# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The mxfp4 LOAD path must hand the model the same layout stock transformers would.

Zoo's `convert_moe_packed_tensors` replacement returns the UN-transposed
[E, D, G*B*2] layout on purpose, so the live loader hook has to restore GPT-OSS's
[E, G*B*2, D]:

  * 4.x             -> module level `mxfp4.dequantize`, called by quantizer_mxfp4
  * 5.1.0 and newer -> `Mxfp4Dequantize` (a ConversionOps) -> `dequantize_convertops`

Zoo only ever patched `dequantize`, so from 5.1.0 the transpose was dropped and
GPT-OSS loaded with dims 1 and 2 swapped; 5.16.0 deleting the function is what
finally turned the drift detector red. The existing mxfp4 tests missed it because
they only exercise the SAVE path, which stayed self-consistent. So assert the
LOADED layout: patch, then compare against pristine transformers on the same input.

Subprocess because patching mutates `transformers.integrations.mxfp4` process-wide;
CPU-only because the comparison is pure layout arithmetic.
"""

import os
import subprocess
import sys
import textwrap

import pytest


# Golden is captured BEFORE the patch, same process and input, so this compares zoo
# against the installed transformers rather than a hardcoded, version-gated shape.
_PROBE = textwrap.dedent(
    """
    import torch
    import transformers
    import transformers.integrations.mxfp4 as m

    if not hasattr(m, "dequantize_convertops"):
        print("SKIP no dequantize_convertops (pre-5.0 ConversionOps path)")
        raise SystemExit(0)

    E, D, G, B = 2, 4, 3, 16
    torch.manual_seed(0)
    blocks = torch.randint(0, 255, (E, D, G, B), dtype=torch.uint8)
    scales = torch.full((E, D, G), 127, dtype=torch.uint8)

    golden = m.dequantize_convertops(blocks.clone(), scales.clone()).clone()

    from unsloth_zoo.temporary_patches.mxfp4 import patch_convert_moe_packed_tensors
    patch_convert_moe_packed_tensors()

    after = m.dequantize_convertops(blocks.clone(), scales.clone())

    # .cpu(): zoo's convert_moe_packed_tensors moves inputs to the accelerator when
    # one is present, which is intended and irrelevant to the layout being checked.
    got, want = tuple(after.shape), tuple(golden.shape)
    if got != want:
        print(
            "FAIL transformers %s: loader got %s, stock transformers gives %s. "
            "Zoo's un-transposed convert_moe_packed_tensors reached the loader with "
            "nothing restoring transpose(1, 2), so GPT-OSS loads with dims 1 and 2 "
            "swapped." % (transformers.__version__, got, want)
        )
        raise SystemExit(1)
    if not torch.equal(after.cpu(), golden.cpu()):
        print("FAIL transformers %s: shape %s matches but values differ"
              % (transformers.__version__, got))
        raise SystemExit(1)
    print("OK transformers %s: loader layout %s matches stock"
          % (transformers.__version__, got))
    """
)


def test_mxfp4_load_path_layout_matches_stock_transformers():
    pytest.importorskip("transformers.integrations.mxfp4")
    env = dict(os.environ)
    # conftest's GPU-free harness does not reach a subprocess, and importing
    # unsloth_zoo calls get_device_type() at import time.
    env["UNSLOTH_ALLOW_CPU"] = "1"
    # The core-upstream lane installs unsloth with `|| true`, so the checkout is
    # allowed to fail, and unsloth_zoo/__init__.py raises
    # ImportError("Please install Unsloth") when find_spec("unsloth") is None.
    # UNSLOTH_IS_PRESENT does NOT cover that: it guards a separate, later check
    # that this import never reaches. Without the line below, one transient git
    # failure turns every 5.x lane red and blames unsloth rather than the layout
    # this test exists to police. Both variables are needed and neither implies
    # the other: this one skips the unsloth requirement, UNSLOTH_ALLOW_CPU keeps
    # get_device_type() from raising on a driverless runner.
    env["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    r = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, env=env,
    )
    if "SKIP" in r.stdout:
        pytest.skip(r.stdout.strip().split("SKIP", 1)[1].strip())
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert "OK" in r.stdout, f"stdout={r.stdout}\nstderr={r.stderr}"

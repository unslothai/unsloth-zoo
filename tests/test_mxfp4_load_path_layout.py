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

"""Loader hook must restore GPT-OSS [E, G*B*2, D] after zoo's un-transposed convert: it patches dequantize (4.x) and
dequantize_convertops (5.0+), since patching only dequantize had left 5.0+ with dims 1 and 2 swapped; subprocess as
patching is process-wide."""

import os
import subprocess
import sys
import textwrap

import pytest


# Golden is captured before the patch in the same process and input, so it compares zoo to the installed transformers, not a hardcoded version-gated shape.
_PROBE = textwrap.dedent(
    """
    import inspect
    import torch
    import transformers
    import transformers.integrations.mxfp4 as m

    if not hasattr(m, "dequantize_convertops"):
        print("SKIP no dequantize_convertops (pre-5.0 ConversionOps path)")
        raise SystemExit(0)

    # Arity-agnostic. transformers 5.0.0 declares
    # (blocks, scales, target_device); 5.1.0 and newer declare (blocks, scales).
    # Hardcoding two positionals made this probe die on 5.0.0 with "missing 1
    # required positional argument: 'target_device'" while capturing the golden,
    # i.e. it went red before it could measure the corruption it exists to detect.
    # Mirror Mxfp4Dequantize.convert, which passes the blocks' own device.
    _N_POS = len([
        p for p in inspect.signature(m.dequantize_convertops).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ])

    def _call(fn, blocks, scales):
        if _N_POS >= 3:
            return fn(blocks, scales, blocks.device)
        return fn(blocks, scales)

    E, D, G, B = 2, 4, 3, 16
    torch.manual_seed(0)
    blocks = torch.randint(0, 255, (E, D, G, B), dtype=torch.uint8)
    scales = torch.full((E, D, G), 127, dtype=torch.uint8)

    golden = _call(m.dequantize_convertops, blocks.clone(), scales.clone()).clone()

    from unsloth_zoo.temporary_patches.mxfp4 import patch_convert_moe_packed_tensors
    patch_convert_moe_packed_tensors()

    after = _call(m.dequantize_convertops, blocks.clone(), scales.clone())

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
    # The subprocess misses conftest's GPU-free harness, and importing unsloth_zoo calls get_device_type() at import time.
    env["UNSLOTH_ALLOW_CPU"] = "1"
    # UNSLOTH_ZOO_DISABLE_GPU_INIT skips the unsloth requirement (a transient git failure in the core-upstream `|| true`
    # unsloth install would turn 5.x lanes red); UNSLOTH_IS_PRESENT guards a later check and does not cover it;
    # UNSLOTH_ALLOW_CPU stops get_device_type() raising on driverless runners.
    env["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    r = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, env=env,
    )
    if "SKIP" in r.stdout:
        pytest.skip(r.stdout.strip().split("SKIP", 1)[1].strip())
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert "OK" in r.stdout, f"stdout={r.stdout}\nstderr={r.stderr}"

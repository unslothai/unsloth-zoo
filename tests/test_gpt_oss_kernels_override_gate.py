# SPDX-License-Identifier: AGPL-3.0-only
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

"""patch_gpt_oss must not claim MXFP4 kernels transformers cannot load; subprocess per case."""
import subprocess
import sys
import textwrap

import pytest


_RUNNER = textwrap.dedent(
    """
    import sys, types
    hub_ok = sys.argv[1] == "hub_ok"
    sys.modules["triton_kernels"] = types.ModuleType("triton_kernels")   # what vLLM registers

    import transformers.utils
    transformers.utils.is_kernels_available = lambda *a, **k: hub_ok
    import transformers.integrations.mxfp4 as m
    if sys.argv[2] == "direct" and not hasattr(m, "_replace_with_mxfp4_linear"):
        m._replace_with_mxfp4_linear = lambda *a, **k: (a[0], True)   # the 4.57 helper
    elif sys.argv[2] == "hub_only" and hasattr(m, "_replace_with_mxfp4_linear"):
        del m._replace_with_mxfp4_linear
    import transformers.quantizers.quantizer_mxfp4 as q
    q.is_kernels_available = lambda *a, **k: hub_ok
    original = q.is_kernels_available

    from unsloth_zoo.temporary_patches.gpt_oss import patch_gpt_oss
    patch_gpt_oss()
    print("OVERRIDDEN" if q.is_kernels_available is not original else "KEPT")
    print("CLAIMS", q.is_kernels_available())
    """
)


def _run(mode, layout = "hub_only"):
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, mode, layout], capture_output = True, text = True, timeout = 600,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    return proc.stdout


def _loader_uses_hub_kernel():
    try:
        import inspect
        import transformers.integrations.mxfp4 as m
        return "get_kernel" in inspect.getsource(m.replace_with_mxfp4_linear)
    except Exception:
        return False


@pytest.mark.skipif(not _loader_uses_hub_kernel(), reason = "transformers loads MXFP4 kernels without the hub")
def test_unreachable_hub_kernels_keep_the_bf16_fallback():
    out = _run("hub_missing")
    assert "KEPT" in out and "CLAIMS False" in out, out


def test_reachable_hub_kernels_still_take_the_native_path():
    pytest.importorskip("transformers.integrations.mxfp4")
    out = _run("hub_ok")
    assert "CLAIMS True" in out, out


@pytest.mark.skipif(not _loader_uses_hub_kernel(), reason = "transformers loads MXFP4 kernels without the hub")
def test_a_direct_replacement_keeps_the_native_path_without_the_hub():
    out = _run("hub_missing", "direct")
    assert "OVERRIDDEN" in out and "CLAIMS True" in out, out

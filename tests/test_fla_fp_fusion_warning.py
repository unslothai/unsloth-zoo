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

"""Triton 3.9 warns once per process when a launch leaves enable_fp_fusion unset, which the vendored fla
launches do. Injecting the vendored tree ignores that one message; every other warning still escapes."""

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

_PROBE = textwrap.dedent("""
    import warnings
    import unsloth_zoo.temporary_patches.fla_vendor as fv
    assert fv.__file__.startswith({root!r}), fv.__file__
    injected, _ = fv._inject_vendored_fla()
    assert injected, "vendored fla was not injected"
    import triton
    from triton import _utils
    def outcome(fn):
        try:
            fn()
            return "silent"
        except UserWarning as e:
            return "raised"
    print("fp_fusion", outcome(lambda: triton.knobs.language.fp_fusion_enabled(None)))
    print("unrelated", outcome(lambda: warnings.warn("unrelated user warning", UserWarning)))
    print("other_triton", outcome(lambda: _utils.warn_once(_utils.CAT_CAN_REORDER_WARNING)))
""")


def _run(*w_flags):
    try:
        from triton import _utils
    except Exception:
        pytest.skip("triton is not installed")
    if not hasattr(_utils, "FP_FUSION_DEFAULT_WARNING"):
        pytest.skip("Triton < 3.9 does not emit the FP-fusion default warning")
    env = dict(os.environ, UNSLOTH_IS_PRESENT = "1", UNSLOTH_VENDORED_FLA_NO_AUTORUN = "1",
               PYTHONPATH = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.pop("TRITON_DEFAULT_FP_FUSION", None)
    env.pop("PYTHONWARNINGS", None)
    cmd = [sys.executable, *w_flags, "-c", _PROBE.format(root = str(ROOT))]
    out = subprocess.run(cmd, capture_output = True, text = True, env = env, timeout = 600)
    assert out.returncode == 0, out.stderr[-3000:]
    return out


def test_fp_fusion_warning_ignored_under_error_but_others_raise():
    # -W error semantics; the CPU fallback notice of the vendored tree is allowed so it can inject on CPU.
    out = _run("-W", "error", "-W", "ignore:Triton is not supported on current platform:UserWarning")
    lines = dict(line.split(" ", 1) for line in out.stdout.splitlines() if " " in line)
    assert lines.get("fp_fusion") == "silent", out.stdout
    assert lines.get("unrelated") == "raised", out.stdout
    assert lines.get("other_triton") == "raised", out.stdout


def test_fp_fusion_warning_not_printed_but_others_are():
    out = _run("-W", "always")
    assert "TRITON_DEFAULT_FP_FUSION is unset" not in out.stderr
    assert "unrelated user warning" in out.stderr
    assert "tl.cat(can_reorder=True) is deprecated" in out.stderr

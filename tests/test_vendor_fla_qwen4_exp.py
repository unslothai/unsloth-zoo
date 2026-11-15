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

"""Qwen4Exp gets vendored fla kernels even when its modeling module is imported before Unsloth."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ZOO_ROOT = Path(__file__).resolve().parents[1]

_SUB = textwrap.dedent(
    """
    import sys
    import transformers.models.qwen4_exp.modeling_qwen4_exp as m  # BEFORE unsloth_zoo
    from transformers.integrations import hub_kernels
    if not hasattr(hub_kernels, "use_kernel_func_from_hub_with_fallback"):
        print("SKIP_NO_KERNEL_HUB"); raise SystemExit(0)
    def resolved(wrapper):
        for cell in wrapper.__closure__ or ():
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if callable(value):
                return value
    assert "unsloth_zoo" not in sys.modules
    before = resolved(m.torch_chunk_gated_delta_rule)
    assert getattr(before, "__module__", "").startswith("transformers."), before
    import unsloth_zoo  # noqa: F401  (runs patch_vendor_fla at import)
    from unsloth_zoo.temporary_patches.fla_vendor import _resolved_implementation, patch_vendor_fla
    patch_vendor_fla()
    fla = sys.modules.get("fla")
    if fla is None or not getattr(fla, "_UNSLOTH_VENDORED_FLA", False):
        print("SKIP_NOT_INJECTED"); raise SystemExit(0)
    live = sys.modules["fla.ops.gated_delta_rule"]
    assert _resolved_implementation(m.torch_chunk_gated_delta_rule) is live.chunk_gated_delta_rule
    assert _resolved_implementation(m.torch_recurrent_gated_delta_rule) is live.recurrent_gated_delta_rule
    print("QWEN4_EXP_REPAIRED_OK")
    """
)


_KILL_SWITCH = textwrap.dedent(
    """
    import importlib.util, sys
    if importlib.util.find_spec("fla") is not None and "unsloth_zoo" not in (importlib.util.find_spec("fla").origin or ""):
        print("SKIP_REAL_FLA_INSTALLED"); raise SystemExit(0)
    import unsloth_zoo  # noqa: F401
    import transformers.models.qwen4_exp.modeling_qwen4_exp as m
    from unsloth_zoo.temporary_patches.fla_vendor import _resolved_implementation
    impl = _resolved_implementation(m.torch_chunk_gated_delta_rule)
    assert getattr(impl, "__module__", "").startswith("transformers."), impl
    print("KILL_SWITCH_OK")
    """
)


@pytest.fixture(scope="module")
def _processes():
    # Fresh interpreters (import order is the subject), started in parallel.
    pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    base = dict(os.environ)
    base["PYTHONPATH"] = str(ZOO_ROOT) + os.pathsep + base.get("PYTHONPATH", "")
    base["HF_HUB_OFFLINE"] = "1"
    for k in ("UNSLOTH_VENDORED_FLA_NO_AUTORUN", "UNSLOTH_DISABLE_VENDORED_FLA"):
        base.pop(k, None)
    kill = dict(base, UNSLOTH_DISABLE_VENDORED_FLA="1")
    procs = {
        name: subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
        for name, code, env in (("imported_first", _SUB, base), ("kill_switch", _KILL_SWITCH, kill))
    }
    results = {}

    def result(name):
        if name not in results:
            results[name] = procs[name].communicate(timeout=600)
        return results[name]

    yield result
    for proc in procs.values():
        if proc.poll() is None:
            proc.kill()
            proc.communicate()


def _check(result, ok):
    stdout, stderr = result
    if "SKIP_" in stdout:
        pytest.skip(stdout.strip())
    assert ok in stdout, f"stdout=\n{stdout}\nstderr=\n{stderr[-3000:]}"


def test_qwen4_exp_imported_first_gets_vendored_kernels(_processes):
    _check(_processes("imported_first"), "QWEN4_EXP_REPAIRED_OK")


def test_qwen4_exp_kill_switch_keeps_torch(_processes):
    _check(_processes("kill_switch"), "KILL_SWITCH_OK")

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

"""Regression guard for XPU SM-capability handling in _get_vllm_state_dict.

Intel XPU (like AMD ROCm) has no NVIDIA SM capability, and
torch.cuda.get_device_capability() fails there. _get_vllm_state_dict must skip
the CUDA query and set sm_cap = 0 on XPU, gating out the SM90-only
CUTLASS/DeepGEMM paths. The full function needs a live vLLM model, so this checks
the source-level invariant (metadata-only, no import / no CUDA probe) the same
way tests/test_zoo_history_regressions_deep.py does.
"""

import importlib.util
import pathlib
import re


def _module_source_text(module_name: str) -> str:
    # find_spec is metadata-only, so importing vllm_utils (and its import-time
    # torch.cuda probes) is avoided on a CPU-only box.
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin in (None, "built-in"):
        raise ImportError(f"could not locate source for {module_name!r}")
    return pathlib.Path(spec.origin).read_text(encoding="utf-8")


def test_get_vllm_state_dict_sm_cap_guards_rocm_and_xpu():
    src = _module_source_text("unsloth_zoo.vllm_utils")

    idx = src.find("sm_cap")
    assert idx != -1, "sm_cap computation missing from vllm_utils.py"
    window = src[max(0, idx - 400): idx + 400]

    # The CUDA capability query must be gated behind BOTH the ROCm and XPU checks
    guard = re.search(r"if\s+not\s+is_hip\(\)\s+and\s+DEVICE_TYPE\s*!=\s*[\"']xpu[\"']\s*:", window)
    assert guard is not None, (
        "sm_cap guard must exclude both ROCm and Intel XPU from the CUDA "
        "capability query, e.g. `if not is_hip() and DEVICE_TYPE != \"xpu\":`. "
        "Regression: XPU would call torch.cuda.get_device_capability() and crash."
    )

    assert re.search(r"sm_cap\s*=\s*0", window) is not None, (
        "XPU/ROCm fallback must set sm_cap = 0."
    )

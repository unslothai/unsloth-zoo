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

"""No i32 multiply may take an expert stride (wraps past 2**31 elements). CPU: lowered to sm90 TTIR."""

import os
import re

import pytest

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

triton = pytest.importorskip("triton")
m = pytest.importorskip("unsloth_zoo.merged_rollout_exact")

try:
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
except Exception:  # noqa: BLE001
    pytest.skip("this Triton cannot lower offline", allow_module_level = True)

_PTRS = {
    "W": "*bf16", "A": "*bf16", "B": "*bf16", "counts": "*i32", "woff": "*i64", "wcnt": "*i16", "words": "*i32",
    "idx_out": "*i32", "val_out": "*i16", "eoff": "*i64", "checksum": "*i64",
}
_CONSTS = {"R": 8, "BLOCK_N": 32, "BLOCK_K": 128, "MANT": 7, "EXPM": 255, "MAX_GAP": 7}


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_expert_offsets_are_int64(mode):
    k = getattr(m, "_exact_merge_kernel", None)
    if k is None:
        pytest.skip("Triton kernels unavailable")
    consts = dict(_CONSTS, mode = mode)
    sig = {}
    for n in k.arg_names:
        if n in _PTRS:
            sig[n] = _PTRS[n]
        elif n == "scale":
            sig[n] = "fp32"
        elif n in consts:
            sig[n] = "constexpr"
        else:
            sig[n] = "i32"
    src = ASTSource(fn = k, signature = sig, constexprs = {(k.arg_names.index(n),): v for n, v in consts.items()})
    try:
        ttir = triton.compile(
            src, target = GPUTarget("cuda", 90, 32), options = {"enable_fp_fusion": False, "num_warps": 8}
        ).asm["ttir"]
    except Exception as exc:  # noqa: BLE001 -- no NVIDIA backend / ptxas in this Triton
        pytest.skip(f"cannot lower offline: {exc}")
    bad = [l.strip() for l in ttir.splitlines() if re.search(r"arith\.muli .*%s[WAB]e\b.* : i32\b", l)]
    assert "arith.muli" in ttir and bad == []

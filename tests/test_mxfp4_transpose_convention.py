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

"""The mxfp4 export path calls convert_moe_packed_tensors directly and must apply an external
transpose(1, 2) for every convention EXCEPT stock transformers >= 4.56.0, which already self-
transposes inside the function. Keying the decision only on the presence of the Unsloth-only
convert_moe_packed_tensors_cpu variant under-transposed exports on stock transformers 4.55.x
(supported by unsloth_zoo's pin: only 4.55.0/4.55.1 are excluded), silently corrupting weights.
Stock behaviour verified against the real transformers source:
  * v4.55.0 / v4.55.4: convert_moe_packed_tensors returns `out` (no transpose); dequantize()
    applied `dequantized.transpose(1, 2).contiguous().to(target_device)`.
  * v4.56.0 / v4.56.2 / v4.57.0 / 4.57.6: returns `out.transpose(1, 2).contiguous()`; dequantize()
    does not transpose.
"""
import torch

from unsloth_zoo.saving_utils import _mxfp4_base_returns_transposed


# A truthy sentinel standing in for the injected convert_moe_packed_tensors_cpu.
_CPU_SENTINEL = object()


def test_decision_matrix_patch_and_version():
    # Unsloth patched base (its _cpu variant present) never self-transposes, in any version.
    assert _mxfp4_base_returns_transposed(_CPU_SENTINEL, "4.55.4") is False
    assert _mxfp4_base_returns_transposed(_CPU_SENTINEL, "4.57.6") is False

    # Stock transformers < 4.56.0 returns the un-transposed layout -> external transpose needed.
    assert _mxfp4_base_returns_transposed(None, "4.55.2") is False
    assert _mxfp4_base_returns_transposed(None, "4.55.3") is False
    assert _mxfp4_base_returns_transposed(None, "4.55.4") is False

    # Stock transformers >= 4.56.0 already self-transposes -> no external transpose.
    assert _mxfp4_base_returns_transposed(None, "4.56.0") is True
    assert _mxfp4_base_returns_transposed(None, "4.56.2") is True
    assert _mxfp4_base_returns_transposed(None, "4.57.0") is True
    assert _mxfp4_base_returns_transposed(None, "4.57.6") is True


def test_unparseable_version_defaults_to_modern_stock():
    # None / garbage version must not crash; assume modern stock (self-transposes).
    assert _mxfp4_base_returns_transposed(None, None) is True
    assert _mxfp4_base_returns_transposed(None, "not-a-version") is True


def test_all_conventions_produce_identical_gpt_oss_layout():
    """The export weight must land in the same transposed GPT-OSS layout regardless of which
    convert_moe_packed_tensors convention produced the raw dequant. The ground truth is the
    stock >= 4.56.0 output: out.transpose(1, 2).contiguous()."""
    torch.manual_seed(0)
    E, D, N = 4, 6, 8  # [experts, rows, cols] of the un-transposed dequant `out`
    out = torch.randn(E, D, N)
    ground_truth = out.transpose(1, 2).contiguous()  # stock >= 4.56.0 self-transposed layout

    def apply(base_out, cpu_variant, version):
        base_returns_transposed = _mxfp4_base_returns_transposed(cpu_variant, version)
        return base_out.contiguous() if base_returns_transposed else base_out.transpose(1, 2).contiguous()

    # Unsloth patched base: returns un-transposed `out`, _cpu present.
    assert torch.equal(apply(out, _CPU_SENTINEL, "4.57.6"), ground_truth)
    # Stock 4.55.x: returns un-transposed `out`, _cpu absent.
    assert torch.equal(apply(out, None, "4.55.4"), ground_truth)
    # Stock >= 4.56.0: returns already-transposed layout, _cpu absent.
    assert torch.equal(apply(ground_truth, None, "4.56.0"), ground_truth)


def test_old_cpu_only_keying_would_corrupt_stock_4_55():
    """Documents the fixed bug: the previous decision (base_returns_transposed == (_cpu is None))
    would treat stock 4.55.x as self-transposing and skip the external transpose, producing the
    wrong (un-transposed) layout."""
    torch.manual_seed(0)
    out = torch.randn(4, 6, 8)
    ground_truth = out.transpose(1, 2).contiguous()

    # Old logic: _cpu absent -> assume self-transposed -> no external transpose.
    old_base_returns_transposed = (None is None)  # i.e. _cpu is None -> True (buggy)
    old_W = out.contiguous() if old_base_returns_transposed else out.transpose(1, 2).contiguous()
    assert not torch.equal(old_W, ground_truth)  # confirms the old path corrupted 4.55.x

    # New logic keyed on version restores correctness.
    new_returns_transposed = _mxfp4_base_returns_transposed(None, "4.55.4")
    new_W = out.contiguous() if new_returns_transposed else out.transpose(1, 2).contiguous()
    assert torch.equal(new_W, ground_truth)

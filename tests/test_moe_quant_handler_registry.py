# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Save-side MoE quant handler registry: FP8 dequant/requant lives in
moe_utils_fp8.py and saving_utils.py consults it via apply_moe_quant_load.

These tests pin the wiring (no inline FP8 helpers in saving_utils anymore),
exercise the FP8 round-trip end-to-end with a synthetic safetensors-like
file, and lock in the _MOE_QUANT_UNSAFE sentinel path for the missing-scale
case.
"""

from __future__ import annotations

import pytest
import torch


def test_saving_utils_uses_registry_not_inline_helpers():
    """saving_utils.py must NOT redefine FP8 dequant/requant locally; those
    helpers live in moe_utils_fp8.py and are imported in."""
    from unsloth_zoo import saving_utils

    # Inline FP8 helpers removed from saving_utils.
    for sym in (
        "_FP8_E4M3_MAX",
        "_fp8_dequant_blockwise",
        "_fp8_requant_blockwise",
        "_FP8_MERGE_UNSAFE",
        "_fp8_load_for_merge",
    ):
        assert not hasattr(saving_utils, sym), (
            f"saving_utils still defines {sym!r}; it should live in moe_utils_fp8."
        )

    # Registry entry points wired in.
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import (
        apply_moe_quant_load,
        _MOE_QUANT_UNSAFE,
    )
    assert saving_utils._apply_moe_quant_load is apply_moe_quant_load
    assert saving_utils._MOE_QUANT_UNSAFE is _MOE_QUANT_UNSAFE
    assert callable(saving_utils._merge_moe_expert_quant_aware)


def test_registry_contains_fp8_handler():
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import (
        _MOE_QUANT_HANDLERS,
        _fp8_save_handler,
    )
    assert _fp8_save_handler in _MOE_QUANT_HANDLERS


def _build_fake_fp8_file():
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import _fp8_requant_blockwise

    torch.manual_seed(0)
    W_real = (torch.randn(128, 256, dtype=torch.float32) * 0.1)
    W_fp8, scale_inv = _fp8_requant_blockwise(W_real, (64, 128), torch.bfloat16)

    class FakeFile:
        def __init__(self, tensors):
            self._tensors = tensors

        def get_tensor(self, key):
            return self._tensors[key]

    header = {
        "X.weight":             {"dtype": "F8_E4M3"},
        "X.weight_scale_inv":   {"dtype": "BF16"},
    }
    file = FakeFile({"X.weight": W_fp8, "X.weight_scale_inv": scale_inv})
    return W_real, W_fp8, scale_inv, header, file


def test_fp8_handler_roundtrip_identity_merge():
    """Identity 'merge' (no LoRA delta) must round-trip the FP8 weight
    bytes within the FP8 quantization error budget."""
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import (
        _fp8_save_handler,
        _fp8_dequant_blockwise,
        _MOE_QUANT_UNSAFE,
    )
    W_real, W_fp8, scale_inv, header, file = _build_fake_fp8_file()

    loaded = _fp8_save_handler(file, header, "X.weight")
    assert loaded is not None
    W_bf16, requant = loaded
    assert W_bf16 is not _MOE_QUANT_UNSAFE
    assert W_bf16.dtype == torch.bfloat16
    assert tuple(W_bf16.shape) == tuple(W_fp8.shape)

    new_fp8, write_dtype, extras = requant(W_bf16)
    assert write_dtype == torch.float8_e4m3fn
    assert len(extras) == 1
    scale_key, new_scale, scale_dtype = extras[0]
    assert scale_key == "X.weight_scale_inv"
    assert scale_dtype == torch.bfloat16
    assert tuple(new_scale.shape) == tuple(scale_inv.shape)

    # Round-trip drift bound: dequant -> requant of bf16 dequant should be
    # tight enough that re-dequantising stays within 2x the single-pass FP8
    # quantisation budget.
    re_dequant = _fp8_dequant_blockwise(new_fp8, new_scale)
    max_err = (re_dequant.to(torch.float32) - W_real).abs().max().item()
    assert max_err < 0.05, f"round-trip drift too large: {max_err}"


def test_fp8_handler_returns_sentinel_when_scale_missing():
    """When the companion scale is absent, the handler must surface the
    _MOE_QUANT_UNSAFE sentinel so saving_utils skips the merge."""
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import (
        _fp8_save_handler,
        _fp8_requant_blockwise,
        _MOE_QUANT_UNSAFE,
    )

    W_real = (torch.randn(64, 128, dtype=torch.float32) * 0.1)
    W_fp8, _scale = _fp8_requant_blockwise(W_real, (32, 64), torch.bfloat16)

    class FakeFile:
        def __init__(self, tensors):
            self._tensors = tensors

        def get_tensor(self, key):
            return self._tensors[key]

    header = {"Y.weight": {"dtype": "F8_E4M3"}}
    file = FakeFile({"Y.weight": W_fp8})

    result = _fp8_save_handler(file, header, "Y.weight")
    assert result == (_MOE_QUANT_UNSAFE, None)


def test_fp8_handler_returns_none_for_non_fp8_keys():
    """Non-FP8 tensors must be ignored so the generic loader path takes over."""
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import _fp8_save_handler

    class FakeFile:
        def get_tensor(self, key):
            return torch.randn(8, 8, dtype=torch.bfloat16)

    # dtype hint is bf16 → handler bails without touching the file.
    header = {"Z.weight": {"dtype": "BF16"}}
    assert _fp8_save_handler(FakeFile(), header, "Z.weight") is None


def test_apply_moe_quant_load_dispatch():
    """apply_moe_quant_load consults handlers in order and falls back to
    file.get_tensor(key) when no handler matches."""
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import apply_moe_quant_load

    class FakeFile:
        def __init__(self):
            self.calls = []

        def get_tensor(self, key):
            self.calls.append(key)
            return torch.tensor([1.0, 2.0, 3.0])

    file = FakeFile()
    header = {"plain.weight": {"dtype": "BF16"}}
    W, requant = apply_moe_quant_load(file, header, "plain.weight")
    assert requant is None
    assert torch.equal(W, torch.tensor([1.0, 2.0, 3.0]))
    assert file.calls == ["plain.weight"]


class _FakeFile:
    def __init__(self, tensors):
        self._tensors = tensors

    def get_tensor(self, key):
        return self._tensors[key]


def test_fp8_handler_partial_block_needs_block_size():
    """A weight dim that is not a block multiple (130 rows, 128x128 blocks, 2x2 scale
    grid) divides the scale grid evenly (130 % 2 == 0), so inference picks bm=65 and
    misapplies scales. The configured weight_block_size must override that."""
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import (
        _fp8_save_handler,
        _fp8_dequant_blockwise,
        _fp8_requant_blockwise,
        _MOE_QUANT_UNSAFE,
    )
    torch.manual_seed(0)
    W_real = torch.randn(130, 256, dtype=torch.float32) * 0.1
    W_fp8, scale_inv = _fp8_requant_blockwise(W_real, (128, 128), torch.float32)
    assert tuple(scale_inv.shape) == (2, 2)  # ceil(130/128), ceil(256/128)

    header = {"X.weight": {"dtype": "F8_E4M3"}, "X.weight_scale_inv": {"dtype": "F32"}}
    file = _FakeFile({"X.weight": W_fp8, "X.weight_scale_inv": scale_inv})

    # Without block_size: inference (bm=65) misapplies the scale grid -> wrong dequant.
    W_inferred, _ = _fp8_save_handler(file, header, "X.weight")
    assert W_inferred is not _MOE_QUANT_UNSAFE
    err_inferred = (W_inferred.float() - W_real).abs().max().item()

    # With configured block size: correct 128x128 tiling -> matches the real weight.
    W_block, requant = _fp8_save_handler(file, header, "X.weight", block_size=(128, 128))
    assert W_block is not _MOE_QUANT_UNSAFE
    err_block = (W_block.float() - W_real).abs().max().item()
    assert err_block < 0.05, f"block-size dequant off: {err_block}"
    assert err_inferred > err_block * 5, (
        f"inference should be materially worse (inferred={err_inferred}, block={err_block})"
    )

    # Requant round-trips for the partial block (no reshape crash, grid preserved).
    new_fp8, write_dtype, extras = requant(W_block)
    assert write_dtype == torch.float8_e4m3fn
    scale_key, new_scale, _ = extras[0]
    assert tuple(new_scale.shape) == (2, 2)
    re = _fp8_dequant_blockwise(new_fp8, new_scale, block_size=(128, 128))
    assert (re.float() - W_real).abs().max().item() < 0.05


def test_fp8_handler_block_size_grid_mismatch_is_unsafe():
    """A scale grid that matches neither the configured block size's ceil grid nor an exact
    division of the weight is rejected (UNSAFE) rather than silently mis-tiled."""
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import (
        _fp8_save_handler, _MOE_QUANT_UNSAFE,
    )
    # 130x256 weight, (3,2) grid: ceil(130/128)=2 != 3 (block mismatch) and 130 % 3 != 0
    # (no exact division) -> UNSAFE.
    W_fp8 = (torch.randn(130, 256) * 0.1).clamp(-448, 448).to(torch.float8_e4m3fn)
    scale_inv = torch.rand(3, 2, dtype=torch.float32) + 0.5
    header = {"X.weight": {"dtype": "F8_E4M3"}, "X.weight_scale_inv": {"dtype": "F32"}}
    file = _FakeFile({"X.weight": W_fp8, "X.weight_scale_inv": scale_inv})
    assert _fp8_save_handler(file, header, "X.weight", block_size=(128, 128)) == (_MOE_QUANT_UNSAFE, None)


def test_fp8_handler_per_channel_scale_with_block_size_merges():
    """A per-channel (rows,1) scale must still merge when a global block_size is configured:
    the block size does not tile it, so the handler falls back to per-channel inference
    instead of aborting the whole FP8 MoE merge."""
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import (
        _fp8_save_handler, _MOE_QUANT_UNSAFE,
    )
    torch.manual_seed(21)
    rows, cols = 256, 128
    W_real = torch.randn(rows, cols, dtype=torch.float32) * 0.1
    scale = (W_real.abs().amax(dim=1, keepdim=True) / 448.0).clamp_min(1e-12)  # (rows, 1)
    W_fp8 = (W_real / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    header = {"X.weight": {"dtype": "F8_E4M3"}, "X.weight_scale_inv": {"dtype": "F32"}}
    file = _FakeFile({"X.weight": W_fp8, "X.weight_scale_inv": scale})

    loaded = _fp8_save_handler(file, header, "X.weight", block_size=(128, 128))
    assert loaded[0] is not _MOE_QUANT_UNSAFE, "per-channel scale wrongly rejected"
    W_bf16, requant = loaded
    assert (W_bf16.float() - W_real).abs().max().item() < 0.05
    # Requant round-trips and preserves the (rows, 1) grid.
    _new_fp8, _dtype, extras = requant(W_bf16)
    _key, new_scale, _sd = extras[0]
    assert tuple(new_scale.shape) == (rows, 1)

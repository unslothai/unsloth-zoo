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

"""End-to-end behaviour for `_merge_moe_expert_quant_aware`.

The registry inversion pushed FP8 dequant/requant out of saving_utils into
moe_utils_fp8. These tests pin both legs:

- bf16 base weight: the helper writes the merged tensor unchanged in shape,
  no companion-scale write, no fallback recorded.
- FP8 base weight: the helper dequantises, applies the LoRA merge in bf16,
  requantises back to FP8, writes the data tensor AND the companion
  weight_scale_inv to the output mm.
- FP8 base without companion scale: the helper records a merge fallback
  and skips writing.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch


@dataclass
class _FakeLoraStats:
    lora_A: torch.Tensor = None
    lora_B: torch.Tensor = None
    scaling: float = 1.0
    module: object = None


class _FakeFile:
    def __init__(self, tensors):
        self._tensors = tensors

    def get_tensor(self, key):
        return self._tensors[key]


class _FakeMM:
    """Stand-in for the mmap-backed safetensors output. Collects writes so
    we can assert on them."""

    def __init__(self):
        self.writes = []  # list of (key, tensor, dtype) tuples


def _identity_merge(W, lora_stats, expert_idx, num_experts, output_dtype):
    """Stand-in 'merge' that returns the input unchanged. Lets us pin the
    quant-aware wrapper without depending on the real LoRA math."""
    return W.to(output_dtype)


@pytest.fixture(autouse=True)
def _stub_write_tensor_direct_torch(monkeypatch):
    """`_write_tensor_direct_torch` does real safetensors header arithmetic on
    a torch storage; for these tests we just capture (key, tensor, dtype)."""
    from unsloth_zoo import saving_utils

    def _capture(mm, header_metadata, length_of_header, key, tensor, dtype):
        mm.writes.append((key, tensor.detach().clone(), dtype))

    monkeypatch.setattr(saving_utils, "_write_tensor_direct_torch", _capture)
    yield


def _make_header_for(keys_to_dtype):
    return {k: {"dtype": d} for k, d in keys_to_dtype.items()}


def test_bf16_weight_writes_only_the_merged_tensor():
    from unsloth_zoo.saving_utils import _merge_moe_expert_quant_aware

    W = torch.randn(64, 128, dtype=torch.bfloat16)
    file = _FakeFile({"layer.0.experts.0.gate_proj.weight": W})
    hdr = _make_header_for({"layer.0.experts.0.gate_proj.weight": "BF16"})
    mm = _FakeMM()

    ok = _merge_moe_expert_quant_aware(
        "gate", "layer.0.experts.0.gate_proj.weight", file, hdr, _FakeLoraStats(),
        0, 1, None, mm, 0, set(), _identity_merge,
    )
    assert ok is True
    assert len(mm.writes) == 1
    written_key, written_tensor, written_dtype = mm.writes[0]
    assert written_key == "layer.0.experts.0.gate_proj.weight"
    assert written_dtype == torch.bfloat16
    assert torch.equal(written_tensor.to(torch.bfloat16), W)


def test_fp8_weight_writes_data_and_scale():
    from unsloth_zoo.saving_utils import _merge_moe_expert_quant_aware
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import _fp8_requant_blockwise

    torch.manual_seed(0)
    W_real = torch.randn(128, 256, dtype=torch.float32) * 0.1
    W_fp8, scale_inv = _fp8_requant_blockwise(W_real, (64, 128), torch.bfloat16)

    file = _FakeFile({
        "layer.0.experts.0.gate_proj.weight":           W_fp8,
        "layer.0.experts.0.gate_proj.weight_scale_inv": scale_inv,
    })
    hdr = _make_header_for({
        "layer.0.experts.0.gate_proj.weight":           "F8_E4M3",
        "layer.0.experts.0.gate_proj.weight_scale_inv": "BF16",
    })
    mm = _FakeMM()

    ok = _merge_moe_expert_quant_aware(
        "gate", "layer.0.experts.0.gate_proj.weight", file, hdr, _FakeLoraStats(),
        0, 1, None, mm, 0, set(), _identity_merge,
    )
    assert ok is True

    # Two writes: the scale companion first (extra_writes), then the data.
    assert len(mm.writes) == 2
    keys = [w[0] for w in mm.writes]
    assert keys == [
        "layer.0.experts.0.gate_proj.weight_scale_inv",
        "layer.0.experts.0.gate_proj.weight",
    ]
    _, scale_written, scale_dtype = mm.writes[0]
    assert scale_dtype == torch.bfloat16
    assert tuple(scale_written.shape) == tuple(scale_inv.shape)
    _, data_written, data_dtype = mm.writes[1]
    assert data_dtype == torch.float8_e4m3fn
    assert tuple(data_written.shape) == tuple(W_fp8.shape)


def test_fp8_weight_missing_scale_records_fallback_and_skips_write():
    from unsloth_zoo import saving_utils
    from unsloth_zoo.saving_utils import _merge_moe_expert_quant_aware

    # Sentinel-trigger: FP8 weight in header but NO companion scale key.
    fp8 = torch.zeros(64, 128, dtype=torch.float8_e4m3fn)
    file = _FakeFile({"layer.0.experts.0.down_proj.weight": fp8})
    hdr = _make_header_for({"layer.0.experts.0.down_proj.weight": "F8_E4M3"})
    mm = _FakeMM()

    fallback_log = []

    def _capture_fallback(*args, **kwargs):
        fallback_log.append((args, kwargs))

    # Patch in-place on the module to observe.
    original = saving_utils._record_moe_merge_fallback
    saving_utils._record_moe_merge_fallback = _capture_fallback
    try:
        ok = _merge_moe_expert_quant_aware(
            "down", "layer.0.experts.0.down_proj.weight", file, hdr, _FakeLoraStats(),
            0, 1, None, mm, 0, set(), _identity_merge,
        )
    finally:
        saving_utils._record_moe_merge_fallback = original

    assert ok is False
    assert mm.writes == []
    assert len(fallback_log) == 1, "fallback must be recorded exactly once"


def test_key_missing_is_a_silent_no_op():
    from unsloth_zoo.saving_utils import _merge_moe_expert_quant_aware

    file = _FakeFile({})
    hdr = {}  # key not present
    mm = _FakeMM()
    assert _merge_moe_expert_quant_aware(
        "gate", "not_in_header.gate_proj.weight", file, hdr, _FakeLoraStats(),
        0, 1, None, mm, 0, set(), _identity_merge,
    ) is False
    assert mm.writes == []

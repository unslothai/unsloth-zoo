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

"""LoRA snapshot isolation in create_lora_statistics, and _merge_lora's magnitude
tripwire for finite garbage the isfinite gate cannot see.

CPU only, and does not import vLLM.
"""
import pytest
import torch
import torch.nn as nn

from unsloth_zoo.saving_utils import (
    LoraStats,
    _absmax,
    _merge_lora,
    _snapshot_lora_weight,
    create_lora_statistics,
)


def _tiny_peft(r = 8, alpha = 16, dora = False, d_in = 32, d_out = 24):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(1234)

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(d_in, d_out, bias = False)

        def forward(self, x):
            return self.q_proj(x)

    model = get_peft_model(_M(), LoraConfig(
        r = r, lora_alpha = alpha, target_modules = ["q_proj"],
        lora_dropout = 0.0, bias = "none", use_dora = dora,
    ))
    # PEFT zero-inits lora_B, which would make every merge a no-op here.
    for name, module in model.named_modules():
        if name.endswith("lora_B.default"):
            with torch.no_grad():
                module.weight.copy_(torch.randn_like(module.weight) * 0.05)
    return model


def _delta_stats(model):
    captured = create_lora_statistics(model, merge_into_original = True)
    lora_weights = captured[0] if isinstance(captured, tuple) else captured
    for stats in lora_weights.values():
        if stats.lora_A is not None and stats.lora_B is not None:
            return stats
    raise AssertionError("no LoRA delta captured")


def _base_weight(model):
    for name, module in model.named_modules():
        if name.endswith("q_proj") and hasattr(module, "base_layer"):
            return module.base_layer.weight
    raise AssertionError("base layer not found")


def test_snapshot_isolates_merge_from_later_adapter_mutation():
    model = _tiny_peft()
    stats = _delta_stats(model)
    W = _base_weight(model).detach().clone().float()

    before = _merge_lora(W.clone(), stats, "q_proj").clone()
    for name, module in model.named_modules():
        if name.endswith("lora_B.default"):
            with torch.no_grad():
                module.weight.mul_(2.0)          # what LoRALayerWeights.optimize() does
    after = _merge_lora(W.clone(), stats, "q_proj")

    assert torch.equal(before, after), \
        "merge followed a live adapter reference instead of the captured snapshot"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_snapshot_is_a_detached_cpu_copy_with_exact_values(dtype):
    weight = nn.Parameter(torch.randn(8, 16, dtype = dtype))
    snapshot = _snapshot_lora_weight(weight)

    assert snapshot.device.type == "cpu"
    assert snapshot.dtype == dtype
    assert not isinstance(snapshot, nn.Parameter)
    assert not snapshot.requires_grad
    assert not snapshot.is_inference()
    assert snapshot.data_ptr() != weight.data_ptr()
    assert torch.equal(snapshot, weight.detach())


def test_snapshot_survives_being_taken_under_inference_mode():
    weight = nn.Parameter(torch.randn(4, 4))

    @torch.inference_mode
    def capture():
        return _snapshot_lora_weight(weight)

    snapshot = capture()
    assert not snapshot.is_inference()
    snapshot.mul_(2.0)          # would raise on an inference tensor


def test_dora_magnitude_is_snapshotted_too():
    model = _tiny_peft(dora = True)
    stats = _delta_stats(model)
    assert stats.magnitude is not None
    assert stats.magnitude.device.type == "cpu"
    assert not isinstance(stats.magnitude, nn.Parameter)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16])
def test_absmax_matches_abs_max(dtype):
    t = (torch.randn(32, 48) * 7.0).to(dtype)
    assert _absmax(t) == pytest.approx(t.abs().max().item(), rel = 1e-3)


def test_absmax_handles_non_contiguous_and_zero_dim():
    t = torch.randn(16, 16).t()
    assert _absmax(t) == pytest.approx(t.abs().max().item())
    assert _absmax(torch.tensor(-4.0)) == pytest.approx(4.0)


def test_absmax_propagates_nan_so_the_isfinite_gate_still_fires():
    assert _absmax(torch.tensor([1.0, float("nan")])) != _absmax(torch.tensor([1.0, 2.0]))
    assert torch.isnan(torch.tensor(_absmax(torch.tensor([1.0, float("nan")])))).item()


def test_tripwire_rejects_finite_garbage_the_isfinite_gate_misses():
    W = torch.randn(64, 64) * 0.02
    stats = LoraStats(
        module = None,
        lora_A = torch.full((8, 64), 1e6),
        lora_B = torch.full((64, 8), 1e6),
        alpha = 1.0,
    )
    with pytest.raises(ValueError, match = r"\|W\|max"):
        _merge_lora(W.clone(), stats, "o_proj")


@pytest.mark.parametrize("rank", [8, 16, 64])
@pytest.mark.parametrize("alpha_ratio", [1, 2, 4, 16, 64])
@pytest.mark.parametrize("base_scale", [1e-3, 1e-1, 1.0])
@pytest.mark.parametrize("mode", ["lora", "dora", "rslora"])
def test_tripwire_accepts_healthy_merges(rank, alpha_ratio, base_scale, mode):
    torch.manual_seed(rank * 977 + alpha_ratio * 13 + int(base_scale * 1000))
    d_in, d_out = 256, 192
    W = torch.randn(d_out, d_in) * base_scale
    lora_A = torch.randn(rank, d_in) / (rank ** 0.5)
    lora_B = torch.randn(d_out, rank) * 0.05
    alpha = (alpha_ratio * rank) / (rank ** 0.5) if mode == "rslora" else float(alpha_ratio)

    stats = LoraStats(module = None, lora_A = lora_A, lora_B = lora_B, alpha = alpha)
    if mode == "dora":
        stats.magnitude = torch.linalg.norm(W + alpha * (lora_B @ lora_A), dim = 1)

    _merge_lora(W.clone(), stats, f"r{rank}_a{alpha_ratio}_{mode}")


def test_tripwire_does_not_fire_on_an_all_zero_base_weight():
    """A zero base leaves only the absolute floor; the ratio arm must not reject
    every adapter."""
    stats = LoraStats(
        module = None,
        lora_A = torch.randn(8, 64),
        lora_B = torch.randn(64, 8),
        alpha = 2.0,
    )
    _merge_lora(torch.zeros(64, 64), stats, "newly_resized_rows")


def test_merge_without_an_adapter_is_untouched():
    W = torch.randn(4, 4, dtype = torch.bfloat16)
    assert _merge_lora(W, LoraStats(None, None, None, 0), "passthrough") is W

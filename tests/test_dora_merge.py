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

"""DoRA (use_dora=True) merge support in the safetensors merge path.

The dense merge must (1) no longer raise a key-mismatch on a DoRA adapter (the magnitude vector
is now captured), and (2) produce the same merged weight as PEFT's own DoRA merge. MoE-expert
DoRA is explicitly refused (fail loud) rather than silently dropping the magnitude.
"""
import copy

import pytest
import torch
import torch.nn as nn

from unsloth_zoo.saving_utils import create_lora_statistics, _merge_lora, LoraStats


class _Tiny(nn.Module):
    def __init__(self, d_in=32, d_out=24):
        super().__init__()
        self.q_proj = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        return self.q_proj(x)


def _find_q_stats(lora_weights):
    for v in lora_weights.values():
        if v.lora_A is not None and v.lora_B is not None:
            return v
    return None


def test_dora_merge_matches_peft():
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    base = _Tiny().to(torch.float32)
    W0 = base.q_proj.weight.detach().clone()

    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj"], use_dora=True)
    pm = get_peft_model(copy.deepcopy(base), cfg)

    # Give the adapter a non-trivial delta and magnitude so DoRA actually rescales.
    for n, p in pm.named_parameters():
        if n.endswith("lora_B.default.weight"):
            with torch.no_grad():
                p.copy_(torch.randn_like(p) * 0.1)
        if n.endswith("lora_magnitude_vector.default.weight"):
            with torch.no_grad():
                p.add_(torch.randn_like(p) * 0.1)

    # Ground truth: PEFT's own DoRA merge.
    merged_peft = copy.deepcopy(pm).merge_and_unload()
    W_peft = None
    for n, p in merged_peft.named_parameters():
        if n.endswith("q_proj.weight"):
            W_peft = p.detach().float().clone()
    assert W_peft is not None

    # Unsloth merge path: capture stats (must NOT raise on DoRA) then fold via _merge_lora.
    result = create_lora_statistics(pm, merge_into_original=True, return_state_dict=True)
    lora_weights = result[0] if isinstance(result, tuple) else result
    stats = _find_q_stats(lora_weights)
    assert stats is not None
    assert stats.magnitude is not None, "DoRA magnitude was not captured"

    W_uns = _merge_lora(W0.clone(), stats, "q_proj").cpu().float()

    max_abs = (W_uns - W_peft).abs().max().item()
    assert torch.allclose(W_uns, W_peft, atol=1e-4, rtol=1e-4), f"max abs diff {max_abs}"
    # Sanity: DoRA actually changed the weight vs the plain base.
    assert (W_uns - W0.float()).abs().max().item() > 1e-3


def test_plain_lora_unaffected():
    """A non-DoRA adapter has magnitude None and merges as W0 + alpha*BA."""
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(1)
    base = _Tiny().to(torch.float32)
    W0 = base.q_proj.weight.detach().clone()
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj"], use_dora=False)
    pm = get_peft_model(copy.deepcopy(base), cfg)
    for n, p in pm.named_parameters():
        if n.endswith("lora_B.default.weight"):
            with torch.no_grad():
                p.copy_(torch.randn_like(p) * 0.1)

    result = create_lora_statistics(pm, merge_into_original=True, return_state_dict=True)
    lora_weights = result[0] if isinstance(result, tuple) else result
    stats = _find_q_stats(lora_weights)
    assert stats is not None and stats.magnitude is None

    W_uns = _merge_lora(W0.clone(), stats, "q_proj").cpu().float()
    expected = W0.float() + stats.alpha * (stats.lora_B.float() @ stats.lora_A.float())
    assert torch.allclose(W_uns, expected, atol=1e-5)


def test_dora_on_moe_expert_is_refused():
    from unsloth_zoo.saving_utils import _merge_moe_fused_gate_up_expert

    E, rank, H, I = 4, 4, 8, 6
    gate_up_W = torch.randn(E, 2 * I, H)
    A = torch.randn(E * rank, H)
    B = torch.randn(2 * I, E * rank)
    stats = LoraStats(None, A, B, 1.0, magnitude=torch.randn(2 * I))
    with pytest.raises(RuntimeError, match="DoRA"):
        _merge_moe_fused_gate_up_expert(gate_up_W, stats, torch.float32)


def test_dora_on_embedding_is_refused():
    """DoRA on an Embedding captures a magnitude but no mergeable lora_A/lora_B (PEFT stores the
    embedding delta as lora_embedding_A/lora_embedding_B). _merge_lora cannot fold it, so the
    magnitude would be silently dropped and assert_same_keys (which now ignores magnitude keys)
    would not catch it. create_lora_statistics must fail loud instead."""
    from peft import LoraConfig, get_peft_model

    class _WithEmbed(nn.Module):
        def __init__(self, vocab=16, hidden=8):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab, hidden)
            self.q_proj = nn.Linear(hidden, hidden, bias=False)

        def forward(self, ids):
            return self.q_proj(self.embed_tokens(ids))

    torch.manual_seed(0)
    cfg = LoraConfig(r=4, lora_alpha=8, use_dora=True,
                     target_modules=["embed_tokens", "q_proj"])
    pm = get_peft_model(_WithEmbed(), cfg)
    with pytest.raises(RuntimeError, match="DoRA"):
        create_lora_statistics(pm, merge_into_original=True, return_state_dict=True)


def test_dora_on_mxfp4_packed_moe_is_refused(monkeypatch, tmp_path):
    """The MXFP4 packed-MoE rewrite (GPT-OSS gate_up_proj/down_proj) merges experts via
    _merge_lora directly, not the dense _merge_moe_*_expert helpers, so it must apply the same
    DoRA refuse. Without it a use_dora adapter on the experts bypasses the guard and _merge_lora
    fails with an opaque shape error on the 3D expert group instead of the clear message."""
    import unsloth_zoo.saving_utils as su
    from safetensors.torch import save_file

    E, rank, H, I = 2, 4, 8, 6
    base = "model.layers.0.mlp.experts.gate_up_proj"
    # Minimal on-disk mxfp4-style file: one packed expert group (blocks + scales).
    save_file(
        {
            base + "_blocks": torch.zeros(E, 2 * I, H // 2, dtype=torch.uint8),
            base + "_scales": torch.zeros(E, 2 * I, 1, dtype=torch.uint8),
        },
        str(tmp_path / "model.safetensors"),
    )
    # Patch the heavy mxfp4 machinery; only the DoRA-refuse behavior is under test.
    monkeypatch.setattr(su, "_convert_lora_keys_to_safetensor_format",
                        lambda lw, keys, model_class_name=None: lw)
    # _merge_and_overwrite_lora_mxfp4 re-imports convert_moe_packed_tensors fresh from
    # transformers.integrations.mxfp4 (so a runtime Unsloth patch is picked up), so stub it
    # there, not just on the saving_utils module, otherwise the real converter runs on this
    # deliberately-minimal packed group and asserts on the block/scale shapes before the
    # DoRA guard is reached.
    _stub_convert = lambda b, s, rows_per_chunk=None: torch.randn(E, 2 * I, H)
    monkeypatch.setattr(su, "convert_moe_packed_tensors", _stub_convert)
    from transformers.integrations import mxfp4 as _mxfp4_mod
    monkeypatch.setattr(_mxfp4_mod, "convert_moe_packed_tensors", _stub_convert)
    monkeypatch.setattr(su, "_choose_mxfp4_processing_strategy",
                        lambda b, s: ("cuda", 0, 1024))

    A = torch.randn(E * rank, H)
    B = torch.randn(2 * I, E * rank)
    stats = LoraStats(None, A, B, 1.0, magnitude=torch.randn(2 * I))
    with pytest.raises(RuntimeError, match="DoRA"):
        su._merge_and_overwrite_lora_mxfp4(
            str(tmp_path), "model.safetensors", {base: stats},
            torch.float32, "GptOssForCausalLM",
            base_model_is_quantized=True, quant_type="mxfp4",
        )

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

"""merged_16bit on a DeepSeek-V4.1 released-layout checkpoint must place every native adapter; before, only names
both layouts share were merged and other deltas were dropped silently."""

from __future__ import annotations

import collections
from dataclasses import dataclass

import pytest
import torch
from safetensors.torch import load_file, save_file

import unsloth_zoo.saving_utils as zs
from unsloth_zoo.saving_utils import LoraStats, _merge_and_overwrite_lora, _reset_moe_merge_state

# The one-to-one renames of unsloth/models/deepseek_v41/convert.py (released name -> module name).
RENAMES = {
    r"^head\.weight": "lm_head.weight",
    r"\.attn\.": ".self_attn.",
    r"\.ffn\.": ".mlp.",
    r"^embed\.weight$": "embed_tokens.weight",
    r"\.attn_norm\.": ".input_layernorm.",
    r"\.ffn_norm\.": ".post_attention_layernorm.",
    r"\.wq_a\.": ".q_a_proj.",
    r"\.q_norm\.": ".q_a_norm.",
    r"\.wq_b\.": ".q_b_proj.",
    r"\.self_attn\.wkv\.": ".self_attn.kv_proj.",
    r"\.compressor\.wkv\.": ".compressor.kv_proj.",
    r"\.compressor\.wgate\.": ".compressor.gate_proj.",
    r"\.wo_a\.": ".o_a_proj.",
    r"\.wo_b\.": ".o_b_proj.",
    r"\.shared_experts\.w1\.": ".shared_experts.gate_proj.",
    r"\.shared_experts\.w2\.": ".shared_experts.down_proj.",
    r"\.shared_experts\.w3\.": ".shared_experts.up_proj.",
}
H, I, E, R, ALPHA = 16, 8, 4, 2, 0.5
# released name -> (native LoRA module, out, in)
DENSE = {
    "layers.0.attn.wq_a": ("model.layers.0.self_attn.q_a_proj", 12, H),
    "layers.0.attn.wq_b": ("model.layers.0.self_attn.q_b_proj", 20, 12),
    "layers.0.attn.wkv": ("model.layers.0.self_attn.kv_proj", 10, H),
    "layers.0.attn.wo_a": ("model.layers.0.self_attn.o_a_proj", 6, 10),
    "layers.0.attn.wo_b": ("model.layers.0.self_attn.o_b_proj", H, 6),
    "layers.0.attn.compressor.wkv": ("model.layers.0.self_attn.compressor.kv_proj", 10, H),
    "layers.0.attn.compressor.wgate": ("model.layers.0.self_attn.compressor.gate_proj", 10, H),
    "layers.0.ffn.shared_experts.w1": ("model.layers.0.mlp.shared_experts.gate_proj", I, H),
    "layers.0.ffn.shared_experts.w3": ("model.layers.0.mlp.shared_experts.up_proj", I, H),
    "layers.0.ffn.shared_experts.w2": ("model.layers.0.mlp.shared_experts.down_proj", H, I),
    "layers.1.engram.wkv": ("model.layers.1.engram.wkv", 10, H),
    "head": ("lm_head", 32, H),
}
UNTOUCHED = {"embed.weight": (32, H), "layers.0.attn_norm.weight": (H,), "layers.0.ffn.gate.weight": (E, H)}


@dataclass
class _Experts:
    num_experts: int


def _shard(path):
    g = torch.Generator().manual_seed(0)
    t = {k + ".weight": torch.randn(o, i, generator = g) for k, (_, o, i) in DENSE.items()}
    for e in range(E):
        p = f"layers.0.ffn.experts.{e}"
        t[f"{p}.w1.weight"] = torch.randn(I, H, generator = g)
        t[f"{p}.w3.weight"] = torch.randn(I, H, generator = g)
        t[f"{p}.w2.weight"] = torch.randn(H, I, generator = g)
    t.update({k: torch.randn(*s, generator = g) for k, s in UNTOUCHED.items()})
    save_file(t, path)
    return t


def _lora():
    g = torch.Generator().manual_seed(1)
    lw = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    for _, (native, o, i) in DENSE.items():
        lw[native] = LoraStats(None, torch.randn(R, i, generator = g), torch.randn(o, R, generator = g), ALPHA)
    experts = "model.layers.0.mlp.experts"
    lw[experts + ".base_layer"] = LoraStats(   # gate_up_proj [E, 2I, H]
        _Experts(E), torch.randn(E * R, H, generator = g), torch.randn(2 * I, E * R, generator = g), ALPHA,
    )
    lw[experts] = LoraStats(                   # down_proj [E, H, I]
        _Experts(E), torch.randn(E * R, I, generator = g), torch.randn(H, E * R, generator = g), ALPHA,
    )
    return lw


def _expert_delta(stats, e):
    a = stats.lora_A[e * R:(e + 1) * R].double()
    b = stats.lora_B.reshape(stats.lora_B.shape[0], R, E)[:, :, e].double()
    return stats.alpha * (b @ a)


@pytest.fixture
def released_class(monkeypatch):
    cls = type("DeepseekV41ForCausalLM", (), {"_unsloth_checkpoint_conversion_mapping": dict(RENAMES)})
    monkeypatch.setitem(zs._MERGE_MODEL_CLASSES, cls.__name__, cls)
    return cls


def test_released_layout_merge_places_every_native_adapter(tmp_path, released_class):
    path = str(tmp_path / "model.safetensors")
    base = _shard(path)
    lw = _lora()
    _reset_moe_merge_state()
    _merge_and_overwrite_lora(
        save_directory = str(tmp_path), filename = "model.safetensors", lora_weights = lw,
        output_dtype = torch.float32, model_class_name = released_class.__name__,
    )
    _reset_moe_merge_state()
    merged = load_file(path)
    assert set(merged) == set(base)

    expected = {}
    for released, (native, _, _) in DENSE.items():
        s = lw[native]
        expected[released + ".weight"] = base[released + ".weight"].double() + s.alpha * (s.lora_B.double() @ s.lora_A.double())
    gu, dn = lw["model.layers.0.mlp.experts.base_layer"], lw["model.layers.0.mlp.experts"]
    for e in range(E):
        p = f"layers.0.ffn.experts.{e}"
        d = _expert_delta(gu, e)
        expected[f"{p}.w1.weight"] = base[f"{p}.w1.weight"].double() + d[:I]
        expected[f"{p}.w3.weight"] = base[f"{p}.w3.weight"].double() + d[I:]
        expected[f"{p}.w2.weight"] = base[f"{p}.w2.weight"].double() + _expert_delta(dn, e)

    missed = sorted(k for k in expected if torch.equal(merged[k], base[k]))
    assert not missed, f"adapters dropped on the released layout: {missed}"
    worst = max(float((merged[k].double() - v).abs().max()) for k, v in expected.items())
    assert worst < 1e-4, worst
    for k in UNTOUCHED:
        assert torch.equal(merged[k], base[k]), k


def test_native_layout_shard_still_merges_by_module_name(tmp_path, released_class):
    """The same class over a checkpoint already in the native layout (transformers 4.x converts the release first)."""
    path = str(tmp_path / "model.safetensors")
    g = torch.Generator().manual_seed(2)
    base = {
        "model.layers.0.self_attn.q_a_proj.weight": torch.randn(12, H, generator = g),
        "model.layers.0.mlp.shared_experts.up_proj.weight": torch.randn(I, H, generator = g),
        "model.layers.0.input_layernorm.weight": torch.randn(H, generator = g),
        "lm_head.weight": torch.randn(32, H, generator = g),
    }
    save_file(base, path)
    lw = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    for native, (o, i) in {
        "model.layers.0.self_attn.q_a_proj": (12, H),
        "model.layers.0.mlp.shared_experts.up_proj": (I, H),
        "lm_head": (32, H),
    }.items():
        lw[native] = LoraStats(None, torch.randn(R, i, generator = g), torch.randn(o, R, generator = g), ALPHA)
    _merge_and_overwrite_lora(
        save_directory = str(tmp_path), filename = "model.safetensors", lora_weights = lw,
        output_dtype = torch.float32, model_class_name = released_class.__name__,
    )
    merged = load_file(path)
    for native in lw:
        s, k = lw[native], native + ".weight"
        want = base[k].double() + s.alpha * (s.lora_B.double() @ s.lora_A.double())
        assert float((merged[k].double() - want).abs().max()) < 1e-4, k
    assert torch.equal(merged["model.layers.0.input_layernorm.weight"], base["model.layers.0.input_layernorm.weight"])

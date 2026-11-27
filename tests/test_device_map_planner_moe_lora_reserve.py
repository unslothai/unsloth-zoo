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

"""offload_layers = "auto" on a 128-expert MoE QLoRA, one 15 GiB T4 (meta device, no GPU, no weights).

Inkling-2-Mini's shape: 24 decoder layers of 0.49 GiB NF4 (128 experts, hidden 2304, expert width 1152), five layer
signatures (so the slot pool holds about ten slots), an untied 201024-row vocabulary, a T4 with 14.41 GiB free and
bitsandbytes' 10% load haircut. Rank-16 LoRA on every expert is 427M params: at the 20 bytes per param a 32-bit
AdamW takes the plan refused every count, although 20 swapped layers train at sequence 512 with a 12.06 GB peak."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from unsloth_zoo import device_map_planner as dmp
from unsloth_zoo.device_map_planner import DeviceMapInfeasible, plan_block_swap

_GiB = 1024 ** 3
_HIDDEN, _WIDTH, _EXPERTS, _VOCAB = 2304, 1152, 128, 201024
# Inkling-2-Mini's layer signatures, first to last (attention-residual blocks and message gates vary).
_PATTERN = [0, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 4, 1, 1, 1, 1, 1, 2]
# The planner's own sizes for unsloth/Inkling-2-Mini-unsloth-bnb-4bit (meta model, Unsloth's MoE NF4 patches on).
_LAYER_BYTES = {0: 524398848, 1: 524408064, 2: 524493056, 3: 531485952, 4: 531570944}
_EMBED_BYTES, _HEAD_BYTES, _TOTAL_BYTES = 926323200, 921867264, 14681366656


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(_EXPERTS, 2 * _WIDTH, _HIDDEN, dtype = torch.bfloat16))
        self.down_proj = nn.Parameter(torch.empty(_EXPERTS, _HIDDEN, _WIDTH, dtype = torch.bfloat16))


class _Layer(nn.Module):
    def __init__(self, kind):
        super().__init__()
        # Attention, gates and norms: 78968 LoRA inputs + outputs per layer, as the real decoder has.
        self.attn = nn.ModuleList([nn.Linear(_HIDDEN, _HIDDEN, bias = False) for _ in range(17)])
        self.attn.append(nn.Linear(316, 316, bias = False))
        self.mlp = nn.Module()
        self.mlp.experts = _Experts()
        # A small kind-specific tensor: a different signature, so its own slots in the pool.
        self.extra = nn.Parameter(torch.empty(kind + 1, _HIDDEN))


class _MoE(nn.Module):
    _no_split_modules = ["_Layer"]

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size = _HIDDEN, vocab_size = _VOCAB, tie_word_embeddings = False)
        self.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN, dtype = torch.bfloat16)
        self.layers = nn.ModuleList([_Layer(k) for k in _PATTERN])
        self.norm = nn.LayerNorm(_HIDDEN)
        self.lm_head = nn.Linear(_HIDDEN, _VOCAB, bias = False, dtype = torch.bfloat16)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


class _Bnb4bit:
    """bitsandbytes 4-bit's adjust_max_memory: 10% held back for its load-time buffers."""

    def adjust_max_memory(self, max_memory):
        return {k: v * 90 // 100 for k, v in max_memory.items()}


def _nf4_sizes(model, hf_quantizer = None):
    # The real checkpoint's bytes (NF4 expert stacks and linears), not the bf16 placeholders' shapes.
    layers = {f"layers.{i}": _LAYER_BYTES[k] for i, k in enumerate(_PATTERN)}
    return {"": _TOTAL_BYTES, "layers": sum(layers.values()), **layers,
            "embed_tokens": _EMBED_BYTES, "lm_head": _HEAD_BYTES}


@pytest.fixture
def t4_moe(monkeypatch):
    monkeypatch.setattr(dmp, "_compute_module_sizes", _nf4_sizes)
    with torch.device("meta"):
        model = _MoE()
    return model, dict(model = model, hf_quantizer = _Bnb4bit(), max_memory = {0: int(14.41 * _GiB)},
                       offload_embedding = True)


@pytest.mark.parametrize("seq_len", [512, 1024])
def test_auto_picks_a_count_for_moe_qlora_on_one_t4(t4_moe, seq_len):
    model, kw = t4_moe
    plan = plan_block_swap(seq_len = seq_len, **kw)
    # 20 swapped layers trained at sequence 512 with a 12.06 GB peak of the T4's 14.56 GiB; 1024 needs one more.
    assert 0 < plan.layers < plan.total_layers == 24
    assert plan.layers == {512: 20, 1024: 21}[seq_len]
    assert plan.offload_embedding
    assert "adamw_8bit" in plan.describe().splitlines()[0]
    assert any("32-bit AdamW" in n for n in plan.notes)


def test_lora_state_is_charged_at_adamw_8bit_rates(t4_moe):
    model, kw = t4_moe
    from unsloth_zoo.block_swap import estimate_training_reserve_bytes, find_decoder_layers, lora_param_count
    params = lora_param_count(find_decoder_layers(model), 16)
    assert 420e6 < params < 430e6
    plan = plan_block_swap(seq_len = 1024, **kw)
    assert plan.reserve_bytes == estimate_training_reserve_bytes(
        model.config, 1024, 1, extra_bytes = params * 10 + params // 32
    )


def test_32bit_adamw_reserve_still_refuses(t4_moe):
    # The same card with the 32-bit AdamW reserve passed explicitly: no count fits, and the plan says so.
    model, kw = t4_moe
    from unsloth_zoo.block_swap import estimate_training_reserve_bytes, find_decoder_layers, lora_param_count
    params = lora_param_count(find_decoder_layers(model), 16)
    reserve = estimate_training_reserve_bytes(model.config, 1024, 1, extra_bytes = params * 20)
    with pytest.raises(DeviceMapInfeasible, match = "23 of 24"):
        plan_block_swap(seq_len = 1024, reserve_bytes = reserve, **kw)


def test_explicit_reserve_has_no_basis(t4_moe):
    model, kw = t4_moe
    plan = plan_block_swap(seq_len = 1024, reserve_bytes = 2 * _GiB, **kw)
    assert plan.reserve_basis == "" and "adamw_8bit" not in plan.describe()

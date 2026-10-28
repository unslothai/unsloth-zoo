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

"""Automatic LoRA targets on a model with one layer of a kind (leaves seen once)."""
import re

import pytest
import torch
import torch.nn as nn

from unsloth_zoo.peft_utils import get_peft_regex


def _nemotron_h(block_types):
    transformers = pytest.importorskip("transformers")
    NemotronHConfig = getattr(transformers, "NemotronHConfig", None)
    if NemotronHConfig is None:
        pytest.skip(reason="native nemotron_h needs transformers 5.x")
    cfg = NemotronHConfig(
        vocab_size=128, hidden_size=16, intermediate_size=32, moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=32, n_routed_experts=4, num_experts_per_tok=2,
        n_group=1, topk_group=1, num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        mamba_num_heads=8, mamba_head_dim=4, ssm_state_size=16, n_groups=1, chunk_size=16,
        layers_block_type=block_types, use_mamba_kernels=False,
    )
    with torch.device("meta"):
        return transformers.NemotronHForCausalLM(cfg)


def _targets(model):
    regex = get_peft_regex(model)
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    return regex, {n.rsplit(".", 1)[-1] for n in names if re.fullmatch(regex, n)}, \
        [n for n in names if re.fullmatch(regex, n)]


def test_one_layer_per_kind_keeps_layer_targets():
    regex, leaves, _ = _targets(_nemotron_h(["mamba", "attention", "moe"]))
    assert not regex.endswith(r"\.(?:))"), regex
    for leaf in ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "up_proj", "down_proj"):
        assert leaf in leaves, (leaf, sorted(leaves))
    assert "model.layers.2.mixer.shared_experts.up_proj" in _targets(_nemotron_h(["mamba", "attention", "moe"]))[2]
    assert "out_proj" not in leaves and "lm_head" not in leaves and "gate" not in leaves


def test_single_layer_matches_two_layer_leaf_set():
    _, one, _ = _targets(_nemotron_h(["mamba", "attention", "moe"]))
    _, two, _ = _targets(_nemotron_h(["mamba", "attention", "moe"] * 2))
    assert one == two, (sorted(one), sorted(two))


def test_top_level_single_leaf_still_excluded():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(8, 8)

    class Toy(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.config = None
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Block() for _ in range(n)])
            self.lm_head = nn.Linear(8, 8)
            self.mlp_proj = nn.Linear(8, 8)

    for n in (1, 3):
        toy = Toy(n)
        regex = get_peft_regex(toy)
        hits = [x for x, m in toy.named_modules() if isinstance(m, nn.Linear) and re.fullmatch(regex, x)]
        assert hits == [f"model.layers.{i}.self_attn.q_proj" for i in range(n)], (n, hits, regex)


def test_shared_experts_do_not_hide_linear_attention():
    # Qwen3-Next shape: reaching shared experts once skipped the fallback that kept linear_attn.*.
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_attn = nn.Module()
            self.linear_attn.in_proj_qkvz = nn.Linear(8, 8)
            self.linear_attn.out_proj = nn.Linear(8, 8)
            self.mlp = nn.Module()
            self.mlp.gate = nn.Linear(8, 4)
            self.mlp.shared_expert = nn.Module()
            self.mlp.shared_expert.up_proj = nn.Linear(8, 8)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = None
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Block() for _ in range(2)])
            self.lm_head = nn.Linear(8, 8)

    toy = Toy()
    regex = get_peft_regex(toy, target_modules=["in_proj_qkvz", "out_proj", "up_proj"],
                           finetune_vision_layers=False)
    hits = {x for x, m in toy.named_modules() if isinstance(m, nn.Linear) and re.fullmatch(regex, x)}
    assert hits == {f"model.layers.{i}.{leaf}" for i in range(2) for leaf in
                    ("linear_attn.in_proj_qkvz", "linear_attn.out_proj", "mlp.shared_expert.up_proj")}, hits


def test_text_branch_stays_inside_the_decoder_stack():
    regex = get_peft_regex(_nemotron_h(["mamba", "attention", "moe"]), finetune_vision_layers=False)
    assert re.fullmatch(regex, "model.layers.1.mixer.q_proj")
    for name in ("vision_tower.model.layers.1.mixer.q_proj", "model.visual.blocks.0.attn.q_proj",
                 "model.layers.1.mixer.q_proj_drop", "model.layers.2.mixer.gate"):
        assert not re.fullmatch(regex, name), name

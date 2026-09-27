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

"""all-linear / automatic LoRA targets on a model with one layer of a kind.

get_peft_regex drops a Linear leaf name that occurs once, to skip heads like lm_head. A
config with a single attention / Mamba / MoE layer (tiny NemotronH test models, hybrids with
one attention block) then lost every per-layer target, and the regex ended in an empty
group: only the routed experts (target_parameters) got LoRA.
"""
import re

import pytest
import torch
import torch.nn as nn

from unsloth_zoo.peft_utils import get_peft_regex


def _nemotron_h(block_types):
    transformers = pytest.importorskip("transformers")
    NemotronHConfig = getattr(transformers, "NemotronHConfig", None)
    if NemotronHConfig is None:
        pytest.skip("transformers has no nemotron_h")
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
    # Mamba out_proj stays excluded (fused kernels), lm_head / router never targeted.
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
            self.mlp_proj = nn.Linear(8, 8)  # a lone head outside any layer stack

    for n in (1, 3):
        toy = Toy(n)
        regex = get_peft_regex(toy)
        hits = [x for x, m in toy.named_modules() if isinstance(m, nn.Linear) and re.fullmatch(regex, x)]
        assert hits == [f"model.layers.{i}.self_attn.q_proj" for i in range(n)], (n, hits, regex)


def test_regex_unchanged_without_shared_experts():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.up_proj = nn.Linear(8, 8)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = None
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Block() for _ in range(2)])

    assert "shared_expert" not in get_peft_regex(Toy())

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

"""The MoE router must not be an automatically chosen LoRA target.

A router leaf is an nn.Linear, so the automatic target search counted it as an
ordinary projection. Llama 4's router subclasses nn.Linear but returns
(scores, logits), so PEFT's lora.Linear.forward reads `.dtype` off a tuple and
the model cannot take a single step. Adapting a router is wrong regardless:
it decides which experts run, not what they compute.

Only the automatic path changes. An explicit target_modules list naming the
router is a request, and is still honoured. CPU only.
"""

from __future__ import annotations

import re

import pytest
import torch.nn as nn

from unsloth_zoo.peft_utils import get_peft_regex

N_LAYERS = 4


class _Cfg:
    def __init__(self, model_type):
        self.model_type = model_type
        self.architectures = [model_type]


def _build(router_leaf="router", model_type="llama4"):
    """A model shaped the way get_peft_regex reads one: named nn.Linear leaves."""
    model = nn.Module()
    model.config = _Cfg(model_type)
    layers = nn.ModuleList()
    for _ in range(N_LAYERS):
        block = nn.Module()
        attn = nn.Module()
        for leaf in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(attn, leaf, nn.Linear(8, 8))
        block.self_attn = attn
        mlp = nn.Module()
        for leaf in ("gate_proj", "up_proj", "down_proj"):
            setattr(mlp, leaf, nn.Linear(8, 8))
        if router_leaf is not None:
            setattr(mlp, router_leaf, nn.Linear(8, 4))
        block.mlp = mlp
        layers.append(block)
    inner = nn.Module()
    inner.layers = layers
    model.model = inner
    model.lm_head = nn.Linear(8, 32)
    return model


def _targets(model, **kwargs):
    regex = get_peft_regex(model, finetune_vision_layers=False, **kwargs)
    return regex, sorted(
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and re.fullmatch(regex, name)
    )


def test_router_is_not_auto_selected():
    _, targets = _targets(_build())
    routers = [t for t in targets if t.rsplit(".")[-1] == "router"]
    assert not routers, f"router auto-selected as a LoRA target: {routers}"
    # and the real projections are all still there, so the skip is not a blanket drop
    leaves = {t.rsplit(".")[-1] for t in targets}
    assert leaves == {"q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"}
    assert len(targets) == N_LAYERS * 7


def test_a_model_without_a_router_gets_an_unchanged_regex():
    """The skip must be invisible to every non-MoE model."""
    regex_with, targets_with = _targets(_build(router_leaf=None))
    assert "router" not in regex_with
    assert len(targets_with) == N_LAYERS * 7


def test_an_explicit_target_modules_list_is_left_alone():
    """Naming the router explicitly is a request, not an accident, so it is
    honoured. Only the automatic search skips it."""
    _, targets = _targets(_build(), target_modules=["q_proj", "router"])
    leaves = {t.rsplit(".")[-1] for t in targets}
    assert "router" in leaves, (
        "an explicitly requested router was dropped; the skip escaped the "
        "automatic branch"
    )
    assert leaves == {"q_proj", "router"}


def test_router_leaf_is_the_only_name_skipped():
    """A 'gate' leaf is the router in several families but also a plain
    projection in others, and no model in the sweep failed because of it, so it
    is deliberately still targetable. Pin that, so widening the set is a
    decision rather than a drift."""
    from unsloth_zoo.peft_utils import MOE_ROUTER_MODULES

    assert MOE_ROUTER_MODULES == frozenset(("router",))
    _, targets = _targets(_build(router_leaf="gate"))
    assert any(t.rsplit(".")[-1] == "gate" for t in targets)


@pytest.mark.parametrize("model_type", ["llama4", "qwen3_moe", "llama", "mixtral"])
def test_skip_does_not_depend_on_the_model_family(model_type):
    _, targets = _targets(_build(model_type=model_type))
    assert not [t for t in targets if t.rsplit(".")[-1] == "router"]


def test_llama4_router_returns_a_tuple_so_peft_cannot_adapt_it():
    """The upstream fact the skip exists for. If Llama 4's router ever becomes a
    plain Linear returning a tensor, this fails and the skip can be revisited."""
    transformers = pytest.importorskip("transformers")
    modeling = pytest.importorskip("transformers.models.llama4.modeling_llama4")
    router_cls = getattr(modeling, "Llama4Router", None)
    if router_cls is None:
        pytest.skip("this transformers has no Llama4Router")
    assert issubclass(router_cls, nn.Linear), (
        "Llama4Router is no longer an nn.Linear, so the automatic search would "
        "not have picked it up in the first place"
    )
    import inspect
    source = inspect.getsource(router_cls.forward)
    assert "return router_scores, router_logits" in source, (
        f"Llama4Router.forward no longer returns a tuple on transformers "
        f"{transformers.__version__}; re-check whether the skip is still needed"
    )

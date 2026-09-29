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

A router leaf is an nn.Linear, so the automatic search counted it as an ordinary
projection; Llama 4's and PhiMoE's return a tuple, which PEFT cannot adapt.
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
    leaves = {t.rsplit(".")[-1] for t in targets}
    assert leaves == {"q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"}
    assert len(targets) == N_LAYERS * 7


def test_a_model_without_a_router_gets_an_unchanged_regex():
    regex_with, targets_with = _targets(_build(router_leaf=None))
    assert "router" not in regex_with
    assert len(targets_with) == N_LAYERS * 7


def test_an_explicit_target_modules_list_is_left_alone():
    _, targets = _targets(_build(), target_modules=["q_proj", "router"])
    leaves = {t.rsplit(".")[-1] for t in targets}
    assert "router" in leaves, (
        "an explicitly requested router was dropped; the skip escaped the "
        "automatic branch"
    )
    assert leaves == {"q_proj", "router"}


def test_router_leaf_is_the_only_name_skipped():
    """Pinned so that widening the set is a decision rather than a drift."""
    from unsloth_zoo.peft_utils import MOE_ROUTER_MODULES

    assert MOE_ROUTER_MODULES == frozenset(("router",))
    _, targets = _targets(_build(router_leaf="gate"))
    assert any(t.rsplit(".")[-1] == "gate" for t in targets)


@pytest.mark.parametrize("model_type", ["llama4", "qwen3_moe", "llama", "mixtral"])
def test_skip_does_not_depend_on_the_model_family(model_type):
    _, targets = _targets(_build(model_type=model_type))
    assert not [t for t in targets if t.rsplit(".")[-1] == "router"]


def _router_class(family):
    """Matched by shape, not spelling: the class has been both Llama4Router and
    Llama4TextRouter, and a getattr on one turns a rename into a silent skip."""
    modeling = pytest.importorskip(f"transformers.models.{family}.modeling_{family}")
    found = sorted(
        (name, obj) for name, obj in vars(modeling).items()
        if isinstance(obj, type) and name.endswith("Router")
        and issubclass(obj, nn.Linear) and obj.__module__ == modeling.__name__
    )
    if not found:
        # No nn.Linear router means the automatic search never saw one here.
        pytest.skip(f"no nn.Linear router class in modeling_{family}")
    return modeling, found[0][1]


@pytest.mark.parametrize("family,config_cls_name", [
    ("llama4", "Llama4TextConfig"),
    ("phimoe", "PhimoeConfig"),
])
def test_the_router_returns_a_tuple_so_peft_cannot_adapt_it(family, config_cls_name):
    """PEFT's lora.Linear.forward does `result.dtype` on the base layer's return,
    so a tuple cannot be adapted. Asserted on the value, not on the source text."""
    import torch

    transformers = pytest.importorskip("transformers")
    modeling, router_cls = _router_class(family)
    config_cls = getattr(
        pytest.importorskip(f"transformers.models.{family}.configuration_{family}"),
        config_cls_name, None,
    )
    if config_cls is None:
        pytest.skip(f"this transformers has no {config_cls_name}")

    config = config_cls(hidden_size=8, num_local_experts=4, num_experts_per_tok=2)
    router = router_cls(config).eval()
    with torch.no_grad():
        out = router(torch.randn(4, 8))

    assert isinstance(out, tuple), (
        f"{router_cls.__name__}.forward returns {type(out).__name__} on "
        f"transformers {transformers.__version__}, not a tuple; re-check "
        f"whether the skip is still needed for {family}"
    )
    assert not hasattr(out, "dtype"), (
        "PEFT reads .dtype off this return value; if it has one, LoRA would work"
    )

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

"""Transformers MoE expert-container drift canary (#5410). Pins
fused_3d vs module_list per tracked block. CPU only."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

transformers = pytest.importorskip("transformers")


def _tiny_config_kwargs():
    return dict(
        hidden_size=32,
        intermediate_size=16,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=2,
        num_local_experts=2,  # Mixtral reads num_local_experts (default 8)
        num_experts_per_tok=1,
        max_position_embeddings=64,
        vocab_size=128,
    )


def _classify_experts_container(experts) -> str:
    gu = getattr(experts, "gate_up_proj", None)
    if isinstance(gu, nn.Parameter) and gu.dim() == 3:
        return "fused_3d"
    if isinstance(experts, nn.ModuleList):
        return "module_list"
    if isinstance(experts, nn.Module):
        for _, child in experts.named_children():
            if isinstance(child, nn.ModuleList):
                return "module_list"
        return f"other:{type(experts).__name__}"
    return f"other:{type(experts).__name__}"


# (module, block, config, experts attr). Extend on new fused-3D arches.
_TRACKED_MOE_ARCHES = [
    ("transformers.models.qwen3_moe.modeling_qwen3_moe",
     "Qwen3MoeSparseMoeBlock", "Qwen3MoeConfig", "experts"),
    ("transformers.models.mixtral.modeling_mixtral",
     "MixtralSparseMoeBlock",  "MixtralConfig",  "experts"),
]


@pytest.mark.parametrize("module_path,block_cls_name,cfg_cls_name,experts_attr",
                         _TRACKED_MOE_ARCHES)
def test_tracked_moe_arch_experts_container_shape(module_path, block_cls_name, cfg_cls_name, experts_attr):
    try:
        mod = __import__(module_path, fromlist=[block_cls_name, cfg_cls_name])
    except Exception as e:
        pytest.skip(f"{module_path} not importable: {e}")

    block_cls = getattr(mod, block_cls_name, None)
    cfg_cls   = getattr(mod, cfg_cls_name,   None)
    if block_cls is None or cfg_cls is None:
        pytest.skip(f"{block_cls_name} / {cfg_cls_name} missing")

    try:
        cfg = cfg_cls(**_tiny_config_kwargs())
    except TypeError:
        try:
            cfg = cfg_cls()
        except Exception as e:
            pytest.skip(f"{cfg_cls_name} could not be instantiated: {e}")
    except Exception as e:
        pytest.skip(f"{cfg_cls_name} could not be instantiated: {e}")

    try:
        block = block_cls(cfg)
    except Exception as e:
        pytest.skip(f"{block_cls_name} could not be instantiated: {e}")

    experts = getattr(block, experts_attr, None)
    assert experts is not None, f"{block_cls_name}.{experts_attr} missing on transformers {transformers.__version__}"

    kind = _classify_experts_container(experts)
    accepted = {"fused_3d", "module_list"}
    assert kind in accepted, (
        f"\nMoE experts-container drift on transformers {transformers.__version__}:\n"
        f"  {block_cls_name}.{experts_attr} kind={kind}; expected one of {accepted}.\n"
        f"Update _merge_moe_*_expert to handle the new container."
    )

    if kind == "fused_3d":
        gu = experts.gate_up_proj
        assert gu.dim() == 3 and gu.shape[0] == cfg.num_experts, (
            f"fused gate_up_proj shape {tuple(gu.shape)} on {block_cls_name}; expected (E, 2I, H)."
        )
    elif kind == "module_list":
        assert len(experts) == cfg.num_experts, (
            f"module_list len={len(experts)} but num_experts={cfg.num_experts}"
        )


def test_at_least_one_tracked_moe_arch_imports():
    import importlib
    ok = sum(1 for module_path, *_ in _TRACKED_MOE_ARCHES
             if _try_import(importlib, module_path))
    assert ok >= 1, (
        f"No tracked MoE arch importable on transformers {transformers.__version__}. "
        f"Update _TRACKED_MOE_ARCHES."
    )


def _try_import(importlib, module_path):
    try:
        importlib.import_module(module_path)
        return True
    except Exception:
        return False

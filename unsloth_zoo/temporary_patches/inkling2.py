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
"""Inkling 2 (inkling2 / inkling2_loop, transformers >= 5.24) MoE experts: softcapped SwiGLU over
gate and up, tanh(g / c_g) * c_g * sigmoid(g) * tanh(u / c_u) * c_u, with fp32 routed outputs;
both expert classes and fp32 reference must declare _apply_gate, else generic backends use
act_fn(gate) * up."""

from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING
from .utils import logger

__all__ = ["patch_inkling2_moe_experts", "INKLING2_EXPERT_CLASSES"]

INKLING2_EXPERT_CLASSES = (
    ("transformers.models.inkling2.modeling_inkling2", "Inkling2Experts"),
    ("transformers.models.inkling2_loop.modeling_inkling2_loop", "Inkling2LoopExperts"),
)


def patch_inkling2_moe_experts():
    import importlib

    for module_name, class_name in INKLING2_EXPERT_CLASSES:
        try:
            klass = getattr(importlib.import_module(module_name), class_name)
        except Exception:
            continue
        if not callable(getattr(klass, "_apply_gate", None)):
            # Layout changed upstream: keep the native forward rather than guess the activation.
            continue
        klass._unsloth_own_apply_gate = True
        klass._unsloth_moe_fp32_reference = True
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(f"Unsloth: {class_name} uses its own softcapped SwiGLU in the MoE backends.")


TEMPORARY_PATCHES.append(patch_inkling2_moe_experts)

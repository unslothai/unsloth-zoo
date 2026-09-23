# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Llama-4 MoE (Scout, Maverick) on Unsloth's grouped expert path.

`Llama4TextExperts` is not a transformers `@use_experts_implementation` class:
its forward takes the routed input only, views it as (E, T, H) and runs one
`torch.bmm` per projection over every expert, and `Llama4TextMoe` feeds it the
whole batch repeated E times with the sigmoid router score multiplied into the
INPUT of the selected expert (zero for the others). Two consequences:

* every token runs through every expert (E x T compute, top_k = 1 on Scout and
  Maverick), and
* the 4-bit quantizer, which claims expert stacks structurally, packed the
  stacks to uint8 which `bmm` cannot read ("shape [671088640, -1, 5120] is
  invalid" on Scout).

This patch keeps Llama-4's semantics (the score scales the expert input, not its
output, which matters because the activation is not linear) while dispatching
only the selected (token, expert) pairs through `forward_moe_backend`:

    x_k = hidden[t] * sigmoid(logit[t, e_k])        one row per (token, k)
    y   = experts(x_k, index = e_k, weight = 1)      grouped GEMM over the pairs
    out = shared_expert(hidden) + sum_k y_k

`Llama4TextExperts` stores gate_up_proj as (E, H, 2I) and down_proj as
(E, I, H), which is grouped_mm's (E, in, out) layout: the class is marked
`is_transposed = True`, the attribute transformers uses for the same layout, so
PEFT's ParamWrapper and Unsloth's LoRA extractor agree on the in/out dims.
"""
import torch

from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING
from .utils import patch_function, logger
from .moe_utils import (
    patch_param_wrapper_for_moe,
    get_forward_moe_backend,
    extract_moe_lora_weights_for_grouped_mm,
)


def _llama4_moe_lora_dims(wrapper):
    """(in, out) of the wrapped stack: gate_up_proj (E, H, 2I), down_proj (E, I, H)."""
    if wrapper is None or not hasattr(wrapper, "get_base_layer"):
        return None, None
    base = wrapper.get_base_layer()
    hidden = getattr(base, "hidden_size", None)
    expert_dim = getattr(base, "expert_dim", getattr(base, "intermediate_size", None))
    if hidden is None or expert_dim is None:
        return None, None
    name = getattr(wrapper, "parameter_name", None)
    if name == "gate_up_proj":
        return hidden, 2 * expert_dim
    if name == "down_proj":
        return expert_dim, hidden
    return None, None


def _llama4_moe_lora_extractor(wrapper, weight_A, weight_B, scaling, num_experts):
    input_dim, output_dim = _llama4_moe_lora_dims(wrapper)
    return extract_moe_lora_weights_for_grouped_mm(
        wrapper, weight_A, weight_B, scaling, num_experts,
        input_dim = input_dim, output_dim = output_dim,
        model_name = "Llama-4 MoE", enable_logging = UNSLOTH_ENABLE_LOGGING, logger_obj = logger,
    )


@torch.compiler.disable
def Llama4TextMoe_forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_scores, router_logits = self.router(hidden_states)
    # router_scores is (T, E): sigmoid(logit) for the top_k experts, 0 elsewhere,
    # so its own top_k recovers the selected experts and their scores exactly.
    top_k = self.top_k
    top_k_weights, top_k_index = torch.topk(router_scores, top_k, dim = -1)
    n_tokens = hidden_states.shape[0]
    # Llama-4 scales the input of the selected expert by its score.
    routed_in = hidden_states.unsqueeze(1) * top_k_weights.unsqueeze(-1).to(hidden_states.dtype)
    routed_in = routed_in.reshape(n_tokens * top_k, self.hidden_dim)
    ones = torch.ones(n_tokens * top_k, 1, device = hidden_states.device, dtype = router_scores.dtype)
    routed_out = self.experts(routed_in, top_k_index.reshape(-1, 1), ones)
    routed_out = routed_out.reshape(n_tokens, top_k, self.hidden_dim).sum(dim = 1)
    out = self.shared_expert(hidden_states)
    # The model adds in place into the shared expert's output, which keeps the
    # residual stream in the model dtype under autocast; a promoting add would
    # turn it float32 and the next layer's grouped GEMM would see mixed dtypes.
    out = out + routed_out.to(out.dtype)
    return out, router_logits


def patch_llama4_moe():
    """Route Llama-4's experts through Unsloth's MoE backend (Split LoRA, 4-bit)."""
    patch_param_wrapper_for_moe()

    try:
        from transformers.models.llama4.modeling_llama4 import Llama4TextExperts, Llama4TextMoe
    except Exception:
        return

    if getattr(Llama4TextExperts, "_unsloth_already_patched", False):
        return

    # The two forwards only work together: the MoE forward hands the experts routing
    # indices and weights, so install both or neither.
    original_experts_forward = Llama4TextExperts.__dict__.get("forward")
    original_moe_forward = Llama4TextMoe.__dict__.get("forward")
    ok = patch_function(Llama4TextMoe, "forward", Llama4TextMoe_forward)
    # Different signature from the model's forward, so force the patch.
    ok = ok and patch_function(Llama4TextExperts, "forward", get_forward_moe_backend(), force = True)
    if not ok:
        if original_moe_forward is not None:
            Llama4TextMoe.forward = original_moe_forward
        if original_experts_forward is not None:
            Llama4TextExperts.forward = original_experts_forward
    else:
        # Separated LoRA on the stacks reads its dims from the module, not the
        # (E, in, out) shape that the shared extractor would otherwise misread.
        Llama4TextExperts._unsloth_lora_extractor_fn = staticmethod(_llama4_moe_lora_extractor)
        # The stacks are (E, in, out). transformers' own name for that layout.
        Llama4TextExperts.is_transposed = True
        Llama4TextExperts.is_concatenated = True   # gate, up = chunk(2)
        Llama4TextExperts.has_bias = False
        Llama4TextExperts.has_gate = True
        Llama4TextExperts._unsloth_already_patched = True

    if UNSLOTH_ENABLE_LOGGING:
        if ok:
            logger.info("Unsloth: Patched Llama-4 MoE experts for Split LoRA support.")
        else:
            logger.warning("Unsloth: Could not patch Llama-4 MoE experts; the model's own dense expert path is used.")
pass

TEMPORARY_PATCHES.append(patch_llama4_moe)

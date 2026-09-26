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

"""Route Llama-4 MoE (Scout, Maverick) through Unsloth's grouped expert path instead of dense bmm over every expert."""
import torch

from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING
from .utils import patch_function, logger
from .moe_utils import (
    patch_param_wrapper_for_moe,
    get_forward_moe_backend,
    extract_moe_lora_weights_for_grouped_mm,
)


def _llama4_moe_lora_dims(wrapper):
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
    experts = self.experts
    # LoRA on the stacks wraps the experts in PEFT's ParamWrapper; the flag lives on the base class.
    base = experts.get_base_layer() if hasattr(experts, "get_base_layer") else experts
    if not getattr(type(base), "_unsloth_already_patched", False):
        # Pre-quantized per-expert checkpoints get SequentialLlama4TextExperts (hidden states only): run dense routing.
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_scores, router_logits = self.router(hidden_states)
        routed_in = hidden_states.repeat(router_scores.shape[1], 1)
        routed_in = routed_in * router_scores.transpose(0, 1).reshape(-1, 1)
        routed_out = self.experts(routed_in)
        out = self.shared_expert(hidden_states)
        out.add_(routed_out.reshape(router_scores.shape[1], -1, routed_out.shape[-1]).sum(dim = 0))
        return out, router_logits
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_scores, router_logits = self.router(hidden_states)
    top_k = self.top_k
    top_k_weights, top_k_index = torch.topk(router_scores, top_k, dim = -1)
    n_tokens = hidden_states.shape[0]
    routed_in = hidden_states.unsqueeze(1) * top_k_weights.unsqueeze(-1).to(hidden_states.dtype)
    routed_in = routed_in.reshape(n_tokens * top_k, self.hidden_dim)
    ones = torch.ones(n_tokens * top_k, 1, device = hidden_states.device, dtype = router_scores.dtype)
    routed_out = self.experts(routed_in, top_k_index.reshape(-1, 1), ones)
    routed_out = routed_out.reshape(n_tokens, top_k, self.hidden_dim).sum(dim = 1)
    out = self.shared_expert(hidden_states)
    # Cast first: a promoting add turns the residual float32 under autocast and mixes dtypes in the next GEMM.
    out = out + routed_out.to(out.dtype)
    return out, router_logits


def patch_llama4_moe():
    patch_param_wrapper_for_moe()

    try:
        from transformers.models.llama4.modeling_llama4 import Llama4TextExperts, Llama4TextMoe
        # Before transformers 4.54 the router returns logits only: those versions keep their own forward.
        from transformers.models.llama4.modeling_llama4 import Llama4Router
    except Exception:
        return

    if getattr(Llama4TextExperts, "_unsloth_already_patched", False):
        return

    # The MoE forward feeds the experts routing indices: install both or neither.
    original_experts_forward = Llama4TextExperts.__dict__.get("forward")
    original_moe_forward = Llama4TextMoe.__dict__.get("forward")
    ok = patch_function(Llama4TextMoe, "forward", Llama4TextMoe_forward)
    ok = ok and patch_function(Llama4TextExperts, "forward", get_forward_moe_backend(), force = True)
    if not ok:
        if original_moe_forward is not None:
            Llama4TextMoe.forward = original_moe_forward
        if original_experts_forward is not None:
            Llama4TextExperts.forward = original_experts_forward
    else:
        # Separated LoRA reads dims from the module; the shared extractor misreads (E, in, out).
        Llama4TextExperts._unsloth_lora_extractor_fn = staticmethod(_llama4_moe_lora_extractor)
        Llama4TextExperts.is_transposed = True
        Llama4TextExperts.is_concatenated = True
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

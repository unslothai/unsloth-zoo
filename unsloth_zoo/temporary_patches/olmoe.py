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
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from .common import (
    TEMPORARY_PATCHES,
    UNSLOTH_ENABLE_LOGGING,
)
from .utils import (
    patch_function,
    logger,
)

# Grouped GEMM kernel integration for MoE training acceleration.
from .moe_utils import (
    patch_param_wrapper_for_moe,
    get_forward_moe_backend,
    extract_moe_lora_weights_for_grouped_mm,
    register_weight_preprocessor,
    get_weight_preprocessor,
)


def _olmoe_weight_preprocessor(weight, proj_type, hidden_dim):
    """OLMoE expert weights are stored in the transformers F.linear layout
    (gate_up (E, 2I, H), down (E, H, I)); grouped_mm wants (E, in, out), so both
    projections always transpose. Registered explicitly because OLMoE's shipping
    shape makes gate_up square (2*1024 == 2048 == hidden_size on OLMoE-1B-7B), where
    shape inspection cannot tell the two layouts apart (unslothai/unsloth-zoo#849) —
    the registry bypasses the shape guess entirely."""
    return weight.transpose(-2, -1)


def _make_olmoe_moe_lora_extractor():
    """LoRA extractor for the fused OlmoeExperts. Same 3D layout as Qwen3MoeExperts
    (gate_up_proj (E, 2*I, H), down_proj (E, H, I)); io-dims read off hidden_dim / intermediate_dim."""
    def _get_olmoe_moe_lora_dims(wrapper):
        if wrapper is None or not hasattr(wrapper, "get_base_layer"):
            return None, None

        base = wrapper.get_base_layer()
        if base is None:
            return None, None
        param_name = getattr(wrapper, "parameter_name", None)
        if param_name == "gate_up_proj":
            input_dim = getattr(base, "hidden_dim", None)
            output_dim = getattr(base, "intermediate_dim", None)
            return input_dim, None if output_dim is None else 2 * output_dim
        if param_name == "down_proj":
            return getattr(base, "intermediate_dim", None), getattr(base, "hidden_dim", None)

        return None, None

    def _olmoe_moe_lora_extractor(wrapper, weight_A, weight_B, scaling, num_experts):
        input_dim, output_dim = _get_olmoe_moe_lora_dims(wrapper)
        return extract_moe_lora_weights_for_grouped_mm(
            wrapper,
            weight_A,
            weight_B,
            scaling,
            num_experts,
            input_dim=input_dim,
            output_dim=output_dim,
            model_name="OLMoE",
            enable_logging=UNSLOTH_ENABLE_LOGGING,
            logger_obj=logger,
        )

    return _olmoe_moe_lora_extractor


def patch_olmoe_moe():
    """Patch OLMoE (olmoe) fused experts for bnb-4bit QLoRA + Split LoRA via grouped GEMM.

    The generic MoE bnb-4bit quantizer already matches OlmoeExperts (any module with
    fused gate_up_proj/down_proj nn.Parameters), but with no per-arch forward patch the
    native OlmoeExperts.forward receives the packed 2D uint8 Params4bit storage and
    crashes on the first forward (unslothai/unsloth-zoo#850): the model loads with
    load_in_4bit=True, VRAM drops, then IndexError/'got Byte' at first use. Routing the
    forward through Unsloth's MoE backend (dequantize + grouped_mm) fixes it, as for
    qwen3_moe / glm4_moe / lfm2_moe / etc. The OlmoeSparseMoeBlock keeps its native
    routing — its gate returns (router_logits, top_k_weights, top_k_index) and it calls
    self.experts(hidden_states, top_k_index, top_k_weights), which matches the
    dispatcher signature — so only the experts forward is replaced.
    """
    # Separated LoRA on the fused experts params. Idempotent (qwen3_moe installs it too).
    patch_param_wrapper_for_moe()

    # Transformers < 5 has no OlmoeExperts (ModuleList experts, handled by
    # moe_grouped_modulelist) -> strict no-op there.
    try:
        from transformers.models.olmoe.modeling_olmoe import OlmoeExperts
    except ImportError:
        return

    if getattr(OlmoeExperts, "_unsloth_already_patched", False):
        return

    # Deterministic grouped_mm layout for the square-dims arch (#849): model_type on
    # the class routes preprocess_weight through the registry instead of shape
    # inference, which is ambiguous for OLMoE's square gate_up.
    OlmoeExperts._unsloth_model_type = "olmoe"
    if get_weight_preprocessor("olmoe") is None:
        register_weight_preprocessor("olmoe", _olmoe_weight_preprocessor)

    # get_forward_moe_backend() prefers the unsloth_compiled_cache copy of moe_utils —
    # a distinct module object with its own (empty) _WEIGHT_PREPROCESSORS, where the
    # package registration above is invisible and square gate_up falls back to shape
    # inference (#849). Register through the executing namespace's own API as well.
    forward_backend = get_forward_moe_backend()
    backend_ns = getattr(forward_backend, "__globals__", None) or {}
    ns_get = backend_ns.get("get_weight_preprocessor")
    ns_register = backend_ns.get("register_weight_preprocessor")
    if (
        callable(ns_get)
        and callable(ns_register)
        and ns_get is not get_weight_preprocessor
        and ns_get("olmoe") is None
    ):
        ns_register("olmoe", _olmoe_weight_preprocessor)

    _olmoe_lora_extractor = _make_olmoe_moe_lora_extractor()
    OlmoeExperts._unsloth_lora_extractor_fn = staticmethod(_olmoe_lora_extractor)

    # Pass the function object directly (no closure): patch_function serializes the source into
    # the compiled cache, so a closure var would be a NameError there. Mirrors qwen3_moe.py.
    patch_function(OlmoeExperts, "forward", forward_backend)
    OlmoeExperts._unsloth_already_patched = True

    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched OLMoE experts for bnb-4bit + Split LoRA support.")


TEMPORARY_PATCHES.append(patch_olmoe_moe)

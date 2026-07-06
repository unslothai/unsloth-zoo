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
"""bitsandbytes 4-bit support for transformers v5 MoE expert parameters.

transformers' bnb integration only quantizes nn.Linear. In v5 the MoE expert
weights are bare 3-D nn.Parameters (gate_up_proj / down_proj on the experts
module), so the integration skips them. These patches teach it to recognize
and quantize those parameters and route the experts forward through the
standard backend after on-the-fly dequantization.
"""

from typing import Optional, List, Union

import torch
import torch.nn as nn

from .common import (
    TEMPORARY_PATCHES,
    UNSLOTH_ENABLE_LOGGING,
    is_transformers_v5_moe_quantization_available,
    logger,
)
from .utils import patch_function, raise_error


try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Params4bit
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    Params4bit = None


__all__ = [
    "patch_bnb4bit_quantize_convert",
    "patch_bnb4bit_quantizer_param_needs_quantization",
    "patch_bnb4bit_quantizer_process_model",
    "patch_bnb4bit_quantizer_weight_conversions",
    "patch_bnb4bit_model_conversion_mapping",
    "patch_bnb4bit_dequantize_plain_params",
    "patch_transformers_weight_converter_kwargs",
    "replace_expert_params_with_bnb_params",
    "forward_moe_backend_bnb4bit",
    "_moe_uses_bnb4bit_expert_weights",
]


# ============================================================================
# Detection
# ============================================================================

def _is_bnb4bit_param(param) -> bool:
    """True iff `param` is a Params4bit with a populated quant_state."""
    return (
        HAS_BNB
        and isinstance(param, Params4bit)
        and getattr(param, "quant_state", None) is not None
    )


def _moe_uses_bnb4bit_expert_weights(self) -> bool:
    """True iff this experts module's gate_up_proj / down_proj are bnb 4-bit."""
    if not HAS_BNB:
        return False
    return _is_bnb4bit_param(getattr(self, "gate_up_proj", None)) or _is_bnb4bit_param(
        getattr(self, "down_proj", None)
    )


def _is_expert_module(module: nn.Module) -> bool:
    """True iff gate_up_proj and down_proj are both nn.Parameter (v5 MoE experts)."""
    return (
        hasattr(module, "gate_up_proj")
        and hasattr(module, "down_proj")
        and isinstance(module.gate_up_proj, nn.Parameter)
        and isinstance(module.down_proj, nn.Parameter)
    )


# ============================================================================
# Dequantization
# ============================================================================

def _dequantize_bnb4bit_expert_weights(weight, target_dtype: torch.dtype):
    """Dequantize a packed Params4bit MoE expert to its logical 3D shape at
    `target_dtype`. Returns None if `weight` isn't a quantized Params4bit.
    """
    if not _is_bnb4bit_param(weight):
        return None
    dequant = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
    original_shape = getattr(weight, "_original_shape", None)
    if original_shape is not None and tuple(dequant.shape) != tuple(original_shape):
        dequant = dequant.reshape(original_shape)
    return dequant.to(target_dtype)


# ============================================================================
# Forward dispatcher
# ============================================================================

_LOGGED_BACKENDS = set()


def _log_moe_bnb4bit_backend_once(experts_module, message: str):
    key = (id(type(experts_module)), message)
    if key in _LOGGED_BACKENDS:
        return
    _LOGGED_BACKENDS.add(key)
    if UNSLOTH_ENABLE_LOGGING:
        logger.info(message)


def forward_moe_backend_bnb4bit(self, hidden_states, top_k_index, top_k_weights):
    """bnb 4-bit MoE forward: dequantize experts to the input dtype, then
    dispatch to the standard MoE backend (grouped_mm / triton / native).

    Mirrors `forward_moe_backend_fp8`. Base weights stay in 4-bit Params4bit
    storage; dequantized copies are temporary.
    """
    from .moe_utils import (
        select_moe_backend,
        forward_native_grouped_mm,
        forward_triton_grouped_gemm,
        forward_native_moe_loop,
        swap_moe_weights_for_call,
        _moe_recompute_default,
    )

    target_dtype = hidden_states.dtype
    if not target_dtype.is_floating_point:
        target_dtype = torch.bfloat16

    # Defer dequant into forward_native_grouped_mm's providers instead of
    # pre-dequantizing the full bf16 stack up front. The 4-bit base prefers recompute
    # by default (prefer_memory=True) so that, even inside a gradient-checkpoint
    # recompute pass, we never materialize and pin the full bf16 expert dequant the
    # 4-bit storage exists to avoid; UNSLOTH_MOE_RECOMPUTE=0 still forces the pin for
    # memory-rich runs. This keeps the packed Params4bit and rebuilds the dense stack
    # on demand; the pre-dequantize path below is only taken when the policy says pin,
    # so we never pre-dequantize into a dense stack and then re-hold it for a backward
    # recompute. Matches the per-source decision in forward_native_grouped_mm.
    if (
        select_moe_backend() == "grouped_mm"
        and _is_bnb4bit_param(self.gate_up_proj)
        and _is_bnb4bit_param(self.down_proj)
        and _moe_recompute_default(prefer_memory = True)
    ):
        _log_moe_bnb4bit_backend_once(self, "Unsloth: MoE bnb4bit grouped_mm with backward-recompute.")
        return forward_native_grouped_mm(self, hidden_states.to(target_dtype), top_k_index, top_k_weights)

    gate_up_weight = _dequantize_bnb4bit_expert_weights(self.gate_up_proj, target_dtype)
    down_weight = _dequantize_bnb4bit_expert_weights(self.down_proj, target_dtype)
    if gate_up_weight is None or down_weight is None:
        return None  # not bnb4bit (mixed state) — fall through to caller

    backend = select_moe_backend()
    if backend == "grouped_mm":
        _log_moe_bnb4bit_backend_once(self, "Unsloth: MoE bnb4bit using dequantize-plus-grouped_mm.")
        forward_fn = forward_native_grouped_mm
    elif backend == "unsloth_triton":
        _log_moe_bnb4bit_backend_once(self, "Unsloth: MoE bnb4bit using dequantize-plus-Triton grouped GEMM.")
        forward_fn = forward_triton_grouped_gemm
    else:
        _log_moe_bnb4bit_backend_once(self, "Unsloth: MoE bnb4bit using dequantize-plus-native_torch loop.")
        forward_fn = forward_native_moe_loop

    return swap_moe_weights_for_call(
        self,
        gate_up_weight,
        down_weight,
        forward_fn,
        hidden_states.to(target_dtype),
        top_k_index,
        top_k_weights,
    )


# ============================================================================
# transformers integration patches
# ============================================================================

def replace_expert_params_with_bnb_params(
    model: nn.Module,
    modules_to_not_convert: Optional[List[str]] = None,
    quantization_config = None,
    pre_quantized: bool = False,
) -> nn.Module:
    """Replace MoE gate_up_proj / down_proj nn.Parameters with Params4bit."""

    try:
        from transformers.quantizers.quantizers_utils import should_convert_module
    except Exception as e:
        return raise_error("transformers.quantizers.quantizers_utils.should_convert_module", e)

    has_been_replaced = False

    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue

        if not _is_expert_module(module):
            continue

        gate_up_proj = module.gate_up_proj
        down_proj = module.down_proj

        if isinstance(gate_up_proj, Params4bit) or isinstance(down_proj, Params4bit):
            continue
        with torch.device("meta"):
            placeholder_gate_up = Params4bit(
                torch.zeros(gate_up_proj.shape),
                requires_grad=False,
                compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                quant_type=quantization_config.bnb_4bit_quant_type,
                quant_storage=quantization_config.bnb_4bit_quant_storage,
            )

            placeholder_down = Params4bit(
                torch.zeros(down_proj.shape),
                requires_grad=False,
                compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                quant_type=quantization_config.bnb_4bit_quant_type,
                quant_storage=quantization_config.bnb_4bit_quant_storage,
            )

        module.gate_up_proj = placeholder_gate_up
        module.down_proj = placeholder_down
        has_been_replaced = True

        if UNSLOTH_ENABLE_LOGGING:
            logger.info(f"Unsloth: Prepared {module_name}'s gate_up_proj & down_proj for BNB 4-bit quantization (shapes: {gate_up_proj.shape}, {down_proj.shape})")

    if not has_been_replaced and UNSLOTH_ENABLE_LOGGING:
        logger.info(
            f"Unsloth: No MoE expert parameters were found to be replaced for "
            f"{getattr(model, 'name_or_path', type(model).__name__)} (expected for non-MoE)"
        )

    return model


def patch_bnb4bit_quantize_convert():
    """Convert nn.Parameter experts to Params4bit during weight loading,
    preserving original shape for PEFT LoRA compatibility.
    """

    try:
        from transformers.integrations.bitsandbytes import Bnb4bitQuantize
    except Exception as e:
        return raise_error("transformers.integrations.bitsandbytes.Bnb4bitQuantize", e)

    if hasattr(Bnb4bitQuantize.convert, "_unsloth_moe_patched"):
        return

    original_convert = Bnb4bitQuantize.convert

    def patched_convert(
        self,
        input_dict: dict[str, Union[list[torch.Tensor], torch.Tensor]],
        full_layer_name: str | None = None,
        model: torch.nn.Module | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        value = list(input_dict.values())[0]
        if isinstance(value, (list, tuple)):
            value = value[0]

        try:
            from transformers.quantizers.quantizers_utils import get_module_from_name
            module, _ = get_module_from_name(model, full_layer_name)

            if _is_expert_module(module):
                old_value = model.get_parameter_or_buffer(full_layer_name)
                old_dict = {k: v for k, v in old_value.__dict__.items()}
                new_value = Params4bit(value, requires_grad=False, **old_dict).to(value.device)
                # _original_shape is needed by PEFT LoRA to recover the logical 3D shape.
                new_value._original_shape = value.shape
                module._is_hf_initialized = True
                return {full_layer_name: new_value}

        except (KeyError, AttributeError) as e:
            if UNSLOTH_ENABLE_LOGGING:
                logger.info(f"Unsloth: expert convert fall-through for {full_layer_name}: {e}")

        return original_convert(self, input_dict, full_layer_name=full_layer_name, model=model, **kwargs)

    patched_convert._unsloth_moe_patched = True
    patch_function(Bnb4bitQuantize, "convert", patched_convert, match_level = "relaxed")

    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched Bnb4bitQuantize.convert for MoE expert parameter support")
pass


def patch_bnb4bit_quantizer_param_needs_quantization():
    """Recognize MoE expert modules of Params4bit type as needing quantization."""

    try:
        from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
        from transformers.quantizers.quantizers_utils import get_module_from_name
    except Exception as e:
        return raise_error("transformers.quantizers.quantizer_bnb_4bit.Bnb4BitHfQuantizer", e)

    original_param_needs_quantization = getattr(Bnb4BitHfQuantizer, "param_needs_quantization", None)
    if original_param_needs_quantization is None:
        return

    if getattr(original_param_needs_quantization, "_unsloth_moe_patched", False):
        return

    def patched_param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        if original_param_needs_quantization(self, model, param_name, **kwargs):
            return True

        try:
            module, name = get_module_from_name(model, param_name)
            if name in ("gate_up_proj", "down_proj"):
                param = getattr(module, name, None)
                if isinstance(param, Params4bit) and not getattr(param, "bnb_quantized", False):
                    return True
        except (KeyError, AttributeError) as e:
            if UNSLOTH_ENABLE_LOGGING:
                logger.info(f"Unsloth: param_needs_quantization fall-through for {param_name}: {e}")

        return False

    patched_param_needs_quantization._unsloth_moe_patched = True
    patch_function(Bnb4BitHfQuantizer, "param_needs_quantization", patched_param_needs_quantization)

    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched Bnb4BitHfQuantizer.param_needs_quantization for MoE expert parameters")
pass


def patch_bnb4bit_quantizer_process_model():
    try:
        from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
    except Exception as e:
        return raise_error("transformers.quantizers.quantizer_bnb_4bit.Bnb4BitHfQuantizer", e)

    if hasattr(Bnb4BitHfQuantizer._process_model_before_weight_loading, "_unsloth_moe_patched"):
        return

    original_process_model_before_weight_loading = Bnb4BitHfQuantizer._process_model_before_weight_loading

    def patched_process_model_before_weight_loading(self, model, device_map, **kwargs):
        original_process_model_before_weight_loading(self, model, device_map, **kwargs)

        model = replace_expert_params_with_bnb_params(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
        )
        return model

    patched_process_model_before_weight_loading._unsloth_moe_patched = True
    patch_function(Bnb4BitHfQuantizer, "_process_model_before_weight_loading", patched_process_model_before_weight_loading, match_level = "relaxed")
pass


def patch_transformers_weight_converter_kwargs():
    """Drop kwargs PEFT 0.19 passes to WeightConverter (distributed_operation,
    quantization_operation, ...) that transformers 5.6.2's __init__ rejects,
    which otherwise raises TypeError in PeftModel.from_pretrained for MoE 4-bit.
    """
    try:
        from transformers.core_model_loading import WeightConverter
    except ImportError:
        return

    if getattr(WeightConverter.__init__, "_unsloth_kwargs_patched", False):
        return

    _original_init = WeightConverter.__init__

    import inspect
    try:
        original_params = set(inspect.signature(_original_init).parameters)
    except (TypeError, ValueError):
        original_params = {"self", "source_patterns", "target_patterns", "operations"}

    def _patched_init(self, *args, **kwargs):
        accepted = {k: v for k, v in kwargs.items() if k in original_params}
        return _original_init(self, *args, **accepted)

    _patched_init._unsloth_kwargs_patched = True
    WeightConverter.__init__ = _patched_init

    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched transformers WeightConverter.__init__ to ignore unknown kwargs (peft 0.19 forward-compat)")
pass


# ============================================================================
# PEFT LoRA support for stacked-MoE bnb 4-bit experts.
# Expose a Params4bit's logical 3-D shape to ParamWrapper and merge/unmerge via a
# dequant -> add -> requant cycle so merge_and_unload() works on packed 4-bit storage.
# ============================================================================

class _ParamShapeProxy:
    """Expose 4bit MoE param attributes correctly for PEFT LoRA; delegate the rest."""

    def __init__(self, param, shape):
        self._param = param
        self._shape = shape
        self._ndim = len(shape)

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def dtype(self):
        # Surface the pre-quant compute dtype so PEFT's `tensor.to(param.dtype)`
        # casts don't truncate bf16 LoRA deltas to the uint8 packed storage dtype.
        quant_state = getattr(self._param, "quant_state", None)
        qs_dtype = getattr(quant_state, "dtype", None) if quant_state is not None else None
        if qs_dtype is not None and (qs_dtype.is_floating_point or qs_dtype.is_complex):
            return qs_dtype
        return self._param.dtype

    def __getattr__(self, name):
        return getattr(self._param, name)


def patch_peft_param_wrapper_4bit_expert_shape():
    """Patch ParamWrapper.get_param() to return a proxy exposing
    .shape = _original_shape for 4bit MoE params (param.shape is wrong for them).
    """
    try:
        from peft.tuners.lora.layer import ParamWrapper
        from peft.utils.integrations import get_bnb_param_type
    except (ImportError, AttributeError):
        return

    if getattr(ParamWrapper.get_param, "_unsloth_4bit_expert_patched", False):
        return

    _original_get_param = ParamWrapper.get_param

    def _patched_get_param(self):
        param = _original_get_param(self)
        if get_bnb_param_type(param) == "4bit":
            shape = getattr(param, "_original_shape", None)
            if shape is not None and len(shape) == 3:
                # Don't touch in_features/out_features: PEFT 0.19's update_layer
                # swaps them for 3D params and calls get_param() again afterwards.
                self.num_experts = shape[0]
                return _ParamShapeProxy(param, shape)
            raise ValueError(
                "unsloth: ParamWrapper.get_param() encountered a 4-bit Params4bit "
                f"without a 3D _original_shape attribute (param shape={tuple(param.shape)}). "
                "This usually means the MoE quantizer patch did not run during model load, "
                "or this parameter is not a stacked MoE expert and should not be in "
                "LoraConfig.target_parameters."
            )
        return param

    _patched_get_param._unsloth_4bit_expert_patched = True
    patch_function(ParamWrapper, "get_param", _patched_get_param)
pass


def patch_peft_param_wrapper_merge_4bit():
    """Override ParamWrapper.merge/unmerge with a dequantize -> add ->
    re-quantize cycle for Params4bit MoE experts with a 3D _original_shape.

    PEFT's in-place `param.data += delta_weight` fails here: `param.data` is
    packed (N_packed, 1) uint8 storage while `delta_weight` is the logical 3D.
    """
    try:
        from peft.tuners.lora.layer import ParamWrapper, check_adapters_to_merge
    except (ImportError, AttributeError):
        return
    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import Params4bit
    except ImportError:
        return

    if getattr(ParamWrapper.merge, "_unsloth_4bit_moe_patched", False):
        return

    _original_merge = ParamWrapper.merge
    _original_unmerge = ParamWrapper.unmerge

    def _is_4bit_moe_param(param):
        return isinstance(param, Params4bit) and getattr(param, "_original_shape", None) is not None

    def _dequant_param_for_merge(param, parameter_name):
        """Dequantize a Params4bit MoE expert to logical 3D at the compute
        dtype; raises if `quant_state` is unpopulated. Shared by merge/unmerge."""
        quant_state = getattr(param, "quant_state", None)
        if quant_state is None:
            raise RuntimeError(
                "unsloth: ParamWrapper saw a Params4bit MoE expert with "
                f"quant_state=None on {parameter_name}. The MoE quantizer "
                "patch likely did not finish before merge_and_unload()."
            )
        compute_dtype = getattr(quant_state, "dtype", None) or torch.bfloat16
        dequant = _dequantize_bnb4bit_expert_weights(param, compute_dtype)
        return dequant, compute_dtype

    def _requantize_like(reference, new_data, original_shape):
        kwargs = dict(
            requires_grad=False,
            blocksize=getattr(reference, "blocksize", 64),
            compress_statistics=getattr(reference, "compress_statistics", True),
            quant_type=getattr(reference, "quant_type", "nf4"),
            quant_storage=getattr(reference, "quant_storage", torch.uint8),
        )
        device = reference.device
        if device.type == "meta":
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        new_param = Params4bit(new_data.detach().to("cpu").contiguous(), **kwargs).to(device)
        new_param._original_shape = torch.Size(original_shape)
        return new_param

    def _patched_merge(self, safe_merge=False, adapter_names=None):
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        base_layer = self.get_base_layer()
        param = getattr(base_layer, self.parameter_name)

        if not _is_4bit_moe_param(param):
            return _original_merge(self, safe_merge=safe_merge, adapter_names=adapter_names)

        for active_adapter in adapter_names:
            if active_adapter not in self.lora_A.keys():
                continue
            param = getattr(base_layer, self.parameter_name)
            original_shape = torch.Size(getattr(param, "_original_shape"))
            dequant, compute_dtype = _dequant_param_for_merge(param, self.parameter_name)
            delta_weight = self.get_delta_weight(active_adapter).to(
                device=dequant.device, dtype=compute_dtype
            )
            if delta_weight.shape != dequant.shape:
                raise RuntimeError(
                    f"unsloth: delta_weight shape {tuple(delta_weight.shape)} does not "
                    f"match dequantized expert shape {tuple(dequant.shape)} for "
                    f"{self.parameter_name}/adapter={active_adapter}."
                )
            merged = dequant + delta_weight

            if safe_merge and not torch.isfinite(merged).all():
                raise ValueError(
                    f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                )

            new_param = _requantize_like(param, merged, original_shape)
            setattr(base_layer, self.parameter_name, new_param)
            self.merged_adapters.append(active_adapter)

    def _patched_unmerge(self):
        if not self.merged:
            import warnings
            warnings.warn("Already unmerged. Nothing to do.")
            return

        base_layer = self.get_base_layer()
        param = getattr(base_layer, self.parameter_name)
        if not _is_4bit_moe_param(param):
            return _original_unmerge(self)

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self.lora_A.keys():
                continue
            param = getattr(base_layer, self.parameter_name)
            original_shape = torch.Size(getattr(param, "_original_shape"))
            dequant, compute_dtype = _dequant_param_for_merge(param, self.parameter_name)
            delta_weight = self.get_delta_weight(active_adapter).to(
                device=dequant.device, dtype=compute_dtype
            )
            unmerged = dequant - delta_weight
            new_param = _requantize_like(param, unmerged, original_shape)
            setattr(base_layer, self.parameter_name, new_param)

    _patched_merge._unsloth_4bit_moe_patched = True
    _patched_unmerge._unsloth_4bit_moe_patched = True
    patch_function(ParamWrapper, "merge", _patched_merge, match_level="relaxed")
    patch_function(ParamWrapper, "unmerge", _patched_unmerge, match_level="relaxed")
pass


# ============================================================================
# Prequantized checkpoint loading (transformers v5)
# ============================================================================

def _bnb4bit_expert_weight_conversions(hf_quantizer):
    """WeightConverters for prequantized bnb 4-bit MoE expert params.

    The stock quantizer registers Bnb4bitDeserialize only for params named
    `weight`, so fused expert params (gate_up_proj/down_proj) load as raw
    packed uint8 and their aux keys are dropped. Register equivalents here.
    """
    from transformers.core_model_loading import WeightConverter
    from transformers.integrations.bitsandbytes import Bnb4bitDeserialize
    from transformers.quantizers.quantizers_utils import get_module_from_name

    class _ExpertDeserialize(Bnb4bitDeserialize):
        def __init__(self, hf_quantizer, param_name):
            super().__init__(hf_quantizer)
            self.param_name = param_name

        def convert(self, input_dict, model=None, full_layer_name=None, **kwargs):
            if len(input_dict) == 1:
                return input_dict  # no quant stats collected -> not prequantized
            input_dict = dict(input_dict)  # avoid mutating the caller's dict
            for key, value in list(input_dict.items()):
                if isinstance(value, list):
                    input_dict[key] = value[0]
            weight = input_dict.pop(self.param_name)
            module, _ = get_module_from_name(model, full_layer_name)
            new_value = Params4bit.from_prequantized(
                data=weight,
                quantized_stats=input_dict,
                requires_grad=False,
                device=weight.device,
                module=module,
            )
            # _original_shape is needed by PEFT LoRA to recover the logical 3D shape.
            if getattr(new_value, "quant_state", None) is not None:
                new_value._original_shape = new_value.quant_state.shape
            module._is_hf_initialized = True
            return {self.param_name: new_value}

    converters = []
    for pname in ("gate_up_proj", "down_proj"):
        converters.append(
            WeightConverter(
                source_patterns=[
                    f"{pname}.nested_absmax",
                    f"{pname}.nested_quant_map",
                    f"{pname}.quant_map",
                    f"{pname}.absmax",
                    f"{pname}.quant_state.bitsandbytes__nf4",
                    f"{pname}.quant_state.bitsandbytes__fp4",
                    pname,
                ],
                target_patterns=pname,
                operations=[_ExpertDeserialize(hf_quantizer, pname)],
            )
        )
    return converters


def patch_bnb4bit_quantizer_weight_conversions():
    """Extend get_weight_conversions so prequantized MoE expert params deserialize."""
    try:
        from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
    except Exception as e:
        return raise_error("transformers.quantizers.quantizer_bnb_4bit.Bnb4BitHfQuantizer", e)

    original_get_weight_conversions = getattr(Bnb4BitHfQuantizer, "get_weight_conversions", None)
    if original_get_weight_conversions is None:
        return
    if getattr(original_get_weight_conversions, "_unsloth_moe_patched", False):
        return

    def patched_get_weight_conversions(self):
        conversions = original_get_weight_conversions(self)
        if self.pre_quantized:
            try:
                conversions = list(conversions) + _bnb4bit_expert_weight_conversions(self)
            except Exception as e:
                if UNSLOTH_ENABLE_LOGGING:
                    logger.info(f"Unsloth: expert weight conversions unavailable: {e}")
        return conversions

    patched_get_weight_conversions._unsloth_moe_patched = True
    patch_function(Bnb4BitHfQuantizer, "get_weight_conversions", patched_get_weight_conversions)
pass


_AUX_SUFFIXES = (
    ".nested_absmax",
    ".nested_quant_map",
    ".quant_map",
    ".absmax",
    ".quant_state.bitsandbytes__nf4",
    ".quant_state.bitsandbytes__fp4",
)


def _quantstate_absmax_fp32(qs):
    """Materialize a QuantState's absmax as flat fp32 (denesting double-quant)."""
    if getattr(qs, "nested", False):
        absmax = bnb.functional.dequantize_blockwise(qs.absmax, qs.state2)
        absmax = absmax + qs.offset
        return absmax.float()
    return qs.absmax.float()


def _bnb4bit_per_expert_conversions(model_conversions, hf_quantizer):
    """Quantized twins of the model's per-expert MoE merge converters.

    Old checkpoints store one quantized tensor per expert per projection; the
    model's MergeModulelist converters byte-concat the packed uint8 as if bf16
    and drop the absmax/quant_map aux keys. Each twin also collects the aux
    keys and rebuilds one stacked Params4bit (byte concat is exact when each
    segment fills whole quant blocks; otherwise it repacks from dequantized
    values to keep block boundaries aligned). Twins must be PREPENDED to
    win the first-match scan; bare-weight patterns are `$`-anchored so they do
    not shadow the model converter in pattern_to_converter.
    """
    from transformers.core_model_loading import WeightConverter, ConversionOps
    from transformers.quantizers.quantizers_utils import get_module_from_name
    from bitsandbytes.functional import QuantState
    from copy import deepcopy

    class _PerExpertStackDeserialize(ConversionOps):
        def __init__(self, base_sources, anchored_sources, original_ops):
            self.base_sources = base_sources          # e.g. ["mlp.experts.*.gate_proj.weight", ...]
            self.anchored_sources = anchored_sources  # same, "$"-anchored (collection keys)
            self.original_ops = original_ops          # model's ops, for unquantized fallback

        def convert(self, input_dict, model=None, full_layer_name=None,
                    target_patterns=None, missing_keys=None, config=None, **kwargs):
            has_aux = any(
                (base + suf) in input_dict
                for base in self.base_sources
                for suf in _AUX_SUFFIXES
            )
            if not has_aux:
                # Unquantized experts in this layer (e.g. dynamic-quant skip layers):
                # replicate the model's own merge exactly.
                out = {
                    orig: input_dict[anch]
                    for orig, anch in zip(self.base_sources, self.anchored_sources)
                    if anch in input_dict
                }
                for op in self.original_ops:
                    out = op.convert(
                        out,
                        source_patterns=self.base_sources,
                        target_patterns=target_patterns,
                        full_layer_name=full_layer_name,
                        model=model,
                        config=config,
                        missing_keys=missing_keys,
                    )
                return out

            input_dict = dict(input_dict)  # avoid mutating the caller's dict
            for key, value in list(input_dict.items()):
                if not isinstance(value, list):
                    input_dict[key] = [value]

            # Only keep sources actually present; guard against missing/mismatched keys.
            present_sources = [
                (base, anch)
                for base, anch in zip(self.base_sources, self.anchored_sources)
                if anch in input_dict
            ]
            if not present_sources:
                raise ValueError(
                    f"Unsloth: no expert weights found in input_dict for {full_layer_name}"
                )

            counts = {len(input_dict[anch]) for _, anch in present_sources}
            if len(counts) != 1:
                raise ValueError(
                    f"Unsloth: inconsistent per-expert tensor counts {counts} for {full_layer_name}"
                )
            num_experts = counts.pop()

            device = input_dict[present_sources[0][1]][0].device
            first_qs = None
            src_shapes = []
            per_src_absmax = []  # [src][expert] fp32 absmax
            per_src_qs = []      # [src][expert] QuantState (for the misaligned fallback)
            for base, anch in present_sources:
                absmax_list, qs_list = [], []
                for e in range(num_experts):
                    qd = {}
                    for suf in _AUX_SUFFIXES:
                        vals = input_dict.get(base + suf)
                        if vals is not None and e < len(vals):
                            qd["weight" + suf] = vals[e]
                    qs = QuantState.from_dict(qs_dict=qd, device=device)
                    if first_qs is None:
                        first_qs = qs
                    if e == 0:
                        src_shapes.append(tuple(qs.shape))
                    absmax_list.append(_quantstate_absmax_fp32(qs))
                    qs_list.append(qs)
                per_src_absmax.append(absmax_list)
                per_src_qs.append(qs_list)

            out_dim = sum(s[0] for s in src_shapes)
            in_dim = src_shapes[0][1]
            if any(s[1] != in_dim for s in src_shapes):
                raise ValueError(f"Unsloth: mismatched expert in_dims {src_shapes} for {full_layer_name}")

            blocksize = first_qs.blocksize
            if any((s[0] * s[1]) % blocksize != 0 for s in src_shapes):
                # A segment ends mid-block, so raw byte/absmax concatenation would
                # scale everything after it with the wrong absmax. Repack from
                # dequantized values instead (exact block boundaries, tiny requant noise).
                full = torch.stack([
                    torch.cat([
                        bnb.functional.dequantize_4bit(
                            input_dict[anch][e].reshape(-1, 1), per_src_qs[i][e]
                        )
                        for i, (_, anch) in enumerate(present_sources)
                    ], dim=0)
                    for e in range(num_experts)
                ])
                data, quant_state = bnb.functional.quantize_4bit(
                    full.to(device, first_qs.dtype).contiguous(),
                    blocksize=blocksize,
                    quant_type=first_qs.quant_type,
                    compress_statistics=False,
                )
                quant_state.shape = torch.Size((num_experts, out_dim, in_dim))
            else:
                packed_rows, absmax_rows = [], []
                for e in range(num_experts):
                    packed_rows.append(torch.cat(
                        [input_dict[anch][e].reshape(-1) for _, anch in present_sources]
                    ))
                    absmax_rows.append(torch.cat([per_src_absmax[i][e] for i in range(len(present_sources))]))
                data = torch.stack(packed_rows).unsqueeze(-1)  # (E, bytes_per_expert, 1)
                absmax = torch.cat(absmax_rows)

                quant_state = QuantState(
                    absmax=absmax,
                    shape=torch.Size((num_experts, out_dim, in_dim)),
                    code=first_qs.code.to(device),
                    blocksize=blocksize,
                    quant_type=first_qs.quant_type,
                    dtype=first_qs.dtype,
                )
            new_param = torch.Tensor._make_subclass(Params4bit, data.to(device))
            new_param.requires_grad = False
            new_param.quant_state = quant_state
            new_param.blocksize = quant_state.blocksize
            new_param.compress_statistics = False
            new_param.quant_type = quant_state.quant_type
            new_param.quant_storage = data.dtype
            new_param.bnb_quantized = True
            # Logical 3D shape for PEFT LoRA's ParamWrapper.get_param (_original_shape).
            new_param._original_shape = torch.Size((num_experts, out_dim, in_dim))
            module, _ = get_module_from_name(model, full_layer_name)
            new_param.module = module
            module._is_hf_initialized = True
            return {target_patterns[0]: new_param}

    class _FusedExpertDeserialize(ConversionOps):
        """Quantized twin of a model's fused expert converter (e.g. a Transpose).

        A fused prequantized checkpoint stores packed uint8 + aux keys that the
        model's own converter would mangle (ops like Transpose cannot apply to
        packed data). With aux keys, rebuild the Params4bit directly; without
        them, replicate the model's ops so unquantized checkpoints are untouched.
        """

        def __init__(self, base_source, anchored_source, original_ops):
            self.base_source = base_source
            self.anchored_source = anchored_source
            self.original_ops = original_ops

        def convert(self, input_dict, model=None, full_layer_name=None,
                    target_patterns=None, missing_keys=None, config=None, **kwargs):
            has_aux = any((self.base_source + suf) in input_dict for suf in _AUX_SUFFIXES)
            if not has_aux:
                out = {self.base_source: input_dict[self.anchored_source]}
                for op in self.original_ops:
                    out = op.convert(
                        out,
                        source_patterns=[self.base_source],
                        target_patterns=target_patterns,
                        full_layer_name=full_layer_name,
                        model=model,
                        config=config,
                        missing_keys=missing_keys,
                    )
                return out

            input_dict = dict(input_dict)  # avoid mutating the caller's dict
            for key, value in list(input_dict.items()):
                if isinstance(value, list):
                    input_dict[key] = value[0]
            weight = input_dict.pop(self.anchored_source)
            stats = {
                (self.base_source + suf): input_dict[self.base_source + suf]
                for suf in _AUX_SUFFIXES
                if (self.base_source + suf) in input_dict
            }
            module, _ = get_module_from_name(model, full_layer_name)
            new_value = Params4bit.from_prequantized(
                data=weight,
                quantized_stats=stats,
                requires_grad=False,
                device=weight.device,
                module=module,
            )
            # Logical 3D shape for PEFT LoRA's ParamWrapper.get_param.
            if getattr(new_value, "quant_state", None) is not None:
                new_value._original_shape = new_value.quant_state.shape
                # Orientation is baked in at quantize time (converters run before
                # the quant op, so supported saves are always model layout). A
                # same-rank mismatch means an unsupported checkpoint layout that a
                # transpose op cannot fix on packed data: fail loudly here.
                try:
                    expected = tuple(model.get_parameter(full_layer_name).shape)
                except Exception:
                    expected = None
                qshape = tuple(new_value.quant_state.shape)
                if (
                    expected is not None
                    and len(expected) == len(qshape)
                    and expected != qshape
                ):
                    raise ValueError(
                        f"Unsloth: prequantized expert `{full_layer_name}` was "
                        f"quantized with shape {qshape} but the model expects "
                        f"{expected}. Requantize the checkpoint from weights in "
                        f"the model layout."
                    )
            module._is_hf_initialized = True
            return {target_patterns[0]: new_value}

    twins = []
    for conv in model_conversions:
        if not isinstance(conv, WeightConverter):
            continue
        # Single-target only: a twin returns target_patterns[0] alone, so a
        # many-to-many mapping would leave its remaining targets unloaded.
        if len(conv.target_patterns) != 1:
            continue
        target = conv.target_patterns[0]
        if not target.endswith(("gate_up_proj", "down_proj")):
            continue
        if all(s.endswith(".weight") for s in conv.source_patterns):
            base_sources = list(conv.source_patterns)
            anchored_sources = [s + "$" for s in base_sources]
            aux_sources = [b + suf for b in base_sources for suf in _AUX_SUFFIXES]
            twins.append(
                WeightConverter(
                    source_patterns=aux_sources + anchored_sources,
                    target_patterns=target,
                    operations=[
                        _PerExpertStackDeserialize(
                            base_sources, anchored_sources, deepcopy(conv.operations)
                        )
                    ],
                )
            )
        elif (
            len(conv.source_patterns) == 1
            and conv.source_patterns[0].endswith(("gate_up_proj", "down_proj"))
        ):
            # Fused passthrough converter (e.g. qwen3_vl_moe's Transpose). The
            # model converter precedes appended quantizer conversions and matching
            # stops at the first hit, so without this twin a fused prequantized
            # checkpoint feeds packed uint8 into the model's ops.
            base = conv.source_patterns[0]
            anchored = base + "$"
            aux_sources = [base + suf for suf in _AUX_SUFFIXES]
            twins.append(
                WeightConverter(
                    source_patterns=aux_sources + [anchored],
                    target_patterns=target,
                    operations=[
                        _FusedExpertDeserialize(base, anchored, deepcopy(conv.operations))
                    ],
                )
            )
    return twins


def patch_bnb4bit_model_conversion_mapping():
    """Prepend per-expert quantized-MoE converters to the model conversion mapping."""
    try:
        import transformers.conversion_mapping as conversion_mapping
        import transformers.modeling_utils as modeling_utils
        from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
    except Exception as e:
        return raise_error("transformers.conversion_mapping", e)

    original = conversion_mapping.get_model_conversion_mapping
    if getattr(original, "_unsloth_moe_patched", False):
        return

    def patched_get_model_conversion_mapping(model, key_mapping=None, hf_quantizer=None, add_legacy=True):
        conversions = original(model, key_mapping, hf_quantizer, add_legacy)
        if isinstance(hf_quantizer, Bnb4BitHfQuantizer) and hf_quantizer.pre_quantized:
            try:
                twins = _bnb4bit_per_expert_conversions(conversions, hf_quantizer)
                if twins:
                    conversions = twins + conversions
            except Exception as e:
                if UNSLOTH_ENABLE_LOGGING:
                    logger.info(f"Unsloth: per-expert bnb4bit converters unavailable: {e}")
        return conversions

    patched_get_model_conversion_mapping._unsloth_moe_patched = True
    conversion_mapping.get_model_conversion_mapping = patched_get_model_conversion_mapping
    if getattr(modeling_utils, "get_model_conversion_mapping", None) is original:
        modeling_utils.get_model_conversion_mapping = patched_get_model_conversion_mapping
pass


def patch_bnb4bit_dequantize_plain_params():
    """Dequantize prequantized Params4bit left on plain nn.Parameter slots.

    Old checkpoints quantize small non-Linear weights too (e.g. the MoE
    router), which under v5 live on bare nn.Parameter slots, so a packed
    Params4bit lands where a float weight is expected. After loading,
    dequantize any such param on a non-Linear4bit, non-experts module.
    """
    try:
        from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
    except Exception as e:
        return raise_error("transformers.quantizers.quantizer_bnb_4bit.Bnb4BitHfQuantizer", e)

    original_after_load = getattr(Bnb4BitHfQuantizer, "_process_model_after_weight_loading", None)
    if original_after_load is None:
        return
    if getattr(original_after_load, "_unsloth_plain_dequant_patched", False):
        return

    def patched_after_load(self, model, **kwargs):
        result = original_after_load(self, model, **kwargs)
        target = result if result is not None else model
        try:
            _dequantize_plain_param_slots(target)
        except Exception as e:
            if UNSLOTH_ENABLE_LOGGING:
                logger.info(f"Unsloth: plain-param dequantize pass failed: {e}")
        return result

    patched_after_load._unsloth_plain_dequant_patched = True
    patch_function(
        Bnb4BitHfQuantizer, "_process_model_after_weight_loading", patched_after_load,
        match_level="relaxed",
    )
pass


def _is_bnb_module(module: nn.Module) -> bool:
    """Any bitsandbytes module (Linear4bit, Embedding4bit, ...) or subclass:
    their forwards consume quant_state, so their weights must stay quantized."""
    return any(
        getattr(klass, "__module__", "").startswith("bitsandbytes")
        for klass in type(module).__mro__
    )


def _dequantize_plain_param_slots(model: nn.Module):
    if not HAS_BNB:
        return
    for module_name, module in model.named_modules():
        if _is_bnb_module(module) or _is_expert_module(module):
            continue
        for name, param in list(module.named_parameters(recurse=False)):
            if not isinstance(param, Params4bit):
                continue
            quant_state = getattr(param, "quant_state", None)
            if quant_state is None:
                continue
            dequant = bnb.functional.dequantize_4bit(param.data, quant_state)
            setattr(module, name, nn.Parameter(dequant, requires_grad=param.requires_grad))
            if UNSLOTH_ENABLE_LOGGING:
                logger.info(
                    f"Unsloth: dequantized non-Linear 4-bit param {module_name}.{name} "
                    f"to {tuple(dequant.shape)} {dequant.dtype}"
                )
pass


def _register_transformers_v5_moe_bnb4bit_patches():
    if not is_transformers_v5_moe_quantization_available():
        return
    TEMPORARY_PATCHES.append(patch_bnb4bit_quantize_convert)
    TEMPORARY_PATCHES.append(patch_bnb4bit_quantizer_param_needs_quantization)
    TEMPORARY_PATCHES.append(patch_bnb4bit_quantizer_process_model)
    TEMPORARY_PATCHES.append(patch_bnb4bit_quantizer_weight_conversions)
    TEMPORARY_PATCHES.append(patch_bnb4bit_model_conversion_mapping)
    TEMPORARY_PATCHES.append(patch_bnb4bit_dequantize_plain_params)
    TEMPORARY_PATCHES.append(patch_transformers_weight_converter_kwargs)
    TEMPORARY_PATCHES.append(patch_peft_param_wrapper_4bit_expert_shape)
    TEMPORARY_PATCHES.append(patch_peft_param_wrapper_merge_4bit)
pass
_register_transformers_v5_moe_bnb4bit_patches()

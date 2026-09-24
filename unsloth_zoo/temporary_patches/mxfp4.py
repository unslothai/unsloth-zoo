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

import re
from typing import Union, List, Optional, Tuple
import functools
import inspect
import torch
import torch.nn as nn
import os
import math
from importlib.metadata import version as importlib_version
from unsloth_zoo.utils import Version
from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING, logger
from .utils import patch_function, raise_error
from ..mxfp4_dequant import Mxfp4ExpertParam, is_mxfp4_expert_param, mxfp4_dequantize, mxfp4_kernel_available

transformers_version = Version(importlib_version("transformers"))

# UNSLOTH_MXFP4_NO_DEQUANTIZE=1 keeps MXFP4 quantized (needs triton_kernels);
# otherwise weights dequantize to bf16 for LoRA training.
UNSLOTH_MXFP4_NO_DEQUANTIZE = os.environ.get("UNSLOTH_MXFP4_NO_DEQUANTIZE", "0") == "1"


def _check_triton_kernels_available():
    """Check if OpenAI's triton_kernels package is available for MXFP4."""
    try:
        from triton_kernels import matmul_ogs, swiglu
        return True
    except ImportError:
        return False


_TRITON_KERNELS_AVAILABLE = None
def is_triton_kernels_available():
    """Cached check for triton_kernels availability."""
    global _TRITON_KERNELS_AVAILABLE
    if _TRITON_KERNELS_AVAILABLE is None:
        _TRITON_KERNELS_AVAILABLE = _check_triton_kernels_available()
    return _TRITON_KERNELS_AVAILABLE


def should_dequantize_mxfp4():
    """Whether MXFP4 should be dequantized to bf16 for training.

    True unless UNSLOTH_MXFP4_NO_DEQUANTIZE="1" and triton_kernels is available.
    """
    if not UNSLOTH_MXFP4_NO_DEQUANTIZE:
        return True  # Default: dequantize for compatibility

    if not is_triton_kernels_available():
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(
                "Unsloth: UNSLOTH_MXFP4_NO_DEQUANTIZE=1 but triton_kernels not available. "
                "Will dequantize MXFP4 to bf16."
            )
        return True  # triton_kernels required for native MXFP4

    return False  # Keep MXFP4 quantized


def get_mxfp4_config_for_training():
    """Return the Mxfp4Config for training (dequantize=True unless
    UNSLOTH_MXFP4_NO_DEQUANTIZE=1 and triton_kernels is available).

    Usage:
        model = AutoModelForCausalLM.from_pretrained(
            "unsloth/gpt-oss-20b",
            quantization_config=get_mxfp4_config_for_training(),
        )
    """
    try:
        from transformers import Mxfp4Config
    except ImportError:
        raise ImportError("transformers.Mxfp4Config not available. Please upgrade transformers.")

    dequantize = should_dequantize_mxfp4()

    if UNSLOTH_ENABLE_LOGGING:
        if dequantize:
            logger.info("Unsloth: MXFP4 will be dequantized to bf16 for training")
        else:
            logger.info("Unsloth: MXFP4 weights will remain quantized (triton_kernels available)")

    return Mxfp4Config(dequantize=dequantize)

_LOAD_OFFLOADS = [False]


def patch_mxfp4_offload_guard():
    try:
        import transformers.modeling_utils as modeling_utils
    except Exception:
        return
    original = getattr(modeling_utils, "_get_device_map", None)
    if original is not None and not getattr(original, "_unsloth_mxfp4_patched", False):

        @functools.wraps(original)
        def _get_device_map(*args, **kwargs):
            device_map = original(*args, **kwargs)
            values = device_map.values() if isinstance(device_map, dict) else ()
            _LOAD_OFFLOADS[0] = any(str(value) in ("cpu", "disk") for value in values)
            return device_map

        _get_device_map._unsloth_mxfp4_patched = True
        modeling_utils._get_device_map = _get_device_map

    # _get_device_map only runs given a device map, so clear the flag where every MXFP4 load passes first.
    try:
        from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer
    except Exception:
        return
    validate = Mxfp4HfQuantizer.validate_environment
    if getattr(validate, "_unsloth_mxfp4_patched", False):
        return

    @functools.wraps(validate)
    def validate_environment(self, *args, **kwargs):
        _LOAD_OFFLOADS[0] = False
        return validate(self, *args, **kwargs)

    validate_environment._unsloth_mxfp4_patched = True
    Mxfp4HfQuantizer.validate_environment = validate_environment
pass
TEMPORARY_PATCHES.append(patch_mxfp4_offload_guard)


def keep_mxfp4_experts_packed() -> bool:
    """GPT-OSS MXFP4 experts stay packed, dequantized per layer by grouped_mm. UNSLOTH_MXFP4_KEEP_PACKED=0 opts out."""
    setting = os.environ.get("UNSLOTH_MXFP4_KEEP_PACKED", "")
    if setting == "0" or transformers_version < Version("5.0.0"):
        return False
    # Offload stores the uint8 blocks without their scales, so they could not be decoded again.
    if _LOAD_OFFLOADS[0]:
        return False
    name = os.environ.get("UNSLOTH_MODEL_NAME", "").lower().replace("-", "_")
    if "gpt_oss" not in name or "_load_in_4bit_" in name:
        return False
    if os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") == "1":
        return False
    try:
        # Only grouped_mm reads experts through _get_base_weight; loop and Triton index the stack.
        from .moe_utils import select_moe_backend
        import transformers.models.gpt_oss.modeling_gpt_oss as modeling_gpt_oss
        if not getattr(modeling_gpt_oss.GptOssExperts, "_unsloth_lora_patched", False):
            return False
        if select_moe_backend() != "grouped_mm":
            return False
    except Exception:
        return False
    return True


def _dequantize_to_gpt_oss_layout(convert, blocks, scales):
    if blocks.is_cuda and blocks.dtype == torch.uint8 and blocks.dim() == 4:
        return mxfp4_dequantize(blocks, scales, transpose = True)
    return convert(blocks, scales).transpose(1, 2).contiguous()


class _Mxfp4ShapeProxy:
    def __init__(self, param):
        self._param = param
        self.shape = param._original_shape
        self.ndim = len(self.shape)

    @property
    def dtype(self):
        return self._param.mxfp4_dtype

    def __getattr__(self, name):
        return getattr(self._param, name)


def patch_peft_param_wrapper_mxfp4():
    """Merge installs a bf16 parameter (a merged weight is not MXFP4); unmerge restores the packed stack."""
    try:
        from peft.tuners.lora.layer import ParamWrapper, check_adapters_to_merge
    except Exception:
        return
    if getattr(ParamWrapper.get_param, "_unsloth_mxfp4_patched", False):
        return
    original_get_param = ParamWrapper.get_param
    original_merge = ParamWrapper.merge
    original_unmerge = ParamWrapper.unmerge

    # wraps(): the bnb 4-bit patches check the signature before layering on top.
    @functools.wraps(original_get_param)
    def get_param(self):
        param = original_get_param(self)
        if is_mxfp4_expert_param(param):
            self.num_experts = param._original_shape[0]
            return _Mxfp4ShapeProxy(param)
        return param

    @functools.wraps(original_merge)
    def merge(self, safe_merge = False, adapter_names = None):
        base_layer = self.get_base_layer()
        param = getattr(base_layer, self.parameter_name, None)
        if not is_mxfp4_expert_param(param):
            return original_merge(self, safe_merge = safe_merge, adapter_names = adapter_names)
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        merged = param.dequantize()
        applied = []
        for adapter in adapter_names:
            if adapter not in self.lora_A.keys():
                continue
            merged = merged + self.get_delta_weight(adapter).to(device = merged.device, dtype = merged.dtype)
            applied.append(adapter)
        if not applied:
            return
        if safe_merge and not torch.isfinite(merged).all():
            raise ValueError(f"NaNs detected in the merged weights. The adapters {applied} seem to be broken")
        self.__dict__.setdefault("_unsloth_mxfp4_packed", {})[self.parameter_name] = param
        setattr(base_layer, self.parameter_name, nn.Parameter(merged, requires_grad = False))
        self.merged_adapters.extend(applied)

    @functools.wraps(original_unmerge)
    def unmerge(self):
        packed = self.__dict__.get("_unsloth_mxfp4_packed", {}).pop(self.parameter_name, None)
        if packed is None:
            return original_unmerge(self)
        base_layer = self.get_base_layer()
        current = getattr(base_layer, self.parameter_name, None)
        if current is not None and current.device != packed.device:
            # The saved stack sits outside the module, so a model move since merge() left it behind.
            holder = nn.Module()
            holder.param = packed
            packed = holder.to(current.device).param
        setattr(base_layer, self.parameter_name, packed)
        self.merged_adapters.clear()

    get_param._unsloth_mxfp4_patched = True
    merge._unsloth_mxfp4_patched = True
    unmerge._unsloth_mxfp4_patched = True
    ParamWrapper.get_param = get_param
    ParamWrapper.merge = merge
    ParamWrapper.unmerge = unmerge
pass
TEMPORARY_PATCHES.append(patch_peft_param_wrapper_mxfp4)


def patch_mxfp4_quantizer_element_size():
    """caching_allocator_warmup counts experts as bf16; packed ones take 17 / 32 bytes per value."""
    try:
        from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer
    except Exception:
        return
    original = getattr(Mxfp4HfQuantizer, "param_element_size", None)
    if original is None or getattr(original, "_unsloth_mxfp4_patched", False):
        return

    @functools.wraps(original)
    def param_element_size(self, model, param_name, param):
        if (
            param_name.endswith((".gate_up_proj", ".down_proj"))
            and getattr(self, "pre_quantized", False)
            and getattr(getattr(self, "quantization_config", None), "dequantize", False)
            and keep_mxfp4_experts_packed()
        ):
            return 17 / 32
        return original(self, model, param_name, param)

    param_element_size._unsloth_mxfp4_patched = True
    Mxfp4HfQuantizer.param_element_size = param_element_size
pass
TEMPORARY_PATCHES.append(patch_mxfp4_quantizer_element_size)


def _dense_packed_linear(module):
    dense = nn.Linear(module.in_features, module.out_features, bias = False, device = "meta")
    dense.weight = nn.Parameter(module.dequantize_weight().cpu(), requires_grad = False)
    if module.bias is not None:
        dense.bias = module.bias
    return dense


def _restore_modules(swaps):
    for parent, child, module, _ in reversed(swaps):
        setattr(parent, child, module)


def _swap_out_packed_modules(model):
    """Swap packed modules for the dense ones the checkpoint names, so the model's own code reloads the save."""
    from ..mxfp4_stacked_experts import dense_expert_modules

    found = []
    for name, module in list(model.named_modules()):
        stacked = getattr(type(module), "_unsloth_mxfp4_stacked_experts", False)
        if not stacked and not getattr(type(module), "_unsloth_mxfp4_packed_linear", False):
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if stacked and hasattr(parent, "base_layer"):
            raise RuntimeError(
                "Unsloth: this model's MXFP4 experts still carry LoRA adapters. Call "
                "`merge_and_unload()` before a full `save_pretrained`, or save the adapter alone."
            )
        found.append((parent, child, module, name, stacked))
    swaps = []
    try:
        for parent, child, module, name, stacked in found:
            dense = dense_expert_modules(module) if stacked else _dense_packed_linear(module)
            setattr(parent, child, dense)
            swaps.append((parent, child, module, name))
    except BaseException:
        _restore_modules(swaps)
        raise
    return swaps


# Set on a per-Linear packed module a PEFT merge made dense; the full save must describe it as dense.
_DENSIFIED_STATE = "_unsloth_mxfp4_packed_state"


def _densified_module_names(model):
    return [name for name, module in model.named_modules() if _DENSIFIED_STATE in module.__dict__]


def _dense_state_dict(state_dict, swaps):
    swapped = {name: (parent, child) for parent, child, _, name in swaps}
    owners = {key.rpartition(".")[0] for key in state_dict} & swapped.keys()
    if not owners:
        return state_dict
    out = {k: v for k, v in state_dict.items() if k.rpartition(".")[0] not in owners}
    for name in owners:
        parent, child = swapped[name]
        out.update(getattr(parent, child).state_dict(prefix = name + "."))
    return out


def _quant_dict(quant):
    if isinstance(quant, dict):
        return quant
    try:
        return quant.to_dict()
    except Exception:
        return {}


def _is_mxfp4_compressed_config(quant) -> bool:
    quant = _quant_dict(quant)
    inner = quant.get("quantization_config")
    if isinstance(inner, dict) and "config_groups" in inner:
        quant = dict(inner, quant_method = quant.get("quant_method"))
    method = str(quant.get("quant_method", "")).lower().replace("_", "-")
    groups = [g for g in (quant.get("config_groups") or {}).values() if isinstance(g, dict)]
    top = quant.get("format")
    return method in ("compressed-tensors", "sparseml") and bool(groups) and all(
        (g.get("format") or top) == "mxfp4-pack-quantized" for g in groups
    )


def _config_for_dense_save(config, names):
    """Drop MXFP4 compressed-tensors configs (sub-configs too), make bnb skip the dense modules; returns an undo."""
    restores = []
    seen, stack = set(), [config]
    while stack:
        cfg = stack.pop()
        if cfg is None or id(cfg) in seen:
            continue
        seen.add(id(cfg))
        stack.extend(
            v for k, v in vars(cfg).items()
            if not k.startswith("_") and hasattr(v, "to_dict") and hasattr(v, "__dict__")
        )
        quant = cfg.__dict__.get("quantization_config", None)
        if quant is None:
            continue
        if _is_mxfp4_compressed_config(quant):
            del cfg.quantization_config
            restores.append(lambda cfg = cfg, quant = quant: setattr(cfg, "quantization_config", quant))
            continue
        method = str(getattr(_quant_dict(quant).get("quant_method", ""), "value", _quant_dict(quant).get("quant_method", "")))
        if "bitsandbytes" not in method.lower():
            continue
        get = quant.get if isinstance(quant, dict) else (lambda k, d = None: getattr(quant, k, d))
        skip = get("llm_int8_skip_modules", None)
        widened = list(skip or []) + [n for n in names if n not in (skip or [])]

        def assign(value, quant = quant):
            if isinstance(quant, dict):
                quant["llm_int8_skip_modules"] = value
            else:
                quant.llm_int8_skip_modules = value

        assign(widened)
        restores.append(lambda assign = assign, skip = skip: assign(skip))

    def restore():
        for undo in reversed(restores):
            undo()
    return restore


def patch_save_pretrained_mxfp4():
    """Full saves write packed experts dequantized, then restore them; adapter-only saves never reach this."""
    try:
        from transformers import PreTrainedModel
    except Exception:
        return
    original = PreTrainedModel.save_pretrained
    if getattr(original, "_unsloth_mxfp4_patched", False):
        return

    @functools.wraps(original)
    def save_pretrained(self, *args, **kwargs):
        swaps = _swap_out_packed_modules(self)
        dense_names = [name for *_, name in swaps] + _densified_module_names(self)
        if not dense_names:
            return _save_pretrained_expert_params(self, *args, **kwargs)
        try:
            restore_config = _config_for_dense_save(self.config, dense_names)
            try:
                try:
                    bound = inspect.signature(original).bind(self, *args, **kwargs)
                except TypeError:
                    bound = None
                if bound is not None and bound.arguments.get("state_dict", None) is not None:
                    bound.arguments["state_dict"] = _dense_state_dict(bound.arguments["state_dict"], swaps)
                    args, kwargs = bound.args[1:], bound.kwargs
                return _save_pretrained_expert_params(self, *args, **kwargs)
            finally:
                restore_config()
        finally:
            _restore_modules(swaps)

    def _save_pretrained_expert_params(self, *args, **kwargs):
        packed = [
            (module, name, param)
            for module in self.modules()
            for name, param in list(module._parameters.items())
            if is_mxfp4_expert_param(param)
        ]
        if not packed:
            return original(self, *args, **kwargs)
        names = {id(param): name for name, param in self.named_parameters(remove_duplicate = False)
                 if is_mxfp4_expert_param(param)}
        dense = {}
        try:
            for module, name, param in packed:
                weight = param.dequantize().cpu()
                dense[names.get(id(param))] = weight
                module._parameters[name] = nn.Parameter(weight, requires_grad = False)
            try:
                bound = inspect.signature(original).bind(self, *args, **kwargs)
            except TypeError:
                return original(self, *args, **kwargs)
            state_dict = bound.arguments.get("state_dict", None)
            if state_dict is not None:
                bound.arguments["state_dict"] = {key: dense.get(key, value) for key, value in state_dict.items()}
            return original(*bound.args, **bound.kwargs)
        finally:
            for module, name, param in packed:
                module._parameters[name] = param

    save_pretrained._unsloth_mxfp4_patched = True
    PreTrainedModel.save_pretrained = save_pretrained
pass
TEMPORARY_PATCHES.append(patch_save_pretrained_mxfp4)


def patch_convert_moe_packed_tensors():
    """Pin the GPU convert_moe_packed_tensors with a smaller default chunk."""
    try:
        import transformers.integrations.mxfp4
        from transformers.integrations.mxfp4 import FP4_VALUES
    except Exception as e:
        return raise_error("transformers.integrations.mxfp4", e)

    def convert_moe_packed_tensors(
        blocks,
        scales,
        *,
        dtype: torch.dtype = torch.bfloat16,
        rows_per_chunk: int = 32768 * 1024,
    ) -> torch.Tensor:
        """Dequantize mxfp4 weights into GPT_OSS-compatible form (GPU path)."""
        # Move CPU tensors to GPU if available.
        if not blocks.is_cuda and torch.cuda.is_available():
            blocks = blocks.cuda()
            scales = scales.cuda()
        if blocks.is_cuda and blocks.dtype == torch.uint8 and dtype in (torch.bfloat16, torch.float16):
            return mxfp4_dequantize(blocks, scales, dtype = dtype)

        scales = scales.to(torch.int32) - 127

        assert blocks.shape[:-1] == scales.shape, f"{blocks.shape=} does not match {scales.shape=}"

        lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

        *prefix_shape, G, B = blocks.shape
        rows_total = math.prod(prefix_shape) * G

        blocks = blocks.reshape(rows_total, B)
        scales = scales.reshape(rows_total, 1)

        out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

        for r0 in range(0, rows_total, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, rows_total)

            blk = blocks[r0:r1]
            exp = scales[r0:r1]

            # nibble indices -> int64
            idx_lo = (blk & 0x0F).to(torch.long)
            idx_hi = (blk >> 4).to(torch.long)

            sub = out[r0:r1]
            sub[:, 0::2] = lut[idx_lo]
            sub[:, 1::2] = lut[idx_hi]

            torch.ldexp(sub, exp, out=sub)
            del idx_lo, idx_hi, blk, exp, sub

        out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
        del blocks, scales, lut
        return out
    patch_function(transformers.integrations.mxfp4, "convert_moe_packed_tensors", convert_moe_packed_tensors)

    """
    Transformers 4.55.4 did dequantized.transpose(1, 2).contiguous().to(target_device)
    but new versions > 4.56.0 removed the transpose(1, 2) and moved it into patch_convert_moe_packed_tensors
    """
    # convert_moe_packed_tensors above returns the UN-transposed [E, D, G*B*2] layout
    # on purpose (saving_utils._mxfp4_base_returns_transposed keys the export path off
    # that convention), so the live loader hook must restore GPT-OSS's [E, G*B*2, D].
    # Which hook is live:
    #   4.x             -> module level mxfp4.dequantize, called by quantizer_mxfp4
    #   5.0.0 and newer -> Mxfp4Dequantize (a ConversionOps) -> dequantize_convertops
    # mxfp4.dequantize survives unreferenced from 5.0.0 until 5.16.0 (upstream PR
    # #47579, the DTensor TP rewrite) deletes it, so patching only dequantize dropped
    # the transpose from 5.0.0 on and loaded GPT-OSS with dims 1 and 2 silently
    # swapped.
    #
    # 5.0.0 alone declares dequantize_convertops(blocks, scales, target_device); every
    # release from 5.1.0 on declares (blocks, scales). A 2-arg replacement against the
    # 3-arg original is refused by can_safely_patch ("Parameter count mismatch: 3 vs
    # 2"), which left 5.0.0 with the un-transposed convert_moe_packed_tensors above and
    # nothing restoring the transpose. So pick the arm that matches what is actually
    # installed.
    #
    # Dispatch on the observed parameter names rather than on transformers_version:
    # arity is the exact property can_safely_patch enforces, so reading it directly
    # cannot disagree with it, whereas a version gate is a proxy that a backport or a
    # fork can falsify. An unrecognised signature falls through to the 2-arg arm and is
    # rejected loudly by can_safely_patch, which is the correct failure mode.
    _convertops = getattr(transformers.integrations.mxfp4, "dequantize_convertops", None)
    if _convertops is not None:
        # 5.x path, called only by Mxfp4Dequantize.convert. Both arms close over the
        # local un-transposed convert_moe_packed_tensors rather than re-reading the
        # module attribute, so the transpose stays correct even if patch_function above
        # did not take and upstream's self-transposing version is still installed.
        try:
            _convertops_params = tuple(inspect.signature(_convertops).parameters)
        except (TypeError, ValueError):
            _convertops_params = ()

        if _convertops_params == ("blocks", "scales", "target_device"):
            # 5.0.0. Mirrors upstream's own body (empty_cache before the move, and the
            # result placed on target_device) with the transpose added back.
            def dequantize_convertops(blocks, scales, target_device):
                if blocks.dtype == torch.uint8 and blocks.dim() == 4 and keep_mxfp4_experts_packed():
                    return Mxfp4ExpertParam(blocks.to(target_device), mxfp4_scales = scales.to(target_device))
                dequantized = _dequantize_to_gpt_oss_layout(convert_moe_packed_tensors, blocks, scales)
                if target_device == "cpu" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return torch.nn.Parameter(dequantized.to(target_device))
        else:
            # 5.1.0 and newer. Upstream leaves placement to its caller here.
            def dequantize_convertops(blocks, scales):
                if blocks.dtype == torch.uint8 and blocks.dim() == 4 and keep_mxfp4_experts_packed():
                    return Mxfp4ExpertParam(blocks, mxfp4_scales = scales)
                return torch.nn.Parameter(_dequantize_to_gpt_oss_layout(convert_moe_packed_tensors, blocks, scales))
        patch_function(transformers.integrations.mxfp4, "dequantize_convertops", dequantize_convertops)

    if transformers_version < Version("5.0.0"):
        # 4.x path. shard_and_distribute_module is imported inside the gate because on
        # 5.16.0+ it still imports fine but is a tombstone that raises when called.
        try:
            import transformers.integrations.mxfp4
            from transformers.integrations.tensor_parallel import shard_and_distribute_module
        except Exception as e:
            return raise_error("transformers.integrations.mxfp4.dequantize", e)

        def dequantize(module, param_name, param_value, target_device, dq_param_name, **kwargs):
            model = kwargs.get("model", None)
            empty_param = kwargs.get("empty_param", None)
            casting_dtype = kwargs.get("casting_dtype", None)
            to_contiguous = kwargs.get("to_contiguous", None)
            rank = kwargs.get("rank", None)
            device_mesh = kwargs.get("device_mesh", None)

            for proj in ["gate_up_proj", "down_proj"]:
                if proj in param_name:
                    if device_mesh is not None:
                        # 8 positionals, no set_param: that kwarg was removed in 4.57.0
                        # (and is absent from 5.x), so passing it TypeErrors there.
                        param_value = shard_and_distribute_module(
                            model,
                            param_value,
                            empty_param,
                            dq_param_name,
                            casting_dtype,
                            to_contiguous,
                            rank,
                            device_mesh,
                        )
                    blocks_attr = f"{proj}_blocks"
                    scales_attr = f"{proj}_scales"
                    setattr(module, param_name.rsplit(".", 1)[1], param_value)
                    if hasattr(module, blocks_attr) and hasattr(module, scales_attr):
                        dequantized = convert_moe_packed_tensors(getattr(module, blocks_attr), getattr(module, scales_attr))
                        # [HERE] we must do transpose(1, 2)
                        dequantized = dequantized.transpose(1, 2).contiguous().to(target_device)
                        # TODO: this is perhaps necessary since if target_device is cpu, and the param was on gpu
                        if target_device == "cpu" and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        setattr(module, proj, torch.nn.Parameter(dequantized))
                        delattr(module, blocks_attr)
                        delattr(module, scales_attr)
        patch_function(transformers.integrations.mxfp4, "dequantize", dequantize)

    """
    Add a new CPU-optimized version of convert_moe_packed_tensors with smaller default chunk size.
    """
    try:
        import transformers.integrations.mxfp4
        from transformers.integrations.mxfp4 import FP4_VALUES
    except Exception as e:
        return raise_error("transformers.integrations.mxfp4_CPU", e)

    def convert_moe_packed_tensors_cpu(
        blocks,
        scales,
        *,
        dtype: torch.dtype = torch.bfloat16,
        rows_per_chunk: int = 1024 * 1024,  # CPU-optimized default (~2.6GB temp memory)
    ) -> torch.Tensor:
        """Dequantize mxfp4 weights into GPT_OSS-compatible form (CPU path,
        smaller default chunk).

        rows_per_chunk default 1M rows; per-chunk memory at B=128: 8192 ~22 MB,
        1M ~2.6 GB, 32M ~90 GB.
        """
        # Force tensors onto CPU.
        if blocks.is_cuda:
            blocks = blocks.cpu()
        if scales.is_cuda:
            scales = scales.cpu()

        scales = scales.to(torch.int32) - 127

        assert blocks.shape[:-1] == scales.shape, f"{blocks.shape[:-1]=} does not match {scales.shape=}"

        lut = torch.tensor(FP4_VALUES, dtype=dtype, device='cpu')

        *prefix_shape, G, B = blocks.shape
        rows_total = math.prod(prefix_shape) * G

        blocks = blocks.reshape(rows_total, B)
        scales = scales.reshape(rows_total, 1)

        out = torch.empty(rows_total, B * 2, dtype=dtype, device='cpu')

        for r0 in range(0, rows_total, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, rows_total)

            blk = blocks[r0:r1]
            exp = scales[r0:r1]

            # nibble indices -> int64
            idx_lo = (blk & 0x0F).to(torch.long)
            idx_hi = (blk >> 4).to(torch.long)

            sub = out[r0:r1]
            sub[:, 0::2] = lut[idx_lo]
            sub[:, 1::2] = lut[idx_hi]

            torch.ldexp(sub, exp, out=sub)
            del idx_lo, idx_hi, blk, exp, sub

        out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
        del blocks, scales, lut
        return out

    if hasattr(transformers.integrations.mxfp4, 'convert_moe_packed_tensors'):
        transformers.integrations.mxfp4.convert_moe_packed_tensors_cpu = convert_moe_packed_tensors_cpu
        if UNSLOTH_ENABLE_LOGGING:
            logger.info("Unsloth: Successfully added convert_moe_packed_tensors_cpu function.")
    else:
        if UNSLOTH_ENABLE_LOGGING:
            logger.info("Unsloth: Failed to add convert_moe_packed_tensors_cpu - original function not found.")
pass
TEMPORARY_PATCHES.append(patch_convert_moe_packed_tensors)

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

def keep_mxfp4_experts_packed() -> bool:
    """Whether GPT-OSS MXFP4 experts stay packed in memory (dequantized per layer on the fly by the
    grouped_mm MoE forward, with expert LoRA on top) instead of a bf16 copy made at load.

    On by default for LoRA loads on transformers 5 with the grouped_mm backend, where the fused
    dequant kernel is verified exact on this device. UNSLOTH_MXFP4_KEEP_PACKED=0 restores the
    load-time dequant; =1 also keeps them packed where the kernel is not verified (ROCm, or no
    Triton), using the slower torch dequant there."""
    setting = os.environ.get("UNSLOTH_MXFP4_KEEP_PACKED", "")
    if setting == "0" or transformers_version < Version("5.0.0"):
        return False
    name = os.environ.get("UNSLOTH_MODEL_NAME", "").lower().replace("-", "_")
    if "gpt_oss" not in name or "_load_in_4bit_" in name:
        return False
    if os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") == "1":
        return False
    try:
        # Only the grouped_mm forward (patch_gpt_oss_moe_for_lora) reads experts through
        # _get_base_weight; the loop and Triton backends index the stack directly.
        from .moe_utils import select_moe_backend
        import transformers.models.gpt_oss.modeling_gpt_oss as modeling_gpt_oss
        if not getattr(modeling_gpt_oss.GptOssExperts, "_unsloth_lora_patched", False):
            return False
        if select_moe_backend() != "grouped_mm":
            return False
    except Exception:
        return False
    if setting == "1":
        return True
    if getattr(torch.version, "hip", None) is not None:
        return False
    return mxfp4_kernel_available()


def _dequantize_to_gpt_oss_layout(convert, blocks, scales):
    """GPT-OSS's (E, in, out) expert stack: the fused kernel writes it transposed in one pass;
    otherwise ``convert`` (the un-transposed Unsloth variant) plus a transposing copy."""
    if blocks.is_cuda and blocks.dtype == torch.uint8 and blocks.dim() == 4:
        return mxfp4_dequantize(blocks, scales, transpose = True)
    return convert(blocks, scales).transpose(1, 2).contiguous()


class _Mxfp4ShapeProxy:
    """What PEFT's ParamWrapper sees for a packed stack: its logical shape and compute dtype."""

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
    """LoRA on packed MXFP4 experts through PEFT ``target_parameters``. ``get_param`` reports the
    logical (E, in, out) shape so the adapters match the bf16 stack's. Merging adds the delta to
    the dequantized stack and installs it as a bf16 parameter (a merged weight is not MXFP4);
    unmerging restores the untouched packed stack."""
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
        setattr(self.get_base_layer(), self.parameter_name, packed)
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
    """transformers pre-allocates the model's size on the GPU before loading
    (caching_allocator_warmup), counting dequantized experts as bf16. Experts kept packed take
    17 / 32 bytes per value (4-bit values plus one e8m0 scale per 32), so count that instead;
    otherwise loading needs the bf16 model's memory the packed experts exist to avoid."""
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


def patch_save_pretrained_mxfp4():
    """A full ``save_pretrained`` of a model whose experts are kept packed writes them dequantized,
    as the load-time path would have, and the packed stacks are put back afterwards. Adapter-only
    saves (PEFT) never reach this and stay as cheap as before."""
    try:
        from transformers import PreTrainedModel
    except Exception:
        return
    original = PreTrainedModel.save_pretrained
    if getattr(original, "_unsloth_mxfp4_patched", False):
        return

    @functools.wraps(original)
    def save_pretrained(self, *args, **kwargs):
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
            state_dict = kwargs.get("state_dict", None)
            if state_dict is not None:
                kwargs["state_dict"] = {key: dense.get(key, value) for key, value in state_dict.items()}
            return original(self, *args, **kwargs)
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
        # One fused Triton pass, bit-identical to the loop below, with no int64 / fp32 temporaries.
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

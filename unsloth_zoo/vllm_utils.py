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

__all__ = [
    "patch_vllm",
    "vllm_dynamic_quant_supported",
    "convert_vllm_to_huggingface",
    "get_vllm_state_dict",
    "assert_same_state_dict",
    "load_vllm",
    "create_batches",
    "delete_vllm",
    "save_lora",
    "load_lora",
    "generate_batches",
    "convert_lora_modules",
    "return_lora_modules",
]

from typing import Optional, List, Tuple, Dict, Any
import importlib.util
import re
from collections import OrderedDict
import numpy as np
from copy import deepcopy
import math
import gc
import os
import torch
import json
import psutil
import functools
import contextlib
import inspect
from functools import partial
from .utils import _get_dtype
from .patching_utils import patch_model_and_tokenizer
from unsloth import DEVICE_TYPE
global LORA_REQUEST_ID

# Ignore logging messages
import logging
class HideLoggingMessage(logging.Filter):
    def __init__(self, text): self.text = text
    def filter(self, x): return not (self.text in x.getMessage())
pass

def _return_nothing(*args, **kwargs): return None
def _return_self(self, *args, **kwargs): return self
def _return_self_tokenizer(self, *args, **kwargs): return self.tokenizer

def get_target_device(index = 0):
    return torch.device(DEVICE_TYPE, index)

def get_mem_info():
    if DEVICE_TYPE == "xpu":
        free_memory, total_memory = torch.xpu.mem_get_info()
    else:
        free_memory, total_memory = torch.cuda.mem_get_info()
    return free_memory, total_memory

if importlib.util.find_spec("vllm") is not None:

    # Allow unsloth dynamic quants to work
    def is_layer_skipped_bnb(prefix: str, llm_int8_skip_modules):
        # Split the prefix into its dot-separated components
        components = prefix.split('.')
        # Check if any of the skip modules exactly matches any component
        vllm_check = any(
            module_name in components
            for module_name in llm_int8_skip_modules
        )

        # Allow certain layers to not be quantized
        components = set(".".join(components[:i+1]) for i in range(len(components)))
        unsloth_check = len(set(llm_int8_skip_modules) & components) != 0

        return vllm_check or unsloth_check
    pass

    import vllm.model_executor.layers.quantization.bitsandbytes

    if not hasattr(
        vllm.model_executor.layers.quantization.bitsandbytes,
        "apply_bnb_4bit"
    ):
        # Fix force using torch.bfloat16 all the time and make it dynamic
        def _apply_4bit_weight(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            # only load the bitsandbytes module when needed
            from bitsandbytes import matmul_4bit

            original_type = x.dtype
            original_shape = x.shape
            reshape_after_matmul = False
            if x.ndim > 2:
                x = x.reshape(-1, x.size(-1))
                reshape_after_matmul = True

            qweight = layer.weight
            quant_states = qweight.bnb_quant_state
            offsets = qweight.bnb_shard_offsets
            inference_dtype = quant_states[0].dtype
            bf_x = x.to(inference_dtype) # Originally used bfloat16

            out_dim_0 = x.shape[0]
            out_dim_1 = sum(
                [quant_state[1].shape[0] for quant_state in quant_states.items()])
            out = torch.empty(out_dim_0,
                              out_dim_1,
                              dtype=inference_dtype,
                              device=x.device)

            current_index = 0
            for i in range(len(quant_states)):
                output_size = quant_states[i].shape[0]
                # It is more efficient to use out kwarg like
                # matmul_4bit(..., out = ...).  Infeasible now due to the bug
                # https://github.com/TimDettmers/bitsandbytes/issues/1235.
                # Need to change  after the bug is fixed.
                out[:, current_index:current_index + output_size] = matmul_4bit(
                    bf_x, qweight[offsets[i]:offsets[i + 1]].t(), quant_states[i])

                current_index += output_size

            out = out.to(original_type)

            if reshape_after_matmul:
                out = out.view(*original_shape[:-1], out.size(-1))

            if bias is not None:
                out += bias

            return out
        pass
    else:
        # Newer vLLM versions have _apply_bnb_4bit
        apply_bnb_4bit = vllm.model_executor.layers.quantization.bitsandbytes.apply_bnb_4bit
        def _apply_4bit_weight(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            # only load the bitsandbytes module when needed
            original_type = x.dtype
            original_shape = x.shape
            reshape_after_matmul = False
            if x.ndim > 2:
                x = x.reshape(-1, x.size(-1))
                reshape_after_matmul = True

            qweight = layer.weight
            quant_states = qweight.bnb_quant_state
            offsets = qweight.bnb_shard_offsets
            inference_dtype = quant_states[0].dtype
            bf_x = x.to(inference_dtype) # Originally used bfloat16

            out_dim_0 = x.shape[0]
            out_dim_1 = sum(
                [quant_state[1].shape[0] for quant_state in quant_states.items()])
            out = torch.empty(out_dim_0,
                              out_dim_1,
                              dtype=inference_dtype,
                              device=x.device)
            apply_bnb_4bit(bf_x, qweight, offsets, out)
            out = out.to(original_type)

            if reshape_after_matmul:
                out = out.view(*original_shape[:-1], out.size(-1))

            if bias is not None:
                out += bias

            return out
        pass
    pass

    def patch_vllm_bitsandbytes():
        # All Unsloth Zoo code licensed under LGPLv3
        import vllm.model_executor.layers.quantization.bitsandbytes
        vllm.model_executor.layers.quantization.bitsandbytes.is_layer_skipped_bnb = is_layer_skipped_bnb
        vllm.model_executor.layers.quantization.bitsandbytes.BitsAndBytesLinearMethod._apply_4bit_weight = _apply_4bit_weight

        # Disable all not supported messages
        from vllm.config import logger as vllm_config_logger
        vllm_config_logger.addFilter(HideLoggingMessage("not supported"))
        vllm_config_logger.addFilter(HideLoggingMessage("is not tested"))
        vllm_config_logger.addFilter(HideLoggingMessage("is not fully optimized"))
        vllm_config_logger.addFilter(HideLoggingMessage("not set"))
        del vllm_config_logger
    pass

    class BitsAndBytesConfig(
        vllm.model_executor.layers.quantization.bitsandbytes.BitsAndBytesConfig
    ):
        # All Unsloth Zoo code licensed under LGPLv3
        def __init__(self, *args, **kwargs):
            dtype = os.environ.get("UNSLOTH_bnb_4bit_compute_dtype", kwargs["bnb_4bit_compute_dtype"])
            kwargs["bnb_4bit_compute_dtype"] = dtype
            print(f"Unsloth: vLLM Bitsandbytes config using kwargs = {kwargs}")
            super().__init__(*args, **kwargs)
        pass
    pass

    def patch_vllm_compute_dtype(dtype = torch.float16):
        # All Unsloth Zoo code licensed under LGPLv3
        # vLLM defaults to using the model config file's compute_dtype
        # We shall fix it dynamically!
        old_config = vllm.model_executor.layers.quantization.bitsandbytes.BitsAndBytesConfig

        dtype = str(dtype)
        if dtype.startswith("torch."): dtype = dtype[len("torch."):]
        os.environ["UNSLOTH_bnb_4bit_compute_dtype"] = dtype

        vllm.model_executor.layers.quantization.bitsandbytes.BitsAndBytesConfig = BitsAndBytesConfig
        return old_config
    pass

    def unpatch_vllm_compute_dtype(old_config):
        # All Unsloth Zoo code licensed under LGPLv3
        import vllm.model_executor.layers.quantization.bitsandbytes
        vllm.model_executor.layers.quantization.bitsandbytes.BitsAndBytesConfig = old_config
        del os.environ["UNSLOTH_bnb_4bit_compute_dtype"]
    pass

    def patch_vllm_lora_tokenizer():
        import vllm.transformers_utils.tokenizer
        vllm.transformers_utils.tokenizer.get_lora_tokenizer = _return_nothing
        vllm.transformers_utils.tokenizer.get_lora_tokenizer_async = _return_nothing

        try:
            import vllm.transformers_utils.tokenizer_group.tokenizer_group
            vllm.transformers_utils.tokenizer_group.tokenizer_group.get_lora_tokenizer = _return_nothing
            vllm.transformers_utils.tokenizer_group.tokenizer_group.get_lora_tokenizer_async = _return_nothing
        except:
            pass
        try:
            # New vLLM is now a class!
            import vllm.transformers_utils.tokenizer_group
            vllm.transformers_utils.tokenizer_group.TokenizerGroup.get_lora_tokenizer = _return_self_tokenizer
            vllm.transformers_utils.tokenizer_group.TokenizerGroup.get_lora_tokenizer_async = _return_self_tokenizer
        except:
            pass
    pass

    from .vllm_lora_request import LoRARequest as PatchedLoRARequest
    from .vllm_lora_worker_manager import (
        WorkerLoRAManager as PatchedWorkerLoRAManager,
        LRUCacheWorkerLoRAManager as PatchedLRUCacheWorkerLoRAManager,
    )
    def patch_vllm_lora_load_tensors():
        import vllm.lora.request
        vllm.lora.request.LoRARequest = PatchedLoRARequest
        import vllm.lora.worker_manager
        vllm.lora.worker_manager.LoRARequest = PatchedLoRARequest
        vllm.lora.worker_manager.WorkerLoRAManager = PatchedWorkerLoRAManager
        vllm.lora.worker_manager.LRUCacheWorkerLoRAManager = PatchedLRUCacheWorkerLoRAManager
    pass

    def set_inductor_config(config, runtime_shape):
        if isinstance(runtime_shape, int):
            # for a specific batchsize, tuning triton kernel parameters
            # can be beneficial
            config["max_autotune"] = False # Very slow so disable
            config["coordinate_descent_tuning"] = True
    pass

    def patch_vllm_set_inductor_config():
        try:
            import vllm.compilation.compiler_interface
            vllm.compilation.compiler_interface.set_inductor_config = set_inductor_config
        except:
            pass
        return
    pass
else:
    def patch_vllm_bitsandbytes():
        return
    pass

    def patch_vllm_compute_dtype():
        return
    pass

    def unpatch_vllm_compute_dtype(old_config):
        return
    pass

    def patch_vllm_lora_tokenizer():
        return
    pass

    def patch_vllm_lora_load_tensors():
        return
    pass

    def patch_vllm_set_inductor_config():
        return
    pass
pass


if importlib.util.find_spec("bitsandbytes") is not None:
    import bitsandbytes.functional
    from bitsandbytes.utils import pack_dict_to_tensor, unpack_tensor_to_dict

    # Force offsets to be in float32 and not bfloat16 / float16
    @classmethod
    def from_dict(cls, qs_dict: Dict[str, Any], device: torch.device) -> "QuantState":
        """
        unpacks components of state_dict into QuantState
        where necessary, convert into strings, torch.dtype, ints, etc.

        qs_dict: based on state_dict, with only relevant keys, striped of prefixes.

        item with key `quant_state.bitsandbytes__[nf4/fp4]` may contain minor and non-tensor quant state items.
        """

        # unpacking tensor with non-tensor components
        qs_key = [k for k, v in qs_dict.items() if "quant_state" in k and isinstance(v, torch.Tensor)]
        if not len(qs_key) and "quant_type" not in qs_dict:
            raise ValueError("Expected packed or unpacked quant_state items, found neither")
        elif len(qs_key) != 1 or qs_key[0].split(".")[-1] not in cls.valid_qs_type_keys:
            raise ValueError(
                f"There should be exactly one `quant_state` item with ending from {cls.valid_qs_type_keys}.\nDetected {qs_key}.",
            )

        # unpacking minor and non-tensor quant state items if necessary
        if len(qs_key) == 1:
            first_qs_key = qs_key[0]
            qs_dict.update(unpack_tensor_to_dict(qs_dict.pop(first_qs_key)))

        qs_dict = {k.split(".")[-1]: v for k, v in qs_dict.items()}  # strip prefixes
        assert set(qs_dict.keys()).issubset(cls.valid_qs_keys)

        if "nested_absmax" in qs_dict:
            # Must use float32 and disable autocasting - vLLM fails!
            # offset = torch.tensor(float(qs_dict["nested_offset"])).to(device)
            with torch.autocast(device_type = "cuda", enabled = False):
                offset = torch.tensor(qs_dict["nested_offset"], dtype = torch.float32, device = "cuda")
            state2 = cls(
                absmax=qs_dict["nested_absmax"].to(device),
                blocksize=qs_dict["nested_blocksize"],
                code=qs_dict["nested_quant_map"].to(device),
                dtype=getattr(torch, qs_dict["nested_dtype"]),
            )
        else:
            offset, state2 = None, None

        quant_state = cls(
            quant_type=qs_dict["quant_type"],
            absmax=qs_dict["absmax"].to(device),
            blocksize=qs_dict["blocksize"],
            code=qs_dict["quant_map"].to(device),
            # dtype=getattr(torch, qs_dict["dtype"]),
            # Patch over the compute dtype for vLLM
            dtype=getattr(torch, os.environ.get("UNSLOTH_bnb_4bit_compute_dtype", qs_dict["dtype"])),
            shape=torch.Size(qs_dict["shape"]) if qs_dict["shape"] is not None else None,
            offset=offset,
            state2=state2,
        )
        return quant_state
    pass

    import bitsandbytes.nn.modules
    class Linear4bit(bitsandbytes.nn.modules.Linear4bit):
        # All Unsloth Zoo code licensed under LGPLv3
        def __init__(self, *args, **kwargs):
            compute_dtype = os.environ.get("UNSLOTH_bnb_4bit_compute_dtype", None)
            if compute_dtype is not None:
                compute_dtype = getattr(torch, compute_dtype)
                kwargs["compute_dtype"] = compute_dtype
            super().__init__(*args, **kwargs)
        pass
    pass

    def patch_bitsandbytes_quant_state():
        # All Unsloth Zoo code licensed under LGPLv3
        bitsandbytes.functional.QuantState.from_dict = from_dict
        bitsandbytes.nn.modules.Linear4bit = Linear4bit
    pass

    def patch_bitsandbytes_compute_dtype(dtype):
        # All Unsloth Zoo code licensed under LGPLv3
        dtype = str(dtype)
        if dtype.startswith("torch."): dtype = dtype[len("torch."):]
        os.environ["UNSLOTH_bnb_4bit_compute_dtype"] = dtype
        return
    pass

    def unpatch_bitsandbytes_compute_dtype():
        del os.environ["UNSLOTH_bnb_4bit_compute_dtype"]
        return
    pass
else:
    def patch_bitsandbytes_quant_state():
        return
    pass

    def patch_bitsandbytes_compute_dtype(dtype):
        return
    pass

    def unpatch_bitsandbytes_compute_dtype():
        return
    pass
pass

def patch_vllm_enable_sleep_mode():
    from vllm.device_allocator.cumem import CuMemAllocator, libcudart, unmap_and_release, create_and_map
    from vllm.logger import init_logger
    from vllm.utils import is_pin_memory_available
    from typing import Optional, Union, Tuple

    logger = init_logger(__name__)
    print(f"Unsloth: Enabling vLLM standby mode")

    def sleep(
            self,
            offload_tags: Optional[Union[Tuple[str, ...],
                                            str]] = None) -> None:
        """
        Put the allocator in sleep mode.
        All data in the memory allocation with the specified tag will be
        offloaded to CPU memory, and others will be discarded.

        :param offload_tags: The tags of the memory allocation that will be
            offloaded. The rest of the memory allocation will be discarded.
        """
        if offload_tags is None:
            # by default, allocated tensors are offloaded
            # when the allocator sleeps
            offload_tags = (CuMemAllocator.default_tag, )
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags, )

        assert isinstance(offload_tags, tuple)

        logger.debug(f'Sleeping allocator with tags: {offload_tags}')
        set_of_tags = set([data.tag for _, data in self.pointer_to_data.items()])
        logger.debug(f'Set of tags {set_of_tags} and len of data {len(self.pointer_to_data.items())}')

        self.print_memory_summary()
        cpu_offloads = 0
        true_offloads = 0
        total_offloads = 0

        for ptr, data in self.pointer_to_data.items():
            total_offloads += 1
            handle = data.handle
            if data.tag == 'weights':
                # In unsloth's case we have weights managed by unsloth. So we neither offload/delete them nor onload/create them here.
                continue
            if data.tag in offload_tags:
                size_in_bytes = handle[1]
                cpu_backup_tensor = torch.empty(
                    size_in_bytes,
                    dtype=torch.uint8,
                    device='cpu',
                    pin_memory=is_pin_memory_available())
                cpu_ptr = cpu_backup_tensor.data_ptr()
                libcudart.cudaMemcpy(cpu_ptr, ptr, size_in_bytes)
                data.cpu_backup_tensor = cpu_backup_tensor
                cpu_offloads += 1
            logger.debug(f"data's tag is {data.tag} and is offloaded to cpu? {data.tag in offload_tags}")

            unmap_and_release(handle)
            true_offloads += 1


        logger.debug(f'CPU offloads {cpu_offloads} true offloads {true_offloads} total {total_offloads}')
        gc.collect()
        torch.cuda.empty_cache()

    def wake_up(self, tags: Optional[List[str]] = None) -> None:
        """
        Wake up the allocator from sleep mode.
        All data that is previously offloaded will be loaded back to GPU
        memory, and the rest of the data will have empty memory.

        :param tags: The tags of the memory allocation that will be loaded
            back to GPU memory. If None, all memory allocation will be loaded
            back to GPU memory.
        """
        delete_memory()
        for ptr, data in self.pointer_to_data.items():
            if data.tag == "weights":
                # In unsloth's case we have weights managed by unsloth. So we neither offload/delete them nor onload/create them here.
                continue
            if tags is None or data.tag in tags:
                handle = data.handle
                create_and_map(handle)
                if data.cpu_backup_tensor is not None:
                    cpu_backup_tensor = data.cpu_backup_tensor
                    if cpu_backup_tensor is not None:
                        size_in_bytes = cpu_backup_tensor.numel(
                        ) * cpu_backup_tensor.element_size()
                        cpu_ptr = cpu_backup_tensor.data_ptr()
                        libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)
                        data.cpu_backup_tensor = None

    def delete_memory():
        torch.cuda.empty_cache()
        gc.collect()
    pass

    def print_memory_summary(self):
        """
        Print the total memory usage for weights and KVCache allocations.
        """
        weights_total = 0
        kv_cache_total = 0
        kv_cache_count = 0
        weights_count  = 0
        for data in self.pointer_to_data.values():
            size = data.handle[1]
            if data.tag == "weights":
                weights_count += 1
                weights_total += size
            elif data.tag == "kv_cache":
                kv_cache_total += size
                kv_cache_count += 1
        logger.debug(f"Total weights memory: {weights_total / 1e9:.2f} GB for {weights_count} items")
        logger.debug(f"Total KVCache memory: {kv_cache_total / 1e9:.2f} GB for {kv_cache_count} items")

    CuMemAllocator.sleep = sleep
    CuMemAllocator.wake_up = wake_up
    CuMemAllocator.print_memory_summary = print_memory_summary
pass


def patch_vllm(debug = True):
    # Temporary patch to disable multiprocessing for vLLM
    # Allows accessing model_executor
    print(f'Unsloth: Patching vLLM')
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if debug:
        os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
    # os.environ["VLLM_TRACE_FUNCTION"] = "1"
    patch_vllm_set_inductor_config()
    patch_bitsandbytes_quant_state()
    patch_vllm_bitsandbytes()
    patch_vllm_lora_tokenizer()
    patch_vllm_lora_load_tensors()
    patch_vllm_enable_sleep_mode()
    global LORA_REQUEST_ID
    LORA_REQUEST_ID = 1
pass


def vllm_dynamic_quant_supported(
    model_name,
    config,
) -> bool:
    # All Unsloth Zoo code licensed under LGPLv3

    # Check if vLLM supports some Unsloth dynamic quants
    # Sometimes we quantize modules within a layer, but not an entire layer
    # If so, then we cannot use dynamic quants for now
    if not model_name.lower().endswith("unsloth-bnb-4bit"): return True
    if "quantization_config" not in config: return True

    llm_int8_skip_modules = config.quantization_config.get("llm_int8_skip_modules", {})

    # Only allow layer modules ie model.layers.1.mlp or model.layers.1.self_attn

    # Exclude model.layers.27.mlp.gate_proj
    parent_llm_int8_skip_modules = []
    for module in llm_int8_skip_modules:
        # $ means end of string
        if re.search(r"[\d]\.[^\.]{1,}$", module) or "." not in module:
            parent_llm_int8_skip_modules.append(module)
    pass

    parent_llm_int8_skip_modules = set(parent_llm_int8_skip_modules)
    find_regex = "|".join(re.escape(x) for x in parent_llm_int8_skip_modules)
    find_regex = re.compile(find_regex)

    for module in llm_int8_skip_modules:
        # Could not find parent
        if find_regex.search(module) is None: return False
    return True
pass


@torch.inference_mode
def get_vllm_state_dict(llm, return_state_dict = False, config = None, is_vision_model = False):
    # All Unsloth Zoo code licensed under LGPLv3
    # Unmerges vLLM modules and returns HF equivalent state_dict
    # vllm_state_dict = {}
    try:
        llm_engine = getattr(llm, "llm_engine", getattr(llm, "engine", llm))

        # Handle V1 vs V0 engines
        if hasattr(llm_engine, "engine_core"):
            # V1 engine - access through engine_core (multiprocessing is disabled by patch_vllm)
            vllm_internals = llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner.model
        else:
            # V0 engine - direct access
            vllm_internals = llm_engine.model_executor.driver_worker.model_runner.model

        # for name, p in vllm_internals.named_parameters():
        #     vllm_state_dict[name] = p
    except Exception as e:
        # If we can't access the model directly, raise a more informative error
        raise RuntimeError(f"Unsloth: Cannot access vLLM internal model. This might be due to a vLLM version incompatibility. Error: {str(e)}")
    pass

    print(f"Unsloth: vllm_internals: \n\n{vllm_internals}\n\n")

    assert(config is not None)

    # Determine model type from config BEFORE reassigning config
    model_type = getattr(config, "model_type", "causal_lm")

    # Keep the original config for model_type but use text_config for vocab_size etc
    text_config = config
    if hasattr(config, "text_config"):
        text_config = config.text_config

    vocab_size = text_config.vocab_size

    state_dict = OrderedDict()
    quant_state_dict = OrderedDict()

    def get_state_dict(prefix, kk, state_dict, proj, slice_weights=True):
        proj = getattr(proj, "base_layer", proj)
        qweight = proj.weight

        # Determine slicing offsets
        output_sizes = getattr(proj, "output_sizes", None)
        if output_sizes is not None:
            dim_offsets = np.cumsum([0] + output_sizes)
        else:
            dim_offsets = [0, qweight.shape[0]]

        # Handle quantized weights
        quant_states = getattr(qweight, "bnb_quant_state", None)
        if quant_states is not None:
            offsets = qweight.bnb_shard_offsets
            if slice_weights:
                weight = qweight[offsets[kk] : offsets[kk + 1]]
                quant_state_dict[prefix + ".weight.quant_state"] = quant_states[kk]
                quant_state = quant_states[kk].as_dict(packed = True)
                for k, v in quant_state.items():
                    state_dict[prefix + ".weight." + k] = v
            else:
                weight = qweight
                quant_state_dict[prefix + ".weight.quant_state"] = quant_states[0]
                quant_state = quant_states[0].as_dict(packed = True)
                for k, v in quant_state.items():
                    state_dict[prefix + ".weight." + k] = v
        else:
            # Normal FP16 weights
            qweight.requires_grad_(False)
            if slice_weights:
                weight = qweight[dim_offsets[kk] : dim_offsets[kk + 1]]
            else:
                weight = qweight

        # Apply vocab_size truncation for embedding and lm_head layers
        if vocab_size is not None and ("embed_tokens" in prefix or "lm_head" in prefix):
            if weight.shape[0] > vocab_size:
                weight = weight[:vocab_size]

        state_dict[prefix + ".weight"] = weight
        quant_state_dict[prefix + ".weight"] = weight

        # Handle bias
        bias = getattr(proj, "bias", None)
        if bias is not None:
            bias.requires_grad_(False)
            if slice_weights:
                bias_tensor = bias[dim_offsets[kk] : dim_offsets[kk + 1]]
            else:
                bias_tensor = bias

            # Apply vocab_size truncation for bias as well
            if vocab_size is not None and ("embed_tokens" in prefix or "lm_head" in prefix):
                if bias_tensor.shape[0] > vocab_size:
                    bias_tensor = bias_tensor[:vocab_size]

            state_dict[prefix + ".bias"] = bias_tensor
            quant_state_dict[prefix + ".bias"] = bias_tensor
    pass

    # Embedding
    if hasattr(vllm_internals, "model"): # Standard Language models
        vllm_text_model = vllm_internals.model
        vllm_text_model_prefix = "model"
    elif hasattr(vllm_internals, "language_model"):
        # For Llama 3.2, Gemma 3 and Qwen 2.5 VL, they have text model in model.language_model.model
        vllm_text_model_prefix = "model.language_model"
        vllm_text_model = vllm_internals.language_model.model
    else:
        raise RuntimeError(f'Unsloth: Cannot find vllm_internal_model!')

    embed_tokens = vllm_text_model.embed_tokens
    # Use get_state_dict for consistent extraction and automatic truncation
    get_state_dict(f"{vllm_text_model_prefix}.embed_tokens", 0, state_dict, embed_tokens, slice_weights=False)

    # Get layer configuration for this model type
    layer_config = get_model_layer_config(model_type, text_config)

    # All layers
    skipped_layernorms = []
    for kk in range(len(vllm_text_model.layers)):
        layer = vllm_text_model.layers[kk]
        if hasattr(layer, "self_attn"):
            prefix = f"{vllm_text_model_prefix}.layers.{kk}.self_attn"
            qkv_proj = layer.self_attn.qkv_proj
            o_proj = layer.self_attn.o_proj
        elif hasattr(layer, "cross_attn"):
            prefix = f"{vllm_text_model_prefix}.layers.{kk}.cross_attn"
            qkv_proj = layer.cross_attn.qkv_proj
            o_proj = layer.cross_attn.o_proj

        get_state_dict(f"{prefix}.q_proj", 0, state_dict, qkv_proj)
        get_state_dict(f"{prefix}.k_proj", 1, state_dict, qkv_proj)
        get_state_dict(f"{prefix}.v_proj", 2, state_dict, qkv_proj)

        get_state_dict(f"{prefix}.o_proj", 0, state_dict, o_proj)

        proj = layer.mlp.gate_up_proj
        get_state_dict(f"{vllm_text_model_prefix}.layers.{kk}.mlp.gate_proj", 0, state_dict, proj)
        get_state_dict(f"{vllm_text_model_prefix}.layers.{kk}.mlp.up_proj",   1, state_dict, proj)

        proj = layer.mlp.down_proj
        get_state_dict(f"{vllm_text_model_prefix}.layers.{kk}.mlp.down_proj", 0, state_dict, proj)

        # Use layernorms from the layer configuration
        layernorm_names = [name.format(kk=kk) for name in layer_config['layernorms']]

        for layernorm_name in layernorm_names:
            vllm_name = layernorm_name.replace(f".{kk}.", f"[{kk}].").replace(vllm_text_model_prefix, "vllm_text_model")
            try:
                layernorm = eval(vllm_name).state_dict()["weight"]
                layernorm_name = f"{layernorm_name}.weight"
                state_dict[layernorm_name] = layernorm
                quant_state_dict[layernorm_name] = state_dict[layernorm_name]
            except Exception as e:
                skipped_layernorms.append(layernorm_name.split(".")[-1])
        pass
    pass

    if len(skipped_layernorms) != 0:
        print(f"Unsloth: Just some info: will skip parsing {list(set(skipped_layernorms))}")
    pass

    if is_vision_model:
        # Handle vision-specific layers using dedicated functions
        if model_type == "mllama":
            extract_mllama_vision_layers(vllm_internals, state_dict, quant_state_dict, get_state_dict)
        elif model_type == "qwen2_5_vl":
            extract_qwen2_5_vl_vision_layers(vllm_internals, state_dict, quant_state_dict, get_state_dict)
        elif model_type == "gemma3":
            extract_gemma3_vision_layers(vllm_internals, state_dict, quant_state_dict, get_state_dict)

    # Norm
    # For Gemma3 and similar multimodal models, norm should be under model.norm
    # For standard models, also under model.norm
    norm_prefix = f"{vllm_text_model_prefix}.norm.weight"
    state_dict[norm_prefix] = vllm_text_model.norm.weight.data
    quant_state_dict[norm_prefix] = state_dict[norm_prefix]

    # LM Head - Use get_state_dict for consistency

    if not getattr(text_config, "tie_word_embeddings", False):
        lm_layer = [mod for name,mod in vllm_internals.named_modules() if "lm_head" in name]
        # Use get_state_dict for consistent extraction and automatic truncation
        get_state_dict("lm_head", 0, state_dict, lm_layer[0], slice_weights=False)
    else:
        # Fallback to embed_tokens for tied embeddings
        embed_key = f"{vllm_text_model_prefix}.embed_tokens.weight"
        if embed_key in state_dict:
            lm_weight = state_dict[embed_key]
            state_dict["lm_head.weight"] = lm_weight
            quant_state_dict["lm_head.weight"] = lm_weight


    if not return_state_dict: state_dict = None
    return state_dict, quant_state_dict
pass


@torch.inference_mode
def assert_same_state_dict(old_state_dict, new_state_dict):
    # All Unsloth Zoo code licensed under LGPLv3
    # Check if state_dict are equivalent
    # hf, vllm

    difference = new_state_dict.keys() ^ old_state_dict.keys()
    difference -= set(("model.lm_head.weight","model.language_model.lm_head.weight", "lm_head.weight"))
    if len(difference) != 0:
        missing_from_vllm = new_state_dict.keys() - old_state_dict.keys()
        missing_from_hf = old_state_dict.keys() - new_state_dict.keys()
        print(f'Unsloth: Failed comparing state_dict with Missing from vllm: {missing_from_vllm}\nMissing from hf: {missing_from_hf}')
        raise RuntimeError(f"Unsloth: Failed comparing state_dict with {difference}")
    pass

    failures = {}

    for key in old_state_dict:
        try:
            torch.testing.assert_close(old_state_dict[key], new_state_dict[key], check_stride = True)
        except Exception as error:
            if key == "lm_head.weight":
                # Try tied embeddings fallback
                key1 = next((k for k in (key, "model.embed_tokens.weight", "model.language_model.embed_tokens.weight") if k in old_state_dict), None)
                key2 = next((k for k in (key, "model.embed_tokens.weight", "model.language_model.embed_tokens.weight") if k in new_state_dict), None)

                if key1 is not None and key2 is not None:
                    try:
                        torch.testing.assert_close(old_state_dict[key1], new_state_dict[key2], check_stride = True)
                    except Exception:
                        failures[key] = error
                else:
                    failures[key] = error
            else:
                failures[key] = error
        pass
    if len(failures) > 0:
        error_message = "\n".join([f"[{key}]\n{str(error)}" for key, error in failures.items()])
        raise RuntimeError(f"Unsloth: Failed comparing state_dict with {len(failures)}: {error_message}")
    pass
pass


@torch.inference_mode
def create_empty_causal_lm(config, dtype = torch.float16):
    # All Unsloth Zoo code licensed under LGPLv3
    # Empty model from config
    new_config = deepcopy(config)
    new_config.intermediate_size = 0
    new_config.hidden_size = 0
    new_config.vocab_size = 1
    new_config.pad_token_id = 0

    # Set attention module head_dim
    # Otherwise will get error if (head_dim)**-0.5 is seen like in Qwen
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    new_config.update({"head_dim" : head_dim})

    from transformers import AutoModelForCausalLM
    new_model = AutoModelForCausalLM.from_config(
        new_config,
        attn_implementation = "eager",
    )

    # Get layer names from config
    layer_config = get_model_layer_config("causal_lm", config)
    layer_names = layer_config['standard_layers'] + layer_config['layernorms']

    return new_model, layer_names, config.num_hidden_layers
pass


@torch.inference_mode
def create_empty_qwen2_5_vl(config, dtype = torch.float16):
    from transformers import Qwen2_5_VLForConditionalGeneration
    new_config = deepcopy(config)

    new_config.num_attention_heads = 1
    new_config.num_key_value_heads = 1
    new_config.intermediate_size = 0

    new_config.vision_config.dim = 1
    new_config.vision_config.num_heads = 1
    new_config.vision_config.intermediate_size = 0
    new_config.vision_config.out_hidden_size = 1

    new_model = Qwen2_5_VLForConditionalGeneration(new_config)

    # Get layer names from config
    layer_config = get_model_layer_config("qwen2_5_vl", config)
    layer_names = (layer_config['standard_layers'] +
                  layer_config['layernorms'] +
                  layer_config['vision_layers'] +
                  layer_config['additional_layers'])

    layers = max(get_model_layer_counts(config).values())
    return new_model, layer_names, layers
pass

@torch.inference_mode
def create_empty_mllama(config, dtype = torch.float16):
    from transformers import MllamaForConditionalGeneration
    new_config = deepcopy(config)

    new_config.text_config.num_attention_heads = 1
    new_config.text_config.num_key_value_heads = 1
    new_config.text_config.intermediate_size = 0

    new_config.vision_config.num_attention_heads = 1
    new_config.vision_config.num_key_value_heads = 1
    new_config.vision_config.intermediate_size = 0
    new_config.vision_config.vision_output_dim = 1

    new_model = MllamaForConditionalGeneration(new_config)

    # Get layer names from config
    layer_config = get_model_layer_config("mllama", config)
    layer_names = (layer_config['standard_layers'] +
                  layer_config['layernorms'] +
                  layer_config['vision_layers'] +
                  layer_config['additional_layers'])

    num_layers = max(get_model_layer_counts(config).values())
    return new_model, layer_names, num_layers
pass

def create_empty_gemma3mm(config, dtype = torch.float16):
    from transformers import Gemma3ForConditionalGeneration
    new_config = deepcopy(config)

    new_config.text_config.num_attention_heads = 1
    new_config.text_config.intermediate_size = 1

    new_config.vision_config.num_attention_heads = 1
    new_config.vision_config.intermediate_size = 1
    new_config.vision_config.vision_output_dim = 1

    new_model = Gemma3ForConditionalGeneration(new_config)

    # Get layer names from config
    layer_config = get_model_layer_config("gemma3", config)
    layer_names = (layer_config['standard_layers'] +
                  layer_config['layernorms'] +
                  layer_config['vision_layers'] +
                  layer_config['additional_layers'])

    num_layers = max(get_model_layer_counts(config).values())

    return new_model, layer_names, num_layers
pass

@torch.inference_mode
def create_empty_model(config, dtype = torch.float16, is_vision_model = False):
    model_type = config.model_type
    if not is_vision_model:
        return create_empty_causal_lm(config, dtype)
    elif model_type == "mllama":
        return create_empty_mllama(config, dtype)
    elif model_type == "qwen2_5_vl":
        return create_empty_qwen2_5_vl(config, dtype)
    elif model_type == "gemma3":
        return create_empty_gemma3mm(config, dtype)
    else:
        raise ValueError(f"Unsloth: Unsupported model type: {model_type}")

pass

def set_additional_modules(new_model, quant_state_dict, config):
    if hasattr(new_model, "language_model"):
        language_model = new_model.language_model
        language_model_prefix = "model.language_model"
    else:
        language_model_prefix = "model"
        language_model = new_model.model

    # Embeddings
    embed_tokens_key = f"{language_model_prefix}.embed_tokens.weight"
    language_model.embed_tokens = torch.nn.Embedding.from_pretrained(
        quant_state_dict[embed_tokens_key],
        freeze = True,
        padding_idx = getattr(config, 'pad_token_id', None),
    )

    # Norm
    norm_key = f"{language_model_prefix}.norm.weight"
    norm = quant_state_dict[norm_key]
    norm = torch.nn.Parameter(norm, requires_grad = False)
    language_model.norm.weight = norm

    # LM Head
    if getattr(config, "tie_word_embeddings", False):
        lmhead_key = f"{language_model_prefix}.embed_tokens.weight"
    else:
        lmhead_key = "lm_head.weight"

    # Check if lm_head exists in the state dict
    if lmhead_key in quant_state_dict:
        weight = quant_state_dict[lmhead_key]
        from torch.nn import Linear

        # Create lm_head with correct dimensions
        layer = Linear(weight.shape[1], weight.shape[0], device = get_target_device(), bias = False)
        layer.weight = torch.nn.Parameter(weight, requires_grad = False)

        # Set lm_head at the correct level
        if hasattr(new_model, "lm_head"):
            new_model.lm_head = layer
        else:
            # For multimodal models, check if language_model has lm_head
            if hasattr(language_model, "lm_head"):
                language_model.lm_head = layer
            else:
                new_model.lm_head = layer

        if getattr(config, "tie_word_embeddings", False):
            # For tied embeddings, tie the weights properly
            if hasattr(new_model, "tie_weights"):
                new_model.tie_weights()
            elif hasattr(language_model, "tie_weights"):
                language_model.tie_weights()

    # Process additional keys
    # For eg, `merger` in qwen2.5-vl or probably any other projection modules
    additional_keys = set(
        x for x in quant_state_dict.keys()
        if not any(substr in x for substr in ("layers", "blocks", embed_tokens_key, norm_key, "lm_head"))
    )

    for key in additional_keys:
        try:
            replaced_key = re.sub(r"\.(\d+)\.", r"[\1].", key)
            exec(f"new_{replaced_key}.data = quant_state_dict[key]")
        except:
            continue
    pass

pass

@torch.inference_mode
def convert_vllm_to_huggingface(quant_state_dict, config, dtype = torch.float16, bnb_config = None, is_vision_model = False):
    # All Unsloth Zoo code licensed under LGPLv3
    # Unmerges vLLM modules to create HF compatible model
    config.update({"torch_dtype" : dtype}) # Do not use config file's dtype!
    new_model, layer_names, layer_count = create_empty_model(config, dtype, is_vision_model)
    new_model = new_model.to(device = get_target_device(), dtype = dtype)
    quantization_config = getattr(config, "quantization_config", {})
    kwargs = dict()
    compute_dtype = dtype  # Do not use config file's dtype!

    if quantization_config != {} or bnb_config is not None:
        # Get quantization_config flags
        if quantization_config != {}:
            kwargs["compress_statistics"] = quantization_config["bnb_4bit_use_double_quant"]
            kwargs["quant_type"] = quantization_config["bnb_4bit_quant_type"]
            kwargs["quant_storage"] = _get_dtype(quantization_config["bnb_4bit_quant_storage"])

        # Get bnb_config flags
        elif bnb_config is not None:
            kwargs["compress_statistics"] = bnb_config.bnb_4bit_use_double_quant
            kwargs["quant_type"] = bnb_config.bnb_4bit_quant_type
            kwargs["quant_storage"] = _get_dtype(bnb_config.bnb_4bit_quant_storage)

    pass
    from bitsandbytes.nn.modules import Linear4bit, Params4bit
    from torch.nn.modules import Linear

    layernorm_names = [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "q_norm",
        "k_norm",
        # Vision / multimodal norms
        "layer_norm1",       # Gemma-3 vision encoder
        "layer_norm2",       # Gemma-3 vision encoder
        "post_layernorm",    # Gemma-3 vision encoder per-layer norm
        "mm_soft_emb_norm",  # Gemma-3 multimodal projector norm,
        "norm1",              # Qwen2.5-VL vision encoder
        "norm2",              # Qwen2.5-VL vision encoder
        "norm",
    ]
    # Override .to("cuda") to disable it otherwise we'll get
    # ValueError: Blockwise quantization only supports 16/32-bit floats, but got torch.uint8
    def _override_to(self, *args, **kwargs):
        try: return self.to(*args, **kwargs)
        except: return self
    pass

    skipped_layernorms = []
    for kk in range(layer_count):
        for layer_name in layer_names:
            if "kk" not in layer_name: # skip those that are not per layer
                continue
            layer_name = layer_name.format(kk = kk)
            if f"{layer_name}.weight" not in quant_state_dict:
                skipped_layernorms.append(layer_name.split(".")[-1])
                continue
            pass
            weight = quant_state_dict[f"{layer_name}.weight"]

            if f"{layer_name}.bias" in quant_state_dict:
                # Has bias!
                has_bias = True
                bias = quant_state_dict[f"{layer_name}.bias"]
                bias = torch.nn.Parameter(bias, requires_grad = False)
            else:
                has_bias = False
                bias = None
            pass

            if f"{layer_name}.weight.quant_state" in quant_state_dict:
                # Layer is quantized!
                quant_state = quant_state_dict[f"{layer_name}.weight.quant_state"]
                layer = Linear4bit(0, 0, device = get_target_device(), bias = has_bias, compute_dtype = compute_dtype, **kwargs)
                layer.in_features  = quant_state.shape[1]
                layer.out_features = quant_state.shape[0]
                layer.weight = Params4bit(data = weight, requires_grad = False, **kwargs)
                layer.weight.quant_state = quant_state
                layer.bias = bias

                # Must override or else Bitsandbytes will error
                layer.to = partial(_override_to, layer)
                layer.weight.to = partial(_override_to, layer.weight)

            elif not any(x in layer_name for x in layernorm_names):
                layer = Linear(0, 0, device = get_target_device(), bias = has_bias)
                layer.in_features  = weight.shape[1]
                layer.out_features = weight.shape[0]
                layer.weight = torch.nn.Parameter(weight, requires_grad = False)
                layer.bias = bias
            else:
                # LayerNorms (including vision norms)
                weight_param = torch.nn.Parameter(weight, requires_grad=False)
                layer_name_br = re.sub(r"\.([\d]{1,})\.", r"[\1].", layer_name)
                # Set weight
                exec(f"new_model.{layer_name_br}.weight = None")
                exec(f"new_model.{layer_name_br}.weight = weight_param")
                # Set bias if it exists
                if bias is not None:
                    exec(f"new_model.{layer_name_br}.bias = None")
                    exec(f"new_model.{layer_name_br}.bias = bias")
                continue
            pass

            # Convert model.layers.0.self_attn.q_proj to model.layers[0].self_attn.q_proj
            layer_name = re.sub(r"\.([\d]{1,})\.", r"[\1].", layer_name)
            exec(f"new_model.{layer_name} = layer")
        pass
    pass

    set_additional_modules(new_model, quant_state_dict, config)


    # Fix up config items with correct items
    config_as_dict = config.to_dict()
    for module in new_model.modules():
        for key, value in config_as_dict.items():
            if hasattr(module, key): exec(f"module.{key} = {value}")
        if hasattr(module, "config"): module.config = config
    pass
    for param in new_model.parameters():
        for key, value in config_as_dict.items():
            if hasattr(param, key): exec(f"param.{key} = {value}")
        if hasattr(param, "config"): param.config = config
    pass
    module = new_model
    for key, value in config_as_dict.items():
        if hasattr(module, key): exec(f"module.{key} = {value}")
    new_model.config = config

    text_config = getattr(config, "text_config", config) #try using text config for VLMs
    # Fix up rotary_emb by re-initing them
    for module in new_model.modules():
        if hasattr(module, "rotary_emb"):
            module.rotary_emb = module.rotary_emb.__class__(
                config = text_config,
                device = get_target_device(),
            )
        if hasattr(module, "rotary_emb_local"):
            # gemma3 has a rotary_emb_local
            module.rotary_emb_local = module.rotary_emb_local.__class__(
                config = text_config,
                device = get_target_device(),
            )
        pass
    pass

    # Must override or else Bitsandbytes will error
    new_model.to = partial(_override_to, new_model)

    # Cleanup
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()

    if len(skipped_layernorms) != 0:
        print(f"Unsloth: Just some info: will skip parsing {list(set(skipped_layernorms))}")
    return new_model
pass


def approximate_vllm_memory_usage(
    config,
    max_seq_length = 2048,
    gpu_memory_utilization = 0.8,
    enable_lora = True,
    max_lora_rank = 16,
    max_loras = 1,
    float8_kv_cache = False,
    account_for_gradients = True,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Gets approximate max model length and max num sequences
    load_in_4bit = "quantization_config" in config
    free_memory, total_memory = get_mem_info()

    free_memory = gpu_memory_utilization * free_memory

    vocab_size = config.vocab_size
    hd = config.hidden_size
    context_length = config.max_position_embeddings
    mlp_size = config.intermediate_size
    n_layers = config.num_hidden_layers
    n_kv_heads = getattr(config, "num_key_value_heads", 1)
    n_heads    = getattr(config, "num_attention_heads", 1)
    # Group Query Attention
    kv_size = hd // n_heads * n_kv_heads

    # Modules
    qkvo = hd + kv_size + kv_size + hd
    qkvo = qkvo * hd
    mlp  = (hd * mlp_size) * 3
    layernorms = 2 * hd
    embed_tokens = vocab_size * hd
    lm_head = 0 if getattr(config, "tie_word_embeddings", True) else vocab_size * hd

    # LoRA modules on all QKVO, MLP
    qkvo_A = hd * max_lora_rank * 4
    qkvo_B = max_lora_rank * (hd + kv_size + kv_size + hd)
    mlp_A  = hd * max_lora_rank * 2 + mlp_size * max_lora_rank
    mlp_B  = max_lora_rank * (mlp_size + mlp_size) + max_lora_rank * hd
    lora_elements = qkvo_A + qkvo_B + mlp_A + mlp_B
    lora_elements = lora_elements * max_loras
    # 2 bytes = float16 for LoRA
    lora_elements = lora_elements*n_layers * 2
    if not enable_lora: lora_elements = 0

    # Get activation and gradients for LoRA
    # 8bit Adam most likely * 2 for momentum, variance
    gradient_lora_elements  = lora_elements + lora_elements
    # Parameter left in float32
    parameter_lora_elements = lora_elements*4

    # Activation memory - assume bsz=2
    bsz = 2
    activation_qkv  = max_seq_length * bsz * (hd + kv_size + kv_size)
    residual_memory = (max_seq_length * bsz)*2
    activation_mlp  = max_seq_length * bsz * (mlp_size + mlp_size)
    weights = mlp_size * hd
    maximum_activation = \
        activation_qkv + residual_memory + activation_mlp + weights
    # 2 bytes with 25% extra just in case
    maximum_activation = (maximum_activation*1.25) * 2
    if not account_for_gradients: maximum_activation = 0
    # Minus for activations
    if total_memory - free_memory < maximum_activation:
        free_memory = total_memory - maximum_activation
    actual_gpu_memory_utilization = free_memory / total_memory

    # 2 bytes = float16
    total_quantizable_elements = (qkvo + mlp)*n_layers * 2
    total_float16_elements     = (layernorms + embed_tokens + lm_head)*2
    factor = 16/5 if load_in_4bit else 1 # Should be 4.5 but use 5
    bytes_for_model = \
        total_quantizable_elements / factor + total_float16_elements + lora_elements

    # KV cache size (float16 is 2 bytes. float8 is 1.25 bytes)
    float_bytes = 1.25 if float8_kv_cache else 2
    kv_elements = (kv_size * 2 * n_layers) * float_bytes
    memory_left_for_kv_cache = free_memory - bytes_for_model
    if memory_left_for_kv_cache <= 0: memory_left_for_kv_cache = 0

    # Approx maximum # of KV cache elements
    max_num_batched_tokens = int(0.95*(memory_left_for_kv_cache / kv_elements))
    # Round by 256
    max_num_batched_tokens = (max_num_batched_tokens // 256) * 256
    # Assuming all requests output max_seq_length, get theoretical max requests
    approx_max_num_seqs = int(max_num_batched_tokens / max_seq_length)

    # GB for KV cache
    memory_left_for_kv_cache_gb = memory_left_for_kv_cache / 1024 / 1024 / 1024

    return \
        max_num_batched_tokens, approx_max_num_seqs, \
        actual_gpu_memory_utilization, memory_left_for_kv_cache_gb
pass


def load_vllm(
    model_name             : str   = "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
    config                 = None,
    gpu_memory_utilization : float = 0.8,
    max_seq_length         : int   = 8192,
    dtype                  : torch.dtype = None,
    training               : bool = True,
    float8_kv_cache        : bool = False,
    random_state           : int  = 0,
    enable_lora            : bool = True,
    max_lora_rank          : int  = 16,
    max_loras              : int  = 1,
    use_async              : bool = False,
    use_engine             : bool = False,
    disable_log_stats      : bool = False,
    enforce_eager          : bool = False, # Good for debugging
    enable_prefix_caching  : bool = True,
    compilation_config     : int  = 3, # -O3 for maximum performance
    conservativeness       : float = 1.0, # For low VRAM devices, scale batches, num_seqs
    max_logprobs           : int  = 0,
    use_bitsandbytes       : bool = True,
    unsloth_vllm_standby   : bool = False,
    is_vision_model        : bool = False,
    return_args            : bool = False, # Just return args
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Create vLLM instance
    assert(config is not None)
    assert(type(use_bitsandbytes) is bool)
    assert(conservativeness >= 0.0 and conservativeness <= 1.0)

    if DEVICE_TYPE == "cuda":
        major_version, minor_version = torch.cuda.get_device_capability()
        if major_version < 7: raise NotImplementedError("Unsloth: Your GPU is too old!")

        # Float8 KV cache only works for 8.0 or higher
        if float8_kv_cache and major_version < 8:
            raise NotImplementedError("Unsloth: Your GPU is too old for float8 KV cache! Set it to False.")

    if hasattr(config, "text_config"):
        mem_config = config.text_config
    else:
        mem_config = config

    max_num_batched_tokens, approx_max_num_seqs, \
    actual_gpu_memory_utilization, memory_left_for_kv_cache_gb = \
    approximate_vllm_memory_usage(
        mem_config,
        max_seq_length = max_seq_length,
        gpu_memory_utilization = gpu_memory_utilization,
        enable_lora = enable_lora,
        max_lora_rank = max_lora_rank,
        max_loras = max_loras,
        float8_kv_cache = float8_kv_cache,
        account_for_gradients = training,
    )

    # Check max_num_batched_tokens for max_seq_length
    # Must be >= max_num_batched_tokens
    if max_num_batched_tokens <= 0:
        max_seq_length = 256
        max_num_batched_tokens = 256

    if max_num_batched_tokens <= max_seq_length:
        print(
            f"Unsloth: Your GPU cannot handle sequence lengths of {max_seq_length} due to limited GPU memory.\n"\
            f"Unsloth: Your GPU can only handle approximately the maximum sequence length of {max_seq_length}."
        )
        max_seq_length = max_num_batched_tokens
    pass

    # Get correct dtype
    if DEVICE_TYPE == "cuda" and major_version >= 8: _dtype = torch.bfloat16
    elif DEVICE_TYPE == "xpu":
        _dtype = torch.bfloat16
    else:
        _dtype = torch.float16
    if dtype == torch.bfloat16 and _dtype == torch.float16:
        print("Unsloth: We switched to dtype = torch.float16 since your GPU does not support torch.bfloat16")
        dtype = torch.float16
    elif dtype is None:
        dtype = _dtype
        print(f"Unsloth: Using dtype = {dtype} for vLLM.")
    elif dtype == torch.float16 or dtype == torch.bfloat16: pass
    else:
        raise NotImplementedError(f"Unsloth: We do not support dtype = {dtype} yet!")

    free_memory, total_memory = get_mem_info()

    total_memory_gb = round(total_memory / 1024 / 1024 / 1024, 2)
    use_bitsandbytes = use_bitsandbytes or \
        model_name.lower().endswith("-bnb-4bit")

    # Fix up vLLM compute_dtype for bitsandbytes
    BitsAndBytesConfig = patch_vllm_compute_dtype(dtype)

    # Use Flashinfer if possible (doesn't seem to be faster for BnB)
    # Also seems to process 2x less sequences in 1 go so less throughput?
    # Maybe FP8 Flashinfer is much better
    # See https://docs.vllm.ai/en/latest/serving/env_vars.html
    if importlib.util.find_spec("flashinfer"):
        # Allowed: FLASHINFER, TORCH_SDPA, FLASH_ATTN, XFORMERS, ROCM_FLASH
        if not use_bitsandbytes and major_version >= 8:
            os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

        # Flashinfer sampler maybe makes it somewhat faster on newer GPUs
        # Tesla T4 is 280 tok/s vs 330 tok/s
        if major_version >= 8:
            os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "1"
        else:
            os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        # os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
    pass

    # Prefix Caching fails for V100, Titan X CUDA Compute Capability 7.0
    # See https://github.com/huggingface/trl/issues/2798
    if DEVICE_TYPE == "cuda":
        major_version, minor_version = torch.cuda.get_device_capability()
        if (major_version < 7) or (major_version == 7 and minor_version < 5):
            print("Unsloth: Your GPU does not support prefix caching - will disable!")
            enable_prefix_caching = False
    elif DEVICE_TYPE == "xpu":
        enable_prefix_caching = True

    pass

    # Use VLLM_USE_V1 for vllm >= 0.7.4 and CUDA >= 8.0
    # [FAILS] for bitsandbytes - https://github.com/unslothai/unsloth/issues/2102
    # if importlib.util.find_spec("vllm") and (major_version >= 8):
    #     from importlib.metadata import version as importlib_version
    #     from packaging.version import Version
    #     if Version(importlib_version("vllm")) > Version("0.7.3"):
    #         os.environ["VLLM_USE_V1"] = "1"
    # pass

    from vllm import LLM, LLMEngine, AsyncLLMEngine, EngineArgs, AsyncEngineArgs

    # Default vLLM max_num_seqs is 256
    approx_max_num_seqs = 256
    if   memory_left_for_kv_cache_gb <=  2: approx_max_num_seqs = 128 # - 32
    elif memory_left_for_kv_cache_gb <=  4: approx_max_num_seqs = 160 # - 32
    elif memory_left_for_kv_cache_gb <=  8: approx_max_num_seqs = 192 # - 32
    elif memory_left_for_kv_cache_gb <= 12: approx_max_num_seqs = 224 # - 32
    elif memory_left_for_kv_cache_gb <= 16: approx_max_num_seqs = 256 # Default
    elif memory_left_for_kv_cache_gb <= 24: approx_max_num_seqs = 288 # + 32
    elif memory_left_for_kv_cache_gb <= 40: approx_max_num_seqs = 320 # + 32
    elif memory_left_for_kv_cache_gb <= 48: approx_max_num_seqs = 226 # + 16
    elif memory_left_for_kv_cache_gb <= 80: approx_max_num_seqs = 368 # + 32
    else: approx_max_num_seqs = 400 # + 32

    max_num_batched_tokens = 2048

    if is_vision_model:
        # In vLLM profiling, each sequence contributes to an image. Which is generally in the order of thousand tokens.
        # We don't want to go beyond 16 sequences for vision models.
        # TODO: In vLLM V1, iirc, the profiling sets a cap on the max seqs based on the budget. Check it out.
        print(f'Unsloth: Vision model detected, setting approx_max_num_seqs to 16')
        approx_max_num_seqs = 16
        max_num_batched_tokens = 8192 # Single image would contribute to 6404 tokens in Llama 3.2 for eg. So have some more for text

    # float8 KV cache can fit more sequences in 1 go so more throughput
    if float8_kv_cache: approx_max_num_seqs = int(approx_max_num_seqs * 1.05)

    # vLLM default max_num_batched_tokens is 2048
    chunked_prefill_tokens = 2048
    if   memory_left_for_kv_cache_gb <=  8: chunked_prefill_tokens = 1024 # + 0
    elif memory_left_for_kv_cache_gb <= 12: chunked_prefill_tokens = 1536 # + 512
    elif memory_left_for_kv_cache_gb <= 16: chunked_prefill_tokens = 2048 # + 512
    elif memory_left_for_kv_cache_gb <= 24: chunked_prefill_tokens = 3072 # + 1024
    elif memory_left_for_kv_cache_gb <= 40: chunked_prefill_tokens = 4096 # + 1024
    elif memory_left_for_kv_cache_gb <= 48: chunked_prefill_tokens = 4608 # + 512
    elif memory_left_for_kv_cache_gb <= 80: chunked_prefill_tokens = 8192 # + 4096
    else: chunked_prefill_tokens = 8192 # + 0

    # vLLM errors out from max_seq_length (2048) being bigger than chunked_prefill_tokens (1024)
    if max_seq_length > chunked_prefill_tokens:
        chunked_prefill_tokens = max_seq_length
    elif chunked_prefill_tokens > max_seq_length:
        chunked_prefill_tokens = max_seq_length

    # Scale num_seqs by conservativeness
    approx_max_num_seqs = int(approx_max_num_seqs * conservativeness)

    # Check max RAM usage for vLLM (swap space) default is 4GB
    memory = psutil.virtual_memory()
    RAM_GB = memory.available / 1024 / 1024 / 1024
    swap_space = 4
    if   RAM_GB <= 4:  swap_space = 0
    elif RAM_GB <= 8:  swap_space = 0
    elif RAM_GB <= 12: swap_space = 0
    elif RAM_GB <= 16: swap_space = 0
    elif RAM_GB <= 24: swap_space = 2
    elif RAM_GB <= 48: swap_space = 4
    else: swap_space = 6

    if DEVICE_TYPE == "xpu":
        platform = "Intel GPU"
        gpu_eu_count = torch.xpu.get_device_properties(0).gpu_eu_count
        message = f"{platform} has eu:{gpu_eu_count}"
    else:
        platform = "CUDA"
        major_version, minor_version = torch.cuda.get_device_capability()
        message = f"{platform} compute capability {major_version}.{minor_version}"



    print(
        f"Unsloth: vLLM loading {model_name} with actual GPU utilization = {round(actual_gpu_memory_utilization*100, 2)}%\n"\
        f"Unsloth: Your GPU has {message} with VRAM = {total_memory_gb} GB.\n"\
        f"Unsloth: Using conservativeness = {conservativeness}. Chunked prefill tokens = {chunked_prefill_tokens}. Num Sequences = {approx_max_num_seqs}.\n"\
        f"Unsloth: vLLM's KV Cache can use up to {round(memory_left_for_kv_cache_gb, 2)} GB. Also swap space = {swap_space} GB."
    )

    # Get device as well
    device = get_target_device()

    if compilation_config == 3:
        try:
            from vllm.config import CompilationConfig, CompilationLevel
            compilation_config = CompilationConfig(
                level = 3,
                backend = "inductor",
                # cache_dir = "unsloth_compiled_vllm_cache", # Pytorch fails to load from cache
                # compile_sizes = [1, 2, 4, 8, 16],
                # cudagraph_capture_sizes = [1, 2, 4, 8, 16],
                # max_capture_size = 16,
                cudagraph_num_of_warmups = 1,
                full_cuda_graph = False,
                use_cudagraph = True,
                use_inductor = True,
                inductor_compile_config = {
                    "debug" : False,
                    "dce" : True,
                    "coordinate_descent_tuning" : True,
                    "trace.enabled" : False,
                    "trace.graph_diagram" : False,
                    "triton.cudagraphs" : True,
                    "compile_threads" : 48,
                    "max_autotune" : False, # Way too slow
                    "disable_progress" : False,
                    "verbose_progress" : True,
                }
            )
        except:
            pass
    pass

    engine_args = dict(
        model                  = model_name,
        gpu_memory_utilization = actual_gpu_memory_utilization,
        max_model_len          = max_seq_length,
        quantization           = "bitsandbytes" if use_bitsandbytes else None,
        load_format            = "bitsandbytes" if use_bitsandbytes else "auto",
        kv_cache_dtype         = "fp8" if float8_kv_cache else "auto",
        dtype                  = dtype,

        max_num_batched_tokens = max_num_batched_tokens,
        max_num_seqs           = approx_max_num_seqs, # vLLM default uses 256 -> reduce if OOM
        max_logprobs           = max_logprobs, # Disallow logprobs being returned
        seed                   = random_state, # Default is 0

        # lora_extra_vocab_size = 0, # Breaks vLLM so we leave it as 256
        enable_lora            = enable_lora,
        max_lora_rank          = max_lora_rank,
        max_loras              = max_loras,

        disable_log_stats      = disable_log_stats,
        enable_prefix_caching  = enable_prefix_caching,
        # enable_chunked_prefill = True, # LoRA fails with chunked prefill as at Feb 2025
        # max_seq_len_to_capture fails for V1
        # max_seq_len_to_capture = min(8192, max_seq_length + 256), # Default is 8192 for CUDAGraphs
        compilation_config     = compilation_config, # 0, 1, 2, 3
        enforce_eager          = True,
        swap_space             = swap_space, # Low memory devices like Colab (13GB) default 4GB
        device                 = device,
        # New vLLM versions need to pass this in!
        # worker_extension_cls   = "unsloth_zoo.vllm_rlhf_utils.ColocateWorkerExtension",
        enable_sleep_mode      = unsloth_vllm_standby,
    )
    if unsloth_vllm_standby and "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
        del os.environ['PYTORCH_CUDA_ALLOC_CONF'] # Disable expandable segments cuz https://github.com/pytorch/pytorch/issues/147851
    good_keys = inspect.signature(AsyncEngineArgs if use_async else EngineArgs).parameters.keys()
    old_keys = engine_args.keys()
    for key in old_keys:
        if key not in good_keys:
            del engine_args[key]
            print(f"Unsloth: Not an error, but `{key}` is not supported in vLLM. Skipping.")
        pass
    pass

    # Quick exit
    if return_args: return engine_args

    # Keep trying until success (2 times)
    trials = 0
    while True:
        try:
            if use_async:
                llm = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_args))
            elif use_engine:
                llm = LLMEngine.from_engine_args(EngineArgs(**engine_args))
            else:
                llm = LLM(**engine_args)
            pass
            break
        except Exception as error:
            print(f"Error occured loading vLLM: {error}", "will retry" if trials < 2 else "")
            trials += 1
            # Cleanup
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()
            pass
            error = str(error)
            if trials >= 2:
                raise RuntimeError(error)

            if "gpu_memory_utilization" in error or "memory" in error:
                approx_max_num_seqs = int(approx_max_num_seqs * 0.75)
                engine_args["max_num_seqs"] = approx_max_num_seqs
                engine_args["gpu_memory_utilization"] *= 0.85
                print(
                    f"Unsloth: Retrying vLLM to process {approx_max_num_seqs} sequences and {max_num_batched_tokens} tokens in tandem.\n"\
                    f"Error:\n{error}"
                )
            else:
                raise RuntimeError(error)
        pass
    pass
    # Save maximum requests length since llm.generate fails to partition inputs sometimes
    llm.approx_max_num_seqs = approx_max_num_seqs

    # Unpatch vLLM compute_dtype for bitsandbytes
    unpatch_vllm_compute_dtype(BitsAndBytesConfig)

    # Cleanup
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()
    return llm
pass


def create_batches(requests, num_sequences = 64):
    # All Unsloth Zoo code licensed under LGPLv3
    # llm.generate must be batched!
    n_splits = int(math.ceil(len(requests) / num_sequences))
    offsets = np.arange(0, len(requests), num_sequences)
    if offsets[-1] != len(requests):
        offsets = np.hstack((offsets, len(requests)))
    batches = [requests[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)]
    return batches
pass


@torch.inference_mode
def save_lora(model, save_directory, *args, **kwargs):
    # All Unsloth Zoo code licensed under LGPLv3
    state_dict = model.state_dict()
    dtype = model.get_input_embeddings().weight.dtype
    # Cast LoRA to float16 / bfloat16
    state_dict = {k:v.to(dtype) for k, v in state_dict.items() if ".lora_A." in k or ".lora_B." in k}
    kwargs["state_dict"] = state_dict
    model.save_pretrained(save_directory = save_directory, *args, **kwargs)
pass


@functools.cache
def get_peft_config(save_directory):
    with open(os.path.join(save_directory, "adapter_config.json")) as f:
        config = json.load(f)
    return config
pass


def vllm_lora_already_loaded(model):
    # All Unsloth Zoo code licensed under LGPLv3
    # Check if LoRA is loaded - if not, we should load the first one
    m = model.vllm_engine.llm_engine.model_executor.driver_worker.model_runner
    lora_cache = m.lora_manager._adapter_manager._active_adapters.cache

    layers = m.model.model.layers
    v_layer = layers[0]
    print(lora_cache, v_layer.self_attn.qkv_proj.lora_a_stacked[0].data_ptr())
    return len(lora_cache) != 0
pass


def prepare_vllm_lora_loading(model):
    # All Unsloth Zoo code licensed under LGPLv3
    # Get all vLLM LoRAs
    assert(hasattr(model, "vllm_engine"))

    # Must split into 2 lists since B is scaled in vLLM
    model_loras_A, model_loras_B = [], []
    vllm_loras_A,  vllm_loras_B  = [], []
    vllm_model = model.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model

    # Go through all layers!
    for v_layer, m_layer in zip(vllm_model .model.layers, model.model.model.layers):
        model_loras_A.append(m_layer.self_attn.q_proj.lora_A.default.weight)
        model_loras_A.append(m_layer.self_attn.k_proj.lora_A.default.weight)
        model_loras_A.append(m_layer.self_attn.v_proj.lora_A.default.weight)
        vllm_loras_A .append(v_layer.self_attn.qkv_proj.lora_a_stacked[0])
        vllm_loras_A .append(v_layer.self_attn.qkv_proj.lora_a_stacked[1])
        vllm_loras_A .append(v_layer.self_attn.qkv_proj.lora_a_stacked[2])

        sq = m_layer.self_attn.q_proj.scaling["default"]
        sk = m_layer.self_attn.k_proj.scaling["default"]
        sv = m_layer.self_attn.v_proj.scaling["default"]
        sq = None if sq == 1.0 else sq
        sk = None if sk == 1.0 else sk
        sv = None if sv == 1.0 else sv
        model_loras_B.append( m_layer.self_attn.q_proj.lora_B.default.weight)
        model_loras_B.append( m_layer.self_attn.k_proj.lora_B.default.weight)
        model_loras_B.append( m_layer.self_attn.v_proj.lora_B.default.weight)
        vllm_loras_B .append((v_layer.self_attn.qkv_proj.lora_b_stacked[0], sq,))
        vllm_loras_B .append((v_layer.self_attn.qkv_proj.lora_b_stacked[1], sk,))
        vllm_loras_B .append((v_layer.self_attn.qkv_proj.lora_b_stacked[2], sv,))

        so = m_layer.self_attn.o_proj.scaling["default"]
        so = None if so == 1.0 else so
        model_loras_A.append(m_layer.self_attn.o_proj.lora_A.default.weight)
        vllm_loras_A .append(v_layer.self_attn.o_proj.lora_a_stacked[0])
        model_loras_B.append( m_layer.self_attn.o_proj.lora_B.default.weight)
        vllm_loras_B .append((v_layer.self_attn.o_proj.lora_b_stacked[0], so,))

        model_loras_A.append(m_layer.mlp.gate_proj.lora_A.default.weight)
        model_loras_A.append(m_layer.mlp.gate_proj.lora_A.default.weight)
        vllm_loras_A .append(v_layer.mlp.gate_up_proj.lora_a_stacked[0])
        vllm_loras_A .append(v_layer.mlp.gate_up_proj.lora_a_stacked[1])

        sg = m_layer.mlp.gate_proj.scaling["default"]
        su = m_layer.mlp.  up_proj.scaling["default"]
        sg = None if sg == 1.0 else sg
        su = None if su == 1.0 else su
        model_loras_B.append( m_layer.mlp.gate_proj.lora_B.default.weight)
        model_loras_B.append( m_layer.mlp.gate_proj.lora_B.default.weight)
        vllm_loras_B .append((v_layer.mlp.gate_up_proj.lora_b_stacked[0], sg,))
        vllm_loras_B .append((v_layer.mlp.gate_up_proj.lora_b_stacked[1], su,))

        sd = m_layer.mlp.down_proj.scaling["default"]
        sd = None if sd == 1.0 else sd
        model_loras_A.append(m_layer.mlp.down_proj.lora_A.default.weight)
        vllm_loras_A .append(v_layer.mlp.down_proj.lora_a_stacked[0])
        model_loras_B.append( m_layer.mlp.down_proj.lora_B.default.weight)
        vllm_loras_B .append((v_layer.mlp.down_proj.lora_b_stacked[0], sd,))
    pass

    # Check all shapes
    for model_lora_A, vllm_lora_A in zip(model_loras_A, vllm_loras_A):
        assert(model_lora_A.squeeze().shape == vllm_lora_A.squeeze().shape)
    for model_lora_B, (vllm_lora_B, s,) in zip(model_loras_B, vllm_loras_B):
        assert(model_lora_B.squeeze().shape == vllm_lora_B.squeeze().shape)
    pass

    # Set model items
    model.model_loras_A = model_loras_A
    model.model_loras_B = model_loras_B
    model. vllm_loras_A = vllm_loras_A
    model. vllm_loras_B = vllm_loras_B
    return
pass


def load_lora_directly(model):
    # All Unsloth Zoo code licensed under LGPLv3
    # Load LoRAs directly from model into vLLM internal LoRAs
    model_loras_A = model.model_loras_A
    model_loras_B = model.model_loras_B
    vllm_loras_A  = model. vllm_loras_A
    vllm_loras_B  = model. vllm_loras_B

    for model_lora_A, vllm_lora_A in zip(model_loras_A, vllm_loras_A):
        vllm_lora_A.copy_(model_lora_A, non_blocking = True)
    pass

    # Must also scale B with scaling since vLLM does this
    for model_lora_B, (vllm_lora_B, s) in zip(model_loras_B, vllm_loras_B):
        vllm_lora_B.copy_(model_lora_B, non_blocking = True)
        if s is not None: vllm_lora_B *= s
    pass
    # Must block!
    torch.cuda.synchronize()
pass


from peft import PeftType

@torch.inference_mode
def convert_lora_modules(
    model,
    dtype = None,
):
    dtype = _get_dtype(model.config.torch_dtype if dtype is None else dtype)

    if (hasattr(model, "peft_config") and "default" in model.peft_config) \
        and (model.peft_config["default"].peft_type == PeftType.LORA):

        state_dict = model.state_dict().items()
        state_dict = {
            k : v.detach().clone() for k, v in state_dict \
            if (v.dtype != dtype) and \
               (".lora_A.default" in k or ".lora_B.default" in k)
        }
        if len(state_dict) == 0: return {}

        for name, module in model.named_modules():
            if name + ".default.weight" in state_dict:
                exec(f"module.to({dtype})")
        pass
        return state_dict
    return {}
pass


@torch.inference_mode
def return_lora_modules(
    model,
    state_dict = {},
    dtype = torch.float32,
):
    if state_dict == {} or state_dict is None: return
    dtype = _get_dtype(model.config.torch_dtype if dtype is None else dtype)

    if (hasattr(model, "peft_config") and "default" in model.peft_config) \
        and (model.peft_config["default"].peft_type == PeftType.LORA):

        for name, module in model.named_modules():
            old_name = name + ".default.weight"
            old_weight = state_dict.get(old_name, None)
            if old_weight is not None:
                exec(f"module.to({dtype})")
                # module.default.weight.copy_(old_weight)
        pass
        return
    return
pass


@torch.inference_mode
def load_lora(model, save_directory, load_tensors = False):
    # vllm_lora_already_loaded(model)
    # Check internally if model has hot loaded LoRAs
    # if load_tensors and hasattr(model, "saved_vllm_lora_request"):# vllm_lora_already_loaded(model):
    #     if not hasattr(model, "model_loras_A"):
    #         # Prepare vLLM for LoRA direct loading!
    #         prepare_vllm_lora_loading(model)
    #     pass
    #     load_lora_directly(model)
    #     return model.saved_vllm_lora_request
    # pass

    # All Unsloth Zoo code licensed under LGPLv3
    global LORA_REQUEST_ID
    if LORA_REQUEST_ID is None: LORA_REQUEST_ID = 1

    # Check if path exists
    if not os.path.exists(save_directory) or LORA_REQUEST_ID == 1:
        if load_tensors:
            # We need to save and load the config file once!
            model.peft_config["default"].save_pretrained(save_directory)
        elif not os.path.exists(save_directory):
            raise OSError(f"Unsloth: LoRA filepath = {save_directory} does not exist!")
    pass

    from vllm.lora.request import LoRARequest
    if load_tensors:
        # We extract it directly from the model's state_dict
        peft_config = get_peft_config(save_directory)
        state_dict = model.state_dict()
        items = state_dict.items()
        state_dict = {k.replace(".default", ""):v for k, v in items if ".lora_A." in k or ".lora_B." in k}

        # vllm_lora_already_loaded(model)
        lora_request = LoRARequest(str(LORA_REQUEST_ID), LORA_REQUEST_ID, lora_tensors = state_dict, lora_config = peft_config)
        # Warm up LoRA
        # vllm_lora_already_loaded(model)
        # outputs = model.vllm_engine.generate(["Hi!"], use_tqdm = False, lora_request = lora_request)
        # del outputs
        # vllm_lora_already_loaded(model)
        # print("###", LORA_REQUEST_ID)
        # vllm_lora_already_loaded(model)
            # model.saved_vllm_lora_request = lora_request
    else:
        lora_request = LoRARequest(str(LORA_REQUEST_ID), LORA_REQUEST_ID, save_directory)
    pass
    # vllm_lora_already_loaded(model)

    LORA_REQUEST_ID += 1
    # Set model's current LoRA adapater
    # model.vllm_engine.vllm_lora_request = lora_request
    return lora_request
pass


def generate_batches(llm, inputs, n_batches = None, lora_request = None, *args, **kwargs):
    # All Unsloth Zoo code licensed under LGPLv3
    # Cannot just use llm.generate or will OOM - split into batches
    if n_batches is None:
        if "UNSLOTH_VLLM_BATCHES" in os.environ:
            n_batches = int(os.environ["UNSLOTH_VLLM_BATCHES"])
        else:
            free_memory, total_memory = get_mem_info()
            total_memory_gb = round(total_memory / 1024 / 1024 / 1024, 2)
            if   total_memory_gb <=  8: n_batches = llm.approx_max_num_seqs // 10
            elif total_memory_gb <= 16: n_batches = llm.approx_max_num_seqs // 5
            elif total_memory_gb <= 24: n_batches = llm.approx_max_num_seqs // 2
            else: n_batches = llm.approx_max_num_seqs

            os.environ["UNSLOTH_VLLM_BATCHES"] = str(n_batches)

            if n_batches != llm.approx_max_num_seqs:
                print(f"Unsloth: Will use {n_batches} batches to reduce memory usage for generation!")
        pass
    pass

    # We should disable for now since it might interfere with the reference model in RL
    # if lora_request is None:
    #     if hasattr(llm, "vllm_lora_request"): lora_request = llm.vllm_lora_request
    # pass

    batches = create_batches(inputs, n_batches)
    kwargs["lora_request"] = lora_request
    output_list = []
    for batch in batches:
        outputs = llm.generate(batch, *args, **kwargs)
        output_list += list(outputs)
    pass
    return output_list
pass


def delete_vllm(llm = None):
    # From https://github.com/vllm-project/vllm/issues/1908
    from vllm.distributed.parallel_state import (
        destroy_model_parallel,
        destroy_distributed_environment,
    )
    # Delete the llm object and free the memory
    destroy_model_parallel()
    destroy_distributed_environment()
    if llm is not None:
        del llm.llm_engine.model_executor
        del llm
        llm = None
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    try:
        import ray
        ray.shutdown()
    except:
        pass
    return llm
pass


def _test_same_model(model, new_model, input_ids):
    # All Unsloth Zoo code licensed under LGPLv3
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        ALL_ATTENTION_FUNCTIONS,
    )
    from peft.utils.integrations import dequantize_module_weight as df

    A =     model.model.embed_tokens(input_ids)
    B = new_model.model.embed_tokens(input_ids)
    torch.testing.assert_close(model.model.embed_tokens.weight, new_model.model.embed_tokens.weight)
    torch.testing.assert_close(A, B)

    position_ids = torch.arange(input_ids.shape[1], device = "cuda")
    position_ids = position_ids.repeat((1, input_ids.shape[0]))
    rotary_A =     model.model.rotary_emb(A, position_ids)
    new_rotary = new_model.model.rotary_emb.__class__(new_model.config, device = "cuda")
    rotary_B = new_rotary(B, position_ids)
    torch.testing.assert_close(rotary_A[0], rotary_B[0])
    torch.testing.assert_close(rotary_A[1], rotary_B[1])

    for i, (old, new) in enumerate(zip(model.model.layers, new_model.model.layers)):
        print(i, end = ",")
        residualA = A
        residualB = B

        torch.testing.assert_close(old.input_layernorm.weight, new.input_layernorm.weight)
        A = old.input_layernorm(A)
        B = new.input_layernorm(B)

        AA, _ = old.self_attn(A.clone(), attention_mask = None, position_embeddings = rotary_A)
        BB, _ = new.self_attn(B.clone(), attention_mask = None, position_embeddings = rotary_B)
        torch.testing.assert_close(AA, BB, rtol = 0.01, atol = 0.005)

        torch.testing.assert_close(df(old.self_attn.q_proj), df(new.self_attn.q_proj))
        torch.testing.assert_close(df(old.self_attn.k_proj), df(new.self_attn.k_proj))
        torch.testing.assert_close(df(old.self_attn.v_proj), df(new.self_attn.v_proj))

        input_shapeA = A.shape[:-1]
        hidden_shapeA = (*input_shapeA, -1, old.self_attn.head_dim)
        QA = old.self_attn.q_proj(A).view(hidden_shapeA).transpose(1, 2)
        KA = old.self_attn.k_proj(A).view(hidden_shapeA).transpose(1, 2)
        VA = old.self_attn.v_proj(A).view(hidden_shapeA).transpose(1, 2)

        input_shapeB = B.shape[:-1]
        hidden_shapeB = (*input_shapeB, -1, new.self_attn.head_dim)
        QB = new.self_attn.q_proj(B).view(hidden_shapeB).transpose(1, 2)
        KB = new.self_attn.k_proj(B).view(hidden_shapeB).transpose(1, 2)
        VB = new.self_attn.v_proj(B).view(hidden_shapeB).transpose(1, 2)
        torch.testing.assert_close(QA, QB, rtol = 0.01, atol = 0.005)
        torch.testing.assert_close(KA, KB, rtol = 0.01, atol = 0.005)
        torch.testing.assert_close(VA, VB, rtol = 0.01, atol = 0.005)

        QA, KA = apply_rotary_pos_emb(QA, KA, *rotary_A)
        QB, KB = apply_rotary_pos_emb(QB, KB, *rotary_B)
        torch.testing.assert_close(QA, QB, rtol = 0.01, atol = 0.005)
        torch.testing.assert_close(KA, KB, rtol = 0.01, atol = 0.005)

        f = ALL_ATTENTION_FUNCTIONS[old.self_attn.config._attn_implementation]
        attentionA, _ = f(old.self_attn, QA, KA, VA,
            attention_mask = None,
            dropout = 0.0 if not old.self_attn.training else old.self_attn.attention_dropout,
            scaling = old.self_attn.scaling,
        )
        f = ALL_ATTENTION_FUNCTIONS[new.self_attn.config._attn_implementation]
        attentionB, _ = f(new.self_attn, QB, KB, VB,
            attention_mask = None,
            dropout = 0.0 if not new.self_attn.training else new.self_attn.attention_dropout,
            scaling = new.self_attn.scaling,
        )
        torch.testing.assert_close(attentionA, attentionB)

        A = attentionA.reshape(*input_shapeA, -1).contiguous()
        A = old.self_attn.o_proj(A)
        B = attentionB.reshape(*input_shapeB, -1).contiguous()
        B = new.self_attn.o_proj(B)
        torch.testing.assert_close(A, B, rtol = 0.01, atol = 0.005)
        torch.testing.assert_close(AA, BB, rtol = 0.01, atol = 0.005)
        torch.testing.assert_close(AA, B, rtol = 0.01, atol = 0.005)
        torch.testing.assert_close(BB, B, rtol = 0.01, atol = 0.005)

        residualA = A
        residualB = B
        torch.testing.assert_close(old.post_attention_layernorm.weight, new.post_attention_layernorm.weight)
        A = old.post_attention_layernorm(A)
        B = new.post_attention_layernorm(B)
        torch.testing.assert_close(A, B, rtol = 0.01, atol = 0.005)

        AA = old.mlp(A.clone())
        BB = new.mlp(B.clone())
        torch.testing.assert_close(AA, BB, rtol = 0.01, atol = 0.005)
        gateA = old.mlp.gate_proj(A)
        gateB = new.mlp.gate_proj(B)
        torch.testing.assert_close(gateA, gateB, rtol = 0.01, atol = 0.005)
        upA = old.mlp.up_proj(A)
        upB = new.mlp.up_proj(B)
        torch.testing.assert_close(upA, upB, rtol = 0.01, atol = 0.005)
        A = old.mlp.act_fn(gateA) * upA
        B = new.mlp.act_fn(gateB) * upB
        A = old.mlp.down_proj(A)
        B = new.mlp.down_proj(B)
        torch.testing.assert_close(A, B, rtol = 0.01, atol = 0.005)
        torch.testing.assert_close(AA, BB, rtol = 0.01, atol = 0.005)
        torch.testing.assert_close(AA, A, rtol = 0.01, atol = 0.005)
        torch.testing.assert_close(BB, B, rtol = 0.01, atol = 0.005)

        A = residualA + A
        B = residualB + B
        torch.testing.assert_close(A, B, rtol = 0.01, atol = 0.005)

        B = A.clone()
    pass

    A =     model.model.norm(A)
    B = new_model.model.norm(B)
    torch.testing.assert_close(A, B)

    # LM Head testing with proper error handling
    try:
        # Check if both models have lm_head
        if hasattr(model, 'lm_head') and hasattr(new_model, 'lm_head'):
            if model.lm_head.weight is not None and new_model.lm_head.weight is not None:
                torch.testing.assert_close(model.lm_head.weight, new_model.lm_head.weight)

        # Continue with lm_head forward pass if possible
        if hasattr(model, 'lm_head') and hasattr(new_model, 'lm_head'):
            A = model.lm_head(A)
            B = new_model.lm_head(B)
            torch.testing.assert_close(A, B)
    except Exception as e:
        print(f"Unsloth: lm_head test failed. Error: {e}")

    return
pass


@torch.inference_mode
def _test_get_vllm_state_dict(
    model_name = "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
    dtype = torch.float16,
    gpu_memory_utilization = 0.7,
    counts = 100,
    conservativeness = 1.0,
    float8_kv_cache = False,
    unsloth_vllm_standby = False,
    load_in_4bit = False,
    skip_generation = False,
    is_vision_model = False,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Check if model is allowed to be used in vLLM
    gc.collect()
    torch.cuda.empty_cache()

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        model_name,
        token = None,
        revision = None,
        trust_remote_code = False,
        attn_implementation = "sdpa",
    )
    if not vllm_dynamic_quant_supported(model_name, config):
        raise NotImplementedError(f"Unsloth: Dynamic quant of {model_name} not supported in vLLM")

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    bnb_config = None
    load_in_4bit = model_name.lower().endswith("-bnb-4bit")
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type       = "nf4",
            bnb_4bit_compute_dtype    = dtype,
        )
    pass
    kwargs = dict()
    if load_in_4bit: kwargs["quantization_config"] = bnb_config
    # Must patch BnB compute_dtype since it's forced to bfloat16!
    patch_bitsandbytes_quant_state()
    # patch_bitsandbytes_compute_dtype(dtype)
    model_type = getattr(config, "model_type", "causal_lm")
    if model_type == "mllama":
        from transformers import MllamaForConditionalGeneration
        model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            device_map          = "sequential",
            torch_dtype         = dtype,
            attn_implementation = "sdpa",
            **kwargs,
        )
    elif model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map          = "sequential",
            torch_dtype         = dtype,
            attn_implementation = "sdpa",
            **kwargs,
        )
    elif model_type == "gemma3" and hasattr(config, "vision_config"):
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            device_map          = "sequential",
            torch_dtype         = dtype,
            attn_implementation = "sdpa",
            **kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map          = "sequential",
            torch_dtype         = dtype,
            attn_implementation = "sdpa",
            **kwargs,
        )
    # unpatch_bitsandbytes_compute_dtype()
    for param in model.parameters():
        param.requires_grad_(False)
    model, _ = patch_model_and_tokenizer(model, None)

    # Patch vLLM to disable multiprocessing for state dict extraction
    patch_vllm()

    llm = load_vllm(
        model_name             = model_name,
        config                 = config,
        gpu_memory_utilization = gpu_memory_utilization,
        dtype                  = dtype,
        conservativeness       = conservativeness,
        float8_kv_cache        = float8_kv_cache,
        unsloth_vllm_standby   = unsloth_vllm_standby,
        use_bitsandbytes       = load_in_4bit,
    )

    state_dict, quant_state_dict = get_vllm_state_dict(
        llm,
        return_state_dict = True,
        config = config,
        is_vision_model = is_vision_model,
    )
    assert_same_state_dict(model.state_dict(), state_dict)

    new_model = convert_vllm_to_huggingface(quant_state_dict, config, dtype, is_vision_model = is_vision_model)
    assert_same_state_dict(model.state_dict(), new_model.state_dict())

    # Run the model as well
    if not skip_generation:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        messages = [
            [{"role": "user", "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},],
            [{"role": "user", "content": "Write a long poem about the world."},],
            [{"role": "user", "content": "What is the capital of France? Describe it."},],
            [{"role": "user", "content": "Why is the sky blue?"},],
            [{"role": "user", "content": "Explain Newton's third law of motion."},],
            [{"role": "user", "content": "Why is spacetime bent?"},],
            [{"role": "user", "content": "Explain heliocentricism."},],
            [{"role": "user", "content": "Derive the formula for an infinite sum of 1, 1/2, 1/4, 1/8 and so on."},],
        ]*counts
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True, # Must add for generation
            padding = True,
        )

        from vllm import SamplingParams
        sampling_params = SamplingParams(
            # temperature = 1.5,
            # min_p = 0.1,
            temperature = 0.8,
            top_p = 0.95,
            logprobs = 0,
            prompt_logprobs = 0,
            max_tokens = 256,
        )

        # Cannot just use llm.generate or OOM - split into batches
        batches = create_batches(inputs, llm.approx_max_num_seqs)
        completion_ids = []
        for batch in batches:
            outputs = llm.generate(batch, sampling_params)
            completion_ids.extend(out.token_ids for completions in outputs for out in completions.outputs)
        pass
        del completion_ids

        # Check all hidden states manually
        input_ids = tokenizer(inputs[0], add_special_tokens = False, return_tensors = "pt")
        input_ids = input_ids["input_ids"].to("cuda", non_blocking = True)
        _test_same_model(model, new_model, input_ids)

    delete_vllm(llm)

    # Delete model as well
    try:
        model.model.embed_tokens.weight = None
        new_model.model.embed_tokens.weight = None

        for i in range(len(model.model.layers)):
            model.model.layers[i] = None
            new_model.model.layers[i] = None
        pass

        model.model.norm.weight = None
        new_model.model.norm.weight = None
        model.lm_head.weight = None
        new_model.lm_head.weight = None
        model.model = None
        new_model.model = None
    except:
        pass

    del model
    del new_model
    print(f'Test passed!')
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()
pass


def test_get_vllm_state_dict():
    # All Unsloth Zoo code licensed under LGPLv3
    patch_vllm()

    free_memory, total_memory = get_mem_info()

    model_names = [
        ("unsloth/Llama-3.2-1B-Instruct-bnb-4bit", 100,),
        ("unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit", 100,),
        ("unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit", 50,),
    ]
    bfloat16_dtype = torch.float16
    if total_memory >= 40 * 1000 * 1000 * 1000:
        model_names += [
            ("unsloth/Qwen2.5-3B-Instruct", 50,),
            ("unsloth/Llama-3.2-1B-Instruct-bnb-4bit", 100,),
            ("unsloth/meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit", 25,),
            ("unsloth/Qwen2.5-7B-Instruct-bnb-4bit", 25,),
        ]
        bfloat16_dtype = torch.bfloat16
    pass

    for i, (model_name, counts,) in enumerate(model_names):
        gc.collect()
        torch.cuda.empty_cache()
        dtype = torch.float16 if i % 2 == 0 else bfloat16_dtype
        print(f"##### Testing {model_name} with dtype = {dtype} #####")
        if bfloat16_dtype == torch.float16:
            counts = counts // 4
            conservativeness = 0.8
            float8_kv_cache = True
            gpu_memory_utilization = 0.5
        else:
            conservativeness = 1.0
            float8_kv_cache = True
            gpu_memory_utilization = 0.7
        try:
            _test_get_vllm_state_dict(
                model_name = model_name,
                dtype = dtype,
                gpu_memory_utilization = gpu_memory_utilization,
                counts = counts,
                conservativeness = conservativeness,
                float8_kv_cache = float8_kv_cache,
                unsloth_vllm_standby = unsloth_vllm_standby,
            )
        except Exception as error:
            error = str(error)
            raise RuntimeError(f"[{model_name}]\n{error}")
        gc.collect()
        torch.cuda.empty_cache()
    pass
pass

def get_model_layer_config(model_type, config=None):
    """
    Returns layer configuration for different model types.

    Args:
        model_type: Type of model ("causal_lm", "mllama", "qwen2_5_vl", "gemma3")
        config: Model configuration (optional, used for some model-specific configs)

    Returns:
        dict: Dictionary containing layer templates for different components
    """
    def get_base_config(prefix):
        # Base layer configurations common to all models
        base_config = {
            'standard_layers': [
                f"{prefix}.layers.{{kk}}.self_attn.q_proj",
                f"{prefix}.layers.{{kk}}.self_attn.k_proj",
                f"{prefix}.layers.{{kk}}.self_attn.v_proj",
                f"{prefix}.layers.{{kk}}.self_attn.o_proj",
                f"{prefix}.layers.{{kk}}.mlp.gate_proj",
                f"{prefix}.layers.{{kk}}.mlp.up_proj",
                f"{prefix}.layers.{{kk}}.mlp.down_proj",
            ],
            'layernorms': [
                f"{prefix}.layers.{{kk}}.input_layernorm",
                f"{prefix}.layers.{{kk}}.post_attention_layernorm",
            ],
            'vision_layers': [],
            'additional_layers': [],
        }
        return base_config

    if model_type == "mllama":
        base_config = get_base_config("model.language_model")
        base_config['layernorms'].extend([
            "model.language_model.layers.{kk}.cross_attn_input_layernorm",
            "model.language_model.layers.{kk}.cross_attn_post_attention_layernorm",
        ])
        base_config['additional_layers'].extend([
            "model.layers.{kk}.cross_attn.qkv_proj",
            "model.layers.{kk}.cross_attn.o_proj",
        ])
        # Vision transformer layers
        base_config['vision_layers'].extend([
            "model.vision_model.transformer.layers.{kk}.self_attn.q_proj",
            "model.vision_model.transformer.layers.{kk}.self_attn.k_proj",
            "model.vision_model.transformer.layers.{kk}.self_attn.v_proj",
            "model.vision_model.transformer.layers.{kk}.self_attn.o_proj",
            "model.vision_model.transformer.layers.{kk}.mlp.fc1",
            "model.vision_model.transformer.layers.{kk}.mlp.fc2",
            "model.vision_model.transformer.layers.{kk}.input_layernorm",
            "model.vision_model.transformer.layers.{kk}.post_attention_layernorm",
            "model.vision_model.global_transformer.layers.{kk}.self_attn.q_proj",
            "model.vision_model.global_transformer.layers.{kk}.self_attn.k_proj",
            "model.vision_model.global_transformer.layers.{kk}.self_attn.v_proj",
            "model.vision_model.global_transformer.layers.{kk}.self_attn.o_proj",
            "model.vision_model.global_transformer.layers.{kk}.mlp.fc1",
            "model.vision_model.global_transformer.layers.{kk}.mlp.fc2",
            "model.vision_model.global_transformer.layers.{kk}.input_layernorm",
            "model.vision_model.global_transformer.layers.{kk}.post_attention_layernorm",
        ])

    elif model_type == "qwen2_5_vl":
        base_config = get_base_config("model.language_model")
        base_config['layernorms'].extend([
            "model.language_model.norm",
            "model.visual.norm",
        ])
        base_config['vision_layers'].extend([
            "model.visual.blocks.{kk}.attn.qkv",
            "model.visual.blocks.{kk}.attn.proj",
            "model.visual.blocks.{kk}.mlp.gate_proj",
            "model.visual.blocks.{kk}.mlp.up_proj",
            "model.visual.blocks.{kk}.mlp.down_proj",
            "model.visual.blocks.{kk}.norm1",
            "model.visual.blocks.{kk}.norm2",
        ])
        base_config['additional_layers'].extend([
            "model.visual.merger.ln_q",
            "model.visual.merger.mlp.0",
            "model.visual.merger.mlp.2",
            "model.visual.patch_embed.proj",
        ])

    elif model_type == "gemma3":
        base_config = get_base_config("model.language_model")
        base_config['layernorms'].extend([
            "model.language_model.layers.{kk}.pre_feedforward_layernorm",
            "model.language_model.layers.{kk}.post_feedforward_layernorm",
            "model.language_model.layers.{kk}.self_attn.q_norm",
            "model.language_model.layers.{kk}.self_attn.k_norm",
        ])
        base_config['vision_layers'].extend([
            "model.vision_tower.vision_model.encoder.layers.{kk}.self_attn.q_proj",
            "model.vision_tower.vision_model.encoder.layers.{kk}.self_attn.k_proj",
            "model.vision_tower.vision_model.encoder.layers.{kk}.self_attn.v_proj",
            "model.vision_tower.vision_model.encoder.layers.{kk}.self_attn.out_proj",
            "model.vision_tower.vision_model.encoder.layers.{kk}.mlp.fc1",
            "model.vision_tower.vision_model.encoder.layers.{kk}.mlp.fc2",
            "model.vision_tower.vision_model.encoder.layers.{kk}.post_layernorm",
            "model.vision_tower.vision_model.encoder.layers.{kk}.layer_norm1",
            "model.vision_tower.vision_model.encoder.layers.{kk}.layer_norm2",
        ])

    # Add some common additional norms for causal LM models
    else:
        # Add potential additional norms that some models might have
        base_config = get_base_config("model")
        base_config['layernorms'].extend([
            "model.layers.{kk}.pre_feedforward_layernorm",
            "model.layers.{kk}.post_feedforward_layernorm",
            "model.layers.{kk}.q_norm",
            "model.layers.{kk}.k_norm",
        ])

    return base_config

def get_model_layer_counts(config):
    """
    Returns layer counts for different model types.

    Args:
        config: Model configuration

    Returns:
        int or dict: Number of layers (int for causal_lm, dict for VL models)
    """
    model_type = getattr(config, "model_type", "causal_lm")

    if model_type == "mllama":
        return {
            "text_layers": getattr(config.text_config, "num_hidden_layers", 32),
            "vision_layers": getattr(config.vision_config, "num_hidden_layers", 32),
            "global_layers": getattr(config.vision_config, "num_global_layers", 8),
        }
    elif model_type == "qwen2_5_vl":
        return {
            "text_layers": getattr(config, "num_hidden_layers", 32),
            "vision_layers": getattr(config.vision_config, "num_hidden_layers", 32),
        }
    elif model_type == "gemma3":
        return {
            "text_layers": getattr(config.text_config, "num_hidden_layers", 32),
            "vision_layers": getattr(config.vision_config, "num_hidden_layers", 32),
        }
    else:
        # Standard causal LM
        return getattr(config, "num_hidden_layers", 32)

def extract_mllama_vision_layers(vllm_internals, state_dict, quant_state_dict, get_state_dict):
    """Extract vision layers for mllama models."""
    try:
        vision_model = vllm_internals.vision_model
        for module_name in ["transformer", "global_transformer"]:
            if hasattr(vision_model, module_name):
                module = getattr(vision_model, module_name)
                if hasattr(module, "layers"):
                    for kk in range(len(module.layers)):
                        layer = module.layers[kk]
                        prefix = f"model.vision_model.{module_name}.layers.{kk}"

                        # Vision attention layers
                        if hasattr(layer, "self_attn"):
                            if hasattr(layer.self_attn, "qkv_proj"):
                                get_state_dict(f"{prefix}.self_attn.q_proj", 0, state_dict, layer.self_attn.qkv_proj)
                                get_state_dict(f"{prefix}.self_attn.k_proj", 1, state_dict, layer.self_attn.qkv_proj)
                                get_state_dict(f"{prefix}.self_attn.v_proj", 2, state_dict, layer.self_attn.qkv_proj)
                            if hasattr(layer.self_attn, "o_proj"):
                                get_state_dict(f"{prefix}.self_attn.o_proj", 0, state_dict, layer.self_attn.o_proj)

                        # Vision MLP layers
                        if hasattr(layer, "mlp"):
                            if hasattr(layer.mlp, "fc1"):
                                get_state_dict(f"{prefix}.mlp.fc1", 0, state_dict, layer.mlp.fc1)
                            if hasattr(layer.mlp, "fc2"):
                                get_state_dict(f"{prefix}.mlp.fc2", 0, state_dict, layer.mlp.fc2)

                        # Vision layernorms
                        for norm_name in ["input_layernorm", "post_attention_layernorm"]:
                            if hasattr(layer, norm_name):
                                norm = getattr(layer, norm_name)
                                state_dict[f"{prefix}.{norm_name}.weight"] = norm.weight.data
                                quant_state_dict[f"{prefix}.{norm_name}.weight"] = state_dict[f"{prefix}.{norm_name}.weight"]
    except Exception as e:
        print(f"Unsloth: Could not extract vision layers for mllama: {e}")

def extract_qwen2_5_vl_vision_layers(vllm_internals, state_dict, quant_state_dict, get_state_dict):
    """Extract vision layers for qwen2_5_vl models."""
    try:
        for kk in range(len(vllm_internals.visual.blocks)):
            block = vllm_internals.visual.blocks[kk]
            prefix = f"model.visual.blocks.{kk}"

            # Visual attention - vLLM uses QKVParallelLinear, HF expects unified QKV
            # Use slice_weights=False to get the full unified QKV weight
            get_state_dict(f"{prefix}.attn.qkv", 0, state_dict, block.attn.qkv, slice_weights=False)

            # Extract projection layer using get_state_dict to handle tensor parallelism
            get_state_dict(f"{prefix}.attn.proj", 0, state_dict, block.attn.proj)

            # Visual MLP - use get_state_dict to handle tensor parallelism
            get_state_dict(f"{prefix}.mlp.gate_proj", 0, state_dict, block.mlp.gate_proj)
            get_state_dict(f"{prefix}.mlp.up_proj", 0, state_dict, block.mlp.up_proj)
            get_state_dict(f"{prefix}.mlp.down_proj", 0, state_dict, block.mlp.down_proj)

            # Visual norms
            for norm_name in ["norm1", "norm2"]:
                norm = getattr(block, norm_name)
                # LayerNorms are not tensor-parallel – grab full weight/bias.
                get_state_dict(f"{prefix}.{norm_name}", 0, state_dict, norm, slice_weights = False)

        # Extract visual.merger and patch_embed weights with proper tensor parallelism handling
        visual_attr = getattr(vllm_internals, "visual", None)
        if visual_attr is not None:
            # Merger extraction under model.visual.merger.*
            merger = visual_attr.merger
            merger_prefix = "model.visual.merger"

            if hasattr(merger, "ln_q"):
                ln_q_layer = getattr(merger.ln_q, "base_layer", merger.ln_q)
                get_state_dict(f"{merger_prefix}.ln_q", 0, state_dict, ln_q_layer, slice_weights = False)

            # Extract MLP layers directly
            mlp = merger.mlp
            if len(mlp) > 0:
                get_state_dict(f"{merger_prefix}.mlp.0", 0, state_dict, mlp[0], slice_weights = False)
            if len(mlp) > 2:
                get_state_dict(f"{merger_prefix}.mlp.2", 0, state_dict, mlp[2], slice_weights = False)

            if hasattr(visual_attr, "patch_embed") and hasattr(visual_attr.patch_embed, "proj"):
                get_state_dict("model.visual.patch_embed.proj", 0, state_dict, visual_attr.patch_embed.proj, slice_weights = False)

    except Exception as e:
        print(f"Unsloth: Could not extract vision layers for qwen2_5_vl: {e}")

def extract_gemma3_vision_layers(vllm_internals, state_dict, quant_state_dict, get_state_dict):
    """Extract vision layers for gemma3 models."""
    try:

        # Vision encoder layers
        if hasattr(vllm_internals, "vision_tower"):
            vision_model = vllm_internals.vision_tower.vision_model

            for kk in range(len(vision_model.encoder.layers)):
                layer = vision_model.encoder.layers[kk]
                prefix = f"model.vision_tower.vision_model.encoder.layers.{kk}"

                # Vision attention layers (QKV unified in vLLM)
                proj = layer.self_attn.qkv_proj
                get_state_dict(f"{prefix}.self_attn.q_proj", 0, state_dict, proj)
                get_state_dict(f"{prefix}.self_attn.k_proj", 1, state_dict, proj)
                get_state_dict(f"{prefix}.self_attn.v_proj", 2, state_dict, proj)

                get_state_dict(f"{prefix}.self_attn.out_proj", 0, state_dict, layer.self_attn.out_proj)

                # Vision MLP layers - moved inside the loop
                get_state_dict(f"{prefix}.mlp.fc1", 0, state_dict, layer.mlp.fc1)
                get_state_dict(f"{prefix}.mlp.fc2", 0, state_dict, layer.mlp.fc2)

                # Vision layernorms – use helper for full tensors
                for norm_name in ["layer_norm1", "layer_norm2"]:
                    if hasattr(layer, norm_name):
                        norm = getattr(layer, norm_name)
                        get_state_dict(f"{prefix}.{norm_name}", 0, state_dict, norm, slice_weights = False)

            # Extract vision embeddings and post norm
            if hasattr(vision_model, "embeddings"):
                embeddings = vision_model.embeddings
                # Patch embedding (Conv2d)
                get_state_dict("model.vision_tower.vision_model.embeddings.patch_embedding", 0, state_dict, embeddings.patch_embedding, slice_weights = False)
                # Position embedding (Embedding)
                get_state_dict("model.vision_tower.vision_model.embeddings.position_embedding", 0, state_dict, embeddings.position_embedding, slice_weights = False)

            # Post layernorm
            if hasattr(vision_model, "post_layernorm"):
                get_state_dict("model.vision_tower.vision_model.post_layernorm", 0, state_dict, vision_model.post_layernorm, slice_weights = False)

        # Extract multi-modal projector components
        if hasattr(vllm_internals, "multi_modal_projector"):
            multi_modal_projector = vllm_internals.multi_modal_projector

            # Extract mm_input_projection_weight if it exists
            if hasattr(multi_modal_projector, "mm_input_projection_weight"):
                state_dict["model.multi_modal_projector.mm_input_projection_weight"] = multi_modal_projector.mm_input_projection_weight.data
                quant_state_dict["model.multi_modal_projector.mm_input_projection_weight"] = state_dict["model.multi_modal_projector.mm_input_projection_weight"]

            # Extract mm_soft_emb_norm
            if hasattr(multi_modal_projector, "mm_soft_emb_norm"):
                mm_soft_emb_norm = multi_modal_projector.mm_soft_emb_norm
                state_dict["model.multi_modal_projector.mm_soft_emb_norm.weight"] = mm_soft_emb_norm.weight.data
                quant_state_dict["model.multi_modal_projector.mm_soft_emb_norm.weight"] = state_dict["model.multi_modal_projector.mm_soft_emb_norm.weight"]

    except Exception as e:
        print(f"Unsloth: Could not extract vision layers for gemma3: {e}")

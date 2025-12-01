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
    "get_lora_supported_ranks",
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
import ast
import sys
import torch
from torch import __version__ as torch_version
import json
import psutil
import functools
import contextlib
import inspect
from functools import partial
from .utils import _get_dtype, get_quant_type, Version
from .empty_model import *
from .hf_utils import (
    dtype_from_config,
    add_dtype_kwargs,
    set_dtype_in_config,
)
from .patching_utils import patch_model_and_tokenizer
from .temporary_patches.common import (
    get_torch_compile_options,
    UNSLOTH_ENABLE_LOGGING,
)
from .log import logger
from .device_type import DEVICE_TYPE
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
    if DEVICE_TYPE == "hip":
        return torch.device("cuda", index)
    return torch.device(DEVICE_TYPE, index)

def get_mem_info():
    if DEVICE_TYPE == "xpu":
        free_memory, total_memory = torch.xpu.mem_get_info()
    else:
        free_memory, total_memory = torch.cuda.mem_get_info()
    return free_memory, total_memory
pass

if importlib.util.find_spec("vllm") is not None:
    from vllm import __version__ as vllm_version

    # Patch excessive warning messages
    if not UNSLOTH_ENABLE_LOGGING:
        # Disable all not supported messages
        # Regarding multimodal models, vLLM currently only supports adding LoRA to language model.
        try:
            from vllm.worker.model_runner import logger as vllm_logger
            vllm_logger.addFilter(HideLoggingMessage("only supports adding LoRA"))
            del vllm_logger
        except:
            pass
        try:
            from vllm.v1.worker.lora_model_runner_mixin import logger as vllm_logger
            vllm_logger.addFilter(HideLoggingMessage("only supports adding LoRA"))
            del vllm_logger
        except:
            pass
    pass

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

    # Since https://github.com/vllm-project/vllm/blob/4959915089f1bcf011f082136464e48b76c7e3d9/vllm/model_executor/model_loader/bitsandbytes_loader.py
    # vLLM dequantizes the Double quant scalars on the fly
    # We disable this
    def dequantize_dq(quant_states):
        return quant_states
    def _dequantize_dq(self, quant_states):
        return quant_states
    try:
        import vllm.model_executor.model_loader.bitsandbytes_loader
        if hasattr(
            vllm.model_executor.model_loader.bitsandbytes_loader,
            "dequantize_dq",
        ):
            vllm.model_executor.model_loader.bitsandbytes_loader.dequantize_dq = dequantize_dq
        elif hasattr(
            vllm.model_executor.model_loader.bitsandbytes_loader.BitsAndBytesModelLoader,
            "_dequantize_dq",
        ):
            vllm.model_executor.model_loader.bitsandbytes_loader.BitsAndBytesModelLoader._dequantize_dq = _dequantize_dq
        pass
    except:
        pass

    # Patch apply_bnb_4bit
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
        try:
            from vllm.config import logger as vllm_config_logger
        except:
            # vLLM refactored a lot of configs. Most of them are backwards compatible for imports. This seems to not be.
            from vllm.config.model import logger as vllm_config_logger
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
        try:
            import vllm.v1.worker.lora_model_runner_mixin
            vllm.v1.worker.lora_model_runner_mixin.LRUCacheWorkerLoRAManager = PatchedLRUCacheWorkerLoRAManager
        except:
            pass
        if os.getenv("UNSLOTH_DO_NOT_PATCH_V0_LRU_LORA_MANAGER", "0") == "1":
            return
        try:
            import vllm.worker.model_runner
            vllm.worker.model_runner.LRUCacheWorkerLoRAManager = PatchedLRUCacheWorkerLoRAManager
        except:
            pass
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
    from vllm.device_allocator.cumem import CuMemAllocator, libcudart, unmap_and_release, create_and_map, AllocationData
    try:
        from vllm.utils import is_pin_memory_available
    except:
        # in some newer versions, this is not available in vllm.utils
        from vllm.utils.platform_utils import is_pin_memory_available
    from typing import Optional, Union, Tuple, Any

    logger.info(f"Unsloth: Enabling vLLM standby mode")

    def __init__(self):
        # This is a replica of the original CuMemAllocator.__init__()
        # with no changes except modification to error message for better readability
        for check in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF",):
            conf = os.environ.get(check, "")
            assert "expandable_segments:True" not in conf, \
                ("Standby mode is not supported with expandable segments.\n"
                f"Please set environment variable {check} without `expandable_segments:True`.\n"
                )

        self.pointer_to_data: dict[int, AllocationData] = {}
        self.current_tag: str = CuMemAllocator.default_tag
        self.allocator_and_pools: dict[str, Any] = {}
        if hasattr(self, '_python_malloc_callback'):
            # vllm changed something recently wrt cumem init
            # new versions have function _python_malloc/free and set it to self.python_malloc/free
            # old versions just have the function self.python_malloc/free so they need no such assignment
            # this check is to make sure it works for both new versions and old alike
            # https://github.com/vllm-project/vllm/commit/9dc30b7068ae07ceca89663e9f8403d00217256d
            self.python_malloc_callback = self._python_malloc_callback
        if hasattr(self, '_python_free_callback'):
            self.python_free_callback = self._python_free_callback

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
        pass

        logger.debug(f'CPU offloads {cpu_offloads} true offloads {true_offloads} total {total_offloads}')
        gc.collect()
        torch.cuda.empty_cache()
    pass

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
            pass
        pass
    pass

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
        # print(f"Total weights memory: {weights_total / 1e9:.2f} GB for {weights_count} items")
        # print(f"Total KVCache memory: {kv_cache_total / 1e9:.2f} GB for {kv_cache_count} items")
    pass

    def get_patched_generate(original_generate):
        def check_sleep_mode(self):
            # LLM object has llm_engine as an attribute
            engine = getattr(self, "llm_engine", self)
            return hasattr(engine, "vllm_config") and hasattr(engine.vllm_config, "model_config") and getattr(engine.vllm_config.model_config, "enable_sleep_mode", False)

        import functools
        @functools.wraps(original_generate)
        def new_generate(self, *args, **kwargs):
            # vLLM internally checks if wake_up is necessary before performing memory allocation.
            if check_sleep_mode(self):
                self.wake_up()
            return original_generate(self,*args, **kwargs)
        return new_generate
    pass

    vllm.LLM.generate = get_patched_generate(vllm.LLM.generate)
    vllm.AsyncLLMEngine.generate = get_patched_generate(vllm.AsyncLLMEngine.generate)

    CuMemAllocator.__init__ = __init__
    CuMemAllocator.sleep = sleep
    CuMemAllocator.wake_up = wake_up
    CuMemAllocator.print_memory_summary = print_memory_summary
pass


def patch_vllm_graph_capture():
    """
    Temporarily disable ``gc.collect`` to speed up CUDA graph capture.
    This is a workaround to avoid the overhead of garbage collection
    during the graph capture with torch.compile.
    """
    from contextlib import contextmanager
    import gc
    import time
    from functools import wraps

    @contextmanager
    def suppress_gc_collect():
        original_gc_collect = gc.collect
        gc.collect = lambda: None
        try:
            yield
        finally:
            gc.collect = original_gc_collect
    pass

    # Patch vLLM v1
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner, logger
        logger.info('Unsloth: Patching vLLM v1 graph capture')
        original_capture_model_v1 = GPUModelRunner.capture_model

        @wraps(original_capture_model_v1)
        def capture_model_wrapper_v1(self, *args, **kwargs):
            logger.info("Unsloth: Running patched vLLM v1 `capture_model`.")
            start_time = time.perf_counter()

            with suppress_gc_collect():
                result = original_capture_model_v1(self, *args, **kwargs)

            end_time = time.perf_counter()
            logger.info(
                "Unsloth: Patched vLLM v1 graph capture finished in %.0f secs.",
                end_time - start_time
            )
            for _ in range(2):
                gc.collect()
                torch.cuda.empty_cache()
            return result
        pass
        GPUModelRunner.capture_model = capture_model_wrapper_v1
    except Exception as e:
        print(f"Unsloth: Could not patch vLLM V1 graph capture: {e}")

    from packaging.utils import Version
    if Version(vllm.__version__) < Version("0.11.0"):
        # Also patch vLLM v0. vLLM v0 is deprecated in vLLM v0.11.0 so only do when appropriate.
        try:
            from vllm.worker.model_runner import GPUModelRunnerBase, logger
            logger.info('Unsloth: Patching vLLM v0 graph capture')
            original_capture_model_v0 = GPUModelRunnerBase.capture_model

            @wraps(original_capture_model_v0)
            def capture_model_wrapper_v0(self, *args, **kwargs):
                logger.info("Unsloth: Running patched vLLM v0 `capture_model`.")
                start_time = time.perf_counter()

                with suppress_gc_collect():
                    result = original_capture_model_v0(self, *args, **kwargs)

                end_time = time.perf_counter()
                logger.info(
                    "Unsloth: Patched vLLM v0 graph capture finished in %.0f secs.",
                    end_time - start_time
                )
                for _ in range(2):
                    gc.collect()
                    torch.cuda.empty_cache()
                return result
            pass
            GPUModelRunnerBase.capture_model = capture_model_wrapper_v0
        except Exception as e:
            print(f"Unsloth: Could not patch vLLM V0 graph capture: {e}")
pass


def patch_vllm(debug = True):
    # Temporary patch to disable multiprocessing for vLLM
    # Allows accessing model_executor
    logger.info(f'Unsloth: Patching vLLM')
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if debug or os.getenv("UNSLOTH_ENABLE_LOGGING", "0") == "1":
        os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
    # os.environ["VLLM_TRACE_FUNCTION"] = "1"
    patch_vllm_set_inductor_config()
    patch_bitsandbytes_quant_state()
    patch_vllm_bitsandbytes()
    patch_vllm_lora_tokenizer()
    patch_vllm_lora_load_tensors()
    if os.getenv("UNSLOTH_VLLM_STANDBY", "0") == "1":
        logger.info(f'Unsloth: Patching vLLM to enable standby.')
        patch_vllm_enable_sleep_mode()
    patch_vllm_graph_capture()
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


def get_vllm_state_dict(llm, return_state_dict = False, config = None, is_vision_model = False):
    # If the vllm state dict was quantized using torchao, we will run into
    # the following error when calling ops like aten.t() in inference mode.
    # This is a bug in PyTorch that affects all tensor subclasses.
    #
    #     Cannot set version_counter for inference tensor
    #
    # For now, we work around this issue by using torch.no_grad in this case.
    # See https://github.com/pytorch/pytorch/issues/164872 for more details
    if get_quant_type(config) == "torchao":
        ctx_manager = torch.no_grad()
    else:
        ctx_manager = torch.inference_mode()
    with ctx_manager:
        return _get_vllm_state_dict(llm, return_state_dict, config, is_vision_model)


def _get_vllm_state_dict(llm, return_state_dict = False, config = None, is_vision_model = False):
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
    except:
        # Using a new VLLM version must use collective_rpc
        try:
            vllm_state_dict = {}
            gpu_ids = llm.collective_rpc("report_device_id", args = tuple())
            weights = llm.collective_rpc("get_weight_ipc_handles", args = tuple())[0]
            weights = weights[gpu_ids[0]]
            for weight_name, (to_cuda_fx, cuda_data,) in weights.items():
                vllm_state_dict[weight_name] = to_cuda_fx(*cuda_data)
            pass
            raise NotImplementedError("Unsloth: Currently vLLM RPC is not yet fully enabled!")
        except Exception as e:
            raise RuntimeError(f"Unsloth: Cannot get internal vLLM states with error = {str(e)}")
    pass

    assert(config is not None)

    # Determine model type from config BEFORE reassigning config
    model_type = getattr(config, "model_type", "causal_lm")

    # Keep the original config for model_type but use text_config for vocab_size etc
    text_config = getattr(config, "text_config", config)

    vocab_size = text_config.vocab_size

    state_dict = OrderedDict()
    quant_state_dict = OrderedDict()

    capability = torch.cuda.get_device_capability()
    sm_cap = capability[0] * 10 + capability[1]


    try:
        # vLLM recently removed the transpose of weight scale for Hopper GPUs.
        # https://github.com/vllm-project/vllm/pull/28431
        # So now we check if the weight process function does a transpose of weight scale before doing so
        # https://github.com/vllm-project/vllm/commit/f9a4087182ffcd9404779fcda876f820b3b26d5f#diff-cce58c0ceb6a9b15a01f117d734b93736acc25ed89921c2eacc58ea05bd34d0eL1155-L1157
        from vllm.model_executor.layers.quantization.utils.fp8_utils import maybe_post_process_fp8_weight_block
        from inspect import getsource
        needs_transpose_check = 'layer.weight_scale.data.T.contiguous()' in getsource(maybe_post_process_fp8_weight_block)
    except Exception as e:
        logger.info(f"Unsloth: Could not import vLLM fp8_utils: {e}")
        needs_transpose_check = False

    is_deep_gemm_supported = False
    cutlass_block_fp8_supported = False
    if needs_transpose_check:
        # Only try to import and check if we need to
        try:
            from vllm.utils.deep_gemm import is_deep_gemm_supported as vllm_is_deep_gemm_supported
            is_deep_gemm_supported = vllm_is_deep_gemm_supported()
        except Exception as e:
            logger.info(f"Unsloth: Could not import vLLM deep_gemm: {e}")

        try:
            cutlass_block_fp8_supported = torch.ops._C.cutlass_scaled_mm_supports_block_fp8(sm_cap)
        except Exception as e:
            logger.info(f"Unsloth: Could not import vLLM cutlass_block_fp8_supported: {e}")
        pass

    def get_state_dict(prefix, kk, state_dict, proj, slice_weights=True, slice_index=-1):
        proj = getattr(proj, "base_layer", proj)
        qweight = proj.weight

        # Determine slicing offsets
        output_sizes = getattr(proj, "output_sizes", None)
        if output_sizes is not None:
            dim_offsets = np.cumsum([0] + output_sizes)
        else:
            dim_offsets = [0, qweight.shape[0]]

        ## Handle FP8 weights. For now only BlockQuantized
        if qweight.dtype == torch.float8_e4m3fn:
            if hasattr(proj, 'weight_scale'):
                weight_scale = proj.weight_scale
            elif hasattr(proj, 'weight_scale_inv'):
                weight_scale = proj.weight_scale_inv
            else:
                raise ValueError(f"Unsloth: Cannot find weight scale for FP8 weight {prefix}")

            offsets = [0] + proj.logical_widths # [q, k, v] sizes
            offsets = np.cumsum(offsets)
            scale_suffix = '.weight_scale'
            if weight_scale.ndim == 2:
                if weight_scale.shape[1] > 1:
                    # Block quantized has 2D weight scale
                    # for qwen 3 for eg, 4096 query and 1024 each for k and v. Weight block size say is [128, 128]
                    # so the shape of qkv is [6144, 4096] and scale.T is [48, 32]. Now 48 should be split into [0, 32, 40, 48]
                    # Also notice that vLLM stores scale in [32,48] which is transpose of what HF expects.
                    scale_suffix = '.weight_scale_inv'
                    block_size = proj.weight_block_size[0]
                    is_compressed_linear = "CompressedTensors" in str(type(getattr(proj, 'quant_method', None)))
                    if is_compressed_linear:
                        # Compressed linear doesn't seem to transpose the weight scale inv
                        # Also preferes the name weight_scale (without _inv suffix)
                        # We detect it based on the quant_method we see in proj's attributes
                        scale_suffix = '.weight_scale'
                    elif needs_transpose_check:
                        should_use_deepgemm = is_deep_gemm_supported and getattr(proj, "orig_dtype", torch.bfloat16) == torch.bfloat16 and qweight.shape[0] % 128 == 0 and qweight.shape[1] % 128 == 0
                        if sm_cap==90 and cutlass_block_fp8_supported and not should_use_deepgemm:
                            # For H100 (at least), the scale seems to be a transpose of what HF expects, while on L4 it is right shape.
                            # This is done by vLLM based on a few checks that we replicated above.
                            # https://github.com/vllm-project/vllm/blob/294c805f1df9ddf62c2290989710da9d48ab4973/vllm/model_executor/layers/quantization/utils/fp8_utils.py#L1172-L1179
                            weight_scale = weight_scale.T
                            logger.info(f"Unsloth: Transposed weight scale for {prefix} for weight shape {qweight.shape} and scale shape {weight_scale.shape}")
                    pass
                    a, b = qweight.shape
                    p, q = weight_scale.shape
                    # This is just a sanity check to ensure that we don't end up with wrongly sliced weight of shape [0, x] :)
                    assert a // p == proj.weight_block_size[0] and b // q == proj.weight_block_size[1], f"Unsloth: vLLM weight for {prefix} has unexpected weight shape {qweight.shape} and scale {weight_scale.shape} and block size {proj.weight_block_size}"
                else:
                    # This is dynamic quantization (aka per row or per column). The scale is of shape [n,1]
                    # The weight here is of shape [4096, 6144]. We need to transpose and then slice
                    qweight = qweight.T
                    block_size = 1

                scale_offsets = [x//block_size for x in offsets]
                if slice_weights:
                    weight_scale = weight_scale[scale_offsets[kk] : scale_offsets[kk + 1]]

            if slice_weights:
                weight = qweight[offsets[kk] : offsets[kk + 1]]
            else:
                weight = qweight

            state_dict[prefix + scale_suffix] = weight_scale
            quant_state_dict[prefix + scale_suffix] = weight_scale

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
        # for mllama, prefer using org_vocab_size which is text_config.vocab_size + 8
        # https://github.com/huggingface/transformers/blob/1cea763ba422b83778a8db0374ea90f43b09992b/src/transformers/models/mllama/modeling_mllama.py#L1147
        shrink_size = getattr(proj,"org_vocab_size", vocab_size)
        if shrink_size and ("embed_tokens" in prefix or "lm_head" in prefix):
            if weight.shape[0] > shrink_size:
                weight = weight[:shrink_size]

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
            if shrink_size is not None and ("embed_tokens" in prefix or "lm_head" in prefix):
                if bias_tensor.shape[0] > shrink_size:
                    bias_tensor = bias_tensor[:shrink_size]

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
    layer_config = get_model_layer_config()

    # All layers
    skipped_layernorms = []
    for kk in range(len(vllm_text_model.layers)):
        layer = vllm_text_model.layers[kk]
        if hasattr(layer, "self_attn"):
            prefix = f"{vllm_text_model_prefix}.layers.{kk}.self_attn"
            qkv_proj = layer.self_attn.qkv_proj
            o_proj = layer.self_attn.o_proj

            get_state_dict(f"{prefix}.q_proj", 0, state_dict, qkv_proj)
            get_state_dict(f"{prefix}.k_proj", 1, state_dict, qkv_proj)
            get_state_dict(f"{prefix}.v_proj", 2, state_dict, qkv_proj)
        elif hasattr(layer, "cross_attn"):
            prefix = f"{vllm_text_model_prefix}.layers.{kk}.cross_attn"
            qkv_proj = layer.cross_attn.qkv_proj
            o_proj = layer.cross_attn.o_proj
            name = re.sub(r"\.(\d+)\.", r"[\1].", prefix.replace('model.language_model','language_model.model', 1) + ".qkv_proj")
            cross_attn_layer = eval(f'vllm_internals.{name}')
            q_proj = cross_attn_layer.proj['q_proj_decoder']
            kv_proj = cross_attn_layer.proj['kv_proj_encoder']
            get_state_dict(f"{prefix}.q_proj", 0, state_dict, q_proj)
            get_state_dict(f"{prefix}.k_proj", 1, state_dict, kv_proj)
            get_state_dict(f"{prefix}.v_proj", 2, state_dict, kv_proj)

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
        extract_vision_layers(vllm_internals, state_dict, quant_state_dict, get_state_dict)
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
        missing_from_hf = new_state_dict.keys() - old_state_dict.keys()
        missing_from_vllm = old_state_dict.keys() - new_state_dict.keys()
        print(f'Unsloth: Failed comparing state_dict with Missing from hf: {missing_from_hf}\nMissing from vllm: {missing_from_vllm}')
        raise RuntimeError(f"Unsloth: Failed comparing state_dict with {difference}")
    pass

    failures = {}

    for key in old_state_dict:
        try:
            old_val = old_state_dict[key]
            new_val = new_state_dict[key]
            if old_val.dtype != new_val.dtype or (new_val.element_size() < 2):
                # upcast both to float32 just for comparison. For FP8, vLLM stores weight scale in FP32 while HF preferes 16bit
                old_val = old_val.to(torch.float32)
                new_val = new_val.to(torch.float32)
            torch.testing.assert_close(old_val, new_val, check_stride = False, atol = 1e-4, rtol = 1e-3)
        except Exception as error:
            if key == "lm_head.weight":
                # Try tied embeddings fallback
                key1 = next((k for k in (key, "model.embed_tokens.weight", "model.language_model.embed_tokens.weight") if k in old_state_dict), None)
                key2 = next((k for k in (key, "model.embed_tokens.weight", "model.language_model.embed_tokens.weight") if k in new_state_dict), None)

                if key1 is not None and key2 is not None:
                    try:
                        torch.testing.assert_close(old_state_dict[key1].contiguous(), new_state_dict[key2].contiguous(), check_stride = True)
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
def convert_vllm_to_huggingface(quant_state_dict, config, dtype = torch.float16, bnb_config = None, is_vision_model = False):
    # All Unsloth Zoo code licensed under LGPLv3
    # Unmerges vLLM modules to create HF compatible model
    set_dtype_in_config(config, dtype)
    new_model, original_meta_model, layer_count, layer_names = create_empty_model(config, dtype, is_vision_model)
    new_model = new_model.to(device = get_target_device(), dtype = dtype)
    quantization_config = getattr(config, "quantization_config", {})
    quant_method = get_quant_type(config)
    kwargs = dict()
    compute_dtype = dtype  # Do not use config file's dtype!

    if quantization_config != {} or bnb_config is not None:
        # Get quantization_config flags
        if quantization_config != {}:
            if quant_method == 'bitsandbytes':
                kwargs["compress_statistics"] = quantization_config["bnb_4bit_use_double_quant"]
                kwargs["quant_type"] = quantization_config["bnb_4bit_quant_type"]
                kwargs["quant_storage"] = _get_dtype(quantization_config["bnb_4bit_quant_storage"])
            elif quant_method == 'fp8':
                kwargs['activation_scheme'] = quantization_config['activation_scheme']
                kwargs['block_size'] = quantization_config['weight_block_size']
                try:
                    from transformers.integrations.finegrained_fp8 import FP8Linear # This has patched forward pass for LoRA and training support. Patched in unsloth/kernels/fp8.py
                except:
                    raise ImportError("Unsloth: FP8 models need importing FP8Linear from `transformers.integrations.finegrained_fp8` but we don't see it.")
            elif quant_method == 'fbgemm_fp8':
                kwargs['input_scale_ub'] = torch.tensor([quantization_config['activation_scale_ub']], device = get_target_device())
                try:
                    from transformers.integrations.fbgemm_fp8 import FbgemmFp8Linear # This has patched forward pass for LoRA and training support
                except:
                    raise ImportError("Unsloth: FP8 models need importing FbgemmFP8Linear from `transformers.integrations.fbgemm_fp8` but we don't see it.")
            elif quant_method == 'compressed-tensors':
                kwargs['activation_scheme'] = 'dynamic' # mark it dynamic for now
                block_size = [128, 128] # The default we override if we find in config
                config_groups = quantization_config.get('config_groups', None)
                group_0 = config_groups.get(0, None) if config_groups else None
                weights = group_0.get('weight', None) if group_0 else None
                block_size = weights.get('block_size', block_size) if weights else block_size
                kwargs['block_size'] = block_size
                try:
                    from transformers.integrations.finegrained_fp8 import FP8Linear # This has patched forward pass for LoRA and training support. Patched in unsloth/kernels/fp8.py
                except:
                    raise ImportError("Unsloth: FP8 models need importing FP8Linear from `transformers.integrations.finegrained_fp8` but we don't see it.")
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
            layer_name = layer_name.format(kk = kk)

            if 'language_model.model' in layer_name:
                # vLLM uses vllm_internals.language_model.model.layers while HF uses model.language_model.layers
                layer_name = layer_name.replace('language_model.model', 'language_model')

            is_weight = True
            if layer_name in quant_state_dict:
                # for attirbutes of type nn.Parameter, there's no .weight
                weight = quant_state_dict[layer_name]
                is_weight = False
            else:
                if f"{layer_name}.weight" not in quant_state_dict:
                    if "norm" in layer_name:
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

            # check if either of layer_name.weight_scale or layer_name.weight_scale_inv exists and set that attribute to fp8_weight_scale
            fp8_weight_scale = None
            if f"{layer_name}.weight_scale" in quant_state_dict:
                fp8_weight_scale = quant_state_dict[f"{layer_name}.weight_scale"]
            elif f"{layer_name}.weight_scale_inv" in quant_state_dict:
                fp8_weight_scale = quant_state_dict[f"{layer_name}.weight_scale_inv"]
            pass

            if fp8_weight_scale is not None: assert fp8_weight_scale.ndim in [1,2], f"we only support row quantized (ndim=1) and block quantized(ndim=2) fp8 but found {fp8_weight_scale.ndim}"

            if layer_name in quant_state_dict:
                # for attributes of type nn.Parameter, there's no .weight
                layer_name_br = re.sub(r"\.([\d]{1,})\.", r"[\1].", layer_name.replace('model.','',1))
                layer = torch.nn.Parameter(weight, requires_grad = False)
                exec(f"new_model.{layer_name_br} = layer")
                continue
            elif fp8_weight_scale is not None:
                if fp8_weight_scale.ndim == 1:
                    # This is FP8 quantized but not block quant. Either dynamic or static
                    layer = FbgemmFp8Linear(in_features = 0, out_features = 0, bias = has_bias, weight_dtype = dtype).to(get_target_device())
                    layer.in_features = weight.shape[1]
                    layer.out_features = weight.shape[0]
                    layer.weight = torch.nn.Parameter(weight, requires_grad = False)
                    layer.bias = bias
                    layer.input_scale_ub = kwargs['input_scale_ub']
                    layer.weight_scale = torch.nn.Parameter(fp8_weight_scale, requires_grad = False)
                    layer.weight.input_scale_ub = kwargs['input_scale_ub']
                    layer.quant_method = "fbgemm_fp8"
                elif fp8_weight_scale.ndim == 2:
                    # This denotes that the model if FP8 dynamic quantized.
                    layer = FP8Linear(in_features = 0, out_features = 0, bias = has_bias, dtype = dtype, block_size = kwargs['block_size'], device = get_target_device(), activation_scheme = kwargs['activation_scheme'])
                    layer.in_features = weight.shape[1]
                    layer.out_features = weight.shape[0]
                    layer.weight = torch.nn.Parameter(weight, requires_grad = False)
                    layer.bias = bias
                    layer.weight_scale_inv = torch.nn.Parameter(fp8_weight_scale, requires_grad = False)
                    layer.quant_method = "fp8"
            elif f"{layer_name}.weight.quant_state" in quant_state_dict:
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
                # from vllm 0.11.1, the .weight is of dtype ModelWeightParameter, so try to extract the 'data' part
                # https://github.com/vllm-project/vllm/commit/de94289a98d7ec52a5ef02719e01a1db8b505170#diff-7d6145ac4ba084231a441c2056c7fca23c3bae33e6542f4f602a6c9d4d2da64dL199-R208
                layer.weight = torch.nn.Parameter(getattr(weight, 'data', weight), requires_grad = False)
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
            layer_name = re.sub(r"\.([\d]{1,})", lambda x: f"[{x.group(1)}]", layer_name)
            exec(f"new_model.{layer_name} = layer")
        pass
    pass

    set_additional_modules(new_model, quant_state_dict, config)

    if original_meta_model is not None:
        copy_attributes(original_meta_model, new_model)

    # # Set config on model and modules using clean approach
    # new_model.config = config
    # for module in new_model.modules():
    #     if hasattr(module, "config"):
    #         module.config = config
    # for param in new_model.parameters():
    #     if hasattr(param, "config"):
    #         param.config = config

    text_config = getattr(config, "text_config", config) #try using text config for VLMs
    vision_config = getattr(config, "vision_config", None)
    # Fix up rotary_emb by re-initing them
    for module in new_model.modules():
        if hasattr(module, "rotary_emb"):
            module.rotary_emb = module.rotary_emb.__class__(
                config = text_config,
                device = get_target_device(),
            )
        if hasattr(module, "rotary_pos_emb"):
            # Qwen 2.5 VL has a rotary_pos_emb in vision submodel
            # https://github.com/huggingface/transformers/blob/a871f6f58d49f3a05ae9dae519caa8aa9d919a07/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L337
            assert vision_config is not None, "Unsloth: vision_config is required for models with vision rotary_pos_emb"
            head_dim = vision_config.hidden_size // vision_config.num_heads
            module.rotary_pos_emb = module.rotary_pos_emb.__class__(head_dim//2).to(get_target_device())
        if hasattr(module, "rotary_emb_local"):
            # gemma3 has a rotary_emb_local
            # https://github.com/huggingface/transformers/blob/008c0ba8e2a1226a6ef5a61c4915a0a8a340c157/src/transformers/models/gemma3/modeling_gemma3.py#L469-L471
            # Gemma3 uses different defaults for local and global RoPE. Copy the config for modification.
            local_rope_config = deepcopy(text_config)
            local_rope_config.rope_theta = text_config.rope_local_base_freq
            local_rope_config.rope_scaling = {"rope_type": "default"}
            # gemma3 has a rotary_emb_local
            module.rotary_emb_local = module.rotary_emb_local.__class__(
                config = local_rope_config,
                device = get_target_device(),
            )
            del local_rope_config
        pass
    pass

    # Must override or else Bitsandbytes will error
    new_model.to = partial(_override_to, new_model)
    new_model.eval()

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
    load_in_4bit = False,
    load_in_8bit = False,
    max_seq_length = 2048,
    gpu_memory_utilization = 0.8,
    enable_lora = True,
    max_lora_rank = 16,
    max_loras = 1,
    float8_kv_cache = False,
    account_for_gradients = True,
    parallel_sequences = 64,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Gets approximate max model length and max num sequences

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
    # factor = 16/5 if load_in_4bit else 1 # Should be 4.5 but use 5
    factor = 1
    if load_in_4bit: factor = 16/5
    elif load_in_8bit: factor = 8/5 # Very vague approximation. Will fix later
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


@functools.cache
def get_lora_supported_ranks():
    possible_max_ranks = [8, 16, 32, 64, 128, 256, 320, 512]
    try:
        import vllm.config.lora
        if hasattr(vllm.config.lora, "MaxLoRARanks"):
            possible_max_ranks = str(vllm.config.lora.MaxLoRARanks)
        else:
            lora_config = inspect.getsource(vllm.config.lora)
            text = "possible_max_ranks"
            l = lora_config.find(text)
            if l != -1:
                r = lora_config.find("\n", l + len(text))
                possible_max_ranks = lora_config[l : r]
    except:
        pass
    if type(possible_max_ranks) is str:
        possible_max_ranks = re.findall(r"[\d]{1,}", possible_max_ranks)
        possible_max_ranks = [int(x) for x in possible_max_ranks]
    return possible_max_ranks
pass


def determine_max_lora_rank(lora_rank = 16):
    """vLLM doesn't allow any LoRA rank, so we need to get the next largest"""
    possible_max_ranks = get_lora_supported_ranks()
    for max_lora_rank in possible_max_ranks:
        if max_lora_rank >= lora_rank:
            return max_lora_rank
    raise RuntimeError(
        f"Unsloth: vLLM does not support LoRA ranks of {lora_rank}.\n"\
        "Only `{possible_max_ranks}` is supported."
    )
pass


def vllm_supports_flashinfer(config) -> bool:
    """
    Approximately checks if a vLLM model supports FLASHINFER by checking
    vLLM's ModelRegistry, then inspecting if an `if self.attn_backend not in { ... }`
    guard excludes FLASHINFER.

    For eg Qwen3-VL does not work with flashinfer.
    """
    try:
        from vllm.model_executor.models.registry import ModelRegistry
    except Exception as e:
        print(
            f"Unsloth: Failed loading vLLM model class for arch {arch} "
            f"during `vllm_supports_flashinfer`.\n{e}"
        )
        return True

    architectures = getattr(config, "architectures", None) or []
    if isinstance(architectures, str):
        architectures = [architectures]

    # --- Get the vLLM model class without using resolve_model_cls() ---
    model_cls = None
    for arch in architectures:
        registered = getattr(ModelRegistry, "models", {}).get(arch)
        if registered is None:
            continue
        try:
            # _BaseRegisteredModel.load_model_cls() – works across versions
            model_cls = registered.load_model_cls()
            break
        except Exception as e:
            print(
                f"Unsloth: Failed loading vLLM model class for arch {arch} "
                f"during `vllm_supports_flashinfer`.\n{e}"
            )
            return True

    if model_cls is None:
        # Unknown architecture for vLLM; let vLLM handle it and don't block FLASHINFER.
        return True

    module = inspect.getmodule(model_cls)
    if module is None:
        return True

    def _module_disallows_flashinfer(module) -> bool:
        ATTENTION_BACKEND_GUARD_REGEX = re.compile(
            r"if\s+self\.attn_backend\s+not\s+in\s*{\s*(?P<body>.*?)\s*}:",
            re.DOTALL,
        )
        try:
            source = inspect.getsource(module)
        except Exception:
            # Can't inspect source – don't claim FLASHINFER is disallowed.
            return False

        matches = list(ATTENTION_BACKEND_GUARD_REGEX.finditer(source))
        if not matches:
            return False

        # For each guard, see if FLASHINFER appears in the allowed set.
        for m in matches:
            body = m.group("body")
            if "FLASHINFER" in body:
                # Some allowed-set includes FLASHINFER → don't disallow.
                return False

        # We found at least one guard, but none of its allowed sets mention FLASHINFER.
        # That's exactly the Qwen3-VL pattern:
        # { FLASH_ATTN, TORCH_SDPA, ROCM_AITER_FA }
        return True

    return not _module_disallows_flashinfer(module)
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
    max_num_seqs           : int = 256, # how many seqs to process in parallel. Default vLLM 256
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Create vLLM instance
    assert(config is not None)
    assert(type(use_bitsandbytes) is bool)
    assert(conservativeness >= 0.0 and conservativeness <= 1.0)

    unsloth_vllm_standby = unsloth_vllm_standby or (os.getenv("UNSLOTH_VLLM_STANDBY", "0") != "0")
    # This would give the flexibility to override the util we set for standby mode. In some extreme cases, this can be helpful.
    standby_util_override = os.getenv("UNSLOTH_VLLM_STANDBY_UTIL_OVERRIDE", "0") != "0"

    free_memory, total_memory = get_mem_info()
    # If T4 ie 15GB, we use 0.85 since it'll rarely OOM. Other GPUs 0.9
    # L4 with ~22GB seems to work at 0.89 but not 0.9 due to larget cuda graphs/large max num sequences we impose
    total_gb = total_memory/1024/1024/1024
    ten_percent = total_gb * 0.1 # 1.46GB for T4
    if   ten_percent >= 4.0: standby_target_gpu_util = 0.925
    elif ten_percent >= 2.5: standby_target_gpu_util = 0.9
    elif ten_percent >= 2.0: standby_target_gpu_util = 0.875
    elif ten_percent >= 1.4: standby_target_gpu_util = 0.85
    elif ten_percent >= 1.0: standby_target_gpu_util = 0.8
    else: standby_target_gpu_util = 0.75
    # Reduce memory usage for newer vLLM versions since it OOMs
    if Version(vllm_version) >= Version("0.11.0"):
        standby_target_gpu_util *= 0.95

    if unsloth_vllm_standby and not standby_util_override:
        if gpu_memory_utilization < standby_target_gpu_util:
            gpu_memory_utilization = standby_target_gpu_util
        print(f"Unsloth: Standby mode is enabled. Changing `gpu_memory_utilization` to {gpu_memory_utilization}.")

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

    # Determine the maximum LoRA rank since vLLM restricts the rank to some values
    new_max_lora_rank = determine_max_lora_rank(max_lora_rank)
    if new_max_lora_rank != max_lora_rank:
        print(f"Unsloth: Changing the maximum lora rank to {new_max_lora_rank} from {max_lora_rank} for vLLM.")
    max_lora_rank = new_max_lora_rank

    quant_method = get_quant_type(config)
    use_bitsandbytes = use_bitsandbytes or \
        model_name.lower().endswith("-bnb-4bit") or (quant_method == "bitsandbytes")

    is_fp8 = "fp8" in model_name.lower() or (quant_method in ("fp8", "fbgemm_fp8"))

    assert not (use_bitsandbytes and is_fp8), f'`load_in_4bit` and `load_in_8bit` should be set to false for loading FP8 quantized models with fast inference'

    max_num_batched_tokens, approx_max_num_seqs, \
    actual_gpu_memory_utilization, memory_left_for_kv_cache_gb = \
    approximate_vllm_memory_usage(
        mem_config,
        load_in_4bit = use_bitsandbytes,
        load_in_8bit = is_fp8,
        max_seq_length = max_seq_length,
        gpu_memory_utilization = gpu_memory_utilization,
        enable_lora = enable_lora,
        max_lora_rank = max_lora_rank,
        max_loras = max_loras,
        float8_kv_cache = float8_kv_cache,
        account_for_gradients = training,
    )

    enable_chunked_prefill = True
    is_mllama = "mllama" in config.model_type
    if is_mllama:
        # chunked prefill is not supported for vLLM V0.
        enable_chunked_prefill = False
        assert not enable_lora, "Unsloth: MLLama does not support LoRA with fast inference"
        assert max_seq_length >= 8192, "Unsloth: MLLama requires max_seq_length >= 8192 for fast inference"

    else:
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
    elif DEVICE_TYPE == "hip":
        _dtype = torch.bfloat16
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

    # Fix up vLLM compute_dtype for bitsandbytes
    BitsAndBytesConfig = patch_vllm_compute_dtype(dtype)

    # Use Flashinfer if possible (doesn't seem to be faster for BnB)
    # Also seems to process 2x less sequences in 1 go so less throughput?
    # Maybe FP8 Flashinfer is much better
    # See https://docs.vllm.ai/en/latest/serving/env_vars.html
    if importlib.util.find_spec("flashinfer"):
        # Check if FLASHINFER is supported - for eg Qwen3-VL and Qwen2-VL do not work
        if "VLLM_ATTENTION_BACKEND" in os.environ and os.environ["VLLM_ATTENTION_BACKEND"] == "":
            del os.environ["VLLM_ATTENTION_BACKEND"]
        elif not vllm_supports_flashinfer(config):
            if os.environ.get("VLLM_ATTENTION_BACKEND", "") == "FLASHINFER":
                print(f"Unsloth: `{model_name} does not support `VLLM_ATTENTION_BACKEND==FLASHINFER`. Will disable")
            if "VLLM_ATTENTION_BACKEND" in os.environ:
                del os.environ["VLLM_ATTENTION_BACKEND"]
        elif os.environ.get("VLLM_ATTENTION_BACKEND", "") != "":
            pass
        elif not use_bitsandbytes and major_version >= 8:
            # Allowed: FLASHINFER, TORCH_SDPA, FLASH_ATTN, XFORMERS, ROCM_FLASH
            os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
        elif Version(vllm_version) >= Version("0.11.0"):
            # On 0.11.0, Flashinfer also works!
            os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

        # Flashinfer sampler maybe makes it somewhat faster on newer GPUs
        # Tesla T4 is 280 tok/s vs 330 tok/s
        if major_version >= 8:
            os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "1"
        elif Version(vllm_version) >= Version("0.11.0"):
            # On 0.11.0, Flashinfer also works!
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
    elif DEVICE_TYPE == "hip":
        enable_prefix_caching = True
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
    # This is how many sequences can be processed in parallel
    # We do 64 on smaller GPUs
    # batch_size = 16, num_generations = 4 for eg.
    # We expand max_batched_tokens to 4096 on small GPUs and 8192 on large ones.
    """
    Benchmarks for max_batched_tokens, max_num_seqs
    Around after max_num_seqs>=64, we see linear increase in memory usage.
    | max_model_len | max_batched_tokens | max_num_seqs | Profiling Time | Non-KV Memory | Torch Peak | Non-Torch Forward | Weights |
    |--------------:|-------------------:|-------------:|---------------:|--------------:|-----------:|------------------:|--------:|
    | 2048          | 2048               | 8            | 11.18s         | 7.87GiB       | 0.18GiB    | 0.13GiB           | 7.56GiB |
    | 4096          | 4096               | 8            | 10.87s         | 8.01GiB       | 0.32GiB    | 0.13GiB           | 7.56GiB |
    | 8192          | 8192               | 8            | 11.24s         | 8.31GiB       | 0.62GiB    | 0.13GiB           | 7.56GiB |
    | 8192          | 8192               | 16           | 11.48s         | 8.31GiB       | 0.62GiB    | 0.13GiB           | 7.56GiB |
    | 8192          | 8192               | 32           | 11.09s         | 8.31GiB       | 0.62GiB    | 0.13GiB           | 7.56GiB |
    | 8192          | 8192               | 64           | 11.09s         | 8.31GiB       | 0.62GiB    | 0.13GiB           | 7.56GiB |
    | 8192          | 8192               | 128          | 11.38s         | 8.45GiB       | 0.76GiB    | 0.13GiB           | 7.56GiB |
    | 8192          | 8192               | 256          | 11.84s         | 9.14GiB       | 1.45GiB    | 0.13GiB           | 7.56GiB |
    | 8192          | 8192               | 512          | 11.50s         | 10.52GiB      | 2.83GiB    | 0.13GiB           | 7.56GiB |
    | 8192          | 8192               | 1024         | 11.03s         | 13.28GiB      | 5.59GiB    | 0.13GiB           | 7.56GiB |
    | 8192          | 8192               | 2048         | 11.63s         | 18.80GiB      | 11.11GiB   | 0.13GiB           | 7.56GiB |
    | 16384         | 16384              | 8            | 11.21s         | 8.89GiB       | 1.20GiB    | 0.13GiB           | 7.56GiB |
    | 32768         | 32768              | 8            | 11.27s         | 10.07GiB      | 2.38GiB    | 0.13GiB           | 7.56GiB |
    """
    approx_max_num_seqs = max_num_seqs # vLLM default is 256
    max_num_batched_tokens = 2048 # vLLM default
    if   memory_left_for_kv_cache_gb <=  2: max_num_batched_tokens, approx_max_num_seqs = 2048, 8   # - 8
    elif memory_left_for_kv_cache_gb <=  4: max_num_batched_tokens, approx_max_num_seqs = 2048, 16  # - 16
    elif memory_left_for_kv_cache_gb <=  8: max_num_batched_tokens, approx_max_num_seqs = 4096, 32  # - 16
    elif memory_left_for_kv_cache_gb <= 12: max_num_batched_tokens, approx_max_num_seqs = 4096, 48  # - 16
    elif memory_left_for_kv_cache_gb <= 16: max_num_batched_tokens, approx_max_num_seqs = 6144, 64  # Default
    elif memory_left_for_kv_cache_gb <= 24: max_num_batched_tokens, approx_max_num_seqs = 6144, 80  # + 16
    elif memory_left_for_kv_cache_gb <= 40: max_num_batched_tokens, approx_max_num_seqs = 8192, 96  # + 16
    elif memory_left_for_kv_cache_gb <= 48: max_num_batched_tokens, approx_max_num_seqs = 8192, 112 # + 16
    elif memory_left_for_kv_cache_gb <= 80: max_num_batched_tokens, approx_max_num_seqs = 8192, 128 # + 16
    elif memory_left_for_kv_cache_gb >  80: max_num_batched_tokens, approx_max_num_seqs = 8192, 256 # + 16

    if is_vision_model:
        # In vLLM profiling, each sequence contributes to an image. Which is generally in the order of thousand tokens.
        # We don't want to go beyond 16 sequences for vision models.
        # TODO: In vLLM V1, iirc, the profiling sets a cap on the max seqs based on the budget. Check it out.
        print(f'Unsloth: Vision model detected, setting approx_max_num_seqs to 1')
        # [TODO] Check this
        approx_max_num_seqs = 1
        # Single image would contribute to 6404 tokens in Llama 3.2 for eg. So have some more for text
        # For qwen 2.5 VL, this single image/video contributes to 16Ki tokens
        max_num_batched_tokens = max(8192, max_seq_length)

    # float8 KV cache can fit more sequences in 1 go so more throughput
    if float8_kv_cache: approx_max_num_seqs = int(approx_max_num_seqs * 1.05)

    # vLLM default max_num_batched_tokens is 2048
    chunked_prefill_tokens = 2048
    if not is_vision_model:
        if   memory_left_for_kv_cache_gb <=  8: chunked_prefill_tokens = 1024 # + 0
        elif memory_left_for_kv_cache_gb <= 12: chunked_prefill_tokens = 1536 # + 512
        elif memory_left_for_kv_cache_gb <= 16: chunked_prefill_tokens = 2048 # + 512
        elif memory_left_for_kv_cache_gb <= 24: chunked_prefill_tokens = 3072 # + 1024
        elif memory_left_for_kv_cache_gb <= 40: chunked_prefill_tokens = 4096 # + 1024
        elif memory_left_for_kv_cache_gb <= 48: chunked_prefill_tokens = 4608 # + 512
        elif memory_left_for_kv_cache_gb <= 80: chunked_prefill_tokens = 8192 # + 4096
        else: chunked_prefill_tokens = 8192 # + 0

        # vLLM errors out from max_seq_length (2048) being bigger than chunked_prefill_tokens (1024)
        chunked_prefill_tokens = max_seq_length

    # Scale num_seqs by conservativeness
    approx_max_num_seqs = int(approx_max_num_seqs * conservativeness)
    approx_max_num_seqs = max(approx_max_num_seqs, 1)

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
    pass

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
            from vllm.config import CompilationConfig

            # Torch versions >= 2.9.0 or vllm_version > 0.11.0
            if Version(vllm_version) > Version("0.11.0") or Version(torch_version) > Version("2.9.0"):
                cudagraphs = False # Weirdly if we set it to True, we get
                # [rank0]: RuntimeError: These storage data ptrs are not allocated in pool (0, 2) but should be {612290048}
                combo_kernels = True # Latest works now only on Llama it seems
                if total_memory_gb <= 70:
                    combo_kernels = False # Too slow on less than 80GB GPUs
                # We still see
                # AttributeError: 'NullKernelHandler' object has no attribute 'index_to_str'
                # Try unsloth/gemma-3-4b-it
                combo_kernels = False
            else:
                cudagraphs = True
                combo_kernels = False

            compile_flags = dict(
                level = 3,
                backend = "inductor",
                # cache_dir = "unsloth_compiled_vllm_cache", # Pytorch fails to load from cache
                # compile_sizes = [1, 2, 4, 8, 16],
                # cudagraph_capture_sizes = [1, 2, 4, 8, 16],
                # max_capture_size = 16,
                cudagraph_num_of_warmups = 1,
                full_cuda_graph = True,
                use_cudagraph = True,
                use_inductor = True,
                inductor_compile_config = get_torch_compile_options(
                    epilogue_fusion = True,
                    max_autotune = False, # Too slow
                    shape_padding = True,
                    debug = False,
                    cudagraphs = cudagraphs,
                    coordinate_descent_tuning = False, # Too slow
                    logging = True, # Enable compile logs
                    combo_kernels = combo_kernels,
                    group_fusion = True,
                    memory_planning = True,
                    use_block_ptr = True,

                    multi_kernel = False, # RuntimeError: name 'multi_kernel_0' is not defined
                    # [rank0]: TypeError: 'NoneType' object does not support the context manager protocol
                )
            )
            good_keys = inspect.signature(CompilationConfig).parameters.keys()
            # Use new cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE mode for maximum performance
            # See https://docs.vllm.ai/en/v0.10.2/api/vllm/config/compilation.html#vllm.config.compilation.CUDAGraphMode
            if "cudagraph_mode" in good_keys:
                try:
                    from vllm.config import CUDAGraphMode
                    compile_flags["cudagraph_mode"] = CUDAGraphMode.FULL_AND_PIECEWISE
                    del compile_flags["full_cuda_graph"]
                except Exception as e:
                    print("Unsloth: Failed getting `from vllm.config import CUDAGraphMode` and `CUDAGraphMode.FULL_AND_PIECEWISE`")
            else:
                print("Unsloth: `cudagraph_mode` is not in `from vllm.config import CompilationConfig`")
            old_keys = list(compile_flags.keys())
            for key in old_keys:
                if key not in good_keys:
                    del compile_flags[key]
                    print(f"Unsloth: Not an error, but `{key}` is not supported in vLLM.config.CompilationConfig. Skipping.")
                pass
            pass
            compilation_config = CompilationConfig(**compile_flags)
        except Exception as e:
            print(f"Unsloth: FAILED getting compilation_config with error = {str(e)}")
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
        enable_chunked_prefill = enable_chunked_prefill, # LoRA fails with chunked prefill as at Feb 2025
        # max_seq_len_to_capture fails for V1
        # max_seq_len_to_capture = min(8192, max_seq_length + 256), # Default is 8192 for CUDAGraphs
        compilation_config     = compilation_config, # 0, 1, 2, 3
        enforce_eager          = enforce_eager,
        swap_space             = swap_space, # Low memory devices like Colab (13GB) default 4GB
        device                 = device,
        # New vLLM versions need to pass this in!
        # worker_extension_cls   = "unsloth_zoo.vllm_rlhf_utils.ColocateWorkerExtension",
        enable_sleep_mode      = unsloth_vllm_standby,
    )
    if is_vision_model:
        # To reduce memory usage, we limit the number of images/videos per prompt
        # TODO: Make it configurable by user
        engine_args["limit_mm_per_prompt"] = {"image": 1, "video": 0}

    # [[CRITICAL for RL on policy]]
    # Check for Cascade Attention which fails on A100 / L40 for vLLM < 0.11.0 versions
    # Ada Lovelace 8.9 and Ampere 8.0
    # See https://github.com/vllm-project/flash-attention/pull/87
    # import vllm.vllm_flash_attn
    # vllm.vllm_flash_attn.__version__ == 2.7.2.post1
    if DEVICE_TYPE == "cuda":
        major_version, minor_version = torch.cuda.get_device_capability()
        if major_version < 9:
            if Version(vllm_version) >= Version("0.11.0"):
                disable_cascade_attn = False
            else:
                # Disable for A100, L40 etc
                disable_cascade_attn = True
                print("Unsloth: Disabling `disable_cascade_attn` in vLLM to allow for better on policy RL!")
            engine_args["disable_cascade_attn"] = disable_cascade_attn
    pass

    good_keys = inspect.signature(AsyncEngineArgs if use_async else EngineArgs).parameters.keys()
    old_keys = list(engine_args.keys())
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
            trials += 1
            # Cleanup
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()
            pass
            error = str(error)
            if trials >= 2 or unsloth_vllm_standby:
                # Sleep mode uses CuMemAllocator which can't run multiple instances in single process.
                # We can't do retry because vLLM will fail to load with said error.
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

    # Check if sleep mode, and send the model to sleep
    # This is to counteract OOMs before GRPO is launched like pre-inference runs
    if unsloth_vllm_standby and not standby_util_override:
        print(f"Unsloth: Standby mode is enabled. Pre-sleeping vLLM model to reduce OOMs.")
        llm.sleep(os.environ.get('VLLM_SLEEP_MODE', "1"))

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
    with open(os.path.join(save_directory, "adapter_config.json"), encoding = "utf-8") as f:
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
    dtype = _get_dtype(dtype_from_config(model.config) if dtype is None else dtype)

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
    dtype = _get_dtype(dtype_from_config(model.config) if dtype is None else dtype)

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

@torch.inference_mode()
def test_model_conversion(original_model, new_model):
    """
    Simplified model testing using clean comparison utilities.
    Replaces the complex _test_same_model function.
    """
    print("=== MODEL CONVERSION TEST ===")

    # Compare model attributes. Wouldn't throw error if some attributes are missing
    compare_attributes(original_model, new_model)

    try:
        # compare state dicts
        assert_same_state_dict(original_model.state_dict(), new_model.state_dict())
        print("✅ State dict comparison passed!")
    except Exception as e:
        print(f"❌ State dict comparison failed: {e}")
        return False

    print("✅ Model conversion test completed!")
    return True

def _test_is_same_vlm(model, new_model, processor, test_backward=False):
    # All Unsloth Zoo code licensed under LGPLv3
    assert model.device == new_model.device
    assert model.dtype == new_model.dtype

    messages = [{
        "role" : "user",
        "content": [
            { "type": "image", "image": "https://files.worldwildlife.org/wwfcmsprod/images/Sloth_Sitting_iStock_3_12_2014/story_full_width/8l7pbjmj29_iStock_000011145477Large_mini__1_.jpg"},
            { "type": "text",  "text" : "Which films does this animal feature in?" }
        ]
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=model.dtype)

    with torch.no_grad():
        original_outputs = model(**inputs)
        new_outputs = new_model(**inputs)
        torch.testing.assert_close(original_outputs.logits, new_outputs.logits)
        print(f'Forward pass logits match!')

    inputs['labels'] = inputs['input_ids']
    original_outputs = model(**inputs)
    new_outputs = new_model(**inputs)
    torch.testing.assert_close(original_outputs.loss, new_outputs.loss)
    print('Losses match !')

    if test_backward:
        # Initialize per-model statistics dictionaries
        original_model_stats = {
            'pre': defaultdict(list),
            'post': defaultdict(list),
            'backward': defaultdict(list)
        }

        new_model_stats = {
            'pre': defaultdict(list),
            'post': defaultdict(list),
            'backward': defaultdict(list)
        }

        # Register hooks for both models
        register_hooks(model, original_model_stats)
        register_hooks(new_model, new_model_stats)

        # Prepare inputs
        from copy import deepcopy
        inputs['labels'] = deepcopy(inputs['input_ids'])
        inputs['input_ids'].requires_grad = True

        # Forward passes
        original_outputs = model(**inputs)
        new_outputs = new_model(**inputs)

        # Check loss matches
        torch.testing.assert_close(original_outputs.loss, new_outputs.loss)
        print('Losses match!')

        # Backward passes
        original_outputs.loss.backward()
        new_outputs.loss.backward()

        # Compare backward gradient statistics
        matches = []
        mismatches = []
        for layer_name in original_model_stats['backward'].keys():
            original_grads = torch.tensor(original_model_stats['backward'][layer_name])
            new_grads = torch.tensor(new_model_stats['backward'][layer_name])
            try:
                torch.testing.assert_close(original_grads, new_grads, atol=1e-6)
                matches.append(layer_name)
            except Exception as e:
                print(f"Gradient mismatch in layer '{layer_name}': {e}")
                mismatches.append(layer_name)
        print(f"Backward gradient statistics match for {len(matches)} layers: {matches}")
        print(f"Backward gradient statistics mismatch for {len(mismatches)} layers: {mismatches}")
pass


def _read_unsloth_vision_source() -> str:
    _VISION_TAIL = ("unsloth", "models", "vision.py")
    from importlib.metadata import files, PackageNotFoundError, PackagePath
    from pathlib import Path
    # 1) Via installed distribution metadata (no import of the package)
    try:
        for entry in files("unsloth") or ():
            if isinstance(entry, PackagePath):
                parts = entry.parts
                if len(parts) >= 3 and tuple(parts[-3:]) == _VISION_TAIL:
                    return entry.read_text(encoding = "utf-8")
    except PackageNotFoundError:
        pass

    # 2) Fallback: scan sys.path for a plain file
    for base in map(Path, sys.path):
        candidate = base.joinpath(*_VISION_TAIL)
        if candidate.is_file():
            return candidate.read_text(encoding = "utf-8")
    raise FileNotFoundError("Could not locate unsloth/models/vision.py without importing it")
pass


def get_vllm_supported_vlm(_VAR_NAME = "VLLM_SUPPORTED_VLM"):
    """
    Parse VLLM_SUPPORTED_VLM from unsloth/models/vision.py as a literal.
    """
    src = _read_unsloth_vision_source()
    tree = ast.parse(src)

    # Support: `VLLM_SUPPORTED_VLM = [...]` and `VLLM_SUPPORTED_VLM: list[str] = [...]`
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(getattr(t, "id", None) == _VAR_NAME for t in node.targets):
                return ast.literal_eval(node.value)
        elif isinstance(node, ast.AnnAssign):
            if getattr(node.target, "id", None) == _VAR_NAME:
                return ast.literal_eval(node.value)
    raise ValueError(f"{_VAR_NAME} not found as a literal in unsloth/models/vision.py")
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
    config.model_name = model_name

    if not vllm_dynamic_quant_supported(model_name, config):
        raise NotImplementedError(f"Unsloth: Dynamic quant of {model_name} not supported in vLLM")

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    bnb_config = None
    load_in_4bit = model_name.lower().endswith("-bnb-4bit") or load_in_4bit
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
    kwargs = add_dtype_kwargs(dtype, kwargs)
    # Must patch BnB compute_dtype since it's forced to bfloat16!
    patch_bitsandbytes_quant_state()
    # patch_bitsandbytes_compute_dtype(dtype)
    model_type = getattr(config, "model_type", "causal_lm")

    enable_lora = model_type != "mllama"

    if not is_vision_model:
        model_class = AutoModelForCausalLM
    else:
        VLLM_SUPPORTED_VLM = get_vllm_supported_vlm()
        if model_type in VLLM_SUPPORTED_VLM:
            import transformers
            model_class = getattr(transformers, config.architectures[0])
        else:
            raise ValueError(f"Unsloth: Model type {model_type} not supported for vision models")

    print(f'Loading model with type {model_class}')
    model = model_class.from_pretrained(
        model_name,
        device_map          = "sequential",
        # torch_dtype         = dtype,  transformers moved torch_dtype to dtype
        attn_implementation = "sdpa",
        low_cpu_mem_usage   = True,
        **kwargs,
    )

    # unpatch_bitsandbytes_compute_dtype()
    for param in model.parameters():
        param.requires_grad_(False)
    model, _ = patch_model_and_tokenizer(model, None)
    model.eval()

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
        is_vision_model        = is_vision_model,
        enable_lora            = enable_lora,
    )

    state_dict, quant_state_dict = get_vllm_state_dict(
        llm,
        return_state_dict = True,
        config = config,
        is_vision_model = is_vision_model,
    )

    assert_same_state_dict(model.state_dict(), state_dict)

    new_model = convert_vllm_to_huggingface(quant_state_dict, config, dtype, is_vision_model = is_vision_model)
    test_model_conversion(model, new_model)

    # Run the model as well
    if not is_vision_model:
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

        if not skip_generation:
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
    else:
        # VLMs dont have a standardised forward pass mechanism. So we just test whole model forward pass and not layer wise
        # TODO: Maybe add layer wise checks
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name)
        _test_is_same_vlm(model, new_model, processor, False)

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

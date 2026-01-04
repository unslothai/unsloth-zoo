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
    "create_huggingface_repo",
    "merge_and_dequantize_lora",
    "merge_and_overwrite_lora",
]
import warnings
from .peft_utils import get_lora_layer_modules
from .utils import _get_dtype
from .hf_utils import dtype_from_config
from .temporary_patches.common import UNSLOTH_ENABLE_LOGGING, logger
from collections import defaultdict

try:
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors, convert_moe_packed_tensors_cpu
except (ImportError, ModuleNotFoundError):
    # Provide a fallback or a clear error if the function isn't available
    # when not using mxfp4.
    convert_moe_packed_tensors     = None
    convert_moe_packed_tensors_cpu = None
pass

MODEL_CARD = \
"""---
base_model: {base_model}
tags:
- text-generation-inference
- transformers
- unsloth
- {model_type}
- {extra}
license: apache-2.0
language:
- en
---

# Uploaded finetuned {method} model

- **Developed by:** {username}
- **License:** apache-2.0
- **Finetuned from model :** {base_model}

This {model_type} model was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth) and Huggingface's TRL library.

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)
"""

import torch
import bitsandbytes as bnb
try:
    from huggingface_hub import get_token
except:
    try:
        from huggingface_hub.utils import get_token
    except:
        # For older versions of huggingface_hub
        from huggingface_hub.utils._token import get_token
    pass
pass
from transformers.modeling_utils import PushToHubMixin
import json
import os
from pathlib import Path
from typing import Union, List, Optional
import tempfile
from peft import PeftModelForCausalLM, PeftModel

def find_skipped_quantized_modules(model):
    skipped_modules = []
    quantized_modules = []
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            if hasattr(module.weight, 'quant_state') and module.weight.quant_state is not None:
                quantized_modules.append(name)
            else:
                skipped_modules.append(name)
        elif isinstance(module, torch.nn.Linear):
            skipped_modules.append(name)
    return skipped_modules, quantized_modules
pass

def create_huggingface_repo(
    model,
    repo_id,
    private = False,
    token = None,
):
    # All Unsloth Zoo code licensed under LGPLv3
    assert(type(repo_id) is str)
    if repo_id.count("/") != 1:
        raise TypeError(f"Unsloth: You are pushing to Hugging Face, but {repo_id} is not a valid repo.")

    from huggingface_hub import ModelCard, HfApi
    if token is None: token = get_token()
    api = HfApi(token = token)
    repo_url = api.create_repo(
        repo_id = repo_id,
        private = private,
        exist_ok = True,  # don't error if repo already exists
    )
    username = repo_id.split("/")[0]

    # Check if base_model is a local path
    base_model = model.config._name_or_path
    if os.path.exists(base_model) and os.path.isdir(base_model):
        # Try to get the original model ID from config
        original_model_id = get_original_model_id(base_model)
        if original_model_id is not None and not os.path.exists(original_model_id):
            # Use the original model ID if it doesn't look like a local path
            base_model = original_model_id
        else:
            # If we can't determine the original model, use repo_id as a generic description
            # that won't cause HF validation errors
            base_model = repo_id

    # Create model card
    content = MODEL_CARD.format(
        username   = username,
        base_model = base_model,
        model_type = model.config.model_type,
        method     = "",
        extra      = "unsloth",
    )
    card = ModelCard(content)
    card.push_to_hub(repo_id, token = token, commit_message = "Unsloth Model Card")

    hf_api = HfApi(token = token)
    return username, repo_id, hf_api
pass


from huggingface_hub import (
    snapshot_download,
    hf_hub_download,
    HfFileSystem,
)
from safetensors import safe_open
from safetensors.torch import save_file
from collections import OrderedDict
from tqdm import tqdm as ProgressBar
import os, shutil, re, functools


def _merge_lora(W, lora_stats, name):
    if lora_stats.lora_A is None or lora_stats.lora_B is None: return W
    W = W.to("cuda", dtype = torch.float32, non_blocking = True)
    W = W.addmm_(
        lora_stats.lora_B.to("cuda", dtype = torch.float32, non_blocking = True),
        lora_stats.lora_A.to("cuda", dtype = torch.float32, non_blocking = True),
        alpha = lora_stats.alpha,
    )
    if not torch.isfinite(torch.amax(W)).item():
        raise ValueError('Unsloth: Merge failed as there are infinite elements in ' + name)
    return W
pass


def check_if_quantized(module: torch.nn.Module) -> bool:
    # All Unsloth Zoo code licensed under LGPLv3
    # Adapted from https://github.com/huggingface/peft/blob/main/src/peft/utils/integrations.py
    if not hasattr(module, "weight"): return False

    if hasattr(module, "W_q"):  # For handling HQQ quantized weight
        # weight = module.dequantize()
        # return weight
        return True
    elif type(module.weight).__module__.startswith("torchao."):
        # check for torchao without requiring any torchao imports
        # weight = module.weight.dequantize()
        # return weight
        return True

    weight = module.weight
    if not isinstance(weight, torch.nn.Parameter):
        if isinstance(weight, torch.Tensor):
            # this is an FSDP-specific edge case
            # return weight  # type: ignore
            return False
        # raise TypeError(f"Input weight should be of type nn.Parameter, got {type(weight)} instead")
        return False

    cls_name = weight.__class__.__name__
    if cls_name not in ("Params4bit", "Int8Params"):
        # return weight
        return False

    quant_state = getattr(module, "state", None)
    device = weight.device
    is_cpu = device.type == torch.device("cpu").type
    return True
    # weight = dequantize_bnb_weight(weight, state=quant_state)  # no-op if not bnb
    # if is_cpu:
    #     # dequantize_bnb_weight for 8bit moves the device in-place, thus we need to move it back to CPU if necessary
    #     module.weight = module.weight.to(device)
    # return weight
pass


def expand_module_keys(name, module, original_keys):
    # All Unsloth Zoo code licensed under LGPLv3
    keys = module.state_dict().keys()
    for key in keys: original_keys.add(name + "." + key)
    return original_keys
pass


from peft.utils.integrations import dequantize_module_weight
import collections
import numpy as np
import inspect
from tqdm import tqdm as ProgressBar
from dataclasses import dataclass

@dataclass
class LoraStats:
    module : torch.nn.Module
    lora_A : torch.Tensor
    lora_B : torch.Tensor
    alpha  : float
pass


def assert_same_keys(model, new_state_dict):
    # All Unsloth Zoo code licensed under LGPLv3
    inner_model = model.base_model.model if hasattr(model, "base_model") else model
    original_keys = inner_model.state_dict().keys()
    all_original_keys = set()
    for x in original_keys:
        where_weight = x.rfind(".weight")
        where_bias   = x.rfind(".bias")
        if where_weight != -1: x = x[:where_weight + len(".weight")]
        elif where_bias != -1: x = x[:where_bias   + len(".bias")  ]
        else: pass

        # Remove LoRA and base_layer
        j = max(x.rfind(".lora_"), x.rfind(".base_layer"))
        if j != -1: x = x[:j] + x[x.rfind("."):]

        all_original_keys.add(x)
    pass
    difference = all_original_keys ^ set(new_state_dict)
    if len(difference) != 0:
        raise RuntimeError(f"Unsloth: Extracted keys = {difference} do not match!")
    pass
pass


@torch.inference_mode
def create_lora_statistics(model, merge_into_original = False, return_state_dict = True):
    # All Unsloth Zoo code licensed under LGPLv3
    # merge_into_original is merging directly into 16bit downloaded model
    # without dequantizing
    Linear_LoRA_Layers = get_lora_layer_modules()
    Linear_LoRA_Layers = tuple(x[0] for x in Linear_LoRA_Layers)

    lora_weights = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    module_count, lora_A_count, lora_B_count, scaling_count = 0, 0, 0, 0

    remove_keys = set()
    keep_keys   = set()

    inner_model = find_lora_base_model(model)
    for name, module in inner_model.named_modules():
        if name == "": continue

        elif name.endswith(".lora_A.default"):
            lora_weights[name[:-len(".lora_A.default")]].lora_A = module.weight
            lora_A_count += 1
            expand_module_keys(name, module, remove_keys)

        elif name.endswith(".lora_B.default"):
            lora_weights[name[:-len(".lora_B.default")]].lora_B = module.weight
            lora_B_count += 1
            expand_module_keys(name, module, remove_keys)

        elif isinstance(module, Linear_LoRA_Layers):
            active_adapter = module.active_adapters[0] if \
                hasattr(module, "active_adapters") else module.active_adapter
            lora_weights[name].alpha = module.scaling[active_adapter]
            scaling_count += 1
            expand_module_keys(name, module, remove_keys)

        elif name.endswith(".base_layer"):
            lora_weights[name[:-len(".base_layer")]].module = module
            module_count += 1
            remove_keys.add(name)
            remove_keys.add(name[:-len(".base_layer")])

        elif (not merge_into_original) and check_if_quantized(module):
            lora_weights[name].module = module
            keep_keys.add(name + ".weight")
            if getattr(module, "bias", None) is not None: keep_keys.add(name + ".bias")
            expand_module_keys(name, module, remove_keys)
            remove_keys.add(name)

        elif ".lora_" in name: continue

        else:
            new_keys = expand_module_keys(name, module, set())
            for key in new_keys:
                if not key.endswith((".weight", ".bias")):
                    # Check if quantized item exactly which has ".weight"
                    if ".weight." in key:
                        remove_keys.add(key)
                    else:
                        # Keep gate_tanh, embedding etc
                        pass
            remove_keys.add(name)
        pass
    pass
    assert(module_count == lora_A_count == lora_B_count == scaling_count)

    # Also return state_dict if needed
    if return_state_dict:
        old_state_dict = inner_model.state_dict()
        state_dict     = collections.OrderedDict()
        for name, param in old_state_dict.items():

            if name.endswith(".base_layer.weight"):
                name = name[:-len(".base_layer.weight")]

            if name in lora_weights:
                state_dict[name + ".weight"]   = lora_weights[name]
                if getattr(lora_weights[name].module, "bias", None) is not None:
                    state_dict[name + ".bias"] = lora_weights[name].module.bias
                continue
            elif name in keep_keys:
                # Quantized modules with no LoRA adapters
                lora_name = name[:-len(".weight")]
                if lora_name in lora_weights:
                    param = lora_weights[lora_name]
                else:
                    # Bias term
                    pass
            elif name in remove_keys: continue

            state_dict[name] = param
        pass
    else:
        state_dict = None
    pass

    if return_state_dict: assert_same_keys(model, state_dict)
    return lora_weights, state_dict
pass


import torch
import gc
import time
import safetensors
import json
import mmap
import ctypes
# Mapping from BF16 to torch.blfloat16 etc
try:
    SAFETENSORS_DTYPES = safetensors.torch._TYPES
except:
    logger.info("Unsloth: `safetensors.torch._TYPES` does not exist. Will set to our default version")
    SAFETENSORS_DTYPES = {
        'F64': torch.float64,
        'F32': torch.float32,
        'F16': torch.float16,
        'BF16': torch.bfloat16,
        'I64': torch.int64,
        'I32': torch.int32,
        'I16': torch.int16,
        'I8': torch.int8,
        'U8': torch.uint8,
        'BOOL': torch.bool,
        'F8_E4M3': torch.float8_e4m3fn,
        'F8_E5M2': torch.float8_e5m2,
        'U64': torch.uint64,
        'U32': torch.uint32,
        'U16': torch.uint16,
    }
pass

@torch.inference_mode
def _merge_and_overwrite_lora(
    save_directory,
    filename,
    lora_weights,
    output_dtype,
    model_class_name,
    base_model_is_quantized = False,
    quant_type = None,
    save_method = "merged_16bit"
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Merges LoRA and overwrites the safetensors file it was merged to
    if base_model_is_quantized and quant_type == "mxfp4" and save_method != "mxfp4":
        if UNSLOTH_ENABLE_LOGGING:
            logger.info("mxfp4 quantized model detected. Using safe rewrite strategy (requires temporary disk space).")
        # Here, we fall back to the complete rewrite logic.
        # This logic is extracted from your original 'working_code'.
        return _merge_and_overwrite_lora_mxfp4(
            save_directory, filename, lora_weights, output_dtype,
            model_class_name, base_model_is_quantized, quant_type,
        )
    pass

    filename_original = os.path.join(save_directory, filename)  # Original file path
    count = 0

    # Convert lora_weights to safetensor format
    converted_lora_weights = _convert_lora_keys_to_safetensor_format(
        lora_weights,
        [],
        model_class_name = model_class_name,
    )

    # Open original file for reading
    raw_pointer = None
    mm = None
    header_metadata = None
    length_of_header = 0

    # Only if overwriting
    try:
        # Memory map the file for direct access
        raw_pointer = open(filename_original, "r+b")
        mm = mmap.mmap(raw_pointer.fileno(), length = 0, access = mmap.ACCESS_WRITE)

        # Parse safetensors header
        length_of_header = int.from_bytes(mm.read(8), "little")
        header_metadata = json.loads(mm.read(length_of_header))
        mm.seek(0)

        with safe_open(filename_original, framework = "pt", device = "cpu") as file:
            safetensor_keys = list(file.keys())

            # Update converted_lora_weights with actual safetensor keys
            converted_lora_weights = _convert_lora_keys_to_safetensor_format(
                lora_weights,
                safetensor_keys,
                model_class_name = model_class_name,
            )

            processed_mxfp4_keys = set()

            for key in safetensor_keys:
                if key in processed_mxfp4_keys:
                    continue

                # FORCE memory cleanup before processing each tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                is_save_mxfp4 = base_model_is_quantized and quant_type == "mxfp4" and save_method == "mxfp4"
                if is_save_mxfp4 and (key.endswith("_blocks") or key.endswith("_scales")):
                    # In this mode, we don't dequantize or modify MXFP4 tensors.
                    # Since we're doing an in-place overwrite on the file,
                    # skipping these keys leaves them untouched in the final model file.
                    if UNSLOTH_ENABLE_LOGGING:
                        logger.info(f"[DEBUG] Preserving MXFP4 tensor: {key}")
                    continue
                pass

                output_key = key
                action_logged = False
                # Standard 16-bit model
                W = file.get_tensor(key)
                W_original_dtype = W.dtype

                if W is None:
                    continue

                # Check for LoRA merge
                lora_key = output_key[:-len(".weight")] if output_key.endswith(".weight") else output_key
                lora_stats = converted_lora_weights.get(lora_key, None)

                if lora_stats is not None and hasattr(lora_stats, 'lora_A') and lora_stats.lora_A is not None:
                    W = _merge_lora(W, lora_stats, output_key)
                    count += 1

                # FIXED: Direct tensor writing using torch
                success = _write_tensor_direct_torch(mm, header_metadata, length_of_header, output_key, W, W_original_dtype)

                if not success:
                    raise RuntimeError(f"Failed to write tensor to model file.")

                del W
                torch.cuda.empty_cache()
            pass
            # Success! Direct overwrite completed
        pass
        mm.flush()
        mm.close()
        raw_pointer.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return count

    except Exception as e:
        raise RuntimeError(f"Model merge failed with error: {e}")

    finally:
        # Cleanup memory mapping
        if mm is not None:
            try:
                mm.close()
            except:
                pass
        if raw_pointer is not None:
            try:
                raw_pointer.close()
            except:
                pass
    return count
pass

@torch.inference_mode
def _merge_and_overwrite_lora_mxfp4(save_directory, filename, lora_weights, output_dtype, model_class_name, base_model_is_quantized=False, quant_type=None):
    # All Unsloth Zoo code licensed under LGPLv3
    # Merges LoRA and overwrites the safetensors file it was merged to
    filename_original = os.path.join(save_directory, filename)  # Original file path
    tensors = OrderedDict()
    count = 0
    import psutil
    import pickle
    limit = 700 * 1024 * 1024  # 700MB

    # Convert lora_weights to safetensor format
    converted_lora_weights = _convert_lora_keys_to_safetensor_format(
        lora_weights,
        [],
        model_class_name = model_class_name,
    )

    with safe_open(filename_original, framework = "pt", device = "cpu") as file: # Open original file for reading
        safetensor_keys = list(file.keys())

        # Update converted_lora_weights with actual safetensor keys
        converted_lora_weights = _convert_lora_keys_to_safetensor_format(
            lora_weights,
            safetensor_keys,
            model_class_name = model_class_name,
        )

        # Set to track mxfp4 keys that have already been processed
        processed_mxfp4_keys = set()

        for key in safetensor_keys:
            if key in processed_mxfp4_keys:
                continue

            # FORCE memory cleanup before processing each tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            W = None
            output_key = key
            action_logged = False
            # --- START OF MODIFIED LOGIC ---

            # This block handles ALL keys from a hybrid MXFP4 file.
            if key.endswith("_blocks"):
                if convert_moe_packed_tensors is None:
                    raise ImportError("MXFP4 dequantization is required, but `convert_moe_packed_tensors` could not be imported.")

                base_name = key[:-len("_blocks")]
                scales_key = base_name + "_scales"
                output_key = base_name # Correct naming without .weight
                if scales_key not in safetensor_keys:
                    warnings.warn(f"Found mxfp4 tensor {key} but missing its scales tensor {scales_key}. Skipping.")
                    continue

                blocks_tensor, scales_tensor = file.get_tensor(key), file.get_tensor(scales_key)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Wait for previous operations to complete
                    torch.cuda.empty_cache()

                # Determine optimal device and chunk size for mxfp4 dequantization
                device_type, device_id, rows_per_chunk = _choose_mxfp4_processing_strategy(
                    blocks_tensor, scales_tensor
                )

                # Apply dequantization with optimal parameters
                if device_type == 'cpu':
                    # Use CPU-optimized version
                    try:
                        from transformers.integrations.mxfp4 import convert_moe_packed_tensors_cpu
                        W = convert_moe_packed_tensors_cpu(
                            blocks_tensor, scales_tensor, rows_per_chunk=rows_per_chunk
                        ).transpose(1, 2).contiguous()
                        if UNSLOTH_ENABLE_LOGGING:
                            logger.info(f"[DEBUG] Using CPU dequantization for {base_name} with {rows_per_chunk:,} rows per chunk")
                    except ImportError:
                        # Fallback to original function
                        W = convert_moe_packed_tensors(
                            blocks_tensor, scales_tensor, rows_per_chunk=rows_per_chunk
                        ).transpose(1, 2).contiguous()
                else:
                    # Use GPU version (original or patched)
                    W = convert_moe_packed_tensors(
                        blocks_tensor, scales_tensor, rows_per_chunk=rows_per_chunk
                    ).transpose(1, 2).contiguous()
                    if UNSLOTH_ENABLE_LOGGING:
                        logger.info(f"[DEBUG] Using GPU dequantization for {base_name} with {rows_per_chunk:,} rows per chunk")

                processed_mxfp4_keys.add(key); processed_mxfp4_keys.add(scales_key)

                lora_stats = converted_lora_weights.get(base_name, None)
                if lora_stats and hasattr(lora_stats, 'lora_A') and lora_stats.lora_A is not None:
                    if UNSLOTH_ENABLE_LOGGING:
                        logger.info(f"[DEBUG] DEQUANTIZING MXFP4 & MERGING LoRA into Key Group: {base_name}")
                    count += 1; W = _merge_lora(W, lora_stats, output_key)
                else:
                    if UNSLOTH_ENABLE_LOGGING:
                        logger.info(f"[DEBUG] DEQUANTIZING MXFP4 Key Group: {base_name}")
                action_logged = True

            elif key.endswith("_scales"):
                continue

            else:
                # Handle the 16-bit tensors (like attention layers)
                # that are present in the same file as the MXFP4 tensors.
                W = file.get_tensor(key)


            # Remove .weight suffix to match LoRA key format
            lora_key = output_key[:-len(".weight")] if output_key.endswith(".weight") else output_key
            lora_stats = converted_lora_weights.get(lora_key, None)

            if W is not None and lora_stats is not None and hasattr(lora_stats, 'lora_A') and lora_stats.lora_A is not None:
                if not action_logged:
                    count += 1
                    W = _merge_lora(W, lora_stats, output_key)  # Assume _merge_lora is defined elsewhere
                    action_logged = True

            if W is None:
                continue

            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
                temp_filename = temp_file.name
                # Save the merged tensor to a unique temp file
                torch.save(W.to(output_dtype), temp_filename, pickle_module=pickle, pickle_protocol=pickle.HIGHEST_PROTOCOL)
                del W
                # Load it back as a memory-mapped object. The OS will manage paging this from disk.
                W = torch.load(temp_filename, map_location="cpu", mmap=True, weights_only=False)

                # Clean up the temporary pickle file immediately after mmaping
            try:
                os.remove(temp_filename)
            except OSError:
                # On Windows, the mmap might keep a handle. The OS will clean it up.
                pass

            tensors[output_key] = W

            # Free up VRAM after each merge
            torch.cuda.empty_cache()

    # CRITICAL: Force cleanup to release file handles on Windows
    if os.name == 'nt':
        gc.collect()
        time.sleep(0.1)  # Give Windows a moment to release file handles

    # Create a temporary file in the same directory for atomic rename
    with tempfile.NamedTemporaryFile(suffix=".safetensors", dir=save_directory, delete=False) as tmpfile:
        temp_filename_safetensors = tmpfile.name

    save_file(tensors, temp_filename_safetensors, metadata={"format": "pt"})  # Save to the temporary safetensors file

    # Replace the temporary file with the original file
    try:
        os.replace(temp_filename_safetensors, filename_original)  # Attempt atomic rename
    except OSError as e:
        # If rename fails (e.g., due to permissions), fall back to copy and remove temporary file
        print(f"Error renaming temporary file: {e}. Attempting copy and replace.")
        import shutil

        # On Windows, we might still have the file locking issue with copy
        if os.name == 'nt':
            # Try a few times with delays
            for attempt in range(3):
                try:
                    shutil.copy2(temp_filename_safetensors, filename_original)
                    break
                except PermissionError:
                    if attempt == 2:  # Last attempt
                        raise
                    time.sleep(0.5)
                    gc.collect()
        else:
            shutil.copy2(temp_filename_safetensors, filename_original)

        # Clean up temp file
        try:
            os.remove(temp_filename_safetensors)
        except:
            pass

    return count
pass

from huggingface_hub import (
    split_state_dict_into_shards_factory,
    get_torch_storage_size,
    get_torch_storage_id,
)

def get_torch_storage_size_new(x, element_size):
    if isinstance(x, LoraStats):
        shape = (x.module.in_features, x.module.out_features)
        return int(np.prod(shape)) * element_size
    else:
        return get_torch_storage_size(x)
pass


def get_torch_storage_id_new(x):
    if isinstance(x, LoraStats):
        return None
    else:
        return get_torch_storage_id(x)
pass


def prepare_saving(
    model,
    save_directory,
    push_to_hub = False,
    max_shard_size = "5GB",
    private = True,
    token = None,
    output_dtype = None,
    merge_into_original = False,
    low_disk_space_usage = False,
    min_size_in_bytes = 100_000_000, # Must be of this size - 100MB default
    use_temp_file = False,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Check size
    from huggingface_hub.serialization._base import parse_size_to_int
    max_shard_size_in_bytes = max_shard_size
    if type(max_shard_size_in_bytes) is not int:
        max_shard_size_in_bytes = parse_size_to_int(max_shard_size)
    pass

    temp_file = None
    username, repo_id, hf_api = None, None, None

    if push_to_hub:
        if token is None: token = get_token()
        username, repo_id, hf_api = create_huggingface_repo(
            model = model,
            repo_id = save_directory,
            private = private,
            token = token,
        )
        # Check if temporary folder is needed
        if os.path.isdir(save_directory) or use_temp_file:
            temp_file = tempfile.TemporaryDirectory(ignore_cleanup_errors = True)
            save_directory = temp_file.name
            use_temp_file = True
        pass
    pass

    if output_dtype is None: output_dtype = _get_dtype(dtype_from_config(model.config))
    assert(output_dtype in (torch.float32, torch.float16, torch.float64, torch.bfloat16))
    assert(type(torch.bfloat16) is torch.dtype)
    element_size = torch.tensor([], dtype = output_dtype).element_size()

    # Get state_dict
    lora_weights, state_dict = create_lora_statistics(
        model,
        merge_into_original = merge_into_original,
        return_state_dict = True,
    )
    # Total save size in bytes
    save_size = sum(get_torch_storage_size_new(x, element_size) for x in state_dict.values())

    # Create folder if it does not exist
    if not os.path.exists(save_directory):
        try:
            os.makedirs(save_directory, exist_ok = True)
        except Exception as error:
            raise RuntimeError(f"Unsloth: Error creating directory {save_directory} with error = {str(error)}")
    pass

    # Check if directory has enough space
    total, used, free = shutil.disk_usage(save_directory)
    free = int(free*0.95)

    def raise_upload_works():
        # Works with individual shard uploading
        raise RuntimeError(
            "Unsloth: Failed saving locally - no disk space left. "\
            "Uploading can work luckily! Use .push_to_hub instead."
        )
    pass

    if free < save_size:
        # Fail if already using temp folder except if individual portions work!
        if use_temp_file:
            if merge_into_original:
                if free > min_size_in_bytes:
                    # Downloading safetensor shards must be min shard size
                    low_disk_space_usage = True
                else: raise_upload_works()
            elif free > 100_000_000:
                if push_to_hub:
                    # Instead we form shards on the fly and push them!
                    low_disk_space_usage = True
                    max_shard_size_in_bytes = free
                else: raise_upload_works()
            else:
                raise RuntimeError("Failed saving - no disk space left!")
        pass

        # Too small - try using the temporary file system (sometimes large like Kaggle)
        try_temp_file = tempfile.TemporaryDirectory(ignore_cleanup_errors = True)
        try_save_directory = try_temp_file.name

        total, used, free = shutil.disk_usage(try_save_directory)
        free = int(free*0.95)
        if not push_to_hub and free > save_size: raise_upload_works()
        elif push_to_hub and free < save_size:
            raise RuntimeError("Unsloth: Failed uploading - no disk space left.")
        elif push_to_hub:
            print(
                f"Unsloth: Saving to {save_directory} will fail, but using a temp folder works! "\
                "Switching to a temp folder then uploading!"
            )
            # Switch to temp directory
            temp_file = try_temp_file
            save_directory = try_save_directory
            use_temp_file = True
        else:
            raise RuntimeError("Failed saving - no disk space left!")
        pass
    pass

    return (
        username, repo_id, hf_api, token,
        output_dtype, element_size,
        lora_weights, state_dict, save_size, free,
        temp_file, save_directory, use_temp_file,
        low_disk_space_usage, max_shard_size_in_bytes,
    )
pass


def _remove_quantization_config(config_path: Path):
    assert config_path.exists(), "Given config does not exist"
    with open(config_path, "r", encoding = "utf-8") as f:
        config = json.load(f)
    if "quantization_config" in config:
        # Remove the quantization_config field
        del config["quantization_config"]
    else:
        return
    # Overwrite the config file
    with open(config_path, "w", encoding = "utf-8") as f:
        json.dump(config, f, indent = 4)
    pass
pass

def fix_tokenizer_config_json(tokenizer, saved_folder):
    # Add "chat_template" to tokenizer_config.json
    tokenizer_config_path = os.path.join(saved_folder, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path) and tokenizer is not None:
        old_chat_template = getattr(tokenizer, "chat_template", None)
        if old_chat_template is not None:
            try:
                with open(tokenizer_config_path, "r") as f:
                    f = json.load(f)
                if "chat_template" not in f or f["chat_template"] is None:
                    f["chat_template"] = tokenizer.chat_template
                with open(tokenizer_config_path, "w") as new_f:
                    json.dump(f, new_f, indent = 2, ensure_ascii = False)
            except:
                pass
        pass

        # Remove chat_template if NULL
        try:
            with open(tokenizer_config_path, "r") as f:
                f = json.load(f)
            if "chat_template" in f and (f["chat_template"] == "" or f["chat_template"] is None):
                del f["chat_template"]
            with open(tokenizer_config_path, "w") as new_f:
                json.dump(f, new_f, indent = 2, ensure_ascii = False)
        except:
            pass
    pass
    # Fix config.json using torch_dtype / dtype
    config_file_path = os.path.join(saved_folder, "config.json")
    if os.path.exists(config_file_path):
        try:
            with open(config_file_path, "r") as f:
                data = f.read()
            data = data.replace('"dtype"', '"torch_dtype"')
            data = data.replace("'dtype'", "'torch_dtype'")
            with open(config_file_path, "w") as f:
                f.write(data)
        except:
            pass
    return
pass

def is_hf_sharded_safetensors(filenames: list[str]) -> bool:
    """Check if filenames follow HF sharded naming: model-00001-of-00005.safetensors"""
    pattern = re.compile(r'^(.+?)-(\d+)-of-(\d+)\.safetensors$')
    
    matches = [pattern.match(f) for f in filenames]
    if not all(matches):
        return False
    
    # Keep strings to check padding
    parsed = [(m.group(1), m.group(2), m.group(3)) for m in matches]
    
    # shard and total have same padding: turned off as deepseekocr padding is different
    # for prefix, shard_str, total_str in parsed:
    #     if len(shard_str) != len(total_str):
    #         return False
    
    # same prefix and total
    prefixes, _, totals = zip(*parsed)
    return len(set(prefixes)) == 1 and len(set(totals)) == 1

@torch.inference_mode
def merge_and_overwrite_lora(
    get_model_name,
    model,
    tokenizer            = None,
    save_directory       = "unsloth_finetuned_merge",
    push_to_hub          = False,
    private              = False,
    token                = None,
    save_method          = "merged_16bit",
    output_dtype         = None,
    low_disk_space_usage = False,
    use_temp_file        = False,
    cleanup_temp_file    = True,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Directly downloads 16bit original weights and merges LoRA
    inner_model = model.base_model.model if isinstance(model, PeftModel) else model
    inner_model = inner_model.base_model if hasattr(model, "base_model") else inner_model
    safetensors_list = []
    max_size_in_bytes = 0
    total_size_in_bytes = 0
    config = model.config

    for loop_iteration in range(2):
        if not isinstance(model, PeftModel):
            warnings.warn("Model is not a PeftModel (no Lora adapters detected). Skipping Merge. Please use save_pretrained() or push_to_hub() instead!")
            return None
        if loop_iteration == 0:
            # Only do on the first iteration since MXFP4 gpt-oss might already have executed this
            try:
                model_name = get_model_name(model.config._name_or_path, load_in_4bit = False)
            except:
                model_name = model.config._name_or_path
            pass
        pass

        final_model_name, is_local_path, source_info, base_model_is_quantized, quant_type = determine_base_model_source(model_name, token)
        if base_model_is_quantized and (quant_type == "nf4" or quant_type == "fp4") and save_method == "merged_16bit":
            warnings.warn("Base model should be a 16bits or mxfp4 base model for a 16bit model merge. Use `save_method=forced_merged_4bit` instead")
            return None
        if final_model_name is None:
            warnings.warn(f"Model {model_name} not found locally or on HuggingFace")
            return None
        model_name = final_model_name

        # Handle case for local model where config._name_or_path is a local os path
        # https://github.com/unslothai/unsloth/issues/2140
        is_local_path = False
        if os.path.exists(model_name) and os.path.isdir(model_name):
            is_local_path = True
            print(f"Detected local model directory: {model_name}")

            # Get safetensors files from local directory
            for file in os.listdir(model_name):
                if file.endswith(".safetensors"):
                    safetensors_list.append(file)
                    file_path = os.path.join(model_name, file)
                    file_size = os.path.getsize(file_path)
                    max_size_in_bytes = max(max_size_in_bytes, file_size)
                    total_size_in_bytes += file_size

            # Check for index file
            index_path = os.path.join(model_name, "model.safetensors.index.json")
            if os.path.exists(index_path):
                try:
                    with open(index_path, 'r', encoding = "utf-8") as f:
                        index_data = json.load(f)
                        # Extract file names from the index if available
                        if "weight_map" in index_data:
                            # Get unique filenames from weight map
                            indexed_files = set(index_data["weight_map"].values())
                            # Only use these if we didn't find files directly
                            if not safetensors_list:
                                safetensors_list = list(indexed_files)
                                # Need to compute sizes for these files
                                for file in safetensors_list:
                                    file_path = os.path.join(model_name, file)
                                    if os.path.exists(file_path):
                                        file_size = os.path.getsize(file_path)
                                        max_size_in_bytes = max(max_size_in_bytes, file_size)
                                        total_size_in_bytes += file_size
                except Exception as e:
                    print(f"Warning: Could not process index file: {e}")
            tokenizer_model_path = os.path.join(model_name, "tokenizer.model")
            if os.path.exists(tokenizer_model_path):
                os.makedirs(save_directory, exist_ok=True)
                # Copy from local
                shutil.copy2(tokenizer_model_path, os.path.join(save_directory, "tokenizer.model"))
                print(f"Copied tokenizer.model from local model directory")
        else:
            # Original HF repo logic
            try:
                file_list = HfFileSystem(token = token).ls(model_name, detail = True)
            except:
                original_model_id = get_original_model_id(model_name)
                model_name = original_model_id
                if original_model_id is None:
                    raise ValueError(f"Could not determine original model ID from {model_name}. "
                                    "If using a local model, ensure the path exists and contains safetensors files.")
                file_list = HfFileSystem(token = token).ls(model_name, detail = True)

            # Process HF file listing
            for x in file_list:
                if not x["name"].endswith(".safetensors"): continue
                safetensors_list.append(os.path.split(x["name"])[-1])
                max_size_in_bytes = max(max_size_in_bytes, x["size"])
                total_size_in_bytes += x["size"]

        if not safetensors_list:
             raise RuntimeError(f"No '.safetensors' files found for the base model: {model_name}")
        assert(max_size_in_bytes != 0 and total_size_in_bytes != 0)

        (
            username, repo_id, hf_api, token,
            output_dtype, element_size,
            lora_weights, state_dict, save_size, free,
            temp_file, save_directory, new_use_temp_file,
            low_disk_space_usage, max_shard_size_in_bytes,
        ) = prepare_saving(
            model = model,
            save_directory = save_directory,
            push_to_hub = push_to_hub,
            max_shard_size = "5GB",
            private = private,
            token = token,
            output_dtype = output_dtype,
            low_disk_space_usage = low_disk_space_usage,
            merge_into_original = True,
            min_size_in_bytes = max_size_in_bytes,
            use_temp_file = use_temp_file,
        )
        use_temp_file = use_temp_file or new_use_temp_file
        _save_dir_path = Path(save_directory)

        # Extra path for gpt-oss-20b-BF16 -> if only attention layers are provided
        all_lora_keys = "\n".join(lora_weights.keys())
        only_attention_loras = all_lora_keys.count("self_attn") == (all_lora_keys.count("\n") + 1)
        if only_attention_loras and save_method == "mxfp4" and model_name.endswith("-BF16"):
            # Check if we have a non -BF16 version which might be MXFP4
            try:
                model_name = get_model_name(model_name.removesuffix("-BF16"), load_in_4bit = False)
                print(f"Unsloth: Found MXFP4 variant = `{model_name}`")
                # Re-get all meta-data from scratch
                safetensors_list = []
                max_size_in_bytes = 0
                total_size_in_bytes = 0
                continue
            except:
                pass
        pass
        # Stop loop and continue
        break
    pass

    n_saved_modules = 0
    def upload_items(filename = None):
        extras = {"repo_id" : repo_id, "repo_type" : "model", "commit_message" : "(Trained with Unsloth)", }
        if filename is None:
            hf_api.upload_folder(folder_path = save_directory, **extras,)
        else:
            hf_api.upload_file(
                path_or_fileobj = os.path.join(save_directory, filename),
                path_in_repo = filename,
                **extras,
            )
        pass
    pass

    # Save config / generation_config via no state_dict and tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(save_directory = save_directory)
        fix_tokenizer_config_json(tokenizer, save_directory)

    # --- Handle 4-bit merging first ---
    if save_method == "merged_4bit" or save_method == "forced_merged_4bit":
        base_model = model.base_model if isinstance(model, PeftModel) else model
        print(f"Unsloth: Merging LoRA weights into 4bit model...")
        if not isinstance(model, PeftModelForCausalLM) and not isinstance(model, PeftModel):
             raise TypeError("Model must be a PeftModelForCausalLM or PeftModel for 'merged_4bit' save.")
        if not getattr(model.config, "quantization_config", None):
             raise ValueError("Model does not appear to be quantized. Cannot use 'merged_4bit'.")

        # Perform the merge
        try:
            # Use the base_model reference which points to the PeftModel's base
            merged_model = base_model.merge_and_unload()
            print(f"Unsloth: Merging finished.")
        except Exception as e:
            raise RuntimeError(f"Failed to merge LoRA weights for 4-bit save: {e}")

        # Check for skipped modules (optional but good practice)
        skipped_modules, _ = find_skipped_quantized_modules(merged_model)
        if len(skipped_modules) > 0:
            print(f"Unsloth: Found skipped modules: {skipped_modules}. Updating config.")
            # Ensure quantization_config exists before modifying
            if not hasattr(merged_model.config, "quantization_config"):
                merged_model.config.quantization_config = {} # Initialize if somehow missing
            merged_model.config.quantization_config["llm_int8_skip_modules"] = skipped_modules

        print(f"Unsloth: Saving merged 4bit model to {save_directory}...")
        try:
            merged_model.save_pretrained(save_directory = save_directory)
            print(f"Unsloth: Merged 4bit model saved.")
        except Exception as e:
             raise RuntimeError(f"Failed to save merged 4-bit model: {e}")
        fix_tokenizer_config_json(tokenizer, save_directory)

        # Upload the saved 4-bit model files
        if push_to_hub:
            upload_items() # Upload the entire directory content

        # Clean up temp file if created
        if cleanup_temp_file and temp_file is not None:
            print("Unsloth: Cleaning up temporary file...")
            try: temp_file.cleanup()
            except Exception as e: print(f"Warning: Failed to cleanup temp file: {e}")

        print("Unsloth: Merged 4bit model process completed.")
        return save_directory # <<<--- EARLY RETURN for 4-bit path
    pass

    # Default handle 16 bit merge and save/push
    # Step 1: Save base model config/architecture (no weights needed here)
    if save_method == "merged_16bit":
        config.save_pretrained(save_directory)
        _remove_quantization_config(config_path = Path(save_directory) / "config.json")
    elif save_method == "mxfp4":
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(
            model_name,
            token = None,
            trust_remote_code = False,
        )
        model_config.save_pretrained(save_directory)
        # Remove the quantization_config in the config.json file if it exists,
    # as we are exporting the model in 16-bit format.

    # Step 2: Initial upload of non-model files (config, tokenizer)
    fix_tokenizer_config_json(tokenizer, save_directory)
    if push_to_hub:
        upload_items()

    # Step 3: Conditional index handling
    import subprocess
    is_t4 = "Tesla T4" in str(torch.cuda.get_device_name(0))
    needs_splitting = should_split_shards(is_t4, config, safetensors_list) if save_method == "merged_16bit" else False
    _hf_cache_dir = _get_hf_cache_dir()
    copied_all_from_cache = False
    copied_tokenizer_model_from_cache = False
    is_hf_sharded = is_hf_sharded_safetensors(safetensors_list)
    safe_tensor_index_files = ["model.safetensors.index.json"] if (len(safetensors_list) > 1 or is_hf_sharded) else []

    # ONLY download/copy the original index if we are NOT dequantizing an MXFP4 model
    if (not (base_model_is_quantized and quant_type == "mxfp4") or (base_model_is_quantized and quant_type == "mxfp4" and save_method == "mxfp4")) and not needs_splitting:
        if is_local_path:
            os.makedirs(save_directory, exist_ok = True)
            # Copy from local
            if safe_tensor_index_files:
                local_index_path = os.path.join(model_name, "model.safetensors.index.json")
                if os.path.exists(local_index_path):
                    try:
                        shutil.copy2(local_index_path, os.path.join(save_directory, "model.safetensors.index.json"))
                    except shutil.SameFileError:
                        pass
                    except Exception as e:
                        print(f"Error copying model.safetensors.index.json: {e}")
                        raise e
        else:
            # Download from HF
            if "model.safetensors.index.json" in [f for f in safe_tensor_index_files]:
                snapshot_download(
                    repo_id = model_name,
                    local_dir = save_directory,
                    allow_patterns = ["model.safetensors.index.json"],
                    local_dir_use_symlinks = False,
                )

        if push_to_hub and safe_tensor_index_files:
            upload_items("model.safetensors.index.json")
        pass
    pass

    # Step 4 : Handle retrieval of original 16-bit shards and tokenizer.model file if exists
    if not is_local_path and _hf_cache_dir is not None:
        copied_all_from_cache = _try_copy_all_from_cache(
            repo_id = model_name,
            filenames_to_check = safetensors_list,
            target_dir_str = save_directory,
            hf_cache_dir = _hf_cache_dir,
            token = token,
        )
        copied_tokenizer_model_from_cache = _try_copy_all_from_cache(
            repo_id=model_name,
            filenames_to_check=["tokenizer.model"],
            target_dir_str=save_directory,
            hf_cache_dir=_hf_cache_dir,
            token=token,
        )

    if not copied_all_from_cache and not low_disk_space_usage and not is_local_path:
        print(f"Downloading safetensors for {model_name}...")
        snapshot_download(
            repo_id = model_name,
            local_dir = save_directory,
            allow_patterns = safe_tensor_index_files + safetensors_list,
            local_dir_use_symlinks = False,
        )

    if not copied_tokenizer_model_from_cache and not low_disk_space_usage and not is_local_path:
        print(f"Attempting to download tokenizer.model for {model_name}...")
        snapshot_download(
            repo_id = model_name,
            local_dir = save_directory,
            allow_patterns = ["tokenizer.model"],
            local_dir_use_symlinks = False,
        )

    final_safetensors_list = []

    # Step 5: Iterate through original shards, merge LoRA, and overwrite/save
    for filename in ProgressBar(safetensors_list, desc = "Unsloth: Preparing safetensor model files"):
        file_path = os.path.join(save_directory, filename)
        # Only download if we didn't get everything from cache AND this specific file doesn't exist
        # AND we're in low disk space mode
        # For local models, copy the file if needed
        if is_local_path and not os.path.exists(file_path):
            local_file_path = os.path.join(model_name, filename)
            if os.path.exists(local_file_path):
                shutil.copy2(local_file_path, file_path)
                print(f"Copied {filename} from local model directory")

        elif not copied_all_from_cache and low_disk_space_usage and not os.path.exists(file_path) and not is_local_path:
            hf_hub_download(
                repo_id = model_name,
                filename = filename,
                repo_type = "model",
                local_dir = save_directory,
            )
        pass

        if needs_splitting:
            resulting_files = split_safetensor_file(filename, save_directory, max_shard_size_gb=1.5)
        else:
            resulting_files = [filename]

        # Collect all resulting files (temp names if split, original names if not)
        final_safetensors_list.extend(resulting_files)
    pass

    if low_disk_space_usage and not is_local_path and not copied_all_from_cache:
        tokenizer_model_path = os.path.join(save_directory, "tokenizer.model")
        if not os.path.exists(tokenizer_model_path):
            try:
                hf_hub_download(
                    repo_id = model_name,
                    filename = "tokenizer.model",
                    repo_type = "model",
                    local_dir = save_directory,
                    token = token,
                )
                print("Downloaded tokenizer.model")
            except Exception as e:
                # It's OK if the file doesn't exist (not all models have it)
                print(f"Note: tokenizer.model not found (this is OK for non-SentencePiece models)")

    if needs_splitting:
        final_safetensors_list = renumber_safetensor_files(final_safetensors_list, save_directory)

    is_final_safetensors_list_sharded = is_hf_sharded_safetensors(final_safetensors_list)
    regenerate_index = ((base_model_is_quantized and quant_type == "mxfp4") or needs_splitting) and (len(final_safetensors_list) > 1 or is_final_safetensors_list_sharded) and save_method != "mxfp4"
    weight_map = {}

    for filename in ProgressBar(final_safetensors_list, desc=f'Unsloth: Merging weights into {"mxfp4" if save_method=="mxfp4" else "16bit"}'):
        n_saved_modules += _merge_and_overwrite_lora(
            save_directory = save_directory,
            filename = filename,
            lora_weights = lora_weights,
            output_dtype = output_dtype,
            model_class_name = find_lora_base_model(model).__class__.__name__,
            base_model_is_quantized = base_model_is_quantized,
            quant_type = quant_type,
            save_method = save_method,
        )
        torch.cuda.empty_cache()

        file_path = os.path.join(save_directory, filename)

        # --- NEW LOGIC: Build the weight_map BEFORE deleting the file ---
        if regenerate_index:
            # We must open the file we just created to get its tensor keys
            with safe_open(file_path, framework = "pt", device = "cpu") as f:
                for key in f.keys():
                    weight_map[key] = filename

        if low_disk_space_usage and push_to_hub:
            upload_items(filename)
            os.remove(os.path.join(save_directory, filename)) # Remove to conserve disk space
        pass
    pass

    # Step 6: Regenerate index ONLY for MXFP4 dequantization
    if regenerate_index:
        # The logic is now simpler: we just write the map we already built.
        print("Unsloth: Regenerating safetensors index for dequantized MXFP4 model...")

        index_data = {"metadata": {}, "weight_map": weight_map}
        index_path = os.path.join(save_directory, "model.safetensors.index.json")
        with open(index_path, "w", encoding = "utf-8") as f:
            json.dump(index_data, f, indent = 4)

        if push_to_hub:
            upload_items("model.safetensors.index.json")

    # Step 7: Final upload of all shards if not using low disk space mode and pushing
    if not low_disk_space_usage and push_to_hub:

        # Explicitly upload all safetensors files if not already handled
        for filename in safetensors_list:
            upload_items(filename)
        upload_items()


    # Step 7: Check for errors
    if len(lora_weights) != n_saved_modules:
        raise RuntimeError(
            f"Unsloth: Saving LoRA finetune failed since # of LoRAs = {len(lora_weights)} "\
            f"does not match # of saved modules = {n_saved_modules}. Please file a bug report!"
        )
    pass

    # --- Cleanup
    if temp_file is not None:
        try: temp_file.cleanup()
        except: pass
    pass
    # need to clean dangling files in the directory if we're pushing to hub,
    if push_to_hub and os.path.exists(save_directory):
        try:
            shutil.rmtree(save_directory)
        except Exception as e:
            print(f"Warning: Failed to remove temporary directory {save_directory}: {e}")
    pass
    print(f"Unsloth: Merge process complete. Saved to `{os.path.abspath(save_directory)}`")

    return save_directory
pass

def _try_copy_all_from_cache(
    repo_id: str,
    filenames_to_check: List[str],
    target_dir_str: str, # Expect string path for target directory
    hf_cache_dir: Optional[Path],
    token: Optional[str],
) -> bool:
    """
    Checks if ALL specified files exist in the HF cache. If yes, creates the
    target_dir_str and copies ALL files into it using os functions.
    Returns True if successful, False otherwise.
    """
    from huggingface_hub.errors import LocalEntryNotFoundError

    if not hf_cache_dir or not filenames_to_check:
        print("Skipping cache check: No cache directory or no files specified.") # Verbose
        return False

    hf_cache_dir_str = str(hf_cache_dir)
    print(f"Checking cache directory for required files...") # Verbose
    cached_paths_map = {}

    all_found = True
    for filename in filenames_to_check:
        try:
            cached_path_str = hf_hub_download(repo_id = repo_id, filename = filename, local_files_only = True)
            cached_paths_map[filename] = Path(cached_path_str) # Store Path for checking
        except LocalEntryNotFoundError:
            print(f"Cache check failed: {filename} not found in local cache.") # Verbose
            all_found = False
            break
        except Exception as check_err:
            print(f"Cache check failed: Error checking for {filename}: {check_err}.")
            all_found = False
            break

    if not all_found:
        print("Not all required files found in cache. Will proceed with downloading.") # Verbose
        return False

    try:
        # Create target directory using os.makedirs
        os.makedirs(target_dir_str, exist_ok = True)
        if not os.access(target_dir_str, os.W_OK | os.X_OK):
             raise PermissionError(f"No write/execute permission for target directory: {target_dir_str}")
    except Exception as dir_err:
        print(f"Cache copy failed: Could not create or access target directory {target_dir_str}: {dir_err}")
        return False

    all_copied = True
    for filename, cached_path in ProgressBar(cached_paths_map.items(), desc = f"Unsloth: Copying {len(filenames_to_check)} files from cache to `{target_dir_str}`"):
        try:
            # Pass string target_dir_str to copy helper
            _copy_file_from_source(cached_path, target_dir_str, filename)
        except (IOError, PermissionError, FileNotFoundError) as copy_err:
             print(f"Cache copy failed: Error copying {filename} from {cached_path} to {target_dir_str}: {copy_err}")
             all_copied = False; break
        except Exception as e:
            print(f"Cache copy failed: An unexpected error occurred copying {filename}: {e}")
            all_copied = False; break
    pass

    if all_copied:
        print(f"Successfully copied all {len(filenames_to_check)} files from cache to `{target_dir_str}`")
        return True
    else:
        print("Failed to copy one or more files from cache. Will proceed with downloading.")
        return False
pass

def _copy_file_from_source(src_path: Union[str, Path], target_dir_str: str, filename: str):
    """Copies a file from src_path to target_dir_str/filename using os.path."""
    src_path = Path(src_path) # Keep Path for source checking ease
    dst_path = os.path.join(target_dir_str, filename) # Use os.path.join for destination

    if not src_path.is_file():
        raise FileNotFoundError(f"Source {src_path} is not a valid file.")
    if not os.access(src_path, os.R_OK):
         raise PermissionError(f"No read permission for source file: {src_path}")
    # Target dir creation and permission check is handled by caller (_try_copy_all_from_cache)
    try:
        shutil.copy2(str(src_path), dst_path) # Use string paths for shutil
    except Exception as e:
        raise IOError(f"Failed to copy {src_path} to {dst_path}: {e}") from e
pass

def _get_hf_cache_dir() -> Optional[Path]:
    """Determines the Hugging Face Hub cache directory."""
    potential_paths = []
    if "HF_HUB_CACHE" in os.environ:
        potential_paths.append(Path(os.environ["HF_HUB_CACHE"]))
    if "HF_HOME" in os.environ:
        potential_paths.append(Path(os.environ["HF_HOME"]) / "hub")
    potential_paths.append(Path.home() / ".cache" / "huggingface" / "hub")

    for cache_dir in potential_paths:
        try:
            # 1. Check if it exists and is a directory
            if cache_dir.is_dir():
                # 2. Check if we have read/write/execute access
                # Need W/X for potential lock files or internal operations by huggingface_hub
                if os.access(cache_dir, os.R_OK | os.W_OK | os.X_OK):
                    print(f"Found HuggingFace hub cache directory: {cache_dir.resolve()}")
                    return cache_dir.resolve() # Return absolute path
                else:
                    print(f"Warning: Found cache directory {cache_dir}, but lack R/W/X permissions. Cannot use cache.")
                    # Don't check other paths if we found the prioritized one but lack permissions
                    return None
            # If it exists but is not a dir, it's problematic, stop checking.
            elif cache_dir.exists():
                 print(f"Warning: Path {cache_dir} exists but is not a directory. Cannot use cache.")
                 return None
            # If it doesn't exist, continue to check the next potential path

        except Exception as e:
            # Handle potential issues like symlink loops, permissions errors during check
            print(f"Warning: Error accessing potential cache path {cache_dir}: {e}. Checking next option.")
            continue # Try the next path

    # If none of the paths worked
    print("No existing and accessible Hugging Face cache directory found.")
    return None
pass


_PUSHING_CODE = \
"""
PushToHubMixin._upload_modified_files(
    PushToHubMixin,
    working_dir = save_directory,
    repo_id = '{repo_id}',
    files_timestamps = files_timestamps,
    commit_message = "Upload Unsloth finetuned model",
    token = token,
    create_pr = False,
    revision = {revision},
    commit_description = "Upload Unsloth finetuned model",
)
if {use_temp_file} and temp_file is not None: temp_file.cleanup()
else:
    shutil.rmtree(save_directory)
    os.makedirs(save_directory, exist_ok = True)
if {use_temp_file}:
    temp_file = tempfile.TemporaryDirectory(ignore_cleanup_errors = True)
    save_directory = temp_file.name
files_timestamps = PushToHubMixin._get_files_timestamps(PushToHubMixin, save_directory)
"""

def incremental_save_pretrained(
    save_pretrained,
    low_disk_space_usage = True,
    use_temp_file = True,
    repo_id = "",
    revision = None,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Move file timestamps out
    makedir = re.search(r"os\.makedirs\(save_directory.+?\n", save_pretrained)
    assert(makedir is not None)
    span = makedir.span(0)
    save_pretrained = save_pretrained[:span[1]-1] + \
        "; files_timestamps = self._get_files_timestamps(save_directory); temp_file = None;\n" + \
        save_pretrained[span[1]:]
    pass

    # Find the main loop
    if "for shard_file, tensors in filename_to_tensors" not in save_pretrained:
        raise RuntimeError("Unsloth: Failed to find `for shard_file, tensors in filename_to_tensors`")
    for_loop = re.search(
        r"for shard_file, tensors in filename_to_tensors\:"\
        r".*?[\n]{1,}[ ]{4}[a-zA-Z0-9\_\#]",
        save_pretrained,
        flags = re.DOTALL | re.MULTILINE,
    )
    assert(for_loop is not None)

    span = for_loop.span(0)
    for_loop = save_pretrained[max(span[0], span[1]-8) : span[1]-1]
    where = re.search(r"[\n]{1,}", for_loop[::-1]).span(0)[0]
    for_loop = save_pretrained[span[0] : span[1]-where-1]
    spaces = len(re.findall(r"\n([ ]{4,})", for_loop)[0])

    first_newline = for_loop.find("\n") + 1
    for_loop = for_loop.rstrip()

    if low_disk_space_usage:
        new_for_loop = for_loop[:first_newline] + \
            for_loop[first_newline:] + \
            " "*spaces + \
            re.sub(r"[ ]{8,}", "",
                   _PUSHING_CODE.format(
                       repo_id = repo_id,
                       revision = revision,
                       use_temp_file = use_temp_file,
                    ).rstrip()
            ).replace("\n", "\n" + " "*spaces)
    else:
        new_for_loop = for_loop
    pass

    new_for_loop = new_for_loop + \
        "\n" + \
        " "*spaces + \
        "for tensor in shard:\n" + \
        " "*(spaces+4) + \
        "if tensor in DEQUANTIZED_KEYS: shard[tensor] = None\n"

    if low_disk_space_usage:
        new_for_loop = new_for_loop + \
            "\n" + \
            " "*(spaces-4) + \
            f"if {use_temp_file}:\n" + \
            " "*(spaces) + \
            "temp_file = tempfile.TemporaryDirectory(ignore_cleanup_errors = True)\n" + \
            " "*(spaces) + \
            "save_directory = temp_file.name\n" + \
            " "*(spaces) + \
            f"repo_id = '{repo_id}'\n"
    pass
    save_pretrained = save_pretrained.replace(for_loop, new_for_loop)

    if not low_disk_space_usage:
        save_pretrained = save_pretrained.replace(
            "for shard_file, tensors in filename_to_tensors",
            "for shard_file, tensors in ProgressBar(filename_to_tensors, desc = 'Unsloth: Saving ' + str(len(filename_to_tensors)) + ' safetensor(s)')",
            1,
        )
    pass
    return save_pretrained
pass


def merge_and_dequantize_lora(
    model,
    tokenizer            = None,
    save_directory       = "unsloth_finetuned_merge",
    push_to_hub          = False,
    max_shard_size       = "5GB",
    safe_serialization   = True,
    token                = None,
    private              = False,
    revision             = None,
    output_dtype         = None,
    low_disk_space_usage = False,
    use_temp_file        = False,
    **kwargs,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Dequantizes model to 16bit weights and merges LoRA
    inner_model = model.base_model.model if isinstance(model, PeftModelForCausalLM) else model
    inner_model = inner_model.base_model if hasattr(model, "base_model") else inner_model

    (
        username, repo_id, hf_api, token,
        output_dtype, element_size,
        lora_weights, state_dict, save_size, free,
        temp_file, save_directory, use_temp_file,
        low_disk_space_usage, max_shard_size_in_bytes,
    ) = prepare_saving(
        model = model,
        save_directory = save_directory,
        push_to_hub = push_to_hub,
        max_shard_size = max_shard_size,
        private = private,
        token = token,
        output_dtype = output_dtype,
        low_disk_space_usage = low_disk_space_usage,
        merge_into_original = False,
        min_size_in_bytes = 100_000_000, # 100MB default
        use_temp_file = use_temp_file,
    )

    import transformers.modeling_utils
    save_pretrained = inspect.getsource(transformers.modeling_utils.PreTrainedModel.save_pretrained)
    spaces = save_pretrained.find("def")
    save_pretrained = save_pretrained.split("\n")
    save_pretrained = "\n".join(x[spaces:] for x in save_pretrained)

    # Now patch for incremental pushing to hub
    if push_to_hub:
        save_pretrained = incremental_save_pretrained(
            save_pretrained = save_pretrained,
            low_disk_space_usage = low_disk_space_usage,
            use_temp_file = use_temp_file,
            repo_id = repo_id,
            revision = revision,
        )
    pass

    functions = dir(transformers.modeling_utils)
    # functions = [x for x in functions if (f"{x}." in save_pretrained or f"{x}(" in save_pretrained) and x != "PreTrainedModel"]
    exec(f"from transformers.modeling_utils import ({', '.join(functions)})", locals(), globals())

    replace_state_dict = f"""
    DEQUANTIZED_KEYS = []

    def merge_lora_weights(state_dict, name):
        x = state_dict[name]
        if type(x) is LoraStats:
            DEQUANTIZED_KEYS.append(name)
            W = dequantize_module_weight(x.module)
            W = _merge_lora(W, x, name)
            x = W.to(device = 'cpu', dtype = {str(output_dtype)}, non_blocking = True)
        # Remove memory leak
        state_dict[name] = None
        return x
    pass
    state_dict_split = split_state_dict_into_shards_factory(
        state_dict,
        max_shard_size   = {max_shard_size_in_bytes},
        filename_pattern = filename_pattern,
        get_storage_size = functools.partial(get_torch_storage_size_new, element_size = {element_size}),
        get_storage_id   = get_torch_storage_id_new,
    )
    """
    left  = save_pretrained.find("state_dict_split = split_torch_state_dict_into_shards")
    if left == -1: raise RuntimeError("Unsloth: Failed to find `state_dict_split`")
    right = save_pretrained.find(")", left) + 1
    save_pretrained = save_pretrained[:left] + replace_state_dict + save_pretrained[right:]

    if "state_dict[tensor].contiguous()" not in save_pretrained:
        raise RuntimeError("Unsloth: Failed to find `state_dict[tensor].contiguous()`")
    save_pretrained = save_pretrained.replace(
        "state_dict[tensor].contiguous()",
        "merge_lora_weights(state_dict, tensor).contiguous()",
        1,
    )

    if "def save_pretrained" not in save_pretrained:
        raise RuntimeError("Unsloth: Failed to find `def save_pretrained`")
    save_pretrained = save_pretrained.replace(
        "def save_pretrained",
        "def save_pretrained_dequantized",
        1,
    )

    functions = {}
    exec(save_pretrained, globals(), functions)
    save_pretrained_dequantized = functions["save_pretrained_dequantized"]
    save_pretrained_dequantized = torch.inference_mode(save_pretrained_dequantized)

    files_timestamps = PushToHubMixin._get_files_timestamps(
        PushToHubMixin,
        save_directory,
    )
    save_pretrained_dequantized(
        inner_model,
        save_directory     = save_directory,
        push_to_hub        = False,
        max_shard_size     = max_shard_size_in_bytes,
        safe_serialization = safe_serialization,
        token              = token,
        private            = private,
        state_dict         = state_dict,
        **kwargs,
    )

    # Save tokenizer
    if tokenizer is not None: tokenizer.save_pretrained(save_directory = save_directory,)

    if push_to_hub:
        commit = PushToHubMixin._upload_modified_files(
            PushToHubMixin,
            working_dir = save_directory,
            repo_id = repo_id,
            files_timestamps = files_timestamps,
            commit_message = "Upload Unsloth finetuned model",
            token = token,
            create_pr = False,
            revision = revision,
            commit_description = "Upload Unsloth finetuned model",
        )
        print(f"Unsloth: Uploaded model to https://huggingface.co/{repo_id}")
        return commit
    pass
    if temp_file is not None:
        try: temp_file.cleanup()
        except: pass
    pass
pass

def get_original_model_id(local_path: str):
    import json
    import os

    config_path = os.path.join(local_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding = "utf-8") as f:
            config = json.load(f)

        # Check for _name_or_path that's not a local path
        # When we load using AutoConfig, the _name_or_path changed into the local path instead
        if "_name_or_path" in config:
            return config["_name_or_path"]

    config_path = os.path.join(local_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding = "utf-8") as f:
            config = json.load(f)

        if "base_model_name_or_path" in config:
            return config["base_model_name_or_path"]

    return None
pass

def _get_checkpoint_conversion_mapping(model_class_name):
    """Get the checkpoint conversion mapping for a specific model class"""
    try:
        # Dynamically import the model class
        module = __import__('transformers', fromlist=[model_class_name])
        model_class = getattr(module, model_class_name)
        return getattr(model_class, '_checkpoint_conversion_mapping', {})  # Returns {} if attribute doesn't exist
    except (ImportError, AttributeError):
        return {}
pass


def detect_keys_format(keys_to_check, forward_mapping):
    if not forward_mapping:
        return "new"

    count_matches_old_pattern = 0
    count_matches_new_pattern = 0

    # Compile regex patterns for efficiency if called multiple times with same mapping (though here it's per call)
    old_regex_compiled = [re.compile(p) for p in forward_mapping.keys()]
    # For new patterns (values of forward_mapping), treat them as literal prefixes to match
    new_regex_compiled = [re.compile(r"^" + re.escape(val)) for val in forward_mapping.values()]

    for key in keys_to_check:
        if not isinstance(key, str): continue

        # A key is "new" if it starts with one of the new_prefix_strings (values of forward_mapping)
        # A key is "old" if it matches one of the old_pattern_regex (keys of forward_mapping)
        #   AND it does NOT start with one of the new_prefix_strings (to avoid double counting if patterns overlap badly)

        matched_new = any(r.match(key) for r in new_regex_compiled)
        matched_old = any(r.match(key) for r in old_regex_compiled)

        if matched_new:
            count_matches_new_pattern += 1
        elif matched_old: # Only count as old if not already counted as new
            count_matches_old_pattern += 1

    # Decision logic
    if count_matches_new_pattern > 0 and count_matches_old_pattern == 0: return "new"
    if count_matches_old_pattern > 0 and count_matches_new_pattern == 0: return "old"

    # If mixed,
    if count_matches_new_pattern > count_matches_old_pattern: return "new"
    if count_matches_old_pattern > count_matches_new_pattern: return "old"

    return "new" # Default, assuming most models/keys will be in the "new" (current HF) format.
pass

def _convert_lora_keys_to_safetensor_format(
    lora_weights,        # Global dict of LoraStats objects
    safetensor_keys,     # List of keys from the CURRENT shard
    model_class_name="PreTrainedModel" # The actual model instance (e.g. Qwen2VLForConditionalGeneration)
):
    import re

    # Get the forward mapping from the model class itself
    forward_mapping = _get_checkpoint_conversion_mapping(model_class_name)

    if not forward_mapping:
        return defaultdict(lora_weights.default_factory, lora_weights)

    # Create reverse mapping
    reverse_mapping = {}
    for pattern, replacement in forward_mapping.items():
        reverse_mapping[replacement] = pattern
    # Determine formats
    lora_key_format_assumed = "new"
    shard_key_format = detect_keys_format(safetensor_keys, forward_mapping)

    converted_lora_weights_output = defaultdict(lora_weights.default_factory)
    conversion_applied_count = 0

    for lora_key_module_name, lora_stats in lora_weights.items():
        if not isinstance(lora_key_module_name, str):
            converted_lora_weights_output[lora_key_module_name] = lora_stats
            continue

        converted_key_for_lookup = lora_key_module_name
        applied_conversion_for_this_key = False

        if lora_key_format_assumed == "new" and shard_key_format == "old":
            # LoRA keys are new format, shard is old style -> convert LoRA key to old style
            # Use reverse mapping
            for pattern, replacement in reverse_mapping.items():
                replacement = re.sub(r"\^?([^(?]+).*", r"\1", replacement.lstrip("^"))
                temp_key, n_replace = re.subn(pattern, replacement, converted_key_for_lookup)
                if n_replace > 0:
                    converted_key_for_lookup = temp_key
                    applied_conversion_for_this_key = True
                    break

        elif lora_key_format_assumed == "old" and shard_key_format == "new":
            # LoRA keys are old format, shard is new format -> convert LoRA key to new style
            for pattern, replacement in forward_mapping.items():
                temp_key, n_replace = re.subn(pattern, replacement, converted_key_for_lookup)
                if n_replace > 0:
                    converted_key_for_lookup = temp_key
                    applied_conversion_for_this_key = True
                    break

        if applied_conversion_for_this_key:
            conversion_applied_count += 1

        converted_lora_weights_output[converted_key_for_lookup] = lora_stats
    return converted_lora_weights_output
pass

def find_lora_base_model(model_to_inspect):
    current = model_to_inspect
    if hasattr(current, "base_model"):
        current = current.base_model
    if hasattr(current, "model"):
        current = current.model
    return current
pass

def check_hf_model_exists(model_name, token=None):
    """Check if model exists on HuggingFace"""
    try:
        file_list = HfFileSystem(token=token).ls(model_name, detail=True)
        return any(x["name"].endswith(".safetensors") for x in file_list)
    except:
        return False
pass

def check_local_model_exists(model_path):
    """
    Check if model exists locally with case insensitive naming patterns.
    Returns the actual path if found, None otherwise.
    """

    def has_safetensors(directory):
        """Check if directory contains safetensors files"""
        if not os.path.exists(directory) or not os.path.isdir(directory):
            return False
        try:
            for file in os.listdir(directory):
                if file.endswith(".safetensors"):
                    return True
            return False
        except (OSError, PermissionError):
            return False

    def find_case_insensitive_path(target_path):
        """Find a path that matches case-insensitively"""
        if os.path.exists(target_path):
            return target_path

        # Split path into components
        parts = target_path.split(os.sep)
        current_path = ""

        for i, part in enumerate(parts):
            if i == 0:
                # Handle first part (could be relative or absolute)
                if part == "":  # absolute path starting with /
                    current_path = os.sep
                    continue
                elif part == ".":
                    current_path = "."
                else:
                    current_path = part
            else:
                current_path = os.path.join(current_path, part)

            # If this exact path exists, continue
            if os.path.exists(current_path):
                continue

            # Try to find case-insensitive match
            parent_path = os.path.dirname(current_path) if i > 0 else "."
            target_name = os.path.basename(current_path).lower()

            if not os.path.exists(parent_path):
                return None

            try:
                found_match = False
                for item in os.listdir(parent_path):
                    if item.lower() == target_name:
                        current_path = os.path.join(parent_path, item)
                        found_match = True
                        break

                if not found_match:
                    return None
            except (OSError, PermissionError):
                return None

        return current_path if os.path.exists(current_path) else None

    # List of path patterns to check
    paths_to_check = []

    # 1. Exact path as given
    paths_to_check.append(model_path)

    # 2. Case-insensitive version of full path
    case_insensitive_full = find_case_insensitive_path(model_path)
    if case_insensitive_full:
        paths_to_check.append(case_insensitive_full)

    # 3. If path contains "/", also check just the model name part
    if "/" in model_path:
        model_name = model_path.split("/")[-1]  # Get part after last "/"

        # Exact model name
        paths_to_check.append(model_name)

        # Case-insensitive model name in current directory
        try:
            for item in os.listdir("."):
                if item.lower() == model_name.lower():
                    paths_to_check.append(item)
                    break
        except (OSError, PermissionError):
            pass

    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for path in paths_to_check:
        if path and path not in seen:
            seen.add(path)
            unique_paths.append(path)

    # Check each path and verify it contains safetensors
    for path in unique_paths:
        if has_safetensors(path):
            return os.path.abspath(path)  # Return absolute path

    return None
pass

def check_model_quantization_status(model_name_or_path, token=None):
    """Check if a model is quantized (works for both HF and local)"""
    config = None
    # Local path
    if os.path.exists(model_name_or_path) and os.path.isdir(model_name_or_path):
        config_path = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding = "utf-8") as f:
                    config = json.load(f)
            except:
                pass
    # HF repo
    else:
        try:
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(
                repo_id = model_name_or_path,
                filename = "config.json",
                cache_dir = None,
                token = token
            )
            with open(config_path, 'r', encoding="utf-8") as f:
                config = json.load(f)
        except:
            pass

    if config and "quantization_config" in config:
        quant_config = config["quantization_config"]

        # Case 2: Check for MXFP4 format first (more specific)
        # We assume the Mxfp4Config serializes with a "quant_method": "mxfp4" key.
        if isinstance(quant_config, dict) and quant_config.get("quant_method") == "mxfp4":
            return (True, "mxfp4")

        # Case 1: Fallback to existing logic for bitsandbytes
        elif isinstance(quant_config, dict):
            is_quantized = quant_config.get("load_in_4bit", False)
            quant_type = quant_config.get("bnb_4bit_quant_type", None)
            if is_quantized:
                # Return the specific type if available, otherwise a generic "bitsandbytes"
                return (True, quant_type if quant_type else "bitsandbytes")

    return (False, None)
pass

def determine_base_model_source(model_name, token=None):
    """
    Determine the best source for base model using branched logic
    Returns: (final_model_name, is_local_path, source_info, is_quantized, quant_type)
    """

    # Check availability
    hf_exists = check_hf_model_exists(model_name, token)
    local_path = check_local_model_exists(model_name)

    # Get quantization status for both if they exist
    hf_is_quantized, hf_quant_type = None, None
    local_is_quantized, local_quant_type = None, None

    if hf_exists:
        hf_is_quantized, hf_quant_type = check_model_quantization_status(model_name, token)

    if local_path:
        local_is_quantized, local_quant_type = check_model_quantization_status(local_path)

    # Priority 1: Local unquantized
    if local_path and not local_is_quantized:
        return (local_path, True, "local_unquantized", False, None)

    # Priority 2: Local mxfp4
    if local_path and local_is_quantized and local_quant_type == "mxfp4":  # local_quant_type == "mxfp4"
        return (local_path, True, "local_mxfp4", True, "mxfp4")

    # Priority 3: HF unquantized
    if hf_exists and not hf_is_quantized:
        return (model_name, False, "HF_unquantized", False, None)

    # Priority 4: HF quantized (covers both "both quantized" and "just HF quantized")
    if hf_exists and hf_is_quantized:
        return (model_name, False, f"HF_{hf_quant_type}", True, hf_quant_type)

    # Priority 5: Local other quantization
    if local_path and local_is_quantized:
        return (local_path, True, f"local_{local_quant_type}", True, local_quant_type)

    # Priority 6: Nothing suitable found
    return (None, False, "", False, None)
pass

def get_memory_stats():
    """Get current memory statistics for CPU and GPU"""
    stats = {}
    import psutil

    # CPU Memory
    cpu_mem = psutil.virtual_memory()
    stats['cpu'] = {
        'total': cpu_mem.total,
        'available': cpu_mem.available,
        'used': cpu_mem.used,
        'percent': cpu_mem.percent,
        'free': cpu_mem.available,  # Available is more accurate than free
    }

    # GPU Memory (for each GPU)
    stats['gpus'] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_mem = torch.cuda.mem_get_info(i)
            total = gpu_mem[1]
            free = gpu_mem[0]
            stats['gpus'].append({
                'device_id': i,
                'name': torch.cuda.get_device_name(i),
                'total': total,
                'free': free,
                'used': total - free,
                'percent': ((total - free) / total) * 100 if total > 0 else 0
            })

    return stats
pass

def format_bytes(bytes_value):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"
pass

def calculate_combined_score(speed_score, chunk_size):
    # Normalize chunk size to 0-1 scale (assuming max reasonable chunk is 100M)
    chunk_factor = min(1.0, chunk_size / (100 * 1024 * 1024))
    # Weight: 60% device speed, 40% chunk efficiency
    return speed_score * 0.6 + chunk_factor * 10 * 0.4  # Scale chunk factor to match speed range
pass

def _choose_mxfp4_processing_strategy(blocks_tensor, scales_tensor):
    """
    Choose optimal device and chunk size for mxfp4 dequantization based on available memory.
    """
    import math

    # Calculate tensor dimensions
    *prefix_shape, G, B = blocks_tensor.shape
    rows_total = math.prod(prefix_shape) * G

    # Estimate memory requirements
    #base_memory_per_row = B * 21 if B else 128 * 21
    base_memory_per_row = B * 35 if B else 128 * 35
    input_size = blocks_tensor.numel() + scales_tensor.numel() * 4
    output_size = rows_total * B * 2 * 2
    persistent_memory = input_size + output_size

    # Device-specific safety factors
    GPU_SAFETY_FACTOR = 0.75  # GPUs can handle higher utilization
    CPU_SAFETY_FACTOR = 0.75  # CPUs need more headroom for OS and other processes

    def calculate_safe_usable_memory(free_memory, safety_factor):
        # Option 1: What we can use from reported free memory (accounting for fragmentation)
        usable_from_free = free_memory * safety_factor

        return usable_from_free


    def calculate_optimal_chunk_size(safe_usable_memory):
        """Calculate the largest chunk size that fits in the safe usable memory"""
        temp_memory_budget = safe_usable_memory - persistent_memory
        if temp_memory_budget <= 0:
            return None

        max_chunk_from_memory = int(temp_memory_budget // base_memory_per_row)
        optimal_chunk = min(rows_total, max_chunk_from_memory)

        if optimal_chunk < 1024:
            return None

        return optimal_chunk

    stats = get_memory_stats()
    suitable_strategies = []

    # Check GPU strategies first (preferred for speed)
    for gpu in stats['gpus']:
        safe_usable_memory = calculate_safe_usable_memory(
            free_memory=gpu['free'],
            safety_factor=GPU_SAFETY_FACTOR,
        )
        chunk_size = calculate_optimal_chunk_size(safe_usable_memory)

        if chunk_size:
            temp_memory = min(chunk_size, rows_total) * base_memory_per_row
            total_memory_needed = persistent_memory + temp_memory

            combined_score = calculate_combined_score(3.0, chunk_size)

            suitable_strategies.append({
                'device_type': 'cuda',
                'device_id': gpu['device_id'],
                'rows_per_chunk': chunk_size,
                'available_memory': gpu['free'] * GPU_SAFETY_FACTOR,
                'total_memory': gpu['total'],
                'safe_usable_memory': safe_usable_memory,
                'needed_memory': total_memory_needed,
                'speed_score': 3.0,
                'efficiency_score': chunk_size,
                'safety_factor': GPU_SAFETY_FACTOR,
                'memory_utilization': total_memory_needed / safe_usable_memory,
                'combined_score': combined_score,
            })

    # Check CPU strategy
    cpu_safe_usable_memory = calculate_safe_usable_memory(
        free_memory=stats['cpu']['available'],
        safety_factor=CPU_SAFETY_FACTOR,
    )
    cpu_chunk_size = calculate_optimal_chunk_size(cpu_safe_usable_memory)

    if cpu_chunk_size:
        temp_memory = min(cpu_chunk_size, rows_total) * base_memory_per_row
        total_memory_needed = persistent_memory + temp_memory
        combined_score = calculate_combined_score(1.0, cpu_chunk_size)  # For CPU
        suitable_strategies.append({
            'device_type': 'cpu',
            'device_id': None,
            'rows_per_chunk': cpu_chunk_size,
            'available_memory': stats['cpu']['available'] * CPU_SAFETY_FACTOR,
            'total_memory': stats['cpu']['total'],
            'safe_usable_memory': cpu_safe_usable_memory,
            'needed_memory': total_memory_needed,
            'speed_score': 1.0,
            'efficiency_score': cpu_chunk_size,
            'safety_factor': CPU_SAFETY_FACTOR,
            'fragmentation_factor': 1.0,
            'memory_utilization': total_memory_needed / cpu_safe_usable_memory,
            'combined_score': combined_score,
        })

    if suitable_strategies:

        # Sort by combined score
        suitable_strategies.sort(key=lambda x: x['combined_score'], reverse=True)

        best = suitable_strategies[0]

        if UNSLOTH_ENABLE_LOGGING:
            logger.info(
                f"[MXFP4] Selected {best['device_type']}:{best['device_id'] or ''} "
                f"with {best['rows_per_chunk']:,} rows per chunk "
                f"(safety factor: {best['safety_factor']:.0%}, "
                f"safe memory utilization: {best['memory_utilization']:.1%}) "
                f"- Need: {format_bytes(best['needed_memory'])}, "
                f"Available: {format_bytes(best['available_memory'])}"
            )

        return (best['device_type'], best['device_id'], best['rows_per_chunk'])

    # Fallback: find device with most memory and use minimal chunk
    fallback_options = []

    # Add CPU fallback
    fallback_options.append({
        'device_type': 'cpu',
        'device_id': None,
        'available': stats['cpu']['available'] * CPU_SAFETY_FACTOR,
        'total_available': stats['cpu']['available']
    })

    # Add GPU fallbacks
    for gpu in stats['gpus']:
        fallback_options.append({
            'device_type': 'cuda',
            'device_id': gpu['device_id'],
            'available': gpu['free'] * GPU_SAFETY_FACTOR,
            'total_available': gpu['free']
        })

    # Sort by available memory (after safety factor)
    fallback_options.sort(key=lambda x: x['available'], reverse=True)
    best_fallback = fallback_options[0]

    # Calculate minimal safe chunk size for fallback
    remaining_memory = best_fallback['available'] - persistent_memory
    if remaining_memory > 0:
        fallback_chunk_size = max(1024, min(8192, int(remaining_memory // base_memory_per_row), rows_total))
    else:
        fallback_chunk_size = min(1024, rows_total)

    warnings.warn(
        f"[MXFP4] Insufficient memory for optimal processing on any device. "
        f"Using {best_fallback['device_type']}:{best_fallback['device_id'] or ''} "
        f"with minimal chunks ({fallback_chunk_size:,}). "
        f"Available: {format_bytes(best_fallback['total_available'])}, "
        f"Required: {format_bytes(persistent_memory)}. "
        f"Processing will be slow."
    )

    return (best_fallback['device_type'], best_fallback['device_id'], fallback_chunk_size)
pass

def should_split_shards(is_t4, model_config, safetensors_list):
    """Determine if we need to split shards based on T4 and GPT-OSS conditions."""
    if not is_t4:
        return False

    if hasattr(model_config, 'model_type'):
        if model_config.model_type.lower() == 'gpt_oss':
            return True

    return False
pass

def split_safetensor_file(filename, save_directory, max_shard_size_gb=2):
    """Split a file if needed, using temporary names to avoid messy numbering."""
    file_path = os.path.join(save_directory, filename)

    if not os.path.exists(file_path):
        return [filename]

    file_size = os.path.getsize(file_path)
    max_shard_size_bytes = max_shard_size_gb * 1024 * 1024 * 1024

    if file_size <= max_shard_size_bytes:
        return [filename]  # No splitting needed

    print(f"Splitting {filename} (size: {file_size / (1024**3):.2f} GB)...")

    try:
        # Split into shards
        shards = split_safetensors_to_shards(file_path, max_shard_size_gb)

        # Create temporary filenames to avoid messy nested numbering
        import uuid
        temp_base = str(uuid.uuid4())[:8]  # Short unique ID
        temp_filenames = []

        for i, shard in enumerate(shards):
            temp_filename = f"temp_split_{temp_base}_{i:03d}.safetensors"
            temp_file_path = os.path.join(save_directory, temp_filename)
            save_file(shard, temp_file_path, metadata={"format": "pt"})
            temp_filenames.append(temp_filename)

            shard_size = sum(tensor.numel() * tensor.element_size() for tensor in shard.values())
            if UNSLOTH_ENABLE_LOGGING:
                logger.info(f"Created temp chunk: {temp_filename} (size: {shard_size / (1024**3):.2f} GB)")

        # Remove original file
        os.remove(file_path)
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(f"Removed original file: {filename}")

        return temp_filenames

    except Exception as e:
        print(f"Error splitting {filename}: {e}")
        return [filename]
pass

def renumber_safetensor_files(file_list, save_directory):
    """Renumber all files with clean sequential names."""
    if len(file_list) <= 1:
        # Single file - rename to model.safetensors
        if len(file_list) == 1 and file_list[0] != "model.safetensors":
            old_path = os.path.join(save_directory, file_list[0])
            new_path = os.path.join(save_directory, "model.safetensors")
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
                if UNSLOTH_ENABLE_LOGGING:
                    logger.info(f"Renamed {file_list[0]} -> model.safetensors")
            return ["model.safetensors"]
        return file_list

    # Multiple files - use clean numbering
    total_files = len(file_list)
    clean_names = [f"model-{i+1:05d}-of-{total_files:05d}.safetensors" for i in range(total_files)]

    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Renumbering safetensor files with sequential numbering...")

    # Create mapping of old -> new names
    rename_pairs = list(zip(file_list, clean_names))

    # Rename files (handle potential conflicts with temp names)
    for old_name, new_name in rename_pairs:
        old_path = os.path.join(save_directory, old_name)
        new_path = os.path.join(save_directory, new_name)

        if os.path.exists(old_path) and old_name != new_name:
            # Use temp name to avoid conflicts
            temp_path = os.path.join(save_directory, f"renaming_{new_name}")
            os.rename(old_path, temp_path)
            os.rename(temp_path, new_path)
            if UNSLOTH_ENABLE_LOGGING:
                logger.info(f"Renamed {old_name} -> {new_name}")

    return clean_names
pass

def split_safetensors_to_shards(file_path, max_shard_size_gb=2):
    """Split a safetensors file into smaller shards."""
    max_shard_size = max_shard_size_gb * 1024 * 1024 * 1024

    with safe_open(file_path, framework="pt", device="cpu") as f:
        all_tensors = {key: f.get_tensor(key) for key in f.keys()}

    shards = []
    current_shard = OrderedDict()
    current_size = 0

    for key, tensor in all_tensors.items():
        tensor_size = tensor.numel() * tensor.element_size()

        if current_size + tensor_size > max_shard_size and current_shard:
            shards.append(current_shard)
            current_shard = OrderedDict()
            current_size = 0

        current_shard[key] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    return shards
pass

def _write_tensor_direct_torch(mm, header_metadata, length_of_header, output_key, tensor, output_dtype):
    """
    Write tensor directly to memory-mapped file using pure PyTorch operations
    """
    try:
        if output_key not in header_metadata:
            return False

        key_metadata = header_metadata[output_key]
        index_L, index_R = key_metadata["data_offsets"]

        # Adjust for header offset
        index_L += 8 + length_of_header
        index_R += 8 + length_of_header

        expected_size = index_R - index_L

        # Convert tensor to the correct format using pure PyTorch
        tensor_formatted = tensor.to(output_dtype).contiguous().cpu()

        # Get tensor data as bytes using PyTorch's storage
        tensor_bytes = tensor_formatted.untyped_storage().nbytes()

        if tensor_bytes != expected_size:
            if UNSLOTH_ENABLE_LOGGING:
                logger.warning(f"Size mismatch for {output_key}: expected {expected_size}, got {tensor_bytes}")
            return False

        # Use PyTorch's internal byte representation directly
        # This avoids numpy conversion and preserves exact format
        tensor_view = tensor_formatted.view(torch.uint8)

        # Convert to bytes using PyTorch's .data_ptr() and ctypes
        import ctypes
        data_ptr = tensor_view.data_ptr()
        byte_data = (ctypes.c_ubyte * tensor_view.numel()).from_address(data_ptr)

        # Write directly to memory map
        mm[index_L:index_R] = bytes(byte_data)

        # Clear memory
        del data_ptr
        del tensor_view
        del tensor_formatted
        del tensor

        return True

    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(f"Direct tensor write failed for {output_key}: {e}")
        return False
pass
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

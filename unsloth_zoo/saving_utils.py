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
from .device_type import DEVICE_TYPE, DEVICE_TYPE_TORCH, device_empty_cache
from .temporary_patches.common import UNSLOTH_ENABLE_LOGGING, logger
from collections import defaultdict

try:
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors, convert_moe_packed_tensors_cpu
except (ImportError, ModuleNotFoundError):
    # Absent unless mxfp4 is in use
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

    # If base_model is a local path, resolve to the original model ID so the
    # card doesn't reference a local dir (which fails HF validation).
    base_model = model.config._name_or_path
    if os.path.exists(base_model) and os.path.isdir(base_model):
        original_model_id = get_original_model_id(base_model)
        if original_model_id is not None and not os.path.exists(original_model_id):
            base_model = original_model_id
        else:
            base_model = repo_id  # fall back to a generic, valid description

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

# Flush the allocator only after a large-tensor merge (embed / lm_head / fused
# experts); a per-key empty_cache/synchronize over every key is wasted GPU stall.
_EMPTY_CACHE_BYTES_THRESHOLD = 256 * 1024 * 1024


@functools.lru_cache(maxsize = 1)
def _active_merge_device():
    """Pick the active accelerator family for LoRA merge math, cached.

    Hardcoding "cuda" breaks ROCm/XPU/MPS; DEVICE_TYPE_TORCH drops MPS (needed
    by the MLX backend's on-host merge). So probe at first call instead.
    """
    if torch.cuda.is_available():
        return "cuda"  # PyTorch ROCm aliases the cuda API, so this also covers HIP
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
pass

# Architectures whose bnb-4bit merged_16bit export must fold the adapter onto the DEQUANTIZED
# 4bit base dequant(W4), not the downloaded 16bit base W16 (see `_merge_lora`). Deliberately
# narrow: the dequant base bakes quant noise into the checkpoint and regresses ordinary models
# (Qwen2.5 perplexity +~20%), so only models where the quant error swamps the fine-tune delta
# (huge base-weight norms from small MuP multipliers over a deep hybrid stack) belong here.
_DEQUANT_MERGE_BASE_MODEL_TYPES = frozenset({"falcon_h1"})


def _model_type_needs_dequant_merge_base(model) -> bool:
    """True iff `model`'s architecture is one whose on-the-fly bnb-4bit merged_16bit
    export must use dequant(W4) as the LoRA merge base instead of the downloaded W16."""
    try:
        cfg = getattr(model, "config", None)
        mt = (getattr(cfg, "model_type", "") or "").lower()
    except Exception:
        return False
    return mt in _DEQUANT_MERGE_BASE_MODEL_TYPES


def _is_bnb_4bit_base(module):
    # True iff `module` is a live bitsandbytes 4bit linear whose weight still
    # carries a quant_state (i.e. an on-the-fly / pre-quantized bnb-4bit base).
    if module is None: return False
    weight = getattr(module, "weight", None)
    if weight is None: return False
    if weight.__class__.__name__ != "Params4bit": return False
    return getattr(weight, "quant_state", None) is not None
pass


def _merge_lora(W, lora_stats, name, use_dequant_base = False):
    if lora_stats.lora_A is None or lora_stats.lora_B is None: return W
    device = _active_merge_device()
    # QLoRA merge-base correctness (gated, see _DEQUANT_MERGE_BASE_MODEL_TYPES). A bnb-4bit
    # adapter is trained against dequant(W4), not the 16bit base W16 that merged_16bit downloads;
    # they differ by quant error q = W16 - dequant(W4). Merging the delta into W16 leaves a stray
    # +q the adapter never saw. For most models ||q|| is negligible (and W16 is the better 16bit
    # weight), but for huge base-weight norms over tiny MuP multipliers (Falcon-H1) +q compounds
    # and swamps the fine-tune. The caller sets use_dequant_base only for gated archs -> strict
    # no-op elsewhere. Assumes a live 4bit base means the adapter trained against dequant(W4)
    # (the standard QLoRA flow); an adapter trained on the 16bit base and only reloaded in 4bit
    # for export has no merge-time provenance signal, so it would also fold onto dequant(W4).
    if use_dequant_base and _is_bnb_4bit_base(getattr(lora_stats, "module", None)):
        try:
            W_dq = dequantize_module_weight(lora_stats.module)
        except Exception as e:
            # For a gated arch the 16bit base is the known-wrong base (the +q error this path
            # exists to remove), so silently folding onto it would emit a corrupt merged_16bit.
            # Surface the failure instead of degrading just this layer to the wrong base.
            raise RuntimeError(
                f"Unsloth: could not dequantize the 4bit base for `{name}` during merged_16bit "
                "export of a model that requires dequant(W4) as the merge base. Falling back to "
                "the 16bit base would corrupt this checkpoint, so the merge was aborted. Free GPU "
                "memory (or merge on CPU) and retry."
            ) from e
        if tuple(W_dq.shape) == tuple(W.shape):
            W = W_dq
        # else: a shape mismatch is structural (e.g. vocab resize handled below), not a dequant
        # failure, so keep the 16bit W here and let the resize path reconcile it.
    W = W.to(device, dtype = torch.float32, non_blocking = True)
    lora_B = lora_stats.lora_B.to(device, dtype = torch.float32, non_blocking = True)
    lora_A = lora_stats.lora_A.to(device, dtype = torch.float32, non_blocking = True)
    # Handle vocab resize: LoRA may have more rows than base safetensors weight
    if lora_B.shape[0] != W.shape[0]:
        new_size = lora_B.shape[0]
        old_size = W.shape[0]
        W_new = torch.zeros(new_size, W.shape[1], dtype=W.dtype, device=W.device)
        W_new[:old_size] = W
        W = W_new.addmm_(lora_B, lora_A, alpha=lora_stats.alpha)
    else:
        W = W.addmm_(lora_B, lora_A, alpha=lora_stats.alpha)
    # DoRA: rescale the merged direction to the learned magnitude. With delta = alpha*(B@A),
    # PEFT's DoRA merge is (m / ||W0 + delta||_row) * (W0 + delta), one L2 norm per output row
    # over the input dim. W already holds W0 + delta here, so fold m onto it.
    magnitude = getattr(lora_stats, "magnitude", None)
    if magnitude is not None:
        magnitude = magnitude.to(device, dtype = torch.float32, non_blocking = True).reshape(-1)
        if magnitude.shape[0] != W.shape[0]:
            raise ValueError(
                f"Unsloth: DoRA magnitude for `{name}` has {magnitude.shape[0]} entries but the "
                f"merged weight has {W.shape[0]} output rows."
            )
        weight_norm = torch.linalg.norm(W, dim = 1).clamp_min(1e-9)
        W = (magnitude / weight_norm).unsqueeze(1) * W
    if not torch.isfinite(torch.amax(W)).item():
        raise ValueError('Unsloth: Merge failed as there are infinite elements in ' + name)
    return W
pass


def _get_modules_to_save_weight(module):
    modules_to_save = getattr(module, "modules_to_save", None)
    if modules_to_save is None:
        return None

    # Prefer the default adapter, else first entry with a weight
    for key in ("default",):
        try:
            candidate = modules_to_save[key]
            if hasattr(candidate, "weight"):
                return candidate.weight
        except Exception:
            continue

    for _, candidate in modules_to_save.items():
        if hasattr(candidate, "weight"):
            return candidate.weight

    return None


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
    magnitude : object = None   # DoRA lora_magnitude_vector weight (None for plain LoRA)
pass


def assert_same_keys(model, new_state_dict):
    """Compare only weight/bias tensors, normalizing MoE helper wrappers
    (base_layer, modules_to_save, original_module) and LoRA suffixes so they
    don't trigger false mismatches.
    """
    inner_model = model.base_model.model if hasattr(model, "base_model") else model

    def _should_ignore(key: str) -> bool:
        # Ignore helper wrappers and raw LoRA adapter tensors; the merged
        # state_dict intentionally omits lora_A / lora_B / DoRA magnitude weights
        # (the magnitude is folded into the merged weight in _merge_lora).
        return (
            "modules_to_save" in key
            or "original_module" in key
            or ".lora_A" in key
            or ".lora_B" in key
            or ".lora_embedding" in key
            or ".lora_magnitude_vector" in key
        )

    def _normalize(key: str) -> str:
        if not (key.endswith(".weight") or key.endswith(".bias")):
            return ""
        # strip helper wrappers
        key = key.replace(".base_layer", "")
        key = key.replace(".modules_to_save.default", "")
        key = key.replace(".original_module", "")
        key = key.replace(".lora_A.default", ".lora_A")
        key = key.replace(".lora_B.default", ".lora_B")
        return key

    original_keys = {
        k
        for k in (_normalize(x) for x in inner_model.state_dict().keys())
        if k and not _should_ignore(k)
    }
    new_keys = {
        k
        for k in (_normalize(x) for x in new_state_dict.keys())
        if k and not _should_ignore(k)
    }

    # On tied-weight models, lm_head.weight shares storage with
    # embed_tokens.weight and may be absent from the safetensors file
    # or the built state_dict depending on wrapping (LoRA vs modules_to_save).
    # Exclude both from the check to avoid false positives.
    base_model = inner_model.model if hasattr(inner_model, "model") else inner_model
    tie_word_embeddings = getattr(
        getattr(base_model, "config", None), "tie_word_embeddings", False
    )
    if tie_word_embeddings:
        _tied_suffixes = ("lm_head.weight", "embed_tokens.weight")
        original_keys = {k for k in original_keys if not any(k.endswith(s) for s in _tied_suffixes)}
        new_keys      = {k for k in new_keys      if not any(k.endswith(s) for s in _tied_suffixes)}

    difference = original_keys ^ new_keys
    if len(difference) != 0:
        raise RuntimeError(f"Unsloth: Extracted keys = {difference} do not match!")
pass


def _get_lora_scaling(module):
    # All Unsloth Zoo code licensed under LGPLv3
    # Resolve plural active_adapters or older singular active_adapter (may be a list);
    # 0.0 if unresolved so counts align. (#2966)
    active_adapters = getattr(module, "active_adapters", None)
    if active_adapters:
        active_adapter = active_adapters[0]
    else:
        active_adapter = getattr(module, "active_adapter", "default")
        if isinstance(active_adapter, (list, tuple)):
            active_adapter = active_adapter[0] if active_adapter else "default"
    try:
        return module.scaling[active_adapter]
    except Exception:
        return 0.0
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

        elif name.endswith(".lora_magnitude_vector.default"):
            # DoRA magnitude vector m; folded onto the merged weight in _merge_lora. Register its
            # key so the key-consistency check does not flag it (the merged model omits it).
            lora_weights[name[:-len(".lora_magnitude_vector.default")]].magnitude = module.weight
            expand_module_keys(name, module, remove_keys)

        elif isinstance(module, Linear_LoRA_Layers):
            lora_weights[name].alpha = _get_lora_scaling(module)
            scaling_count += 1
            expand_module_keys(name, module, remove_keys)

        # LoRA wrappers (MoE/quant/older peft) not subclassing Linear_LoRA_Layers:
        # capture alpha so counts align. Require lora_A/lora_B so a non-LoRA module
        # with its own `scaling` + `active_adapter` isn't misclassified. (#2966)
        elif hasattr(module, "scaling") and \
            (hasattr(module, "lora_A") or hasattr(module, "lora_B")) and \
            (hasattr(module, "active_adapters") or hasattr(module, "active_adapter")):
            lora_weights[name].alpha = _get_lora_scaling(module)
            scaling_count += 1
            expand_module_keys(name, module, remove_keys)

        elif name.endswith(".base_layer"):
            lora_weights[name[:-len(".base_layer")]].module = module
            module_count += 1
            remove_keys.add(name)
            remove_keys.add(name[:-len(".base_layer")])

        elif getattr(module, "modules_to_save", None) is not None:
            saved_weight = _get_modules_to_save_weight(module)
            if saved_weight is not None:
                lora_weights[name].module = module
                expand_module_keys(name, module, remove_keys)
                remove_keys.add(name)
            else:
                new_keys = expand_module_keys(name, module, set())
                remove_keys.update(new_keys)
                remove_keys.add(name)

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
                    # Drop quantized sub-keys (".weight."); keep gate_tanh, embedding etc
                    if ".weight." in key:
                        remove_keys.add(key)
                    else:
                        pass
            remove_keys.add(name)
        pass
    pass
    # Custom MoE LoRA wrappers (e.g. GPT-OSS expert LoRA) match the fallback
    # branch and have lora_A/B/scaling but no .base_layer child, so module
    # stays None while the other three counters are incremented.
    # Count them here to align module_count (#3405, #3701).
    for _key, _stats in lora_weights.items():
        if (
            _stats.lora_A is not None
            and _stats.lora_B is not None
            and _stats.module is None
        ):
            module_count += 1

    # DoRA on a non-dense target (e.g. an Embedding / tied lm_head trained with
    # use_dora=True) captures a lora_magnitude_vector but no mergeable lora_A/lora_B
    # (PEFT stores the embedding delta as lora_embedding_A/lora_embedding_B, which
    # this merge does not read). _merge_lora only folds the magnitude onto W0+delta
    # for a dense nn.Linear; here it would early-return the base weight and the
    # magnitude (and the embedding delta) would be silently dropped -- and since
    # assert_same_keys now ignores lora_magnitude_vector keys, that wrong merge would
    # not even trip the key check. Fail loud instead (matches _refuse_dora_on_moe).
    for _key, _stats in lora_weights.items():
        if getattr(_stats, "magnitude", None) is not None and (
            _stats.lora_A is None or _stats.lora_B is None
        ):
            raise RuntimeError(
                f"Unsloth: DoRA (use_dora=True) merging is not yet supported for `{_key}` "
                "(a non-Linear target such as an embedding / tied lm_head has a DoRA "
                "magnitude but no mergeable LoRA delta, so the magnitude would be silently "
                "dropped). Fine-tune such layers without DoRA, or open an issue at "
                "https://github.com/unslothai/unsloth/issues."
            )

    if not (module_count == lora_A_count == lora_B_count == scaling_count):
        print(
            f"[Unsloth merge debug] LoRA count mismatch: modules={module_count}, "
            f"lora_A={lora_A_count}, lora_B={lora_B_count}, scaling={scaling_count}"
        )
        try:
            items = list(lora_weights.items())
            print(f"[Unsloth merge debug] Total LoRA keys: {len(lora_weights)}")
            for k, v in items[:10]:
                param_name = getattr(v.module, "parameter_name", None)
                a_shape = tuple(v.lora_A.shape) if v.lora_A is not None else None
                b_shape = tuple(v.lora_B.shape) if v.lora_B is not None else None
                print(f"  key={k} param={param_name} A={a_shape} B={b_shape}")
        except Exception:
            pass

    # Also return state_dict if needed
    if return_state_dict:
        old_state_dict = inner_model.state_dict()
        state_dict     = collections.OrderedDict()
        for name, param in old_state_dict.items():

            if name.endswith(".base_layer.weight"):
                name = name[:-len(".base_layer.weight")]

            # modules_to_save wraps embed_tokens / lm_head; strip the wrapper
            # so the key matches lora_weights entries created by the branch above.
            # Only strip .weight variant; the lora_weights branch adds both
            # .weight and .bias from the module so we don't need a separate bias entry.
            elif name.endswith(".modules_to_save.default.weight"):
                name = name[:-len(".modules_to_save.default.weight")]

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
    save_method = "merged_16bit",
    counted_lora_modules = None,
    tie_word_embeddings = False,
    weight_block_size = None,
    use_dequant_base = False,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Merges LoRA and overwrites the safetensors file it was merged to
    if base_model_is_quantized and quant_type == "mxfp4" and save_method != "mxfp4":
        if UNSLOTH_ENABLE_LOGGING:
            logger.info("mxfp4 quantized model detected. Using safe rewrite strategy (requires temporary disk space).")
        # mxfp4 needs the full-rewrite path
        return _merge_and_overwrite_lora_mxfp4(
            save_directory, filename, lora_weights, output_dtype,
            model_class_name, base_model_is_quantized, quant_type,
        )
    pass

    # FP8 grows to 16bit and drops its scales, so full-rewrite (mmap can't resize).
    # 16bit merge only; other save methods keep the quant config and must not dequantize.
    # MoE-expert LoRA is left to the in-place quant-aware path below (the dense rewrite cannot
    # fuse per-expert adapters); dense / non-expert FP8 LoRA dequantizes here.
    _fp8_moe_expert_lora = any(
        isinstance(k, str) and (".experts" in k or ".moe" in k) for k in lora_weights
    )
    if base_model_is_quantized and quant_type == "fp8" and save_method == "merged_16bit" and not _fp8_moe_expert_lora:
        if UNSLOTH_ENABLE_LOGGING:
            logger.info("FP8 quantized model detected. Dequantizing to 16bit via full rewrite.")
        return _merge_and_overwrite_lora_fp8(
            save_directory, filename, lora_weights, output_dtype,
            model_class_name, tie_word_embeddings = tie_word_embeddings,
            weight_block_size = weight_block_size,
        )
    pass

    filename_original = os.path.join(save_directory, filename)  # Original file path
    count = 0
    # Collect keys for this shard so the caller can aggregate without re-reading the file (avoids
    # an extra safetensors pass purely for tied-embedding bookkeeping).
    safetensor_keys_seen = set()
    processed_moe_gate = set()  # track (fused_key, expert_idx) processed for gate_up_proj
    if counted_lora_modules is None:
        counted_lora_modules = set()   # fused lora keys counted toward n_saved_modules

    # built once below, once the real safetensor keys are known.
    raw_pointer = None
    mm = None
    header_metadata = None
    length_of_header = 0

    try:
        # Memory-map for in-place overwrite
        raw_pointer = open(filename_original, "r+b")
        mm = mmap.mmap(raw_pointer.fileno(), length = 0, access = mmap.ACCESS_WRITE)

        # Parse safetensors header
        length_of_header = int.from_bytes(mm.read(8), "little")
        header_metadata = json.loads(mm.read(length_of_header))
        mm.seek(0)

        with safe_open(filename_original, framework = "pt", device = "cpu") as file:
            safetensor_keys = list(file.keys())
            safetensor_keys_seen.update(safetensor_keys)

            # Pre-compute number of experts per layer prefix from shard keys
            moe_num_experts = {}
            for _k in safetensor_keys:
                m = re.match(r"^(.*mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", _k)
                if m:
                    prefix, idx, _ = m.groups()
                    idx = int(idx)
                    moe_num_experts[prefix] = max(moe_num_experts.get(prefix, -1), idx + 1)
                # Legacy Mixtral disk keys: count under the renamed mlp.experts.
                m = re.match(r"^(.*)\.block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight$", _k)
                if m:
                    prefix = m.group(1) + ".mlp.experts"
                    idx = int(m.group(2))
                    moe_num_experts[prefix] = max(moe_num_experts.get(prefix, -1), idx + 1)

            # Update converted_lora_weights with actual safetensor keys
            converted_lora_weights = _convert_lora_keys_to_safetensor_format(
                lora_weights,
                safetensor_keys,
                model_class_name = model_class_name,
            )
            # #5410: use the wrapped module's num_experts, not the shard-local one.
            for _lk, _ls in converted_lora_weights.items():
                if not isinstance(_lk, str): continue
                _pm = re.match(r"^(.*mlp\.experts)(?:\.base_layer)?$", _lk)
                if not _pm: continue
                _prefix = _pm.group(1)
                _ne = _resolve_num_experts_from_lora_stats(_ls, fallback = None)
                if _ne is not None and _ne > 1:
                    moe_num_experts[_prefix] = _ne
            processed_mxfp4_keys = set()
            if UNSLOTH_ENABLE_LOGGING:
                try:
                    logger.info(f"[merge_debug] Converted LoRA keys (sample): {list(converted_lora_weights.keys())[:6]}")
                except Exception:
                    pass

            # Fast path for MoE models with stacked experts: merge by iterating LoRA modules instead of per-weight scan
            # Matches standard (.mlp.experts), Gemma4 (.experts without .mlp), and .moe naming
            if any(".experts" in k or ".moe" in k for k in converted_lora_weights.keys()):
                count = _merge_moe_experts_file(
                    mm = mm,
                    header_metadata = header_metadata,
                    length_of_header = length_of_header,
                    file = file,
                    converted_lora_weights = converted_lora_weights,
                    moe_num_experts = moe_num_experts,
                    output_dtype = output_dtype,
                    counted_lora_modules = counted_lora_modules,
                    processed_mxfp4_keys = processed_mxfp4_keys,
                    weight_block_size = weight_block_size,
                )

            for key in safetensor_keys:
                if key in processed_mxfp4_keys:
                    continue

                if (
                    UNSLOTH_ENABLE_LOGGING
                    and count == 0
                    and len(processed_moe_gate) == 0
                    and len(processed_mxfp4_keys) == 0
                ):
                    logger.info(f"[merge_debug] First shard key example: {key}")

                # Legacy Mixtral disk keys: transformers v5 fuses them under mlp.experts
                # in-memory (where the LoRA lives), so map disk w1->gate, w3->up, w2->down.
                m_legacy = re.match(
                    r"^(.*)\.block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight$", key
                )
                if m_legacy:
                    layer_prefix, expert_idx, w_name = m_legacy.groups()
                    expert_idx = int(expert_idx)
                    base_prefix = layer_prefix + ".mlp.experts"
                    available_experts = moe_num_experts.get(base_prefix, None)
                    if available_experts is not None and expert_idx >= available_experts:
                        continue
                    # Fallback count only: the merge helpers re-resolve num_experts from the
                    # wrapped module (lora_stats), so a shard-local value here is corrected.
                    num_experts = available_experts
                    if w_name == "w2":
                        # down_proj LoRA: experts module, else fused .down_proj when unwrapped.
                        fused_key = base_prefix
                        lora_stats = converted_lora_weights.get(fused_key)
                        if lora_stats is None:
                            fused_key = base_prefix + ".down_proj"
                            lora_stats = converted_lora_weights.get(fused_key)
                        merge_role = "down"
                        merge_fn = _merge_moe_down_proj_expert
                    else:
                        # gate_up_proj LoRA: experts.base_layer, else .gate_up_proj when unwrapped.
                        fused_key = base_prefix + ".base_layer"
                        lora_stats = converted_lora_weights.get(fused_key)
                        if lora_stats is None:
                            fused_key = base_prefix + ".gate_up_proj"
                            lora_stats = converted_lora_weights.get(fused_key)
                        merge_role = "gate" if w_name == "w1" else "up"
                        merge_fn = _merge_moe_gate_expert if w_name == "w1" else _merge_moe_up_expert
                    if lora_stats is not None and lora_stats.lora_A is not None and lora_stats.lora_B is not None:
                        # Quant-aware: FP8/compressed shards dequant->merge->requant; 16-bit pass through.
                        merged = _merge_moe_expert_quant_aware(
                            merge_role, key, file, header_metadata, lora_stats,
                            expert_idx, num_experts, output_dtype, mm, length_of_header,
                            processed_mxfp4_keys, merge_fn,
                            weight_block_size = weight_block_size,
                        )
                        if merged and fused_key not in counted_lora_modules:
                            count += 1
                            counted_lora_modules.add(fused_key)
                        continue

                # MoE stacked experts: gate_up_proj is fused in the model but
                # sharded as per-expert gate_proj/up_proj on disk.
                m_gate = re.match(r"^(.*mlp\.experts)\.(\d+)\.(gate_proj|up_proj)\.weight$", key)
                if m_gate:
                    if UNSLOTH_ENABLE_LOGGING and len(processed_moe_gate) < 2:
                        logger.info(f"[merge_debug] Matched gate/up key {key}")
                    base_prefix, expert_idx, proj_type = m_gate.groups()
                    expert_idx = int(expert_idx)

                    # Skip experts not present in this shard (defensive)
                    available_experts = moe_num_experts.get(base_prefix, None)
                    if available_experts is not None and expert_idx >= available_experts:
                        continue

                    # PEFT stores gate_up_proj LoRA on experts.base_layer
                    fused_key = base_prefix + ".base_layer"
                    lora_stats = converted_lora_weights.get(fused_key)
                    if lora_stats is None:
                        fused_key = base_prefix + ".gate_up_proj"
                        lora_stats = converted_lora_weights.get(fused_key)

                    if lora_stats is not None and lora_stats.lora_A is not None and lora_stats.lora_B is not None:
                        num_experts = moe_num_experts.get(base_prefix, None)

                        W = file.get_tensor(key)

                        if proj_type == "gate_proj":
                             merged_W = _merge_moe_gate_expert(
                                W, lora_stats, expert_idx, num_experts, output_dtype or W.dtype
                            )
                        else:
                             merged_W = _merge_moe_up_expert(
                                W, lora_stats, expert_idx, num_experts, output_dtype or W.dtype
                            )

                        _write_tensor_direct_torch(mm, header_metadata, length_of_header, key, merged_W, W.dtype)
                        processed_mxfp4_keys.add(key)

                        # Count the module once any part of it is processed
                        if fused_key not in counted_lora_modules:
                            count += 1
                            counted_lora_modules.add(fused_key)
                        continue

                m_down = re.match(r"^(.*mlp\.experts)\.(\d+)\.down_proj\.weight$", key)
                if m_down:
                    base_prefix, expert_idx = m_down.groups()
                    expert_idx = int(expert_idx)
                    available_experts = moe_num_experts.get(base_prefix, None)
                    if available_experts is not None and expert_idx >= available_experts:
                        continue
                    fused_key = base_prefix  # down_proj LoRA stored on experts module
                    lora_stats = converted_lora_weights.get(fused_key)
                    if lora_stats is None and len(processed_moe_gate) < 3:
                        if UNSLOTH_ENABLE_LOGGING:
                            logger.info(f"[merge_debug] No LoRA found for down_proj prefix {base_prefix}")
                    if lora_stats is not None and lora_stats.lora_A is not None and lora_stats.lora_B is not None:
                        num_experts = moe_num_experts.get(base_prefix, None)
                        if UNSLOTH_ENABLE_LOGGING:
                            logger.info(f"[merge_debug] Applying down_proj LoRA for {fused_key} expert {expert_idx}")
                        down_W = file.get_tensor(key)
                        merged_down = _merge_moe_down_proj_expert(
                            down_W, lora_stats, expert_idx, num_experts, output_dtype or down_W.dtype
                        )
                        _write_tensor_direct_torch(mm, header_metadata, length_of_header, key, merged_down, down_W.dtype)
                        processed_mxfp4_keys.add(key)
                        if fused_key not in counted_lora_modules:
                            count += 1
                            counted_lora_modules.add(fused_key)
                        continue

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
                # Standard 16-bit tensor
                W = file.get_tensor(key)
                W_original_dtype = W.dtype

                if W is None:
                    continue

                lora_key = output_key[:-len(".weight")] if output_key.endswith(".weight") else output_key
                lora_stats = converted_lora_weights.get(lora_key, None)
                # Fallback: handle Gemma4ClippableLinear (.linear.weight -> .weight)
                if lora_stats is None and lora_key.endswith(".linear"):
                    lora_stats = converted_lora_weights.get(
                        lora_key[:-len(".linear")], None
                    )
                # Tied embeddings can omit lm_head.weight from safetensors. If lm_head has LoRA
                # adapters, apply them onto embed_tokens.weight since both share one base tensor.
                if (
                    lora_stats is None
                    and tie_word_embeddings
                    and lora_key.endswith("embed_tokens")
                ):
                    lm_head_key = lora_key[:-len("embed_tokens")] + "lm_head"
                    lora_stats = converted_lora_weights.get(lm_head_key, None)
                    if lora_stats is None and lm_head_key.startswith("model."):
                        lora_stats = converted_lora_weights.get(lm_head_key[len("model."):], None)
                    if lora_stats is None and not lm_head_key.startswith("model."):
                        lora_stats = converted_lora_weights.get("model." + lm_head_key, None)

                if lora_stats is not None:
                    # Prefer modules_to_save weights when there's no LoRA delta
                    if getattr(lora_stats, "lora_A", None) is None and getattr(lora_stats, "module", None) is not None:
                        saved_weight = _get_modules_to_save_weight(lora_stats.module)
                        if saved_weight is None and hasattr(lora_stats.module, "weight"):
                            saved_weight = lora_stats.module.weight
                        if saved_weight is not None:
                            target_dtype = output_dtype if output_dtype is not None else W_original_dtype
                            W = saved_weight.to(W.device, dtype = target_dtype, non_blocking = True)
                            count += 1
                    elif hasattr(lora_stats, 'lora_A') and lora_stats.lora_A is not None:
                        W = _merge_lora(W, lora_stats, output_key, use_dequant_base = use_dequant_base)
                        count += 1

                success = _write_tensor_direct_torch(mm, header_metadata, length_of_header, output_key, W, W_original_dtype)

                if not success:
                    # Tensor was resized (e.g. vocab grew); track for rewrite
                    if not hasattr(_merge_and_overwrite_lora, "_resized"):
                        _merge_and_overwrite_lora._resized = {}
                    _merge_and_overwrite_lora._resized[output_key] = W.to(
                        dtype=W_original_dtype, device="cpu"
                    )

                nbytes = W.numel() * W.element_size()
                del W
                # flush only after a large tensor; per-key flush was pure stall.
                if nbytes >= _EMPTY_CACHE_BYTES_THRESHOLD:
                    device_empty_cache()
            pass
        pass
        mm.flush()
        mm.close()
        raw_pointer.close()

        # Vocab grew -> resized tensors no longer fit their byte slots; rewrite the
        # shard. Stream to a temp file when disk allows, else rewrite in place.
        resized = getattr(_merge_and_overwrite_lora, "_resized", {})
        if resized:
            _merge_and_overwrite_lora._resized = {}
            gc.collect()
            device_empty_cache()

            temp_dir = os.path.dirname(os.path.abspath(filename_original))
            est_bytes = _estimate_resized_shard_bytes(header_metadata, resized, length_of_header)
            try:
                free_bytes = shutil.disk_usage(temp_dir).free
            except OSError:
                free_bytes = None
            margin = 64 * 1024 * 1024
            if free_bytes is not None and free_bytes < est_bytes + margin:
                # No room for the atomic temp+replace rewrite. The in-place fallback
                # overwrites the shard non-atomically (a failed write truncates it,
                # worse on Windows mmap), so refuse by default; opt in to allow it.
                if os.environ.get("UNSLOTH_ALLOW_NON_ATOMIC_RESIZED_REWRITE") != "1":
                    raise RuntimeError(
                        f"Unsloth: not enough free disk to rewrite resized shard "
                        f"{filename_original} atomically (free={free_bytes}, "
                        f"need~={est_bytes + margin}). The original shard was left "
                        f"intact. Free disk space or point the save directory at a "
                        f"larger volume, or set "
                        f"UNSLOTH_ALLOW_NON_ATOMIC_RESIZED_REWRITE=1 to allow the "
                        f"non-atomic in-place rewrite (which can corrupt the shard if "
                        f"the write is interrupted)."
                    )
                logger.warning(
                    f"Unsloth: low disk to rewrite resized shard {filename_original} "
                    f"(free={free_bytes}, need~={est_bytes}); rewriting in place via "
                    f"UNSLOTH_ALLOW_NON_ATOMIC_RESIZED_REWRITE. This is non-atomic, so "
                    f"do not interrupt the save."
                )
                _inplace_rewrite_resized_shard(filename_original, header_metadata, resized)
            else:
                _stream_rewrite_resized_shard_and_replace(
                    filename_original, temp_dir, header_metadata, length_of_header, resized,
                )
            del resized
            gc.collect()
            device_empty_cache()

        device_empty_cache()
        return count, safetensor_keys_seen

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Model merge failed with error: {e}") from e

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
    return count, safetensor_keys_seen
pass

# Per-expert MoE LoRA merge helpers (#5410). PEFT 0.18 = "swapped", PEFT 0.19+
# = "standard" (huggingface/peft#3165). Layout detected by shape vs the
# per-expert disk weight; unrecognised shapes increment fallback and the
# outer raises so partial merges cannot save.

_MOE_MERGE_STATE = {
    "attempted":   0,
    "applied":     0,
    "fallback":    0,
    "first_error": None,
}


def _reset_moe_merge_state():
    _MOE_MERGE_STATE["attempted"]   = 0
    _MOE_MERGE_STATE["applied"]     = 0
    _MOE_MERGE_STATE["fallback"]    = 0
    _MOE_MERGE_STATE["first_error"] = None


def _record_moe_merge_fallback(role, expert_idx, reason, lora_stats, W_shape):
    _MOE_MERGE_STATE["fallback"] += 1
    if _MOE_MERGE_STATE["first_error"] is None:
        try:
            a_shape = tuple(lora_stats.lora_A.shape) if lora_stats is not None and lora_stats.lora_A is not None else None
            b_shape = tuple(lora_stats.lora_B.shape) if lora_stats is not None and lora_stats.lora_B is not None else None
        except Exception:
            a_shape, b_shape = None, None
        _MOE_MERGE_STATE["first_error"] = {
            "role":          role,
            "expert_idx":    expert_idx,
            "reason":        str(reason),
            "lora_A_shape":  a_shape,
            "lora_B_shape":  b_shape,
            "per_expert_W":  W_shape,
        }
    logger.warning(
        f"[Unsloth MoE merge fallback] role={role} expert={expert_idx} reason={reason}. "
        f"per_expert_W={W_shape}. The base weight is being written through; "
        "the merged checkpoint will be missing this delta."
    )


def _resolve_num_experts_from_lora_stats(lora_stats, fallback):
    """Walk module.base_layer for num_experts; bounded so cycles cannot hang."""
    try:
        module = getattr(lora_stats, "module", None)
    except Exception:
        return fallback
    seen = set()
    for _ in range(16):
        if module is None:
            break
        try:
            mid = id(module)
        except Exception:
            break
        if mid in seen:
            break
        seen.add(mid)
        for attr in (
            "num_experts",
            "num_experts_per_group",
            "num_routed_experts",
            "num_local_experts",
            "num_moe_experts",
        ):
            try:
                value = getattr(module, attr, None)
            except Exception:
                value = None
            if isinstance(value, int) and value > 1:
                return value
        try:
            module = getattr(module, "base_layer", None)
        except Exception:
            break
    return fallback


# MoE-quant save-side dispatch. Each quant kind (FP8 today, room for bnb4bit
# later) registers a handler in `unsloth_zoo.temporary_patches.moe_utils_fp8.
# _MOE_QUANT_HANDLERS`; we consult that list via `apply_moe_quant_load`
# instead of inlining quant-specific dequant/requant math here.
try:
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import (
        apply_moe_quant_load as _apply_moe_quant_load,
        _MOE_QUANT_UNSAFE,
    )
except ImportError:
    _apply_moe_quant_load = None
    _MOE_QUANT_UNSAFE = object()


def _merge_moe_expert_quant_aware(
    role: str,
    key: str,
    file,
    header_metadata,
    lora_stats,
    expert_idx: int,
    num_experts: int,
    output_dtype,
    mm,
    length_of_header,
    processed_mxfp4_keys,
    merge_fn,
    weight_block_size = None,
) -> bool:
    """Read one expert weight (dequantising via the quant registry), apply
    `merge_fn`, requantise if the handler gave a requant closure, and write
    data + companion scales back to `mm`. True on merge+write, False on skip.
    """
    if key not in header_metadata:
        return False

    if _apply_moe_quant_load is None:
        W = file.get_tensor(key)
        requant_fn = None
    else:
        loaded = _apply_moe_quant_load(file, header_metadata, key, block_size = weight_block_size)
        if loaded[0] is _MOE_QUANT_UNSAFE:
            _record_moe_merge_fallback(
                role, expert_idx,
                f"{role} base at {key} has no usable quant companion scale; "
                "skipping LoRA merge to avoid scale-loss corruption",
                lora_stats, None,
            )
            return False
        W, requant_fn = loaded

    merged = merge_fn(W, lora_stats, expert_idx, num_experts, output_dtype or W.dtype)

    if requant_fn is None:
        write_dtype = W.dtype
    else:
        merged, write_dtype, extra_writes = requant_fn(merged)
        for extra_key, extra_tensor, extra_dtype in extra_writes:
            _write_tensor_direct_torch(
                mm, header_metadata, length_of_header,
                extra_key, extra_tensor, extra_dtype,
            )
            processed_mxfp4_keys.add(extra_key)

    _write_tensor_direct_torch(mm, header_metadata, length_of_header, key, merged, write_dtype)
    processed_mxfp4_keys.add(key)
    return True


def _peft_paramwrapper_swaps_in_out():
    """True if PEFT's ParamWrapper.update_layer swaps
    `(experts, in, out) -> (experts, out, in)` for 3D non-transposed MoE params
    (PEFT 0.19+; absent in 0.18-).

    Uses `peft.__version__`, falling back to a source check for
    `_did_swap_in_out_features` when no version string is available.
    """
    try:
        from peft import __version__ as _peft_version
        major, minor = _peft_version.split(".")[:2]
        return (int(major), int(minor)) >= (0, 19)
    except Exception:
        pass
    try:
        from peft.tuners.lora.layer import ParamWrapper
        # PEFT 0.19's update_layer references `_did_swap_in_out_features`; the
        # attribute is set on instances, not the class, so a hasattr check on
        # the class doesn't suffice — fall back to source inspection.
        import inspect
        return "_did_swap_in_out_features" in inspect.getsource(ParamWrapper.update_layer)
    except Exception:
        return True  # assume modern PEFT


_PEFT_AMBIGUOUS_LAYOUT_DEFAULT = "standard" if _peft_paramwrapper_swaps_in_out() else "swapped"


def _detect_moe_lora_layout(lora_A, lora_B, num_experts, out_dim, in_dim, lora_module=None):
    """Shape-classify as 'swapped' / 'standard' / 'unknown'; returns (layout, r).

    When 2*I == H for the fused gate_up_proj, both checks pass. That ambiguous
    case resolves on whether PEFT swapped in/out features: 0.19+ swaps ->
    "standard", 0.18- doesn't -> "swapped". The default comes from
    `_PEFT_AMBIGUOUS_LAYOUT_DEFAULT` (computed at import) but a caller may
    override via a `lora_module` with `_did_swap_in_out_features` set.
    """
    total_rank_A, dim_A = lora_A.shape
    dim_B, total_rank_B = lora_B.shape
    if total_rank_A != total_rank_B or num_experts is None or num_experts <= 0:
        return "unknown", 0
    if total_rank_A % num_experts != 0:
        return "unknown", 0
    r = total_rank_A // num_experts
    standard_match = (dim_A == in_dim and dim_B == out_dim)
    swapped_match  = (dim_A == out_dim and dim_B == in_dim)
    if standard_match and swapped_match:
        # Ambiguous (typically 2*I == H): prefer the per-wrapper signal
        # (_did_swap_in_out_features=True -> "standard"), else version default.
        if lora_module is not None and hasattr(lora_module, "_did_swap_in_out_features"):
            return ("standard" if lora_module._did_swap_in_out_features else "swapped"), r
        return _PEFT_AMBIGUOUS_LAYOUT_DEFAULT, r
    if standard_match:
        return "standard", r
    if swapped_match:
        return "swapped", r
    return "unknown", r


def _refuse_dora_on_moe(lora_stats):
    """DoRA on MoE experts is not yet supported: the expert merge helpers fold only the LoRA
    delta, not the DoRA magnitude (the dense path handles it in _merge_lora). Fail loud rather
    than emit a checkpoint with the magnitude silently dropped."""
    if getattr(lora_stats, "magnitude", None) is not None:
        raise RuntimeError(
            "Unsloth: DoRA (use_dora=True) merging is not yet supported for MoE expert layers. "
            "Fine-tune only the non-expert (attention/MLP) layers with DoRA, or open an issue at "
            "https://github.com/unslothai/unsloth/issues."
        )
pass


def _merge_moe_gate_or_up_expert(W, lora_stats, expert_idx, num_experts, output_dtype, *, role):
    """Per-expert merge for gate_proj/up_proj (role='gate' -> first I, 'up' -> last I)."""
    if lora_stats is None or lora_stats.lora_A is None or lora_stats.lora_B is None:
        return W
    _refuse_dora_on_moe(lora_stats)
    _MOE_MERGE_STATE["attempted"] += 1
    try:
        num_experts = _resolve_num_experts_from_lora_stats(lora_stats, num_experts)
        if num_experts is None or num_experts <= 0:
            rank = getattr(lora_stats, "rank", 0) or 0
            if rank <= 0:
                # No num_experts and no rank: `total_rank // 1` would give a
                # degenerate r=1 slicing that passes the shape check but emits a
                # wrong delta silently. Refuse to merge.
                _record_moe_merge_fallback(
                    role, expert_idx,
                    "num_experts and lora_stats.rank both missing — cannot derive per-expert slicing",
                    lora_stats, tuple(W.shape),
                )
                return W
            num_experts = lora_stats.lora_A.shape[0] // rank

        I, H = W.shape
        layout, r = _detect_moe_lora_layout(
            lora_stats.lora_A, lora_stats.lora_B,
            num_experts = num_experts, out_dim = 2 * I, in_dim = H,
            lora_module = getattr(lora_stats, "module", None),
        )
        if layout == "unknown" or r <= 0:
            _record_moe_merge_fallback(
                role, expert_idx,
                f"layout not detected (A={tuple(lora_stats.lora_A.shape)}, B={tuple(lora_stats.lora_B.shape)})",
                lora_stats, (I, H),
            )
            return W

        start, end = expert_idx * r, (expert_idx + 1) * r
        if end > num_experts * r:
            _record_moe_merge_fallback(role, expert_idx, "expert_idx out of range", lora_stats, (I, H))
            return W

        # unsloth's MoE forward (temporary_patches/moe_utils.py
        # `_canonical_lora_weights_for_grouped_mm`) views lora_B as
        # (out, num_experts, r) — contiguous-r columns per expert. Must match
        # here so the merged checkpoint reproduces the training-time forward.
        # (unsloth bypasses PEFT's get_delta_weight via patch_param_wrapper_for_moe.)
        a_slice = lora_stats.lora_A[start:end, :]
        b_slice = lora_stats.lora_B[:, start:end]
        device  = _active_merge_device()
        a_f     = a_slice.to(device, dtype = torch.float32, non_blocking = True)
        b_f     = b_slice.to(device, dtype = torch.float32, non_blocking = True)

        if layout == "swapped":
            half = a_f[:, :I] if role == "gate" else a_f[:, I:]
            delta = b_f @ half
            merged = W.to(device, dtype = torch.float32, non_blocking = True).add(
                delta.transpose(0, 1), alpha = lora_stats.alpha,
            )
        else:
            half = b_f[:I, :] if role == "gate" else b_f[I:, :]
            delta = half @ a_f
            merged = W.to(device, dtype = torch.float32, non_blocking = True).add(
                delta, alpha = lora_stats.alpha,
            )

        _MOE_MERGE_STATE["applied"] += 1
        return merged.to(output_dtype)
    except Exception as exc:
        _record_moe_merge_fallback(role, expert_idx, repr(exc), lora_stats, tuple(W.shape))
        return W


def _merge_moe_gate_expert(gate_W, lora_stats, expert_idx, num_experts, output_dtype):
    return _merge_moe_gate_or_up_expert(
        gate_W, lora_stats, expert_idx, num_experts, output_dtype, role = "gate",
    )


def _merge_moe_up_expert(up_W, lora_stats, expert_idx, num_experts, output_dtype):
    return _merge_moe_gate_or_up_expert(
        up_W, lora_stats, expert_idx, num_experts, output_dtype, role = "up",
    )
pass


def _merge_moe_down_proj_expert(down_W, lora_stats, expert_idx, num_experts, output_dtype):
    if lora_stats is None or lora_stats.lora_A is None or lora_stats.lora_B is None:
        return down_W
    _refuse_dora_on_moe(lora_stats)
    _MOE_MERGE_STATE["attempted"] += 1
    try:
        num_experts = _resolve_num_experts_from_lora_stats(lora_stats, num_experts)
        if num_experts is None or num_experts <= 0:
            rank = getattr(lora_stats, "rank", 0) or 0
            if rank <= 0:
                _record_moe_merge_fallback(
                    "down", expert_idx,
                    "num_experts and lora_stats.rank both missing — cannot derive per-expert slicing",
                    lora_stats, tuple(down_W.shape),
                )
                return down_W
            num_experts = lora_stats.lora_A.shape[0] // rank

        H, I = down_W.shape
        layout, r = _detect_moe_lora_layout(
            lora_stats.lora_A, lora_stats.lora_B,
            num_experts = num_experts, out_dim = H, in_dim = I,
            lora_module = getattr(lora_stats, "module", None),
        )
        if layout == "unknown" or r <= 0:
            _record_moe_merge_fallback(
                "down", expert_idx,
                f"layout not detected (A={tuple(lora_stats.lora_A.shape)}, B={tuple(lora_stats.lora_B.shape)})",
                lora_stats, (H, I),
            )
            return down_W

        start, end = expert_idx * r, (expert_idx + 1) * r
        if end > num_experts * r:
            _record_moe_merge_fallback("down", expert_idx, "expert_idx out of range", lora_stats, (H, I))
            return down_W

        # See _merge_moe_gate_or_up_expert for the slicing convention rationale.
        a_slice = lora_stats.lora_A[start:end, :]
        b_slice = lora_stats.lora_B[:, start:end]
        device  = _active_merge_device()
        a_f     = a_slice.to(device, dtype = torch.float32, non_blocking = True)
        b_f     = b_slice.to(device, dtype = torch.float32, non_blocking = True)

        delta = b_f @ a_f
        if layout == "swapped":
            merged = down_W.to(device, dtype = torch.float32, non_blocking = True).add(
                delta.transpose(0, 1), alpha = lora_stats.alpha,
            )
        else:
            merged = down_W.to(device, dtype = torch.float32, non_blocking = True).add(
                delta, alpha = lora_stats.alpha,
            )

        _MOE_MERGE_STATE["applied"] += 1
        return merged.to(output_dtype)
    except Exception as exc:
        _record_moe_merge_fallback("down", expert_idx, repr(exc), lora_stats, tuple(down_W.shape))
        return down_W
pass


def _resolve_moe_num_experts(prefix, lora_stats, moe_num_experts):
    if prefix in moe_num_experts and moe_num_experts[prefix] > 0:
        return moe_num_experts[prefix]

    if lora_stats is None:
        return None

    module = getattr(lora_stats, "module", None)
    if module is not None:
        for attr in (
            "num_experts",
            "num_experts_per_group",
            "num_experts_per_tok",
            "num_experts_per_token",
            "num_moe_experts",
            "num_routed_experts",
        ):
            value = getattr(module, attr, None)
            if isinstance(value, int) and value > 1:
                moe_num_experts[prefix] = value
                if UNSLOTH_ENABLE_LOGGING:
                    try:
                        logger.info(
                            f"[merge_debug] Derived num_experts={value} for {prefix} from module.{attr}"
                        )
                    except Exception:
                        pass
                return value

    lora_A = getattr(lora_stats, "lora_A", None)
    if lora_A is None:
        return None

    total_rank = lora_A.shape[0]
    rank = getattr(lora_stats, "rank", None)
    if not rank:
        lora_B = getattr(lora_stats, "lora_B", None)
        rank = lora_B.shape[-1] if lora_B is not None else None

    if not rank or rank == 0:
        return None
    if total_rank % rank != 0:
        return None

    candidate = total_rank // rank
    moe_num_experts[prefix] = candidate
    if UNSLOTH_ENABLE_LOGGING:
        try:
            logger.info(
                f"[merge_debug] Derived num_experts={candidate} for {prefix} from LoRA stats"
            )
        except Exception:
            pass

    return candidate



# Per-expert disk-tensor naming schemes for the per-expert (2D) MoE layout.
# Default scheme is the standard gate_proj/up_proj/down_proj (DeepSeek/Qwen3 on
# old transformers). LFM2 (lfm2_moe) and some ERNIE variants instead store the
# experts on disk as w1 (gate) / w3 (up) / w2 (down) while the runtime module is
# the fused Lfm2MoeExperts/Ernie4_5_MoeExperts (LoRA on gate_up_proj/down_proj).
# (Lfm2MoeMLP.forward = w2(silu(w1(x)) * w3(x))  ->  w1=gate, w3=up, w2=down.)
_MOE_PEREXPERT_SCHEMES = (
    ("gate_proj", "up_proj", "down_proj"),
    ("w1", "w3", "w2"),
)


def _detect_moe_perexpert_scheme(prefix, header_metadata):
    """The (gate, up, down) per-expert disk-tensor name scheme for ``prefix`` in the shard
    header, or None. Detected from the header (any expert index, any projection), not by model
    name, so a shard holding only later experts (e.g. prefix.7.w3.weight) is still matched."""
    esc = re.escape(prefix)
    for gate_name, up_name, down_name in _MOE_PEREXPERT_SCHEMES:
        names = "|".join(re.escape(n) for n in (gate_name, up_name, down_name))
        pat = re.compile(rf"^{esc}\.\d+\.(?:{names})\.weight$")
        if any(pat.match(k) for k in header_metadata):
            return gate_name, up_name, down_name
    return None


def _count_moe_experts_in_header(prefix, header_metadata, scheme):
    """Expert count (max per-expert index + 1) for ``prefix`` under ``scheme`` in the shard
    header, or None. Seeds moe_num_experts for schemes the pre-scan misses (w1/w3/w2), so a
    down_proj-only adapter still merges every expert instead of deriving 1."""
    names = "|".join(re.escape(n) for n in scheme)
    pat = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.(?:{names})\.weight$")
    max_idx = -1
    for k in header_metadata:
        m = pat.match(k)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return (max_idx + 1) if max_idx >= 0 else None


def _resolve_moe_num_experts_with_header(prefix, resolution_stats, moe_num_experts, header_metadata, scheme):
    """Expert count for ``prefix``, preferring the authoritative source: the live module's
    num_experts then the fused-LoRA shape (both shard-independent), else the shard header's
    (max index + 1) only when neither resolves (e.g. a down_proj-only w1/w3/w2 adapter that
    derives 1). The header may only RAISE a missing/too-low count, never lower an authoritative
    one -- a low-index shard subset under-counts and would corrupt the per-expert slicing
    stride. Records a raised count and returns it (possibly None)."""
    num_experts = _resolve_moe_num_experts(prefix, resolution_stats, moe_num_experts)
    hdr_ne = _count_moe_experts_in_header(prefix, header_metadata, scheme)
    if hdr_ne and hdr_ne > (num_experts or 0):
        num_experts = hdr_ne
        moe_num_experts[prefix] = hdr_ne
    return num_experts


def _merge_moe_experts_file(mm, header_metadata, length_of_header, file, converted_lora_weights, moe_num_experts, output_dtype, counted_lora_modules, processed_mxfp4_keys, weight_block_size = None):
    count = 0
    debug_logged = 0
    file_path = getattr(file, "path", None)
    if UNSLOTH_ENABLE_LOGGING:
        try:
            logger.info(
                f"[merge_debug] Running MoE expert merge for {file_path or 'safetensors shard'}"
            )
        except Exception:
            pass

    # Check if this is GPT-OSS format (fused 3D tensors instead of per-expert 2D tensors)
    # GPT-OSS uses: model.layers.X.mlp.experts.gate_up_proj (3D tensor)
    # Gemma4 uses:  model.layers.X.experts.gate_up_proj (3D tensor, no mlp prefix)
    # Standard MoE uses: model.layers.X.mlp.experts.0.gate_proj (2D tensor per expert)
    is_gpt_oss_format = False
    for key in header_metadata.keys():
        if key.endswith(".gate_up_proj") or key.endswith(".down_proj"):
            if ".experts." in key:
                is_gpt_oss_format = True
                break

    # Determine if fused 3D format is transposed (GPT-OSS) or standard (Gemma4)
    # from gate_up_proj shape: (E, H, 2*I) → transposed, (E, 2*I, H) → standard
    _is_transposed_format = None
    if is_gpt_oss_format:
        for key, meta in header_metadata.items():
            if key.endswith(".gate_up_proj") and ".experts." in key:
                shape = meta.get("shape", [])
                if len(shape) == 3:
                    _is_transposed_format = shape[2] > shape[1]
                break

    if is_gpt_oss_format and UNSLOTH_ENABLE_LOGGING:
        try:
            logger.info(f"[merge_debug] Detected fused 3D tensor format (transposed={_is_transposed_format})")
        except Exception:
            pass

    # Build mapping from model LoRA module path -> safetensor expert prefix
    # Handles Gemma4 (.experts without .mlp), standard (.mlp.experts), and .moe patterns
    _moe_lora_to_shard_prefix = {}
    for lora_key in converted_lora_weights.keys():
        if ".experts" in lora_key or ".moe" in lora_key:
            base = lora_key.replace(".base_layer", "")
            # Try direct match first (standard models)
            if (
                f"{base}.gate_up_proj" in header_metadata
                or f"{base}.down_proj" in header_metadata
                or _detect_moe_perexpert_scheme(base, header_metadata) is not None
            ):
                _moe_lora_to_shard_prefix[lora_key] = base
            else:
                # Try remapping moe -> experts (Gemma4)
                remapped = base.replace(".moe", ".experts")
                if (
                    f"{remapped}.gate_up_proj" in header_metadata
                    or f"{remapped}.down_proj" in header_metadata
                    or _detect_moe_perexpert_scheme(remapped, header_metadata) is not None
                ):
                    _moe_lora_to_shard_prefix[lora_key] = remapped

    for lora_key, lora_stats in converted_lora_weights.items():
        shard_prefix = _moe_lora_to_shard_prefix.get(lora_key)
        if shard_prefix is None:
            continue
        is_gate = lora_key.endswith(".base_layer")
        prefix = shard_prefix

        # Handle GPT-OSS fused 3D tensor format
        if is_gpt_oss_format:
            module_updated = False
            already_counted = lora_key in counted_lora_modules

            if is_gate:
                # gate_up_proj is stored as 3D tensor: (num_experts, 2*intermediate_dim, hidden_dim)
                gate_up_key = f"{shard_prefix}.gate_up_proj"
                if gate_up_key in header_metadata:
                    gate_up_W = file.get_tensor(gate_up_key)
                    if gate_up_W.dtype in _FP8_WEIGHT_DTYPES:
                        raise RuntimeError(
                            "Unsloth: FP8 fused MoE expert LoRA merge (gate_up_proj) is not "
                            "supported; merging raw FP8 without its companion scale would "
                            "corrupt the delta. Please open an issue at "
                            "https://github.com/unslothai/unsloth/issues."
                        )
                    # Merge LoRA into fused 3D tensor
                    merged_gate_up = _merge_moe_fused_gate_up_expert(
                        gate_up_W, lora_stats, output_dtype or gate_up_W.dtype, is_transposed=_is_transposed_format
                    )
                    _write_tensor_direct_torch(
                        mm,
                        header_metadata,
                        length_of_header,
                        gate_up_key,
                        merged_gate_up,
                        gate_up_W.dtype,
                    )
                    processed_mxfp4_keys.add(gate_up_key)
                    module_updated = True
                    if UNSLOTH_ENABLE_LOGGING and debug_logged < 8:
                        try:
                            logger.info(
                                f"[merge_debug] Merged GPT-OSS gate_up_proj for {shard_prefix}"
                            )
                            debug_logged += 1
                        except Exception:
                            pass
            else:
                # down_proj is stored as 3D tensor: (num_experts, hidden_dim, intermediate_dim)
                down_key = f"{shard_prefix}.down_proj"
                if down_key in header_metadata:
                    down_W = file.get_tensor(down_key)
                    if down_W.dtype in _FP8_WEIGHT_DTYPES:
                        raise RuntimeError(
                            "Unsloth: FP8 fused MoE expert LoRA merge (down_proj) is not "
                            "supported; merging raw FP8 without its companion scale would "
                            "corrupt the delta. Please open an issue at "
                            "https://github.com/unslothai/unsloth/issues."
                        )
                    # Merge LoRA into fused 3D tensor
                    merged_down = _merge_moe_fused_down_proj_expert(
                        down_W, lora_stats, output_dtype or down_W.dtype, is_transposed=_is_transposed_format
                    )
                    _write_tensor_direct_torch(
                        mm,
                        header_metadata,
                        length_of_header,
                        down_key,
                        merged_down,
                        down_W.dtype,
                    )
                    processed_mxfp4_keys.add(down_key)
                    module_updated = True
                    if UNSLOTH_ENABLE_LOGGING and debug_logged < 8:
                        try:
                            logger.info(
                                f"[merge_debug] Merged GPT-OSS down_proj for {shard_prefix}"
                            )
                            debug_logged += 1
                        except Exception:
                            pass

            if module_updated and not already_counted:
                count += 1
                counted_lora_modules.add(lora_key)
            continue

        # Standard per-expert format (DeepSeek, Qwen3, GLM4, etc.): default
        # gate_proj/up_proj/down_proj, or w1/w3/w2 (LFM2 / some ERNIE) per the shard layout.
        _scheme = _detect_moe_perexpert_scheme(prefix, header_metadata)
        gate_name, up_name, down_name = _scheme or ("gate_proj", "up_proj", "down_proj")
        resolution_stats = lora_stats
        if getattr(resolution_stats, "module", None) is None:
            base_stats = converted_lora_weights.get(prefix + ".base_layer")
            if (
                base_stats is not None
                and getattr(base_stats, "module", None) is not None
            ):
                resolution_stats = base_stats
        num_experts = _resolve_moe_num_experts_with_header(
            prefix, resolution_stats, moe_num_experts,
            header_metadata, (gate_name, up_name, down_name),
        )
        if UNSLOTH_ENABLE_LOGGING and num_experts is not None and debug_logged < 2:
            try:
                logger.info(
                    f"[merge_debug] {lora_key}: merging {num_experts} experts via MoE merge path"
                )
                debug_logged += 1
            except Exception:
                pass
        if num_experts is None or num_experts == 0:
            if UNSLOTH_ENABLE_LOGGING and debug_logged < 4:
                try:
                    logger.info(f"[merge_debug] Skipping {lora_key}: num_experts missing")
                    debug_logged += 1
                except Exception:
                    pass
            continue

        module_updated = False
        already_counted = lora_key in counted_lora_modules
        if UNSLOTH_ENABLE_LOGGING and debug_logged < 8:
            try:
                logger.info(f"[merge_debug] Merging {lora_key} is_gate={is_gate} num_experts={num_experts} A={tuple(lora_stats.lora_A.shape) if getattr(lora_stats,'lora_A',None) is not None else None} B={tuple(lora_stats.lora_B.shape) if getattr(lora_stats,'lora_B',None) is not None else None}")
                debug_logged += 1
            except Exception:
                pass
        for expert_idx in range(num_experts):
            if is_gate:
                gate_key = f"{prefix}.{expert_idx}.{gate_name}.weight"
                up_key   = f"{prefix}.{expert_idx}.{up_name}.weight"
                if _merge_moe_expert_quant_aware(
                    "gate", gate_key, file, header_metadata, lora_stats,
                    expert_idx, num_experts, output_dtype, mm, length_of_header,
                    processed_mxfp4_keys, _merge_moe_gate_expert,
                    weight_block_size = weight_block_size,
                ):
                    module_updated = True
                if _merge_moe_expert_quant_aware(
                    "up", up_key, file, header_metadata, lora_stats,
                    expert_idx, num_experts, output_dtype, mm, length_of_header,
                    processed_mxfp4_keys, _merge_moe_up_expert,
                    weight_block_size = weight_block_size,
                ):
                    module_updated = True
            else:
                down_key = f"{prefix}.{expert_idx}.{down_name}.weight"
                if _merge_moe_expert_quant_aware(
                    "down", down_key, file, header_metadata, lora_stats,
                    expert_idx, num_experts, output_dtype, mm, length_of_header,
                    processed_mxfp4_keys, _merge_moe_down_proj_expert,
                    weight_block_size = weight_block_size,
                ):
                    module_updated = True
        if module_updated and not already_counted:
            count += 1
            counted_lora_modules.add(lora_key)
    return count


def _merge_moe_fused_gate_up_expert(gate_up_W, lora_stats, output_dtype, is_transposed=None):
    """
    Merge LoRA for fused gate_up_proj 3D tensor.
    Supports both formats:
      - Transposed (GPT-OSS): (E, H, 2*I) with lora_A (E*R, H), lora_B (2*I, E*R)
      - Standard (Gemma4):    (E, 2*I, H) with lora_A (E*R, H), lora_B (2*I, E*R)
    is_transposed: if provided, overrides dimension-based heuristic (needed when dims are equal).
    """
    _refuse_dora_on_moe(lora_stats)
    _MOE_MERGE_STATE["attempted"] += 1
    try:
        if lora_stats.lora_A is None or lora_stats.lora_B is None:
            _record_moe_merge_fallback(
                "fused_gate_up", -1, "lora_A or lora_B is None",
                lora_stats, tuple(gate_up_W.shape),
            )
            return gate_up_W

        num_experts, dim1, dim2 = gate_up_W.shape
        total_rank, dim_A = lora_stats.lora_A.shape
        dim_B, total_rank_B = lora_stats.lora_B.shape

        if total_rank_B != total_rank:
            _record_moe_merge_fallback(
                "fused_gate_up", -1,
                f"total_rank mismatch (A.shape[0]={total_rank}, B.shape[1]={total_rank_B})",
                lora_stats, tuple(gate_up_W.shape),
            )
            return gate_up_W

        rank = total_rank // num_experts
        if total_rank % num_experts != 0:
            _record_moe_merge_fallback(
                "fused_gate_up", -1,
                f"total_rank {total_rank} not divisible by num_experts {num_experts}",
                lora_stats, tuple(gate_up_W.shape),
            )
            return gate_up_W

        # LoRA dims fix the layout: standard (E,out,in) has dim_A==dim2, dim_B==dim1;
        # transposed (E,in,out) the reverse. Hint only breaks the square (in==out) tie.
        std = (dim_A == dim2 and dim_B == dim1)
        trn = (dim_A == dim1 and dim_B == dim2)
        if std and not trn:
            use_transpose = False
        elif trn and not std:
            use_transpose = True
        elif is_transposed is not None:
            use_transpose = is_transposed
        else:
            _record_moe_merge_fallback(
                "fused_gate_up", -1,
                f"layout ambiguous (W.shape={tuple(gate_up_W.shape)}, A.shape={tuple(lora_stats.lora_A.shape)}, B.shape={tuple(lora_stats.lora_B.shape)})",
                lora_stats, tuple(gate_up_W.shape),
            )
            return gate_up_W

        device = _active_merge_device()
        gate_up_merged = gate_up_W.to(device, dtype=torch.float32, non_blocking=True)
        # Move lora_A/lora_B to device once, then slice per expert.
        lora_A_dev = lora_stats.lora_A.to(device, dtype=torch.float32, non_blocking=True)
        lora_B_dev = lora_stats.lora_B.to(device, dtype=torch.float32, non_blocking=True)

        for expert_idx in range(num_experts):
            start, end = expert_idx * rank, (expert_idx + 1) * rank
            # lora_B is contiguous-r per expert (see moe_utils.py
            # _canonical_lora_weights_for_grouped_mm); must match for the saved
            # checkpoint to reproduce the training-time forward.
            delta = lora_B_dev[:, start:end] @ lora_A_dev[start:end, :]

            gate_up_merged[expert_idx] = gate_up_merged[expert_idx].add(
                delta.T if use_transpose else delta, alpha=lora_stats.alpha
            )

        _MOE_MERGE_STATE["applied"] += 1
        return gate_up_merged.to(output_dtype)
    except Exception as exc:
        _record_moe_merge_fallback(
            "fused_gate_up", -1, repr(exc),
            lora_stats, tuple(gate_up_W.shape),
        )
        return gate_up_W


def _merge_moe_fused_down_proj_expert(down_W, lora_stats, output_dtype, is_transposed=None):
    """
    Merge LoRA for fused down_proj 3D tensor.
    Supports both formats:
      - Transposed (GPT-OSS): (E, H, I) with lora_A (E*R, I), lora_B (H, E*R)
      - Standard (Gemma4):    (E, H, I) with lora_A (E*R, H), lora_B (I, E*R)
    is_transposed: if provided, overrides dimension-based heuristic (needed when H==I).
    """
    _refuse_dora_on_moe(lora_stats)
    _MOE_MERGE_STATE["attempted"] += 1
    try:
        if lora_stats.lora_A is None or lora_stats.lora_B is None:
            _record_moe_merge_fallback(
                "fused_down", -1, "lora_A or lora_B is None",
                lora_stats, tuple(down_W.shape),
            )
            return down_W

        num_experts, dim1, dim2 = down_W.shape
        total_rank, dim_A = lora_stats.lora_A.shape
        dim_B, total_rank_B = lora_stats.lora_B.shape

        if total_rank_B != total_rank:
            _record_moe_merge_fallback(
                "fused_down", -1,
                f"total_rank mismatch (A.shape[0]={total_rank}, B.shape[1]={total_rank_B})",
                lora_stats, tuple(down_W.shape),
            )
            return down_W

        rank = total_rank // num_experts
        if total_rank % num_experts != 0:
            _record_moe_merge_fallback(
                "fused_down", -1,
                f"total_rank {total_rank} not divisible by num_experts {num_experts}",
                lora_stats, tuple(down_W.shape),
            )
            return down_W

        # LoRA dims fix the layout: standard (E,out,in) has dim_A==dim2, dim_B==dim1;
        # transposed (E,in,out) the reverse. Hint only breaks the square (in==out) tie.
        std = (dim_A == dim2 and dim_B == dim1)
        trn = (dim_A == dim1 and dim_B == dim2)
        if std and not trn:
            use_transpose = False
        elif trn and not std:
            use_transpose = True
        elif is_transposed is not None:
            use_transpose = is_transposed
        else:
            _record_moe_merge_fallback(
                "fused_down", -1,
                f"layout ambiguous (W.shape={tuple(down_W.shape)}, A.shape={tuple(lora_stats.lora_A.shape)}, B.shape={tuple(lora_stats.lora_B.shape)})",
                lora_stats, tuple(down_W.shape),
            )
            return down_W

        device = _active_merge_device()
        down_merged = down_W.to(device, dtype=torch.float32, non_blocking=True)
        # Move lora_A/lora_B to device once, then slice per expert.
        lora_A_dev = lora_stats.lora_A.to(device, dtype=torch.float32, non_blocking=True)
        lora_B_dev = lora_stats.lora_B.to(device, dtype=torch.float32, non_blocking=True)

        for expert_idx in range(num_experts):
            start, end = expert_idx * rank, (expert_idx + 1) * rank
            # See _merge_moe_fused_gate_up_expert for the slicing rationale.
            delta = lora_B_dev[:, start:end] @ lora_A_dev[start:end, :]

            down_merged[expert_idx] = down_merged[expert_idx].add(
                delta.T if use_transpose else delta, alpha=lora_stats.alpha
            )

        _MOE_MERGE_STATE["applied"] += 1
        return down_merged.to(output_dtype)
    except Exception as exc:
        _record_moe_merge_fallback(
            "fused_down", -1, repr(exc),
            lora_stats, tuple(down_W.shape),
        )
        return down_W


@torch.inference_mode
def _merge_and_overwrite_lora_mxfp4(save_directory, filename, lora_weights, output_dtype, model_class_name, base_model_is_quantized=False, quant_type=None):
    # All Unsloth Zoo code licensed under LGPLv3
    # Merges LoRA and overwrites the safetensors file it was merged to
    filename_original = os.path.join(save_directory, filename)  # Original file path
    tensors = OrderedDict()
    count = 0
    safetensor_keys_seen = set()
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
        safetensor_keys_seen.update(safetensor_keys)

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

            W = None
            output_key = key
            action_logged = False

            # Handle all keys from a hybrid MXFP4 file
            if key.endswith("_blocks"):
                if convert_moe_packed_tensors is None:
                    raise ImportError("MXFP4 dequantization is required, but `convert_moe_packed_tensors` could not be imported.")

                base_name = key[:-len("_blocks")]
                scales_key = base_name + "_scales"
                output_key = base_name  # name without .weight
                if scales_key not in safetensor_keys:
                    warnings.warn(f"Found mxfp4 tensor {key} but missing its scales tensor {scales_key}. Skipping.")
                    continue

                blocks_tensor, scales_tensor = file.get_tensor(key), file.get_tensor(scales_key)

                # Free the allocator before the large dequant alloc.
                device_empty_cache()

                # Pick device + chunk size for mxfp4 dequantization
                device_type, device_id, rows_per_chunk = _choose_mxfp4_processing_strategy(
                    blocks_tensor, scales_tensor
                )

                if device_type == 'cpu':
                    try:
                        from transformers.integrations.mxfp4 import convert_moe_packed_tensors_cpu
                        W = convert_moe_packed_tensors_cpu(
                            blocks_tensor, scales_tensor, rows_per_chunk=rows_per_chunk
                        ).transpose(1, 2).contiguous()
                        if UNSLOTH_ENABLE_LOGGING:
                            logger.info(f"[DEBUG] Using CPU dequantization for {base_name} with {rows_per_chunk:,} rows per chunk")
                    except ImportError:
                        W = convert_moe_packed_tensors(
                            blocks_tensor, scales_tensor, rows_per_chunk=rows_per_chunk
                        ).transpose(1, 2).contiguous()
                else:
                    W = convert_moe_packed_tensors(
                        blocks_tensor, scales_tensor, rows_per_chunk=rows_per_chunk
                    ).transpose(1, 2).contiguous()
                    if UNSLOTH_ENABLE_LOGGING:
                        logger.info(f"[DEBUG] Using GPU dequantization for {base_name} with {rows_per_chunk:,} rows per chunk")

                processed_mxfp4_keys.add(key); processed_mxfp4_keys.add(scales_key)

                lora_stats = converted_lora_weights.get(base_name, None)
                if lora_stats and hasattr(lora_stats, 'lora_A') and lora_stats.lora_A is not None:
                    # Packed MoE experts (GPT-OSS gate_up_proj/down_proj) take this path, not the
                    # dense _merge_moe_*_expert helpers, so mirror their DoRA guard here. Otherwise a
                    # use_dora adapter on the experts bypasses the refuse and _merge_lora fails with an
                    # opaque shape error on the 3D expert group instead of the clear message.
                    _refuse_dora_on_moe(lora_stats)
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
                # 16-bit tensors (e.g. attention) coexisting with MXFP4 tensors
                W = file.get_tensor(key)


            lora_key = output_key[:-len(".weight")] if output_key.endswith(".weight") else output_key
            lora_stats = converted_lora_weights.get(lora_key, None)
            # Gemma4 ClippableLinear (.linear.weight -> .weight), mirror the standard merge loop
            if lora_stats is None and lora_key.endswith(".linear"):
                lora_stats = converted_lora_weights.get(lora_key[: -len(".linear")], None)

            if W is not None and lora_stats is not None and hasattr(lora_stats, 'lora_A') and lora_stats.lora_A is not None:
                if not action_logged:
                    count += 1
                    W = _merge_lora(W, lora_stats, output_key)
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

            # Free VRAM only after a large dequant/merge, not every small tensor.
            if W.numel() * W.element_size() >= _EMPTY_CACHE_BYTES_THRESHOLD:
                device_empty_cache()

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

    return count, safetensor_keys_seen
pass

# FP8 weight dtypes and companion scale suffixes (dropped on merge). Underscore
# variants cover fused params whose scale is <key>_scale(_inv), not <key>.weight_scale.
_FP8_WEIGHT_DTYPES = tuple(
    getattr(torch, _n) for _n in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, _n)
)
_FP8_SCALE_SUFFIXES = (".weight_scale_inv", ".weight_scale", ".input_scale",
                       "_scale_inv", "_scale")
# safetensors header dtype tags for FP8 weights (used to find genuine scale companions).
_FP8_HEADER_DTYPES = ("F8_E4M3", "F8_E5M2")

def _fp8_dequantize_weight(file, header_metadata, weight_key, weight_block_size = None, extra_scale_lookup = None):
    """Dequantize one FP8 weight; return (W_real, [scale_keys to drop]).

    Raises if FP8 with no usable scale (never silent). Handles every dense scale layout
    (per-tensor, 1-D, 2-D per-channel/block) plus 3-D fused MoE experts (per-expert).
    weight_block_size (bm, bn) is needed for a weight dim that isn't a block multiple.
    extra_scale_lookup(key) loads a scale companion from a sibling shard when HF sharding
    placed the weight and its scale in different files.
    """
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import _fp8_dequant_blockwise
    W = file.get_tensor(weight_key)
    if W.dtype not in _FP8_WEIGHT_DTYPES:
        return W, []
    base = weight_key[: -len(".weight")] if weight_key.endswith(".weight") else weight_key
    # <base>.weight_scale(_inv) for .weight; <key>_scale(_inv) for fused params.
    scale_inv = None
    for suffix in (".weight_scale_inv", ".weight_scale", "_scale_inv", "_scale"):
        cand = base + suffix
        if cand in header_metadata:
            scale_inv = file.get_tensor(cand)
            break
    if scale_inv is None and extra_scale_lookup is not None:
        # Scale companion landed in a different shard (HF shards weight/scale independently).
        for suffix in (".weight_scale_inv", ".weight_scale", "_scale_inv", "_scale"):
            scale_inv = extra_scale_lookup(base + suffix)
            if scale_inv is not None:
                break
    if scale_inv is None:
        raise RuntimeError(
            f"Unsloth: FP8 weight '{weight_key}' has no companion weight_scale / "
            "weight_scale_inv; cannot dequantize to 16bit. The merged model would be "
            "corrupted. Please file a bug at https://github.com/unslothai/unsloth/issues."
        )
    # Drop every companion scale for this weight (input_scale too on static FP8).
    scale_keys = [base + s for s in _FP8_SCALE_SUFFIXES if base + s in header_metadata]
    # 3-D fused MoE experts (E, M, N): dequantize each expert with its scale slice.
    # Expert-LoRA merges are refused upstream, so only base experts reach here.
    if W.ndim == 3:
        n_experts = W.shape[0]
        if scale_inv.numel() == 1:
            expert_scales = [scale_inv] * n_experts
        elif scale_inv.shape[0] == n_experts:
            expert_scales = [scale_inv[e] for e in range(n_experts)]
        else:
            raise RuntimeError(
                f"Unsloth: FP8 fused-expert weight '{weight_key}' shape {tuple(W.shape)} has "
                f"no per-expert scale aligned to its leading dim (scale {tuple(scale_inv.shape)})."
            )
        W_real = torch.stack(
            [_fp8_dequant_blockwise(W[e], expert_scales[e], block_size = weight_block_size)
             for e in range(n_experts)],
            dim = 0,
        )
        return W_real, scale_keys
    if W.ndim != 2:
        raise RuntimeError(
            f"Unsloth: FP8 weight '{weight_key}' is {W.ndim}-D; only 2-D dense and 3-D "
            "fused-expert FP8 weights are supported on a 16bit merge."
        )
    rows, cols = W.shape
    _scale_ok = (
        scale_inv.numel() == 1
        or (scale_inv.ndim == 1 and scale_inv.shape[0] in (rows, cols))
        or (scale_inv.ndim == 2 and rows % scale_inv.shape[0] == 0 and cols % scale_inv.shape[1] == 0)
        # Configured block size: only if its ceil grid matches the stored scale grid, else a
        # grid for a different block size would silently dequantize with inferred tiles.
        or (scale_inv.ndim == 2 and weight_block_size is not None and len(weight_block_size) == 2
            and scale_inv.shape[0] == -(-rows // weight_block_size[0])
            and scale_inv.shape[1] == -(-cols // weight_block_size[1]))
    )
    if not _scale_ok:
        raise RuntimeError(
            f"Unsloth: FP8 weight '{weight_key}' shape {tuple(W.shape)} is not tiled by "
            f"its scale {tuple(scale_inv.shape)}; cannot dequantize safely."
        )
    W_real = _fp8_dequant_blockwise(W, scale_inv, block_size = weight_block_size)
    return W_real, scale_keys

def _build_cross_shard_fp8_index(save_directory, current_filename):
    """{key: (filename, dtype)} for tensors in the OTHER *.safetensors shards (headers only,
    no tensor reads). HF shards a weight and its scale companion independently, so an FP8
    weight and its `*.weight_scale(_inv)` can land in different files; this lets the rewrite
    find a scale elsewhere and drop a scale whose weight was dequantized elsewhere."""
    index = {}
    try:
        siblings = [f for f in os.listdir(save_directory)
                    if f.endswith(".safetensors") and f != current_filename]
    except OSError:
        return index
    for fn in siblings:
        try:
            with open(os.path.join(save_directory, fn), "rb") as fp:
                length = int.from_bytes(fp.read(8), "little")
                hdr = json.loads(fp.read(length))
        except Exception:
            continue
        for k, meta in hdr.items():
            if k != "__metadata__" and isinstance(meta, dict):
                index.setdefault(k, (fn, meta.get("dtype")))
    return index
pass

def _fp8_scale_key_weight_bases(scale_key):
    """Candidate FP8 weight keys a companion scale belongs to (most specific suffix wins),
    used to drop a scale whose dequantized weight lives in another shard."""
    for suffix in (".weight_scale_inv", ".weight_scale", ".input_scale", "_scale_inv", "_scale"):
        if scale_key.endswith(suffix):
            base = scale_key[: -len(suffix)]
            return (base + ".weight", base)  # .weight_* -> <base>.weight ; fused -> <base>
    return ()
pass

def _collect_fp8_weight_keys(save_directory, filenames):
    """Keys whose current on-disk dtype is FP8 (header-only read). Capture this BEFORE the
    FP8 -> 16bit rewrite so the post-rewrite scale cleanup can anchor on the weights that were
    actually FP8, rather than on a name-suffix heuristic that would also match unrelated
    `*_scale` buffers."""
    fp8_keys = set()
    for filename in filenames:
        path = os.path.join(save_directory, filename)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as fp:
                length = int.from_bytes(fp.read(8), "little")
                header = json.loads(fp.read(length))
        except Exception:
            continue
        for key, meta in header.items():
            if key != "__metadata__" and isinstance(meta, dict) and meta.get("dtype") in _FP8_HEADER_DTYPES:
                fp8_keys.add(key)
    return fp8_keys
pass

def _drop_resolved_fp8_scales_after_rewrite(save_directory, filenames, prerewrite_fp8_keys):
    """Order-independent cleanup run AFTER every FP8 shard is rewritten: drop each companion
    scale of a weight that WAS FP8 before the rewrite (prerewrite_fp8_keys), even when the
    weight and its scale were split across shards. Per-shard processing only drops same-shard
    companions, so a cross-shard scale would otherwise survive regardless of processing order.

    Anchored on the pre-rewrite FP8 weight set, NOT on a post-rewrite `*_scale` name match:
    once every weight is 16bit, a name heuristic ("base weight now exists and is non-FP8")
    would also delete an unrelated `*_scale` buffer (router / logit scales, etc.) whose base
    weight happens to exist. Returns the set of removed keys."""
    if not prerewrite_fp8_keys:
        return set()
    headers = {}
    for filename in filenames:
        path = os.path.join(save_directory, filename)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as fp:
                length = int.from_bytes(fp.read(8), "little")
                header = json.loads(fp.read(length))
        except Exception:
            continue
        headers[filename] = header

    removed = set()
    for filename, header in headers.items():
        drop = set()
        for key in header:
            if key == "__metadata__":
                continue
            if any(wk in prerewrite_fp8_keys for wk in _fp8_scale_key_weight_bases(key)):
                drop.add(key)
        if not drop:
            continue
        path = os.path.join(save_directory, filename)
        tensors = OrderedDict()
        with safe_open(path, framework = "pt", device = "cpu") as f:
            for key in f.keys():
                if key not in drop:
                    tensors[key] = f.get_tensor(key).contiguous()
        with tempfile.NamedTemporaryFile(suffix = ".safetensors", dir = save_directory, delete = False) as tmp:
            tmp_path = tmp.name
        save_file(tensors, tmp_path, metadata = {"format": "pt"})
        os.replace(tmp_path, path)
        removed.update(drop)
    return removed
pass

def _merge_and_overwrite_lora_fp8(save_directory, filename, lora_weights, output_dtype, model_class_name, tie_word_embeddings = False, weight_block_size = None):
    # All Unsloth Zoo code licensed under LGPLv3
    # Dequantize FP8 to 16bit, merge LoRA, drop scales, atomically rewrite the shard.
    filename_original = os.path.join(save_directory, filename)
    tensors = OrderedDict()
    count = 0
    safetensor_keys_seen = set()

    with safe_open(filename_original, framework = "pt", device = "cpu") as file:
        safetensor_keys = list(file.keys())
        safetensor_keys_seen.update(safetensor_keys)

        # Read the header to skip scale companions without a tensor read. Read-only: the merge
        # writes a temp file and os.replace()s it, so no write handle is needed (and "r+b"
        # alongside safe_open's handle can trigger a PermissionError on Windows).
        raw_pointer = open(filename_original, "rb")
        try:
            length_of_header = int.from_bytes(raw_pointer.read(8), "little")
            header_metadata = json.loads(raw_pointer.read(length_of_header))
        finally:
            raw_pointer.close()

        converted_lora_weights = _convert_lora_keys_to_safetensor_format(
            lora_weights, safetensor_keys, model_class_name = model_class_name,
        )

        # Dense path has no MoE fusion; refuse a fused-expert LoRA rather than drop it.
        if any(isinstance(k, str) and (".experts" in k or ".moe" in k) for k in converted_lora_weights):
            raise RuntimeError(
                "Unsloth: FP8 dequant-on-merge does not yet support LoRA adapters on "
                "MoE experts. Please open an issue at "
                "https://github.com/unslothai/unsloth/issues."
            )

        # FP8 weight/scale pairs can be split across shards, so index the other shards once.
        cross_shard = _build_cross_shard_fp8_index(save_directory, filename)

        def _load_cross_shard_scale(scale_key):
            entry = cross_shard.get(scale_key)
            if entry is None:
                return None
            with safe_open(os.path.join(save_directory, entry[0]), framework = "pt", device = "cpu") as f2:
                return f2.get_tensor(scale_key)

        # Companion scales of an actual FP8 weight (by header dtype) are folded in by the
        # dequant, so they are dropped. Derive them from the FP8 weights rather than by raw
        # suffix, so unrelated `*_scale` / `*_scale_inv` tensors (logit_scale, router
        # per_expert_scale, ...) are not silently lost.
        scale_keys_to_drop = set()
        for key in safetensor_keys:
            if header_metadata.get(key, {}).get("dtype") not in _FP8_HEADER_DTYPES:
                continue
            base = key[: -len(".weight")] if key.endswith(".weight") else key
            scale_keys_to_drop.update(base + s for s in _FP8_SCALE_SUFFIXES if base + s in header_metadata)
        # A scale whose FP8 weight lives in ANOTHER shard is left in place here and removed by
        # the order-independent post-rewrite cleanup (_drop_resolved_fp8_scales_after_rewrite),
        # so it survives until that weight's shard has loaded it cross-shard.

        for key in safetensor_keys:
            if key in scale_keys_to_drop:
                continue

            was_fp8 = header_metadata.get(key, {}).get("dtype") in _FP8_HEADER_DTYPES
            merged = False
            W, _scale_keys = _fp8_dequantize_weight(file, header_metadata, key, weight_block_size = weight_block_size, extra_scale_lookup = _load_cross_shard_scale)

            output_key = key
            lora_key = output_key[:-len(".weight")] if output_key.endswith(".weight") else output_key
            lora_stats = converted_lora_weights.get(lora_key, None)
            if lora_stats is None and lora_key.endswith(".linear"):
                lora_stats = converted_lora_weights.get(lora_key[: -len(".linear")], None)
            # Tied embeddings: fold an lm_head LoRA onto embed_tokens (shared base tensor).
            if lora_stats is None and tie_word_embeddings and lora_key.endswith("embed_tokens"):
                lm_head_key = lora_key[: -len("embed_tokens")] + "lm_head"
                lora_stats = converted_lora_weights.get(lm_head_key, None)
                if lora_stats is None and lm_head_key.startswith("model."):
                    lora_stats = converted_lora_weights.get(lm_head_key[len("model."):], None)
                if lora_stats is None and not lm_head_key.startswith("model."):
                    lora_stats = converted_lora_weights.get("model." + lm_head_key, None)
            if lora_stats is not None:
                if getattr(lora_stats, "lora_A", None) is None and getattr(lora_stats, "module", None) is not None:
                    # modules_to_save (e.g. resized embed/lm_head): take the saved weight.
                    saved_weight = _get_modules_to_save_weight(lora_stats.module)
                    if saved_weight is None:
                        saved_weight = getattr(lora_stats.module, "weight", None)
                    if saved_weight is not None:
                        W = saved_weight.to(W.device, dtype = torch.float32)
                        merged = True
                        count += 1
                elif getattr(lora_stats, "lora_A", None) is not None:
                    W = _merge_lora(W, lora_stats, output_key)
                    merged = True
                    count += 1

            # Dequantized FP8 or LoRA-merged tensors take output_dtype; untouched non-FP8
            # buffers (int64/bool/fp32, e.g. inv_freq) keep their dtype, as the in-place path does.
            write_dtype = output_dtype if (was_fp8 or merged) else W.dtype
            tensors[output_key] = W.to(device = "cpu", dtype = write_dtype).contiguous()
            del W
            if tensors[output_key].numel() * tensors[output_key].element_size() >= _EMPTY_CACHE_BYTES_THRESHOLD:
                device_empty_cache()

        # Remove the dropped scale companions from the rewritten shard.
        for sk in scale_keys_to_drop:
            tensors.pop(sk, None)
        for k in list(safetensor_keys):
            if k in scale_keys_to_drop:
                safetensor_keys_seen.discard(k)

    if os.name == 'nt':
        gc.collect()
        time.sleep(0.1)

    with tempfile.NamedTemporaryFile(suffix=".safetensors", dir=save_directory, delete=False) as tmpfile:
        temp_filename_safetensors = tmpfile.name
    save_file(tensors, temp_filename_safetensors, metadata={"format": "pt"})
    try:
        os.replace(temp_filename_safetensors, filename_original)
    except OSError as e:
        print(f"Error renaming temporary file: {e}. Attempting copy and replace.")
        shutil.copy2(temp_filename_safetensors, filename_original)
        try: os.remove(temp_filename_safetensors)
        except: pass

    return count, safetensor_keys_seen
pass

from huggingface_hub import (
    split_state_dict_into_shards_factory,
    get_torch_storage_size,
    get_torch_storage_id,
)

def get_torch_storage_size_new(x, element_size):
    if isinstance(x, LoraStats):
        mod = x.module
        # modules_to_save: use the saved weight shape directly
        saved_w = _get_modules_to_save_weight(mod)
        if saved_w is None and hasattr(mod, "weight"):
            saved_w = mod.weight
        if saved_w is not None and hasattr(saved_w, "shape"):
            return int(np.prod(saved_w.shape)) * element_size
        # MoE LoRA wrappers with no .base_layer: infer merged shape from lora matrices
        if mod is None and x.lora_A is not None and x.lora_B is not None:
            shape = (x.lora_B.shape[0], x.lora_A.shape[1])
            return int(np.prod(shape)) * element_size
        # Fallback for Linear-like modules
        shape = (mod.in_features, mod.out_features)
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
    # Strip quantization_config from the top level AND nested sub-configs. VLMs keep it under
    # text_config/vision_config, so a merged_16bit export would ship bf16 weights still labelled
    # load_in_4bit there -> on reload transformers builds the bnb quantizer for full-precision
    # weights ("Cannot copy out of meta tensor"). Recurse to remove it wherever it lives.
    def _strip_quantization_config(obj):
        removed = False
        if isinstance(obj, dict):
            if "quantization_config" in obj:
                del obj["quantization_config"]
                removed = True
            for value in obj.values():
                if _strip_quantization_config(value):
                    removed = True
        elif isinstance(obj, list):
            for value in obj:
                if _strip_quantization_config(value):
                    removed = True
        return removed
    if not _strip_quantization_config(config):
        return
    # Overwrite the config file
    with open(config_path, "w", encoding = "utf-8") as f:
        json.dump(config, f, indent = 4)
    pass
pass

def _remove_transformers_version(config_path: Path):
    if not config_path.exists():
        return
    try:
        with open(config_path, "r", encoding = "utf-8") as f:
            config = json.load(f)
    except Exception:
        return
    if "transformers_version" not in config:
        return
    del config["transformers_version"]
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
                with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                    f = json.load(f)
                if "chat_template" not in f or f["chat_template"] is None:
                    f["chat_template"] = tokenizer.chat_template
                with open(tokenizer_config_path, "w", encoding="utf-8") as new_f:
                    json.dump(f, new_f, indent = 2, ensure_ascii = False)
            except:
                pass
        pass

        # Remove chat_template if NULL
        try:
            with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                f = json.load(f)
            if "chat_template" in f and (f["chat_template"] == "" or f["chat_template"] is None):
                del f["chat_template"]
            with open(tokenizer_config_path, "w", encoding="utf-8") as new_f:
                json.dump(f, new_f, indent = 2, ensure_ascii = False)
        except:
            pass
    pass
    # Fix config.json using torch_dtype / dtype
    config_file_path = os.path.join(saved_folder, "config.json")
    if os.path.exists(config_file_path):
        try:
            with open(config_file_path, "r", encoding="utf-8") as f:
                data = f.read()
            data = data.replace('"dtype"', '"torch_dtype"')
            data = data.replace("'dtype'", "'torch_dtype'")
            with open(config_file_path, "w", encoding="utf-8") as f:
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
        # For a 16bit merge of an FP8 base, prefer an existing 16bit sibling (e.g.
        # unsloth/GLM-5.2-FP8 -> unsloth/GLM-5.2) and merge LoRA onto full-precision
        # weights, mirroring the 4bit flow. Only dequantize the FP8 if no sibling exists.
        if base_model_is_quantized and quant_type == "fp8" and save_method == "merged_16bit":
            _sibling = _resolve_fp8_16bit_sibling(model_name, token)
            if _sibling is not None:
                if UNSLOTH_ENABLE_LOGGING:
                    logger.info(f"Unsloth: FP8 base detected; merging onto 16bit sibling `{_sibling}`.")
                model_name = _sibling
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

            # Get safetensors files from local directory.
            # Mistral-7B-v0.3, Codestral-22B, Mistral-Nemo and Mistral-Small ship a
            # consolidated.safetensors that duplicates their proper shards. When
            # proper shards coexist we drop it (it would double download/disk and
            # cause T4 OOM during the merge pass), but if it is the only file we
            # keep it so the user can still merge such a local directory.
            _local_st = [f for f in os.listdir(model_name) if f.endswith(".safetensors")]
            _has_proper_shards = any(f != "consolidated.safetensors" for f in _local_st)
            for file in _local_st:
                if _has_proper_shards and file == "consolidated.safetensors":
                    continue
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
                            else:
                                # Drop stale/duplicate shards the index doesn't reference
                                # (mirrors the HF-repo branch below): a local snapshot can carry a
                                # leftover non-indexed shard set (e.g. granite-3.2-8b) whose shapes
                                # differ -> "Bad in-place call". Filter only when extra shards
                                # exist, so well-formed dirs are untouched.
                                _indexed = {os.path.split(v)[-1] for v in index_data["weight_map"].values()}
                                if _indexed and not set(safetensors_list).issubset(_indexed):
                                    _kept = [s for s in safetensors_list if s in _indexed]
                                    if _kept and len(_kept) != len(safetensors_list):
                                        safetensors_list    = _kept
                                        max_size_in_bytes   = 0
                                        total_size_in_bytes = 0
                                        for _s in safetensors_list:
                                            _sp = os.path.join(model_name, _s)
                                            if os.path.exists(_sp):
                                                _sz = os.path.getsize(_sp)
                                                max_size_in_bytes   = max(max_size_in_bytes, _sz)
                                                total_size_in_bytes += _sz
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

            # Process HF file listing. Same soft filter as the local branch above:
            # drop consolidated.safetensors only when proper shards coexist.
            _hf_entries = [x for x in file_list if x["name"].endswith(".safetensors")]
            _hf_has_proper = any(
                os.path.split(x["name"])[-1] != "consolidated.safetensors"
                for x in _hf_entries
            )
            for x in _hf_entries:
                fname = os.path.split(x["name"])[-1]
                if _hf_has_proper and fname == "consolidated.safetensors":
                    continue
                safetensors_list.append(fname)
                max_size_in_bytes = max(max_size_in_bytes, x["size"])
                total_size_in_bytes += x["size"]

            # Drop stale/duplicate shard sets the index doesn't reference. Some repos (e.g.
            # granite-3.2-8b-instruct) ship a leftover second shard set next to the real one while
            # the index references only the real set; merging into a stale shard whose shapes
            # differ raises "Bad in-place call". Filter only when extra shards exist.
            try:
                from huggingface_hub import hf_hub_download as _hf_hub_download
                _idx_path = _hf_hub_download(
                    repo_id  = model_name,
                    filename = "model.safetensors.index.json",
                    token    = token,
                )
                with open(_idx_path, "r", encoding = "utf-8") as f:
                    _weight_map = json.load(f).get("weight_map", {})
                _indexed = {os.path.split(v)[-1] for v in _weight_map.values()}
                if _indexed and not set(safetensors_list).issubset(_indexed):
                    _kept = [s for s in safetensors_list if s in _indexed]
                    if _kept and len(_kept) != len(safetensors_list):
                        _sizes = {os.path.split(x["name"])[-1] : x["size"] for x in _hf_entries}
                        safetensors_list      = _kept
                        max_size_in_bytes     = 0
                        total_size_in_bytes   = 0
                        for _s in safetensors_list:
                            _sz = _sizes.get(_s, 0)
                            max_size_in_bytes   = max(max_size_in_bytes, _sz)
                            total_size_in_bytes += _sz
            except Exception:
                # Index-based filtering is best-effort: if the index cannot be
                # fetched/parsed, fall back to the full shard list found above.
                pass

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
        _remove_transformers_version(config_path = Path(save_directory) / "config.json")
        # #5410: keep trained eos / sampling defaults on reload.
        try:
            gen_cfg = getattr(model, "generation_config", None)
            if gen_cfg is None:
                gen_cfg = getattr(getattr(model, "config", None), "generation_config", None)
            if gen_cfg is not None:
                gen_cfg.save_pretrained(save_directory)
        except Exception as gen_cfg_err:
            print(f"Unsloth: failed to save generation_config.json: {gen_cfg_err}")
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
    is_t4 = DEVICE_TYPE == "cuda" and "Tesla T4" in torch.cuda.get_device_name(0)
    needs_splitting = should_split_shards(is_t4, config, safetensors_list, max_size_in_bytes) if save_method == "merged_16bit" else False
    _hf_cache_dir = _get_hf_cache_dir()
    copied_all_from_cache = False
    copied_tokenizer_model_from_cache = False
    is_hf_sharded = is_hf_sharded_safetensors(safetensors_list)
    safe_tensor_index_files = ["model.safetensors.index.json"] if (len(safetensors_list) > 1 or is_hf_sharded) else []

    # The original index lists scale keys, so it goes stale on MXFP4/FP8 dequant; skip
    # copying it (regenerated below). FP8 only dequantizes on a merged_16bit save, so an
    # FP8 base saved another way keeps its scales and must reuse the original index.
    _is_quant_dequant = (
        base_model_is_quantized and quant_type == "mxfp4" and save_method != "mxfp4"
    ) or (base_model_is_quantized and quant_type == "fp8" and save_method == "merged_16bit")
    # ONLY download/copy the original index if we are NOT dequantizing a quantized model
    if not _is_quant_dequant and not needs_splitting:
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
                    cache_dir = _hf_cache_dir,
                    token = token,
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
            cache_dir = _hf_cache_dir,
            token = token,
        )

    if not copied_tokenizer_model_from_cache and not low_disk_space_usage and not is_local_path:
        print(f"Attempting to download tokenizer.model for {model_name}...")
        snapshot_download(
            repo_id = model_name,
            local_dir = save_directory,
            allow_patterns = ["tokenizer.model"],
            local_dir_use_symlinks = False,
            cache_dir = _hf_cache_dir,
            token = token,
        )

    final_safetensors_list = []

    _reset_moe_merge_state()  # #5410

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
                cache_dir = _hf_cache_dir,
                token = token,
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
                    cache_dir = _hf_cache_dir,
                    token = token,
                )
                print("Downloaded tokenizer.model")
            except Exception as e:
                # It's OK if the file doesn't exist (not all models have it)
                print(f"Note: tokenizer.model not found (this is OK for non-SentencePiece models)")

    if needs_splitting:
        final_safetensors_list = renumber_safetensor_files(final_safetensors_list, save_directory)

    is_final_safetensors_list_sharded = is_hf_sharded_safetensors(final_safetensors_list)
    # FP8 dequant drops companion scale keys, so a sharded index must be rebuilt too.
    # Mirror the actual dequant conditions (mxfp4: any non-mxfp4 save; fp8: merged_16bit)
    # so a non-dequantizing FP8 save keeps a correct index instead of none.
    _quant_dequant_index = (
        base_model_is_quantized and quant_type == "mxfp4" and save_method != "mxfp4"
    ) or (base_model_is_quantized and quant_type == "fp8" and save_method == "merged_16bit")
    regenerate_index = (_quant_dequant_index or needs_splitting) and (len(final_safetensors_list) > 1 or is_final_safetensors_list_sharded) and save_method != "mxfp4"
    weight_map = {}

    # Collect all tensor keys encountered across shards so we can reason about tied embeddings
    # (embed_tokens/lm_head) in the final sanity check without assuming both tensors exist on disk.
    safetensor_keys_seen = set()
    counted_lora_modules_global = set()

    # Per-shard-invariant: resolve the base model + tie flag once, not per shard.
    _merge_base_model = find_lora_base_model(model)
    _merge_model_class_name = _merge_base_model.__class__.__name__
    _merge_tie_word_embeddings = bool(
        getattr(_merge_base_model.config, "tie_word_embeddings", False)
    )
    # FP8 block dequant needs the block size for partial final blocks; capture it
    # before merge16bit strips the config. Top-level for finegrained_fp8/fbgemm;
    # under config_groups[*].weights.block_structure for compressed-tensors.
    _merge_weight_block_size = None
    if base_model_is_quantized and quant_type == "fp8":
        _qc = getattr(_merge_base_model.config, "quantization_config", None)
        _qc_get = (_qc.get if isinstance(_qc, dict) else (lambda k, d = None: getattr(_qc, k, d))) if _qc is not None else None
        if _qc_get is not None:
            _wbs = _qc_get("weight_block_size", None)
            if _wbs is None:
                # config_groups (and its weights) may be dicts or transformers objects.
                _grps = _qc_get("config_groups", None) or {}
                _grps_iter = _grps.values() if hasattr(_grps, "values") else _grps
                for _grp in _grps_iter:
                    _w = _grp.get("weights") if isinstance(_grp, dict) else getattr(_grp, "weights", None)
                    _bs = _w.get("block_structure") if isinstance(_w, dict) else getattr(_w, "block_structure", None)
                    if isinstance(_bs, (list, tuple)) and len(_bs) == 2:
                        _wbs = _bs
                        break
            if isinstance(_wbs, (list, tuple)) and len(_wbs) == 2:
                _merge_weight_block_size = tuple(int(x) for x in _wbs)
    # Gated archs + 16bit merge only: fold each LoRA delta onto dequant(W4) instead of W16
    # (see _merge_lora). Strict no-op for every other model/merge.
    _use_dequant_base = (
        save_method == "merged_16bit"
        and _model_type_needs_dequant_merge_base(_merge_base_model)
    )
    if _use_dequant_base:
        warnings.warn(
            "Unsloth: merging each LoRA delta onto the dequantized 4bit base "
            "dequant(W4) (the weights the QLoRA adapter trained against) instead of "
            "the downloaded 16bit base, to keep the merged_16bit checkpoint faithful "
            f"for model_type={getattr(getattr(_merge_base_model, 'config', None), 'model_type', '?')}."
        )

    # FP8 MoE-expert LoRA + merged_16bit: the dense FP8 rewrite cannot fuse per-expert
    # adapters, so dequantize the whole model to 16bit first (dense rewrite with no LoRA +
    # cross-shard scale cleanup), then merge the expert adapters with the standard 16bit MoE
    # path in the loop below. Keeps a genuine 16bit output instead of FP8-labelled-as-16bit.
    if (base_model_is_quantized and quant_type == "fp8" and save_method == "merged_16bit"
            and any(isinstance(k, str) and (".experts" in k or ".moe" in k) for k in lora_weights)):
        if UNSLOTH_ENABLE_LOGGING:
            logger.info("FP8 MoE-expert LoRA detected: dequantizing to 16bit, then merging experts.")
        _prerewrite_fp8_keys = _collect_fp8_weight_keys(save_directory, final_safetensors_list)
        for _fn in final_safetensors_list:
            _merge_and_overwrite_lora_fp8(
                save_directory, _fn, defaultdict(lambda: None), output_dtype, _merge_model_class_name,
                tie_word_embeddings = _merge_tie_word_embeddings,
                weight_block_size = _merge_weight_block_size,
            )
        _drop_resolved_fp8_scales_after_rewrite(save_directory, final_safetensors_list, _prerewrite_fp8_keys)
        # Model is now 16bit on disk; merge expert LoRA via the standard (non-quantized) path.
        base_model_is_quantized = False
        quant_type = None
        _merge_weight_block_size = None
    pass

    # FP8 merged_16bit can load a weight's scale from a sibling shard, so the index build and
    # the low-disk upload/remove must wait until every shard is rewritten and the cross-shard
    # scale cleanup has run (otherwise an early shard is removed or indexed with a stale scale).
    _fp8_post_cleanup = base_model_is_quantized and quant_type == "fp8" and save_method == "merged_16bit"
    # Capture FP8 weight keys BEFORE the loop rewrites them to 16bit, so the post-rewrite scale
    # cleanup anchors on weights that were actually FP8 (not a `*_scale` name heuristic).
    _fp8_prerewrite_keys = _collect_fp8_weight_keys(save_directory, final_safetensors_list) if _fp8_post_cleanup else set()
    _defer_low_disk = low_disk_space_usage and push_to_hub and _fp8_post_cleanup
    for filename in ProgressBar(final_safetensors_list, desc=f'Unsloth: Merging weights into {"mxfp4" if save_method=="mxfp4" else "16bit"}'):
        merged_count, shard_keys = _merge_and_overwrite_lora(
            save_directory = save_directory,
            filename = filename,
            lora_weights = lora_weights,
            output_dtype = output_dtype,
            model_class_name = _merge_model_class_name,
            base_model_is_quantized = base_model_is_quantized,
            quant_type = quant_type,
            save_method = save_method,
            counted_lora_modules = counted_lora_modules_global,
            tie_word_embeddings = _merge_tie_word_embeddings,
            weight_block_size = _merge_weight_block_size,
            use_dequant_base = _use_dequant_base,
        )
        n_saved_modules += merged_count
        safetensor_keys_seen.update(shard_keys)
        device_empty_cache()

        file_path = os.path.join(save_directory, filename)

        # --- NEW LOGIC: Build the weight_map BEFORE deleting the file ---
        if regenerate_index and not _fp8_post_cleanup:
            # We must open the file we just created to get its tensor keys
            with safe_open(file_path, framework = "pt", device = "cpu") as f:
                for key in f.keys():
                    weight_map[key] = filename

        if low_disk_space_usage and push_to_hub and not _defer_low_disk:
            upload_items(filename)
            os.remove(os.path.join(save_directory, filename)) # Remove to conserve disk space
        pass
    pass

    # FP8 cross-shard scale cleanup: order-independent pass after every shard is rewritten.
    if _fp8_post_cleanup:
        for _removed_key in _drop_resolved_fp8_scales_after_rewrite(save_directory, final_safetensors_list, _fp8_prerewrite_keys):
            safetensor_keys_seen.discard(_removed_key)
        if regenerate_index:
            for filename in final_safetensors_list:
                file_path = os.path.join(save_directory, filename)
                if not os.path.exists(file_path):
                    continue
                with safe_open(file_path, framework = "pt", device = "cpu") as f:
                    for key in f.keys():
                        weight_map[key] = filename
        if _defer_low_disk:
            for filename in final_safetensors_list:
                upload_items(filename)
                try:
                    os.remove(os.path.join(save_directory, filename))
                except FileNotFoundError:
                    pass
    pass

    # Step 6: Regenerate index for MXFP4 dequantization or shard splitting
    if regenerate_index:
        # The logic is now simpler: we just write the map we already built.
        print("Unsloth: Regenerating safetensors index...")

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
    # Count only LoRA modules backed by a saved tensor, using the merge loop's key
    # resolution (remap, Gemma4 .linear, fused MoE, tied lm_head -> embed_tokens), so the
    # count equals what the merge writes and needs no tied discount. len(lora_weights)
    # over-counts unbacked targets such as a vision tower absent from the base.
    _base = find_lora_base_model(model)
    # Native mxfp4 save preserves _blocks/_scales instead of merging, so a LoRA on a packed
    # tensor is not written; don't count it as backed there.
    _count_packed_mxfp4 = not (base_model_is_quantized and quant_type == "mxfp4" and save_method == "mxfp4")
    effective_loras = _count_backed_lora_modules(
        lora_weights,
        safetensor_keys_seen,
        _base.__class__.__name__,
        bool(getattr(_base.config, "tie_word_embeddings", False)),
        count_packed_mxfp4 = _count_packed_mxfp4,
    )

    if effective_loras != n_saved_modules:
        raise RuntimeError(
            f"Unsloth: Saving LoRA finetune failed since # of LoRAs = {effective_loras} "\
            f"does not match # of saved modules = {n_saved_modules}. Please file a bug report!"
        )
    pass

    if _MOE_MERGE_STATE["fallback"] > 0:  # #5410: never claim success on a partial merge
        err = _MOE_MERGE_STATE.get("first_error") or {}
        raise RuntimeError(
            "Unsloth: MoE LoRA merge fell back to the base weight on "
            f"{_MOE_MERGE_STATE['fallback']} per-expert tensor(s) "
            f"(of {_MOE_MERGE_STATE['attempted']} attempted, {_MOE_MERGE_STATE['applied']} applied). "
            "The merged checkpoint would be missing the expert LoRA delta. "
            f"First failure: role={err.get('role')} expert={err.get('expert_idx')} "
            f"reason={err.get('reason')} lora_A={err.get('lora_A_shape')} "
            f"lora_B={err.get('lora_B_shape')} per_expert_W={err.get('per_expert_W')}. "
            "This usually means an unrecognised PEFT/Transformers MoE LoRA layout. "
            "Please file a bug at https://github.com/unslothai/unsloth-zoo/issues "
            "with these shapes."
        )
    if _MOE_MERGE_STATE["attempted"] > 0:
        print(
            f"Unsloth: MoE LoRA merge applied to "
            f"{_MOE_MERGE_STATE['applied']}/{_MOE_MERGE_STATE['attempted']} "
            "per-expert tensors."
        )

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
    """If ALL files exist in the HF cache, copy them into target_dir_str.
    Returns True on success, False otherwise.
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
            cached_path_str = hf_hub_download(
                repo_id = repo_id,
                filename = filename,
                local_files_only = True,
                repo_type = "model",
                cache_dir = hf_cache_dir,
                token = token,
            )
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
    # Resolve through hf_cache._active_caches so cache reuse sees the same
    # location Hub uses (XDG_CACHE_HOME, legacy HUGGINGFACE_HUB_CACHE,
    # expanded env values, unresolvable-home handling).
    from .hf_cache import _active_caches
    _, cache_dir, _ = _active_caches()

    if cache_dir is not None:
        try:
            if cache_dir.is_dir():
                # Need R/W/X for HF's lock files and internal operations
                if os.access(cache_dir, os.R_OK | os.W_OK | os.X_OK):
                    print(f"Found HuggingFace hub cache directory: {cache_dir.resolve()}")
                    return cache_dir.resolve()
                else:
                    print(f"Warning: Found cache directory {cache_dir}, but lack R/W/X permissions. Cannot use cache.")
                    return None
            elif cache_dir.exists():
                 # Exists but not a directory: bail
                 print(f"Warning: Path {cache_dir} exists but is not a directory. Cannot use cache.")
                 return None
        except Exception as e:
            # symlink loops, permission errors, etc.
            print(f"Warning: Error accessing potential cache path {cache_dir}: {e}.")

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

    # transformers 5.x rewrote PreTrainedModel.save_pretrained -- the
    # source-string anchors zoo's LoRA-merge optimization relies on are
    # gone. Detect that upfront and fall back to vanilla save_pretrained
    # so users on 5.x don't see a hard `Failed to find ...` RuntimeError
    # from the per-anchor checks below. The LoRA merge won't run, so
    # callers must `model.merge_and_unload()` (or equivalent) themselves
    # before saving on 5.x.
    _required_anchors = [
        "state_dict_split = split_torch_state_dict_into_shards",
        "state_dict[tensor].contiguous()",
        "def save_pretrained",
    ]
    if push_to_hub:
        _required_anchors.append("for shard_file, tensors in filename_to_tensors")
    _missing_anchors = [a for a in _required_anchors if a not in save_pretrained]
    if _missing_anchors:
        import transformers as _tx
        warnings.warn(
            "Unsloth: transformers "
            f"{getattr(_tx, '__version__', 'unknown')} rewrote "
            f"PreTrainedModel.save_pretrained -- the source-string "
            f"anchors {_missing_anchors!r} are missing, so the "
            "LoRA-merge-on-save optimization is skipped. Calling "
            "vanilla model.save_pretrained instead; merge LoRA "
            "explicitly (e.g. model.merge_and_unload()) before "
            "saving if you need the merged weights on disk.",
            stacklevel = 2,
        )
        model.save_pretrained(
            save_directory     = save_directory,
            push_to_hub        = push_to_hub,
            max_shard_size     = max_shard_size,
            safe_serialization = safe_serialization,
            token              = token,
            private            = private,
            revision           = revision,
        )
        if tokenizer is not None:
            tokenizer.save_pretrained(save_directory = save_directory)
        return

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
    """Get a model class's _checkpoint_conversion_mapping ({} if absent)."""
    try:
        module = __import__('transformers', fromlist=[model_class_name])
        model_class = getattr(module, model_class_name)
        return getattr(model_class, '_checkpoint_conversion_mapping', {})
    except (ImportError, AttributeError):
        return {}
pass


def detect_keys_format(keys_to_check, forward_mapping):
    if not forward_mapping:
        return "new"

    count_matches_old_pattern = 0
    count_matches_new_pattern = 0

    old_regex_compiled = [re.compile(p) for p in forward_mapping.keys()]
    # New patterns (mapping values) are matched as literal prefixes
    new_regex_compiled = [re.compile(r"^" + re.escape(val)) for val in forward_mapping.values()]

    for key in keys_to_check:
        if not isinstance(key, str): continue

        # "new" = starts with a mapping value; "old" = matches a mapping key
        # but not a new prefix (avoids double counting on overlap)
        matched_new = any(r.match(key) for r in new_regex_compiled)
        matched_old = any(r.match(key) for r in old_regex_compiled)

        if matched_new:
            count_matches_new_pattern += 1
        elif matched_old:
            count_matches_old_pattern += 1

    if count_matches_new_pattern > 0 and count_matches_old_pattern == 0: return "new"
    if count_matches_old_pattern > 0 and count_matches_new_pattern == 0: return "old"

    # Mixed: go with the majority
    if count_matches_new_pattern > count_matches_old_pattern: return "new"
    if count_matches_old_pattern > count_matches_new_pattern: return "old"

    return "new" # Default to current HF format
pass

def _safetensor_module_key(sf):
    """LoRA module path a .weight key backs: strip .weight, then a trailing .linear
    (Gemma4 ClippableLinear). None for non-.weight keys."""
    if not isinstance(sf, str) or not sf.endswith(".weight"):
        return None
    module_key = sf[: -len(".weight")]
    if module_key.endswith(".linear"):
        module_key = module_key[: -len(".linear")]
    return module_key
pass

def _build_valid_prefixes(keys_set, count_packed_mxfp4 = True):
    """Component-boundary parent prefixes of merge-backing tensors (.weight, or mxfp4
    _blocks paired with _scales). Lets _lora_key_has_backing test MoE descendants with an
    O(1) `cand in prefixes` lookup instead of an O(N) scan per key on large checkpoints."""
    valid_prefixes = set()
    for s in keys_set:
        if not isinstance(s, str):
            continue
        if s.endswith(".weight"):
            pass
        elif count_packed_mxfp4 and s.endswith("_blocks") and (s[: -len("_blocks")] + "_scales") in keys_set:
            pass
        else:
            continue
        parts = s.split(".")
        for i in range(1, len(parts)):
            valid_prefixes.add(".".join(parts[: i]))
    return valid_prefixes
pass

def _lora_key_has_backing(key, keys_set, count_packed_mxfp4 = True, valid_prefixes = None):
    """True if a converted LoRA module path is backed by a tensor the merge consumes:
    direct <key>.weight, Gemma4 <key>.linear.weight, mxfp4 packed <key>_blocks/_scales,
    per-expert MoE descendants, or fused 3D <prefix>.gate_up_proj/.down_proj (incl. the
    .moe -> .experts alias). Shared by the save-count check and the remap fallback so both
    match what the merge writes. count_packed_mxfp4=False mirrors the native mxfp4 save,
    which preserves _blocks/_scales (so a LoRA on a packed tensor is not written)."""
    if not isinstance(key, str):
        return False
    if (key + ".weight") in keys_set:
        return True
    if (key + ".linear.weight") in keys_set:                 # Gemma4 ClippableLinear
        return True
    if count_packed_mxfp4 and (key + "_blocks") in keys_set and (key + "_scales") in keys_set:
        return True                                          # mxfp4 packed (dequantized on save)
    if ".experts" in key or ".moe" in key:                   # fused / per-expert MoE
        base = key.replace(".base_layer", "")
        cands = set()
        # Disk aliases: .moe -> .experts (Gemma4); .mlp.experts -> .block_sparse_moe.experts (legacy Mixtral).
        for b in (
            base,
            base.replace(".moe", ".experts"),
            base.replace(".mlp.experts", ".block_sparse_moe.experts"),
        ):
            cands.add(b)
            # fused-named key (.gate_up_proj/.down_proj) is also backed by its per-expert
            # descendants, since the merge maps <experts>.<e>.<proj>.weight onto it.
            for suf in (".gate_up_proj", ".down_proj"):
                if b.endswith(suf):
                    cands.add(b[: -len(suf)])
        for cand in cands:
            if (cand + ".gate_up_proj") in keys_set or (cand + ".down_proj") in keys_set:
                return True                                  # fused 3D (GPT-OSS / Gemma4)
            if valid_prefixes is not None:
                # O(1): cand is a parent prefix of some per-expert .weight (or packed) tensor.
                if cand in valid_prefixes:
                    return True
            else:
                for s in keys_set:
                    if not (isinstance(s, str) and s.startswith(cand + ".")):
                        continue
                    if s.endswith(".weight"):                # per-expert 2D
                        return True
                    if count_packed_mxfp4 and s.endswith("_blocks") and (s[: -len("_blocks")] + "_scales") in keys_set:
                        return True                          # per-expert packed mxfp4
    return False
pass

def _infer_prefix_and_remap(lora_weights, safetensor_keys):
    """Infer missing key prefixes by matching LoRA keys against safetensor keys.

    Composite models may store safetensors under an extra prefix
    (``model.language_model.``) differing from the runtime ``model.`` namespace: keep
    already-matching keys, remap single-candidate keys, and let unmatched keys (e.g.
    fused MoE) inherit the most common inferred prefix. Also handles reordered path
    components (``model.language_model.`` <-> ``language_model.model.`` on Mistral 3
    VLMs) via a dominant prefix-substitution rule learned from common-suffix matches.
    Returns the remapped ``defaultdict``, or ``None`` if nothing was remapped.
    """
    if not safetensor_keys:
        return None

    sf_key_set = set(safetensor_keys)
    valid_prefixes = _build_valid_prefixes(sf_key_set)  # O(1) MoE backing lookups
    remapped = defaultdict(lora_weights.default_factory)
    changed = False
    inferred_prefixes = []  # track prefixes from successful per-key matches
    unmatched_keys = []     # keys that couldn't be matched at all

    for k, v in lora_weights.items():
        if not isinstance(k, str):
            remapped[k] = v
            continue
        # Already matches a safetensor key (direct, or Gemma4 ClippableLinear .linear.weight)
        if (k + ".weight") in sf_key_set or (k + ".linear.weight") in sf_key_set:
            remapped[k] = v
            continue
        # unique prefix candidates; also accept a .linear.weight shard (Gemma4) so a
        # prefix-add onto it is not dropped.
        candidates = list(dict.fromkeys(
            sf_key[: -len(suffix)]
            for suffix in (k + ".weight", k + ".linear.weight")
            for sf_key in safetensor_keys
            if sf_key.endswith(suffix) and sf_key[: -len(suffix)]
        ))
        if len(candidates) == 1:
            remapped[candidates[0] + k] = v
            inferred_prefixes.append(candidates[0])
            changed = True
        else:
            # No exact/prefix match -- defer to global substitution below
            unmatched_keys.append((k, v))

    # Discover a dominant prefix substitution from unmatched-key suffix matches; the
    # multiset guard below prevents false matches across sub-modules (vision vs language).
    if unmatched_keys:
        from collections import Counter as _Counter2
        substitution_votes = _Counter2()
        # A vote is recorded only for a true reordering of the same path components
        # (multiset guard below), so short trailing suffixes are safe to seed and cannot
        # pull a key across namespaces. Bucket sf keys by trailing 1..3 components so each
        # unmatched key scans only plausible buckets (~O(n)).
        sf_parts_by_suffix = defaultdict(list)
        for sf in safetensor_keys:
            module_key = _safetensor_module_key(sf)   # strips .weight and a trailing .linear
            if module_key is None:
                continue
            sf_parts = module_key.split(".")
            for sl in range(1, min(3, len(sf_parts)) + 1):
                sf_parts_by_suffix[(sl, tuple(sf_parts[-sl:]))].append(sf_parts)
        for k, _ in unmatched_keys:
            if not isinstance(k, str):
                continue
            k_parts = k.split(".")
            seen_sf = set()
            for sl in range(min(3, len(k_parts)), 0, -1):
                for sf_parts in sf_parts_by_suffix.get((sl, tuple(k_parts[-sl:])), ()):
                    sf_id = tuple(sf_parts)
                    if sf_id in seen_sf:        # process each sf once, at its longest match
                        continue
                    seen_sf.add(sf_id)
                    # Extend the common suffix as far as it goes.
                    common_suffix_len = sl
                    for i in range(sl + 1, min(len(k_parts), len(sf_parts)) + 1):
                        if k_parts[-i] == sf_parts[-i]:
                            common_suffix_len = i
                        else:
                            break
                    lora_prefix = ".".join(k_parts[: -common_suffix_len]) + "." if common_suffix_len < len(k_parts) else ""
                    sf_prefix = ".".join(sf_parts[: -common_suffix_len]) + "." if common_suffix_len < len(sf_parts) else ""
                    # Only a true reordering (same components, different order) seeds a vote.
                    # sorted() equality is the multiset guard (cheaper than a Counter per pair).
                    if (lora_prefix and sf_prefix and lora_prefix != sf_prefix
                            and sorted(p for p in lora_prefix.split(".") if p)
                                == sorted(p for p in sf_prefix.split(".") if p)):
                        substitution_votes[(lora_prefix, sf_prefix)] += 1

        # Apply substitutions for true reorderings only (e.g. model.language_model. <->
        # language_model.model.), never cross-namespace; claim only an on-disk target not
        # already taken, so an unmatched vision key can't overwrite a real language tensor.
        if substitution_votes:
            remaining_unmatched = list(unmatched_keys)
            applied_prefixes = set()
            for (lora_prefix, sf_prefix), _ in substitution_votes.most_common():
                if lora_prefix in applied_prefixes:
                    continue
                if sorted(p for p in lora_prefix.split(".") if p) != \
                   sorted(p for p in sf_prefix.split(".") if p):
                    continue
                applied_prefixes.add(lora_prefix)
                still_unmatched = []
                for k, v in remaining_unmatched:
                    new_key = sf_prefix + k[len(lora_prefix):] if k.startswith(lora_prefix) else None
                    if new_key is not None and new_key not in remapped and _lora_key_has_backing(new_key, sf_key_set, valid_prefixes = valid_prefixes):
                        remapped[new_key] = v
                        inferred_prefixes.append(sf_prefix)
                        changed = True
                    else:
                        still_unmatched.append((k, v))
                remaining_unmatched = still_unmatched
            unmatched_keys = remaining_unmatched

    from collections import Counter as _Counter
    common_prefix = _Counter(inferred_prefixes).most_common(1)[0][0] if inferred_prefixes else None

    # A LoRA target may carry an extra wrapper prefix the base lacks (model.vision_tower.*
    # vs vision_tower.* on Mistral 3). Strip only GENERIC wrappers (model/base_model/module),
    # never a semantic namespace (vision_tower/language_model) -- else an unbacked vision
    # adapter could strip to a bare language suffix and merge onto the wrong tensor. Requires
    # an exact on-disk backing for the stripped key.
    if unmatched_keys:
        _WRAPPER_COMPONENTS = {"model", "base_model", "module"}
        still_unmatched = []
        for k, v in unmatched_keys:
            target = None
            if isinstance(k, str):
                parts = k.split(".")
                for i in range(1, len(parts)):
                    if any(p not in _WRAPPER_COMPONENTS for p in parts[: i]):
                        break  # would strip a semantic namespace -> stop
                    cand = ".".join(parts[i:])
                    if cand not in remapped and _lora_key_has_backing(cand, sf_key_set, valid_prefixes = valid_prefixes):
                        target = cand
                        break
            if target is not None:
                remapped[target] = v
                changed = True
            else:
                still_unmatched.append((k, v))
        unmatched_keys = still_unmatched

    if not changed:
        return None

    # Apply the most common inferred prefix to remaining unmatched keys, but only when it
    # lands on a real backing tensor; otherwise leave the key so the merge skips a genuinely
    # unbacked target instead of rewriting it onto a wrong key. Backing covers MoE experts
    # and Gemma4 .linear, not just direct .weight.
    for k, v in unmatched_keys:
        if (
            common_prefix is not None and isinstance(k, str)
            and (common_prefix + k) not in remapped
            and _lora_key_has_backing(common_prefix + k, sf_key_set, valid_prefixes = valid_prefixes)
        ):
            remapped[common_prefix + k] = v
        else:
            remapped[k] = v

    return remapped


def _convert_lora_keys_to_safetensor_format(
    lora_weights,        # Global dict of LoraStats objects
    safetensor_keys,     # List of keys from the CURRENT shard
    model_class_name="PreTrainedModel" # The actual model instance (e.g. Qwen2VLForConditionalGeneration)
):
    import re

    forward_mapping = _get_checkpoint_conversion_mapping(model_class_name)

    if not forward_mapping:
        remapped = _infer_prefix_and_remap(lora_weights, safetensor_keys)
        if remapped is not None:
            return remapped
        return defaultdict(lora_weights.default_factory, lora_weights)

    reverse_mapping = {}
    for pattern, replacement in forward_mapping.items():
        reverse_mapping[replacement] = pattern
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
            # New LoRA keys, old shard -> convert LoRA key to old via reverse mapping
            for pattern, replacement in reverse_mapping.items():
                replacement = re.sub(r"\^?([^(?]+).*", r"\1", replacement.lstrip("^"))
                temp_key, n_replace = re.subn(pattern, replacement, converted_key_for_lookup)
                if n_replace > 0:
                    converted_key_for_lookup = temp_key
                    applied_conversion_for_this_key = True
                    break

        elif lora_key_format_assumed == "old" and shard_key_format == "new":
            # Old LoRA keys, new shard -> convert LoRA key to new
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

def _count_backed_lora_modules(lora_weights, safetensor_keys_seen, model_class_name, tie_word_embeddings, count_packed_mxfp4 = True):
    """Count LoRA modules backed by a saved tensor, mirroring the merge loop's key
    resolution (remap, Gemma4 .linear, fused MoE, mxfp4 packed, tied lm_head ->
    embed_tokens). The count equals what the merge writes, so the Step-7 sanity check
    needs no tied discount; genuinely unbacked targets (a vision tower absent from the
    base, or a bare lm_head whose composite-VLM embed sits at an unbridgeable prefix) are
    excluded, matching the merge.
    """
    converted = _convert_lora_keys_to_safetensor_format(
        lora_weights, safetensor_keys_seen, model_class_name = model_class_name,
    )
    # Pre-build parent prefixes once so the MoE backing check is O(1) per key, not O(N).
    valid_prefixes = _build_valid_prefixes(safetensor_keys_seen, count_packed_mxfp4 = count_packed_mxfp4)

    def _backed(key):
        if not isinstance(key, str):
            return False
        # all backing cases mirror the merge loop via the shared helper.
        if _lora_key_has_backing(key, safetensor_keys_seen, count_packed_mxfp4 = count_packed_mxfp4, valid_prefixes = valid_prefixes):
            return True
        if tie_word_embeddings and key.endswith("lm_head"):    # tied: merged onto embed_tokens
            # The merge folds an lm_head LoRA onto an on-disk embed_tokens.weight only when
            # that embed is not itself a target. Mirror that, keying off the DISK embed prefix
            # with the merge's model. strip/add, so a bare lm_head resolves to
            # model.embed_tokens.weight but a deep composite-VLM embed prefix does not.
            for sf in safetensor_keys_seen:
                if not (isinstance(sf, str) and sf.endswith("embed_tokens.weight")):
                    continue
                embed_key = sf[: -len(".weight")]
                if embed_key in converted:           # embed is a target -> lm_head dropped
                    continue
                lm_head_key = embed_key[: -len("embed_tokens")] + "lm_head"
                bridged = lm_head_key[len("model."):] if lm_head_key.startswith("model.") else "model." + lm_head_key
                if key == lm_head_key or key == bridged:
                    return True
        return False

    return sum(1 for key in converted if _backed(key))
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

def _is_fp8_quant_config(quant_config):
    """True for a dense 8-bit FP8 scheme (finegrained_fp8, fbgemm_fp8, compressed-tensors
    float-quantized) to dequantize on a 16bit merge. Excludes microscaling (mxfp8/mxfp4)
    and sub-8-bit floats (e.g. NVFP4)."""
    if not isinstance(quant_config, dict):
        return False
    method = str(quant_config.get("quant_method", "")).lower()
    # Dense FP8 method, but not microscaling (mx* keep their own block scales).
    if "fp8" in method and not method.startswith("mx"):
        return True
    # compressed-tensors: only 8-bit dense float groups. mxfp8/NVFP4 reuse type
    # "float"/num_bits 8, so exclude their format markers first.
    if method == "compressed-tensors":
        fmt = str(quant_config.get("format", "")).lower()
        if "mx" in fmt or "nvfp4" in fmt:
            return False
        if fmt == "float-quantized":
            return True
        for group in (quant_config.get("config_groups") or {}).values():
            weights = group.get("weights") if isinstance(group, dict) else None
            if not isinstance(weights, dict):
                continue
            wfmt = str(weights.get("format", "")).lower()
            if "mx" in wfmt or "nvfp4" in wfmt:
                continue
            try:
                num_bits = int(weights.get("num_bits", 8))
            except (TypeError, ValueError):
                num_bits = 0  # malformed (null / non-int) -> do not classify as FP8
            if str(weights.get("type", "")).lower() == "float" and num_bits == 8:
                return True
    return False
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

    # Detection keys off config.json["quantization_config"]. NVIDIA ModelOpt FP8 checkpoints
    # (e.g. *-Nemotron-*-FP8) instead carry their spec in a separate hf_quant_config.json
    # ("quantization": {"quant_algo": "FP8"}) with no config.json quantization_config, so they
    # are not detected here and a 16bit merge will not dequantize them. The dense FP8 dequant
    # math already handles their per-tensor layout; only this detection is missing. Tracked as
    # a follow-up (ModelOpt also emits NVFP4 / INT4_AWQ, which must NOT take the 8-bit path).
    if config and "quantization_config" in config:
        quant_config = config["quantization_config"]

        # Case 2: Check for MXFP4 format first (more specific)
        # We assume the Mxfp4Config serializes with a "quant_method": "mxfp4" key.
        if isinstance(quant_config, dict) and quant_config.get("quant_method") == "mxfp4":
            return (True, "mxfp4")

        # Case 3: FP8 (merged-16bit dequantizes instead of writing raw FP8).
        elif isinstance(quant_config, dict) and _is_fp8_quant_config(quant_config):
            return (True, "fp8")

        # Case 1: Fallback to existing logic for bitsandbytes
        elif isinstance(quant_config, dict):
            is_quantized = quant_config.get("load_in_4bit", False)
            quant_type = quant_config.get("bnb_4bit_quant_type", None)
            if is_quantized:
                # Return the specific type if available, otherwise a generic "bitsandbytes"
                return (True, quant_type if quant_type else "bitsandbytes")

    return (False, None)
pass

def _strip_fp8_suffix(model_name):
    """Strip a trailing FP8 quant marker (-FP8, -FP8-Dynamic/Static/Block/Row, -fp8, ...)
    and everything after it. Returns the 16bit base name, or None if there is no marker.
    Uses the last marker so an `fp8` inside a path/base name is not mistaken for it."""
    low = str(model_name).lower()
    idx = max(low.rfind("-fp8"), low.rfind("_fp8"))
    if idx <= 0:
        return None
    return model_name[:idx] or None
pass

def _resolve_fp8_16bit_sibling(model_name, token=None):
    """If model_name is an FP8 variant with an existing, non-quantized 16bit sibling
    (e.g. unsloth/GLM-5.2-FP8 -> unsloth/GLM-5.2), return the sibling so a 16bit merge
    folds LoRA onto full-precision weights instead of dequantizing the FP8. Else None."""
    base = _strip_fp8_suffix(model_name)
    if not base:
        return None
    try:
        local = check_local_model_exists(base)
        if local and not check_model_quantization_status(local)[0]:
            return local
        if check_hf_model_exists(base, token) and not check_model_quantization_status(base, token)[0]:
            return base
    except Exception:
        return None
    return None
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

def should_split_shards(is_t4, model_config, safetensors_list, max_size_in_bytes=0):
    """Determine if we need to split shards based on T4 and GPT-OSS conditions."""
    if not is_t4:
        return False

    if hasattr(model_config, 'model_type'):
        if model_config.model_type.lower() == 'gpt_oss':
            return True

    # Split single-file models larger than 5GB on T4 to avoid OOM during merge
    MAX_SINGLE_SHARD_BYTES = 5 * 1024 * 1024 * 1024  # 5GB
    if len(safetensors_list) == 1 and max_size_in_bytes > MAX_SINGLE_SHARD_BYTES:
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

def _stream_rewrite_resized_shard(src_path, dst_path, header_metadata, length_of_header, resized):
    # Stream one tensor at a time (peak RAM ~ one tensor): resized tensors from
    # `resized`, the rest byte-copied from src. Tensor-identical to dict+save_file.
    import struct
    src_data_start = 8 + length_of_header
    meta = header_metadata.get("__metadata__", None)
    tensor_keys = [k for k in header_metadata.keys() if k != "__metadata__"]
    tensor_keys.sort(key = lambda k: header_metadata[k]["data_offsets"][0])

    # Cast resized tensors to the header dtype so bytes match the label.
    res_t = {}
    for k in resized:
        dt = SAFETENSORS_DTYPES[header_metadata[k]["dtype"]]
        res_t[k] = resized[k].detach().to(dt).contiguous().cpu()

    new_header = {}
    if meta is not None:
        new_header["__metadata__"] = meta
    layout = []  # (key, src_off0, nbytes, is_resized)
    cursor = 0
    for k in tensor_keys:
        entry = header_metadata[k]
        if k in res_t:
            t = res_t[k]
            shape = list(t.shape)
            nbytes = t.numel() * t.element_size()
        else:
            shape = list(entry["shape"])
            o0, o1 = entry["data_offsets"]
            nbytes = o1 - o0
        # dtype is unchanged by a vocab resize
        new_header[k] = {"dtype": entry["dtype"], "shape": shape,
                         "data_offsets": [cursor, cursor + nbytes]}
        layout.append((k, entry["data_offsets"][0], nbytes, k in res_t))
        cursor += nbytes

    header_bytes = json.dumps(new_header, separators = (",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - (len(header_bytes) % 8)) % 8)  # 8-byte align data start

    CHUNK = 64 * 1024 * 1024
    with open(dst_path, "wb") as out, open(src_path, "rb") as src:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)
        for k, src_off0, nbytes, is_resized in layout:
            if is_resized:
                # reshape(-1) so 0-dim scalars (e.g. Gemma-4 audio min/max) view as bytes
                out.write(memoryview(res_t[k].reshape(-1).view(torch.uint8).numpy()))
            else:
                src.seek(src_data_start + src_off0)
                remaining = nbytes
                while remaining > 0:
                    chunk = src.read(min(CHUNK, remaining))
                    if not chunk:
                        raise RuntimeError(
                            f"Unsloth: unexpected EOF reading {src_path} for tensor {k}")
                    out.write(chunk)
                    remaining -= len(chunk)
    res_t.clear()
pass


def _estimate_resized_shard_bytes(header_metadata, resized, length_of_header):
    # Rewritten-shard size: resized tensors at new byte count, rest at current slot, + header.
    total = 0
    for k, entry in header_metadata.items():
        if k == "__metadata__":
            continue
        if k in resized:
            t = resized[k]
            total += t.numel() * t.element_size()
        else:
            o0, o1 = entry["data_offsets"]
            total += o1 - o0
    return total + 8 + length_of_header


def _inplace_rewrite_resized_shard(filename_original, header_metadata, resized):
    # Low-disk fallback: reload the shard, swap in resized tensors, save_file over
    # the original (higher RAM, no transient 2x-shard disk; non-atomic).
    meta = header_metadata.get("__metadata__", None)
    tensors = {}
    with safe_open(filename_original, framework = "pt", device = "cpu") as f:
        for key in f.keys():
            tensors[key] = resized[key] if key in resized else f.get_tensor(key)
    save_file(tensors, filename_original, metadata = meta)
    tensors.clear()


def _stream_rewrite_resized_shard_and_replace(filename_original, temp_dir, header_metadata, length_of_header, resized):
    # Stream the resized shard to a temp file then atomically os.replace it in
    # (peak RAM ~ one tensor). Used when disk has room for the transient copy.
    max_retries = 5
    base_delay  = 0.2  # seconds
    try:
        original_mode = os.stat(filename_original).st_mode
    except OSError:
        original_mode = None

    fd, tmp_path = tempfile.mkstemp(dir=temp_dir, suffix=".safetensors.tmp")
    os.close(fd)

    try:
        _stream_rewrite_resized_shard(
            src_path = filename_original,
            dst_path = tmp_path,
            header_metadata = header_metadata,
            length_of_header = length_of_header,
            resized = resized,
        )
        if original_mode is not None:
            try:
                os.chmod(tmp_path, original_mode)
            except OSError:
                pass

        gc.collect()
        device_empty_cache()

        for attempt in range(max_retries):
            try:
                os.replace(tmp_path, filename_original)
                tmp_path = None
                break
            except OSError as e:
                winerror  = getattr(e, "winerror", None)
                error_msg = str(e).lower()
                is_lock_error = (
                    winerror in {32, 1224}
                    or (
                        winerror == 5 and (
                            "user-mapped" in error_msg
                            or "being used by another process" in error_msg
                            or "sharing violation" in error_msg
                        )
                    )
                    or "user-mapped" in error_msg
                    or "being used by another process" in error_msg
                )
                if is_lock_error and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    if UNSLOTH_ENABLE_LOGGING:
                        logger.warning(
                            f"[Retry {attempt + 1}/{max_retries}] Windows file lock "
                            f"detected for {filename_original}: {e}. "
                            f"Waiting {delay:.1f}s before retry..."
                        )
                    gc.collect()
                    time.sleep(delay)
                    continue
                if is_lock_error:
                    raise RuntimeError(
                        f"Failed to rewrite {filename_original} after {max_retries} "
                        f"attempts due to Windows file lock. Original shard is intact "
                        f"(atomic replace never committed). "
                        f"Solutions: 1) Restart Unsloth Studio 2) Disable antivirus "
                        f"3) Close File Explorer windows"
                    ) from e
                raise RuntimeError(
                    f"Model merge failed while rewriting {filename_original}: {e}"
                ) from e
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
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

        # Zero-copy write into the mmap; avoids the bytes() copy that doubled peak
        # RAM on large tensors. tensor_formatted is already contiguous CPU; reshape(-1)
        # gives the 1D buffer the mmap slice needs (multi-dim assignment errors on some Pythons).
        tensor_view = tensor_formatted.detach().reshape(-1).view(torch.uint8)
        mm[index_L:index_R] = memoryview(tensor_view.numpy())

        # Clear memory
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

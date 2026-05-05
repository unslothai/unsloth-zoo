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
    "create_empty_model",
    "set_additional_modules",
    "finalize_huggingface_model",
    "patch_gemma4_vllm_lora_support",
    "patch_gemma4_vllm_k_eq_v_support",
    "extract_gdn_layers",
    "extract_vision_layers",
    "get_model_layer_config",
    "compare_attributes",
    "copy_attributes",
]

import torch
import re
import os
from copy import deepcopy
from .utils import get_quant_type
from .log import logger
from .hf_utils import HAS_TORCH_DTYPE, dtype_from_config, set_dtype_in_config

def _is_gemma4_config(config):
    if config is None:
        return False
    model_type = getattr(config, "model_type", None)
    text_config = getattr(config, "text_config", config)
    text_model_type = getattr(text_config, "model_type", None)
    return model_type == "gemma4" or text_model_type in ("gemma4", "gemma4_text")
pass

def is_comparable(val):
    # Don't treat tensors as comparable, only basic types
    from enum import Enum
    return isinstance(val, (int, float, bool, str, list, tuple, type(None), torch.dtype, Enum))

def compare_dicts(orig_dict, new_dict, prefix=""):
    all_keys = set(orig_dict.keys()) | set(new_dict.keys())
    for key in sorted(all_keys):
        orig_val = orig_dict.get(key, None)
        new_val = new_dict.get(key, None)
        key_path = f"{prefix}.{key}" if prefix else key
        if isinstance(orig_val, dict) and isinstance(new_val, dict):
            compare_dicts(orig_val, new_val, prefix=key_path)
        elif is_comparable(orig_val) and is_comparable(new_val):
            if orig_val != new_val:
                print(f"Dict key {key_path} mismatch: original {orig_val} != new model {new_val}")
        elif type(orig_val) != type(new_val):
            print(f"Dict key {key_path} type mismatch: original {type(orig_val)} != new model {type(new_val)}")

def compare_attributes(original_model, new_model):
    try:
        from transformers.configuration_utils import PreTrainedConfig
        PretrainedConfig = PreTrainedConfig
    except:
        from transformers.configuration_utils import PretrainedConfig

    print("=== ATTRIBUTE COMPARISON REPORT ===")
    missing_attrs = []
    type_mismatches = []
    value_mismatches = []

    # Extract all config keys at any level
    config_keys = _extract_all_config_keys(original_model.config) if hasattr(original_model, 'config') else set()
    config_keys = config_keys | {'config'}

    for (name, module), (orig_name, original_module) in zip(
        new_model.named_modules() if new_model is not None else [],
        original_model.named_modules() if original_model is not None else []
    ):
        orig_attrs = {attr for attr in dir(original_module) if not attr.startswith('_')}
        new_attrs = {attr for attr in dir(module) if not attr.startswith('_')}
        buffer_names = {name for name,_ in original_module.named_buffers(recurse=False)}


        # Find missing attributes (in original but not in new)
        missing_in_new = orig_attrs - new_attrs
        missing_in_new = missing_in_new - {'hf_device_map', 'source_cls'}
        if missing_in_new:
            for attr in sorted(missing_in_new):
                missing_attrs.append(f"{name}.{attr}")

        # Find extra attributes (in new but not in original)
        extra_in_new = new_attrs - orig_attrs
        if extra_in_new:
            print(f'Found some extra attributes like: {list(extra_in_new)[:5]}...')
            # for attr in sorted(extra_in_new):
            #     print(f"EXTRA ATTRIBUTE: {name}.{attr} (exists in new model but not original)")

        # Compare common attributes and buffer names
        common_attrs = orig_attrs & new_attrs
        common_buffers = orig_attrs | buffer_names
        for attr in sorted(common_attrs):
            if attr.startswith('.'):
                continue
            try:
                original_val = getattr(original_module, attr)
                new_val = getattr(module, attr)
            except Exception:
                continue

            original_comparable = is_comparable(original_val)
            new_comparable = is_comparable(new_val)

            # Check type mismatches first
            if type(original_val) != type(new_val):
                if original_comparable or new_comparable:
                    type_mismatches.append(f"{name}.{attr}: original {type(original_val).__name__} != new {type(new_val).__name__}")
                continue

            try:
                if isinstance(original_val, dict) and isinstance(new_val, dict):
                    if attr in config_keys:
                        # only compare those attributes that are relevant
                        compare_dicts(original_val, new_val, prefix=f"{name}.{attr}")
                elif original_comparable and new_comparable:
                    if original_val != new_val:
                        value_mismatches.append(f"{name}.{attr}: original {original_val} != new {new_val}")
            except Exception as e:
                type_mismatches.append(f"{name}.{attr}: comparison failed - {str(e)}")

            try:
                if isinstance(original_val, PretrainedConfig) and isinstance(new_val, PretrainedConfig):
                    compare_dicts(original_val.to_dict(), new_val.to_dict(), prefix=f"{name}.{attr}")
            except Exception as e:
                type_mismatches.append(f"{name}.{attr}: comparison failed - {str(e)}")

    # Print summary
    if missing_attrs:
        print(f"\n🚨 MISSING ATTRIBUTES ({len(missing_attrs)}):")
        for attr in missing_attrs:
            print(f"  - {attr}")

    if type_mismatches:
        print(f"\n⚠️  TYPE MISMATCHES ({len(type_mismatches)}):")
        for mismatch in type_mismatches:
            print(f"  - {mismatch}")

    if value_mismatches:
        print(f"\n📝 VALUE MISMATCHES ({len(value_mismatches)}):")
        for mismatch in value_mismatches:
            print(f"  - {mismatch}")

    if not missing_attrs and not type_mismatches and not value_mismatches:
        print("\n✅ No missing attributes or type mismatches found!")

def _extract_all_config_keys(config):
    """Extract all keys from config at any nesting level"""
    keys = set()

    def _extract_keys(obj, prefix=""):
        if hasattr(obj, 'to_dict'):
            obj = obj.to_dict()

        if isinstance(obj, dict):
            for key, value in obj.items():
                keys.add(key)
                if isinstance(value, dict):
                    _extract_keys(value, f"{prefix}.{key}" if prefix else key)
                elif hasattr(value, 'to_dict'):
                    _extract_keys(value, f"{prefix}.{key}" if prefix else key)

    _extract_keys(config)
    return keys

def copy_attributes(original_model, new_model):
    from transformers.configuration_utils import PretrainedConfig
    if original_model is None or new_model is None:
        print("Cannot copy attributes: one of the models is None")
        return

    # Extract all config keys at any level
    config_keys = _extract_all_config_keys(original_model.config) if hasattr(original_model, 'config') else set()
    config_keys = config_keys | {'config'}
    extra_attrs = {'hf_quantizer', }

    copied_count = 0
    skipped_count = 0
    skipped_attrs = []
    dict_copied_count = 0
    dict_skipped_count = 0

    for (name, module), (_, original_module) in zip(new_model.named_modules(), original_model.named_modules()):
        buffer_names = [name for name,_ in original_module.named_buffers(recurse=False)]
        for attr in dir(original_module):
            if attr.startswith('_'):
                continue

            try:
                original_val = getattr(original_module, attr)

                if attr in buffer_names:
                    # Some models like gemma3 have embed_scale and position_ids as buffers
                    # Lets copy them over to avoid inconsistencies
                    setattr(module, attr, original_val.to(new_model.device))
                elif is_comparable(original_val):
                    setattr(module, attr, original_val)
                    copied_count += 1
                elif isinstance(original_val, dict):
                    # Only copy dictionaries whose attribute name exists in config keys
                    if attr in config_keys:
                        setattr(module, attr, deepcopy(original_val))
                        copied_count += 1
                        dict_copied_count += 1
                    else:
                        skipped_count += 1
                        skipped_attrs.append(f"{attr} (dict not in config)")
                        dict_skipped_count += 1
                elif isinstance(original_val, PretrainedConfig):
                    # Sometimes the .config in original model is of config class and not a dict. Copy it as is.
                    setattr(module, attr, deepcopy(original_val))
                    copied_count += 1
                elif attr in extra_attrs:
                    setattr(module, attr, getattr(original_module, attr))
            except:
                skipped_count += 1
                skipped_attrs.append(attr)

    if os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1":
        print(f"✅ Copied {copied_count} attributes (including {dict_copied_count} config-related dicts)")
        if dict_skipped_count > 0:
            print(f"📋 Skipped {dict_skipped_count} non-config dictionaries")
        if skipped_count > 0:
            print(f"⏭️ Skipped {skipped_count} total attributes (tensors, modules, non-config dicts, etc.)")
            if skipped_count <= 10:
                print(f"    Skipped: {skipped_attrs}")
            else:
                print(f"    Sample: {skipped_attrs[:5]}... and {skipped_count-5} more")
pass


@torch.inference_mode
def create_empty_causal_lm(config, dtype = torch.float16):
    # All Unsloth Zoo code licensed under LGPLv3
    from transformers import AutoModelForCausalLM
    from accelerate import init_empty_weights
    # Suppress warning on uninited weights
    old_warn = os.environ.get("UNSLOTH_WARN_UNINITIALIZED", "1")
    os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"
    model_name = getattr(config, 'model_name', None)
    kwargs = {"torch_dtype" if HAS_TORCH_DTYPE else "dtype" : dtype_from_config(config)}
    original_meta_model = None
    error = None
    # [NOTE] init_empty_weights(include_buffers = True) is wrong
    # include_buffers=False is required because buffers (non-trainable tensors like
    # embed_scale, position_ids) must be initialized with actual values, not on meta
    # device. Models like Gemma 3 use embed_scale as a buffer in their embedding layer.
    # With include_buffers=True, buffers become empty meta tensors with no data,
    # causing attribute access failures during inference.
    with init_empty_weights(include_buffers = False):
        if model_name is not None:
            try:
                # This would persist quantization information for FP8 weights
                original_meta_model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            except Exception as e:
                error = str(e)
                original_meta_model = None
        if original_meta_model is None:
            try:
                # We must do this for 4.57.0 and above
                original_meta_model = AutoModelForCausalLM.from_config(config)
            except Exception as e:
                error = str(e)
                original_meta_model = None
    pass
    # Suppress warning on uninited weights
    os.environ["UNSLOTH_WARN_UNINITIALIZED"] = old_warn
    if error is not None and original_meta_model is None:
        print(f"Failed to create original_meta_model for AutoModelForCausalLM. Error {error}")
        original_meta_model = None

    new_config = deepcopy(config)
    new_config.intermediate_size = 1
    new_config.hidden_size = 1
    new_config.num_attention_heads = 1
    new_config.num_key_value_heads = 1
    new_config.head_dim = 1
    new_config.vocab_size = 1
    new_config.pad_token_id = 0

    _set_config_attrs(new_config, {
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 1,
        "linear_key_head_dim": 1,
        "linear_value_head_dim": 1,
        "linear_conv_kernel_dim": 1,
    })

    # Set attention module head_dim
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    new_config.update({"head_dim" : head_dim})

    new_model = AutoModelForCausalLM.from_config(
        new_config,
        attn_implementation = "eager",
    )

    return new_model, original_meta_model, config.num_hidden_layers

def _set_config_attrs(config_obj, attrs_to_set):
    """Helper to set multiple attributes on a config object if they exist."""
    for attr, value in attrs_to_set.items():
        if hasattr(config_obj, attr):
            setattr(config_obj, attr, value)
pass

def _get_model_device(model):
    for tensor in model.parameters():
        return tensor.device
    for tensor in model.buffers():
        return tensor.device
    return torch.device("cpu")
pass

def patch_gemma4_vllm_lora_support():
    from functools import wraps
    from vllm.model_executor.models import interfaces as vllm_model_interfaces
    from vllm.lora import model_manager as vllm_lora_model_manager
    try:
        from vllm.v1.worker import lora_model_runner_mixin
    except ImportError:
        lora_model_runner_mixin = None
    from unsloth_zoo import vllm_lora_worker_manager

    gemma4_lora_classes = []
    classes_to_patch = []
    try:
        from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration
        classes_to_patch.append(Gemma4ForConditionalGeneration)
        gemma4_lora_classes.append("Gemma4ForConditionalGeneration")
    except Exception:
        pass
    try:
        from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
        classes_to_patch.append(Gemma4ForCausalLM)
        gemma4_lora_classes.append("Gemma4ForCausalLM")
    except Exception:
        pass
    if not classes_to_patch:
        return
    gemma4_lora_classes = set(gemma4_lora_classes)

    for cls in classes_to_patch:
        if not getattr(cls, "_unsloth_gemma4_class_patched", False):
            cls.supports_lora = True
            if not hasattr(cls, "embedding_modules"):
                cls.embedding_modules = {}
            cls._unsloth_gemma4_class_patched = True

    original_supports_lora = getattr(
        lora_model_runner_mixin, "supports_lora", vllm_model_interfaces.supports_lora
    )
    if not hasattr(original_supports_lora, "_unsloth_gemma4_patch"):
        def patched_supports_lora(model):
            if model.__class__.__name__ in gemma4_lora_classes:
                return True
            return original_supports_lora(model)

        patched_supports_lora._unsloth_gemma4_patch = True
        if lora_model_runner_mixin is not None:
            lora_model_runner_mixin.supports_lora = patched_supports_lora
        vllm_model_interfaces.supports_lora = patched_supports_lora

    if not hasattr(vllm_lora_model_manager.create_lora_manager, "_unsloth_gemma4_patch"):
        original_create_lora_manager = vllm_lora_model_manager.create_lora_manager

        @wraps(original_create_lora_manager)
        def patched_create_lora_manager(model, *args, **kwargs):
            if model.__class__.__name__ in gemma4_lora_classes:
                kwargs.setdefault("lora_manager_cls", vllm_lora_model_manager.LoRAModelManager)
            return original_create_lora_manager(model, *args, **kwargs)

        patched_create_lora_manager._unsloth_gemma4_patch = True
        vllm_lora_model_manager.create_lora_manager = patched_create_lora_manager
        vllm_lora_worker_manager.create_lora_manager = patched_create_lora_manager
pass

# Prequantized BnB Gemma4 k_eq_v layers lack a synthetic v quant-state shard;
# we duplicate K -> V at loader-side quant-state stacking time.
def patch_gemma4_vllm_k_eq_v_support():
    from vllm.model_executor.model_loader.bitsandbytes_loader import (
        BitsAndBytesModelLoader,
    )

    stack_quantization_states = getattr(
        BitsAndBytesModelLoader, "_stack_quantization_states", None,
    )
    if stack_quantization_states is None:
        return
    if hasattr(stack_quantization_states, "_unsloth_gemma4_k_eq_v_patch"):
        return

    original_stack_quantization_states = stack_quantization_states

    def _get_gemma4_text_config(model):
        config = getattr(model, "config", None)
        if not _is_gemma4_config(config):
            return None
        return getattr(config, "text_config", config)

    def _get_gemma4_k_eq_v_pairs(model):
        text_config = _get_gemma4_text_config(model)
        if text_config is None or not getattr(text_config, "attention_k_eq_v", False):
            return ()

        param_names = set(name for name, _ in model.named_parameters())
        pairs = []
        for layer_idx, layer_type in enumerate(getattr(text_config, "layer_types", ())):
            if layer_type != "full_attention":
                continue

            for prefix in ("model.language_model", "language_model.model", "model"):
                k_name = f"{prefix}.layers.{layer_idx}.self_attn.k_proj.weight"
                v_name = f"{prefix}.layers.{layer_idx}.self_attn.v_proj.weight"
                qkv_name = f"{prefix}.layers.{layer_idx}.self_attn.qkv_proj.weight"
                if k_name in param_names:
                    pairs.append(("split", k_name, v_name))
                    break
                if qkv_name in param_names:
                    pairs.append(("packed", qkv_name, None))
                    break
        return tuple(pairs)

    def patched_stack_quantization_states(self, model, quant_state_dict):
        stacked_quant_state_dict = original_stack_quantization_states(
            self, model, quant_state_dict
        )

        for kind, source, target in _get_gemma4_k_eq_v_pairs(model):
            quant_states = stacked_quant_state_dict.get(source)
            if quant_states is None:
                continue

            # k_eq_v reuses K as V: the raw-weight loader already duplicates
            # k_proj -> v_proj, so prequant BnB needs the matching QuantState.
            if kind == "packed":
                if isinstance(quant_states, dict) and 2 not in quant_states and 1 in quant_states:
                    quant_states[2] = deepcopy(quant_states[1])
            elif kind == "split":
                if target not in stacked_quant_state_dict:
                    stacked_quant_state_dict[target] = deepcopy(quant_states)

        return stacked_quant_state_dict

    patched_stack_quantization_states._unsloth_gemma4_k_eq_v_patch = True
    BitsAndBytesModelLoader._stack_quantization_states = (
        patched_stack_quantization_states
    )
pass


@torch.inference_mode
def create_empty_vision_model(config, dtype = torch.float16):
    # All Unsloth Zoo code licensed under LGPLv3
    model_type = get_model_type(config)

    from transformers.models.siglip.modeling_siglip import SiglipVisionModel

    # Patch SiglipVisionModel to skip weight init on meta device.
    if not hasattr(SiglipVisionModel, "_original_initialize_weights"):
        SiglipVisionModel._original_initialize_weights = SiglipVisionModel._init_weights
        # Patch _init_weights to a no-op with correct signature
        def _init_weights(self, module):
            return
        SiglipVisionModel._init_weights = _init_weights

    import transformers
    model_cls = getattr(transformers, config.architectures[0])

    try:
        # Use accelerate's init_empty_weights, not transformers.modeling_utils
        from accelerate import init_empty_weights
        # [NOTE] init_empty_weights(include_buffers = True) is wrong
        # include_buffers=False is required because buffers (non-trainable tensors like
        # embed_scale, position_ids) must be initialized with actual values, not on meta
        # device. Models like Gemma 3 use embed_scale as a buffer in their embedding layer.
        # With include_buffers=True, buffers become empty meta tensors with no data,
        # causing attribute access failures during inference.
        with init_empty_weights():
            original_meta_model = model_cls(config)
    except Exception as e:
        print(f"Failed to create original_meta_model for {model_cls.__name__}. Error {e}")
        import traceback
        traceback.print_exc()
        original_meta_model = None

    # Restore original SiglipVisionModel weight init
    if hasattr(SiglipVisionModel, "_original_initialize_weights"):
        SiglipVisionModel._init_weights = SiglipVisionModel._original_initialize_weights
        del SiglipVisionModel._original_initialize_weights


    new_config = deepcopy(config)

    # Common text attributes
    _set_config_attrs(new_config.text_config, {
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "hidden_size": 1,
        "vocab_size": 8,
        "intermediate_size": 1,
        "head_dim": 1,
        "pad_token_id": 1,
    })
    # Qwen 3.5 or GDN related attrs
    _set_config_attrs(new_config.text_config, {
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 1,
        "linear_key_head_dim": 1,
        "linear_value_head_dim": 1,
        "linear_conv_kernel_dim": 1,
    })

    # Common vision attributes
    _set_config_attrs(new_config.vision_config, {
        "hidden_size": 1,
        "intermediate_size": 1,
        "patch_size": 1,
        "image_size": 1,
        "vision_output_dim": 1,
        # The following are different names for the same concept
        "num_heads": 1,
        "attention_heads": 1,
        "num_attention_heads": 1,
    })

    text_layers = config.text_config.num_hidden_layers
    vision_layers = getattr(config.vision_config, "num_hidden_layers", None) or getattr(config.vision_config, "depth", 0)

    if model_type in ("qwen2_5_vl", "qwen3_vl", "qwen3_5"):
        new_config.vision_config.out_hidden_size = 1

    num_layers = max(text_layers, vision_layers)
    new_model = model_cls(new_config)

    return new_model, original_meta_model, num_layers


@torch.inference_mode
def create_empty_model(config, dtype = torch.float16, is_vision_model = False):
    # All Unsloth Zoo code licensed under LGPLv3

    if is_vision_model:
        new_model, original_meta_model, num_layers = create_empty_vision_model(config, dtype)
    else:
        new_model, original_meta_model, num_layers = create_empty_causal_lm(config, dtype)

    # Get layer names from config
    layer_templates = get_model_layer_config(return_non_layered=False)
    layer_names = sum(layer_templates.values(), [])

    return new_model, original_meta_model, num_layers, layer_names


@torch.inference_mode
def set_additional_modules(new_model, quant_state_dict, config):
    def _unwrap_tensor(val):
        return getattr(val, "data", val)

    if hasattr(new_model, "language_model"):
        language_model = new_model.language_model
        language_model_prefix = "model.language_model"
    elif hasattr(new_model, "model") and hasattr(new_model.model, "language_model"):
        language_model = new_model.model.language_model
        language_model_prefix = "model.language_model"
    else:
        language_model_prefix = "model"
        language_model = new_model.model

    embed_tokens_key = f"{language_model_prefix}.embed_tokens.weight"
    # Explicit None check since pad_token_id=0 is valid.
    pad_token_id = getattr(config, "pad_token_id", None)
    if pad_token_id is None:
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            pad_token_id = getattr(text_config, "pad_token_id", None)
    if pad_token_id is not None: assert pad_token_id < quant_state_dict[embed_tokens_key].shape[0], f"Pad token id {pad_token_id} out of bounds for vocab size {quant_state_dict[embed_tokens_key].shape[0]}"

    # gemma3 uses Gemma3TextScaledWordEmbedding (nn.Embedding subclass with
    # an embed_scale); in-place weight assignment preserves its forward.
    def set_embedding(module, embed_tokens_key, pad_token_id, requires_grad=False):
        num_embeddings, embedding_dim = quant_state_dict[embed_tokens_key].shape
        embeddings = _unwrap_tensor(quant_state_dict[embed_tokens_key])
        if isinstance(embeddings, torch.Tensor):
            # Newer vLLM returns a plain tensor; wrap it so it can be assigned.
            embeddings = torch.nn.Parameter(embeddings, requires_grad = requires_grad)
        module.weight = embeddings
        module.padding_idx = pad_token_id
        module.num_embeddings = num_embeddings
        module.embedding_dim = embedding_dim

    set_embedding(language_model.embed_tokens, embed_tokens_key, pad_token_id) # This sets the embedding that we generally find in language (sub)model

    if 'model.visual.pos_embed.weight' in quant_state_dict:
        # This is to handle visual embeddings in Qwen 3 VL
        set_embedding(new_model.model.visual.pos_embed, 'model.visual.pos_embed.weight', None, requires_grad=False)

    norm_key = f"{language_model_prefix}.norm.weight"
    norm = quant_state_dict[norm_key]
    norm = _unwrap_tensor(norm)
    norm = torch.nn.Parameter(norm, requires_grad = False)
    language_model.norm.weight = norm

    # LM Head. Do note that for some models, like Mistral3ForConditionalGeneration,
    # there can be mismatch in the value of tie_word_embeddings between config and text_config
    # we prefer picking the one in text_config. If you notice any issue later, please report it!
    text_config = getattr(config, "text_config", config)
    if getattr(text_config, "tie_word_embeddings", False):
        lmhead_key = f"{language_model_prefix}.embed_tokens.weight"
    else:
        lmhead_key = "lm_head.weight"

    if lmhead_key in quant_state_dict:
        weight = _unwrap_tensor(quant_state_dict[lmhead_key])
        from torch.nn import Linear

        # Zero-dim Linear skips default weight allocation before we assign the real one.
        layer = Linear(0, 0, device=weight.device, bias=False)
        layer.in_features = weight.shape[1]
        layer.out_features = weight.shape[0]
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)

        if hasattr(new_model, "lm_head"):
            new_model.lm_head = layer
        elif hasattr(language_model, "lm_head"):
            language_model.lm_head = layer
        else:
            new_model.lm_head = layer

        if getattr(config, "tie_word_embeddings", False):
            if hasattr(new_model, "tie_weights"):
                new_model.tie_weights()
            elif hasattr(language_model, "tie_weights"):
                language_model.tie_weights()

    # Non-layered components (norms, embeddings, conv-style layers).
    non_layered_components = get_model_layer_config()["non_layered_components"]
    exact_non_layered = {n for n in non_layered_components if "{kk}" not in n}
    additional_keys = set(
        x for x in quant_state_dict.keys()
        if (
            any(x == n or x.startswith(n + ".") for n in exact_non_layered)
            or not any(substr in x for substr in ("layers", "blocks", embed_tokens_key, norm_key, "lm_head", "mlp", "linear", "list"))
        )
    )
    print(f'Performing substitution for {additional_keys=}')

    for key in additional_keys:
        # sometimes it can be in new_model.model. instead of new_model.
        for prefix in ['new_', 'new_model.']:
            try:
                val = quant_state_dict[key]
                val = _unwrap_tensor(val)
                if isinstance(val, torch.Tensor):
                    val = torch.nn.Parameter(val,requires_grad=False)
                exec(f"{prefix}{key} = val")
                break
            except:
                continue

    pass
pass

@torch.inference_mode
def finalize_huggingface_model(
    new_model,
    original_meta_model,
    config,
    dtype,
    quantization_config = None,
    bnb_config = None,
):
    if original_meta_model is not None:
        copy_attributes(original_meta_model, new_model)

    if hasattr(new_model, "language_model"):
        lm_root = new_model.language_model
    elif hasattr(new_model, "model") and hasattr(new_model.model, "language_model"):
        lm_root = new_model.model.language_model
    else:
        lm_root = getattr(new_model, "model", None)

    if lm_root is not None and hasattr(lm_root, "layers"):
        for layer_idx, layer in enumerate(lm_root.layers):
            if hasattr(layer, "layer_idx"):
                layer.layer_idx = layer_idx
            for attr_name in ("self_attn", "cross_attn", "mlp", "linear_attn"):
                submodule = getattr(layer, attr_name, None)
                if submodule is not None and hasattr(submodule, "layer_idx"):
                    submodule.layer_idx = layer_idx

    known_configs = {id(config)}
    for sub_name in ("text_config", "vision_config", "audio_config"):
        sub_cfg = getattr(config, sub_name, None)
        if sub_cfg is not None:
            known_configs.add(id(sub_cfg))

    live_root = getattr(new_model, "config", None)
    if live_root is not None and id(live_root) not in known_configs:
        set_dtype_in_config(live_root, dtype)
        known_configs.add(id(live_root))
        for sub_name in ("text_config", "vision_config", "audio_config"):
            sub_cfg = getattr(live_root, sub_name, None)
            if sub_cfg is not None and id(sub_cfg) not in known_configs:
                set_dtype_in_config(sub_cfg, dtype)
                known_configs.add(id(sub_cfg))

    for module in new_model.modules():
        module_config = getattr(module, "config", None)
        if module_config is not None and id(module_config) in known_configs:
            set_dtype_in_config(module_config, dtype)

    target_device = _get_model_device(new_model)
    text_config = getattr(config, "text_config", config)
    vision_config = getattr(config, "vision_config", None)

    vision_config_ids = set()
    if vision_config is not None:
        vision_config_ids.add(id(vision_config))
    live_vision_config = getattr(live_root, "vision_config", None) if live_root is not None else None
    if live_vision_config is not None:
        vision_config_ids.add(id(live_vision_config))

    local_rope_config = None
    for module_name, module in new_model.named_modules():
        if hasattr(module, "rotary_emb"):
            current_rotary_config = getattr(module.rotary_emb, "config", None)
            is_vision_rotary = vision_config is not None and (
                "vision_tower" in module_name
                or "vision_model" in module_name
                or (current_rotary_config is not None and id(current_rotary_config) in vision_config_ids)
            )
            rotary_config = vision_config if is_vision_rotary else text_config
            reinit_ok = True
            try:
                module.rotary_emb = module.rotary_emb.__class__(
                    config = rotary_config,
                    device = target_device,
                )
            except Exception as rotary_reinit_error:
                reinit_ok = False
                logger.warning(
                    f"Unsloth: skipped rotary_emb reinit for {module_name}: {rotary_reinit_error}"
                )
            if reinit_ok:
                for buffer_name, buffer in list(module.rotary_emb._buffers.items()):
                    if torch.is_tensor(buffer) and buffer.is_floating_point():
                        module.rotary_emb._buffers[buffer_name] = buffer.to(
                            device = target_device,
                            dtype = torch.float32,
                        )
        if hasattr(module, "rotary_pos_emb") and vision_config is not None:
            head_dim = vision_config.hidden_size // vision_config.num_heads
            module.rotary_pos_emb = module.rotary_pos_emb.__class__(head_dim//2).to(target_device)
        if hasattr(module, "rotary_emb_local"):
            if local_rope_config is None:
                local_rope_config = deepcopy(text_config)
                local_rope_config.rope_theta = text_config.rope_local_base_freq
                local_rope_config.rope_scaling = {"rope_type": "default"}
            module.rotary_emb_local = module.rotary_emb_local.__class__(
                config = local_rope_config,
                device = target_device,
            )

    if (quantization_config or {}) == {} and bnb_config is None:
        new_model = new_model.to(device = target_device, dtype = dtype)
        for module in new_model.modules():
            rotary_emb = getattr(module, "rotary_emb", None)
            if rotary_emb is None:
                continue
            for buffer_name, buffer in list(rotary_emb._buffers.items()):
                if torch.is_tensor(buffer) and buffer.is_floating_point():
                    rotary_emb._buffers[buffer_name] = buffer.to(
                        device = target_device,
                        dtype = torch.float32,
                    )
    return new_model
pass

def get_model_layer_config(return_non_layered=True):
    """
    Returns a unified layer configuration containing the union of layer names
    from all supported vision models. Serves as a fallback.

    Returns:
        dict: Dictionary containing layer templates for different components.
    """
    layer_templates = {
        'standard_layers': {
            "model.language_model.layers.{kk}.layer_scalar",
            "model.language_model.layers.{kk}.self_attn.q_proj",
            "model.language_model.layers.{kk}.self_attn.k_proj",
            "model.language_model.layers.{kk}.self_attn.v_proj",
            "model.language_model.layers.{kk}.self_attn.qkv_proj", # for extracting from vLLM (phi3 architecture)
            "model.language_model.layers.{kk}.self_attn.o_proj",
            "model.language_model.layers.{kk}.mlp.gate_proj",
            "model.language_model.layers.{kk}.mlp.up_proj",
            "model.language_model.layers.{kk}.mlp.gate_up_proj", # for extracting from vLLM (phi3 architecture)
            "model.language_model.layers.{kk}.mlp.down_proj",

            "model.layers.{kk}.layer_scalar",
            "model.layers.{kk}.self_attn.q_proj",
            "model.layers.{kk}.self_attn.k_proj",
            "model.layers.{kk}.self_attn.v_proj",
            "model.layers.{kk}.self_attn.qkv_proj", # for extracting from vLLM (phi3 architecture)
            "model.layers.{kk}.self_attn.o_proj",
            "model.layers.{kk}.mlp.gate_proj",
            "model.layers.{kk}.mlp.up_proj",
            "model.layers.{kk}.mlp.gate_up_proj", # for extracting from vLLM (phi3 architecture)
            "model.layers.{kk}.mlp.down_proj",
            "model.language_model.layers.{kk}.linear_attn.in_proj_qkv",
            "model.language_model.layers.{kk}.linear_attn.in_proj_z",
            "model.language_model.layers.{kk}.linear_attn.in_proj_b",
            "model.language_model.layers.{kk}.linear_attn.in_proj_a",
            "model.language_model.layers.{kk}.linear_attn.conv1d",
            "model.language_model.layers.{kk}.linear_attn.out_proj",
            "model.language_model.layers.{kk}.linear_attn.dt_bias",
            "model.language_model.layers.{kk}.linear_attn.A_log",

            "model.layers.{kk}.linear_attn.in_proj_qkv",
            "model.layers.{kk}.linear_attn.in_proj_z",
            "model.layers.{kk}.linear_attn.in_proj_b",
            "model.layers.{kk}.linear_attn.in_proj_a",
            "model.layers.{kk}.linear_attn.conv1d",
            "model.layers.{kk}.linear_attn.out_proj",
            "model.layers.{kk}.linear_attn.dt_bias",
            "model.layers.{kk}.linear_attn.A_log",

            # Gemma4 per-layer input modules
            "model.language_model.layers.{kk}.per_layer_input_gate",
            "model.language_model.layers.{kk}.per_layer_projection",
            "model.layers.{kk}.per_layer_input_gate",
            "model.layers.{kk}.per_layer_projection",
        },
        'layernorms': {
            "model.language_model.layers.{kk}.input_layernorm",
            "model.language_model.layers.{kk}.post_attention_layernorm",
            "model.language_model.layers.{kk}.pre_feedforward_layernorm",
            "model.language_model.layers.{kk}.post_feedforward_layernorm",
            "model.language_model.layers.{kk}.self_attn.q_norm",
            "model.language_model.layers.{kk}.self_attn.k_norm",
            "model.language_model.layers.{kk}.cross_attn.q_norm",
            "model.language_model.layers.{kk}.cross_attn.k_norm",
            "model.layers.{kk}.input_layernorm",
            "model.layers.{kk}.post_attention_layernorm",
            "model.layers.{kk}.pre_feedforward_layernorm",
            "model.layers.{kk}.post_feedforward_layernorm",
            "model.layers.{kk}.self_attn.q_norm",
            "model.layers.{kk}.self_attn.k_norm",
            "model.visual.blocks.{kk}.norm1",
            "model.visual.blocks.{kk}.norm2",
            "model.vision_tower.vision_model.encoder.layers.{kk}.post_layernorm",
            "model.vision_tower.vision_model.encoder.layers.{kk}.layer_norm1",
            "model.vision_tower.vision_model.encoder.layers.{kk}.layer_norm2",
            "model.vision_tower.encoder.layers.{kk}.input_layernorm",
            "model.vision_tower.encoder.layers.{kk}.post_attention_layernorm",
            "model.vision_tower.encoder.layers.{kk}.pre_feedforward_layernorm",
            "model.vision_tower.encoder.layers.{kk}.post_feedforward_layernorm",
            "model.vision_tower.encoder.layers.{kk}.self_attn.q_norm",
            "model.vision_tower.encoder.layers.{kk}.self_attn.k_norm",

            # Mistral3 vision norms
            "model.vision_tower.transformer.layers.{kk}.attention_norm",
            "model.vision_tower.transformer.layers.{kk}.ffn_norm",

            # qwen3 vl
            "model.visual.deepstack_merger_list.{kk}.norm",
            "model.language_model.layers.{kk}.linear_attn.norm",
            "model.layers.{kk}.linear_attn.norm",

            # Gemma4 per-layer input norm
            "model.language_model.layers.{kk}.post_per_layer_input_norm",
            "model.layers.{kk}.post_per_layer_input_norm",
        },
        'vision_layers': {

            # These will be used while converting from vLLM to HF
            "model.vision_model.transformer.layers.{kk}.self_attn.q_proj",
            "model.vision_model.transformer.layers.{kk}.self_attn.k_proj",
            "model.vision_model.transformer.layers.{kk}.self_attn.v_proj",
            "model.vision_model.transformer.layers.{kk}.self_attn.qkv_proj", # for extracting from vLLM
            "model.vision_model.transformer.layers.{kk}.self_attn.o_proj",
            'model.vision_model.global_transformer.layers.{kk}.gate_attn',
            "model.vision_model.transformer.layers.{kk}.input_layernorm",
            "model.vision_model.transformer.layers.{kk}.post_attention_layernorm",
            "model.vision_model.global_transformer.layers.{kk}.input_layernorm",
            "model.vision_model.global_transformer.layers.{kk}.post_attention_layernorm",

            "model.vision_model.transformer.layers.{kk}.mlp.fc1",
            "model.vision_model.transformer.layers.{kk}.mlp.fc2",

            "model.language_model.layers.{kk}.cross_attn.q_proj",
            "model.language_model.layers.{kk}.cross_attn.k_proj",
            "model.language_model.layers.{kk}.cross_attn.v_proj",
            "model.language_model.layers.{kk}.cross_attn.qkv_proj",
            "model.language_model.layers.{kk}.cross_attn.o_proj",
            "model.language_model.layers.{kk}.cross_attn_input_layernorm",
            "model.language_model.layers.{kk}.cross_attn_post_attention_layernorm",

            "model.vision_model.global_transformer.layers.{kk}.self_attn.q_proj",
            "model.vision_model.global_transformer.layers.{kk}.self_attn.k_proj",
            "model.vision_model.global_transformer.layers.{kk}.self_attn.v_proj",
            "model.vision_model.global_transformer.layers.{kk}.self_attn.qkv_proj",
            "model.vision_model.global_transformer.layers.{kk}.self_attn.o_proj",

            "model.vision_model.global_transformer.layers.{kk}.mlp.fc1",
            "model.vision_model.global_transformer.layers.{kk}.mlp.fc2",

            "model.vision_tower.vision_model.encoder.layers.{kk}.self_attn.q_proj",
            "model.vision_tower.vision_model.encoder.layers.{kk}.self_attn.k_proj",
            "model.vision_tower.vision_model.encoder.layers.{kk}.self_attn.v_proj",
            "model.vision_tower.vision_model.encoder.layers.{kk}.self_attn.qkv_proj",
            "model.vision_tower.vision_model.encoder.layers.{kk}.self_attn.out_proj",

            "model.vision_tower.vision_model.encoder.layers.{kk}.mlp.fc1",
            "model.vision_tower.vision_model.encoder.layers.{kk}.mlp.fc2",
            "model.vision_tower.encoder.layers.{kk}.self_attn.q_proj.linear",
            "model.vision_tower.encoder.layers.{kk}.self_attn.k_proj.linear",
            "model.vision_tower.encoder.layers.{kk}.self_attn.v_proj.linear",
            "model.vision_tower.encoder.layers.{kk}.self_attn.o_proj.linear",
            "model.vision_tower.encoder.layers.{kk}.mlp.gate_proj.linear",
            "model.vision_tower.encoder.layers.{kk}.mlp.up_proj.linear",
            "model.vision_tower.encoder.layers.{kk}.mlp.down_proj.linear",

            # qwen2.5_vl style
            "model.visual.blocks.{kk}.attn.qkv",
            "model.visual.blocks.{kk}.attn.proj",

            "model.visual.blocks.{kk}.mlp.gate_up_proj",
            "model.visual.blocks.{kk}.mlp.gate_proj",
            "model.visual.blocks.{kk}.mlp.up_proj",
            "model.visual.blocks.{kk}.mlp.down_proj",

            # Mistral 3
            "model.vision_tower.transformer.layers.{kk}.attention.q_proj",
            "model.vision_tower.transformer.layers.{kk}.attention.k_proj",
            "model.vision_tower.transformer.layers.{kk}.attention.v_proj",
            "model.vision_tower.transformer.layers.{kk}.attention.qkv_proj",
            "model.vision_tower.transformer.layers.{kk}.attention.o_proj",
            "model.vision_tower.transformer.layers.{kk}.feed_forward.gate_up_proj",
            "model.vision_tower.transformer.layers.{kk}.feed_forward.gate_proj",
            "model.vision_tower.transformer.layers.{kk}.feed_forward.up_proj",
            "model.vision_tower.transformer.layers.{kk}.feed_forward.down_proj",

            # qwen 3 vl
            "model.visual.blocks.{kk}.mlp.linear_fc1",
            "model.visual.blocks.{kk}.mlp.linear_fc2",

        },
        'additional_layers': {
            # Primarily for layers that are neither language decoder layers or vision transformer layers/blocks.
            # Basically anything that is a merger, convertor or bridge in between. Preferably iterable layers

            "model.visual.merger.mlp.{kk}",
            "model.visual.merger.mlp.{kk}",
            'model.language_model.model.layers.{kk}.cross_attn_mlp_gate',
            'model.language_model.model.layers.{kk}.cross_attn_attn_gate',
            'model.vision_model.global_transformer.layers.{kk}.gate_ffn',

            # Mistral3
            "model.multi_modal_projector.patch_merger.merging_layer",
            "model.multi_modal_projector.linear_{kk}",
            # "model.multi_modal_projector.linear_2",

            # qwen 3 vl
            "model.visual.deepstack_merger_list.{kk}.linear_fc1",
            "model.visual.deepstack_merger_list.{kk}.linear_fc2",

        },
        "non_layered_components":{
            # we do not handle quantization for these layers yet
            # the set_additional_modules would process these layers
            "model.visual.merger.linear_fc1",
            "model.visual.merger.linear_fc2",
            "model.multi_modal_projector",
            "model.language_model.norm",
            'model.vision_model.layernorm_pre',
            'model.vision_model.layernorm_post',
            'model.vision_model.class_embedding',
            "model.visual.norm",
            "model.visual.merger.ln_q",
            "model.visual.patch_embed.proj",
            "model.multi_modal_projector.mm_soft_emb_norm",
            "model.multi_modal_projector.mm_input_projection_weight",
            "model.vision_tower.vision_model.embeddings.patch_embedding",
            "model.vision_tower.vision_model.embeddings.position_embedding",
            "model.vision_tower.vision_model.post_layernorm",
            "model.multi_modal_projector.mm_input_projection_weight",
            "model.vision_model.post_tile_positional_embedding.gate",
            "model.vision_model.gated_positional_embedding.tile_embedding",
            "model.vision_model.pre_tile_positional_embedding.embedding",
            "model.vision_model.gated_positional_embedding",
            "model.vision_model.post_tile_positional_embedding.embedding",
            "model.vision_model.pre_tile_positional_embedding.gate",

            # Mistral3
            "model.vision_tower.patch_positional_embedding",
            "model.vision_tower.patch_conv",
            "model.vision_tower.ln_pre",
            "model.vision_tower.std_bias",
            "model.vision_tower.std_scale",
            "model.vision_tower.patch_embedder.position_embedding_table",
            "model.vision_tower.patch_embedder.input_proj",
            "model.embed_vision.embedding_projection",

            # qwen 3 vl
            "model.visual.pos_embed",
            "model.visual.merger.norm",

            # Gemma4 top-level per-layer-input modules
            "model.language_model.embed_tokens_per_layer",
            "model.language_model.per_layer_model_projection",
            "model.language_model.per_layer_projection_norm",
        }
    }

    # Convert sets to sorted lists for deterministic order
    return {key: sorted(list(value)) for key, value in layer_templates.items() if key!='non_layered_components' or return_non_layered}

def get_model_type(config):
    model_type = getattr(config, "model_type", "causal_lm")
    if hasattr(config, "vision_config"):
        # vllm curretly seems to be having qwen 2.5 vl model type as qwen2_5_vl_text for some reason
        # aka vllm_config.model_type is qwen2_5_vl_text but config.vision_config.model_type is qwen2_5_vl
        model_type = getattr(config.vision_config, "model_type", model_type)
    return model_type

def get_model_layer_counts(config):
    """
    Returns layer counts for different model types.

    Args:
        config: Model configuration

    Returns:
        int or dict: Number of layers (int for causal_lm, dict for VL models)
    """
    model_type = get_model_type(config)

    if model_type == "mllama":
        return {
            "text_layers": getattr(config.text_config, "num_hidden_layers", 32),
            "vision_layers": getattr(config.vision_config, "num_hidden_layers", 32),
            "global_layers": getattr(config.vision_config, "num_global_layers", 8),
        }
    elif model_type == "qwen2_5_vl":
        return {
            "text_layers": getattr(config, "num_hidden_layers", 32),
            "vision_layers": getattr(config.vision_config, "depth", 32),
        }
    elif model_type == "qwen3_vl":
        return {
            "text_layers": getattr(config, "num_hidden_layers", 36),
            "vision_layers": getattr(config.vision_config, "depth", 27),
            "deepstack_layers": getattr(config.vision_config, "deepstack_depth", 3),
        }
    elif model_type == "gemma4":
        return {
            "text_layers": getattr(config.text_config, "num_hidden_layers", 32),
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


def _get_nested_attr(obj, attr_path: str):
    parts = attr_path.split(".")
    if parts[0] == "model" and not hasattr(obj, "model"):
        parts = parts[1:]
    cur = obj
    try:
        for part in parts:
            if part.isdigit():
                cur = cur[int(part)]
            else:
                cur = getattr(cur, part)
        return cur
    except (AttributeError, IndexError):
        return None
    return None


def extract_gdn_layers(gdn_module, prefix, state_dict, quant_state_dict, get_state_dict):
    gdn = gdn_module

    def _unwrap(v):
        return getattr(v, "data", v)

    def store(name, value):
        state_dict[name] = value
        quant_state_dict[name] = value

    def _store_quant_state(name, quant_state):
        if quant_state is None:
            return
        quant_state_dict[f"{name}.weight.quant_state"] = quant_state
        try:
            for k, v in quant_state.as_dict(packed=True).items():
                state_dict[f"{name}.weight.{k}"] = v
        except Exception:
            pass

    if hasattr(gdn, "in_proj_qkvz"):
        proj = getattr(gdn.in_proj_qkvz, "base_layer", gdn.in_proj_qkvz)
        raw_weight = proj.weight
        weight = _unwrap(raw_weight)

        output_sizes = getattr(proj, "output_sizes", None)
        if output_sizes is None:
            key_dim = getattr(gdn, "key_dim", None)
            value_dim = getattr(gdn, "value_dim", None)
            if key_dim is None or value_dim is None:
                raise RuntimeError(
                    "Unsloth: cannot infer GDN in_proj_qkvz shards without "
                    "proj.output_sizes or gdn.key_dim / gdn.value_dim"
                )
            output_sizes = [key_dim, key_dim, value_dim, value_dim]
        output_sizes = list(output_sizes)
        offsets = [0]
        for s in output_sizes:
            offsets.append(offsets[-1] + s)
        if len(offsets) < 5:
            raise RuntimeError(
                f"Unsloth: GDN in_proj_qkvz expected 4 shards (q,k,v,z); got sizes={output_sizes}"
            )

        qkv_weight = weight[offsets[0]:offsets[3]]
        z_weight = weight[offsets[3]:offsets[4]]

        qs_attr = getattr(raw_weight, "bnb_quant_state", getattr(weight, "bnb_quant_state", None))
        qkv_states = [qs_attr.get(i) for i in (0, 1, 2)] if isinstance(qs_attr, dict) else [None, None, None]
        if sum(qs is not None for qs in qkv_states) > 1:
            try:
                from bitsandbytes.functional import dequantize_4bit
            except Exception:
                raise RuntimeError(
                    "Unsloth: prequantized BnB Qwen3.5 GDN requires bitsandbytes for fused in_proj_qkv reconstruction."
                )
            parts = []
            for i, qs in enumerate(qkv_states):
                shard = weight[offsets[i]:offsets[i + 1]]
                parts.append(dequantize_4bit(shard, quant_state=qs) if qs is not None else shard)
            store(f"{prefix}.in_proj_qkv.weight", torch.cat(parts, dim=0))
        else:
            store(f"{prefix}.in_proj_qkv.weight", qkv_weight)
            if isinstance(qs_attr, dict):
                _store_quant_state(f"{prefix}.in_proj_qkv", qkv_states[0])
        store(f"{prefix}.in_proj_z.weight", z_weight)
        if isinstance(qs_attr, dict):
            _store_quant_state(f"{prefix}.in_proj_z", qs_attr.get(3))

        if weight.dtype == torch.float8_e4m3fn:
            scale_attr = None
            if hasattr(proj, "weight_scale"):
                scale_attr = "weight_scale"
            elif hasattr(proj, "weight_scale_inv"):
                scale_attr = "weight_scale_inv"
            ws = _unwrap(getattr(proj, scale_attr)) if scale_attr is not None else None
            if ws is not None:
                if ws.ndim == 2 and ws.shape[1] > 1:
                    block_size = proj.weight_block_size[0]
                    scale_offsets = [x // block_size for x in offsets]
                    qkv_scale = ws[scale_offsets[0]:scale_offsets[3]]
                    z_scale = ws[scale_offsets[3]:scale_offsets[4]]
                else:
                    qkv_scale = ws[offsets[0]:offsets[3]]
                    z_scale = ws[offsets[3]:offsets[4]]
                store(f"{prefix}.in_proj_qkv.{scale_attr}", qkv_scale)
                store(f"{prefix}.in_proj_z.{scale_attr}", z_scale)
    else:
        get_state_dict(f"{prefix}.in_proj_qkv", 0, state_dict, gdn.in_proj_qkv, slice_weights=False)
        get_state_dict(f"{prefix}.in_proj_z", 0, state_dict, gdn.in_proj_z, slice_weights=False)

    ba_layer = getattr(gdn.in_proj_ba, "base_layer", gdn.in_proj_ba)
    raw_ba_weight = ba_layer.weight
    ba_weight = _unwrap(raw_ba_weight)
    mid = ba_weight.shape[0] // 2

    ba_qs = getattr(raw_ba_weight, "bnb_quant_state", getattr(ba_weight, "bnb_quant_state", None))
    ba_states = [ba_qs.get(i) for i in (0, 1)] if isinstance(ba_qs, dict) else [None, None]
    if isinstance(ba_qs, dict) and ba_states[0] is not None and ba_states[1] is None:
        try:
            from bitsandbytes.functional import dequantize_4bit
        except Exception:
            raise RuntimeError(
                "Unsloth: prequantized BnB Qwen3.5 GDN requires bitsandbytes for in_proj_ba split."
            )
        full = dequantize_4bit(ba_weight, quant_state=ba_states[0])
        full_mid = full.shape[0] // 2
        store(f"{prefix}.in_proj_b.weight", full[:full_mid])
        store(f"{prefix}.in_proj_a.weight", full[full_mid:])
    else:
        store(f"{prefix}.in_proj_b.weight", ba_weight[:mid])
        store(f"{prefix}.in_proj_a.weight", ba_weight[mid:])
        if isinstance(ba_qs, dict):
            _store_quant_state(f"{prefix}.in_proj_b", ba_states[0])
            _store_quant_state(f"{prefix}.in_proj_a", ba_states[1])

    if ba_weight.dtype == torch.float8_e4m3fn:
        scale_attr = None
        if hasattr(ba_layer, "weight_scale"):
            scale_attr = "weight_scale"
        elif hasattr(ba_layer, "weight_scale_inv"):
            scale_attr = "weight_scale_inv"
        ws = _unwrap(getattr(ba_layer, scale_attr)) if scale_attr is not None else None
        if ws is not None:
            if ws.ndim == 2 and ws.shape[1] > 1:
                block_size = ba_layer.weight_block_size[0]
                scale_mid = mid // block_size
                b_scale = ws[:scale_mid]
                a_scale = ws[scale_mid:]
            else:
                b_scale = ws[:mid]
                a_scale = ws[mid:]
            store(f"{prefix}.in_proj_b.{scale_attr}", b_scale)
            store(f"{prefix}.in_proj_a.{scale_attr}", a_scale)

    store(f"{prefix}.conv1d.weight", gdn.conv1d.weight.data)
    store(f"{prefix}.dt_bias", gdn.dt_bias.data)
    store(f"{prefix}.A_log", gdn.A_log.data)

    if hasattr(gdn, "norm") and hasattr(gdn.norm, "weight"):
        store(f"{prefix}.norm.weight", gdn.norm.weight.data)

    get_state_dict(f"{prefix}.out_proj", 0, state_dict, gdn.out_proj)
pass


def extract_vision_layers(vllm_internals, state_dict, quant_state_dict, get_state_dict):
    """
    Extracts vision layers for any supported vision model by dynamically using
    a model-specific configuration. This approach is more robust and avoids
    failures by correctly identifying layer paths and parameters.
    """
    model_type = get_model_type(vllm_internals.config)
    layer_config = get_model_layer_config()

    all_layered_templates = (
        layer_config.get('vision_layers', []) +
        layer_config.get('layernorms', []) +
        layer_config.get('additional_layers', [])
    )

    layer_counts = get_model_layer_counts(vllm_internals.config)
    num_layers_to_iterate = max(layer_counts.values()) if isinstance(layer_counts, dict) else layer_counts

    # Process layered components
    for kk in range(num_layers_to_iterate):
        for layer_template in all_layered_templates:
            layer_path = layer_template.format(kk=kk)
            layer_module = _get_nested_attr(vllm_internals, layer_path)

            if 'language_model.model' in layer_path:
                # vLLM uses vllm_internals.language_model.model.layers while HF uses model.language_model.layers
                layer_path = layer_path.replace('language_model.model', 'language_model')


            if layer_module is not None:
                if "qkv" in layer_path:
                    if model_type in ("qwen2_5_vl", "qwen3_vl", "qwen3_5"):
                        # If the HF model too prefers having merged qkv, we do this
                        # This is evident in qwen-2.5-vl and qwen-3-vl so far.
                        get_state_dict(layer_path, 0, state_dict, layer_module, slice_weights=False)
                    else:
                         get_state_dict(f"{layer_path.replace('qkv_proj', 'q_proj')}", 0, state_dict, layer_module)
                         get_state_dict(f"{layer_path.replace('qkv_proj', 'k_proj')}", 1, state_dict, layer_module)
                         get_state_dict(f"{layer_path.replace('qkv_proj', 'v_proj')}", 2, state_dict, layer_module)
                elif "gate_up_proj" in layer_path:
                    # vLLM seems to have merged gate and up proj recently for qwen vl. This is to handle new variant
                    # https://github.com/jeejeelee/vllm/commit/a71e4765cc0c1534f2a8891aaf628e1751f6df07
                    get_state_dict(f"{layer_path.replace('gate_up_proj','gate_proj')}", 0, state_dict, layer_module)
                    get_state_dict(f"{layer_path.replace('gate_up_proj','up_proj')}", 1, state_dict, layer_module)
                elif "fc" in layer_path or "proj" in layer_path:
                    get_state_dict(layer_path, 0, state_dict, layer_module)
                else: # Handle other layers, especially layernorms
                    if isinstance(layer_module, torch.nn.Module):
                        if hasattr(layer_module, 'weight'):
                            get_state_dict(layer_path, 0, state_dict, layer_module)
                    elif isinstance(layer_module, torch.Tensor):
                        state_dict[f"{layer_path}"] = layer_module.data
                        quant_state_dict[f"{layer_path}"] = state_dict[f"{layer_path}"]
                    else:
                        print(f"Unsloth: Skipping layer '{layer_path}' of unexpected type: {type(layer_module)}")

    # Extract non-layered vision components using a more robust method
    non_layered_components = layer_config.get('non_layered_components', [])
    for component_path in non_layered_components:
        component = _get_nested_attr(vllm_internals, component_path)

        if component is not None:
            if hasattr(component, 'weight'):
                # Prefer using get_state_dict when possible
                get_state_dict(component_path, 0, state_dict, component)
            elif isinstance(component, torch.Tensor):
                state_dict[component_path] = component.data
                quant_state_dict[component_path] = component.data
            elif isinstance(component, torch.nn.Module):
                for param_name, param in component.named_parameters():
                    # if the parameter is to be extracted separately, skip it
                    if param_name.replace('.weight', '') in non_layered_components: continue
                    full_param_path = f"{component_path}.{param_name}"
                    if hasattr(param, 'weight'):
                        get_state_dict(full_param_path, 0, state_dict, param)
                    elif hasattr(param, 'data'):
                        state_dict[full_param_path] = param.data
                        quant_state_dict[full_param_path] = param.data
            else:
                print(f"Unsloth: Skipping non-layered component '{component_path}' of unexpected type: {type(component)}")

    # for mllama. vLLM uses ColumnParallelConv2dPatch which has _linear.weight of shape torch.Size([1280, 588])
    # hf expects patch_embedding.weight of shape torch.Size([1280, 3, 14, 14])
    path = "model.vision_model.patch_embedding"
    component = _get_nested_attr(vllm_internals, path)
    if component is not None:
        weight = component._linear.weight
        state_dict[f'{path}.weight'] = weight.reshape(weight.shape[0], 3, 14, 14)
        quant_state_dict[f'{path}.weight'] = state_dict[f'{path}.weight']

    # for qwen3 vl, only needed in specific vllm which had this PR which uses Linear instead of Conv3d
    # https://github.com/vllm-project/vllm/pull/27418
    path = "model.visual.patch_embed.proj"
    vision_config = vllm_internals.config.vision_config
    component = _get_nested_attr(vllm_internals, path)
    if component is not None:
        weight = component.weight
        state_dict[f'{path}.weight'] = weight.reshape(vision_config.hidden_size, vision_config.in_channels, vision_config.temporal_patch_size, vision_config.patch_size, vision_config.patch_size)
        quant_state_dict[f'{path}.weight'] = state_dict[f'{path}.weight']

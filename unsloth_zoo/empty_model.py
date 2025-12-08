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
from .hf_utils import HAS_TORCH_DTYPE, dtype_from_config

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

    # Set minimal sizes for different model types
    if model_type == "qwen2_5_vl":
        new_config.vision_config.out_hidden_size = 1
    elif model_type == "qwen3_vl":
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
    if hasattr(new_model, "language_model"):
        language_model = new_model.language_model
        language_model_prefix = "model.language_model"
    else:
        language_model_prefix = "model"
        language_model = new_model.model

    # Embeddings
    embed_tokens_key = f"{language_model_prefix}.embed_tokens.weight"
    pad_token_id = getattr(config, "pad_token_id", None) or getattr(config, "text_config", None) and getattr(config.text_config, "pad_token_id", None)
    if pad_token_id: assert pad_token_id <= quant_state_dict[embed_tokens_key].shape[0], f"Pad token id {pad_token_id} out of bounds for vocab size {quant_state_dict[embed_tokens_key].shape[0]}"

    # language_model.embed_tokens = torch.nn.Embedding.from_pretrained(
    #     quant_state_dict[embed_tokens_key],
    #     freeze = True,
    #     padding_idx = pad_token_id,
    # )
    # we cannot use the normal embedding init because gemma3 uses Gemma3TextScaledWordEmbedding which wraps around nn.Embedding and has a scaling factor. This new init ensures that we respect the forward from original model.
    def set_embedding(module, embed_tokens_key, pad_token_id, requires_grad=False):
        num_embeddings, embedding_dim = quant_state_dict[embed_tokens_key].shape
        embeddings = quant_state_dict[embed_tokens_key]
        if isinstance(embeddings, torch.Tensor):
            # in the newer vLLM versions, this seems to return a tensor which can't be assigned to embedding weight
            # we need to convert that to nn.Paramter and then pass it on
            embeddings = torch.nn.Parameter(embeddings, requires_grad = requires_grad)
        module.weight = embeddings
        module.padding_idx = pad_token_id
        module.num_embeddings = num_embeddings
        module.embedding_dim = embedding_dim

    set_embedding(language_model.embed_tokens, embed_tokens_key, pad_token_id) # This sets the embedding that we generally find in language (sub)model

    if 'model.visual.pos_embed.weight' in quant_state_dict:
        # This is to handle visual embeddings in Qwen 3 VL
        set_embedding(new_model.model.visual.pos_embed, 'model.visual.pos_embed.weight', None, requires_grad=False)

    # Norm
    norm_key = f"{language_model_prefix}.norm.weight"
    norm = quant_state_dict[norm_key]
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

    # Check if lm_head exists in the state dict
    if lmhead_key in quant_state_dict:
        weight = quant_state_dict[lmhead_key]
        from torch.nn import Linear

        # Create Linear layer with zero dimensions to avoid any weight allocation
        layer = Linear(0, 0, device=weight.device, bias=False)
        # Set correct dimensions
        layer.in_features = weight.shape[1]
        layer.out_features = weight.shape[0]
        # Assign the weight directly (no deletion needed since no weight was allocated)
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)

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
    # For any layers that are potentially in non layered components.
    # Preferably norms, embeddings and convolution type layers.
    additional_keys = set(
        x for x in quant_state_dict.keys()
        if not any(substr in x for substr in ("layers", "blocks", embed_tokens_key, norm_key, "lm_head", "mlp", "linear", "list"))
    )
    print(f'Performing substitution for {additional_keys=}')

    for key in additional_keys:
        # sometimes it can be in new_model.model. instead of new_model.
        for prefix in ['new_', 'new_model.']:
            try:
                val = quant_state_dict[key]
                if isinstance(val, torch.Tensor):
                    val = torch.nn.Parameter(val,requires_grad=False)
                exec(f"{prefix}{key} = val")
                break
            except:
                continue

    pass
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
            "model.language_model.layers.{kk}.self_attn.q_proj",
            "model.language_model.layers.{kk}.self_attn.k_proj",
            "model.language_model.layers.{kk}.self_attn.v_proj",
            "model.language_model.layers.{kk}.self_attn.o_proj",
            "model.language_model.layers.{kk}.mlp.gate_proj",
            "model.language_model.layers.{kk}.mlp.up_proj",
            "model.language_model.layers.{kk}.mlp.down_proj",

            "model.layers.{kk}.self_attn.q_proj",
            "model.layers.{kk}.self_attn.k_proj",
            "model.layers.{kk}.self_attn.v_proj",
            "model.layers.{kk}.self_attn.o_proj",
            "model.layers.{kk}.mlp.gate_proj",
            "model.layers.{kk}.mlp.up_proj",
            "model.layers.{kk}.mlp.down_proj",
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

            # Mistral3 vision norms
            "model.vision_tower.transformer.layers.{kk}.attention_norm",
            "model.vision_tower.transformer.layers.{kk}.ffn_norm",

            # qwen3 vl
            "model.visual.deepstack_merger_list.{kk}.norm",
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
            "model.visual.merger.linear_fc{kk}",

        },
        "non_layered_components":{
            # we do not handle quantization for these layers yet
            # the set_additional_modules would process these layers
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

            # qwen 3 vl
            "model.visual.pos_embed",
            "model.visual.merger.norm",
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
                    if model_type in ("qwen2_5_vl", "qwen3_vl"):
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
                    elif isinstance(layer_module, torch.nn.Parameter):
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
            elif isinstance(component, torch.nn.Parameter):
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

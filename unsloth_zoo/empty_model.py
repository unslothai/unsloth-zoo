__all__ = [
    "create_empty_model",
    "set_additional_modules",
    "extract_mllama_vision_layers",
    "extract_qwen2_5_vl_vision_layers",
    "extract_gemma3_vision_layers",
    "get_model_layer_config",
    "compare_attributes",
    "copy_attributes",
]

import torch
import re
from collections import OrderedDict
from copy import deepcopy

def is_comparable(val):
    # Don't treat tensors as comparable, only basic types
    return isinstance(val, (int, float, bool, str, list, type(None)))

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
    from transformers.configuration_utils import PretrainedConfig
    print("=== ATTRIBUTE COMPARISON REPORT ===")
    missing_attrs = []
    type_mismatches = []
    value_mismatches = []

    for (name, module), (orig_name, original_module) in zip(
        new_model.named_modules() if new_model is not None else [],
        original_model.named_modules() if original_model is not None else []
    ):
        orig_attrs = {attr for attr in dir(original_module) if not attr.startswith('_')}
        new_attrs = {attr for attr in dir(module) if not attr.startswith('_')}

        # Find missing attributes (in original but not in new)
        missing_in_new = orig_attrs - new_attrs
        missing_in_new = missing_in_new - {'hf_device_map'}
        if missing_in_new:
            for attr in sorted(missing_in_new):
                missing_attrs.append(f"{name}.{attr}")

        # Find extra attributes (in new but not in original)
        extra_in_new = new_attrs - orig_attrs
        if extra_in_new:
            for attr in sorted(extra_in_new):
                print(f"EXTRA ATTRIBUTE: {name}.{attr} (exists in new model but not original)")

        # Compare common attributes
        common_attrs = orig_attrs & new_attrs
        for attr in sorted(common_attrs):
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

    copied_count = 0
    skipped_count = 0
    skipped_attrs = []
    dict_copied_count = 0
    dict_skipped_count = 0

    for (name, module), (_, original_module) in zip(new_model.named_modules(), original_model.named_modules()):
        for attr in dir(original_module):
            if attr.startswith('_'):
                continue

            try:
                original_val = getattr(original_module, attr)

                if original_model.config.model_type == 'gemma3' and attr == 'embed_scale':
                    # Gemma3 has this value as tensor. We generally skip copying tensors.
                    # We might want to force copy this attribute
                    setattr(module, attr, original_val)

                if is_comparable(original_val):
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
            except:
                skipped_count += 1
                skipped_attrs.append(attr)

    print(f"✅ Copied {copied_count} attributes (including {dict_copied_count} config-related dicts)")
    if dict_skipped_count > 0:
        print(f"📋 Skipped {dict_skipped_count} non-config dictionaries")
    if skipped_count > 0:
        print(f"⏭️  Skipped {skipped_count} total attributes (tensors, modules, non-config dicts, etc.)")
        if skipped_count <= 10:
            print(f"   Skipped: {skipped_attrs}")
        else:
            print(f"   Sample: {skipped_attrs[:5]}... and {skipped_count-5} more")


@torch.inference_mode()
def create_empty_causal_lm(config, dtype = torch.float16):
    # All Unsloth Zoo code licensed under LGPLv3
    from transformers import AutoModelForCausalLM
    try:
        from accelerate import init_empty_weights
        with init_empty_weights():
            original_meta_model = AutoModelForCausalLM.from_config(config)
    except Exception as e:
        print(f"Failed to create original_meta_model for AutoModelForCausalLM. Error {e}")
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

    # Get layer names from config
    layer_config = get_model_layer_config("causal_lm", config)
    layer_names = layer_config['standard_layers'] + layer_config['layernorms']

    return new_model, original_meta_model, layer_names, config.num_hidden_layers

def _set_config_attrs(config_obj, attrs_to_set):
    """Helper to set multiple attributes on a config object if they exist."""
    for attr, value in attrs_to_set.items():
        if hasattr(config_obj, attr):
            setattr(config_obj, attr, value)
pass


@torch.inference_mode()
def create_empty_vision_model(config, dtype = torch.float16):
    # All Unsloth Zoo code licensed under LGPLv3
    model_type = config.model_type

    from transformers.models.siglip.modeling_siglip import SiglipVisionModel

    # Patch SiglipVisionModel to skip weight init on meta device.
    if not hasattr(SiglipVisionModel, "_original_initialize_weights"):
        SiglipVisionModel._original_initialize_weights = SiglipVisionModel._init_weights
        # Patch _init_weights to a no-op with correct signature
        def _init_weights(self, module):
            return
        SiglipVisionModel._init_weights = _init_weights

    if model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model_cls = Qwen2_5_VLForConditionalGeneration
    elif model_type == "mllama":
        from transformers import MllamaForConditionalGeneration
        model_cls = MllamaForConditionalGeneration
    elif model_type == "gemma3":
        from transformers import Gemma3ForConditionalGeneration
        model_cls = Gemma3ForConditionalGeneration
    else:
        raise ValueError(f"Unsloth: Unsupported vision model type: {model_type}")

    try:
        # Use accelerate's init_empty_weights, not transformers.modeling_utils
        from accelerate import init_empty_weights
        with init_empty_weights():
            original_meta_model = model_cls(config)
            print(f'Initialised dummy model for config')
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


    num_layers = max(text_layers, vision_layers)
    new_model = model_cls(new_config)

    # Get layer names from config
    layer_config = get_model_layer_config(model_type, config)
    layer_names = (layer_config['standard_layers'] +
                  layer_config['layernorms'] +
                  layer_config['vision_layers'] +
                  layer_config['additional_layers'])

    return new_model, original_meta_model, layer_names, num_layers


@torch.inference_mode()
def create_empty_model(config, dtype = torch.float16, is_vision_model = False):
    # All Unsloth Zoo code licensed under LGPLv3
    if is_vision_model:
        return create_empty_vision_model(config, dtype)
    else:
        return create_empty_causal_lm(config, dtype)

@torch.inference_mode()
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
        padding_idx = config.pad_token_id,
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

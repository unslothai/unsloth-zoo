# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Lightweight FastLanguageModel for Apple Silicon / MLX.

No GPU dependencies — uses mlx-lm for model loading and LoRA.
Supports both text-only models (mlx-lm) and VLMs (mlx-vlm).
This avoids importing unsloth.models (which pulls in CUDA kernels).
"""

import json
import importlib
import inspect
import os
import types
import warnings
from dataclasses import asdict, dataclass
from fnmatch import fnmatch

from .mlx_compile import (
    explain_compile_support,
    get_compile_qualification,
    get_compile_trait_report,
    get_backend_compile_qualifications,
    install_mlx_compile_patches,
    normalize_mlx_patch_mode,
    trace_compile_application,
)

_vlm_model_types_cache = None
_SAFE_TEXT_SANITIZE_PATCHED: set[str] = set()
_MULTIMODAL_STRIP_KEYS = (
    "vision_tower",
    "audio_tower",
    "embed_audio",
    "embed_vision",
    "multi_modal_projector",
    "mm_projector",
)


def _convert_mlx_dtype(model, target_dtype) -> None:
    """Cast floating-point params to target_dtype (preserves quantized ints)
    while honoring the model's optional path-based ``cast_predicate``."""
    import mlx.core as mx
    from mlx.utils import tree_map_with_path
    cast_pred = getattr(model, "cast_predicate", lambda _: True)
    model.update(tree_map_with_path(
        lambda k, v: v.astype(target_dtype)
        if cast_pred(k) and mx.issubdtype(v.dtype, mx.floating) else v,
        model.parameters(),
    ))
    mx.eval(model.parameters())


def _is_vlm(config: dict) -> bool:
    """Detect whether a model config describes a VLM.

    Checks:
    1. "vision_config" key in config → True
    2. model_type is in mlx_vlm's supported model set (discovered dynamically)
    """
    if "vision_config" in config:
        return True

    architectures = config.get("architectures") or ()
    if isinstance(architectures, str):
        architectures = (architectures,)
    if any(str(arch).endswith("ForCausalLM") for arch in architectures):
        return False

    model_type = config.get("model_type", "")
    if not model_type:
        return False

    global _vlm_model_types_cache
    if _vlm_model_types_cache is None:
        _vlm_model_types_cache = _build_vlm_model_types()

    return model_type in _vlm_model_types_cache


_KNOWN_MLX_LM_STRICT_FALLBACKS = {
    "gemma4_text": {
        "message_tokens": (
            "parameters not in model",
            "self_attn.k_proj",
            "self_attn.v_proj",
        ),
        "notice": (
            "Unsloth: Gemma4 text checkpoint has extra KV-sharing weights - "
            "loading with mlx-lm strict=False."
        ),
    },
}


_KNOWN_VLM_EXTRA_WEIGHT_FILTERS = {
    "gemma4": {
        "message_tokens": (
            "parameters not in model",
            "per_layer_model_projection",
            "scales",
            "biases",
        ),
        "notice": (
            "Unsloth: Gemma4 VLM checkpoint has extra quantized "
            "per-layer projection state - ignoring only those known keys."
        ),
        "allowed_extra": frozenset({
            "language_model.model.per_layer_model_projection.biases",
            "language_model.model.per_layer_model_projection.scales",
        }),
    },
}


def _message_matches_known_fallback(message, rule):
    return all(token in message for token in rule.get("message_tokens", ()))


def _load_mlx_lm_with_strict_fallback(
    model_name,
    model_type,
    mlx_load,
    mlx_load_kwargs,
):
    """Load text models through mlx-lm, retrying strict=False for known safe mismatches.

    Some upstream checkpoints contain extra tensors that the current mlx-lm
    implementation intentionally does not instantiate. mlx-lm's public load()
    does not expose strict=False, so use the internal loader only for registered
    mismatch signatures.
    """
    try:
        return mlx_load(model_name, **mlx_load_kwargs)
    except ValueError as error:
        message = str(error)
        rule = _KNOWN_MLX_LM_STRICT_FALLBACKS.get(model_type)
        if rule is None or not _message_matches_known_fallback(message, rule):
            raise

        print(rule["notice"])
        from mlx_lm.utils import _download, load_model, load_tokenizer

        model_path = _download(model_name, revision=mlx_load_kwargs.get("revision"))
        model, config = load_model(
            model_path,
            lazy=mlx_load_kwargs.get("lazy", False),
            strict=False,
            model_config=mlx_load_kwargs.get("model_config"),
        )
        tokenizer = load_tokenizer(
            model_path,
            mlx_load_kwargs.get("tokenizer_config"),
            eos_token_ids=config.get("eos_token_id", None),
        )
        if mlx_load_kwargs.get("return_config", False):
            return model, tokenizer, config
        return model, tokenizer


def _load_mlx_vlm_with_extra_weight_filter(
    model_name,
    model_type,
    vlm_load,
    vlm_kwargs,
):
    """Load VLMs, filtering known extra checkpoint tensors on retry.

    Some upstream VLM checkpoints include tensors for modules that the current
    mlx-vlm class does not instantiate. mlx-vlm does not expose strict=False
    through load(), so retry with a temporary load_weights shim only for
    registered mismatch signatures and exact allow-listed keys.
    """
    try:
        return vlm_load(model_name, **vlm_kwargs)
    except ValueError as error:
        message = str(error)
        rule = _KNOWN_VLM_EXTRA_WEIGHT_FILTERS.get(model_type)
        if rule is None or not _message_matches_known_fallback(message, rule):
            raise

        print(rule["notice"])
        import mlx.nn as nn

        original_load_weights = nn.Module.load_weights
        allowed_extra = rule["allowed_extra"]

        def _load_weights_without_projection_quant_state(self, file_or_weights, strict=True):
            if isinstance(file_or_weights, list):
                file_or_weights = [
                    (key, value)
                    for key, value in file_or_weights
                    if key not in allowed_extra
                ]
            return original_load_weights(self, file_or_weights, strict=strict)

        nn.Module.load_weights = _load_weights_without_projection_quant_state
        try:
            return vlm_load(model_name, **vlm_kwargs)
        finally:
            nn.Module.load_weights = original_load_weights


def _build_vlm_model_types():
    """Build the set of model_type strings that mlx_vlm supports.

    Uses dynamic discovery via pkgutil + MODEL_REMAPPING keys/values.
    Returns frozenset; cached at module level by _is_vlm().
    """
    types_set = set()
    try:
        import mlx_vlm.models as vlm_models_pkg
        import pkgutil
        for importer, modname, ispkg in pkgutil.iter_modules(vlm_models_pkg.__path__):
            if ispkg:
                types_set.add(modname)
    except ImportError:
        pass

    try:
        from mlx_vlm.utils import MODEL_REMAPPING
        # Add both source and target keys
        for src, tgt in MODEL_REMAPPING.items():
            types_set.add(src)
            types_set.add(tgt)
    except (ImportError, AttributeError):
        pass

    return frozenset(types_set)


def _fix_missing_no_grad(model):
    """Ensure every nn.Module submodule has _no_grad / _training.

    Works around upstream model definitions that use __new__ without __init__
    (e.g. gemma4 AudioRelativePositionEmbedding).
    """
    import mlx.nn as nn
    for _, mod in model.named_modules():
        if isinstance(mod, nn.Module):
            if not hasattr(mod, "_no_grad"):
                object.__setattr__(mod, "_no_grad", set())
            if not hasattr(mod, "_training"):
                object.__setattr__(mod, "_training", True)


class _TrainingKVStore:
    """Lightweight KV store for Gemma4 KV-sharing during training.

    Gemma4 E2B/E4B have shared attention layers that borrow K/V from earlier
    "store" layers via the KV cache. During training (cache=None), the shared
    layers silently fall through to computing their own K/V from wrong hidden
    states. This class provides a minimal interface so store layers can write
    K/V and shared layers can read them, without autoregressive offset tracking.

    Implements the subset of KVCache that Attention.__call__ needs:
    - offset (always 0 for training — no prior tokens)
    - state property (returns stored K/V)
    - update_and_fetch (stores K/V from store layer)
    """
    __slots__ = ("keys", "values")

    def __init__(self):
        self.keys = None
        self.values = None

    @property
    def offset(self):
        return 0

    @property
    def state(self):
        return (self.keys, self.values)

    def update_and_fetch(self, keys, values):
        self.keys = keys
        self.values = values
        return keys, values


def _fix_gemma4_kv_sharing(model):
    """Fix Gemma4 KV-shared layers producing wrong K/V during training.

    Gemma4 E2B/E4B have num_kv_shared_layers shared attention layers that
    borrow K/V from earlier "store" layers via the KV cache. When cache=None
    (training), shared layers fall through to computing their own K/V from
    the wrong hidden state — silently producing incorrect attention.

    Fix: monkey-patch the text backbone's __call__ to create _TrainingKVStore
    objects when cache=None, so store layers populate them and shared layers
    read correct K/V.
    """
    lm = getattr(model, "language_model", None)
    if lm is None:
        return
    backbone = getattr(lm, "model", None)
    if backbone is None:
        return

    first_shared = getattr(backbone, "first_kv_shared_layer_idx", None)
    num_layers = getattr(backbone, "num_hidden_layers", None)
    if first_shared is None or num_layers is None or first_shared >= num_layers:
        return  # No shared layers

    cls = backbone.__class__
    if getattr(cls, "_kv_sharing_patched", False):
        return  # Already patched

    original_call = cls.__call__
    n_stores = first_shared  # number of store-layer cache slots

    def patched_call(self, inputs=None, inputs_embeds=None, mask=None,
                     cache=None, per_layer_inputs=None, **kwargs):
        if cache is None:
            # Objects created once under mx.compile tracing; data flow through
            # update_and_fetch/state is captured in the computation graph.
            cache = [_TrainingKVStore() for _ in range(n_stores)]
        return original_call(
            self, inputs=inputs, inputs_embeds=inputs_embeds, mask=mask,
            cache=cache, per_layer_inputs=per_layer_inputs, **kwargs,
        )

    cls.__call__ = patched_call
    cls._kv_sharing_patched = True
    n_shared = num_layers - first_shared
    print(f"Unsloth: Fixed Gemma4 KV-sharing for training "
          f"({n_shared} shared layers now read correct K/V).")


def _fix_qwen35_attention_cache(model):
    """Fix Qwen3.5 attention crash when cache=None during training.

    mlx-vlm's Qwen3.5 attention does `cache.offset + 1` without checking
    if cache is None. During training cache is always None. Patch the
    attention __call__ to handle cache=None by computing position_ids
    from scratch.
    """
    try:
        import importlib
        mod = importlib.import_module("mlx_vlm.models.qwen3_5.language")
        attn_cls = getattr(mod, "Qwen3_5Attention", None)
        if attn_cls is None:
            return
    except (ImportError, AttributeError):
        return

    if getattr(attn_cls, "_unsloth_cache_patched", False):
        return

    original_attn_call = attn_cls.__call__

    def patched_attn_call(self, x, mask=None, cache=None, position_ids=None):
        # When training (cache=None) and position_ids=None, compute them
        if cache is None and position_ids is None:
            import mlx.core as mx
            L = x.shape[1]
            position_ids = mx.arange(L)
            position_ids = mx.expand_dims(position_ids, axis=0)
            position_ids = mx.tile(position_ids, (3, 1, 1))
        return original_attn_call(self, x, mask=mask, cache=cache, position_ids=position_ids)

    attn_cls.__call__ = patched_attn_call
    attn_cls._unsloth_cache_patched = True
    print("Unsloth: Fixed Qwen3.5 attention for training (cache=None).")


def _safe_getsource(obj) -> str:
    try:
        return inspect.getsource(obj)
    except Exception:
        return ""


def _has_multimodal_strip_sanitize(model_or_cls) -> bool:
    """Return whether a loader-side sanitize path strips multimodal towers.

    We use this as a generic signal for "text-only load of a multimodal wrapper"
    instead of hardcoding every Gemma-like family by name.
    """

    cls = model_or_cls if inspect.isclass(model_or_cls) else type(model_or_cls)
    sanitize = getattr(cls, "sanitize", None)
    if sanitize is None:
        return False
    source = _safe_getsource(sanitize)
    if not source:
        return False
    return any(token in source for token in _MULTIMODAL_STRIP_KEYS)


def _get_mlx_lm_model_class(model_type: str):
    if not model_type:
        return None
    try:
        module = importlib.import_module(f"mlx_lm.models.{model_type}")
    except Exception:
        return None
    return getattr(module, "Model", None)


def _prefer_vlm_loader_for_text(config: dict, model_type: str) -> bool:
    """Return whether a multimodal wrapper should stay on the VLM load path.

    We still want a plain tokenizer API for text-only training, but some repos
    are fundamentally multimodal wrappers whose `mlx_lm` text path works only
    by stripping modality towers in `sanitize()`. That is a strong signal that
    the text loader is reconstructing a different object graph than the actual
    checkpoint. When that happens, keeping the runtime on the VLM model path is
    more robust than trying to maintain one sanitizer workaround per family.
    """

    if not _is_vlm(config):
        return False

    cls = _get_mlx_lm_model_class(model_type)
    if cls is None:
        return False

    return _has_multimodal_strip_sanitize(cls)


def _ensure_safe_text_wrapper_sanitize(model_type: str) -> None:
    """Patch nested-weight sanitize assumptions for text-only multimodal loads.

    Some `mlx_lm` multimodal wrappers sanitize text-only checkpoints by first
    unflattening weights and then blindly indexing `weights["model"]`. That is
    brittle across upstream packing changes: some checkpoints expose the text
    wrapper under `"model"`, others expose the same multimodal towers at the
    top level. We patch the sanitize method by behavior, not by one exact
    architecture, so any future loader with the same nested-vs-top-level
    assumption is handled the same way.
    """

    if not model_type or model_type in _SAFE_TEXT_SANITIZE_PATCHED:
        return

    try:
        module = importlib.import_module(f"mlx_lm.models.{model_type}")
    except Exception:
        return

    cls = getattr(module, "Model", None)
    sanitize = getattr(cls, "sanitize", None)
    if cls is None or sanitize is None:
        return

    source = _safe_getsource(sanitize)
    if 'weights["model"]' not in source or not any(token in source for token in _MULTIMODAL_STRIP_KEYS):
        return

    tree_unflatten = getattr(module, "tree_unflatten", None)
    tree_flatten = getattr(module, "tree_flatten", None)
    if tree_unflatten is None or tree_flatten is None:
        return

    original_sanitize = sanitize

    def patched_sanitize(self, weights):
        structured = tree_unflatten(list(weights.items()))
        target = structured.get("model")
        if not isinstance(target, dict):
            target = structured

        for key in _MULTIMODAL_STRIP_KEYS:
            if isinstance(target, dict):
                target.pop(key, None)

        if target is not structured and isinstance(structured, dict):
            structured["model"] = target
        return dict(tree_flatten(structured))

    cls.sanitize = patched_sanitize
    _SAFE_TEXT_SANITIZE_PATCHED.add(model_type)


def _fp16_needs_bf16_modules(model):
    """Return modules that should stay bf16 under fp16 training.

    Some Pixtral/Mistral3-family VLMs emit vision hidden states above fp16's
    finite range on real OCR-style images. The projector output remains small,
    but the selected vision features can exceed 65,504 before projection, so
    a plain `model.set_dtype(mx.float16)` overflows inside get_input_embeddings.

    Text-only loads of multimodal wrapper models can also be numerically shaky
    in fp16. We detect those by behavior: if the wrapper sanitize path strips
    multimodal towers before handing off to a text backbone, we keep that
    backbone in bf16 under an fp16 training request.
    """
    model_module = type(model).__module__
    vision_tower = getattr(model, "vision_tower", None)
    vision_module = type(vision_tower).__module__ if vision_tower is not None else ""

    modules = []
    if (
        "mlx_vlm.models.mistral3.mistral3" in model_module
        or "mlx_vlm.models.pixtral" in vision_module
    ):
        if vision_tower is not None:
            modules.append(vision_tower)

        for attr in ("multi_modal_projector", "mm_projector", "connector", "aligner"):
            module = getattr(model, attr, None)
            if module is not None:
                modules.append(module)
                break

    if _has_multimodal_strip_sanitize(model):
        language_backbone = getattr(model, "language_model", None) or getattr(model, "model", None)
        if language_backbone is not None:
            modules.append(language_backbone)

    if getattr(model, "_unsloth_text_only_vlm", False):
        language_backbone = getattr(model, "language_model", None) or getattr(model, "model", None)
        if language_backbone is not None:
            modules.append(language_backbone)

    return tuple(modules)


def _patch_mixed_precision_set_dtype(model):
    """Patch set_dtype so unstable fp16 vision towers keep a safer dtype."""
    if getattr(model, "_unsloth_mixed_precision_set_dtype_patched", False):
        return

    original_set_dtype = model.set_dtype

    def patched_set_dtype(self, dtype):
        result = original_set_dtype(dtype)

        try:
            import mlx.core as mx
        except ImportError:
            return result

        safe_modules = _fp16_needs_bf16_modules(self)
        if dtype == mx.float16 and safe_modules:
            for module in safe_modules:
                if hasattr(module, "set_dtype"):
                    module.set_dtype(mx.bfloat16)
            self._unsloth_vision_precision_override = "bf16"
        else:
            self._unsloth_vision_precision_override = None

        return result

    model.set_dtype = types.MethodType(patched_set_dtype, model)
    model._unsloth_mixed_precision_set_dtype_patched = True


# ---------------------------------------------------------------------------
# Runtime VLM quantization via monkey-patching mlx_vlm.utils.load_model
# ---------------------------------------------------------------------------
_vlm_load_model_patched = False
_original_vlm_load_model = None
_vlm_prompt_utils_patched = False
_original_vlm_apply_chat_template = None

_MULTIMODAL_ITEM_TYPES = frozenset(
    {
        "image",
        "image_url",
        "input_image",
        "audio",
        "input_audio",
        "video",
    }
)
_NON_USER_ROLES = frozenset({"system", "assistant"})
_ROLE_PROMPT_NAMES = {
    "user": "Human",
    "assistant": "Assistant",
    "system": "System",
}

# Fragments that identify text modules skipped by default for both LLMs and VLMs.
_DEFAULT_QUANT_SKIP_FRAGMENTS = (
    "lm_head", "embed_tokens",
)

# Fragments that identify multimodal sub-networks we must *never* quantize.
_MULTIMODAL_SKIP_FRAGMENTS = (
    "multi_modal_projector", "mm_projector", "connector", "aligner",
    "projector",
    "vision_tower", "vision_model", "vision_encoder", "visual",
    "embed_vision", "vision_embed_tokens", "img_processor", "img_projection",
    "audio_encoder", "audio_projection", "embed_audio",
)

_MLX_QUANT_MODE_DEFAULTS = {
    "affine": (64, 4),
    "mxfp4": (32, 4),
    "nvfp4": (16, 4),
    "mxfp8": (32, 8),
}


@dataclass
class _MLXQuantizationSpec:
    enabled: bool = False
    bits: int | None = None
    group_size: int | None = None
    mode: str | None = None
    source: str = "none"
    quantize_modules: tuple[str, ...] | None = None
    has_callable_predicate: bool = False
    force_requantize: bool = False

    def to_metadata(self):
        data = asdict(self)
        if self.quantize_modules is not None:
            data["quantize_modules"] = list(self.quantize_modules)
        return data

_LORA_TARGET_ALIASES = {
    "q_proj": {"qkv", "qkv_proj", "query_key_value", "Wqkv"},
    "k_proj": {"qkv", "qkv_proj", "query_key_value", "Wqkv"},
    "v_proj": {"qkv", "qkv_proj", "query_key_value", "Wqkv"},
    "o_proj": {"proj", "out_proj", "dense"},
    "gate_proj": {"gate_up_proj"},
    "up_proj": {"gate_up_proj"},
}


def _lora_name_matches_target(name, target_modules):
    if target_modules is None:
        return True
    if not name:
        return False
    parts = name.split(".")
    leaf = parts[-1]
    parent_leaf = parts[-2] if len(parts) >= 2 else ""
    if (
        leaf in target_modules
        or parent_leaf in target_modules
        or name in target_modules
    ):
        return True
    return any(
        alias in (leaf, parent_leaf)
        for target in target_modules
        for alias in _LORA_TARGET_ALIASES.get(target, ())
    )


def _get_existing_mlx_quantization(config_data: dict):
    if not isinstance(config_data, dict):
        return None
    quantization = config_data.get("quantization", None)
    if quantization:
        return quantization
    quantization_config = config_data.get("quantization_config", None)
    if quantization_config:
        return quantization_config
    return None


def _vlm_config_is_already_quantized(config_data: dict) -> bool:
    """Return True when config indicates pre-quantized MLX/HF weights."""
    quantization = _get_existing_mlx_quantization(config_data)
    if quantization:
        return True
    return False


def _quant_config_to_dict(quantization_config):
    if quantization_config is None:
        return {}
    if isinstance(quantization_config, dict):
        return dict(quantization_config)
    if hasattr(quantization_config, "to_dict"):
        return dict(quantization_config.to_dict())
    data = {}
    for key in (
        "load_in_4bit", "load_in_8bit", "bnb_4bit_compute_dtype",
        "bnb_4bit_quant_type", "bnb_4bit_use_double_quant",
        "llm_int8_threshold", "llm_int8_skip_modules",
    ):
        if hasattr(quantization_config, key):
            data[key] = getattr(quantization_config, key)
    return data


def _reject_unsupported_hf_quantization_fields(config_dict):
    unsupported = []
    allowed_defaults = {
        "bnb_4bit_compute_dtype": ("float32", None),
        "bnb_4bit_quant_type": ("fp4", None),
        "bnb_4bit_use_double_quant": (False, None),
        "bnb_4bit_quant_storage": ("uint8", None),
        "llm_int8_threshold": (6.0, None),
        "llm_int8_skip_modules": (None, [], {}),
        "llm_int8_enable_fp32_cpu_offload": (False, None),
        "llm_int8_has_fp16_weight": (False, None),
    }
    for key in (
        "bnb_4bit_compute_dtype", "bnb_4bit_quant_type",
        "bnb_4bit_use_double_quant", "bnb_4bit_quant_storage",
        "llm_int8_threshold", "llm_int8_skip_modules",
        "llm_int8_enable_fp32_cpu_offload", "llm_int8_has_fp16_weight",
    ):
        value = config_dict.get(key, None)
        if value not in allowed_defaults[key]:
            unsupported.append(key)
    if unsupported:
        raise ValueError(
            "Unsloth: MLX quantization does not support these "
            f"BitsAndBytesConfig fields: {', '.join(sorted(unsupported))}. "
            "Use mlx_quantization_config/q_bits/q_group_size/q_mode instead."
        )


def _normalize_quantize_modules(value):
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(x) for x in value)
    raise TypeError("Unsloth: quantize_modules must be a string or list of strings.")


def _infer_snapshot_commit(path):
    if not path:
        return None
    parts = os.path.normpath(str(path)).split(os.sep)
    try:
        index = parts.index("snapshots")
    except ValueError:
        return None
    if index + 1 >= len(parts):
        return None
    commit = parts[index + 1]
    return commit or None


def _effective_mlx_quantization_map(model):
    quantized = {}
    quantized.update(_quantization_config_to_path_map(
        _get_existing_mlx_quantization(getattr(model, "_config", None))
    ))
    quantized.update(_quantization_config_to_path_map(
        getattr(model, "_unsloth_quantization_config", None)
    ))
    for path, module in model.named_modules():
        if not path:
            continue
        if type(module).__name__ not in {"QuantizedLinear", "QuantizedEmbedding"}:
            continue
        path = _canonical_mlx_quantization_path(path)
        entry = {}
        for key in ("bits", "group_size", "mode"):
            if hasattr(module, key):
                value = getattr(module, key)
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    pass
                entry[key] = value
        quantized[path] = entry
    return quantized


def _quantization_config_to_path_map(config):
    if not isinstance(config, dict):
        return {}
    defaults = {
        key: config.get(key)
        for key in ("bits", "group_size", "mode")
        if config.get(key) is not None
    }
    if defaults and "mode" not in defaults:
        defaults["mode"] = "affine"
    reserved = {
        "bits", "group_size", "mode", "quant_method", "skip_vision",
        "skip_projector", "skip_lm_head",
    }
    quantized = {}
    for path, value in config.items():
        if path in reserved or not isinstance(value, dict):
            continue
        entry = dict(defaults)
        entry.update(value)
        if "bits" not in entry or "group_size" not in entry:
            continue
        if "mode" not in entry:
            entry["mode"] = "affine"
        quantized[_canonical_mlx_quantization_path(str(path))] = {
            key: int(item) if key in ("bits", "group_size") else item
            for key, item in entry.items()
            if key in ("bits", "group_size", "mode")
        }
    return quantized


def _canonical_mlx_quantization_path(path):
    # mlx-lm LoRALinear stores the wrapped base layer under ".linear".
    # Adapter metadata must describe the underlying base path so validation can
    # run before load_adapters re-wraps the module.
    if path.endswith(".linear"):
        return path[:-len(".linear")]
    return path


def _normalize_quantization_map(value):
    if not value:
        return {}
    normalized = {}
    for path, config in dict(value).items():
        path = _canonical_mlx_quantization_path(str(path))
        if not isinstance(config, dict):
            normalized[path] = config
            continue
        entry = {}
        for key in ("bits", "group_size", "mode"):
            if key not in config:
                continue
            item = config[key]
            try:
                item = int(item)
            except (TypeError, ValueError):
                pass
            entry[key] = item
        normalized[path] = entry
    return normalized


def _validate_mlx_adapter_base(model, adapter_cfg):
    expected_map = _normalize_quantization_map(
        adapter_cfg.get("base_resolved_quantization_map")
        or adapter_cfg.get("base_quantization_map")
    )
    if expected_map:
        live_map = _normalize_quantization_map(_effective_mlx_quantization_map(model))
        if live_map != expected_map:
            missing = sorted(set(expected_map) - set(live_map))
            extra = sorted(set(live_map) - set(expected_map))
            changed = sorted(
                path for path in set(expected_map) & set(live_map)
                if expected_map[path] != live_map[path]
            )
            details = []
            if missing:
                details.append(f"missing quantized modules: {missing[:5]!r}")
            if extra:
                details.append(f"unexpected quantized modules: {extra[:5]!r}")
            if changed:
                details.append(f"changed quantized modules: {changed[:5]!r}")
            detail_text = "; ".join(details) if details else "quantization map differs"
            raise ValueError(
                "Unsloth: This MLX adapter was saved for a base model with a "
                f"different resolved quantization map ({detail_text}). Reload "
                "the adapter with the exact saved base and quantization settings, "
                "or export a standalone merged/quantized model."
            )

    expected_config = adapter_cfg.get("base_quantization_config")
    if isinstance(expected_config, dict) and {"bits", "group_size"} <= set(expected_config):
        expected = _global_quant_params(expected_config)
        live = (
            _global_quant_params(getattr(model, "_unsloth_quantization_config", None))
            or _global_quant_params(
                _get_existing_mlx_quantization(getattr(model, "_config", None))
            )
        )
        if expected is not None and live is None and not expected_map:
            raise ValueError(
                "Unsloth: This MLX adapter was saved with base quantization "
                f"{expected!r}, but the reloaded base has no verifiable MLX "
                "quantization config. Reload the adapter with the exact saved "
                "base and quantization settings, or export a standalone "
                "merged/quantized model."
            )
        if expected is not None and live is not None and live != expected:
            raise ValueError(
                "Unsloth: This MLX adapter was saved with base quantization "
                f"{expected!r}, but the reloaded base has {live!r}."
            )


def _quant_config_from_resolved_map(resolved_map):
    resolved_map = _normalize_quantization_map(resolved_map)
    if not resolved_map:
        return None
    configs = {
        (
            config.get("bits"),
            config.get("group_size"),
            config.get("mode", "affine"),
        )
        for config in resolved_map.values()
    }
    if len(configs) != 1:
        return None
    bits, group_size, mode = next(iter(configs))
    if bits is None or group_size is None:
        return None
    return {
        "bits": bits,
        "group_size": group_size,
        "mode": mode,
        "quantize_modules": sorted(resolved_map),
    }


def _adapter_actual_quant_config(adapter_cfg, resolved_map):
    expected = _global_quant_params(adapter_cfg.get("base_quantization_config"))
    if expected is not None:
        expected["quantize_modules"] = None
        return expected
    return _quant_config_from_resolved_map(resolved_map)


def _adapter_base_revision(adapter_cfg):
    return (
        adapter_cfg.get("base_model_commit_hash")
        or adapter_cfg.get("base_model_revision")
    )


def _adapter_needs_runtime_quantization(adapter_cfg, quant_policy):
    if adapter_cfg.get("requires_unsloth_mlx_runtime_quantization") is not None:
        return bool(adapter_cfg.get("requires_unsloth_mlx_runtime_quantization"))
    source = adapter_cfg.get("base_quantized_source")
    if source is not None:
        return source == "runtime"
    return bool(quant_policy.get("enabled"))


def _resolve_mlx_quantization_spec(
    *,
    load_in_4bit,
    load_in_8bit,
    load_in_16bit,
    load_in_fp8,
    full_finetuning,
    q_bits,
    q_group_size,
    q_mode,
    mlx_quantization_config,
    quantization_config,
    quant_predicate,
    quantize_modules,
    force_requantize,
):
    if full_finetuning:
        load_in_4bit = load_in_8bit = load_in_fp8 = False
        load_in_16bit = True

    mlx_config = dict(mlx_quantization_config or {})
    if "quantize_modules" in mlx_config and quantize_modules is None:
        quantize_modules = mlx_config.get("quantize_modules")
    if "modules" in mlx_config and quantize_modules is None:
        quantize_modules = mlx_config.get("modules")

    hf_config = _quant_config_to_dict(quantization_config)
    if hf_config:
        if not mlx_config and any(
            key in hf_config
            for key in (
                "bits", "q_bits", "group_size", "q_group_size", "mode", "q_mode",
                "quantize_modules", "modules",
            )
        ):
            mlx_config = dict(hf_config)
        _reject_unsupported_hf_quantization_fields(hf_config)
        if hf_config.get("load_in_4bit"):
            load_in_4bit = True
        if hf_config.get("load_in_8bit"):
            load_in_8bit = True
        if quantize_modules is None:
            if "quantize_modules" in hf_config:
                quantize_modules = hf_config.get("quantize_modules")
            elif "modules" in hf_config:
                quantize_modules = hf_config.get("modules")

    explicit_mlx_quant_config = bool(mlx_config) or q_bits is not None or q_mode is not None
    if explicit_mlx_quant_config and load_in_4bit and not any((load_in_8bit, load_in_fp8, load_in_16bit)):
        # FastMLXModel defaults load_in_4bit=True for CUDA API parity, but
        # explicit MLX quantization knobs should not be silently shadowed by
        # that implicit default.
        if not hf_config.get("load_in_4bit"):
            print(
                "Unsloth: Explicit MLX quantization settings detected - "
                "disabling the default load_in_4bit=True."
            )
            load_in_4bit = False

    load_flags = {
        "load_in_4bit": bool(load_in_4bit),
        "load_in_8bit": bool(load_in_8bit),
        "load_in_fp8": bool(load_in_fp8),
        "load_in_16bit": bool(load_in_16bit),
    }
    if load_flags["load_in_4bit"] and any(
        load_flags[x] for x in ("load_in_8bit", "load_in_fp8", "load_in_16bit")
    ):
        # FastMLXModel keeps CUDA Unsloth's load_in_4bit=True default. Let
        # explicit non-4bit flags override that default for ergonomic calls like
        # from_pretrained(..., load_in_8bit=True).
        load_flags["load_in_4bit"] = False
        load_in_4bit = False
    active = [name for name, enabled in load_flags.items() if enabled]
    if len([x for x in active if x != "load_in_16bit"]) > 1:
        raise ValueError(
            "Unsloth: pass only one MLX quantization load flag among "
            "load_in_4bit, load_in_8bit, and load_in_fp8."
        )
    if load_in_16bit and any(load_flags[x] for x in ("load_in_4bit", "load_in_8bit", "load_in_fp8")):
        raise ValueError("Unsloth: load_in_16bit conflicts with quantized load flags.")

    if load_in_16bit:
        return _MLXQuantizationSpec(
            enabled=False,
            source="load_in_16bit",
            quantize_modules=_normalize_quantize_modules(quantize_modules),
            has_callable_predicate=quant_predicate is not None,
            force_requantize=bool(force_requantize),
        )

    source = "none"
    if load_in_4bit:
        bits, mode, source = 4, "affine", "load_in_4bit"
    elif load_in_8bit:
        bits, mode, source = 8, "mxfp8", "load_in_8bit"
    elif load_in_fp8:
        bits, mode, source = 8, "mxfp8", "load_in_fp8"
    elif mlx_config:
        bits = mlx_config.get("bits", mlx_config.get("q_bits", q_bits))
        mode = mlx_config.get("mode", mlx_config.get("q_mode", q_mode))
        if bits is None and mode is None:
            bits, mode = 4, "affine"
        source = "mlx_quantization_config"
    else:
        bits, mode = q_bits, q_mode
        source = "q_args" if bits is not None or mode is not None else "none"

    if source.startswith("load_in"):
        # q_group_size is the only low-level override intentionally allowed
        # for Unsloth-style load flags.
        group_size = q_group_size
    else:
        group_size = mlx_config.get("group_size", mlx_config.get("q_group_size", q_group_size))

    if mode is None and bits is not None:
        mode = "mxfp8" if bits == 8 else "affine"
    if mode is not None:
        if mode not in _MLX_QUANT_MODE_DEFAULTS:
            raise ValueError(
                f"Unsloth: Unsupported MLX q_mode={mode!r}. "
                f"Supported modes: {sorted(_MLX_QUANT_MODE_DEFAULTS)}"
            )
        default_group, default_bits = _MLX_QUANT_MODE_DEFAULTS[mode]
        bits = bits or default_bits
        group_size = group_size or default_group

    enabled = bits is not None or mode is not None
    return _MLXQuantizationSpec(
        enabled=bool(enabled),
        bits=int(bits) if bits is not None else None,
        group_size=int(group_size) if group_size is not None else None,
        mode=mode,
        source=source,
        quantize_modules=_normalize_quantize_modules(quantize_modules),
        has_callable_predicate=quant_predicate is not None,
        force_requantize=bool(force_requantize),
    )


def _global_quant_params(quantization):
    if not isinstance(quantization, dict):
        return None
    if {"bits", "group_size"} <= set(quantization):
        return {
            "bits": quantization.get("bits"),
            "group_size": quantization.get("group_size"),
            "mode": quantization.get("mode", "affine"),
        }
    return None


def _ensure_quantization_compatible(config_data, spec: _MLXQuantizationSpec, model_name):
    existing = _get_existing_mlx_quantization(config_data)
    if not existing or not spec.enabled:
        return "none"
    existing_global = _global_quant_params(existing)
    requested = {
        "bits": spec.bits,
        "group_size": spec.group_size,
        "mode": spec.mode,
    }
    is_full_model_request = spec.quantize_modules is None and not spec.has_callable_predicate
    if existing_global == requested and is_full_model_request:
        return "compatible"
    if (
        is_full_model_request
        and spec.source in ("load_in_4bit", "load_in_8bit", "load_in_fp8")
        and existing_global is not None
        and existing_global.get("bits") == spec.bits
    ):
        return "compatible"
    if (
        is_full_model_request
        and spec.source == "load_in_4bit"
        and existing_global is not None
    ):
        return "compatible"
    if spec.force_requantize:
        warnings.warn(
            f"Unsloth: '{model_name}' is already quantized but force_requantize=True; "
            "attempting requested MLX requantization.",
            stacklevel=2,
        )
        return "force"
    if existing_global is None and isinstance(existing, dict):
        raise ValueError(
            f"Unsloth: '{model_name}' has per-module MLX quantization metadata "
            f"{existing!r}, but requested {requested!r}. Unsloth cannot treat "
            "a partial quantization config as a globally quantized model. Load "
            "without additional quantization, request matching per-module "
            "quantization, or pass force_requantize=True if you intend to "
            "replace the existing quantized modules."
        )
    raise ValueError(
        f"Unsloth: '{model_name}' is already quantized with {existing!r}, "
        f"but requested incompatible MLX quantization {requested!r}. "
        "Load without quantization, request matching quantization, or pass "
        "force_requantize=True to explicitly attempt requantization."
    )


def _path_matches_any(path, patterns):
    if patterns is None:
        return True
    if not patterns:
        return False
    parts = path.split(".")
    return any(
        path == p
        or path.endswith(f".{p}")
        or p in parts
        or fnmatch(path, p)
        for p in patterns
    )


def _compose_mlx_quant_predicate(model, spec: _MLXQuantizationSpec, *, is_vlm, user_predicate=None):
    model_predicate = getattr(model, "quant_predicate", None)
    include_patterns = spec.quantize_modules
    try:
        from mlx_vlm.utils import skip_multimodal_module
    except ImportError:
        skip_multimodal_module = None

    def predicate(path: str, module):
        path_parts = path.split(".")
        explicitly_included = (
            include_patterns is not None and _path_matches_any(path, include_patterns)
        )

        if is_vlm:
            if skip_multimodal_module is not None and skip_multimodal_module(path):
                return False
            if any(fragment in path_parts for fragment in _MULTIMODAL_SKIP_FRAGMENTS):
                return False

        if not explicitly_included:
            if any(fragment in path_parts for fragment in _DEFAULT_QUANT_SKIP_FRAGMENTS):
                return False
            if type(module).__name__ in {"Embedding", "QuantizedEmbedding"}:
                return False

        if include_patterns is not None and not _path_matches_any(path, include_patterns):
            return False
        if model_predicate is not None:
            model_result = model_predicate(path, module)
            if model_result is False:
                return False
            if isinstance(model_result, dict):
                return model_result
        if user_predicate is not None:
            user_result = user_predicate(path, module)
            if user_result is not True:
                return user_result
        return True

    return predicate


def _dequantize_selected_mlx_modules(model, predicate):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten

    replacements = []
    for path, module in model.named_modules():
        if not predicate(path, module):
            continue
        if isinstance(module, nn.QuantizedLinear):
            weight = mx.dequantize(
                module.weight,
                module.scales,
                module.biases,
                group_size=module.group_size,
                bits=module.bits,
                mode=module.mode,
            )
            output_dims, input_dims = weight.shape
            replacement = nn.Linear(input_dims, output_dims, bias="bias" in module)
            replacement.weight = weight
            if "bias" in module:
                replacement.bias = module.bias
            replacements.append((path, replacement))
        elif isinstance(module, nn.QuantizedEmbedding):
            weight = mx.dequantize(
                module.weight,
                module.scales,
                module.biases,
                group_size=module.group_size,
                bits=module.bits,
                mode=module.mode,
            )
            num_embeddings, dims = weight.shape
            replacement = nn.Embedding(num_embeddings, dims)
            replacement.weight = weight
            replacements.append((path, replacement))

    if replacements:
        model.update_modules(tree_unflatten(replacements))
        mx.eval(model.parameters())
    return len(replacements)


def _apply_mlx_quantization(model, config, spec: _MLXQuantizationSpec, *, is_vlm, user_predicate=None):
    if not spec.enabled:
        model._unsloth_quantization_config = None
        model._unsloth_quantization_policy = spec.to_metadata()
        model._unsloth_quantized_source = "none"
        return model, config

    from mlx_lm.utils import quantize_model

    predicate = _compose_mlx_quant_predicate(
        model, spec, is_vlm=is_vlm, user_predicate=user_predicate,
    )
    if spec.force_requantize:
        existing_quantization = _get_existing_mlx_quantization(config)
        is_selective = is_vlm or spec.quantize_modules is not None or user_predicate is not None
        if is_selective and _global_quant_params(existing_quantization):
            raise ValueError(
                "Unsloth: selective force_requantize=True on a globally quantized "
                "MLX model is unsupported because MLX config metadata cannot "
                "safely express dequantized exceptions. Load a non-quantized base "
                "model first, or force-requantize the full model."
            )
        n_dequantized = _dequantize_selected_mlx_modules(model, predicate)
        if n_dequantized == 0 and _get_existing_mlx_quantization(config):
            raise ValueError(
                "Unsloth: force_requantize=True was requested for an already "
                "quantized model, but no selected MLX quantized modules could "
                "be dequantized for requantization."
            )
        if not is_vlm and spec.quantize_modules is None and user_predicate is None:
            config = dict(config or {})
            config.pop("quantization", None)
            config.pop("quantization_config", None)
    if is_vlm or spec.quantize_modules is not None or user_predicate is not None:
        config = dict(config or {})
        config.setdefault("quantization", {})
    model, updated_config = quantize_model(
        model,
        config,
        group_size=spec.group_size,
        bits=spec.bits,
        mode=spec.mode or "affine",
        quant_predicate=predicate,
    )
    model._config = updated_config
    model._unsloth_quantization_config = updated_config.get(
        "quantization_config", updated_config.get("quantization")
    )
    model._unsloth_quantization_policy = spec.to_metadata()
    model._unsloth_quantized_source = "runtime"
    return model, updated_config


def _build_vlm_quant_predicate(model):
    """Build a quant_predicate for mlx_lm.utils.quantize_model.

    Two layers of filtering:
    1. Hard-skip multimodal modules (vision tower, projectors, embeddings, …)
    2. Delegate to model.quant_predicate for model-specific rules (MoE gates, …)
    """
    try:
        from mlx_vlm.utils import skip_multimodal_module
    except ImportError:
        skip_multimodal_module = None

    # Trigger model-specific setup (e.g. phi4mm LoRA merge side-effect)
    model_predicate = getattr(model, "quant_predicate", None)

    def predicate(path: str, module):
        # 1. Hard skip — mlx_vlm's own multimodal check
        if skip_multimodal_module is not None and skip_multimodal_module(path):
            return False
        # 2. Hard skip — extra fragments (embeddings, projectors, …)
        path_parts = path.split(".")
        for frag in _MULTIMODAL_SKIP_FRAGMENTS:
            if frag in path_parts:
                return False
        # 3. Model-specific predicate (MoE gates → 8-bit, phi4mm exclusions, …)
        if model_predicate is not None:
            return model_predicate(path, module)
        return True

    return predicate


def _patched_vlm_load_model(model_path, lazy=False, **kwargs):
    """Drop-in replacement for mlx_vlm.utils.load_model with runtime quantization."""
    import mlx.core as mx

    quantization_spec = kwargs.pop("quantization_spec", None)
    user_quant_predicate = kwargs.pop("quant_predicate", None)
    q_bits = kwargs.pop("q_bits", None)
    q_group_size = kwargs.pop("q_group_size", 64)
    q_mode = kwargs.pop("q_mode", None)
    quantize_modules = kwargs.pop("quantize_modules", None)

    # Load float weights (always lazy so quantization runs on lazy arrays)
    model = _original_vlm_load_model(model_path, lazy=True, **kwargs)

    if quantization_spec is None and q_bits is not None:
        quantization_spec = _MLXQuantizationSpec(
            enabled=True,
            bits=q_bits,
            group_size=q_group_size,
            mode=q_mode or ("mxfp8" if q_bits == 8 else "affine"),
            source="q_args",
            quantize_modules=_normalize_quantize_modules(quantize_modules),
            has_callable_predicate=user_quant_predicate is not None,
        )

    if quantization_spec is not None and quantization_spec.enabled:
        from mlx_vlm.utils import load_config

        config = load_config(model_path)
        model, updated_config = _apply_mlx_quantization(
            model,
            config,
            quantization_spec,
            is_vlm=True,
            user_predicate=user_quant_predicate,
        )
        model._config = updated_config

    if not lazy:
        mx.eval(model.parameters())

    return model


def _ensure_vlm_load_model_patched():
    """Idempotent installer — patches mlx_vlm.utils.load_model on first call."""
    global _vlm_load_model_patched, _original_vlm_load_model

    if _vlm_load_model_patched:
        return

    import mlx_vlm.utils as _vlm_utils

    _original_vlm_load_model = _vlm_utils.load_model
    _vlm_utils.load_model = _patched_vlm_load_model
    _vlm_load_model_patched = True


def _content_has_structured_multimodal_markers(content):
    """Return True when content already contains explicit image/audio/video items."""
    if isinstance(content, list):
        for item in content:
            if _content_has_structured_multimodal_markers(item):
                return True
        return False

    if isinstance(content, dict):
        item_type = str(content.get("type", "")).lower()
        if item_type in _MULTIMODAL_ITEM_TYPES:
            return True
        nested = content.get("content", None)
        if nested is not None and nested is not content:
            return _content_has_structured_multimodal_markers(nested)
        return False

    return False


def _normalize_prompt_messages(prompt_utils_module, prompt):
    """Normalize prompt-like items into a message list without discarding content."""
    messages = []
    for item in prompt:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue

        role_content = prompt_utils_module._get_role_content(item)
        if role_content is not None:
            role, content = role_content
            messages.append({"role": role, "content": content})
            continue

        messages.append({"role": "user", "content": str(item)})
    return messages


def _messages_have_structured_multimodal_content(messages):
    """Return True when any normalized message already carries media markers."""
    return any(
        _content_has_structured_multimodal_markers(message.get("content", ""))
        for message in messages
    )


def _first_media_user_message_index(messages):
    """Return the first user-like turn that should own conversation-level media."""
    for i, message in enumerate(messages):
        role = str(message.get("role", "user")).lower()
        if role not in _NON_USER_ROLES:
            return i
    return -1


def _anchor_conversation_media_to_first_user_turn(
    prompt_utils_module,
    model_type,
    messages,
    *,
    num_images,
    num_audios,
    kwargs,
):
    """Rebuild text-only chat while keeping conversation-level media on turn 1.

    mlx-vlm's prompt helper used to attach `num_images` / `num_audios` to the
    last user turn for list prompts. That shifts the image token between turns
    during multi-turn chat, which breaks prompt-cache reuse for models like
    Qwen3.5 that pre-compute multimodal rope positions from the full prompt.
    """
    target_idx = _first_media_user_message_index(messages)
    if target_idx < 0:
        return messages

    anchored = []
    for i, message in enumerate(messages):
        role = str(message.get("role", "user"))
        content = prompt_utils_module.extract_text_from_content(
            message.get("content", "")
        )
        is_target = i == target_idx and role.lower() not in _NON_USER_ROLES
        anchored.append(
            prompt_utils_module.get_message_json(
                model_type,
                content,
                role,
                skip_image_token=not is_target,
                skip_audio_token=not is_target,
                num_images=num_images,
                num_audios=num_audios,
                **kwargs,
            )
        )
    return anchored


def _get_vlm_image_token(processor):
    """Best-effort image token string for manual prompt fallbacks."""
    image_token = getattr(processor, "image_token", None)
    if isinstance(image_token, str) and image_token:
        return image_token
    tokenizer = getattr(processor, "tokenizer", None)
    image_token = getattr(tokenizer, "image_token", None)
    if isinstance(image_token, str) and image_token:
        return image_token
    return "<image>"


def _flatten_multimodal_content_for_prompt(
    content,
    image_token,
    *,
    audio_token="<audio>",
    video_token="<video>",
):
    """Flatten OpenAI-style multimodal content into a plain prompt string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            parts.append(
                _flatten_multimodal_content_for_prompt(
                    item,
                    image_token,
                    audio_token=audio_token,
                    video_token=video_token,
                )
            )
        stitched = []
        prev_marker = False
        markers = {image_token, audio_token, video_token}
        for part in parts:
            if not part:
                continue
            is_marker = part in markers
            if prev_marker and not is_marker and not part[0].isspace():
                stitched.append(" ")
            stitched.append(part)
            prev_marker = is_marker
        return "".join(stitched).strip()
    if isinstance(content, dict):
        item_type = str(content.get("type", "")).lower()
        if item_type in ("image", "image_url", "input_image"):
            return image_token
        if item_type in ("audio", "input_audio"):
            return audio_token
        if item_type == "video":
            return video_token
        nested = content.get("content", None)
        if nested is not None and nested is not content:
            return _flatten_multimodal_content_for_prompt(
                nested,
                image_token,
                audio_token=audio_token,
                video_token=video_token,
            )
        text = content.get("text", "") or content.get("content", "")
        return str(text) if text else ""
    return str(content) if content is not None else ""


def _build_role_prompt_fallback(processor, messages, *, add_generation_prompt):
    """Render a plain role-prefixed prompt while preserving media markers."""
    image_token = f"<|vision_start|>{_get_vlm_image_token(processor)}<|vision_end|>"
    lines = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        flattened = _flatten_multimodal_content_for_prompt(content, image_token)
        prefix = _ROLE_PROMPT_NAMES.get(role.lower(), role.capitalize())
        lines.append(f"{prefix}: {flattened}" if flattened else f"{prefix}:")

    if add_generation_prompt:
        lines.append("Assistant:")
    return "\n".join(lines).strip()


_EMPTY_VLM_CHAT_TEMPLATE_FALLBACKS = {
    # Upstream Qwen2-VL MLX template expects a flat content list instead of
    # standard role-wrapped chat messages, so normal chat rendering can return
    # an empty string. Fall back to a simple role-prefixed prompt instead.
    "qwen2_vl": _build_role_prompt_fallback,
}


def _render_empty_template_fallback(model_type, processor, messages, *, add_generation_prompt):
    """Render a model-specific fallback when an upstream template is unusable."""
    builder = _EMPTY_VLM_CHAT_TEMPLATE_FALLBACKS.get(model_type)
    if builder is None:
        return None
    return builder(
        processor,
        messages,
        add_generation_prompt=add_generation_prompt,
    )


def _prepare_vlm_template_messages(
    prompt_utils_module,
    model_type,
    prompt,
    *,
    num_images,
    num_audios,
    kwargs,
):
    """Normalize prompt input and apply the smallest rewrite needed for VLM chat.

    There are two cases where we bypass mlx-vlm's higher-level helper and render
    messages ourselves:
    1. The prompt already uses structured multimodal content and must survive
       intact instead of being flattened back to plain text.
    2. The caller passes conversation-level `num_images` / `num_audios`. In
       that case we anchor those media tokens to the first user turn so prompt
       cache reuse does not silently change rope positions between turns.
    """
    prompt_items = prompt if isinstance(prompt, list) else [prompt]
    normalized_messages = _normalize_prompt_messages(prompt_utils_module, prompt_items)
    has_structured_multimodal = _messages_have_structured_multimodal_content(
        normalized_messages
    )
    needs_media_anchor = (
        not has_structured_multimodal and (num_images > 0 or num_audios > 0)
    )

    template_messages = normalized_messages
    if needs_media_anchor:
        template_messages = _anchor_conversation_media_to_first_user_turn(
            prompt_utils_module,
            model_type,
            normalized_messages,
            num_images=num_images,
            num_audios=num_audios,
            kwargs=kwargs,
        )

    return normalized_messages, template_messages, (
        has_structured_multimodal or needs_media_anchor
    )


def _render_vlm_template_or_fallback(
    prompt_utils_module,
    model_type,
    processor,
    messages,
    *,
    add_generation_prompt,
    kwargs,
):
    """Render a message list, falling back only when the upstream template is empty."""
    rendered = prompt_utils_module.get_chat_template(
        processor,
        messages,
        add_generation_prompt,
        **kwargs,
    )
    if isinstance(rendered, str) and rendered.strip():
        return rendered

    fallback = _render_empty_template_fallback(
        model_type,
        processor,
        messages,
        add_generation_prompt=add_generation_prompt,
    )
    if fallback is not None:
        return fallback
    return rendered


def _ensure_vlm_prompt_utils_patched():
    """Patch mlx-vlm chat-template helper for stable multi-turn multimodal chat."""
    global _vlm_prompt_utils_patched, _original_vlm_apply_chat_template

    if _vlm_prompt_utils_patched:
        return

    import importlib

    prompt_utils = importlib.import_module("mlx_vlm.prompt_utils")
    _original_vlm_apply_chat_template = prompt_utils.apply_chat_template

    def patched_apply_chat_template(
        processor,
        config,
        prompt,
        add_generation_prompt=True,
        return_messages=False,
        num_images=0,
        num_audios=0,
        **kwargs,
    ):
        config_data = config if isinstance(config, dict) else config.__dict__
        model_type = config_data["model_type"]

        if not isinstance(prompt, (dict, list)):
            return _original_vlm_apply_chat_template(
                processor,
                config,
                prompt,
                add_generation_prompt=add_generation_prompt,
                return_messages=return_messages,
                num_images=num_images,
                num_audios=num_audios,
                **kwargs,
            )

        normalized_messages, template_messages, needs_custom_render = (
            _prepare_vlm_template_messages(
                prompt_utils,
                model_type,
                prompt,
                num_images=num_images,
                num_audios=num_audios,
                kwargs=kwargs,
            )
        )
        if needs_custom_render:
            if return_messages:
                return template_messages
            return _render_vlm_template_or_fallback(
                prompt_utils,
                model_type,
                processor,
                template_messages,
                add_generation_prompt=add_generation_prompt,
                kwargs=kwargs,
            )

        rendered = _original_vlm_apply_chat_template(
            processor,
            config,
            prompt,
            add_generation_prompt=add_generation_prompt,
            return_messages=return_messages,
            num_images=num_images,
            num_audios=num_audios,
            **kwargs,
        )
        if return_messages or not (isinstance(rendered, str) and not rendered.strip()):
            return rendered

        return _render_vlm_template_or_fallback(
            prompt_utils,
            model_type,
            processor,
            normalized_messages,
            add_generation_prompt=add_generation_prompt,
            kwargs=kwargs,
        )

    prompt_utils.apply_chat_template = patched_apply_chat_template

    for modname in (
        "mlx_vlm.chat",
        "mlx_vlm.generate",
        "mlx_vlm.server",
        "mlx_vlm.evals.utils",
    ):
        try:
            module = importlib.import_module(modname)
        except Exception:
            continue
        if hasattr(module, "apply_chat_template"):
            module.apply_chat_template = patched_apply_chat_template

    _vlm_prompt_utils_patched = True


def _mlx_save_pretrained_merged(self, save_directory, tokenizer=None, **kwargs):
    from .mlx_utils import save_pretrained_merged
    tokenizer = tokenizer or self._tokenizer
    save_pretrained_merged(self, tokenizer, save_directory, **kwargs)


def _mlx_save_pretrained_gguf(self, save_directory, tokenizer=None,
                               quantization_method="fast_quantized", **kwargs):
    from .mlx_utils import save_pretrained_gguf
    tokenizer = tokenizer or self._tokenizer
    save_pretrained_gguf(self, tokenizer, save_directory,
                         quantization_method=quantization_method)


def _mlx_push_to_hub_merged(self, repo_id, tokenizer=None, **kwargs):
    from .mlx_utils import push_to_hub_merged
    tokenizer = tokenizer or self._tokenizer
    push_to_hub_merged(self, tokenizer, repo_id, repo_id=repo_id, **kwargs)


def _mlx_push_to_hub_gguf(self, repo_id, tokenizer=None,
                            quantization_method="fast_quantized", **kwargs):
    from .mlx_utils import push_to_hub_gguf
    tokenizer = tokenizer or self._tokenizer
    push_to_hub_gguf(self, tokenizer, repo_id, repo_id=repo_id,
                     quantization_method=quantization_method, **kwargs)


def _mlx_save_lora_adapters(self, path, adapter_config=None):
    from .mlx_utils import save_lora_adapters
    save_lora_adapters(self, path, adapter_config=adapter_config)


def _patch_mlx_saving(model, tokenizer):
    """Attach save/push methods to the model, matching unsloth's CUDA pattern."""
    model._tokenizer = tokenizer
    model.save_pretrained_merged = types.MethodType(_mlx_save_pretrained_merged, model)
    model.save_pretrained_gguf   = types.MethodType(_mlx_save_pretrained_gguf, model)
    model.push_to_hub_merged     = types.MethodType(_mlx_push_to_hub_merged, model)
    model.push_to_hub_gguf       = types.MethodType(_mlx_push_to_hub_gguf, model)
    model.save_lora_adapters     = types.MethodType(_mlx_save_lora_adapters, model)


def _lora_walk_module(
    model,
    lora_config,
    target_modules,
    attr_names,
    *,
    match_all_linear=False,
):
    """Walk a module tree and replace matching Linear/QuantizedLinear with LoRA.

    Used for vision encoders that don't have the flat `.layers` structure
    expected by mlx-lm's `linear_to_lora_layers`.
    """
    import mlx.nn as nn
    try:
        from mlx_lm.tuner.lora import LoRALinear
    except ImportError:
        return

    target_modules = set(target_modules or ())

    for attr_name in attr_names:
        root = getattr(model, attr_name, None)
        if root is None:
            continue

        def _walk(module):
            for name, child in module.named_modules():
                if not match_all_linear and not _lora_name_matches_target(name, target_modules):
                    continue
                if isinstance(child, (nn.Linear, nn.QuantizedLinear)):
                    lora_layer = LoRALinear.from_base(
                        child,
                        r=lora_config["rank"],
                        dropout=lora_config.get("dropout", 0.0),
                        scale=lora_config["scale"],
                    )
                    # Navigate to parent and replace
                    parts = name.split(".")
                    parent = root
                    for p in parts[:-1]:
                        try:
                            parent = parent[int(p)]
                        except (ValueError, TypeError):
                            parent = getattr(parent, p)
                    setattr(parent, parts[-1], lora_layer)

        _walk(root)
        break  # These are alternative names for the same component — stop after first hit


def _resolve_lora_keys(model, target_modules):
    """Resolve user-facing target module names to mlx-lm layer-local keys."""
    import mlx.nn as nn

    target_modules = set(target_modules or ())
    if not target_modules:
        return None

    keys = set()
    roots = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        roots.extend(model.model.layers)
    elif hasattr(model, "layers"):
        roots.extend(model.layers)
    else:
        roots.append(model)

    for root in roots:
        for name, module in root.named_modules():
            if not isinstance(module, (nn.Linear, nn.QuantizedLinear)):
                continue
            if _lora_name_matches_target(name, target_modules):
                keys.add(name)

    return keys


class FastMLXModel:
    """MLX model loader for Apple Silicon.

    Mirrors the unsloth GPU API so notebooks work with minimal changes:
        model, tokenizer = FastLanguageModel.from_pretrained(...)
        model = FastLanguageModel.get_peft_model(model, r=16)

    Pass any HuggingFace model name directly — mlx-lm handles loading:
        "mlx-community/Llama-3.2-1B-Instruct-4bit"   (pre-quantized MLX)
        "mlx-community/Llama-3.2-1B-Instruct-8bit"   (8-bit MLX)
        "meta-llama/Llama-3.2-1B-Instruct"            (full precision)
        "Qwen/Qwen2.5-7B-Instruct"                    (any HF model)
    """

    @staticmethod
    def from_pretrained(
        model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        load_in_8bit=False,
        load_in_16bit=False,
        load_in_fp8=False,
        full_finetuning=False,
        token=None,
        trust_remote_code=False,
        text_only=None,
        patch_mode="patched",
        revision=None,
        random_state=3407,
        **kwargs,  # Accept and ignore GPU-only kwargs
    ):
        """Load a model via mlx-lm (text) or mlx-vlm (vision) on Apple Silicon.

        Args:
            model_name: Any HuggingFace repo name.
            max_seq_length: Maximum sequence length for training.
            dtype: Target floating-point dtype. ``None`` (default) keeps the
                model's native dtype. Accepts ``"float16"``, ``"bfloat16"``,
                ``"float32"`` or the corresponding ``mx.*`` constants.
                Quantized integer weights are preserved regardless. On M1/M2,
                bf16 is emulated and ~40-70%% slower in prefill — fp16 is
                recommended on those chips.
            load_in_4bit: Accepted for API compat with CUDA unsloth.
            full_finetuning: When True, force-disable runtime quantization
                (``load_in_4bit`` etc.) so the full-precision weights are
                trainable. ``get_peft_model`` becomes a no-op for models
                loaded this way.
            token: HuggingFace token for gated models.
            text_only: Loading mode:
                None  — auto-detect from config (default)
                True  — force text-only via mlx-lm
                False — force VLM via mlx-vlm
        """
        if full_finetuning and (load_in_4bit or load_in_8bit or load_in_fp8):
            print(
                "Unsloth: full_finetuning=True — disabling quantized loads "
                "(quantized weights cannot be trained directly)."
            )
            load_in_4bit = False
            load_in_8bit = False
            load_in_fp8 = False
            load_in_16bit = True
        target_dtype = None
        if dtype is not None:
            import mlx.core as mx
            if isinstance(dtype, str):
                target_dtype = getattr(mx, dtype, None)
            elif dtype in (mx.float16, mx.bfloat16, mx.float32):
                target_dtype = dtype
            if target_dtype not in (mx.float16, mx.bfloat16, mx.float32):
                raise ValueError(
                    f"Unsloth: Unsupported dtype {dtype!r}. "
                    f"Use 'float16', 'bfloat16', or 'float32'."
                )
            chip = mx.device_info().get("device_name", "") or ""
            if target_dtype == mx.bfloat16 and chip.startswith(("Apple M1", "Apple M2")):
                warnings.warn(
                    f"Unsloth: {chip} lacks native bf16 GPU support — "
                    f"bf16 will be emulated (~40-70%% slower prefill). "
                    f"Pass dtype='float16' on M1/M2.",
                    stacklevel=2,
                )
        try:
            from mlx_lm import load as mlx_load
            from mlx_lm.utils import _download
        except ImportError:
            raise ImportError(
                "Unsloth: mlx-lm is required for Apple Silicon. "
                "Install via: pip install unsloth-zoo[mlx]"
            )

        chat_template = kwargs.pop("chat_template", None)
        patch_mode = normalize_mlx_patch_mode(kwargs.pop("patch_mode", patch_mode))
        q_bits = kwargs.pop("q_bits", None)
        q_group_size = kwargs.pop("q_group_size", None)
        q_mode = kwargs.pop("q_mode", None)
        mlx_quantization_config = kwargs.pop("mlx_quantization_config", None)
        quantization_config = kwargs.pop("quantization_config", None)
        caller_supplied_quant_config = (
            mlx_quantization_config is not None
            or quantization_config is not None
            or q_bits is not None
            or q_group_size is not None
            or q_mode is not None
            or load_in_8bit
            or load_in_fp8
        )
        quant_predicate = kwargs.pop("quant_predicate", None)
        quantize_modules = kwargs.pop(
            "quantize_modules",
            kwargs.pop("modules_to_quantize", kwargs.pop("mlx_quantize_modules", None)),
        )
        force_requantize = bool(kwargs.pop("force_requantize", False))
        quantization_spec = _resolve_mlx_quantization_spec(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            load_in_16bit=load_in_16bit,
            load_in_fp8=load_in_fp8,
            full_finetuning=full_finetuning,
            q_bits=q_bits,
            q_group_size=q_group_size,
            q_mode=q_mode,
            mlx_quantization_config=mlx_quantization_config,
            quantization_config=quantization_config,
            quant_predicate=quant_predicate,
            quantize_modules=quantize_modules,
            force_requantize=force_requantize,
        )

        # Seed mlx random state so any randomness during model construction
        # (e.g. layer init for runtime-quantized models) is reproducible.
        try:
            import mlx.core as mx
            mx.random.seed(int(random_state))
        except Exception:
            pass

        # Step 1: Download config to decide loading path
        try:
            local_path = str(_download(model_name, revision=revision))
            config_path = local_path + "/config.json"
            with open(config_path, "r") as f:
                config_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            config_data = {}
            local_path = None

        adapter_cfg_path = os.path.join(local_path, "adapter_config.json") if local_path else None
        if adapter_cfg_path and os.path.exists(adapter_cfg_path):
            try:
                with open(adapter_cfg_path, "r") as f:
                    adapter_cfg = json.load(f)
                base_model_id = adapter_cfg.get("base_model_name_or_path", "")
                if base_model_id:
                    print(f"Unsloth: Detected LoRA adapter, loading base model '{base_model_id}'...")
                    adapter_quant_policy = adapter_cfg.get("base_quantization_policy") or {}
                    adapter_quant_map = (
                        adapter_cfg.get("base_resolved_quantization_map")
                        or adapter_cfg.get("base_quantization_map")
                    )
                    adapter_base_revision = _adapter_base_revision(adapter_cfg)
                    adapter_requires_runtime_quant = _adapter_needs_runtime_quantization(
                        adapter_cfg,
                        adapter_quant_policy,
                    )
                    adapter_has_quant_metadata = (
                        adapter_quant_policy.get("enabled")
                        or adapter_quant_map
                        or adapter_cfg.get("base_quantization_config") is not None
                    )
                    if adapter_has_quant_metadata:
                        adapter_mlx_quant_config = None
                        if adapter_requires_runtime_quant and adapter_quant_map:
                            adapter_mlx_quant_config = _quant_config_from_resolved_map(
                                adapter_quant_map
                            )
                        if adapter_requires_runtime_quant and adapter_quant_policy.get("has_callable_predicate") and adapter_mlx_quant_config is None:
                            raise ValueError(
                                "Unsloth: This adapter was saved after loading with a "
                                "custom quant_predicate and its resolved quantization "
                                "map cannot be replayed exactly. Reload the matching "
                                "base model with the same quant_predicate and apply the "
                                "adapter manually, or export a standalone merged model."
                            )
                        if full_finetuning or load_in_16bit:
                            raise ValueError(
                                "Unsloth: This adapter was saved against a quantized "
                                "base model. load_in_16bit=True/full_finetuning=True "
                                "would change the adapter base weights; load without "
                                "those overrides or retrain the adapter."
                            )
                        if adapter_requires_runtime_quant and adapter_mlx_quant_config is None:
                            adapter_mlx_quant_config = {
                                "bits": adapter_quant_policy.get("bits"),
                                "group_size": adapter_quant_policy.get("group_size"),
                                "mode": adapter_quant_policy.get("mode"),
                                "quantize_modules": adapter_quant_policy.get("quantize_modules"),
                            }
                        if caller_supplied_quant_config:
                            actual_quant_config = _adapter_actual_quant_config(
                                adapter_cfg, adapter_quant_map
                            )
                            expected_quantize_modules = None
                            if actual_quant_config is not None:
                                expected_quantize_modules = actual_quant_config.get("quantize_modules")
                            elif adapter_mlx_quant_config is not None:
                                expected_quantize_modules = adapter_mlx_quant_config.get("quantize_modules")
                            elif adapter_quant_policy.get("quantize_modules") is not None:
                                expected_quantize_modules = adapter_quant_policy.get("quantize_modules")
                            requested_adapter_config = {
                                "bits": quantization_spec.bits,
                                "group_size": quantization_spec.group_size,
                                "mode": quantization_spec.mode,
                                "quantize_modules": quantization_spec.quantize_modules,
                            }
                            expected_source = (
                                actual_quant_config
                                or adapter_mlx_quant_config
                                or adapter_quant_policy
                            )
                            expected_adapter_config = {
                                "bits": expected_source.get("bits"),
                                "group_size": expected_source.get("group_size"),
                                "mode": expected_source.get("mode"),
                                "quantize_modules": (
                                    tuple(expected_quantize_modules)
                                    if expected_quantize_modules is not None
                                    else None
                                ),
                            }
                            if requested_adapter_config != expected_adapter_config:
                                raise ValueError(
                                    "Unsloth: This adapter was saved with base "
                                    f"quantization {expected_adapter_config!r}, but "
                                    f"the load request resolved to {requested_adapter_config!r}. "
                                    "Use the saved base quantization policy or retrain "
                                    "the adapter for the requested base quantization."
                                )
                        model, tokenizer = FastMLXModel.from_pretrained(
                            base_model_id,
                            max_seq_length=max_seq_length,
                            dtype=dtype,
                            load_in_4bit=False,
                            load_in_8bit=False,
                            load_in_16bit=False,
                            load_in_fp8=False,
                            full_finetuning=full_finetuning,
                            token=token,
                            trust_remote_code=trust_remote_code,
                            text_only=text_only,
                            patch_mode=patch_mode,
                            revision=adapter_base_revision,
                            random_state=random_state,
                            **(
                                {"mlx_quantization_config": adapter_mlx_quant_config}
                                if adapter_mlx_quant_config is not None
                                else {}
                            ),
                        )
                        _validate_mlx_adapter_base(model, adapter_cfg)
                        from mlx_lm.tuner.utils import load_adapters
                        model = load_adapters(model, local_path)
                        loaded_model_config = getattr(model, "_config", None)
                    else:
                        model, tokenizer = mlx_load(
                            base_model_id,
                            adapter_path=local_path,
                            tokenizer_config={"token": token} if token else None,
                            revision=adapter_base_revision,
                        )
                        loaded_model_config = None
                    is_vlm_model = bool(getattr(model, "_is_vlm_model", False))
                    processor = getattr(model, "_processor", None)

                    base_local = str(_download(base_model_id, revision=adapter_base_revision))
                    base_config_path = os.path.join(base_local, "config.json")
                    if os.path.exists(base_config_path):
                        with open(base_config_path, "r") as f:
                            config_data = json.load(f)
                    if loaded_model_config is not None:
                        model._config = loaded_model_config
                    else:
                        model._config = config_data
                    model._hf_repo = base_model_id
                    model._src_path = base_local
                    model._unsloth_base_revision = adapter_base_revision
                    model._unsloth_base_commit_hash = (
                        adapter_cfg.get("base_model_commit_hash")
                        or _infer_snapshot_commit(base_local)
                    )
                    model._is_vlm_model = is_vlm_model
                    if processor is not None:
                        model._processor = processor
                    model.max_seq_length = max_seq_length
                    model._unsloth_full_finetuning = bool(full_finetuning)
                    if adapter_quant_policy:
                        model._unsloth_quantization_policy = adapter_quant_policy
                        model._unsloth_quantization_config = adapter_cfg.get(
                            "base_quantization_config"
                        )
                        model._unsloth_quantized_source = adapter_cfg.get(
                            "base_quantized_source"
                        )
                    _patch_mlx_saving(model, tokenizer)
                    return model, tokenizer
            except Exception as e:
                if isinstance(e, ValueError) and "Unsloth:" in str(e):
                    raise
                print(f"Unsloth: LoRA adapter detection failed ({e}), falling back to standard load.")

        # Step 2: Check unsloth custom loader registry
        model_type = config_data.get("model_type", "")
        try:
            from unsloth.models.mlx import get_unsloth_loader
            custom_loader = get_unsloth_loader(model_type)
        except (ImportError, AttributeError, NotImplementedError):
            custom_loader = None

        if custom_loader is not None:
            model, tokenizer_or_processor = custom_loader(
                model_name, config_data, max_seq_length=max_seq_length, token=token
            )
            if text_only is False or _is_vlm(config_data):
                from .mlx_utils import normalize_vlm_processor_chat_template

                tokenizer_or_processor = normalize_vlm_processor_chat_template(
                    tokenizer_or_processor,
                    chat_template=chat_template,
                    model_name=model_name,
                    model_type=model_type,
                    strict=False,
                )
                model._is_vlm_model = True
                model._processor = tokenizer_or_processor
                _patch_mixed_precision_set_dtype(model)
            elif chat_template is not None:
                from .mlx_utils import normalize_mlx_chat_template

                tokenizer_or_processor = normalize_mlx_chat_template(
                    tokenizer_or_processor,
                    chat_template=chat_template,
                    model_name=model_name,
                    model_type=model_type,
                    is_vlm=False,
                    strict=False,
                )
            model._config = config_data
            model._hf_repo = model_name
            model._src_path = local_path
            model._unsloth_base_revision = revision
            model._unsloth_base_commit_hash = _infer_snapshot_commit(local_path)
            model.max_seq_length = max_seq_length
            model._unsloth_patch_mode = patch_mode
            model._unsloth_full_finetuning = bool(full_finetuning)
            _patch_mlx_saving(model, tokenizer_or_processor)
            return model, tokenizer_or_processor

        # Step 3: Route based on text_only
        is_vlm = False
        force_vlm_text_path = bool(text_only is True and _prefer_vlm_loader_for_text(config_data, model_type))

        if text_only is True and not force_vlm_text_path:
            is_vlm = False
        elif text_only is False:
            is_vlm = True
        else:
            is_vlm = _is_vlm(config_data)

        extra_kwargs = {}
        if token:
            extra_kwargs["token"] = token
        if trust_remote_code:
            extra_kwargs["trust_remote_code"] = True

        if is_vlm:
            # VLM path via mlx-vlm
            try:
                from mlx_vlm import load as vlm_load
            except ImportError:
                raise ImportError(
                    "Unsloth: mlx-vlm is required for Vision Language Models. "
                    "Install via: pip install mlx-vlm\n"
                    "Or pass text_only=True to load as text-only via mlx-lm."
                )

            if text_only is False and not _is_vlm(config_data):
                warnings.warn(
                    f"text_only=False but '{model_name}' does not appear to be a VLM. "
                    f"Attempting mlx_vlm.load() anyway — this may fail.",
                    stacklevel=2,
                )

            if patch_mode == "patched":
                install_mlx_compile_patches()
            _ensure_vlm_prompt_utils_patched()

            quant_state = _ensure_quantization_compatible(
                config_data, quantization_spec, model_name,
            )
            want_runtime_quant = quantization_spec.enabled and quant_state != "compatible"

            if quantization_spec.enabled and quant_state == "compatible":
                warnings.warn(
                    f"Unsloth: '{model_name}' is already quantized — "
                    "using existing compatible MLX quantization.",
                    stacklevel=2,
                )

            if want_runtime_quant:
                _ensure_vlm_load_model_patched()
                print(f"Unsloth: Loading {model_name} via mlx-vlm (VLM, "
                      f"runtime {quantization_spec.bits}-bit {quantization_spec.mode} quantization)...")
                model, processor = vlm_load(
                    model_name,
                    quantization_spec=quantization_spec,
                    quant_predicate=quant_predicate,
                    lazy=True,
                    revision=revision,
                    **extra_kwargs,
                )
            else:
                print(f"Unsloth: Loading {model_name} via mlx-vlm (VLM)...")
                # Lazy-load when we need to convert dtype so weights are
                # only materialized once in the target dtype.
                vlm_kwargs = dict(extra_kwargs)
                vlm_kwargs["revision"] = revision
                if target_dtype is not None:
                    vlm_kwargs["lazy"] = True
                model, processor = _load_mlx_vlm_with_extra_weight_filter(
                    model_name,
                    model_type,
                    vlm_load,
                    vlm_kwargs,
                )

            if target_dtype is not None:
                _convert_mlx_dtype(model, target_dtype)
            elif want_runtime_quant:
                import mlx.core as mx
                mx.eval(model.parameters())

            from .mlx_utils import (
                normalize_mlx_chat_template,
                normalize_vlm_processor_chat_template,
            )

            processor = normalize_vlm_processor_chat_template(
                processor,
                chat_template=chat_template,
                model_name=model_name,
                model_type=model_type,
                strict=False,
            )
            if force_vlm_text_path:
                print(
                    "Unsloth: text_only=True requested for a multimodal wrapper; "
                    "keeping the model on the mlx-vlm path and returning its tokenizer."
                )
                model._unsloth_text_only_vlm = True
            model._is_vlm_model = True
            model._processor = processor
            _fix_gemma4_kv_sharing(model)

            model._config = getattr(model, "_config", config_data)
            model._hf_repo = model_name
            model._src_path = local_path
            model._unsloth_base_revision = revision
            model._unsloth_base_commit_hash = _infer_snapshot_commit(local_path)
            model.max_seq_length = max_seq_length
            model._unsloth_patch_mode = patch_mode
            model._unsloth_full_finetuning = bool(full_finetuning)
            if quant_state == "compatible":
                model._unsloth_quantization_config = _get_existing_mlx_quantization(config_data)
                model._unsloth_quantization_policy = quantization_spec.to_metadata()
                model._unsloth_quantized_source = "mlx_config"
            model._unsloth_compile_trait_report = get_compile_trait_report(model)
            model._unsloth_compile_qualification = get_compile_qualification(model)
            model._unsloth_compile_backend_qualifications = get_backend_compile_qualifications(model)
            model._unsloth_compile_trace = trace_compile_application(model)
            model._unsloth_compile_explain = explain_compile_support(model)
            _patch_mixed_precision_set_dtype(model)

            public_target = processor
            if force_vlm_text_path:
                public_target = normalize_mlx_chat_template(
                    getattr(processor, "tokenizer", processor),
                    chat_template=chat_template,
                    model_name=model_name,
                    model_type=model_type,
                    is_vlm=False,
                    strict=False,
                )
                model._tokenizer = public_target

            _patch_mlx_saving(model, public_target)
            return model, public_target
        else:
            # Text path via mlx-lm (original behavior)
            quant_state = _ensure_quantization_compatible(
                config_data, quantization_spec, model_name,
            )
            want_runtime_quant = quantization_spec.enabled and quant_state != "compatible"

            if want_runtime_quant:
                print(
                    f"Unsloth: Loading {model_name} via mlx-lm "
                    f"(runtime {quantization_spec.bits}-bit {quantization_spec.mode} quantization)..."
                )
            else:
                print(f"Unsloth: Loading {model_name} via mlx-lm...")
                if quantization_spec.enabled and quant_state == "compatible":
                    warnings.warn(
                        f"Unsloth: '{model_name}' is already quantized — "
                        "using existing compatible MLX quantization.",
                        stacklevel=2,
                    )
            _ensure_safe_text_wrapper_sanitize(model_type)

            mlx_load_kwargs = dict(
                tokenizer_config=extra_kwargs if extra_kwargs else None,
                return_config=True,
                revision=revision,
            )
            if target_dtype is not None or want_runtime_quant:
                mlx_load_kwargs["lazy"] = True
            model, tokenizer, config = _load_mlx_lm_with_strict_fallback(
                model_name,
                model_type,
                mlx_load,
                mlx_load_kwargs,
            )

            if want_runtime_quant:
                model, config = _apply_mlx_quantization(
                    model,
                    config,
                    quantization_spec,
                    is_vlm=False,
                    user_predicate=quant_predicate,
                )
                print(
                    f"Unsloth: Quantized text model to "
                    f"{quantization_spec.bits}-bit {quantization_spec.mode}."
                )

            if target_dtype is not None:
                _convert_mlx_dtype(model, target_dtype)
            elif want_runtime_quant:
                import mlx.core as mx
                mx.eval(model.parameters())
            from .mlx_utils import normalize_mlx_chat_template

            tokenizer = normalize_mlx_chat_template(
                tokenizer,
                chat_template=chat_template,
                model_name=model_name,
                model_type=model_type,
                is_vlm=False,
                strict=False,
            )
            model._is_vlm_model = False

            model._config = config
            model._hf_repo = model_name
            model._src_path = local_path
            model._unsloth_base_revision = revision
            model._unsloth_base_commit_hash = _infer_snapshot_commit(local_path)
            model.max_seq_length = max_seq_length
            model._unsloth_patch_mode = patch_mode
            model._unsloth_full_finetuning = bool(full_finetuning)
            if quant_state == "compatible":
                model._unsloth_quantization_config = _get_existing_mlx_quantization(config_data)
                model._unsloth_quantization_policy = quantization_spec.to_metadata()
                model._unsloth_quantized_source = "mlx_config"
            _patch_mixed_precision_set_dtype(model)

            _patch_mlx_saving(model, tokenizer)
            return model, tokenizer

    @staticmethod
    def get_peft_model(
        model,
        r=16,
        target_modules=None,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="mlx",
        random_state=3407,
        max_seq_length=2048,
        train_vision=False,
        train_projector=False,
        **kwargs,  # Accept and ignore GPU-only kwargs
    ):
        """Apply LoRA via mlx-lm on Apple Silicon.

        For VLMs, applies LoRA to the language model and optionally to the
        vision tower (train_vision=True) and projector (train_projector=True).

        When the model was loaded with ``full_finetuning=True``, this is a
        no-op: the full-precision parameters stay trainable and the model
        is returned as-is.
        """
        if getattr(model, "_unsloth_full_finetuning", False):
            print(
                "Unsloth: full_finetuning=True — skipping LoRA, training "
                "all model parameters directly."
            )
            return model
        try:
            from mlx_lm.tuner.utils import linear_to_lora_layers
        except ImportError:
            raise ImportError(
                "Unsloth: mlx-lm is required for LoRA on Apple Silicon. "
                "Install via: pip install unsloth-zoo[mlx]"
            )

        # Seed mlx random state so LoRA matrix init (lora_b is zero, lora_a
        # is random Gaussian) is reproducible across runs.
        try:
            import mlx.core as mx
            mx.random.seed(int(random_state))
        except Exception:
            pass

        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

        # "all-linear" is a PEFT special keyword meaning all linear layers.
        # _resolve_lora_keys handles None target_modules → discovers all.
        if target_modules == ["all-linear"] or target_modules == "all-linear":
            target_modules = None

        lora_config = {
            "rank": r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "scale": lora_alpha / r,
        }

        is_vlm = getattr(model, "_is_vlm_model", False)

        if is_vlm:
            # VLM path: freeze everything, then apply LoRA selectively
            _fix_missing_no_grad(model)
            _fix_gemma4_kv_sharing(model)
            model.freeze()

            # Apply LoRA to the language model (filtered by target_modules)
            lm = model.language_model
            num_layers = 0
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                num_layers = len(lm.model.layers)
            language_lora_keys = _resolve_lora_keys(lm, target_modules)
            linear_to_lora_layers(
                lm,
                num_layers=num_layers,
                config={**lora_config, "keys": language_lora_keys},
                use_dora=False,
            )

            # Optionally apply LoRA to vision tower
            if train_vision:
                _lora_walk_module(model, lora_config, target_modules,
                                  attr_names=("vision_tower", "vision_model",
                                              "vision_encoder"))

            # Optionally train the multimodal projector / connector. Prefer
            # projector LoRA over unfreezing raw weights because many MLX VLM
            # checkpoints expose projector layers as QuantizedLinear, and MLX
            # does not backprop into quantized weights directly.
            if train_projector:
                _lora_walk_module(
                    model,
                    lora_config,
                    target_modules=(),
                    attr_names=(
                        "multi_modal_projector",
                        "mm_projector",
                        "connector",
                        "aligner",
                        "embed_vision",
                    ),
                    match_all_linear=True,
                )

            # Unfreeze all LoRA params across the entire tree
            model.unfreeze(keys=["lora_a", "lora_b"], strict=False)
        else:
            # Text-only path — filter by target_modules
            # Fix missing _no_grad on modules that use __new__ without __init__
            # (e.g. Gemma4 AudioRelativePositionEmbedding loaded via VLM path)
            _fix_missing_no_grad(model)

            num_layers = 0
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                num_layers = len(model.model.layers)
            language_lora_keys = _resolve_lora_keys(model, target_modules)
            linear_to_lora_layers(
                model,
                num_layers=num_layers,
                config={**lora_config, "keys": language_lora_keys},
                use_dora=False,
            )

            model.freeze()
            model.unfreeze(keys=["lora_a", "lora_b"], strict=False)

        # Apply gradient checkpointing if requested
        # "mlx" (default) or True → apply; False or "none" → skip
        if isinstance(use_gradient_checkpointing, str):
            _apply_gc = use_gradient_checkpointing.lower() not in ("false", "none", "")
        else:
            _apply_gc = bool(use_gradient_checkpointing)

        if _apply_gc:
            from .mlx_utils import apply_gradient_checkpointing
            apply_gradient_checkpointing(model)

        import mlx.utils
        trainable = sum(v.size for _, v in mlx.utils.tree_flatten(model.trainable_parameters()))
        total = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
        pct = 100.0 * trainable / total if total > 0 else 0
        print(
            f"Unsloth: LoRA applied — {trainable:,} trainable params "
            f"({pct:.2f}% of {total:,} total)"
        )
        return model


# Aliases for backward compat
FastLanguageModel = FastMLXModel
FastModel = FastMLXModel

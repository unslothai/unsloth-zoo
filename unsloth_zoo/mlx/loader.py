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
import math
import os
import sys
import types
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from fnmatch import fnmatch

from .compile import (
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


import threading
_HF_TOKEN_ENV_LOCK = threading.RLock()
_LOAD_WEIGHTS_PATCH_LOCK = threading.RLock()


@contextmanager
def _temporary_hf_token_env(token):
    """Expose an explicit HF token to libraries that only read auth from env.

    mlx-lm / mlx-vlm do not consistently thread a ``token=`` argument through
    every internal download path. Keep explicit token loads local to this call
    without mutating persistent Hugging Face login state.
    """
    if not token or not isinstance(token, str):
        yield
        return

    env_names = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    with _HF_TOKEN_ENV_LOCK:
        previous = {name: os.environ.get(name) for name in env_names}
        try:
            for name in env_names:
                os.environ[name] = token
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _is_force_float32_arch(model_type: str) -> bool:
    """Case-insensitive lookup of ``model_type`` in
    ``unsloth_zoo.FORCE_FLOAT32``. Strips ``-``/``_`` and treats a
    trailing comma on a list entry as an exact-match marker."""
    if not model_type:
        return False
    from ..model_lists import FORCE_FLOAT32
    def _norm(s: str) -> str:
        return s.lower().replace("-", "").replace("_", "")
    norm_input = _norm(model_type)
    for entry in FORCE_FLOAT32:
        e = entry.lower()
        if e.endswith(","):
            e = e[:-1]
        if _norm(e) == norm_input:
            return True
    return False


def _convert_mlx_dtype(model, target_dtype, model_type: str = "") -> None:
    """Cast floating-point params to target_dtype (preserves quantized ints)
    while honoring the model's optional path-based ``cast_predicate``.

    Warns on bfloat16 -> float16 for architectures in
    ``unsloth_zoo.FORCE_FLOAT32`` (Gemma3 family, gpt_oss, Qwen3.5) where
    fp16's narrower range silently NaN/Infs at training time.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map_with_path
    from ..model_lists import FORCE_FLOAT32
    cast_pred = getattr(model, "cast_predicate", lambda _: True)

    needs_cast = False
    has_bf16 = False
    for k, v in tree_flatten(model.parameters()):
        if cast_pred(k) and mx.issubdtype(v.dtype, mx.floating) and v.dtype != target_dtype:
            needs_cast = True
            if v.dtype == mx.bfloat16:
                has_bf16 = True
    if not needs_cast:
        return

    if has_bf16 and target_dtype == mx.float16 and _is_force_float32_arch(model_type):
        warnings.warn(
            f"Unsloth: downcasting bfloat16 -> float16 on {model_type!r}, "
            "which is known to NaN/Inf in fp16. Pass dtype=None to keep "
            "native bf16, or dtype='float32' for full precision.",
            stacklevel=2,
        )

    model.update(tree_map_with_path(
        lambda k, v: v.astype(target_dtype)
        if cast_pred(k) and mx.issubdtype(v.dtype, mx.floating) else v,
        model.parameters(),
    ))
    mx.eval(model.parameters())


def _seed_mlx_random_state(random_state):
    try:
        seed = int(random_state)
    except (TypeError, ValueError):
        raise TypeError("Unsloth: random_state must be an integer.")

    import mlx.core as mx
    mx.random.seed(seed)


def _collect_all_linear_target_names(model):
    """Return suffix names of every Linear / QuantizedLinear leaf in `model`.

    Mirrors PEFT's ``target_modules="all-linear"``: discover every
    ``nn.Linear`` (and quantized variant) in the live module tree, and
    return the leaf-suffix name of each. Numeric tokens (list indices like
    ``experts.0``) are skipped — we want the semantic name (e.g. ``w1``,
    ``router``, ``q_proj``, ``qkv_proj``, ``lm_head``).

    The result is the set of names against which mlx-lm's
    ``linear_to_lora_layers`` matches, so applying LoRA to this set covers
    every Linear in the model — including fused-QKV archs, MoE
    routers/experts, multimodal projector, and untied LM heads.
    """
    import mlx.nn as nn
    linear_types = (nn.Linear, nn.QuantizedLinear)
    names = set()
    try:
        for path, mod in model.named_modules():
            if not isinstance(mod, linear_types):
                continue
            for token in reversed(str(path).split(".")):
                if token and not token.isdigit():
                    names.add(token)
                    break
    except Exception:
        # Defensive: never let target-module discovery raise during LoRA
        # setup. Caller will fall back to the canonical 7-name default.
        return []
    return sorted(names)


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
    hf_token=None,
):
    """Load text models through mlx-lm, retrying strict=False for known safe mismatches.

    Some upstream checkpoints contain extra tensors that the current mlx-lm
    implementation intentionally does not instantiate. mlx-lm's public load()
    does not expose strict=False, so use the internal loader only for registered
    mismatch signatures.
    """
    # why: mlx-lm 0.22.0 load() rejects return_config / revision; bypass it
    # so signature drift between mlx-lm releases doesn't break loading.
    from mlx_lm.utils import _download, load_model, load_tokenizer

    tokenizer_config = mlx_load_kwargs.get("tokenizer_config") or {}
    model_config = mlx_load_kwargs.get("model_config") or {}
    lazy = mlx_load_kwargs.get("lazy", False)
    revision = mlx_load_kwargs.get("revision")
    want_config = mlx_load_kwargs.get("return_config", False)

    with _temporary_hf_token_env(hf_token):
        model_path = _download(model_name, revision=revision)

    try:
        model, config = load_model(
            model_path,
            lazy=lazy,
            model_config=model_config,
        )
    except ValueError as error:
        message = str(error)
        rule = _KNOWN_MLX_LM_STRICT_FALLBACKS.get(model_type)
        if rule is None or not _message_matches_known_fallback(message, rule):
            raise
        print(rule["notice"])
        model, config = load_model(
            model_path,
            lazy=lazy,
            strict=False,
            model_config=model_config,
        )

    tokenizer = load_tokenizer(
        model_path,
        tokenizer_config,
        eos_token_ids=config.get("eos_token_id", None),
    )
    if want_config:
        return model, tokenizer, config
    return model, tokenizer


def _load_mlx_vlm_with_extra_weight_filter(
    model_name,
    model_type,
    vlm_load,
    vlm_kwargs,
    hf_token=None,
):
    """Load VLMs, filtering known extra checkpoint tensors on retry.

    Some upstream VLM checkpoints include tensors for modules that the current
    mlx-vlm class does not instantiate. mlx-vlm does not expose strict=False
    through load(), so retry with a temporary load_weights shim only for
    registered mismatch signatures and exact allow-listed keys.
    """
    try:
        with _temporary_hf_token_env(hf_token):
            return vlm_load(model_name, **vlm_kwargs)
    except ValueError as error:
        message = str(error)
        rule = _KNOWN_VLM_EXTRA_WEIGHT_FILTERS.get(model_type)
        if rule is None or not _message_matches_known_fallback(message, rule):
            raise

        print(rule["notice"])
        import mlx.nn as nn

        allowed_extra = rule["allowed_extra"]

        # why: nn.Module.load_weights is patched process-globally; lock so
        # a concurrent load doesn't see the filtered version.
        with _LOAD_WEIGHTS_PATCH_LOCK:
            original_load_weights = nn.Module.load_weights

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
                with _temporary_hf_token_env(hf_token):
                    return vlm_load(model_name, **vlm_kwargs)
            finally:
                nn.Module.load_weights = original_load_weights


def _read_json_file(path):
    """Read a JSON object, returning an empty dict for missing/bad sidecars."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}


def _resolve_mlx_vlm_processor_class(model_type, processor_class_name):
    """Resolve a custom mlx-vlm or Transformers processor class by name."""
    if not processor_class_name:
        return None

    module_model_type = (model_type or "").replace("-", "_")
    module_candidates = (
        f"mlx_vlm.models.{module_model_type}.processing",
        f"mlx_vlm.models.{module_model_type}.processing_{module_model_type}",
    )
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        processor_class = getattr(module, processor_class_name, None)
        if processor_class is not None:
            return processor_class

    try:
        import transformers
        return getattr(transformers, processor_class_name, None)
    except Exception:
        return None


def _build_vlm_image_processor_from_config(model_path, processor_config, preprocessor_config):
    """Recreate the image processor from saved processor sidecar configs."""
    image_config = processor_config.get("image_processor")
    if not isinstance(image_config, dict):
        image_config = preprocessor_config
    if not isinstance(image_config, dict):
        image_config = {}

    image_processor_type = (
        image_config.get("image_processor_type")
        or preprocessor_config.get("image_processor_type")
    )
    image_kwargs = dict(image_config)
    image_kwargs.pop("image_processor_type", None)
    image_kwargs.pop("processor_class", None)

    if image_processor_type:
        try:
            import transformers
            image_processor_class = getattr(transformers, image_processor_type, None)
            if image_processor_class is not None:
                return image_processor_class(**image_kwargs)
        except Exception:
            pass

    try:
        from transformers import AutoImageProcessor
        return AutoImageProcessor.from_pretrained(model_path)
    except Exception:
        return None


def _repair_degraded_vlm_processor(
    processor,
    model_path,
    model_type,
    *,
    token=None,
    trust_remote_code=False,
):
    """Rebuild VLM processors when mlx-vlm falls back to tokenizer-only.

    mlx-vlm registers several custom processors through an AutoProcessor patch.
    If the custom processor's image processor cannot be constructed through
    AutoImageProcessor, the patch falls back to the prior tokenizer-only loader.
    Rebuild from the source processor configs so downstream saves preserve real
    multimodal processor metadata.
    """
    if processor is None or getattr(processor, "image_processor", None) is not None:
        return processor

    if not model_path or not os.path.isdir(str(model_path)):
        return processor

    processor_config = _read_json_file(
        os.path.join(str(model_path), "processor_config.json")
    )
    preprocessor_config = _read_json_file(
        os.path.join(str(model_path), "preprocessor_config.json")
    )
    processor_class_name = (
        processor_config.get("processor_class")
        or preprocessor_config.get("processor_class")
    )
    processor_class = _resolve_mlx_vlm_processor_class(
        model_type, processor_class_name,
    )
    if processor_class is None:
        return processor

    image_processor = _build_vlm_image_processor_from_config(
        model_path, processor_config, preprocessor_config,
    )
    if image_processor is None:
        return processor

    tokenizer = getattr(processor, "tokenizer", None) or processor
    if tokenizer is None or not hasattr(tokenizer, "save_pretrained"):
        try:
            from transformers import AutoTokenizer
            tokenizer_kwargs = {"trust_remote_code": trust_remote_code}
            if token:
                tokenizer_kwargs["token"] = token
            tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
        except Exception:
            return processor

    chat_template = getattr(processor, "chat_template", None)
    if chat_template is not None and getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = chat_template

    try:
        repaired = processor_class(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )
    except TypeError:
        try:
            repaired = processor_class(
                image_processor=image_processor,
                tokenizer=tokenizer,
            )
        except Exception:
            return processor
    except Exception:
        return processor

    if chat_template is not None and getattr(repaired, "chat_template", None) is None:
        repaired.chat_template = chat_template
    return repaired


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

    def patched_call(self, inputs=None, inputs_embeds=None, mask=None,
                     cache=None, per_layer_inputs=None, **kwargs):
        if cache is None:
            # why: read n_stores from the live instance so a second Gemma4
            # variant with a different first_kv_shared_layer_idx isn't given
            # the first model's count.
            n_stores = getattr(self, "first_kv_shared_layer_idx", None)
            if n_stores is None:
                n_stores = 0
            cache = [_TrainingKVStore() for _ in range(int(n_stores))]
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


def _resolve_full_finetune_dtype(target_dtype, float32_mixed_precision, mx):
    if target_dtype == mx.bfloat16:
        if type(float32_mixed_precision) is not bool:
            # Match the Torch post-patch default: bf16 full finetuning stays
            # bf16 unless float32_mixed_precision=True is explicitly requested.
            float32_mixed_precision = False
        if float32_mixed_precision is False:
            return mx.bfloat16, False
    return mx.float32, True


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
# VLM prompt/template compatibility patches
# ---------------------------------------------------------------------------
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


def _normalize_mlx_lora_module_paths(module_paths):
    """Normalize stored module paths into a list of non-empty strings.

    Accepts str, list/tuple/set, dict ({"language": [...], "vision": [...]}),
    and pathlib.Path; iterating a bare string would walk characters and
    never wrap a real LoRA layer.
    """
    if module_paths is None:
        return []
    if isinstance(module_paths, str):
        return [module_paths] if module_paths else []
    if isinstance(module_paths, os.PathLike):
        s = os.fspath(module_paths)
        return [s] if s else []
    if isinstance(module_paths, dict):
        out = []
        for value in module_paths.values():
            out.extend(_normalize_mlx_lora_module_paths(value))
        return out
    if isinstance(module_paths, (list, tuple, set)):
        out = []
        for p in module_paths:
            if isinstance(p, str):
                if p:
                    out.append(p)
            elif isinstance(p, os.PathLike):
                s = os.fspath(p)
                if s:
                    out.append(s)
        return out
    return []


def _infer_rank_from_saved_adapter(adapter_weights_file, module_path):
    """Read the rank dimension off the saved LoRA tensor for `module_path`.

    Compatibility shim for legacy direct-save adapters that wrote
    `unsloth_mlx_lora_module_paths` without persisting rank/scale/dropout.
    Returns None when the file/path is absent or imports fail.
    """
    if not adapter_weights_file or not os.path.exists(adapter_weights_file):
        return None
    try:
        from safetensors import safe_open
    except Exception:
        return None
    candidate_suffixes = (
        ".lora_a.weight", ".lora_a",
        ".lora_A.weight", ".lora_A",
        ".lora_embedding_a.weight", ".lora_embedding_a",
    )
    # Suffixes whose 2-D shape is (rank, in_dims): mlx-lm's layer-backed
    # `lora_a.weight` and PEFT-style uppercase `lora_A` / `lora_A.weight`.
    # Raw lowercase `lora_a` keeps the older (in_dims, rank) shape so falls
    # through to the default shape[-1] branch. `.lora_embedding_a*` is NOT
    # in this set: LoRAEmbedding saves A as (num_embeddings, rank).
    _rank_first_2d_suffixes = frozenset({
        ".lora_a.weight",
        ".lora_A",
        ".lora_A.weight",
    })
    try:
        with safe_open(adapter_weights_file, framework="numpy") as _f:
            keys = set(_f.keys())
            for suffix in candidate_suffixes:
                key = f"{module_path}{suffix}"
                if key not in keys:
                    continue
                tensor = _f.get_tensor(key)
                shape = tuple(tensor.shape)
                if not shape:
                    return None
                # MoE/switch: (experts, rank, in_dims) → rank is shape[-2].
                if len(shape) >= 3:
                    return int(shape[-2])
                if suffix in _rank_first_2d_suffixes:
                    return int(shape[0])
                return int(shape[-1])
    except Exception:
        return None
    return None


def _apply_lora_at_paths(model, module_paths, adapter_cfg, adapter_weights_file=None):
    """Recreate LoRA/DoRA layers at saved module paths so vision/projector
    and MoE/embedding LoRA survives reload (mlx-lm's load_adapters only
    rebuilds the language tower's Linear layers).

    Returns the number of new wrappers attached. `adapter_weights_file` is
    a last-resort rank source for legacy adapters that wrote
    `unsloth_mlx_lora_module_paths` without rank/scale/dropout.
    """
    import mlx.nn as nn
    from mlx_lm.tuner.lora import LoRALinear

    module_paths = _normalize_mlx_lora_module_paths(module_paths)
    if not module_paths:
        return 0

    # Lazy import: tolerate older mlx-lm without switch / embedding LoRA.
    try:
        from mlx_lm.tuner.lora import LoRASwitchLinear
    except Exception:
        LoRASwitchLinear = None
    try:
        from mlx_lm.tuner.lora import LoRAEmbedding
    except Exception:
        LoRAEmbedding = None
    try:
        from mlx_lm.models.switch_layers import (
            QuantizedSwitchLinear,
            SwitchLinear,
        )
        switch_types = (SwitchLinear, QuantizedSwitchLinear)
    except Exception:
        switch_types = ()

    embedding_types = tuple(
        t for t in (getattr(nn, "Embedding", None), getattr(nn, "QuantizedEmbedding", None))
        if t is not None
    )

    # DoRA path: refuse silent downgrade to plain LoRA when the saved
    # model had DoRA modules; the per-module loop raises a specific
    # ImportError if it hits one that needs the missing class.
    fine_tune_type = str(adapter_cfg.get("fine_tune_type", "lora")).lower()
    use_dora = fine_tune_type == "dora"
    DoRALinear_cls = DoRAEmbedding_cls = None
    if use_dora:
        try:
            from mlx_lm.tuner.dora import DoRALinear as DoRALinear_cls
        except Exception:
            DoRALinear_cls = None
        try:
            from mlx_lm.tuner.dora import DoRAEmbedding as DoRAEmbedding_cls
        except Exception:
            DoRAEmbedding_cls = None

    # Defer rank validation until the first module actually needs wrapping;
    # a legacy adapter with no rank but paths already wrapped by mlx-lm.
    # load_adapters should not crash here.
    _metadata = {"rank": None, "scale": None, "dropout": None}

    def _ensure_metadata(module_path=None):
        if _metadata["rank"] is not None:
            return
        lora_params = adapter_cfg.get("lora_parameters") or {}
        raw_rank = lora_params.get("rank", adapter_cfg.get("rank"))
        if raw_rank is None and module_path is not None:
            # Legacy direct-save fallback: recover rank from the saved
            # tensor shape; scale/dropout stay at their 1.0 / 0.0 defaults.
            raw_rank = _infer_rank_from_saved_adapter(
                adapter_weights_file, module_path,
            )
        if raw_rank is None:
            raise ValueError(
                "Unsloth MLX: adapter_config.json is missing LoRA rank "
                "metadata and rank could not be inferred from "
                "adapters.safetensors; refusing to recreate adapter "
                "wrappers with placeholder rank=8. Re-save the adapter "
                "with rank/scale/dropout populated."
            )
        # Wrap the int/float conversions in a namespaced ValueError so
        # malformed metadata (e.g. `"rank": "not-an-int"` in a hand-
        # authored or stale config) raises the same "Unsloth MLX:" prefix
        # the outer adapter detection catch preserves. Without this the
        # plain `invalid literal for int()` ValueError can be swallowed
        # by the fallback into a silent standard-load.
        try:
            _metadata["rank"] = int(raw_rank)
            _metadata["scale"] = float(
                lora_params.get("scale", adapter_cfg.get("scale", 1.0))
            )
            _metadata["dropout"] = float(
                lora_params.get("dropout", adapter_cfg.get("dropout", 0.0))
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Unsloth MLX: adapter_config.json has invalid LoRA "
                "rank/scale/dropout metadata; refusing to recreate adapter "
                "wrappers with placeholder defaults. Re-save the adapter "
                "with numeric rank, scale, and dropout values."
            ) from exc

    by_name = dict(model.named_modules())
    linear_types = (nn.Linear, nn.QuantizedLinear)
    attached = 0
    # Track per-path skip reasons for the final warning so the user knows
    # which paths could not be re-wrapped (the saved-vs-live key diff is
    # the actual gate that raises).
    _skipped_paths = []
    for name in module_paths:
        module = by_name.get(name)
        if module is None:
            _skipped_paths.append((name, "module_missing"))
            continue
        # Skip already-wrapped paths so we don't nest LoRALinear(LoRALinear).
        if hasattr(module, "lora_a") and hasattr(module, "lora_b"):
            continue
        lora_cls = None
        if isinstance(module, linear_types):
            if use_dora:
                # Fail loud: silent downgrade to plain LoRA would drop the
                # saved q_proj.m magnitude tensor on strict=False.
                if DoRALinear_cls is None:
                    raise ImportError(
                        "Unsloth MLX: adapter_config.json says "
                        "fine_tune_type='dora' but mlx_lm.tuner.dora."
                        "DoRALinear is unavailable; refusing to silently "
                        "downgrade to plain LoRA. Upgrade mlx-lm."
                    )
                lora_cls = DoRALinear_cls
            else:
                lora_cls = LoRALinear
        elif switch_types and isinstance(module, switch_types):
            # No DoRA on switch layers in mlx-lm; fail loud rather than
            # silently dropping saved switch LoRA tensors.
            if LoRASwitchLinear is None:
                raise ImportError(
                    "Unsloth MLX: adapter_config.json contains a saved "
                    f"switch LoRA path {name!r}, but mlx_lm.tuner.lora."
                    "LoRASwitchLinear is unavailable; refusing to silently "
                    "drop the saved switch LoRA tensors. Upgrade mlx-lm "
                    "or re-save without switch LoRA."
                )
            lora_cls = LoRASwitchLinear
        elif embedding_types and isinstance(module, embedding_types):
            if use_dora:
                if DoRAEmbedding_cls is None:
                    raise ImportError(
                        "Unsloth MLX: adapter_config.json says "
                        "fine_tune_type='dora' but mlx_lm.tuner.dora."
                        "DoRAEmbedding is unavailable; refusing to silently "
                        "downgrade to plain LoRAEmbedding. Upgrade mlx-lm."
                    )
                lora_cls = DoRAEmbedding_cls
            else:
                # Fail loud for the embedding path too: silent lora_cls=None
                # would drop saved embedding LoRA tensors on strict=False.
                if LoRAEmbedding is None:
                    raise ImportError(
                        "Unsloth MLX: adapter_config.json contains a saved "
                        f"embedding LoRA path {name!r}, but mlx_lm.tuner."
                        "lora.LoRAEmbedding is unavailable; refusing to "
                        "silently drop the saved embedding LoRA tensors. "
                        "Upgrade mlx-lm or re-save without embedding LoRA."
                    )
                lora_cls = LoRAEmbedding
        if lora_cls is None:
            _skipped_paths.append(
                (name, f"unhandled_type:{type(module).__name__}"),
            )
            continue
        _ensure_metadata(module_path=name)
        wrapped = _lora_from_base_compat(
            lora_cls, module,
            _metadata["rank"], _metadata["scale"], _metadata["dropout"],
        )
        # Walk numeric path segments (e.g. `vision_tower.merger.layers.0`)
        # via `parent[int(seg)]` first then getattr; same try-int /
        # setattr pattern on the leaf so list-indexed wrappers install.
        parts = name.split(".")
        parent = model
        parent_reachable = True
        for seg in parts[:-1]:
            try:
                parent = parent[int(seg)]
            except (ValueError, TypeError):
                next_parent = getattr(parent, seg, None)
                if next_parent is None:
                    parent_reachable = False
                    break
                parent = next_parent
            except (IndexError, KeyError):
                parent_reachable = False
                break
        if not parent_reachable or parent is None:
            _skipped_paths.append((name, "parent_unreachable"))
            continue
        leaf = parts[-1]
        try:
            parent[int(leaf)] = wrapped
        except (ValueError, TypeError):
            if not hasattr(parent, leaf):
                _skipped_paths.append((name, "parent_unreachable"))
                continue
            setattr(parent, leaf, wrapped)
        except (IndexError, KeyError):
            _skipped_paths.append((name, "parent_unreachable"))
            continue
        attached += 1

    if _skipped_paths:
        _preview = ", ".join(
            f"{name} ({reason})" for name, reason in _skipped_paths[:5]
        )
        if len(_skipped_paths) > 5:
            _preview += f", ... (+{len(_skipped_paths) - 5} more)"
        warnings.warn(
            f"Unsloth MLX: skipped {len(_skipped_paths)} saved auxiliary "
            f"LoRA path(s) during reload: {_preview}. Their saved tensors "
            "will not bind into the live model; the saved-vs-live key diff "
            "may also raise.",
            stacklevel=2,
        )
    return attached


def _eval_mlx_model_after_adapter_reload(model):
    try:
        model.eval()
    except Exception:
        pass
    return model


# Saved-file suffixes used by _warn_missing_adapter_keys. `.m` / `.m.weight`
# cover raw and layer-wrapped DoRA magnitudes.
_ADAPTER_LORA_KEY_SUFFIXES = (
    ".lora_a", ".lora_b",
    ".lora_a.weight", ".lora_b.weight",
    ".lora_A", ".lora_B",
    ".lora_A.weight", ".lora_B.weight",
    ".lora_embedding_a", ".lora_embedding_b",
    ".lora_embedding_a.weight", ".lora_embedding_b.weight",
    ".m", ".m.weight",
)


def _warn_missing_adapter_keys(model, adapter_weights_file):
    """Diff saved adapter LoRA keys against live params and warn.

    Compares both presence AND shape so a wrong-rank live wrapper at a
    matching key (e.g. mlx-lm wrapped the language tower at default
    rank=8 over saved rank-4) is reported as missing. Called on both the
    success and fallback branches; never blocks the following
    load_weights() (exceptions are swallowed into a skip warning).
    Returns the sorted missing-key list so the fallback can raise instead
    of returning a partial adapter; `[]` means clean OR skipped.
    """
    if not adapter_weights_file or not os.path.exists(adapter_weights_file):
        return []
    try:
        from safetensors import safe_open
        from mlx.utils import tree_flatten as _tree_flatten

        with safe_open(adapter_weights_file, framework="numpy") as _f:
            _saved_shapes = {
                k: tuple(_f.get_tensor(k).shape)
                for k in _f.keys()
                if k.endswith(_ADAPTER_LORA_KEY_SUFFIXES)
            }
        _bound_shapes = {}
        for k, v in _tree_flatten(model.parameters()):
            if not k.endswith(_ADAPTER_LORA_KEY_SUFFIXES):
                continue
            shape = getattr(v, "shape", None)
            _bound_shapes[k] = tuple(shape) if shape is not None else None
        # Compare presence AND shape so a wrong-rank live wrapper (e.g.
        # default rank=8 over saved rank-4) is reported missing; a pure
        # key-set diff would call it "fully bound" and strict=False would
        # drop the saved tensors.
        _missing = sorted(
            k for k, saved_shape in _saved_shapes.items()
            if k not in _bound_shapes or _bound_shapes[k] != saved_shape
        )
        if _missing:
            def _describe(k):
                live = _bound_shapes.get(k)
                if live is None:
                    return f"{k} saved={_saved_shapes[k]} live=<missing>"
                return f"{k} saved={_saved_shapes[k]} live={live}"

            _preview = ", ".join(_describe(k) for k in _missing[:5])
            if len(_missing) > 5:
                _preview += f", ... (+{len(_missing) - 5} more)"
            warnings.warn(
                f"Unsloth MLX: {len(_missing)} saved LoRA adapter "
                f"tensor(s) are missing or shape-incompatible with the "
                f"live module tree and will not load: {_preview}",
                stacklevel=3,
            )
        return _missing
    except Exception as _diff_exc:
        warnings.warn(
            f"Unsloth MLX: skipped saved-vs-live adapter key diff "
            f"({_diff_exc!r}); silently-dropped LoRA tensors will not "
            f"be surfaced.",
            stacklevel=3,
        )
        return []


def _apply_lora_metadata_to_wrapper(wrapped, scale, dropout):
    """Restore scale + dropout on a LoRA wrapper after a no-kwarg from_base()."""
    if hasattr(wrapped, "scale"):
        try:
            wrapped.scale = scale
        except Exception:
            pass
    _drop = getattr(wrapped, "dropout", None)
    if _drop is not None:
        if hasattr(_drop, "_p_1"):
            try:
                _drop._p_1 = float(1.0 - float(dropout))
            except Exception:
                pass
        elif hasattr(_drop, "p"):
            try:
                _drop.p = float(dropout)
            except Exception:
                pass
    return wrapped


_FROM_BASE_SIGNATURE_NEEDLES = (
    "unexpected keyword",
    "got an unexpected",
    "got multiple values",
    "no argument named",
    "takes no keyword",
    # Manual-rejection phrasings from custom shims. Intentionally narrow:
    # _is_from_base_kwarg_typeerror only treats these as signature
    # mismatches when the error ALSO names one of our kwargs, so
    # "rank 4 not supported" cannot silently downgrade rank.
    "not accepted",
    "not supported",
)


def _is_from_base_kwarg_typeerror(exc, kwarg=None, kwargs=None):
    """True when `exc` is a `from_base()` signature mismatch (older mlx-lm
    rejecting r/scale/dropout), not an internal wrapper TypeError.

    Falling back through fewer-arg signatures on an unrelated TypeError
    would silently downgrade rank to r=8 and mis-bind a wrong-rank wrapper.
    """
    import re

    msg = str(exc).lower()
    if kwargs is None:
        if kwarg is not None:
            kwargs = (kwarg,)
        else:
            kwargs = ("r", "scale", "dropout")
    kwargs_lower = tuple(k.lower() for k in kwargs)

    # CPython quotes the rejected keyword; accept either quote style.
    quoted = tuple(f"'{k}'" for k in kwargs_lower) + tuple(
        f'"{k}"' for k in kwargs_lower
    )
    if any(q in msg for q in quoted):
        return True

    if not any(needle in msg for needle in _FROM_BASE_SIGNATURE_NEEDLES):
        return False

    # Require the kwarg as a standalone word so short kwargs like "r" do
    # not match "rank" / "wrapper" / "are" in unrelated semantic errors.
    return any(
        re.search(rf"(?<![\w]){re.escape(k)}(?![\w])", msg)
        for k in kwargs_lower
    )


def _no_rank_fallback_or_fail(from_base, module, rank):
    """Call `from_base(module)` only when rank is the upstream default (8).

    Any other rank must NOT silently downgrade; strict=False would then
    drop the saved different-rank tensors.
    """
    if int(rank) != 8:
        raise ValueError(
            f"Unsloth MLX: this mlx-lm wrapper's from_base() does not "
            f"accept a LoRA rank argument; refusing to recreate a rank-"
            f"{rank} adapter as the upstream default rank=8. Upgrade "
            f"mlx-lm or re-save the adapter at rank=8."
        )
    return from_base(module)


def _lora_from_base_compat(lora_cls, module, rank, scale, dropout):
    """Call lora_cls.from_base with older-signature fallback.

    Only retries on signature-mismatch TypeErrors; internal wrapper
    TypeErrors propagate so we never silently downgrade rank to r=8.
    """
    try:
        return lora_cls.from_base(module, r=rank, scale=scale, dropout=dropout)
    except TypeError as exc:
        if not _is_from_base_kwarg_typeerror(exc):
            raise
        try:
            wrapped = lora_cls.from_base(module, r=rank)
        except TypeError as exc2:
            if not _is_from_base_kwarg_typeerror(exc2, kwargs=("r",)):
                raise
            wrapped = _no_rank_fallback_or_fail(lora_cls.from_base, module, rank)
        return _apply_lora_metadata_to_wrapper(wrapped, scale, dropout)


def _patch_mlx_lora_from_base_compat():
    """Monkey-patch mlx_lm LoRA/DoRA `from_base()` to accept scale/dropout
    kwargs on older mlx-lm wheels.

    The canonical mlx-lm walk (linear_to_lora_layers / load_adapters) calls
    `from_base(..., scale=..., dropout=...)` directly with no fallback, so
    without this an older wheel produces an asymmetric partial-LoRA model.
    Idempotent: each class gets `_unsloth_from_base_compat = True`.
    """
    patch_targets = (
        ("mlx_lm.tuner.lora", ("LoRALinear", "LoRASwitchLinear", "LoRAEmbedding")),
        ("mlx_lm.tuner.dora", ("DoRALinear", "DoRAEmbedding")),
    )
    for module_name, class_names in patch_targets:
        try:
            _mod = __import__(module_name, fromlist=["_"])
        except Exception:
            continue

        for cls_name in class_names:
            lora_cls = getattr(_mod, cls_name, None)
            if lora_cls is None or getattr(lora_cls, "_unsloth_from_base_compat", False):
                continue

            # Stubs without from_base: skip rather than patch the wrong
            # attribute; the upstream walk will surface AttributeError.
            original_from_base = getattr(lora_cls, "from_base", None)
            if original_from_base is None:
                continue

            def _compat_from_base(
                cls,
                module,
                r=8,
                dropout=0.0,
                scale=20.0,
                _orig=original_from_base,
            ):
                # Pass-through on new mlx-lm; fall back through older
                # signatures the same way _lora_from_base_compat does.
                # Positional order MUST match upstream
                # `LoRALinear.from_base(linear, r=8, dropout=0.0, scale=20.0)`
                # so positional callers don't get scale/dropout swapped.
                try:
                    return _orig(module, r=r, dropout=dropout, scale=scale)
                except TypeError as exc:
                    if not _is_from_base_kwarg_typeerror(exc):
                        raise
                    try:
                        wrapped = _orig(module, r=r)
                    except TypeError as exc2:
                        if not _is_from_base_kwarg_typeerror(exc2, kwargs=("r",)):
                            raise
                        wrapped = _no_rank_fallback_or_fail(_orig, module, r)
                    return _apply_lora_metadata_to_wrapper(wrapped, scale, dropout)

            # Best-effort patch; simulation stubs with __slots__ raise on
            # assignment, which is the right surface for those builds.
            try:
                lora_cls.from_base = classmethod(_compat_from_base)
                lora_cls._unsloth_from_base_compat = True
            except (AttributeError, TypeError):
                continue


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
    load_in_mxfp4,
    load_in_nvfp4,
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
        load_in_mxfp4 = load_in_nvfp4 = False
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
        if hf_config.get("load_in_mxfp4"):
            load_in_mxfp4 = True
        if hf_config.get("load_in_nvfp4"):
            load_in_nvfp4 = True
        if quantize_modules is None:
            if "quantize_modules" in hf_config:
                quantize_modules = hf_config.get("quantize_modules")
            elif "modules" in hf_config:
                quantize_modules = hf_config.get("modules")

    explicit_mlx_quant_config = bool(mlx_config) or q_bits is not None or q_mode is not None
    if explicit_mlx_quant_config and load_in_4bit and not any((
        load_in_8bit, load_in_fp8, load_in_mxfp4, load_in_nvfp4, load_in_16bit,
    )):
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
        "load_in_mxfp4": bool(load_in_mxfp4),
        "load_in_nvfp4": bool(load_in_nvfp4),
        "load_in_16bit": bool(load_in_16bit),
    }
    if load_flags["load_in_4bit"] and any(
        load_flags[x] for x in (
            "load_in_8bit", "load_in_fp8", "load_in_mxfp4",
            "load_in_nvfp4", "load_in_16bit",
        )
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
            "load_in_4bit, load_in_8bit, load_in_fp8, load_in_mxfp4, "
            "and load_in_nvfp4."
        )
    if load_in_16bit and any(load_flags[x] for x in (
        "load_in_4bit", "load_in_8bit", "load_in_fp8",
        "load_in_mxfp4", "load_in_nvfp4",
    )):
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
        bits, mode, source = 8, "affine", "load_in_8bit"
    elif load_in_fp8:
        bits, mode, source = 8, "mxfp8", "load_in_fp8"
    elif load_in_mxfp4:
        bits, mode, source = 4, "mxfp4", "load_in_mxfp4"
    elif load_in_nvfp4:
        bits, mode, source = 4, "nvfp4", "load_in_nvfp4"
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
        mode = "affine"
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
    from .utils import save_pretrained_merged
    tokenizer = tokenizer or self._tokenizer
    save_pretrained_merged(self, tokenizer, save_directory, **kwargs)


def _mlx_supported_kwargs(kwargs, supported):
    """Keep CUDA-compatible kwargs out of MLX-only save/export APIs."""
    return {key: kwargs[key] for key in supported if key in kwargs}


def _mlx_save_pretrained_gguf(self, save_directory, tokenizer=None,
                               quantization_method="fast_quantized", **kwargs):
    from .utils import save_pretrained_gguf
    tokenizer = tokenizer or self._tokenizer
    kwargs = _mlx_supported_kwargs(kwargs, ("first_conversion",))
    save_pretrained_gguf(self, tokenizer, save_directory,
                         quantization_method=quantization_method, **kwargs)


def _mlx_push_to_hub_merged(self, repo_id, tokenizer=None, save_directory=None, **kwargs):
    from .utils import push_to_hub_merged
    tokenizer = tokenizer or self._tokenizer
    # If save_directory wasn't given, fall back to repo_id (relative dir
    # named after the repo). Callers that already saved locally should
    # pass save_directory= to avoid a redundant re-save.
    save_directory = save_directory or repo_id
    push_to_hub_merged(self, tokenizer, save_directory, repo_id=repo_id, **kwargs)


def _mlx_push_to_hub_gguf(self, repo_id, tokenizer=None,
                            quantization_method="fast_quantized", **kwargs):
    from .utils import push_to_hub_gguf
    tokenizer = tokenizer or self._tokenizer
    kwargs = _mlx_supported_kwargs(kwargs, ("first_conversion", "token", "private"))
    push_to_hub_gguf(self, tokenizer, repo_id, repo_id=repo_id,
                     quantization_method=quantization_method, **kwargs)


def _mlx_save_lora_adapters(self, path, adapter_config=None):
    from .utils import save_lora_adapters
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

    if target_modules is None:
        match_all_linear = True
        target_modules = set()
    else:
        target_modules = set(target_modules or ())

    replacements = 0

    for attr_name in attr_names:
        root = getattr(model, attr_name, None)
        if root is None:
            continue

        def _walk(module):
            nonlocal replacements
            for name, child in list(module.named_modules()):
                if not match_all_linear and not _lora_name_matches_target(name, target_modules):
                    continue
                if isinstance(child, (nn.Linear, nn.QuantizedLinear)):
                    lora_layer = _lora_from_base_compat(
                        LoRALinear,
                        child,
                        rank=lora_config["rank"],
                        scale=lora_config["scale"],
                        dropout=lora_config.get("dropout", 0.0),
                    )
                    replacements += 1
                    if name == "":
                        setattr(model, attr_name, lora_layer)
                        continue
                    # Navigate to parent and replace. The final segment can
                    # be a numeric string (list index) for some VLM trees
                    # like Qwen2.5-VL's vision merger.
                    parts = name.split(".")
                    parent = root
                    for p in parts[:-1]:
                        try:
                            parent = parent[int(p)]
                        except (ValueError, TypeError):
                            parent = getattr(parent, p)
                    leaf = parts[-1]
                    try:
                        parent[int(leaf)] = lora_layer
                    except (ValueError, TypeError):
                        setattr(parent, leaf, lora_layer)

        _walk(root)
        break  # These are alternative names for the same component — stop after first hit
    return replacements


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


def _raise_no_lora_targets(target_modules):
    raise ValueError(
        "Unsloth: No MLX LoRA target modules were found for "
        f"target_modules={target_modules!r}. Check the module names or use "
        "target_modules='all-linear'."
    )


def _validate_mlx_init_lora_weights(init_lora_weights):
    if (
        type(init_lora_weights) is bool
        or init_lora_weights == "gaussian"
    ):
        return
    peft_only_initializers = ("eva", "olora", "pissa", "corda", "loftq", "orthogonal")
    if (
        init_lora_weights in peft_only_initializers
        or (
            isinstance(init_lora_weights, str)
            and init_lora_weights.startswith("pissa_niter_")
        )
    ):
        raise NotImplementedError(
            f"Unsloth: init_lora_weights={init_lora_weights!r} is not "
            "supported for MLX LoRA yet."
        )
    raise ValueError(
        'Unsloth: init_lora_weights must be one of [True, False, "gaussian"]. '
        "MLX does not support PEFT-only data-driven or quantization-aware "
        "initializers yet."
    )


def _apply_mlx_lora_initialization(model, init_lora_weights):
    """Match PEFT LoRA initialization for MLX LoRA modules."""
    if init_lora_weights is True:
        # mlx-lm LoRA constructors already use Linear-style A init and zero B.
        return

    import mlx.core as mx

    def _lora_tensor_shape(tensor):
        # mlx-lm wraps lora tensors in nn.Linear; unwrap to .weight when
        # the bare object doesn't expose .shape.
        if hasattr(tensor, "shape"):
            return tensor.shape
        weight = getattr(tensor, "weight", None)
        if weight is not None and hasattr(weight, "shape"):
            return weight.shape
        return None

    def _assign_lora_tensor(module, attr, value):
        # mlx-lm wraps lora_a / lora_b in nn.Linear; assigning a raw
        # mx.array to the slot replaces the entire wrapper, destroying
        # the layer object and any methods/parameters that came with it.
        # Update .weight when the slot is a layer; only fall back to a
        # direct setattr when the slot is a bare array.
        current = getattr(module, attr, None)
        if current is not None and hasattr(current, "weight") \
                and hasattr(getattr(current, "weight"), "shape"):
            try:
                current.weight = value
                return
            except Exception:
                pass
        setattr(module, attr, value)

    for _, module in model.named_modules():
        if not (hasattr(module, "lora_a") and hasattr(module, "lora_b")):
            continue
        a_shape = _lora_tensor_shape(module.lora_a)
        b_shape = _lora_tensor_shape(module.lora_b)
        if a_shape is None or b_shape is None:
            continue
        if init_lora_weights == "gaussian":
            if hasattr(module, "embedding"):
                _assign_lora_tensor(module, "lora_a", mx.zeros(a_shape))
                _assign_lora_tensor(module, "lora_b", mx.random.normal(shape=b_shape))
            else:
                _assign_lora_tensor(
                    module, "lora_a",
                    mx.random.normal(shape=a_shape) * (1.0 / b_shape[0]),
                )
                _assign_lora_tensor(module, "lora_b", mx.zeros(b_shape))
        elif init_lora_weights is False:
            _assign_lora_tensor(
                module, "lora_a",
                mx.random.uniform(
                    low=-1.0 / math.sqrt(a_shape[0]),
                    high=1.0 / math.sqrt(a_shape[0]),
                    shape=a_shape,
                ),
            )
            _assign_lora_tensor(
                module, "lora_b",
                mx.random.uniform(
                    low=-1.0 / math.sqrt(b_shape[0]),
                    high=1.0 / math.sqrt(b_shape[0]),
                    shape=b_shape,
                ),
            )


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
        load_in_mxfp4=False,
        load_in_nvfp4=False,
        full_finetuning=False,
        token=None,
        trust_remote_code=False,
        text_only=None,
        patch_mode="patched",
        revision=None,
        random_state=3407,
        float32_mixed_precision=None,
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
                trainable. By default MLX mirrors Unsloth Torch full
                finetuning and upcasts trainable floating weights to float32;
                pass ``float32_mixed_precision=False`` to keep native bf16
                weights on bf16-capable Apple Silicon. ``get_peft_model``
                becomes a no-op for models loaded this way.
            token: HuggingFace token for gated models.
            text_only: Loading mode:
                None  — auto-detect from config (default)
                True  — force text-only via mlx-lm
                False — force VLM via mlx-vlm
        """
        if full_finetuning and (
            load_in_4bit or load_in_8bit or load_in_fp8
            or load_in_mxfp4 or load_in_nvfp4
        ):
            print(
                "Unsloth: full_finetuning=True — disabling quantized loads "
                "(quantized weights cannot be trained directly)."
            )
            load_in_4bit = False
            load_in_8bit = False
            load_in_fp8 = False
            load_in_mxfp4 = False
            load_in_nvfp4 = False
            load_in_16bit = True
        import mlx.core as mx
        chip = mx.device_info().get("device_name", "") or ""
        bf16_supported = not chip.startswith(("Apple M1", "Apple M2"))
        target_dtype = None
        if dtype is None:
            target_dtype = mx.bfloat16 if bf16_supported else mx.float16
        else:
            if isinstance(dtype, str):
                target_dtype = getattr(mx, dtype, None)
            elif dtype in (mx.float16, mx.bfloat16, mx.float32):
                target_dtype = dtype
            if target_dtype not in (mx.float16, mx.bfloat16, mx.float32):
                raise ValueError(
                    f"Unsloth: Unsupported dtype {dtype!r}. "
                    f"Use 'float16', 'bfloat16', or 'float32'."
                )
            if target_dtype == mx.bfloat16 and not bf16_supported:
                warnings.warn(
                    f"Unsloth: {chip} lacks native bf16 GPU support — "
                    f"bf16 will be emulated (~40-70%% slower prefill). "
                    f"Pass dtype='float16' on M1/M2.",
                    stacklevel=2,
                )
        if full_finetuning:
            original_target_dtype = target_dtype
            target_dtype, using_float32_full_ft = _resolve_full_finetune_dtype(
                target_dtype,
                float32_mixed_precision,
                mx,
            )
            if not using_float32_full_ft:
                print(
                    "Unsloth: Using bfloat16 MLX full finetuning. "
                    "This reduces memory but can differ from Unsloth Torch's "
                    "float32_mixed_precision=True path."
                )
            else:
                if original_target_dtype != mx.float32:
                    print(
                        "Unsloth: Using float32 MLX full finetuning to match "
                        "Unsloth Torch's explicit float32_mixed_precision=True "
                        "path."
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
            or load_in_mxfp4
            or load_in_nvfp4
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
            load_in_mxfp4=load_in_mxfp4,
            load_in_nvfp4=load_in_nvfp4,
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
        _seed_mlx_random_state(random_state)

        # Split download from config-read so a missing config.json
        # does not clear local_path. LoRA-adapter directories carry
        # adapter_config.json but no config.json; the adapter branch
        # below needs local_path either way.
        local_path = None
        try:
            with _temporary_hf_token_env(token):
                local_path = str(_download(model_name, revision=revision))
        except Exception:
            local_path = None

        config_data = {}
        if local_path:
            config_path = os.path.join(local_path, "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        config_data = json.load(f)
                except (json.JSONDecodeError, KeyError):
                    config_data = {}

        # Reject full_finetuning against a pre-quantized repo. The weights on
        # disk are int4/int8 packed; full FT would need them in a trainable
        # float dtype, and our CCE backward returns mx.zeros for the quantized
        # weight gradient (since dequant→grad→requant is not implemented).
        # Without this check the user would silently train only the
        # non-quantized params (LayerNorms, biases) while the bulk of the
        # model's quantized linears never updated.
        if full_finetuning and _get_existing_mlx_quantization(config_data):
            raise ValueError(
                f"Unsloth: full_finetuning=True was requested against "
                f"'{model_name}', which is a pre-quantized repo. The "
                "quantized weights on disk cannot be trained directly: our "
                "CCE backward zeros the weight gradient for quantized "
                "linears, so full FT would silently update only the "
                "non-quantized params (LayerNorms, biases) and leave every "
                "quantized linear unchanged. Either:\n"
                "  - load the unquantized base (drop the '-4bit' / '-8bit' "
                "suffix from the repo name) for full fine-tuning, or\n"
                "  - keep this quantized base and use LoRA "
                "(full_finetuning=False, the default) which trains only the "
                "adapter matrices."
            )

        adapter_cfg_path = os.path.join(local_path, "adapter_config.json") if local_path else None
        adapter_cfg = None
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
                    adapter_mlx_quant_config = None
                    if adapter_has_quant_metadata:
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
                    # Always reload the base via FastMLXModel.from_pretrained
                    # (works for both text and VLM); the previous mlx_lm.load
                    # fallback for non-quant-metadata adapters silently broke
                    # VLM adapters because mlx-lm's load is text-only.
                    model, tokenizer = FastMLXModel.from_pretrained(
                        base_model_id,
                        max_seq_length=max_seq_length,
                        dtype=dtype,
                        load_in_4bit=False,
                        load_in_8bit=False,
                        load_in_16bit=False,
                        load_in_fp8=False,
                        load_in_mxfp4=False,
                        load_in_nvfp4=False,
                        full_finetuning=full_finetuning,
                        token=token,
                        trust_remote_code=trust_remote_code,
                        text_only=text_only,
                        patch_mode=patch_mode,
                        revision=adapter_base_revision,
                        random_state=random_state,
                        float32_mixed_precision=float32_mixed_precision,
                        **(
                            {"mlx_quantization_config": adapter_mlx_quant_config}
                            if adapter_mlx_quant_config is not None
                            else {}
                        ),
                    )
                    _validate_mlx_adapter_base(model, adapter_cfg)
                    # why: load_adapters only rebuilds language-tower LoRA;
                    # vision/projector LoRA must be re-attached at the saved
                    # paths first so load_weights binds the trained tensors.
                    _saved_lora_paths = _normalize_mlx_lora_module_paths(
                        adapter_cfg.get("unsloth_mlx_lora_module_paths"),
                    )
                    # Call load_adapters FIRST (canonical walk rebuilds the
                    # language tower); _apply_lora_at_paths runs AFTER and
                    # skips already-wrapped paths, leaving only auxiliary
                    # (vision / projector / MoE / embedding) to attach. The
                    # old order produced nested LoRALinear(LoRALinear).
                    from mlx_lm.tuner.utils import load_adapters
                    # Compatibility patch so older mlx-lm's load_adapters
                    # accepts scale=/dropout= without forcing a manual-wrap fallback.
                    _patch_mlx_lora_from_base_compat()
                    adapter_weights_file = os.path.join(local_path, "adapters.safetensors")
                    _load_adapters_ok = False
                    _load_adapters_exc = None
                    # Staging-only R29 DoRA pre-validate: catch missing
                    # mlx_lm.tuner.dora BEFORE load_adapters silently rebuilds
                    # plain LoRA and drops saved DoRA `.m` tensors via
                    # strict=False. Distinct from _apply_lora_at_paths's
                    # post-check, which fires per-module after wrapping.
                    if adapter_cfg.get("fine_tune_type") == "dora":
                        try:
                            import mlx_lm.tuner.dora  # noqa: F401
                        except Exception as _dora_exc:
                            raise RuntimeError(
                                "Unsloth MLX: adapter_config declares "
                                "fine_tune_type='dora' but mlx_lm.tuner.dora "
                                "is unavailable; install a DoRA-capable "
                                "mlx-lm or convert the adapter to plain "
                                "LoRA before reload."
                            ) from _dora_exc
                    try:
                        model = load_adapters(model, local_path)
                        _load_adapters_ok = True
                    except Exception as _exc:
                        # Fall through to the manual-wrap + strict=False
                        # fallback below (needed for adapter sets lacking
                        # mlx-lm metadata, e.g. missing num_layers).
                        _load_adapters_exc = _exc

                    _aux_attached = 0
                    if _saved_lora_paths:
                        # Attach auxiliary paths (vision / projector / MoE)
                        # that linear_to_lora_layers does not walk; the
                        # skip-if-wrapped guard makes language-tower paths
                        # no-ops here.
                        try:
                            _aux_attached = _apply_lora_at_paths(
                                model, _saved_lora_paths, adapter_cfg,
                                adapter_weights_file=adapter_weights_file,
                            ) or 0
                        except (ValueError, ImportError):
                            # Caller-actionable contracts (missing rank
                            # metadata / DoRA class); never downgrade.
                            raise
                        except Exception as _exc:
                            warnings.warn(
                                f"Unsloth MLX: failed to re-attach auxiliary "
                                f"LoRA wrappers ({_exc!r}); some adapter "
                                f"tensors may not load.",
                                stacklevel=2,
                            )

                    # Bind aux tensors via a follow-up load_weights and run
                    # the shared key diff so silent drops surface here too.
                    if _load_adapters_ok and os.path.exists(adapter_weights_file):
                        # Always diff (even when _aux_attached == 0): a
                        # caller-declared aux path the live tree no longer
                        # satisfies would otherwise sit in adapters.safetensors
                        # with no live module to bind into.
                        _missing_after_success = _warn_missing_adapter_keys(
                            model, adapter_weights_file,
                        )
                        if _aux_attached > 0:
                            model.load_weights(adapter_weights_file, strict=False)
                        # Refuse to return a partial adapter on ANY shape
                        # mismatch (e.g. stale rank=8 over rank-4) so the
                        # success branch stays in lockstep with the fallback.
                        if _missing_after_success:
                            _preview = ", ".join(_missing_after_success[:5])
                            if len(_missing_after_success) > 5:
                                _preview += (
                                    f", ... (+{len(_missing_after_success) - 5} more)"
                                )
                            raise RuntimeError(
                                "Unsloth MLX: load_adapters succeeded but "
                                f"{len(_missing_after_success)} saved LoRA "
                                "tensor(s) are missing or shape-incompatible "
                                "with the live module tree "
                                f"({_preview}). Refusing to return a "
                                "partially loaded adapter."
                            )

                    if not _load_adapters_ok:
                        # No wrappers → strict=False would silently drop
                        # every saved tensor and return a base model;
                        # re-raise the original load_adapters error.
                        if _aux_attached == 0:
                            if _load_adapters_exc is not None:
                                raise _load_adapters_exc
                            raise RuntimeError(
                                "Unsloth MLX: adapter load failed and no "
                                "live LoRA wrappers exist to bind the "
                                "saved tensors against."
                            )
                        if os.path.exists(adapter_weights_file):
                            # Diff saved-vs-live before strict=False; the
                            # shared helper covers DoRA `.m` + lora_{a,b}.
                            _missing_saved_keys = _warn_missing_adapter_keys(
                                model, adapter_weights_file,
                            )
                            # Refuse an aux-only partial adapter (the language
                            # tower would silently mis-train); re-raise the
                            # original load_adapters error chained.
                            if _missing_saved_keys:
                                _preview = ", ".join(_missing_saved_keys[:5])
                                if len(_missing_saved_keys) > 5:
                                    _preview += (
                                        f", ... (+{len(_missing_saved_keys) - 5} more)"
                                    )
                                _partial_err = RuntimeError(
                                    "Unsloth MLX: adapter load failed and "
                                    "the manual fallback would only bind "
                                    f"part of the adapter "
                                    f"({len(_missing_saved_keys)} saved "
                                    f"LoRA tensor(s) have no live module: "
                                    f"{_preview}). Refusing to return a "
                                    "partially loaded adapter."
                                )
                                if _load_adapters_exc is not None:
                                    raise _partial_err from _load_adapters_exc
                                raise _partial_err
                            model.load_weights(adapter_weights_file, strict=False)
                        else:
                            # No safetensors fallback; surface the
                            # original load_adapters failure.
                            if _load_adapters_exc is not None:
                                raise _load_adapters_exc
                            raise RuntimeError(
                                "Unsloth MLX: adapter load failed and "
                                "adapters.safetensors is missing."
                            )
                    model = _eval_mlx_model_after_adapter_reload(model)
                    loaded_model_config = getattr(model, "_config", None)
                    is_vlm_model = bool(getattr(model, "_is_vlm_model", False))
                    processor = getattr(model, "_processor", None)

                    with _temporary_hf_token_env(token):
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
                # Preserve Unsloth-tagged ValueError/ImportError/RuntimeError
                # so adapter-config-level failures aren't swallowed into
                # a silent base-model fallback.
                _msg = str(e)
                _is_unsloth = (
                    "Unsloth:" in _msg or "Unsloth MLX:" in _msg
                )
                if _is_unsloth and isinstance(
                    e, (ValueError, ImportError, RuntimeError)
                ):
                    raise
                # Also preserve the namespaced Unsloth MLX RuntimeError
                # signal from _apply_lora_at_paths.
                if isinstance(e, RuntimeError) and "Unsloth MLX:" in str(e):
                    raise
                # If adapter_config declared a LoRA/DoRA artifact, refuse
                # the silent base-model fallback for any other exception.
                _is_lora_adapter = False
                if isinstance(adapter_cfg, dict):
                    _peft_type = str(adapter_cfg.get("peft_type", "")).upper()
                    _ft_type = str(adapter_cfg.get("fine_tune_type", "")).lower()
                    _is_lora_adapter = (
                        _peft_type == "LORA"
                        or _ft_type in {"lora", "dora"}
                    )
                if _is_lora_adapter:
                    raise RuntimeError(
                        "Unsloth MLX: failed to load the LoRA adapter declared "
                        f"in adapter_config.json ({type(e).__name__}: {e}); "
                        "refusing to silently fall back to a base model load."
                    ) from e
                print(f"Unsloth: LoRA adapter detection failed ({e}), falling back to standard load.")

        model_type = config_data.get("model_type", "")

        # Step 2: Route based on text_only
        is_vlm = False
        force_vlm_text_path = bool(
            text_only is True and _prefer_vlm_loader_for_text(config_data, model_type)
        )

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
                import mlx.core as mx
                from mlx_vlm.utils import load_config as _vlm_load_config
                print(f"Unsloth: Loading {model_name} via mlx-vlm (VLM, "
                      f"runtime {quantization_spec.bits}-bit {quantization_spec.mode} quantization)...")
                with _temporary_hf_token_env(token):
                    model, processor = vlm_load(
                        model_name,
                        lazy=True,
                        revision=revision,
                        **extra_kwargs,
                    )
                    vlm_cfg = _vlm_load_config(local_path or model_name)
                model, vlm_cfg = _apply_mlx_quantization(
                    model, vlm_cfg, quantization_spec,
                    is_vlm=True, user_predicate=quant_predicate,
                )
                model._config = vlm_cfg
                mx.eval(model.parameters())
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
                    hf_token=token,
                )

            processor = _repair_degraded_vlm_processor(
                processor,
                local_path or model_name,
                model_type,
                token=token,
                trust_remote_code=trust_remote_code,
            )

            if target_dtype is not None:
                _convert_mlx_dtype(model, target_dtype, model_type=model_type)
            elif want_runtime_quant:
                import mlx.core as mx
                mx.eval(model.parameters())

            from .utils import (
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
                hf_token=token,
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
                _convert_mlx_dtype(model, target_dtype, model_type=model_type)
            elif want_runtime_quant:
                import mlx.core as mx
                mx.eval(model.parameters())
            from .utils import normalize_mlx_chat_template

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
        use_rslora=False,
        init_lora_weights=True,
        use_gradient_checkpointing="mlx",
        random_state=3407,
        max_seq_length=2048,
        train_vision=False,
        train_projector=False,
        finetune_vision_layers=None,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        finetune_last_n_layers=None,
        **kwargs,  # Accept and ignore GPU-only kwargs
    ):
        """Apply LoRA via mlx-lm on Apple Silicon.

        For VLMs, applies LoRA to the language model and optionally to the
        vision tower (train_vision=True) and projector (train_projector=True).

        When the model was loaded with ``full_finetuning=True``, this is a
        no-op: the full-precision parameters stay trainable and the model
        is returned as-is.
        """
        loftq_config = kwargs.pop("loftq_config", None)
        if loftq_config is not None:
            raise NotImplementedError(
                "Unsloth: loftq_config is not supported for MLX LoRA yet."
            )
        qat_scheme = kwargs.pop("qat_scheme", None)
        if qat_scheme is not None:
            raise NotImplementedError(
                "Unsloth: qat_scheme is not supported for MLX LoRA yet."
            )
        if bias not in (None, False, "none"):
            print(
                "Unsloth: bias is not supported for MLX LoRA yet - "
                "ignoring bias={!r}.".format(bias)
            )
        if type(use_rslora) is not bool:
            raise TypeError("Unsloth: use_rslora must be True or False.")
        _validate_mlx_init_lora_weights(init_lora_weights)

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

        # finetune_vision_layers (None = use train_vision arg; bool overrides it)
        if finetune_vision_layers is not None:
            train_vision = bool(finetune_vision_layers)


        # PEFT/CUDA semantics: target_modules="all-linear" means literally
        # every nn.Linear in the model — fused QKV (qkv_proj), MoE routers
        # and experts, vision tower linears, multimodal projector, untied
        # lm_head, etc. Walk the model tree and collect those names instead
        # of silently collapsing to the canonical 7-name list (which would
        # leave fused-attention archs and MoEs with no LoRA on most of
        # their linears).
        if target_modules == ["all-linear"] or target_modules == "all-linear":
            target_modules = _collect_all_linear_target_names(model)
            if not target_modules:
                # No Linear modules discovered (shouldn't happen on a real
                # model). Fall back to the canonical default.
                target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ]

        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

        # Filter target_modules by finetune_attention_modules / finetune_mlp_modules.
        # Applies whether target_modules came from the user as an explicit list,
        # was built from defaults, or was normalized from "all-linear" — so
        # toggling these flags always has effect.
        if isinstance(target_modules, list) and len(target_modules) > 0:
            _ATTN = {
                "q_proj", "k_proj", "v_proj", "o_proj",
                "qkv", "qkv_proj", "Wqkv", "in_proj",
                "c_attn", "out_proj",
            }
            _MLP = {
                "gate_proj", "up_proj", "down_proj",
                "fc1", "fc2", "w1", "w2", "w3",
            }
            filtered = []
            for m in target_modules:
                if m in _ATTN and not finetune_attention_modules:
                    continue
                if m in _MLP and not finetune_mlp_modules:
                    continue
                filtered.append(m)
            target_modules = filtered

        lora_config = {
            "rank": r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "scale": lora_alpha / (math.sqrt(r) if use_rslora else r),
        }

        is_vlm = getattr(model, "_is_vlm_model", False)

        if is_vlm:
            # VLM path: freeze everything, then apply LoRA selectively
            _fix_missing_no_grad(model)
            _fix_gemma4_kv_sharing(model)
            model.freeze()

            # Apply LoRA to the language model (filtered by target_modules)
            language_lora_count = 0
            if finetune_language_layers and (
                target_modules is None or (isinstance(target_modules, list) and len(target_modules) > 0)
            ):
                lm = model.language_model
                num_layers = 0
                if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                    num_layers = len(lm.model.layers)
                if finetune_last_n_layers is not None and num_layers > 0:
                    num_layers = max(1, min(int(finetune_last_n_layers), num_layers))
                language_lora_keys = _resolve_lora_keys(lm, target_modules)
                if language_lora_keys is None or len(language_lora_keys) > 0:
                    # Compatibility patch (older mlx-lm rejects
                    # scale=/dropout= on from_base); applied before
                    # _seed_mlx_random_state since monkey-patching
                    # Python classes does not advance mx.random.
                    _patch_mlx_lora_from_base_compat()
                    # Match mlx_lm/tuner/lora.py (def train) -- seed
                    # mx.random immediately before LoRA init; lazy MLX
                    # state advances otherwise leak into lora_a sampling.
                    _seed_mlx_random_state(random_state)
                    linear_to_lora_layers(
                        lm,
                        num_layers=num_layers,
                        config={**lora_config, "keys": language_lora_keys},
                        use_dora=False,
                    )
                    language_lora_count = len(language_lora_keys) if language_lora_keys is not None else num_layers

            # Optionally apply LoRA to vision tower
            vision_lora_count = 0
            if train_vision:
                vision_lora_count = _lora_walk_module(
                    model,
                    lora_config,
                    target_modules,
                    attr_names=("vision_tower", "vision_model", "vision_encoder"),
                )

            # Optionally train the multimodal projector / connector. Prefer
            # projector LoRA over unfreezing raw weights because many MLX VLM
            # checkpoints expose projector layers as QuantizedLinear, and MLX
            # does not backprop into quantized weights directly.
            projector_lora_count = 0
            if train_projector:
                projector_lora_count = _lora_walk_module(
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

            if (
                language_lora_count == 0
                and vision_lora_count == 0
                and projector_lora_count == 0
            ):
                if not finetune_language_layers and not train_vision and not train_projector:
                    raise ValueError(
                        "Unsloth: no trainable LoRA targets — every layer-group "
                        "flag is off (finetune_language_layers=False, "
                        "finetune_vision_layers=False, train_projector=False). "
                        "Enable at least one. To LoRA only MLP modules of the "
                        "language model, set finetune_language_layers=True, "
                        "finetune_attention_modules=False, finetune_mlp_modules=True."
                    )
                if isinstance(target_modules, list) and len(target_modules) == 0:
                    raise ValueError(
                        "Unsloth: target_modules became empty after filtering by "
                        "finetune_attention_modules / finetune_mlp_modules. Enable "
                        "at least one of these flags."
                    )
                _raise_no_lora_targets(target_modules)

            # Unfreeze all LoRA params across the entire tree
            model.unfreeze(keys=["lora_a", "lora_b"], strict=False)
        else:
            # Text-only path — filter by target_modules
            # Fix missing _no_grad on modules that use __new__ without __init__
            # (e.g. Gemma4 AudioRelativePositionEmbedding loaded via VLM path)
            _fix_missing_no_grad(model)

            if not finetune_language_layers:
                warnings.warn(
                    "Unsloth: finetune_language_layers=False on a text-only model — "
                    "no LoRA will be applied; the model has no trainable parameters.",
                    stacklevel=2,
                )
            else:
                num_layers = 0
                if hasattr(model, "model") and hasattr(model.model, "layers"):
                    num_layers = len(model.model.layers)
                if finetune_last_n_layers is not None and num_layers > 0:
                    num_layers = max(1, min(int(finetune_last_n_layers), num_layers))
                language_lora_keys = _resolve_lora_keys(model, target_modules)
                if language_lora_keys is not None and len(language_lora_keys) == 0:
                    _raise_no_lora_targets(target_modules)
                # Compatibility patch (older mlx-lm rejects scale=/dropout=
                # on from_base); applied before _seed_mlx_random_state
                # since monkey-patching does not advance mx.random.
                _patch_mlx_lora_from_base_compat()
                # Match mlx_lm/tuner/lora.py (def train) -- seed
                # mx.random immediately before LoRA init; lazy MLX
                # state advances otherwise leak into lora_a sampling.
                _seed_mlx_random_state(random_state)
                linear_to_lora_layers(
                    model,
                    num_layers=num_layers,
                    config={**lora_config, "keys": language_lora_keys},
                    use_dora=False,
                )

            model.freeze()
            model.unfreeze(keys=["lora_a", "lora_b"], strict=False)

        _apply_mlx_lora_initialization(model, init_lora_weights)

        # Apply gradient checkpointing if requested
        # "mlx" (default) or True → apply; False or "none" → skip
        if isinstance(use_gradient_checkpointing, str):
            _apply_gc = use_gradient_checkpointing.lower() not in ("false", "none", "")
        else:
            _apply_gc = bool(use_gradient_checkpointing)

        if _apply_gc:
            from .utils import apply_gradient_checkpointing
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

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

"""Lightweight FastLanguageModel for Apple Silicon / MLX.

No GPU deps: uses mlx-lm (text) and mlx-vlm (VLM) instead of unsloth.models
(which pulls in CUDA kernels).
"""

import copy
import gc
import json
import importlib
import inspect
import math
import os
import re
import shutil
import sys
import tempfile
import types
import warnings
import weakref
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from functools import wraps
from pathlib import Path

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
_AUDIO_CONV_SANITIZE_PATCHED: set[str] = set()
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
    """Expose an explicit HF token via env for the duration of the call.

    mlx-lm / mlx-vlm don't thread ``token=`` through every download path; this
    keeps the token local without mutating persistent HF login state.
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
    """Case-insensitive ``model_type`` lookup in ``unsloth_zoo.FORCE_FLOAT32``
    (strips ``-``/``_``; trailing comma marks an exact-match entry)."""
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
    """Cast float params to target_dtype (quantized ints preserved), honoring
    the model's optional path-based ``cast_predicate``.

    Warns on bf16 -> fp16 for ``unsloth_zoo.FORCE_FLOAT32`` archs (Gemma3,
    gpt_oss, Qwen3.5) where fp16's narrower range NaN/Infs in training.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map_with_path
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


def _keep_norm_parameters_float32(model) -> None:
    """Prepare MLX training by keeping normalization parameters in fp32."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map_with_path
    from .utils import is_mlx_norm_parameter_path

    try:
        parameters = model.parameters()
    except AttributeError:
        return

    needs_cast = False
    for k, v in tree_flatten(parameters):
        if (
            is_mlx_norm_parameter_path(k)
            and mx.issubdtype(v.dtype, mx.floating)
            and v.dtype != mx.float32
        ):
            needs_cast = True
            break
    if not needs_cast:
        return

    model.update(tree_map_with_path(
        lambda k, v: v.astype(mx.float32)
        if is_mlx_norm_parameter_path(k) and mx.issubdtype(v.dtype, mx.floating)
        else v,
        parameters,
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
    """Leaf-suffix names of every Linear / QuantizedLinear in `model`.

    Mirrors PEFT's ``target_modules="all-linear"``: walk the live tree and
    return each leaf's semantic name (``w1``, ``q_proj``, ``lm_head``, ...),
    skipping numeric list indices. mlx-lm's ``linear_to_lora_layers`` matches
    on these, so LoRA covers fused-QKV, MoE, projector, and untied heads.
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
        # Never let discovery raise during LoRA setup; caller falls back to
        # the canonical 7-name default.
        return []
    return sorted(names)


def _is_vlm(config: dict) -> bool:
    """Detect whether a config describes a VLM (via "vision_config" or a
    model_type in mlx_vlm's supported set)."""
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


def _raise_if_qk_norm_version_gap(model_type, message, error):
    """A strict mlx load rejecting q_norm / k_norm means mlx-lm / mlx-vlm is too
    old (or regressed, e.g. 0.31.3 - mlx-lm #1242) for a QK-norm arch; dropping
    those weights breaks the model, so raise a clear error instead."""
    if "parameters not in model" not in message:
        return
    if not any(marker in message for marker in ("k_norm", "q_norm")):
        return
    # gemma4/gemma3n KV-sharing tails ship DEAD k_proj/v_proj/k_norm (never
    # q_norm); dropping them via strict=False is safe (mlx-lm #1242). A real gap
    # rejects q_norm/k_norm WITHOUT the paired projections - raise only then.
    kv_sharing_dead_tail = (
        "self_attn.k_proj" in message
        and "self_attn.v_proj" in message
        and "q_norm" not in message
    )
    if kv_sharing_dead_tail:
        return
    versions = []
    for pkg in ("mlx-lm", "mlx-vlm"):
        try:
            from importlib.metadata import version as _dist_version

            versions.append(f"{pkg}={_dist_version(pkg)}")
        except Exception:
            # Best-effort hint; omit a missing dist.
            pass
    installed = f" Installed: {', '.join(versions)}." if versions else ""
    raise ValueError(
        f"Unsloth: cannot load MLX {model_type or 'model'} - the installed "
        f"mlx-lm / mlx-vlm rejects its QK-norm (q_norm/k_norm) weights, so it is "
        f"too old or regressed for this architecture (mlx-lm 0.31.3 broke "
        f"gemma4 / qwen3_5). Reinstall an arch-complete build, e.g. "
        f'`pip install -U "mlx-lm>=0.22.0,!=0.31.3" "mlx-vlm"`. See mlx-lm #1242.{installed}'
    ) from error


def _patch_deepseek_ocr_transformers_import_compat(model_type):
    """Let DeepSeek-OCR remote config imports survive newer Transformers.

    The MLX path does not instantiate the Torch Llama flash-attention class,
    but DeepSeek-OCR's tokenizer/config import still imports that symbol from
    Transformers. Recent Transformers releases removed it, so provide the
    nearest eager-attention alias only for this import-time compatibility case.
    """
    if model_type not in {"deepseekocr", "deepseekocr_2", "deepseek_vl_v2"}:
        return
    try:
        from transformers.models.llama import modeling_llama
    except Exception:
        return
    if (
        not hasattr(modeling_llama, "LlamaFlashAttention2")
        and hasattr(modeling_llama, "LlamaAttention")
    ):
        modeling_llama.LlamaFlashAttention2 = modeling_llama.LlamaAttention
    try:
        from transformers.utils import import_utils
    except Exception:
        return
    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: False


def _deepseek_ocr_config_model_type(config_data):
    architectures = config_data.get("architectures") or ()
    if isinstance(architectures, str):
        architectures = (architectures,)
    normalized = {str(arch).lower() for arch in architectures}
    if "deepseekocrforcausallm" in normalized:
        return "deepseekocr"
    if "deepseekocr2forcausallm" in normalized:
        return "deepseekocr_2"
    return None


def _tokenizer_supports_list_extra_special_tokens():
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except Exception:
        return True
    init = PreTrainedTokenizerBase.__init__
    original_init = getattr(init, "_unsloth_original_init", init)

    class _ProbeTokenizer(PreTrainedTokenizerBase):
        model_input_names = ["input_ids"]
        padding_side = "right"
        truncation_side = "right"

        def _add_tokens(self, new_tokens, special_tokens=False):
            return len(new_tokens)

        def get_vocab(self):
            return {}

        @property
        def vocab_size(self):
            return 0

    try:
        probe = _ProbeTokenizer.__new__(_ProbeTokenizer)
        original_init(probe, extra_special_tokens=["<|unsloth_probe|>"])
    except AttributeError as error:
        if "keys" in str(error):
            return False
        return True
    except TypeError as error:
        if "extra_special_tokens" in str(error):
            return False
        return True
    except Exception:
        return True
    return True


def _normalize_tokenizer_config_extra_special_tokens(
    tokenizer_config,
    *,
    supports_list_extra_special_tokens=None,
):
    extra_special_tokens = tokenizer_config.get("extra_special_tokens")
    if not isinstance(extra_special_tokens, list):
        return tokenizer_config, False
    if supports_list_extra_special_tokens is None:
        supports_list_extra_special_tokens = (
            _tokenizer_supports_list_extra_special_tokens()
        )
    if supports_list_extra_special_tokens:
        return tokenizer_config, False

    patched_config = dict(tokenizer_config)
    additional_special_tokens = patched_config.get("additional_special_tokens")
    merged_additional = []
    if isinstance(additional_special_tokens, list):
        merged_additional.extend(additional_special_tokens)
    for token in extra_special_tokens:
        if token not in merged_additional:
            merged_additional.append(token)
    patched_config["additional_special_tokens"] = merged_additional

    model_specific_tokens = patched_config.get("model_specific_special_tokens")
    patched_config["extra_special_tokens"] = (
        dict(model_specific_tokens) if isinstance(model_specific_tokens, dict) else {}
    )
    return patched_config, True


def _normalize_tokenizer_config_backend_class(
    tokenizer_config,
    *,
    backend_class_available=None,
):
    if tokenizer_config.get("tokenizer_class") != "TokenizersBackend":
        return tokenizer_config, False
    if backend_class_available is None:
        try:
            from transformers.models.auto.tokenization_auto import (
                tokenizer_class_from_name,
            )
            backend_class_available = (
                tokenizer_class_from_name("TokenizersBackend") is not None
            )
        except Exception:
            return tokenizer_config, False
    if backend_class_available:
        return tokenizer_config, False

    patched_config = dict(tokenizer_config)
    patched_config["tokenizer_class"] = "PreTrainedTokenizerFast"
    return patched_config, True


def _materialize_mlx_vlm_config_data(local_path, config_data):
    override_dir = tempfile.mkdtemp(prefix="unsloth_mlx_vlm_config_")
    for name in os.listdir(local_path):
        src = os.path.join(local_path, name)
        dst = os.path.join(override_dir, name)
        if name == "config.json":
            continue
        _link_or_copy_path(src, dst)
    with open(os.path.join(override_dir, "config.json"), "w") as f:
        json.dump(config_data, f, indent=2)
    return override_dir


def _mlx_vlm_config_override_data(config_data):
    corrected_model_type = _deepseek_ocr_config_model_type(config_data)
    if (
        corrected_model_type is None
        or config_data.get("model_type") == corrected_model_type
    ):
        return None

    patched_config = dict(config_data)
    patched_config["model_type"] = corrected_model_type
    # mlx-vlm supplies the model/processor implementation locally. Keeping the
    # Torch remote-code auto_map here makes AutoProcessor import incompatible
    # DeepSeek OCR Torch modules during MLX loads.
    patched_config.pop("auto_map", None)
    print(
        "Unsloth: Routing DeepSeek OCR checkpoint through "
        f"mlx-vlm model_type={corrected_model_type!r}."
    )
    return patched_config


def _keep_mlx_vlm_config_view_alive(model, override_dir):
    override_dir = str(override_dir)
    paths = list(getattr(model, "_unsloth_mlx_config_view_paths", ()))
    paths.append(override_dir)
    model._unsloth_mlx_config_view_paths = paths
    try:
        finalizers = list(getattr(model, "_unsloth_mlx_config_view_finalizers", ()))
        finalizers.append(weakref.finalize(model, shutil.rmtree, override_dir, ignore_errors=True))
        model._unsloth_mlx_config_view_finalizers = finalizers
    except TypeError:
        pass


def _materialize_mlx_vlm_config_override(
    local_path,
    config_data,
    *,
    normalize_tokenizer_config=False,
    supports_list_extra_special_tokens=None,
):
    """Return a load path whose sidecars are compatible with mlx-vlm loaders."""
    if not local_path:
        return local_path, config_data
    patched_files = {}

    corrected_model_type = _deepseek_ocr_config_model_type(config_data)
    patched_config = config_data
    if (
        corrected_model_type is not None
        and config_data.get("model_type") != corrected_model_type
    ):
        patched_config = dict(config_data)
        patched_config["model_type"] = corrected_model_type
        # Drop the Torch remote-code auto_map so AutoProcessor doesn't import
        # incompatible DeepSeek OCR Torch modules during MLX loads.
        patched_config.pop("auto_map", None)
        patched_files["config.json"] = patched_config

    if normalize_tokenizer_config:
        tokenizer_config = _read_json_file(
            os.path.join(local_path, "tokenizer_config.json")
        )
        patched_tokenizer_config, patched_backend = (
            _normalize_tokenizer_config_backend_class(tokenizer_config)
        )
        patched_tokenizer_config, patched_tokenizer = (
            _normalize_tokenizer_config_extra_special_tokens(
                patched_tokenizer_config,
                supports_list_extra_special_tokens=supports_list_extra_special_tokens,
            )
        )
        if patched_backend or patched_tokenizer:
            patched_files["tokenizer_config.json"] = patched_tokenizer_config

    if not patched_files:
        return local_path, config_data

    override_dir = tempfile.mkdtemp(prefix="unsloth_mlx_vlm_config_")
    for name in os.listdir(local_path):
        src = os.path.join(local_path, name)
        dst = os.path.join(override_dir, name)
        if name in patched_files:
            continue
        try:
            os.symlink(src, dst)
        except FileExistsError:
            pass
    for name, data in patched_files.items():
        with open(os.path.join(override_dir, name), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    if corrected_model_type is not None and "config.json" in patched_files:
        print(
            "Unsloth: Routing DeepSeek OCR checkpoint through "
            f"mlx-vlm model_type={corrected_model_type!r}."
        )
    return override_dir, patched_config


def _load_mlx_lm_with_strict_fallback(
    model_name,
    model_type,
    mlx_load,
    mlx_load_kwargs,
    hf_token=None,
):
    """Load text models via mlx-lm, retrying strict=False for known safe
    mismatches (extra checkpoint tensors mlx-lm doesn't instantiate).

    mlx-lm's public load() doesn't expose strict=False, so we use the internal
    loader only for registered mismatch signatures.
    """
    # why: mlx-lm 0.22.0 load() rejects return_config / revision; bypass it so
    # signature drift between releases doesn't break loading.
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
        # Active-layer QK-norm weights are load-bearing: never strict=False past
        # them (the dead KV-sharing tail still falls through to the fallback).
        _raise_if_qk_norm_version_gap(model_type, message, error)
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


def _mlx_lm_metadata_allow_patterns():
    return [
        "*.json",
        "*.py",
        "tokenizer.model",
        "*.tiktoken",
        "tiktoken.model",
        "*.txt",
        "*.jsonl",
        "*.jinja",
    ]


def _link_or_copy_path(src, dst):
    try:
        os.symlink(src, dst)
        return
    except FileExistsError:
        return
    except OSError:
        pass

    try:
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            os.link(src, dst)
        return
    except FileExistsError:
        return
    except OSError:
        pass

    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def _mlx_lm_snapshot_view(model_path, *, weight_files=()):
    src_root = Path(model_path).resolve()
    dst_root = Path(tempfile.mkdtemp(prefix="unsloth_mlx_dist_view_"))
    allowed_weights = {str(file) for file in (weight_files or ())}
    for root, dirs, files in os.walk(src_root):
        rel_root = os.path.relpath(root, src_root)
        dst_dir = dst_root if rel_root == "." else dst_root / rel_root
        dst_dir.mkdir(parents=True, exist_ok=True)
        for directory in dirs:
            (dst_dir / directory).mkdir(exist_ok=True)
        for name in files:
            rel_file = name if rel_root == "." else os.path.join(rel_root, name)
            is_weight = name.endswith(".safetensors")
            if (
                is_weight
                and rel_file not in allowed_weights
                and name not in allowed_weights
            ):
                continue
            src = os.path.join(root, name)
            dst = dst_dir / name
            _link_or_copy_path(src, dst)
    return dst_root


@contextmanager
def _temporary_mlx_lm_snapshot_view(model_path, *, weight_files=()):
    view_path = _mlx_lm_snapshot_view(model_path, weight_files=weight_files)
    try:
        yield view_path
    finally:
        shutil.rmtree(view_path, ignore_errors=True)


def _load_mlx_lm_distributed(
    model_name,
    model_type,
    mlx_load_kwargs,
    *,
    pipeline_group=None,
    tensor_group=None,
    hf_token=None,
):
    """Load text models following mlx-lm's sharded_load ordering.

    Pipeline placement must happen before downloading/loading local weight
    shards; post-load sharding defeats the memory behavior of pipeline
    parallelism.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.utils import _download, load_model, load_tokenizer

    pipeline_group, tensor_group = _mlx_active_distributed_groups(
        pipeline_group,
        tensor_group,
    )
    tokenizer_config = mlx_load_kwargs.get("tokenizer_config") or {}
    model_config = mlx_load_kwargs.get("model_config") or {}
    revision = mlx_load_kwargs.get("revision")
    want_config = mlx_load_kwargs.get("return_config", False)

    with _temporary_hf_token_env(hf_token):
        model_path = _download(
            model_name,
            revision=revision,
            allow_patterns=_mlx_lm_metadata_allow_patterns(),
        )
        with _temporary_mlx_lm_snapshot_view(model_path) as metadata_model_path:
            model, config = load_model(
                metadata_model_path,
                lazy=True,
                strict=False,
                model_config=model_config,
            )

            mode = _mlx_distributed_sharding_mode(
                model,
                pipeline_group=pipeline_group,
                tensor_group=tensor_group,
                model_name=model_name,
            )
            if mode == "tensor":
                tensor_load_kwargs = dict(mlx_load_kwargs)
                tensor_load_kwargs["lazy"] = True
                tensor_load_kwargs["return_config"] = True
                model, tokenizer, config = _load_mlx_lm_with_strict_fallback(
                    model_name,
                    model_type,
                    None,
                    tensor_load_kwargs,
                    hf_token=hf_token,
                )
                _apply_mlx_distributed_sharding(
                    model,
                    tensor_group=tensor_group,
                    model_name=model_name,
                )
                mx.eval(model.parameters())
                mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
                if want_config:
                    return model, tokenizer, config
                return model, tokenizer

            if mode == "pipeline":
                _apply_mlx_distributed_sharding(
                    model,
                    pipeline_group=pipeline_group,
                    model_name=model_name,
                )
                try:
                    with open(
                        metadata_model_path / "model.safetensors.index.json",
                        "r",
                    ) as file:
                        index_data = json.load(file)
                        weight_index = index_data.get("weight_map")
                        if not isinstance(weight_index, dict):
                            raise ValueError(
                                "Unsloth: MLX pipeline distributed loading requires a "
                                "valid 'weight_map' in model.safetensors.index.json."
                            )
                except FileNotFoundError as error:
                    raise ValueError(
                        "Unsloth: MLX pipeline distributed loading requires a "
                        "converted MLX checkpoint with model.safetensors.index.json."
                    ) from error

                local_files = set()
                for key, _value in tree_flatten(model.parameters()):
                    for indexed_key in _mlx_pipeline_index_keys_for_parameter(
                        key,
                        weight_index,
                    ):
                        file_name = weight_index.get(indexed_key)
                        if file_name is None:
                            raise ValueError(
                                "Unsloth: MLX pipeline distributed loading is only "
                                "supported for converted MLX models with indexed weights."
                            )
                        local_files.add(file_name)

        _download(model_name, revision=revision, allow_patterns=sorted(local_files))
        final_model_path = _mlx_lm_snapshot_view(model_path, weight_files=local_files)
        cleanup_final_model_path = True

        try:
            tokenizer = load_tokenizer(
                final_model_path,
                tokenizer_config,
                eos_token_ids=config.get("eos_token_id", None),
            )
            model, _config = load_model(
                final_model_path,
                lazy=True,
                strict=False,
                model_config=model_config,
            )
            config = _config
            _apply_mlx_distributed_sharding(
                model,
                pipeline_group=pipeline_group,
                tensor_group=tensor_group,
                model_name=model_name,
            )
            mx.eval(model.parameters())

            if pipeline_group is not None or tensor_group is not None:
                mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
        finally:
            if cleanup_final_model_path:
                shutil.rmtree(final_model_path, ignore_errors=True)

    if want_config:
        return model, tokenizer, config
    return model, tokenizer


def _resolve_distributed_runtime_quantization(
    model_name,
    quantization_spec,
    *,
    distributed_requested,
    want_runtime_quant,
):
    if not distributed_requested or not want_runtime_quant:
        return quantization_spec, want_runtime_quant
    if quantization_spec.source == "load_in_4bit":
        warnings.warn(
            "Unsloth: distributed MLX inference does not support runtime "
            f"quantization for '{model_name}'. Loading the full-precision MLX "
            "checkpoint instead of applying the default load_in_4bit=True. "
            "Use a pre-quantized MLX repo for distributed quantized inference.",
            stacklevel=3,
        )
        return (
            _MLXQuantizationSpec(
                enabled=False,
                source="distributed_full_precision",
                quantize_modules=quantization_spec.quantize_modules,
                has_callable_predicate=quantization_spec.has_callable_predicate,
                force_requantize=quantization_spec.force_requantize,
            ),
            False,
        )
    raise ValueError(
        "Unsloth: distributed MLX inference requires a pre-quantized "
        "or full-precision MLX repo. Runtime MLX quantization while "
        "sharding is unsupported; use an mlx-community *-4bit/*-8bit "
        "repo or load without quantized load flags."
    )


def _mlx_pipeline_index_keys_for_parameter(key, weight_index):
    yield key
    if not key.endswith(".weight"):
        return
    prefix = key[:-len(".weight")]
    for suffix in (".scales", ".biases", ".bias"):
        sibling = f"{prefix}{suffix}"
        if sibling in weight_index:
            yield sibling


def _load_mlx_vlm_with_extra_weight_filter(
    model_name,
    model_type,
    vlm_load,
    vlm_kwargs,
    hf_token=None,
):
    """Load VLMs, filtering known extra checkpoint tensors on retry.

    Some VLM checkpoints carry tensors mlx-vlm doesn't instantiate; since
    load() lacks strict=False, retry with a temporary load_weights shim for
    registered mismatch signatures and exact allow-listed keys only.
    """
    _patch_deepseek_ocr_transformers_import_compat(model_type)
    try:
        with _temporary_hf_token_env(hf_token):
            return vlm_load(model_name, **vlm_kwargs)
    except ValueError as error:
        message = str(error)
        # QK-norm weights are load-bearing: check before the extra-weight filter.
        _raise_if_qk_norm_version_gap(model_type, message, error)
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
            except ValueError as retry_error:
                # Filtering the extras can unmask a q_norm/k_norm version gap on
                # the retry; surface the actionable error instead of the raw one.
                _raise_if_qk_norm_version_gap(model_type, str(retry_error), retry_error)
                raise
            finally:
                nn.Module.load_weights = original_load_weights


def _load_mlx_vlm_distributed(
    model_name,
    model_type,
    *,
    pipeline_group=None,
    tensor_group=None,
    hf_token=None,
    revision=None,
    config_override_data=None,
):
    pipeline_group, tensor_group = _mlx_active_distributed_groups(
        pipeline_group,
        tensor_group,
    )
    mode = "tensor" if tensor_group is not None else "pipeline"
    model_label = f"'{model_name}'"
    if model_type:
        model_label = f"{model_label} (model_type={model_type!r})"

    try:
        from mlx_vlm.utils import get_model_path, sharded_load
    except ImportError as error:
        raise ImportError(
            "Unsloth: distributed MLX VLM inference requires mlx-vlm with "
            "sharded_load support. Install or upgrade mlx-vlm on Apple Silicon."
        ) from error

    try:
        with _temporary_hf_token_env(hf_token):
            load_target = get_model_path(model_name, revision=revision)
            if config_override_data is not None:
                load_target = _materialize_mlx_vlm_config_data(
                    str(load_target),
                    config_override_data,
                )
                try:
                    model, processor = sharded_load(
                        load_target,
                        tensor_group=tensor_group,
                        pipeline_group=pipeline_group,
                    )
                except Exception:
                    shutil.rmtree(load_target, ignore_errors=True)
                    raise
                _keep_mlx_vlm_config_view_alive(model, load_target)
                return model, processor
            return sharded_load(
                load_target,
                tensor_group=tensor_group,
                pipeline_group=pipeline_group,
            )
    except ValueError as error:
        message = str(error)
        lower_message = message.lower()
        support_error = (
            "does not support" in lower_message
            or "not supported" in lower_message
            or "unsupported model type" in lower_message
        )
        if not support_error:
            raise
        raise ValueError(
            f"Unsloth: {model_label} does not support MLX {mode} parallel "
            f"VLM inference through mlx-vlm. {message}"
        ) from error
    except TypeError as error:
        message = str(error)
        if "tensor_group" not in message and "pipeline_group" not in message:
            raise
        raise ImportError(
            "Unsloth: distributed MLX VLM inference requires a newer mlx-vlm "
            "with sharded_load(repo, tensor_group=..., pipeline_group=...) support."
        ) from error


def _read_json_file(path):
    """Read a JSON object, returning an empty dict for missing/bad sidecars."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}
    # Non-object JSON (list/str/number/null) is as malformed; callers expect a dict.
    return data if isinstance(data, dict) else {}


def _resolve_mlx_vlm_processor_class(
    model_type,
    processor_class_name,
    processor_class=None,
):
    """Resolve a custom mlx-vlm or Transformers processor class by name."""
    if not processor_class_name:
        return None

    raw_model_type = str(model_type or "")
    module_model_type = raw_model_type.replace("-", "_")
    module_types = []
    # Aliased model types live under their MODEL_REMAPPING target package.
    try:
        from mlx_vlm.utils import MODEL_REMAPPING

        remapped = MODEL_REMAPPING.get(raw_model_type)
        if remapped is None:
            remapped = MODEL_REMAPPING.get(module_model_type)
        if remapped:
            module_types.append(str(remapped).replace("-", "_"))
    except Exception:
        pass
    if module_model_type and module_model_type not in module_types:
        module_types.append(module_model_type)
    module_candidates = [
        name
        for module_type in module_types
        for name in (
            f"mlx_vlm.models.{module_type}",
            f"mlx_vlm.models.{module_type}.processing",
            f"mlx_vlm.models.{module_type}.processing_{module_type}",
            f"mlx_vlm.models.{module_type}.image_processing_{module_type}",
            f"mlx_vlm.models.{module_type}.audio_feature_extractor",
            f"mlx_vlm.models.{module_type}.feature_extraction_{module_type}",
            f"mlx_vlm.models.{module_type}.video_processing_{module_type}",
        )
    ]
    for base_class in getattr(processor_class, "__mro__", ()):
        module_name = getattr(base_class, "__module__", "")
        if not module_name.startswith("mlx_vlm.models."):
            continue
        for candidate in (module_name, module_name.rsplit(".", 1)[0]):
            if candidate not in module_candidates:
                module_candidates.append(candidate)
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        resolved_class = getattr(module, processor_class_name, None)
        if isinstance(resolved_class, type):
            return resolved_class

    try:
        import transformers
        resolved_class = getattr(transformers, processor_class_name, None)
        return resolved_class if isinstance(resolved_class, type) else None
    except Exception:
        return None


_VLM_PROCESSOR_COMPONENT_TYPES = {
    "image_processor": (("image_processor_type",), ("image_processor_type",)),
    "video_processor": (("video_processor_type",), ("video_processor_type",)),
    "feature_extractor": (("feature_extractor_type",), ("feature_extractor_type",)),
    "audio_processor": (
        ("audio_processor_type", "feature_extractor_type"),
        ("audio_processor_type",),
    ),
}
_VLM_PROCESSOR_COMPONENT_BASES = {
    "image_processor": (
        "transformers.image_processing_utils",
        "ImageProcessingMixin",
    ),
    "video_processor": (
        "transformers.video_processing_utils",
        "BaseVideoProcessor",
    ),
    "feature_extractor": (
        "transformers.feature_extraction_utils",
        "FeatureExtractionMixin",
    ),
    "audio_processor": (
        "transformers.feature_extraction_utils",
        "FeatureExtractionMixin",
    ),
    "tokenizer": (
        "transformers.tokenization_utils_base",
        "PreTrainedTokenizerBase",
    ),
}


def _vlm_processor_attributes(processor_class):
    get_attributes = getattr(processor_class, "get_attributes", None)
    if callable(get_attributes):
        try:
            attributes = get_attributes()
        except (AttributeError, TypeError, ValueError):
            pass
        else:
            return tuple(dict.fromkeys(
                name for name in attributes if isinstance(name, str)
            ))
    return tuple(dict.fromkeys(
        name
        for name in getattr(processor_class, "attributes", ())
        if isinstance(name, str)
    ))


def _is_native_mlx_vlm_factory(processor_class):
    from_pretrained = getattr(processor_class, "from_pretrained", None)
    factory = inspect.unwrap(getattr(from_pretrained, "__func__", from_pretrained))
    return getattr(factory, "__module__", "").startswith("mlx_vlm.models.")


def _declared_vlm_processor_components(
    processor_config,
    preprocessor_config,
    video_processor_config=None,
):
    """Return modality attribute -> class name declarations from sidecars."""
    sidecars = (processor_config, preprocessor_config, video_processor_config or {})
    components = {}
    for attribute_name, (nested_keys, flat_keys) in (
        _VLM_PROCESSOR_COMPONENT_TYPES.items()
    ):
        for config in sidecars:
            nested = config.get(attribute_name)
            class_name = None
            if isinstance(nested, dict):
                class_name = next(
                    (nested.get(key) for key in nested_keys if nested.get(key)),
                    None,
                )
            class_name = class_name or next(
                (config.get(key) for key in flat_keys if config.get(key)),
                None,
            )
            if isinstance(class_name, str) and class_name:
                components[attribute_name] = class_name
                break
    return components


def _matches_vlm_component_kind(attribute_name, argument):
    module_and_class = _VLM_PROCESSOR_COMPONENT_BASES.get(attribute_name)
    if module_and_class is None:
        return False
    try:
        component_base = getattr(
            importlib.import_module(module_and_class[0]),
            module_and_class[1],
        )
    except (ImportError, AttributeError):
        return False
    matches = (
        issubclass(argument, component_base)
        if isinstance(argument, type)
        else isinstance(argument, component_base)
    )
    if not matches or attribute_name != "image_processor":
        return matches
    try:
        video_base = getattr(
            importlib.import_module("transformers.video_processing_utils"),
            "BaseVideoProcessor",
        )
    except (ImportError, AttributeError):
        return True
    return not (
        issubclass(argument, video_base)
        if isinstance(argument, type)
        else isinstance(argument, video_base)
    )


def _vlm_processor_class_contract(
    processor_class,
    components,
    model_type,
):
    """Return custom lookup and exact component identities for one processor."""
    custom_lookup = {}
    accepted = {}
    attributes = set(_vlm_processor_attributes(processor_class)) | set(components)
    for attribute_name in attributes:
        component = components.get(attribute_name)
        class_names = getattr(processor_class, f"{attribute_name}_class", None)
        class_names = (
            list(class_names) if isinstance(class_names, tuple) else [class_names]
        )
        if component is not None:
            class_names.insert(0, component[0])
        declared_classes = []
        for class_name in class_names:
            if not isinstance(class_name, str) or not class_name:
                continue
            resolved_class = _resolve_mlx_vlm_processor_class(
                model_type,
                class_name,
                processor_class,
            )
            module_name = getattr(resolved_class, "__module__", "")
            if module_name.startswith("mlx_vlm.models."):
                custom_lookup[class_name] = resolved_class
            if resolved_class is not None and not module_name.startswith(
                "transformers.models.auto."
            ):
                declared_classes.append(resolved_class)

        accepted[attribute_name] = set(declared_classes)
    return custom_lookup, accepted


def _valid_vlm_processor_contract(
    processor,
    processor_class,
    processor_attributes,
    components,
    accepted_classes,
    *,
    allow_native,
):
    required = set(components)
    if "tokenizer" in processor_attributes:
        required.add("tokenizer")
    if any(getattr(processor, name, None) is None for name in required):
        return False

    native_factory = _is_native_mlx_vlm_factory(processor_class)
    modality_attributes = (
        processor_attributes | set(components)
    ) & set(_VLM_PROCESSOR_COMPONENT_TYPES)
    for attribute_name in modality_attributes:
        argument = getattr(processor, attribute_name, None)
        if argument is None or type(argument) in accepted_classes.get(
            attribute_name, ()
        ):
            continue
        if (
            allow_native
            and native_factory
            # Native factories may intentionally substitute a backend-safe
            # implementation of the same modality (for example, fast -> slow).
            and _matches_vlm_component_kind(attribute_name, argument)
        ):
            continue
        return False
    return True


def _vlm_processor_needs_repair(
    processor,
    processor_class,
    components,
):
    """Detect tokenizer-only or structurally incomplete processor results."""
    if processor is None:
        return True
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        if isinstance(processor, PreTrainedTokenizerBase):
            return True
    except ImportError:
        pass

    attributes = set(_vlm_processor_attributes(processor_class))
    required = set(components)
    if "tokenizer" in attributes:
        required.add("tokenizer")
    return any(getattr(processor, name, None) is None for name in required)


def _reconstruct_declared_vlm_processor(
    processor_class,
    model_path,
    components,
    contract,
    *,
    token=None,
    trust_remote_code=False,
    processor_kwargs=None,
):
    """Run a declared processor's native factory with local custom lookup."""
    from_pretrained = getattr(processor_class, "from_pretrained", None)
    original_lookup = getattr(processor_class, "get_possibly_dynamic_module", None)
    if not callable(from_pretrained) or not callable(original_lookup):
        return None

    custom_components, accepted_classes = contract
    processor_attribute_order = _vlm_processor_attributes(processor_class)
    processor_attributes = set(processor_attribute_order)

    def get_possibly_dynamic_module(class_name):
        component_class = custom_components.get(class_name)
        if component_class is not None:
            return component_class
        return original_lookup(class_name)

    base_check = getattr(processor_class, "check_argument_for_proper_class", None)

    def valid_component(attribute_name, argument):
        if type(argument) in accepted_classes.get(attribute_name, ()):
            return True
        return (
            attribute_name in _VLM_PROCESSOR_COMPONENT_TYPES
            # Apply the same native-factory substitution contract while the
            # temporary ProcessorMixin subclass validates constructor args.
            and _is_native_mlx_vlm_factory(processor_class)
            and _matches_vlm_component_kind(attribute_name, argument)
        )

    def check_argument_for_proper_class(self, attribute_name, argument):
        try:
            return base_check(self, attribute_name, argument)
        except TypeError:
            if valid_component(attribute_name, argument):
                return None
            raise

    subclass_attrs = {
        "__module__": processor_class.__module__,
        "get_possibly_dynamic_module": staticmethod(get_possibly_dynamic_module),
    }
    if callable(getattr(processor_class, "get_attributes", None)):
        subclass_attrs["get_attributes"] = classmethod(
            lambda _cls: list(processor_attribute_order)
        )
    for attribute_name, (class_name, _component_class) in components.items():
        if (
            attribute_name in processor_attributes
            and getattr(processor_class, f"{attribute_name}_class", None) is None
        ):
            subclass_attrs[f"{attribute_name}_class"] = class_name
    if callable(base_check):
        subclass_attrs["check_argument_for_proper_class"] = check_argument_for_proper_class
    temporary_class = type(processor_class.__name__, (processor_class,), subclass_attrs)

    factory_kwargs = dict(processor_kwargs or {})
    if trust_remote_code:
        factory_kwargs.setdefault("trust_remote_code", True)
    if token:
        factory_kwargs.setdefault("token", token)
    try:
        repaired = temporary_class.from_pretrained(model_path, **factory_kwargs)
    except Exception:
        return None
    if repaired is None:
        return None

    if not isinstance(repaired, temporary_class):
        return None
    for attribute_name in processor_attributes:
        argument = getattr(repaired, attribute_name, None)
        if argument is None:
            continue
        if not callable(base_check):
            continue
        try:
            temporary_class.check_argument_for_proper_class(
                repaired,
                attribute_name,
                argument,
            )
        except (TypeError, ValueError):
            return None
    if not _valid_vlm_processor_contract(
        repaired,
        processor_class,
        processor_attributes,
        components,
        accepted_classes,
        allow_native=True,
    ):
        return None

    if isinstance(repaired, temporary_class):
        try:
            repaired.__class__ = processor_class
        except TypeError:
            return None
    if type(repaired) is not processor_class:
        return None
    return repaired


def _set_mlx_vlm_processor_runtime_state(
    processor,
    model_path,
    eos_token_ids=None,
):
    """Attach mlx-vlm generation state to a newly reconstructed processor."""
    try:
        import mlx_vlm.utils as vlm_utils

        tokenizer = getattr(processor, "tokenizer", processor)
        detokenizer_class = vlm_utils.load_tokenizer(
            Path(model_path),
            return_tokenizer=False,
        )
        processor.detokenizer = detokenizer_class(tokenizer)
        final_eos_ids = eos_token_ids
        if final_eos_ids is None:
            final_eos_ids = getattr(tokenizer, "eos_token_ids", None)
        if final_eos_ids is None:
            final_eos_ids = getattr(tokenizer, "eos_token_id", None)
        criteria = vlm_utils.StoppingCriteria(final_eos_ids, tokenizer)
        if hasattr(processor, "tokenizer"):
            tokenizer.stopping_criteria = criteria
        else:
            processor.stopping_criteria = criteria
    except Exception:
        return False
    return True


def _recoverable_mlx_vlm_processor_error(error):
    message = str(error)
    if isinstance(error, TypeError):
        return (
            message.startswith("Received a ")
            and " for argument " in message
            and ", but a " in message
            and message.endswith(" was expected.")
        )
    return (
        isinstance(error, ValueError)
        and message.startswith("Could not find module ")
        and " in `transformers`." in message
    )


def _ensure_mlx_vlm_processor_repair():
    """Recover native processor resolution and component mismatch failures."""
    try:
        import mlx_vlm.utils as vlm_utils
    except ImportError:
        return

    load_processor = getattr(vlm_utils, "load_processor", None)
    if load_processor is None or getattr(
        load_processor, "_unsloth_processor_repair", False
    ):
        return

    @wraps(load_processor)
    def repairing_load_processor(
        model_path,
        add_detokenizer=True,
        eos_token_ids=None,
        **kwargs,
    ):
        try:
            return load_processor(
                model_path,
                add_detokenizer=add_detokenizer,
                eos_token_ids=eos_token_ids,
                **kwargs,
            )
        except (TypeError, ValueError) as error:
            if not _recoverable_mlx_vlm_processor_error(error):
                raise
            try:
                config = _read_json_file(os.path.join(str(model_path), "config.json"))
                repaired = _repair_degraded_vlm_processor(
                    None,
                    model_path,
                    config.get("model_type"),
                    token=kwargs.get("token"),
                    trust_remote_code=bool(kwargs.get("trust_remote_code", False)),
                    processor_kwargs=kwargs,
                    add_detokenizer=add_detokenizer,
                    eos_token_ids=eos_token_ids,
                )
            except Exception:
                repaired = None
            if repaired is None:
                raise
        return repaired

    repairing_load_processor._unsloth_processor_repair = True
    repairing_load_processor._unsloth_original = load_processor
    vlm_utils.load_processor = repairing_load_processor


_MLX_VLM_PT_ONLY_ERROR = (
    "Failed to process inputs with error: "
    "Only returning PyTorch tensors is currently supported."
)


def _convert_pt_vlm_output(value, return_tensors):
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.is_conj():
            tensor = tensor.resolve_conj()
        if tensor.is_neg():
            tensor = tensor.resolve_neg()
        if return_tensors in {"mlx", "mx"}:
            import mlx.core as mx
            from_dlpack = getattr(mx, "from_dlpack", None)
            if from_dlpack is not None:
                return from_dlpack(tensor.contiguous())
            if tensor.dtype is torch.bfloat16:
                tensor = tensor.float()
            return mx.array(tensor.numpy())
        if tensor.dtype is torch.bfloat16:
            tensor = tensor.float()
        return tensor.numpy()
    if isinstance(value, Mapping):
        converted_items = {
            key: _convert_pt_vlm_output(item, return_tensors)
            for key, item in value.items()
        }
        try:
            converted = copy.copy(value)
            for key, item in converted_items.items():
                converted[key] = item
            return converted
        except (AttributeError, TypeError):
            try:
                return type(value)(converted_items)
            except TypeError:
                return converted_items
    if isinstance(value, tuple):
        converted = tuple(
            _convert_pt_vlm_output(item, return_tensors) for item in value
        )
        if hasattr(value, "_fields"):
            return type(value)(*converted)
        if type(value) is not tuple:
            try:
                return type(value)(converted)
            except TypeError:
                pass
        return converted
    if isinstance(value, list):
        converted = [_convert_pt_vlm_output(item, return_tensors) for item in value]
        if type(value) is list:
            return converted
        try:
            copied = copy.copy(value)
            copied[:] = converted
            return copied
        except (AttributeError, TypeError):
            return converted
    return value


def _ensure_mlx_vlm_pt_output_fallback():
    """Retry processors that only implement PyTorch tensor output."""
    try:
        import mlx_vlm.utils as vlm_utils
    except ImportError:
        return

    process_inputs = getattr(vlm_utils, "process_inputs_with_fallback", None)
    if process_inputs is None or getattr(process_inputs, "_unsloth_pt_fallback", False):
        return

    @wraps(process_inputs)
    def process_inputs_with_pt_fallback(
        processor,
        prompts,
        images,
        audio,
        add_special_tokens=False,
        return_tensors="mlx",
        **kwargs,
    ):
        try:
            return process_inputs(
                processor,
                prompts,
                images,
                audio,
                add_special_tokens=add_special_tokens,
                return_tensors=return_tensors,
                **kwargs,
            )
        except ValueError as error:
            if (
                return_tensors not in {"mlx", "mx", "np"}
                or str(error) != _MLX_VLM_PT_ONLY_ERROR
            ):
                raise
        outputs = process_inputs(
            processor,
            prompts,
            images,
            audio,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
            **kwargs,
        )
        return _convert_pt_vlm_output(outputs, return_tensors)

    process_inputs_with_pt_fallback._unsloth_pt_fallback = True
    process_inputs_with_pt_fallback._unsloth_original = process_inputs
    vlm_utils.process_inputs_with_fallback = process_inputs_with_pt_fallback


def _build_vlm_image_processor_from_config(
    model_path, processor_config, preprocessor_config, model_type=None,
):
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
        # mlx-vlm models can ship their own image processor classes.
        try:
            image_processor_class = _resolve_mlx_vlm_processor_class(
                model_type,
                image_processor_type,
            )
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
    processor_kwargs=None,
    add_detokenizer=True,
    eos_token_ids=None,
):
    """Rebuild VLM processors when mlx-vlm falls back to tokenizer-only.

    mlx-vlm degrades to a tokenizer-only processor when AutoImageProcessor
    fails; rebuild from the source sidecar configs so downstream saves keep
    real multimodal processor metadata.
    """
    if not model_path or not os.path.isdir(str(model_path)):
        return processor

    processor_config = _read_json_file(
        os.path.join(str(model_path), "processor_config.json")
    )
    preprocessor_config = _read_json_file(
        os.path.join(str(model_path), "preprocessor_config.json")
    )
    video_processor_config = _read_json_file(
        os.path.join(str(model_path), "video_preprocessor_config.json")
    )
    tokenizer_config = _read_json_file(
        os.path.join(str(model_path), "tokenizer_config.json")
    )
    processor_class_name = (
        processor_config.get("processor_class")
        or preprocessor_config.get("processor_class")
        or video_processor_config.get("processor_class")
        or tokenizer_config.get("processor_class")
    )
    if not isinstance(processor_class_name, str) or not processor_class_name:
        return processor

    processor_class = _resolve_mlx_vlm_processor_class(model_type, processor_class_name)
    if processor_class is None:
        return processor
    declared_components = _declared_vlm_processor_components(
        processor_config,
        preprocessor_config,
        video_processor_config,
    )
    processor_attributes = set(_vlm_processor_attributes(processor_class))
    components = {}
    for attribute_name, class_name in declared_components.items():
        component_class = _resolve_mlx_vlm_processor_class(
            model_type,
            class_name,
            processor_class,
        )
        custom_component = getattr(component_class, "__module__", "").startswith(
            "mlx_vlm.models."
        )
        if attribute_name in processor_attributes or custom_component:
            components[attribute_name] = (class_name, component_class)
    if not _vlm_processor_needs_repair(processor, processor_class, components):
        return processor

    contract = _vlm_processor_class_contract(processor_class, components, model_type)
    repaired = _reconstruct_declared_vlm_processor(
        processor_class,
        model_path,
        components,
        contract,
        token=token,
        trust_remote_code=trust_remote_code,
        processor_kwargs=processor_kwargs,
    )
    accepted_classes = contract[1]
    if repaired is None and processor is not None and "image_processor" in components:
        image_processor = _build_vlm_image_processor_from_config(
            model_path, processor_config, preprocessor_config, model_type,
        )
        tokenizer = getattr(processor, "tokenizer", None) or processor
        chat_template = getattr(processor, "chat_template", None)
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
                repaired = None
        except Exception:
            repaired = None

        if (
            repaired is None
            or type(repaired) is not processor_class
            or _vlm_processor_needs_repair(repaired, processor_class, components)
            or not _valid_vlm_processor_contract(
                repaired,
                processor_class,
                processor_attributes,
                components,
                accepted_classes,
                allow_native=False,
            )
        ):
            repaired = None

    if repaired is None:
        return processor

    target_tokenizer = getattr(repaired, "tokenizer", None)
    source_tokenizer = getattr(processor, "tokenizer", None) or processor
    chat_template = getattr(processor, "chat_template", None) or getattr(
        source_tokenizer, "chat_template", None
    )
    if chat_template is not None and getattr(repaired, "chat_template", None) is None:
        repaired.chat_template = chat_template
    if (
        chat_template is not None
        and target_tokenizer is not None
        and getattr(target_tokenizer, "chat_template", None) is None
    ):
        target_tokenizer.chat_template = chat_template
    if eos_token_ids is None and source_tokenizer is not None:
        source_criteria = getattr(source_tokenizer, "stopping_criteria", None)
        eos_token_ids = getattr(source_criteria, "eos_token_ids", None)
    if add_detokenizer and not _set_mlx_vlm_processor_runtime_state(
        repaired,
        model_path,
        eos_token_ids=eos_token_ids,
    ):
        return processor
    return repaired


def _build_vlm_model_types():
    """Frozenset of model_type strings mlx_vlm supports (discovered via
    pkgutil + MODEL_REMAPPING); cached at module level by _is_vlm()."""
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
        for src, tgt in MODEL_REMAPPING.items():
            types_set.add(src)
            types_set.add(tgt)
    except (ImportError, AttributeError):
        pass

    return frozenset(types_set)


def _fix_missing_no_grad(model):
    """Ensure every submodule has _no_grad / _training.

    Works around upstream modules using __new__ without __init__ (e.g. gemma4
    AudioRelativePositionEmbedding).
    """
    import mlx.nn as nn
    for _, mod in model.named_modules():
        if isinstance(mod, nn.Module):
            if not hasattr(mod, "_no_grad"):
                object.__setattr__(mod, "_no_grad", set())
            if not hasattr(mod, "_training"):
                object.__setattr__(mod, "_training", True)


class _TrainingKVStore:
    """Minimal KV store for Gemma4 KV-sharing during training.

    Gemma4 E2B/E4B shared layers borrow K/V from earlier "store" layers via
    the cache; with cache=None they'd recompute K/V from the wrong hidden
    states. This lets store layers write and shared layers read, with no
    autoregressive offset tracking. Implements just the KVCache surface
    Attention.__call__ needs: offset (0), state, update_and_fetch.
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


def _gemma4_has_native_shared_kv(backbone):
    """Return whether mlx-vlm already threads Gemma4 shared K/V for training."""
    layers = getattr(backbone, "layers", None) or []
    previous_kvs = getattr(backbone, "previous_kvs", None)
    if (
        isinstance(previous_kvs, (list, tuple))
        and layers
        and len(previous_kvs) == len(layers)
    ):
        return True

    try:
        call_params = inspect.signature(backbone.__class__.__call__).parameters
    except (TypeError, ValueError):
        return False

    return "shared_kv_sink" in call_params


def _fix_gemma4_kv_sharing(model):
    """Fix legacy Gemma4 KV-shared layers producing wrong K/V during training.

    Gemma4 E2B/E4B have num_kv_shared_layers shared attention layers that
    borrow K/V from earlier "store" layers via the KV cache. When cache=None
    (training), legacy shared layers recompute K/V from the wrong hidden state.

    mlx-vlm 0.5.0+ threads shared_kv natively; only older backbones need the
    _TrainingKVStore cache shim.
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

    if _gemma4_has_native_shared_kv(backbone):
        return  # Native mlx-vlm shared_kv support

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

    mlx-vlm's Qwen3.5 attention does `cache.offset + 1` without a None check,
    but training always passes cache=None. Patch __call__ to compute
    position_ids from scratch in that case.
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

    def patched_attn_call(self, x, mask=None, cache=None, position_ids=None, **kwargs):
        # Compute position_ids when training (cache=None, position_ids=None).
        if cache is None and position_ids is None:
            import mlx.core as mx
            L = x.shape[1]
            position_ids = mx.arange(L)
            position_ids = mx.expand_dims(position_ids, axis=0)
            position_ids = mx.tile(position_ids, (3, 1, 1))
        return original_attn_call(self, x, mask=mask, cache=cache, position_ids=position_ids, **kwargs)

    attn_cls.__call__ = patched_attn_call
    attn_cls._unsloth_cache_patched = True
    print("Unsloth: Fixed Qwen3.5 attention for training (cache=None).")


def _fix_gemma3_vision_post_layernorm_eps(model):
    """Match HF Gemma3/SigLIP final vision LayerNorm epsilon.

    mlx-vlm constructs ``post_layernorm`` with MLX's default eps=1e-5, while
    the checkpoint config and Transformers path use vision_config.layer_norm_eps
    (1e-6 for Gemma3). The mismatch only appears after the full vision tower,
    so it is easy to misdiagnose as attention drift.
    """

    vision_tower = getattr(model, "vision_tower", None)
    vision_model = getattr(vision_tower, "vision_model", None)
    post_layernorm = getattr(vision_model, "post_layernorm", None)
    if post_layernorm is None or not hasattr(post_layernorm, "eps"):
        return False

    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    if vision_config is None and isinstance(config, dict):
        vision_config = config.get("vision_config")

    eps = None
    if isinstance(vision_config, dict):
        eps = vision_config.get("layer_norm_eps")
    elif vision_config is not None:
        eps = getattr(vision_config, "layer_norm_eps", None)
    if eps is None:
        return False

    eps = float(eps)
    if float(getattr(post_layernorm, "eps")) == eps:
        return False

    post_layernorm.eps = eps
    model._unsloth_gemma3_vision_post_layernorm_eps = eps
    return True


def _fix_gemma3_text_rmsnorm_fp32(model=None):
    """Match HF Gemma3 text RMSNorm: fp32 math, then cast back to activation dtype."""

    try:
        import mlx.core as mx
        language_module = importlib.import_module("mlx_vlm.models.gemma3.language")
    except Exception:
        return False

    rmsnorm_cls = getattr(language_module, "RMSNorm", None)
    if rmsnorm_cls is None:
        return False
    if getattr(rmsnorm_cls, "_unsloth_fp32_rmsnorm_patched", False):
        if model is not None:
            model._unsloth_gemma3_text_rmsnorm_fp32 = True
        return True

    def patched_rmsnorm_call(self, x):
        orig_dtype = x.dtype
        x_f = x.astype(mx.float32)
        y = x_f * mx.rsqrt(mx.mean(x_f * x_f, axis=-1, keepdims=True) + self.eps)
        if "weight" in self:
            y = y * (1.0 + self.weight.astype(mx.float32))
        return y.astype(orig_dtype)

    try:
        rmsnorm_cls.__call__ = patched_rmsnorm_call
        rmsnorm_cls._unsloth_fp32_rmsnorm_patched = True
    except Exception:
        return False
    if model is not None:
        model._unsloth_gemma3_text_rmsnorm_fp32 = True
    return True


def _fix_gemma3_vision_attention_fp32_sdpa(model=None):
    """Run Gemma3 vision SDPA in fp32, then cast back before the output proj."""

    try:
        import mlx.core as mx
        vision_module = importlib.import_module("mlx_vlm.models.gemma3.vision")
    except Exception:
        return False

    attention_cls = getattr(vision_module, "Attention", None)
    if attention_cls is None:
        return False
    if getattr(attention_cls, "_unsloth_fp32_sdpa_patched", False):
        return False

    def patched_attention_call(self, x, mask=None):
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        orig_dtype = queries.dtype

        num_heads = self.num_heads
        batch_size, query_length, hidden_size = queries.shape
        _, key_length, _ = keys.shape
        queries = queries.reshape(
            batch_size, query_length, num_heads, -1,
        ).transpose(0, 2, 1, 3).astype(mx.float32)
        keys = keys.reshape(
            batch_size, key_length, num_heads, -1,
        ).transpose(0, 2, 1, 3).astype(mx.float32)
        values = values.reshape(
            batch_size, key_length, num_heads, -1,
        ).transpose(0, 2, 1, 3).astype(mx.float32)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask,
        )
        output = output.astype(orig_dtype)
        output = output.transpose(0, 2, 1, 3).reshape(
            batch_size, query_length, hidden_size,
        )
        return self.out_proj(output)

    try:
        attention_cls.__call__ = patched_attention_call
        attention_cls._unsloth_fp32_sdpa_patched = True
    except Exception:
        return False
    if model is not None:
        model._unsloth_gemma3_vision_attention_fp32_sdpa = True
    return True


def _fix_gemma3_vision_mlp_fp32_activation(model=None):
    """Match CUDA SigLIP GELU: compute activation in fp32, then cast back."""

    try:
        import mlx.core as mx
        vision_module = importlib.import_module("mlx_vlm.models.gemma3.vision")
    except Exception:
        return False

    mlp_cls = getattr(vision_module, "MLP", None)
    if mlp_cls is None:
        return False
    if getattr(mlp_cls, "_unsloth_fp32_activation_patched", False):
        if model is not None:
            model._unsloth_gemma3_vision_mlp_fp32_activation = True
        return True

    def patched_mlp_call(self, x):
        x = self.fc1(x)
        orig_dtype = x.dtype
        x = self.activation_fn(x.astype(mx.float32)).astype(orig_dtype)
        x = self.fc2(x)
        return x

    try:
        mlp_cls.__call__ = patched_mlp_call
        mlp_cls._unsloth_fp32_activation_patched = True
    except Exception:
        return False
    if model is not None:
        model._unsloth_gemma3_vision_mlp_fp32_activation = True
    return True


def _fix_gemma3_language_mlp_fp32_activation(model=None):
    """Compute the language-tower GEGLU activation in fp32, then cast back.

    Mirrors _fix_gemma3_vision_mlp_fp32_activation for the language tower.
    Without this, the gelu_approx forward intermediates (x^3 term in the
    tanh-form) overflow in bf16 / fp16 for some prompts and produce NaN
    gradients in the backward pass — observed at language layers 0-11 of
    gemma-3-4b-it during multimodal LoRA finetuning. The vision-side fix
    above is not sufficient because the language MLP has its own activation
    site that does not share that code path.
    """

    try:
        import mlx.core as mx
        import mlx.nn as nn
        language_module = importlib.import_module("mlx_vlm.models.gemma3.language")
    except Exception:
        return False

    mlp_cls = getattr(language_module, "MLP", None)
    if mlp_cls is None:
        return False
    if getattr(mlp_cls, "_unsloth_fp32_activation_patched", False):
        if model is not None:
            model._unsloth_gemma3_language_mlp_fp32_activation = True
        return True

    def patched_mlp_call(self, x):
        orig_dtype = x.dtype
        gate_pre = self.gate_proj(x)
        gate_post = nn.gelu_approx(gate_pre.astype(mx.float32)).astype(orig_dtype)
        up = self.up_proj(x)
        return self.down_proj(gate_post * up)

    try:
        mlp_cls.__call__ = patched_mlp_call
        mlp_cls._unsloth_fp32_activation_patched = True
    except Exception:
        return False
    if model is not None:
        model._unsloth_gemma3_language_mlp_fp32_activation = True
    return True


def _fix_gemma3_vision_encoder_fp32_layernorm(model=None):
    """Match CUDA SigLIP LayerNorm math in fp32 while preserving bf16 activations."""

    try:
        import mlx.core as mx
        vision_module = importlib.import_module("mlx_vlm.models.gemma3.vision")
    except Exception:
        return False

    encoder_layer_cls = getattr(vision_module, "EncoderLayer", None)
    if encoder_layer_cls is None:
        return False
    if getattr(encoder_layer_cls, "_unsloth_fp32_layernorm_patched", False):
        if model is not None:
            model._unsloth_gemma3_vision_encoder_fp32_layernorm = True
        return True

    def torch_like_layer_norm(norm, x):
        orig_dtype = x.dtype
        x_f = x.astype(mx.float32)
        mean = mx.mean(x_f, axis=-1, keepdims=True)
        centered = x_f - mean
        var = mx.mean(centered * centered, axis=-1, keepdims=True)
        y = centered * mx.rsqrt(var + norm.eps)
        if "weight" in norm:
            y = y * norm.weight.astype(mx.float32)
        if "bias" in norm:
            y = y + norm.bias.astype(mx.float32)
        return y.astype(orig_dtype)

    def patched_encoder_layer_call(self, x, mask=None):
        r = self.self_attn(torch_like_layer_norm(self.layer_norm1, x), mask)
        h = x + r
        r = self.mlp(torch_like_layer_norm(self.layer_norm2, h))
        return h + r

    try:
        encoder_layer_cls.__call__ = patched_encoder_layer_call
        encoder_layer_cls._unsloth_fp32_layernorm_patched = True
    except Exception:
        return False
    if model is not None:
        model._unsloth_gemma3_vision_encoder_fp32_layernorm = True
    return True


def _fix_gemma3_vision_post_layernorm_fp32(model=None):
    """Run Gemma3 final SigLIP vision LayerNorm in fp32, then cast back."""

    try:
        import mlx.core as mx
        vision_module = importlib.import_module("mlx_vlm.models.gemma3.vision")
    except Exception:
        return False

    siglip_cls = getattr(vision_module, "SigLipVisionModel", None)
    if siglip_cls is None:
        return False
    if getattr(siglip_cls, "_unsloth_fp32_post_layernorm_patched", False):
        if model is not None:
            model._unsloth_gemma3_vision_post_layernorm_fp32 = True
        return True

    def torch_like_layer_norm(norm, x):
        orig_dtype = x.dtype
        x_f = x.astype(mx.float32)
        mean = mx.mean(x_f, axis=-1, keepdims=True)
        centered = x_f - mean
        var = mx.mean(centered * centered, axis=-1, keepdims=True)
        y = centered * mx.rsqrt(var + norm.eps)
        if "weight" in norm:
            y = y * norm.weight.astype(mx.float32)
        if "bias" in norm:
            y = y + norm.bias.astype(mx.float32)
        return y.astype(orig_dtype)

    def patched_siglip_call(self, x, output_hidden_states=None):
        x = self.embeddings(x)
        encoder_outputs = self.encoder(
            x=x, output_hidden_states=output_hidden_states, mask=None,
        )
        pooler_output = torch_like_layer_norm(self.post_layernorm, encoder_outputs[0])
        return pooler_output, x, encoder_outputs[-1]

    try:
        siglip_cls.__call__ = patched_siglip_call
        siglip_cls._unsloth_fp32_post_layernorm_patched = True
    except Exception:
        return False
    if model is not None:
        model._unsloth_gemma3_vision_post_layernorm_fp32 = True
    return True


def _fix_gemma3_multimodal_image_feature_scale(model=None):
    """Use text embedding width when compensating Gemma3 image feature scaling."""

    try:
        import mlx.core as mx
        gemma3_module = importlib.import_module("mlx_vlm.models.gemma3.gemma3")
    except Exception:
        return False

    model_cls = getattr(gemma3_module, "Model", None)
    masked_scatter = getattr(gemma3_module, "masked_scatter", None)
    if model_cls is None or masked_scatter is None:
        return False
    if getattr(model_cls, "_unsloth_image_feature_scale_patched", False):
        if model is not None:
            model._unsloth_gemma3_image_feature_scale = "text_embed_dim"
        return True

    def prepare_inputs_for_multimodal(
        hidden_size,
        pad_token_id,
        image_token_index,
        image_features,
        inputs_embeds,
        input_ids,
        attention_mask,
    ):
        del hidden_size
        embed_dim = image_features.shape[-1]
        batch_size, sequence_length = input_ids.shape
        # Gemma3's language model scales all inputs_embeds by sqrt(text hidden
        # size). Compensate image features with the actual embedding width, not
        # the top-level multimodal config hidden_size.
        scaled_image_features = image_features / (embed_dim**0.5)
        final_embedding = mx.zeros((batch_size, sequence_length, embed_dim))

        pad_token_id = pad_token_id if pad_token_id is not None else 0
        text_mask = (input_ids != image_token_index) & (input_ids != pad_token_id)
        image_mask = input_ids == image_token_index
        pad_mask = input_ids == pad_token_id

        text_mask_expanded = mx.repeat(mx.expand_dims(text_mask, -1), embed_dim, axis=-1)
        pad_mask_expanded = mx.repeat(mx.expand_dims(pad_mask, -1), embed_dim, axis=-1)
        image_mask_expanded = mx.repeat(mx.expand_dims(image_mask, -1), embed_dim, axis=-1)

        final_embedding = mx.where(text_mask_expanded, inputs_embeds, final_embedding)
        final_embedding = mx.where(
            pad_mask_expanded, mx.zeros_like(final_embedding), final_embedding,
        )
        final_embedding = masked_scatter(
            final_embedding, image_mask_expanded, scaled_image_features,
        )

        attention_mask_expanded_1 = mx.expand_dims(attention_mask, 1)
        attention_mask_expanded_2 = mx.expand_dims(attention_mask, 2)
        final_attention_mask_4d = mx.expand_dims(
            attention_mask_expanded_1 * attention_mask_expanded_2,
            1,
        )
        return final_embedding.astype(inputs_embeds.dtype), final_attention_mask_4d

    try:
        model_cls.prepare_inputs_for_multimodal = staticmethod(
            prepare_inputs_for_multimodal,
        )
        model_cls._unsloth_image_feature_scale_patched = True
    except Exception:
        return False
    if model is not None:
        model._unsloth_gemma3_image_feature_scale = "text_embed_dim"
    return True
def _disable_fused_mrope(model):
    """Flip fused_apply off so MRoPE training uses the differentiable
    cos/sin fallback; the fused Metal kernel has no VJP."""
    count = 0
    try:
        modules = model.modules()
    except Exception:
        return
    for module in modules:
        if getattr(module, "fused_apply", False):
            module.fused_apply = False
            count += 1
    if count:
        print(f"Unsloth: Disabled fused MRoPE kernel on {count} modules for training (no VJP).")


def _safe_getsource(obj) -> str:
    try:
        return inspect.getsource(obj)
    except Exception:
        return ""


def _has_multimodal_strip_sanitize(model_or_cls) -> bool:
    """Whether a sanitize path strips multimodal towers, a generic signal for
    "text-only load of a multimodal wrapper" (vs hardcoding families)."""

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
    """Whether a multimodal wrapper should stay on the VLM load path.

    Some repos are multimodal wrappers whose mlx_lm text path works only by
    stripping modality towers in `sanitize()`, meaning it reconstructs a
    different object graph than the checkpoint. Keeping the VLM path is more
    robust than a per-family sanitizer workaround.
    """

    if not _is_vlm(config):
        return False

    cls = _get_mlx_lm_model_class(model_type)
    if cls is None:
        return False

    return _has_multimodal_strip_sanitize(cls)


def _ensure_safe_text_wrapper_sanitize(model_type: str) -> None:
    """Patch nested-weight sanitize assumptions for text-only multimodal loads.

    Some mlx_lm wrappers unflatten then blindly index `weights["model"]`, which
    breaks when towers sit at the top level instead. Patch by behavior, not by
    one architecture, so any loader with the same assumption is handled.
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


def _resolve_mlx_vlm_model_class(model_type):
    """Resolve the mlx_vlm ``Model`` class for a model_type (honoring remaps)."""
    if not model_type:
        return None
    module_type = str(model_type).replace("-", "_")
    try:
        from mlx_vlm.utils import MODEL_REMAPPING
        remapped = MODEL_REMAPPING.get(model_type, MODEL_REMAPPING.get(module_type))
        if remapped:
            module_type = str(remapped).replace("-", "_")
    except Exception:
        pass
    for candidate in (
        f"mlx_vlm.models.{module_type}.{module_type}",
        f"mlx_vlm.models.{module_type}",
    ):
        try:
            module = importlib.import_module(candidate)
        except Exception:
            continue
        cls = getattr(module, "Model", None)
        if cls is not None:
            return cls
    return None


def _lookup_module_array(root, dotted_key):
    """Resolve a parameter (mx.array) by its checkpoint dotted key, else None."""
    obj = root
    for part in dotted_key.split("."):
        if isinstance(obj, Mapping):
            if part not in obj:
                return None
            obj = obj[part]
        elif isinstance(obj, (list, tuple)):
            if not part.isdigit() or int(part) >= len(obj):
                return None
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


# Inverse of the unconditional PyTorch->MLX conv transpose mlx-vlm applies in
# sanitize(): Conv1d transpose(0, 2, 1) is self-inverse, Conv2d transpose(
# 0, 2, 3, 1) inverts to (0, 3, 1, 2).
_MLX_CONV_TRANSPOSE_INVERSE = {3: (0, 2, 1), 4: (0, 3, 1, 2)}


def _ensure_audio_conv_sanitize(model_type: str) -> None:
    """Guard mlx-vlm audio-tower conv sanitize against a double transpose.

    Pre-converted MLX checkpoints (e.g. mlx-community/gemma-4-e2b-it-4bit) store
    the audio SubSampleConvProjection Conv2d weights and Conformer depthwise
    Conv1d weights already in MLX channel-last layout. mlx-vlm's ``sanitize``
    unconditionally re-applies the PyTorch->MLX transpose, corrupting the shape
    and failing the load with an HTTP 500 shape mismatch (e.g. "Expected shape
    (128, 3, 3, 1) but received shape (128, 3, 1, 3)").

    When a checkpoint weight already matches the instantiated module's
    channel-last target shape, pre-apply the inverse permutation so the upstream
    transpose round-trips to the correct layout instead of double transposing.
    We only intervene on an exact target-shape match, so raw PyTorch checkpoints
    (which genuinely need the transpose) are untouched. Patched by behavior
    (sanitize source signature), not by hardcoding a single family.
    """
    if not model_type or model_type in _AUDIO_CONV_SANITIZE_PATCHED:
        return
    cls = _resolve_mlx_vlm_model_class(model_type)
    if cls is None:
        return
    sanitize = getattr(cls, "sanitize", None)
    if sanitize is None:
        return
    source = _safe_getsource(sanitize)
    if not source:
        return
    compact = re.sub(r"\s+", "", source)
    conv2d_transpose = "subsample_conv_projection" in source and "transpose(0,2,3,1)" in compact
    conv1d_transpose = "depthwise_conv1d.weight" in source and "transpose(0,2,1)" in compact
    if not (conv2d_transpose or conv1d_transpose):
        return

    original_sanitize = sanitize

    def _is_audio_conv_key(key, ndim):
        if ndim == 4 and "subsample_conv_projection" in key and "conv.weight" in key:
            return conv2d_transpose
        if ndim == 3 and "depthwise_conv1d.weight" in key:
            return conv1d_transpose
        return False

    def patched_sanitize(self, weights):
        prepared = {}
        for key, value in weights.items():
            ndim = getattr(value, "ndim", 0)
            inverse = _MLX_CONV_TRANSPOSE_INVERSE.get(ndim)
            if inverse is not None and _is_audio_conv_key(key, ndim):
                target = _lookup_module_array(self, key)
                target_shape = tuple(getattr(target, "shape", ()) or ())
                if target_shape and target_shape == tuple(value.shape):
                    # Already channel-last: undo the upcoming upstream transpose.
                    value = value.transpose(*inverse)
            prepared[key] = value
        return original_sanitize(self, prepared)

    cls.sanitize = patched_sanitize
    _AUDIO_CONV_SANITIZE_PATCHED.add(model_type)


def _fp16_needs_bf16_modules(model):
    """Modules that should stay bf16 under fp16 training.

    Some Pixtral/Mistral3 VLMs emit vision features above fp16's range (>65,504)
    before projection, so `model.set_dtype(mx.float16)` overflows in
    get_input_embeddings. Text-only loads of multimodal wrappers are also shaky
    in fp16; detect them by the strip-tower sanitize path and keep the text
    backbone bf16.
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
            # Match Torch: bf16 full finetuning stays bf16 unless
            # float32_mixed_precision=True is explicit.
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
    # CUDA bnb NF4 parity (diagnostic): quantize then dequantize; not memory-saving.
    "nf4_dense": (64, 4),
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
    import mlx.nn as nn
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
        # isinstance, not an exact class-name match: a training-time subclass of
        # the quantized layer (e.g. NEFTune's _NEFTuneEmbed) must still be
        # recognised, else embed_tokens is silently dropped from the map.
        if not isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
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
    # mlx-lm LoRA/DoRA wrappers store the wrapped base layer under ".linear"
    # (LoRALinear) or ".embedding" (LoRAEmbedding/DoRAEmbedding). Adapter
    # metadata must describe the underlying base path so validation can run
    # before load_adapters re-wraps the module.
    for suffix in (".linear", ".embedding"):
        if path.endswith(suffix):
            return path[:-len(suffix)]
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
    """Normalize stored module paths to a list of non-empty strings.

    Accepts str, list/tuple/set, dict, and pathlib.Path; a bare string is
    wrapped (not iterated char-by-char).
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
    """Read the rank from the saved LoRA tensor for `module_path`.

    Compat shim for legacy adapters that saved module paths without
    rank/scale/dropout. Returns None if the file/path is absent or imports fail.
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
    # Suffixes whose 2-D shape is (rank, in_dims): mlx-lm's `lora_a.weight` and
    # PEFT-style `lora_A` / `lora_A.weight`. Raw lowercase `lora_a` keeps the
    # older (in_dims, rank) shape (default shape[-1] branch). `.lora_embedding_a*`
    # is excluded: LoRAEmbedding saves A as (num_embeddings, rank).
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
                # MoE/switch (experts, rank, in_dims): rank is shape[-2].
                if len(shape) >= 3:
                    return int(shape[-2])
                if suffix in _rank_first_2d_suffixes:
                    return int(shape[0])
                return int(shape[-1])
    except Exception:
        return None
    return None


def _apply_lora_at_paths(model, module_paths, adapter_cfg, adapter_weights_file=None):
    """Recreate LoRA/DoRA at saved module paths so vision/projector and
    MoE/embedding LoRA survive reload (mlx-lm's load_adapters rebuilds only the
    language tower).

    Returns the number of wrappers attached. `adapter_weights_file` is a
    last-resort rank source for legacy adapters lacking rank/scale/dropout.
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

    # DoRA path: refuse silent downgrade to plain LoRA; the per-module loop
    # raises ImportError if it needs a missing DoRA class.
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

    # Defer rank validation until a module needs wrapping, so a legacy adapter
    # with no rank (but already wrapped by load_adapters) doesn't crash here.
    _metadata = {"rank": None, "scale": None, "dropout": None}

    def _ensure_metadata(module_path=None):
        if _metadata["rank"] is not None:
            return
        lora_params = adapter_cfg.get("lora_parameters") or {}
        raw_rank = lora_params.get("rank", adapter_cfg.get("rank"))
        if raw_rank is None and module_path is not None:
            # Legacy fallback: recover rank from the saved tensor shape;
            # scale/dropout stay at 1.0 / 0.0.
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
        # Re-raise conversion errors with the "Unsloth MLX:" prefix the outer
        # catch preserves; otherwise a plain `int()` ValueError on malformed
        # metadata gets swallowed into a silent standard-load.
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
    # Track per-path skip reasons for the final warning (the saved-vs-live key
    # diff is the gate that actually raises).
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
                # Fail loud: a plain-LoRA downgrade drops the saved DoRA `.m`
                # tensor on strict=False.
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
            # mlx-lm has no DoRA on switch layers; fail loud, don't drop saved
            # switch LoRA tensors.
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
                # Fail loud for embeddings too: lora_cls=None would drop saved
                # embedding LoRA tensors on strict=False.
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
        # Resolve numeric path segments (e.g. `...layers.0`) via parent[int(seg)]
        # then getattr; same pattern on the leaf so list-indexed wrappers install.
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

    Compares presence AND shape so a wrong-rank live wrapper (e.g. default
    rank=8 over saved rank-4) counts as missing. Never blocks the following
    load_weights() (exceptions become a skip warning). Returns the sorted
    missing-key list so the fallback can raise; `[]` means clean or skipped.
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
        # Presence AND shape: a pure key-set diff would call a wrong-rank
        # wrapper "fully bound" and strict=False would drop the saved tensors.
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
    # Manual-rejection phrasings; only treated as signature mismatches when the
    # error also names one of our kwargs, so "rank 4 not supported" can't
    # downgrade rank.
    "not accepted",
    "not supported",
)


def _is_from_base_kwarg_typeerror(exc, kwarg=None, kwargs=None):
    """True when `exc` is a `from_base()` signature mismatch (older mlx-lm
    rejecting r/scale/dropout), not an internal wrapper TypeError.

    Falling back on an unrelated TypeError would downgrade rank to r=8 and
    mis-bind a wrong-rank wrapper.
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

    # Require the kwarg as a whole word so "r" doesn't match "rank"/"are" in
    # unrelated errors.
    return any(
        re.search(rf"(?<![\w]){re.escape(k)}(?![\w])", msg)
        for k in kwargs_lower
    )


def _no_rank_fallback_or_fail(from_base, module, rank):
    """Call `from_base(module)` only when rank is the upstream default (8); any
    other rank must not downgrade or strict=False drops the saved tensors."""
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

    Retries only on signature-mismatch TypeErrors; internal wrapper TypeErrors
    propagate so rank is never downgraded to r=8.
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
    """Monkey-patch mlx_lm LoRA/DoRA `from_base()` to accept scale/dropout on
    older wheels.

    The canonical walk calls `from_base(..., scale=..., dropout=...)` with no
    fallback, so an older wheel would yield a partial-LoRA model. Idempotent
    via `_unsloth_from_base_compat`.
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

            # Stubs without from_base: skip; the upstream walk surfaces the
            # AttributeError.
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
                # Pass-through on new mlx-lm; else fall back like
                # _lora_from_base_compat. Positional order MUST match upstream
                # `from_base(linear, r=8, dropout=0.0, scale=20.0)` so callers
                # don't swap scale/dropout.
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

            # Best-effort; simulation stubs with __slots__ raise on assignment.
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


def _nf4_dense_dequantize_weight(weight, group_size=64, use_double_quant=False):
    import mlx.core as mx

    codebook = mx.array(
        [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ],
        dtype=mx.float32,
    )

    def _bnb_dynamic_codebook():
        data = []
        max_exponent_bits = 7
        total_bits = 8
        non_sign_bits = total_bits - 1
        additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
        for i in range(max_exponent_bits):
            fraction_items = int(2 ** (i + non_sign_bits - max_exponent_bits) + 1)
            boundaries = mx.linspace(0.1, 1.0, fraction_items, dtype=mx.float32)
            means = (boundaries[:-1] + boundaries[1:]) / 2.0
            scale = 10 ** (-(max_exponent_bits - 1) + i)
            data.extend((scale * means).tolist())
            data.extend((-scale * means).tolist())
        if additional_items > 0:
            boundaries = mx.linspace(0.1, 1.0, additional_items + 1, dtype=mx.float32)
            means = (boundaries[:-1] + boundaries[1:]) / 2.0
            scale = 10 ** (-(max_exponent_bits - 1) + i)
            data.extend((scale * means).tolist())
            data.extend((-scale * means).tolist())
        data.append(0.0)
        data.append(1.0)
        data.sort()
        return mx.array(data, dtype=mx.float32)

    def _bnb_nested_absmax(absmax):
        dynamic_codebook = _bnb_dynamic_codebook()
        original_size = (
            absmax.numel()
            if callable(getattr(absmax, "numel", None))
            else (absmax.size() if callable(getattr(absmax, "size", None)) else absmax.size)
        )
        offset = mx.mean(absmax)
        shifted = (absmax - offset).reshape((-1,))
        pad = (-original_size) % 256
        if pad:
            shifted = mx.concatenate([shifted, mx.zeros((pad,), dtype=mx.float32)])
        scale_groups = shifted.reshape((-1, 256))
        scale_absmax = mx.max(mx.abs(scale_groups), axis=1, keepdims=True)
        scale_denom = mx.where(scale_absmax > 0, scale_absmax, mx.ones_like(scale_absmax))
        scaled = scale_groups / scale_denom
        scale_indices = mx.argmin(mx.abs(scaled[..., None] - dynamic_codebook), axis=-1)
        nested = (dynamic_codebook[scale_indices] * scale_absmax).reshape((-1,))[:original_size]
        return nested + offset

    original_shape = weight.shape
    original_dtype = weight.dtype
    flat = weight.astype(mx.float32).reshape((-1,))
    original_size = (
        flat.numel()
        if callable(getattr(flat, "numel", None))
        else (flat.size() if callable(getattr(flat, "size", None)) else flat.size)
    )
    pad = (-original_size) % group_size
    if pad:
        flat = mx.concatenate([flat, mx.zeros((pad,), dtype=mx.float32)])
    groups = flat.reshape((-1, group_size))
    absmax = mx.max(mx.abs(groups), axis=1, keepdims=True)
    denom = mx.where(absmax > 0, absmax, mx.ones_like(absmax))
    scaled = groups / denom
    indices = mx.argmin(mx.abs(scaled[..., None] - codebook), axis=-1)
    # Only simulate the nested (double-quantized) absmax when double quant is
    # requested. The accepted BitsAndBytesConfig path rejects
    # bnb_4bit_use_double_quant=True, and CUDA bitsandbytes dequantizes plain
    # NF4 with the raw absmax, so default NF4 must keep un-nested scales.
    if use_double_quant:
        absmax = _bnb_nested_absmax(absmax.reshape((-1,))).reshape((-1, 1))
    dequantized = (codebook[indices] * absmax).reshape((-1,))[:original_size]
    return dequantized.reshape(original_shape).astype(original_dtype)


def _apply_dense_nf4_quantization(model, config, spec: _MLXQuantizationSpec, predicate):
    import mlx.core as mx

    quantized = {}
    for path, module in model.named_modules():
        if not predicate(path, module):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or len(getattr(weight, "shape", ())) != 2:
            continue
        module.weight = _nf4_dense_dequantize_weight(weight, spec.group_size or 64)
        quantized[path] = {
            "bits": 4,
            "group_size": spec.group_size or 64,
            "mode": "nf4_dense",
            "storage": "dense_dequantized",
        }
        mx.eval(module.weight)

    updated_config = dict(config or {}) if isinstance(config, dict) else {}
    updated_config["quantization"] = quantized
    updated_config["quantization_config"] = quantized
    model._config = updated_config
    model._unsloth_quantization_config = quantized
    model._unsloth_quantization_policy = {
        **spec.to_metadata(),
        "storage": "dense_dequantized",
        "warning": (
            "nf4_dense is a diagnostic CUDA bitsandbytes NF4 parity mode. "
            "Weights are stored densely after quantize/dequantize and this "
            "does not reduce memory like QLoRA."
        ),
    }
    model._unsloth_quantized_source = "runtime_dense_nf4"
    return model, updated_config


# Real (non-stub) bitsandbytes modules, cached after the first import. bnb
# registers torch custom operators at import time, which cannot be registered
# twice, so it must be imported once per process and reused — re-importing
# after purging it from sys.modules raises "Tried to register an operator".
_REAL_BITSANDBYTES_MODULES = {}
# Serialize the stub swap in _dequantize_bnb_to_tempdir: concurrent callers
# would race the lifted-stub window and each other's sys.modules restore (and
# two multi-GB dequants at once risks OOM on Apple Silicon).
_BNB_IMPORT_LOCK = threading.Lock()


def _bnb_module_names():
    return [
        name for name in sys.modules
        if name == "bitsandbytes" or name.startswith("bitsandbytes.")
    ]


def _dequantize_bnb_to_tempdir(source, *, token, trust_remote_code):
    """Dequantize a bitsandbytes (NF4) repo to fp16 and write a clean,
    non-quantized copy to a temp dir, returning its path.

    unsloth_zoo stubs out bitsandbytes on Apple Silicon (stubs/bitsandbytes_stub),
    so the real wheel is imported here with the stub temporarily lifted and
    restored afterwards. bnb itself dequantizes the NF4 weights; the caller then
    re-quantizes via MLX's affine path. Raises if bitsandbytes (or its dequant)
    is unavailable so the caller can fall back to the clear bnb-unsupported error.
    """
    global _REAL_BITSANDBYTES_MODULES
    with _BNB_IMPORT_LOCK:
        saved_meta = list(sys.meta_path)
        stub_modules = {name: sys.modules[name] for name in _bnb_module_names()}
        sys.meta_path[:] = [
            finder for finder in sys.meta_path if type(finder).__name__ != "_BnbFinder"
        ]
        for name in _bnb_module_names():
            del sys.modules[name]
        # Reuse the already-initialized real bnb if we imported it earlier; only a
        # cold process re-imports (and re-registers torch ops) for the first time.
        sys.modules.update(_REAL_BITSANDBYTES_MODULES)
        try:
            import bitsandbytes  # noqa: F401 — real wheel; ImportError => fall back
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = "mps" if torch.backends.mps.is_available() else "cpu"

            # Only text bnb repos reach here; VLM bnb repos are rejected by the
            # caller (mlx-vlm dequant is not wired up yet). AutoModelForCausalLM
            # dequantizes the NF4 weights to fp16. Pass torch_dtype, not dtype:
            # the supported transformers 4.x range only accepts torch_dtype (a
            # `dtype` kwarg raises TypeError there), and 5.x still accepts it.
            model = AutoModelForCausalLM.from_pretrained(
                source,
                torch_dtype=torch.float16,
                device_map={"": device},
                token=token,
                trust_remote_code=trust_remote_code,
            ).dequantize()
            # After dequantize, the model config still carries bnb's
            # quantization_config plus _pre_quantization_dtype (a torch.dtype).
            # The dtype is not JSON-serializable and breaks save_pretrained; the
            # dequantized weights no longer need any of this metadata.
            def _strip_quant_meta(cfg):
                if cfg is None:
                    return
                for _attr in ("quantization_config", "_pre_quantization_dtype"):
                    if hasattr(cfg, _attr):
                        try:
                            delattr(cfg, _attr)
                        except Exception:
                            pass
                # Walk known sub-config attrs that models use to nest configs.
                for _sub in (
                    "vision_config", "text_config", "audio_config",
                    "speech_config", "image_config", "encoder_config",
                    "decoder_config",
                ):
                    _strip_quant_meta(getattr(cfg, _sub, None))
            _strip_quant_meta(model.config)
            tmpdir = tempfile.mkdtemp(prefix="unsloth_bnb_dequant_")
            try:
                # transformers 5.x: the dequantized model still carries
                # weight-name conversions that can't be reversed for quantized
                # weights, so save_pretrained's revert_weight_conversion raises.
                # The dequantized weights need no conversion; clear it. No-op on
                # 4.x, where the attribute doesn't exist.
                try:
                    model._weight_conversions = []
                except Exception:
                    pass
                model.save_pretrained(tmpdir, safe_serialization=True)
                # Save the tokenizer so the downstream mlx-lm load can read it.
                AutoTokenizer.from_pretrained(
                    source, token=token, trust_remote_code=trust_remote_code,
                ).save_pretrained(tmpdir)
            except BaseException:
                # Don't leak the multi-GB fp16 scratch copy: BaseException,
                # because a Ctrl-C during the long safetensors write is the
                # likeliest abort and must clean up too.
                shutil.rmtree(tmpdir, ignore_errors=True)
                raise
            # Release the fp16 model and bnb's MPS allocator cache so the caller's
            # MLX re-quantization (and any later loads) aren't starved of memory.
            del model
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            return tmpdir
        finally:
            _REAL_BITSANDBYTES_MODULES = {
                name: sys.modules[name] for name in _bnb_module_names()
            }
            for name in _bnb_module_names():
                del sys.modules[name]
            sys.modules.update(stub_modules)
            sys.meta_path[:] = saved_meta


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
    if spec.mode == "nf4_dense":
        return _apply_dense_nf4_quantization(model, config, spec, predicate)
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


def _mlx_distributed_rank_size(group=None):
    """Return ``(rank, world_size)`` for an optional MLX distributed group."""
    if group is None:
        return 0, 1
    rank = int(group.rank())
    world_size = int(group.size())
    if world_size < 1:
        raise ValueError(f"Invalid MLX distributed world_size={world_size}.")
    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"Invalid MLX distributed rank={rank} for world_size={world_size}."
        )
    return rank, world_size


def _mlx_distributed_group_size(group):
    return _mlx_distributed_rank_size(group)[1]


def _mlx_group_is_distributed(group) -> bool:
    size = _mlx_distributed_group_size(group)
    return group is not None and size != 1


def _mlx_active_distributed_groups(pipeline_group=None, tensor_group=None):
    active_pipeline = pipeline_group if _mlx_group_is_distributed(pipeline_group) else None
    active_tensor = tensor_group if _mlx_group_is_distributed(tensor_group) else None
    if active_pipeline is not None and active_tensor is not None:
        raise ValueError(
            "Unsloth: MLX distributed loading accepts only one parallel mode. "
            "Pass either pipeline_group or tensor_group, not both."
        )
    return active_pipeline, active_tensor


def _mlx_distributed_requested(pipeline_group=None, tensor_group=None) -> bool:
    return any(
        _mlx_group_is_distributed(group)
        for group in (pipeline_group, tensor_group)
    )


def _mlx_distributed_sharding_mode(
    model,
    *,
    pipeline_group=None,
    tensor_group=None,
    model_name="model",
):
    pipeline_group, tensor_group = _mlx_active_distributed_groups(
        pipeline_group,
        tensor_group,
    )
    if pipeline_group is None and tensor_group is None:
        return None

    has_pipelining = hasattr(getattr(model, "model", None), "pipeline")
    has_tensor_parallel = hasattr(model, "shard")

    if tensor_group is not None:
        if has_tensor_parallel:
            return "tensor"
        else:
            raise ValueError(
                f"Unsloth: '{model_name}' does not support MLX tensor "
                "parallelism. The model must expose shard(group)."
            )
    else:
        if has_pipelining:
            return "pipeline"
        else:
            raise ValueError(
                f"Unsloth: '{model_name}' does not support MLX pipeline "
                "parallelism. The model must expose model.pipeline(group)."
            )


def _apply_mlx_distributed_sharding(
    model,
    *,
    pipeline_group=None,
    tensor_group=None,
    model_name="model",
):
    mode = _mlx_distributed_sharding_mode(
        model,
        pipeline_group=pipeline_group,
        tensor_group=tensor_group,
        model_name=model_name,
    )
    if mode is None:
        return None

    if mode == "tensor":
        model.shard(tensor_group)
    else:
        model.model.pipeline(pipeline_group)
    model._unsloth_mlx_distributed_parallel_mode = mode
    return mode


def _structured_multimodal_counts(content):
    """Count explicit image, audio, and video items in structured content."""
    if isinstance(content, list):
        counts = [_structured_multimodal_counts(item) for item in content]
        return tuple(sum(values) for values in zip(*counts)) if counts else (0, 0, 0)
    if not isinstance(content, dict):
        return (0, 0, 0)

    item_type = str(content.get("type", "")).lower()
    if item_type in ("image", "image_url", "input_image"):
        return (1, 0, 0)
    if item_type in ("audio", "input_audio"):
        return (0, 1, 0)
    if item_type == "video":
        return (0, 0, 1)
    nested = content.get("content", None)
    if nested is not None and nested is not content:
        return _structured_multimodal_counts(nested)
    return (0, 0, 0)


def _content_has_structured_multimodal_markers(content):
    """Return True when content already contains explicit image/audio/video items."""
    return any(_structured_multimodal_counts(content))


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


def _structured_media_matches_count_renderer(messages, num_images, num_audios):
    """Return whether upstream count-based rendering preserves media ownership."""
    last_target_idx = -1
    media_indices = set()
    totals = [0, 0, 0]
    for i, message in enumerate(messages):
        role = str(message.get("role", "user"))
        if role not in _NON_USER_ROLES:
            last_target_idx = i
        counts = _structured_multimodal_counts(message.get("content", ""))
        if any(counts):
            media_indices.add(i)
            totals = [total + count for total, count in zip(totals, counts)]

    return (
        last_target_idx >= 0
        and str(messages[last_target_idx].get("role", "user")) == "user"
        and media_indices == {last_target_idx}
        and any(totals)
        and totals == [num_images, num_audios, 0]
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
    """Rebuild text-only chat, keeping conversation-level media on turn 1.

    mlx-vlm attached `num_images`/`num_audios` to the last user turn, shifting
    the image token between turns and breaking prompt-cache reuse for models
    like Qwen3.5 that precompute multimodal rope positions from the full prompt.
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
    # Qwen2-VL's MLX template expects a flat content list, so role-wrapped chat
    # can render empty; fall back to a role-prefixed prompt.
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
    """Normalize prompt input and apply the smallest VLM-chat rewrite.

    Bypass mlx-vlm's helper and render ourselves when (1) the prompt already
    has structured multimodal content (must survive intact) or (2) the caller
    passes conversation-level `num_images`/`num_audios` (anchor media to the
    first user turn so cache reuse doesn't shift rope positions).
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


def _has_vlm_template_result(rendered):
    """Return whether a chat renderer produced a usable prompt or message list."""
    if isinstance(rendered, str):
        return bool(rendered.strip())
    if isinstance(rendered, (list, tuple, dict)):
        return bool(rendered)
    return rendered is not None


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
        if (
            not return_messages
            and not kwargs.get("video")
            and model_type in getattr(prompt_utils, "MODEL_CONFIG", {})
            and _structured_media_matches_count_renderer(
                normalized_messages, num_images, num_audios
            )
        ):
            try:
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
            except Exception:
                pass
            else:
                if _has_vlm_template_result(rendered):
                    return rendered

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
    from .utils import collect_mlx_lora_adapter_tensors, save_pretrained_merged
    tokenizer = tokenizer or self._tokenizer
    if "save_method" not in kwargs and not collect_mlx_lora_adapter_tensors(self):
        kwargs["save_method"] = "merged_16bit"
    kwargs = _mlx_supported_kwargs(
        kwargs,
        (
            "save_method", "push_to_hub", "token", "private", "tags",
            "repo_id", "commit_message", "commit_description",
            "create_pr", "revision",
        ),
    )
    save_pretrained_merged(self, tokenizer, save_directory, **kwargs)


def _mlx_supported_kwargs(kwargs, supported):
    """Keep CUDA-compatible kwargs out of MLX-only save/export APIs."""
    return {key: kwargs[key] for key in supported if key in kwargs}


def _mlx_push_to_hub(self, repo_id, *args, **kwargs):
    """Upload MLX LoRA adapters through the HF-style push_to_hub API."""
    import tempfile
    tokenizer = kwargs.pop("tokenizer", None) or getattr(self, "_tokenizer", None)
    save_directory = kwargs.pop("save_directory", None)
    kwargs = _mlx_supported_kwargs(
        kwargs,
        (
            "token", "private", "tags", "commit_message",
            "commit_description", "create_pr", "revision",
        ),
    )
    if save_directory is not None:
        _mlx_save_pretrained_merged(
            self,
            save_directory,
            tokenizer=tokenizer,
            push_to_hub=True,
            repo_id=repo_id,
            **kwargs,
        )
        return
    with tempfile.TemporaryDirectory() as tmp_dir:
        _mlx_save_pretrained_merged(
            self,
            tmp_dir,
            tokenizer=tokenizer,
            push_to_hub=True,
            repo_id=repo_id,
            **kwargs,
        )


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
    # Default save_directory to repo_id; callers that already saved locally
    # should pass save_directory= to avoid a re-save.
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


def _mlx_prompt_to_ids(prompt):
    """Normalize HF/torch/MLX tokenizer outputs into one token-id list."""
    if prompt is None:
        raise ValueError("Unsloth MLX: generate() requires input_ids or a prompt.")
    if isinstance(prompt, dict):
        prompt = prompt.get("input_ids")
    if isinstance(prompt, str):
        return prompt
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, tuple):
        prompt = list(prompt)
    if isinstance(prompt, list):
        if len(prompt) == 1 and isinstance(prompt[0], (list, tuple)):
            prompt = list(prompt[0])
        elif prompt and isinstance(prompt[0], (list, tuple)):
            raise ValueError("Unsloth MLX: generate() only supports batch size 1.")
        return [int(token) for token in prompt]
    raise TypeError(
        "Unsloth MLX: generate() expected input_ids as a string, list, "
        "or tensor-like object with .tolist()."
    )


def _mlx_attention_mask_to_list(attention_mask):
    """Normalize HF/torch/MLX attention masks into one mask list."""
    if attention_mask is None:
        return None
    if hasattr(attention_mask, "tolist"):
        attention_mask = attention_mask.tolist()
    if isinstance(attention_mask, tuple):
        attention_mask = list(attention_mask)
    if isinstance(attention_mask, list):
        if len(attention_mask) == 1 and isinstance(attention_mask[0], (list, tuple)):
            attention_mask = list(attention_mask[0])
        elif attention_mask and isinstance(attention_mask[0], (list, tuple)):
            raise ValueError("Unsloth MLX: generate() only supports batch size 1.")
        return [int(token) for token in attention_mask]
    raise TypeError(
        "Unsloth MLX: generate() expected attention_mask as a list "
        "or tensor-like object with .tolist()."
    )


def _mlx_apply_attention_mask(prompt_ids, attention_mask):
    """Drop padded prompt ids before MLX generation and max_length math."""
    mask = _mlx_attention_mask_to_list(attention_mask)
    if mask is None or isinstance(prompt_ids, str):
        return prompt_ids
    if len(mask) != len(prompt_ids):
        raise ValueError(
            "Unsloth MLX: attention_mask length must match input_ids length."
        )
    return [token for token, keep in zip(prompt_ids, mask) if keep != 0]


def _mlx_generate_output(prompt_ids, generated_ids):
    """Build a Transformers-friendly batched generate return value."""
    sequences = [list(prompt_ids) + list(generated_ids)]
    try:
        # Broad except: a torch that is installed but broken (bad native libs)
        # raises OSError/RuntimeError, not ImportError; fall back to numpy so
        # MLX generation keeps working instead of failing hard.
        import torch
        return torch.tensor(sequences, dtype=torch.long)
    except Exception:
        import numpy as np
        return np.asarray(sequences, dtype=np.int64)


def _mlx_eos_token_id_set(eos_token_id):
    """Normalize HF-style eos_token_id values into a set of token ids."""
    if eos_token_id is None:
        return None
    if hasattr(eos_token_id, "tolist"):
        eos_token_id = eos_token_id.tolist()
    if isinstance(eos_token_id, tuple):
        eos_token_id = list(eos_token_id)
    if isinstance(eos_token_id, list):
        if len(eos_token_id) == 1 and isinstance(eos_token_id[0], (list, tuple)):
            eos_token_id = list(eos_token_id[0])
        return {int(token) for token in eos_token_id}
    return {int(eos_token_id)}


def _mlx_override_tokenizer_eos_ids(tokenizer, eos_token_id):
    """Temporarily override mlx-lm tokenizer EOS ids for one generate call."""
    eos_ids = _mlx_eos_token_id_set(eos_token_id)
    if eos_ids is None:
        return None
    had_attr = hasattr(tokenizer, "eos_token_ids")
    original = getattr(tokenizer, "eos_token_ids", None)
    try:
        tokenizer.eos_token_ids = eos_ids
    except Exception:
        return None
    return (had_attr, original)


def _mlx_restore_tokenizer_eos_ids(tokenizer, restore_state):
    """Restore tokenizer EOS ids after a temporary generate override."""
    if restore_state is None:
        return
    had_attr, original = restore_state
    try:
        if had_attr:
            tokenizer.eos_token_ids = original
        else:
            delattr(tokenizer, "eos_token_ids")
    except Exception:
        pass


def _mlx_put_streamer_tokens(streamer, token_ids):
    """Send token ids to a HF TextStreamer-compatible object."""
    if streamer is None:
        return
    import mlx.core as mx
    streamer.put(mx.array(token_ids))


def _mlx_token_to_int(token):
    """Convert an MLX scalar / Python scalar token into a Python int."""
    if token is None:
        return None
    if hasattr(token, "item"):
        token = token.item()
    return int(token)


def _mlx_generate_vlm(self, *args, **kwargs):
    """HF-style VLM generate() shim backed by mlx-vlm stream_generate."""
    from mlx_vlm import stream_generate
    from .utils import _to_mx_vlm_batch

    processor = getattr(self, "_tokenizer", None)
    if processor is None:
        raise ValueError("Unsloth MLX: VLM generate() requires model._tokenizer.")

    inputs = {}
    if args:
        if len(args) > 1:
            raise TypeError(
                "Unsloth MLX: VLM generate() accepts at most one positional "
                "input argument."
            )
        positional = args[0]
        if isinstance(positional, dict):
            inputs.update(positional)
        elif "input_ids" not in kwargs:
            inputs["input_ids"] = positional
        else:
            raise TypeError(
                "Unsloth MLX: pass input_ids either positionally or by keyword, "
                "not both."
            )
    inputs.update(kwargs)

    streamer = inputs.pop("streamer", None)
    max_tokens = inputs.pop("max_tokens", None)
    max_new_tokens = inputs.pop("max_new_tokens", None)
    max_length = inputs.pop("max_length", None)

    # HF generation flags commonly present in notebooks but not consumed by
    # mlx-vlm's generation loop.
    do_sample = inputs.pop("do_sample", None)
    inputs.pop("use_cache", None)
    inputs.pop("return_dict_in_generate", None)
    inputs.pop("output_scores", None)
    inputs.pop("output_attentions", None)
    inputs.pop("output_hidden_states", None)
    inputs.pop("pad_token_id", None)

    eos_token_id = inputs.pop("eos_token_id", None)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if hasattr(tokenizer, "stopping_criteria"):
        eos = eos_token_id
        if eos is None:
            eos = getattr(getattr(self, "config", None), "eos_token_id", None)
        tokenizer.stopping_criteria.reset(eos)

    batch = _to_mx_vlm_batch(inputs)
    batch = {
        key: value for key, value in batch.items()
        if key != "labels" and not key.startswith("_unsloth_")
    }
    input_ids = batch.get("input_ids")
    if input_ids is None:
        raise ValueError("Unsloth MLX: VLM generate() requires input_ids.")
    if len(input_ids.shape) != 2 or input_ids.shape[0] != 1:
        raise ValueError("Unsloth MLX: VLM generate() only supports batch size 1.")

    if "mask" not in batch and "attention_mask" in batch:
        batch["mask"] = batch.pop("attention_mask")
    else:
        batch.pop("attention_mask", None)

    prompt_ids = [int(token) for token in input_ids.flatten().tolist()]
    prompt_ids = _mlx_apply_attention_mask(prompt_ids, batch.get("mask", None))
    if max_tokens is None:
        if max_new_tokens is not None:
            max_tokens = int(max_new_tokens)
        elif max_length is not None:
            max_tokens = max(0, int(max_length) - len(prompt_ids))
        else:
            max_tokens = 256

    if "temp" in batch and "temperature" not in batch:
        batch["temperature"] = batch.pop("temp")
    if do_sample is False:
        batch["temperature"] = 0.0
        batch.pop("temp", None)
        batch.pop("top_p", None)
        batch.pop("top_k", None)
        batch.pop("min_p", None)
    elif do_sample is True and batch.get("temperature", None) is None:
        batch["temperature"] = 1.0
    elif batch.get("temperature", None) is None:
        batch.pop("temperature", None)

    _mlx_put_streamer_tokens(streamer, [prompt_ids])

    generated_ids = []
    last_generation_tokens = None
    for response in stream_generate(
        self,
        processor,
        "",
        max_tokens=max_tokens,
        **batch,
    ):
        token_id = _mlx_token_to_int(getattr(response, "token", None))
        if token_id is None:
            continue
        generation_tokens = getattr(response, "generation_tokens", None)
        if (
            generation_tokens is not None
            and generation_tokens == last_generation_tokens
        ):
            continue
        last_generation_tokens = generation_tokens
        generated_ids.append(token_id)
        _mlx_put_streamer_tokens(streamer, [token_id])

    if streamer is not None:
        streamer.end()
    return _mlx_generate_output(prompt_ids, generated_ids)


def _mlx_generate(self, *args, **kwargs):
    """HF-style text generate() shim backed by mlx-lm stream_generate."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    if getattr(self, "_is_vlm_model", False):
        return _mlx_generate_vlm(self, *args, **kwargs)

    tokenizer = getattr(self, "_tokenizer", None)
    if tokenizer is None:
        raise ValueError("Unsloth MLX: generate() requires model._tokenizer.")

    prompt = kwargs.pop("input_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    if args:
        if prompt is None:
            prompt = args[0]
            if attention_mask is None and isinstance(prompt, dict):
                attention_mask = prompt.get("attention_mask")
        else:
            raise TypeError(
                "Unsloth MLX: pass prompt input either positionally or as "
                "input_ids, not both."
            )
    prompt_ids = _mlx_prompt_to_ids(prompt)
    prompt_ids = _mlx_apply_attention_mask(prompt_ids, attention_mask)
    if isinstance(prompt_ids, str):
        add_special_tokens = (
            getattr(tokenizer, "bos_token", None) is None
            or not prompt_ids.startswith(getattr(tokenizer, "bos_token", ""))
        )
        try:
            prompt_ids = tokenizer.encode(
                prompt_ids,
                add_special_tokens=add_special_tokens,
            )
        except TypeError:
            prompt_ids = tokenizer.encode(prompt_ids)

    streamer = kwargs.pop("streamer", None)
    max_tokens = kwargs.pop("max_tokens", None)
    max_new_tokens = kwargs.pop("max_new_tokens", None)
    max_length = kwargs.pop("max_length", None)
    if max_tokens is None:
        if max_new_tokens is not None:
            max_tokens = int(max_new_tokens)
        elif max_length is not None:
            max_tokens = max(0, int(max_length) - len(prompt_ids))
        else:
            max_tokens = 256

    do_sample = kwargs.pop("do_sample", None)
    eos_token_id = kwargs.pop("eos_token_id", None)
    sampler = kwargs.pop("sampler", None)
    _missing_temperature = object()
    temp = kwargs.pop("temperature", kwargs.pop("temp", _missing_temperature))
    if temp is _missing_temperature:
        temp = 1.0 if do_sample is True else 0.0
    if temp is None:
        temp = 0.0
    top_p = float(kwargs.pop("top_p", 0.0) or 0.0)
    min_p = float(kwargs.pop("min_p", 0.0) or 0.0)
    top_k = int(kwargs.pop("top_k", 0) or 0)
    if do_sample is False and sampler is None:
        temp = 0.0
        top_p = 0.0
        min_p = 0.0
        top_k = 0
    sampler = sampler or make_sampler(
        temp=float(temp),
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
    )
    logits_processors = kwargs.pop("logits_processors", None)
    if logits_processors is None:
        logits_processors = make_logits_processors(
            logit_bias=kwargs.pop("logit_bias", None),
            repetition_penalty=kwargs.pop("repetition_penalty", None),
            presence_penalty=kwargs.pop("presence_penalty", None),
            frequency_penalty=kwargs.pop("frequency_penalty", None),
        )

    stream_kwargs = _mlx_supported_kwargs(
        kwargs,
        (
            "max_kv_size", "prompt_cache", "prefill_step_size",
            "kv_bits", "kv_group_size", "quantized_kv_start",
            "prompt_progress_callback", "input_embeddings",
        ),
    )

    # HF TextStreamer(skip_prompt=True) expects one prompt callback first.
    _mlx_put_streamer_tokens(streamer, [prompt_ids])

    generated_ids = []
    eos_restore_state = _mlx_override_tokenizer_eos_ids(tokenizer, eos_token_id)
    try:
        for response in stream_generate(
            self,
            tokenizer,
            prompt_ids,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            **stream_kwargs,
        ):
            token = getattr(response, "token", None)
            token_id = _mlx_token_to_int(token)
            if token_id is None:
                continue
            generated_ids.append(token_id)
            _mlx_put_streamer_tokens(streamer, [token_id])
    finally:
        _mlx_restore_tokenizer_eos_ids(tokenizer, eos_restore_state)

    if streamer is not None:
        streamer.end()
    return _mlx_generate_output(prompt_ids, generated_ids)


def _mlx_chat_template_batch_encoding(output):
    """Wrap tokenized chat-template output in a HF mapping when requested."""
    from transformers import BatchEncoding

    if isinstance(output, BatchEncoding):
        return output
    if isinstance(output, Mapping):
        return BatchEncoding(dict(output))
    return BatchEncoding({"input_ids": output})


def _patch_mlx_tokenizer_call(tokenizer):
    """Patch mlx-lm TokenizerWrapper to match HF notebook tokenizer APIs."""
    if tokenizer is None:
        return
    cls = type(tokenizer)
    if cls.__name__ != "TokenizerWrapper":
        return
    if "__call__" not in cls.__dict__:
        if hasattr(tokenizer, "_tokenizer") and callable(tokenizer._tokenizer):
            def tokenizer_wrapper_call(self, *args, **kwargs):
                return self._tokenizer(*args, **kwargs)

            tokenizer_wrapper_call._unsloth_mlx_call = True
            cls.__call__ = tokenizer_wrapper_call

    if getattr(cls, "_unsloth_mlx_apply_chat_template", False):
        return
    original_apply_chat_template = getattr(cls, "apply_chat_template", None)
    if original_apply_chat_template is None:
        return

    def tokenizer_wrapper_apply_chat_template(self, *args, tokenize=True, **kwargs):
        return_dict = bool(kwargs.get("return_dict", False))
        inner_tokenizer = getattr(self, "_tokenizer", None)
        if (
            tokenize
            and return_dict
            and getattr(self, "_chat_template", None) is None
            and hasattr(inner_tokenizer, "apply_chat_template")
        ):
            if "enable_thinking" not in kwargs:
                kwargs["enable_thinking"] = getattr(self, "has_thinking", False)
            output = inner_tokenizer.apply_chat_template(
                *args,
                tokenize=tokenize,
                **kwargs,
            )
            return _mlx_chat_template_batch_encoding(output)

        output = original_apply_chat_template(
            self,
            *args,
            tokenize=tokenize,
            **kwargs,
        )
        if tokenize and return_dict:
            return _mlx_chat_template_batch_encoding(output)
        return output

    tokenizer_wrapper_apply_chat_template._unsloth_mlx_call = True
    cls.apply_chat_template = tokenizer_wrapper_apply_chat_template
    cls._unsloth_mlx_apply_chat_template = True


def _patch_mlx_saving(model, tokenizer):
    """Attach save/push methods to the model, matching unsloth's CUDA pattern."""
    _patch_mlx_tokenizer_call(tokenizer)
    model._tokenizer = tokenizer
    model.generate               = types.MethodType(_mlx_generate, model)
    model.save_pretrained        = types.MethodType(_mlx_save_pretrained_merged, model)
    model.save_pretrained_merged = types.MethodType(_mlx_save_pretrained_merged, model)
    model.save_pretrained_gguf   = types.MethodType(_mlx_save_pretrained_gguf, model)
    model.push_to_hub            = types.MethodType(_mlx_push_to_hub, model)
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


def _remap_unsloth_bnb_hub_id_for_mlx(model_name, revision):
    """Map an unsloth/*-bnb-4bit Hub ID to its full-precision base repo.

    mlx-lm cannot read bitsandbytes-packed weights, so load the base repo and
    let MLX quantize to 4-bit. Returns (name, revision, remapped_from); the
    bnb-pinned revision is dropped since it does not apply to the base repo.
    Local paths and third-party repos keep the exact name given.
    """
    if (
        not isinstance(model_name, str)
        or not model_name.startswith("unsloth/")
        or os.path.exists(model_name)
    ):
        return model_name, revision, None
    for _bnb_suffix in ("-unsloth-bnb-4bit", "-bnb-4bit"):
        if model_name.endswith(_bnb_suffix):
            return model_name[: -len(_bnb_suffix)], None, model_name
    return model_name, revision, None


def _coerce_list_extra_special_tokens():
    # why: the MLX path skips unsloth's TEMPORARY_PATCHES. Old transformers crash
    # on a list extra_special_tokens; v5 accepts it, so only coerce on failure.
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except Exception:
        return
    init = PreTrainedTokenizerBase.__init__
    if getattr(init, "_unsloth_extra_special_tokens_patched", False):
        return

    def patched_init(*args, **kwargs):
        if not isinstance(kwargs.get("extra_special_tokens"), list):
            return init(*args, **kwargs)
        try:
            return init(*args, **kwargs)
        except AttributeError as e:
            if "keys" not in str(e):
                raise
            kwargs["extra_special_tokens"] = {}
            return init(*args, **kwargs)

    patched_init._unsloth_extra_special_tokens_patched = True
    patched_init._unsloth_original_init = init
    PreTrainedTokenizerBase.__init__ = patched_init


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
        pipeline_group=None,
        tensor_group=None,
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
            full_finetuning: When True, disable runtime quantization so
                full-precision weights are trainable. MLX mirrors Unsloth Torch
                and upcasts trainable float weights to float32; pass
                ``float32_mixed_precision=False`` to keep native bf16 on
                bf16-capable chips. ``get_peft_model`` becomes a no-op.
            token: HuggingFace token for gated models.
            text_only: Loading mode:
                None  — auto-detect from config (default)
                True  — force text-only via mlx-lm
                False — force VLM via mlx-vlm
            pipeline_group: Optional MLX distributed group for pipeline
                parallel text or VLM inference.
            tensor_group: Optional MLX distributed group for tensor parallel
                text or VLM inference.
        """
        _coerce_list_extra_special_tokens()
        _mlx_active_distributed_groups(pipeline_group, tensor_group)

        model_name, revision, _bnb_remapped_from = _remap_unsloth_bnb_hub_id_for_mlx(
            model_name, revision
        )
        if _bnb_remapped_from is not None:
            print(
                f"Unsloth: mlx-lm cannot load bitsandbytes 4-bit weights; "
                f"loading base '{model_name}' and applying MLX 4-bit "
                f"quantization instead of '{_bnb_remapped_from}'."
            )
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
        distributed_requested = _mlx_distributed_requested(pipeline_group, tensor_group)

        # Seed mlx random state so construction-time randomness (e.g. runtime
        # quant layer init) is reproducible.
        _seed_mlx_random_state(random_state)

        # Separate download from config-read so a missing config.json doesn't
        # clear local_path: LoRA-adapter dirs have adapter_config.json but no
        # config.json, and the adapter branch needs local_path.
        local_path = None
        original_local_path = None
        vlm_config_override_data = None
        try:
            with _temporary_hf_token_env(token):
                config_allow_patterns = None
                if distributed_requested:
                    config_allow_patterns = _mlx_lm_metadata_allow_patterns()
                local_path = str(
                    _download(
                        model_name,
                        revision=revision,
                        allow_patterns=config_allow_patterns,
                    )
                )
                original_local_path = local_path
        except Exception:
            if distributed_requested:
                raise
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
            original_local_path = local_path
            original_config_data = dict(config_data)
            if distributed_requested:
                patched_config_data = _mlx_vlm_config_override_data(config_data)
                if patched_config_data is not None:
                    config_data = patched_config_data
                    vlm_config_override_data = dict(config_data)
            else:
                local_path, config_data = _materialize_mlx_vlm_config_override(
                    local_path,
                    config_data,
                )
                if local_path != original_local_path or config_data != original_config_data:
                    vlm_config_override_data = dict(config_data)

        # bitsandbytes-quantized repos store NF4 weights MLX cannot read. When
        # the real bitsandbytes wheel is importable, let bnb dequantize to fp16
        # and re-enter the normal load on a clean copy so MLX's affine path
        # re-quantizes it (the user gets an MLX 4-bit model, never seeing bnb).
        # If bitsandbytes is unavailable, fall back to a clear, actionable error.
        _existing_quant = _get_existing_mlx_quantization(config_data)
        # A LoRA adapter dir can carry a copied base config.json (bitsandbytes
        # when the base was a bnb repo). It must take the adapter branch below,
        # not be dequantized as a full model, so skip the bnb path for it.
        _is_adapter_dir = bool(local_path) and os.path.exists(
            os.path.join(local_path, "adapter_config.json")
        )
        if (
            not _is_adapter_dir
            and isinstance(_existing_quant, dict)
            and _existing_quant.get("quant_method") == "bitsandbytes"
        ):
            _suggested = re.sub(
                r"-(?:unsloth-)?bnb-\d+bit$", "", model_name,
            ) or model_name
            _suggestion_line = (
                f"  - Try the non-bnb variant: '{_suggested}'\n"
                if _suggested != model_name
                else "  - Try the non-bnb variant of this model (drop the "
                     "'-bnb-4bit' suffix)\n"
            )
            # VLM bitsandbytes repos are out of scope on the MLX path: the
            # dequant path only handles text models (mlx-lm), and mlx-vlm
            # dequant is not wired up. Reject with the clear, actionable error
            # instead of attempting an unverified VLM dequant + mlx-vlm load.
            # DeepSeek-OCR is a VLM routed through mlx-vlm despite its
            # *ForCausalLM architecture. Its config carries a top-level
            # vision_config, so `_is_vlm` already flags it; the model_type
            # check keeps the rejection from hinging on that one config key.
            if _is_vlm(config_data) or _deepseek_ocr_config_model_type(config_data):
                raise ValueError(
                    f"Unsloth: '{model_name}' is a bitsandbytes-quantized VLM, "
                    "which isn't supported on the MLX path yet.\n"
                    f"{_suggestion_line}"
                )
            try:
                _dequant_dir = _dequantize_bnb_to_tempdir(
                    local_path or model_name,
                    token=token,
                    trust_remote_code=trust_remote_code,
                )
            except ImportError:
                raise ValueError(
                    f"Unsloth: '{model_name}' is a bitsandbytes-quantized model "
                    "and bitsandbytes is not available to dequantize it on this "
                    "machine.\n"
                    f"{_suggestion_line}"
                    "  - Or install a bitsandbytes build that runs here so "
                    "Unsloth can dequantize and re-quantize via MLX"
                )
            except Exception as _bnb_exc:
                # bnb-4bit loading in transformers requires `accelerate`, which
                # is excluded from the darwin-arm64 deps — exactly where this
                # path runs. Detect that specific failure and point the user at
                # the real fix instead of misreporting it as a bnb problem.
                if "accelerate" in str(_bnb_exc).lower():
                    raise ValueError(
                        f"Unsloth: loading the bitsandbytes model '{model_name}' "
                        "requires the `accelerate` package, which isn't installed "
                        "on this machine.\n"
                        "  - Install it with: pip install accelerate\n"
                        "  - Then re-run; Unsloth will dequantize and re-quantize "
                        "via MLX."
                    ) from _bnb_exc
                raise ValueError(
                    f"Unsloth: failed to dequantize the bitsandbytes model "
                    f"'{model_name}' "
                    f"({type(_bnb_exc).__name__}: {_bnb_exc}).\n"
                    f"{_suggestion_line}"
                ) from _bnb_exc
            try:
                model, tokenizer = FastMLXModel.from_pretrained(
                    _dequant_dir,
                    max_seq_length=max_seq_length,
                    dtype=dtype,
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    load_in_16bit=load_in_16bit,
                    load_in_fp8=load_in_fp8,
                    load_in_mxfp4=load_in_mxfp4,
                    load_in_nvfp4=load_in_nvfp4,
                    full_finetuning=full_finetuning,
                    token=token,
                    trust_remote_code=trust_remote_code,
                    text_only=text_only,
                    patch_mode=patch_mode,
                    revision=None,
                    random_state=random_state,
                    float32_mixed_precision=float32_mixed_precision,
                    chat_template=chat_template,
                    q_bits=q_bits,
                    q_group_size=q_group_size,
                    q_mode=q_mode,
                    quant_predicate=quant_predicate,
                    quantize_modules=quantize_modules,
                    force_requantize=force_requantize,
                    **(
                        {"mlx_quantization_config": mlx_quantization_config}
                        if mlx_quantization_config is not None
                        else {}
                    ),
                    **(
                        {"quantization_config": quantization_config}
                        if quantization_config is not None
                        else {}
                    ),
                )
            finally:
                # MLX has materialized its weights by now; the fp16 scratch copy
                # (several GB) is no longer needed.
                shutil.rmtree(_dequant_dir, ignore_errors=True)
            model._hf_repo = model_name
            model._src_path = local_path
            model._unsloth_base_revision = revision
            model._unsloth_base_commit_hash = _infer_snapshot_commit(local_path)
            return model, tokenizer

        # Reject full_finetuning on a pre-quantized repo: int4/int8 weights
        # aren't trainable (our CCE backward zeros the quantized weight grad),
        # so full FT would silently update only LayerNorms/biases.
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
                    active_pipeline_group, active_tensor_group = _mlx_active_distributed_groups(
                        pipeline_group,
                        tensor_group,
                    )
                    if active_pipeline_group is not None or active_tensor_group is not None:
                        raise ValueError(
                            "Unsloth: MLX distributed loading for LoRA adapter "
                            "repos is not supported yet. Merge/export the adapter "
                            "into an MLX model before distributed inference."
                        )
                    adapter_quant_policy = adapter_cfg.get("base_quantization_policy") or {}
                    adapter_quant_map = (
                        adapter_cfg.get("base_resolved_quantization_map")
                        or adapter_cfg.get("base_quantization_map")
                    )
                    adapter_base_revision = _adapter_base_revision(adapter_cfg)
                    # Keep the base repo consistent across the recursive load,
                    # metadata download, and recorded _hf_repo when the adapter
                    # base is an unsloth/*-bnb-4bit Hub ID.
                    base_model_id, adapter_base_revision, _adapter_base_bnb = (
                        _remap_unsloth_bnb_hub_id_for_mlx(base_model_id, adapter_base_revision)
                    )
                    if _adapter_base_bnb is not None:
                        print(
                            f"Unsloth: adapter base '{_adapter_base_bnb}' is bitsandbytes "
                            f"4-bit; loading base '{base_model_id}' for MLX instead."
                        )
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
                    # A bnb-4bit base always needs MLX 4-bit quantization. Force it
                    # whenever the metadata branch above did not yield a usable MLX
                    # config -- no metadata at all, or only a CUDA/bitsandbytes
                    # base_quantization_config with no MLX map -- so we never reload a
                    # full-precision base for a 4-bit-trained adapter. A usable MLX map
                    # already set adapter_mlx_quant_config, so this leaves it intact.
                    if _adapter_base_bnb is not None and adapter_mlx_quant_config is None:
                        adapter_mlx_quant_config = {
                            "bits": 4,
                            "group_size": _MLX_QUANT_MODE_DEFAULTS["affine"][0],
                            "mode": "affine",
                        }
                    # Reload the base via FastMLXModel.from_pretrained (text +
                    # VLM); the old mlx_lm.load fallback broke VLM adapters
                    # (mlx-lm load is text-only).
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
                        pipeline_group=pipeline_group,
                        tensor_group=tensor_group,
                        **(
                            {"mlx_quantization_config": adapter_mlx_quant_config}
                            if adapter_mlx_quant_config is not None
                            else {}
                        ),
                    )
                    _validate_mlx_adapter_base(model, adapter_cfg)
                    # why: load_adapters rebuilds only language-tower LoRA;
                    # vision/projector LoRA must be re-attached so load_weights
                    # binds the trained tensors.
                    _saved_lora_paths = _normalize_mlx_lora_module_paths(
                        adapter_cfg.get("unsloth_mlx_lora_module_paths"),
                    )
                    # load_adapters FIRST (rebuilds the language tower);
                    # _apply_lora_at_paths runs after and skips wrapped paths,
                    # attaching only auxiliary (vision/projector/MoE/embedding).
                    # The old order nested LoRALinear(LoRALinear).
                    from mlx_lm.tuner.utils import load_adapters
                    # Let older mlx-lm load_adapters accept scale=/dropout=.
                    _patch_mlx_lora_from_base_compat()
                    adapter_weights_file = os.path.join(local_path, "adapters.safetensors")
                    _load_adapters_ok = False
                    _load_adapters_exc = None
                    # Pre-validate DoRA: catch missing mlx_lm.tuner.dora before
                    # load_adapters rebuilds plain LoRA and drops saved DoRA
                    # `.m` via strict=False (distinct from the per-module
                    # post-check in _apply_lora_at_paths).
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
                        # Fall through to the manual-wrap + strict=False fallback
                        # (for adapters lacking mlx-lm metadata, e.g. num_layers).
                        _load_adapters_exc = _exc

                    _aux_attached = 0
                    if _saved_lora_paths:
                        # Attach auxiliary paths (vision/projector/MoE) that
                        # linear_to_lora_layers skips; language-tower paths are
                        # no-ops via the skip-if-wrapped guard.
                        try:
                            _aux_attached = _apply_lora_at_paths(
                                model, _saved_lora_paths, adapter_cfg,
                                adapter_weights_file=adapter_weights_file,
                            ) or 0
                        except (ValueError, ImportError):
                            # Caller-actionable (missing rank / DoRA class); never
                            # downgrade.
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
                        # Always diff (even _aux_attached == 0): a declared aux
                        # path the live tree no longer satisfies has no module
                        # to bind into.
                        _missing_after_success = _warn_missing_adapter_keys(
                            model, adapter_weights_file,
                        )
                        if _aux_attached > 0:
                            model.load_weights(adapter_weights_file, strict=False)
                        # Refuse a partial adapter on any shape mismatch (e.g.
                        # stale rank=8 over rank-4), matching the fallback.
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
                        # No wrappers: strict=False would drop every saved
                        # tensor and return a base model; re-raise instead.
                        if _aux_attached == 0:
                            if _load_adapters_exc is not None:
                                raise _load_adapters_exc
                            raise RuntimeError(
                                "Unsloth MLX: adapter load failed and no "
                                "live LoRA wrappers exist to bind the "
                                "saved tensors against."
                            )
                        if os.path.exists(adapter_weights_file):
                            # Diff saved-vs-live before strict=False (covers DoRA
                            # `.m` + lora_{a,b}).
                            _missing_saved_keys = _warn_missing_adapter_keys(
                                model, adapter_weights_file,
                            )
                            # Refuse an aux-only partial adapter (language tower
                            # would mis-train); chain the original error.
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
                            # No safetensors; surface the original failure.
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
                        base_allow_patterns = None
                        if _mlx_distributed_requested(pipeline_group, tensor_group):
                            base_allow_patterns = _mlx_lm_metadata_allow_patterns()
                        base_local = str(
                            _download(
                                base_model_id,
                                revision=adapter_base_revision,
                                allow_patterns=base_allow_patterns,
                            )
                        )
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
                        # A bnb-4bit base was remapped to its full-precision repo; the
                        # adapter's recorded base_model_commit_hash is the bnb repo's
                        # commit and need not exist in the base repo, so infer from the
                        # downloaded base snapshot to avoid writing an unresolvable rev.
                        _infer_snapshot_commit(base_local)
                        if _adapter_base_bnb is not None
                        else (
                            adapter_cfg.get("base_model_commit_hash")
                            or _infer_snapshot_commit(base_local)
                        )
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
                # Preserve Unsloth-tagged Value/Import/RuntimeError so
                # adapter-config failures aren't swallowed into a base-model
                # fallback.
                _msg = str(e)
                _is_unsloth = (
                    "Unsloth:" in _msg or "Unsloth MLX:" in _msg
                )
                if _is_unsloth and isinstance(
                    e, (ValueError, ImportError, RuntimeError)
                ):
                    raise
                # Preserve the Unsloth MLX RuntimeError from _apply_lora_at_paths.
                if isinstance(e, RuntimeError) and "Unsloth MLX:" in str(e):
                    raise
                # If a LoRA/DoRA artifact was declared, refuse the base-model
                # fallback for any other exception.
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

        # Route based on text_only.
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

            if local_path:
                local_path, config_data = _materialize_mlx_vlm_config_override(
                    local_path,
                    config_data,
                    normalize_tokenizer_config=True,
                )

            if patch_mode == "patched":
                install_mlx_compile_patches()
            _ensure_vlm_prompt_utils_patched()
            _ensure_mlx_vlm_processor_repair()
            _ensure_mlx_vlm_pt_output_fallback()
            _ensure_audio_conv_sanitize(model_type)

            quant_state = _ensure_quantization_compatible(
                config_data, quantization_spec, model_name,
            )
            want_runtime_quant = quantization_spec.enabled and quant_state != "compatible"
            quantization_spec, want_runtime_quant = _resolve_distributed_runtime_quantization(
                model_name,
                quantization_spec,
                distributed_requested=distributed_requested,
                want_runtime_quant=want_runtime_quant,
            )

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
                _patch_deepseek_ocr_transformers_import_compat(model_type)
                vlm_load_target = local_path or model_name
                with _temporary_hf_token_env(token):
                    try:
                        model, processor = vlm_load(
                            vlm_load_target,
                            lazy=True,
                            revision=revision,
                            **extra_kwargs,
                        )
                    except ValueError as error:
                        # Pre-quantize load bypasses the extra-weight filter, so
                        # surface the QK-norm version gap here too.
                        _raise_if_qk_norm_version_gap(model_type, str(error), error)
                        raise
                    vlm_cfg = _vlm_load_config(vlm_load_target)
                model, vlm_cfg = _apply_mlx_quantization(
                    model, vlm_cfg, quantization_spec,
                    is_vlm=True, user_predicate=quant_predicate,
                )
                model._config = vlm_cfg
                mx.eval(model.parameters())
            elif distributed_requested:
                if text_only is False and not _is_vlm(config_data):
                    raise ValueError(
                        "Unsloth: distributed MLX VLM inference requires a "
                        f"supported mlx-vlm architecture. '{model_name}' does "
                        f"not appear to be a VLM (model_type={model_type!r})."
                    )
                active_pipeline_group, active_tensor_group = _mlx_active_distributed_groups(
                    pipeline_group,
                    tensor_group,
                )
                mode = "tensor" if active_tensor_group is not None else "pipeline"
                print(f"Unsloth: Loading {model_name} via mlx-vlm (distributed {mode} VLM)...")
                model, processor = _load_mlx_vlm_distributed(
                    model_name,
                    model_type,
                    pipeline_group=active_pipeline_group,
                    tensor_group=active_tensor_group,
                    hf_token=token,
                    revision=revision,
                    config_override_data=vlm_config_override_data,
                )
            else:
                print(f"Unsloth: Loading {model_name} via mlx-vlm (VLM)...")
                # Lazy-load when converting dtype so weights materialize once.
                vlm_kwargs = dict(extra_kwargs)
                vlm_kwargs["revision"] = revision
                if target_dtype is not None:
                    vlm_kwargs["lazy"] = True
                model, processor = _load_mlx_vlm_with_extra_weight_filter(
                    local_path or model_name,
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
            _fix_gemma3_text_rmsnorm_fp32(model)

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
            _fix_gemma3_vision_post_layernorm_eps(model)
            _fix_gemma3_vision_attention_fp32_sdpa(model)
            _fix_gemma3_vision_encoder_fp32_layernorm(model)
            _fix_gemma3_vision_post_layernorm_fp32(model)
            _fix_gemma3_vision_mlp_fp32_activation(model)
            _fix_gemma3_language_mlp_fp32_activation(model)
            _fix_gemma3_multimodal_image_feature_scale(model)

            model._config = getattr(model, "_config", config_data)
            model._hf_repo = model_name
            model._src_path = original_local_path or local_path
            # _src_path is the original snapshot (for commit inference); sidecar
            # saving needs the mlx-vlm patched dir when one was materialized
            # (e.g. DeepSeek OCR), else saved adapters copy the unpatched
            # model_type/auto_map the override drops.
            model._config_src_path = local_path or original_local_path
            model._unsloth_base_revision = revision
            model._unsloth_base_commit_hash = _infer_snapshot_commit(
                original_local_path or local_path
            )
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
            quantization_spec, want_runtime_quant = _resolve_distributed_runtime_quantization(
                model_name,
                quantization_spec,
                distributed_requested=distributed_requested,
                want_runtime_quant=want_runtime_quant,
            )

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
            if _mlx_distributed_requested(pipeline_group, tensor_group):
                mlx_load_kwargs["lazy"] = True
                model, tokenizer, config = _load_mlx_lm_distributed(
                    model_name,
                    model_type,
                    mlx_load_kwargs,
                    pipeline_group=pipeline_group,
                    tensor_group=tensor_group,
                    hf_token=token,
                )
            else:
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
            model._src_path = original_local_path or local_path
            # Mirror the VLM branch: sidecar-saving uses the patched dir when one
            # was materialized (no-op for text, where local_path == original).
            model._config_src_path = local_path or original_local_path
            model._unsloth_base_revision = revision
            model._unsloth_base_commit_hash = _infer_snapshot_commit(
                original_local_path or local_path
            )
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

        For VLMs, applies LoRA to the language model and optionally the vision
        tower (train_vision=True) and projector (train_projector=True). No-op
        when the model was loaded with ``full_finetuning=True``.
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


        # "all-linear" means every nn.Linear (fused QKV, MoE routers/experts,
        # vision linears, projector, untied lm_head). Walk the tree for those
        # names rather than collapsing to the canonical 7, which would leave
        # fused-attention archs and MoEs mostly un-LoRA'd.
        if target_modules == ["all-linear"] or target_modules == "all-linear":
            target_modules = _collect_all_linear_target_names(model)
            if not target_modules:
                # No Linear modules found; fall back to the canonical default.
                target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ]

        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

        # Filter by finetune_attention_modules / finetune_mlp_modules,
        # whatever the source of target_modules, so these flags always apply.
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
            # Freeze everything, then apply LoRA selectively.
            _fix_missing_no_grad(model)
            _fix_gemma4_kv_sharing(model)
            model.freeze()

            # LoRA the language model (filtered by target_modules).
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
                    # Compat patch (older mlx-lm rejects scale=/dropout= on
                    # from_base); before the seed since monkey-patching doesn't
                    # advance mx.random.
                    _patch_mlx_lora_from_base_compat()
                    # Seed mx.random immediately before LoRA init (like
                    # mlx_lm/tuner/lora.py train); otherwise lazy state
                    # advances leak into lora_a sampling.
                    _seed_mlx_random_state(random_state)
                    linear_to_lora_layers(
                        lm,
                        num_layers=num_layers,
                        config={**lora_config, "keys": language_lora_keys},
                        use_dora=False,
                    )
                    language_lora_count = len(language_lora_keys) if language_lora_keys is not None else num_layers

            # Optionally LoRA the vision tower.
            vision_lora_count = 0
            if train_vision:
                vision_lora_count = _lora_walk_module(
                    model,
                    lora_config,
                    target_modules,
                    attr_names=("vision_tower", "vision_model", "vision_encoder"),
                )

            # Optionally LoRA the projector/connector. LoRA beats unfreezing
            # raw weights since many projectors are QuantizedLinear and MLX
            # can't backprop into quantized weights.
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

            # Unfreeze all LoRA params across the tree.
            model.unfreeze(keys=["lora_a", "lora_b"], strict=False)
        else:
            # Text-only path. _fix_missing_no_grad handles modules using
            # __new__ without __init__ (e.g. Gemma4 AudioRelativePosition...).
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
                # Compat patch (older mlx-lm rejects scale=/dropout= on
                # from_base); before the seed since monkey-patching doesn't
                # advance mx.random.
                _patch_mlx_lora_from_base_compat()
                # Seed mx.random immediately before LoRA init (like
                # mlx_lm/tuner/lora.py train); otherwise lazy state advances
                # leak into lora_a sampling.
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

        # Gradient checkpointing: "mlx"/True -> apply; False/"none" -> skip.
        if isinstance(use_gradient_checkpointing, str):
            _apply_gc = use_gradient_checkpointing.lower() not in ("false", "none", "")
        else:
            _apply_gc = bool(use_gradient_checkpointing)

        if _apply_gc:
            from .utils import apply_gradient_checkpointing
            apply_gradient_checkpointing(model)

        if hasattr(model, "trainable_parameters") and hasattr(model, "parameters"):
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

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["FastCPUModel"]

import contextlib
import gc
import functools
import inspect
import itertools
import os
import re
import sys
import types
import warnings
import weakref

import torch
from unsloth_zoo.utils import is_activation_checkpointing as _is_activation_checkpointing
from unsloth_zoo.utils import _is_out_of_memory

from unsloth_zoo.private_staging import keep_mode, precreate_private_temp
from .runtime import available_ram, cpu_bf16_supported, cpu_bnb_4bit_supported
from .ops import (
    Bf16StoredLinear, _plain_lm_head, patch_causal_conv1d_fallback, patch_chunked_causal_lm_loss, set_bf16_stored_dtype,
    store_frozen_linears_in_bf16,
)

_DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# unsloth/Qwen3-0.6B-unsloth-bnb-4bit -> unsloth/Qwen3-0.6B
_PREQUANTIZED_SUFFIX = re.compile(r"(-unsloth)?-bnb-4bit$", re.IGNORECASE)


def _gguf_dequantize_config():
    """`GgufConfig(dequantize = True)` where transformers has a GGUF quantizer (5.17+), else None.

    Left at its default that quantizer keeps Qwen3.5 / LFM2 weights packed where the ggml kernel resolves, and for
    every other architecture stays attached with `is_trainable = False`, so Trainer refuses LoRA and full
    finetuning. Dequantizing drops it: the dense model a GGUF load gave before 5.17."""
    try:
        from transformers import GgufConfig
    except ImportError:
        return None
    if "dequantize" not in inspect.signature(GgufConfig.__init__).parameters:
        return None
    return GgufConfig(dequantize = True)


# GPU-only kwargs never reach AutoModelForCausalLM.from_pretrained, whose __init__ rejects them.
_GPU_LOADER_KWARGS_IGNORED = (
    "load_in_4bit", "load_in_8bit", "quantization_config", "use_exact_model_name",
    "max_lora_rank", "gpu_memory_utilization", "float8_kv_cache", "random_state",
    "fix_tokenizer", "unsloth_tiled_mlp", "offload_embedding",
    "device_map_planner_kwargs", "float32_mixed_precision", "disable_log_stats",
    "fullgraph", "unsloth_force_compile", "text_only", "whisper_language", "whisper_task",
)
# Features the CPU cannot run: dropped when off, refused when set, never silently skipped.
_GPU_LOADER_KWARGS_REFUSED = {
    # The GPU route refuses CPU too: float8 GEMMs need fp8 tensor cores (compute capability 8.9+).
    "float8_training": "float8_training needs an NVIDIA GPU with compute capability 8.9+ (fp8 tensor cores)",
    "fp8_lora": "fp8 LoRA needs an NVIDIA GPU with compute capability 8.9+ (fp8 tensor cores)",
    "load_in_fp8": "load_in_fp8 needs an NVIDIA GPU",
    "qat_scheme": "qat_scheme (quantization aware training) is not supported on the CPU backend",
    "auto_model": "auto_model is not supported on the CPU backend, which loads AutoModelForCausalLM",
    "user_config": "user_config is not supported on the CPU backend",
    "target_parameters": "target_parameters is not supported on the CPU backend",
}


def _pop_gpu_loader_kwargs(kwargs):
    for key, reason in _GPU_LOADER_KWARGS_REFUSED.items():
        value = kwargs.pop(key, None)
        if value not in (None, False):
            raise NotImplementedError(f"Unsloth (CPU): {reason}; got {key} = {value!r}. Unset it to train on CPU.")
    for key in _GPU_LOADER_KWARGS_IGNORED:
        kwargs.pop(key, None)
    # GPU loaders take rope_scaling = None as "unchanged"; transformers would apply it to the config.
    if "rope_scaling" in kwargs and kwargs["rope_scaling"] is None:
        kwargs.pop("rope_scaling")


# fp32 weights + fp32 grads + AdamW's two fp32 states, per parameter.
_FP32_MASTER_BYTES_PER_PARAM = 16


def _multi_process():
    """True under a launcher with several training processes (torchrun, accelerate, Intel MPI, Open MPI, MVAPICH) or
    an initialized process group of several ranks."""
    from .runtime import _local_world_size
    if _local_world_size() > 1:
        return True
    for name in ("WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE"):
        try:
            if int(os.environ.get(name, "")) > 1:
                return True
        except ValueError:
            pass
    try:
        import torch.distributed as dist
        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    except Exception:
        return False


def _fp32_master_fit(model_name, token, revision, trust_remote_code, resize_vocab = None, /, **kwargs):
    """(bytes, free RAM, fits) for float32 full finetuning: weights, grads and AdamW states take at most half of the
    free RAM; activations are sized separately (gradient checkpointing auto-off). ``kwargs`` are the model load's: an
    explicit config object decides, else the checkpoint's config with the load's overrides (as AutoModel reads it,
    ``vocab_size`` included). ``resize_vocab`` is a pending resize, counted when larger. Positional-only, so no load
    kwarg can collide with them. None when the parameter count or the free RAM is unknown."""
    try:
        from transformers import AutoConfig, PretrainedConfig
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, _ = AutoConfig.from_pretrained(model_name, token = token, revision = revision,
                                                   trust_remote_code = trust_remote_code, return_unused_kwargs = True,
                                                   **kwargs)
    except Exception:
        return None
    vocab_size = resize_vocab
    if resize_vocab is not None:
        loaded = getattr(getattr(config, "get_text_config", lambda: config)(), "vocab_size", None)
        if isinstance(loaded, int) and not isinstance(loaded, bool):
            vocab_size = max(loaded, resize_vocab)
    count = _dense_parameter_count(config, vocab_size = vocab_size)
    available = available_ram()
    if count is None or available is None:
        return None
    needed = count * _FP32_MASTER_BYTES_PER_PARAM
    return needed, available, needed * 2 <= available


def _print_full_finetuning_precision(dtype, choice, fit = None):
    """One line on full finetuning precision. bfloat16 weights take AdamW's bfloat16 update directly, so updates below
    a weight's bfloat16 spacing round away (SmolLM2-135M, lr 2e-5, 4 steps: 3.9% of the weights changed, against
    99.99% with float32 master weights)."""
    gb = lambda value: f"{value / 1024 ** 3:.1f} GB"  # noqa: E731
    if choice == "fits":
        _print(f"full finetuning keeps float32 master weights and computes in bfloat16 autocast: float32 weights, grads "
               f"and AdamW states ({gb(fit[0])}) fit in RAM ({gb(fit[1])} free), and bfloat16 weights round away "
               "small updates. float32_mixed_precision = False (or dtype = torch.bfloat16) keeps bfloat16 weights.")
    elif choice == "forced":
        if dtype == torch.float32 and cpu_bf16_supported():
            _print("float32_mixed_precision = True: full finetuning keeps float32 weights and optimizer states, "
                   "computing in bfloat16 autocast.")
    elif dtype == torch.bfloat16:
        if choice in ("too_big", "distributed"):
            if choice == "distributed":
                reason = "several training processes each read their own free RAM and could pick different dtypes"
            elif fit is not None:
                reason = f"float32 weights, grads and AdamW states ({gb(fit[0])}) do not fit in RAM ({gb(fit[1])} free)"
            else:
                reason = "the float32 training state could not be sized"
            _print(f"bfloat16 full finetuning: {reason}, so weights stay bfloat16 and updates below a weight's "
                   "bfloat16 spacing can be rounded away at small learning rates. float32_mixed_precision = True "
                   "forces float32 master weights.")
        else:
            _print("bfloat16 full finetuning (your dtype / float32_mixed_precision): halves memory, but updates below "
                   "a weight's bfloat16 spacing can be rounded away at small learning rates. float32_mixed_precision "
                   "= True keeps float32 master weights.")


def _dtype_kwarg():
    # from_pretrained took `torch_dtype` before transformers 4.56; older ones drop `dtype` silently.
    from transformers import __version__
    from unsloth_zoo.utils import Version
    return "dtype" if Version(__version__) >= Version("4.56.0") else "torch_dtype"


_DTYPE_KWARG = _dtype_kwarg()


def _print(message):
    print(f"Unsloth (CPU): {message}")


def _dense_parameter_count(config, vocab_size = None):
    """Parameters of a dense decoder from its config (embeddings, attention, gated MLP), None when unknown, MoE, or a
    per-layer layout (list widths, per-layer embeddings) the formula does not cover. ``vocab_size`` overrides the
    config's (a pending resize)."""
    try:
        text = getattr(config, "get_text_config", lambda: config)()
        get = lambda name: getattr(text, name, None)  # noqa: E731
        if any(get(name) for name in ("num_experts", "num_local_experts", "n_routed_experts", "moe_intermediate_size",
                                      "vocab_size_per_layer_input", "hidden_size_per_layer_input")):
            return None
        hidden, layers = get("hidden_size"), get("num_hidden_layers")
        vocab = vocab_size if vocab_size is not None else get("vocab_size")
        heads = get("num_attention_heads") or 1
        head_dim = get("head_dim") or (hidden // heads if isinstance(hidden, int) and isinstance(heads, int) else None)
        kv_heads = get("num_key_value_heads") or heads
        intermediate = get("intermediate_size") or (4 * hidden if isinstance(hidden, int) else None)
        if not all(isinstance(value, int) and not isinstance(value, bool)
                   for value in (hidden, layers, vocab, heads, head_dim, kv_heads, intermediate)):
            return None
        kv = kv_heads * head_dim
        per_layer = hidden * heads * head_dim * 2 + 2 * hidden * kv + 3 * hidden * intermediate
        embeddings = vocab * hidden * (1 if getattr(config, "tie_word_embeddings", True) else 2)
        return embeddings + layers * per_layer
    except Exception:
        return None


def _prefer_16bit_weights(model_name, dtype, token, revision, trust_remote_code, **kwargs):
    """4-bit on CPU dequantizes every weight in forward, backward and the checkpointed recompute (~12% more CPU time
    per step on SmolLM2-135M, no lower peak RSS at that size). When the 16-bit weights take at most a quarter of the
    free RAM, train on them instead. UNSLOTH_CPU_LOAD_IN_4BIT=1 keeps 4-bit. Returns the 16-bit size or None."""
    if os.environ.get("UNSLOTH_CPU_LOAD_IN_4BIT", "").strip().lower() in ("1", "true", "yes", "on"):
        return None
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, token = token, revision = revision,
                                            trust_remote_code = trust_remote_code, **kwargs)
    except Exception:
        return None
    count = _dense_parameter_count(config)
    available = available_ram()
    if count is None or available is None:
        return None
    size = count * torch.empty((), dtype = dtype).element_size()
    return size if size * 4 <= available else None


# float32 frozen weights may take at most this share of the free RAM (the rest: activations, LoRA, optimizer).
_FP32_FROZEN_RAM_SHARE = 0.25


def _fp32_frozen_storage_pays():
    """x86 without native bf16, on Windows and on Linux CPUs with AVX-512. There a bf16-stored weight's upcast on every
    forward, backward and decoded token was 30-34% of a 2-thread Qwen3-0.6B GRPO step (aten::copy_, Xeon 8370C /
    8573C), and float32 storage made GRPO 9-21% faster on Windows. On a Linux AVX-512 Xeon, decoding through float32
    weights was also faster than the per-call upcast (59 vs 101 ms per Qwen3-0.6B token pass). On AVX2-only Linux x86
    (EPYC 7763) decoding through float32 reads twice the bytes and GRPO was 1-6% slower, so bf16 storage stays there;
    Apple Silicon has its fp32 rollout instead."""
    import platform
    if platform.machine().lower() not in ("amd64", "x86_64"):
        return False
    if sys.platform == "win32":
        return True
    if sys.platform.startswith("linux"):
        from .runtime import cpu_flags
        return "avx512f" in cpu_flags()
    return False


def _fp32_weights_fit(model_name, token, revision, trust_remote_code, **kwargs):
    """(bytes, free RAM) when float32 frozen storage pays on this host (``_fp32_frozen_storage_pays``) and the whole
    model in float32 takes at most a quarter of the free RAM, else None (also when either is unknown). Then the frozen
    projections are kept in float32 instead of bfloat16, at 2 more bytes per frozen weight. None under several training
    processes: each would read the same free RAM for its own copy."""
    if not _fp32_frozen_storage_pays() or _multi_process():
        return None
    try:
        from transformers import AutoConfig
        config = kwargs.get("config")
        if config is None:
            config = AutoConfig.from_pretrained(model_name, token = token, revision = revision,
                                                trust_remote_code = trust_remote_code,
                                                **{k: v for k, v in kwargs.items() if k != "config"})
    except Exception:
        return None
    count = _dense_parameter_count(config)
    available = available_ram()
    if count is None or available is None:
        return None
    size = 4 * count
    return (size, available) if size <= _FP32_FROZEN_RAM_SHARE * available else None


def _checkpoint_dtype(model_name, **kwargs):
    """The dtype the checkpoint's weights are stored in, from its config (None when unknown)."""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, **kwargs)
    except Exception:
        return None
    for cfg in (config, getattr(config, "get_text_config", lambda: config)()):
        value = getattr(cfg, "dtype", None) or getattr(cfg, "torch_dtype", None)
        if isinstance(value, str):
            value = getattr(torch, value.replace("torch.", ""), None)
        if isinstance(value, torch.dtype):
            return value
    return None


# Where composite configs keep the language model's config (and so its quantization_config).
_SUB_CONFIG_KEYS = ("text_config", "language_config", "llm_config", "decoder", "generator")


def _quant_method_of(config):
    """quant_method of the quantization_config in ``config`` (a dict or a PretrainedConfig), at its root or in a
    language-model sub-config, else None."""
    def field(holder, name):
        return holder.get(name) if isinstance(holder, dict) else getattr(holder, name, None)

    if config is None:
        return None
    for holder in (config, *(field(config, key) for key in _SUB_CONFIG_KEYS)):
        if holder is None:
            continue
        quant = field(holder, "quantization_config")
        if quant is not None and not isinstance(quant, dict):
            quant = quant.to_dict() if hasattr(quant, "to_dict") else {"quant_method": getattr(quant, "quant_method", None)}
        method = quant.get("quant_method") if isinstance(quant, dict) else None
        if method:
            return str(getattr(method, "value", method)).lower()
    return None


def _checkpoint_quant_method(model_name, config = None, **kwargs):
    """quant_method of a quantization_config the checkpoint ships, else None. An explicit ``config`` is what the
    model load uses, so it decides; ``kwargs`` carry the hub arguments (``subfolder`` included)."""
    if config is not None and not isinstance(config, (str, os.PathLike)):
        return _quant_method_of(config)
    try:
        from transformers import PretrainedConfig

        config_dict, _ = PretrainedConfig.get_config_dict(config or model_name, **kwargs)
    except Exception:
        return None
    return _quant_method_of(config_dict) if isinstance(config_dict, dict) else None


def _is_fp8_dequantize(op):
    # By type name: transformers.integrations.finegrained_fp8.Fp8Dequantize, without importing that module.
    return type(op).__name__ == "Fp8Dequantize"


def _has_fp8_dequantize_conversion(model):
    """True when a load conversion of ``model`` dequantizes FP8 (transformers' Fp8Dequantize operation)."""
    try:
        return any(
            _is_fp8_dequantize(op)
            for conversion in getattr(model, "_weight_conversions", None) or ()
            for op in getattr(conversion, "operations", None) or ()
        )
    except Exception:
        return False


def _drop_fp8_dequantize_conversions(model):
    """Mirror of unsloth.models.loader_utils._drop_fp8_dequantize_conversions (unimportable without Triton)."""
    conversions = getattr(model, "_weight_conversions", None)
    if not conversions:
        return 0

    def _strip(pattern):
        return pattern[:-1] if pattern.endswith("$") else pattern

    def _targets(conversion):
        return tuple(_strip(p) for p in getattr(conversion, "_original_target_patterns", None) or [])

    def _sources(patterns):
        patterns = [_strip(p) for p in patterns or []]
        return tuple(sorted(p for p in patterns if not p.endswith((".weight_scale_inv", ".activation_scale"))))

    def _source_forms(conversion):
        return {
            _sources(getattr(conversion, "_original_source_patterns", None)),
            _sources(getattr(conversion, "source_patterns", None)),
        }

    def _ops(conversion):
        operations = getattr(conversion, "operations", None) or []
        return tuple(type(op).__name__ for op in operations if not _is_fp8_dequantize(op))

    def _signature(conversion):
        return (
            type(conversion),
            getattr(conversion, "scope_prefix", None),
            min(_source_forms(conversion)),
            _targets(conversion),
            _ops(conversion),
        )

    def _has_dequant(conversion):
        return any(_is_fp8_dequantize(op) for op in getattr(conversion, "operations", None) or [])

    plain = None
    kept, changed = [], 0
    retained = {_signature(c) for c in conversions if not _has_dequant(c)}
    for conversion in conversions:
        operations = getattr(conversion, "operations", None) or []
        if not _has_dequant(conversion):
            kept.append(conversion)
            continue
        if getattr(getattr(model, "config", None), "quantization_config", None) is None:
            model._unsloth_fp8_dequantized = True
        if all(_is_fp8_dequantize(op) for op in operations):
            changed += 1
            continue
        if plain is None:
            try:
                from transformers.conversion_mapping import get_model_conversion_mapping

                plain = get_model_conversion_mapping(model, None, None)
            except Exception:
                plain = []
        # Dequantization can lose scope_prefix; restore every matching scoped counterpart.
        scope = getattr(conversion, "scope_prefix", None)
        sources = _source_forms(conversion)
        matches = [
            c
            for c in plain
            if type(c) is type(conversion)
            and _targets(c) == _targets(conversion)
            and _source_forms(c) & sources
            and _ops(c) == _ops(conversion)
            and (scope is None or getattr(c, "scope_prefix", None) == scope)
        ]
        if not matches:
            _print(
                "an fp8 load conversion has no 16bit counterpart; a merged save may write fp8 weights for "
                f"{list(_targets(conversion))}."
            )
            kept.append(conversion)
            continue
        for match in matches:
            signature = _signature(match)
            if signature not in retained:
                retained.add(signature)
                kept.append(match)
        changed += 1
    if changed:
        model._weight_conversions = kept
    return changed


def _drop_reverse_fp8_conversions(model):
    """Drop kept Fp8Dequantize conversions, which a merged save reverses into FP8 bytes without FP8 metadata."""
    if not _has_fp8_dequantize_conversion(model):
        return 0
    try:
        return _drop_fp8_dequantize_conversions(model)
    except Exception as error:
        _print(f"could not drop the reverse fp8 conversions ({error}); a merged save may write fp8 weights.")
        return 0


def _resolve_dtype(dtype):
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype.replace("torch.", ""), None)
    if dtype is None:
        return torch.bfloat16 if cpu_bf16_supported() else torch.float32
    if dtype == torch.float16:
        # fp16 GEMMs are emulated on almost every CPU and fp16 training needs a grad scaler.
        replacement = torch.bfloat16 if cpu_bf16_supported() else torch.float32
        _print(f"float16 is slow and unstable on CPU, using {replacement} instead.")
        return replacement
    if dtype == torch.bfloat16 and not cpu_bf16_supported():
        _print("this CPU has no native bfloat16 (AMX / AVX512-BF16 / Arm BF16); bfloat16 will be emulated and slow.")
    return dtype


def _compile_layers(model):
    """torch.compile (Inductor C++) per decoder layer: one graph reused by every layer."""
    inner = getattr(model, getattr(model, "base_model_prefix", "model"), None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        return 0
    dynamic = os.environ.get("UNSLOTH_CPU_COMPILE_DYNAMIC", "1") == "1"
    for layer in layers:
        layer.compile(dynamic = dynamic)
    _fall_back_to_eager_on_compile_error(layers)
    return len(layers)


def _is_control_flow(error):
    """Exceptions torch raises on purpose through a layer call: non-reentrant checkpointing stops a recomputation
    early with _StopRecomputationError once it has every saved tensor."""
    import torch.utils.checkpoint as checkpoint
    stop = getattr(checkpoint, "_StopRecomputationError", None)
    return stop is not None and isinstance(error, stop)


def _first_error_line(error, limit):
    """Dynamo's BackendCompilerFailed reads "backend='inductor' raised:" on its first line; the cause is the next one."""
    lines = [line.strip() for line in str(error).strip().splitlines() if line.strip()]
    if len(lines) > 1 and lines[0].endswith(":"):
        return f"{lines[0]} {lines[1]}"[:limit]
    return lines[0][:limit] if lines else ""


@contextlib.contextmanager
def _backward_lowered_in_forward():
    """AOTAutograd lowers the backward graph lazily, at the first backward, when the forward saves no symbolic sizes,
    and re-defers it after a failed eager attempt otherwise; a backward that cannot build would then raise from
    loss.backward(), out of reach of the fallback. Forcing the lowering into the forward call puts it inside it.
    Entered only with grad enabled (no backward graph otherwise), so generate() and eval skip it."""
    config = getattr(getattr(torch, "_functorch", None), "config", None)
    if config is None or not hasattr(config, "force_non_lazy_backward_lowering"):
        yield  # older torch: a backward lowering error still surfaces in backward
        return
    previous = config.force_non_lazy_backward_lowering
    config.force_non_lazy_backward_lowering = True
    try:
        yield
    finally:
        config.force_non_lazy_backward_lowering = previous


def _in_backward():
    """True inside autograd's backward pass, where checkpointing recomputes a layer's forward."""
    current = getattr(torch._C, "_current_graph_task_id", None)
    try:
        return current is not None and current() != -1
    except Exception:
        return False


_CACHE_KWARGS = ("past_key_values", "past_key_value", "layer_past", "cache_params", "past_key_values_cache")


def _carries_cache(args, kwargs):
    """A decoder layer call that would write into a KV / state cache."""
    if any(kwargs.get(name) is not None for name in _CACHE_KWARGS):
        return True
    try:
        from transformers.cache_utils import Cache
    except Exception:
        return False
    return any(isinstance(value, Cache) for value in (*args, *kwargs.values()))


def _checkpoints_inside_forward(layer):
    """transformers' GradientCheckpointingLayer opens its checkpoint in __call__ up to 5.22 and wraps ``forward`` from
    5.23 on (module hooks stay outside the checkpoint), so a compiled layer call then carries its own checkpoint."""
    try:
        from transformers.modeling_layers import GradientCheckpointingLayer
    except Exception:
        return False
    # __init_subclass__, not a missing __call__: temporary_patches.misc installs its own __call__ on every version.
    return isinstance(layer, GradientCheckpointingLayer) and "__init_subclass__" in GradientCheckpointingLayer.__dict__


class _RestartEagerly(Exception):
    """Raised out of a layer's failed compiled call so its __call__ reruns the whole call, checkpoint included."""


_RESTARTABLE_CLASSES = {}


def _restartable(layer):
    """Gives ``layer`` a __call__ that reruns itself once on _RestartEagerly. transformers' GradientCheckpointingLayer
    opens the checkpoint in __call__, so the rerun gets a fresh checkpoint instead of one still holding what a
    partly compiled forward (a graph break, then a segment that fails to compile) saved before the failure."""
    cls = type(layer)
    if getattr(cls, "_unsloth_restartable", False):
        return
    sub = _RESTARTABLE_CLASSES.get(cls)
    if sub is None:
        def __call__(self, *args, **kwargs):
            self.__dict__["_unsloth_restart_depth"] = self.__dict__.get("_unsloth_restart_depth", 0) + 1
            try:
                try:
                    return cls.__call__(self, *args, **kwargs)
                except _RestartEagerly as error:
                    # The aborted checkpoint's generator still holds its saved-tensor hooks (non-reentrant
                    # checkpoint never closes it when the function raises) and the traceback keeps it alive: drop
                    # both before the retry, or its tensors land in the aborted frame and backward raises.
                    error.__traceback__ = None
                    del error
                gc.collect()
                return cls.__call__(self, *args, **kwargs)
            finally:
                self.__dict__["_unsloth_restart_depth"] -= 1
        sub = type(cls.__name__, (cls,), {
            "__call__": __call__, "_unsloth_restartable": True,
            "__module__": cls.__module__, "__qualname__": cls.__qualname__,
        })
        _RESTARTABLE_CLASSES[cls] = sub
    layer.__class__ = sub


def _fall_back_to_eager_on_compile_error(layers):
    """The toolchain probe compiles one small kernel; a decoder layer can still fail to trace or build (an op or a
    patched module Dynamo / Inductor cannot handle; Dynamo can raise plain AssertionErrors). The first failure of a
    compiled call drops every layer back to eager and restarts that layer call eagerly (its checkpoint included, so
    nothing a compiled segment saved before the failure stays behind), so a genuine model error still raises, now
    from eager code. A layer whose compiled forward is still waiting for its checkpoint recomputation
    (an earlier layer of the same step) replays it compiled once more: an eager recompute saves different tensors
    and checkpointing raises CheckpointError. It goes eager after that."""
    layers = list(layers)
    state = {"eager": False}
    # id(layer) -> one entry per compiled checkpointed forward whose backward has not recomputed it yet, keyed by
    # its first tensor input: two checkpoints of one layer can be outstanding, and only the compiled one replays
    # compiled (an eagerly retried one must recompute eagerly).
    pending = {}
    # id(layer) -> checkpointed forwards rerun eagerly after an out-of-memory compiled call, keyed like pending, until
    # recomputed or their input is freed (a layer that was not really checkpointed). While any is outstanding the layer
    # runs eagerly, as after a compile failure, so checkpoints sharing an input never mix modes; then it compiles again.
    eager_window = {}

    def first_tensor(args):
        for value in args:
            if isinstance(value, torch.Tensor):
                return value
        return None

    def remember(key, args, pending = pending):
        x = first_tensor(args)
        entry = (None, None, None) if x is None else (weakref.ref(x), x.data_ptr(), tuple(x.shape))
        entries = pending.setdefault(key, [])
        entries.append(entry)
        del entries[:-64]  # forwards never backpropagated (grad-enabled eval) must not pile up

    def window_open(key):
        entries = eager_window.get(key)
        if entries:
            entries[:] = [entry for entry in entries if entry[0]() is not None]
        return bool(entries)

    def take(key, args, pending = pending):
        entries = pending.get(key)
        if not entries:
            return False
        x = first_tensor(args)
        for index in range(len(entries) - 1, -1, -1):
            ref, ptr, shape = entries[index]
            alive = None if ref is None else ref()
            if (
                ref is None
                or x is None
                or alive is x
                # reentrant checkpointing recomputes on detached copies: same storage and shape
                or (alive is None and ptr == x.data_ptr() and shape == tuple(x.shape))
            ):
                del entries[index]
                return True
        return False

    def drop(layer):
        if not pending.get(id(layer)):
            layer._compiled_call_impl = None

    def disable(error):
        if not state["eager"]:
            state["eager"] = True
            _print(f"torch.compile failed on a decoder layer ({type(error).__name__}: "
                   f"{_first_error_line(error, 160)}), training eagerly.")
        for layer in layers:
            drop(layer)

    def eager_once(key, layer, args, kwargs):
        out = layer._call_impl(*args, **kwargs)
        if not _in_backward() and checkpointed(layer) and first_tensor(args) is not None:
            remember(key, args, eager_window)
        return out

    def checkpointed(layer):
        # Only a checkpoint opened around the layer call recomputes through it. transformers >= 5.23 checkpoints
        # inside forward instead, so the recomputation never comes back here and there is nothing to replay.
        if _checkpoints_inside_forward(layer):
            return False
        return torch.is_grad_enabled() and layer.training and getattr(layer, "gradient_checkpointing", True)

    for layer in layers:
        compiled = layer._compiled_call_impl
        if compiled is None:
            continue

        def guarded(*args, _compiled = compiled, _layer = layer, **kwargs):
            key = id(_layer)
            backward = _in_backward()
            if backward and take(key, args):
                # Checkpoint recomputation of a compiled forward: replay it compiled, as the forward saved it.
                if state["eager"]:
                    drop(_layer)
                return _compiled(*args, **kwargs)
            if backward and window_open(key) and take(key, args, eager_window):
                # One of the eager out-of-memory retries recomputing: eagerly, as it ran.
                return _layer._call_impl(*args, **kwargs)
            if _layer.__dict__.pop("_unsloth_eager_once", False) or (not backward and window_open(key)):
                return eager_once(key, _layer, args, kwargs)  # the restart after an out-of-memory compiled call
            if state["eager"] or _carries_cache(args, kwargs):
                # A KV cache is mutated in place (update() appends): a compiled segment that ran before a later one
                # failed could not be undone by a retry, so cached forwards (inference) always run eagerly.
                return _layer._call_impl(*args, **kwargs)
            try:
                if not torch.is_grad_enabled():
                    out = _compiled(*args, **kwargs)
                else:
                    with _backward_lowered_in_forward():
                        out = _compiled(*args, **kwargs)
            except Exception as error:
                if _is_control_flow(error) or isinstance(error, _RestartEagerly):
                    raise
                if _is_out_of_memory(error):
                    # Not a compile failure: this call reruns eagerly (restarted like below), the next one compiles.
                    if not backward and _layer.__dict__.get("_unsloth_restart_depth", 0) > 0:
                        _layer.__dict__["_unsloth_eager_once"] = True
                        raise _RestartEagerly() from error
                    return eager_once(key, _layer, args, kwargs)
                disable(error)
                if not backward and _layer.__dict__.get("_unsloth_restart_depth", 0) > 0:
                    # A compiled segment before a graph break may already have saved tensors into this call's
                    # checkpoint: start the call over (eager now) rather than rerun inside it.
                    raise _RestartEagerly() from error
                return _layer._call_impl(*args, **kwargs)
            if not backward and checkpointed(_layer):
                remember(key, args)
            return out

        layer._compiled_call_impl = guarded
        try:
            _restartable(layer)
        except TypeError:
            pass  # a class that cannot be subclassed / reassigned keeps the in-place eager rerun


def _load_msvc_environment():
    """Windows: Inductor needs `cl` on PATH (a Developer Prompt); load vcvars64.bat's environment. True if cl is reachable."""
    import shutil
    import subprocess
    if shutil.which("cl"):
        return True
    root = os.environ.get("ProgramFiles(x86)") or r"C:\Program Files (x86)"
    vswhere = os.path.join(root, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.isfile(vswhere):
        return False
    hidden = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    try:
        done = subprocess.run(
            [vswhere, "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
             "-property", "installationPath", "-utf8"],
            capture_output = True, text = True, encoding = "utf-8", errors = "replace", timeout = 60, **hidden,
        )
        install = (done.stdout or "").strip().splitlines()
        if done.returncode != 0 or not install:
            return False
        vcvars = os.path.join(install[0].strip(), "VC", "Auxiliary", "Build", "vcvars64.bat")
        if not os.path.isfile(vcvars):
            return False
        # /u: UTF-16 `set` output, so non-ASCII PATH entries survive.
        done = subprocess.run(f'cmd /d /u /s /c ""{vcvars}" >nul 2>&1 && set"', capture_output = True,
                              timeout = 120, **hidden)
    except Exception:
        return False
    if done.returncode != 0:
        return False
    env = {}
    for line in (done.stdout or b"").decode("utf-16-le", errors = "replace").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.upper() in ("PATH", "INCLUDE", "LIB", "LIBPATH"):
            env[key.upper()] = value
    if "PATH" not in env or "INCLUDE" not in env:
        return False
    os.environ.update(env)
    return bool(shutil.which("cl"))


@functools.cache
def _cpu_compile_unavailable():
    """None when Inductor can build and run a CPU kernel here (it needs a C++ compiler at run time), else why not."""
    import sys
    if sys.platform == "win32" and not _load_msvc_environment():
        return ("no MSVC: install Visual Studio Build Tools with the C++ workload "
                "(winget install Microsoft.VisualStudio.2022.BuildTools --override \"--add "
                "Microsoft.VisualStudio.Workload.VCTools --includeRecommended\")")
    if os.environ.get("TORCHDYNAMO_DISABLE", "").strip().lower() in ("1", "true"):
        # Studio's Windows Triton gate sets it; torch.compile then runs eagerly and the probe below would still pass.
        return "TORCHDYNAMO_DISABLE=1"
    try:
        import torch._dynamo
        if getattr(torch._dynamo.config, "disable", False):
            return "torch._dynamo.config.disable is set"
        x = torch.linspace(-4, 4, 64)  # not randn: the probe must not move the RNG that seeds LoRA init
        out = torch.compile(lambda t: torch.nn.functional.silu(t) * 2, dynamic = False)(x)
        if not torch.allclose(out, torch.nn.functional.silu(x) * 2):
            return "probe kernel gave wrong results"
        return None
    except Exception as error:
        return f"{type(error).__name__}: {_first_error_line(error, 200)}"
    finally:
        try:
            torch._dynamo.reset()
        except Exception:
            pass


# Checked compiled vs eager in fp32 (tiny Llama / Mistral / Qwen2 / Qwen3 / Phi-3 / Gemma / OLMo-2, SmolLM2-135M).
_CPU_COMPILE_MODEL_TYPES = frozenset(("llama", "mistral", "qwen2", "qwen3", "phi3", "gemma", "olmo2"))


def _parallel_kernel_builds():
    """Windows: build a compile's C++ kernels in parallel. torch sets compile_threads = 1 on win32 for its Triton
    worker pool, so MSVC builds ran one at a time (cold Qwen3 layer: 45 s first step on 4 vCPU, 21 s parallel). CPU
    kernels build on Inductor's in-process thread pool; the Triton pool stays off (TORCH_WARM_POOL=0). A user's
    TORCHINDUCTOR_COMPILE_THREADS, a raised compile_threads or UNSLOTH_FORCE_SINGLE_COMPILE_WORKER=1 is kept."""
    if sys.platform != "win32" or "TORCHINDUCTOR_COMPILE_THREADS" in os.environ:
        return None
    if os.environ.get("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", "0") == "1":
        return None
    try:
        import torch._inductor.config as inductor_config
    except Exception:
        return None
    if getattr(inductor_config, "compile_threads", 1) not in (None, 1):
        return None
    count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    threads = min(32, count)
    if threads <= 1:
        return None
    os.environ.setdefault("TORCH_WARM_POOL", "0")
    inductor_config.compile_threads = threads
    return threads


def compile_cpu_layers(model, explicit = False):
    """Compiles the decoder layers when Inductor can build CPU kernels here. Returns the layer count (0 = eager)."""
    if getattr(model, "_unsloth_cpu_compiled", False):
        return 0
    reason = _cpu_compile_unavailable()
    if reason is not None:
        if explicit:
            _print(f"UNSLOTH_CPU_COMPILE=1 ignored, training eagerly: torch.compile cannot build CPU kernels here ({reason}).")
        else:
            _print(f"torch.compile cannot build CPU kernels here ({reason}), training eagerly.")
        return 0
    _parallel_kernel_builds()
    count = _compile_layers(model)
    model._unsloth_cpu_compiled = True
    _print(f"torch.compile (Inductor C++) on {count} decoder layers." + ("" if explicit else " UNSLOTH_CPU_COMPILE=0 trains eagerly."))
    return count


def _cpu_compile_wanted(model):
    """"explicit" for UNSLOTH_CPU_COMPILE=1, "auto" for the dense architectures whose compiled layers were checked
    against eager (fp32 loss equal, gradients within 1e-6 relative; ~24% less CPU time per step after a one-off
    compile), None otherwise or for UNSLOTH_CPU_COMPILE=0. Quantized models stay eager (bitsandbytes ops)."""
    value = os.environ.get("UNSLOTH_CPU_COMPILE", "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return "explicit"
    if value in ("0", "false", "no", "off"):
        return None
    config = getattr(model, "config", None)
    if getattr(config, "quantization_config", None) is not None or getattr(model, "is_loaded_in_4bit", False) \
            or getattr(model, "is_loaded_in_8bit", False):
        return None
    text = getattr(config, "get_text_config", lambda: config)()
    if getattr(text, "model_type", None) not in _CPU_COMPILE_MODEL_TYPES:
        return None
    try:
        return "auto" if _plain_lm_head(model) else None
    except Exception:
        return None


def _non_reentrant_checkpointing_on(model):
    # Re-enabling would drop an every_n_layers selection; reentrant or unknown funcs still get switched.
    if not _is_activation_checkpointing(model):
        return False
    funcs = [m.__dict__["_gradient_checkpointing_func"] for m in model.modules() if "_gradient_checkpointing_func" in m.__dict__]
    return bool(funcs) and all(isinstance(getattr(f, "keywords", None), dict) and f.keywords.get("use_reentrant", True) is False for f in funcs)


def _disable_bnb_cpu_weight_packing(model):
    """bitsandbytes (0.50) repacks a Linear4bit weight in place for its AVX512-BF16 kernel on the first
    eval forward without grad, and never unpacks it: backward through the packed weight is then wrong,
    so a single evaluation or generate() before / during training silently stalls QLoRA learning.
    Keep every layer on the trainable layout."""
    count = 0
    for module in model.modules():
        if hasattr(module, "support_avx512bf16_for_cpu"):
            module.support_avx512bf16_for_cpu = False
            count += 1
    return count


class FastCPUModel:
    """FastLanguageModel for hosts with no accelerator: stock transformers + PEFT on CPU."""

    @staticmethod
    def from_pretrained(
        model_name = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
        load_in_8bit = False,
        load_in_16bit = False,
        full_finetuning = False,
        token = None,
        device_map = None,
        trust_remote_code = False,
        revision = None,
        fast_inference = False,
        use_gradient_checkpointing = "unsloth",
        attn_implementation = None,
        **kwargs,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if fast_inference:
            raise NotImplementedError("Unsloth (CPU): fast_inference (vLLM) needs a GPU.")
        # A quantization config the caller built is a choice of quantization: never swapped for 16-bit weights.
        user_quantization = kwargs.get("quantization_config", None) is not None
        # Full finetuning on a native-bf16 CPU: True keeps float32 master weights, False bfloat16 weights; unset (and no
        # dtype) picks float32 when its training state fits in RAM.
        float32_mixed_precision = kwargs.get("float32_mixed_precision", None)
        # Before anything downloads: a refused GPU-only feature fails fast.
        _pop_gpu_loader_kwargs(kwargs)
        # Chunked CE returns logits = None in training steps; return_logits keeps the full-logit loss.
        return_logits = kwargs.pop("return_logits", False)
        # A separate tokenizer repo, as FastLanguageModel accepts on GPU; never a model kwarg.
        tokenizer_name = kwargs.pop("tokenizer_name", None) or model_name
        if load_in_8bit:
            _print("8-bit loading is not supported on CPU, loading 16-bit weights instead.")
        if token is None:
            token = os.environ.get("HF_TOKEN")
        # The transformers spelling: honoured, and never passed twice next to the resolved dtype.
        torch_dtype = kwargs.pop("torch_dtype", None)
        if dtype is None and torch_dtype is not None:
            dtype = torch_dtype
        user_dtype = dtype is not None
        master_fp32 = bool(full_finetuning) and float32_mixed_precision is True
        master_choice = "forced" if master_fp32 else "user"
        # "auto" keeps the checkpoint's own dtype (fp16 checkpoints still move off fp16 after loading).
        auto_dtype = isinstance(dtype, str) and dtype == "auto" and not master_fp32
        # Forced float32 master weights load as float32 straight from the checkpoint, whatever dtype was asked for.
        dtype = torch.float32 if master_fp32 else _resolve_dtype(None if auto_dtype else dtype)

        # 4-bit only where bitsandbytes' CPU backend really runs; full finetuning never quantizes.
        use_4bit = bool(load_in_4bit) and not load_in_8bit and not load_in_16bit and not full_finetuning
        if use_4bit and kwargs.get("gguf_file"):
            # transformers refuses a BitsAndBytesConfig beside a GGUF, whose weights are unpacked at load anyway.
            _print("`gguf_file` weights are dequantized at load, so load_in_4bit is ignored.")
            use_4bit = False
        # A saved LoRA checkpoint (notebooks reload "lora_model"): load its base, then the adapter on top,
        # trainable, so training resumes and merged exports see a PeftModel.
        adapter_name = adapter_revision = None
        hub_kwargs = {k: kwargs[k] for k in ("cache_dir", "local_files_only") if k in kwargs}
        adapter_kwargs = dict(hub_kwargs, **({"subfolder": kwargs["subfolder"]} if kwargs.get("subfolder") else {}))
        base = _adapter_base_model(model_name, token, revision, **adapter_kwargs)
        if base is not None:
            # The caller's revision names the adapter repo's branch; the base keeps the one the adapter recorded.
            adapter_name, adapter_revision = model_name, revision
            model_name, revision = base
            # A subfolder locates the adapter inside its repo, not the base model.
            kwargs.pop("subfolder", None)
            _print(f"{adapter_name} is a LoRA adapter; loading its base {model_name} first.")
        # The config the model load reads: an explicit config= decides, else the checkpoint's (in its subfolder).
        probe_kwargs = dict(hub_kwargs, **({"subfolder": kwargs["subfolder"]} if kwargs.get("subfolder") else {}))
        if use_4bit:
            # A checkpoint quantized with another method (fp8, compressed-tensors, mxfp4) keeps its own config, as
            # on GPU: transformers refuses a BitsAndBytesConfig on top of it.
            quant_method = _checkpoint_quant_method(
                model_name, config = kwargs.get("config"), token = token, revision = revision, **probe_kwargs,
            )
            if quant_method not in (None, "bitsandbytes"):
                _print(f"{model_name} is already quantized with {quant_method}; loading it with that "
                       f"quantization config instead of bitsandbytes 4-bit.")
                use_4bit = False
        if use_4bit and not cpu_bnb_4bit_supported():
            _print("bitsandbytes 4-bit does not work on this CPU, loading 16-bit weights instead.")
            use_4bit = False
        if use_4bit and not user_quantization and kwargs.get("config") is None \
                and not _PREQUANTIZED_SUFFIX.search(model_name) and quant_method is None:
            # A pre-quantized repo was picked to download less; its 16-bit base would be a new download.
            size = _prefer_16bit_weights(model_name, dtype, token, revision, trust_remote_code, **probe_kwargs)
            if size is not None:
                _print(f"load_in_4bit = True, but loading 16-bit weights instead: they fit in RAM "
                       f"({size / 1024 ** 3:.2f} GB of {available_ram() / 1024 ** 3:.0f} GB free) and train faster than "
                       "4-bit on CPU. Set UNSLOTH_CPU_LOAD_IN_4BIT=1 to keep 4-bit.")
                use_4bit = False
        if not use_4bit and _PREQUANTIZED_SUFFIX.search(model_name):
            base = _PREQUANTIZED_SUFFIX.sub("", model_name)
            _print(f"{model_name} is pre-quantized; loading {base} for 16-bit training.")
            model_name = base
        # Studio's progress / stall-watchdog callback, never a model kwarg; adapter before base, as on GPU.
        on_model_resolved = kwargs.pop("on_model_resolved", None)
        if on_model_resolved is not None:
            if adapter_name is not None and adapter_name != model_name:
                on_model_resolved(adapter_name)
            on_model_resolved(model_name)
        resize_model_vocab = kwargs.pop("resize_model_vocab", None)
        master_fit = None
        if full_finetuning and float32_mixed_precision is None and not user_dtype and dtype == torch.bfloat16:
            # No dtype and no float32_mixed_precision from the caller: float32 master weights when they fit, sized on
            # the checkpoint that loads (after the redirect above), its overrides and a pending vocabulary resize.
            # One process only: ranks reading different free RAM would pick different dtypes.
            if _multi_process():
                master_choice = "distributed"
            else:
                master_fit = _fp32_master_fit(model_name, token, revision, trust_remote_code, resize_model_vocab,
                                              **kwargs)
                master_choice = "too_big"
                if master_fit is not None and master_fit[2]:
                    master_fp32, master_choice, dtype = True, "fits", torch.float32

        # dtype = "auto" loads the checkpoint's dtype (float16 moves to bfloat16 / float32 after loading, as below).
        loaded_dtype = dtype
        if auto_dtype and full_finetuning:
            loaded_dtype = _checkpoint_dtype(model_name, token = token, revision = revision,
                                             trust_remote_code = trust_remote_code, **probe_kwargs)
            if loaded_dtype == torch.float16:
                loaded_dtype = torch.bfloat16 if cpu_bf16_supported() else torch.float32
        if full_finetuning and loaded_dtype == torch.float32 and not (master_fit is not None and master_fit[2]) \
                and not _multi_process():
            # float32 weights, grads and AdamW states (every CPU without native bf16: Apple Silicon, AVX2 x86) that do not
            # fit in RAM swap: Qwen3-0.6B on a 7 GB Apple M1 took 266 s per step instead of about 3 s.
            fp32_fit = master_fit or _fp32_master_fit(model_name, token, revision, trust_remote_code,
                                                      resize_model_vocab, **kwargs)
            if fp32_fit is not None and fp32_fit[0] > fp32_fit[1]:
                _print(f"float32 full finetuning needs about {fp32_fit[0] / 1024 ** 3:.1f} GB for weights, grads and "
                       f"AdamW states, but only {fp32_fit[1] / 1024 ** 3:.1f} GB of RAM is free: training will swap "
                       "and can be 100x slower. LoRA (get_peft_model) or a smaller model fits.")

        quantization_config = None
        if use_4bit and not _PREQUANTIZED_SUFFIX.search(model_name):
            from unsloth_zoo.peft_utils import SKIP_QUANTIZATION_MODULES
            quantization_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_quant_type = "nf4",
                bnb_4bit_use_double_quant = True,
                bnb_4bit_compute_dtype = dtype,
                # As the GPU loaders: lm_head, MoE routers, Mamba and vision / audio towers stay 16-bit.
                llm_int8_skip_modules = list(SKIP_QUANTIZATION_MODULES),
            )

        if quantization_config is None and kwargs.get("gguf_file"):
            quantization_config = _gguf_dequantize_config()
        if quantization_config is not None:
            # Only when set: an explicit None also discards a pre-quantized checkpoint's own config.
            kwargs["quantization_config"] = quantization_config
        # Left to transformers when unset: SDPA where the class supports it, eager otherwise (Bloom, MPT, ...).
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        # fp32 training of a bf16 checkpoint can keep the frozen projections in bf16 (lossless, half their RAM), at the
        # cost of an upcast per call. Default where fp32 is the default (no native bf16), except on Windows x86 and
        # AVX-512 Linux x86 when the fp32 weights fit in a quarter of the free RAM; UNSLOTH_CPU_FROZEN_BF16=1 / 0 forces
        # it on / off, =training keeps it whatever the RAM and OS (bf16 for training, fp32 for generation where that pays).
        frozen_env = os.environ.get("UNSLOTH_CPU_FROZEN_BF16", "").strip().lower()
        frozen_wanted = frozen_env in ("1", "true", "yes", "on", "training") or (
            frozen_env not in ("0", "false", "no", "off") and not cpu_bf16_supported()
        )
        frozen_bf16 = (
            frozen_wanted and dtype == torch.float32 and not auto_dtype
            and not use_4bit and not full_finetuning
            and _checkpoint_dtype(model_name, token = token, revision = revision,
                                  trust_remote_code = trust_remote_code, **probe_kwargs) == torch.bfloat16
        )
        frozen_fp32 = False
        if frozen_bf16 and frozen_env not in ("1", "true", "yes", "on", "training"):
            fp32_fit = _fp32_weights_fit(model_name, token, revision, trust_remote_code,
                                         **probe_kwargs, **({"config": kwargs["config"]} if "config" in kwargs else {}))
            if fp32_fit is not None:
                # Still loaded in bf16, then upcast one tensor at a time: a float32 load of a bf16 checkpoint peaked at
                # 4.7 GB RSS on Qwen3-0.6B, against 1.9 GB for the bf16 load.
                frozen_bf16, frozen_fp32 = False, True
                _print(f"frozen weights kept in float32: they fit in RAM ({fp32_fit[0] / 1024 ** 3:.2f} GB of "
                       f"{fp32_fit[1] / 1024 ** 3:.0f} GB free), and bfloat16 storage would upcast them on every "
                       "call on this CPU. UNSLOTH_CPU_FROZEN_BF16=training keeps bfloat16 storage.")
        load_kwargs = dict(
            **{_DTYPE_KWARG: "auto" if auto_dtype else (torch.bfloat16 if frozen_bf16 or frozen_fp32 else dtype)},
            device_map = "cpu",
            token = token,
            revision = revision,
            trust_remote_code = trust_remote_code,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = None
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs, **kwargs)
            except RuntimeError as error:
                # bitsandbytes' CPU packing rejects very narrow layers (hidden size < 64).
                if quantization_config is None or not use_4bit:
                    raise
                _print(f"bitsandbytes 4-bit failed on CPU ({error}), loading 16-bit weights instead.")
            if model is None:
                # Outside the handler: its traceback keeps the failed load's half-built model alive.
                import gc
                gc.collect()
                kwargs.pop("quantization_config", None)
                use_4bit = False
                model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs, **kwargs)
        _drop_reverse_fp8_conversions(model)
        # The load may import a modeling module for the first time (Unsloth's native DeepSeek-V4.1 is imported
        # lazily): its device-independent patches (checkpoint replay, config aliases) apply only once it exists.
        from . import runtime as _runtime
        _runtime.apply_cpu_security_patches()
        if auto_dtype:
            dtype = model.dtype
            if dtype == torch.float16 and not use_4bit:
                dtype = _resolve_dtype(torch.float16)
                model = model.to(dtype)
        if full_finetuning:
            _print_full_finetuning_precision(dtype, master_choice, master_fit)
        max_seq_length = _clamp_to_learned_positions(model, max_seq_length)
        tokenizer_kwargs = dict(
            token = token, revision = revision, trust_remote_code = trust_remote_code,
            model_max_length = max_seq_length, padding_side = "right",
            **{k: kwargs[k] for k in ("cache_dir", "local_files_only", "subfolder") if k in kwargs},
        )
        try:
            if adapter_name is not None and tokenizer_name == adapter_name:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **{**tokenizer_kwargs, **adapter_kwargs, "revision": adapter_revision})
            elif adapter_name is not None and tokenizer_name != model_name:
                # A separate tokenizer repo: the caller's revision, not the one the adapter recorded for its base.
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **{**tokenizer_kwargs, "revision": adapter_revision})
            else:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
        except (OSError, ValueError):
            if adapter_name is None or tokenizer_name != adapter_name:
                raise
            # An adapter saved without its tokenizer: the base model's.
            tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # for_inference / for_training flip its padding side, as the GPU path does.
        model._saved_temp_tokenizer = tokenizer
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        if frozen_fp32:
            for tensor in itertools.chain(model.parameters(), model.buffers()):
                if tensor.dtype == torch.bfloat16:
                    tensor.data = tensor.data.float()
        if frozen_bf16:
            _print(f"{store_frozen_linears_in_bf16(model)} frozen Linear weights kept in bfloat16, computing in float32.")
            # Generation upcasts them back to fp32 unless UNSLOTH_CPU_FROZEN_BF16=1 forced the storage: the per-call
            # upcast costs decode speed (about 3x per projection at one token), training pays nothing for it.
            model._unsloth_cpu_frozen_bf16 = "forced" if frozen_env in ("1", "true", "yes", "on") else "training"
        patch_causal_conv1d_fallback(model)
        # Also for a checkpoint that ships its own 4-bit quantization_config.
        _disable_bnb_cpu_weight_packing(model)
        model.max_seq_length = max_seq_length
        model._unsloth_cpu_gradient_checkpointing = use_gradient_checkpointing
        if full_finetuning:
            for param in model.parameters():
                param.requires_grad_(True)
            FastCPUModel._prepare(model, use_gradient_checkpointing)
        chunked = False if return_logits else patch_chunked_causal_lm_loss(model)
        _print(
            f"{type(model).__name__} on {torch.get_num_threads()} CPU threads. "
            f"dtype = {dtype}, 4-bit = {use_4bit}, chunked CE = {chunked}."
        )
        if resize_model_vocab is not None:
            model.resize_token_embeddings(resize_model_vocab)
        FastCPUModel._attach(model)
        if adapter_name is not None:
            from peft import PeftModel
            FastCPUModel._prepare(model, use_gradient_checkpointing)
            if use_gradient_checkpointing in (True, "unsloth") and hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            embeddings = model.get_input_embeddings()
            if embeddings is not None and len(tokenizer) > embeddings.num_embeddings:
                # Tokens added during finetuning: the saved embed_tokens / lm_head are that size.
                model.resize_token_embeddings(len(tokenizer))
            model = PeftModel.from_pretrained(
                model, adapter_name, is_trainable = True, token = token, revision = adapter_revision, **adapter_kwargs,
            )
            model.max_seq_length = max_seq_length
            FastCPUModel._attach(model)
        return model, tokenizer

    @staticmethod
    def _prepare(model, use_gradient_checkpointing):
        if use_gradient_checkpointing in (True, "unsloth") and not _non_reentrant_checkpointing_on(model):
            # "unsloth" offloads activations to host RAM on GPUs; on CPU they already live there,
            # so it is plain recomputation.
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
        # Training forwards never reuse a KV cache; generate() turns it back on via for_inference.
        model.config.use_cache = False
        wanted = _cpu_compile_wanted(model)
        if wanted == "auto":
            # Decided by the trainer once it has timed eager steps: a short run never repays the cold compile.
            model._unsloth_cpu_compile_pending = True
        elif wanted and not getattr(model, "_unsloth_cpu_compiled", False):
            compile_cpu_layers(model, explicit = True)

    @staticmethod
    def get_peft_model(
        model,
        r = 16,
        target_modules = None,
        lora_alpha = 16,
        lora_dropout = 0.0,
        bias = "none",
        layers_to_transform = None,
        layers_pattern = None,
        use_gradient_checkpointing = None,
        random_state = 3407,
        max_seq_length = None,
        use_rslora = False,
        modules_to_save = None,
        init_lora_weights = True,
        loftq_config = None,
        **kwargs,
    ):
        from peft import LoraConfig, PeftModel, get_peft_model

        if isinstance(model, PeftModel):
            # Reloaded LoRA checkpoint: a second get_peft_model would reset the saved adapter weights.
            _print("this model already has LoRA adapters, keeping them.")
            return model
        # A model built by the caller (not through from_pretrained) may have imported its modeling module since.
        from . import runtime as _runtime
        _runtime.apply_cpu_security_patches()
        if use_gradient_checkpointing is None:
            use_gradient_checkpointing = getattr(model, "_unsloth_cpu_gradient_checkpointing", "unsloth")
        # The trainer reads the choice back (SFTConfig would otherwise turn checkpointing on for False).
        model._unsloth_cpu_gradient_checkpointing = use_gradient_checkpointing
        leaf_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
        # Models with their own projection names (Unsloth's native DeepSeek-V4.1) translate q/k/v/o/gate/up/down_proj,
        # "all-linear" and None into their real Linears, exactly as the GPU get_peft_model does.
        remap = getattr(model, "unsloth_lora_target_modules", None)
        remapped_targets = False
        if callable(remap) and kwargs.get("finetune_language_layers", True):
            embeddings = []
            if target_modules is not None and not isinstance(target_modules, str):
                embeddings = [m for m in target_modules if m in ("embed_tokens", "lm_head")]
                target_modules = [m for m in target_modules if m not in embeddings]
            remapped = remap(
                target_modules,
                attention = kwargs.get("finetune_attention_modules", True),
                mlp = kwargs.get("finetune_mlp_modules", True),
            )
            if remapped is not target_modules:
                target_modules = remapped
                remapped_targets = True
            # Trained in full whatever the remapper returned (it may hand back the list it was given).
            if embeddings:
                modules_to_save = list(dict.fromkeys([*(modules_to_save or []), *embeddings]))
        if remapped_targets:
            pass
        elif target_modules is None:
            # Unsloth's usual projections where the model has them, else PEFT's per-architecture default.
            candidates = _with_fused_projections(_DEFAULT_TARGET_MODULES, leaf_names)
            target_modules = [m for m in candidates if m in leaf_names]
            # Only a set that reaches attention counts; a partial match (MPT: Wqkv + up/down_proj) would
            # leave attention frozen, so hand those architectures to PEFT's own per-model default.
            if not set(target_modules) & {"q_proj", "k_proj", "v_proj", "qkv_proj"}:
                target_modules = None
        elif not isinstance(target_modules, str):
            # As on GPU: embed_tokens / lm_head in target_modules are trained in full (continued pretraining).
            embeddings = [m for m in target_modules if m in ("embed_tokens", "lm_head")]
            if embeddings:
                target_modules = [m for m in target_modules if m not in embeddings]
                modules_to_save = list(dict.fromkeys([*(modules_to_save or []), *embeddings]))
            target_modules = _with_fused_projections(list(target_modules), leaf_names)
        for key in ("finetune_vision_layers", "finetune_language_layers", "finetune_attention_modules",
                    "finetune_mlp_modules", "temporary_location", "qat_scheme"):
            kwargs.pop(key, None)
        torch.manual_seed(random_state)
        import inspect
        ensure_weight_tying = kwargs.pop("ensure_weight_tying", None)
        if ensure_weight_tying is not None and "ensure_weight_tying" in inspect.signature(LoraConfig).parameters:
            kwargs["ensure_weight_tying"] = ensure_weight_tying
        config = LoraConfig(
            r = r,
            lora_alpha = lora_alpha,
            lora_dropout = lora_dropout,
            bias = bias,
            target_modules = target_modules,
            layers_to_transform = layers_to_transform,
            layers_pattern = layers_pattern,
            use_rslora = use_rslora,
            modules_to_save = modules_to_save,
            init_lora_weights = init_lora_weights,
            **({"loftq_config": loftq_config} if loftq_config is not None else {}),
            task_type = "CAUSAL_LM",
            **kwargs,
        )
        FastCPUModel._prepare(model, use_gradient_checkpointing)
        if use_gradient_checkpointing in (True, "unsloth") and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        peft_model = get_peft_model(model, config)
        _frozen_bf16_for(peft_model, training = True)  # after PEFT froze the base weights
        if max_seq_length is not None:
            peft_model.max_seq_length = max_seq_length
        FastCPUModel._attach(peft_model)
        return peft_model

    @staticmethod
    def for_training(model, use_gradient_checkpointing = None):
        model.train()
        _frozen_bf16_for(model, training = True)
        _set_inference_mode(model, False)
        # None keeps whatever from_pretrained / get_peft_model set up; an explicit value switches it.
        if use_gradient_checkpointing is not None and hasattr(model, "gradient_checkpointing_enable"):
            model._unsloth_cpu_gradient_checkpointing = use_gradient_checkpointing
            if use_gradient_checkpointing:
                if not _non_reentrant_checkpointing_on(model):
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
            elif _is_activation_checkpointing(model):
                model.gradient_checkpointing_disable()
        return model

    @staticmethod
    def for_inference(model):
        model.eval()
        _frozen_bf16_for(model, training = False)
        _set_inference_mode(model, True)
        return model

    @staticmethod
    def _attach(model):
        generate = getattr(model, "generate", None)
        func = getattr(generate, "__func__", None)
        if func is not None and generate.__self__ is model and not getattr(func, "_unsloth_cpu", False):
            # Bound per instance (MethodType), so deepcopy rebinds it to the copy.
            model.generate = types.MethodType(_unpack_encoding_generate(func), model)
        model.for_training = types.MethodType(lambda self, *a, **k: FastCPUModel.for_training(self, *a, **k), model)
        model.for_inference = types.MethodType(lambda self: FastCPUModel.for_inference(self), model)
        model.save_pretrained_merged = types.MethodType(save_pretrained_merged, model)
        model.push_to_hub_merged = types.MethodType(push_to_hub_merged, model)
        model.save_pretrained_gguf = types.MethodType(save_pretrained_gguf, model)


# Unsloth's projection names -> the fused module some architectures use instead (Phi-3: qkv_proj, gate_up_proj).
_FUSED_PROJECTIONS = {"q_proj": "qkv_proj", "k_proj": "qkv_proj", "v_proj": "qkv_proj",
                      "gate_proj": "gate_up_proj", "up_proj": "gate_up_proj"}


def _with_fused_projections(target_modules, leaf_names):
    """Add the fused projection a split name maps to when the model has it, so a Llama-style target list does not
    silently leave Phi-3's Q/K/V and MLP input projections frozen."""
    out = list(target_modules)
    for name in target_modules:
        fused = _FUSED_PROJECTIONS.get(name)
        if fused and fused in leaf_names and name not in leaf_names and fused not in out:
            out.append(fused)
    return out


def _adapter_base_model(model_name, token, revision, **hub_kwargs):
    """(base model, base revision) when model_name is a PEFT adapter checkpoint (local dir or Hub repo), else None."""
    try:
        local = os.path.join(model_name, hub_kwargs.get("subfolder") or "")
        if os.path.isdir(model_name) and not os.path.isfile(os.path.join(local, "adapter_config.json")):
            return None
        from peft import PeftConfig
        # Cache-aware (works offline for a cached adapter); a plain model repo raises here.
        config = PeftConfig.from_pretrained(model_name, revision = revision, token = token, **hub_kwargs)
        if not config.base_model_name_or_path:
            return None
        base = config.base_model_name_or_path, getattr(config, "revision", None)
    except Exception:
        return None
    # A Hub adapter names its base itself: on Windows a UNC base would make the load probe a host the repo chose (SMB, NTLM).
    if not os.path.isdir(model_name):
        from unsloth_zoo.vision_utils import _is_network_share
        if _is_network_share(str(base[0])):
            raise ValueError(
                f"Unsloth: The adapter {model_name} names a base model on a network share ({base[0]!r}). "
                "Refusing to load it: download the base model and load the adapter from a local folder instead."
            )
    return base


def _clamp_to_learned_positions(model, max_seq_length):
    """GPT-2 / OPT learned position tables cannot be extended, so a longer max_seq_length would index past them; RoPE models can run past their trained length."""
    config = getattr(model, "config", None)
    limit = getattr(config, "max_position_embeddings", None) or getattr(config, "n_positions", None)
    if not isinstance(limit, int) or max_seq_length is None or max_seq_length <= limit:
        return max_seq_length
    inputs = model.get_input_embeddings()
    learned = any(
        isinstance(m, torch.nn.Embedding) and m is not inputs and limit <= m.num_embeddings <= limit + 2
        for m in model.modules()
    )
    if learned:
        _print(f"max_seq_length = {max_seq_length} is longer than this model's {limit} learned positions, using {limit}.")
        return limit
    return max_seq_length


def _set_inference_mode(model, inference):
    """KV cache on and left padding for generate(), off and right padding for training, as on GPU."""
    inner = model
    for _ in range(3):
        if hasattr(inner, "config"):
            inner.config.use_cache = inference
        tokenizer = getattr(inner, "_saved_temp_tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left" if inference else "right"
        inner = getattr(inner, "base_model", None) or getattr(inner, "model", None)
        if inner is None:
            break


def _frozen_bf16_mode(model):
    inner, mode = model, None
    for _ in range(4):
        mode = mode or getattr(inner, "_unsloth_cpu_frozen_bf16", None)
        inner = getattr(inner, "base_model", None) or getattr(inner, "model", None)
        if inner is None:
            break
    return mode


# A rollout's fp32 copy of the frozen weights may use at most this share of the RAM still available.
_FP32_ROLLOUT_RAM_SHARE = 0.5


def _slow_bf16_upcast():
    """Apple Silicon: upcasting the bf16-stored projections on every decoded token made a Qwen3-0.6B GRPO step 1.6x
    slower (M1). On x86 without native bf16 (AVX2) it cost about 2%, not worth the fp32 copy's RAM."""
    import platform
    return sys.platform == "darwin" and platform.machine().lower() in ("arm64", "aarch64")


def _fp32_rollout_fits(model):
    """True on Apple Silicon when the frozen bf16 projections' fp32 copy (2 more bytes per weight) fits in RAM."""
    if not _slow_bf16_upcast():
        return False
    extra = 2 * sum(m.weight.numel() for m in model.modules()
                    if isinstance(m, Bf16StoredLinear) and m.weight.dtype == torch.bfloat16 and not m.weight.requires_grad)
    if extra == 0:
        return False
    available = available_ram()
    return available is not None and extra <= _FP32_ROLLOUT_RAM_SHARE * available


def _frozen_bf16_for(model, training):
    """Default frozen bf16 storage is for training only: generation upcasts the weights to fp32 (decode speed), and
    training stores them in bf16 again (half their RAM). UNSLOTH_CPU_FROZEN_BF16=1 keeps bf16 throughout."""
    if _frozen_bf16_mode(model) == "training" and isinstance(model, torch.nn.Module):
        set_bf16_stored_dtype(model, torch.bfloat16 if training else torch.float32)


def _unpack_encoding_generate(generate):
    """Notebooks pass `input_ids = tokenizer.apply_chat_template(..., return_tensors = "pt")`, which is a
    BatchEncoding on transformers 5; generate() then fails on `.shape`. The GPU path unpacks it, so do the same."""
    @functools.wraps(generate)
    def wrapper(self, *args, **kwargs):
        encoding = kwargs.get("input_ids", None)
        if encoding is not None and not isinstance(encoding, torch.Tensor) and hasattr(encoding, "items"):
            for key, value in kwargs.pop("input_ids").items():
                kwargs.setdefault(key, value)
        restore = False
        if isinstance(self, torch.nn.Module) and not self.training:
            # Inference-only loads and for_inference: fp32 frozen weights (a training-time callback keeps bf16).
            _frozen_bf16_for(self, training = False)
        elif isinstance(self, torch.nn.Module) and _frozen_bf16_mode(self) == "training" and _fp32_rollout_fits(self):
            # Generating mid-training (GRPO / online DPO rollouts, train mode): decode with fp32 frozen weights too, as
            # upcasting every projection per token made a CPU GRPO step 1.6x slower on Apple Silicon, then store bf16.
            # inference_mode(False): converted under a caller's inference_mode, the weights would stay inference
            # tensors, and the next training forward could not save them for backward.
            with torch.inference_mode(False):
                restore = set_bf16_stored_dtype(self, torch.float32) > 0
        try:
            set_stance = getattr(torch.compiler, "set_stance", None)
            modules = self.modules() if isinstance(self, torch.nn.Module) else ()
            if set_stance is not None and any(getattr(m, "_compiled_call_impl", None) is not None for m in modules):
                # Layers compiled for training would recompile per layer against the KV cache, up to the limit.
                with set_stance("force_eager"):
                    return generate(self, *args, **kwargs)
            return generate(self, *args, **kwargs)
        finally:
            if restore:
                with torch.inference_mode(False):
                    set_bf16_stored_dtype(self, torch.bfloat16)
    wrapper._unsloth_cpu = True
    return wrapper


def _merged_copy(model, save_method):
    if save_method == "lora":
        return model, True
    if not hasattr(model, "merge_and_unload"):
        # Full finetuning: nothing to merge, but merged_16bit still means 16-bit weights on disk.
        if save_method == "merged_16bit" and getattr(model, "dtype", None) == torch.float32:
            import copy
            model = copy.deepcopy(model).to(torch.bfloat16 if cpu_bf16_supported() else torch.float16)
        return model, False
    if getattr(getattr(model, "config", None), "quantization_config", None) is not None or any(
        type(m).__name__ == "Linear4bit" for m in model.modules()
    ):
        raise NotImplementedError(
            "Unsloth (CPU): merging LoRA into a 4-bit model is not supported on CPU yet. "
            "Load with load_in_4bit = False to save merged 16-bit weights, or save_method = 'lora'."
        )
    import copy
    merged = copy.deepcopy(model)
    for module in merged.modules():
        # bf16-stored frozen weights: merge in fp32 so the merged / GGUF weights are unrounded.
        if isinstance(module, Bf16StoredLinear) and module.weight.dtype == torch.bfloat16:
            module.weight.data = module.weight.data.float()
    merged = merged.merge_and_unload()
    if save_method == "merged_16bit" and merged.dtype == torch.float32:
        merged = merged.to(torch.bfloat16 if cpu_bf16_supported() else torch.float16)
    return merged, False


def _stream_merge_source(model):
    """The transformers model whose live weights a streamed merge reads, or None when the merge needs the full copy.
    Allowlist: plain LoRA on Linear layers (no DoRA, lora_bias, already merged adapters), no other PEFT tuner
    (IA3, trainable token adapters, modules_to_save, ...), no 4-bit, no remote code (save_pretrained also copies its
    modeling files)."""
    try:
        from peft import PeftModel
        from peft.tuners.lora import LoraLayer
        from peft.tuners.lora.layer import Linear as LoraLinear
        from peft.tuners.tuners_utils import BaseTunerLayer
    except Exception:
        return None
    if os.environ.get("UNSLOTH_CPU_STREAM_MERGE", "1") == "0":
        return None
    if isinstance(model, PeftModel):
        configs = getattr(model, "peft_config", None) or {}
        if any(str(getattr(config, "peft_type", "")).rsplit(".", 1)[-1] != "LORA" for config in configs.values()):
            return None
    hf = model.get_base_model() if isinstance(model, PeftModel) else model
    if not hasattr(hf, "save_pretrained") or getattr(hf, "_auto_class", None) is not None:
        return None
    if getattr(getattr(hf, "config", None), "quantization_config", None) is not None:
        return None
    if getattr(hf, "_checkpoint_conversion_mapping", None):
        return None  # transformers 4.x VLMs: save_pretrained renames keys back to the checkpoint layout
    for module in hf.modules():
        name = type(module).__name__
        if name in ("ModulesToSaveWrapper", "TrainableTokensWrapper", "Linear4bit", "Linear8bitLt") or \
            "AuxiliaryTrainingWrapper" in (cls.__name__ for cls in type(module).__mro__):
            return None
        if isinstance(module, BaseTunerLayer) and not isinstance(module, LoraLayer):
            return None  # IA3, trainable token adapters, other tuners: only merge_and_unload knows their merge
        if isinstance(module, LoraLayer):
            if type(module) is not LoraLinear or getattr(module, "merged_adapters", None):
                return None
            if any(getattr(module, "lora_bias", {}).get(name, False) for name in module.active_adapters):
                return None
            if any(getattr(module, "use_dora", {}).get(name, False) for name in module.active_adapters):
                return None
    return hf


def _save_keeps_live_layout(hf, entries):
    """True when save_pretrained would write the live names / shapes. transformers 5 reverts its checkpoint conversions
    on save (e.g. Qwen3MoE fused experts back to per-expert weights): dry-run that revert on meta tensors and stream
    only when it changes nothing."""
    try:
        from transformers.core_model_loading import revert_weight_conversion
    except Exception:
        return True  # transformers 4.x: no save-time conversion beyond _checkpoint_conversion_mapping (checked above)
    meta = {out: torch.empty(shape, dtype = dtype, device = "meta") for out, _, dtype, _, shape in entries}
    try:
        reverted = revert_weight_conversion(hf, dict(meta))
    except Exception:
        return False
    return set(reverted) == set(meta) and all(tuple(reverted[k].shape) == tuple(meta[k].shape) for k in meta)


def _merged_weight(module, to_float):
    """PEFT's own merge on a scratch copy of the base weight: bit-identical to merge_and_unload, live model unchanged."""
    base = module.get_base_layer()
    live = base.weight.data
    scratch = live.float() if to_float else live.clone()
    base.weight.data = scratch
    try:
        module.merge()
    finally:
        base.weight.data = live
        module.merged_adapters.clear()
    return scratch


def _shard_bytes(size):
    """save_pretrained's max_shard_size: an int (bytes) or "5GB" / "500MB" / "2GiB" (decimal units, *iB binary)."""
    if isinstance(size, int):
        return size
    match = re.fullmatch(r"\s*([0-9.]+)\s*([KMGT]?)(I?)B?\s*", str(size).upper())
    if not match:
        raise ValueError(f"Unsloth (CPU): max_shard_size = {size!r} is not a size like '5GB'.")
    value, unit, binary = match.groups()
    base = 1024 if binary else 1000
    return int(float(value) * base ** "_KMGT".index(unit or "_"))


_SAFETENSORS_DTYPES = {
    torch.float64: "F64", torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16", torch.int64: "I64",
    torch.int32: "I32", torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL",
}


def _safetensors_dtype_rank():
    """safetensors' save_file lays tensors out by dtype (its own order, widest first) then name: read that order from
    the installed safetensors so the streamed file is byte-identical to save_file's."""
    import json
    import struct
    try:
        from safetensors.torch import save
        blob = save({str(i): torch.zeros(1, dtype = dtype) for i, dtype in enumerate(_SAFETENSORS_DTYPES)})
        order = list(json.loads(blob[8:8 + struct.unpack("<Q", blob[:8])[0]]))
        dtypes = list(_SAFETENSORS_DTYPES)
        return {dtypes[int(k)]: rank for rank, k in enumerate(k for k in order if k != "__metadata__")}
    except Exception:
        return {dtype: rank for rank, dtype in enumerate((torch.int64, torch.float64, torch.float32, torch.int32,
            torch.bfloat16, torch.float16, torch.int16, torch.int8, torch.uint8, torch.bool))}


def _write_safetensors(path, entries, rank = None):
    """A safetensors file written one tensor at a time: the header comes from shapes / dtypes, then each tensor is
    made, written and dropped, so at most one merged tensor is alive (safetensors' save_file holds the whole file as
    bytes). Same layout as save_file (metadata first, dtype then name order). Returns the payload size in bytes."""
    import json
    import struct
    rank = rank or _safetensors_dtype_rank()
    entries = sorted(entries, key = lambda entry: (rank.get(entry[2], len(rank)), entry[0]))
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for out, _, dtype, numel, shape in entries:
        if dtype not in _SAFETENSORS_DTYPES:
            raise TypeError(f"Unsloth (CPU): no safetensors dtype for {dtype}")
        nbytes = numel * torch.empty((), dtype = dtype).element_size()
        header[out] = {"dtype": _SAFETENSORS_DTYPES[dtype], "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    encoded = json.dumps(header, separators = (",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for out, make, dtype, numel, shape in entries:
            value = make().to(dtype).contiguous()
            if tuple(value.shape) != shape:
                raise RuntimeError(f"Unsloth (CPU): merged {out} has shape {tuple(value.shape)}, expected {shape}")
            if value.numel():
                f.write(memoryview(value.view(-1).view(torch.uint8).numpy()))
            del value
    return offset


def _default_shard_size(hf):
    """save_pretrained's own default (5GB on transformers 4.x, 50GB on 5.x)."""
    import inspect
    try:
        default = inspect.signature(type(hf).save_pretrained).parameters["max_shard_size"].default
        return default if isinstance(default, (int, str)) else "5GB"
    except Exception:
        return "5GB"


def _split_shards(entries, limit):
    """Shard grouping and names from huggingface_hub's splitter (what save_pretrained uses), on meta tensors."""
    try:
        from huggingface_hub import split_torch_state_dict_into_shards
        meta = {entry[0]: torch.empty(entry[4], dtype = entry[2], device = "meta") for entry in entries}
        split = split_torch_state_dict_into_shards(meta, filename_pattern = "model{suffix}.safetensors",
                                                   max_shard_size = limit)
        by_name = {entry[0]: entry for entry in entries}
        return [(name, [by_name[k] for k in keys]) for name, keys in split.filename_to_tensors.items()], split.metadata
    except Exception:
        pass
    shards, current, size = [], [], 0
    for entry in entries:
        nbytes = entry[3] * torch.empty((), dtype = entry[2]).element_size()
        if current and size + nbytes > limit:
            shards.append(current)
            current, size = [], 0
        current.append(entry)
        size += nbytes
    if current:
        shards.append(current)
    names = ["model.safetensors"] if len(shards) == 1 else \
        [f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors" for i in range(len(shards))]
    total = sum(e[3] * torch.empty((), dtype = e[2]).element_size() for e in entries)
    return list(zip(names, shards)), {"total_size": total}


_STALE_SHARD = re.compile(r"model-\d{5}-of-\d{5}\.safetensors")


def _remove_stale_weights(save_directory, names):
    """Re-exporting into the same folder: drop an earlier export's weights that are not part of this one, as
    save_pretrained does, so a different shard count leaves no obsolete shards (or a single file / index that
    transformers would load instead). Only model.safetensors, model-NNNNN-of-NNNNN.safetensors and the index; other
    files (adapters, tokenizer, other names) are never touched. Rank 0 only."""
    if os.environ.get("RANK", "0").strip() not in ("", "0"):
        return
    keep = set(names)
    if len(names) > 1:
        keep.add("model.safetensors.index.json")
    for filename in os.listdir(save_directory):
        if filename in keep:
            continue
        if filename in ("model.safetensors", "model.safetensors.index.json") or _STALE_SHARD.fullmatch(filename):
            path = os.path.join(save_directory, filename)
            if os.path.isfile(path):
                os.remove(path)


def _stream_merged_save(model, save_directory, save_method, max_shard_size = None):
    """Writes the merged checkpoint shard by shard from the live model (peak RSS about the model plus one shard,
    instead of a full deepcopy). Same tensors, names, dtypes and config as merge_and_unload().save_pretrained().
    Returns False when the model needs the copying path."""
    hf = _stream_merge_source(model)
    if hf is None:
        return False
    import copy
    import json
    from peft.tuners.lora import LoraLayer

    lora = {name: module for name, module in hf.named_modules() if isinstance(module, LoraLayer)}
    bf16_stored = {name.replace(".base_layer", "") for name, module in hf.named_modules() if isinstance(module, Bf16StoredLinear)}
    model_dtype = hf.dtype
    target = (torch.bfloat16 if cpu_bf16_supported() else torch.float16) \
        if save_method == "merged_16bit" and model_dtype == torch.float32 else None
    # bf16-stored frozen weights are upcast to fp32 first, as the copying path does (unrounded merge / GGUF staging).
    shared = {}
    for name, module in hf.named_parameters(remove_duplicate = False):
        shared.setdefault(module.untyped_storage().data_ptr(), []).append(name)
    for name, module in lora.items():
        weight = module.get_base_layer().weight
        if len(shared.get(weight.untyped_storage().data_ptr(), ())) > 1:
            return False  # a LoRA on a tied weight: merge_and_unload edits the shared tensor, keep its semantics

    entries, seen = [], set()
    for key, tensor in hf.state_dict(keep_vars = True).items():
        if ".lora_" in key:
            continue
        out = key.replace(".base_layer.", ".")
        storage = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape))
        if storage in seen:
            continue  # tied (lm_head = embed_tokens): save_pretrained keeps the first name only
        seen.add(storage)
        owner = out.rsplit(".", 1)[0]
        stored = owner in bf16_stored and out.endswith(".weight") and tensor.dtype == torch.bfloat16
        if owner in lora and key.endswith(".base_layer.weight"):
            make = functools.partial(_merged_weight, lora[owner], stored)
            dtype = torch.float32 if stored else tensor.dtype
        elif stored:
            make, dtype = (lambda t = tensor: t.detach().float()), torch.float32
        else:
            make, dtype = (lambda t = tensor: t.detach()), tensor.dtype
        if target is not None and dtype.is_floating_point:
            dtype = target
        entries.append((out, make, dtype, tensor.numel(), tuple(tensor.shape)))
    if not _save_keeps_live_layout(hf, entries):
        return False

    limit = _shard_bytes(max_shard_size or _default_shard_size(hf))
    shards, split_metadata = _split_shards(entries, limit)
    names = [name for name, _ in shards]
    os.makedirs(save_directory, exist_ok = True)
    # Each file is written under a temporary name and moved into place only after every shard is complete: the live
    # weights may be memory-mapped from the files being replaced (a model loaded from save_directory), and an
    # interrupted export must not leave a half-written model.safetensors that loaders would pick.
    rank, staged = _safetensors_dtype_rank(), []
    # A different shard count writes new names: those take the mode of an earlier export's weights in this folder.
    earlier = [os.path.join(save_directory, f) for f in sorted(os.listdir(save_directory))
               if f == "model.safetensors" or _STALE_SHARD.fullmatch(f)]
    try:
        with torch.no_grad():
            for name, shard in shards:
                temporary = os.path.join(save_directory, f".{name}.unsloth-tmp")
                staged.append((temporary, os.path.join(save_directory, name)))
                # A new shard name (shard count changed) is kept private while written like the earlier export's.
                precreate_private_temp(temporary, next((p for p in (staged[-1][1], *earlier) if os.path.isfile(p)), staged[-1][1]))
                _write_safetensors(temporary, shard, rank)
        if len(shards) > 1:
            total_parameters = sum(p.numel() for n, p in hf.named_parameters() if ".lora_" not in n)
            index = {"metadata": {"total_parameters": total_parameters, **split_metadata},
                     "weight_map": {entry[0]: name for name, shard in shards for entry in shard}}
            temporary = os.path.join(save_directory, ".model.safetensors.index.json.unsloth-tmp")
            staged.append((temporary, os.path.join(save_directory, "model.safetensors.index.json")))
            precreate_private_temp(temporary, staged[-1][1])
            with open(temporary, "w", encoding = "utf-8") as f:
                f.write(json.dumps(index, indent = 2, sort_keys = True) + "\n")
        for temporary, final in staged:
            keep_mode(temporary, final, like = earlier)
            os.replace(temporary, final)
    finally:
        for temporary, _ in staged:
            if os.path.exists(temporary):
                os.remove(temporary)
    _remove_stale_weights(save_directory, names)
    config = copy.deepcopy(hf.config)
    # save_pretrained records the first floating parameter's dtype under the field this transformers reads.
    setattr(config, _DTYPE_KWARG, str(target or model_dtype).split(".")[1])
    config.architectures = [type(hf).__name__]
    config.save_pretrained(save_directory)
    generation = getattr(hf, "generation_config", None)
    if generation is not None and getattr(hf, "can_generate", lambda: False)():
        generation.save_pretrained(save_directory)
    return True


def _save_merged(model, save_directory, save_method, **kwargs):
    """Streamed merge when the model allows it, else the full merged copy."""
    if kwargs.get("safe_serialization", True) and _stream_merged_save(
        model, save_directory, save_method, kwargs.get("max_shard_size"),
    ):
        return
    merged, _ = _merged_copy(model, save_method)
    merged.save_pretrained(save_directory, **{k: v for k, v in kwargs.items() if k in ("safe_serialization", "max_shard_size")})
    del merged


def _check_save_method(save_method):
    if save_method not in ("lora", "merged_16bit"):
        raise NotImplementedError(f"Unsloth (CPU): save_method = {save_method!r} is not supported on CPU.")


def save_pretrained_merged(self, save_directory, tokenizer = None, save_method = "merged_16bit", **kwargs):
    """CPU save_pretrained_merged: "lora" (adapters) or "merged_16bit"."""
    _check_save_method(save_method)
    for key in ("maximum_memory_usage", "temporary_location", "max_shard_size_gb"):
        kwargs.pop(key, None)
    if save_method == "lora":
        self.save_pretrained(save_directory, **{k: v for k, v in kwargs.items() if k in ("safe_serialization", "max_shard_size")})
    else:
        _save_merged(self, save_directory, save_method, **kwargs)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_directory)
    return save_directory


def _refuse_public_repo_for_private_push(repo_id, token, private):
    # transformers' push_to_hub calls create_repo(private=True, exist_ok=True), which keeps an existing public repo
    # public: refuse before the first upload, as saving_utils.create_huggingface_repo does on the GPU path.
    if not private: return
    from huggingface_hub import HfApi
    try:
        api = HfApi(token = token)
        url = api.create_repo(repo_id = repo_id, private = True, exist_ok = True)
        is_private = getattr(api.repo_info(repo_id = getattr(url, "repo_id", None) or repo_id, repo_type = "model"), "private", None)
    except Exception as exc:
        print(f"Unsloth: Could not confirm the visibility of {repo_id} ({exc}); continuing.")
        return
    if is_private is False:
        raise RuntimeError(
            f"Unsloth: private=True was requested but the Hugging Face repo {repo_id!r} already exists and is "
            "public. Refusing to upload so private weights are not published. Make the repo private on the Hub, "
            "push to a new repo id, or pass private=False."
        )


def push_to_hub_merged(self, repo_id, tokenizer = None, save_method = "merged_16bit", token = None, private = None, **kwargs):
    _check_save_method(save_method)
    _refuse_public_repo_for_private_push(repo_id, token, private)
    model, _ = _merged_copy(self, save_method)
    # Branch / PR / message options shared by model and tokenizer push_to_hub on transformers 4.57 and 5.x.
    hub = {k: kwargs[k] for k in ("revision", "create_pr", "commit_message", "commit_description", "tags") if k in kwargs}
    info = model.push_to_hub(repo_id, token = token, private = private, **hub)
    if tokenizer is not None:
        pr_revision = getattr(info, "pr_revision", None)
        if hub.get("create_pr") and pr_revision:
            # Add the tokenizer to the model's PR instead of opening a second, model-less one.
            hub = {**hub, "create_pr": False, "revision": pr_revision}
        tokenizer.push_to_hub(repo_id, token = token, private = private, **hub)
    return repo_id


# Shared with the MLX GGUF export (one guard for both converter paths).
from unsloth_zoo.llama_cpp import loaded_remote_code as _loaded_remote_code  # noqa: E402
from unsloth_zoo.llama_cpp import converter_remote_code_guard as _converter_remote_code  # noqa: E402


def save_pretrained_gguf(self, save_directory, tokenizer = None, quantization_method = "q8_0", **kwargs):
    """Merge to 16-bit, then llama.cpp's converter (+ llama-quantize for k-quants). Returns the GGUF paths."""
    from unsloth_zoo.llama_cpp import install_llama_cpp, convert_to_gguf, quantize_gguf

    methods = [quantization_method] if isinstance(quantization_method, str) else list(quantization_method)
    # Unsloth's GGUF aliases, as on the GPU save path.
    aliases = {"not_quantized": "bf16", "fast_quantized": "q8_0", "quantized": "q4_k_m", "none": "q8_0"}
    methods = list(dict.fromkeys(aliases.get(str(m).lower(), str(m).lower()) for m in methods))
    from unsloth_zoo.llama_cpp import quant_requires_imatrix
    needs_imatrix = [m for m in methods if quant_requires_imatrix(m)]
    if needs_imatrix:
        # Fail before the merge and conversion, not after them.
        raise NotImplementedError(
            f"Unsloth (CPU): {needs_imatrix} need an importance matrix, which the CPU GGUF export does not build. "
            "Use q4_k_m / q5_k_m / q8_0 / bf16, or quantize the bf16 GGUF with llama-quantize --imatrix."
        )
    os.makedirs(save_directory, exist_ok = True)
    merged_directory = os.path.join(save_directory, "merged_16bit")
    # Stage merged weights unrounded: the converter casts once per requested type instead of rounding through fp16/bf16
    # first; fp16 without native bf16 would also clip bf16's range.
    unrounded = [m for m in methods if m in ("f32", "f16")]
    _save_merged(self, merged_directory, "merged_32bit")
    if tokenizer is not None:
        tokenizer.save_pretrained(merged_directory)
    quantizer, converter = install_llama_cpp(print_output = False)
    base_name = os.path.basename(os.path.normpath(save_directory)) or "model"
    direct = {"bf16", "f16", "q8_0", "f32"}
    # llama-quantize refuses to requantize from q8_0, so k-quants are made from a bf16 GGUF.
    first = "bf16" if len(methods) > 1 or any(m not in direct for m in methods) else methods[0]
    approved = getattr(self, "_unsloth_trust_remote_code", False) is True or _loaded_remote_code(self, tokenizer)
    with _converter_remote_code(approved):
        outputs, _ = convert_to_gguf(
            os.path.join(save_directory, base_name), merged_directory, model_dtype = "bf16",
            quantization_type = first, converter_location = converter,
        )
    first_file = outputs[0]
    results = list(outputs)
    for method in methods:
        if method in direct and method == first:
            continue
        if method in unrounded:
            # Straight from the unrounded staging checkpoint; llama-quantize would re-cast the bf16 GGUF.
            with _converter_remote_code(approved):
                direct_outputs, _ = convert_to_gguf(
                    os.path.join(save_directory, base_name), merged_directory, model_dtype = "bf16",
                    quantization_type = method, converter_location = converter,
                )
            results.extend(direct_outputs)
            continue
        target = first_file.replace(f".{first.upper()}.gguf", f".{method.upper()}.gguf")
        if target == first_file:
            target = os.path.join(save_directory, f"{base_name}.{method.upper()}.gguf")
        quantize_gguf(first_file, target, method, quantizer_location = quantizer, print_output = False)
        results.append(target)
    if first not in methods:
        # The bf16 GGUF only fed llama-quantize; drop it (all shards) like the GPU export does.
        for path in outputs:
            if os.path.exists(path):
                os.remove(path)
            if path in results:
                results.remove(path)
    return results

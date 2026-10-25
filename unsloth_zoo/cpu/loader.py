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

import functools
import os
import re
import types
import warnings

import torch

from .runtime import cpu_bf16_supported, cpu_bnb_4bit_supported
from .ops import Bf16StoredLinear, patch_causal_conv1d_fallback, patch_chunked_causal_lm_loss, store_frozen_linears_in_bf16

_DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# unsloth/Qwen3-0.6B-unsloth-bnb-4bit -> unsloth/Qwen3-0.6B
_PREQUANTIZED_SUFFIX = re.compile(r"(-unsloth)?-bnb-4bit$", re.IGNORECASE)


def _dtype_kwarg():
    # from_pretrained took `torch_dtype` before transformers 4.56; older ones drop `dtype` silently.
    from transformers import __version__
    from unsloth_zoo.utils import Version
    return "dtype" if Version(__version__) >= Version("4.56.0") else "torch_dtype"


_DTYPE_KWARG = _dtype_kwarg()


def _print(message):
    print(f"Unsloth (CPU): {message}")


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
    return len(layers)


@functools.cache
def _cpu_compile_unavailable():
    """None when Inductor can build and run a CPU kernel here, else why not. Inductor needs a C++
    compiler at run time (MSVC `cl` on an initialised Windows shell, Xcode clang on macOS, gcc / clang on
    Linux); without one the first training step would raise instead of falling back."""
    try:
        import torch._dynamo
        x = torch.randn(64)
        out = torch.compile(lambda t: torch.nn.functional.silu(t) * 2, dynamic = False)(x)
        if not torch.allclose(out, torch.nn.functional.silu(x) * 2):
            return "probe kernel gave wrong results"
        return None
    except Exception as error:
        message = str(error).strip().splitlines()
        return f"{type(error).__name__}: {message[0][:200] if message else ''}"
    finally:
        try:
            torch._dynamo.reset()
        except Exception:
            pass


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
        # "auto" keeps the checkpoint's own dtype (fp16 checkpoints still move off fp16 after loading).
        auto_dtype = isinstance(dtype, str) and dtype == "auto"
        dtype = _resolve_dtype(None if auto_dtype else dtype)

        # 4-bit only where bitsandbytes' CPU backend really runs; full finetuning never quantizes.
        use_4bit = bool(load_in_4bit) and not load_in_8bit and not load_in_16bit and not full_finetuning
        if use_4bit and not cpu_bnb_4bit_supported():
            _print("bitsandbytes 4-bit does not work on this CPU, loading 16-bit weights instead.")
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
        if not use_4bit and _PREQUANTIZED_SUFFIX.search(model_name):
            base = _PREQUANTIZED_SUFFIX.sub("", model_name)
            _print(f"{model_name} is pre-quantized; loading {base} for 16-bit training.")
            model_name = base

        quantization_config = None
        if use_4bit and not _PREQUANTIZED_SUFFIX.search(model_name):
            from ..peft_utils import SKIP_QUANTIZATION_MODULES
            quantization_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_quant_type = "nf4",
                bnb_4bit_use_double_quant = True,
                bnb_4bit_compute_dtype = dtype,
                # As the GPU loaders: lm_head, MoE routers, Mamba and vision / audio towers stay 16-bit.
                llm_int8_skip_modules = list(SKIP_QUANTIZATION_MODULES),
            )
        resize_model_vocab = kwargs.pop("resize_model_vocab", None)
        for key in ("load_in_4bit", "load_in_8bit", "quantization_config", "use_exact_model_name",
                    "max_lora_rank", "gpu_memory_utilization", "float8_kv_cache", "random_state",
                    "fix_tokenizer", "unsloth_tiled_mlp", "offload_embedding"):
            kwargs.pop(key, None)

        if quantization_config is not None:
            # Only when set: an explicit None also discards a pre-quantized checkpoint's own config.
            kwargs["quantization_config"] = quantization_config
        # Left to transformers when unset: SDPA where the class supports it, eager otherwise (Bloom, MPT, ...).
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        # Opt-in for fp32 training: a bf16 checkpoint keeps its frozen Linear weights in bf16 (lossless,
        # about a quarter less resident memory) while everything trainable and every matmul stays fp32.
        frozen_bf16 = (
            os.environ.get("UNSLOTH_CPU_FROZEN_BF16", "0") == "1" and dtype == torch.float32 and not auto_dtype
            and not use_4bit and not full_finetuning
            and _checkpoint_dtype(model_name, token = token, revision = revision,
                                  trust_remote_code = trust_remote_code, **hub_kwargs) == torch.bfloat16
        )
        load_kwargs = dict(
            **{_DTYPE_KWARG: "auto" if auto_dtype else (torch.bfloat16 if frozen_bf16 else dtype)},
            device_map = "cpu",
            token = token,
            revision = revision,
            trust_remote_code = trust_remote_code,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs, **kwargs)
            except RuntimeError as error:
                # bitsandbytes' CPU packing rejects very narrow layers (hidden size < 64).
                if quantization_config is None:
                    raise
                _print(f"bitsandbytes 4-bit failed on CPU ({error}), loading 16-bit weights instead.")
                kwargs.pop("quantization_config", None)
                use_4bit = False
                model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs, **kwargs)
        if auto_dtype:
            dtype = model.dtype
            if dtype == torch.float16 and not use_4bit:
                dtype = _resolve_dtype(torch.float16)
                model = model.to(dtype)
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

        if frozen_bf16:
            _print(f"{store_frozen_linears_in_bf16(model)} frozen Linear weights kept in bfloat16, computing in float32.")
        patch_causal_conv1d_fallback(model)
        # Also for a checkpoint that ships its own 4-bit quantization_config.
        _disable_bnb_cpu_weight_packing(model)
        model.max_seq_length = max_seq_length
        model._unsloth_cpu_gradient_checkpointing = use_gradient_checkpointing
        if full_finetuning:
            for param in model.parameters():
                param.requires_grad_(True)
            FastCPUModel._prepare(model, use_gradient_checkpointing)
        chunked = patch_chunked_causal_lm_loss(model)
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
        if use_gradient_checkpointing in (True, "unsloth"):
            # "unsloth" offloads activations to host RAM on GPUs; on CPU they already live there,
            # so it is plain recomputation.
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
        # Training forwards never reuse a KV cache; generate() turns it back on via for_inference.
        model.config.use_cache = False
        if os.environ.get("UNSLOTH_CPU_COMPILE", "0") == "1" and not getattr(model, "_unsloth_cpu_compiled", False):
            reason = _cpu_compile_unavailable()
            if reason is not None:
                _print(f"UNSLOTH_CPU_COMPILE=1 ignored, training eagerly: torch.compile cannot build CPU kernels here ({reason}).")
                return
            count = _compile_layers(model)
            model._unsloth_cpu_compiled = True
            _print(f"torch.compile (Inductor C++) on {count} decoder layers.")

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
        if use_gradient_checkpointing is None:
            use_gradient_checkpointing = getattr(model, "_unsloth_cpu_gradient_checkpointing", "unsloth")
        leaf_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
        if target_modules is None:
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
        if max_seq_length is not None:
            peft_model.max_seq_length = max_seq_length
        FastCPUModel._attach(peft_model)
        return peft_model

    @staticmethod
    def for_training(model, use_gradient_checkpointing = None):
        model.train()
        _set_inference_mode(model, False)
        # None keeps whatever from_pretrained / get_peft_model set up; an explicit value switches it.
        if use_gradient_checkpointing is not None and hasattr(model, "gradient_checkpointing_enable"):
            if use_gradient_checkpointing:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
            elif getattr(model, "is_gradient_checkpointing", False):
                model.gradient_checkpointing_disable()
        return model

    @staticmethod
    def for_inference(model):
        model.eval()
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
    """Add the fused projection a requested split name maps to, when the model has it and not the split one,
    so a Llama-style target list does not silently leave Phi-3's Q/K/V and MLP input projections frozen."""
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
        return config.base_model_name_or_path, getattr(config, "revision", None)
    except Exception:
        return None


def _clamp_to_learned_positions(model, max_seq_length):
    """GPT-2 / OPT style learned position tables cannot be extended here (RoPE models can run past their
    trained length), so a longer max_seq_length would index past the table during training."""
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


def _unpack_encoding_generate(generate):
    """Notebooks pass `input_ids = tokenizer.apply_chat_template(..., return_tensors = "pt")`, which is a
    BatchEncoding on transformers 5; generate() then fails on `.shape`. The GPU path unpacks it, so do the same."""
    @functools.wraps(generate)
    def wrapper(self, *args, **kwargs):
        encoding = kwargs.get("input_ids", None)
        if encoding is not None and not isinstance(encoding, torch.Tensor) and hasattr(encoding, "items"):
            for key, value in kwargs.pop("input_ids").items():
                kwargs.setdefault(key, value)
        set_stance = getattr(torch.compiler, "set_stance", None)
        modules = self.modules() if isinstance(self, torch.nn.Module) else ()
        if set_stance is not None and any(getattr(m, "_compiled_call_impl", None) is not None for m in modules):
            # UNSLOTH_CPU_COMPILE layers are compiled for training; with a KV cache every layer would recompile
            # (the cache guards on each layer's state) until the recompile limit, costing seconds per layer.
            with set_stance("force_eager"):
                return generate(self, *args, **kwargs)
        return generate(self, *args, **kwargs)
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


def _check_save_method(save_method):
    if save_method not in ("lora", "merged_16bit"):
        raise NotImplementedError(f"Unsloth (CPU): save_method = {save_method!r} is not supported on CPU.")


def save_pretrained_merged(self, save_directory, tokenizer = None, save_method = "merged_16bit", **kwargs):
    """CPU save_pretrained_merged: "lora" (adapters) or "merged_16bit"."""
    _check_save_method(save_method)
    model, _ = _merged_copy(self, save_method)
    for key in ("maximum_memory_usage", "temporary_location", "max_shard_size_gb"):
        kwargs.pop(key, None)
    model.save_pretrained(save_directory, **{k: v for k, v in kwargs.items() if k in ("safe_serialization", "max_shard_size")})
    if tokenizer is not None:
        tokenizer.save_pretrained(save_directory)
    return save_directory


def push_to_hub_merged(self, repo_id, tokenizer = None, save_method = "merged_16bit", token = None, private = None, **kwargs):
    _check_save_method(save_method)
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
    # Stage the merged weights unrounded: the converter casts once to each requested type, instead of
    # rounding through fp16 / bf16 first (fp16 on a CPU without native bf16 would also clip bf16's range).
    unrounded = [m for m in methods if m in ("f32", "f16")]
    staged, _ = _merged_copy(self, "merged_32bit")
    staged.save_pretrained(merged_directory)
    if tokenizer is not None:
        tokenizer.save_pretrained(merged_directory)
    quantizer, converter = install_llama_cpp(print_output = False)
    base_name = os.path.basename(os.path.normpath(save_directory)) or "model"
    direct = {"bf16", "f16", "q8_0", "f32"}
    # llama-quantize refuses to requantize from q8_0, so k-quants are made from a bf16 GGUF.
    first = "bf16" if len(methods) > 1 or any(m not in direct for m in methods) else methods[0]
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

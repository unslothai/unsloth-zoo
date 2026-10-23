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
from .ops import patch_chunked_causal_lm_loss

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
        if load_in_8bit:
            _print("8-bit loading is not supported on CPU, loading 16-bit weights instead.")
        if token is None:
            token = os.environ.get("HF_TOKEN")
        dtype = _resolve_dtype(dtype)

        # 4-bit only where bitsandbytes' CPU backend really runs; full finetuning never quantizes.
        use_4bit = bool(load_in_4bit) and not load_in_8bit and not load_in_16bit and not full_finetuning
        if use_4bit and not cpu_bnb_4bit_supported():
            _print("bitsandbytes 4-bit does not work on this CPU, loading 16-bit weights instead.")
            use_4bit = False
        if not use_4bit and _PREQUANTIZED_SUFFIX.search(model_name):
            base = _PREQUANTIZED_SUFFIX.sub("", model_name)
            _print(f"{model_name} is pre-quantized; loading {base} for 16-bit training.")
            model_name = base

        quantization_config = None
        if use_4bit and not _PREQUANTIZED_SUFFIX.search(model_name):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_quant_type = "nf4",
                bnb_4bit_use_double_quant = True,
                bnb_4bit_compute_dtype = dtype,
            )
        for key in ("load_in_4bit", "load_in_8bit", "quantization_config", "use_exact_model_name",
                    "max_lora_rank", "gpu_memory_utilization", "float8_kv_cache", "random_state",
                    "fix_tokenizer", "resize_model_vocab", "unsloth_tiled_mlp", "offload_embedding"):
            kwargs.pop(key, None)

        if quantization_config is not None:
            # Only when set: an explicit None also discards a pre-quantized checkpoint's own config.
            kwargs["quantization_config"] = quantization_config
        # Left to transformers when unset: SDPA where the class supports it, eager otherwise (Bloom, MPT, ...).
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        load_kwargs = dict(
            **{_DTYPE_KWARG: dtype},
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
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, token = token, revision = revision, trust_remote_code = trust_remote_code,
            model_max_length = max_seq_length, padding_side = "right",
            **{k: kwargs[k] for k in ("cache_dir", "local_files_only", "subfolder") if k in kwargs},
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id

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
        FastCPUModel._attach(model)
        return model, tokenizer

    @staticmethod
    def _prepare(model, use_gradient_checkpointing):
        if use_gradient_checkpointing in (True, "unsloth"):
            # "unsloth" offloads activations to host RAM on GPUs; on CPU they already live there,
            # so it is plain recomputation.
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
            model.config.use_cache = False
        if os.environ.get("UNSLOTH_CPU_COMPILE", "0") == "1":
            count = _compile_layers(model)
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
        from peft import LoraConfig, get_peft_model

        if use_gradient_checkpointing is None:
            use_gradient_checkpointing = getattr(model, "_unsloth_cpu_gradient_checkpointing", "unsloth")
        if target_modules is None:
            # Unsloth's usual projections where the model has them, else PEFT's per-architecture default.
            leaf_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
            present = [m for m in _DEFAULT_TARGET_MODULES if m in leaf_names]
            target_modules = present or None
        elif not isinstance(target_modules, str):
            # As on GPU: embed_tokens / lm_head in target_modules are trained in full (continued pretraining).
            embeddings = [m for m in target_modules if m in ("embed_tokens", "lm_head")]
            if embeddings:
                target_modules = [m for m in target_modules if m not in embeddings]
                modules_to_save = list(dict.fromkeys([*(modules_to_save or []), *embeddings]))
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
    def for_training(model, use_gradient_checkpointing = True):
        model.train()
        _set_use_cache(model, False)
        return model

    @staticmethod
    def for_inference(model):
        model.eval()
        _set_use_cache(model, True)
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


def _set_use_cache(model, value):
    inner = model
    for _ in range(3):
        if hasattr(inner, "config"):
            inner.config.use_cache = value
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
    merged = copy.deepcopy(model).merge_and_unload()
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
    model.push_to_hub(repo_id, token = token, private = private, **hub)
    if tokenizer is not None:
        tokenizer.push_to_hub(repo_id, token = token, private = private, **hub)
    return repo_id


def save_pretrained_gguf(self, save_directory, tokenizer = None, quantization_method = "q8_0", **kwargs):
    """Merge to 16-bit, then llama.cpp's converter (+ llama-quantize for k-quants). Returns the GGUF paths."""
    from unsloth_zoo.llama_cpp import install_llama_cpp, convert_to_gguf, quantize_gguf

    methods = [quantization_method] if isinstance(quantization_method, str) else list(quantization_method)
    # Unsloth's GGUF aliases, as on the GPU save path.
    aliases = {"not_quantized": "bf16", "fast_quantized": "q8_0", "quantized": "q4_k_m", "none": "q8_0"}
    methods = list(dict.fromkeys(aliases.get(str(m).lower(), str(m).lower()) for m in methods))
    os.makedirs(save_directory, exist_ok = True)
    merged_directory = os.path.join(save_directory, "merged_16bit")
    save_pretrained_merged(self, merged_directory, tokenizer, save_method = "merged_16bit")
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
        target = first_file.replace(f".{first.upper()}.gguf", f".{method.upper()}.gguf")
        if target == first_file:
            target = os.path.join(save_directory, f"{base_name}.{method.upper()}.gguf")
        quantize_gguf(first_file, target, method, quantizer_location = quantizer, print_output = False)
        results.append(target)
    return results

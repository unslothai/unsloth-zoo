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
MLX utilities for Apple Silicon training.

Provides loss functions (CCE via mlx-cce, baseline CE), data batching,
weight extraction helpers, and model save/load/export for LoRA adapters
and merged models (safetensors, GGUF, HuggingFace Hub).
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import json
import os
import sys
import shutil
import tempfile
from pathlib import Path


def apply_gradient_checkpointing(model):
    """Apply gradient checkpointing to all transformer layers.

    Patches the layer class's __call__ to use mx.checkpoint, which
    recomputes activations during backward instead of storing them.
    Trades ~30% more compute for significant memory savings.

    Follows the same pattern as mlx_lm.tuner.trainer.grad_checkpoint.
    """
    layers = getattr(model, 'layers', None)
    if layers is None or len(layers) == 0:
        return
    layer_cls = type(layers[0])
    if getattr(layer_cls, '_orig_call', None) is not None:
        return  # already applied
    layer_cls._orig_call = layer_cls.__call__
    fn = layer_cls.__call__

    def checkpointed_fn(self, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            self.update(params)
            return fn(self, *args, **kwargs)
        return mx.checkpoint(inner_fn)(self.trainable_parameters(), *args, **kwargs)

    layer_cls.__call__ = checkpointed_fn


def remove_gradient_checkpointing(model):
    """Remove gradient checkpointing, restoring original layer __call__."""
    layers = getattr(model, 'layers', None)
    if layers is None or len(layers) == 0:
        return
    layer_cls = type(layers[0])
    orig = getattr(layer_cls, '_orig_call', None)
    if orig is not None:
        layer_cls.__call__ = orig
        del layer_cls._orig_call


def _get_lm_head_layer(model):
    """Get the raw LM head layer (QuantizedLinear or Linear/Embedding).

    Checks for a separate lm_head first (untied models like Qwen), then
    falls back to embed_tokens (tied models like Gemma/Llama).

    Returns the layer object (not its weight), so callers can access
    .weight, .scales, .biases, .group_size, .bits for quantized layers.
    """
    if hasattr(model, "lm_head") and model.lm_head is not None:
        return model.lm_head
    return model.model.embed_tokens


def _is_quantized_layer(layer):
    """Check if a layer has quantized weights (has .scales attribute)."""
    return hasattr(layer, "scales")


def has_cce_kernel():
    """Check if mx.fast.cce_loss is available (requires mlx-cce)."""
    return hasattr(mx.fast, "cce_loss")


def _get_logit_softcap(model):
    """Get logit softcapping value if model uses it (e.g. Gemma-2), else 0.0."""
    softcap = getattr(model, "final_logit_softcapping", None)
    if softcap is None and hasattr(model, "args"):
        softcap = getattr(model.args, "final_logit_softcapping", None)
    return float(softcap) if softcap is not None and softcap > 0 else 0.0


def _is_lm_head_trainable(model):
    """Check if the LM head weight is trainable (not frozen by LoRA).

    For LoRA training, the LM head weight is frozen — computing its gradient
    in CCE is a wasted V x chunk_size x H matmul per chunk. Returns False
    when the weight should be wrapped with mx.stop_gradient.
    """
    trainable = dict(mlx.utils.tree_flatten(model.trainable_parameters()))
    for key in trainable:
        if 'lora' not in key:
            if 'lm_head' in key or 'embed_tokens.weight' in key:
                return True
    return len(trainable) == 0  # no LoRA = full fine-tuning = trainable


def make_cce_loss_fn(model):
    """Create a CCE loss function using mx.fast.cce_loss from mlx-cce.

    CCE computes cross-entropy directly from hidden states and the LM head weight,
    avoiding full logit materialization. This saves significant memory for large
    vocabularies. For V=256K+ models, CCE is both faster and more memory-efficient.

    If the LM head is quantized, passes raw uint32 weight + scales + biases
    directly to cce_loss, using fused quantized matmul kernels.

    Returns:
        A function (model, batch, lengths) -> (loss, ntoks).
    """
    if not has_cce_kernel():
        raise RuntimeError(
            "mx.fast.cce_loss not available. Install mlx-cce: pip install mlx-cce"
        )

    softcap = _get_logit_softcap(model)
    if softcap > 0:
        print(f"Unsloth: CCE using logit_softcap={softcap} for this model.")

    lm_layer = _get_lm_head_layer(model)
    use_quantized = _is_quantized_layer(lm_layer)

    if use_quantized:
        group_size = getattr(lm_layer, "group_size", 64)
        bits = getattr(lm_layer, "bits", 4)
        print(f"Unsloth: CCE using quantized matmul (group_size={group_size}, bits={bits})")
        _has_lm_head_q = (hasattr(model, "lm_head")
                          and model.lm_head is not None
                          and hasattr(model.lm_head, "scales"))
        _has_biases = hasattr(lm_layer, "biases")

        def loss_fn(model, batch, lengths):
            inputs, targets = batch[:, :-1], batch[:, 1:]
            hidden = model.model(inputs)
            layer = model.lm_head if _has_lm_head_q else model.model.embed_tokens
            w = layer.weight
            sc = layer.scales
            bi = layer.biases if _has_biases else None
            steps = mx.arange(1, targets.shape[1] + 1)
            mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
            masked_targets = mx.where(mask, targets, -100)
            ntoks = mask.sum()
            loss = mx.fast.cce_loss(
                hidden, w, masked_targets,
                scales=sc, biases=bi,
                group_size=group_size, bits=bits,
                ignore_index=-100,
                logit_softcap=softcap,
            )
            loss = loss.astype(mx.float32).sum() / ntoks
            return loss, ntoks
    else:
        _has_lm_head = (hasattr(model, "lm_head")
                        and model.lm_head is not None
                        and hasattr(model.lm_head, "weight"))
        _skip_weight_grad = not _is_lm_head_trainable(model)
        if _skip_weight_grad:
            print("Unsloth: CCE skipping weight gradient (LM head is frozen).")

        def loss_fn(model, batch, lengths):
            inputs, targets = batch[:, :-1], batch[:, 1:]
            hidden = model.model(inputs)
            w = model.lm_head.weight if _has_lm_head else model.model.embed_tokens.weight
            if _skip_weight_grad:
                w = mx.stop_gradient(w)
            steps = mx.arange(1, targets.shape[1] + 1)
            mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
            masked_targets = mx.where(mask, targets, -100)
            ntoks = mask.sum()
            loss = mx.fast.cce_loss(
                hidden, w, masked_targets,
                ignore_index=-100,
                logit_softcap=softcap,
            )
            loss = loss.astype(mx.float32).sum() / ntoks
            return loss, ntoks

    return loss_fn


def make_baseline_loss_fn():
    """Create a standard cross-entropy loss function.

    Uses the full logit computation through the LM head, then applies
    nn.losses.cross_entropy. This is the fallback when CCE is not available.

    Returns:
        A function (model, batch, lengths) -> (loss, ntoks).
    """
    def loss_fn(model, batch, lengths):
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = model(inputs)
        steps = mx.arange(1, targets.shape[1] + 1)
        mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
        ce = nn.losses.cross_entropy(logits, targets) * mask
        ntoks = mask.sum()
        loss = ce.astype(mx.float32).sum() / ntoks
        return loss, ntoks

    return loss_fn


def _prepare_dataset(dataset, tokenizer, dataset_text_field="text",
                     formatting_func=None):
    """Wrap a HuggingFace dataset into mlx-lm's dataset classes.

    Uses TextDataset + CacheDataset from mlx_lm so that tokenization
    (including EOS appending) matches mlx-lm's own training pipeline exactly.

    If a formatting_func is provided, each item is pre-formatted into a
    ``{"text": ...}`` dict before wrapping.

    Returns:
        A CacheDataset ready for ``iterate_batches``.
    """
    from mlx_lm.tuner.datasets import TextDataset, CacheDataset

    # Pre-format items into [{"text": str}, ...] so TextDataset can consume them.
    formatted = []
    for item in dataset:
        if formatting_func is not None:
            result = formatting_func(item)
            texts = result if isinstance(result, list) else [result]
        elif isinstance(item, dict):
            texts = []
            if dataset_text_field in item:
                texts = [item[dataset_text_field]]
            else:
                for key in ("text", "content", "instruction"):
                    if key in item:
                        texts = [item[key]]
                        break
        elif isinstance(item, str):
            texts = [item]
        else:
            continue

        for text in texts:
            if text:
                formatted.append({"text": text})

    if not formatted:
        raise ValueError(
            f"No text data found. Provide a dataset with a "
            f"'{dataset_text_field}' column."
        )

    return CacheDataset(TextDataset(formatted, tokenizer, text_key="text"))


def create_batches(dataset, tokenizer, batch_size, max_seq_length,
                   num_batches=None, seed=42, dataset_text_field="text",
                   formatting_func=None):
    """Pre-tokenize and batch a HuggingFace dataset for MLX training.

    Uses iterate_batches from mlx_lm for efficient dynamic-padding batching:
    samples are sorted by length, grouped into batches, and padded to the
    max length within each batch (rounded up to the nearest multiple of 32),
    capped at max_seq_length.

    Tokenization is delegated to mlx_lm's TextDataset (appends EOS, etc.)
    so behaviour matches ``mlx_lm.lora`` exactly.

    Returns:
        List of (batch, lengths) tuples, where batch has shape
        (batch_size, padded_length) and lengths has shape (batch_size, 2)
        with [offset, length] per sequence (from iterate_batches).
    """
    from mlx_lm.tuner.trainer import iterate_batches

    ds = _prepare_dataset(
        dataset, tokenizer, dataset_text_field, formatting_func
    )

    batch_pairs = []
    for batch, lengths_info in iterate_batches(
        ds, batch_size, max_seq_length,
        loop=(num_batches is not None),
        seed=seed,
    ):
        batch_pairs.append((batch, lengths_info))
        if num_batches is not None and len(batch_pairs) >= num_batches:
            break

    mx.eval([b for b, _ in batch_pairs] + [l for _, l in batch_pairs])
    return batch_pairs


def iterate_training_batches(dataset, tokenizer, batch_size, max_seq_length,
                             seed=42, dataset_text_field="text",
                             formatting_func=None):
    """Streaming batch generator for MLX training.

    Wraps mlx-lm's iterate_batches(loop=True) as a generator, avoiding
    materializing all batches in memory at once. Useful for large datasets.

    Yields:
        (batch, lengths) tuples — same format as create_batches.
    """
    from mlx_lm.tuner.trainer import iterate_batches

    ds = _prepare_dataset(
        dataset, tokenizer, dataset_text_field, formatting_func
    )

    for batch, lengths_info in iterate_batches(
        ds, batch_size, max_seq_length,
        loop=True,
        seed=seed,
    ):
        yield batch, lengths_info


def save_lora_adapters(model, path, adapter_config=None):
    """Save LoRA adapter weights to disk.

    Args:
        model: MLX model with LoRA layers.
        path: Directory to save adapters.
        adapter_config: Optional dict with LoRA config metadata.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Collect only trainable (LoRA) parameters — flatten nested dict for safetensors
    trainable = dict(mlx.utils.tree_flatten(model.trainable_parameters()))

    if trainable:
        mx.save_safetensors(str(path / "adapters.safetensors"), trainable)

    if adapter_config:
        with open(path / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=2)


def _get_model_config(model):
    """Extract config dict from an MLX model.

    mlx-lm stores the raw config dict at model._config when loaded.
    Falls back to reconstructing from model.args dataclass.
    """
    # Prefer the raw config dict stashed by our loader
    if hasattr(model, "_config") and isinstance(model._config, dict):
        return dict(model._config)

    # Reconstruct from the ModelArgs dataclass
    if hasattr(model, "args"):
        import dataclasses
        if dataclasses.is_dataclass(model.args):
            return dataclasses.asdict(model.args)

    return {}


def _get_src_path(model):
    """Get the original model source path/repo for copying auxiliary files."""
    return getattr(model, "_src_path", None)


def save_merged_model(model, tokenizer, path):
    """Fuse LoRA weights and save the full merged model.

    Produces an HF-compatible directory with sharded safetensors,
    config.json, tokenizer files, and a model card. The output can
    be reloaded with ``mlx_lm.load()`` or uploaded to HuggingFace Hub.

    Args:
        model: MLX model with LoRA layers.
        tokenizer: Tokenizer to save alongside.
        path: Directory to save merged model.
    """
    from mlx_lm.utils import save_model, save_config, create_model_card
    from mlx.utils import tree_unflatten

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Fuse LoRA weights into base model using mlx-lm's pattern
    model.eval()
    fused_linears = [
        (n, m.fuse())
        for n, m in model.named_modules()
        if hasattr(m, "fuse")
    ]
    if fused_linears:
        model.update_modules(tree_unflatten(fused_linears))
    de_lora_model = model

    # Save sharded safetensors + index.json
    save_model(path, de_lora_model, donate_model=False)

    # Save config.json
    config = _get_model_config(model)
    if config:
        save_config(config, config_path=path / "config.json")

    # Save tokenizer
    tokenizer.save_pretrained(str(path))

    # Copy auxiliary files (generation_config.json, *.py) from source
    src_path = _get_src_path(model)
    if src_path is not None:
        src_path = Path(src_path)
        if src_path.exists():
            import glob as globmod
            for pattern in ["generation_config.json", "*.py"]:
                for f in globmod.glob(str(src_path / pattern)):
                    shutil.copy(f, path)

    # Model card
    hf_repo = getattr(model, "_hf_repo", None)
    try:
        create_model_card(path, hf_repo)
    except Exception:
        # Fails if hf_repo doesn't exist on Hub — create a minimal card
        readme = path / "README.md"
        if not readme.exists():
            readme.write_text("---\nlibrary_name: mlx\ntags:\n- mlx\n- unsloth\n---\n")

    print(f"Unsloth: Merged model saved to {path}")


def save_pretrained_merged(
    model,
    tokenizer,
    save_directory,
    push_to_hub=False,
    token=None,
    private=None,
    tags=None,
):
    """Save LoRA-fused model in HF-compatible format.

    This is the user-facing API matching the CUDA path's
    ``model.save_pretrained_merged()``.

    Args:
        model: MLX model with LoRA layers.
        tokenizer: Tokenizer to save alongside.
        save_directory: Output directory path.
        push_to_hub: If True, upload to HuggingFace Hub after saving.
        token: HuggingFace token for pushing.
        private: Whether the HF repo should be private.
        tags: Additional tags for the model card.
    """
    save_merged_model(model, tokenizer, save_directory)

    if push_to_hub:
        push_to_hub_merged(
            model, tokenizer, save_directory,
            token=token, private=private, tags=tags,
        )


def _install_llama_cpp_macos(llama_cpp_folder="llama.cpp"):
    """Install llama.cpp on macOS by cloning and building with cmake."""
    import subprocess

    if not os.path.exists(llama_cpp_folder):
        print("Unsloth: Cloning llama.cpp...")
        subprocess.run(
            ["git", "clone", "https://github.com/ggml-org/llama.cpp", llama_cpp_folder],
            check=True,
        )

    # Install Python dependencies — use gguf from the cloned repo to stay in sync
    gguf_py_dir = os.path.join(llama_cpp_folder, "gguf-py")
    if os.path.exists(gguf_py_dir):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", gguf_py_dir,
             "protobuf", "sentencepiece"],
            check=True, capture_output=True,
        )
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "gguf",
             "protobuf", "sentencepiece"],
            check=True, capture_output=True,
        )

    # Build with cmake (Metal support on macOS)
    build_dir = os.path.join(llama_cpp_folder, "build")
    print("Unsloth: Building llama.cpp with cmake...")
    subprocess.run(
        ["cmake", llama_cpp_folder, "-B", build_dir,
         "-DBUILD_SHARED_LIBS=OFF", "-DGGML_METAL=ON"],
        check=True, capture_output=True,
    )

    import psutil
    n_jobs = psutil.cpu_count() or 4
    targets = ["llama-quantize", "llama-cli", "llama-gguf-split"]
    target_args = []
    for t in targets:
        target_args += ["--target", t]

    subprocess.run(
        ["cmake", "--build", build_dir, "--config", "Release",
         f"-j{n_jobs}", "--clean-first"] + target_args,
        check=True, capture_output=True,
    )

    # Copy binaries to llama.cpp root
    bin_dir = os.path.join(build_dir, "bin")
    if os.path.exists(bin_dir):
        import glob as globmod
        for binary in globmod.glob(os.path.join(bin_dir, "llama-*")):
            shutil.copy(binary, llama_cpp_folder)

    print("Unsloth: llama.cpp installed successfully.")


def save_pretrained_gguf(
    model,
    tokenizer,
    save_directory,
    quantization_method="fast_quantized",
):
    """Save LoRA-fused model in GGUF format for llama.cpp inference.

    Follows the same pipeline as unsloth's CUDA path:
    1. Merge LoRA and save as HF-compatible safetensors
    2. Install/check llama.cpp
    3. Download and patch convert_hf_to_gguf.py
    4. Convert safetensors -> GGUF (bf16/f16 intermediate)
    5. Quantize to target format if needed

    Args:
        model: MLX model (with or without LoRA).
        tokenizer: Tokenizer to save alongside.
        save_directory: Output directory for GGUF file(s).
        quantization_method: Quantization to apply. Options:
            "not_quantized" - bf16, no quantization
            "fast_quantized" - q8_0 (fast, good quality)
            "quantized" - q4_k_m (small, fast inference)
            Or any llama.cpp quant type: q2_k, q3_k_m, q4_k_m, q5_k_m,
            q6_k, q8_0, f16, bf16, f32, etc.
    """
    from .llama_cpp import (
        convert_to_gguf,
        quantize_gguf,
        install_llama_cpp,
        check_llama_cpp,
        _download_convert_hf_to_gguf,
    )

    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    # Map friendly names to llama.cpp quant types
    quant_map = {
        "not_quantized": "bf16",
        "fast_quantized": "q8_0",
        "quantized": "q4_k_m",
        None: "q8_0",
    }
    quant_type = quant_map.get(quantization_method, quantization_method)

    # Apple Silicon always supports bf16
    model_dtype = "bf16"

    # Determine first_conversion (intermediate GGUF format before quantizing)
    # Same logic as unsloth CUDA path's save_to_gguf()
    if quant_type in ("bf16", "f16", "f32"):
        first_conversion = quant_type
    elif quant_type == "q8_0":
        # q8_0 can be done directly by convert_hf_to_gguf.py
        first_conversion = "None"
    else:
        # For all other quant types, first convert to bf16 then quantize
        first_conversion = "bf16"

    first_conversion_dtype = "" if first_conversion == "None" else first_conversion

    # Step 1: Save merged model to a temp HF-format directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "merged"
        print("Unsloth: Merging LoRA weights and saving to 16-bit...")
        save_merged_model(model, tokenizer, tmp_path)

        # Step 2: Ensure llama.cpp is installed
        llama_cpp_folder = "llama.cpp"
        try:
            check_llama_cpp(llama_cpp_folder)
        except Exception:
            print("Unsloth: Installing llama.cpp (this only happens once)...")
            _install_llama_cpp_macos(llama_cpp_folder)

        # Step 3: Download and patch convert_hf_to_gguf.py
        converter = os.path.join(llama_cpp_folder, "unsloth_convert_hf_to_gguf.py")
        supported_text_archs = None
        supported_vision_archs = None
        if not os.path.exists(converter):
            result = _download_convert_hf_to_gguf()  # no args — uses defaults
            if isinstance(result, tuple) and len(result) >= 3:
                converter, supported_text_archs, supported_vision_archs = result[:3]

        # Step 4: Get model name for output filename
        hf_repo = getattr(model, "_hf_repo", None)
        if hf_repo:
            model_name = hf_repo.split("/")[-1]
        else:
            model_name = "model"

        output_base = str(save_directory / model_name)

        # Step 5: Convert HF -> GGUF
        print(f"Unsloth: Converting to GGUF format...")
        kwargs = dict(
            model_name=output_base,
            input_folder=str(tmp_path),
            model_dtype=model_dtype,
            quantization_type=first_conversion,
            converter_location=converter,
            is_vlm=False,
            is_gpt_oss=False,
            print_output=True,
        )
        if supported_text_archs is not None:
            kwargs["supported_text_archs"] = supported_text_archs
            kwargs["supported_vision_archs"] = supported_vision_archs
        convert_to_gguf(**kwargs)

        # Step 6: Quantize if the target quant differs from first_conversion
        if quant_type not in ("bf16", "f16", "f32") and first_conversion != "None":
            quantizer = os.path.join(llama_cpp_folder, "llama-quantize")
            base_gguf = f"{output_base}.{first_conversion.upper()}.gguf"
            final_gguf = f"{output_base}.{quant_type.upper()}.gguf"

            print(f"Unsloth: Quantizing to {quant_type}...")
            quantize_gguf(
                input_gguf=base_gguf,
                output_gguf=final_gguf,
                quant_type=quant_type,
                quantizer_location=quantizer,
                print_output=True,
            )
            # Remove intermediate bf16 gguf to save space
            if os.path.exists(base_gguf) and base_gguf != final_gguf:
                os.remove(base_gguf)
                print(f"Unsloth: Removed intermediate {Path(base_gguf).name}")

    # List produced files
    gguf_files = sorted(save_directory.glob("*.gguf"))
    for f in gguf_files:
        size_gb = f.stat().st_size / (1024**3)
        print(f"Unsloth: Saved {f.name} ({size_gb:.2f} GB)")
    print(f"Unsloth: GGUF export complete -> {save_directory}")


def push_to_hub_merged(
    model,
    tokenizer,
    save_directory,
    repo_id=None,
    token=None,
    private=None,
    tags=None,
):
    """Push merged model to HuggingFace Hub.

    Args:
        model: MLX model.
        tokenizer: Tokenizer.
        save_directory: Local path with saved model (or where to save).
        repo_id: HuggingFace repo ID (e.g. "username/model-name").
            If None, uses save_directory as repo_id.
        token: HuggingFace token.
        private: Whether repo should be private.
        tags: Additional tags.
    """
    from mlx_lm.utils import upload_to_hub

    save_directory = Path(save_directory)

    # Save first if not already saved
    if not (save_directory / "model.safetensors.index.json").exists():
        save_merged_model(model, tokenizer, save_directory)

    if repo_id is None:
        repo_id = save_directory.name

    upload_to_hub(str(save_directory), repo_id)
    print(f"Unsloth: Pushed to https://huggingface.co/{repo_id}")


def push_to_hub_gguf(
    model,
    tokenizer,
    save_directory,
    repo_id,
    quantization_method="fast_quantized",
    token=None,
    private=None,
):
    """Export to GGUF and push to HuggingFace Hub.

    Args:
        model: MLX model.
        tokenizer: Tokenizer.
        save_directory: Local path for GGUF output.
        repo_id: HuggingFace repo ID.
        quantization_method: GGUF quantization type.
        token: HuggingFace token.
        private: Whether repo should be private.
    """
    from huggingface_hub import HfApi

    save_directory = Path(save_directory)

    # Export to GGUF
    save_pretrained_gguf(model, tokenizer, save_directory, quantization_method)

    # Upload GGUF files
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private)

    gguf_files = list(save_directory.glob("*.gguf"))
    for gguf_file in gguf_files:
        api.upload_file(
            path_or_fileobj=str(gguf_file),
            path_in_repo=gguf_file.name,
            repo_id=repo_id,
        )

    print(f"Unsloth: GGUF pushed to https://huggingface.co/{repo_id}")

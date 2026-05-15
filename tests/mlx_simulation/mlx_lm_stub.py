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

# Unsloth Zoo - Utilities for Unsloth
# mlx_lm stub — load, utils, tuner.lora, tuner.utils, sample_utils, stream_generate
"""
mlx_lm — sample/load/tuner facade.

PR-A imports a fixed list of symbols from mlx_lm; PR-B's inference
path uses sample_utils and stream_generate.  See plan §8.10 for the
catalog.

Phase 1 ships skeleton stubs (no-op or NotImplementedError on call).
Phase 5 wires real implementations under mlx_helpers/.
"""

from __future__ import annotations

import sys
import types

import torch


# ---------------------------------------------------------------------------
# mlx_lm.load — top-level entry
# ---------------------------------------------------------------------------
def load(repo_path, *args, **kwargs):
    """Load a model + tokenizer pair.

    Phase 1 raises with a clear message; Phase 5 wires HF transformers.
    """
    from .mlx_helpers.mlx_lm_compat import load as real_load
    return real_load(repo_path, *args, **kwargs)


# ---------------------------------------------------------------------------
# mlx_lm.stream_generate — generation streaming
# ---------------------------------------------------------------------------
def stream_generate(model, tokenizer, prompt, *args, **kwargs):
    from .mlx_helpers.stream_generate import stream_generate as real_sg
    yield from real_sg(model, tokenizer, prompt, *args, **kwargs)


# ---------------------------------------------------------------------------
# Submodules — populate after inject_into_sys_modules
# ---------------------------------------------------------------------------
def _pkg(name):
    m = types.ModuleType(name)
    m.__path__ = []
    return m


utils_module = _pkg("mlx_lm.utils")


def _utils_download(repo_id, *args, **kwargs):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo_id, **{
        k: v for k, v in kwargs.items()
        if k in ("revision", "cache_dir", "local_dir", "allow_patterns",
                 "ignore_patterns", "token", "endpoint")
    })


def _utils_load_model(model_path, *args, **kwargs):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model_path, *args, **kwargs)


def _utils_load_tokenizer(model_path, *args, **kwargs):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, *args, **kwargs)


def _utils_quantize_model(model, *args, **kwargs):
    """Phase 1 stub — return model unchanged.  Phase 5 implements affine quant."""
    return model


def _utils_save_model(model, path, *args, **kwargs):
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(path)
    else:
        from safetensors.torch import save_file
        save_file(model.parameters() if callable(getattr(model, "parameters", None)) else model, path)


def _utils_save_config(config, path):
    import json
    import os
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def _utils_create_model_card(model_name, *args, **kwargs):
    return f"# {model_name}\n\nMLX model card placeholder.\n"


utils_module._download = _utils_download
utils_module.load_model = _utils_load_model
utils_module.load_tokenizer = _utils_load_tokenizer
utils_module.quantize_model = _utils_quantize_model
utils_module.save_model = _utils_save_model
utils_module.save_config = _utils_save_config
utils_module.create_model_card = _utils_create_model_card


# --- mlx_lm.tuner.lora.LoRALinear --------------------------------------
tuner_module = _pkg("mlx_lm.tuner")
tuner_lora_module = _pkg("mlx_lm.tuner.lora")
tuner_utils_module = _pkg("mlx_lm.tuner.utils")
tuner_datasets_module = _pkg("mlx_lm.tuner.datasets")
tuner_trainer_module = _pkg("mlx_lm.tuner.trainer")


def _placeholder_lora_linear(*args, **kwargs):
    from .mlx_helpers.lora_linear import LoRALinear
    return LoRALinear(*args, **kwargs)


# Expose the LoRALinear class itself for `isinstance` checks.
class LoRALinear:
    """Placeholder until mlx_helpers.lora_linear is wired in Phase 5.

    isinstance(x, LoRALinear) returns True only for instances of this
    exact class.  Phase 5 makes the helper subclass this so existing
    callsites work.
    """

    def __init__(self, *args, **kwargs):
        from .mlx_helpers.lora_linear import LoRALinear as _Real
        self._real = _Real(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __call__(self, x):
        return self._real(x)


tuner_lora_module.LoRALinear = LoRALinear


def _placeholder_linear_to_lora(model, num_layers, config, use_dora=False):
    from .mlx_helpers.lora_linear import linear_to_lora_layers as real
    return real(model, num_layers, config, use_dora=use_dora)


def _placeholder_load_adapters(model, adapter_path):
    from .mlx_helpers.lora_linear import load_adapters as real
    return real(model, adapter_path)


tuner_utils_module.linear_to_lora_layers = _placeholder_linear_to_lora
tuner_utils_module.load_adapters = _placeholder_load_adapters


# --- mlx_lm.tuner.datasets ---------------------------------------------
class TextDataset:
    def __init__(self, tokenizer, dataset, max_seq_length=2048, *args, **kwargs):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]
        if isinstance(text, dict) and "text" in text:
            text = text["text"]
        encoded = self.tokenizer(text, truncation=True, max_length=self.max_seq_length)
        return encoded["input_ids"], encoded["input_ids"]


class CacheDataset:
    def __init__(self, dataset, *args, **kwargs):
        self.dataset = dataset
        self._cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx not in self._cache:
            self._cache[idx] = self.dataset[idx]
        return self._cache[idx]


tuner_datasets_module.TextDataset = TextDataset
tuner_datasets_module.CacheDataset = CacheDataset


# --- mlx_lm.tuner.trainer ----------------------------------------------
def _iterate_batches(dataset, batch_size, max_seq_length, train=True, *args, **kwargs):
    """Yield (batch, lengths) pairs.  Simple pad-to-max-in-batch."""
    indices = list(range(len(dataset)))
    if train:
        import random
        random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch_pairs = [dataset[i] for i in batch_indices]
        # Each item is (input_ids, labels) — pad to max length in batch
        if not batch_pairs:
            continue
        first = batch_pairs[0]
        if isinstance(first, tuple):
            input_ids = [pair[0] for pair in batch_pairs]
        else:
            input_ids = batch_pairs
        max_len = min(max(len(x) for x in input_ids), max_seq_length)
        padded = [list(x[:max_len]) + [0] * (max_len - len(x[:max_len])) for x in input_ids]
        lengths = [min(len(x), max_seq_length) for x in input_ids]
        yield torch.tensor(padded, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


tuner_trainer_module.iterate_batches = _iterate_batches


# --- mlx_lm.sample_utils -----------------------------------------------
sample_utils_module = _pkg("mlx_lm.sample_utils")


def _make_sampler(temp=1.0, top_p=1.0, top_k=0, min_tokens_to_keep=1, **kw):
    """Build a callable that takes logits and returns a sampled token id."""
    def sampler(logits):
        if temp == 0:
            return torch.argmax(logits, dim=-1)
        scaled = logits / max(temp, 1e-6)
        if top_k > 0:
            top_vals, top_idx = torch.topk(scaled, k=min(top_k, scaled.shape[-1]), dim=-1)
            mask = torch.full_like(scaled, float("-inf"))
            mask.scatter_(-1, top_idx, top_vals)
            scaled = mask
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(scaled, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(sorted_probs, dim=-1)
            remove = cum > top_p
            # always keep top min_tokens_to_keep
            remove[..., :min_tokens_to_keep] = False
            sorted_logits[remove] = float("-inf")
            scaled = torch.gather(sorted_logits, -1, torch.argsort(sorted_idx, dim=-1))
        probs = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    return sampler


def _make_logits_processors(*args, **kwargs):
    """Repetition penalty etc.  For Phase 1: empty list (no-op processors)."""
    return []


sample_utils_module.make_sampler = _make_sampler
sample_utils_module.make_logits_processors = _make_logits_processors


# --- mlx_lm.models.gated_delta -----------------------------------------
models_module = _pkg("mlx_lm.models")
gated_delta_module = _pkg("mlx_lm.models.gated_delta")


def _gated_delta_placeholder(*args, **kwargs):
    raise NotImplementedError(
        "mlx-shim: mlx_lm.models.gated_delta not implemented in Phase 1; "
        "use the unsloth_zoo.gated_delta_vjp helpers directly."
    )


gated_delta_module.gated_delta_update = _gated_delta_placeholder
models_module.gated_delta = gated_delta_module


# ---------------------------------------------------------------------------
__path__ = []


def __getattr__(name):
    from .mlx_stub import _Noop
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return _Noop(f"mlx_lm.{name}")


def inject_into_sys_modules():
    this = sys.modules[__name__]
    this.utils = utils_module
    this.tuner = tuner_module
    this.sample_utils = sample_utils_module
    this.models = models_module
    tuner_module.lora = tuner_lora_module
    tuner_module.utils = tuner_utils_module
    tuner_module.datasets = tuner_datasets_module
    tuner_module.trainer = tuner_trainer_module
    sys.modules.update({
        "mlx_lm": this,
        "mlx_lm.utils": utils_module,
        "mlx_lm.tuner": tuner_module,
        "mlx_lm.tuner.lora": tuner_lora_module,
        "mlx_lm.tuner.utils": tuner_utils_module,
        "mlx_lm.tuner.datasets": tuner_datasets_module,
        "mlx_lm.tuner.trainer": tuner_trainer_module,
        "mlx_lm.sample_utils": sample_utils_module,
        "mlx_lm.models": models_module,
        "mlx_lm.models.gated_delta": gated_delta_module,
    })

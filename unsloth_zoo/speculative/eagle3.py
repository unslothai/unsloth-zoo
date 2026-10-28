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

"""EAGLE-3 drafters for targets without a native MTP head, in the vllm-project/speculators layout.

Contract (speculators `models/eagle3/core.py`, vLLM `llama_eagle3.py`, unslothai/llama.cpp
`src/models/eagle3.cpp`):
- aux id k = residual stream entering target layer k (k == L: last layer output, pre final norm);
  fused g = fc(cat[aux]);
- row t = (g_t, x_{t+1}) at position t + 1 predicts x_{t+2}; one llama decoder layer whose
  attention reads cat[input_layernorm(embed(x)), hidden_norm(g)], residual = raw g;
- logits = lm_head(norm(h)) over the draft vocabulary, draft id d -> target id d + d2t[d];
- step k > 0 feeds back the previous step's pre-norm output h; training-time test unrolls the
  steps with keys = causal step-0 rows plus each later step's own row (diagonal).
"""

__all__ = [
    "EAGLE3_TARGET_TYPES",
    "eagle3_supported",
    "build_eagle3_vocab_map",
    "Eagle3Drafter",
    "build_eagle3_drafter",
    "save_eagle3_drafter",
    "load_eagle3_drafter",
    "convert_eagle3_to_gguf",
    "eagle3_loss",
    "Eagle3TrainingArguments",
    "UnslothEagle3Trainer",
]

import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer, TrainingArguments

from .trainer import token_average_scale
from .capture import effective_lm_head_weight, resolve_text_parts, run_frozen_target

# Target text model_types where llama.cpp exposes layer inputs (`t_layer_inp`) AND vLLM 0.31
# implements SupportsEagle3. Mistral checkpoints convert to llama.cpp's llama arch.
EAGLE3_TARGET_TYPES = {
    "llama"           : "llama",
    "mistral"         : "llama",
    "qwen3"           : "qwen3",
    "qwen3_moe"       : "qwen3moe",
    "qwen3_5"         : "qwen35",
    "qwen3_5_text"    : "qwen35",
    "qwen3_5_moe"     : "qwen35moe",
    "qwen3_5_moe_text": "qwen35moe",
    "qwen3_next"      : "qwen3next",
    "gemma4"          : "gemma4",
    "gemma4_text"     : "gemma4",
    "gpt_oss"         : "openai-moe",
    "nemotron_h"      : "nemotron-h",
    "minimax_m2"      : "minimax-m2",
}

_UNSUPPORTED_REASONS = {
    "gemma3"     : "llama.cpp's gemma3 graph does not expose the layer inputs EAGLE-3 reads, and vLLM has no EAGLE-3 path for Gemma 3",
    "gemma3_text": "llama.cpp's gemma3 graph does not expose the layer inputs EAGLE-3 reads, and vLLM has no EAGLE-3 path for Gemma 3",
    "mistral3"   : "llama.cpp's mistral3 graph does not expose the layer inputs EAGLE-3 reads",
    "ministral3" : "llama.cpp's mistral3 graph does not expose the layer inputs EAGLE-3 reads",
    "qwen2"      : "llama.cpp's qwen2 graph does not expose the layer inputs EAGLE-3 reads",
}


def _text_config(config):
    if hasattr(config, "get_text_config"):
        return config.get_text_config()
    return getattr(config, "text_config", None) or config


def eagle3_supported(config):
    """(supported, reason): can an EAGLE-3 drafter for this target run in both vLLM and llama.cpp?"""
    text = _text_config(config)
    model_type = getattr(text, "model_type", None)
    if model_type in EAGLE3_TARGET_TYPES:
        return True, f"{model_type}: EAGLE-3 runs in vLLM and llama.cpp ({EAGLE3_TARGET_TYPES[model_type]})"
    reason = _UNSUPPORTED_REASONS.get(model_type)
    if reason is None:
        reason = f"{model_type}: no EAGLE-3 support in both vLLM and llama.cpp"
    return False, reason


def build_eagle3_vocab_map(token_id_lists, target_vocab_size, draft_vocab_size = 32000):
    """(d2t, t2d) from token frequencies: the `draft_vocab_size` most frequent ids (ties by id),
    padded with the lowest unused ids, sorted. target id = draft index + d2t[index]."""
    target_vocab_size = int(target_vocab_size)
    draft_vocab_size = min(int(draft_vocab_size), target_vocab_size)
    counts = Counter()
    for ids in token_id_lists:
        counts.update(int(t) for t in ids if 0 <= int(t) < target_vocab_size)
    selected = sorted(counts, key = lambda t: (-counts[t], t))[:draft_vocab_size]
    if len(selected) < draft_vocab_size:
        chosen = set(selected)
        for t in range(target_vocab_size):
            if t not in chosen:
                selected.append(t)
                if len(selected) >= draft_vocab_size:
                    break
    selected.sort()
    d2t = torch.tensor(selected, dtype = torch.long) - torch.arange(draft_vocab_size, dtype = torch.long)
    t2d = torch.zeros(target_vocab_size, dtype = torch.bool)
    t2d[selected] = True
    return d2t, t2d


def _attr(config, name, default = None):
    # transformers 5 heterogeneous configs (Gemma 4) raise on a per-layer attribute read globally.
    try:
        return getattr(config, name, default)
    except Exception:
        return default


def _draft_layer_config(text, attn_implementation = "sdpa"):
    """A one-layer llama config shaped like the target (speculators `_build_draft_config`)."""
    import inspect

    from transformers import LlamaConfig

    hidden = int(text.hidden_size)
    heads = int(text.num_attention_heads)
    kv_heads = int(_attr(text, "num_key_value_heads") or heads)
    head_dim = _attr(text, "head_dim")
    if head_dim and hidden % heads != 0 and hidden % head_dim == 0:
        heads = hidden // head_dim
        if heads % kv_heads != 0:
            kv_heads = heads
    head_dim = int(head_dim or hidden // heads)
    intermediate = _attr(text, "intermediate_size")
    moe = _attr(text, "num_experts") or _attr(text, "num_local_experts")
    # MoE targets have no dense FFN width; speculators uses 3 x hidden.
    intermediate = intermediate if isinstance(intermediate, int) and not moe else 3 * hidden
    rope = _attr(text, "rope_parameters") or _attr(text, "rope_scaling") or {}
    if isinstance(rope, dict) and "rope_theta" not in rope and isinstance(
            rope.get("full_attention") or rope.get("sliding_attention"), dict):
        rope = dict(rope.get("full_attention") or rope.get("sliding_attention"))
    theta = (rope.get("rope_theta") if isinstance(rope, dict) else None) or _attr(text, "rope_theta") or 10000.0
    rope_type = rope.get("rope_type", rope.get("type", "default")) if isinstance(rope, dict) else "default"
    # llama.cpp's eagle3 graph and vLLM's llama drafter both run plain (or llama3-scaled) RoPE.
    if rope_type in ("llama3", "linear"):
        rope_parameters = {k: v for k, v in rope.items() if k not in ("type", "mrope_section", "partial_rotary_factor")}
        rope_parameters["rope_theta"] = float(theta)
    else:
        rope_parameters = {"rope_type": "default", "rope_theta": float(theta)}
    if "rope_parameters" in inspect.signature(LlamaConfig.__init__).parameters:
        rope_kwargs = {"rope_parameters": rope_parameters}
    else:
        scaling = {k: v for k, v in rope_parameters.items() if k != "rope_theta"}
        rope_kwargs = {"rope_theta": float(theta), "rope_scaling": scaling if scaling.get("rope_type") != "default" else None}
    config = LlamaConfig(
        vocab_size = int(text.vocab_size),
        hidden_size = hidden,
        intermediate_size = int(intermediate),
        num_hidden_layers = 1,
        num_attention_heads = heads,
        num_key_value_heads = kv_heads,
        head_dim = head_dim,
        # llama.cpp's eagle3 FFN is SiLU-gated regardless of the target's activation.
        hidden_act = "silu",
        max_position_embeddings = int(_attr(text, "max_position_embeddings", 4096) or 4096),
        initializer_range = float(_attr(text, "initializer_range", 0.02) or 0.02),
        rms_norm_eps = float(_attr(text, "rms_norm_eps") or 1e-6),
        tie_word_embeddings = False,
        attention_bias = False,
        mlp_bias = False,
        **rope_kwargs,
    )
    config._attn_implementation = attn_implementation
    return config


class _Eagle3Layer(nn.Module):
    """speculators `LlamaDecoderEagle3FirstLayer`: q/k/v read 2H = [embeds; hidden]."""

    def __init__(self, config, norm_before_residual = False):
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

        base = LlamaDecoderLayer(config, 0)
        self.self_attn = base.self_attn
        self.mlp = base.mlp
        self.input_layernorm = base.input_layernorm
        self.post_attention_layernorm = base.post_attention_layernorm
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.norm_before_residual = bool(norm_before_residual)
        H, hd = config.hidden_size, config.head_dim
        self.self_attn.q_proj = nn.Linear(2 * H, config.num_attention_heads * hd, bias = config.attention_bias)
        self.self_attn.k_proj = nn.Linear(2 * H, config.num_key_value_heads * hd, bias = config.attention_bias)
        self.self_attn.v_proj = nn.Linear(2 * H, config.num_key_value_heads * hd, bias = config.attention_bias)
        import inspect

        params = inspect.signature(self.self_attn.forward).parameters
        self._cache_kw = "past_key_values" if "past_key_values" in params else "past_key_value"

    def forward(self, embeds, hidden, attention_mask, position_embeddings, past_key_values):
        residual = hidden
        embeds = self.input_layernorm(embeds)
        hidden = self.hidden_norm(hidden)
        if self.norm_before_residual:
            residual = hidden
        out, _ = self.self_attn(
            hidden_states = torch.cat([embeds, hidden], dim = -1),
            position_embeddings = position_embeddings,
            attention_mask = attention_mask,
            **{self._cache_kw: past_key_values},
        )
        hidden = residual + out
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


def _additive(allowed, dtype):
    mask = torch.zeros(allowed.shape, dtype = dtype, device = allowed.device)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)


class Eagle3Drafter(nn.Module):
    """One-layer EAGLE-3 drafter. The target's embedding matrix is shared (frozen, never saved);
    `lm_head` starts as the target's lm_head rows of the draft vocabulary and stays frozen
    (speculators `DraftVocabMixin`)."""

    def __init__(self, layer_config, draft_vocab_size, aux_layer_ids, target_hidden_size = None,
                 norm_before_residual = False):
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

        self.config = layer_config
        H = layer_config.hidden_size
        self.hidden_size = H
        self.target_hidden_size = int(target_hidden_size or H)
        self.aux_layer_ids = [int(i) for i in aux_layer_ids]
        if len(self.aux_layer_ids) != 3:
            raise ValueError("Unsloth: EAGLE-3 needs exactly 3 auxiliary target layers (llama.cpp requires 3).")
        self.draft_vocab_size = int(draft_vocab_size)
        self.target_vocab_size = int(layer_config.vocab_size)
        self.norm_before_residual = bool(norm_before_residual)
        self.fc = nn.Linear(3 * self.target_hidden_size, H, bias = False)
        self.layers = nn.ModuleList([_Eagle3Layer(layer_config, norm_before_residual)])
        self.norm = LlamaRMSNorm(H, eps = layer_config.rms_norm_eps)
        self.lm_head = nn.Linear(H, self.draft_vocab_size, bias = False)
        self.lm_head.weight.requires_grad_(False)
        self.register_buffer("d2t", torch.zeros(self.draft_vocab_size, dtype = torch.long))
        self.register_buffer("t2d", torch.ones(self.target_vocab_size, dtype = torch.bool)
                             if self.draft_vocab_size == self.target_vocab_size else
                             torch.zeros(self.target_vocab_size, dtype = torch.bool))
        self.rotary_emb = LlamaRotaryEmbedding(layer_config)
        object.__setattr__(self, "_embed_weight", None)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def target_layer_ids(self):
        return self.aux_layer_ids

    @property
    def uses_draft_vocab(self):
        return self.draft_vocab_size != self.target_vocab_size

    def set_embeddings(self, weight):
        """Share the target's (raw, unscaled) input embedding matrix; both runtimes do a plain lookup."""
        object.__setattr__(self, "_embed_weight", weight)

    def embed(self, ids):
        weight = self._embed_weight
        if weight is None:
            raise ValueError("Unsloth: call `set_embeddings(target_embedding_weight)` before running the drafter.")
        return F.embedding(ids.to(weight.device), weight).to(device = self.fc.weight.device, dtype = self.fc.weight.dtype)

    def draft_to_target(self, draft_ids):
        return draft_ids + self.d2t[draft_ids] if self.uses_draft_vocab else draft_ids

    def logits(self, hidden):
        return self.lm_head(self.norm(hidden))

    def fuse(self, aux_hidden):
        return self.fc(aux_hidden.to(self.fc.weight.dtype))

    def _run(self, embeds, hidden, mask, positions, cache):
        position_embeddings = self.rotary_emb(hidden, positions)
        return self.layers[0](embeds, hidden, mask, position_embeddings, cache)

    def forward(self, aux_hidden, input_ids, attention_mask = None, ttt_steps = 3, return_hidden = False,
                *, loss_kwargs = None):
        out = self.forward_train(aux_hidden, input_ids, attention_mask, ttt_steps,
                                 return_hidden = return_hidden or loss_kwargs is not None)
        if loss_kwargs is not None:
            # DDP's unused-parameter search must see norm.weight in the returned loss graph.
            return eagle3_chunked_loss(self, out, **loss_kwargs)
        return out

    def forward_train(self, aux_hidden, input_ids, attention_mask = None, ttt_steps = 3, return_hidden = False):
        """Training-time test over full sequences (speculators `Eagle3DraftModel.forward` after
        `shift_batch`). aux_hidden [B, S, 3Ht] from the target pass over input_ids [B, S].
        Returns per-step logits [B, S-1, Vd]; step k row t predicts x_{t+2+k}."""
        from transformers import DynamicCache

        B, S = input_ids.shape
        T = S - 1
        device = input_ids.device
        dtype = self.fc.weight.dtype
        valid = torch.ones(B, T, dtype = torch.bool, device = device) if attention_mask is None \
            else attention_mask[:, 1:].bool().to(device)
        hidden = self.fuse(aux_hidden[:, :-1])
        ids = input_ids[:, 1:]
        positions = torch.arange(1, S, device = device).unsqueeze(0).expand(B, -1)
        idx = torch.arange(T, device = device)
        causal = (idx[None, :, None] >= idx[None, None, :]) & valid[:, :, None] & valid[:, None, :]
        diag = (idx[:, None] == idx[None, :]).unsqueeze(0).expand(B, -1, -1)
        allowed = causal | diag
        cache = DynamicCache()
        out = []
        for k in range(ttt_steps):
            if k > 0:
                ids = torch.cat([input_ids[:, 1 + k:], input_ids.new_zeros(B, k)], dim = 1)
                allowed = torch.cat([allowed, diag], dim = -1)
            mask = _additive(allowed.unsqueeze(1), dtype)
            hidden = self._run(self.embed(ids), hidden, mask, positions + k, cache)
            out.append(hidden if return_hidden else self.logits(hidden))
        return out

    @torch.no_grad()
    def draft_greedy(self, aux_hidden, input_ids, anchor, num_steps):
        """Greedy drafts (target ids) after the committed tokens input_ids[:, :anchor + 1]; the
        target has produced aux states for positions < anchor + 1 (row anchor - 1 sees x_anchor)."""
        from transformers import DynamicCache

        c = int(anchor)
        cache = DynamicCache()
        hidden = self.fuse(aux_hidden[:, :c])
        ids = input_ids[:, 1:c + 1]
        positions = torch.arange(1, c + 1, device = input_ids.device).unsqueeze(0)
        idx = torch.arange(c, device = input_ids.device)
        mask = _additive((idx[:, None] >= idx[None, :])[None, None], hidden.dtype)
        h = self._run(self.embed(ids), hidden, mask, positions, cache)[:, -1:]
        drafts = []
        for m in range(num_steps):
            token = self.draft_to_target(self.logits(h).argmax(-1))
            drafts.append(token)
            if m + 1 == num_steps:
                break
            pos = torch.full((1, 1), c + 1 + m, device = input_ids.device, dtype = torch.long)
            h = self._run(self.embed(token), h, None, pos, cache)
        return torch.cat(drafts, dim = 1)

    def target_logits(self, final_hidden, lm_head_weight):
        """Teacher logits over the draft vocabulary from the target's post-norm final hidden."""
        weight = lm_head_weight[self.t2d.to(lm_head_weight.device)] if self.uses_draft_vocab else lm_head_weight
        return F.linear(final_hidden.to(weight.dtype), weight)


def _aux_capture_ids(aux_layer_ids, num_layers):
    """Aux id k (input of layer k) is the output of decoder layer k - 1."""
    ids = []
    for k in aux_layer_ids:
        # llama.cpp reads id == n_layer from the nextn output, not a layer input (common/speculative.cpp).
        if not 1 <= k < num_layers:
            raise ValueError(f"Unsloth: EAGLE-3 aux layer id {k} must be in [1, {num_layers - 1}].")
        ids.append(k - 1)
    return ids


def aux_hidden_states(captured, aux_layer_ids, device = None):
    # A target split over GPUs (device_map) leaves each captured layer on its own device.
    device = device or captured[aux_layer_ids[0] - 1].device
    return torch.cat([captured[k - 1].to(device) for k in aux_layer_ids], dim = -1)


def _default_aux_ids(num_layers):
    ids = [2, num_layers // 2, num_layers - 3]
    if len(set(ids)) != 3 or min(ids) < 1 or max(ids) > num_layers:
        raise ValueError(
            f"Unsloth: default EAGLE-3 aux layers {ids} are invalid for a {num_layers}-layer target; "
            "pass aux_layer_ids explicitly.")
    return ids


def build_eagle3_drafter(target_model, d2t, t2d, aux_layer_ids = None, dtype = None, device = None,
                         norm_before_residual = False):
    parts = resolve_text_parts(target_model)
    text = parts["config"]
    supported, reason = eagle3_supported(text)
    if not supported:
        raise NotImplementedError(f"Unsloth: EAGLE-3 is not supported for this model: {reason}.")
    num_layers = int(text.num_hidden_layers)
    aux = [int(i) for i in (aux_layer_ids or _default_aux_ids(num_layers))]
    _aux_capture_ids(aux, num_layers)
    layer_config = _draft_layer_config(text)
    t2d = t2d.bool()
    drafter = Eagle3Drafter(layer_config, int(t2d.sum()), aux, target_hidden_size = text.hidden_size,
                            norm_before_residual = norm_before_residual)
    lm_weight = effective_lm_head_weight(parts["lm_head"])
    with torch.no_grad():
        drafter.d2t.copy_(d2t.long())
        drafter.t2d.copy_(t2d)
        rows = lm_weight[t2d.to(lm_weight.device)] if drafter.uses_draft_vocab else lm_weight
        drafter.lm_head.weight.copy_(rows.detach().float().to(drafter.lm_head.weight.device))
    if device is None:
        device = lm_weight.device
    drafter.to(device = device, dtype = dtype or torch.float32)
    drafter.set_embeddings(parts["embed_tokens"].weight)
    return drafter


def _speculators_config(drafter, verifier_name_or_path, speculative_tokens):
    layer = drafter.config.to_dict()
    for key in ("_attn_implementation_autoset", "torch_dtype", "dtype", "transformers_version",
                "architectures", "_name_or_path"):
        layer.pop(key, None)
    layer["model_type"] = "llama"
    layer["num_hidden_layers"] = 1
    return {
        "architectures": ["Eagle3Speculator"],
        "speculators_model_type": "eagle3",
        "speculators_version": "0.3.0",
        "transformer_layer_config": layer,
        "draft_vocab_size": drafter.draft_vocab_size,
        "target_hidden_size": drafter.target_hidden_size,
        "eagle_aux_hidden_state_layer_ids": list(drafter.aux_layer_ids),
        # vLLM defaults a missing norm_before_residual to True: always write it.
        "norm_before_residual": drafter.norm_before_residual,
        "norm_before_fc": False,
        "fc_norm": False,
        "norm_output": False,
        "embed_requires_grad": False,
        "has_no_defaults_at_init": False,
        "speculators_config": {
            "algorithm": "eagle3",
            "default_proposal_method": "greedy",
            "proposal_methods": [{"proposal_type": "greedy", "speculative_tokens": int(speculative_tokens),
                                  "verifier_accept_k": 1, "accept_tolerance": 0.0}],
            "verifier": {"name_or_path": verifier_name_or_path, "architectures": []},
        },
    }


def save_eagle3_drafter(directory, drafter, verifier_name_or_path, speculative_tokens = 3, dtype = torch.bfloat16):
    """Write the drafter in the speculators layout (config.json + model.safetensors)."""
    from safetensors.torch import save_file

    os.makedirs(directory, exist_ok = True)
    state = {}
    for name, tensor in drafter.state_dict().items():
        if name.startswith("rotary_emb."):
            continue
        if name in ("d2t", "t2d"):
            if drafter.uses_draft_vocab:
                state[name] = tensor.detach().cpu().contiguous()
            continue
        state[name] = tensor.detach().to(dtype).cpu().contiguous()
    save_file(state, os.path.join(directory, "model.safetensors"), metadata = {"format": "pt"})
    config = _speculators_config(drafter, verifier_name_or_path, speculative_tokens)
    with open(os.path.join(directory, "config.json"), "w", encoding = "utf-8") as f:
        json.dump(config, f, indent = 2)
    return directory


def load_eagle3_drafter(directory, target_model, dtype = None, device = None):
    from safetensors.torch import load_file
    from transformers import LlamaConfig

    with open(os.path.join(directory, "config.json"), encoding = "utf-8") as f:
        config = json.load(f)
    if config.get("speculators_model_type") != "eagle3":
        raise ValueError(f"Unsloth: {directory} is not a speculators EAGLE-3 checkpoint.")
    layer = dict(config["transformer_layer_config"])
    layer.pop("model_type", None)
    layer_config = LlamaConfig(**layer)
    layer_config._attn_implementation = "sdpa"
    drafter = Eagle3Drafter(layer_config, config["draft_vocab_size"], config["eagle_aux_hidden_state_layer_ids"],
                            target_hidden_size = config.get("target_hidden_size"),
                            norm_before_residual = config.get("norm_before_residual", False))
    state = load_file(os.path.join(directory, "model.safetensors"))
    missing, unexpected = drafter.load_state_dict(state, strict = False)
    missing = [m for m in missing if not m.startswith("rotary_emb.") and m not in ("d2t", "t2d")]
    if missing or unexpected:
        raise ValueError(f"Unsloth: EAGLE-3 checkpoint mismatch (missing {missing[:5]}, unexpected {unexpected[:5]}).")
    if not drafter.uses_draft_vocab:
        drafter.t2d.fill_(True)
    drafter.lm_head.weight.requires_grad_(False)
    parts = resolve_text_parts(target_model)
    if device is None:
        device = parts["lm_head"].weight.device
    drafter.to(device = device, dtype = dtype or torch.float32)
    drafter.set_embeddings(parts["embed_tokens"].weight)
    return drafter


def _converter_path():
    """The converter Unsloth's GGUF export uses (pinned / staged unslothai/llama.cpp revision)."""
    from unsloth_zoo.llama_cpp import _download_convert_hf_to_gguf
    return _download_convert_hf_to_gguf()[0]


def convert_eagle3_to_gguf(drafter_dir, target_dir, out_path, outtype = "f16", converter = None, python = None):
    """Convert a speculators EAGLE-3 drafter with llama.cpp (needs the target dir for the tokenizer)."""
    converter = converter or _converter_path()
    cmd = [python or sys.executable, converter, str(drafter_dir), "--target-model-dir", str(target_dir),
           "--outtype", outtype, "--outfile", str(out_path)]
    result = subprocess.run(cmd, capture_output = True, text = True)
    if result.returncode != 0 or not os.path.isfile(out_path):
        raise RuntimeError(f"Unsloth: EAGLE-3 GGUF conversion failed:\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}")
    return out_path


def eagle3_loss(step_logits, teacher_logits, loss_mask, decay = 1.0):
    """Sum over TTT steps of masked mean forward KL(teacher || draft) (speculators `compute_metrics`
    with kl_div). Returns (loss, stats with per-step loss / top-1 accuracy / counts)."""
    total = 0.0
    stats = {"loss": [], "acc": [], "count": []}
    for k, logits in enumerate(step_logits):
        T = logits.shape[1]
        s_logits = logits[:, :T - k] if k else logits
        s_teacher = teacher_logits[:, k:]
        s_mask = loss_mask[:, k:].to(torch.float32)
        logp = F.log_softmax(s_logits.float(), dim = -1)
        p = F.softmax(s_teacher.float(), dim = -1)
        kl = F.kl_div(logp, p, reduction = "none").sum(-1)
        denom = s_mask.sum() + 1e-5
        step_loss = (kl * s_mask).sum() / denom
        total = total + (decay ** k) * step_loss
        with torch.no_grad():
            hit = (s_logits.argmax(-1) == s_teacher.argmax(-1)).float() * s_mask
            stats["loss"].append(float(step_loss.detach()))
            stats["acc"].append(float(hit.sum() / denom))
            stats["count"].append(float(s_mask.sum()))
    return total, stats


def eagle3_chunked_loss(drafter, step_hidden, final_hidden, lm_head_weight, loss_mask, decay = 1.0,
                        chunk_rows = 1024, normalizer = None, teacher_logit_softcap = None):
    """`eagle3_loss` from the drafter's hidden states, on supervised rows only, in checkpointed chunks:
    draft and teacher logits never exist for the whole batch ([B, T, Vd] fp32 per step OOMs 24 GB cards).
    `normalizer` (supervised tokens in the whole accumulated batch) replaces the per-microbatch count,
    so gradient accumulation weights every token equally."""
    from torch.utils.checkpoint import checkpoint

    weight = lm_head_weight[drafter.t2d.to(lm_head_weight.device)] if drafter.uses_draft_vocab else lm_head_weight
    # A target sharded over GPUs keeps its lm_head on another device than the drafter.
    weight = weight.to(drafter.fc.weight.device)

    def chunk_kl(h, th):
        logp = F.log_softmax(drafter.logits(h).float(), dim = -1)
        with torch.no_grad():
            teacher = F.linear(th.to(weight.dtype), weight).to(logp.dtype)
            if teacher_logit_softcap is not None:
                teacher = teacher_logit_softcap * torch.tanh(teacher / teacher_logit_softcap)
            p = F.softmax(teacher.float(), dim = -1)
            hits = (logp.argmax(-1) == teacher.argmax(-1)).sum()
        return F.kl_div(logp, p, reduction = "sum"), hits

    total = 0.0
    stats = {"loss": [], "acc": [], "count": []}
    for k, hidden in enumerate(step_hidden):
        T = hidden.shape[1]
        h = hidden[:, :T - k] if k else hidden
        th = final_hidden[:, k:]
        mask = loss_mask[:, k:].bool()
        rows, trows = h[mask], th[mask]
        n = int(rows.shape[0])
        kl_sum = rows.new_zeros((), dtype = torch.float32) + 0.0 * hidden.sum()
        hit_sum = 0
        for start in range(0, n, chunk_rows):
            kl, hits = checkpoint(chunk_kl, rows[start:start + chunk_rows], trows[start:start + chunk_rows],
                                  use_reentrant = False)
            kl_sum = kl_sum + kl
            hit_sum += int(hits)
        denom = n + 1e-5
        step_loss = kl_sum / denom
        total = total + (decay ** k) * (step_loss if normalizer is None else kl_sum / normalizer)
        stats["loss"].append(float(step_loss.detach()))
        stats["acc"].append(hit_sum / denom)
        stats["count"].append(float(n))
    return total, stats


@dataclass
class Eagle3TrainingArguments(TrainingArguments):
    eagle3_ttt_steps: int = field(default = 3, metadata = {"help": "Training-time-test steps unrolled per batch."})
    eagle3_step_decay: float = field(default = 1.0, metadata = {"help": "Loss weight decay per TTT step."})
    remove_unused_columns: bool = False


class _SupervisedRowLabels:
    """Labels that count only what the loss supervises (row t is scored by labels[t + 1], so never
    position 0 or padding): every transformers version counts num_items_in_batch from raw labels."""

    def __init__(self, collate):
        self.collate = collate

    def __call__(self, features):
        batch = self.collate(features)
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.clone()
            labels[:, 0] = -100
            mask = batch.get("attention_mask")
            if mask is not None:
                labels = labels.masked_fill(~mask.bool(), -100)
            batch["labels"] = labels
        return batch


class UnslothEagle3Trainer(Trainer):
    """Trains an EAGLE-3 drafter (`model`) against a frozen target, one target pass per batch.

    Rows need `input_ids` and optionally `labels` (-100 = not supervised, e.g. the prompt).
    """

    def __init__(self, model = None, args = None, target_model = None, **kwargs):
        if target_model is None:
            raise ValueError("Unsloth: UnslothEagle3Trainer needs `target_model`.")
        if args is None:
            args = Eagle3TrainingArguments(output_dir = "eagle3_outputs")
        if not isinstance(args, Eagle3TrainingArguments):
            raise TypeError("Unsloth: UnslothEagle3Trainer needs Eagle3TrainingArguments.")
        self.target_model = target_model
        for p in target_model.parameters():
            p.requires_grad_(False)
        target_model.eval()
        parts = resolve_text_parts(target_model)
        self._lm_head_weight = effective_lm_head_weight(parts["lm_head"])
        self._teacher_logit_softcap = getattr(parts["config"], "final_logit_softcapping", None)
        self._num_layers = int(parts["config"].num_hidden_layers)
        model.lm_head.weight.requires_grad_(False)
        if model._embed_weight is None:
            model.set_embeddings(parts["embed_tokens"].weight)
        if kwargs.get("data_collator") is None:
            from .trainer import MTPDataCollator
            tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
            tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
            pad = getattr(tokenizer, "pad_token_id", None)
            if pad is None:
                pad = getattr(tokenizer, "eos_token_id", 0) or 0
            kwargs["data_collator"] = MTPDataCollator(pad)
        kwargs["data_collator"] = _SupervisedRowLabels(kwargs["data_collator"])
        super().__init__(model = model, args = args, **kwargs)
        # compute_loss normalizes by num_items_in_batch itself, so the Trainer must not also divide by GA.
        self.model_accepts_loss_kwargs = True
        self._eagle3_log = []

    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels", input_ids)
        if attention_mask is not None and bool((attention_mask[:, 0] == 0).any()):
            raise ValueError("Unsloth: UnslothEagle3Trainer needs right-padded batches.")
        drafter = self.accelerator.unwrap_model(model)
        ids = _aux_capture_ids(drafter.aux_layer_ids, self._num_layers)
        captured = run_frozen_target(self.target_model, input_ids, attention_mask = attention_mask, layer_ids = ids)
        device = drafter.fc.weight.device
        aux = aux_hidden_states(captured, drafter.aux_layer_ids, device = device)
        captured["final"] = captured["final"].to(device)
        # speculators `shift_batch` pairs row t = (g_t, x_{t+1}) with loss_mask[t + 1].
        loss_mask = (labels[:, 1:] != -100)
        if attention_mask is not None:
            loss_mask = loss_mask & attention_mask[:, 1:].bool()
        normalizer = None
        if num_items_in_batch is not None:
            normalizer = torch.as_tensor(num_items_in_batch, device = loss_mask.device, dtype = torch.float32).clamp(min = 1)
        loss_kwargs = dict(final_hidden = captured["final"][:, 1:], lm_head_weight = self._lm_head_weight,
                           loss_mask = loss_mask, decay = self.args.eagle3_step_decay, normalizer = normalizer,
                           teacher_logit_softcap = self._teacher_logit_softcap)
        if isinstance(model, nn.DataParallel):
            # DataParallel scatters forward kwargs along dim 0 and cannot gather Python stats: score after the gather.
            step_hidden = model(aux, input_ids, attention_mask, self.args.eagle3_ttt_steps, return_hidden = True)
            loss, stats = eagle3_chunked_loss(drafter, step_hidden, **loss_kwargs)
        else:
            loss, stats = model(aux, input_ids, attention_mask, self.args.eagle3_ttt_steps, loss_kwargs = loss_kwargs)
        if model.training:
            self._eagle3_log.append(stats)
        loss = loss * token_average_scale(self, num_items_in_batch)
        return (loss, None) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)

    def log(self, logs, *args, **kwargs):
        rows = [s for s in self._eagle3_log if s["count"] and s["count"][0]]
        if rows:
            for k in range(len(rows[0]["loss"])):
                n = sum(s["count"][k] for s in rows)
                if n:
                    logs[f"eagle3_loss_{k}"] = round(sum(s["loss"][k] * s["count"][k] for s in rows) / n, 4)
                    logs[f"eagle3_acc_{k}"] = round(sum(s["acc"][k] * s["count"][k] for s in rows) / n, 4)
        self._eagle3_log = []
        return super().log(logs, *args, **kwargs)

    def _save(self, output_dir = None, state_dict = None):
        output_dir = output_dir or self.args.output_dir
        verifier = getattr(getattr(self.target_model, "config", None), "_name_or_path", None)
        save_eagle3_drafter(output_dir, self.model, verifier, self.args.eagle3_ttt_steps,
                            dtype = next(self.model.parameters()).dtype)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

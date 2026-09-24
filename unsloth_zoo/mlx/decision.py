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

"""Typed-decision models (ModernBERT encoder + decision head) on MLX, inference only.

Reads the Laya checkpoint layout as is and returns the decision logits of laya's torch
`DecisionModel`; prompt building and calibration stay with the `laya` package."""

import json
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

__all__ = [
    "DecisionModel",
    "load_decision_model",
]


def _rope_base(config, layer_type):
    params = (config.get("rope_parameters") or {}).get(layer_type) or {}
    if params.get("rope_type", "default") != "default":
        raise ValueError(f"Unsupported ModernBERT RoPE type {params['rope_type']!r}")
    if layer_type == "sliding_attention":
        fallback = config.get("local_rope_theta", 10000.0)
    else:
        fallback = config.get("global_rope_theta", 160000.0)
    return float(params.get("rope_theta", fallback))


@partial(mx.compile, shapeless = True)
def _gelu_gate(value, gate):
    return nn.gelu(value) * gate


def _gather_rows(x, rows):
    return mx.take_along_axis(x, mx.broadcast_to(rows[:, :, None], (*rows.shape, x.shape[-1])), axis = 1)


class _EncoderAttention(nn.Module):
    def __init__(self, dims, heads, bias, base):
        super().__init__()
        self.heads = heads
        self.base = base
        self.Wqkv = nn.Linear(dims, 3 * dims, bias = bias)
        self.Wo = nn.Linear(dims, dims, bias = bias)

    def __call__(self, x, mask):
        B, L, D = x.shape
        qkv = self.Wqkv(x).reshape(B, L, 3, self.heads, -1).transpose(2, 0, 3, 1, 4)
        head_dim = qkv.shape[-1]
        q, k = (
            mx.fast.rope(t, head_dim, traditional = False, base = self.base, scale = 1.0, offset = 0)
            for t in (qkv[0], qkv[1])
        )
        out = mx.fast.scaled_dot_product_attention(q, k, qkv[2], scale = head_dim**-0.5, mask = mask)
        return self.Wo(out.transpose(0, 2, 1, 3).reshape(B, L, D))


class _EncoderMLP(nn.Module):
    def __init__(self, dims, hidden, bias):
        super().__init__()
        self.Wi = nn.Linear(dims, 2 * hidden, bias = bias)
        self.Wo = nn.Linear(hidden, dims, bias = bias)

    def __call__(self, x):
        value, gate = mx.split(self.Wi(x), 2, axis = -1)
        return self.Wo(_gelu_gate(value, gate))


class _EncoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        dims, eps, norm_bias = config["hidden_size"], config.get("norm_eps", 1e-5), config.get("norm_bias", False)
        self.layer_type = config["layer_types"][index]
        self.attn_norm = nn.Identity() if index == 0 else nn.LayerNorm(dims, eps = eps, bias = norm_bias)
        self.attn = _EncoderAttention(
            dims, config["num_attention_heads"], config.get("attention_bias", False), _rope_base(config, self.layer_type)
        )
        self.mlp_norm = nn.LayerNorm(dims, eps = eps, bias = norm_bias)
        self.mlp = _EncoderMLP(dims, config["intermediate_size"], config.get("mlp_bias", False))

    def __call__(self, x, mask):
        x = x + self.attn(self.attn_norm(x), mask)
        return x + self.mlp(self.mlp_norm(x))


class _Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.norm = nn.LayerNorm(config["hidden_size"], eps = config.get("norm_eps", 1e-5), bias = config.get("norm_bias", False))

    def __call__(self, input_ids):
        return self.norm(self.tok_embeddings(input_ids))


class _Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.window = config.get("local_attention", 128) // 2
        self.embeddings = _Embeddings(config)
        self.layers = [_EncoderLayer(config, i) for i in range(config["num_hidden_layers"])]
        self.final_norm = nn.LayerNorm(
            config["hidden_size"], eps = config.get("norm_eps", 1e-5), bias = config.get("norm_bias", False)
        )

    def __call__(self, input_ids, keys):
        positions = mx.arange(input_ids.shape[1])
        near = mx.abs(positions[:, None] - positions[None, :]) <= self.window
        masks = {"full_attention": keys, "sliding_attention": keys & near}
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, masks[layer.layer_type])
        return self.final_norm(x)


class _HeadAttention(nn.Module):
    # torch.nn.MultiheadAttention: fused biased in-projection, one output projection.
    def __init__(self, dims, heads):
        super().__init__()
        self.heads = heads
        self.in_proj = nn.Linear(dims, 3 * dims)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(self, x, mask, rows = None):
        B, _, D = x.shape
        q, k, v = mx.split(self.in_proj(x), 3, axis = -1)
        if rows is not None:
            q = _gather_rows(q, rows)
        q, k, v = (t.reshape(B, t.shape[1], self.heads, -1).transpose(0, 2, 1, 3) for t in (q, k, v))
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale = q.shape[-1]**-0.5, mask = mask)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(B, -1, D))


class _HeadLayer(nn.Module):
    # torch.nn.TransformerEncoderLayer(norm_first = True) with its default ReLU feed-forward.
    def __init__(self, dims, heads):
        super().__init__()
        self.self_attn = _HeadAttention(dims, heads)
        self.norm1 = nn.LayerNorm(dims)
        self.norm2 = nn.LayerNorm(dims)
        self.linear1 = nn.Linear(dims, 4 * dims)
        self.linear2 = nn.Linear(4 * dims, dims)

    def __call__(self, x, mask, rows = None):
        # With `rows`, keys and values still span every token; attention output and feed-forward cover only those rows.
        attended = self.self_attn(self.norm1(x), mask, rows)
        x = (x if rows is None else _gather_rows(x, rows)) + attended
        return x + self.linear2(nn.relu(self.linear1(self.norm2(x))))


class _Head(nn.Module):
    def __init__(self, dims, count):
        super().__init__()
        self.layers = [_HeadLayer(dims, max(1, dims // 64)) for _ in range(count)]


class DecisionModel(nn.Module):
    """Upstream `DecisionModel` without the act head. Parameter names follow the checkpoint."""

    def __init__(self, encoder_config, head_layers):
        super().__init__()
        dims = encoder_config["hidden_size"]
        self.encoder = _Encoder(encoder_config)
        self.head = _Head(dims, head_layers)
        self.type_emb = nn.Embedding(3, dims)
        self.scorer = [nn.LayerNorm(dims), nn.Linear(dims, dims), nn.GELU(), nn.Linear(dims, 1)]

    def __call__(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        keys = attention_mask.astype(mx.bool_)[:, None, None, :]
        h = self.encoder(input_ids, keys) + self.type_emb(qtype)[:, None, :]
        layers = self.head.layers
        for layer in layers[:-1]:
            h = layer(h, keys)
        # Only the marker rows are scored, so the last head layer answers for those rows alone.
        x = layers[-1](h, keys, marker_pos) if layers else _gather_rows(h, marker_pos)
        for layer in self.scorer:
            x = layer(x)
        return mx.where(marker_mask, x.squeeze(-1).astype(mx.float32), -1e4)

    def logits(self, batch):
        """Decision logits for a collated batch of numpy or torch arrays, as float32 numpy."""
        arrays = {k: mx.array(np.asarray(batch[k])) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}
        out = self(**arrays)
        mx.eval(out)
        return np.array(out)


def _checkpoint_name(name):
    return name.replace(".in_proj_weight", ".in_proj.weight").replace(".in_proj_bias", ".in_proj.bias")


def load_decision_model(folder, dtype = mx.float32):
    """Load a Laya checkpoint folder (`encoder/config.json`, `rl_agent_config.json`, `model.safetensors`).

    float32 matches laya's torch CPU logits to ~1e-4; float16 drifts calibrated probabilities by up to ~1e-2."""
    folder = Path(folder)
    encoder_config = json.loads((folder / "encoder" / "config.json").read_text())
    agent_config = json.loads((folder / "rl_agent_config.json").read_text())
    if encoder_config.get("model_type") != "modernbert":
        raise ValueError(f"The decision model needs a ModernBERT encoder, got {encoder_config.get('model_type')!r}")
    if encoder_config.get("hidden_activation", "gelu") != "gelu":
        raise ValueError(f"Unsupported ModernBERT activation {encoder_config['hidden_activation']!r}")
    every = encoder_config.get("global_attn_every_n_layers", 3)
    encoder_config.setdefault("layer_types", [
        "full_attention" if i % every == 0 else "sliding_attention"
        for i in range(encoder_config["num_hidden_layers"])
    ])
    model = DecisionModel(encoder_config, agent_config.get("head_layers", 2))
    # The act head is not served, and calibration is read from rl_agent_config.json, not this buffer.
    weights = [
        (_checkpoint_name(name), value.astype(dtype))
        for name, value in mx.load(str(folder / "model.safetensors")).items()
        if not name.startswith("act_head.") and name != "temperature"
    ]
    model.load_weights(weights, strict = True)
    model.eval()
    mx.eval(model.parameters())
    return model

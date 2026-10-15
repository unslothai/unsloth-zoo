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

"""Native MTP heads, trainable outside the target model.

Qwen3.5 / Qwen3.6 use the hand-written Qwen35MTPHead below; every other family in
mtp_families (GLM, DeepSeek, Qwen3-Next, Nemotron-H, Step, ...) uses nextn.NextNHead.

The forward follows llama.cpp's `graph_mtp` (src/models/qwen35.cpp) and its
`draft-mtp` driver (common/speculative.cpp): MTP position p consumes the target's
post-`output_norm` hidden of position p-1 and the token at p, and predicts the
token at p+1. Its KV cache is filled over every target position with those
pairs, and recursive draft steps feed the head's own normed output back as h.
"""

__all__ = [
    "MTP_PREFIX_RE",
    "mtp_family_name",
    "find_mtp_tensor_names",
    "load_mtp_state_dict",
    "Qwen35MTPHead",
    "build_mtp_head",
]

import json
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Qwen3.5 writes `mtp.*`; nested text wrappers prefix `model.` / `language_model.`.
MTP_PREFIX_RE = re.compile(r"^((?:(?:model|language_model)\.)*mtp\.)")
MTP_CONFIG_KEY = "mtp_num_hidden_layers"
_SUPPORTED_MODEL_TYPES = ("qwen3_5_text",)
_MOE_MODEL_TYPES = ("qwen3_5_moe_text",)


def _weight_map(source, revision = None):
    """{tensor name: shard file} for a local checkpoint dir or a Hub repo, headers only."""
    path = Path(str(source))
    if path.is_dir():
        index = path / "model.safetensors.index.json"
        if index.is_file():
            with index.open("r", encoding = "utf-8") as f:
                return dict(json.load(f)["weight_map"]), path
        from safetensors import safe_open
        weight_map = {}
        for shard in sorted(path.glob("*.safetensors")):
            with safe_open(str(shard), framework = "pt", device = "cpu") as f:
                for name in f.keys():
                    weight_map[name] = shard.name
        return weight_map, path
    from huggingface_hub import get_safetensors_metadata
    metadata = get_safetensors_metadata(str(source), revision = revision)
    weight_map = dict(metadata.weight_map)
    return weight_map, None


def _source_text_config(source, revision = None):
    """The text config of a local folder or Hub repo, or None when it cannot be read."""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(str(source), revision = revision)
    except Exception:
        return None
    return config.get_text_config() if hasattr(config, "get_text_config") else config


def _nextn_family(config):
    if config is None or getattr(config, "model_type", None) in _SUPPORTED_MODEL_TYPES + _MOE_MODEL_TYPES:
        return None
    from .mtp_families import mtp_family_for_config
    return mtp_family_for_config(config)


def mtp_family_name(config):
    """`"qwen3_5"` for Qwen3.5 / 3.6, the NextN family name, or None when unsupported."""
    config = _text_config(config) if config is not None else None
    if getattr(config, "model_type", None) in _SUPPORTED_MODEL_TYPES + _MOE_MODEL_TYPES:
        return "qwen3_5"
    family = _nextn_family(config)
    return family.name if family is not None else None


def find_mtp_tensor_names(source, revision = None, config = None):
    """{full tensor name: shard file} for every MTP tensor in the checkpoint. `config` (the
    target's) picks the family; read from `source` when omitted."""
    weight_map, _ = _weight_map(source, revision = revision)
    text = _text_config(config) if config is not None else _source_text_config(source, revision)
    family = _nextn_family(text)
    if family is not None:
        n = int(text.num_hidden_layers)
        return {name: shard for name, shard in weight_map.items() if family.is_mtp_name(name, n)}
    return {name: shard for name, shard in weight_map.items() if MTP_PREFIX_RE.match(name)}


def _declared_mtp_layers(source, revision = None):
    path = Path(str(source))
    try:
        if path.is_dir():
            config_path = path / "config.json"
        else:
            from huggingface_hub import hf_hub_download
            config_path = Path(hf_hub_download(str(source), "config.json", revision = revision))
        with open(config_path, "r", encoding = "utf-8") as f:
            config = json.load(f)
    except Exception:
        return None
    for holder in (config, config.get("text_config") or {}):
        if isinstance(holder, dict) and holder.get(MTP_CONFIG_KEY):
            return int(holder[MTP_CONFIG_KEY])
    return None


def load_mtp_state_dict(source, revision = None, config = None):
    """Read only the MTP tensors of a checkpoint (transformers drops them on load).

    Returns (state_dict, info). Qwen3.5 names are head-relative (`fc.weight`); other families
    keep full checkpoint names (info["prefix"] is then ""). Only the shards that hold the head
    are downloaded or opened. `config` (the target's) picks the family; read from `source`
    when omitted.
    """
    from safetensors import safe_open
    text = _text_config(config) if config is not None else _source_text_config(source, revision)
    family = _nextn_family(text)
    if family is not None:
        from .nextn import load_nextn_state_dict
        canonical, info = load_nextn_state_dict(source, text, revision = revision)
        mapping = info.pop("ckpt_names")
        state_dict = {mapping[k]: v for k, v in canonical.items()}
        info["ckpt_prefix"], info["prefix"] = info["prefix"], ""
        return state_dict, info
    names = find_mtp_tensor_names(source, revision = revision)
    if not names:
        declared = _declared_mtp_layers(source, revision = revision)
        if declared:
            raise ValueError(
                f"Unsloth: `{source}` declares `{MTP_CONFIG_KEY}={declared}` but its weights carry no "
                "`mtp.*` tensors, so there is no MTP head to fine-tune. Load the original (non "
                "pre-quantized) checkpoint, or build a randomly initialised head."
            )
        raise ValueError(f"Unsloth: `{source}` has no `mtp.*` tensors, so it has no native MTP head.")

    prefixes = {MTP_PREFIX_RE.match(name).group(1) for name in names}
    if len(prefixes) != 1:
        raise ValueError(f"Unsloth: `{source}` mixes MTP prefixes {sorted(prefixes)}.")
    prefix = prefixes.pop()

    path = Path(str(source))
    by_shard = {}
    for name, shard in names.items():
        by_shard.setdefault(shard, []).append(name)

    state_dict = {}
    files = []
    for shard, shard_names in sorted(by_shard.items()):
        if path.is_dir():
            shard_path = str(path / shard)
        else:
            from huggingface_hub import hf_hub_download
            shard_path = hf_hub_download(str(source), shard, revision = revision)
        files.append(shard_path)
        with safe_open(shard_path, framework = "pt", device = "cpu") as f:
            for name in shard_names:
                state_dict[name[len(prefix):]] = f.get_tensor(name)

    layer_ids = {
        int(m.group(1)) for m in (re.match(r"^layers\.(\d+)\.", k) for k in state_dict) if m
    }
    dtypes = {str(t.dtype) for t in state_dict.values()}
    info = {
        "source": str(source),
        "revision": revision,
        "prefix": prefix,
        "names": sorted(names),
        "files": files,
        "dtype": dtypes.pop() if len(dtypes) == 1 else sorted(dtypes),
        "shapes": {k: tuple(v.shape) for k, v in state_dict.items()},
        "num_layers": len(layer_ids),
    }
    return state_dict, info


def _text_config(target_model_or_config):
    config = getattr(target_model_or_config, "config", target_model_or_config)
    if hasattr(config, "get_text_config"):
        config = config.get_text_config()
    return config


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim = -1)


def _apply_rope(x, cos, sin):
    # x [B, heads, S, D]; cos/sin [B, S, rot]; partial rotary keeps the tail as is.
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rot = cos.shape[-1]
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    return torch.cat([x_rot * cos + _rotate_half(x_rot) * sin, x_pass], dim = -1)


class Qwen35MTPHead(nn.Module):
    """One native Qwen3.5 MTP block; `state_dict()` names equal the checkpoint's minus `mtp.`.

    The token embedding and lm_head are not owned: they are the target's, passed in frozen.
    """

    def __init__(self, config):
        super().__init__()
        import copy

        model_type = getattr(config, "model_type", None)
        if model_type in _MOE_MODEL_TYPES:
            # Same block as dense plus the routed + shared-expert MoE MLP (llama.cpp qwen35moe graph_mtp).
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                Qwen3_5MoeDecoderLayer as DecoderLayer,
                Qwen3_5MoeRMSNorm as RMSNorm,
                Qwen3_5MoeTextRotaryEmbedding as RotaryEmbedding,
            )
        elif model_type in _SUPPORTED_MODEL_TYPES:
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5DecoderLayer as DecoderLayer,
                Qwen3_5RMSNorm as RMSNorm,
                Qwen3_5TextRotaryEmbedding as RotaryEmbedding,
            )
        else:
            raise NotImplementedError(
                f"Unsloth: native MTP head training supports {_SUPPORTED_MODEL_TYPES + _MOE_MODEL_TYPES}, not `{model_type}`."
            )
        num_layers = int(getattr(config, MTP_CONFIG_KEY, 0) or 1)
        if num_layers != 1:
            # llama.cpp's qwen35 graph_mtp asserts n_layer_nextn == 1.
            raise NotImplementedError(f"Unsloth: Qwen3.5 MTP supports one MTP layer, got {num_layers}.")

        head_config = copy.deepcopy(config)
        head_config.num_hidden_layers = num_layers
        head_config.layer_types = ["full_attention"] * num_layers
        self.config = head_config
        hidden_size = head_config.hidden_size
        eps = head_config.rms_norm_eps

        self.fc = nn.Linear(2 * hidden_size, hidden_size, bias = False)
        self.pre_fc_norm_embedding = RMSNorm(hidden_size, eps = eps)
        self.pre_fc_norm_hidden = RMSNorm(hidden_size, eps = eps)
        self.layers = nn.ModuleList([DecoderLayer(head_config, i) for i in range(num_layers)])
        self.norm = RMSNorm(hidden_size, eps = eps)
        self.rotary_emb = RotaryEmbedding(head_config)
        # Ablation switch for alignment checks only; llama.cpp concatenates embedding first.
        self._embedding_first = True

    @property
    def layer(self):
        return self.layers[0]

    def _rope(self, x, positions):
        # Text tokens: every mrope section gets the same position, i.e. plain partial rope.
        return self.rotary_emb(x, positions[None].expand(3, -1, -1))

    def _project_inputs(self, token_embeds, hidden):
        e = self.pre_fc_norm_embedding(token_embeds)
        h = self.pre_fc_norm_hidden(hidden)
        pair = torch.cat([e, h] if self._embedding_first else [h, e], dim = -1)
        return self.fc(pair)

    def _qkv(self, residual, positions):
        attn = self.layer.self_attn
        x = self.layer.input_layernorm(residual)
        B, S, _ = x.shape
        D = attn.head_dim
        q, gate = torch.chunk(attn.q_proj(x).view(B, S, -1, D * 2), 2, dim = -1)
        gate = gate.reshape(B, S, -1)
        q = attn.q_norm(q).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).view(B, S, -1, D)).transpose(1, 2)
        v = attn.v_proj(x).view(B, S, -1, D).transpose(1, 2)
        cos, sin = self._rope(x, positions)
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin), v, gate

    def _attend(self, q, k, v, mask):
        # q [B, Hq, Sq, D], k/v [B, Hkv, Sk, D], mask bool [B, 1, Sq, Sk] (True = attend).
        attn = self.layer.self_attn
        groups = q.shape[1] // k.shape[1]
        if groups > 1:
            k = k.repeat_interleave(groups, dim = 1)
            v = v.repeat_interleave(groups, dim = 1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask = mask, scale = attn.scaling)
        return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)

    def _finish(self, attn_out, gate, residual):
        layer = self.layer
        h = residual + layer.self_attn.o_proj(attn_out * torch.sigmoid(gate))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        return self.norm(h)

    def forward(self, hidden_final, input_ids, embed_tokens, attention_mask = None, num_steps = 1, position_ids = None):
        # Wrappers such as DDP only sync gradients for calls that go through forward().
        return self.forward_train(
            hidden_final, input_ids, embed_tokens,
            attention_mask = attention_mask, num_steps = num_steps, position_ids = position_ids,
        )

    def forward_train(
        self,
        hidden_final,
        input_ids,
        embed_tokens,
        attention_mask = None,
        num_steps = 1,
        position_ids = None,
    ):
        """Teacher-forced recursive drafting over a whole sequence.

        Index p of every step is the chain whose first MTP input is
        (hidden_final[p-1], x_p) at position p. Step k of that chain sits at
        position p+k with token x_{p+k} and h = step k-1's output at p, and predicts
        x_{p+k+1} (= x_{t+k+2} with t = p-1). Step k attends to step 0's keys at
        positions <= p (llama.cpp fills those from the target's h) plus the chain's
        own keys of steps 1..k, exactly what the draft loop sees.

        Returns one dict per step: hidden (normed, pre lm_head), labels (-100 where
        undefined), positions, valid.
        """
        B, S = input_ids.shape
        device = input_ids.device
        if attention_mask is None:
            valid = torch.ones(B, S, dtype = torch.bool, device = device)
        else:
            valid = attention_mask.to(device = device).bool()
        if position_ids is None:
            position_ids = torch.arange(S, device = device).unsqueeze(0).expand(B, -1)

        hidden_final = hidden_final.to(self.fc.weight.dtype)
        h_prev = torch.cat([torch.zeros_like(hidden_final[:, :1]), hidden_final[:, :-1]], dim = 1)
        idx = torch.arange(S, device = device)
        causal = idx[None, :] <= idx[:, None]
        diagonal = idx[None, :] == idx[:, None]

        steps = []
        keys, values = [], []
        h_in = h_prev
        for k in range(num_steps):
            # Token x_{p+k}: shift left by k, out-of-range slots padded (their labels are -100).
            tokens = input_ids.new_zeros(B, S)
            tok_valid = torch.zeros_like(valid)
            label = input_ids.new_full((B, S), -100)
            if k < S:
                tokens[:, : S - k] = input_ids[:, k:]
                tok_valid[:, : S - k] = valid[:, k:]
            if k + 1 < S:
                label[:, : S - k - 1] = input_ids[:, k + 1 :]
                lab_valid = torch.zeros_like(valid)
                lab_valid[:, : S - k - 1] = valid[:, k + 1 :]
            else:
                lab_valid = torch.zeros_like(valid)
            step_valid = tok_valid & lab_valid & valid & (idx[None, :] >= 1)
            label = label.masked_fill(~step_valid, -100)
            positions = position_ids + k

            residual = self._project_inputs(embed_tokens(tokens).to(self.fc.weight.dtype), h_in)
            q, kk, vv, gate = self._qkv(residual, positions)
            keys.append(kk)
            values.append(vv)

            # Block 0: step-0 keys j <= p (plus self so padded rows never go all -inf).
            block0 = (causal[None] & valid[:, None, :]) | diagonal[None]
            blocks = [block0] + [diagonal[None].expand(B, -1, -1)] * k
            mask = torch.cat(blocks, dim = -1)[:, None]
            attn_out = self._attend(q, torch.cat(keys, dim = 2), torch.cat(values, dim = 2), mask)
            out = self._finish(attn_out, gate, residual)
            steps.append({"hidden": out, "labels": label, "positions": positions, "valid": step_valid})
            h_in = out
        return steps

    @staticmethod
    def new_cache():
        return {"k": None, "v": None}

    def draft(self, hidden, token_ids, positions, embed_tokens, cache):
        """Incremental decode of new MTP positions, appended to `cache` (llama.cpp's MTP KV).

        hidden [B, n, H] is the h paired with each token (target h of the previous
        position during catch-up, the head's own output while drafting). Returns
        the normed outputs [B, n, H].
        """
        residual = self._project_inputs(embed_tokens(token_ids).to(self.fc.weight.dtype), hidden.to(self.fc.weight.dtype))
        q, k, v, gate = self._qkv(residual, positions)
        cache["k"] = k if cache["k"] is None else torch.cat([cache["k"], k], dim = 2)
        cache["v"] = v if cache["v"] is None else torch.cat([cache["v"], v], dim = 2)
        n = token_ids.shape[1]
        total = cache["k"].shape[2]
        past = total - n
        qi = torch.arange(n, device = token_ids.device)[:, None] + past
        ki = torch.arange(total, device = token_ids.device)[None, :]
        mask = (ki <= qi)[None, None].expand(token_ids.shape[0], 1, -1, -1)
        return self._finish(self._attend(q, cache["k"], cache["v"], mask), gate, residual)

    @staticmethod
    def truncate_cache(cache, length):
        if cache["k"] is not None:
            cache["k"] = cache["k"][:, :, :length]
            cache["v"] = cache["v"][:, :, :length]
        return cache


def _init_unset_parameters(head, std):
    """MoE experts / router are created with torch.empty / zeros (transformers leaves them to
    _init_weights); give a random head finite, non-degenerate values."""
    for name, param in head.named_parameters():
        if name.endswith(("experts.gate_up_proj", "experts.down_proj", "mlp.gate.weight")):
            with torch.no_grad():
                param.normal_(0.0, std)


def build_mtp_head(target_model_or_config, state_dict = None, dtype = None, device = None, strict = True,
                   target_embed = None, target_lm_head = None, ckpt_prefix = None):
    """Build the MTP head for a target; load `state_dict` (as returned by load_mtp_state_dict).

    target_embed / target_lm_head: the target's matrices, so head-serialized copies equal to
    them are not held twice.
    """
    config = _text_config(target_model_or_config)
    family = _nextn_family(config)
    if family is not None:
        from .nextn import build_nextn_head
        canonical = None
        prefixes = set()
        if state_dict is not None:
            n = int(config.num_hidden_layers)
            canonical = {}
            for name, tensor in state_dict.items():
                if name.startswith("heads."):
                    canonical[name] = tensor
                    continue
                mapped = family.from_ckpt(name, n)
                if mapped is None:
                    raise ValueError(f"Unsloth: `{name}` is not a {family.name} MTP tensor.")
                prefixes.add(mapped[0])
                canonical[mapped[1]] = tensor
        head = build_nextn_head(
            config, state_dict = canonical, dtype = dtype, device = device, strict = strict, family = family,
            target_embed = target_embed, target_lm_head = target_lm_head,
        )
        # "" is a discovered prefix (bare `layers.N.*` names), not a missing one.
        if ckpt_prefix is None:
            ckpt_prefix = prefixes.pop() if len(prefixes) == 1 else family.default_prefix()
        head._ckpt_prefix = ckpt_prefix
        return head
    with torch.device("meta" if state_dict is not None else (device or "cpu")):
        head = Qwen35MTPHead(config)
    if state_dict is None:
        _init_unset_parameters(head, getattr(config, "initializer_range", 0.02))
    else:
        head = head.to_empty(device = device or "cpu")
        # to_empty leaves non-persistent rope buffers uninitialised.
        head.rotary_emb = type(head.rotary_emb)(head.config).to(device or "cpu")
        missing, unexpected = head.load_state_dict(state_dict, strict = False)
        if strict and (missing or unexpected):
            raise ValueError(
                f"Unsloth: MTP state_dict does not match the head: missing={missing}, unexpected={unexpected}"
            )
    if dtype is not None:
        # Keep the rope frequencies fp32: a bf16 round trip cannot be undone by .float().
        frequencies = {
            name: buffer.detach().clone().float()
            for name, buffer in head.rotary_emb.named_buffers(recurse = False) if "inv_freq" in name
        }
        head = head.to(dtype)
        for name, buffer in frequencies.items():
            setattr(head.rotary_emb, name, buffer)
    return head

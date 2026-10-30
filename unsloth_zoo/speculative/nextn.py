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

"""Generic NextN/MTP head built from the target's decoder layer class; teacher forcing follows llama.cpp's draft-mtp
driver (chain p starts from (h[p-1], x_p)); single-head families reuse head 0, chained families run head k at step k."""

__all__ = [
    "NextNHead",
    "build_nextn_head",
    "pack_experts",
    "unpack_experts",
]

import importlib
import inspect
import re

import torch
import torch.nn as nn

from .mtp_families import mtp_family_for_config

_PER_EXPERT = re.compile(r"^(.*experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
_PACKED = re.compile(r"^(.*experts)\.(gate_up_proj|up_proj|down_proj)$")


class _ChainKV:
    """Minimal KV store: the attention modules only call `update`."""

    is_compileable = False

    def __init__(self):
        self.keys = {}
        self.values = {}
        self.positions = None  # key positions [B, K], for sliding-window masks

    def add_positions(self, positions):
        self.positions = positions if self.positions is None else torch.cat([self.positions, positions], dim = 1)

    def update(self, key_states, value_states, layer_idx, cache_kwargs = None):
        if layer_idx in self.keys:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], key_states], dim = -2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], value_states], dim = -2)
        else:
            self.keys[layer_idx] = key_states
            self.values[layer_idx] = value_states
        return self.keys[layer_idx], self.values[layer_idx]

    def update_indexer(self, indexer_key_states, layer_idx):
        # DSA indexer keys [B, S, D] (transformers' DynamicCache.update_indexer contract).
        key = ("indexer", layer_idx)
        previous = self.keys.get(key)
        self.keys[key] = indexer_key_states if previous is None else torch.cat([previous, indexer_key_states], dim = 1)
        return self.keys[key]

    def get_seq_length(self, layer_idx = 0):
        attention = [v for k, v in self.keys.items() if not isinstance(k, tuple)]
        return attention[0].shape[-2] if attention else 0


def _window_mask(mask, positions, cache, window):
    """Restrict an additive mask to sliding-window keys (attend iff q_pos - k_pos < window, as HF and llama.cpp); key positions are tracked on the cache."""
    key_positions = positions if cache.positions is None else torch.cat([cache.positions, positions], dim = 1)
    distance = positions[:, :, None] - key_positions[:, None, :]
    outside = (distance >= window)[:, None]
    return mask.masked_fill(outside, torch.finfo(mask.dtype).min)


class _WeightOnlyLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = (x - x.mean(-1, keepdim = True)) * torch.rsqrt(x.var(-1, keepdim = True, unbiased = False) + self.eps)
        return (self.weight.float() * x).to(dtype)


def _module(family):
    return importlib.import_module(f"transformers.models.{family.module}.modeling_{family.module}")


def _eps(config):
    for key in ("rms_norm_eps", "layer_norm_epsilon", "layer_norm_eps", "norm_eps"):
        value = getattr(config, key, None)
        if value is not None:
            return float(value)
    return 1e-6


def _make_norm(norm_cls, config):
    try:
        return norm_cls(config.hidden_size, eps = _eps(config))
    except TypeError:
        return norm_cls(config.hidden_size, _eps(config))


class _NextNLayer(nn.Module):
    def __init__(self, family, config, head_index, layer_cls, norm_cls, num_hidden_layers):
        super().__init__()
        hidden = config.hidden_size
        self.enorm = _make_norm(norm_cls, config)
        self.hnorm = _make_norm(norm_cls, config)
        self.eh_proj = nn.Linear(2 * hidden, hidden, bias = False)
        self.block = nn.ModuleList([
            layer_cls(config, family.block_layer_idx(num_hidden_layers, head_index, j))
            for j in range(len(family.blocks))
        ])
        if family.head_norm == "layer":
            self.shared_head_norm = _WeightOnlyLayerNorm(hidden, _eps(config))
        else:
            self.shared_head_norm = _make_norm(norm_cls, config)
        # Head-serialized copies of the embedding / lm_head (frozen; llama.cpp prefers them).
        self.register_buffer("embed_weight", None, persistent = False)
        self.register_buffer("lm_head_weight", None, persistent = False)


def _call_kwargs(layer):
    params = inspect.signature(layer.forward).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return None
    return set(params)


class NextNHead(nn.Module):
    """Native MTP head(s) of a DeepSeek-V3 style family; state_dict() uses canonical names, mtp_state_dict() the
    checkpoint's; embedding and lm_head are the target's or frozen copies."""

    def __init__(self, config, family = None):
        super().__init__()
        family = family or mtp_family_for_config(config)
        if family is None:
            raise NotImplementedError(f"Unsloth: no MTP family for `{getattr(config, 'model_type', None)}`.")
        num_heads = max(1, family.num_heads(config))
        if num_heads > family.max_heads:
            raise NotImplementedError(
                f"Unsloth: {family.name} MTP supports up to {family.max_heads} heads, the config declares {num_heads}."
            )
        self.family = family
        self.num_hidden_layers = int(config.num_hidden_layers)
        self.config = family.head_config(config)
        module = _module(family)
        layer_cls = getattr(module, family.layer_cls)
        norm_cls = getattr(module, family.norm_cls)
        self.heads = nn.ModuleList([
            _NextNLayer(family, self.config, i, layer_cls, norm_cls, self.num_hidden_layers)
            for i in range(num_heads)
        ])
        self.rotary_emb = getattr(module, family.rotary_cls)(self.config) if family.rotary_cls else None
        layer_types = getattr(self.config, "layer_types", None)
        window = getattr(self.config, "sliding_window", None)
        for i, head in enumerate(self.heads):
            for j, block in enumerate(head.block):
                idx = family.block_layer_idx(self.num_hidden_layers, i, j)
                kind = layer_types[idx] if isinstance(layer_types, (list, tuple)) and idx < len(layer_types) else None
                block._unsloth_layer_type = kind
                block._unsloth_window = int(window) if kind == "sliding_attention" and window else None
        self._block_kwargs = _call_kwargs(self.heads[0].block[0])
        self._embedding_first = True

    @property
    def h_source(self):
        """Which target hidden the head consumes: "final" (post norm) or "last_layer"."""
        return self.family.h_source

    def mtp_state_dict(self):
        """Trainable head tensors under the checkpoint's own names (frozen copies excluded)."""
        prefix = getattr(self, "_ckpt_prefix", None)
        return export_nextn_state_dict(self, prefix = prefix)

    def stitch_kwargs(self):
        """What stitch.write_mtp_tensors needs to find this family's tensors in a checkpoint."""
        n = self.num_hidden_layers
        frozen = {"embed_tokens", "shared_head_head"}

        def is_trainable_mtp_name(name):
            if not self.family.is_mtp_name(name, n):
                return False
            mapped = self.family.from_ckpt(name, n)
            return mapped is not None and mapped[1].split(".")[2] not in frozen

        # A full-model save drops the MTP layer, so owned copies must be appended back; duplicates are not, since
        # llama.cpp and the loader use the target's matrices.
        prefix = getattr(self, "_ckpt_prefix", None)
        prefix = self.family.default_prefix() if prefix is None else prefix
        owned = {}
        for i, head in enumerate(self.heads):
            for canon, tensor in ((f"heads.{i}.embed_tokens.weight", head.embed_weight),
                                  (f"heads.{i}.shared_head_head.weight", head.lm_head_weight)):
                if tensor is not None:
                    owned[self.family.to_ckpt(canon, prefix, n)] = tensor
        return {
            "is_mtp_name": is_trainable_mtp_name,
            "config_key": self.family.count_keys[0],
            "num_layers": len(self.heads),
            "append_extra": owned,
        }

    @property
    def chained(self):
        return self.family.chained and len(self.heads) > 1

    def _head_for_step(self, k):
        return self.heads[min(k, len(self.heads) - 1)] if self.chained else self.heads[0]

    def _block(self, block, x, mask, positions, cache, position_embeddings):
        kwargs = {
            "attention_mask": mask,
            "position_ids": positions,
            "past_key_values": cache,
            "use_cache": True,
            "position_embeddings": position_embeddings,
        }
        if self._block_kwargs is not None:
            kwargs = {k: v for k, v in kwargs.items() if k in self._block_kwargs}
        out = block(x, **kwargs)
        return out[0] if isinstance(out, (tuple, list)) else out

    def attach_frozen(self, state_dict, target_embed = None, target_lm_head = None):
        """Pop head-serialized embed/lm_head copies from state_dict: copies equal to the target's are dropped, others
        stay as frozen buffers; exports never write either; returns {canonical name: duplicate | owned}."""
        kinds = {}
        for i, head in enumerate(self.heads):
            for canon, attr, target in (
                (f"heads.{i}.embed_tokens.weight", "embed_weight", target_embed),
                (f"heads.{i}.shared_head_head.weight", "lm_head_weight", target_lm_head),
            ):
                tensor = state_dict.pop(canon, None)
                if tensor is None:
                    continue
                if target is not None and tuple(target.shape) == tuple(tensor.shape) and \
                        torch.equal(target.detach().to(device = tensor.device, dtype = tensor.dtype), tensor):
                    kinds[canon] = "duplicate"
                    continue
                kinds[canon] = "owned"
                setattr(head, attr, tensor.detach())
        return kinds

    def _embed(self, head, embed_tokens, tokens):
        if head.embed_weight is not None:
            return torch.nn.functional.embedding(tokens, head.embed_weight.to(tokens.device))
        return embed_tokens(tokens)

    def _run(self, head, tokens_embeds, hidden, mask, positions, cache):
        dtype = head.eh_proj.weight.dtype
        e = head.enorm(tokens_embeds.to(dtype))
        h = head.hnorm(hidden.to(dtype))
        x = head.eh_proj(torch.cat([e, h] if self._embedding_first else [h, e], dim = -1))
        rope_positions = positions[None].expand(3, -1, -1) if self.family.mrope else positions
        for block in head.block:
            position_embeddings = None
            if self.rotary_emb is not None:
                if self.family.rotary_layer_type:
                    position_embeddings = self.rotary_emb(x, rope_positions, block._unsloth_layer_type)
                else:
                    position_embeddings = self.rotary_emb(x, rope_positions)
            window = block._unsloth_window
            block_mask = mask if window is None else _window_mask(mask, positions, cache, window)
            x = self._block(block, x, block_mask, positions, cache, position_embeddings)
        cache.add_positions(positions)
        normed = head.shared_head_norm(x)
        return normed, (normed if self.family.feedback == "normed" else x)

    def forward(self, hidden_in, input_ids, embed_tokens, attention_mask = None, num_steps = 1, position_ids = None):
        # Wrappers such as DDP only sync gradients for calls that go through forward().
        return self.forward_train(
            hidden_in, input_ids, embed_tokens,
            attention_mask = attention_mask, num_steps = num_steps, position_ids = position_ids,
        )

    def forward_train(self, hidden_in, input_ids, embed_tokens, attention_mask = None, num_steps = 1, position_ids = None):
        """Teacher-forced drafting, same contract as Qwen35MTPHead.forward_train; hidden_in is the target hidden the
        family feeds (post final norm, or last layer output for h_source="last_layer")."""
        B, S = input_ids.shape
        device = input_ids.device
        valid = torch.ones(B, S, dtype = torch.bool, device = device) if attention_mask is None \
            else attention_mask.to(device = device).bool()
        if position_ids is None:
            position_ids = torch.arange(S, device = device).unsqueeze(0).expand(B, -1)
        dtype = self.heads[0].eh_proj.weight.dtype
        hidden_in = hidden_in.to(dtype)
        h0 = torch.cat([torch.zeros_like(hidden_in[:, :1]), hidden_in[:, :-1]], dim = 1)
        idx = torch.arange(S, device = device)
        causal = idx[None, :] <= idx[:, None]
        diagonal = idx[None, :] == idx[:, None]
        block0 = (causal[None] & valid[:, None, :]) | diagonal[None]
        neg = torch.finfo(dtype).min

        def row_inputs(m):
            tokens = input_ids.new_zeros(B, S)
            if m < S:
                tokens[:, : S - m] = input_ids[:, m:]
            return tokens

        def mask_for(m):
            allowed = torch.cat([block0] + [diagonal[None].expand(B, -1, -1)] * m, dim = -1)
            return torch.zeros(allowed.shape, dtype = dtype, device = device).masked_fill(~allowed, neg)[:, None]

        caches = {}
        rows_done = {}
        feedback = [h0]  # feedback[m] = h fed to row m
        steps = []
        for k in range(num_steps):
            head = self._head_for_step(k)
            key = id(head)
            cache = caches.setdefault(key, _ChainKV())
            out = None
            for m in range(rows_done.get(key, 0), k + 1):
                normed, fed = self._run(
                    head, self._embed(head, embed_tokens, row_inputs(m)), feedback[m], mask_for(m), position_ids + m, cache,
                )
                out = normed
                if m == k and len(feedback) == k + 1:
                    feedback.append(fed)
            rows_done[key] = k + 1

            label = input_ids.new_full((B, S), -100)
            lab_valid = torch.zeros_like(valid)
            tok_valid = torch.zeros_like(valid)
            if k < S:
                tok_valid[:, : S - k] = valid[:, k:]
            if k + 1 < S:
                label[:, : S - k - 1] = input_ids[:, k + 1 :]
                lab_valid[:, : S - k - 1] = valid[:, k + 1 :]
            step_valid = tok_valid & lab_valid & valid & (idx[None, :] >= 1)
            steps.append({
                "lm_head_weight": head.lm_head_weight,
                "hidden": out,
                "labels": label.masked_fill(~step_valid, -100),
                "positions": position_ids + k,
                "valid": step_valid,
            })
        return steps

    @torch.no_grad()
    def draft_greedy(self, hidden_in, input_ids, embed_tokens, lm_head_weight, anchor, num_steps):
        """Greedy draft of num_steps tokens after input_ids[:, :anchor + 1]; step k uses head k if chained else head 0;
        KV covers rows (x_j, h[j-1]) for j <= anchor; returns token ids [B, num_steps]."""
        B = input_ids.shape[0]
        device = input_ids.device
        dtype = self.heads[0].eh_proj.weight.dtype
        prefix = input_ids[:, : anchor + 1]
        h_prev = torch.cat([torch.zeros_like(hidden_in[:, :1]), hidden_in[:, :anchor]], dim = 1).to(dtype)
        positions = torch.arange(anchor + 1, device = device).unsqueeze(0).expand(B, -1)
        L = anchor + 1
        causal = torch.arange(L, device = device)[None, :] <= torch.arange(L, device = device)[:, None]
        neg = torch.finfo(dtype).min
        mask0 = torch.zeros((B, 1, L, L), dtype = dtype, device = device).masked_fill(~causal[None, None], neg)
        drafted = []
        rows_tokens = [prefix[:, -1:]]
        rows_h = []
        last_normed, last_fed = None, None
        for k in range(num_steps):
            head = self._head_for_step(k)
            cache = _ChainKV()
            normed, fed = self._run(head, self._embed(head, embed_tokens, prefix), h_prev, mask0, positions, cache)
            last_normed, last_fed = normed[:, -1:], fed[:, -1:]
            # rows 1..k of the chain: (drafted token, previous step's fed state)
            for m in range(1, k + 1):
                total = L + m
                mask = torch.zeros((B, 1, 1, total), dtype = dtype, device = device)
                normed_m, fed_m = self._run(
                    head, self._embed(head, embed_tokens, rows_tokens[m]), rows_h[m - 1], mask,
                    positions[:, -1:] + m, cache,
                )
                last_normed, last_fed = normed_m, fed_m
            weight = head.lm_head_weight if head.lm_head_weight is not None else lm_head_weight
            logits = last_normed.float() @ weight.float().T
            token = logits.argmax(-1)
            drafted.append(token)
            if len(rows_h) == k:
                rows_h.append(last_fed)
            if len(rows_tokens) == k + 1:
                rows_tokens.append(token)
        return torch.cat(drafted, dim = 1)


def pack_experts(state_dict, reference):
    """Pack per-expert checkpoint tensors into the packed module layout that reference (the module's state_dict) uses; returns (state_dict, packed prefixes)."""
    groups, out = {}, {}
    for name, tensor in state_dict.items():
        m = _PER_EXPERT.match(name)
        if m and any(f"{m.group(1)}.{p}" in reference for p in ("gate_up_proj", "up_proj", "down_proj")):
            groups.setdefault(m.group(1), {}).setdefault(int(m.group(2)), {})[m.group(3)] = tensor
        else:
            out[name] = tensor
    for prefix, experts in groups.items():
        order = sorted(experts)
        if order != list(range(len(order))):
            raise ValueError(f"Unsloth: MTP experts under `{prefix}` are not numbered 0..E-1.")
        if f"{prefix}.gate_up_proj" in reference:
            packed = torch.stack([torch.cat([experts[e]["gate_proj"], experts[e]["up_proj"]], 0) for e in order])
            out[f"{prefix}.gate_up_proj"] = _match_shape(packed, reference[f"{prefix}.gate_up_proj"])
        elif f"{prefix}.up_proj" in reference:
            packed = torch.stack([experts[e]["up_proj"] for e in order])
            out[f"{prefix}.up_proj"] = _match_shape(packed, reference[f"{prefix}.up_proj"])
        packed = torch.stack([experts[e]["down_proj"] for e in order])
        out[f"{prefix}.down_proj"] = _match_shape(packed, reference[f"{prefix}.down_proj"])
    return out, sorted(groups)


def _match_shape(tensor, reference):
    if tuple(tensor.shape) == tuple(reference.shape):
        return tensor
    if tensor.ndim == 3 and tuple(tensor.transpose(1, 2).shape) == tuple(reference.shape):
        return tensor.transpose(1, 2).contiguous()
    raise ValueError(f"Unsloth: packed MTP expert shape {tuple(tensor.shape)} does not fit {tuple(reference.shape)}.")


def unpack_experts(state_dict, prefixes, transposed = ()):
    """Inverse of pack_experts for the given packed prefixes (module layout -> per expert)."""
    out = {}
    for name, tensor in state_dict.items():
        m = _PACKED.match(name)
        if m is None or m.group(1) not in prefixes:
            out[name] = tensor
            continue
        prefix, kind = m.group(1), m.group(2)
        if prefix in transposed:
            tensor = tensor.transpose(1, 2)
        for e in range(tensor.shape[0]):
            if kind == "gate_up_proj":
                gate, up = tensor[e].chunk(2, dim = 0)
                out[f"{prefix}.{e}.gate_proj.weight"] = gate.contiguous()
                out[f"{prefix}.{e}.up_proj.weight"] = up.contiguous()
            else:
                out[f"{prefix}.{e}.{kind}.weight"] = tensor[e].contiguous()
    return out


def build_nextn_head(config, state_dict = None, dtype = None, device = None, strict = True, family = None,
                     target_embed = None, target_lm_head = None):
    """Build a NextNHead; `state_dict` uses canonical names (per-expert or packed experts)."""
    with torch.device("meta" if state_dict is not None else (device or "cpu")):
        head = NextNHead(config, family = family)
    if state_dict is None:
        _init_unset(head, getattr(config, "initializer_range", 0.02))
    else:
        head = head.to_empty(device = device or "cpu")
        if head.rotary_emb is not None:
            head.rotary_emb = type(head.rotary_emb)(head.config).to(device or "cpu")
        state_dict = dict(state_dict)
        head._frozen_kinds = head.attach_frozen(state_dict, target_embed, target_lm_head)
        for h in head.heads:
            for attr in ("embed_weight", "lm_head_weight"):
                if getattr(h, attr) is not None:
                    setattr(h, attr, getattr(h, attr).to(device or "cpu"))
        reference = head.state_dict()
        state_dict, _ = pack_experts(state_dict, reference)
        missing, unexpected = head.load_state_dict(state_dict, strict = False)
        # Non-persistent or derived buffers are not in checkpoints.
        missing = [k for k in missing if k in dict(head.named_parameters())]
        if strict and (missing or unexpected):
            raise ValueError(
                f"Unsloth: MTP state_dict does not match the head: missing={missing}, unexpected={unexpected}"
            )
        for name, buffer in head.named_buffers():
            # to_empty leaves persistent buffers the checkpoint lacks uninitialised, so float ones are zeroed; never the
            # frozen embedding / lm_head copies attached above.
            if name in reference and name not in state_dict and buffer.is_floating_point():
                buffer.zero_()
    if dtype is not None:
        keep = {}
        if head.rotary_emb is not None:
            keep = {n: b.detach().clone().float() for n, b in head.rotary_emb.named_buffers(recurse = False) if "inv_freq" in n}
        frozen = {(i, attr): getattr(h, attr) for i, h in enumerate(head.heads)
                  for attr in ("embed_weight", "lm_head_weight") if getattr(h, attr) is not None}
        # transformers keeps e.g. MoE routing correction biases in fp32 (_keep_in_fp32_modules*); rounding them to bf16 reorders the top-k experts.
        fp32_names = _keep_in_fp32_names(head.family)
        keep_fp32 = {
            name: tensor.detach().clone().float()
            for name, tensor in list(head.named_parameters()) + list(head.named_buffers())
            if tensor is not None and tensor.is_floating_point() and any(k in name for k in fp32_names)
        }
        head = head.to(dtype)
        for name, value in keep_fp32.items():
            module_name, _, attr = name.rpartition(".")
            module = head.get_submodule(module_name)
            current = getattr(module, attr)
            if isinstance(current, nn.Parameter):
                current.data = value.to(current.device)
            else:
                setattr(module, attr, value.to(current.device))
        for name, buffer in keep.items():
            setattr(head.rotary_emb, name, buffer)
        # Frozen embedding / lm_head copies keep their checkpoint dtype (they can be large).
        for (i, attr), tensor in frozen.items():
            setattr(head.heads[i], attr, tensor.to(device = getattr(head.heads[i], attr).device))
    return head


def _keep_in_fp32_names(family):
    """Module-name fragments the family's transformers model keeps in float32."""
    names = set()
    module = _module(family)
    for obj in vars(module).values():
        if isinstance(obj, type):
            for attr in ("_keep_in_fp32_modules_strict", "_keep_in_fp32_modules"):
                names.update(getattr(obj, attr, None) or ())
    return sorted(names)


def _init_unset(head, std):
    for name, param in head.named_parameters():
        if re.search(r"experts\.(gate_up_proj|up_proj|down_proj)$|\.gate\.weight$", name):
            with torch.no_grad():
                param.normal_(0.0, std)
    for name, buffer in head.named_buffers():
        if buffer.is_floating_point() and not torch.isfinite(buffer).all():
            buffer.zero_()


def load_nextn_state_dict(source, config, revision = None):
    """Read only MTP tensors of a NextN family checkpoint; returns (state_dict with canonical names, per-expert layout as in checkpoint, info)."""
    from safetensors import safe_open
    from .mtp import _weight_map

    family = mtp_family_for_config(config)
    if family is None:
        raise NotImplementedError(f"Unsloth: no MTP family for `{getattr(config, 'model_type', None)}`.")
    n = int(config.num_hidden_layers)
    weight_map, path = _weight_map(source, revision = revision)
    names = {name: shard for name, shard in weight_map.items() if family.is_mtp_name(name, n)}
    if not names:
        raise ValueError(
            f"Unsloth: `{source}` has no {family.name} MTP tensors, so it has no native MTP head. "
            "Load the original (non pre-quantized) checkpoint, or build a randomly initialised head."
        )
    quantized = sorted(n for n in names if re.search(r"\.(absmax|quant_map|quant_state|nested_absmax|SCB)", n))
    if quantized:
        raise ValueError(
            f"Unsloth: `{source}` stores its MTP head pre-quantized (`{quantized[0]}`). Pass `source=` "
            "the original BF16 / FP8 repo; only its MTP shards are downloaded."
        )
    by_shard = {}
    for name, shard in names.items():
        by_shard.setdefault(shard, []).append(name)
    state_dict, prefixes, files, mapping = {}, set(), [], {}
    for shard, shard_names in sorted(by_shard.items()):
        if path is not None:
            shard_path = str(path / shard)
        else:
            from huggingface_hub import hf_hub_download
            shard_path = hf_hub_download(str(source), shard, revision = revision)
        files.append(shard_path)
        with safe_open(shard_path, framework = "pt", device = "cpu") as f:
            for name in shard_names:
                prefix, canon = family.from_ckpt(name, n)
                prefixes.add(prefix)
                mapping[canon] = name
                state_dict[canon] = f.get_tensor(name)
    if len(prefixes) != 1:
        raise ValueError(f"Unsloth: `{source}` mixes MTP prefixes {sorted(prefixes)}.")
    heads = {int(k.split(".")[1]) for k in state_dict}
    declared = family.num_heads(config)
    if declared and len(heads) != declared:
        raise ValueError(f"Unsloth: `{source}` declares {declared} MTP heads but carries {len(heads)}.")
    dtypes = {str(t.dtype) for t in state_dict.values()}
    info = {
        "family": family.name,
        "source": str(source),
        "revision": revision,
        "prefix": prefixes.pop(),
        "names": sorted(names),
        "ckpt_names": mapping,
        "num_hidden_layers": n,
        "num_layers": len(heads),
        "files": files,
        "dtype": dtypes.pop() if len(dtypes) == 1 else sorted(dtypes),
    }
    return state_dict, info


def export_nextn_state_dict(head, prefix = None, num_hidden_layers = None):
    """Head weights under the family's checkpoint names, experts per expert, frozen copies excluded."""
    family = head.family
    n = head.num_hidden_layers if num_hidden_layers is None else num_hidden_layers
    prefix = family.default_prefix() if prefix is None else prefix
    persistent = {k: v for k, v in head.state_dict().items() if k.startswith("heads.")}
    packed = sorted({m.group(1) for m in (_PACKED.match(k) for k in persistent) if m})
    canon = unpack_experts(persistent, packed)
    return {family.to_ckpt(k, prefix, n): v for k, v in canon.items()}


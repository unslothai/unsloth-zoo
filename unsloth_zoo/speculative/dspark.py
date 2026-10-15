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

"""DSpark drafters for Qwen3 targets (DeepSpec `Qwen3DSparkModel` schema).

Contract (unslothai/llama.cpp `src/models/dflash.cpp`, `common/speculative.cpp` draft-dspark,
`conversion/qwen.py` DSparkModel, sample_from_anchor = true for this schema):
- context = concat of target decoder-layer outputs `target_layer_ids` (pre final norm), each
  position p injected as K/V of `hidden_norm(fc(context))` at position p;
- a block at anchor a holds [x_a, mask, ..., mask] at positions a..a+B-1, attends non-causally to
  itself and to context positions < a; slot j predicts x[a+j+1];
- logits = lm_head(norm(h)) + markov_w2(markov_w1(prev)), prev = x_a for slot 0, then the previous
  slot's token (teacher forced in training, greedy argmax in llama.cpp);
- confidence = sigmoid(proj([norm(h); markov_w1(prev)])), trained against 1 - TV(target, draft).
"""

__all__ = [
    "Qwen3DSparkDrafter",
    "DSparkTrainingArguments",
    "UnslothDSparkTrainer",
    "sample_anchor_positions",
    "dspark_loss",
]

import json
import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import Trainer, TrainingArguments

from . import losses
from .capture import resolve_text_parts, run_frozen_target

_CONFIG_EXTRAS = (
    "block_size", "target_layer_ids", "mask_token_id", "num_anchors", "enable_confidence_head",
    "confidence_head_with_markov", "markov_rank", "markov_head_type", "num_target_layers",
)


class _Rotary(nn.Module):
    """Default RoPE; own copy since Unsloth replaces the transformers Qwen3 rotary class."""

    def __init__(self, config):
        super().__init__()
        rope = getattr(config, "rope_parameters", None) or {}
        rope_type = rope.get("rope_type", "default")
        if rope_type != "default":
            raise NotImplementedError(f"Unsloth: DSpark drafter supports default RoPE, got {rope_type!r}.")
        theta = float(rope.get("rope_theta", getattr(config, "rope_theta", 10000.0)))
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        # A plain attribute, not a buffer: `.to(bfloat16)` on the drafter must not round the
        # frequencies (bf16 inv_freq drifts badly at long positions).
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype = torch.int64).float() / dim))

    @torch.no_grad()
    def forward(self, x, position_ids):
        freqs = position_ids[..., None].float() * self.inv_freq.to(position_ids.device)
        emb = torch.cat((freqs, freqs), dim = -1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat((-x2, x1), dim = -1)


class _Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        H = config.hidden_size
        self.head_dim = getattr(config, "head_dim", None) or H // config.num_attention_heads
        self.n_heads = config.num_attention_heads
        self.n_kv = config.num_key_value_heads
        bias = bool(getattr(config, "attention_bias", False))
        self.q_proj = nn.Linear(H, self.n_heads * self.head_dim, bias = bias)
        self.k_proj = nn.Linear(H, self.n_kv * self.head_dim, bias = bias)
        self.v_proj = nn.Linear(H, self.n_kv * self.head_dim, bias = bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, H, bias = bias)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps = config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps = config.rms_norm_eps)

    def forward(self, hidden, context, cos, sin, mask):
        # cos / sin cover [context positions ; block positions]; queries use the block part.
        B, Q, _ = hidden.shape
        C = context.shape[1]
        q = self.q_norm(self.q_proj(hidden).view(B, Q, self.n_heads, self.head_dim)).transpose(1, 2)
        k = torch.cat([self.k_proj(context), self.k_proj(hidden)], dim = 1).view(B, C + Q, self.n_kv, self.head_dim)
        v = torch.cat([self.v_proj(context), self.v_proj(hidden)], dim = 1).view(B, C + Q, self.n_kv, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        q = q * cos[:, :, C:] + _rotate_half(q) * sin[:, :, C:]
        k = k * cos + _rotate_half(k) * sin
        if self.n_kv != self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv, dim = 1)
            v = v.repeat_interleave(self.n_heads // self.n_kv, dim = 1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask = mask, scale = self.head_dim ** -0.5)
        return self.o_proj(out.transpose(1, 2).reshape(B, Q, self.n_heads * self.head_dim))


class _Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm
        self.self_attn = _Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps = config.rms_norm_eps)

    def forward(self, hidden, context, cos, sin, mask):
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), context, cos, sin, mask)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class _VanillaMarkov(nn.Module):
    def __init__(self, vocab_size, rank):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias = False)


class _ConfidenceHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, features):
        return self.proj(features).squeeze(-1)


def sample_anchor_positions(loss_mask, num_anchors, generator = None):
    """DeepSpec anchor sampler: anchors a with loss_mask[a] and loss_mask[a+1], sorted, padded.

    Returns (anchors [B, N] long, keep [B, N] bool); padded slots point at 0 with keep False.
    """
    B, S = loss_mask.shape
    device = loss_mask.device
    n_cand = max(S - 1, 0)
    if n_cand == 0:
        return (torch.zeros(B, num_anchors, dtype = torch.long, device = device),
                torch.zeros(B, num_anchors, dtype = torch.bool, device = device))
    valid = (loss_mask[:, :n_cand] > 0) & (loss_mask[:, 1:n_cand + 1] > 0)
    idx = torch.arange(n_cand, device = device).expand(B, -1)
    masked = torch.where(valid, idx, torch.full_like(idx, S + 1))
    noise = torch.rand(B, n_cand, device = device, generator = generator)
    noise = torch.where(valid, noise, torch.full_like(noise, 2.0))
    order = noise.sort(dim = 1).indices
    picked = torch.gather(masked, 1, order)
    if n_cand < num_anchors:
        picked = torch.cat([picked, torch.full((B, num_anchors - n_cand), S + 1, dtype = picked.dtype, device = device)], 1)
    anchors = picked[:, :num_anchors].sort(dim = 1).values
    keep = torch.arange(num_anchors, device = device)[None] < valid.sum(1, keepdim = True).clamp(max = num_anchors)
    return torch.where(keep, anchors, torch.zeros_like(anchors)), keep


class Qwen3DSparkDrafter(nn.Module):
    """DeepSpec-compatible Qwen3 DSpark drafter; state_dict names match `Qwen3DSparkModel`."""

    def __init__(self, config):
        super().__init__()
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        self.config = config
        if str(getattr(config, "markov_head_type", "vanilla")).lower() != "vanilla":
            raise NotImplementedError("Unsloth: only the vanilla DSpark Markov head runs in llama.cpp.")
        H, V = config.hidden_size, config.vocab_size
        self.target_layer_ids = [int(i) for i in config.target_layer_ids]
        self.block_size = int(config.block_size)
        self.mask_token_id = int(config.mask_token_id)
        self.embed_tokens = nn.Embedding(V, H)
        self.layers = nn.ModuleList([_Layer(config) for _ in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(H, eps = config.rms_norm_eps)
        self.rotary_emb = _Rotary(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * H, H, bias = False)
        self.hidden_norm = Qwen3RMSNorm(H, eps = config.rms_norm_eps)
        self.lm_head = nn.Linear(H, V, bias = False)
        rank = int(getattr(config, "markov_rank", 0))
        self.markov_head = _VanillaMarkov(V, rank) if rank > 0 else None
        self.confidence_head = None
        if bool(getattr(config, "enable_confidence_head", False)):
            with_markov = bool(getattr(config, "confidence_head_with_markov", False))
            if with_markov and self.markov_head is None:
                raise ValueError("Unsloth: confidence_head_with_markov needs markov_rank > 0.")
            self.confidence_head = _ConfidenceHead(H + (rank if with_markov else 0))
        self._confidence_with_markov = bool(getattr(config, "confidence_head_with_markov", False))

    # ---- construction / io -------------------------------------------------------------
    @staticmethod
    def _config_from_dict(d):
        from transformers import Qwen3Config
        extras = {k: d[k] for k in _CONFIG_EXTRAS if k in d}
        base = {k: v for k, v in d.items() if k not in extras and k not in ("architectures", "model_type")}
        config = Qwen3Config(**base)
        for k, v in extras.items():
            setattr(config, k, v)
        return config

    @classmethod
    def from_pretrained(cls, path, dtype = None, device = None):
        from safetensors.torch import load_file
        with open(os.path.join(path, "config.json"), encoding = "utf-8") as f:
            d = json.load(f)
        if "Qwen3DSparkModel" not in (d.get("architectures") or []):
            raise ValueError(f"Unsloth: {path} is not a Qwen3DSparkModel checkpoint.")
        drafter = cls(cls._config_from_dict(d))
        state = {}
        for name in sorted(os.listdir(path)):
            if name.endswith(".safetensors"):
                state.update(load_file(os.path.join(path, name)))
        missing, unexpected = drafter.load_state_dict(state, strict = False)
        missing = [m for m in missing if not m.startswith("rotary_emb.")]
        if missing or unexpected:
            raise ValueError(f"Unsloth: DSpark checkpoint mismatch, missing={missing} unexpected={unexpected}")
        return drafter.to(device = device, dtype = dtype)

    @classmethod
    def from_target(cls, target_model, target_layer_ids, mask_token_id, num_draft_layers = 5,
                    block_size = 7, num_anchors = 512, markov_rank = 256, confidence_head = True,
                    dtype = None, device = None):
        """Fresh drafter shaped like the target's layers, sharing (copies of) its embedding / lm_head."""
        parts = resolve_text_parts(target_model)
        d = parts["config"].to_dict()
        for key in ("quantization_config", "_name_or_path", "auto_map", "architectures", "torch_dtype"):
            d.pop(key, None)
        n_target = int(d["num_hidden_layers"])
        ids = [int(i) for i in target_layer_ids]
        if not ids or ids != sorted(set(ids)) or ids[0] < -1 or ids[-1] >= n_target:
            raise ValueError(f"Unsloth: target_layer_ids {ids} must be increasing within [-1, {n_target - 1}].")
        d.update(
            num_target_layers = n_target, num_hidden_layers = int(num_draft_layers),
            layer_types = ["full_attention"] * int(num_draft_layers), block_size = int(block_size),
            target_layer_ids = ids, mask_token_id = int(mask_token_id), num_anchors = int(num_anchors),
            enable_confidence_head = bool(confidence_head), confidence_head_with_markov = bool(confidence_head),
            markov_rank = int(markov_rank), markov_head_type = "vanilla", tie_word_embeddings = False,
            use_sliding_window = False, sliding_window = None,
        )
        drafter = cls(cls._config_from_dict(d))
        drafter.apply(drafter._init_weights)
        with torch.no_grad():
            drafter.embed_tokens.weight.copy_(parts["embed_tokens"].weight.detach().float())
            drafter.lm_head.weight.copy_(parts["lm_head"].weight.detach().float())
        return drafter.to(device = device or parts["embed_tokens"].weight.device, dtype = dtype)

    def _init_weights(self, module):
        std = float(getattr(self.config, "initializer_range", 0.02))
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std = std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std = std)

    def freeze_shared(self):
        """Embedding and lm_head stay the target's (DeepSpec freezes them too)."""
        self.embed_tokens.requires_grad_(False)
        self.lm_head.requires_grad_(False)
        return self

    def save_pretrained(self, path, dtype = torch.bfloat16):
        from safetensors.torch import save_file
        os.makedirs(path, exist_ok = True)
        d = self.config.to_dict()
        for k in _CONFIG_EXTRAS:
            if hasattr(self.config, k):
                d[k] = getattr(self.config, k)
        d["architectures"] = ["Qwen3DSparkModel"]
        d["model_type"] = "qwen3"
        d["dtype"] = str(dtype).replace("torch.", "")
        for key in ("_attn_implementation_autoset", "_name_or_path", "chunk_size_feed_forward", "id2label",
                    "label2id", "is_encoder_decoder", "output_attentions", "output_hidden_states", "problem_type",
                    "return_dict"):
            d.pop(key, None)
        with open(os.path.join(path, "config.json"), "w", encoding = "utf-8") as f:
            json.dump(d, f, indent = 2)
        state = {k: v.detach().to(dtype).contiguous().cpu() for k, v in self.state_dict().items() if not k.startswith("rotary_emb.")}
        save_file(state, os.path.join(path, "model.safetensors"), metadata = {"format": "pt"})

    # ---- forward -------------------------------------------------------------------------
    def context_states(self, captured):
        """Concat of the target decoder-layer outputs this drafter reads (-1 = embedding output)."""
        missing = [i for i in self.target_layer_ids if i not in captured]
        if missing:
            raise KeyError(f"Unsloth: target layers {missing} were not captured.")
        # A target sharded over GPUs leaves each layer's states on that layer's device.
        device = self.lm_head.weight.device
        return torch.cat([captured[i].to(device) for i in self.target_layer_ids], dim = -1)

    def _block_mask(self, anchors, keep, S):
        B, N = anchors.shape
        bs = self.block_size
        q_block = torch.arange(N * bs, device = anchors.device) // bs
        ctx = torch.arange(S, device = anchors.device)[None, None] < anchors[:, q_block][..., None]
        own = (q_block[:, None] == q_block[None, :])[None].expand(B, -1, -1)
        # A dropped block sees only itself: never all-masked (SDPA would give NaN), never read.
        ctx = ctx & keep[:, q_block][..., None]
        return torch.cat([ctx, own], dim = -1)[:, None]

    def forward(self, context, input_ids, loss_mask, num_anchors = None, generator = None, anchors = None):
        # Wrappers such as DDP only sync gradients for calls that go through forward().
        return self.forward_train(context, input_ids, loss_mask, num_anchors, generator, anchors)

    def forward_train(self, context, input_ids, loss_mask, num_anchors = None, generator = None, anchors = None):
        """Teacher-forced DeepSpec training forward over sampled anchor blocks.

        Returns hidden (normed draft states [B, N, bs, H]), target_ids, prev_ids, eval_mask,
        label_index (position of each label), keep.
        """
        B, S = input_ids.shape
        bs = self.block_size
        if anchors is None:
            anchors, keep = sample_anchor_positions(loss_mask, int(num_anchors or self.config.num_anchors), generator)
        else:
            anchors, keep = anchors
        N = anchors.shape[1]
        device = input_ids.device
        noise_ids = torch.full((B, N, bs), self.mask_token_id, dtype = torch.long, device = device)
        anchor_tok = torch.gather(input_ids, 1, anchors)
        noise_ids[:, :, 0] = torch.where(keep, anchor_tok, torch.full_like(anchor_tok, self.mask_token_id))
        hidden = self.embed_tokens(noise_ids.view(B, N * bs))
        ctx = self.hidden_norm(self.fc(context.to(hidden.dtype)))
        offsets = torch.arange(bs, device = device)
        pos = torch.cat([torch.arange(S, device = device).expand(B, -1), (anchors[..., None] + offsets).view(B, N * bs)], 1)
        cos, sin = self.rotary_emb(hidden, pos)
        mask = self._block_mask(anchors, keep, S)
        for layer in self.layers:
            hidden = layer(hidden, ctx, cos, sin, mask)
        hidden = self.norm(hidden).view(B, N, bs, -1)

        label_index = anchors[..., None] + offsets + 1
        in_range = label_index < S
        safe = torch.where(keep[..., None], label_index.clamp(max = S - 1), torch.zeros_like(label_index))
        target_ids = torch.gather(input_ids[:, None].expand(-1, N, -1), 2, safe)
        lm = torch.gather(loss_mask[:, None].expand(-1, N, -1), 2, safe) > 0
        eval_mask = (in_range & lm & keep[..., None]).int().cumprod(-1).bool()
        prev_ids = torch.cat([anchor_tok[..., None], target_ids[..., :-1]], dim = -1)
        return {
            "hidden": hidden, "target_ids": target_ids, "prev_ids": prev_ids, "eval_mask": eval_mask,
            "label_index": safe, "keep": keep, "anchors": anchors,
        }

    def logits(self, hidden, prev_ids):
        out = self.lm_head(hidden)
        if self.markov_head is not None:
            out = out + self.markov_head.markov_w2(self.markov_head.markov_w1(prev_ids))
        return out

    def confidence_logits(self, hidden, prev_ids):
        if self.confidence_head is None:
            return None
        feats = hidden
        if self._confidence_with_markov:
            feats = torch.cat([hidden, self.markov_head.markov_w1(prev_ids).to(hidden.dtype)], dim = -1)
        return self.confidence_head(feats).float()


def _row_terms(hidden, prev_emb, lm_w, mk_w2, teacher, labels):
    logits = (hidden @ lm_w.T).float()
    if prev_emb is not None:
        logits = logits + (prev_emb @ mk_w2.T).float()
    logp = logits.log_softmax(-1)
    ce = -logp.gather(-1, labels[:, None]).squeeze(-1)
    correct = (logits.argmax(-1) == labels).float()
    if teacher is None:
        return ce, torch.zeros_like(ce), correct
    l1 = (logp.exp() - teacher).abs().sum(-1)
    return ce, l1, correct


def dspark_loss(drafter, out, teacher_hidden = None, ce_alpha = 0.1, l1_alpha = 0.9,
                confidence_alpha = 1.0, decay_gamma = 4.0, chunk_size = 1024, teacher_lm_head_weight = None):
    """DeepSpec DSpark loss, chunked over supervised rows so [rows, vocab] logits never coexist.

    teacher_hidden: target post-norm states [B, S, H]; slot j's teacher is the target's own
    prediction at label_index - 1, projected with `teacher_lm_head_weight` (the target's lm_head;
    defaults to the drafter's copy). Returns (loss, stats).
    """
    eval_mask = out["eval_mask"]
    rows = eval_mask.nonzero(as_tuple = True)
    hidden = out["hidden"][rows]
    labels = out["target_ids"][rows]
    prev = out["prev_ids"][rows]
    slot = rows[2]
    weight = torch.exp(-slot.float() / float(decay_gamma)) if decay_gamma and decay_gamma > 0 else torch.ones_like(slot, dtype = torch.float32)
    lm_w = drafter.lm_head.weight
    teacher_w = lm_w if teacher_lm_head_weight is None else teacher_lm_head_weight.to(device = lm_w.device, dtype = hidden.dtype)
    # A chunk holds ~6 fp32 [rows, V] tensors (logits, log-softmax, probs, teacher, l1 terms).
    chunk_size = max(1, min(chunk_size, losses.LOGITS_BUDGET_BYTES // (lm_w.shape[0] * 4 * 6)))
    mk = drafter.markov_head
    prev_emb = mk.markov_w1(prev).to(hidden.dtype) if mk is not None else None
    need_teacher = l1_alpha > 0 or (drafter.confidence_head is not None and confidence_alpha > 0)
    teacher_rows = None
    if need_teacher:
        if teacher_hidden is None:
            raise ValueError("Unsloth: the l1 / confidence terms need the target's final hidden states.")
        b, n, j = rows
        teacher_rows = teacher_hidden[b, out["label_index"][b, n, j] - 1].to(hidden.dtype)
    ce_all, l1_all, correct_all = [], [], []
    for s in range(0, hidden.shape[0], chunk_size):
        e = slice(s, s + chunk_size)
        teacher = None
        if teacher_rows is not None:
            with torch.no_grad():
                teacher = (teacher_rows[e] @ teacher_w.T).float().softmax(-1)
        args = (hidden[e], None if prev_emb is None else prev_emb[e], lm_w, None if mk is None else mk.markov_w2.weight, teacher, labels[e])
        if torch.is_grad_enabled() and hidden.requires_grad:
            ce, l1, correct = checkpoint(_row_terms, *args, use_reentrant = False)
        else:
            ce, l1, correct = _row_terms(*args)
        ce_all.append(ce)
        l1_all.append(l1)
        correct_all.append(correct)
    zero = hidden.new_zeros((), dtype = torch.float32)
    if not ce_all:
        return zero + 0.0 * sum(p.sum() for p in drafter.parameters() if p.requires_grad), {"n_tokens": 0}
    ce, l1, correct = torch.cat(ce_all), torch.cat(l1_all), torch.cat(correct_all)
    den = weight.sum() + 1e-6
    ce_loss = (ce * weight).sum() / den
    loss = ce_alpha * ce_loss
    stats = {"n_tokens": int(ce.numel()), "ce_loss": float(ce_loss.detach())}
    if need_teacher:
        accept = (1.0 - 0.5 * l1.detach()).clamp(0.0, 1.0)
        l1_loss = (l1 * weight).sum() / den
        if l1_alpha > 0:
            loss = loss + l1_alpha * l1_loss
        stats["l1_loss"] = float(l1_loss.detach())
        stats["accept_rate"] = float(accept.mean())
        conf = drafter.confidence_logits(hidden, prev)
        if conf is not None and confidence_alpha > 0:
            conf_loss = (F.binary_cross_entropy_with_logits(conf, accept, reduction = "none") * weight).sum() / den
            loss = loss + confidence_alpha * conf_loss
            stats["confidence_loss"] = float(conf_loss.detach())
    bs = out["hidden"].shape[2]
    per_slot = torch.zeros(bs, device = slot.device).index_add_(0, slot, correct)
    count = torch.zeros(bs, device = slot.device).index_add_(0, slot, torch.ones_like(correct))
    stats["top1_per_slot"] = (per_slot / count.clamp_min(1)).tolist()
    stats["count_per_slot"] = count.tolist()
    return loss, stats


@dataclass
class DSparkTrainingArguments(TrainingArguments):
    dspark_num_anchors: int = field(default = 512, metadata = {"help": "Anchor blocks sampled per sequence."})
    dspark_ce_alpha: float = 0.1
    dspark_l1_alpha: float = 0.9
    dspark_confidence_alpha: float = 1.0
    dspark_decay_gamma: float = 4.0
    dspark_chunk_size: int = 1024
    remove_unused_columns: bool = False


class UnslothDSparkTrainer(Trainer):
    """Trains a DSpark drafter (`model`) against a frozen target, one target pass per batch.

    Rows need `input_ids` and optionally `labels` (-100 = not supervised, e.g. the prompt).
    """

    def __init__(self, model = None, args = None, target_model = None, **kwargs):
        if target_model is None:
            raise ValueError("Unsloth: UnslothDSparkTrainer needs `target_model`.")
        if args is None:
            args = DSparkTrainingArguments(output_dir = "dspark_outputs")
        if not isinstance(args, DSparkTrainingArguments):
            raise TypeError("Unsloth: UnslothDSparkTrainer needs DSparkTrainingArguments.")
        self.target_model = target_model
        for p in target_model.parameters():
            p.requires_grad_(False)
        target_model.eval()
        model.freeze_shared()
        if kwargs.get("data_collator") is None:
            from .trainer import MTPDataCollator
            tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
            tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
            pad = getattr(tokenizer, "pad_token_id", None)
            if pad is None:
                pad = getattr(tokenizer, "eos_token_id", 0) or 0
            kwargs["data_collator"] = MTPDataCollator(pad)
        super().__init__(model = model, args = args, **kwargs)
        self._dspark_log = []

    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels", input_ids)
        if attention_mask is not None and bool((attention_mask[:, 0] == 0).any()):
            raise ValueError("Unsloth: UnslothDSparkTrainer needs right-padded batches (left padding misaligns positions).")
        loss_mask = labels != -100
        if attention_mask is not None:
            loss_mask = loss_mask & attention_mask.bool()
        drafter = self.accelerator.unwrap_model(model)
        # DeepSpec's -1 is the embedding output, not the last layer (a Python index to the capture).
        ids = drafter.target_layer_ids
        captured = run_frozen_target(self.target_model, input_ids, attention_mask = attention_mask,
                                     layer_ids = [i for i in ids if i >= 0], embed = -1 in ids)
        if -1 in ids:
            captured[-1] = captured.pop("embed")
        dtype = next(p for p in drafter.parameters()).dtype
        out = model(drafter.context_states(captured).to(dtype), input_ids, loss_mask, self.args.dspark_num_anchors)
        loss, stats = dspark_loss(
            drafter, out, teacher_hidden = captured["final"].to(device = out["hidden"].device, dtype = dtype), ce_alpha = self.args.dspark_ce_alpha,
            l1_alpha = self.args.dspark_l1_alpha, confidence_alpha = self.args.dspark_confidence_alpha,
            decay_gamma = self.args.dspark_decay_gamma, chunk_size = self.args.dspark_chunk_size,
            teacher_lm_head_weight = resolve_text_parts(self.target_model)["lm_head"].weight,
        )
        if model.training:
            self._dspark_log.append(stats)
        return (loss, None) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys = None):
        # The drafter has no forward(), so the stock prediction_step cannot call model(**inputs).
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)

    def log(self, logs, *args, **kwargs):
        rows = [s for s in self._dspark_log if s.get("n_tokens")]
        if rows:
            n = sum(s["n_tokens"] for s in rows)
            for key in ("ce_loss", "l1_loss", "confidence_loss", "accept_rate"):
                vals = [(s[key], s["n_tokens"]) for s in rows if key in s]
                if vals:
                    logs[f"dspark_{key}"] = round(sum(v * c for v, c in vals) / n, 4)
            bs = len(rows[0]["top1_per_slot"])
            for j in range(bs):
                c = sum(s["count_per_slot"][j] for s in rows)
                if c:
                    logs[f"dspark_top1_{j}"] = round(sum(s["top1_per_slot"][j] * s["count_per_slot"][j] for s in rows) / c, 4)
        self._dspark_log = []
        return super().log(logs, *args, **kwargs)

    def _save(self, output_dir = None, state_dict = None):
        output_dir = output_dir or self.args.output_dir
        # Checkpoints keep the training dtype so resuming does not round fp32 weights to bf16.
        self.model.save_pretrained(output_dir, dtype = next(self.model.parameters()).dtype)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

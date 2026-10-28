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

__all__ = [
    "MTPTrainingArguments",
    "UnslothMTPTrainer",
    "MTPDataCollator",
    "mtp_step_losses",
]

import os
from dataclasses import dataclass, field

import torch
from transformers import Trainer, TrainingArguments

from .capture import effective_lm_head_weight, resolve_text_parts, run_frozen_target
from . import losses
from .losses import chunked_lm_loss, fastmtp_step_weights


@dataclass
class MTPTrainingArguments(TrainingArguments):
    mtp_num_steps: int = field(default = 3, metadata = {"help": "Recursive draft depth trained per anchor."})
    mtp_step_beta: float = field(default = 0.6, metadata = {"help": "FastMTP step-weight decay."})
    mtp_loss_type: str = field(default = "ce", metadata = {"help": "ce | kl_div | tv | lk_hybrid"})
    mtp_label_source: str = field(default = "data", metadata = {"help": "ce labels: data tokens or teacher argmax."})
    mtp_chunk_size: int = field(default = 1024, metadata = {"help": "Max rows per vocab-projection chunk (also capped by a 256 MB logits budget)."})
    remove_unused_columns: bool = False


class MTPDataCollator:
    """Right-pads `input_ids` (+ optional `labels`) and builds the attention mask."""

    def __init__(self, pad_token_id, max_length = None):
        self.pad_token_id = int(pad_token_id)
        self.max_length = max_length

    def __call__(self, features):
        rows = []
        for f in features:
            ids = list(f["input_ids"])
            labels = list(f["labels"]) if f.get("labels") is not None else list(ids)
            if f.get("attention_mask") is not None:
                # Pre-padded rows: keep only the real tokens, then re-pad on the right.
                keep = [i for i, m in enumerate(f["attention_mask"]) if m]
                ids, labels = [ids[i] for i in keep], [labels[i] for i in keep]
            if self.max_length is not None:
                ids, labels = ids[:self.max_length], labels[:self.max_length]
            rows.append((ids, labels))
        width = max(len(ids) for ids, _ in rows)
        input_ids = torch.full((len(rows), width), self.pad_token_id, dtype = torch.long)
        labels = torch.full((len(rows), width), -100, dtype = torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype = torch.long)
        for i, (ids, lab) in enumerate(rows):
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype = torch.long)
            labels[i, :len(lab)] = torch.tensor(lab, dtype = torch.long)
            attention_mask[i, :len(ids)] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def _shift_window(tensor, start, length, fill):
    """`tensor[:, start:start + length]`, padded with `fill` past the end."""
    out = torch.full((tensor.shape[0], length) + tuple(tensor.shape[2:]), fill, dtype = tensor.dtype, device = tensor.device)
    available = max(0, min(length, tensor.shape[1] - start))
    if available > 0:
        out[:, :available] = tensor[:, start:start + available]
    return out


def mtp_step_losses(head, parts, final_hidden, input_ids, attention_mask, data_labels, args, normalizer = None,
                    head_hidden = None):
    """Weighted MTP loss and per-step (loss, n_tokens, n_correct); normalizer = supervised tokens in the accumulated
    batch, so microbatches are weighted by tokens, not means; chain p starts from (final_hidden[p-1], x[p]); step k
    predicts x[p+k+1] with teacher final_hidden[:, p+k]."""
    steps = head(
        final_hidden if head_hidden is None else head_hidden,
        input_ids,
        parts["embed_tokens"],
        attention_mask = attention_mask,
        num_steps = args.mtp_num_steps,
    )
    weights = fastmtp_step_weights(args.mtp_step_beta, len(steps))
    # The effective head: an unmerged LoRA on the target's lm_head is part of the teacher (and the draft).
    target_lm_head_weight = effective_lm_head_weight(parts["lm_head"])
    supervised = data_labels != -100
    total = final_hidden.new_zeros((), dtype = torch.float32)
    stats = []
    for k, (step, weight) in enumerate(zip(steps, weights)):
        hidden, labels = step["hidden"], step["labels"]
        # Heads that ship their own lm_head copy draft with it (llama.cpp does); the teacher stays the target's.
        lm_head_weight = step.get("lm_head_weight")
        if lm_head_weight is None:
            lm_head_weight = target_lm_head_weight
        length = labels.shape[1]
        keep = _shift_window(supervised, k + 1, length, False)
        labels = torch.where(keep, labels, torch.full_like(labels, -100))
        teacher_hidden = None
        if args.mtp_loss_type != "ce" or args.mtp_label_source == "teacher":
            teacher_hidden = _shift_window(final_hidden, k, length, 0).reshape(-1, final_hidden.shape[-1])
        if args.mtp_label_source == "teacher":
            # Same logits budget as chunked_lm_loss: one [rows, V] projection per chunk.
            per_row = target_lm_head_weight.shape[0] * target_lm_head_weight.element_size()
            rows = max(1, min(args.mtp_chunk_size, losses.LOGITS_BUDGET_BYTES // per_row))
            with torch.no_grad():
                w = target_lm_head_weight
                teacher_ids = torch.cat([
                    (chunk.to(device = w.device, dtype = w.dtype) @ w.T).argmax(-1).to(labels.device)
                    for chunk in teacher_hidden.split(rows)
                ]).view_as(labels)
            labels = torch.where(labels != -100, teacher_ids, labels)
        loss_sum, n_tokens, n_correct = chunked_lm_loss(
            hidden.reshape(-1, hidden.shape[-1]),
            lm_head_weight,
            labels.reshape(-1),
            loss_type = args.mtp_loss_type,
            teacher_hidden = teacher_hidden,
            teacher_lm_head_weight = None if lm_head_weight is target_lm_head_weight else target_lm_head_weight,
            chunk_size = args.mtp_chunk_size,
        )
        n = max(int(n_tokens), 1)
        total = total + weight * loss_sum / (n if normalizer is None else normalizer)
        stats.append((float(loss_sum.detach()) / n, int(n_tokens), int(n_correct)))
    return total, stats


def token_average_scale(trainer, num_items_in_batch):
    """Scale Trainer.compute_loss applies to a loss when num_items_in_batch counts tokens across ranks (DDP's gradient
    mean cancels it); compute_loss overrides must match."""
    args = trainer.args
    if num_items_in_batch is None or not getattr(args, "average_tokens_across_devices", False):
        return 1
    return trainer.accelerator.num_processes if args.n_gpu <= 1 else args.n_gpu


class UnslothMTPTrainer(Trainer):
    """Trains a draft MTP head against a frozen target, one target pass per batch; model is the only trainable module, target_model stays frozen."""

    def __init__(self, model = None, args = None, target_model = None, **kwargs):
        if target_model is None:
            raise ValueError("Unsloth: UnslothMTPTrainer needs `target_model`, the frozen model the head drafts for.")
        if args is None:
            args = MTPTrainingArguments(output_dir = "mtp_outputs")
        if not isinstance(args, MTPTrainingArguments):
            raise TypeError("Unsloth: UnslothMTPTrainer needs MTPTrainingArguments.")
        if args.mtp_loss_type not in ("ce", "kl_div", "tv", "lk_hybrid"):
            raise ValueError(f"Unsloth: unknown mtp_loss_type {args.mtp_loss_type!r}.")
        if args.mtp_label_source not in ("data", "teacher"):
            raise ValueError(f"Unsloth: unknown mtp_label_source {args.mtp_label_source!r}.")
        self.target_model = target_model
        self.target_parts = resolve_text_parts(target_model)
        effective_lm_head_weight(self.target_parts["lm_head"])  # unsupported adapted heads fail here, not mid-run
        for p in target_model.parameters():
            p.requires_grad_(False)
        target_model.eval()
        if kwargs.get("data_collator") is None:
            tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
            tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
            pad = getattr(tokenizer, "pad_token_id", None)
            if pad is None:
                pad = getattr(tokenizer, "eos_token_id", 0) or 0
            kwargs["data_collator"] = MTPDataCollator(pad)
        super().__init__(model = model, args = args, **kwargs)
        # compute_loss normalizes by num_items_in_batch itself, so the Trainer must not also divide by GA.
        self.model_accepts_loss_kwargs = True
        self._mtp_log = []

    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        data_labels = inputs.get("labels", input_ids)
        if attention_mask is not None and bool((attention_mask[:, 0] == 0).any()):
            raise ValueError("Unsloth: UnslothMTPTrainer needs right-padded batches (left padding misaligns MTP positions).")
        head_module = getattr(model, "module", model)
        pre_norm = getattr(head_module, "h_source", "final") == "last_layer"
        captured = run_frozen_target(
            self.target_model, input_ids, attention_mask = attention_mask, layer_ids = [-1] if pre_norm else (),
        )
        param = next(model.parameters())
        final_hidden = captured["final"].to(device = param.device, dtype = param.dtype)
        head_hidden = captured[-1].to(device = param.device, dtype = param.dtype) if pre_norm else None
        normalizer = None
        if num_items_in_batch is not None:
            normalizer = torch.as_tensor(num_items_in_batch, device = param.device, dtype = torch.float32).clamp(min = 1)
        loss, stats = mtp_step_losses(
            model, self.target_parts, final_hidden, input_ids, attention_mask, data_labels, self.args,
            normalizer = normalizer, head_hidden = head_hidden,
        )
        if model.training:
            self._mtp_log.append(stats)
        loss = loss * token_average_scale(self, num_items_in_batch)
        return (loss, None) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys = None):
        # The head has no forward(), so the stock prediction_step cannot call model(**inputs).
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)

    def _load_from_checkpoint(self, resume_from_checkpoint, model = None):
        from .stitch import load_saved_mtp_head
        loaded = load_saved_mtp_head(resume_from_checkpoint)
        if loaded is None:
            return super()._load_from_checkpoint(resume_from_checkpoint, model)
        (model or self.model).load_state_dict(loaded[0], strict = True)

    def _load_best_model(self):
        from .stitch import load_saved_mtp_head
        loaded = load_saved_mtp_head(self.state.best_model_checkpoint)
        if loaded is None:
            return super()._load_best_model()
        self.model.load_state_dict(loaded[0], strict = True)

    def log(self, logs, *args, **kwargs):
        if self._mtp_log:
            depth = max(len(s) for s in self._mtp_log)
            for k in range(depth):
                rows = [s[k] for s in self._mtp_log if len(s) > k]
                n = sum(r[1] for r in rows)
                if n:
                    logs[f"mtp_loss_{k}"] = round(sum(r[0] * r[1] for r in rows) / n, 4)
                    logs[f"mtp_top1_{k}"] = round(sum(r[2] for r in rows) / n, 4)
            self._mtp_log = []
        return super().log(logs, *args, **kwargs)

    def _save(self, output_dir = None, state_dict = None):
        from .stitch import save_mtp_head
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok = True)
        head = self.model
        state_dict = state_dict if state_dict is not None else head.state_dict()
        save_mtp_head(output_dir, state_dict, getattr(head, "_unsloth_mtp_info", {}))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

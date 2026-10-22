# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("accelerate")  # TrainingArguments needs it; absent on MLX-only installs
import torch
from transformers import TrainingArguments

import unsloth_zoo.training_utils as training_utils


class _TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        self.config = SimpleNamespace(torch_dtype = torch.bfloat16, dtype = torch.bfloat16)
        self.forward_calls = 0

    def forward(self, input_ids, labels, n_items):
        self.forward_calls += 1
        x = torch.nn.functional.one_hot(input_ids % 4, 4).float()
        loss = self.proj(x).float().pow(2).sum() / n_items
        return SimpleNamespace(loss = loss)


def _collate(rows):
    ids = torch.tensor([r["input_ids"] for r in rows])
    return {"input_ids": ids, "labels": ids.clone()}


def _train(monkeypatch, tmp_path, n_rows, **args):
    # Route the loop's "cuda:0" tensors to CPU so it runs on any host.
    monkeypatch.setattr(training_utils, "DEVICE_TYPE", "xpu")
    monkeypatch.setattr(training_utils, "DEVICE_TYPE_TORCH", "cpu", raising = False)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *a, **k: self)
    model = _TinyLM()
    trainer = SimpleNamespace(
        args = TrainingArguments(
            output_dir = str(tmp_path), report_to = "none", dataloader_pin_memory = False,
            per_device_train_batch_size = 1, logging_steps = 1000, **args,
        ),
        model = model,
        train_dataset = [{"input_ids": [1, 2, 3]}] * n_rows,
        data_collator = _collate,
    )
    before = model.proj.weight.detach().clone()
    training_utils.unsloth_train(trainer)
    return model, before


def test_max_grad_norm_zero_disables_clipping_instead_of_zeroing_grads(monkeypatch, tmp_path):
    model, before = _train(monkeypatch, tmp_path, 4, max_steps = 2, max_grad_norm = 0.0, learning_rate = 1e-2)
    assert not torch.equal(model.proj.weight.detach(), before)


def test_max_steps_stops_before_an_extra_epoch(monkeypatch, tmp_path):
    # 5 rows, batch 1, GA 2: 3 steps per epoch (2 + 2 + 1 batches); max_steps = 3 gives 2 epochs.
    model, _ = _train(monkeypatch, tmp_path, 5, max_steps = 3, gradient_accumulation_steps = 2)
    assert model.forward_calls == 5

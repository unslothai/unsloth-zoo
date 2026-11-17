# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("accelerate")  # TrainingArguments needs it; absent on MLX-only installs
import torch
from transformers import TrainingArguments

import unsloth_zoo.training_utils as training_utils


class _OverflowOnceLM(torch.nn.Module):
    """float16 model whose first loss is inf; small fixed weights keep later scaled grads finite in fp16."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        torch.nn.init.constant_(self.proj.weight, 1e-2)
        torch.nn.init.zeros_(self.proj.bias)
        self.config = SimpleNamespace(torch_dtype = torch.float16, dtype = torch.float16)
        self.forward_calls = 0

    def forward(self, input_ids, labels, n_items):
        self.forward_calls += 1
        x = torch.nn.functional.one_hot(input_ids % 4, 4).float()
        loss = self.proj(x).float().pow(2).sum() / n_items
        if self.forward_calls == 1:
            loss = loss * float("inf")
        return SimpleNamespace(loss = loss)


def _collate(rows):
    ids = torch.tensor([r["input_ids"] for r in rows])
    return {"input_ids": ids, "labels": ids.clone()}


# Fused AdamW is always called and skips inside its kernel, so only the scale shows the skip.
@pytest.mark.parametrize("optim", ["adamw_torch", "adamw_torch_fused"])
def test_lr_schedule_does_not_advance_on_a_skipped_scaler_step(monkeypatch, tmp_path, optim):
    monkeypatch.setattr(training_utils, "DEVICE_TYPE", "xpu")
    monkeypatch.setattr(training_utils, "DEVICE_TYPE_TORCH", "cpu", raising = False)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *a, **k: self)

    scheduler_steps = []
    real_get_scheduler = training_utils.transformers_get_scheduler

    def spy_get_scheduler(**kwargs):
        scheduler = real_get_scheduler(**kwargs)
        real_step = scheduler.step

        def counted_step(*args, **kw):
            scheduler_steps.append(1)
            return real_step(*args, **kw)

        scheduler.step = counted_step
        return scheduler

    monkeypatch.setattr(training_utils, "transformers_get_scheduler", spy_get_scheduler)

    model = _OverflowOnceLM()
    trainer = SimpleNamespace(
        args = TrainingArguments(
            output_dir = str(tmp_path), report_to = "none", dataloader_pin_memory = False,
            per_device_train_batch_size = 1, logging_steps = 1000, max_steps = 2,
            learning_rate = 1e-2, optim = optim,
        ),
        model = model,
        train_dataset = [{"input_ids": [1, 2, 3]}] * 2,
        data_collator = _collate,
    )
    training_utils.unsloth_train(trainer)

    assert model.forward_calls == 2
    assert len(scheduler_steps) == 1

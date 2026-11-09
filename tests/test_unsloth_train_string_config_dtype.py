# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import TrainingArguments

from unsloth_zoo import training_utils


class _TinyLM(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        # Unsloth stores the config dtype as a string on some transformers versions.
        self.config = SimpleNamespace(torch_dtype = dtype, dtype = dtype)

    def forward(self, input_ids, labels, n_items):
        x = torch.nn.functional.one_hot(input_ids % 4, 4).float()
        return SimpleNamespace(loss = self.proj(x).float().pow(2).sum() / n_items)


def _collate(rows):
    ids = torch.tensor([r["input_ids"] for r in rows])
    return {"input_ids": ids, "labels": ids.clone()}


def _autocast_dtype(monkeypatch, tmp_path, config_dtype):
    monkeypatch.setattr(training_utils, "DEVICE_TYPE", "xpu")
    monkeypatch.setattr(training_utils, "DEVICE_TYPE_TORCH", "cpu", raising = False)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *a, **k: self)
    seen = []
    real = torch.amp.autocast

    def recording_autocast(*args, **kwargs):
        seen.append(kwargs.get("dtype"))
        return real(*args, **kwargs)

    monkeypatch.setattr(torch.amp, "autocast", recording_autocast)
    trainer = SimpleNamespace(
        args = TrainingArguments(
            output_dir = str(tmp_path), report_to = "none", dataloader_pin_memory = False,
            per_device_train_batch_size = 1, logging_steps = 1000, max_steps = 1,
        ),
        model = _TinyLM(config_dtype),
        train_dataset = [{"input_ids": [1, 2, 3]}] * 2,
        data_collator = _collate,
    )
    training_utils.unsloth_train(trainer)
    return seen[0]


def test_string_float16_config_dtype_uses_fp16_autocast(monkeypatch, tmp_path):
    assert _autocast_dtype(monkeypatch, tmp_path, "float16") == torch.float16


def test_torch_float16_config_dtype_uses_fp16_autocast(monkeypatch, tmp_path):
    assert _autocast_dtype(monkeypatch, tmp_path, torch.float16) == torch.float16

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.

"""unsloth_train must feed batches to the model's input device, not a hard-coded cuda:0; otherwise a model on another
GPU or on CPU gets wrong-device batches and no-CUDA hosts crash before step 0."""

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
        self.config = SimpleNamespace(torch_dtype = torch.float32, dtype = torch.float32)
        self.embed = torch.nn.Embedding(16, 8)
        self.head = torch.nn.Linear(8, 16)
        self.seen_devices = []

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, labels, n_items = None):
        self.seen_devices.append((input_ids.device, labels.device))
        logits = self.head(self.embed(input_ids)).float()
        loss = torch.nn.functional.cross_entropy(logits.view(-1, 16), labels.view(-1))
        return SimpleNamespace(loss = loss)


def _collate(rows):
    ids = torch.tensor([row["input_ids"] for row in rows])
    return {"input_ids": ids, "labels": ids.clone()}


def test_batches_follow_the_model_device(tmp_path):
    model = _TinyLM()
    trainer = SimpleNamespace(
        args = TrainingArguments(
            output_dir = str(tmp_path),
            max_steps = 2,
            per_device_train_batch_size = 2,
            report_to = "none",
            use_cpu = True,
            dataloader_pin_memory = False,
        ),
        model = model,
        train_dataset = [{"input_ids": [1, 2, 3, 4]}] * 4,
        data_collator = _collate,
    )
    training_utils.unsloth_train(trainer)

    assert model.seen_devices, "the model never ran"
    expected = model.embed.weight.device
    assert all(ids == expected and labels == expected for ids, labels in model.seen_devices)


def test_offloaded_embedding_uses_the_hook_execution_device(tmp_path):
    # With accelerate offloading the embedding weight is meta between forwards, so batches must go to the hook's execution device, never to meta.
    model = _TinyLM()
    meta_embed = torch.nn.Embedding(16, 8, device = "meta")
    meta_embed._hf_hook = SimpleNamespace(execution_device = torch.device("cpu"))
    model.get_input_embeddings = lambda: meta_embed
    trainer = SimpleNamespace(
        args = TrainingArguments(
            output_dir = str(tmp_path),
            max_steps = 1,
            per_device_train_batch_size = 2,
            report_to = "none",
            use_cpu = True,
            dataloader_pin_memory = False,
        ),
        model = model,
        train_dataset = [{"input_ids": [1, 2, 3, 4]}] * 2,
        data_collator = _collate,
    )
    training_utils.unsloth_train(trainer)
    assert model.seen_devices and all(ids.type == "cpu" for ids, _ in model.seen_devices)

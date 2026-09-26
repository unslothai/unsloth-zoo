# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("accelerate")  # TrainingArguments needs it; absent on MLX-only installs
import torch
from transformers import TrainingArguments

import unsloth_zoo.training_utils as training_utils


class _Stop(Exception):
    pass


def _scheduler_kwargs(monkeypatch, tmp_path, **args):
    seen = {}

    def fake_get_scheduler(**kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(training_utils, "transformers_get_scheduler", fake_get_scheduler)
    trainer = SimpleNamespace(
        args = TrainingArguments(
            output_dir = str(tmp_path), max_steps = 2, report_to = "none", **args
        ),
        model = torch.nn.Linear(4, 4),
        train_dataset = [{"input_ids": [1, 2, 3]}] * 4,
        data_collator = lambda rows: rows,
    )
    with pytest.raises(_Stop):
        training_utils.unsloth_train(trainer)
    return seen


def test_default_lr_scheduler_kwargs_reach_the_scheduler(monkeypatch, tmp_path):
    kwargs = _scheduler_kwargs(monkeypatch, tmp_path)
    assert set(kwargs) == {"name", "optimizer", "num_warmup_steps", "num_training_steps"}


def test_given_lr_scheduler_kwargs_are_still_passed(monkeypatch, tmp_path):
    kwargs = _scheduler_kwargs(
        monkeypatch,
        tmp_path,
        lr_scheduler_type = "cosine_with_restarts",
        lr_scheduler_kwargs = {"num_cycles": 3},
    )
    assert kwargs["num_cycles"] == 3

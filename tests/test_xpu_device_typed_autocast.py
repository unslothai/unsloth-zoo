# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "unsloth_zoo"


def test_nested_quant_state_offset_follows_device():
    F = pytest.importorskip("bitsandbytes.functional")
    vllm_utils = pytest.importorskip("unsloth_zoo.vllm_utils")
    _, qs = F.quantize_4bit(torch.randn(64, 64), compress_statistics = True, quant_type = "nf4")
    state = vllm_utils.from_dict.__func__(F.QuantState, qs.as_dict(packed = True), device = torch.device("cpu"))
    assert state.offset.device.type == "cpu"
    assert state.offset.dtype == torch.float32
    assert torch.equal(state.offset, qs.offset.float())


def test_gemma3n_fp32_region_overrides_enclosing_autocast():
    from unsloth_zoo.temporary_patches import gemma3n

    linear = torch.nn.Linear(4, 4)
    with torch.autocast("cpu", dtype = torch.bfloat16):
        with gemma3n._fp32_autocast("cpu"):
            out = linear(torch.randn(2, 4))
    assert out.dtype == torch.float32
    # A backend autocast does not know falls back to no context instead of raising.
    with gemma3n._fp32_autocast("not_a_backend"):
        pass


def test_gemma3n_has_no_cuda_only_autocast():
    tree = ast.parse((ROOT / "temporary_patches" / "gemma3n.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "autocast":
            for kw in node.keywords:
                assert not (kw.arg == "device_type" and isinstance(kw.value, ast.Constant)), ast.unparse(node)


class _Stop(Exception):
    pass


@pytest.mark.parametrize(
    "device_type, device_type_torch, expected",
    [("xpu", "xpu", "xpu"), ("npu", "npu", "npu"), ("cuda", "cuda", "cuda"), ("hip", "cuda", "cuda")],
)
def test_unsloth_train_autocasts_on_the_accelerator(monkeypatch, tmp_path, device_type, device_type_torch, expected):
    pytest.importorskip("accelerate")
    from transformers import TrainingArguments
    import unsloth_zoo.training_utils as training_utils

    seen = {}

    def fake_autocast(device_type, **kwargs):
        seen["device_type"] = device_type
        raise _Stop

    monkeypatch.setattr(training_utils, "DEVICE_TYPE", device_type)
    monkeypatch.setattr(training_utils, "DEVICE_TYPE_TORCH", device_type_torch, raising = False)
    monkeypatch.setattr(torch.amp, "autocast", fake_autocast)
    model = torch.nn.Linear(4, 4)
    model.config = SimpleNamespace(torch_dtype = torch.bfloat16, dtype = torch.bfloat16)
    trainer = SimpleNamespace(
        args = TrainingArguments(output_dir = str(tmp_path), max_steps = 2, report_to = "none"),
        model = model,
        train_dataset = [{"input_ids": [1, 2, 3]}] * 4,
        data_collator = lambda rows: rows,
    )
    with pytest.raises(_Stop):
        training_utils.unsloth_train(trainer)
    assert seen["device_type"] == expected

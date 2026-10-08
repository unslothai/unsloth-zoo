# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.

from __future__ import annotations

import ast
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "unsloth_zoo"


def _patched_from_dict():
    """The patched `QuantState.from_dict`, sliced by AST so CPU CI runs it without bitsandbytes."""
    tree = ast.parse((ROOT / "vllm_utils.py").read_text(encoding = "utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "from_dict")
    fn.decorator_list = []
    ns = {"torch": torch, "os": __import__("os"), "Dict": dict, "Any": object,
          "unpack_tensor_to_dict": lambda t: dict(t.extra)}
    exec(compile(ast.Module([fn], []), "vllm_utils.py", "exec"), ns)
    return ns["from_dict"]


class _QuantState:
    valid_qs_type_keys = ["bitsandbytes__nf4"]
    valid_qs_keys = {"absmax", "quant_map", "nested_absmax", "nested_quant_map", "quant_state",
                     "quant_type", "blocksize", "dtype", "shape", "nested_blocksize", "nested_dtype",
                     "nested_offset"}

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_nested_quant_state_offset_follows_device():
    packed = torch.zeros(1)
    packed.extra = {"quant_type": "nf4", "blocksize": 64, "dtype": "float16", "shape": (64, 64),
                    "nested_blocksize": 256, "nested_dtype": "float32", "nested_offset": 0.25}
    qs_dict = {"absmax": torch.zeros(4, dtype = torch.uint8), "quant_map": torch.zeros(16),
               "nested_absmax": torch.zeros(1), "nested_quant_map": torch.zeros(256),
               "quant_state.bitsandbytes__nf4": packed}
    state = _patched_from_dict()(_QuantState, qs_dict, device = torch.device("cpu"))
    assert state.offset.device.type == "cpu"
    assert state.offset.dtype == torch.float32
    assert state.offset.item() == 0.25


def test_gemma3n_region_keeps_enclosing_non_cuda_autocast():
    from unsloth_zoo.temporary_patches import gemma3n

    # bf16 weights, fp32 input, as in the patched forwards: an fp32 "region" off CUDA only disables
    # the enclosing autocast and the matmul then fails on mixed dtypes.
    linear = torch.nn.Linear(4, 4).to(torch.bfloat16)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with torch.autocast("cpu", dtype = torch.bfloat16):
            with gemma3n._fp32_autocast("cpu"):
                out = linear(torch.randn(2, 4, dtype = torch.float32))
    assert out.dtype == torch.bfloat16
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

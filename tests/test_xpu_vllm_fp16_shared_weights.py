# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""vLLM refusing bf16 on an Intel Arc A770 must not leave HF on bf16: load_vllm switches the engine to fp16, but
released unsloth (no xpu_vllm_refuses_bf16) still passes dtype = bf16 to convert_vllm_to_huggingface and post_patch;
both must follow the shared fp16 weights."""

import ast
import inspect
import types

import pytest
import torch

pytest.importorskip("bitsandbytes")
pytest.importorskip("peft")


@pytest.fixture
def vu():
    try:
        from unsloth_zoo import vllm_utils
    except ImportError as e:
        pytest.skip(f"vllm_utils does not import here: {e}")
    return vllm_utils


@pytest.mark.parametrize(
    "device_type, requested, embed, expected",
    [
        ("xpu", torch.bfloat16, torch.float16, torch.float16),   # vLLM refused bf16 and ran fp16
        ("xpu", torch.bfloat16, torch.bfloat16, torch.bfloat16),
        ("xpu", torch.float16, torch.float16, torch.float16),
        ("xpu", torch.float16, torch.bfloat16, torch.bfloat16),   # loader fell back to fp16, vLLM's XPU runs bf16
        ("xpu", torch.float16, torch.float32, torch.float16),     # only fp16 / bf16 embeddings are followed
        ("xpu", torch.bfloat16, None, torch.bfloat16),            # nothing to read: keep the request
        ("cuda", torch.bfloat16, torch.float16, torch.bfloat16),  # only the XPU fallback is followed
        ("cuda", torch.float16, torch.bfloat16, torch.float16),
    ],
)
def test_shared_weight_dtype_follows_the_vllm_embedding(vu, monkeypatch, device_type, requested, embed, expected):
    monkeypatch.setattr(vu, "DEVICE_TYPE", device_type)
    state = {} if embed is None else {"model.embed_tokens.weight": torch.zeros(4, 2, dtype = embed)}
    assert vu._shared_weight_dtype(state, requested) is expected


def test_shared_weight_dtype_reads_vision_embedding_and_parameters(vu, monkeypatch):
    monkeypatch.setattr(vu, "DEVICE_TYPE", "xpu")
    weight = torch.nn.Parameter(torch.zeros(4, 2, dtype = torch.float16), requires_grad = False)
    state = {"model.language_model.embed_tokens.weight": weight}
    assert vu._shared_weight_dtype(state, torch.bfloat16) is torch.float16


def test_shared_weight_dtype_reads_nemotron_h_embeddings(vu, monkeypatch):
    # get_vllm_state_dict stores Nemotron-H's embedding as `embeddings`, not `embed_tokens`.
    monkeypatch.setattr(vu, "DEVICE_TYPE", "xpu")
    state = {"model.embeddings.weight": torch.zeros(4, 2, dtype = torch.float16)}
    assert vu._shared_weight_dtype(state, torch.bfloat16) is torch.float16


def test_vllm_automatic_dtype_judges_the_engine_target():
    from unsloth_zoo import vllm_utils as vu
    source = inspect.getsource(vu.load_vllm)
    assert "_xpu_bf16_supported(get_target_device())" in source
    assert "device = get_target_device()" in source


def test_convert_builds_in_the_shared_dtype_and_marks_the_model(vu):
    source = inspect.getsource(vu.convert_vllm_to_huggingface)
    tree = ast.parse(source)
    func = tree.body[0]
    first = [node for node in func.body if not isinstance(node, (ast.FunctionDef, ast.Expr))][0]
    # The dtype is settled before the config or the empty model see it.
    assert ast.unparse(first) == "dtype = _shared_weight_dtype(quant_state_dict, dtype)"
    assert "new_model._unsloth_vllm_dtype = dtype" in source
    # The rebuild still runs under inference mode, and the dtype probe stays a plain function.
    assert source.lstrip().startswith("@torch.inference_mode\n")
    assert not inspect.getsource(vu._shared_weight_dtype).lstrip().startswith("@")


def _model():
    model = torch.nn.Module()
    model.config = types.SimpleNamespace(to_dict = lambda: {})
    model.layer = torch.nn.Linear(4, 4).to(torch.float16)
    model.leftover = torch.nn.Linear(4, 4).to(torch.bfloat16)
    return model


def _patch(model, dtype):
    from unsloth_zoo.patching_utils import patch_model_and_tokenizer

    patch_model_and_tokenizer(model, None, downcast_rope = False, fix_embeddings = False, correct_dtype = dtype)


def test_post_patch_follows_vllms_fp16_over_a_bf16_request():
    model = _model()
    model._unsloth_vllm_dtype = torch.float16
    _patch(model, torch.bfloat16)
    assert model.leftover.weight.dtype == torch.float16
    assert model.layer.weight.dtype == torch.float16


def test_post_patch_follows_vllms_bf16_over_an_fp16_request(monkeypatch):
    # A released unsloth picked fp16 off an unused emulating XPU while vLLM's XPU runs bf16.
    import unsloth_zoo.patching_utils as pu
    monkeypatch.setattr(pu, "DEVICE_TYPE", "xpu")
    model = _model()
    model._unsloth_vllm_dtype = torch.bfloat16
    shared = model.leftover.weight.data
    _patch(model, torch.float16)
    # The bf16 weights vLLM shares are not cast to fp16 (which would detach and duplicate them).
    assert model.leftover.weight.dtype == torch.bfloat16
    assert model.leftover.weight.data.data_ptr() == shared.data_ptr()


def test_post_patch_fp16_request_off_xpu_is_unchanged(monkeypatch):
    import unsloth_zoo.patching_utils as pu
    monkeypatch.setattr(pu, "DEVICE_TYPE", "cuda")
    model = _model()
    model._unsloth_vllm_dtype = torch.bfloat16
    _patch(model, torch.float16)
    assert model.leftover.weight.dtype == torch.float16


def test_post_patch_keeps_bf16_without_the_vllm_fallback():
    model = _model()
    _patch(model, torch.bfloat16)
    assert model.leftover.weight.dtype == torch.bfloat16
    model = _model()
    model._unsloth_vllm_dtype = torch.bfloat16
    _patch(model, torch.bfloat16)
    assert model.leftover.weight.dtype == torch.bfloat16

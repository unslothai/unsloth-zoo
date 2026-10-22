# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""fp16 fused lm_head + CE forward must see the GradScaler: the kernel saves d(loss)/d(hidden) in the forward, unscaled it underflows."""

import inspect
import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

SCALE = 65536.0


class _FakeScaler:
    def get_scale(self):
        return SCALE


def _hidden_grad(model, ids, scaler):
    seen = {}

    def hook(module, args, output):
        output.register_hook(lambda g: seen.__setitem__("g", g.float()))

    handle = model.model.norm.register_forward_hook(hook)
    if scaler is not None:
        model.accelerator_scaler = scaler
    out = model(input_ids = ids, labels = ids)
    handle.remove()
    assert out.logits.numel() == 0, "the fused CE forward did not run"
    scale = SCALE if scaler is not None else 1.0
    (out.loss * scale).backward()
    return seen["g"] / scale


def test_fused_forward_fp16_uses_accelerator_scaler(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FUSED_FORWARD", "1")
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    from transformers import LlamaConfig, LlamaForCausalLM
    from unsloth_zoo.fused_losses import forward_install as fi

    original = LlamaForCausalLM.forward
    was_patched = LlamaForCausalLM.__qualname__ in fi._PATCHED
    try:
        assert fi.install_for_class(LlamaForCausalLM)
        torch.manual_seed(0)
        config = LlamaConfig(
            vocab_size = 2048, hidden_size = 64, intermediate_size = 128, num_hidden_layers = 1,
            num_attention_heads = 2, num_key_value_heads = 1, tie_word_embeddings = False,
            attn_implementation = "eager",
        )
        ref = LlamaForCausalLM(config).float()
        half = LlamaForCausalLM(config).half()
        half.load_state_dict(ref.state_dict())
        ids = torch.randint(0, config.vocab_size, (8, 1024), generator = torch.Generator().manual_seed(0))

        g_ref = _hidden_grad(ref, ids, None)
        g_half = _hidden_grad(half, ids, _FakeScaler())
        assert torch.isfinite(g_half).all()
        rel = ((g_half - g_ref).norm() / g_ref.norm()).item()
        assert rel < 5e-3, rel
    finally:
        if not was_patched:
            LlamaForCausalLM.forward = original
            with fi._REGISTRY_LOCK:
                fi._PATCHED.pop(LlamaForCausalLM.__qualname__, None)


def test_fused_forward_bf16_without_scaler_is_unchanged(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FUSED_FORWARD", "1")
    from unsloth_zoo.fused_losses import forward_adapter as fa

    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return torch.zeros(())

    monkeypatch.setattr(fa, "unsloth_fused_ce_loss", fake)
    head = torch.nn.Linear(4, 8, bias = False)
    h = torch.zeros(1, 3, 4)
    labels = torch.zeros(1, 3, dtype = torch.long)
    fa.unsloth_fused_lm_head_loss(h, head, labels, accelerator_scaler = None)
    assert "scaling" not in seen and "accelerator_scaler" not in seen
    fa.unsloth_fused_lm_head_loss(h, head, labels, accelerator_scaler = _FakeScaler())
    assert isinstance(seen["scaling"], _FakeScaler)
    fa.unsloth_fused_lm_head_loss(h, head, labels, scaling = 2.0, accelerator_scaler = _FakeScaler())
    assert seen["scaling"] == 2.0
    fa.unsloth_fused_lm_head_loss(h, head, labels, scaling = 0.0, accelerator_scaler = _FakeScaler())
    assert seen["scaling"] == 0.0


class _TinyHead(torch.nn.Module):
    def __init__(self, train_head):
        super().__init__()
        from transformers.loss.loss_utils import ForCausalLMLoss

        self.config = SimpleNamespace(vocab_size = 2048)
        self.lm_head = torch.nn.Linear(16, self.config.vocab_size, bias = False).half()
        self.lm_head.requires_grad_(train_head)
        self.loss_function = ForCausalLMLoss
        self.accelerator_scaler = _FakeScaler()

    def forward(self, hidden_states, labels = None, **kwargs):
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config.vocab_size, **kwargs)
        return loss, logits


@pytest.mark.parametrize("train_head", [False, True])
def test_ast_fp16_gradients_match_scaled_reference(monkeypatch, train_head):
    from unsloth_zoo.fused_losses.ast_rewriter import rewrite_forward_source
    from unsloth_zoo.fused_losses import forward_adapter as fa, forward_install as fi

    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    torch.manual_seed(0)
    model = _TinyHead(train_head)
    hidden = torch.randn(2, 128, 16, dtype = torch.float16, requires_grad = True)
    labels = torch.randint(0, model.config.vocab_size, (2, 128))
    inputs = (hidden, model.lm_head.weight) if train_head else (hidden,)

    reference_loss, _ = model(hidden, labels)
    reference_grads = torch.autograd.grad(reference_loss * SCALE, inputs)

    source, capture = rewrite_forward_source(inspect.getsource(_TinyHead.forward))
    assert source is not None and capture is not None
    namespace = dict(
        os = os, unsloth_fused_lm_head_loss = fa.unsloth_fused_lm_head_loss,
        EMPTY_LOGITS = fa.EMPTY_LOGITS, _can_fuse_loss = fi._can_fuse_loss,
    )
    exec(compile(source, "<fp16-scaler-test>", "exec"), namespace)
    # One eager chunk makes the arithmetic match the reference exactly.
    loss, logits = namespace["forward"](
        model, hidden, labels, torch_compile = False, n_chunks = 1,
    )
    assert logits.numel() == 0, "the fused CE forward did not run"
    grads = torch.autograd.grad(loss * SCALE, inputs)

    torch.testing.assert_close(loss, reference_loss, rtol = 0, atol = 0)
    for actual, expected in zip(grads, reference_grads):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol = 0, atol = 0)

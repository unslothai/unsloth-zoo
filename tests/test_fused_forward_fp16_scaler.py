# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""fp16 fused lm_head + CE forward must see the GradScaler: the kernel saves d(loss)/d(hidden) in the forward, unscaled it underflows."""

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

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""HF output kwargs (output_attentions / output_hidden_states) must not push the fused lm_head CE off the chunked
kernel (_use_chunked_ce) onto the torch.func fallback."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _route(monkeypatch):
    from unsloth_zoo.fused_losses import cross_entropy_loss as ce
    from unsloth_zoo.fused_losses import fused_linear as fl

    seen = {"torch_func": 0}
    orig = ce.UnslothFusedLoss.forward

    def forward(ctx, *a, **k):
        seen["torch_func"] += 1
        return orig(ctx, *a, **k)

    monkeypatch.setattr(ce.UnslothFusedLoss, "forward", staticmethod(forward))
    return seen, fl.CALLS


@pytest.mark.parametrize("extra", [{}, {"output_attentions": False, "output_hidden_states": False}])
def test_lm_head_loss_hf_output_kwargs_keep_kernel_route(monkeypatch, extra):
    from unsloth_zoo.fused_losses.forward_adapter import unsloth_fused_lm_head_loss

    seen, calls = _route(monkeypatch)
    torch.manual_seed(0)
    head = torch.nn.Linear(32, 257, bias = False)
    hidden = torch.randn(2, 9, 32, requires_grad = True)
    labels = torch.randint(0, 257, (2, 9))
    labels[0, :3] = -100
    before = calls["ce"]
    loss = unsloth_fused_lm_head_loss(hidden, head, labels, vocab_size = 257, num_items_in_batch = 11, **extra)
    loss.backward()
    assert calls["ce"] - before == 1, "the chunked fused-linear CE kernel did not run"
    assert seen["torch_func"] == 0, "fell back to the torch.func UnslothFusedLoss path"

    ref_h = hidden.detach().clone().requires_grad_(True)
    logits = head(ref_h).float()[:, :-1].reshape(-1, 257)
    tgt = labels[:, 1:].reshape(-1)
    ref = torch.nn.functional.cross_entropy(logits, tgt, ignore_index = -100, reduction = "sum") / 11
    ref.backward()
    torch.testing.assert_close(loss.detach(), ref.detach(), rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(hidden.grad, ref_h.grad, rtol = 1e-4, atol = 1e-6)


def test_installed_forward_with_output_kwargs_keeps_kernel_route(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FUSED_FORWARD", "1")
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    from transformers import LlamaConfig, LlamaForCausalLM
    from unsloth_zoo.fused_losses import forward_install as fi

    original = LlamaForCausalLM.forward
    was_patched = LlamaForCausalLM.__qualname__ in fi._PATCHED
    try:
        assert fi.install_for_class(LlamaForCausalLM)
        torch.manual_seed(0)
        config = LlamaConfig(vocab_size = 512, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 1,
                             num_attention_heads = 2, num_key_value_heads = 1, tie_word_embeddings = False,
                             attn_implementation = "eager")
        model = LlamaForCausalLM(config).float()
        ids = torch.randint(0, 512, (2, 16), generator = torch.Generator().manual_seed(0))
        seen, calls = _route(monkeypatch)
        before = calls["ce"]
        out = model(input_ids = ids, labels = ids, output_attentions = False, output_hidden_states = False)
        assert out.logits.numel() == 0, "the fused CE forward did not run"
        out.loss.backward()
        assert calls["ce"] - before == 1, "the chunked fused-linear CE kernel did not run"
        assert seen["torch_func"] == 0, "fell back to the torch.func UnslothFusedLoss path"
    finally:
        LlamaForCausalLM.forward = original
        if not was_patched:
            fi._PATCHED.pop(LlamaForCausalLM.__qualname__, None)

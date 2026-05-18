# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for the fused lm_head + cross_entropy auto-installer.

Covers:
  - AST rewriter recognises the canonical HF triplet shape (keyword form,
    positional vocab_size, `.float()` wrapper, no-`loss = None` initialiser).
  - AST rewriter declines on non-matching forwards (no triplet, missing
    if-labels block, missing loss_function call).
  - install_for_class:
      * no-op when UNSLOTH_FUSED_FORWARD is off
      * patches a synthetic *ForCausalLM whose forward matches the triplet
      * leaves a hand-crafted bespoke forward in _UNMATCHED
      * is idempotent
  - Numerical equivalence of the rewritten forward vs the original at
    small shapes (mean MSE under 1e-4 on bf16 -> fp32).
"""

from __future__ import annotations

import os
import sys
import types

import pytest


# Reset module state between tests so install registries don't bleed.
@pytest.fixture
def fresh_install():
    from unsloth_zoo.fused_losses import forward_install as fi
    with fi._REGISTRY_LOCK:
        fi._PATCHED.clear()
        fi._UNMATCHED.clear()
        fi._FAILED.clear()
        fi._CANONICAL_FORWARDS.clear()
    yield fi
    with fi._REGISTRY_LOCK:
        fi._PATCHED.clear()
        fi._UNMATCHED.clear()
        fi._FAILED.clear()
        fi._CANONICAL_FORWARDS.clear()


@pytest.fixture
def enable_env(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FUSED_FORWARD", "1")


# ---------------------------------------------------------------------------
# AST rewriter unit tests
# ---------------------------------------------------------------------------


CANONICAL_KW_SRC = """
def forward(self, input_ids=None, labels=None, logits_to_keep=0, **kwargs):
    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])
    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
    return (loss, logits)
"""

CANONICAL_POS_SRC = """
def forward(self, input_ids=None, labels=None, **kwargs):
    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs.last_hidden_state
    lm_logits = self.lm_head(hidden_states).float()
    loss = None
    if labels is not None:
        loss = self.loss_function(lm_logits, labels, self.config.vocab_size, **kwargs)
    return (loss, lm_logits)
"""

NON_CANONICAL_SRC = """
def forward(self, input_ids=None, labels=None, **kwargs):
    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs.last_hidden_state
    logits = self.lm_head(hidden_states)
    if labels is not None:
        loss_fct = object()  # legacy CrossEntropyLoss path
        loss = loss_fct(logits, labels)
    else:
        loss = None
    return (loss, logits)
"""


def test_ast_rewriter_matches_keyword_form():
    from unsloth_zoo.fused_losses.ast_rewriter import rewrite_forward_source
    new_src, cap = rewrite_forward_source(CANONICAL_KW_SRC)
    assert new_src is not None
    assert cap is not None
    assert cap.head_attr == "lm_head"
    assert cap.logits_name == "logits"
    assert "unsloth_fused_lm_head_loss" in new_src
    assert "EMPTY_LOGITS" in new_src
    # self.loss_function appears exactly once: on the UNSLOTH_RETURN_LOGITS
    # opt-in path, where we materialise logits via the original RHS and
    # route the loss through it to avoid a second lm_head matmul.
    assert new_src.count("self.loss_function") == 1
    # Labels branch carries the UNSLOTH_RETURN_LOGITS opt-in; the hidden-
    # states branch is intentionally absent (handled by the compiled
    # forward in unsloth_zoo/compiler.py).
    assert "UNSLOTH_RETURN_LOGITS" in new_src
    assert "UNSLOTH_RETURN_HIDDEN_STATES" not in new_src


def test_ast_rewriter_matches_positional_with_float_wrapper():
    from unsloth_zoo.fused_losses.ast_rewriter import rewrite_forward_source
    new_src, cap = rewrite_forward_source(CANONICAL_POS_SRC)
    assert new_src is not None
    assert cap is not None
    assert cap.head_attr == "lm_head"
    assert cap.logits_name == "lm_logits"
    assert "unsloth_fused_lm_head_loss" in new_src


def test_ast_rewriter_declines_non_canonical():
    from unsloth_zoo.fused_losses.ast_rewriter import rewrite_forward_source
    new_src, cap = rewrite_forward_source(NON_CANONICAL_SRC)
    assert new_src is None
    assert cap is None


COHERE_REBINDING_SRC = """
def forward(self, input_ids=None, labels=None, **kwargs):
    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs.last_hidden_state
    logits = self.lm_head(hidden_states)
    logits = logits * self.logit_scale
    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
    return (loss, logits)
"""


def test_ast_rewriter_declines_when_logits_rebound():
    # Cohere-style `logits = logits * self.logit_scale` between lm_head and
    # the if-labels block: removing the lm_head call would leave the
    # rebinding referencing an undefined `logits`. The rewriter must refuse.
    from unsloth_zoo.fused_losses.ast_rewriter import rewrite_forward_source
    new_src, cap = rewrite_forward_source(COHERE_REBINDING_SRC)
    assert new_src is None
    assert cap is None


GEMMA_SOFTCAP_SRC = """
def forward(self, input_ids=None, labels=None, **kwargs):
    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs.last_hidden_state
    logits = self.lm_head(hidden_states)
    if self.final_logit_softcapping is not None:
        logits = logits / self.final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * self.final_logit_softcapping
    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
    return (loss, logits)
"""


def test_ast_rewriter_declines_when_intermediate_touches_logits():
    # Gemma-style softcap mutates logits between lm_head and the labels-if.
    # Wholesale rewriting would skip that step and feed un-softcapped logits
    # to the fused loss; refuse and let the backstop handle it.
    from unsloth_zoo.fused_losses.ast_rewriter import rewrite_forward_source
    new_src, cap = rewrite_forward_source(GEMMA_SOFTCAP_SRC)
    assert new_src is None
    assert cap is None


CSM_ALIASED_LABELS_SRC = """
def forward(self, input_ids=None, labels=None, backbone_labels=None, **kwargs):
    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs.last_hidden_state
    logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=backbone_labels, vocab_size=self.config.vocab_size, **kwargs)
    return (loss, logits)
"""


def test_ast_rewriter_declines_when_labels_aliased():
    # CSM-style: gates on `labels is not None` but passes a different
    # aliased name to loss_function. Wholesale rewrite would forward the
    # wrong tensor; refuse.
    from unsloth_zoo.fused_losses.ast_rewriter import rewrite_forward_source
    new_src, cap = rewrite_forward_source(CSM_ALIASED_LABELS_SRC)
    assert new_src is None
    assert cap is None


MULTISTMT_LABEL_BRANCH_SRC = """
def forward(self, input_ids=None, labels=None, **kwargs):
    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs.last_hidden_state
    logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        aux_loss = self.aux_loss_coef * compute_aux(outputs)
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        loss = loss + aux_loss
    return (loss, logits)
"""


def test_ast_rewriter_declines_non_trivial_labels_branch():
    # MoE-style auxiliary loss inside the labels branch would be silently
    # dropped by a wholesale rewrite. The rewriter must refuse.
    from unsloth_zoo.fused_losses.ast_rewriter import rewrite_forward_source
    new_src, cap = rewrite_forward_source(MULTISTMT_LABEL_BRANCH_SRC)
    assert new_src is None
    assert cap is None


EXTRA_LOSS_KW_SRC = """
def forward(self, input_ids=None, labels=None, **kwargs):
    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs.last_hidden_state
    logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        loss = self.loss_function(
            logits=logits, labels=labels, vocab_size=self.config.vocab_size,
            num_items_in_batch=kwargs.get("num_items_in_batch"),
        )
    return (loss, logits)
"""


def test_ast_rewriter_forwards_explicit_extra_kwargs():
    # Bloom-style: loss_function gets explicit `num_items_in_batch=...` even
    # though there's no **kwargs unpack. The rewriter must preserve that
    # kwarg in the call to unsloth_fused_lm_head_loss.
    from unsloth_zoo.fused_losses.ast_rewriter import rewrite_forward_source
    new_src, cap = rewrite_forward_source(EXTRA_LOSS_KW_SRC)
    assert new_src is not None
    assert cap is not None
    assert ("num_items_in_batch", ) == tuple(name for name, _ in cap.extra_loss_kws)
    assert "num_items_in_batch=" in new_src


# ---------------------------------------------------------------------------
# install_for_class
# ---------------------------------------------------------------------------


_SYNTH_COUNTER = 0


def _make_synthetic_class(forward_src: str, name: str = "SyntheticForCausalLM"):
    """Build a class whose forward source is recoverable via inspect.getsource.

    `inspect.getsource` relies on `linecache`. Exec'd functions without a
    real file backing return OSError, which is what the installer falls
    back on. To exercise the rewriter we register a unique synthetic file
    name with `linecache` and compile through it.
    """
    import linecache
    global _SYNTH_COUNTER
    _SYNTH_COUNTER += 1
    fake_path = f"<unsloth-test-synthetic-{_SYNTH_COUNTER}.py>"
    src = forward_src.lstrip("\n")
    linecache.cache[fake_path] = (
        len(src), None, [line + "\n" for line in src.splitlines()], fake_path,
    )
    namespace = {}
    code = compile(src, fake_path, "exec")
    exec(code, namespace)
    forward_fn = namespace["forward"]
    cls = type(name, (), {"forward": forward_fn})
    cls.__module__ = "transformers.models.synthetic.modeling_synthetic"
    return cls


def test_install_noop_when_disabled(fresh_install, monkeypatch):
    # On by default; UNSLOTH_FUSED_FORWARD=0 is the explicit opt-out.
    monkeypatch.setenv("UNSLOTH_FUSED_FORWARD", "0")
    cls = _make_synthetic_class(CANONICAL_KW_SRC)
    original = cls.forward
    assert fresh_install.install_for_class(cls) is False
    assert cls.forward is original


def test_install_default_is_on(fresh_install, monkeypatch):
    # With no env var set, the installer must be active.
    monkeypatch.delenv("UNSLOTH_FUSED_FORWARD", raising=False)
    cls = _make_synthetic_class(CANONICAL_KW_SRC)
    assert fresh_install.is_enabled() is True
    assert fresh_install.install_for_class(cls) is True


def test_install_skips_ineligible_name(fresh_install, enable_env):
    cls = _make_synthetic_class(CANONICAL_KW_SRC, name="SyntheticModel")
    original = cls.forward
    assert fresh_install.install_for_class(cls) is False
    assert cls.forward is original


def test_install_skips_for_conditional_generation(fresh_install, enable_env):
    # *ForConditionalGeneration uses aligned labels (seq2seq); the fused
    # kernel hardcodes a causal shift and would produce off-by-one losses.
    # Such classes must be skipped.
    cls = _make_synthetic_class(CANONICAL_KW_SRC, name="SyntheticForConditionalGeneration")
    original = cls.forward
    assert fresh_install.install_for_class(cls) is False
    assert cls.forward is original


COMPOSITE_HEAD_SRC = """
def forward(self, input_ids=None, labels=None, **kwargs):
    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs.last_hidden_state
    logits = self.cls(hidden_states)
    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
    return (loss, logits)
"""


def test_install_skips_composite_head(fresh_install, enable_env):
    # BigBird-style `self.cls(...)` (BigBirdOnlyMLMHead) is a composite head,
    # not a Linear; the adapter would crash on `lm_head.weight`. The
    # installer must reject heads that aren't in the _LINEAR_HEAD_ATTRS
    # allowlist.
    cls = _make_synthetic_class(COMPOSITE_HEAD_SRC, name="SyntheticForCausalLM")
    original = cls.forward
    assert fresh_install.install_for_class(cls) is False
    assert cls.forward is original
    assert cls.__qualname__ in fresh_install._UNMATCHED
    assert "non-linear-head" in fresh_install._UNMATCHED[cls.__qualname__]


def test_install_patches_canonical_forward(fresh_install, enable_env):
    cls = _make_synthetic_class(CANONICAL_KW_SRC)
    original = cls.forward
    ok = fresh_install.install_for_class(cls)
    assert ok is True
    assert cls.forward is not original
    rep = fresh_install._PATCHED[cls.__qualname__]
    assert rep["tier"] == "2-ast-triplet"
    assert rep["head_attr"] == "lm_head"


def test_install_idempotent(fresh_install, enable_env):
    cls = _make_synthetic_class(CANONICAL_KW_SRC)
    first = fresh_install.install_for_class(cls)
    patched_fn = cls.forward
    second = fresh_install.install_for_class(cls)
    assert first is True and second is True
    assert cls.forward is patched_fn


def test_install_leaves_non_canonical_in_unmatched(fresh_install, enable_env):
    cls = _make_synthetic_class(NON_CANONICAL_SRC)
    ok = fresh_install.install_for_class(cls)
    assert ok is False
    assert cls.__qualname__ in fresh_install._UNMATCHED


def test_install_function_override_fast_path(fresh_install, enable_env):
    from unsloth_zoo.fused_losses.forward_install import _structural_hash
    cls = _make_synthetic_class(CANONICAL_KW_SRC)
    target_hash = _structural_hash(cls.forward)
    assert target_hash is not None

    sentinel = []
    def _replacement(self, *a, **kw):
        sentinel.append(True)
        return (None, None)

    fresh_install.register_canonical(target_hash, _replacement)
    ok = fresh_install.install_for_class(cls)
    assert ok is True
    rep = fresh_install._PATCHED[cls.__qualname__]
    assert rep["tier"] == "1-function-override"
    cls.forward(object())
    assert sentinel == [True]


def test_audit_dump(fresh_install, enable_env):
    cls = _make_synthetic_class(CANONICAL_KW_SRC)
    fresh_install.install_for_class(cls)
    out = fresh_install.audit()
    assert out["enabled"] is True
    assert out["n_patched"] >= 1
    assert cls.__qualname__ in out["patched"]


# ---------------------------------------------------------------------------
# Numerical equivalence on a small toy model
# ---------------------------------------------------------------------------


def _toy_forward_src():
    # Mirrors the canonical HF template enough to be rewriter-eligible.
    return """
def forward(self, hidden_states, labels=None, **kwargs):
    logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
    return (loss, logits)
"""


def test_rewritten_forward_loss_matches_reference(fresh_install, enable_env):
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
        pytest.skip("fused CE kernel requires a CUDA device")

    cls = _make_synthetic_class(_toy_forward_src(), name="ToyForCausalLM")

    # Wire a config + lm_head + reference loss_function.
    B, T, H, V = 2, 8, 32, 64

    class _Config:
        vocab_size = V

    instance = cls()
    instance.config = _Config()
    instance.lm_head = torch.nn.Linear(H, V, bias=False).cuda().to(torch.bfloat16)

    def _reference_loss(logits, labels, vocab_size, **kw):
        # unsloth_fused_ce_loss shifts labels by one (causal LM convention).
        # Mirror that here so the two losses are apples-to-apples.
        shifted = labels.clone()
        shifted[..., :-1] = labels[..., 1:]
        shifted[..., -1] = -100
        return torch.nn.functional.cross_entropy(
            logits.float().view(-1, vocab_size),
            shifted.view(-1),
            ignore_index=-100,
        )
    instance.loss_function = _reference_loss

    hidden = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, V, (B, T), device="cuda")

    ref_loss, _ = instance.forward(hidden, labels=labels)
    ref_loss_value = float(ref_loss.detach().cpu().item())

    # Install fused forward.
    ok = fresh_install.install_for_class(cls)
    assert ok is True

    # The instance still binds the old forward (Python attribute lookup hits
    # the class on call), so we re-fetch from the class.
    fused_loss, fused_logits = cls.forward(instance, hidden, labels=labels)
    fused_loss_value = float(fused_loss.detach().cpu().item())

    # Loss should match the reference to within bf16 -> fp32 rounding noise.
    assert abs(fused_loss_value - ref_loss_value) < 0.05, (
        f"fused loss {fused_loss_value} diverged from reference {ref_loss_value}"
    )
    # logits slot becomes the EMPTY_LOGITS sentinel under fused path.
    assert fused_logits.numel() == 0


def test_fused_kernel_respects_ignore_index():
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
        pytest.skip("fused CE kernel requires a CUDA device")
    from unsloth_zoo.fused_losses import unsloth_fused_ce_loss

    B, T, H, V = 1, 16, 8, 32
    hidden = torch.randn(B, T, H, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(V, H, device="cuda", dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, V, (B, T), device="cuda")
    labels[0, 0] = 99  # would be a CUDA-side assert if not masked out

    loss = unsloth_fused_ce_loss(
        trainer=None,
        hidden_states=hidden,
        lm_head_weight=weight,
        lm_head_bias=None,
        labels=labels,
        torch_compile=False,
        ignore_index=99,
    )
    assert torch.isfinite(loss), f"loss not finite with ignore_index=99: {loss}"


def test_fused_kernel_accepts_int_n_items():
    # HF Trainer / gradient accumulation passes a Python int for
    # num_items_in_batch. The kernel must promote it to a tensor before
    # the DataParallel .numel()/.ravel() guard.
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
        pytest.skip("fused CE kernel requires a CUDA device")
    from unsloth_zoo.fused_losses import unsloth_fused_ce_loss

    B, T, H, V = 1, 8, 8, 16
    hidden = torch.randn(B, T, H, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(V, H, device="cuda", dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, V, (B, T), device="cuda")

    loss = unsloth_fused_ce_loss(
        trainer=None, hidden_states=hidden, lm_head_weight=weight, lm_head_bias=None,
        labels=labels, torch_compile=False, n_items=3,  # int, not tensor
    )
    assert torch.isfinite(loss), f"loss not finite with int n_items: {loss}"


def _ce_reference(hidden, lm_head, labels, shift_labels=None, n_items=None,
                  ignore_index=-100, label_smoothing=0.0):
    """Reference: F.cross_entropy on materialised logits. Mirrors HF
    ForCausalLMLoss when shift_labels is supplied, otherwise does the
    canonical causal shift itself."""
    import torch
    logits = torch.nn.functional.linear(hidden, lm_head.weight,
                                        getattr(lm_head, "bias", None))
    if shift_labels is None:
        # Standard causal shift: predict token t+1 from position t.
        target = torch.full_like(labels, ignore_index)
        target[..., :-1] = labels[..., 1:]
    else:
        target = shift_labels
    reduction = "sum" if n_items is not None else "mean"
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.shape[-1]).float(),
        target.reshape(-1).to(logits.device),
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )
    if n_items is not None:
        loss = loss / float(n_items)
    return loss


def test_adapter_auto_shift_matches_F_cross_entropy():
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
        pytest.skip("requires CUDA")
    from unsloth_zoo.fused_losses import unsloth_fused_lm_head_loss

    torch.manual_seed(0)
    B, T, H, V = 2, 32, 64, 128
    hidden = torch.randn(B, T, H, device="cuda", dtype=torch.float32, requires_grad=True)
    lm_head = torch.nn.Linear(H, V, bias=False).cuda().float()
    labels = torch.randint(0, V, (B, T), device="cuda")
    labels[0, 5:8] = -100  # sprinkle ignore_index

    fused = unsloth_fused_lm_head_loss(hidden, lm_head, labels, vocab_size=V)
    ref = _ce_reference(hidden, lm_head, labels)
    assert torch.allclose(fused, ref, atol=1e-5, rtol=1e-5), (
        f"fused auto-shift {fused.item()} != reference {ref.item()}"
    )


def test_adapter_pre_shifted_tensor_matches_F_cross_entropy():
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
        pytest.skip("requires CUDA")
    from unsloth_zoo.fused_losses import unsloth_fused_lm_head_loss

    torch.manual_seed(1)
    B, T, H, V = 2, 32, 64, 128
    hidden = torch.randn(B, T, H, device="cuda", dtype=torch.float32, requires_grad=True)
    lm_head = torch.nn.Linear(H, V, bias=False).cuda().float()
    labels = torch.randint(0, V, (B, T), device="cuda")
    # Simulate trl padding_free pre-shifted target: shift labels left by 1,
    # last position becomes ignore_index. Same shape as logits.
    shift = torch.full_like(labels, -100)
    shift[..., :-1] = labels[..., 1:]

    fused = unsloth_fused_lm_head_loss(
        hidden, lm_head, labels=labels, vocab_size=V, shift_labels=shift,
    )
    ref = _ce_reference(hidden, lm_head, labels=None, shift_labels=shift)
    assert torch.allclose(fused, ref, atol=1e-5, rtol=1e-5), (
        f"fused pre-shifted {fused.item()} != reference {ref.item()}"
    )


def test_adapter_shift_labels_false_matches_F_cross_entropy():
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
        pytest.skip("requires CUDA")
    from unsloth_zoo.fused_losses import unsloth_fused_lm_head_loss

    torch.manual_seed(2)
    B, T, H, V = 2, 32, 64, 128
    hidden = torch.randn(B, T, H, device="cuda", dtype=torch.float32, requires_grad=True)
    lm_head = torch.nn.Linear(H, V, bias=False).cuda().float()
    # Caller hands us labels that are already pre-shifted (the bool=False
    # contract: do not shift again, treat labels as the target tensor).
    target = torch.randint(0, V, (B, T), device="cuda")
    target[..., -1] = -100  # canonical pre-shift fills last position
    fused = unsloth_fused_lm_head_loss(
        hidden, lm_head, labels=target, vocab_size=V, shift_labels=False,
    )
    ref = _ce_reference(hidden, lm_head, labels=None, shift_labels=target)
    assert torch.allclose(fused, ref, atol=1e-5, rtol=1e-5), (
        f"fused shift_labels=False {fused.item()} != reference {ref.item()}"
    )


def test_adapter_num_items_in_batch_divides_correctly():
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
        pytest.skip("requires CUDA")
    from unsloth_zoo.fused_losses import unsloth_fused_lm_head_loss

    torch.manual_seed(3)
    B, T, H, V = 2, 16, 32, 64
    hidden = torch.randn(B, T, H, device="cuda", dtype=torch.float32, requires_grad=True)
    lm_head = torch.nn.Linear(H, V, bias=False).cuda().float()
    labels = torch.randint(0, V, (B, T), device="cuda")
    labels[:, :2] = -100  # pad-like prefix

    # Effective token count after causal shift: only positions where the
    # shifted target is not ignore_index count.
    target = torch.full_like(labels, -100)
    target[..., :-1] = labels[..., 1:]
    n_items = int((target != -100).sum().item())

    fused = unsloth_fused_lm_head_loss(
        hidden, lm_head, labels, vocab_size=V, num_items_in_batch=n_items,
    )
    ref = _ce_reference(hidden, lm_head, labels, n_items=n_items)
    assert torch.allclose(fused, ref, atol=1e-5, rtol=1e-5), (
        f"fused (num_items={n_items}) {fused.item()} != reference {ref.item()}"
    )


def test_adapter_num_items_in_batch_as_int_and_tensor_equal():
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
        pytest.skip("requires CUDA")
    from unsloth_zoo.fused_losses import unsloth_fused_lm_head_loss

    torch.manual_seed(4)
    B, T, H, V = 2, 16, 32, 64
    hidden = torch.randn(B, T, H, device="cuda", dtype=torch.float32, requires_grad=True)
    lm_head = torch.nn.Linear(H, V, bias=False).cuda().float()
    labels = torch.randint(0, V, (B, T), device="cuda")
    n_items_int = 17
    n_items_tensor = torch.tensor(17, device="cuda")

    fused_int = unsloth_fused_lm_head_loss(
        hidden, lm_head, labels, vocab_size=V, num_items_in_batch=n_items_int,
    )
    fused_tensor = unsloth_fused_lm_head_loss(
        hidden, lm_head, labels, vocab_size=V, num_items_in_batch=n_items_tensor,
    )
    assert torch.allclose(fused_int, fused_tensor, atol=1e-6), (
        f"int vs tensor n_items disagree: {fused_int.item()} vs {fused_tensor.item()}"
    )


# ---------------------------------------------------------------------------
# Env-var branch: UNSLOTH_RETURN_LOGITS
# ---------------------------------------------------------------------------


def _install_toy_cls(fresh_install):
    cls = _make_synthetic_class(_toy_forward_src(), name="EnvVarToyForCausalLM")
    assert fresh_install.install_for_class(cls) is True
    return cls


def _toy_instance(cls, dtype, V=64, H=32):
    import torch
    class _Config:
        vocab_size = V
    inst = cls()
    inst.config = _Config()
    inst.lm_head = torch.nn.Linear(H, V, bias=False).cuda().to(dtype)
    def _reference_loss(logits, labels, vocab_size, **kw):
        shifted = labels.clone()
        shifted[..., :-1] = labels[..., 1:]
        shifted[..., -1] = -100
        return torch.nn.functional.cross_entropy(
            logits.float().view(-1, vocab_size),
            shifted.view(-1),
            ignore_index=-100,
        )
    inst.loss_function = _reference_loss
    return inst


def test_rewritten_forward_returns_real_logits_when_env_set(
    fresh_install, enable_env, monkeypatch,
):
    # UNSLOTH_RETURN_LOGITS=1 with labels present: logits slot carries
    # real lm_head output and loss is computed via self.loss_function on
    # those materialised logits. Critically, only ONE lm_head matmul
    # happens (no fused-kernel re-matmul).
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
        pytest.skip("requires CUDA")

    cls = _install_toy_cls(fresh_install)
    B, T, H, V = 2, 8, 32, 64
    inst = _toy_instance(cls, dtype=torch.bfloat16, V=V, H=H)
    hidden = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(0, V, (B, T), device="cuda")

    # Count lm_head invocations to prove single-matmul on the opt-in path.
    lm_head_calls = {"n": 0}
    orig_call = inst.lm_head.__class__.__call__
    def _counting_call(self, *a, **kw):
        lm_head_calls["n"] += 1
        return orig_call(self, *a, **kw)
    monkeypatch.setattr(inst.lm_head.__class__, "__call__", _counting_call)

    # Also count self.loss_function invocations to confirm the opt-in path
    # routes through the model's own loss function.
    loss_fn_calls = {"n": 0}
    real_loss_fn = inst.loss_function
    def _counting_loss_fn(*a, **kw):
        loss_fn_calls["n"] += 1
        return real_loss_fn(*a, **kw)
    inst.loss_function = _counting_loss_fn

    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "1")
    loss, logits = cls.forward(inst, hidden, labels=labels)
    assert loss is not None
    assert torch.isfinite(loss)
    assert logits.shape == (B, T, V), (
        f"expected real logits {(B, T, V)}, got {tuple(logits.shape)}"
    )
    assert lm_head_calls["n"] == 1, (
        f"opt-in path must do exactly one lm_head matmul, observed {lm_head_calls['n']}"
    )
    assert loss_fn_calls["n"] == 1, (
        f"opt-in path must call self.loss_function once, observed {loss_fn_calls['n']}"
    )


def test_rewritten_forward_default_labels_branch_yields_empty_logits(
    fresh_install, enable_env, monkeypatch,
):
    # Defaults (env var unset): labels branch returns EMPTY_LOGITS so we
    # don't pay the lm_head matmul on the hot path. Byte-identical to
    # the pre-PR behaviour.
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
        pytest.skip("requires CUDA")

    cls = _install_toy_cls(fresh_install)
    B, T, H, V = 2, 8, 32, 64
    inst = _toy_instance(cls, dtype=torch.bfloat16, V=V, H=H)
    hidden = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(0, V, (B, T), device="cuda")

    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising=False)
    loss, logits = cls.forward(inst, hidden, labels=labels)
    assert loss is not None
    assert torch.isfinite(loss)
    assert logits.numel() == 0


def test_fused_kernel_label_smoothing_changes_loss():
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
        pytest.skip("fused CE kernel requires a CUDA device")
    from unsloth_zoo.fused_losses import unsloth_fused_ce_loss

    B, T, H, V = 1, 8, 8, 16
    hidden = torch.randn(B, T, H, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(V, H, device="cuda", dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, V, (B, T), device="cuda")

    loss_plain = unsloth_fused_ce_loss(
        trainer=None, hidden_states=hidden, lm_head_weight=weight, lm_head_bias=None,
        labels=labels, torch_compile=False,
    )
    loss_smoothed = unsloth_fused_ce_loss(
        trainer=None, hidden_states=hidden, lm_head_weight=weight, lm_head_bias=None,
        labels=labels, torch_compile=False, label_smoothing=0.1,
    )
    assert float(loss_plain.item()) != float(loss_smoothed.item()), (
        "label_smoothing kwarg was ignored: smoothed loss equals plain loss"
    )

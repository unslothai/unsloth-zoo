# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Fused CE on an lm_head carrying a live PEFT adapter must match lm_head.forward(): the kernel used to read only
lm_head.weight (the base weight on a tuner layer), so the adapter got no gradient and the loss was the base head's.
A live adapter now takes the head's own forward; plain heads and modules_to_save stay fused (CPU)."""

import inspect
import os
import textwrap
import types

import pytest
import torch

peft = pytest.importorskip("peft")
gemma3 = pytest.importorskip("transformers.models.gemma3.modeling_gemma3")
from transformers.loss.loss_utils import ForCausalLMLoss  # noqa: E402
from unsloth_zoo import compiler  # noqa: E402
from unsloth_zoo.fused_losses import cross_entropy_loss as ce  # noqa: E402
from unsloth_zoo.fused_losses import forward_adapter  # noqa: E402
from unsloth_zoo.fused_losses import unsloth_fused_ce_loss, unsloth_fused_lm_head_loss  # noqa: E402
from unsloth_zoo.fused_losses.forward_adapter import (  # noqa: E402
    lm_head_live_adapter,
    lm_head_needs_module_forward,
    unsloth_module_head_ce_loss,
)
from unsloth_zoo.fused_losses.forward_install import _can_fuse_loss  # noqa: E402

V, H = 96, 32


@pytest.fixture(autouse = True)
def _eager_fused_ce(monkeypatch):
    # The CE dispatcher reads its compile flag at import: set the module value, not the environment.
    monkeypatch.setattr(ce, "_FUSED_CE_COMPILE_DISABLED", True)
    monkeypatch.setattr(forward_adapter, "_WARNED_MODULE_HEADS", set())


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(V, H)
        self.proj = torch.nn.Linear(H, H)
        self.lm_head = torch.nn.Linear(H, V, bias = False)

    def forward(self, ids):
        return self.lm_head(self.proj(self.embed(ids)))


def _ids():
    return torch.randint(0, V, (2, 9), generator = torch.Generator().manual_seed(1))


def _wrap(dropout = 0.0, mts = False, **extra):
    torch.manual_seed(0)
    if mts:
        cfg = peft.LoraConfig(r = 4, target_modules = ["proj"], modules_to_save = ["lm_head"])
    else:
        cfg = peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = ["proj", "lm_head"], lora_dropout = dropout,
                              init_lora_weights = False, **extra)
    model = peft.get_peft_model(Toy(), cfg).train()
    inner = model.base_model.model
    if mts:
        with torch.no_grad():
            inner.lm_head.modules_to_save["default"].weight.add_(0.1)
    return model, inner


def _grad(loss, params):
    return torch.autograd.grad(loss, params, allow_unused = True, materialize_grads = True)


def _stock_and_fused(inner, params, ids, **fused_kw):
    hidden = lambda: inner.proj(inner.embed(ids))  # noqa: E731
    torch.manual_seed(3)
    want = ForCausalLMLoss(inner.lm_head(hidden()).float(), ids, vocab_size = V)
    want_grads = _grad(want, params)
    torch.manual_seed(3)
    got = unsloth_fused_lm_head_loss(hidden(), inner.lm_head, ids, vocab_size = V, torch_compile = False, **fused_kw)
    got_grads = _grad(got, params)
    return want, want_grads, got, got_grads


def _assert_match(inner, model, ids, rtol = 1e-4, atol = 1e-6):
    params = [p for p in model.parameters() if p.requires_grad]
    want, want_grads, got, got_grads = _stock_and_fused(inner, params, ids)
    torch.testing.assert_close(got, want, rtol = rtol, atol = atol)
    for g, w in zip(got_grads, want_grads):
        torch.testing.assert_close(g, w, rtol = rtol, atol = atol)
    return dict(zip([n for n, p in model.named_parameters() if p.requires_grad], got_grads))


class _FusedSpy:
    def __init__(self, monkeypatch):
        self.calls = 0
        real = forward_adapter.unsloth_fused_ce_loss

        def spy(*args, **kwargs):
            self.calls += 1
            return real(*args, **kwargs)
        monkeypatch.setattr(forward_adapter, "unsloth_fused_ce_loss", spy)


@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_lora_head_trains_through_its_forward(dropout, monkeypatch):
    spy = _FusedSpy(monkeypatch)
    model, inner = _wrap(dropout = dropout)
    grads = _assert_match(inner, model, _ids())
    assert grads["base_model.model.lm_head.lora_B.default.weight"].abs().sum() > 0
    assert spy.calls == 0  # the head's own forward, not the kernel


@pytest.mark.parametrize("extra", [dict(use_dora = True), dict(lora_bias = True)])
def test_lora_variants_train_through_the_heads_forward(extra, capsys):
    model, inner = _wrap(**extra)
    _assert_match(inner, model, _ids())
    assert "the loss calls the head's own forward" in capsys.readouterr().out


def test_adalora_head_uses_its_projection():
    torch.manual_seed(0)
    model = peft.get_peft_model(
        Toy(), peft.AdaLoraConfig(target_modules = ["proj", "lm_head"], total_step = 10, init_r = 4, init_lora_weights = False)
    ).train()
    inner = model.base_model.model
    assert lm_head_live_adapter(inner.lm_head) is not None
    _assert_match(inner, model, _ids())


class _FakeQuantLinear(torch.nn.Linear):
    """A quantized base whose forward is not x @ weight.T (int8 round trip): only its own forward is right."""

    def forward(self, x):
        scale = self.weight.detach().abs().amax(dim = 1, keepdim = True) / 127
        return torch.nn.functional.linear(x, torch.round(self.weight / scale).clamp(-127, 127) * scale)


def test_quantized_base_under_lora_uses_the_heads_forward():
    model, inner = _wrap()
    base = inner.lm_head.base_layer
    quant = _FakeQuantLinear(H, V, bias = False)
    with torch.no_grad():
        quant.weight.copy_(base.weight)
    quant.weight.requires_grad_(False)
    inner.lm_head.base_layer = quant
    _assert_match(inner, model, _ids())


def test_fp16_lora_head_matches_stock():
    """A large LoRA scale: PEFT applies it after B in the base dtype; the head's own forward keeps that order."""
    torch.manual_seed(0)
    model = peft.get_peft_model(
        Toy(), peft.LoraConfig(r = 1, lora_alpha = 64, target_modules = ["lm_head"], init_lora_weights = False)
    ).half().train()
    inner = model.base_model.model
    params = [p for p in model.parameters() if p.requires_grad]
    want, want_grads, got, got_grads = _stock_and_fused(inner, params, _ids())
    assert torch.isfinite(want)
    torch.testing.assert_close(got, want)
    for g, w in zip(got_grads, want_grads):
        assert torch.isfinite(g).all()
        torch.testing.assert_close(g, w, rtol = 1e-2, atol = 1e-3)


def test_disabled_and_merged_heads_stay_fused(monkeypatch):
    spy = _FusedSpy(monkeypatch)
    model, inner = _wrap()
    ids = _ids()
    inner.proj.base_layer.weight.requires_grad_(True)  # disabling freezes every adapter: keep one grad to compare
    with model.disable_adapter():
        _assert_match(inner, model, ids)
    assert spy.calls == 1
    inner.lm_head.merge()
    _assert_match(inner, model, ids)
    assert spy.calls == 2


def test_merged_then_disabled_head_unmerges_like_its_forward():
    model, inner = _wrap()
    ids = _ids()
    inner.proj.base_layer.weight.requires_grad_(True)
    inner.lm_head.merge()
    with model.disable_adapter():
        assert lm_head_live_adapter(inner.lm_head) is not None
        _assert_match(inner, model, ids)
        assert not inner.lm_head.merged


def test_modules_to_save_head_trains_the_copy(monkeypatch):
    spy = _FusedSpy(monkeypatch)
    model, inner = _wrap(mts = True)
    assert lm_head_live_adapter(inner.lm_head) is None
    grads = _assert_match(inner, model, _ids())
    assert "base_model.model.lm_head.original_module.weight" not in grads
    assert grads["base_model.model.lm_head.modules_to_save.default.weight"].abs().sum() > 0
    assert spy.calls == 1


def test_per_example_adapter_names_take_the_heads_forward():
    model, inner = _wrap()
    model.eval()  # PEFT takes adapter_names only in eval (labelled evaluation)
    with model.base_model._enable_peft_forward_hooks(adapter_names = ["__base__", "default"]):
        assert lm_head_live_adapter(inner.lm_head).endswith("(per-example adapter_names)")
        _assert_match(inner, model, _ids())  # row 0 scores the base head, row 1 the adapter, as the forward does


# Module CE reduction: the same targets, divisor and transforms as the fused kernel.
def _module_vs_kernel(head, hidden, labels, **kw):
    want = unsloth_fused_ce_loss(
        trainer = None, hidden_states = hidden, lm_head_weight = head.weight, lm_head_bias = None,
        labels = labels, torch_compile = False, **kw,
    )
    got = unsloth_module_head_ce_loss(hidden, head, labels, **kw)
    torch.testing.assert_close(got, want, rtol = 1e-5, atol = 1e-6)
    return got


@pytest.mark.parametrize("shift", [True, False])
def test_module_ce_masks_shifted_and_pre_aligned_targets(shift):
    torch.manual_seed(0)
    head = torch.nn.Linear(H, V, bias = False)
    hidden = torch.randn(2, 9, H)
    labels = _ids().to(torch.int32)  # int32 ids, accepted like the kernel
    mask = torch.ones(2, 9, dtype = torch.long)
    mask[0, 4:] = 0
    got = unsloth_module_head_ce_loss(hidden, head, labels, mask = mask, shift_labels = shift, n_items = torch.tensor(7))
    # The chunked kernel's targets (chunked_logprob._ce_targets): the mask applies to shifted and pre-aligned labels.
    targets = labels.long()
    keep = mask
    if shift:
        targets = torch.cat((targets[:, 1:], torch.full_like(targets[:, :1], -100)), dim = 1)
        keep = torch.cat((mask[:, 1:], torch.ones_like(mask[:, :1])), dim = 1)
    targets = targets.masked_fill(keep == 0, -100)
    want = torch.nn.functional.cross_entropy(head(hidden).float().reshape(-1, V), targets.reshape(-1), reduction = "sum") / 7
    torch.testing.assert_close(got, want)
    if shift:
        _module_vs_kernel(head, hidden, labels, mask = mask, n_items = torch.tensor(7))


def test_module_ce_transforms_bias_and_smoothing():
    torch.manual_seed(0)
    head = torch.nn.Linear(H, V, bias = False)
    hidden = torch.randn(2, 9, H)
    bias = torch.randn(V, dtype = torch.float64)
    _module_vs_kernel(head, hidden, _ids(), logit_softcapping = 5.0, logit_scale_multiply = 2.0,
                      logit_scale_divide = 3.0, label_smoothing = 0.1, logits_bias = bias)
    # Tensor scales: no Python truth test, 0 still means off.
    got = unsloth_module_head_ce_loss(hidden, head, _ids(), logit_softcapping = torch.tensor(0.0))
    torch.testing.assert_close(got, unsloth_module_head_ce_loss(hidden, head, _ids()))


def test_module_ce_all_ignored_is_zero_and_vocab_mismatch_raises():
    torch.manual_seed(0)
    head = torch.nn.Linear(H, V, bias = False)
    hidden = torch.randn(2, 9, H)
    ignored = torch.full((2, 9), -100)
    assert unsloth_module_head_ce_loss(hidden, head, ignored).item() == 0.0
    assert unsloth_module_head_ce_loss(hidden, head, ignored, n_items = 0).item() == 0.0
    # logits_to_keep style: 4 kept positions against 9 labels must not be re-read as wider rows.
    with pytest.raises((ValueError, RuntimeError)):
        unsloth_module_head_ce_loss(hidden[:, -4:], head, _ids())


def test_module_ce_follows_the_label_smoothing_override():
    torch.manual_seed(0)
    head = torch.nn.Linear(H, V, bias = False)
    hidden = torch.randn(2, 9, H)
    with ce.unsloth_label_smoothing_override(0.2, torch.tensor(5.0)) as state:
        got = unsloth_module_head_ce_loss(hidden, head, _ids())
    assert state.engaged == 1
    want = unsloth_module_head_ce_loss(hidden, head, _ids(), label_smoothing = 0.2, n_items = torch.tensor(5.0))
    torch.testing.assert_close(got, want)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason = "needs a second device for the head")
def test_split_device_hidden_states_move_to_the_head():
    model, inner = _wrap()
    inner.lm_head.cuda()
    hidden = inner.proj(inner.embed(_ids()))
    loss = unsloth_fused_lm_head_loss(hidden, inner.lm_head, _ids(), torch_compile = False)
    loss.backward()
    assert inner.lm_head.lora_B["default"].weight.grad is not None


def test_fullgraph_trace_with_the_notice(capsys):
    model, inner = _wrap()
    ids = _ids()

    def loss_fn(hidden):
        return unsloth_fused_lm_head_loss(hidden, inner.lm_head, ids, torch_compile = False)

    hidden = inner.proj(inner.embed(ids)).detach()
    compiled = torch.compile(loss_fn, fullgraph = True, backend = "eager")
    torch.testing.assert_close(compiled(hidden), loss_fn(hidden))
    # Printed once, by the eager call, never inside the trace.
    assert capsys.readouterr().out.count("the loss calls the head's own forward") == 1


def _fused_gemma3_forward():
    # Gemma 3 as in test_fused_lm_head_qat.py: the conftest's unsloth import leaves its forward stock.
    source = inspect.getsource(gemma3.Gemma3ForCausalLM.forward)
    source = compiler.fixup_fused_lm_head(source)
    source, _ = compiler.apply_fused_lm_head(source, "Gemma3ForCausalLM")
    assert "unsloth_fused_ce_loss" in source
    from _compiled_module_names import fused_loss_header_names

    scope = dict(vars(gemma3))
    scope.update(fused_loss_header_names())
    scope.update(
        os = os,
        UNSLOTH_ENABLE_CCE = False,
        HAS_CUT_CROSS_ENTROPY = False,
        UNSLOTH_COMPILE_DISABLE = True,
        unsloth_fused_ce_loss = unsloth_fused_ce_loss,
        _can_fuse_loss = _can_fuse_loss,
        unsloth_training_stance = lambda: None,
        EMPTY_LOGITS = torch.empty(0),
    )
    source = textwrap.dedent(source)
    exec(source[source.index("def forward"):], scope)
    return scope["forward"], source


def _gemma3_model(softcap, use_dora):
    config = gemma3.Gemma3TextConfig(
        vocab_size = V, hidden_size = H, intermediate_size = 64, num_hidden_layers = 1, num_attention_heads = 2,
        num_key_value_heads = 1, head_dim = 16, final_logit_softcapping = softcap, attn_implementation = "eager",
    )
    torch.manual_seed(0)
    model = peft.get_peft_model(
        gemma3.Gemma3ForCausalLM(config).float(),
        peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj", "lm_head"], init_lora_weights = False,
                        use_dora = use_dora),
    ).train()
    inner = model.base_model.model
    inner.loss_function = ForCausalLMLoss
    return model, inner


@pytest.mark.parametrize("use_dora", [False, True])
@pytest.mark.parametrize("softcap", [None, 30.0])
def test_pattern2_forward_trains_the_head_adapter(softcap, use_dora, monkeypatch):
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    model, inner = _gemma3_model(softcap, use_dora)
    ids = _ids()
    params = [p for p in model.parameters() if p.requires_grad]
    want = model(input_ids = ids, labels = ids).loss
    want_grads = _grad(want, params)
    forward, source = _fused_gemma3_forward()
    assert "explicit_shift_labels" in source and "_needs_module_head" in source  # cross_entropy_replacement_2
    out = forward(inner, input_ids = ids, labels = ids)
    assert out.logits.numel() > 0  # the head's own logits, as the exact branch returns
    got_grads = _grad(out.loss, params)
    torch.testing.assert_close(out.loss, want)
    for g, w in zip(got_grads, want_grads):
        torch.testing.assert_close(g, w, rtol = 1e-4, atol = 1e-6)


def test_pattern2_follows_the_label_smoothing_override_and_ignored_batches(monkeypatch):
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    model, inner = _gemma3_model(None, True)
    forward, _ = _fused_gemma3_forward()
    ids = _ids()
    with ce.unsloth_label_smoothing_override(0.1, torch.tensor(8.0)) as state:
        got = forward(inner, input_ids = ids, labels = ids).loss
    assert state.engaged == 1
    logits = model(input_ids = ids).logits.float()
    want = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1), reduction = "sum", label_smoothing = 0.1,
    ) / 8.0
    torch.testing.assert_close(got, want)
    ignored = torch.full_like(ids, -100)
    assert forward(inner, input_ids = ids, labels = ignored).loss.item() == 0.0


# Patterns 1 and 3 have no exact-logits branch under labels: a live head adapter goes through its own forward there.
PATTERN_1 = """
def forward(self, input_ids=None, labels=None, **kwargs):
    hidden_states = self.proj(self.embed(input_ids))
    logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        logits = logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
    return types.SimpleNamespace(loss=loss, logits=logits)
"""

PATTERN_3 = """
def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
    hidden_states = self.proj(self.embed(input_ids))
    logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:]
            shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
            shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
        )
    return types.SimpleNamespace(loss=loss, logits=logits)
"""

AST_CANONICAL = """
def forward(self, input_ids=None, labels=None, **kwargs):
    hidden_states = self.proj(self.embed(input_ids))
    logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size, **kwargs)
    return types.SimpleNamespace(loss=loss, logits=logits)
"""


def _exec_forward(source, extra_scope = None):
    from _compiled_module_names import fused_loss_header_names

    scope = dict(torch = torch, nn = torch.nn, CrossEntropyLoss = torch.nn.CrossEntropyLoss, types = types, os = os)
    if extra_scope is not None:
        scope.update(fused_loss_header_names())
        scope.update(extra_scope)
    source = textwrap.dedent(source)
    exec(source[source.index("def forward"):], scope)
    return scope["forward"]


def _check_against_stock(stock, rewritten, model, inner, **kw):
    params = [p for p in model.parameters() if p.requires_grad]
    want = stock(inner, **kw).loss
    want_grads = torch.autograd.grad(want, params)
    out = rewritten(inner, **kw)
    got_grads = torch.autograd.grad(out.loss, params)
    torch.testing.assert_close(out.loss, want)
    for g, w in zip(got_grads, want_grads):
        torch.testing.assert_close(g, w, rtol = 1e-4, atol = 1e-6)
    return out


def _regex_forward(pattern, marker):
    new, fused = compiler.apply_fused_lm_head(pattern, "ToyForCausalLM")
    assert fused and marker in new and "lm_head_needs_module_forward(self.lm_head)" in new
    return _exec_forward(new, dict(
        UNSLOTH_ENABLE_CCE = False, HAS_CUT_CROSS_ENTROPY = False, UNSLOTH_COMPILE_DISABLE = True,
        unsloth_fused_ce_loss = unsloth_fused_ce_loss, _can_fuse_loss = _can_fuse_loss,
        unsloth_training_stance = lambda: None, EMPTY_LOGITS = torch.empty(0),
    ))


PATTERNS = [
    (PATTERN_1, "labels, mask = None, n_items = n_items"),  # cross_entropy_replacement_1's module branch
    (PATTERN_3, "labels, mask = _mask, n_items = n_items"),  # cross_entropy_replacement_3's module branch
]


def _pattern_kwargs(pattern, ids):
    kw = dict(input_ids = ids, labels = ids)
    if pattern is PATTERN_3:
        mask = torch.ones_like(ids)
        mask[0, 5:] = 0
        kw["attention_mask"] = mask
    return kw


@pytest.mark.parametrize("use_dora", [False, True])
@pytest.mark.parametrize("pattern, marker", PATTERNS, ids = ["pattern1", "pattern3"])
def test_patterns_without_exact_branch_train_the_head_adapter(pattern, marker, use_dora, monkeypatch):
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    rewritten = _regex_forward(pattern, marker)
    model, inner = _wrap(use_dora = use_dora)
    inner.config = types.SimpleNamespace(vocab_size = V)
    _check_against_stock(_exec_forward(pattern), rewritten, model, inner, **_pattern_kwargs(pattern, _ids()))


@pytest.mark.parametrize("pattern, marker", PATTERNS, ids = ["pattern1", "pattern3"])
def test_patterns_reuse_the_returned_projection(pattern, marker, monkeypatch):
    """UNSLOTH_RETURN_LOGITS=1: the loss scores the logits it returns (one projection, one dropout draw)."""
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "1")
    rewritten = _regex_forward(pattern, marker)
    model, inner = _wrap(dropout = 0.7)
    inner.config = types.SimpleNamespace(vocab_size = V)
    ids = _ids()
    kw = _pattern_kwargs(pattern, ids)
    calls = []
    inner.lm_head.register_forward_hook(lambda *a: calls.append(1))
    out = rewritten(inner, **kw)
    assert len(calls) == 1
    labels = ids[:, 1:].clone()
    if "attention_mask" in kw:
        labels[kw["attention_mask"][:, 1:] == 0] = -100
    want = torch.nn.functional.cross_entropy(out.logits[:, :-1].float().reshape(-1, V), labels.reshape(-1))
    torch.testing.assert_close(out.loss, want)


@pytest.mark.parametrize("use_dora", [False, True])
def test_ast_rewritten_forward_trains_the_head_adapter(use_dora, monkeypatch):
    from unsloth_zoo.fused_losses.ast_rewriter import rewrite_forward_source

    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    new, _ = rewrite_forward_source(AST_CANONICAL)
    assert "unsloth_fused_lm_head_loss(" in new
    # Only the names forward_install puts in the namespace: the head check lives inside the adapter.
    rewritten = _exec_forward(new, dict(
        unsloth_fused_lm_head_loss = unsloth_fused_lm_head_loss, _can_fuse_loss = _can_fuse_loss,
        EMPTY_LOGITS = torch.empty(0),
    ))
    model, inner = _wrap(use_dora = use_dora)
    inner.config = types.SimpleNamespace(vocab_size = V)
    inner.loss_function = ForCausalLMLoss
    ids = _ids()
    out = _check_against_stock(_exec_forward(AST_CANONICAL), rewritten, model, inner, input_ids = ids, labels = ids)
    assert out.logits.numel() == 0


def test_plain_head_needs_no_module_forward():
    assert not lm_head_needs_module_forward(torch.nn.Linear(H, V))

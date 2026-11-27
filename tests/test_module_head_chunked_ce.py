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

"""Chunked lm_head forward for live PEFT adapters: full logits never exist, and loss and grads must
match the all-rows path (UNSLOTH_CHUNKED_MODULE_HEAD=0), on CPU float64."""

import copy

import pytest
import torch

peft = pytest.importorskip("peft")
from unsloth_zoo.fused_losses import cross_entropy_loss as ce  # noqa: E402
from unsloth_zoo.fused_losses import unsloth_fused_ce_loss  # noqa: E402

V, H, B, T = 50, 16, 2, 11


@pytest.fixture(autouse = True)
def _fresh_state(monkeypatch):
    monkeypatch.setattr(ce, "_FUSED_CE_COMPILE_DISABLED", True)
    monkeypatch.setattr(ce, "_WARNED_MODULE_HEADS", set())
    monkeypatch.setattr(ce, "_CHUNKED_MODULE_HEAD", True, raising = False)


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(H, H)
        self.lm_head = torch.nn.Linear(H, V, bias = True)


def _head(dropout = 0.0, train_base = False, dtype = torch.float64):
    torch.manual_seed(0)
    cfg = peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = ["lm_head"], lora_dropout = dropout,
                          init_lora_weights = False, bias = "all")
    head = peft.get_peft_model(_Toy().to(dtype), cfg).train().base_model.model.lm_head
    if train_base:
        head.base_layer.weight.requires_grad_(True)
    return head


def _labels():
    labels = torch.randint(0, V, (B, T), generator = torch.Generator().manual_seed(1))
    labels[0, :5] = -100  # prompt rows
    labels[1, 7] = -100
    return labels


def _close(got, want, got_g = (), want_g = ()):
    # The loss upcasts the logits to float32 (not float64): chunk sums differ in the last float32 bits.
    torch.testing.assert_close(got, want, rtol = 1e-6, atol = 1e-7)
    assert len(got_g) == len(want_g)
    for g, w in zip(got_g, want_g):
        torch.testing.assert_close(g, w, rtol = 1e-5, atol = 1e-7)


def _run(head, hidden, labels, chunked, rows = 0, monkeypatch = None, **kw):
    monkeypatch.setattr(ce, "_CHUNKED_MODULE_HEAD", chunked, raising = False)
    monkeypatch.setattr(ce, "_MODULE_HEAD_ROWS", rows, raising = False)
    before = getattr(ce, "_MODULE_HEAD_CHUNKED_CALLS", 0)
    params = [p for p in head.parameters() if p.requires_grad]
    for p in params:
        p.grad = None
    h = hidden.detach().requires_grad_(hidden.requires_grad)
    loss = unsloth_fused_ce_loss(trainer = None, hidden_states = h, lm_head_weight = head.base_layer.weight,
                                 lm_head_bias = head.base_layer.bias, labels = labels, torch_compile = False,
                                 lm_head_module = head, **kw)
    grads = []
    if loss.requires_grad:
        (loss * 3.0).backward()  # a non-unit upstream gradient
        grads = [p.grad.clone() for p in params] + ([h.grad.clone()] if h.requires_grad else [])
    return loss.detach(), grads, getattr(ce, "_MODULE_HEAD_CHUNKED_CALLS", 0) - before


@pytest.mark.parametrize("rows", [0, 1, 4, 5, 7, 64])
@pytest.mark.parametrize("hidden_grad", [True, False])
def test_chunked_matches_all_rows_loss_and_every_gradient(rows, hidden_grad, monkeypatch):
    head = _head(train_base = True)
    hidden = torch.randn(B, T, H, dtype = torch.float64).requires_grad_(hidden_grad)
    want, want_g, n0 = _run(head, hidden, _labels(), False, monkeypatch = monkeypatch)
    got, got_g, n1 = _run(head, hidden, _labels(), True, rows, monkeypatch = monkeypatch)
    assert (n0, n1) == (0, 1)
    # lora_A, lora_B, base weight, base bias (+ hidden)
    assert len(got_g) == 4 + hidden_grad and all(g.abs().sum() > 0 for g in got_g)
    _close(got, want, got_g, want_g)


@pytest.mark.parametrize("shift", [True, False])
def test_chunked_matches_with_transforms_count_and_mask(shift, monkeypatch):
    head = _head()
    hidden = torch.randn(B, T, H, dtype = torch.float64, requires_grad = True)
    mask = torch.ones(B, T, dtype = torch.long)
    mask[1, 8:] = 0
    kw = dict(mask = mask, n_items = torch.tensor(13), shift_labels = shift, logit_softcapping = 3.0,
              logit_scale_multiply = 1.5, logit_scale_divide = 2.0, label_smoothing = 0.1,
              logits_bias = torch.randn(V, dtype = torch.float64))
    want, want_g, _ = _run(head, hidden, _labels(), False, monkeypatch = monkeypatch, **kw)
    got, got_g, n1 = _run(head, hidden, _labels(), True, 3, monkeypatch = monkeypatch, **kw)
    assert n1 == 1
    _close(got, want, got_g, want_g)


def test_fp32_head_matches_and_grad_scale_is_divided_out(monkeypatch):
    head = _head(dtype = torch.float32)
    hidden = torch.randn(B, T, H, requires_grad = True)
    want, want_g, _ = _run(head, hidden, _labels(), False, monkeypatch = monkeypatch, scaling = 1024.0)
    got, got_g, n1 = _run(head, hidden, _labels(), True, 4, monkeypatch = monkeypatch, scaling = 1024.0)
    assert n1 == 1
    torch.testing.assert_close(got, want, rtol = 1e-6, atol = 1e-6)
    for g, w in zip(got_g, want_g):
        torch.testing.assert_close(g, w, rtol = 1e-5, atol = 1e-6)


def test_head_never_sees_all_rows_nor_ignored_rows(monkeypatch):
    # Flip: the all-rows forward called the head once on every row (full logits, ignored rows included).
    head = _head()
    seen = []
    head.register_forward_hook(lambda m, args, out: seen.append(args[0].reshape(-1, H).shape[0]))
    hidden = torch.randn(B, T, H, dtype = torch.float64, requires_grad = True)
    monkeypatch.setattr(ce, "_MODULE_HEAD_ROWS", 4, raising = False)
    labels = _labels()
    loss = unsloth_fused_ce_loss(trainer = None, hidden_states = hidden, lm_head_weight = head.base_layer.weight,
                                 lm_head_bias = head.base_layer.bias, labels = labels, torch_compile = False,
                                 lm_head_module = head)
    loss.backward()
    labelled = int((labels[..., 1:] != -100).sum())
    assert sum(seen) == labelled and max(seen) <= 4, seen
    assert len(seen) == -(-labelled // 4)


@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_live_dropout_keeps_the_all_rows_forward(dropout, monkeypatch):
    head = _head(dropout = dropout)
    hidden = torch.randn(B, T, H, dtype = torch.float64, requires_grad = True)
    *_, n = _run(head, hidden, _labels(), True, 4, monkeypatch = monkeypatch)
    assert n == (1 if dropout == 0.0 else 0)
    head.eval()  # eval dropout draws nothing: chunked again
    want, want_g, _ = _run(head, hidden, _labels(), False, monkeypatch = monkeypatch)
    got, got_g, n = _run(head, hidden, _labels(), True, 4, monkeypatch = monkeypatch)
    assert n == 1
    _close(got, want, got_g, want_g)


def test_no_grad_all_ignored_and_frozen_head(monkeypatch):
    head = _head()
    hidden = torch.randn(B, T, H, dtype = torch.float64)
    with torch.no_grad():
        want, _, _ = _run(head, hidden, _labels(), False, monkeypatch = monkeypatch)
        got, _, n = _run(head, hidden, _labels(), True, 4, monkeypatch = monkeypatch)
    assert n == 1
    _close(got, want)
    ignored = torch.full((B, T), -100)
    got, got_g, n = _run(head, hidden.requires_grad_(True), ignored, True, 4, monkeypatch = monkeypatch)
    assert n == 1 and got.item() == 0.0 and all(g.abs().sum() == 0 for g in got_g)
    for p in head.parameters():
        p.requires_grad_(False)
    want, _, _ = _run(head, hidden.detach(), _labels(), False, monkeypatch = monkeypatch)
    got, _, n = _run(head, hidden.detach(), _labels(), True, 4, monkeypatch = monkeypatch)
    assert n == 1
    _close(got, want)


def test_traced_call_keeps_the_all_rows_forward(monkeypatch):
    head = _head()
    labels = _labels()

    def loss_fn(hidden):
        return unsloth_fused_ce_loss(trainer = None, hidden_states = hidden, lm_head_weight = head.base_layer.weight,
                                     lm_head_bias = head.base_layer.bias, labels = labels, torch_compile = False,
                                     lm_head_module = head)
    hidden = torch.randn(B, T, H, dtype = torch.float64)
    before = getattr(ce, "_MODULE_HEAD_CHUNKED_CALLS", 0)
    got = torch.compile(loss_fn, fullgraph = True, backend = "eager")(hidden)
    assert getattr(ce, "_MODULE_HEAD_CHUNKED_CALLS", 0) == before
    _close(got, loss_fn(hidden))


def _fp16_case(monkeypatch, chunked, scaling = None):
    """A float16 LoRA head (trainable bias), 32000 vocabulary, 1648 labelled rows: per-element gradients sit at the
    float16 subnormal floor. Returns each trainable tensor's error against the float32 head, and the engagement."""
    torch.manual_seed(0)
    toy = torch.nn.Module()
    toy.lm_head = torch.nn.Linear(512, 32000, bias = True)
    cfg = peft.LoraConfig(r = 16, lora_alpha = 32, target_modules = ["lm_head"], init_lora_weights = False,
                          bias = "all")
    head16 = peft.get_peft_model(toy, cfg).base_model.model.lm_head
    ref = copy.deepcopy(head16)
    head16.half()
    torch.manual_seed(1)
    labels = torch.randint(0, 32000, (1, 2048))
    labels[:, :400] = -100
    hidden = torch.randn(1, 2048, 512) * 0.5
    monkeypatch.setattr(ce, "_MODULE_HEAD_ROWS", 300, raising = False)
    monkeypatch.setattr(ce, "_CHUNKED_MODULE_HEAD", chunked, raising = False)
    upstream = 2.0 * (scaling or 1.0)  # a loss weight of 2, times the GradScaler scale when there is one
    out = []
    for head, h in ((head16, hidden.half()), (ref, hidden)):
        before = getattr(ce, "_MODULE_HEAD_CHUNKED_CALLS", 0)
        loss = unsloth_fused_ce_loss(trainer = None, hidden_states = h, lm_head_weight = head.base_layer.weight,
                                     lm_head_bias = head.base_layer.bias, labels = labels, torch_compile = False,
                                     lm_head_module = head, scaling = scaling)
        engaged = getattr(ce, "_MODULE_HEAD_CHUNKED_CALLS", 0) - before
        (loss * upstream).backward()
        out.append(([p.grad.float() / upstream for p in head.parameters() if p.requires_grad], engaged))
    (got, engaged), (want, _) = out
    return [((g - w).norm() / w.norm()).item() for g, w in zip(got, want)], engaged


def test_fp16_without_a_grad_scaler_keeps_the_all_rows_forward(monkeypatch):
    # No loss scale: fp16 keeps all-rows, since chunked grads round before the upstream factor.
    want, _ = _fp16_case(monkeypatch, chunked = False)
    got, engaged = _fp16_case(monkeypatch, chunked = True)
    torch.testing.assert_close(got, want, rtol = 1e-3, atol = 0)
    assert engaged == 0


def test_fp16_with_a_grad_scaler_stays_chunked(monkeypatch):
    want, _ = _fp16_case(monkeypatch, chunked = False, scaling = 1024.0)
    got, engaged = _fp16_case(monkeypatch, chunked = True, scaling = 1024.0)
    assert engaged == 1
    assert all(g <= 1.5 * w + 2e-3 for g, w in zip(got, want)), (got, want)


@pytest.mark.parametrize("hook_name", ["_adapter_names_pre_forward_hook", "_alora_offsets_pre_forward_hook"])
def test_per_example_head_hooks_keep_the_all_rows_forward(hook_name, monkeypatch):
    # PEFT's adapter_names / aLoRA offsets hooks index the original batch and positions, not row chunks.
    head = _head()
    hidden = torch.randn(B, T, H, dtype = torch.float64, requires_grad = True)
    hook = lambda module, args, kwargs: None  # noqa: E731
    hook.__name__ = hook_name
    handle = head.register_forward_pre_hook(hook, with_kwargs = True)
    try:
        *_, n = _run(head, hidden, _labels(), True, 4, monkeypatch = monkeypatch)
    finally:
        handle.remove()
    assert n == 0
    *_, n = _run(head, hidden, _labels(), True, 4, monkeypatch = monkeypatch)
    assert n == 1


def test_non_lora_adapters_and_flat_parameters_keep_the_all_rows_forward(monkeypatch):
    # LoHa / LoKr draw rank / module dropout once per forward; FSDP1 flat parameters hide the head's tensors.
    torch.manual_seed(0)
    cfg = peft.LoHaConfig(r = 4, alpha = 8, target_modules = ["lm_head"], rank_dropout = 0.5, init_weights = False)
    loha = peft.get_peft_model(_Toy().to(torch.float64), cfg).train().base_model.model.lm_head
    hidden = torch.randn(B, T, H, dtype = torch.float64, requires_grad = True)
    *_, n = _run(loha, hidden, _labels(), True, 4, monkeypatch = monkeypatch)
    assert n == 0
    head = _head()
    head._is_fsdp_managed_module = True
    *_, n = _run(head, hidden, _labels(), True, 4, monkeypatch = monkeypatch)
    assert n == 0
    del head._is_fsdp_managed_module
    *_, n = _run(head, hidden, _labels(), True, 4, monkeypatch = monkeypatch)
    assert n == 1

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

"""Inkling 2 applies its final logit softcap through a module (`logits = self.logit_softcap(logits)`); the fused
LM-head CE must take it as `logit_softcapping` instead of declining and materialising the fp32 logits."""

import importlib
import inspect
import os

import pytest
import torch
from torch import nn

from unsloth_zoo import compiler
from unsloth_zoo.fused_losses import forward_install
from unsloth_zoo.fused_losses.ast_rewriter import inline_softcap_modules, rewrite_forward_source
from unsloth_zoo.fused_losses.forward_adapter import EMPTY_LOGITS

_INKLING = {
    "inkling2": ("Inkling2ForConditionalGeneration", "Inkling2ForCausalLM"),
    "inkling2_loop": ("Inkling2LoopForConditionalGeneration", "Inkling2LoopForCausalLM"),
}
_STATEMENT = "logits = torch.tanh(logits / self.logit_softcap.softcap) * self.logit_softcap.softcap"
_MODULE_CALL = "logits = self.logit_softcap(logits)"


def _inkling_classes():
    out = []
    for model_type, names in _INKLING.items():
        try:
            mod = importlib.import_module(f"transformers.models.{model_type}.modeling_{model_type}")
        except ImportError:
            continue
        out += [(mod, getattr(mod, n)) for n in names if hasattr(mod, n)]
    return out


@pytest.mark.parametrize("entry", _inkling_classes() or [None], ids = lambda e: e[1].__name__ if e else "absent")
def test_inkling_forwards_fuse_with_the_module_softcap(entry):
    if entry is None:
        pytest.skip("transformers has no Inkling 2 (needs >= 5.24)")
    mod, cls = entry
    src = inspect.getsource(cls.forward)
    if "unsloth_fused_lm_head_loss(" in src:
        # The import hook already installed the fused forward (*ForCausalLM once unsloth_zoo is imported).
        assert forward_install.audit()["patched"][cls.__qualname__]["head_attr"] == "lm_head"
        fused = next(line for line in src.splitlines() if "unsloth_fused_lm_head_loss(" in line)
        assert "logit_softcapping=self.logit_softcap.softcap" in fused and "vocab_size=None" in fused
        assert _STATEMENT not in src and src.count(_MODULE_CALL) == 2
        return
    assert "self.logit_softcap(logits)" in src   # the upstream form this handles
    inlined = inline_softcap_modules(src, cls)
    assert _STATEMENT in inlined and "self.logit_softcap(logits)" not in inlined
    new, route, _ = compiler.fused_lm_head_forward(cls.__name__, cls, mod.__name__, src)
    assert route == "ast", route
    fused = next(line for line in new.splitlines() if "unsloth_fused_lm_head_loss(" in line)
    assert "logit_softcapping=self.logit_softcap.softcap" in fused
    # `vocab_size=logits.shape[-1]` would read logits the fused branch never binds.
    assert "vocab_size=None" in fused and "logits.shape" not in fused
    # Generation (no labels) and the custom-loss path call the softcap module as upstream does (hooks run).
    assert _STATEMENT not in new and new.count(_MODULE_CALL) == 2
    assert "isinstance(self.logit_softcap.softcap, (int, float))" in new


class _Softcap(nn.Module):
    def __init__(self, softcap):
        super().__init__()
        self.softcap = softcap

    def forward(self, hidden_states):
        return torch.tanh(hidden_states / self.softcap) * self.softcap


class _ShiftedSoftcap(nn.Module):
    """Looks alike but is not a plain softcap: must not be inlined."""

    def __init__(self, softcap):
        super().__init__()
        self.softcap = softcap

    def forward(self, hidden_states):
        return torch.tanh(hidden_states / self.softcap) * self.softcap + 1.0


class _Out:
    def __init__(self, loss, logits):
        self.loss, self.logits = loss, logits


def _make_class(name, cap_cls):
    from transformers.loss.loss_utils import ForCausalLMLoss

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.body = nn.Linear(16, 16)
            self.lm_head = nn.Linear(16, 300, bias = False)
            with torch.no_grad():
                self.lm_head.weight.mul_(40)   # logits well past the cap, so the softcap matters
            self.logit_softcap = cap_cls(30.0)
            self.loss_function = ForCausalLMLoss

        def forward(self, x = None, labels = None, **kwargs):
            hidden_states = self.body(x)
            logits = self.lm_head(hidden_states).float()
            logits = self.logit_softcap(logits)
            loss = None
            if labels is not None:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=logits.shape[-1], **kwargs)
            return _Out(loss, logits)

    Toy.__name__ = Toy.__qualname__ = name
    return Toy


def test_lookalike_module_call_is_left_alone():
    cls = _make_class("ShiftedToyForCausalLM", _ShiftedSoftcap)
    src = inspect.getsource(cls.forward)
    assert inline_softcap_modules(src, cls) == src
    assert rewrite_forward_source(inline_softcap_modules(src, cls))[1].head_attr == "logit_softcap"
    assert not forward_install.install_for_class(cls)


def test_fused_loss_and_grads_match_the_materialised_softcap(monkeypatch):
    cls = _make_class("SoftcapToyForCausalLM", _Softcap)
    assert forward_install.install_for_class(cls)
    model = cls()
    x = torch.randn(2, 12, 16)
    labels = torch.randint(0, 300, (2, 12))
    labels[0, :4] = -100      # prompt mask
    labels[1, 9:] = -100      # right padding
    params = [model.body.weight, model.body.bias, model.lm_head.weight]
    for n_items in (None, torch.tensor(2 * int((labels[:, 1:] != -100).sum()))):
        kw = {} if n_items is None else {"num_items_in_batch": n_items}
        got = {}
        for mode in ("1", "0"):
            monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", mode)
            for p in params:
                p.grad = None
            out = model(x = x, labels = labels, **kw)
            out.loss.backward()
            got[mode] = (out.loss.detach(), out.logits, [p.grad.clone() for p in params])
        assert got["0"][1] is EMPTY_LOGITS and got["1"][1].numel() == 2 * 12 * 300
        assert float(got["1"][1].abs().max()) <= 30.0
        torch.testing.assert_close(got["0"][0], got["1"][0], rtol = 1e-5, atol = 1e-6)
        for a, b in zip(got["0"][2], got["1"][2]):
            torch.testing.assert_close(a, b, rtol = 1e-4, atol = 1e-6)


def test_zero_softcap_takes_the_exact_path(monkeypatch):
    """The kernel reads a zero cap as "no softcap"; the module's tanh(x / 0) * 0 zeroes the logits (loss log V)."""
    cls = _make_class("ZeroCapToyForCausalLM", _Softcap)
    model = cls()
    model.logit_softcap.softcap = 0.0
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16)
    labels = torch.randint(0, 300, (2, 8))
    want = model(x = x, labels = labels).loss
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    assert forward_install.install_for_class(cls)
    got = model(x = x, labels = labels).loss
    torch.testing.assert_close(got, want)


def _lora_head(base):
    """A PEFT LoRA Linear around `base` with a nonzero adapter: PEFT 0.18 takes r / lora_alpha / init_lora_weights
    as keywords, later releases also need the LoraConfig."""
    peft = pytest.importorskip("peft")
    from peft.tuners.lora.layer import Linear as LoraLinear

    kwargs = dict(r = 2, lora_alpha = 2)
    if "config" in inspect.signature(LoraLinear.__init__).parameters:
        kwargs["config"] = peft.LoraConfig(r = 2, lora_alpha = 2, init_lora_weights = False)
    else:
        kwargs["init_lora_weights"] = False
    return LoraLinear(base, "default", **kwargs)


def test_unmerged_peft_adapter_on_the_head_still_trains(monkeypatch):
    """A PEFT LoRA head adds its adapter in forward; the fused kernel reads only lm_head.weight, so such a head takes
    the exact path and its adapter gets the gradient UNSLOTH_RETURN_LOGITS=1 gives. Disabled and unmerged: fused."""
    cls = _make_class("AdapterHeadToyForCausalLM", _Softcap)
    assert forward_install.install_for_class(cls)
    torch.manual_seed(0)
    model = cls()
    model.lm_head = _lora_head(model.lm_head)
    x = torch.randn(2, 8, 16)
    labels = torch.randint(0, 300, (2, 8))
    grads, losses = [], []
    for mode in ("1", "0"):
        monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", mode)
        model.zero_grad(set_to_none = True)
        out = model(x = x, labels = labels)
        out.loss.backward()
        grads.append(model.lm_head.lora_B["default"].weight.grad)
        losses.append(out.loss.detach())
    assert grads[1] is not None
    torch.testing.assert_close(grads[1], grads[0])
    torch.testing.assert_close(losses[1], losses[0])
    with _disabled(model.lm_head):
        assert model(x = x, labels = labels).logits.numel() == 0   # adapter off: the fused kernel engages again


class _disabled:
    def __init__(self, layer):
        self.layer = layer

    def __enter__(self):
        self.layer.enable_adapters(False)

    def __exit__(self, *exc):
        self.layer.enable_adapters(True)


def test_merged_then_disabled_peft_head_takes_the_exact_path(monkeypatch):
    """PEFT unmerges a merged head inside its forward when the adapters are disabled; the fused kernel would read the
    merged weight (the adapter's loss) and leave the head merged. Merged and enabled runs base_layer alone on the
    merged weight, which the kernel reads, so it stays fused."""
    cls = _make_class("MergedHeadToyForCausalLM", _Softcap)
    assert forward_install.install_for_class(cls)
    torch.manual_seed(0)
    model = cls()
    model.lm_head = head = _lora_head(model.lm_head)
    x = torch.randn(2, 8, 16)
    labels = torch.randint(0, 300, (2, 8))
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "1")
    head.enable_adapters(False)
    want_base = model(x = x, labels = labels).loss.detach()
    head.enable_adapters(True)
    want_adapter = model(x = x, labels = labels).loss.detach()
    assert not torch.allclose(want_base, want_adapter)   # the adapter changes the loss
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    head.merge()
    out = model(x = x, labels = labels)
    assert out.logits.numel() == 0   # merged and enabled: fused, on the merged weight
    torch.testing.assert_close(out.loss.detach(), want_adapter)
    head.enable_adapters(False)
    out = model(x = x, labels = labels)
    assert out.logits.numel() > 0 and not head.merged
    torch.testing.assert_close(out.loss.detach(), want_base)
    assert model(x = x, labels = labels).logits.numel() == 0   # unmerged and disabled: fused again


def _compiles_fullgraph(model, x, labels):
    torch._dynamo.reset()
    loss = torch.compile(model, backend = "aot_eager", fullgraph = True, dynamic = True)(x = x, labels = labels).loss
    loss.backward()
    return loss.detach()


def test_tensor_softcap_keeps_its_gradient_and_fullgraph(monkeypatch):
    """The kernel takes the cap as a Python float: a trainable cap (nn.Parameter) takes the exact path, keeping its
    gradient, and the guard is a trace-time isinstance (no data-dependent branch under fullgraph)."""
    cls = _make_class("ParamCapToyForCausalLM", _Softcap)
    torch.manual_seed(0)
    model = cls()
    model.logit_softcap.softcap = nn.Parameter(torch.tensor(30.0))
    x = torch.randn(2, 8, 16)
    labels = torch.randint(0, 300, (2, 8))
    want = model(x = x, labels = labels).loss
    want.backward()
    want_grad = model.logit_softcap.softcap.grad.clone()
    assert forward_install.install_for_class(cls)
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    model.zero_grad(set_to_none = True)
    got = model(x = x, labels = labels)
    got.loss.backward()
    torch.testing.assert_close(got.loss.detach(), want.detach())
    torch.testing.assert_close(model.logit_softcap.softcap.grad, want_grad)
    model.zero_grad(set_to_none = True)
    torch.testing.assert_close(_compiles_fullgraph(model, x, labels), want.detach())
    torch.testing.assert_close(model.logit_softcap.softcap.grad, want_grad)


def test_softcap_module_hooks_run_in_the_materialised_branches(monkeypatch):
    """Generation, the no-label call and UNSLOTH_RETURN_LOGITS=1 call the softcap module as upstream does, so its
    forward hooks run there; only the fused branch takes the scalar cap."""
    cls = _make_class("HookedCapToyForCausalLM", _Softcap)
    torch.manual_seed(0)
    model = cls()
    x = torch.randn(2, 8, 16)
    labels = torch.randint(0, 300, (2, 8))
    calls = []
    model.logit_softcap.register_forward_hook(lambda module, args, out: calls.append(1) or out * 0)
    assert forward_install.install_for_class(cls)
    for mode in ("0", "1"):
        monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", mode)
        calls.clear()
        logits = model(x = x).logits
        assert calls == [1] and float(logits.abs().max()) == 0.0
    calls.clear()
    assert float(model(x = x, labels = labels).logits.abs().max()) == 0.0 and calls == [1]
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    assert model(x = x, labels = labels).logits is EMPTY_LOGITS


def test_fused_linear_kill_switch_keeps_the_stock_forward(monkeypatch):
    """UNSLOTH_FUSED_LINEAR=0 runs fused CE behind torch.compiler.disable: the softcap forward is not rewritten, so it
    stays as compilable as upstream. The default switch fuses and compiles in one graph."""
    from unsloth_zoo.fused_losses import chunked_logprob

    x = torch.randn(2, 8, 16)
    labels = torch.randint(0, 300, (2, 8))
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    cls = _make_class("FusedOnToyForCausalLM", _Softcap)
    assert forward_install.install_for_class(cls)
    model = cls()
    want = model(x = x, labels = labels)
    assert want.logits is EMPTY_LOGITS
    torch.testing.assert_close(_compiles_fullgraph(model, x, labels), want.loss.detach())
    monkeypatch.setattr(chunked_logprob, "_FUSED_LINEAR", False)
    cls = _make_class("FusedOffToyForCausalLM", _Softcap)
    src = inspect.getsource(cls.forward)
    assert inline_softcap_modules(src, cls) == src
    assert not forward_install.install_for_class(cls)
    assert "unsloth_fused_lm_head_loss" not in inspect.getsource(cls.forward)

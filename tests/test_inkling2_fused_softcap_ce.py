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
    # Generation (no labels) and the custom-loss path keep the softcap.
    assert new.count(_STATEMENT) == 2


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

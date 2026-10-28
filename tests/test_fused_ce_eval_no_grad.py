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

"""Fused CE under no_grad / inference_mode: same loss, no gradient pass, hidden states untouched."""
import pytest
import torch

pytestmark = pytest.mark.usefixtures("compile_enabled")


def _ce_module():
    try:
        import unsloth_zoo.fused_losses.cross_entropy_loss as ce
    except ImportError as e:
        pytest.skip(f"unsloth_zoo import unavailable: {e}")
    return ce


def _inputs(head_grad, bias, seed = 0):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(2, 37, 16, generator = g, requires_grad = True)
    weight = (torch.randn(64, 16, generator = g) * 0.5).requires_grad_(head_grad)
    b = (torch.randn(64, generator = g) * 0.1).requires_grad_(head_grad) if bias else None
    labels = torch.randint(0, 64, (2, 37), generator = g)
    labels[0, :5] = -100
    return hidden, weight, b, labels


def _loss(ce, hidden, weight, b, labels, overwrite = False):
    return ce.unsloth_fused_ce_loss(
        None, hidden, weight, b, labels,
        torch_compile = False, overwrite = overwrite, n_chunks = 3,
    )


@pytest.mark.parametrize("mode", ["no_grad", "inference_mode"])
@pytest.mark.parametrize("head_grad", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("overwrite", [False, True])
def test_eval_skips_gradients(monkeypatch, mode, head_grad, bias, overwrite):
    ce = _ce_module()
    hidden, weight, b, labels = _inputs(head_grad, bias)
    reference = _loss(ce, hidden.detach().clone().requires_grad_(), weight, b, labels)

    calls = []
    real = torch.func.grad_and_value
    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(torch.func, "grad_and_value", counting)

    before = hidden.detach().clone()
    context = torch.no_grad() if mode == "no_grad" else torch.inference_mode()
    with context:
        loss = _loss(ce, hidden, weight, b, labels, overwrite = overwrite)
    assert calls == []
    assert not loss.requires_grad
    assert torch.equal(loss, reference.detach())
    assert torch.equal(hidden.detach(), before)


@pytest.mark.parametrize("bias", [False, True])
def test_training_after_eval_matches_reference(bias):
    ce = _ce_module()
    hidden, weight, b, labels = _inputs(True, bias)
    with torch.no_grad():
        _loss(ce, hidden, weight, b, labels)
    loss = _loss(ce, hidden, weight, b, labels)
    loss.backward()

    h2 = hidden.detach().clone().requires_grad_()
    w2 = weight.detach().clone().requires_grad_()
    b2 = b.detach().clone().requires_grad_() if bias else None
    logits = h2[:, :-1] @ w2.t() + (b2 if bias else 0)
    ref = torch.nn.functional.cross_entropy(logits.reshape(-1, 64).float(), labels[:, 1:].reshape(-1))
    ref.backward()
    torch.testing.assert_close(loss, ref)
    torch.testing.assert_close(hidden.grad, h2.grad)
    torch.testing.assert_close(weight.grad, w2.grad)
    if bias:
        torch.testing.assert_close(b.grad, b2.grad)


def test_eval_probe_does_not_settle_compile_support(monkeypatch):
    ce = _ce_module()
    monkeypatch.setattr(ce, "_FUSED_CE_COMPILE_SUPPORTED", None)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: (lambda *a, **k: fn(*a, **k)))
    hidden, weight, b, labels = _inputs(False, False)
    with torch.no_grad():
        ce.unsloth_fused_ce_loss(None, hidden, weight, b, labels, torch_compile = True, n_chunks = 3)
    assert ce._FUSED_CE_COMPILE_SUPPORTED is None
    ce.unsloth_fused_ce_loss(None, hidden, weight, b, labels, torch_compile = True, n_chunks = 3)
    assert ce._FUSED_CE_COMPILE_SUPPORTED is True

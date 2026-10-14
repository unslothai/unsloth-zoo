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

import importlib
import importlib.util
import math
import os

import pytest
import torch
import torch.nn.functional as F


def _load(name):
    try:
        return importlib.import_module(f"unsloth_zoo.speculative.{name}")
    except ImportError:
        path = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth_zoo", "speculative", f"{name}.py")
        spec = importlib.util.spec_from_file_location(f"_speculative_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


losses = _load("losses")


def _reference(hidden, weight, labels, loss_type, teacher_hidden = None, ignore_index = -100, lk_eta = 3.0):
    valid = labels != ignore_index
    logits = hidden[valid] @ weight.T
    y = labels[valid]
    if loss_type == "ce":
        return F.cross_entropy(logits, y, reduction = "sum")
    with torch.no_grad():
        p = torch.softmax(teacher_hidden[valid] @ weight.T, -1)
    q = torch.softmax(logits, -1)
    kl = (p * (p.log() - q.log())).sum(-1)
    tv = 0.5 * (p - q).abs().sum(-1)
    if loss_type == "kl_div":
        return kl.sum()
    if loss_type == "tv":
        return tv.sum()
    lam = torch.exp(-lk_eta * torch.minimum(p, q).sum(-1)).detach()
    return (lam * kl + (1 - lam) * tv).sum()


def test_step_weights():
    w = losses.fastmtp_step_weights(0.6, 3)
    assert sum(w) == pytest.approx(1.0)
    assert w == pytest.approx([1 / 1.96, 0.6 / 1.96, 0.36 / 1.96])
    assert losses.fastmtp_step_weights(0.6, 1) == [1.0]
    with pytest.raises(ValueError):
        losses.fastmtp_step_weights(0.6, 0)


def test_closed_form_values():
    # V = 2: draft logits [0, 0] -> q = [.5, .5]; teacher logits [log 3, 0] -> p = [.75, .25].
    weight = torch.eye(2, dtype = torch.float64)
    hidden = torch.zeros(1, 2, dtype = torch.float64)
    teacher = torch.tensor([[math.log(3.0), 0.0]], dtype = torch.float64)
    labels = torch.tensor([0])
    ce, n, _ = losses.chunked_lm_loss(hidden, weight, labels, "ce")
    assert n == 1 and ce.item() == pytest.approx(math.log(2))
    kl, _, _ = losses.chunked_lm_loss(hidden, weight, labels, "kl_div", teacher_hidden = teacher)
    assert kl.item() == pytest.approx(0.75 * math.log(1.5) + 0.25 * math.log(0.5))
    tv, _, _ = losses.chunked_lm_loss(hidden, weight, labels, "tv", teacher_hidden = teacher)
    assert tv.item() == pytest.approx(0.25)
    lk, _, _ = losses.chunked_lm_loss(hidden, weight, labels, "lk_hybrid", teacher_hidden = teacher, lk_eta = 3.0)
    lam = math.exp(-3.0 * 0.75)
    assert lk.item() == pytest.approx(lam * kl.item() + (1 - lam) * 0.25)


@pytest.mark.parametrize("loss_type", losses.LOSS_TYPES)
@pytest.mark.parametrize("chunk_size", [1, 3, 7, 64])
def test_matches_unchunked_reference(loss_type, chunk_size):
    torch.manual_seed(0)
    N, H, V = 11, 5, 13
    weight = torch.randn(V, H, dtype = torch.float64)
    hidden = torch.randn(N, H, dtype = torch.float64, requires_grad = True)
    teacher = torch.randn(N, H, dtype = torch.float64)
    labels = torch.randint(0, V, (N,))
    labels[[2, 5, 9]] = -100
    got, n, n_correct = losses.chunked_lm_loss(
        hidden, weight, labels, loss_type, teacher_hidden = teacher, chunk_size = chunk_size,
    )
    (g_got,) = torch.autograd.grad(got, hidden)
    ref = _reference(hidden, weight, labels, loss_type, teacher_hidden = teacher)
    (g_ref,) = torch.autograd.grad(ref, hidden)
    assert n == 8
    assert torch.allclose(got, ref, atol = 1e-10)
    assert torch.allclose(g_got, g_ref, atol = 1e-10)
    assert g_got[[2, 5, 9]].abs().sum() == 0
    valid = labels != -100
    assert n_correct == int(((hidden[valid] @ weight.T).argmax(-1) == labels[valid]).sum())


def test_frozen_head_still_passes_grad_and_bf16_upcast():
    torch.manual_seed(0)
    weight = torch.randn(17, 8, dtype = torch.bfloat16)
    hidden = torch.randn(6, 8, dtype = torch.bfloat16, requires_grad = True)
    labels = torch.randint(0, 17, (6,))
    loss, n, _ = losses.chunked_lm_loss(hidden, weight, labels, "ce", chunk_size = 4)
    assert loss.dtype == torch.float32
    loss.backward()
    assert weight.grad is None and hidden.grad is not None and hidden.grad.abs().sum() > 0


def test_all_ignored_is_zero_and_backward_safe():
    hidden = torch.randn(4, 3, requires_grad = True)
    loss, n, n_correct = losses.chunked_lm_loss(hidden, torch.randn(5, 3), torch.full((4,), -100))
    assert n == 0 and n_correct == 0 and loss.item() == 0.0
    loss.backward()
    assert hidden.grad.abs().sum() == 0


def test_lk_hybrid_lambda_is_stop_grad():
    # With lam detached, d(lk)/dh = lam * dKL + (1 - lam) * dTV exactly.
    torch.manual_seed(1)
    weight = torch.randn(9, 4, dtype = torch.float64)
    hidden = torch.randn(3, 4, dtype = torch.float64, requires_grad = True)
    teacher = torch.randn(3, 4, dtype = torch.float64)
    labels = torch.zeros(3, dtype = torch.long)
    with torch.no_grad():
        p = torch.softmax(teacher @ weight.T, -1)
        q = torch.softmax(hidden @ weight.T, -1)
        lam = torch.exp(-3.0 * torch.minimum(p, q).sum(-1))
    def kl_rows(h):
        return (p * (p.log() - torch.log_softmax(h @ weight.T, -1))).sum(-1)
    def tv_rows(h):
        return 0.5 * (p - torch.softmax(h @ weight.T, -1)).abs().sum(-1)
    (g_expected,) = torch.autograd.grad((lam * kl_rows(hidden) + (1 - lam) * tv_rows(hidden)).sum(), hidden)
    lk, _, _ = losses.chunked_lm_loss(hidden, weight, labels, "lk_hybrid", teacher_hidden = teacher, chunk_size = 2)
    (g_got,) = torch.autograd.grad(lk, hidden)
    assert torch.allclose(g_got, g_expected, atol = 1e-10)


def _saved_bytes(fn):
    saved = []
    def pack(tensor):
        saved.append(tensor.numel() * tensor.element_size())
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        out = fn()
    return sum(saved), out


def test_saved_activations_scale_with_chunk_not_rows(monkeypatch):
    # Checkpointed chunks keep only their inputs for backward, never the [N, V] logits.
    torch.manual_seed(0)
    N, H, V = 256, 2, 4096
    weight = torch.randn(V, H)
    labels = torch.randint(0, V, (N,))
    full_logits_bytes = N * V * 4
    hidden = torch.randn(N, H, requires_grad = True)
    ckpt_bytes, (loss, _, _) = _saved_bytes(lambda: losses.chunked_lm_loss(hidden, weight, labels, "ce", chunk_size = 32))
    loss.backward()
    grad_ckpt = hidden.grad.clone()
    monkeypatch.setattr(losses, "checkpoint", lambda fn, *args, **kwargs: fn(*args))
    hidden.grad = None
    plain_bytes, (loss, _, _) = _saved_bytes(lambda: losses.chunked_lm_loss(hidden, weight, labels, "ce", chunk_size = 32))
    loss.backward()
    assert plain_bytes >= full_logits_bytes
    assert ckpt_bytes < full_logits_bytes / 4
    assert torch.allclose(grad_ckpt, hidden.grad)


def test_errors():
    h, w, y = torch.randn(2, 3), torch.randn(4, 3), torch.zeros(2, dtype = torch.long)
    with pytest.raises(ValueError):
        losses.chunked_lm_loss(h, w, y, "bogus")
    with pytest.raises(ValueError):
        losses.chunked_lm_loss(h, w, y, "kl_div")
    with pytest.raises(ValueError):
        losses.chunked_lm_loss(h, w, torch.zeros(3, dtype = torch.long))

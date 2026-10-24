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

"""Chunked cross entropy (gradient taken per chunk in forward, unlabelled rows skipped) against an fp64 oracle and
the torch.func fused CE it replaces in unsloth_fused_ce_loss."""

import pytest
import torch

import unsloth_zoo.fused_losses.cross_entropy_loss as cel
from unsloth_zoo.fused_losses.chunked_logprob import UnslothChunkedCE, unsloth_chunked_ce_loss

B, S, H, V = 2, 40, 16, 64


def _oracle(hidden, weight, bias, labels, mask, n_items, cap, mul, div):
    hd = hidden.detach().double().requires_grad_(True)
    wd = weight.detach().double().requires_grad_(weight.requires_grad)
    bd = None if bias is None else bias.detach().double().requires_grad_(bias.requires_grad)
    z = hd @ wd.t() + (0 if bd is None else bd)
    if mul:
        z = z * mul
    if div:
        z = z / div
    if cap:
        z = cap * torch.tanh(z / cap)
    shifted = torch.full_like(labels, -100)
    shifted[:, :-1] = labels[:, 1:]
    if mask is not None:
        shifted[:, :-1][mask[:, 1:] == 0] = -100
    loss = torch.nn.functional.cross_entropy(z.reshape(-1, V), shifted.reshape(-1), reduction = "sum")
    count = (shifted != -100).sum()
    loss = loss / (n_items if n_items is not None else (count if count > 0 else 1))
    loss.backward()
    return loss, hd.grad, wd.grad, None if bd is None else bd.grad


@pytest.mark.parametrize("rows", [1, 7, 1000])
@pytest.mark.parametrize("cap,mul,div", [(0.0, 0.0, 0.0), (30.0, 0.0, 0.0), (5.0, 2.0, 0.0), (0.0, 0.0, 3.0)])
@pytest.mark.parametrize("train_head", [False, True])
@pytest.mark.parametrize("use_mask,n_items,scaling", [(False, None, None), (True, 23, None), (False, None, 512.0)])
def test_chunked_ce_matches_fp64(rows, cap, mul, div, train_head, use_mask, n_items, scaling):
    torch.manual_seed(0)
    hidden = torch.randn(B, S, H, requires_grad = True)
    weight = (torch.randn(V, H) / 2).requires_grad_(train_head)
    bias = torch.randn(V).requires_grad_(train_head)
    labels = torch.randint(0, V, (B, S))
    labels[:, :9] = -100
    mask = torch.ones(B, S, dtype = torch.long)
    mask[1, -6:] = 0
    mask = mask if use_mask else None
    loss = unsloth_chunked_ce_loss(
        hidden, weight, bias, labels, mask = mask, n_items = None if n_items is None else torch.tensor(n_items),
        scaling = scaling, logit_softcapping = cap, logit_scale_multiply = mul, logit_scale_divide = div,
        rows_per_chunk = rows,
    )
    loss.backward()
    ref, gh, gw, gb = _oracle(hidden, weight, bias, labels, mask, n_items, cap, mul, div)
    torch.testing.assert_close(loss.double(), ref.detach(), rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(hidden.grad.double(), gh, rtol = 1e-4, atol = 1e-7)
    if train_head:
        torch.testing.assert_close(weight.grad.double(), gw, rtol = 1e-4, atol = 1e-7)
        torch.testing.assert_close(bias.grad.double(), gb, rtol = 1e-4, atol = 1e-7)


def test_all_masked_gives_zero_loss_and_grads():
    hidden = torch.randn(1, 6, H, requires_grad = True)
    weight = torch.randn(V, H, requires_grad = True)
    loss = unsloth_chunked_ce_loss(hidden, weight, None, torch.full((1, 6), -100))
    loss.backward()
    assert float(loss) == 0.0
    assert torch.count_nonzero(hidden.grad) == 0 and torch.count_nonzero(weight.grad) == 0


def test_unlabelled_rows_never_reach_the_head(monkeypatch):
    seen = []
    forward = UnslothChunkedCE.forward

    def spy(ctx, hidden, *args):
        seen.append(hidden.shape[0])
        return forward(ctx, hidden, *args)

    monkeypatch.setattr(UnslothChunkedCE, "forward", staticmethod(spy))
    labels = torch.randint(0, V, (B, S))
    labels[:, :30] = -100
    unsloth_chunked_ce_loss(torch.randn(B, S, H), torch.randn(V, H), None, labels)
    assert seen == [B * (S - 30)]  # shifted: positions 29..38 predict tokens 30..39


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_fused_ce_routes_plain_ce_only(monkeypatch, label_smoothing):
    calls = {"new": 0}
    real = cel._chunked_ce_opaque

    def spy(*args):
        calls["new"] += 1
        return real(*args)

    monkeypatch.setattr(cel, "_chunked_ce_opaque", spy)
    monkeypatch.setattr(cel, "_CHUNKED_CE", True)
    torch.manual_seed(1)
    hidden = torch.randn(B, S, H, requires_grad = True)
    weight = torch.randn(V, H)
    labels = torch.randint(0, V, (B, S))
    loss = cel.unsloth_fused_ce_loss(None, hidden, weight, None, labels, label_smoothing = label_smoothing)
    assert calls["new"] == (1 if label_smoothing == 0.0 else 0)
    loss.backward()
    monkeypatch.setattr(cel, "_CHUNKED_CE", False)
    hidden2 = hidden.detach().clone().requires_grad_(True)
    old = cel.unsloth_fused_ce_loss(None, hidden2, weight, None, labels, label_smoothing = label_smoothing)
    old.backward()
    torch.testing.assert_close(loss, old, rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(hidden.grad, hidden2.grad, rtol = 1e-4, atol = 1e-6)


def test_zero_scaling_gives_finite_gradients():
    hidden = torch.randn(B, S, H, requires_grad = True)
    labels = torch.randint(0, V, (B, S))
    loss = unsloth_chunked_ce_loss(hidden, torch.randn(V, H), None, labels, scaling = 0.0)
    loss.backward()
    assert torch.isfinite(hidden.grad).all() and hidden.grad.abs().sum() > 0


@pytest.mark.parametrize("flag", ["FP32_HEAD_GRAD", "_FUSED_CE_COMPILE_DISABLED"])
def test_old_path_opt_ins_keep_the_old_path(monkeypatch, flag):
    calls = {"new": 0}
    monkeypatch.setattr(cel, "_chunked_ce_opaque", lambda *a: calls.__setitem__("new", 1))
    monkeypatch.setattr(cel, "_CHUNKED_CE", True)
    monkeypatch.setattr(cel, flag, True)
    hidden = torch.randn(B, S, H, requires_grad = True)
    loss = cel.unsloth_fused_ce_loss(None, hidden, torch.randn(V, H), None, torch.randint(0, V, (B, S)))
    loss.backward()
    assert calls["new"] == 0 and torch.isfinite(hidden.grad).all()

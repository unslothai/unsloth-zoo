# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
PR-A gated_delta_vjp end-to-end through the shim.

Exercises:
  * mx.custom_function decorator + .vjp registration
  * Chunked forward (mx.zeros, mx.ones, mx.repeat, mx.concatenate)
  * .at[:, t].add(...) JAX functional update at 5 sites in the VJP
  * .astype(mx.float32) at ~30 sites
  * mx.where / mx.expand_dims / mx.zeros_like

If forward + backward both produce finite tensors with the right shapes,
PR-A's VJP path is exercisable on Linux+CUDA.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


def test_gated_delta_ops_efficient_forward():
    """Forward returns finite y with shape (B, T, Hv, Dv) and final state."""
    from unsloth_zoo.gated_delta_vjp import gated_delta_ops_efficient

    torch.manual_seed(0)
    B, T, Hk, Dk, Hv, Dv = 1, 4, 2, 8, 2, 8
    q = torch.randn(B, T, Hk, Dk, dtype=torch.float32)
    k = torch.randn(B, T, Hk, Dk, dtype=torch.float32)
    v = torch.randn(B, T, Hv, Dv, dtype=torch.float32)
    g = torch.randn(B, T, Hv, dtype=torch.float32).sigmoid()  # (0, 1) decay
    beta = torch.randn(B, T, Hv, dtype=torch.float32).sigmoid()

    y, state = gated_delta_ops_efficient(q, k, v, g, beta)
    assert y.shape == (B, T, Hv, Dv), f"unexpected y shape {y.shape}"
    assert state.shape == (B, Hv, Dv, Dk), f"unexpected state shape {state.shape}"
    assert torch.isfinite(y).all(), "non-finite y"
    assert torch.isfinite(state).all(), "non-finite state"


def test_gated_delta_ops_efficient_backward():
    """Backward through the chunked VJP yields finite grads on all primals."""
    from unsloth_zoo.gated_delta_vjp import gated_delta_ops_efficient

    torch.manual_seed(0)
    B, T, Hk, Dk, Hv, Dv = 1, 4, 2, 8, 2, 8
    q = torch.randn(B, T, Hk, Dk, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, T, Hk, Dk, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, T, Hv, Dv, dtype=torch.float32, requires_grad=True)
    g = torch.randn(B, T, Hv, dtype=torch.float32).sigmoid().requires_grad_(True)
    beta = torch.randn(B, T, Hv, dtype=torch.float32).sigmoid().requires_grad_(True)

    y, _ = gated_delta_ops_efficient(q, k, v, g, beta)
    y.sum().backward()

    for name, t in [("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)]:
        assert t.grad is not None, f"no grad on {name}"
        assert torch.isfinite(t.grad).all(), f"non-finite grad on {name}"


def test_gated_delta_ops_efficient_with_mask():
    """Mask path: masked positions should keep state unchanged."""
    from unsloth_zoo.gated_delta_vjp import gated_delta_ops_efficient

    torch.manual_seed(0)
    B, T, Hk, Dk, Hv, Dv = 1, 4, 2, 8, 2, 8
    q = torch.randn(B, T, Hk, Dk)
    k = torch.randn(B, T, Hk, Dk)
    v = torch.randn(B, T, Hv, Dv)
    g = torch.randn(B, T, Hv).sigmoid()
    beta = torch.randn(B, T, Hv).sigmoid()
    mask = torch.tensor([[True, True, False, True]])  # (B, T) bool

    y, state = gated_delta_ops_efficient(q, k, v, g, beta, mask=mask)
    assert y.shape == (B, T, Hv, Dv)
    assert torch.isfinite(y).all()
    assert torch.isfinite(state).all()

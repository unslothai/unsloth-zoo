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

"""unsloth_fused_ce_loss must not call inspect.signature per call.

Under torch.compile dynamo ignores functools.cache, so a per-call signature lookup is traced
and graph-breaks every compiled model forward that reaches the fused loss.
"""
import importlib

import pytest


@pytest.fixture(autouse = True)
def _allow_cpu_import(monkeypatch):
    monkeypatch.setenv("UNSLOTH_ALLOW_CPU", "1")


def _ce():
    try:
        return importlib.import_module("unsloth_zoo.fused_losses.cross_entropy_loss")
    except ImportError as e:
        pytest.skip(f"unsloth_zoo import unavailable: {e}")


def test_fused_ce_loss_does_not_inspect_signature_per_call(monkeypatch):
    import torch

    ce = _ce()
    ce._get_mapping.cache_clear()

    def no_signature(*args, **kwargs):
        raise AssertionError("inspect.signature called on the fused loss hot path")

    monkeypatch.setattr(ce.inspect, "signature", no_signature)
    torch.manual_seed(0)
    hidden = torch.randn(2, 8, 16, requires_grad = True)
    weight = torch.randn(32, 16, requires_grad = True)
    labels = torch.randint(0, 32, (2, 8))
    loss = ce.unsloth_fused_ce_loss(
        None, hidden, weight, None, labels, target_gb = 1, torch_compile = False,
    )
    loss.backward()

    reference = torch.nn.functional.cross_entropy(
        (hidden[:, :-1] @ weight.T).float().reshape(-1, 32), labels[:, 1:].reshape(-1),
    )
    assert torch.allclose(loss.detach(), reference.detach(), atol = 1e-4)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_precomputed_mapping_matches_forward_signature():
    import inspect

    ce = _ce()
    parameters = dict(inspect.signature(ce.UnslothFusedLoss.forward).parameters)
    parameters.pop("ctx", None)
    assert ce._FUSED_LOSS_PARAMETERS == tuple(parameters)
    assert ce._FUSED_LOSS_DEFAULTS == tuple(p.default for p in parameters.values())

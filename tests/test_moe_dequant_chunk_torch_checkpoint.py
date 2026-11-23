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

"""The automatic dequant chunk decision under torch's own non-reentrant checkpoint (Inkling 2 pipeline stages): the
recompute must repeat its forward's decision even when free memory moved in between, else it saves a different number
of tensors and backward raises CheckpointError. CPU only: the headroom reading is patched."""

import os

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

import unsloth_zoo.temporary_patches.moe_utils as mu

MINI = [
    (128, (2304, 2304)),
    (128, (2304, 1152)),
]  # Inkling-2-Mini gate_up, down: 55 experts per chunk on an L4
CUDA0 = torch.device("cuda", 0)
needs_hook_query = pytest.mark.skipif(
    not hasattr(torch._C._autograd, "_top_saved_tensors_default_hooks"),
    reason="torch without the saved-tensors hook query",
)


class _Experts(nn.Module):
    """Runs its experts in the chunks the automatic decision picks; every chunk saves its own tensors."""

    _unsloth_dequant_scratch_fraction = 0.025

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(128, 4, 4) * 0.1)
        self.chunks = []

    def forward(self, x):
        chunk = mu._auto_dequant_chunk_experts(
            MINI,
            torch.float16,
            self,
            CUDA0,
            checkpoint_call=mu._gc_moe_checkpoint_call(),
        )
        self.chunks.append(chunk)
        step = chunk or 128
        return torch.cat(
            [
                torch.einsum("bi,eij->bej", x, self.w[i : i + step]).tanh()
                for i in range(0, 128, step)
            ],
            1,
        )


def _memory(monkeypatch, forward_gib, backward_gib):
    """Free memory reads `forward_gib` outside backward and `backward_gib` inside it (the recompute)."""
    monkeypatch.delenv("UNSLOTH_MOE_DEQUANT_MAX_MB", raising=False)
    monkeypatch.delenv("UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS", raising=False)
    monkeypatch.setitem(mu._DEVICE_TOTAL_BYTES, 0, int(21.96 * 2**30))  # L4

    def fake(index):
        gib = backward_gib if torch._C._current_graph_task_id() != -1 else forward_gib
        free = int(gib * 2**30)
        return free, lambda: free

    monkeypatch.setattr(mu, "_device_headroom_bytes", fake)


def _grads(layer, x, use_checkpoint):
    layer.w.grad = None
    x = x.detach().requires_grad_()
    out = checkpoint(layer, x, use_reentrant=False) if use_checkpoint else layer(x)
    (out * out).sum().backward()
    return layer.w.grad.clone(), x.grad.clone()


@needs_hook_query
@pytest.mark.parametrize("forward_gib, backward_gib", [(8.0, 2.5), (2.5, 8.0)])
def test_non_reentrant_recompute_repeats_the_forward_decision(
    monkeypatch, forward_gib, backward_gib
):
    torch.manual_seed(0)
    layer, x = _Experts(), torch.randn(3, 4)
    _memory(monkeypatch, forward_gib, forward_gib)
    reference = _grads(layer, x, False)
    _memory(monkeypatch, forward_gib, backward_gib)
    layer.chunks.clear()
    got = _grads(layer, x, True)
    assert layer.chunks[0] == layer.chunks[1] == (55 if forward_gib < 3 else 0)
    for a, b in zip(got, reference):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@needs_hook_query
def test_in_flight_microbatches_keep_their_own_decisions(monkeypatch):
    """Pipeline schedules run several checkpointed forwards before their backwards, in either order: each recompute
    takes its own forward's decision, not the latest one on the module."""
    torch.manual_seed(0)
    layer = _Experts()
    xs = [torch.randn(3, 4, requires_grad=True) for _ in range(2)]
    outs = []
    for x, gib in zip(xs, (8.0, 2.5)):
        _memory(monkeypatch, gib, gib)
        outs.append(checkpoint(layer, x, use_reentrant=False))
    assert layer.chunks == [0, 55]
    for i, gib in ((1, 8.0), (0, 2.5)):
        _memory(monkeypatch, gib, gib)
        (outs[i] * outs[i]).sum().backward()
    assert layer.chunks == [0, 55, 55, 0]


def test_outside_a_checkpoint_every_call_decides_fresh(monkeypatch):
    layer, x = _Experts(), torch.randn(3, 4, requires_grad=True)
    for gib, want in ((8.0, 0), (2.5, 55), (8.0, 0)):
        _memory(monkeypatch, gib, gib)
        layer(x)
        assert layer.chunks[-1] == want


def test_explicit_chunk_setting_still_wins(monkeypatch):
    _memory(monkeypatch, 8.0, 2.5)
    monkeypatch.setenv("UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS", "0")
    layer, x = _Experts(), torch.randn(3, 4, requires_grad=True)
    (checkpoint(layer, x, use_reentrant=False) ** 2).sum().backward()
    assert layer.chunks == [0, 0]

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

"""The routed gather hidden_states[sorted_indices // top_k] must have a reproducible backward
under torch.compile (Unsloth block compile puts the MoE inside one Inductor region).

Autograd differentiates that gather into an accumulating index_put with top_k duplicate rows
per token. Inductor lowers it to tl.atomic_add, so two identical backward passes in one
process gave different dX, and every LoRA gradient upstream of the first MoE layer differed
(attention LoRA included, even with frozen experts). gather_routed_rows sums the top_k rows in
a fixed order instead.
"""

import pytest
import torch
import torch.nn as nn

from unsloth_zoo.temporary_patches import moe_utils as mu

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or not mu._check_torch_grouped_mm_supported(),
    reason = "needs CUDA grouped_mm",
)
DT = torch.bfloat16


class Qwen3Experts(nn.Module):
    def __init__(self, E, H, I, gen):
        super().__init__()
        self.num_experts = E
        rnd = lambda *s, scale: nn.Parameter((torch.randn(*s, generator = gen) * scale).to("cuda", DT), requires_grad = False)
        self.gate_up_proj = rnd(E, 2 * I, H, scale = H ** -0.5)
        self.down_proj = rnd(E, H, I, scale = I ** -0.5)
        self.act_fn = nn.SiLU()


def _case(T = 2048, E = 16, H = 256, I = 128, k = 8, rank = 8):
    gen = torch.Generator().manual_seed(0)
    m = Qwen3Experts(E, H, I, gen)
    x = (torch.randn(T, H, generator = gen) * 0.5).to("cuda", DT)
    topv, topi = torch.topk(torch.randn(T, E, generator = gen).to("cuda"), k, dim = -1)
    w = torch.softmax(topv, dim = -1).to(DT)
    dy = torch.randn(T, H, generator = gen).to("cuda", DT)
    mk = lambda *s: (torch.randn(*s, generator = gen) * 0.05).to("cuda", DT).requires_grad_(True)
    lora = {"gate_up_proj": (mk(E, H, rank), mk(E, rank, 2 * I), 2.0), "down_proj": (mk(E, I, rank), mk(E, rank, H), 2.0)}
    return m, x, topi, w, dy, lora


@needs_cuda
@pytest.mark.parametrize("top_k", [4, 8])
@pytest.mark.parametrize("dynamic", [False, True])
def test_compiled_grouped_mm_backward_has_no_atomics_and_is_bitwise_reproducible(monkeypatch, top_k, dynamic):
    from torch._inductor.utils import run_and_get_code

    m, x, topi, w, dy, lora = _case(k = top_k)
    monkeypatch.setattr(mu, "take_moe_lora_stash", lambda module, name: lora.get(name))
    # Eager reference: the sorted index_put backward, reproducible but not the compiled path.
    xe = x.detach().clone().requires_grad_(True)
    mu.forward_native_grouped_mm(m, xe, topi, w).backward(dy)
    for a, b, _ in lora.values():
        a.grad = b.grad = None
    torch._dynamo.reset()
    # Block compile marks the sequence dim dynamic; an unrolled spelling was wrong only there.
    compiled = torch.compile(mu.forward_native_grouped_mm, fullgraph = False, dynamic = dynamic)

    def step():
        for a, b, _ in lora.values():
            a.grad = b.grad = None
        xg = x.detach().clone().requires_grad_(True)
        compiled(m, xg, topi, w).backward(dy)
        torch.cuda.synchronize()
        return [xg.grad.clone()] + [t.grad.clone() for a, b, _ in lora.values() for t in (a, b)]

    with torch._inductor.config.patch(fx_graph_cache = False):
        first, codes = run_and_get_code(step)
        # The gather's backward is the only accumulating scatter in this region.
        assert not any("tl.atomic_add(" in c for c in codes), "routed gather backward lowered to atomics"
        rel = ((first[0].double() - xe.grad.double()).norm() / xe.grad.double().norm()).item()
        assert rel < 2e-2, rel
        for _ in range(5):
            again = step()
            for g0, g1 in zip(first, again):
                assert torch.equal(g0, g1)
    torch._dynamo.reset()


@needs_cuda
@pytest.mark.parametrize("top_k", [1, 2, 8])
def test_gather_routed_rows_matches_indexing(top_k):
    T, H, E = 512, 64, 16
    gen = torch.Generator().manual_seed(top_k)
    x = torch.randn(T, H, generator = gen, dtype = torch.float64).cuda()
    flat = torch.randint(0, E, (T * top_k,), generator = gen).cuda()
    sorted_indices = flat.argsort(stable = True)
    g = torch.randn(T * top_k, H, generator = gen, dtype = torch.float64).cuda()

    xr = x.clone().requires_grad_(True)
    ref = xr[sorted_indices // top_k]
    ref.backward(g)

    fn = torch.compile(mu.gather_routed_rows, fullgraph = True)
    xc = x.clone().requires_grad_(True)
    got = fn(xc, sorted_indices, top_k)
    got.backward(g)
    assert torch.equal(got, ref)
    torch.testing.assert_close(xc.grad, xr.grad, rtol = 0, atol = 1e-12)

    # Eager keeps plain indexing (identical forward and backward to before).
    xe = x.clone().requires_grad_(True)
    mu.gather_routed_rows(xe, sorted_indices, top_k).backward(g)
    assert torch.equal(xe.grad, xr.grad)
    torch._dynamo.reset()

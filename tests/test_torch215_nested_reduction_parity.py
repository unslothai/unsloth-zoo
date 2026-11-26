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

"""torch 2.15 turns Inductor's nested-reduction fusion on by default. It is the same fusion family as the
2.10-2.14 mix-order reduction that wrote wrong trainable-norm gradients, so pin its numerics: compiled norm
training blocks with the knob on must match the knob off and an fp64 eager reference, and the fusion must
really fire on the norm + block-amax graph it targets (else the parity cases prove nothing)."""

import pytest
import torch
import torch.nn.functional as F

EPS = 1e-6


def _has_nested_reduction():
    try:
        import torch._inductor.config as config
    except Exception:
        return False
    return hasattr(config.triton, "nested_reduction")


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _has_nested_reduction(),
    reason = "needs CUDA and a torch with triton.nested_reduction (2.14+, on by default from 2.15)",
)


def _rms_llama(x, res, w, lin):
    h = x + res
    hf = h.float()
    y = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim = True) + EPS)
    return F.linear(w * y.to(x.dtype), lin), h


def _rms_gemma(x, res, w, lin):
    h = x + res
    hf = h.float()
    y = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim = True) + EPS)
    return F.linear((y * (1.0 + w.float())).to(x.dtype), lin), h


def _layer_norm(x, res, w, lin):
    h = x + res
    y = F.layer_norm(h.float(), (h.shape[-1],), w.float(), None, EPS).to(x.dtype)
    return F.linear(y, lin), h


def _compile(fn, nested):
    import torch._inductor.config as config
    torch._dynamo.reset()
    config.triton.nested_reduction = nested
    return torch.compile(fn, fullgraph = True, dynamic = False)


def _grads(fn, inputs, dtype, go, goh):
    leaves = [t.detach().to(dtype).requires_grad_(True) for t in inputs]
    out, h = fn(*leaves)
    ((out.double() * go).sum() + (h.double() * goh).sum()).backward()
    return [t.grad.double() for t in leaves]


def _rel(a, ref):
    assert torch.isfinite(a).all()
    return ((a - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()


@pytest.mark.parametrize("fn", [_rms_llama, _rms_gemma, _layer_norm], ids = ["rms", "rms_1p", "layernorm"])
@pytest.mark.parametrize("hidden,rows", [(896, 7), (4096, 4096), (5120, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids = ["fp32", "bf16"])
def test_trainable_norm_grads_match_with_nested_reduction(fn, hidden, rows, dtype):
    import torch._inductor.config as config
    default = config.triton.nested_reduction
    g = torch.Generator(device = "cuda").manual_seed(hidden + rows)
    rand = lambda *s: torch.randn(*s, device = "cuda", generator = g, dtype = torch.float64)
    w = 0.1 * rand(hidden) + (0.0 if fn is _rms_gemma else 1.0)
    inputs = [rand(rows, hidden), rand(rows, hidden), w, rand(256, hidden) / hidden**0.5]
    go, goh = rand(rows, 256), rand(rows, hidden)
    ref = _grads(fn, inputs, torch.float64, go, goh)
    eager = _grads(fn, inputs, dtype, go, goh)
    try:
        on = _grads(_compile(fn, True), inputs, dtype, go, goh)
        off = _grads(_compile(fn, False), inputs, dtype, go, goh)
    finally:
        config.triton.nested_reduction = default
        torch._dynamo.reset()
    floor = 1e-5 if dtype == torch.float32 else 2e-2
    for name, r, e, a, b in zip(("x", "residual", "weight", "linear"), ref, eager, on, off):
        bound = max(2 * _rel(e, r), floor)
        assert _rel(a, r) <= bound, (name, _rel(a, r), bound)
        assert _rel(b, r) <= bound, (name, _rel(b, r), bound)


def test_nested_reduction_fires_on_norm_block_amax_and_matches():
    import torch._inductor.config as config
    from torch._inductor import metrics

    def norm_block_amax(x, w):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim = True) + EPS) * w
        return y.to(x.dtype), y.view(x.shape[0], -1, 128).abs().amax(-1)

    g = torch.Generator(device = "cuda").manual_seed(0)
    x = torch.randn(512, 2048, device = "cuda", generator = g, dtype = torch.bfloat16)
    w = torch.randn(2048, device = "cuda", generator = g)
    default = config.triton.nested_reduction
    fired, outs = {}, {}
    try:
        for nested in (True, False):
            metrics.reset()
            outs[nested] = _compile(norm_block_amax, nested)(x, w)
            fired[nested] = metrics.codegen_nested_reduction
    finally:
        config.triton.nested_reduction = default
        torch._dynamo.reset()
    assert fired[True] > 0 and fired[False] == 0, fired
    eager = norm_block_amax(x, w)
    for a, b, e in zip(outs[True], outs[False], eager):
        torch.testing.assert_close(a, b, rtol = 0, atol = 0)
        torch.testing.assert_close(a.float(), e.float(), rtol = 2e-2, atol = 2e-2)

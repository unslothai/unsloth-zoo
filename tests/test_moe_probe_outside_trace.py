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

"""Lazy grouped_mm probes must run eagerly, never on FakeTensors inside a Dynamo trace."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import unsloth_zoo.temporary_patches.moe_utils as M

# Dynamo settings `import unsloth` applies.
_UNSLOTH_DYNAMO = dict(suppress_errors=True, capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True)


@pytest.fixture
def fresh_probes(monkeypatch):
    monkeypatch.setattr(M, "_TORCH_GROUPED_MM_SUPPORTED", None)
    monkeypatch.setattr(M, "_TRANSPOSED_VIEW_GROUPED_MM_SAFE", None)
    M.select_moe_backend.cache_clear()
    torch._dynamo.reset()
    yield
    M.select_moe_backend.cache_clear()
    torch._dynamo.reset()


@pytest.mark.parametrize("check, probe, flag", [
    ("_check_torch_grouped_mm_supported", "_probe_torch_grouped_mm_supported", "_TORCH_GROUPED_MM_SUPPORTED"),
    ("_transposed_view_grouped_mm_is_safe", "_probe_transposed_view_grouped_mm_is_safe", "_TRANSPOSED_VIEW_GROUPED_MM_SAFE"),
])
def test_probe_body_runs_eagerly_under_compile(fresh_probes, monkeypatch, check, probe, flag):
    seen = []

    def fake_probe():
        compiling = torch.compiler.is_compiling()
        seen.append(compiling)
        monkeypatch.setattr(M, flag, not compiling)
        return not compiling

    monkeypatch.setattr(M, probe, fake_probe)
    monkeypatch.setattr(M, "_TORCH_GROUPED_MM_AVAILABLE", True)
    monkeypatch.setattr(M, "_grouped_mm_probe_device", lambda: torch.device("cpu"))
    fn = getattr(M, check)

    def f(x):
        return x + 1 if fn() else x - 1

    with torch._dynamo.config.patch(**_UNSLOTH_DYNAMO):
        out = torch.compile(f, backend="eager", dynamic=True)(torch.zeros(3))
    assert seen == [False], seen
    assert getattr(M, flag) is True
    assert torch.equal(out, torch.ones(3))
    torch._dynamo.reset()
    with torch._dynamo.config.patch(**_UNSLOTH_DYNAMO):
        torch.compile(f, backend="eager", dynamic=True)(torch.zeros(3))
    assert seen == [False]


def _gpu_grouped_mm_ok():
    if not torch.cuda.is_available() or not hasattr(torch, "_grouped_mm"):
        return False
    try:
        torch._grouped_mm(torch.ones(1, 8, device="cuda", dtype=torch.bfloat16),
                          torch.ones(1, 8, 8, device="cuda", dtype=torch.bfloat16),
                          offs=torch.tensor([1], device="cuda", dtype=torch.int32))
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_grouped_mm_ok(), reason="needs a GPU with torch._grouped_mm")
def test_first_moe_forward_inside_compile_matches_reference(fresh_probes):
    torch.manual_seed(0)
    E, H, I, T, K = 8, 64, 32, 48, 2

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts, self.hidden_dim, self.intermediate_dim = E, H, I
            self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * I, H) * 0.1)
            self.down_proj = nn.Parameter(torch.randn(E, H, I) * 0.1)
            self.act_fn = nn.SiLU()
        forward = M.forward_moe_backend

    m = Experts().cuda().to(torch.bfloat16)
    hs = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    idx = torch.stack([torch.randperm(E, device="cuda")[:K] for _ in range(T)])
    w = torch.softmax(torch.randn(T, K, device="cuda"), -1)
    ref = torch.zeros(T, H, device="cuda")
    for e in range(E):
        t, p = torch.where(idx == e)
        g, u = F.linear(hs[t].float(), m.gate_up_proj[e].float()).chunk(2, -1)
        ref.index_add_(0, t, F.linear(F.silu(g) * u, m.down_proj[e].float()) * w[t, p, None])

    def block(mod, x, i, ww):
        return mod(x * 1.0, i, ww)

    with torch.no_grad(), torch._dynamo.config.patch(**_UNSLOTH_DYNAMO):
        out = torch.compile(block, dynamic=True)(m, hs, idx, w).float()
    assert M._TORCH_GROUPED_MM_SUPPORTED is True
    assert M.select_moe_backend() == "grouped_mm"
    rel = ((out - ref).norm() / ref.norm()).item()
    assert rel < 2e-2, rel


def test_definitive_negative_does_not_break_a_fullgraph_trace(monkeypatch):
    import torch
    from unsloth_zoo.temporary_patches import moe_utils

    monkeypatch.setattr(moe_utils, "_TORCH_GROUPED_MM_SUPPORTED", None)
    monkeypatch.setattr(moe_utils, "_TRANSPOSED_VIEW_GROUPED_MM_SAFE", None)
    monkeypatch.setattr(moe_utils, "_grouped_mm_probe_device", lambda: None)
    torch._dynamo.reset()

    @torch.compile(fullgraph = True, backend = "eager")
    def f(x):
        if moe_utils._check_torch_grouped_mm_supported() or moe_utils._transposed_view_grouped_mm_is_safe():
            return x + 1
        return x - 1

    assert torch.equal(f(torch.zeros(2)), torch.full((2,), -1.0))
    assert moe_utils._TORCH_GROUPED_MM_SUPPORTED is False
    assert moe_utils._TRANSPOSED_VIEW_GROUPED_MM_SAFE is False

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

"""gpt-oss bnb-4bit experts: eager inference runs only the routed experts.

torch_native_forward's eval branch used to run every expert for every token and
zero the unrouted ones through routing_weights: num_experts / top_k times the
work, and 2 * num_experts expert launches per layer (64 for gpt-oss-20b), which
left eager decode launch bound. Eager inference now runs only the experts the
router picked; a compiled forward keeps the dense, sync-free branch.

Checked on CPU with plain nn.Linear experts:
  * only the routed experts are called in eager inference, all of them while compiling,
    during a CUDA graph capture, or with UNSLOTH_GPTOSS_ROUTED_INFERENCE=0,
  * the routed result matches an fp64 reference at least as well as the dense one,
    in bfloat16 and float16, for decode (q_len 1) and prefill shapes,
  * output shape and dtype are unchanged.
"""
import os
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

try:
    from unsloth_zoo.temporary_patches.gpt_oss import torch_native_forward, swiglu_torch_forward
except Exception as e:  # pragma: no cover - unsloth_zoo import needs an accelerator
    pytest.skip(f"cannot import unsloth_zoo gpt_oss patches: {e}", allow_module_level=True)

NUM_EXPERTS, TOP_K, HIDDEN, INTER = 8, 2, 32, 16


class _Counted(torch.nn.Linear):
    """Counts calls; like bitsandbytes Linear4bit, computes in the weight dtype and
    returns the input dtype."""
    calls = 0

    def forward(self, x):
        type(self).calls += 1
        return super().forward(x.to(self.weight.dtype)).to(x.dtype)


def _experts(dtype):
    torch.manual_seed(0)
    _Counted.calls = 0
    gate_up = [_Counted(HIDDEN, 2 * INTER).to(dtype) for _ in range(NUM_EXPERTS)]
    down = [_Counted(INTER, HIDDEN).to(dtype) for _ in range(NUM_EXPERTS)]
    return SimpleNamespace(hidden_size = HIDDEN, gate_up_projs = gate_up, down_projs = down,
                           alpha = 1.702, limit = 7.0, training = False)


def _routing(num_tokens, dtype, unused = (3, 6)):
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(num_tokens, NUM_EXPERTS, generator = g)
    logits[:, list(unused)] = -1e9  # some experts never routed
    top_vals, top_idx = logits.topk(TOP_K, dim = -1)
    weights = torch.zeros(num_tokens, NUM_EXPERTS).scatter_(1, top_idx, top_vals.softmax(-1))
    return top_idx, weights.to(dtype)


def _reference(m, x, idx, w):
    out = torch.zeros(x.shape[0], HIDDEN, dtype = torch.float64)
    for t in range(x.shape[0]):
        for e in idx[t].tolist():
            gu = torch.nn.functional.linear(x[t].double(), m.gate_up_projs[e].weight.double(), m.gate_up_projs[e].bias.double())
            h = swiglu_torch_forward(gu, m.alpha, m.limit, dtype = torch.float64)
            y = torch.nn.functional.linear(h, m.down_projs[e].weight.double(), m.down_projs[e].bias.double())
            out[t] += w[t, e].double() * y
    return out


def _run(m, x, idx, w, batch, compiling, monkeypatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)
    _Counted.calls = 0
    y = torch_native_forward(m, x.view(batch, -1, HIDDEN), idx, w)
    return y, _Counted.calls


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids = str)
@pytest.mark.parametrize("batch,q_len", [(1, 1), (4, 1), (2, 7)])
def test_eager_inference_runs_only_routed_experts(dtype, batch, q_len, monkeypatch):
    m = _experts(dtype)
    n = batch * q_len
    x = torch.randn(n, HIDDEN).to(dtype)
    idx, w = _routing(n, dtype)
    routed, routed_calls = _run(m, x, idx, w, batch, False, monkeypatch)
    dense, dense_calls = _run(m, x, idx, w, batch, True, monkeypatch)
    used = len(set(idx.flatten().tolist()))
    assert routed_calls == 2 * used < 2 * NUM_EXPERTS
    assert dense_calls == 2 * NUM_EXPERTS
    assert routed.shape == dense.shape == (batch, q_len, HIDDEN)
    assert routed.dtype == dense.dtype == dtype
    ref = _reference(m, x, idx, w).view(batch, q_len, HIDDEN)
    err_routed = (routed.double() - ref).abs().max().item()
    err_dense = (dense.double() - ref).abs().max().item()
    # fp32 accumulation over the routed experts only: never worse than the dense sum.
    assert err_routed <= err_dense + torch.finfo(dtype).eps * ref.abs().max().item()


def test_dense_branch_kept_for_capture_and_kill_switch(monkeypatch):
    m = _experts(torch.bfloat16)
    x = torch.randn(4, HIDDEN).to(torch.bfloat16)
    idx, w = _routing(4, torch.bfloat16)
    monkeypatch.setenv("UNSLOTH_GPTOSS_ROUTED_INFERENCE", "0")
    _, calls = _run(m, x, idx, w, 4, False, monkeypatch)
    assert calls == 2 * NUM_EXPERTS
    monkeypatch.delenv("UNSLOTH_GPTOSS_ROUTED_INFERENCE")
    if torch.cuda.is_available():
        # A real CUDA graph capture: the routed branch's host sync would make it fail.
        m = SimpleNamespace(**{**vars(m), "gate_up_projs": [l.cuda() for l in m.gate_up_projs],
                               "down_projs": [l.cuda() for l in m.down_projs]})
        xc, ic, wc = x.cuda().view(4, -1, HIDDEN), idx.cuda(), w.cuda()
        monkeypatch.setenv("UNSLOTH_GPTOSS_ROUTED_INFERENCE", "0")
        torch_native_forward(m, xc, ic, wc)  # warm up the dense branch outside the capture
        monkeypatch.delenv("UNSLOTH_GPTOSS_ROUTED_INFERENCE")
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        _Counted.calls = 0
        with torch.cuda.graph(graph):
            captured = torch_native_forward(m, xc, ic, wc)
        assert _Counted.calls == 2 * NUM_EXPERTS
        graph.replay(); torch.cuda.synchronize()
        monkeypatch.setenv("UNSLOTH_GPTOSS_ROUTED_INFERENCE", "0")
        assert torch.equal(captured, torch_native_forward(m, xc, ic, wc))

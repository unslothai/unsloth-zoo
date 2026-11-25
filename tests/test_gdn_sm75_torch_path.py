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

"""sm75 (T4) gated delta: Triton >= 3.3 has no tensor-core dot on sm75, so fla's chunk_gated_delta_rule runs
unsloth_zoo.gdn_torch_chunk there. CPU tests: the torch path against an fp64 token-by-token recurrence (forward,
final state and every gradient), the scan op's autograd, its traceability, and which calls the routing takes."""

import os
import sys
import types

import pytest

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_VENDORED_FLA_NO_AUTORUN", "1")

torch = pytest.importorskip("torch")
if not hasattr(getattr(torch, "library", None), "custom_op"):
    pytest.skip("torch.library.custom_op needs torch >= 2.4", allow_module_level = True)
try:
    from unsloth_zoo import gdn_torch_chunk as G
    from unsloth_zoo.temporary_patches import fla_vendor
except ImportError as _e:
    pytest.skip(f"unsloth_zoo unavailable: {_e}", allow_module_level = True)


def _recurrent_fp64(q, k, v, g, beta, scale, h0 = None):
    """fla's gated delta rule token by token in float64 (fused_recurrent semantics, in-kernel l2norm)."""
    q, k, v, g, beta = (x.double() for x in (q, k, v, g, beta))
    groups = v.shape[2] // q.shape[2]
    q, k = q.repeat_interleave(groups, 2), k.repeat_interleave(groups, 2)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim = True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim = True) + 1e-6)
    B, T, H, K = q.shape
    S = torch.zeros(B, H, K, v.shape[-1], dtype = torch.float64) if h0 is None else h0.double()
    out = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        u = beta[:, t][..., None] * (v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], S))
        S = S + k[:, t][..., :, None] * u[..., None, :]
        out.append(scale * torch.einsum("bhk,bhkv->bhv", q[:, t], S))
    return torch.stack(out, 1), S


def _inputs(B, T, H, HV, K, V, seed = 0, dtype = torch.float64):
    gen = torch.Generator().manual_seed(seed)
    q, k = torch.randn(B, T, H, K, generator = gen), torch.randn(B, T, H, K, generator = gen)
    v = torch.randn(B, T, HV, V, generator = gen)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, HV, generator = gen)) * 0.5
    beta = torch.rand(B, T, HV, generator = gen)
    h0 = torch.randn(B, HV, K, V, generator = gen) * 0.1
    return [x.to(dtype) for x in (q, k, v, g, beta, h0)]


@pytest.mark.parametrize("T", [1, 63, 64, 130])
@pytest.mark.parametrize("H,HV", [(2, 2), (2, 4)])
def test_torch_chunk_matches_fp64_recurrence_forward_and_backward(T, H, HV):
    """float64 runs the same code in float64: forward, final state and the grads of q, k, v, g, beta, h0
    agree with the token recurrence to rounding. Covers padding (T % 64), one token, GVA (H < HV)."""
    ins = [x.requires_grad_() for x in _inputs(2, T, H, HV, 16, 8)]
    ref_ins = [x.detach().clone().requires_grad_() for x in ins]
    o, ht = G.torch_chunk_gated_delta_rule(*ins[:5], scale = 0.3, initial_state = ins[5], output_final_state = True,
                                           use_qk_l2norm_in_kernel = True)
    o_ref, ht_ref = _recurrent_fp64(*ref_ins[:5], 0.3, ref_ins[5])
    torch.testing.assert_close(o, o_ref, atol = 1e-11, rtol = 1e-10)
    torch.testing.assert_close(ht, ht_ref, atol = 1e-11, rtol = 1e-10)
    gen = torch.Generator().manual_seed(9)
    do, dht = torch.randn(o.shape, generator = gen, dtype = o.dtype), torch.randn(ht.shape, generator = gen, dtype = ht.dtype)
    grads = torch.autograd.grad((o * do).sum() + (ht * dht).sum(), ins)
    ref = torch.autograd.grad((o_ref * do).sum() + (ht_ref * dht).sum(), ref_ins)
    for name, a, b in zip(("dq", "dk", "dv", "dg", "dbeta", "dh0"), grads, ref):
        torch.testing.assert_close(a, b, atol = 1e-10, rtol = 1e-9, msg = name)


@pytest.mark.parametrize("lengths", [[70, 50], [1, 63, 64, 65, 1, 7], [128, 64], [200]])
def test_torch_chunk_packed_documents_match_fp64_per_document(lengths):
    """cu_seqlens (B = 1): each document equals its own recurrence from a zero state, forward and every grad.
    Documents shorter than a chunk, ending on a chunk edge, one token long, and one spanning several chunks."""
    T = sum(lengths)
    ins = [x.requires_grad_() for x in _inputs(1, T, 2, 4, 16, 8, seed = len(lengths))[:5]]
    ref_ins = [x.detach().clone().requires_grad_() for x in ins]
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), dtype = torch.int32)
    o, ht = G.torch_chunk_gated_delta_rule(*ins, scale = 0.3, use_qk_l2norm_in_kernel = True, cu_seqlens = cu)
    assert ht is None
    outs = []
    for a, b in zip(cu[:-1].tolist(), cu[1:].tolist()):
        outs.append(_recurrent_fp64(*(x[:, a:b] for x in ref_ins), 0.3)[0])
    o_ref = torch.cat(outs, 1)
    torch.testing.assert_close(o, o_ref, atol = 1e-11, rtol = 1e-10)
    do = torch.randn(o.shape, generator = torch.Generator().manual_seed(4), dtype = o.dtype)
    grads = torch.autograd.grad((o * do).sum(), ins)
    ref = torch.autograd.grad((o_ref * do).sum(), ref_ins)
    for name, a, b in zip(("dq", "dk", "dv", "dg", "dbeta"), grads, ref):
        torch.testing.assert_close(a, b, atol = 1e-10, rtol = 1e-9, msg = name)


def _half_inputs(*shape, seed = 0):
    """float16 q / k / v / beta (what sm75 trains in), float32 g and state, like transformers' GatedDeltaNet."""
    q, k, v, g, beta, h0 = _inputs(*shape, seed = seed)
    return q.half(), k.half(), v.half(), g.float(), beta.half(), h0.float()


def test_torch_chunk_float32_and_no_final_state():
    q, k, v, g, beta, _ = _inputs(1, 100, 2, 2, 16, 8, dtype = torch.float32)
    o, ht = G.torch_chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel = True)
    o_ref, _ = _recurrent_fp64(q, k, v, g, beta, 16 ** -0.5)
    assert ht is None and o.dtype == torch.float32
    torch.testing.assert_close(o.double(), o_ref, atol = 2e-5, rtol = 1e-4)


def test_scan_op_gradcheck_and_opcheck():
    scan, _ = G.scan_ops()
    gen = torch.Generator().manual_seed(1)
    B, H, N, C, K, V = 1, 2, 3, 4, 3, 5

    def r(*s):
        return torch.randn(*s, generator = gen, dtype = torch.float64)

    args = (r(B, H, N, C, K), r(B, H, N, C, V), r(B, H, N, K, C), torch.rand(B, H, N, generator = gen, dtype = torch.float64), r(B, H, K, V))
    assert torch.autograd.gradcheck(scan, tuple(a.clone().requires_grad_() for a in args))
    torch.library.opcheck(scan, args)


def test_compiles_fullgraph_one_graph_for_every_length():
    """The recurrence is opaque: no graph break, and dynamic lengths share one graph (no unrolled chunk loop)."""
    from torch._dynamo.utils import counters
    torch._dynamo.reset()
    counters.clear()
    f = torch.compile(G.torch_chunk_gated_delta_rule, backend = "aot_eager", fullgraph = True, dynamic = True)
    for T in (70, 130):
        ins = [x.float().requires_grad_() for x in _inputs(1, T, 2, 2, 16, 8)[:5]]
        o, _ = f(*ins, use_qk_l2norm_in_kernel = True)
        o_eager, _ = G.torch_chunk_gated_delta_rule(*ins, use_qk_l2norm_in_kernel = True)
        torch.testing.assert_close(o, o_eager, atol = 1e-6, rtol = 1e-5)
        o.sum().backward()
    assert sum(counters["graph_break"].values()) == 0
    assert counters["stats"]["unique_graphs"] <= 2


def test_packed_rows_compile_fullgraph_across_document_counts():
    """cu_seqlens: document ids from a bucketize, masks from them: no host sync, so no graph break, and new
    lengths / document counts reuse the graph."""
    from torch._dynamo.utils import counters
    torch._dynamo.reset()
    counters.clear()
    f = torch.compile(G.torch_chunk_gated_delta_rule, backend = "aot_eager", fullgraph = True, dynamic = True)
    for lengths in ([70, 50], [1, 64, 65, 7]):
        T = sum(lengths)
        cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), dtype = torch.int32)
        ins = [x.float().requires_grad_() for x in _inputs(1, T, 2, 2, 16, 8)[:5]]
        o, _ = f(*ins, use_qk_l2norm_in_kernel = True, cu_seqlens = cu)
        o_eager, _ = G.torch_chunk_gated_delta_rule(*ins, use_qk_l2norm_in_kernel = True, cu_seqlens = cu)
        torch.testing.assert_close(o, o_eager, atol = 1e-6, rtol = 1e-5)
        o.sum().backward()
    assert sum(counters["graph_break"].values()) == 0
    assert counters["stats"]["unique_graphs"] <= 2


# Routing: a fake fla with fla's public signature, whose kernel records each call.
def _fake_fla(monkeypatch):
    calls = []

    def chunk_gated_delta_rule(q, k, v, g, beta, scale = None, initial_state = None, output_final_state = False,
                               use_qk_l2norm_in_kernel = False, use_beta_sigmoid_in_kernel = False,
                               allow_neg_eigval = False, state_v_first = False, cu_seqlens = None,
                               cu_seqlens_cpu = None, cp_context = None, **kwargs):
        calls.append(kwargs)
        return "fla", None

    chunk_gated_delta_rule = torch.compiler.disable(chunk_gated_delta_rule)
    package = types.ModuleType("fla.ops.gated_delta_rule")
    chunk = types.ModuleType("fla.ops.gated_delta_rule.chunk")
    package.chunk_gated_delta_rule = chunk.chunk_gated_delta_rule = chunk_gated_delta_rule
    modeling = types.ModuleType("transformers.models.qwen3_5.modeling_qwen3_5")
    modeling.chunk_gated_delta_rule = chunk_gated_delta_rule
    for m in (package, chunk):
        monkeypatch.setitem(sys.modules, m.__name__, m)
    monkeypatch.setitem(sys.modules, modeling.__name__, modeling)
    return package, chunk, modeling, calls


def _sm75_host(monkeypatch, caps = {0: (7, 5)}, triton_ok = True, budget = 1 << 62):
    """Pretend the visible GPUs have ``caps``, and that CPU test tensors live on device 0 of them."""
    monkeypatch.setitem(fla_vendor._sm75_state, "capability", dict(caps))
    monkeypatch.setitem(fla_vendor._sm75_state, "calls", 0)
    monkeypatch.setitem(fla_vendor._sm75_state, "over_budget", 0)
    monkeypatch.setitem(fla_vendor._sm75_state, "budget", {0: budget})
    monkeypatch.setattr(fla_vendor, "_triton_lacks_sm75_dots", lambda: triton_ok)
    monkeypatch.setattr(fla_vendor, "_on_sm75", lambda t: isinstance(t, torch.Tensor) and caps.get(0) == (7, 5))


def test_sm75_routes_every_export_and_matches_fp64(monkeypatch):
    package, chunk, modeling, calls = _fake_fla(monkeypatch)
    _sm75_host(monkeypatch)
    monkeypatch.delenv("UNSLOTH_GDN_SM75_TORCH", raising = False)
    assert fla_vendor._install_sm75_torch_path()
    assert fla_vendor._install_sm75_torch_path()  # idempotent
    q, k, v, g, beta, h0 = _half_inputs(2, 70, 2, 4, 16, 8)
    o_ref, ht_ref = _recurrent_fp64(q, k, v, g, beta, 0.3, h0)
    for fn in (package.chunk_gated_delta_rule, chunk.chunk_gated_delta_rule, modeling.chunk_gated_delta_rule):
        o, ht = fn(q, k, v, g = g, beta = beta, scale = 0.3, initial_state = h0, output_final_state = True,
                   use_qk_l2norm_in_kernel = True)
        assert o.dtype == torch.float16
        torch.testing.assert_close(o.double(), o_ref, atol = 2e-3, rtol = 2e-3)
        torch.testing.assert_close(ht.double(), ht_ref, atol = 2e-3, rtol = 2e-3)
    assert calls == [] and fla_vendor._sm75_state["calls"] == 3
    # state_v_first: the state is [V, K] in and out.
    o2, ht2 = package.chunk_gated_delta_rule(q, k, v, g = g, beta = beta, scale = 0.3,
                                             initial_state = h0.transpose(-1, -2).contiguous(), output_final_state = True,
                                             use_qk_l2norm_in_kernel = True, state_v_first = True)
    torch.testing.assert_close(ht2.transpose(-1, -2).double(), ht_ref, atol = 2e-3, rtol = 2e-3)
    # beta from logits in kernel.
    o3, _ = package.chunk_gated_delta_rule(q, k, v, g = g, beta = torch.logit(beta.float()).half(), scale = 0.3, initial_state = h0,
                                           use_qk_l2norm_in_kernel = True, use_beta_sigmoid_in_kernel = True)
    torch.testing.assert_close(o3.double(), o_ref, atol = 2e-3, rtol = 2e-3)


def test_sm75_routes_packed_rows(monkeypatch):
    package, _, _, calls = _fake_fla(monkeypatch)
    _sm75_host(monkeypatch)
    assert fla_vendor._install_sm75_torch_path()
    q, k, v, g, beta, _ = _half_inputs(1, 70, 2, 4, 16, 8)
    cu = torch.tensor([0, 30, 70])
    o, ht = package.chunk_gated_delta_rule(q, k, v, g = g, beta = beta, scale = 0.3, use_qk_l2norm_in_kernel = True,
                                           cu_seqlens = cu, cu_seqlens_cpu = cu)
    ref = torch.cat([_recurrent_fp64(*(x[:, a:b] for x in (q, k, v, g, beta)), 0.3)[0] for a, b in ((0, 30), (30, 70))], 1)
    torch.testing.assert_close(o.double(), ref, atol = 2e-3, rtol = 2e-3)
    assert ht is None and calls == [] and fla_vendor._sm75_state["calls"] == 1


def test_repeated_patch_passes_wrap_once(monkeypatch):
    """patch_vendor_fla runs more than once: the CPU and sm75 installers each wrap the kernel once in total, so the
    kernel-hub closure and Unsloth's caller walk see one sm75 router over one CPU wrapper over fla."""
    package, chunk, modeling, calls = _fake_fla(monkeypatch)
    _sm75_host(monkeypatch)
    kernel = package.chunk_gated_delta_rule
    for _ in range(3):
        fla_vendor._install_cpu_torch_path()
        assert fla_vendor._install_sm75_torch_path()
    top = package.chunk_gated_delta_rule
    assert getattr(top, fla_vendor._SM75_TORCH_MARK, None) == "torch"
    cpu = top.__wrapped__
    assert getattr(cpu, fla_vendor._SM75_TORCH_MARK, None) is None and getattr(cpu, fla_vendor._CPU_TORCH_MARK, None) is not None
    assert cpu.__wrapped__ is kernel
    assert chunk.chunk_gated_delta_rule is top and modeling.chunk_gated_delta_rule is top


def test_sm75_calls_over_the_memory_budget_keep_fla(monkeypatch):
    """The torch path's workspace estimate (tokens x value heads x bytes) over the budget: fla, counted; under it:
    the torch path. The edge is exact, so the decision is a pure function of the shape."""
    package, _, _, calls = _fake_fla(monkeypatch)
    per = fla_vendor._SM75_BYTES_PER_TOKEN_HEAD["float16"]
    B, T, HV = 2, 70, 4
    _sm75_host(monkeypatch, budget = B * T * HV * per)
    assert fla_vendor._install_sm75_torch_path()
    q, k, v, g, beta, _ = _half_inputs(B, T, 2, HV, 16, 8)
    o, _ = package.chunk_gated_delta_rule(q, k, v, g = g, beta = beta, use_qk_l2norm_in_kernel = True)
    assert o.dtype == torch.float16 and calls == [] and fla_vendor._sm75_state["calls"] == 1
    q, k, v, g, beta, _ = _half_inputs(B, T + 1, 2, HV, 16, 8)
    o, _ = package.chunk_gated_delta_rule(q, k, v, g = g, beta = beta, use_qk_l2norm_in_kernel = True)
    assert o == "fla" and len(calls) == 1
    assert fla_vendor._sm75_state["calls"] == 1 and fla_vendor._sm75_state["over_budget"] == 1
    # Non-float16 inputs (float32 / bfloat16 matmuls) count more bytes per token.
    q, k, v, g, beta, _ = _half_inputs(B, T, 2, HV, 16, 8)
    o, _ = package.chunk_gated_delta_rule(q.float(), k.float(), v.float(), g = g, beta = beta.float(), use_qk_l2norm_in_kernel = True)
    assert o == "fla" and fla_vendor._sm75_state["over_budget"] == 2


def test_sm75_budget_is_read_once_per_device(monkeypatch):
    """A fraction of driver-free + allocator-unused memory at the first call, cached: a checkpoint recompute (less
    free memory) decides like its forward. UNSLOTH_GDN_SM75_MEM_FRACTION overrides the fraction."""
    reads = []

    def mem_get_info(index):
        reads.append(index)
        return (8 << 30) - len(reads), 16 << 30

    monkeypatch.setitem(fla_vendor._sm75_state, "budget", {})
    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda index: 3 << 30)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda index: 1 << 30)
    monkeypatch.setenv("UNSLOTH_GDN_SM75_MEM_FRACTION", "0.25")
    first = fla_vendor._sm75_budget_impl(0)
    assert first == int(((8 << 30) - 1 + (2 << 30)) * 0.25)
    assert fla_vendor._sm75_budget_impl(0) == first and reads == [0]
    fla_vendor._sm75_budget_impl(1)
    assert reads == [0, 1]
    # No memory reading: no budget, fla.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: 1 / 0)
    assert fla_vendor._sm75_budget_impl(2) == 0


def test_sm75_budget_compiles_without_graph_break(monkeypatch):
    """Traced calls read the budget as a constant (no graph break); the shape check is a guard, so lengths under the
    budget share the torch graph and a call over it compiles once more, to fla."""
    from torch._dynamo.utils import counters
    package, _, _, calls = _fake_fla(monkeypatch)
    per = fla_vendor._SM75_BYTES_PER_TOKEN_HEAD["other"]
    _sm75_host(monkeypatch, budget = 1 * 200 * 2 * per)
    assert fla_vendor._install_sm75_torch_path()
    torch._dynamo.reset()
    counters.clear()
    fn = package.chunk_gated_delta_rule
    f = torch.compile(lambda *a: fn(*a, use_qk_l2norm_in_kernel = True)[0], backend = "aot_eager", fullgraph = True, dynamic = True)
    for T in (70, 130, 200):
        ins = [x.float().requires_grad_() for x in _inputs(1, T, 2, 2, 16, 8)[:5]]
        o = f(*ins)
        o.sum().backward()
    assert sum(counters["graph_break"].values()) == 0 and calls == []
    assert counters["stats"]["unique_graphs"] <= 2
    g = torch.compile(lambda *a: fn(*a, use_qk_l2norm_in_kernel = True)[0], backend = "aot_eager", dynamic = True)
    ins = [x.float() for x in _inputs(1, 201, 2, 2, 16, 8)[:5]]
    assert g(*ins) == "fla" and len(calls) == 1


@pytest.mark.parametrize("extra", [
    dict(cu_seqlens = torch.tensor([0, 30, 70]), output_final_state = True),
    dict(cu_seqlens = torch.tensor([0, 30, 70]), initial_state = torch.zeros(2, 4, 16, 8)),
    dict(cu_seqlens = torch.tensor([0, 30, 70]), batch = 2),
    dict(cp_context = object()),
    dict(use_gate_in_kernel = True, A_log = torch.zeros(4)),
    dict(head_first = False),
    dict(some_future_flag = True),
    dict(allow_neg_eigval = True),
])
def test_sm75_calls_the_torch_path_does_not_implement_keep_fla(monkeypatch, extra):
    package, _, _, calls = _fake_fla(monkeypatch)
    _sm75_host(monkeypatch)
    assert fla_vendor._install_sm75_torch_path()
    B = extra.pop("batch", 1 if "cu_seqlens" in extra else 2)
    q, k, v, g, beta, _ = _half_inputs(B, 70, 2, 4, 16, 8)
    assert package.chunk_gated_delta_rule(q, k, v, g = g, beta = beta, **extra)[0] == "fla"
    assert len(calls) == 1 and fla_vendor._sm75_state["calls"] == 0


@pytest.mark.parametrize("case", ["kill_switch", "ampere", "no_gpu", "triton_3_2"])
def test_not_installed_off_sm75_or_when_disabled(monkeypatch, case):
    package, _, _, _ = _fake_fla(monkeypatch)
    original = package.chunk_gated_delta_rule
    _sm75_host(monkeypatch, caps = {0: (8, 0), 1: (8, 9)} if case == "ampere" else ({} if case == "no_gpu" else {0: (7, 5)}),
               triton_ok = case != "triton_3_2")
    if case == "kill_switch":
        monkeypatch.setenv("UNSLOTH_GDN_SM75_TORCH", "0")
    assert not fla_vendor._install_sm75_torch_path()
    assert package.chunk_gated_delta_rule is original


@pytest.mark.parametrize("name,routed", [
    ("Tesla T4", True), ("NVIDIA GeForce RTX 2080 Ti", True), ("Quadro RTX 6000", True),
    ("NVIDIA GeForce GTX 1660 SUPER", False), ("Quadro T1000", False), ("NVIDIA T400", False),
])
def test_sm75_needs_tensor_cores(monkeypatch, name, routed):
    """sm75 parts without tensor cores (GTX 16xx, Quadro T-series) report sm75 too: they keep fla."""
    monkeypatch.setitem(fla_vendor._sm75_state, "capability", None)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch, "__version__", str(torch.__version__).split("+")[0] + ("+rocm7.13.0" if None else "+cu130"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i = 0: (7, 5))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i = 0: name)
    assert (fla_vendor._sm75_capabilities().get(0) == (7, 5)) is routed


def test_rocm_reports_no_sm75(monkeypatch):
    """ROCm builds report CUDA-style capabilities; the probe returns none there, so the path never installs."""
    monkeypatch.setitem(fla_vendor._sm75_state, "capability", None)
    monkeypatch.setattr(torch.version, "hip", "6.4.0", raising = False)
    monkeypatch.setattr(torch, "__version__", str(torch.__version__).split("+")[0] + ("+rocm7.13.0" if "6.4.0" else "+cu130"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i = 0: (7, 5))
    assert fla_vendor._sm75_capabilities() == {}


def test_triton_gate():
    import triton
    expected = fla_vendor._version_at_least(triton.__version__, "3.3.0") if hasattr(triton, "__version__") else False
    assert fla_vendor._triton_lacks_sm75_dots() == expected

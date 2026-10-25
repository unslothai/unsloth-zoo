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
"""sonic-moe zero-copy gate_up (concat_layout=("B",)) layout algebra.

The CPU tests pin the layout facts the zero-copy path depends on, emulating quack's
`concat_to_interleave(t, 1 - t.leading_dim)` with `as_strided`. The GPU test runs only
where the kernels-community/sonic-moe build loads (sm90 / sm100 with `kernels`).
"""
import pytest
import torch
import torch.nn.functional as F

from unsloth_zoo.temporary_patches import moe_utils_sonic as S


def _quack_concat_view_b(weight_t):
    """What the kernel reads for B = `weight_t` ([E, K, N]) with "B" in concat_layout:
    the non-contiguous of (N, K) is split as (2, half) with strides (half * s, s)."""
    E, Kdim, N = weight_t.shape
    sE, sK, sN = weight_t.stride()
    if sK == 1:      # K-major: N is the non-contiguous dim -> N interleaved
        half = N // 2
        return torch.as_strided(weight_t, (E, Kdim, half, 2), (sE, sK, sN, half * sN)).reshape(E, Kdim, N)
    half = Kdim // 2  # N-major: K is the non-contiguous dim -> K interleaved
    return torch.as_strided(weight_t, (E, half, 2, N), (sE, sK, half * sK, sN)).reshape(E, Kdim, N)


def test_concat_view_of_k_major_is_the_interleaved_weight():
    E, H, inter = 3, 16, 8
    W = torch.randn(E, 2 * inter, H)            # HF storage [E, 2I, H]
    Wt = W.transpose(-2, -1)                # grouped_mm view [E, H, 2I], K-major
    assert S._is_k_major_gate_up(Wt)
    assert torch.equal(_quack_concat_view_b(Wt), S._interleave_columns(Wt))


def test_concat_view_of_n_major_interleaves_the_contraction_dim():
    # Contiguous [E, H, 2I] interleaves K instead: rejected.
    E, H, inter = 2, 16, 8
    Wn = torch.randn(E, H, 2 * inter)
    assert not S._is_k_major_gate_up(Wn)
    assert not torch.equal(_quack_concat_view_b(Wn), S._interleave_columns(Wn))


def test_dx_concat_on_hf_storage_matches_chunked_dx():
    E, H, inter, M = 1, 16, 8, 5
    W = torch.randn(E, 2 * inter, H, dtype=torch.float64)
    dpre = torch.randn(M, 2 * inter, dtype=torch.float64)
    dx_ref = dpre @ W[0]
    dx = S._interleave_columns(dpre) @ _quack_concat_view_b(W)[0]
    torch.testing.assert_close(dx, dx_ref)


def test_sqrt2_signature_of_column_order_mismatch():
    # An interleaved preact compared against a chunked reference: rel err ~ sqrt(2).
    torch.manual_seed(0)
    ref = torch.randn(4096, 512, dtype=torch.float64)
    rel = ((S._interleave_columns(ref) - ref).norm() / ref.norm()).item()
    assert abs(rel - 2 ** 0.5) < 0.05


def test_interleaved_lora_up_projection_gives_interleaved_delta():
    H, inter, R, M = 16, 8, 4, 6
    x = torch.randn(M, H, dtype=torch.float64)
    A = torch.randn(H, R, dtype=torch.float64)
    B = torch.randn(R, 2 * inter, dtype=torch.float64, requires_grad=True)
    delta_il = (x @ A) @ S._interleave_columns(B)
    torch.testing.assert_close(delta_il, S._interleave_columns((x @ A) @ B))
    g = torch.randn_like(delta_il)
    (delta_il * g).sum().backward()
    grad_il = B.grad.clone()
    B.grad = None
    g_chunked = torch.cat([g[:, 0::2], g[:, 1::2]], dim=-1)
    ((x @ A) @ B * g_chunked).sum().backward()
    torch.testing.assert_close(grad_il, B.grad)


def test_swiglu_backward_interleaved_matches_autograd():
    M, inter = 7, 12
    pre = torch.randn(M, 2 * inter, dtype=torch.float64, requires_grad=True)
    out = F.silu(pre[:, :inter]) * pre[:, inter:]
    g = torch.randn(M, inter, dtype=torch.float64)
    (out * g).sum().backward()
    d_il = S._swiglu_backward_interleaved(S._interleave_columns(pre.detach()), g)
    torch.testing.assert_close(d_il, S._interleave_columns(pre.grad))


def test_k_major_check_rejects_misaligned_strides():
    W = torch.randn(2, 6, 7, dtype=torch.bfloat16)   # H=7: 14-byte stride, not TMA aligned
    assert not S._is_k_major_gate_up(W.transpose(-2, -1))
    assert not S._is_k_major_gate_up(torch.randn(2, 7, 16).transpose(-2, -1))  # odd 2I
    assert not S._is_k_major_gate_up(torch.randn(2, 24, 16).transpose(-2, -1))  # inter=12, rows not 16B


def _sonic_ready():
    if not torch.cuda.is_available():
        return False
    try:
        return S.sonic_moe_available() and S._concat_layout_supported(S.get_sonic_module())
    except Exception:
        return False


@pytest.mark.skipif(not _sonic_ready(), reason="kernels-community/sonic-moe with concat_layout not available")
@pytest.mark.parametrize("lora", [False, True])
def test_zero_copy_matches_grouped_mm(lora, monkeypatch):
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts
    from unsloth_zoo.temporary_patches import moe_utils as MU

    torch.manual_seed(0)
    dev = torch.device("cuda")
    H, inter, E, K, T, R = 256, 128, 8, 2, 512, 8
    cfg = Qwen3MoeConfig(hidden_size=H, moe_intermediate_size=inter, num_experts=E,
                         num_experts_per_tok=K, hidden_act="silu")
    experts = Qwen3MoeExperts(cfg).to(dev)
    with torch.no_grad():
        experts.gate_up_proj.normal_(0, H ** -0.5)
        experts.down_proj.normal_(0, inter ** -0.5)
    experts = experts.to(torch.bfloat16).requires_grad_(False)
    X0 = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    topv, topi = torch.randn(T, E, device=dev).topk(K, dim=-1)
    W0 = torch.softmax(topv, -1).to(torch.bfloat16)
    lora_w = [torch.randn(E, H, R, device=dev) * H ** -0.5, torch.randn(E, R, 2 * inter, device=dev) * 0.05]

    def run(fn):
        X = X0.clone().requires_grad_(True)
        Wt = W0.clone().requires_grad_(True)
        lw = [t.clone().requires_grad_(True) for t in lora_w]
        if lora:
            experts._unsloth_lora_gate_up_proj = (lw[0], lw[1], 2.0)
        else:
            experts.__dict__.pop("_unsloth_lora_gate_up_proj", None)
        out = fn(experts, X, topi, Wt)
        out.float().square().sum().backward()
        grads = [X.grad, Wt.grad] + ([t.grad for t in lw] if lora else [])
        return [out] + grads

    assert S._sonic_eligible(experts, X0) == (True, None), S._sonic_eligible(experts, X0)
    ref = run(MU.forward_native_grouped_mm)
    monkeypatch.setenv("UNSLOTH_MOE_SONIC_GATED", "1")
    monkeypatch.setenv("UNSLOTH_MOE_SONIC_ZERO_COPY", "1")
    S._INTERLEAVE_CACHE.clear()
    got = run(S.forward_sonic_moe)
    assert S._cached_bytes() == 0, "zero-copy path must not build the interleaved duplicate"
    got2 = run(S.forward_sonic_moe)
    for a, b, c in zip(got, ref, got2):
        rel = ((a.float() - b.float()).norm() / b.float().norm()).item()
        assert rel < 2e-2, rel
        assert torch.equal(a, c), "zero-copy fused path is not bitwise reproducible"


@pytest.mark.skipif(not torch.cuda.is_available() or S.triton is None, reason="needs CUDA + Triton")
def test_triton_swiglu_backward_matches_torch(monkeypatch):
    torch.manual_seed(0)
    pre = torch.randn(3001, 2 * 96, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(3001, 96, device="cuda", dtype=torch.bfloat16)
    got = S._swiglu_backward_interleaved(pre, g)
    monkeypatch.setenv("UNSLOTH_MOE_SONIC_TRITON_DGATED", "0")
    ref = S._swiglu_backward_interleaved(pre, g)
    torch.testing.assert_close(got.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_interleave_cache_is_tied_to_the_source_weight(monkeypatch):
    # data_ptr is reused after a model is freed: a dead or different owner must never hit.
    import gc

    monkeypatch.setenv("UNSLOTH_MOE_SONIC_CACHE_GIB", "1")
    S._INTERLEAVE_CACHE.clear()
    param = torch.randn(2, 8, 6)
    view = param.transpose(-2, -1).transpose(-2, -1)
    first = S._interleaved_gate_up(view)
    assert S._interleaved_gate_up(param.transpose(-2, -1).transpose(-2, -1)) is first
    key = next(iter(S._INTERLEAVE_CACHE))
    other = torch.randn(2, 8, 6)
    entry = S._INTERLEAVE_CACHE[key]
    S._INTERLEAVE_CACHE[key] = entry[:4] + (__import__("weakref").ref(other),)
    assert S._interleaved_gate_up(param.transpose(-2, -1).transpose(-2, -1)) is not first
    del param, view, first
    gc.collect()
    assert not S._INTERLEAVE_CACHE


def test_sonic_availability_probe_runs_outside_dynamo(monkeypatch):
    from unsloth_zoo.temporary_patches import moe_utils as MU

    seen = []
    monkeypatch.setattr(S, "sonic_moe_available", lambda: seen.append(torch.compiler.is_compiling()) or False)
    monkeypatch.delenv("UNSLOTH_DISABLE_MOE_SONIC", raising = False)

    @torch.compile(backend = "eager")
    def f(x):
        return x + int(MU._check_sonic_moe_available())

    f(torch.zeros(1))
    assert seen == [False]


def _cpu_experts(act_fn):
    """A CPU Qwen3-style experts module that passes every _sonic_eligible check but the one under test."""
    mod = torch.nn.Module()
    mod.gate_up_proj = torch.nn.Parameter(torch.randn(4, 32, 16, dtype=torch.bfloat16), requires_grad=False)
    mod.down_proj = torch.nn.Parameter(torch.randn(4, 16, 16, dtype=torch.bfloat16), requires_grad=False)
    mod.num_experts = 4
    if act_fn is not None:
        mod.act_fn = act_fn
    return mod


def test_eligibility_rejects_non_silu_activations(monkeypatch):
    monkeypatch.setattr(S, "sonic_moe_available", lambda: True)
    monkeypatch.setattr(S, "_device_ok", lambda device: True, raising=False)
    x = torch.zeros(3, 16, dtype=torch.bfloat16)
    for act in (torch.nn.Hardswish(), torch.nn.GELU(), torch.tanh):
        ok, reason = S._sonic_eligible(_cpu_experts(act), x)
        assert not ok and "not SiLU" in reason, (act, reason)
    for act in (torch.nn.SiLU(), F.silu, None):
        assert S._sonic_eligible(_cpu_experts(act), x) == (True, None), act


def test_eligibility_checks_the_operand_device(monkeypatch):
    monkeypatch.setattr(S, "sonic_moe_available", lambda: True)
    ok, reason = S._sonic_eligible(_cpu_experts(torch.nn.SiLU()), torch.zeros(3, 16, dtype=torch.bfloat16))
    assert not ok and "device" in reason


def test_device_ok_trusts_only_the_probed_capability(monkeypatch):
    caps = {0: (10, 0), 1: (8, 9), 2: (10, 0)}
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=None: caps[i])
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(S, "_DEVICE_OK", {})
    monkeypatch.setattr(S, "_HOME_CAPABILITY", None)
    S._record_device_capabilities()
    assert S._device_ok(torch.device("cuda", 0)) and S._device_ok(torch.device("cuda", 2))
    assert not S._device_ok(torch.device("cuda", 1))
    assert not S._device_ok(torch.device("cpu"))


def test_alignment_verdict_survives_compile_and_declines_unverified():
    def run(weight, source):
        @torch.compile(backend="eager", fullgraph=True)
        def f(w):
            return w.sum() * (2.0 if S._is_k_major_gate_up(w, source) else 3.0)
        torch._dynamo.reset()
        return round((f(weight) / weight.sum()).item())

    aligned = torch.nn.Parameter(torch.randn(2, 32, 16, dtype=torch.bfloat16), requires_grad=False)
    flat = torch.randn(2 * 32 * 16 + 1, dtype=torch.bfloat16)
    misaligned = torch.nn.Parameter(flat[1:].view(2, 32, 16), requires_grad=False)  # 2-byte offset
    assert misaligned.storage_offset() == 1
    # No eager verdict yet: zero-copy is declined under compile, not assumed.
    assert run(aligned.transpose(-2, -1), aligned) == 3
    assert run(misaligned.transpose(-2, -1), misaligned) == 3
    # Eager verdicts, then the compiled region reads them.
    assert S._is_k_major_gate_up(aligned.transpose(-2, -1), aligned)
    assert not S._is_k_major_gate_up(misaligned.transpose(-2, -1), misaligned)
    assert run(aligned.transpose(-2, -1), aligned) == 2
    assert run(misaligned.transpose(-2, -1), misaligned) == 3


def test_interleave_cache_fills_once_under_compile(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_SONIC_CACHE_GIB", "1")
    S._INTERLEAVE_CACHE.clear()
    param = torch.randn(2, 8, 6)

    @torch.compile(backend = "eager", fullgraph = False)
    def f(w):
        return S._interleaved_gate_up(w.transpose(-2, -1).transpose(-2, -1)) * 1.0

    a = f(param)
    assert len(S._INTERLEAVE_CACHE) == 1
    cached = next(iter(S._INTERLEAVE_CACHE.values()))[3]
    f(param)
    assert len(S._INTERLEAVE_CACHE) == 1 and next(iter(S._INTERLEAVE_CACHE.values()))[3] is cached
    torch.testing.assert_close(a, cached)
    S._INTERLEAVE_CACHE.clear()


def test_probe_through_the_custom_op_sees_the_module(monkeypatch):
    # FASTCALL=0: the probe re-enters get_sonic_module() via the custom op.
    import sys
    import types

    fake = types.SimpleNamespace(name = "sonic")
    kernels = types.ModuleType("kernels")
    kernels.get_kernel = lambda *a, **k: fake
    monkeypatch.setitem(sys.modules, "kernels", kernels)
    for name in ("_apply_cutlass_dsl_shim", "_install_sonic_module_path", "_record_device_capabilities",
                 "_concat_layout_supported", "_pinned_ops"):
        monkeypatch.setattr(S, name, lambda *a, **k: None)
    monkeypatch.setattr(S, "_device_is_supported", lambda: (True, None))

    def probe(module):
        if S.get_sonic_module() is not module:
            raise RuntimeError("custom op saw no module")

    monkeypatch.setattr(S, "_probe_sonic", probe)
    monkeypatch.setattr(S, "_SONIC_PROBED", False)
    monkeypatch.setattr(S, "_SONIC_DONE", False)
    monkeypatch.setattr(S, "_SONIC_MODULE", None)
    monkeypatch.delenv("UNSLOTH_DISABLE_MOE_SONIC", raising = False)
    assert S.get_sonic_module() is fake


def test_forward_uses_the_shared_routing_sort_and_skips_the_cache_when_compiling():
    # Source-level: the forward needs SM100 kernels.
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(S.forward_sonic_moe))
    tree = ast.parse(src)
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "moe_routing_sort" in calls and "argsort" not in src
    guarded = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.If) and any(
            isinstance(b, ast.Assign) and isinstance(b.value, ast.Call)
            and getattr(b.value.func, "id", None) == "_interleaved_gate_up"
            for b in n.body
        )
    ]
    assert guarded, "duplicate-cache call must sit under a guard"
    test = ast.unparse(guarded[0].test)
    assert "_interleave_budget_bytes() > 0" in test and "is_compiling()" in test


def test_concurrent_first_probe_waits_for_the_result(monkeypatch):
    import sys
    import threading
    import time
    import types

    fake = types.SimpleNamespace(name = "sonic")
    kernels = types.ModuleType("kernels")
    kernels.get_kernel = lambda *a, **k: fake
    monkeypatch.setitem(sys.modules, "kernels", kernels)
    for name in ("_apply_cutlass_dsl_shim", "_install_sonic_module_path", "_record_device_capabilities",
                 "_concat_layout_supported", "_pinned_ops"):
        monkeypatch.setattr(S, name, lambda *a, **k: None)
    monkeypatch.setattr(S, "_device_is_supported", lambda: (True, None))
    started = threading.Event()

    def slow_probe(module):
        started.set()
        time.sleep(0.3)

    monkeypatch.setattr(S, "_probe_sonic", slow_probe)
    monkeypatch.setattr(S, "_SONIC_PROBED", False)
    monkeypatch.setattr(S, "_SONIC_DONE", False)
    monkeypatch.setattr(S, "_SONIC_MODULE", None)
    monkeypatch.delenv("UNSLOTH_DISABLE_MOE_SONIC", raising = False)
    got = {}
    first = threading.Thread(target = lambda: got.__setitem__("first", S.get_sonic_module()))
    first.start()
    started.wait(5)
    got["second"] = S.get_sonic_module()
    first.join()
    assert got["first"] is fake and got["second"] is fake


def test_rejected_operand_falls_back_by_its_own_device(monkeypatch):
    from unsloth_zoo.temporary_patches import moe_utils as MU

    calls = []
    monkeypatch.setattr(S, "_sonic_eligible", lambda *a: (False, "unsupported operand device"))
    monkeypatch.setattr(MU, "forward_native_grouped_mm", lambda *a: calls.append("grouped_mm"))
    monkeypatch.setattr(MU, "forward_native_moe_loop", lambda *a: calls.append("loop"))
    seen = []
    monkeypatch.setattr(MU, "_grouped_mm_supported_on", lambda device: seen.append(device) or False)

    class FakeCuda:
        is_cuda = True
        device = "cuda:1"

    S.forward_sonic_moe(object(), FakeCuda(), None, None)
    assert calls == ["loop"] and seen == ["cuda:1"]
    monkeypatch.setattr(MU, "_grouped_mm_supported_on", lambda device: True)
    S.forward_sonic_moe(object(), FakeCuda(), None, None)
    assert calls == ["loop", "grouped_mm"]

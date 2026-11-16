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

"""A model kept on the CPU of a GPU host must not send CPU tensors into fla's Triton kernels.

The vendored fla makes fla importable on every GPU host and transformers picks fla over its torch path once per
import, not per device, so a CPU model (device_map="cpu", a CPU eval or merge, the tiny test models) called the
Triton kernels with CPU tensors: "Pointer argument cannot be accessed from Triton (cpu tensor?)". These tests run
patch_vendor_fla's routing step against a fake fla whose kernels fail the same way on CPU tensors.
"""

import os
import sys
import types

import pytest

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_VENDORED_FLA_NO_AUTORUN", "1")

torch = pytest.importorskip("torch")
try:
    from unsloth_zoo.temporary_patches import fla_vendor
except ImportError as _e:
    pytest.skip(f"unsloth_zoo unavailable: {_e}", allow_module_level = True)

pytest.importorskip("transformers.models.qwen3_next.modeling_qwen3_next")


def _recurrent_fp64(q, k, v, g, beta, scale, h0 = None):
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


def _inputs(B, T, H, HV, K, V, seed = 0):
    gen = torch.Generator().manual_seed(seed)
    q, k = torch.randn(B, T, H, K, generator = gen), torch.randn(B, T, H, K, generator = gen)
    v = torch.randn(B, T, HV, V, generator = gen)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, HV, generator = gen)) * 0.5
    beta = torch.rand(B, T, HV, generator = gen)
    return q, k, v, g, beta


def _install_fake_fla(monkeypatch):
    """fla's two gated-delta entry points with its signatures; CPU tensors fail the way Triton's launcher does."""
    calls = []

    def _launch(name, q):
        if q.device.type == "cpu":
            raise ValueError("Pointer argument cannot be accessed from Triton (cpu tensor?)")
        calls.append(name)
        return name, None

    def chunk_gated_delta_rule(q, k, v, g, beta, scale = None, initial_state = None, output_final_state = False,
                               use_qk_l2norm_in_kernel = False, use_beta_sigmoid_in_kernel = False,
                               allow_neg_eigval = False, state_v_first = False, cu_seqlens = None,
                               cu_seqlens_cpu = None, cp_context = None, **kwargs):
        return _launch("chunk", q)

    def fused_recurrent_gated_delta_rule(q, k, v, g = None, gk = None, gv = None, beta = None, scale = None,
                                         initial_state = None, output_final_state = False,
                                         use_qk_l2norm_in_kernel = False, use_gate_in_kernel = False, A_log = None,
                                         dt_bias = None, use_beta_sigmoid_in_kernel = False, allow_neg_eigval = False,
                                         state_v_first = False, cu_seqlens = None, **kwargs):
        return _launch("recurrent", q)

    # fla's entry points are compiler-disabled; the routing wrapper must not inherit that.
    chunk_gated_delta_rule = torch.compiler.disable(chunk_gated_delta_rule)
    fused_recurrent_gated_delta_rule = torch.compiler.disable(fused_recurrent_gated_delta_rule)
    package = types.ModuleType("fla.ops.gated_delta_rule")
    chunk = types.ModuleType("fla.ops.gated_delta_rule.chunk")
    recurrent = types.ModuleType("fla.ops.gated_delta_rule.fused_recurrent")
    package.chunk_gated_delta_rule = chunk.chunk_gated_delta_rule = chunk_gated_delta_rule
    package.fused_recurrent_gated_delta_rule = recurrent.fused_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
    # An older transformers modeling module that bound fla's functions at import.
    modeling = types.ModuleType("transformers.models.qwen3_next.modeling_qwen3_next_fake")
    modeling.chunk_gated_delta_rule = chunk_gated_delta_rule
    modeling.fused_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
    for m in (package, chunk, recurrent):
        monkeypatch.setitem(sys.modules, m.__name__, m)
    monkeypatch.setitem(sys.modules, "transformers.models.qwen3_5.modeling_qwen3_5", modeling)
    return package, chunk, recurrent, modeling, calls


def _run_patch_vendor_fla(monkeypatch):
    """patch_vendor_fla with the vendored injection off and the steps unrelated to device routing stubbed out."""
    monkeypatch.setenv("UNSLOTH_DISABLE_VENDORED_FLA", "1")
    monkeypatch.setenv("FLA_FLASH_QLA", "0")
    for name in ("_patch_l2norm_fp32_on_torch_path", "_alias_missing_gated_delta_names", "_install_fp32_ieee_dots",
                 "_install_gdn_backend", "_repair_kernel_hub_closures", "_install_glm5_kda_safe_gate",
                 "_install_rocm_fp32_torch_path"):
        monkeypatch.setattr(fla_vendor, name, lambda *a, **k: None)
    monkeypatch.setattr(fla_vendor, "_gpu_lacks_dot_instructions", lambda *a, **k: False)
    fla_vendor.patch_vendor_fla()


def test_cpu_tensors_take_the_torch_path_and_match_fp64(monkeypatch):
    package, chunk, recurrent, modeling, calls = _install_fake_fla(monkeypatch)
    _run_patch_vendor_fla(monkeypatch)
    B, T, H, HV, K, V = 2, 70, 2, 4, 16, 8
    q, k, v, g, beta = _inputs(B, T, H, HV, K, V)
    h0 = torch.randn(B, HV, K, V, generator = torch.Generator().manual_seed(3)) * 0.1
    o_ref, ht_ref = _recurrent_fp64(q, k, v, g, beta, 0.3, h0)
    # Every route a model can take: the package, its submodule, a modeling module that bound it at import.
    for fn in (package.chunk_gated_delta_rule, chunk.chunk_gated_delta_rule, modeling.chunk_gated_delta_rule):
        o, ht = fn(q, k, v, g = g, beta = beta, scale = 0.3, initial_state = h0, output_final_state = True,
                   use_qk_l2norm_in_kernel = True)
        torch.testing.assert_close(o.double(), o_ref, atol = 2e-5, rtol = 1e-4)
        torch.testing.assert_close(ht.double(), ht_ref, atol = 2e-5, rtol = 1e-4)
    # One decode step from the prefix state, through fla's fused-recurrent contract.
    q1, k1, v1, g1, b1 = _inputs(B, 1, H, HV, K, V, seed = 7)
    o1_ref, ht1_ref = _recurrent_fp64(q1, k1, v1, g1, b1, K ** -0.5, ht_ref)
    for fn in (package.fused_recurrent_gated_delta_rule, recurrent.fused_recurrent_gated_delta_rule,
               modeling.fused_recurrent_gated_delta_rule):
        o1, ht1 = fn(q1, k1, v1, g = g1, beta = b1, initial_state = ht_ref.float(), output_final_state = True,
                     use_qk_l2norm_in_kernel = True)
        torch.testing.assert_close(o1.double(), o1_ref, atol = 2e-5, rtol = 1e-4)
        torch.testing.assert_close(ht1.double(), ht1_ref, atol = 2e-5, rtol = 1e-4)
    assert calls == []


def test_gpu_tensors_still_reach_fla(monkeypatch):
    package, chunk, recurrent, modeling, calls = _install_fake_fla(monkeypatch)
    _run_patch_vendor_fla(monkeypatch)
    _run_patch_vendor_fla(monkeypatch)  # idempotent: one wrapper, not two
    wrapped = package.chunk_gated_delta_rule
    assert getattr(wrapped, fla_vendor._CPU_TORCH_MARK, None) == "torch"
    assert getattr(wrapped.__wrapped__, fla_vendor._CPU_TORCH_MARK, None) is None
    assert not getattr(wrapped, "_torchdynamo_disable", False)
    gpu = types.SimpleNamespace(device = types.SimpleNamespace(type = "cuda"))
    assert package.chunk_gated_delta_rule(gpu, gpu, gpu, gpu, gpu) == ("chunk", None)
    assert modeling.fused_recurrent_gated_delta_rule(q = gpu, k = gpu, v = gpu, g = gpu, beta = gpu) == ("recurrent", None)
    assert calls == ["chunk", "recurrent"]
    # The torch path refuses what it cannot compute instead of computing something else.
    q, k, v, g, beta = _inputs(1, 1, 2, 2, 8, 8)
    with pytest.raises(RuntimeError, match = "gk / gv"):
        package.fused_recurrent_gated_delta_rule(q, k, v, gk = g, beta = beta)


def test_cpu_path_compiles_without_a_graph_break(monkeypatch):
    """The stock torch path compiles whole on a CPU-only host; a GPU host must not turn it into a break."""
    package, chunk, recurrent, modeling, calls = _install_fake_fla(monkeypatch)
    _run_patch_vendor_fla(monkeypatch)
    q, k, v, g, beta = _inputs(1, 8, 2, 2, 8, 8)
    fn = modeling.chunk_gated_delta_rule

    def layer(q, k, v, g, beta):
        return fn(q, k, v, g = g, beta = beta, use_qk_l2norm_in_kernel = True)[0] * 2

    torch._dynamo.reset()
    try:
        out = torch.compile(layer, backend = "eager", fullgraph = True)(q, k, v, g, beta)
    finally:
        torch._dynamo.reset()
    torch.testing.assert_close(out, layer(q, k, v, g, beta))
    assert calls == []


def test_cpu_path_repeated_compiled_calls_reuse_one_graph(monkeypatch):
    """A diagnostic call counter bumped inside the traced CPU wrapper is a guarded global: every call recompiled
    and fullgraph=True hit FailOnRecompileLimitHit on call nine. Same shapes must keep one graph."""
    from torch._dynamo.testing import CompileCounter

    package, chunk, recurrent, modeling, calls = _install_fake_fla(monkeypatch)
    _run_patch_vendor_fla(monkeypatch)
    q, k, v, g, beta = _inputs(1, 8, 2, 2, 8, 8)
    fn = modeling.chunk_gated_delta_rule

    def layer(q, k, v, g, beta):
        return fn(q, k, v, g = g, beta = beta, use_qk_l2norm_in_kernel = True)[0] * 2

    counter = CompileCounter()
    torch._dynamo.reset()
    try:
        compiled = torch.compile(layer, backend = counter, fullgraph = True)
        for _ in range(12):
            out = compiled(q, k, v, g, beta)
    finally:
        torch._dynamo.reset()
    torch.testing.assert_close(out, layer(q, k, v, g, beta))
    assert counter.frame_count == 1
    assert calls == []

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


def _bind_every_alias(monkeypatch, package, chunk, recurrent):
    """Installed fla's and the vendored tree's other spellings: chunk_gdn / fused_recurrent_gdn in the kernel
    submodules and the package, the fla.ops re-exports and the decode alias (recurrent_gated_delta_rule)."""
    ops = types.ModuleType("fla.ops")
    ops.gated_delta_rule = package
    root = types.ModuleType("fla")
    root.ops = ops
    chunk_fn, recurrent_fn = package.chunk_gated_delta_rule, package.fused_recurrent_gated_delta_rule
    for module in (package, ops):
        module.chunk_gated_delta_rule = module.chunk_gdn = chunk_fn
        module.fused_recurrent_gated_delta_rule = module.fused_recurrent_gdn = recurrent_fn
    package.recurrent_gated_delta_rule = recurrent_fn
    chunk.chunk_gdn = chunk_fn
    recurrent.fused_recurrent_gdn = recurrent_fn
    monkeypatch.setitem(sys.modules, "fla", root)
    monkeypatch.setitem(sys.modules, "fla.ops", ops)
    return ops


_CHUNK_SPELLINGS = [("package", "chunk_gdn"), ("chunk", "chunk_gdn"), ("ops", "chunk_gated_delta_rule"), ("ops", "chunk_gdn")]
_RECURRENT_SPELLINGS = [
    ("package", "recurrent_gated_delta_rule"), ("package", "fused_recurrent_gdn"), ("recurrent", "fused_recurrent_gdn"),
    ("ops", "fused_recurrent_gated_delta_rule"), ("ops", "fused_recurrent_gdn"),
]


@pytest.mark.parametrize("where, name", _CHUNK_SPELLINGS + _RECURRENT_SPELLINGS)
def test_every_gated_delta_spelling_takes_the_cpu_path(monkeypatch, where, name):
    package, chunk, recurrent, modeling, calls = _install_fake_fla(monkeypatch)
    ops = _bind_every_alias(monkeypatch, package, chunk, recurrent)
    _run_patch_vendor_fla(monkeypatch)
    _run_patch_vendor_fla(monkeypatch)  # a second install keeps every spelling on the one wrapper
    fn = getattr({"package": package, "chunk": chunk, "recurrent": recurrent, "ops": ops}[where], name)
    assert getattr(fn, fla_vendor._CPU_TORCH_MARK, None) == "torch"
    B, T = 1, (6 if (where, name) in _CHUNK_SPELLINGS else 1)
    q, k, v, g, beta = _inputs(B, T, 2, 4, 8, 8)
    h0 = torch.randn(B, 4, 8, 8, generator = torch.Generator().manual_seed(5)) * 0.1
    o_ref, ht_ref = _recurrent_fp64(q, k, v, g, beta, 8 ** -0.5, h0)
    o, ht = fn(q, k, v, g = g, beta = beta, initial_state = h0, output_final_state = True, use_qk_l2norm_in_kernel = True)
    torch.testing.assert_close(o.double(), o_ref, atol = 2e-5, rtol = 1e-4)
    torch.testing.assert_close(ht.double(), ht_ref, atol = 2e-5, rtol = 1e-4)
    assert calls == []


def test_rocm_wrapper_installed_first_keeps_chunk_gdn_on_the_cpu_path(monkeypatch):
    """The gfx11 float32 wrapper replaces the canonical chunk callable first; chunk_gdn must follow it, or the
    CPU installer (which matches by identity) leaves the shorthand on the raw kernel."""
    package, chunk, recurrent, modeling, calls = _install_fake_fla(monkeypatch)
    ops = _bind_every_alias(monkeypatch, package, chunk, recurrent)
    monkeypatch.setattr(fla_vendor, "_rocm_has_fp32_faulting_gpu", lambda *a, **k: True)
    assert fla_vendor._install_rocm_fp32_torch_path()
    assert fla_vendor._install_cpu_torch_path()
    q, k, v, g, beta = _inputs(1, 6, 2, 2, 8, 8)
    o_ref, _ = _recurrent_fp64(q, k, v, g, beta, 8 ** -0.5)
    for fn in (package.chunk_gated_delta_rule, package.chunk_gdn, chunk.chunk_gdn, ops.chunk_gdn):
        o, _ = fn(q, k, v, g = g, beta = beta, use_qk_l2norm_in_kernel = True)
        torch.testing.assert_close(o.double(), o_ref, atol = 2e-5, rtol = 1e-4)
    assert calls == []


def test_a_separately_implemented_export_is_never_rebound(monkeypatch):
    package, chunk, recurrent, modeling, calls = _install_fake_fla(monkeypatch)

    def upstream_recurrent(q, k, v, g, beta):
        return "upstream"

    package.recurrent_gated_delta_rule = package.fused_recurrent_gdn = upstream_recurrent
    _run_patch_vendor_fla(monkeypatch)
    assert getattr(package.fused_recurrent_gated_delta_rule, fla_vendor._CPU_TORCH_MARK, None) == "torch"
    assert package.recurrent_gated_delta_rule is upstream_recurrent
    assert package.fused_recurrent_gdn is upstream_recurrent


def test_cpu_decode_reset_gate_clears_the_state_instead_of_nan(monkeypatch):
    """g = -inf is a full reset (decay exp(g) = 0): fla's fused_recurrent kernel and the fp64 recurrence
    give a finite output and state; the chunk reference forms -inf - -inf = NaN in its decay matrix."""
    package, chunk, recurrent, modeling, calls = _install_fake_fla(monkeypatch)
    _run_patch_vendor_fla(monkeypatch)
    q, k, v, g, beta = _inputs(1, 1, 2, 4, 8, 8)
    g = torch.full_like(g, -float("inf"))
    h0 = torch.ones(1, 4, 8, 8)
    o_ref, ht_ref = _recurrent_fp64(q, k, v, g, beta, 8 ** -0.5, h0)
    assert torch.isfinite(o_ref).all() and torch.isfinite(ht_ref).all()
    o, ht = package.fused_recurrent_gated_delta_rule(
        q, k, v, g = g, beta = beta, initial_state = h0, output_final_state = True, use_qk_l2norm_in_kernel = True,
    )
    torch.testing.assert_close(o.double(), o_ref, atol = 2e-5, rtol = 1e-4)
    torch.testing.assert_close(ht.double(), ht_ref, atol = 2e-5, rtol = 1e-4)
    assert calls == []


def test_cpu_decode_keeps_fla_defaults_and_output_dtype(monkeypatch):
    """fla's fused_recurrent: beta defaults to 1 per value head, the output is allocated like v."""
    package, chunk, recurrent, modeling, calls = _install_fake_fla(monkeypatch)
    _run_patch_vendor_fla(monkeypatch)
    q, k, v, g, beta = _inputs(1, 1, 2, 4, 8, 8)
    o_ref, _ = _recurrent_fp64(q, k, v, g, torch.ones_like(beta), 8 ** -0.5)
    for passed in ({}, {"beta": None}):
        o, _ = package.fused_recurrent_gated_delta_rule(q, k, v, g = g, use_qk_l2norm_in_kernel = True, **passed)
        torch.testing.assert_close(o.double(), o_ref, atol = 2e-5, rtol = 1e-4)
    o, _ = package.fused_recurrent_gated_delta_rule(q, k, v.bfloat16(), g = g, beta = beta, use_qk_l2norm_in_kernel = True)
    assert o.dtype == torch.bfloat16
    assert calls == []


def test_cold_reference_cache_compiles_one_graph(monkeypatch):
    from torch._dynamo.testing import CompileCounter

    monkeypatch.setitem(fla_vendor._rocm_fp32_state, "reference", None)
    package, chunk, recurrent, modeling, calls = _install_fake_fla(monkeypatch)
    _run_patch_vendor_fla(monkeypatch)
    q, k, v, g, beta = _inputs(1, 8, 2, 2, 8, 8)
    fn = package.chunk_gated_delta_rule
    counter = CompileCounter()
    torch._dynamo.reset()
    try:
        compiled = torch.compile(lambda *t: fn(*t, use_qk_l2norm_in_kernel = True)[0] * 2, backend = counter)
        for _ in range(3):
            compiled(q, k, v, g, beta)
    finally:
        torch._dynamo.reset()
    assert counter.frame_count == 1


def test_olmo_hybrid_bound_before_the_patch_takes_the_cpu_path(monkeypatch):
    """olmo_hybrid is left out of the general kernel-hub repair; a closure that captured the raw kernel the
    CPU wrapper now wraps is still re-resolved, so CPU decode never reaches Triton."""
    hub = pytest.importorskip("transformers.integrations.hub_kernels")
    decorate = getattr(hub, "use_kernel_func_from_hub_with_fallback", None)
    if decorate is None:
        pytest.skip("transformers without use_kernel_func_from_hub_with_fallback")
    package, chunk, recurrent, modeling, calls = _install_fake_fla(monkeypatch)
    _bind_every_alias(monkeypatch, package, chunk, recurrent)
    from transformers.models.qwen3_next import modeling_qwen3_next as qwen3_next

    olmo = types.ModuleType("transformers.models.olmo_hybrid.modeling_olmo_hybrid")
    olmo.torch_recurrent_gated_delta_rule = decorate("fused_recurrent_gated_delta_rule", "fla")(
        inspect_unwrap(qwen3_next.torch_recurrent_gated_delta_rule))
    assert fla_vendor._resolved_implementation(olmo.torch_recurrent_gated_delta_rule) is package.fused_recurrent_gated_delta_rule
    monkeypatch.setitem(sys.modules, olmo.__name__, olmo)
    monkeypatch.setenv("UNSLOTH_DISABLE_VENDORED_FLA", "1")
    monkeypatch.setenv("FLA_FLASH_QLA", "0")
    for name in ("_patch_l2norm_fp32_on_torch_path", "_alias_missing_gated_delta_names", "_install_fp32_ieee_dots",
                 "_install_gdn_backend", "_install_glm5_kda_safe_gate", "_install_rocm_fp32_torch_path"):
        monkeypatch.setattr(fla_vendor, name, lambda *a, **k: None)
    monkeypatch.setattr(fla_vendor, "_gpu_lacks_dot_instructions", lambda *a, **k: False)
    fla_vendor.patch_vendor_fla()
    q, k, v, g, beta = _inputs(1, 1, 2, 2, 8, 8)
    o_ref, _ = _recurrent_fp64(q, k, v, g, beta, 8 ** -0.5)
    o, _ = olmo.torch_recurrent_gated_delta_rule(q, k, v, g = g, beta = beta, use_qk_l2norm_in_kernel = True)
    torch.testing.assert_close(o.double(), o_ref, atol = 2e-5, rtol = 1e-4)
    assert calls == []


def inspect_unwrap(fn):
    import inspect
    return inspect.unwrap(fn)


@pytest.mark.parametrize("entry", ["chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"])
def test_gpu_host_cold_reference_first_call_compiles_fullgraph(monkeypatch, entry):
    """A GPU host defers the reference to the first CPU call. When that first call is a fullgraph compile and no
    Qwen modeling module is imported, resolving it must not call importlib.import_module, which Dynamo refuses."""
    package, chunk, recurrent, modeling, calls = _install_fake_fla(monkeypatch)
    for key in ("reference", "recurrent_reference"):
        monkeypatch.setitem(fla_vendor._rocm_fp32_state, key, None)
    for model in ("qwen3_next", "qwen3_5_moe"):
        name = f"transformers.models.{model}.modeling_{model}"
        if name in sys.modules:
            monkeypatch.delitem(sys.modules, name)
    with monkeypatch.context() as m:
        m.setattr(torch.cuda, "is_available", lambda: True)
        _run_patch_vendor_fla(monkeypatch)
    assert fla_vendor._rocm_fp32_state["reference"] is None
    q, k, v, g, beta = _inputs(1, 6, 2, 2, 8, 8)
    fn = getattr(package, entry)

    def layer(q, k, v, g, beta):
        return fn(q, k, v, g = g, beta = beta, use_qk_l2norm_in_kernel = True)[0] * 2

    torch._dynamo.reset()
    try:
        out = torch.compile(layer, backend = "eager", fullgraph = True)(q, k, v, g, beta)
    finally:
        torch._dynamo.reset()
    want, _ = _recurrent_fp64(q, k, v, g, beta, 8 ** -0.5)
    torch.testing.assert_close(out.double(), 2 * want, atol = 1e-4, rtol = 1e-4)
    assert calls == []


@pytest.mark.parametrize("has_gpu", [True, False])
def test_install_imports_no_modeling_module_on_a_gpu_host(monkeypatch, has_gpu):
    """Routing the live fla at `import unsloth` must not import transformers' Qwen3-Next modeling on a GPU host (about
    a second for every model): there the reference comes from a modeling module already imported, or from the first
    CPU call. A host with no GPU still resolves it at install, outside any trace."""
    import importlib

    package, chunk, recurrent, modeling, calls = _install_fake_fla(monkeypatch)
    for key in ("reference", "recurrent_reference"):
        monkeypatch.setitem(fla_vendor._rocm_fp32_state, key, None)
    saved = {}
    for model in ("qwen3_next", "qwen3_5_moe"):
        name = f"transformers.models.{model}.modeling_{model}"
        if name in sys.modules:
            saved[name] = sys.modules[name]
            monkeypatch.delitem(sys.modules, name)
    imported = []
    real_import = importlib.import_module

    def import_module(name, *args, **kwargs):
        imported.append(name)
        return saved[name] if name in saved else real_import(name, *args, **kwargs)

    monkeypatch.setattr(fla_vendor.importlib, "import_module", import_module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: has_gpu)
    assert fla_vendor._install_cpu_torch_path()
    modeling_imports = [n for n in imported if ".modeling_" in n]
    if has_gpu:
        assert modeling_imports == []
        assert fla_vendor._rocm_fp32_state["reference"] is None
    else:
        assert "transformers.models.qwen3_next.modeling_qwen3_next" in modeling_imports
        assert fla_vendor._rocm_fp32_state["reference"] is not None
    # The first CPU call resolves it and runs the torch path.
    q, k, v, g, beta = _inputs(1, 6, 2, 2, 8, 8)
    out, _ = package.chunk_gated_delta_rule(q, k, v, g = g, beta = beta, use_qk_l2norm_in_kernel = True)
    want, _ = _recurrent_fp64(q, k, v, g, beta, 8 ** -0.5)
    torch.testing.assert_close(out.double(), want, atol = 1e-4, rtol = 1e-4)
    assert fla_vendor._rocm_fp32_state["reference"] is not None
    assert calls == []


def test_cpu_decode_with_half_qk_and_float_v_keeps_values_past_half_range(monkeypatch):
    """fla's fused_recurrent computes in float32 and stores into an output allocated like v. With fp16 q / k and
    fp32 v, an output of 100000 (past fp16's 65504) must come back as 100000, not inf from a pass through q's dtype."""
    package, chunk, recurrent, modeling, calls = _install_fake_fla(monkeypatch)
    _run_patch_vendor_fla(monkeypatch)
    K = 256
    q = torch.ones(1, 1, 1, K, dtype = torch.float16)
    k = torch.ones(1, 1, 1, K, dtype = torch.float16)
    v = torch.full((1, 1, 1, 4), 6250.0)
    g = torch.zeros(1, 1, 1)
    beta = torch.ones(1, 1, 1)
    o, ht = package.fused_recurrent_gated_delta_rule(
        q, k, v, g = g, beta = beta, scale = 1 / 16, output_final_state = True,
    )
    assert o.dtype == torch.float32
    torch.testing.assert_close(o, torch.full_like(o, 100000.0))
    assert ht.dtype == torch.float32 and torch.isfinite(ht).all()
    # The chunk entry point returns q's dtype, as fla's chunk does (o.to(q.dtype)).
    o, _ = package.chunk_gated_delta_rule(q, k, v, g = g, beta = beta, scale = 1 / 16)
    assert o.dtype == torch.float16
    assert calls == []

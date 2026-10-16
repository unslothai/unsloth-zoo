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

"""CPU fakes of each triton_kernels API generation's index semantics: a translation mistake gives a wrong MoE output."""

import dataclasses
import logging
import sys
import types

import pytest
import torch

import unsloth_zoo.triton_kernels_compat as tkc
from unsloth_zoo.temporary_patches import gpt_oss as go

_REAL_IMPORT_VLLM_VENDORED = tkc._import_vllm_vendored


def _expert_rows(x_rows, w, bias, hist):
    out, lo = [], 0
    for e, count in enumerate(hist.tolist()):
        y = x_rows[lo : lo + count] @ w[e]
        if bias is not None:
            y = y + bias[e]
        out.append(y)
        lo += count
    return torch.cat(out)


def _activate(y, reduction_n):
    # Stand-in fused activation: sums column pairs, so the reduction width must arrive.
    return y.view(y.shape[0], -1, reduction_n).sum(-1)


def _make_fake(name, api):
    pkg = types.ModuleType(name)
    pkg.__path__ = []
    mods = {"": pkg}

    def add(sub):
        m = types.ModuleType(f"{name}.{sub}")
        mods[sub] = m
        return m

    tensor = add("tensor")
    tensor.FP4 = "fp4"
    tensor.convert_layout = lambda t, layout, **kw: t
    tensor.wrap_torch_tensor = lambda t, dtype = None: t
    add("tensor_details").__path__ = []
    add("tensor_details.layout")
    add("swiglu").swiglu_fn = "swiglu_fn"
    calls = pkg.calls = []

    @dataclasses.dataclass(frozen = True)
    class InFlexData:
        pass

    @dataclasses.dataclass(frozen = True)
    class FlexCtx:
        rhs_data: object = None

    if api == tkc.API_OGS_ROUTING:
        @dataclasses.dataclass(frozen = True)
        class FnSpecs:
            name: str
            fn: object
            fn_arg_names: tuple

        @dataclasses.dataclass(frozen = True)
        class FusedActivation:
            specs: object = None
            fn_args: tuple = ()
            reduction_n: int = 1
    else:
        @dataclasses.dataclass(frozen = True)
        class FnSpecs:
            name: str
            fn: object
            fn_arg_names: tuple = ()
            fn_arg_do_not_specialize: tuple = ()
            reduction_n: int = 1

        @dataclasses.dataclass(frozen = True)
        class FusedActivation:
            specs: object = None
            fn_args: tuple = ()

    def reduction_of(act):
        if act is None:
            return 1
        return act.reduction_n if api == tkc.API_OGS_ROUTING else act.specs.reduction_n

    if api == tkc.API_MATMUL_RAGGED:
        mm = add("matmul")

        @dataclasses.dataclass
        class PrecisionConfig:
            flex_ctx: object = None
            b_mx_scale: object = None
            b_microblock_size: int = None
            out_dtype: object = None

        def matmul(a, b, bias, a_ragged_metadata = None, b_ragged_metadata = None, gather_indx = None,
                   scatter_indx = None, precision_config = None, betas = None, gammas = None,
                   fused_activation = None, **unused):
            calls.append(("matmul", gather_indx, scatter_indx, precision_config))
            assert gather_indx is None or isinstance(gather_indx, torch.Tensor)
            assert scatter_indx is None or isinstance(scatter_indx, torch.Tensor)
            pc = precision_config or PrecisionConfig()
            assert pc.b_mx_scale is None or pc.b_microblock_size == 32
            rows = a if gather_indx is None else a[gather_indx.long()]
            y = _expert_rows(rows.float(), b, bias, a_ragged_metadata.slice_sizes)
            if gammas is not None:
                y = y * gammas[:, None]
            y = _activate(y, reduction_of(fused_activation))
            dtype = pc.out_dtype or a.dtype
            if scatter_indx is None:
                return y.to(dtype)
            out = torch.full((scatter_indx.shape[0], y.shape[-1]), float("nan"), dtype = dtype)
            out[scatter_indx.long()] = y.to(dtype)
            return out

        mm.matmul = matmul
    else:
        mm = add("matmul_ogs")

        @dataclasses.dataclass
        class PrecisionConfig:
            flex_ctx: object = None
            weight_scale: object = None
            out_dtype: object = None

        def matmul_ogs(x, w, bias, routing_data = None, gather_indx = None, scatter_indx = None,
                       precision_config = None, betas = None, gammas = None, fused_activation = None, **unused):
            calls.append(("matmul_ogs", gather_indx, scatter_indx, precision_config))
            if api == tkc.API_OGS_RAGGED:
                assert routing_data.expt_data is not None and hasattr(routing_data.expt_data, "slice_sizes")
            k = routing_data.n_expts_act
            rows = x if gather_indx is None else x[(gather_indx.src_indx // k).long()]
            y = _expert_rows(rows.float(), w, bias, routing_data.expt_hist)
            if gammas is not None:
                y = y * gammas[:, None]
            y = _activate(y, reduction_of(fused_activation))
            if scatter_indx is None:
                return y.to(x.dtype)
            out = torch.zeros((y.shape[0] // k, y.shape[-1]))
            out.index_add_(0, (scatter_indx.dst_indx // k).long(), y)
            return out.to(x.dtype)

        mm.matmul_ogs = matmul_ogs

    mm.PrecisionConfig, mm.FlexCtx, mm.InFlexData = PrecisionConfig, FlexCtx, InFlexData
    mm.FnSpecs, mm.FusedActivation = FnSpecs, FusedActivation

    def sorted_topk(logits, k):
        vals, indx = torch.topk(logits, k, dim = -1)
        # Dense (n, E) weights: topk_to_routing_tensors reads an (n, k) array as dense when k == E.
        dense = torch.zeros_like(logits).scatter_(1, indx, torch.softmax(vals, -1))
        return indx, dense

    if api == tkc.API_OGS_ROUTING:
        rt = add("routing")
        holder = rt
    else:
        holder = mm
        topk_mod = add("topk")
        ragged = add("tensor_details.ragged_tensor")

        def topk(logits, k, apply_softmax = True, **unused):
            assert apply_softmax
            indx, dense = sorted_topk(logits, k)
            gate_scal, hist, combine, dispatch = go.topk_to_routing_tensors(indx, dense, logits.shape[-1])
            # vals in expert order per row, as topk returns them; gate_scal = vals.flatten()[combine]
            indx = torch.sort(indx, dim = 1).values
            vals = torch.gather(dense, 1, indx.long())
            meta = types.SimpleNamespace(col_sum = hist, col_sorted_indx = combine, row_sorted_indx = dispatch)
            return types.SimpleNamespace(vals = vals, indx = indx, mask_metadata = meta)

        @dataclasses.dataclass
        class RaggedTensorMetadata:
            slice_sizes: torch.Tensor
            slice_offs: torch.Tensor

        RaggedTensorMetadata.__module__ = ragged.__name__

        def make_ragged_tensor_metadata(slice_sizes, n_total_rows):
            assert slice_sizes.dtype == torch.int32
            offs = torch.cat((slice_sizes.new_zeros(1), torch.cumsum(slice_sizes, 0, dtype = torch.int32)))
            assert int(offs[-1]) == n_total_rows
            return RaggedTensorMetadata(slice_sizes, offs)

        topk_mod.topk = topk
        ragged.make_ragged_tensor_metadata = make_ragged_tensor_metadata

    if api != tkc.API_MATMUL_RAGGED:
        @dataclasses.dataclass
        class GatherIndx:
            src_indx: torch.Tensor
            dst_indx: torch.Tensor

        @dataclasses.dataclass
        class ScatterIndx:
            src_indx: torch.Tensor
            dst_indx: torch.Tensor

        @dataclasses.dataclass
        class RoutingData:
            gate_scal: torch.Tensor
            expt_hist: torch.Tensor
            n_expts_tot: int
            n_expts_act: int
            expt_data: object = None

        for cls in (GatherIndx, ScatterIndx, RoutingData):
            cls.__module__ = holder.__name__
            setattr(holder, cls.__name__, cls)

    if api == tkc.API_OGS_ROUTING:
        def routing(logits, n_expts_act):
            indx, vals = sorted_topk(logits, n_expts_act)
            gate_scal, hist, combine, dispatch = go.topk_to_routing_tensors(indx, vals, logits.shape[-1])
            return (
                rt.RoutingData(gate_scal, hist, logits.shape[-1], n_expts_act, None),
                rt.GatherIndx(combine, dispatch), rt.ScatterIndx(dispatch, combine),
            )

        rt.routing = routing
    return pkg, mods


def _install(monkeypatch, name, api):
    pkg, mods = _make_fake(name, api)
    for sub, mod in mods.items():
        monkeypatch.setitem(sys.modules, name if not sub else f"{name}.{sub}", mod)
    return pkg


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(tkc, "_MODULE", tkc._UNRESOLVED)
    monkeypatch.setattr(tkc, "_BACKENDS", {})
    monkeypatch.setattr(tkc, "_LOGGED", set())
    monkeypatch.setattr(tkc, "_DEVICE_OK", {})
    monkeypatch.setattr(go, "_TRITON_KERNELS_AVAILABLE", None)
    for key in [k for k in sys.modules if k == "triton_kernels" or k.startswith("triton_kernels.")]:
        monkeypatch.delitem(sys.modules, key)
    monkeypatch.delenv("UNSLOTH_TRITON_KERNELS_SOURCE", raising = False)
    monkeypatch.delenv("UNSLOTH_MXFP4_MATMUL_OGS", raising = False)
    monkeypatch.setattr(tkc, "_import_vllm_vendored", lambda: None)
    before = set(sys.modules)
    yield monkeypatch
    # monkeypatch restores modules it replaced, not ones a test imported fresh: a fake `vllm`
    # left behind makes a later importorskip("vllm") on this worker return it instead of skipping.
    for key in [k for k in sys.modules if k not in before and k.split(".", 1)[0] in ("vllm", "triton_kernels")]:
        del sys.modules[key]


ALL_APIS = (tkc.API_OGS_ROUTING, tkc.API_OGS_RAGGED, tkc.API_MATMUL_RAGGED)



@pytest.mark.parametrize("api", ALL_APIS)
def test_each_generation_resolves_and_is_detected(fresh, api):
    pkg = _install(fresh, "triton_kernels", api)
    assert tkc.get_triton_kernels() is pkg
    assert tkc.selected_triton_kernels() == ("triton_kernels", api)
    assert go._check_triton_kernels_available() is True


def test_selection_is_logged_once(fresh, caplog):
    from unsloth_zoo.log import logger
    _install(fresh, "triton_kernels", tkc.API_MATMUL_RAGGED)
    fresh.setattr(logger, "level", logging.INFO)
    with caplog.at_level(logging.INFO, logger = logger.name):
        tkc.get_triton_kernels()
        tkc.get_triton_kernels_backend()
        fresh.setattr(tkc, "_MODULE", tkc._UNRESOLVED)
        tkc.get_triton_kernels()
    lines = [r.getMessage() for r in caplog.records if "MXFP4 experts use triton_kernels" in r.getMessage()]
    assert lines == ["Unsloth: MXFP4 experts use triton_kernels 'triton_kernels' (matmul+ragged API)."]


def test_matmul_ogs_without_routing_or_ragged_metadata_is_refused(fresh):
    _, mods = _make_fake("triton_kernels", tkc.API_OGS_RAGGED)
    del mods["topk"]
    for sub, mod in mods.items():
        fresh.setitem(sys.modules, "triton_kernels" if not sub else f"triton_kernels.{sub}", mod)
    fresh.setitem(sys.modules, "triton_kernels.topk", None)
    assert tkc.get_triton_kernels() is None
    assert go._check_triton_kernels_available() is False


def test_matmul_without_topk_is_refused(fresh):
    _, mods = _make_fake("triton_kernels", tkc.API_MATMUL_RAGGED)
    del mods["tensor_details.ragged_tensor"]
    for sub, mod in mods.items():
        fresh.setitem(sys.modules, "triton_kernels" if not sub else f"triton_kernels.{sub}", mod)
    fresh.setitem(sys.modules, "triton_kernels.tensor_details.ragged_tensor", None)
    assert tkc.get_triton_kernels() is None


def test_none_installed(fresh):
    fresh.setattr(tkc.importlib.util, "find_spec", lambda name, *a: None)
    assert tkc.get_triton_kernels() is None
    assert tkc.get_triton_kernels_backend() is None
    assert tkc.selected_triton_kernels() is None
    assert tkc.matmul_ogs_available("cpu") is False
    assert go._ogs_modules.__wrapped__() is None


@pytest.mark.parametrize("source, want", [("auto", "triton_kernels"), ("triton_kernels", "triton_kernels"),
                                          ("vllm", "vllm.third_party.triton_kernels")])
def test_source_can_be_forced_when_both_exist(fresh, source, want):
    _install(fresh, "triton_kernels", tkc.API_MATMUL_RAGGED)
    vendored = _install(fresh, "vllm.third_party.triton_kernels", tkc.API_OGS_ROUTING)
    fresh.setattr(tkc, "_import_vllm_vendored", lambda: tkc._probe(vendored))
    fresh.setenv("UNSLOTH_TRITON_KERNELS_SOURCE", source)
    assert tkc.get_triton_kernels().__name__ == want
    backend = tkc.get_triton_kernels_backend()
    assert backend.source == ("vllm" if source == "vllm" else "triton_kernels")


def test_vendored_copy_is_used_when_top_level_is_missing(fresh):
    vendored = _install(fresh, "vllm.third_party.triton_kernels", tkc.API_OGS_ROUTING)
    fresh.setitem(sys.modules, "triton_kernels", None)
    fresh.setattr(tkc, "_import_top_level", lambda: None)
    fresh.setattr(tkc, "_import_vllm_vendored", lambda: tkc._probe(vendored))
    assert tkc.selected_triton_kernels() == ("vllm.third_party.triton_kernels", tkc.API_OGS_ROUTING)
    assert go._check_triton_kernels_available() is False


def test_forced_vendored_import_ignores_cached_top_level_children(fresh, tmp_path):
    """Its absolute `triton_kernels.topk` import must load the vendored topk, not a cached top-level one."""
    vendored_dir = tmp_path / "vllm" / "third_party" / "triton_kernels"
    vendored_dir.mkdir(parents = True)
    (tmp_path / "vllm" / "__init__.py").write_text("")
    (tmp_path / "vllm" / "third_party" / "__init__.py").write_text("")
    (vendored_dir / "__init__.py").write_text("")
    (vendored_dir / "topk.py").write_text("SOURCE = 'vendored'\n")
    (vendored_dir / "routing.py").write_text("from triton_kernels.topk import SOURCE\n")
    for key in [k for k in sys.modules if k == "vllm" or k.startswith("vllm.")]:
        fresh.delitem(sys.modules, key)
    fresh.syspath_prepend(str(tmp_path))
    # Unsloth's import fixes may hide vllm when the installed one is unusable.
    fresh.setattr(sys, "meta_path", [f for f in sys.meta_path if "vllm" not in type(f).__name__.lower()])
    find_spec = tkc.importlib.util.find_spec
    fresh.setattr(tkc.importlib.util, "find_spec", getattr(find_spec, "_unsloth_original_find_spec", find_spec))
    top, top_topk = types.ModuleType("triton_kernels"), types.ModuleType("triton_kernels.topk")
    top_topk.SOURCE = "top-level"
    fresh.setitem(sys.modules, "triton_kernels", top)
    fresh.setitem(sys.modules, "triton_kernels.topk", top_topk)
    fresh.setattr(tkc, "_probe", lambda module: module)
    assert _REAL_IMPORT_VLLM_VENDORED().__name__ == "vllm.third_party.triton_kernels"
    assert sys.modules["vllm.third_party.triton_kernels.routing"].SOURCE == "vendored"
    assert sys.modules["triton_kernels"] is top and sys.modules["triton_kernels.topk"] is top_topk



def _moe_inputs(T = 7, H = 8, I = 6, E = 4, seed = 0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(T, H, generator = gen)
    logits = torch.randn(T, E, generator = gen)
    w1, b1 = torch.randn(E, H, 2 * I, generator = gen), torch.randn(E, 2 * I, generator = gen)
    w2, b2 = torch.randn(E, I, H, generator = gen), torch.randn(E, H, generator = gen)
    return x, logits, w1, b1, w2, b2


def _reference_moe(x, logits, w1, b1, w2, b2, k):
    vals, experts = torch.topk(logits, k, dim = -1)
    weights = torch.softmax(vals, -1)
    out = torch.zeros(x.shape[0], w2.shape[-1])
    for t in range(x.shape[0]):
        for slot in range(k):
            e = experts[t, slot]
            h = _activate((x[t : t + 1] @ w1[e] + b1[e]), 2)
            out[t] += weights[t, slot] * (h @ w2[e] + b2[e])[0]
    return out


def _backend_moe(backend, x, logits, w1, b1, w2, b2, k, routing = None):
    rd, gather, scatter = routing or backend.routing(logits, k)
    act = backend.swiglu_activation(1.702, 7.0)
    pc = backend.precision("scale")
    h = backend.gate_up(x, w1, b1, rd, gather, pc, fused_activation = act)
    return backend.down(h, w2, b2, rd, scatter, pc, gammas = rd.gate_scal)


@pytest.mark.parametrize("api", ALL_APIS)
@pytest.mark.parametrize("k", [1, 2, 4])
def test_moe_matches_dense_reference_on_every_generation(fresh, api, k):
    _install(fresh, "triton_kernels", api)
    backend = tkc.get_triton_kernels_backend()
    inputs = _moe_inputs()
    got = _backend_moe(backend, *inputs, k)
    want = _reference_moe(*inputs, k)
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, rtol = 1e-5, atol = 1e-5)


@pytest.mark.parametrize("api", ALL_APIS)
def test_routing_from_router_topk_matches_backend_routing(fresh, api):
    import contextlib
    fresh.setattr(go, "torch_cuda_device", lambda device: contextlib.nullcontext())
    _install(fresh, "triton_kernels", api)
    backend = tkc.get_triton_kernels_backend()
    x, logits, w1, b1, w2, b2 = _moe_inputs(T = 9)
    k = 2
    vals, experts = torch.topk(logits, k, dim = -1)
    module = types.SimpleNamespace(num_experts = logits.shape[-1])
    module.gate_up_proj = w1
    ours = go._mxfp4_routing_from_topk(module, experts, torch.softmax(vals, -1))
    theirs = backend.routing(logits, k)
    assert type(ours[0]) is type(theirs[0])
    for a, b in zip(ours, theirs):
        for field in dataclasses.fields(a):
            va, vb = getattr(a, field.name), getattr(b, field.name)
            if isinstance(va, torch.Tensor):
                torch.testing.assert_close(va, vb)
            elif field.name != "expt_data":
                assert va == vb
    torch.testing.assert_close(
        _backend_moe(backend, x, logits, w1, b1, w2, b2, k, routing = ours), _reference_moe(x, logits, w1, b1, w2, b2, k),
        rtol = 1e-5, atol = 1e-5,
    )


def test_old_generation_calls_are_passed_through_unchanged(fresh):
    pkg = _install(fresh, "triton_kernels", tkc.API_OGS_ROUTING)
    backend = tkc.get_triton_kernels_backend()
    x, logits, w1, b1, w2, b2 = _moe_inputs()
    rd, gather, scatter = backend.routing(logits, 2)
    pc = backend.precision("scale")
    assert pc.weight_scale == "scale" and pc.flex_ctx == sys.modules["triton_kernels.matmul_ogs"].FlexCtx(
        rhs_data = sys.modules["triton_kernels.matmul_ogs"].InFlexData())
    act = backend.swiglu_activation(1.702, 7.0)
    assert act.reduction_n == 2 and act.fn_args == (1.702, 7.0) and act.specs.fn == "swiglu_fn"
    backend.gate_up(x, w1, b1, rd, gather, pc, fused_activation = act)
    backend.down(torch.randn(x.shape[0] * 2, 6), w2, b2, rd, scatter, pc, gammas = rd.gate_scal)
    assert [(c[0], c[1], c[2], c[3]) for c in pkg.calls] == [
        ("matmul_ogs", gather, None, pc), ("matmul_ogs", None, scatter, pc),
    ]


def test_new_generation_translates_indices_and_precision(fresh):
    pkg = _install(fresh, "triton_kernels", tkc.API_MATMUL_RAGGED)
    backend = tkc.get_triton_kernels_backend()
    x, logits, w1, b1, w2, b2 = _moe_inputs()
    k = 2
    rd, gather, scatter = backend.routing(logits, k)
    assert isinstance(rd, tkc.RoutingData) and rd.n_expts_act == k
    pc = backend.precision("scale")
    assert pc.b_mx_scale == "scale" and pc.b_microblock_size == 32
    assert backend.weight_scale(pc) == "scale"
    act = backend.swiglu_activation(1.702, 7.0)
    assert act.specs.reduction_n == 2 and act.fn_args == (1.702, 7.0)
    h = backend.gate_up(x, w1, b1, rd, gather, pc, fused_activation = act)
    backend.down(h, w2, b2, rd, scatter, pc, gammas = rd.gate_scal)
    (_, g, _, _), (_, _, s, pc_down) = pkg.calls
    torch.testing.assert_close(g, gather.src_indx // k)
    torch.testing.assert_close(s, scatter.dst_indx)
    assert pc_down is pc and pc.out_dtype is None


def test_new_generation_down_rows_stay_in_output_dtype(fresh):
    # fp32 matmul rows would double every down projection's (tokens * k, hidden) buffer.
    pkg = _install(fresh, "triton_kernels", tkc.API_MATMUL_RAGGED)
    backend = tkc.get_triton_kernels_backend()
    x, logits, w1, b1, w2, b2 = _moe_inputs()
    k = 2
    rd, gather, scatter = backend.routing(logits, k)
    h = torch.randn(x.shape[0] * k, w2.shape[1]).to(torch.bfloat16)
    out = backend.down(h, w2, b2, rd, scatter, None, gammas = rd.gate_scal)
    (_, _, _, pc_down), = pkg.calls
    assert pc_down is None or pc_down.out_dtype in (None, torch.bfloat16)
    assert out.dtype == torch.bfloat16
    rows = sys.modules["triton_kernels.matmul"].matmul(
        h, w2, b2, a_ragged_metadata = rd.expt_data, scatter_indx = scatter.dst_indx, gammas = rd.gate_scal,
    )
    assert rows.dtype == torch.bfloat16
    torch.testing.assert_close(out, rows.view(-1, k, rows.shape[-1]).float().sum(1).to(torch.bfloat16), rtol = 0, atol = 0)


def test_dense_training_follows_the_copy_that_built_ragged_routing(fresh):
    # Routing came from the top-level copy: resolve it from the ragged metadata, not the SOURCE env var.
    _install(fresh, "triton_kernels", tkc.API_MATMUL_RAGGED)
    vendored = _install(fresh, "vllm.third_party.triton_kernels", tkc.API_OGS_ROUTING)
    fresh.setattr(tkc, "_import_vllm_vendored", lambda: tkc._probe(vendored))
    fresh.setenv("UNSLOTH_TRITON_KERNELS_SOURCE", "vllm")
    assert tkc.selected_triton_kernels()[0] == "vllm.third_party.triton_kernels"
    x, logits, *_ = _moe_inputs()
    routing_data, _, _ = tkc.triton_kernels_backend_for("triton_kernels").routing(logits, 2)
    assert isinstance(routing_data, tkc.RoutingData)
    module = torch.nn.Module()  # LoRA / training forward never sets _unsloth_tk_root
    assert go._ogs_root_for(torch.zeros(1), routing_data, module) == "triton_kernels"
    assert go._ogs_backend_for(torch.zeros(1), routing_data, module).api == tkc.API_MATMUL_RAGGED


def test_unswizzle_handles_both_layout_contracts(fresh):
    _install(fresh, "triton_kernels", tkc.API_MATMUL_RAGGED)
    backend = tkc.get_triton_kernels_backend()
    data = torch.arange(6)
    old_style = types.SimpleNamespace(
        storage = types.SimpleNamespace(data = data, layout = types.SimpleNamespace(unswizzle_data = lambda d: d + 1)),
    )
    torch.testing.assert_close(backend.unswizzle(old_style), data + 1)

    @dataclasses.dataclass(frozen = True)
    class StridedLayout:
        major_dim: int = -1

    seen = []

    def convert_layout(t, layout, **kw):
        seen.append(layout)
        return types.SimpleNamespace(storage = types.SimpleNamespace(data = t.storage.data * 2, layout = layout))

    fresh.setattr(backend.layout, "StridedLayout", StridedLayout, raising = False)
    fresh.setattr(backend.tensor, "convert_layout", convert_layout)
    new_style = types.SimpleNamespace(storage = types.SimpleNamespace(data = data, layout = object()))
    torch.testing.assert_close(backend.unswizzle(new_style), data * 2)
    # Packed along K (dim -2), the checkpoint's order, not the canonical last-dim packing.
    assert seen == [StridedLayout(major_dim = -2)]


@pytest.mark.parametrize("api", ALL_APIS)
def test_decode_custom_op_runs_on_every_generation(fresh, api):
    _install(fresh, "triton_kernels", api)
    go._ogs_modules.cache_clear()
    try:
        backend = go._ogs_kernels()
        assert backend is not None and backend.api == api
        inputs = _moe_inputs()
        torch.testing.assert_close(_backend_moe(backend, *inputs, 2), _reference_moe(*inputs, 2), rtol = 1e-5, atol = 1e-5)
    finally:
        go._ogs_modules.cache_clear()



# Verbatim from triton_kernels matmul_details/_matmul.py (release/3.8.x, main 4eda6cc).
MATMUL_HOPPER_ONLY_SOURCE = '''
@triton.jit
def _matmul(Y, W, WMxScale, SWIZZLE_MX_VALUE: tl.constexpr, is_w_microscaled: tl.constexpr):
    if is_w_microscaled:
        tl.static_assert(SWIZZLE_MX_VALUE == "HOPPER_VALUE" or SWIZZLE_MX_VALUE == "STRIDED", "Only Hopper swizzling is supported for values")
'''


def test_renamed_kernel_hopper_only_assert_is_detected(monkeypatch):
    go._source_rejects_blackwell_value_swizzle.cache_clear()
    assert go._source_rejects_blackwell_value_swizzle(MATMUL_HOPPER_ONLY_SOURCE) is True
    root = "fake_vendored_root.triton_kernels"
    for name in ("fake_vendored_root", root, root + ".matmul_details"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, root + ".matmul_ogs_details", None)
    leaf = types.ModuleType(root + ".matmul_details._matmul")
    leaf._matmul = types.SimpleNamespace(src = MATMUL_HOPPER_ONLY_SOURCE)
    monkeypatch.setitem(sys.modules, leaf.__name__, leaf)
    monkeypatch.setitem(sys.modules, "triton_kernels.matmul_ogs_details._matmul_ogs", None)
    monkeypatch.setitem(sys.modules, "triton_kernels.matmul_details._matmul", None)
    assert go._blackwell_value_swizzle_unsupported() is False
    assert go._blackwell_value_swizzle_unsupported(root) is True
    go._source_rejects_blackwell_value_swizzle.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason = "needs CUDA")
def test_real_routing_matches_a_dense_moe_across_experts():
    """The installed copy's own routing (topk metadata on newer APIs) with k = 4 of 8 experts."""
    backend = tkc.get_triton_kernels_backend()
    if backend is None:
        pytest.skip("no triton_kernels copy installed")
    from unsloth_zoo.mxfp4_dequant import mxfp4_dequantize_torch

    g = torch.Generator(device = "cuda").manual_seed(0)
    E, N, K, T, k = 8, 256, 512, 64, 4
    blocks = torch.randint(0, 256, (E, N, K // 32, 16), dtype = torch.uint8, device = "cuda", generator = g)
    scales = torch.randint(118, 128, (E, N, K // 32), dtype = torch.uint8, device = "cuda", generator = g)
    w, precision = tkc.mxfp4_ogs_weight(blocks, scales)
    dense = mxfp4_dequantize_torch(blocks, scales, dtype = torch.float32)
    x = torch.randn(T, K, device = "cuda", generator = g).to(torch.bfloat16)
    logits = torch.randn(T, E, device = "cuda", generator = g)
    rdata, gather, scatter = backend.routing(logits, k)
    out = backend.matmul_ogs(
        x, w, None, rdata, gather_indx = gather, scatter_indx = scatter,
        precision_config = precision, gammas = rdata.gate_scal,
    ).float()
    vals, idx = torch.topk(logits, k, dim = 1)
    ref = torch.einsum("tk,tkn->tn", torch.softmax(vals, dim = 1), torch.einsum("tc,tknc->tkn", x.float(), dense[idx]))
    assert ((out - ref).norm() / ref.norm()).item() < 1e-2



def test_routing_follows_the_copy_the_patched_forward_launches(fresh):
    _install(fresh, "triton_kernels", tkc.API_MATMUL_RAGGED)
    vendored = _install(fresh, "vllm.third_party.triton_kernels", tkc.API_OGS_ROUTING)
    fresh.setattr(tkc, "_import_vllm_vendored", lambda: tkc._probe(vendored))
    fresh.setenv("UNSLOTH_TRITON_KERNELS_SOURCE", "vllm")
    module = torch.nn.Module()
    module.__dict__["gate_up_proj"] = torch.zeros(1)
    assert go._triton_kernels_root(module) == "vllm.third_party.triton_kernels"
    module.__dict__["_unsloth_tk_root"] = "triton_kernels"
    assert go._triton_kernels_root(module) == "triton_kernels"


def test_dense_training_follows_the_copy_the_patched_forward_launches(fresh):
    _install(fresh, "triton_kernels", tkc.API_MATMUL_RAGGED)
    vendored = _install(fresh, "vllm.third_party.triton_kernels", tkc.API_OGS_ROUTING)
    fresh.setattr(tkc, "_import_vllm_vendored", lambda: tkc._probe(vendored))
    fresh.setenv("UNSLOTH_TRITON_KERNELS_SOURCE", "vllm")
    weight = torch.zeros(1)
    routing_data = type("RoutingData", (), {})()
    module = torch.nn.Module()
    assert go._ogs_root_for(weight, routing_data, module) == "vllm.third_party.triton_kernels"
    module.__dict__["_unsloth_tk_root"] = "triton_kernels"
    assert go._ogs_root_for(weight, routing_data, module) == "triton_kernels"
    assert go._ogs_backend_for(weight, routing_data, module).name == "triton_kernels"

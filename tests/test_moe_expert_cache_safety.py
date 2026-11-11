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
"""UNSLOTH_MOE_EXPERT_CACHE lifetime / staleness / capture / stream / OOM rules and the NF4 auto op's error and cap
policy, on CPU with a simulated device memory."""
import gc
import types

import pytest
import torch

from unsloth_zoo.temporary_patches import moe_expert_cache as ec

GiB = 2**30


def _qparam():
    """A frozen packed stand-in with a QuantState-like object (in-place edits bump real version counters)."""
    p = torch.nn.Parameter(torch.zeros(16, dtype = torch.uint8), requires_grad = False)
    p.quant_state = types.SimpleNamespace(
        absmax = torch.ones(4), code = torch.linspace(-1, 1, 16), offset = None, state2 = None,
        dtype = torch.bfloat16, blocksize = 64, quant_type = "nf4", shape = torch.Size((2, 4, 4)),
    )
    return p


def _stack(gib = 1):
    return torch.zeros(1, dtype = torch.bfloat16).expand(int(gib * GiB // 2))


@pytest.fixture
def mem(monkeypatch):
    state = {"total": 40 * GiB, "peak": 20 * GiB, "free_base": 40 * GiB}
    monkeypatch.setattr(ec, "_mem", lambda device: (state["free_base"] - ec._cached_bytes(), state["total"], state["peak"]))
    monkeypatch.setattr(ec, "_CHECK_EVERY_S", 0.0)
    monkeypatch.setenv("UNSLOTH_MOE_EXPERT_CACHE", "1")
    ec.clear_expert_cache()
    for k in ec.EXPERT_CACHE_STATS:
        ec.EXPERT_CACHE_STATS[k] = 0
    yield state
    ec.clear_expert_cache()


def _admit(p, w):
    for _ in range(ec.ADMIT_AFTER):
        ec.expert_cache_offer(p, w)
    assert ec.expert_cache_lookup(p) is w


@pytest.mark.parametrize("raw", ["0", "off", "OFF", "false", "no", "0.0", "-1", "abc", "", " 0 "])
def test_one_parser_every_gate_off(monkeypatch, raw):
    from unsloth_zoo.temporary_patches import moe_nf4_grouped as nf
    from unsloth_zoo.temporary_patches import moe_utils as mu
    monkeypatch.setenv("UNSLOTH_MOE_EXPERT_CACHE", raw)
    assert ec.parse_expert_cache_mode(raw) is None
    assert not ec.expert_cache_enabled()
    assert not mu._expert_cache_env_on()
    assert mu._expert_cache_module() is None
    monkeypatch.setenv("UNSLOTH_MOE_NF4_FUSED", "auto")
    monkeypatch.delenv("UNSLOTH_MOE_NF4_AUTO_IN_OP", raising = False)
    monkeypatch.delenv("UNSLOTH_MOE_SELECTIVE_DEQUANT", raising = False)
    monkeypatch.delenv("UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS", raising = False)
    assert nf.nf4_auto_in_op()


def test_parser_on_values():
    assert ec.parse_expert_cache_mode("1") == float("inf")
    assert ec.parse_expert_cache_mode("auto") == float("inf")
    assert ec.parse_expert_cache_mode("2.5") == 2.5 * GiB


def test_traced_gate_off_strings_keep_one_graph(monkeypatch):
    from unsloth_zoo.temporary_patches import moe_utils as mu
    monkeypatch.setenv("UNSLOTH_MOE_EXPERT_CACHE", "off")
    monkeypatch.setenv("UNSLOTH_MOE_EXPERT_CACHE_TRACED", "1")
    torch._dynamo.reset()

    @torch.compile(backend = "eager", fullgraph = True)
    def f(x):
        if mu._expert_cache_env_on():
            return x + 1
        return x * 2

    assert torch.equal(f(torch.ones(3)), torch.full((3,), 2.0))
    torch._dynamo.reset()


def test_dead_param_frees_its_stack(mem):
    p = _qparam()
    w = _stack()
    _admit(p, w)
    ref = __import__("weakref").ref(w)
    del w, p
    gc.collect()
    assert ec._cached_bytes() == 0 and not ec._CACHE and not ec._REFS
    assert ref() is None and ec.EXPERT_CACHE_STATS["bytes"] == 0


@pytest.mark.parametrize("edit", ["copy_", "load_state_dict", "absmax_inplace", "quant_state_swap", "data_swap"])
def test_in_place_change_invalidates(mem, edit):
    p = _qparam()
    _admit(p, _stack())
    if edit == "copy_":
        with torch.no_grad():
            p.copy_(torch.full_like(p, 7))   # writes through .data bypass version tracking: clear_expert_cache()
    elif edit == "load_state_dict":
        m = torch.nn.Module()
        m.w = p
        m.load_state_dict({"w": torch.full((16,), 3, dtype = torch.uint8)})
    elif edit == "absmax_inplace":
        p.quant_state.absmax.mul_(2)
    elif edit == "quant_state_swap":
        p.quant_state = types.SimpleNamespace(**vars(p.quant_state))
    else:
        p.data = torch.zeros(16, dtype = torch.uint8)
    assert ec.expert_cache_lookup(p) is None
    assert ec._cached_bytes() == 0


def test_no_lookup_while_capturing(mem, monkeypatch):
    p = _qparam()
    w = _stack()
    _admit(p, w)
    monkeypatch.setattr(ec, "_capturing", lambda: True)
    assert ec.expert_cache_lookup(p) is None
    monkeypatch.setattr(ec, "_capturing", lambda: False)
    assert ec.expert_cache_lookup(p) is w


def test_only_the_producing_stream_hits(mem, monkeypatch):
    p = _qparam()
    w = _stack()
    monkeypatch.setattr(ec, "_stream_id", lambda t: 11)
    _admit(p, w)
    monkeypatch.setattr(ec, "_stream_id", lambda t: 22)
    assert ec.expert_cache_lookup(p) is None
    monkeypatch.setattr(ec, "_stream_id", lambda t: 11)
    assert ec.expert_cache_lookup(p) is w


def test_inference_tensor_not_admitted(mem):
    p = _qparam()
    with torch.inference_mode():
        w = torch.zeros(4, dtype = torch.bfloat16)
    for _ in range(ec.ADMIT_AFTER + 1):
        ec.expert_cache_offer(p, w)
    assert ec.expert_cache_lookup(p) is None


def test_dtype_change_replaces_entry(mem):
    p = _qparam()
    _admit(p, _stack())
    w16 = torch.zeros(4, dtype = torch.float16)
    ec.expert_cache_offer(p, w16)
    assert ec.expert_cache_lookup(p) is w16


def test_offer_does_not_count_the_offered_stack_twice(mem):
    # free is measured with the decoded stack already allocated: 2.5 GiB free >= 2 GiB margin admits a 1 GiB stack.
    p = _qparam()
    w = _stack(1)
    mem["free_base"] = int(2.5 * GiB)
    for _ in range(ec.ADMIT_AFTER):
        ec.expert_cache_offer(p, w)
    assert ec.EXPERT_CACHE_STATS["admits"] == 1 and ec.EXPERT_CACHE_STATS["declined"] == 0


def test_oom_clear_keeps_the_uncached_baseline(mem):
    p = _qparam()
    _admit(p, _stack(10))
    mem["peak"] = 20 * GiB + 10 * GiB    # the allocator peak now includes the cache
    ec.clear_expert_cache(oom = True)
    q = _qparam()
    w = _stack(10)
    for _ in range(ec.ADMIT_AFTER):
        ec.expert_cache_offer(q, w)
    assert ec.expert_cache_lookup(q) is w   # budget 40 - 20 - 2 = 18 GiB, not 40 - 30 - 2 = 8


def test_oom_retry_clears_then_retries(mem):
    p = _qparam()
    _admit(p, _stack())
    calls = []

    def fn():
        calls.append(ec._cached_bytes())
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return "ok"

    assert ec.with_expert_cache_oom_retry(fn) == "ok"
    assert calls[0] > 0 and calls[1] == 0 and ec.EXPERT_CACHE_STATS["oom_clears"] == 1

    def always():
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        ec.with_expert_cache_oom_retry(always)    # nothing cached: no retry


def test_selective_decode_and_cast_recover_from_oom(mem, monkeypatch):
    from unsloth_zoo.temporary_patches import moe_utils as mu
    p = _qparam()
    _admit(p, _stack())
    n = {"sel": 0, "full": 0}

    def sel_once(param, experts, target_dtype = None):
        n["sel"] += 1
        if n["sel"] == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return torch.zeros(1)

    monkeypatch.setattr(mu, "_dequantize_bnb4bit_experts_once", sel_once)
    assert mu._dequantize_bnb4bit_experts(p, slice(0, 1)) is not None
    assert n["sel"] == 2 and ec.EXPERT_CACHE_STATS["oom_clears"] == 1

    q = _qparam()
    _admit(q, _stack())

    class OOMOnCast(torch.Tensor):
        pass

    def full(param):
        n["full"] += 1
        t = torch.zeros(4, dtype = torch.bfloat16)
        if n["full"] == 1:
            t = t.as_subclass(OOMOnCast)
        return t

    real_to = torch.Tensor.to

    def to(self, *a, **k):
        if type(self) is OOMOnCast:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return real_to(self, *a, **k)

    monkeypatch.setattr(OOMOnCast, "to", to)
    monkeypatch.setattr(mu, "_dequantize_bnb4bit_full", full)
    out = mu._get_bnb4bit_base_weight(q, torch.float16)
    assert out.dtype == torch.float16 and n["full"] == 2 and ec.EXPERT_CACHE_STATS["oom_clears"] == 2


def test_fused_probe_checks_dtype(mem, monkeypatch):
    from unsloth_zoo.temporary_patches import moe_utils as mu
    p = _qparam()
    _admit(p, torch.zeros(4, dtype = torch.bfloat16))
    hits = ec.EXPERT_CACHE_STATS["hits"]
    cache = mu._expert_cache_module()
    assert mu._expert_cache_get(cache, p, torch.float16, count = False) is None
    assert mu._expert_cache_get(cache, p, torch.bfloat16, count = False) is not None
    assert ec.EXPERT_CACHE_STATS["hits"] == hits


# ---- NF4 auto op policy ------------------------------------------------------------------------------------------

def _nf():
    pytest.importorskip("triton")
    from unsloth_zoo.temporary_patches import moe_nf4_grouped as nf
    return nf


@pytest.mark.parametrize("cap", [0, -1])
def test_nonpositive_cap_matches_python_choice(monkeypatch, cap):
    nf = _nf()
    route = []
    monkeypatch.setattr(nf, "_dequant_grouped_mm", lambda *a, **k: route.append("dequant"))
    monkeypatch.setattr(nf, "_launch", lambda *a, **k: route.append("fused"))
    a = torch.zeros(8, 4)
    ends = torch.tensor([4, 8], dtype = torch.int32)
    nf._auto_launch(a, *([None] * 6), ends, 4, 4, 4, 1, 64, 256, False, torch.bfloat16, cap)
    assert route == ["dequant"]
    monkeypatch.setenv("UNSLOTH_MOE_NF4_AUTO_MAX_ROWS", str(cap))
    assert not nf.nf4_auto_fused(8, 2)


def _flow_errors():
    from torch.utils import checkpoint
    errs = [torch.cuda.OutOfMemoryError("CUDA out of memory")]
    if hasattr(checkpoint, "_StopRecomputationError"):
        errs.append(checkpoint._StopRecomputationError())
    return errs


@pytest.mark.parametrize("which", [0, 1])
def test_fused_launch_propagates_oom_and_checkpoint_errors(monkeypatch, which):
    nf = _nf()
    errs = _flow_errors()
    if which >= len(errs):
        pytest.skip("no _StopRecomputationError in this torch")
    err = errs[which]

    def boom(*a, **k):
        raise err

    monkeypatch.setattr(nf, "_run_kernel", boom)
    monkeypatch.setattr(nf, "FAILED_SHAPES", {})
    launches = nf.NF4_FUSED_STATS["launches"]
    a = torch.zeros(4, 8)
    ends = torch.tensor([2, 4], dtype = torch.int32)
    with pytest.raises(type(err)):
        nf._launch(a, None, None, None, None, None, None, ends, 8, 6, 6, 1, 64, 256, False, torch.bfloat16)
    assert nf.FAILED_SHAPES == {} and nf.NF4_FUSED_STATS["launches"] == launches


@pytest.mark.parametrize("which", [0, 1])
def test_auto_decoder_propagates_oom_and_checkpoint_errors(monkeypatch, which):
    nf = _nf()
    pytest.importorskip("bitsandbytes")
    from unsloth_zoo.temporary_patches import moe_triton_kernels as mk
    errs = _flow_errors()
    if which >= len(errs):
        pytest.skip("no _StopRecomputationError in this torch")
    err = errs[which]

    def boom(*a, **k):
        raise err

    torch_fallback = []
    monkeypatch.setattr(mk, "nf4_dequant_triton", boom)
    monkeypatch.setattr(nf, "_decode_torch", lambda *a, **k: torch_fallback.append(1))
    absmax = torch.ones(2)
    with pytest.raises(type(err)):
        nf._dequant_flat(torch.zeros(64, dtype = torch.uint8), absmax, absmax, absmax, absmax,
                         torch.linspace(-1, 1, 16), 128, 64, 256, False, torch.bfloat16)
    assert not torch_fallback


@pytest.mark.parametrize("env", [("UNSLOTH_MOE_EXPERT_CACHE", "1"), ("UNSLOTH_MOE_SELECTIVE_DEQUANT", "1"),
                                 ("UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS", "2")])
def test_in_op_auto_yields_to_explicit_dequant_policy(monkeypatch, env):
    nf = _nf()
    for k in ("UNSLOTH_MOE_EXPERT_CACHE", "UNSLOTH_MOE_SELECTIVE_DEQUANT", "UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS",
              "UNSLOTH_MOE_NF4_AUTO_IN_OP"):
        monkeypatch.delenv(k, raising = False)
    assert nf.nf4_auto_in_op()
    monkeypatch.setenv(*env)
    assert not nf.nf4_auto_in_op()


def test_auto_ops_are_cudagraph_unsafe():
    nf = _nf()
    if nf._AUTO_OP is None or not hasattr(torch.Tag, "cudagraph_unsafe"):
        pytest.skip("custom_op tags unavailable")
    for op in (torch.ops.unsloth_zoo.nf4_grouped_mm_auto.default, torch.ops.unsloth_zoo.nf4_grouped_mm_auto_dx.default):
        assert torch.Tag.cudagraph_unsafe in op.tags


def test_gpt_oss_decode_oom_keeps_kernels(monkeypatch):
    god = pytest.importorskip("unsloth_zoo.gpt_oss_decode_fused")
    monkeypatch.setattr(god, "_ENABLED", True)
    assert god._disable(torch.cuda.OutOfMemoryError("CUDA out of memory")) is None
    assert god._ENABLED
    god._disable(RuntimeError("no compiler"))
    assert not god._ENABLED

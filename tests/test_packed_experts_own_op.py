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
"""Packed expert stacks of a format this module does not know (Unsloth's GGUF stacks) trace through torch.compile
without a graph break, and never read expert offsets on the host inside a CUDA graph capture or a decode step planned
for one. CPU, plus CUDA graph captures where a GPU is present."""

import pytest
import torch

import unsloth_zoo.mxfp4_stacked_experts as mse
import unsloth_zoo.packed_experts as pe
from unsloth_zoo.decode_graph_state import graph_decode_step, graph_generate
from unsloth_zoo.mxfp4_stacked_experts import _grouped_mm

_CUDA = torch.cuda.is_available() and torch.version.hip is None
# Dynamo before 2.10 (checked: 2.6) traces a Parameter subclass held outside a module as a plain object, so the call
# graph-breaks there as it did before the op (same results); 2.10, 2.11 and 2.14 trace it whole.
_TRACES_PARAM_SUBCLASS = tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 10)
needs_tracing = pytest.mark.skipif(not _TRACES_PARAM_SUBCLASS, reason = "Dynamo before torch 2.10")


class _ToyExpertParam(pe._PackedExpertParam):
    """int8 codes (E, N, K) times one fp32 scale per row (`packed_scales`, (E, N)), plus two plain constructor
    arguments of its own, as a GGUF stack has (ggml type, column count)."""

    packed_format = "toy_own_op"

    def __new__(cls, data, requires_grad = False, toy_shift = 0, toy_name = "x", **kwargs):
        param = super().__new__(cls, data, requires_grad, **kwargs)
        param.toy_shift = int(toy_shift)
        param.toy_name = str(toy_name)
        return param

    def _init_kwargs(self):
        kwargs = super()._init_kwargs()
        kwargs.update(toy_shift = self.toy_shift, toy_name = self.toy_name)
        return kwargs

    def packed_dims(self):
        return tuple(self.data.shape)

    def _decode(self, codes, scales, glob, dtype, transpose, token_counts, out = None):
        dense = ((codes.view(torch.int8).to(torch.float32) + self.toy_shift) * scales[..., None]).to(dtype)
        return dense.transpose(-2, -1).contiguous() if transpose else dense


class _ToyOwnGemm(_ToyExpertParam):
    """Brings its own grouped GEMM over the dequantized stack (device offsets only)."""

    def packed_grouped_mm(self, x, counts, transpose_b):
        dense = self.dequantize(x.dtype, transpose = False)
        weight = dense.transpose(-1, -2) if transpose_b else dense
        return _grouped_mm(x, weight, torch.cumsum(counts, 0, dtype = torch.int32))


def _toy(cls = _ToyExpertParam, E = 4, N = 32, K = 64, device = "cpu", shift = 3):
    g = torch.Generator().manual_seed(0)
    codes = torch.randint(-127, 128, (E, N, K), dtype = torch.int8, generator = g).view(torch.uint8)
    scales = torch.rand((E, N), generator = g) * 0.01
    return cls(codes.to(device), packed_scales = scales.to(device), toy_shift = shift, toy_name = "abc")


def _reference(x, param, counts, transpose_b):
    dense = param.dequantize(x.dtype, transpose = False)
    weight = dense.transpose(-1, -2) if transpose_b else dense
    return _grouped_mm(x, weight, torch.cumsum(counts, 0, dtype = torch.int32))


def test_own_op_flag_is_published():
    assert pe.PACKED_EXPERT_OWN_GROUPED_OP is True
    assert pe._register_own_op() is not None


@needs_tracing
@pytest.mark.parametrize("cls", [_ToyExpertParam, _ToyOwnGemm])
def test_own_format_traces_without_a_graph_break(cls):
    param = _toy(cls)
    counts = torch.tensor([2, 3, 0, 4])

    def f(x, counts):
        return pe.packed_expert_grouped_linear(x, param, counts, True) * 2

    torch._dynamo.reset()
    explain = torch._dynamo.explain(f)(torch.randn(9, 64), counts)
    assert explain.graph_break_count == 0, [str(b.reason).splitlines()[0] for b in explain.break_reasons]
    assert explain.graph_count == 1


@needs_tracing
@pytest.mark.parametrize("transpose_b", [True, False])
@pytest.mark.parametrize("cls", [_ToyExpertParam, _ToyOwnGemm])
def test_own_format_fullgraph_forward_and_dx_equal_eager(cls, transpose_b):
    param = _toy(cls)
    counts = torch.tensor([2, 3, 0, 4])
    in_dim = 64 if transpose_b else 32

    def f(x, counts):
        return pe.packed_expert_grouped_linear(x, param, counts, transpose_b)

    torch._dynamo.reset()
    compiled = torch.compile(f, backend = "aot_eager", fullgraph = True)
    x = torch.randn(9, in_dim, requires_grad = True)
    x2 = x.detach().clone().requires_grad_(True)
    out, ref = compiled(x, counts), _reference(x2, param, counts, transpose_b)
    assert torch.equal(out, ref)
    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    assert torch.equal(x.grad, x2.grad)
    assert param.grad is None


def test_op_rebuilds_the_stack_with_its_own_arguments():
    param = _toy(shift = 5)
    meta = pe._own_op_meta(param)
    assert meta is not None and "toy_shift=i5" in meta and "toy_name=sabc" in meta
    scales, _ = param._companions()
    rebuilt = pe._own_rebuild(type(param)._unsloth_own_key, param.data, scales, scales.new_empty((0,)), meta)
    assert type(rebuilt) is type(param)
    assert rebuilt.toy_shift == 5 and rebuilt.toy_name == "abc" and rebuilt.packed_global is None
    assert rebuilt.packed_dtype == param.packed_dtype and rebuilt.packed_transposed == param.packed_transposed
    assert torch.equal(rebuilt.dequantize(torch.float32), param.dequantize(torch.float32))


def test_classes_of_one_name_keep_their_own_keys():
    def make(shift):
        class _Factory(_ToyExpertParam):
            def _decode(self, codes, scales, glob, dtype, transpose, token_counts, out = None):
                dense = super()._decode(codes, scales, glob, dtype, transpose, token_counts)
                return dense + shift
        return _Factory

    a, b = make(1.0), make(2.0)
    assert a._unsloth_own_key != b._unsloth_own_key
    assert pe._OWN_CLASSES[a._unsloth_own_key] is a and pe._OWN_CLASSES[b._unsloth_own_key] is b
    counts = torch.tensor([2, 3, 0, 4])
    x = torch.randn(9, 64)
    for cls in (a, b):
        param = _toy(cls)
        assert torch.equal(pe.packed_expert_grouped_linear(x, param, counts, True), _reference(x, param, counts, True))


def test_unencodable_arguments_keep_the_eager_path():
    class _Odd(_ToyExpertParam):
        def _init_kwargs(self):
            kwargs = super()._init_kwargs()
            kwargs["toy_extra"] = (1, 2)
            return kwargs

    param = _toy(_Odd)
    assert pe._own_op_meta(param) is None
    counts = torch.tensor([2, 3, 0, 4])
    x = torch.randn(9, 64)
    assert torch.equal(pe.packed_expert_grouped_linear(x, param, counts, True), _reference(x, param, counts, True))


def _no_tolist(monkeypatch):
    def refuse(self, *args, **kwargs):
        raise AssertionError("expert offsets read on the host")
    monkeypatch.setattr(torch.Tensor, "tolist", refuse)


@pytest.mark.skipif(not hasattr(torch, "_grouped_mm"), reason = "torch without _grouped_mm")
@pytest.mark.parametrize("transpose_b", [True, False])
def test_multi_chunk_in_a_graph_decode_step_reads_no_host_offsets(monkeypatch, transpose_b):
    # The grouped GEMM itself on device offsets (on CPU the zoo's falls back to a per-expert loop with host bounds).
    monkeypatch.setattr(mse, "_grouped_mm", lambda inputs, weight, offsets: torch._grouped_mm(inputs, weight, offs = offsets))
    param = _toy()
    counts = torch.tensor([2, 3, 0, 4])
    x = torch.randn(9, 64 if transpose_b else 32)
    one = pe.packed_expert_grouped_linear(x, param, counts, transpose_b)
    monkeypatch.setattr(mse, "_chunk_bytes", lambda: 32 * 64 * 4)  # one expert per chunk
    host = pe.packed_expert_grouped_linear(x, param, counts, transpose_b)
    _no_tolist(monkeypatch)
    with graph_generate(), graph_decode_step():
        device = pe.packed_expert_grouped_linear(x, param, counts, transpose_b)
    torch.testing.assert_close(device, one)
    torch.testing.assert_close(device, host)


def test_gguf_grouped_fallback_in_a_graph_decode_step_reads_no_host_offsets(monkeypatch):
    from unsloth_zoo.gguf_kernels import GGMLType, gguf_grouped_mm
    from unsloth_zoo.gguf_kernels.testing import random_packed

    E, N, K = 3, 16, 256
    g = torch.Generator().manual_seed(0)
    packed = random_packed(GGMLType.Q8_0, E * N, K, special = False).view(E, N, -1)
    counts = torch.tensor([3, 0, 4], dtype = torch.int32)
    x = torch.randn((7, K), generator = g)
    eager = gguf_grouped_mm(x, packed, GGMLType.Q8_0, (E, N, K), counts)
    _no_tolist(monkeypatch)
    with graph_generate(), graph_decode_step():
        device = gguf_grouped_mm(x, packed, GGMLType.Q8_0, (E, N, K), counts)
    torch.testing.assert_close(device, eager)


@pytest.mark.skipif(not _CUDA, reason = "needs an NVIDIA CUDA device")
@pytest.mark.parametrize("multi_chunk", [False, True])
def test_own_format_captures_in_a_cuda_graph(monkeypatch, multi_chunk):
    param = _toy(_ToyExpertParam, device = "cuda")
    if multi_chunk:
        monkeypatch.setattr(mse, "_chunk_bytes", lambda: 32 * 64 * 2)  # one bf16 expert per chunk
    counts = torch.tensor([2, 3, 0, 4], device = "cuda")
    x = torch.randn(9, 64, device = "cuda", dtype = torch.bfloat16)
    eager = pe.packed_expert_grouped_linear(x, param, counts, True)
    static_x = x.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        pe.packed_expert_grouped_linear(static_x, param, counts, True)  # warm-up off the capture
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = pe.packed_expert_grouped_linear(static_x, param, counts, True)
    static_x.copy_(x * 2)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(static_out, pe.packed_expert_grouped_linear(x * 2, param, counts, True))
    static_x.copy_(x)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(static_out, eager)


@pytest.mark.skipif(not _CUDA, reason = "needs an NVIDIA CUDA device")
def test_gguf_grouped_float32_captures_in_a_cuda_graph():
    from unsloth_zoo.gguf_kernels import GGMLType, gguf_grouped_mm
    from unsloth_zoo.gguf_kernels.testing import random_packed

    E, N, K = 3, 16, 256
    g = torch.Generator().manual_seed(0)
    packed = random_packed(GGMLType.Q8_0, E * N, K, special = False).view(E, N, -1).cuda()
    counts = torch.tensor([3, 0, 4], dtype = torch.int32, device = "cuda")
    x = torch.randn((7, K), generator = g).cuda()
    eager = gguf_grouped_mm(x, packed, GGMLType.Q8_0, (E, N, K), counts)
    static_x = x.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        gguf_grouped_mm(static_x, packed, GGMLType.Q8_0, (E, N, K), counts)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = gguf_grouped_mm(static_x, packed, GGMLType.Q8_0, (E, N, K), counts)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(static_out, eager)


@needs_tracing
@pytest.mark.skipif(not _CUDA, reason = "needs an NVIDIA CUDA device")
def test_own_format_under_inductor_cuda_graphs():
    param = _toy(_ToyOwnGemm, device = "cuda")
    counts = torch.tensor([2, 3, 0, 4], device = "cuda")

    def f(x, counts):
        return pe.packed_expert_grouped_linear(x, param, counts, True)

    from torch._dynamo.utils import counters
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(f, mode = "reduce-overhead", fullgraph = True)
    x = torch.randn(9, 64, device = "cuda", dtype = torch.bfloat16)
    for _ in range(3):
        out = compiled(x, counts)
    torch.testing.assert_close(out, f(x, counts))
    assert counters["inductor"].get("cudagraph_skips", 0) == 0


def test_multi_chunk_in_a_graph_decode_step_with_the_loop_fallback():
    # On CPU the zoo's grouped GEMM is the per-expert loop, whose output stops at the last offset it is given.
    param = _toy()
    counts = torch.tensor([2, 3, 0, 4])
    x = torch.randn(9, 64)
    one = pe.packed_expert_grouped_linear(x, param, counts, True)
    import unsloth_zoo.mxfp4_stacked_experts as stacked
    original = stacked._chunk_bytes
    stacked._chunk_bytes = lambda: 32 * 64 * 4  # one expert per chunk
    try:
        with graph_generate(), graph_decode_step():
            device = pe.packed_expert_grouped_linear(x, param, counts, True)
    finally:
        stacked._chunk_bytes = original
    torch.testing.assert_close(device, one)


@needs_tracing
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_float32_inputs_under_autocast_match_eager_when_compiled(backend):
    param = _toy()
    counts = torch.tensor([2, 3, 0, 4])

    def f(x, counts):
        return pe.packed_expert_grouped_linear(x, param, counts, True)

    x = torch.randn(9, 64, requires_grad = True)
    x2 = x.detach().clone().requires_grad_(True)
    torch._dynamo.reset()
    compiled = torch.compile(f, backend = backend, fullgraph = True)
    with torch.autocast("cpu", dtype = torch.bfloat16):
        eager = f(x, counts)
        out = compiled(x2, counts)
    assert eager.dtype == out.dtype == torch.bfloat16
    assert torch.equal(out, eager)
    g = torch.randn_like(eager)
    eager.backward(g)
    out.backward(g)
    assert x.grad.dtype == x2.grad.dtype == torch.float32
    assert torch.equal(x.grad, x2.grad)


@pytest.mark.parametrize("multi_chunk", [False, True])
@pytest.mark.parametrize("transpose_b", [True, False])
def test_graph_decode_step_avoids_host_sync_grouped_gemm_fallbacks(monkeypatch, multi_chunk, transpose_b):
    # On CPU (as for float32 CUDA rows) the zoo's grouped GEMM is the per-expert loop, which reads its offsets on
    # the host: a capture must not reach it, with one chunk or several.
    param = _toy()
    counts = torch.tensor([2, 3, 0, 4])
    x = torch.randn(9, 64 if transpose_b else 32)
    eager = pe.packed_expert_grouped_linear(x, param, counts, transpose_b)
    if multi_chunk:
        monkeypatch.setattr(mse, "_chunk_bytes", lambda: 32 * 64 * 4)  # one expert per chunk
    _no_tolist(monkeypatch)
    with graph_generate(), graph_decode_step():
        device = pe.packed_expert_grouped_linear(x, param, counts, transpose_b)
    torch.testing.assert_close(device, eager)


class _FakeCuda:
    """Operand stand-in on a CUDA device (CPU host): only what the dispatch predicates read."""

    def __init__(self, shape, dtype = torch.bfloat16, stride = None):
        self.shape = torch.Size(shape)
        self.dtype = dtype
        self.device = torch.device("cuda", 0)
        self.is_cuda = True
        self._stride = stride or torch.empty(shape, device = "meta").stride()

    def dim(self):
        return len(self.shape)

    def stride(self):
        return self._stride

    def element_size(self):
        return torch.empty((), dtype = self.dtype).element_size()


def _dispatch(monkeypatch, wanted = False, triton_ok = True, supported = True, torch_loop = False):
    from unsloth_zoo.temporary_patches import moe_grouped_fp16, moe_utils

    monkeypatch.setattr(mse, "_CAPTURE_SAFE", {}, raising = False)
    monkeypatch.setattr(moe_utils, "_triton_grouped_mm_wanted", lambda inputs, weight: wanted)
    monkeypatch.setattr(moe_grouped_fp16, "triton_grouped_available", lambda device: triton_ok)
    monkeypatch.setattr(moe_utils, "_grouped_mm_supported_on", lambda device: supported)
    monkeypatch.setattr(moe_utils, "_torch_grouped_mm_takes_fallback", lambda *args: torch_loop)


# (Triton wanted, Triton self-check passed, torch._grouped_mm supported and not latched off, torch serves it with its
# own per-expert loop) -> capture-safe. Ampere bf16: torch's loop; a failed Triton self-check falls back to it.
@pytest.mark.parametrize("wanted,triton_ok,supported,torch_loop,safe", [
    (True, True, True, True, True),
    (True, False, True, True, False),
    (True, False, True, False, True),
    (True, False, False, False, False),
    (False, True, True, True, False),
    (False, True, True, False, True),
    (False, True, False, False, False),
])
def test_capture_safety_follows_the_backend_that_runs(monkeypatch, wanted, triton_ok, supported, torch_loop, safe):
    _dispatch(monkeypatch, wanted, triton_ok, supported, torch_loop)
    assert mse._grouped_mm_capture_safe(_FakeCuda((9, 64)), _FakeCuda((4, 64, 32))) is safe


def test_capture_safety_needs_16_byte_rows_for_torch_grouped_mm(monkeypatch):
    # An unaligned row makes torch._grouped_mm raise on its strides, and the zoo then takes the per-expert loop.
    _dispatch(monkeypatch)
    assert mse._grouped_mm_capture_safe(_FakeCuda((9, 64)), _FakeCuda((4, 64, 32)))
    assert not mse._grouped_mm_capture_safe(_FakeCuda((9, 60)), _FakeCuda((4, 60, 32)))
    assert not mse._grouped_mm_capture_safe(_FakeCuda((9, 64)), _FakeCuda((4, 64, 30)))


def test_capture_reuses_the_warm_up_verdict_and_probes_nothing(monkeypatch):
    from unsloth_zoo.temporary_patches import moe_utils

    _dispatch(monkeypatch)
    inputs, weight = _FakeCuda((9, 64)), _FakeCuda((4, 64, 32))
    assert mse._grouped_mm_capture_safe(inputs, weight)  # warm-up: verdict taken and kept

    def probe(*args):
        raise AssertionError("dispatch probed inside a capture")

    monkeypatch.setattr(moe_utils, "_grouped_mm_supported_on", probe)
    monkeypatch.setattr(moe_utils, "_triton_grouped_mm_wanted", probe)
    monkeypatch.setattr(mse, "_tracing_or_capturing", lambda device = None: True)
    assert mse._grouped_mm_capture_safe(inputs, weight)
    # Shapes the warm-up never saw: the masked path.
    assert not mse._grouped_mm_capture_safe(_FakeCuda((5, 64)), weight)


@needs_tracing
@pytest.mark.parametrize("dtype,autocast_dtype", [(torch.float16, torch.bfloat16), (torch.bfloat16, torch.float16)])
def test_low_precision_inputs_take_a_different_autocast_dtype(dtype, autocast_dtype):
    param = _toy()
    counts = torch.tensor([2, 3, 0, 4])

    def f(x, counts):
        return pe.packed_expert_grouped_linear(x, param, counts, True)

    x = (torch.randn(9, 64) * 0.5).to(dtype)
    torch._dynamo.reset()
    compiled = torch.compile(f, backend = "aot_eager", fullgraph = True)
    with torch.autocast("cpu", dtype = autocast_dtype):
        eager = f(x, counts)
        out = compiled(x, counts)
    assert eager.dtype == out.dtype == autocast_dtype
    assert torch.equal(out, eager)
    torch.testing.assert_close(eager, _reference(x.to(autocast_dtype), param, counts, True))


def test_graph_decode_warm_up_halves_a_chunk_that_does_not_fit(monkeypatch):
    # Eager warm-up of a graph decode step: a chunk over free memory is halved and retried, as on the host path.
    param = _toy()
    counts = torch.tensor([2, 3, 0, 4])
    x = torch.randn(9, 64)
    eager = pe.packed_expert_grouped_linear(x, param, counts, True)
    monkeypatch.setattr(mse, "_chunk_bytes", lambda: 2 * 32 * 64 * 4)  # two experts per chunk
    real = mse._dequantize_chunk
    sizes = []

    def dequantize(param, start, end, *args):
        sizes.append(end - start)
        if end - start > 1:
            raise torch.cuda.OutOfMemoryError("simulated: chunk does not fit")
        return real(param, start, end, *args)

    monkeypatch.setattr(mse, "_dequantize_chunk", dequantize)
    with graph_generate(), graph_decode_step():
        device = pe.packed_expert_grouped_linear(x, param, counts, True)
    torch.testing.assert_close(device, eager)
    assert sizes[0] == 2 and set(sizes[1:]) == {1}


def test_an_oom_inside_a_real_capture_still_raises(monkeypatch):
    param = _toy()
    counts = torch.tensor([2, 3, 0, 4])
    monkeypatch.setattr(mse, "_chunk_bytes", lambda: 2 * 32 * 64 * 4)
    monkeypatch.setattr(mse, "_tracing_or_capturing", lambda device = None: True)

    def dequantize(*args):
        raise torch.cuda.OutOfMemoryError("simulated")

    monkeypatch.setattr(mse, "_dequantize_chunk", dequantize)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        mse._chunked_grouped_mm_on_device(torch.randn(9, 64), param, counts, torch.float32, True, 2, 32)


@needs_tracing
@pytest.mark.skipif(not _CUDA, reason = "needs an NVIDIA CUDA device")
def test_compiled_first_call_moves_left_behind_companions_onto_the_stack():
    # Accelerate rebuilds a stack on the GPU but leaves its companions on the CPU; a compiled first call must move
    # them on the stack itself, not on the op's temporary rebuild (which would copy them on every call).
    param = _toy(device = "cuda")
    param.packed_scales = param.packed_scales.cpu()
    counts = torch.tensor([2, 3, 0, 4], device = "cuda")
    x = torch.randn(9, 64, device = "cuda", dtype = torch.bfloat16)

    def f(x, counts):
        return pe.packed_expert_grouped_linear(x, param, counts, True)

    torch._dynamo.reset()
    out = torch.compile(f, backend = "aot_eager")(x, counts)
    assert param.packed_scales.is_cuda
    moved = param.packed_scales
    torch.testing.assert_close(out, _reference(x, param, counts, True))
    torch.compile(f, backend = "aot_eager")(x, counts)
    assert param.packed_scales is moved


@pytest.mark.parametrize("transpose_b", [True, False])
def test_own_op_passes_opcheck(transpose_b):
    # Schema, fake (meta) impl, autograd registration and aliasing, on a plain uint8 operand as a trace passes it.
    if not hasattr(torch.library, "opcheck"):
        pytest.skip("torch.library.opcheck needs torch >= 2.4")
    param = _toy()
    scales, _ = param._companions()
    key, meta = type(param)._unsloth_own_key, pe._own_op_meta(param)
    E, N, K = param.shape
    x = torch.randn(9, K if transpose_b else N, requires_grad = True)
    args = (x, param.as_subclass(torch.Tensor), scales, scales.new_empty((0,)), torch.tensor([2, 3, 0, 4]),
            key, meta, N, K, transpose_b)
    torch.library.opcheck(torch.ops.unsloth_zoo.packed_experts_own_mm_v1.default, args)


def test_kill_switch_is_read_at_call_time(monkeypatch):
    param = _toy()
    counts = torch.tensor([2, 3, 0, 4])
    x = torch.randn(9, 64)
    seen = []
    original = pe._OwnPackedGroupedLinear.apply
    monkeypatch.setattr(pe._OwnPackedGroupedLinear, "apply", lambda *a: seen.append(1) or original(*a))
    ref = _reference(x, param, counts, True)
    monkeypatch.setenv("UNSLOTH_PACKED_EXPERT_OWN_OP", "0")
    assert torch.equal(pe.packed_expert_grouped_linear(x, param, counts, True), ref) and seen == [1]
    monkeypatch.setenv("UNSLOTH_PACKED_EXPERT_OWN_OP", "1")
    assert torch.equal(pe.packed_expert_grouped_linear(x, param, counts, True), ref) and seen == [1]

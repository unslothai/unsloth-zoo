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

"""Packed rows (`seq_idx`) through the Triton GatedDeltaNet causal conv: no tap crosses a sequence."""

import inspect
import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from unsloth_zoo.temporary_patches import gdn_causal_conv1d as gcc  # noqa: E402

_HAS_CUDA = torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason = "needs a CUDA GPU")


def _seq_idx_from_lengths(rows, device = "cpu"):
    """rows: per batch row, the document lengths (all rows the same total)."""
    out = []
    for lengths in rows:
        out.append(torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths)))
    return torch.stack(out).to(device = device, dtype = torch.int32)


def _inputs(B, D, T, W, dtype, bias, seed = 0, device = "cuda"):
    g = torch.Generator(device = device).manual_seed(seed)
    x = torch.randn(B, T, D, device = device, generator = g).to(dtype).transpose(1, 2)
    w = (torch.randn(D, W, device = device, generator = g) * 0.5).to(dtype)
    b = (torch.randn(D, device = device, generator = g) * 0.1).to(dtype) if bias else None
    dy = torch.randn(B, D, T, device = device, generator = g).to(dtype)
    return x, w, b, dy


def _run(fn, x, w, b, dy, act, seq_idx):
    x = x.detach().clone().requires_grad_(True)
    w = w.detach().clone().requires_grad_(True)
    b = b.detach().clone().requires_grad_(True) if b is not None else None
    y = fn(x, w, b, act, seq_idx)
    y.backward(dy)
    return y.detach(), x.grad, w.grad, (b.grad if b is not None else None)


def _triton(x, w, b, act, seq_idx):
    return gcc.triton_causal_conv1d(x, w, b, act, seq_idx)


def _reference(x, w, b, act, seq_idx):
    return gcc._causal_conv1d_reference_seq_idx(x, w, b, act, seq_idx)


def _per_document(x, w, b, act, seq_idx):
    """Each document run alone through the plain transformers torch conv, then concatenated."""
    rows = []
    for i in range(x.shape[0]):
        ids = seq_idx[i]
        starts = [0] + (torch.nonzero(ids[1:] != ids[:-1]).flatten() + 1).tolist() + [ids.numel()]
        rows.append(torch.cat([
            gcc.causal_conv1d_reference(x[i:i + 1, :, s:e], w, b, act) for s, e in zip(starts[:-1], starts[1:])
        ], dim = -1))
    return torch.cat(rows, dim = 0)


def _rel(a, ref):
    a, ref = a.double(), ref.double()
    return ((a - ref).norm() / ref.norm().clamp_min(1e-30)).item()


def _to64(t):
    return t.double() if t is not None else None


# (D, W, rows of document lengths): docs shorter than W, 1-token docs, trailing pad, tile edges, B > 1.
CASES = [
    (64, 4, [[5, 1, 7]]),
    (64, 4, [[1, 1, 1, 1, 1, 1, 1, 1]]),
    (96, 4, [[2, 3, 1, 40, 1, 3, 14]]),
    (130, 4, [[31, 2, 33, 1, 61], [64, 64]]),
    (70, 3, [[17, 1, 2, 80], [50, 50]]),
    (128, 2, [[1, 99, 1, 28], [3, 125, 1], [129]]),
    (256, 4, [[300, 1, 211, 512], [1, 1, 1021, 1]]),
]


@requires_cuda
@pytest.mark.parametrize("D,W,rows", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("act", ["silu", None])
def test_seq_idx_matches_reference_and_per_document(D, W, rows, dtype, bias, act):
    B, T = len(rows), sum(rows[0])
    x, w, b, dy = _inputs(B, D, T, W, dtype, bias)
    seq_idx = _seq_idx_from_lengths(rows, "cuda")
    ours = _run(_triton, x, w, b, dy, act, seq_idx)
    torch_ = _run(_reference, x, w, b, dy, act, seq_idx)
    per_doc = _run(_per_document, x, w, b, dy, act, seq_idx)
    oracle = _run(_per_document, _to64(x), _to64(w), _to64(b), _to64(dy), act, seq_idx)
    for name, o, t, p, r in zip(("y", "dx", "dw", "db"), ours, torch_, per_doc, oracle):
        if r is None:
            assert o is None
            continue
        e_ours, e_torch = _rel(o, r), _rel(t, r)
        if dtype == torch.float32:
            assert e_ours < 1e-5, (name, e_ours)
            # fp32 torch conv may use cuDNN TF32 (torch < 2.9).
            assert _rel(o, p) < 1e-5 + 2 * _rel(p, r), (name, _rel(o, p), _rel(p, r))
        else:
            floor = 4e-3 if dtype == torch.bfloat16 else 5e-4
            assert e_ours <= 2 * e_torch + floor, (name, e_ours, e_torch)
            assert _rel(o, p) <= 2 * max(e_torch, _rel(p, r)) + 2 * floor, (name, _rel(o, p))


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_seq_idx_forward_exact_per_document(dtype):
    rows = [[5, 1, 7, 2, 1, 48]]
    x, w, b, _ = _inputs(1, 192, 64, 4, dtype, True)
    seq_idx = _seq_idx_from_lengths(rows, "cuda")
    y = gcc.triton_causal_conv1d(x, w, b, "silu", seq_idx)
    starts = [0, 5, 6, 13, 15, 16, 64]
    alone = torch.cat([
        gcc.triton_causal_conv1d(x[:, :, s:e], w, b, "silu") for s, e in zip(starts[:-1], starts[1:])
    ], dim = -1)
    assert torch.equal(y, alone)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_single_sequence_seq_idx_is_bitwise_no_seq_idx(dtype):
    x, w, b, dy = _inputs(2, 320, 515, 4, dtype, True)
    zeros = torch.zeros(2, 515, dtype = torch.int32, device = "cuda")
    plain = _run(lambda x, w, b, a, s: gcc.triton_causal_conv1d(x, w, b, a), x, w, b, dy, "silu", None)
    masked = _run(_triton, x, w, b, dy, "silu", zeros)
    for p, m in zip(plain, masked):
        assert torch.equal(p, m)


@requires_cuda
def test_seq_idx_leak_is_visible_without_the_mask():
    # Flip check: without seq_idx the oracle must fail.
    rows = [[5, 1, 7, 3, 48]]
    x, w, b, dy = _inputs(1, 64, 64, 4, torch.float32, True)
    seq_idx = _seq_idx_from_lengths(rows, "cuda")
    per_doc = _run(_per_document, _to64(x), _to64(w), _to64(b), _to64(dy), "silu", seq_idx)
    leaky = _run(lambda x, w, b, a, s: gcc.triton_causal_conv1d(x, w, b, a), x, w, b, dy, "silu", None)
    ours = _run(_triton, x, w, b, dy, "silu", seq_idx)
    for name, l, o, p in zip(("y", "dx", "dw", "db"), leaky, ours, per_doc):
        if name == "db":
            continue
        assert _rel(l, p) > 1e-2, name
        assert _rel(o, p) < 1e-5, name


@requires_cuda
def test_seq_idx_dtypes_and_layouts():
    rows = [[3, 1, 12], [16]]
    x, w, b, _ = _inputs(2, 64, 16, 4, torch.bfloat16, True)
    ref = gcc._causal_conv1d_reference_seq_idx(x, w, b, "silu", _seq_idx_from_lengths(rows, "cuda"))
    for dtype in (torch.int32, torch.int64, torch.int16):
        seq_idx = _seq_idx_from_lengths(rows, "cuda").to(dtype)
        y = gcc.triton_causal_conv1d(x, w, b, "silu", seq_idx)
        assert _rel(y, ref) < 4e-3
    y = gcc.triton_causal_conv1d(x.contiguous(), w, b, "silu", _seq_idx_from_lengths(rows, "cuda"))
    assert _rel(y, ref) < 4e-3


def test_seq_idx_shape_mismatch_raises():
    x, w = torch.randn(1, 8, 10), torch.randn(8, 4)
    with pytest.raises(ValueError, match = "seq_idx"):
        gcc._legacy_causal_conv1d_fn(x = x, weight = w, seq_idx = torch.zeros(1, 9, dtype = torch.int32))
    with pytest.raises(ValueError, match = "seq_idx"):
        gcc._legacy_causal_conv1d_fn(x = x, weight = w, seq_idx = torch.zeros(1, 10))
    for dtype in (torch.bool, torch.complex64):
        with pytest.raises(ValueError, match = "seq_idx"):
            gcc._legacy_causal_conv1d_fn(x = x, weight = w, seq_idx = torch.zeros(1, 10, dtype = dtype))
    # Optional inputs default like the public entry points.
    torch.testing.assert_close(gcc._causal_conv1d_any(x, w), gcc.causal_conv1d_reference(x, w))


def test_dispatch_signatures_name_seq_idx():
    assert "seq_idx" in inspect.signature(gcc._legacy_causal_conv1d_fn).parameters

    def causal_conv1d_fn(hidden_states, weight, bias = None, activation = None, **kwargs):
        return gcc.causal_conv1d_reference(hidden_states, weight, bias, activation)

    dispatch = gcc._make_hub_dispatch(causal_conv1d_fn)
    assert gcc._is_marked(dispatch)
    assert dispatch.__wrapped__ is causal_conv1d_fn
    assert "seq_idx" in inspect.signature(dispatch).parameters
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 8, 10, generator = g, dtype = torch.float64)
    w = torch.randn(8, 4, generator = g, dtype = torch.float64)
    b = torch.randn(8, generator = g, dtype = torch.float64)
    seq_idx = _seq_idx_from_lengths([[3, 1, 6], [1, 9]])
    torch.testing.assert_close(dispatch(x, w, b, activation = "silu", seq_idx = seq_idx), _per_document(x, w, b, "silu", seq_idx))
    torch.testing.assert_close(dispatch(x, w, b, activation = "silu"), causal_conv1d_fn(x, w, b, "silu"))


def test_dispatch_signature_ignores_copied_signature():
    # functools.wraps copies an explicit __signature__; the dispatch must still name seq_idx.
    def causal_conv1d_fn(hidden_states, weight, bias = None, activation = None, **kwargs):
        return gcc.causal_conv1d_reference(hidden_states, weight, bias, activation)

    causal_conv1d_fn.__signature__ = inspect.signature(causal_conv1d_fn)
    dispatch = gcc._make_hub_dispatch(causal_conv1d_fn)
    assert "seq_idx" in inspect.signature(dispatch).parameters
    assert "seq_idx" not in inspect.signature(causal_conv1d_fn).parameters


def test_hub_wrapper_forwards_seq_idx(monkeypatch):
    hub_kernels = pytest.importorskip("transformers.integrations.hub_kernels")
    original = getattr(hub_kernels, "use_kernel_func_from_hub_with_fallback", None)
    if original is None:
        pytest.skip("transformers without use_kernel_func_from_hub_with_fallback")
    original = original.__wrapped__ if getattr(original, gcc._HUB_MARK, False) else original
    import transformers.integrations as integrations
    # Start unpatched (a CUDA host patched it at import), then patch as the product does.
    monkeypatch.setattr(hub_kernels, "use_kernel_func_from_hub_with_fallback", original)
    monkeypatch.setattr(integrations, "use_kernel_func_from_hub_with_fallback", original)
    monkeypatch.setattr(gcc, "_real_causal_conv1d_available", lambda: False)
    patched = gcc._patch_hub_decorator()
    seen = []
    any_conv = gcc._causal_conv1d_any

    def spy(*args, **kwargs):
        seen.append(kwargs.get("seq_idx", args[4] if len(args) > 4 else None))
        return any_conv(*args, **kwargs)

    monkeypatch.setattr(gcc, "_causal_conv1d_any", spy)

    def causal_conv1d_fn(hidden_states, weight, bias = None, activation = None, **kwargs):
        return gcc.causal_conv1d_reference(hidden_states, weight, bias, activation)

    causal_conv1d_fn.__module__ = "transformers.models.qwen3_5.modeling_qwen3_5"
    # transformers >= 5.20 raises on seq_idx in its own torch path without causal-conv1d.
    hub_fn = patched("causal_conv1d_fn", "causal_conv1d")(causal_conv1d_fn)
    assert gcc._is_marked(hub_fn)
    assert "seq_idx" in inspect.signature(hub_fn).parameters
    x, w = torch.randn(1, 8, 6, dtype = torch.float64), torch.randn(8, 4, dtype = torch.float64)
    seq_idx = _seq_idx_from_lengths([[2, 4]])
    out = hub_fn(x, w, None, activation = "silu", seq_idx = seq_idx, cu_seq_lens_q = None, position_ids = None)
    assert seen and seen[-1] is seq_idx
    torch.testing.assert_close(out, _per_document(x, w, None, "silu", seq_idx))
    torch.testing.assert_close(
        hub_fn(x, w, None, activation = "silu", seq_idx = seq_idx, max_length_q = 4),
        _per_document(x, w, None, "silu", seq_idx),
    )
    torch.testing.assert_close(hub_fn(x, w, None, activation = "silu"), causal_conv1d_fn(x, w, None, "silu"))



def _fake_kernels_decorator(layer_name):
    """What `kernels.use_kernel_forward_from_hub` does to a function: a layer instance whose
    class carries `kernel_layer_name`."""
    def decorator(fn):
        class Func(torch.nn.Module):
            def forward(self, *args, **kwargs):
                return fn(*args, **kwargs)

        Func.kernel_layer_name = layer_name
        return Func()
    return decorator


def _hub_conv(hidden_states, weight, bias = None, activation = None, **kwargs):
    return gcc.causal_conv1d_reference(hidden_states, weight, bias, activation)


_hub_conv.__module__ = "transformers.models.qwen3_5.modeling_qwen3_5"


def _check_hub_layer_routing(hub_fn):
    x, w = torch.randn(1, 8, 6, dtype = torch.float64), torch.randn(8, 4, dtype = torch.float64)
    seq_idx = _seq_idx_from_lengths([[2, 4]])
    torch.testing.assert_close(hub_fn(x, w, None, activation = "silu", seq_idx = seq_idx),
                               _per_document(x, w, None, "silu", seq_idx))
    torch.testing.assert_close(hub_fn(x, w, None, activation = "silu"), _hub_conv(x, w, None, "silu"))


def test_hub_layer_instance_keeps_its_kernel_metadata():
    # kernels' decorator returns a layer instance; use_kernelized_func checks `kernel_layer_name`
    # and kernelize reads it from type(fn), so the GDN routing must not replace it with a function.
    hub_fn = gcc._hub_gdn_function(_fake_kernels_decorator("causal_conv1d_fn"), _hub_conv)
    assert isinstance(hub_fn, torch.nn.Module)
    assert type(hub_fn).kernel_layer_name == "causal_conv1d_fn"
    assert gcc._is_marked(hub_fn)
    assert "seq_idx" in inspect.signature(hub_fn).parameters
    _check_hub_layer_routing(hub_fn)

    # kernelize swaps the instance forward for the hub kernel's; that must take effect.
    class Kernel(torch.nn.Module):
        def forward(self, x, *args, **kwargs):
            return x * 0 + 7

    from types import MethodType
    hub_fn.forward = MethodType(Kernel.forward, hub_fn)
    x, w = torch.randn(1, 8, 6), torch.randn(8, 4)
    assert torch.equal(hub_fn(x, w), torch.full_like(x, 7))


def test_hub_layer_passes_real_kernels_validation():
    kernels = pytest.importorskip("kernels")
    hub_fn = gcc._hub_gdn_function(kernels.use_kernel_forward_from_hub("causal_conv1d_fn"), _hub_conv)
    assert hasattr(hub_fn, "kernel_layer_name")

    @kernels.use_kernelized_func(hub_fn)
    class Layer(torch.nn.Module):
        pass

    assert Layer()._kernel_funcs
    _check_hub_layer_routing(hub_fn)


def _wide_ids(lengths, labels, dtype, device):
    return torch.repeat_interleave(
        torch.tensor(labels, dtype = torch.int64), torch.tensor(lengths),
    ).view(1, -1).to(device = device, dtype = dtype)


@requires_cuda
def test_seq_idx_wide_int64_ids_stay_distinct():
    # 0, 2**32, -2**32 and 2**33 + 1 collapse to {0, 1} in int32: taps must still stop at each boundary.
    lengths, labels = [3, 4, 2, 5, 6], [0, 2**32, -2**32, 2**33 + 1, 1]
    x, w, b, dy = _inputs(1, 64, sum(lengths), 4, torch.float32, True)
    seq_idx = _wide_ids(lengths, labels, torch.int64, "cuda")
    oracle = _run(_per_document, _to64(x), _to64(w), _to64(b), _to64(dy), "silu", seq_idx)
    for fn in (_triton, _reference, gcc._legacy_causal_conv1d_fn):
        for name, o, r in zip(("y", "dx", "dw", "db"), _run(fn, x, w, b, dy, "silu", seq_idx), oracle):
            assert _rel(o, r) < 1e-5, (fn, name, _rel(o, r))


@requires_cuda
def test_seq_idx_small_and_unsigned_dtypes_match_int32():
    lengths, labels = [3, 1, 6, 6], [0, 1, 2, 3]
    x, w, b, _ = _inputs(1, 64, sum(lengths), 4, torch.bfloat16, True)
    ref = gcc.triton_causal_conv1d(x, w, b, "silu", _wide_ids(lengths, labels, torch.int32, "cuda"))
    dtypes = [torch.int8, torch.uint8, torch.int16, torch.int64]
    dtypes += [getattr(torch, n) for n in ("uint16", "uint32", "uint64") if hasattr(torch, n)]
    for dtype in dtypes:
        seq_idx = _wide_ids(lengths, labels, dtype, "cuda")
        assert torch.equal(gcc.triton_causal_conv1d(x, w, b, "silu", seq_idx), ref), dtype
        y = gcc._causal_conv1d_reference_seq_idx(x, w, b, "silu", seq_idx)
        assert _rel(y, ref) < 4e-3, dtype


@requires_cuda
def test_seq_idx_on_cpu_with_cuda_activations(monkeypatch):
    rows = [[3, 1, 12], [16]]
    x, w, b, dy = _inputs(2, 64, 16, 4, torch.float32, True)
    seq_idx = _seq_idx_from_lengths(rows, "cuda")
    oracle = _run(_per_document, _to64(x), _to64(w), _to64(b), _to64(dy), "silu", seq_idx)
    assert gcc._check_seq_idx(x, seq_idx.cpu()).device == x.device
    before = gcc.GDN_CAUSAL_CONV1D_STATS["triton_seq_idx"]
    for fn in (_triton, _reference, gcc._legacy_causal_conv1d_fn):
        for name, o, r in zip(("y", "dx", "dw", "db"), _run(fn, x, w, b, dy, "silu", seq_idx.cpu()), oracle):
            assert _rel(o, r) < 1e-5, (fn, name, _rel(o, r))
    # Direct entry and dispatch both ran the kernel, not the fallback.
    assert gcc.GDN_CAUSAL_CONV1D_STATS["triton_seq_idx"] - before == 2
    monkeypatch.setenv(gcc._KILL_SWITCH, "1")
    y = gcc._legacy_causal_conv1d_fn(x, w, b, "silu", seq_idx = seq_idx.cpu())
    assert _rel(y, oracle[0]) < 1e-5


def _random_seq_idx(g, T, W, dtype):
    """Random document lengths (many shorter than W); IDs gapped, negative, huge, and reused
    once at least W - 1 tokens away (equal IDs inside one window would join the runs)."""
    pool = [-7, -1, 0, 3, 5, 11, 2**31 - 1, 2**32, 2**32 + 3, -2**40]
    pool = [v for v in pool if torch.iinfo(dtype).min <= v <= torch.iinfo(dtype).max]
    n_docs = int(torch.randint(1, min(12, T) + 1, (), generator = g))
    cuts = (torch.randperm(T - 1, generator = g)[:n_docs - 1] + 1).sort().values.tolist()
    ids = []
    for s, e in zip([0] + cuts, cuts + [T]):
        recent = set(ids[max(0, s - (W - 1)):])
        label = pool[int(torch.randint(len(pool), (), generator = g))]
        while label in recent:
            label = pool[int(torch.randint(len(pool), (), generator = g))]
        ids += [label] * (e - s)
    return ids


@requires_cuda
def test_seq_idx_randomized_against_per_document():
    g = torch.Generator().manual_seed(1234)
    for case in range(50):
        B, T, W = 1 + case % 3, int(torch.randint(8, 100, (), generator = g)), 2 + case % 7
        dtype = (torch.int64, torch.int32)[case % 2]
        x, w, b, dy = _inputs(B, 64, T, W, torch.float32, True, seed = case)
        rows = [_random_seq_idx(g, T, W, dtype) for _ in range(B)]
        # Strided (B, T) view of a wider buffer.
        storage = torch.full((B, 2 * T), 99, dtype = dtype, device = "cuda")
        seq_idx = storage[:, ::2]
        seq_idx.copy_(torch.tensor(rows, dtype = dtype))
        assert not seq_idx.is_contiguous()
        ours = _run(_triton, x, w, b, dy, "silu", seq_idx)
        oracle = _run(_per_document, _to64(x), _to64(w), _to64(b), _to64(dy), "silu", seq_idx)
        for name, o, r in zip(("y", "dx", "dw", "db"), ours, oracle):
            assert _rel(o, r) < 2e-5, (case, name, _rel(o, r), rows)

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
    # Channel-last storage, as the GatedDeltaNet passes it (transposed (B, T, D)).
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


# (D, W, rows of document lengths). Covers documents shorter than W, 1-token documents,
# a trailing pad segment (one long last "document"), tile-boundary crossings and B > 1.
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
            assert _rel(o, p) < 1e-5, (name, _rel(o, p))
        else:
            floor = 4e-3 if dtype == torch.bfloat16 else 5e-4
            assert e_ours <= 2 * e_torch + floor, (name, e_ours, e_torch)
            assert _rel(o, p) <= 2 * max(e_torch, _rel(p, r)) + 2 * floor, (name, _rel(o, p))


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_seq_idx_forward_exact_per_document(dtype):
    # The forward reads exactly the same values a lone document would: bitwise in fp32 / bf16.
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
    # Flip check: dropping seq_idx must fail the per-document oracle by a wide margin.
    rows = [[5, 1, 7, 3, 48]]
    x, w, b, dy = _inputs(1, 64, 64, 4, torch.float32, True)
    seq_idx = _seq_idx_from_lengths(rows, "cuda")
    per_doc = _run(_per_document, x, w, b, dy, "silu", seq_idx)
    leaky = _run(lambda x, w, b, a, s: gcc.triton_causal_conv1d(x, w, b, a), x, w, b, dy, "silu", None)
    ours = _run(_triton, x, w, b, dy, "silu", seq_idx)
    for name, l, o, p in zip(("y", "dx", "dw", "db"), leaky, ours, per_doc):
        if name == "db":
            continue  # the bias gradient does not depend on the taps
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
    # Channel-first storage too.
    y = gcc.triton_causal_conv1d(x.contiguous(), w, b, "silu", _seq_idx_from_lengths(rows, "cuda"))
    assert _rel(y, ref) < 4e-3


def test_seq_idx_shape_mismatch_raises():
    x, w = torch.randn(1, 8, 10), torch.randn(8, 4)
    with pytest.raises(ValueError, match = "seq_idx"):
        gcc._legacy_causal_conv1d_fn(x = x, weight = w, seq_idx = torch.zeros(1, 9, dtype = torch.int32))
    with pytest.raises(ValueError, match = "seq_idx"):
        gcc._legacy_causal_conv1d_fn(x = x, weight = w, seq_idx = torch.zeros(1, 10))


def test_dispatch_signatures_name_seq_idx():
    # Unsloth's hybrid packing gate and the transformers hub filter key on the named parameter.
    assert "seq_idx" in inspect.signature(gcc._legacy_causal_conv1d_fn).parameters

    def causal_conv1d_fn(hidden_states, weight, bias = None, activation = None, **kwargs):
        return gcc.causal_conv1d_reference(hidden_states, weight, bias, activation)

    dispatch = gcc._make_hub_dispatch(causal_conv1d_fn)
    assert gcc._is_marked(dispatch)
    assert dispatch.__wrapped__ is causal_conv1d_fn
    assert "seq_idx" in inspect.signature(dispatch).parameters
    # CPU: the torch path honours seq_idx and equals per-document runs.
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 8, 10, generator = g, dtype = torch.float64)
    w = torch.randn(8, 4, generator = g, dtype = torch.float64)
    b = torch.randn(8, generator = g, dtype = torch.float64)
    seq_idx = _seq_idx_from_lengths([[3, 1, 6], [1, 9]])
    torch.testing.assert_close(dispatch(x, w, b, activation = "silu", seq_idx = seq_idx), _per_document(x, w, b, "silu", seq_idx))
    torch.testing.assert_close(dispatch(x, w, b, activation = "silu"), causal_conv1d_fn(x, w, b, "silu"))


def test_hub_wrapper_forwards_seq_idx(monkeypatch):
    hub_kernels = pytest.importorskip("transformers.integrations.hub_kernels")
    original = getattr(hub_kernels, "use_kernel_func_from_hub_with_fallback", None)
    if original is None:
        pytest.skip("transformers without use_kernel_func_from_hub_with_fallback")
    original = original.__wrapped__ if getattr(original, gcc._HUB_MARK, False) else original
    monkeypatch.setattr(gcc, "_real_causal_conv1d_available", lambda: False)
    seen = []

    def causal_conv1d_fn(hidden_states, weight, bias = None, activation = None, **kwargs):
        return gcc.causal_conv1d_reference(hidden_states, weight, bias, activation)

    causal_conv1d_fn.__module__ = "transformers.models.qwen3_5.modeling_qwen3_5"
    dispatch = gcc._make_hub_dispatch(causal_conv1d_fn)

    def spy(*args, seq_idx = None, **kwargs):
        seen.append(seq_idx)
        return dispatch(*args, seq_idx = seq_idx, **kwargs)

    spy.__signature__ = inspect.signature(dispatch)
    hub_fn = original("causal_conv1d_fn", "causal_conv1d")(spy)
    x, w = torch.randn(1, 8, 6, dtype = torch.float64), torch.randn(8, 4, dtype = torch.float64)
    seq_idx = _seq_idx_from_lengths([[2, 4]])
    hub_fn(x, w, None, activation = "silu", seq_idx = seq_idx, cu_seq_lens_q = None, position_ids = None)
    assert seen and seen[-1] is seq_idx
    # The real dispatch through the hub wrapper matches per-document runs.
    hub_fn = original("causal_conv1d_fn", "causal_conv1d")(dispatch)
    torch.testing.assert_close(
        hub_fn(x, w, None, activation = "silu", seq_idx = seq_idx, max_length_q = 4),
        _per_document(x, w, None, "silu", seq_idx),
    )

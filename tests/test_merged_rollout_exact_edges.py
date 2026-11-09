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

"""Exact fold / unfold at the edges of fp16 / bf16: every one of the 65,536 bit patterns of W against deltas that are
signed zeros, subnormal, half-ulp ties, binade crossings, much larger than W, overflowing to inf, inf and NaN.

The proof runs the real kernel. On CUDA it runs in this process; without CUDA, test_cpu_interpreter_proof reruns this
file under TRITON_INTERPRET=1 on CPU tensors. Each delta is driven through A, B and scale: rank 1 with A = 1, so
s * B @ A = fp32(b * s) exactly and a torch emulation knows which elements cannot round-trip."""

import os
import subprocess
import sys

import pytest
import torch

from unsloth_zoo import merged_rollout_exact as ex

DTYPES = (torch.float16, torch.bfloat16)
# Exact in both dtypes: per-row multipliers of the scale.
MULTIPLIERS = (1.0, -1.0, 1.5, -1.5, 1.25, -0.75, 2.0**-3, -(2.0**-7))
INF, NAN = float("inf"), float("nan")
# fp32 deltas around each dtype's representation edges; every W binade sees ties, crossings and overflow somewhere.
# Signed zero deltas come from scale 0.0 and the negative multipliers (d = -1 * 0.0 = -0.0); a -0.0 scale itself is not
# a LoRA scaling and loses its sign as a Triton float argument.
SCALES = {
    torch.float16: (
        0.0, 2.0**-149, 2.0**-130, 2.0**-40, 2.0**-25, 2.0**-24, 2.0**-15, 2.0**-14, 2.0**-11, 2.0**-10,
        2.0**-5, 0.5, 1.0, 3.0, 1000.0, 32768.0, 65504.0, 1e6, INF, -INF, NAN,
    ),
    torch.bfloat16: (
        0.0, 2.0**-149, 2.0**-134, 2.0**-133, 2.0**-127, 2.0**-100, 2.0**-9, 2.0**-8, 2.0**-3, 1.0, 3.0,
        1e10, 1e30, 3.3e38, INF, -INF, NAN,
    ),
}


def _bits(t):
    return t.view(torch.int16)


def _all_patterns(dtype, rows, device):
    w = torch.arange(-32768, 32768, dtype = torch.int32).to(torch.int16).view(dtype)
    return w.repeat(rows, 1).to(device).contiguous()


def _emulate(W, d):
    """(folded W, which elements cannot round-trip): the kernel's arithmetic on the host, rank 1 with A = 1."""
    new = (W.float() + d).to(W.dtype)
    back = (new.float() - d).to(W.dtype)
    return new, _bits(back) != _bits(W)


def _ord(b):
    b = b.to(torch.int32)
    return torch.where(b < 0, -b - 32769, b)


def _emulate_store(W, d):
    """(code width per element, escapes): the store layout the kernel must produce, from the module docstring's rule."""
    new = (W.float() + d).to(W.dtype)
    back = (new.float() - d).to(W.dtype)
    mant, expm = (10, 31) if W.dtype == torch.float16 else (7, 255)
    nb, bb = _bits(new).to(torch.int32), _bits(back).to(torch.int32)
    r = ((nb >> mant) & expm).clamp(min = 1) - ((bb >> mant) & expm).clamp(min = 1)
    k = torch.where((r >= 1) & (r <= ex._MAX_GAP), r + 1, 0)
    c = _ord(_bits(W)) - _ord(bb)
    half = 1 << (k.clamp(min = 1) - 1)
    bad = _bits(back) != _bits(W)
    esc = bad & ((k == 0) | (c <= -half) | (c >= half))
    return k, esc


# Which code widths / escape kinds the edge cross reached (asserted once the cross has run).
COVERAGE = {}


def _check_cycle(W, A, B, scale, d = None):
    W0 = W.clone()
    t = ex.ExactFoldTarget(W.unsqueeze(0), A.unsqueeze(0), B.unsqueeze(0), scale)
    stats = ex.exact_fold([t])
    store = t.host_idx.to(torch.int64).cpu()
    vals = t.host_val.cpu()
    words = t.host_words.cpu().clone()
    folded = W.clone()
    ex.exact_unfold([t])
    assert torch.equal(_bits(W), _bits(W0)), "unfold is not bit exact"
    # Escapes keep original bits, once per element.
    assert store.numel() == torch.unique(store).numel() <= stats["bad"]
    assert torch.equal(_bits(vals), _bits(W0.reshape(-1).cpu()[store]))
    # The CPU interpreter has no bf16: it converts fp32 -> bf16 by truncation and flushes subnormals, so only the
    # restore is checked there (it stays exact under any deterministic rounding); CUDA and fp16 check every bit.
    if d is not None and (W.is_cuda or W.dtype == torch.float16):
        want, bad = _emulate(W0.cpu(), d.cpu())
        k, esc = _emulate_store(W0.cpu(), d.cpu())
        nan = torch.isnan(want) | torch.isnan(W0.cpu())
        kept = torch.zeros(W0.numel(), dtype = torch.bool)
        kept[store] = True
        kept = kept.view(W0.shape)
        # NaN payloads may be canonicalised by the device conversion; everything else is checked bit for bit.
        assert int(bad[~nan].sum()) <= stats["bad"] <= int(bad[~nan].sum()) + int(nan.sum())
        assert torch.equal(kept[~nan], esc[~nan]), "escapes are not exactly the out-of-code elements"
        assert torch.equal(_bits(folded.cpu())[~nan], _bits(want)[~nan]), "fold differs from fp32(W + d) rounded once"
        assert bool(kept[nan & torch.isnan(W0.cpu()) & ~torch.isnan(want)].all())
        # The bit stream holds every coded element: at least the code bits, at most one padding word per tile.
        n_tiles = (-(-W0.shape[0] // ex._BLOCK_N)) * (-(-W0.shape[1] // ex._BLOCK_K))
        bits = int(k[~nan].sum())
        assert bits <= 32 * words.numel() <= int(k.sum()) + 8 * int(nan.sum()) + 32 * n_tiles
        for width in torch.unique(k[~nan & bad & ~esc]).tolist():
            COVERAGE["code%d" % width] = True
        if bool((esc & (k > 0) & ~nan).any()): COVERAGE["escape_code"] = True
        if bool((esc & (k == 0) & ~nan).any()): COVERAGE["escape_no_code"] = True
        if bool((nan & kept).any()): COVERAGE["escape_nan"] = True
    return stats


def run_negative(device):
    """A corrupted bit stream, a corrupted escape and a changed adapter must each raise, never restore silently."""
    g = torch.Generator(device = device).manual_seed(3)
    for dtype in DTYPES:
        for corrupt in ("word", "escape", "adapter"):
            W = (torch.randn(64, 1024, device = device, generator = g) * 0.02).to(dtype)
            W[0, :64] = 2.0**-20   # tiny weights under a large delta: escapes (fp16 subnormal crossings, bf16 ties)
            A = torch.randn(1, 4, 1024, device = device, generator = g).to(dtype)
            B = torch.randn(1, 64, 4, device = device, generator = g).to(dtype)
            scale = 0.05 * 0.02 / (B[0].float() @ A[0].float()).abs().mean().item()
            t = ex.ExactFoldTarget(W.unsqueeze(0), A, B, scale)
            ex.exact_fold([t])
            if corrupt == "word":
                nz = (t.host_words != 0).nonzero()
                if not nz.numel() and device == "cpu" and dtype == torch.bfloat16:
                    ex.exact_unfold([t]); continue   # truncating interpreter bf16: every inexact element escapes
                assert nz.numel(), "no coded element to corrupt"
                t.host_words[nz[0, 0]] ^= 1 << 3
            elif corrupt == "escape":
                if t.host_val.numel() == 0:
                    ex.exact_unfold([t]); continue
                t.host_val.view(torch.int16)[0] ^= 1
            else:
                B.add_(0.5)
            try:
                ex.exact_unfold([t])
            except RuntimeError as error:
                assert "checksum mismatch" in str(error)
            else:
                raise AssertionError(f"corrupted {corrupt} restored without an error ({dtype})")


def run_edges(device, scales = None, dtypes = DTYPES, multipliers = MULTIPLIERS):
    """Every bit pattern x every delta of SCALES (x multipliers rows); returns {dtype: fix-up fraction per scale}."""
    out = {}
    rows = len(multipliers)
    table = SCALES if scales is None else scales
    for dtype in dtypes:
        fractions = []
        for s in table[dtype]:
            W = _all_patterns(dtype, rows, device)
            A = torch.ones(1, W.shape[1], dtype = dtype, device = device)
            B = torch.tensor(multipliers, dtype = dtype, device = device).view(rows, 1)
            d = (B.float() * s).expand(rows, W.shape[1])   # fp32(b * s): fma(b, 1, 0) * s in the kernel
            stats = _check_cycle(W, A, B, s, d)
            fractions.append((s, stats["bad"] / stats["elements"]))
        out[dtype] = fractions
    return out


def run_random(device, cycles = 30, rows = 256):
    """Trained-like adapters (rank 16, deltas 1e-3 to 10x of W) and many fold / unfold cycles with a moving B."""
    g = torch.Generator(device = device).manual_seed(0)
    for dtype in DTYPES:
        for rel in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
            W = (torch.randn(rows, 1024, device = device, generator = g) * 0.02).to(dtype)
            A = torch.randn(16, 1024, device = device, generator = g).to(dtype)
            B = torch.randn(rows, 16, device = device, generator = g).to(dtype)
            scale = rel * 0.02 / (B.float() @ A.float()).abs().mean().item()
            _check_cycle(W, A, B, scale)
        # The patterns themselves under a random adapter: arbitrary fp32 deltas on every bit pattern.
        W = _all_patterns(dtype, 4, device)
        A = torch.randn(8, W.shape[1], device = device, generator = g).to(dtype)
        B = torch.randn(4, 8, device = device, generator = g).to(dtype)
        _check_cycle(W, A, B, 0.37)
        W = (torch.randn(rows // 2, 512, device = device, generator = g) * 0.02).to(dtype)
        W0 = W.clone()
        A = torch.randn(4, 512, device = device, generator = g).to(dtype)
        B = torch.randn(rows // 2, 4, device = device, generator = g).to(dtype)
        for step in range(cycles):
            B.add_(torch.randn(B.shape, device = device, generator = g).to(dtype) * 0.05)
            t = ex.ExactFoldTarget(W.unsqueeze(0), A.unsqueeze(0), B.unsqueeze(0), 0.01 * (1 + step))
            ex.exact_fold([t])
            ex.exact_unfold([t])
        assert torch.equal(_bits(W), _bits(W0))


def _cuda():
    return torch.cuda.is_available() and ex.triton is not None


@pytest.mark.skipif(not _cuda(), reason = "needs CUDA + Triton")
def test_every_bit_pattern_at_every_edge_restores_exactly():
    run_edges("cuda")


@pytest.mark.skipif(not _cuda(), reason = "needs CUDA + Triton")
def test_every_code_width_and_escape_kind_is_reached():
    COVERAGE.clear()
    run_edges("cuda")
    want = {"code%d" % k for k in range(2, ex._MAX_GAP + 2)} | {"escape_code", "escape_no_code", "escape_nan"}
    assert want <= set(COVERAGE), sorted(want - set(COVERAGE))


@pytest.mark.skipif(not _cuda(), reason = "needs CUDA + Triton")
def test_corrupted_store_or_changed_adapter_raises():
    run_negative("cuda")


@pytest.mark.skipif(not _cuda(), reason = "needs CUDA + Triton")
def test_random_adapters_and_cycles_restore_exactly():
    run_random("cuda", cycles = 200)


@pytest.mark.skipif(_cuda() or ex.triton is None, reason = "CUDA runs the proof directly; this is the CPU interpreter")
def test_cpu_interpreter_proof():
    # The same unsloth_zoo tree as this process (a script run puts tests/ first on sys.path, not the repo).
    root = os.path.dirname(os.path.dirname(os.path.abspath(ex.__file__)))
    path = os.pathsep.join(p for p in (root, os.environ.get("PYTHONPATH", "")) if p)
    env = dict(os.environ, TRITON_INTERPRET = "1", CUDA_VISIBLE_DEVICES = "", PYTHONPATH = path)
    result = subprocess.run(
        [sys.executable, __file__], env = env, capture_output = True, text = True, timeout = 1800,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert "EDGES OK" in result.stdout


if __name__ == "__main__":
    # TRITON_INTERPRET=1 on CPU: the exhaustive cross with large interpreter tiles (the arithmetic is per element, the
    # tiles only change indexing), then the production tiles on a subset, then random adapters and cycles.
    assert os.environ.get("TRITON_INTERPRET") == "1", "run under TRITON_INTERPRET=1"
    blocks = (ex._BLOCK_N, ex._BLOCK_K)
    ex._BLOCK_N, ex._BLOCK_K = 8, 8192
    report = run_edges("cpu")
    want = {"code%d" % k for k in range(2, ex._MAX_GAP + 2)} | {"escape_code", "escape_no_code", "escape_nan"}
    assert want <= set(COVERAGE), sorted(want - set(COVERAGE))
    for dtype, fractions in report.items():
        print(dtype, " ".join(f"{s:g}:{100 * f:.1f}%" for s, f in fractions))
    run_random("cpu", cycles = 20, rows = 32)
    ex._BLOCK_N, ex._BLOCK_K = blocks
    subset = {torch.float16: (2.0**-25, 2.0**-11, 65504.0), torch.bfloat16: (2.0**-134, 2.0**-9, 3.3e38)}
    run_edges("cpu", scales = subset, multipliers = (1.0, -1.5))
    run_negative("cpu")
    print("EDGES OK")

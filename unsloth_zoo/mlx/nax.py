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

"""Apple GPU neural accelerators (NAX) and the MLX dispatch gaps unsloth routes around."""

import functools
import hashlib
import itertools
import json
import logging
import mmap
import os
import platform
import re
import subprocess
import sys
from threading import Lock
from typing import NamedTuple

import mlx.core as mx


logger = logging.getLogger(__name__)


# MLX's `is_nax_available`, which Python cannot call: macOS 26.2 and a generation-17 GPU,
# 18 for the `p` class.
_MIN_MACOS = (26, 2)
_GPU_ARCHITECTURE_PATTERN = re.compile(r"applegpu_g(\d+)([a-z])")


def _gpu_architecture():
    try:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    except RuntimeError:
        return ""
    return str(info.get("architecture", ""))


def _gpu_generation():
    match = _GPU_ARCHITECTURE_PATTERN.fullmatch(_gpu_architecture())
    return int(match.group(1)) if match else None


@functools.cache
def _gpu_core_count():
    """The IORegistry `gpu-core-count`, which `mx.device_info()` does not report; None if unreadable."""
    try:
        result = subprocess.run(["/usr/sbin/ioreg", "-rc", "AGXAccelerator", "-d", "1", "-k", "gpu-core-count"],
                                capture_output = True, text = True, timeout = 5)
    except Exception:
        return None
    match = re.search(r'"gpu-core-count" = (\d+)', result.stdout) if result.returncode == 0 else None
    return int(match.group(1)) if match else None


@functools.cache
def _nax_gpu():
    if platform.system() != "Darwin" or not mx.metal.is_available():
        return False
    try:
        release = tuple(int(part) for part in platform.mac_ver()[0].split(".")[:2])
    except ValueError:
        return False
    match = _GPU_ARCHITECTURE_PATTERN.fullmatch(_gpu_architecture())
    if match is None or release < _MIN_MACOS:
        return False
    return int(match.group(1)) >= (18 if match.group(2) == "p" else 17)


@functools.cache
def _stock_nax_kernels():
    # A wheel built for macOS below 26.2 compiles MLX_METAL_NO_NAX and ships none.
    path = os.path.join(os.path.dirname(mx.__file__), "lib", "mlx.metallib")
    try:
        with open(path, "rb") as file, mmap.mmap(file.fileno(), 0, access = mmap.ACCESS_READ) as data:
            return data.find(b"_nax_") >= 0
    except (OSError, ValueError):
        return False


def nax_available():
    """Whether stock MLX dispatches its NAX kernels here. `UNSLOTH_MLX_NAX=0` turns every route off."""
    return os.environ.get("UNSLOTH_MLX_NAX", "1") != "0" and _nax_gpu() and _stock_nax_kernels()


class Gap(NamedTuple):
    open_in: tuple  # MLX releases measured to leave the gap open
    closed_on_main: bool


# A release missing from `open_in` keeps a gap only while MLX main has not closed it, so both
# the first release with the upstream fix and older unmeasured releases turn the route off.
_GAPS = {
    # A few rows of a transposed quantized matmul run qmv/qmv_wide below the qmv batch limit,
    # then qmm_t_splitk until the output is wide enough for qmm_nax.
    "small_m_qmm": Gap(open_in = ("0.32.2", "0.32.3"), closed_on_main = False),
}


def gap_open(name):
    gap = _GAPS[name]
    return mx.__version__ in gap.open_in or not gap.closed_on_main


_PROBE_PATH = os.path.join(os.path.expanduser("~"), ".cache", "unsloth", "mlx_nax_probes.json")
_PROBE_TIMEOUT = 60
_PROBES = {}
_PROBE_LOCK = Lock()


@functools.cache
def _os_build():
    try:
        return subprocess.run(["sysctl", "-n", "kern.osversion"], capture_output = True,
                              text = True, timeout = 5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _stored_probes():
    try:
        with open(_PROBE_PATH) as file:
            stored = json.load(file)
        return stored if isinstance(stored, dict) else {}
    except (OSError, ValueError):
        return {}


def _store_probe(entry, passed):
    stored = _stored_probes()
    stored[entry] = passed
    try:
        os.makedirs(os.path.dirname(_PROBE_PATH), exist_ok = True)
        temporary = f"{_PROBE_PATH}.{os.getpid()}"
        with open(temporary, "w") as file:
            json.dump(stored, file)
        os.replace(temporary, _PROBE_PATH)
    except OSError:
        pass


def _run_probe(module, function):
    # `-I` keeps the working directory off the child's path: it imports only what this process can.
    paths = [os.path.abspath(path) for path in sys.path if path]
    code = (f"import sys; sys.path[:] = {paths!r}; import importlib; "
            f"getattr(importlib.import_module({module!r}), {function!r})()")
    try:
        result = subprocess.run([sys.executable, "-I", "-c", code], capture_output = True,
                                text = True, timeout = _PROBE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None
    except OSError as error:
        return f"{type(error).__name__}: {error}"
    if result.returncode == 0:
        return ""
    lines = (result.stderr or result.stdout or "").strip().splitlines()
    return lines[-1] if lines else f"exit code {result.returncode}"


def kernel_probe_passed(key: str, module: str, function: str) -> bool:
    """Run `module.function()` once in a subprocess; cache pass/fail per (macOS build, MLX version, key).

    A Metal build failure inside `mx.fast.metal_kernel` can abort the process, so each NAX kernel is
    first built and checked here, where a failure only disables it.
    """
    try:
        if not nax_available():
            return False
        entry = f"{platform.mac_ver()[0]}|{_os_build()}|{mx.__version__}|{key}"
        with _PROBE_LOCK:
            if entry not in _PROBES:
                passed = _stored_probes().get(entry)
                if not isinstance(passed, bool):
                    failure = _run_probe(module, function)
                    passed = failure == ""
                    if failure is not None:  # a timeout is retried by the next process
                        _store_probe(entry, passed)
                    if not passed:
                        logger.warning("NAX kernel %s failed its probe (%s); the native path stays in use",
                                       key, failure or "timed out")
                _PROBES[entry] = passed
            return _PROBES[entry]
    except Exception as error:
        logger.warning("NAX kernel %s could not be probed (%s); the native path stays in use", key, error)
        return False


# Small-row affine quantized matmul `x @ W^T` on matmul2d, reading the stock packed codes, scales
# and biases in place. Each quant group is one TM x TN x group_size matmul into fp32, folded as
# `acc += P * scale + sum(x over the group) * bias`: stock qmv's factorization on the same codes,
# so only the fp32 summation order differs. K is split across threadgroups so that a few rows
# still fill the GPU, and the splits' fp32 partials are then added in a fixed order.
_QMM_BITS = (4, 8)
_QMM_GROUP_SIZES = (32, 64, 128)
_QMM_MAX_ROWS = 16
# Threadgroups per call by row tile. In a chain of dependent projections one kernel runs at a time,
# so its own grid must fill the GPU: these fill the 16-core M5 Pro, and scale with the core count.
_QMM_THREADGROUPS = {8: 1400, 16: 700}
_QMM_MEASURED_CORES = 16
_QMM_THREADGROUP_MEMORY = 32768

_QMM_HEADER = """
#include <metal_tensor>
#include <metal_type_traits>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

_QMM_SOURCE = """
    constexpr int G = K / GS;
    constexpr int SG = TN / 16;
    constexpr int E = TM * TN / (SG * 32);
    constexpr uint STEPS = (G + U - 1) / U;
    constexpr uint SPLIT_GROUPS = (STEPS + KSPLIT - 1) / KSPLIT * U;
    using W = metal::conditional_t<BITS == 4, uint4b_format, uchar>;
    const int M = x_shape[0];
    const int N = w_shape[0];
    const uint n0 = threadgroup_position_in_grid.x * TN;
    const uint split = threadgroup_position_in_grid.y;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint g0 = split * STEPS / KSPLIT * U;
    const uint g1 = min((split + 1) * STEPS / KSPLIT * U, uint(G));

    threadgroup float sums[SPLIT_GROUPS * TM];
    threadgroup float out[TM * TN];
    // At group size 32 a ragged tile's dynamic slices are slow: stage the split's rows, zero past M.
    constexpr uint SK = SPLIT_GROUPS * GS;
    constexpr bool STAGE = EDGE && GS == 32 && TM * (SK * sizeof(T) + SPLIT_GROUPS * 4 + TN * 4) <= 32768;
    threadgroup T xs[STAGE ? TM * SK : 1];
    if constexpr (STAGE) {
        for (uint e = (simd * 32 + lane) * 4; e < TM * SK; e += SG * 128) {
            const uint r = e / SK, k = e % SK;
            vec<T, 4> v = vec<T, 4>(0);
            if (int(r) < M && k < (g1 - g0) * GS) v = *(const device vec<T, 4>*)(x + ulong(r) * K + g0 * GS + k);
            *(threadgroup vec<T, 4>*)(xs + e) = v;
        }
    }
    for (uint i = simd; i < (g1 - g0) * TM; i += SG) {
        const uint r = i % TM;
        float v = 0.0f;
        if (int(r) < M) {
            const device T* xr = x + ulong(r) * K + (g0 + i / TM) * GS;
            for (uint j = lane; j < GS; j += 32) v += float(xr[j]);
            v = simd_sum(v);
        }
        if (lane == 0) sums[i] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto a = tensor((device T*)x, dextents<int, 2>{K, M}, array<int, 2>{1, K});
    tensor<device W, dextents<int, 2>, tensor_inline> b(
        (device uchar*)w + ulong(n0) * K * BITS / 8, dextents<int, 2>{K, TN}, array<int, 2>{1, K});
    constexpr auto desc = matmul2d_descriptor(TM, TN, GS, false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto staged = tensor((threadgroup T*)xs, dextents<int, 2>{int(SK), TM}, array<int, 2>{1, int(SK)});
    // Static-extent tiles are not bounds-checked; with fewer than TM rows the dynamic ones keep reads within M.
    auto a_tile = [&](int k0) {
        if constexpr (STAGE) return staged.template slice<GS, TM>(k0 - int(g0 * GS), 0);
        else if constexpr (EDGE) return a.slice(k0, 0);
        else return a.template slice<GS, TM>(k0, 0);
    };
    auto a0 = a_tile(int(g0 * GS));
    auto b0 = b.template slice<GS, TN>(0, 0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    uint col[E], row[E];
    bool ok[E];
    for (ushort i = 0; i < E; ++i) {
        acc[i] = 0.0f;
        ok[i] = acc.is_valid_element(i);
        auto idx = acc.get_multidimensional_index(i);
        col[i] = ok[i] ? uint(idx[0]) : 0u;
        row[i] = ok[i] ? uint(idx[1]) : 0u;
    }
    const device T* sp = scales + ulong(n0) * G;
    const device T* bp = biases + ulong(n0) * G;

    // U (2, 4 or 8) matmuls are issued before any is folded, so their latencies overlap.
    uint g = g0;
    for (; g + U <= g1; g += U) {
        float s[U][E], c[U][E];
        for (ushort j = 0; j < U; ++j) {
            for (ushort i = 0; i < E; ++i) {
                s[j][i] = float(sp[col[i] * G + g + j]);
                c[j][i] = float(bp[col[i] * G + g + j]);
            }
        }
        // Cooperative tensors cannot form an array.
        decltype(acc) p0, p1, p2, p3, p4, p5, p6, p7;
        auto run = [&](ushort j, thread decltype(acc)& p) {
            auto as = a_tile((g + j) * GS);
            auto bs = b.template slice<GS, TN>((g + j) * GS, 0);
            op.run(as, bs, p);
        };
        auto fold = [&](ushort j, thread decltype(acc)& p) {
            for (ushort i = 0; i < E; ++i) {
                acc[i] += p[i] * s[j][i] + sums[(g + j - g0) * TM + row[i]] * c[j][i];
            }
        };
        run(0, p0);
        run(1, p1);
        if constexpr (U > 2) { run(2, p2); run(3, p3); }
        if constexpr (U > 4) { run(4, p4); run(5, p5); run(6, p6); run(7, p7); }
        fold(0, p0);
        fold(1, p1);
        if constexpr (U > 2) { fold(2, p2); fold(3, p3); }
        if constexpr (U > 4) { fold(4, p4); fold(5, p5); fold(6, p6); fold(7, p7); }
    }
    for (; g < g1; ++g) {
        decltype(acc) p;
        auto as = a_tile(g * GS);
        auto bs = b.template slice<GS, TN>(g * GS, 0);
        op.run(as, bs, p);
        for (ushort i = 0; i < E; ++i) {
            acc[i] += p[i] * float(sp[col[i] * G + g]) + sums[(g - g0) * TM + row[i]] * float(bp[col[i] * G + g]);
        }
    }

    for (ushort i = 0; i < E; ++i) {
        if (ok[i]) out[row[i] * TN + col[i]] = acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = simd * 32 + lane; o < uint(M) * TN; o += SG * 32) {
        y[(ulong(split) * M + o / TN) * N + n0 + o % TN] = OT(out[o]);
    }
"""

_QMM_REDUCE_SOURCE = """
    const uint i = thread_position_in_grid.x * 4;
    const uint size = p_shape[0] / KSPLIT;
    if (i >= size) return;
    float4 v = 0.0f;
    for (int k = 0; k < KSPLIT; ++k) v += *(const device float4*)(p + ulong(k) * size + i);
    *(device vec<T, 4>*)(y + i) = vec<T, 4>(v);
"""

QMM_PROBE_KEY = "small_m_qmm:" + hashlib.sha256(
    (_QMM_HEADER + _QMM_SOURCE + _QMM_REDUCE_SOURCE).encode()).hexdigest()[:12]


@functools.cache
def _qmm_kernels():
    return (mx.fast.metal_kernel(name = "unsloth_nax_small_m_qmm", input_names = ["x", "w", "scales", "biases"],
                                 output_names = ["y"], header = _QMM_HEADER, source = _QMM_SOURCE),
            mx.fast.metal_kernel(name = "unsloth_nax_small_m_qmm_reduce", input_names = ["p"],
                                 output_names = ["y"], source = _QMM_REDUCE_SOURCE))


def small_m_qmm_supported(N, K, group_size, bits, mode):
    """Whether the small-row kernel covers an `[N, K]` weight quantized this way; the rest stay native."""
    return (mode == "affine" and bits in _QMM_BITS and group_size in _QMM_GROUP_SIZES
            and N > 0 and N % 64 == 0 and K > 0 and K % group_size == 0)


def small_m_qmm_geometry(M, N, K, group_size):
    """The (row tile, column tile, K splits, matmuls per fold) variant a call of M rows runs."""
    row_tile = 8 if M <= 8 else 16
    column_tile = 128 if N % 128 == 0 else 64
    # Each thread holds matmuls-per-fold x row_tile / 2 partials; more spills registers.
    unroll = min(256 // group_size, 64 // row_tile)
    steps = -(-(K // group_size) // unroll)
    cores = _gpu_core_count() or _QMM_MEASURED_CORES
    threadgroups = _QMM_THREADGROUPS[row_tile] * cores // _QMM_MEASURED_CORES
    splits = max(1, min(threadgroups // (N // column_tile), steps))
    splits = -(-steps // -(-steps // splits))   # no more splits than the longest one needs
    while row_tile * (-(-steps // splits) * unroll + column_tile) * 4 > _QMM_THREADGROUP_MEMORY:
        splits += 1
    return row_tile, column_tile, splits, unroll


def small_m_qmm(x, w, scales, biases, group_size, bits):
    """`quantized_matmul(x, w, scales, biases, transpose=True)` for a 2-D x of at most 16 rows."""
    M, K = x.shape
    N = w.shape[0]
    row_tile, column_tile, splits, unroll = small_m_qmm_geometry(M, N, K, group_size)
    main, reduce = _qmm_kernels()
    partials = main(
        inputs = [x, w, scales, biases],
        template = [("T", x.dtype), ("OT", x.dtype if splits == 1 else mx.float32), ("K", K),
                    ("GS", group_size), ("BITS", bits), ("TM", row_tile), ("TN", column_tile),
                    ("KSPLIT", splits), ("U", unroll), ("EDGE", M % row_tile != 0)],
        grid = (2 * N, splits, 1), threadgroup = (2 * column_tile, 1, 1),
        output_shapes = [(M, N) if splits == 1 else (splits * M * N,)],
        output_dtypes = [x.dtype if splits == 1 else mx.float32],
    )[0]
    if splits == 1:
        return partials
    return reduce(inputs = [partials], template = [("T", x.dtype), ("KSPLIT", splits)],
                  grid = (M * N // 4, 1, 1), threadgroup = (min(256, M * N // 4), 1, 1),
                  output_shapes = [(M, N)], output_dtypes = [x.dtype])[0]


# Rows per call where the kernel beat stock by >= 1.05x on chained projections (M5 Pro; the M5
# family is assumed to match its bandwidth per core). First match wins: (bits, group size or None,
# fewest weights, widest N, fewest rows, most rows). Heads wider than 64K never start below the
# narrower outputs' entry: a head alone gains too little. Unmeasured generations keep >= 1.3x rows.
_QMM_ROWS_BY_GPU = {
    17: (
        (4, None, 1 << 22, 8192, 6, 16), (4, None, 1 << 22, 65536, 6, 15), (4, None, 1 << 22, None, 6, 16),
        (4, None, 1 << 21, None, 6, 16), (8, 32, 1 << 23, 8192, 11, 16), (8, None, 1 << 23, 8192, 7, 16),
        (8, None, 1 << 23, 65536, 7, 14), (8, 32, 1 << 23, None, 11, 16), (8, None, 1 << 23, None, 7, 16),
        (8, None, 1 << 21, None, 11, 16),
    ),
}
_QMM_ROWS_UNMEASURED = (
    (4, None, 1 << 22, 8192, 11, 16), (4, None, 1 << 22, 65536, 7, 12), (4, None, 1 << 22, None, 7, 12),
    (8, 32, 1 << 23, 8192, 16, 16), (8, None, 1 << 23, 8192, 11, 16), (8, None, 1 << 23, 65536, 11, 12),
    (8, None, 1 << 23, None, 11, 12),
)


def small_m_qmm_row_range(N, K, group_size, bits):
    """The row counts (lowest, highest) where the kernel is measured faster than stock for `[N, K]`."""
    for entry in _QMM_ROWS_BY_GPU.get(_gpu_generation(), _QMM_ROWS_UNMEASURED):
        entry_bits, entry_group_size, fewest, widest, low, high = entry
        if (bits == entry_bits and entry_group_size in (None, group_size) and N * K >= fewest
                and (widest is None or N <= widest)):
            return max(low, 2), min(high, _QMM_MAX_ROWS)  # one row stays on stock qmv, at the bandwidth roofline
    return 0, -1


def probe_small_m_qmm():
    """Subprocess probe for the small-row kernel: at each bit width, both row tiles full and ragged,
    both column tiles, split and unsplit K, every group size, bf16 and fp16."""
    for dtype in (mx.bfloat16, mx.float16):
        for bits in _QMM_BITS:
            for (N, K), group_size in itertools.product(((384, 512), (320, 256)), _QMM_GROUP_SIZES):
                w = mx.random.normal((N, K), key = mx.random.key(group_size)) * 0.05
                w, scales, biases = mx.quantize(w.astype(dtype), group_size = group_size, bits = bits)
                for M in (3, 8, 13, 16):
                    x = mx.random.normal((M, K), key = mx.random.key(M)).astype(dtype)
                    native = mx.quantized_matmul(x, w, scales, biases, transpose = True,
                                                 group_size = group_size, bits = bits).astype(mx.float32)
                    got = small_m_qmm(x, w, scales, biases, group_size, bits).astype(mx.float32)
                    error = mx.abs(got - native).max().item()
                    assert error <= 0.02 * mx.abs(native).max().item(), (bits, group_size, M, error)



# Prefill `x @ W^T` with int8 activations (oMLX's QxA8 design), reading the stock packed codes,
# scales and biases in place. Each row's activation group of x (the quant group, at most 64 wide)
# becomes int8 codes a with an fp32 scale sa and code sum Ra; each is one exact int8 x int8 -> int32
# matmul P against the codes, folded in fp32 as `acc += sa * s * P`. The bias term, the sum over
# groups of sa * Ra * b', is then one matmul of U (sa * Ra summed over each quant group), stored as
# bf16 high and low halves, against b'. Codes under 8 bits enter as they are with b' = b; 8-bit codes are
# centred (q - 128) and b' = b + 128 * s, applied as two matmuls so b' is never rounded.
_INT8_QMM_GROUP_SIZES = (32, 64, 128)
_INT8_QMM_BITS = (3, 4, 5, 6, 8)


def _int8_qmm_activation_group(group_size):
    return min(group_size, 64)


_INT8_QMM_HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
"""

# One threadgroup per row, a simdgroup per quant group. The GEMM's lanes each read a quarter of an
# activation group (16 codes, or 8 for 32-wide groups), 4 codes per matmul step; for 4-bit each word's
# even and odd nibbles are separate steps, so the codes of a step are k, k+2, k+4, k+6 and the
# activations are stored in that order. A group with a non-finite value gets a NaN scale, so the output
# keeps the NaN.
_INT8_ACTIVATION_QUANTIZE_SOURCE = """
    constexpr int GW = K / GS;
    constexpr int GP = (GW + 15) / 16 * 16;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.x;
    const uint M = x_shape[0];
    const device T* xr = x + ulong(row) * K;
    device int8_t* qr = qa + ulong(row) * K;
    auto slot = [](uint k) {
        if (BITS != 4) return k;
        const uint r = k & 15;
        return (k & ~15u) | (((r >> 3) * 2 + (r & 1)) * 4 + ((r & 7) >> 1));
    };
    for (uint gw = simd; gw < GW; gw += 8) {
        float u = 0.0f;
        for (uint g = gw * (GS / AG); g < (gw + 1) * (GS / AG); ++g) {
            float v[AG / 32];
            float amax = 0.0f;
            bool finite = true;
            _Pragma("clang loop unroll(full)")
            for (int i = 0; i < AG / 32; ++i) {
                v[i] = float(xr[g * AG + lane + 32 * i]);
                amax = max(amax, abs(v[i]));
                finite = finite && isfinite(v[i]);
            }
            amax = simd_max(amax);
            finite = !simd_any(!finite);
            const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
            int sum = 0;
            _Pragma("clang loop unroll(full)")
            for (int i = 0; i < AG / 32; ++i) {
                const int q = int(clamp(rint(v[i] * inv), -127.0f, 127.0f));
                qr[g * AG + slot(lane + 32 * i)] = int8_t(q);
                sum += q;
            }
            const float scale = finite ? amax / 127.0f : NAN;
            u += scale * float(simd_sum(sum));
            if (lane == 0) sa[ulong(g) * M + row] = scale;
        }
        if (lane == 0) {
            uh[ulong(row) * GP + gw] = bfloat(u);
            ul[ulong(row) * GP + gw] = bfloat(u - float(bfloat(u)));
        }
    }
    if (simd == 0 && lane < GP - GW) {
        uh[ulong(row) * GP + GW + lane] = bfloat(0.0f);
        ul[ulong(row) * GP + GW + lane] = bfloat(0.0f);
    }
"""

# A simdgroup computes 2 x 16 rows by 32 columns with single-simdgroup 16x32x16 int8 matmuls whose
# operands are filled from registers; a threadgroup is WM x WN simdgroups. The bias matmul uses bf16 x T
# operands in the same register layout: an fp32 operand would change it. Rows past the tile's valid
# range load its last valid row and are never stored. GATHER: `w` holds one matrix per expert and
# the rows are sorted by expert; `starts` gives each expert's first row and `tiles` its first tile,
# so every tile belongs to one expert. MAP: row i's activations are row `rmap[i]` of the quantized x,
# which holds one row per token however many experts it is routed to.
_INT8_QMM_SOURCE = """
    static_assert(sizeof(T) == 2, "16-bit activations, scales and biases only");
    constexpr int G = K / AG;
    constexpr int GW = K / GS;
    constexpr int WORDS = AG * BITS / 32;
    constexpr int RUN = WORDS / 4;   // words of a lane's run, at 4 and 8 bits
    constexpr bool STREAM = BITS != 4 && BITS != 8;
    constexpr int RUN_BITS = AG / 4 * BITS;
    constexpr int BM = 32 * WM;
    constexpr int BN = 32 * WN;
    const int M = qa_shape[0];
    const int N = GATHER ? w_shape[1] : w_shape[0];
    const uint simd = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const int tile = int(threadgroup_position_in_grid.y);

    int row_lo = tile * BM, row_hi = min(row_lo + BM, M);
    const device uint* wp = w;
    const device T* sp = scales;
    const device T* bp = biases;
    if (GATHER) {
        const int E = w_shape[0];
        if (tile >= tiles[E]) return;
        int lo = 0, hi = E;   // tiles[lo] <= tile < tiles[hi]
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (tiles[mid] <= tile) lo = mid; else hi = mid;
        }
        row_lo = starts[lo] + (tile - tiles[lo]) * BM;
        row_hi = min(row_lo + BM, starts[lo + 1]);
        wp += ulong(lo) * N * (G * WORDS);
        sp += ulong(lo) * N * GW;
        bp += ulong(lo) * N * GW;
    }

    // The matmul fragments' lane layout.
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    const int cx = fn >> 2;
    const int row_base = row_lo + int(simd % WM) * 32 + fm;
    const int col_base = int(threadgroup_position_in_grid.x) * BN + int(simd / WM) * 32;

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto desc_set = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    mpp::tensor_ops::matmul2d<desc_set, metal::execution_simdgroup> op_set;
    auto ct_a = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
    auto ct_b = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
    auto acc0 = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), int32_t>();
    auto acc1 = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), int32_t>();

    int rows[4];
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 4; ++i) rows[i] = min(row_base + i * 8, row_hi - 1);   // rows +0, +8, +16, +24
    int arow[4];   // where each row's activations were quantized
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 4; ++i) arow[i] = MAP ? int(rmap[rows[i]]) : rows[i];
    float out[2][16];
    _Pragma("clang loop unroll(full)")
    for (int h = 0; h < 2; ++h)
        _Pragma("clang loop unroll(full)")
        for (int e = 0; e < 16; ++e) out[h][e] = 0.0f;
    const device uint* wlane = wp + ulong(col_base + fm) * (G * WORDS);
    const device T* slane = sp + ulong(col_base + fn) * GW;
    float s[8], scale[4];
    auto load_scales = [&](int g) {
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < 8; ++j)   // columns fn + (j & 3) + (j >> 2) * 16
            s[j] = float(slane[((j & 3) + (j >> 2) * 16) * GW + g / (GS / AG)]);
        _Pragma("clang loop unroll(full)")
        for (int i = 0; i < 4; ++i) scale[i] = sa[ulong(g) * M + arow[i]];
    };

    for (int g = 0; g < G; ++g) {
        constexpr bool EARLY = BITS == 4 || BITS == 6;   // scales before the matmuls or after them: measured
        if (EARLY) load_scales(g);
        uint4 wv[4];
        _Pragma("clang loop unroll(full)")
        for (int q = 0; q < 4; ++q) {   // columns +0, +8, +16, +24
            const device uint* wr = wlane + ulong((q & 1) * 8 + (q >> 1) * 16) * (G * WORDS) + g * WORDS;
            wv[q] = uint4(0u);
            if (STREAM) {   // 3-, 5- and 6-bit codes: the words holding the run, realigned to its first bit
                const device uint* p = wr + (cx * RUN_BITS >> 5);
                const uint s0 = cx * RUN_BITS & 31;
                wv[q].x = p[0];
                if (s0 + RUN_BITS > 32) wv[q].y = p[1];
                if (s0 + RUN_BITS > 64) wv[q].z = p[2];
                if (s0) wv[q] = uint4((wv[q].x >> s0) | (wv[q].y << (32 - s0)),
                                      (wv[q].y >> s0) | (wv[q].z << (32 - s0)), wv[q].z >> s0, 0u);
            }
            else if (RUN == 4) wv[q] = ((const device uint4*)wr)[cx];
            else if (RUN == 2) wv[q].xy = ((const device uint2*)wr)[cx];
            else wv[q].x = wr[cx];
            if (BITS == 8) wv[q] ^= uint4(0x80808080u);
        }
        _Pragma("clang loop unroll(full)")
        for (int t = 0; t < AG / 16; ++t) {
            _Pragma("clang loop unroll(full)")
            for (int q = 0; q < 4; ++q) {
                const int base = (q >> 1) * 8 + (q & 1) * 4;
                char4 c;
                if (STREAM) {
                    const int o = t * 4 * BITS, i = o >> 5, sh = o & 31;
                    const uint f = sh ? (wv[q][i] >> sh) | (wv[q][i + 1] << (32 - sh)) : wv[q][i];
                    constexpr uint m = (1u << BITS) - 1;
                    c = as_type<char4>((f & m) | ((f << (8 - BITS)) & (m << 8))
                                       | ((f << (16 - 2 * BITS)) & (m << 16)) | ((f << (24 - 3 * BITS)) & (m << 24)));
                }
                else c = BITS == 8 ? as_type<char4>(wv[q][t])
                                   : as_type<char4>(((t & 1) ? wv[q][t >> 1] >> 4 : wv[q][t >> 1]) & 0x0f0f0f0fu);
                ct_b[base] = c.x; ct_b[base + 1] = c.y; ct_b[base + 2] = c.z; ct_b[base + 3] = c.w;
            }
            _Pragma("clang loop unroll(full)")
            for (int h = 0; h < 2; ++h) {
                _Pragma("clang loop unroll(full)")
                for (int r = 0; r < 2; ++r) {
                    const char4 c = as_type<char4>(*(const device uint*)(qa + ulong(arow[h * 2 + r]) * K
                                                                        + g * AG + cx * (AG / 4) + t * 4));
                    ct_a[r * 4] = c.x; ct_a[r * 4 + 1] = c.y; ct_a[r * 4 + 2] = c.z; ct_a[r * 4 + 3] = c.w;
                }
                if (t == 0) { if (h == 0) op_set.run(ct_a, ct_b, acc0); else op_set.run(ct_a, ct_b, acc1); }
                else if (h == 0) op.run(ct_a, ct_b, acc0);
                else op.run(ct_a, ct_b, acc1);
            }
        }
        if (!EARLY) load_scales(g);
        _Pragma("clang loop unroll(full)")
        for (int e = 0; e < 16; ++e) {
            const int j = (e & 3) + (e >> 3) * 4, r = (e >> 2) & 1;
            out[0][e] = metal::fma(scale[r], s[j] * float(acc0[e]), out[0][e]);
            out[1][e] = metal::fma(scale[2 + r], s[j] * float(acc1[e]), out[1][e]);
        }
    }
    {
        constexpr int GP = (GW + 15) / 16 * 16;
        constexpr auto bdesc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<bdesc, metal::execution_simdgroup> bop;
        auto bt_a = bop.template get_left_input_cooperative_tensor<bfloat, T, float>();
        auto bt_b = bop.template get_right_input_cooperative_tensor<bfloat, T, float>();
        auto bt_s = bop.template get_right_input_cooperative_tensor<bfloat, T, float>();
        auto bo0 = bop.template get_destination_cooperative_tensor<decltype(bt_a), decltype(bt_b), float>();
        auto bo1 = bop.template get_destination_cooperative_tensor<decltype(bt_a), decltype(bt_b), float>();
        _Pragma("clang loop unroll(full)")
        for (int e = 0; e < 16; ++e) { bo0[e] = out[0][e]; bo1[e] = out[1][e]; }
        for (int g0 = 0; g0 < GP; g0 += 16) {
            const int k = g0 + cx * 4;
            _Pragma("clang loop unroll(full)")
            for (int q = 0; q < 4; ++q) {
                const int base = (q >> 1) * 8 + (q & 1) * 4;
                const ulong at = ulong(col_base + fm + (q & 1) * 8 + (q >> 1) * 16) * GW + k;
                _Pragma("clang loop unroll(full)")
                for (int i = 0; i < 4; ++i) {
                    bt_b[base + i] = k + i < GW ? bp[at + i] : T(0);
                    if (BITS == 8) bt_s[base + i] = k + i < GW ? T(128) * sp[at + i] : T(0);
                }
            }
            _Pragma("clang loop unroll(full)")
            for (int part = 0; part < 2; ++part) {
                _Pragma("clang loop unroll(full)")
                for (int h = 0; h < 2; ++h) {
                    _Pragma("clang loop unroll(full)")
                    for (int r = 0; r < 2; ++r) {
                        const device bfloat* ur = (part ? ul : uh) + ulong(arow[h * 2 + r]) * GP + k;
                        _Pragma("clang loop unroll(full)")
                        for (int i = 0; i < 4; ++i) bt_a[r * 4 + i] = ur[i];
                    }
                    if (h == 0) bop.run(bt_a, bt_b, bo0); else bop.run(bt_a, bt_b, bo1);
                    if (BITS == 8) { if (h == 0) bop.run(bt_a, bt_s, bo0); else bop.run(bt_a, bt_s, bo1); }
                }
            }
        }
        _Pragma("clang loop unroll(full)")
        for (int e = 0; e < 16; ++e) { out[0][e] = bo0[e]; out[1][e] = bo1[e]; }
    }
    _Pragma("clang loop unroll(full)")
    for (int h = 0; h < 2; ++h) {
        _Pragma("clang loop unroll(full)")
        for (int e = 0; e < 16; ++e) {
            const int m = row_base + h * 16 + ((e >> 2) & 1) * 8;
            if (m < row_hi) y[ulong(m) * N + col_base + (e >> 3) * 16 + fn + (e & 3)] = T(out[h][e]);
        }
    }
"""

INT8_QMM_PROBE_KEY = "int8_qmm:" + hashlib.sha256(
    (_INT8_QMM_HEADER + _INT8_ACTIVATION_QUANTIZE_SOURCE + _INT8_QMM_SOURCE).encode()).hexdigest()[:12]


@functools.cache
def _int8_qmm_kernels():
    return (mx.fast.metal_kernel(name = "unsloth_nax_int8_activation_quantize", input_names = ["x"],
                                 output_names = ["qa", "sa", "uh", "ul"], source = _INT8_ACTIVATION_QUANTIZE_SOURCE),
            mx.fast.metal_kernel(name = "unsloth_nax_int8_qmm",
                                 input_names = ["qa", "sa", "uh", "ul", "w", "scales", "biases", "starts", "tiles",
                                                "rmap"],
                                 output_names = ["y"], header = _INT8_QMM_HEADER, source = _INT8_QMM_SOURCE))


def int8_qmm_supported(N, K, group_size, bits, mode):
    """Whether the int8-activation kernel covers an `[N, K]` weight quantized this way."""
    return (mode == "affine" and bits in _INT8_QMM_BITS and group_size in _INT8_QMM_GROUP_SIZES
            and N > 0 and N % 64 == 0 and K > 0 and K % group_size == 0)


def _int8_quantize_activations(x, group_size, bits):
    M, K = x.shape
    span, padded = _int8_qmm_activation_group(group_size), -(-(K // group_size) // 16) * 16
    return _int8_qmm_kernels()[0](
        inputs = [x], template = [("T", x.dtype), ("K", K), ("BITS", bits), ("GS", group_size), ("AG", span)],
        grid = (M * 256, 1, 1), threadgroup = (256, 1, 1),
        output_shapes = [(M, K), (K // span, M), (M, padded), (M, padded)],
        output_dtypes = [mx.int8, mx.float32, mx.bfloat16, mx.bfloat16])


def _int8_qmm(x, w, scales, biases, group_size, bits, simdgroups, starts = None, tiles = None, row_tiles = None,
             token_rows = None):
    M, K = x.shape if token_rows is None else (token_rows.shape[0], x.shape[1])
    N = w.shape[-2]
    rows_per_simdgroup, columns = simdgroups
    qa, sa, uh, ul = _int8_quantize_activations(x, group_size, bits)
    gather = starts is not None
    if not gather:
        starts = tiles = mx.zeros((1,), mx.int32)
        row_tiles = -(-M // (32 * rows_per_simdgroup))
    mapped = token_rows is not None
    return _int8_qmm_kernels()[1](
        inputs = [qa, sa, uh, ul, w, scales, biases, starts, tiles, token_rows if mapped else starts],
        template = [("T", x.dtype), ("K", K), ("BITS", bits), ("GS", group_size),
                    ("AG", _int8_qmm_activation_group(group_size)), ("WM", rows_per_simdgroup), ("WN", columns),
                    ("GATHER", gather), ("MAP", mapped)],
        grid = (N // (32 * columns) * 32 * rows_per_simdgroup * columns, row_tiles, 1),
        threadgroup = (32 * rows_per_simdgroup * columns, 1, 1),
        output_shapes = [(M, N)], output_dtypes = [x.dtype])[0]


# (row, column) simdgroups per threadgroup: 32 x 64 tiles under 8 bits, 64 x 64 for 8-bit; experts
# use 32-row tiles, which waste fewer rows on short expert segments.
_INT8_QMM_SIMDGROUPS = {3: (1, 2), 4: (1, 2), 5: (1, 2), 6: (1, 2), 8: (2, 2)}
_INT8_GATHER_QMM_SIMDGROUPS = {bits: (1, 2) for bits in _INT8_QMM_BITS}


@functools.cache
def _differentiable_int8_qmm(group_size, bits, gather):
    """The kernel call with the stock op's vjp for x, scales and biases (Metal kernels have none); packed
    weights and indices, which stock cannot differentiate, get zeros."""
    def stock(x, w, scales, biases, indices = None, token_rows = None):
        if gather:
            x = x if token_rows is None else x[token_rows]
            return mx.gather_qmm(x[:, None], w, scales, biases, rhs_indices = indices, transpose = True,
                                 group_size = group_size, bits = bits, sorted_indices = True)[:, 0]
        return mx.quantized_matmul(x, w, scales, biases, transpose = True, group_size = group_size, bits = bits)

    @mx.custom_function
    def routed(x, w, scales, biases, *indices):
        if gather:
            return _int8_gather_qmm(x, w, scales, biases, group_size, bits, *indices)
        return _int8_qmm(x, w, scales, biases, group_size, bits, _INT8_QMM_SIMDGROUPS[bits])

    @routed.vjp
    def routed_vjp(primals, cotangent, output):
        x, w, scales, biases, *indices = primals
        _, (dx, ds, db) = mx.vjp(lambda x, s, b: stock(x, w, s, b, *indices), (x, scales, biases), (cotangent,))
        return (dx, mx.zeros_like(w), ds, db, *map(mx.zeros_like, indices))

    return routed


def int8_qmm(x, w, scales, biases, bits):
    """`quantized_matmul(x, w, scales, biases, transpose=True)` for a 2-D x, with int8 activations and
    the group size the scales imply; x, scales and biases must be float16 or bfloat16."""
    return _differentiable_int8_qmm(x.shape[-1] // scales.shape[-1], bits, False)(x, w, scales, biases)


def int8_gather_qmm(x, w, scales, biases, indices, bits, token_rows = None):
    """`gather_qmm(x, w, ..., rhs_indices=indices, transpose=True, sorted_indices=True)` for x of
    shape [T, K] and sorted expert indices [T], with int8 activations; returns [T, N]. x, scales and
    biases must be float16 or bfloat16. With `token_rows`, row i reads `x[token_rows[i]]`, so each
    row of x is quantized once however many experts it is routed to."""
    routed = _differentiable_int8_qmm(x.shape[-1] // scales.shape[-1], bits, True)
    if token_rows is None:
        return routed(x, w, scales, biases, indices)
    return routed(x, w, scales, biases, indices, token_rows)


def _int8_gather_qmm(x, w, scales, biases, group_size, bits, indices, token_rows = None):
    E = w.shape[0]
    rows = 32 * _INT8_GATHER_QMM_SIMDGROUPS[bits][0]
    starts = (indices[None, :] < mx.arange(E + 1, dtype = indices.dtype)[:, None]).sum(axis = 1).astype(mx.int32)
    counts = starts[1:] - starts[:-1]
    tiles = mx.concatenate([mx.zeros((1,), mx.int32), mx.cumsum((counts + rows - 1) // rows)])
    # Each expert adds at most one partial tile; threadgroups past the last tile return at once.
    row_tiles = indices.shape[0] // rows + min(E, indices.shape[0])
    return _int8_qmm(x, w, scales, biases, group_size, bits, _INT8_GATHER_QMM_SIMDGROUPS[bits], starts, tiles,
                     row_tiles, token_rows)


# Measured on the M5 Pro and used on every NAX GPU, as the route is opt-in: every width beats stock end to
# end from the first row count the small-row kernel leaves to stock. Weights under 1024 wide or deep gain
# too little; heads over 64K wide are evaluated at one position in prefill.
_INT8_PREFILL_MIN_ROWS = _QMM_MAX_ROWS + 1
_INT8_PREFILL_MIN_N = _INT8_PREFILL_MIN_K = 1024
_INT8_PREFILL_MAX_N = 65536
# Rows per routed expert by bits: (calls that quantize each token once for all its experts, every other
# call). Down projections lose at 12 rows per expert except at 4 bits.
_INT8_PREFILL_EXPERT_ROWS = {3: (8, 32), 4: (8, 8), 5: (8, 32), 6: (8, 32), 8: (16, 32)}


def int8_prefill_min_rows(N, K):
    """The fewest rows at which the int8 route speeds up prefill for an `[N, K]` weight, or 0."""
    return _INT8_PREFILL_MIN_ROWS if _INT8_PREFILL_MIN_N <= N <= _INT8_PREFILL_MAX_N and K >= _INT8_PREFILL_MIN_K else 0


def int8_prefill_expert_min_rows(E, bits, per_token = False):
    """The fewest sorted rows per call at which the gathered int8 route is measured faster, or 0;
    `per_token` for a call whose rows read one quantized copy of each token."""
    return _INT8_PREFILL_EXPERT_ROWS.get(bits, (0, 0))[not per_token] * E


def probe_int8_qmm():
    """Subprocess probe for the int8 kernels: every bit width and group size, dense, gathered and
    gathered through a token map, bf16 and fp16."""
    for dtype, bits, group_size in itertools.product((mx.bfloat16, mx.float16), _INT8_QMM_BITS, _INT8_QMM_GROUP_SIZES):
        w = mx.random.normal((4, 128, 256), key = mx.random.key(bits)) * 0.05
        w, scales, biases = mx.quantize(w.astype(dtype), group_size = group_size, bits = bits)
        x = mx.random.normal((70, 256), key = mx.random.key(1)).astype(dtype)
        indices = mx.sort(mx.random.randint(0, 4, (70,), key = mx.random.key(2)).astype(mx.uint32))
        tokens = mx.random.randint(0, 70, (70,), key = mx.random.key(3)).astype(mx.uint32)
        for got, native in (
                (int8_qmm(x, w[0], scales[0], biases[0], bits),
                 mx.quantized_matmul(x, w[0], scales[0], biases[0], transpose = True, group_size = group_size,
                                     bits = bits)),
                (int8_gather_qmm(x, w, scales, biases, indices, bits),
                 mx.gather_qmm(x[:, None], w, scales, biases, rhs_indices = indices, transpose = True,
                               group_size = group_size, bits = bits, sorted_indices = True)[:, 0]),
                (int8_gather_qmm(x, w, scales, biases, indices, bits, tokens),
                 mx.gather_qmm(x[tokens][:, None], w, scales, biases, rhs_indices = indices, transpose = True,
                               group_size = group_size, bits = bits, sorted_indices = True)[:, 0])):
            error = mx.abs(got.astype(mx.float32) - native.astype(mx.float32)).max().item()
            assert error <= 0.05 * mx.abs(native.astype(mx.float32)).max().item(), (bits, group_size, dtype, error)

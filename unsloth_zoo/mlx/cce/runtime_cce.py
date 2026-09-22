# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Copyright © 2026

"""Chunked cross-entropy helpers built from MLX runtime custom kernels."""

from collections import OrderedDict
from typing import Callable

import mlx.core as mx

__all__ = [
    "make_chunked_cross_entropy_loss",
    "make_runtime_cce_loss_fused_finalize",
]


def _get_memory_budget() -> int:
    """Return a per-chunk logit byte budget based on available hardware memory.

    Two considerations in tension:
      1. Smaller devices need aggressive chunking (more chunks, smaller each)
         to avoid OOM.
      2. The MLX scheduler benefits from small chunks regardless of device size
         — but too many chunks incurs kernel-launch overhead.

    We use 0.1% of the device's recommended working set as the budget, capped
    at 128 MB and floored at 4 MB.  The cap ensures the scheduler always gets
    enough granularity (≥16 chunks for 128K vocab at any batch size ≤4), while
    the hardware scaling ensures small devices chunk even more aggressively.

      M4 Max 128GB → 103 GB recommended → min(103 MB, 128 MB) = 103 MB
      M3 Pro 36GB  →  27 GB recommended → min(27 MB, 128 MB)  =  27 MB
      M2 16GB      →  12 GB recommended → min(12 MB, 128 MB)  =  12 MB
      M1 8GB       →   6 GB recommended → min(6 MB, 128 MB)   =   6 MB

    Falls back to 128 MB if device info is unavailable.
    """
    try:
        import mlx.core as _mx
        info = _mx.device_info()
        recommended = info.get("max_recommended_working_set_size", 0)
        if recommended > 0:
            hw_budget = int(recommended * 0.001)
            return max(4 * 1024 * 1024, min(hw_budget, 128 * 1024 * 1024))
    except Exception:
        pass
    return 128 * 1024 * 1024


_CHUNK_BUDGET: int | None = None
_CHUNK_PLAN_CACHE_MAX_ENTRIES = 16


def _resolve_chunk_size(
    requested_chunk_size: int,
    n_tokens: int,
    vocab_size: int,
    *,
    bytes_per_element: int = 4,
) -> int:
    if requested_chunk_size > 0:
        return min(requested_chunk_size, vocab_size)

    global _CHUNK_BUDGET
    if _CHUNK_BUDGET is None:
        _CHUNK_BUDGET = _get_memory_budget()

    # Goal: choose chunk_v so the MLX scheduler can free each chunk's logit
    # tensor before the next chunk is computed, while keeping chunks large
    # enough for efficient GEMM.
    #
    # Strategy: target 16 chunks as the baseline granularity.  The per-chunk
    # byte budget (derived from hardware memory) caps how large each chunk can
    # be.  On large-memory devices the cap is ~100 MB; on small devices it
    # scales down to force more aggressive chunking.
    #
    # Constraints:
    #   - min 2048 vocab entries per chunk  (GEMM efficiency floor)
    #   - target 16 chunks                 (scheduler granularity sweet spot)
    #   - per-chunk bytes ≤ hw budget       (adapts to device memory, ≤128 MB)

    min_chunk_v = 2048
    target_chunks = 16

    # Compute chunk_v from target chunk count
    chunk_v = (vocab_size + target_chunks - 1) // target_chunks

    # Enforce per-chunk byte limit (adapts to hardware)
    chunk_bytes = n_tokens * chunk_v * bytes_per_element
    if chunk_bytes > _CHUNK_BUDGET and chunk_v > min_chunk_v:
        chunk_v = max(min_chunk_v, _CHUNK_BUDGET // (max(1, n_tokens) * bytes_per_element))

    # Align to 256 for Metal efficiency
    chunk_v = max(min_chunk_v, (chunk_v // 256) * 256)
    return min(chunk_v, vocab_size)


def _normalize_label_smoothing(value) -> float:
    """Shared domain check for every loss entry point: finite real 0<=eps<=1."""
    import numbers

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(
            f"label_smoothing must be a real number in [0, 1], got {value!r}"
        )
    try:
        eps = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f"label_smoothing must be a real number in [0, 1], got {value!r}"
        )
    if not (eps == eps and 0.0 <= eps <= 1.0):
        raise ValueError(
            f"label_smoothing must be a finite value in [0, 1], got {value!r}"
        )
    return eps


def _apply_softcap(logits: mx.array, logit_softcap: float) -> mx.array:
    if logit_softcap <= 0.0:
        return logits
    softcap = mx.array(logit_softcap, dtype=mx.float32)
    return softcap * mx.tanh(logits / softcap)


def _target_validity_masks(
    targets: mx.array,
    vocab_size: int,
    ignore_index: int,
) -> tuple[mx.array, mx.array]:
    # Cast unsigned labels to int64 first: direct `>= 0` on uint16/32/64 crashes
    # the torch-backed MLX shim ("ge_cpu" not implemented).
    _unsigned_safe_to_i64 = tuple(
        dtype for dtype in (
            getattr(mx, "uint8", None),
            getattr(mx, "uint16", None),
            getattr(mx, "uint32", None),
        )
        if dtype is not None
    )
    _uint64_dtype = getattr(mx, "uint64", None)
    if _uint64_dtype is not None and targets.dtype == _uint64_dtype:
        # uint64 -> int64: values >= 2**63 wrap negative. Route those to an
        # out-of-vocab sentinel (1<<62) so wrap artifacts (e.g. 2**64-100)
        # NaN-poison instead of colliding with ignore_index. Avoid float
        # validation: float32 loses precision above 2**24.
        targets_i64 = targets.astype(mx.int64)
        overflow = targets_i64 < 0
        invalid_sentinel = mx.array(1 << 62, dtype=mx.int64)
        targets_for_validation = mx.where(
            overflow, invalid_sentinel, targets_i64,
        )
        in_vocab = (targets_for_validation >= 0) & (
            targets_for_validation < vocab_size
        )
        not_ignored = targets_for_validation != ignore_index
        return not_ignored & in_vocab, not_ignored & ~in_vocab

    targets_for_validation = (
        targets.astype(mx.int64)
        if targets.dtype in _unsigned_safe_to_i64
        else targets
    )
    in_vocab = (targets_for_validation >= 0) & (targets_for_validation < vocab_size)
    not_ignored = targets_for_validation != ignore_index
    return not_ignored & in_vocab, not_ignored & ~in_vocab


def _poison_invalid_targets(values: mx.array, invalid: mx.array) -> mx.array:
    # mx.full (real tensor), not a 0-d scalar: a scalar bakes into the Metal
    # kernel as the literal token `nan`, which MSL rejects.
    return mx.where(
        invalid,
        mx.full(values.shape, float("nan"), dtype=values.dtype),
        values,
    )


def _chunk_matmul(
    x: mx.array,
    weight: mx.array,
    *,
    scales: mx.array | None = None,
    biases: mx.array | None = None,
    group_size: int | None = None,
    bits: int | None = None,
    mode: str = "affine",
    transpose: bool = True,
) -> mx.array:
    if scales is None:
        return x @ (weight.T if transpose else weight)
    return mx.quantized_matmul(
        x,
        weight,
        scales,
        biases=biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )


def _build_forward_update_kernel() -> Callable:
    source = """
        uint gid = thread_position_in_grid.x;
        uint row = gid / 256;
        uint n = logits_shape[0];
        if (row >= n) {
            return;
        }

        uint lid = gid % 256;
        uint tpg = 256;
        uint chunk_v = logits_shape[1];
        int base = int(row * chunk_v);
        int target = targets[row];
        int v_start = v_start_arr[0];
        int ignore_index = ignore_index_arr[0];
        float softcap = softcap_arr[0];

        threadgroup float max_buf[256];
        threadgroup float sum_buf[256];
        threadgroup float target_buf[256];
        threadgroup uint found_buf[256];

        float local_max = -INFINITY;
        for (uint col = lid; col < chunk_v; col += tpg) {
            float raw = logits[base + int(col)];
            float val = raw;
            if (softcap > 0.0f) {
                val = softcap * fast::tanh(raw / softcap);
            }
            local_max = metal::max(local_max, val);
        }

        max_buf[lid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tpg / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                max_buf[lid] = metal::max(max_buf[lid], max_buf[lid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float chunk_max = max_buf[0];

        float local_sum = 0.0f;
        float local_target = 0.0f;
        uint found_target = 0;
        for (uint col = lid; col < chunk_v; col += tpg) {
            float raw = logits[base + int(col)];
            float val = raw;
            if (softcap > 0.0f) {
                val = softcap * fast::tanh(raw / softcap);
            }
            local_sum += fast::exp(val - chunk_max);
            int global_v = v_start + int(col);
            if (global_v == target) {
                local_target = val;
                found_target = 1;
            }
        }

        sum_buf[lid] = local_sum;
        target_buf[lid] = local_target;
        found_buf[lid] = found_target;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tpg / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                sum_buf[lid] += sum_buf[lid + stride];
                if (found_buf[lid] == 0 && found_buf[lid + stride] != 0) {
                    found_buf[lid] = 1;
                    target_buf[lid] = target_buf[lid + stride];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lid == 0) {
            float old_max = running_max_in[row];
            float old_sum = running_sum_in[row];
            float new_max = metal::max(old_max, chunk_max);
            float new_sum = old_sum * fast::exp(old_max - new_max) +
                            sum_buf[0] * fast::exp(chunk_max - new_max);

            running_max_out[row] = new_max;
            running_sum_out[row] = new_sum;
            if (target != ignore_index && found_buf[0] != 0) {
                target_out[row] = target_buf[0];
            } else {
                target_out[row] = target_in[row];
            }
        }
    """

    return mx.fast.metal_kernel(
        name="cce_runtime_forward_update_clean",
        input_names=[
            "logits",
            "targets",
            "running_max_in",
            "running_sum_in",
            "target_in",
            "v_start_arr",
            "ignore_index_arr",
            "softcap_arr",
        ],
        output_names=["running_max_out", "running_sum_out", "target_out"],
        source=source,
        ensure_row_contiguous=True,
    )


# INVARIANT: kernel emits finite lse_out/loss_out for every row (no vocab_size).
# Callers MUST _poison_invalid_targets on loss and lse before the dlogits kernel,
# else invalid rows silently get finite wrong gradients.
def _build_forward_update_finalize_kernel() -> Callable:
    source = """
        uint gid = thread_position_in_grid.x;
        uint row = gid / 256;
        uint n = logits_shape[0];
        if (row >= n) {
            return;
        }

        uint lid = gid % 256;
        uint tpg = 256;
        uint chunk_v = logits_shape[1];
        int base = int(row * chunk_v);
        int target = targets[row];
        int v_start = v_start_arr[0];
        int ignore_index = ignore_index_arr[0];
        float softcap = softcap_arr[0];

        threadgroup float max_buf[256];
        threadgroup float sum_buf[256];
        threadgroup float target_buf[256];
        threadgroup uint found_buf[256];

        float local_max = -INFINITY;
        for (uint col = lid; col < chunk_v; col += tpg) {
            float raw = logits[base + int(col)];
            float val = raw;
            if (softcap > 0.0f) {
                val = softcap * fast::tanh(raw / softcap);
            }
            local_max = metal::max(local_max, val);
        }

        max_buf[lid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tpg / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                max_buf[lid] = metal::max(max_buf[lid], max_buf[lid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float chunk_max = max_buf[0];

        float local_sum = 0.0f;
        float local_target = 0.0f;
        uint found_target = 0;
        for (uint col = lid; col < chunk_v; col += tpg) {
            float raw = logits[base + int(col)];
            float val = raw;
            if (softcap > 0.0f) {
                val = softcap * fast::tanh(raw / softcap);
            }
            local_sum += fast::exp(val - chunk_max);
            int global_v = v_start + int(col);
            if (global_v == target) {
                local_target = val;
                found_target = 1;
            }
        }

        sum_buf[lid] = local_sum;
        target_buf[lid] = local_target;
        found_buf[lid] = found_target;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tpg / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                sum_buf[lid] += sum_buf[lid + stride];
                if (found_buf[lid] == 0 && found_buf[lid + stride] != 0) {
                    found_buf[lid] = 1;
                    target_buf[lid] = target_buf[lid + stride];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lid == 0) {
            float old_max = running_max_in[row];
            float old_sum = running_sum_in[row];
            float new_max = metal::max(old_max, chunk_max);
            float new_sum = old_sum * fast::exp(old_max - new_max) +
                            sum_buf[0] * fast::exp(chunk_max - new_max);
            float new_target = target_in[row];
            if (target != ignore_index && found_buf[0] != 0) {
                new_target = target_buf[0];
            }

            running_max_out[row] = new_max;
            running_sum_out[row] = new_sum;
            target_out[row] = new_target;

            float lse = new_max + fast::log(new_sum + 1e-9f);
            lse_out[row] = lse;
            if (target == ignore_index) {
                loss_out[row] = 0.0f;
            } else {
                loss_out[row] = lse - new_target;
            }
        }
    """

    return mx.fast.metal_kernel(
        name="cce_runtime_forward_update_finalize_clean",
        input_names=[
            "logits",
            "targets",
            "running_max_in",
            "running_sum_in",
            "target_in",
            "v_start_arr",
            "ignore_index_arr",
            "softcap_arr",
        ],
        output_names=["running_max_out", "running_sum_out", "target_out", "loss_out", "lse_out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _build_dlogits_kernel() -> Callable:
    source = """
        uint tid = thread_position_in_grid.x;
        const uint N_READS = 4;
        uint n = logits_shape[0];
        uint chunk_v = logits_shape[1];
        uint total = n * chunk_v;
        uint base_elem = tid * N_READS;
        if (base_elem >= total) {
            return;
        }

        int ignore_index = ignore_index_arr[0];
        int v_start = v_start_arr[0];
        float softcap = softcap_arr[0];
        for (uint i = 0; i < N_READS; i++) {
            uint elem = base_elem + i;
            if (elem >= total) {
                continue;
            }

            uint row = elem / chunk_v;
            uint col = elem % chunk_v;
            int target = targets[row];

            // Invalid rows arrive with lse=NaN. fast::exp() is not IEEE-754
            // strict (MSL 6.5.1) so emit NaN explicitly via 0/0. Must run
            // before the ignore_index check so wide invalid labels that
            // narrow to ignore_index still get NaN, not zero.
            if (isnan(lse[row])) {
                d_logits[elem] = static_cast<O>(0.0f / 0.0f);
                continue;
            }

            if (target == ignore_index) {
                d_logits[elem] = static_cast<O>(0.0f);
                continue;
            }

            int global_v = v_start + int(col);
            float raw = logits[elem];
            float capped = raw;
            if (softcap > 0.0f) {
                capped = softcap * fast::tanh(raw / softcap);
            }

            float prob = fast::exp(capped - lse[row]);
            float grad = (prob - float(global_v == target)) * grad_output[row];
            if (softcap > 0.0f) {
                float t = fast::tanh(raw / softcap);
                grad *= (1.0f - t * t);
            }
            d_logits[elem] = static_cast<O>(grad);
        }
    """

    return mx.fast.metal_kernel(
        name="cce_runtime_dlogits_clean",
        input_names=[
            "logits",
            "lse",
            "targets",
            "grad_output",
            "v_start_arr",
            "ignore_index_arr",
            "softcap_arr",
        ],
        output_names=["d_logits"],
        source=source,
        ensure_row_contiguous=True,
    )


def _with_single_simd_forward(fallback: Callable, finalize: bool) -> Callable:
    source = """
        uint gid = thread_position_in_grid.x;
        uint row = gid / 32;
        uint n = logits_shape[0];
        if (row >= n) {
            return;
        }

        uint lid = gid % 32;
        uint tpg = 32;
        uint chunk_v = logits_shape[1];
        int base = int(row * chunk_v);
        int target = targets[row];
        int v_start = v_start_arr[0];
        int ignore_index = ignore_index_arr[0];
        float local_max = -INFINITY;
        for (uint col = lid; col < chunk_v; col += tpg) {
            float raw = logits[base + int(col)];
            local_max = metal::max(local_max, raw);
        }
        float chunk_max = simd_max(local_max);
        float local_sum = 0.0f;
        for (uint col = lid; col < chunk_v; col += tpg) {
            float raw = logits[base + int(col)];
            local_sum += fast::exp(raw - chunk_max);
        }
        float chunk_sum = simd_sum(local_sum);
        bool found_target = target >= v_start && target < v_start + int(chunk_v);
        float chunk_target = 0.0f;
        if (lid == 0 && target != ignore_index && found_target) {
            chunk_target = logits[base + target - v_start];
        }
        if (lid == 0) {
            float old_max = running_max_in[row];
            float old_sum = running_sum_in[row];
            float new_max = metal::max(old_max, chunk_max);
            float new_sum = old_sum * fast::exp(old_max - new_max) +
                            chunk_sum * fast::exp(chunk_max - new_max);
            float new_target = target_in[row];
            if (target != ignore_index && found_target) {
                new_target = chunk_target;
            }

            running_max_out[row] = new_max;
            running_sum_out[row] = new_sum;
            target_out[row] = new_target;
    """
    if finalize:
        source += """
            float lse = new_max + fast::log(new_sum + 1e-9f);
            lse_out[row] = lse;
            if (target == ignore_index) {
                loss_out[row] = 0.0f;
            } else {
                loss_out[row] = lse - new_target;
            }
    """
    source += "    }\n"
    output_names = ["running_max_out", "running_sum_out", "target_out"]
    if finalize:
        output_names += ["loss_out", "lse_out"]
    kernel = mx.fast.metal_kernel(
        name="cce_runtime_forward_single_simd_" + str(finalize),
        input_names=["logits", "targets", "running_max_in", "running_sum_in",
                     "target_in", "v_start_arr", "ignore_index_arr", "softcap_arr"],
        output_names=output_names, source=source, ensure_row_contiguous=True,
    )

    def call(**kwargs):
        logits = kwargs["inputs"][0]
        rows, width = logits.shape
        # Small grids need more SIMD groups; wide rows need more lanes per row.
        if rows >= 256 and width <= 2048 and logits.dtype in (mx.float16, mx.bfloat16):
            kwargs["grid"] = (rows * 32, 1, 1)
            kwargs["threadgroup"] = (32, 1, 1)
            return kernel(**kwargs)
        return fallback(**kwargs)
    return call


def _build_kernel_set(
    logit_softcap: float = 0.0,
) -> tuple[Callable | None, Callable | None, Callable | None]:
    if not mx.metal.is_available():
        return None, None, None

    update = _build_forward_update_kernel()
    finalize = _build_forward_update_finalize_kernel()
    if logit_softcap <= 0.0:
        update = _with_single_simd_forward(update, False)
        finalize = _with_single_simd_forward(finalize, True)
    return update, finalize, _build_dlogits_kernel()


def _forward_chunked_fused_finalize(
    hidden: mx.array,
    weight: mx.array,
    targets: mx.array,
    *,
    scales: mx.array | None,
    biases: mx.array | None,
    group_size: int | None,
    bits: int | None,
    mode: str,
    ignore_index: int,
    logit_softcap: float,
    chunk_size: int,
    forward_update_kernel: Callable | None,
    forward_update_finalize_kernel: Callable | None,
    label_smoothing: float = 0.0,
) -> tuple[mx.array, mx.array]:
    hidden_compute = hidden
    weight_compute = weight
    # Validate in the original dtype so wide ints can't wrap into a valid class
    # id or ignore_index after the int32 narrow.
    targets_raw = targets

    n, _ = hidden_compute.shape
    vocab_size = weight_compute.shape[0]
    # Reject rank-2 targets up front; they slip past the length check and crash the kernels.
    if len(targets_raw.shape) != 1:
        raise ValueError(
            "MLX CCE: targets must be a flat 1D vector "
            f"(hidden.shape={hidden_compute.shape}, targets.shape={targets_raw.shape})."
        )
    if n == 0:
        # Surface upstream shape mismatch instead of silently dropping labels.
        if targets_raw.shape[0] != 0:
            raise ValueError(
                "MLX CCE: hidden has 0 tokens but targets is non-empty "
                f"(targets.shape={targets_raw.shape})."
            )
        # Separate allocations so the VJP can't alias loss into lse.
        return (
            mx.zeros((0,), dtype=mx.float32),
            mx.zeros((0,), dtype=mx.float32),
        )
    if targets_raw.shape[0] != n:
        raise ValueError(
            "MLX CCE: targets length does not match hidden token count "
            f"(hidden.shape={hidden_compute.shape}, targets.shape={targets_raw.shape})."
        )
    valid_pre, invalid_pre = _target_validity_masks(
        targets_raw, vocab_size, ignore_index,
    )
    targets = targets_raw.astype(mx.int32)
    compute_bytes = 2 if hidden_compute.dtype in (mx.float16, mx.bfloat16) else 4
    if label_smoothing > 0.0:
        compute_bytes = 4  # smoothing casts each logits chunk to fp32
    chunk_size = _resolve_chunk_size(
        chunk_size,
        n,
        vocab_size,
        bytes_per_element=compute_bytes,
    )
    running_max = mx.full((n,), -mx.inf, dtype=mx.float32)
    running_sum_exp = mx.zeros((n,), dtype=mx.float32)
    target_logit = mx.zeros((n,), dtype=mx.float32)
    # HF LabelSmoother accumulates the smoothed vocabulary term in float32.
    sum_capped = mx.zeros((n,), dtype=mx.float32) if label_smoothing > 0.0 else None

    if forward_update_kernel is None or forward_update_finalize_kernel is None:
        for v_start in range(0, vocab_size, chunk_size):
            v_end = min(v_start + chunk_size, vocab_size)
            w_chunk = weight_compute[v_start:v_end]
            scales_chunk = None if scales is None else scales[v_start:v_end]
            biases_chunk = None if biases is None else biases[v_start:v_end]
            logits = _chunk_matmul(
                hidden_compute,
                w_chunk,
                scales=scales_chunk,
                biases=biases_chunk,
                group_size=group_size,
                bits=bits,
                mode=mode,
            )
            logits = _apply_softcap(logits, logit_softcap)
            if sum_capped is not None:
                # eps>0 always takes this python path (kernels disabled), so
                # match the Metal kernels' float32 LSE accumulation here.
                logits = logits.astype(mx.float32)

            chunk_max = mx.max(logits, axis=-1)
            chunk_sum_exp = mx.sum(mx.exp(logits - mx.expand_dims(chunk_max, -1)), axis=-1)

            new_max = mx.maximum(running_max, chunk_max)
            running_sum_exp = running_sum_exp * mx.exp(running_max - new_max)
            running_sum_exp = running_sum_exp + chunk_sum_exp * mx.exp(chunk_max - new_max)
            running_max = new_max

            in_chunk = (targets >= v_start) & (targets < v_end)
            local_targets = mx.clip(targets - v_start, 0, v_end - v_start - 1)
            chunk_target = mx.take_along_axis(logits, mx.expand_dims(local_targets, -1), axis=1).squeeze(-1)
            target_logit = mx.where(in_chunk, chunk_target, target_logit)
            if sum_capped is not None:
                # logits is already float32 here (cast above under the same guard)
                sum_capped = sum_capped + logits.sum(axis=-1)

        lse = running_max + mx.log(running_sum_exp + 1e-9)
        if sum_capped is not None:
            # loss = lse - (1-eps)*target - eps*mean_v(logits): equals
            # (1-eps)*NLL + eps*uniform smoothing (HF LabelSmoother form).
            eps = label_smoothing
            token_loss = lse - (1.0 - eps) * target_logit - eps * (sum_capped / vocab_size)
        else:
            token_loss = lse - target_logit
        loss = mx.where(valid_pre, token_loss, mx.zeros_like(lse))
        loss = _poison_invalid_targets(loss, invalid_pre)
        lse = _poison_invalid_targets(lse, invalid_pre)
        return loss, lse

    ignore_arr = mx.array([ignore_index], dtype=mx.int32)
    softcap_arr = mx.array([logit_softcap], dtype=mx.float32)
    chunk_starts = [mx.array([v_start], dtype=mx.int32) for v_start in range(0, vocab_size, chunk_size)]
    last_chunk_idx = len(chunk_starts) - 1

    for v_start in range(0, vocab_size, chunk_size):
        chunk_idx = v_start // chunk_size
        v_end = min(v_start + chunk_size, vocab_size)
        w_chunk = weight_compute[v_start:v_end]
        scales_chunk = None if scales is None else scales[v_start:v_end]
        biases_chunk = None if biases is None else biases[v_start:v_end]
        logits = _chunk_matmul(
            hidden_compute,
            w_chunk,
            scales=scales_chunk,
            biases=biases_chunk,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )

        if chunk_idx == last_chunk_idx:
            _, _, _, loss, lse = forward_update_finalize_kernel(
                inputs=[
                    logits,
                    targets,
                    running_max,
                    running_sum_exp,
                    target_logit,
                    chunk_starts[chunk_idx],
                    ignore_arr,
                    softcap_arr,
                ],
                output_shapes=[running_max.shape, running_sum_exp.shape, target_logit.shape, (n,), (n,)],
                output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32, mx.float32],
                grid=(n * 256, 1, 1),
                threadgroup=(256, 1, 1),
            )
            loss = _poison_invalid_targets(loss, invalid_pre)
            lse = _poison_invalid_targets(lse, invalid_pre)
            return loss, lse

        running_max, running_sum_exp, target_logit = forward_update_kernel(
            inputs=[
                logits,
                targets,
                running_max,
                running_sum_exp,
                target_logit,
                chunk_starts[chunk_idx],
                ignore_arr,
                softcap_arr,
            ],
            output_shapes=[running_max.shape, running_sum_exp.shape, target_logit.shape],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
            grid=(n * 256, 1, 1),
            threadgroup=(256, 1, 1),
        )

    raise RuntimeError("Unreachable: fused finalize path did not return outputs.")


def _forward_with_hidden_gradient(
    hidden: mx.array,
    weight: mx.array,
    targets: mx.array,
    *,
    scales: mx.array | None,
    biases: mx.array | None,
    group_size: int | None,
    bits: int | None,
    mode: str,
    ignore_index: int,
    logit_softcap: float,
    chunk_size: int,
    forward_update_kernel: Callable,
    forward_update_finalize_kernel: Callable,
    dlogits_kernel: Callable,
) -> tuple[mx.array, mx.array]:
    """Token losses and each token's loss gradient per unit cotangent.

    With a frozen head the hidden gradient is ``cotangent * (sum_v p_v s'_v w_v -
    s'_y w_y)``, and the bracket does not depend on the cotangent. It is built here
    one vocabulary chunk at a time: each chunk's softmax weights are taken against
    the logsumexp so far, and the running sum is rescaled as that logsumexp grows.
    No vocabulary-wide buffer outlives its chunk, and the backward is a product.
    """
    n, dim = hidden.shape
    vocab_size = weight.shape[0]
    if n == 0 or len(targets.shape) != 1 or targets.shape[0] != n:
        # Shape validation and the empty batch, exactly as the loss-only forward.
        loss, _ = _forward_chunked_fused_finalize(
            hidden, weight, targets, scales=scales, biases=biases, group_size=group_size,
            bits=bits, mode=mode, ignore_index=ignore_index, logit_softcap=logit_softcap,
            chunk_size=chunk_size, forward_update_kernel=None, forward_update_finalize_kernel=None,
        )
        return loss, mx.zeros((n, dim), dtype=hidden.dtype)
    valid, invalid = _target_validity_masks(targets, vocab_size, ignore_index)
    targets = targets.astype(mx.int32)
    ignore_arr = mx.array([ignore_index], dtype=mx.int32)
    softcap_arr = mx.array([logit_softcap], dtype=mx.float32)
    # A start no target can reach leaves only the softmax term of the derivative.
    unreachable_start = mx.array([-(1 << 30)], dtype=mx.int32)
    unit_cotangent = mx.ones((n,), dtype=mx.float32)
    quantization = dict(group_size=group_size, bits=bits, mode=mode)

    running_max = mx.full((n,), -mx.inf, dtype=mx.float32)
    running_sum_exp = mx.zeros((n,), dtype=mx.float32)
    target_logit = mx.zeros((n,), dtype=mx.float32)
    lse = mx.full((n,), -mx.inf, dtype=mx.float32)
    weighted_rows = mx.zeros((n, dim), dtype=mx.float32)
    for v_start in range(0, vocab_size, chunk_size):
        v_end = min(v_start + chunk_size, vocab_size)
        w_chunk = weight[v_start:v_end]
        scales_chunk = None if scales is None else scales[v_start:v_end]
        biases_chunk = None if biases is None else biases[v_start:v_end]
        logits = _chunk_matmul(hidden, w_chunk, scales=scales_chunk, biases=biases_chunk, **quantization)
        inputs = [
            logits, targets, running_max, running_sum_exp, target_logit,
            mx.array([v_start], dtype=mx.int32), ignore_arr, softcap_arr,
        ]
        previous_lse = lse
        if v_end == vocab_size:
            # The loss-only forward's finalize kernel, so losses match it bit for bit.
            running_max, running_sum_exp, target_logit, loss, lse = forward_update_finalize_kernel(
                inputs=inputs,
                output_shapes=[(n,)] * 5,
                output_dtypes=[mx.float32] * 5,
                grid=(n * 256, 1, 1),
                threadgroup=(256, 1, 1),
            )
        else:
            running_max, running_sum_exp, target_logit = forward_update_kernel(
                inputs=inputs,
                output_shapes=[(n,)] * 3,
                output_dtypes=[mx.float32] * 3,
                grid=(n * 256, 1, 1),
                threadgroup=(256, 1, 1),
            )
            lse = running_max + mx.log(running_sum_exp + 1e-9)
        n_reads = 4
        probs = dlogits_kernel(
            inputs=[logits, lse, targets, unit_cotangent, unreachable_start, ignore_arr, softcap_arr],
            output_shapes=[logits.shape],
            output_dtypes=[logits.dtype],
            template=[("O", logits.dtype)],
            grid=((logits.size + n_reads - 1) // n_reads, 1, 1),
            threadgroup=(256, 1, 1),
        )[0]
        chunk_rows = _chunk_matmul(
            probs.astype(hidden.dtype), w_chunk, scales=scales_chunk, biases=biases_chunk,
            transpose=False, **quantization,
        )
        weighted_rows = mx.exp(previous_lse - lse)[:, None] * weighted_rows + chunk_rows.astype(mx.float32)
        # Tie each chunk's consumers to its logits so the chunk is freed before the next.
        weighted_rows, lse, target_logit = mx.depends([weighted_rows, lse, target_logit], [probs])

    safe_targets = mx.where(valid, targets, 0)
    if scales is None:
        target_rows = weight[safe_targets].astype(mx.float32)
    else:
        target_rows = mx.dequantize(
            weight[safe_targets], scales[safe_targets],
            None if biases is None else biases[safe_targets], **quantization,
        ).astype(mx.float32)
    if logit_softcap > 0.0:
        # target_logit is capped, so its tanh is target_logit / softcap.
        target_rows = target_rows * (1.0 - mx.square(target_logit / logit_softcap))[:, None]
    loss = mx.where(valid, loss, 0.0)
    # Held until the backward, so store it at the precision the hidden gradient gets.
    gradient = mx.where(valid[:, None], weighted_rows - target_rows, 0.0).astype(hidden.dtype)
    return _poison_invalid_targets(loss, invalid), _poison_invalid_targets(gradient, invalid[:, None])


# Requires lse pre-poisoned with NaN for invalid rows: this fallback does not
# re-check vocab bounds and relies on NaN propagation for the gradient.
def _fallback_dlogits(
    logits: mx.array,
    lse: mx.array,
    targets: mx.array,
    grad_output: mx.array,
    *,
    v_start: int,
    v_end: int,
    ignore_index: int,
    logit_softcap: float,
    label_smoothing: float = 0.0,
    vocab_size: int = 0,
) -> mx.array:
    if label_smoothing > 0.0 and vocab_size <= 0:
        raise ValueError("vocab_size must be positive when label_smoothing > 0")
    capped = _apply_softcap(logits, logit_softcap)
    probs = mx.exp(capped - mx.expand_dims(lse, -1))

    local_targets = targets - v_start
    target_mask = mx.expand_dims(local_targets, -1) == mx.arange(v_end - v_start)
    valid = (targets >= v_start) & (targets < v_end) & (targets != ignore_index)
    target_mask = target_mask & mx.expand_dims(valid, -1)

    if label_smoothing > 0.0:
        # d/dlogit_v of the smoothed loss: p_v - (1-eps)*onehot_v - eps/V.
        d_capped = (
            probs
            - (1.0 - label_smoothing) * target_mask.astype(mx.float32)
            - label_smoothing / vocab_size
        )
    else:
        d_capped = probs - target_mask.astype(mx.float32)
    d_capped = d_capped * mx.expand_dims(grad_output, -1)

    if logit_softcap > 0.0:
        softcap = mx.array(logit_softcap, dtype=mx.float32)
        t = mx.tanh(logits / softcap)
        d_capped = d_capped * (1.0 - t * t)

    # NaN grad on lse-NaN rows BEFORE the ignore_index mask, so wide invalid
    # labels that narrow to ignore_index do not silently zero-grad.
    invalid_lse = mx.isnan(lse)
    nan_grad = mx.full(d_capped.shape, float("nan"), dtype=d_capped.dtype)
    d_capped = mx.where(mx.expand_dims(invalid_lse, -1), nan_grad, d_capped)

    ignore_mask = (targets == ignore_index) & ~invalid_lse
    return mx.where(mx.expand_dims(ignore_mask, -1), mx.zeros_like(d_capped), d_capped)


def make_runtime_cce_loss_fused_finalize(
    *,
    ignore_index: int,
    logit_softcap: float,
    chunk_size: int,
    quantized: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: str = "affine",
    label_smoothing: float = 0.0,
    weight_is_frozen: bool = False,
    precompute_hidden_gradient: bool = False,
):
    label_smoothing = _normalize_label_smoothing(label_smoothing)
    forward_update_kernel, forward_update_finalize_kernel, dlogits_kernel = _build_kernel_set(
        logit_softcap,
    )
    if label_smoothing > 0.0:
        # Smoothing lives in the chunked python path; the fused Metal kernels
        # do not carry the vocabulary-sum term. eps=0 keeps the kernel path.
        forward_update_kernel = forward_update_finalize_kernel = dlogits_kernel = None
    use_metal_kernel = dlogits_kernel is not None
    ignore_arr = mx.array([ignore_index], dtype=mx.int32)
    softcap_arr = mx.array([logit_softcap], dtype=mx.float32)
    chunk_plan_cache: OrderedDict[
        tuple,
        tuple[int, tuple[int, ...], tuple[mx.array, ...], tuple[mx.array, ...]],
    ] = OrderedDict()
    cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
    quantized_layout = (
        bool(quantized),
        group_size if quantized else None,
        bits if quantized else None,
        mode if quantized else None,
    )

    def get_chunk_plan(
        hidden: mx.array,
        weight: mx.array,
    ) -> tuple[int, tuple[int, ...], tuple[mx.array, ...], tuple[mx.array, ...]]:
        n_tokens = hidden.shape[0]
        vocab_size = weight.shape[0]
        compute_bytes = 2 if hidden.dtype in (mx.float16, mx.bfloat16) else 4
        if label_smoothing > 0.0:
            compute_bytes = 4  # smoothing casts each logits chunk to fp32
        resolved_chunk_size = _resolve_chunk_size(
            chunk_size,
            n_tokens,
            vocab_size,
            bytes_per_element=compute_bytes,
        )
        # A vocabulary small enough that the default 16-chunk split lands under
        # 4096 gives the GEMM too little work per launch. Widening to 4096 is
        # worth 1.01-1.47x on the backward for such heads, but only while every
        # buffer it grows stays small: the token-side buffers, the weight slice
        # (hidden), and the classifier gradient when the head is trained.
        # Each is held to 8 MB, which is what confines this to compact heads.
        #
        # A trainable bfloat16 head on the kernel path is the one case where the token
        # side is not one buffer: dlogits_out_dtype below writes d_logits in float32,
        # and the hidden GEMM then needs it cast back, so logits + d_logits + the cast
        # are live at once, 4x what counting the logits chunk alone allowed. Promoting
        # there costs memory rather than saving it: measured on an M1 at n_tokens=512,
        # hidden=512, vocab=16384, the 4096 plan peaked at 146324012 bytes against
        # 143211072 for 2048. Everywhere else the derivative matches the logits dtype
        # and the original single-buffer bound is what applies. Label smoothing is
        # excluded because it disables the kernels, so dlogits_out_dtype never runs and
        # compute_bytes is already 4.
        promoted_chunk = 4096
        promoted_bytes = promoted_chunk * compute_bytes
        token_bytes = compute_bytes
        if label_smoothing == 0.0 and hidden.dtype == mx.bfloat16 and not weight_is_frozen:
            token_bytes = compute_bytes + 4 + compute_bytes
        if (chunk_size <= 0 and not quantized
                and hidden.dtype == weight.dtype and hidden.dtype in (mx.bfloat16, mx.float32)
                and n_tokens >= 256 and resolved_chunk_size < promoted_chunk
                and vocab_size >= 16384
                and n_tokens * promoted_chunk * token_bytes <= min(8 * 1024 * 1024, _CHUNK_BUDGET)
                and hidden.shape[1] * promoted_bytes <= 8 * 1024 * 1024
                and (weight_is_frozen
                     or hidden.shape[1] * promoted_chunk * 4 <= 8 * 1024 * 1024)):
            resolved_chunk_size = promoted_chunk
        key = (
            vocab_size,
            resolved_chunk_size,
            hidden.dtype,
            quantized_layout,
        )
        if key in chunk_plan_cache:
            cache_stats["hits"] += 1
            chunk_plan_cache.move_to_end(key)
            return chunk_plan_cache[key]

        cache_stats["misses"] += 1
        starts = tuple(range(0, vocab_size, resolved_chunk_size))
        start_arrays = tuple(
            mx.array([v_start], dtype=mx.int32) for v_start in starts
        )
        weight_start_arrays = (
            ()
            if quantized
            else tuple(
                mx.array([v_start, 0], dtype=mx.int32)
                for v_start in starts
            )
        )
        chunk_plan_cache[key] = (
            resolved_chunk_size,
            starts,
            start_arrays,
            weight_start_arrays,
        )
        if len(chunk_plan_cache) > _CHUNK_PLAN_CACHE_MAX_ENTRIES:
            chunk_plan_cache.popitem(last=False)
            cache_stats["evictions"] += 1
        return chunk_plan_cache[key]

    def get_chunk_plan_cache_info():
        return {
            "entries": len(chunk_plan_cache),
            "max_entries": _CHUNK_PLAN_CACHE_MAX_ENTRIES,
            "hits": cache_stats["hits"],
            "misses": cache_stats["misses"],
            "evictions": cache_stats["evictions"],
        }

    if precompute_hidden_gradient and use_metal_kernel and label_smoothing == 0.0 and (quantized or weight_is_frozen):
        @mx.custom_function
        def hidden_gradient_loss_full(hidden, weight, scales, biases, targets):
            return _forward_with_hidden_gradient(
                hidden,
                weight,
                targets,
                scales=scales,
                biases=biases,
                group_size=group_size,
                bits=bits,
                mode=mode,
                ignore_index=ignore_index,
                logit_softcap=logit_softcap,
                chunk_size=get_chunk_plan(hidden, weight)[0],
                forward_update_kernel=forward_update_kernel,
                forward_update_finalize_kernel=forward_update_finalize_kernel,
                dlogits_kernel=dlogits_kernel,
            )

        @hidden_gradient_loss_full.vjp
        def hidden_gradient_loss_vjp(primals, cotangents, outputs):
            hidden = primals[0]
            grad_output = cotangents[0] if isinstance(cotangents, (tuple, list)) else cotangents
            if grad_output is None:
                grad_hidden = mx.zeros_like(hidden)
            else:
                grad_hidden = (grad_output.astype(mx.float32)[:, None] * outputs[1]).astype(hidden.dtype)
            return (grad_hidden, *(None if p is None else mx.zeros_like(p) for p in primals[1:]))

        def hidden_gradient_loss(hidden, weight, *rest):
            scales, biases, targets = rest if quantized else (None, None, rest[0])
            losses, gradient = hidden_gradient_loss_full(hidden, weight, scales, biases, targets)
            # Same guard as the loss-only path: keep the VJP's output live under mx.compile.
            return losses + gradient[:, 0] * mx.array(0.0, dtype=mx.float32)

        hidden_gradient_loss._unsloth_chunk_plan_cache_info = get_chunk_plan_cache_info
        return hidden_gradient_loss, use_metal_kernel

    if quantized:
        @mx.custom_function
        def runtime_cce_loss_full(
            hidden: mx.array,
            weight: mx.array,
            scales: mx.array,
            biases: mx.array,
            targets: mx.array,
        ):
            losses, lse = _forward_chunked_fused_finalize(
                hidden,
                weight,
                targets,
                scales=scales,
                biases=biases,
                group_size=group_size,
                bits=bits,
                mode=mode,
                ignore_index=ignore_index,
                logit_softcap=logit_softcap,
                chunk_size=get_chunk_plan(hidden, weight)[0],
                forward_update_kernel=forward_update_kernel,
                forward_update_finalize_kernel=forward_update_finalize_kernel,
                label_smoothing=label_smoothing,
            )
            return losses, lse

        @runtime_cce_loss_full.vjp
        def runtime_cce_loss_vjp(primals, cotangents, outputs):
            hidden, weight, scales, biases, targets = primals
            grad_output = cotangents[0] if isinstance(cotangents, tuple) else cotangents

            hidden_compute = hidden
            weight_compute = weight
            targets32 = targets.astype(mx.int32)
            if hidden_compute.shape[0] == 0:
                return (
                    mx.zeros_like(hidden),
                    mx.zeros_like(weight),
                    mx.zeros_like(scales),
                    mx.zeros_like(biases),
                    mx.zeros_like(targets),
                )
            if grad_output is None:
                grad_output = mx.zeros_like(outputs[0])
            grad_output32 = grad_output.astype(mx.float32)

            resolved_chunk_size, chunk_starts_int, chunk_starts_arr, _ = get_chunk_plan(hidden, weight)
            vocab_size = weight_compute.shape[0]
            lse = outputs[1].astype(mx.float32)

            grad_hidden = mx.zeros_like(hidden_compute)
            n_reads = 4

            for chunk_idx, v_start in enumerate(chunk_starts_int):
                v_end = min(v_start + resolved_chunk_size, vocab_size)
                weight_chunk = weight_compute[v_start:v_end]
                scales_chunk = scales[v_start:v_end]
                biases_chunk = None if biases is None else biases[v_start:v_end]

                logits = _chunk_matmul(
                    hidden_compute,
                    weight_chunk,
                    scales=scales_chunk,
                    biases=biases_chunk,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                )

                if dlogits_kernel is not None:
                    total_threads = (logits.size + n_reads - 1) // n_reads
                    dlogits_out_dtype = logits.dtype
                    d_logits = dlogits_kernel(
                        inputs=[
                            logits,
                            lse,
                            targets32,
                            grad_output32,
                            chunk_starts_arr[chunk_idx],
                            ignore_arr,
                            softcap_arr,
                        ],
                        output_shapes=[logits.shape],
                        output_dtypes=[dlogits_out_dtype],
                        template=[("O", dlogits_out_dtype)],
                        grid=(total_threads, 1, 1),
                        threadgroup=(256, 1, 1),
                    )[0]
                else:
                    d_logits = _fallback_dlogits(
                        logits,
                        lse,
                        targets32,
                        grad_output32,
                        v_start=v_start,
                        v_end=v_end,
                        ignore_index=ignore_index,
                        logit_softcap=logit_softcap,
                        label_smoothing=label_smoothing,
                        vocab_size=vocab_size,
                    ).astype(logits.dtype)

                d_logits_compute = d_logits.astype(hidden_compute.dtype)
                grad_hidden = grad_hidden + _chunk_matmul(
                    d_logits_compute,
                    weight_chunk,
                    scales=scales_chunk,
                    biases=biases_chunk,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                    transpose=False,
                )
                grad_hidden = mx.depends([grad_hidden], [d_logits])[0]

            # Quantized weight gradients are zero: correct for LoRA (frozen LM head,
            # gradients flow only through grad_hidden). Full fine-tuning of quantized
            # models would need dequantize -> grad -> requantize, which is unsupported.
            return (
                grad_hidden.astype(hidden.dtype),
                mx.zeros_like(weight),
                mx.zeros_like(scales),
                None if biases is None else mx.zeros_like(biases),
                mx.zeros_like(targets),
            )

        def runtime_cce_loss(
            hidden: mx.array,
            weight: mx.array,
            scales: mx.array,
            biases: mx.array,
            targets: mx.array,
        ) -> mx.array:
            losses, lse = runtime_cce_loss_full(
                hidden, weight, scales, biases, targets
            )
            # Keep lse live for the custom VJP under mx.compile (read from outputs
            # during backward); zero-weight add preserves losses.
            return losses + lse * mx.array(0.0, dtype=mx.float32)

        runtime_cce_loss._unsloth_chunk_plan_cache_info = get_chunk_plan_cache_info
        return runtime_cce_loss, use_metal_kernel

    @mx.custom_function
    def runtime_cce_loss_full(hidden: mx.array, weight: mx.array, targets: mx.array):
        losses, lse = _forward_chunked_fused_finalize(
            hidden,
            weight,
            targets,
            scales=None,
            biases=None,
            group_size=None,
            bits=None,
            mode="affine",
            ignore_index=ignore_index,
            logit_softcap=logit_softcap,
            chunk_size=get_chunk_plan(hidden, weight)[0],
            forward_update_kernel=forward_update_kernel,
            forward_update_finalize_kernel=forward_update_finalize_kernel,
            label_smoothing=label_smoothing,
        )
        return losses, lse

    @runtime_cce_loss_full.vjp
    def runtime_cce_loss_vjp(primals, cotangents, outputs):
        hidden, weight, targets = primals
        grad_output = cotangents[0] if isinstance(cotangents, tuple) else cotangents

        hidden_compute = hidden
        weight_compute = weight
        targets32 = targets.astype(mx.int32)
        if hidden_compute.shape[0] == 0:
            return mx.zeros_like(hidden), mx.zeros_like(weight), mx.zeros_like(targets)
        if grad_output is None:
            grad_output = mx.zeros_like(outputs[0])
        grad_output32 = grad_output.astype(mx.float32)
        lse = outputs[1].astype(mx.float32)

        resolved_chunk_size, chunk_starts_int, chunk_starts_arr, weight_chunk_starts = get_chunk_plan(
            hidden,
            weight,
        )
        vocab_size = weight_compute.shape[0]

        grad_hidden = mx.zeros_like(hidden_compute)
        # Vocabulary chunks are disjoint; only each GEMM needs fp32 accumulation.
        grad_weight = mx.zeros(weight_compute.shape, dtype=weight.dtype)
        hidden_f32 = hidden_compute.astype(mx.float32)
        n_reads = 4

        for chunk_idx, v_start in enumerate(chunk_starts_int):
            v_end = min(v_start + resolved_chunk_size, vocab_size)
            weight_chunk = weight_compute[v_start:v_end]

            logits = hidden_compute @ weight_chunk.T

            if dlogits_kernel is not None:
                total_threads = (logits.size + n_reads - 1) // n_reads
                dlogits_out_dtype = (mx.float32 if logits.dtype == mx.bfloat16
                                     and not weight_is_frozen else logits.dtype)
                d_logits = dlogits_kernel(
                    inputs=[
                        logits,
                        lse,
                        targets32,
                        grad_output32,
                        chunk_starts_arr[chunk_idx],
                        ignore_arr,
                        softcap_arr,
                    ],
                    output_shapes=[logits.shape],
                    output_dtypes=[dlogits_out_dtype],
                    template=[("O", dlogits_out_dtype)],
                    grid=(total_threads, 1, 1),
                    threadgroup=(256, 1, 1),
                )[0]
            else:
                d_logits = _fallback_dlogits(
                    logits,
                    lse,
                    targets32,
                    grad_output32,
                    v_start=v_start,
                    v_end=v_end,
                    ignore_index=ignore_index,
                    logit_softcap=logit_softcap,
                    label_smoothing=label_smoothing,
                    vocab_size=vocab_size,
                ).astype(logits.dtype)

            d_logits_compute = d_logits.astype(hidden_compute.dtype)
            grad_hidden = grad_hidden + d_logits_compute @ weight_chunk
            # Weight gradient GEMM in float32 for accumulation precision
            d_logits_f32 = d_logits.astype(mx.float32)
            grad_weight_chunk = (d_logits_f32.T @ hidden_f32).astype(weight.dtype)
            grad_weight = mx.slice_update(
                grad_weight,
                grad_weight_chunk,
                start_indices=weight_chunk_starts[chunk_idx],
                axes=(0, 1),
            )
            if weight_is_frozen:
                # Under mx.compile the gradient sum would otherwise run as one
                # fused chain that keeps every chunk's d_logits alive.
                grad_hidden = mx.depends([grad_hidden], [d_logits])[0]

        return grad_hidden.astype(hidden.dtype), grad_weight.astype(weight.dtype), mx.zeros_like(targets)

    def runtime_cce_loss(hidden: mx.array, weight: mx.array, targets: mx.array) -> mx.array:
        losses, lse = runtime_cce_loss_full(hidden, weight, targets)
        # Return losses, but keep lse live for the custom VJP under mx.compile
        # (it reads lse from custom-function outputs during backward).
        return losses + lse * mx.array(0.0, dtype=mx.float32)

    runtime_cce_loss._unsloth_chunk_plan_cache_info = get_chunk_plan_cache_info
    return runtime_cce_loss, use_metal_kernel


def make_chunked_cross_entropy_loss(
    *,
    ignore_index: int = -100,
    logit_softcap: float = 0.0,
    chunk_size: int = 0,
    quantized: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: str = "affine",
    label_smoothing: float = 0.0,
    weight_is_frozen: bool = False,
    precompute_hidden_gradient: bool = False,
):
    """Return a standalone CCE loss and a kernel-usage flag.

    Set weight_is_frozen only when classifier gradients will not be requested.
    precompute_hidden_gradient builds a frozen or quantized head's hidden gradient
    in the forward (Metal kernels, no label smoothing), which saves backward memory
    but doubles the cost of a call that is never differentiated.
    """

    return make_runtime_cce_loss_fused_finalize(
        ignore_index=ignore_index,
        logit_softcap=logit_softcap,
        chunk_size=chunk_size,
        quantized=quantized,
        group_size=group_size,
        bits=bits,
        mode=mode,
        label_smoothing=label_smoothing,
        weight_is_frozen=weight_is_frozen,
        precompute_hidden_gradient=precompute_hidden_gradient,
    )

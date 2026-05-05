# Copyright © 2026

"""Chunked cross-entropy helpers built from MLX runtime custom kernels."""

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
    at 128 MB.  The cap ensures the scheduler always gets enough granularity
    (≥16 chunks for 128K vocab at any batch size ≤4), while the hardware
    scaling ensures small devices chunk even more aggressively.

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


def _apply_softcap(logits: mx.array, logit_softcap: float) -> mx.array:
    if logit_softcap <= 0.0:
        return logits
    softcap = mx.array(logit_softcap, dtype=mx.float32)
    return softcap * mx.tanh(logits / softcap)


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
            if (target == ignore_index) {
                d_logits[elem] = 0.0f;
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
            d_logits[elem] = grad;
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


def _build_kernel_set() -> tuple[Callable | None, Callable | None, Callable | None]:
    if not mx.metal.is_available():
        return None, None, None

    return (
        _build_forward_update_kernel(),
        _build_forward_update_finalize_kernel(),
        _build_dlogits_kernel(),
    )


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
) -> tuple[mx.array, mx.array]:
    hidden_compute = hidden
    weight_compute = weight
    targets = targets.astype(mx.int32)

    n, _ = hidden_compute.shape
    vocab_size = weight_compute.shape[0]
    compute_bytes = 2 if hidden_compute.dtype in (mx.float16, mx.bfloat16) else 4
    chunk_size = _resolve_chunk_size(
        chunk_size,
        n,
        vocab_size,
        bytes_per_element=compute_bytes,
    )
    running_max = mx.full((n,), -mx.inf, dtype=mx.float32)
    running_sum_exp = mx.zeros((n,), dtype=mx.float32)
    target_logit = mx.zeros((n,), dtype=mx.float32)

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

        lse = running_max + mx.log(running_sum_exp + 1e-9)
        valid = targets != ignore_index
        loss = mx.where(valid, lse - target_logit, mx.zeros_like(lse))
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
) -> mx.array:
    capped = _apply_softcap(logits, logit_softcap)
    probs = mx.exp(capped - mx.expand_dims(lse, -1))

    local_targets = targets - v_start
    target_mask = mx.expand_dims(local_targets, -1) == mx.arange(v_end - v_start)
    valid = (targets >= v_start) & (targets < v_end) & (targets != ignore_index)
    target_mask = target_mask & mx.expand_dims(valid, -1)

    d_capped = probs - target_mask.astype(mx.float32)
    d_capped = d_capped * mx.expand_dims(grad_output, -1)

    if logit_softcap > 0.0:
        softcap = mx.array(logit_softcap, dtype=mx.float32)
        t = mx.tanh(logits / softcap)
        d_capped = d_capped * (1.0 - t * t)

    ignore_mask = targets == ignore_index
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
):
    forward_update_kernel, forward_update_finalize_kernel, dlogits_kernel = _build_kernel_set()
    use_metal_kernel = dlogits_kernel is not None

    ignore_arr = mx.array([ignore_index], dtype=mx.int32)
    softcap_arr = mx.array([logit_softcap], dtype=mx.float32)
    chunk_plan_cache: dict[
        tuple,
        tuple[int, list[int], list[mx.array], list[mx.array]],
    ] = {}

    def get_chunk_plan(
        hidden: mx.array,
        weight: mx.array,
    ) -> tuple[int, list[int], list[mx.array], list[mx.array]]:
        n_tokens = hidden.shape[0]
        vocab_size = weight.shape[0]
        # why: chunk size depends on dtype-derived compute_bytes; without dtype
        # in the key, a bf16-cached plan is reused for fp32 and OOMs.
        key = (n_tokens, vocab_size, hidden.dtype)
        if key in chunk_plan_cache:
            return chunk_plan_cache[key]

        compute_bytes = 2 if hidden.dtype in (mx.float16, mx.bfloat16) else 4
        resolved_chunk_size = _resolve_chunk_size(
            chunk_size,
            n_tokens,
            vocab_size,
            bytes_per_element=compute_bytes,
        )
        starts = list(range(0, vocab_size, resolved_chunk_size))
        start_arrays = [mx.array([v_start], dtype=mx.int32) for v_start in starts]
        weight_start_arrays = [mx.array([v_start, 0], dtype=mx.int32) for v_start in starts]
        chunk_plan_cache[key] = (resolved_chunk_size, starts, start_arrays, weight_start_arrays)
        return chunk_plan_cache[key]

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
            )
            return losses, lse

        @runtime_cce_loss_full.vjp
        def runtime_cce_loss_vjp(primals, cotangents, outputs):
            hidden, weight, scales, biases, targets = primals
            grad_output = cotangents[0] if isinstance(cotangents, tuple) else cotangents

            hidden_compute = hidden
            weight_compute = weight
            targets32 = targets.astype(mx.int32)
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
                biases_chunk = biases[v_start:v_end]

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
                    dlogits_out_dtype = mx.float32 if logits.dtype == mx.bfloat16 else logits.dtype
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

            # Quantized weight gradients are zero — correct for LoRA where the
            # LM head is frozen and gradients flow through grad_hidden only.
            # Full fine-tuning of quantized models would need
            # dequantize → grad → requantize, which is not supported.
            return (
                grad_hidden.astype(hidden.dtype),
                mx.zeros_like(weight),
                mx.zeros_like(scales),
                mx.zeros_like(biases),
                mx.zeros_like(targets),
            )

        def runtime_cce_loss(
            hidden: mx.array,
            weight: mx.array,
            scales: mx.array,
            biases: mx.array,
            targets: mx.array,
        ) -> mx.array:
            return runtime_cce_loss_full(hidden, weight, scales, biases, targets)[0]

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
        )
        return losses, lse

    @runtime_cce_loss_full.vjp
    def runtime_cce_loss_vjp(primals, cotangents, outputs):
        hidden, weight, targets = primals
        grad_output = cotangents[0] if isinstance(cotangents, tuple) else cotangents

        hidden_compute = hidden
        weight_compute = weight
        targets32 = targets.astype(mx.int32)
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
        # Accumulate weight gradient in float32 to avoid precision loss
        # when hidden is float16/bfloat16. Only matters for full fine-tuning
        # (LoRA freezes the LM head so this VJP path is never reached).
        grad_weight = mx.zeros(weight_compute.shape, dtype=mx.float32)
        n_reads = 4

        for chunk_idx, v_start in enumerate(chunk_starts_int):
            v_end = min(v_start + resolved_chunk_size, vocab_size)
            weight_chunk = weight_compute[v_start:v_end]

            logits = hidden_compute @ weight_chunk.T

            if dlogits_kernel is not None:
                total_threads = (logits.size + n_reads - 1) // n_reads
                dlogits_out_dtype = mx.float32 if logits.dtype == mx.bfloat16 else logits.dtype
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
                ).astype(logits.dtype)

            d_logits_compute = d_logits.astype(hidden_compute.dtype)
            grad_hidden = grad_hidden + d_logits_compute @ weight_chunk
            # Weight gradient GEMM in float32 for accumulation precision
            d_logits_f32 = d_logits.astype(mx.float32)
            hidden_f32 = hidden_compute.astype(mx.float32)
            grad_weight_chunk = d_logits_f32.T @ hidden_f32
            grad_weight = mx.slice_update(
                grad_weight,
                grad_weight_chunk,
                start_indices=weight_chunk_starts[chunk_idx],
                axes=(0, 1),
            )

        return grad_hidden.astype(hidden.dtype), grad_weight.astype(weight.dtype), mx.zeros_like(targets)

    def runtime_cce_loss(hidden: mx.array, weight: mx.array, targets: mx.array) -> mx.array:
        return runtime_cce_loss_full(hidden, weight, targets)[0]

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
):
    """Return a standalone chunked CCE loss callable and a kernel-usage flag."""

    return make_runtime_cce_loss_fused_finalize(
        ignore_index=ignore_index,
        logit_softcap=logit_softcap,
        chunk_size=chunk_size,
        quantized=quantized,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )

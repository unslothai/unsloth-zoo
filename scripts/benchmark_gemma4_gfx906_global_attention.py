#!/usr/bin/env python3
"""Reproduce Gemma-4 31B D=512 attention time/VRAM on gfx906.

Run each backend in a fresh process so allocator state cannot leak between
measurements, for example:

    python scripts/benchmark_gemma4_gfx906_global_attention.py --backend sdpa
    python scripts/benchmark_gemma4_gfx906_global_attention.py --backend triton

The default shape matches Gemma-4 31B global attention: B=1, Hq=32, Hkv=4,
D=512, fp32, causal, scale=1.0, sequence length 4096.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("sdpa", "triton"), required=True)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp32")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    return parser.parse_args()


def dtype_from_name(name):
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def sync():
    torch.cuda.synchronize()


def make_inputs(seq_len, dtype, seed):
    torch.manual_seed(seed)
    q = torch.randn(1, 32, seq_len, 512, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(1, 4, seq_len, 512, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    grad = torch.randn(1, 32, seq_len, 512, device="cuda", dtype=dtype)
    return q, k, v, grad


def load_triton_attention():
    """Load the private kernel once, before warmup and measured timing."""
    path = (
        Path(__file__).resolve().parents[1]
        / "unsloth_zoo"
        / "temporary_patches"
        / "_gemma4_gfx906_global_kernels.py"
    )
    spec = importlib.util.spec_from_file_location("_gemma4_gfx906_bench_kernel", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.gemma4_gfx906_global_attention


def run_once(
    backend,
    seq_len,
    dtype,
    scale,
    seed,
    *,
    measure,
    triton_attention=None,
):
    q, k, v, grad = make_inputs(seq_len, dtype, seed)
    if measure:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_alloc = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()

    sync()
    t0 = time.perf_counter()
    if backend == "sdpa":
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=0.0,
            scale=scale,
            enable_gqa=True,
        )
    else:
        assert triton_attention is not None
        out = triton_attention(q, k, v, scale)
    sync()
    t1 = time.perf_counter()

    if measure:
        peak_fwd_alloc = torch.cuda.max_memory_allocated()
        peak_fwd_reserved = torch.cuda.max_memory_reserved()

    out.backward(grad)
    sync()
    t2 = time.perf_counter()

    # Snapshot measured peaks before finiteness validation allocates temporary
    # tensors.  The benchmark reports end-to-end allocated memory for this
    # attention call (outputs/saved tensors/gradients included), not "workspace".
    if measure:
        peak_total_alloc = torch.cuda.max_memory_allocated()
        peak_total_reserved = torch.cuda.max_memory_reserved()

    row = {
        "forward_sec": t1 - t0,
        "backward_sec": t2 - t1,
        "total_sec": t2 - t0,
        "finite_output": bool(torch.isfinite(out).all()),
        "finite_q_grad": bool(torch.isfinite(q.grad).all()),
        "finite_k_grad": bool(torch.isfinite(k.grad).all()),
        "finite_v_grad": bool(torch.isfinite(v.grad).all()),
    }
    if measure:
        row.update(
            baseline_allocated_gib=baseline_alloc / 2**30,
            baseline_reserved_gib=baseline_reserved / 2**30,
            peak_forward_allocated_gib=peak_fwd_alloc / 2**30,
            peak_forward_reserved_gib=peak_fwd_reserved / 2**30,
            incremental_forward_allocated_gib=(peak_fwd_alloc - baseline_alloc) / 2**30,
            incremental_forward_reserved_gib=(peak_fwd_reserved - baseline_reserved) / 2**30,
            peak_total_allocated_gib=peak_total_alloc / 2**30,
            peak_total_reserved_gib=peak_total_reserved / 2**30,
            incremental_total_allocated_gib=(peak_total_alloc - baseline_alloc) / 2**30,
            incremental_total_reserved_gib=(peak_total_reserved - baseline_reserved) / 2**30,
        )
    return row


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device required")
    if getattr(torch.version, "hip", None) is None:
        raise SystemExit("ROCm/HIP runtime required")

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = str(getattr(props, "gcnArchName", "")).split(":", 1)[0]
    if args.backend == "triton" and arch != "gfx906":
        raise SystemExit(f"Triton fallback benchmark requires gfx906, got {arch!r}")

    dtype = dtype_from_name(args.dtype)
    triton_attention = load_triton_attention() if args.backend == "triton" else None
    for i in range(args.warmup):
        run_once(
            args.backend,
            args.seq_len,
            dtype,
            args.scale,
            args.seed + i,
            measure=False,
            triton_attention=triton_attention,
        )
        torch.cuda.empty_cache()

    rows = [
        run_once(
            args.backend,
            args.seq_len,
            dtype,
            args.scale,
            args.seed + 10_000 + i,
            measure=True,
            triton_attention=triton_attention,
        )
        for i in range(args.steps)
    ]

    def mean(key):
        return sum(row[key] for row in rows) / len(rows)

    result = {
        "backend": args.backend,
        "shape": [1, 32, args.seq_len, 512],
        "kv_heads": 4,
        "dtype": args.dtype,
        "scale": args.scale,
        "seed": args.seed,
        "warmup": args.warmup,
        "steps": args.steps,
        "device_name": props.name,
        "gcn_arch": str(getattr(props, "gcnArchName", "")),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "python": platform.python_version(),
        "mean_forward_sec": mean("forward_sec"),
        "mean_backward_sec": mean("backward_sec"),
        "mean_total_sec": mean("total_sec"),
        "max_peak_forward_allocated_gib": max(row["peak_forward_allocated_gib"] for row in rows),
        "max_peak_forward_reserved_gib": max(row["peak_forward_reserved_gib"] for row in rows),
        "max_peak_total_allocated_gib": max(row["peak_total_allocated_gib"] for row in rows),
        "max_peak_total_reserved_gib": max(row["peak_total_reserved_gib"] for row in rows),
        "max_incremental_total_allocated_gib": max(row["incremental_total_allocated_gib"] for row in rows),
        "max_incremental_total_reserved_gib": max(row["incremental_total_reserved_gib"] for row in rows),
        "all_finite": all(
            row["finite_output"]
            and row["finite_q_grad"]
            and row["finite_k_grad"]
            and row["finite_v_grad"]
            for row in rows
        ),
        "runs": rows,
    }
    if args.backend == "triton":
        import triton

        result["triton"] = triton.__version__

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

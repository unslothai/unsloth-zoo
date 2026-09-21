#!/usr/bin/env python3
"""Compile the gfx906 Gemma-4 D=512 kernels and report Triton resources."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch


def _load_kernel_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "unsloth_zoo"
        / "temporary_patches"
        / "_gemma4_gfx906_global_kernels.py"
    )
    spec = importlib.util.spec_from_file_location("_gemma4_gfx906_metadata_kernel", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _compiled_kernels(jit_function):
    """Return unique CompiledKernel-like objects held by a Triton JITFunction."""
    roots = []
    for name in ("cache", "device_caches"):
        value = getattr(jit_function, name, None)
        if value is not None:
            roots.append(value)

    found = []
    seen_containers = set()

    def walk(value):
        if value is None:
            return
        if hasattr(value, "metadata") and (
            hasattr(value, "run") or hasattr(value, "launch_metadata")
        ):
            found.append(value)
            return
        ident = id(value)
        if ident in seen_containers:
            return
        seen_containers.add(ident)
        if isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple, set)):
            for child in value:
                walk(child)

    for root in roots:
        walk(root)

    unique = []
    seen = set()
    for kernel in found:
        if id(kernel) not in seen:
            seen.add(id(kernel))
            unique.append(kernel)
    return unique


def _metadata_row(kernel, fallback_name):
    metadata = kernel.metadata
    return {
        "name": getattr(metadata, "name", None)
        or getattr(kernel, "name", None)
        or fallback_name,
        "num_ctas": getattr(metadata, "num_ctas", None),
        "num_stages": getattr(metadata, "num_stages", None),
        "num_warps": getattr(metadata, "num_warps", None),
        "registers_per_thread": getattr(kernel, "n_regs", None),
        "shared_bytes": getattr(metadata, "shared", None),
        "spills": getattr(kernel, "n_spills", None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), required=True)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise SystemExit("ROCm/HIP device required")

    module = _load_kernel_module()
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]
    torch.manual_seed(args.seed)
    S = args.seq_len
    q = torch.randn(1, 32, S, 512, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(1, 4, S, 512, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    out = module.gemma4_gfx906_global_attention(q, k, v, 1.0)
    out.backward(torch.randn_like(out))
    torch.cuda.synchronize()

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    groups = {
        "forward": (module._fwd_kernel, "_fwd_kernel"),
        "backward_dq": (module._bwd_dq_kernel, "_bwd_dq_kernel"),
        "backward_dkv": (module._bwd_dkv_kernel, "_bwd_dkv_kernel"),
    }
    kernels = {
        label: [
            _metadata_row(compiled, fallback_name)
            for compiled in _compiled_kernels(jit_function)
        ]
        for label, (jit_function, fallback_name) in groups.items()
    }
    if any(not rows for rows in kernels.values()):
        missing = [name for name, rows in kernels.items() if not rows]
        raise RuntimeError(f"Triton compiled-kernel metadata unavailable for: {missing}")

    import triton

    result = {
        "device_name": props.name,
        "dtype": args.dtype,
        "finite": all(
            bool(torch.isfinite(x).all())
            for x in (out, q.grad, k.grad, v.grad)
        ),
        "gcn_arch": str(getattr(props, "gcnArchName", "")),
        "hip": getattr(torch.version, "hip", None),
        "kernels": kernels,
        "seq_len": S,
        "torch": torch.__version__,
        "triton": triton.__version__,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
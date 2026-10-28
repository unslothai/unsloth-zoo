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

"""Every Triton launch states enable_fp_fusion explicitly: Triton 3.9 flips the default to False."""

import ast
import json
import os
import pathlib
import subprocess
import sys

import pytest

ZOO = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo"

# Subscript calls that are not Triton launches.
_NOT_LAUNCHES = {
    "fused_losses/fused_linear.py": {"_OPS"},
    "temporary_patches/gpt_oss.py": {"self.gate_up_projs", "self.down_projs"},
    "temporary_patches/moe_grouped_modulelist.py": {"spec"},
    "temporary_patches/moe_utils.py": {"registry"},
    # Custom ops out of the pinned-op tuple, and a weakref read out of a cache entry.
    "temporary_patches/moe_utils_sonic.py": {"_pinned_ops()", "hit"},
}
# Launches that deliberately disable contraction to match an eager reference bit for bit.
_DELIBERATE_FALSE = {
    "merged_rollout_exact.py:_exact_merge_kernel",
    "temporary_patches/gpt_oss_grouped_qlora.py:_nf4_dequant_stack_kernel",
    "temporary_patches/moe_routed.py:_nf4_select_dequant_kernel",
    "temporary_patches/moe_nf4_grouped.py:_nf4_grouped_kernel",
    "temporary_patches/moe_nf4_grouped.py:_nf4_grouped_kernel.fn",
    "temporary_patches/moe_fused_swiglu.py:_swiglu_fwd_kernel",
    "temporary_patches/moe_fused_swiglu.py:_swiglu_bwd_kernel",
    "temporary_patches/moe_triton_kernels.py:kernel",
    "temporary_patches/moe_triton_kernels.py:fwd",
    "temporary_patches/moe_triton_kernels.py:bwd_dy",
    "temporary_patches/moe_triton_kernels.py:bwd_dw",
    "gpt_oss_decode_fused.py:_add_rms_norm_kernel",
    "gpt_oss_decode_fused.py:_sink_scale_kernel",
}


def _kwargs_dicts(tree):
    """id(launch call) -> the dict(...) / {...} each `**name` it splats was assigned in its function."""
    out = {}
    # Module-level `NAME = {...}` splats too (gpt_oss_decode_fused's `_NO_FMA`); a function's own wins.
    module = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
    }
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        dicts = dict(module)
        for node in ast.walk(func):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                dicts[node.targets[0].id] = node.value
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                splats = [k.value.id for k in node.keywords if k.arg is None and isinstance(k.value, ast.Name)]
                out[id(node)] = [dicts[n] for n in splats if n in dicts]
    return out


def _fp_fusion_values(node, splatted):
    values = [k.value for k in node.keywords if k.arg == "enable_fp_fusion"]
    # `{} if TRITON_INTERPRET else {...}`: the compiled branch is the one that states it.
    splatted = [v.orelse if isinstance(v, ast.IfExp) else v for v in splatted]
    for value in splatted:
        if isinstance(value, ast.Call) and getattr(value.func, "id", None) == "dict":
            values += [k.value for k in value.keywords if k.arg == "enable_fp_fusion"]
        elif isinstance(value, ast.Dict):
            values += [v for k, v in zip(value.keys, value.values) if isinstance(k, ast.Constant) and k.value == "enable_fp_fusion"]
    return values


def _launch_sites():
    for path in sorted(ZOO.rglob("*.py")):
        rel = path.relative_to(ZOO).as_posix()
        if "_vendored" in rel:
            continue
        src = path.read_text(encoding = "utf-8")
        if "triton" not in src:
            continue
        tree = ast.parse(src)
        splats = _kwargs_dicts(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
                name = ast.get_source_segment(src, node.func.value)
                if name in _NOT_LAUNCHES.get(rel, ()):
                    continue
                yield rel, name, node, splats.get(id(node), [])


def test_every_launch_states_fp_fusion():
    sites = list(_launch_sites())
    assert len(sites) >= 20, [(r, n) for r, n, _, _ in sites]
    missing, wrong = [], []
    for rel, name, node, splatted in sites:
        values = _fp_fusion_values(node, splatted)
        if not values:
            missing.append(f"{rel}:{node.lineno}:{name}")
            continue
        want = f"{rel}:{name}" not in _DELIBERATE_FALSE
        if not (isinstance(values[0], ast.Constant) and values[0].value is want):
            wrong.append(f"{rel}:{node.lineno}:{name} expected {want}")
    assert not missing, f"Triton launches without enable_fp_fusion: {missing}"
    assert not wrong, wrong



def _compiled_options(kernel):
    from triton.runtime.jit import JITFunction

    seen = []
    while not isinstance(kernel, JITFunction) and hasattr(kernel, "fn"):
        kernel = kernel.fn
    if not isinstance(kernel, JITFunction):
        return seen
    cache = getattr(kernel, "cache", None)
    if isinstance(cache, dict):  # Triton <= 3.2: cache[device][key]
        tables = list(cache.values())
    else:  # Triton 3.3+: device_caches[device] = (kernel_cache, ...)
        tables = [entry[0] for entry in kernel.device_caches.values()]
    for table in tables:
        for compiled in table.values():
            meta = compiled.metadata
            value = meta.get("enable_fp_fusion") if isinstance(meta, dict) else getattr(meta, "enable_fp_fusion", None)
            seen.append(value)
    return seen


def exercise(device = "cuda"):
    import torch

    from unsloth_zoo import mxfp4_dequant, mxfp4_gemm
    from unsloth_zoo.temporary_patches import gdn_causal_conv1d as gcc
    from unsloth_zoo.temporary_patches import moe_grouped_fp16 as mg

    out = {}
    g = torch.Generator(device = device).manual_seed(0)
    E, R, C = 4, 64, 128
    blocks = torch.randint(0, 256, (E, R, C // 32, 16), dtype = torch.uint8, device = device, generator = g)
    scales = torch.randint(118, 130, (E, R, C // 32), dtype = torch.uint8, device = device, generator = g)
    bf16 = torch.cuda.get_device_capability(device) >= (8, 0)
    dt = torch.bfloat16 if bf16 else torch.float16
    if bf16:
        out["mxfp4_decode"] = (mxfp4_gemm._decode_for_test(blocks, scales),)
        out["mxfp4_dequant"] = (mxfp4_dequant.mxfp4_dequantize(blocks, scales, torch.bfloat16),)
    # Triton 3.2 aborts the process (MLIR assertion) compiling the grouped MXFP4 GEMM, base included.
    if bf16 and _triton_at_least(3, 3) and mxfp4_gemm.mxfp4_grouped_mm_available(torch.device(device)):
        counts = torch.tensor([3, 0, 9, 4], dtype = torch.int32, device = device)
        x = torch.randn(16, C, generator = g, device = device).to(torch.bfloat16)
        out["mxfp4_grouped_mm"] = (mxfp4_gemm.mxfp4_grouped_mm(x, blocks, scales, counts),)
        out["mxfp4_grouped_gemv"] = (mxfp4_gemm.mxfp4_grouped_mm(x[:1], blocks, scales, torch.tensor([1, 0, 0, 0], dtype = torch.int32, device = device)),)

    if mg.triton_grouped_available(torch.device(device)):
        counts = torch.tensor([5, 0, 7, 4], dtype = torch.int32, device = device)
        a = torch.randn(16, 64, generator = g, device = device).half()
        w = torch.randn(4, 32, 64, generator = g, device = device).half()
        out["grouped_gemm"] = (mg.grouped_gemm(a, w, counts, torch.float32, b_trans = True, backend = "triton"),)
        dy = torch.randn(16, 32, generator = g, device = device).half()
        out["grouped_wgrad"] = (mg.grouped_wgrad(dy, a, counts, torch.float32, backend = "triton"),)

    x = torch.randn(2, 64, 32, generator = g, device = device).to(dt).requires_grad_()
    cw = torch.randn(64, 4, generator = g, device = device).to(dt).requires_grad_()
    y = gcc.triton_causal_conv1d(x, cw, None, "silu")
    y.float().square().sum().backward()
    out["causal_conv1d"] = (y.detach(), x.grad, cw.grad)
    torch.cuda.synchronize()

    kernels = {
        "mxfp4_gemm._decode_only_kernel": mxfp4_gemm._decode_only_kernel,
        "mxfp4_gemm._mxfp4_grouped_mm_kernel": mxfp4_gemm._mxfp4_grouped_mm_kernel,
        "mxfp4_gemm._mxfp4_grouped_gemv_kernel": mxfp4_gemm._mxfp4_grouped_gemv_kernel,
        "mxfp4_dequant._mxfp4_dequant_kernel": mxfp4_dequant._mxfp4_dequant_kernel,
        "moe_grouped_fp16._grouped_gemm_kernel": mg._grouped_gemm_kernel,
        "moe_grouped_fp16._grouped_wgrad_kernel": mg._grouped_wgrad_kernel,
    }
    if gcc._kernels is not None:
        kernels["gdn_causal_conv1d.fwd"], (kernels["gdn_causal_conv1d.bwd_g"], kernels["gdn_causal_conv1d.bwd_dxdw"]) = gcc._kernels
    meta = {name: _compiled_options(k) for name, k in kernels.items()}
    return out, {name: v for name, v in meta.items() if v}


def _child():
    _, meta = exercise()
    print("A1META " + json.dumps(meta))


def _has_gpu():
    try:
        import torch

        return torch.cuda.is_available() and not getattr(torch.cuda, "_unsloth_consolidated_spoof", False)
    except Exception:
        return False


def _triton_at_least(major, minor):
    import triton

    return tuple(int(x) for x in triton.__version__.split(".")[:2]) >= (major, minor)


def _has_bf16():
    import torch

    return torch.cuda.get_device_capability() >= (8, 0)


@pytest.mark.parametrize("default", ["0", "1"])
def test_compiled_kernels_keep_fp_fusion(tmp_path, default):
    if not _has_gpu():
        pytest.skip("needs a CUDA / ROCm GPU")
    pytest.importorskip("triton")
    env = dict(os.environ)
    env["TRITON_DEFAULT_FP_FUSION"] = default
    env.setdefault("UNSLOTH_IS_PRESENT", "1")
    env["TRITON_CACHE_DIR"] = str(tmp_path / "triton")
    env.pop("TRITON_FORCE_DISABLE_FP_FUSION", None)
    code = f"import sys; sys.path.insert(0, {str(pathlib.Path(__file__).parent)!r}); import test_triton_fp_fusion as t; t._child()"
    proc = subprocess.run([sys.executable, "-c", code], env = env, capture_output = True, text = True, timeout = 900)
    line = [l for l in proc.stdout.splitlines() if l.startswith("A1META ")]
    assert proc.returncode == 0 and line, proc.stdout[-3000:] + proc.stderr[-6000:]
    meta = json.loads(line[-1][len("A1META ") :])
    expected = ["gdn_causal_conv1d.fwd", "gdn_causal_conv1d.bwd_g", "gdn_causal_conv1d.bwd_dxdw"]
    if _triton_at_least(3, 3):
        expected += ["moe_grouped_fp16._grouped_gemm_kernel", "moe_grouped_fp16._grouped_wgrad_kernel"]
    if _has_bf16():
        expected += ["mxfp4_gemm._decode_only_kernel"]
    for name in expected:
        assert name in meta, (name, sorted(meta))
    bad = {name: v for name, v in meta.items() if any(x is not True for x in v)}
    assert not bad, f"compiled without enable_fp_fusion=True: {bad}"

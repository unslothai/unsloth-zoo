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
    # aten rounds every multiply before the add that follows it (_NO_FMA in that module).
    "gpt_oss_decode_fused.py:_add_rms_norm_kernel",
    "gpt_oss_decode_fused.py:_sink_scale_kernel",
}


def _assigned_names(scope_body):
    return {
        node.targets[0].id: node.value
        for node in scope_body
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
    }


def _kwargs_dicts(tree):
    """id(launch call) -> the dict(...) / {...} each `**name` it splats was assigned in its function, or at
    module level (a function-local name wins). `a if cond else b` contributes both branches."""
    module = _assigned_names(tree.body)
    out = {}
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        dicts = dict(module)
        dicts.update(_assigned_names(ast.walk(func)))
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                splats = [k.value.id for k in node.keywords if k.arg is None and isinstance(k.value, ast.Name)]
                values = [dicts[n] for n in splats if n in dicts]
                out[id(node)] = [
                    branch
                    for value in values
                    for branch in ((value.body, value.orelse) if isinstance(value, ast.IfExp) else (value,))
                ]
    return out


def _fp_fusion_values(node, splatted):
    values = [k.value for k in node.keywords if k.arg == "enable_fp_fusion"]
    for value in splatted:
        if isinstance(value, ast.Call) and getattr(value.func, "id", None) == "dict":
            values += [k.value for k in value.keywords if k.arg == "enable_fp_fusion"]
        elif isinstance(value, ast.Dict):
            values += [v for k, v in zip(value.keys, value.values) if isinstance(k, ast.Constant) and k.value == "enable_fp_fusion"]
    return values


def _params(func):
    args = func.args
    names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
    names += [a.arg for a in (args.vararg, args.kwarg) if a is not None]
    return set(names)


def _enclosing(tree):
    """id(node) -> the innermost function holding it, and id(node) -> the `if` tests above it there."""
    func_of, tests_of = {}, {}

    def visit(node, func, tests):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child, [])
                continue
            func_of[id(child)], tests_of[id(child)] = func, tests
            if isinstance(node, ast.If) and child in node.body:
                visit(child, func, tests + [node.test])
            else:
                visit(child, func, tests)

    visit(tree, None, [])
    return func_of, tests_of


def _options_names(tree):
    """Names the module's functions collect launch options under (`def launch(..., **options)`)."""
    return {
        f.args.kwarg.arg for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.args.kwarg is not None
    }


def _forwards_options(node, func, options):
    """A launch inside a generic launcher: it splats the options the launcher was called with, so the
    flag comes from the launcher's callers, which are checked instead. Splatting only constexprs
    does not count."""
    return func is not None and any(
        k.arg is None and isinstance(k.value, ast.Name) and k.value.id in options & _params(func)
        for k in node.keywords
    )


def _interpreter_only(tests):
    # TRITON_INTERPRET runs the kernel in Python: nothing is compiled, so there is no fusion to state.
    return any(isinstance(t, ast.Name) and "INTERPRET" in t.id for t in tests)


def _module_trees():
    for path in sorted(ZOO.rglob("*.py")):
        rel = path.relative_to(ZOO).as_posix()
        if "_vendored" in rel:
            continue
        src = path.read_text(encoding = "utf-8")
        if "triton" not in src:
            continue
        yield rel, src, ast.parse(src)


def _launcher_entry_points(trees):
    """{module dotted name: {function}} for functions taking **options in a module whose launches
    forward options: calling one of them is a launch."""
    out = {}
    for rel, _src, tree in trees:
        func_of, _ = _enclosing(tree)
        options = _options_names(tree)
        forwarding = any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
            and _forwards_options(node, func_of.get(id(node)), options)
            for node in ast.walk(tree)
        )
        if forwarding:
            module = rel[: -len(".py")].replace("/", ".")
            out[module] = {
                f.name for f in tree.body if isinstance(f, ast.FunctionDef) and f.args.kwarg is not None
            }
    return out


def _launcher_aliases(rel, tree, entry_points):
    """Local names bound to a launcher entry point by `from .module import f [as g]`."""
    package = rel.split("/")[:-1]
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            base = package[: len(package) - (node.level - 1)] if node.level else []
            module = ".".join(base + node.module.split(".")) if node.level else node.module.removeprefix("unsloth_zoo.")
            for alias in node.names:
                if alias.name in entry_points.get(module, ()):
                    aliases[alias.asname or alias.name] = alias.name
    return aliases


def _launch_sites():
    trees = list(_module_trees())
    entry_points = _launcher_entry_points(trees)
    for rel, src, tree in trees:
        splats = _kwargs_dicts(tree)
        func_of, tests_of = _enclosing(tree)
        options = _options_names(tree)
        aliases = _launcher_aliases(rel, tree, entry_points)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Subscript):
                name = ast.get_source_segment(src, node.func.value)
                if name in _NOT_LAUNCHES.get(rel, ()):
                    continue
                if _forwards_options(node, func_of.get(id(node)), options) or _interpreter_only(
                    tests_of.get(id(node), [])
                ):
                    continue
            elif isinstance(node.func, ast.Name) and node.func.id in aliases and node.args:
                name = ast.get_source_segment(src, node.args[0])
            else:
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

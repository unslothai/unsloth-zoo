# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Per-call token / row counts of the zoo MoE kernels get no % 16 class (jit_any_alignment keeps == 1, do_not_specialize
drops both), and the routing
scratch slices keep a fixed pointer alignment, so a new prefill length or batch size compiles nothing after warm-up
(w5_recompile G1: these kernels re-specialized with the same constexprs on Kaggle T4)."""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo"


def test_helper_drops_alignment_only():
    pytest.importorskip("triton")
    import triton.language as tl

    from unsloth_zoo.triton_specialize import jit_any_alignment

    @jit_any_alignment("T")
    def kernel(x_ptr, T, N, BLOCK: tl.constexpr):  # BLOCK: elements per program
        pass

    if not hasattr(kernel, "params"):  # TRITON_INTERPRET=1 returns an InterpretedFunction
        pytest.skip("interpreted Triton function has no params")
    params = {p.name: p for p in kernel.params}
    if not hasattr(params["T"], "do_not_specialize_on_alignment"):
        pytest.skip("this Triton has no do_not_specialize_on_alignment")
    assert params["T"].do_not_specialize_on_alignment and not params["T"].do_not_specialize
    assert not params["N"].do_not_specialize_on_alignment


EXPECTED = {
    "temporary_patches/moe_fused_swiglu.py": {"_swiglu_fwd_kernel": "M", "_swiglu_bwd_kernel": "M"},
    "temporary_patches/moe_triton_kernels.py": {"_combine_fwd_kernel": "T", "_combine_bwd_dy_kernel": "N"},
    "temporary_patches/moe_router_triton.py": {"_router_fwd_kernel": "T", "_router_bwd_kernel": "T"},
    "mxfp4_gemm.py": {"_mxfp4_grouped_mm_kernel": "M"},
}


@pytest.mark.parametrize("path", sorted(EXPECTED))
def test_token_counts_alignment_unspecialized(path):
    tree = ast.parse((ROOT / path).read_text(encoding = "utf-8"))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in EXPECTED[path]:
            names = []
            for d in node.decorator_list:
                if not isinstance(d, ast.Call):
                    continue
                if getattr(d.func, "id", "").lstrip("_") == "jit_any_alignment":
                    names += [a.value for a in d.args]
                elif ast.unparse(d.func) == "triton.jit":
                    names += [e.value for k in d.keywords if k.arg == "do_not_specialize" for e in k.value.elts]
            assert len(names) == 1, (node.name, names)
            assert names[0] in [a.arg for a in node.args.args], (node.name, names[0])
            found[node.name] = names[0]
    assert found == EXPECTED[path]


def test_routing_scratch_slices_stay_aligned():
    from unsloth_zoo.temporary_patches.moe_routing_triton import _route_scratch

    # The real allocation: every slice 16-byte aligned for any token block count and expert count.
    for NB in range(1, 40):
        for E in (7, 8, 31, 32, 128, 255):
            hist, rank_base, starts = _route_scratch(NB, E, "cpu")
            assert all(t.data_ptr() % 16 == 0 for t in (hist, rank_base, starts)), (NB, E)
            assert hist.numel() >= NB * (E + 1) and rank_base.numel() >= NB * (E + 1) and starts.numel() == E + 1


def test_nf4_grouped_row_counts_unspecialized():
    tree = ast.parse((ROOT / "temporary_patches/moe_nf4_grouped.py").read_text(encoding = "utf-8"))
    (fn,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_nf4_grouped_kernel"]
    (jit,) = [d for d in fn.decorator_list if isinstance(d, ast.Call) and ast.unparse(d.func) == "triton.jit"]
    (kw,) = [k for k in jit.keywords if k.arg == "do_not_specialize"]
    assert [e.value for e in kw.value.elts] == ["M", "M_HINT"]
    # Unused in the body, so not specializing them cannot change the kernel.
    body = {n.id for stmt in fn.body for n in ast.walk(stmt) if isinstance(n, ast.Name)}
    assert not body & {"M", "M_HINT"}

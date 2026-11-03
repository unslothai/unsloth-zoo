# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Per-call token / row counts of the zoo MoE kernels keep Triton's == 1 binary but no % 16 class, and the routing
scratch slices keep a fixed pointer alignment, so a new prefill length or batch size compiles nothing after warm-up
(w5_recompile G1: these kernels re-specialized with the same constexprs on Kaggle T4)."""

import ast
import pathlib

import pytest

triton = pytest.importorskip("triton")
import triton.language as tl

from unsloth_zoo.triton_specialize import jit_any_alignment

ROOT = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo"


def test_helper_drops_alignment_only():
    @jit_any_alignment("T")
    def kernel(x_ptr, T, N, BLOCK: tl.constexpr):  # BLOCK: elements per program
        pass

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
            (deco,) = node.decorator_list
            assert isinstance(deco, ast.Call) and deco.func.id == "_jit_any_alignment", node.name
            name = deco.args[0].value
            assert name in [a.arg for a in node.args.args], (node.name, name)
            found[node.name] = name
    assert found == EXPECTED[path]


def test_routing_scratch_slices_stay_aligned():
    source = (ROOT / "temporary_patches/moe_routing_triton.py").read_text(encoding = "utf-8")
    assert "size = (NB * (E + 1) + 3) // 4 * 4" in source
    # int32 scratch: every slice offset is a multiple of 4 elements = 16 bytes, for any NB and E.
    for NB in range(1, 40):
        for E in (7, 8, 31, 32, 128, 255):
            size = (NB * (E + 1) + 3) // 4 * 4
            assert (size * 4) % 16 == 0 and (2 * size * 4) % 16 == 0 and size >= NB * (E + 1)

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Per-call sizes (batch, token / row counts, split indices) stay runtime Triton arguments with
do_not_specialize, so they do not compile a new variant per batch size, decode step or prompt length."""

import ast
import pathlib

import pytest

ZOO = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo"

CASES = {
    "_vendored/fla/ops/common/chunk_o.py": {"chunk_bwd_kernel_dqkwg": ("B", "T")},
    "temporary_patches/moe_router_triton.py": {"_router_fwd_kernel": ("T",), "_router_bwd_kernel": ("T",)},
    "fused_losses/fused_linear.py": {
        "_fl_partial_kernel": ("n_rows", "stride_k", "split0"),
        "_fl_combine_kernel": ("n_rows", "stride_k"),
        "_fl_flash_p_kernel": ("n_rows", "split0", "stride_k"),
        "_fl_flash_u_kernel": ("n_rows", "stride_k"),
        "_fl_scale_kernel": ("n",),
    },
    "merged_rollout_exact.py": {"_exact_merge_kernel": ("mode",)},
    "temporary_patches/gdn_causal_conv1d.py": {
        "_causal_conv1d_fwd_kernel": ("T",),
        "_causal_conv1d_bwd_g_kernel": ("T",),
        "_causal_conv1d_bwd_dxdw_kernel": ("T", "NT"),
    },
    "temporary_patches/moe_fused_swiglu.py": {"_swiglu_fwd_kernel": ("M",), "_swiglu_bwd_kernel": ("M",)},
}


def _defs(path):
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def _do_not_specialize(fn):
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Call) and ast.unparse(dec.func).endswith("jit"):
            for kw in dec.keywords:
                if kw.arg == "do_not_specialize":
                    return {e.value for e in kw.value.elts}
    return set()


@pytest.mark.parametrize("rel", sorted(CASES))
def test_runtime_sizes_are_not_specialized(rel):
    defs = _defs(ZOO / rel)
    for name, args in CASES[rel].items():
        fn = defs[name]
        params = {a.arg: a for a in fn.args.args}
        dns = _do_not_specialize(fn)
        for arg in args:
            assert arg in params, (name, arg)
            ann = params[arg].annotation
            assert ann is None or "constexpr" not in ast.unparse(ann), (name, arg)
            assert arg in dns, (name, arg, sorted(dns))

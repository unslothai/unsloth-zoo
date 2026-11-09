# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Per-call token / row counts of the zoo MoE kernels get no % 16 class (jit_any_alignment keeps == 1, do_not_specialize
drops both), and the routing
scratch slices keep a fixed pointer alignment, so a new prefill length or batch size compiles nothing after warm-up
(w5_recompile G1: these kernels re-specialized with the same constexprs on Kaggle T4)."""

import ast
import inspect
import pathlib
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo"


def test_helper_drops_alignment_only(monkeypatch):
    from unsloth_zoo import triton_specialize as ts

    # Under TRITON_INTERPRET=1 triton.jit returns an InterpretedFunction, which keeps no params to inspect.
    monkeypatch.setenv("TRITON_INTERPRET", "0")

    def kernel(x_ptr, T, N):
        pass

    built = ts.jit_any_alignment("T")(kernel)
    if ts.triton is None:  # CPU-only torch wheels ship no Triton: the function comes back unbuilt
        assert built is kernel
        return
    params = {p.name: p for p in built.params}
    if "do_not_specialize_on_alignment" in inspect.signature(ts.triton.jit).parameters:
        assert params["T"].do_not_specialize_on_alignment and not params["T"].do_not_specialize
        assert not params["N"].do_not_specialize_on_alignment
    else:
        # A Triton without the option gets plain triton.jit, the behaviour before the helper.
        assert not any(getattr(p, "do_not_specialize_on_alignment", False) for p in built.params)


def test_helper_falls_back_to_plain_jit(monkeypatch):
    import unsloth_zoo.triton_specialize as ts

    calls = []

    def old_jit(fn, **kwargs):  # a Triton whose jit has no do_not_specialize_on_alignment
        calls.append(kwargs)
        if kwargs:
            raise TypeError("jit() got an unexpected keyword argument 'do_not_specialize_on_alignment'")
        return ("jitted", fn)

    def kernel():
        pass

    monkeypatch.setattr(ts, "triton", types.SimpleNamespace(jit = old_jit))
    assert ts.jit_any_alignment("T")(kernel) == ("jitted", kernel)
    assert calls == [{"do_not_specialize_on_alignment": ["T"]}, {}]
    monkeypatch.setattr(ts, "triton", None)  # no Triton at all: the function is returned unbuilt
    assert ts.jit_any_alignment("T")(kernel) is kernel


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


def test_routing_scratch_slices_stay_aligned(monkeypatch):
    import torch

    from unsloth_zoo.temporary_patches import moe_routing_triton as mr

    # Record the pointer alignment of every tensor the routing launch hands its kernels: a slice whose alignment
    # follows the token count is a new Triton binary per prefill length.
    misaligned = []

    class _Recorder:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                for i, a in enumerate(args):
                    if isinstance(a, torch.Tensor) and a.data_ptr() % 16:
                        misaligned.append((self.name, i, N, E))
            return launch

    for name in ("_route_single_kernel", "_route_hist_kernel", "_route_scan_kernel", "_route_starts_kernel",
                 "_route_scatter_kernel"):
        monkeypatch.setattr(mr, name, _Recorder(name), raising = False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: None)  # CPU ids: launch on "this" device

    for N in range(mr._SINGLE_BLOCK_MAX + 1, mr._SINGLE_BLOCK_MAX + 16 * mr._MULTI_BLOCK, 97):
        for E in (7, 8, 32, 128, 255):
            mr._launch(torch.zeros((N // 2, 2), dtype = torch.int64), E)
    assert not misaligned, misaligned[:8]


def test_nf4_grouped_row_counts_unspecialized():
    tree = ast.parse((ROOT / "temporary_patches/moe_nf4_grouped.py").read_text(encoding = "utf-8"))
    (fn,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_nf4_grouped_kernel"]
    (jit,) = [d for d in fn.decorator_list if isinstance(d, ast.Call) and ast.unparse(d.func) == "triton.jit"]
    (kw,) = [k for k in jit.keywords if k.arg == "do_not_specialize"]
    assert [e.value for e in kw.value.elts] == ["M", "M_HINT"]
    # Unused in the body, so not specializing them cannot change the kernel.
    body = {n.id for stmt in fn.body for n in ast.walk(stmt) if isinstance(n, ast.Name)}
    assert not body & {"M", "M_HINT"}

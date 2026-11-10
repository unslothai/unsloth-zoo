# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Per-call token / row counts of the combine and MXFP4 grouped GEMM kernels get no % 16 class (jit_any_alignment
keeps == 1), and the routing scratch slices keep a fixed pointer alignment, so a new prefill length compiles nothing
after warm-up. The do_not_specialize kernels (router, SwiGLU, NF4 grouped) are pinned by test_triton_runtime_size_args."""

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
    "temporary_patches/moe_triton_kernels.py": {"_combine_fwd_kernel": "T", "_combine_bwd_dy_kernel": "N"},
    "mxfp4_gemm.py": {"_mxfp4_grouped_mm_kernel": "M"},
}


@pytest.mark.parametrize("path", sorted(EXPECTED))
def test_token_counts_alignment_unspecialized(path):
    tree = ast.parse((ROOT / path).read_text(encoding = "utf-8"))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in EXPECTED[path]:
            names = [
                a.value for d in node.decorator_list
                if isinstance(d, ast.Call) and getattr(d.func, "id", "").lstrip("_") == "jit_any_alignment"
                for a in d.args
            ]
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


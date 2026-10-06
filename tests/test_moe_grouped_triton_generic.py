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

"""Generic MoE Triton grouped GEMM (unsloth_zoo::grouped_mm_triton): torch._grouped_mm semantics on
moe_grouped_fp16's kernels, behind a static shape / arch gate in _grouped_mm_with_backward_fix,
moe_grouped_modulelist._grouped_mm_fix and transformers' _grouped_mm. On sm90 / sm100 the gate's auto
mode declines, so the kernel tests force UNSLOTH_MOE_GROUPED_TRITON=1."""

import logging
import os

import pytest
import torch

from unsloth_zoo.temporary_patches import moe_grouped_fp16 as MG
from unsloth_zoo.temporary_patches import moe_utils as MU

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason = "needs CUDA")
needs_kernel = pytest.mark.skipif(
    not CUDA or MG.triton is None or torch.version.hip is not None or MU._GROUPED_MM_TRITON_OP is None,
    reason = "needs CUDA, Triton and torch.library.custom_op",
)
_ENV = ("UNSLOTH_MOE_GROUPED_TRITON", "UNSLOTH_DISABLE_MOE_TRITON", "UNSLOTH_MOE_GROUPED_TRITON_MAX_ROWS")


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising = False)
    saved_cap = dict(MU._TRITON_GROUPED_MM_CAPABILITY)
    MU._TRITON_GROUPED_MM_POLICY.clear()
    yield
    MU._TRITON_GROUPED_MM_CAPABILITY.clear()
    MU._TRITON_GROUPED_MM_CAPABILITY.update(saved_cap)
    MU._TRITON_GROUPED_MM_POLICY.clear()


def _force_on(monkeypatch, max_rows = None):
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "1")
    if max_rows is not None:
        monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON_MAX_ROWS", str(max_rows))


def _mock_cap(cap, index = 0):
    MU._TRITON_GROUPED_MM_CAPABILITY[index] = cap
    MU._TRITON_GROUPED_MM_POLICY.clear()


def _problem(counts, K, N, dtype, transposed = True, seed = 0, device = "cuda"):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    counts = torch.tensor(counts, dtype = torch.int64)
    E, M = counts.numel(), int(counts.sum())
    x = torch.randn(M, K, generator = g).to(device, dtype)
    if transposed:   # [E, N, K] parameter seen as an [E, K, N] view, as the MoE path passes it
        param = (torch.randn(E, N, K, generator = g) * 0.1).to(device, dtype)
    else:
        param = (torch.randn(E, K, N, generator = g) * 0.1).to(device, dtype)
    dy = torch.randn(M, N, generator = g).to(device, dtype)
    offs = counts.cumsum(0).to(device = device, dtype = torch.int32)
    return x, param, dy, offs, counts


def _weight(param, transposed):
    return param.transpose(-2, -1) if transposed else param


def _run(fn, x, param, dy, offs, transposed):
    x = x.detach().clone().requires_grad_(True)
    p = param.detach().clone().requires_grad_(True)
    y = fn(x, _weight(p, transposed), offs)
    y.backward(dy)
    return y.detach(), x.grad.detach(), p.grad.detach()


def _fp64(x, param, dy, counts, transposed):
    w = param.double()
    w = w.transpose(-2, -1) if transposed else w        # [E, K, N]
    e_of = torch.repeat_interleave(torch.arange(counts.numel()), counts).to(x.device)
    we = w[e_of]
    y = torch.einsum("mk,mkn->mn", x.double(), we)
    dx = torch.einsum("mn,mkn->mk", dy.double(), we)
    dw = torch.zeros_like(w)
    for e in range(counts.numel()):
        rows = e_of == e
        dw[e] = x.double()[rows].T @ dy.double()[rows]
    if transposed:
        dw = dw.transpose(-2, -1)
    return y, dx, dw


def _rel(a, ref):
    return ((a.double() - ref).norm() / ref.norm().clamp_min(1e-30)).item()


def _calls():
    return dict(MG.GENERIC_CALLS)


# ---------------------------------------------------------------- gate table (capability mocked) ----------------


_KINDS = ("lora", "many", "few", "dw")
_AUTO = {
    (8, 0): {"lora": -1, "many": 2048, "few": 256, "dw": 128},
    (12, 0): {"lora": -1, "many": 2048, "few": 256, "dw": 64},
    (8, 9): {"lora": -1, "many": 32, "few": 32, "dw": 0},
}


@pytest.mark.parametrize("cap", [(7, 5), (8, 0), (8, 6), (8, 7), (8, 9), (9, 0), (10, 0), (12, 0), (12, 1)])
@pytest.mark.parametrize("kind", _KINDS)
def test_gate_table(monkeypatch, cap, kind):
    if MG.triton is None or MU._GROUPED_MM_TRITON_OP is None or torch.version.hip is not None:
        pytest.skip("gate is off without Triton / custom_op / on HIP")
    _mock_cap(cap)
    auto = _AUTO.get(cap, {}).get(kind, 0)
    forced = -1 if cap >= (8, 0) else 0
    assert MU._triton_grouped_mm_max_rows(0, kind) == auto
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "auto")
    assert MU._triton_grouped_mm_max_rows(0, kind) == auto
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "1")
    assert MU._triton_grouped_mm_max_rows(0, kind) == forced
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON_MAX_ROWS", "64")
    assert MU._triton_grouped_mm_max_rows(0, kind) == (64 if forced else 0)
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "auto")
    assert MU._triton_grouped_mm_max_rows(0, kind) == (64 if auto else 0)
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "0")
    assert MU._triton_grouped_mm_max_rows(0, kind) == 0
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "1")
    monkeypatch.setenv("UNSLOTH_DISABLE_MOE_TRITON", "1")
    assert MU._triton_grouped_mm_max_rows(0, kind) == 0


def test_gate_kind():
    assert MU._triton_grouped_mm_kind(2048, 16, 128) == "lora"     # LoRA A
    assert MU._triton_grouped_mm_kind(16, 1536, 128) == "lora"     # LoRA B
    assert MU._triton_grouped_mm_kind(2048, 1536, 128) == "many"   # Qwen3 gate_up
    assert MU._triton_grouped_mm_kind(2048, 1536, 64) == "many"
    assert MU._triton_grouped_mm_kind(4096, 28672, 8) == "few"     # Mixtral
    assert MU._triton_grouped_mm_kind(2880, 5760, 32) == "few"     # gpt-oss
    assert MU._triton_grouped_mm_kind(64, 2048, 128) == "many"     # LoRA r=64: base limits


@needs_cuda
@pytest.mark.parametrize("rows_per_expert, triton_dw", [(100, True), (200, False)])
def test_base_dw_above_limit_uses_torch(rows_per_expert, triton_dw):
    """A trainable base stack's dW (full finetuning) above the "dw" limit goes to torch._grouped_mm while
    forward and dX stay on Triton; both give the fp64 answer."""
    if MU._GROUPED_MM_TRITON_OP is None or MG.triton is None or not MU._check_torch_grouped_mm_supported():
        pytest.skip("no custom_op / Triton / torch._grouped_mm")
    index = torch.cuda.current_device()
    _mock_cap((8, 0), index)                                     # few: 256 rows, dw: 128 rows
    E, K, N = 4, 64, 96
    x, param, dy, offs, counts = _problem([rows_per_expert] * E, K, N, torch.bfloat16)
    before = _calls()
    y, dx, dw = _run(MU._grouped_mm_with_backward_fix, x, param, dy, offs, True)
    after = _calls()
    assert after["gemm"] - before["gemm"] == 2                   # forward + dX on Triton
    assert after["wgrad"] - before["wgrad"] == (1 if triton_dw else 0)
    ref_y, ref_dx, ref_dw = _fp64(x, param, dy, counts, True)
    assert _rel(y, ref_y) < 1e-2 and _rel(dx, ref_dx) < 1e-2 and _rel(dw, ref_dw) < 1e-2


def test_gate_off_on_hip(monkeypatch):
    _mock_cap((8, 0))
    monkeypatch.setattr(torch.version, "hip", "6.4", raising = False)
    MU._TRITON_GROUPED_MM_POLICY.clear()
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "1")
    assert MU._triton_grouped_mm_max_rows(0) == 0


@needs_cuda
def test_gate_rows_threshold_and_operands(monkeypatch):
    if MU._GROUPED_MM_TRITON_OP is None or MG.triton is None:
        pytest.skip("no custom_op / Triton")
    index = torch.cuda.current_device()
    _mock_cap((8, 0), index)
    E, K, N = 4, 40, 48                                          # "few" (< 64 experts, both dims > 32)
    w = torch.empty(E, N, K, device = "cuda", dtype = torch.bfloat16).transpose(-2, -1)
    at = lambda m, dtype = torch.bfloat16, weight = w: MU._triton_grouped_mm_wanted(
        torch.empty(m, weight.shape[1], device = "cuda", dtype = dtype), weight)
    assert at(256 * E) and not at(256 * E + 1)
    many = torch.empty(64, 40, 48, device = "cuda", dtype = torch.bfloat16)
    assert at(2048 * 64, weight = many) and not at(2048 * 64 + 1, weight = many)
    lora = torch.empty(E, 40, 16, device = "cuda", dtype = torch.bfloat16)
    assert at(100000 * E, weight = lora)                          # LoRA class: no row limit
    _mock_cap((8, 9), index)
    assert at(32 * E) and not at(32 * E + 1)
    assert at(100000 * E, weight = lora)
    _mock_cap((9, 0), index)
    assert not at(1)
    _mock_cap((8, 0), index)
    assert not at(8, torch.float32, w.float())                  # fp32
    assert not at(8, torch.float16)                             # mixed dtypes
    assert at(8, torch.float16, w.half())
    assert not MU._triton_grouped_mm_wanted(torch.empty(8, K, dtype = torch.bfloat16), w.cpu())  # CPU
    assert not MU._triton_grouped_mm_wanted(torch.empty(2, 8, K, device = "cuda", dtype = torch.bfloat16), w)
    assert not at(8, torch.bfloat16, torch.empty(0, K, N, device = "cuda", dtype = torch.bfloat16))  # no experts
    class _Sub(torch.Tensor):
        pass
    assert not at(8, torch.bfloat16, w.as_subclass(_Sub))           # tensor subclass (DTensor, float8, ...)
    assert at(8, torch.bfloat16, torch.nn.Parameter(w.detach().contiguous(), requires_grad = False))
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON_MAX_ROWS", "16")
    assert at(16 * E) and not at(16 * E + 1)
    assert at(16 * E, weight = lora) and not at(16 * E + 1, weight = lora)
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "0")
    assert not at(1)


# ---------------------------------------------------------------- numerics vs fp64 -------------------------------

_CASES = {
    "empty_skewed": ([0, 1, 37, 0, 200, 3, 0, 15], 64, 96),
    "odd_dims": ([5, 0, 19, 7], 72, 40),
    "odd_nk": ([9, 33, 0, 2], 33, 17),
    "lora_A_r1": ([12, 0, 40, 7], 64, 1),
    "lora_A_r7": ([12, 0, 40, 7], 64, 7),
    "lora_A_r8": ([12, 0, 40, 7], 64, 8),
    "lora_A_r16": ([12, 0, 40, 7], 64, 16),
    "lora_A_r64": ([12, 0, 40, 7], 64, 64),
    "lora_B_r1": ([12, 0, 40, 7], 1, 96),
    "lora_B_r7": ([12, 0, 40, 7], 7, 96),
    "lora_B_r16": ([12, 0, 40, 7], 16, 96),
    "lora_B_r64": ([12, 0, 40, 7], 64, 96),
    "many_experts": ([3, 0, 1, 2] * 32, 128, 64),
}


def _per_expert_matmul(x, w, offs):
    outs, start = [], 0
    for e, end in enumerate(offs.tolist()):
        outs.append(x[start:end] @ w[e])
        start = end
    return torch.cat(outs)


def _torch_reference(monkeypatch, x, param, dy, offs, transposed):
    """The main-branch path (torch._grouped_mm). Its native backward rejects rows that are not 16-byte
    aligned (LoRA rank 1 / 7: forward_native_grouped_mm pads those ranks first), so there the
    reference is the same-dtype per-expert matmul."""
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "0")
    before = _calls()
    try:
        out = _run(MU._grouped_mm_with_backward_fix, x, param, dy, offs, transposed)
    except RuntimeError as exc:
        if "16 bytes" not in str(exc):
            raise
        out = _run(_per_expert_matmul, x, param, dy, offs, transposed)
    assert _calls() == before
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "1")
    return out


@needs_kernel
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("transposed", [True, False])
@pytest.mark.parametrize("case", list(_CASES))
def test_matches_fp64_at_least_as_well_as_torch(monkeypatch, dtype, transposed, case):
    counts, K, N = _CASES[case]
    x, param, dy, offs, counts_t = _problem(counts, K, N, dtype, transposed)
    ref = _fp64(x, param, dy, counts_t, transposed)
    base = _torch_reference(monkeypatch, x, param, dy, offs, transposed)
    _force_on(monkeypatch)
    before = _calls()
    got = _run(MU._grouped_mm_with_backward_fix, x, param, dy, offs, transposed)
    after = _calls()
    assert after["gemm"] - before["gemm"] == 2 and after["wgrad"] - before["wgrad"] == 1, (before, after)
    for name, g, b, r in zip(("out", "dX", "dW"), got, base, ref):
        assert g.shape == r.shape and g.dtype == dtype, name
        assert torch.isfinite(g).all(), name
        eg, eb = _rel(g, r), _rel(b, r)
        # same fp32 accumulation, one rounding to dtype: at most torch's error (5% slack for summation order)
        assert eg <= eb * 1.05 + 1e-7, (name, eg, eb)


@needs_kernel
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_zero_rows(monkeypatch, dtype):
    _force_on(monkeypatch)
    x, param, dy, offs, _ = _problem([0, 0, 0], 32, 48, dtype)
    y, dx, dw = _run(MU._grouped_mm_with_backward_fix, x, param, dy, offs, True)
    assert y.shape == (0, 48) and dx.shape == (0, 32)
    assert dw.shape == param.shape and (dw == 0).all()


@needs_kernel
def test_rows_past_last_offset_are_ignored(monkeypatch):
    """torch._grouped_mm leaves rows past offs[-1] unwritten (transformers masks its EP sentinel tail); the
    op matches: those rows never feed the valid output or dW, whatever they hold."""
    _force_on(monkeypatch)
    x, param, dy, offs, counts = _problem([6, 0, 11], 40, 24, torch.bfloat16)
    M = x.shape[0]
    tail = 5
    x_pad = torch.cat([x, torch.full((tail, x.shape[1]), float("nan"), device = "cuda", dtype = x.dtype)])
    dy_pad = torch.cat([dy, torch.full((tail, dy.shape[1]), float("nan"), device = "cuda", dtype = dy.dtype)])
    y, dx, dw = _run(MU._grouped_mm_with_backward_fix, x_pad, param, dy_pad, offs, True)
    y0, dx0, dw0 = _run(MU._grouped_mm_with_backward_fix, x, param, dy, offs, True)
    assert torch.equal(y[:M], y0) and torch.equal(dx[:M], dx0) and torch.equal(dw, dw0)


@needs_kernel
def test_engagement_counter_and_modulelist(monkeypatch):
    from unsloth_zoo.temporary_patches import moe_grouped_modulelist as ML
    x, param, dy, offs, _ = _problem([4, 9, 0, 3], 32, 48, torch.bfloat16)
    # auto off here (sm90 / sm100 / sm75); the declined path is torch._grouped_mm itself (absent on torch 2.7)
    if MU._triton_grouped_mm_max_rows(torch.cuda.current_device()) == 0 and hasattr(torch, "_grouped_mm"):
        before = _calls()
        ML._grouped_mm_fix(x, param.transpose(-2, -1), offs)
        MU._grouped_mm_with_backward_fix(x, param.transpose(-2, -1), offs)
        assert _calls() == before
    _force_on(monkeypatch)
    before = _calls()
    y_ml = ML._grouped_mm_fix(x, param.transpose(-2, -1), offs)
    y_mu = MU._grouped_mm_with_backward_fix(x, param.transpose(-2, -1), offs)
    after = _calls()
    assert after["gemm"] - before["gemm"] == 2 and after["fallback"] == before["fallback"]
    assert torch.equal(y_ml, y_mu)


@needs_kernel
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_eager_twin_is_bitwise_the_custom_op(monkeypatch, dtype):
    """Eager calls take the autograd.Function twin (no custom-op dispatch cost), traced calls the op."""
    _force_on(monkeypatch)
    x, param, dy, offs, _ = _problem([12, 0, 40, 7, 1], 64, 96, dtype)
    twin = _run(MU._triton_grouped_mm, x, param, dy, offs, True)
    op = _run(MU._GROUPED_MM_TRITON_OP, x, param, dy, offs, True)
    for a, b in zip(twin, op):
        assert torch.equal(a, b)


@needs_kernel
@pytest.mark.parametrize("backend", ["triton", "cublas"])
def test_ends_matches_counts(backend):
    """grouped_gemm / grouped_wgrad(ends = True) read cumulative ends like rows per expert."""
    x, param, dy, offs, counts = _problem([12, 0, 40, 7, 1], 64, 96, torch.bfloat16)
    c = counts.to("cuda", torch.int32)
    w = param.transpose(-2, -1)
    a = MG.grouped_gemm(x, w, c, x.dtype, b_trans = False, backend = backend)
    b = MG.grouped_gemm(x, w, offs, x.dtype, b_trans = False, backend = backend, ends = True)
    assert torch.equal(a, b)
    a = MG.grouped_wgrad(x, dy, c, x.dtype, num_experts = 5, backend = backend)
    b = MG.grouped_wgrad(x, dy, offs, x.dtype, num_experts = 5, backend = backend, ends = True)
    assert torch.equal(a, b)


# ---------------------------------------------------------------- declined path == main ---------------------------


def _main_branch(inputs, weight, offsets):
    # _grouped_mm_with_backward_fix before the Triton branch (zoo main 3e6a7be3).
    if (
        inputs.dtype == torch.float16 and weight.dtype == torch.float16
        and MU._GROUPED_MM_FP16_OP is not None and torch.compiler.is_compiling()
    ):
        return MU._GROUPED_MM_FP16_OP(inputs, weight, offsets)
    return MU._grouped_mm_eager(inputs, weight, offsets)


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", ["kill_switch", "disable_triton", "sm90_auto", "sm100_auto", "sm80_above_rows"])
def test_declined_path_is_bitwise_main(monkeypatch, dtype, mode):
    if not MU._check_torch_grouped_mm_supported():
        pytest.skip("torch._grouped_mm unsupported here")
    index = torch.cuda.current_device()
    counts = [40, 0, 70, 30]
    if mode == "kill_switch":
        monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "0")
    elif mode == "disable_triton":
        monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "1")
        monkeypatch.setenv("UNSLOTH_DISABLE_MOE_TRITON", "1")
    elif mode == "sm90_auto":
        _mock_cap((9, 0), index)
    elif mode == "sm100_auto":
        _mock_cap((10, 0), index)
    else:
        _mock_cap((8, 0), index)
        monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON_MAX_ROWS", "16")   # 140 rows > 16 * 4
    x, param, dy, offs, _ = _problem(counts, 64, 96, dtype)
    assert not MU._triton_grouped_mm_wanted(x, param.transpose(-2, -1))
    before = _calls()
    head = _run(MU._grouped_mm_with_backward_fix, x, param, dy, offs, True)
    main = _run(_main_branch, x, param, dy, offs, True)
    assert _calls() == before
    for a, b in zip(head, main):
        assert torch.equal(a, b)


def test_declined_branch_is_the_main_code():
    """Everything after the Triton branch is the main-branch body, unchanged."""
    import inspect
    src = inspect.getsource(MU._grouped_mm_with_backward_fix)
    body = src.split('"""')[-1]
    assert body.strip().startswith(
        "if _triton_grouped_mm_wanted(inputs, weight):\n        return _triton_grouped_mm(inputs, weight, offsets)"
    )
    main_body = inspect.getsource(_main_branch).split("zoo main 3e6a7be3).")[-1]
    norm = lambda s: " ".join(s.replace("MU.", "").split())
    assert norm(main_body) in norm(body)


# ---------------------------------------------------------------- self-check fallback -----------------------------


@needs_kernel
def test_self_check_failure_falls_back_to_torch(monkeypatch, caplog):
    if not MU._check_torch_grouped_mm_supported():
        pytest.skip("torch._grouped_mm unsupported here")
    _force_on(monkeypatch)
    monkeypatch.setattr(MG, "_GENERIC_SELF_CHECKED", {})
    monkeypatch.setattr(MG, "_GENERIC_DISABLED_REASON", None)

    def broken(device):
        raise RuntimeError("simulated miscompile")

    monkeypatch.setattr(MG, "_self_check_generic", broken)
    x, param, dy, offs, _ = _problem([5, 0, 19, 7], 64, 96, torch.bfloat16)
    before = _calls()
    with caplog.at_level(logging.WARNING):
        y, dx, dw = _run(MU._grouped_mm_with_backward_fix, x, param, dy, offs, True)
        _run(MU._grouped_mm_with_backward_fix, x, param, dy, offs, True)
    after = _calls()
    assert after["gemm"] == before["gemm"] and after["wgrad"] == before["wgrad"]
    assert after["fallback"] - before["fallback"] == 6     # fwd, dX, dW twice
    assert sum("failed its self-check" in r.getMessage() for r in caplog.records) == 1
    assert "simulated miscompile" in MG._GENERIC_DISABLED_REASON
    y_ref = MU._grouped_mm_eager(x, param.transpose(-2, -1), offs)
    assert torch.equal(y, y_ref)


@needs_kernel
def test_self_check_oom_propagates(monkeypatch):
    _force_on(monkeypatch)
    monkeypatch.setattr(MG, "_GENERIC_SELF_CHECKED", {})
    monkeypatch.setattr(MG, "_GENERIC_DISABLED_REASON", None)

    def oom(device):
        raise torch.OutOfMemoryError("simulated")

    monkeypatch.setattr(MG, "_self_check_generic", oom)
    x, param, dy, offs, _ = _problem([5, 0, 19, 7], 64, 96, torch.bfloat16)
    with pytest.raises(torch.OutOfMemoryError):
        MU._grouped_mm_with_backward_fix(x, param.transpose(-2, -1), offs)
    assert MG._GENERIC_DISABLED_REASON is None


@needs_kernel
def test_real_self_check_passes():
    MG._self_check_generic(torch.device("cuda", torch.cuda.current_device()))
    assert MG.triton_grouped_available(torch.device("cuda", torch.cuda.current_device()))


# ---------------------------------------------------------------- opcheck / compile --------------------------------


@needs_kernel
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_opcheck(dtype):
    if not hasattr(torch.library, "opcheck"):
        pytest.skip("torch.library.opcheck missing")
    x, param, dy, offs, _ = _problem([5, 0, 19, 7], 64, 96, dtype)
    w = param.detach().clone().requires_grad_(True)
    torch.library.opcheck(torch.ops.unsloth_zoo.grouped_mm_triton.default,
                          (x.clone().requires_grad_(True), w.transpose(-2, -1), offs))
    torch.library.opcheck(torch.ops.unsloth_zoo.grouped_mm_triton.default,
                          (x, param.transpose(-2, -1).contiguous(), offs))
    torch.library.opcheck(torch.ops.unsloth_zoo.grouped_mm_triton_wgrad.default, (x, dy, offs))


def _lora_block(x, base, lora_a, lora_b, offs):
    # frozen base through a transposed view + a LoRA pair, as forward_native_grouped_mm runs them
    y = MU._grouped_mm_with_backward_fix(x, base.transpose(-2, -1), offs)
    h = MU._grouped_mm_with_backward_fix(x, lora_a, offs)
    return y + 2.0 * MU._grouped_mm_with_backward_fix(h, lora_b, offs)


def _lora_case(M_counts, dtype, seed = 0):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    E, K, N, R = len(M_counts), 64, 96, 16
    counts = torch.tensor(M_counts)
    x = torch.randn(int(counts.sum()), K, generator = g).to("cuda", dtype)
    base = (torch.randn(E, N, K, generator = g) * 0.1).to("cuda", dtype)
    a = (torch.randn(E, K, R, generator = g) * 0.1).to("cuda", dtype)
    b = (torch.randn(E, R, N, generator = g) * 0.1).to("cuda", dtype)
    offs = counts.cumsum(0).to("cuda", torch.int32)
    return x, base, a, b, offs


def _lora_run(fn, case):
    x, base, a, b, offs = case
    x = x.clone().requires_grad_(True)
    a = a.clone().requires_grad_(True)
    b = b.clone().requires_grad_(True)
    y = fn(x, base, a, b, offs)
    (y.float().square().sum()).backward()
    return y.detach(), x.grad, a.grad, b.grad


@needs_kernel
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_compile_fullgraph_matches_eager(monkeypatch, dtype):
    _force_on(monkeypatch)
    torch._dynamo.reset()
    case = _lora_case([30, 0, 50, 20], dtype)
    eager = _lora_run(_lora_block, case)
    before = _calls()
    compiled_fn = torch.compile(_lora_block, fullgraph = True)
    compiled = _lora_run(compiled_fn, case)
    after = _calls()
    assert after["gemm"] - before["gemm"] == 6 and after["wgrad"] - before["wgrad"] == 2, (before, after)
    for name, e, c in zip(("out", "dX", "dA", "dB"), eager, compiled):
        assert torch.isfinite(c).all(), name
        # the GEMMs are opaque and deterministic; only the inductor glue (y + 2 * delta) may round differently
        assert torch.allclose(e.float(), c.float(), rtol = 2e-2, atol = 2e-2), name
    for name, e, c in zip(("dA", "dB"), eager[2:], compiled[2:]):
        assert torch.equal(e, c) or torch.allclose(e.float(), c.float(), rtol = 1e-2, atol = 1e-2), name


@needs_kernel
def test_compile_ops_bitwise_eager(monkeypatch):
    """The op alone, compiled: output and grads bitwise the eager op's."""
    _force_on(monkeypatch)
    torch._dynamo.reset()
    x, param, dy, offs, _ = _problem([30, 0, 50, 20], 64, 96, torch.bfloat16)
    fn = lambda x, w, offs: MU._grouped_mm_with_backward_fix(x, w, offs)
    eager = _run(fn, x, param, dy, offs, True)
    compiled = _run(torch.compile(fn, fullgraph = True), x, param, dy, offs, True)
    for e, c in zip(eager, compiled):
        assert torch.equal(e, c)


@needs_kernel
def test_compile_recompiles_at_most_once_across_threshold(monkeypatch):
    """Under dynamic shapes M <= limit * E is a guard: crossing it recompiles once, then nothing."""
    if not MU._check_torch_grouped_mm_supported():
        pytest.skip("torch._grouped_mm unsupported here")
    from torch._dynamo.testing import CompileCounterWithBackend
    _force_on(monkeypatch, max_rows = 16)     # E = 4: Triton up to 64 rows
    torch._dynamo.reset()
    counter = CompileCounterWithBackend("inductor")
    compiled_fn = torch.compile(_lora_block, fullgraph = True, dynamic = True, backend = counter)
    seen = []
    for counts in ([8, 0, 20, 12], [10, 5, 20, 12], [16, 16, 16, 16], [40, 0, 50, 20], [60, 10, 50, 30], [9, 3, 7, 11]):
        case = _lora_case(counts, torch.bfloat16)
        before = _calls()
        out = _lora_run(compiled_fn, case)
        ref = _lora_run(_lora_block, case)
        engaged = _calls()["gemm"] > before["gemm"]
        seen.append((sum(counts), engaged))
        assert engaged == (sum(counts) <= 64), seen
        assert torch.allclose(out[0].float(), ref[0].float(), rtol = 2e-2, atol = 2e-2)
    assert counter.frame_count <= 2, (counter.frame_count, seen)


@needs_kernel
def test_compile_kill_switch_never_reaches_kernel(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "0")
    if not MU._check_torch_grouped_mm_supported():
        pytest.skip("torch._grouped_mm unsupported here")
    torch._dynamo.reset()
    case = _lora_case([30, 0, 50, 20], torch.bfloat16)
    before = _calls()
    _lora_run(torch.compile(_lora_block, fullgraph = True), case)
    assert _calls() == before


# ---------------------------------------------------------------- transformers 5.x _grouped_mm ---------------------


def _transformers_moe():
    try:
        import transformers.integrations.moe as tm
    except Exception:
        return None
    return tm if hasattr(tm, "_grouped_mm") else None


@needs_kernel
def test_transformers_grouped_mm_wrapper(monkeypatch):
    tm = _transformers_moe()
    if tm is None:
        pytest.skip("transformers without integrations.moe._grouped_mm")
    from unsloth_zoo.temporary_patches import moe_experts_interface as MEI
    MEI._patch_transformers_grouped_mm()
    wrapped = tm._grouped_mm
    assert getattr(wrapped, "_unsloth_patched", False)
    MEI._patch_transformers_grouped_mm()
    assert tm._grouped_mm is wrapped                     # idempotent
    original = wrapped._unsloth_original

    x, param, dy, offs, counts = _problem([5, 0, 19, 7], 64, 96, torch.bfloat16)
    # declined (auto on sm90 / sm100, kill switch): the original's output, bitwise
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "0")
    before = _calls()
    assert torch.equal(tm._grouped_linear(x, param, offs), original(x, param.transpose(-2, -1), offs))
    assert _calls() == before
    # engaged: through the op, with grads
    _force_on(monkeypatch)
    got = _run(lambda a, w, o: tm._grouped_linear(a, w.transpose(-2, -1), o, is_transposed = False),
               x, param, dy, offs, True)
    after = _calls()
    assert after["gemm"] - before["gemm"] == 2 and after["wgrad"] - before["wgrad"] == 1
    ref = _fp64(x, param, dy, counts, True)
    for name, g, r in zip(("out", "dX", "dW"), got, ref):
        assert _rel(g, r) < 1e-2, name
    # mixed dtypes (fp32 activations, bf16 experts): the original, which casts
    before = _calls()
    xf = x.float()
    assert torch.equal(tm._grouped_mm(xf, param.transpose(-2, -1), offs), original(xf, param.transpose(-2, -1), offs))
    assert _calls() == before

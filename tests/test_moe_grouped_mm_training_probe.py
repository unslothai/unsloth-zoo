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

"""A grouped_mm that serves one fp16 expert but refuses bf16 multi-expert (Windows wheels) must not be selected; a
later refusal falls back once; an OOM never latches. CPU only."""

import pytest
import torch
import torch.nn as nn

from unsloth_zoo.temporary_patches import moe_utils as mu

REFUSAL = "grouped mm is not supported on your system"
CPU = torch.device("cpu")


def _reference_grouped_mm(x, w, offs = None):
    ends = offs.tolist()
    rows = [x[s:e] @ w[g] for g, (s, e) in enumerate(zip([0] + ends[:-1], ends))]
    return torch.cat(rows) if rows else x.new_empty((0, w.shape[-1]))


def _windows_grouped_mm(calls):
    """Serves one fp16 expert; refuses bf16 or several experts, as the Windows wheels' bf16 kernel stub does."""
    def grouped_mm(x, w, offs = None):
        calls.append((x.dtype, int(offs.numel())))
        if x.dtype == torch.bfloat16 or offs.numel() > 1:
            raise RuntimeError(REFUSAL)
        return _reference_grouped_mm(x, w, offs)
    return grouped_mm


@pytest.fixture
def fresh(monkeypatch):
    for name, value in (
        ("_TORCH_GROUPED_MM_AVAILABLE", True), ("_TORCH_GROUPED_MM_SUPPORTED", None),
        ("_TRANSPOSED_VIEW_GROUPED_MM_SAFE", None), ("_GROUPED_MM_HOME", None), ("_GROUPED_MM_HOMOGENEOUS", False),
        ("_GROUPED_MM_SURVIVES", None), ("_SELECTED_MOE_BACKEND", None),
        ("_GROUPED_MM_TRAINING_PROBE_FAILURE", None), ("_GROUPED_MM_TRITON_STANDIN", False),
    ):
        monkeypatch.setattr(mu, name, value, raising = False)
    for name in ("_GROUPED_MM_FOREIGN", "_TORCH_GROUPED_MM_SUPPORTED_BY_KIND", "_GROUPED_MM_SURVIVES_BY_KIND",
                 "_TRANSPOSED_VIEW_SAFE_BY_KIND", "_TRITON_GROUPED_MM_POLICY"):
        monkeypatch.setattr(mu, name, {}, raising = False)
    monkeypatch.setattr(mu, "_GROUPED_MM_FALLBACK_LOGGED", set(), raising = False)
    monkeypatch.setattr(mu, "_GROUPED_MM_RUNTIME_REFUSALS", [0], raising = False)
    monkeypatch.setattr(mu, "_GROUPED_MM_LAST_REFUSAL", [None], raising = False)
    monkeypatch.setattr(mu, "_grouped_mm_probe_device", lambda: CPU)
    monkeypatch.setattr(mu, "_check_grouped_gemm_available", lambda: False)
    for var in ("UNSLOTH_MOE_BACKEND", "UNSLOTH_MOE_GROUPED_TRITON", "UNSLOTH_DISABLE_MOE_TRITON",
                "UNSLOTH_MOE_GROUPED_MM_PROBE"):
        monkeypatch.delenv(var, raising = False)
    return monkeypatch


class _Experts(nn.Module):

    def __init__(self, E = 4, H = 32, I = 16, rank = 4):
        super().__init__()
        gen = torch.Generator().manual_seed(0)
        self.num_experts = E
        self.gate_up_proj = nn.Parameter((torch.randn(E, 2 * I, H, generator = gen) * H ** -0.5).bfloat16(), requires_grad = False)
        self.down_proj = nn.Parameter((torch.randn(E, H, I, generator = gen) * I ** -0.5).bfloat16(), requires_grad = False)
        self.act_fn = nn.SiLU()
        lora = lambda *s: (torch.randn(*s, generator = gen) * 0.1).bfloat16().requires_grad_(True)
        self.lora = {"gate_up_proj": (lora(E, H, rank), lora(E, rank, 2 * I), 2.0),
                     "down_proj": (lora(E, I, rank), lora(E, rank, H), 2.0)}


def _batch(T = 24, H = 32, E = 4, k = 2):
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(T, H, generator = gen).bfloat16()
    logits = torch.randn(T, E, generator = gen)
    logits[:, 1] = -1e4
    top_v, top_i = torch.topk(logits, k, dim = -1)
    return x, top_i, torch.softmax(top_v, dim = -1).bfloat16()


def _train_step(monkeypatch, forward):
    module = _Experts()
    monkeypatch.setattr(mu, "take_moe_lora_stash", lambda m, name: module.lora.get(name))
    x, top_i, top_w = _batch()
    loss = forward(module, x, top_i, top_w).float().square().mean()
    loss.backward()
    return loss.item(), [t.grad.float() for pair in module.lora.values() for t in pair[:2]]


def _selected_forward(module, x, top_i, top_w):
    backend = mu.select_moe_backend()
    if backend == "grouped_mm":
        return mu.forward_native_grouped_mm(module, x, top_i, top_w)
    return mu.forward_native_moe_loop(module, x, top_i, top_w)


def test_windows_refusal_selects_the_fallback_and_trains_like_native(fresh, capsys):
    calls = []
    fresh.setattr(torch, "_grouped_mm", _windows_grouped_mm(calls), raising = False)
    loss, grads = _train_step(fresh, _selected_forward)
    assert mu.select_moe_backend() == "native_torch"
    assert mu._check_torch_grouped_mm_supported() is False
    assert calls == [(torch.float16, 1), (torch.bfloat16, 4)]
    out = capsys.readouterr().out
    assert REFUSAL in out and "MoE backend 'native_torch'" in out and "UNSLOTH_MOE_GROUPED_MM_PROBE=0" in out

    fresh.setattr(mu, "_SELECTED_MOE_BACKEND", "native_torch")
    native_loss, native_grads = _train_step(fresh, mu.forward_native_moe_loop)
    assert loss == native_loss
    for got, want in zip(grads, native_grads):
        assert torch.equal(got, want)


def test_explicit_grouped_mm_runs_on_the_triton_stand_in(fresh, capsys):
    calls = []
    fresh.setattr(torch, "_grouped_mm", _windows_grouped_mm(calls), raising = False)
    fresh.setenv("UNSLOTH_MOE_BACKEND", "grouped_mm")
    fresh.setattr(mu, "_grouped_mm_triton_standin_available", lambda device: True)
    assert mu._select_moe_backend() == "grouped_mm"
    assert mu._GROUPED_MM_TRITON_STANDIN is True
    assert "Unsloth's Triton grouped GEMM kernels" in capsys.readouterr().out
    fresh.setattr(mu, "_SELECTED_MOE_BACKEND", "grouped_mm")
    del calls[:]
    loss, grads = _train_step(fresh, mu.forward_native_grouped_mm)
    assert calls == []
    native_loss, native_grads = _train_step(fresh, mu.forward_native_moe_loop)
    assert loss == pytest.approx(native_loss, rel = 2e-2)
    for got, want in zip(grads, native_grads):
        torch.testing.assert_close(got, want, rtol = 5e-2, atol = 5e-3)


def test_auto_prefers_the_triton_stand_in_for_a_training_refusal(fresh):
    fresh.setattr(torch, "_grouped_mm", _windows_grouped_mm([]), raising = False)
    fresh.setattr(mu, "_grouped_mm_triton_standin_available", lambda device: True)
    assert mu._select_moe_backend() == "grouped_mm"
    assert mu._GROUPED_MM_TRITON_STANDIN is True


@pytest.mark.parametrize("explicit", [False, True])
def test_stand_in_opt_out_keeps_the_backend_chain(fresh, explicit):
    fresh.setattr(torch, "_grouped_mm", _windows_grouped_mm([]), raising = False)
    fresh.setenv("UNSLOTH_MOE_GROUPED_TRITON", "0")
    if explicit:
        fresh.setenv("UNSLOTH_MOE_BACKEND", "grouped_mm")
    assert mu._select_moe_backend() == "native_torch"
    assert mu._GROUPED_MM_TRITON_STANDIN is False


def test_a_failing_single_expert_call_keeps_the_old_auto_chain(fresh):
    # torch with no working op at all (2.8 off sm90): auto never switches to the stand-in, only an explicit request does.
    def refuse_all(x, w, offs = None):
        raise RuntimeError("torch._grouped_mm is only supported on CUDA devices with compute capability = 9.0")
    fresh.setattr(torch, "_grouped_mm", refuse_all, raising = False)
    fresh.setattr(mu, "_grouped_mm_triton_standin_available", lambda device: True)
    assert mu._select_moe_backend() == "native_torch"
    assert mu._GROUPED_MM_TRAINING_PROBE_FAILURE is None


def test_stand_in_forces_triton_for_every_kind(fresh):
    fresh.setattr(mu, "_triton_grouped_mm_capability", lambda index: (10, 0))
    from unsloth_zoo.temporary_patches import moe_grouped_fp16
    if moe_grouped_fp16.triton is None or mu._GROUPED_MM_TRITON_OP is None:
        pytest.skip("needs triton and the grouped_mm_triton op")
    fresh.setattr(mu, "is_rocm_torch", lambda torch_module: False)
    for kind in ("lora", "many", "few", "dw", "dw_few"):
        assert mu._triton_grouped_mm_policy(0, kind, None, None, None) == 0
        assert mu._triton_grouped_mm_policy(0, kind, None, None, None, True) == -1
        assert mu._triton_grouped_mm_policy(0, kind, "0", None, None, True) == 0


def test_a_correct_grouped_mm_still_selects_grouped_mm(fresh, capsys):
    calls = []
    def grouped_mm(x, w, offs = None):
        calls.append((x.dtype, int(offs.numel())))
        return _reference_grouped_mm(x, w, offs)
    fresh.setattr(torch, "_grouped_mm", grouped_mm, raising = False)
    cpu_rng = torch.get_rng_state()
    assert mu._select_moe_backend() == "grouped_mm"
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert calls[:2] == [(torch.float16, 1), (torch.bfloat16, 4)]
    assert mu._GROUPED_MM_TRAINING_PROBE_FAILURE is None and mu._GROUPED_MM_TRITON_STANDIN is False
    assert "cannot run MoE training" not in capsys.readouterr().out


@pytest.mark.parametrize("defect", ["zeros", "dx", "empty_dw", "no_backward"])
def test_the_probe_checks_forward_and_both_grads(fresh, defect):
    class Wrong(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w, offs):
            ctx.save_for_backward(x, w, offs)
            return torch.zeros(x.shape[0], w.shape[-1], dtype = x.dtype) if defect == "zeros" else \
                _reference_grouped_mm(x, w, offs)
        @staticmethod
        def backward(ctx, grad):
            if defect == "no_backward":
                raise RuntimeError(REFUSAL)
            x, w, offs = ctx.saved_tensors
            with torch.enable_grad():
                x_, w_ = x.detach().requires_grad_(), w.detach().requires_grad_()
                dx, dw = torch.autograd.grad(_reference_grouped_mm(x_, w_, offs), (x_, w_), grad)
            if defect == "dx":
                dx = dx * 2
            if defect == "empty_dw":
                dw[1] = 1
            return dx, dw, None
    def grouped_mm(x, w, offs = None):
        return _reference_grouped_mm(x, w, offs) if offs.numel() == 1 else Wrong.apply(x, w, offs)
    fresh.setattr(torch, "_grouped_mm", grouped_mm, raising = False)
    assert mu._check_torch_grouped_mm_supported() is False
    assert mu._GROUPED_MM_TRAINING_PROBE_FAILURE


def test_probe_opt_out_restores_the_single_expert_check(fresh):
    fresh.setattr(torch, "_grouped_mm", _windows_grouped_mm([]), raising = False)
    fresh.setenv("UNSLOTH_MOE_GROUPED_MM_PROBE", "0")
    assert mu._check_torch_grouped_mm_supported() is True


def test_an_oom_in_the_probe_is_no_verdict(fresh):
    def grouped_mm(x, w, offs = None):
        if offs.numel() > 1:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 MiB")
        return _reference_grouped_mm(x, w, offs)
    fresh.setattr(torch, "_grouped_mm", grouped_mm, raising = False)
    assert mu._check_torch_grouped_mm_supported() is True
    assert mu._GROUPED_MM_TRAINING_PROBE_FAILURE is None


def test_probe_dtype_is_the_training_compute_dtype(monkeypatch):
    monkeypatch.setattr(mu, "is_rocm_torch", lambda torch_module: False)
    for capability, dtype in (((7, 5), torch.float16), ((8, 0), torch.bfloat16), ((10, 0), torch.bfloat16)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None, c = capability: c)
        assert mu._grouped_mm_probe_dtype(torch.device("cuda", 0)) is dtype
    monkeypatch.setattr(mu, "is_rocm_torch", lambda torch_module: True)
    assert mu._grouped_mm_probe_dtype(torch.device("cuda", 0)) is torch.bfloat16
    assert mu._grouped_mm_probe_dtype(torch.device("xpu", 0)) is torch.bfloat16


def _supported_then(fresh, behaviour):
    calls = []
    def grouped_mm(x, w, offs = None):
        calls.append(x.dtype)
        return behaviour(x, w, offs)
    fresh.setattr(mu, "_TORCH_GROUPED_MM_SUPPORTED", True)
    fresh.setattr(mu, "_TRANSPOSED_VIEW_GROUPED_MM_SAFE", True)
    fresh.setattr(torch, "_grouped_mm", grouped_mm, raising = False)
    x = torch.randn(10, 32).bfloat16()
    w = torch.randn(4, 32, 16).bfloat16()
    offs = torch.tensor([3, 3, 7, 10], dtype = torch.int32)
    return calls, x, w, offs


def test_a_real_call_refusal_falls_back_once_for_the_process(fresh, capsys):
    def refuse(x, w, offs):
        raise RuntimeError(f"{REFUSAL}\nException raised from bf16bf16_grouped_mm")
    calls, x, w, offs = _supported_then(fresh, refuse)
    got = mu._grouped_mm_with_backward_fix(x, w, offs)
    torch.testing.assert_close(got.float(), _reference_grouped_mm(x, w, offs).float(), rtol = 2e-2, atol = 2e-2)
    assert calls == [torch.bfloat16]
    assert mu._TORCH_GROUPED_MM_SUPPORTED is False and mu._GROUPED_MM_RUNTIME_REFUSALS == [1]
    assert REFUSAL in capsys.readouterr().out
    mu._grouped_mm_with_backward_fix(x, w, offs)
    assert calls == [torch.bfloat16]
    assert mu._GROUPED_MM_RUNTIME_REFUSALS == [1]


def test_a_runtime_oom_propagates_and_never_latches(fresh):
    state = {"oom": True}
    def maybe_oom(x, w, offs):
        if state["oom"]:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 GiB")
        return _reference_grouped_mm(x, w, offs)
    calls, x, w, offs = _supported_then(fresh, maybe_oom)
    with pytest.raises(torch.OutOfMemoryError):
        mu._grouped_mm_with_backward_fix(x, w, offs)
    assert mu._TORCH_GROUPED_MM_SUPPORTED is True and mu._GROUPED_MM_RUNTIME_REFUSALS == [0]
    state["oom"] = False
    mu._grouped_mm_with_backward_fix(x, w, offs)
    assert calls == [torch.bfloat16, torch.bfloat16]


def test_other_runtime_errors_still_raise(fresh):
    def shape_error(x, w, offs):
        raise RuntimeError("mat_a and mat_b shapes cannot be multiplied")
    calls, x, w, offs = _supported_then(fresh, shape_error)
    with pytest.raises(RuntimeError, match = "cannot be multiplied"):
        mu._grouped_mm_with_backward_fix(x, w, offs)
    assert mu._TORCH_GROUPED_MM_SUPPORTED is True


def test_the_fallback_line_prints_once_across_the_compiled_cache_copy(fresh, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("unsloth_cached_moe_utils_g97", mu.__file__)
    copy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copy)
    mu._log_grouped_mm_fallback("bfloat16, 4 experts: refused", "MoE backend 'native_torch'")
    copy._log_grouped_mm_fallback("bfloat16, 4 experts: refused", "MoE backend 'native_torch'")
    assert capsys.readouterr().out.count("cannot run MoE training") == 1

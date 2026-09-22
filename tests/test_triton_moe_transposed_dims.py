# SPDX-License-Identifier: AGPL-3.0-only
"""forward_triton_grouped_gemm must tune its GEMMs for the real intermediate size
whether the experts store gate_up as (E, 2I, H) or transposed as (E, H, 2I) (Llama-4)."""
import pytest
import torch
import torch.nn as nn

from unsloth_zoo.temporary_patches import moe_utils


class _Stop(Exception):
    pass


def _experts(E, H, I, transposed):
    m = nn.Module()
    shape = (E, H, 2 * I) if transposed else (E, 2 * I, H)
    m.gate_up_proj = nn.Parameter(torch.zeros(shape))
    m.down_proj = nn.Parameter(torch.zeros((E, I, H) if transposed else (E, H, I)))
    m.num_experts = E
    m._unsloth_moe_configs = None
    return m


@pytest.mark.parametrize("transposed", [False, True])
def test_triton_backend_tunes_for_the_real_intermediate_size(monkeypatch, transposed):
    E, H, I = 4, 32, 48
    seen = []

    def fake_autotune(*args, **kwargs):
        seen.append(kwargs)
        raise _Stop()

    autotune_cache = pytest.importorskip("unsloth.kernels.moe.autotune_cache")
    monkeypatch.setattr(autotune_cache, "get_or_autotune_moe_kernels", fake_autotune)
    experts = _experts(E, H, I, transposed)
    hidden = torch.zeros(3, H)
    top_k_index = torch.zeros(3, 1, dtype = torch.long)
    top_k_weights = torch.ones(3, 1)
    with pytest.raises(_Stop):
        moe_utils.forward_triton_grouped_gemm(experts, hidden, top_k_index, top_k_weights)
    assert seen and seen[0]["hidden_dim"] == H
    assert seen[0]["intermediate_dim"] == 2 * I


@pytest.mark.parametrize("name, interleaved", [("GptOssExperts", True), ("Qwen3MoeExperts", False)])
def test_triton_backend_is_skipped_for_interleaved_gate_up(monkeypatch, name, interleaved):
    # The Triton kernels chunk gate_up into halves with SiLU and no bias, which is not
    # GPT-OSS's interleaved, clamped, biased activation.
    calls = []
    monkeypatch.setattr(moe_utils, "select_moe_backend", lambda: "unsloth_triton")
    monkeypatch.setattr(moe_utils, "forward_triton_grouped_gemm", lambda *a: calls.append("triton"))
    monkeypatch.setattr(moe_utils, "forward_native_moe_loop", lambda *a: calls.append("loop"))
    experts = type(name, (nn.Module,), {})()
    experts.gate_up_proj = nn.Parameter(torch.zeros(2, 8, 4))
    moe_utils.forward_moe_backend(experts, torch.zeros(1, 4), torch.zeros(1, 1, dtype = torch.long), torch.ones(1, 1))
    assert calls == (["loop"] if interleaved else ["triton"])


@pytest.mark.parametrize("quant", ["bnb4bit", "fp8"])
@pytest.mark.parametrize("name, interleaved", [("GptOssExperts", True), ("Qwen3MoeExperts", False)])
def test_quantized_dispatchers_skip_triton_for_interleaved_gate_up(monkeypatch, quant, name, interleaved):
    from unsloth_zoo.temporary_patches import moe_utils_bnb4bit, moe_utils_fp8

    calls = []
    monkeypatch.setattr(moe_utils, "select_moe_backend", lambda: "unsloth_triton")
    monkeypatch.setattr(moe_utils, "forward_triton_grouped_gemm", lambda *a: calls.append("triton"))
    monkeypatch.setattr(moe_utils, "forward_native_moe_loop", lambda *a: calls.append("loop"))
    experts = type(name, (nn.Module,), {})()
    experts.gate_up_proj = nn.Parameter(torch.zeros(2, 8, 4))
    experts.down_proj = nn.Parameter(torch.zeros(2, 4, 4))
    dense = torch.zeros(2, 8, 4)
    if quant == "bnb4bit":
        monkeypatch.setattr(moe_utils_bnb4bit, "_dequantize_bnb4bit_expert_weights", lambda w, dtype: dense)
        dispatch = moe_utils_bnb4bit.forward_moe_backend_bnb4bit
    else:
        monkeypatch.setattr(moe_utils_fp8, "_get_moe_weight_and_quant_info", lambda m, n: (dense, None, None))
        monkeypatch.setattr(moe_utils_fp8, "_dequantize_full_expert_weights", lambda *a, **k: dense)
        dispatch = moe_utils_fp8.forward_moe_backend_fp8
    dispatch(experts, torch.zeros(1, 4), torch.zeros(1, 1, dtype = torch.long), torch.ones(1, 1))
    assert calls == (["loop"] if interleaved else ["triton"])


@pytest.mark.parametrize("declared", [True, False])
def test_square_stacks_follow_the_declared_layout(monkeypatch, declared):
    # Llama-4 with 2I == H stores gate_up as a square (E, H, H): the shape cannot tell the
    # orientation, so the declared is_transposed has to pick w1.
    E, H = 4, 32
    I = H // 2
    autotune_cache = pytest.importorskip("unsloth.kernels.moe.autotune_cache")
    interface = pytest.importorskip("unsloth.kernels.moe.grouped_gemm.interface")
    monkeypatch.setattr(autotune_cache, "get_or_autotune_moe_kernels", lambda **k: (None, None, None))
    seen = []

    def fake_grouped_gemm(**kwargs):
        seen.append(kwargs["W"])
        raise _Stop()

    monkeypatch.setattr(interface, "grouped_gemm", fake_grouped_gemm)
    experts = _experts(E, H, I, transposed = declared)
    experts.gate_up_proj = nn.Parameter(torch.randn(E, H, 2 * I))
    experts.is_transposed = declared
    hidden = torch.zeros(3, H)
    with pytest.raises(_Stop):
        moe_utils.forward_triton_grouped_gemm(
            experts, hidden, torch.zeros(3, 1, dtype = torch.long), torch.ones(3, 1)
        )
    want = experts.gate_up_proj.transpose(-2, -1) if declared else experts.gate_up_proj
    assert torch.equal(seen[0], want)


def test_native_loop_follows_the_declared_layout_for_square_stacks():
    # native_torch fallback: a declared is_transposed square (E, H, H) gate_up must be
    # transposed like the Triton path does.
    E, H = 2, 8
    I = H // 2
    torch.manual_seed(0)
    experts = _experts(E, H, I, transposed = True)
    experts.gate_up_proj = nn.Parameter(torch.randn(E, H, 2 * I))
    experts.down_proj = nn.Parameter(torch.randn(E, I, H))
    experts.is_transposed = True
    experts.act_fn = torch.nn.functional.silu
    hidden = torch.randn(3, H)
    top_k_index = torch.zeros(3, 1, dtype = torch.long)
    top_k_weights = torch.ones(3, 1)
    got = moe_utils.forward_native_moe_loop(experts, hidden, top_k_index, top_k_weights)
    gate, up = (hidden @ experts.gate_up_proj[0]).chunk(2, dim = -1)
    want = (torch.nn.functional.silu(gate) * up) @ experts.down_proj[0]
    torch.testing.assert_close(got, want)

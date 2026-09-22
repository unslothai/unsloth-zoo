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

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Capability decisions follow the operand's GPU on a mixed host (CPU only, devices mocked):
the Triton MoE caller hands the operands' device to the config picker, and the SDPA mem-efficient
eligibility check is asked on the query's card, not the current one."""

import contextlib
import sys
import types

import pytest
import torch

from unsloth_zoo.temporary_patches import moe_utils as M
from unsloth_zoo.temporary_patches import sdpa_gqa


def _tuner_with_device(num_experts, hidden_dim, intermediate_dim, top_k, dtype, force_autotune = False,
                       seq_len = 8192, weight_dtype = None, device = None):
    return None


def _tuner_without_device(num_experts, hidden_dim, intermediate_dim, top_k, dtype, force_autotune = False,
                          seq_len = 8192, weight_dtype = None):
    return None


def test_config_picker_gets_the_operand_cuda_device():
    experts = types.SimpleNamespace(gate_up_proj = torch.zeros(1, dtype = torch.bfloat16))
    card = torch.device("cuda", 1)
    assert M._moe_weight_dtype_kwargs(_tuner_with_device, experts, "gate_up_proj", torch.bfloat16, card) == {"device": card}
    assert M._moe_weight_dtype_kwargs(_tuner_with_device, experts, "gate_up_proj", torch.float32, card) == {
        "weight_dtype": torch.bfloat16, "device": card,
    }


def test_older_unsloth_and_non_cuda_keep_the_old_kwargs():
    experts = types.SimpleNamespace(gate_up_proj = torch.zeros(1, dtype = torch.bfloat16))
    card = torch.device("cuda", 1)
    for device in (card, None):
        assert M._moe_weight_dtype_kwargs(_tuner_without_device, experts, "gate_up_proj", torch.float32, device) == {
            "weight_dtype": torch.bfloat16,
        }
    assert M._moe_weight_dtype_kwargs(_tuner_with_device, experts, "gate_up_proj", torch.bfloat16, torch.device("cpu")) == {}
    assert M._moe_weight_dtype_kwargs(_tuner_with_device, experts, "gate_up_proj", torch.bfloat16) == {}


class _Stop(Exception):
    pass


def test_triton_forward_passes_the_hidden_states_device(monkeypatch):
    asked = []

    def kwargs(tuner, module, name, dtype, device = None):
        asked.append((name, device))
        return {}

    def fake_autotune(**kw):
        if len(asked) == 2:
            raise _Stop
        return ("fwd", "dX", "dW")

    autotune = types.ModuleType("unsloth.kernels.moe.autotune_cache")
    autotune.get_or_autotune_moe_kernels = fake_autotune
    interface = types.ModuleType("unsloth.kernels.moe.grouped_gemm.interface")
    interface.grouped_gemm = None
    monkeypatch.setitem(sys.modules, autotune.__name__, autotune)
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    monkeypatch.setattr(M, "_moe_weight_dtype_kwargs", kwargs)
    E, I, H = 4, 48, 64
    experts = torch.nn.Module()
    experts.num_experts = E
    experts.gate_up_proj = torch.nn.Parameter(torch.zeros(E, 2 * I, H))
    experts.down_proj = torch.nn.Parameter(torch.zeros(E, H, I))
    x = torch.zeros(3, H)
    with pytest.raises(_Stop):
        M.forward_triton_grouped_gemm(experts, x, torch.zeros(3, 2, dtype = torch.long), torch.zeros(3, 2))
    assert asked == [("gate_up_proj", x.device), ("down_proj", x.device)]


@pytest.fixture
def two_cards(monkeypatch):
    # Current cuda:0 rejects the mem-efficient kernel for this call, the query's cuda:1 accepts it.
    state = {"current": 0, "switches": 0}

    @contextlib.contextmanager
    def device(d):
        previous, state["current"] = state["current"], torch.device(d).index
        state["switches"] += 1
        try:
            yield
        finally:
            state["current"] = previous

    monkeypatch.setattr(torch.cuda, "current_device", lambda: state["current"])
    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch.backends.cuda, "SDPAParams", lambda *a: a, raising = False)
    monkeypatch.setattr(
        torch.backends.cuda, "can_use_efficient_attention", lambda params, debug: state["current"] == 1, raising = False,
    )
    return state


def _query(index):
    return types.SimpleNamespace(is_cuda = True, device = torch.device("cuda", index))


def test_efficient_check_asks_the_query_card(two_cards):
    assert sdpa_gqa._efficient_takes(_query(1), None, None, None, 0.0, True, False) is True
    assert two_cards["current"] == 0


def test_efficient_check_on_the_current_card_does_not_switch(two_cards):
    two_cards["current"] = 1
    assert sdpa_gqa._efficient_takes(_query(1), None, None, None, 0.0, True, False) is True
    two_cards["current"] = 0
    assert sdpa_gqa._efficient_takes(_query(0), None, None, None, 0.0, True, False) is False
    assert two_cards["switches"] == 0


def test_rejected_repeated_kv_is_freed_before_the_fallback(monkeypatch):
    # The repeat is rejected too (fp32 on ROCm CK): its k/v copies must not stay alive during the fallback call.
    import weakref
    repeated = []

    def efficient(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa):
        if not enable_gqa:
            repeated.extend((weakref.ref(key), weakref.ref(value)))
        return False

    monkeypatch.setattr(sdpa_gqa, "_efficient_takes", efficient)
    monkeypatch.setattr(sdpa_gqa, "_flash_takes_gqa", lambda *a: False)
    monkeypatch.setattr(torch.backends.cuda, "mem_efficient_sdp_enabled", lambda: True)
    alive = []

    def sdpa(query, key, value, **kwargs):
        alive.append([r() is not None for r in repeated])
        assert kwargs["enable_gqa"] is True and key.shape[1] == 2
        return "fallback"

    gqa_sdpa = sdpa_gqa.make_gqa_sdpa(sdpa)
    query = types.SimpleNamespace(is_cuda = True, device = torch.device("cuda", 0))
    key, value = torch.zeros(1, 2, 16, 8), torch.zeros(1, 2, 16, 8)
    assert gqa_sdpa(query, key, value, None, 0.0, True, None, 4) == "fallback"
    assert len(repeated) == 2 and alive == [[False, False]]

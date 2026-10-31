# SPDX-License-Identifier: AGPL-3.0-only
import pytest
import torch
import torch.nn as nn

from unsloth_zoo.temporary_patches import moe_utils


class _Stop(Exception):
    pass


def _experts(E = 4, H = 32, I = 48):
    m = nn.Module()
    m.gate_up_proj = nn.Parameter(torch.zeros(E, 2 * I, H))
    m.down_proj = nn.Parameter(torch.zeros(E, H, I))
    m.num_experts = E
    return m


@pytest.fixture
def tuner_calls(monkeypatch):
    autotune_cache = pytest.importorskip("unsloth.kernels.moe.autotune_cache")
    pytest.importorskip("unsloth.kernels.moe.grouped_gemm.interface")
    calls = []

    def fake_autotune(*args, **kwargs):
        calls.append(kwargs["dtype"])
        return (object(), object(), object())

    def stop(*args, **kwargs):
        raise _Stop()

    monkeypatch.setattr(autotune_cache, "get_or_autotune_moe_kernels", fake_autotune)
    monkeypatch.setattr(moe_utils, "moe_routing_sort", stop)
    monkeypatch.setattr(moe_utils, "_should_use_separated_lora", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return calls


def _call(experts, dtype, H = 32):
    with pytest.raises(_Stop):
        moe_utils.forward_triton_grouped_gemm(
            experts, torch.zeros(3, H, dtype = dtype), torch.zeros(3, 1, dtype = torch.long), torch.ones(3, 1, dtype = dtype),
        )


@pytest.mark.parametrize("first, second", [(torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32)])
def test_configs_reselected_after_module_dtype_change(monkeypatch, tuner_calls, first, second):
    monkeypatch.setenv("UNSLOTH_MOE_TRITON_FP32", "1")
    experts = _experts().to(first)
    _call(experts, first)
    _call(experts, first)
    assert tuner_calls == [first, first]  # gate_up and down tuned once
    experts.to(second)
    _call(experts, second)
    assert tuner_calls == [first, first, second, second]


def test_preset_configs_without_key_are_reused(tuner_calls):
    experts = _experts()
    preset = (48, (None, None, None), (None, None, None))
    experts._unsloth_moe_configs = preset
    _call(experts, torch.float32)
    _call(experts, torch.float32)
    assert tuner_calls == []
    assert experts._unsloth_moe_configs is preset

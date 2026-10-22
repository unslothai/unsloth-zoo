# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""forward_triton_grouped_gemm hands the experts' device to the autotuner when it takes one, so a
device_map layer on another GPU model is tuned, keyed and SMEM-fitted for its own card. An older
autotuner without the parameter is called exactly as before."""

import sys
import types

import pytest
import torch

from unsloth_zoo.temporary_patches import moe_utils as M


class _Stop(Exception):
    pass


def _run(monkeypatch, tuner):
    autotune = types.ModuleType("unsloth.kernels.moe.autotune_cache")
    autotune.get_or_autotune_moe_kernels = tuner
    interface = types.ModuleType("unsloth.kernels.moe.grouped_gemm.interface")
    interface.grouped_gemm = None
    monkeypatch.setitem(sys.modules, autotune.__name__, autotune)
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    E, I, H = 4, 48, 64
    experts = torch.nn.Module()
    experts.num_experts = E
    experts.gate_up_proj = torch.nn.Parameter(torch.zeros(E, 2 * I, H))
    experts.down_proj = torch.nn.Parameter(torch.zeros(E, H, I))
    with pytest.raises(_Stop):
        M.forward_triton_grouped_gemm(
            experts, torch.zeros(3, H), torch.zeros(3, 2, dtype = torch.long), torch.zeros(3, 2)
        )


def test_a_tuner_with_a_device_parameter_gets_the_experts_device(monkeypatch):
    seen = []

    def tuner(num_experts, hidden_dim, intermediate_dim, top_k, dtype, weight_dtype = None, device = None):
        seen.append(device)
        if len(seen) == 2:
            raise _Stop
        return ("fwd", "dX", "dW")

    _run(monkeypatch, tuner)
    assert seen == [torch.device("cpu"), torch.device("cpu")]


def test_an_older_tuner_is_called_without_it(monkeypatch):
    seen = []

    def tuner(num_experts, hidden_dim, intermediate_dim, top_k, dtype, weight_dtype = None):
        seen.append(dtype)
        if len(seen) == 2:
            raise _Stop
        return ("fwd", "dX", "dW")

    _run(monkeypatch, tuner)
    assert len(seen) == 2

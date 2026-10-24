# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Experts' CUDA device reaches the autotuner when it accepts one; older autotuners and non-CUDA
operands are called as before (no device kwarg)."""

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


def _tuner_recording_device(seen):
    def tuner(num_experts, hidden_dim, intermediate_dim, top_k, dtype, weight_dtype = None, device = None):
        seen.append(device)
        if len(seen) == 2:
            raise _Stop
        return ("fwd", "dX", "dW")
    return tuner


def test_a_tuner_with_a_device_parameter_gets_the_experts_cuda_device(monkeypatch):
    # The operands report cuda:1 to the kwarg builder (no GPU needed): the forward must hand it to the tuner.
    card = torch.device("cuda", 1)
    real = M._moe_weight_dtype_kwargs
    monkeypatch.setattr(M, "_moe_weight_dtype_kwargs", lambda t, m, n, d, device = None: real(t, m, n, d, card))
    seen = []
    _run(monkeypatch, _tuner_recording_device(seen))
    assert seen == [card, card]


def test_cpu_operands_pass_no_device(monkeypatch):
    seen = []
    _run(monkeypatch, _tuner_recording_device(seen))
    assert seen == [None, None]


def test_an_older_tuner_is_called_without_it(monkeypatch):
    seen = []

    def tuner(num_experts, hidden_dim, intermediate_dim, top_k, dtype, weight_dtype = None):
        seen.append(dtype)
        if len(seen) == 2:
            raise _Stop
        return ("fwd", "dX", "dW")

    _run(monkeypatch, tuner)
    assert len(seen) == 2

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""forward_triton_grouped_gemm must autotune each GEMM at its real N: the autotuner builds a
(E, 2 * intermediate_dim, hidden_dim) weight, so a doubled argument tunes a BLOCK_SIZE_N that may
not divide the real N (CompileTimeAssertionFailure at the first real launch)."""

import sys
import types

import pytest
import torch

from unsloth_zoo.temporary_patches import moe_utils as M


class _Stop(Exception):
    pass


def test_each_gemm_is_tuned_at_its_real_n(monkeypatch):
    seen = []

    def fake_autotune(**kwargs):
        seen.append(kwargs)
        if len(seen) == 2:
            raise _Stop
        return ("fwd", "dX", "dW")

    autotune = types.ModuleType("unsloth.kernels.moe.autotune_cache")
    autotune.get_or_autotune_moe_kernels = fake_autotune
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

    gemm1, gemm2 = seen
    assert (2 * gemm1["intermediate_dim"], gemm1["hidden_dim"]) == (2 * I, H)
    assert (2 * gemm2["intermediate_dim"], gemm2["hidden_dim"]) == (H, I)

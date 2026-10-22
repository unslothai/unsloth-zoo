# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""A cold select_moe_backend() inside torch.compile must not recompile its caller on the next call."""
import pytest
import torch

import unsloth_zoo.temporary_patches.moe_utils as M


@pytest.mark.parametrize("mode", [{}, {"fullgraph": True}, {"fullgraph": True, "dynamic": True}])
def test_cold_backend_selection_compiles_once(monkeypatch, mode):
    monkeypatch.setattr(M, "_select_moe_backend", lambda: "native_torch")
    monkeypatch.setattr(M, "_unsloth_importing", lambda: False)
    M.select_moe_backend.cache_clear()
    torch._dynamo.reset()
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    def f(x):
        return x + (1 if M.select_moe_backend() == "native_torch" else 2)

    try:
        compiled = torch.compile(f, backend = backend, **mode)
        for _ in range(3):
            assert compiled(torch.ones(4)).tolist() == [2.0] * 4
        assert len(graphs) == 1
        # Eager calls still cache the choice.
        assert M.select_moe_backend() == "native_torch" and M._SELECTED_MOE_BACKEND == "native_torch"
    finally:
        M.select_moe_backend.cache_clear()
        torch._dynamo.reset()

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.

"""Unpatch must drop every staging buffer before emptying the cache: resize_(0) keeps the storage, so a buffer still
referenced (e.g. by a loop variable) stays reserved after unpatch returns."""

import sys
import types
import weakref

import torch
import transformers
from unsloth_zoo import gradient_checkpointing as G


def test_unpatch_drops_all_staging_tensors_before_empty_cache(monkeypatch):
    modeling = types.ModuleType("transformers.modeling_utils")
    modeling.checkpoint = torch.utils.checkpoint.checkpoint
    monkeypatch.setitem(sys.modules, modeling.__name__, modeling)
    monkeypatch.setattr(transformers, "modeling_utils", modeling, raising = False)
    monkeypatch.setattr(torch.utils.checkpoint, "_old_CheckpointFunction",
                        torch.utils.checkpoint.CheckpointFunction, raising = False)
    monkeypatch.setattr(torch.utils.checkpoint, "CheckpointFunction", G.UnslothCheckpointFunction)
    state = {
        "CPU_BUFFERS": [], "GPU_BUFFERS": (torch.empty(128), torch.empty(64)),
        "GPU_BUFFERS_B": None, "USE_DOUBLE_BUFFER": False,
        "BUFFER_EVENTS_A": None, "BUFFER_EVENTS_B": None, "NEXT_BUFFER_SLOT": [0, 0],
    }
    for name in state:
        monkeypatch.setattr(G, name, state[name], raising = False)
    refs = [weakref.ref(x) for x in state.pop("GPU_BUFFERS")]
    calls = []

    def empty_cache():
        calls.append([ref() is None for ref in refs])

    monkeypatch.setattr(G.torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(G, "DEVICE_TYPE", "cuda", raising = False)
    G.unpatch_unsloth_smart_gradient_checkpointing()
    assert calls == [[True, True]], f"staging buffers alive during empty_cache: {calls}"

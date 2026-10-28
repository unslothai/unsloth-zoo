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

"""Forced-float32 pass calls empty_cache once after syncing only the model's own GPUs (not device_count()): per-module
calls freed cached blocks on other GPUs mid-write (illegal access later), and syncing unused GPUs creates a CUDA
context per card (hundreds of MB per rank under DDP)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM

from unsloth_zoo.patching_utils import patch_model_and_tokenizer


def _tiny_model():
    config = LlamaConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        max_position_embeddings = 64,
    )
    return LlamaForCausalLM(config)


class _CudaCallRecorder:
    """Record every synchronise / empty_cache the pass makes, then replay it."""

    def __init__(self):
        self.synchronised = []
        self.empty_cache_calls = 0
        self._synchronize = torch.cuda.synchronize
        self._empty_cache = torch.cuda.empty_cache

    def __enter__(self):
        def synchronize(device = None):
            self.synchronised.append(device)
            return self._synchronize(device)

        def empty_cache():
            self.empty_cache_calls += 1
            return self._empty_cache()

        torch.cuda.synchronize = synchronize
        torch.cuda.empty_cache = empty_cache
        return self

    def __exit__(self, *exception):
        torch.cuda.synchronize = self._synchronize
        torch.cuda.empty_cache = self._empty_cache
        return False


def test_cpu_model_never_touches_a_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)

    model = _tiny_model()
    with _CudaCallRecorder() as recorder:
        patch_model_and_tokenizer(model, None, do_forced_float32 = True)

    assert recorder.synchronised == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_gpu_model_syncs_its_own_devices_once():
    model = _tiny_model().to("cuda:0")
    n_modules = len(list(model.named_modules()))
    expected = {p.device for p in model.parameters()}
    expected |= {b.device for b in model.buffers()}

    with _CudaCallRecorder() as recorder:
        patch_model_and_tokenizer(model, None, do_forced_float32 = True)

    assert set(recorder.synchronised) == expected
    assert len(recorder.synchronised) == len(expected)
    # One empty_cache for the whole pass; the trailing gc loop in patch_model_and_tokenizer adds a fixed three on top,
    # so the total (4) is bounded by a constant, not model-sized.
    assert recorder.empty_cache_calls < n_modules
    assert recorder.empty_cache_calls == 4

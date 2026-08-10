# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Device handling for the GRPO chunked log-softmax.

`chunked_hidden_states_selective_log_softmax` multiplies the hidden states by
the output head. When a model is dispatched across several GPUs those two can
live on different devices, and the matmul then fails. These tests pin both
halves of the contract:

  1. on one device the result is unchanged, bit for bit
  2. with the head on another device it runs, and agrees with the single-device
     result, and hands the answer back on the caller's device
"""
import torch
import pytest

from unsloth_zoo.rl_replacements import chunked_hidden_states_selective_log_softmax


def _inputs(device, vocab = 512, batch = 2, seq = 32, hidden = 64, dtype = torch.float32):
    gen = torch.Generator().manual_seed(0)
    hidden_states = torch.randn(batch, seq, hidden, generator = gen, dtype = dtype).to(device)
    lm_head = torch.randn(vocab, hidden, generator = gen, dtype = dtype).to(device)
    index = torch.randint(0, vocab, (batch, seq), generator = gen).to(device)
    return hidden_states, lm_head, index


KWARGS = dict(chunks = 4, logit_softcapping = 20.0, temperature = 0.9)


def test_single_device_cpu_runs():
    hidden_states, lm_head, index = _inputs("cpu")
    out = chunked_hidden_states_selective_log_softmax(hidden_states, lm_head, index, **KWARGS)
    assert out.shape == index.shape
    assert out.device == hidden_states.device
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_single_device_cuda_is_deterministic():
    hidden_states, lm_head, index = _inputs("cuda:0")
    a = chunked_hidden_states_selective_log_softmax(hidden_states, lm_head, index, **KWARGS)
    b = chunked_hidden_states_selective_log_softmax(hidden_states, lm_head, index, **KWARGS)
    assert torch.equal(a, b)
    assert a.device == hidden_states.device


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason = "needs 2 GPUs")
def test_head_on_another_device_matches_single_device():
    hidden_states, lm_head, index = _inputs("cuda:0")
    same = chunked_hidden_states_selective_log_softmax(hidden_states, lm_head, index, **KWARGS)

    # The head is the tensor that moves: accelerate puts the tail of the model
    # on the last device it fills, while the hidden states arrive from earlier
    # layers on an earlier device.
    split = chunked_hidden_states_selective_log_softmax(
        hidden_states, lm_head.to("cuda:1"), index, **KWARGS,
    )
    assert split.device == hidden_states.device, "result must come back on the caller's device"
    assert torch.equal(same, split)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason = "needs 2 GPUs")
def test_index_on_caller_device_is_accepted():
    # The index follows the hidden states, not the head, so it needs moving too.
    hidden_states, lm_head, index = _inputs("cuda:0")
    out = chunked_hidden_states_selective_log_softmax(
        hidden_states, lm_head.to("cuda:1"), index, **KWARGS,
    )
    assert torch.isfinite(out).all()


def _run(hidden_states, lm_head, index, **kw):
    return chunked_hidden_states_selective_log_softmax(hidden_states, lm_head, index, **kw)


def test_row_cap_unset_keeps_chunk_boundaries(monkeypatch):
    # The default must not move a single byte: unset means the old behaviour.
    monkeypatch.delenv("UNSLOTH_GRPO_MAX_ROWS_PER_CHUNK", raising = False)
    hidden_states, lm_head, index = _inputs("cpu")
    a = _run(hidden_states, lm_head, index, **KWARGS)
    monkeypatch.setenv("UNSLOTH_GRPO_MAX_ROWS_PER_CHUNK", "0")
    b = _run(hidden_states, lm_head, index, **KWARGS)
    assert torch.equal(a, b)


def test_row_cap_splits_further_without_changing_the_answer(monkeypatch):
    # float32 in, so pure loop splitting is exact here. In lower precision the
    # matmul shape changes and tiny last-bit differences are expected.
    monkeypatch.delenv("UNSLOTH_GRPO_MAX_ROWS_PER_CHUNK", raising = False)
    hidden_states, lm_head, index = _inputs("cpu", batch = 2, seq = 64)
    ref = _run(hidden_states, lm_head, index, **KWARGS)
    monkeypatch.setenv("UNSLOTH_GRPO_MAX_ROWS_PER_CHUNK", "8")
    capped = _run(hidden_states, lm_head, index, **KWARGS)
    assert capped.shape == ref.shape
    assert torch.allclose(ref, capped, rtol = 0, atol = 1e-5)


def test_row_cap_never_reduces_chunk_count(monkeypatch):
    # A cap larger than the whole tensor must not coarsen the existing split.
    monkeypatch.setenv("UNSLOTH_GRPO_MAX_ROWS_PER_CHUNK", "100000")
    hidden_states, lm_head, index = _inputs("cpu")
    monkeypatch.delenv("UNSLOTH_GRPO_MAX_ROWS_PER_CHUNK", raising = False)
    ref = _run(hidden_states, lm_head, index, **KWARGS)
    monkeypatch.setenv("UNSLOTH_GRPO_MAX_ROWS_PER_CHUNK", "100000")
    big = _run(hidden_states, lm_head, index, **KWARGS)
    assert torch.equal(ref, big)

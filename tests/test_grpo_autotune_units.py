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

"""GRPO mini-batch autotune: rows per forward in real bytes, and callers that never exceed the fit."""

import inspect
import math
import textwrap
from types import SimpleNamespace

import pytest
import torch

from unsloth_zoo import rl_replacements as rr

GiB = 1024**3
ROWS, SEQ, HIDDEN, VOCAB = 8, 1024, 1024, 151936


def _estimate(rows, dtype_bytes, total = ROWS, seq = SEQ, multiplier = 4):
    """The autotune's estimate for `rows` rows: one hidden state plus the chunked logits."""
    hidden = rows * seq * HIDDEN * dtype_bytes
    logits = (rows / total) * rows * seq * VOCAB * dtype_bytes / multiplier
    return hidden + logits


def _gpu(monkeypatch, free, total = 80 * GiB, reserved = 0, allocated = 0, fraction = 1.0):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (int(free), int(total)))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: int(reserved))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: int(allocated))
    monkeypatch.setattr(
        torch.cuda, "get_per_process_memory_fraction", lambda *a, **k: fraction, raising = False
    )


def _free_for(rows, dtype_bytes = 2):
    """Free memory that fits exactly `rows` rows (the autotune budgets 80% of it)."""
    return (_estimate(rows, dtype_bytes) + _estimate(rows + 1, dtype_bytes)) / 2 / 0.8


def test_estimate_is_in_bytes_and_fits_five_of_eight(monkeypatch):
    _gpu(monkeypatch, _free_for(5))
    assert rr.autotune_batch_and_chunks(ROWS, SEQ, HIDDEN, VOCAB, 2, None) == (5, 4)


@pytest.mark.parametrize("bits, nbytes", [(16, 2), (32, 4)])
def test_bits_from_an_older_caller_are_read_as_bytes(monkeypatch, bits, nbytes):
    _gpu(monkeypatch, _free_for(5, nbytes))
    assert rr.autotune_batch_and_chunks(ROWS, SEQ, HIDDEN, VOCAB, bits, None) == (5, 4)


@pytest.mark.parametrize("rows", [1, 2, 5, 8, 17])
def test_nothing_fits_gives_one_row_per_forward(monkeypatch, rows):
    _gpu(monkeypatch, 1)
    assert rr.autotune_batch_and_chunks(rows, SEQ, HIDDEN, VOCAB, 2, None) == (1, 4)


def test_cached_allocator_blocks_count_as_free(monkeypatch):
    # 1 row of free device memory, plus 4 rows' worth the caching allocator holds unused.
    _gpu(monkeypatch, _free_for(1), reserved = _free_for(5) - _free_for(1) + GiB, allocated = GiB)
    assert rr.autotune_batch_and_chunks(ROWS, SEQ, HIDDEN, VOCAB, 2, None)[0] == 5


def test_per_process_fraction_caps_the_budget(monkeypatch):
    # Plenty free on the device, but set_per_process_memory_fraction leaves this process 5 rows.
    total = 80 * GiB
    _gpu(monkeypatch, 70 * GiB, total = total, allocated = GiB,
         fraction = (_free_for(5) + GiB) / total)
    assert rr.autotune_batch_and_chunks(ROWS, SEQ, HIDDEN, VOCAB, 2, None)[0] == 5


class _Stop(Exception):
    pass


class _Model(torch.nn.Module):
    def __init__(self, hidden, vocab, device = None):
        super().__init__()
        self.head = torch.nn.Linear(hidden, vocab, bias = False, device = device)

    def get_output_embeddings(self):
        return self.head


def _plan(monkeypatch, free, rows = ROWS, mini_batch = None):
    """(dtype_bytes, rows per forward) grpo_accumulated_loss picks; stops at its chunker."""
    _gpu(monkeypatch, free)
    seen = {}
    real = rr.autotune_batch_and_chunks

    def recording(total_rows, seq_len, hidden, vocab, dtype_bytes, multiplier):
        seen["dtype_bytes"] = dtype_bytes
        return real(total_rows, seq_len, hidden, vocab, dtype_bytes, multiplier)

    def chunker(vision_inputs, total_samples, batch_size):
        seen["batch_size"] = batch_size
        raise _Stop()

    monkeypatch.delenv("UNSLOTH_RETURN_HIDDEN_STATES", raising = False)
    monkeypatch.setattr(rr, "autotune_batch_and_chunks", recording)
    monkeypatch.setattr(rr, "grpo_vision_chunks", chunker)
    model = _Model(HIDDEN, VOCAB, device = "meta")
    trainer = SimpleNamespace(
        model = model,
        processing_class = SimpleNamespace(pad_token_id = 0),
        accelerator = SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m),
        use_vllm = False,
        _autocast_dtype = torch.bfloat16,
        _autocast_enabled = True,
        args = SimpleNamespace(unsloth_grpo_mini_batch = mini_batch, unsloth_logit_chunk_multiplier = None),
    )
    ids = torch.ones(rows, SEQ, dtype = torch.long)
    with pytest.raises(_Stop):
        rr.grpo_accumulated_loss(
            trainer, ids, torch.ones_like(ids), SEQ // 2, torch.ones(rows, SEQ // 2, dtype = torch.long),
            torch.ones(rows), None, None,
        )
    return seen


def test_loss_sizes_bfloat16_in_bytes_and_honours_the_fit(monkeypatch):
    seen = _plan(monkeypatch, _free_for(5))
    assert seen["dtype_bytes"] == 2
    # 5 rows fit, so 8 rows go as 2 forwards of 4, never 1 forward of 8.
    assert seen["batch_size"] == 4


def test_loss_plan_is_unchanged_when_every_row_fits(monkeypatch):
    assert _plan(monkeypatch, 70 * GiB)["batch_size"] == ROWS


@pytest.mark.parametrize("total_rows, rows_fit", [(8, 5), (16, 9), (16, 15), (7, 4), (3, 2), (8, 8), (8, 4)])
def test_split_count_never_exceeds_the_fit(monkeypatch, total_rows, rows_fit):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    seen = {}

    def chunker(vision_inputs, total_samples, batch_size):
        seen["batch_size"] = batch_size
        raise _Stop()

    monkeypatch.setattr(rr, "autotune_batch_and_chunks", lambda *a: (rows_fit, 4))
    monkeypatch.setattr(rr, "grpo_vision_chunks", chunker)
    model = _Model(4, 16)
    trainer = SimpleNamespace(
        model = model,
        processing_class = SimpleNamespace(pad_token_id = 0),
        accelerator = SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m),
        use_vllm = False, _autocast_dtype = None,
        args = SimpleNamespace(unsloth_grpo_mini_batch = None, unsloth_logit_chunk_multiplier = None),
    )
    ids = torch.ones(total_rows, 8, dtype = torch.long)
    with pytest.raises(_Stop):
        rr.grpo_accumulated_loss(
            trainer, ids, torch.ones_like(ids), 4, torch.ones(total_rows, 4, dtype = torch.long),
            torch.ones(total_rows), None, None,
        )
    assert seen["batch_size"] <= rows_fit
    assert seen["batch_size"] == math.ceil(total_rows / math.ceil(total_rows / rows_fit))


def test_offload_gate_sees_the_chunk_width_not_the_split_count():
    """A 12-row chunk at one split must reach efficient_log_softmax as 12 rows wide."""
    src = inspect.getsource(rr.grpo_accumulated_loss)
    start = src.rindex("\n", 0, src.index("def compute_logprobs_chunk(")) + 1
    end = src.index("\n        for (", start)
    seen = {}

    def efficient_log_softmax(hidden, lm_head, index, chunks = 32, batch_size = 8, **kwargs):
        seen["batch_size"] = batch_size
        return index.new_zeros(index.shape, dtype = torch.float32)

    ns = {
        "torch": torch, "B": 1, "multiplier": 1, "lm_head": torch.zeros(16, 4),
        "_refresh_lm_head": lambda: None, "_fp32_bind": lambda *a: None,
        "_need_entropy": False, "_fused_route": False, "efficient_log_softmax": efficient_log_softmax,
        "logit_scale_multiply": 0.0, "logit_scale_divide": 0.0, "logit_softcapping": 0.0,
        "temperature": 1.0, "lm_head_bias": None,
    }
    exec(textwrap.dedent(src[start:end]), ns)
    ns["compute_logprobs_chunk"](
        torch.zeros(12, 5, 4), torch.zeros(12, 5, dtype = torch.long), torch.zeros(12, 9, dtype = torch.long),
    )
    assert seen["batch_size"] == 12


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

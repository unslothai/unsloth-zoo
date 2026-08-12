# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Equal-chunk padding in ``chunked_hidden_states_selective_log_softmax``.

The GRPO log-softmax chunks its rows with ``torch.chunk(x, chunks = N)``, which
leaves a ragged last chunk of ``n - (N - 1)*ceil(n/N)`` rows. Inside one dynamic
graph that tail is a symbolic expression in ``n``, and torch 2.12's Inductor
cannot prove a vocab-wide node over it splits evenly, so the GRPO step dies with

    InductorError: CantSplit: 202048*s47*s87 - 3434816*(((s47*s87 + 17)//18))
    not divisible by s47*s87 - 17*(((s47*s87 + 17)//18))

Padding the row count up to a multiple of ``chunks`` removes the ragged tail.
These tests pin the two things that has to be true: the padding never changes a
returned value, and the chunk sizes really are all equal.

The function is lifted out with ``ast`` rather than imported, so the tests stay
CPU-only and do not need the rest of unsloth_zoo. Decorators are stripped: the
change is shape bookkeeping, and eager exercises it exactly.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest
import torch


SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth_zoo" / "rl_replacements.py"
FUNCTION_NAME = "chunked_hidden_states_selective_log_softmax"


def _load_chunked_log_softmax():
    tree = ast.parse(SOURCE_PATH.read_text(encoding = "utf-8"), filename = str(SOURCE_PATH))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == FUNCTION_NAME:
            node.decorator_list = []
            namespace: dict = {"torch": torch, "os": os}
            exec(
                compile(ast.Module(body = [node], type_ignores = []), str(SOURCE_PATH), "exec"),
                namespace,
            )
            return namespace[FUNCTION_NAME]
    raise AssertionError(f"{FUNCTION_NAME} not found in {SOURCE_PATH}")


chunked_log_softmax = _load_chunked_log_softmax()


def unpadded_reference(
    hidden_states, lm_head, index, chunks,
    logit_scale_multiply = 0.0, logit_scale_divide = 0.0,
    logit_softcapping = 0.0, temperature = 1.0,
):
    """The pre-fix body: torch.chunk straight through, ragged tail and all."""
    flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_index = index.reshape(-1)
    out = []
    for chunk_hidden_states, chunk_index in zip(
        torch.chunk(flat_hidden_states, chunks = chunks, dim = 0),
        torch.chunk(flat_index, chunks = chunks, dim = 0),
    ):
        chunk_hidden_states = chunk_hidden_states.to(device = lm_head.device, dtype = lm_head.dtype)
        chunk_index = chunk_index.to(lm_head.device)
        chunk_logits = chunk_hidden_states @ lm_head.t()
        if logit_scale_multiply != 0.0:
            chunk_logits = chunk_logits * logit_scale_multiply
        if logit_scale_divide != 0.0:
            chunk_logits = chunk_logits / logit_scale_divide
        if logit_softcapping != 0.0:
            chunk_logits = logit_softcapping * torch.tanh(chunk_logits / logit_softcapping)
        chunk_logits = chunk_logits.to(torch.float32)
        if temperature != 1.0:
            chunk_logits = chunk_logits / temperature
        selected = torch.gather(chunk_logits, dim = -1, index = chunk_index.unsqueeze(-1)).squeeze(-1)
        out.append((selected - torch.logsumexp(chunk_logits, dim = -1)).to(hidden_states.device))
    return torch.concat(out).reshape((hidden_states.shape[0], hidden_states.shape[1]))


# (batch, seq, chunks). The Muse Glimmer shape that raised CantSplit is a
# ragged split at 18 chunks, so ragged cases lead.
SHAPES = [
    (4, 137, 18),   # ragged, the reported shape
    (5, 100,  7),   # ragged, odd chunk count
    (1,  17, 18),   # one row short of the chunk count
    (1,  19, 18),   # torch.chunk returns fewer chunks than asked
    (1,   5, 18),   # rows well below the chunk count
    (1,   1, 18),   # single row
    (1,   1,  1),   # single row, single chunk
    (2,  50,  1),   # one chunk, nothing to pad
    (6,  36, 18),   # already an exact multiple
    (4,   9,  4),   # exact multiple, small
]

SCALINGS = [
    {},
    {"logit_softcapping": 30.0, "temperature": 0.7},
    {"logit_scale_multiply": 2.0},
    {"logit_scale_divide": 4.0},
]


@pytest.mark.parametrize("batch, seq, chunks", SHAPES)
@pytest.mark.parametrize("scaling", SCALINGS)
def test_padding_never_changes_a_returned_value(batch, seq, chunks, scaling):
    torch.manual_seed(0)
    vocab, hidden = 512, 32
    lm_head = torch.randn(vocab, hidden)
    hidden_states = torch.randn(batch, seq, hidden)
    index = torch.randint(0, vocab, (batch, seq))

    got = chunked_log_softmax(hidden_states, lm_head, index, chunks, **scaling)
    expected = unpadded_reference(hidden_states, lm_head, index, chunks, **scaling)

    assert got.shape == (batch, seq)
    # Bit-identical, not close: padding repeats a row and slices it back off,
    # so it must not perturb any surviving value at all.
    assert torch.equal(got, expected)


@pytest.mark.parametrize("batch, seq, chunks", SHAPES)
def test_every_chunk_is_the_same_size(batch, seq, chunks):
    """The property the Inductor fix rests on, checked on the padded count."""
    n_rows = batch * seq
    rows_per_chunk = (n_rows + chunks - 1) // chunks
    padded = rows_per_chunk * chunks

    sizes = [c.shape[0] for c in torch.chunk(torch.empty(padded, 4), chunks = chunks, dim = 0)]

    assert len(set(sizes)) == 1, f"ragged split survived: {sizes}"
    assert len(sizes) == chunks
    assert sum(sizes) == padded
    assert padded - n_rows < chunks


def test_the_unpadded_reference_really_is_ragged():
    """Guard against the suite passing because the reference was fixed too."""
    sizes = [c.shape[0] for c in torch.chunk(torch.empty(548, 4), chunks = 18, dim = 0)]
    assert len(set(sizes)) > 1, "expected torch.chunk to leave a ragged tail at 548/18"
    assert sizes[-1] != sizes[0]

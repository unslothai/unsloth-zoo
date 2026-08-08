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

"""The PrefixGrouper forward needs the same width dispatch as the packed path.

`.logits` carries hidden states only when the forward is the Unsloth generated
one honouring UNSLOTH_RETURN_HIDDEN_STATES. The packed call site and its
verifier both dispatch on width; `_pg_grad_forward` handed `.logits` straight to
`extract_logps`, which always feeds its helper the lm_head matmul signature. A
forward returning real [T, vocab] logits therefore raised the reduction-dim
mismatch inside PrefixGrouper, the outer handler untrusted the signature, and
the whole verify forward was repeated and thrown away on every later step.

The block is exec'd from its own source so the dispatch is exercised, not read.

The second half pins the property the reshape comment in
chunked_hidden_states_selective_log_softmax protects: a wrong-width caller must
still fail with a message naming BOTH operands, never be silently reshaped.
"""

import contextlib
import inspect
import os
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

import unsloth_zoo.rl_replacements as rr


VOCAB, HIDDEN = 23, 6
L = 4                      # padded row length
W = 4                      # scatter width
ROWS = [[3, 5, 7, 9],      # row 0
        [3, 5, 11, 13]]    # row 1, shares the [3, 5] prefix
# PrefixGrouper's flat stream: shared prefix once, then each row's suffix.
FLAT = [3, 5, 7, 9, 11, 13]
POS  = [0, 1, 2, 3, 2, 3]
# One entry per predicted completion token, aligned 1:1.
TGT_ROWS = [0, 0, 0, 1, 1, 1]
TGT_COLS = [1, 2, 3, 1, 2, 3]
TGT_PRED = [0, 1, 2, 0, 1, 4]   # flat index of the predicting token
TGT_FLAT = [1, 2, 3, 1, 4, 5]   # flat index of the target token itself


class _Model(torch.nn.Module):
    """`hidden_states = False` ignores UNSLOTH_RETURN_HIDDEN_STATES and returns
    real [.., vocab] logits; True is the Unsloth generated forward.

    Position-local, so the grouped forward and a per-row forward agree exactly.
    """
    def __init__(self, hidden_states = False, vocab = VOCAB, hidden = HIDDEN):
        super().__init__()
        torch.manual_seed(0)
        self.emb  = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab, bias = False)
        self.hidden_states = hidden_states

    def forward(self, input_ids = None, position_ids = None,
                prefix_seg_info = None, use_cache = None, **kwargs):
        h = torch.tanh(self.emb(input_ids))
        return SimpleNamespace(logits = h if self.hidden_states else self.head(h))


def _pg_forward_source():
    src = inspect.getsource(rr.grpo_accumulated_loss)
    start = src.index("    def _pg_grad_forward():")
    end   = src.index("    if _pg_layout is not None:")
    return textwrap.dedent(src[start:end])


def _layout():
    """A real GroupLayout, so extract_logps runs its own code."""
    pg = pytest.importorskip("unsloth.utils.prefix_grouper")
    t = lambda xs: torch.tensor(xs, dtype = torch.long)
    return pg.GroupLayout(
        flat_ids        = t(FLAT).unsqueeze(0),
        position_ids    = t(POS).unsqueeze(0),
        prefix_seg_info = None,
        tgt_rows        = t(TGT_ROWS),
        tgt_cols        = t(TGT_COLS),
        tgt_pred        = t(TGT_PRED),
        tgt_flat        = t(TGT_FLAT),
        total_rows      = len(ROWS),
        L               = L,
        W               = W,
        tok_r           = 1.0,
        signature       = ("test",),
    )


def _run_pg_forward(hidden_states):
    model = _Model(hidden_states = hidden_states)
    ns = {
        "os": rr.os, "torch": torch,
        "chunked_hidden_states_selective_log_softmax":
            rr.chunked_hidden_states_selective_log_softmax,
        "chunked_selective_log_softmax": rr.chunked_selective_log_softmax,
        "device_synchronize": lambda *a, **k: None,
        "unwrapped_model": model,
        "lm_head": model.head.weight,          # [vocab, hidden]
        "_pg_layout": _layout(),
        "total_rows": len(ROWS),
        "multiplier": 1,
        "autocaster": contextlib.nullcontext(),
        "logit_scale_multiply": 0.0, "logit_scale_divide": 0.0,
        "logit_softcapping": 0.0, "temperature": 1.0,
    }
    exec(_pg_forward_source(), ns)
    return ns["_pg_grad_forward"](), model


def _reference():
    """Per-row logprobs straight from the model, no grouping involved."""
    model = _Model(hidden_states = False)
    out = torch.zeros(len(ROWS), L)
    for r, row in enumerate(ROWS):
        ids = torch.tensor(row).unsqueeze(0)
        with torch.no_grad():
            logps = torch.log_softmax(model(input_ids = ids).logits.float(), dim = -1)[0]
        for j in range(1, len(row)):
            out[r, j] = logps[j - 1, row[j]]
    return out[:, -W:]


def test_prefix_grouper_forward_survives_a_forward_that_returns_real_logits():
    got, _ = _run_pg_forward(hidden_states = False)
    assert got.shape == (len(ROWS), W)
    assert torch.allclose(got.float(), _reference(), atol = 1e-5), (got, _reference())


def test_prefix_grouper_forward_still_matches_on_the_hidden_states_path():
    got, _ = _run_pg_forward(hidden_states = True)
    assert torch.allclose(got.float(), _reference(), atol = 1e-5), (got, _reference())


def test_pg_forward_source_dispatches_on_width():
    """Both helpers must be reachable from the block: a source that mentions only
    the matmul helper cannot have a raw-logits branch at all."""
    src = _pg_forward_source()
    assert "chunked_selective_log_softmax(" in src
    assert "lm_head.shape[1]" in src


# --- the property the reshape comment protects -------------------------------

@pytest.mark.parametrize("vocab,hidden", [
    (23, 6),    # vocab not divisible by hidden
    (24, 6),    # divisible: reshaping on lm_head's width would silently succeed
])
def test_wrong_width_caller_fails_naming_both_operands(vocab, hidden):
    lm_head = torch.nn.Linear(hidden, vocab, bias = False).weight   # [vocab, hidden]
    wrong   = torch.randn(1, 5, vocab)                              # raw logits, not hidden
    index   = torch.randint(0, vocab, (1, 5))
    with pytest.raises(Exception) as excinfo:
        rr.chunked_hidden_states_selective_log_softmax(wrong, lm_head, index, 1)
    msg = str(excinfo.value)
    assert str(vocab) in msg and str(hidden) in msg, (
        f"the mismatch must name both operands, got: {msg!r}"
    )
    assert "Expected cond to be True" not in msg


def test_correct_width_caller_is_unaffected():
    lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias = False).weight
    hidden  = torch.randn(2, 5, HIDDEN)
    index   = torch.randint(0, VOCAB, (2, 5))
    got = rr.chunked_hidden_states_selective_log_softmax(hidden, lm_head, index, 1)
    ref = torch.gather(
        torch.log_softmax((hidden @ lm_head.t()).float(), dim = -1),
        -1, index.unsqueeze(-1),
    ).squeeze(-1)
    assert torch.allclose(got.float(), ref, atol = 1e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

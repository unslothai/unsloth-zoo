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

"""The GRPO chunked log-softmax and its reduction dim.

The reduction dim is not the bug. In

    a and b must have same reduction dim, but got
    [((s47*s87 + 255)//256), s33] X [1536, 151936]

`s33` is a backed symbol and `Eq(s33, 1536)` is False -- the tensor really was
the wrong width. Switching the reduction dim to `lm_head.shape[-1]` plus a bare
`torch._check` left the guard sets identical (verified with TORCH_LOGS=guards)
and only made the message worse. The real cause and fix live in
tests/test_grpo_packed_raw_logits_dispatch.py.

What is left here is the numeric contract, kept as a regression guard.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = (ROOT / "unsloth_zoo" / "rl_replacements.py").read_text(encoding="utf-8")


# ---- the source ------------------------------------------------------------

def _fn_source():
    import ast
    tree = ast.parse(SRC)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and \
                node.name == "chunked_hidden_states_selective_log_softmax":
            return ast.get_source_segment(SRC, node)
    raise AssertionError("function not found")


def test_the_reshape_uses_its_own_last_dim():
    """A no-op reshape. Against `lm_head.shape[-1]` a mismatched caller whose
    element count divides gets its row count silently rewritten."""
    body = _fn_source()
    assert "hidden_states.reshape(-1, hidden_states.shape[-1])" in body


def test_no_message_less_check_is_reintroduced():
    """`torch._check(cond)` names neither operand and Dynamo rejects a callable
    message, so the matmul must be left to raise -- it prints both."""
    import ast
    calls = [
        n for n in ast.walk(ast.parse(_fn_source()))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_check"
    ]
    assert calls == [], "torch._check without a message is back"


def test_the_sibling_helper_is_left_alone():
    """The sibling reshapes against its own last dim too, but has no matmul."""
    assert "logits.reshape(-1, logits.shape[-1])" in SRC


def test_the_function_is_still_compiled_by_default():
    """The point is to make the compiled path work, not to opt out of it."""
    body = _fn_source()
    i = SRC.index(body)
    preceding = SRC[max(0, i - 300):i]
    assert "_maybe_compile(dynamic = True, fullgraph = True" in preceding


# ---- the numbers -----------------------------------------------------------

def _setup():
    torch = pytest.importorskip("torch")
    from unsloth_zoo.rl_replacements import (
        chunked_hidden_states_selective_log_softmax as fn,
    )
    if torch.cuda.is_available():
        return torch, fn, "cuda", torch.bfloat16, 128, 256, 2e-2
    # CPU keeps the shapes small: compiling 1536x151936 there is a wait, not a test.
    return torch, fn, "cpu", torch.float32, 32, 64, 1e-4


def _reference(torch, hidden_states, lm_head, index, temperature = 1.0):
    import torch.nn.functional as F
    logits = (hidden_states.to(lm_head.dtype) @ lm_head.t()).float() / temperature
    return torch.gather(
        F.log_softmax(logits, dim = -1), -1, index.unsqueeze(-1),
    ).squeeze(-1)


@pytest.mark.parametrize("batch,seq,chunks,temperature", [
    (2, 16, 4, 1.0),
    (4, 24, 256, 1.0),   # chunks > rows, where the ceil-division bites
    (3, 40, 8, 0.7),
    (1, 7, 3, 1.0),      # rows not divisible by chunks
])
def test_it_still_matches_a_plain_log_softmax(batch, seq, chunks, temperature):
    torch, fn, dev, dtype, hidden, vocab, tol = _setup()
    torch.manual_seed(0)
    hidden_states = torch.randn(batch, seq, hidden, device = dev, dtype = dtype)
    lm_head = torch.randn(vocab, hidden, device = dev, dtype = dtype) / 30
    index = torch.randint(0, vocab, (batch, seq), device = dev)

    # The compiled trainer marks the hidden dim dynamic; a bare call does not.
    torch._dynamo.mark_dynamic(hidden_states, 2)

    got = fn(hidden_states, lm_head, index,
             chunks = chunks, temperature = temperature)
    expected = _reference(torch, hidden_states, lm_head, index,
                          temperature = temperature)
    assert tuple(got.shape) == (batch, seq)
    assert (got - expected).abs().max().item() < tol


def test_a_mismatched_lm_head_fails_loudly():
    """Wrong-width hidden states can reshape cleanly when the element count
    divides (2x16x768 against a 1536-wide lm_head gives 16 rows, not 32), so the
    wrongness must surface here rather than later."""
    torch, fn, dev, dtype, hidden, vocab, _ = _setup()
    hidden_states = torch.randn(2, 16, hidden // 2, device = dev, dtype = dtype)
    lm_head = torch.randn(vocab, hidden, device = dev, dtype = dtype)
    index = torch.randint(0, vocab, (2, 16), device = dev)
    with pytest.raises(Exception):
        fn(hidden_states, lm_head, index, chunks = 4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

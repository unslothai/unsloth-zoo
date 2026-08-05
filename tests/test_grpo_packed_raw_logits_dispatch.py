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

"""The GRPO packed path fed raw logits into the lm_head matmul.

    a and b must have same reduction dim, but got
    [((s47*s87 + 255)//256), s33] X [1536, 151936]

is not a dynamic-shape guard failure. `Eq(s33, 1536)` evaluates to False: the
tensor really was the wrong width. `lm_head` is
`trainer.model.get_output_embeddings().weight`, an nn.Parameter of
[vocab, hidden] = [151936, 1536], which is why `b` prints concrete. The first
operand's width therefore had a hint that was not 1536 -- vocab-wide raw
logits, matching the 151936 in the same message.

`grpo_accumulated_loss` sets UNSLOTH_RETURN_HIDDEN_STATES=1, but `.logits`
only carries hidden states when the model's forward is the Unsloth generated
one that reads that variable. The padded path already copes: see the
`new_hidden_states_chunk.shape[-1] == lm_head.shape[1]` dispatch in
`compute_logprobs_chunk`, which falls back to `chunked_selective_log_softmax`.
The sequence-packing path -- default-on, `UNSLOTH_GRPO_SEQ_PACKING=1` -- had no
such dispatch and handed `.logits` straight to the matmul.

So this file covers the two halves of the gap:

  * the packed call site dispatches on width, like the padded one;
  * when the widths genuinely disagree, the error still names both operands,
    rather than the bare `Expected cond to be True, but got False` that a
    message-less `torch._check` produces. A `torch._check` that *would* name
    them is not available: Dynamo rejects a callable message with "Failed to
    convert args/kwargs to proxy", so the matmul's own error is the best one
    there is.
"""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = (ROOT / "unsloth_zoo" / "rl_replacements.py").read_text(encoding = "utf-8")
# One parse, shared: nodes from two separate parses never compare equal, which
# would make every containment test below silently vacuous.
TREE = ast.parse(SRC)


# ---- the message ----------------------------------------------------------

def _mismatched_call(fn, torch):
    """Vocab-wide logits where hidden states are expected: what an unpatched
    forward returns.

    `lm_head` is an nn.Parameter because that is what
    `get_output_embeddings().weight` is, and it matters here: parameters are
    shape-static under `force_parameter_static_shapes`, which is why the real
    report printed a concrete `[1536, 151936]` next to a symbolic `s33`.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hidden, vocab = 32, 128
    logits  = torch.randn(2, 8, vocab, device = dev, dtype = torch.float32)
    lm_head = torch.nn.Parameter(
        torch.randn(vocab, hidden, device = dev, dtype = torch.float32)
    )
    index   = torch.randint(0, vocab, (2, 8), device = dev)
    return lambda: fn(logits, lm_head, index, chunks = 4)


def test_the_width_mismatch_names_both_operands():
    """A bare `torch._check` reports only "Expected cond to be True, but got
    False", which names neither operand -- strictly less than the matmul error
    it would replace. Since a message-carrying check is not available under
    Dynamo, the matmul must be left to raise."""
    torch = pytest.importorskip("torch")
    from unsloth_zoo.rl_replacements import (
        chunked_hidden_states_selective_log_softmax as fn,
    )
    with pytest.raises(RuntimeError) as excinfo:
        _mismatched_call(fn, torch)()
    message = str(excinfo.value)
    assert "Expected cond to be True" not in message, message
    # Both widths, so the reader can see which operand is wrong.
    assert "32" in message and "128" in message, message


# ---- the dispatch ---------------------------------------------------------

def _packed_block():
    """The `if _pack_T >= 2 ...` body inside `grpo_accumulated_loss`.

    Located by AST rather than by string search so reformatting the file
    cannot make this test silently vacuous.
    """
    for node in ast.walk(TREE):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and \
                func.id == "chunked_hidden_states_selective_log_softmax":
            args = [a for a in node.args if isinstance(a, ast.Name)]
            if any(a.id == "_pack_tid" or a.id.startswith("_pack") for a in args):
                return node
    return None


def _enclosing_width_test(call_node):
    """The nearest `if` whose test compares a `.shape[-1]` against
    `lm_head.shape[...]` and which contains `call_node`."""
    for node in ast.walk(TREE):
        if not isinstance(node, ast.If):
            continue
        if call_node not in list(ast.walk(node)):
            continue
        test = ast.dump(node.test)
        if "lm_head" in test and "shape" in test and "Compare" in test:
            return node
    return None


def test_the_packed_call_site_dispatches_on_width():
    call = _packed_block()
    assert call is not None, "packed call to the hidden-states helper not found"
    guard = _enclosing_width_test(call)
    assert guard is not None, (
        "the packed path calls chunked_hidden_states_selective_log_softmax "
        "without first comparing the tensor width against lm_head, so an "
        "unpatched forward returning real logits reaches the lm_head matmul"
    )


def test_the_packed_fallback_is_the_raw_logits_helper():
    """Raw logits must not go through the matmul path at all; scale and
    softcapping were already applied by the forward, exactly as
    `compute_logprobs_chunk` documents."""
    call = _packed_block()
    guard = _enclosing_width_test(call)
    assert guard is not None
    names = {n.func.id for n in ast.walk(ast.Module(body = guard.orelse, type_ignores = []))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "chunked_selective_log_softmax" in names, sorted(names)


def test_the_padded_path_still_dispatches():
    """The behaviour being mirrored. If this ever goes away, the packed
    dispatch above is no longer "the same as the padded one"."""
    assert "new_hidden_states_chunk.shape[-1] == lm_head.shape[1]" in SRC


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

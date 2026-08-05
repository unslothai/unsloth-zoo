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

`grpo_accumulated_loss` sets UNSLOTH_RETURN_HIDDEN_STATES=1, but `.logits`
carries hidden states only when the forward is the Unsloth generated one. The
padded path copes via the `new_hidden_states_chunk.shape[-1] == lm_head.shape[1]`
dispatch in `compute_logprobs_chunk`; the packed path -- default-on,
`UNSLOTH_GRPO_SEQ_PACKING=1` -- had none, so vocab-wide logits hit the matmul:

    a and b must have same reduction dim, but got
    [((s47*s87 + 255)//256), s33] X [1536, 151936]

That is not a guard failure -- `Eq(s33, 1536)` is simply False. So this file
covers both halves: the packed call site dispatches on width, and a genuine
mismatch still names both operands, unlike the message-less `torch._check` that
would replace it (Dynamo rejects a message-carrying one).
"""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = (ROOT / "unsloth_zoo" / "rl_replacements.py").read_text(encoding = "utf-8")
# One shared parse: nodes from separate parses never compare equal, which would
# make the containment tests below vacuous.
TREE = ast.parse(SRC)


# ---- the message ----------------------------------------------------------

def _mismatched_call(fn, torch):
    """Vocab-wide logits where hidden states are expected, as an unpatched
    forward returns. `lm_head` is an nn.Parameter, like
    `get_output_embeddings().weight`, and so is shape-static -- which is why the
    real report printed a concrete `[1536, 151936]` beside a symbolic `s33`.
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
    """A bare `torch._check` names neither operand, and Dynamo rejects a
    message-carrying one, so the matmul must be left to raise."""
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
    """The packed call inside `grpo_accumulated_loss`, located by AST so
    reformatting cannot make this test vacuous."""
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
    """Raw logits skip the matmul path entirely: scale and softcapping were
    already applied by the forward."""
    call = _packed_block()
    guard = _enclosing_width_test(call)
    assert guard is not None
    names = {n.func.id for n in ast.walk(ast.Module(body = guard.orelse, type_ignores = []))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "chunked_selective_log_softmax" in names, sorted(names)


def test_the_padded_path_still_dispatches():
    """The behaviour being mirrored."""
    assert "new_hidden_states_chunk.shape[-1] == lm_head.shape[1]" in SRC


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

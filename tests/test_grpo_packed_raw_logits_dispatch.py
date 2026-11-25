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

"""Packed GRPO path (UNSLOTH_GRPO_SEQ_PACKING=1) fed vocab-wide logits to lm_head; it must dispatch
on hidden width as compute_logprobs_chunk does, and a real mismatch must still name both operands
(a message-less torch._check would not; Dynamo rejects message-carrying ones)."""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = (ROOT / "unsloth_zoo" / "rl_replacements.py").read_text(encoding = "utf-8")
# One shared parse: AST nodes from separate parses never compare equal, which would make the containment tests below vacuous.
TREE = ast.parse(SRC)



def _mismatched_call(fn, torch):
    """Vocab-wide logits for hidden states; real report [1536, 151936] beside symbolic s33."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hidden, vocab = 32, 128
    logits  = torch.randn(2, 8, vocab, device = dev, dtype = torch.float32)
    lm_head = torch.nn.Parameter(
        torch.randn(vocab, hidden, device = dev, dtype = torch.float32)
    )
    index   = torch.randint(0, vocab, (2, 8), device = dev)
    return lambda: fn(logits, lm_head, index, chunks = 4)


def test_the_width_mismatch_names_both_operands():
    """A bare torch._check names neither operand and Dynamo rejects one with a message, so the matmul itself must be left to raise, naming both widths."""
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



def _packed_block():
    """Locates the packed call inside grpo_accumulated_loss by AST, so reformatting cannot make the test vacuous."""
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
    """Returns the nearest if whose test compares a .shape[-1] against lm_head.shape[...] and contains call_node."""
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
    """Raw logits skip the matmul path: scale and softcapping were already applied by the forward, so the fallback is the raw-logits helper."""
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

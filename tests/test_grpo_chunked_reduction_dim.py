"""The GRPO chunked log-softmax took its reduction dim from the wrong operand.

`chunked_hidden_states_selective_log_softmax` is compiled with dynamic = True
and fullgraph = True, and opened with

    flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

Under dynamic = True that last dim stays a free symbol. The very next thing the
function does is matmul it against lm_head, whose hidden size is concrete, and
Dynamo cannot prove the two are equal, so the guard fails before a single
element is computed:

    a and b must have same reduction dim, but got
    [((s47*s87 + 255)//256), s33] X [1536, 151936]

That is NeMo-Gym-Sudoku at cell 21, inside the generated UnslothGRPOTrainer,
which carries a copy of this function's source. Isolation with
UNSLOTH_COMPILE_DISABLE=1 (once that flag actually reached these helpers, zoo
b6638c070) made the error disappear entirely, which is what says the problem is
the guard rather than the chunk arithmetic: the ceil-division in the first
operand is just `torch.chunk` and is fine.

The fix takes the reduction dim from the operand it has to match, and states the
equality to the symbolic shape system with torch._check. Both sides are then the
same expression and there is nothing left to prove. The two are the same number
whenever the matmul is legal at all, so nothing about the maths changes -- which
is what the numeric tests here are for.
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


def test_the_reduction_dim_comes_from_lm_head():
    body = _fn_source()
    assert "hidden_states.reshape(-1, lm_head.shape[-1])" in body
    assert "hidden_states.reshape(-1, hidden_states.shape[-1])" not in body


def test_the_equality_is_stated_to_the_shape_system():
    body = _fn_source()
    assert "torch._check(hidden_states.shape[-1] == lm_head.shape[-1])" in body


def test_the_check_runs_before_the_reshape():
    """After the reshape it is too late to say anything useful."""
    body = _fn_source()
    assert body.index("torch._check(") < body.index("reshape(-1, lm_head")


def test_the_sibling_helper_is_left_alone():
    """`chunked_selective_log_softmax` reshapes against its own last dim too,
    but it has no matmul, so there is nothing for a guard to fail on and no
    reason to touch it."""
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
    # CPU keeps the shapes small; compiling this at 1536x151936 on CPU is not
    # a test, it is a wait.
    return torch, fn, "cpu", torch.float32, 32, 64, 1e-4


def _reference(torch, hidden_states, lm_head, index, temperature = 1.0):
    import torch.nn.functional as F
    logits = (hidden_states.to(lm_head.dtype) @ lm_head.t()).float() / temperature
    return torch.gather(
        F.log_softmax(logits, dim = -1), -1, index.unsqueeze(-1),
    ).squeeze(-1)


@pytest.mark.parametrize("batch,seq,chunks,temperature", [
    (2, 16, 4, 1.0),
    (4, 24, 256, 1.0),   # chunks > rows, which is where the ceil-division bites
    (3, 40, 8, 0.7),
    (1, 7, 3, 1.0),      # rows not divisible by chunks
])
def test_it_still_matches_a_plain_log_softmax(batch, seq, chunks, temperature):
    torch, fn, dev, dtype, hidden, vocab, tol = _setup()
    torch.manual_seed(0)
    hidden_states = torch.randn(batch, seq, hidden, device = dev, dtype = dtype)
    lm_head = torch.randn(vocab, hidden, device = dev, dtype = dtype) / 30
    index = torch.randint(0, vocab, (batch, seq), device = dev)

    # What the real run has and a bare call does not: the hidden dim arrives
    # marked dynamic from the surrounding compiled trainer.
    torch._dynamo.mark_dynamic(hidden_states, 2)

    got = fn(hidden_states, lm_head, index,
             chunks = chunks, temperature = temperature)
    expected = _reference(torch, hidden_states, lm_head, index,
                          temperature = temperature)
    assert tuple(got.shape) == (batch, seq)
    assert (got - expected).abs().max().item() < tol


def test_a_mismatched_lm_head_fails_loudly():
    """Without the check, `hidden_states` of the wrong width can still reshape
    cleanly whenever the element count happens to divide -- 2x16x768 against a
    1536-wide lm_head becomes 16 rows instead of 32 -- and the wrongness only
    surfaces later, as a confusing reshape error about the output."""
    torch, fn, dev, dtype, hidden, vocab, _ = _setup()
    hidden_states = torch.randn(2, 16, hidden // 2, device = dev, dtype = dtype)
    lm_head = torch.randn(vocab, hidden, device = dev, dtype = dtype)
    index = torch.randint(0, vocab, (2, 16), device = dev)
    with pytest.raises(Exception):
        fn(hidden_states, lm_head, index, chunks = 4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

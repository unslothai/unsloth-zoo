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

"""A trainable output head must get correct gradients when a chunk would hold one row.

Sequence-packed GRPO full fine-tuning chunks the micro-batch's completion tokens, so any
token count can leave a one-row chunk (49 tokens in 16 chunks: 12 of 4 rows, then 1).
Inductor compiled that chunk's head gradient into garbage: an illegal memory access with
a large vocabulary, silently wrong gradients with a small one. Each case runs in its own
process because the illegal memory access poisons the CUDA context.
"""
import json
import subprocess
import sys
import textwrap

import pytest
import torch

CHILD = textwrap.dedent(
    """
    import json, sys
    import torch
    from unsloth_zoo.rl_replacements import chunked_hidden_states_selective_log_softmax as f

    rows, head_grad = int(sys.argv[1]), sys.argv[2] == "1"
    vocab, hidden = 1024, 64
    gen = torch.Generator().manual_seed(0)
    lm_head = (torch.randn(vocab, hidden, generator = gen) * 0.05).cuda().requires_grad_(head_grad)
    states = torch.randn(1, rows, hidden, generator = gen).cuda().requires_grad_(True)
    index = torch.randint(0, vocab, (1, rows), generator = gen).cuda()
    weights = torch.randn(1, rows, generator = gen).cuda()

    out = f(states, lm_head, index, 16)
    (out * weights).sum().backward()
    got = [out.detach(), states.grad.clone()] + ([lm_head.grad.clone()] if head_grad else [])
    states.grad = None
    lm_head.grad = None

    logits = states.reshape(-1, hidden) @ lm_head.t()
    ref = (logits.gather(-1, index.reshape(-1, 1)).squeeze(-1) - logits.logsumexp(-1)).reshape(1, rows)
    (ref * weights).sum().backward()
    want = [ref.detach(), states.grad] + ([lm_head.grad] if head_grad else [])
    print(json.dumps([float((a - b).abs().max() / b.abs().max()) for a, b in zip(got, want)]))
    """
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "the fault is in Inductor's CUDA codegen")
@pytest.mark.parametrize("head_grad", [True, False])
@pytest.mark.parametrize("rows", [8, 13, 17, 24, 25, 49])
def test_one_row_chunk_matches_eager(rows, head_grad):
    run = subprocess.run(
        [sys.executable, "-c", CHILD, str(rows), "1" if head_grad else "0"],
        capture_output = True, text = True, timeout = 600,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    errors = json.loads(run.stdout.strip().splitlines()[-1])
    # logps, hidden-state gradient, head gradient: float32 against one unchunked matmul.
    assert max(errors) < 1e-4, errors

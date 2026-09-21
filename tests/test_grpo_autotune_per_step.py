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

"""`grpo_accumulated_loss` froze its mini-batch/chunk plan at the first step.

The plan was cached into `trainer.args.unsloth_grpo_mini_batch`, and the branch
meant to refresh it read

    if trainer.args.unsloth_grpo_mini_batch is None:
        if not hasattr(trainer, "_has_autotuned"):
            trainer._has_autotuned = True
            ...
            trainer.args.unsloth_grpo_mini_batch = max(1, total_rows//B)
        elif trainer._step % trainer.current_gradient_accumulation_steps == 0:
            del trainer._has_autotuned
            del trainer.args.unsloth_grpo_mini_batch
            ...

Reaching the `elif` needs `unsloth_grpo_mini_batch is None` *and*
`_has_autotuned` set, and the only writer of `_has_autotuned` makes the value
non-None in the same statement, so it never ran. GRPO completion lengths vary
step to step: a plan sized on a short first step under-chunks every later long
one (at seq_len 65536 the multiplier stays 4 where the sizing rule asks for 16).

unsloth's own copy of this function, `unsloth/models/rl_replacements.py`, sizes
per call and caches nothing, which is what this file pins here.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _Stop(Exception):
    """Raised by the stub right after the sizing block, so no GPU work is needed."""


def _trainer(torch, mini_batch = None, multiplier = None):
    lm_head = SimpleNamespace(weight = torch.zeros(37, 11))
    return SimpleNamespace(
        model = SimpleNamespace(get_output_embeddings = lambda: lm_head),
        processing_class = SimpleNamespace(pad_token_id = 0),
        use_vllm = False,
        args = SimpleNamespace(
            unsloth_grpo_mini_batch = mini_batch,
            unsloth_logit_chunk_multiplier = multiplier,
        ),
    )


def _call(rr, torch, trainer, seq_len, rows = 8):
    input_ids = torch.zeros(rows, seq_len, dtype = torch.long)
    with pytest.raises(_Stop):
        rr.grpo_accumulated_loss(
            trainer,
            input_ids,
            torch.ones(rows, seq_len, dtype = torch.long),
            seq_len,
            torch.ones(rows, seq_len, dtype = torch.long),
            torch.zeros(rows),
            None,
            None,
        )


@pytest.fixture
def sized(monkeypatch):
    """Record every sizing call and stop the function right after the block."""
    torch = pytest.importorskip("torch")
    from unsloth_zoo import rl_replacements as rr

    calls = []
    real = rr.autotune_batch_and_chunks

    def recording(total_rows, seq_len, hidden, vocab, dtype_bytes, multiplier):
        calls.append((total_rows, seq_len, multiplier))
        return real(total_rows, seq_len, hidden, vocab, dtype_bytes, multiplier)

    monkeypatch.setattr(rr, "autotune_batch_and_chunks", recording)
    monkeypatch.setattr(
        rr, "calculate_pad_tokens_in_prompt", lambda *a, **k: (_ for _ in ()).throw(_Stop())
    )
    return rr, torch, calls


def test_the_plan_is_sized_from_the_current_step(sized):
    rr, torch, calls = sized
    trainer = _trainer(torch)

    for seq_len in (16, 4096, 65536):
        _call(rr, torch, trainer, seq_len)

    assert [c[1] for c in calls] == [16, 4096, 65536], calls
    # Nothing is written back, so a later step is not handed the earlier answer.
    assert trainer.args.unsloth_grpo_mini_batch is None
    assert trainer.args.unsloth_logit_chunk_multiplier is None


def test_a_user_set_mini_batch_is_never_autotuned(sized):
    rr, torch, calls = sized
    trainer = _trainer(torch, mini_batch = 2)

    _call(rr, torch, trainer, 4096)
    _call(rr, torch, trainer, 65536)

    assert calls == [], "a mini batch size from the config must be left alone"
    assert trainer.args.unsloth_grpo_mini_batch == 2


def test_a_user_set_multiplier_survives_every_step(sized):
    rr, torch, calls = sized
    trainer = _trainer(torch, multiplier = 8)

    _call(rr, torch, trainer, 4096)
    _call(rr, torch, trainer, 65536)

    # Both calls see the config value; caching the autotuned answer back into args used to
    # overwrite it, and the refresh branch would then have dropped it to None.
    assert [c[2] for c in calls] == [8, 8], calls
    assert trainer.args.unsloth_logit_chunk_multiplier == 8

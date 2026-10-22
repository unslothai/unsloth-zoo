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

"""A raise inside grpo_accumulated_loss must restore the caller's UNSLOTH_RETURN_HIDDEN_STATES."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FLAG = "UNSLOTH_RETURN_HIDDEN_STATES"


def _run_until(monkeypatch, exc, prior):
    """Raise `exc` inside grpo_accumulated_loss after the flag is set; return the flag seen there."""
    torch = pytest.importorskip("torch")
    from unsloth_zoo import rl_replacements as rr

    if prior is None:
        monkeypatch.delenv(FLAG, raising = False)
    else:
        monkeypatch.setenv(FLAG, prior)

    seen = {}

    def boom(*_a, **_k):
        seen["flag"] = os.environ.get(FLAG)
        raise exc

    monkeypatch.setattr(rr, "calculate_pad_tokens_in_prompt", boom)
    lm_head = SimpleNamespace(weight = torch.zeros(37, 11))
    trainer = SimpleNamespace(
        model = SimpleNamespace(get_output_embeddings = lambda: lm_head),
        processing_class = SimpleNamespace(pad_token_id = 0),
        use_vllm = False,
        args = SimpleNamespace(unsloth_grpo_mini_batch = None, unsloth_logit_chunk_multiplier = None),
    )
    rows, seq_len = 4, 6
    with pytest.raises(type(exc)):
        rr.grpo_accumulated_loss(
            trainer,
            torch.zeros(rows, seq_len, dtype = torch.long),
            torch.ones(rows, seq_len, dtype = torch.long),
            seq_len,
            torch.ones(rows, seq_len, dtype = torch.long),
            torch.zeros(rows),
            None,
            None,
        )
    return seen.get("flag")


@pytest.mark.parametrize("prior", [None, "0", "1"])
@pytest.mark.parametrize(
    "exc", [RuntimeError("CUDA out of memory"), KeyboardInterrupt()], ids = ["error", "interrupt"]
)
def test_a_raise_inside_puts_the_callers_value_back(monkeypatch, exc, prior):
    during = _run_until(monkeypatch, exc, prior)
    # Control: the raise came after the flag was set.
    assert during == "1"
    assert os.environ.get(FLAG) == prior

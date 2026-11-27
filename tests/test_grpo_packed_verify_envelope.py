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

"""Grad packing self-verify cost: a passing verify trusts up to 2x its shape, a re-verify of trusted packing scores
at most 3 rows alone (always the last document of the stream), the first verify scores every row, and a
document-boundary leak still disables packing on either. The real grad block is exec'd."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

torch = pytest.importorskip("torch")

import unsloth_zoo.rl_replacements as rr
import test_grpo_packed_grad_oom_recovery as h


class _LeakModel(h._Model):
    """Exact unless `leak`: the packed forward then moves every document after the first, which is what attention
    that ignores packed_seq_lengths does. Counts per-row (unpacked) forwards."""

    def __init__(self):
        super().__init__()
        self.leak = False
        self.per_row = 0

    def forward(self, input_ids = None, packed_seq_lengths = None, **kwargs):
        out = super().forward(input_ids = input_ids, packed_seq_lengths = packed_seq_lengths, **kwargs)
        if packed_seq_lengths is None:
            self.per_row += 1
            return out
        if self.leak:
            logits = out.logits.clone()
            first = int(packed_seq_lengths[0])
            logits[0, first:] = logits[0, first:].roll(1, dims = -1) * 3
            out.logits = logits
        return out


def _batch(rows, keep = h.KEEP, prompt = 3):
    g = torch.Generator().manual_seed(rows)
    ids = torch.randint(1, h.VOCAB, (rows, prompt + keep), generator = g)
    ids[0, 0] = h.PAD_ID
    return ids


def _run(model, rows):
    model.per_row = 0
    return h._run(model, _batch(rows))


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_GRPO_SEQ_PACKING_VERIFY", raising = False)
    monkeypatch.delenv("UNSLOTH_GRPO_SEQ_PACKING_VERIFY_GROWTH", raising = False)


def _trusted():
    model = _LeakModel()
    ns = _run(model, 3)
    assert ns["_pack_use"] is True and model.per_row == 3
    assert model._unsloth_seq_packing_grad_verified_T == 2 * ns["_pack_T"]
    return model, ns["_pack_T"]


def test_within_twice_the_verified_shape_skips_the_verify():
    model, T = _trusted()
    ns = _run(model, 5)
    assert ns["_pack_T"] <= 2 * T
    assert ns["_pack_use"] is True and model.per_row == 0


def test_reverify_past_the_bound_scores_three_rows():
    model, T = _trusted()
    ns = _run(model, 8)
    assert ns["_pack_T"] > 2 * T
    assert ns["_pack_use"] is True and model.per_row == 3
    assert ns["_pack_vrows"][-1] == 7


def test_boundary_leak_fails_the_bounded_reverify():
    model, _ = _trusted()
    model.leak = True
    ns = _run(model, 8)
    assert model.per_row == 3
    assert ns["_pack_use"] is False and ns["_pack_diff"] >= 1.5
    assert model._unsloth_seq_packing_grad_ok is False


def test_boundary_leak_fails_the_first_verify():
    model = _LeakModel()
    model.leak = True
    ns = _run(model, 8)
    assert model.per_row == 8
    assert ns["_pack_use"] is False and model._unsloth_seq_packing_grad_ok is False


def test_force_verify_scores_every_row(monkeypatch):
    model, _ = _trusted()
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING_VERIFY", "1")
    _run(model, 8)
    assert model.per_row == 8


def test_growth_one_keeps_the_exact_shape(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING_VERIFY_GROWTH", "1")
    model = _LeakModel()
    ns = _run(model, 3)
    assert model._unsloth_seq_packing_grad_verified_T == ns["_pack_T"]


def test_rope_switch_caps_the_bound():
    longrope = SimpleNamespace(config = SimpleNamespace(
        rope_scaling = {"rope_type": "longrope", "original_max_position_embeddings": 4096}))
    assert rr.grpo_packing_trusted_bound(longrope, 3000) == 4096
    assert rr.grpo_packing_trusted_bound(longrope, 5000) == 5000
    assert rr.grpo_packing_trusted_bound(SimpleNamespace(config = SimpleNamespace()), 300) == 600


def test_length_dependent_rope_reverifies_every_row():
    """Rows on either side of a LongRoPE switch run different frequencies alone, so a 3-row sample could miss the
    short side: such models keep scoring every row."""
    model, T = _trusted()
    model.config = SimpleNamespace(rope_scaling = {"rope_type": "longrope", "original_max_position_embeddings": 4096})
    ns = _run(model, 8)
    assert ns["_pack_T"] > 2 * T
    assert model.per_row == 8 and ns["_pack_use"] is True
    assert rr.grpo_packing_verify_rows([5, 6, 7, 8, 9], [1] * 5, True, model = model) == [0, 1, 2, 3, 4]

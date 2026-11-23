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

"""The GRPO policy pass must keep packing and PrefixGrouper within one padded mini-batch's tokens, and keep
PrefixGrouper off when forwards are collective (sharded FSDP2), so every rank runs the same forwards."""

import collections
import types

import pytest
import torch

import unsloth_zoo.rl_replacements as rr

V, H = 16, 8
_IDS = torch.tensor([
    [0, 0, 1, 2, 3, 4, 5, 6],
    [0, 0, 1, 2, 7, 8, 9, 0],
    [1, 2, 6, 7, 3, 4, 6, 9],
    [1, 2, 6, 7, 5, 0, 0, 0],
    [0, 3, 2, 3, 5, 9, 8, 7],
    [0, 3, 2, 3, 11, 12, 0, 0],
])
_COMPLETION = 4
ROWS, L = _IDS.shape


class _Model(torch.nn.Module):
    """Position-local, so packed, grouped and per-row forwards agree; records every forward."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(V, H)
        self.mix = torch.nn.Linear(H, H)
        self.head = torch.nn.Linear(H, V, bias = False)
        self.device = torch.device("cpu")
        self.calls = []

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids = None, prefix_seg_info = None, packed_seq_lengths = None, **kwargs):
        kind = "grouped" if prefix_seg_info is not None else ("packed" if packed_seq_lengths is not None else "padded")
        self.calls.append((kind, input_ids.numel()))
        return types.SimpleNamespace(logits = torch.tanh(self.mix(self.emb(input_ids))))


def _trainer(model, mini_batch):
    return types.SimpleNamespace(
        args = types.SimpleNamespace(
            unsloth_grpo_mini_batch = mini_batch, unsloth_logit_chunk_multiplier = 1, off_policy_mask_threshold = None,
        ),
        processing_class = types.SimpleNamespace(pad_token_id = 0),
        model = model,
        accelerator = types.SimpleNamespace(
            unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None, gather = lambda t: t,
        ),
        control = types.SimpleNamespace(should_evaluate = False),
        _metrics = {"train": collections.defaultdict(list), "eval": collections.defaultdict(list)},
        use_vllm = False,
        _autocast_dtype = None,
        beta = 0.0,
        num_generations = 2,
    )


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    import unsloth_zoo.fused_losses.fused_grpo as _fused_grpo
    monkeypatch.setattr(_fused_grpo, "GRPO_FWD_GRAD", False)
    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_SKIP_ZERO_ADVANTAGE", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "1")


def _prefix_grouper(monkeypatch):
    """PrefixGrouper with the real layout builder, no token-ratio gate and no verify (trusted at once)."""
    pg = pytest.importorskip("unsloth.utils.prefix_grouper")
    build = lambda *a, **k: pg.build_group_layout(*a, apply_tokr_gate = False, **k)
    funcs = (build, lambda model = None: True, lambda: False, pg.tol_ok, pg.TOL_KILL)
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "1")
    monkeypatch.setattr(rr.grpo_accumulated_loss, "_pg_funcs", funcs, raising = False)


def _run(mini_batch, model = None):
    model = model or _Model()
    model.train(True)
    mask = (_IDS[:, -_COMPLETION:] != 0).long()
    out = rr.grpo_accumulated_loss(
        _trainer(model, mini_batch), _IDS, (_IDS != 0).long(), _COMPLETION, mask, torch.ones(ROWS), None, None,
        loss_type = "dapo", max_completion_length = 8, num_items_in_batch = mask.sum().float(),
        current_gradient_accumulation_steps = 1, epsilon_low = 0.2, epsilon_high = 0.28,
    )
    out[0].backward()
    return model.calls


def test_packing_stays_within_the_mini_batch(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    # 6 splits: one row of L tokens per forward.
    calls = _run(mini_batch = ROWS)
    assert all(kind == "padded" for kind, _ in calls), calls
    assert max(n for _, n in calls) <= L, calls


def test_packing_still_runs_when_it_fits(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    assert "packed" in [kind for kind, _ in _run(mini_batch = 1)]


def test_prefix_grouper_stays_within_the_mini_batch(monkeypatch):
    _prefix_grouper(monkeypatch)
    calls = _run(mini_batch = ROWS)
    assert all(kind == "padded" for kind, _ in calls), calls
    assert max(n for _, n in calls) <= L, calls


def test_prefix_grouper_still_runs_when_it_fits(monkeypatch):
    _prefix_grouper(monkeypatch)
    assert [kind for kind, _ in _run(mini_batch = 1)] == ["grouped"]


def test_prefix_grouper_is_off_when_forwards_are_collective(monkeypatch):
    """Sharded FSDP2: unsloth turns packing off; PrefixGrouper must follow, or a rank whose batch groups runs a
    different number of forwards than one whose batch does not, and the all-gathers hang."""
    _prefix_grouper(monkeypatch)
    monkeypatch.setattr(rr, "grpo_forwards_are_collective", lambda model: True)
    monkeypatch.setattr(rr, "grpo_agree_mini_batch", lambda B, device: B)
    for mini_batch in (1, 2):
        model = _Model()
        model._unsloth_seq_packing_grad_ok = False
        calls = _run(mini_batch, model)
        assert [kind for kind, _ in calls] == ["padded"] * mini_batch, calls


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

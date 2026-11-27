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

"""Score centering keeps GRPO sequence packing on: the packed forward gathers the centering head at the packed
completion rows, and the centered loss inputs match the padded loop's in fp64."""

import inspect
import types

import pytest
import torch

import unsloth_zoo.rl_replacements as _rl
import unsloth_zoo.score_centering as _sc

HIDDEN, VOCAB, K, PAD = 6, 11, 3, 0
D64 = torch.float64

# Left-padded prompts | right-padded completions (logits_to_keep = 4), pad id 0.
INPUT_IDS = torch.tensor([
    [0, 0, 1, 2, 3, 4, 5, 0],
    [0, 7, 8, 9, 6, 5, 4, 3],
    [1, 2, 3, 4, 9, 0, 0, 0],
    [0, 0, 0, 5, 7, 8, 6, 0],
])
LOGITS_TO_KEEP = 4
COMPLETION_MASK = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 0]])
_REAL_FORWARD = _rl.UnslothEfficientGRPO.forward


class _Model(torch.nn.Module):
    """Token-local decoder (no attention), so the packed and padded forwards are equal and only the row
    bookkeeping differs."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(3)
        self.emb = torch.nn.Parameter(torch.randn(VOCAB, HIDDEN, generator = g, dtype = D64))
        self.mix = torch.nn.Parameter(torch.randn(HIDDEN, HIDDEN, generator = g, dtype = D64))
        self.head = torch.nn.Linear(HIDDEN, VOCAB, bias = False, dtype = D64)
        with torch.no_grad():
            self.head.weight.copy_(torch.randn(VOCAB, HIDDEN, generator = g, dtype = D64))
        self.device = torch.device("cpu")
        self.packed_calls = 0

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids = None, packed_seq_lengths = None, **kwargs):
        if packed_seq_lengths is not None:
            self.packed_calls += 1
        return types.SimpleNamespace(logits = torch.tanh(self.emb[input_ids] @ self.mix))


def _heads():
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(0, VOCAB, (INPUT_IDS.shape[0], LOGITS_TO_KEEP, K), generator = g).to(torch.int32)
    logps = torch.log_softmax(torch.randn(INPUT_IDS.shape[0], LOGITS_TO_KEEP, VOCAB, generator = g), -1)
    logps = logps.gather(-1, ids.long()).to(torch.float32)
    ids[2] = -1  # an unmatched row
    logps[2] = -float("inf")
    ids[1, 3, 1:] = -1  # a short head
    logps[1, 3, 1:] = -float("inf")
    return ids, logps


def _run(monkeypatch, packing, bins = None):
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "1" if packing else "0")
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_SKIP_ZERO_ADVANTAGE", "0")
    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "1")
    import unsloth_zoo.fused_losses.fused_grpo as _fused_grpo
    monkeypatch.setattr(_fused_grpo, "GRPO_FWD_GRAD", False)
    monkeypatch.setattr(_rl, "grpo_score_centering_k", lambda *a, **kw: K)
    if bins is not None:
        monkeypatch.setattr(_rl, "grpo_packing_bins", lambda *a, **kw: bins)
    ids, logps = _heads()
    monkeypatch.setattr(_sc, "lookup_rollout_topk", lambda *a, **kw: (ids, logps, torch.ones(ids.shape[0], dtype = torch.bool)))
    seen = {}
    real = _REAL_FORWARD

    class _Loss:
        @staticmethod
        def apply(*args):
            bound = inspect.signature(real).bind(None, *args).arguments
            seen["new"] = bound["_new_logps"]
            seen["sc"] = bound.get("_sc_correction")
            seen["mask"] = bound["_mask"]
            zero = torch.zeros((), dtype = D64)
            return zero, zero, zero, zero, zero, zero

    monkeypatch.setattr(_rl, "UnslothEfficientGRPO", _Loss)
    model = _Model()
    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(unsloth_grpo_mini_batch = 1, unsloth_logit_chunk_multiplier = 1),
        processing_class = types.SimpleNamespace(pad_token_id = PAD),
        model = model,
        accelerator = types.SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None),
        use_vllm = True,
        _autocast_dtype = None,
        beta = 0.0,
        _metrics = {"train": {}},
    )
    _rl.grpo_accumulated_loss(
        trainer, INPUT_IDS, (INPUT_IDS != PAD).long(), LOGITS_TO_KEEP, COMPLETION_MASK,
        torch.linspace(-1, 1, INPUT_IDS.shape[0]), None, None,
        loss_type = "reinforce", num_items_in_batch = int(COMPLETION_MASK.sum()),
    )
    assert seen.get("sc") is not None, "the centering term did not reach the loss"
    mask = seen["mask"].to(D64)
    # A scalar that depends on both loss inputs, for the gradient check.
    total = (seen["new"].to(D64) * mask).sum() + (seen["sc"].to(D64) * mask).sum()
    params = [model.emb, model.mix, model.head.weight]
    grads = torch.autograd.grad(total, params)
    return seen, grads, model.packed_calls


@pytest.mark.parametrize("bins", [None, [(0, 2), (2, 4)]])
def test_score_centering_keeps_sequence_packing(monkeypatch, bins):
    padded, padded_grads, padded_calls = _run(monkeypatch, packing = False)
    packed, packed_grads, packed_calls = _run(monkeypatch, packing = True, bins = bins)
    assert padded_calls == 0 and packed_calls == (1 if bins is None else len(bins))
    mask = padded["mask"].bool()
    assert torch.equal(mask, packed["mask"].bool())
    assert (packed["new"][mask] - padded["new"][mask]).abs().max().item() < 1e-6
    assert padded["sc"].abs().max().item() > 0  # the term is not trivially zero
    assert (packed["sc"][mask] - padded["sc"][mask]).abs().max().item() < 1e-6
    for a, b in zip(packed_grads, padded_grads):
        assert (a - b).abs().max().item() < 1e-6

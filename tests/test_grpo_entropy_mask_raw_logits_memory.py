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

"""Raw-logit entropies used only for the top_entropy_quantile mask carry no gradient (TRL uses no_grad), so they must
not keep full-vocab fp32 tensors; with the entropy bonus on, the entropy keeps its graph."""
import types

import pytest
import torch

import unsloth_zoo.rl_replacements as rr

HIDDEN, VOCAB = 4, 16


class _RawLogitsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(4)
        self.emb = torch.nn.Embedding(VOCAB, HIDDEN)
        self.head = torch.nn.Linear(HIDDEN, VOCAB, bias = False)
        self.device = torch.device("cpu")

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids = None, logits_to_keep = None, **kwargs):
        logits = self.head(self.emb(input_ids))
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return types.SimpleNamespace(logits = logits)


def _run(monkeypatch, quantile, coef):
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "0")
    monkeypatch.setattr(rr.chunked_hidden_states_selective_log_softmax_recompute, "_use_compiled", False, raising = False)

    class _Loss:
        @staticmethod
        def apply(*args):
            zero = torch.zeros(())
            return zero, zero, zero, zero, zero, zero

    monkeypatch.setattr(rr, "UnslothEfficientGRPO", _Loss)
    seen = {}
    args = types.SimpleNamespace(
        unsloth_grpo_mini_batch = 1, unsloth_logit_chunk_multiplier = 1, top_entropy_quantile = quantile,
        entropy_coef = coef,
    )
    model = _RawLogitsModel()

    def get_high_entropy_mask(ent, mask, q):
        seen["entropies"] = ent.clone()
        return torch.ones_like(mask, dtype = torch.bool)

    trainer = types.SimpleNamespace(
        args = args, model = model, processing_class = types.SimpleNamespace(pad_token_id = 0),
        accelerator = types.SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None),
        use_vllm = False, beta = 0.0, vllm_importance_sampling_correction = False, _autocast_dtype = None,
        get_high_entropy_mask = get_high_entropy_mask, _metrics = {"train": {}, "eval": {}},
        control = types.SimpleNamespace(should_evaluate = False),
    )
    ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 6, 7, 3]])
    saved = []

    def pack(t):
        if t.dtype == torch.float32 and t.dim() >= 1 and t.shape[-1] == VOCAB and t.dim() == 3:
            saved.append(tuple(t.shape))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        rr.grpo_accumulated_loss(
            trainer, ids, torch.ones_like(ids), 3, torch.ones(2, 3, dtype = torch.long),
            torch.ones(2), None, None, loss_type = "grpo", num_items_in_batch = 6,
        )
    with torch.no_grad():
        logits = model(ids, logits_to_keep = 4).logits[:, :-1].float()
        log_p = logits.log_softmax(-1)
        want = -(log_p.exp() * log_p).sum(-1)
    return saved, seen.get("entropies"), want


def test_mask_only_entropy_keeps_no_vocab_tensors_for_backward(monkeypatch):
    base, _, _ = _run(monkeypatch, quantile = 1.0, coef = 0.0)
    saved, got, want = _run(monkeypatch, quantile = 0.5, coef = 0.0)
    assert len(saved) == len(base), (saved, base)
    torch.testing.assert_close(got, want, rtol = 1e-6, atol = 1e-6)


def test_entropy_bonus_still_keeps_its_graph(monkeypatch):
    base, _, _ = _run(monkeypatch, quantile = 1.0, coef = 0.0)
    saved, _, _ = _run(monkeypatch, quantile = 1.0, coef = 0.1)
    assert len(saved) > len(base)

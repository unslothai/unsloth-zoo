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

"""GRPO grad path: a pad id that is one of the EOS ids (Qwen-style pad <|endoftext|>, stop <|im_end|>) must not
turn PrefixGrouper off unless some completion really ends on the pad id."""
import inspect
import types

import pytest
import torch

import unsloth_zoo.rl_replacements as _rl

HIDDEN, VOCAB = 4, 12
IM_END, ENDOFTEXT = 10, 11   # stop token, pad token; both are generation EOS ids
PROMPT_LEN = 3
_REAL_FORWARD = _rl.UnslothEfficientGRPO.forward


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    import unsloth_zoo.fused_losses.fused_grpo as _fused_grpo
    monkeypatch.setattr(_fused_grpo, "GRPO_FWD_GRAD", False)
    monkeypatch.setenv("UNSLOTH_GRPO_SKIP_ZERO_ADVANTAGE", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "1")
    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "0")


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(VOCAB, HIDDEN)
        self.head = torch.nn.Linear(HIDDEN, VOCAB, bias = False)
        self.device = torch.device("cpu")
        self.config = types.SimpleNamespace()

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids = None, attention_mask = None, logits_to_keep = None, **kwargs):
        hidden = self.emb(input_ids)
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        return types.SimpleNamespace(logits = hidden)


def _trl_completion_mask(completion_ids, eos_ids):
    is_eos = torch.isin(completion_ids, torch.as_tensor(eos_ids))
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype = torch.long)
    eos_idx[is_eos.any(dim = 1)] = is_eos.int().argmax(dim = 1)[is_eos.any(dim = 1)]
    return (torch.arange(is_eos.size(1)).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()


def _run(monkeypatch, completions, ref_logps = False):
    """grpo_accumulated_loss on 2 prompts x 2 generations; returns (PrefixGrouper layout calls, loss mask)."""
    P = ENDOFTEXT
    prompts = [[P, 1, 2], [P, 1, 2], [3, 4, 5], [3, 4, 5]]
    input_ids = torch.tensor([p + c for p, c in zip(prompts, completions)])
    keep = input_ids.shape[1] - PROMPT_LEN
    completion_mask = _trl_completion_mask(input_ids[:, PROMPT_LEN:], [IM_END, ENDOFTEXT])
    attention_mask = torch.cat([(input_ids[:, :PROMPT_LEN] != P).long(), completion_mask], dim = 1)
    calls = []

    def _build_layout(*args, **kwargs):
        calls.append(args)
        return None   # no layout: the full-row path still runs, so only the gate is under test

    monkeypatch.setattr(
        _rl.grpo_accumulated_loss, "_pg_funcs",
        (_build_layout, lambda model = None: True, lambda: True, lambda *a: True, 1.0), raising = False,
    )
    seen = {}

    class _Loss:
        @staticmethod
        def apply(*args):
            seen["mask"] = inspect.signature(_REAL_FORWARD).bind(None, *args).arguments["_mask"].clone()
            zero = torch.zeros(())
            return zero, zero, zero, zero, zero, zero

    monkeypatch.setattr(_rl, "UnslothEfficientGRPO", _Loss)
    ref = None
    if ref_logps:
        width = keep + int(_rl.calculate_pad_tokens_in_prompt(input_ids, keep, P).max())
        ref = torch.zeros(input_ids.shape[0], width)
    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(unsloth_grpo_mini_batch = 4, unsloth_logit_chunk_multiplier = 1),
        processing_class = types.SimpleNamespace(pad_token_id = P, eos_token_id = IM_END),
        eos_token_ids = [IM_END, ENDOFTEXT],
        model = _Model(),
        accelerator = types.SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None),
        use_vllm = False,
        num_generations = 2,
        _autocast_dtype = None,
        beta = 0.0 if ref is None else 0.1,
        _unsloth_grpo_logps_cover_eos = True,
    )
    _rl.grpo_accumulated_loss(
        trainer, input_ids, attention_mask, keep, completion_mask,
        torch.zeros(input_ids.shape[0]), None, ref,
        loss_type = "dapo", num_items_in_batch = int(completion_mask.sum()),
    )
    return len(calls), seen["mask"].bool(), completion_mask.bool()


@pytest.mark.parametrize("ref_logps", [False, True])
def test_pad_in_eos_ids_keeps_prefix_grouper_on(monkeypatch, ref_logps):
    # Every completion stops on <|im_end|>; <|endoftext|> is only padding.
    P = ENDOFTEXT
    completions = [[6, IM_END, P, P], [7, 8, IM_END, P], [9, IM_END, P, P], [6, 7, 8, IM_END]]
    calls, mask, trl = _run(monkeypatch, completions, ref_logps = ref_logps)
    assert calls == 1, "PrefixGrouper must engage when no completion ends on the pad id"
    assert mask.sum(dim = 1).tolist() == trl.sum(dim = 1).tolist() == [2, 3, 2, 4]


def test_completion_ending_on_the_pad_id_keeps_prefix_grouper_off(monkeypatch):
    # Row 1 stopped on <|endoftext|> itself: its EOS slot is kept, which PrefixGrouper's pad-id layout would drop.
    P = ENDOFTEXT
    completions = [[6, IM_END, P, P], [7, 8, P, P], [9, IM_END, P, P], [6, 7, 8, IM_END]]
    calls, mask, trl = _run(monkeypatch, completions)
    assert calls == 0
    assert mask.sum(dim = 1).tolist() == trl.sum(dim = 1).tolist() == [2, 3, 2, 4]

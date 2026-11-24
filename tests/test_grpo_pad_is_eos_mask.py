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

"""GRPO text path with pad == eos: the loss mask keeps the first EOS of each completion, as TRL's completion_mask does."""
import inspect
import types

import pytest
import torch

import unsloth_zoo.rl_replacements as _rl

HIDDEN, VOCAB = 4, 10
EOS, OTHER_PAD = 9, 0
PROMPT_LEN = 3
_REAL_FORWARD = _rl.UnslothEfficientGRPO.forward


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    import unsloth_zoo.fused_losses.fused_grpo as _fused_grpo
    monkeypatch.setattr(_fused_grpo, "GRPO_FWD_GRAD", False)
    monkeypatch.setenv("UNSLOTH_GRPO_SKIP_ZERO_ADVANTAGE", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "0")


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(VOCAB, HIDDEN)
        self.head = torch.nn.Linear(HIDDEN, VOCAB, bias = False)
        self.device = torch.device("cpu")

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids = None, attention_mask = None, logits_to_keep = None, **kwargs):
        hidden = self.emb(input_ids)
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        return types.SimpleNamespace(logits = hidden)


def _trl_completion_mask(completion_ids, eos):
    # GRPOTrainer._generate_and_score_completions (TRL 0.22.2 to 1.18): keep through the first EOS.
    is_eos = torch.isin(completion_ids, torch.as_tensor(eos))
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype = torch.long)
    eos_idx[is_eos.any(dim = 1)] = is_eos.int().argmax(dim = 1)[is_eos.any(dim = 1)]
    return (torch.arange(is_eos.size(1)).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()


def _batch(pad):
    """Left-padded prompts, right-padded completions of mixed lengths; row 2 is truncated (no EOS)."""
    P = pad
    prompts = [[P, 1, 2], [3, 4, 5], [P, P, 6], [7, 1, 2]]
    completions = [[3, 4, EOS, P, P], [5, EOS, P, P, P], [1, 2, 3, 4, 5], [EOS, P, P, P, P]]
    return torch.tensor([p + c for p, c in zip(prompts, completions)])


def _loss_mask(monkeypatch, input_ids, pad, ref_logps = None, eos_ids = EOS, **trainer_attrs):
    seen = {}

    class _Loss:
        @staticmethod
        def apply(*args):
            seen["mask"] = inspect.signature(_REAL_FORWARD).bind(None, *args).arguments["_mask"].clone()
            zero = torch.zeros(())
            return zero, zero, zero, zero, zero, zero

    monkeypatch.setattr(_rl, "UnslothEfficientGRPO", _Loss)
    keep = input_ids.shape[1] - PROMPT_LEN
    completion_mask = _trl_completion_mask(input_ids[:, PROMPT_LEN:], eos_ids)
    attention_mask = torch.cat([(input_ids[:, :PROMPT_LEN] != pad).long(), completion_mask], dim = 1)
    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(unsloth_grpo_mini_batch = 2, unsloth_logit_chunk_multiplier = 1),
        processing_class = types.SimpleNamespace(pad_token_id = pad, eos_token_id = EOS),
        model = _Model(),
        accelerator = types.SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None),
        use_vllm = False,
        _autocast_dtype = None,
        beta = 0.0 if ref_logps is None else 0.1,
        **trainer_attrs,
    )
    _rl.grpo_accumulated_loss(
        trainer, input_ids, attention_mask, keep, completion_mask,
        torch.zeros(input_ids.shape[0]), None, ref_logps,
        loss_type = "dapo", num_items_in_batch = int(completion_mask.sum()),
    )
    return seen["mask"].bool(), completion_mask.bool()


def _per_row_kept(mask):
    return mask.sum(dim = 1).tolist()


def _frozen_id_rule(input_ids, keep, pad):
    """The mask before this change: shift past the left pad, then drop every pad id."""
    left_pad = (input_ids[:, :-keep] == pad).sum(dim = 1)
    max_left_pad = int(left_pad.max())
    packed = _rl.left_pack_padding(input_ids, pad)[:, -(keep + max_left_pad):]
    idx = torch.arange(packed.shape[1]).unsqueeze(0)
    return (idx >= (max_left_pad - left_pad).unsqueeze(1)) & (packed != pad)


def test_pad_is_eos_keeps_the_first_eos_of_each_completion(monkeypatch):
    ids = _batch(EOS)
    mask, trl = _loss_mask(monkeypatch, ids, EOS)
    # Kept tokens per row equal TRL's (EOS included); the truncated row gains nothing.
    assert _per_row_kept(mask) == _per_row_kept(trl) == [3, 2, 5, 1]


def test_pad_is_eos_kept_slot_holds_the_eos_token(monkeypatch):
    ids = _batch(EOS)
    mask, _ = _loss_mask(monkeypatch, ids, EOS)
    keep = ids.shape[1] - PROMPT_LEN
    left_pad = _rl.calculate_pad_tokens_in_prompt(ids, keep, EOS)
    window = _rl.left_pack_padding(ids, EOS)[:, -(keep + int(left_pad.max())):]
    for row in range(ids.shape[0]):
        kept = window[row][mask[row]].tolist()
        real = ids[row, PROMPT_LEN:][_trl_completion_mask(ids[:, PROMPT_LEN:], EOS)[row].bool()].tolist()
        assert kept == real


def test_pad_distinct_from_eos_is_bitwise_unchanged(monkeypatch):
    ids = _batch(OTHER_PAD)
    mask, trl = _loss_mask(monkeypatch, ids, OTHER_PAD)
    assert torch.equal(mask, _frozen_id_rule(ids, ids.shape[1] - PROMPT_LEN, OTHER_PAD))
    assert _per_row_kept(mask) == _per_row_kept(trl)


def test_create_completion_attention_mask_default_is_the_id_rule():
    torch.manual_seed(0)
    for _ in range(20):
        ids = torch.randint(0, 4, (6, 9))
        keep = 5
        left_pad = _rl.calculate_pad_tokens_in_prompt(ids, keep, 0)
        max_left_pad = int(left_pad.max())
        window = _rl.left_pack_padding(ids, 0)[:, -(keep + max_left_pad):]
        got = _rl.create_completion_attention_mask(window, left_pad, max_left_pad, 0)
        assert torch.equal(got, _frozen_id_rule(ids, keep, 0))
        assert _rl.grpo_pad_is_eos_rows(
            types.SimpleNamespace(processing_class = types.SimpleNamespace(pad_token_id = 0, eos_token_id = 3)),
            ids[:, -keep:],
        ) is None


def test_trl_eos_token_ids_count_as_eos(monkeypatch):
    # TRL 1.x stops on generation_config EOS ids too (trainer.eos_token_ids).
    trainer = types.SimpleNamespace(
        processing_class = types.SimpleNamespace(pad_token_id = 8, eos_token_id = EOS), eos_token_ids = [EOS, 8],
    )
    rows = _rl.grpo_pad_is_eos_rows(trainer, torch.tensor([[1, 8, 8], [1, 2, 3]]))
    assert rows.tolist() == [True, False]


def test_other_eos_terminator_with_pad_id_padding_gets_no_extra_slot(monkeypatch):
    # pad 9 is one of TRL 1.x's eos_token_ids [8, 9]; a row that stopped on 8 is then padded with 9.
    # Only a row whose first EOS-set token is the pad id needs the extra slot.
    trainer = types.SimpleNamespace(
        processing_class = types.SimpleNamespace(pad_token_id = EOS, eos_token_id = EOS), eos_token_ids = [8, EOS],
    )
    completions = torch.tensor([[1, 8, EOS, EOS], [1, EOS, EOS, EOS], [8, EOS, EOS, EOS], [1, 2, 3, 4]])
    assert _rl.grpo_pad_is_eos_rows(trainer, completions).tolist() == [False, True, False, False]

    prompts = torch.tensor([[EOS, 1, 2], [3, 4, 5], [EOS, EOS, 6], [7, 1, 2]])
    input_ids = torch.cat([prompts, completions], dim = 1)
    mask, completion_mask = _loss_mask(monkeypatch, input_ids, EOS, eos_ids = [8, EOS], eos_token_ids = [8, EOS])
    # [1, 8, 9, 9] -> [1, 1, 0, 0], as TRL's completion_mask; [1, 9, 9, 9] keeps its EOS.
    assert completion_mask.int().tolist()[0] == [1, 1, 0, 0]
    assert _per_row_kept(mask) == completion_mask.sum(dim = 1).tolist() == [2, 2, 1, 4]


def test_ref_logps_without_eos_coverage_keep_the_old_mask(monkeypatch):
    # Unsloth's packed old / ref path skips pad-id tokens, so without its promise the EOS slot stays out.
    ids = _batch(EOS)
    keep = ids.shape[1] - PROMPT_LEN
    width = keep + int(_rl.calculate_pad_tokens_in_prompt(ids, keep, EOS).max())
    ref = torch.zeros(ids.shape[0], width)
    mask, _ = _loss_mask(monkeypatch, ids, EOS, ref_logps = ref)
    assert torch.equal(mask, _frozen_id_rule(ids, keep, EOS))
    mask, trl = _loss_mask(monkeypatch, ids, EOS, ref_logps = ref, _unsloth_grpo_logps_cover_eos = True)
    assert _per_row_kept(mask) == _per_row_kept(trl)


class _PositionLocal(torch.nn.Module):
    """Position-local logits, so the packed forward and the per-row forward agree exactly."""
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(VOCAB, HIDDEN)
        self.head = torch.nn.Linear(HIDDEN, VOCAB, bias = False)

    def forward(self, input_ids = None, position_ids = None, packed_seq_lengths = None, use_cache = None, **kwargs):
        return types.SimpleNamespace(logits = self.head(torch.tanh(self.emb(input_ids))))


def test_packed_forward_scores_the_kept_eos(monkeypatch):
    """The sequence-packing block (default on) must pack and score the EOS that shares the pad id."""
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "1")
    import contextlib
    import textwrap
    src = inspect.getsource(_rl.grpo_accumulated_loss)
    start = src.rindex("\n", 0, src.index("    new_logprobs = None")) + 1
    end = src.rindex("\n", 0, src.index("    # ---- PrefixGrouper resolution")) + 1
    ids = _batch(EOS)
    keep = ids.shape[1] - PROMPT_LEN
    trainer = types.SimpleNamespace(processing_class = types.SimpleNamespace(pad_token_id = EOS, eos_token_id = EOS))
    eos_rows = _rl.grpo_pad_is_eos_rows(trainer, ids[:, -keep:])
    left_pad = _rl.calculate_pad_tokens_in_prompt(ids, keep, EOS)
    max_left_pad = int(left_pad.max())
    packed = _rl.left_pack_padding(ids, EOS)
    width = keep + max_left_pad
    completion_mask = _rl.create_completion_attention_mask(packed[:, -width:], left_pad, max_left_pad, EOS, eos_rows = eos_rows)
    attention_mask = packed != EOS
    attention_mask[:, -width:] |= completion_mask
    model = _PositionLocal()
    lm_head = model.head.weight
    ns = {
        "os": _rl.os, "torch": torch,
        "chunked_hidden_states_selective_log_softmax": _rl.chunked_hidden_states_selective_log_softmax,
        "chunked_selective_log_softmax": _rl.chunked_selective_log_softmax,
        "_refresh_lm_head": lambda: None, "_fp32_bind": lambda entropy = False: None, "lm_head_bias": None,
        "_fused_route": False, "_fused_pack": None, "_pg_layout": None,
        "create_completion_attention_mask": _rl.create_completion_attention_mask,
        "device_synchronize": lambda *a, **k: None, "UNSLOTH_ENABLE_LOGGING": False,
        "trainer": trainer, "unwrapped_model": model, "lm_head": lm_head,
        "input_ids": packed, "attention_mask": attention_mask, "pad_eos_rows": eos_rows,
        "left_pad_tokens_per_prompt": left_pad, "max_left_pad": max_left_pad,
        "completion_mask": completion_mask, "logits_to_keep": keep,
        "total_rows": ids.shape[0], "batch_size": ids.shape[0], "_collective_forwards": False, "multiplier": 1,
        "autocaster": contextlib.nullcontext(), "pixel_values": None, "token_type_ids": None, "mm_token_type_ids": None,
        "_pg_skip_pack": False, "logit_scale_multiply": 0.0, "logit_scale_divide": 0.0,
        "logit_softcapping": 0.0, "temperature": 1.0, "_need_entropy": False,
    }
    exec(textwrap.dedent(src[start:end]), ns)
    assert ns["_pack_use"], "the packed path must engage"
    assert torch.equal(ns["_pack_keep"], attention_mask)
    got = ns["_pack_result"][:, -width:]
    with torch.no_grad():
        logps = torch.log_softmax(model(input_ids = packed).logits.float(), dim = -1)
    # The kept EOS column holds the EOS log-prob given the row's real tokens.
    for row in range(ids.shape[0]):
        cols = completion_mask[row].nonzero().squeeze(1)
        if not bool(eos_rows[row]):
            continue
        col = int(cols[-1])
        full_col = packed.shape[1] - width + col
        want = logps[row, full_col - 1, EOS]
        torch.testing.assert_close(got[row, col].float(), want, rtol = 1e-4, atol = 1e-4)
        assert packed[row, full_col] == EOS

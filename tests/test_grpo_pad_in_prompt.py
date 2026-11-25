# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO text path: prompt tokens sharing the pad id (Qwen <|im_end|>) stay in prompt, since the id
rule let left_pack_padding drop them; padding comes from TRL's attention mask on disagreement;
policy sees TRL prompt+completion, loss mask is TRL's completion_mask; composes with EOS sharing
pad id."""

import os
import inspect
from types import SimpleNamespace

os.environ.setdefault("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
os.environ.setdefault("UNSLOTH_GRPO_SKIP_ZERO_ADVANTAGE", "0")

import pytest
import torch

import unsloth_zoo.rl_replacements as rl
import unsloth_zoo.fused_losses.fused_grpo as fg

VOCAB, HIDDEN, KEEP = 24, 6, 4
EOS = 2          # pad == eos, also every chat turn end
DISTINCT_PAD = 0


class _CausalModel(torch.nn.Module):
    """Hidden state = tanh(running mean of attended embeddings); honours attention_mask and packing."""

    def __init__(self, seed = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = torch.nn.Embedding(VOCAB, HIDDEN)
        self.head = torch.nn.Linear(HIDDEN, VOCAB, bias = False)
        self.device = torch.device("cpu")
        self.seen = []

    def get_output_embeddings(self):
        return self.head

    @staticmethod
    def _run(e, keep):
        e = e * keep.unsqueeze(-1)
        return torch.tanh(e.cumsum(1) / keep.cumsum(1).clamp(min = 1).unsqueeze(-1))

    def forward(self, input_ids = None, attention_mask = None, packed_seq_lengths = None,
                logits_to_keep = None, **kwargs):
        e = self.emb(input_ids)
        if packed_seq_lengths is not None:
            sizes = [int(n) for n in packed_seq_lengths]
            self.seen.append([p.tolist() for p in torch.split(input_ids[0], sizes)])
            h = torch.cat([self._run(p[None], torch.ones(1, p.shape[0]))[0]
                           for p in torch.split(e[0], sizes)])[None]
        else:
            keep = torch.ones(input_ids.shape) if attention_mask is None else attention_mask.float()
            self.seen.append([r[k.bool()].tolist() for r, k in zip(input_ids, keep)])
            h = self._run(e, keep)
        if logits_to_keep:
            h = h[:, -logits_to_keep:]
        return SimpleNamespace(logits = h)


def _batch(pad):
    # Prompts left padded; EOS inside the prompt is a real chat turn end (TRL's prompt_mask = 1).
    prompts = [[pad, pad, 5, EOS, 6, 7], [8, EOS, 9, EOS, 10, 11], [pad, 12, 13, 14, EOS, 15]]
    prompt_mask = [[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]]
    comps = [[16, 17, EOS, pad], [18, 19, 20, 21], [22, EOS, pad, pad]]
    ids = torch.tensor([p + c for p, c in zip(prompts, comps)])
    cmask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]])
    if pad != EOS:
        # pad != eos: prompt EOS keeps its id, and only real pads are masked.
        prompt_mask = [[int(t != pad) for t in p] for p in prompts]
    att = torch.cat([torch.tensor(prompt_mask), cmask], 1)
    return ids, att, cmask


def _trainer(pad, model, beta = 0.0):
    return SimpleNamespace(
        args = SimpleNamespace(unsloth_grpo_mini_batch = 1, unsloth_logit_chunk_multiplier = 1),
        processing_class = SimpleNamespace(pad_token_id = pad, eos_token_id = EOS),
        model = model,
        accelerator = SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None),
        use_vllm = False, _autocast_dtype = None, beta = beta,
    )


def _loss_inputs(trainer, ids, att, cmask, old = None, ref = None, **kwargs):
    seen = {}
    real_forward = rl.UnslothEfficientGRPO.forward

    class _Capture:
        @staticmethod
        def apply(*args):
            bound = inspect.signature(real_forward).bind(None, *args).arguments
            seen.update(
                new = bound["_new_logps"].detach().clone(), mask = bound["_mask"].clone(),
                ids = bound["_input_ids"].clone(),
            )
            z = torch.zeros(())
            return z, z, z, z, z, z

    real, fwd_grad = rl.UnslothEfficientGRPO, fg.GRPO_FWD_GRAD
    rl.UnslothEfficientGRPO, fg.GRPO_FWD_GRAD = _Capture, False
    try:
        rl.grpo_accumulated_loss(trainer, ids, att, KEEP, cmask, torch.zeros(ids.shape[0]), old, ref,
                                 loss_type = "dapo", num_items_in_batch = int(cmask.sum()), **kwargs)
    finally:
        rl.UnslothEfficientGRPO, fg.GRPO_FWD_GRAD = real, fwd_grad
    return seen


def _intended(ids, att, pad, cmask = None):
    # TRL's prompt tokens + the completion: TRL's completion_mask when given (EOS kept), else the id rule.
    out = []
    for r in range(ids.shape[0]):
        prompt = ids[r, :-KEEP][att[r, :-KEEP].bool()].tolist()
        if cmask is None:
            out.append(prompt + [t for t in ids[r, -KEEP:].tolist() if t != pad])
        else:
            out.append(prompt + ids[r, -KEEP:][cmask[r].bool()].tolist())
    return out


def _reference_logps(model, rows):
    """Unpacked per-row logprob of each completion token, right-aligned per row."""
    out = []
    with torch.no_grad():
        for row in rows:
            x = torch.tensor([row])
            h = model._run(model.emb(x), torch.ones(1, len(row)))
            lp = torch.log_softmax(h @ model.head.weight.t(), -1)[0]
            out.append([float(lp[j - 1, row[j]]) for j in range(1, len(row))])
    return out


def _at_mask(t, mask):
    return [t[r][mask[r].bool()].tolist() for r in range(t.shape[0])]


@pytest.mark.parametrize("packing", [True, False])
def test_prompt_eos_is_seen_and_scored(packing, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "1" if packing else "0")
    ids, att, cmask = _batch(EOS)
    model = _CausalModel()
    seen = _loss_inputs(_trainer(EOS, model), ids, att, cmask)
    # <|im_end|>-style id: pad, every prompt turn end, and the completion EOS (rows 0 and 2).
    intended = _intended(ids, att, EOS, cmask)
    assert model.seen[0] == intended, "the policy forward did not see TRL's prompt + completion"
    # The loss mask is TRL's completion_mask: same tokens, EOS included.
    assert _at_mask(seen["ids"], seen["mask"]) == _at_mask(ids[:, -KEEP:], cmask)
    # Completion logps equal an unpacked per-row forward over the intended tokens.
    ref = _reference_logps(model, intended)
    got = _at_mask(seen["new"], seen["mask"])
    for r, row in enumerate(got):
        assert len(row) == cmask[r].sum()
        assert torch.allclose(torch.tensor(row), torch.tensor(ref[r][-len(row):]), atol = 1e-5), (r, row, ref[r])


def test_prompt_real_mask_is_none_without_a_real_pad_id_prompt_token():
    ids, att, _ = _batch(DISTINCT_PAD)
    assert rl.grpo_prompt_real_mask(ids, att, KEEP, DISTINCT_PAD) is None
    ids, att, _ = _batch(EOS)
    got = rl.grpo_prompt_real_mask(ids, att, KEEP, EOS)
    assert got is not None
    assert torch.equal(got[:, :-KEEP], att[:, :-KEEP].bool())
    assert torch.equal(got[:, -KEEP:], ids[:, -KEEP:] != EOS)
    assert rl.grpo_prompt_real_mask(ids, None, KEEP, EOS) is None
    # Only when the pad id is an EOS id: an uninformative all-ones mask over real pads changes nothing.
    assert rl.grpo_prompt_real_mask(ids, att, KEEP, EOS, eos_token_ids = {EOS}) is not None
    assert rl.grpo_prompt_real_mask(ids, att, KEEP, EOS, eos_token_ids = {7}) is None
    ids, _att, _ = _batch(DISTINCT_PAD)
    assert rl.grpo_prompt_real_mask(ids, torch.ones_like(ids), KEEP, DISTINCT_PAD, eos_token_ids = {EOS}) is None


def _pad_is_not_eos_batches():
    gen = torch.Generator().manual_seed(0)
    for _ in range(10):
        rows = 4
        ids = torch.randint(3, VOCAB, (rows, 6 + KEEP), generator = gen)
        att = torch.ones_like(ids)
        for r in range(rows):
            lp = int(torch.randint(0, 4, (1,), generator = gen))
            ids[r, :lp] = DISTINCT_PAD
            att[r, :lp] = 0
            cl = int(torch.randint(1, KEEP + 1, (1,), generator = gen))
            ids[r, 6 + cl:] = DISTINCT_PAD
            att[r, 6 + cl:] = 0
        yield ids, att, att[:, -KEEP:].clone()


@pytest.mark.parametrize("packing", [True, False])
def test_pad_is_not_eos_is_bitwise_unchanged(packing, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "1" if packing else "0")
    for ids, att, cmask in list(_pad_is_not_eos_batches()) + [_batch(DISTINCT_PAD)]:
        head = _loss_inputs(_trainer(DISTINCT_PAD, _CausalModel()), ids, att, cmask)
        with monkeypatch.context() as m:
            m.setattr(rl, "grpo_prompt_real_mask", lambda *a, **k: None, raising = False)
            base = _loss_inputs(_trainer(DISTINCT_PAD, _CausalModel()), ids, att, cmask)
        assert torch.equal(head["new"], base["new"]) and torch.equal(head["mask"], base["mask"])


def test_old_or_ref_logps_without_the_promise_keep_the_id_layout(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "1")
    ids, att, cmask = _batch(EOS)
    lp = rl.calculate_pad_tokens_in_prompt(ids, KEEP, EOS)
    width = KEEP + int(lp.max())
    old = torch.zeros(ids.shape[0], width)
    model = _CausalModel()
    got = _loss_inputs(_trainer(EOS, model, beta = 0.04), ids, att, cmask, old = old, ref = old.clone())
    with monkeypatch.context() as m:
        m.setattr(rl, "grpo_prompt_real_mask", lambda *a, **k: None, raising = False)
        base_model = _CausalModel()
        base = _loss_inputs(_trainer(EOS, base_model, beta = 0.04), ids, att, cmask, old = old, ref = old.clone())
    assert torch.equal(got["new"], base["new"]) and torch.equal(got["mask"], base["mask"])
    assert model.seen[0] == base_model.seen[0]

    # With the scorer's promise the prompt layout is used with old / ref logps too.
    real = rl.grpo_prompt_real_mask(ids, att, KEEP, EOS)
    width = KEEP + int((~real[:, :-KEEP]).sum(1).max())
    old = torch.zeros(ids.shape[0], width)
    trainer = _trainer(EOS, _CausalModel(), beta = 0.04)
    trainer._unsloth_grpo_logps_prompt_from_mask = True
    _loss_inputs(trainer, ids, att, cmask, old = old, ref = old.clone())
    assert trainer.model.seen[0] == _intended(ids, att, EOS)


def test_stored_old_logps_keep_the_prompt_layout_on_later_iterations(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "1")
    ids, att, cmask = _batch(EOS)
    trainer = _trainer(EOS, _CausalModel())
    _loss_inputs(trainer, ids, att, cmask, old_logps_from_new = True, store_old_logps = True)
    stored = trainer._unsloth_grpo_stored_old_logps
    assert stored._unsloth_grpo_prompt_from_mask is True
    assert stored._unsloth_grpo_covers_eos is True
    assert not getattr(trainer, "_unsloth_grpo_logps_prompt_from_mask", False)
    # The next num_iterations pass gets them back as old_logps and keeps the same layout.
    trainer.model = _CausalModel()
    again = _loss_inputs(trainer, ids, att, cmask, old = stored)
    assert trainer.model.seen[0] == _intended(ids, att, EOS, cmask)
    assert _at_mask(again["ids"], again["mask"]) == _at_mask(ids[:, -KEEP:], cmask)
    assert torch.equal(again["new"], stored)
    # An unmarked copy (external old logps) keeps the previous layout.
    trainer.model = _CausalModel()
    _loss_inputs(trainer, ids, att, cmask, old = stored.clone())
    assert trainer.model.seen[0] != _intended(ids, att, EOS)

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

"""vLLM's processed_logprobs vs the trainer's full-vocabulary logprobs under top_k / top_p / min_p.

The reference below follows vLLM v1 (0.11.2 .. 0.31): temperature, then the min_p logits
processor, then apply_top_k_top_p (sort ascending, top-k mask, cumsum <= 1 - p mask, the
largest token always kept), then log_softmax of the masked logits.
"""

import inspect
import logging
import types

import pytest
import torch

import unsloth_zoo.rl_replacements as rr


def _vllm_processed_logprobs(logits, temperature, top_k, top_p, min_p):
    logits = logits.float() / temperature
    if min_p > 0.0:
        probs = torch.softmax(logits, dim = -1)
        logits = logits.masked_fill(probs < min_p * probs.amax(dim = -1, keepdim = True), float("-inf"))
    logits_sort, logits_idx = logits.sort(dim = -1, descending = False)
    if top_k > 0:
        kth = logits_sort.gather(1, torch.full((logits.shape[0], 1), logits.shape[-1] - top_k))
        logits_sort = logits_sort.masked_fill(logits_sort < kth, float("-inf"))
    if top_p < 1.0:
        probs_sum = torch.cumsum(logits_sort.softmax(dim = -1), dim = -1)
        mask = probs_sum <= 1 - top_p
        mask[:, -1] = False
        logits_sort = logits_sort.masked_fill(mask, float("-inf"))
    logits = logits_sort.scatter(dim = -1, index = logits_idx, src = logits_sort)
    return torch.log_softmax(logits, dim = -1)


def _sample_kept(processed, generator):
    return torch.multinomial(processed.exp(), 1, generator = generator).squeeze(-1)


CASES = [
    (0.9, 0, 0.0),
    (1.0, 20, 0.0),
    (1.0, 0, 0.05),
    (0.95, 50, 0.02),
]


@pytest.mark.parametrize("top_p, top_k, min_p", CASES)
def test_trainer_logps_are_biased_by_kept_mass(top_p, top_k, min_p):
    g = torch.Generator().manual_seed(0)
    T, V, temperature = 64, 512, 0.8
    logits = torch.randn(T, V, generator = g) * 4
    processed = _vllm_processed_logprobs(logits, temperature, top_k, top_p, min_p)
    ids = _sample_kept(processed, g)
    vllm_logps = processed.gather(1, ids[:, None]).squeeze(1)
    trainer_logps = rr.chunked_selective_log_softmax(logits[None], ids[None], temperature, 4)[0]

    full = torch.log_softmax(logits / temperature, dim = -1)
    kept_mass = torch.logsumexp(full.masked_fill(~torch.isfinite(processed), float("-inf")), dim = -1)
    # vLLM's logp exceeds the trainer's by exactly -log(kept mass) on every sampled token.
    torch.testing.assert_close(vllm_logps - trainer_logps, -kept_mass, atol = 1e-4, rtol = 0)
    assert (vllm_logps - trainer_logps).mean() > 0.01


@pytest.mark.parametrize("top_p, top_k, min_p", CASES)
def test_support_scoring_matches_vllm_processed_logprobs(top_p, top_k, min_p):
    g = torch.Generator().manual_seed(1)
    B, L, H, V, temperature = 2, 16, 32, 384, 0.7
    hidden = torch.randn(B, L, H, generator = g)
    lm_head = torch.randn(V, H, generator = g) * 0.5
    logits = hidden @ lm_head.t()
    processed = _vllm_processed_logprobs(logits.reshape(-1, V), temperature, top_k, top_p, min_p)
    ids = _sample_kept(processed, g).reshape(B, L)
    expected = processed.gather(1, ids.reshape(-1, 1)).reshape(B, L)

    kwargs = dict(support_top_k = top_k, support_top_p = top_p, support_min_p = min_p)
    got_logits = rr.chunked_selective_log_softmax(logits, ids, temperature, 3, **kwargs)
    got_hidden = rr.chunked_hidden_states_selective_log_softmax(
        hidden, lm_head, ids, 3, temperature = temperature, **kwargs,
    )
    torch.testing.assert_close(got_logits, expected, atol = 1e-4, rtol = 1e-4)
    torch.testing.assert_close(got_hidden, expected, atol = 1e-4, rtol = 1e-4)


def test_token_outside_recomputed_set_stays_finite_and_grads_flow():
    g = torch.Generator().manual_seed(2)
    hidden = torch.randn(1, 8, 16, generator = g, requires_grad = True)
    lm_head = torch.randn(64, 16, generator = g)
    logits = hidden.detach() @ lm_head.t()
    ids = logits.argmin(dim = -1)  # never inside a top_k = 4 set
    out = rr.chunked_hidden_states_selective_log_softmax(
        hidden, lm_head, ids, 2, support_top_k = 4,
    )
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(hidden.grad).all() and hidden.grad.abs().sum() > 0


def test_defaults_are_full_vocabulary():
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(2, 5, 100, generator = g)
    ids = torch.randint(0, 100, (2, 5), generator = g)
    expected = torch.log_softmax(logits / 0.9, dim = -1).gather(-1, ids[..., None]).squeeze(-1)
    torch.testing.assert_close(rr.chunked_selective_log_softmax(logits, ids, 0.9, 2), expected)


def _trainer(use_vllm = True, is_correction = True, off_policy = None, **args):
    base = dict(top_k = 0, top_p = 1.0, min_p = None, generation_kwargs = None,
                off_policy_mask_threshold = off_policy)
    base.update(args)
    return types.SimpleNamespace(
        use_vllm = use_vllm,
        vllm_importance_sampling_correction = is_correction,
        args = types.SimpleNamespace(**base),
    )


def test_sampling_support_resolution():
    assert rr.grpo_sampling_support(_trainer()) is None
    assert rr.grpo_sampling_support(_trainer(top_k = -1, top_p = 1.0, min_p = 0.0)) is None
    assert rr.grpo_sampling_support(_trainer(top_k = None)) is None
    assert rr.grpo_sampling_support(_trainer(top_p = 0.9)) == (0, 0.9, 0.0)
    assert rr.grpo_sampling_support(_trainer(generation_kwargs = {"top_k": 20})) == (20, 1.0, 0.0)
    t = _trainer(top_p = 0.9)
    t.args.vllm_sampling_params = types.SimpleNamespace(_set_kwargs = {"top_p": 1.0, "min_p": 0.1})
    assert rr.grpo_sampling_support(t) == (0, 1.0, 0.1)


def test_warns_once_only_when_vllm_logps_are_consumed(monkeypatch, caplog):
    monkeypatch.delenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", raising = False)
    caplog.set_level(logging.WARNING)
    rr.logger.propagate = True

    quiet = _trainer(top_p = 0.9, is_correction = False)
    assert rr.grpo_truncated_support_kwargs(quiet) == {}
    assert "UNSLOTH_GRPO_TRUNCATED_SUPPORT" not in caplog.text

    for t in (_trainer(top_p = 0.9), _trainer(top_k = 20, is_correction = False, off_policy = 0.5)):
        caplog.clear()
        assert rr.grpo_truncated_support_kwargs(t) == {}
        assert rr.grpo_truncated_support_kwargs(t) == {}
        assert caplog.text.count("UNSLOTH_GRPO_TRUNCATED_SUPPORT=1") == 1

    caplog.clear()
    assert rr.grpo_truncated_support_kwargs(_trainer(use_vllm = False, top_p = 0.9)) == {}
    assert rr.grpo_truncated_support_kwargs(_trainer()) == {}
    assert caplog.text == ""


def test_opt_in_returns_support_kwargs(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", "1")
    assert rr.grpo_truncated_support_kwargs(_trainer(top_p = 0.9, top_k = 20, min_p = 0.05)) == {
        "support_top_k": 20, "support_top_p": 0.9, "support_min_p": 0.05,
    }
    assert rr.grpo_truncated_support_kwargs(_trainer()) == {}


def test_loss_binds_the_raw_logits_helper_locally_so_the_support_partial_reaches_it():
    src = inspect.getsource(rr.grpo_accumulated_loss)
    assert "grpo_truncated_support_kwargs(trainer)" in src
    assert "_unsloth_truncated_support_scored" in src
    imports = src.split("vision_inputs = _grpo_get_vision_inputs(kwargs)", 1)[0]
    assert "chunked_selective_log_softmax," in imports

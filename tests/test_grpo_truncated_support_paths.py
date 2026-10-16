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

"""UNSLOTH_GRPO_TRUNCATED_SUPPORT=1 across the paths around the scoring helpers: a tied top-p
cutoff, the logprob recompute helper, entropy scoring, and raw vllm.SamplingParams overrides."""

import inspect
import os
import sys
import types

import pytest
import torch

import unsloth_zoo.rl_replacements as rr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_grpo_truncated_support import _vllm_processed_logprobs  # noqa: E402


@pytest.mark.parametrize("top_p", [0.9, 0.1])
def test_tied_top_p_cutoff_matches_vllm_kept_count(top_p):
    # vLLM masks sorted positions, so a cutoff inside a run of equal logits keeps only part of it.
    logits = torch.zeros(1, 100)
    processed = _vllm_processed_logprobs(logits, 1.0, 0, top_p, 0.0)
    kept = int(torch.isfinite(processed).sum())
    assert kept == round(100 * top_p)
    assert int(rr._truncated_support_mask(logits, 0, top_p, 0.0).sum()) == kept
    # Every token is tied with the kept ones, so any of them scores as a kept token.
    for token in (0, 50, 99):
        got = rr._truncated_support_logps(logits, torch.tensor([token]), 0, top_p, 0.0)
        torch.testing.assert_close(got, torch.tensor([-float(torch.log(torch.tensor(float(kept))))]))


def test_bf16_rounded_logits_match_vllm():
    g = torch.Generator().manual_seed(0)
    logits = (torch.randn(16, 4096, generator = g) * 3).to(torch.bfloat16).float()
    processed = _vllm_processed_logprobs(logits, 1.0, 0, 0.9, 0.0)
    assert torch.equal(rr._truncated_support_mask(logits, 0, 0.9, 0.0).sum(-1), torch.isfinite(processed).sum(-1))
    ids = torch.multinomial(processed.exp(), 1, generator = g).squeeze(-1)
    want = processed.gather(1, ids[:, None]).squeeze(1)
    got = rr._truncated_support_logps(logits, ids, 0, 0.9, 0.0)
    torch.testing.assert_close(got, want, atol = 1e-5, rtol = 0)


HIDDEN, VOCAB = 4, 8


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(4)
        self.emb = torch.nn.Embedding(VOCAB, HIDDEN)
        self.head = torch.nn.Linear(HIDDEN, VOCAB, bias = False)
        self.device = torch.device("cpu")

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids = None, logits_to_keep = None, **kwargs):
        hidden = self.emb(input_ids)
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        return types.SimpleNamespace(logits = hidden)


def _new_logps(monkeypatch, support, recompute = False, entropy = False):
    """The new logprobs grpo_accumulated_loss hands to the loss for a top_k = 2 vLLM rollout."""
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", "1" if support else "0")
    monkeypatch.setenv("UNSLOTH_GRPO_LOGPROB_RECOMPUTE", "1" if recompute else "0")
    monkeypatch.setattr(rr.chunked_hidden_states_selective_log_softmax_recompute, "_use_compiled", False, raising = False)
    seen = {}

    class _Loss:
        @staticmethod
        def apply(*args):
            seen["new"] = inspect.signature(_REAL_FORWARD).bind(None, *args).arguments["_new_logps"].detach().clone()
            zero = torch.zeros(())
            return zero, zero, zero, zero, zero, zero

    monkeypatch.setattr(rr, "UnslothEfficientGRPO", _Loss)
    args = types.SimpleNamespace(
        unsloth_grpo_mini_batch = 1, unsloth_logit_chunk_multiplier = 1,
        top_k = 2, top_p = 1.0, min_p = 0.0, generation_kwargs = None,
    )
    trainer = types.SimpleNamespace(
        args = args, model = _Model(), processing_class = types.SimpleNamespace(pad_token_id = 0),
        accelerator = types.SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None),
        use_vllm = True, beta = 0.0, vllm_importance_sampling_correction = False, _autocast_dtype = None,
    )
    if entropy:
        args.top_entropy_quantile = 0.5
        trainer.get_high_entropy_mask = lambda ent, mask, q: torch.ones_like(mask, dtype = torch.bool)
    ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 6, 7, 3]])
    rr.grpo_accumulated_loss(
        trainer, ids, torch.ones_like(ids), 3, torch.ones(2, 3, dtype = torch.long),
        torch.ones(2), None, None, loss_type = "grpo", num_items_in_batch = 6,
    )
    return seen["new"]


_REAL_FORWARD = rr.UnslothEfficientGRPO.forward


def test_recompute_helper_takes_the_truncated_support(monkeypatch):
    truncated = _new_logps(monkeypatch, support = True)
    full = _new_logps(monkeypatch, support = False)
    assert not torch.allclose(truncated, full)
    torch.testing.assert_close(_new_logps(monkeypatch, support = True, recompute = True), truncated)
    # Off, the recompute helper is untouched.
    torch.testing.assert_close(_new_logps(monkeypatch, support = False, recompute = True), full)


def test_entropy_scoring_keeps_the_truncated_support(monkeypatch):
    truncated = _new_logps(monkeypatch, support = True)
    full = _new_logps(monkeypatch, support = False)
    torch.testing.assert_close(_new_logps(monkeypatch, support = True, entropy = True), truncated)
    torch.testing.assert_close(_new_logps(monkeypatch, support = False, entropy = True), full)


class _RawSamplingParams:
    """vllm.SamplingParams built directly, so it has no _set_kwargs (explicit keyword signature,
    as grpo_update_SamplingParams reads it)."""

    def __init__(
        self, n = 1, temperature = 1.0, top_p = 1.0, top_k = 0, min_p = 0.0, repetition_penalty = 1.0,
        presence_penalty = 0.0, frequency_penalty = 0.0, min_tokens = 0, logit_bias = None,
        bad_words = None, allowed_token_ids = None, logits_processors = None, seed = None, stop = None,
    ):
        for key, value in dict(locals()).items():
            if key != "self":
                setattr(self, key, value)


# unsloth >= #11595 grpo_update_SamplingParams (unsloth/models/rl_replacements.py), verbatim; the
# compiled trainer module defines it, so the trainer class's module carries it.
_UNSLOTH_UPDATE = '''
import inspect
def grpo_update_SamplingParams(
    SamplingParams,
    generation_kwargs,
    vllm_sampling_params = None,
):
    good_sampling_params_keys = inspect.signature(SamplingParams).parameters.keys()

    new_generation_kwargs = {}
    for key in generation_kwargs.keys():
        if key in good_sampling_params_keys:
            new_generation_kwargs[key] = generation_kwargs[key]
    generation_kwargs = new_generation_kwargs

    if vllm_sampling_params is not None:
        overwrites = getattr(vllm_sampling_params, "_set_kwargs", None)
        if overwrites is None:
            default_sampling_params = SamplingParams()
            overwrites = {}
            for key in good_sampling_params_keys:
                if key.startswith("_") or not hasattr(vllm_sampling_params, key):
                    continue
                overwrited_key = getattr(vllm_sampling_params, key)
                if overwrited_key != getattr(default_sampling_params, key, None):
                    overwrites[key] = overwrited_key
        for key, overwrited_key in overwrites.items():
            if key in good_sampling_params_keys and key not in (
                "seed",
                "n",
                "temperature",
                "max_tokens",
                "logprobs",
            ):
                generation_kwargs[key] = overwrited_key
    return generation_kwargs

class UnslothGRPOTrainer:
    pass
'''


def _compiled_trainer_class(monkeypatch):
    module = types.ModuleType("unsloth_compiled_cache_test.UnslothGRPOTrainer")
    exec(_UNSLOTH_UPDATE, module.__dict__)
    module.UnslothGRPOTrainer.__module__ = module.__name__
    monkeypatch.setitem(__import__("sys").modules, module.__name__, module)
    return module.UnslothGRPOTrainer


def _raw_trainer(trainer_class = None, **raw):
    trainer = (trainer_class or types.SimpleNamespace)()
    trainer.use_vllm = True
    trainer.temperature = 1.0
    trainer.vllm_importance_sampling_correction = False
    trainer.args = types.SimpleNamespace(
        top_k = 0, top_p = 0.9, min_p = 0.0, temperature = 1.0, repetition_penalty = 1.0,
        generation_kwargs = None, vllm_sampling_params = _RawSamplingParams(**raw),
    )
    return trainer


def test_raw_sampling_params_follow_unsloth_generation(monkeypatch):
    # Current unsloth applies the raw fields that differ from SamplingParams(), so scoring does too.
    monkeypatch.setenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", "1")
    klass = _compiled_trainer_class(monkeypatch)
    assert rr.grpo_truncated_support_kwargs(_raw_trainer(klass, top_k = 20)) == {
        "support_top_k": 20, "support_top_p": 0.9, "support_min_p": 0.0,
    }
    assert rr.grpo_truncated_support_kwargs(_raw_trainer(klass, top_p = 0.8)) == {
        "support_top_k": 0, "support_top_p": 0.8, "support_min_p": 0.0,
    }
    # Fields left at the SamplingParams default do not override the config.
    assert rr.grpo_truncated_support_kwargs(_raw_trainer(klass, seed = 3)) == {
        "support_top_k": 0, "support_top_p": 0.9, "support_min_p": 0.0,
    }
    assert rr.grpo_truncated_support_blockers(_raw_trainer(klass, repetition_penalty = 1.1)) == ["repetition_penalty=1.1"]
    assert rr.grpo_truncated_support_kwargs(_raw_trainer(klass, repetition_penalty = 1.1)) == {}


def test_raw_sampling_params_follow_zoo_generation_helper(monkeypatch):
    # Without the unsloth copy (unsloth before #11595), generation goes through this module's
    # grpo_update_SamplingParams, which only copies non-empty list / tuple fields: raw scalars never
    # reach the rollout, so scoring must not use them either.
    monkeypatch.setenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", "1")
    applied = rr.grpo_update_SamplingParams(_RawSamplingParams, {}, _RawSamplingParams(top_k = 20, top_p = 0.8, stop = ["x"]))
    assert applied == {"stop": ["x"]}
    assert rr.grpo_truncated_support_kwargs(_raw_trainer(top_k = 20, top_p = 0.8)) == {
        "support_top_k": 0, "support_top_p": 0.9, "support_min_p": 0.0,
    }
    assert rr.grpo_truncated_support_blockers(_raw_trainer(repetition_penalty = 1.1)) == []


def test_boundary_tied_sampled_token_is_in_the_denominator():
    # 100 equal logits at top_p = 0.9 keep 90 of them. A sampled token the sort dropped must take one
    # kept token's place: logprob -log(90) and d/dlogit = 1 - 1/90, as for a kept token.
    logits = torch.zeros(1, 100)
    keep = rr._truncated_support_mask(logits, 0, 0.9, 0.0)[0]
    for token in (int((~keep).nonzero()[0]), int(keep.nonzero()[0])):
        x = logits.clone().requires_grad_(True)
        got = rr._truncated_support_logps(x, torch.tensor([token]), 0, 0.9, 0.0)
        torch.testing.assert_close(got, torch.tensor([-float(torch.log(torch.tensor(90.0)))]))
        (grad,) = torch.autograd.grad(got.sum(), [x])
        torch.testing.assert_close(grad[0, token], torch.tensor(1.0 - 1.0 / 90))
        assert int((grad[0] != 0).sum()) == 90
        torch.testing.assert_close(grad.sum(), torch.tensor(0.0), atol = 1e-6, rtol = 0)
    # A token strictly below the kept set is still added on top of it (91 terms).
    x = torch.cat([torch.zeros(1, 100), torch.full((1, 1), -30.0)], dim = 1).requires_grad_(True)
    got = rr._truncated_support_logps(x, torch.tensor([100]), 0, 0.9, 0.0)
    want = -30.0 - torch.logaddexp(torch.log(torch.tensor(90.0)), torch.tensor(-30.0))
    torch.testing.assert_close(got, want.reshape(1))


def test_raw_sampling_params_temperature_is_not_a_false_blocker(monkeypatch):
    # Neither generation helper applies a raw SamplingParams temperature (unsloth's skips it, this
    # module's copies list fields only): the rollout samples at the trainer temperature, so a raw
    # temperature must not block truncated scoring.
    monkeypatch.setenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", "1")
    want = {"support_top_k": 0, "support_top_p": 0.9, "support_min_p": 0.0}
    klass = _compiled_trainer_class(monkeypatch)
    for trainer in (_raw_trainer(klass, temperature = 0.7), _raw_trainer(temperature = 0.7)):
        assert rr.grpo_truncated_support_blockers(trainer) == []
        assert rr.grpo_truncated_support_kwargs(trainer) == want

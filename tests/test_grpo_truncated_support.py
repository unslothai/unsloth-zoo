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
largest token always kept; vLLM's own function, copied verbatim), then log_softmax of the masked logits.
"""

import inspect
import logging
import os
import sys
import types

import pytest
import torch

import unsloth_zoo.rl_replacements as rr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _vllm_topk_topp_reference import apply_top_k_top_p_pytorch  # noqa: E402


def _vllm_processed_logprobs(logits, temperature, top_k, top_p, min_p):
    logits = logits.float() / temperature
    if min_p > 0.0:
        probs = torch.softmax(logits, dim = -1)
        logits = logits.masked_fill(probs < min_p * probs.amax(dim = -1, keepdim = True), float("-inf"))
    rows = logits.shape[0]
    # vLLM's own apply_top_k_top_p_pytorch (verbatim copy), with k / p as the int32 / float32 tensors
    # vLLM's input batch holds: the top-p cutoff `1 - p` is a float32 subtraction.
    k = torch.full((rows,), top_k, dtype = torch.int32) if top_k > 0 else None
    p = torch.full((rows,), top_p, dtype = torch.float32) if top_p < 1.0 else None
    logits = apply_top_k_top_p_pytorch(logits, k, p)
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


def test_excluded_ids_alone_drop_padded_vocab_rows():
    # Padded vocab rows the engine never samples: the scorer must drop them even without top_k / top_p / min_p.
    g = torch.Generator().manual_seed(4)
    hidden = torch.randn(1, 8, 32, generator = g)
    lm_head = torch.randn(128, 32, generator = g) * 0.3
    ids = torch.randint(0, 64, (1, 8), generator = g)
    logits = (hidden @ lm_head.t()).double()
    expected = torch.log_softmax(logits[..., :64], dim = -1).gather(-1, ids[..., None]).squeeze(-1)
    got = rr.chunked_hidden_states_selective_log_softmax(
        hidden, lm_head, ids, 4, support_excluded_ids = torch.arange(64, 128),
    )
    torch.testing.assert_close(got.double(), expected, atol = 1e-5, rtol = 0)


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


def _trainer(use_vllm = True, is_correction = True, off_policy = None, logprobs_mode = "processed_logprobs", **args):
    base = dict(top_k = 0, top_p = 1.0, min_p = None, generation_kwargs = None,
                off_policy_mask_threshold = off_policy)
    base.update(args)
    trainer = types.SimpleNamespace(
        use_vllm = use_vllm,
        vllm_importance_sampling_correction = is_correction,
        args = types.SimpleNamespace(**base),
    )
    if logprobs_mode is not None:
        # The colocated engine as vLLM v1 exposes it: llm.llm_engine.model_config.logprobs_mode.
        trainer.llm = types.SimpleNamespace(llm_engine = types.SimpleNamespace(
            model_config = types.SimpleNamespace(logprobs_mode = logprobs_mode)))
    return trainer


def test_sampling_support_resolution():
    assert rr.grpo_sampling_support(_trainer()) is None
    assert rr.grpo_sampling_support(_trainer(top_k = -1, top_p = 1.0, min_p = 0.0)) is None
    assert rr.grpo_sampling_support(_trainer(top_k = None)) is None
    assert rr.grpo_sampling_support(_trainer(top_p = 0.9)) == (0, 0.9, 0.0)
    assert rr.grpo_sampling_support(_trainer(generation_kwargs = {"top_k": 20})) == (20, 1.0, 0.0)
    t = _trainer(top_p = 0.9)
    t.args.vllm_sampling_params = types.SimpleNamespace(_set_kwargs = {"top_p": 1.0, "min_p": 0.1})
    assert rr.grpo_sampling_support(t) == (0, 1.0, 0.1)


def _support_only(kwargs):
    # support_triton_sampler depends on the installed vLLM; the decision is the kept-set triple.
    return {k: v for k, v in kwargs.items() if k in ("support_top_k", "support_top_p", "support_min_p")}


def test_kill_switch_warns_once_only_when_vllm_logps_are_consumed(monkeypatch, caplog):
    # =0 keeps the full vocabulary everywhere, silently (the user chose it).
    monkeypatch.setenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", "0")
    caplog.set_level(logging.WARNING)
    rr.logger.propagate = True
    for t in (_trainer(top_p = 0.9, is_correction = False), _trainer(top_p = 0.9),
              _trainer(top_k = 20, is_correction = False, off_policy = 0.5)):
        assert rr.grpo_truncated_support_kwargs(t) == {}
        assert rr.grpo_truncated_support_kwargs(t) == {}
    assert caplog.text == ""


@pytest.mark.parametrize("value", [None, "", "auto"])
def test_unset_is_on_exactly_when_truncated_logps_are_consumed(monkeypatch, caplog, value):
    if value is None:
        monkeypatch.delenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", value)
    caplog.set_level(logging.WARNING)
    rr.logger.propagate = True
    # Truncated and consumed (IS correction, or the off-policy mask alone): kept-set scoring, no warning.
    for t, triple in (
        (_trainer(top_p = 0.9), (0, 0.9, 0.0)),
        (_trainer(min_p = 0.1), (0, 1.0, 0.1)),
        (_trainer(top_k = 20, is_correction = False, off_policy = 0.5), (20, 1.0, 0.0)),
        (_trainer(top_k = 20, top_p = 0.95, min_p = 0.05), (20, 0.95, 0.05)),
    ):
        got = rr.grpo_truncated_support_kwargs(t)
        assert _support_only(got) == dict(zip(("support_top_k", "support_top_p", "support_min_p"), triple))
    assert caplog.text == ""
    # vllm_config.model_config (other vLLM layouts) and an enum-valued mode are read too.
    t = _trainer(top_p = 0.9, logprobs_mode = None)
    mode = types.SimpleNamespace(value = "processed_logprobs")
    t.vllm_generation = types.SimpleNamespace(llm = types.SimpleNamespace(llm_engine = types.SimpleNamespace(
        vllm_config = types.SimpleNamespace(model_config = types.SimpleNamespace(logprobs_mode = mode)))))
    assert _support_only(rr.grpo_truncated_support_kwargs(t))["support_top_p"] == 0.9
    assert caplog.text == ""
    # Truncated but not consumed: the vLLM logprobs reach nothing, so nothing changes and nothing warns.
    assert rr.grpo_truncated_support_kwargs(_trainer(top_p = 0.9, is_correction = False)) == {}
    # Not truncated, no vLLM: nothing.
    assert rr.grpo_truncated_support_kwargs(_trainer(use_vllm = False, top_p = 0.9)) == {}
    assert rr.grpo_truncated_support_kwargs(_trainer()) == {}
    assert caplog.text == ""


def test_decision_table(monkeypatch):
    # (setting, truncated, IS correction, off-policy mask) -> kept-set scoring?
    cases = []
    for setting in (None, "0", "1"):
        for truncated in (False, True):
            for is_correction in (False, True):
                for off_policy in (None, 0.5):
                    consumed = is_correction or off_policy is not None
                    expected = truncated and (setting == "1" or (setting is None and consumed))
                    cases.append((setting, truncated, is_correction, off_policy, expected))
    for setting, truncated, is_correction, off_policy, expected in cases:
        if setting is None:
            monkeypatch.delenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", raising = False)
        else:
            monkeypatch.setenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", setting)
        t = _trainer(top_p = 0.9 if truncated else 1.0, is_correction = is_correction, off_policy = off_policy)
        assert bool(rr.grpo_truncated_support_kwargs(t)) is expected, (setting, truncated, is_correction, off_policy)


@pytest.mark.parametrize("logprobs_mode", ["raw_logprobs", "raw_logits", None])
def test_unset_stays_full_vocabulary_unless_the_engine_returns_processed_logprobs(monkeypatch, caplog, logprobs_mode):
    # raw_logprobs are full-vocabulary: a kept-set trainer score would bias the ratio the other way.
    # None = no colocated engine to read (server mode): unknown, so the automatic default stays off.
    monkeypatch.delenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", raising = False)
    caplog.set_level(logging.WARNING)
    rr.logger.propagate = True
    t = _trainer(top_p = 0.9, logprobs_mode = logprobs_mode)
    assert rr.grpo_truncated_support_kwargs(t) == {}
    assert caplog.text.count("UNSLOTH_GRPO_TRUNCATED_SUPPORT=1") == 1
    # =1 still forces it (the user vouches for the engine).
    monkeypatch.setenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", "1")
    assert _support_only(rr.grpo_truncated_support_kwargs(_trainer(top_p = 0.9, logprobs_mode = logprobs_mode)))


def test_setting_parse(monkeypatch):
    for value, setting in (("1", "on"), ("true", "on"), ("ON", "on"), ("0", "off"), ("false", "off"),
                           ("off", "off"), ("", "auto"), ("auto", "auto"), ("weird", "auto")):
        monkeypatch.setenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", value)
        assert rr.grpo_truncated_support_setting() == setting
    monkeypatch.delenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT")
    assert rr.grpo_truncated_support_setting() == "auto"


def test_automatic_blockers_fall_back_with_an_automatic_message(monkeypatch, caplog):
    monkeypatch.delenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", raising = False)
    caplog.set_level(logging.WARNING)
    rr.logger.propagate = True
    t = _trainer(top_p = 0.9, temperature = 1.0, repetition_penalty = 1.1)
    t.temperature = 1.0
    assert rr.grpo_truncated_support_kwargs(t) == {}
    assert rr.grpo_truncated_support_kwargs(t) == {}
    assert caplog.text.count("(UNSLOTH_GRPO_TRUNCATED_SUPPORT, automatic)") == 1
    assert "ignoring UNSLOTH_GRPO_TRUNCATED_SUPPORT=1" not in caplog.text


def test_opt_in_returns_support_kwargs(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", "1")
    assert rr.grpo_truncated_support_kwargs(_trainer(top_p = 0.9, top_k = 20, min_p = 0.05)) == {
        "support_top_k": 20, "support_top_p": 0.9, "support_min_p": 0.05,
    }
    assert rr.grpo_truncated_support_kwargs(_trainer()) == {}


@pytest.mark.parametrize("overrides", [
    dict(generation_kwargs = {"temperature": 0.7}),
    dict(vllm_sampling_params = types.SimpleNamespace(_set_kwargs = {"temperature": 1.3})),
    dict(repetition_penalty = 1.1),
    dict(generation_kwargs = {"presence_penalty": 0.5}),
    dict(generation_kwargs = {"logit_bias": {5: 2.0}}),
    dict(vllm_sampling_params = types.SimpleNamespace(_set_kwargs = {"bad_words": ["x"]})),
    dict(generation_kwargs = {"allowed_token_ids": [1, 2]}),
    dict(generation_kwargs = {"min_tokens": 4}),
    dict(vllm_structured_outputs_regex = "a+"),
], ids = ["gen_temperature", "vllm_temperature", "repetition_penalty", "presence_penalty",
          "logit_bias", "bad_words", "allowed_token_ids", "min_tokens", "structured_regex"])
def test_opt_in_refuses_settings_vllm_applies_before_truncation(monkeypatch, caplog, overrides):
    # vLLM v1 Sampler: apply_logits_processors (allowed ids, bad words, logit bias / min tokens,
    # penalties), then temperature, then min_p, then top_k / top_p, at the rollout's own temperature.
    monkeypatch.setenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", "1")
    caplog.set_level(logging.WARNING)
    rr.logger.propagate = True
    t = _trainer(**{"top_p": 0.9, "temperature": 1.0, "repetition_penalty": 1.0, **overrides})
    t.temperature = 1.0
    assert rr.grpo_truncated_support_kwargs(t) == {}
    assert rr.grpo_truncated_support_kwargs(t) == {}
    assert caplog.text.count("ignoring UNSLOTH_GRPO_TRUNCATED_SUPPORT=1") == 1


def test_opt_in_keeps_neutral_processor_settings(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", "1")
    t = _trainer(top_p = 0.9, temperature = 0.8, repetition_penalty = 1.0, vllm_structured_outputs_regex = None,
                 generation_kwargs = {"temperature": 0.8, "presence_penalty": 0.0, "logit_bias": {}, "min_tokens": 0})
    t.temperature = 0.8
    assert rr.grpo_truncated_support_kwargs(t) == {
        "support_top_k": 0, "support_top_p": 0.9, "support_min_p": 0.0,
    }


def test_loss_binds_the_raw_logits_helper_locally_so_the_support_partial_reaches_it():
    src = inspect.getsource(rr.grpo_accumulated_loss)
    assert "grpo_truncated_support_kwargs(trainer)" in src
    assert "_unsloth_truncated_support_scored" in src
    imports = src.split("vision_inputs = _grpo_get_vision_inputs(kwargs)", 1)[0]
    assert "chunked_selective_log_softmax," in imports


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
@pytest.mark.parametrize("sampling, triple", [
    (dict(min_p = 0.1), (0, 1.0, 0.1)),
    (dict(top_p = 0.9), (0, 0.9, 0.0)),
    (dict(top_k = 20), (20, 1.0, 0.0)),
])
def test_cuda_automatic_kwargs_score_the_kept_set(monkeypatch, sampling, triple):
    # Unset + truncated + IS correction: the kwargs the GRPO loss receives score the renormalised
    # kept set on CUDA, so identical trainer / sampler logits give an IS ratio of 1 (full
    # vocabulary would give the kept mass).
    monkeypatch.delenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", raising = False)
    monkeypatch.setenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT_SAMPLER", "pytorch")
    kwargs = rr.grpo_truncated_support_kwargs(_trainer(**sampling))
    assert _support_only(kwargs) == dict(zip(("support_top_k", "support_top_p", "support_min_p"), triple))
    g = torch.Generator().manual_seed(7)
    hidden = torch.randn(2, 16, 64, generator = g)
    head = torch.randn(4096, 64, generator = g) * 0.3
    logits = (hidden.double() @ head.double().t())
    probs = torch.softmax(logits, -1)
    ids = torch.multinomial(probs.reshape(-1, 4096).float(), 1, generator = g).reshape(2, 16)
    # fp64 kept set, vLLM order (min_p, then top_k / top_p), always keeping the scored token.
    keep = torch.ones_like(logits, dtype = torch.bool)
    top_k, top_p, min_p = triple
    if min_p > 0:
        keep &= probs >= min_p * probs.amax(-1, keepdim = True)
    if top_k > 0:
        keep &= logits >= logits.topk(top_k, -1).values[..., -1:]
    if top_p < 1:
        s, order = logits.masked_fill(~keep, float("-inf")).sort(-1)
        cum = torch.softmax(s, -1).cumsum(-1)
        drop_sorted = cum <= 1 - top_p
        drop_sorted[..., -1] = False
        keep &= ~torch.zeros_like(keep).scatter(-1, order, drop_sorted)
    keep.scatter_(-1, ids[..., None], True)
    masked = logits.masked_fill(~keep, float("-inf"))
    oracle = torch.log_softmax(masked, -1).gather(-1, ids[..., None]).squeeze(-1)
    full = torch.log_softmax(logits, -1).gather(-1, ids[..., None]).squeeze(-1)
    got = rr.chunked_hidden_states_selective_log_softmax(
        hidden.cuda(), head.cuda(), ids.cuda(), 2, **{k: v for k, v in kwargs.items()}
    ).cpu().double()
    torch.testing.assert_close(got, oracle, atol = 2e-4, rtol = 0)
    assert (oracle - full).abs().max() > 1e-3  # the kept set really differs from the full vocabulary


def test_loss_autotune_counts_the_kept_set_logits_twice():
    src = inspect.getsource(rr.grpo_accumulated_loss)
    head, tail = src.split("dtype_bytes = 16 if _autocast_on", 1)
    block = tail.split("total_rows = input_ids.shape[0]", 1)[0]
    assert "if _support_kwargs:" in block and "dtype_bytes = dtype_bytes * 2" in block
    # Free memory that fits every row at the plain estimate still fits them doubled: unchanged plan.
    rows_plain, _ = rr.autotune_batch_and_chunks(8, 512, 1024, 151936, 16, None)
    rows_kept, _ = rr.autotune_batch_and_chunks(8, 512, 1024, 151936, 32, None)
    assert rows_kept <= rows_plain

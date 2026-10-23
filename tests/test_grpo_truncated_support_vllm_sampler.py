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

"""UNSLOTH_GRPO_TRUNCATED_SUPPORT=1 against the sampler vLLM actually runs.

* PyTorch sampler (vLLM < 0.17, or no Triton): checked against vLLM's own functions, copied
  verbatim in _vllm_topk_topp_reference.py (and the installed vLLM's, when it imports).
* Triton sampler (vLLM >= 0.17 with Triton): its kernels need a GPU or TRITON_INTERPRET, so the
  expected kept sets below are the outputs of vLLM v0.32.0's apply_top_k_top_p_triton run under
  TRITON_INTERPRET=1 on CPU: exactly top_k tokens (ties broken by vocabulary index), and a row of
  equal logits is kept whole.
* vllm_sampling_params only count where Unsloth routes them into generation.
"""

import logging
import os
import sys
import types

import pytest
import torch

import unsloth_zoo.rl_replacements as rr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _vllm_topk_topp_reference import apply_top_k_top_p_pytorch, apply_top_k_top_p_v0_11_2  # noqa: E402


def _vllm_kept(fn, logits, top_k, top_p):
    rows = logits.shape[0]
    k = torch.full((rows,), top_k, dtype = torch.int32) if top_k > 0 else None
    p = torch.full((rows,), top_p, dtype = torch.float32) if top_p < 1.0 else None
    return torch.isfinite(fn(logits.clone(), k, p))


def _references():
    refs = [apply_top_k_top_p_pytorch, apply_top_k_top_p_v0_11_2]
    try:
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch as installed
        refs.append(installed)
    except Exception:
        pass
    return refs


def _tied_rows():
    rows = []
    for n in (3, 7, 10, 20, 64, 100):
        x = torch.full((1, 256), -30.0)
        x[:, :n] = 0.0
        rows.append(x)
    return torch.cat(rows)


@pytest.mark.parametrize("top_p", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
def test_top_p_cutoff_uses_vllm_float32_threshold(top_p):
    # vLLM compares the sorted cumsum against `1 - p` with p a float32 tensor. Python's 1.0 - top_p
    # rounds differently, e.g. ten tied logits at top_p = 0.8 keep 9 in vLLM but 8 with the double.
    logits = _tied_rows()
    keep = rr._truncated_support_mask(logits, 0, top_p, 0.0)
    for fn in _references():
        assert torch.equal(keep.sum(-1), _vllm_kept(fn, logits, 0, top_p).sum(-1)), fn


def test_ten_tied_top_k_top_p_matches_vllm():
    logits = torch.full((1, 1000), -30.0)
    logits[:, :10] = 0.0
    vllm = _vllm_kept(apply_top_k_top_p_pytorch, logits, 10, 0.8)
    assert int(vllm.sum()) == 9
    assert int(rr._truncated_support_mask(logits, 10, 0.8, 0.0).sum()) == 9
    got = rr._truncated_support_logps(logits, torch.tensor([0]), 10, 0.8, 0.0)
    torch.testing.assert_close(got, torch.log_softmax(logits.masked_fill(~vllm, float("-inf")), -1)[:, 0])


@pytest.mark.parametrize("top_k, top_p", [(20, 1.0), (50, 1.0), (0, 0.9), (50, 0.9), (20, 0.5)])
def test_pytorch_sampler_bf16_rows_match_vllm(top_k, top_p):
    g = torch.Generator().manual_seed(0)
    logits = (torch.randn(32, 4096, generator = g) * 3).to(torch.bfloat16).float()
    logits[0] = 0.0
    keep = rr._truncated_support_mask(logits, top_k, top_p, 0.0)
    for fn in _references():
        assert torch.equal(keep.sum(-1), _vllm_kept(fn, logits, top_k, top_p).sum(-1)), fn


def test_triton_sampler_top_k_keeps_exactly_k():
    # vLLM v0.32.0 Triton (TRITON_INTERPRET=1): 8 logits at 3 and 32 at 2, top_k = 16, keep tokens
    # 0..15 and token 0 scores -2.392703. The PyTorch sampler keeps all 40 (-2.984274).
    logits = torch.full((1, 128256), -10.0)
    logits[:, :8] = 3.0
    logits[:, 8:40] = 2.0
    keep = rr._truncated_support_mask(logits, 16, 1.0, 0.0, True)
    assert keep[0].nonzero().flatten().tolist() == list(range(16))
    assert int(rr._truncated_support_mask(logits, 16, 1.0, 0.0).sum()) == 40
    got = rr._truncated_support_logps(logits, torch.tensor([0]), 16, 1.0, 0.0, True)
    torch.testing.assert_close(got, torch.tensor([-2.392703]), atol = 1e-6, rtol = 0)
    # A sampled 2-valued token the index budget dropped scores as one of the 16 kept, not a 17th.
    for token in (8, 39):
        got = rr._truncated_support_logps(logits, torch.tensor([token]), 16, 1.0, 0.0, True)
        want = 2.0 - torch.logsumexp(torch.tensor([3.0] * 8 + [2.0] * 8), 0)
        torch.testing.assert_close(got, want.reshape(1))


def test_triton_sampler_top_k_at_the_row_minimum_keeps_only_larger_logits():
    # vLLM v0.32.0 Triton: 4 logits at 0 and 60 at -1, top_k = 8 keep only the 4 (its pivot cannot
    # drop below the row minimum); one logit at -50 under the tied run restores exactly 8.
    logits = torch.full((1, 64), -1.0)
    logits[:, :4] = 0.0
    assert int(rr._truncated_support_mask(logits, 8, 1.0, 0.0, True).sum()) == 4
    logits[0, -1] = -50.0
    assert int(rr._truncated_support_mask(logits, 8, 1.0, 0.0, True).sum()) == 8


def test_triton_sampler_skips_top_k_with_at_most_k_finite_logits():
    # vLLM v0.24 - v0.32 Triton masks top-k only when the row has more than top_k finite logits
    # (`num_finite_total > k`). 4 at 0 and 4 at -1 with the rest -inf and top_k = 8: all 8 stay,
    # though the 8th logit is the row minimum. With 5 at -1 the row-minimum rule applies again.
    logits = torch.full((1, 64), float("-inf"))
    logits[0, :4] = 0.0
    logits[0, 4:8] = -1.0
    assert int(rr._truncated_support_mask(logits, 8, 1.0, 0.0, True).sum()) == 8
    logits[0, 8] = -1.0
    assert int(rr._truncated_support_mask(logits, 8, 1.0, 0.0, True).sum()) == 4
    # The same row reached through engine-excluded ids, on the scoring helper.
    raw = torch.full((1, 64), -30.0)
    raw[0, :4] = 0.0
    raw[0, 4:8] = -1.0
    excluded = torch.arange(8, 64)
    got = rr._truncated_support_logps(raw, torch.tensor([5]), 8, 1.0, 0.0, True, excluded)
    want = -1.0 - torch.logsumexp(torch.tensor([0.0] * 4 + [-1.0] * 4), 0)
    torch.testing.assert_close(got, want.reshape(1))


def test_triton_sampler_keeps_uniform_rows_whole():
    # vLLM v0.32.0 Triton, monolithic kernel and the CUDA split pipeline alike: 100 equal logits at
    # top_p = 0.9 keep all 100 (-log 100); the PyTorch sampler keeps 90 (-log 90).
    logits = torch.zeros(1, 100)
    assert int(rr._truncated_support_mask(logits, 0, 0.9, 0.0, True).sum()) == 100
    assert int(rr._truncated_support_mask(logits, 0, 0.9, 0.0).sum()) == 90
    got = rr._truncated_support_logps(logits, torch.tensor([7]), 0, 0.9, 0.0, True)
    torch.testing.assert_close(got, torch.tensor([-float(torch.log(torch.tensor(100.0)))]))
    # Also with an active top_k (v0.32.0 Triton keeps the whole 4096-wide row at top_k = 20).
    assert int(rr._truncated_support_mask(torch.zeros(1, 4096), 20, 0.5, 0.0, True).sum()) == 4096


def test_triton_sampler_bf16_rows_keep_k_in_vocab_order():
    g = torch.Generator().manual_seed(0)
    logits = (torch.randn(24, 4096, generator = g) * 3).to(torch.bfloat16).float()
    keep = rr._truncated_support_mask(logits, 20, 1.0, 0.0, True)
    assert torch.equal(keep.sum(-1), torch.full((24,), 20))
    kth = torch.topk(logits, 20, dim = -1).values[:, -1:]
    assert bool((keep <= (logits >= kth)).all())
    # Top-p alone on non-uniform rows: the Triton kernels agree with the PyTorch count.
    pt = _vllm_kept(apply_top_k_top_p_pytorch, logits, 0, 0.9).sum(-1)
    assert torch.equal(rr._truncated_support_mask(logits, 0, 0.9, 0.0, True).sum(-1), pt)


def test_chunked_helpers_take_the_triton_flag():
    logits = torch.full((1, 2, 64), -10.0)
    logits[..., :4] = 3.0
    logits[..., 4:20] = 2.0
    ids = torch.tensor([[0, 10]])
    got = rr.chunked_selective_log_softmax(logits, ids, 1.0, 1, support_top_k = 8, support_triton_sampler = True)
    kept = torch.tensor([3.0] * 4 + [2.0] * 4)
    want = torch.stack([3.0 - torch.logsumexp(kept, 0), 2.0 - torch.logsumexp(kept, 0)])
    torch.testing.assert_close(got[0], want)


def _trainer(**args):
    base = dict(top_k = 0, top_p = 0.9, min_p = 0.0, temperature = 1.0, generation_kwargs = None)
    base.update(args)
    return types.SimpleNamespace(
        use_vllm = True, temperature = 1.0, vllm_importance_sampling_correction = True,
        args = types.SimpleNamespace(**base),
    )


@pytest.fixture
def sampler(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", "1")
    monkeypatch.delenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT_SAMPLER", raising = False)

    def use(version, has_triton = True):
        from packaging.version import Version
        monkeypatch.setattr(rr, "_vllm_sampler_version", lambda: None if version is None else Version(version), raising = False)
        monkeypatch.setattr(rr, "_vllm_has_triton", lambda: has_triton, raising = False)
    return use


@pytest.mark.parametrize("version, has_triton, triton", [
    ("0.11.2", True, False), ("0.16.0", True, False), ("0.32.0", False, False),
    ("0.24.0", True, True), ("0.30.0", True, True), ("0.32.0", True, True),
])
def test_sampler_backend_follows_installed_vllm(sampler, version, has_triton, triton):
    sampler(version, has_triton)
    kwargs = rr.grpo_truncated_support_kwargs(_trainer(top_k = 20))
    assert kwargs.get("support_triton_sampler", False) is triton
    assert kwargs["support_top_k"] == 20


def test_sampler_backend_warns_where_ties_are_not_reproducible(sampler, caplog):
    caplog.set_level(logging.WARNING)
    rr.logger.propagate = True
    # vLLM >= 0.30, top_k or top_p alone: reproduced exactly, no warning.
    sampler("0.32.0")
    for t in (_trainer(top_k = 20, top_p = 1.0), _trainer(top_p = 0.9)):
        rr.grpo_truncated_support_kwargs(t)
    assert "Triton" not in caplog.text
    # Batch-size dependent dispatch (0.17 - 0.29) and a k + p tie budget: warned once.
    for version, top_k, top_p, phrase in [
        ("0.29.0", 20, 1.0, "fewer than 8 sequences"),
        ("0.20.0", 20, 1.0, "one token more or fewer"),
        ("0.32.0", 20, 0.9, "tied at the top_p cutoff"),
    ]:
        caplog.clear()
        sampler(version)
        t = _trainer(top_k = top_k, top_p = top_p)
        rr.grpo_truncated_support_kwargs(t)
        rr.grpo_truncated_support_kwargs(t)
        assert caplog.text.count(phrase) == 1, (version, caplog.text)


def test_sampler_backend_env_override(sampler, monkeypatch):
    sampler("0.32.0")
    monkeypatch.setenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT_SAMPLER", "pytorch")
    assert "support_triton_sampler" not in rr.grpo_truncated_support_kwargs(_trainer(top_k = 20))
    sampler("0.11.2")
    monkeypatch.setenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT_SAMPLER", "triton")
    assert rr.grpo_truncated_support_kwargs(_trainer(top_k = 20))["support_triton_sampler"] is True


class _VLLMGeneration:
    def generate(self, *args, **kwargs):
        pass
    generate._unsloth_vllm_generation_lora_wrapped = True


class _UnwrappedVLLMGeneration:
    def generate(self, *args, **kwargs):
        pass


def _generation_trainer(generation_class = _VLLMGeneration, shared = False, llm = True, **args):
    trainer = _trainer(**args)
    trainer.args.vllm_sampling_params = types.SimpleNamespace(_set_kwargs = {"top_p": 0.5})
    if generation_class is not None:
        trainer.vllm_generation = generation_class()
        trainer.vllm_generation.llm = types.SimpleNamespace(shared_weights = shared) if llm else None
    return trainer


@pytest.mark.parametrize("make, rollout_top_p", [
    # TRL >= 0.28: Unsloth's generate wrapper applies vllm_sampling_params only to a shared engine.
    (lambda: _generation_trainer(shared = True), 0.5),
    (lambda: _generation_trainer(shared = False), 0.9),
    (lambda: _generation_trainer(llm = False), 0.9),
    (lambda: _generation_trainer(_UnwrappedVLLMGeneration, shared = True), 0.9),
    # Server mode samples with TRL's parameters.
    (lambda: _generation_trainer(shared = True, vllm_mode = "server"), 0.9),
    # TRL < 0.28: rl.py rewrites the trainer's own colocate SamplingParams call.
    (lambda: _generation_trainer(None), 0.5),
])
def test_vllm_sampling_params_count_only_where_generation_applies_them(sampler, make, rollout_top_p):
    sampler("0.11.2")
    assert rr.grpo_truncated_support_kwargs(make())["support_top_p"] == rollout_top_p


def _padded_engine_trainer(ids, bias = float("-inf"), **args):
    trainer = _trainer(**args)
    trainer.llm = types.SimpleNamespace(_unsloth_padded_vocab_bias = (tuple(ids), bias))
    return trainer


def test_engine_suppressed_padded_rows_leave_the_scored_support(sampler):
    # zoo#1831: 61 real tokens, 64 alignment rows the engine biases to -inf on every request, equal
    # logits. vLLM adds logit_bias before computing processed logprobs (sampler.py
    # apply_logits_processors, builtin.py LogitBiasLogitsProcessor: logits += bias), so a rollout
    # token scores -log(61); the trainer scored -log(125), a ratio of 0.488 with identical weights.
    sampler("0.11.2")
    pad = list(range(61, 125))
    logits = torch.zeros(1, 2, 125)
    ids = torch.tensor([[0, 60]])
    biased = logits.clone()
    biased[..., pad] += float("-inf")
    vllm = torch.log_softmax(biased, -1).gather(-1, ids[..., None]).squeeze(-1)
    full = rr.chunked_selective_log_softmax(logits, ids, 1.0, 1)
    torch.testing.assert_close(full, torch.full((1, 2), -float(torch.log(torch.tensor(125.0)))))

    # No top_k / top_p / min_p: the padded rows alone turn support scoring on.
    kwargs = rr.grpo_truncated_support_kwargs(_padded_engine_trainer(pad, top_p = 1.0))
    assert kwargs["support_top_k"] == 0 and kwargs["support_top_p"] == 1.0
    assert kwargs["support_excluded_ids"].tolist() == pad
    x = logits.clone().requires_grad_(True)
    got = rr.chunked_selective_log_softmax(x, ids, 1.0, 1, **kwargs)
    torch.testing.assert_close(got, vllm)
    (grad,) = torch.autograd.grad(got.sum(), [x])
    assert bool((grad[..., pad] == 0).all())
    # Together with truncation, on the hidden-states helper as well.
    kwargs = rr.grpo_truncated_support_kwargs(_padded_engine_trainer(pad, top_k = 30, top_p = 1.0))
    assert kwargs["support_top_k"] == 30 and "support_excluded_ids" in kwargs
    hidden = torch.eye(125)[None, :2]
    head = torch.zeros(125, 125)
    got = rr.chunked_hidden_states_selective_log_softmax(hidden, head, ids, 1, 1.0, 1.0, 0.0, 1.0, **kwargs)
    torch.testing.assert_close(got, torch.full((1, 2), -float(torch.log(torch.tensor(61.0)))))


def test_padded_rows_only_count_where_the_engine_suppresses_them(sampler, monkeypatch, caplog):
    sampler("0.11.2")
    pad = list(range(61, 125))
    # No record (TRL-created engine, server mode) or a finite bias: nothing is excluded.
    assert rr.grpo_truncated_support_kwargs(_trainer(top_p = 1.0)) == {}
    assert rr.grpo_truncated_support_kwargs(_padded_engine_trainer(pad, bias = -100.0, top_p = 1.0)) == {}
    # TRL >= 0.28: the record sits on vllm_generation.llm.
    trainer = _trainer(top_p = 1.0)
    trainer.vllm_generation = types.SimpleNamespace(llm = types.SimpleNamespace(_unsloth_padded_vocab_bias = (tuple(pad), float("-inf"))))
    assert rr.grpo_truncated_support_kwargs(trainer)["support_excluded_ids"].tolist() == pad
    # Opt-in off with the vLLM logprobs consumed: warned once, naming the padded rows.
    monkeypatch.delenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT")
    caplog.set_level(logging.WARNING)
    rr.logger.propagate = True
    t = _padded_engine_trainer(pad, top_p = 1.0)
    assert rr.grpo_truncated_support_kwargs(t) == {}
    assert rr.grpo_truncated_support_kwargs(t) == {}
    assert caplog.text.count("64 padded vocabulary rows suppressed") == 1

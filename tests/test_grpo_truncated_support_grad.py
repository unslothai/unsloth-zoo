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

"""Automatic kept-set scoring corrects the IS ratio only; the policy keeps the full-vocabulary gradient.

With UNSLOTH_GRPO_TRUNCATED_SUPPORT unset, truncated vLLM rollouts (top_k / top_p / min_p) whose
processed logprobs feed the IS correction are scored over the rollouts' kept set S, so the IS ratio
compares like with like (~1, not p(S)). The gradient must stay TRL's and verl's d log p(y) over the
full vocabulary: d log (p(y) / p(S)) is exactly zero whenever S is one token (top_k = 1, or any
confident token under min_p / top_p), so a negative advantage could never push such a token down.
=1 still opts into the support-conditioned objective.
"""
import types

import pytest
import torch

import unsloth_zoo.rl_replacements as rr

SUPPORTS = [dict(support_top_k = 1), dict(support_min_p = 0.1), dict(support_top_p = 0.9), dict(support_top_k = 5),
            dict(support_top_k = 20, support_top_p = 0.95, support_min_p = 0.05)]


def _trainer(logprobs_mode = "processed_logprobs", **args):
    base = dict(top_k = 0, top_p = 1.0, min_p = None, generation_kwargs = None, off_policy_mask_threshold = None)
    base.update(args)
    t = types.SimpleNamespace(use_vllm = True, vllm_importance_sampling_correction = True,
                              args = types.SimpleNamespace(**base))
    t.llm = types.SimpleNamespace(llm_engine = types.SimpleNamespace(
        model_config = types.SimpleNamespace(logprobs_mode = logprobs_mode)))
    return t


@pytest.fixture
def cuda_platform(monkeypatch):
    # The automatic default is gated to NVIDIA CUDA; the decision itself is platform independent.
    monkeypatch.setattr(rr, "_grpo_truncated_support_auto_platform", lambda: True)
    monkeypatch.setenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT_SAMPLER", "pytorch")


def test_automatic_kwargs_keep_full_vocabulary_gradient(monkeypatch, cuda_platform):
    monkeypatch.delenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", raising = False)
    assert rr.grpo_truncated_support_kwargs(_trainer(min_p = 0.1)).get("support_full_vocab_grad") is True
    # =1 is the explicit support-conditioned objective.
    monkeypatch.setenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", "1")
    assert "support_full_vocab_grad" not in rr.grpo_truncated_support_kwargs(_trainer(min_p = 0.1))


@pytest.mark.parametrize("mode, expected", [("processed_logprobs", True), ("processed_logits", False),
                                            ("raw_logprobs", False), ("raw_logits", False)])
def test_automatic_needs_normalised_processed_logprobs(monkeypatch, cuda_platform, mode, expected):
    # vLLM processed_logits are the processed logits before log_softmax, not logprobs.
    monkeypatch.delenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", raising = False)
    assert bool(rr.grpo_truncated_support_kwargs(_trainer(logprobs_mode = mode, top_p = 0.9))) is expected


def _full_vocab_oracle(logits, ids, temperature):
    x = logits.detach().double().requires_grad_(True)
    lp = torch.log_softmax(x / temperature, -1).gather(-1, ids[..., None]).squeeze(-1)
    return x, lp


@pytest.mark.parametrize("support", SUPPORTS)
@pytest.mark.parametrize("helper", ["logits", "hidden"])
def test_value_is_kept_set_and_gradient_is_full_vocabulary(support, helper):
    g = torch.Generator().manual_seed(5)
    B, L, H, V, T = 3, 6, 16, 257, 0.8
    hidden = torch.randn(B, L, H, generator = g, dtype = torch.float64)
    head = torch.randn(V, H, generator = g, dtype = torch.float64) * 0.7
    logits = hidden @ head.t()
    ids = torch.multinomial(torch.softmax(logits / T, -1).reshape(-1, V), 1, generator = g).reshape(B, L)
    ids[0, :2] = logits[0, :2].argmax(-1)  # confident picks: singleton kept sets under top_k = 1
    weights = torch.randn(B, L, generator = g, dtype = torch.float64)

    def score(full_grad):
        if helper == "logits":
            x = logits.clone().requires_grad_(True)
            out = rr.chunked_selective_log_softmax(x, ids, T, 2, support_full_vocab_grad = full_grad, **support)
        else:
            x = head.clone().requires_grad_(True)
            out = rr.chunked_hidden_states_selective_log_softmax(
                hidden, x, ids, 2, temperature = T, support_full_vocab_grad = full_grad, **support)
        (out * weights).sum().backward()
        return out.detach(), x.grad

    kept_value, kept_grad = score(False)
    value, grad = score(True)
    assert torch.equal(value, kept_value)  # bitwise the kept-set logprob the IS ratio needs
    x, lp = _full_vocab_oracle(logits, ids, T)
    (lp * weights).sum().backward()
    want = x.grad if helper == "logits" else torch.einsum("blv,blh->vh", x.grad, hidden)
    # The helpers score in float32 (as the trainer does); the oracle is float64.
    torch.testing.assert_close(grad, want, rtol = 1e-4, atol = 1e-5)
    if support.get("support_top_k") == 1 and helper == "logits":
        # The support-conditioned gradient vanishes on a singleton kept set (argmax picks under
        # top_k = 1): the defect this fixes. The full-vocabulary gradient there is not zero.
        assert kept_grad[0, :2].abs().max() == 0 and grad[0, :2].abs().max() > 1e-3


def _grpo(new, old, sampling, adv):
    mask = torch.ones_like(new)
    out = rr.grpo_compute_loss(
        None, new, old, sampling, torch.zeros_like(mask), mask, 0.0, adv,
        loss_type = "grpo", use_vllm = True, vllm_importance_sampling_correction = True,
        vllm_importance_sampling_mode = "token_truncate", vllm_importance_sampling_clip_max = 3.0,
        max_completion_length = new.shape[1], importance_sampling_level = "token",
    )
    return out[0], out[4]


@pytest.mark.parametrize("logits, support", [
    ([2.0, 1.0, 0.0], dict(support_top_k = 1)),   # review repro
    ([5.0, 0.0, 0.0], dict(support_min_p = 0.1)), # min_p 0.1 keeps only the confident token
])
@pytest.mark.parametrize("adv", [1.0, -1.0])
def test_grpo_gradient_survives_a_singleton_kept_set(logits, support, adv):
    # One iteration, synced weights: vLLM's processed logprob equals the trainer's kept-set score.
    x = torch.tensor([[logits]], dtype = torch.float64, requires_grad = True)
    ids = torch.zeros(1, 1, dtype = torch.long)
    sampling = rr.chunked_selective_log_softmax(x.detach(), ids, 1.0, 1, **support)
    kwargs = dict(support, support_full_vocab_grad = True)
    new = rr.chunked_selective_log_softmax(x, ids, 1.0, 1, **kwargs)
    loss, is_ratio = _grpo(new, new.detach(), sampling, torch.tensor([adv], dtype = torch.float64))
    loss.backward()
    assert is_ratio.item() == 1.0  # no log(kept mass) bias
    # fp64 oracle: IS 1, coef 1, so loss = -adv * exp(lp - lp.detach()) and d/dx = -adv * d log p(y) / dx.
    ox, lp = _full_vocab_oracle(x, ids, 1.0)
    (-adv * lp).sum().backward()
    torch.testing.assert_close(x.grad, ox.grad, rtol = 1e-6, atol = 1e-7)
    assert x.grad.abs().max() > 1e-3


def test_compiled_full_vocab_grad_matches_eager():
    g = torch.Generator().manual_seed(9)
    logits = torch.randn(2, 5, 97, generator = g)
    ids = torch.randint(0, 97, (2, 5), generator = g)
    kw = dict(support_top_p = 0.9, support_min_p = 0.05, support_full_vocab_grad = True)
    fn = getattr(rr.chunked_selective_log_softmax, "_torchdynamo_orig_callable", rr.chunked_selective_log_softmax)
    fn = getattr(fn, "__wrapped__", fn)

    def run(f):
        x = logits.clone().requires_grad_(True)
        out = f(x, ids, 1.0, 2, **kw)
        out.sum().backward()
        return out.detach(), x.grad

    torch._dynamo.reset()
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    eager = run(fn)
    compiled = run(torch.compile(fn, fullgraph = True, dynamic = True, backend = backend))
    assert len(graphs) == 1
    torch.testing.assert_close(compiled[0], eager[0], rtol = 0, atol = 0)
    torch.testing.assert_close(compiled[1], eager[1], rtol = 1e-6, atol = 1e-7)

# Under no_grad (old / ref logps) the full-vocabulary term carries no gradient: it must not run.
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
import unsloth_zoo.rl_replacements as _rr  # noqa: E402

_fn = getattr(_rr.chunked_selective_log_softmax, "__wrapped__", _rr.chunked_selective_log_softmax)


class _Count(TorchDispatchMode):
    def __init__(self): super().__init__(); self.n = 0
    def __torch_dispatch__(self, func, types, args = (), kwargs = None):
        self.n += 1
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("support", [dict(support_top_k = 5), dict(support_min_p = 0.1), dict(support_top_p = 0.9)])
def test_no_grad_full_vocab_flag_runs_no_extra_ops(support):
    z = torch.randn(2, 6, 97, generator = torch.Generator().manual_seed(0))
    ids = z.argmax(-1)
    with torch.no_grad():
        with _Count() as kept: a = _fn(z, ids, 1.0, 2, **support)
        with _Count() as auto: b = _fn(z, ids, 1.0, 2, support_full_vocab_grad = True, **support)
    assert torch.equal(a, b)
    assert auto.n == kept.n, f"{auto.n - kept.n} extra aten ops under no_grad"


def test_grad_still_full_vocabulary():
    z = torch.randn(2, 6, 97, generator = torch.Generator().manual_seed(1)).requires_grad_(True)
    ids = z.detach().argmax(-1)
    _fn(z, ids, 1.0, 2, support_top_k = 1, support_full_vocab_grad = True).sum().backward()
    want = torch.nn.functional.one_hot(ids, 97) - torch.softmax(z.detach(), -1)
    torch.testing.assert_close(z.grad, want, rtol = 1e-5, atol = 1e-6)

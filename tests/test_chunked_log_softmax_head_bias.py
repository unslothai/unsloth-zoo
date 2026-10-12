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

"""GRPO scores hidden states through the output head itself, so a head with a bias (Phi) must add it
before the logit transforms, and the bias must receive its gradient."""
import pytest
import torch

from unsloth_zoo.rl_replacements import chunked_hidden_states_selective_log_softmax


def _reference(states, weight, bias, index, softcap, temperature):
    logits = torch.nn.functional.linear(states, weight, bias)
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    logits = logits.float() / temperature
    return logits.gather(-1, index.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)


@pytest.mark.parametrize("softcap, temperature", [(0.0, 1.0), (30.0, 0.7)])
def test_head_bias_matches_linear(softcap, temperature):
    gen = torch.Generator().manual_seed(0)
    vocab, hidden = 96, 16
    weight = torch.randn(vocab, hidden, generator = gen, dtype = torch.float64).requires_grad_(True)
    bias = (3 * torch.randn(vocab, generator = gen, dtype = torch.float64)).requires_grad_(True)
    states = torch.randn(2, 5, hidden, generator = gen, dtype = torch.float64).requires_grad_(True)
    index = torch.randint(0, vocab, (2, 5), generator = gen)
    upstream = torch.randn(2, 5, generator = gen, dtype = torch.float64)

    out = chunked_hidden_states_selective_log_softmax(
        states, weight, index, 2, 0.0, 0.0, softcap, temperature, lm_head_bias = bias,
    )
    (out * upstream).sum().backward()
    got = [out.detach()] + [t.grad.clone() for t in (states, weight, bias)]
    for t in (states, weight, bias):
        t.grad = None

    ref = _reference(states, weight, bias, index, softcap, temperature)
    (ref * upstream).sum().backward()
    want = [ref.detach()] + [t.grad for t in (states, weight, bias)]
    for a, b in zip(got, want):
        torch.testing.assert_close(a.double(), b.double(), rtol = 1e-5, atol = 1e-5)
    assert (ref - _reference(states, weight, None, index, softcap, temperature)).abs().max() > 0.1


def test_no_bias_is_unchanged():
    gen = torch.Generator().manual_seed(1)
    weight = torch.randn(64, 8, generator = gen)
    states = torch.randn(3, 4, 8, generator = gen)
    index = torch.randint(0, 64, (3, 4), generator = gen)
    a = chunked_hidden_states_selective_log_softmax(states, weight, index, 2)
    b = chunked_hidden_states_selective_log_softmax(states, weight, index, 2, lm_head_bias = None)
    assert torch.equal(a, b)

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

"""UNSLOTH_VLLM_FP32_LM_HEAD=1 must reach the recompute projector too, else new logps (recompute) and old/ref logps
(plain) differ by 16-bit head rounding on identical weights."""

import inspect
import math
import os
import pathlib
import subprocess
import sys
import typing

import pytest
import torch

import unsloth_zoo
import unsloth_zoo.rl_replacements as rr

ROOT = pathlib.Path(unsloth_zoo.__file__).resolve().parents[1]


def _standalone():
    # The trainer copies this source without its module, so run it the same way.
    namespace = {"torch": torch, "os": os, "math": math, "Optional": typing.Optional}
    source = inspect.getsource(rr.chunked_hidden_states_selective_log_softmax_recompute)
    exec(compile(source, "<recompute>", "exec"), namespace)
    function = namespace["chunked_hidden_states_selective_log_softmax_recompute"]
    function._use_compiled = False
    return function


def _inputs(seed = 0, bias = False):
    gen = torch.Generator().manual_seed(seed)
    vocab, hidden = 512, 64
    weight = (torch.randn(vocab, hidden, generator = gen) * 0.5).to(torch.bfloat16)
    states = (torch.randn(2, 7, hidden, generator = gen) * 2).to(torch.bfloat16)
    index = torch.randint(0, vocab, (2, 7), generator = gen)
    head_bias = (torch.randn(vocab, generator = gen)).to(torch.bfloat16) if bias else None
    return states, weight, index, head_bias


def _reference(states, weight, index, head_bias, temperature):
    logits = states.double() @ weight.double().t()
    if head_bias is not None:
        logits = logits + head_bias.double()
    logits = logits / temperature
    logp = logits.log_softmax(-1)
    selected = logp.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    entropy = -(logp.exp() * logp).sum(-1)
    return selected, entropy


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_recompute_fp32_head_matches_plain_fp32_head(bias, temperature):
    states, weight, index, head_bias = _inputs(bias = bias)
    ref_logps, ref_entropy = _reference(states, weight, index, head_bias, temperature)
    recompute = _standalone()
    with torch.autocast(device_type = "cpu", dtype = torch.bfloat16):
        plain = rr.chunked_hidden_states_selective_log_softmax(
            states, weight, index, 2, 0.0, 0.0, 0.0, temperature,
            lm_head_bias = head_bias, fp32_lm_head = True,
        )
        logps, entropy = recompute(
            states, weight, index, 2, 0.0, 0.0, 0.0, temperature,
            lm_head_bias = head_bias, return_entropy = True, fp32_lm_head = True,
        )
    assert (logps.double() - ref_logps).abs().max().item() < 1e-4
    assert (entropy.double() - ref_entropy).abs().max().item() < 1e-4
    # new logps (recompute) and old / ref logps (plain) agree on identical weights.
    assert (logps - plain).abs().max().item() < 1e-5


def test_recompute_fp32_head_gradients_match_plain():
    states, weight, index, head_bias = _inputs(1, bias = True)
    grads = []
    for function in (rr.chunked_hidden_states_selective_log_softmax, _standalone()):
        leaves = [t.clone().requires_grad_(True) for t in (states, weight, head_bias)]
        with torch.autocast(device_type = "cpu", dtype = torch.bfloat16):
            out = function(
                leaves[0], leaves[1], index, 3, 0.0, 0.0, 0.0, 1.0,
                lm_head_bias = leaves[2], fp32_lm_head = True,
            )
        out.float().sum().backward()
        assert [leaf.grad.dtype for leaf in leaves] == [torch.bfloat16] * 3
        grads.append([leaf.grad.float() for leaf in leaves])
    for a, b in zip(*grads):
        assert (a - b).abs().max().item() <= 2e-2 * b.abs().max().item()


def test_recompute_off_is_bit_identical_to_default():
    states, weight, index, _ = _inputs(2)
    recompute = _standalone()
    with torch.autocast(device_type = "cpu", dtype = torch.bfloat16):
        a = recompute(states, weight, index, 2, 0.0, 0.0, 0.0, 0.9)
        b = recompute(states, weight, index, 2, 0.0, 0.0, 0.0, 0.9, fp32_lm_head = False)
    assert torch.equal(a, b)


@pytest.mark.parametrize("value, expected", [(None, "False False"), ("0", "False False"), ("1", "True True")])
def test_env_sets_both_defaults(value, expected):
    env = {k: v for k, v in os.environ.items() if k != "UNSLOTH_VLLM_FP32_LM_HEAD"}
    if value is not None:
        env["UNSLOTH_VLLM_FP32_LM_HEAD"] = value
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    code = (
        "import inspect\n"
        "import unsloth_zoo.rl_replacements as rr\n"
        "a = inspect.signature(rr.chunked_hidden_states_selective_log_softmax).parameters['fp32_lm_head'].default\n"
        "b = inspect.signature(rr.chunked_hidden_states_selective_log_softmax_recompute).parameters['fp32_lm_head'].default\n"
        "print(a, b)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], env = env, capture_output = True, text = True, timeout = 600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == expected

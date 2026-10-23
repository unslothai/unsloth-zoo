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

"""grpo_accumulated_loss's efficient_log_softmax: the fused scorer recomputes logits in backward, but when GRPO falls
back to the old chunked scorer (fp32 head, truncated support, UNSLOTH_GRPO_CHUNKED_LOGPROBS=0) every chunk's logits
are kept for backward, so long completions must keep using Unsloth_Offloaded_Log_Softmax."""
import inspect
import os
import textwrap
from functools import partial

import pytest
import torch

from unsloth_zoo import rl_replacements as rr

_ENVS = ("UNSLOTH_GRPO_CHUNKED_LOGPROBS", "UNSLOTH_VLLM_FP32_LM_HEAD", "UNSLOTH_GRPO_LOGPROB_RECOMPUTE")

# name -> (env, truncated support kwargs)
_CASES = {
    "default": ({}, {}),
    "fp32_head": ({"UNSLOTH_VLLM_FP32_LM_HEAD": "1"}, {}),
    "truncated_support": ({}, {"support_top_k": 20}),
    "kill_switch": ({"UNSLOTH_GRPO_CHUNKED_LOGPROBS": "0"}, {}),
}


def _set_env(monkeypatch, env):
    for name in _ENVS:
        monkeypatch.delenv(name, raising = False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


def _efficient_log_softmax(support, scorer):
    # The offload helpers and efficient_log_softmax are closures inside grpo_accumulated_loss: rebuild them alone.
    src = inspect.getsource(rr.grpo_accumulated_loss)
    start = src.rindex("\n", 0, src.index("def to_device")) + 1
    end = src.rindex("\n", 0, src.index("def compute_logprobs_chunk")) + 1
    ns = {"torch": torch, "os": os, "_support_kwargs": support,
          "chunked_hidden_states_selective_log_softmax": scorer}
    exec(textwrap.dedent(src[start:end]), ns)
    return ns


def test_grpo_scorer_keeps_logits_truth_table(monkeypatch):
    _set_env(monkeypatch, {})
    assert not rr.grpo_scorer_keeps_logits()
    assert rr.grpo_scorer_keeps_logits({"support_top_k": 20})
    assert rr.grpo_scorer_keeps_logits({"support_top_p": 0.9})
    assert rr.grpo_scorer_keeps_logits({"support_min_p": 0.05})
    assert rr.grpo_scorer_keeps_logits({"support_excluded_ids": torch.arange(4)})
    assert rr.grpo_scorer_keeps_logits(fp32_lm_head = True)
    monkeypatch.setenv("UNSLOTH_VLLM_FP32_LM_HEAD", "1")
    assert rr.grpo_scorer_keeps_logits()
    _set_env(monkeypatch, {"UNSLOTH_GRPO_CHUNKED_LOGPROBS": "0"})
    assert rr.grpo_scorer_keeps_logits()


@pytest.mark.parametrize("case, length, batch, recompute, offloaded", [
    ("default", 2000, 2, False, False),         # fused scorer: recomputes in backward, never offloads
    ("default", 100, 16, False, False),
    ("fp32_head", 2000, 2, False, True),        # old scorer keeps logits: long completions offload
    ("truncated_support", 2000, 2, False, True),
    ("kill_switch", 2000, 2, False, True),
    ("kill_switch", 100, 16, False, True),      # batch > 8 offloads too, as on main
    ("kill_switch", 100, 4, False, False),      # short completions, small batch: the plain path, as before
    ("kill_switch", 100, 1, False, False),
    ("kill_switch", 2000, 2, True, False),      # UNSLOTH_GRPO_LOGPROB_RECOMPUTE=1 never keeps logits
    ("truncated_support", 2000, 2, True, True),
])
def test_efficient_log_softmax_offloads_only_for_the_old_scorer(monkeypatch, case, length, batch, recompute, offloaded):
    env, support = _CASES[case]
    _set_env(monkeypatch, dict(env, **({"UNSLOTH_GRPO_LOGPROB_RECOMPUTE": "1"} if recompute else {})))
    calls = []
    ns = _efficient_log_softmax(support, lambda *a, **k: calls.append("plain"))
    ns["Unsloth_Offloaded_Log_Softmax"].apply = staticmethod(lambda *a: calls.append("offloaded"))
    ns["efficient_log_softmax"](
        torch.zeros(batch, length, 4), torch.zeros(8, 4), torch.zeros(batch, length, dtype = torch.long),
        batch_size = batch,
    )
    assert calls == ["offloaded" if offloaded else "plain"]


@pytest.mark.parametrize("case", list(_CASES))
def test_long_completion_saves_no_logits_for_backward(monkeypatch, case):
    # Real scorers, long completion: no route may keep rows x vocab logits for backward, and gradients are unchanged.
    env, support = _CASES[case]
    _set_env(monkeypatch, env)
    batch, length, hidden, vocab = 2, 1100, 8, 128
    gen = torch.Generator().manual_seed(0)
    states = torch.randn(batch, length, hidden, generator = gen)
    head = torch.randn(vocab, hidden, generator = gen) / 3
    index = torch.randint(0, vocab, (batch, length), generator = gen)
    scorer = partial(rr.grpo_chunked_hidden_states_logprobs, **support)
    ns = _efficient_log_softmax(support, scorer)

    saved = []
    def pack(tensor):
        saved.append(tensor.numel() * tensor.element_size())
        return tensor

    leaf = states.clone().requires_grad_(True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        out = ns["efficient_log_softmax"](leaf, head, index, chunks = 4, batch_size = batch)
    logits_bytes = batch * length * vocab * 4
    assert sum(saved) < logits_bytes // 2, (case, sum(saved), logits_bytes)

    out.sum().backward()
    ref_leaf = states.clone().requires_grad_(True)
    scorer(ref_leaf, head, index, 4).sum().backward()
    torch.testing.assert_close(leaf.grad, ref_leaf.grad, rtol = 1e-4, atol = 1e-5)


def test_old_scorer_keeps_logits_control():
    # Control for the test above: the old chunked scorer itself saves at least rows x vocab bytes.
    batch, length, hidden, vocab = 2, 1100, 8, 128
    states = torch.randn(batch, length, hidden, requires_grad = True)
    head = torch.randn(vocab, hidden)
    index = torch.randint(0, vocab, (batch, length))
    saved = []
    def pack(tensor):
        saved.append(tensor.numel() * tensor.element_size())
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        kwargs = {}
        if "fused_linear" in inspect.signature(rr.chunked_hidden_states_selective_log_softmax).parameters:
            kwargs["fused_linear"] = False  # the old chunked path, not the fused kernel
        rr.chunked_hidden_states_selective_log_softmax(states, head, index, 4, **kwargs)
    assert sum(saved) >= batch * length * vocab * 4

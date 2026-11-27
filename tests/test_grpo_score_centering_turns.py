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

"""Score centering store: per-turn head placement (multi-turn tools, carried partial rollouts) and bounded memory."""
import inspect
import types

import pytest
import torch

from unsloth_zoo import score_centering as sc
from unsloth_zoo.rl_replacements import grpo_accumulated_loss, grpo_compute_loss

D64 = torch.float64
PAD = 0


class _LP:
    def __init__(self, logprob, rank):
        self.logprob, self.rank = logprob, rank


def _output(prompt, completion, V, k, gen):
    """A vLLM RequestOutput stand-in plus the true head (ids, float32 logps) of every generated token."""
    positions, truth = [], []
    for tok in completion:
        logq = torch.log_softmax(torch.randn(V, generator = gen, dtype = D64), -1).to(torch.float32)
        order = logq.argsort(descending = True).tolist()
        rank = {t: r + 1 for r, t in enumerate(order)}
        d = {tok: _LP(float(logq[tok]), rank[tok])}
        for t in order[:k]:
            d[t] = _LP(float(logq[t]), rank[t])
        positions.append(d)
        truth.append((order[:k], logq[order[:k]]))
    comp = types.SimpleNamespace(token_ids = list(completion), logprobs = positions)
    return types.SimpleNamespace(prompt_token_ids = list(prompt), outputs = [comp]), truth


def _batch(rows, prompt_width, width):
    """TRL layout: left-padded prompt | right-padded completion."""
    input_ids = torch.full((len(rows), prompt_width + width), PAD, dtype = torch.long)
    for b, (p, c) in enumerate(rows):
        input_ids[b, prompt_width - len(p):prompt_width] = torch.tensor(p)
        input_ids[b, prompt_width:prompt_width + len(c)] = torch.tensor(c)
    return input_ids


def _reference_heads(B, L, k, segments):
    """segments: (row, first completion column, truth list) placed by hand, per turn."""
    ids = torch.full((B, L, k), -1, dtype = torch.long)
    logps = torch.full((B, L, k), float("-inf"), dtype = torch.float32)
    for b, col, truth in segments:
        for j, (top, vals) in enumerate(truth):
            if col + j >= L:
                break
            ids[b, col + j] = torch.tensor(top)
            logps[b, col + j] = vals
    return ids, logps


def _centered_loss(logits, sampled, ids, logps, mask, adv):
    logp = torch.log_softmax(logits, -1)
    new = logp.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    head_mask = (ids >= 0) & mask.to(torch.bool).unsqueeze(-1)
    wfn = sc.score_centering_weight_fn("token_truncate", None, 2.0)
    corr, _, _ = sc.score_centering_correction(logp.gather(-1, ids.clamp_min(0)), logps.to(D64), wfn, head_mask)
    old = new.detach() + 0.1
    kw = dict(
        loss_type = "reinforce", use_vllm = True, vllm_importance_sampling_correction = True,
        vllm_importance_sampling_mode = "token_truncate", vllm_importance_sampling_clip_min = None,
        vllm_importance_sampling_clip_max = 2.0, rollout_logps_as_old = True,
        current_gradient_accumulation_steps = 1, num_items_in_batch = 1.0, score_centering_correction = corr,
    )
    return grpo_compute_loss(None, new, old, None, sampled, mask, 0.0, adv, **kw)[0]


def test_multi_turn_tool_row_gets_each_turns_head_fp64_oracle():
    sc.reset_score_centering()
    gen = torch.Generator().manual_seed(0)
    V, k = 64, 6
    P, C1, R1, C2 = [5, 6, 7], [11, 12, 13], [21, 22], [31, 32, 33, 34]
    P2, C1b = [8, 9], [14, 15]  # a sibling row (same prompt as nothing else) and a single-turn row
    t1, tr1 = _output(P, C1, V, k, gen)
    t1b, tr1b = _output(P2, C1b, V, k, gen)
    t2, tr2 = _output(P + C1 + R1, C2, V, k, gen)  # TRL: turn 2's prompt is prompt + completion + tool ids
    sc.record_rollout_topk([t1, t1b], k)
    sc.record_rollout_topk([t2], k)
    L, Tp = 12, 4
    input_ids = _batch([(P, C1 + R1 + C2), (P2, C1b)], Tp, L)
    ids, logps, matched = sc.lookup_rollout_topk(input_ids, L, k, PAD)
    assert matched.tolist() == [True, True]
    ref_ids, ref_logps = _reference_heads(2, L, k, [(0, 0, tr1), (0, len(C1 + R1), tr2), (1, 0, tr1b)])
    assert torch.equal(ids, ref_ids)  # C1 columns hold turn 1's head, tool columns none, C2 columns turn 2's
    assert torch.equal(logps, ref_logps)

    # Centered REINFORCE loss and gradient equal the per-turn reference (fp64).
    logits = torch.randn(2, L, V, generator = gen, dtype = D64, requires_grad = True)
    sampled = input_ids[:, Tp:]
    mask = torch.zeros(2, L, dtype = D64)
    mask[0, :len(C1)] = 1  # tool tokens are masked out of the loss
    mask[0, len(C1 + R1):len(C1 + R1 + C2)] = 1
    mask[1, :len(C1b)] = 1
    adv = torch.tensor([1.3, -0.7], dtype = D64)
    loss = _centered_loss(logits, sampled, ids, logps, mask, adv)
    ref = _centered_loss(logits, sampled, ref_ids, ref_logps, mask, adv)
    assert abs(loss.item() - ref.item()) < 1e-12
    (g,) = torch.autograd.grad(loss, logits)
    (gr,) = torch.autograd.grad(ref, logits)
    assert (g - gr).abs().max().item() < 1e-12


def test_carried_partial_rollout_places_both_segments():
    sc.reset_score_centering()
    gen = torch.Generator().manual_seed(1)
    V, k = 50, 4
    P, A, Bt = [3, 4], [10, 11], [12, 13, 14]
    a, tra = _output(P, A, V, k, gen)  # aborted at step t, carried
    sc.record_rollout_topk([a], k)
    sc.lookup_rollout_topk(_batch([([1], [2])], 1, 1), 1, k, PAD)  # step t trains other rows: rollout closes
    b, trb = _output(P + A, Bt, V, k, gen)  # resumed from prompt + carried tokens at step t + 1
    sc.record_rollout_topk([b], k)
    L = 6
    ids, logps, matched = sc.lookup_rollout_topk(_batch([(P, A + Bt)], 2, L), L, k, PAD)
    ref_ids, ref_logps = _reference_heads(1, L, k, [(0, 0, tra), (0, len(A), trb)])
    assert matched.tolist() == [True]
    assert torch.equal(ids, ref_ids) and torch.equal(logps, ref_logps)


def test_truncated_last_turn_keeps_only_its_tokens():
    sc.reset_score_centering()
    gen = torch.Generator().manual_seed(2)
    V, k = 40, 4
    P, C1, R1, C2 = [5, 6], [11, 12], [21], [31, 32, 33]
    t1, tr1 = _output(P, C1, V, k, gen)
    t2, tr2 = _output(P + C1 + R1, C2, V, k, gen)
    sc.record_rollout_topk([t1], k)
    sc.record_rollout_topk([t2], k)
    # TRL cut turn 2 to fit max_completion_length: the row keeps C2[:1] then padding.
    L = 6
    ids, logps, matched = sc.lookup_rollout_topk(_batch([(P, C1 + R1 + C2[:1])], 2, L), L, k, PAD)
    ref_ids, _ = _reference_heads(1, L, k, [(0, 0, tr1), (0, len(C1 + R1), tr2[:1])])
    assert matched.tolist() == [True]
    assert torch.equal(ids, ref_ids)


def test_row_that_differs_from_the_record_gets_no_head():
    sc.reset_score_centering()
    gen = torch.Generator().manual_seed(3)
    V, k = 40, 4
    P, C1 = [5, 6], [11, 12, 13]
    t1, _ = _output(P, C1, V, k, gen)
    sc.record_rollout_topk([t1], k)
    # Same prompt, a completion that diverges at its second token: no column may take the record's head.
    ids, _, matched = sc.lookup_rollout_topk(_batch([(P, [11, 99, 13])], 2, 4), 4, k, PAD)
    assert matched.tolist() == [False] and (ids == -1).all()


def test_store_is_bounded_without_a_lookup():
    sc.reset_score_centering()
    gen = torch.Generator().manual_seed(4)
    out, _ = _output([1, 2], [3, 4, 5], 20, 4, gen)
    for step in range(500):
        out.prompt_token_ids = [1, 2, step + 100]
        sc.record_rollout_topk([out], 4)
    stored = sum(len(r["raw"]) for r in sc._STATE.rollouts.values())
    assert stored <= sc._MAX_ROLLOUTS * sc._MAX_RECORD_CALLS, stored
    # The newest calls are still found.
    ids, _, matched = sc.lookup_rollout_topk(_batch([([1, 2, 599], [3, 4, 5])], 3, 3), 3, 4, PAD)
    assert matched.tolist() == [True]


def test_pause_stops_recording_and_the_logprob_raise():
    sc.reset_score_centering()
    gen = torch.Generator().manual_seed(5)
    k = 4
    seen = []

    class _LLM:
        def generate(self, prompts, sampling_params = None):
            seen.append(sampling_params.logprobs)
            return [_output([1, 2], [3, 4], 20, k, gen)[0]]

    llm = sc.wrap_vllm_generate_topk(_LLM(), k)
    llm.generate(["a"], sampling_params = types.SimpleNamespace(logprobs = None))
    assert seen[-1] == k and sc.score_centering_stats()["recorded"] == 1
    assert sc.pause_score_centering_record() is True
    assert not sc._STATE.rollouts
    for _ in range(5):
        llm.generate(["a"], sampling_params = types.SimpleNamespace(logprobs = None))
    assert seen[-1] is None and sc.score_centering_stats()["recorded"] == 1 and not sc._STATE.rollouts
    # A centering lookup turns recording back on for the next rollout.
    sc.lookup_rollout_topk(_batch([([1, 2], [3, 4])], 2, 2), 2, k, PAD)
    llm.generate(["a"], sampling_params = types.SimpleNamespace(logprobs = None))
    assert seen[-1] == k and sc.score_centering_stats()["recorded"] == 2


def test_grpo_loss_pauses_recording_when_it_does_not_center():
    src = inspect.getsource(grpo_accumulated_loss)
    lookup = src.index("_sc_lookup(")
    pause = src.index("_sc_pause()")
    assert lookup < pause
    assert "if not _sc_k and" in src[lookup:pause]

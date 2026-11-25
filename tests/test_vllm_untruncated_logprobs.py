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

"""GRPO vLLM IS correction: the colocated engine returns log_softmax(logits / T) from before min_p / top_k / top_p,
so the trainer scores the full vocabulary instead of rebuilding the kept set. Exercised on stand-ins with the
call structure of vLLM's V1 sampler (vllm/v1/sample/sampler.py, 0.11-0.33) and V2 sampler
(vllm/v1/worker/gpu/sample/sampler.py, 0.20-0.34+)."""

import ast
import functools
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "UNSLOTH_VLLM_UNTRUNCATED_LOGPROBS",
    "_UNTRUNCATED_CAPTURE",
    "patch_vllm_untruncated_logprobs",
    "set_vllm_untruncated_logprobs",
)


def _load():
    source = (ROOT / "unsloth_zoo" / "vllm_utils.py").read_text(encoding = "utf-8")
    nodes = []
    for node in ast.parse(source).body:
        name = getattr(node, "name", None)
        if name is None and isinstance(node, ast.Assign):
            name = getattr(node.targets[0], "id", None)
        if name in NAMES:
            nodes.append(node)
    assert len(nodes) == len(NAMES)
    ns = {"torch": torch, "functools": functools, "os": os}
    exec(compile(ast.Module(nodes, []), "vllm_utils.py", "exec"), ns)
    return ns


def _min_p_mask(logits, min_p):
    probs = logits.softmax(-1)
    return logits.masked_fill(probs < min_p * probs.amax(-1, keepdim = True), float("-inf"))


def _fake_v1(monkeypatch, static = True):
    def _temperature(logits, temp, all_random):
        return logits.div_(temp.unsqueeze(1))

    class Sampler:
        def __init__(self, mode):
            self.logprobs_mode = mode

        if static:
            apply_temperature = staticmethod(_temperature)
        else:  # vLLM 0.11.0: an instance method
            def apply_temperature(self, logits, temp, all_random):
                return logits.div_(temp.unsqueeze(1))

        def sample(self, logits, sampling_metadata, logprobs_mode_override = None):
            logits = self.apply_temperature(logits, sampling_metadata.temperature, True)
            logits.copy_(_min_p_mask(logits, 0.1))
            return logits.argmax(-1), torch.log_softmax(logits, -1)

    module = types.ModuleType("vllm.v1.sample.sampler")
    module.Sampler = Sampler
    for name in ("vllm", "vllm.v1", "vllm.v1.sample"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.sample.sampler", module)
    return Sampler


def _fake_v2(monkeypatch):
    class SamplingStates:
        def __init__(self, temp):
            self.temp = temp

        def apply_temperature(self, logits):
            logits.div_(self.temp.unsqueeze(1))

        def apply_min_p(self, logits):
            logits.copy_(_min_p_mask(logits, 0.1))

    sampler = types.ModuleType("vllm.v1.worker.gpu.sample.sampler")
    sampler.compute_topk_scores = lambda logits, sampled: torch.log_softmax(logits, -1).gather(1, sampled[:, None])

    class Sampler:
        def __init__(self, mode, temp):
            self.logprobs_mode = mode
            self.sampling_states = SamplingStates(temp)

        def __call__(self, logits):
            processed = logits.clone()
            self.sampling_states.apply_temperature(processed)
            self.sampling_states.apply_min_p(processed)
            sampled = processed.argmax(-1)
            # The module global is looked up at call time, as in vLLM.
            return sampler.compute_topk_scores(processed, sampled)

    sampler.Sampler = Sampler
    states = types.ModuleType("vllm.v1.worker.gpu.sample.states")
    states.SamplingStates = SamplingStates
    for name in ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu", "vllm.v1.worker.gpu.sample"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.sample.sampler", sampler)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.sample.states", states)
    return Sampler


def _logits():
    torch.manual_seed(0)
    return torch.randn(4, 50) * 3


@pytest.mark.parametrize("static", [True, False])
@pytest.mark.parametrize("on", [True, False])
def test_v1_returns_scaled_untruncated_logprobs(monkeypatch, on, static):
    Sampler = _fake_v1(monkeypatch, static)
    ns = _load()
    monkeypatch.delenv("UNSLOTH_VLLM_UNTRUNCATED_LOGPROBS", raising = False)
    assert ns["set_vllm_untruncated_logprobs"](on) is on
    temp = torch.full((4,), 0.7)
    logits = _logits()
    md = SimpleNamespace(temperature = temp, max_num_logprobs = 0, logprob_token_ids = None)
    _, logprobs = Sampler("processed_logprobs").sample(logits.clone(), md)
    scaled = torch.log_softmax(logits / 0.7, -1)
    truncated = torch.log_softmax(_min_p_mask(logits / 0.7, 0.1), -1)
    torch.testing.assert_close(logprobs, scaled if on else truncated)
    assert (ns["UNSLOTH_VLLM_UNTRUNCATED_LOGPROBS"]["calls"] > 0) is on


def test_v1_raw_mode_and_steps_without_logprobs_are_untouched(monkeypatch):
    Sampler = _fake_v1(monkeypatch)
    ns = _load()
    ns["set_vllm_untruncated_logprobs"](True)
    logits = _logits()
    truncated = torch.log_softmax(_min_p_mask(logits / 0.7, 0.1), -1)
    md = SimpleNamespace(temperature = torch.full((4,), 0.7), max_num_logprobs = None, logprob_token_ids = None)
    torch.testing.assert_close(Sampler("processed_logprobs").sample(logits.clone(), md)[1], truncated)
    md.max_num_logprobs = 0
    torch.testing.assert_close(Sampler("raw_logprobs").sample(logits.clone(), md)[1], truncated)
    assert ns["UNSLOTH_VLLM_UNTRUNCATED_LOGPROBS"]["calls"] == 0


@pytest.mark.parametrize("on", [True, False])
def test_v2_returns_scaled_untruncated_logprobs(monkeypatch, on):
    Sampler = _fake_v2(monkeypatch)
    ns = _load()
    ns["set_vllm_untruncated_logprobs"](on)
    logits = _logits()
    out = Sampler("processed_logprobs", torch.full((4,), 1.3))(logits.clone())
    ref = logits / 1.3 if on else _min_p_mask(logits / 1.3, 0.1)
    sampled = _min_p_mask(logits / 1.3, 0.1).argmax(-1)
    torch.testing.assert_close(out, torch.log_softmax(ref, -1).gather(1, sampled[:, None]))


def test_kill_switch_and_unknown_vllm(monkeypatch):
    _fake_v1(monkeypatch)
    ns = _load()
    monkeypatch.setenv("UNSLOTH_VLLM_UNTRUNCATED_LOGPROBS", "0")
    assert ns["set_vllm_untruncated_logprobs"](True) is False
    monkeypatch.delenv("UNSLOTH_VLLM_UNTRUNCATED_LOGPROBS")
    for name in list(sys.modules):
        if name == "vllm" or name.startswith("vllm."):
            monkeypatch.setitem(sys.modules, name, None)
    ns = _load()
    assert ns["set_vllm_untruncated_logprobs"](True) is False


def test_trainer_scores_the_full_vocabulary_when_untruncated(monkeypatch):
    rl = pytest.importorskip("unsloth_zoo.rl_replacements")
    vllm_utils = pytest.importorskip("unsloth_zoo.vllm_utils")
    state = vllm_utils.UNSLOTH_VLLM_UNTRUNCATED_LOGPROBS
    monkeypatch.setitem(state, "on", True)
    monkeypatch.setitem(state, "installed", True)
    monkeypatch.setattr(rl, "grpo_sampling_support", lambda trainer: (0, 1.0, 0.1))
    monkeypatch.setattr(rl, "_grpo_engine_excluded_ids", lambda trainer: None)
    monkeypatch.delenv("UNSLOTH_GRPO_TRUNCATED_SUPPORT", raising = False)
    trainer = SimpleNamespace(use_vllm = True, vllm_importance_sampling_correction = True,
                              _unsloth_vllm_untruncated_logprobs = True)
    assert rl.grpo_rollout_logprobs_untruncated(trainer) is True
    assert rl.grpo_truncated_support_automatic(trainer) is False
    assert rl.grpo_truncated_support_kwargs(trainer) == {}
    excluded = torch.tensor([7, 9])
    monkeypatch.setattr(rl, "_grpo_engine_excluded_ids", lambda trainer: excluded)
    kwargs = rl.grpo_truncated_support_kwargs(trainer)
    assert kwargs["support_min_p"] == 0.0 and kwargs["support_top_p"] == 1.0 and kwargs["support_top_k"] == 0
    assert kwargs["support_excluded_ids"] is excluded
    trainer._unsloth_vllm_untruncated_logprobs = False
    assert rl.grpo_rollout_logprobs_untruncated(trainer) is False

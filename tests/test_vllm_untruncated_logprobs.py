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
    "_vllm_engine_speculative",
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

        def get_logprobs_dims(self, idx_mapping_np):
            return None if self.no_logprobs else (0, 0)

        no_logprobs = False

        def __call__(self, logits, input_batch = None):
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
    out = Sampler("processed_logprobs", torch.full((4,), 1.3))(logits.clone(), SimpleNamespace(idx_mapping_np = None))
    ref = logits / 1.3 if on else _min_p_mask(logits / 1.3, 0.1)
    sampled = _min_p_mask(logits / 1.3, 0.1).argmax(-1)
    torch.testing.assert_close(out, torch.log_softmax(ref, -1).gather(1, sampled[:, None]))


def test_v2_batches_without_logprobs_skip_the_capture(monkeypatch):
    Sampler = _fake_v2(monkeypatch)
    ns = _load()
    ns["set_vllm_untruncated_logprobs"](True)
    sampler = Sampler("processed_logprobs", torch.full((4,), 1.3))
    sampler.no_logprobs = True
    copies = []
    original = torch.Tensor.clone
    monkeypatch.setattr(torch.Tensor, "clone", lambda self, *a, **k: copies.append(1) or original(self, *a, **k))
    sampler(_logits(), SimpleNamespace(idx_mapping_np = None))
    assert len(copies) == 1  # the stand-in's own copy only
    assert ns["UNSLOTH_VLLM_UNTRUNCATED_LOGPROBS"]["calls"] == 0


def test_v1_token_specific_logprobs_are_left_alone(monkeypatch):
    Sampler = _fake_v1(monkeypatch)
    ns = _load()
    ns["set_vllm_untruncated_logprobs"](True)
    logits = _logits()
    md = SimpleNamespace(temperature = torch.full((4,), 0.7), max_num_logprobs = 0, logprob_token_ids = {0: [1]})
    truncated = torch.log_softmax(_min_p_mask(logits / 0.7, 0.1), -1)
    torch.testing.assert_close(Sampler("processed_logprobs").sample(logits.clone(), md)[1], truncated)


def test_speculative_engines_keep_the_kept_set_path(monkeypatch):
    _fake_v1(monkeypatch)
    ns = _load()
    spec = SimpleNamespace(llm_engine = SimpleNamespace(vllm_config = SimpleNamespace(speculative_config = {"method": "ngram"})))
    plain = SimpleNamespace(llm_engine = SimpleNamespace(vllm_config = SimpleNamespace(speculative_config = None)))
    assert ns["set_vllm_untruncated_logprobs"](True, spec) is False
    assert ns["set_vllm_untruncated_logprobs"](True, plain) is True


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


class MinPLogitsProcessor:
    # Name-matched stand-in for vLLM's built-in (vllm/v1/sample/logits_processor/builtin.py).
    def __init__(self, count):
        self.min_p_count = count

    def apply(self, logits):
        return _min_p_mask(logits, 0.1) if self.min_p_count else logits


def _fake_v1_metadata_driven(monkeypatch):
    # V1 Sampler.sample: temperature, argmax-invariant processors, then top_k (the stand-in's only filter).
    class Sampler:
        def __init__(self, mode):
            self.logprobs_mode = mode

        @staticmethod
        def apply_temperature(logits, temp, all_random):
            return logits.div_(temp.unsqueeze(1))

        def sample(self, logits, md, logprobs_mode_override = None):
            logits = self.apply_temperature(logits, md.temperature, True)
            for processor in md.logitsprocs.argmax_invariant:
                logits = processor.apply(logits)
            if md.top_k is not None:
                kth = logits.topk(int(md.top_k[0]), -1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            mode = logprobs_mode_override or self.logprobs_mode
            processed = logits if mode == "processed_logits" else logits.log_softmax(dim = -1, dtype = torch.float32)
            return logits.argmax(-1), processed

    module = types.ModuleType("vllm.v1.sample.sampler")
    module.Sampler = Sampler
    for name in ("vllm", "vllm.v1", "vllm.v1.sample"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.sample.sampler", module)
    return Sampler


def _v1_md(min_p_count = 0, top_k = None, processors = None):
    if processors is None:
        processors = [MinPLogitsProcessor(min_p_count)]
    return SimpleNamespace(
        temperature = torch.full((4,), 0.7), max_num_logprobs = 0, logprob_token_ids = None,
        top_k = top_k, top_p = None, logitsprocs = SimpleNamespace(argmax_invariant = processors),
    )


def _count_clones(monkeypatch):
    copies = []
    original = torch.Tensor.clone
    monkeypatch.setattr(torch.Tensor, "clone", lambda self, *a, **k: copies.append(1) or original(self, *a, **k))
    return copies


@pytest.mark.parametrize("mode", ["processed_logprobs", "processed_logits"])
def test_v1_unfiltered_steps_skip_the_capture_with_identical_logprobs(monkeypatch, mode):
    Sampler = _fake_v1_metadata_driven(monkeypatch)
    ns = _load()
    state = ns["UNSLOTH_VLLM_UNTRUNCATED_LOGPROBS"]
    logits = _logits()
    # The capture path's answer for the same unfiltered step (an unrecognised metadata layout forces it).
    forced = _v1_md()
    forced.logitsprocs = SimpleNamespace(argmax_invariant = [SimpleNamespace(apply = lambda x: x)])
    ns["set_vllm_untruncated_logprobs"](True)
    _, captured = Sampler(mode).sample(logits.clone(), forced)
    assert state["calls"] == 1 and state.get("unfiltered", 0) == 0
    inputs = logits.clone()
    copies = _count_clones(monkeypatch)
    sampled, fast = Sampler(mode).sample(inputs, _v1_md())
    assert copies == [] and state["calls"] == 1 and state.get("unfiltered", 0) == 1
    assert torch.equal(fast, captured)
    ns["set_vllm_untruncated_logprobs"](False)
    _, off = Sampler(mode).sample(logits.clone(), _v1_md())
    assert torch.equal(fast, off)


@pytest.mark.parametrize("case", ["min_p", "top_k", "unknown_processor"])
def test_v1_filtered_steps_keep_the_capture(monkeypatch, case):
    Sampler = _fake_v1_metadata_driven(monkeypatch)
    ns = _load()
    ns["set_vllm_untruncated_logprobs"](True)
    md = {
        "min_p": _v1_md(min_p_count = 1),
        "top_k": _v1_md(top_k = torch.full((4,), 5)),
        "unknown_processor": _v1_md(processors = [MinPLogitsProcessor(0), SimpleNamespace(apply = lambda x: x)]),
    }[case]
    logits = _logits()
    _, logprobs = Sampler("processed_logprobs").sample(logits.clone(), md)
    torch.testing.assert_close(logprobs, torch.log_softmax(logits / 0.7, -1))
    state = ns["UNSLOTH_VLLM_UNTRUNCATED_LOGPROBS"]
    assert state["calls"] == 1 and state.get("unfiltered", 0) == 0


def _fake_v2_states(monkeypatch, vocab):
    Sampler = _fake_v2(monkeypatch)
    import numpy as np

    def states(sampler, min_p = 0.0, top_p = 1.0, top_k = None):
        st = sampler.sampling_states
        st.vocab_size = vocab
        st.min_p = SimpleNamespace(np = np.full(8, min_p, dtype = np.float32))
        st.top_p = SimpleNamespace(np = np.full(8, top_p, dtype = np.float32))
        st.top_k = SimpleNamespace(np = np.full(8, vocab if top_k is None else top_k, dtype = np.int32))
        return sampler

    def apply_min_p(self, logits):
        # vLLM skips the min_p kernel when no request uses it.
        if (self.min_p.np != 0.0).any():
            logits.copy_(_min_p_mask(logits, 0.1))

    monkeypatch.setattr(sys.modules["vllm.v1.worker.gpu.sample.states"].SamplingStates, "apply_min_p", apply_min_p)
    return Sampler, states


def test_v2_unfiltered_batches_skip_the_capture_with_identical_logprobs(monkeypatch):
    import numpy as np
    Sampler, states = _fake_v2_states(monkeypatch, 50)
    ns = _load()
    state = ns["UNSLOTH_VLLM_UNTRUNCATED_LOGPROBS"]
    ns["set_vllm_untruncated_logprobs"](True)
    logits = _logits()
    batch = SimpleNamespace(idx_mapping_np = np.arange(4))
    # Capture path for the same batch (no idx_mapping_np: the layout is not recognised).
    captured = states(Sampler("processed_logprobs", torch.full((4,), 1.3)))(logits.clone(), SimpleNamespace())
    assert state["calls"] == 1 and state.get("unfiltered", 0) == 0
    inputs = logits.clone()
    copies = _count_clones(monkeypatch)
    fast = states(Sampler("processed_logprobs", torch.full((4,), 1.3)))(inputs, batch)
    assert len(copies) == 1  # the stand-in's own copy only
    assert state["calls"] == 1 and state.get("unfiltered", 0) == 1
    assert torch.equal(fast, captured)


@pytest.mark.parametrize("case", ["min_p", "top_p", "top_k"])
def test_v2_filtered_batches_keep_the_capture(monkeypatch, case):
    import numpy as np
    Sampler, states = _fake_v2_states(monkeypatch, 50)
    ns = _load()
    ns["set_vllm_untruncated_logprobs"](True)
    kwargs = {"min_p": {"min_p": 0.1}, "top_p": {"top_p": 0.9}, "top_k": {"top_k": 5}}[case]
    sampler = states(Sampler("processed_logprobs", torch.full((4,), 1.3)), **kwargs)
    sampler(_logits(), SimpleNamespace(idx_mapping_np = np.arange(4)))
    state = ns["UNSLOTH_VLLM_UNTRUNCATED_LOGPROBS"]
    assert state.get("unfiltered", 0) == 0
    # The stand-in's min_p only runs for min_p (top_k / top_p are not modelled): the capture still swapped in.
    assert state["calls"] == 1

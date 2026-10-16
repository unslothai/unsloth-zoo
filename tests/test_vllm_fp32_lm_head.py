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

"""UNSLOTH_VLLM_FP32_LM_HEAD=1 runs the output head in fp32 on both sides of GRPO.

Trainer: chunked_hidden_states_selective_log_softmax must produce fp32 logits even
under the bf16 autocast the GRPO loss runs it in, and be unchanged when off.
vLLM: load_vllm passes hf_overrides head_dtype=float32, only where vLLM honours it
(LogitsProcessor._apply_head, vLLM >= 0.26.0), and refuses otherwise.
load_vllm needs a GPU, so the vLLM side is checked on the helpers and the AST.
"""

import ast
import os
import pathlib
import subprocess
import sys
import types

import pytest
import torch

import unsloth_zoo
from unsloth_zoo.rl_replacements import chunked_hidden_states_selective_log_softmax

ROOT = pathlib.Path(unsloth_zoo.__file__).resolve().parents[1]


def _inputs(seed = 0):
    gen = torch.Generator().manual_seed(seed)
    vocab, hidden = 512, 64
    weight = (torch.randn(vocab, hidden, generator = gen) * 0.5).to(torch.bfloat16)
    states = (torch.randn(2, 7, hidden, generator = gen) * 2).to(torch.bfloat16)
    index = torch.randint(0, vocab, (2, 7), generator = gen)
    return states, weight, index


def _reference(states, weight, index, temperature):
    logits = (states.double() @ weight.double().t()) / temperature
    return logits.gather(-1, index.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_fp32_head_under_autocast_matches_fp64(temperature):
    states, weight, index = _inputs()
    ref = _reference(states, weight, index, temperature)
    with torch.autocast(device_type = "cpu", dtype = torch.bfloat16):
        fp32 = chunked_hidden_states_selective_log_softmax(
            states, weight, index, 2, 0.0, 0.0, 0.0, temperature, fp32_lm_head = True,
        )
        bf16 = chunked_hidden_states_selective_log_softmax(
            states, weight, index, 2, 0.0, 0.0, 0.0, temperature, fp32_lm_head = False,
        )
    assert fp32.dtype == torch.float32
    fp32_err = (fp32.double() - ref).abs().max().item()
    bf16_err = (bf16.double() - ref).abs().max().item()
    assert fp32_err < 1e-4, fp32_err
    # The 16-bit head rounds logits of magnitude ~20 to steps of ~0.1.
    assert bf16_err > 20 * fp32_err, (bf16_err, fp32_err)


def test_fp32_head_gradients_flow():
    states, weight, index = _inputs(1)
    states = states.requires_grad_(True)
    weight = weight.requires_grad_(True)
    out = chunked_hidden_states_selective_log_softmax(
        states, weight, index, 3, 0.0, 0.0, 0.0, 1.0, fp32_lm_head = True,
    )
    out.sum().backward()
    assert states.grad is not None and states.grad.dtype == torch.bfloat16
    assert weight.grad is not None and weight.grad.dtype == torch.bfloat16


def test_off_is_bit_identical_to_default():
    states, weight, index = _inputs(2)
    with torch.autocast(device_type = "cpu", dtype = torch.bfloat16):
        a = chunked_hidden_states_selective_log_softmax(states, weight, index, 2, 0.0, 0.0, 0.0, 0.9)
        b = chunked_hidden_states_selective_log_softmax(
            states, weight, index, 2, 0.0, 0.0, 0.0, 0.9, fp32_lm_head = False,
        )
    assert torch.equal(a, b)


@pytest.mark.parametrize("value, expected", [(None, "False"), ("0", "False"), ("1", "True")])
def test_env_sets_the_default(value, expected):
    # The default is bound when the function is defined, so probe a fresh interpreter.
    env = {k: v for k, v in os.environ.items() if k != "UNSLOTH_VLLM_FP32_LM_HEAD"}
    if value is not None:
        env["UNSLOTH_VLLM_FP32_LM_HEAD"] = value
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    code = (
        "import inspect\n"
        "from unsloth_zoo.rl_replacements import chunked_hidden_states_selective_log_softmax as f\n"
        "print(inspect.signature(f).parameters['fp32_lm_head'].default)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], env = env, capture_output = True, text = True, timeout = 600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == expected


def _vllm_utils_tree():
    return ast.parse((ROOT / "unsloth_zoo" / "vllm_utils.py").read_text(encoding = "utf-8-sig"))


def _helpers(monkeypatch, has_apply_head, version):
    tree = _vllm_utils_tree()
    wanted = {"_fp32_lm_head_requested", "_vllm_supports_fp32_lm_head", "_vllm_fp32_lm_head_overrides"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    ns = {"os": os, "vllm_version": version}
    exec(compile(ast.Module(body = nodes, type_ignores = []), "vllm_utils_helpers", "exec"), ns)

    class LogitsProcessor:
        pass
    if has_apply_head:
        LogitsProcessor._apply_head = lambda self: None
    names = ["vllm", "vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.logits_processor"]
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules[names[-1]].LogitsProcessor = LogitsProcessor
    return ns


def test_overrides_off_by_default(monkeypatch):
    monkeypatch.delenv("UNSLOTH_VLLM_FP32_LM_HEAD", raising = False)
    ns = _helpers(monkeypatch, has_apply_head = False, version = "0.11.2")
    assert ns["_vllm_fp32_lm_head_overrides"](True) == {}


def test_overrides_on_supported_vllm(monkeypatch):
    monkeypatch.setenv("UNSLOTH_VLLM_FP32_LM_HEAD", "1")
    ns = _helpers(monkeypatch, has_apply_head = True, version = "0.29.0")
    assert ns["_vllm_fp32_lm_head_overrides"](True) == {"head_dtype": "float32"}
    # Inference-only engines produce no rollout logprobs, so they keep the 16-bit head.
    assert ns["_vllm_fp32_lm_head_overrides"](False) == {}


def test_overrides_refuse_old_vllm(monkeypatch):
    monkeypatch.setenv("UNSLOTH_VLLM_FP32_LM_HEAD", "1")
    ns = _helpers(monkeypatch, has_apply_head = False, version = "0.24.0")
    with pytest.raises(RuntimeError, match = "vLLM >= 0.26.0"):
        ns["_vllm_fp32_lm_head_overrides"](True)
    # Inference-only engines never request the override, so an old vLLM must not block them.
    assert ns["_vllm_fp32_lm_head_overrides"](False) == {}


def test_load_vllm_merges_override_before_arg_filter():
    load_vllm = next(
        n for n in ast.walk(_vllm_utils_tree()) if isinstance(n, ast.FunctionDef) and n.name == "load_vllm"
    )
    src = ast.unparse(load_vllm)
    call = src.find("_vllm_fp32_lm_head_overrides(training)")
    merge = src.find("**fp32_head_overrides")
    arg_filter = src.find("good_keys = inspect.signature(AsyncEngineArgs", merge)
    assert -1 < call < merge < arg_filter
    # Merged, not assigned, so the rope / gemma4 / torchao overrides set earlier survive.
    assert "engine_args.get('hf_overrides', {})" in src[call:arg_filter]

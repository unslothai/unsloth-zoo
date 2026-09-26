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

"""GRPO reuses load_vllm's engine, so its sampling logprobs must be temperature
scaled like the training logprobs (TRL builds its own engine with
logprobs_mode="processed_logprobs"); vLLM's raw_logprobs default skews the
importance sampling ratio and off-policy mask. AST-level: load_vllm needs a GPU.
"""

import ast
import pathlib

import pytest


def _engine_args_logprobs_mode() -> ast.expr:
    path = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo" / "vllm_utils.py"
    tree = ast.parse(path.read_text(encoding = "utf-8-sig"))
    load_vllm = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "load_vllm"
    )
    for node in ast.walk(load_vllm):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "engine_args" for t in node.targets
        ) and isinstance(node.value, ast.Call) and getattr(node.value.func, "id", "") == "dict":
            for kw in node.value.keywords:
                if kw.arg == "logprobs_mode":
                    return kw.value
    raise AssertionError("load_vllm's engine_args no longer sets logprobs_mode")


@pytest.mark.parametrize("training, expected", [
    (True,  "processed_logprobs"),
    (False, "raw_logprobs"),
])
def test_logprobs_mode_follows_training(training, expected):
    expr = ast.Expression(_engine_args_logprobs_mode())
    got = eval(compile(expr, "<logprobs_mode>", "eval"), {}, {"training": training})
    assert got == expected


def test_modes_are_valid_vllm_values():
    model_config = pytest.importorskip("vllm.config.model")
    import typing
    modes = set(typing.get_args(getattr(model_config, "LogprobsMode", None)))
    if not modes:
        pytest.skip("this vLLM predates logprobs_mode; load_vllm drops the key")
    assert {"processed_logprobs", "raw_logprobs"} <= modes


def test_v1_sampler_keeps_fast_topk_topp_when_no_logprobs(monkeypatch):
    sampler_mod = pytest.importorskip("vllm.v1.sample.sampler")
    from types import SimpleNamespace
    import unsloth_zoo.vllm_utils as vu

    seen = []
    monkeypatch.setattr(
        sampler_mod.Sampler, "forward",
        lambda self, logits, md, *a, **k: seen.append(self.topk_topp_sampler.logprobs_mode),
    )
    vu.patch_vllm_processed_logprobs_fast_path()
    s = sampler_mod.Sampler(logprobs_mode = "processed_logprobs")
    s.forward(None, SimpleNamespace(max_num_logprobs = None))
    s.forward(None, SimpleNamespace(max_num_logprobs = 0))
    s.forward(None, SimpleNamespace(max_num_logprobs = None), logprobs_mode_override = "processed_logits")
    assert seen == ["raw_logprobs", "processed_logprobs", "processed_logprobs"]
    assert s.topk_topp_sampler.logprobs_mode == "processed_logprobs"

    raw = sampler_mod.Sampler(logprobs_mode = "raw_logprobs")
    raw.forward(None, SimpleNamespace(max_num_logprobs = None))
    assert seen[-1] == "raw_logprobs" and "_unsloth_raw_topk_topp_sampler" not in raw.__dict__

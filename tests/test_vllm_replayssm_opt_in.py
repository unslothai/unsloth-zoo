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

"""UNSLOTH_VLLM_REPLAYSSM=1 gating in load_vllm. AST-level: load_vllm needs a GPU."""

import ast
import importlib.util
import inspect
import pathlib
from types import SimpleNamespace

import pytest

VLLM_UTILS = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo" / "vllm_utils.py"
TREE = ast.parse(VLLM_UTILS.read_text(encoding = "utf-8-sig"))
HELPERS = ("_vllm_replayssm_skip_reason", "_is_vllm_replayssm_refusal")


def _helpers():
    nodes = [
        n for n in TREE.body
        if (isinstance(n, ast.FunctionDef) and n.name in HELPERS)
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_REPLAYSSM_MODEL_TYPES" for t in n.targets))
    ]
    assert len(nodes) == len(HELPERS) + 1
    ns = {}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(VLLM_UTILS), "exec"), ns)
    return ns


NS = _helpers()
KEYS = {"model", "use_replayssm"}


def _config(model_type, nested = False):
    inner = SimpleNamespace(model_type = model_type)
    return SimpleNamespace(model_type = "wrapper", text_config = inner) if nested else inner


@pytest.mark.parametrize("nested", [False, True])
def test_nemotron_h_on_cuda_is_requested(nested):
    assert NS["_vllm_replayssm_skip_reason"](_config("nemotron_h", nested), KEYS, "cuda") is None


@pytest.mark.parametrize("config, keys, device, why", [
    (_config("qwen3_next"), KEYS, "cuda", "model type"),
    (_config("llama"), KEYS, "cuda", "model type"),
    (_config("nemotron_h"), KEYS, "hip", "CUDA only"),
    (_config("nemotron_h"), KEYS, "xpu", "CUDA only"),
    (_config("nemotron_h"), {"model"}, "cuda", "no `use_replayssm`"),
])
def test_skip_reasons(config, keys, device, why):
    assert why in NS["_vllm_replayssm_skip_reason"](config, keys, device)


def test_refusal_matcher_rejects_unrelated_errors():
    is_refusal = NS["_is_vllm_replayssm_refusal"]
    assert not is_refusal("CUDA out of memory. Tried to allocate 2.00 GiB")
    assert not is_refusal("Free memory on device cuda:0 (1.2/80 GiB) on startup is less than desired")


def _load_vllm():
    return next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "load_vllm")


def test_requested_only_for_engines_load_vllm_builds():
    # return_args callers (synthetic data `vllm serve`) never get the flag without the runner env.
    fn = _load_vllm()
    return_args_line = next(
        n.lineno for n in ast.walk(fn)
        if isinstance(n, ast.If) and getattr(n.test, "id", "") == "return_args"
    )
    request_lines = [
        n.lineno for n in ast.walk(fn)
        if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Subscript) and getattr(t.slice, "value", None) == "use_replayssm"
            for t in n.targets
        )
    ]
    assert request_lines and min(request_lines) > return_args_line
    src = ast.get_source_segment(VLLM_UTILS.read_text(encoding = "utf-8-sig"), fn)
    assert 'os.environ.get("UNSLOTH_VLLM_REPLAYSSM", "0") == "1"' in src
    assert '"VLLM_USE_V2_MODEL_RUNNER" not in os.environ' in src


# Drift guards against the installed vLLM; skipped where vLLM is absent or predates ReplaySSM.
def _vllm_file(*parts):
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("vLLM not installed")
    path = pathlib.Path(list(spec.submodule_search_locations)[0]).joinpath(*parts)
    if not path.exists():
        pytest.skip(f"{path} not in this vLLM")
    return ast.parse(path.read_text(encoding = "utf-8"))


def test_refusal_matcher_matches_vllm_replayssm_errors():
    tree = _vllm_file("config", "vllm.py")
    messages = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            text = "".join(
                c.value for a in node.exc.args for c in ast.walk(a)
                if isinstance(c, ast.Constant) and isinstance(c.value, str)
            )
            if "ReplaySSM" in text:
                messages.append(text)
    if not messages:
        pytest.skip("this vLLM has no ReplaySSM validation")
    assert all(NS["_is_vllm_replayssm_refusal"](m) for m in messages)


@pytest.mark.parametrize("model_type", NS["_REPLAYSSM_MODEL_TYPES"])
def test_model_types_declare_supports_replayssm(model_type):
    tree = _vllm_file("model_executor", "models", f"{model_type}.py")
    bases = {
        getattr(b, "id", getattr(b, "attr", None))
        for n in ast.walk(tree) if isinstance(n, ast.ClassDef) for b in n.bases
    }
    if "SupportsReplaySSM" not in bases and "SupportsReplaySSM" not in ast.dump(tree):
        pytest.skip("this vLLM predates ReplaySSM")
    assert "SupportsReplaySSM" in bases


def test_engine_arg_name_matches_vllm():
    try:
        from vllm.engine.arg_utils import EngineArgs
    except Exception as e:
        pytest.skip(f"vLLM EngineArgs unavailable: {e}")
    params = inspect.signature(EngineArgs).parameters
    if not any("replayssm" in p for p in params):
        pytest.skip("this vLLM predates ReplaySSM")
    assert "use_replayssm" in params

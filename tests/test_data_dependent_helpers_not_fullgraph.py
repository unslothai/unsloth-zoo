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

"""Helpers doing .nonzero() / .tolist() / .item() must not be emitted fullgraph = True."""

import ast
import importlib.util
import json
import os
import subprocess
import sys
import textwrap

import pytest

from unsloth_zoo.compiler import data_dependent_helpers, has_data_dependent_call


def test_has_data_dependent_call():
    assert has_data_dependent_call("    n = x.sum().item()\n")
    assert has_data_dependent_call("    sizes = grid_thw.tolist()\n")
    assert has_data_dependent_call("    idx = (x == 2).nonzero()\n")
    assert not has_data_dependent_call("    return x * 2 + y.sum()\n")


def _write_and_import(tmp_path, name, source):
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(source), encoding = "utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_data_dependent_helpers_selects_only_readable_data_dependent_functions(tmp_path):
    module = _write_and_import(tmp_path, "synthetic_dd_helpers", """
        import torch

        def uses_item(x):
            return int(x.max().item())

        def uses_tolist(x, lengths):
            return x.split(lengths.tolist(), dim = 0)

        def plain(x):
            return x * 2

        class LooksLikeAHelper:
            def forward(self, x):
                return x.item()

        # Rebound to objects inspect.getsource cannot read: must be skipped, not raise.
        def sourceless(x):
            return x.item()
        sourceless = eval(compile("lambda x: x.item()", "<generated>", "eval"))

        def builtin_alias(x):
            return x.item()
        builtin_alias = len
    """)
    called = [
        "uses_item", "uses_tolist", "plain", "LooksLikeAHelper",
        "sourceless", "builtin_alias", "not_an_attribute",
    ]
    assert data_dependent_helpers(module, called) == ["uses_item", "uses_tolist"]


def _run_compiler_child(script, cwd):
    """Own interpreter: the rewriter patches the modeling module in place (one-way)."""
    env = dict(os.environ)
    env["UNSLOTH_ALLOW_CPU"] = "1"
    env["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    # Pin the child to THIS checkout, else it reports on the site-packages compiler.py.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    # Either would change the decorators emitted and make the assertions meaningless.
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    env.pop("UNSLOTH_FULLGRAPH", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd = str(cwd),
        env = env,
        capture_output = True,
        text = True,
        timeout = 1800,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    payload = None
    for line in result.stdout.splitlines():
        if line.startswith("@@@"):
            payload = json.loads(line[3:])
    assert payload is not None, result.stdout[-4000:]
    return payload


def _decorators(generated):
    tree = ast.parse(generated)
    return {
        node.name: "\n".join(ast.get_source_segment(generated, d) for d in node.decorator_list)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _fullgraph_functions_calling(generated, names):
    tree = ast.parse(generated)
    offenders = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        decorators = " ".join(ast.get_source_segment(generated, d) for d in node.decorator_list)
        if "fullgraph = True" not in decorators:
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id in names
            ):
                offenders.append((node.name, inner.func.id))
    return offenders


_COMPILE = r'''
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    unsloth_compile_transformers(
        model_type = MT, fast_lora_forwards = False, fullgraph = True,
        import_from_cache = False, disable = False, supports_sdpa = [None],
    )
generated = open(
    os.path.join("unsloth_compiled_cache", f"unsloth_compiled_module_{MT}.py"),
    encoding = "utf-8",
).read()
'''


# Real package on disk so inspect.getsource works and the compiler's import_module finds it.
_SYNTHETIC_CHILD = r'''
import os, sys, json, importlib, io, contextlib, textwrap
os.environ["UNSLOTH_COMPILE_LOCATION"] = "unsloth_compiled_cache"
import torch
import transformers.models
from unsloth_zoo.compiler import unsloth_compile_transformers

MT = "unsloth_synthetic_dd"
root = os.path.join(os.getcwd(), "synthetic_transformers_models")
package = os.path.join(root, MT)
os.makedirs(package)
open(os.path.join(package, "__init__.py"), "w").close()
with open(os.path.join(package, f"modeling_{MT}.py"), "w", encoding = "utf-8") as handle:
    handle.write(textwrap.dedent("""
        import torch
        import torch.nn as nn


        def synthetic_item_helper(hidden_states, lengths):
            longest = lengths.max().item()
            return hidden_states[:, :longest]


        def synthetic_plain_helper(hidden_states):
            return hidden_states * 2


        def synthetic_helper_caller(hidden_states, lengths):
            return synthetic_item_helper(hidden_states, lengths) + 1


        class SyntheticDataDependentCaller(nn.Module):
            def __init__(self, hidden_size = 8):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(hidden_size))

            def forward(self, hidden_states, lengths):
                return synthetic_item_helper(hidden_states, lengths) * self.scale


        # Only here so synthetic_helper_caller is called, which puts it in called_functions.
        class SyntheticIndirectCaller(nn.Module):
            def __init__(self, hidden_size = 8):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(hidden_size))

            def forward(self, hidden_states, lengths):
                return synthetic_helper_caller(hidden_states, lengths) * self.scale


        class SyntheticPlainCaller(nn.Module):
            def __init__(self, hidden_size = 8):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(hidden_size))

            def forward(self, hidden_states):
                return synthetic_plain_helper(hidden_states) * self.scale
    """))

transformers.models.__path__.append(root)
modeling_file = importlib.import_module(f"transformers.models.{MT}.modeling_{MT}")
''' + _COMPILE + r'''
print("@@@" + json.dumps({
    "generated": generated,
    "modeling_file": modeling_file.__file__,
    "log": buf.getvalue()[-4000:],
}))
'''


def test_synthetic_data_dependent_helper_is_disabled_and_its_callers_demoted(tmp_path):
    payload = _run_compiler_child(_SYNTHETIC_CHILD, tmp_path)
    assert payload["modeling_file"].startswith(str(tmp_path)), payload["modeling_file"]
    generated = payload["generated"]
    decorators = _decorators(generated)
    context = "\n--- compiler log ---\n" + payload["log"]

    assert "torch.compiler.disable(recursive = False)" in decorators.get(
        "synthetic_item_helper", ""
    ), (
        "synthetic_item_helper calls lengths.max().item(); emitting it fullgraph = True "
        "is a hard GuardOnDataDependentSymNode error on its first call.\n"
        + decorators.get("synthetic_item_helper", "<not emitted>") + context
    )
    assert "fullgraph = True" in decorators.get("synthetic_plain_helper", ""), (
        decorators.get("synthetic_plain_helper", "<not emitted>") + context
    )
    assert "fullgraph = False" in decorators.get("synthetic_helper_caller", ""), (
        decorators.get("synthetic_helper_caller", "<not emitted>") + context
    )
    assert "fullgraph = False" in decorators.get("SyntheticDataDependentCaller_forward", ""), (
        decorators.get("SyntheticDataDependentCaller_forward", "<not emitted>") + context
    )
    assert "fullgraph = True" in decorators.get("SyntheticPlainCaller_forward", ""), (
        decorators.get("SyntheticPlainCaller_forward", "<not emitted>") + context
    )


_QWEN3_OMNI = "qwen3_omni_moe"
_QWEN3_OMNI_HELPERS = ["chunk_and_pad_features", "get_valid_indices", "get_audio_cu_seqlens"]


def _qwen3_omni_defines_the_helpers():
    """Read, not import, so no half-patched module leaks out."""
    try:
        spec = importlib.util.find_spec(
            f"transformers.models.{_QWEN3_OMNI}.modeling_{_QWEN3_OMNI}"
        )
    except Exception:
        return False
    if spec is None or not spec.origin:
        return False
    try:
        with open(spec.origin, encoding = "utf-8") as handle:
            source = handle.read()
    except OSError:
        return False
    return "def chunk_and_pad_features(" in source


_QWEN3_OMNI_CHILD = r'''
import os, sys, json, importlib, io, contextlib
os.environ["UNSLOTH_COMPILE_LOCATION"] = "unsloth_compiled_cache"
import torch
from unsloth_zoo.compiler import unsloth_compile_transformers

MT = "qwen3_omni_moe"
modeling_file = importlib.import_module(f"transformers.models.{MT}.modeling_{MT}")
original = modeling_file.chunk_and_pad_features
input_features = torch.randn(128, 300)
feature_lens = torch.tensor([200, 100])
expected = original(input_features, feature_lens, 50)
''' + _COMPILE + r'''
# The rewriter swaps the emitted helper back into the modeling module, so this is the
# function the audio encoder calls.
patched = modeling_file.chunk_and_pad_features
error = None
matches = None
try:
    got = patched(input_features, feature_lens, 50)
    matches = all(torch.equal(a, b) for a, b in zip(got, expected))
except Exception as exception:
    error = f"{type(exception).__name__}: {str(exception).strip().splitlines()[0][:300]}"

print("@@@" + json.dumps({
    "generated": generated,
    "replaced": patched is not original,
    "error": error,
    "matches": matches,
    "log": buf.getvalue()[-4000:],
}))
'''


@pytest.mark.skipif(
    not _qwen3_omni_defines_the_helpers(),
    reason = "the installed transformers does not define qwen3_omni_moe.chunk_and_pad_features",
)
def test_qwen3_omni_audio_helpers_are_not_fullgraph(tmp_path):
    payload = _run_compiler_child(_QWEN3_OMNI_CHILD, tmp_path)
    generated = payload["generated"]
    decorators = _decorators(generated)
    context = "\n--- compiler log ---\n" + payload["log"]

    for name in _QWEN3_OMNI_HELPERS:
        assert "torch.compiler.disable(recursive = False)" in decorators.get(name, ""), (
            f"{name} pulls tensor data into Python (.tolist() / .item() / .nonzero()) "
            f"but was emitted as:\n{decorators.get(name, '<not emitted>')}" + context
        )
    assert _fullgraph_functions_calling(generated, set(_QWEN3_OMNI_HELPERS)) == []

    assert "fullgraph = True" in decorators.get("_get_feat_extract_output_lengths", ""), (
        decorators.get("_get_feat_extract_output_lengths", "<not emitted>") + context
    )

    assert payload["replaced"], "the compiled cache did not replace chunk_and_pad_features"
    assert payload["error"] is None, payload["error"]
    assert payload["matches"] is True

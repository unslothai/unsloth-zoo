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

"""A module whose __init__ builds a compile-disabled attention module must stay eager.

Compiled, it graph-breaks at every call ("Skip inlining torch.compiler.disable()d function"):
GraniteSpeechConformerBlock (4 per step on the tiny model), Blip2QFormerAttention (12),
Gemma3nAudioConformerAttention / Gemma3nAudioConformerBlock (6). Its other compiled
children still compile.
"""

import ast
import json
import os
import subprocess
import sys

import pytest


def test_wrappers_are_found_transitively():
    from unsloth_zoo.compiler import modules_wrapping_disabled_attention

    inits = {
        "ConformerAttention": "self.attn = AudioAttention(config)",
        "ConformerBlock": "self.attention = ConformerAttention(config)\nself.ffw = FeedForward(config)",
        "Encoder": "self.blocks = nn.ModuleList([ConformerBlock(config) for _ in range(n)])",
        "FeedForward": "self.fc = nn.Linear(4, 4)",
        "AudioAttentionPool": "self.pool = AudioAttentionPooler(config)",
    }
    found = modules_wrapping_disabled_attention(inits, ["AudioAttention"])
    assert found == {"ConformerAttention", "ConformerBlock", "Encoder"}


def test_nothing_disabled_changes_nothing():
    from unsloth_zoo.compiler import modules_wrapping_disabled_attention

    assert modules_wrapping_disabled_attention({"A": "self.b = B(config)"}, []) == set()


_CHILD = r"""
import contextlib, io, json, os, sys
os.environ["UNSLOTH_COMPILE_LOCATION"] = "unsloth_compiled_cache"
from unsloth_zoo.compiler import unsloth_compile_transformers
model_type = sys.argv[1]
with contextlib.redirect_stdout(io.StringIO()):
    unsloth_compile_transformers(
        model_type = model_type, fast_lora_forwards = False, fullgraph = True,
        import_from_cache = False, disable = False, supports_sdpa = [None],
    )
path = f"unsloth_compiled_cache/unsloth_compiled_module_{model_type}.py"
print("@@@" + json.dumps(open(path, encoding = "utf-8").read()))
"""


def _decorators(tmp_path, model_type):
    env = dict(os.environ, UNSLOTH_ALLOW_CPU = "1", UNSLOTH_ZOO_DISABLE_GPU_INIT = "1", CUDA_VISIBLE_DEVICES = "")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join([repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    env.pop("UNSLOTH_FULLGRAPH", None)
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, model_type],
        cwd = tmp_path, env = env, capture_output = True, text = True, timeout = 1800,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    generated = json.loads(next(line[3:] for line in result.stdout.splitlines() if line.startswith("@@@")))
    tree = ast.parse(generated)
    return {
        node.name: " ".join(ast.get_source_segment(generated, d) for d in node.decorator_list)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


@pytest.mark.parametrize(
    "model_type, disabled, eager, compiled",
    [
        (
            "granite_speech",
            "GraniteSpeechConformerAttention",
            ["GraniteSpeechConformerBlock"],
            ["GraniteSpeechConformerFeedForward", "GraniteSpeechConformerConvModule"],
        ),
        (
            "gemma3n",
            "Gemma3nAudioAttention",
            ["Gemma3nAudioConformerAttention", "Gemma3nAudioConformerBlock"],
            ["Gemma3nAudioConformerFeedForward", "Gemma3nAudioCumulativeGroupNorm"],
        ),
        (
            "blip_2",
            "Blip2QFormerMultiHeadAttention",
            ["Blip2QFormerAttention"],
            ["Blip2QFormerSelfOutput", "Blip2QFormerIntermediate", "Blip2QFormerOutput"],
        ),
    ],
)
def test_disabled_attention_wrappers_are_not_compiled(tmp_path, model_type, disabled, eager, compiled):
    pytest.importorskip(f"transformers.models.{model_type}.modeling_{model_type}")
    decorators = _decorators(tmp_path, model_type)
    assert "torch_compiler_disable_unless_decode" in decorators[f"{disabled}_forward"]
    for name in eager:
        assert "torch_compile" not in decorators.get(f"{name}_forward", ""), (name, decorators.get(f"{name}_forward"))
    for name in compiled:
        assert "torch_compile_with_fallback" in decorators.get(f"{name}_forward", ""), name

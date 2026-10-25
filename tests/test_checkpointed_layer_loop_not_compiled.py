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

"""A torch module looping over `GradientCheckpointingLayer`s must not be compiled.

Under gradient checkpointing each layer call goes through the checkpoint function, which
Dynamo skips, so a compiled Wav2Vec2FeatureEncoder graph broke once per conv layer plus
once on its `hidden_states.requires_grad = True`.
"""

import inspect
import json
import os
import re
import subprocess
import sys

import pytest

wav2vec2 = pytest.importorskip("transformers.models.wav2vec2.modeling_wav2vec2")

from unsloth_zoo.compiler import loops_over_checkpointed_layers


def _check(module, name):
    cls = getattr(module, name)
    return loops_over_checkpointed_layers(
        module, inspect.getsource(cls.__init__), inspect.getsource(cls.forward)
    )


def test_loops_over_checkpointed_layers():
    assert _check(wav2vec2, "Wav2Vec2FeatureEncoder")
    # No loop / no checkpointed children.
    assert not _check(wav2vec2, "Wav2Vec2FeatureProjection")
    assert not _check(wav2vec2, "Wav2Vec2FeedForward")
    assert not _check(wav2vec2, "Wav2Vec2PositionalConvEmbedding")


_CHILD = r'''
import os, io, contextlib, json
os.environ["UNSLOTH_COMPILE_LOCATION"] = "unsloth_compiled_cache"
from unsloth_zoo.compiler import unsloth_compile_transformers
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    unsloth_compile_transformers(
        model_type = "wav2vec2", fast_lora_forwards = False, fullgraph = True,
        import_from_cache = False, disable = False, supports_sdpa = [None],
    )
generated = open(
    os.path.join("unsloth_compiled_cache", "unsloth_compiled_module_wav2vec2.py"),
    encoding = "utf-8",
).read()
print("@@@" + json.dumps({"generated": generated, "log": buf.getvalue()}))
'''


def test_wav2vec2_feature_encoder_not_compiled(tmp_path):
    env = dict(os.environ)
    env["UNSLOTH_ALLOW_CPU"] = "1"
    env["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    env.pop("UNSLOTH_FULLGRAPH", None)
    result = subprocess.run(
        [sys.executable, "-c", _CHILD], cwd = str(tmp_path), env = env,
        capture_output = True, text = True, timeout = 1800,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    payload = None
    for line in result.stdout.splitlines():
        if line.startswith("@@@"):
            payload = json.loads(line[3:])
    assert payload is not None, result.stdout[-4000:]
    generated = payload["generated"]

    def compiled(name):
        return re.search(
            r"@torch_compile_with_fallback\([^\n]*\)\ndef " + name + r"_forward\(", generated
        ) is not None

    assert not compiled("Wav2Vec2FeatureEncoder"), payload["log"][-3000:]
    # Siblings without a checkpointed-layer loop keep compiling.
    assert compiled("Wav2Vec2FeatureProjection")
    assert compiled("Wav2Vec2FeedForward")

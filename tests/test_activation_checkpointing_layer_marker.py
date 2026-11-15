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

"""Layers subclassing ActivationCheckpointingLayer (transformers#49249) take the GradientCheckpointingLayer path (unsloth-zoo#2398)."""

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("transformers.models.vit.modeling_vit")
pytest.importorskip("transformers.models.whisper.modeling_whisper")


_CHILD = r'''
import contextlib, importlib, inspect, io, json, os, sys
os.environ["UNSLOTH_COMPILE_LOCATION"] = "unsloth_compiled_cache"
marker = sys.argv[1]
targets = {
    importlib.import_module("transformers.models.vit.modeling_vit"),
    importlib.import_module("transformers.models.whisper.modeling_whisper"),
}
_getsource = inspect.getsource
def getsource(obj):
    source = _getsource(obj)
    if obj in targets:
        source = source.replace("(GradientCheckpointingLayer)", "(" + marker + ")")
    return source
inspect.getsource = getsource
from unsloth_zoo.compiler import unsloth_compile_transformers
payload = {}
for model_type in ("vit", "whisper"):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        unsloth_compile_transformers(
            model_type = model_type, fast_lora_forwards = False,
            import_from_cache = False, disable = False, supports_sdpa = [None],
        )
    generated = open(
        os.path.join("unsloth_compiled_cache", f"unsloth_compiled_module_{model_type}.py"),
        encoding = "utf-8",
    ).read()
    payload[model_type] = {"generated": generated, "log": buf.getvalue()}
print("@@@" + json.dumps(payload))
'''


@pytest.mark.parametrize(
    "marker", ["GradientCheckpointingLayer", "ActivationCheckpointingLayer"]
)
def test_checkpointing_layer_marker(tmp_path, marker):
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
        [sys.executable, "-c", _CHILD, marker], cwd = str(tmp_path), env = env,
        capture_output = True, text = True, timeout = 1800,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    payload = None
    for line in result.stdout.splitlines():
        if line.startswith("@@@"):
            payload = json.loads(line[3:])
    assert payload is not None, result.stdout[-4000:]
    vit, whisper = payload["vit"], payload["whisper"]

    # ViTLayer checkpoints itself, so ViTModel's layer loop must not be wrapped in a second checkpoint.
    assert "def ViTModel_forward(" not in vit["generated"], vit["log"][-3000:]
    assert "Patched ViTModel by adding gradient checkpointing" not in vit["log"]
    assert "def WhisperDecoder_forward(" in whisper["generated"], whisper["log"][-3000:]
    assert "with gradient checkpointing" not in whisper["log"], whisper["log"][-3000:]

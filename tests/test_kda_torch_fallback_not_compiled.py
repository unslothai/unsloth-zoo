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

"""Kimi delta attention torch fallbacks (glm5_next, kimi_linear) must stay eager: compiled, the chunk loop
took 25+ minutes of AOT compile on GLM-5.3-Flash's first checkpoint replay."""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

from unsloth_zoo.compiler import DISABLE_COMPILE_FUNCTIONS

KDA = ("chunk_kimi_delta_attention", "recurrent_kimi_delta_attention")


def test_kda_fallbacks_are_listed():
    for name in KDA:
        assert name in DISABLE_COMPILE_FUNCTIONS


_CHILD = r'''
import os, sys, json, io, contextlib
os.environ["UNSLOTH_COMPILE_LOCATION"] = "unsloth_compiled_cache"
from unsloth_zoo.compiler import unsloth_compile_transformers
MT = sys.argv[1]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    unsloth_compile_transformers(
        model_type = MT, fast_lora_forwards = False, fullgraph = True,
        import_from_cache = False, disable = False, supports_sdpa = [None],
    )
src = open(os.path.join("unsloth_compiled_cache", f"unsloth_compiled_module_{MT}.py")).read()
out = {}
for name in ("chunk_kimi_delta_attention", "recurrent_kimi_delta_attention"):
    at = src.find(f"def {name}(")
    head = src[:at].rstrip().rsplit("\n\n", 1)[-1] if at != -1 else None
    out[name] = head
print("@@@" + json.dumps(out))
'''


@pytest.mark.parametrize("model_type", ["glm5_next", "kimi_linear"])
def test_kda_fallbacks_not_wrapped_in_torch_compile(model_type, tmp_path):
    try:
        spec = importlib.util.find_spec(f"transformers.models.{model_type}.modeling_{model_type}")
    except ModuleNotFoundError:  # parent package missing on older transformers
        spec = None
    if spec is None:
        pytest.skip(reason = f"installed transformers predates {model_type}")
    env = dict(os.environ)
    env["UNSLOTH_ALLOW_CPU"] = "1"
    env["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join([repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    r = subprocess.run([sys.executable, "-c", _CHILD, model_type], cwd = str(tmp_path), env = env,
                       capture_output = True, text = True, timeout = 1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    payload = next(json.loads(l[3:]) for l in r.stdout.splitlines() if l.startswith("@@@"))
    for name, head in payload.items():
        if head is None:
            continue  # imported, not emitted from source
        assert "torch_compile_with_fallback" not in head and "torch.compile(" not in head, (name, head)

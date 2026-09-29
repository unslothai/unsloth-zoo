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

"""Routers rewritten for the routing-weights cast stay compiled, and the compiler swaps dict entries only
for the exact class it replaced (a substring match turned Jamba's attention decoder layer into bare attention)."""

import json
import os
import subprocess
import sys

import pytest

_CHILD = r'''
import os, io, json, contextlib, importlib
os.environ["UNSLOTH_COMPILE_LOCATION"] = os.path.abspath("unsloth_compiled_cache")
from unsloth_zoo.compiler import unsloth_compile_transformers

def decorator_of(generated, name):
    where = generated.find(f"def {name}(")
    return None if where == -1 else generated[:where].rsplit("\n@", 1)[-1].strip()

out = {}
for mt, router in (("ernie4_5_moe", "Ernie4_5_MoeTopKRouter"), ("laguna", "LagunaTopKRouter"), ("jamba", None)):
    try:
        modeling = importlib.import_module(f"transformers.models.{mt}.modeling_{mt}")
    except Exception:
        continue
    # transformers 4.57 routes Ernie 4.5 inside its MoE block: no router class to check.
    if router is not None and not hasattr(modeling, router):
        continue
    with contextlib.redirect_stdout(io.StringIO()):
        unsloth_compile_transformers(
            model_type = mt, fast_lora_forwards = False, fullgraph = True,
            import_from_cache = False, disable = False, supports_sdpa = [None],
        )
    path = os.path.join(os.environ["UNSLOTH_COMPILE_LOCATION"], f"unsloth_compiled_module_{mt}.py")
    generated = open(path, encoding = "utf-8").read()
    if router is not None:
        out[mt] = decorator_of(generated, router + "_forward")
    else:
        out[mt] = {k: v.__name__ for k, v in modeling.ALL_DECODER_LAYER_TYPES.items()}
print("RESULT " + json.dumps(out))
'''


def test_cast_routers_compiled_and_decoder_layer_types_kept(tmp_path):
    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join([repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("UNSLOTH_ZOO_DISABLE_GPU_INIT", "1")
    proc = subprocess.run([sys.executable, "-c", _CHILD], cwd = tmp_path, capture_output = True, text = True,
                          timeout = 900, env = env)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line is not None, proc.stdout[-3000:] + proc.stderr[-3000:]
    r = json.loads(line[len("RESULT "):])
    if not r:
        pytest.skip(reason = "transformers has none of the Ernie 4.5 / Laguna routers or Jamba")
    for mt in ("ernie4_5_moe", "laguna"):
        if mt in r:
            assert r[mt] is not None and r[mt].startswith("torch_compile_with_fallback("), (mt, r[mt])
    if "jamba" in r:
        assert r["jamba"] == {"attention": "JambaAttentionDecoderLayer", "mamba": "JambaMambaDecoderLayer"}, r["jamba"]

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

"""Compiled module must parse with `import copy` and `state.copy()`; stdlib source not pasted."""

import ast
import json
import os
import subprocess
import sys

_CHILD = r'''
import os, sys, json, importlib, io, contextlib, textwrap
os.environ["UNSLOTH_COMPILE_LOCATION"] = os.path.abspath("unsloth_compiled_cache")
import torch
import transformers.models
from unsloth_zoo.compiler import unsloth_compile_transformers

MT = "unsloth_synthetic_module_named_like_method"
root = os.path.join(os.getcwd(), "synthetic_transformers_models")
package = os.path.join(root, MT)
os.makedirs(package)
open(os.path.join(package, "__init__.py"), "w").close()
with open(os.path.join(package, f"modeling_{MT}.py"), "w", encoding = "utf-8") as handle:
    handle.write(textwrap.dedent("""
        import copy

        import torch
        import torch.nn as nn

        __all__ = ["SyntheticModel"]


        class _State:
            def __init__(self):
                self.values = {}

            def copy(self):
                out = _State()
                out.values = copy.copy(self.values)
                return out


        class SyntheticModel(nn.Module):
            def __init__(self, hidden_size = 8):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.state = _State()

            def forward(self, hidden_states):
                state = self.state.copy()
                state.values["scale"] = 2.0
                return hidden_states * self.weight * state.values["scale"]
    """))
transformers.models.__path__.append(root)
importlib.import_module(f"transformers.models.{MT}.modeling_{MT}")
error = None
try:
    with contextlib.redirect_stdout(io.StringIO()):
        unsloth_compile_transformers(
            model_type = MT, fast_lora_forwards = False, fullgraph = True,
            import_from_cache = False, disable = True, supports_sdpa = [None],
        )
except Exception as exception:
    error = f"{type(exception).__name__}: {exception}"
path = os.path.join(os.environ["UNSLOTH_COMPILE_LOCATION"], f"unsloth_compiled_module_{MT}.py")
print("@@@" + json.dumps({"error": error, "generated": open(path, encoding = "utf-8").read() if os.path.exists(path) else None}))
'''


def test_generated_module_parses_when_an_imported_module_shares_a_method_name(tmp_path):
    env = dict(os.environ)
    env["UNSLOTH_ALLOW_CPU"] = "1"
    env["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join([repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    result = subprocess.run(
        [sys.executable, "-c", _CHILD], cwd = str(tmp_path), env = env, capture_output = True, text = True,
        timeout = 900,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    payload = json.loads(next(l for l in result.stdout.splitlines() if l.startswith("@@@"))[3:])
    generated = payload["generated"]
    assert generated is not None, payload["error"]
    ast.parse(generated)  # SyntaxError when a module's source was pasted in
    assert "Generic (shallow and deep) copying operations" not in generated
    assert payload["error"] is None, payload["error"]

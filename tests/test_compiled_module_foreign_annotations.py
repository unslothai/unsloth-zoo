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

"""unslothai/unsloth#13245: imported closure re-emitted with unimported annotations (NameError)."""

import importlib.util
import inspect
import json
import os
import subprocess
import sys
import types

from unsloth_zoo.compiler import (
    function_defined_elsewhere,
    signature_with_string_annotations,
)


def test_signature_annotations_are_quoted():
    namespace = {}
    exec(
        "import collections.abc, typing\n"
        "class Cache: pass\n"
        "def f(a: Cache, b: Cache | None, c: collections.abc.Callable | None = None, d = 1, *args: int, **kwargs) -> Cache: pass\n",
        namespace,
    )
    rendered = str(signature_with_string_annotations(inspect.signature(namespace["f"])))
    exec(f"def g{rendered}: pass", {})
    assert "d=1" in rendered.replace(" ", "")
    assert rendered.count("'") >= 8, rendered


def test_function_defined_elsewhere():
    module = types.ModuleType("fake_modeling")
    module.__file__ = __file__
    module.local = test_function_defined_elsewhere
    module.foreign = json.dumps
    module.unknown = len
    assert not function_defined_elsewhere(module, "local")
    assert function_defined_elsewhere(module, "foreign")
    assert not function_defined_elsewhere(module, "unknown")
    assert not function_defined_elsewhere(module, "missing")


_CHILD = r'''
import os, sys, json, importlib, io, contextlib, textwrap, inspect
os.environ["UNSLOTH_COMPILE_LOCATION"] = "unsloth_compiled_cache"
import torch
import transformers.models
from unsloth_zoo.compiler import unsloth_compile_transformers

MT = "unsloth_synthetic_foreign_annotations"
root = os.path.join(os.getcwd(), "synthetic_transformers_models")
package = os.path.join(root, MT)
os.makedirs(package)
open(os.path.join(package, "__init__.py"), "w").close()

# A sibling module the generated file never imports: str(signature) renders its classes as
# `transformers.models.<MT>.synthetic_utils.SyntheticCache`.
with open(os.path.join(package, "synthetic_utils.py"), "w", encoding = "utf-8") as handle:
    handle.write(textwrap.dedent("""
        import inspect
        import collections.abc
        import torch

        class SyntheticCache:
            pass

        def build_mask(inputs_embeds: torch.Tensor, cache: SyntheticCache | None, mask_fn: collections.abc.Callable | None = None, unused_flag = None):
            return inputs_embeds.new_ones(inputs_embeds.shape[:-1])

        # The temporary_patches/misc.py shape: a closure that borrows the signature.
        def wrap(original):
            hidden = 1
            def wrapper(*args, **kwargs):
                return original(*args, **kwargs) * hidden
            wrapper.__signature__ = inspect.signature(original)
            return wrapper

        build_mask = wrap(build_mask)
    """))

with open(os.path.join(package, f"modeling_{MT}.py"), "w", encoding = "utf-8") as handle:
    handle.write(textwrap.dedent("""
        import torch
        import torch.nn as nn

        from .synthetic_utils import SyntheticCache, build_mask

        __all__ = ["SyntheticModel"]


        def local_helper(hidden_states: torch.Tensor, cache: SyntheticCache | None, unused_arg = None, scale = None):
            return hidden_states * 2


        class SyntheticModel(nn.Module):
            def __init__(self, hidden_size = 8):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))

            @staticmethod
            def build_mask(inputs_embeds, cache = None):
                return build_mask(inputs_embeds, cache)

            def forward(self, hidden_states):
                mask = self.build_mask(hidden_states)
                hidden_states = local_helper(hidden_states, None)
                return hidden_states * self.weight + mask.unsqueeze(-1)
    """))

transformers.models.__path__.append(root)
modeling_file = importlib.import_module(f"transformers.models.{MT}.modeling_{MT}")

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    unsloth_compile_transformers(
        model_type = MT, fast_lora_forwards = False, fullgraph = True,
        import_from_cache = False, disable = True, supports_sdpa = [None],
    )

path = os.path.join("unsloth_compiled_cache", f"unsloth_compiled_module_{MT}.py")
generated = open(path, encoding = "utf-8").read()
error = None
forward_error = None
try:
    spec = importlib.util.spec_from_file_location("synthetic_generated", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
except Exception as exception:
    error = f"{type(exception).__name__}: {exception}"
else:
    # The method shadows the foreign helper by name but calls it bare: it must still resolve.
    try:
        output = module.SyntheticModel()(torch.full((2, 3, 8), 0.5))
        assert torch.equal(output, torch.full((2, 3, 8), 2.0)), output
    except Exception as exception:
        forward_error = f"{type(exception).__name__}: {exception}"

print("@@@" + json.dumps({
    "generated"     : generated,
    "modeling_file" : modeling_file.__file__,
    "error"         : error,
    "forward_error" : forward_error,
    "log"           : buf.getvalue()[-4000:],
}))
'''


def _run_child(cwd):
    env = dict(os.environ)
    env["UNSLOTH_ALLOW_CPU"] = "1"
    env["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    result = subprocess.run(
        [sys.executable, "-c", _CHILD],
        cwd = str(cwd),
        env = env,
        capture_output = True,
        text = True,
        timeout = 1800,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    for line in result.stdout.splitlines():
        if line.startswith("@@@"):
            return json.loads(line[3:])
    raise AssertionError(result.stdout[-4000:])


def test_generated_module_imports_with_foreign_wrapper_and_annotations(tmp_path):
    payload = _run_child(tmp_path)
    generated = payload["generated"]
    assert payload["modeling_file"].startswith(str(tmp_path)), payload["modeling_file"]
    assert payload["error"] is None, (
        payload["error"] + "\n" + generated[-3000:] + "\n--- log ---\n" + payload["log"]
    )
    assert payload["forward_error"] is None, payload["forward_error"] + "\n" + generated[-3000:]
    assert "\ndef build_mask(" not in generated, generated[-3000:]
    assert "\ndef local_helper(" in generated, payload["log"]
    assert "unused_arg" not in generated.split("\ndef local_helper(", 1)[1].split("\n", 1)[0]

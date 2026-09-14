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

"""Listing a helper in DISABLE_COMPILE_FUNCTIONS must reach its CALLERS too.
`called_functions` holds only names the modeling file both calls AND defines, so an
imported-only helper is imported raw for Dynamo to inline, and the CALLER is demoted."""

import importlib.util
import inspect
import json
import os
import subprocess
import sys

import pytest
import transformers

from unsloth_zoo import compiler as compiler_module
from unsloth_zoo.compiler import (
    DISABLE_COMPILE_FUNCTIONS,
    calls_disable_compile_function,
)


def test_bare_calls_are_detected_and_attribute_calls_are_not():
    """Several VL files define a METHOD of the same name, so matching those would demote
    modules that never touch the helper."""
    listed = DISABLE_COMPILE_FUNCTIONS[0]

    assert calls_disable_compile_function(
        f"    x = {listed}(grid_thw, 2)\n", DISABLE_COMPILE_FUNCTIONS
    ) == [listed]
    assert calls_disable_compile_function(
        f"    x = {listed} (grid_thw)\n", DISABLE_COMPILE_FUNCTIONS
    ) == [listed]

    assert calls_disable_compile_function(
        f"    x = self.{listed}(grid_thw, 2)\n", DISABLE_COMPILE_FUNCTIONS
    ) == []
    assert calls_disable_compile_function(
        f"    x = vision_utils.{listed}(grid_thw, 2)\n", DISABLE_COMPILE_FUNCTIONS
    ) == []
    assert calls_disable_compile_function(
        f"    x = wrapped_{listed}(grid_thw, 2)\n", DISABLE_COMPILE_FUNCTIONS
    ) == []
    assert calls_disable_compile_function(
        f'    """See {listed} for details."""\n', DISABLE_COMPILE_FUNCTIONS
    ) == []


def test_every_fullgraph_emit_site_consults_the_detector():
    """Miss one emit site and an imported helper is inlined into a fullgraph region again."""
    source = inspect.getsource(compiler_module)
    assert source.count("calls_disable_compile_function(") == 4, (
        "expected one definition plus three call sites (module scan + the two "
        "generated-source emit sites); a fullgraph = True emit no longer "
        "consults DISABLE_COMPILE_FUNCTIONS membership of the CALLEE."
    )


def _run_compiler_child(script, cwd):
    """Its own interpreter: the rewriter sets `__UNSLOTH_PATCHED__` on the modeling module
    and swaps the classes it emits back into it, so a compile run is a one-way trip for
    the process that does it."""
    # conftest's GPU-free harness does not reach a subprocess.
    env = dict(os.environ)
    env["UNSLOTH_ALLOW_CPU"] = "1"
    env["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    # Pin the child to THIS checkout, else it reports on the site-packages compiler.py.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    # An inherited UNSLOTH_COMPILE_DISABLE=1 forces disable=True and emits the wrong decorator.
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
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


def _decorator_above(generated, function_name):
    """The emit sites put the decorator immediately above `def <Class>_forward`, so the
    text after the last `\\n@` is the one that governs this function."""
    where = generated.find(f"def {function_name}(")
    assert where != -1, (
        f"{function_name} is not in the generated cache at all, so its decorator "
        f"proves nothing.\n" + generated[:2000]
    )
    return generated[:where].rsplit("\n@", 1)[-1]


# The synthetic fixture owns the end-to-end proof, because no shipped model presents the
# shape any more (see `_upstream_shape_is_shipped`). It writes a real modeling package to
# disk so inspect.getsource and linecache behave exactly as they do upstream, hangs it off
# transformers.models.__path__ so the compiler's own
# `importlib.import_module(f"transformers.models.{mt}.modeling_{mt}")` finds it, and then
# drives the real `unsloth_compile_transformers`.
_SYNTHETIC_CHILD = r'''
import os, sys, json, importlib, io, contextlib, textwrap
os.environ["UNSLOTH_COMPILE_LOCATION"] = "unsloth_compiled_cache"
import torch
import transformers.models
from unsloth_zoo.compiler import DISABLE_COMPILE_FUNCTIONS, unsloth_compile_transformers

# Any listed name proves the rule, and index 0 keeps working if the list is re-ordered.
LISTED   = DISABLE_COMPILE_FUNCTIONS[0]
MT       = "unsloth_synthetic_vl"
CALLER   = "SyntheticImportedHelperCaller"
INNOCENT = "SyntheticPlainBlock"

root = os.path.join(os.getcwd(), "synthetic_transformers_models")
package = os.path.join(root, MT)
os.makedirs(package)
with open(os.path.join(package, "__init__.py"), "w", encoding = "utf-8") as handle:
    handle.write("")

# The helper lives in a SIBLING module. The caller only imports it, so it never reaches
# `called_functions`, and the grid_thw.tolist() that makes it untraceable is invisible to
# every source scan the compiler runs over the modeling file.
with open(os.path.join(package, "synthetic_vision_utils.py"), "w", encoding = "utf-8") as handle:
    handle.write(textwrap.dedent(f"""
        def {LISTED}(grid_thw, spatial_merge_size):
            sizes = grid_thw.tolist()
            return grid_thw.new_zeros(len(sizes) * spatial_merge_size)
    """))

# Neither __init__ mentions nn.Linear or nn.ModuleList, so both modules arrive at the
# emit site with fullgraph = True and only the DISABLE_COMPILE_FUNCTIONS rule can part them.
with open(os.path.join(package, f"modeling_{MT}.py"), "w", encoding = "utf-8") as handle:
    handle.write(textwrap.dedent(f"""
        import torch
        import torch.nn as nn

        from .synthetic_vision_utils import {LISTED}

        __all__ = ["{CALLER}", "{INNOCENT}"]


        class {CALLER}(nn.Module):
            def __init__(self, hidden_size = 8, spatial_merge_size = 2):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(hidden_size))
                self.spatial_merge_size = spatial_merge_size

            def forward(self, hidden_states, grid_thw):
                indices = {LISTED}(grid_thw, self.spatial_merge_size)
                return hidden_states * self.scale + indices.sum()


        class {INNOCENT}(nn.Module):
            def __init__(self, hidden_size = 8):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))

            def forward(self, hidden_states):
                return hidden_states * self.weight
    """))

# transformers.models is a real package, so extending its __path__ hands the fixture to
# the ordinary import machinery rather than faking sys.modules entries around it.
transformers.models.__path__.append(root)
location = f"transformers.models.{MT}.modeling_{MT}"
modeling_file = importlib.import_module(location)

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

print("@@@" + json.dumps({
    "generated"     : generated,
    "modeling_file" : modeling_file.__file__,
    "listed"        : LISTED,
    "caller"        : CALLER,
    "innocent"      : INNOCENT,
    "log"           : buf.getvalue()[-4000:],
}))
'''


def test_synthetic_imported_helper_demotes_its_caller_and_nothing_else(tmp_path):
    payload = _run_compiler_child(_SYNTHETIC_CHILD, tmp_path)

    generated = payload["generated"]
    listed = payload["listed"]
    caller = payload["caller"]
    innocent = payload["innocent"]

    # The fixture has to have been the module the compiler actually read, not a
    # same-named leftover somewhere else on sys.path.
    assert payload["modeling_file"].startswith(str(tmp_path)), payload["modeling_file"]

    # The precondition the whole rule rests on: the helper is imported, never defined
    # here, so `called_functions` cannot see it and Dynamo inlines the raw upstream body.
    assert f"def {listed}(" not in generated, (
        f"the fixture now defines {listed} itself, which puts it in `called_functions` "
        f"and makes this test prove nothing."
    )

    assert "fullgraph = False" in _decorator_above(generated, f"{caller}_forward"), (
        f"{caller}_forward calls the imported {listed}, whose first line is "
        f"grid_thw.tolist(); emitting it fullgraph = True kills the first forward with "
        f"`Backend compiler exception ... aten._local_scalar_dense.default`.\n"
        + _decorator_above(generated, f"{caller}_forward")
        + "\n--- compiler log ---\n"
        + payload["log"]
    )

    # Demoting everything is not a fix. A module that calls nothing on the list keeps
    # fullgraph = True, so this test fails a blanket `fullgraph = False` just as loudly.
    assert "fullgraph = True" in _decorator_above(generated, f"{innocent}_forward"), (
        f"{innocent}_forward calls nothing in DISABLE_COMPILE_FUNCTIONS, so demoting it "
        f"gives up fullgraph for every leaf module in every model.\n"
        + _decorator_above(generated, f"{innocent}_forward")
        + "\n--- compiler log ---\n"
        + payload["log"]
    )


_UPSTREAM_MODEL_TYPE = "minimax_m3_vl"
_UPSTREAM_CLASS = "MiniMaxM3VL3DRotaryEmbedding"


def _upstream_shape_is_shipped():
    """transformers 5.17 renamed this class to MiniMaxM3VLVisionRotaryEmbedding and lifted
    the get_vision_position_ids call out of it into MiniMaxM3VLVisionModel.forward. That is
    a PreTrainedModel subclass, which is never compiled, so re-pointing at the new name
    asserts nothing. Across all of transformers 5.17 the only plain nn.Module left calling
    the helper is paddleocr_vl's encoder, and paddleocr_vl DEFINES the helper itself.

    Read the source text instead of importing: importing a modeling module here would leak
    a half-patched transformers into every test that runs after this one."""
    try:
        import transformers.vision_utils  # noqa: F401

        spec = importlib.util.find_spec(
            f"transformers.models.{_UPSTREAM_MODEL_TYPE}.modeling_{_UPSTREAM_MODEL_TYPE}"
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
    return f"class {_UPSTREAM_CLASS}(" in source


# Its own interpreter: the rewriter sets `__UNSLOTH_PATCHED__` on the modeling module.
_UPSTREAM_CHILD = r'''
import os, sys, json, importlib, io, contextlib
os.environ["UNSLOTH_COMPILE_LOCATION"] = "unsloth_compiled_cache"
import torch
from unsloth_zoo.compiler import unsloth_compile_transformers

MT = "minimax_m3_vl"
loc = f"transformers.models.{MT}.modeling_{MT}"
mf = importlib.import_module(loc)

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

error = None
error_type = None
shapes = None
try:
    rope = mf.MiniMaxM3VL3DRotaryEmbedding(head_dim = 80, spatial_merge_size = 2)
    cos, sin = rope(torch.tensor([[1, 24, 32]], dtype = torch.long),
                    torch.device("cpu"), torch.float32)
    shapes = [tuple(cos.shape), tuple(sin.shape)]
except Exception as exception:
    error_type = type(exception).__name__
    error = f"{error_type}: {str(exception).strip().splitlines()[0]}"

print("@@@" + json.dumps({
    "generated"   : generated,
    "error"       : error,
    "error_type"  : error_type,
    "shapes"      : shapes,
}))
'''


@pytest.mark.skipif(
    not _upstream_shape_is_shipped(),
    reason = (
        f"transformers {transformers.__version__} no longer ships a compiled leaf module "
        f"that calls an imported-only DISABLE_COMPILE_FUNCTIONS helper: "
        f"{_UPSTREAM_CLASS} is gone from modeling_{_UPSTREAM_MODEL_TYPE} (renamed to "
        f"MiniMaxM3VLVisionRotaryEmbedding, with the get_vision_position_ids call moved "
        f"into the uncompiled MiniMaxM3VLVisionModel). "
        f"test_synthetic_imported_helper_demotes_its_caller_and_nothing_else covers the rule."
    ),
)
def test_imported_helper_is_not_inlined_into_a_fullgraph_region(tmp_path):
    payload = _run_compiler_child(_UPSTREAM_CHILD, tmp_path)

    generated = payload["generated"]
    assert "def get_vision_position_ids(" not in generated, (
        "minimax_m3_vl now defines the helper itself; pick another model type "
        "that only imports it, or this test proves nothing."
    )
    where = generated.find(f"def {_UPSTREAM_CLASS}_forward")
    assert where != -1, f"{_UPSTREAM_CLASS}_forward is no longer generated"
    decorators = generated[:where]
    assert "fullgraph = False" in decorators.rsplit("\n@", 1)[-1], (
        f"{_UPSTREAM_CLASS}_forward calls the raw upstream "
        "get_vision_position_ids, whose first line is grid_thw.tolist(); "
        "emitting it fullgraph = True kills the first vision forward with "
        "`Backend compiler exception ... aten._local_scalar_dense.default`.\n"
        + decorators[-400:]
    )

    # Bonus check: a bare CI runner may lack a working inductor, and the regression itself
    # surfaces as a different exception type.
    if payload.get("error_type") == "BackendCompilerFailed":
        pytest.skip(f"inductor cannot compile here ({payload['error']}); decorator asserted above")
    assert payload["error"] is None, payload["error"]
    assert payload["shapes"] == [[768, 78], [768, 78]], payload["shapes"]

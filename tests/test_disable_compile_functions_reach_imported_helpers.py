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

import inspect
import os
import subprocess
import sys

import pytest

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


_HAS_VISION_UTILS = False
try:
    import transformers.vision_utils  # noqa: F401
    import transformers.models.minimax_m3_vl  # noqa: F401

    _HAS_VISION_UTILS = True
except Exception:
    pass


# Its own interpreter: the rewriter sets `__UNSLOTH_PATCHED__` on the modeling module.
_CHILD = r'''
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
    not _HAS_VISION_UTILS,
    reason = "needs transformers >= 5.9 (shared vision_utils) with minimax_m3_vl",
)
def test_imported_helper_is_not_inlined_into_a_fullgraph_region(tmp_path):
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
        [sys.executable, "-c", _CHILD],
        cwd = str(tmp_path),
        env = env,
        capture_output = True,
        text = True,
        timeout = 1800,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    payload = None
    for line in result.stdout.splitlines():
        if line.startswith("@@@"):
            import json

            payload = json.loads(line[3:])
    assert payload is not None, result.stdout[-4000:]

    generated = payload["generated"]
    assert "def get_vision_position_ids(" not in generated, (
        "minimax_m3_vl now defines the helper itself; pick another model type "
        "that only imports it, or this test proves nothing."
    )
    where = generated.find("def MiniMaxM3VL3DRotaryEmbedding_forward")
    assert where != -1, "MiniMaxM3VL3DRotaryEmbedding_forward is no longer generated"
    decorators = generated[:where]
    assert "fullgraph = False" in decorators.rsplit("\n@", 1)[-1], (
        "MiniMaxM3VL3DRotaryEmbedding_forward calls the raw upstream "
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

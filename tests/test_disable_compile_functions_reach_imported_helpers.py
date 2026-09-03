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

`unsloth_compile_transformers` builds `called_functions` from names that the
modeling file both calls AND defines::

    defined = re.findall(r"\\bdef[\\s]{1,}" + re.escape(function), full_source, ...)
    called  = re.findall(r"[\\s]{1,}" + re.escape(function) + r"\\(.+?\\)", full_source, ...)
    if len(defined) != 0 and len(called) != 0:

so a modeling file that only *imports* a helper never reaches either
`disable_compile_functions` branch, and `create_new_function` imports the raw
upstream function into the generated module. transformers >= 5.9 moved the
vision grid helpers into `transformers/vision_utils.py`, and at 5.16.1 47 of the
61 (modeling file, helper) import pairs are imported-only. Two of those land
inside a `fullgraph = True` region::

    MiniMaxM3VL3DRotaryEmbedding.forward       -> get_vision_position_ids
    Kimi_K25VisionPositionEmbeddings.forward   -> get_vision_interpolation_indices_and_weights

which on torch < 2.10 ends the first vision forward with

    torch._dynamo.exc.Unsupported: Backend compiler exception
    Backend compiler `inductor` failed with aten._local_scalar_dense.default
      ... in get_vision_position_ids: grid_thw.tolist()

The fix demotes any caller of a listed helper to `fullgraph = False`. Demoting the
caller, rather than only re-emitting the helper as `@torch.compiler.disable`, is
deliberate: a disabled callee is itself a graph break, and a graph break under
`fullgraph = True` is a hard error, so disabling alone swaps one crash for another.

The first two tests are pure source inspection and run anywhere. The last one
drives the real rewriter and is skipped below transformers 5.9, where the shared
vision_utils does not exist.
"""

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
    """The detector has to see an imported helper and ignore a same-named method.

    qwen3_vl, glm4v, qwen2_5_vl and paddleocr_vl all define a METHOD called
    `get_vision_position_ids` next to the module-level import, so matching the
    bare name would demote modules that never touch the helper.
    """
    listed = DISABLE_COMPILE_FUNCTIONS[0]

    assert calls_disable_compile_function(
        f"    x = {listed}(grid_thw, 2)\n", DISABLE_COMPILE_FUNCTIONS
    ) == [listed]
    # whitespace between name and paren, and a call as the very first token
    assert calls_disable_compile_function(
        f"    x = {listed} (grid_thw)\n", DISABLE_COMPILE_FUNCTIONS
    ) == [listed]

    assert calls_disable_compile_function(
        f"    x = self.{listed}(grid_thw, 2)\n", DISABLE_COMPILE_FUNCTIONS
    ) == []
    assert calls_disable_compile_function(
        f"    x = vision_utils.{listed}(grid_thw, 2)\n", DISABLE_COMPILE_FUNCTIONS
    ) == []
    # a longer name that merely ends with a listed one must not match
    assert calls_disable_compile_function(
        f"    x = wrapped_{listed}(grid_thw, 2)\n", DISABLE_COMPILE_FUNCTIONS
    ) == []
    # a mention that is not a call (a docstring, an export list) must not match
    assert calls_disable_compile_function(
        f'    """See {listed} for details."""\n', DISABLE_COMPILE_FUNCTIONS
    ) == []


def test_every_fullgraph_emit_site_consults_the_detector():
    """All three places that can stamp fullgraph = True have to ask.

    The module scan (`no_fullgraph_modules`) plus both generated-source emit
    sites in `compile_function_calls`. Miss one and an imported helper is
    inlined into a fullgraph region again, which the runtime tests here cannot
    see because they do not go through the rewriter.
    """
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


# Runs the rewriter for real, so it goes in its own interpreter: it sets
# `__UNSLOTH_PATCHED__` on the modeling module and writes a cache directory.
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
    """The regression itself, end to end through the real rewriter.

    minimax_m3_vl only imports `get_vision_position_ids`, so it is never in
    `called_functions` and the generated module calls the raw upstream helper.
    The forward that calls it must therefore be emitted `fullgraph = False`, and
    calling it must produce position embeddings rather than a Dynamo error.
    """
    # conftest's GPU-free harness does not reach a subprocess, and importing
    # unsloth_zoo on a GPU-less runner would otherwise go looking for one.
    env = dict(os.environ)
    env["UNSLOTH_ALLOW_CPU"] = "1"
    env["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    # Pin the child to THIS checkout. Without it the subprocess imports whatever
    # unsloth_zoo is installed in site-packages, so the test silently reports on
    # a different copy of compiler.py than the one under review.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    # The core-drift job sets UNSLOTH_COMPILE_DISABLE=1 at job level, and this
    # copies the whole environment. Inheriting it makes unsloth_compile_transformers
    # force disable=True, so the forward is emitted @torch.compiler.disable instead
    # of the torch_compile_with_fallback this test is about, and the assertion below
    # fails for a reason unrelated to the regression. Drop it for the child.
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
    # The premise: the helper really is imported raw rather than re-emitted.
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

    # The decorator asserted above IS the regression guard: emitting this forward
    # fullgraph = True is precisely the bug, and that check is deterministic and
    # runs everywhere. Actually calling the forward is a bonus end-to-end check
    # that additionally needs a working inductor toolchain. A bare CI runner has
    # no Triton and its inductor CPU backend can fail outright, which says nothing
    # about this change, so tolerate exactly that and nothing else. The regression
    # this test exists for surfaces as torch._dynamo.exc.Unsupported, a different
    # type, so it is still caught here.
    if payload.get("error_type") == "BackendCompilerFailed":
        pytest.skip(f"inductor cannot compile here ({payload['error']}); decorator asserted above")
    assert payload["error"] is None, payload["error"]
    assert payload["shapes"] == [[768, 78], [768, 78]], payload["shapes"]

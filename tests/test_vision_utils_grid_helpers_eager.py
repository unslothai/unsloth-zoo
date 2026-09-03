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

"""The transformers/vision_utils.py grid helpers must never be compiled fullgraph.

transformers 5.9 moved the per-image grid helpers every VL tower calls into a shared
`transformers/vision_utils.py`, and compiler.py's rewriter used to stamp
fullgraph = True on them. Each opens by reading the grid off the device with
`grid_thw.tolist()`, so the shapes built from those unbacked SymInts are not
guardable (`UserError: Could not guard on data-dependent expression Eq((u2//u3), 0)`
out of `reshape((h // merge, merge, w // merge, merge))`). Under fullgraph = True
that is a hard error rather than a graph break, so every Qwen3.5 / Qwen3-VL /
GLM-4V / PaddleOCR-VL run died on the first vision forward. fullgraph = False alone
is not enough either, since the `.tolist()` sync breaks the graph on line 1, so the
helpers are listed in `DISABLE_COMPILE_FUNCTIONS` and emitted eager.

Which helpers trip is torch dependent (pytorch#162354 made the guards non-throwing
in 2.10), so `test_uncompilable_grid_helpers_are_listed` probes upstream and asserts
an implication rather than a fixed list of names: it passes on both sides of that
boundary and picks up new helpers without an edit here, which is how 5.16's
`get_vision_interpolation_indices_and_weights` was caught.

CPU only: the failure is in the meta/shape layer.
"""

import inspect

import pytest
import torch
import transformers

from unsloth_zoo.compiler import DISABLE_COMPILE_FUNCTIONS

vision_utils = pytest.importorskip(
    "transformers.vision_utils",
    reason="transformers < 5.9 has no shared vision_utils module",
)


# Height and width are multiples of every merge size below, so the helpers have real
# work to do.
GRID_THW = torch.tensor([[1, 24, 32]], dtype=torch.long)

# Filled in by name so an upstream signature reordering (5.16 inserts
# `include_temporal` into get_vision_position_ids) cannot silently pass the wrong
# value. `config` only decides whether flash attention is requested; a default one
# answers no, the path get_vision_attention_seqlens takes for every non-flash run,
# and without it the probe below would skip that helper.
_ARGS = {
    "grid_thw": GRID_THW,
    "spatial_merge_size": 2,
    "window_size": 112,
    "patch_size": 14,
    "num_grid_per_side": 32,
    "config": transformers.PreTrainedConfig(),
}


def _grid_helpers():
    """Every public vision_utils helper the modeling files call with an image grid.

    Not filtered on `.tolist()`: get_vision_cu_seqlens reaches the same unbacked
    shapes via `repeat_interleave`, and only the "cannot be traced fullgraph"
    property below matters, not how it got there.
    """
    found = {}
    for name in dir(vision_utils):
        if not name.startswith("get_vision_"):
            continue
        fn = getattr(vision_utils, name)
        if not inspect.isfunction(fn):
            continue
        params = inspect.signature(fn).parameters
        if "grid_thw" not in params:
            continue
        if any(
            p not in _ARGS
            for p, spec in params.items()
            if spec.default is inspect.Parameter.empty
        ):
            # A required argument this test cannot build; skip rather than guess.
            continue
        found[name] = (fn, {p: _ARGS[p] for p in params if p in _ARGS})
    return found


def test_grid_helpers_are_discoverable():
    """Guard the probe itself: if this finds nothing the tests below are vacuous."""
    helpers = _grid_helpers()
    assert "get_vision_position_ids" in helpers, (
        "transformers.vision_utils no longer exposes a get_vision_position_ids taking "
        f"grid_thw; found {sorted(helpers)}. Re-check what the modeling files import "
        "before trusting the rest of this file."
    )


@pytest.mark.parametrize("name", sorted(_grid_helpers()))
def test_uncompilable_grid_helpers_are_listed(name):
    """Anything Dynamo cannot capture fullgraph must be in DISABLE_COMPILE_FUNCTIONS.

    An implication rather than a flat list of names, so upstream making a helper
    compilable does not fail, while a new uncompilable helper does.
    """
    fn, kwargs = _grid_helpers()[name]

    eager = fn(**kwargs)

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True, dynamic=True)
    try:
        compiled(**kwargs)
    except Exception as exception:
        assert name in DISABLE_COMPILE_FUNCTIONS, (
            f"transformers.vision_utils.{name} cannot be traced with fullgraph = True "
            f"({type(exception).__name__}: "
            f"{str(exception).strip().splitlines()[0][:200]}), but it is not in "
            "compiler.py's DISABLE_COMPILE_FUNCTIONS, so the rewriter will stamp "
            "@torch_compile_with_fallback(fullgraph = True, ...) on it and every VL "
            "model importing it dies on the first vision forward. Add it to the list."
        )
    finally:
        torch._dynamo.reset()

    # Eager has to produce something usable either way, else the probe above proved
    # nothing about the real call path. `None` is allowed: get_vision_attention_seqlens
    # documents a None max_seqlen without flash attention, the config built here.
    outputs = eager if isinstance(eager, tuple) else (eager,)
    assert any(o is not None for o in outputs)
    assert all(
        isinstance(o, torch.Tensor) and o.numel() > 0 for o in outputs if o is not None
    )


def test_get_vision_position_ids_eager_result_is_the_block_major_layout():
    """The layout the disabled path has to keep producing.

    Cheap oracle so "we stopped compiling it" cannot become "we stopped computing
    it correctly": for a t=1 image the (row, col) pairs are the h*w grid walked in
    spatial-merge-block order.
    """
    merge_size = 2
    _, height, width = GRID_THW[0].tolist()

    position_ids = vision_utils.get_vision_position_ids(GRID_THW, merge_size)

    assert position_ids.shape == (height * width, 2)

    expected_rows, expected_cols = torch.meshgrid(
        torch.arange(height), torch.arange(width), indexing="ij"
    )
    block_shape = (
        height // merge_size,
        merge_size,
        width // merge_size,
        merge_size,
    )
    expected = torch.stack(
        [
            expected_rows.reshape(block_shape).transpose(1, 2).flatten(),
            expected_cols.reshape(block_shape).transpose(1, 2).flatten(),
        ],
        dim=-1,
    )
    assert torch.equal(position_ids.cpu(), expected)


def test_disable_compile_functions_selects_the_disable_decorator():
    """The list only works because both emit sites consult it before compiling.

    Membership is checked ahead of the `torch_compile_with_fallback` branch. Invert
    that ordering and the names go back to being compiled while the runtime tests
    above still pass, so pin the wiring too.
    """
    source = inspect.getsource(
        __import__("unsloth_zoo.compiler", fromlist=["compiler"])
    )

    assert source.count("if module in disable_compile_functions:") == 2, (
        "compiler.py no longer gates both generated-source emit sites on "
        "disable_compile_functions; DISABLE_COMPILE_FUNCTIONS entries may no "
        "longer take effect."
    )
    for name in (
        "get_vision_position_ids",
        "get_vision_cu_seqlens",
        "get_vision_attention_seqlens",
        "get_vision_window_index",
        "get_vision_interpolation_indices_and_weights",
        "get_vision_bilinear_indices_and_weights",
    ):
        assert f'"{name}"' in source, (
            f"compiler.py's DISABLE_COMPILE_FUNCTIONS no longer lists {name}; "
            "transformers 5.9+ VL models will crash with "
            "'Could not guard on data-dependent expression' on the first vision forward."
        )

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

"""Grid helpers in transformers/vision_utils.py must never be compiled fullgraph:
`grid_thw.tolist()` builds unguardable shapes from unbacked SymInts, a hard UserError on
the first vision forward. The probe below enumerates vision_utils rather than hard-coding
today's names."""

import inspect

import pytest
import torch
import transformers

from unsloth_zoo.compiler import DISABLE_COMPILE_FUNCTIONS

vision_utils = pytest.importorskip(
    "transformers.vision_utils",
    reason="transformers < 5.9 has no shared vision_utils module",
)


GRID_THW = torch.tensor([[1, 24, 32]], dtype=torch.long)

# Keyed by name so an upstream signature reorder cannot pass the wrong value; without a
# default `config` the probe below skips get_vision_attention_seqlens.
_ARGS = {
    "grid_thw": GRID_THW,
    "spatial_merge_size": 2,
    "window_size": 112,
    "patch_size": 14,
    "num_grid_per_side": 32,
    "config": transformers.PreTrainedConfig(),
}


def _grid_helpers():
    """Public vision_utils helpers taking an image grid. Not filtered on `.tolist()`:
    get_vision_cu_seqlens reaches the same unbacked shapes via `repeat_interleave`."""
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
            continue
        found[name] = (fn, {p: _ARGS[p] for p in params if p in _ARGS})
    return found


def test_grid_helpers_are_discoverable():
    helpers = _grid_helpers()
    assert "get_vision_position_ids" in helpers, (
        "transformers.vision_utils no longer exposes a get_vision_position_ids taking "
        f"grid_thw; found {sorted(helpers)}. Re-check what the modeling files import "
        "before trusting the rest of this file."
    )


@pytest.mark.parametrize("name", sorted(_grid_helpers()))
def test_uncompilable_grid_helpers_are_listed(name):
    """Anything Dynamo cannot capture fullgraph must be in DISABLE_COMPILE_FUNCTIONS. The
    assert fires only inside `except`, so a helper turning traceable upstream cannot fail it."""
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

    # `None` is documented for get_vision_attention_seqlens without flash attention.
    outputs = eager if isinstance(eager, tuple) else (eager,)
    assert any(o is not None for o in outputs)
    assert all(
        isinstance(o, torch.Tensor) and o.numel() > 0 for o in outputs if o is not None
    )


def test_get_vision_position_ids_eager_result_is_the_block_major_layout():
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
    """Membership is checked ahead of the `torch_compile_with_fallback` branch; invert that
    and these names compile again while the runtime tests above still pass."""
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

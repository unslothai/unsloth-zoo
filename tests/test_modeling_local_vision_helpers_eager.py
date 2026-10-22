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

"""Grid helpers defined in modeling files need the same eager fallback as vision_utils.

Discover helpers from source, then import only matching modules to probe tracing."""

import inspect
import os
import pathlib
import re

import pytest
import torch

from unsloth_zoo.compiler import DISABLE_COMPILE_FUNCTIONS

transformers = pytest.importorskip("transformers")

# Match module-level helpers; methods are not emitted standalone.
_DEFINITION = re.compile(r"^def (get_vision_\w+)\s*\(", re.MULTILINE)

# Dimensions must be divisible by the merge and kernel sizes below.
GRID_THW = torch.tensor([[1, 24, 32]], dtype = torch.long)

# Bind fixtures by parameter name; skip helpers with unknown required parameters.
_ARGS = {
    "grid_thw": GRID_THW,
    "merge_size": 2,
    "spatial_merge_size": 2,
    "kernel_height": 2,
    "kernel_width": 2,
}


def _models_root():
    root = pathlib.Path(inspect.getfile(transformers)).parent / "models"
    return root if root.is_dir() else None


def _locally_defined_grid_helpers():
    """Find locally defined and called helpers without importing every modeling module."""
    root = _models_root()
    found = {}
    if root is None:
        return found
    for path in sorted(root.glob("*/modeling_*.py")):
        try:
            source = path.read_text(encoding = "utf-8")
        except OSError:
            continue
        for name in _DEFINITION.findall(source):
            # Exclude the definition while retaining calls before and after it.
            callers = re.sub(r"^def " + re.escape(name) + r"\s*\(", "", source, flags = re.MULTILINE)
            if not re.search(r"[^\w.]" + re.escape(name) + r"\s*\(", callers):
                continue
            found.setdefault(name, (path.parent.name, path))
    return found


_HELPERS = _locally_defined_grid_helpers()


def _load(name):
    """Import the modeling module and return its helper, or skip."""
    model_type, _ = _HELPERS[name]
    module = pytest.importorskip(
        f"transformers.models.{model_type}.modeling_{model_type}",
        reason = f"{model_type} needs optional deps this environment lacks",
    )
    function = getattr(module, name, None)
    if not inspect.isfunction(function):
        pytest.skip(f"{model_type}.{name} is no longer a module-level function")
    return function


def _call_args(function):
    parameters = inspect.signature(function).parameters
    missing = [
        p for p, spec in parameters.items()
        if spec.default is inspect.Parameter.empty and p not in _ARGS
    ]
    if missing:
        pytest.skip(f"no fixture value for required parameter(s) {missing}")
    return {p: _ARGS[p] for p in parameters if p in _ARGS}


# Check file existence independently so broken discovery cannot silently skip coverage.
_EXPECTED = {
    "muse_glimmer": ["get_vision_pixel_shuffle_index"],
    "kimi_k25": ["get_vision_frame_index", "get_vision_temporal_merge_index"],
}


def test_the_sweep_finds_the_known_helpers():
    """Require discovery to find known helpers whenever their modeling files exist."""
    root = _models_root()
    if root is None:
        pytest.skip(f"transformers {transformers.__version__} ships no models package")

    shipped = {
        model_type: names
        for model_type, names in _EXPECTED.items()
        if (root / model_type / f"modeling_{model_type}.py").is_file()
    }
    if not shipped:
        pytest.skip(
            f"transformers {transformers.__version__} ships none of {sorted(_EXPECTED)}; "
            "nothing to discover, and the vision_utils probe covers the exported helpers"
        )

    for model_type, names in shipped.items():
        for name in names:
            assert name in _HELPERS, (
                f"transformers {transformers.__version__} ships "
                f"modeling_{model_type}.py, but the sweep did not find {name} in it. "
                "Either upstream renamed or moved the helper, or _DEFINITION / the "
                "called-check stopped matching. Until this is resolved every "
                "test_uncompilable_local_grid_helpers_are_listed case is silently gone."
            )
            assert _HELPERS[name][0] == model_type, _HELPERS[name]

    assert all(os.path.exists(path) for _, path in _HELPERS.values())


@pytest.mark.parametrize("name", sorted(_HELPERS))
def test_uncompilable_local_grid_helpers_are_listed(name):
    """Helpers that fail fullgraph tracing must be listed for eager execution.

    The eager backend isolates Dynamo failures from Inductor and Triton setup.
    Helpers that become traceable on newer torch versions may remain listed."""
    model_type, _ = _HELPERS[name]
    function = _load(name)
    kwargs = _call_args(function)

    eager = function(**kwargs)

    torch._dynamo.reset()
    try:
        torch.compile(function, fullgraph = True, dynamic = True, backend = "eager")(**kwargs)
    except Exception as exception:
        assert name in DISABLE_COMPILE_FUNCTIONS, (
            f"transformers.models.{model_type} defines {name}, which cannot be traced "
            f"with fullgraph = True ({type(exception).__name__}: "
            f"{str(exception).strip().splitlines()[0][:200]}), but it is not in "
            "compiler.py's DISABLE_COMPILE_FUNCTIONS. It is defined AND called in the "
            "modeling file, so it reaches `called_functions` and the rewriter stamps "
            "@torch_compile_with_fallback(fullgraph = True, ...) directly on it -- "
            f"every {model_type} vision forward then dies on the first image. Add it "
            "to the list."
        )
    finally:
        torch._dynamo.reset()

    outputs = eager if isinstance(eager, tuple) else (eager,)
    assert all(
        isinstance(o, torch.Tensor) and o.numel() > 0 for o in outputs if o is not None
    )


@pytest.mark.parametrize(
    "name",
    [
        "get_vision_pixel_shuffle_index",
        "get_vision_frame_index",
        "get_vision_temporal_merge_index",
    ],
)
def test_the_known_offenders_stay_listed(name):
    """Keep coverage when models are absent or helpers trace on newer torch versions.

    temporal_merge_index fails on torch 2.9.1 but traces on 2.10.0 and 2.13.0."""
    assert name in DISABLE_COMPILE_FUNCTIONS, (
        f"{name} guards on a value read out of grid_thw.tolist(); dropping it from "
        "DISABLE_COMPILE_FUNCTIONS puts back 'Could not guard on data-dependent "
        "expression' on the first image of a muse_glimmer or kimi_k25 vision fine-tune. "
        "Check the oldest torch pyproject admits before deciding one of these is "
        "traceable now."
    )

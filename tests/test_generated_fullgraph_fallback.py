# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Generated `fullgraph = True` regions must not hard-fail on cache exhaustion.

`patch_function` already routes its own fullgraph compiles through
`_fall_back_to_eager_on_recompile_limit`, but the modules written into
`unsloth_compiled_cache` decorate their functions directly and never reach it,
so every one of those regions kept the failure the wrapper exists to remove.

Gemma4 has ten. Its vision tower drives `Gemma4RMSNorm_forward` past the
budget and training stops at step 0 with

    FailOnRecompileLimitHit: Hard failure due to fullgraph=True

raised from `unsloth_compiled_module_gemma4.py`. Measured on the real notebook
path with `recompile_limit = 2`: released zoo and this branch without the fix
both die at step 0; with it, all 7 steps run and the loss falls 3.187 -> 1.202.
Lowering the limit is what makes the failure reachable on any GPU -- a T4
reaches it unaided, a B200 at the default never does.
"""

import re
from pathlib import Path

import pytest
import torch

from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback

COMPILER = Path(__file__).resolve().parents[1] / "unsloth_zoo" / "compiler.py"


# ---- what the compiler emits ---------------------------------------------

def test_no_emitter_writes_a_bare_fullgraph_compile():
    """Every emitted `fullgraph` decorator has to carry the fallback.

    Source-level because the alternative is generating a module per model, and
    a single missed emitter is exactly how the cross-entropy template kept its
    bare decorator after the first four were fixed.
    """
    src = COMPILER.read_text()
    bare = [line.strip() for line in src.splitlines()
            if "@torch.compile(fullgraph" in line and not line.lstrip().startswith("#")]
    assert not bare, bare


def test_the_generated_preamble_imports_the_helper():
    """The decorator name has to resolve inside the generated module."""
    src = COMPILER.read_text()
    assert "from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback" in src


def test_the_cross_entropy_template_carries_its_own_import():
    """That block is spliced in on its own, so it cannot borrow the preamble."""
    src = COMPILER.read_text()
    block = src.split("_cross_entropy_code = \"\"\"", 1)[1].split("\"\"\"", 1)[0]
    assert "@torch_compile_with_fallback(" in block
    assert "import torch_compile_with_fallback" in block


# ---- what the helper does -------------------------------------------------

def test_fullgraph_false_is_returned_untouched():
    """Dynamo already falls back on its own there, so a wrapper could never
    fire and would only add a frame."""
    def f(x):
        return x + 1

    wrapped = torch_compile_with_fallback(fullgraph = False)(f)
    assert not hasattr(wrapped, "_unsloth_fallback_state")


def test_fullgraph_true_is_wrapped():
    def f(x):
        return x + 1

    wrapped = torch_compile_with_fallback(fullgraph = True)(f)
    assert hasattr(wrapped, "_unsloth_fallback_state")


def test_the_wrapper_still_computes_the_right_answer():
    def f(x):
        return x * 2 + 1

    wrapped = torch_compile_with_fallback(fullgraph = True)(f)
    x = torch.arange(4, dtype = torch.float32)
    assert torch.equal(wrapped(x), f(x))


def test_a_label_is_recorded_for_the_warning():
    """The give-up message names the region; an unnamed one is unactionable."""
    def some_forward(x):
        return x

    wrapped = torch_compile_with_fallback(fullgraph = True)(some_forward)
    assert "some_forward" in wrapped._unsloth_fallback_label


def test_the_eager_path_is_the_original_function():
    """The fallback runs this, so a compiled object here would defeat it."""
    def some_forward(x):
        return x

    wrapped = torch_compile_with_fallback(fullgraph = True)(some_forward)
    assert wrapped.__wrapped__ is some_forward

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

ZOO = Path(__file__).resolve().parents[1] / "unsloth_zoo"
COMPILER = ZOO / "compiler.py"
# Every module that hands torch a fullgraph region, not just the compiler. The
# first pass scanned compiler.py alone and passed while GRPO's own
# `grpo_compute_loss_slow` and `accumulate_chunk` still carried bare
# decorators, so cache exhaustion there stayed fatal.
FULLGRAPH_SITES = (COMPILER, ZOO / "rl_replacements.py",
                   ZOO / "temporary_patches" / "common.py")


# ---- what the compiler emits ---------------------------------------------

def test_no_emitter_writes_a_bare_fullgraph_compile():
    """Every emitted `fullgraph` decorator has to carry the fallback.

    Source-level because the alternative is generating a module per model, and
    a single missed emitter is exactly how the cross-entropy template kept its
    bare decorator after the first four were fixed.
    """
    bare = []
    for path in FULLGRAPH_SITES:
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "@torch.compile(fullgraph" in line or "@torch.compile(dynamic" in line:
                bare.append(f"{path.name}: {stripped}")
    assert not bare, bare


def test_the_grpo_loss_regions_are_wrapped():
    """The two sites the first pass missed. `grpo_compute_loss_slow` is emitted
    as source into the generated trainer, and `accumulate_chunk` is compiled at
    call time inside UnslothEfficientGRPO's backward."""
    from unsloth_zoo.rl_replacements import RL_REPLACEMENTS
    slow = RL_REPLACEMENTS["grpo_compute_loss_slow"]
    assert "@torch_compile_with_fallback(" in slow
    assert "import torch_compile_with_fallback" in slow, \
        "the name has to resolve inside the generated module"
    src = (ZOO / "rl_replacements.py").read_text()
    assert "torch_compile_with_fallback(\n            fullgraph = True," in src, \
        "accumulate_chunk is still compiled bare"


def test_maybe_compile_routes_fullgraph_through_the_fallback():
    """Three more fullgraph regions go through this one helper, so wiring it
    covers them without touching each decorator."""
    from unsloth_zoo.temporary_patches import common
    src = (ZOO / "temporary_patches" / "common.py").read_text()
    body = src.split("def _maybe_compile(", 1)[1].split("\ndef ", 1)[0]
    assert "torch_compile_with_fallback" in body
    # Without fullgraph Dynamo already falls back by itself, so leave it alone.
    assert 'if not kwargs.get("fullgraph"):' in body

    def f(x):
        return x
    wrapped = common._maybe_compile(fullgraph = True, dynamic = True)(f)
    assert hasattr(wrapped, "_unsloth_fallback_state")
    plain = common._maybe_compile(dynamic = True)(f)
    assert not hasattr(plain, "_unsloth_fallback_state")


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


# aot_eager, not inductor. Inductor codegens C++ and shells out to a host
# compiler, so this raised `Compiler: cl is not found` on the Windows runner --
# a statement about MSVC being absent, not about the wrapper. What is under test
# is that the wrapper returns what the eager function returns, and every backend
# answers that.
_BACKEND = "aot_eager"


def test_the_wrapper_still_computes_the_right_answer():
    def f(x):
        return x * 2 + 1

    wrapped = torch_compile_with_fallback(fullgraph = True, backend = _BACKEND)(f)
    x = torch.arange(4, dtype = torch.float32)
    assert torch.equal(wrapped(x), f(x))


def test_the_helper_takes_a_backend_like_torch_compile_does():
    """It forwards **compile_kwargs, and the generated decorators rely on that
    for `dynamic` and `options`; a dropped kwarg would compile the wrong thing."""
    import inspect
    sig = inspect.signature(torch_compile_with_fallback)
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in sig.parameters.values())

    def f(x):
        return x

    # No host compiler needed, and no exception: the kwarg reached torch.compile.
    assert torch_compile_with_fallback(fullgraph = True, backend = "eager")(f)(1) == 1


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

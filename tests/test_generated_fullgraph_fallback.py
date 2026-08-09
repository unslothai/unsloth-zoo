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

Gemma4 has ten. Its vision tower drives `Gemma4RMSNorm_forward` past the budget
and training stops at step 0 with

    FailOnRecompileLimitHit: Hard failure due to fullgraph=True

from `unsloth_compiled_module_gemma4.py`. Measured on the real notebook path
with `recompile_limit = 2`: released zoo and this branch without the fix both
die at step 0; with it all 7 steps run and the loss falls 3.187 -> 1.202.
Lowering the limit makes the failure reachable on any GPU -- a T4 gets there
unaided, a B200 at the default never does.
"""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback

ZOO = Path(__file__).resolve().parents[1] / "unsloth_zoo"
COMPILER = ZOO / "compiler.py"
# Every module that hands torch a fullgraph region, not just the compiler: the
# first pass scanned compiler.py alone and passed while GRPO's own
# `grpo_compute_loss_slow` and `accumulate_chunk` stayed bare, and so fatal.
FULLGRAPH_SITES = (COMPILER, ZOO / "rl_replacements.py",
                   ZOO / "temporary_patches" / "common.py")


def _run_in_fresh_interpreter(body, compile_disable):
    """Run `body` in a new interpreter with UNSLOTH_COMPILE_DISABLE pinned.

    `None` clears the variable, a string sets it. unsloth's consolidated CI runs
    this whole suite with `UNSLOTH_COMPILE_DISABLE=1` at the job level, so a test
    that needs the other setting cannot get it from the ambient environment.

    `common.py` reacts to that variable in two different ways, and only one of
    them needs this:

    * **Bound at import.** `torch_compile`, `_torch_compile` and
      `_raw_torch_compile` are assigned under a module-level
      `if UNSLOTH_COMPILE_DISABLE:` and are already `noop` function objects by
      the time any test runs. Rebinding `common.UNSLOTH_COMPILE_DISABLE`
      afterwards cannot undo that, so asserting on these three needs a fresh
      interpreter.
    * **Read per call.** `_maybe_compile` reads `UNSLOTH_COMPILE_DISABLE` and
      `UNSLOTH_COMPILE_DISABLE_PARTIAL` in its own body on every call.
      `monkeypatch.setattr` on the module attributes is enough there, and is
      what the `_maybe_compile` tests below use; a subprocess would only buy a
      five second package import per assertion.

    A subprocess, not `importlib.reload`, because reloading the module
    mid-session rebinds objects other tests in the same run already hold.
    """
    env = dict(os.environ)
    if compile_disable is None:
        env.pop("UNSLOTH_COMPILE_DISABLE", None)
    else:
        env["UNSLOTH_COMPILE_DISABLE"] = compile_disable
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output = True, text = True, env = env,
        cwd = str(ZOO.parent),
    )
    assert done.returncode == 0, done.stdout + done.stderr


def _run_with_compile_enabled(body):
    _run_in_fresh_interpreter(body, None)


def _run_with_compile_disabled(body):
    _run_in_fresh_interpreter(body, "1")


# ---- what the compiler emits ---------------------------------------------

def test_no_emitter_writes_a_bare_fullgraph_compile():
    """Every emitted `fullgraph` decorator has to carry the fallback.

    Source-level because the alternative is generating a module per model, and
    one missed emitter is how the cross-entropy template kept its bare decorator
    after the first four were fixed."""
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


def test_maybe_compile_routes_fullgraph_through_the_fallback(monkeypatch):
    """Three more fullgraph regions go through this one helper, so wiring it
    covers them without touching each decorator."""
    from unsloth_zoo.temporary_patches import common
    src = (ZOO / "temporary_patches" / "common.py").read_text()
    body = src.split("def _maybe_compile(", 1)[1].split("\ndef ", 1)[0]
    assert "torch_compile_with_fallback" in body
    # Without fullgraph Dynamo already falls back by itself, so leave it alone.
    assert 'if not kwargs.get("fullgraph"):' in body

    # In process, unlike the alias below: `_maybe_compile` reads both switches
    # in its own body on every call, so pinning the module attributes is exact
    # and the assertion then holds in every configuration the suite runs under.
    monkeypatch.setattr(common, "UNSLOTH_COMPILE_DISABLE", False)
    monkeypatch.setattr(common, "UNSLOTH_COMPILE_DISABLE_PARTIAL", False)

    def f(x):
        return x
    wrapped = common._maybe_compile(fullgraph = True, dynamic = True)(f)
    assert hasattr(wrapped, "_unsloth_fallback_state")

    # The other direction needs a second observable. `_unsloth_fallback_state`
    # is stamped on only under fullgraph (utils.py returns the plain compiled
    # object before it), so its absence cannot tell a non-fullgraph compile that
    # wrongly went through the helper from one that did not. Record the call
    # instead: `_maybe_compile` resolves the name off `utils` at call time, so
    # rebinding the attribute is enough. Verified by replacing the guard's body
    # with `pass`, which the `hasattr` form did not notice in any switch
    # position and this does.
    import unsloth_zoo.temporary_patches.utils as U
    calls = []

    def _record(**kwargs):
        calls.append(kwargs)
        return lambda fn: fn

    monkeypatch.setattr(U, "torch_compile_with_fallback", _record)
    common._maybe_compile(fullgraph = True, dynamic = True)(f)
    assert calls, "the recorder never saw the fullgraph compile, so it proves nothing"
    calls.clear()
    common._maybe_compile(dynamic = True)(f)
    assert not calls, f"a non-fullgraph compile was wrapped: {calls}"


@pytest.mark.parametrize("flag", ["UNSLOTH_COMPILE_DISABLE",
                                  "UNSLOTH_COMPILE_DISABLE_PARTIAL"])
def test_maybe_compile_is_a_no_op_when_compilation_is_switched_off(monkeypatch, flag):
    """The other half of the switch. Turning compilation off has to drop the
    fallback wrapper as well, or torch is still handed a fullgraph region by the
    one helper three of them go through.

    Monkeypatch, again because the read is per call: setting the variable in the
    environment instead would do nothing to an already imported module, and a
    subprocess would be spelling `False -> True` the long way round."""
    from unsloth_zoo.temporary_patches import common

    monkeypatch.setattr(common, "UNSLOTH_COMPILE_DISABLE", False)
    monkeypatch.setattr(common, "UNSLOTH_COMPILE_DISABLE_PARTIAL", False)
    monkeypatch.setattr(common, flag, True)

    def f(x):
        return x
    assert common._maybe_compile(fullgraph = True, dynamic = True)(f) is f, \
        f"{flag} did not switch the fullgraph wrapper off"


def test_the_alias_is_the_no_op_when_compilation_is_switched_off():
    """The same half of the switch for the import-bound side.

    Nothing else pins this: the source scan below only recognises the enabled
    spelling, `functools.partial(...)`, so a disabled branch that bound bare
    `torch.compile` would pass every other test in this file while handing
    unsloth's CPU CI job the fullgraph regions it asked not to have. Needs a
    fresh interpreter for the reason in `_run_in_fresh_interpreter`: these three
    names are bound once, at import."""
    _run_with_compile_disabled("""
        from unsloth_zoo.temporary_patches import common

        for name in ("torch_compile", "_torch_compile", "_raw_torch_compile"):
            assert getattr(common, name) is common.noop, name
    """)


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


# aot_eager, not inductor: inductor shells out to a host compiler, so this
# raised `Compiler: cl is not found` on the Windows runner, a statement about
# MSVC and not about the wrapper. Any backend answers what is under test, that
# the wrapper returns what the eager function returns.
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


# --- what the seventh review round found ------------------------------------

def _bare_fullgraph_alias_sites():
    """Every `@torch_compile(... fullgraph = True)` in the package.

    The scan above reads three named files and only recognises a literal
    `torch.compile`, so the alias sites in gpt_oss / qwen3_vl_moe / gemma sat
    outside it: `torch_compile` was `functools.partial(torch.compile)`, straight
    to Dynamo, where cache exhaustion under fullgraph is fatal."""
    import re
    pattern = re.compile(r"\b_?torch_compile\s*\([^)]*fullgraph\s*=\s*True", re.S)
    found = []
    for path in sorted((ZOO / "temporary_patches").glob("*.py")):
        if path.name in ("common.py", "utils.py"):
            continue
        text = path.read_text(encoding = "utf-8")
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            if text.splitlines()[line - 1].lstrip().startswith("#"):
                continue
            found.append(f"{path.name}:{line}")
    return found


def test_the_alias_sites_exist_and_are_covered_by_the_alias_itself():
    """They are not rewritten one by one; the alias routes them. Fixing ten
    decorators leaves the eleventh, so `torch_compile` and `_torch_compile` go
    through `_compile_or_fall_back`, which hands any fullgraph compile to
    `torch_compile_with_fallback`."""
    sites = _bare_fullgraph_alias_sites()
    assert sites, "the scan found no alias sites at all; has the spelling changed?"

    common = (ZOO / "temporary_patches" / "common.py").read_text(encoding = "utf-8")
    assert "def _compile_or_fall_back" in common
    assert "torch_compile_with_fallback" in common
    # By line, not by substring: `_torch_compile = ...` is a substring of
    # `_raw_torch_compile = ...`, which deliberately DOES partial torch.compile
    # (compile_with_eager_fallback applies the wrapper itself, and wrapping a
    # wrapper leaves the inner one swallowing the exhaustion).
    lines = common.split("\n")
    for name in ("torch_compile", "_torch_compile"):
        starts = [i for i, ln in enumerate(lines)
                  if ln.strip().startswith(f"{name} = functools.partial(")]
        assert starts, f"{name} is no longer a partial; has the alias moved?"
        for i in starts:
            body = "\n".join(lines[i:i + 4])
            assert "_compile_or_fall_back" in body, \
                f"{name} still partials torch.compile directly, so {sites} stay fatal"
    raw = [ln for ln in lines
           if ln.strip().startswith("_raw_torch_compile = functools.partial(")]
    assert raw, "the raw alias compile_with_eager_fallback needs is gone"


def test_the_alias_routes_a_fullgraph_compile_through_the_fallback():
    _run_with_compile_enabled("""
        import unsloth_zoo.temporary_patches.utils as U
        seen = {}

        def _fake(**kwargs):
            seen.update(kwargs)
            return lambda fn: fn

        # `common` imports the helper lazily inside the call, so patching the
        # attribute on utils before the decorator runs is enough.
        U.torch_compile_with_fallback = _fake

        from unsloth_zoo.temporary_patches import common as C

        @C.torch_compile(dynamic = True, fullgraph = True)
        def _f(x): return x

        assert seen.get("fullgraph") is True, seen
    """)


def test_the_alias_still_accepts_a_function_positionally():
    """`prepare = torch_compile(prepare, fullgraph = True)` is how gemma.py and
    gpt_oss.py call it, and a kwargs-only helper would TypeError there."""
    from unsloth_zoo.temporary_patches import common as C

    def _f(x): return x * 2
    wrapped = C.torch_compile(_f, dynamic = True, fullgraph = True)
    assert callable(wrapped)
    assert wrapped(3) == 6


def test_a_non_fullgraph_compile_is_left_alone():
    """Dynamo already falls back by itself without fullgraph, so wrapping there
    would add a layer that can never fire.

    In a fresh interpreter, because in process this asserted nothing whenever
    the switch was off, which is the position unsloth's Core job runs the whole
    suite in: the alias is `noop` there, so the helper is unreachable and the
    recorder stays at zero however the routing is written. Removing the
    `if not kwargs.get("fullgraph")` guard from `_compile_or_fall_back` left the
    in-process form green under `UNSLOTH_COMPILE_DISABLE=1` and `partial`.

    `_torch_compile` comes along for free here; nothing else ran it. A positive
    control first, or every assertion below passes the moment the recorder
    stops recording."""
    _run_with_compile_enabled("""
        import unsloth_zoo.temporary_patches.utils as U
        calls = []

        def _record(**kwargs):
            calls.append(kwargs)
            return lambda fn: fn

        # Both aliases import the helper lazily inside the call, so rebinding
        # the attribute on utils before the decorator runs is enough.
        U.torch_compile_with_fallback = _record

        from unsloth_zoo.temporary_patches import common as C

        def _f(x): return x

        for name in ("torch_compile", "_torch_compile"):
            calls.clear()
            getattr(C, name)(dynamic = True, fullgraph = True)(_f)
            assert calls, f"the recorder never saw {name} under fullgraph"
            calls.clear()
            # `is _f` because the recorder's decorator is the identity: reaching
            # the helper is not enough, the function has to be handed to it
            # rather than the undecorated decorator returned to the caller.
            assert getattr(C, name)(_f, dynamic = True, fullgraph = True) is _f, \
                f"positional {name} did not apply the decorator"
            assert calls, f"the recorder never saw positional {name}"
            calls.clear()
            getattr(C, name)(dynamic = True)(_f)
            assert not calls, f"{name} wrapped a non-fullgraph compile: {calls}"

        # `_raw_torch_compile` is excluded on purpose: its one caller,
        # compile_with_eager_fallback, applies the wrapper itself, and wrapping a
        # wrapper leaves the inner one swallowing the exhaustion.
        calls.clear()
        C._raw_torch_compile(_f, fullgraph = True)
        assert not calls, calls
    """)

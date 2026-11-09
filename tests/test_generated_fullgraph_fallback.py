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
"""Generated fullgraph regions fall back to eager on recompile-limit hits, not FailOnRecompileLimitHit"""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback

pytestmark = pytest.mark.usefixtures("compile_enabled")

ZOO = Path(__file__).resolve().parents[1] / "unsloth_zoo"
COMPILER = ZOO / "compiler.py"
# Scanning compiler.py alone missed grpo_compute_loss_slow and accumulate_chunk, left bare and fatal.
FULLGRAPH_SITES = (COMPILER, ZOO / "rl_replacements.py",
                   ZOO / "temporary_patches" / "common.py")


def _run_in_fresh_interpreter(body, compile_disable):
    """Runs body in a fresh interpreter with UNSLOTH_COMPILE_DISABLE pinned (None clears); needed because CI sets it
    ambiently and common binds these names at import; not importlib.reload, which rebinds objects other tests hold"""
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



def test_no_emitter_writes_a_bare_fullgraph_compile():
    """Every emitted fullgraph decorator must carry the fallback; a source scan catches missed emitters"""
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
    """grpo_compute_loss_slow is emitted as source; accumulate_chunk is compiled at call time"""
    from unsloth_zoo.rl_replacements import RL_REPLACEMENTS
    slow = RL_REPLACEMENTS["grpo_compute_loss_slow"]
    assert "@torch_compile_with_fallback(" in slow
    assert "import torch_compile_with_fallback" in slow, \
        "the name has to resolve inside the generated module"
    src = (ZOO / "rl_replacements.py").read_text()
    assert "torch_compile_with_fallback(\n            fullgraph = True," in src, \
        "accumulate_chunk is still compiled bare"


def test_maybe_compile_routes_fullgraph_through_the_fallback(monkeypatch):
    """Three more fullgraph regions go through _maybe_compile, so wiring the helper covers them all."""
    from unsloth_zoo.temporary_patches import common
    src = (ZOO / "temporary_patches" / "common.py").read_text()
    body = src.split("def _maybe_compile(", 1)[1].split("\ndef ", 1)[0]
    assert "torch_compile_with_fallback" in body
    # Without fullgraph Dynamo already falls back by itself, so leave it alone.
    assert 'if not kwargs.get("fullgraph"):' in body

    # _maybe_compile reads both switches per call, so monkeypatching the module attributes is exact.
    monkeypatch.setattr(common, "UNSLOTH_COMPILE_DISABLE", False)
    monkeypatch.setattr(common, "UNSLOTH_COMPILE_DISABLE_PARTIAL", False)

    def f(x):
        return x
    wrapped = common._maybe_compile(fullgraph = True, dynamic = True)(f)
    assert hasattr(wrapped, "_unsloth_fallback_state")

    # _unsloth_fallback_state is stamped only under fullgraph, so hasattr misses non-fullgraph calls.
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
    """With compilation off the fallback wrapper must also drop, or torch still gets fullgraph regions."""
    from unsloth_zoo.temporary_patches import common

    monkeypatch.setattr(common, "UNSLOTH_COMPILE_DISABLE", False)
    monkeypatch.setattr(common, "UNSLOTH_COMPILE_DISABLE_PARTIAL", False)
    monkeypatch.setattr(common, flag, True)

    def f(x):
        return x
    assert common._maybe_compile(fullgraph = True, dynamic = True)(f) is f, \
        f"{flag} did not switch the fullgraph wrapper off"


def test_the_alias_is_the_no_op_when_compilation_is_switched_off():
    """Import-bound aliases must be no-ops if compilation is off; bare torch.compile evades the scan"""
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



def test_fullgraph_false_is_returned_untouched():
    """Without fullgraph Dynamo falls back itself, so a wrapper would never fire and only add a frame."""
    def f(x):
        return x + 1

    wrapped = torch_compile_with_fallback(fullgraph = False)(f)
    assert not hasattr(wrapped, "_unsloth_fallback_state")


def test_fullgraph_true_is_wrapped():
    def f(x):
        return x + 1

    wrapped = torch_compile_with_fallback(fullgraph = True)(f)
    assert hasattr(wrapped, "_unsloth_fallback_state")


# aot_eager, not inductor: inductor shells out to a host compiler (cl missing on Windows)
_BACKEND = "aot_eager"


def test_the_wrapper_still_computes_the_right_answer():
    def f(x):
        return x * 2 + 1

    wrapped = torch_compile_with_fallback(fullgraph = True, backend = _BACKEND)(f)
    x = torch.arange(4, dtype = torch.float32)
    assert torch.equal(wrapped(x), f(x))


def test_the_helper_takes_a_backend_like_torch_compile_does():
    """Forwards **compile_kwargs (generated decorators need dynamic and options); dropping one miscompiles"""
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



def _bare_fullgraph_alias_sites():
    """Finds torch_compile(fullgraph=True) alias sites the literal scan misses; cache exhaustion is fatal"""
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
    """Alias sites are covered by routing torch_compile and _torch_compile through _compile_or_fall_back"""
    sites = _bare_fullgraph_alias_sites()
    assert sites, "the scan found no alias sites at all; has the spelling changed?"

    common = (ZOO / "temporary_patches" / "common.py").read_text(encoding = "utf-8")
    assert "def _compile_or_fall_back" in common
    assert "torch_compile_with_fallback" in common
    # Match by line, not substring: `_torch_compile = ` occurs inside `_raw_torch_compile = `, which deliberately does
    # partial torch.compile; compile_with_eager_fallback applies that wrapper itself, and wrapping a wrapper leaves the
    # inner one swallowing exhaustion.
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
    """torch_compile takes fn positionally (gemma.py, gpt_oss.py); kwargs-only would TypeError"""
    from unsloth_zoo.temporary_patches import common as C

    def _f(x): return x * 2
    wrapped = C.torch_compile(_f, dynamic = True, fullgraph = True)
    assert callable(wrapped)
    assert wrapped(3) == 6


def test_a_non_fullgraph_compile_is_left_alone():
    """Non-fullgraph compiles skip the wrapper; run in a fresh interpreter (in-process it passes vacuously when compile
    is off) and take a positive control first, or every assertion passes once the recorder stops recording"""
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

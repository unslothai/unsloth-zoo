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

import json
import os
import re
import subprocess
import sys
import textwrap
from functools import lru_cache
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


def test_maybe_compile_asks_for_the_fallback_only_under_fullgraph():
    """Three more fullgraph regions go through this one helper, so wiring it
    covers them without touching each decorator. What it does at runtime is in
    `test_every_fullgraph_entry_point_carries_the_fallback` below, which has to
    pick the compile switch's position rather than inherit the runner's."""
    src = (ZOO / "temporary_patches" / "common.py").read_text()
    body = src.split("def _maybe_compile(", 1)[1].split("\ndef ", 1)[0]
    assert "torch_compile_with_fallback" in body
    # Without fullgraph Dynamo already falls back by itself, so leave it alone.
    assert 'if not kwargs.get("fullgraph"):' in body


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


def test_the_alias_still_accepts_a_function_positionally():
    """`prepare = torch_compile(prepare, fullgraph = True)` is how gemma.py and
    gpt_oss.py call it, and a kwargs-only helper would TypeError there.

    True in either position of the compile switch: with it off the alias is
    `noop`, which takes a function positionally too."""
    from unsloth_zoo.temporary_patches import common as C

    def _f(x): return x * 2
    wrapped = C.torch_compile(_f, dynamic = True, fullgraph = True)
    assert callable(wrapped)
    assert wrapped(3) == 6


# ---- routing, with the compile switch put in each position -----------------
# `UNSLOTH_COMPILE_DISABLE` is read into module constants at import, and both
# the alias and `_maybe_compile` short-circuit on it, so anything checked in
# this process only ever sees the position the runner happened to start in.
# unsloth's consolidated-tests-ci.yml pins the flag to 1 for the whole job,
# where the two runtime checks that used to live here read a correct no-op as a
# missing fallback and went red on every unsloth PR. Set the position in a
# subprocess instead, same shape as tests/test_rl_replacements_compile_disable.py,
# and assert both halves of the contract: routed when compiling, nothing
# compiled at all when the escape hatch is on.

_ROUTING_SCRIPT = textwrap.dedent("""
    import json
    from unsloth_zoo.temporary_patches import common as C

    def _f(x): return x

    # `torch_compile_with_fallback` is the only thing that stamps this on, so
    # its presence is the routing, not a spelling of it.
    def _routed(fn): return hasattr(fn, "_unsloth_fallback_state")
    def _dynamo_off(fn): return getattr(fn, "_torchdynamo_disable", False) is True

    out = {"flag": bool(C.UNSLOTH_COMPILE_DISABLE)}
    for name, alias in (("torch_compile", C.torch_compile),
                        ("_torch_compile", C._torch_compile)):
        out[name + "_fullgraph"]     = _routed(alias(dynamic = True, fullgraph = True)(_f))
        out[name + "_positional"]    = _routed(alias(_f, dynamic = True, fullgraph = True))
        out[name + "_plain"]         = _routed(alias(dynamic = True)(_f))
        out[name + "_fullgraph_off"] = _dynamo_off(alias(dynamic = True, fullgraph = True)(_f))
    out["maybe_fullgraph"]  = _routed(C._maybe_compile(fullgraph = True, dynamic = True)(_f))
    out["maybe_plain"]      = _routed(C._maybe_compile(dynamic = True)(_f))
    out["maybe_identity"]   = C._maybe_compile(fullgraph = True, dynamic = True)(_f) is _f
    # `_raw_torch_compile` deliberately does NOT route: compile_with_eager_fallback
    # applies the wrapper itself and wrapping a wrapper leaves the inner one
    # swallowing the exhaustion.
    out["raw_fullgraph"] = _routed(C._raw_torch_compile(_f, fullgraph = True))

    # `_unsloth_fallback_state` is stamped on only under fullgraph, so it cannot
    # tell a non-fullgraph compile that wrongly went through the helper from one
    # that did not: `torch_compile_with_fallback(fullgraph = False)` hands back a
    # plain compiled object either way. Record the calls instead. Both routers
    # resolve the name off `utils` at call time, so rebinding it is enough.
    import unsloth_zoo.temporary_patches.utils as U
    calls = []
    _real = U.torch_compile_with_fallback
    def _record(**kwargs):
        calls.append(kwargs)
        return lambda fn: fn
    U.torch_compile_with_fallback = _record
    try:
        for name, alias in (("torch_compile", C.torch_compile),
                            ("_torch_compile", C._torch_compile)):
            for label, kw in (("plain", dict(dynamic = True)),
                              ("fullgraph", dict(dynamic = True, fullgraph = True))):
                calls.clear()
                alias(**kw)(_f)
                out[f"{name}_{label}_reached_helper"] = bool(calls)
        for label, kw in (("plain", dict(dynamic = True)),
                          ("fullgraph", dict(fullgraph = True, dynamic = True))):
            calls.clear()
            C._maybe_compile(**kw)(_f)
            out[f"maybe_{label}_reached_helper"] = bool(calls)
        calls.clear()
        C._raw_torch_compile(_f, fullgraph = True)
        out["raw_reached_helper"] = bool(calls)
    finally:
        U.torch_compile_with_fallback = _real
    print("PROBE " + json.dumps(out))
""")


@lru_cache(maxsize = None)
def _routing_probe(value):
    env = dict(
        os.environ,
        # Prepend, never replace: the checkout has to win over an installed
        # copy, but the parent process may be carrying paths the import needs.
        PYTHONPATH = os.pathsep.join(
            [str(ZOO.parent)] + [p for p in [os.environ.get("PYTHONPATH", "")] if p]
        ),
        UNSLOTH_COMPILE_DISABLE = value,
        # unsloth_zoo/__init__ calls get_device_type() at import and raises on a
        # GPU-less runner; same escape the sibling compile-disable probe uses.
        UNSLOTH_ZOO_DISABLE_GPU_INIT = "1",
    )
    r = subprocess.run([sys.executable, "-c", _ROUTING_SCRIPT],
                       capture_output = True, text = True, timeout = 900, env = env)
    line = [l for l in r.stdout.splitlines() if l.startswith("PROBE ")]
    assert line, (r.stdout[-2000:], r.stderr[-3000:])
    return json.loads(line[0][len("PROBE "):])


def test_every_fullgraph_entry_point_carries_the_fallback():
    """`torch_compile`, `_torch_compile` and `_maybe_compile` are the three ways
    a fullgraph region is built, and all three have to reach the wrapper. The
    decorator and positional spellings are both in use (gemma.py, gpt_oss.py)."""
    got = _routing_probe("0")
    assert got["flag"] is False, got
    for key in ("torch_compile_fullgraph", "torch_compile_positional",
                "_torch_compile_fullgraph", "_torch_compile_positional",
                "maybe_fullgraph"):
        assert got[key] is True, f"{key} does not reach torch_compile_with_fallback: {got}"


def test_a_non_fullgraph_compile_is_left_alone():
    """Dynamo already falls back by itself without fullgraph, so wrapping there
    would add a layer that can never fire. `_raw_torch_compile` is excluded on
    purpose; its one caller applies the wrapper itself."""
    got = _routing_probe("0")
    # Positive control first: without it every assertion below passes for free
    # the moment the call recorder stops recording.
    for key in ("torch_compile_fullgraph_reached_helper",
                "_torch_compile_fullgraph_reached_helper",
                "maybe_fullgraph_reached_helper"):
        assert got[key] is True, f"the recorder never saw {key}: {got}"
    for key in ("torch_compile_plain_reached_helper",
                "_torch_compile_plain_reached_helper",
                "maybe_plain_reached_helper", "raw_reached_helper"):
        assert got[key] is False, f"{key} was wrapped and cannot ever fire: {got}"
    for key in ("torch_compile_plain", "_torch_compile_plain", "maybe_plain",
                "raw_fullgraph"):
        assert got[key] is False, f"{key} carries a fallback state it cannot use: {got}"


@pytest.mark.parametrize("value", ["1", "partial"])
def test_the_escape_hatch_still_turns_compilation_off(value):
    """The routing must not become a way back in. With the flag set nothing is
    compiled: the aliases hand back a Dynamo-disabled callable and
    `_maybe_compile` hands back the function itself, so no wrapper is stamped
    on. This is the position unsloth's Core job runs the whole suite in."""
    got = _routing_probe(value)
    assert got["flag"] is True, got
    for key in ("torch_compile_fullgraph", "_torch_compile_fullgraph",
                "maybe_fullgraph"):
        assert got[key] is False, f"{key} routed with compilation disabled: {got}"
    for key in ("torch_compile_fullgraph_off", "_torch_compile_fullgraph_off"):
        assert got[key] is True, f"{key} still reaches Dynamo: {got}"
    assert got["maybe_identity"] is True, got

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

"""`inspect.getsource` raising `tokenize.TokenError` must not fail a model load.
TokenError subclasses Exception directly, so it escaped the OSError/TypeError catches;
it fires when a generated compile-folder file is read mid-rewrite. Subprocesses: the
compiler mutates torch.nn process-wide and reads UNSLOTH_COMPILE_LOCATION at import.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import tokenize
from pathlib import Path

import pytest


pytest.importorskip("transformers")
pytest.importorskip("unsloth_zoo.compiler")

ROOT = Path(__file__).resolve().parents[1]


def _run(body: str, cache_dir: Path, timeout: int = 900):
    script = textwrap.dedent(
        f"""
        import importlib, json, os, sys, tokenize
        sys.path.insert(0, {str(ROOT)!r})
        import unsloth_zoo.compiler as compiler

        CACHE = os.environ["UNSLOTH_COMPILE_LOCATION"]

        def fresh_llama():
            mod = importlib.import_module(
                "transformers.models.llama.modeling_llama",
            )
            if hasattr(mod, "__UNSLOTH_PATCHED__"):
                delattr(mod, "__UNSLOTH_PATCHED__")
            return mod

        # Warm-up on real source. This is what replaces the torch.nn forwards
        # with generated ones, i.e. what puts the compile folder on the far end
        # of a later inspect.getsource.
        fresh_llama()
        compiler.unsloth_compile_transformers("llama", disable=True)
        """
    ) + textwrap.dedent(body)

    cache_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["UNSLOTH_COMPILE_LOCATION"] = str(cache_dir)
    env["UNSLOTH_COMPILE_DISABLE"] = "1"
    # On a CPU-only runner the zoo's get_device_type raises "Unsloth cannot find
    # any torch accelerator" during the child's import, before either TokenError
    # handler is reached. Measured: neuter tests/conftest.py's UNSLOTH_ALLOW_CPU
    # setdefault and all three tests here die on the import.
    env.setdefault("UNSLOTH_ALLOW_CPU", "1")
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=timeout, env=env,
    )


def _result(proc, what: str):
    if proc.returncode != 0:
        pytest.fail(
            f"{what}: the probe died, so tokenize.TokenError escaped "
            f"unsloth_compile_transformers instead of being handled.\n"
            f"STDOUT:\n{proc.stdout[-2500:]}\nSTDERR:\n{proc.stderr[-3000:]}"
        )
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        pytest.fail(
            f"{what}: probe produced no RESULT line.\n"
            f"STDOUT:\n{proc.stdout[-2500:]}\nSTDERR:\n{proc.stderr[-3000:]}"
        )
    return json.loads(line[-1][len("RESULT "):])


def test_tokenerror_is_not_an_oserror_or_typeerror():
    """The catch must name TokenError explicitly: it subclasses neither."""
    assert not issubclass(tokenize.TokenError, OSError)
    assert not issubclass(tokenize.TokenError, TypeError)


def test_compile_transformers_survives_tokenerror_from_getsource(tmp_path):
    """Every getsource raises; the pipeline must still return."""
    proc = _run(
        """
        calls = []

        def raise_tokenerror(obj, *args, **kwargs):
            calls.append(getattr(obj, "__name__", repr(obj)))
            raise tokenize.TokenError("EOF in multi-line string", (1, 0))

        compiler.inspect.getsource = raise_tokenerror

        mod = fresh_llama()
        compiler.unsloth_compile_transformers("llama", disable=True)

        print("RESULT " + json.dumps({
            "calls": len(calls),
            "supports_sdpa": getattr(mod, "__UNSLOTH_SUPPORTS_SDPA__", None),
        }))
        """,
        tmp_path / "cache_all",
    )
    out = _result(proc, "all-getsource-raises")

    assert out["calls"] > 0, (
        "inspect.getsource was never called, so the probe proved nothing "
        "about the TokenError handlers"
    )
    assert out["supports_sdpa"] is False, (
        f"the unreadable-source branch must set __UNSLOTH_SUPPORTS_SDPA__ "
        f"False, got {out['supports_sdpa']!r}"
    )


# The markers #967 added to every forward _patch_torch_dtype_modules installs.
# Dropping them puts a compile-folder forward back in the state the patcher treats
# as "not mine yet", which is what the rewrite path below needs to be reachable.
_STRIP_DTYPE_MARKERS = """
        for _name in compiler._patch_functions:
            _cls = getattr(torch.nn, _name, None)
            _fwd = getattr(_cls, "forward", None) if _cls is not None else None
            if _fwd is None:
                continue
            for _marker in (
                "__unsloth_dtype_wrapped__",
                "__unsloth_dtype_disable__",
                "__unsloth_dtype_original__",
            ):
                if hasattr(_fwd, _marker):
                    delattr(_fwd, _marker)
"""


def test_a_second_pass_does_not_read_its_own_generated_forward(tmp_path):
    """#967's idempotence guarantee, which is what makes the probe below artificial.

    The warm-up leaves every patched torch.nn forward served out of the compile
    folder AND marked `__unsloth_dtype_wrapped__`. A second pass must recognise its
    own output and skip it, because re-reading it would stack a second dtype
    prologue and return a bf16 activation as fp32. Measured on this tree: 12 of 12
    generated forwards are marked, so the second pass calls getsource on none.
    """
    proc = _run(
        """
        import torch

        generated = [
            name for name in compiler._patch_functions
            if getattr(getattr(getattr(torch.nn, name, None), "forward", None),
                       "__code__", None) is not None
            and getattr(torch.nn, name).forward.__code__.co_filename.startswith(CACHE)
        ]
        marked = [
            name for name in generated
            if getattr(torch.nn.__dict__[name].forward,
                       "__unsloth_dtype_wrapped__", False)
        ]

        real_getsource = compiler.inspect.getsource
        hits = []

        def record_generated(obj, *args, **kwargs):
            code = getattr(obj, "__code__", None)
            if code is not None and code.co_filename.startswith(CACHE):
                hits.append(code.co_filename)
            return real_getsource(obj, *args, **kwargs)

        compiler.inspect.getsource = record_generated

        fresh_llama()
        compiler.unsloth_compile_transformers("llama", disable=True)

        print("RESULT " + json.dumps({
            "generated": len(generated),
            "marked": len(marked),
            "hits": len(hits),
        }))
        """,
        tmp_path / "cache_idempotent",
    )
    out = _result(proc, "second-pass-skips-own-output")

    assert out["generated"] > 0, (
        "no torch.nn forward was served out of the compile folder after a "
        "warm-up compile, so this probe proves nothing about a second pass"
    )
    assert out["marked"] == out["generated"], (
        f"{out['generated'] - out['marked']} generated forwards carry no "
        f"__unsloth_dtype_wrapped__ marker, so a second pass would rewrite them "
        f"again and stack a dtype prologue"
    )
    assert out["hits"] == 0, (
        f"a second pass called inspect.getsource on {out['hits']} compile-folder "
        f"forwards, so it is reading back its own generated output"
    )


def test_dtype_patcher_survives_tokenerror_on_its_generated_forward(tmp_path):
    """The handler itself: getsource raising on a compile-folder forward must not fail the load.

    The markers are stripped first. Before #967 a second pass reached this on its
    own, which is how the case was found; #967 closed that route deliberately, so
    reproducing the state has to be explicit now. The handler is still live for a
    forward built by exec or one whose file has gone, and the contract it owes is
    unchanged: wrap rather than skip, so the casts survive without the source.
    """
    proc = _run(
        """
        import torch
        """ + _STRIP_DTYPE_MARKERS + """
        generated = sorted({
            code.co_filename
            for code in (
                getattr(getattr(torch.nn, name).forward, "__code__", None)
                for name in compiler._patch_functions
                if hasattr(torch.nn, name)
            )
            if code is not None and code.co_filename.startswith(CACHE)
        })

        real_getsource = compiler.inspect.getsource
        hits = []

        def raise_on_generated(obj, *args, **kwargs):
            code = getattr(obj, "__code__", None)
            if code is not None and code.co_filename.startswith(CACHE):
                hits.append(code.co_filename)
                raise tokenize.TokenError("EOF in multi-line statement", (1, 0))
            return real_getsource(obj, *args, **kwargs)

        compiler.inspect.getsource = raise_on_generated

        fresh_llama()
        compiler.unsloth_compile_transformers("llama", disable=True)
        compiler.inspect.getsource = real_getsource

        # Skipping instead of wrapping would only move the failure here.
        ran = tuple(torch.nn.LayerNorm(4)(torch.randn(2, 4)).shape) == (2, 4)

        print("RESULT " + json.dumps({
            "generated": len(generated),
            "hits": len(hits),
            "forward_runs": ran,
        }))
        """,
        tmp_path / "cache_generated",
    )
    out = _result(proc, "generated-forward-getsource-raises")

    assert out["generated"] > 0, (
        "no torch.nn forward was served out of the compile folder after a "
        "warm-up compile, so this probe does not cover the concurrent-rewrite "
        "case it exists for"
    )
    assert out["hits"] > 0, (
        "no getsource call resolved into the compile folder, so the dtype "
        "patcher's TokenError handler was never exercised. If the marker names "
        "in _STRIP_DTYPE_MARKERS have drifted from compiler.py, this is what "
        "that looks like"
    )
    assert out["forward_runs"], (
        "a torch.nn forward stopped working after getsource raised, so the "
        "handler skipped the module instead of wrapping it"
    )


def test_the_probe_sets_cpu_mode_itself_rather_than_inheriting_it():
    """The subprocess must not depend on conftest for its own importability.

    A fresh interpreter inherits none of pytest's setup. These probes ran on a
    CPU-only runner only because tests/conftest.py sets UNSLOTH_ALLOW_CPU at
    import time and dict(os.environ) happened to carry it into the child;
    neutering that single line makes every test in this file die inside the
    child's `import unsloth_zoo.compiler`, before the handlers under test.

    The checkout pin is already handled: the generated script starts with
    sys.path.insert(0, ROOT), so the child imports this tree and not an
    installed copy. Only the CPU flag was missing.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    setup = source[source.index("env = dict(os.environ)"):source.index("return subprocess.run(")]
    assert 'env.setdefault("UNSLOTH_ALLOW_CPU", "1")' in setup
    assert "sys.path.insert(0, {str(ROOT)!r})" in source, (
        "the child must keep pinning this checkout, or it validates a "
        "different compiler.py than the one under review"
    )


def test_the_stripped_markers_are_the_ones_compiler_sets():
    """A rename in compiler.py must fail here, not quietly disarm the probe above.

    Stripping names nothing sets would leave every forward marked, the second pass
    would skip them all, and the TokenError test would go back to reporting zero
    hits -- the exact failure this file was carrying on main.
    """
    compiler_source = (ROOT / "unsloth_zoo" / "compiler.py").read_text(encoding="utf-8")
    for marker in (
        "__unsloth_dtype_wrapped__",
        "__unsloth_dtype_disable__",
        "__unsloth_dtype_original__",
    ):
        assert marker in _STRIP_DTYPE_MARKERS, f"{marker} is not stripped by the probe"
        assert marker in compiler_source, (
            f"{marker} is stripped by the probe but compiler.py no longer sets it"
        )

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


def test_dtype_patcher_survives_tokenerror_on_its_generated_forward(tmp_path):
    """Raises only on the generated forward, so the pipeline reaches the dtype patcher."""
    proc = _run(
        """
        import torch

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

        print("RESULT " + json.dumps({
            "generated": len(generated),
            "hits": len(hits),
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
        "patcher's TokenError handler was never exercised"
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

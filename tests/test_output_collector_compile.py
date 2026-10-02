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


"""A capture hook inside a compiled submodule reads the eagerly set collector without a graph break."""
import contextvars
import subprocess
import sys
import textwrap
import threading

import pytest
import torch

output_capturing = pytest.importorskip("transformers.utils.output_capturing")
if not hasattr(output_capturing, "CompileableContextVar"):
    pytest.skip("this transformers has no CompileableContextVar", allow_module_level = True)

_PROBE = textwrap.dedent("""
    import torch
    from transformers.utils.output_capturing import CompileableContextVar
    {patch}
    var = CompileableContextVar("probe")

    @torch.compile(fullgraph = True, backend = "eager")
    def hook(x):
        collected = var.get()
        collected["k"].append(x * 2)
        return x + 1

    collected = {{"k": []}}
    token = var.set(collected)
    try:
        hook(torch.ones(2))
    finally:
        var.reset(token)
    assert len(collected["k"]) == 1 and torch.equal(collected["k"][0], torch.full((2,), 2.0))
    print("FULLGRAPH_OK")
""")


def _run(patch):
    code = _PROBE.format(patch = patch)
    return subprocess.run([sys.executable, "-c", code], capture_output = True, text = True, timeout = 600)


def test_compiled_reader_sees_the_eager_collector_without_a_graph_break():
    patch = (
        "from unsloth_zoo.temporary_patches.misc import patch_output_collector_for_compiled_submodules\n"
        "patch_output_collector_for_compiled_submodules()"
    )
    out = _run(patch)
    assert "FULLGRAPH_OK" in out.stdout, out.stderr[-3000:]


def test_unpatched_reader_breaks_the_graph():
    out = _run("")
    if "FULLGRAPH_OK" in out.stdout:
        pytest.skip(f"torch {torch.__version__} traces ContextVar.get, so there is no graph break to remove")
    assert "ContextVar" in out.stderr, out.stderr[-3000:]


def _patched_var(name):
    from unsloth_zoo.temporary_patches.misc import patch_output_collector_for_compiled_submodules
    patch_output_collector_for_compiled_submodules()
    return output_capturing.CompileableContextVar(name)


def _compiled_get(var, monkeypatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    try:
        return var.get()
    finally:
        monkeypatch.undo()


def test_eager_reads_stay_per_thread_and_nesting_restores(monkeypatch):
    var = _patched_var("eager")
    outer, inner = {"k": []}, {"k": []}
    t_outer = var.set(outer)
    seen = []
    thread = threading.Thread(target = lambda: seen.append(var.get()))
    thread.start(); thread.join()
    assert seen == [None] and var.get() is outer
    t_inner = var.set(inner)
    assert var.get() is inner and _compiled_get(var, monkeypatch) is inner
    var.reset(t_inner)
    assert var.get() is outer and _compiled_get(var, monkeypatch) is outer
    var.reset(t_outer)
    assert var.get() is None and _compiled_get(var, monkeypatch) is None


def test_overlapping_threads_never_read_each_others_collector(monkeypatch):
    var = _patched_var("threads")
    a, b = {"k": []}, {"k": []}
    steps = {name: threading.Event() for name in ("a_set", "b_set", "a_reset", "b_done")}
    seen = {}

    def thread_a():
        token = var.set(a)
        steps["a_set"].set(); steps["b_set"].wait()
        var.reset(token)
        steps["a_reset"].set()

    def thread_b():
        steps["a_set"].wait()
        token = var.set(b)
        seen["overlap"] = var._unsloth_eager_single
        steps["b_set"].set(); steps["a_reset"].wait()
        seen["after_a_reset"] = (var._unsloth_eager_single, var._unsloth_eager_value is b)
        var.reset(token)
        steps["b_done"].set()

    threads = [threading.Thread(target = thread_a), threading.Thread(target = thread_b)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert seen["overlap"] is False
    assert seen["after_a_reset"] == (True, True)
    assert var._unsloth_eager_single and _compiled_get(var, monkeypatch) is None


def test_overlapping_same_thread_contexts_never_read_each_others_collector(monkeypatch):
    # asyncio tasks or greenlets share one thread but each runs in its own context.
    var = _patched_var("contexts")
    a, b = {"k": []}, {"k": []}
    ctx_a, ctx_b = contextvars.copy_context(), contextvars.copy_context()
    token_a = ctx_a.run(var.set, a)
    token_b = ctx_b.run(var.set, b)
    assert var._unsloth_eager_single is False
    assert ctx_a.run(_compiled_get, var, monkeypatch) is a
    ctx_a.run(var.reset, token_a)
    assert var._unsloth_eager_single and _compiled_get(var, monkeypatch) is b
    ctx_b.run(var.reset, token_b)
    assert _compiled_get(var, monkeypatch) is None

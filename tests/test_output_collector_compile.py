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


"""A capture hook inside a compiled submodule must read the collector capture_outputs set eagerly
without a graph break (TRL >= 1 captures MoE router logits for its aux loss)."""
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
    assert "FULLGRAPH_OK" not in out.stdout
    assert "ContextVar" in out.stderr, out.stderr[-3000:]


def test_eager_reads_stay_per_thread_and_nesting_restores():
    from unsloth_zoo.temporary_patches.misc import patch_output_collector_for_compiled_submodules
    patch_output_collector_for_compiled_submodules()
    var = output_capturing.CompileableContextVar("eager")
    outer, inner = {"k": []}, {"k": []}
    t_outer = var.set(outer)
    seen = []
    thread = threading.Thread(target = lambda: seen.append(var.get()))
    thread.start(); thread.join()
    assert seen == [None] and var.get() is outer
    t_inner = var.set(inner)
    assert var.get() is inner and var._unsloth_eager_value is inner
    var.reset(t_inner)
    assert var.get() is outer and var._unsloth_eager_value is outer
    var.reset(t_outer)
    assert var.get() is None and var._unsloth_eager_value is None

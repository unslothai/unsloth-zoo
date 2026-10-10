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

"""Patched F.layer_norm under a Dynamo trace after init_process_group rebuilt the rule map (vLLM Qwen3-VL ViT)."""

import os
import subprocess
import sys
import textwrap

import torch

_CHILD = textwrap.dedent(
    """
    import torch, torch.nn.functional as F
    from unsloth_zoo.patch_torch_functions import patch_torch_functions
    patch_torch_functions()
    import torch._dynamo
    # What torch.distributed.init_process_group does: the rebuilt map lists the replacement as in-graph.
    torch._dynamo.trace_rules.clear_lru_cache()
    assert torch._dynamo.trace_rules.lookup(F.layer_norm).__name__ == "TorchInGraphFunctionVariable"

    def f(x, w, b):
        return F.layer_norm(x * 2, (x.shape[-1],), w, b, 1e-6) + 1

    x, w, b = torch.randn(7, 64), torch.randn(64), torch.randn(64)
    torch._dynamo.mark_dynamic(x, 0)
    y = torch.compile(f, dynamic = True)(x, w, b)
    ref = F._uncompiled_layer_norm(x * 2, (64,), w, b, 1e-6) + 1
    torch.testing.assert_close(y, ref)
    print("CHILD_OK")
    """
)


_CHILD_NO_RESET = textwrap.dedent(
    """
    import torch, torch.nn.functional as F
    from unsloth_zoo.patch_torch_functions import patch_torch_functions
    patch_torch_functions()
    import torch._dynamo
    from torch._dynamo.utils import counters

    def f(x, w, b):
        return F.layer_norm(x * 2, (x.shape[-1],), w, b, 1e-6) + 1

    x, w, b = torch.randn(7, 64), torch.randn(64), torch.randn(64)
    counters.clear()
    y = torch.compile(f)(x, w, b)
    torch.testing.assert_close(y, F._uncompiled_layer_norm(x * 2, (64,), w, b, 1e-6) + 1)
    assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])
    print("CHILD_OK")
    """
)


def _run_child(code):
    env = dict(os.environ, UNSLOTH_IS_PRESENT = "1")
    out = subprocess.run(
        [sys.executable, "-c", code], env = env, capture_output = True, text = True, timeout = 600,
    )
    assert out.returncode == 0 and "CHILD_OK" in out.stdout, (out.returncode, out.stderr[-2000:])


def test_patched_layer_norm_does_not_break_the_graph_it_is_traced_in():
    _run_child(_CHILD_NO_RESET)


def test_patched_layer_norm_survives_in_graph_classification():
    _run_child(_CHILD)


def test_fake_mode_probe_is_false_for_real_tensors():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from unsloth_zoo.patch_torch_functions import _in_fake_or_proxy_mode

    assert not _in_fake_or_proxy_mode(torch.randn(2, 3))
    with FakeTensorMode():
        assert _in_fake_or_proxy_mode(torch.empty(2, 3))

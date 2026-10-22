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

"""The below-sm80 device set is built on an eager call, never inside a trace: torch.cuda.device_count()
and get_device_capability() there break the graph, and fullgraph raises. A trace that asks first takes
the below-sm80 answer (one-token decode only), and the eager prefill builds the set for the decode."""
import os
import types

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
pytest.importorskip("triton")

from unsloth_zoo.temporary_patches import moe_routed as MR  # noqa: E402
from unsloth_zoo.temporary_patches import gpt_oss_routed as GR  # noqa: E402


def test_a_cold_traced_call_compiles_fullgraph(monkeypatch):
    monkeypatch.setattr(MR, "_SUB_SM80_DEVICES", None)

    def f(x):
        sub = MR.sub_sm80_devices()
        return x + (1 if sub is None else 3 if sub else 2)

    torch._dynamo.reset()
    out = torch.compile(f, fullgraph = True, backend = "eager")(torch.zeros(2))
    torch._dynamo.reset()
    assert out.tolist() == [1.0, 1.0]
    assert MR._SUB_SM80_DEVICES is None


def test_a_traced_call_before_the_set_takes_the_below_sm80_answer(monkeypatch):
    monkeypatch.setattr(MR, "_SUB_SM80_DEVICES", None)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    def no_query(*args, **kwargs):
        raise AssertionError("torch.cuda queried while tracing")

    monkeypatch.setattr(torch.cuda, "device_count", no_query)
    monkeypatch.setattr(torch.cuda, "get_device_capability", no_query)
    assert MR.sub_sm80_devices() is None
    assert MR.routed_16bit_supported(torch.device("cuda", 0)) is False


def test_the_eager_prefill_builds_the_set(monkeypatch):
    monkeypatch.setattr(MR, "_SUB_SM80_DEVICES", None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i = None: (7, 5) if i == 1 else (9, 0))
    monkeypatch.setattr(GR, "_routed_disabled", lambda: False)
    hidden = types.SimpleNamespace(is_cuda = True, shape = (1, 64, 8), numel = lambda: 64 * 8)
    mlp = types.SimpleNamespace(experts = object(), router = types.SimpleNamespace(top_k = 4))
    with torch.no_grad():
        assert GR.routed_mlp_forward(mlp, hidden, max_slots = 16) is None   # prefill: too large to route
    assert MR._SUB_SM80_DEVICES == frozenset({1})

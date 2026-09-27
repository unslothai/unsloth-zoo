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

"""_Gemma4KVSharedSafeProxy is built inside compiled gemma-4 forwards via
get_text_config() (26B-A4B / 31B, num_kv_shared_layers == 0): it must trace."""
import copy
import pickle

import pytest
import torch

from unsloth_zoo.temporary_patches import gemma4 as g4


class _Cfg:
    def __init__(self):
        self.num_kv_shared_layers = 0
        self.hidden_size = 8
        self.final_logit_softcapping = 30.0
        self.layer_types = ["sliding_attention", "full_attention"]


def _count_breaks(fn, *args):
    from torch._dynamo.utils import counters
    torch._dynamo.reset()
    counters.clear()
    out = torch.compile(fn, backend = "eager")(*args)
    n = sum(counters["graph_break"].values()) + sum(counters["unimplemented"].values())
    return out, n


def test_proxy_reads_trace_without_graph_break():
    cfg = _Cfg()

    def f(x):
        p = g4._Gemma4KVSharedSafeProxy(cfg)
        y = x * p.hidden_size + p.final_logit_softcapping
        if hasattr(p, "num_kv_shared_layers"):
            y = y + 1000
        if "layer_types" in p and "num_kv_shared_layers" not in p:
            y = y + len(p.layer_types)
        return y + p.get_text_config().hidden_size

    x = torch.ones(3)
    out, n = _count_breaks(f, x)
    assert torch.equal(out, f(x))
    assert torch.equal(out, x * 8 + 30.0 + 2 + 8)
    assert n == 0, f"{n} graph breaks / abandoned frames from _Gemma4KVSharedSafeProxy"


def test_proxy_semantics_unchanged():
    cfg = _Cfg()
    p = g4._Gemma4KVSharedSafeProxy(cfg)
    assert not hasattr(p, "num_kv_shared_layers")
    assert p.hidden_size == 8 and p.get_text_config() is p
    assert p == g4._Gemma4KVSharedSafeProxy(cfg) and p == cfg
    assert hash(p) == hash(cfg) and bool(p)
    assert "_Cfg" in repr(p)
    bare = g4._Gemma4KVSharedSafeProxy.__new__(g4._Gemma4KVSharedSafeProxy)
    with pytest.raises(AttributeError):
        bare.hidden_size
    c = copy.copy(p)
    assert c.hidden_size == 8 and not hasattr(c, "num_kv_shared_layers")
    pk = pickle.loads(pickle.dumps(p))
    assert pk.hidden_size == 8 and not hasattr(pk, "num_kv_shared_layers")

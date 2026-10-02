# SPDX-License-Identifier: AGPL-3.0-only
"""Stash scan of a forward whose globals hold an lru_cache wrapper: no graph break, no fullgraph failure.

torch 2.11 graph-breaks on `hasattr(<lru_cache wrapper>, "__func__")` while tracing the scan, and later
versions raise under fullgraph. Unsloth's MoE forwards live in moe_utils, whose globals hold two.
"""
import functools

import pytest
import torch

MU = pytest.importorskip("unsloth_zoo.temporary_patches.moe_utils")


def _experts(reads_stash):
    namespace = {"cached": functools.lru_cache(maxsize = 1)(lambda: None), "take_moe_lora_stash": None}
    body = "    take_moe_lora_stash(self)\n" if reads_stash else ""
    exec("def forward(self, x):\n    cached()\n" + body + "    return x\n", namespace)
    return type("Experts", (torch.nn.Module,), {"forward": namespace["forward"]})()


@pytest.mark.parametrize("reads_stash", [True, False])
@pytest.mark.parametrize("fullgraph", [True, False])
def test_scan_compiles_without_breaking(reads_stash, fullgraph):
    from torch._dynamo.utils import counters

    experts = _experts(reads_stash)
    assert MU._forward_statically_reads_stash(experts) is reads_stash

    def scan(x):
        return x + (1.0 if MU._forward_statically_reads_stash(experts) else 0.0)

    torch._dynamo.reset()
    counters.clear()
    out = torch.compile(scan, fullgraph = fullgraph)(torch.zeros(3))
    assert torch.equal(out, torch.full((3,), float(reads_stash)))
    assert sum(counters["graph_break"].values()) == 0

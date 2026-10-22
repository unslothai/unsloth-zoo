# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
"""The tiled MLP dropout predicate must trace under fullgraph=True after the module tree changes.

Eager calls cache the dropout list keyed on a process-wide tree version and track modules in a
WeakValueDictionary. Refreshing that cache inside a traced region creates weakrefs, which Dynamo
rejects, so a fullgraph trace failed (and a normal one broke the graph) once any tracked MLP
gained a child. CPU only.
"""
import pytest
import torch

from unsloth_zoo import tiled_mlp


class _MLP(torch.nn.Module):
    def __init__(self, p = 0.0):
        super().__init__()
        self.up_proj = torch.nn.Linear(8, 8, bias = False)
        self.drop = torch.nn.Dropout(p)

    def forward(self, x):
        return self.drop(self.up_proj(x))


@pytest.mark.parametrize("p", [0.0, 0.1])
def test_predicate_traces_fullgraph_after_a_tree_change(p):
    torch._dynamo.reset()
    mlp = _MLP(p).train()
    assert tiled_mlp._mlp_needs_rng_state(mlp) == (p > 0)     # eager: cache filled

    def f(x):
        return x.cos() + (1 if tiled_mlp._mlp_needs_rng_state(mlp) else 0)

    compiled = torch.compile(f, backend = "eager", fullgraph = True)
    x = torch.randn(4, 8)
    torch.testing.assert_close(compiled(x), f(x))
    other = _MLP()
    tiled_mlp._mlp_needs_rng_state(other)                      # tracked
    other.extra = torch.nn.Identity()                          # bumps the tree version
    torch.testing.assert_close(compiled(x), f(x))
    mlp.eval()                                                 # still follows train / eval
    torch.testing.assert_close(compiled(x), x.cos())


def test_eager_cache_still_sees_lora_dropout_added_later():
    mlp = _MLP(0.0).train()
    assert not tiled_mlp._mlp_needs_rng_state(mlp)
    mlp.lora_dropout = torch.nn.Dropout(0.2)
    assert tiled_mlp._mlp_needs_rng_state(mlp)

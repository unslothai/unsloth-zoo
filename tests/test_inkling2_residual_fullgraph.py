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

"""The Inkling 2 block attention residuals stay out of Inductor (wrong input grads compiled on CUDA). The decorator the
compiler emits for them must still let an enclosing fullgraph = True training compile trace through (no graph break,
no Unsupported), with the eager forward and gradients."""

import importlib
import re

import pytest
import torch

from unsloth_zoo import compiler
from unsloth_zoo.temporary_patches import utils as patch_utils

_CLASSES = {
    "inkling2": ("Inkling2BlockAttentionResidual", "Inkling2TextConfig"),
    "inkling2_loop": ("Inkling2LoopBlockAttentionResidual", "Inkling2LoopTextConfig"),
}


def _generated_decorator(monkeypatch, model_type, class_name):
    location, modeling = compiler.resolve_modeling_module(model_type)
    if modeling is None or not hasattr(modeling, class_name):
        pytest.skip(f"transformers without {class_name}")
    import transformers
    monkeypatch.setattr(compiler, "transformers", transformers, raising = False)
    source = compiler.create_standalone_class(class_name, location, dir(modeling), disable = True)
    name = re.search(rf"^@(\w+)\s*\ndef {class_name}_forward\b", source, re.M).group(1)
    for where in ("unsloth_zoo.temporary_patches.utils", "unsloth_zoo.temporary_patches.inkling2"):
        decorator = getattr(importlib.import_module(where), name, None)
        if decorator is not None:
            return modeling, decorator
    raise AssertionError(f"generated decorator {name} not found")


def _run(layer, forward, compiled, lengths = (5, 7)):
    def outer(block_states, partial):
        outs = forward(layer, block_states = block_states, partial_block_states = partial)
        return sum((out * (i + 1)).sum() for i, out in enumerate(outs))

    fn = torch.compile(outer, backend = "aot_eager", fullgraph = True, dynamic = True) if compiled else outer
    results = []
    for length in lengths:
        g = torch.Generator().manual_seed(length)
        block_states = torch.randn(2, length, 3, 32, generator = g, requires_grad = True)
        partial = torch.randn(2, length, 32, generator = g, requires_grad = True)
        layer.zero_grad(set_to_none = True)
        loss = fn(block_states, partial)
        loss.backward()
        query_grad = layer.query.grad.clone() if layer.query.grad is not None else None
        results.append((loss.detach(), block_states.grad, partial.grad, query_grad))
    return results


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
@pytest.mark.parametrize("num_queries", [1, 3])
@pytest.mark.parametrize("model_type", sorted(_CLASSES))
def test_enclosing_fullgraph_compile_traces_the_eager_residual(monkeypatch, model_type, num_queries, attn):
    class_name, config_name = _CLASSES[model_type]
    modeling, decorator = _generated_decorator(monkeypatch, model_type, class_name)
    config = getattr(importlib.import_module(f"transformers.models.{model_type}.configuration_{model_type}"), config_name)(
        hidden_size = 32)
    if attn != "sdpa" and not hasattr(getattr(modeling, class_name)(config), "config"):
        pytest.skip("this transformers calls sdpa directly in the residual")
    config._attn_implementation = attn
    torch.manual_seed(0)
    layer = getattr(modeling, class_name)(config, num_queries = num_queries)
    with torch.no_grad():
        layer.query.normal_()
    stock = getattr(modeling, class_name).forward
    torch._dynamo.reset()
    expected = _run(layer, stock, compiled = False)
    got = _run(layer, decorator(stock), compiled = True)
    for want, have in zip(expected, got):
        for a, b in zip(want, have):
            torch.testing.assert_close(b, a, rtol = 1e-5, atol = 1e-5)


def test_frozen_query_and_decode_compile(monkeypatch):
    class_name, config_name = _CLASSES["inkling2"]
    modeling, decorator = _generated_decorator(monkeypatch, "inkling2", class_name)
    from transformers.models.inkling2.configuration_inkling2 import Inkling2TextConfig
    config = Inkling2TextConfig(hidden_size = 32)
    config._attn_implementation = "eager"
    layer = getattr(modeling, class_name)(config)
    layer.query.requires_grad_(False)  # LoRA training: the residual query stays frozen
    stock = getattr(modeling, class_name).forward
    torch._dynamo.reset()
    expected = _run(layer, stock, compiled = False)
    got = _run(layer, decorator(stock), compiled = True)
    for want, have in zip(expected, got):
        assert have[3] is None
        for a, b in zip(want[:3], have[:3]):
            torch.testing.assert_close(b, a, rtol = 1e-5, atol = 1e-5)
    # A compiled decode step inlines the plain forward, as before.
    previous = patch_utils.UNSLOTH_DECODE_COMPILE[0]
    patch_utils.UNSLOTH_DECODE_COMPILE[0] = True
    try:
        torch._dynamo.reset()
        decorated = decorator(stock)
        fn = torch.compile(lambda b, p: decorated(layer, b, p)[0], backend = "eager", fullgraph = True)
        with torch.no_grad():
            b, p = torch.randn(1, 1, 3, 32), torch.randn(1, 1, 32)
            torch.testing.assert_close(fn(b, p), stock(layer, b, p)[0])
    finally:
        patch_utils.UNSLOTH_DECODE_COMPILE[0] = previous

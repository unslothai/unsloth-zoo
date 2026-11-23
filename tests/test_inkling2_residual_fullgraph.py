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


def _autocast(precision):
    if precision == "fp32":
        return torch.autocast("cpu", enabled = False)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    try:
        with torch.autocast("cpu", dtype = dtype):
            torch.ones(2, 2) @ torch.ones(2, 2)
    except Exception:
        pytest.skip(f"CPU autocast without {precision}")
    return torch.autocast("cpu", dtype = dtype)


def _run_typed(forward, layer, precision, compiled):
    def outer(block_states, partial):
        with _autocast(precision):
            outs = forward(layer, block_states, partial)
        return outs, sum((out.float() * (i + 1)).sum() for i, out in enumerate(outs))

    fn = torch.compile(outer, backend = "aot_eager", fullgraph = True, dynamic = True) if compiled else outer
    g = torch.Generator().manual_seed(3)
    block_states = torch.randn(2, 5, 3, 32, generator = g, requires_grad = True)
    partial = torch.randn(2, 5, 32, generator = g, requires_grad = True)
    layer.zero_grad(set_to_none = True)
    outs, loss = fn(block_states, partial)
    loss.backward()
    return [o.detach() for o in outs], block_states.grad, partial.grad, layer.query.grad


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
@pytest.mark.parametrize("attn", ["eager", "sdpa"])
@pytest.mark.parametrize("num_queries", [1, 3])
def test_residual_op_follows_the_caller_autocast(monkeypatch, num_queries, attn, precision):
    """Under autocast the op returns what the eager module returns (dtype, values, grads), eagerly and inside a
    fullgraph aot_eager compile, with the backward recompute under the forward's autocast."""
    from unsloth_zoo.temporary_patches import inkling2 as patch
    class_name, config_name = _CLASSES["inkling2"]
    modeling, decorator = _generated_decorator(monkeypatch, "inkling2", class_name)
    from transformers.models.inkling2.configuration_inkling2 import Inkling2TextConfig
    config = Inkling2TextConfig(hidden_size = 32)
    config._attn_implementation = attn
    torch.manual_seed(0)
    layer = getattr(modeling, class_name)(config, num_queries = num_queries)
    with torch.no_grad():
        layer.query.normal_()
    stock = getattr(modeling, class_name).forward

    def op_forward(module, block_states, partial):
        return patch.inkling2_block_residual_op(
            block_states, partial, module.query, module.norm.eps, sdpa = attn == "sdpa").unbind(-2)

    # Half precision: the sdpa module and the matmul reference round differently in a few elements.
    tol = {} if precision == "fp32" else dict(rtol = 2e-2, atol = 5e-2)
    torch._dynamo.reset()
    want = _run_typed(stock, layer, precision, compiled = False)
    for name, forward, compiled in (
        ("decorated compiled", decorator(stock), True),
        ("op eager", op_forward, False),
        ("op compiled", op_forward, True),
    ):
        got = _run_typed(forward, layer, precision, compiled)
        for a, b in zip(want[0], got[0]):
            assert b.dtype == a.dtype, (name, b.dtype, a.dtype)
            torch.testing.assert_close(b, a, **tol, msg = lambda m: f"{name}: {m}")
        for a, b in zip(want[1:], got[1:]):
            assert b.dtype == a.dtype, (name, b.dtype, a.dtype)
            torch.testing.assert_close(b, a, **tol, msg = lambda m: f"{name}: {m}")


@pytest.mark.parametrize("backend", ["default", "math"])
@pytest.mark.parametrize("model_type, hidden, num_queries, fill", [
    ("inkling2", 512, 1, 200.0),
    ("inkling2_loop", 2304, 3, 64.0),
])
def test_fp16_residual_stays_finite_where_sdpa_does(monkeypatch, model_type, hidden, num_queries, fill, backend):
    """A large finite query: hidden * |q| overflows fp16 unless the softmax scale is applied before the product,
    as SDPA does. The op must stay finite and match the stock sdpa module, values and grads, on the default and
    the math SDPA backend (the one a wide head falls back to)."""
    import contextlib
    from torch.nn.attention import SDPBackend, sdpa_kernel
    kernel = sdpa_kernel(SDPBackend.MATH) if backend == "math" else contextlib.nullcontext()
    class_name, config_name = _CLASSES[model_type]
    modeling, decorator = _generated_decorator(monkeypatch, model_type, class_name)
    config = getattr(importlib.import_module(f"transformers.models.{model_type}.configuration_{model_type}"), config_name)(
        hidden_size = hidden)
    config._attn_implementation = "sdpa"
    layer = getattr(modeling, class_name)(config, num_queries = num_queries)
    with torch.no_grad():
        layer.query.fill_(fill)
    stock = getattr(modeling, class_name).forward

    def run(forward, compiled):
        def outer(block_states, partial):
            with _autocast("fp16"):
                outs = forward(layer, block_states, partial)
            return outs, sum((out.float() * (i + 1)).sum() for i, out in enumerate(outs))

        fn = torch.compile(outer, backend = "aot_eager", fullgraph = True, dynamic = True) if compiled else outer
        g = torch.Generator().manual_seed(5)
        block_states = torch.randn(1, 3, 2, hidden, generator = g).abs().requires_grad_()
        partial = torch.randn(1, 3, hidden, generator = g).abs().requires_grad_()
        layer.zero_grad(set_to_none = True)
        outs, loss = fn(block_states, partial)
        loss.backward()
        return [o.detach() for o in outs], block_states.grad, partial.grad, layer.query.grad

    torch._dynamo.reset()
    with kernel:
        want = run(stock, compiled = False)
        got = run(decorator(stock), compiled = True)
    for a, b in zip([*want[0], *want[1:]], [*got[0], *got[1:]]):
        assert a.isfinite().all()
        assert b.isfinite().all(), f"{b.isnan().sum().item()} NaN of {b.numel()}"
        torch.testing.assert_close(b, a, rtol = 1e-3, atol = 1e-3)


def test_hooked_norm_keeps_the_original_boundary(monkeypatch):
    """The op computes the norm inline: a hook on the norm must make the residual fall back to the stock module."""
    class_name, config_name = _CLASSES["inkling2"]
    modeling, decorator = _generated_decorator(monkeypatch, "inkling2", class_name)
    from transformers.models.inkling2.configuration_inkling2 import Inkling2TextConfig
    config = Inkling2TextConfig(hidden_size = 32)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    layer = getattr(modeling, class_name)(config)
    with torch.no_grad():
        layer.query.normal_()
    stock = getattr(modeling, class_name).forward
    decorated = decorator(stock)
    handle = layer.norm.register_forward_hook(lambda module, args, out: out * 0)
    try:
        torch._dynamo.reset()
        fn = torch.compile(lambda b, p: decorated(layer, b, p)[0], backend = "aot_eager")
        b, p = torch.randn(1, 2, 3, 32), torch.randn(1, 2, 32)
        want = stock(layer, b, p)[0]
        assert want.abs().max() == 0  # the hook ran: zero states, uniform attention, zero output
        torch.testing.assert_close(fn(b, p), want)
    finally:
        handle.remove()

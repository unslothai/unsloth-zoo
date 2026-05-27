# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import math
import sys

import pytest


mx = pytest.importorskip("mlx.core")
if "mlx_simulation" in str(getattr(mx, "__file__", "")):
    pytest.skip("requires real MLX runtime", allow_module_level=True)


def _stable_norm(values):
    max_abs = mx.array(0.0, dtype=mx.float32)
    for value in values:
        value32 = value.astype(mx.float32)
        max_abs = mx.maximum(max_abs, mx.max(mx.abs(value32)))

    denom = mx.maximum(max_abs, mx.array(1e-30, dtype=mx.float32))
    norm_sq = mx.array(0.0, dtype=mx.float32)
    for value in values:
        scaled = value.astype(mx.float32) / denom
        norm_sq = norm_sq + mx.sum(scaled * scaled)
    return denom * mx.sqrt(norm_sq)


def _skip_torch_shim():
    if any(name.startswith("mlx_simulation") for name in sys.modules):
        pytest.skip("requires real MLX runtime")


def test_runtime_cce_zero_tokens_with_non_empty_targets_raises():
    # hidden=0 with non-empty targets must raise, not silently drop labels.
    _skip_torch_shim()
    from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

    runtime_cce, _ = make_chunked_cross_entropy_loss(
        ignore_index=-100,
        chunk_size=16,
    )
    hidden = mx.zeros((0, 16), dtype=mx.float32)
    weight = mx.zeros((32, 16), dtype=mx.float32)
    targets = mx.array([0, 1, 2], dtype=mx.int32)

    with pytest.raises(ValueError, match="hidden has 0 tokens"):
        runtime_cce(hidden, weight, targets)


def test_runtime_cce_int64_wrap_to_ignore_index_poisons_gradients():
    # Wide labels (e.g. 2**32-100) narrow to -100 in int32; backward must
    # propagate NaN from the poisoned lse instead of zeroing the gradient.
    _skip_torch_shim()
    from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

    runtime_cce, _ = make_chunked_cross_entropy_loss(
        ignore_index=-100,
        chunk_size=16,
    )
    hidden = mx.ones((3, 16), dtype=mx.float32)
    weight = mx.ones((32, 16), dtype=mx.float32)
    targets = mx.array([0, 2**32 - 100, -100], dtype=mx.int64)

    def loss_fn(h, w):
        return runtime_cce(h, w, targets).astype(mx.float32).sum()

    loss, grads = mx.value_and_grad(loss_fn, argnums=(0, 1))(hidden, weight)
    grad_hidden, _ = grads
    rows = mx.sum(mx.abs(grad_hidden).astype(mx.float32), axis=1)
    mx.eval(loss, rows)

    assert math.isnan(loss.item())
    assert math.isfinite(rows[0].item()), "valid row must have finite grad"
    assert math.isnan(rows[1].item()), "wrap-to-ignore_index row must NaN grad"
    assert rows[2].item() == pytest.approx(0.0), "real ignore_index row zero"


@pytest.mark.parametrize(
    "bad_target",
    [2**32, -(2**32), 2**32 + 5, 2**32 - 100],
)
def test_runtime_cce_int64_invalid_labels_do_not_wrap_to_valid(bad_target):
    # int64 labels outside int32 range must NaN, not wrap to valid ids.
    _skip_torch_shim()
    from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

    runtime_cce, _ = make_chunked_cross_entropy_loss(
        ignore_index=-100,
        chunk_size=16,
    )
    hidden = mx.ones((1, 16), dtype=mx.float32)
    weight = mx.ones((32, 16), dtype=mx.float32)
    targets = mx.array([bad_target], dtype=mx.int64)

    losses = runtime_cce(hidden, weight, targets)
    mx.eval(losses)
    assert math.isnan(losses[0].item())


def test_runtime_cce_rejects_non_flat_targets():
    # Rank-2 / scalar targets must raise ValueError, not crash kernels.
    _skip_torch_shim()
    from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

    runtime_cce, _ = make_chunked_cross_entropy_loss(
        ignore_index=-100,
        chunk_size=16,
    )
    hidden = mx.zeros((4, 16), dtype=mx.float32)
    weight = mx.zeros((32, 16), dtype=mx.float32)
    targets_2d = mx.zeros((4, 1), dtype=mx.int32)
    targets_scalar = mx.array(0, dtype=mx.int32)

    with pytest.raises(ValueError, match="flat 1D vector"):
        runtime_cce(hidden, weight, targets_2d)
    with pytest.raises(ValueError, match="flat 1D vector"):
        runtime_cce(hidden, weight, targets_scalar)


def test_runtime_cce_zero_tokens_returns_empty_losses_and_zero_gradients():
    _skip_torch_shim()
    from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

    runtime_cce, _ = make_chunked_cross_entropy_loss(
        ignore_index=-100,
        chunk_size=16,
    )
    hidden = mx.zeros((0, 16), dtype=mx.float32)
    weight = mx.zeros((32, 16), dtype=mx.float32)
    targets = mx.zeros((0,), dtype=mx.int32)

    losses = runtime_cce(hidden, weight, targets)
    mx.eval(losses)
    assert losses.shape == (0,)

    def loss_fn(h, w):
        return runtime_cce(h, w, targets).astype(mx.float32).sum()

    loss, grads = mx.value_and_grad(loss_fn, argnums=(0, 1))(hidden, weight)
    mx.eval(loss, *grads)

    assert loss.item() == pytest.approx(0.0)
    assert grads[0].shape == hidden.shape
    assert grads[1].shape == weight.shape
    assert mx.sum(mx.abs(grads[0]).astype(mx.float32)).item() == pytest.approx(0.0)
    assert mx.sum(mx.abs(grads[1]).astype(mx.float32)).item() == pytest.approx(0.0)


def test_quantized_runtime_cce_zero_tokens_returns_empty_losses_and_zero_gradients():
    _skip_torch_shim()
    import mlx.nn as nn

    from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

    linear = nn.Linear(32, 32, bias=False)
    qlinear = nn.QuantizedLinear.from_linear(linear, group_size=32, bits=4)
    runtime_cce, _ = make_chunked_cross_entropy_loss(
        ignore_index=-100,
        chunk_size=16,
        quantized=True,
        group_size=qlinear.group_size,
        bits=qlinear.bits,
    )
    hidden = mx.zeros((0, 32), dtype=mx.float32)
    targets = mx.zeros((0,), dtype=mx.int32)

    losses = runtime_cce(
        hidden,
        qlinear.weight,
        qlinear.scales,
        qlinear.biases,
        targets,
    )
    mx.eval(losses)
    assert losses.shape == (0,)

    def loss_fn(h):
        return runtime_cce(
            h,
            qlinear.weight,
            qlinear.scales,
            qlinear.biases,
            targets,
        ).astype(mx.float32).sum()

    loss, grad = mx.value_and_grad(loss_fn)(hidden)
    mx.eval(loss, grad)

    assert loss.item() == pytest.approx(0.0)
    assert grad.shape == hidden.shape
    assert mx.sum(mx.abs(grad).astype(mx.float32)).item() == pytest.approx(0.0)


@pytest.mark.parametrize("bad_target", [-1, 32])
def test_runtime_cce_invalid_labels_poison_loss_and_gradients(bad_target):
    _skip_torch_shim()
    from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

    runtime_cce, _ = make_chunked_cross_entropy_loss(
        ignore_index=-100,
        chunk_size=16,
    )
    hidden = mx.ones((3, 16), dtype=mx.float32)
    weight = mx.ones((32, 16), dtype=mx.float32)
    targets = mx.array([0, bad_target, -100], dtype=mx.int32)

    losses = runtime_cce(hidden, weight, targets)
    mx.eval(losses)

    assert losses[0].item() == pytest.approx(math.log(32.0), rel=1e-5)
    assert math.isnan(losses[1].item())
    assert losses[2].item() == pytest.approx(0.0)

    def loss_fn(h, w):
        return runtime_cce(h, w, targets).astype(mx.float32).sum()

    loss, grads = mx.value_and_grad(loss_fn, argnums=(0, 1))(hidden, weight)
    grad_hidden, grad_weight = grads
    grad_norm = _stable_norm(grads)
    mx.eval(loss, grad_norm, grad_hidden)

    assert math.isnan(loss.item())
    assert math.isnan(grad_norm.item())
    # Per-row check: only the invalid row's grad_hidden goes NaN. Otherwise
    # a leak into valid rows would be masked by grad_weight's NaN in grad_norm.
    grad_hidden_rows = mx.sum(mx.abs(grad_hidden).astype(mx.float32), axis=1)
    mx.eval(grad_hidden_rows)
    assert math.isfinite(grad_hidden_rows[0].item()), "valid row must have finite grad_hidden"
    assert math.isnan(grad_hidden_rows[1].item()), "invalid row must have NaN grad_hidden"
    assert grad_hidden_rows[2].item() == pytest.approx(0.0), "ignore_index row must zero-grad"


@pytest.mark.parametrize("bad_target", [-1, 32])
def test_compiled_runtime_cce_invalid_labels_poison_loss(bad_target):
    # Both negative and >= vocab_size labels under mx.compile.
    _skip_torch_shim()
    from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

    runtime_cce, _ = make_chunked_cross_entropy_loss(
        ignore_index=-100,
        chunk_size=16,
    )
    hidden = mx.ones((2, 16), dtype=mx.float32)
    weight = mx.ones((32, 16), dtype=mx.float32)
    targets = mx.array([0, bad_target], dtype=mx.int32)

    def losses_fn(h, w, t):
        return runtime_cce(h, w, t)

    losses = mx.compile(losses_fn)(hidden, weight, targets)
    mx.eval(losses)

    assert losses[0].item() == pytest.approx(math.log(32.0), rel=1e-5)
    assert math.isnan(losses[1].item())


@pytest.mark.parametrize("bad_target", [-1, 32])
def test_compiled_runtime_cce_invalid_labels_poison_gradients(bad_target):
    # NaN must survive aux-lse storage into the VJP under compile.
    _skip_torch_shim()
    from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

    runtime_cce, _ = make_chunked_cross_entropy_loss(
        ignore_index=-100,
        chunk_size=16,
    )
    hidden = mx.ones((3, 16), dtype=mx.float32)
    weight = mx.ones((32, 16), dtype=mx.float32)
    targets = mx.array([0, bad_target, -100], dtype=mx.int32)

    def loss_fn(h, w):
        return runtime_cce(h, w, targets).astype(mx.float32).sum()

    compiled = mx.compile(mx.value_and_grad(loss_fn, argnums=(0, 1)))
    loss, grads = compiled(hidden, weight)
    grad_hidden, _ = grads
    mx.eval(loss, grad_hidden)

    assert math.isnan(loss.item())
    rows = mx.sum(mx.abs(grad_hidden).astype(mx.float32), axis=1)
    mx.eval(rows)
    assert math.isfinite(rows[0].item()), "compiled: valid row must have finite grad_hidden"
    assert math.isnan(rows[1].item()), "compiled: invalid row must have NaN grad_hidden"
    assert rows[2].item() == pytest.approx(0.0), "compiled: ignore_index row must zero-grad"


def test_quantized_runtime_cce_invalid_labels_poison_loss():
    _skip_torch_shim()
    import mlx.nn as nn

    from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

    linear = nn.Linear(32, 32, bias=False)
    linear.weight = mx.ones((32, 32), dtype=mx.float32)
    qlinear = nn.QuantizedLinear.from_linear(linear, group_size=32, bits=4)
    runtime_cce, _ = make_chunked_cross_entropy_loss(
        ignore_index=-100,
        chunk_size=16,
        quantized=True,
        group_size=qlinear.group_size,
        bits=qlinear.bits,
    )
    hidden = mx.ones((2, 32), dtype=mx.float32)
    targets = mx.array([0, 32], dtype=mx.int32)

    losses = runtime_cce(
        hidden,
        qlinear.weight,
        qlinear.scales,
        qlinear.biases,
        targets,
    )
    mx.eval(losses)

    assert losses[0].item() == pytest.approx(math.log(32.0), rel=1e-5)
    assert math.isnan(losses[1].item())


def test_quantized_runtime_cce_invalid_labels_poison_gradients():
    _skip_torch_shim()
    import mlx.nn as nn

    from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

    linear = nn.Linear(32, 32, bias=False)
    linear.weight = mx.ones((32, 32), dtype=mx.float32)
    qlinear = nn.QuantizedLinear.from_linear(linear, group_size=32, bits=4)
    runtime_cce, _ = make_chunked_cross_entropy_loss(
        ignore_index=-100,
        chunk_size=16,
        quantized=True,
        group_size=qlinear.group_size,
        bits=qlinear.bits,
    )
    hidden = mx.ones((2, 32), dtype=mx.float32)
    targets = mx.array([0, 32], dtype=mx.int32)

    def loss_fn(h):
        return runtime_cce(
            h,
            qlinear.weight,
            qlinear.scales,
            qlinear.biases,
            targets,
        ).astype(mx.float32).sum()

    loss, grad = mx.value_and_grad(loss_fn)(hidden)
    mx.eval(loss, grad)

    assert math.isnan(loss.item())
    grad_rows = mx.sum(mx.abs(grad).astype(mx.float32), axis=1)
    mx.eval(grad_rows)
    assert math.isfinite(grad_rows[0].item()), "quantized: valid row must have finite grad"
    assert math.isnan(grad_rows[1].item()), "quantized: invalid row must have NaN grad"


def test_compiled_runtime_cce_preserves_aux_lse_for_gradients():
    _skip_torch_shim()
    from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

    runtime_cce, _ = make_chunked_cross_entropy_loss(
        ignore_index=-100,
        chunk_size=32,
    )
    hidden = (mx.arange(64 * 32, dtype=mx.float32).reshape(64, 32) / 97.0) - 1.0
    weight = (mx.arange(128 * 32, dtype=mx.float32).reshape(128, 32) / 113.0) - 1.0
    targets = (mx.arange(64, dtype=mx.int32) * 7) % 128
    targets = mx.where(mx.arange(64) % 11 == 0, -100, targets)
    ntoks = mx.maximum(
        mx.sum((targets != -100).astype(mx.float32)),
        mx.array(1.0, dtype=mx.float32),
    )

    def loss_and_grad_norm(h, w):
        def loss_fn(hh, ww):
            losses = runtime_cce(hh, ww, targets)
            return losses.astype(mx.float32).sum() / ntoks

        loss, grads = mx.value_and_grad(loss_fn, argnums=(0, 1))(h, w)
        return loss, _stable_norm(grads)

    eager_loss, eager_norm = loss_and_grad_norm(hidden, weight)
    compiled_loss, compiled_norm = mx.compile(loss_and_grad_norm)(hidden, weight)
    mx.eval(eager_loss, eager_norm, compiled_loss, compiled_norm)

    assert compiled_loss.item() == pytest.approx(eager_loss.item(), rel=1e-5)
    assert compiled_norm.item() == pytest.approx(eager_norm.item(), rel=1e-4)


def test_compiled_quantized_runtime_cce_preserves_aux_lse_for_gradients():
    _skip_torch_shim()
    import mlx.nn as nn

    from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

    linear = nn.Linear(32, 128, bias=False)
    linear.weight = (
        mx.arange(128 * 32, dtype=mx.float32).reshape(128, 32) / 113.0
    ) - 1.0
    qlinear = nn.QuantizedLinear.from_linear(linear, group_size=32, bits=4)
    runtime_cce, _ = make_chunked_cross_entropy_loss(
        ignore_index=-100,
        chunk_size=32,
        quantized=True,
        group_size=qlinear.group_size,
        bits=qlinear.bits,
    )
    hidden = (mx.arange(64 * 32, dtype=mx.float32).reshape(64, 32) / 97.0) - 1.0
    targets = (mx.arange(64, dtype=mx.int32) * 7) % 128
    targets = mx.where(mx.arange(64) % 11 == 0, -100, targets)
    ntoks = mx.maximum(
        mx.sum((targets != -100).astype(mx.float32)),
        mx.array(1.0, dtype=mx.float32),
    )

    def loss_and_grad_norm(h):
        def loss_fn(hh):
            losses = runtime_cce(
                hh,
                qlinear.weight,
                qlinear.scales,
                qlinear.biases,
                targets,
            )
            return losses.astype(mx.float32).sum() / ntoks

        loss, grad = mx.value_and_grad(loss_fn)(h)
        return loss, _stable_norm((grad,))

    eager_loss, eager_norm = loss_and_grad_norm(hidden)
    compiled_loss, compiled_norm = mx.compile(loss_and_grad_norm)(hidden)
    mx.eval(eager_loss, eager_norm, compiled_loss, compiled_norm)

    assert compiled_loss.item() == pytest.approx(eager_loss.item(), rel=1e-5)
    assert compiled_norm.item() == pytest.approx(eager_norm.item(), rel=1e-4)

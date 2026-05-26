# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

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

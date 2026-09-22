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


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("finalize", [False, True])
def test_single_simd_forward_preserves_each_row(dtype, finalize):
    _skip_torch_shim()
    if not mx.metal.is_available():
        pytest.skip("requires Metal kernels")
    from unsloth_zoo.mlx.cce import runtime_cce as rt

    mx.random.seed(718)
    rows, width = 267, 2048
    targets = mx.where(mx.arange(rows) % 5 == 0, -100, mx.arange(rows) * 17)
    inputs = [
        mx.random.normal((rows, width)).astype(dtype), targets,
        mx.random.uniform(shape=(rows,)), mx.random.uniform(shape=(rows,)),
        mx.random.normal((rows,)), mx.array([1024], mx.int32),
        mx.array([-100], mx.int32), mx.array([0.0], mx.float32),
    ]
    original = (rt._build_forward_update_finalize_kernel() if finalize
                else rt._build_forward_update_kernel())

    def unreachable(**kwargs):
        raise AssertionError("fell back from the single SIMD kernel")

    optimized = rt._with_single_simd_forward(unreachable, finalize)
    kwargs = dict(
        inputs=inputs, output_shapes=[(rows,)] * (5 if finalize else 3),
        output_dtypes=[mx.float32] * (5 if finalize else 3),
        grid=(rows * 256, 1, 1), threadgroup=(256, 1, 1),
    )
    expected, actual = original(**kwargs), optimized(**kwargs)
    mx.eval(expected, actual)
    for left, right in zip(expected, actual):
        assert mx.allclose(left, right, atol=1e-5, rtol=1e-5).item()


@pytest.mark.parametrize("frozen,dim", [(False, 512), (False, 1024), (True, 1024)])
def test_automatic_chunks_reduce_backward_peak(frozen, dim):
    _skip_torch_shim()
    if not mx.metal.is_available():
        # Both halves need Metal: the peak assertion needs Metal memory accounting, and
        # off-Metal the bf16 LSE accumulates at the logits dtype (the float32 cast in
        # _forward_chunked_fused_finalize is gated on label smoothing), so the loss moves
        # with the chunk size and the two plans do not agree.
        pytest.skip("requires Metal memory accounting")
    import gc
    from unsloth_zoo.mlx.cce import _get_runtime_cce

    mx.random.seed(812)
    hidden = mx.random.normal((512, dim)).astype(mx.bfloat16)
    weight = (mx.random.normal((16384, dim)) * 0.05).astype(mx.bfloat16)
    targets = mx.where(mx.arange(512) % 7 == 0, -100, mx.arange(512) * 31)
    mx.eval(hidden, weight, targets)
    functions = []
    for chunk in (2048, 0):
        runtime = _get_runtime_cce(
            ignore_index=-100, logit_softcap=0, chunk_size=chunk, weight_is_frozen=frozen,
        )
        def loss(h, w, runtime=runtime):
            return runtime(h, w, targets).sum() / 512
        functions.append(mx.compile(mx.value_and_grad(loss, argnums=0 if frozen else (0, 1))))
    expected, actual = [fn(hidden, weight) for fn in functions]
    mx.eval(expected, actual)
    assert actual[0].item() == pytest.approx(expected[0].item(), abs=2e-5)
    pairs = [(expected[1], actual[1])] if frozen else zip(expected[1], actual[1])
    for left, right in pairs:
        assert mx.allclose(left, right, atol=2e-5, rtol=0.02).item()
    del expected, actual, left, right, pairs
    peaks = []
    for fn in functions:
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        resident = mx.get_active_memory()
        mx.reset_peak_memory()
        result = fn(hidden, weight)
        mx.eval(result)
        peaks.append(mx.get_peak_memory() - resident)
        del result
    assert peaks[1] <= peaks[0]
    # Only a frozen head is admitted at these shapes. A trainable bfloat16 head writes
    # d_logits in float32 and then casts it back, so the token-side buffers the promotion
    # grows are 4x the logits chunk and exceed what the plan allows; measured on an M1,
    # promoting there cost 146324012 bytes against 143211072.
    if frozen:
        assert peaks[1] < peaks[0]


def test_a_trainable_bfloat16_head_is_not_promoted_to_the_wide_chunk():
    """The promotion rule itself, as integers.

    The peak assertions above can only discriminate on hardware where the promoted
    plan is actually worse, which so far is one runner, and they skip entirely off
    Metal. The rule is a pure integer predicate, so pin it directly: at these shapes
    a frozen head takes the wide chunk and a trainable bfloat16 head must not, on
    every device.
    """
    _skip_torch_shim()
    import unsloth_zoo.mlx.cce.runtime_cce as runtime_cce_module
    from unsloth_zoo.mlx.cce import _get_runtime_cce, clear_cce_cache

    hidden = mx.zeros((512, 512), dtype=mx.bfloat16)
    weight = mx.zeros((16384, 512), dtype=mx.bfloat16)
    saved_budget = runtime_cce_module._CHUNK_BUDGET
    try:
        # Every budget _get_memory_budget can return, from its 4 MB floor (smallest
        # supported device) to its 128 MB cap. The answer must not depend on which.
        for budget_mib in (4, 6, 12, 27, 103, 128):
            runtime_cce_module._CHUNK_BUDGET = budget_mib * 1024 * 1024
            clear_cce_cache()
            plans = {}
            for frozen in (True, False):
                runtime = _get_runtime_cce(
                    ignore_index=-100,
                    logit_softcap=0.0,
                    chunk_size=0,
                    weight_is_frozen=frozen,
                )
                plans[frozen] = runtime._unsloth_get_chunk_plan(hidden, weight)[0]
            assert plans == {True: 4096, False: 2048}, (budget_mib, plans)
    finally:
        runtime_cce_module._CHUNK_BUDGET = saved_budget
        clear_cce_cache()


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


def test_quantized_runtime_cce_cache_omits_weight_gradient_start_arrays():
    import inspect
    import mlx.nn as nn

    from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

    linear = nn.Linear(32, 32, bias=False)
    qlinear = nn.QuantizedLinear.from_linear(linear, group_size=32, bits=4)
    runtime_cce, _ = make_chunked_cross_entropy_loss(
        chunk_size=16,
        quantized=True,
        group_size=qlinear.group_size,
        bits=qlinear.bits,
    )
    losses = runtime_cce(
        mx.ones((2, 32), dtype=mx.float32),
        qlinear.weight,
        qlinear.scales,
        qlinear.biases,
        mx.array([0, 1], dtype=mx.int32),
    )
    mx.eval(losses)

    cache_info = runtime_cce._unsloth_chunk_plan_cache_info
    assert cache_info()["entries"] == 1
    cache = inspect.getclosurevars(cache_info).nonlocals["chunk_plan_cache"]
    assert all(plan[3] == () for plan in cache.values())


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


def test_quantized_runtime_cce_rejects_missing_affine_biases():
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
    ntoks = mx.maximum(
        mx.sum((targets != -100).astype(mx.float32)),
        mx.array(1.0, dtype=mx.float32),
    )

    def loss_fn(hh):
        losses = runtime_cce(hh, qlinear.weight, qlinear.scales, None, targets)
        return losses.astype(mx.float32).sum() / ntoks

    with pytest.raises(ValueError, match="Biases must be provided for affine"):
        mx.eval(loss_fn(hidden))


def test_label_smoothing_matches_closed_form():
    # label_smoothing>0 disables the Metal kernels and takes the python chunk
    # path with the HF LabelSmoother loss/gradient (eps=0 kernel-path no-op is
    # covered by recorded validation).
    _skip_torch_shim()
    from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

    mx.random.seed(3)
    hidden, weight = mx.random.normal((5, 8)), mx.random.normal((16, 8))
    targets = mx.array([1, 7, 15, -100, 4], dtype=mx.int32)
    valid, eps = targets != -100, 0.1
    cce, _ = make_chunked_cross_entropy_loss(
        ignore_index=-100, chunk_size=4, label_smoothing=eps)

    def manual(h, w):
        lg = (h @ w.T).astype(mx.float32)
        safe = mx.where(valid, targets, mx.zeros_like(targets))
        tgt = mx.take_along_axis(lg, mx.expand_dims(safe, -1), -1).squeeze(-1)
        tok = mx.logsumexp(lg, -1) - (1.0 - eps) * tgt - eps * lg.mean(-1)
        return mx.where(valid, tok, mx.zeros_like(tok)).sum()

    lc, gc = mx.value_and_grad(
        lambda h, w: cce(h, w, targets).astype(mx.float32).sum(), argnums=(0, 1))(hidden, weight)
    lm, gm = mx.value_and_grad(manual, argnums=(0, 1))(hidden, weight)
    mx.eval(lc, gc, lm, gm)
    assert float(lc.item()) == pytest.approx(float(lm.item()), rel=1e-5)
    assert max(float(mx.abs(a - b).max().item()) for a, b in zip(gc, gm)) < 2e-5


@pytest.mark.parametrize("quantized, budget_mib", [(True, 12), (False, 88)])
def test_runtime_cce_backward_peak_memory(quantized, budget_mib):
    _skip_torch_shim()
    if not mx.metal.is_available():
        pytest.skip("requires Metal memory accounting")
    from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

    mx.random.seed(73)
    hidden = mx.random.normal((128, 1024)).astype(mx.bfloat16)
    weight = (mx.random.normal((8192, 1024)) * 0.03).astype(mx.bfloat16)
    targets = (mx.arange(128) * 137 % 8192).astype(mx.int32)
    targets = mx.where(mx.arange(128) % 7 == 2, -100, targets)
    side = ()
    if quantized:
        weight, scales, biases = mx.quantize(weight, group_size=64, bits=4)
        side = (scales, biases)
    mx.eval(hidden, weight, targets, *side)
    cce, _ = make_chunked_cross_entropy_loss(
        chunk_size=2048, quantized=quantized, group_size=64 if quantized else None, bits=4 if quantized else None,
    )

    def loss(h, w):
        return cce(h, w, *side, targets).mean()

    run = mx.compile(mx.value_and_grad(loss, argnums=0 if quantized else (0, 1)))
    warmup = run(hidden, weight)
    mx.eval(warmup)
    del warmup
    mx.synchronize()
    mx.clear_cache()
    resident = mx.get_active_memory()
    mx.reset_peak_memory()
    result = run(hidden, weight)
    mx.eval(result)
    peak = mx.get_peak_memory() - resident
    assert mx.isfinite(result[0]).item()
    assert peak < budget_mib * 1024**2, f"CCE backward used {peak / 1024**2:.2f} MiB"


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("softcap", [0.0, 5.0])
def test_frozen_head_gradient_is_built_in_the_forward(monkeypatch, quantized, softcap):
    _skip_torch_shim()
    if not mx.metal.is_available():
        # precompute_hidden_gradient is admitted only when the Metal kernels exist, so
        # off Metal the loss-only path is returned and `calls == [1]` fails rather than
        # skips. The silent fallback is the documented contract, not a defect.
        pytest.skip("requires Metal kernels")
    from unsloth_zoo.mlx.cce import runtime_cce

    mx.random.seed(5)
    hidden = mx.random.normal((67, 64)) * 0.5
    weight = mx.random.normal((4099, 64)) * 0.3
    # Targets in every chunk, ignored rows, and one out-of-range label.
    targets = mx.where(mx.arange(67) % 11 == 7, -100, (mx.arange(67) * 613) % 4099)
    targets = mx.where(mx.arange(67) == 30, 4099, targets)
    cotangent = mx.linspace(-0.7, 0.9, 67)
    if quantized:
        head = mx.quantize(weight, group_size=64, bits=4)
        weight = mx.dequantize(*head, group_size=64, bits=4)
        loss = runtime_cce.make_chunked_cross_entropy_loss(
            quantized=True, group_size=64, bits=4, chunk_size=1024, logit_softcap=softcap,
            precompute_hidden_gradient=True)[0]
    else:
        head = (weight,)
        loss = runtime_cce.make_chunked_cross_entropy_loss(
            weight_is_frozen=True, chunk_size=1024, logit_softcap=softcap, precompute_hidden_gradient=True)[0]
    reference = runtime_cce.make_chunked_cross_entropy_loss(chunk_size=1024, logit_softcap=softcap)[0]

    def run(fn, h, y, g, *args):
        return mx.value_and_grad(lambda x: (fn(x, *args, y) * g).sum())(h)

    calls, forward = [], runtime_cce._forward_with_hidden_gradient
    monkeypatch.setattr(runtime_cce, "_forward_with_hidden_gradient",
                        lambda *args, **kwargs: (calls.append(1), forward(*args, **kwargs))[1])
    actual = run(loss, hidden, targets, cotangent, *head)
    mx.eval(actual)
    assert calls == [1]
    compiled = mx.compile(lambda *args: run(loss, *args))(hidden, targets, cotangent, *head)
    expected = run(reference, hidden, targets, cotangent, mx.stop_gradient(weight))
    mx.eval(compiled, expected)
    assert mx.array_equal(loss(hidden, *head, targets), reference(hidden, weight, targets), equal_nan=True).item()
    for got in (actual, compiled):
        assert mx.all(mx.isnan(got[1][30])).item()
        assert mx.allclose(got[1], expected[1], atol=2e-6, rtol=0, equal_nan=True).item()


@pytest.mark.parametrize("frozen", [False, True])
def test_vlm_cce_passes_a_frozen_dense_head_to_the_runtime_cce(monkeypatch, frozen):
    """The dense VLM loss must tell the runtime whether the head is trainable.

    This lives here rather than beside the other VLM tests because that file runs on
    the torch shim, whose Module.freeze/unfreeze are `return self` no-ops
    (tests/mlx_simulation/mlx_nn_stub.py). Under the shim trainable_parameters() still
    reports lm_head.weight, so the freeze below cannot be read back and the frozen case
    can never be observed. Real MLX runs the whole chain: freeze -> trainable_parameters
    -> _is_lm_head_trainable -> _skip_weight_grad -> weight_is_frozen.
    """
    _skip_torch_shim()
    from unsloth_zoo.mlx import utils as U

    class _Backbone(U.nn.Module):
        pass

    class _LM(U.nn.Module):
        def __call__(self, x):
            return x

    class _VLM(U.nn.Module):
        def __init__(self):
            super().__init__()
            backbone = _Backbone()
            backbone.embed_tokens = U.nn.Embedding(96, 32)
            language_model = _LM()
            language_model.model = backbone
            language_model.lm_head = U.nn.Linear(32, 96, bias=False)
            self.language_model = language_model

        def get_input_embeddings(self):
            return None

    model = _VLM()
    if frozen:
        adapter = U.nn.Linear(32, 32)
        adapter.lora_a = U.mx.zeros((4, 32))
        adapter.lora_b = U.mx.zeros((32, 4))
        model.language_model.model.proj = adapter
        model.freeze()
        adapter.unfreeze(keys=["lora_a", "lora_b"])
    # The derivation is left real. Pinning it here would only re-assert the literal
    # `frozen` and would stop the nested `language_model.lm_head` head-prefix
    # resolution from being covered at all.
    assert U._is_lm_head_trainable(model) is (not frozen)

    factories = []
    monkeypatch.setattr(U, "_get_runtime_cce", lambda **kwargs: factories.append(kwargs))
    U.make_vlm_cce_loss_fn(model)
    assert {kwargs.get("weight_is_frozen") for kwargs in factories} == {frozen}
    # A frozen head also builds the training-mode runtime that precomputes the hidden
    # gradient, which is this PR's actual optimisation. Ordered, not collapsed to a set:
    # a bare set would still pass with the training arm deleted outright.
    assert [kwargs.get("precompute_hidden_gradient") for kwargs in factories] == (
        [None, True] if frozen else [None]
    )

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

"""Quantized-KV attention over a dequantized copy, against the runtime's own path, on real Metal."""

from __future__ import annotations

import sys

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

# "Is mlx importable" is not "is this real mlx", and this file needs the second.
# simulate_mlx_on_torch() installs process-wide and one sibling module calls it
# while being IMPORTED (test_mlx_arrays_cache_advance.py, which sorts just before
# this one), so collection alone is enough to make the importorskip above succeed
# against the shim. These tests then run against a simulation that refuses the
# operation they are built on:
#
#     NotImplementedError: mlx-shim: mx.quantize is not implemented in the
#     simulation. Use real MLX on a Mac to produce quantized weights.
#
# 40 failures, every one of them on a box that simply has no MLX. That is the
# trap test_mlx_real_detection_is_order_independent.py documents; it guards the
# find_spec spellings of it, and importorskip is a third spelling that lies the
# same way. Alone this file skips and passes, which is why it stayed hidden here
# while failing unsloth's Core job, the one place that runs this suite as a
# whole directory rather than a curated list.
from mlx_simulation import mlx_is_simulated  # noqa: E402

if mlx_is_simulated():
    pytest.skip(
        "needs real MLX: these compare a fused Metal kernel against the "
        "runtime's own path, and the shim implements neither",
        allow_module_level = True,
    )

from unsloth_zoo.mlx.attention import (
    _PATCH_FLAG,
    _PATCH_TARGETS,
    dequantized_sdpa,
    dequantizing_is_smaller,
    fused_kernel_exists,
    install_quantized_attention,
    quantized_sdpa_over,
)

GS = 64
# bf16 rounding in a different order meets at ~1.5e-2; a wrong operand, scale or mask at 1e-1.
MAX_DIVERGENCE = 3e-2


def _quant(shape, bits, group_size=GS, dtype=mx.bfloat16, seed=0):
    mx.random.seed(seed)
    return mx.quantize(mx.random.normal(shape).astype(dtype), group_size, bits)


def _cache(B, HKV, S, D, bits, group_size=GS, dtype=mx.bfloat16, Dv=None, value_dtype=None):
    q = (_quant((B, HKV, S, D), bits, group_size, dtype),
         _quant((B, HKV, S, Dv or D), bits, group_size, value_dtype or dtype, seed=100))
    mx.eval(q)
    return q


def _queries(B, HQ, L, D, dtype=mx.bfloat16, seed=1):
    mx.random.seed(seed)
    return mx.random.normal((B, HQ, L, D)).astype(dtype)


def _divergence(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.max(mx.abs(a - b)) / mx.maximum(mx.max(mx.abs(a)), 1e-6)).item()


def _runtime(name):
    return pytest.importorskip(f"{name}.models.base").quantized_scaled_dot_product_attention


def _peak(call):
    mx.synchronize()
    mx.clear_cache()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    out = call()
    mx.eval(out)
    peak = mx.get_peak_memory() - base
    del out
    mx.synchronize()
    mx.clear_cache()
    return peak


# `last`: the last query count on the runtime's path; a row costs its bf16 width plus packing.
@pytest.mark.parametrize("HQ,HKV,D,Dv,bits,query_dtype,last", [
    (32, 8, 128, 128, 4, mx.bfloat16, 82),
    (32, 8, 128, 128, 8, mx.bfloat16, 98),
    (8, 8, 128, 128, 4, mx.bfloat16, 328),
    (16, 2, 64, 64, 4, mx.bfloat16, 20),     # the narrowest head dim with a fused kernel
    (32, 8, 128, 128, 4, mx.float16, 73),    # both sides at float32, with the result
])
def test_the_query_count_against_the_cache_geometry_decides_the_route(HQ, HKV, D, Dv, bits,
                                                                      query_dtype, last):
    cache = _cache(1, HKV, 512, D, bits, Dv=Dv)
    taken = []

    def unfused(queries, q_keys, q_values, scale, mask, group_size=64, bits=8):
        taken.append(queries.shape[-2])
        return mx.zeros((1, HQ, queries.shape[-2], Dv))

    wrapped = quantized_sdpa_over(unfused)
    for L in (1, last, last + 1):
        queries = _queries(1, HQ, L, D, dtype=query_dtype)
        assert (dequantizing_is_smaller(queries, *cache, GS)
                and fused_kernel_exists(queries, *cache, GS)) is (L > last), L
        mx.eval(wrapped(queries, *cache, D ** -0.5, mask="causal", group_size=GS, bits=bits))
    assert taken == [1, last]


@pytest.mark.parametrize("D,Dv,group_size", [
    (256, 256, 64),   # Gemma-2 and Gemma-3: no full kernel at head_dim 256
    (192, 128, 64),   # MLA-shaped: a latent wider than the values
    (160, 160, 32),   # not a supported head dim at all
    (128, 64, 64),    # values narrower than the keys
])
def test_a_geometry_with_no_fused_kernel_never_leaves_the_runtime(D, Dv, group_size):
    """Past 8 queries mlx has a full kernel only at head_dim 64/72/80/96/128 with D == Dv; anywhere
    else it materializes the same scores, so dequantizing on top of them is strictly worse."""
    cache = _cache(1, 8, 512, D, 4, group_size, Dv=Dv)
    taken = []

    def unfused(queries, q_keys, q_values, scale, mask, group_size=64, bits=8):
        taken.append(queries.shape[-2])
        return mx.zeros((1, 32, queries.shape[-2], Dv))

    wrapped = quantized_sdpa_over(unfused)
    for L in (16, 1024, 65536):
        queries = _queries(1, 32, L, D)
        assert fused_kernel_exists(queries, *cache, group_size) is False, L
        mx.eval(wrapped(queries, *cache, D ** -0.5, "causal", group_size=group_size, bits=4))
    assert taken == [16, 1024, 65536]
    # and the guard is what declined it: on bytes alone the deepest call would have dequantized.
    assert dequantizing_is_smaller(_queries(1, 32, 65536, D), *cache, group_size) is True


@pytest.mark.parametrize("bits,group_size,last", [(4, 64, 82), (8, 64, 98), (2, 32, 76)])
@pytest.mark.parametrize("mask", ["causal", "bool", "additive"])
@pytest.mark.parametrize("runtime", ["mlx_lm", "mlx_vlm"])
def test_matches_the_runtime_path_on_both_sides_of_the_threshold(runtime, mask, bits, group_size,
                                                                 last):
    """mlx-vlm runs batched, as its array masks are; those masks also drop every seventh key.

    Both array masks carry a leading 1: `mlx_vlm<0.6.5` -- which is what `mlx-vlm<0.7.0` against
    this repo's `transformers` cap resolves to -- broadcasts a `[B, 1, L, S]` mask against its own
    5-D grouped scores only when B is 1. A per-row batched mask above the tie is covered by
    `test_a_batched_mask_the_pinned_runtime_cannot_broadcast` below.
    """
    unfused = _runtime(runtime)
    B, HQ, HKV, S, D = (2 if runtime == "mlx_vlm" else 1), 32, 8, 640, 128
    cache = _cache(B, HKV, S, D, bits, group_size)
    wrapped = quantized_sdpa_over(unfused)
    for L, exact in ((last, True), (last + 1, False)):
        queries = _queries(B, HQ, L, D)
        if mask == "causal":
            m = "causal"
        else:
            keep = mx.arange(S - L, S)[:, None] >= mx.arange(S)[None]
            keep = keep & (mx.arange(S)[None] % 7 != 3)
            m = (keep[None, None] if mask == "bool"
                 else mx.where(keep, 0.0, -mx.inf).astype(queries.dtype)[None, None])
        call = dict(scale=D ** -0.5, mask=m, group_size=group_size, bits=bits)
        expected = unfused(mx.array(queries), *cache, **call)
        actual = wrapped(mx.array(queries), *cache, **call)
        mx.eval(expected, actual)
        assert actual.shape == expected.shape and actual.dtype == expected.dtype, L
        if exact:
            assert mx.array_equal(actual, expected).item(), L
        else:
            assert _divergence(expected, actual) < MAX_DIVERGENCE, L


@pytest.mark.parametrize("runtime", ["mlx_lm", "mlx_vlm"])
def test_a_batched_mask_the_pinned_runtime_cannot_broadcast(runtime):
    """A per-row `[B, 1, L, S]` mask over grouped heads, which above the tie no longer reaches the
    runtime. `mlx_lm` 0.31.3 and `mlx_vlm` 0.6.4 both raise on it; the fused kernel takes it as
    given, so gate on a float32 reference rather than on their answer."""
    unfused = _runtime(runtime)
    B, HQ, HKV, S, L, D, bits = 2, 32, 8, 640, 256, 128, 4
    cache = _cache(B, HKV, S, D, bits)
    queries = _queries(B, HQ, L, D)
    keep = mx.arange(S - L, S)[:, None] >= mx.arange(S)[None]
    rows = mx.arange(B)[:, None, None, None] == 0
    m = mx.broadcast_to(keep, (B, 1, L, S)) & (rows | (mx.arange(S)[None] % 7 != 3))
    call = dict(scale=D ** -0.5, mask=m, group_size=GS, bits=bits)

    assert dequantizing_is_smaller(queries, *cache, GS) is True
    keys = mx.dequantize(*cache[0], group_size=GS, bits=bits).astype(mx.float32)
    values = mx.dequantize(*cache[1], group_size=GS, bits=bits).astype(mx.float32)
    keys, values = mx.repeat(keys, HQ // HKV, 1), mx.repeat(values, HQ // HKV, 1)
    scores = (queries.astype(mx.float32) @ mx.swapaxes(keys, -1, -2)) * D ** -0.5
    expected = mx.softmax(mx.where(m, scores, mx.finfo(mx.float32).min), axis=-1,
                          precise=True) @ values
    actual = quantized_sdpa_over(unfused)(mx.array(queries), *cache, **call)
    mx.eval(expected, actual)
    assert actual.shape == expected.shape
    assert _divergence(expected, actual) < MAX_DIVERGENCE


def test_the_transient_is_one_dequantized_cache_not_the_scores():
    """On the prefix view a real cache hands over, which is compacted first."""
    from mlx_lm.models.cache import QuantizedKVCache

    B, HQ, HKV, S, L, D, bits = 1, 32, 8, 32768 - 100, 2048, 128, 4
    cache = QuantizedKVCache(group_size=GS, bits=bits)
    mx.random.seed(0)
    q_keys, q_values = cache.update_and_fetch(
        mx.random.normal((B, HKV, S, D)).astype(mx.bfloat16),
        mx.random.normal((B, HKV, S, D)).astype(mx.bfloat16))
    queries = _queries(B, HQ, L, D)
    mx.eval(q_keys, q_values, queries)
    assert q_keys[0].shape[2] == S < cache.keys[0].shape[2]

    peak = _peak(lambda: dequantized_sdpa(queries, q_keys, q_values, D ** -0.5, mask="causal",
                                          group_size=GS, bits=bits))
    dequantized = 2 * B * HKV * S * D * 2
    scores = B * HQ * L * S * 2
    assert dequantized < peak < 2 * dequantized < scores, (peak, dequantized, scores)


@pytest.mark.parametrize("query_dtype,key_dtype,value_dtype", [
    (mx.bfloat16, mx.bfloat16, mx.bfloat16),
    (mx.float16, mx.float16, mx.float16),
    (mx.float16, mx.bfloat16, mx.bfloat16),
    (mx.bfloat16, mx.float32, mx.float32),
    (mx.float16, mx.float16, mx.bfloat16),
])
def test_the_result_dtype_is_the_one_the_runtime_returns(query_dtype, key_dtype, value_dtype):
    unfused = _runtime("mlx_lm")
    B, HQ, HKV, S, L, D, bits = 1, 32, 8, 256, 128, 128, 4
    cache = _cache(B, HKV, S, D, bits, dtype=key_dtype, value_dtype=value_dtype)
    queries = _queries(B, HQ, L, D, dtype=query_dtype)
    call = dict(scale=D ** -0.5, mask="causal", group_size=GS, bits=bits)
    expected = unfused(mx.array(queries), *cache, **call)
    actual = dequantized_sdpa(queries, *cache, **call)
    assert actual.dtype == expected.dtype
    assert _divergence(expected, actual) < MAX_DIVERGENCE


@pytest.mark.parametrize("mask_dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_an_additive_mask_the_fused_kernel_would_reject_stays_on_the_runtime(mask_dtype):
    """`scores += mask` widens the runtime's result, where the fused kernel refuses any mask that
    does not promote to its output. Whichever it is, the caller must get the runtime's answer."""
    unfused = _runtime("mlx_lm")
    B, HQ, HKV, S, L, D, bits = 1, 32, 8, 256, 128, 128, 4
    cache = _cache(B, HKV, S, D, bits)
    queries = _queries(B, HQ, L, D)
    keep = mx.arange(S - L, S)[:, None] >= mx.arange(S)[None]
    m = mx.where(keep, 0.0, -mx.inf).astype(mask_dtype)
    call = dict(scale=D ** -0.5, mask=m, group_size=GS, bits=bits)
    promotes = mask_dtype == mx.bfloat16
    assert fused_kernel_exists(queries, *cache, GS, m) is promotes
    expected = unfused(mx.array(queries), *cache, **call)
    actual = quantized_sdpa_over(unfused)(mx.array(queries), *cache, **call)
    mx.eval(expected, actual)
    assert actual.dtype == expected.dtype, mask_dtype
    assert _divergence(expected, actual) < MAX_DIVERGENCE


def test_an_argument_the_wrapper_predates_goes_to_the_runtime():
    """A future `sinks=` has no fused equivalent, so it must delegate, not be dropped."""
    seen = []

    def unfused(queries, q_keys, q_values, scale, mask=None, group_size=64, bits=8, sinks=None):
        seen.append(sinks)
        return mx.zeros((1, 32, queries.shape[-2], 128), dtype=queries.dtype)

    cache = _cache(1, 8, 512, 128, 4)
    wrapped = quantized_sdpa_over(unfused)
    over_the_tie = _queries(1, 32, 4096, 128)
    assert dequantizing_is_smaller(over_the_tie, *cache, GS) is True
    mx.eval(wrapped(over_the_tie, *cache, 128 ** -0.5, "causal", group_size=GS, bits=4, sinks="S"))
    assert seen == ["S"], "an unknown argument must send the call to the runtime"


def test_installing_the_patch_redirects_both_runtimes():
    pytest.importorskip("mlx_vlm.models.base")
    assert _PATCH_TARGETS == ("mlx_lm.models.base", "mlx_vlm.models.base")
    modules = [sys.modules[path] for path in _PATCH_TARGETS]
    saved = [(m, m.quantized_scaled_dot_product_attention, getattr(m, _PATCH_FLAG, False))
             for m in modules]
    try:
        originals = []
        for module, function, flagged in saved:
            if flagged:
                delattr(module, _PATCH_FLAG)
                function = module.quantized_scaled_dot_product_attention = function.__wrapped__
            originals.append(function)
        assert install_quantized_attention() == _PATCH_TARGETS
        for module, original in zip(modules, originals):
            assert module.quantized_scaled_dot_product_attention.__wrapped__ is original
        assert install_quantized_attention() == ()
        for module, original in zip(modules, originals):
            assert module.quantized_scaled_dot_product_attention.__wrapped__ is original
    finally:
        for module, function, flagged in saved:
            module.quantized_scaled_dot_product_attention = function
            if flagged:
                setattr(module, _PATCH_FLAG, True)
            elif hasattr(module, _PATCH_FLAG):
                delattr(module, _PATCH_FLAG)


def test_every_model_load_path_installs_the_patch():
    """Checked as a shape: every model-returning `return` in `from_pretrained` goes through
    `_finish_load`, since reaching each branch needs repositories this suite lacks."""
    import ast
    import inspect

    from unsloth_zoo.mlx import loader

    tree = ast.parse(inspect.getsource(loader))
    entry = next(n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained")

    def own_returns(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Return) and child.value is not None:
                yield child
            yield from own_returns(child)

    def funnelled(stmt):
        return (isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id == "_finish_load")

    returns = list(own_returns(entry))
    bypassing = [n.lineno for n in returns if not funnelled(n)]
    assert not bypassing, f"value returned without _finish_load at lines {bypassing}"
    assert len(returns) >= 4, f"loader return paths changed shape ({len(returns)} found)"

    helper = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_finish_load")
    shape = [type(n).__name__ for n in helper.body]
    assert shape == ["Expr", "Expr", "Return"], f"_finish_load is no longer straight-line: {shape}"
    assert isinstance(helper.body[0].value, ast.Constant), "expected a docstring first"
    install = helper.body[1].value
    assert (isinstance(install, ast.Call) and isinstance(install.func, ast.Name)
            and install.func.id == "install_quantized_attention"), \
        "_finish_load does not install the patch before returning"

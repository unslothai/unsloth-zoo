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

import functools
import math
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

from unsloth_zoo.mlx import nax
from unsloth_zoo.mlx.attention import (
    _PATCH_FLAG,
    _PATCH_TARGETS,
    _assert_rounding_close,
    _fp32_prefill,
    _probe_cases,
    dequantized_sdpa,
    dequantizing_is_smaller,
    fast_prefill_supported,
    forced_padding_row_bytes,
    fused_kernel_exists,
    install_fused_attention,
    install_quantized_attention,
    invariant_attention_available,
    runtime_fuses,
    invariant_attention_supported,
    draft_attention_supported,
    probe_draft_attention,
    probe_invariant_attention,
    probe_fast_prefill,
    probe_simd_prefill,
    quantized_sdpa_over,
    select_blocks,
    simd_prefill_attention,
    simd_prefill_supported,
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
    (256, 256, 64),   # Gemma-2 and Gemma-3: fused only for long causal or array-masked prefill
    (192, 128, 64),   # MLA-shaped: a latent wider than the values
    (160, 160, 32),   # not a supported head dim at all
    (128, 64, 64),    # values narrower than the keys
])
def test_a_geometry_with_no_fused_kernel_never_leaves_the_runtime(D, Dv, group_size):
    """mlx fuses none of these, so dequantizing on top of its scores is strictly worse."""
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


@pytest.mark.parametrize("case,expected", [
    ({}, True),
    ({"L": 1023}, False),
    ({"S": 1023}, False),   # more queries than keys
    ({"mask": None}, False),
    ({"mask": "bool"}, True),
    ({"mask": "additive"}, True),
    ({"mask": "bool", "L": 1023}, False),
    ({"Dv": 128}, False),
    ({"nax": False}, False),
    ({"nax": False, "L": 2048, "S": 2048}, True),
    ({"nax": False, "L": 2048, "S": 4096}, False),
    ({"nax": False, "L": 2048, "S": 2048, "dtype": mx.float32}, False),
    ({"nax": False, "L": 2048, "S": 2048, "mask": "bool"}, False),
    ({"dtype": mx.float16}, True),
    ({"tf32": "0"}, True),
    ({"dtype": mx.float32}, True),
    ({"dtype": mx.float32, "tf32": "0"}, False),
    ({"dtype": mx.float32, "qdtype": mx.bfloat16, "tf32": "0"}, False),   # the result dtype decides
    ({"dtype": mx.float32, "tf32": None}, True),   # unset means on
    ({"dtype": mx.float32, "tf32": " 0"}, False),
    ({"dtype": mx.float32, "tf32": "abc"}, False),
    ({"dtype": mx.float32, "tf32": "-1"}, True),
    ({"D": 512, "L": 4096, "S": 4096}, True),
    ({"D": 512, "L": 4096, "S": 4096, "nax": False}, False),
    ({"D": 512, "L": 4096, "S": 4096, "dtype": mx.float32, "tf32": "0"}, False),
])
def test_wide_head_prefill_is_fused_where_mlx_fuses_it(monkeypatch, case, expected):
    from unsloth_zoo.mlx import attention, nax

    c = {"L": 1024, "S": 1024, "D": 256, "mask": "causal", "nax": True, "dtype": mx.bfloat16,
         "tf32": "1", **case}
    c.setdefault("Dv", c["D"])
    monkeypatch.setattr(nax, "nax_available", lambda: c["nax"])
    if c["tf32"] is None:
        monkeypatch.delenv("MLX_ENABLE_TF32", raising=False)
    else:
        monkeypatch.setenv("MLX_ENABLE_TF32", c["tf32"])
    cache = _cache(1, 2, c["S"], c["D"], 8, dtype=c["dtype"], Dv=c["Dv"])
    queries = _queries(1, 8, c["L"], c["D"], dtype=c.get("qdtype", c["dtype"]))
    mask = {"bool": mx.ones((c["L"], c["S"]), mx.bool_),
            "additive": mx.zeros((c["L"], c["S"]), c["dtype"])}.get(c["mask"], c["mask"])
    attention._tf32_enabled.cache_clear()
    try:
        assert fused_kernel_exists(queries, *cache, GS, mask) is expected
    finally:
        attention._tf32_enabled.cache_clear()


def test_head_dim_256_prefill_on_nax_is_the_dequantized_kernel(monkeypatch):
    from unsloth_zoo.mlx import nax

    unfused = _runtime("mlx_lm")
    real_nax = nax.nax_available()
    B, HQ, HKV, S, D, bits = 1, 16, 4, 1280, 256, 8
    cache = _cache(B, HKV, S, D, bits)
    wrapped = quantized_sdpa_over(unfused)
    for L, on_nax, fused in ((1023, True, False), (1024, True, True), (1024, False, False)):
        monkeypatch.setattr(nax, "nax_available", lambda: on_nax)
        queries = _queries(B, HQ, L, D)
        call = dict(scale=D ** -0.5, mask="causal", group_size=GS, bits=bits)
        expected = unfused(mx.array(queries), *cache, **call)
        actual = wrapped(mx.array(queries), *cache, **call)
        dequantized = dequantized_sdpa(mx.array(queries), *cache, **call)
        mx.eval(expected, actual, dequantized)
        assert mx.array_equal(actual, dequantized if fused else expected).item(), (L, on_nax)
        assert _divergence(expected, actual) < MAX_DIVERGENCE, (L, on_nax)
        if fused and real_nax:   # mlx really fuses it: no scores are materialized
            peak = _peak(lambda: wrapped(queries, *cache, **call))
            assert peak < B * HQ * L * S * 2, peak


@pytest.mark.parametrize("D,Dv,HQ,L,S,mask,min_keys,expected", [
    (512, 512, 32, 1024, 1024, "causal", None, True),
    (512, 512, 16, 1024, 1024, "causal", None, False),
    (512, 512, 32, 1024, 1024, "bool", None, False),
    (512, 512, 32, 1024, 1023, "causal", None, False),
    (512, 512, 64, 1000, 1024, "causal", None, False),
    (512, 512, 30, 1089, 1089, "causal", None, True),   # query blocks round up
    (512, 512, 16, 1, 1024, None, None, True),
    (512, 512, 16, 1, 1023, None, None, False),
    (512, 512, 8, 1, 1024, None, None, False),
    (512, 512, 16, 2, 1024, None, None, False),
    (512, 512, 16, 1, 1024, "bool", None, False),
    (512, 512, 16, 1, 512, None, "512", True),
    (512, 512, 8, 2, 16, "bool", "0", True),
    (96, 64, 8, 16, 512, None, None, True),
    (96, 64, 8, 1, 512, None, None, True),
    (96, 64, 8, 32, 16, "causal", None, False),
    (96, 64, 8, 8, 4, None, None, False),
])
def test_head_dim_512_and_96_64_are_fused_where_mlx_has_a_kernel(monkeypatch, D, Dv, HQ, L, S, mask,
                                                                min_keys, expected):
    from unsloth_zoo.mlx import nax

    monkeypatch.setattr(nax, "nax_available", lambda: True)
    if min_keys is None:
        monkeypatch.delenv("MLX_SDPA_D512_MIN_KL", raising=False)
    else:
        monkeypatch.setenv("MLX_SDPA_D512_MIN_KL", min_keys)
    mask = mx.ones((L, S), mx.bool_) if mask == "bool" else mask
    cache = _cache(1, 2, S, D, 8, group_size=32, Dv=Dv)
    assert fused_kernel_exists(_queries(1, HQ, L, D), *cache, 32, mask) is expected


@pytest.mark.parametrize("D,Dv,HQ,HKV,L,S,mask", [
    (128, 128, 8, 8, 16, 512, None), (192, 192, 8, 8, 16, 512, None), (64, 64, 8, 8, 1, 512, None),
    (256, 256, 8, 2, 16, 512, "causal"), (256, 256, 8, 2, 1024, 1024, "causal"),
    (256, 256, 8, 2, 1024, 1024, "bool"), (256, 256, 8, 2, 2048, 2048, "causal"),
    (512, 512, 32, 4, 1024, 1024, "causal"), (512, 512, 16, 4, 1024, 1024, "causal"),
    (512, 512, 16, 2, 1, 1024, None), (512, 512, 16, 2, 1, 1023, None),
    (96, 64, 8, 8, 16, 512, None), (96, 64, 8, 8, 1, 512, None),
    (96, 64, 8, 8, 32, 16, "causal"), (128, 128, 8, 8, 32, 16, "causal"),
    (128, 128, 8, 8, 8, 4, None),
])
def test_fused_kernel_exists_matches_what_the_installed_mlx_fuses(monkeypatch, D, Dv, HQ, HKV, L, S,
                                                                  mask):
    """The default call is bit-identical to `force_fused=True` exactly when mlx fuses."""
    monkeypatch.delenv("UNSLOTH_MLX_NAX", raising=False)
    q_keys, q_values = _cache(1, HKV, S, D, 8, group_size=32, Dv=Dv)
    queries = _queries(1, HQ, L, D)
    keys, values = (mx.dequantize(*q, group_size=32, bits=8).astype(queries.dtype)
                    for q in (q_keys, q_values))
    mask = mx.random.uniform(shape=(L, S)) > 0.1 if mask == "bool" else mask
    stock = getattr(mx.fast.scaled_dot_product_attention, "__wrapped__",
                    mx.fast.scaled_dot_product_attention)
    sdpa = functools.partial(stock, queries, keys, values, scale=D ** -0.5, mask=mask)
    try:
        fused = mx.array_equal(sdpa(force_fused=True), sdpa()).item()
    except ValueError:
        fused = False
    assert fused_kernel_exists(queries, q_keys, q_values, 32, mask) is fused


@pytest.mark.parametrize("bits,group_size,last", [(4, 64, 82), (8, 64, 98), (2, 32, 76)])
@pytest.mark.parametrize("mask", ["causal", "bool", "additive"])
@pytest.mark.parametrize("runtime", ["mlx_lm", "mlx_vlm"])
def test_matches_the_runtime_path_on_both_sides_of_the_threshold(runtime, mask, bits, group_size,
                                                                 last):
    """mlx-vlm runs batched, as its array masks are; those masks also drop every seventh key.

    Both array masks carry a leading 1: `mlx_vlm<0.6.5` -- which is what `mlx-vlm<=0.7.11` against
    unsloth's `transformers<=5.5.0` resolves to -- broadcasts a `[B, 1, L, S]` mask against its own
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
    runtime. `mlx_lm` 0.31.3 and `mlx_vlm` 0.6.4 raise on it (mlx_lm 0.32 broadcasts it); the fused kernel takes it as
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


@pytest.fixture
def fused_attention():
    stock = mx.fast.scaled_dot_product_attention
    stock = getattr(stock, "__wrapped__", stock)
    mx.fast.scaled_dot_product_attention = stock
    install_fused_attention()
    yield stock
    mx.fast.scaled_dot_product_attention = stock


def _d256_call(L, S, dtype=mx.bfloat16):
    q = mx.random.normal((1, 8, L, 256), key=mx.random.key(0)).astype(dtype)
    k, v = (mx.random.normal((1, 2, S, 256), key=mx.random.key(i)).astype(dtype) for i in (1, 2))
    return q, k, v


def _peak_of(call):
    mx.synchronize()
    mx.clear_cache()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    mx.eval(call())
    return mx.get_peak_memory() - base


@pytest.mark.parametrize("L,S", [(2048, 2048), (2047, 2558)])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_native_window_dispatch_matches_or_improves_fp32(fused_attention, monkeypatch, L, S, dtype):
    from unsloth_zoo.mlx import attention
    if "window_size" not in (fused_attention.__doc__ or "") or not mx.metal.is_available():
        pytest.skip("requires MLX native window support")
    calls = []
    def spy(*args, **kwargs): calls.append(kwargs); return fused_attention(*args, **kwargs)
    spy.__doc__ = fused_attention.__doc__
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", attention.fused_sdpa_over(spy))
    q, k, v = _d256_call(L, S, dtype)
    rows, cols = (mx.arange(L) + S - L)[:, None], mx.arange(S)[None, :]
    mask = (cols <= rows) & (cols > rows - 512)
    ordinary = fused_attention(q, k, v, scale=1.0, mask=mask)
    got = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=attention.WindowMask(mask, 512))
    assert calls[-1]["mask"] == "causal" and calls[-1]["window_size"] == 512 and calls[-1]["force_fused"] is True
    keys, values = (mx.repeat(x.astype(mx.float32), 4, axis=1) for x in (k, v))
    scores = q.astype(mx.float32) @ keys.swapaxes(-1, -2)
    ref = mx.softmax(mx.where(mask, scores, -mx.inf), axis=-1, precise=True) @ values
    for reduce in (mx.max, mx.mean):
        assert reduce(mx.abs(got.astype(mx.float32) - ref)).item() <= reduce(mx.abs(ordinary.astype(mx.float32) - ref)).item()
    last = fused_attention(q[..., -1:, :], k, v, scale=1.0, mask=mask[..., -1:, :])
    trimmed = mx.fast.scaled_dot_product_attention(q[..., -1:, :], k, v, scale=1.0, mask=attention.WindowMask(mask, 512))
    assert calls[-1]["mask"].shape == (1, S) and mx.array_equal(trimmed.view(mx.uint16), last.view(mx.uint16)).item()
    grad = lambda m: mx.grad(lambda x: mx.fast.scaled_dot_product_attention(x, k, v, scale=1.0, mask=m).astype(mx.float32).sum())(q)
    via_window = grad(attention.WindowMask(mask, 512))
    assert calls[-1]["mask"] is mask and mx.array_equal(via_window.view(mx.uint16), grad(mask).view(mx.uint16)).item()
    from unsloth_zoo.mlx import utils
    monkeypatch.setattr(utils, "mlx_training_patches_active", lambda: True)
    mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=attention.WindowMask(mask, 512))
    assert calls[-1]["mask"] is mask
    monkeypatch.setattr(utils, "mlx_training_patches_active", lambda: False)
    monkeypatch.setenv("UNSLOTH_MLX_NATIVE_WINDOW", "0")
    restored = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=attention.WindowMask(mask, 512))
    assert calls[-1]["mask"] is mask and mx.array_equal(restored.view(mx.uint16), ordinary.view(mx.uint16)).item()


@pytest.mark.parametrize("guard", [None, "vision", "padding", "keep", "offset", "training", "old_runtime", "binding",
                                   "wrapped_builder", "sourceless_builder"])
@pytest.mark.parametrize("compiled", [False, True])
def test_native_window_scope_owns_only_causal_unpadded_masks(fused_attention, monkeypatch, guard, compiled):
    from unsloth_zoo.mlx import attention, inference
    from mlx_vlm.models.gemma4.language import Gemma4TextModel, TextConfig, RotatingKVCache, KVCache
    if "window_size" not in (fused_attention.__doc__ or "") or not mx.metal.is_available():
        pytest.skip("requires MLX native window support")
    stock, scope = Gemma4TextModel._make_masks, {}
    if guard == "wrapped_builder":
        monkeypatch.setattr(Gemma4TextModel, "_make_masks", functools.wraps(stock)(lambda *args: stock(*args)))
    elif guard == "sourceless_builder":
        exec("def masks(*args): return stock(*args)", {"stock": stock}, scope)
        monkeypatch.setattr(Gemma4TextModel, "_make_masks", scope["masks"])
    if compiled:
        from unsloth_zoo.mlx import compile as patches
        monkeypatch.setattr(Gemma4TextModel, "_make_masks", Gemma4TextModel._make_masks)
        monkeypatch.setattr(patches, "_PATCHED_ARCHES", set(patches._PATCHED_ARCHES))
        monkeypatch.setattr(patches, "_PATCH_BINDINGS", set(patches._PATCH_BINDINGS))
        patches._install_gemma4_compile_patches()
        monkeypatch.setattr(mx.fast, "_unsloth_safe_sdpa_mask_patch", False, raising=False)
        patches._install_safe_fused_sdpa_mask_patches()
    model = Gemma4TextModel(TextConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=1, num_key_value_heads=1, vocab_size=32, hidden_size_per_layer_input=0,
        num_kv_shared_layers=1, layer_types=["sliding_attention", "full_attention", "sliding_attention", "sliding_attention"]))
    model.train(guard == "training")
    cache = [RotatingKVCache(512), KVCache(), RotatingKVCache(512), None]
    if guard in ("padding", "keep", "offset"):
        setattr(cache[2], {"padding": "left_padding", "keep": "keep", "offset": "offset"}[guard], 1)
    if guard == "old_runtime":
        monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", lambda *args, **kwargs: fused_attention(*args, **kwargs))
    h, layer_types = mx.zeros((1, 1024, 16)), [type(layer) for layer in model.layers]
    with inference.native_window_attention(model), inference.native_window_attention(model):
        if guard == "binding":
            monkeypatch.setattr(Gemma4TextModel, "_make_masks", lambda *args, **kwargs: [None] * 4)
        masks = model._make_masks(h, cache, mx.zeros((1, 1024), mx.int32) if guard == "vision" else None)
        assert [isinstance(m, attention.WindowMask) for m in masks] == ([True, False, True, True] if guard is None else [False] * 4)
        assert guard != "binding" or masks == [None] * 4
        assert guard is not None or all(m.window == 512 and mx.array_equal(m.original, want).item()
                                        for m, want in zip(masks, stock(model, h, cache)) if isinstance(m, attention.WindowMask))
        assert [type(layer) for layer in model.layers] == layer_types
    assert type(model) is Gemma4TextModel and "_unsloth_window_scopes" not in model.__dict__
    assert [type(layer) for layer in model.layers] == layer_types


def test_safe_sdpa_patch_installed_during_training_survives_its_release(monkeypatch):
    from unsloth_zoo.mlx import compile as patches, utils
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", mx.fast.scaled_dot_product_attention)
    monkeypatch.setattr(mx.fast, "_unsloth_safe_sdpa_mask_patch", False, raising=False)
    utils.acquire_mlx_training_patches()
    try:
        patches._install_safe_fused_sdpa_mask_patches()
        safe = mx.fast.scaled_dot_product_attention
    finally:
        utils.release_mlx_training_patches()
    assert mx.fast.scaled_dot_product_attention is safe
    q = mx.ones((1, 1, 2, 8))
    out = safe(q, q, q, scale=1.0, mask=mx.array([[-mx.inf, -mx.inf], [0.0, 0.0]]))
    assert out[0, 0, 0].tolist() == [0.0] * 8


@pytest.mark.skipif(nax.nax_available(), reason="NAX fuses these chunks by default")
def test_long_head_dim_256_prefill_holds_no_scores_and_stays_close_to_fp32(fused_attention):
    q, k, v = _d256_call(512, 8192)
    mx.eval(q, k, v)
    call = lambda sdpa, *a: sdpa(*a, q, k, v, scale=256 ** -0.5, mask="causal")
    scores = 8 * 512 * 8192 * 2
    assert _peak_of(lambda: call(fused_attention)) >= scores
    assert _peak_of(lambda: call(mx.fast.scaled_dot_product_attention)) < scores / 4
    got = mx.fast.scaled_dot_product_attention(q, k, v, scale=256 ** -0.5, mask="causal")
    want = fused_attention(*(x.astype(mx.float32) for x in (q, k, v)), scale=256 ** -0.5, mask="causal")
    assert mx.abs(got.astype(mx.float32) - want).max().item() <= 0.02 * mx.abs(want).max().item()


@pytest.mark.parametrize("change", ["queries", "keys", "more_queries_than_keys", "float32", "array_mask", "sinks",
                                    "head_dim", "training"])
def test_only_the_measured_call_shape_is_forced(fused_attention, monkeypatch, change):
    from unsloth_zoo.mlx import utils

    L, S, dtype, mask = 16, 8192, mx.bfloat16, "causal"
    extra = {}
    if change == "queries":
        L = 15
    elif change == "keys":
        S = 8191
    elif change == "more_queries_than_keys":
        L = 8193
    elif change == "float32":
        dtype = mx.float32
    elif change == "array_mask":
        mask = mx.ones((L, S), dtype=mx.bool_)
    elif change == "sinks":
        extra["sinks"] = mx.zeros((1, 8, 1, 256), dtype=dtype)
    elif change == "training":
        monkeypatch.setattr(utils, "mlx_training_patches_active", lambda: True)
    q, k, v = _d256_call(L, S, dtype)
    if change == "head_dim":
        q, k, v = (x[..., :128] for x in (q, k, v))
    forced = []

    def spy(*a, **kw):
        forced.append(kw.get("force_fused", False))

    from unsloth_zoo.mlx.attention import fused_sdpa_over
    fused_sdpa_over(spy)(q, k, v, scale=1.0, mask=mask, **extra)
    assert forced == [False]
    q, k, v = _d256_call(16, 8192)
    fused_sdpa_over(spy)(q, k, v, scale=1.0, mask="causal")
    assert forced[-1] is (change != "training")


@pytest.mark.parametrize("head_dim, value_dim, on_nax, widths", [
    (256, 256, False, (256, 256)), (256, 256, True, (256, 256)),
    (192, 128, False, (192, 192)), (192, 128, True, (256, 256)),
    (512, 512, True, None), (512, 512, False, None),
    (192, 192, False, None), (192, 192, True, None),
    (128, 128, False, None), (160, 160, False, None),
])
def test_each_width_is_forced_at_the_width_measured_to_help(fused_attention, monkeypatch, head_dim, value_dim, on_nax,
                                                            widths):
    monkeypatch.setattr(nax, "nax_available", lambda: on_nax)
    q = mx.zeros((1, 8, 16, head_dim), dtype=mx.bfloat16)
    k = mx.zeros((1, 2, 8192, head_dim), dtype=mx.bfloat16)
    v = mx.zeros((1, 2, 8192, value_dim), dtype=mx.bfloat16)
    seen = []

    def spy(q, k, v, **kw):
        seen.append((q.shape[-1], k.shape[-1], v.shape[-1], kw.get("force_fused", False)))
        return mx.zeros((1, 8, 16, v.shape[-1]), dtype=mx.bfloat16)

    from unsloth_zoo.mlx.attention import fused_sdpa_over
    out = fused_sdpa_over(spy)(q, k, v, scale=1.0, mask="causal")
    assert out.shape[-1] == value_dim
    if widths is None:
        assert seen == [(head_dim, head_dim, value_dim, False)]
    else:
        assert seen == [(widths[0], widths[0], widths[1], True)]


def test_the_padded_copies_count_against_dequantizing_and_only_where_a_call_is_forced(monkeypatch):
    monkeypatch.setattr(nax, "nax_available", lambda: True)
    assert forced_padding_row_bytes(192, 128, 2, 2, 8191) == 0
    # Scores outgrow the 4-bit copy plus V padded to 192, but not the copy plus K and V padded to 256.
    q_keys, q_values = _cache(1, 64, 8192, 192, 4, group_size=64, Dv=128)
    queries = _queries(1, 64, 768, 192)
    assert not dequantizing_is_smaller(queries, q_keys, q_values, 64)
    monkeypatch.setattr(nax, "nax_available", lambda: False)
    assert dequantizing_is_smaller(queries, q_keys, q_values, 64)


@pytest.mark.parametrize("on_nax", [False, True])
def test_padded_192_128_prefill_holds_no_scores_and_matches_the_unpadded_attention(fused_attention, monkeypatch,
                                                                                   on_nax):
    monkeypatch.setattr(nax, "nax_available", lambda: on_nax)
    q = mx.random.normal((1, 8, 512, 192), key=mx.random.key(0)).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, 8192, 192), key=mx.random.key(1)).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, 8192, 128), key=mx.random.key(2)).astype(mx.bfloat16)
    mx.eval(q, k, v)
    scores = 8 * 512 * 8192 * 2
    call = lambda sdpa, *a: sdpa(*a, q, k, v, scale=192 ** -0.5, mask="causal")
    assert _peak_of(lambda: call(fused_attention)) >= scores
    # Only the padded copies of K and V remain, and the zoo reports what they cost.
    padding = forced_padding_row_bytes(192, 128, 2, 2, 8192) * 8192
    assert padding == 2 * 8192 * (512 if on_nax else 192) * 2
    assert _peak_of(lambda: call(mx.fast.scaled_dot_product_attention)) < padding + scores / 8
    got = call(mx.fast.scaled_dot_product_attention)
    want = fused_attention(*(x.astype(mx.float32) for x in (q, k, v)), scale=192 ** -0.5, mask="causal")
    assert got.shape == want.shape
    assert mx.abs(got.astype(mx.float32) - want).max().item() <= 0.02 * mx.abs(want).max().item()


@pytest.mark.parametrize("L, expected", [(1023, False), (1024, True)])
def test_mlxs_blocked_head_dim_512_route_counts_as_holding_no_full_scores(L, expected):
    if nax.nax_available():
        pytest.skip("the blocked route is off-NAX")
    q_keys, q_values = _cache(1, 1, 4096, 512, 8, group_size=32)
    queries = _queries(1, 16, L, 512)
    assert runtime_fuses(queries, q_keys, q_values, 32, "causal") is expected
    keys, values = (mx.dequantize(*c, group_size=32, bits=8).astype(queries.dtype) for c in (q_keys, q_values))
    mx.eval(keys, values, queries)
    scores = 16 * L * 4096 * 2
    stock = getattr(mx.fast.scaled_dot_product_attention, "__wrapped__", mx.fast.scaled_dot_product_attention)
    peak = _peak_of(lambda: stock(queries, keys, values, scale=512 ** -0.5, mask="causal"))
    assert (peak < scores / 2) is expected


@pytest.mark.parametrize("opt_out", ["0", "1"])
def test_unsloths_nax_opt_out_leaves_mlxs_own_nax_dispatch_in_place(monkeypatch, opt_out):
    monkeypatch.setattr(nax, "_nax_gpu", lambda: True)
    monkeypatch.setattr(nax, "_stock_nax_kernels", lambda: True)
    monkeypatch.setenv("UNSLOTH_MLX_NAX", opt_out)
    q_keys, q_values = _cache(1, 1, 4096, 512, 8, group_size=32)
    blocked_only = _queries(1, 16, 1024, 512)
    assert not runtime_fuses(blocked_only, q_keys, q_values, 32, "causal")
    assert fused_kernel_exists(_queries(1, 16, 2048, 512), q_keys, q_values, 32, "causal")


def test_a_quantized_head_dim_256_prefill_dequantizes_on_the_forced_route(fused_attention):
    q_keys, q_values = _cache(1, 2, 8192, 256, 8, group_size=64)
    queries = _queries(1, 8, 64, 256)
    assert runtime_fuses(queries, q_keys, q_values, 64, "causal")
    assert not runtime_fuses(queries[:, :, :15], q_keys, q_values, 64, "causal")
    assert not fused_kernel_exists(queries, q_keys, q_values, 64, "causal") or nax.nax_available()
    never = lambda *a, **kw: pytest.fail("the unfused runtime answered a forced call")
    # 512 rows: at 64 the copy still costs more than the scores.
    queries = _queries(1, 8, 512, 256)
    got = quantized_sdpa_over(never)(queries, q_keys, q_values, 256 ** -0.5, "causal", group_size=64, bits=8)
    want = dequantized_sdpa(queries, q_keys, q_values, 256 ** -0.5, "causal", group_size=64, bits=8)
    assert mx.array_equal(got, want).item()


def test_the_wrapper_takes_mlxs_own_keywords(fused_attention):
    q, k, v = _d256_call(1, 4)
    got = mx.fast.scaled_dot_product_attention(q=q, k=k, v=v, scale=1.0, mask="causal", stream=mx.gpu)
    assert mx.array_equal(got, fused_attention(q=q, k=k, v=v, scale=1.0, mask="causal")).item()


def test_installing_the_fused_route_twice_wraps_once(fused_attention):
    once = mx.fast.scaled_dot_product_attention
    install_fused_attention()
    assert mx.fast.scaled_dot_product_attention is once and once.__wrapped__ is fused_attention


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
    assert set(shape[:-1]) == {"Expr"} and shape[-1] == "Return", \
        f"_finish_load is no longer straight-line: {shape}"
    assert isinstance(helper.body[0].value, ast.Constant), "expected a docstring first"
    calls = [stmt.value.func.id for stmt in helper.body[1:-1]
             if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name)]
    assert {"install_quantized_attention", "install_fused_attention"} <= set(calls), \
        "_finish_load does not install the patches before returning"


def _sparse_training_case(kind, seed=13):
    from types import SimpleNamespace

    glm = pytest.importorskip("mlx_vlm.models.glm5_next.language")
    qwen = pytest.importorskip("mlx_vlm.models.qwen4_exp.language")
    qsa = pytest.importorskip("mlx_vlm.models.qwen4_exp.qsa_kernel")
    mx.random.seed(seed)
    q = mx.random.normal((2, 4, 81, 64)).astype(mx.bfloat16)
    k, v = [mx.random.normal((2, 2, 96, 64)).astype(mx.bfloat16) for _ in range(2)]
    if kind == "indexed":
        module = pytest.importorskip("mlx_vlm.models.sparse_attention")
        idx = mx.random.randint(0, 96, (2, 81, 8))
        idx = mx.where(mx.arange(8) == 7, -1, idx)
        idx = mx.where(mx.arange(81)[None, :, None] == 4, -1, idx)
        call = lambda q, k, v: glm._sparse_prefill_attention(q, k, v, idx, 64 ** -.5)
        kernel_name = "_indexed_sparse_attention_kernel"
    else:
        module, kernel_name = qsa, "_qsa_sparse_attention_kernel"
        ends = mx.broadcast_to(mx.arange(10, 91), (2, 81))
        blocks = mx.broadcast_to(mx.array([0, 1]), (2, 81, 2))
        blocks = mx.where((mx.arange(2)[:, None, None] == 1) & (ends[..., None] >= 12),
                          blocks + 1, blocks)
        blocks = mx.where(mx.arange(81)[None, :, None] == 4, -1, blocks)
        ends = mx.where(mx.arange(81)[None, :] == 4, 0, ends)
        selection = SimpleNamespace(
            selected_blocks=blocks, query_ends=ends, complete_counts=ends // 4,
            left_padding=mx.zeros((2,), mx.int32), key_len=96)
        indexer = SimpleNamespace(compress_ratio=4, block_topk=2)
        factory = lambda: qwen.Qwen4ExpQSAIndexer.build_mask(indexer, selection)
        call = lambda q, k, v: qsa.dispatch_qsa_attention(
            q, k, v, blocks, ends, cache=None, scale=64 ** -.5,
            block_size=4, causal=True, mask="causal", mask_factory=factory)
    return (q, k, v), call, module, kernel_name


@pytest.mark.parametrize("kind", ["indexed", "qsa"])
@pytest.mark.parametrize("seed", [13, 41])
def test_sparse_training_matches_fused_forward_and_preserves_inference(kind, seed, monkeypatch):
    import contextvars
    from unsloth_zoo.mlx import utils

    inputs, call, module, kernel_name = _sparse_training_case(kind, seed)
    dispatches = []
    original = getattr(module, kernel_name)

    def observed(*args, **kwargs):
        kernel = original(*args, **kwargs)
        def invoke(*args, **kwargs):
            dispatches.append(True)
            return kernel(*args, **kwargs)
        return invoke

    monkeypatch.setattr(module, kernel_name, observed)
    expected = call(*inputs)
    mx.eval(expected)
    assert len(dispatches) == 1
    utils.acquire_mlx_training_patches()
    try:
        actual = call(*inputs)
        gradients = mx.grad(lambda q, k, v: call(q, k, v).astype(mx.float32).square().sum(),
                            argnums=(0, 1, 2))(*inputs)
        mx.eval(actual, gradients)
        assert len(dispatches) == 1
        assert _divergence(expected, actual) < 5e-3
        for row in range(2):
            assert _divergence(expected[row], actual[row]) < 5e-3
            for gradient in gradients:
                assert mx.all(mx.isfinite(gradient[row])).item()
                assert mx.max(mx.abs(gradient[row])).item() > 0
        assert mx.all(actual[:, :, 4] == 0).item()
        # A concurrent inference context and nested evaluation both keep the kernel.
        other = contextvars.Context().run(call, *inputs)
        mx.eval(other)
        paused = utils.pause_mlx_training_patches()
        try:
            evaluated = call(*inputs)
            mx.eval(evaluated)
        finally:
            utils.resume_mlx_training_patches(paused)
        assert len(dispatches) == 3
        assert mx.array_equal(expected, other).item()
        assert mx.array_equal(expected, evaluated).item()
    finally:
        utils.release_mlx_training_patches()
    after = call(*inputs)
    mx.eval(after)
    assert len(dispatches) == 4
    assert mx.array_equal(expected, after).item()


@pytest.mark.parametrize("key_kind", ["array", "tuple", "list", "take"])
def test_training_index_consumers_detach_only_integer_keys(key_kind):
    import contextvars
    from unsloth_zoo.mlx import utils

    original = mx.array.__getitem__
    x = mx.array([[1.1, 0.2, 2.3], [2.1, 1.2, 0.3]])
    def loss(x):
        idx = x[:, :2].astype(mx.int32)
        values = x.T
        if key_kind == "take":
            return mx.take(values, indices=idx, axis=0).sum()
        key = idx if key_kind == "array" else (idx,)
        if key_kind == "list":
            key = [idx]
        return values[key].sum()

    with pytest.raises(ValueError, match="VJP with respect to indices"):
        mx.eval(mx.grad(loss)(x))
    utils.acquire_mlx_training_patches()
    try:
        with pytest.raises(ValueError, match="VJP with respect to indices"):
            contextvars.Context().run(lambda: mx.eval(mx.grad(loss)(x)))
        gradient = mx.grad(loss)(x)
        mx.eval(gradient)
        assert mx.array_equal(gradient, mx.array([[1., 2., 1.], [1., 2., 1.]])).item()
    finally:
        utils.release_mlx_training_patches()
    assert mx.array.__getitem__ is original


def test_sparse_training_installs_for_imported_aliases_and_missing_modules(monkeypatch):
    import types
    from unsloth_zoo.mlx import attention, utils

    sparse = pytest.importorskip("mlx_vlm.models.sparse_attention")
    original = getattr(sparse.indexed_sparse_attention,
                       "_unsloth_sparse_attention_original", sparse.indexed_sparse_attention)
    consumer = types.ModuleType("mlx_vlm.models.future_consumer")
    consumer.other_name = original
    monkeypatch.setitem(sys.modules, consumer.__name__, consumer)
    utils.acquire_mlx_training_patches()
    try:
        assert consumer.other_name is sparse.indexed_sparse_attention
        assert consumer.other_name(None) is None
    finally:
        utils.release_mlx_training_patches()
    for path in ("mlx_vlm.models.sparse_attention", "mlx_vlm.models.qwen4_exp.qsa_kernel"):
        monkeypatch.delitem(sys.modules, path, raising=False)
    attention.install_sparse_attention_training()


def test_qsa_wrapper_forwards_unknown_kwargs_outside_training():
    from unsloth_zoo.mlx import attention

    def dispatch(queries, keys, values, block_indices, query_ends, *, cache, scale,
                 block_size, causal, mask, mask_factory, allow_sparse_decode=False,
                 future_flag=None):
        return future_flag

    wrapped = attention._qsa_attention_training_over(dispatch)
    assert wrapped(*[None] * 5, cache=None, scale=1., block_size=4, causal=True,
                   mask=None, mask_factory=None, future_flag="kept") == "kept"


def _stock_selection(scores, complete, topk):
    # mlx-vlm 0.7.4 Qwen4ExpQSAIndexer.select_from_projected after the per-head scores, sorted.
    blocks = mx.sum(mx.maximum(scores, 0), axis = 1) / math.sqrt(128)
    blocks = mx.where(mx.arange(scores.shape[-1])[None, None] < complete[..., None], blocks, -mx.inf)
    picks = mx.argpartition(blocks, kth = -topk, axis = -1)[..., -topk:].astype(mx.int32)
    return mx.sort(mx.where(picks < complete[..., None], picks, -1), axis = -1)


@pytest.mark.parametrize("kind", ["normal", "ties", "zeros", "nan", "negzero"])
def test_qsa_block_selection_picks_argpartitions_set(kind):
    mx.random.seed(len(kind))
    for B, L, nb in [(1, 1, 1025), (1, 2, 70000), (2, 3, 4096), (1, 300, 1500), (1, 64, 40000)]:
        scores = mx.random.normal((B, 4, L, nb))
        draw = mx.random.uniform(shape = scores.shape)
        scores = {"ties": mx.round(scores * 2) / 2, "zeros": mx.where(draw < 0.9, -1.0, scores),
                  "nan": mx.where(draw < 0.001, mx.nan, scores), "negzero": mx.where(draw < 0.5, -0.0, scores),
                  }.get(kind, scores)
        # Rows straddle the top-k: some see fewer complete blocks than it, the rest more.
        complete = mx.broadcast_to(nb - L + mx.arange(L, dtype = mx.int32), (B, L)) - (B * L > 3) * (nb - 600)
        assert mx.array_equal(select_blocks(scores, complete, 512, 128), _stock_selection(scores, complete, 512)).item()


def test_qsa_block_selection_refuses_a_zero_budget():
    scores, complete = mx.random.normal((1, 4, 2, 64)), mx.full((1, 2), 64, dtype = mx.int32)
    with pytest.raises(ValueError, match = "topk >= 1"):
        select_blocks(scores, complete, 0, 128)


def test_qsa_fast_prefill_matches_fp32_attention():
    if not nax.nax_available():
        pytest.skip("needs the neural accelerators")
    probe_fast_prefill()


@pytest.mark.parametrize("change", [None, "batch", "heads", "kv_heads", "width", "dtype", "topk", "block", "rows"])
def test_qsa_fast_prefill_covers_only_the_measured_layout(change):
    B, H, KV, D, L, topk, block, dtype = 1, 24, 2, 256, 64, 512, 4, mx.bfloat16
    B, H, KV, D, L, topk, block, dtype = {
        "batch": (2, H, KV, D, L, topk, block, dtype), "heads": (B, 12, KV, D, L, topk, block, dtype),
        "kv_heads": (B, H, 4, D, L, topk, block, dtype), "width": (B, H, KV, 128, L, topk, block, dtype),
        "dtype": (B, H, KV, D, L, topk, block, mx.float16), "topk": (B, H, KV, D, L, 256, block, dtype),
        "block": (B, H, KV, D, L, topk, 8, dtype), "rows": (B, H, KV, D, 63, topk, block, dtype),
    }.get(change, (B, H, KV, D, L, topk, block, dtype))
    queries, keys = mx.zeros((B, H, L, D), dtype), mx.zeros((B, KV, 4096, D), dtype)
    assert fast_prefill_supported(queries, keys, mx.zeros((B, L, topk), mx.int32), block) == (change is None)


def test_qsa_prefill_reference_matches_mlx_vlm():
    from mlx_vlm.models.qwen4_exp.qsa_kernel import qsa_sparse_attention
    for q, k, v, picks, ends in _probe_cases():
        _assert_rounding_close(_fp32_prefill(q, k, v, picks, ends, 256 ** -0.5),
                               qsa_sparse_attention(q, k, v, picks, ends, scale = 256 ** -0.5, block_size = 4),
                               q.shape[2:3] + k.shape[2:3])


def test_qsa_simd_prefill_matches_fp32_attention():
    probe_simd_prefill()


@pytest.mark.parametrize("heads, topk", [(4, 512), (32, 512), (24, 256)])
def test_qsa_simd_prefill_matches_other_query_groups(heads, topk):
    from mlx_vlm.models.qwen4_exp.qsa_kernel import qsa_sparse_attention
    L, N = 64, 2400
    q = mx.random.normal((1, heads, L, 256), key = mx.random.key(1)).astype(mx.bfloat16)
    k, v = (mx.random.normal((1, 2, N, 256), key = mx.random.key(i)).astype(mx.bfloat16) for i in (2, 3))
    ends = mx.arange(N - L + 1, N + 1, dtype = mx.int32)[None]
    picks = mx.argsort(mx.random.uniform(shape = (1, L, 584), key = mx.random.key(4)), axis = -1)[..., :topk]  # complete blocks
    native = qsa_sparse_attention(q, k, v, picks, ends, scale = 0.0625, block_size = 4).astype(mx.float32)
    got = simd_prefill_attention(q, k, v, picks, ends, 0.0625).astype(mx.float32)
    assert mx.abs(got - native).max().item() <= 0.02 * mx.abs(native).max().item()


@pytest.mark.parametrize("change", [None, "group", "topk", "batch", "uneven", "wide_group", "width", "dtype", "block",
                                    "rows"])
def test_qsa_simd_prefill_covers_query_groups_up_to_16(change):
    B, H, KV, D, L, topk, block, dtype = {
        "group": (1, 8, 1, 256, 64, 512, 4, mx.bfloat16), "topk": (1, 24, 2, 256, 64, 256, 4, mx.bfloat16),
        "batch": (2, 24, 2, 256, 64, 512, 4, mx.bfloat16), "uneven": (1, 24, 5, 256, 64, 512, 4, mx.bfloat16),
        "wide_group": (1, 34, 2, 256, 64, 512, 4, mx.bfloat16), "width": (1, 24, 2, 128, 64, 512, 4, mx.bfloat16),
        "dtype": (1, 24, 2, 256, 64, 512, 4, mx.float16), "block": (1, 24, 2, 256, 64, 512, 8, mx.bfloat16),
        "rows": (1, 24, 2, 256, 63, 512, 4, mx.bfloat16),
    }.get(change, (1, 24, 2, 256, 64, 512, 4, mx.bfloat16))
    queries, keys = mx.zeros((B, H, L, D), dtype), mx.zeros((B, KV, 4096, D), dtype)
    covered = change in (None, "group", "topk")
    assert simd_prefill_supported(queries, keys, mx.zeros((B, L, topk), mx.int32), block) == covered


def test_invariant_attention_rows_are_their_one_row_calls_and_near_fp32():
    if not invariant_attention_available():
        pytest.skip("needs an Apple GPU")
    probe_invariant_attention()


def test_draft_attention_follows_stock_over_a_window_read_in_place():
    if not invariant_attention_available():
        pytest.skip("needs an Apple GPU")
    probe_draft_attention()


@pytest.mark.parametrize("change", [None, "batch", "block_heads", "block_dim", "block_values", "block_dtype"])
def test_draft_attention_covers_a_lone_batch_with_matching_block_keys(change):
    q, k = mx.zeros((2 if change == "batch" else 1, 32, 8, 128), mx.bfloat16), mx.zeros((1, 8, 100, 128), mx.bfloat16)
    block = mx.zeros((1, 4 if change == "block_heads" else 8, 12, 64 if change == "block_dim" else 128),
                     mx.float16 if change == "block_dtype" else mx.bfloat16)
    values = block[:, :, :-1] if change == "block_values" else block
    if change == "batch":
        k = mx.zeros((2, 8, 100, 128), mx.bfloat16)
    assert draft_attention_supported(q, k, k, block, values) == (change is None)


@pytest.mark.parametrize("tensor_units, nax_on, apple, expected", [
    (True, True, True, True), (True, False, True, False), (False, False, True, True), (False, False, False, False)])
def test_invariant_attention_runs_on_apple_gpus_and_honors_the_nax_switch_on_tensor_units(
        monkeypatch, tensor_units, nax_on, apple, expected):
    monkeypatch.setattr(nax, "_nax_gpu", lambda: tensor_units)
    monkeypatch.setattr(nax, "nax_available", lambda: nax_on)
    monkeypatch.setattr(nax, "_gpu_generation", lambda: 15 if apple else None)
    assert invariant_attention_available() is expected


@pytest.mark.parametrize("change", [None, "group", "width", "fp16", "dtypes", "rows", "batch", "heads", "values", "dim"])
def test_invariant_attention_covers_kv_cache_layouts(change):
    shapes = {"group": (1, 12, 3), "rows": (1, 24, 4, 4), "batch": (2, 24, 4, 4096)}.get(change, (1, 24, 4))
    B, H, KV, N = (*shapes, 4096)[:4]
    D = {"width": 128, "dim": 96}.get(change, 256)
    dtype = mx.float16 if change == "fp16" else mx.bfloat16
    q = mx.zeros((B, H, 5, D), dtype)
    k = mx.zeros((B if change != "batch" else 1, KV if change != "heads" else 5, N, D), dtype)
    v = k.astype(mx.float16) if change == "dtypes" else k[:, :, :-1] if change == "values" else k
    covered = change in (None, "group", "width", "fp16")
    assert invariant_attention_supported(q, k, v) == covered

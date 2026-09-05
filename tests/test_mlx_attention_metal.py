"""Quantized-KV attention over a dequantized copy, against the runtime's own path, on real Metal."""

from __future__ import annotations

import sys

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from unsloth_zoo.mlx.attention import (
    _PATCH_FLAG,
    _PATCH_TARGETS,
    dequantized_sdpa,
    dequantizing_is_smaller,
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
    (16, 1, 192, 128, 4, mx.bfloat16, 25),   # MLA-shaped: a latent wider than the values
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
        assert dequantizing_is_smaller(queries, *cache, GS) is (L > last), L
        mx.eval(wrapped(queries, *cache, D ** -0.5, mask="causal", group_size=GS, bits=bits))
    assert taken == [1, last]


@pytest.mark.parametrize("bits,group_size,last", [(4, 64, 82), (8, 64, 98), (2, 32, 76)])
@pytest.mark.parametrize("mask", ["causal", "bool", "additive"])
@pytest.mark.parametrize("runtime", ["mlx_lm", "mlx_vlm"])
def test_matches_the_runtime_path_on_both_sides_of_the_threshold(runtime, mask, bits, group_size,
                                                                 last):
    """mlx-vlm runs batched, as its array masks are; those masks also drop every seventh key."""
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
            m = (mx.broadcast_to(keep, (B, 1, L, S)) if mask == "bool"
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

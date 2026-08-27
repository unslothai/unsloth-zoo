# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""What the patch intercepts, and -- more importantly -- what it must not.

`mx.quantized_matmul` serves every quantized projection in MLX, including attention
itself when the KV cache is quantized. Intercepting the wrong call would corrupt
generation silently, so the negative cases here matter more than the positive ones.
"""

import pytest

from _mlx_int8_helpers import (
    make_quantized_linear,
    make_quantized_model,
    reset_int8_state,
)


@pytest.fixture(autouse = True)
def clean_state():
    reset_int8_state()
    yield
    reset_int8_state()


@pytest.fixture
def make_ql():
    return make_quantized_linear


@pytest.fixture
def quantized_model():
    return make_quantized_model()


import mlx.core as mx
import mlx.nn as nn

from unsloth_zoo.mlx import int8_prefill
from unsloth_zoo.mlx.int8_prefill import patch, registry
from unsloth_zoo.mlx.int8_prefill.eligibility import ROW_THRESHOLD

K, N = 1024, 2048


def _enable_with(ql, name="w"):
    int8_prefill.enable(force=True)
    ok, why = registry.register_module(ql, name)
    assert ok, why
    return ql["weight"], ql["scales"], ql["biases"], ql.group_size, ql.bits


def _hits(monkeypatch):
    seen = []
    orig = patch._dispatch
    monkeypatch.setattr(patch, "_dispatch", lambda x, e: seen.append(e.name) or orig(x, e))
    return seen


class TestArgumentBinding:
    """Every calling convention in the wild must bind identically."""

    @pytest.mark.parametrize("form", ["kwargs", "positional", "star_args", "mixed", "defaults"])
    def test_intercepted(self, make_ql, monkeypatch, form):
        ql = make_ql(K, N)
        w, s, b, gs, bits = _enable_with(ql)
        seen = _hits(monkeypatch)
        x = mx.random.normal((ROW_THRESHOLD, K)).astype(mx.bfloat16)

        calls = {
            "kwargs": lambda: mx.quantized_matmul(
                x, w, scales=s, biases=b, transpose=True, group_size=gs, bits=bits),
            "positional": lambda: mx.quantized_matmul(x, w, s, b, True, gs, bits),
            # The literal form at mlx-lm/mlx_lm/models/base.py:84.
            "star_args": lambda: mx.quantized_matmul(
                x, *(w, s, b), transpose=True, group_size=gs, bits=bits),
            "mixed": lambda: mx.quantized_matmul(
                x, w, s, biases=b, transpose=True, group_size=gs, bits=bits),
            "defaults": lambda: mx.quantized_matmul(
                x, w, scales=s, biases=b, group_size=gs, bits=bits),
        }
        out = calls[form]()
        mx.eval(out)
        assert seen == ["w"], f"{form} did not dispatch"


class TestFallthrough:
    """Bit-identical passthrough is the bar: these must reach MLX's op untouched."""

    def _assert_untouched(self, monkeypatch, call):
        seen = _hits(monkeypatch)
        out = call()
        mx.eval(out)
        assert seen == [], f"unexpectedly intercepted: {seen}"
        return out

    def test_rows_below_threshold(self, make_ql, monkeypatch):
        ql = make_ql(K, N)
        w, s, b, gs, bits = _enable_with(ql)
        x = mx.random.normal((ROW_THRESHOLD - 1, K)).astype(mx.bfloat16)
        got = self._assert_untouched(
            monkeypatch, lambda: mx.quantized_matmul(x, w, s, b, True, gs, bits))
        want = patch._ORIG_QMM(x, w, s, b, True, gs, bits, "affine")
        mx.eval(want)
        assert mx.array_equal(got, want).item()

    def test_unregistered_weight(self, make_ql, monkeypatch):
        ql = make_ql(K, N)
        _enable_with(ql)
        other = make_ql(K, N)
        x = mx.random.normal((ROW_THRESHOLD, K)).astype(mx.bfloat16)
        self._assert_untouched(monkeypatch, lambda: other(x))

    def test_transpose_false(self, make_ql, monkeypatch):
        ql = make_ql(K, N)
        w, s, b, gs, bits = _enable_with(ql)
        x = mx.random.normal((ROW_THRESHOLD, N)).astype(mx.bfloat16)
        self._assert_untouched(
            monkeypatch, lambda: mx.quantized_matmul(x, w, s, b, False, gs, bits))

    def test_explicit_stream(self, make_ql, monkeypatch):
        """A metal_kernel launch cannot honour a caller's stream, so never intercept."""
        ql = make_ql(K, N)
        w, s, b, gs, bits = _enable_with(ql)
        x = mx.random.normal((ROW_THRESHOLD, K)).astype(mx.bfloat16)
        stream = mx.default_stream(mx.default_device())
        self._assert_untouched(
            monkeypatch,
            lambda: mx.quantized_matmul(x, w, s, b, True, gs, bits, "affine", stream=stream))

    def test_non_affine_mode(self, make_ql, monkeypatch):
        """mxfp4 carries no biases and uses e8m0 scales; the affine requant is invalid."""
        ql = make_ql(K, N)
        _enable_with(ql)
        try:
            mxfp4 = nn.QuantizedLinear.from_linear(nn.Linear(K, N, bias=False), mode="mxfp4")
        except Exception:
            pytest.skip("mxfp4 unsupported in this MLX build")
        x = mx.random.normal((ROW_THRESHOLD, K)).astype(mx.bfloat16)
        self._assert_untouched(monkeypatch, lambda: mxfp4(x))

    def test_quantized_kv_attention_shape(self, make_ql, monkeypatch):
        """The sdpa path at models/base.py:84 passes 3-D per-step cache tensors."""
        ql = make_ql(K, N)
        _enable_with(ql)
        x = mx.random.normal((2, ROW_THRESHOLD, K)).astype(mx.bfloat16)
        w3 = mx.random.randint(0, 2**31 - 1, (2, N, K // 8)).astype(mx.uint32)
        s3 = mx.random.normal((2, N, K // 64)).astype(mx.bfloat16)
        b3 = mx.random.normal((2, N, K // 64)).astype(mx.bfloat16)
        seen = _hits(monkeypatch)
        try:
            out = mx.quantized_matmul(x, w3, s3, b3, True, 64, 4)
            mx.eval(out)
        except Exception:
            pass  # a raising fallthrough is still a fallthrough
        assert seen == []


class TestCoverage:
    """The call sites a layer-level patch would have missed."""

    def test_quantized_linear(self, make_ql, monkeypatch):
        ql = make_ql(K, N)
        _enable_with(ql, "ql")
        seen = _hits(monkeypatch)
        out = ql(mx.random.normal((ROW_THRESHOLD, K)).astype(mx.bfloat16))
        mx.eval(out)
        assert seen == ["ql"]

    def test_embedding_as_linear(self, monkeypatch):
        """Tied lm_heads route through QuantizedEmbedding, not QuantizedLinear."""
        emb = nn.QuantizedEmbedding.from_embedding(
            nn.Embedding(N, K), group_size=64, bits=4)
        int8_prefill.enable(force=True)
        ok, why = registry.register_module(emb, "emb")
        assert ok, why
        seen = _hits(monkeypatch)
        out = emb.as_linear(mx.random.normal((ROW_THRESHOLD, K)).astype(mx.bfloat16))
        mx.eval(out)
        assert seen == ["emb"]


class TestLifecycle:
    def test_idempotent(self, make_ql):
        assert int8_prefill.enable(force=True) is True
        assert patch.apply() is False, "second apply must not nest the wrapper"

    def test_disable_restores(self, make_ql):
        int8_prefill.enable(force=True)
        assert mx.quantized_matmul is not patch._ORIG_QMM
        int8_prefill.disable()
        assert mx.quantized_matmul is patch._ORIG_QMM
        assert registry.size() == 0

    def test_wraps_roundtrip(self):
        int8_prefill.enable(force=True)
        assert mx.quantized_matmul.__wrapped__ is patch._ORIG_QMM

    def test_registry_identity_recheck(self, make_ql):
        """An id() outliving its array must not select another weight's scales."""
        ql = make_ql(K, N)
        w, s, b, gs, bits = _enable_with(ql)
        entry = registry.get(w)
        assert entry is not None
        entry.w = make_ql(K, N)["weight"]  # simulate id reuse
        assert registry.get(w) is None

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
"""The properties that keep this from breaking anyone who does not want it.

Three of these guard failure modes that would be invisible until they were expensive:
a capability probe that says yes on hardware that cannot run the kernels, an eval on the
hot path that only raises once someone wraps generation in `mx.compile`, and a missing
vjp that only surfaces partway through a fine-tune.
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
import numpy as np

from unsloth_zoo.mlx import int8_prefill
from unsloth_zoo.mlx.int8_prefill import capability, eligibility, patch, registry
from unsloth_zoo.mlx.int8_prefill.eligibility import ROW_THRESHOLD

K, N = 1024, 2048


class TestCapability:
    # These guard the no-op promise, so they must run wherever the module is not
    # supported -- which emphatically includes Apple silicon without M5 int8 tensor ops,
    # the configuration every current Mac user is in. Keying the skip on
    # `mx.metal.is_available()` instead of on the verdict skipped all three on the macOS
    # runner, which is the one place they matter most.

    def test_declines_where_unsupported(self):
        if capability.is_supported():
            pytest.skip("int8 path is supported here; the negative case cannot be shown")
        assert capability.is_supported() is False
        assert capability.reason()  # always explains itself, whatever the layer

    def test_enable_is_a_noop_when_unsupported(self, make_ql):
        if capability.is_supported():
            pytest.skip("int8 path is supported here")
        assert int8_prefill.enable() is False
        assert int8_prefill.is_enabled() is False
        assert mx.quantized_matmul is patch._ORIG_QMM

    def test_model_forward_bit_identical_when_unsupported(self, quantized_model):
        """The whole no-op promise in one assertion."""
        if capability.is_supported():
            pytest.skip("int8 path is supported here")
        x = mx.random.normal((ROW_THRESHOLD, 1024)).astype(mx.bfloat16)
        before = quantized_model(x)
        mx.eval(before)

        int8_prefill.enable()
        int8_prefill.warmup(quantized_model)
        after = quantized_model(x)
        mx.eval(after)

        assert mx.array_equal(before, after).item()

    def test_kill_switch(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_MLX_INT8_PREFILL", "0")
        capability.reset()
        assert capability.is_supported() is False
        assert "disabled by" in capability.reason()

    def test_probe_reference_is_computable_and_exact(self):
        """The capability probe's reference must actually evaluate.

        Regression: the probe originally compared its kernel against
        `mx.matmul(int32, int32)`, which MLX rejects outright -- "Only inexact types are
        supported". That made the probe raise, be caught by its own `except BaseException`,
        and report unsupported. It looked correct on an M1 (which should decline anyway)
        while guaranteeing the module could never enable on an M5 either. Nothing on
        Linux caught it because the probe short-circuits at `sys.platform`.

        So assert the two properties the probe relies on, on any backend: integer matmul
        is unavailable, and the float32 substitute is exact at the probe's size.
        """
        M = N = 128
        K = 256
        xq = ((mx.arange(M * K) % 251) - 125).reshape(M, K).astype(mx.int8)
        wq = ((mx.arange(N * K) % 241) - 120).reshape(N, K).astype(mx.int8)

        with pytest.raises(ValueError, match="inexact"):
            mx.eval(mx.matmul(xq.astype(mx.int32), wq.astype(mx.int32).T))

        got = mx.matmul(xq.astype(mx.float32), wq.astype(mx.float32).T).astype(mx.int32)
        mx.eval(got)
        want = np.matmul(
            np.array(xq, copy=False).astype(np.int64),
            np.array(wq, copy=False).astype(np.int64).T,
        )
        assert np.array_equal(np.array(got, copy=False).astype(np.int64), want)

        # The exactness above holds only while the largest partial sum stays inside
        # float32's exact integer range. Pin the headroom so raising K trips this.
        assert 127 * 127 * K < 2**24

    def test_probe_never_raises(self, monkeypatch):
        """Whatever goes wrong inside, callers see False, not an exception."""
        monkeypatch.setattr(
            capability, "_decide", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        capability.reset()
        assert capability.is_supported() is False
        assert "boom" in capability.reason()


class TestModuleSweep:
    """The rebind sweep must never be the thing that raises.

    Regression: apply() and revert() probed `quantized_matmul` on every entry in
    sys.modules. transformers installs a lazy module __getattr__ that imports a submodule
    for any attribute name it recognises, so a bare getattr for an unrelated name raised
    ModuleNotFoundError from inside somebody else's optional dependency. Observed in CI as
    "No module named 'torchvision'" during teardown, turning a passing test into an error.
    """

    def test_survives_a_module_whose_getattr_explodes(self, make_ql):
        import sys
        import types

        hostile = types.ModuleType("mlx_hostile_probe")

        def _boom(name):
            raise ModuleNotFoundError(f"No module named 'not_installed' (probing {name})")

        hostile.__getattr__ = _boom
        sys.modules["mlx_hostile_probe"] = hostile
        try:
            assert int8_prefill.enable(force = True) is True
            int8_prefill.disable()
            assert mx.quantized_matmul is patch._ORIG_QMM
        finally:
            sys.modules.pop("mlx_hostile_probe", None)

    def test_sweep_skips_unrelated_packages(self):
        """Narrow by name as well as guarding, so unrelated lazy imports are never even
        triggered. Only mlx and unsloth packages could hold the symbol."""
        import sys
        import types

        probed = []
        watcher = types.ModuleType("totally_unrelated_pkg")

        def _record(name):
            probed.append(name)
            raise AttributeError(name)

        watcher.__getattr__ = _record
        sys.modules["totally_unrelated_pkg"] = watcher
        try:
            int8_prefill.enable(force = True)
            int8_prefill.disable()
        finally:
            sys.modules.pop("totally_unrelated_pkg", None)
        assert probed == [], f"sweep touched an unrelated package: {probed}"


class TestEligibility:
    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_4bit_all_group_sizes_accepted(self, group_size):
        ok, why = eligibility.is_eligible(2048, 1024, 4, group_size)
        assert ok, why

    def test_8bit_rejected_by_default(self, monkeypatch):
        monkeypatch.delenv("UNSLOTH_MLX_INT8_ALLOW_8BIT", raising=False)
        ok, why = eligibility.is_eligible(2048, 1024, 8, 64)
        assert not ok
        assert "ALLOW_8BIT" in why

    def test_8bit_accepted_when_opted_in(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_MLX_INT8_ALLOW_8BIT", "1")
        ok, why = eligibility.is_eligible(2048, 1024, 8, 64)
        assert ok, why

    @pytest.mark.parametrize("bits", [2, 3, 5, 6])
    def test_odd_bit_widths_rejected(self, bits):
        """3/5/6 are a dense bitstream whose values straddle uint32 words."""
        ok, _ = eligibility.is_eligible(2048, 1024, bits, 64)
        assert not ok

    def test_non_affine_rejected(self):
        ok, why = eligibility.is_eligible(2048, 1024, 4, 32, mode="mxfp4", has_biases=False)
        assert not ok
        assert "affine" in why

    def test_n_must_tile(self):
        """N is not ceil-divided when building the launch grid, so a partial tile would
        silently drop output columns."""
        ok, why = eligibility.is_eligible(2048 + 64, 1024, 4, 64)
        assert not ok
        assert "multiple of 128" in why

    def test_lm_head_sized_output_rejected(self):
        ok, why = eligibility.is_eligible(151936 // 128 * 128, 4096, 4, 64)
        assert not ok
        assert "lm_head" in why


class TestCompileSafety:
    def test_compiled_forward_runs(self, make_ql):
        """Every mx.eval lives in warmup, so the hot path is trace-safe. An eval inside
        a compile trace raises '[eval] Attempting to eval an array during function
        transformations'."""
        ql = make_ql(K, N)
        int8_prefill.enable(force=True)
        ok, why = registry.register_module(ql, "w")
        assert ok, why

        x = mx.random.normal((ROW_THRESHOLD, K)).astype(mx.bfloat16)
        compiled = mx.compile(lambda t: ql(t))
        out = compiled(x)
        mx.eval(out)
        assert out.shape == (ROW_THRESHOLD, N)

    def test_compiled_matches_eager(self, make_ql):
        ql = make_ql(K, N)
        int8_prefill.enable(force=True)
        registry.register_module(ql, "w")
        x = mx.random.normal((ROW_THRESHOLD, K)).astype(mx.bfloat16)

        eager = ql(x)
        compiled = mx.compile(lambda t: ql(t))(x)
        mx.eval(eager, compiled)
        assert mx.allclose(eager, compiled, atol=1e-2).item()


class TestGradients:
    """`mx.fast.metal_kernel` outputs carry no vjp, so without custom_function a LoRA
    backward through an intercepted prefill dies inside the trainer."""

    def test_gradient_flows(self, make_ql):
        ql = make_ql(K, N)
        int8_prefill.enable(force=True)
        registry.register_module(ql, "w")
        x = mx.random.normal((ROW_THRESHOLD, K)).astype(mx.float32)

        grad = mx.grad(lambda t: ql(t).sum())(x)
        mx.eval(grad)
        assert grad.shape == x.shape
        assert bool(mx.isfinite(grad).all().item())

    def test_gradient_matches_unpatched(self, make_ql):
        """The vjp delegates to the stock 4-bit op, so it should be the exact gradient
        of the unquantized-activation forward, not an approximation of it."""
        ql = make_ql(K, N)
        x = mx.random.normal((ROW_THRESHOLD, K)).astype(mx.float32)

        want = mx.grad(lambda t: ql(t).sum())(x)
        mx.eval(want)

        int8_prefill.enable(force=True)
        registry.register_module(ql, "w")
        got = mx.grad(lambda t: ql(t).sum())(x)
        mx.eval(got)

        assert mx.allclose(got, want, atol=1e-3).item()


class TestSelfTest:
    def test_passes_on_a_registered_weight(self, make_ql):
        ql = make_ql(K, N)
        int8_prefill.enable(force=True)
        registry.register_module(ql, "w")
        ok, detail = int8_prefill.self_test()
        assert ok, detail

    def test_empty_registry_is_not_a_failure(self):
        int8_prefill.enable(force=True)
        ok, detail = int8_prefill.self_test()
        assert ok
        assert "no registered" in detail

    def test_failure_disables_dispatch(self, make_ql, monkeypatch):
        """A wrong kernel must stop dispatching, not emit wrong tokens."""
        ql = make_ql(K, N)
        int8_prefill.enable(force=True)
        registry.register_module(ql, "w")

        from unsloth_zoo.mlx.int8_prefill import backends
        backend = backends.select()
        monkeypatch.setattr(
            backend, "matmul",
            lambda x, e, out_dtype=mx.bfloat16: mx.zeros((x.shape[0], e.n), dtype=out_dtype),
        )
        ok, detail = int8_prefill.self_test()
        assert not ok, detail
        assert patch._disabled_by_selftest is True

        # And from here on, calls fall through untouched.
        x = mx.random.normal((ROW_THRESHOLD, K)).astype(mx.bfloat16)
        got = ql(x)
        want = patch._ORIG_QMM(
            x, ql["weight"], ql["scales"], ql["biases"], True, ql.group_size, ql.bits, "affine")
        mx.eval(got, want)
        assert mx.array_equal(got, want).item()


class TestWarmup:
    def test_registers_eligible_only(self, quantized_model):
        int8_prefill.enable(force=True)
        registered, skipped = int8_prefill.warmup(quantized_model)
        assert registered == 2, int8_prefill.registered()
        assert skipped >= 1  # the 64x128 projection is far below MIN_DIM

    def test_mlp_scope_filters_by_name(self, quantized_model):
        int8_prefill.enable(force=True)
        registered, _ = int8_prefill.warmup(quantized_model, scope="mlp")
        assert all("mlp" in name for name in int8_prefill.registered())
        assert registered == 2

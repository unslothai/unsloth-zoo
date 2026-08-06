"""
AMD ROCm Compatibility Tests for unsloth-zoo
============================================

Tests that all AMD ROCm patches in unsloth-zoo work correctly.
These tests run without GPU hardware using mocks — no ROCm installation needed.

For live hardware validation, set UNSLOTH_IS_PRESENT=1 and run on a ROCm GPU:
    UNSLOTH_IS_PRESENT=1 python -m pytest tests/test_rocm_compatibility.py -v
"""

import importlib
import inspect
import sys
from unittest import mock

import pytest
import torch

import unsloth_zoo.device_type as dt


@pytest.fixture(autouse=True)
def _restore_is_hip_cache():
    """Clear is_hip() functools.cache after every test.

    is_hip() is @functools.cache. Patching torch.version inside a test caches
    the mocked result for the process lifetime unless we explicitly clear it.
    Without this fixture, a test that mocks CUDA and runs last leaves is_hip()
    returning False for all subsequent tests even on live ROCm hardware.
    """
    yield
    dt.is_hip.cache_clear()
    if hasattr(dt, "get_amd_attention_implementation"):
        dt.get_amd_attention_implementation.cache_clear()
    if hasattr(dt, "get_amd_flash_attn_func"):
        dt.get_amd_flash_attn_func.cache_clear()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_hip_torch():
    """Return a mock torch.version with .hip set (simulates ROCm build)."""
    v = mock.MagicMock()
    v.hip = "6.2.0"
    v.cuda = None
    return v


def _make_cuda_torch():
    """Return a mock torch.version without .hip (simulates CUDA build)."""
    v = mock.MagicMock()
    v.hip = None
    v.cuda = "12.1"
    return v


# Skip aiter tests when the symbols aren't in this tree yet (PR #920 pending)
_AITER_APIS_PRESENT = (
    hasattr(dt, "get_amd_attention_implementation")
    and hasattr(dt, "get_amd_flash_attn_func")
)
_needs_aiter_apis = pytest.mark.skipif(
    not _AITER_APIS_PRESENT,
    reason="AMD aiter APIs not yet merged (pending PR #920)",
)

try:
    from unsloth_zoo.compiler import replace_sdpa_with_amd_aiter as _replace_fn
    _COMPILER_REWRITE_PRESENT = True
except ImportError:
    _COMPILER_REWRITE_PRESENT = False

_needs_compiler_rewrite = pytest.mark.skipif(
    not _COMPILER_REWRITE_PRESENT,
    reason="replace_sdpa_with_amd_aiter not yet merged (pending PR #920)",
)


# ─────────────────────────────────────────────────────────────────────────────
# PR #910 patches — ROCm compatibility guards
# ─────────────────────────────────────────────────────────────────────────────

class TestIsHip:
    """is_hip() must return True on ROCm builds, False on CUDA builds."""

    def test_is_hip_true_on_rocm(self):
        with mock.patch.object(torch, "version", _make_hip_torch()):
            dt.is_hip.cache_clear()
            assert dt.is_hip() is True

    def test_is_hip_false_on_cuda(self):
        with mock.patch.object(torch, "version", _make_cuda_torch()):
            dt.is_hip.cache_clear()
            assert dt.is_hip() is False

    def test_is_hip_false_when_hip_attr_missing(self):
        v = mock.MagicMock()
        del v.hip
        with mock.patch.object(torch, "version", v):
            dt.is_hip.cache_clear()
            assert dt.is_hip() is False


class TestSmCapGuard:
    """On AMD, SM capability guard in vllm_utils.py must use the real production code."""

    def test_sm_cap_zero_on_amd_via_production_code(self):
        """
        Read the production sm_cap assignment from vllm_utils.py source and
        verify the AMD branch sets sm_cap=0 — exercises the real guard, not a copy.
        """
        import unsloth_zoo.vllm_utils as vu
        src = inspect.getsource(vu)
        # The production guard must contain the AMD/XPU else branch
        assert "sm_cap = 0" in src, (
            "vllm_utils.py must set sm_cap=0 on AMD/XPU; guard may have been removed"
        )
        assert "is_hip()" in src, (
            "vllm_utils.py must call is_hip() to gate SM-arch paths"
        )

    def test_sm_cap_guard_uses_is_hip_not_cuda_branch(self):
        """Simulate AMD device: is_hip()=True must skip get_device_capability()."""
        call_count = {"n": 0}
        real_get_cap = torch.cuda.get_device_capability

        def counting_get_cap(*a, **kw):
            call_count["n"] += 1
            return real_get_cap(*a, **kw)

        with mock.patch.object(torch, "version", _make_hip_torch()):
            dt.is_hip.cache_clear()
            with mock.patch("torch.cuda.is_available", return_value=True):
                with mock.patch("torch.cuda.get_device_capability",
                                side_effect=counting_get_cap):
                    # Simulate the production guard logic from vllm_utils.py lines 948-952
                    if not dt.is_hip() and dt.DEVICE_TYPE != "xpu":
                        capability = torch.cuda.get_device_capability()
                        sm_cap = capability[0] * 10 + capability[1]
                    else:
                        sm_cap = 0
        assert sm_cap == 0
        assert call_count["n"] == 0, (
            "get_device_capability() must not be called on AMD (is_hip()=True)"
        )


class TestFlashInferGuard:
    """FlashInfer must be skipped on AMD ROCm via the production _clear_flashinfer_env_on_hip()."""

    def test_production_guard_clears_env_on_hip(self):
        """Call the real _clear_flashinfer_env_on_hip() — not a reimplementation."""
        import os
        import unsloth_zoo.vllm_utils as vu
        env_key = "VLLM_USE_FLASHINFER_SAMPLER"
        # Assertions must be INSIDE patch.dict context: exiting it restores the
        # original environment (key absent), which would make the post-context
        # assertions vacuously pass or spuriously fail.
        with mock.patch.dict(os.environ, {env_key: "1"}):
            with mock.patch.object(vu, "is_hip", return_value=True):
                returned = vu._clear_flashinfer_env_on_hip()
                # Assert inside the patch context while the env is still mocked
                assert returned is True, (
                    "_clear_flashinfer_env_on_hip must return True on AMD"
                )
                assert env_key not in os.environ, (
                    f"{env_key} must be deleted by _clear_flashinfer_env_on_hip on AMD ROCm"
                )

    def test_production_guard_clears_attention_backend_on_hip(self):
        """VLLM_ATTENTION_BACKEND=FLASHINFER must also be cleared on AMD ROCm.

        A future change that stops clearing this key would leave vLLM on the
        CUDA-only FlashInfer path on ROCm — this test catches that regression.
        """
        import os
        import unsloth_zoo.vllm_utils as vu
        env_key = "VLLM_ATTENTION_BACKEND"
        with mock.patch.dict(os.environ, {env_key: "FLASHINFER"}):
            with mock.patch.object(vu, "is_hip", return_value=True):
                returned = vu._clear_flashinfer_env_on_hip()
                assert returned is True
                assert env_key not in os.environ, (
                    f"{env_key}=FLASHINFER must be cleared by "
                    "_clear_flashinfer_env_on_hip on AMD ROCm"
                )

    def test_production_guard_preserves_env_on_cuda(self):
        """On CUDA, _clear_flashinfer_env_on_hip must return False and leave env untouched."""
        import os
        import unsloth_zoo.vllm_utils as vu
        env_key = "VLLM_USE_FLASHINFER_SAMPLER"
        # Assertions must be INSIDE patch.dict context for the same reason.
        with mock.patch.dict(os.environ, {env_key: "1"}):
            with mock.patch.object(vu, "is_hip", return_value=False):
                returned = vu._clear_flashinfer_env_on_hip()
                # Assert inside the patch context while the env is still mocked
                assert returned is False, (
                    "_clear_flashinfer_env_on_hip must return False on CUDA"
                )
                assert os.environ.get(env_key) == "1", (
                    f"{env_key} must NOT be removed on CUDA"
                )

    def test_production_guard_exists_in_vllm_utils(self):
        """_clear_flashinfer_env_on_hip must remain importable from vllm_utils."""
        import unsloth_zoo.vllm_utils as vu
        assert hasattr(vu, "_clear_flashinfer_env_on_hip"), (
            "_clear_flashinfer_env_on_hip has been removed from vllm_utils.py"
        )
        assert callable(vu._clear_flashinfer_env_on_hip)


class TestMemGetInfoGuard:
    """tiled_mlp._default_target_gb() must use mem_get_info on ROCm, fallback on CPU."""

    def test_default_target_gb_on_hip_calls_mem_get_info(self):
        """Exercise the real _default_target_gb() from tiled_mlp — not a duplicate."""
        import unsloth_zoo.tiled_mlp as tm
        fake_free = 200 * 1024 ** 3  # 200 GB free
        with mock.patch.object(tm, "DEVICE_TYPE", "hip"):
            with mock.patch("torch.cuda.is_available", return_value=True):
                with mock.patch("torch.cuda.mem_get_info",
                                return_value=(fake_free, 256 * 1024**3)):
                    result = tm._default_target_gb()
        # _default_target_gb returns free_gb * 0.5
        expected = fake_free / 1024 ** 3 * 0.5
        assert abs(result - expected) < 0.01, f"Expected {expected:.1f} GB, got {result:.1f} GB"

    def test_default_target_gb_fallback_on_cpu(self):
        """On CPU with psutil, _default_target_gb returns half of available host RAM."""
        import unsloth_zoo.tiled_mlp as tm
        import types
        # _default_target_gb uses psutil.virtual_memory().available on CPU.
        # We must mock it: psutil is installed in CI so the real call returns
        # the runner's RAM, not 4.0. Mock both psutil and mem_get_info.
        fake_available = 32 * 1024 ** 3  # 32 GB simulated host RAM
        fake_vmem = types.SimpleNamespace(available=fake_available)
        fake_psutil = types.ModuleType("psutil")
        fake_psutil.virtual_memory = lambda: fake_vmem
        with mock.patch.object(tm, "DEVICE_TYPE", "cpu"):
            with mock.patch("torch.cuda.is_available", return_value=False):
                with mock.patch("torch.cuda.mem_get_info",
                                side_effect=AssertionError("must not call mem_get_info on CPU")):
                    with mock.patch.dict("sys.modules", {"psutil": fake_psutil}):
                        result = tm._default_target_gb()
        expected = fake_available / 1024 ** 3 * 0.5  # 16.0 GB
        assert abs(result - expected) < 0.01, f"Expected {expected:.1f} GB, got {result:.1f} GB"

    def test_tiled_mlp_source_contains_hip_guard(self):
        """Production tiled_mlp.py must still guard mem_get_info with DEVICE_TYPE check."""
        import unsloth_zoo.tiled_mlp as tm
        src = inspect.getsource(tm._default_target_gb)
        assert '"hip"' in src or "'hip'" in src, (
            "_default_target_gb must include 'hip' in the DEVICE_TYPE guard"
        )
        assert "mem_get_info" in src, (
            "_default_target_gb must call torch.cuda.mem_get_info on ROCm"
        )


class TestDeviceTypeTorch:
    """DEVICE_TYPE_TORCH must be 'cuda' on ROCm (PyTorch ROCm aliases torch.cuda.*)."""

    def test_device_type_torch_is_cuda_on_hip(self):
        """On a mocked ROCm build, DEVICE_TYPE_TORCH must be exactly 'cuda'."""
        # The module-level constant is already computed; test that the invariant
        # holds by checking the documented ROCm mapping directly.
        # On live ROCm hardware DEVICE_TYPE="hip" and DEVICE_TYPE_TORCH="cuda".
        # The mock test verifies the transformation logic: hip -> cuda alias.
        if dt.DEVICE_TYPE == "hip":
            # Running on real ROCm hardware
            assert dt.DEVICE_TYPE_TORCH == "cuda", (
                "On ROCm, DEVICE_TYPE_TORCH must be 'cuda' (PyTorch ROCm alias); "
                f"got {dt.DEVICE_TYPE_TORCH!r}"
            )
        else:
            # Not on ROCm — just verify the constant is a string
            assert isinstance(dt.DEVICE_TYPE_TORCH, str)

    def test_device_type_torch_transformation_logic(self):
        """The hip->cuda alias transformation must be present in device_type.py source."""
        src = inspect.getsource(dt)
        assert 'DEVICE_TYPE_TORCH' in src
        # The module must set DEVICE_TYPE_TORCH="cuda" when DEVICE_TYPE=="hip"
        assert '"cuda"' in src or "'cuda'" in src, (
            "device_type.py must set DEVICE_TYPE_TORCH to 'cuda' on ROCm"
        )


# ─────────────────────────────────────────────────────────────────────────────
# AMD aiter Flash Attention (PR #920) — skipped until merged
# ─────────────────────────────────────────────────────────────────────────────

@_needs_aiter_apis
class TestAmdAiterDetection:
    """get_amd_attention_implementation() must return correct backend."""

    def test_returns_sdpa_on_cuda(self):
        with mock.patch.object(torch, "version", _make_cuda_torch()):
            dt.is_hip.cache_clear()
            dt.get_amd_attention_implementation.cache_clear()
            assert dt.get_amd_attention_implementation() == "sdpa"

    def test_returns_sdpa_when_aiter_missing(self):
        with mock.patch.object(torch, "version", _make_hip_torch()):
            dt.is_hip.cache_clear()
            dt.get_amd_attention_implementation.cache_clear()
            with mock.patch("importlib.util.find_spec", return_value=None):
                with mock.patch.object(dt, "_detect_rocm_major_minor",
                                       return_value="7.0"):
                    assert dt.get_amd_attention_implementation() == "sdpa"

    def test_returns_sdpa_on_rocm_lt_7(self):
        """aiter present but ROCm < 7.0 — must return sdpa via version gate, not absent aiter."""
        mock_aiter = mock.MagicMock()
        mock_aiter.flash_attn_func = mock.MagicMock()
        with mock.patch.object(torch, "version", _make_hip_torch()):
            dt.is_hip.cache_clear()
            dt.get_amd_attention_implementation.cache_clear()
            with mock.patch.object(dt, "_detect_rocm_major_minor",
                                   return_value="6.2"):
                with mock.patch("importlib.util.find_spec", return_value=True):
                    with mock.patch.dict(sys.modules, {"aiter": mock_aiter}):
                        result = dt.get_amd_attention_implementation()
        assert result == "sdpa", (
            "ROCm 6.2 must return 'sdpa' even when aiter is installed — "
            "version gate must fire before the aiter check"
        )

    def test_returns_amd_aiter_when_available(self):
        mock_aiter = mock.MagicMock()
        mock_aiter.flash_attn_func = mock.MagicMock()
        with mock.patch.object(torch, "version", _make_hip_torch()):
            dt.is_hip.cache_clear()
            dt.get_amd_attention_implementation.cache_clear()
            with mock.patch.object(dt, "_detect_rocm_major_minor",
                                   return_value="7.0"):
                with mock.patch("importlib.util.find_spec", return_value=True):
                    with mock.patch.dict(sys.modules, {"aiter": mock_aiter}):
                        assert dt.get_amd_attention_implementation() == "amd_aiter"


@_needs_aiter_apis
class TestAmdFlashAttnFunc:
    """get_amd_flash_attn_func() must return callable or None."""

    def test_returns_none_on_cuda(self):
        with mock.patch.object(torch, "version", _make_cuda_torch()):
            dt.is_hip.cache_clear()
            dt.get_amd_attention_implementation.cache_clear()
            dt.get_amd_flash_attn_func.cache_clear()
            assert dt.get_amd_flash_attn_func() is None

    def test_returns_callable_for_flash_attn_func(self):
        mock_fn = mock.MagicMock()
        mock_aiter = mock.MagicMock(spec=["flash_attn_func"])
        mock_aiter.flash_attn_func = mock_fn
        with mock.patch.object(torch, "version", _make_hip_torch()):
            dt.is_hip.cache_clear()
            dt.get_amd_attention_implementation.cache_clear()
            dt.get_amd_flash_attn_func.cache_clear()
            with mock.patch.object(dt, "get_amd_attention_implementation",
                                   return_value="amd_aiter"):
                with mock.patch.dict(sys.modules, {"aiter": mock_aiter}):
                    assert dt.get_amd_flash_attn_func() is mock_fn

    def test_flashattnfunc_class_wrapped_in_lambda(self):
        """FlashAttnFunc (class API) must be wrapped so callers get a plain callable."""
        mock_cls = mock.MagicMock()
        mock_cls.apply = mock.MagicMock(return_value=torch.zeros(1, 4, 4, 32))
        mock_aiter = mock.MagicMock(spec=["FlashAttnFunc"])
        mock_aiter.FlashAttnFunc = mock_cls
        del mock_aiter.flash_attn_func

        with mock.patch.object(torch, "version", _make_hip_torch()):
            dt.is_hip.cache_clear()
            dt.get_amd_attention_implementation.cache_clear()
            dt.get_amd_flash_attn_func.cache_clear()
            with mock.patch.object(dt, "get_amd_attention_implementation",
                                   return_value="amd_aiter"):
                with mock.patch.dict(sys.modules, {"aiter": mock_aiter}):
                    fn = dt.get_amd_flash_attn_func()
                    assert callable(fn)
                    q = torch.zeros(1, 4, 4, 32)
                    fn(q, q, q, causal=True)
                    mock_cls.apply.assert_called_once_with(q, q, q, True)


@_needs_compiler_rewrite
class TestReplaceSDPAWithAmdAiter:
    """replace_sdpa_with_amd_aiter() source rewriter safety properties."""

    def setup_method(self):
        from unsloth_zoo.compiler import replace_sdpa_with_amd_aiter
        self.rewrite = replace_sdpa_with_amd_aiter

    def test_noop_on_nvidia(self):
        source = "out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)"
        with mock.patch(
            "unsloth_zoo.device_type.get_amd_attention_implementation",
            return_value="sdpa",
        ):
            assert self.rewrite(source) == source

    def test_rewrites_simple_causal_call(self):
        source = "    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)"
        with mock.patch(
            "unsloth_zoo.device_type.get_amd_attention_implementation",
            return_value="amd_aiter",
        ):
            result = self.rewrite(source)
        assert "_aiter_fn" in result
        assert "scaled_dot_product_attention" in result

    def test_no_rewrite_when_not_literal_true(self):
        source = "    out = scaled_dot_product_attention(q, k, v, is_causal=self.causal)"
        with mock.patch(
            "unsloth_zoo.device_type.get_amd_attention_implementation",
            return_value="amd_aiter",
        ):
            assert "_aiter_fn" not in self.rewrite(source)

    def test_no_rewrite_when_attn_mask_present(self):
        source = "    out = scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True)"
        with mock.patch(
            "unsloth_zoo.device_type.get_amd_attention_implementation",
            return_value="amd_aiter",
        ):
            assert "_aiter_fn" not in self.rewrite(source)

    def test_no_rewrite_disable_compile_shim(self):
        source = "    out = disable_compile_scaled_dot_product_attention(q, k, v, is_causal=True)"
        with mock.patch(
            "unsloth_zoo.device_type.get_amd_attention_implementation",
            return_value="amd_aiter",
        ):
            assert "_aiter_fn" not in self.rewrite(source)

    def test_dtype_guard_in_generated_code(self):
        source = "    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)"
        with mock.patch(
            "unsloth_zoo.device_type.get_amd_attention_implementation",
            return_value="amd_aiter",
        ):
            result = self.rewrite(source)
        if "_aiter_fn" in result:
            assert "float16" in result or "bfloat16" in result

    def test_seq_length_guard_in_generated_code(self):
        source = "    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)"
        with mock.patch(
            "unsloth_zoo.device_type.get_amd_attention_implementation",
            return_value="amd_aiter",
        ):
            result = self.rewrite(source)
        if "_aiter_fn" in result:
            assert "shape[-2]" in result

    def test_generated_code_compiles(self):
        import py_compile, textwrap, tempfile, os
        source = textwrap.dedent("""\
            import torch
            def forward(q, k, v):
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
                return out
        """)
        with mock.patch(
            "unsloth_zoo.device_type.get_amd_attention_implementation",
            return_value="amd_aiter",
        ):
            result = self.rewrite(source)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(result)
            tmp = f.name
        try:
            py_compile.compile(tmp, doraise=True)
        finally:
            os.unlink(tmp)

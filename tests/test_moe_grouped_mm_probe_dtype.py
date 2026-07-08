"""The torch._grouped_mm support probe must use bfloat16, not float16.

The non-scaled torch._grouped_mm grouped kernel is bf16-only. On torch 2.8 a
float16 probe raises outright (hard TORCH_CHECK, no fallback) and caches the
backend as unsupported even on hardware that supports it; on torch >= 2.9 a
float16 probe only "passes" through the slow per-group fallback loop, not the
grouped kernel the backend actually targets. Probe with bfloat16, the dtype
the backend runs with.
"""

import torch

from unsloth_zoo.temporary_patches import moe_utils


def test_probe_uses_bf16_and_detects_support(monkeypatch):
    recorded_dtypes = []

    def fake_grouped_mm(x, w, offs=None):
        recorded_dtypes.append(x.dtype)
        if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
            raise RuntimeError(
                f"Expected inputs of BF16 type but got mat_a.dtype={x.dtype}"
            )
        return torch.zeros(x.shape[0], w.shape[-1], dtype=x.dtype)

    # Pretend a GPU is present but keep probe tensors on CPU so the test is
    # deterministic on CPU-only CI and GPU hosts alike.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch, "_grouped_mm", fake_grouped_mm, raising=False)
    monkeypatch.setattr(moe_utils, "_TORCH_GROUPED_MM_AVAILABLE", True)
    # Reset the module-level cache so the probe re-runs; monkeypatch restores it.
    monkeypatch.setattr(moe_utils, "_TORCH_GROUPED_MM_SUPPORTED", None)

    assert moe_utils._check_torch_grouped_mm_supported() is True
    assert recorded_dtypes == [torch.bfloat16]


def test_backend_selection_is_dtype_aware(monkeypatch):
    """A passing bf16 probe must not route non-bf16 forwards to grouped_mm.

    torch._grouped_mm's grouped kernel is bf16-only (raises on torch 2.8 eager
    and under compile's meta checks; slow per-group fallback on >= 2.9), so with
    the probe fixed to bf16 the backend selection has to stay dtype-aware:
    non-bf16 activations go to the Triton path (or the native loop), and only
    dtype=None keeps the permissive legacy behavior.
    """
    monkeypatch.delenv("UNSLOTH_MOE_BACKEND", raising=False)
    monkeypatch.setattr(
        moe_utils, "_check_torch_grouped_mm_supported", lambda: True
    )
    monkeypatch.setattr(moe_utils, "_check_grouped_gemm_available", lambda: True)

    moe_utils.select_moe_backend.cache_clear()
    try:
        assert moe_utils.select_moe_backend(torch.bfloat16) == "grouped_mm"
        assert moe_utils.select_moe_backend(torch.float16) == "unsloth_triton"
        assert moe_utils.select_moe_backend(torch.float32) == "unsloth_triton"
        # Callers that don't know the activation dtype keep the old behavior.
        assert moe_utils.select_moe_backend() == "grouped_mm"

        # Without Triton, non-bf16 falls through to the native loop.
        monkeypatch.setattr(
            moe_utils, "_check_grouped_gemm_available", lambda: False
        )
        moe_utils.select_moe_backend.cache_clear()
        assert moe_utils.select_moe_backend(torch.float16) == "native_torch"
        assert moe_utils.select_moe_backend(torch.bfloat16) == "grouped_mm"
    finally:
        # Drop entries computed against the patched availability checks.
        moe_utils.select_moe_backend.cache_clear()

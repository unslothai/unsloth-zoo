"""The torch._grouped_mm support probe must use bfloat16, not float16.

The non-scaled torch._grouped_mm is bf16-only, so a float16 probe always raises
and caches the backend as unsupported even on hardware that supports it. Probe
with bfloat16 so supported hardware is detected.
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

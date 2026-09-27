"""The SM90 `_scaled_grouped_mm` FP8 MoE path: rhs must be (E, K, N) column major, and
`weight_scale_inv` is a dequant multiplier, not a divisor. Checked on meta + CPU (no Hopper needed)."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

m = pytest.importorskip("unsloth_zoo.temporary_patches.moe_utils_fp8")
FP8 = getattr(torch, "float8_e4m3fn", None)
E, H, I = 2, 32, 48


def _experts(kind):
    torch.manual_seed(0)
    ex = nn.Module()
    ex.num_experts = E
    ex.gate_up_proj = nn.Parameter((torch.randn(E, 2 * I, H) * 50).to(FP8), requires_grad = False)
    ex.down_proj = nn.Parameter((torch.randn(E, H, I) * 50).to(FP8), requires_grad = False)
    setattr(ex, f"gate_up_proj_{kind}", nn.Parameter(torch.rand(E, 2 * I) * 0.01 + 1e-3, requires_grad = False))
    setattr(ex, f"down_proj_{kind}", nn.Parameter(torch.rand(E, H) * 0.01 + 1e-3, requires_grad = False))
    return ex


@pytest.mark.skipif(FP8 is None or not hasattr(torch, "_scaled_grouped_mm"), reason = "needs fp8 + _scaled_grouped_mm")
@pytest.mark.parametrize("kind", ["scale_inv", "scale"])
@pytest.mark.parametrize("proj,proj_type,out_dim", [("gate_up_proj", "gate_up", 2 * I), ("down_proj", "down", H)])
def test_prepared_rhs_and_scale(kind, proj, proj_type, out_dim):
    ex = _experts(kind)
    prepared = m._prepare_scaled_grouped_mm_weight(ex, proj, proj_type, H, None)
    assert prepared is not None
    rhs, scale = prepared
    in_dim = H if proj_type == "gate_up" else I
    assert tuple(rhs.shape) == (E, in_dim, out_dim)
    # torch's own meta checks for _scaled_grouped_mm (column-major mat_b, contraction dim, scale shape).
    torch._scaled_grouped_mm(
        torch.empty(8, in_dim, dtype = FP8, device = "meta"),
        torch.empty_strided(rhs.shape, rhs.stride(), dtype = FP8, device = "meta"),
        torch.empty(8, device = "meta"),
        torch.empty(scale.shape, device = "meta"),
        offs = torch.tensor([4, 8], dtype = torch.int32, device = "meta"),
        out_dtype = torch.bfloat16,
    )
    q = getattr(ex, proj).detach()
    s = getattr(ex, f"{proj}_{kind}").detach()
    x = torch.randn(3, in_dim)
    for e in range(E):
        got = x @ (rhs[e].float() * scale[e][None, :])
        ref = F.linear(x, q[e].float() * s[e][:, None])
        torch.testing.assert_close(got, ref)


def test_scaled_grouped_mm_path_is_opt_in(monkeypatch):
    import torch
    from unsloth_zoo.temporary_patches import moe_utils_fp8 as m

    monkeypatch.setattr(m, "_TORCH_SCALED_GROUPED_MM_SUPPORTED", None)
    monkeypatch.setattr(m, "_TORCH_SCALED_GROUPED_MM_AVAILABLE", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (9, 0))
    monkeypatch.delenv("UNSLOTH_FP8_SCALED_GROUPED_MM", raising=False)
    assert m._check_torch_scaled_grouped_mm_supported() is False

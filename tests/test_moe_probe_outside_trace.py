"""The lazy grouped_mm probes must never be evaluated inside a Dynamo trace.

Their first call is usually the first MoE forward, which in a compiled model runs inside
torch.compile. Traced, `torch._grouped_mm` sees FakeTensors and can raise where the real
kernel works (torch 2.14 / B200: "Float16 grouped_mm requires cuBLASLt grouped GEMM
support"); with Unsloth's dynamo config the probe's `except` branch was committed and
`_TORCH_GROUPED_MM_SUPPORTED` latched False, sending bf16 ERNIE-4.5 MoE to a wrong backend
(PPL 1550 vs 20.7).
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import unsloth_zoo.temporary_patches.moe_utils as M

# The Dynamo settings `import unsloth` applies, under which the bug reproduced.
_UNSLOTH_DYNAMO = dict(suppress_errors=True, capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True)


@pytest.fixture
def fresh_probes(monkeypatch):
    monkeypatch.setattr(M, "_TORCH_GROUPED_MM_SUPPORTED", None)
    monkeypatch.setattr(M, "_TRANSPOSED_VIEW_GROUPED_MM_SAFE", None)
    M.select_moe_backend.cache_clear()
    torch._dynamo.reset()
    yield
    M.select_moe_backend.cache_clear()
    torch._dynamo.reset()


@pytest.mark.parametrize("check, probe, flag", [
    ("_check_torch_grouped_mm_supported", "_probe_torch_grouped_mm_supported", "_TORCH_GROUPED_MM_SUPPORTED"),
    ("_transposed_view_grouped_mm_is_safe", "_probe_transposed_view_grouped_mm_is_safe", "_TRANSPOSED_VIEW_GROUPED_MM_SAFE"),
])
def test_probe_body_runs_eagerly_under_compile(fresh_probes, monkeypatch, check, probe, flag):
    seen = []

    def fake_probe():
        compiling = torch.compiler.is_compiling()
        seen.append(compiling)
        # A traced probe answers differently from the device, as the FakeTensor one did.
        monkeypatch.setattr(M, flag, not compiling)
        return not compiling

    monkeypatch.setattr(M, probe, fake_probe)
    # Pretend an accelerator with the op exists, so the check reaches its (fake) kernel probe.
    monkeypatch.setattr(M, "_TORCH_GROUPED_MM_AVAILABLE", True)
    monkeypatch.setattr(M, "_grouped_mm_probe_device", lambda: torch.device("cpu"))
    fn = getattr(M, check)

    def f(x):
        return x + 1 if fn() else x - 1

    with torch._dynamo.config.patch(**_UNSLOTH_DYNAMO):
        out = torch.compile(f, backend="eager", dynamic=True)(torch.zeros(3))
    assert seen == [False], seen            # probed once, eagerly
    assert getattr(M, flag) is True         # the real answer was cached
    assert torch.equal(out, torch.ones(3))  # and the compiled branch used it
    # Cached now: a second trace reads the flag and never re-probes.
    torch._dynamo.reset()
    with torch._dynamo.config.patch(**_UNSLOTH_DYNAMO):
        torch.compile(f, backend="eager", dynamic=True)(torch.zeros(3))
    assert seen == [False]


def _gpu_grouped_mm_ok():
    if not torch.cuda.is_available() or not hasattr(torch, "_grouped_mm"):
        return False
    try:
        torch._grouped_mm(torch.ones(1, 8, device="cuda", dtype=torch.bfloat16),
                          torch.ones(1, 8, 8, device="cuda", dtype=torch.bfloat16),
                          offs=torch.tensor([1], device="cuda", dtype=torch.int32))
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_grouped_mm_ok(), reason="needs a GPU with torch._grouped_mm")
def test_first_moe_forward_inside_compile_matches_reference(fresh_probes):
    torch.manual_seed(0)
    E, H, I, T, K = 8, 64, 32, 48, 2

    class Experts(nn.Module):  # ERNIE-4.5 / Qwen3-MoE fused layout
        def __init__(self):
            super().__init__()
            self.num_experts, self.hidden_dim, self.intermediate_dim = E, H, I
            self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * I, H) * 0.1)
            self.down_proj = nn.Parameter(torch.randn(E, H, I) * 0.1)
            self.act_fn = nn.SiLU()
        forward = M.forward_moe_backend

    m = Experts().cuda().to(torch.bfloat16)
    hs = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    idx = torch.stack([torch.randperm(E, device="cuda")[:K] for _ in range(T)])
    w = torch.softmax(torch.randn(T, K, device="cuda"), -1)  # fp32 router weights, as ERNIE's
    ref = torch.zeros(T, H, device="cuda")
    for e in range(E):
        t, p = torch.where(idx == e)
        g, u = F.linear(hs[t].float(), m.gate_up_proj[e].float()).chunk(2, -1)
        ref.index_add_(0, t, F.linear(F.silu(g) * u, m.down_proj[e].float()) * w[t, p, None])

    def block(mod, x, i, ww):
        return mod(x * 1.0, i, ww)

    with torch.no_grad(), torch._dynamo.config.patch(**_UNSLOTH_DYNAMO):
        out = torch.compile(block, dynamic=True)(m, hs, idx, w).float()
    assert M._TORCH_GROUPED_MM_SUPPORTED is True
    assert M.select_moe_backend() == "grouped_mm"
    rel = ((out - ref).norm() / ref.norm()).item()
    assert rel < 2e-2, rel


def test_definitive_negative_does_not_break_a_fullgraph_trace(monkeypatch):
    """With no accelerator (or no torch._grouped_mm) the answer is known without a kernel
    probe, so a fullgraph trace must get False instead of an Unsupported graph break."""
    import torch
    from unsloth_zoo.temporary_patches import moe_utils

    monkeypatch.setattr(moe_utils, "_TORCH_GROUPED_MM_SUPPORTED", None)
    monkeypatch.setattr(moe_utils, "_TRANSPOSED_VIEW_GROUPED_MM_SAFE", None)
    monkeypatch.setattr(moe_utils, "_grouped_mm_probe_device", lambda: None)
    torch._dynamo.reset()

    @torch.compile(fullgraph = True, backend = "eager")
    def f(x):
        if moe_utils._check_torch_grouped_mm_supported() or moe_utils._transposed_view_grouped_mm_is_safe():
            return x + 1
        return x - 1

    assert torch.equal(f(torch.zeros(2)), torch.full((2,), -1.0))
    assert moe_utils._TORCH_GROUPED_MM_SUPPORTED is False
    assert moe_utils._TRANSPOSED_VIEW_GROUPED_MM_SAFE is False

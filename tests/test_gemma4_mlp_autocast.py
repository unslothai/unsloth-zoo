import os
import sys
import types
import importlib.util

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
if "unsloth" not in sys.modules:
    _stub = types.ModuleType("unsloth")
    _stub.__spec__ = importlib.util.spec_from_loader("unsloth", loader=None)
    _stub.__path__ = []
    sys.modules["unsloth"] = _stub

import pytest
import torch
import torch.nn as nn

from unsloth_zoo.temporary_patches import gemma4 as g4


def _make_mlp_class():
    class Gemma4TextMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(64, 2048, bias=False)
            self.up_proj = nn.Linear(64, 2048, bias=False)
            self.down_proj = nn.Linear(2048, 64, bias=False)
            self.act_fn = nn.GELU(approximate="tanh")
            with torch.no_grad():
                for p in (
                    self.gate_proj.weight,
                    self.up_proj.weight,
                    self.down_proj.weight,
                ):
                    p.fill_(0.5)

        def forward(self, x):
            return self.down_proj(
                self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            )

    return Gemma4TextMLP


def _install_module_stub(monkeypatch, cls):
    fake = types.ModuleType("transformers.models.gemma4.modeling_gemma4")
    fake.Gemma4TextMLP = cls
    for pkg in (
        "transformers",
        "transformers.models",
        "transformers.models.gemma4",
    ):
        if pkg not in sys.modules:
            p = types.ModuleType(pkg)
            p.__path__ = []
            monkeypatch.setitem(sys.modules, pkg, p)
    monkeypatch.setitem(
        sys.modules, "transformers.models.gemma4.modeling_gemma4", fake
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_bf16_weights_fp16_autocast_stabilizes(monkeypatch):
    # On bf16-capable GPUs a user can wrap inference in torch.amp.autocast(fp16)
    # over a bf16-loaded model. x.dtype stays bf16, but self.gate_proj(x)
    # executes in fp16 and can overflow. The gate.dtype guard must detect
    # this and enter stabilization; an x.dtype guard would bypass it.
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    cls = _make_mlp_class()
    _install_module_stub(monkeypatch, cls)
    g4.patch_Gemma4TextMLP()
    m = cls().cuda().to(torch.bfloat16).eval()
    torch.manual_seed(0)
    x = torch.randn(2, 64, dtype=torch.bfloat16, device="cuda") * 20.0
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        out = m(x)
    assert torch.all(torch.isfinite(out)).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp32_weights_fp16_autocast_stabilizes(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    cls = _make_mlp_class()
    _install_module_stub(monkeypatch, cls)
    g4.patch_Gemma4TextMLP()
    m = cls().cuda().eval()
    torch.manual_seed(0)
    x = torch.randn(2, 64, device="cuda") * 20.0
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        out = m(x)
    assert torch.all(torch.isfinite(out)).item()


def test_idempotent_patch_install(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    cls = _make_mlp_class()
    _install_module_stub(monkeypatch, cls)
    g4.patch_Gemma4TextMLP()
    first = cls.forward
    g4.patch_Gemma4TextMLP()
    second = cls.forward
    # Second call should leave the class in a patched, working state.
    assert second is not None
    assert second.__name__ == "forward"
    m = cls().half().eval()
    torch.manual_seed(0)
    x = torch.randn(2, 64, dtype=torch.float16) * 0.1
    with torch.no_grad():
        out = m(x)
    assert torch.all(torch.isfinite(out)).item()
    assert (out != 0).any().item()


def test_pure_bf16_path_bypasses_even_with_overflow_scale(monkeypatch):
    # Pure bf16 path should NEVER enter stabilization regardless of input
    # magnitude, because bf16's dynamic range does not overflow at these
    # scales. gate.dtype guard must send it to upstream verbatim.
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    cls = _make_mlp_class()
    _install_module_stub(monkeypatch, cls)
    upstream = cls.forward
    g4.patch_Gemma4TextMLP()
    m = cls().to(torch.bfloat16).eval()
    torch.manual_seed(0)
    x = torch.randn(2, 64, dtype=torch.bfloat16) * 50.0
    with torch.no_grad():
        patched = m(x)
        expected = upstream(m, x)
    assert torch.equal(patched, expected)
    assert patched.dtype == torch.bfloat16

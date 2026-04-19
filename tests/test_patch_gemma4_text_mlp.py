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


def test_noop_when_force_float32_unset(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FORCE_FLOAT32", raising=False)
    cls = _make_mlp_class()
    _install_module_stub(monkeypatch, cls)
    original = cls.forward
    g4.patch_Gemma4TextMLP()
    assert cls.forward is original


def test_noop_when_force_float32_zero(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "0")
    cls = _make_mlp_class()
    _install_module_stub(monkeypatch, cls)
    original = cls.forward
    g4.patch_Gemma4TextMLP()
    assert cls.forward is original


def test_upstream_without_patch_overflows_fp16():
    cls = _make_mlp_class()
    m = cls().half().eval()
    torch.manual_seed(0)
    x = torch.randn(2, 64, dtype=torch.float16) * 20.0
    with torch.no_grad():
        out = m(x)
    assert (~torch.isfinite(out)).any().item()


def test_fp16_overflow_output_is_finite(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    cls = _make_mlp_class()
    _install_module_stub(monkeypatch, cls)
    g4.patch_Gemma4TextMLP()
    m = cls().half().eval()
    torch.manual_seed(0)
    x = torch.randn(2, 64, dtype=torch.float16) * 20.0
    with torch.no_grad():
        out = m(x)
    assert torch.all(torch.isfinite(out)).item()
    assert out.dtype == torch.float16


def test_fp16_nan_to_num_replaces_with_zero(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    cls = _make_mlp_class()
    _install_module_stub(monkeypatch, cls)
    g4.patch_Gemma4TextMLP()
    m = cls().half().eval()
    torch.manual_seed(0)
    x = torch.randn(2, 64, dtype=torch.float16) * 20.0
    with torch.no_grad():
        out = m(x)
    assert out.abs().max().item() == 0.0


def test_fp16_normal_input_produces_nonzero_output(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    cls = _make_mlp_class()
    _install_module_stub(monkeypatch, cls)
    g4.patch_Gemma4TextMLP()
    m = cls().half().eval()
    torch.manual_seed(0)
    x = torch.randn(2, 64, dtype=torch.float16) * 0.1
    with torch.no_grad():
        out = m(x)
    assert torch.all(torch.isfinite(out)).item()
    assert (out != 0).any().item()


def test_bf16_input_matches_upstream(monkeypatch):
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


def test_fp32_input_matches_upstream(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    cls = _make_mlp_class()
    _install_module_stub(monkeypatch, cls)
    upstream = cls.forward
    g4.patch_Gemma4TextMLP()
    m = cls().eval()
    torch.manual_seed(0)
    x = torch.randn(2, 64)
    with torch.no_grad():
        patched = m(x)
        expected = upstream(m, x)
    assert torch.equal(patched, expected)


def test_idempotent_patch_install(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    cls = _make_mlp_class()
    _install_module_stub(monkeypatch, cls)
    g4.patch_Gemma4TextMLP()
    g4.patch_Gemma4TextMLP()
    assert cls.forward.__name__ == "forward"
    m = cls().half().eval()
    torch.manual_seed(0)
    x = torch.randn(2, 64, dtype=torch.float16) * 0.1
    with torch.no_grad():
        out = m(x)
    assert torch.all(torch.isfinite(out)).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_bf16_weights_fp16_autocast_stabilizes(monkeypatch):
    # bf16-capable GPU + user-wrapped torch.amp.autocast(fp16): x.dtype stays
    # bf16 but self.gate_proj(x) runs in fp16 and can overflow. The
    # gate.dtype guard must enter stabilization here; an x.dtype guard would
    # bypass and leave the overflow unfixed.
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

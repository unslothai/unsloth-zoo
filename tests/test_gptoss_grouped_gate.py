"""Env gates of GptOssExpertsBnb4bit._grouped_bnb4bit_ready: ON by default, OFF with UNSLOTH_GPTOSS_GROUPED=0, ON with
UNSLOTH_COMPILE_DISABLE=1/partial; probe stays self-contained (compiler copies its source); no CUDA compute, but
importing unsloth_zoo needs a torch accelerator."""
import os
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

bnb = pytest.importorskip("bitsandbytes")
from bitsandbytes.nn import Params4bit

try:
    import unsloth_zoo.temporary_patches.moe_utils as mu
    from unsloth_zoo.temporary_patches.gpt_oss import GptOssExpertsBnb4bit
except Exception as e:  # pragma: no cover - unsloth_zoo import needs an accelerator
    pytest.skip(f"cannot import unsloth_zoo gpt_oss patches: {e}", allow_module_level=True)

from unsloth_zoo.stubs.bitsandbytes_stub import real_bitsandbytes_available

if not real_bitsandbytes_available():
    pytest.skip("bitsandbytes is the unsloth_zoo stub on this host", allow_module_level=True)


_NF4_CODE = torch.linspace(-1, 1, 16)


def _fake_expert_linear():
    """LoRA-free bnb Linear4bit-like expert passing all readiness checks: Params4bit stays unquantized on CPU with a hand-built quant_state, so no GPU dequant runs."""
    w = Params4bit(
        torch.zeros(4, 32, 64, dtype=torch.bfloat16),
        requires_grad=False,
        quant_type="nf4",
    )
    # Every field a real QuantState carries that the readiness format check reads.
    w.quant_state = SimpleNamespace(
        blocksize=64,
        shape=torch.Size([4, 32, 64]),
        quant_type="nf4",
        dtype=torch.bfloat16,
        nested=False,
        code=_NF4_CODE,
        absmax=torch.ones(4 * 32, dtype=torch.float32),
    )
    return SimpleNamespace(weight=w, bias=torch.zeros(4, dtype=torch.bfloat16))


@pytest.fixture
def experts(monkeypatch):
    # Hold the hardware/torch capability check True so only the env gates decide.
    monkeypatch.setattr(mu, "_check_torch_grouped_mm_supported", lambda: True)
    fe = SimpleNamespace(
        gate_up_projs=[_fake_expert_linear()],
        down_projs=[_fake_expert_linear()],
    )
    return fe


def _ready(experts):
    # Call the unbound method against the fake experts object.
    return GptOssExpertsBnb4bit._grouped_bnb4bit_ready(experts)


def test_grouped_ready_default_on(experts, monkeypatch):
    monkeypatch.delenv("UNSLOTH_GPTOSS_GROUPED", raising=False)
    monkeypatch.delenv("UNSLOTH_COMPILE_DISABLE", raising=False)
    assert _ready(experts) is True


def test_grouped_disabled_by_gptoss_grouped_zero(experts, monkeypatch):
    monkeypatch.delenv("UNSLOTH_COMPILE_DISABLE", raising=False)
    monkeypatch.setenv("UNSLOTH_GPTOSS_GROUPED", "0")
    assert _ready(experts) is False


@pytest.mark.parametrize("value", ["1", "partial"])
def test_compile_disable_keeps_grouped_on(experts, monkeypatch, value):
    monkeypatch.delenv("UNSLOTH_GPTOSS_GROUPED", raising=False)
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", value)
    assert _ready(experts) is True


def test_compile_disable_zero_keeps_grouped_on(experts, monkeypatch):
    monkeypatch.delenv("UNSLOTH_GPTOSS_GROUPED", raising=False)
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "0")
    assert _ready(experts) is True

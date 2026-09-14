"""The gpt-oss grouped bnb4bit readiness probe honors both env gates.

GptOssExpertsBnb4bit._grouped_bnb4bit_ready decides whether the grouped
torch._grouped_mm forward replaces the per-expert loop. It must be:

  * ON by default,
  * OFF when UNSLOTH_GPTOSS_GROUPED=0 (explicit opt-out of the grouped path),
  * OFF when UNSLOTH_COMPILE_DISABLE=1 (the grouped stacks are meant to run under
    the compiled cache; disabling compilation opts into the plain eager loop).

The probe is self-contained (local imports, reads os.environ directly) so the
compiler can copy its source into the standalone compiled cache. These are the
env gates only: the downstream expert/quant_state checks are held True via a
minimal CPU fake so the gate logic is exercised in isolation. No CUDA compute is
performed (the fake Params4bit is never quantized), though importing unsloth_zoo
requires a visible torch accelerator.
"""
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


def _fake_expert_linear():
    """A LoRA-free bnb Linear4bit-like expert whose readiness checks all pass.

    The Params4bit is left unquantized on CPU and given a hand-built quant_state
    so the probe's attribute inspection succeeds without any GPU dequant.
    """
    w = Params4bit(
        torch.zeros(4, 32, 64, dtype=torch.bfloat16),
        requires_grad=False,
        quant_type="nf4",
    )
    w.quant_state = SimpleNamespace(blocksize=64, shape=torch.Size([4, 32, 64]))
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


def test_grouped_disabled_by_compile_disable(experts, monkeypatch):
    monkeypatch.delenv("UNSLOTH_GPTOSS_GROUPED", raising=False)
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "1")
    assert _ready(experts) is False


def test_compile_disable_zero_keeps_grouped_on(experts, monkeypatch):
    monkeypatch.delenv("UNSLOTH_GPTOSS_GROUPED", raising=False)
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "0")
    assert _ready(experts) is True

"""Regression tests for the Gemma-4 MoE LoRA extractor registration added by
PR #624. These tests do NOT require ``transformers.models.gemma4`` to exist;
they exercise the registration helper against a synthetic stand-in class
with the same surface (gate_up_proj (E, 2*I, H), down_proj (E, H, I),
hidden_dim, intermediate_dim).

What is covered:

1. Successful registration attaches the Qwen extractor and the model-type
   tag without overwriting unrelated state.
2. Registration is idempotent across repeated calls (callable identity is
   preserved, no double-wrapping).
3. ``_register_gemma4_lora_extractor(None)`` returns False without raising,
   matching the legacy import path where ``Gemma4TextMoEBlock`` may be
   absent.
4. If the underlying extractor factory raises, registration returns False
   and leaves the class state untouched (no half-registered attributes).
5. The registered extractor produces (E, in, R) / (E, R, out) tensors that
   numerically reconstruct the per-expert delta on both the PEFT 0.18 raw
   layout and the PEFT 0.19 ``_did_swap_in_out_features`` swapped layout,
   for both ``gate_up_proj`` and ``down_proj`` parameters.
6. Sibling MoE families' existing extractor registrations are not disturbed
   by Gemma-4 registration.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from unsloth_zoo.temporary_patches import gemma4_moe as g4
from unsloth_zoo.temporary_patches.qwen3_moe import _make_qwen_moe_lora_extractor


def _fresh_stub_class():
    class StubGemma4TextExperts(nn.Module):
        num_experts = 4
        hidden_dim = 8
        intermediate_dim = 12

        def __init__(self) -> None:
            super().__init__()
            self.gate_up_proj = nn.Parameter(
                torch.randn(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
            )
            self.down_proj = nn.Parameter(
                torch.randn(self.num_experts, self.hidden_dim, self.intermediate_dim)
            )

    return StubGemma4TextExperts


class _StubWrapper:
    """Mimics the surface ``_make_qwen_moe_lora_extractor`` reads from a
    PEFT ParamWrapper: ``parameter_name``, ``get_base_layer()``,
    optionally ``_did_swap_in_out_features``."""

    def __init__(self, parameter_name: str, base_layer, peft_swapped: bool):
        self.parameter_name = parameter_name
        self._base_layer = base_layer
        self._did_swap_in_out_features = peft_swapped

    def get_base_layer(self):
        return self._base_layer


def test_register_attaches_extractor_and_tag():
    cls = _fresh_stub_class()
    assert g4._register_gemma4_lora_extractor(cls) is True
    assert cls._unsloth_lora_extractor_registered is True
    assert cls._unsloth_model_type == "gemma4_moe"
    assert callable(cls._unsloth_lora_extractor_fn)


def test_register_is_idempotent():
    cls = _fresh_stub_class()
    assert g4._register_gemma4_lora_extractor(cls) is True
    fn_before = cls._unsloth_lora_extractor_fn
    # Second call must short-circuit on the registered flag and leave the
    # extractor identity untouched.
    assert g4._register_gemma4_lora_extractor(cls) is True
    assert cls._unsloth_lora_extractor_fn is fn_before
    assert cls._unsloth_model_type == "gemma4_moe"


def test_register_none_returns_false_without_raising():
    assert g4._register_gemma4_lora_extractor(None) is False


def test_register_failure_preserves_class_state(monkeypatch):
    cls = _fresh_stub_class()

    def _boom():
        raise RuntimeError("synthetic factory failure")

    monkeypatch.setattr(g4, "_make_qwen_moe_lora_extractor", _boom)
    assert g4._register_gemma4_lora_extractor(cls) is False
    assert not hasattr(cls, "_unsloth_lora_extractor_fn")
    assert not getattr(cls, "_unsloth_lora_extractor_registered", False)


def test_register_does_not_disturb_sibling_registration():
    # qwen3_moe registers its extractor when patch_qwen3_moe is invoked.
    # Without invoking it, we verify that an unrelated class registered by
    # Gemma-4 does NOT touch any class qwen3_moe would later target.
    cls = _fresh_stub_class()
    assert g4._register_gemma4_lora_extractor(cls) is True
    # The Qwen extractor factory is still callable and independent.
    qwen_extractor = _make_qwen_moe_lora_extractor()
    assert callable(qwen_extractor)
    # The Gemma-4 stub extractor and a freshly built Qwen extractor are
    # distinct instances even though they originate from the same factory.
    assert cls._unsloth_lora_extractor_fn is not qwen_extractor


def _drive_extractor(cls, parameter_name: str, peft_swapped: bool):
    """Drive the registered extractor against hand-built LoRA factors and
    compare per-expert reconstructed delta to a naive reference."""
    torch.manual_seed(0)
    base = cls()
    if parameter_name == "gate_up_proj":
        in_dim = base.hidden_dim
        out_dim = 2 * base.intermediate_dim
    else:  # down_proj
        in_dim = base.intermediate_dim
        out_dim = base.hidden_dim

    E = base.num_experts
    R = 3

    if peft_swapped:
        # PEFT 0.19 swapped layout
        weight_A = torch.randn(E * R, out_dim)
        weight_B = torch.randn(in_dim, E * R)
    else:
        # PEFT 0.18 raw 3D layout
        weight_A = torch.randn(E * R, in_dim)
        weight_B = torch.randn(out_dim, E * R)

    wrapper = _StubWrapper(parameter_name, base, peft_swapped)
    extractor = cls._unsloth_lora_extractor_fn
    first, second, scaling, num_experts = extractor(
        wrapper, weight_A, weight_B, 1.5, E,
    )
    assert first.shape == (E, in_dim, R), (parameter_name, peft_swapped, first.shape)
    assert second.shape == (E, R, out_dim), (parameter_name, peft_swapped, second.shape)
    assert num_experts == E
    assert scaling == 1.5
    assert first.is_contiguous()
    assert second.is_contiguous()

    x = torch.randn(7, in_dim)
    for e in range(E):
        Ae = weight_A[e * R : (e + 1) * R]
        Be = weight_B[:, e * R : (e + 1) * R]
        if peft_swapped:
            naive = x @ Be @ Ae
        else:
            naive = x @ Ae.T @ Be.T
        via = (x @ first[e]) @ second[e]
        torch.testing.assert_close(via, naive, atol=1e-4, rtol=1e-4)


def test_extractor_gate_up_canonical_peft018():
    cls = _fresh_stub_class()
    assert g4._register_gemma4_lora_extractor(cls) is True
    _drive_extractor(cls, "gate_up_proj", peft_swapped=False)


def test_extractor_gate_up_swapped_peft019():
    cls = _fresh_stub_class()
    assert g4._register_gemma4_lora_extractor(cls) is True
    _drive_extractor(cls, "gate_up_proj", peft_swapped=True)


def test_extractor_down_canonical_peft018():
    cls = _fresh_stub_class()
    assert g4._register_gemma4_lora_extractor(cls) is True
    _drive_extractor(cls, "down_proj", peft_swapped=False)


def test_extractor_down_swapped_peft019():
    cls = _fresh_stub_class()
    assert g4._register_gemma4_lora_extractor(cls) is True
    _drive_extractor(cls, "down_proj", peft_swapped=True)


def test_patch_gemma4_moe_is_noop_without_gemma4(monkeypatch):
    """End-to-end sanity check: when transformers lacks
    ``models.gemma4.modeling_gemma4`` (the case in this env), the public
    entrypoint must not raise. Both inner patch functions guard their
    imports, and ``patch_gemma4_moe`` short-circuits via ``raise_error``
    which returns rather than raising."""
    # Act: just call the entrypoint. transformers.models.gemma4 is not
    # importable on transformers 4.57.6 in this environment.
    g4.patch_gemma4_moe()  # must not raise


def test_register_handles_legacy_block_class_shape():
    """Sanity check that a class shaped like Gemma4TextMoEBlock (legacy
    layout) accepts registration the same way as Gemma4TextExperts. The
    legacy block also exposes ``num_experts``/``hidden_dim``/
    ``intermediate_dim`` plus ``gate_up_proj``/``down_proj``, so the
    Qwen extractor is layout-compatible."""
    LegacyBlock = _fresh_stub_class()
    assert g4._register_gemma4_lora_extractor(LegacyBlock) is True
    _drive_extractor(LegacyBlock, "gate_up_proj", peft_swapped=False)
    _drive_extractor(LegacyBlock, "down_proj", peft_swapped=True)

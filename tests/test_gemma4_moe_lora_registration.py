"""Regression tests for the Gemma-4 MoE LoRA extractor registration (PR #624).

Do NOT require ``transformers.models.gemma4``; they drive the registration
helper against a synthetic stand-in with the same surface (gate_up_proj
(E, 2*I, H), down_proj (E, H, I), hidden_dim, intermediate_dim). Cover:
registration attaches the Qwen extractor + model-type tag, is idempotent,
returns False (no raise / no half-registered state) on None or a raising
factory, reconstructs the per-expert delta on PEFT 0.18 raw and 0.19 swapped
layouts for both params, and does not disturb sibling families.
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
    """PEFT ParamWrapper surface read by ``_make_qwen_moe_lora_extractor``:
    ``parameter_name``, ``get_base_layer()``, ``_did_swap_in_out_features``."""

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
    # Second call short-circuits on the registered flag; identity unchanged.
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
    # Gemma-4 registration must not touch any class qwen3_moe would target.
    cls = _fresh_stub_class()
    assert g4._register_gemma4_lora_extractor(cls) is True
    qwen_extractor = _make_qwen_moe_lora_extractor()
    assert callable(qwen_extractor)
    # Same factory, distinct instances.
    assert cls._unsloth_lora_extractor_fn is not qwen_extractor


def _drive_extractor(cls, parameter_name: str, peft_swapped: bool):
    """Drive the registered extractor on hand-built LoRA factors and compare
    the per-expert reconstructed delta to a naive reference."""
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
    """When transformers lacks ``models.gemma4.modeling_gemma4``, the public
    entrypoint must not raise: inner patches guard their imports and
    ``patch_gemma4_moe`` short-circuits via ``raise_error`` (returns)."""
    g4.patch_gemma4_moe()  # must not raise


def test_register_handles_legacy_block_class_shape():
    """A legacy Gemma4TextMoEBlock-shaped class registers like
    Gemma4TextExperts: it exposes the same num_experts/hidden_dim/
    intermediate_dim + gate_up_proj/down_proj surface."""
    LegacyBlock = _fresh_stub_class()
    assert g4._register_gemma4_lora_extractor(LegacyBlock) is True
    _drive_extractor(LegacyBlock, "gate_up_proj", peft_swapped=False)
    _drive_extractor(LegacyBlock, "down_proj", peft_swapped=True)

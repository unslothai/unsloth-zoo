# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""merge_and_unload() re-quantizes the expert stack, so it overflows too.

Quantizing and dequantizing an oversized stack in slices is not enough on its
own: ParamWrapper.merge dequantizes, adds the adapter delta and re-quantizes,
and that re-quantize is a second single bitsandbytes call over the same element
count. It aborts the process at csrc/ops.cu line 74 exactly like the load-time
one, after a full fine-tune rather than at load.

The kernel aborts rather than raising, so a test cannot let it run. The
threshold is lowered instead and every quantize call is recorded, which pins
the property that matters: no single call is ever handed a stack bitsandbytes
cannot count.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

bnb = pytest.importorskip("bitsandbytes")
peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model  # noqa: E402

from unsloth_zoo.temporary_patches import moe_utils_bnb4bit as M  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NUM_EXPERTS = 4
TWO_INTER = 128
HIDDEN = 64
RANK = 2


class _ToyExperts(nn.Module):
    num_experts = NUM_EXPERTS

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.empty(NUM_EXPERTS, TWO_INTER, HIDDEN, dtype=torch.bfloat16).normal_()
        )

    def forward(self, x):
        return torch.einsum("bh,eih->bei", x, self.gate_up_proj)


class _ToyMoE(nn.Module):
    """PEFT refuses an nn.Parameter on the top-level module, and real GPT-OSS
    holds it at ...mlp.experts.gate_up_proj, so nest it one level."""

    num_experts = NUM_EXPERTS

    def __init__(self):
        super().__init__()
        self.experts = _ToyExperts()

    def forward(self, x):
        return self.experts(x)


def _peft_supports_target_parameters() -> bool:
    try:
        LoraConfig(r=1, target_parameters=["dummy"])
        return True
    except TypeError:
        return False
    except Exception:
        return True


requires_target_parameters = pytest.mark.skipif(
    not _peft_supports_target_parameters(), reason="PEFT < 0.18 lacks target_parameters"
)


def _live_param(peft_model):
    """The Params4bit as it stands now. PEFT replaces the attribute with a
    ParamWrapper module, so reading it off the original module returns the
    wrapper and hides every merge."""
    for _, module in peft_model.named_modules():
        if type(module).__name__ == "ParamWrapper":
            assert getattr(type(module).merge, "_unsloth_4bit_moe_patched", False), (
                "ParamWrapper.merge is not the patched one, so this test would "
                "exercise PEFT's merge instead of the sliced path"
            )
            return getattr(module.get_base_layer(), module.parameter_name)
    raise AssertionError("PEFT did not wrap the expert parameter")


@pytest.fixture
def quantize_spy(monkeypatch):
    """Record the element count of every bitsandbytes 4-bit quantize call."""
    seen = []
    real = bnb.functional.quantize_4bit

    def _spy(a, *args, **kwargs):
        seen.append(a.numel())
        return real(a, *args, **kwargs)

    monkeypatch.setattr(bnb.functional, "quantize_4bit", _spy)
    return seen


@requires_target_parameters
def test_merge_never_hands_bitsandbytes_an_uncountable_stack(monkeypatch, quantize_spy):
    torch.manual_seed(0)
    base = _ToyMoE().cuda()
    value = base.experts.gate_up_proj.data.clone()

    # One slice per two experts, so merge must split rather than issue one call.
    threshold = 2 * TWO_INTER * HIDDEN
    monkeypatch.setattr(M, "_BNB_MAX_QUANTIZE_NUMEL", threshold)

    param = M._make_expert_params4bit(
        value, requires_grad=False, blocksize=64, quant_type="nf4",
    )
    param._original_shape = value.shape
    base.experts.gate_up_proj = param

    M.patch_peft_param_wrapper_merge_4bit()

    model = get_peft_model(base, LoraConfig(
        r=RANK, lora_alpha=RANK * 2, lora_dropout=0.0, bias="none",
        target_parameters=["experts.gate_up_proj"],
    ))

    quantize_spy.clear()
    model.merge_adapter()

    assert quantize_spy, "merge did not re-quantize, so this test proves nothing"
    too_big = [n for n in quantize_spy if n >= threshold]
    assert not too_big, (
        f"merge issued {len(too_big)} quantize call(s) of {too_big} elements with the "
        f"threshold at {threshold}; bitsandbytes aborts the process on those."
    )

    merged = _live_param(model)
    assert tuple(merged._original_shape) == tuple(value.shape)
    assert tuple(merged.quant_state.shape) == tuple(value.shape)
    assert merged.data.numel() * 2 == value.numel()


@requires_target_parameters
def test_merged_weights_are_the_adapter_delta_added_to_the_base(monkeypatch, quantize_spy):
    """The sliced re-quantize has to be a re-quantize, not a discard: merging a
    zeroed adapter must leave the stack where it was."""
    torch.manual_seed(0)
    base = _ToyMoE().cuda()
    value = base.experts.gate_up_proj.data.clone()
    monkeypatch.setattr(M, "_BNB_MAX_QUANTIZE_NUMEL", 2 * TWO_INTER * HIDDEN)

    param = M._make_expert_params4bit(
        value, requires_grad=False, blocksize=64, quant_type="nf4",
    )
    param._original_shape = value.shape
    base.experts.gate_up_proj = param
    before = M._dequantize_bnb4bit_expert_weights(param, torch.bfloat16).clone()

    M.patch_peft_param_wrapper_merge_4bit()
    model = get_peft_model(base, LoraConfig(
        r=RANK, lora_alpha=RANK * 2, lora_dropout=0.0, bias="none",
        target_parameters=["experts.gate_up_proj"],
    ))
    # lora_B is zero-initialised, so the delta is exactly zero.
    model.merge_adapter()

    after = M._dequantize_bnb4bit_expert_weights(_live_param(model), torch.bfloat16)
    assert after.shape == before.shape
    assert torch.equal(after, before), (
        "a zero adapter delta changed the merged weights, so the sliced "
        "re-quantize is not round-tripping the stack"
    )


def test_small_stacks_still_take_the_single_call_merge_path(monkeypatch, quantize_spy):
    """Below the threshold nothing about the merge path changes, including the
    double quantization the sliced path has to turn off."""
    torch.manual_seed(0)
    value = torch.empty(NUM_EXPERTS, TWO_INTER, HIDDEN, dtype=torch.bfloat16, device="cuda").normal_()
    monkeypatch.setattr(M, "_BNB_MAX_QUANTIZE_NUMEL", 2 ** 31)
    param = M._make_expert_params4bit(
        value, requires_grad=False, blocksize=64, quant_type="nf4",
        compress_statistics=True,
    )
    assert param.quant_state.nested
    assert quantize_spy == [value.numel()], (
        f"expected exactly one whole-stack quantize call, got {quantize_spy}"
    )


@requires_target_parameters
@pytest.mark.parametrize("quant_storage", [torch.uint8, torch.bfloat16])
def test_merge_keeps_the_configured_quant_storage(monkeypatch, quantize_spy, quant_storage):
    """The sliced merge must not quietly downgrade storage to uint8 either."""
    torch.manual_seed(0)
    base = _ToyMoE().cuda()
    value = base.experts.gate_up_proj.data.clone()
    monkeypatch.setattr(M, "_BNB_MAX_QUANTIZE_NUMEL", 2 * TWO_INTER * HIDDEN)

    param = M._make_expert_params4bit(
        value, requires_grad=False, blocksize=64, quant_type="nf4",
        quant_storage=quant_storage,
    )
    param._original_shape = value.shape
    base.experts.gate_up_proj = param

    M.patch_peft_param_wrapper_merge_4bit()
    model = get_peft_model(base, LoraConfig(
        r=RANK, lora_alpha=RANK * 2, lora_dropout=0.0, bias="none",
        target_parameters=["experts.gate_up_proj"],
    ))
    model.merge_adapter()

    merged = _live_param(model)
    assert merged.data.dtype == quant_storage
    assert merged.quant_storage == quant_storage

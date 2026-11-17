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


"""No-grad matmul_ogs decode reads the base experts only, so it must step aside while expert LoRA is active."""
import types

import torch

from unsloth_zoo.temporary_patches import gpt_oss


def _lora_experts(merged = False, disabled = False):
    base = torch.nn.Module()
    wrapper = torch.nn.Module()
    wrapper.base_layer = base
    wrapper.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 1, bias = False)})
    wrapper.merged = merged
    wrapper.disable_adapters = disabled
    return wrapper


def _decode(monkeypatch, experts):
    unwrapped = []
    monkeypatch.setattr(gpt_oss, "_ogs_kernels", lambda: object())
    # Plain (non-MXFP4) base params make the decode return None right after the unwrap.
    monkeypatch.setattr(
        gpt_oss,
        "_unwrap_peft_experts",
        lambda e: unwrapped.append(e) or types.SimpleNamespace(gate_up_proj = None, down_proj = None),
    )
    layer = types.SimpleNamespace(experts = experts)
    with torch.no_grad():
        out = gpt_oss._mxfp4_ogs_decode(layer, torch.zeros(2, 4))
    return out, unwrapped


def test_active_expert_lora_skips_matmul_ogs(monkeypatch):
    out, unwrapped = _decode(monkeypatch, _lora_experts())
    assert out is None and unwrapped == []


def test_merged_or_disabled_lora_still_reaches_matmul_ogs(monkeypatch):
    for experts in (_lora_experts(merged = True), _lora_experts(disabled = True)):
        out, unwrapped = _decode(monkeypatch, experts)
        assert out is None and len(unwrapped) == 1

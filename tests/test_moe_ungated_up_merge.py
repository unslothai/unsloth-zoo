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

"""Ungated MoE experts (NemotronH experts.up_proj of I rows, no gate_proj) merge their up LoRA."""
import torch

from unsloth_zoo import saving_utils as SU
from unsloth_zoo.saving_utils import LoraStats, _merge_moe_gate_expert, _merge_moe_up_expert


def _peft_expert_lora_b(lora_B, expert_idx, num_experts):
    return lora_B.reshape(lora_B.shape[0], -1, num_experts)[:, :, expert_idx]


def _reset():
    for k in list(SU._MOE_MERGE_STATE):
        v = SU._MOE_MERGE_STATE[k]
        SU._MOE_MERGE_STATE[k] = type(v)() if not isinstance(v, int) else 0


def test_ungated_up_standard_layout_nemotronh_shapes():
    torch.manual_seed(0)
    E, r, I, H, alpha = 4, 8, 32, 16, 2.0
    W = torch.randn(I, H)
    A = torch.randn(E * r, H)
    B = torch.randn(I, E * r)
    _reset()
    for e in range(E):
        out = _merge_moe_up_expert(W.clone(), LoraStats(module=None, lora_A=A, lora_B=B, alpha=alpha),
                                   e, E, torch.float32)
        exp = W + alpha * (_peft_expert_lora_b(B, e, E) @ A[e * r:(e + 1) * r])
        torch.testing.assert_close(out.cpu(), exp, atol=1e-4, rtol=1e-4)
    assert SU._MOE_MERGE_STATE["applied"] == E
    assert SU._MOE_MERGE_STATE["attempted"] == E


def test_ungated_up_swapped_layout():
    torch.manual_seed(1)
    E, r, I, H, alpha = 4, 3, 8, 12, 4.0
    W = torch.randn(I, H)
    A = torch.randn(E * r, I)
    B = torch.randn(H, E * r)
    for e in range(E):
        out = _merge_moe_up_expert(W.clone(), LoraStats(module=None, lora_A=A, lora_B=B, alpha=alpha),
                                   e, E, torch.float32)
        exp = W + alpha * (_peft_expert_lora_b(B, e, E) @ A[e * r:(e + 1) * r]).T
        torch.testing.assert_close(out.cpu(), exp, atol=1e-4, rtol=1e-4)


def test_gate_role_never_takes_the_ungated_branch():
    E, r, I, H = 4, 2, 8, 12
    W = torch.randn(I, H)
    A, B = torch.randn(E * r, H), torch.randn(I, E * r)
    out = _merge_moe_gate_expert(W.clone(), LoraStats(module=None, lora_A=A, lora_B=B, alpha=1.0),
                                 0, E, torch.float32)
    assert torch.equal(out, W)


def test_fused_gate_up_unchanged():
    torch.manual_seed(2)
    E, r, I, H, alpha = 4, 3, 8, 12, 8.0
    W = torch.randn(I, H)
    A = torch.randn(E * r, H)
    B = torch.randn(2 * I, E * r)
    for e in range(E):
        out = _merge_moe_up_expert(W.clone(), LoraStats(module=None, lora_A=A, lora_B=B, alpha=alpha),
                                   e, E, torch.float32)
        exp = W + alpha * (_peft_expert_lora_b(B, e, E)[I:] @ A[e * r:(e + 1) * r])
        torch.testing.assert_close(out.cpu(), exp, atol=1e-4, rtol=1e-4)


def _merge_shard(tmp_path, parameter_name):
    from collections import defaultdict
    from safetensors import safe_open
    from safetensors.torch import save_file
    from unsloth_zoo.saving_utils import _merge_and_overwrite_lora

    torch.manual_seed(3)
    E, r, I, H = 4, 2, 32, 16
    pre = "backbone.layers.2.mixer.experts"
    up = {e: torch.randn(I, H) for e in range(E)}
    down = {e: torch.randn(H, I) for e in range(E)}
    shard = tmp_path / "model.safetensors"
    save_file({**{f"{pre}.{e}.up_proj.weight": up[e] for e in up},
               **{f"{pre}.{e}.down_proj.weight": down[e] for e in down}},
              str(shard), metadata={"format": "pt"})
    A, B = torch.randn(E * r, H), torch.randn(I, E * r)
    lora = defaultdict(lambda: LoraStats(None, None, None, 0))
    # A lone expert wrapper is keyed on `experts` (no .base_layer) whichever parameter it wraps.
    lora[pre] = LoraStats(None, A, B, 1.0, parameter_name=parameter_name)
    _merge_and_overwrite_lora(
        save_directory=str(tmp_path), filename="model.safetensors", lora_weights=lora,
        output_dtype=torch.float32, model_class_name="NemotronHForCausalLM",
    )
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        for e in range(E):
            exp_up = up[e] + _peft_expert_lora_b(B, e, E) @ A[e * r:(e + 1) * r]
            yield f.get_tensor(f"{pre}.{e}.up_proj.weight"), exp_up, f.get_tensor(f"{pre}.{e}.down_proj.weight"), down[e]


def test_up_only_expert_lora_lands_on_up_proj_not_down_proj(tmp_path):
    _reset()
    for got_up, exp_up, got_down, orig_down in _merge_shard(tmp_path, "up_proj"):
        torch.testing.assert_close(got_up, exp_up, atol=1e-4, rtol=1e-4)
        assert torch.equal(got_down, orig_down)

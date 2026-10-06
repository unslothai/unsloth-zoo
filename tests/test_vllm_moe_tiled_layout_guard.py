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
"""Shared MoE experts must never be vLLM's TRT-LLM tiled layout, even when the shape stays 3-D.

vLLM >= 0.31 converts to the TRT-LLM layout in place without changing the shape, so if the
override that keeps vLLM on Triton ever stops matching, only the class in use or the bytes
themselves can tell.
"""
import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from safetensors.torch import save_file

from unsloth_zoo.empty_model import extract_moe_layers, vllm_moe_expert_weights

E, I, H = 2, 64, 128
CFG = types.SimpleNamespace(num_experts = E, moe_intermediate_size = I, hidden_size = H)
PREFIX = "model.language_model.layers.0.mlp"


def tile_w13(w13):
    # Closed form of FlashInfer's BlockMajorK + shuffled w13 (scripts/trtllm_layout/layout_check.py)
    E, M, H = w13.shape; I = M // 2
    x = w13.view(E, 2, I // 16, 8, 2, H // 64, 64).flip(1)
    return x.permute(0, 5, 2, 4, 1, 3, 6).reshape(E, H // 64, M, 64)


def tile_w2(w2):
    E, H, I = w2.shape
    x = w2.view(E, H // 32, 8, 4, I // 64, 64)
    return x.permute(0, 4, 1, 3, 2, 5).reshape(E, I // 64, H, 64)


def _reference():
    g = torch.Generator().manual_seed(0)
    return (torch.randn(E, 2 * I, H, generator = g).bfloat16(),
            torch.randn(E, H, I, generator = g).bfloat16())


def _routed(w13, w2, backend = "TRITON", quant_method = None):
    routed = torch.nn.Module()
    routed.w13_weight = torch.nn.Parameter(w13, requires_grad = False)
    routed.w2_weight = torch.nn.Parameter(w2, requires_grad = False)
    if quant_method is None:
        quant_method = types.SimpleNamespace(unquantized_backend = types.SimpleNamespace(name = backend))
    routed.quant_method = quant_method
    return routed


def _checkpoint(tmp_path, w13, w2, key_prefix = PREFIX):
    save_file({
        f"{key_prefix}.experts.gate_up_proj": w13.contiguous(),
        f"{key_prefix}.experts.down_proj": w2.contiguous(),
        "mtp.layers.0.mlp.experts.gate_up_proj": torch.zeros_like(w13),
    }, str(tmp_path / "model.safetensors"))
    return str(tmp_path)


def _extract(routed):
    block = torch.nn.Module()
    block.experts = torch.nn.Module()
    block.experts.routed_experts = routed
    sd, qsd = {}, {}
    extract_moe_layers(block, PREFIX, sd, qsd, lambda *a, **k: None, config = CFG)
    return qsd


# vLLM 0.11.2 TrtLlmGenExperts; 0.29 / 0.31 TrtLlmBf16LoRAExperts, TrtLlmBf16ExpertsModular.
TrtLlmGenExperts = type("TrtLlmGenExperts", (), {})
TrtLlmBf16LoRAExperts = type("TrtLlmBf16LoRAExperts", (), {})
TrtLlmBf16ExpertsMonolithic = type("TrtLlmBf16ExpertsMonolithic", (), {})


@pytest.mark.parametrize("holder", ["moe_kernel", "experts_cls", "renamed_kernel", "old_quant_method"])
@pytest.mark.parametrize("cls", [TrtLlmGenExperts, TrtLlmBf16LoRAExperts, TrtLlmBf16ExpertsMonolithic])
def test_a_trtllm_experts_class_is_refused_even_without_a_backend_field(holder, cls):
    w13, w2 = _reference()
    # No unquantized_backend anywhere: only the class in use can reveal the TRT-LLM path.
    if holder == "moe_kernel":  # FusedMoEModularMethod under LoRA
        qm = types.SimpleNamespace(moe_kernel = types.SimpleNamespace(fused_experts = cls()))
    elif holder == "experts_cls":
        qm = types.SimpleNamespace(experts_cls = cls)
    elif holder == "renamed_kernel":
        qm = types.SimpleNamespace(kernel = types.SimpleNamespace(impl = cls()))
    else:
        qm = types.SimpleNamespace(old_quant_method = types.SimpleNamespace(experts_cls = cls))
    with pytest.raises(NotImplementedError, match = cls.__name__):
        vllm_moe_expert_weights(_routed(w13, w2, quant_method = qm), "layer 0", CFG)


def test_triton_kernel_classes_are_not_flagged():
    w13, w2 = _reference()
    TritonExperts = type("TritonExperts", (), {})
    qm = types.SimpleNamespace(
        unquantized_backend = types.SimpleNamespace(name = "TRITON"),
        experts_cls = TritonExperts,
        moe_kernel = types.SimpleNamespace(fused_experts = TritonExperts()),
    )
    got13, got2 = vllm_moe_expert_weights(_routed(w13, w2, quant_method = qm), "layer 0", CFG)
    assert got13.data_ptr() == w13.data_ptr() and got2.data_ptr() == w2.data_ptr()


def test_the_correct_layout_passes_the_content_check(tmp_path):
    w13, w2 = _reference()
    from unsloth_zoo.empty_model import verify_vllm_moe_experts_match_checkpoint
    qsd = _extract(_routed(w13.clone(), w2.clone()))
    assert verify_vllm_moe_experts_match_checkpoint(qsd, _checkpoint(tmp_path, w13, w2)) is True


@pytest.mark.parametrize("which", ["w13", "w2"])
def test_in_place_tiled_bytes_with_the_hf_shape_are_refused(tmp_path, which):
    w13, w2 = _reference()
    # vLLM >= 0.31: TRT-LLM bytes written back into the original 3-D tensor.
    v13 = tile_w13(w13).reshape(w13.shape).contiguous() if which == "w13" else w13.clone()
    v2 = tile_w2(w2).reshape(w2.shape).contiguous() if which == "w2" else w2.clone()
    assert v13.shape == w13.shape and v2.shape == w2.shape
    qsd = _extract(_routed(v13, v2))  # shape, dtype and backend checks all pass
    from unsloth_zoo.empty_model import verify_vllm_moe_experts_match_checkpoint
    with pytest.raises(RuntimeError, match = "do not match the checkpoint"):
        verify_vllm_moe_experts_match_checkpoint(qsd, _checkpoint(tmp_path, w13, w2))


def test_a_differently_prefixed_checkpoint_is_still_found(tmp_path):
    w13, w2 = _reference()
    from unsloth_zoo.empty_model import verify_vllm_moe_experts_match_checkpoint
    path = _checkpoint(tmp_path, w13, w2, key_prefix = "model.layers.0.mlp")
    assert verify_vllm_moe_experts_match_checkpoint(_extract(_routed(w13.clone(), w2.clone())), path) is True
    tiled = tile_w13(w13).reshape(w13.shape).contiguous()
    with pytest.raises(RuntimeError):
        verify_vllm_moe_experts_match_checkpoint(_extract(_routed(tiled, w2.clone())), path)


def test_a_missing_checkpoint_does_not_block_loading(tmp_path):
    w13, w2 = _reference()
    from unsloth_zoo.empty_model import verify_vllm_moe_experts_match_checkpoint
    qsd = _extract(_routed(w13, w2))
    assert verify_vllm_moe_experts_match_checkpoint(qsd, str(tmp_path / "missing")) is None
    assert verify_vllm_moe_experts_match_checkpoint(qsd, None) is None
    assert verify_vllm_moe_experts_match_checkpoint({}, str(tmp_path)) is None

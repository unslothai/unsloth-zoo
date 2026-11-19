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

"""Patched Qwen3-VL-MoE blocks must return router logits at index 1 (transformers 4.57.x); returning
only hidden states makes the aux loss read hidden states as logits."""
import inspect

import pytest
import torch

Q = pytest.importorskip("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe")
if not hasattr(Q, "Qwen3VLMoeTextExperts"):
    pytest.skip("patched sparse block path needs Qwen3VLMoeTextExperts", allow_module_level = True)

import unsloth_zoo.temporary_patches.qwen3_vl_moe as V


def _tiny_model(hidden_size):
    from transformers import Qwen3VLMoeConfig

    config = Qwen3VLMoeConfig(
        text_config = dict(
            vocab_size = 128, hidden_size = hidden_size, intermediate_size = 32, moe_intermediate_size = 16,
            num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16,
            num_experts = 16, num_experts_per_tok = 4, decoder_sparse_step = 1, mlp_only_layers = [],
            router_aux_loss_coef = 0.001, rope_scaling = {"rope_type": "default", "mrope_section": [2, 3, 3]},
        ),
        vision_config = dict(
            depth = 1, hidden_size = 16, intermediate_size = 16, num_heads = 2, out_hidden_size = hidden_size,
            deepstack_visual_indexes = [0],
        ),
    )
    torch.manual_seed(0)
    return Q.Qwen3VLMoeForConditionalGeneration(config).eval()


def _saved_forwards():
    return {
        (Q.Qwen3VLMoeTextExperts, "forward"): Q.Qwen3VLMoeTextExperts.__dict__["forward"],
        (Q.Qwen3VLMoeTextExperts, "__init__"): Q.Qwen3VLMoeTextExperts.__dict__["__init__"],
        (Q.Qwen3VLMoeTextSparseMoeBlock, "forward"): Q.Qwen3VLMoeTextSparseMoeBlock.__dict__["forward"],
        (Q.Qwen3VLMoeModel, "forward"): Q.Qwen3VLMoeModel.__dict__["forward"],
        (Q.Qwen3VLMoeForConditionalGeneration, "forward"): Q.Qwen3VLMoeForConditionalGeneration.__dict__["forward"],
    }


def _run_and_check(model, top_level_router_logits = True):
    num_experts = model.config.text_config.num_experts
    gate_logits = []
    def hook(module, args, output):
        gate_logits.append((output[0] if isinstance(output, tuple) else output).detach())
    handles = [layer.mlp.gate.register_forward_hook(hook) for layer in model.model.language_model.layers]
    try:
        ids = torch.randint(0, 128, (2, 7))
        with torch.no_grad():
            out = model(input_ids = ids, labels = ids, output_router_logits = True)
    finally:
        for h in handles:
            h.remove()

    assert len(gate_logits) == 2
    if top_level_router_logits:
        assert len(out.router_logits) == 2
        for got, want in zip(out.router_logits, gate_logits):
            assert got.shape == (14, num_experts)
            torch.testing.assert_close(got, want)
    assert torch.isfinite(out.aux_loss)
    expected_aux = Q.load_balancing_loss_func(tuple(gate_logits), num_experts, 4, None)
    torch.testing.assert_close(out.aux_loss, expected_aux)


# hidden_size > num_experts crashed one_hot ("Class values must be smaller than num_classes"), as on the
# real 30B / 235B configs (2048 / 4096 hidden, 128 experts); hidden_size < num_experts gave a shape error.
@pytest.mark.parametrize("hidden_size", [64, 8])
def test_patched_block_returns_gate_router_logits(hidden_size):
    saved = _saved_forwards()
    try:
        V.patch_qwen3_vl_moe()
        assert Q.Qwen3VLMoeTextSparseMoeBlock.forward is not saved[(Q.Qwen3VLMoeTextSparseMoeBlock, "forward")]
        _run_and_check(_tiny_model(hidden_size))
    finally:
        for (cls, name), fn in saved.items():
            setattr(cls, name, fn)


def test_router_logits_survive_stripped_output_capture_decorator():
    # The Unsloth compiler rewrites Qwen3VLMoeModel / Qwen3VLMoeForConditionalGeneration and strips
    # @check_model_inputs. On 4.57.x Qwen3VLMoeModel.forward then drops the language model's router_logits.
    saved = _saved_forwards()
    try:
        for cls in (Q.Qwen3VLMoeModel, Q.Qwen3VLMoeForConditionalGeneration):
            cls.forward = inspect.unwrap(cls.__dict__["forward"])
        V.patch_qwen3_vl_moe()
        # The stripped top-level forward never attaches router_logits; TRL only reads aux_loss.
        _run_and_check(_tiny_model(64), top_level_router_logits = False)
    finally:
        for (cls, name), fn in saved.items():
            setattr(cls, name, fn)

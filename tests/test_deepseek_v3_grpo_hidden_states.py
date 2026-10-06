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


"""The patched DeepseekV3ForCausalLM.forward hands GRPO the hidden states (no router_logits on these outputs)."""
import pytest
import torch

# Runs on every transformers with DeepseekV3: the wrapper must not depend on DeepseekV3NaiveMoe (absent on 4.x and >= 5.13).
modeling = pytest.importorskip("transformers.models.deepseek_v3.modeling_deepseek_v3")


def test_return_hidden_states_gives_hidden_states(monkeypatch):
    from unsloth_zoo.temporary_patches.deepseek_v3_moe import patch_deepseek_v3
    patch_deepseek_v3()
    config = modeling.DeepseekV3Config(
        vocab_size = 64, hidden_size = 16, intermediate_size = 32, moe_intermediate_size = 8,
        num_hidden_layers = 2, num_attention_heads = 2, num_key_value_heads = 2, first_k_dense_replace = 0,
        n_routed_experts = 4, num_experts_per_tok = 2, n_group = 1, topk_group = 1, n_shared_experts = 1,
        q_lora_rank = 8, kv_lora_rank = 8, qk_rope_head_dim = 4, qk_nope_head_dim = 4, v_head_dim = 4,
    )
    torch.manual_seed(0)
    model = modeling.DeepseekV3ForCausalLM(config).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "1")
    with torch.no_grad():
        out = model(input_ids = ids)
        hidden = model.model(input_ids = ids).last_hidden_state
    assert out.logits.shape == (1, 5, config.hidden_size)
    torch.testing.assert_close(out.logits, hidden)

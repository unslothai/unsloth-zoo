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

"""A bf16 DeepSeek-V4 must run a plain forward; norms match DeepSeek's reference `(weight * x).to(dtype)`."""

import pytest
import torch

modeling = pytest.importorskip("transformers.models.deepseek_v4.modeling_deepseek_v4")

import unsloth_zoo.temporary_patches  # noqa: F401  (populates TEMPORARY_PATCHES)
from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES


def _apply():
    for fn in TEMPORARY_PATCHES:
        if "deepseek_v4" in fn.__name__:
            fn()


def _tiny_config():
    from transformers import DeepseekV4Config

    return DeepseekV4Config(
        vocab_size = 256, hidden_size = 64, num_hidden_layers = 4, num_attention_heads = 4,
        num_key_value_heads = 1, head_dim = 32, qk_rope_head_dim = 8, q_lora_rank = 32, o_lora_rank = 16,
        o_groups = 2, n_routed_experts = 4, n_shared_experts = 1, num_experts_per_tok = 2,
        moe_intermediate_size = 32, index_n_heads = 2, index_head_dim = 16, index_topk = 4,
        sliding_window = 8, compress_ratios = [0, 128, 4, 128], num_hash_layers = 1,
        max_position_embeddings = 256,
    )


def test_bf16_forward_without_autocast():
    _apply()
    torch.manual_seed(0)
    model = modeling.DeepseekV4ForCausalLM(_tiny_config()).to(torch.bfloat16).eval()
    for name, module in model.named_modules():
        if isinstance(module, modeling.DeepseekV4RMSNorm):
            module.float()
    ids = torch.randint(0, 256, (1, 24))
    with torch.no_grad():
        out = model(input_ids = ids, labels = ids, use_cache = False)
    assert torch.isfinite(out.loss)
    hidden = model.model(input_ids = ids, use_cache = False).last_hidden_state
    assert hidden.dtype == torch.bfloat16


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_matches_deepseek_reference_norm(dtype):
    _apply()
    torch.manual_seed(0)
    norm = modeling.DeepseekV4RMSNorm(64, eps = 1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(64))
    x = torch.randn(3, 5, 64).to(dtype)
    got = norm(x)
    # DeepSeek inference/model.py RMSNorm.forward
    xf = x.float()
    want = (norm.weight * (xf * torch.rsqrt(xf.square().mean(-1, keepdim = True) + 1e-6))).to(dtype)
    assert got.dtype == dtype
    assert torch.equal(got, want)


def test_idempotent():
    _apply()
    first = modeling.DeepseekV4RMSNorm.forward
    _apply()
    assert modeling.DeepseekV4RMSNorm.forward is first

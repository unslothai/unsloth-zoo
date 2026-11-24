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

"""generate with a StaticCache on hybrid_sliding models past the sliding window, on a tiny Inkling 2 Mini CPU fake."""

import pytest
import torch

modeling = pytest.importorskip("transformers.models.inkling2_loop.modeling_inkling2_loop")
from transformers.cache_utils import DynamicCache, StaticCache  # noqa: E402
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig  # noqa: E402
import transformers.masking_utils as masking_utils  # noqa: E402

from unsloth_zoo.temporary_patches import inkling2 as ink  # noqa: E402


@pytest.fixture(scope = "module")
def tiny():
    torch.set_num_threads(1)
    config = Inkling2LoopTextConfig(
        vocab_size = 64, unpadded_vocab_size = 64, hidden_size = 32, num_hidden_layers = 6, num_attention_heads = 2,
        num_key_value_heads = 1, head_dim = 16, swa_num_attention_heads = 2, swa_num_key_value_heads = 1,
        swa_head_dim = 16, sliding_window_size = 4, rel_extent = 16, d_rel = 4, moe_intermediate_size = 16,
        n_routed_experts = 4, num_experts_per_tok = 2, n_layers_per_attn_res_block = 2, loop_layer_index_start = 2,
        loop_layer_index_end = 4, loop_num_passes = 3, loop_gate_rank = 4, eos_token_id = None, pad_token_id = 0,
        layer_types = ["hybrid_sliding", "hybrid"] * 3,
    )
    torch.manual_seed(0)
    model = modeling.Inkling2LoopForCausalLM(config).eval()
    input_ids = torch.randint(1, 64, (1, 12), generator = torch.Generator().manual_seed(1))  # 12 > window 4
    return config, model, input_ids


def _generate(model, input_ids, cache):
    with torch.no_grad():
        return model.generate(input_ids, max_new_tokens = 4, do_sample = False, cache = cache)


def test_static_cache_past_window_matches_dynamic(tiny, monkeypatch):
    config, model, input_ids = tiny
    entry = masking_utils.LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING["hybrid_sliding"]
    monkeypatch.setitem(entry, "sliding_attention", masking_utils.create_causal_mask)  # the transformers 5.24 table
    dynamic = _generate(model, input_ids, DynamicCache(config = config))
    with pytest.raises(RuntimeError, match = "size of tensor"):
        _generate(model, input_ids, StaticCache(config = config, max_cache_len = 20))
    ink.patch_hybrid_sliding_generate_masks()
    assert entry["sliding_attention"] is masking_utils.create_sliding_window_causal_mask
    static = _generate(model, input_ids, StaticCache(config = config, max_cache_len = 20))
    assert torch.equal(static, dynamic)


def test_noop_when_transformers_maps_it(monkeypatch):
    entry = masking_utils.LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING["hybrid_sliding"]
    # A transformers that already maps the sliding side (or maps it to anything but the full causal mask) is untouched.
    sentinel = lambda **kwargs: None  # noqa: E731
    monkeypatch.setitem(entry, "sliding_attention", sentinel)
    ink.patch_hybrid_sliding_generate_masks()
    assert entry["sliding_attention"] is sentinel
    monkeypatch.setitem(entry, "sliding_attention", masking_utils.create_sliding_window_causal_mask)
    ink.patch_hybrid_sliding_generate_masks()
    assert entry["sliding_attention"] is masking_utils.create_sliding_window_causal_mask

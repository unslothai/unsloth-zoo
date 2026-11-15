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


"""CsmDepthDecoderModel decode steps over a StaticCache capture as one CUDA graph and replay at the next codebook
position: the positions are computed on the device (transformers 5.x reads the cache length on the host with
torch.arange(past_seen_tokens, ...) and branches on position_ids[0] == 0). CUDA only."""
import inspect
import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
modeling_csm = pytest.importorskip("transformers.models.csm.modeling_csm")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")


def _depth():
    from transformers import CsmDepthDecoderConfig, StaticCache

    config = CsmDepthDecoderConfig(
        hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2, num_attention_heads = 2,
        num_key_value_heads = 1, head_dim = 16, num_codebooks = 8, vocab_size = 64, backbone_hidden_size = 32,
        max_position_embeddings = 9,
    )
    torch.manual_seed(0)
    model = modeling_csm.CsmDepthDecoderForCausalLM._from_config(config, dtype = torch.float32).cuda().eval()
    return model, StaticCache(config = config, max_cache_len = 9)


def _state(cache):
    return [(t, t.clone()) for layer in cache.layers for t in (layer.keys, layer.values, layer.cumulative_length)]


def _restore(state):
    for live, saved in state:
        live.copy_(saved)


def test_depth_decode_step_replays_at_the_next_position():
    try:
        from unsloth_zoo.temporary_patches.misc import patch_CsmDepthDecoderModel_forward
    except ImportError:
        patch_CsmDepthDecoderModel_forward = None
    if patch_CsmDepthDecoderModel_forward is not None:
        patch_CsmDepthDecoderModel_forward()
    model, cache = _depth()
    if "cache_position" in inspect.signature(modeling_csm.CsmDepthDecoderModel.forward).parameters:
        pytest.skip("transformers 4.x passes cache_position")
    dev = "cuda"
    with torch.no_grad():
        prefill = torch.tensor([[0, 5]], device = dev)
        model(input_ids = prefill, backbone_last_hidden_state = torch.randn(1, 32, device = dev),
              past_key_values = cache, use_cache = True, logits_to_keep = 1)
        token = torch.tensor([[7]], device = dev)
        # generate()'s decode step: one token, its own logits (logits_to_keep = 1)
        step = lambda: model(input_ids = token, past_key_values = cache, use_cache = True, logits_to_keep = 1).logits  # noqa: E731

        start = _state(cache)
        eager_1 = step().clone()
        eager_2 = step().clone()  # the next codebook position, from the advanced device counter
        _restore(start)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            step()
        torch.cuda.current_stream().wait_stream(side)
        _restore(start)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = step()
        _restore(start)
        graph.replay()
        first = out.clone()
        graph.replay()
        torch.cuda.synchronize()
    assert torch.equal(first, eager_1)
    assert torch.equal(out, eager_2)
    assert not torch.equal(eager_1, eager_2)

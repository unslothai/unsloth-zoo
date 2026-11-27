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

import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
modeling_csm = pytest.importorskip("transformers.models.csm.modeling_csm")


def _tiny_csm():
    from transformers import CsmConfig

    depth = dict(
        hidden_size = 32, intermediate_size = 64, num_hidden_layers = 1,
        num_attention_heads = 2, num_key_value_heads = 1, head_dim = 16,
        num_codebooks = 4, vocab_size = 64, backbone_hidden_size = 32,
        max_position_embeddings = 8,
    )
    codec = dict(
        model_type = "mimi", hidden_size = 32, upsample_groups = 32, num_filters = 4, num_hidden_layers = 1,
        num_attention_heads = 2, num_key_value_heads = 2, head_dim = 16, intermediate_size = 64,
        codebook_dim = 16, vector_quantization_hidden_dimension = 16, codebook_size = 64, num_quantizers = 4, num_semantic_quantizers = 1,
    )
    config = CsmConfig(
        hidden_size = 32, intermediate_size = 64, num_hidden_layers = 1,
        num_attention_heads = 2, num_key_value_heads = 1, head_dim = 16,
        num_codebooks = 4, vocab_size = 64, text_vocab_size = 128,
        depth_decoder_config = depth, codec_config = codec,
        audio_token_id = 100, audio_eos_token_id = 101,
    )
    torch.manual_seed(0)
    return modeling_csm.CsmForConditionalGeneration._from_config(config, dtype = torch.float32).eval()


def test_merge_encodes_whole_frames():
    # kyutai's encoder (CSM's labels) zero-pads clips to whole frames (unsloth#19438).
    from unsloth_zoo.temporary_patches.misc import patch_CsmForConditionalGeneration_merge

    patch_CsmForConditionalGeneration_merge()
    model = _tiny_csm()
    frame = model.codec_model.config.frame_size
    lengths = [2 * frame + 333, frame - 7, 3 * frame]
    frames = [-(-n // frame) for n in lengths]

    audio_id, eos_id = model.config.audio_token_id, model.config.audio_eos_token_id
    ids, cutoffs = [5], [0]
    for f in frames:
        ids += [audio_id] * f + [eos_id, 6]
        cutoffs.append(cutoffs[-1] + lengths[len(cutoffs) - 1])
    input_ids = torch.tensor([ids])
    torch.manual_seed(1)
    input_values = torch.randn(1, 1, cutoffs[-1] + 50) * 0.1
    input_values_cutoffs = torch.tensor([cutoffs[1:]])

    seen = []
    encode = model.codec_model.encode
    def recording_encode(x, *args, **kwargs):
        seen.append(x.detach().clone())
        return encode(x, *args, **kwargs)
    model.codec_model.encode = recording_encode

    with torch.no_grad():
        out = model._merge_input_ids_with_input_values(
            input_ids, input_values, input_values_cutoffs, torch.full_like(input_ids, -100),
        )

    assert [x.shape[-1] for x in seen] == [f * frame for f in frames]
    for x, n, start in zip(seen, lengths, cutoffs):
        assert torch.equal(x[..., :n], input_values[0, :, start:start + n].unsqueeze(0))
        assert not x[..., n:].any()

    labels = out["labels"][0]
    audio_rows = (input_ids[0] == audio_id).nonzero().squeeze(-1)
    expected = torch.cat([
        encode(torch.nn.functional.pad(input_values[..., s:s + n], (0, -n % frame))).audio_codes[0].T
        for n, s in zip(lengths, cutoffs)
    ])
    assert torch.equal(labels[audio_rows], expected)

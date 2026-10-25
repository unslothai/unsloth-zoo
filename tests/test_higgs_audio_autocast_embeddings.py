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

"""Higgs Audio v2 under CUDA autocast: the summed codebook embeddings must keep the embedding dtype."""

import pytest

torch = pytest.importorskip("torch")
modeling = pytest.importorskip("transformers.models.higgs_audio_v2.modeling_higgs_audio_v2")

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA autocast runs sum() in float32")


def _embeddings(device):
    from transformers import HiggsAudioV2Config

    config = HiggsAudioV2Config(
        hidden_size = 16, num_attention_heads = 2, num_key_value_heads = 2, num_codebooks = 4, codebook_size = 8
    )
    return modeling.HiggsAudioV2Embeddings(config).to(device = device, dtype = torch.bfloat16)


def test_patch_is_idempotent():
    from unsloth_zoo.temporary_patches.misc import patch_HiggsAudioV2Embeddings_autocast_dtype

    patch_HiggsAudioV2Embeddings_autocast_dtype()
    patched = modeling.HiggsAudioV2Embeddings.forward
    patch_HiggsAudioV2Embeddings_autocast_dtype()
    assert modeling.HiggsAudioV2Embeddings.forward is patched
    assert patched._unsloth_patched


@needs_cuda
def test_autocast_sum_is_float32_without_the_patch():
    """Without this the next test would prove nothing on a torch whose autocast keeps sum() in bfloat16."""
    emb = _embeddings("cuda")
    ids = torch.randint(0, 8, (2, 5, 4), device = "cuda")
    with torch.autocast("cuda", dtype = torch.bfloat16):
        assert emb.embed_audio_tokens(ids).sum(dim = -2).dtype == torch.float32


@needs_cuda
def test_patched_embeddings_keep_the_weight_dtype():
    from unsloth_zoo.temporary_patches.misc import patch_HiggsAudioV2Embeddings_autocast_dtype

    patch_HiggsAudioV2Embeddings_autocast_dtype()
    emb = _embeddings("cuda")
    ids = torch.randint(0, 8, (2, 5, 4), device = "cuda")
    with torch.autocast("cuda", dtype = torch.bfloat16):
        out = emb(ids)
    assert out.dtype == torch.bfloat16
    reference = emb.embed_audio_tokens(ids + emb.audio_tokens_offsets).float().sum(dim = -2)
    torch.testing.assert_close(out.float(), reference, rtol = 2e-2, atol = 2e-2)

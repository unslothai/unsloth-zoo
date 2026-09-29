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
    config = CsmConfig(
        hidden_size = 32, intermediate_size = 64, num_hidden_layers = 1,
        num_attention_heads = 2, num_key_value_heads = 1, head_dim = 16,
        num_codebooks = 4, vocab_size = 64, text_vocab_size = 128,
        depth_decoder_config = depth,
    )
    torch.manual_seed(0)
    return modeling_csm.CsmForConditionalGeneration._from_config(config, dtype = torch.float32).train()


def _patched():
    from unsloth_zoo.temporary_patches.misc import patch_CsmForConditionalGeneration_forward

    patch_CsmForConditionalGeneration_forward()
    assert "unsloth_zoo" in modeling_csm.CsmForConditionalGeneration.forward.__module__


def test_no_depth_decoder_frames_trains_backbone_only():
    # CsmProcessor(depth_decoder_labels_ratio=0.0) masks codebooks 1: on every frame.
    _patched()
    model = _tiny_csm()
    input_ids = torch.randint(0, 60, (2, 5, 4))
    labels = input_ids.clone()
    labels[:, :, 1:] = -100

    out = model(input_ids = input_ids, labels = labels)

    assert torch.isfinite(out.loss)
    assert out.depth_decoder_loss.item() == 0.0
    torch.testing.assert_close(out.loss, out.backbone_loss)
    out.loss.backward()


def test_depth_decoder_frames_still_train():
    _patched()
    model = _tiny_csm()
    input_ids = torch.randint(0, 60, (2, 5, 4))
    labels = input_ids.clone()
    labels[0, :, 1:] = -100

    out = model(input_ids = input_ids, labels = labels)

    assert out.depth_decoder_loss.item() > 0
    torch.testing.assert_close(out.loss, out.backbone_loss + out.depth_decoder_loss)

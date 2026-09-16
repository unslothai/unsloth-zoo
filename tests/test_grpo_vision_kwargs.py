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

"""The GRPO multimodal key tuple and the chunker both logprob paths share.

unslothai/unsloth#6960: a hard coded list of four keys dropped spatial_shapes, num_tiles
and image_position_ids, and dropped pixel_values entirely for a model with no
image_grid_thw, silently recomputing the reference logprobs from the text alone.
"""

import inspect

import torch

from unsloth_zoo.rl_replacements import (
    GRPO_VISION_KEYS,
    grpo_accumulated_loss,
    grpo_get_vision_inputs,
    grpo_vision_chunks,
)


def test_key_tuple_covers_the_trl_multimodal_kwargs():
    for key in (
        "pixel_values",
        "image_grid_thw",
        "pixel_attention_mask",
        "image_sizes",
        "spatial_shapes",
        "num_tiles",
        "image_position_ids",
        "num_images",
        "token_type_ids",
        "mm_token_type_ids",
    ):
        assert key in GRPO_VISION_KEYS, key


def test_get_vision_inputs_reads_every_key_and_nothing_else():
    source = {"pixel_values": 1, "spatial_shapes": 2, "advantages": 3}
    collected = grpo_get_vision_inputs(source)
    assert set(collected) == set(GRPO_VISION_KEYS)
    assert collected["pixel_values"] == 1
    assert collected["spatial_shapes"] == 2
    assert "advantages" not in collected


def test_lfm2vl_tiles_are_sliced_by_num_tiles():
    num_tiles = [3, 1, 2, 4]
    total_tiles = sum(num_tiles)
    vision = {
        "pixel_values": torch.arange(total_tiles).reshape(total_tiles, 1).float(),
        "pixel_attention_mask": torch.arange(total_tiles).reshape(total_tiles, 1),
        "spatial_shapes": torch.arange(2 * total_tiles).reshape(total_tiles, 2),
        "num_tiles": num_tiles,
        "num_images": [1, 1, 1, 1],
    }
    chunks = grpo_vision_chunks(vision, total_samples = 4, batch_size = 2)
    assert len(chunks) == 2
    # TRL's own arithmetic: cum_tiles = [0, 3, 4, 6, 10].
    assert chunks[0]["pixel_values"].shape[0] == 4
    assert chunks[1]["pixel_values"].shape[0] == 6
    assert torch.equal(chunks[0]["spatial_shapes"], vision["spatial_shapes"][0:4])
    assert torch.equal(chunks[1]["spatial_shapes"], vision["spatial_shapes"][4:10])
    assert torch.equal(chunks[1]["pixel_attention_mask"], vision["pixel_attention_mask"][4:10])
    for chunk in chunks:
        assert "image_grid_thw" not in chunk


def test_gemma4_image_position_ids_are_sliced_by_image():
    num_images = [2, 1, 1]
    total_images = sum(num_images)
    vision = {
        "pixel_values": torch.arange(total_images).reshape(total_images, 1).float(),
        "image_position_ids": torch.arange(3 * total_images).reshape(total_images, 3),
        "num_images": num_images,
    }
    chunks = grpo_vision_chunks(vision, total_samples = 3, batch_size = 2)
    assert chunks[0]["pixel_values"].shape[0] == 3
    assert chunks[1]["pixel_values"].shape[0] == 1
    assert torch.equal(chunks[0]["image_position_ids"], vision["image_position_ids"][0:3])
    assert torch.equal(chunks[1]["image_position_ids"], vision["image_position_ids"][3:4])


def test_internvl_num_tiles_alone_slices_pixel_values_by_tile():
    num_tiles = [5, 2]
    vision = {
        "pixel_values": torch.arange(7).reshape(7, 1).float(),
        "num_tiles": num_tiles,
        "num_images": [1, 1],
    }
    chunks = grpo_vision_chunks(vision, total_samples = 2, batch_size = 1)
    assert chunks[0]["pixel_values"].shape[0] == 5
    assert chunks[1]["pixel_values"].shape[0] == 2


def test_pixel_values_are_never_dropped_for_a_model_without_image_grid_thw():
    num_images = [1, 2]
    vision = {
        "pixel_values": torch.arange(3).reshape(3, 1, 1, 1).float(),
        "num_images": num_images,
    }
    chunks = grpo_vision_chunks(vision, total_samples = 2, batch_size = 1)
    assert chunks[0]["pixel_values"].shape[0] == 1
    assert chunks[1]["pixel_values"].shape[0] == 2
    for chunk in chunks:
        assert chunk["pixel_values"] is not None
        assert chunk["pixel_values"].numel() > 0


def test_one_pixel_values_row_per_sample_stays_on_the_sample_axis():
    vision = {"pixel_values": torch.arange(4).reshape(4, 1).float()}
    chunks = grpo_vision_chunks(vision, total_samples = 4, batch_size = 2)
    assert torch.equal(chunks[0]["pixel_values"], vision["pixel_values"][0:2])
    assert torch.equal(chunks[1]["pixel_values"], vision["pixel_values"][2:4])


def test_text_only_rows_produce_empty_chunks():
    chunks = grpo_vision_chunks({}, total_samples = 4, batch_size = 2)
    assert chunks == [{}, {}]


def test_count_lists_that_are_not_per_sample_are_ignored():
    vision = {
        "pixel_values": torch.arange(4).reshape(4, 1).float(),
        "num_tiles": [1, 1, 1, 1, 1, 1],
    }
    chunks = grpo_vision_chunks(vision, total_samples = 4, batch_size = 4)
    assert chunks[0]["pixel_values"].shape[0] == 4


def test_token_type_ids_stay_on_the_sample_axis():
    vision = {
        "pixel_values": torch.zeros(4, 1),
        "token_type_ids": torch.arange(8).reshape(4, 2),
        "mm_token_type_ids": torch.arange(8).reshape(4, 2),
    }
    chunks = grpo_vision_chunks(vision, total_samples = 4, batch_size = 2)
    assert chunks[0]["token_type_ids"].shape[0] == 2
    assert chunks[1]["mm_token_type_ids"].shape[0] == 2


def test_gradient_pass_forwards_the_whole_chunk():
    source = inspect.getsource(grpo_accumulated_loss)
    assert "grpo_vision_chunks" in source
    assert "**vision_chunk" in source
    assert "image_grid_thw = image_grid_thw_chunk" not in source
    assert "pixel_values = pixel_values_chunk" not in source

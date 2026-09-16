"""The GRPO multimodal key tuple and the chunker both logprob paths share.

Regression cover for unslothai/unsloth#6960. TRL's GRPO trainer forwards every key the
processor produced into its per-token logprob helper, including `spatial_shapes`
(LFM2-VL), `num_tiles` (LFM2-VL and InternVL) and `image_position_ids` (Gemma 4). The
Unsloth replacements used to read a hard coded list of four keys and to drop
`pixel_values` outright for any model that does not emit `image_grid_thw`, which silently
recomputed the reference logprobs from the text alone.
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
    """LFM2-VL: pixel_values, pixel_attention_mask and spatial_shapes are tile indexed."""
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
    # Nothing tile indexed may be sliced on the sample axis.
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
    """The bug behind #6960: no image_grid_thw used to mean no images at all."""
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
    """A num_tiles of the wrong length must not become a cumulative sample index."""
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
    """grpo_accumulated_loss must splat the chunk, not name four keys by hand."""
    source = inspect.getsource(grpo_accumulated_loss)
    assert "grpo_vision_chunks" in source
    assert "**vision_chunk" in source
    assert "image_grid_thw = image_grid_thw_chunk" not in source
    assert "pixel_values = pixel_values_chunk" not in source

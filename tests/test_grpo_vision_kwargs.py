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


def test_released_unsloth_can_still_see_num_images_in_this_function():
    """unsloth 2026.9.4 on PyPI gates multi-image GRPO on a source-text probe:

        _supports_num_images = "num_images" in inspect.signature(grpo_accumulated_loss).parameters
        if not _supports_num_images:
            _supports_num_images = "num_images" in inspect.getsource(grpo_accumulated_loss)
        if not _supports_num_images:
            raise RuntimeError("Multi-image GRPO requires an unsloth_zoo build ... upgrade unsloth_zoo")

    The two packages ship independently, so a zoo that moves num_images handling out of this
    function tells every released-unsloth user with two images in a sample to upgrade the zoo
    they just upgraded. This runs that probe verbatim rather than describing it.
    """
    from unsloth_zoo.rl_replacements import grpo_accumulated_loss

    source = inspect.getsource(grpo_accumulated_loss)
    supports = "num_images" in inspect.signature(grpo_accumulated_loss).parameters
    if not supports:
        supports = "num_images" in source
    assert supports, (
        "released unsloth would raise 'Please upgrade unsloth_zoo' for a multi-image sample: "
        "grpo_accumulated_loss no longer mentions num_images"
    )

    # The probe above is satisfied by a COMMENT mentioning num_images, which the periodic
    # comment-reduction pass is entitled to delete. So pin the binding itself: parsed, not
    # grepped, so the guarantee survives any rewording of the text around it.
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(source))
    bound = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "num_images" in bound, (
        "num_images survives only in a comment; a comment-reduction pass would delete it and "
        "released unsloth would start refusing multi-image GRPO again"
    )


# The gate that keeps the two logprob passes on the same inputs when the installed unsloth
# still carries its own hard coded no-grad key list. See grpo_companion_vision_keys.

import sys
import textwrap
import types

import pytest

import unsloth_zoo.rl_replacements as _rl


_RELEASED_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "pixel_attention_mask",
    "image_sizes",
    "num_images",
    "token_type_ids",
    "mm_token_type_ids",
)


def _install_companion(monkeypatch, tmp_path, body, name = "companion_rl"):
    """Put a module named like unsloth's under sys.modules with real, readable source.

    inspect.getsource needs a file, so the patcher cannot be built with exec or a lambda.
    """
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(body))
    module = types.ModuleType(name)
    module.__file__ = str(path)
    code = compile(path.read_text(), str(path), "exec")
    exec(code, module.__dict__)
    monkeypatch.setitem(sys.modules, "unsloth.models.rl_replacements", module)
    monkeypatch.setattr(_rl, "_GRPO_COMPANION_VISION_KEYS", None)
    return module


@pytest.fixture(autouse = True)
def _reset_companion_memo(monkeypatch):
    monkeypatch.setattr(_rl, "_GRPO_COMPANION_VISION_KEYS", None)
    monkeypatch.delitem(sys.modules, "unsloth.models.rl_replacements", raising = False)


def test_an_unsloth_with_a_hard_coded_no_grad_list_holds_the_gradient_pass_to_it(
    monkeypatch, tmp_path
):
    _install_companion(
        monkeypatch,
        tmp_path,
        '''
        def grpo_trainer__get_per_token_logps_and_entropies(function_name, function):
            def _get_per_token_logps_and_entropies(self, model, **kwargs):
                pixel_values = kwargs.get("pixel_values", None)
                image_grid_thw = kwargs.get("image_grid_thw", None)
                pixel_attention_mask = kwargs.get("pixel_attention_mask", None)
                image_sizes = kwargs.get("image_sizes", None)
                num_images = kwargs.get("num_images", None)
                token_type_ids = kwargs.get("token_type_ids", None)
                mm_token_type_ids = kwargs.get("mm_token_type_ids", None)
                return pixel_values, image_grid_thw, pixel_attention_mask, image_sizes, \\
                    num_images, token_type_ids, mm_token_type_ids
            return _get_per_token_logps_and_entropies
        ''',
    )
    assert _rl.grpo_companion_vision_keys() == _RELEASED_KEYS

    source = dict.fromkeys(_rl.GRPO_VISION_KEYS, 1)
    shared = _rl.grpo_shared_vision_inputs(source)
    assert set(shared) == set(_RELEASED_KEYS)
    for dropped in ("spatial_shapes", "num_tiles", "image_position_ids"):
        assert dropped not in shared


def test_an_unsloth_that_shares_the_helper_lifts_the_restriction(monkeypatch, tmp_path):
    _install_companion(
        monkeypatch,
        tmp_path,
        '''
        def grpo_trainer__get_per_token_logps_and_entropies(function_name, function):
            def _get_per_token_logps_and_entropies(self, model, **kwargs):
                from unsloth_zoo.rl_replacements import grpo_get_vision_inputs
                return grpo_get_vision_inputs(kwargs)
            return _get_per_token_logps_and_entropies
        ''',
    )
    assert _rl.grpo_companion_vision_keys() == _rl.GRPO_VISION_KEYS
    assert set(_rl.grpo_shared_vision_inputs(dict.fromkeys(_rl.GRPO_VISION_KEYS, 1))) == set(
        _rl.GRPO_VISION_KEYS
    )


def test_an_unsloth_that_only_names_the_chunker_also_lifts_it(monkeypatch, tmp_path):
    """The companion change (unslothai/unsloth#11031) collects the keys in a module level
    helper and names only grpo_vision_chunks inside the replacement itself."""
    _install_companion(
        monkeypatch,
        tmp_path,
        '''
        def grpo_trainer__get_per_token_logps_and_entropies(function_name, function):
            def _get_per_token_logps_and_entropies(self, model, **kwargs):
                from unsloth_zoo.rl_replacements import grpo_vision_chunks
                return grpo_vision_chunks(kwargs, 1, 1)
            return _get_per_token_logps_and_entropies
        ''',
    )
    assert _rl.grpo_companion_vision_keys() == _rl.GRPO_VISION_KEYS


def test_no_companion_in_sys_modules_forwards_everything():
    """unsloth_zoo used on its own, or before unsloth has imported that module."""
    assert _rl.grpo_companion_vision_keys() == _rl.GRPO_VISION_KEYS


def test_a_no_grad_pass_this_reader_cannot_parse_does_not_turn_vision_off(
    monkeypatch, tmp_path
):
    """Fail open: a rewritten replacement that names no keys must not be read as naming none
    of them, which would drop pixel_values and train on the text alone.

    The source here is READABLE and merely unrecognised, which is the opposite case to
    test_a_companion_whose_source_cannot_be_read_falls_back_to_the_released_keys: there the
    text cannot be obtained at all, so there is nothing to conclude and the released set is
    assumed. Here the companion is visibly doing something this reader does not model, and
    narrowing on that would turn vision off on a guess."""
    _install_companion(
        monkeypatch,
        tmp_path,
        '''
        def grpo_trainer__get_per_token_logps_and_entropies(function_name, function):
            def _get_per_token_logps_and_entropies(self, model, **kwargs):
                return {key: kwargs.get(key) for key in SOME_TUPLE_DEFINED_ELSEWHERE}
            return _get_per_token_logps_and_entropies
        ''',
    )
    assert _rl.grpo_companion_vision_keys() == _rl.GRPO_VISION_KEYS


def test_a_companion_whose_source_cannot_be_read_falls_back_to_the_released_keys(
    monkeypatch, tmp_path
):
    """Source-stripped, frozen or dynamically wrapped packaging.

    The companion is THERE -- the patcher exists -- and the only thing missing is the ability
    to see which keys it forwards. Leaving the full tuple there was the unsafe half of that
    guess: the gradient pass would forward spatial_shapes, num_tiles and the position ids that
    every released unsloth omits, and the importance ratio and the KL term would then compare
    two different policies. The released set is the answer for every unsloth that does not
    carry the companion change, so it is the one an unknown companion gets.
    """
    module = _install_companion(
        monkeypatch,
        tmp_path,
        '''
        def grpo_trainer__get_per_token_logps_and_entropies(function_name, function):
            def _get_per_token_logps_and_entropies(self, model, **kwargs):
                return kwargs
            return _get_per_token_logps_and_entropies
        ''',
    )

    def _no_source(target):
        raise OSError("source not available")

    monkeypatch.setattr(_rl.inspect, "getsource", _no_source)
    assert _rl.grpo_companion_vision_keys() == _RELEASED_KEYS
    assert _rl.GRPO_RELEASED_VISION_KEYS == _RELEASED_KEYS
    shared = _rl.grpo_shared_vision_inputs(dict.fromkeys(_rl.GRPO_VISION_KEYS, 1))
    for dropped in ("spatial_shapes", "num_tiles", "image_position_ids"):
        assert dropped not in shared, dropped
    assert "pixel_values" in shared

    # Not memoized: this is a failure to look, not a fact about the process, so a run that can
    # read the source afterwards is not stuck with the conservative answer.
    assert _rl._GRPO_COMPANION_VISION_KEYS is None
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "unsloth.models.rl_replacements", module)
    monkeypatch.setattr(_rl, "_GRPO_COMPANION_VISION_KEYS", None)
    assert _rl.grpo_companion_vision_keys() == _rl.GRPO_VISION_KEYS


def test_the_companion_probe_reads_the_really_installed_unsloth():
    """Not a mock: whatever unsloth is installed here must be classified, and if it does not
    share the helper the answer must be a strict subset that still carries pixel_values."""
    pytest.importorskip("unsloth.models.rl_replacements")
    import unsloth.models.rl_replacements as installed

    keys = _rl.grpo_companion_vision_keys()
    patcher = getattr(installed, "grpo_trainer__get_per_token_logps_and_entropies", None)
    assert patcher is not None, "unsloth no longer exposes the no-grad replacement to probe"
    # The gate's own marker list, not a copy of it. Either name means the companion reads this
    # module's tuple, and unslothai/unsloth#11031 is chunker-only: it collects the keys in a
    # module level helper and names only grpo_vision_chunks inside the replacement. Naming one
    # marker here meant that once that companion was installed production returned all eleven
    # keys while this test asserted a strict subset, so the integration test failed against
    # the exact companion it exists to validate. Reusing the constant is right for THIS test,
    # whose job is only to classify whatever package is installed; the four mock tests above
    # pin each shape with literal source text, so a wrong constant is still caught there.
    installed_source = inspect.getsource(patcher)
    shares = any(marker in installed_source for marker in _rl.GRPO_SHARED_HELPER_MARKERS)
    if shares:
        assert keys == _rl.GRPO_VISION_KEYS
    else:
        assert set(keys) < set(_rl.GRPO_VISION_KEYS)
        assert "pixel_values" in keys


def test_the_gradient_pass_goes_through_the_gate():
    source = inspect.getsource(grpo_accumulated_loss)
    assert "grpo_shared_vision_inputs" in source


# Real tensors, because the chunker slices them and the branch under test reads the one it
# is handed rather than a marker string.
_PIXELS = torch.zeros(1, 4)
_MASK = torch.ones(1, 4)
_GRID = torch.tensor([[1, 1, 1]])
_SIZES = torch.tensor([[4, 4]])
_TTI = torch.tensor([[0, 0, 1, 1]])


class _TookTheTextBranch(Exception):
    """left_pack_padding was entered: the batch was repacked and the mask rebuilt."""


class _ReachedTheVisionBranch(Exception):
    """The branch was not entered: input_ids and completion_mask are as they arrived."""


def _which_branch(monkeypatch, vision_kwargs):
    """Which arrangement branch `grpo_accumulated_loss` takes for these vision kwargs.

    Observed, not read off the returned mapping. `pixel_values` is a sentinel in the caller as
    well as a model input: a None there makes the GRADIENT pass recompute max_left_pad,
    left-pack input_ids and rebuild completion_mask through
    create_completion_attention_mask, and it also switches the sequence-packing path on. The
    companion's no-grad pass decides both on its own pixel_values, which is not None. So a
    gate that blanks the key does not make the two passes agree; it makes them disagree about
    something worse. Every other test in this file passes by inspecting the dict, which is
    exactly how that got through, so this one runs the function and reports which branch it
    took -- each side raises its own sentinel the moment it is reached.
    """
    import types

    def _no_packing(*_a, **_k):
        raise _TookTheTextBranch

    def _no_vision(*_a, **_k):
        raise _ReachedTheVisionBranch

    monkeypatch.setattr(_rl, "left_pack_padding", _no_packing)
    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(
            unsloth_grpo_mini_batch = 1,
            unsloth_logit_chunk_multiplier = 1,
        ),
        processing_class = types.SimpleNamespace(pad_token_id = 0),
        model = types.SimpleNamespace(
            get_output_embeddings = lambda: types.SimpleNamespace(
                weight = torch.zeros(8, 4)
            )
        ),
        accelerator = types.SimpleNamespace(unwrap_model = _no_vision),
        use_vllm = False,
        _autocast_dtype = torch.bfloat16,
    )
    input_ids = torch.tensor([[0, 5, 6, 7]])
    try:
        _rl.grpo_accumulated_loss(
            trainer,
            input_ids,
            torch.ones_like(input_ids),
            2,
            torch.ones(1, 2),
            torch.zeros(1),
            None,
            None,
            **dict(vision_kwargs),
        )
    except _TookTheTextBranch:
        return "text"
    except _ReachedTheVisionBranch:
        return "vision"
    raise AssertionError("neither branch was reached; the harness has drifted")


def test_the_branch_harness_tells_the_two_apart(monkeypatch):
    """The control for `_which_branch`. A harness that answered "vision" whatever it was given
    would make the assertion above pass for nothing, which is the failure mode this whole
    exercise is about."""
    assert _which_branch(monkeypatch, {}) == "text"
    assert (
        _which_branch(monkeypatch, {"pixel_values": _PIXELS, "image_grid_thw": _GRID})
        == "vision"
    )


def test_a_companion_without_the_chunker_gets_no_pixels_for_a_gridless_vlm(
    monkeypatch, tmp_path
):
    """Intersecting key NAMES cannot express a shape.

    A companion that does not share the chunker slices the pixels with a loop of its own, and
    that loop appends None for pixel_values unless image_grid_thw is there to slice them by.
    For a VLM that carries no grid -- Gemma 3, InternVL, LFM2-VL -- forwarding pixels on the
    gradient side alone leaves the current policy looking at images the reference policy never
    saw, so the ratio and the KL term compare two different policies however well the names
    match.
    """
    _install_companion(
        monkeypatch,
        tmp_path,
        '''
        def grpo_trainer__get_per_token_logps_and_entropies(function_name, function):
            def _get_per_token_logps_and_entropies(self, model, **kwargs):
                pixel_values = kwargs.get("pixel_values", None)
                image_grid_thw = kwargs.get("image_grid_thw", None)
                pixel_attention_mask = kwargs.get("pixel_attention_mask", None)
                image_sizes = kwargs.get("image_sizes", None)
                num_images = kwargs.get("num_images", None)
                token_type_ids = kwargs.get("token_type_ids", None)
                mm_token_type_ids = kwargs.get("mm_token_type_ids", None)
                return pixel_values, image_grid_thw, pixel_attention_mask, image_sizes, \\
                    num_images, token_type_ids, mm_token_type_ids
            return _get_per_token_logps_and_entropies
        ''',
    )
    monkeypatch.setattr(_rl, "_GRPO_COMPANION_PIXELS_WARNED", False, raising = False)

    gridless = {
        "pixel_values": _PIXELS,
        "pixel_attention_mask": _MASK,
        "image_sizes": _SIZES,
        "token_type_ids": _TTI,
        "num_images": [1],
    }
    shared = _rl.grpo_shared_vision_inputs(gridless)
    # The SENTINEL survives. `grpo_accumulated_loss` reads pixel_values to choose between the
    # vision branch and the text branch, and the companion chooses on its own pixel_values,
    # which is a real tensor: it appends None per chunk inside its loop, after the branch is
    # already decided. Blanking it here would stop the two passes disagreeing about pixels and
    # start them disagreeing about how the sequences are ARRANGED -- left-packed input_ids, a
    # rebuilt completion_mask and the packed forward on one side only -- which is the worse
    # comparison of the two, for exactly the models this is for.
    assert shared["pixel_values"] is _PIXELS, shared
    assert shared["pixel_attention_mask"] is _MASK, shared
    # What is actually FORWARDED carries no pixels, which is where the companion drops them.
    chunk = _rl.grpo_vision_chunks(shared, 1, 1)[0]
    assert "pixel_values" not in chunk, chunk
    assert "pixel_attention_mask" not in chunk, chunk
    # Everything the companion DOES forward is still forwarded, or the two passes disagree in
    # the other direction.
    assert shared["image_sizes"] is _SIZES
    assert shared["token_type_ids"] is _TTI
    assert shared["num_images"] == [1]
    assert torch.equal(chunk["image_sizes"], _SIZES)
    assert torch.equal(chunk["token_type_ids"], _TTI)

    # And the branch is observed, not read off the mapping: the gradient pass must still take
    # the vision branch for this batch, leaving input_ids exactly as the companion has them.
    assert _which_branch(monkeypatch, gridless) == "vision"

    # With a grid the companion slices the pixels, so they are forwarded on both sides.
    withgrid = dict(gridless, image_grid_thw = _GRID)
    shared = _rl.grpo_shared_vision_inputs(withgrid)
    assert shared["pixel_values"] is _PIXELS
    assert shared["image_grid_thw"] is _GRID
    assert shared["pixel_attention_mask"] is _MASK
    chunk = _rl.grpo_vision_chunks(shared, 1, 1)[0]
    assert chunk["pixel_values"] is not None
    assert chunk["pixel_attention_mask"] is not None


def test_a_companion_that_shares_the_chunker_keeps_the_pixels(monkeypatch, tmp_path):
    """The narrowing is only for a companion that slices with a loop of its own. Where the two
    passes run the same chunker there is nothing to disagree about, and dropping the pixels
    there would throw away the whole point of sharing it."""
    _install_companion(
        monkeypatch,
        tmp_path,
        '''
        def grpo_trainer__get_per_token_logps_and_entropies(function_name, function):
            def _get_per_token_logps_and_entropies(self, model, **kwargs):
                from unsloth_zoo.rl_replacements import grpo_vision_chunks
                return grpo_vision_chunks(kwargs, 1, 1)
            return _get_per_token_logps_and_entropies
        ''',
    )
    shared = _rl.grpo_shared_vision_inputs(
        {"pixel_values": "PIXELS", "spatial_shapes": "SHAPES", "num_tiles": "TILES"}
    )
    assert shared["pixel_values"] == "PIXELS"
    assert shared["spatial_shapes"] == "SHAPES"
    assert shared["num_tiles"] == "TILES"


def test_a_padded_image_tensor_stays_on_the_sample_axis():
    """SmolVLM and Idefics pad to the widest sample instead of flattening by image.

    `num_images = [2, 0]` over two samples makes the two readings numerically identical --
    two image rows, two samples -- and the length comparison alone then sliced a padded
    tensor by image: with one sample per chunk both of the first sample's rows went to the
    first chunk and an empty tensor to the second, which is a batch-shape failure at best and
    pixels attached to the wrong sample at worst. The layouts differ in RANK, which is what
    the counts cannot express.
    """
    padded = torch.arange(2 * 2 * 3 * 4 * 4).reshape(2, 2, 3, 4, 4)
    sizes = torch.arange(2 * 2 * 2).reshape(2, 2, 2)
    chunks = grpo_vision_chunks(
        {
            "pixel_values": padded,
            "image_sizes": sizes,
            "num_images": [2, 0],
        },
        2,
        1,
    )
    assert len(chunks) == 2
    assert torch.equal(chunks[0]["pixel_values"], padded[0:1])
    assert torch.equal(chunks[1]["pixel_values"], padded[1:2])
    assert torch.equal(chunks[0]["image_sizes"], sizes[0:1])
    assert torch.equal(chunks[1]["image_sizes"], sizes[1:2])


def test_a_flattened_image_tensor_is_still_sliced_by_image():
    """The control. One row per image, [N, C, H, W], with the same counts: here the image
    axis really is the first one, and slicing by sample would hand the second sample rows
    that belong to the first."""
    flat = torch.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
    sizes = torch.arange(2 * 2).reshape(2, 2)
    chunks = grpo_vision_chunks(
        {
            "pixel_values": flat,
            "image_sizes": sizes,
            "num_images": [2, 0],
        },
        2,
        1,
    )
    assert torch.equal(chunks[0]["pixel_values"], flat[0:2])
    assert chunks[1]["pixel_values"].shape[0] == 0
    assert torch.equal(chunks[0]["image_sizes"], sizes[0:2])

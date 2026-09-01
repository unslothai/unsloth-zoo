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
#
# What a text-only multimodal load has to keep consistent, tested under the
# MLX-on-torch shim so it runs on Linux CI. The rest of this coverage lives in
# test_mlx_vlm_label_masks.py, which needs a real MLX runtime and so never runs
# off Apple Silicon.
#
# The vision grid form is chosen from a family set keyed on the canonical
# model_type, while mlx-vlm resolves an aliased config to that canonical name
# through MODEL_REMAPPING before loading its module. Both ends have to resolve
# the same way: a config spelled `muse-glimmer` loads the `muse_glimmer` tower,
# whose first line is `grid_thw.tolist()`, and a Python tuple has no `tolist`.

from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


@pytest.fixture
def mutils():
    import unsloth_zoo.mlx.utils as mutils
    return mutils


@pytest.fixture
def remapping(monkeypatch):
    """Install a MODEL_REMAPPING the way a real mlx-vlm ships one."""
    def _install(mapping):
        module = sys.modules["mlx_vlm.utils"]
        monkeypatch.setattr(module, "MODEL_REMAPPING", mapping, raising = False)
    return _install


def _prepare(mutils, model_type, **grids):
    batch = {key: value for key, value in grids.items()}
    return mutils._prepare_vlm_batch_for_compile(batch, {"model_type": model_type})


def _is_array(value):
    return not isinstance(value, (tuple, list)) and hasattr(value, "shape")


# --- the array-grid families keep an indexable grid -------------------------


@pytest.mark.parametrize("model_type", ["glm4v", "glm_ocr", "muse_glimmer"])
def test_array_grid_families_get_an_indexable_grid(mutils, model_type):
    out = _prepare(mutils, model_type, image_grid_thw = [[1, 2, 2]])
    assert _is_array(out["image_grid_thw"])


@pytest.mark.parametrize("model_type", ["qwen2_vl", "qwen2_5_vl", "paddle_ocr"])
def test_compile_patched_families_keep_the_traceable_tuple(mutils, model_type):
    # An mx.array becomes a tracer under mx.compile and its .tolist() raises,
    # which is why these keep Python tuples.
    out = _prepare(mutils, model_type, image_grid_thw = [[1, 2, 2]])
    assert isinstance(out["image_grid_thw"], tuple)


# --- aliases resolve the same way the loader resolves them ------------------


def test_an_aliased_config_still_reaches_the_array_form(mutils, remapping):
    """`muse-glimmer` loads the muse_glimmer tower, so it needs that tower's grid."""
    remapping({"muse-glimmer": "muse_glimmer"})
    out = _prepare(mutils, "muse-glimmer", image_grid_thw = [[1, 2, 2]])
    assert _is_array(out["image_grid_thw"])


def test_an_alias_of_a_tuple_family_stays_a_tuple(mutils, remapping):
    remapping({"qwen2-vl": "qwen2_vl"})
    out = _prepare(mutils, "qwen2-vl", image_grid_thw = [[1, 2, 2]])
    assert isinstance(out["image_grid_thw"], tuple)


def test_the_video_grid_follows_the_image_grid(mutils, remapping):
    remapping({"muse-glimmer": "muse_glimmer"})
    out = _prepare(
        mutils, "muse-glimmer",
        image_grid_thw = [[1, 2, 2]], video_grid_thw = [[2, 2, 2]],
    )
    assert _is_array(out["image_grid_thw"])
    assert _is_array(out["video_grid_thw"])


# --- resolution never becomes a new way to fail -----------------------------


def test_an_unlisted_architecture_is_untouched(mutils, remapping):
    remapping({"something-else": "something_else"})
    out = _prepare(mutils, "something-else", image_grid_thw = [[1, 2, 2]])
    assert isinstance(out["image_grid_thw"], tuple)


@pytest.mark.parametrize(
    "hostile",
    [
        pytest.param(None, id = "not a mapping"),
        pytest.param(object(), id = "no get"),
    ],
)
def test_a_broken_remapping_falls_back_to_the_raw_name(mutils, remapping, hostile):
    """An old or oddly shaped mlx-vlm must not turn batch prep into a crash."""
    remapping(hostile)
    out = _prepare(mutils, "muse_glimmer", image_grid_thw = [[1, 2, 2]])
    assert _is_array(out["image_grid_thw"])


def test_the_loader_and_the_grid_agree_on_the_same_alias(remapping):
    """One helper, so the two ends cannot drift apart again."""
    import unsloth_zoo.mlx.loader as loader
    import unsloth_zoo.mlx.utils as mutils
    remapping({"muse-glimmer": "muse_glimmer"})
    assert loader._mlx_vlm_text_path_is_verified("muse-glimmer") is True
    assert mutils._mlx_vlm_canonical_model_type("muse-glimmer") == "muse_glimmer"


def test_an_empty_model_type_resolves_to_nothing(mutils):
    assert mutils._mlx_vlm_canonical_model_type("") == ""
    assert mutils._mlx_vlm_canonical_model_type(None) == ""


def test_a_capitalised_model_type_resolves_like_mlx_vlm_resolves_it(mutils):
    """mlx-vlm lower-cases before it remaps, so a `Muse_Glimmer` config loads."""
    assert mutils._mlx_vlm_canonical_model_type("Muse_Glimmer") == "muse_glimmer"


# --- what the losses accept from a model call -------------------------------


def test_logits_come_back_from_a_wrapper_and_from_a_raw_array(mutils):
    import types
    array = object()
    assert mutils._model_logits(array) is array
    wrapped = types.SimpleNamespace(logits=array)
    assert mutils._model_logits(wrapped) is array


def test_a_wrapper_with_no_logits_says_so(mutils):
    """Unwrapping to None instead fails deep inside cross-entropy."""
    import types
    with pytest.raises(ValueError, match="`logits` is None"):
        mutils._model_logits(types.SimpleNamespace(logits=None))


# --- generate() picks the processor by presence, not truthiness -------------


def test_generate_keeps_a_falsy_processor(monkeypatch):
    """A processor is preferred because a tokenizer cannot preprocess images.

    Resolution is observed through the object whose stopping criteria generate
    resets, which is the first thing it does with whatever it picked.
    """
    import types
    import unsloth_zoo.mlx.loader as loader
    import unsloth_zoo.mlx.utils as mutils

    picked = []

    def _side(tag):
        criteria = types.SimpleNamespace(reset = lambda eos: picked.append(tag))
        return types.SimpleNamespace(stopping_criteria = criteria)

    class _EmptyProcessor:
        """Falsy, the way a mapping-like processor with no entries would be."""
        def __init__(self):
            self.tokenizer = _side("processor")
        def __len__(self):
            return 0

    processor = _EmptyProcessor()
    model = types.SimpleNamespace(_processor = processor, _tokenizer = _side("tokenizer"))

    def _fake_to_mx_vlm_batch(inputs):
        raise _Stop

    # The module object, not a dotted string: a string target is resolved by walking
    # attributes, which fails on a real MLX runtime where the submodule is not bound
    # on its parent package yet.
    monkeypatch.setattr(mutils, "_to_mx_vlm_batch", _fake_to_mx_vlm_batch)
    with pytest.raises(_Stop):
        loader._mlx_generate_vlm(model, "hello")
    assert picked == ["processor"]


class _Stop(Exception):
    pass

# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""The LoRA target group must match a whole leaf name, never the tail of one.

get_peft_regex collects the leaf names of every nn.Linear into one alternation.
Nemotron-H's vision tower has a Linear named exactly `proj` (`...attn.proj`), so
the bare `proj` joins the group. Without a dot in front of the group, the `.*?`
before it absorbs `...mixer.fc1_latent_` and `proj` matches the rest, so the
nn.Identity placeholders the mixer keeps in `fc1_latent_proj` / `fc2_latent_proj`
on every layer without a latent projection were selected, and get_peft_model
stopped with "Target module Identity() is not supported".

The models here are built to that shape (no downloads); every test states the arm
it measures and fails on a tree without the anchor.
"""
import re

import pytest
import torch
import torch.nn as nn

from unsloth_zoo.peft_utils import get_peft_regex


class _Mixer(nn.Module):
    def __init__(self, dim, latent):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        # Nemotron-H: a Linear when the layer uses a latent projection, an
        # nn.Identity placeholder when it does not. Both names end in "proj".
        self.fc1_latent_proj = nn.Linear(dim, dim) if latent else nn.Identity()
        self.fc2_latent_proj = nn.Linear(dim, dim) if latent else nn.Identity()


class _Layer(nn.Module):
    def __init__(self, dim, latent):
        super().__init__()
        self.mixer = _Mixer(dim, latent)


class _VisionBlock(nn.Module):
    """The half that is easy to miss: an attention output Linear whose leaf name
    is exactly `proj`. That is what puts the bare `proj` in the group."""

    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)


class _Model(nn.Module):
    def __init__(self, layers = 4, dim = 8, latent_layers = ()):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList(_Layer(dim, i in latent_layers) for i in range(layers))
        self.vision_tower = nn.Module()
        self.vision_tower.blocks = nn.ModuleList(nn.Module() for _ in range(2))
        for block in self.vision_tower.blocks:
            block.attn = _VisionBlock(dim)

    class config:
        _name_or_path = "test/leaf-anchor"


def _matched(model, regex):
    # PEFT selects with re.fullmatch over module names.
    return {n for n, _ in model.named_modules() if n and re.fullmatch(regex, n, flags = re.DOTALL)}


def _linears(model):
    return {n for n, m in model.named_modules() if isinstance(m, nn.Linear)}


def test_bare_proj_puts_proj_in_the_group():
    """Precondition of the bug: the vision `proj` Linear is a group entry."""
    regex = get_peft_regex(_Model())
    assert re.search(r"\|proj[|)]", regex) or "(?:proj" in regex, regex


def test_identity_placeholders_are_not_selected():
    """The arm that fails without the anchor: Identity modules were targets."""
    model = _Model()
    matched = _matched(model, get_peft_regex(model))
    assert matched, "the matcher selected nothing, so this proves nothing"
    non_linear = matched - _linears(model)
    assert not non_linear, sorted(non_linear)


def test_every_real_target_is_still_selected():
    """Lossless: every Linear under an attention or mlp component is reached."""
    # Two latent layers: a leaf name carried by a single Linear is treated as a
    # projection head and left out of the group on purpose, so one would not
    # test what this test is about.
    model = _Model(latent_layers = (0, 1))
    matched = _matched(model, get_peft_regex(model))
    for layer in range(4):
        for leaf in ("in_proj", "out_proj"):
            assert f"language_model.layers.{layer}.mixer.{leaf}" in matched
    # layers 0 and 1 have real latent Linears: they stay targetable by their own name
    for layer in (0, 1):
        assert f"language_model.layers.{layer}.mixer.fc1_latent_proj" in matched
        assert f"language_model.layers.{layer}.mixer.fc2_latent_proj" in matched
    for block in range(2):
        assert f"vision_tower.blocks.{block}.attn.proj" in matched
        assert f"vision_tower.blocks.{block}.attn.qkv" in matched
    # and the Identity copies on layers 2-3 are not: they share the leaf name
    # with a real Linear, so the anchor cannot tell them apart; they are left out
    # by their exact name because they carry no parameters.
    assert not (matched - _linears(model)), sorted(matched - _linears(model))


def test_placeholders_are_excluded_by_exact_name_only():
    model = _Model(latent_layers = (0, 1))
    regex = get_peft_regex(model)
    # the exclusion names each placeholder verbatim (escaped for the regex)
    for layer in (2, 3):
        assert re.escape(f"language_model.layers.{layer}.mixer.fc1_latent_proj") in regex
        assert re.escape(f"language_model.layers.{layer}.mixer.fc2_latent_proj") in regex
    assert re.escape("language_model.layers.0.mixer.fc1_latent_proj") not in regex


def test_model_without_placeholders_gets_no_exclusion():
    model = _Model(latent_layers = (0, 1, 2, 3))
    assert not get_peft_regex(model).startswith("(?!")


def test_group_is_anchored_to_a_dot():
    regex = get_peft_regex(_Model())
    assert r".*?\.(?:" in regex, regex


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_selection_does_not_depend_on_dtype(dtype):
    model = _Model().to(dtype)
    matched = _matched(model, get_peft_regex(model))
    assert not (matched - _linears(model))

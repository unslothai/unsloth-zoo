# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# SPDX-License-Identifier: AGPL-3.0-or-later

# With no gpu given the visual-server env builder keeps one card: the first entry of an
# inherited CUDA_VISIBLE_DEVICES (it used to force "0", a GPU the mask may hide), GPU 0
# when none is inherited; a given gpu replaces the mask verbatim.

from __future__ import annotations

import inspect

import pytest

from unsloth_zoo.diffusion_studio import visual_engine as V

def _env(base_env, **kw):
    return V._build_subprocess_env("/dg/srv", base_env = base_env, os_name = "posix", **kw)


@pytest.mark.parametrize("mask,expected", [
    ("", ""), ("-1", "-1"), (",2", ",2"), (" , 1", " , 1"), ("-1,0", "-1,0"),
    ("2,1", "2"), ("0,1", "0"), (" 3 , 0", "3"),
    ("GPU-8a1b2c3d-0000-1111-2222-333344445555", "GPU-8a1b2c3d-0000-1111-2222-333344445555"),
    ("GPU-aaaa,GPU-bbbb", "GPU-aaaa"),
    ("MIG-0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0", "MIG-0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"),
    ("MIG-GPU-8a1b2c3d/1/0", "MIG-GPU-8a1b2c3d/1/0"),
])
def test_inherited_mask_keeps_one_visible_gpu_by_default(mask, expected):
    # Identical GPUs under "0,1" still run on GPU 0 alone, as before; a hidden GPU 0 is never used.
    assert _env({"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": mask})["CUDA_VISIBLE_DEVICES"] == expected


def test_unset_mask_keeps_gpu_0_by_default():
    # As before: no inherited mask still pins one card, not every GPU.
    assert _env({"PATH": "/usr/bin"})["CUDA_VISIBLE_DEVICES"] == "0"


@pytest.mark.parametrize("gpu", ["0", "3", "", "2,1", "GPU-8a1b2c3d", 1])
def test_explicit_gpu_replaces_inherited_mask(gpu):
    # Studio always passes an explicit token ("" = CPU-masked); that must still win.
    env = _env({"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "5"}, gpu = gpu)
    assert env["CUDA_VISIBLE_DEVICES"] == str(gpu)


def test_visual_server_default_gpu_is_inherit():
    assert inspect.signature(V.VisualServer.__init__).parameters["gpu"].default is None

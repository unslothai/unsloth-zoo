# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# SPDX-License-Identifier: AGPL-3.0-or-later

# The visual-server env builder must keep an inherited CUDA_VISIBLE_DEVICES when no
# gpu is given (it used to force "0"), and replace it verbatim when one is.

from __future__ import annotations

import inspect

import pytest

from unsloth_zoo.diffusion_studio import visual_engine as V

MASKS = ["", "-1", "2,1", "GPU-8a1b2c3d-0000-1111-2222-333344445555",
         "MIG-0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0", "MIG-GPU-8a1b2c3d/1/0"]


def _env(base_env, **kw):
    return V._build_subprocess_env("/dg/srv", base_env = base_env, os_name = "posix", **kw)


@pytest.mark.parametrize("mask", MASKS)
def test_inherited_mask_kept_by_default(mask):
    assert _env({"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": mask})["CUDA_VISIBLE_DEVICES"] == mask


def test_unset_mask_stays_unset_by_default():
    assert "CUDA_VISIBLE_DEVICES" not in _env({"PATH": "/usr/bin"})


def test_gpu_none_keeps_inherited_mask():
    env = _env({"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "2,1"}, gpu = None)
    assert env["CUDA_VISIBLE_DEVICES"] == "2,1"


@pytest.mark.parametrize("gpu", ["0", "3", "", "2,1", "GPU-8a1b2c3d", 1])
def test_explicit_gpu_replaces_inherited_mask(gpu):
    # Studio always passes an explicit token ("" = CPU-masked); that must still win.
    env = _env({"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "5"}, gpu = gpu)
    assert env["CUDA_VISIBLE_DEVICES"] == str(gpu)


def test_visual_server_default_gpu_is_inherit():
    assert inspect.signature(V.VisualServer.__init__).parameters["gpu"].default is None

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# SPDX-License-Identifier: AGPL-3.0-or-later

# Coverage for visual_engine._canvas_maxtok: the per-turn DiffusionGemma canvas is
# non-causal, so its compute buffer is ~quadratic in size; oversized values must defer
# to the server's MAXTOK=0 auto-size path instead of reserving a buffer that OOMs.

from __future__ import annotations

import pytest

from unsloth_zoo.diffusion_studio.visual_engine import _canvas_maxtok


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, 0),          # auto-size sentinel stays auto
        (1, 1),          # small explicit honored
        (4096, 4096),    # below threshold honored
        (8192, 8192),    # threshold honored
        (8193, 0),       # just over -> auto
        (32768, 0),      # full model context -> auto (avoids the ~77 GB buffer OOM)
        (262144, 0),     # very large -> auto
    ],
)
def test_canvas_maxtok_defers_oversized_to_autosize(value, expected):
    assert _canvas_maxtok(value) == expected


def test_canvas_maxtok_coerces_str():
    assert _canvas_maxtok("4096") == 4096
    assert _canvas_maxtok("99999") == 0

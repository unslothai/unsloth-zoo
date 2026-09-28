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


import importlib

import numpy as np
import pytest
import torch
from PIL import Image

from unsloth_zoo.temporary_patches.misc import patch_idefics2_image_processor_leading_text_row


def _processor_classes():
    found = []
    for module, name in (
        ("transformers.models.idefics2.image_processing_idefics2", "Idefics2ImageProcessor"),
        ("transformers.models.idefics2.image_processing_pil_idefics2", "Idefics2ImageProcessorPil"),
        ("transformers.models.idefics2.image_processing_idefics2_fast", "Idefics2ImageProcessorFast"),
    ):
        try:
            cls = importlib.import_module(module).__dict__.get(name)
        except Exception:
            continue
        if cls is not None and cls not in found:
            found.append(cls)
    return found


CLASSES = _processor_classes()


@pytest.fixture(scope = "module", autouse = True)
def _patched():
    patch_idefics2_image_processor_leading_text_row()
    patch_idefics2_image_processor_leading_text_row()


def _image():
    return Image.fromarray((np.random.RandomState(0).rand(40, 56, 3) * 255).astype(np.uint8))


@pytest.mark.skipif(not CLASSES, reason = "no Idefics2 image processor in this transformers")
@pytest.mark.parametrize("cls", CLASSES, ids = lambda c: c.__name__)
@pytest.mark.parametrize("layout", [[0, 1], [0, 1, 0], [0, 0, 1]])
def test_leading_text_only_row(cls, layout):
    processor = cls()
    image = _image()
    ref = processor([[image], []], return_tensors = "pt")
    out = processor([[image] if flag else [] for flag in layout], return_tensors = "pt")
    k = layout.index(1)
    if len(ref["pixel_values"]) == 1:
        # transformers 4.57's fast processor drops text-only rows even with the image row first,
        # so the patch's job there is only to not raise: the image row comes back as it would.
        assert len(out["pixel_values"]) == 1
        assert torch.equal(torch.as_tensor(out["pixel_values"][0]), torch.as_tensor(ref["pixel_values"][0]))
        return
    assert out["pixel_values"].shape[0] == len(layout)
    assert torch.equal(torch.as_tensor(out["pixel_values"][k]), torch.as_tensor(ref["pixel_values"][0]))
    for i in range(len(layout)):
        if i != k and "pixel_attention_mask" in out:
            assert not torch.as_tensor(out["pixel_attention_mask"][i]).any()


@pytest.mark.skipif(not CLASSES, reason = "no Idefics2 image processor in this transformers")
@pytest.mark.parametrize("cls", CLASSES, ids = lambda c: c.__name__)
def test_patch_is_idempotent_and_keeps_wrapped(cls):
    preprocess = cls.__dict__["preprocess"]
    assert getattr(preprocess, "_unsloth_leading_text_row", False)
    assert not getattr(preprocess.__wrapped__, "_unsloth_leading_text_row", False)

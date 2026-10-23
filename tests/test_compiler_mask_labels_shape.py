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

"""The generated mask_attention_mask_out only masks labels with an attention mask over the same positions."""

import re
from pathlib import Path

import torch

_SOURCE = (Path(__file__).resolve().parents[1] / "unsloth_zoo" / "compiler.py").read_text(encoding = "utf-8")


def _helper():
    block = re.search(r"\ndef mask_attention_mask_out\(.*?\n    return labels\n", _SOURCE, flags = re.DOTALL).group(0)
    ns = {}
    exec(block, ns)
    return ns["mask_attention_mask_out"]


def test_matching_mask_still_masks_padding():
    labels = torch.tensor([[5, 6, 7], [8, 9, 10]])
    out = _helper()(labels.clone(), torch.tensor([[1, 1, 0], [1, 0, 0]]))
    assert out.tolist() == [[5, 6, -100], [8, -100, -100]]


def test_text_encoder_mask_leaves_musicgen_codebook_labels_alone():
    labels = torch.randint(0, 10, (2, 50, 4))  # decoder frames x codebooks
    out = _helper()(labels.clone(), torch.ones(2, 7, dtype = torch.long))  # text encoder mask
    assert torch.equal(out, labels)


def test_codebook_labels_are_left_alone_even_when_lengths_match():
    labels = torch.randint(0, 10, (2, 7, 4))  # 7 decoder frames, 7 text tokens: same length, other modality
    mask = torch.tensor([[1] * 7, [1] * 5 + [0] * 2])
    assert torch.equal(_helper()(labels.clone(), mask), labels)

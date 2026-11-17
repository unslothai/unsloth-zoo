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

"""The patched Gemma3Processor keeps every row of a batch that mixes image and text-only examples."""
import pytest

pytest.importorskip("transformers")
from PIL import Image

TINY = "trl-internal-testing/tiny-Gemma3ForConditionalGeneration"


def test_mixed_image_and_text_rows_are_all_kept():
    from transformers import AutoProcessor

    import unsloth_zoo.temporary_patches.gemma as gemma

    gemma.patch_Gemma3Processor()
    try:
        proc = AutoProcessor.from_pretrained(TINY)
    except OSError as e:
        pytest.skip(f"{TINY} not downloadable: {e}")
    img = Image.new("RGB", (32, 32), (255, 0, 0))
    text = [f"{proc.boi_token} What color?", "Just text.", "More text."]
    out = proc(text = text, images = [[img], [], []], padding = True)
    assert len(out["input_ids"]) == 3
    alone = proc(text = ["More text."], padding = True)["input_ids"][0]
    row = [t for t in out["input_ids"][2] if t != proc.tokenizer.pad_token_id]
    assert row == [t for t in alone if t != proc.tokenizer.pad_token_id]

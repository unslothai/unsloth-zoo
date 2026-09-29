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

"""The patched Gemma3Processor.__call__ must mark the image soft tokens in token_type_ids.

Gemma3 attends bidirectionally over the tokens token_type_ids marks as image. transformers >= 5.15
points processor.image_token_id at the BOI token, so building token_type_ids from it marked only
<start_of_image> and left the 256 soft tokens causal during training.
"""
import pytest

pytest.importorskip("transformers")
from PIL import Image

TINY = "trl-internal-testing/tiny-Gemma3ForConditionalGeneration"


def _processor():
    from transformers import AutoProcessor
    try:
        return AutoProcessor.from_pretrained(TINY)
    except Exception as e:
        pytest.skip(f"{TINY} processor unavailable: {type(e).__name__}: {e}")


@pytest.mark.parametrize("batch", [1, 2])
def test_token_type_ids_mark_image_soft_tokens(batch):
    from unsloth_zoo.temporary_patches.gemma import patch_Gemma3Processor
    patch_Gemma3Processor()
    proc = _processor()
    soft = proc.tokenizer.image_token_id
    img = Image.new("RGB", (32, 32), (255, 0, 0))
    texts = [f"{proc.tokenizer.bos_token}<start_of_turn>user\n{proc.boi_token}What is this{'?' * (i + 1)}<end_of_turn>\n"
             for i in range(batch)]
    out = proc(text=texts, images=[[img]] * batch, padding=True, return_tensors="pt")
    for ids, tti in zip(out["input_ids"].tolist(), out["token_type_ids"].tolist()):
        assert tti == [int(i == soft) for i in ids]
        assert sum(tti) == proc.image_seq_length

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

"""A text-only tokenizer handed to UnslothVisionDataCollator (AutoTokenizer, or a local
VLM folder without preprocessor_config.json) must say what was passed and how to fix it
(unslothai/notebooks#124). CPU-pure and offline."""

import pytest

from unsloth_zoo.vision_utils import UnslothVisionDataCollator


def _text_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast(tokenizer_object = Tokenizer(WordLevel({"[UNK]": 0}, unk_token = "[UNK]")))


def test_text_tokenizer_error_names_type_and_fix():
    tok = _text_tokenizer()
    with pytest.raises(TypeError) as e:
        UnslothVisionDataCollator(object(), tok)
    msg = str(e.value)
    assert "requires an image or audio processor" in msg
    assert type(tok).__name__ in msg
    assert "FastVisionModel.from_pretrained" in msg
    assert "preprocessor_config.json" in msg

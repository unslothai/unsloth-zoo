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

"""TRL tools= sets response_template; AddedToken-converting its {"content": ...} fields broke tokenizer save."""

import copy
import json

import pytest

TEMPLATE = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|im_start|>assistant\n",
    "fields": {
        "reasoning_content": {"open": "<think>", "close_pattern": r"</think>\s*", "content": "text"},
        "tool_calls": {
            "open": "<tool_call>",
            "close_pattern": r"</tool_call>\s*",
            "repeats": True,
            "content": "json",
            "transform": {"type": "function", "function": "{content}"},
        },
        "content": {"close_pattern": r"<\|im_end\|>\s*", "content": "text"},
    },
}
SPECIAL = {"content": "<x>", "lstrip": False, "rstrip": False, "normalized": False, "single_word": False, "special": True}


@pytest.fixture(scope="module")
def tokenizer_base():
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from unsloth_zoo.temporary_patches.misc import patch_tokenizer_convert_added_tokens

    patch_tokenizer_convert_added_tokens()
    assert PreTrainedTokenizerBase.convert_added_tokens.__func__.__name__ == "patched_convert_added_tokens"
    return PreTrainedTokenizerBase


@pytest.mark.parametrize("key", ["response_template", "response_schema"])
@pytest.mark.parametrize("save", [False, True])
def test_parser_untouched_and_serialisable(tokenizer_base, key, save):
    config = {key: copy.deepcopy(TEMPLATE), "eos_token": "<|im_end|>"}
    out = tokenizer_base.convert_added_tokens(config, save=save, add_type_field=True)
    assert out[key] == TEMPLATE
    json.dumps(out)


def test_dict_special_tokens_still_load_as_added_token(tokenizer_base):
    from transformers.tokenization_utils_base import AddedToken

    config = {"additional_special_tokens": [dict(SPECIAL)], "response_template": copy.deepcopy(TEMPLATE)}
    out = tokenizer_base.convert_added_tokens(config, save=False)
    token = out["additional_special_tokens"][0]
    assert isinstance(token, AddedToken) and token.content == "<x>" and token.special
    assert out["response_template"] == TEMPLATE


def test_save_round_trips_added_token(tokenizer_base):
    from transformers.tokenization_utils_base import AddedToken

    token = AddedToken("<x>", special=True, normalized=False)
    out = tokenizer_base.convert_added_tokens({"pad_token": token}, save=True, add_type_field=True)
    assert out["pad_token"]["__type"] == "AddedToken" and out["pad_token"]["content"] == "<x>"
    json.dumps(out)

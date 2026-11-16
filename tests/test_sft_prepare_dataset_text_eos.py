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

"""Plain-text SFT rows end in exactly one EOS, as in TRL; chat-formatted rows and rows that already end in EOS are unchanged."""
import pytest
from datasets import Dataset

from unsloth_zoo.dataset_utils import sft_prepare_dataset

transformers = pytest.importorskip("transformers")

TOKENIZER = "trl-internal-testing/tiny-Qwen3ForCausalLM"


@pytest.fixture(scope = "module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    except Exception as e:
        pytest.skip(f"{TOKENIZER} unavailable: {e}")


class Args:
    max_length = 512
    dataset_text_field = "text"
    remove_unused_columns = True


class Trainer:
    model = None
    data_collator = None


def _ids(tokenizer, texts):
    out = sft_prepare_dataset(Trainer(), Dataset.from_dict({"text": texts}), tokenizer, Args(), False, None, "train")
    return [row["input_ids"] for row in out]


def test_plain_text_gets_one_eos(tokenizer):
    eos = tokenizer.eos_token_id
    plain, already = _ids(tokenizer, ["The sky is blue.", "The sky is blue." + tokenizer.eos_token])
    assert plain[-1] == eos and plain.count(eos) == 1
    assert already == plain


def test_chat_formatted_text_is_unchanged(tokenizer):
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}], tokenize = False
    )
    assert not chat.endswith(tokenizer.eos_token), "the template ends after the turn marker"
    (ids,) = _ids(tokenizer, [chat])
    assert ids == tokenizer(chat, add_special_tokens = False)["input_ids"]


class _AppendsEos:
    """A tokenizer whose post-processor appends EOS (add_eos_token = True)."""
    eos_token, eos_token_id, bos_token, chat_template = "</s>", 2, None, None

    def __call__(self, text, add_special_tokens = True, **kwargs):
        def ids(t):
            return [3 + (ord(c) % 20) for c in t] + ([2] if add_special_tokens else [])
        return {"input_ids": [ids(t) for t in text] if isinstance(text, list) else ids(text)}


def test_a_tokenizer_that_appends_eos_gets_no_second_one():
    (ids,) = _ids(_AppendsEos(), ["The sky is blue."])
    assert ids == _AppendsEos()("The sky is blue.")["input_ids"] and ids.count(2) == 1

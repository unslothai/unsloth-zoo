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

"""assistant_only_loss on conversational prompt/completion rows drops tool turns inside the completion."""

import re

import pytest
from datasets import Dataset

import unsloth_zoo.dataset_utils as dataset_utils
from unsloth_zoo.dataset_utils import sft_prepare_dataset

trl_sft = pytest.importorskip("trl.trainer.sft_trainer")
transformers = pytest.importorskip("transformers")

TOKENIZER = "trl-internal-testing/tiny-Qwen3ForCausalLM"
TOOL_RESULT = "TOOLRESULT sunny and 21 degrees"
FINAL = "FINALANSWER it is sunny"


@pytest.fixture(scope = "module")
def tokenizer():
    try:
        tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    except Exception as e:
        pytest.skip(f"{TOKENIZER} unavailable: {e}")
    assert "generation" in tok.chat_template
    return tok


class Args:
    max_length = 512
    dataset_text_field = "text"
    remove_unused_columns = True
    assistant_only_loss = True
    completion_only_loss = True


class Trainer:
    def __init__(self):
        self.model = None
        self.completion_only_loss = True
        self.data_collator = trl_sft.DataCollatorForLanguageModeling(pad_token_id = 0)


def rows():
    tool_call = {"type": "function", "function": {"name": "weather", "arguments": {"city": "Paris"}}}
    return [{
        "prompt": [{"role": "user", "content": "Weather in Paris?"}],
        "completion": [
            {"role": "assistant", "content": "", "tool_calls": [tool_call]},
            {"role": "tool", "content": TOOL_RESULT},
            {"role": "assistant", "content": FINAL},
        ],
    }]


def supervised_text(dataset, tok):
    kwargs = {"pad_token_id": 0}
    if "completion_only_loss" in trl_sft.DataCollatorForLanguageModeling.__dataclass_fields__:
        kwargs["completion_only_loss"] = True
    batch = trl_sft.DataCollatorForLanguageModeling(**kwargs)([dataset[0]])
    ids = [t for t, y in zip(batch["input_ids"][0].tolist(), batch["labels"][0].tolist()) if y != -100]
    return tok.decode(ids)


@pytest.mark.parametrize("markers", ["generation", "fallback"])
def test_prompt_completion_tool_turn_not_supervised(tokenizer, markers, monkeypatch):
    monkeypatch.setattr(tokenizer, "chat_template", tokenizer.chat_template)
    if markers == "fallback":
        tokenizer.chat_template = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", tokenizer.chat_template)
    out = sft_prepare_dataset(Trainer(), Dataset.from_list(rows()), tokenizer, Args(), False, None, "train")
    text = supervised_text(out, tokenizer)
    assert FINAL in text
    assert "weather" in text
    assert "sunny and 21" not in text
    assert "Weather in Paris" not in text


def test_prompt_completion_masks_truncated_with_ids(tokenizer):
    args = Args()
    args.max_length = 20
    out = sft_prepare_dataset(Trainer(), Dataset.from_list(rows()), tokenizer, args, False, None, "train")
    row = out[0]
    assert len(row["input_ids"]) == 20
    for column in ("completion_mask", "assistant_masks", "labels"):
        if column in row:
            assert len(row[column]) == 20


def test_plain_prompt_completion_rejects_assistant_only_loss(tokenizer):
    dataset = Dataset.from_list([{"prompt": "Hi", "completion": " there"}])
    with pytest.raises(ValueError, match = "assistant_only_loss"):
        sft_prepare_dataset(Trainer(), dataset, tokenizer, Args(), False, None, "train")


def test_prompt_completion_without_assistant_only_loss_unchanged(tokenizer):
    args = Args()
    args.assistant_only_loss = False
    out = sft_prepare_dataset(Trainer(), Dataset.from_list(rows()), tokenizer, args, False, None, "train")
    assert "assistant_masks" not in out.column_names
    assert "sunny and 21" in supervised_text(out, tokenizer)


def test_prompt_completion_row_without_effective_target_is_dropped(tokenizer):
    # The prompt's historical assistant turn is masked in, but the completion is cut away entirely.
    long_row = {
        "prompt": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello there"},
            {"role": "user", "content": "word " * 200},
        ],
        "completion": [{"role": "assistant", "content": FINAL}],
    }
    args = Args()
    args.max_length = 64
    out = sft_prepare_dataset(Trainer(), Dataset.from_list([long_row] + rows()), tokenizer, args, False, None, "train")
    assert len(out) == 1

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

"""messages and prompt/completion datasets render one chat template call per batch, with the same token ids,
masks, row order and errors as one call per row. CPU-only."""

import json
from types import SimpleNamespace

import pytest

from unsloth_zoo.dataset_utils import sft_prepare_dataset

transformers = pytest.importorskip("transformers")
datasets = pytest.importorskip("datasets")

TOOLS = [{"type": "function", "function": {"name": "mul", "description": "Multiply.", "parameters": {
    "type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}}}]


def _tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
    except Exception as e:  # offline runner
        pytest.skip(f"cannot load tokenizer: {e}")


def _conversation(i):
    return [
        {"role": "user", "content": f"Question {i}: what is {i} plus {i}? " * (1 + i % 3)},
        {"role": "assistant", "content": f"The answer is {2 * i}. " * (1 + i % 4)},
    ]


def _prepare(tokenizer, rows, assistant_only_loss = False, max_length = 4096):
    trainer = SimpleNamespace(
        chat_template = None, _unsloth_assistant_mask_fallback = True, data_collator = None,
        model = None, completion_only_loss = None,
    )
    args = SimpleNamespace(
        max_length = max_length, dataset_text_field = "text", remove_unused_columns = True,
        assistant_only_loss = assistant_only_loss, dataset_num_proc = 1, packing_strategy = "bfd",
        truncation_mode = "keep_start",
    )
    return sft_prepare_dataset(trainer, datasets.Dataset.from_list(rows), tokenizer, args, False, None, "train")


class _Counter:
    def __init__(self, tokenizer, monkeypatch):
        self.calls = 0
        original = tokenizer.apply_chat_template
        def counted(*args, **kwargs):
            self.calls += 1
            return original(*args, **kwargs)
        monkeypatch.setattr(tokenizer, "apply_chat_template", counted)


def test_messages_rows_share_one_template_call_per_batch(monkeypatch):
    tokenizer = _tokenizer()
    rows = [{"messages": _conversation(i)} for i in range(64)]
    expected = [tokenizer.apply_chat_template(r["messages"], tokenize = True, return_dict = True)["input_ids"] for r in rows]
    counter = _Counter(tokenizer, monkeypatch)
    out = _prepare(tokenizer, rows)
    assert out["input_ids"] == [list(x)[:4096] for x in expected]
    assert counter.calls == 1, counter.calls


def test_prompt_completion_rows_share_template_calls(monkeypatch):
    tokenizer = _tokenizer()
    rows = [{"prompt": _conversation(i)[:1], "completion": _conversation(i)[1:]} for i in range(64)]
    counter = _Counter(tokenizer, monkeypatch)
    out = _prepare(tokenizer, rows)
    assert counter.calls == 2, counter.calls
    for row, ids, mask in zip(rows, out["input_ids"], out["completion_mask"]):
        full = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize = True, return_dict = True)["input_ids"]
        prompt = tokenizer.apply_chat_template(row["prompt"], tokenize = True, add_generation_prompt = True)
        assert ids == list(full)
        assert mask == [0] * len(prompt) + [1] * (len(full) - len(prompt))


@pytest.mark.parametrize("assistant_only_loss", [False, True])
def test_rows_with_tools_or_template_kwargs_match_one_call_per_row(assistant_only_loss):
    tokenizer = _tokenizer()
    rows = []
    for i in range(24):
        rows.append({
            "messages": _conversation(i),
            "tools": json.dumps(TOOLS) if i % 3 == 1 else None,
            "chat_template_kwargs": {"enable_thinking": False} if i % 4 == 2 else None,
        })
    out = _prepare(tokenizer, rows, assistant_only_loss = assistant_only_loss)
    assert len(out) == len(rows)
    for row, ids in zip(rows, out["input_ids"]):
        kwargs = dict(row["chat_template_kwargs"] or {})
        tools = json.loads(row["tools"]) if row["tools"] else None
        full = tokenizer.apply_chat_template(row["messages"], tools = tools, tokenize = True, return_dict = True, **kwargs)
        assert ids == list(full["input_ids"])


def test_a_failing_row_raises_the_row_by_row_error():
    tokenizer = _tokenizer()
    rows = [{"messages": _conversation(i)} for i in range(8)]
    rows[5]["messages"] = []
    with pytest.raises(IndexError):
        _prepare(tokenizer, rows)

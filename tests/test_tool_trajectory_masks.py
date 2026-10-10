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

"""Label masks for multi-turn tool-calling (agent trajectory) SFT.

Assistant tool calls and answers must be trained; tool results, user and system turns masked.
Covers ``train_on_responses_only`` on real chat templates and the ``messages`` dataset path of
``sft_prepare_dataset`` (``assistant_only_loss`` with TRL's training template, and the
train_on_responses_only fallback used when TRL has no training template for the tokenizer).
"""

from types import SimpleNamespace

import pytest

from unsloth_zoo.dataset_utils import sft_prepare_dataset, train_on_responses_only

transformers = pytest.importorskip("transformers")
datasets = pytest.importorskip("datasets")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two integers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        },
    }
]


def _call(a, b):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "multiply", "arguments": {"a": a, "b": b}}}],
    }


CONVERSATION = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 77 * 88? USERQ_1"},
    _call(77, 88),
    {"role": "tool", "name": "multiply", "content": "TOOLOUT_9137"},
    {"role": "assistant", "content": "FINALANS_5521"},
    {"role": "user", "content": "And 12 * 13? USERQ_2"},
    _call(12, 13),
    {"role": "tool", "name": "multiply", "content": "TOOLOUT_4242"},
    {"role": "assistant", "content": "FINALANS_7777"},
]


def _tokenizer(repo):
    try:
        return transformers.AutoTokenizer.from_pretrained(repo)
    except Exception as e:  # offline runner
        pytest.skip(f"cannot load {repo}: {e}")


def _assert_trajectory_masks(trained):
    assert '"a": 77' in trained and '"b": 88' in trained, trained
    assert '"a": 12' in trained and '"b": 13' in trained, trained
    assert "FINALANS_5521" in trained and "FINALANS_7777" in trained, trained
    assert "TOOLOUT" not in trained, trained
    assert "USERQ" not in trained, trained
    assert "You are helpful" not in trained, trained


def _trained_text(tokenizer, input_ids, labels):
    return tokenizer.decode([t for t, label in zip(input_ids, labels) if label != -100])


@pytest.mark.parametrize("repo", ["Qwen/Qwen3-0.6B", "unsloth/Qwen3-0.6B", "unsloth/Llama-3.2-1B-Instruct"])
def test_train_on_responses_only_trains_tool_calls(repo):
    tokenizer = _tokenizer(repo)
    text = tokenizer.apply_chat_template(CONVERSATION, tools = TOOLS, tokenize = False)
    input_ids = tokenizer(text, add_special_tokens = False)["input_ids"]
    labels = train_on_responses_only(None, tokenizer = tokenizer, return_function = True)(
        {"input_ids": [input_ids]}
    )["labels"][0]
    _assert_trajectory_masks(_trained_text(tokenizer, input_ids, labels))


def test_gpt_oss_role_marker_is_not_cut_to_shared_opener():
    # "<|start|>assistant" used to resolve to the bare "<|start|>" opener ("assistant" was treated as an
    # optional BPE edge), so every system, user and tool message counted as a response.
    tokenizer = _tokenizer("unsloth/gpt-oss-20b")
    text = tokenizer.apply_chat_template(CONVERSATION, tools = TOOLS, tokenize = False)
    input_ids = tokenizer(text, add_special_tokens = False)["input_ids"]
    labels = train_on_responses_only(
        None,
        tokenizer = tokenizer,
        return_function = True,
        instruction_part = "<|start|>user<|message|>",
        response_part = "<|start|>assistant",
    )({"input_ids": [input_ids]})["labels"][0]
    trained = _trained_text(tokenizer, input_ids, labels)
    _assert_trajectory_masks(trained)
    assert "<|call|>" in trained  # the model must learn to end a tool call


def _prepare(tokenizer, chat_template = None, fallback = False, assistant_only_loss = True, text = None, max_length = 4096, return_dataset = False):
    rows = [{"messages": CONVERSATION, "tools": TOOLS}] * 2
    if text is not None:
        rows = [dict(row, text = text) for row in rows]
    trainer = SimpleNamespace(
        chat_template = chat_template,
        _unsloth_assistant_mask_fallback = fallback,
        data_collator = None,
    )
    args = SimpleNamespace(
        max_length = max_length,
        dataset_text_field = "text",
        dataset_num_proc = None,
        assistant_only_loss = assistant_only_loss,
        completion_only_loss = None,
        packing = False,
    )
    dataset = sft_prepare_dataset(
        trainer, datasets.Dataset.from_list(rows), tokenizer, args, False, None, "train"
    )
    if return_dataset:
        return dataset
    row = dataset[0]
    if "labels" in row:
        labels = row["labels"]
    elif "assistant_masks" in row:
        labels = [t if m else -100 for t, m in zip(row["input_ids"], row["assistant_masks"])]
    else:
        labels = list(row["input_ids"])
    return row["input_ids"], labels


def test_messages_dataset_assistant_only_loss_with_trl_training_template():
    chat_template_utils = pytest.importorskip("trl.chat_template_utils")
    if not hasattr(chat_template_utils, "get_training_chat_template"):
        pytest.skip("TRL without get_training_chat_template")
    tokenizer = _tokenizer("Qwen/Qwen3-0.6B")
    training_template = chat_template_utils.get_training_chat_template(tokenizer)
    input_ids, labels = _prepare(tokenizer, chat_template = training_template)
    _assert_trajectory_masks(_trained_text(tokenizer, input_ids, labels))


@pytest.mark.parametrize(
    "repo", ["unsloth/Qwen3-0.6B", "unsloth/Llama-3.2-1B-Instruct", "unsloth/gpt-oss-20b"]
)
def test_messages_dataset_assistant_only_loss_marker_fallback(repo):
    tokenizer = _tokenizer(repo)
    input_ids, labels = _prepare(tokenizer, fallback = True)
    _assert_trajectory_masks(_trained_text(tokenizer, input_ids, labels))


def test_messages_dataset_without_assistant_only_loss_trains_everything():
    tokenizer = _tokenizer("unsloth/Qwen3-0.6B")
    input_ids, labels = _prepare(tokenizer, assistant_only_loss = False)
    expected = tokenizer.apply_chat_template(CONVERSATION, tools = TOOLS, tokenize = True, return_dict = True)["input_ids"]
    assert input_ids == list(expected)
    assert all(label != -100 for label in labels)


@pytest.mark.parametrize("repo", ["unsloth/Qwen3-0.6B", "unsloth/Llama-3.2-1B-Instruct"])
def test_messages_dataset_assistant_only_loss_without_generation_markers(repo):
    # TRL before 1.7 never swaps a training template and Unsloth sets no fallback flag there, so a
    # template without {% generation %} must still mask by markers instead of training on nothing.
    tokenizer = _tokenizer(repo)
    input_ids, labels = _prepare(tokenizer, chat_template = None, fallback = False)
    _assert_trajectory_masks(_trained_text(tokenizer, input_ids, labels))


def test_assistant_only_loss_reads_messages_beside_a_text_column():
    tokenizer = _tokenizer("unsloth/Qwen3-0.6B")
    text = tokenizer.apply_chat_template(CONVERSATION, tools = TOOLS, tokenize = False)
    input_ids, labels = _prepare(tokenizer, fallback = True, text = text)
    _assert_trajectory_masks(_trained_text(tokenizer, input_ids, labels))
    # Without assistant_only_loss the text column is still what gets tokenized, as before.
    input_ids, labels = _prepare(tokenizer, assistant_only_loss = False, text = "plain text row")
    assert tokenizer.decode(input_ids).endswith("plain text row")


def test_rows_truncated_before_any_assistant_token_are_dropped(monkeypatch):
    tokenizer = _tokenizer("unsloth/Qwen3-0.6B")
    long_system = {"role": "system", "content": "Be careful. " * 400}
    rows = [
        {"messages": [long_system] + CONVERSATION[1:], "tools": TOOLS},
        {"messages": CONVERSATION, "tools": TOOLS},
    ]
    monkeypatch.setattr(datasets.Dataset, "from_list", classmethod(lambda cls, _: _ORIGINAL_FROM_LIST(rows)))
    dataset = _prepare(tokenizer, fallback = True, max_length = 1024, return_dataset = True)
    assert len(dataset) == 1
    row = dataset[0]
    assert any(row["assistant_masks"]) if "assistant_masks" in row else any(l != -100 for l in row["labels"])


_ORIGINAL_FROM_LIST = datasets.Dataset.from_list


@pytest.mark.parametrize("fallback", [True, False])
def test_assistant_only_loss_refuses_a_formatting_func(fallback):
    tokenizer = _tokenizer("unsloth/Qwen3-0.6B")
    trainer = SimpleNamespace(chat_template = None, _unsloth_assistant_mask_fallback = fallback, data_collator = None)
    args = SimpleNamespace(
        max_length = 4096, dataset_text_field = "text", dataset_num_proc = None,
        assistant_only_loss = True, completion_only_loss = None, packing = False,
    )
    rows = datasets.Dataset.from_list([{"messages": CONVERSATION, "tools": TOOLS}] * 2)
    formatting = lambda row: tokenizer.apply_chat_template(row["messages"], tokenize = False)
    with pytest.raises(ValueError, match = "assistant_only_loss"):
        sft_prepare_dataset(trainer, rows, tokenizer, args, False, formatting, "train")


def test_named_template_set_does_not_break_marker_detection():
    tokenizer = _tokenizer("unsloth/Llama-3.2-1B-Instruct")
    tokenizer.chat_template = {"default": tokenizer.chat_template, "tool_use": tokenizer.chat_template}
    input_ids, labels = _prepare(tokenizer, chat_template = None, fallback = False)
    _assert_trajectory_masks(_trained_text(tokenizer, input_ids, labels))

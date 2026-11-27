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

"""train_on_responses_only around tool turns: GLM <|observation|>, Mistral [/TOOL_RESULTS], gpt-oss tool calls."""

import re

import pytest

from unsloth_zoo.dataset_utils import train_on_responses_only


class _AddedToken:
    def __init__(self, content):
        self.content, self.special = content, True


class _Tokenizer:
    bos_token_id = None
    all_special_ids = ()

    def __init__(self, markers, eos = None):
        self.markers = {marker: i for i, marker in enumerate(markers, start = 1)}
        self.eos_token_id = self.markers.get(eos)
        self.added_tokens_decoder = {i: _AddedToken(marker) for marker, i in self.markers.items()}
        self.pattern = "|".join(map(re.escape, self.markers)) + "|(?s:.)"

    def __call__(self, text, add_special_tokens = False):
        ids = [self.markers.get(p) or 100 + ord(p) for p in re.findall(self.pattern, text)]
        return type("Encoding", (), {"input_ids": ids})()

    def convert_ids_to_tokens(self, ids):
        return str(ids)

    def decode(self, ids):
        names = {i: marker for marker, i in self.markers.items()}
        return "".join(names.get(i) or chr(i - 100) for i in ids)


def _trained(markers, text, instruction_part, response_part, eos = None, **kwargs):
    tokenizer = _Tokenizer(markers, eos)
    masker = train_on_responses_only(
        None, tokenizer = tokenizer, return_function = True,
        instruction_part = instruction_part, response_part = response_part, **kwargs,
    )
    input_ids = tokenizer(text).input_ids
    labels = masker({"input_ids": [input_ids]})["labels"][0]
    return tokenizer.decode([label for label in labels if label != -100])


def test_glm_observation_ends_the_assistant_span():
    markers = ("<|user|>", "<|assistant|>", "<|observation|>")
    text = "<|user|>Q<|assistant|>CALL<|observation|>RESULT<|assistant|>ANSWER<|user|>Q2<|assistant|>B"
    assert _trained(markers, text, "<|user|>", "<|assistant|>") == "CALLANSWERB"


@pytest.mark.parametrize("last_response_only, expected", [(False, "CALL</s>ANSWER</s>"), (True, "ANSWER</s>")])
def test_mistral_answer_after_tool_results_is_trained(last_response_only, expected):
    markers = ("[INST]", "[/INST]", "[TOOL_RESULTS]", "[/TOOL_RESULTS]", "</s>")
    text = "[INST]Q[/INST]CALL</s>[TOOL_RESULTS]R1[/TOOL_RESULTS][TOOL_RESULTS]R2[/TOOL_RESULTS]ANSWER</s>[INST]Q2"
    assert _trained(
        markers, text, "[INST]", "[/INST]", eos = "</s>", last_response_only = last_response_only,
    ) == expected


TOOLS = [{"type": "function", "function": {
    "name": "multiply", "description": "Multiply.",
    "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
}}]
CALL = {"id": "abcdefghi", "type": "function", "function": {"name": "multiply", "arguments": {"a": 77, "b": 88}}}
TOOL_CHAT = [
    {"role": "user", "content": "USERQ"},
    {"role": "assistant", "content": "", "tool_calls": [CALL]},
    {"role": "tool", "tool_call_id": "abcdefghi", "name": "multiply", "content": "TOOLOUT_9137"},
    {"role": "assistant", "content": "FINALANS_5521"},
]


@pytest.mark.parametrize("repo, call_text", [
    ("unsloth/GLM-4.5-Air", "<arg_value>77</arg_value>"),
    ("unsloth/Mistral-Small-3.2-24B-Instruct-2506", '{"a": 77, "b": 88}'),
    ("unsloth/gpt-oss-20b", '{"a": 77, "b": 88}<|call|>'),
])
def test_cached_tokenizer_tool_conversation(repo, call_text):
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(repo)
    except Exception as e:  # offline runner
        pytest.skip(f"cannot load {repo}: {e}")
    text = tokenizer.apply_chat_template(TOOL_CHAT, tools = TOOLS, tokenize = False)
    input_ids = tokenizer(text, add_special_tokens = False).input_ids
    labels = train_on_responses_only(None, tokenizer = tokenizer, return_function = True)(
        {"input_ids": [input_ids]}
    )["labels"][0]
    trained = tokenizer.decode([t for t, label in zip(input_ids, labels) if label != -100])
    assert call_text in trained and "FINALANS_5521" in trained, trained
    assert "TOOLOUT" not in trained and "USERQ" not in trained, trained

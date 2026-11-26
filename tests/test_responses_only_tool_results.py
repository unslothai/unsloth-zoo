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

"""train_on_responses_only must not train on a tool result rendered inside the model turn (Gemma 4,
FunctionGemma). The model stops on the opening marker; the result and closing marker come from the tool."""

import re

import pytest

from unsloth_zoo.dataset_utils import train_on_responses_only

FAMILIES = {
    "gemma4": ("<|turn>", "<turn|>", "<|tool_response>", "<tool_response|>"),
    "functiongemma": ("<start_of_turn>", "<end_of_turn>", "<start_function_response>", "<end_function_response>"),
}


class _AddedToken:
    def __init__(self, content):
        self.content, self.special = content, True


class _Tokenizer:
    """One id per marker, one id per other character."""

    bos_token_id = eos_token_id = None
    all_special_ids = ()

    def __init__(self, markers):
        self.markers = {marker: i for i, marker in enumerate(markers, start = 1)}
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


def _trained(family, text):
    """The conversation with every untrained token dropped."""
    markers = FAMILIES[family]
    tokenizer = _Tokenizer(markers)
    turn = markers[0]
    masker = train_on_responses_only(
        None, tokenizer = tokenizer, return_function = True,
        instruction_part = f"{turn}user\n", response_part = f"{turn}model\n",
    )
    input_ids = tokenizer(text).input_ids
    labels = masker({"input_ids": [input_ids]})["labels"][0]
    return tokenizer.decode([label for label in labels if label != -100])


@pytest.mark.parametrize("family", FAMILIES)
def test_tool_results_are_not_trained(family):
    turn, end, opener, closer = FAMILIES[family]
    text = (
        f"{turn}user\nQ{end}\n{turn}model\nCALL1{opener}RESULT1{closer}{opener}RESULT2{closer}"
        f"ANSWER{end}\n"
    )
    assert _trained(family, text) == f"CALL1{opener}{opener}ANSWER{end}\n"


@pytest.mark.parametrize("family", FAMILIES)
def test_unanswered_call_does_not_mask_later_turns(family):
    turn, end, opener, _ = FAMILIES[family]
    text = (
        f"{turn}user\nQ1{end}\n{turn}model\nCALL1{opener}{end}\n"
        f"{turn}user\nQ2{end}\n{turn}model\nANSWER{end}\n"
    )
    assert _trained(family, text) == f"CALL1{opener}ANSWER{end}\n"


@pytest.mark.parametrize("family", FAMILIES)
def test_model_marker_quoted_by_a_result_stays_masked(family):
    turn, end, opener, closer = FAMILIES[family]
    text = f"{turn}user\nQ{end}\n{turn}model\nCALL{opener}A{turn}model\nB{closer}ANSWER{end}\n"
    assert _trained(family, text) == f"CALL{opener}ANSWER{end}\n"


@pytest.mark.parametrize("repo, opener", [
    ("unsloth/gemma-4-E4B-it", "<|tool_response>"),
    ("unsloth/functiongemma-270m-it", "<start_function_response>"),
])
def test_cached_tokenizer_tool_conversation(repo, opener):
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(repo)
    except Exception as e:  # offline runner
        pytest.skip(f"cannot load {repo}: {e}")
    call = {"type": "function", "function": {"name": "multiply", "arguments": {"a": 77, "b": 88}}}
    messages = [
        {"role": "user", "content": "USERQ"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "name": "multiply", "content": "TOOLOUT_9137"},
        {"role": "assistant", "content": "FINALANS_5521"},
    ]
    tools = [{"type": "function", "function": {
        "name": "multiply", "description": "Multiply.",
        "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
    }}]
    input_ids = tokenizer.apply_chat_template(messages, tools = tools, tokenize = True, return_dict = False)
    labels = train_on_responses_only(None, tokenizer = tokenizer, return_function = True)(
        {"input_ids": [input_ids]}
    )["labels"][0]
    trained = tokenizer.decode([t for t, label in zip(input_ids, labels) if label != -100])
    assert "call:multiply" in trained and opener in trained and "FINALANS_5521" in trained, trained
    assert "TOOLOUT" not in trained and "USERQ" not in trained, trained

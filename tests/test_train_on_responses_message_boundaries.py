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

"""``train_on_responses_only`` must stop each response span at a message boundary.

The span loop used to run from the assistant marker to the next *user* marker, so a
tool conversation - which has no user turn in between - supervised the tool result
and the role headers around it, and after packing the last span ran on into the next
sample's BOS and system prompt.

Every fixture gives the turn terminator an id that is NOT ``eos_token_id``, so a span
stopping in the right place proves the message opener was recognised rather than the
EOS check having covered for it.

CPU-pure and offline: the tokenizers are local stubs, no weights are loaded.
"""

import pytest

from unsloth_zoo.dataset_utils import train_on_responses_only


class _AddedToken:
    """Stands in for ``tokenizers.AddedToken`` as ``added_tokens_decoder`` stores it."""

    def __init__(self, content, special):
        self.content = content
        self.special = special

    def __str__(self):
        return self.content


class StubTokenizer:
    """Whitespace-free lookup tokenizer with an explicit special-token registry."""

    def __init__(self, vocab, added, all_special_ids, bos_token_id, eos_token_id):
        self._vocab = vocab
        self.added_tokens_decoder = {i: _AddedToken(t, s) for i, (t, s) in added.items()}
        self.all_special_ids = list(all_special_ids)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def __call__(self, text, add_special_tokens=False):
        ids, rest = [], text
        while rest:
            for token, i in self._vocab.items():
                if rest.startswith(token):
                    ids.append(i)
                    rest = rest[len(token):]
                    break
            else:  # pragma: no cover - a fixture typo, not a code path
                raise AssertionError(f"stub tokenizer cannot encode {rest!r}")

        class _Result:
            input_ids = ids
        return _Result()


# ChatML: <|im_start|> is an added special but NOT an attribute special, the way Qwen,
# Llama 3, Gemma and Phi-4 store their openers. <|im_end|> ends a turn, <|endoftext|> is EOS.
IM_START, IM_END, NL, EOS, BOS = 1, 7, 6, 9, 8
USER, ASSISTANT, TOOL, SYSTEM = 2, 3, 4, 5
CHATML_VOCAB = {
    "<|im_start|>": IM_START, "<|im_end|>": IM_END, "<|endoftext|>": EOS,
    "<|bos|>": BOS, "user": USER, "assistant": ASSISTANT, "tool": TOOL,
    "system": SYSTEM, "\n": NL,
}
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"

QUESTION, TOOLCALL, TOOLRESULT, FINAL, SYSPROMPT = 101, 102, 103, 104, 105


def chatml_tokenizer():
    return StubTokenizer(
        vocab=CHATML_VOCAB,
        added={IM_START: ("<|im_start|>", True), IM_END: ("<|im_end|>", True),
               EOS: ("<|endoftext|>", True), BOS: ("<|bos|>", True)},
        all_special_ids=[EOS, BOS],   # what transformers actually reports
        bos_token_id=BOS, eos_token_id=EOS,
    )


def turn(role, body):
    return [IM_START, role, NL, body, IM_END, NL]


def supervised(input_ids, labels):
    """The ids that carry a target, as a set of positions and as ids."""
    return [i for i, l in zip(input_ids, labels) if l != -100]


def mask(tokenizer, input_ids, last_response_only=False, labels=None):
    fn = train_on_responses_only(
        None, INSTRUCTION_PART, RESPONSE_PART, tokenizer=tokenizer,
        return_function=True, last_response_only=last_response_only,
    )
    examples = {"input_ids": [list(input_ids)]}
    if labels is not None: examples["labels"] = [list(labels)]
    return fn(examples)["labels"][0]


def test_a_tool_result_is_not_supervised():
    """assistant -> tool -> assistant: the tool body and the headers stay masked."""
    tokenizer = chatml_tokenizer()
    row = (turn(USER, QUESTION) + turn(ASSISTANT, TOOLCALL) +
           turn(TOOL, TOOLRESULT) + turn(ASSISTANT, FINAL) + [EOS])

    labels = mask(tokenizer, row)

    assert TOOLRESULT not in supervised(row, labels), "the tool result was trained on"
    assert TOOLCALL in supervised(row, labels)
    assert FINAL in supervised(row, labels)
    # No role header may be supervised: every <|im_start|> is a message opener.
    for position, token in enumerate(row):
        if token == IM_START:
            assert labels[position] == -100, f"role header at {position} was trained on"


def test_last_response_only_takes_the_final_assistant_turn():
    """The earlier assistant turn must not ride along inside the last span."""
    tokenizer = chatml_tokenizer()
    row = (turn(USER, QUESTION) + turn(ASSISTANT, TOOLCALL) +
           turn(TOOL, TOOLRESULT) + turn(ASSISTANT, FINAL) + [EOS])

    labels = mask(tokenizer, row, last_response_only=True)

    trained = supervised(row, labels)
    assert FINAL in trained
    assert TOOLCALL not in trained, "an earlier assistant message was retained"
    assert TOOLRESULT not in trained


def test_consecutive_assistant_messages_are_separate_spans():
    tokenizer = chatml_tokenizer()
    row = turn(USER, QUESTION) + turn(ASSISTANT, TOOLCALL) + turn(ASSISTANT, FINAL) + [EOS]

    labels = mask(tokenizer, row)

    assert TOOLCALL in supervised(row, labels)
    assert FINAL in supervised(row, labels)
    # The second assistant's own header sits between them and must stay masked.
    assert labels[row.index(ASSISTANT, row.index(ASSISTANT) + 1)] == -100


def test_a_concatenated_next_sample_is_not_supervised():
    """Masking after packing must not run the last span into the next sample."""
    tokenizer = chatml_tokenizer()
    first = turn(USER, QUESTION) + turn(ASSISTANT, FINAL)
    second = [BOS] + turn(SYSTEM, SYSPROMPT) + turn(USER, QUESTION) + turn(ASSISTANT, TOOLCALL)

    labels = mask(tokenizer, first + second)

    trained = supervised(first + second, labels)
    assert SYSPROMPT not in trained, "the next sample's system prompt was trained on"
    assert BOS not in trained
    assert trained.count(FINAL) == 1 and TOOLCALL in trained


def test_padding_that_reuses_eos_is_not_supervised():
    tokenizer = chatml_tokenizer()
    row = turn(USER, QUESTION) + turn(ASSISTANT, FINAL) + [EOS] * 6

    labels = mask(tokenizer, row)

    # The response keeps its own EOS target; the five pad copies do not.
    assert [l for l in labels if l == EOS] == [EOS]


def test_existing_labels_are_preserved_inside_a_retained_span():
    tokenizer = chatml_tokenizer()
    row = turn(USER, QUESTION) + turn(ASSISTANT, FINAL) + [EOS]
    existing = [-100] * len(row)
    existing[row.index(FINAL)] = FINAL

    labels = mask(tokenizer, row, labels=existing)

    assert labels == existing, "an existing label mask was overwritten"


# NVIDIA Nemotron: the markers are not atomic and their shared prefix is a bare "\n"
# that the tokenizer does register as a special token.
NEMO_NL, NEMO_USER, NEMO_ASSISTANT, NEMO_END, NEMO_EOS = 11, 12, 13, 14, 15
NEMO_ANSWER_A, NEMO_ANSWER_B = 201, 202


def test_a_whitespace_special_token_is_not_a_message_boundary():
    """A special "\\n" must not end the span, or multi-line answers get cut."""
    tokenizer = StubTokenizer(
        vocab={"\n": NEMO_NL, "User": NEMO_USER, "Assistant": NEMO_ASSISTANT,
               "<end>": NEMO_END, "<eos>": NEMO_EOS},
        added={NEMO_NL: ("\n", True), NEMO_END: ("<end>", True), NEMO_EOS: ("<eos>", True)},
        all_special_ids=[NEMO_EOS],
        bos_token_id=None, eos_token_id=NEMO_EOS,
    )
    fn = train_on_responses_only(
        None, "\nUser\n", "\nAssistant\n", tokenizer=tokenizer, return_function=True)
    # ...Assistant\n ANSWER_A \n ANSWER_B <end> <eos>: the answer spans a newline.
    row = [NEMO_NL, NEMO_USER, NEMO_NL, 200,
           NEMO_NL, NEMO_ASSISTANT, NEMO_NL, NEMO_ANSWER_A, NEMO_NL, NEMO_ANSWER_B,
           NEMO_END, NEMO_EOS]

    labels = fn({"input_ids": [list(row)]})["labels"][0]

    trained = supervised(row, labels)
    assert NEMO_ANSWER_A in trained
    assert NEMO_ANSWER_B in trained, "the answer was cut at an internal newline"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

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

"""train_on_responses_only must not fold a user/instruction marker token at the last position of
input_ids into the trained span; rows truncated right at a turn boundary end exactly like that.
CPU-only stub tokenizer."""

from unsloth_zoo.dataset_utils import train_on_responses_only

USER, ASST, NL, TXT = 10, 20, 30, 40


class StubTokenizer:
    """Whole-word id tokenizer: 'Q:' -> USER, 'A:' -> ASST, '' -> NL, else TXT."""

    bos_token_id = None
    eos_token_id = None
    added_tokens_decoder = {}

    def __call__(self, text, add_special_tokens = False, **kwargs):
        out = []
        for piece in text.split(" "):
            if piece == "Q:":
                out.append(USER)
            elif piece == "A:":
                out.append(ASST)
            elif piece == "":
                out.append(NL)
            else:
                out.extend([TXT] * len(piece))
        return type("Enc", (), {"input_ids": out})()


def run(rows):
    fn = train_on_responses_only(
        None,
        instruction_part = "Q:",
        response_part = "A:",
        tokenizer = StubTokenizer(),
        return_function = True,
    )
    return fn({"input_ids": rows})["labels"]


def test_trailing_user_marker_is_masked():
    # [A: t t t Q:] -- the trailing Q: opens a new user turn with nothing after it.
    labels = run([[ASST, 41, 42, 43, USER]])[0]
    assert list(labels) == [-100, 41, 42, 43, -100]


def test_trailing_user_marker_after_content_is_masked():
    labels = run([[ASST, 41, 42, NL, USER]])[0]
    assert list(labels) == [-100, 41, 42, NL, -100]


def test_plain_tail_without_marker_still_trains_to_the_end():
    labels = run([[ASST, 41, 42, 43]])[0]
    assert list(labels) == [-100, 41, 42, 43]

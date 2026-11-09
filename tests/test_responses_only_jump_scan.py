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

"""Plain-list rows take the list.index jump path; labels must equal the stepping path a list subclass takes."""

import random

import pytest
import torch

from unsloth_zoo.dataset_utils import train_on_responses_only

BOS, EOS, START, USER, ASSISTANT, END = 1, 2, 3, 4, 5, 6
VOCAB = {"<s>": BOS, "</s>": EOS, "<|start|>": START, "user": USER, "assistant": ASSISTANT, "<|end|>": END, ":": 7, "\n": 8}


class _AddedToken:
    def __init__(self, content, special):
        self.content, self.special = content, special


class _Tokenizer:
    def __init__(self, with_specials):
        self.added_tokens_decoder = {i: _AddedToken(t, True) for t, i in VOCAB.items() if i <= END} if with_specials else {}
        self.all_special_ids = [BOS, EOS] if with_specials else []
        self.bos_token_id = BOS if with_specials else None
        self.eos_token_id = EOS if with_specials else None

    def __call__(self, text, add_special_tokens = False):
        ids, rest = [], text
        while rest:
            for token, i in VOCAB.items():
                if rest.startswith(token):
                    ids.append(i); rest = rest[len(token):]; break
            else:
                ids.append(100 + ord(rest[0])); rest = rest[1:]

        class _Result:
            input_ids = ids
        return _Result()

    def convert_ids_to_tokens(self, ids):
        return [str(i) for i in ids]


class _Row(list):
    pass


MARKERS = [
    ("<|start|>user", "<|start|>assistant"),
    ("<|start|>user\n", "<|start|>assistant:\n"),
    ("user:", "assistant:"),
    ("<|end|><|start|>user", "<|end|><|start|>assistant"),
    ("<|start|>", "<|start|>"),
]


def _rows(seed, pieces, count = 300):
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        row = []
        for _ in range(rng.randint(0, 30)):
            if rng.random() < 0.5:
                piece = rng.choice(pieces)
                row += piece[:rng.randint(0, len(piece))] if rng.random() < 0.3 else piece
            else:
                row += [rng.randrange(1, 12) for _ in range(rng.randint(1, 4))]
        rows.append(row)
    return rows


@pytest.mark.parametrize("with_specials", [True, False])
@pytest.mark.parametrize("markers", MARKERS)
@pytest.mark.parametrize("last_response_only", [False, True])
@pytest.mark.parametrize("force_match", [True, False])
def test_jump_scan_matches_stepping_scan(with_specials, markers, last_response_only, force_match):
    tok = _Tokenizer(with_specials)
    fn = train_on_responses_only(
        None, markers[0], markers[1], tokenizer = tok, return_function = True,
        force_match = force_match, last_response_only = last_response_only, mask_out_tokens = [":"],
    )
    pieces = [tok(m).input_ids for m in markers] + [[BOS], [EOS], [START], [END]]
    rows = _rows(repr((with_specials, markers, last_response_only, force_match)), pieces)
    fast = fn({"input_ids": rows})["labels"]
    slow = fn({"input_ids": [_Row(r) for r in rows]})["labels"]
    assert fast == slow
    assert any(v != -100 for row in fast for v in row)

    old = [[v if i % 3 else -100 for i, v in enumerate(r)] for r in rows]
    assert fn({"input_ids": rows, "labels": old})["labels"] == fn({"input_ids": [_Row(r) for r in rows], "labels": old})["labels"]


def test_tensor_batch_still_returns_int64_tensor():
    tok = _Tokenizer(True)
    fn = train_on_responses_only(None, "<|start|>user", "<|start|>assistant", tokenizer = tok, return_function = True)
    rows = [[BOS, START, USER, 9, 9, START, ASSISTANT, 10, 11, EOS], [START, ASSISTANT, 10, START, USER, 9, START, ASSISTANT, 11, 11]]
    out = fn({"input_ids": torch.tensor(rows)})["labels"]
    assert out.dtype == torch.int64
    assert out.tolist() == fn({"input_ids": [_Row(r) for r in rows]})["labels"]
    assert out.tolist() == [[-100] * 7 + [10, 11, EOS], [-100, -100, 10, -100, -100, -100, -100, -100, 11, 11]]

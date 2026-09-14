# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""``_find_common_token_ids`` when the matched core is not found in the component.

The core is the common sublist across probe variants that each carry an appended
``[0]`` sentinel, so it need not appear in ``original``. On no match the loop fell
through and sliced at the last index, returning optional tokens that were never
matched; on an empty tokenization ``j`` was never bound at all.

An unlocated core is also never returned: it can be the bare sentinel (whose first
id 0 would then mask on every ``<unk>``/``<pad>`` in the corpus) or a real id with
the sentinel glued on (which masks the whole dataset). Both come back as no core,
and ``train_on_responses_only`` turns that into a named error.
"""

import pytest

from unsloth_zoo.dataset_utils import _find_common_token_ids, train_on_responses_only


class StubTokenizer:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def __call__(self, text, add_special_tokens=False):
        class _Result:
            pass

        result = _Result()
        result.input_ids = (
            list(self.mapping[text]) if text in self.mapping else [ord(c) for c in text]
        )
        return result


# Variants share only the [0] sentinel, so the core is [0] - absent from "X" ([1, 2]).
NO_COMMON_CORE = {
    "X": [1, 2],
    "\nX": [9, 1, 2],
    "\nX\n": [9, 1, 2, 9],
    "X\n": [1, 2, 9],
    "\n\nX": [9, 9, 1, 2],
}


def test_core_absent_from_component_reports_no_core():
    """An unlocated core is reported as no core, with no optional context."""
    tokenizer = StubTokenizer(NO_COMMON_CORE)

    substring, optional_left, optional_right = _find_common_token_ids("X", tokenizer, False)

    original = tokenizer("X").input_ids
    assert original, "fixture no longer exercises the no-match path; update it"
    matched = bool(substring) and any(
        original[j : j + len(substring)] == substring for j in range(len(original))
    )
    assert not matched, "fixture no longer exercises the no-match path; update it"
    assert substring == [], (
        f"the core was never located in {original}, so it must not be reported as a "
        f"match, but the helper returned {substring} - its first id becomes A_first / "
        "Q_first and masks on every token that happens to carry it"
    )
    assert optional_left == [] and optional_right == []


def test_component_tokenizing_to_nothing_does_not_raise_unbound_local():
    """An empty tokenization must not blow up on an unbound loop variable."""
    tokenizer = StubTokenizer()

    try:
        substring, optional_left, optional_right = _find_common_token_ids("", tokenizer, True)
    except UnboundLocalError as exception:  # pragma: no cover - the bug being fixed
        pytest.fail(f"empty component raised UnboundLocalError: {exception}")

    assert substring == []
    assert optional_left == [] and optional_right == []


def test_empty_explicit_marker_raises_named_error():
    """Empty explicit marker -> named error. Auto-detect cannot reach this."""
    tokenizer = StubTokenizer()

    with pytest.raises(ValueError, match="response_part could not be resolved"):
        train_on_responses_only(
            None, "user:", "", tokenizer=tokenizer, return_function=True
        )

    with pytest.raises(ValueError, match="instruction_part could not be resolved") as caught:
        train_on_responses_only(
            None, "", "assistant:", tokenizer=tokenizer, return_function=True
        )
    # force_match defaults to True here, so retrying with it is not a fix worth naming.
    assert "force_match" not in str(caught.value)


def test_unlocated_core_raises_before_a_masking_function_exists():
    """The caller-level consequence: a marker whose core was never located must not
    reach the masking closure. Its core used to come back as the bare [0] sentinel,
    so A_first == 0 and a sample of [41, 0, 42, 0, 43] was masked to
    [-100, -100, 42, -100, -100] - token 42 trained on although neither marker was
    ever recovered. Raising here means no such closure is ever built."""
    tokenizer = StubTokenizer(NO_COMMON_CORE)

    with pytest.raises(ValueError, match="instruction_part could not be resolved") as caught:
        train_on_responses_only(
            None,
            "X",
            "X",
            force_match=False,
            tokenizer=tokenizer,
            return_function=True,
        )
    # This one IS fixable by force_match=True, per the test below, so say so.
    assert "force_match = True" in str(caught.value)


def test_same_marker_with_force_match_still_builds_a_masking_function():
    """No-regression guard: only the unrecoverable case is rejected. The same marker
    at force_match=True has a core equal to its own tokenization, so it still works."""
    tokenizer = StubTokenizer(NO_COMMON_CORE)

    mask_fn = train_on_responses_only(
        None, "X", "X", force_match=True, tokenizer=tokenizer, return_function=True
    )

    assert callable(mask_fn)
    assert _find_common_token_ids("X", tokenizer, True) == ([1, 2], [], [])


def test_normal_marker_still_recovers_optional_context():
    """When the core is found, surrounding tokens are still returned."""
    tokenizer = StubTokenizer()

    substring, optional_left, optional_right = _find_common_token_ids("\nAB\n", tokenizer, False)

    original = tokenizer("\nAB\n").input_ids
    assert substring, "expected a non-empty core for a normal marker"
    where = next(
        j for j in range(len(original)) if original[j : j + len(substring)] == substring
    )
    assert optional_left == original[:where]
    assert optional_right == original[where + len(substring) :]


def test_force_match_marker_matches_at_index_zero():
    """force_match=True: core is the whole tokenization, no optional context."""
    tokenizer = StubTokenizer()

    substring, optional_left, optional_right = _find_common_token_ids("AB", tokenizer, True)

    assert substring == tokenizer("AB").input_ids
    assert optional_left == [] and optional_right == []

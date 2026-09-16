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

"""Pick the role column by alias membership, not a cardinality tie, and match aliases
regardless of case or whitespace (unsloth#1766)."""

import pytest
from datasets import Dataset

from unsloth_zoo.dataset_utils import standardize_data_formats


CASES = {
    "two_row_tie": [
        {"conversations": [{"from": "human", "value": "Braund, Mr. Owen Harris"},
                           {"from": "gpt", "value": "0"}]},
        {"conversations": [{"from": "human", "value": "Braund, Mr. Owen Harris"},
                           {"from": "gpt", "value": "0"}]},
    ],
    "one_row_tie": [
        {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]},
    ],
    "role_content": [
        {"conversations": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]},
    ],
    "system_first": [
        {"conversations": [{"from": "system", "value": "s"}, {"from": "human", "value": "hi"},
                           {"from": "gpt", "value": "yo"}]},
    ],
    "single_turn": [
        {"conversations": [{"from": "human", "value": "hi"}]},
    ],
    "mixed_case_tie": [
        {"conversations": [{"from": "Human", "value": "hi"}, {"from": "GPT", "value": "yo"}]},
    ],
    "mixed_case_role_content": [
        {"conversations": [{"role": "User", "content": "hi"},
                           {"role": "Assistant", "content": "yo"}]},
    ],
    "padded_role_tie": [
        {"conversations": [{"from": "user ", "value": "hi"}, {"from": " assistant", "value": "yo"}]},
    ],
    "mixed_case_and_padded_many_turns": [
        {"conversations": [{"from": " System", "value": "s"}, {"from": "HUMAN ", "value": "hi"},
                           {"from": "Gpt", "value": "yo"}]},
    ],
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_role_column_is_resolved_by_alias(name):
    out = standardize_data_formats(Dataset.from_list(CASES[name]))
    for row in out:
        for message in row["conversations"]:
            assert message["role"] in ("system", "user", "assistant"), message
            assert message["content"] not in ("system", "user", "assistant", "human", "gpt")


def test_many_row_dataset_still_resolves():
    rows = [
        {"conversations": [{"from": "human", "value": f"q{i}"}, {"from": "gpt", "value": f"a{i}"}]}
        for i in range(20)
    ]
    out = standardize_data_formats(Dataset.from_list(rows))
    assert [m["role"] for m in out[0]["conversations"]] == ["user", "assistant"]


def test_mixed_case_roles_resolve_on_a_large_dataset():
    """No tie here: the alias check itself compared "Human" against a lowercase list."""
    rows = [
        {"conversations": [{"from": "Human", "value": f"q{i}"}, {"from": "GPT", "value": f"a{i}"}]}
        for i in range(20)
    ]
    out = standardize_data_formats(Dataset.from_list(rows))
    assert [m["role"] for m in out[0]["conversations"]] == ["user", "assistant"]


def test_caller_supplied_aliases_are_also_normalized():
    rows = [
        {"conversations": [{"from": "Questioner ", "value": "hi"},
                           {"from": "ANSWERER", "value": "yo"}]},
    ]
    out = standardize_data_formats(
        Dataset.from_list(rows),
        aliases_for_user      = ["Questioner"],
        aliases_for_assistant = [" answerer "],
    )
    assert [m["role"] for m in out[0]["conversations"]] == ["user", "assistant"]


def test_an_unknown_role_still_raises():
    """Normalization must not turn the alias check into a no-op."""
    rows = [
        {"conversations": [{"from": "moderator", "value": "hi"},
                           {"from": "moderator", "value": "yo"}]},
    ]
    with pytest.raises(TypeError):
        standardize_data_formats(Dataset.from_list(rows))


def test_a_non_sharegpt_dataset_is_returned_untouched():
    rows = [{"text": "hello"}]
    dataset = Dataset.from_list(rows)
    assert standardize_data_formats(dataset) is dataset


@pytest.mark.parametrize("start_method", ["fork", "spawn"])
def test_the_resolved_columns_survive_a_multiprocess_map(start_method):
    """What `_standardize_dataset` closed over has to reach the workers. Windows has no
    fork, so spawn is what runs there, and spawn is where a closure can fail to arrive."""
    multiprocess = pytest.importorskip("multiprocess")

    previous = multiprocess.get_start_method(allow_none = True)
    try:
        multiprocess.set_start_method(start_method, force = True)
    except (RuntimeError, ValueError) as error:
        pytest.skip(f"{start_method} is unavailable here: {error}")
    try:
        rows = [
            {"conversations": [{"from": "Human ", "value": "Braund, Mr. Owen Harris"},
                               {"from": "GPT", "value": "0"}]},
        ] * 60
        out = standardize_data_formats(Dataset.from_list(rows), num_proc = 2)
        assert [m["role"] for m in out[0]["conversations"]] == ["user", "assistant"]
        assert out[0]["conversations"][0]["content"] == "Braund, Mr. Owen Harris"
    finally:
        # `previous is None` is the normal state of a fresh process, and leaving the
        # parametrised method pinned makes every later test order-dependent.
        # `set_start_method(None, force = True)` clears `_actual_context`, it does not pick.
        multiprocess.set_start_method(previous, force = True)


def test_an_explicit_num_proc_still_resolves_the_columns():
    rows = [
        {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]},
    ] * 40
    out = standardize_data_formats(Dataset.from_list(rows), num_proc = 2)
    assert [m["role"] for m in out[0]["conversations"]] == ["user", "assistant"]


def test_a_non_ascii_role_is_reported_not_crashed():
    rows = [{"conversations": [{"from": "uzytkownik", "value": "czesc"},
                               {"from": "asystent", "value": "hej"}]}]
    with pytest.raises(TypeError, match = "aliases"):
        standardize_data_formats(Dataset.from_list(rows))


def test_a_role_padded_with_a_non_breaking_space_resolves():
    rows = [{"conversations": [{"from": "human ", "value": "hi"},
                               {"from": " gpt", "value": "yo"}]}]
    out = standardize_data_formats(Dataset.from_list(rows))
    assert [m["role"] for m in out[0]["conversations"]] == ["user", "assistant"]


def test_many_spellings_of_one_role_all_resolve():
    """More distinct spellings than the memo cap must still all resolve."""
    spellings = ["human", "Human", "HUMAN", " human", "human ", "  HuMaN  "]
    spellings += [" " * n + "human" for n in range(2, 90)]
    rows = [
        {"conversations": [{"from": s, "value": f"q{i}"}, {"from": "gpt", "value": f"a{i}"}]}
        for i, s in enumerate(spellings)
    ]
    out = standardize_data_formats(Dataset.from_list(rows))
    assert len(out) == len(spellings)
    for row in out:
        assert [m["role"] for m in row["conversations"]] == ["user", "assistant"]


def test_the_memo_never_changes_the_resolved_role():
    """The memo may not depend on row order or on where a batch boundary falls."""
    pairs = [("human", "user"), ("GPT", "assistant"), ("System", "system"),
             (" input ", "user"), ("OUTPUT", "assistant"), ("human", "user")]
    rows = [{"conversations": [{"from": role, "value": "x"}]} for role, _ in pairs]
    out = standardize_data_formats(Dataset.from_list(rows), batch_size = 2)
    assert [row["conversations"][0]["role"] for row in out] == [want for _, want in pairs]


def test_an_unknown_role_beyond_the_sampled_rows_still_raises():
    """Only the first ten rows decide the column, so a later non-alias role has always
    blown up inside the map. The memo fast path must not swallow it."""
    rows = [{"conversations": [{"from": "human", "value": "hi"},
                               {"from": "gpt", "value": "yo"}]} for _ in range(12)]
    rows.append({"conversations": [{"from": "moderator", "value": "stop"}]})
    with pytest.raises(Exception):
        standardize_data_formats(Dataset.from_list(rows), num_proc = 1)


def test_aliases_that_normalize_into_each_other_are_refused():
    """"Human" for user and "human" for assistant were distinct keys and both worked; after
    normalization the later assignment would silently relabel every such message."""
    rows = [
        {"conversations": [{"from": "Human", "value": "hi"},
                           {"from": "gpt", "value": "yo"}]},
    ]
    with pytest.raises(TypeError, match = "cannot also mean"):
        standardize_data_formats(
            Dataset.from_list(rows),
            aliases_for_user      = ["Human"],
            aliases_for_assistant = ["human", "gpt"],
        )


def test_a_repeated_alias_within_one_role_is_fine():
    rows = [
        {"conversations": [{"from": "Human", "value": "hi"},
                           {"from": "gpt", "value": "yo"}]},
    ]
    out = standardize_data_formats(
        Dataset.from_list(rows),
        aliases_for_user      = ["human", "Human", " HUMAN "],
        aliases_for_assistant = ["gpt"],
    )
    assert [m["role"] for m in out[0]["conversations"]] == ["user", "assistant"]


def test_the_default_alias_lists_are_disjoint():
    """Nobody who passes nothing may ever see the refusal above."""
    import inspect

    from unsloth_zoo.dataset_utils import _normalize_role_alias

    defaults = inspect.signature(standardize_data_formats).parameters
    groups = [
        {_normalize_role_alias(alias) for alias in defaults[name].default}
        for name in ("aliases_for_system", "aliases_for_user", "aliases_for_assistant")
    ]
    for i, first in enumerate(groups):
        for second in groups[i + 1:]:
            assert not (first & second), (first, second)


def test_the_multiprocess_map_test_leaves_the_start_method_as_it_found_it():
    """Restoring only a non-None `previous` leaves the parametrised method globally pinned,
    and `test_start_method_probe_matches_the_pool_multiprocess_would_build` then skips,
    switching a real assertion off."""
    multiprocess = pytest.importorskip("multiprocess")

    before = multiprocess.get_start_method(allow_none = True)
    multiprocess.set_start_method(None, force = True)
    assert multiprocess.get_start_method(allow_none = True) is None
    try:
        test_the_resolved_columns_survive_a_multiprocess_map("spawn")
        assert multiprocess.get_start_method(allow_none = True) is None, (
            "the multiprocess map test pinned a start method and did not give it back"
        )
    finally:
        multiprocess.set_start_method(before, force = True)


_BOTH_ALIAS_CASES = {
    # CONTENT is a role word, so both columns are alias-only with one unique value and the
    # cardinality heuristic used to take the second key and swap them.
    "role_content_one_turn": (
        [{"conversations": [{"role": "user", "content": "assistant"}]}],
        [("user", "assistant")],
    ),
    "from_value_one_turn": (
        [{"conversations": [{"from": "human", "value": "gpt"}]}],
        [("user", "gpt")],
    ),
    "role_content_two_turns": (
        [{"conversations": [{"role": "user", "content": "assistant"},
                            {"role": "assistant", "content": "user"}]}],
        [("user", "assistant"), ("assistant", "user")],
    ),
    "mixed_case_both_sides": (
        [{"conversations": [{"from": " Human ", "value": "GPT"}]}],
        [("user", "GPT")],
    ),
}


@pytest.mark.parametrize("name", sorted(_BOTH_ALIAS_CASES))
def test_the_conventional_key_name_breaks_a_two_alias_tie(name):
    rows, expected = _BOTH_ALIAS_CASES[name]
    out = standardize_data_formats(Dataset.from_list(rows * 2))
    got = [(m["role"], m["content"]) for m in out[0]["conversations"]]
    assert got == expected


def test_an_unconventional_key_pair_still_uses_the_value_evidence():
    """NEGATIVE CONTROL: the key names may only break a tie the values cannot."""
    rows = [
        {"conversations": [{"speaker": "human", "utterance": "hello there"},
                           {"speaker": "gpt", "utterance": "hi"}]},
    ] * 4
    out = standardize_data_formats(Dataset.from_list(rows))
    assert [m["role"] for m in out[0]["conversations"]] == ["user", "assistant"]
    assert out[0]["conversations"][0]["content"] == "hello there"


def test_an_ordinary_dataset_is_unaffected_by_the_tie_break():
    """A normal dataset never reaches the new branch: its content is not alias-only."""
    rows = [
        {"conversations": [{"content": "what is 2 + 2", "role": "user"},
                           {"content": "4", "role": "assistant"}]},
    ] * 4
    out = standardize_data_formats(Dataset.from_list(rows))
    assert [m["role"] for m in out[0]["conversations"]] == ["user", "assistant"]
    assert out[0]["conversations"][0]["content"] == "what is 2 + 2"

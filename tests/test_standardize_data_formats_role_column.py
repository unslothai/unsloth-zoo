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

"""standardize_data_formats must pick the role column by alias membership, not by a
unique-value cardinality tie, and must match aliases without regard to case or
surrounding whitespace (unsloth#1766)."""

import pytest
from datasets import Dataset

from unsloth_zoo.dataset_utils import standardize_data_formats


CASES = {
    # Both columns have two unique values, so the cardinality heuristic ties.
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
    # Capitalised role names are common in hand-built ShareGPT exports.
    "mixed_case_tie": [
        {"conversations": [{"from": "Human", "value": "hi"}, {"from": "GPT", "value": "yo"}]},
    ],
    "mixed_case_role_content": [
        {"conversations": [{"role": "User", "content": "hi"},
                           {"role": "Assistant", "content": "yo"}]},
    ],
    # Trailing and leading whitespace around the role name.
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
    """No cardinality tie here: the role column already wins on unique count. What used
    to fail is the alias check, which compared "Human" against the lowercase alias list
    and raised TypeError."""
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


# --- the map that carries this decision into worker processes ---------------

@pytest.mark.parametrize("start_method", ["fork", "spawn"])
def test_the_resolved_columns_survive_a_multiprocess_map(start_method):
    """`_standardize_dataset` is handed to `Dataset.map(num_proc = ...)`, so the
    role and content columns it closed over have to reach the workers. datasets
    pools through `multiprocess`, whose default context is its own; Windows has no
    fork at all, so the spawn case is what runs there and it is the one where a
    closure over a module-level helper can fail to arrive."""
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
        if previous is not None:
            multiprocess.set_start_method(previous, force = True)


def test_an_explicit_num_proc_still_resolves_the_columns():
    rows = [
        {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]},
    ] * 40
    out = standardize_data_formats(Dataset.from_list(rows), num_proc = 2)
    assert [m["role"] for m in out[0]["conversations"]] == ["user", "assistant"]


def test_a_non_ascii_role_is_reported_not_crashed():
    """Normalising means calling `.strip().lower()` on whatever the column holds.
    An unknown role in another script must still come out as the aliases message."""
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
    """The map memoises each raw spelling it sees so the common already-normal role costs
    a single dict hit. The memo is capped, so a dataset carrying more distinct spellings
    than the cap must still resolve every one of them, cap or no cap."""
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
    """Every spelling must map to what normalising it alone would have produced, whatever
    order the rows arrive in, and whichever batch boundary falls between them."""
    pairs = [("human", "user"), ("GPT", "assistant"), ("System", "system"),
             (" input ", "user"), ("OUTPUT", "assistant"), ("human", "user")]
    rows = [{"conversations": [{"from": role, "value": "x"}]} for role, _ in pairs]
    # A system turn on its own makes the role column the only all-alias column either way.
    out = standardize_data_formats(Dataset.from_list(rows), batch_size = 2)
    assert [row["conversations"][0]["role"] for row in out] == [want for _, want in pairs]


def test_an_unknown_role_beyond_the_sampled_rows_still_raises():
    """Only the first ten rows decide the column, so a role that appears later and is not
    an alias has always blown up inside the map. The fast path must not silently accept it."""
    rows = [{"conversations": [{"from": "human", "value": "hi"},
                               {"from": "gpt", "value": "yo"}]} for _ in range(12)]
    rows.append({"conversations": [{"from": "moderator", "value": "stop"}]})
    with pytest.raises(Exception):
        standardize_data_formats(Dataset.from_list(rows), num_proc = 1)


def test_aliases_that_normalize_into_each_other_are_refused():
    """Normalization is what makes the lookup case and whitespace insensitive, and it is
    also what can collapse two aliases the caller meant to keep apart. Before it,
    "Human" for user and "human" for assistant were distinct keys and both worked;
    after it they are one key, the later assignment wins, and every such message is
    silently relabelled assistant. Silently mislabelled training data is worse than a
    refusal, so this raises with the pair named."""
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
    """Only a collision ACROSS roles is ambiguous; listing a spelling twice for the same
    role says nothing contradictory."""
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

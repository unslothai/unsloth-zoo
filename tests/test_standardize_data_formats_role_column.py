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

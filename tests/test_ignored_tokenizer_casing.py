# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression test: fix_untrained_tokens' IGNORED_TOKENIZER_NAMES guard must match
case-insensitively.

The list is exported (via UNSLOTH_IGNORED_TOKENIZER_NAMES) fully lowercased, but
fix_untrained_tokens compared model.config._name_or_path (the canonical mixed-case
id, e.g. unsloth/Qwen2.5-Coder-7B-Instruct) against it. The guard never matched, so
fix_untrained_tokens ran on the very models the list exists to leave alone.
"""

from types import SimpleNamespace

import pytest

from unsloth_zoo.tokenizer_utils import fix_untrained_tokens


class _GuardPassed(Exception):
    """Raised once execution proceeds past the ignored-name guard."""


class _SentinelWeight:
    # .shape is only touched when the guard does NOT match (execution continues
    # past the early return), so raising here detects that fix_untrained_tokens
    # did NOT skip an ignored model.
    @property
    def shape(self):
        raise _GuardPassed()


def _model(name):
    emb = SimpleNamespace(weight = _SentinelWeight())
    return SimpleNamespace(
        get_input_embeddings = lambda: emb,
        get_output_embeddings = lambda: emb,
        config = SimpleNamespace(_name_or_path = name),
    )


# IGNORED_TOKENIZER_NAMES is stored and exported lowercased.
_IGNORED = ["unsloth/qwen2.5-coder-7b-instruct"]
_TOK = SimpleNamespace(chat_template = None)


def test_ignored_name_matched_case_insensitively():
    # The canonical mixed-case id must be recognized and skipped (returns early).
    result = fix_untrained_tokens(
        _model("unsloth/Qwen2.5-Coder-7B-Instruct"), _TOK, None,
        IGNORED_TOKENIZER_NAMES = _IGNORED,
    )
    assert result is None


def test_non_ignored_name_still_processed():
    # A model not in the list must not be skipped; it proceeds past the guard.
    with pytest.raises(_GuardPassed):
        fix_untrained_tokens(
            _model("unsloth/Llama-3.1-8B-Instruct"), _TOK, None,
            IGNORED_TOKENIZER_NAMES = _IGNORED,
        )


if __name__ == "__main__":
    test_ignored_name_matched_case_insensitively()
    test_non_ignored_name_still_processed()
    print("ok")

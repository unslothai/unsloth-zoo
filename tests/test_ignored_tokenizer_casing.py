# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression test: fix_untrained_tokens' IGNORED_TOKENIZER_NAMES guard must match
case-insensitively.

The production list is exported (via UNSLOTH_IGNORED_TOKENIZER_NAMES) fully
lowercased, but callers may also provide canonical mixed-case IDs. Both the model
name and configured entries must therefore be normalized before comparison.
"""

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("UNSLOTH_ZOO_DISABLE_GPU_INIT", "1")

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


_MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct"
_IGNORED_LOWERCASE = ["unsloth/qwen2.5-coder-7b-instruct"]
_IGNORED_CANONICAL = [_MODEL_NAME]
_TOK = SimpleNamespace(chat_template = None)


def test_ignored_name_matched_case_insensitively():
    # The lowercased production entry must match the canonical mixed-case ID.
    result = fix_untrained_tokens(
        _model(_MODEL_NAME), _TOK, None,
        IGNORED_TOKENIZER_NAMES = _IGNORED_LOWERCASE,
    )
    assert result is None


def test_canonical_ignored_entry_still_matched():
    # Preserve callers that pass the canonical ID instead of a lowercased entry.
    result = fix_untrained_tokens(
        _model(_MODEL_NAME), _TOK, None,
        IGNORED_TOKENIZER_NAMES = _IGNORED_CANONICAL,
    )
    assert result is None


def test_non_ignored_name_still_processed():
    # A model not in the list must not be skipped; it proceeds past the guard.
    with pytest.raises(_GuardPassed):
        fix_untrained_tokens(
            _model("unsloth/Llama-3.1-8B-Instruct"), _TOK, None,
            IGNORED_TOKENIZER_NAMES = _IGNORED_LOWERCASE,
        )


if __name__ == "__main__":
    test_ignored_name_matched_case_insensitively()
    test_canonical_ignored_entry_still_matched()
    test_non_ignored_name_still_processed()
    print("ok")

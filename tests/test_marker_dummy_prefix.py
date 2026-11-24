"""Marker token cores drop the bare SentencePiece dummy-prefix piece a standalone tokenization starts with.

transformers 5 SentencePiece tokenizers tokenize Zephyr's "\\n<|assistant|>\\n" on its own as "▁", "<0x0A>", "<", ...
Mid-text, after the previous turn, the marker has no leading "▁", so a core that keeps it matches nothing and
train_on_responses_only either refuses to auto-detect or trains on no tokens. The stub below reproduces that shape
without a downloaded tokenizer, so this runs offline (tests/test_zephyr_marker_context.py needs a cached one).
"""

import pytest

dataset_utils = pytest.importorskip("unsloth_zoo.dataset_utils")

PREFIX = "▁"


class _Encoding(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class _DummyPrefixTokenizer:
    """One piece per character; standalone text that starts with "\\n" gets a lone dummy-prefix piece first."""

    def __init__(self):
        self.vocab = {PREFIX: 1}

    def _id(self, piece):
        return self.vocab.setdefault(piece, len(self.vocab) + 1)

    def __call__(self, text, add_special_tokens = False):
        pieces = ([PREFIX] if text.startswith("\n") else []) + list(text)
        return _Encoding(input_ids = [self._id(p) for p in pieces])

    def convert_ids_to_tokens(self, ids):
        inverse = {v: k for k, v in self.vocab.items()}
        if isinstance(ids, int):
            return inverse[ids]
        return [inverse[i] for i in ids]


@pytest.mark.parametrize("force_match", [True, False])
def test_the_core_has_no_dummy_prefix(force_match):
    tok = _DummyPrefixTokenizer()
    marker = "\n<|assistant|>\n"
    core, _, _ = dataset_utils._find_common_token_ids(marker, tok, force_match)
    assert core and tok.convert_ids_to_tokens(core[0]) != PREFIX
    # The core is what the marker tokenizes to mid-text, so it is found inside a running conversation.
    conversation = tok("x</s>\n<|assistant|>\nANSWER</s>").input_ids
    assert any(conversation[i : i + len(core)] == core for i in range(len(conversation) - len(core) + 1))


def test_text_that_starts_with_a_space_keeps_its_first_piece():
    tok = _DummyPrefixTokenizer()
    ids = tok(" x").input_ids
    assert dataset_utils._without_dummy_prefix(ids, " x", tok) == ids


def test_a_lone_piece_is_kept():
    tok = _DummyPrefixTokenizer()
    ids = [tok._id(PREFIX)]
    assert dataset_utils._without_dummy_prefix(ids, "\n", tok) == ids

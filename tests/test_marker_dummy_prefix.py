"""Marker cores drop the bare SentencePiece dummy-prefix piece: transformers 5 emits a leading ▁ for
standalone text, but mid-text the marker lacks it, so keeping it matches nothing
(train_on_responses_only refuses to auto-detect or trains on no tokens); the stub runs offline."""

import pytest

dataset_utils = pytest.importorskip("unsloth_zoo.dataset_utils")

PREFIX = "▁"


class _Encoding(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class _DummyPrefixTokenizer:
    """One piece per character, except the added <CHAT> token and '>' merging with a following
    backslash (aya-vision-32b); standalone text starting with a newline gets a lone dummy-prefix
    piece first."""

    ADDED = "<CHAT>"
    MERGE = ">\\"

    def __init__(self):
        self.vocab = {PREFIX: 1}
        self.added_tokens_decoder = {self._id(self.ADDED): self.ADDED}

    def _id(self, piece):
        return self.vocab.setdefault(piece, len(self.vocab) + 1)

    def __call__(self, text, add_special_tokens = False):
        pieces = [PREFIX] if text.startswith("\n") else []
        i = 0
        while i < len(text):
            for unit in (self.ADDED, self.MERGE):
                if text.startswith(unit, i):
                    pieces.append(unit)
                    i += len(unit)
                    break
            else:
                pieces.append(text[i])
                i += 1
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


def test_a_mergeable_suffix_is_still_trimmed_after_the_prefix_is_dropped():
    """>' merging into the answer's backslash must be trimmed, or the core matches no answer."""
    tok = _DummyPrefixTokenizer()
    core, _, right = dataset_utils._find_common_token_ids("\n<CHAT>>", tok, True)
    assert tok.convert_ids_to_tokens(core) == ["\n", "<CHAT>"]
    assert tok.convert_ids_to_tokens(right) == [">"]
    conversation = tok("x</s>\n<CHAT>>\\sigma").input_ids
    assert any(conversation[i : i + len(core)] == core for i in range(len(conversation) - len(core) + 1))

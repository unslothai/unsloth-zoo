"""lm_head is tied to the embeddings, so it may be absent from either state dict.

The symmetric-difference check excuses the tied lm_head names, but the per-key compare
that follows then looked them up in the other dict unconditionally, so a key excused on
one line raised KeyError two lines later. Idefics3 hits this with
`model.text_model.lm_head.weight`.

Being excused must not weaken the check: when both sides do carry the weight and the
values disagree, that still has to be reported.
"""
import pytest
import torch

from unsloth_zoo.vllm_utils import (
    TIED_EMBED_KEYS,
    TIED_LM_HEAD_KEYS,
    assert_same_state_dict,
)

PAIRS = [
    ("lm_head.weight", "model.embed_tokens.weight"),
    ("model.lm_head.weight", "model.embed_tokens.weight"),
    ("model.language_model.lm_head.weight", "model.language_model.embed_tokens.weight"),
    ("model.text_model.lm_head.weight", "model.text_model.embed_tokens.weight"),
]


def w(value = 0.0):
    return torch.full((4, 4), value)


def test_every_tied_name_is_covered_by_both_sets():
    assert "model.text_model.lm_head.weight" in TIED_LM_HEAD_KEYS
    assert "model.text_model.embed_tokens.weight" in TIED_EMBED_KEYS
    for head, embed in PAIRS:
        assert head in TIED_LM_HEAD_KEYS
        assert embed in TIED_EMBED_KEYS


@pytest.mark.parametrize("head, embed", PAIRS)
def test_tied_lm_head_may_be_missing_from_either_side(head, embed):
    base = {embed: w()}
    assert_same_state_dict(dict(base), {**base, head: w()})   # only in vllm
    assert_same_state_dict({**base, head: w()}, dict(base))   # only in hf
    assert_same_state_dict({**base, head: w()}, {**base, head: w()})


@pytest.mark.parametrize("head, embed", PAIRS)
def test_a_tied_lm_head_that_disagrees_is_still_reported(head, embed):
    base = {embed: w(0.0)}
    with pytest.raises(Exception):
        assert_same_state_dict({**base, head: w(1.0)}, dict(base))


def test_genuine_key_differences_still_raise():
    base = {"model.embed_tokens.weight": w()}
    with pytest.raises(RuntimeError):
        assert_same_state_dict(dict(base), {**base, "model.layers.0.q_proj.weight": w()})
    with pytest.raises(RuntimeError):
        assert_same_state_dict({**base, "model.layers.0.q_proj.weight": w()}, dict(base))


def test_genuine_value_differences_still_raise():
    with pytest.raises(Exception):
        assert_same_state_dict({"a.weight": w(0.0)}, {"a.weight": w(1.0)})


def test_identical_state_dicts_pass():
    sd = {"model.embed_tokens.weight": w(), "model.layers.0.q_proj.weight": w()}
    assert_same_state_dict(dict(sd), dict(sd))

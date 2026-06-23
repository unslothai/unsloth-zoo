# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.
"""Offline tests for unsloth_zoo.pad_token.fix_pad_token (no network, no torch)."""

import importlib.util
import os

import pytest

# Load pad_token.py directly from its file so the test stays truly offline: it
# does not run unsloth_zoo/__init__.py (which pulls the full stack and can fail
# without the companion unsloth package installed).
_PAD_TOKEN_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "unsloth_zoo", "pad_token.py",
)
_spec = importlib.util.spec_from_file_location("_unsloth_pad_token_offline", _PAD_TOKEN_PATH)
_pad_token = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pad_token)
fix_pad_token = _pad_token.fix_pad_token
VISION_RESERVED_TOKENS = _pad_token.VISION_RESERVED_TOKENS


class FakeTokenizer:
    """Minimal stand-in: vocab is a {token: id} dict; reserved tokens are 'added'."""

    def __init__(self, vocab, pad_token, eos_token, added=None):
        self._vocab = dict(vocab)
        self.pad_token = pad_token
        self.eos_token = eos_token
        # added_tokens_decoder maps id -> token content (subset treated as "added")
        added = added if added is not None else list(vocab)
        self.added_tokens_decoder = {self._vocab[t]: t for t in added if t in self._vocab}

    # ids
    @property
    def pad_token_id(self):
        return self._vocab.get(self.pad_token)

    @property
    def eos_token_id(self):
        return self._vocab.get(self.eos_token)

    def get_vocab(self):
        return dict(self._vocab)

    def __call__(self, text, add_special_tokens=False):
        # Each known token is a single id; unknown text splits into >1 id.
        ids = [self._vocab[text]] if text in self._vocab else [0, 1]
        return type("Enc", (), {"input_ids": ids})()

    def add_special_tokens(self, mapping):
        tok = mapping["pad_token"]
        if tok not in self._vocab:
            self._vocab[tok] = max(self._vocab.values()) + 1
            self.added_tokens_decoder[self._vocab[tok]] = tok
        self.pad_token = tok


def _qwen3_text():
    # Qwen3 text tokenizer shipping a vision pad_token (the #3155 bug).
    vocab = {
        "<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645,
        "<|vision_pad|>": 151654, "<|image_pad|>": 151655, "<|video_pad|>": 151656,
    }
    return FakeTokenizer(vocab, pad_token="<|vision_pad|>", eos_token="<|im_end|>")


def test_qwen3_vision_pad_heals_to_endoftext():
    tok = _qwen3_text()
    res = fix_pad_token(tok)
    assert res["changed"] and res["reason"] == "vision_pad"
    assert tok.pad_token == "<|endoftext|>"          # not a vision token
    assert tok.pad_token != tok.eos_token


def test_valid_distinct_pad_is_noop():
    # Mistral-style: pad <pad>, eos </s>, distinct -> untouched.
    tok = FakeTokenizer({"<s>": 1, "</s>": 2, "<pad>": 0, "[control_8]": 10},
                        pad_token="<pad>", eos_token="</s>")
    res = fix_pad_token(tok)
    assert res["changed"] is False and tok.pad_token == "<pad>"


def test_llama31_finetune_pad_is_noop():
    tok = FakeTokenizer({"<|eot_id|>": 128009, "<|finetune_right_pad_id|>": 128004},
                        pad_token="<|finetune_right_pad_id|>", eos_token="<|eot_id|>")
    assert fix_pad_token(tok)["changed"] is False


def test_pad_equals_eos_picks_distinct_reserved():
    tok = FakeTokenizer({"</s>": 2, "<pad>": 0}, pad_token="</s>", eos_token="</s>")
    res = fix_pad_token(tok)
    assert res["reason"] == "equals_eos" and tok.pad_token == "<pad>"
    assert tok.pad_token != tok.eos_token


def test_missing_pad_picks_reserved():
    tok = FakeTokenizer({"<|eot_id|>": 1, "<|reserved_special_token_0|>": 5},
                        pad_token=None, eos_token="<|eot_id|>")
    res = fix_pad_token(tok)
    assert res["reason"] == "missing"
    assert tok.pad_token == "<|reserved_special_token_0|>"


def test_allow_add_false_defers_when_nothing_found():
    # pad==eos and no reserved candidate; tokenizer-only path must not add/raise.
    tok = FakeTokenizer({"</s>": 2}, pad_token="</s>", eos_token="</s>")
    res = fix_pad_token(tok, allow_add=False)
    assert res["changed"] is False and tok.pad_token == "</s>"


def test_manual_fallback_raises():
    tok = FakeTokenizer({"</s>": 2}, pad_token="</s>", eos_token="</s>")
    with pytest.raises(RuntimeError):
        fix_pad_token(tok, allow_add=True)


# --- bug regressions vs the original reference logic ---

def test_closed_bracket_uses_full_token_not_prefix():
    # Pixtral <SPECIAL_NNN>: prefix "<SPECIAL_" is open, full token is closed.
    tok = FakeTokenizer({"</s>": 2, "<SPECIAL_20>": 20, "<SPECIAL_21>": 21},
                        pad_token="</s>", eos_token="</s>")
    res = fix_pad_token(tok)
    assert res["changed"] and tok.pad_token.startswith("<SPECIAL_") and tok.pad_token.endswith(">")


def test_zero_token_candidate_does_not_crash():
    # A reserved entry present but encoding to !=1 id must be skipped, not crash.
    class Weird(FakeTokenizer):
        def __call__(self, text, add_special_tokens=False):
            return type("Enc", (), {"input_ids": []})()  # zero ids for everything
    tok = Weird({"</s>": 2, "<pad>": 0}, pad_token="</s>", eos_token="</s>")
    res = fix_pad_token(tok, allow_add=False)   # <pad> rejected (0 ids) -> defer, no IndexError
    assert res["changed"] is False


def test_vision_token_never_selected_for_text_model():
    tok = _qwen3_text()
    fix_pad_token(tok)
    assert tok.pad_token not in VISION_RESERVED_TOKENS


def test_out_of_range_pad_is_repicked_when_model_known():
    # pad looks fine (not eos, not vision) but its id is past the model vocab,
    # e.g. an earlier model-less swap picked a high-id added token. The
    # model-aware call must catch it and re-pick an in-range reserved token.
    vocab = {
        "<|eot_id|>": 0,                       # eos
        "<|reserved_special_token_0|>": 100,   # in-range reserved
        "<pad>": 5000,                         # current pad, beyond vocab_size
    }
    tok = FakeTokenizer(vocab, pad_token = "<pad>", eos_token = "<|eot_id|>")
    cfg = type("Cfg", (), {"vocab_size": 1000})()
    res = fix_pad_token(tok, model_config = cfg)
    assert res["reason"] == "out_of_range"
    assert tok.pad_token == "<|reserved_special_token_0|>"


def test_in_range_valid_pad_is_noop_with_vocab_size():
    tok = FakeTokenizer({"<s>": 1, "</s>": 2, "<pad>": 0},
                        pad_token = "<pad>", eos_token = "</s>")
    cfg = type("Cfg", (), {"vocab_size": 1000})()
    assert fix_pad_token(tok, model_config = cfg)["changed"] is False


def test_vision_model_keeps_its_vision_pad():
    # A real multimodal model (model_type without "vl", e.g. llava) must be
    # detected as vision so its vision pad_token is left untouched.
    tok = _qwen3_text()  # pad_token = "<|vision_pad|>"
    cfg = type("Cfg", (), {"model_type": "llava"})()
    res = fix_pad_token(tok, model_config = cfg)
    assert res["changed"] is False
    assert tok.pad_token == "<|vision_pad|>"


def test_unk_token_fallback_when_no_reserved():
    # Llama-2 style: no pad and no reserved-token family, but <unk> exists, so
    # reuse it rather than hard-failing (matches the original patch_tokenizer).
    tok = FakeTokenizer({"<s>": 1, "</s>": 2, "<unk>": 0},
                        pad_token = None, eos_token = "</s>")
    tok.unk_token = "<unk>"
    res = fix_pad_token(tok, allow_add = False)
    assert res["changed"] and tok.pad_token == "<unk>"


def test_vision_model_reuses_vision_pad_when_only_option():
    # Vision model with pad == eos and only a modality pad available -> reuse it
    # instead of adding/raising.
    tok = FakeTokenizer({"<|im_end|>": 1, "<|vision_pad|>": 2},
                        pad_token = "<|im_end|>", eos_token = "<|im_end|>")
    cfg = type("Cfg", (), {"model_type": "qwen2_vl"})()
    res = fix_pad_token(tok, model_config = cfg)
    assert res["changed"] and tok.pad_token == "<|vision_pad|>"


def test_audio_pad_not_denylisted_for_audio_model():
    # <|audio_pad|> is not a denied token (we cannot detect audio-only models),
    # so an audio tokenizer keeping it as pad is a no-op.
    tok = FakeTokenizer({"<|im_end|>": 1, "<|audio_pad|>": 2},
                        pad_token = "<|audio_pad|>", eos_token = "<|im_end|>")
    assert fix_pad_token(tok)["changed"] is False
    assert "<|audio_pad|>" not in VISION_RESERVED_TOKENS

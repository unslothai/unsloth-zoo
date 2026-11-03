# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""fix_pad_token keeps a new pad that needed an embedding resize: the pad id starts one past the 256 rows with no
reserved / unk candidate, and afterwards model and tokenizer agree (rows, pad id in range, stamped on the config)."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from unsloth_zoo.pad_token import fix_pad_token
from unsloth_zoo.tokenizer_utils import patch_tokenizer

ROWS = 256


def _model(pad_token_id = None):
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size = ROWS, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 1,
        num_attention_heads = 2, num_key_value_heads = 1, max_position_embeddings = 64,
        pad_token_id = pad_token_id, bos_token_id = ROWS - 1, eos_token_id = ROWS - 1,
    )
    return LlamaForCausalLM(cfg).eval()


def _tokenizer(tmp_path, pad_aliases_eos = False):
    vocab = {f"t{i}": i for i in range(ROWS - 1)}
    vocab["<s>"] = ROWS - 1  # fills the embedding; "<pad>" is not in it, so it is added at id ROWS
    tok = Tokenizer(models.WordLevel(vocab = vocab, unk_token = "t0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    pad = "<s>" if pad_aliases_eos else "<pad>"
    PreTrainedTokenizerFast(
        tokenizer_object = tok, pad_token = pad, bos_token = "<s>", eos_token = "<s>",
    ).save_pretrained(tmp_path)
    loaded = transformers.AutoTokenizer.from_pretrained(tmp_path)
    loaded.unk_token = None  # no unk fallback, so only an added pad can repair
    return loaded


def _assert_agree(model, tok, pad):
    rows_in = model.get_input_embeddings().weight.shape[0]
    rows_out = model.get_output_embeddings().weight.shape[0]
    pid = tok.pad_token_id
    assert tok.pad_token == pad
    assert rows_in == rows_out >= len(tok)
    assert pid is not None and pid < rows_in and pid != tok.eos_token_id
    assert tok.convert_ids_to_tokens(pid) == pad
    assert model.config.pad_token_id == pid
    assert model.generation_config.pad_token_id == pid
    enc = tok(["t5 t6 t7", "t9"], padding = True, return_tensors = "pt")
    assert int(enc.input_ids.max()) < rows_in
    with torch.no_grad():
        model(input_ids = enc.input_ids, attention_mask = enc.attention_mask)


def test_out_of_range_pad_is_added_and_resized_without_raising(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSLOTH_VOCAB_PAD_MULTIPLE", raising = False)
    tok = _tokenizer(tmp_path)
    assert tok.pad_token_id >= ROWS  # the bug's precondition: pad id past the embedding
    model = _model()
    res = fix_pad_token(tok, model = model)
    assert res["changed"] and res["added"] and res["reason"] == "out_of_range"
    _assert_agree(model, tok, res["new_pad"])
    assert fix_pad_token(tok, model = model)["changed"] is False


@pytest.mark.parametrize("pad_aliases_eos", [False, True])
def test_patch_tokenizer_returns_agreeing_model_and_tokenizer(tmp_path, monkeypatch, pad_aliases_eos):
    # The loader entry point (FastLanguageModel calls it unguarded, FastModel falls back on any error).
    monkeypatch.delenv("UNSLOTH_VOCAB_PAD_MULTIPLE", raising = False)
    tok = _tokenizer(tmp_path, pad_aliases_eos = pad_aliases_eos)
    model, tok = patch_tokenizer(_model(), tok)
    assert tok.pad_token != "<s>"
    _assert_agree(model, tok, tok.pad_token)


def test_resize_with_pad_multiple_keeps_pad(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_VOCAB_PAD_MULTIPLE", "64")
    tok = _tokenizer(tmp_path)
    model = _model()
    res = fix_pad_token(tok, model = model)
    assert model.get_input_embeddings().weight.shape[0] == 320
    _assert_agree(model, tok, res["new_pad"])


def test_failed_resize_still_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSLOTH_VOCAB_PAD_MULTIPLE", raising = False)
    tok = _tokenizer(tmp_path)
    model = _model()

    def _boom(*args, **kwargs):
        raise RuntimeError("resize refused")
    monkeypatch.setattr(model, "resize_token_embeddings", _boom)
    with pytest.raises(RuntimeError, match = "Could not find a valid pad token"):
        fix_pad_token(tok, model = model)
    assert model.get_input_embeddings().weight.shape[0] == ROWS


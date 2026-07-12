def test_mlx_vlm_bpe_incomplete_utf8_fallback(monkeypatch):
    import pytest
    tokenizer_utils = pytest.importorskip("mlx_vlm.tokenizer_utils")
    from unsloth_zoo.mlx.loader import _ensure_mlx_vlm_bpe_utf8_fallback, _mlx_vlm_bpe_needs_utf8_fallback
    def strict_add_token(self, token, skip_special_token_ids=[]):
        if token == 0:
            self._unflushed = self.tokenmap[token]
            return
        bytearray(self._byte_decoder[char] for char in self._unflushed).decode("utf-8")
    strict = type("StrictBPEStreamingDetokenizer", (tokenizer_utils.BPEStreamingDetokenizer,), {"add_token": strict_add_token})
    monkeypatch.setattr(tokenizer_utils, "BPEStreamingDetokenizer", strict)
    strict.make_byte_decoder()
    byte_chars = {byte: char for char, byte in strict._byte_decoder.items()}
    def make_detokenizer():
        detokenizer = object.__new__(strict)
        detokenizer.trim_space = False
        detokenizer.tokenmap = [byte_chars[0xC3], byte_chars[32] + "x"]
        detokenizer.reset()
        return detokenizer
    _ensure_mlx_vlm_bpe_utf8_fallback()
    detokenizer = make_detokenizer()
    detokenizer.add_token(0)
    detokenizer.add_token(1)
    assert (detokenizer.text, detokenizer._unflushed) == ("\ufffd", byte_chars[32] + "x")
    assert not _mlx_vlm_bpe_needs_utf8_fallback(strict)
    detokenizer = make_detokenizer()
    detokenizer._unflushed = byte_chars[0xFF]
    with pytest.raises(UnicodeDecodeError, match="invalid start byte"):
        detokenizer.add_token(1)
    def mutating_add_token(self, token, skip_special_token_ids=[]):
        if token == 0:
            return strict_add_token(self, token)
        self._unflushed = "changed"
        raise UnicodeDecodeError("utf-8", b"\xc3", 0, 1, "unexpected end of data")
    monkeypatch.setattr(strict, "add_token", mutating_add_token)
    assert not _mlx_vlm_bpe_needs_utf8_fallback(strict)

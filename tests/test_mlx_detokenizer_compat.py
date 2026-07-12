def test_mlx_vlm_bpe_incomplete_utf8_fallback(monkeypatch):
    import sys
    import types
    import pytest
    pytest.importorskip("mlx.core")
    from unsloth_zoo.mlx.loader import _ensure_mlx_vlm_bpe_utf8_fallback, _mlx_vlm_bpe_needs_utf8_fallback
    tokenizer_utils = types.ModuleType("mlx_vlm.tokenizer_utils")
    tokenizer_utils._remove_space = lambda text: text[1:] if text.startswith(" ") else text
    monkeypatch.setitem(sys.modules, "mlx_vlm", types.ModuleType("mlx_vlm"))
    monkeypatch.setitem(sys.modules, "mlx_vlm.tokenizer_utils", tokenizer_utils)
    class StrictBPEStreamingDetokenizer:
        _byte_decoder = {"Ã": 0xC3, "¿": 0xBF, " ": 32}
        make_byte_decoder = classmethod(lambda cls: None)
        def reset(self):
            self.offset, self._unflushed, self.text, self.tokens = 0, "", "", []
        def add_token(self, token, skip_special_token_ids=[]):
            value = self.tokenmap[token]
            if token in (2, 3, 4):
                encoding = "synthetic-codec" if token == 2 else "utf-8"
                self.failure = UnicodeDecodeError(encoding, b"\xc3", 0, 1, "unexpected end of data")
                if token == 3:
                    self.text = "changed"
                if token == 4:
                    self._unflushed = "changed"
                raise self.failure
            if self._byte_decoder[value[0]] == 32:
                bytearray(self._byte_decoder[char] for char in self._unflushed).decode("utf-8")
                self._unflushed = value
            else:
                self._unflushed += value
    strict = StrictBPEStreamingDetokenizer
    tokenizer_utils.BPEStreamingDetokenizer = strict
    byte_chars = {byte: char for char, byte in strict._byte_decoder.items()}
    def make_detokenizer(tokenmap=None):
        detokenizer = object.__new__(strict)
        detokenizer.trim_space = False
        detokenizer.tokenmap = tokenmap or [byte_chars[0xC3], byte_chars[32] + "x", byte_chars[32], byte_chars[32], byte_chars[32]]
        detokenizer.reset()
        return detokenizer
    _ensure_mlx_vlm_bpe_utf8_fallback()
    detokenizer = make_detokenizer()
    detokenizer.add_token(0)
    detokenizer.add_token(1)
    assert (detokenizer.text, detokenizer._unflushed) == ("\ufffd", byte_chars[32] + "x")
    assert not _mlx_vlm_bpe_needs_utf8_fallback(strict)
    detokenizer = make_detokenizer({123: byte_chars[0xBF], 220: byte_chars[32], 2: byte_chars[32]})
    for token in (123, 220):
        detokenizer.add_token(token)
    assert (detokenizer.text, detokenizer._unflushed) == ("\ufffd", byte_chars[32])
    detokenizer = make_detokenizer()
    detokenizer.add_token(0)
    with pytest.raises(UnicodeDecodeError, match="synthetic-codec"):
        detokenizer.add_token(2)
    for token in (3, 4):
        detokenizer = make_detokenizer()
        detokenizer.add_token(0)
        with pytest.raises(UnicodeDecodeError) as raised:
            detokenizer.add_token(token)
        assert raised.value is detokenizer.failure
    def incomplete_safe_add_token(self, token, skip_special_token_ids=[]):
        value = self.tokenmap[token]
        if token == 0 or self._unflushed == byte_chars[0xC3]:
            self._unflushed = value
            return
        bytearray(self._byte_decoder[char] for char in self._unflushed).decode("utf-8")
    monkeypatch.setattr(strict, "add_token", incomplete_safe_add_token)
    assert _mlx_vlm_bpe_needs_utf8_fallback(strict)

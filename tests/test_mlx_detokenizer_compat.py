def test_mlx_vlm_bpe_incomplete_utf8_fallback(monkeypatch):
    import sys
    import types
    import pytest
    pytest.importorskip("mlx.core")
    from unsloth_zoo.mlx.loader import _ensure_mlx_vlm_bpe_utf8_fallback, _mlx_vlm_bpe_needs_utf8_fallback
    package = types.ModuleType("mlx_vlm")
    package.__path__ = []
    tokenizer_utils = types.ModuleType("mlx_vlm.tokenizer_utils")
    tokenizer_utils._remove_space = lambda text: text[1:] if text.startswith(" ") else text
    package.tokenizer_utils = tokenizer_utils
    monkeypatch.setitem(sys.modules, "mlx_vlm", package)
    monkeypatch.setitem(sys.modules, "mlx_vlm.tokenizer_utils", tokenizer_utils)
    class StrictBPEStreamingDetokenizer:
        _byte_decoder = {"Ã": 0xC3, " ": 32, "ÿ": 0xFF}
        make_byte_decoder = classmethod(lambda cls: None)
        def reset(self):
            self.offset, self._unflushed, self.text, self.tokens = 0, "", "", []
        def add_token(self, token, skip_special_token_ids=[]):
            if token == 0:
                self._unflushed = self.tokenmap[token]
                return
            bytearray(self._byte_decoder[char] for char in self._unflushed).decode("utf-8")
    strict = StrictBPEStreamingDetokenizer
    tokenizer_utils.BPEStreamingDetokenizer = strict
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
            self._unflushed = self.tokenmap[token]
            return
        self._unflushed = "changed"
        raise UnicodeDecodeError("utf-8", b"\xc3", 0, 1, "unexpected end of data")
    monkeypatch.setattr(strict, "add_token", mutating_add_token)
    assert not _mlx_vlm_bpe_needs_utf8_fallback(strict)

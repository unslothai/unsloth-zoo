"""The verifier's GGUF reader must read a file exactly as the stock GGUFReader does,
for everything the verifier looks at, without one numpy view per vocabulary entry.

The stock reader splits every array into per-entry views (two per token string), which
is what made reading a 262k-vocabulary GGUF back take seconds. Long arrays now come back
as one view (numeric) or their first entry (strings). Field names, offsets, types,
numeric contents, string-array emptiness and tensors must match the stock reader, and
a truncated or corrupt file must fail with the same exception type and text.
"""
import struct

import pytest

gguf = pytest.importorskip("gguf")
np = pytest.importorskip("numpy")
from gguf import GGUFEndian, GGUFWriter  # noqa: E402
from gguf.gguf_reader import GGUFReader  # noqa: E402

from test_gguf_post_conversion_validation import _load_llama_cpp_module  # noqa: E402


@pytest.fixture(scope = "module")
def llama_cpp():
    return _load_llama_cpp_module()


VOCAB = 5000


def _write(path, vocab = VOCAB, endian = GGUFEndian.LITTLE):
    w = GGUFWriter(str(path), "llama", endianess = endian)
    w.add_context_length(4096)
    w.add_embedding_length(64)
    w.add_block_count(2)
    w.add_tokenizer_model("gpt2")
    w.add_token_list([("▁té" if i % 3 == 0 else "t") + str(i) for i in range(vocab)])
    w.add_token_merges([f"t{i} t{i + 1}" for i in range(max(vocab - 10, 0))])
    w.add_token_scores([float(i % 7) for i in range(vocab)])
    w.add_token_types([1 + i % 2 for i in range(vocab)])
    w.add_array("llama.per_layer", [3, 4])
    w.add_eos_token_id(2)
    w.add_tensor("token_embd.weight", np.arange(64 * 8, dtype = np.float32).reshape(8, 64))
    w.add_tensor("output_norm.weight", np.ones(64, dtype = np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def _summary(reader):
    fields = []
    for name, field in reader.fields.items():
        types = [int(t) for t in field.types]
        string_array = bool(types) and types[0] == 9 and types[-1] == 8
        record = [name, int(field.offset), types, bool(field.data)]
        if string_array:
            record.append(bytes(field.parts[field.data[0]]) if field.data else None)
        else:
            # Raw bytes of the value: the same whether an array is one view or many,
            # and available on gguf releases older than ReaderField.contents().
            record.append(b"".join(field.parts[i].tobytes() for i in field.data))
        fields.append(record)
    tensors = [
        (t.name, [int(x) for x in t.shape], int(t.tensor_type), int(t.n_bytes),
         int(t.data_offset), bytes(t.data))
        for t in reader.tensors
    ]
    return fields, tensors


def _outcome(open_fn, path):
    try:
        return "ok", _summary(open_fn(path))
    except Exception as error:
        return "error", f"{type(error).__name__}: {error}"


def _stock(path):
    return GGUFReader(str(path), "r")


@pytest.mark.parametrize("endian", [GGUFEndian.LITTLE, GGUFEndian.BIG])
def test_reader_matches_stock_without_per_entry_views(llama_cpp, tmp_path, endian):
    path = _write(tmp_path / "model.gguf", endian = endian)
    stock_outcome = _outcome(_stock, path)
    assert _outcome(llama_cpp._open_gguf_reader, str(path)) == stock_outcome
    if stock_outcome[0] == "error":
        # An old gguf on numpy 2 cannot read the file either way; the same error is the contract.
        pytest.skip(f"the installed gguf cannot read this file: {stock_outcome[1]}")
    fast = llama_cpp._open_gguf_reader(str(path))
    stock = _stock(path)
    assert _summary(fast) == _summary(stock)
    if hasattr(stock.fields["tokenizer.ggml.tokens"], "contents"):
        tokens = fast.fields["tokenizer.ggml.tokens"]
        assert tokens.contents(0) == stock.fields["tokenizer.ggml.tokens"].contents(0)
        for key in ("tokenizer.ggml.scores", "tokenizer.ggml.token_type"):
            assert fast.fields[key].contents() == stock.fields[key].contents()
    # The point: the vocabulary is not split into one view per entry.
    fast_parts = sum(len(field.parts) for field in fast.fields.values())
    stock_parts = sum(len(field.parts) for field in stock.fields.values())
    assert stock_parts > 4 * VOCAB
    assert fast_parts < 200, (fast_parts, stock_parts)


def test_small_and_empty_vocabularies_are_stock(llama_cpp, tmp_path):
    for vocab in (0, 1, 1024, 1025):
        path = _write(tmp_path / f"v{vocab}.gguf", vocab = vocab)
        assert _outcome(llama_cpp._open_gguf_reader, str(path)) == _outcome(_stock, path)


def test_corrupt_files_fail_like_stock(llama_cpp, tmp_path):
    raw = _write(tmp_path / "model.gguf").read_bytes()
    key = raw.find(b"tokenizer.ggml.tokens")
    count_at = key + len("tokenizer.ggml.tokens") + 8
    first = count_at + 8
    middle = first
    for _ in range(VOCAB // 2):
        middle += 8 + struct.unpack_from("<Q", raw, middle)[0]
    merges = raw.find(b"tokenizer.ggml.merges")
    scores = raw.find(b"tokenizer.ggml.scores")
    variants = {}
    for cut in (0, 7, 30, first + 3, first + 9, middle + 5, merges - 2, merges + 60,
                scores + 40, scores + 1000, len(raw) - 70, len(raw) - 1):
        variants[f"cut{cut}"] = raw[:cut]

    def patched(offset, value):
        data = bytearray(raw)
        data[offset:offset + 8] = struct.pack("<Q", value)
        return bytes(data)

    variants["huge_first"] = patched(first, 1 << 40)
    variants["huge_middle"] = patched(middle, 1 << 33)
    variants["short_middle"] = patched(middle, 3)
    variants["huge_count"] = patched(count_at, 1 << 50)
    variants["short_count"] = patched(count_at, VOCAB - 7)
    for name, data in variants.items():
        path = tmp_path / f"{name}.gguf"
        path.write_bytes(data)
        assert _outcome(llama_cpp._open_gguf_reader, str(path)) == _outcome(_stock, path), name


@pytest.mark.parametrize("limit, value", [
    ("GGUF_MAX_ARRAY_ELEMENTS", 2000),
    ("GGUF_MAX_STRING_LENGTH", 100),
])
def test_stock_length_limits_are_left_to_the_stock_parser(llama_cpp, tmp_path, monkeypatch, limit, value):
    # Newer gguf-py rejects arrays and strings past module-level bounds inside the
    # per-entry parser. Past a bound the fast path must step aside, so the stock
    # parser raises (or, on releases without the check, reads) exactly as before.
    import gguf.gguf_reader as stock_module

    path = tmp_path / "long_token.gguf"
    w = GGUFWriter(str(path), "llama")
    w.add_block_count(2)
    w.add_token_list(["t" * 300 if i == 3000 else f"t{i}" for i in range(VOCAB)])
    w.add_token_scores([float(i % 7) for i in range(VOCAB)])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    monkeypatch.setattr(stock_module, limit, value, raising = False)
    stock_outcome = _outcome(_stock, path)
    assert _outcome(llama_cpp._open_gguf_reader, str(path)) == stock_outcome
    if stock_outcome[0] == "ok":
        fast = llama_cpp._open_gguf_reader(str(path))
        stock = _stock(path)
        for key in ("tokenizer.ggml.tokens", "tokenizer.ggml.scores") if limit.endswith("ELEMENTS") else ("tokenizer.ggml.tokens",):
            assert len(fast.fields[key].parts) == len(stock.fields[key].parts), key

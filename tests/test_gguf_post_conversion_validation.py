"""Post-conversion GGUF verification: unsloth#6056, unsloth#8360, unsloth#8513.

On main the validator in `unsloth_zoo/llama_cpp.py` is unreachable, blind and
broken at once: nothing calls it, its reader is rewritten so that only the
final tensor is built, its vocabulary comparison sits behind
`hasattr(dict, "tokenizer.ggml.tokens")` and never runs, and its tensor
comparison raises a torch broadcast error on a correct export. There is no
check at all that a GGUF carries the metadata llama.cpp requires.

Everything here builds its GGUFs with `gguf.GGUFWriter`, so the suite needs no
network, no model and no llama.cpp build.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")
from gguf import GGMLQuantizationType, GGUFWriter  # noqa: E402


def _load_llama_cpp_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "unsloth_zoo" / "llama_cpp.py"
    spec = importlib.util.spec_from_file_location("llama_cpp_gguf_verify", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope = "module")
def llama_cpp():
    return _load_llama_cpp_module()


# The three keys `llama_model_base::load_hparams` reads for every architecture.
UNIVERSAL = {
    "llama.context_length": 4096,
    "llama.embedding_length": 64,
    "llama.block_count": 2,
}
INDEXER = {
    "llama.attention.indexer.head_count": 4,
    "llama.attention.indexer.key_length": 32,
    "llama.attention.indexer.top_k": 8,
}


def write_gguf(path, architecture = "llama", keys = None, tensors = None,
               raw_dtype = None):
    """A minimal GGUF with exactly the KV keys and tensors asked for.

    `tensors` maps a name to an array, or to None for a default 2x2 of ones.
    `raw_dtype` declares the arrays as a quantized GGML type, so the bytes are
    block-format bytes rather than values; that is what the gate reads to decide
    whether a float view of them would mean anything.
    """
    writer = GGUFWriter(str(path), architecture)
    for key, value in (keys or {}).items():
        if isinstance(value, str):
            writer.add_string(key, value)
        else:
            writer.add_uint32(key, int(value))
    for name, array in (tensors or {}).items():
        if array is None:
            array = np.ones((2, 2), dtype = np.float32)
        if raw_dtype is None:
            writer.add_tensor(name, array)
        else:
            writer.add_tensor(
                name, array, raw_shape = array.shape, raw_dtype = raw_dtype,
            )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return str(path)


# ---------------------------------------------------------------------------
# The required metadata gate (unsloth#8360, unsloth#8513)
# ---------------------------------------------------------------------------

def test_complete_file_has_no_problems(llama_cpp, tmp_path):
    """The gate must be silent on a correct file. A check that fires on a good
    export is worse than no check."""
    path = write_gguf(tmp_path / "ok.gguf", keys = UNIVERSAL,
                      tensors = {"blk.0.attn_q.weight": None})
    assert llama_cpp.gguf_metadata_problems(path) == []


@pytest.mark.parametrize("missing", sorted(UNIVERSAL))
def test_missing_universal_key_is_reported(llama_cpp, tmp_path, missing):
    keys = {k: v for k, v in UNIVERSAL.items() if k != missing}
    path = write_gguf(tmp_path / "bad.gguf", keys = keys,
                      tensors = {"blk.0.attn_q.weight": None})
    problems = llama_cpp.gguf_metadata_problems(path)
    assert any(missing in problem for problem in problems), problems


def test_missing_architecture_is_reported(llama_cpp, tmp_path):
    """Without `general.architecture` llama.cpp cannot pick a loader at all."""
    path = write_gguf(tmp_path / "noarch.gguf", architecture = "",
                      keys = UNIVERSAL, tensors = {"blk.0.attn_q.weight": None})
    problems = llama_cpp.gguf_metadata_problems(path)
    assert any("general.architecture" in problem for problem in problems), problems


def test_indexer_tensor_without_indexer_metadata_is_reported(llama_cpp, tmp_path):
    """unsloth#8360 / unsloth#8513 in one file: the MiniMax M3 quants carried
    `blk.N.indexer.*` tensors but not `{arch}.attention.indexer.head_count`,
    which every architecture defining those tensors reads as required."""
    path = write_gguf(
        tmp_path / "indexer.gguf", keys = UNIVERSAL,
        tensors = {"blk.0.attn_q.weight": None, "blk.0.indexer.k_proj.weight": None},
    )
    problems = llama_cpp.gguf_metadata_problems(path)
    reported = " ".join(problems)
    for key in INDEXER:
        assert key in reported, problems
    assert "blk.0.indexer.k_proj.weight" in reported, problems


def test_the_minimax_shape_is_still_refused_for_a_requiring_architecture(llama_cpp, tmp_path):
    """The exemption below must not weaken the case the gate exists for.
    minimax-m3.cpp:21-25 reads all three keys with no `false`."""
    path = write_gguf(
        tmp_path / "minimax.gguf", architecture = "minimax-m3", keys = {
            "minimax-m3.context_length": 8,
            "minimax-m3.embedding_length": 4,
            "minimax-m3.block_count": 1,
        },
        tensors = {"blk.0.attn_q.weight": None, "blk.0.indexer.k_proj.weight": None},
    )
    reported = " ".join(llama_cpp.gguf_metadata_problems(path))
    assert "minimax-m3.attention.indexer.head_count" in reported


def test_an_architecture_that_reads_the_indexer_keys_optionally_is_exempt(llama_cpp, tmp_path):
    """src/models/hy-v4.cpp:46-48 reads head_count, key_length and top_k with a
    trailing `false`, and creates the indexer tensors only when top_k came out
    non-zero, so a `hy_v4` file without them loads rather than failing. The gate
    refuses to publish, so it must not fire on a file llama.cpp accepts."""
    path = write_gguf(
        tmp_path / "hyv4.gguf", architecture = "hy_v4", keys = {
            "hy_v4.context_length": 8,
            "hy_v4.embedding_length": 4,
            "hy_v4.block_count": 1,
        },
        tensors = {"blk.0.attn_q.weight": None, "blk.0.indexer.k_proj.weight": None},
    )
    assert llama_cpp.gguf_metadata_problems(path) == []


def test_every_conditional_entry_declares_its_exempt_architectures(llama_cpp):
    """Shape guard: the exempt set is the fourth element, and a three element
    entry would be silently unpacked wrong by the scan."""
    tables = (
        llama_cpp.GGUF_CONDITIONAL_REQUIRED_KEYS,
        llama_cpp.GGUF_CONDITIONAL_ADVISORY_KEYS,
    )
    for table in tables:
        for entry in table:
            marker, templates, exempt, reason = entry
            assert isinstance(marker, str) and marker
            assert templates and all("{arch}" in t for t in templates)
            assert isinstance(exempt, frozenset)
            assert isinstance(reason, str) and reason


def test_indexer_metadata_present_is_accepted(llama_cpp, tmp_path):
    path = write_gguf(
        tmp_path / "indexer_ok.gguf", keys = {**UNIVERSAL, **INDEXER},
        tensors = {"blk.0.attn_q.weight": None, "blk.0.indexer.k_proj.weight": None},
    )
    assert llama_cpp.gguf_metadata_problems(path) == []


def test_indexer_compressor_tensors_also_imply_the_keys(llama_cpp, tmp_path):
    """llama.cpp spells some of them with an underscore
    (`blk.N.indexer_compressor_kv`), so a `indexer.` prefix test would miss them."""
    path = write_gguf(
        tmp_path / "compressor.gguf", keys = UNIVERSAL,
        tensors = {"blk.0.indexer_compressor_kv.weight": None},
    )
    problems = llama_cpp.gguf_metadata_problems(path)
    assert any("indexer.head_count" in problem for problem in problems), problems


def test_ssm_tensor_implies_conv_kernel(llama_cpp, tmp_path):
    path = write_gguf(tmp_path / "ssm.gguf", keys = UNIVERSAL,
                      tensors = {"blk.0.ssm_conv1d.weight": None})
    problems = llama_cpp.gguf_metadata_problems(path)
    assert any("llama.ssm.conv_kernel" in problem for problem in problems), problems


def test_expert_tensors_imply_expert_counts_as_a_warning(llama_cpp, tmp_path):
    """Without `expert_count` llama.cpp creates no expert tensor and ignores
    every one it finds, so the model loads and is silently wrong.

    A WARNING, not a refusal: llama.cpp does load the file, and a converter can
    legitimately reach this state (conversion/hunyuan.py pops `num_experts`
    before super().set_gguf_parameters() and restores it afterwards), so
    refusing would turn an export that works today into a hard failure."""
    path = write_gguf(tmp_path / "moe.gguf", keys = UNIVERSAL,
                      tensors = {"blk.0.ffn_gate_exps.weight": None})
    assert llama_cpp.gguf_metadata_problems(path) == []
    warnings = llama_cpp.gguf_metadata_warnings(path)
    reported = " ".join(warnings)
    assert "llama.expert_count" in reported, warnings
    assert "llama.expert_used_count" in reported, warnings


def test_an_moe_export_missing_expert_count_is_not_refused(llama_cpp, tmp_path, caplog):
    """The gate must publish it and say so, not raise."""
    import logging
    path = write_gguf(tmp_path / "moe_ok.gguf", keys = UNIVERSAL,
                      tensors = {"blk.0.ffn_gate_exps.weight": None})
    with caplog.at_level(logging.WARNING):
        llama_cpp._verify_converted_gguf([path])
    assert "llama.expert_count" in caplog.text


def test_mmproj_is_exempt(llama_cpp, tmp_path):
    """`load_hparams` returns before reading any hparam when the architecture is
    `clip`, the dummy every mmproj file carries, so requiring them would reject
    every projector Unsloth produces."""
    path = write_gguf(tmp_path / "mmproj.gguf", architecture = "clip", keys = {},
                      tensors = {"blk.0.attn_q.weight": None, "mm.0.weight": None})
    assert llama_cpp.gguf_metadata_problems(path) == []


def test_file_without_blocks_is_exempt(llama_cpp, tmp_path):
    """A vocabulary-only GGUF has no transformer blocks and llama.cpp skips the
    hparams for it."""
    path = write_gguf(tmp_path / "vocab.gguf", keys = {},
                      tensors = {"token_embd.weight": None})
    assert llama_cpp.gguf_metadata_problems(path) == []


def test_unreadable_file_is_reported_not_raised(llama_cpp, tmp_path):
    path = tmp_path / "junk.gguf"
    path.write_bytes(b"not a gguf at all")
    problems = llama_cpp.gguf_metadata_problems(str(path))
    assert problems and "could not be read" in problems[0], problems


# ---------------------------------------------------------------------------
# Split exports
# ---------------------------------------------------------------------------

def test_any_shard_resolves_to_the_whole_set(llama_cpp, tmp_path):
    """Only shard 1 carries the KV metadata. Handed shard 2, the gate must read
    shard 1's metadata rather than report a missing architecture, and must see
    shard 2's tensors."""
    write_gguf(tmp_path / "m-00001-of-00002.gguf", keys = UNIVERSAL,
               tensors = {"blk.0.attn_q.weight": None})
    second = write_gguf(tmp_path / "m-00002-of-00002.gguf", architecture = "llama",
                        keys = {}, tensors = {"blk.1.indexer.k_proj.weight": None})

    siblings = llama_cpp._gguf_shard_siblings(second)
    assert [Path(p).name for p in siblings] == [
        "m-00001-of-00002.gguf", "m-00002-of-00002.gguf",
    ]
    problems = llama_cpp.gguf_metadata_problems(second)
    # The architecture came from shard 1, and the indexer tensor in shard 2 was
    # seen, so the missing indexer keys are what is reported.
    assert not any("general.architecture" in problem for problem in problems), problems
    assert any("indexer.head_count" in problem for problem in problems), problems


def test_a_lone_file_is_its_own_shard_set(llama_cpp, tmp_path):
    path = write_gguf(tmp_path / "solo.gguf", keys = UNIVERSAL,
                      tensors = {"blk.0.attn_q.weight": None})
    assert llama_cpp._gguf_shard_siblings(path) == [path]


# ---------------------------------------------------------------------------
# Tensor sanity (unsloth#6056)
# ---------------------------------------------------------------------------

def test_a_damaged_non_final_tensor_is_caught(llama_cpp, tmp_path):
    """The point of unsloth#6056. On main the reader was rewritten to build
    `tensors_fields[-1:]`, so only the last tensor existed and a zeroed text
    tower was structurally invisible."""
    path = write_gguf(
        tmp_path / "zeroed.gguf", keys = UNIVERSAL,
        tensors = {
            "blk.0.attn_q.weight": np.ones((8, 8), dtype = np.float32),
            "blk.1.attn_q.weight": np.zeros((8, 8), dtype = np.float32),
            "output_norm.weight":  np.ones((8,), dtype = np.float32),
        },
    )
    problems = llama_cpp.gguf_tensor_problems(path)
    assert any("blk.1.attn_q.weight" in problem and "entirely zero" in problem
               for problem in problems), problems


def test_a_clean_file_has_no_tensor_problems(llama_cpp, tmp_path):
    path = write_gguf(
        tmp_path / "clean.gguf", keys = UNIVERSAL,
        tensors = {f"blk.{i}.attn_q.weight": np.full((8, 8), 0.5, dtype = np.float32)
                   for i in range(6)},
    )
    assert llama_cpp.gguf_tensor_problems(path) == []


def test_non_finite_values_are_caught(llama_cpp, tmp_path):
    bad = np.ones((8, 8), dtype = np.float32)
    bad[0, 0] = np.nan
    path = write_gguf(tmp_path / "nan.gguf", keys = UNIVERSAL,
                      tensors = {"blk.0.attn_q.weight": bad})
    problems = llama_cpp.gguf_tensor_problems(path)
    assert any("NaN or Inf" in problem for problem in problems), problems


def test_a_zero_bias_is_not_a_problem(llama_cpp, tmp_path):
    """A zero bias is normal; a zero weight matrix is not."""
    path = write_gguf(
        tmp_path / "bias.gguf", keys = UNIVERSAL,
        tensors = {
            "blk.0.attn_q.weight": np.ones((8, 8), dtype = np.float32),
            "blk.0.attn_q.bias":   np.zeros((8,), dtype = np.float32),
        },
    )
    assert llama_cpp.gguf_tensor_problems(path) == []


def test_a_zeroed_f16_tensor_is_caught(llama_cpp, tmp_path):
    """gguf-py hands F16 and BF16 tensors back as raw bytes rather than as a
    numpy float dtype, so a plain `np.issubdtype(..., np.floating)` guard
    skipped every half precision tensor, which is what a bf16 export is made
    of."""
    path = write_gguf(
        tmp_path / "f16.gguf", keys = UNIVERSAL,
        tensors = {
            "blk.0.attn_q.weight": np.ones((8, 8), dtype = np.float16),
            "blk.1.attn_q.weight": np.zeros((8, 8), dtype = np.float16),
        },
    )
    problems = llama_cpp.gguf_tensor_problems(path)
    assert any("blk.1.attn_q.weight" in problem for problem in problems), problems


def test_an_f16_infinity_is_caught(llama_cpp, tmp_path):
    bad = np.ones((8, 8), dtype = np.float16)
    bad[3, 3] = np.inf
    path = write_gguf(tmp_path / "f16inf.gguf", keys = UNIVERSAL,
                      tensors = {"blk.0.attn_q.weight": bad})
    problems = llama_cpp.gguf_tensor_problems(path)
    assert any("NaN or Inf" in problem for problem in problems), problems


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_sampling_is_deterministic_and_covers_the_ends(llama_cpp):
    first = llama_cpp._gguf_sample_indices(236, 8, ("model.gguf", 236))
    again = llama_cpp._gguf_sample_indices(236, 8, ("model.gguf", 236))
    assert first == again
    assert first[0] == 0 and first[-1] == 235
    assert len(first) == 10
    other = llama_cpp._gguf_sample_indices(236, 8, ("other.gguf", 236))
    assert other != first


def test_sampling_returns_everything_for_a_small_file(llama_cpp):
    assert llama_cpp._gguf_sample_indices(5, 8, ("x", 5)) == [0, 1, 2, 3, 4]
    assert llama_cpp._gguf_sample_indices(0, 8, ("x", 0)) == []


def test_sample_size_env_is_read_at_the_call(llama_cpp, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GGUF_VERIFY_TENSORS", "3")
    assert llama_cpp._gguf_sample_size() == 3
    monkeypatch.setenv("UNSLOTH_GGUF_VERIFY_TENSORS", "not a number")
    assert llama_cpp._gguf_sample_size() == llama_cpp.GGUF_VERIFY_SAMPLE_DEFAULT
    monkeypatch.delenv("UNSLOTH_GGUF_VERIFY_TENSORS")
    assert llama_cpp._gguf_sample_size() == llama_cpp.GGUF_VERIFY_SAMPLE_DEFAULT


# ---------------------------------------------------------------------------
# The gate `convert_to_gguf` runs
# ---------------------------------------------------------------------------

def test_the_gate_refuses_a_file_missing_required_metadata(llama_cpp, tmp_path):
    keys = {k: v for k, v in UNIVERSAL.items() if k != "llama.block_count"}
    path = write_gguf(tmp_path / "bad.gguf", keys = keys,
                      tensors = {"blk.0.attn_q.weight": None})
    with pytest.raises(RuntimeError, match = "did not pass post conversion verification"):
        llama_cpp._verify_converted_gguf([path])


def test_the_gate_accepts_a_good_file(llama_cpp, tmp_path):
    path = write_gguf(tmp_path / "ok.gguf", keys = UNIVERSAL,
                      tensors = {"blk.0.attn_q.weight": None})
    llama_cpp._verify_converted_gguf([path])


def test_a_legitimately_zero_tensor_outside_the_blocks_is_allowed(llama_cpp, tmp_path):
    """BERT-family `token_types.weight` is all zeros whenever the model has a
    single trained segment type, and `BertModel` is in llama.cpp's converter
    registry. Outside `blk.` zero is a real weight, so rejecting it would refuse
    a good export."""
    path = write_gguf(
        tmp_path / "bert.gguf", keys = UNIVERSAL, tensors = {
            "blk.0.attn_q.weight": np.ones((8, 8), dtype = np.float32),
            "token_types.weight": np.zeros((2, 8), dtype = np.float32),
            "position_embd.weight": np.zeros((4, 8), dtype = np.float32),
        },
    )
    assert llama_cpp.gguf_tensor_problems(path) == []


def test_a_zero_block_weight_is_still_caught(llama_cpp, tmp_path):
    """The complement, and the unsloth#6056 detection itself."""
    path = write_gguf(
        tmp_path / "damaged.gguf", keys = UNIVERSAL, tensors = {
            "blk.0.attn_q.weight": np.ones((8, 8), dtype = np.float32),
            "blk.5.attn_q.weight": np.zeros((8, 8), dtype = np.float32),
        },
    )
    reported = " ".join(llama_cpp.gguf_tensor_problems(path))
    assert "blk.5.attn_q.weight" in reported


def test_a_non_finite_value_outside_the_blocks_is_still_caught(llama_cpp, tmp_path):
    """Narrowing the all-zero test must not narrow the NaN test: a non-finite
    value is never a legitimate weight anywhere in the file."""
    bad = np.ones((8, 8), dtype = np.float32)
    bad[0, 0] = np.nan
    path = write_gguf(
        tmp_path / "nan_outside.gguf", keys = UNIVERSAL, tensors = {
            "blk.0.attn_q.weight": np.ones((8, 8), dtype = np.float32),
            "token_embd.weight": bad,
        },
    )
    reported = " ".join(llama_cpp.gguf_tensor_problems(path))
    assert "token_embd.weight" in reported


def test_f16_takes_the_float_dtype_path(llama_cpp, tmp_path):
    """The F16 entry was removed from the bytewise table because gguf-py returns
    a real float16 dtype. If that ever changes, every F16 tensor would silently
    stop being checked, so pin it."""
    bad = np.ones((8, 8), dtype = np.float16)
    bad[0, 0] = np.nan
    path = write_gguf(tmp_path / "f16.gguf", keys = UNIVERSAL,
                      tensors = {"blk.0.attn_q.weight": bad})
    readers = llama_cpp._gguf_open_shards(path)
    tensor = readers[0][1].tensors[0]
    values, mask = llama_cpp._gguf_float_view(tensor)
    assert values is not None and mask is None, "F16 no longer a numpy float dtype"
    assert "blk.0.attn_q.weight" in " ".join(llama_cpp.gguf_tensor_problems(path))


def test_the_gate_skips_tensor_values_on_a_quantized_export(llama_cpp, tmp_path):
    """A quantized block format is bytes, not values, so the tensor checks run
    only for a file whose tensors are plain floats. The metadata gate still runs.

    Read off the FILE, not off the caller's `quantization_type`: one conversion
    writes a `q4_k_m` text half and a bf16 projector, and a per-call dtype left
    the projector unchecked.
    """
    zeros = np.zeros((32, 32), dtype = np.float32)
    # Float tensors: the zeroed weight is refused.
    plain = write_gguf(tmp_path / "plain.gguf", keys = UNIVERSAL,
                       tensors = {"blk.0.attn_q.weight": zeros})
    with pytest.raises(RuntimeError, match = "entirely zero"):
        llama_cpp._verify_converted_gguf([plain])
    # The same bytes declared as a quantized type: no float view exists, so the
    # tensor checks do not run and the file passes.
    quant = write_gguf(tmp_path / "quant.gguf", keys = UNIVERSAL,
                       tensors = {"blk.0.attn_q.weight": zeros},
                       raw_dtype = GGMLQuantizationType.Q8_0)
    llama_cpp._verify_converted_gguf([quant])


def test_the_dtype_argument_is_ignored(llama_cpp, tmp_path):
    """`quantization_type` is accepted for compatibility and ignored: the file is
    the authority on what it holds, which is what makes a mixed-dtype VLM export
    checkable at all."""
    zeros = np.zeros((32, 32), dtype = np.float32)
    path = write_gguf(tmp_path / "ignored.gguf", keys = UNIVERSAL,
                      tensors = {"blk.0.attn_q.weight": zeros})
    for claimed in (None, "bf16", "q4_k_m", "nonsense"):
        with pytest.raises(RuntimeError, match = "entirely zero"):
            llama_cpp._verify_converted_gguf([path], claimed)


def test_a_quantized_file_is_detected_from_its_own_tensors(llama_cpp, tmp_path):
    ones = np.ones((32, 32), dtype = np.float32)
    plain = write_gguf(tmp_path / "f32.gguf", keys = UNIVERSAL,
                       tensors = {"blk.0.attn_q.weight": ones})
    quant = write_gguf(tmp_path / "q8.gguf", keys = UNIVERSAL,
                       tensors = {"blk.0.attn_q.weight": ones},
                       raw_dtype = GGMLQuantizationType.Q8_0)
    assert llama_cpp._gguf_holds_only_float_tensors(llama_cpp._gguf_open_shards(plain))
    assert not llama_cpp._gguf_holds_only_float_tensors(llama_cpp._gguf_open_shards(quant))


def test_the_gate_can_be_turned_off(llama_cpp, tmp_path, monkeypatch):
    keys = {k: v for k, v in UNIVERSAL.items() if k != "llama.block_count"}
    path = write_gguf(tmp_path / "bad.gguf", keys = keys,
                      tensors = {"blk.0.attn_q.weight": None})
    monkeypatch.setenv("UNSLOTH_GGUF_VERIFY", "0")
    llama_cpp._verify_converted_gguf([path])


def test_the_gate_checks_a_split_set_once(llama_cpp, tmp_path):
    """Handed every shard of one export, the gate must not re-read the set once
    per shard."""
    first = write_gguf(tmp_path / "m-00001-of-00002.gguf", keys = UNIVERSAL,
                       tensors = {"blk.0.attn_q.weight": None})
    second = write_gguf(tmp_path / "m-00002-of-00002.gguf", keys = {},
                        tensors = {"blk.1.attn_q.weight": None})
    opened = []
    original = llama_cpp._open_gguf_reader
    def counting(path):
        opened.append(path)
        return original(path)
    llama_cpp._open_gguf_reader = counting
    try:
        llama_cpp._verify_converted_gguf([first, second])
    finally:
        llama_cpp._open_gguf_reader = original
    assert len(opened) == 2, opened


# ---------------------------------------------------------------------------
# What the rebuild removed
# ---------------------------------------------------------------------------

def test_the_reader_is_no_longer_rebuilt_with_exec(llama_cpp):
    """The old validator ran `inspect.getsource(GGUFReader.__init__)` through
    `exec` with `self._build_tensors(offs, tensors_fields` rewritten to
    `[-1:]`. Measured on a 542 MB export that bought nothing (6.69 s and 736 MB
    peak against 7.48 s and the same 736 MB, because the cost is
    `_build_fields` parsing the vocabulary) and it raised whenever gguf-py
    refactored that line."""
    # The docstrings still name the removed rewrite so its absence stays
    # explained, so this parses the file rather than grepping it.
    import ast
    source = Path(llama_cpp.__file__).read_text()
    tree = ast.parse(source)
    verifiers = {
        "gguf_metadata_problems", "gguf_tensor_problems", "_verify_converted_gguf",
        "_assert_correct_gguf", "assert_correct_gguf", "_open_gguf_reader",
        "_gguf_open_shards", "_gguf_degenerate_problem", "_gguf_float_view",
        "_gguf_tokenizer_problems", "_gguf_shape_problems", "_gguf_sample_indices",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in verifiers:
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                assert inner.func.id not in ("exec", "eval", "compile"), \
                    f"{node.name} still calls {inner.func.id}"


def test_the_public_surface_is_exported(llama_cpp):
    for name in ("assert_correct_gguf", "gguf_metadata_problems", "gguf_tensor_problems"):
        assert name in llama_cpp.__all__, name
        assert callable(getattr(llama_cpp, name)), name


def test_an_unreadable_output_warns_rather_than_refusing(llama_cpp, tmp_path, caplog):
    """The installed `gguf` can be older than the converter that wrote the file,
    so a reader failure is not evidence the export is bad. Refusing here would
    break conversions that work today, and `convert_to_gguf` already rejects a
    missing or truncated output before this runs."""
    import logging
    path = tmp_path / "stub.gguf"
    path.write_bytes(b"not a gguf")
    with caplog.at_level(logging.WARNING):
        llama_cpp._verify_converted_gguf([str(path)])
    assert any("was not verified" in record.message for record in caplog.records), caplog.text


def test_a_split_set_missing_its_first_shard_says_so(llama_cpp, tmp_path):
    """Only shard 1 carries the KV metadata, so without it every required key
    reads as missing. The report has to name the absent file instead."""
    second = write_gguf(tmp_path / "m-00002-of-00002.gguf", keys = {},
                        tensors = {"blk.1.attn_q.weight": None})
    problems = llama_cpp.gguf_metadata_problems(second)
    assert len(problems) == 1, problems
    assert "m-00001-of-00002.gguf" in problems[0], problems
    assert "not present" in problems[0], problems

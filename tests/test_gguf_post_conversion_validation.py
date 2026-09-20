# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Post-conversion GGUF verification: unsloth#6056, unsloth#8360, unsloth#8513.
Builds its GGUFs with `gguf.GGUFWriter`, so no network, model or llama.cpp build.
"""

import importlib.util
import os
import pathlib
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
    block-format bytes rather than values.
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


# --- The required metadata gate (unsloth#8360, unsloth#8513) ---

def test_complete_file_has_no_problems(llama_cpp, tmp_path):
    """The gate must be silent on a correct file."""
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
    """unsloth#8360 / unsloth#8513: `blk.N.indexer.*` tensors without
    `{arch}.attention.indexer.head_count`."""
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
    """The exemption below must not weaken the case the gate exists for:
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
    """src/models/hy-v4.cpp:46-48 reads the three keys with a trailing `false`, so
    a `hy_v4` file without them loads and the gate must not fire on it."""
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
    """Shape guard: the exempt set is the fourth element of every entry."""
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
    """llama.cpp spells some with an underscore (`blk.N.indexer_compressor_kv`), so
    an `indexer.` prefix test would miss them."""
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
    """Without `expert_count` the model loads and is silently wrong, but a converter
    can legitimately reach this state, so it is a warning and not a refusal."""
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
    """`load_hparams` reads no hparam for the `clip` architecture every mmproj carries."""
    path = write_gguf(tmp_path / "mmproj.gguf", architecture = "clip", keys = {},
                      tensors = {"blk.0.attn_q.weight": None, "mm.0.weight": None})
    assert llama_cpp.gguf_metadata_problems(path) == []


def test_file_without_blocks_is_exempt(llama_cpp, tmp_path):
    """A vocabulary-only GGUF has no blocks and llama.cpp skips the hparams."""
    path = write_gguf(tmp_path / "vocab.gguf", keys = {},
                      tensors = {"token_embd.weight": None})
    assert llama_cpp.gguf_metadata_problems(path) == []


def test_unreadable_file_is_reported_not_raised(llama_cpp, tmp_path):
    path = tmp_path / "junk.gguf"
    path.write_bytes(b"not a gguf at all")
    problems = llama_cpp.gguf_metadata_problems(str(path))
    assert problems and "could not be read" in problems[0], problems


# --- Split exports ---

def test_any_shard_resolves_to_the_whole_set(llama_cpp, tmp_path):
    """Only shard 1 carries the KV metadata, so handed shard 2 the gate must read
    shard 1 and still see shard 2's tensors."""
    write_gguf(tmp_path / "m-00001-of-00002.gguf", keys = UNIVERSAL,
               tensors = {"blk.0.attn_q.weight": None})
    second = write_gguf(tmp_path / "m-00002-of-00002.gguf", architecture = "llama",
                        keys = {}, tensors = {"blk.1.indexer.k_proj.weight": None})

    siblings = llama_cpp._gguf_shard_siblings(second)
    assert [Path(p).name for p in siblings] == [
        "m-00001-of-00002.gguf", "m-00002-of-00002.gguf",
    ]
    problems = llama_cpp.gguf_metadata_problems(second)
    assert not any("general.architecture" in problem for problem in problems), problems
    assert any("indexer.head_count" in problem for problem in problems), problems


def test_a_lone_file_is_its_own_shard_set(llama_cpp, tmp_path):
    path = write_gguf(tmp_path / "solo.gguf", keys = UNIVERSAL,
                      tensors = {"blk.0.attn_q.weight": None})
    assert llama_cpp._gguf_shard_siblings(path) == [path]


def test_a_missing_middle_shard_is_reported(llama_cpp, tmp_path):
    """Shard 1 present is not the same as the set being complete. The declared count
    lives only in the `-of-NNNNN` of the filename."""
    first = write_gguf(tmp_path / "m-00001-of-00003.gguf", keys = UNIVERSAL,
                       tensors = {"blk.0.attn_q.weight": None})
    write_gguf(tmp_path / "m-00003-of-00003.gguf", architecture = "llama",
               keys = {}, tensors = {"blk.2.attn_q.weight": None})
    # Shard 2 was never written.

    assert llama_cpp._gguf_missing_shards(first) == ["m-00002-of-00003.gguf"]
    problems = llama_cpp.gguf_metadata_problems(first)
    assert any("m-00002-of-00003.gguf" in problem for problem in problems), problems
    assert any("incomplete" in problem for problem in problems), problems
    # Reported from any member of the set, not only from shard 1.
    third = tmp_path / "m-00003-of-00003.gguf"
    assert any(
        "m-00002-of-00003.gguf" in problem
        for problem in llama_cpp.gguf_metadata_problems(str(third))
    )


def test_a_corrupt_later_shard_is_reported_as_a_problem(llama_cpp, tmp_path):
    """Present but unreadable is exactly as unloadable as absent, and the empty list
    is what a caller uses to ACCEPT a set."""
    first = write_gguf(tmp_path / "b-00001-of-00002.gguf", keys = UNIVERSAL,
                       tensors = {"blk.0.attn_q.weight": None})
    corrupt = tmp_path / "b-00002-of-00002.gguf"
    corrupt.write_bytes(b"not a gguf at all")

    problems = llama_cpp.gguf_metadata_problems(first)
    assert any("b-00002-of-00002.gguf" in problem for problem in problems), problems
    assert any("could not be read" in problem for problem in problems), problems

    # The fatal pass has it, so reporting it twice would read like two faults.
    assert llama_cpp.gguf_metadata_warnings(first) == []


def test_a_complete_shard_set_reports_nothing_missing(llama_cpp, tmp_path):
    """The control: the check must not fire on a set that is all there."""
    first = write_gguf(tmp_path / "c-00001-of-00002.gguf", keys = UNIVERSAL,
                       tensors = {"blk.0.attn_q.weight": None})
    write_gguf(tmp_path / "c-00002-of-00002.gguf", architecture = "llama",
               keys = {}, tensors = {"blk.1.attn_q.weight": None})

    assert llama_cpp._gguf_missing_shards(first) == []
    assert llama_cpp.gguf_metadata_problems(first) == []
    # A file that is not part of a split set has no declared count to be short of.
    solo = write_gguf(tmp_path / "solo2.gguf", keys = UNIVERSAL,
                      tensors = {"blk.0.attn_q.weight": None})
    assert llama_cpp._gguf_missing_shards(solo) == []


def test_assert_correct_gguf_checks_each_split_set_once(llama_cpp, tmp_path, monkeypatch):
    """A caller hands back the whole shard LIST, and every member used to resolve to
    the same set and revalidate it, reparsing a 40 shard model 40 times."""
    shards = [
        str(write_gguf(tmp_path / f"m-{index:05d}-of-00004.gguf",
                       keys = UNIVERSAL if index == 1 else {},
                       tensors = {f"blk.{index - 1}.attn_q.weight": None}))
        for index in range(1, 5)
    ]
    seen = []
    monkeypatch.setattr(
        llama_cpp, "_assert_correct_gguf",
        lambda name, *args, **kwargs: seen.append(name),
    )

    llama_cpp.assert_correct_gguf(shards, model = None, tokenizer = None)

    assert seen == [shards[0]], seen

    # Two genuinely different models are still both checked.
    other = str(write_gguf(tmp_path / "other.gguf", keys = UNIVERSAL,
                           tensors = {"blk.0.attn_q.weight": None}))
    seen.clear()
    llama_cpp.assert_correct_gguf(shards + [other], model = None, tokenizer = None)
    assert seen == [shards[0], other], seen


# --- Tensor sanity (unsloth#6056) ---

def test_a_damaged_non_final_tensor_is_caught(llama_cpp, tmp_path):
    """unsloth#6056: the reader built `tensors_fields[-1:]`, so only the last tensor
    existed and a zeroed text tower was invisible."""
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


def test_a_zeroed_identity_projection_is_not_a_problem(llama_cpp, tmp_path):
    """LLaMA Pro block expansion (arXiv 2401.02415) leaves o_proj and down_proj at
    exactly zero, and llama.cpp loads that file."""
    path = write_gguf(
        tmp_path / "identity.gguf", keys = UNIVERSAL,
        tensors = {
            "blk.0.attn_q.weight":      np.ones((8, 8), dtype = np.float32),
            "blk.3.attn_output.weight": np.zeros((8, 8), dtype = np.float32),
            "blk.3.ffn_down.weight":    np.zeros((8, 8), dtype = np.float32),
        },
    )
    assert llama_cpp.gguf_tensor_problems(path) == []


def test_an_ordinary_zeroed_block_weight_is_still_refused(llama_cpp, tmp_path):
    """The exemption above must not turn the all-zero check off elsewhere."""
    path = write_gguf(
        tmp_path / "damaged.gguf", keys = UNIVERSAL,
        tensors = {
            "blk.0.attn_q.weight":   np.ones((8, 8), dtype = np.float32),
            "blk.5.attn_q.weight":   np.zeros((8, 8), dtype = np.float32),
            "blk.5.ffn_up.weight":   np.zeros((8, 8), dtype = np.float32),
        },
    )
    problems = llama_cpp.gguf_tensor_problems(path)
    assert any("blk.5.attn_q.weight" in problem for problem in problems), problems
    assert any("blk.5.ffn_up.weight" in problem for problem in problems), problems


def test_a_zeroed_f16_tensor_is_caught(llama_cpp, tmp_path):
    """gguf-py can hand half precision tensors back as raw bytes, which a plain
    `np.issubdtype(..., np.floating)` guard skipped."""
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


# --- Sampling ---

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


# --- The gate `convert_to_gguf` runs ---

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
    """BERT-family `token_types.weight` is legitimately all zeros: outside `blk.`
    zero is a real weight."""
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
    """Narrowing the all-zero test must not narrow the NaN test."""
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
    """gguf-py returns a real float16 dtype; if that changes, every F16 tensor would
    silently stop being checked."""
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
    """A quantized block format is bytes, not values, so the tensor checks run only
    for float tensors, read off the FILE rather than the caller's dtype."""
    zeros = np.zeros((32, 32), dtype = np.float32)
    plain = write_gguf(tmp_path / "plain.gguf", keys = UNIVERSAL,
                       tensors = {"blk.0.attn_q.weight": zeros})
    with pytest.raises(RuntimeError, match = "entirely zero"):
        llama_cpp._verify_converted_gguf([plain])
    # The same bytes declared as a quantized type: no float view exists.
    quant = write_gguf(tmp_path / "quant.gguf", keys = UNIVERSAL,
                       tensors = {"blk.0.attn_q.weight": zeros},
                       raw_dtype = GGMLQuantizationType.Q8_0)
    llama_cpp._verify_converted_gguf([quant])


def test_the_dtype_argument_is_ignored(llama_cpp, tmp_path):
    """`quantization_type` is accepted for compatibility and ignored: the file is
    the authority on what it holds."""
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
    assert llama_cpp._gguf_holds_any_float_tensor(llama_cpp._gguf_open_shards(plain))
    assert not llama_cpp._gguf_holds_any_float_tensor(llama_cpp._gguf_open_shards(quant))


# --- Mixed files: float tensors stay checked, quantized blocks stay unchecked ---

def write_mixed_gguf(path, float_tensors, quantized_tensors, architecture = "llama",
                     keys = None, raw_dtype = GGMLQuantizationType.Q8_0):
    """One GGUF holding float tensors and quantized tensors together."""
    writer = GGUFWriter(str(path), architecture)
    for key, value in (keys or {}).items():
        if isinstance(value, str):
            writer.add_string(key, value)
        else:
            writer.add_uint32(key, int(value))
    for name, array in float_tensors.items():
        writer.add_tensor(name, array)
    for name, array in quantized_tensors.items():
        writer.add_tensor(name, array, raw_shape = array.shape, raw_dtype = raw_dtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return str(path)


def test_a_nan_projector_beside_quantized_blocks_is_still_caught(llama_cpp, tmp_path):
    """A `q4_k_m` VLM writes `mm.0.weight` at f16, and NaN there is just as fatal."""
    nan = np.full((32, 32), np.nan, dtype = np.float32)
    ones = np.ones((32, 32), dtype = np.float32)
    path = write_mixed_gguf(
        tmp_path / "mixed.gguf", keys = UNIVERSAL,
        float_tensors = {"mm.0.weight": nan},
        quantized_tensors = {"blk.0.attn_q.weight": ones},
    )
    with pytest.raises(RuntimeError, match = "NaN or Inf"):
        llama_cpp._verify_converted_gguf([path], "q4_k_m")


def test_a_nan_float_block_weight_beside_quantized_blocks_is_caught(llama_cpp, tmp_path):
    nan = np.full((32, 32), np.nan, dtype = np.float32)
    ones = np.ones((32, 32), dtype = np.float32)
    path = write_mixed_gguf(
        tmp_path / "mixed_blk.gguf", keys = UNIVERSAL,
        float_tensors = {"blk.0.attn_norm.weight": nan},
        quantized_tensors = {"blk.0.attn_q.weight": ones},
    )
    with pytest.raises(RuntimeError, match = "NaN or Inf"):
        llama_cpp._verify_converted_gguf([path], "q4_k_m")


def test_a_mixed_file_is_not_counted_as_value_skipped(llama_cpp, tmp_path, caplog):
    """A mixed file did get a value check, so the closing line must not claim the
    tensor pass was skipped."""
    ones = np.ones((32, 32), dtype = np.float32)
    path = write_mixed_gguf(
        tmp_path / "clean_mixed.gguf", keys = UNIVERSAL,
        float_tensors = {"mm.0.weight": ones},
        quantized_tensors = {"blk.0.attn_q.weight": ones},
    )
    with caplog.at_level("INFO"):
        llama_cpp._verify_converted_gguf([path], "q4_k_m", print_output = True)
    assert "metadata only" not in caplog.text.lower()


def test_quantized_block_bytes_are_left_alone_in_a_mixed_file(llama_cpp, tmp_path):
    """A quantized tensor whose block bytes are entirely zero must not be refused
    just because a float tensor put the file back in scope."""
    zeros = np.zeros((32, 32), dtype = np.float32)
    ones = np.ones((32, 32), dtype = np.float32)
    path = write_mixed_gguf(
        tmp_path / "zero_block.gguf", keys = UNIVERSAL,
        float_tensors = {"mm.0.weight": ones},
        quantized_tensors = {"blk.0.attn_q.weight": zeros},
    )
    llama_cpp._verify_converted_gguf([path], "q4_k_m")


def test_a_fully_quantized_file_still_gets_the_metadata_gate_only(llama_cpp, tmp_path):
    zeros = np.zeros((32, 32), dtype = np.float32)
    path = write_gguf(tmp_path / "allquant.gguf", keys = UNIVERSAL,
                      tensors = {"blk.0.attn_q.weight": zeros},
                      raw_dtype = GGMLQuantizationType.Q8_0)
    llama_cpp._verify_converted_gguf([path], "q8_0")


def test_verify_gguf_still_checks_quantized_tensors_by_default(llama_cpp, tmp_path):
    """`float_tensors_only` is the export gate's choice, not the default."""
    zeros = np.zeros((32, 32), dtype = np.float32)
    path = write_gguf(tmp_path / "user.gguf", keys = UNIVERSAL,
                      tensors = {"blk.0.attn_q.weight": zeros},
                      raw_dtype = GGMLQuantizationType.Q8_0)
    assert any("entirely zero" in p for p in llama_cpp.gguf_tensor_problems(path))
    assert llama_cpp.gguf_tensor_problems(path, float_tensors_only = True) == []


def test_the_gate_can_be_turned_off(llama_cpp, tmp_path, monkeypatch):
    keys = {k: v for k, v in UNIVERSAL.items() if k != "llama.block_count"}
    path = write_gguf(tmp_path / "bad.gguf", keys = keys,
                      tensors = {"blk.0.attn_q.weight": None})
    monkeypatch.setenv("UNSLOTH_GGUF_VERIFY", "0")
    llama_cpp._verify_converted_gguf([path])


def test_the_gate_checks_a_split_set_once(llama_cpp, tmp_path):
    """Handed every shard of one export, the gate reads the set once."""
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


# --- What the rebuild removed ---

def test_the_reader_is_no_longer_rebuilt_with_exec(llama_cpp):
    """The old validator rebuilt `GGUFReader.__init__` through `exec` to read only
    the final tensor. Parsed rather than grepped: the docstrings still name it."""
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
    """The installed `gguf` can be older than the converter that wrote the file, so
    a reader failure is not evidence the export is bad."""
    import logging
    path = tmp_path / "stub.gguf"
    path.write_bytes(b"not a gguf")
    with caplog.at_level(logging.WARNING):
        llama_cpp._verify_converted_gguf([str(path)])
    assert any("was not verified" in record.message for record in caplog.records), caplog.text


def test_an_unreadable_output_is_refused_when_its_own_writer_cannot_reopen_it(
    llama_cpp, tmp_path, monkeypatch
):
    """Version skew is the ONLY reason the failure above is a warning: a file its own
    writer cannot reopen is malformed or truncated."""
    import contextlib
    path = tmp_path / "stub.gguf"
    path.write_bytes(b"not a gguf")

    @contextlib.contextmanager
    def _pinned(directory):
        yield

    monkeypatch.setattr(llama_cpp, "use_local_gguf", _pinned)
    with pytest.raises(RuntimeError, match = "cannot read it back"):
        llama_cpp._verify_converted_gguf([str(path)], gguf_py_dir = str(tmp_path))


def test_a_writer_tree_that_cannot_be_imported_warns_rather_than_refusing(
    llama_cpp, tmp_path, monkeypatch, caplog
):
    """An ImportError never reached the file's bytes: a tree that imports in the
    child can still fail in the parent over a dependency the child had."""
    import contextlib, logging
    path = tmp_path / "fine.gguf"
    write_gguf(path, keys = UNIVERSAL, tensors = {"blk.0.attn_q.weight": None})

    @contextlib.contextmanager
    def _pinned(directory):
        yield

    def _import_fails(_path):
        raise ImportError("No module named 'sentencepiece'")

    monkeypatch.setattr(llama_cpp, "use_local_gguf", _pinned)
    monkeypatch.setattr(llama_cpp, "_open_gguf_reader", _import_fails)
    with caplog.at_level(logging.WARNING):
        llama_cpp._verify_converted_gguf([str(path)], gguf_py_dir = str(tmp_path))
    assert any("was not verified" in record.message for record in caplog.records), caplog.text
    assert any("sentencepiece" in record.message for record in caplog.records), caplog.text


def test_a_quantized_export_does_not_claim_its_values_were_checked(
    llama_cpp, tmp_path, capsys
):
    """The tensor pass does not run on a quantized export, so a bare "Verified"
    would claim a value check that never happened."""
    path = write_gguf(
        tmp_path / "quant.gguf", keys = UNIVERSAL,
        tensors = {"blk.0.attn_q.weight": np.zeros((32, 32), dtype = np.float32)},
        raw_dtype = GGMLQuantizationType.Q8_0,
    )
    llama_cpp._verify_converted_gguf([str(path)], print_output = True)
    out = capsys.readouterr().out
    assert "Verified 1 GGUF file(s)" in out, out
    assert "metadata only" in out, out


def test_the_refusal_names_the_file_and_the_way_out(llama_cpp, tmp_path, monkeypatch):
    """The refusal names the file and the opt-out."""
    import contextlib
    path = tmp_path / "broken-model.gguf"
    path.write_bytes(b"not a gguf")

    @contextlib.contextmanager
    def _pinned(directory):
        yield

    monkeypatch.setattr(llama_cpp, "use_local_gguf", _pinned)
    with pytest.raises(RuntimeError) as raised:
        llama_cpp._verify_converted_gguf([str(path)], gguf_py_dir = str(tmp_path))
    message = str(raised.value)
    assert "broken-model.gguf" in message, message
    assert "UNSLOTH_GGUF_VERIFY=0" in message, message


def test_a_split_set_missing_its_first_shard_says_so(llama_cpp, tmp_path):
    """Without shard 1 every required key reads as missing, so the report has to
    name the absent file instead."""
    second = write_gguf(tmp_path / "m-00002-of-00002.gguf", keys = {},
                        tensors = {"blk.1.attn_q.weight": None})
    problems = llama_cpp.gguf_metadata_problems(second)
    assert len(problems) == 1, problems
    assert "m-00001-of-00002.gguf" in problems[0], problems
    assert "not present" in problems[0], problems


def test_the_read_back_announces_itself_and_reports_its_cost(llama_cpp, tmp_path, capsys, monkeypatch):
    """A silent stall after "Successfully saved" reads like a hang, so the read-back
    says what it is doing, that the cost is flat, and how to turn it off."""
    path = tmp_path / "m.gguf"
    path.write_bytes(b"not a gguf")
    monkeypatch.setattr(llama_cpp, "_gguf_open_shards", lambda f: [(str(path), None)])

    llama_cpp._verify_converted_gguf([str(path)], "bf16", print_output = True)

    out = capsys.readouterr().out
    assert "Reading the GGUF back" in out
    assert "UNSLOTH_GGUF_VERIFY=0" in out, "the opt-out must be named where the cost is paid"
    assert "does not grow with the model" in out
    # Not "Verified 1": this file could not be read, so no check ran on it.
    assert "Verified" not in out, out
    assert "No GGUF file could be read back, so none was verified." in out


def test_a_file_that_was_really_checked_is_the_one_counted(llama_cpp, tmp_path, capsys):
    """The other half: a readable export still reports its count and its cost."""
    path = write_gguf(tmp_path / "ok.gguf", keys = UNIVERSAL,
                      tensors = {"blk.0.attn_q.weight": None})
    llama_cpp._verify_converted_gguf([path], "bf16", print_output = True)
    out = capsys.readouterr().out
    assert "Verified 1 GGUF file(s) in " in out and "s." in out, out


def test_an_unreadable_shard_is_not_counted_alongside_a_readable_file(
    llama_cpp, tmp_path, capsys, monkeypatch
):
    """Two exports, one readable and one not: the count is one, not two."""
    good = write_gguf(tmp_path / "good.gguf", keys = UNIVERSAL,
                      tensors = {"blk.0.attn_q.weight": None})
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"not a gguf")

    real_open = llama_cpp._gguf_open_shards

    def _open(target):
        if os.path.basename(str(target)) == "bad.gguf":
            return [(str(target), None)]
        return real_open(target)

    monkeypatch.setattr(llama_cpp, "_gguf_open_shards", _open)
    llama_cpp._verify_converted_gguf([good, str(bad)], "bf16", print_output = True)
    out = capsys.readouterr().out
    assert "Verified 1 GGUF file(s) in " in out, out
    assert "Verified 2" not in out, out


def test_the_announcement_is_not_printed_when_verification_is_off(llama_cpp, tmp_path, capsys,
                                                                 monkeypatch):
    monkeypatch.setenv("UNSLOTH_GGUF_VERIFY", "0")
    path = tmp_path / "m.gguf"
    path.write_bytes(b"not a gguf")
    llama_cpp._verify_converted_gguf([str(path)], "bf16", print_output = True)
    assert "Reading the GGUF back" not in capsys.readouterr().out


def test_the_announcement_is_printed_once_for_a_split_set(llama_cpp, tmp_path, capsys, monkeypatch):
    """One announcement per export, not one per shard."""
    paths = []
    for i in (1, 2, 3):
        p = tmp_path / f"m-{i:05d}-of-00003.gguf"
        p.write_bytes(b"not a gguf")
        paths.append(str(p))
    monkeypatch.setattr(llama_cpp, "_gguf_open_shards", lambda f: [(paths[0], None)])
    llama_cpp._verify_converted_gguf(paths, "bf16", print_output = True)
    assert capsys.readouterr().out.count("Reading the GGUF back") == 1


# --- The shape comparison covers the whole model, not just the first shard ---

class _FakeParameter:
    def __init__(self, shape):
        self.shape = shape


class _FakeModel:
    """Just enough of a torch model for `_model_tensor_shapes`."""

    def __init__(self, shapes):
        self._shapes = shapes

    def named_parameters(self):
        return [(name, _FakeParameter(shape)) for name, shape in self._shapes.items()]


def _two_shard_readers(llama_cpp, tmp_path, second_shape):
    """A two shard set holding one block each, the second sized by the caller.

    Give a test that builds two sets its own directory for the second: these readers
    hold the files mapped, and Windows refuses to rewrite a mapped file.
    """
    tmp_path = pathlib.Path(tmp_path)
    tmp_path.mkdir(parents = True, exist_ok = True)
    shapes = {
        "model.layers.0.self_attn.q_proj.weight": (4, 4),
        "model.layers.1.self_attn.q_proj.weight": (4, 4),
    }
    first = write_gguf(
        tmp_path / "model-00001-of-00002.gguf", keys = UNIVERSAL,
        tensors = {"blk.0.attn_q.weight": np.ones((4, 4), dtype = np.float32)},
    )
    second = write_gguf(
        tmp_path / "model-00002-of-00002.gguf", keys = UNIVERSAL,
        tensors = {"blk.1.attn_q.weight": np.ones(second_shape, dtype = np.float32)},
    )
    readers = [
        (first, llama_cpp._open_gguf_reader(first)),
        (second, llama_cpp._open_gguf_reader(second)),
    ]
    return _FakeModel(shapes), readers


def test_a_wrong_shape_past_the_first_shard_is_reported(llama_cpp, tmp_path):
    """The shape comparison was handed `readers[0][1]` alone, so a wrong shape past
    the first shard was never compared with the model."""
    model, readers = _two_shard_readers(llama_cpp, tmp_path, (2, 2))
    problems = llama_cpp._gguf_shape_problems(model, readers, sample_size = 8)
    assert any("blk.1.attn_q.weight" in problem for problem in problems), problems
    assert any("(2, 2)" in problem for problem in problems), problems


def test_a_correct_split_export_reports_nothing(llama_cpp, tmp_path):
    """Walking every shard must not start reporting problems on a good export."""
    model, readers = _two_shard_readers(llama_cpp, tmp_path, (4, 4))
    assert llama_cpp._gguf_shape_problems(model, readers, sample_size = 8) == []


def test_an_unreadable_later_shard_does_not_break_the_shape_pass(llama_cpp, tmp_path):
    """An unreadable shard is already fatal through the metadata pass, so the shape
    pass skips it and still checks the shards it can read."""
    model, readers = _two_shard_readers(llama_cpp, tmp_path, (2, 2))
    readers = [readers[0], (readers[1][0], None)]
    assert llama_cpp._gguf_shape_problems(model, readers, sample_size = 8) == []

    # With only the unreadable one left, nothing to compare rather than an exception.
    # Its own directory: the readers above still hold the first set mapped.
    model, readers = _two_shard_readers(llama_cpp, tmp_path / "second", (2, 2))
    assert llama_cpp._gguf_shape_problems(
        model, [(readers[0][0], None)], sample_size = 8,
    ) == []


# --- The read-back uses the gguf that wrote the file, not whatever the parent has ---

def _fake_gguf_tree(tmp_path, name = "gguf-py"):
    """A directory laid out the way an importable gguf-py tree is."""
    package = tmp_path / name / "gguf"
    package.mkdir(parents = True)
    (package / "__init__.py").write_text("", encoding = "utf-8")
    return str(tmp_path / name), str(package / "__init__.py")


def test_the_tree_is_derived_from_the_probe_report(llama_cpp, tmp_path):
    tree, location = _fake_gguf_tree(tmp_path)
    assert llama_cpp._gguf_tree_of_location(location) == tree


@pytest.mark.parametrize("location", [None, "", 5, "/nowhere/gguf/__init__.py"])
def test_a_location_that_is_not_a_package_is_refused(llama_cpp, location):
    """Putting the wrong directory on the parent's sys.path is worse than not pinning."""
    assert llama_cpp._gguf_tree_of_location(location) is None


def test_a_location_outside_a_gguf_package_is_refused(llama_cpp, tmp_path):
    _tree, location = _fake_gguf_tree(tmp_path, name = "tree")
    moved = Path(location).parent.parent / "notgguf"
    moved.mkdir()
    (moved / "__init__.py").write_text("", encoding = "utf-8")
    assert llama_cpp._gguf_tree_of_location(str(moved / "__init__.py")) is None


def test_the_readback_always_uses_the_tree_that_wrote_the_file(llama_cpp, tmp_path):
    """No version comparison: which package wrote the bytes is the only thing known,
    and version ordering cannot establish compatibility."""
    tree, location = _fake_gguf_tree(tmp_path)
    import sys as _sys
    import types as _types
    for parent_version in ("0.9.0", "0.17.1", "0.99.0", None):
        _sys.modules["gguf"] = _types.SimpleNamespace(__version__ = parent_version)
        try:
            assert llama_cpp._gguf_readback_tree(
                {"location": location, "version": "0.17.1"},
            ) == tree, parent_version
        finally:
            _sys.modules.pop("gguf", None)

    # And with no version reported by the child either.
    _sys.modules["gguf"] = _types.SimpleNamespace(__version__ = "0.17.1")
    try:
        assert llama_cpp._gguf_readback_tree({"location": location}) == tree
    finally:
        _sys.modules.pop("gguf", None)


def test_a_report_with_no_usable_location_still_changes_nothing(llama_cpp, tmp_path):
    """A location that is not a `gguf` package directory is still refused."""
    assert llama_cpp._gguf_readback_tree(None) is None
    assert llama_cpp._gguf_readback_tree({}) is None
    assert llama_cpp._gguf_readback_tree({"location": "/nowhere/gguf/__init__.py"}) is None


def test_the_conversion_passes_the_derived_tree_to_the_verifier(llama_cpp):
    """The derivation is only worth anything if `convert_to_gguf` uses it. Checked
    through the AST so a refactor of the call site does not fail it."""
    import ast as _ast
    tree = _ast.parse(Path(llama_cpp.__file__).read_text())
    fn = next(n for n in _ast.walk(tree)
              if isinstance(n, _ast.FunctionDef) and n.name == "convert_to_gguf")

    # Names assigned from an expression that mentions _gguf_readback_tree.
    derived = set()
    for node in _ast.walk(fn):
        if isinstance(node, _ast.Assign) and any(
            isinstance(c, _ast.Name) and c.id == "_gguf_readback_tree"
            for c in _ast.walk(node.value)
        ):
            derived.update(t.id for t in node.targets if isinstance(t, _ast.Name))
    assert derived, "convert_to_gguf never derives a read-back tree"

    # Every verification call must be handed one of those names, or the expression.
    calls = [n for n in _ast.walk(fn) if isinstance(n, _ast.Call)
             and isinstance(n.func, _ast.Name)
             and n.func.id in ("_verify_converted_gguf", "_verify_run_outputs")]
    assert calls, "convert_to_gguf no longer verifies what it wrote"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "gguf_py_dir" in kw, "a verification call lost gguf_py_dir"
        value = kw["gguf_py_dir"]
        names = {c.id for c in _ast.walk(value) if isinstance(c, _ast.Name)}
        assert names & (derived | {"_gguf_readback_tree"}), (
            "gguf_py_dir is not the derived read-back tree"
        )



# --- A projector has no vocabulary, and that is not a defect ---

class _BareTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = None
    unk_token_id = None


def test_a_projector_gguf_is_not_rejected_for_having_no_vocabulary(llama_cpp, tmp_path):
    """An mmproj holds a vision encoder, not a vocabulary, so it legitimately carries
    no `tokenizer.ggml.tokens`."""
    path = write_gguf(
        tmp_path / "model.F16-mmproj.gguf", architecture = "clip",
        tensors = {"v.blk.0.attn_q.weight": None},
    )
    # No raise: this is the whole assertion.
    llama_cpp.assert_correct_gguf(path, _FakeModel({}), _BareTokenizer(), sample_size = 4)


def test_the_text_model_still_has_its_vocabulary_checked(llama_cpp, tmp_path):
    """A text GGUF with no vocabulary is still a defect."""
    path = write_gguf(
        tmp_path / "text.gguf", architecture = "llama", keys = UNIVERSAL,
        tensors = {"blk.0.attn_q.weight": None},
    )
    with pytest.raises(RuntimeError) as raised:
        llama_cpp.assert_correct_gguf(path, _FakeModel({}), _BareTokenizer(), sample_size = 4)
    assert "tokenizer" in str(raised.value).lower(), raised.value


def test_the_exemption_is_read_from_the_metadata_pass_constant(llama_cpp):
    """Spelled once: a second copy of "clip" would drift."""
    source = Path(llama_cpp.__file__).read_text()
    body = source[source.index("def _assert_correct_gguf("):source.index("def assert_correct_gguf(")]
    assert "GGUF_METADATA_EXEMPT_ARCHITECTURES" in body
    assert '"clip"' not in body, "the architecture name is re-spelled instead of imported"


def test_the_named_tree_takes_precedence_over_every_other_path_entry(llama_cpp, tmp_path):
    """The tree is NAMED, not searched for, so it must beat an ambient gguf-py
    checkout sitting at `sys.path[0]`."""
    tree, _location = _fake_gguf_tree(tmp_path)
    decoy = tmp_path / "decoy"
    (decoy / "gguf").mkdir(parents = True)
    (decoy / "gguf" / "__init__.py").write_text("", encoding = "utf-8")

    import sys as _sys

    original = list(_sys.path)
    _sys.path.insert(0, str(decoy))
    try:
        with llama_cpp.use_local_gguf(tree):
            assert _sys.path[0] == tree, _sys.path[:3]
    finally:
        _sys.path[:] = original


def test_two_threads_cannot_swap_gguf_trees_at_once(llama_cpp, tmp_path):
    """`sys.path` and `sys.modules` are process-global, so the tree swap has to be
    serialised or two conversions interleave."""
    import threading

    first, _location = _fake_gguf_tree(tmp_path / "one")
    second, _location = _fake_gguf_tree(tmp_path / "two")

    inside = threading.Event()
    release = threading.Event()
    seen = []

    def hold():
        with llama_cpp.use_local_gguf(first):
            inside.set()
            release.wait(timeout = 10)
            seen.append(("holder", sys.path[0]))

    def overlap():
        inside.wait(timeout = 10)
        with llama_cpp.use_local_gguf(second):
            seen.append(("waiter", sys.path[0]))

    holder = threading.Thread(target = hold)
    waiter = threading.Thread(target = overlap)
    holder.start()
    waiter.start()
    assert inside.wait(timeout = 10), "the first swap never started"
    waiter.join(timeout = 0.5)
    assert waiter.is_alive(), "the second swap entered while the first still held the tree"
    release.set()
    holder.join(timeout = 10)
    waiter.join(timeout = 10)
    assert not holder.is_alive() and not waiter.is_alive()

    assert seen == [("holder", first), ("waiter", second)], seen


class _RecordingNameMap:
    """Stands in for gguf-py's TensorNameMap and records how it was sized."""

    calls: list = []

    def __init__(self, arch, n_blocks):
        type(self).calls.append(n_blocks)
        self.mapping = {}


def _many_parameter_model(llama_cpp, tmp_path, block_count_key = True):
    """Two blocks, and a great many separately named MoE expert parameters."""
    shapes = {
        "model.layers.0.self_attn.q_proj.weight": (4, 4),
        "model.layers.1.self_attn.q_proj.weight": (4, 4),
    }
    for block in (0, 1):
        for expert in range(2000):
            shapes[f"model.layers.{block}.mlp.experts.{expert}.down_proj.weight"] = (4, 4)
    keys = dict(UNIVERSAL)
    if not block_count_key:
        keys.pop("llama.block_count")
    path = write_gguf(
        tmp_path / "model.gguf", keys = keys,
        tensors = {"blk.0.attn_q.weight": np.ones((4, 4), dtype = np.float32)},
    )
    return _FakeModel(shapes), [(path, llama_cpp._open_gguf_reader(path))]


@pytest.mark.parametrize("block_count_key", [True, False])
def test_the_name_map_is_sized_by_the_blocks_not_the_parameters(
    llama_cpp, tmp_path, monkeypatch, block_count_key,
):
    """`TensorNameMap`'s second argument is the BLOCK count, and sizing it by the
    parameter count builds names for thousands of layers that do not exist. The file's
    declared block count is used, with the parameter names as the fallback."""
    import gguf.tensor_mapping as tensor_mapping

    _RecordingNameMap.calls = []
    monkeypatch.setattr(tensor_mapping, "TensorNameMap", _RecordingNameMap)
    model, readers = _many_parameter_model(llama_cpp, tmp_path, block_count_key)
    assert llama_cpp._gguf_shape_problems(model, readers, sample_size = 8) == []
    assert _RecordingNameMap.calls == [2], _RecordingNameMap.calls


def test_a_corrupt_shard_is_reported_for_a_projector_too(llama_cpp, tmp_path):
    """The architecture exemption is about which metadata KEYS a loader reads: a
    projector with a corrupt shard is still unloadable."""
    first = write_gguf(tmp_path / "p-00001-of-00002.gguf", architecture = "clip", keys = {},
                       tensors = {"blk.0.attn_q.weight": None, "mm.0.weight": None})
    (tmp_path / "p-00002-of-00002.gguf").write_bytes(b"not a gguf at all")

    problems = llama_cpp.gguf_metadata_problems(first)
    assert any("p-00002-of-00002.gguf" in problem for problem in problems), problems
    assert any("could not be read" in problem for problem in problems), problems

    # An intact projector is still exempt.
    ok_first = write_gguf(tmp_path / "q-00001-of-00002.gguf", architecture = "clip", keys = {},
                          tensors = {"blk.0.attn_q.weight": None, "mm.0.weight": None})
    write_gguf(tmp_path / "q-00002-of-00002.gguf", architecture = "clip", keys = {},
               tensors = {"mm.1.weight": None})
    assert llama_cpp.gguf_metadata_problems(ok_first) == []


# --- A failed OPTIONAL run is a text-only downgrade, not an aborted export ---

def test_a_required_run_that_fails_verification_raises(llama_cpp, tmp_path):
    keys = {k: v for k, v in UNIVERSAL.items() if k != "llama.block_count"}
    path = write_gguf(tmp_path / "text.gguf", keys = keys,
                      tensors = {"blk.0.attn_q.weight": None})
    with pytest.raises(RuntimeError, match = "did not pass post conversion verification"):
        llama_cpp._verify_run_outputs(
            [path], "text model", True, "bf16", print_output = False,
        )
    assert os.path.exists(path), "a required run's output is left for the caller to inspect"


def test_a_failed_optional_run_is_dropped_and_the_export_continues(llama_cpp, tmp_path, capsys):
    """The projector is an OPTIONAL run, and aborting would throw away a text model
    that is present and valid."""
    keys = {k: v for k, v in UNIVERSAL.items() if k != "llama.block_count"}
    path = write_gguf(tmp_path / "mmproj.gguf", keys = keys,
                      tensors = {"blk.0.attn_q.weight": None})
    kept = llama_cpp._verify_run_outputs(
        [path], "vision projector", False, "bf16", print_output = False,
    )
    assert kept is False
    assert not os.path.exists(path), "the bad projector must be removed so nothing uploads it"
    out = capsys.readouterr().out
    assert "vision projector" in out
    assert "text model was converted" in out


def test_a_clean_optional_run_is_kept(llama_cpp, tmp_path):
    path = write_gguf(tmp_path / "mmproj.gguf", architecture = "clip",
                      keys = {}, tensors = {"mm.0.weight": None})
    assert llama_cpp._verify_run_outputs(
        [path], "vision projector", False, "bf16", print_output = False,
    ) is True
    assert os.path.exists(path)


def test_the_conversion_drops_only_the_failed_optional_runs_files(llama_cpp):
    """The loop removes only that run's files and flips is_vlm, rather than aborting.
    Pinned through the AST."""
    import ast as _ast
    tree = _ast.parse(Path(llama_cpp.__file__).read_text())
    fn = next(n for n in _ast.walk(tree)
              if isinstance(n, _ast.FunctionDef) and n.name == "convert_to_gguf")
    calls = [n for n in _ast.walk(fn) if isinstance(n, _ast.Call)
             and isinstance(n.func, _ast.Name) and n.func.id == "_verify_run_outputs"]
    assert calls, "convert_to_gguf no longer verifies per run"
    # The required flag must be passed through, not hardcoded.
    for call in calls:
        args = [a for a in call.args if isinstance(a, _ast.Name)]
        assert any(a.id.endswith("required") for a in args), \
            "the run's required flag is not reaching the verifier"

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
import os
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


def test_a_missing_middle_shard_is_reported(llama_cpp, tmp_path):
    """Shard 1 present is not the same as the set being complete.

    The missing-shard check only looked at whether the FIRST shard was absent, because
    that is the one carrying the metadata. With shard 1 on disk every check runs happily
    on whatever survived and finds nothing wrong, so an incomplete split model, which
    llama.cpp cannot load, was reported ready to publish or quantize. The declared count
    lives only in the `-of-NNNNN` of the filename.
    """
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
    """Present but unreadable is exactly as unloadable as absent.

    Shard 1 is fine, so every metadata check passes and the declared count is satisfied; the
    corrupt shard was only logged, and the documented empty list is what a caller uses to
    ACCEPT a downloaded split model. That caller was being told the set was fine.
    """
    first = write_gguf(tmp_path / "b-00001-of-00002.gguf", keys = UNIVERSAL,
                       tensors = {"blk.0.attn_q.weight": None})
    corrupt = tmp_path / "b-00002-of-00002.gguf"
    corrupt.write_bytes(b"not a gguf at all")

    problems = llama_cpp.gguf_metadata_problems(first)
    assert any("b-00002-of-00002.gguf" in problem for problem in problems), problems
    assert any("could not be read" in problem for problem in problems), problems

    # The advisory pass stays quiet: this is a structural fault and the fatal pass has it, so
    # reporting it twice would read like two separate faults.
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
    """`convert_to_gguf` returns the shard LIST, and that list is what a caller hands back.

    Every member then resolved to the same complete set and revalidated it, so a 40 shard
    model was reopened and reparsed 40 times. Almost all of that cost is GGUFReader
    parsing shard 1's vocabulary, which is seconds on its own, so the redundancy is the
    whole wait rather than a rounding error. `_verify_converted_gguf` already dedupes.
    """
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


def test_a_zeroed_identity_projection_is_not_a_problem(llama_cpp, tmp_path):
    """LLaMA Pro block expansion (arXiv 2401.02415) zero-initialises o_proj and
    down_proj so a copied block is an exact identity, and a LoRA that does not
    target them leaves the merged weight at exactly zero. llama.cpp loads that
    file, so refusing it would break a real export."""
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
    """The negative control for the exemption above: it must not turn the
    all-zero check off for the tensors unsloth#6056's damage actually lands on."""
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


def test_an_unreadable_output_is_refused_when_its_own_writer_cannot_reopen_it(
    llama_cpp, tmp_path, monkeypatch
):
    """Version skew is the ONLY reason the failure above is a warning.

    When the converter child's own `gguf` tree is the one doing the reading, that reason is
    gone: a file its writer cannot reopen is malformed or truncated, and the checks before
    this one are existence and shard numbering only. A converter that exits zero after
    writing a broken GGUF would otherwise be published with nothing but a warning.
    """
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
    """An ImportError never reached the file's bytes, so it is not evidence the
    file is malformed. The reader runs in the PARENT, and pinning the writer's
    tree does not reproduce the child's PYTHONPATH or NO_LOCAL_GGUF, so a tree
    that imports in the child can fail in the parent over a dependency the child
    had. Refusing there rejects a healthy export and offers a re-run that cannot
    help."""
    import contextlib, logging
    path = tmp_path / "fine.gguf"
    # A real, readable GGUF: the only thing wrong is the reader.
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
    """A quantized block format holds bytes rather than values, so the tensor
    pass does not run. On a q4_k_m or MXFP4 export, which is what most people
    publish, that is every file, and reporting a bare "Verified" told the user a
    value check had happened when none had."""
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
    """A user holding a file this refuses needs to know which one it was, why it is not
    skew, and that the gate can be turned off if they want the artifact anyway."""
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
    """Only shard 1 carries the KV metadata, so without it every required key
    reads as missing. The report has to name the absent file instead."""
    second = write_gguf(tmp_path / "m-00002-of-00002.gguf", keys = {},
                        tensors = {"blk.1.attn_q.weight": None})
    problems = llama_cpp.gguf_metadata_problems(second)
    assert len(problems) == 1, problems
    assert "m-00001-of-00002.gguf" in problems[0], problems
    assert "not present" in problems[0], problems


def test_the_read_back_announces_itself_and_reports_its_cost(llama_cpp, tmp_path, capsys, monkeypatch):
    """The gate costs seconds on a large vocabulary, and it runs straight after
    "Successfully saved". A silent stall there reads like a hang, so it must say what it
    is doing, that the cost does not grow with the model, and how to turn it off."""
    path = tmp_path / "m.gguf"
    path.write_bytes(b"not a gguf")
    monkeypatch.setattr(llama_cpp, "_gguf_open_shards", lambda f: [(str(path), None)])

    llama_cpp._verify_converted_gguf([str(path)], "bf16", print_output = True)

    out = capsys.readouterr().out
    assert "Reading the GGUF back" in out
    assert "UNSLOTH_GGUF_VERIFY=0" in out, "the opt-out must be named where the cost is paid"
    assert "does not grow with the model" in out
    # Not "Verified 1": this file could not be read, so no metadata or tensor check ran on
    # it. Reporting it as verified told the user a gate had passed that never executed.
    assert "Verified" not in out, out
    assert "No GGUF file could be read back, so none was verified." in out


def test_a_file_that_was_really_checked_is_the_one_counted(llama_cpp, tmp_path, capsys):
    """The other half. A readable, correct export still reports its count and its cost, so
    the change is about honesty rather than about going quiet."""
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


# ---------------------------------------------------------------------------
# The shape comparison covers the whole model, not just the first shard
# ---------------------------------------------------------------------------

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
    """A two shard set holding one block each, the second sized by the caller."""
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
    """A split export puts most of its tensors in shards 1..N.

    Every other check in `assert_correct_gguf` is handed the whole shard set; the shape
    comparison was handed `readers[0][1]` alone, so a tensor whose dimensions were wrong
    anywhere past the first file was never compared with the model at all, and the export
    passed verification on the strength of checks that only prove the bytes are not
    degenerate and the metadata is present.
    """
    model, readers = _two_shard_readers(llama_cpp, tmp_path, (2, 2))
    problems = llama_cpp._gguf_shape_problems(model, readers, sample_size = 8)
    assert any("blk.1.attn_q.weight" in problem for problem in problems), problems
    assert any("(2, 2)" in problem for problem in problems), problems


def test_a_correct_split_export_reports_nothing(llama_cpp, tmp_path):
    """The other half, and the one that matters more: walking every shard must not start
    reporting problems on a good export."""
    model, readers = _two_shard_readers(llama_cpp, tmp_path, (4, 4))
    assert llama_cpp._gguf_shape_problems(model, readers, sample_size = 8) == []


def test_an_unreadable_later_shard_does_not_break_the_shape_pass(llama_cpp, tmp_path):
    """An unreadable shard is already fatal through the metadata pass, so this one skips it
    rather than saying it twice, and still checks the shards it can read."""
    model, readers = _two_shard_readers(llama_cpp, tmp_path, (2, 2))
    readers = [readers[0], (readers[1][0], None)]
    assert llama_cpp._gguf_shape_problems(model, readers, sample_size = 8) == []

    # And with only the unreadable one left there is nothing to compare against, rather
    # than an exception out of `reader.tensors` on None.
    model, readers = _two_shard_readers(llama_cpp, tmp_path, (2, 2))
    assert llama_cpp._gguf_shape_problems(
        model, [(readers[0][0], None)], sample_size = 8,
    ) == []


# ---------------------------------------------------------------------------
# The read-back uses the gguf that wrote the file, not whatever the parent has
# ---------------------------------------------------------------------------

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
    """No version comparison. Which package wrote the bytes is the only thing actually
    known, and it is what the file has to be read with.

    Version ordering cannot establish compatibility here: llama.cpp's vendored `gguf-py`
    and the PyPI wheel both report their own numbers, a fork and a release can report the
    SAME number with different contents, and a parent that omits `__version__` says nothing
    at all. Every one of those used to fall back to the parent's package, which is the
    degradation this helper exists to prevent.
    """
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

    # And with no version reported by the child either, since the child's report is only
    # ever consulted for `location` now.
    _sys.modules["gguf"] = _types.SimpleNamespace(__version__ = "0.17.1")
    try:
        assert llama_cpp._gguf_readback_tree({"location": location}) == tree
    finally:
        _sys.modules.pop("gguf", None)


def test_a_report_with_no_usable_location_still_changes_nothing(llama_cpp, tmp_path):
    """The one refusal that stays: a location that is not a `gguf` package directory would
    put the wrong tree on the parent's sys.path, which is worse than the warning."""
    assert llama_cpp._gguf_readback_tree(None) is None
    assert llama_cpp._gguf_readback_tree({}) is None
    assert llama_cpp._gguf_readback_tree({"location": "/nowhere/gguf/__init__.py"}) is None


def test_the_conversion_passes_the_derived_tree_to_the_verifier(llama_cpp):
    """The derivation is only worth anything if `convert_to_gguf` uses it. A call site still
    passing `_gguf_py_pin` alone would leave every case above passing while the common path
    read the file back with the parent's gguf exactly as before."""
    source = Path(llama_cpp.__file__).read_text()
    assert "gguf_py_dir = _gguf_py_pin or _gguf_readback_tree(_gguf_report)," in source


# ---------------------------------------------------------------------------
# A projector has no vocabulary, and that is not a defect
# ---------------------------------------------------------------------------

class _BareTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = None
    unk_token_id = None


def test_a_projector_gguf_is_not_rejected_for_having_no_vocabulary(llama_cpp, tmp_path):
    """A VLM conversion returns the text model AND the `clip` mmproj, and callers hand that
    whole list to `assert_correct_gguf`.

    An mmproj holds a vision encoder, not a vocabulary, so it legitimately carries no
    `tokenizer.ggml.tokens`. Running the tokenizer pass over it reported a missing
    vocabulary and rejected every otherwise valid multimodal conversion. The metadata pass
    already exempts `clip` for the same reason.
    """
    path = write_gguf(
        tmp_path / "model.F16-mmproj.gguf", architecture = "clip",
        tensors = {"v.blk.0.attn_q.weight": None},
    )
    # No raise: this is the whole assertion.
    llama_cpp.assert_correct_gguf(path, _FakeModel({}), _BareTokenizer(), sample_size = 4)


def test_the_text_model_still_has_its_vocabulary_checked(llama_cpp, tmp_path):
    """The exemption must not become a hole. A text GGUF with no vocabulary is still a
    defect, which is what the tokenizer pass is for."""
    path = write_gguf(
        tmp_path / "text.gguf", architecture = "llama", keys = UNIVERSAL,
        tensors = {"blk.0.attn_q.weight": None},
    )
    with pytest.raises(RuntimeError) as raised:
        llama_cpp.assert_correct_gguf(path, _FakeModel({}), _BareTokenizer(), sample_size = 4)
    assert "tokenizer" in str(raised.value).lower(), raised.value


def test_the_exemption_is_read_from_the_metadata_pass_constant(llama_cpp):
    """Spelled once. A second copy of "clip" would drift the moment another dummy
    architecture is added to the metadata exemption."""
    source = Path(llama_cpp.__file__).read_text()
    body = source[source.index("def _assert_correct_gguf("):source.index("def assert_correct_gguf(")]
    assert "GGUF_METADATA_EXEMPT_ARCHITECTURES" in body
    assert '"clip"' not in body, "the architecture name is re-spelled instead of imported"


def test_the_named_tree_takes_precedence_over_every_other_path_entry(llama_cpp, tmp_path):
    """`sys.path[0]` is the script or working directory, and a process launched from inside
    another gguf-py checkout has that checkout there.

    The tree handed to this context manager is NAMED, not searched for: it is the one the
    converter child reported, and reading the file back with the package that wrote it is
    the entire point. Inserting it after position 0 let the ambient checkout win the
    reimport, so the verification ran against a reader that may not understand what was
    written -- and then warned and skipped rather than failing loudly.
    """
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
    """`TensorNameMap`'s second argument is the BLOCK count and it expands every per-block
    template once per index.

    A model with separately named MoE experts has tens of thousands of parameters and a few
    dozen blocks, so sizing the map by the parameter count builds names for thousands of
    layers that do not exist before a single tensor is compared. The file declares its own
    block count -- llama.cpp refuses to load without it -- and the parameter names are the
    fallback when it is somehow absent.
    """
    import gguf.tensor_mapping as tensor_mapping

    _RecordingNameMap.calls = []
    monkeypatch.setattr(tensor_mapping, "TensorNameMap", _RecordingNameMap)
    model, readers = _many_parameter_model(llama_cpp, tmp_path, block_count_key)
    assert llama_cpp._gguf_shape_problems(model, readers, sample_size = 8) == []
    assert _RecordingNameMap.calls == [2], _RecordingNameMap.calls


def test_a_corrupt_shard_is_reported_for_a_projector_too(llama_cpp, tmp_path):
    """The architecture exemption is about which metadata KEYS a loader reads.

    A projector whose second shard is corrupt is as unloadable as any other model with a
    corrupt shard, and returning the empty list on the exemption first told a caller of the
    exported API that an unloadable projector was fine.
    """
    first = write_gguf(tmp_path / "p-00001-of-00002.gguf", architecture = "clip", keys = {},
                       tensors = {"blk.0.attn_q.weight": None, "mm.0.weight": None})
    (tmp_path / "p-00002-of-00002.gguf").write_bytes(b"not a gguf at all")

    problems = llama_cpp.gguf_metadata_problems(first)
    assert any("p-00002-of-00002.gguf" in problem for problem in problems), problems
    assert any("could not be read" in problem for problem in problems), problems

    # An intact projector is still exempt, which is the whole reason the exemption exists.
    ok_first = write_gguf(tmp_path / "q-00001-of-00002.gguf", architecture = "clip", keys = {},
                          tensors = {"blk.0.attn_q.weight": None, "mm.0.weight": None})
    write_gguf(tmp_path / "q-00002-of-00002.gguf", architecture = "clip", keys = {},
               tensors = {"mm.1.weight": None})
    assert llama_cpp.gguf_metadata_problems(ok_first) == []

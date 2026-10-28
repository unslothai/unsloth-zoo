"""The in-place 16-bit merge must not read or rewrite tensors nothing replaces.

An untouched tensor (no LoRA, no trained bias, no live vocab rows) already holds its
bytes in the shard, so reading it and writing the same bytes back is pure I/O. The
merged shards must stay byte-identical: run_case asserts every pass-through tensor
equals the base tensor exactly and every adapted one matches base + scale * B @ A.
"""
from __future__ import annotations

import pytest

import _merge_e2e_helpers as H


@pytest.mark.parametrize("scenario", ["attn_only", "mlp_only"])
def test_merge_writes_only_adapted_tensors(scenario, tmp_path, monkeypatch):
    if not H.family_available("llama"):
        pytest.skip("llama unavailable in this transformers")
    import unsloth_zoo.saving_utils as sv

    written = []
    real_write = sv._write_tensor_direct_torch

    def write(mm, header_metadata, length_of_header, output_key, tensor, output_dtype):
        written.append(output_key)
        return real_write(mm, header_metadata, length_of_header, output_key, tensor, output_dtype)

    monkeypatch.setattr(sv, "_write_tensor_direct_torch", write)
    n_adapted, n_pass = H.run_case("llama", scenario, str(tmp_path))
    assert n_adapted >= 1 and n_pass >= 1

    marker = ".self_attn." if scenario == "attn_only" else ".mlp."
    # Vocab layers carry the live embedding rows (new tokens), so they are rewritten.
    vocab = ("embed_tokens.weight", "lm_head.weight")
    untouched = sorted(k for k in written if marker not in k and not k.endswith(vocab))
    assert written, "the merge wrote nothing"
    assert untouched == [], f"untouched tensors were rewritten: {untouched}"

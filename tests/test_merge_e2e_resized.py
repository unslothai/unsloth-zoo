# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""End-to-end LoRA merge correctness for the vocab-grow / resized rewrite path.

Growing the tokenizer makes embed_tokens / lm_head larger than the base shard, so
the merge falls back to the resized-shard rewrite (streaming, or the disk-aware
in-place variant). Checks the rewritten embed/lm_head match the resized weights, old
rows are preserved, attention LoRA still merges, everything else is byte-identical.
"""

from __future__ import annotations

import os

import pytest
import torch

import _merge_e2e_helpers as H

NEW_VOCAB = 80  # base vocab is 64 (H._VOCAB)


def _build_resized_case(tmp_path, *, dtype=torch.float32, force_inplace=False,
                        optin=True, monkeypatch=None, calls=None):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    H.set_offline_cpu_env()
    if not H.family_available("llama"):
        pytest.skip("llama unavailable")
    spec = H.make_spec("llama")
    base_dir = os.path.join(str(tmp_path), "base")
    out_dir = os.path.join(str(tmp_path), "merged")

    torch.manual_seed(H.SEED)
    model = AutoModelForCausalLM.from_config(spec.config).to(dtype)
    model.save_pretrained(base_dir, safe_serialization=True)
    model.config._name_or_path = base_dir
    base = H.read_safetensors_dir(base_dir)

    model.resize_token_embeddings(NEW_VOCAB)
    torch.manual_seed(H.SEED)
    pm = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["embed_tokens", "lm_head"]))
    H.seed_lora(pm)

    ref_embed = pm.get_input_embeddings().weight.detach().cpu().clone().to(dtype)
    ref_head = pm.get_output_embeddings().weight.detach().cpu().clone().to(dtype)
    adapted = H.extract_adapted(pm)  # q_proj / v_proj only

    if force_inplace:
        # streaming-vs-in-place is chosen purely from free disk vs estimated shard
        # size; force in-place via an enormous estimate and spy on both paths.
        import unsloth_zoo.saving_utils as SU
        real_inplace = SU._inplace_rewrite_resized_shard
        real_stream = SU._stream_rewrite_resized_shard_and_replace

        def _spy_inplace(*a, **k):
            if calls is not None:
                calls["inplace"] = calls.get("inplace", 0) + 1
            return real_inplace(*a, **k)

        def _spy_stream(*a, **k):
            if calls is not None:
                calls["stream"] = calls.get("stream", 0) + 1
            return real_stream(*a, **k)

        monkeypatch.setattr(SU, "_estimate_resized_shard_bytes", lambda *a, **k: 1 << 60)
        monkeypatch.setattr(SU, "_inplace_rewrite_resized_shard", _spy_inplace)
        monkeypatch.setattr(SU, "_stream_rewrite_resized_shard_and_replace", _spy_stream)
        # in-place is non-atomic -> refused by default; opt in to exercise it.
        if optin:
            monkeypatch.setenv("UNSLOTH_ALLOW_NON_ATOMIC_RESIZED_REWRITE", "1")
        else:
            monkeypatch.delenv("UNSLOTH_ALLOW_NON_ATOMIC_RESIZED_REWRITE", raising=False)

    H.run_merge(pm, base_dir, out_dir, save_dtype=dtype)
    merged = H.read_safetensors_dir(out_dir)
    return base, merged, ref_embed, ref_head, adapted, dtype


def _check(base, merged, ref_embed, ref_head, adapted, dtype):
    embed_key = "model.embed_tokens.weight"
    head_key = "lm_head.weight"

    assert merged[embed_key].shape[0] == NEW_VOCAB
    assert merged[head_key].shape[0] == NEW_VOCAB
    # resized weights written through exactly
    assert torch.equal(merged[embed_key], ref_embed), "resized embed mismatch"
    assert torch.equal(merged[head_key], ref_head), "resized lm_head mismatch"
    # old vocab rows preserved from the base
    assert torch.equal(merged[embed_key][: base[embed_key].shape[0]], base[embed_key]), \
        "old embed rows not preserved"

    atol, rtol = H._TOL[dtype]
    for key, mt in merged.items():
        if key in (embed_key, head_key):
            continue
        if key in adapted:
            ref = H._ref_dense(base[key], adapted[key]).to(dtype)
            H._assert_close("resized", key, mt, ref, atol, rtol, adapted=True)
        else:
            H._assert_equal("resized", key, mt, base[key])


def test_resized_vocab_grow_modules_to_save(tmp_path):
    _check(*_build_resized_case(tmp_path))


def test_resized_vocab_grow_low_disk_inplace(tmp_path, monkeypatch):
    """In-place branch (opt-in) produces the same correct output as streaming.
    Forced via an enormous shard-size estimate; spies confirm which path ran."""
    calls = {"inplace": 0, "stream": 0}
    res = _build_resized_case(tmp_path, force_inplace=True, optin=True,
                              monkeypatch=monkeypatch, calls=calls)
    assert calls["inplace"] >= 1, "in-place resized rewrite branch was not exercised"
    assert calls["stream"] == 0, "streaming branch ran despite forced low disk"
    _check(*res)


def test_resized_vocab_grow_low_disk_fail_closed(tmp_path, monkeypatch):
    """Without the opt-in, a low-disk resized rewrite fails closed (raises, shard
    untouched) instead of the non-atomic in-place rewrite."""
    calls = {"inplace": 0, "stream": 0}
    with pytest.raises(RuntimeError, match="not enough free disk"):
        _build_resized_case(tmp_path, force_inplace=True, optin=False,
                            monkeypatch=monkeypatch, calls=calls)
    assert calls["inplace"] == 0, "non-atomic in-place rewrite ran despite fail-closed default"
    assert calls["stream"] == 0, "streaming branch ran despite forced low disk"

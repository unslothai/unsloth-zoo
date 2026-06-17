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
`_write_tensor_direct_torch` cannot overwrite in place and the merge falls back to
the resized-shard rewrite (streaming on PR #777, or the disk-aware in-place
variant). This test checks the rewritten embed/lm_head equal the model's resized
weights, old rows are preserved, attention LoRA still merges, and everything else
stays byte-identical.
"""

from __future__ import annotations

import os

import pytest
import torch

import _merge_e2e_helpers as H

NEW_VOCAB = 80  # base vocab is 64 (H._VOCAB)


def _build_resized_case(tmp_path, *, dtype=torch.float32, low_disk=False):
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

    H.run_merge(pm, base_dir, out_dir, save_dtype=dtype, low_disk_space_usage=low_disk)
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


def test_resized_vocab_grow_low_disk_fallback(tmp_path):
    """Same correctness when the disk-aware in-place fallback is requested."""
    _check(*_build_resized_case(tmp_path, low_disk=True))

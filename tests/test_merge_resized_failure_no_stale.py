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

"""A merge that fails after a resized (vocab grown) tensor was staged must not keep that tensor alive,
and must not leak it into the next merge of a different model in the same process."""

from __future__ import annotations

import gc
import os
import weakref

import pytest
import torch

import _merge_e2e_helpers as H
import unsloth_zoo.saving_utils as SU

NEW_VOCAB = 80  # base vocab is 64 (H._VOCAB)


def _peft(model):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(H.SEED)
    pm = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["embed_tokens", "lm_head"]))
    H.seed_lora(pm)
    return pm


def _base(tmp_path, name):
    from transformers import AutoModelForCausalLM
    H.set_offline_cpu_env()
    if not H.family_available("llama"):
        pytest.skip("llama unavailable")
    spec = H.make_spec("llama")
    base_dir = os.path.join(str(tmp_path), name)
    torch.manual_seed(H.SEED)
    model = AutoModelForCausalLM.from_config(spec.config).to(torch.float32)
    model.save_pretrained(base_dir, safe_serialization=True)
    model.config._name_or_path = base_dir
    return model, base_dir


def _staged():
    return dict(getattr(SU._merge_and_overwrite_lora, "_resized", {}) or {})


def _failed_resized_merge(tmp_path, monkeypatch):
    """Vocab-grown merge that dies after at least one resized tensor was staged."""
    model, base_dir = _base(tmp_path, "base_a")
    model.resize_token_embeddings(NEW_VOCAB)
    pm = _peft(model)

    real_write = SU._write_tensor_direct_torch
    seen = {"resized": 0, "refs": []}

    def _write(mm, header_metadata, length_of_header, output_key, tensor, output_dtype):
        ok = real_write(mm, header_metadata, length_of_header, output_key, tensor, output_dtype)
        if not ok:
            seen["resized"] += 1
            # float32 on CPU, so the staged copy IS this tensor (the trained lm_head / embed parameter).
            seen["refs"].append(weakref.ref(tensor))
            return ok
        if seen["resized"] and output_key.startswith("model.layers."):
            raise RuntimeError("injected merge failure")
        return ok

    monkeypatch.setattr(SU, "_write_tensor_direct_torch", _write)
    with pytest.raises(RuntimeError, match="injected merge failure"):
        H.run_merge(pm, base_dir, os.path.join(str(tmp_path), "merged_a"), save_dtype=torch.float32)
    monkeypatch.setattr(SU, "_write_tensor_direct_torch", real_write)
    assert seen["resized"] >= 1, "no resized tensor was staged before the injected failure (case not exercised)"
    return seen["refs"]



def test_failed_resized_merge_releases_staged_tensors(tmp_path, monkeypatch):
    # The model is gone once the helper returns: nothing but merge state may keep its staged tensors alive.
    refs = _failed_resized_merge(tmp_path, monkeypatch)
    gc.collect()
    alive = [r() for r in refs if r() is not None]
    assert not alive, (
        f"{len(alive)} resized tensor(s) from a failed merge outlive their model "
        f"(_merge_and_overwrite_lora._resized keys: {sorted(_staged())})"
    )


def test_merge_after_failed_resized_merge_is_not_stale(tmp_path, monkeypatch):
    _failed_resized_merge(tmp_path, monkeypatch)
    gc.collect()
    # A later merge of a different, NOT resized model must write its own lm_head / embed (64 rows).
    model_b, base_dir_b = _base(tmp_path, "base_b")
    pm_b = _peft(model_b)
    ref_head = pm_b.get_output_embeddings().weight.detach().cpu().clone()
    ref_embed = pm_b.get_input_embeddings().weight.detach().cpu().clone()
    out_b = os.path.join(str(tmp_path), "merged_b")
    H.run_merge(pm_b, base_dir_b, out_b, save_dtype=torch.float32)
    merged = H.read_safetensors_dir(out_b)
    assert merged["lm_head.weight"].shape[0] == ref_head.shape[0] != NEW_VOCAB, \
        f"second merge wrote a {merged['lm_head.weight'].shape[0]} row lm_head left over from the failed merge"
    assert torch.equal(merged["lm_head.weight"], ref_head), "lm_head of the second merge is stale"
    assert torch.equal(merged["model.embed_tokens.weight"], ref_embed), "embed of the second merge is stale"
    assert not _staged()

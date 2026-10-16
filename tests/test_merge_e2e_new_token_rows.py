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

"""LoRA on embed_tokens / lm_head after add_new_tokens: the merged rows past the base
checkpoint's vocab must be the live base rows plus the delta, not zeros plus the delta."""

from __future__ import annotations

import copy
import os

import pytest
import torch

import _merge_e2e_helpers as H

NEW_VOCAB = 72  # base vocab is 64 (H._VOCAB)


def _merged_vs_peft(tmp_path, target_modules):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    H.set_offline_cpu_env()
    if not H.family_available("llama"):
        pytest.skip("llama unavailable")
    base_dir = os.path.join(str(tmp_path), "base")
    out_dir = os.path.join(str(tmp_path), "merged")

    torch.manual_seed(H.SEED)
    model = AutoModelForCausalLM.from_config(H.make_spec("llama").config).to(torch.float32)
    model.save_pretrained(base_dir, safe_serialization = True)
    model.config._name_or_path = base_dir

    # add_new_tokens: grow, then seed the new rows from the old ones (offset so a zero row can't pass).
    model.resize_token_embeddings(NEW_VOCAB)
    with torch.no_grad():
        for w in (model.get_input_embeddings().weight, model.get_output_embeddings().weight):
            w[H._VOCAB:] = w[:H._VOCAB].mean(0) + 0.5

    torch.manual_seed(H.SEED)
    pm = get_peft_model(model, LoraConfig(
        r = 4, lora_alpha = 8, lora_dropout = 0.0, bias = "none", target_modules = target_modules))
    H.seed_lora(pm)
    with torch.no_grad():
        for name, p in pm.named_parameters():
            if "lora_embedding" in name:
                p.copy_(torch.randn_like(p) * 0.02 + 0.01)

    expected = {
        k: v.detach().clone() for k, v in copy.deepcopy(pm).merge_and_unload().state_dict().items()
    }
    H.run_merge(pm, base_dir, out_dir, save_dtype = torch.float32)
    return H.read_safetensors_dir(out_dir), expected


@pytest.mark.parametrize("target_modules, keys", [
    (["embed_tokens", "lm_head", "q_proj"], ("model.embed_tokens.weight", "lm_head.weight")),
    (["lm_head"], ("lm_head.weight",)),
    (["embed_tokens"], ("model.embed_tokens.weight",)),
])
def test_new_token_rows_keep_their_base_values(tmp_path, target_modules, keys):
    merged, expected = _merged_vs_peft(tmp_path, target_modules)
    for key in keys:
        assert merged[key].shape[0] == NEW_VOCAB
        new_rows, want = merged[key][H._VOCAB:], expected[key][H._VOCAB:]
        assert torch.allclose(new_rows, want, atol = 1e-5, rtol = 1e-5), (
            f"{key} new rows: max diff {(new_rows - want).abs().max().item():.4f}"
        )
        assert torch.allclose(merged[key][:H._VOCAB], expected[key][:H._VOCAB], atol = 1e-5, rtol = 1e-5)


def test_shrunk_vocab_merges_onto_the_kept_rows():
    from unsloth_zoo.saving_utils import _merge_lora, LoraStats

    torch.manual_seed(0)
    W = torch.randn(8, 4)
    A, B = torch.randn(2, 4), torch.randn(6, 2)
    stats = LoraStats(None, A, B, 0.5)
    merged = _merge_lora(W.clone(), stats, "lm_head")
    assert torch.allclose(merged.cpu(), W[:6] + 0.5 * (B @ A), atol = 1e-6)

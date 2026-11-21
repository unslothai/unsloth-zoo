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

"""CPU merged save streamed from the live model: same checkpoint as merge_and_unload()."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


_TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def _tiny_model():
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(_TINY, dtype = torch.float32, attn_implementation = "sdpa")
    except Exception as error:  # offline without the tiny model cached
        pytest.skip(f"{_TINY} unavailable: {error}")


def _read_checkpoint(directory):
    import glob
    import json
    from safetensors.torch import load_file
    tensors = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.safetensors"))):
        tensors.update(load_file(path))
    with open(os.path.join(directory, "config.json")) as f:
        config = json.load(f)
    config.pop("transformers_version", None)
    return tensors, config


def _lora_model(name, dtype, frozen_bf16 = False):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from unsloth_zoo.cpu.ops import store_frozen_linears_in_bf16
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype = torch.bfloat16 if frozen_bf16 else dtype)
    except Exception as error:
        pytest.skip(f"{name} unavailable: {error}")
    if frozen_bf16:
        store_frozen_linears_in_bf16(model)
    torch.manual_seed(0)
    config = LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj", "v_proj", "up_proj", "down_proj"],
                        init_lora_weights = False)
    return get_peft_model(model, config)


@pytest.mark.parametrize("case", ["bf16", "fp32", "fp32_frozen_bf16", "full_ft_fp32", "sharded"])
@pytest.mark.parametrize("save_method", ["merged_16bit", "merged_32bit"])
def test_streamed_merge_matches_the_copying_path(tmp_path, monkeypatch, case, save_method):
    from unsloth_zoo.cpu import loader
    if case == "full_ft_fp32":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(_TINY, dtype = torch.float32)
    else:
        model = _lora_model(_TINY, torch.float32 if case.startswith("fp32") else torch.bfloat16,
                            frozen_bf16 = case == "fp32_frozen_bf16")
    shard = "1MB" if case == "sharded" else None
    before = {k: v.clone() for k, v in model.state_dict().items()}
    reference, streamed = tmp_path / "copy", tmp_path / "stream"
    merged, _ = loader._merged_copy(model, save_method)
    merged.save_pretrained(reference, **({"max_shard_size": shard} if shard else {}))
    del merged
    assert loader._stream_merged_save(model, str(streamed), save_method, shard) is True
    ref_tensors, ref_config = _read_checkpoint(reference)
    new_tensors, new_config = _read_checkpoint(streamed)
    assert set(ref_tensors) == set(new_tensors)
    for key, value in ref_tensors.items():
        assert new_tensors[key].dtype == value.dtype, key
        assert torch.equal(new_tensors[key], value), key
    assert new_config == ref_config
    if case == "sharded":
        assert os.path.exists(streamed / "model.safetensors.index.json")
    # The live model is untouched (weights, adapters, merge state).
    after = model.state_dict()
    assert set(after) == set(before) and all(torch.equal(after[k], before[k]) for k in before)


def test_streamed_merge_refuses_what_it_cannot_reproduce(tmp_path):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    from unsloth_zoo.cpu import loader
    model = AutoModelForCausalLM.from_pretrained(_TINY, dtype = torch.float32)
    lora = get_peft_model(model, LoraConfig(r = 4, target_modules = ["q_proj"], modules_to_save = ["lm_head"]))
    assert loader._stream_merged_save(lora, str(tmp_path / "x"), "merged_16bit") is False


def test_streamed_merge_tied_embeddings_match(tmp_path):
    """Tied lm_head / embed_tokens: one name on disk, as save_pretrained writes it."""
    import transformers
    from unsloth_zoo.cpu import loader
    config = transformers.LlamaConfig(hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2,
                                      num_attention_heads = 4, num_key_value_heads = 2, vocab_size = 256,
                                      tie_word_embeddings = True)
    torch.manual_seed(0)
    transformers.LlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(tmp_path / "base")
    model = _lora_model(str(tmp_path / "base"), torch.bfloat16)
    merged, _ = loader._merged_copy(model, "merged_16bit")
    merged.save_pretrained(tmp_path / "copy")
    assert loader._stream_merged_save(model, str(tmp_path / "stream"), "merged_16bit") is True
    ref, ref_config = _read_checkpoint(tmp_path / "copy")
    new, new_config = _read_checkpoint(tmp_path / "stream")
    assert set(ref) == set(new) and "lm_head.weight" not in new
    assert all(torch.equal(ref[k], new[k]) for k in ref) and ref_config == new_config

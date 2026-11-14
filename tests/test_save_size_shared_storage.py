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

import functools

import pytest
import torch
from huggingface_hub import split_state_dict_into_shards_factory

from unsloth_zoo.saving_utils import get_torch_storage_id_new, get_torch_storage_size_new


def _arena_views(n = 8, numel = 1024):
    # Block Swap layout: every offloaded weight is a view into one host arena.
    arena = torch.zeros(n * numel * 4, dtype = torch.uint8)
    return arena, {f"layers.{i}.weight": arena[i * numel * 4:(i + 1) * numel * 4].view(torch.float32) for i in range(n)}


def test_disk_estimate_counts_each_view_once():
    arena, sd = _arena_views()
    assert sum(get_torch_storage_size_new(v, 2, own_bytes = True) for v in sd.values()) == arena.nbytes
    t = torch.zeros(3, 5, dtype = torch.bfloat16)
    assert get_torch_storage_size_new(t, 2, own_bytes = True) == 30
    assert get_torch_storage_size_new(torch.nn.Parameter(t), 2, own_bytes = True) == 30


def test_sharder_sizes_a_shared_storage_by_what_torch_save_writes():
    # torch.save writes a view's whole storage, so the views stay grouped and the shard is sized by the arena.
    arena, sd = _arena_views()
    split = split_state_dict_into_shards_factory(
        sd,
        max_shard_size = 3 * 4096,
        filename_pattern = "model{suffix}.bin",
        get_storage_size = functools.partial(get_torch_storage_size_new, element_size = 2),
        get_storage_id = get_torch_storage_id_new,
    )
    assert split.metadata["total_size"] == arena.nbytes
    assert len(set(split.tensor_to_filename.values())) == 1


def test_prepare_saving_estimate_with_block_swap_host_views(tmp_path):
    peft = pytest.importorskip("peft")
    from transformers import LlamaConfig, LlamaForCausalLM
    from unsloth_zoo.saving_utils import prepare_saving

    cfg = LlamaConfig(vocab_size = 128, hidden_size = 64, intermediate_size = 128, num_hidden_layers = 4,
                      num_attention_heads = 4, num_key_value_heads = 2, tie_word_embeddings = False)
    model = peft.get_peft_model(LlamaForCausalLM(cfg), peft.LoraConfig(r = 4, target_modules = ["q_proj"]))
    layers = model.base_model.model.model.layers
    params = [p for l in layers for n, p in l.named_parameters() if "lora_" not in n and "base_layer" not in n]
    arena = torch.empty(sum(p.nbytes for p in params), dtype = torch.uint8)
    host, used = {}, 0
    for p in params:
        host[id(p)] = arena[used:used + p.nbytes].view(p.dtype).view(p.shape).copy_(p.data)
        used += p.nbytes

    def hook(module, sd, prefix, meta):
        for n, p in module.named_parameters(remove_duplicate = False):
            if id(p) in host and prefix + n in sd:
                sd[prefix + n] = host[id(p)]

    for l in layers:
        l._register_state_dict_hook(hook)
    out = prepare_saving(model, save_directory = str(tmp_path), output_dtype = torch.float32)
    assert out[8] == sum(p.nbytes for p in model.merge_and_unload().parameters())

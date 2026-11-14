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

import torch
def test_disk_check_does_not_refuse_a_tied_save_that_fits(tmp_path, monkeypatch):
    import shutil
    import transformers
    from peft import LoraConfig, get_peft_model
    from unsloth_zoo import saving_utils

    cfg = transformers.Qwen2Config(
        vocab_size = 8192, hidden_size = 64, intermediate_size = 128, num_hidden_layers = 1,
        num_attention_heads = 4, num_key_value_heads = 2, tie_word_embeddings = True,
    )
    model = transformers.Qwen2ForCausalLM(cfg).to(torch.bfloat16)
    model.tie_weights()
    model = get_peft_model(model, LoraConfig(r = 4, target_modules = ["q_proj"]))
    emb = model.base_model.model.model.embed_tokens.weight
    weights = sum(p.nbytes for p in model.base_model.model.state_dict().values()) - emb.nbytes
    free = int((weights + emb.nbytes // 2) / 0.95)  # room for the save, not for a second embedding
    monkeypatch.setattr(shutil, "disk_usage", lambda path: (2 * free, free, free))
    out = saving_utils.prepare_saving(model, str(tmp_path), output_dtype = torch.bfloat16)
    assert out[8] < free

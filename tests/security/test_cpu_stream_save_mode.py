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

import os, stat
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "unsloth_zoo" / "cpu" / "loader.py"


@pytest.mark.skipif(os.name != "posix", reason = "POSIX modes")
def test_streamed_merged_resave_keeps_restricted_mode(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    from peft import LoraConfig, get_peft_model
    from unsloth_zoo.cpu import loader
    assert Path(loader.__file__).resolve() == SRC  # this arm's own tree

    config = transformers.LlamaConfig(vocab_size = 64, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 1,
                                      num_attention_heads = 2, num_key_value_heads = 2, tie_word_embeddings = False)
    torch.manual_seed(0)
    model = get_peft_model(transformers.LlamaForCausalLM(config), LoraConfig(r = 4, target_modules = ["q_proj"]))
    out = tmp_path / "merged"
    assert loader._stream_merged_save(model, str(out), "merged_16bit")
    weights = out / "model.safetensors"
    os.chmod(weights, 0o600)  # the user restricted the private weights
    assert loader._stream_merged_save(model, str(out), "merged_16bit")
    assert stat.S_IMODE(os.stat(weights).st_mode) == 0o600

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

import os, types, pytest, torch
from unittest import mock
from transformers import AutoModelForCausalLM

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _loader():
    import unsloth_zoo.cpu.loader as L
    return L


def _model(L):
    m = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype = torch.float32)
    m.push_to_hub_merged = types.MethodType(L.push_to_hub_merged, m)
    m.save_pretrained_merged = types.MethodType(L.save_pretrained_merged, m)
    return m


def _fake_api(private_flag):
    api = mock.MagicMock()
    api.create_repo.return_value = types.SimpleNamespace(repo_id = "me/existing")
    api.repo_info.return_value = types.SimpleNamespace(private = private_flag)
    return api


def test_private_push_to_existing_public_repo_refused():
    L = _loader()
    m = _model(L)
    with mock.patch("huggingface_hub.HfApi", return_value = _fake_api(False)), \
         mock.patch.object(type(m), "push_to_hub") as push:
        with pytest.raises(RuntimeError, match = "already exists and is public"):
            m.push_to_hub_merged("me/existing", save_method = "lora", token = "x", private = True)
    push.assert_not_called()


def test_private_push_to_private_repo_and_public_push_unchanged():
    L = _loader()
    m = _model(L)
    with mock.patch("huggingface_hub.HfApi", return_value = _fake_api(True)), \
         mock.patch.object(type(m), "push_to_hub") as push:
        m.push_to_hub_merged("me/existing", save_method = "lora", token = "x", private = True)
        m.push_to_hub_merged("me/existing", save_method = "lora", token = "x", private = None)
    assert push.call_count == 2


def test_peft_lora_save_is_adapter_only(tmp_path):
    from peft import LoraConfig, get_peft_model
    L = _loader()
    p = get_peft_model(AutoModelForCausalLM.from_pretrained(TINY), LoraConfig(r = 4, target_modules = ["q_proj"]))
    L.FastCPUModel._attach(p)
    p.save_pretrained_merged(str(tmp_path), save_method = "lora")
    files = sorted(os.listdir(tmp_path))
    assert "adapter_model.safetensors" in files and "model.safetensors" not in files, files

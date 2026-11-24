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

"""A low disk push (what push_to_hub_merged runs) must upload the base repo's tokenizer.model.

Step 2 is the only folder upload in low disk mode, and tokenizer.model is copied from the HF
cache or downloaded after it. Hub writes are mocked; the tiny base repo is read from the Hub.
"""

import os
import shutil

import pytest
import torch

from unsloth_zoo import saving_utils
from unsloth_zoo.saving_utils import merge_and_overwrite_lora

SPM_REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"     # ships tokenizer.model
BPE_REPO = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"       # no tokenizer.model at all
TARGET = "fakeuser/merged"


@pytest.fixture(autouse = True)
def _cpu_and_mocked_hub_writes(monkeypatch, tmp_path):
    os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
    monkeypatch.setattr(saving_utils, "_active_merge_device", lambda: "cpu")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "NoGPU")
    from huggingface_hub import HfApi, ModelCard

    events = []

    def upload_folder(self, *args, folder_path = None, **kwargs):
        names = set()
        for root, _, files in os.walk(folder_path):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), folder_path)
                if not rel.startswith(".cache"): names.add(rel)
        events.append(("folder", names))

    def upload_file(self, *args, path_or_fileobj = None, path_in_repo = None, **kwargs):
        assert os.path.isfile(path_or_fileobj), path_or_fileobj
        events.append(("file", path_in_repo))

    class _Info:
        private = True

    monkeypatch.setattr(HfApi, "upload_folder", upload_folder)
    monkeypatch.setattr(HfApi, "upload_file", upload_file)
    monkeypatch.setattr(HfApi, "create_repo", lambda self, *a, **k: f"https://huggingface.co/{TARGET}")
    _real_repo_info = HfApi.repo_info
    monkeypatch.setattr(
        HfApi, "repo_info",
        lambda self, repo_id = None, *a, **k: _Info() if repo_id == TARGET else _real_repo_info(self, repo_id, *a, **k),
    )
    monkeypatch.setattr(ModelCard, "push_to_hub", lambda self, *a, **k: None)
    monkeypatch.chdir(tmp_path)
    return events


def _peft_model(repo):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype = torch.float32)
    model = get_peft_model(model, LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj", "v_proj"]))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name: param.normal_(0, 0.01)
    return model


def _writes_tokenizer_model(tokenizer, tmp_path):
    probe = os.path.join(str(tmp_path), "probe")
    tokenizer.save_pretrained(probe)
    present = os.path.exists(os.path.join(probe, "tokenizer.model"))
    shutil.rmtree(probe)
    return present


def _tokenizer_without_spm_file(tmp_path):
    """A tokenizer saved with tokenizer.json only, as transformers 5.x saves a LoRA directory."""
    from transformers import AutoTokenizer
    tok_dir = os.path.join(str(tmp_path), "lora_tokenizer")
    AutoTokenizer.from_pretrained(SPM_REPO).save_pretrained(tok_dir)
    if os.path.exists(os.path.join(tok_dir, "tokenizer.model")):
        os.remove(os.path.join(tok_dir, "tokenizer.model"))
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    assert not _writes_tokenizer_model(tokenizer, tmp_path)
    return tokenizer


def _tokenizer_writing(tmp_path, payload):
    """A Hub tokenizer whose save_pretrained writes tokenizer.model with `payload` bytes."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SPM_REPO)
    original = tokenizer.save_pretrained

    def save_pretrained(save_directory, *args, **kwargs):
        result = original(save_directory, *args, **kwargs)
        with open(os.path.join(save_directory, "tokenizer.model"), "wb") as f: f.write(payload)
        return result
    tokenizer.save_pretrained = save_pretrained
    assert _writes_tokenizer_model(tokenizer, tmp_path)
    return tokenizer


def _base_tokenizer_model_bytes():
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(SPM_REPO, "tokenizer.model"), "rb") as f: return f.read()


def _push(repo, tokenizer, low_disk_space_usage):
    merge_and_overwrite_lora(
        lambda *a, **k: repo, _peft_model(repo), tokenizer = tokenizer, save_directory = TARGET,
        push_to_hub = True, token = "hf_fake", save_method = "merged_16bit",
        low_disk_space_usage = low_disk_space_usage,
    )


def _tokenizer_model_uploads(events):
    """How many uploads carried tokenizer.model, by kind."""
    return (
        sum(1 for kind, x in events if kind == "folder" and "tokenizer.model" in x),
        sum(1 for kind, x in events if kind == "file" and x == "tokenizer.model"),
    )


def test_low_disk_push_uploads_base_tokenizer_model_once(_cpu_and_mocked_hub_writes, tmp_path):
    events = _cpu_and_mocked_hub_writes
    _push(SPM_REPO, _tokenizer_without_spm_file(tmp_path), low_disk_space_usage = True)
    assert _tokenizer_model_uploads(events) == (0, 1), events
    # The shards still go up one by one, after Step 2's folder upload.
    assert [kind for kind, _ in events].count("folder") == 1
    assert ("file", "model.safetensors") in events


def test_low_disk_push_does_not_reupload_tokenizer_model_already_sent(_cpu_and_mocked_hub_writes, tmp_path):
    events = _cpu_and_mocked_hub_writes
    _push(SPM_REPO, _tokenizer_writing(tmp_path, _base_tokenizer_model_bytes()), low_disk_space_usage = True)
    assert _tokenizer_model_uploads(events) == (1, 0), events


def test_low_disk_push_resends_tokenizer_model_replaced_after_step_2(_cpu_and_mocked_hub_writes, tmp_path):
    """Step 4 overwrites it with the base repo's copy: the Hub gets the bytes the local save ends with."""
    events = _cpu_and_mocked_hub_writes
    _push(SPM_REPO, _tokenizer_writing(tmp_path, b"not the base tokenizer.model"), low_disk_space_usage = True)
    assert _tokenizer_model_uploads(events) == (1, 1), events


def test_full_disk_push_unchanged(_cpu_and_mocked_hub_writes, tmp_path):
    """Outside low disk mode the final folder upload already carries it; no single file upload is added."""
    events = _cpu_and_mocked_hub_writes
    _push(SPM_REPO, _tokenizer_without_spm_file(tmp_path), low_disk_space_usage = False)
    assert _tokenizer_model_uploads(events) == (1, 0), events
    assert events[-1][0] == "folder" and "tokenizer.model" in events[-1][1]


def test_low_disk_push_without_sentencepiece_uploads_no_tokenizer_model(_cpu_and_mocked_hub_writes):
    from transformers import AutoTokenizer
    events = _cpu_and_mocked_hub_writes
    _push(BPE_REPO, AutoTokenizer.from_pretrained(BPE_REPO), low_disk_space_usage = True)
    assert _tokenizer_model_uploads(events) == (0, 0), events
    assert ("file", "model.safetensors") in events

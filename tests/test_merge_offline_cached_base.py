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

"""HF_HUB_OFFLINE with the base in the Hugging Face cache: a 16bit merge must work.

The resolver asked the Hub about any repo id, and offline that raised "could not reach the
Hub", so a merge whose base was already cached failed before reading a byte.
"""

from __future__ import annotations

import os
import shutil

import pytest
import torch

import _merge_e2e_helpers as H
from unsloth_zoo import saving_utils
from unsloth_zoo.saving_utils import merge_and_overwrite_lora

FAMILY = "llama"
REPO = "ns/tiny-cached-base"
REVISION = "0" * 40


@pytest.fixture(autouse = True)
def _merge_on_the_cpu(monkeypatch):
    monkeypatch.setattr(saving_utils, "_active_merge_device", lambda: "cpu")


def _offline_cache(monkeypatch, tmp_path, base_dir, with_safetensors = True):
    """`base_dir` laid out as the cached `main` snapshot of REPO, with offline mode on."""
    from huggingface_hub import constants

    cache = tmp_path / "hub"
    repo_dir = cache / f"models--{REPO.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / REVISION
    shutil.copytree(base_dir, snapshot)
    if not with_safetensors:
        for f in snapshot.glob("*.safetensors"):
            f.unlink()
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text(REVISION)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))


def _base(tmp_path):
    if not H.family_available(FAMILY):
        pytest.skip(f"{FAMILY} unavailable in this transformers")
    H.set_offline_cpu_env()
    spec = H.make_spec(FAMILY)
    base_dir = os.path.join(str(tmp_path), "base")
    model = H.build_and_save_base(spec, base_dir)
    peft_model = H.attach_lora(model, spec, "full")
    H.seed_lora(peft_model)
    return peft_model, base_dir


def test_offline_resolves_a_cached_repo_to_its_snapshot(monkeypatch, tmp_path):
    _, base_dir = _base(tmp_path)
    _offline_cache(monkeypatch, tmp_path, base_dir)
    snapshot = str(tmp_path / "hub" / f"models--{REPO.replace('/', '--')}" / "snapshots" / REVISION)
    assert saving_utils.check_local_model_exists(REPO) == snapshot
    assert saving_utils.determine_base_model_source(REPO)[:3] == (snapshot, True, "local_unquantized")
    assert saving_utils.check_local_model_exists("ns/not-in-the-cache") is None


def test_online_never_reads_the_cache(monkeypatch, tmp_path):
    from huggingface_hub import constants
    _, base_dir = _base(tmp_path)
    _offline_cache(monkeypatch, tmp_path, base_dir)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    assert saving_utils.check_local_model_exists(REPO) is None


def test_a_cached_snapshot_without_safetensors_is_not_a_local_base(monkeypatch, tmp_path):
    _, base_dir = _base(tmp_path)
    _offline_cache(monkeypatch, tmp_path, base_dir, with_safetensors = False)
    assert saving_utils.check_local_model_exists(REPO) is None


def test_merged_16bit_from_a_cached_base_offline(monkeypatch, tmp_path):
    peft_model, base_dir = _base(tmp_path)
    adapted = H.extract_adapted(peft_model)
    _offline_cache(monkeypatch, tmp_path, base_dir)
    out_dir = os.path.join(str(tmp_path), "merged")
    merge_and_overwrite_lora(
        get_model_name = lambda *a, **k: REPO,
        model = peft_model,
        tokenizer = None,
        save_directory = out_dir,
        save_method = "merged_16bit",
        output_dtype = torch.float32,
        push_to_hub = False,
    )
    assert any(f.endswith(".safetensors") for f in os.listdir(out_dir))
    H.assert_merge_correct(
        family = FAMILY, base_tensors = H.read_safetensors_dir(base_dir), out_dir = out_dir,
        save_dtype = torch.float32, adapted = adapted,
    )

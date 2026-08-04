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

"""`merge_and_overwrite_lora` end to end: what happens when the Hub is down.

The rest of this change is verified at the level of `check_hf_model_exists` and
`determine_base_model_source`, which cannot tell whether anything was written. So this
file drives the merge itself with a real PeftModel and asserts the two things a user
experiences: an exception is raised, and no output directory is left behind that could
be mistaken for a successful export.

Uses the tiny synthetic models from `_merge_e2e_helpers`, so this is sub-second. Its
`set_offline_cpu_env()` already forbids the network; the Hub entry points are
monkeypatched on top so a failure is deterministic.

The merge math is pinned to CPU. `set_offline_cpu_env()` sets `UNSLOTH_ALLOW_CPU`,
which permits CPU rather than requiring it, so `_active_merge_device()` would still
prefer any accelerator present. Only control flow is under test here, so leaving the
device to the host only adds a hardware dependency: macos-14's virtualized MPS reports
7.93 GiB available and then refuses a 256 byte allocation.
"""

from __future__ import annotations

import os

import pytest

import _merge_e2e_helpers as H
from unsloth_zoo import saving_utils
from unsloth_zoo.saving_utils import merge_and_overwrite_lora


FAMILY = "llama"


@pytest.fixture(autouse = True)
def _merge_on_the_cpu(monkeypatch):
    """Pin `_merge_lora`'s device, for the reason in the module docstring.
    `_active_merge_device` is `lru_cache`d, so replacing the module attribute is both
    what takes effect and what monkeypatch undoes cleanly."""
    monkeypatch.setattr(saving_utils, "_active_merge_device", lambda: "cpu")


def _skip_if_missing():
    if not H.family_available(FAMILY):
        pytest.skip(f"{FAMILY} unavailable in this transformers")


def _peft_model(tmp_path):
    """A real PeftModel over a tiny base, plus the base directory on disk."""
    H.set_offline_cpu_env()
    spec = H.make_spec(FAMILY)
    base_dir = os.path.join(str(tmp_path), "base")
    model = H.build_and_save_base(spec, base_dir)
    peft_model = H.attach_lora(model, spec, "full")
    H.seed_lora(peft_model)
    return peft_model, base_dir


def _hub_raises(monkeypatch, error):
    """Both Hub round trips fail: the existence probe and the config fetch."""
    import huggingface_hub

    def fake_ls(self, path, detail = True, **kwargs):
        raise error
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)

    def fake_download(*args, **kwargs):
        raise error
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download, raising = True)


def _looks_like_a_successful_export(out_dir):
    """A directory a user or a downstream step would take for a finished model."""
    if not os.path.isdir(out_dir):
        return False
    return any(name.endswith(".safetensors") for name in os.listdir(out_dir))


# The regression this whole branch exists to remove.

def test_unreachable_hub_raises_and_writes_nothing(monkeypatch, tmp_path):
    """The bug, end to end: a base only on the Hub, a rate limited Hub and a 16bit
    merge. The old code warned, returned None and created no output directory."""
    _skip_if_missing()
    peft_model, _base_dir = _peft_model(tmp_path)
    out_dir = os.path.join(str(tmp_path), "merged")
    error = ConnectionError("Temporary failure in name resolution")
    _hub_raises(monkeypatch, error)

    with pytest.raises(RuntimeError) as excinfo:
        merge_and_overwrite_lora(
            get_model_name = lambda *a, **k: "ns/a-base-only-on-the-hub",
            model = peft_model,
            tokenizer = None,
            save_directory = out_dir,
            save_method = "merged_16bit",
            push_to_hub = False,
        )

    message = str(excinfo.value)
    assert "connectivity" in message or "rate limiting" in message, message
    assert not _looks_like_a_successful_export(out_dir), (
        "a partial output directory was left behind, which is exactly what makes "
        "the failure look like a success"
    )


def test_the_failure_is_not_reported_only_as_a_warning(monkeypatch, tmp_path):
    """This path must not merely warn and hand back None."""
    _skip_if_missing()
    peft_model, _base_dir = _peft_model(tmp_path)
    out_dir = os.path.join(str(tmp_path), "merged")
    _hub_raises(monkeypatch, ConnectionError("dns failure"))

    result = "<not set>"
    try:
        result = merge_and_overwrite_lora(
            get_model_name = lambda *a, **k: "ns/a-base-only-on-the-hub",
            model = peft_model,
            tokenizer = None,
            save_directory = out_dir,
            save_method = "merged_16bit",
            push_to_hub = False,
        )
    except RuntimeError:
        return  # raising is the contract
    pytest.fail(f"returned {result!r} for an unreachable Hub instead of raising")


def test_a_genuinely_absent_base_also_raises(monkeypatch, tmp_path):
    """Not just transport: a name that resolves nowhere also used to warn, return None
    and write nothing. The message now says what to check."""
    _skip_if_missing()
    peft_model, _base_dir = _peft_model(tmp_path)
    out_dir = os.path.join(str(tmp_path), "merged")

    def fake_ls(self, path, detail = True, **kwargs):
        raise FileNotFoundError(f"{path} (repository not found)")
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)

    with pytest.raises(RuntimeError) as excinfo:
        merge_and_overwrite_lora(
            get_model_name = lambda *a, **k: "ns/definitely-not-a-real-repo",
            model = peft_model,
            tokenizer = None,
            save_directory = out_dir,
            save_method = "merged_16bit",
            push_to_hub = False,
        )

    message = str(excinfo.value)
    assert "Nothing was written" in message, message
    assert "gated" in message, "the raise should name the cause a user can act on"
    assert not _looks_like_a_successful_export(out_dir)


def test_a_4bit_base_for_a_16bit_merge_raises_rather_than_warning(tmp_path):
    """The last silent no-op on this path, and it needed no Hub at all. `merged_16bit`
    off an nf4/fp4 base cannot be done, and the merge warned and returned None. The
    recovery it names, `forced_merged_4bit`, was already right."""
    _skip_if_missing()
    peft_model, base_dir = _peft_model(tmp_path)
    out_dir = os.path.join(str(tmp_path), "merged")

    # Make the local base look nf4 to `check_model_quantization_status`.
    import json
    config_path = os.path.join(base_dir, "config.json")
    with open(config_path, encoding = "utf-8") as f:
        config = json.load(f)
    config["quantization_config"] = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"}
    with open(config_path, "w", encoding = "utf-8") as f:
        json.dump(config, f)

    with pytest.raises(RuntimeError) as excinfo:
        merge_and_overwrite_lora(
            get_model_name = lambda *a, **k: base_dir,
            model = peft_model,
            tokenizer = None,
            save_directory = out_dir,
            save_method = "merged_16bit",
            push_to_hub = False,
        )

    message = str(excinfo.value)
    assert "Nothing was written" in message, message
    assert "forced_merged_4bit" in message, "the raise must still name the recovery"
    assert not _looks_like_a_successful_export(out_dir)


def test_an_hf_uri_is_still_addressed_as_a_repo(monkeypatch):
    """`hf://namespace/name` is documented HfFileSystem syntax that `ls` accepts, so its
    two slashes must not read as a filesystem path."""
    assert saving_utils._is_hub_repo_id("hf://openai-community/gpt2") is True
    assert saving_utils._is_hub_repo_id("openai-community/gpt2") is True
    # Still a path, scheme or no scheme.
    assert saving_utils._is_hub_repo_id("hf://a/b/c") is False
    assert saving_utils._is_hub_repo_id("/abs/base") is False


# The complement: an unreachable Hub must not break a merge that never needed it.

def test_a_local_base_still_merges_with_the_hub_down(monkeypatch, tmp_path):
    """The base is on disk, so nothing about this merge is the Hub's to decide: it must
    complete and write real weights with every Hub entry point failing."""
    _skip_if_missing()
    peft_model, base_dir = _peft_model(tmp_path)
    out_dir = os.path.join(str(tmp_path), "merged")
    _hub_raises(monkeypatch, ConnectionError("dns failure"))

    H.run_merge(peft_model, base_dir, out_dir, save_dtype = None)

    assert _looks_like_a_successful_export(out_dir), os.listdir(out_dir)
    tensors = H.read_safetensors_dir(out_dir)
    assert tensors, "the merge wrote no tensors"
    remnants = [k for k in tensors if ".lora_A" in k or ".lora_B" in k]
    assert remnants == [], f"adapter keys survived the merge: {remnants[:4]}"

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
`determine_base_model_source`. This file goes through the merge itself with a real
PeftModel, because the promise being made is about the merge:

    "Never return None here. `save_pretrained_merged` would look like it
     succeeded while creating no output directory at all, and the only signal
     would be a UserWarning that scrolls past in a notebook."

That promise had no behavioural coverage. It was pinned only by an
`inspect.getsource` string match and an AST walk, neither of which can tell
whether anything was actually written. So the two things asserted here are the two
things a user experiences: an exception is raised, and no output directory is left
behind that could be mistaken for a successful export.

Uses the tiny synthetic models from `_merge_e2e_helpers`, so this is CPU-only and
sub-second. `set_offline_cpu_env()` there already forbids the network; the Hub
entry points are monkeypatched on top so a failure is deterministic rather than
dependent on being genuinely offline.
"""

from __future__ import annotations

import os

import pytest

import _merge_e2e_helpers as H
from unsloth_zoo import saving_utils
from unsloth_zoo.saving_utils import merge_and_overwrite_lora


FAMILY = "llama"


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


# ---------------------------------------------------------------------------
# The regression this whole branch exists to remove.
# ---------------------------------------------------------------------------

def test_unreachable_hub_raises_and_writes_nothing(monkeypatch, tmp_path):
    """The bug, end to end. A base that is only on the Hub, a Hub that is rate
    limited, and a 16bit merge: the old code warned, returned None, and created no
    output directory, so the export looked like it had worked."""
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
    """A warning is the signal that scrolls past in a notebook. Whatever else it
    does, this path must not merely warn and hand back None."""
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
    """Not just transport. A name that resolves nowhere used to warn and return
    None too, writing nothing, and the message now says what to check."""
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
    """The last silent no-op on this path, and it needed no Hub at all.

    `merged_16bit` off an nf4/fp4 base cannot be done: the merge answered
    `warnings.warn` plus `return None` and wrote nothing, which is exactly the
    shape that cost a training run in the case this branch was opened for. The
    recovery it names, `forced_merged_4bit`, was already right; only the
    reporting was wrong.
    """
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
    """`hf://namespace/name` is the documented HfFileSystem URI form and `ls`
    accepts it. Its two slashes must not read as a filesystem path, or a name that
    worked before this change stops reaching the Hub at all."""
    assert saving_utils._is_hub_repo_id("hf://openai-community/gpt2") is True
    assert saving_utils._is_hub_repo_id("openai-community/gpt2") is True
    # Still a path, scheme or no scheme.
    assert saving_utils._is_hub_repo_id("hf://a/b/c") is False
    assert saving_utils._is_hub_repo_id("/abs/base") is False


# ---------------------------------------------------------------------------
# The complement: an unreachable Hub must not break a merge that never needed it.
# ---------------------------------------------------------------------------

def test_a_local_base_still_merges_with_the_hub_down(monkeypatch, tmp_path):
    """The other half of the contract, and the one a regression would silently
    take away. The base is on disk, so nothing about this merge is the Hub's to
    decide, and it must complete and write real weights with every Hub entry
    point failing."""
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

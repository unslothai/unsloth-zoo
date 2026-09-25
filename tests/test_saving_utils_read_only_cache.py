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

"""Export when base model files are read-only (huggingface_hub 1.x cache blobs are 0444)."""

from __future__ import annotations

import importlib.util
import os
import stat
import sys
import types

# CPU-only bitsandbytes stub; needs a real __spec__ so find_spec() probes do not raise.
if importlib.util.find_spec("bitsandbytes") is None:
    from importlib.machinery import ModuleSpec
    _bnb = types.ModuleType("bitsandbytes")
    _bnb.__spec__ = ModuleSpec("bitsandbytes", loader=None, is_package=True)
    _bnb.__path__ = []
    _bnb_nn = types.ModuleType("bitsandbytes.nn")
    _bnb_nn.__spec__ = ModuleSpec("bitsandbytes.nn", loader=None)
    for _cls in ("Linear8bitLt", "Linear4bit", "Int8Params", "Params4bit"):
        setattr(_bnb_nn, _cls, type(_cls, (object,), {}))
    _bnb.nn = _bnb_nn
    sys.modules["bitsandbytes"] = _bnb
    sys.modules["bitsandbytes.nn"] = _bnb_nn

from unsloth_zoo.saving_utils import (  # noqa: E402
    _copy_file_from_source,
    _ensure_shard_writable,
)


def _mode(p):
    return stat.S_IMODE(os.stat(p).st_mode)


def test_read_only_shard_becomes_owner_writable(tmp_path):
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"x" * 16)
    os.chmod(shard, 0o444)
    _ensure_shard_writable(str(shard))
    assert _mode(shard) == 0o644
    with open(shard, "r+b") as f:
        f.write(b"y")


def test_writable_shard_mode_untouched(tmp_path):
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"x")
    os.chmod(shard, 0o640)
    _ensure_shard_writable(str(shard))
    assert _mode(shard) == 0o640


def test_hard_linked_shard_gets_private_copy(tmp_path):
    blob = tmp_path / "blob"
    blob.write_bytes(b"cache")
    os.chmod(blob, 0o444)
    shard = tmp_path / "out" / "model.safetensors"
    shard.parent.mkdir()
    os.link(blob, shard)
    _ensure_shard_writable(str(shard))
    assert os.stat(shard).st_nlink == 1 and os.stat(blob).st_nlink == 1
    with open(shard, "r+b") as f:
        f.write(b"MERGE")
    assert blob.read_bytes() == b"cache" and _mode(blob) == 0o444
    assert shard.read_bytes() == b"MERGE"


def test_cache_copy_replaces_read_only_destination(tmp_path):
    src = tmp_path / "cache" / "tokenizer.model"
    src.parent.mkdir()
    src.write_bytes(b"new")
    os.chmod(src, 0o444)
    out = tmp_path / "out"
    out.mkdir()
    (out / "tokenizer.model").write_bytes(b"old")
    os.chmod(out / "tokenizer.model", 0o444)
    _copy_file_from_source(src, str(out), "tokenizer.model")
    assert (out / "tokenizer.model").read_bytes() == b"new"
    assert _mode(out / "tokenizer.model") == 0o644
    assert not [p for p in out.iterdir() if p.name.startswith(".unsloth-copy-")]
    assert _mode(src) == 0o444


def test_local_read_only_source_merges_over_read_only_tokenizer(tmp_path):
    import pytest
    if os.geteuid() == 0:
        pytest.skip("root ignores the read-only bit")
    import _merge_e2e_helpers as H
    H.set_offline_cpu_env()
    spec = H.make_spec("llama")
    base_dir, out_dir = str(tmp_path / "base"), str(tmp_path / "merged")
    model = H.build_and_save_base(spec, base_dir)
    base_tensors = H.read_safetensors_dir(base_dir)
    (tmp_path / "base" / "tokenizer.model").write_bytes(b"spm")
    for name in os.listdir(base_dir):
        os.chmod(os.path.join(base_dir, name), 0o444)
    os.makedirs(out_dir)
    (tmp_path / "merged" / "tokenizer.model").write_bytes(b"stale")
    os.chmod(out_dir + "/tokenizer.model", 0o444)
    peft_model = H.attach_lora(model, spec, "full")
    adapted = H.extract_adapted(peft_model)
    H.run_merge(peft_model, base_dir, out_dir, save_dtype = H.torch.float32)
    H.assert_merge_correct(
        family = "llama", base_tensors = base_tensors, out_dir = out_dir,
        save_dtype = H.torch.float32, adapted = adapted, base_dir = base_dir,
    )
    assert (tmp_path / "merged" / "tokenizer.model").read_bytes() == b"spm"

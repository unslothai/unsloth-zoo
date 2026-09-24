# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Export when the base model's files are read-only (huggingface_hub 1.x cache blobs are 0444).

Every copy into the output directory keeps the source mode, so the in-place LoRA merge
("r+b") failed with EACCES and the cache copy could not overwrite a read-only file already
there. CPU-only, no download.
"""

from __future__ import annotations

import importlib.util
import os
import stat
import sys
import types

# Stub bitsandbytes (imported at module scope) for CPU-only runs. Needs a real __spec__
# so find_spec() probes don't raise; built inline so package init can't pull the deps
# the stub avoids.
if importlib.util.find_spec("bitsandbytes") is None:
    from importlib.machinery import ModuleSpec
    _bnb = types.ModuleType("bitsandbytes")
    _bnb.__spec__ = ModuleSpec("bitsandbytes", loader=None, is_package=True)
    _bnb.__path__ = []
    _bnb_nn = types.ModuleType("bitsandbytes.nn")
    _bnb_nn.__spec__ = ModuleSpec("bitsandbytes.nn", loader=None)
    # Subclassable placeholders for older peft `class X(bnb.nn.Y)` import-time subclassing.
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
    with open(shard, "r+b") as f:  # the merge's in-place open
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
    assert blob.read_bytes() == b"cache" and _mode(blob) == 0o444  # cache untouched
    assert shard.read_bytes() == b"MERGE"


def test_cache_copy_replaces_read_only_destination(tmp_path):
    src = tmp_path / "cache" / "tokenizer.model"
    src.parent.mkdir()
    src.write_bytes(b"new")
    os.chmod(src, 0o444)
    out = tmp_path / "out"
    out.mkdir()
    (out / "tokenizer.model").write_bytes(b"old")
    os.chmod(out / "tokenizer.model", 0o444)  # an earlier step put a read-only copy here
    _copy_file_from_source(src, str(out), "tokenizer.model")
    assert (out / "tokenizer.model").read_bytes() == b"new"
    assert _mode(out / "tokenizer.model") == 0o644
    assert not [p for p in out.iterdir() if p.name.startswith(".unsloth-copy-")]
    assert _mode(src) == 0o444

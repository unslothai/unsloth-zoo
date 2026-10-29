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

"""The merge reads the base config at the Hub commit its shards are pinned to."""
import ast
import inspect
import json
import os
import textwrap

import torch


def _config_file(tmp_path, payload):
    path = os.path.join(str(tmp_path), "config.json")
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(payload, f)
    return path


def test_compressed_config_reads_follow_the_commit(monkeypatch, tmp_path):
    import huggingface_hub
    from unsloth_zoo import saving_utils as S
    path = _config_file(tmp_path, {"quantization_config": {
        "quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized",
        "config_groups": {"g": {"format": "mxfp4-pack-quantized"}},
    }})
    asked = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda **kw: asked.append(kw["revision"]) or path)
    commit = "c" * 40
    assert S._compressed_packed_format("org/repo", None, commit) == "mxfp4-pack-quantized"
    # An explicit revision in the name still wins over the pinned commit.
    assert S._compressed_packed_format("org/repo@dev", None, commit) == "mxfp4-pack-quantized"
    # No commit: unchanged, the default branch.
    assert S._compressed_packed_format("org/repo", None) == "mxfp4-pack-quantized"
    assert asked == [commit, "dev", None]


def test_base_config_read_follows_the_commit(monkeypatch):
    import transformers
    from unsloth_zoo import saving_utils as S
    calls = []

    def from_pretrained(name, **kwargs):
        calls.append(kwargs)
        return "config"
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", from_pretrained)
    model = torch.nn.Linear(1, 1)
    model.config = type("C", (), {"_name_or_path": "org/repo"})()
    commit = "d" * 40
    assert S._read_export_base_config("org/repo", None, model, commit = commit) == "config"
    assert calls[-1]["revision"] == commit and calls[-1]["trust_remote_code"] is False
    # Without a commit the call is exactly the old one.
    assert S._read_export_base_config("org/repo", None, model) == "config"
    assert "revision" not in calls[-1]


def test_the_merge_passes_the_pinned_commit_to_every_config_read():
    from unsloth_zoo import saving_utils as S
    source = textwrap.dedent(inspect.getsource(inspect.unwrap(S.merge_and_overwrite_lora)))
    readers = {
        "_compressed_packed_format", "_compressed_int_pack_schemes",
        "_read_export_base_config", "_source_config_dict",
    }
    seen = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in readers:
            names = {n.id for arg in [*node.args, *(k.value for k in node.keywords)]
                     for n in ast.walk(arg) if isinstance(n, ast.Name)}
            seen.setdefault(node.func.id, []).append("_config_commit" in names)
    assert set(seen) == readers, seen
    assert all(all(flags) for flags in seen.values()), seen

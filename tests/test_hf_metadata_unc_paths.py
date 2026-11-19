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

"""Repo metadata (model_index.json keys, modules.json paths) must never name a probed path outside the snapshot.

On Windows `snapshot_dir / r"\\\\host\\share"` is the UNC path itself, and probing it opens an SMB connection that
sends the user's NTLM hash. The join is POSIX here, so the test records which names get probed instead.
"""
import json
from pathlib import Path

from unsloth_zoo_downloads import hf_cache_state, hf_xet_fallback

UNC = "\\\\attacker.example\\share"


def _record_probes(monkeypatch):
    probed = []
    real_is_dir, real_iterdir = Path.is_dir, Path.iterdir

    def is_dir(self, *a, **k):
        probed.append(str(self))
        return real_is_dir(self, *a, **k)

    def iterdir(self):
        probed.append(str(self))
        return real_iterdir(self)

    monkeypatch.setattr(Path, "is_dir", is_dir)
    monkeypatch.setattr(Path, "iterdir", iterdir)
    return probed


def _snapshot(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"x")
    return snap


def test_model_index_unc_component_not_probed(tmp_path, monkeypatch):
    snap = _snapshot(tmp_path)
    (snap / "model_index.json").write_text(json.dumps(
        {"_class_name": "P", UNC: ["diffusers", "UNet"], "..\\..\\up": ["diffusers", "VAE"]}
    ))
    probed = _record_probes(monkeypatch)
    hf_xet_fallback._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None,
    )
    assert not [p for p in probed if "attacker.example" in p or "up" == Path(p).name], probed
    assert hf_cache_state._diffusers_declared_components(snap) is None


def test_modules_json_unc_path_not_probed(tmp_path, monkeypatch):
    snap = _snapshot(tmp_path)
    (snap / "modules.json").write_text(json.dumps(
        [{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
         {"idx": 1, "name": "1", "path": UNC, "type": "sentence_transformers.models.Dense"}]
    ))
    probed = _record_probes(monkeypatch)
    hf_cache_state._sentence_transformers_subfolder_incomplete(snap, strict = True)
    hf_cache_state._sentence_transformers_subfolder_incomplete(snap, strict = False)
    assert not [p for p in probed if "attacker.example" in p], probed


def test_normal_component_and_module_still_checked(tmp_path):
    snap = _snapshot(tmp_path)
    (snap / "model_index.json").write_text(json.dumps({"_class_name": "P", "unet": ["diffusers", "UNet"]}))
    assert hf_cache_state._diffusers_declared_components(snap) == {"unet"}
    (snap / "modules.json").write_text(json.dumps(
        [{"path": "2_Dense", "type": "sentence_transformers.models.Dense"}]
    ))
    # The declared Dense module has no weight on disk: still reported incomplete.
    assert hf_cache_state._sentence_transformers_subfolder_incomplete(snap, strict = False) is True

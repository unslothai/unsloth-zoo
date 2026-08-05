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

"""A base that exists but ships no safetensors must not be called missing.

`check_hf_model_exists` returns True only when the listing contains a
`.safetensors` file, so a public `.bin`-only repo such as `unsloth/bge-m3`
answers False exactly like a repo that is not there, and the caller blames
the name. Diagnosis only: which repos resolve is unchanged.
"""

import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unsloth_zoo.saving_utils as su  # noqa: E402


class _FakeFS:
    """Stands in for HfFileSystem: one listing, or an exception."""
    payload = None

    def __init__(self, token = None):
        pass

    def ls(self, name, detail = True):
        if isinstance(self.__class__.payload, Exception):
            raise self.__class__.payload
        return [{"name": f"{name}/{n}"} for n in self.__class__.payload]


@pytest.fixture
def fs(monkeypatch):
    def _set(payload):
        _FakeFS.payload = payload
        monkeypatch.setattr(su, "HfFileSystem", _FakeFS)
    return _set


def test_bin_only_repo_is_named(fs):
    fs(["config.json", "pytorch_model.bin", "tokenizer.json"])
    assert su._hub_repo_weights_without_safetensors("org/m") == ["pytorch_model.bin"]


def test_pt_and_bin_are_both_reported(fs):
    # bge-m3's actual layout.
    fs(["config.json", "colbert_linear.pt", "pytorch_model.bin", "sparse_linear.pt"])
    assert sorted(su._hub_repo_weights_without_safetensors("org/m")) == [
        "colbert_linear.pt", "pytorch_model.bin", "sparse_linear.pt"]


def test_safetensors_repo_says_nothing(fs):
    # This one resolves normally, so there is no second diagnosis to offer.
    fs(["config.json", "model.safetensors"])
    assert su._hub_repo_weights_without_safetensors("org/m") is None


def test_sharded_safetensors_say_nothing(fs):
    fs(["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
    assert su._hub_repo_weights_without_safetensors("org/m") is None


def test_repo_with_no_weights_at_all_says_nothing(fs):
    # Nothing useful to add; the original "could not be read" message stands.
    fs(["README.md", "config.json"])
    assert su._hub_repo_weights_without_safetensors("org/m") is None


def test_unreachable_hub_says_nothing(fs):
    # Absent, gated and unreachable are already described by the caller, and a
    # guess here would override a more accurate message.
    fs(ConnectionError("boom"))
    assert su._hub_repo_weights_without_safetensors("org/m") is None


def test_local_directory_is_not_probed(fs):
    fs(["pytorch_model.bin"])
    assert su._hub_repo_weights_without_safetensors("./outputs/mymodel") is None


def test_openvino_subdirectory_bins_do_not_count(fs):
    # all-MiniLM-L6-v2 ships openvino/*.bin beside a root model.safetensors.
    fs(["model.safetensors", "openvino/openvino_model.bin"])
    assert su._hub_repo_weights_without_safetensors("org/m") is None


def test_caller_raises_the_specific_message():
    """The bin-only branch must reach the user, not just exist.

    Assert on the RENDERED message: the literal is split across implicit
    concatenation, so no phrase a user would read is contiguous in the source.
    """
    import ast, re
    src = Path(su.__file__).read_text(encoding = "utf-8")
    i = src.index("_bin_only = _hub_repo_weights_without_safetensors")
    j = src.index("could not be read locally or on Hugging Face")
    assert i < j, "the bin-only check must run before the generic message"

    # Render every f-string in the module, then find the one this branch raises.
    rendered = [
        re.sub(r"\s+", " ", "".join(v.value for v in n.values
                                    if isinstance(v, ast.Constant)))
        for n in ast.walk(ast.parse(src)) if isinstance(n, ast.JoinedStr)
    ]
    msg = [m for m in rendered if "ships no safetensors weights" in m]
    assert msg, "the bin-only message is not raised anywhere"
    assert "Nothing was written to" in msg[0]
    assert "Convert the base to safetensors" in msg[0]


# An env var, not a pytest flag: `--live` was never registered through
# pytest_addoption, so `pytest --live` exits with "unrecognized arguments"
# before collection and this test could never actually be run.
@pytest.mark.skipif(
    os.environ.get("UNSLOTH_LIVE_TESTS", "0") != "1",
    reason = "needs network; set UNSLOTH_LIVE_TESTS=1",
)
def test_live_bge_m3():
    got = su._hub_repo_weights_without_safetensors("unsloth/bge-m3")
    assert got and "pytorch_model.bin" in got
    assert su._hub_repo_weights_without_safetensors(
        "sentence-transformers/all-MiniLM-L6-v2") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

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


"""merged_16bit keeps config.json's transformers_version, else transformers warns to set fix_mistral_regex (unsloth#4996)."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import types

import pytest

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

from unsloth_zoo.saving_utils import _remove_transformers_version  # noqa: E402


def _write(path, version):
    config = {"model_type": "llama"}
    if version is not None:
        config["transformers_version"] = version
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


@pytest.mark.parametrize("version", ["4.57.3", "4.57.3.dev0", "4.57.6", "5.0.0.dev0", "5.17.0"])
def test_version_without_the_4_57_2_crash_is_kept(tmp_path, version):
    path = _write(tmp_path / "config.json", version)
    _remove_transformers_version(path)
    assert json.loads(path.read_text())["transformers_version"] == version


@pytest.mark.parametrize("version", ["4.56.2", "4.57.1", "4.57.2", "4.57.2.dev0", "4.57.2rc1", "not-a-version"])
def test_version_that_crashes_transformers_4_57_2_is_dropped(tmp_path, version):
    path = _write(tmp_path / "config.json", version)
    _remove_transformers_version(path)
    assert "transformers_version" not in json.loads(path.read_text())


def test_reloaded_llama_tokenizer_gets_no_mistral_regex_warning(tmp_path):
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from packaging.version import Version
    if Version(transformers.__version__).release < (5,):
        pytest.skip("transformers 4.57.3-4.57.6 warn for every locally saved tokenizer")
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    # transformers only runs its Mistral regex check on vocabularies over 100k (Llama 3 has 128k).
    vocab = {f"t{i}": i for i in range(100_001)}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="t0"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    PreTrainedTokenizerFast(tokenizer_object=backend).save_pretrained(tmp_path)
    # What config.save_pretrained wrote before merge_and_overwrite_lora post-processes it.
    _remove_transformers_version(_write(tmp_path / "config.json", transformers.__version__))

    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger("transformers")
    logger.addHandler(handler)
    # Importing unsloth_zoo can lower transformers' verbosity, which would hide the warning.
    verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_warning()
    try:
        AutoTokenizer.from_pretrained(str(tmp_path))
    finally:
        transformers.logging.set_verbosity(verbosity)
        logger.removeHandler(handler)
    assert not any("incorrect regex pattern" in r.getMessage() for r in records)

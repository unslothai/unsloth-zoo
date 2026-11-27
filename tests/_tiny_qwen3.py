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

"""The tiny Qwen3 the CPU preference-loss tests train, as a complete local snapshot or a skip."""

import os

import pytest

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"
_TINY_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def tiny_snapshot() -> str:
    """With the hub unreachable a partly cached repo does not raise OSError: the snapshot resolves, the slow Qwen2
    tokenizer then opens vocab_file=None (TypeError), and the test fails instead of skipping. So check the files
    are there rather than trusting the load."""
    from huggingface_hub import snapshot_download
    try:
        path = snapshot_download(TINY)
    except Exception as error:  # noqa: BLE001 - offline, rate limited
        pytest.skip(f"{TINY} unavailable: {type(error).__name__}: {error}")
    missing = [f for f in _TINY_FILES if not os.path.isfile(os.path.join(path, f))]
    if missing:
        pytest.skip(f"{TINY} only partly cached and the hub is unreachable: missing {missing}")
    return path

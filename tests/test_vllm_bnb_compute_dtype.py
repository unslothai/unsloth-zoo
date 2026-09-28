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

"""vLLM >= 0.28 builds the vllm-bnb-plugin config with no kwargs for online
quantization, so the compute dtype override must not index a missing key."""

import pytest

from unsloth_zoo import vllm_utils
ENV = "UNSLOTH_bnb_4bit_compute_dtype"


def _resolve_bnb_compute_dtype(kwargs):
    return vllm_utils._resolve_bnb_compute_dtype(kwargs)


@pytest.mark.parametrize("bf16, expected", [(True, "bfloat16"), (False, "float16")])
def test_no_kwargs_falls_back_to_the_gpu_dtype(monkeypatch, bf16, expected):
    monkeypatch.delenv(ENV, raising = False)
    monkeypatch.setattr(vllm_utils, "device_is_bf16_supported", lambda: bf16)
    assert _resolve_bnb_compute_dtype({}) == expected


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv(ENV, "float16")
    monkeypatch.setattr(vllm_utils, "device_is_bf16_supported", lambda: True)
    assert _resolve_bnb_compute_dtype({}) == "float16"
    assert _resolve_bnb_compute_dtype({"bnb_4bit_compute_dtype": "float32"}) == "float16"


def test_checkpoint_dtype_kept_without_override(monkeypatch):
    monkeypatch.delenv(ENV, raising = False)
    monkeypatch.setattr(vllm_utils, "device_is_bf16_supported", lambda: True)
    assert _resolve_bnb_compute_dtype({"bnb_4bit_compute_dtype": "float32"}) == "float32"


def test_patched_config_builds_with_no_kwargs(monkeypatch):
    if getattr(vllm_utils, "_vllm_bnb", None) is None:
        pytest.skip("vLLM bitsandbytes (in tree or vllm-bnb-plugin) not installed")
    monkeypatch.delenv(ENV, raising = False)
    monkeypatch.setattr(vllm_utils, "device_is_bf16_supported", lambda: False, raising = False)
    config = vllm_utils.BitsAndBytesConfig()
    assert config.bnb_4bit_compute_dtype == "float16"

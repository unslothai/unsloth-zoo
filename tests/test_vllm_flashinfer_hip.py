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

from unittest import mock

from unsloth_zoo import vllm_utils


def test_hip_clears_forced_flashinfer_even_when_package_absent():
    # Regression: on ROCm with a forced FlashInfer backend inherited from the env
    # but the flashinfer package NOT installed, the vars must still be cleared so
    # vLLM does not try to use FlashInfer.
    env = {"VLLM_ATTENTION_BACKEND": "FLASHINFER", "VLLM_USE_FLASHINFER_SAMPLER": "1"}
    with mock.patch.object(vllm_utils, "is_hip", return_value=True), \
         mock.patch("importlib.util.find_spec", return_value=None), \
         mock.patch.dict(vllm_utils.os.environ, env, clear=False):
        handled = vllm_utils._clear_flashinfer_env_on_hip()
        assert handled is True
        assert "VLLM_ATTENTION_BACKEND" not in vllm_utils.os.environ
        assert "VLLM_USE_FLASHINFER_SAMPLER" not in vllm_utils.os.environ


def test_non_hip_leaves_env_untouched_and_returns_false():
    env = {"VLLM_ATTENTION_BACKEND": "FLASHINFER"}
    with mock.patch.object(vllm_utils, "is_hip", return_value=False), \
         mock.patch.dict(vllm_utils.os.environ, env, clear=False):
        handled = vllm_utils._clear_flashinfer_env_on_hip()
        assert handled is False
        assert vllm_utils.os.environ.get("VLLM_ATTENTION_BACKEND") == "FLASHINFER"


def test_hip_without_forced_env_is_a_noop_but_still_handled():
    with mock.patch.object(vllm_utils, "is_hip", return_value=True), \
         mock.patch("importlib.util.find_spec", return_value=None), \
         mock.patch.dict(vllm_utils.os.environ, {}, clear=False):
        vllm_utils.os.environ.pop("VLLM_ATTENTION_BACKEND", None)
        vllm_utils.os.environ.pop("VLLM_USE_FLASHINFER_SAMPLER", None)
        handled = vllm_utils._clear_flashinfer_env_on_hip()
        assert handled is True

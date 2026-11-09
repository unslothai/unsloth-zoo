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

"""CPU loader: GPU FastLanguageModel keywords never reach the model constructor."""

import pytest
import torch

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"

# The GPU from_pretrained defaults, as a caller forwarding every keyword passes them.
GPU_DEFAULTS = dict(
    device_map_planner_kwargs = None,
    rope_scaling = None,
    float32_mixed_precision = None,
    disable_log_stats = True,
    qat_scheme = None,
    load_in_fp8 = False,
    text_only = False,
    return_logits = False,
    fullgraph = True,
    auto_model = None,
    whisper_language = None,
    whisper_task = None,
    unsloth_force_compile = False,
    target_parameters = None,
    fp8_lora = None,
    user_config = None,
    float8_training = False,
)


def _loader():
    from unsloth_zoo.cpu import loader
    return loader


def test_gpu_defaults_load_on_cpu():
    model, _ = _loader().FastCPUModel.from_pretrained(
        TINY, full_finetuning = True, load_in_4bit = False, dtype = torch.float32, **GPU_DEFAULTS
    )
    assert type(model).__name__ == "Qwen3ForCausalLM"


@pytest.mark.parametrize(
    "key, value",
    [
        ("float8_training", True),
        ("float8_training", "tensorwise"),
        ("fp8_lora", "fp8_lora_tensorwise"),
        ("load_in_fp8", True),
        ("qat_scheme", "int4"),
        ("auto_model", object),
        ("target_parameters", ["mlp.experts.gate_up_proj"]),
    ],
)
def test_gpu_only_feature_is_refused(key, value):
    with pytest.raises(NotImplementedError, match = key):
        _loader().FastCPUModel.from_pretrained(TINY, full_finetuning = True, load_in_4bit = False, **{key: value})


def test_integer_zero_is_off_as_on_gpu():
    # The GPU loader gates with `not in (None, False)`, so 0 means off there too.
    model, _ = _loader().FastCPUModel.from_pretrained(
        TINY, full_finetuning = True, load_in_4bit = False, float8_training = 0, load_in_fp8 = 0
    )
    assert type(model).__name__ == "Qwen3ForCausalLM"


@pytest.mark.parametrize("return_logits", [False, True])
def test_return_logits_keeps_the_full_logit_loss(return_logits):
    model, _ = _loader().FastCPUModel.from_pretrained(
        TINY, full_finetuning = True, load_in_4bit = False, return_logits = return_logits
    )
    assert getattr(model, "_unsloth_cpu_chunked_ce", False) is (not return_logits)

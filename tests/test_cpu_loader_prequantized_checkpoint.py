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

"""CPU loader: a checkpoint quantized with another method (fp8, compressed-tensors) keeps its own config.

load_in_4bit = True is the default; passing a BitsAndBytesConfig on top of an FP8 release made transformers raise
"The model is quantized with FineGrainedFP8Config but you are passing a BitsAndBytesConfig"."""

import json
import os

import pytest

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"


class _Captured(Exception):
    pass


def _checkpoint(tmp_path, quantization_config):
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(TINY, "config.json"), encoding = "utf-8") as f:
        config = json.load(f)
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    os.makedirs(tmp_path, exist_ok = True)
    with open(os.path.join(tmp_path, "config.json"), "w", encoding = "utf-8") as f:
        json.dump(config, f)
    return str(tmp_path)


def _load_kwargs(monkeypatch, path):
    import transformers
    from unsloth_zoo.cpu import loader

    seen = {}

    def capture(name, *args, **kwargs):
        # Keyword names and the quantization config type only: the loader also forwards the hub token.
        seen.update({k: type(v).__name__ for k, v in kwargs.items()})
        raise _Captured

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", capture)
    with pytest.raises(_Captured):
        loader.FastCPUModel.from_pretrained(path, load_in_4bit = True)
    return seen


@pytest.mark.parametrize(
    "quant",
    [
        {"quant_method": "fp8", "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4"},
        {"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized"},
    ],
)
def test_a_checkpoint_quantized_another_way_gets_no_bitsandbytes_config(tmp_path, monkeypatch, quant):
    seen = _load_kwargs(monkeypatch, _checkpoint(tmp_path, quant))
    assert "quantization_config" not in seen


def test_a_plain_checkpoint_still_loads_in_4bit(tmp_path, monkeypatch):
    from unsloth_zoo.cpu.runtime import cpu_bnb_4bit_supported

    if not cpu_bnb_4bit_supported():
        pytest.skip("bitsandbytes 4-bit does not run on this CPU")
    seen = _load_kwargs(monkeypatch, _checkpoint(tmp_path, None))
    assert seen.get("quantization_config") == "BitsAndBytesConfig"


def test_quant_method_is_read_from_the_text_config(tmp_path):
    from unsloth_zoo.cpu.loader import _checkpoint_quant_method

    path = _checkpoint(tmp_path, None)
    with open(os.path.join(path, "config.json"), encoding = "utf-8") as f:
        config = json.load(f)
    config["text_config"] = {"quantization_config": {"quant_method": "FP8"}}
    with open(os.path.join(path, "config.json"), "w", encoding = "utf-8") as f:
        json.dump(config, f)
    assert _checkpoint_quant_method(path) == "fp8"

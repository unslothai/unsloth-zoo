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
"The model is quantized with FineGrainedFP8Config but you are passing a BitsAndBytesConfig". The decision reads the
config the model load itself uses: an explicit config=, the subfolder, and composite configs' language model."""

import json
import os

import pytest

FP8 = {"quant_method": "fp8", "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4"}
NVFP4 = {"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized"}

# A tiny, local Qwen3 config: no hub access needed.
_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "vocab_size": 128, "hidden_size": 16,
    "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
    "head_dim": 8, "max_position_embeddings": 64,
}


class _Captured(Exception):
    pass


def _write(path, quantization_config = None, **extra):
    config = dict(_CONFIG, **extra)
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    os.makedirs(path, exist_ok = True)
    with open(os.path.join(path, "config.json"), "w", encoding = "utf-8") as f:
        json.dump(config, f)
    return str(path)


def _load_kwargs(monkeypatch, path, bnb_cpu = True, **load):
    import transformers
    from unsloth_zoo.cpu import loader

    seen = {}
    printed = []

    def capture(name, *args, **kwargs):
        # Keyword names and the quantization config type only: the loader also forwards the hub token.
        seen.update({k: type(v).__name__ for k, v in kwargs.items()})
        raise _Captured

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", capture)
    # The decision is under test, not the host's bitsandbytes CPU backend.
    monkeypatch.setattr(loader, "cpu_bnb_4bit_supported", lambda: bnb_cpu)
    monkeypatch.setattr(loader, "_print", printed.append)
    with pytest.raises(_Captured):
        loader.FastCPUModel.from_pretrained(path, load_in_4bit = True, **load)
    return seen, printed


@pytest.mark.parametrize("quant", [FP8, NVFP4])
def test_a_checkpoint_quantized_another_way_gets_no_bitsandbytes_config(tmp_path, monkeypatch, quant):
    seen, printed = _load_kwargs(monkeypatch, _write(tmp_path, quant))
    assert "quantization_config" not in seen
    assert any(quant["quant_method"] in line for line in printed), printed


def test_a_plain_checkpoint_still_loads_in_4bit(tmp_path, monkeypatch):
    seen, _ = _load_kwargs(monkeypatch, _write(tmp_path))
    assert seen.get("quantization_config") == "BitsAndBytesConfig"


def test_the_subfolder_the_load_reads_decides(tmp_path, monkeypatch):
    # An FP8 root with a plain checkpoint in the subfolder still loads the subfolder in 4-bit, and the inverse
    # layout keeps the subfolder's own FP8 config.
    root = _write(tmp_path / "a", FP8)
    _write(tmp_path / "a" / "child")
    seen, _ = _load_kwargs(monkeypatch, root, subfolder = "child")
    assert seen.get("quantization_config") == "BitsAndBytesConfig"
    root = _write(tmp_path / "b")
    _write(tmp_path / "b" / "child", FP8)
    seen, _ = _load_kwargs(monkeypatch, root, subfolder = "child")
    assert "quantization_config" not in seen


def test_an_explicit_config_decides(tmp_path, monkeypatch):
    from transformers import AutoConfig

    path = _write(tmp_path / "fp8", FP8)
    plain = AutoConfig.from_pretrained(_write(tmp_path / "plain"))
    seen, _ = _load_kwargs(monkeypatch, path, config = plain)
    assert seen.get("quantization_config") == "BitsAndBytesConfig"


def test_the_hardware_fallback_names_the_checkpoints_own_quantization(tmp_path, monkeypatch):
    _, printed = _load_kwargs(monkeypatch, _write(tmp_path, FP8), bnb_cpu = False)
    assert any("fp8" in line for line in printed) and not any("does not work on this CPU" in line for line in printed)


@pytest.mark.parametrize("key", ["text_config", "language_config", "llm_config", "decoder", "generator"])
def test_quant_method_is_read_from_language_model_sub_configs(key):
    from unsloth_zoo.cpu.loader import _quant_method_of

    assert _quant_method_of({key: {"quantization_config": {"quant_method": "FP8"}}}) == "fp8"
    assert _quant_method_of({key: None}) is None
    assert _quant_method_of([]) is None


def test_reverse_fp8_conversions_are_dropped_after_load(monkeypatch):
    import sys
    import types

    from unsloth_zoo.cpu import loader

    calls = []
    fake = types.ModuleType("unsloth.models.loader_utils")
    fake._drop_fp8_dequantize_conversions = lambda model: calls.append(model) or 3
    monkeypatch.setitem(sys.modules, "unsloth.models.loader_utils", fake)
    model = object()
    assert loader._drop_reverse_fp8_conversions(model) == 3 and calls == [model]
    # Without Unsloth (or with a helper that fails) the load carries on.
    fake._drop_fp8_dequantize_conversions = lambda model: 1 / 0
    assert loader._drop_reverse_fp8_conversions(model) == 0

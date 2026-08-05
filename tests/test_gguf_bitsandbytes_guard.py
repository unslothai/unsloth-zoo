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

"""Tests the bitsandbytes guard in llama_cpp.convert_to_gguf.

llama.cpp has no bitsandbytes dequantizer, and only raises
`NotImplementedError: Quant method is not yet supported: 'bitsandbytes'` after
reading the entire model, so the guard has to fire before the subprocess.

The detector searches nested dicts because VLMs keep their quantization_config
under a sub-config, the same reason _remove_quantization_config recurses.
"""

import ast
from pathlib import Path

import pytest

LLAMA_CPP = Path(__file__).resolve().parents[1] / "unsloth_zoo" / "llama_cpp.py"
_SRC = LLAMA_CPP.read_text(encoding = "utf-8")


def _load():
    for node in ast.parse(_SRC).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_find_bitsandbytes_quantization":
            ns = {}
            exec(ast.get_source_segment(_SRC, node), ns)
            return ns[node.name]
    raise AssertionError("_find_bitsandbytes_quantization not found")


find = _load()


def test_top_level_bitsandbytes_is_found():
    cfg = {"quantization_config": {"quant_method": "bitsandbytes",
                                   "load_in_4bit": True}}
    assert find(cfg) == "config.json"


def test_nested_text_config_is_found():
    # VLMs keep it under a sub-config.
    cfg = {"text_config": {"quantization_config": {"quant_method": "bitsandbytes"}}}
    assert find(cfg) == "config.json['text_config']"


def test_deeply_nested_is_found():
    cfg = {"a": {"b": {"quantization_config": {"quant_method": "bitsandbytes"}}}}
    assert find(cfg) == "config.json['a']['b']"


def test_legacy_config_without_quant_method():
    # Older checkpoints only carry the bnb flags.
    assert find({"quantization_config": {"load_in_4bit": True}}) == "config.json"
    assert find({"quantization_config": {"load_in_8bit": True}}) == "config.json"


def test_clean_16bit_config_passes():
    assert find({"model_type": "llama", "num_hidden_layers": 32}) is None


def test_other_quant_methods_are_not_flagged():
    # fp8 and gptq have their own handling; only bitsandbytes is refused here.
    for method in ("fp8", "gptq", "awq", "compressed-tensors", "mxfp4"):
        assert find({"quantization_config": {"quant_method": method}}) is None, method


def test_non_dict_inputs_are_safe():
    assert find(None) is None
    assert find([1, 2, 3]) is None
    assert find({"quantization_config": "not-a-dict"}) is None
    assert find({"text_config": None}) is None


def test_guard_is_wired_into_convert_to_gguf():
    body = _SRC[_SRC.index("def convert_to_gguf("):]
    assert "_find_bitsandbytes_quantization(config_file)" in body
    # The whole point is failing fast, so the guard has to come before the
    # converter subprocess rather than after a multi-GB read.
    guard = body.index("_find_bitsandbytes_quantization(config_file)")
    launch = body.index("subprocess.run")
    assert guard < launch


def test_error_message_is_actionable():
    assert "load_in_4bit = False" in _SRC
    # An 8bit checkpoint carries the same quant_method and hits the same guard,
    # so the remediation has to name its flag too.
    assert "load_in_8bit = False" in _SRC
    assert "merged_16bit" in _SRC


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

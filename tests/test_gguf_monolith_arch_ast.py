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

"""Architecture discovery must read the converter, never execute it.

convert_hf_to_gguf.py is downloaded from llama.cpp's master branch at runtime.
The monolith layout used to import it to read ModelBase._model_classes, which
executes the freshly downloaded file inside the training process. It is now
AST-parsed instead, the same way the newer conversion/ package layout already
is.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_MODULE = None


def _load_llama_cpp_module():
    """Loaded once per session: re-executing llama_cpp for every test churns
    global torch state and slows the suite down for no extra coverage."""
    global _MODULE
    if _MODULE is None:
        repo_root = Path(__file__).resolve().parents[1]
        module_path = repo_root / "unsloth_zoo" / "llama_cpp.py"
        spec = importlib.util.spec_from_file_location("llama_cpp_under_test", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _MODULE = module
    return _MODULE


MONOLITH = '''
import gguf

class ModelBase:
    @classmethod
    def register(cls, *names, **kwargs):
        def wrap(c): return c
        return wrap

class MmprojModel(ModelBase): pass

@ModelBase.register("LlamaForCausalLM", "MistralForCausalLM")
class LlamaModel(ModelBase): pass

@ModelBase.register("Qwen2ForCausalLM")
class Qwen2Model(ModelBase): pass

@ModelBase.register("Gemma3ForConditionalGeneration", model_type=ModelType.MMPROJ)
class Gemma3VisionModel(MmprojModel): pass

@ModelBase.register("WhisperForConditionalGeneration")
class WhisperEncoderModel(MmprojModel): pass

@ModelBase.register("UltravoxModel")
class UltravoxWhisperEncoderModel(WhisperEncoderModel): pass
'''


def test_registered_archs_are_extracted_statically():
    llama_cpp = _load_llama_cpp_module()
    text, vision = llama_cpp._extract_archs_from_monolith_source(MONOLITH.encode())

    assert text == {"LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM"}
    # Direct, keyword-tagged and transitively-inherited mmproj classes all count.
    assert vision == {
        "Gemma3ForConditionalGeneration",
        "WhisperForConditionalGeneration",
        "UltravoxModel",
    }


def test_source_is_never_executed(tmp_path):
    llama_cpp = _load_llama_cpp_module()
    marker = tmp_path / "pwned"
    hostile = (
        f"import pathlib\n"
        f"pathlib.Path({str(marker)!r}).write_text('pwned')\n"
        + MONOLITH
    )

    text, _ = llama_cpp._extract_archs_from_monolith_source(hostile.encode())

    assert "LlamaForCausalLM" in text
    assert not marker.exists(), "the downloaded converter was executed"


def test_unparseable_source_yields_empty_sets():
    llama_cpp = _load_llama_cpp_module()
    assert llama_cpp._extract_archs_from_monolith_source(b"def broken(:\n") == (set(), set())


def test_no_in_process_module_loader_remains():
    llama_cpp = _load_llama_cpp_module()
    assert not hasattr(llama_cpp, "_load_module_from_path")


STATIC_REGISTRY = '''
from enum import IntEnum


class ModelType(IntEnum):
    TEXT = 0
    MMPROJ = 1


class ModelBase:
    _model_classes = {
        ModelType.TEXT: {"LlamaForCausalLM": object, "MistralForCausalLM": object},
        ModelType.MMPROJ: {"Gemma3ForConditionalGeneration": object},
    }
'''


def test_statically_initialized_registries_are_harvested():
    """Some converters seed _model_classes literally instead of decorating; the
    old import-based introspection saw those entries, so the parser must too."""
    llama_cpp = _load_llama_cpp_module()
    text, vision = llama_cpp._extract_archs_from_monolith_source(STATIC_REGISTRY.encode())

    assert text == {"LlamaForCausalLM", "MistralForCausalLM"}
    assert vision == {"Gemma3ForConditionalGeneration"}


def test_static_and_decorated_registries_combine():
    llama_cpp = _load_llama_cpp_module()
    text, vision = llama_cpp._extract_archs_from_monolith_source(
        (STATIC_REGISTRY + MONOLITH).encode()
    )

    assert {"LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM"} <= text
    assert {"Gemma3ForConditionalGeneration", "UltravoxModel"} <= vision


def test_the_repo_monolith_fixture_still_yields_its_arch():
    """tests/test_convert_hf_to_gguf_patcher.py models a converter of exactly
    this shape; it must not regress to an empty allowlist."""
    llama_cpp = _load_llama_cpp_module()
    fixture = Path(__file__).with_name("test_convert_hf_to_gguf_patcher.py").read_text()
    start = fixture.index('_MONOLITH = b"""\\\n') + len('_MONOLITH = b"""\\\n')
    body = fixture[start:fixture.index('"""', start)]

    text, _ = llama_cpp._extract_archs_from_monolith_source(body.encode())
    assert "LlamaForCausalLM" in text

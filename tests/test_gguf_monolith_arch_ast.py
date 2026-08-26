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


def _load_llama_cpp_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "unsloth_zoo" / "llama_cpp.py"
    spec = importlib.util.spec_from_file_location("llama_cpp_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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

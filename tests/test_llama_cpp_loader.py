import sys
import textwrap
import importlib.util
from pathlib import Path


def _load_llama_cpp_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "unsloth_zoo" / "llama_cpp.py"
    spec = importlib.util.spec_from_file_location("llama_cpp_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_converter_is_parsed_not_imported(tmp_path):
    """The downloaded converter is AST-parsed, so a script whose sibling
    imports cannot resolve (and whose top level has side effects) still yields
    its registered architectures without executing anything.

    This replaces the old `_load_module_from_path` sibling-import test: that
    helper imported the freshly downloaded llama.cpp script in-process and has
    been removed.
    """
    llama_cpp = _load_llama_cpp_module()
    marker = tmp_path / "executed"
    source = textwrap.dedent(
        f"""
        from conversion import VALUE  # unresolvable sibling import
        import pathlib
        pathlib.Path({str(marker)!r}).write_text("executed")

        class ModelBase: pass

        @ModelBase.register("LlamaForCausalLM")
        class LlamaModel(ModelBase): pass
        """
    ).encode()

    sys.modules.pop("conversion", None)
    original_path = sys.path[:]
    try:
        text_archs, vision_archs = llama_cpp._extract_archs_from_monolith_source(source)

        assert text_archs == {"LlamaForCausalLM"}
        assert vision_archs == set()
        assert not marker.exists()
        assert str(tmp_path) not in sys.path
    finally:
        sys.path[:] = original_path

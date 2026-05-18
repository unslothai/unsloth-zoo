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


def test_load_module_from_path_resolves_sibling_imports(tmp_path):
    llama_cpp = _load_llama_cpp_module()
    (tmp_path / "conversion.py").write_text("VALUE = 'loaded from sibling'\n")
    script = tmp_path / "original_gguf_test.py"
    script.write_text(
        textwrap.dedent(
            """
            from conversion import VALUE

            RESULT = VALUE
            """
        )
    )

    sys.modules.pop("conversion", None)
    original_path = sys.path[:]
    try:
        module = llama_cpp._load_module_from_path(str(script), "original_gguf_test")

        assert module.RESULT == "loaded from sibling"
        assert str(tmp_path) not in sys.path
    finally:
        sys.path[:] = original_path

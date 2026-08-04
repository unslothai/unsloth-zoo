"""A model must not fail to LOAD because we could not read its source.

`unsloth_compile_transformers` calls

    full_source = inspect.getsource(modeling_file)

and everything after it is source-level feature detection and regex class
discovery -- work done purely to make the model FASTER. `inspect.getsource`
reads through linecache, and when that comes back empty it raises

    OSError: could not get source code

Unguarded, that propagates out of `FastModel.from_pretrained`, so the model
does not load at all, and the message names neither unsloth nor the module it
could not read. Hit while collecting the whisper saving test.

Returning early is the honest degradation: LoRA forwards are already patched
by that point, and the model loads without the source-level optimisations.

What we must NOT do is carry on with an empty string. Several checks have the
shape

    "_supports_sdpa = False" not in full_source

which is TRUE for an empty string, so an empty `full_source` would silently
enable paths the model never claimed to support. Slower is fine; wrong is not.
"""

import ast
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COMPILER = ROOT / "unsloth_zoo" / "compiler.py"
SRC = COMPILER.read_text(encoding="utf-8")


def _guard_region() -> str:
    i = SRC.index("full_source = inspect.getsource(modeling_file)")
    return SRC[max(0, i - 400):i + 1600]


# ---- the guard ------------------------------------------------------------

def test_getsource_is_wrapped():
    region = _guard_region()
    assert "try:" in region
    assert "full_source = inspect.getsource(modeling_file)" in region


def test_it_catches_what_getsource_actually_raises():
    """OSError for unreadable source, TypeError for a built-in or C module."""
    region = _guard_region()
    assert "except (OSError, TypeError)" in region


def test_the_real_exception_type_is_oserror():
    """Guards the premise: if inspect ever stopped raising OSError here, the
    except clause would be catching nothing."""
    with pytest.raises((OSError, TypeError)):
        inspect.getsource(sys)          # built-in module, no source file


def test_it_returns_rather_than_continuing_with_empty_source():
    """The critical half. Continuing with full_source = "" would make every
    `X not in full_source` check true."""
    i = SRC.index("except (OSError, TypeError)")
    window = SRC[i:i + 1400]
    assert "\n        return\n" in window, (
        "the handler must return, not fall through to the source-based logic")
    assert 'full_source = ""' not in window, (
        "an empty full_source silently flips `not in` checks to True")


def test_it_warns_so_the_slowdown_is_not_silent():
    i = SRC.index("except (OSError, TypeError)")
    window = SRC[i:i + 1400]
    assert "logger.warning" in window


def test_the_logger_name_exists_in_this_module():
    """compiler.py imports `logger` from .log; there is no module-level
    `logger_compiler` at runtime. A warning that raises NameError inside an
    exception handler would replace a clear failure with a confusing one."""
    import unsloth_zoo.compiler as C
    assert hasattr(C, "logger")


def test_the_warning_says_the_model_still_works():
    i = SRC.index("except (OSError, TypeError)")
    window = SRC[i:i + 1400]
    assert "still works" in window, (
        "a warning during model load must say whether it is fatal")


# ---- what must be preserved ----------------------------------------------

def test_lora_patching_happens_before_the_guard():
    """Returning early is only acceptable because the LoRA forwards have
    already been patched by this point."""
    lora = SRC.index("patch_lora_forwards(torch_compile_options)")
    guard = SRC.index("full_source = inspect.getsource(modeling_file)")
    assert lora < guard


def test_the_patched_marker_is_set_before_the_guard():
    marker = SRC.index("modeling_file.__UNSLOTH_PATCHED__ = True")
    guard = SRC.index("full_source = inspect.getsource(modeling_file)")
    assert marker < guard, (
        "an early return must not leave the module looking unpatched, or a "
        "later pass would try again and fail again")


def test_the_module_still_parses():
    ast.parse(SRC)


def test_a_bare_return_is_the_functions_contract():
    """unsloth_compile_transformers returns None and already has early
    returns, so this one is consistent with the rest."""
    tree = ast.parse(SRC)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "unsloth_compile_transformers")
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert returns, "expected early returns in this function"
    assert any(r.value is None for r in returns)


# ---- the second site: loading a SECOND model in one process --------------

def _nn_patch_region() -> str:
    i = SRC.index("source = inspect.getsource(function.forward).rstrip()")
    return SRC[max(0, i - 700):i + 1400]


def test_the_nn_forward_patch_loop_is_guarded():
    """`inspect.getsource` on a forward it cannot retrieve raises
    `OSError: could not get source code`, which propagates out of
    `FastModel.from_pretrained` -- so the model does not load, over source we
    only wanted in order to patch a dtype cast.

    Observed while collecting tests/saving/text_to_speech_models, after
    another model had been loaded in the same process.

    Scope, stated honestly: the precondition is NOT fully characterised. Two
    plain Qwen loads do not trigger it, and after such a load no torch.nn
    forward is unreadable -- both measured, not assumed. This guards a state
    we have observed rather than encoding a theory about how it arises.
    """
    region = _nn_patch_region()
    assert "try:" in region
    assert "except (OSError, TypeError)" in region


def test_an_unreadable_forward_is_skipped_not_fatal():
    """Every other unsupported case in this loop uses `continue`; this must
    too. Raising abandons the remaining modules as well as this one.

    Checked on the parsed handler rather than its text: the comment inside it
    contains the word "raises", which a substring search reads as a `raise`
    statement.
    """
    tree = ast.parse(SRC)
    # Exactly the try whose body IS this assignment -- several other blocks
    # also call getsource on a forward, and their handlers legitimately do
    # something else.
    target = "source = inspect.getsource(function.forward).rstrip()"
    handlers = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try) or len(node.body) != 1:
            continue
        try:
            body_src = ast.unparse(node.body[0])
        except Exception:
            continue
        if body_src.replace(" ", "") == target.replace(" ", ""):
            handlers.extend(node.handlers)
    assert handlers, f"no try/except whose body is exactly `{target}`"
    for h in handlers:
        body = list(ast.walk(ast.Module(body=h.body, type_ignores=[])))
        assert any(isinstance(n, ast.Continue) for n in body), (
            "the handler must `continue` to the next module")
        assert not any(isinstance(n, ast.Raise) for n in body), (
            "raising here abandons every remaining module too")


def test_the_compiler_config_check_is_kept():
    """It catches torch.compile wrappers, which is a different case from our
    own exec'd replacements -- the new guard adds to it, not replaces it."""
    region = _nn_patch_region()
    assert 'hasattr(function.forward, "get_compiler_config")' in region


def test_both_getsource_guards_are_present():
    """Two distinct sites, two distinct failures. Fixing only the first one
    just moves the crash later, which is exactly what happened."""
    assert SRC.count("except (OSError, TypeError)") >= 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

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
    """The modeling-file guard and its handler.

    Anchored on the call, not on "except (OSError, TypeError)": the torch
    forward guard uses the same clause and is defined earlier in the file.
    """
    i = SRC.index("full_source = inspect.getsource(modeling_file)")
    return SRC[max(0, i - 400):i + 2200]


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
    window = _guard_region()
    assert "\n        return\n" in window, (
        "the handler must return, not fall through to the source-based logic")
    assert 'full_source = ""' not in window, (
        "an empty full_source silently flips `not in` checks to True")


def test_it_warns_so_the_slowdown_is_not_silent():
    assert "logger.warning" in _guard_region()


def test_the_logger_name_exists_in_this_module():
    """compiler.py imports `logger` from .log; there is no module-level
    `logger_compiler` at runtime. A warning that raises NameError inside an
    exception handler would replace a clear failure with a confusing one."""
    import unsloth_zoo.compiler as C
    assert hasattr(C, "logger")


def test_the_warning_says_the_model_still_works():
    window = _guard_region()
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
    return SRC[max(0, i - 1600):i + 1400]


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


# ---- what the fallback must still do -------------------------------------
#
# Returning early is a degradation, and a degradation has to be honest about
# which of the surrounding work it gave up. Three separate questions, and they
# do not have the same answer.


def _break_getsource(monkeypatch, module):
    """Make exactly one module's source unreadable, the way the wild failure
    does: the file is gone as far as inspect and linecache are concerned."""
    import linecache

    path = inspect.getfile(module)
    real = inspect.getfile

    def broken(obj):
        found = real(obj)
        if found == path:
            raise OSError("could not get source code")
        return found

    monkeypatch.setattr(inspect, "getfile", broken)
    linecache.checkcache(path)


@pytest.fixture
def unreadable_llama(monkeypatch):
    import transformers.models.llama.modeling_llama as modeling

    _break_getsource(monkeypatch, modeling)
    # Both markers, with their values: restoring only the patched one leaves a
    # module that reports itself done and then declines to answer, so a later
    # test's supports_sdpa accumulator is silently left untouched.
    _MISSING = object()
    saved = {name: getattr(modeling, name, _MISSING)
             for name in ("__UNSLOTH_PATCHED__", "__UNSLOTH_SUPPORTS_SDPA__")}
    for name in saved:
        if saved[name] is not _MISSING:
            delattr(modeling, name)
    yield modeling
    for name, value in saved.items():
        if hasattr(modeling, name):
            delattr(modeling, name)
        if value is not _MISSING:
            setattr(modeling, name, value)


def test_sdpa_is_reported_unsupported(unreadable_llama):
    """The caller seeds `[True]` and only the normal path writes it, so a
    silent return leaves SDPA selected for a model whose source never claimed
    it. Eager always exists; guessing does not."""
    from unsloth_zoo import compiler

    supports_sdpa = [True]
    compiler.unsloth_compile_transformers(
        "llama", disable = True, compile_torch_modules = False,
        supports_sdpa = supports_sdpa,
    )
    assert supports_sdpa == [False]


def test_the_answer_survives_a_second_call(unreadable_llama):
    """`__UNSLOTH_PATCHED__` is set before the guard, so every later call takes
    the already-patched branch and reads `__UNSLOTH_SUPPORTS_SDPA__` instead.
    Unset, that branch leaves the caller's optimistic value alone."""
    from unsloth_zoo import compiler

    compiler.unsloth_compile_transformers(
        "llama", disable = True, compile_torch_modules = False,
        supports_sdpa = [True],
    )
    again = [True]
    compiler.unsloth_compile_transformers(
        "llama", disable = True, compile_torch_modules = False,
        supports_sdpa = again,
    )
    assert again == [False]


def test_a_supports_sdpa_of_none_is_still_allowed(unreadable_llama):
    """It is optional; the normal path guards it and so must this one."""
    from unsloth_zoo import compiler

    compiler.unsloth_compile_transformers(
        "llama", disable = True, compile_torch_modules = False,
    )


def test_the_torch_dtype_patches_still_run():
    """They read torch's source, not the model's, so an unreadable model is no
    reason to skip them. Skipping turned a load-time failure into a first
    forward pass failure."""
    tree = ast.parse(SRC)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "unsloth_compile_transformers")
    for node in ast.walk(fn):
        if not isinstance(node, ast.Try):
            continue
        if "inspect.getsource(modeling_file)" not in ast.unparse(node.body[0]):
            continue
        handler = ast.unparse(node.handlers[0])
        assert "_patch_torch_dtype_modules" in handler
        return
    pytest.fail("no try whose body reads the modeling file source")


def test_gradient_accumulation_needs_that_same_source():
    """Which is why the early return does not skip anything there: the patch
    re-reads the model's source itself and gives up on the same failure. This
    test exists so a future reader does not add a fallback for a case that
    cannot occur."""
    import transformers.models.llama.modeling_llama as modeling
    from unsloth_zoo.compiler import patch_gradient_accumulation

    with pytest.MonkeyPatch.context() as monkeypatch:
        _break_getsource(monkeypatch, modeling)
        patched = [
            name for name in dir(modeling)
            if isinstance(getattr(modeling, name, None), type)
            and patch_gradient_accumulation(modeling, name) is not None
        ]
    assert patched == []


# ---- the wrapper that stands in for an unreadable torch forward ----------


class _Weighted:
    def __init__(self, dtype):
        import torch

        self.weight = torch.zeros(1, dtype = dtype)


def _run(is_conv, disable, input_dtype, weight_dtype):
    """Returns (dtype the original forward saw, dtype the caller got back)."""
    import torch

    from unsloth_zoo.compiler import _dtype_safe_forward

    seen = {}

    def original(self, x, *args, **kwargs):
        seen["dtype"] = x.dtype
        return x.to(torch.float32)

    forward = _dtype_safe_forward(original, is_conv, disable)
    out = forward(_Weighted(weight_dtype), torch.zeros(2, dtype = input_dtype))
    return seen["dtype"], out.dtype


def test_a_conv_input_is_cast_to_the_weight_dtype():
    import torch

    saw, got = _run(True, False, torch.float16, torch.bfloat16)
    assert saw == torch.bfloat16, "eager F.conv1d crashes on mismatched dtypes"
    assert got == torch.float16, "the caller must get its own dtype back"


def test_an_eager_norm_input_is_cast_too():
    import torch

    saw, got = _run(False, True, torch.bfloat16, torch.float32)
    assert (saw, got) == (torch.float32, torch.bfloat16)


def test_a_compiled_norm_input_is_left_alone():
    """The source rewrite only casts the result when compiling is on; casting
    the input as well changes batched numerics."""
    import torch

    saw, got = _run(False, False, torch.bfloat16, torch.float32)
    assert (saw, got) == (torch.bfloat16, torch.bfloat16)


def test_an_affineless_norm_has_no_weight_to_match():
    import torch

    from unsloth_zoo.compiler import _dtype_safe_forward

    class _NoWeight:
        weight = None

    forward = _dtype_safe_forward(lambda self, x: x.to(torch.float32), False, True)
    assert forward(_NoWeight(), torch.zeros(2, dtype = torch.bfloat16)).dtype \
        == torch.bfloat16


def test_the_wrapper_is_marked_so_it_is_not_wrapped_twice():
    """Loading a second model runs the loop again; without a marker each load
    would add another layer of casts."""
    from unsloth_zoo.compiler import _dtype_safe_forward

    assert _dtype_safe_forward(lambda self, x: x, True, False).__unsloth_dtype_wrapped__
    assert 'getattr(function.forward, "__unsloth_dtype_wrapped__", False)' in SRC


def test_the_unreadable_forward_is_wrapped_rather_than_dropped():
    region = _nn_patch_region()
    assert "_dtype_safe_forward(" in region


def test_the_tensor_may_arrive_by_keyword():
    """These become the public forward. torch.nn.RMSNorm declares `x`, not
    `input`, and `rms_norm(x = t)` works today, so a hard-coded parameter name
    would start raising TypeError on a call that used to be fine."""
    import torch

    from unsloth_zoo.compiler import _dtype_safe_forward

    def original(self, x):
        return x.to(torch.float32)

    forward = _dtype_safe_forward(original, False, False)
    assert forward(_Weighted(torch.float32),
                   x = torch.zeros(2, dtype = torch.bfloat16)).dtype \
        == torch.bfloat16


def test_a_call_it_cannot_read_is_passed_straight_through():
    """Better to cast nothing than to guess which argument is the activation."""
    import torch

    from unsloth_zoo.compiler import _dtype_safe_forward

    seen = {}

    def original(self, **kwargs):
        seen.update(kwargs)
        return torch.zeros(1)

    forward = _dtype_safe_forward(original, False, False)
    forward(_Weighted(torch.float32), other = 1)
    assert seen == {"other": 1}


def test_the_wrapper_records_the_compile_mode_it_was_built_for():
    """`disable` decides whether a norm casts its input, and it is baked into
    the closure, so a second load with the other setting must be able to tell."""
    from unsloth_zoo.compiler import _dtype_safe_forward

    def original(self, x):
        return x

    for disable in (True, False):
        forward = _dtype_safe_forward(original, False, disable)
        assert forward.__unsloth_dtype_disable__ is disable
        assert forward.__unsloth_dtype_original__ is original


def test_a_rebuild_wraps_the_original_not_the_wrapper():
    """Otherwise each load in a process adds another layer of casts."""
    from unsloth_zoo.compiler import _dtype_safe_forward

    def original(self, x):
        return x

    once = _dtype_safe_forward(original, False, True)
    twice = _dtype_safe_forward(once.__unsloth_dtype_original__, False, False)
    assert twice.__unsloth_dtype_original__ is original


def test_the_loop_rebuilds_on_a_mode_change():
    region = _nn_patch_region()
    i = SRC.index('__unsloth_dtype_wrapped__", False)')
    window = SRC[i:i + 800]
    assert "__unsloth_dtype_disable__" in window, (
        "a wrapper built for the other compile mode must not be reused")
    assert "__unsloth_dtype_original__" in window, (
        "rebuild from the original, or the wrappers stack")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

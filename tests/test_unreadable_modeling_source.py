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

"""Unreadable modeling source must not fail the load: unsloth_compile_transformers returns early on OSError, never with
an empty full_source, since a not-in check is true for an empty string and would silently enable unsupported paths;
slower is fine, wrong is not."""

import ast
import inspect
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COMPILER = ROOT / "unsloth_zoo" / "compiler.py"
SRC = COMPILER.read_text(encoding="utf-8")


def _require_index(anchor: str, what: str) -> int:
    """Looks up an anchor in SRC with a message naming the contract; a bare .index miss names neither, and issue 967's
    rename failed six tests at the test file's own line."""
    i = SRC.find(anchor)
    assert i != -1, (
        f"{what}: no `{anchor}` in compiler.py. Either the guard is gone, or "
        f"it was renamed and this anchor needs to follow it.")
    return i


def _guard_region() -> str:
    """The modeling-file guard and its handler.

    Anchored on the call, not on "except (OSError, TypeError)": the torch
    forward guard uses the same clause and is defined earlier in the file.
    """
    i = _require_index(
        "full_source = inspect.getsource(modeling_file)",
        "the modeling-file guard")
    return SRC[max(0, i - 400):i + 2200]


def _caught_names(handler) -> set:
    """The exception names an `except` clause catches, spelled as they are written."""
    node = handler.type
    if node is None:
        return set()
    parts = node.elts if isinstance(node, ast.Tuple) else [node]
    return {ast.unparse(part) for part in parts}


def _catches(caught: set, name: str) -> bool:
    """Whether name is caught, matched only by a spelling this module can resolve: compiler.py imports the tokenize
    module, not TokenError, so a bare handler would be a NameError."""
    if name in caught:
        return True
    bare = name.rsplit(".", 1)[-1]
    return bare in caught and re.search(
        rf"^from [\w.]+ import [^\n]*\b{bare}\b", SRC, re.M
    ) is not None


def _getsource_guards(call: str) -> list:
    """What each try whose body calls the given call catches, one union per try, matched on the parse; only the nearest
    enclosing try counts. Widening to tokenize.TokenError (issue 1149) failed three tests while catching more."""
    matching = [
        node for node in ast.walk(ast.parse(SRC))
        if isinstance(node, ast.Try)
        and any(call in ast.unparse(stmt) for stmt in node.body)
    ]
    return [
        set().union(set(), *(_caught_names(h) for h in node.handlers))
        for node in matching
        if not any(
            other is not node and any(d is other for d in ast.walk(node))
            for other in matching
        )
    ]


def test_getsource_is_wrapped():
    region = _guard_region()
    assert "try:" in region
    assert "full_source = inspect.getsource(modeling_file)" in region


def test_it_catches_what_getsource_actually_raises():
    """OSError for unreadable source, TypeError for a built-in or C module."""
    guards = _getsource_guards("inspect.getsource(modeling_file)")
    assert guards, "the modeling-file getsource is no longer inside a try"
    for caught in guards:
        assert _catches(caught, "OSError") and _catches(caught, "TypeError"), (
            f"the modeling-file guard stopped catching one of them: {sorted(caught)}")
    # Widened by #1149: getblock's TokenError subclasses Exception directly.
    assert any(_catches(caught, "tokenize.TokenError") for caught in guards)


def test_the_real_exception_type_is_oserror():
    """Guards the premise: if inspect.getsource stopped raising OSError for a built-in module, the except clause would catch nothing."""
    with pytest.raises((OSError, TypeError)):
        inspect.getsource(sys)          # built-in module, no source file


def test_it_returns_rather_than_continuing_with_empty_source():
    """The critical half: continuing with an empty full_source would make every X not in full_source check true, so the handler must return."""
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


def test_lora_patching_happens_before_the_guard():
    """An early return is acceptable only because the LoRA forwards are already patched by the time the modeling-file guard runs."""
    lora = _require_index(
        "patch_lora_forwards(torch_compile_options)", "the LoRA forward patch")
    guard = _require_index(
        "full_source = inspect.getsource(modeling_file)", "the modeling-file guard")
    assert lora < guard


def test_the_patched_marker_is_set_before_the_guard():
    marker = _require_index(
        "modeling_file.__UNSLOTH_PATCHED__ = True", "the patched marker")
    guard = _require_index(
        "full_source = inspect.getsource(modeling_file)", "the modeling-file guard")
    assert marker < guard, (
        "an early return must not leave the module looking unpatched, or a "
        "later pass would try again and fail again")


def test_the_module_still_parses():
    ast.parse(SRC)


def test_a_bare_return_is_the_functions_contract():
    """unsloth_compile_transformers returns None and already has early returns, so a bare return matches the function's contract."""
    tree = ast.parse(SRC)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "unsloth_compile_transformers")
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert returns, "expected early returns in this function"
    assert any(r.value is None for r in returns)


def _nn_forward_read() -> tuple:
    """Returns (name, statement) of the forward read, matched by shape so a correct rename (issue 967 moved
    function.forward to original_forward) does not fail it; pinned by
    test_the_forward_source_is_read_from_the_pinned_original."""
    found = []
    for node in ast.walk(ast.parse(SRC)):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and target.id == "source"):
            continue
        call = node.value
        if not (isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "rstrip"):
            continue
        inner = call.func.value
        if not (isinstance(inner, ast.Call)
                and ast.unparse(inner.func) == "inspect.getsource"
                and len(inner.args) == 1
                and isinstance(inner.args[0], ast.Name)):
            continue
        found.append((inner.args[0].id, ast.unparse(node)))

    assert found, (
        "no `source = inspect.getsource(<name>).rstrip()` in compiler.py at "
        "all. The torch.nn forward patch loop no longer reads the forward's "
        "source the way every test below assumes.")
    assert len({name for name, _ in found}) == 1, (
        f"several forward-source reads with different operands: {sorted(found)}. "
        "These tests assume one site and would guard an arbitrary one of them.")
    return found[0]


NN_FORWARD_NAME, NN_FORWARD_READ = _nn_forward_read()
NN_FORWARD_GETSOURCE = f"inspect.getsource({NN_FORWARD_NAME})"


def _nn_patch_region() -> str:
    i = _require_index(NN_FORWARD_READ, "the torch.nn forward patch loop")
    return SRC[max(0, i - 1600):i + 1400]


def test_the_forward_source_is_read_from_the_pinned_original():
    """The read operand must be function.forward or a local assigned from it in the same loop (issue 967 pin), keeping
    the read idempotent across a vision load's two passes; otherwise _nn_forward_read could latch onto a non-forward
    rename and every test would still pass."""
    if NN_FORWARD_NAME == "function":
        pytest.skip("read directly off `function.forward`, nothing to pin")
    pin = f"{NN_FORWARD_NAME} = function.forward"
    i = _require_index(pin, "the pristine-forward pin")
    read = _require_index(NN_FORWARD_READ, "the forward source read")
    assert i < read, (
        f"`{pin}` must precede `{NN_FORWARD_READ}`, or the loop is not reading "
        "the forward it pinned")


def test_the_nn_forward_patch_loop_is_guarded():
    """An unreadable torch.nn forward raises OSError out of FastModel.from_pretrained, so the model fails to load; seen
    in tests/saving/text_to_speech_models after another model loaded in the same process; trigger not fully
    characterised, so this guards the observed state."""
    guards = _getsource_guards(NN_FORWARD_GETSOURCE)
    assert guards, f"`{NN_FORWARD_GETSOURCE}` is no longer inside a try"
    for caught in guards:
        assert _catches(caught, "OSError") and _catches(caught, "TypeError"), (
            f"the forward guard stopped catching one of them: {sorted(caught)}")
    assert any(_catches(caught, "tokenize.TokenError") for caught in guards)


def test_an_unreadable_forward_is_skipped_not_fatal():
    """The handler must continue like every other unsupported case in this loop, since raising would abandon the
    remaining modules too; checked on the parsed AST, not the text."""
    tree = ast.parse(SRC)
    # Matches only the try whose body is this assignment; other blocks also call getsource on a forward and their handlers may legitimately do something else.
    target = NN_FORWARD_READ
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
    """The hasattr get_compiler_config check catches torch.compile wrappers, a different case from our exec'd
    replacements, so the new guard adds to it rather than replacing it."""
    region = _nn_patch_region()
    assert 'hasattr(function.forward, "get_compiler_config")' in region


def test_both_getsource_guards_are_present():
    """Two distinct getsource sites with distinct failures; fixing only the first just moved the crash later, which is exactly what happened."""
    sites = {"inspect.getsource(modeling_file)", NN_FORWARD_GETSOURCE}
    for call in sites:
        guards = _getsource_guards(call)
        assert guards and all(
            _catches(c, "OSError") and _catches(c, "TypeError") for c in guards
        ), (
            f"{call} is unguarded, so an unreadable source is fatal again")


# An early return is a degradation, so say which surrounding work it gives up; the three separate questions here do not have the same answer.


def _break_getsource(monkeypatch, module):
    """Makes one module's source unreadable the way the wild failure does: the file is gone as far as inspect and linecache are concerned."""
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
    # Both markers must be restored with their saved values: restoring only __UNSLOTH_PATCHED__ leaves a module that
    # reports itself done yet declines to answer, and a later test's supports_sdpa accumulator is silently left
    # untouched.
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
    """The caller seeds supports_sdpa as [True] and only the normal path writes it, so a silent return leaves SDPA
    selected for a model whose source never claimed it; eager always exists, guessing does not."""
    from unsloth_zoo import compiler

    supports_sdpa = [True]
    compiler.unsloth_compile_transformers(
        "llama", disable = True, compile_torch_modules = False,
        supports_sdpa = supports_sdpa,
    )
    assert supports_sdpa == [False]


def test_the_answer_survives_a_second_call(unreadable_llama):
    """__UNSLOTH_PATCHED__ is set before the guard, so later calls read __UNSLOTH_SUPPORTS_SDPA__ instead; unset, that branch keeps the caller's optimistic value."""
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
    """The torch dtype patches read torch's source, not the model's, so an unreadable model is no reason to skip them;
    skipping turned a load-time failure into a first-forward-pass failure."""
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
    """patch_gradient_accumulation re-reads the model's source and gives up on the same failure, so the early return skips nothing there and no fallback is needed."""
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
    """The source rewrite casts the result only when compiling is on; casting the input as well changes batched numerics."""
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
    """Loading a second model reruns the loop; without the __unsloth_dtype_wrapped__ marker each load would add another layer of casts."""
    from unsloth_zoo.compiler import _dtype_safe_forward

    assert _dtype_safe_forward(lambda self, x: x, True, False).__unsloth_dtype_wrapped__
    assert 'getattr(function.forward, "__unsloth_dtype_wrapped__", False)' in SRC


def test_the_unreadable_forward_is_wrapped_rather_than_dropped():
    region = _nn_patch_region()
    assert "_dtype_safe_forward(" in region


def test_the_tensor_may_arrive_by_keyword():
    """The wrapper becomes the public forward; torch.nn.RMSNorm declares x, not input, so a hard-coded parameter name
    would break keyword calls such as rms_norm(x = t) with TypeError."""
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
    """disable decides whether a norm casts its input and is baked into the closure, so a later load with the other setting must be detectable."""
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
    i = _require_index(
        '__unsloth_dtype_wrapped__", False)', "the already-wrapped check")
    window = SRC[i:i + 800]
    assert "__unsloth_dtype_disable__" in window, (
        "a wrapper built for the other compile mode must not be reused")
    assert "__unsloth_dtype_original__" in window, (
        "rebuild from the original, or the wrappers stack")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

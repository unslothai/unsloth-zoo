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

"""Blocking `import flashinfer` for vLLM must not be a one-way door.

Clearing VLLM_ATTENTION_BACKEND / VLLM_USE_FLASHINFER_SAMPLER does not steer vLLM on a
cu128 runtime image without nvcc or ninja: vLLM picks FlashInfer from its own sm_100
default backend list and dies JIT compiling trtllm-gen kernels. Hiding the package via
`sys.modules["flashinfer"] = None` is what makes its `has_flashinfer()` find_spec probe
report the package absent, so `get_attn_backend_cls` falls through to FLASH_ATTN.

That hide is process wide, so left permanent a session that installs nvcc/ninja
afterwards could never get FlashInfer back, and the damage would be invisible to
find_spec itself. Everything below is about the block being exactly reversible, and
about which statements sit inside the package-presence guard.

No vLLM and no FlashInfer are needed: the package is faked in sys.modules and the
branch structure is read out of load_vllm as AST.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
import types

import pytest


vllm_utils = importlib.import_module("unsloth_zoo.vllm_utils")


@pytest.fixture
def fake_flashinfer(monkeypatch):
    root = types.ModuleType("flashinfer")
    root.__spec__ = importlib.machinery.ModuleSpec("flashinfer", loader = None)
    root.__path__ = []
    sub = types.ModuleType("flashinfer.decode")
    sub.__spec__ = importlib.machinery.ModuleSpec("flashinfer.decode", loader = None)

    monkeypatch.setitem(sys.modules, "flashinfer", root)
    monkeypatch.setitem(sys.modules, "flashinfer.decode", sub)
    monkeypatch.setattr(vllm_utils, "_UNSLOTH_BLOCKED_FLASHINFER_MODULES", {})
    return types.SimpleNamespace(root = root, sub = sub)


def _import_flashinfer_fails() -> bool:
    try:
        importlib.import_module("flashinfer")
    except ImportError:
        return True
    return False


def test_block_then_unblock_restores_the_modules(fake_flashinfer):
    vllm_utils._block_flashinfer_import()
    assert sys.modules["flashinfer"] is None
    assert "flashinfer.decode" not in sys.modules
    assert importlib.util.find_spec("flashinfer") is None
    assert _import_flashinfer_fails()

    vllm_utils._unblock_flashinfer_import()
    assert sys.modules["flashinfer"] is fake_flashinfer.root
    assert sys.modules["flashinfer.decode"] is fake_flashinfer.sub
    assert importlib.util.find_spec("flashinfer") is not None
    assert not _import_flashinfer_fails()


def test_unblock_leaves_a_package_that_was_never_imported_absent(monkeypatch):
    monkeypatch.delitem(sys.modules, "flashinfer", raising = False)
    monkeypatch.setattr(vllm_utils, "_UNSLOTH_BLOCKED_FLASHINFER_MODULES", {})

    vllm_utils._block_flashinfer_import()
    assert sys.modules["flashinfer"] is None

    vllm_utils._unblock_flashinfer_import()
    assert "flashinfer" not in sys.modules


def test_unblock_restores_a_pre_existing_none_entry(monkeypatch):
    """Someone else's import block is a real sys.modules state, not the same thing as
    the package being absent, so the unblock has to put it back rather than delete it."""
    monkeypatch.setitem(sys.modules, "flashinfer", None)
    monkeypatch.setattr(vllm_utils, "_UNSLOTH_BLOCKED_FLASHINFER_MODULES", {})

    vllm_utils._block_flashinfer_import()
    assert sys.modules["flashinfer"] is None

    vllm_utils._unblock_flashinfer_import()
    assert "flashinfer" in sys.modules
    assert sys.modules["flashinfer"] is None


def test_unblock_without_a_block_is_a_no_op(fake_flashinfer):
    vllm_utils._unblock_flashinfer_import()
    assert sys.modules["flashinfer"] is fake_flashinfer.root


def test_repeated_blocks_keep_the_original_modules(fake_flashinfer):
    vllm_utils._block_flashinfer_import()
    # A second call must not record the None sentinel as the thing to restore.
    vllm_utils._block_flashinfer_import()
    vllm_utils._unblock_flashinfer_import()
    assert sys.modules["flashinfer"] is fake_flashinfer.root
    assert sys.modules["flashinfer.decode"] is fake_flashinfer.sub


def test_a_blocked_flashinfer_is_invisible_to_the_probe_vllm_uses(fake_flashinfer):
    """vLLM's `has_flashinfer()` is a `find_spec` call, so blocking the import is what
    actually stops it selecting the backend."""
    assert importlib.util.find_spec("flashinfer") is not None
    vllm_utils._block_flashinfer_import()
    assert importlib.util.find_spec("flashinfer") is None
    assert _import_flashinfer_fails()


def _flashinfer_chain():
    """load_vllm's FlashInfer branch chain as AST: which statements sit inside the
    find_spec guard is the whole question, so structure rather than substrings."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(vllm_utils.load_vllm)))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Call)
            and getattr(node.test.func, "id", "") == "_clear_flashinfer_env_on_hip"
        ):
            return node
    raise AssertionError("the FlashInfer branch chain is gone")


def _clears_forced_selection(body):
    """True when this statement list writes VLLM_USE_FLASHINFER_SAMPLER=0 itself, not nested."""
    import ast

    for stmt in body:
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if (
                isinstance(target, ast.Subscript)
                and getattr(target.value, "attr", "") == "environ"
                and getattr(target.slice, "value", "") == "VLLM_USE_FLASHINFER_SAMPLER"
                and getattr(stmt.value, "value", "") == "0"
            ):
                return True
    return False


def test_load_vllm_lifts_the_block_when_the_opt_out_is_cleared():
    import inspect

    source = inspect.getsource(vllm_utils.load_vllm)
    unblock = source.find("_unblock_flashinfer_import()")
    probe = source.find('elif importlib.util.find_spec("flashinfer"):')
    assert unblock != -1, "load_vllm never lifts a previously installed block"
    assert probe != -1
    assert unblock < probe, "the block must be lifted before the FlashInfer probe"


def test_the_unblock_is_conditional_on_the_opt_out_being_clear():
    """Unblocking unconditionally would undo the opt-out on every subsequent call."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(vllm_utils.load_vllm)))
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and getattr(node.test.operand, "id", "") == "_no_flashinfer"
    ]
    assert len(guards) == 1, "the unblock is not guarded on the opt-out being clear"
    calls = [
        getattr(getattr(stmt, "value", None), "func", None)
        for stmt in guards[0].body
        if isinstance(stmt, ast.Expr)
    ]
    assert [getattr(c, "id", "") for c in calls] == ["_unblock_flashinfer_import"]


def test_a_preset_opt_out_still_blocks_flashinfer():
    """A pre-set opt-out must still block: an installed FlashInfer is selected by vLLM anyway."""
    import inspect

    source = inspect.getsource(vllm_utils.load_vllm)
    optout = source.find("elif _no_flashinfer:")
    assert optout != -1, "load_vllm has no explicit pre-set opt-out branch"
    probe = source.find('elif importlib.util.find_spec("flashinfer"):')
    assert probe != -1, "the toolchain probe branch is gone"
    assert optout < probe, "the opt-out must be handled before the toolchain probe"

    # the opt-out branch must actually block, not merely skip our tuning
    branch = source[optout:probe]
    assert "_block_flashinfer_import()" in branch, branch
    assert 'os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"' in branch, branch


def test_only_the_import_blocker_is_guarded_on_flashinfer_being_installed():
    """A forced selection has to be cleared whether or not FlashInfer is installed: vLLM's
    TopKTopPSampler does an unguarded `from flashinfer import ...`, so an inherited
    VLLM_USE_FLASHINFER_SAMPLER=1 with the package absent raises instead of falling back."""
    import ast

    node = _flashinfer_chain()
    optout = node.orelse[0]
    assert isinstance(optout, ast.If) and getattr(optout.test, "id", "") == "_no_flashinfer"

    assert _clears_forced_selection(optout.body), (
        "the env cleanup is nested under the package-presence check again"
    )

    guards = [
        stmt
        for stmt in optout.body
        if isinstance(stmt, ast.If) and "find_spec" in ast.dump(stmt.test)
    ]
    assert len(guards) == 1, ast.dump(optout)
    assert not _clears_forced_selection(guards[0].body)
    calls = [
        getattr(getattr(stmt, "value", None), "func", None)
        for stmt in guards[0].body
        if isinstance(stmt, ast.Expr)
    ]
    assert [getattr(c, "id", "") for c in calls] == ["_block_flashinfer_import"], ast.dump(
        guards[0]
    )


def test_the_default_path_clears_a_forced_selection_too():
    """No opt-out, not ROCm, FlashInfer absent: the same unguarded vLLM import waits there."""
    import ast

    node = _flashinfer_chain()
    while node.orelse and len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
        node = node.orelse[0]
    assert node.orelse, "the chain has no final else, so an absent FlashInfer clears nothing"
    assert _clears_forced_selection(node.orelse)


def test_rocm_is_reached_before_any_blocking():
    """AMD takes its own path and FlashInfer never applies there, so the HIP branch must
    stay first in the chain and must not fall into the blocker."""
    import ast

    node = _flashinfer_chain()
    assert not any(
        getattr(getattr(stmt, "func", None), "id", "") == "_block_flashinfer_import"
        for stmt in ast.walk(node.test)
    )
    assert [type(stmt).__name__ for stmt in node.body] == ["Pass"], ast.dump(node)


def test_a_working_toolchain_is_not_blocked():
    """The probe branch only blocks when nvcc / ninja / libcuda.so are missing. Blocking
    in the `else` would disable FlashInfer on every healthy host."""
    import ast

    node = _flashinfer_chain()
    probe = node.orelse[0].orelse[0]
    assert isinstance(probe, ast.If) and "find_spec" in ast.dump(probe.test)

    missing = [stmt for stmt in probe.body if isinstance(stmt, ast.If)]
    assert len(missing) == 1, ast.dump(probe)
    blocked_when_missing = [
        stmt
        for stmt in missing[0].body
        if isinstance(stmt, ast.Expr)
        and getattr(getattr(stmt.value, "func", None), "id", "") == "_block_flashinfer_import"
    ]
    assert blocked_when_missing, "a broken toolchain no longer blocks the import"
    assert "_block_flashinfer_import" not in ast.dump(ast.Module(body = missing[0].orelse, type_ignores = [])), (
        "the healthy-toolchain branch blocks FlashInfer, which would disable it everywhere"
    )


def test_the_block_is_not_hoisted_above_the_model_support_check():
    """`vllm_supports_flashinfer` only rules out this MODEL, and the sampler still wants
    the package, so that branch must clear env vars and never hide the import."""
    import ast

    node = _flashinfer_chain()
    probe = node.orelse[0].orelse[0]
    support = [
        stmt
        for stmt in ast.walk(probe)
        if isinstance(stmt, ast.If)
        and "vllm_supports_flashinfer" in ast.dump(stmt.test)
    ]
    assert len(support) == 1, "the model support check is gone"
    assert "_block_flashinfer_import" not in ast.dump(support[0])


def test_patch_vllm_graph_capture_returns_early_without_vllm(monkeypatch):
    """It reads `vllm.__version__`, but `vllm` is only bound inside the module-level
    find_spec guard, so without the guard below this is a bare NameError.

    Both halves of that state have to be faked, otherwise this passes for the wrong
    reason on a host that does have vLLM: dropping the module global is what makes the
    version read fail, and blocking the import is what the guard has to catch."""
    monkeypatch.delattr(vllm_utils, "vllm", raising = False)
    monkeypatch.setitem(sys.modules, "vllm", None)
    assert vllm_utils.patch_vllm_graph_capture() is None


def test_load_vllm_raises_an_actionable_import_error_without_vllm(monkeypatch):
    """Without vLLM, `vllm_version` is an unbound module global and load_vllm used to
    die with NameError. It now raises ImportError, which is a user visible change."""
    monkeypatch.delattr(vllm_utils, "vllm_version", raising = False)
    with pytest.raises(ImportError, match = "vLLM is required"):
        vllm_utils.load_vllm(config = object())

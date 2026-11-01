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

"""Blocking flashinfer import must be reversible (process-wide); clearing VLLM_ATTENTION_BACKEND or
VLLM_USE_FLASHINFER_SAMPLER does not steer vLLM on cu128 without nvcc or ninja (sm_100 default picks FlashInfer;
trtllm-gen JIT dies); sys.modules None makes find_spec report absent so FLASH_ATTN wins."""

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
    """A pre-existing None entry in sys.modules is someone else's block, not an absent package, so unblock must restore it rather than delete it."""
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
    """vLLM's `has_flashinfer()` is a find_spec call, so blocking the import is what actually stops it selecting the FlashInfer backend."""
    assert importlib.util.find_spec("flashinfer") is not None
    vllm_utils._block_flashinfer_import()
    assert importlib.util.find_spec("flashinfer") is None
    assert _import_flashinfer_fails()


def _flashinfer_chain():
    """load_vllm's FlashInfer branch chain is checked as AST, since which statements sit inside the find_spec guard matters more than substrings."""
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

    def _tests_the_opt_out_is_clear(test):
        """Recognises `not _no_flashinfer` as the opt-out test, alone or as an `and` conjunct; the re-probe unblock adds
        a narrowing owner check (test_the_reprobe_unblock_also_respects_owners)."""
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
            return any(_tests_the_opt_out_is_clear(v) for v in test.values)
        return (
            isinstance(test, ast.UnaryOp)
            and isinstance(test.op, ast.Not)
            and getattr(test.operand, "id", "") == "_no_flashinfer"
        )

    tree = ast.parse(textwrap.dedent(inspect.getsource(vllm_utils.load_vllm)))
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and _tests_the_opt_out_is_clear(node.test)
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
    """Only the import blocker is guarded on FlashInfer being installed: forced env vars must still clear, since an
    inherited VLLM_USE_FLASHINFER_SAMPLER=1 makes TopKTopPSampler's unguarded import raise when the package is
    absent."""
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
    """AMD/HIP takes its own path where FlashInfer never applies, so the HIP branch must stay first in the chain and must not fall into the blocker."""
    import ast

    node = _flashinfer_chain()
    assert not any(
        getattr(getattr(stmt, "func", None), "id", "") == "_block_flashinfer_import"
        for stmt in ast.walk(node.test)
    )
    assert [type(stmt).__name__ for stmt in node.body] == ["Pass"], ast.dump(node)


def test_a_working_toolchain_is_not_blocked():
    """The probe branch blocks only when nvcc, ninja or libcuda.so is missing; blocking in the else would disable FlashInfer on every healthy host."""
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
    """`vllm_supports_flashinfer` only rules out this model while the sampler still needs the package, so that branch clears env vars and never hides the import."""
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
    """patch_vllm_graph_capture reads vllm.__version__ and vllm is bound only inside the find_spec guard; both the
    module global and the import block must be faked or the test passes for the wrong reason."""
    monkeypatch.delattr(vllm_utils, "vllm", raising = False)
    monkeypatch.setitem(sys.modules, "vllm", None)
    assert vllm_utils.patch_vllm_graph_capture() is None


def test_load_vllm_raises_an_actionable_import_error_without_vllm(monkeypatch):
    """Without vLLM, `vllm_version` is an unbound module global, so load_vllm raises an actionable ImportError rather than NameError (user-visible)."""
    monkeypatch.delattr(vllm_utils, "vllm_version", raising = False)
    with pytest.raises(ImportError, match = "vLLM is required"):
        vllm_utils.load_vllm(config = object())

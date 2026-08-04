# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for the import-time hf_transfer default.

``unsloth_zoo/__init__.py`` turns ``HF_HUB_ENABLE_HF_TRANSFER`` on so downloads
take the Rust transfer path. On Windows on ARM that path cannot complete a
download at all: every fetch dies with "an error occurred while downloading
using hf_transfer", and the identical fetch succeeds once it is off. Measured on
a ``windows-11-arm`` runner, where disabling it took the same GGUF load from a
500 to 4/4 passing inference checks.

``sys.platform`` cannot be faked in a child that then imports the package (the
stdlib reaches for ``_winapi``), which is why the sibling Windows branch in
``test_alloc_conf_platform_matrix`` is covered by source inspection rather than
by a real import. These tests go one step further than a string match: the
detector, its assignment and the enablement block are extracted by AST and
**executed** against a fake platform, with a fake ``ctypes`` swapped into
``sys.modules`` so the IsWow64Process2 branch can be driven from Linux.
"""

from __future__ import annotations

import ast
import os
import sys
import types
from unittest import mock

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INIT = os.path.join(_REPO_ROOT, "unsloth_zoo", "__init__.py")
_SOURCE = open(_INIT).read()
_TREE = ast.parse(_SOURCE)


def _statements():
    """The detector, the ``_windows_on_arm`` assignment and the enablement ``if``."""
    detect = assign = enable = None
    for node in _TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_detect_windows_on_arm":
            detect = node
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_windows_on_arm"
                for t in node.targets
            )
        ):
            assign = node
        if isinstance(node, ast.If) and "HF_HUB_ENABLE_HF_TRANSFER" in ast.dump(node):
            enable = node
    assert detect is not None, "_detect_windows_on_arm not found in __init__.py"
    assert assign is not None, "_windows_on_arm assignment not found in __init__.py"
    assert enable is not None, "the hf_transfer enablement block not found"
    return detect, assign, enable


def _fake_ctypes(*, native_machine = None, call_fails = False, api_absent = False):
    """Stand in for ctypes so the IsWow64Process2 path can be driven on Linux."""
    mod = types.ModuleType("ctypes")

    class _UShort:
        def __init__(self, value = 0):
            self.value = value

    def _is_wow64_process2(_handle, _process, native):
        if call_fails:
            return 0
        native.value = native_machine
        return 1

    kernel32 = types.SimpleNamespace(GetCurrentProcess = lambda: 1)
    if not api_absent:
        kernel32.IsWow64Process2 = _is_wow64_process2

    mod.c_ushort = _UShort
    mod.byref = lambda obj: obj
    mod.windll = types.SimpleNamespace(kernel32 = kernel32)
    return mod


def _resolve(*, platform_name, machine, environ, offline = False, ctypes_module = None):
    """Execute the three statements against a fake platform and return the env."""
    detect, assign, enable = _statements()
    env = dict(environ)
    namespace = {
        "sys": types.SimpleNamespace(platform = platform_name),
        "platform": types.SimpleNamespace(machine = lambda: machine),
        "os": types.SimpleNamespace(environ = env),
        "_offline_env": offline,
    }
    body = [detect, assign, enable]
    code = compile(ast.Module(body = body, type_ignores = []), _INIT, "exec")
    # The detector does `import ctypes` at call time, so swapping sys.modules is
    # what lets a Linux box drive the Windows branch.
    patched = {"ctypes": ctypes_module} if ctypes_module is not None else {}
    with mock.patch.dict(sys.modules, patched):
        if ctypes_module is None:
            sys.modules.pop("ctypes", None)
        exec(code, namespace)
    return env


class TestWindowsOnArm:
    def test_native_arm64_python_leaves_hf_transfer_off(self):
        env = _resolve(platform_name = "win32", machine = "ARM64", environ = {})
        assert "HF_HUB_ENABLE_HF_TRANSFER" not in env

    def test_aarch64_spelling_is_caught_too(self):
        env = _resolve(platform_name = "win32", machine = "aarch64", environ = {})
        assert "HF_HUB_ENABLE_HF_TRANSFER" not in env


class TestEmulatedX64:
    """An x64 Python emulated on ARM64 reports AMD64 on Python < 3.12, which reads
    only PROCESSOR_ARCHITECTURE/ARCHITEW6432 -- and Windows sets the latter for
    32-bit processes only, so nothing in the env names the host. This is the
    configuration anyone who installs the default python.org amd64 build on a
    Windows ARM device lands in, and machine() alone leaves it broken."""

    _ARM64 = 0xAA64  # IMAGE_FILE_MACHINE_ARM64
    _AMD64 = 0x8664  # IMAGE_FILE_MACHINE_AMD64

    def test_native_machine_arm64_is_caught_despite_machine_saying_amd64(self):
        env = _resolve(
            platform_name = "win32",
            machine = "AMD64",
            environ = {},
            ctypes_module = _fake_ctypes(native_machine = self._ARM64),
        )
        assert "HF_HUB_ENABLE_HF_TRANSFER" not in env

    def test_a_real_x64_host_still_gets_hf_transfer(self):
        env = _resolve(
            platform_name = "win32",
            machine = "AMD64",
            environ = {},
            ctypes_module = _fake_ctypes(native_machine = self._AMD64),
        )
        assert env.get("HF_HUB_ENABLE_HF_TRANSFER") == "1"

    def test_a_failed_call_falls_back_to_todays_behaviour(self):
        env = _resolve(
            platform_name = "win32",
            machine = "AMD64",
            environ = {},
            ctypes_module = _fake_ctypes(call_fails = True),
        )
        assert env.get("HF_HUB_ENABLE_HF_TRANSFER") == "1"

    def test_windows_without_the_api_does_not_raise(self):
        # IsWow64Process2 is Windows 10 1709+; older hosts must not blow up at
        # import, they just keep today's behaviour.
        env = _resolve(
            platform_name = "win32",
            machine = "AMD64",
            environ = {},
            ctypes_module = _fake_ctypes(api_absent = True),
        )
        assert env.get("HF_HUB_ENABLE_HF_TRANSFER") == "1"

    def test_no_ctypes_at_all_does_not_raise(self):
        env = _resolve(platform_name = "win32", machine = "AMD64", environ = {})
        assert env.get("HF_HUB_ENABLE_HF_TRANSFER") == "1"

    def test_a_native_arm64_host_never_reaches_the_api(self):
        # machine() already answers, so the ctypes path is not consulted; a
        # fake that would report AMD64 must not flip the verdict.
        env = _resolve(
            platform_name = "win32",
            machine = "ARM64",
            environ = {},
            ctypes_module = _fake_ctypes(native_machine = self._AMD64),
        )
        assert "HF_HUB_ENABLE_HF_TRANSFER" not in env

    def test_linux_never_touches_the_windows_api(self):
        env = _resolve(
            platform_name = "linux",
            machine = "x86_64",
            environ = {},
            ctypes_module = _fake_ctypes(native_machine = self._ARM64),
        )
        assert env.get("HF_HUB_ENABLE_HF_TRANSFER") == "1"


class TestEveryOtherPlatformIsUnchanged:
    @pytest.mark.parametrize(
        "platform_name,machine",
        [
            ("win32", "AMD64"),
            ("linux", "x86_64"),
            ("linux", "aarch64"),
            ("darwin", "arm64"),
            ("darwin", "x86_64"),
        ],
    )
    def test_hf_transfer_still_enabled(self, platform_name, machine):
        env = _resolve(platform_name = platform_name, machine = machine, environ = {})
        assert env.get("HF_HUB_ENABLE_HF_TRANSFER") == "1"

    def test_x64_windows_stays_enabled_whatever_the_env_says(self):
        # machine() is the only signal; a leftover PROCESSOR_ARCHITEW6432 must
        # not drag an x64 box into the guard.
        env = _resolve(
            platform_name = "win32",
            machine = "AMD64",
            environ = {"PROCESSOR_ARCHITEW6432": "ARM64"},
        )
        assert env.get("HF_HUB_ENABLE_HF_TRANSFER") == "1"


class TestPrecedence:
    def test_an_explicit_choice_is_never_overridden(self):
        # Someone who sets it on Windows on ARM keeps it; the guard only chooses
        # the default.
        env = _resolve(
            platform_name = "win32",
            machine = "ARM64",
            environ = {"HF_HUB_ENABLE_HF_TRANSFER": "1"},
        )
        assert env["HF_HUB_ENABLE_HF_TRANSFER"] == "1"

    def test_an_explicit_zero_survives_on_x64(self):
        env = _resolve(
            platform_name = "win32",
            machine = "AMD64",
            environ = {"HF_HUB_ENABLE_HF_TRANSFER": "0"},
        )
        assert env["HF_HUB_ENABLE_HF_TRANSFER"] == "0"

    def test_offline_still_wins(self):
        env = _resolve(
            platform_name = "linux", machine = "x86_64", environ = {}, offline = True
        )
        assert "HF_HUB_ENABLE_HF_TRANSFER" not in env


class TestWiring:
    def test_the_guard_is_actually_read_by_the_enablement_block(self):
        # Executing the statements in isolation proves the logic; this pins that
        # the block in the file is the one the guard feeds, so the two cannot
        # drift apart.
        _, _, enable = _statements()
        assert "_windows_on_arm" in ast.dump(enable.test)
        assert "_offline_env" in ast.dump(enable.test)

    def test_no_later_statement_re_enables_the_variable(self):
        # Executing two statements in isolation says nothing about the other few
        # hundred. Without this, appending one os.environ[...] = "1" anywhere
        # below leaves all of the above green while Windows on ARM is broken
        # again: a test that passes for the wrong reason.
        detect, assign, enable = _statements()
        for node in _TREE.body:
            if node in (detect, assign, enable):
                continue
            assert "HF_HUB_ENABLE_HF_TRANSFER" not in ast.dump(node), (
                f"line {node.lineno} of __init__.py also writes the variable"
            )

    def test_the_guard_is_defined_above_the_block_that_reads_it(self):
        # _statements() collects by node type, not source order, so it would
        # happily execute an assignment that really sits below its use.
        detect, assign, enable = _statements()
        assert detect.lineno < assign.lineno < enable.lineno

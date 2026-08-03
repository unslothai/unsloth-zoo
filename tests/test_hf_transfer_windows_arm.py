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
by a real import. These tests go one step further than a string match: the two
source statements are extracted by AST and **executed** against a fake platform,
so the branch logic itself is exercised on every matrix entry.
"""

from __future__ import annotations

import ast
import os
import types

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INIT = os.path.join(_REPO_ROOT, "unsloth_zoo", "__init__.py")
_SOURCE = open(_INIT).read()
_TREE = ast.parse(_SOURCE)


def _statements():
    """The ``_windows_on_arm`` assignment and the enablement ``if``, in order."""
    assign = enable = None
    for node in _TREE.body:
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
    assert assign is not None, "_windows_on_arm assignment not found in __init__.py"
    assert enable is not None, "the hf_transfer enablement block not found"
    return assign, enable


def _resolve(*, platform_name, machine, environ, offline = False):
    """Execute the two statements against a fake platform and return the env."""
    assign, enable = _statements()
    env = dict(environ)
    namespace = {
        "sys": types.SimpleNamespace(platform = platform_name),
        "platform": types.SimpleNamespace(machine = lambda: machine),
        "os": types.SimpleNamespace(environ = env),
        "_offline_env": offline,
    }
    exec(compile(ast.Module(body = [assign, enable], type_ignores = []), _INIT, "exec"), namespace)
    return env


class TestWindowsOnArm:
    def test_native_arm64_python_leaves_hf_transfer_off(self):
        env = _resolve(platform_name = "win32", machine = "ARM64", environ = {})
        assert "HF_HUB_ENABLE_HF_TRANSFER" not in env

    def test_emulated_x64_python_on_an_arm_machine_is_caught(self):
        # An x64 Python under emulation reports AMD64; the machine is still ARM,
        # and Windows advertises that through PROCESSOR_ARCHITEW6432.
        env = _resolve(
            platform_name = "win32",
            machine = "AMD64",
            environ = {"PROCESSOR_ARCHITEW6432": "ARM64"},
        )
        assert "HF_HUB_ENABLE_HF_TRANSFER" not in env

    def test_aarch64_spelling_is_caught_too(self):
        env = _resolve(platform_name = "win32", machine = "aarch64", environ = {})
        assert "HF_HUB_ENABLE_HF_TRANSFER" not in env


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

    def test_x64_windows_is_not_caught_by_a_stale_architew6432(self):
        # The var is only meaningful for an emulated process; on a real x64 box
        # it is absent, so an x64/x64 pairing must stay enabled.
        env = _resolve(
            platform_name = "win32",
            machine = "AMD64",
            environ = {"PROCESSOR_ARCHITEW6432": "AMD64"},
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
        _, enable = _statements()
        assert "_windows_on_arm" in ast.dump(enable.test)
        assert "_offline_env" in ast.dump(enable.test)

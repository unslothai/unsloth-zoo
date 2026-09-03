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

"""`install_package` must not read EOF on a terminal as consent.

`input()` raises EOFError both for a headless stdin and for Ctrl-D at a live prompt.
Only the first is an implicit ENTER; the second must keep cancelling.
"""

from __future__ import annotations

import builtins
import importlib
import os
import subprocess
import sys
import types

import pytest


llama_cpp = importlib.import_module("unsloth_zoo.llama_cpp")


class _RecordedPopen:

    def __init__(self, calls):
        self.calls = calls

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append(cmd)
        return _ContextProc(
            types.SimpleNamespace(stdout = iter(()), terminate = lambda: None)
        )


class _ContextProc:
    def __init__(self, proc):
        self.proc = proc

    def __enter__(self):
        return self.proc

    def __exit__(self, *exc):
        return False


@pytest.fixture
def installer(monkeypatch):
    calls = []
    monkeypatch.setattr(llama_cpp, "IS_WINDOWS", False)
    monkeypatch.setattr(llama_cpp, "IS_COLAB_ENVIRONMENT", False)
    monkeypatch.setattr(llama_cpp, "IS_KAGGLE_ENVIRONMENT", False)
    monkeypatch.setattr(llama_cpp.subprocess, "Popen", _RecordedPopen(calls))

    def _run(answer, isatty, auto_install = None):
        def _input(prompt = ""):
            if isinstance(answer, BaseException):
                raise answer
            return answer

        monkeypatch.setattr(builtins, "input", _input)
        monkeypatch.setattr(
            llama_cpp.sys, "stdin", types.SimpleNamespace(isatty = lambda: isatty)
        )
        if auto_install is None:
            monkeypatch.delenv("UNSLOTH_AUTO_INSTALL", raising = False)
        else:
            monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", auto_install)
        llama_cpp.install_package("cmake", system_type = "debian")
        return calls

    _run.calls = calls
    return _run


def test_eof_on_a_terminal_cancels(installer):
    with pytest.raises(RuntimeError, match = "was cancelled"):
        installer(EOFError(), isatty = True)
    assert installer.calls == [], "an interactive cancel still ran the installer"


def test_eof_without_a_terminal_accepts(installer):
    calls = installer(EOFError(), isatty = False)
    assert calls == ["apt-get install cmake -y"]


def test_eof_without_a_terminal_respects_the_opt_out(installer):
    with pytest.raises(RuntimeError, match = "UNSLOTH_AUTO_INSTALL=0"):
        installer(EOFError(), isatty = False, auto_install = "0")
    assert installer.calls == []


def test_typed_no_cancels_on_a_terminal(installer):
    with pytest.raises(RuntimeError, match = "was cancelled"):
        installer("NO", isatty = True)
    assert installer.calls == []


def test_enter_accepts(installer):
    calls = installer("", isatty = True)
    assert calls == ["apt-get install cmake -y"]


def test_a_stdin_whose_isatty_raises_is_not_treated_as_a_terminal(installer, monkeypatch):
    # An isatty() that raises must not crash out of the EOF handler.
    monkeypatch.setattr(builtins, "input", lambda prompt = "": (_ for _ in ()).throw(EOFError()))
    monkeypatch.setattr(llama_cpp, "IS_WINDOWS", False)
    monkeypatch.setattr(llama_cpp, "IS_COLAB_ENVIRONMENT", False)
    monkeypatch.setattr(llama_cpp, "IS_KAGGLE_ENVIRONMENT", False)
    monkeypatch.delenv("UNSLOTH_AUTO_INSTALL", raising = False)

    def _raises():
        raise OSError("detached")

    monkeypatch.setattr(llama_cpp.sys, "stdin", types.SimpleNamespace(isatty = _raises))
    llama_cpp.install_package("cmake", system_type = "debian")
    assert installer.calls == ["apt-get install cmake -y"]


def test_the_real_builtin_input_raises_RuntimeError_when_stdin_is_None(monkeypatch):
    """Pin CPython's real behaviour: `sys.stdin = None` raises RuntimeError, NOT
    EOFError, so a test stubbing the pair asserts a combination that cannot occur."""
    monkeypatch.setattr(llama_cpp.sys, "stdin", None)
    with pytest.raises(RuntimeError, match = "lost sys.stdin"):
        input("prompt")


def test_stdin_None_accepts_via_the_real_input(monkeypatch):
    # No `input()` stub: `sys.stdin = None` makes the real builtin raise RuntimeError.
    calls = []
    monkeypatch.setattr(llama_cpp, "IS_WINDOWS", False)
    monkeypatch.setattr(llama_cpp, "IS_COLAB_ENVIRONMENT", False)
    monkeypatch.setattr(llama_cpp, "IS_KAGGLE_ENVIRONMENT", False)
    monkeypatch.setattr(llama_cpp.subprocess, "Popen", _RecordedPopen(calls))
    monkeypatch.delenv("UNSLOTH_AUTO_INSTALL", raising = False)
    monkeypatch.setattr(llama_cpp.sys, "stdin", None)

    llama_cpp.install_package("cmake", system_type = "debian")
    assert calls == ["apt-get install cmake -y"]


def test_stdin_None_respects_the_opt_out(monkeypatch):
    calls = []
    monkeypatch.setattr(llama_cpp, "IS_WINDOWS", False)
    monkeypatch.setattr(llama_cpp, "IS_COLAB_ENVIRONMENT", False)
    monkeypatch.setattr(llama_cpp, "IS_KAGGLE_ENVIRONMENT", False)
    monkeypatch.setattr(llama_cpp.subprocess, "Popen", _RecordedPopen(calls))
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")
    monkeypatch.setattr(llama_cpp.sys, "stdin", None)

    with pytest.raises(RuntimeError, match = "UNSLOTH_AUTO_INSTALL=0"):
        llama_cpp.install_package("cmake", system_type = "debian")
    assert calls == []


def test_a_non_interactive_stdin_that_answers_still_respects_the_opt_out(installer):
    """The opt-out must be checked before the prompt: a headless stdin is not always a
    closed one (`docker run -i` feeds a newline), so input() returns and never raises."""
    with pytest.raises(RuntimeError, match = "UNSLOTH_AUTO_INSTALL=0"):
        installer("", isatty = False, auto_install = "0")
    assert installer.calls == [], "UNSLOTH_AUTO_INSTALL=0 still ran the installer"


@pytest.mark.parametrize("hosted", ["IS_COLAB_ENVIRONMENT", "IS_KAGGLE_ENVIRONMENT"])
def test_the_opt_out_applies_on_colab_and_kaggle(monkeypatch, hosted):
    # Colab and Kaggle skip the prompt block, so an opt-out nested inside never applied.
    calls = []
    monkeypatch.setattr(llama_cpp, "IS_WINDOWS", False)
    monkeypatch.setattr(llama_cpp, "IS_COLAB_ENVIRONMENT", hosted == "IS_COLAB_ENVIRONMENT")
    monkeypatch.setattr(llama_cpp, "IS_KAGGLE_ENVIRONMENT", hosted == "IS_KAGGLE_ENVIRONMENT")
    monkeypatch.setattr(llama_cpp.subprocess, "Popen", _RecordedPopen(calls))
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")

    with pytest.raises(RuntimeError, match = "UNSLOTH_AUTO_INSTALL=0"):
        llama_cpp.install_package("cmake", system_type = "debian")
    assert calls == []


@pytest.mark.parametrize("hosted", ["IS_COLAB_ENVIRONMENT", "IS_KAGGLE_ENVIRONMENT"])
def test_colab_and_kaggle_still_install_without_the_opt_out(monkeypatch, hosted):
    # Control: hoisting the check must not turn the hosted no-prompt path into refusal.
    calls = []
    monkeypatch.setattr(llama_cpp, "IS_WINDOWS", False)
    monkeypatch.setattr(llama_cpp, "IS_COLAB_ENVIRONMENT", hosted == "IS_COLAB_ENVIRONMENT")
    monkeypatch.setattr(llama_cpp, "IS_KAGGLE_ENVIRONMENT", hosted == "IS_KAGGLE_ENVIRONMENT")
    monkeypatch.setattr(llama_cpp.subprocess, "Popen", _RecordedPopen(calls))
    monkeypatch.delenv("UNSLOTH_AUTO_INSTALL", raising = False)

    llama_cpp.install_package("cmake", system_type = "debian")
    assert calls == ["apt-get install cmake -y"]


def test_the_opt_out_covers_the_windows_branch(monkeypatch):
    # The Windows arm returns from inside its own loop, so an opt-out placed after the
    # platform branch never reached winget.
    calls = []
    monkeypatch.setattr(llama_cpp, "IS_WINDOWS", True)
    monkeypatch.setattr(llama_cpp, "IS_COLAB_ENVIRONMENT", False)
    monkeypatch.setattr(llama_cpp, "IS_KAGGLE_ENVIRONMENT", False)
    monkeypatch.setattr(
        llama_cpp.subprocess, "run", lambda *a, **k: calls.append(a) or None
    )
    monkeypatch.setattr(llama_cpp.shutil, "which", lambda name: "C:\\winget.exe")
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")

    with pytest.raises(RuntimeError, match = "UNSLOTH_AUTO_INSTALL=0"):
        llama_cpp.install_package("cmake")
    assert calls == [], "winget ran with the opt-out set"


def test_windows_still_installs_without_the_opt_out(monkeypatch):
    # Control: hoisting the check must not turn Windows into a refusal.
    calls = []

    class _Result:
        returncode = 0

    monkeypatch.setattr(llama_cpp, "IS_WINDOWS", True)
    monkeypatch.setattr(llama_cpp, "IS_COLAB_ENVIRONMENT", False)
    monkeypatch.setattr(llama_cpp, "IS_KAGGLE_ENVIRONMENT", False)
    monkeypatch.setattr(
        llama_cpp.subprocess, "run", lambda cmd, **k: calls.append(cmd) or _Result()
    )
    monkeypatch.setattr(llama_cpp.shutil, "which", lambda name: "C:\\winget.exe")
    monkeypatch.delenv("UNSLOTH_AUTO_INSTALL", raising = False)

    llama_cpp.install_package("cmake")
    assert calls and calls[0][:2] == ["winget", "install"]


def test_an_unrelated_RuntimeError_is_not_read_as_consent(installer, monkeypatch):
    # RuntimeError for a lost stdout/stderr with a live stdin is not a missing prompt.
    with pytest.raises(RuntimeError, match = "lost sys.stdout"):
        installer(RuntimeError("input(): lost sys.stdout"), isatty = False)
    assert installer.calls == []


@pytest.mark.skipif(os.name != "posix", reason = "fd 0 is closed with a POSIX shell redirect")
def test_a_closed_fd0_really_produces_a_None_stdin():
    """The unstubbed precondition: fd 0 closed gives `sys.stdin is None`, and the real
    `input()` then raises RuntimeError, not EOFError."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    probe = (
        "import sys\n"
        "print(sys.stdin is None)\n"
        "try:\n"
        "    input('p')\n"
        "except BaseException as e:\n"
        "    print(type(e).__name__)\n"
    )
    r = subprocess.run(
        ["/bin/sh", "-c", 'exec "$0" -c "$1" 0<&-', sys.executable, probe],
        capture_output = True, text = True, env = env, timeout = 120,
    )
    assert r.stdout.split() == ["True", "RuntimeError"], r.stdout + r.stderr

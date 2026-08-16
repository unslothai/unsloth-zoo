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

`input()` raises EOFError both for a headless stdin (Docker without a TTY,
a CI job) and for a user pressing Ctrl-D at a live prompt. Only the first is
an implicit ENTER; the second is the documented way to back out, so it has to
keep cancelling instead of authorising `apt-get` / `yum` / `pacman`.

Nothing here can reach a package manager: `subprocess.Popen` is replaced for
every test, and the accept path asserts against the recorded command only.
"""

from __future__ import annotations

import builtins
import importlib
import types

import pytest


llama_cpp = importlib.import_module("unsloth_zoo.llama_cpp")


class _RecordedPopen:
    """Stand-in for the installer subprocess: records, runs nothing."""

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
    """Call install_package with a stubbed prompt, stdin and subprocess."""
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
    # Ctrl-D at a live prompt. Anything but a cancel authorises apt-get.
    with pytest.raises(RuntimeError, match = "was cancelled"):
        installer(EOFError(), isatty = True)
    assert installer.calls == [], "an interactive cancel still ran the installer"


def test_eof_without_a_terminal_accepts(installer):
    # Docker without a TTY / headless CI: the implicit ENTER this is for.
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


def test_a_stdin_without_isatty_is_not_treated_as_a_terminal(installer, monkeypatch):
    # pythonw / a detached kernel can leave sys.stdin as None or as an object
    # whose isatty() raises; neither may crash out of the EOF handler.
    monkeypatch.setattr(builtins, "input", lambda prompt = "": (_ for _ in ()).throw(EOFError()))
    monkeypatch.setattr(llama_cpp, "IS_WINDOWS", False)
    monkeypatch.setattr(llama_cpp, "IS_COLAB_ENVIRONMENT", False)
    monkeypatch.setattr(llama_cpp, "IS_KAGGLE_ENVIRONMENT", False)
    monkeypatch.delenv("UNSLOTH_AUTO_INSTALL", raising = False)

    def _raises():
        raise OSError("detached")

    for stdin in (None, types.SimpleNamespace(isatty = _raises)):
        installer.calls.clear()
        monkeypatch.setattr(llama_cpp.sys, "stdin", stdin)
        llama_cpp.install_package("cmake", system_type = "debian")
        assert installer.calls == ["apt-get install cmake -y"]

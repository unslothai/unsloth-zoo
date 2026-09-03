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
import os
import subprocess
import sys
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


def test_a_stdin_whose_isatty_raises_is_not_treated_as_a_terminal(installer, monkeypatch):
    # A detached kernel can leave an object whose isatty() raises; that must
    # not crash out of the EOF handler.
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
    """Pin the CPython behaviour the handler is written against.

    `builtin_input_impl` null-checks `sys.stdin` before reading and raises
    RuntimeError("input(): lost sys.stdin"), NOT EOFError. A test that stubs
    `input()` to raise EOFError while setting `sys.stdin = None` asserts a
    combination that cannot occur, so the real path has to be pinned here.
    """
    monkeypatch.setattr(llama_cpp.sys, "stdin", None)
    with pytest.raises(RuntimeError, match = "lost sys.stdin"):
        input("prompt")


def test_stdin_None_accepts_via_the_real_input(monkeypatch):
    # No `input()` stub: `sys.stdin = None` makes the genuine builtin raise
    # RuntimeError, which is exactly what a process launched with fd 0 closed
    # hits. The installer must still run rather than abort.
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
    """The opt-out has to be checked before the prompt, not inside its handler.

    A headless stdin is not always a closed one: `docker run -i` without `-t`,
    a `yes ""` pipe or a here-doc all feed a newline, so `input()` RETURNS and
    never raises. Checked only from the EOF/RuntimeError handler, the opt-out
    was then skipped and the package manager ran anyway.
    """
    with pytest.raises(RuntimeError, match = "UNSLOTH_AUTO_INSTALL=0"):
        installer("", isatty = False, auto_install = "0")
    assert installer.calls == [], "UNSLOTH_AUTO_INSTALL=0 still ran the installer"


@pytest.mark.parametrize("hosted", ["IS_COLAB_ENVIRONMENT", "IS_KAGGLE_ENVIRONMENT"])
def test_the_opt_out_applies_on_colab_and_kaggle(monkeypatch, hosted):
    # Colab and Kaggle skip the prompt block entirely, so an opt-out nested
    # inside it never applied on the two platforms these notebooks run on.
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
    # Control for the pair above: hoisting the check must not turn the hosted
    # no-prompt path into a refusal for everyone else.
    calls = []
    monkeypatch.setattr(llama_cpp, "IS_WINDOWS", False)
    monkeypatch.setattr(llama_cpp, "IS_COLAB_ENVIRONMENT", hosted == "IS_COLAB_ENVIRONMENT")
    monkeypatch.setattr(llama_cpp, "IS_KAGGLE_ENVIRONMENT", hosted == "IS_KAGGLE_ENVIRONMENT")
    monkeypatch.setattr(llama_cpp.subprocess, "Popen", _RecordedPopen(calls))
    monkeypatch.delenv("UNSLOTH_AUTO_INSTALL", raising = False)

    llama_cpp.install_package("cmake", system_type = "debian")
    assert calls == ["apt-get install cmake -y"]


def test_an_unrelated_RuntimeError_is_not_read_as_consent(installer, monkeypatch):
    # `input()` raises RuntimeError for a lost sys.stdout/sys.stderr too. With
    # a live stdin that is not a missing prompt, so it must propagate rather
    # than authorise a package manager command.
    with pytest.raises(RuntimeError, match = "lost sys.stdout"):
        installer(RuntimeError("input(): lost sys.stdout"), isatty = False)
    assert installer.calls == []


@pytest.mark.skipif(os.name != "posix", reason = "fd 0 is closed with a POSIX shell redirect")
def test_a_closed_fd0_really_produces_a_None_stdin():
    """The environmental precondition behind the tests above, unstubbed.

    A process started with fd 0 closed gets `sys.stdin is None`, and the real
    builtin `input()` then raises RuntimeError rather than EOFError. The child
    is pinned to THIS checkout via PYTHONPATH so it cannot answer from an
    installed copy of unsloth_zoo (it imports nothing from it here, but the
    pin keeps that true if the probe ever grows).
    """
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

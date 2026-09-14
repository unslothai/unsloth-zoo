# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Cross-platform and precedence properties of the llama.cpp install consent.

Two things this file exists to rule out.

**Inverted precedence.** "EOF with no terminal means consent" must never
outrank an explicit `UNSLOTH_AUTO_INSTALL=0`. If it did, the opt-out would be
useless in exactly the headless context it was written for.

**Platform-shaped stdin.** The consent decision reads `sys.stdin`, which differs
by platform in ways Linux alone does not exercise:

- Windows `pythonw.exe` runs with no console at all and `sys.stdin is None`.
  CPython raises `RuntimeError`, not `EOFError`, for that.
- `os.environ` is case-insensitive on Windows, so a user setting
  `unsloth_auto_install=0` gets a working opt-out there and not here. The test
  below pins that we read the canonical spelling, which is what `os.environ`
  normalises to on Windows.
- macOS behaves as Linux for all of these; it is covered by the same paths.

Everything here is simulated, so it runs on any host. Real Windows and macOS
coverage comes from the staging cross-platform CI run, not from this file.
"""

from __future__ import annotations

import builtins
import subprocess
import sys
import types

import pytest

import unsloth_zoo.llama_cpp as llama_cpp


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append(cmd)

        class _Proc:
            stdout = iter(())
            def terminate(self): pass

        class _Ctx:
            def __enter__(self_inner): return _Proc()
            def __exit__(self_inner, *exc): return False

        return _Ctx()


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(llama_cpp, "IS_WINDOWS", False)
    monkeypatch.setattr(llama_cpp, "IS_COLAB_ENVIRONMENT", False)
    monkeypatch.setattr(llama_cpp, "IS_KAGGLE_ENVIRONMENT", False)
    monkeypatch.setattr(llama_cpp.subprocess, "Popen", rec)
    monkeypatch.delenv("UNSLOTH_AUTO_INSTALL", raising = False)
    return rec


# ------------------------------------------------------------- precedence

@pytest.mark.parametrize("falsy", ["0", "no", "off", "false", "", "  "])
def test_explicit_opt_out_beats_eof_implies_consent(recorder, monkeypatch, falsy):
    """The decisive precedence test. A headless process hits BOTH conditions at
    once: stdin raises EOFError, and the user has said no. The user wins."""
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", falsy)
    monkeypatch.setattr(
        builtins, "input",
        lambda prompt = "": (_ for _ in ()).throw(EOFError()),
    )
    monkeypatch.setattr(
        llama_cpp.sys, "stdin", types.SimpleNamespace(isatty = lambda: False),
    )
    with pytest.raises(RuntimeError, match = "UNSLOTH_AUTO_INSTALL=0"):
        llama_cpp.install_package("cmake", system_type = "debian")
    assert recorder.calls == [], f"opt-out {falsy!r} was overridden by EOF consent"


@pytest.mark.parametrize("truthy", ["1", "on", "TRUE", "yes", " Yes "])
def test_truthy_spellings_keep_the_installer_enabled(recorder, monkeypatch, truthy):
    """The control. Only a falsy value opts out; the default stays enabled."""
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", truthy)
    monkeypatch.setattr(
        builtins, "input",
        lambda prompt = "": (_ for _ in ()).throw(EOFError()),
    )
    monkeypatch.setattr(
        llama_cpp.sys, "stdin", types.SimpleNamespace(isatty = lambda: False),
    )
    llama_cpp.install_package("cmake", system_type = "debian")
    assert recorder.calls == ["apt-get install cmake -y"]


def test_the_opt_out_is_read_at_the_attempt_not_at_import(recorder, monkeypatch):
    """A user who sees the prompt and sets the variable afterwards must be
    obeyed without re-importing."""
    monkeypatch.setattr(
        builtins, "input",
        lambda prompt = "": (_ for _ in ()).throw(EOFError()),
    )
    monkeypatch.setattr(
        llama_cpp.sys, "stdin", types.SimpleNamespace(isatty = lambda: False),
    )
    llama_cpp.install_package("cmake", system_type = "debian")
    assert len(recorder.calls) == 1
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")
    with pytest.raises(RuntimeError, match = "UNSLOTH_AUTO_INSTALL=0"):
        llama_cpp.install_package("cmake", system_type = "debian")
    assert len(recorder.calls) == 1, "the later opt-out was ignored"


# ------------------------------------------------------------- platform shapes

def test_windows_pythonw_has_no_stdin_and_is_treated_as_consent(recorder, monkeypatch):
    """`pythonw.exe` runs without a console: sys.stdin is None and CPython raises
    RuntimeError rather than EOFError. Nobody is there to answer, so this is the
    same implicit ENTER the prompt documents."""
    monkeypatch.setattr(llama_cpp.sys, "stdin", None)
    llama_cpp.install_package("cmake", system_type = "debian")
    assert recorder.calls == ["apt-get install cmake -y"]


def test_windows_pythonw_still_respects_the_opt_out(recorder, monkeypatch):
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")
    monkeypatch.setattr(llama_cpp.sys, "stdin", None)
    with pytest.raises(RuntimeError, match = "UNSLOTH_AUTO_INSTALL=0"):
        llama_cpp.install_package("cmake", system_type = "debian")
    assert recorder.calls == []


def test_the_canonical_env_spelling_is_what_is_read(monkeypatch):
    """os.environ is case-insensitive on Windows and normalises to upper case, so
    reading the upper-case name is what makes a lower-case setting work there.
    Pinning the spelling keeps that true."""
    import inspect
    source = inspect.getsource(llama_cpp._auto_install_enabled)
    assert '"UNSLOTH_AUTO_INSTALL"' in source
    assert ".upper()" in source, "value comparison must be case-insensitive too"


def test_a_detached_stdin_whose_isatty_raises_is_not_a_terminal(recorder, monkeypatch):
    """Seen with a closed pseudo-terminal and with some notebook kernels."""
    def _boom():
        raise OSError("detached")

    monkeypatch.setattr(
        builtins, "input",
        lambda prompt = "": (_ for _ in ()).throw(EOFError()),
    )
    monkeypatch.setattr(
        llama_cpp.sys, "stdin", types.SimpleNamespace(isatty = _boom),
    )
    llama_cpp.install_package("cmake", system_type = "debian")
    assert recorder.calls == ["apt-get install cmake -y"]


# ------------------------------------------------------------- hosted paths

@pytest.mark.parametrize("system_type", ["debian", "rpm", "arch"])
def test_the_opt_out_covers_every_package_manager(monkeypatch, system_type):
    """The check sits above the platform branch, so it must hold for all of them,
    not only the apt path the other tests use."""
    rec = _Recorder()
    monkeypatch.setattr(llama_cpp, "IS_WINDOWS", False)
    monkeypatch.setattr(llama_cpp, "IS_COLAB_ENVIRONMENT", False)
    monkeypatch.setattr(llama_cpp, "IS_KAGGLE_ENVIRONMENT", False)
    monkeypatch.setattr(llama_cpp.subprocess, "Popen", rec)
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")
    with pytest.raises(RuntimeError, match = "UNSLOTH_AUTO_INSTALL=0"):
        llama_cpp.install_package("cmake", system_type = system_type)
    assert rec.calls == []

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
import errno
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


# ------------------------------------------------- stdin closed after startup

@pytest.mark.parametrize(
    "exc, why",
    [
        (ValueError("I/O operation on closed file."), "sys.stdin.close()"),
        (OSError(9, "Bad file descriptor"), "os.close(0)"),
    ],
)
def test_a_stdin_closed_after_startup_is_treated_as_no_one_there(
    recorder, monkeypatch, exc, why,
):
    """input() raises neither EOFError nor RuntimeError when stdin is closed after the
    interpreter started, which is what daemon and process wrappers do to fd 0.

    Measured on CPython 3.13: sys.stdin.close() gives ValueError, os.close(0) gives
    OSError(EBADF). Both used to propagate straight through install_package, so a
    headless export still died on the prompt this branch exists to survive."""
    monkeypatch.setattr(builtins, "input", lambda prompt = "": (_ for _ in ()).throw(exc))
    # A closed sys.stdin makes isatty() raise; a closed fd 0 makes it return False.
    monkeypatch.setattr(
        llama_cpp.sys, "stdin",
        types.SimpleNamespace(isatty = lambda: False, closed = True),
    )
    llama_cpp.install_package("cmake", system_type = "debian")
    assert recorder.calls == ["apt-get install cmake -y"], why


@pytest.mark.parametrize(
    "exc", [ValueError("I/O operation on closed file."), OSError(9, "Bad file descriptor")],
)
def test_a_closed_stdin_still_respects_the_opt_out(recorder, monkeypatch, exc):
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")
    monkeypatch.setattr(builtins, "input", lambda prompt = "": (_ for _ in ()).throw(exc))
    monkeypatch.setattr(
        llama_cpp.sys, "stdin",
        types.SimpleNamespace(isatty = lambda: False, closed = True),
    )
    with pytest.raises(RuntimeError, match = "UNSLOTH_AUTO_INSTALL=0"):
        llama_cpp.install_package("cmake", system_type = "debian")
    assert recorder.calls == []


@pytest.mark.parametrize(
    "exc", [ValueError("I/O operation on closed file."), OSError(9, "Bad file descriptor")],
)
def test_a_closed_stdin_on_a_terminal_still_cancels(recorder, monkeypatch, exc):
    """The control that keeps Ctrl-D meaningful. If something claims to be a terminal,
    a failed read is not evidence that nobody was asked."""
    monkeypatch.setattr(builtins, "input", lambda prompt = "": (_ for _ in ()).throw(exc))
    monkeypatch.setattr(
        llama_cpp.sys, "stdin",
        types.SimpleNamespace(isatty = lambda: True, closed = True),
    )
    with pytest.raises(RuntimeError, match = "was cancelled"):
        llama_cpp.install_package("cmake", system_type = "debian")
    assert recorder.calls == []


def test_the_real_builtin_raises_ValueError_on_a_closed_stdin():
    """Pin the CPython behaviour the handler now depends on, rather than trusting it."""
    import subprocess as _sp
    # input() writes its prompt to stdout before failing, so mark the answer.
    probe = (
        "import sys\n"
        "sys.stdin.close()\n"
        "try:\n"
        "    input('')\n"
        "except BaseException as e:\n"
        "    print('RESULT=' + type(e).__name__)\n"
    )
    r = _sp.run([sys.executable, "-c", probe], capture_output = True, text = True, timeout = 120)
    assert "RESULT=ValueError" in r.stdout, r.stdout + r.stderr


# ------------------------------------------------- narrowness of the new arms

def test_an_unrelated_OSError_on_a_live_stdin_is_not_consent(recorder, monkeypatch):
    """EIO from a serial console is an I/O failure, not an empty answer. Only EBADF
    establishes that the descriptor is gone."""
    exc = OSError(errno.EIO, "Input/output error")
    monkeypatch.setattr(builtins, "input", lambda prompt = "": (_ for _ in ()).throw(exc))
    monkeypatch.setattr(
        llama_cpp.sys, "stdin",
        types.SimpleNamespace(isatty = lambda: False, closed = False),
    )
    with pytest.raises(OSError):
        llama_cpp.install_package("cmake", system_type = "debian")
    assert recorder.calls == [], "an unrelated I/O error ran the installer"


def test_a_ValueError_from_a_live_stdin_wrapper_is_not_consent(recorder, monkeypatch):
    """A wrapper raising ValueError while still open is not a closed stream."""
    monkeypatch.setattr(
        builtins, "input",
        lambda prompt = "": (_ for _ in ()).throw(ValueError("wrapper blew up")),
    )
    monkeypatch.setattr(
        llama_cpp.sys, "stdin",
        types.SimpleNamespace(isatty = lambda: False, closed = False),
    )
    with pytest.raises(ValueError):
        llama_cpp.install_package("cmake", system_type = "debian")
    assert recorder.calls == []


# ------------------------------------------------- the opt-out covers the whole install

def test_the_opt_out_blocks_the_prebuilt_download_and_the_privileged_update(monkeypatch):
    """install_llama_cpp fetches a prebuilt binary, and probes for elevation by RUNNING
    `apt-get update`, both before install_package is ever reached. Checking the flag only
    inside install_package left an explicit refusal installing llama.cpp anyway."""
    calls = []
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")
    monkeypatch.setattr(
        llama_cpp, "_maybe_install_llama_cpp_prebuilt",
        lambda *a, **k: calls.append("prebuilt"),
    )
    monkeypatch.setattr(
        llama_cpp, "do_we_need_sudo", lambda *a, **k: calls.append("elevate") or False,
    )
    monkeypatch.setattr(llama_cpp, "check_build_requirements", lambda *a, **k: ([], "debian"))

    with pytest.raises(RuntimeError, match = "UNSLOTH_AUTO_INSTALL=0"):
        llama_cpp.install_llama_cpp(llama_cpp_folder = "/nonexistent-unsloth-test-path")

    assert calls == [], "the opt-out was bypassed: %s" % calls


def test_an_existing_install_is_unaffected_by_the_opt_out(monkeypatch, tmp_path):
    """The gate must only fire when something would actually be installed."""
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")
    monkeypatch.setattr(
        llama_cpp, "check_llama_cpp",
        lambda *a, **k: (str(tmp_path / "quantize"), str(tmp_path / "convert")),
    )
    # Nothing to build or clone, so no refusal: the flag governs installation, not use.
    import inspect as _inspect
    src = _inspect.getsource(llama_cpp.install_llama_cpp)
    assert "(needs_build or needs_clone) and not _auto_install_enabled()" in src


def test_a_refusal_does_not_wipe_a_corrupted_folder(monkeypatch, tmp_path):
    """The corrupted-checkout branch used to rmtree before the opt-out was consulted, so
    an explicit refusal still destroyed the folder and only then reported the refusal.
    llama_cpp_folder can be a custom path holding the user's own files."""
    folder = tmp_path / "llama.cpp"
    folder.mkdir()
    # No src/ggml/common and no prebuilt marker, so it reads as corrupted.
    keep = folder / "my_own_notes.txt"
    keep.write_text("do not delete me")

    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")
    with pytest.raises(RuntimeError, match = "UNSLOTH_AUTO_INSTALL=0"):
        llama_cpp.install_llama_cpp(llama_cpp_folder = str(folder))

    assert folder.is_dir(), "the folder was deleted despite the refusal"
    assert keep.read_text() == "do not delete me", "user files were destroyed"


def test_a_corrupted_folder_is_still_wiped_when_installing_is_allowed(monkeypatch, tmp_path):
    """Deferring the deletion must not stop it happening on the normal path."""
    folder = tmp_path / "llama.cpp"
    folder.mkdir()
    (folder / "junk.txt").write_text("stale")

    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "1")
    seen = {}

    def _stop_here(*args, **kwargs):
        seen["reached_install"] = True
        raise RuntimeError("stop after the wipe")

    monkeypatch.setattr(llama_cpp, "_maybe_install_llama_cpp_prebuilt", _stop_here)
    # tmp_path is neither under ~/.unsloth nor ./llama.cpp, so the real guard would
    # refuse. That guard is unrelated to this test and is covered on its own.
    monkeypatch.setattr(llama_cpp, "_is_safe_to_delete", lambda _path: True)

    with pytest.raises(RuntimeError, match = "stop after the wipe"):
        llama_cpp.install_llama_cpp(llama_cpp_folder = str(folder))

    assert seen.get("reached_install"), "never got past the wipe"
    assert not folder.exists(), "the corrupted folder was not re-cloned from scratch"


def test_the_converter_repair_honours_the_opt_out(monkeypatch):
    """An existing working checkout returns out of install_llama_cpp before its gate, so
    the converter self-heal was the one remaining way a refusal still reached pip. It runs
    `pip install --upgrade --force-reinstall`, which mutates the environment as much as
    any install does."""
    import inspect as _inspect
    src = _inspect.getsource(llama_cpp)
    call = src.index("_reinstall_converter_deps(command[0]")
    guard = src.rindex("_auto_install_enabled()", 0, call)
    between = src[guard:call]
    assert "_looks_like_converter_dep_error" in between, (
        "the opt-out is not checked on the path that reaches the repair"
    )


def test_the_repair_still_runs_when_installing_is_allowed(monkeypatch):
    """The gate must not disable the self-heal for everyone else."""
    monkeypatch.delenv("UNSLOTH_AUTO_INSTALL", raising = False)
    assert llama_cpp._auto_install_enabled() is True
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")
    assert llama_cpp._auto_install_enabled() is False


# ------------------------------------------- EBADF must belong to stdin, not to stdout

def test_ebadf_from_a_closed_stdout_is_not_consent(recorder, monkeypatch):
    """input() writes the prompt before it reads, so a closed fd 1 raises EBADF while fd 0
    is open, non-tty and holding an unread answer. Reproduced on CPython: `python -u` with
    fd 1 closed gives OSError(9) with sys.stdin.closed and isatty() both False. Accepting
    that as an implicit ENTER installs packages while the real answer went unread."""
    monkeypatch.setattr(
        builtins, "input",
        lambda prompt = "": (_ for _ in ()).throw(OSError(errno.EBADF, "Bad file descriptor")),
    )
    # A live stdin: not closed, not a tty, and with a real descriptor behind it.
    monkeypatch.setattr(
        llama_cpp.sys, "stdin",
        types.SimpleNamespace(isatty = lambda: False, closed = False, fileno = lambda: 0),
    )
    with pytest.raises(OSError):
        llama_cpp.install_package("cmake", system_type = "debian")
    assert recorder.calls == [], "consented while stdin still had an answer waiting"


def test_ebadf_from_a_genuinely_dead_stdin_is_still_consent(recorder, monkeypatch):
    """The case the branch exists for must keep working: fd 0 really is gone."""
    monkeypatch.setattr(
        builtins, "input",
        lambda prompt = "": (_ for _ in ()).throw(OSError(errno.EBADF, "Bad file descriptor")),
    )
    def _dead_fileno():
        raise OSError(errno.EBADF, "Bad file descriptor")
    monkeypatch.setattr(
        llama_cpp.sys, "stdin",
        types.SimpleNamespace(isatty = lambda: False, closed = True, fileno = _dead_fileno),
    )
    llama_cpp.install_package("cmake", system_type = "debian")
    assert recorder.calls == ["apt-get install cmake -y"]


def test_stdin_is_usable_handles_every_broken_shape(monkeypatch):
    for stdin in (
        None,
        types.SimpleNamespace(),                                  # no fileno at all
        types.SimpleNamespace(fileno = lambda: (_ for _ in ()).throw(ValueError("closed"))),
        types.SimpleNamespace(fileno = lambda: -1),
    ):
        monkeypatch.setattr(llama_cpp.sys, "stdin", stdin)
        assert llama_cpp._stdin_is_usable() is False, stdin

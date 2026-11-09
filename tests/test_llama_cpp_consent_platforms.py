# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Install consent: EOF with no terminal (implied consent) must never outrank UNSLOTH_AUTO_INSTALL=0; covers sys.stdin
None under pythonw.exe and Windows env case; simulated, real Windows/macOS in staging CI."""

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
    """Headless run with EOFError on stdin and an explicit opt-out: the user's no wins over implied consent."""
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
    """The opt-out is read at the attempt, so a variable set after the prompt is obeyed without re-importing the module."""
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
    """pythonw.exe has no console: sys.stdin is None and CPython raises RuntimeError, not EOFError; nobody can answer, so it counts as the implicit ENTER consent."""
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
    """os.environ is case-insensitive on Windows and upper-cases names, so reading the upper-case UNSLOTH_AUTO_INSTALL keeps lower-case settings working there."""
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
    """The opt-out check sits above the platform branch, so it must hold for every package manager, not only the apt path."""
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
    """A stdin closed after startup, as daemons do to fd 0: CPython 3.13 input() raises ValueError after sys.stdin.close() and OSError(EBADF) after os.close(0)."""
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
    """Control for Ctrl-D: when stdin claims to be a terminal, a failed read is not evidence that nobody was asked, so it still cancels."""
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
    """EIO from a serial console is an I/O failure, not an empty answer; only EBADF shows the descriptor is gone."""
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
    """install_llama_cpp fetches a prebuilt and probes elevation via apt-get update before install_package runs, so the opt-out must be checked there too."""
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
    """A refused install must not rmtree a corrupted llama_cpp_folder first, since a custom path can hold the user's own files."""
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
    # tmp_path is outside ~/.unsloth and ./llama.cpp, so the real _is_safe_to_delete guard would refuse; it is stubbed here and covered separately.
    monkeypatch.setattr(llama_cpp, "_is_safe_to_delete", lambda _path: True)

    with pytest.raises(RuntimeError, match = "stop after the wipe"):
        llama_cpp.install_llama_cpp(llama_cpp_folder = str(folder))

    assert seen.get("reached_install"), "never got past the wipe"
    assert not folder.exists(), "the corrupted folder was not re-cloned from scratch"


def test_the_converter_repair_honours_the_opt_out(monkeypatch):
    """A working checkout returns before the install gate, so the pip install --upgrade --force-reinstall converter repair must honour the opt-out itself."""
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
    """input() writes the prompt first, so a closed fd 1 raises EBADF (OSError 9) while fd 0 still holds an unread answer; that is not implicit ENTER consent."""
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

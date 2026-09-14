# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Safety properties of the notebook dependency auto-installer.

These cover the ways a default-on auto-installer can hurt a user who never asked
for it, rather than the ways it helps one who did. Each test names the concrete
failure it prevents.

The headline one is the resolver constraint. `timm` is the package this feature
is most likely to install (the TimmWrapper path in Gemma3N and Qwen3-VL), timm
requires torchvision, and torchvision pins an exact torch. Measured against a
simulated `torch==2.13.0+cu130` install, an unconstrained `pip install timm`
plans `torch-2.14.0` plus a fresh CUDA stack, i.e. it uninstalls the user's CUDA
torch in the middle of a live session. With the constraints file pip selects
`torchvision-0.28.0` instead and leaves torch alone.

All CPU only, no real installs: `subprocess.run` is stubbed throughout.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
import threading
import time
import types

import pytest

nd = importlib.import_module("unsloth_zoo.temporary_patches.notebook_deps")


@pytest.fixture(autouse = True)
def _reset(monkeypatch):
    nd._attempted.clear()
    monkeypatch.setattr(nd, "_constraints_path", None)
    monkeypatch.delenv("UNSLOTH_AUTO_INSTALL", raising = False)
    monkeypatch.delenv("UNSLOTH_NOTEBOOK_DEPS_NO_AUTORUN", raising = False)
    yield
    nd._attempted.clear()
    monkeypatch.setattr(nd, "_constraints_path", None)


class _Recorder:
    """Stands in for subprocess.run and records the argv it was handed."""

    def __init__(self, returncode = 0, stderr = ""):
        self.calls = []
        self.returncode = returncode
        self.stderr = stderr

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append(list(cmd))
        return types.SimpleNamespace(
            returncode = self.returncode, stdout = "", stderr = self.stderr,
        )


# --------------------------------------------------------------- constraints

def test_pip_command_carries_a_constraints_file(monkeypatch):
    cmd = nd._pip_command("timm")
    assert "-c" in cmd, "no constraints file: pip is free to replace torch"
    assert cmd[cmd.index("-c") + 1].endswith(".txt")
    assert cmd[-1] == "timm", "the package must stay the last argument"


def test_uv_command_carries_a_constraints_file(monkeypatch):
    cmd = nd._uv_command("timm")
    assert "-c" in cmd
    assert cmd[-1] == "timm"


def test_constraints_pin_installed_critical_distributions(monkeypatch):
    """torch is the one that matters. It must be pinned to the EXACT installed
    version including the local label, so a same-public-version swap of a CUDA
    build for a PyPI build is refused too."""
    installed = {"torch": "2.13.0+cu130", "numpy": "2.4.6"}

    def _version(dist):
        if dist in installed:
            return installed[dist]
        raise nd.importlib.metadata.PackageNotFoundError(dist)

    monkeypatch.setattr(nd.importlib.metadata, "version", _version)
    body = open(nd._constraints_file()).read()
    assert "torch==2.13.0+cu130" in body
    assert "numpy==2.4.6" in body


def test_constraints_omit_distributions_that_are_not_installed(monkeypatch):
    """Pinning something absent would force it to be INSTALLED, which is the
    opposite of the intent."""
    def _version(dist):
        if dist == "torch":
            return "2.13.0"
        raise nd.importlib.metadata.PackageNotFoundError(dist)

    monkeypatch.setattr(nd.importlib.metadata, "version", _version)
    body = open(nd._constraints_file()).read()
    assert body.strip() == "torch==2.13.0"
    assert "vllm" not in body and "xformers" not in body


def test_constraints_file_is_built_once(monkeypatch):
    calls = []

    def _version(dist):
        calls.append(dist)
        raise nd.importlib.metadata.PackageNotFoundError(dist)

    monkeypatch.setattr(nd.importlib.metadata, "version", _version)
    first, second = nd._constraints_file(), nd._constraints_file()
    assert first == second
    assert len(calls) == len(nd._PINNED_DISTRIBUTIONS), "rebuilt on every install"


def test_torch_is_in_the_pinned_set():
    for critical in ("torch", "torchvision", "numpy", "triton"):
        assert critical in nd._PINNED_DISTRIBUTIONS


# --------------------------------------------------------------- the allow list

def test_pip_install_refuses_a_non_allow_listed_package(monkeypatch):
    """Defence in depth. _try_install_and_import already filters, but this is the
    last point before a name reaches a command line, and on the check_imports path
    that name comes out of a downloaded trust_remote_code file."""
    recorder = _Recorder()
    monkeypatch.setattr(subprocess, "run", recorder)
    assert nd._pip_install("evil-package") is False
    assert recorder.calls == [], "a non allow-listed name reached pip"


def test_pip_install_accepts_an_allow_listed_package(monkeypatch):
    """The control for the test above."""
    recorder = _Recorder()
    monkeypatch.setattr(subprocess, "run", recorder)
    monkeypatch.setattr(nd.shutil, "which", lambda _name: None)
    nd._pip_install("einops")
    assert len(recorder.calls) == 1
    assert recorder.calls[0][-1] == "einops"


# --------------------------------------------------------------- concurrency

def test_two_threads_racing_one_package_run_pip_once(monkeypatch):
    """Without the lock both threads pass the `pkg in _attempted` test before
    either adds to it, and two pip processes hit one prefix at once."""
    recorder = _Recorder()
    barrier = threading.Barrier(2)

    def _slow_run(cmd, *args, **kwargs):
        barrier.wait(timeout = 10)  # maximise the overlap
        return recorder(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _slow_run)
    monkeypatch.setattr(nd.shutil, "which", lambda _name: None)

    def _worker():
        try:
            nd._pip_install("einops")
        except threading.BrokenBarrierError:
            pass

    threads = [threading.Thread(target = _worker) for _ in range(2)]
    for t in threads: t.start()
    # The loser never calls subprocess, so the barrier never completes. That is
    # the proof, and the timeout below is what collects it.
    for t in threads: t.join(timeout = 15)
    assert len(recorder.calls) <= 1, f"pip ran {len(recorder.calls)} times concurrently"


# --------------------------------------------------------------- diagnostics

def test_a_timeout_is_reported_as_a_timeout(monkeypatch, caplog):
    def _timeout(cmd, *args, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 300)

    monkeypatch.setattr(subprocess, "run", _timeout)
    ok, retry = nd._run_install("einops", ["pip", "install", "einops"])
    assert ok is False and retry is False
    assert "timed out" in caplog.text.lower()
    assert "UNSLOTH_AUTO_INSTALL=0" in caplog.text, "no way out offered to the user"


def test_no_autorun_accepts_the_documented_truthy_spellings(monkeypatch):
    """A bare != "1" meant UNSLOTH_NOTEBOOK_DEPS_NO_AUTORUN=true silently did
    nothing, which is exactly the bug the other flags were fixed for."""
    for spelling in ("1", "true", "TRUE", "yes", "on", " True "):
        monkeypatch.setenv("UNSLOTH_NOTEBOOK_DEPS_NO_AUTORUN", spelling)
        assert nd._env_is_true("UNSLOTH_NOTEBOOK_DEPS_NO_AUTORUN") is True, spelling
    for spelling in ("0", "", "no", "off"):
        monkeypatch.setenv("UNSLOTH_NOTEBOOK_DEPS_NO_AUTORUN", spelling)
        assert nd._env_is_true("UNSLOTH_NOTEBOOK_DEPS_NO_AUTORUN") is False, spelling


# --------------------------------------------------------------- patch loop safety

def test_the_registered_patch_never_raises(monkeypatch):
    """unsloth/models/_utils.py runs TEMPORARY_PATCHES with `except (ValueError,
    TypeError)`, and that handler RE-INVOKES the patch instead of skipping it.
    Anything else aborts `import unsloth` and skips every patch behind us. A
    dependency installer is never worth breaking the import over."""
    def _explode():
        raise RuntimeError("pip is on fire")

    monkeypatch.setattr(nd, "patch_notebook_deps_autoinstall", _explode)
    nd._patch_notebook_deps_autoinstall_safe()  # must not raise


@pytest.mark.parametrize(
    "exc", [RuntimeError, ImportError, OSError, ValueError, TypeError, KeyboardInterrupt],
)
def test_the_registered_patch_swallows_every_exception_type(monkeypatch, exc):
    def _explode():
        raise exc("boom")

    monkeypatch.setattr(nd, "patch_notebook_deps_autoinstall", _explode)
    if issubclass(exc, Exception):
        nd._patch_notebook_deps_autoinstall_safe()
    else:
        # BaseException (KeyboardInterrupt) must still propagate: swallowing a
        # Ctrl-C would make the import uninterruptible.
        with pytest.raises(exc):
            nd._patch_notebook_deps_autoinstall_safe()


def test_notebook_deps_is_registered_last(monkeypatch):
    """Position is the blast radius. Registered first, a raise loses every model
    patch behind it; registered last, there is nothing behind it."""
    import unsloth_zoo.temporary_patches as tp
    from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES

    names = [getattr(p, "__name__", "") for p in TEMPORARY_PATCHES]
    assert "_patch_notebook_deps_autoinstall_safe" in names, names
    ours = names.index("_patch_notebook_deps_autoinstall_safe")
    assert ours == len(names) - 1, (
        f"notebook_deps is at {ours} of {len(names)}; everything after it is at "
        f"risk if it ever raises: {names[ours + 1:]}"
    )


# --------------------------------------------------------------- source access

def test_module_source_reads_a_zipimported_module(tmp_path, monkeypatch):
    """A transformers installed from a zip (or any loader that is not a plain
    file) used to be skipped silently, and a skip is indistinguishable from a
    successful replay, so the caller carried on into a module whose guarded name
    was never bound. Asking the loader handles it properly."""
    import zipfile

    body = "import os\nMARKER = 'from the zip'\n"
    archive = tmp_path / "pkg.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("zipped_mod.py", body)

    monkeypatch.syspath_prepend(str(archive))
    module = importlib.import_module("zipped_mod")
    try:
        assert module.MARKER == "from the zip"
        # The old code path: __file__ ends in .py but open() cannot read it.
        assert module.__file__.endswith(".py")
        with pytest.raises(OSError):
            open(module.__file__).read()
        # The new path asks the loader, and gets the real source.
        source = nd._module_source(module)
        assert source is not None, "zipimported source still unreadable"
        assert "MARKER = 'from the zip'" in source
    finally:
        sys.modules.pop("zipped_mod", None)


def test_module_source_returns_none_for_a_builtin():
    """A C builtin has no source and never will. None is the right answer, and
    the caller must treat it as 'nothing to replay here', not as a failure."""
    assert nd._module_source(sys.modules["sys"]) is None


def test_module_source_falls_back_to_reading_the_file(tmp_path, monkeypatch):
    """A loader without get_source must still work via __file__."""
    path = tmp_path / "plain_mod.py"
    path.write_text("VALUE = 1\n")
    module = types.ModuleType("plain_mod")
    module.__file__ = str(path)
    module.__loader__ = object()          # no get_source attribute
    assert nd._module_source(module) == "VALUE = 1\n"


def test_a_waiting_thread_sees_the_other_thread_s_success(monkeypatch):
    """The lock alone made the loser fail fast, which is its own wrong answer.

    Thread B used to find the package already claimed, return False immediately, and let
    its caller re-raise the original ImportError for a package thread A was about to
    install successfully. B now waits for A's attempt and then re-probes."""
    started = threading.Event()
    release = threading.Event()
    installed = {"done": False}

    def _slow_run(cmd, *args, **kwargs):
        started.set()
        release.wait(timeout = 10)
        installed["done"] = True
        return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")

    monkeypatch.setattr(subprocess, "run", _slow_run)
    monkeypatch.setattr(nd.shutil, "which", lambda _name: None)
    # Importability follows the install, so a premature answer is visible as False.
    monkeypatch.setattr(nd, "_importable", lambda _name: installed["done"])
    monkeypatch.setattr(nd, "_auto_install_enabled", lambda: True)
    monkeypatch.setattr(nd, "_no_network", lambda: False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)

    results = {}

    def _owner():
        results["a"] = nd._try_install_and_import("einops")

    def _waiter():
        started.wait(timeout = 10)      # make sure A owns the attempt first
        results["b"] = nd._try_install_and_import("einops")

    a = threading.Thread(target = _owner)
    b = threading.Thread(target = _waiter)
    a.start(); b.start()
    started.wait(timeout = 10)
    release.set()
    a.join(timeout = 20); b.join(timeout = 20)

    assert results.get("a") is True, "the owning thread should have installed it"
    assert results.get("b") is True, (
        "the waiting thread reported failure for a package the other thread installed"
    )


def test_the_waiter_is_bounded(monkeypatch):
    """A hung installer must not hang every other thread forever."""
    assert isinstance(nd._ATTEMPT_WAIT_SECONDS, (int, float))
    assert nd._ATTEMPT_WAIT_SECONDS > 300, (
        "the wait has to outlive the 300s subprocess timeout plus the pip fallback"
    )


# ------------------------------------------------- the default is interactive-only

class ZMQInteractiveShell:
    pass


class TerminalInteractiveShell:
    pass


class _EmbeddedShell:
    pass


def _fake_ipython(shell):
    return types.SimpleNamespace(get_ipython = lambda: shell)


def test_an_unset_flag_does_not_run_pip_in_a_plain_script(monkeypatch):
    """Running pip as a side effect of `import unsloth` is defensible when a person is
    watching a notebook cell. In CI, an inference server or a scheduled script nobody
    reads the warning, so the default there is off."""
    monkeypatch.delenv("UNSLOTH_AUTO_INSTALL", raising = False)
    monkeypatch.delitem(sys.modules, "IPython", raising = False)
    assert nd._auto_install_enabled() is False


def test_an_unset_flag_installs_inside_a_live_kernel(monkeypatch):
    monkeypatch.delenv("UNSLOTH_AUTO_INSTALL", raising = False)
    for shell in (ZMQInteractiveShell(), TerminalInteractiveShell()):
        monkeypatch.setitem(sys.modules, "IPython", _fake_ipython(shell))
        assert nd._auto_install_enabled() is True, type(shell).__name__


def test_importable_ipython_without_a_running_shell_is_not_interactive(monkeypatch):
    """IPython is a transitive dependency of plenty of non-interactive installs, so its
    mere presence must not turn the feature on."""
    monkeypatch.delenv("UNSLOTH_AUTO_INSTALL", raising = False)
    monkeypatch.setitem(sys.modules, "IPython", _fake_ipython(None))
    assert nd._auto_install_enabled() is False
    # An unrecognised embedded shell is not a person at a prompt either.
    monkeypatch.setitem(sys.modules, "IPython", _fake_ipython(_EmbeddedShell()))
    assert nd._auto_install_enabled() is False
    # A get_ipython that raises must not propagate out of an import.
    def _boom():
        raise RuntimeError("no shell")
    monkeypatch.setitem(sys.modules, "IPython", types.SimpleNamespace(get_ipython = _boom))
    assert nd._auto_install_enabled() is False


def test_an_explicit_flag_beats_the_context_in_both_directions(monkeypatch):
    """The env var stays authoritative: a server that wants this can opt in, and a
    notebook that does not want it can opt out."""
    monkeypatch.delitem(sys.modules, "IPython", raising = False)
    for value in ("1", "ON", "true", " yes "):
        monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", value)
        assert nd._auto_install_enabled() is True, value

    monkeypatch.setitem(sys.modules, "IPython", _fake_ipython(ZMQInteractiveShell()))
    for value in ("0", "off", "no", "anything-else"):
        monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", value)
        assert nd._auto_install_enabled() is False, value


def test_an_empty_flag_falls_back_to_the_context(monkeypatch):
    """`UNSLOTH_AUTO_INSTALL=` exported by a wrapper script must not read as a refusal
    any more than it reads as consent."""
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "   ")
    monkeypatch.setitem(sys.modules, "IPython", _fake_ipython(ZMQInteractiveShell()))
    assert nd._auto_install_enabled() is True
    monkeypatch.delitem(sys.modules, "IPython", raising = False)
    assert nd._auto_install_enabled() is False


def test_two_different_packages_do_not_install_concurrently(monkeypatch):
    """Per-package events only deduplicate one package. requires_backends can want several
    backends at once, and two pip processes against one prefix can both rewrite a shared
    dependency and its .dist-info."""
    overlap = []
    live = []
    live_lock = threading.Lock()

    def _slow_run(cmd, *args, **kwargs):
        with live_lock:
            live.append(cmd)
            overlap.append(len(live))
        time.sleep(0.2)
        with live_lock:
            live.remove(cmd)
        return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")

    monkeypatch.setattr(subprocess, "run", _slow_run)
    monkeypatch.setattr(nd.shutil, "which", lambda _name: None)
    monkeypatch.setattr(nd, "_auto_install_enabled", lambda: True)

    threads = [
        threading.Thread(target = nd._pip_install, args = (pkg,))
        for pkg in ("einops", "timm", "av", "jieba")
    ]
    for t in threads: t.start()
    for t in threads: t.join(timeout = 30)

    assert len(overlap) == 4, "not every package was attempted: %s" % overlap
    assert max(overlap) == 1, "installers overlapped: %s concurrent" % max(overlap)

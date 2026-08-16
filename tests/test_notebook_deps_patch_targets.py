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

"""Regression tests for where `notebook_deps` puts its hooks and its installs.

Two things this pins, both of which silently defeated the auto-installer:

1. `requires_backends` is re-exported by `transformers/utils/__init__.py`, and
   HF modeling files import it from there (`modeling_timm_wrapper.py` does
   `from ...utils import auto_docstring, is_timm_available, requires_backends`).
   Each of those is a separate name bound to the same function object, so
   rebinding only `transformers.utils.import_utils.requires_backends` leaves
   the alias the model actually calls pointing at the unwrapped original and
   the installer never runs.

2. `uv pip install` resolves its target environment from VIRTUAL_ENV /
   CONDA_PREFIX / a discovered `.venv`, not from the interpreter that spawned
   it. A notebook kernel whose `sys.executable` differs from the inherited
   environment gets the package installed somewhere else, so the follow-up
   `find_spec` still fails. `--python sys.executable` pins it.

Everything here is offline: the transformers module tree is synthetic (no
torch import, no real HF state is mutated) and every installer call is stubbed,
so no test in this file can reach pip, uv or the network.
"""

from __future__ import annotations

import functools
import importlib
import sys
import types

import pytest


notebook_deps = importlib.import_module("unsloth_zoo.temporary_patches.notebook_deps")


# ---------------------------------------------------------------------------
# Synthetic transformers tree.
#
# `patch_requires_backends_autoinstall` reaches transformers through
# `from transformers.utils import import_utils`, which is served straight out
# of `sys.modules` when the entries are already there. Building a fake tree
# keeps the test hermetic (the real transformers, if installed, is untouched)
# and lets us assert against a pristine, definitely-unpatched starting state
# even though importing unsloth_zoo has already patched the real one.
# ---------------------------------------------------------------------------

_BACKEND = "timm"


def _make_original():
    """Stand-in for the real `requires_backends`: raises for a missing backend."""

    def requires_backends(obj, backends):
        wanted = backends if isinstance(backends, (list, tuple)) else [backends]
        missing = [b for b in wanted if not fake_transformers_state["available"].get(b, True)]
        if missing:
            name = getattr(obj, "__name__", type(obj).__name__)
            raise ImportError(f"{name} requires the {', '.join(missing)} library")
        return None

    return requires_backends


fake_transformers_state = {"available": {}}


@pytest.fixture
def fake_transformers(monkeypatch):
    """Install a synthetic `transformers` tree and yield its three aliases.

    The three modules mirror the real layout that matters here:
      * `transformers.utils.import_utils` -- where the function is defined.
      * `transformers.utils`              -- the public re-export.
      * `...timm_wrapper.modeling_timm_wrapper` -- a consumer that copied the
        function into its own globals with `from ...utils import ...`.
    """
    fake_transformers_state["available"] = {_BACKEND: False}
    original = _make_original()

    import_utils = types.ModuleType("transformers.utils.import_utils")
    import_utils.requires_backends = original
    import_utils.BACKENDS_MAPPING = {_BACKEND: (lambda: False, "{0} needs timm")}

    utils = types.ModuleType("transformers.utils")
    utils.import_utils = import_utils
    utils.requires_backends = original          # the public re-export

    root = types.ModuleType("transformers")
    root.utils = utils

    modeling = types.ModuleType(
        "transformers.models.timm_wrapper.modeling_timm_wrapper"
    )
    modeling.requires_backends = original       # `from ...utils import ...`

    for name, module in (
        ("transformers", root),
        ("transformers.utils", utils),
        ("transformers.utils.import_utils", import_utils),
        ("transformers.models.timm_wrapper.modeling_timm_wrapper", modeling),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    yield types.SimpleNamespace(
        import_utils=import_utils,
        utils=utils,
        modeling=modeling,
        original=original,
    )
    fake_transformers_state["available"] = {}


@pytest.fixture
def record_installs(monkeypatch):
    """Replace the installer with a recorder, so nothing can reach pip/uv."""
    calls = []

    def _stub(pkg):
        calls.append(pkg)
        return False        # "install failed" -> the wrapper re-raises

    monkeypatch.setattr(notebook_deps, "_try_install_and_import", _stub)
    monkeypatch.setattr(notebook_deps, "_AUTO_INSTALL", True)
    monkeypatch.setattr(notebook_deps, "_NO_NETWORK", False)
    return calls


class _Consumer:
    __name__ = "TimmWrapperModel"


# ---------------------------------------------------------------------------
# 1. requires_backends: every alias must route into the installer.
# ---------------------------------------------------------------------------


ALIAS_IDS = [
    "transformers.utils.import_utils.requires_backends",
    "transformers.utils.requires_backends",
    "modeling_timm_wrapper.requires_backends",
]


@pytest.mark.parametrize("holder", ["import_utils", "utils", "modeling"], ids=ALIAS_IDS)
def test_every_requires_backends_alias_is_wrapped(fake_transformers, holder):
    notebook_deps.patch_requires_backends_autoinstall()
    alias = getattr(fake_transformers, holder).requires_backends
    assert getattr(alias, "_unsloth_patched", False), (
        f"{holder}.requires_backends is still the unwrapped original; a model "
        f"calling it raises before the auto-installer is reached"
    )


@pytest.mark.parametrize("holder", ["import_utils", "utils", "modeling"], ids=ALIAS_IDS)
def test_every_requires_backends_alias_reaches_the_installer(
    fake_transformers, record_installs, holder
):
    notebook_deps.patch_requires_backends_autoinstall()
    alias = getattr(fake_transformers, holder).requires_backends
    with pytest.raises(ImportError):
        alias(_Consumer, [_BACKEND])
    assert record_installs == [_BACKEND], (
        f"calling {holder}.requires_backends did not reach the installer; "
        f"recorded calls: {record_installs}"
    )


def test_all_aliases_share_one_wrapper_object(fake_transformers):
    """One wrapper everywhere, so `_unsloth_patched` stays a single sentinel."""
    notebook_deps.patch_requires_backends_autoinstall()
    wrapper = fake_transformers.import_utils.requires_backends
    assert fake_transformers.utils.requires_backends is wrapper
    assert fake_transformers.modeling.requires_backends is wrapper


def test_second_pass_does_not_double_wrap_but_still_rebinds_new_modules(
    fake_transformers, monkeypatch
):
    """The hook runs at import time and again from the TEMPORARY_PATCHES pass.

    The second pass must not stack another wrapper on top of the first, but it
    must still reach a module imported in between, which is holding its own
    copy of the original.
    """
    notebook_deps.patch_requires_backends_autoinstall()
    wrapper = fake_transformers.import_utils.requires_backends

    late = types.ModuleType("transformers.models.late_arrival.modeling_late")
    late.requires_backends = fake_transformers.original
    monkeypatch.setitem(sys.modules, late.__name__, late)

    notebook_deps.patch_requires_backends_autoinstall()

    assert fake_transformers.import_utils.requires_backends is wrapper, (
        "second pass re-wrapped an already-wrapped function"
    )
    assert late.requires_backends is wrapper, (
        "a module imported after the first pass kept the unwrapped original"
    )


def test_unrelated_requires_backends_is_left_alone(fake_transformers, monkeypatch):
    """The rebind is identity-scoped, so a same-named function elsewhere stays."""
    other = types.ModuleType("some_unrelated_package")
    sentinel = lambda obj, backends: None  # noqa: E731
    other.requires_backends = sentinel
    monkeypatch.setitem(sys.modules, other.__name__, other)

    notebook_deps.patch_requires_backends_autoinstall()

    assert other.requires_backends is sentinel


def test_patch_survives_transformers_without_requires_backends(monkeypatch):
    """An older/newer transformers lacking the helper must not break the patch."""
    import_utils = types.ModuleType("transformers.utils.import_utils")
    utils = types.ModuleType("transformers.utils")
    utils.import_utils = import_utils
    root = types.ModuleType("transformers")
    root.utils = utils
    for name, module in (
        ("transformers", root),
        ("transformers.utils", utils),
        ("transformers.utils.import_utils", import_utils),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    notebook_deps.patch_requires_backends_autoinstall()  # must not raise

    assert not hasattr(import_utils, "requires_backends")


# ---------------------------------------------------------------------------
# 2. _pip_install: uv must be aimed at the running interpreter.
# ---------------------------------------------------------------------------


@pytest.fixture
def install_spy(monkeypatch):
    """Force the uv branch and capture every command, running none of them."""
    commands = []
    results = []

    def _run(cmd, *args, **kwargs):
        commands.append(list(cmd))
        returncode, stderr = results.pop(0) if results else (0, "")
        return types.SimpleNamespace(returncode=returncode, stdout="", stderr=stderr)

    monkeypatch.setattr(notebook_deps, "subprocess", types.SimpleNamespace(run=_run))
    monkeypatch.setattr(notebook_deps.shutil, "which", lambda exe: "/usr/bin/uv")
    monkeypatch.setattr(notebook_deps, "_in_venv", lambda: True)
    monkeypatch.setattr(notebook_deps, "_attempted", set())
    return types.SimpleNamespace(commands=commands, results=results)


def test_uv_install_targets_the_running_interpreter(install_spy):
    assert notebook_deps._pip_install("timm") is True
    cmd = install_spy.commands[0]
    assert cmd[:3] == ["uv", "pip", "install"]
    assert "--python" in cmd, (
        "uv resolves its target from VIRTUAL_ENV / CONDA_PREFIX, so without "
        "--python a notebook kernel installs into a different environment"
    )
    assert cmd[cmd.index("--python") + 1] == sys.executable
    assert cmd[-1] == "timm"


def test_uv_usage_error_falls_back_to_pip(install_spy):
    """A uv too old for --python exits 2 before the network; retry via pip."""
    install_spy.results.append((2, "error: unexpected argument '--python' found"))
    install_spy.results.append((0, ""))

    assert notebook_deps._pip_install("timm") is True

    assert len(install_spy.commands) == 2, install_spy.commands
    assert install_spy.commands[0][0] == "uv"
    assert install_spy.commands[1][:4] == [sys.executable, "-m", "pip", "install"]


def test_uv_interpreter_discovery_error_falls_back_to_pip(install_spy):
    """If uv cannot resolve the interpreter we handed it, pip still can."""
    install_spy.results.append((
        2,
        "error: No virtual environment or system Python installation found for "
        "path `/some/python`; run `uv venv` to create an environment",
    ))
    install_spy.results.append((0, ""))

    assert notebook_deps._pip_install("timm") is True

    assert len(install_spy.commands) == 2, install_spy.commands
    assert install_spy.commands[1][:4] == [sys.executable, "-m", "pip", "install"]


def test_genuine_uv_failure_is_not_retried_through_pip(install_spy):
    """Only an argument-parser error is retried, so real failures cost one run."""
    install_spy.results.append((1, "error: No solution found when resolving"))

    assert notebook_deps._pip_install("timm") is False

    assert len(install_spy.commands) == 1, install_spy.commands
    assert install_spy.commands[0][0] == "uv"


def test_pip_fallback_uses_sys_executable(install_spy, monkeypatch):
    monkeypatch.setattr(notebook_deps.shutil, "which", lambda exe: None)

    assert notebook_deps._pip_install("timm") is True

    cmd = install_spy.commands[0]
    assert cmd[:4] == [sys.executable, "-m", "pip", "install"]


def test_repeated_requests_for_the_same_package_run_once(install_spy):
    assert notebook_deps._pip_install("timm") is True
    assert notebook_deps._pip_install("timm") is False
    assert len(install_spy.commands) == 1


# ---------------------------------------------------------------------------
# 4. A successful install must invalidate the cached availability probe.
# ---------------------------------------------------------------------------


def _fake_lru_cached_transformers(monkeypatch, installed):
    """A synthetic transformers shaped like 5.x: the availability probe behind
    `BACKENDS_MAPPING` is an `lru_cache` wrapper, and `requires_backends`
    decides from it, exactly as `transformers.utils.import_utils` does."""

    @functools.lru_cache(maxsize = None)
    def is_backend_available():
        return installed[_BACKEND]

    import_utils = types.ModuleType("transformers.utils.import_utils")
    import_utils.BACKENDS_MAPPING = {_BACKEND: (is_backend_available, "{0} needs timm")}

    def original(obj, backends):
        wanted = backends if isinstance(backends, (list, tuple)) else [backends]
        failed = [b for b in wanted if not import_utils.BACKENDS_MAPPING[b][0]()]
        if failed:
            name = getattr(obj, "__name__", type(obj).__name__)
            raise ImportError(f"{name} requires the {', '.join(failed)} library")
        return None

    import_utils.requires_backends = original

    utils = types.ModuleType("transformers.utils")
    utils.import_utils = import_utils
    utils.requires_backends = original

    root = types.ModuleType("transformers")
    root.utils = utils

    for name, module in (
        ("transformers", root),
        ("transformers.utils", utils),
        ("transformers.utils.import_utils", import_utils),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return import_utils


def test_successful_install_invalidates_the_cached_availability_probe(monkeypatch):
    """On transformers 5.x `BACKENDS_MAPPING[b][0]` is an `lru_cache` wrapper
    that has already cached False by the time we get here. Without clearing it
    the retry re-reads the stale answer and raises the very ImportError the
    install just removed, so reaching the installer buys nothing."""
    installed = {_BACKEND: False}
    import_utils = _fake_lru_cached_transformers(monkeypatch, installed)

    # Prime the cache the way the first failing import does.
    assert import_utils.BACKENDS_MAPPING[_BACKEND][0]() is False

    def _stub(pkg):
        installed[pkg] = True       # the install genuinely succeeded
        return True

    monkeypatch.setattr(notebook_deps, "_try_install_and_import", _stub)
    monkeypatch.setattr(notebook_deps, "_AUTO_INSTALL", True)
    monkeypatch.setattr(notebook_deps, "_NO_NETWORK", False)

    notebook_deps.patch_requires_backends_autoinstall()

    # Must not raise: the retry has to observe the freshly installed package.
    import_utils.requires_backends(_Consumer, [_BACKEND])


def test_refresh_tolerates_a_backend_with_no_cached_probe(monkeypatch):
    """A plain callable (transformers 4.x style) and an unknown backend must
    both be no-ops rather than raising out of the installer path."""
    import_utils = types.ModuleType("transformers.utils.import_utils")
    import_utils.BACKENDS_MAPPING = {_BACKEND: (lambda: False, "{0} needs timm")}

    notebook_deps._refresh_backend_availability(import_utils, _BACKEND)
    notebook_deps._refresh_backend_availability(import_utils, "not-a-backend")

    # transformers 4.x kept a module level flag; it is set when it exists.
    import_utils._timm_available = False
    notebook_deps._refresh_backend_availability(import_utils, _BACKEND)
    assert import_utils._timm_available is True

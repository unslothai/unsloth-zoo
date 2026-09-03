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
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "1")
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
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "1")
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


# ---------------------------------------------------------------------------
# 5. A backend whose install failed must not be marked available.
# ---------------------------------------------------------------------------


_SECOND_BACKEND = "av"


def _fake_flag_based_transformers(monkeypatch, installed):
    """A synthetic transformers shaped like 4.x: one module level
    ``_<backend>_available`` flag per backend, and the ``BACKENDS_MAPPING``
    probe reading it. Verified against the real transformers 4.57.6, where
    ``is_av_available()`` is ``return _av_available`` and the mapping entry
    carries no ``cache_clear``."""
    import_utils = types.ModuleType("transformers.utils.import_utils")

    def _probe(backend):
        return lambda: getattr(import_utils, f"_{backend}_available")

    for backend, available in installed.items():
        setattr(import_utils, f"_{backend}_available", available)
    import_utils.BACKENDS_MAPPING = {
        backend: (_probe(backend), "{0} needs " + backend) for backend in installed
    }

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


def test_a_failed_install_is_not_marked_available(monkeypatch):
    """One backend installs, another does not: only the first may be refreshed.

    Refreshing every requested backend sets `_<backend>_available = True` for
    the one that is still missing, so the retry succeeds and the caller walks
    into a bare ModuleNotFoundError further down instead of the actionable
    ImportError it was about to get.
    """
    import_utils = _fake_flag_based_transformers(
        monkeypatch, {_BACKEND: False, _SECOND_BACKEND: False}
    )

    def _stub(pkg):
        return pkg == _BACKEND      # `av` cannot be installed here

    monkeypatch.setattr(notebook_deps, "_try_install_and_import", _stub)
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "1")
    monkeypatch.setattr(notebook_deps, "_NO_NETWORK", False)

    notebook_deps.patch_requires_backends_autoinstall()

    with pytest.raises(ImportError) as excinfo:
        import_utils.requires_backends(_Consumer, [_BACKEND, _SECOND_BACKEND])
    assert _SECOND_BACKEND in str(excinfo.value)
    assert getattr(import_utils, f"_{_SECOND_BACKEND}_available") is False
    # The one that did install is still refreshed, or the retry buys nothing.
    assert getattr(import_utils, f"_{_BACKEND}_available") is True


# ---------------------------------------------------------------------------
# 4. _in_venv: the running interpreter decides, not an inherited variable.
#
# The mirror image of (2). `--python sys.executable` stopped uv installing into
# whatever VIRTUAL_ENV / CONDA_PREFIX happened to name, but `_in_venv()` still
# read those same variables, and it gates two decisions: whether `_pip_install`
# hands the job to uv at all, and whether `_pip_command` probes for write access
# and falls back to `--user`. Under the kernel mismatch this module exists to
# handle (interpreter A running, variables inherited from environment B) a stale
# variable made every one of those decisions as if A were a venv, while the
# installers kept targeting A's site-packages.
# ---------------------------------------------------------------------------


@pytest.fixture
def two_environments(tmp_path, monkeypatch):
    """Interpreter A is running; B is a real, different directory it is not in."""
    a = tmp_path / "envA"
    b = tmp_path / "envB"
    a.mkdir()
    b.mkdir()
    monkeypatch.delattr(sys, "real_prefix", raising = False)
    monkeypatch.setattr(sys, "prefix", str(a))
    monkeypatch.setattr(sys, "base_prefix", str(a))   # A is not itself a venv
    monkeypatch.delenv("VIRTUAL_ENV", raising = False)
    monkeypatch.delenv("CONDA_PREFIX", raising = False)
    return types.SimpleNamespace(a = a, b = b)


@pytest.fixture
def unwritable_site(tmp_path, monkeypatch):
    """A site-packages path the write probe cannot create, for any uid."""
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("")
    target = blocker / "site-packages"
    monkeypatch.setattr(notebook_deps.site, "getsitepackages", lambda: [str(target)])
    monkeypatch.setattr(notebook_deps.os, "geteuid", lambda: 1000, raising = False)
    return target


@pytest.mark.parametrize("variable", ["VIRTUAL_ENV", "CONDA_PREFIX"])
def test_in_venv_ignores_an_activation_variable_for_another_environment(
    two_environments, monkeypatch, variable,
):
    monkeypatch.setenv(variable, str(two_environments.b))
    assert notebook_deps._in_venv() is False, (
        f"{variable} points at an environment the running interpreter is not in, "
        f"so it says nothing about where sys.executable installs to"
    )


@pytest.mark.parametrize("variable", ["VIRTUAL_ENV", "CONDA_PREFIX"])
def test_in_venv_trusts_a_variable_naming_the_running_prefix(
    two_environments, monkeypatch, variable,
):
    """conda environments have base_prefix == prefix, so the variable is all they have."""
    monkeypatch.setenv(variable, str(two_environments.a))
    assert notebook_deps._in_venv() is True


def test_in_venv_reads_the_interpreter_when_no_variable_is_set(two_environments, monkeypatch):
    assert notebook_deps._in_venv() is False
    monkeypatch.setattr(sys, "base_prefix", str(two_environments.b))  # A is a venv now
    assert notebook_deps._in_venv() is True


def test_in_venv_ignores_a_variable_pointing_at_a_deleted_environment(
    two_environments, monkeypatch,
):
    gone = two_environments.b / "removed"
    monkeypatch.setenv("VIRTUAL_ENV", str(gone))
    assert notebook_deps._in_venv() is False


def test_pip_command_falls_back_to_user_under_a_kernel_mismatch(
    two_environments, unwritable_site, monkeypatch,
):
    """The stale variable must not skip the write probe and --user."""
    monkeypatch.setenv("VIRTUAL_ENV", str(two_environments.b))
    cmd = notebook_deps._pip_command("timm")
    assert cmd[:4] == [sys.executable, "-m", "pip", "install"]
    assert "--user" in cmd, (
        "pip runs under sys.executable, whose site-packages is unwritable here; "
        "an inherited VIRTUAL_ENV does not make it writable"
    )


def test_pip_command_keeps_user_off_a_real_venv(two_environments, unwritable_site, monkeypatch):
    """pip refuses --user inside a virtualenv, so a genuine venv must not get it."""
    monkeypatch.setattr(sys, "base_prefix", str(two_environments.b))
    cmd = notebook_deps._pip_command("timm")
    assert "--user" not in cmd


def test_pip_install_does_not_hand_a_mismatched_kernel_to_uv(
    two_environments, unwritable_site, monkeypatch,
):
    """uv has no user-site fallback, and its permission failure is not retried.

    Observed for real: with VIRTUAL_ENV naming another environment, `uv pip
    install --python /usr/bin/python3 addict` exits 2 with "Permission denied
    (os error 13)". That text matches none of the retry triggers in
    `_run_install`, so `_pip_install` returned False without ever reaching pip,
    which installs the same package into the user site and exits 0.
    """
    commands = []

    def _run(cmd, *args, **kwargs):
        commands.append(list(cmd))
        if cmd[0] == "uv":
            return types.SimpleNamespace(
                returncode = 2, stdout = "",
                stderr = "error: Failed to install: addict-2.4.0-py3-none-any.whl\n"
                         "  Caused by: Permission denied (os error 13)",
            )
        return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")

    monkeypatch.setattr(notebook_deps, "subprocess", types.SimpleNamespace(run = _run))
    monkeypatch.setattr(notebook_deps.shutil, "which", lambda exe: "/usr/bin/uv")
    monkeypatch.setattr(notebook_deps, "_attempted", set())
    monkeypatch.setenv("VIRTUAL_ENV", str(two_environments.b))

    assert notebook_deps._pip_install("addict") is True
    assert [c[0] for c in commands] == [sys.executable], commands
    assert "--user" in commands[0]


# ---------------------------------------------------------------------------
# 6. UNSLOTH_AUTO_INSTALL is read at the moment of the attempt.
#
# `import unsloth` imports this module (and runs the hooks at import time), so
# the documented opt-out is routinely set AFTER that: before loading a model,
# or straight after `_run_install`'s warning names the variable. Captured once
# at import, the opt-out would be honoured only by callers who happened to set
# it before the very first Unsloth import, and everyone else would get pip run
# against an explicit refusal.
# ---------------------------------------------------------------------------


@pytest.fixture
def package_manager_spy(monkeypatch):
    """Record every command the module would hand to uv/pip, run none of them."""
    commands = []

    def _run(cmd, *args, **kwargs):
        commands.append(list(cmd))
        return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")

    monkeypatch.setattr(notebook_deps, "subprocess", types.SimpleNamespace(run = _run))
    monkeypatch.setattr(notebook_deps, "_attempted", set())
    monkeypatch.setattr(notebook_deps, "_NO_NETWORK", False)
    return commands


@pytest.fixture
def fake_dynamic_module_utils(monkeypatch):
    """Synthetic `transformers.dynamic_module_utils` raising the Deepseek-OCR
    style "This modeling file requires ..." ImportError."""
    def check_imports(filename):
        raise ImportError(
            "This modeling file requires the following packages that were not "
            "found in your environment: addict. Run `pip install addict`"
        )

    dmu = types.ModuleType("transformers.dynamic_module_utils")
    dmu.check_imports = check_imports
    root = types.ModuleType("transformers")
    root.dynamic_module_utils = dmu
    monkeypatch.setitem(sys.modules, "transformers", root)
    monkeypatch.setitem(sys.modules, "transformers.dynamic_module_utils", dmu)
    return dmu


def test_requires_backends_honours_an_opt_out_set_after_import(
    fake_transformers, package_manager_spy, monkeypatch
):
    # Spied one level above pip as well, because the allow-listed package may
    # already be importable on the machine running the test, in which case the
    # installer returns before it would have reached uv/pip.
    reached = []
    monkeypatch.setattr(
        notebook_deps, "_try_install_and_import",
        lambda pkg: reached.append(pkg) or False,
    )
    notebook_deps.patch_requires_backends_autoinstall()
    # Only now, exactly as a user reacting to the auto-install warning does.
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")

    with pytest.raises(ImportError):
        fake_transformers.utils.requires_backends(_Consumer, [_BACKEND])
    assert reached == [], f"the opt-out was ignored and the installer ran: {reached}"
    assert package_manager_spy == []


def test_check_imports_honours_an_opt_out_set_after_import(
    fake_dynamic_module_utils, package_manager_spy, monkeypatch
):
    notebook_deps.patch_check_imports_autoinstall()
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")

    with pytest.raises(ImportError):
        fake_dynamic_module_utils.check_imports("modeling_deepseekocr.py")
    assert package_manager_spy == []


def test_the_ipython_chain_repair_honours_an_opt_out_set_after_import(
    package_manager_spy, monkeypatch
):
    reached = []
    monkeypatch.setattr(
        notebook_deps, "_try_install_and_import",
        lambda pkg: reached.append(pkg) or False,
    )
    monkeypatch.setattr(notebook_deps, "_ipython_chain_is_broken", lambda: True)
    # traitlets absent, so the repair would have something to install.
    monkeypatch.setattr(notebook_deps, "importlib", types.SimpleNamespace(
        util = types.SimpleNamespace(find_spec = lambda name: None),
    ))
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")

    notebook_deps._ensure_notebook_chain()
    assert reached == []
    assert package_manager_spy == []


def test_the_installer_itself_honours_an_opt_out_set_after_import(
    package_manager_spy, monkeypatch
):
    """The gate the two wrappers share, reached directly."""
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")
    assert notebook_deps._try_install_and_import("addict") is False
    assert package_manager_spy == []


def test_clearing_the_opt_out_at_runtime_re_enables_the_installer(
    fake_transformers, monkeypatch
):
    """The same liveness in the other direction, so the fix cannot be "always off"."""
    reached = []
    monkeypatch.setattr(
        notebook_deps, "_try_install_and_import",
        lambda pkg: reached.append(pkg) or False,
    )
    monkeypatch.setattr(notebook_deps, "_NO_NETWORK", False)
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")
    notebook_deps.patch_requires_backends_autoinstall()

    with pytest.raises(ImportError):
        fake_transformers.utils.requires_backends(_Consumer, [_BACKEND])
    assert reached == []

    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "1")
    with pytest.raises(ImportError):
        fake_transformers.utils.requires_backends(_Consumer, [_BACKEND])
    assert reached == [_BACKEND]

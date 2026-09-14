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

"""A --user install has to reach sys.path in the session that ran it.

site.addusersitepackages adds the user site only `if os.path.isdir(user_site)`, decided at
interpreter start. On a non-root interpreter with no user site yet, the first --user install
creates the directory and nothing can import out of it, so the auto-installer would re-raise
the very ImportError it ran pip to fix.
"""

from __future__ import annotations

import importlib
import site
import subprocess
import sys
import types

import pytest


notebook_deps = importlib.import_module("unsloth_zoo.temporary_patches.notebook_deps")


@pytest.fixture
def fresh_user_site(tmp_path, monkeypatch):
    """A user site that does not exist yet, created by the 'install'; sys.path is restored after."""
    target = tmp_path / "usersite"
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(target))
    monkeypatch.setattr(site, "ENABLE_USER_SITE", True, raising = False)
    before = list(sys.path)

    def _run(cmd, **kwargs):
        target.mkdir(parents = True, exist_ok = True)   # what pip --user does
        return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")

    monkeypatch.setattr(subprocess, "run", _run)
    try:
        yield target
    finally:
        sys.path[:] = before


def test_a_user_install_puts_the_new_directory_on_sys_path(fresh_user_site):
    assert str(fresh_user_site) not in sys.path, "precondition: it is not there yet"

    ok, retry = notebook_deps._run_install("snac", ["pip", "install", "--user", "snac"])

    assert ok and not retry
    assert str(fresh_user_site) in sys.path, (
        "the package was installed somewhere nothing can import from, so the caller "
        "re-raises the ImportError it just ran pip to fix"
    )


def test_an_install_without_user_still_refreshes_the_user_site(fresh_user_site):
    """This used to assert the opposite, that only a --user install may touch sys.path.

    That premise is wrong, because the flag is not what decides where pip installs.
    decide_user_install (pip/_internal/commands/install.py) returns True whenever
    site_packages_writable() is False, logging "Defaulting to user installation because
    normal site-packages is not writeable". And we never pass --user on Windows at all:
    the writability probe in _pip_command is gated on os.geteuid, which does not exist
    there. So a non-admin Windows install into Program Files takes pip's silent
    fallback, lands in %APPDATA%\\Python, and stays off sys.path, which made a
    successful install read as a failed one."""
    assert str(fresh_user_site) not in sys.path, "precondition: it is not there yet"

    ok, _ = notebook_deps._run_install("snac", ["pip", "install", "snac"])

    assert ok
    assert str(fresh_user_site) in sys.path, (
        "pip chose a user install on its own and we never exposed the directory"
    )


def test_a_disabled_user_site_is_not_overridden(fresh_user_site, monkeypatch):
    """-s, PYTHONNOUSERSITE or a venv: pip's --user would have refused too, so do not add it back."""
    monkeypatch.setattr(site, "ENABLE_USER_SITE", False, raising = False)

    ok, _ = notebook_deps._run_install("snac", ["pip", "install", "--user", "snac"])

    assert ok
    assert str(fresh_user_site) not in sys.path


def test_adding_it_twice_does_not_duplicate_the_entry(fresh_user_site):
    notebook_deps._run_install("snac", ["pip", "install", "--user", "snac"])
    notebook_deps._run_install("snac", ["pip", "install", "--user", "snac"])

    assert sys.path.count(str(fresh_user_site)) == 1


def test_a_failed_install_adds_nothing(fresh_user_site, monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: types.SimpleNamespace(returncode = 1, stdout = "", stderr = "boom"),
    )

    ok, _ = notebook_deps._run_install("snac", ["pip", "install", "--user", "snac"])

    assert not ok
    assert str(fresh_user_site) not in sys.path


def test_the_stdlib_really_gates_on_the_directory_existing():
    """Premise pin: if CPython adds a user site that does not exist yet, the helper is dead code."""
    import inspect

    src = inspect.getsource(site)
    body = src[src.index("def addusersitepackages"):]
    body = body[: body.index("\ndef ", 1)]
    assert "os.path.isdir(user_site)" in body, body


def test_user_site_is_refreshed_even_without_an_explicit_user_flag(monkeypatch):
    """pip can choose a user install by itself, and we would never notice.

    decide_user_install (pip/_internal/commands/install.py) returns True whenever
    site_packages_writable() is False, logging "Defaulting to user installation
    because normal site-packages is not writeable". We never pass --user on Windows,
    because the writability probe is gated on os.geteuid, which does not exist there.
    So a non-admin Windows install into Program Files takes pip's silent fallback and
    the package lands somewhere that is not on sys.path, making a successful install
    look like a failed one."""
    import subprocess
    import types

    import unsloth_zoo.temporary_patches.notebook_deps as nd

    called = []
    monkeypatch.setattr(nd, "_add_user_site", lambda: called.append(True))
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, *a, **k: types.SimpleNamespace(returncode = 0, stdout = "", stderr = ""),
    )
    # No --user anywhere in the command, exactly as on Windows.
    ok, retry = nd._run_install("einops", ["python", "-m", "pip", "install", "einops"])
    assert ok is True and retry is False
    assert called, "a successful install did not refresh the user site directory"


def test_user_site_is_not_refreshed_after_a_failed_install(monkeypatch):
    """The control. Nothing was installed, so there is nothing new to expose."""
    import subprocess
    import types

    import unsloth_zoo.temporary_patches.notebook_deps as nd

    called = []
    monkeypatch.setattr(nd, "_add_user_site", lambda: called.append(True))
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, *a, **k: types.SimpleNamespace(returncode = 1, stdout = "", stderr = "boom"),
    )
    nd._run_install("einops", ["python", "-m", "pip", "install", "einops"])
    assert not called

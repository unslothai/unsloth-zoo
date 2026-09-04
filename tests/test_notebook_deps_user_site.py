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

site.addusersitepackages adds the user site only `if os.path.isdir(user_site)`, and
that is decided at interpreter start. So on a non-root system interpreter with no user
site yet, the first --user install creates the directory and nothing can import out of
it: the auto-installer changes the environment and then re-raises the very ImportError
it was invoking pip to fix, telling the user to install what they have just installed.
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
    """A user site that does not exist yet, created by the 'install', with sys.path
    restored afterwards so one test cannot leak a path into the next."""
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


def test_an_install_without_user_leaves_sys_path_alone(fresh_user_site):
    """Non-vacuity, and the blast radius: only a --user install may touch sys.path."""
    before = list(sys.path)

    ok, _ = notebook_deps._run_install("snac", ["pip", "install", "snac"])

    assert ok
    assert sys.path == before


def test_a_disabled_user_site_is_not_overridden(fresh_user_site, monkeypatch):
    """-s, PYTHONNOUSERSITE or a venv: the interpreter was told to ignore the user
    site, and pip's --user would have refused too, so adding it back is not ours."""
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
    """Premise pin. If CPython ever starts adding a user site that does not exist yet,
    the helper above becomes dead code and should go rather than linger."""
    import inspect

    src = inspect.getsource(site)
    body = src[src.index("def addusersitepackages"):]
    body = body[: body.index("\ndef ", 1)]
    assert "os.path.isdir(user_site)" in body, body

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

"""The wheel must ship unsloth_zoo and nothing else at the top level.

setuptools auto-discovery used to package the repo's tests/ and scripts/ trees,
which land in site-packages as top-level `tests` and `scripts`. Those names are
shared with other wheels, so the last install wins and any uninstall deletes the
survivor's files, leaving a RECORD that points at a file nothing recreates.
Studio's post-update integrity check then reports the tree as damaged and
refuses to start. These CPU-only tests pin the pyproject config that fixes it.
"""
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Names other wheels also write into, so unsloth_zoo must never claim them.
_SHARED_TOP_LEVEL = ("test", "tests", "doc", "docs", "example", "examples", "scripts")


def _find_config():
    if sys.version_info < (3, 11):
        pytest.skip("tomllib needs Python 3.11+")
    import tomllib

    with open(os.path.join(_REPO_ROOT, "pyproject.toml"), "rb") as f:
        data = tomllib.load(f)
    return data["tool"]["setuptools"]["packages"]["find"]


def _discover(config):
    # PEP420PackageFinder is what auto-discovery ran, so a directory without
    # __init__.py (tests/, scripts/) still counts as a package here.
    from setuptools.discovery import PEP420PackageFinder

    found = PEP420PackageFinder.find(
        where = _REPO_ROOT,
        include = tuple(config.get("include", ("*",))),
        exclude = tuple(config.get("exclude", ())),
    )
    return sorted(p for p in found if "." not in p)


def test_config_excludes_the_shared_trees():
    config = _find_config()
    assert config.get("include") == ["unsloth_zoo", "unsloth_zoo.*"]
    for name in ("tests*", "scripts*"):
        assert name in config.get("exclude", []), f"{name} must stay excluded"


def test_a_sibling_of_the_package_is_not_discovered(tmp_path):
    # "unsloth_zoo*" would match unsloth_zoo_cache/ and ship it as its own
    # top-level package, which is the shape this whole change exists to stop.
    from setuptools.discovery import PEP420PackageFinder

    (tmp_path / "unsloth_zoo").mkdir()
    (tmp_path / "unsloth_zoo" / "__init__.py").touch()
    (tmp_path / "unsloth_zoo_cache").mkdir()
    (tmp_path / "unsloth_zoo_cache" / "blob.py").touch()
    config = _find_config()
    found = PEP420PackageFinder.find(
        where = str(tmp_path),
        include = tuple(config["include"]),
        exclude = tuple(config["exclude"]),
    )
    assert sorted(found) == ["unsloth_zoo"]


def test_only_unsloth_zoo_is_discovered():
    top_level = _discover(_find_config())
    assert top_level == ["unsloth_zoo"], f"unexpected top-level packages: {top_level}"


def test_shared_names_are_never_discovered():
    top_level = set(_discover(_find_config()))
    assert not top_level.intersection(_SHARED_TOP_LEVEL)


def test_package_payload_is_untouched():
    # The exclusion must not thin out unsloth_zoo itself.
    from setuptools.discovery import PEP420PackageFinder

    config = _find_config()
    found = PEP420PackageFinder.find(
        where = _REPO_ROOT,
        include = tuple(config["include"]),
        exclude = tuple(config["exclude"]),
    )
    assert "unsloth_zoo" in found
    assert "unsloth_zoo._vendored" in found
    assert len(found) > 10, f"only found {len(found)} packages, discovery is wrong"

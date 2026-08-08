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
    assert config.get("include") == ["unsloth_zoo*"]
    for name in ("tests*", "scripts*"):
        assert name in config.get("exclude", []), f"{name} must stay excluded"


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

"""Distro detection and package-manager mapping in llama_cpp.py.

Covers the Alpine / openSUSE / Gentoo additions: `check_linux_type` picks them
up from their release files or /etc/os-release, the per-distro package maps
have an entry for each, and the package-not-found detection matches each
manager's own phrase without matching an ordinary "not found" line.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


def _load_llama_cpp_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "unsloth_zoo" / "llama_cpp.py"
    spec = importlib.util.spec_from_file_location("llama_cpp_under_test_distro", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def llama_cpp():
    return _load_llama_cpp_module()


def _only_these_files_exist(monkeypatch, present):
    real_exists = os.path.exists

    def fake_exists(path):
        if str(path).startswith("/etc/"):
            return str(path) in present
        return real_exists(path)

    monkeypatch.setattr(os.path, "exists", fake_exists)


@pytest.mark.parametrize(
    "release_file, expected",
    [
        ("/etc/debian_version", "debian"),
        ("/etc/fedora-release", "rpm"),
        ("/etc/arch-release", "arch"),
        ("/etc/alpine-release", "alpine"),
        ("/etc/gentoo-release", "gentoo"),
    ],
)
def test_check_linux_type_from_release_file(llama_cpp, monkeypatch, release_file, expected):
    monkeypatch.setattr(llama_cpp.platform, "system", lambda: "Linux")
    monkeypatch.setattr(llama_cpp, "_os_release_ids", lambda: set())
    _only_these_files_exist(monkeypatch, {release_file})
    assert llama_cpp.check_linux_type() == expected


@pytest.mark.parametrize(
    "ids, expected",
    [
        ({"opensuse-leap", "suse", "opensuse"}, "suse"),  # openSUSE Leap: ID + ID_LIKE
        ({"opensuse-tumbleweed", "suse", "opensuse"}, "suse"),
        ({"sles"}, "suse"),  # SLES sets ID=sles, ID_LIKE=suse on most releases
        ({"ubuntu", "debian"}, "unknown"),  # no release file matched, os-release is not suse
        (set(), "unknown"),
    ],
)
def test_check_linux_type_from_os_release(llama_cpp, monkeypatch, ids, expected):
    monkeypatch.setattr(llama_cpp.platform, "system", lambda: "Linux")
    monkeypatch.setattr(llama_cpp, "_os_release_ids", lambda: ids)
    _only_these_files_exist(monkeypatch, set())
    assert llama_cpp.check_linux_type() == expected


def test_os_release_ids_parses_id_and_id_like(llama_cpp, tmp_path, monkeypatch):
    os_release = tmp_path / "os-release"
    os_release.write_text('NAME="openSUSE Leap"\nID="opensuse-leap"\nID_LIKE="suse opensuse"\nVERSION_ID="15.6"\n')
    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == "/etc/os-release":
            return real_open(os_release, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    assert llama_cpp._os_release_ids() == {"opensuse-leap", "suse", "opensuse"}


def test_os_release_ids_is_empty_when_unreadable(llama_cpp, monkeypatch):
    def fake_open(path, *args, **kwargs):
        raise FileNotFoundError(path)

    monkeypatch.setattr("builtins.open", fake_open)
    assert llama_cpp._os_release_ids() == set()


@pytest.mark.parametrize("system_type", ["rpm", "arch", "alpine", "suse", "gentoo"])
def test_every_distro_has_a_manager_name_and_build_tools(llama_cpp, system_type):
    assert system_type in llama_cpp.PACKAGE_MANAGER_NAMES
    assert "build-essential" in llama_cpp.DISTRO_PACKAGES[system_type]


@pytest.mark.parametrize(
    "line",
    [
        "E: Unable to locate package libcurl4-openssl-dev",  # apt-get
        "Error: No match for argument: libcurl-devel",  # dnf
        "error: target not found: curl",  # pacman
        "ERROR: unable to select packages:",  # apk
        "No provider of 'libcurl-devel' found.",  # zypper
        "emerge: there are no ebuilds to satisfy \"net-misc/curl\".",  # emerge
    ],
)
def test_package_not_found_matches_each_manager(llama_cpp, line):
    assert any(marker in line for marker in llama_cpp.PACKAGE_NOT_FOUND)


@pytest.mark.parametrize(
    "line",
    [
        "Setting up libcurl4-openssl-dev (8.5.0-2ubuntu10) ...",
        "(1/3) Installing curl-dev (8.9.1-r1)",
        "Package 'libgomp1' is not found in cache, downloading.",  # a plain "not found" that is not an error
        "  Fetching https://... (cache not found, retrying)",
    ],
)
def test_package_not_found_ignores_ordinary_output(llama_cpp, line):
    assert not any(marker in line for marker in llama_cpp.PACKAGE_NOT_FOUND)

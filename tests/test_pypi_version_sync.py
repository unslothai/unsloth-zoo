# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Checks __version__ on main is >= the latest PyPI release so a rewind cannot ship a smaller version; networked,
skipped if PyPI is unreachable (CI allowlists pypi.org:443)."""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request

import pytest


PACKAGE_NAME = "unsloth_zoo"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"


def _parse_version(value: str):
    """Best-effort PEP 440 parse via packaging.version, else a numeric-aware split safe for the simple X.Y.Z[.devN|rcN|postN] shape zoo uses."""
    try:
        from packaging.version import Version
        return Version(value)
    except Exception:
        parts = value.split("+", 1)[0].split("-", 1)[0]
        nums, _, _suffix = parts.partition("rc")
        ints = []
        for token in nums.split("."):
            try:
                ints.append(int(token))
            except ValueError:
                ints.append(0)
        return tuple(ints)


def _get_pypi_latest_version(timeout: float = 10.0):
    """Fetches unsloth_zoo's latest PyPI version from the JSON API; returns None on network failure so the test skips rather than fails."""
    request = urllib.request.Request(
        PYPI_JSON_URL,
        headers = {"User-Agent": "unsloth-zoo-ci/test_pypi_version_sync"},
    )
    try:
        with urllib.request.urlopen(request, timeout = timeout) as response:
            metadata = json.load(response)
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError):
        return None
    info = metadata.get("info") or {}
    version = info.get("version")
    if not version:
        return None
    return version


def _get_main_version():
    """Reads __version__ from unsloth_zoo/__init__.py without importing the package, since import triggers device detection on runners without CUDA/XPU/HIP."""
    import pathlib
    import re

    # repo root is parents[1] (parents[0] = tests/).
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    init_py = repo_root / "unsloth_zoo" / "__init__.py"
    text = init_py.read_text(encoding = "utf-8")
    match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']',
        text,
        re.MULTILINE,
    )
    if not match:
        raise AssertionError(
            f"Could not find `__version__ = '...'` in {init_py}. The "
            "pyproject.toml dynamic-version stanza reads from this "
            "attribute; if it's gone, the wheel build also breaks."
        )
    return match.group(1)


def test_pypi_version_is_not_ahead_of_main():
    """Fails when a version bump shipped on a release branch never made it back to main; cherry-pick it to main before the next release."""
    if os.environ.get("UNSLOTH_SKIP_PYPI_VERSION_SYNC"):
        pytest.skip(
            "UNSLOTH_SKIP_PYPI_VERSION_SYNC env var set -- bypassed "
            "(should NEVER be set in default CI; use only for "
            "transient pypi.org outages)."
        )

    main_version_str = _get_main_version()
    pypi_version_str = _get_pypi_latest_version()
    if pypi_version_str is None:
        pytest.skip(
            "Could not reach pypi.org -- skipping version-sync check. "
            "(harden-runner allowlist on default CI runners includes "
            "pypi.org:443, so this should never fire in CI; only on "
            "fully-offline dev machines.)"
        )

    main_v = _parse_version(main_version_str)
    pypi_v = _parse_version(pypi_version_str)

    assert main_v >= pypi_v, (
        f"VERSION REGRESSION DETECTED.\n"
        f"  unsloth_zoo/__init__.__version__ on main: {main_version_str}\n"
        f"  latest version on PyPI:                   {pypi_version_str}\n"
        f"\n"
        f"PyPI is AHEAD of main. The next `python -m build && twine "
        f"upload` from main would publish {main_version_str}, which "
        f"is LESS than {pypi_version_str} -- breaking pip's "
        f"`--upgrade` resolver for every user.\n"
        f"\n"
        f"Resolution: cherry-pick the version bump from the release "
        f"branch back to main before opening this PR."
    )


def test_main_version_string_is_parseable():
    """Checks the __version__ string in unsloth_zoo/__init__.py parses as PEP 440, catching typos and accidental versions like 1.0 without a patch part."""
    main_version_str = _get_main_version()
    try:
        from packaging.version import Version
        Version(main_version_str)
    except ImportError:
        pytest.skip("packaging not installed -- can't validate PEP 440 shape")
    except Exception as exc:
        raise AssertionError(
            f"unsloth_zoo/__init__.__version__ is not a valid PEP 440 "
            f"version: {main_version_str!r} ({exc})"
        )

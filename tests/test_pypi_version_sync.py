# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""PyPI version-sync regression test.

Catches the class of bug where someone bumps `__version__` on a
release branch, ships a wheel to PyPI, then a subsequent PR
accidentally REWINDS `__version__` on main below what PyPI
already serves. The next release would then publish a SMALLER
version than the previous one, breaking pip's resolver for every
user who runs `pip install --upgrade unsloth_zoo`.

Invariant pinned: `__version__ on main >= latest published version
on PyPI`. (Equality is OK -- nothing has changed since the last
release. Greater-than is OK -- we're preparing the next release.
Less-than is the bug.)

Networked. Skipped automatically when PyPI is unreachable, but on
GitHub Actions CI the harden-runner allowlist explicitly includes
`pypi.org:443`, so the skip should never fire there.
"""

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
    """Best-effort PEP 440 parser. Falls back to packaging.version if
    available; otherwise a lexicographic numeric-aware split that's
    safe for the simple `X.Y.Z[.devN|rcN|postN]` shape zoo uses.
    """
    try:
        from packaging.version import Version
        return Version(value)
    except Exception:
        # Tuple of ints + suffix string -- monotonically orderable for
        # versions of the same shape. Good enough for the simple bump
        # case this test guards against.
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
    """Fetch the latest published version of unsloth_zoo from PyPI's
    JSON API. Returns None on network failure (we skip the test
    rather than fail it -- the gate only matters when PyPI is
    reachable).
    """
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
    """Read the `__version__` attribute zoo's setuptools dynamic
    metadata reads from `unsloth_zoo/__init__.py`. We do this WITHOUT
    importing `unsloth_zoo` because importing the package on a
    CI runner without CUDA / XPU / HIP fires the device-type
    detection which is intercepted by tests/conftest.py only when
    pytest is collecting -- safer to read the source file directly.
    """
    import pathlib
    import re

    # __file__ is .../<repo-root>/tests/test_pypi_version_sync.py
    # so the repo root is parents[1] (parents[0] = tests/).
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
    """`__version__` on main MUST be >= latest published version on PyPI.

    If this fails, someone bumped __version__ on a release branch +
    published to PyPI, but the bump didn't make it back to main.
    Resolution: cherry-pick the version bump back to main BEFORE
    the next release.
    """
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
    """The version string in unsloth_zoo/__init__.py must be a valid
    PEP 440 version. Catches typos / accidental "1.0" without patch.
    """
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

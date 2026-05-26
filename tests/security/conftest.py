"""Shared fixtures for the security regression suite.

The scanner scripts under audit are designed to be offline-safe. Pin
that invariant by autouse-installing a function-scoped network blocker
that refuses any non-loopback `socket.connect()` from inside the test
process. If a future test (or a scanner regression) accidentally tries
to reach the public internet, pytest fails loudly instead of leaking
the request.

Scope is intentionally ``function`` rather than ``session``: the swap
mutates a module-global (``socket.socket``), and a session-scoped swap
keeps the patch live for every test pytest runs after the first
security test in the same session -- which silently broke every
network-using test elsewhere in the tree (e.g.
``tests/test_upstream_pinned_symbols_transformers.py`` which fetches
HF modeling source over HTTPS). Per-function setup/teardown costs
~10us and contains the blast radius to security tests only.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest


# Make `scripts/` importable as a package so tests can grab the scanner
# constants directly. The repo root sits two levels above this file.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_LOOPBACK_PREFIXES = ("127.", "::1", "localhost")


def _is_loopback(host: str | bytes) -> bool:
    if isinstance(host, bytes):
        try:
            host = host.decode("utf-8")
        except UnicodeDecodeError:
            return False
    if not host:
        return False
    host = host.strip()
    if host in {"::1", "localhost", "0.0.0.0"}:
        return True
    return host.startswith("127.")


class _BlockedSocket(socket.socket):
    """Socket subclass that refuses any non-loopback connect()."""

    def connect(self, address):  # type: ignore[override]
        host = None
        if isinstance(address, tuple) and address:
            host = address[0]
        if not _is_loopback(host or ""):
            raise RuntimeError(
                f"network access blocked by tests/security/conftest.py "
                f"(attempted connect to {address!r}); the scanner suite "
                "must run fully offline"
            )
        return super().connect(address)

    def connect_ex(self, address):  # type: ignore[override]
        host = None
        if isinstance(address, tuple) and address:
            host = address[0]
        if not _is_loopback(host or ""):
            raise RuntimeError(
                f"network access blocked by tests/security/conftest.py "
                f"(attempted connect_ex to {address!r})"
            )
        return super().connect_ex(address)


@pytest.fixture(scope = "function", autouse = True)
def network_blocker():
    """Function-scoped fixture; replaces `socket.socket` with a blocker.

    Yields nothing; the swap is the side effect. Restored at teardown
    so the *next* test (security or otherwise) sees the real socket.
    Session scope was a footgun: it leaked the patch into every
    network-using test in the parent ``tests/`` tree once a single
    security test ran. See the module docstring for the regression.
    """
    original = socket.socket
    socket.socket = _BlockedSocket  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket = original  # type: ignore[assignment]


@pytest.fixture(scope = "session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope = "session")
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"

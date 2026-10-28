"""Shared fixtures for security tests: an autouse, function-scoped network blocker refusing non-loopback
socket.connect(); function scope because a session-scoped socket.socket swap broke network tests elsewhere."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest


# Add the repo root to sys.path so scripts/ imports as a package and tests can grab scanner constants (root is two levels above this file).
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
    """Swaps socket.socket for a blocker and restores it at teardown; function scope because session scope leaked the
    patch into every network-using test in the parent tests/ tree."""
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

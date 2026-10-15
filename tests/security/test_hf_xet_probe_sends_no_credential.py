# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""The Xet reachability probe must not carry the user's Hugging Face token.

A hand-rolled `urllib` Request ignores `HF_HUB_DISABLE_IMPLICIT_TOKEN` and keeps
`Authorization` across a cross-host 3xx. Loopback only, enforced by the package conftest.
"""

import importlib.util
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

DUMMY_TOKEN = "hf_dummyTokenForTestsOnly000000000000"


def _load_hf_xet_health():
    """Load the module standalone: importing `unsloth_zoo` runs import-time device detection.

    Its one relative import needs a parent package, which the stub supplies.
    """
    import types

    pkg_name = "unsloth_zoo_stub_for_xet_test"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(REPO_ROOT / "unsloth_zoo")]
        sys.modules[pkg_name] = pkg

    name = f"{pkg_name}.hf_xet_health"
    path = REPO_ROOT / "unsloth_zoo" / "hf_xet_health.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _RecordingHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the pytest output clean
        pass

    def _record(self):
        self.server.seen.append((self.path, self.headers.get("Authorization")))

    def do_GET(self):
        self._record()
        if self.path.startswith("/redirect"):
            self.send_response(302)
            self.send_header("Location", self.server.redirect_to)
            self.end_headers()
            return
        if self.server.status != 200:
            # Sized body: a truncated one is a transport error, which is a different code path.
            body = b'{"error": "Unauthorized"}'
            self.send_response(self.server.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = b'{"casUrl": ""}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        self._record()
        self.send_response(200)
        self.end_headers()


def _serve():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
    server.seen = []
    server.redirect_to = ""
    server.status = 200
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    return server


@pytest.fixture
def origin():
    server = _serve()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def elsewhere():
    server = _serve()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def module(monkeypatch, origin):
    mod = _load_hf_xet_health()
    host, port = origin.server_address[0], origin.server_address[1]
    monkeypatch.setattr(mod, "_endpoint", lambda: f"http://{host}:{port}", raising = False)
    return mod


def test_probe_sends_no_authorization_even_with_hf_token_set(module, monkeypatch, origin):
    monkeypatch.setenv("HF_TOKEN", DUMMY_TOKEN)
    module._probe_cas_reachable_inner()
    assert origin.seen, "the probe did not reach the fixture"
    for path, auth in origin.seen:
        assert auth is None, f"probe sent Authorization to {path}"


def test_probe_sends_no_authorization_across_a_cross_host_redirect(
    module, monkeypatch, origin, elsewhere
):
    """The regression this test exists for: the header used to survive the 3xx."""
    monkeypatch.setenv("HF_TOKEN", DUMMY_TOKEN)
    other_host, other_port = elsewhere.server_address[0], elsewhere.server_address[1]
    origin.redirect_to = f"http://{other_host}:{other_port}/landed"
    monkeypatch.setattr(
        module,
        "_endpoint",
        lambda: f"http://{origin.server_address[0]}:{origin.server_address[1]}/redirect",
        raising = False,
    )

    module._probe_cas_reachable_inner()

    assert elsewhere.seen, "the redirect was not followed, so the test proves nothing"
    for path, auth in elsewhere.seen:
        assert auth is None, f"probe carried Authorization across a redirect to {path}"


def test_probe_does_not_read_a_token_at_all(module, monkeypatch, origin):
    """No credential lookup, so HF_HUB_DISABLE_IMPLICIT_TOKEN cannot be contradicted."""
    # HF_TOKEN must be ABSENT or the old code short-circuits before get_token and the spy proves
    # nothing on the unpatched tree either.
    monkeypatch.delenv("HF_TOKEN", raising = False)
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

    called = []
    hub_utils = pytest.importorskip(
        "huggingface_hub.utils", reason = "the cached-token fallback needs huggingface_hub"
    )
    assert hasattr(hub_utils, "get_token"), "huggingface_hub.utils.get_token is the path guarded"
    monkeypatch.setattr(
        hub_utils, "get_token", lambda *a, **k: called.append(1) or DUMMY_TOKEN
    )

    module._probe_cas_reachable_inner()

    assert origin.seen, "the probe did not reach the fixture"
    assert not called, "the probe still looks up a cached token"
    for path, auth in origin.seen:
        assert auth is None, f"probe sent Authorization to {path}"


@pytest.mark.parametrize("with_env_token", [True, False])
def test_a_401_is_inconclusive_and_still_carries_no_credential(
    module, monkeypatch, origin, with_env_token
):
    """A 401 proves the endpoint is REACHABLE, and both arms must agree: nothing is consulted."""
    if with_env_token:
        monkeypatch.setenv("HF_TOKEN", DUMMY_TOKEN)
    else:
        monkeypatch.delenv("HF_TOKEN", raising = False)
    origin.status = 401

    ok, reason = module._probe_cas_reachable_inner()

    assert origin.seen, "the probe did not reach the fixture"
    assert ok is None, reason
    assert "inconclusive" in reason
    for path, auth in origin.seen:
        assert auth is None, f"probe sent Authorization to {path}"


def test_source_carries_no_authorization_header():
    """A lint-shaped guard: nothing in this module may add an Authorization header.

    EXECUTABLE strings only: a text scan must exempt the comment explaining the removal, and
    that exemption then blesses every other mention, a real header included.
    """
    import ast

    path = REPO_ROOT / "unsloth_zoo" / "hf_xet_health.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))

    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in docstrings:
            continue
        lowered = node.value.lower()
        assert "authorization" not in lowered, (
            f"hf_xet_health.py:{node.lineno} names Authorization in code; "
            "the probe must stay anonymous"
        )
        assert "bearer " not in lowered, (
            f"hf_xet_health.py:{node.lineno} formats a Bearer credential again"
        )

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Dataset media fetches must not hand ~/.netrc credentials to a host the dataset picked: never the `default`
entry, never over plain http. An https host the netrc names explicitly, and URL-embedded credentials, still work."""

from __future__ import annotations

import base64
import http.server
import threading

import pytest

vision_utils = pytest.importorskip("unsloth_zoo.vision_utils")


@pytest.fixture
def server():
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target = httpd.serve_forever, daemon = True)
    thread.start()
    try:
        yield f"127.0.0.1:{httpd.server_address[1]}", seen
    finally:
        httpd.shutdown()
        httpd.server_close()


def _netrc(tmp_path, monkeypatch, text):
    path = tmp_path / "netrc"
    path.write_text(text)
    path.chmod(0o600)
    monkeypatch.setenv("NETRC", str(path))
    # The test server is on loopback.
    monkeypatch.setenv(vision_utils.UNSLOTH_ALLOW_PRIVATE_URL_FETCH_VAR, "1")


def test_default_netrc_entry_not_sent_to_dataset_host(server, tmp_path, monkeypatch):
    host, seen = server
    _netrc(tmp_path, monkeypatch, "default login someone password hunter2\n")
    assert vision_utils.fetch_remote_media_bytes(f"http://{host}/a.png").getvalue() == b"ok"
    assert seen == [None]


def test_explicit_netrc_entry_not_sent_over_plain_http(server, tmp_path, monkeypatch):
    host, seen = server
    _netrc(tmp_path, monkeypatch, "machine 127.0.0.1 login someone password hunter2\n")
    vision_utils.fetch_remote_media_bytes(f"http://{host}/a.png")
    assert seen == [None]


def test_url_embedded_credentials_still_sent(server, tmp_path, monkeypatch):
    host, seen = server
    _netrc(tmp_path, monkeypatch, "")
    vision_utils.fetch_remote_media_bytes(f"http://user:pw@{host}/a.png")
    assert seen == ["Basic " + base64.b64encode(b"user:pw").decode()]


def test_https_explicit_machine_keeps_netrc_auth(tmp_path, monkeypatch):
    _netrc(tmp_path, monkeypatch, "machine media.example login someone password hunter2\ndefault login d password q\n")
    auth = vision_utils._media_auth("https://media.example/a.png")
    assert tuple(auth) == ("someone", "hunter2")
    assert vision_utils._media_auth("https://other.example/a.png") is vision_utils._no_media_auth


def test_tilde_netrc_path_is_expanded_like_requests(tmp_path, monkeypatch):
    path = tmp_path / "media.netrc"
    path.write_text("machine media.example login someone password hunter2\n")
    path.chmod(0o600)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # expanduser on Windows
    monkeypatch.setenv("NETRC", "~/media.netrc")
    assert tuple(vision_utils._media_auth("https://media.example/a.png")) == ("someone", "hunter2")

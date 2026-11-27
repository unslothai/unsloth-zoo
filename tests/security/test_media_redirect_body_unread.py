# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A redirect hop's body must never be read: requests' `response.next` precompute would otherwise pull every
hop body into memory, outside UNSLOTH_MAX_MEDIA_DOWNLOAD_MB."""

from __future__ import annotations

import http.server
import threading

import pytest

vision_utils = pytest.importorskip("unsloth_zoo.vision_utils")

HOP_MB = 64


@pytest.fixture
def server():
    sent = {"hop_mb": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/hop":
                self.send_response(302)
                self.send_header("Location", "/final")
                self.send_header("Content-Length", str(HOP_MB << 20))
                self.end_headers()
                try:
                    for _ in range(HOP_MB):
                        self.wfile.write(b"A" * (1 << 20))
                        sent["hop_mb"] += 1
                except OSError:
                    pass
            else:
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target = httpd.serve_forever, daemon = True).start()
    try:
        yield f"127.0.0.1:{httpd.server_address[1]}", sent
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_redirect_hop_body_is_not_consumed(server, monkeypatch):
    host, sent = server
    monkeypatch.setenv(vision_utils.UNSLOTH_ALLOW_PRIVATE_URL_FETCH_VAR, "1")
    monkeypatch.setenv(vision_utils.UNSLOTH_MAX_MEDIA_DOWNLOAD_MB_VAR, "1")
    assert vision_utils.fetch_remote_media_bytes(f"http://{host}/hop").getvalue() == b"ok"
    # Only what fits the socket buffers may leave the server; the old path drained all of it.
    assert sent["hop_mb"] < HOP_MB // 2, sent


def test_guarded_session_never_resolves_redirects():
    with vision_utils._guarded_session() as session:
        assert list(session.resolve_redirects(None, None)) == []

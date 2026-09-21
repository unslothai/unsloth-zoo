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

"""The DiffusionGemma shim is a local sidecar, and only local callers may drive it.

Unsloth Studio starts this listener on loopback by itself when a DiffusionGemma
GGUF is selected, with no credential of any kind. Loopback is not a boundary a
browser respects: a page the user happens to be visiting can POST to 127.0.0.1
cross-site with no preflight (a text/plain body is a CORS simple request), and a
hostname rebound to 127.0.0.1 makes the reply readable as well. These tests pin
the request binding that closes both, and the ceiling that stops one caller from
holding the single-sequence decoder's lock for an arbitrary length of time.

Everything here runs against the ASGI app in-process; no socket is opened.
"""

from __future__ import annotations

import pytest

# The shim imports fastapi/uvicorn, which the CI [core] extras do not ship.
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402

from unsloth_zoo.diffusion_studio import shim  # noqa: E402


# The shim reads the Host header, so the client must present a real one.
LOCAL = "http://127.0.0.1:8123"
BODY = {"model": "diffusiongemma", "messages": [{"role": "user", "content": "hi"}]}
# 256 tokens per canvas block, so 64 blocks is a 16384-token answer.
CEILING = 64


@pytest.fixture
def client(monkeypatch):
    """A shim with the GPU child replaced by a recorder, so nothing is launched."""
    calls = []

    def _generate_visual(server, messages, **kwargs):
        calls.append(kwargs)
        return "ok"

    monkeypatch.setattr(shim.V, "generate_visual", _generate_visual)
    monkeypatch.setitem(shim._STATE, "server", object())
    monkeypatch.setitem(shim._STATE, "host", "127.0.0.1")
    with TestClient(shim.app, base_url = LOCAL) as test_client:
        test_client.calls = calls
        yield test_client


# --- A browser page the user visits is not the local user ---

def test_a_cross_site_origin_is_refused(client):
    """The CORS-simple POST: text/plain needs no preflight, so nothing else stops it."""
    response = client.post(
        "/v1/chat/completions",
        headers = {"Origin": "https://evil.example", "Content-Type": "text/plain"},
        content = '{"messages": [{"role": "user", "content": "attacker prompt"}]}',
    )
    assert response.status_code == 403, response.text
    assert client.calls == [], "the decoder ran for a cross-site caller"


def test_a_cross_site_referer_is_refused(client):
    """Fetch modes that omit Origin still carry a Referer."""
    response = client.post(
        "/v1/chat/completions",
        headers = {"Referer": "https://evil.example/page"},
        json = BODY,
    )
    assert response.status_code == 403, response.text
    assert client.calls == []


def test_a_rebound_hostname_is_refused(client):
    """DNS rebinding is how this listener's replies become readable cross-origin.
    The attacker's name resolves to 127.0.0.1, so only the Host tells them apart."""
    for path in ("/health", "/v1/models"):
        response = client.get(path, headers = {"Host": "attacker.example:8123"})
        assert response.status_code == 403, (path, response.text)
    response = client.post(
        "/v1/chat/completions", headers = {"Host": "attacker.example:8123"}, json = BODY,
    )
    assert response.status_code == 403, response.text
    assert client.calls == []


# --- An ordinary local client keeps working ---

def test_a_plain_local_client_is_served(client):
    """curl, the OpenAI SDK and llama.cpp clients send no Origin and a loopback Host."""
    assert client.get("/health").status_code == 200
    assert client.get("/v1/models").status_code == 200
    response = client.post("/v1/chat/completions", json = BODY)
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "ok"


@pytest.mark.parametrize("origin", [
    "http://localhost:3000",     # Unsloth Studio's own frontend, a different port
    "http://127.0.0.1:8123",
    "http://[::1]:8123",
])
def test_a_local_page_is_served(client, origin):
    """Same machine, any port: the shim is not trying to separate local apps."""
    response = client.post(
        "/v1/chat/completions", headers = {"Origin": origin}, json = BODY,
    )
    assert response.status_code == 200, response.text


def test_a_published_listener_still_answers_its_own_name(client, monkeypatch):
    """--host 0.0.0.0 is the operator publishing this deliberately, so the Host is
    whatever name they reach it by. The Origin rule still holds."""
    monkeypatch.setitem(shim._STATE, "host", "0.0.0.0")
    response = client.get("/health", headers = {"Host": "workstation.lan:8123"})
    assert response.status_code == 200, response.text


# --- One caller cannot take the decoder for as long as it likes ---

def test_max_tokens_is_clamped(client):
    """_max_blocks sizes the generation, and the request holds the process-wide
    lock for the whole of it."""
    response = client.post(
        "/v1/chat/completions", json = {**BODY, "max_tokens": 10 ** 9},
    )
    assert response.status_code == 200, response.text
    assert client.calls[0]["max_blocks"] == CEILING
    assert shim.DEFAULT_MAX_BLOCKS == CEILING


def test_an_ordinary_request_is_not_clamped(client):
    """The default 2048 tokens, and anything under the ceiling, is untouched."""
    assert shim._max_blocks({}) == 8
    assert shim._max_blocks({"max_tokens": 4096}) == 16
    assert shim._max_blocks({"max_tokens": 1}) == 1


def test_the_ceiling_is_configurable(client, monkeypatch):
    monkeypatch.setenv("DG_MAX_BLOCKS", "128")
    assert shim._max_blocks({"max_tokens": 10 ** 9}) == 128
    monkeypatch.setenv("DG_MAX_BLOCKS", "not a number")
    assert shim._max_blocks({"max_tokens": 10 ** 9}) == CEILING

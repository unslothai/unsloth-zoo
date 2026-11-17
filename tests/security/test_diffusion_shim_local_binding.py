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

"""Pins the Host and Origin checks of the DiffusionGemma shim (no credential; loopback is no boundary against cross-site
POST or DNS rebinding) and its generation ceiling that stops one caller holding the single-sequence decoder lock;
in-process, no socket"""

from __future__ import annotations

import json

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
        # The real server closes each turn with STATS; a stand-in that omits them would make every finish-reason assertion unfailable.
        blocks = kwargs.get("max_blocks", 8)
        on_stats = kwargs.get("on_stats")
        if on_stats is not None:
            on_stats({"blocks": blocks, "predicted_n": blocks * 256, "prompt_n": 4,
                      "wall_ms": 1.0, "decode_ms": 1.0, "canvas": 256})
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
    """DNS rebinding makes replies readable cross-origin, since the attacker's name resolves to 127.0.0.1; only the Host header tells them apart."""
    for path in ("/health", "/v1/models"):
        response = client.get(path, headers = {"Host": "attacker.example:8123"})
        assert response.status_code == 403, (path, response.text)
    response = client.post(
        "/v1/chat/completions", headers = {"Host": "attacker.example:8123"}, json = BODY,
    )
    assert response.status_code == 403, response.text
    assert client.calls == []


# --- An ordinary local client keeps working ---

def test_an_unparseable_host_is_refused_not_a_500(client):
    """An unparseable Host must be refused with 403: an uncaught urlsplit error would be a 500, and such a host is the one least worth trusting."""
    response = client.post("/v1/chat/completions", json = BODY, headers = {"Host": "[oops]"})
    assert response.status_code == 403, response.text


def test_userinfo_in_the_host_is_refused(client):
    """Not Host syntax: `urlsplit` reads only the part after `@` and calls it local."""
    assert client.post("/v1/chat/completions", json = BODY,
                       headers = {"Host": "evil.example@localhost"}).status_code == 403


def test_a_second_origin_header_cannot_ride_behind_a_local_one(client):
    """A single `.get` answers with the first value; the foreign second is never seen."""
    request = client.build_request("POST", "/v1/chat/completions", json = BODY)
    raw = [(k, v) for k, v in request.headers.raw if k.lower() != b"origin"]
    raw += [(b"origin", b"http://localhost:3000"), (b"origin", b"https://evil.example")]
    request.headers = type(request.headers)(raw)
    assert client.send(request).status_code == 403


@pytest.mark.parametrize("spelling", ["127.1", "2130706433", "0x7f000001"])
def test_a_short_loopback_spelling_is_still_this_machine(spelling):
    """Short loopback spellings such as 127.1 are this machine, since a numeric literal cannot be rebound; Studio's host_policy reads them the same way."""
    assert shim._is_local_name(spelling) is True
    assert shim._is_local_name("1") is False          # 0.0.0.1, not loopback
    assert shim._is_local_name("evil.example") is False


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
    """With --host 0.0.0.0 the operator publishes the listener on purpose, so any Host name is accepted; the Origin rule still applies."""
    monkeypatch.setitem(shim._STATE, "host", "0.0.0.0")
    response = client.get("/health", headers = {"Host": "workstation.lan:8123"})
    assert response.status_code == 200, response.text


# --- One caller cannot take the decoder for as long as it likes ---

def test_max_tokens_is_clamped(client):
    """max_tokens is clamped because _max_blocks sizes the generation and the request holds the process-wide lock for all of it."""
    response = client.post(
        "/v1/chat/completions", json = {**BODY, "max_tokens": 10 ** 9},
    )
    assert response.status_code == 200, response.text
    assert client.calls[0]["max_blocks"] == CEILING
    assert shim.DEFAULT_MAX_BLOCKS == CEILING


def test_an_ordinary_request_is_not_clamped(client):
    """The default 2048 tokens, and anything under the ceiling, is untouched."""
    assert shim._max_blocks({}) == (8, False)
    assert shim._max_blocks({"max_tokens": 4096}) == (16, False)
    assert shim._max_blocks({"max_tokens": 1}) == (1, False)


def test_the_ceiling_is_configurable(client, monkeypatch):
    monkeypatch.setenv("DG_MAX_BLOCKS", "128")
    assert shim._max_blocks({"max_tokens": 10 ** 9}) == (128, True)
    monkeypatch.setenv("DG_MAX_BLOCKS", "not a number")
    assert shim._max_blocks({"max_tokens": 10 ** 9}) == (CEILING, True)


# --- A reply this server cut short does not report a natural stop ---

def test_a_capped_generation_reports_length(client):
    """A capped generation must report finish_reason length, or a truncated answer looks complete and clients skip the continuation they run on other servers."""
    client.calls.clear()
    response = client.post("/v1/chat/completions", json = {**BODY, "max_tokens": 10 ** 9})
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["finish_reason"] == "length"


def test_an_uncapped_generation_still_reports_stop(client):
    """A generation that ends before the ceiling reports stop; reporting length there would be the same bug reversed."""
    response = client.post("/v1/chat/completions", json = BODY)
    assert response.json()["choices"][0]["finish_reason"] == "stop"


def test_a_capped_request_that_stops_early_still_reports_stop(client, monkeypatch):
    """A capped request that stops early reports stop: fewer blocks than the ceiling means the ceiling did not end generation."""
    def _short(server, messages, seed = 3407, max_blocks = 8, on_frame = None,
               on_commit = None, on_stats = None, tools = None):
        if on_stats is not None:
            on_stats({"blocks": 1, "predicted_n": 256, "prompt_n": 4})
        return "done early"
    monkeypatch.setattr(shim.V, "generate_visual", _short)
    response = client.post("/v1/chat/completions", json = {**BODY, "max_tokens": 10 ** 9})
    assert response.json()["choices"][0]["finish_reason"] == "stop"


def test_a_capped_streaming_generation_reports_length(client):
    """Streaming emits its own terminal chunk, and delta readers see only its finish_reason, so the capped stream must report length too."""
    response = client.post(
        "/v1/chat/completions", json = {**BODY, "max_tokens": 10 ** 9, "stream": True},
    )
    assert response.status_code == 200, response.text
    reasons = [
        choice["finish_reason"]
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
        for choice in json.loads(line[6:]).get("choices", [])
        if choice.get("finish_reason")
    ]
    assert reasons == ["length"], response.text


def test_an_uncapped_streaming_generation_reports_stop(client):
    response = client.post("/v1/chat/completions", json = {**BODY, "stream": True})
    reasons = [
        choice["finish_reason"]
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
        for choice in json.loads(line[6:]).get("choices", [])
        if choice.get("finish_reason")
    ]
    assert reasons == ["stop"], response.text

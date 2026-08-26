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

"""Dataset supplied image/video URLs must not reach internal addresses.

A VLM dataset row can carry `{"type": "image_url", "image_url": "http://..."}`,
and the collators resolve it server side through fetch_image()/fetch_video().
Without a destination check that lets a dataset drive requests from the
training worker to localhost, the LAN, or the cloud metadata endpoint. Public
URLs and every local-file form must keep working unchanged.
"""

from __future__ import annotations

import glob
import io
import os
import tempfile

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from unsloth_zoo import vision_utils  # noqa: E402


BLOCKED_URLS = [
    "http://127.0.0.1:9/x.png",
    "http://localhost:8080/x.png",
    "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
    "http://metadata.google.internal/computeMetadata/v1/",
    "http://10.0.0.5/x.png",
    "http://192.168.1.10/x.png",
    "http://172.16.3.4/x.png",
    "http://172.31.255.254/x.png",
    "http://[::1]:8000/x.png",
    "http://[fd00::1]/x.png",
    "http://[fe80::1]/x.png",
    "http://[::ffff:127.0.0.1]/x.png",
    "http://0.0.0.0/x.png",
    "http://127.1/x.png",
    "http://2130706433/x.png",
    "http://user:pass@127.0.0.1/x.png",
    # RFC 6598 carrier-grade NAT: ipaddress.is_private is False for it on
    # every Python from 3.9 to 3.13, so the guard needs its own entry.
    "http://100.64.0.1/x.png",
    "http://224.0.0.1/x.png",
    "http://240.0.0.1/x.png",
]


@pytest.fixture(autouse=True)
def _default_policy(monkeypatch):
    monkeypatch.delenv("UNSLOTH_ALLOW_PRIVATE_URL_FETCH", raising=False)
    monkeypatch.delenv("UNSLOTH_MAX_MEDIA_DOWNLOAD_MB", raising=False)
    vision_utils._is_blocked_address.cache_clear()
    yield
    vision_utils._is_blocked_address.cache_clear()


@pytest.fixture
def no_network(monkeypatch):
    """Fail loudly if anything actually issues a request."""
    calls = []

    def boom(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError(f"network call escaped the guard: {args} {kwargs}")

    monkeypatch.setattr(vision_utils.requests, "get", boom)
    return calls


def _png_bytes(size=(32, 32), color=(1, 2, 3)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class _FakeResponse:
    is_redirect = False
    is_permanent_redirect = False

    def __init__(self, content=b"", status=200, headers=None, redirect_to=None):
        self._content = content
        self.status_code = status
        self.headers = headers or {}
        if redirect_to is not None:
            self.is_redirect = True
            self.headers["location"] = redirect_to

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"HTTP {self.status_code}")

    def close(self):
        pass


@pytest.fixture
def public_dns(monkeypatch):
    """Resolve every host to a public address unless it is a literal IP."""
    import ipaddress
    import socket

    real = socket.getaddrinfo

    def fake_getaddrinfo(host, *args, **kwargs):
        try:
            ipaddress.ip_address(host)
        except ValueError:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
        return real(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


@pytest.mark.parametrize("url", BLOCKED_URLS)
def test_fetch_image_blocks_internal_urls(url, no_network):
    with pytest.raises(ValueError):
        vision_utils.fetch_image({"image": url})
    assert no_network == []


@pytest.mark.parametrize("url", BLOCKED_URLS)
def test_fetch_image_blocks_internal_urls_via_image_url_dict(url, no_network):
    with pytest.raises(ValueError):
        vision_utils.fetch_image({"image_url": {"url": url}})
    assert no_network == []


def test_fetch_image_blocks_non_http_scheme_in_url_dict(no_network):
    # The dict {"url": ...} branch previously had no scheme check at all.
    with pytest.raises(ValueError):
        vision_utils.fetch_image({"image": {"url": "file:///etc/passwd"}})
    assert no_network == []


def test_process_vision_info_blocks_internal_urls(no_network):
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": "http://169.254.169.254/metadata"},
        ],
    }]
    with pytest.raises(ValueError):
        vision_utils.process_vision_info(messages)
    assert no_network == []


def test_public_url_is_still_fetched(monkeypatch, public_dns):
    seen = []

    def fake_get(url, **kwargs):
        seen.append(url)
        return _FakeResponse(_png_bytes())

    monkeypatch.setattr(vision_utils.requests, "get", fake_get)

    image = vision_utils.fetch_image({"image": "https://example.com/cat.png"})
    assert seen == ["https://example.com/cat.png"]
    assert image.size[0] > 0 and image.size[1] > 0


def test_redirect_to_internal_address_is_blocked(monkeypatch, public_dns):
    seen = []

    def fake_get(url, **kwargs):
        seen.append(url)
        return _FakeResponse(redirect_to="http://127.0.0.1:9/secret.png")

    monkeypatch.setattr(vision_utils.requests, "get", fake_get)

    with pytest.raises(ValueError):
        vision_utils.fetch_image({"image": "https://example.com/redir.png"})
    assert seen == ["https://example.com/redir.png"], "the internal hop must not be requested"


def test_size_cap_is_enforced(monkeypatch, public_dns):
    monkeypatch.setenv("UNSLOTH_MAX_MEDIA_DOWNLOAD_MB", "0.0001")  # ~100 bytes
    monkeypatch.setattr(
        vision_utils.requests, "get",
        lambda url, **kwargs: _FakeResponse(_png_bytes((256, 256), (7, 8, 9))),
    )
    with pytest.raises(ValueError):
        vision_utils.fetch_image({"image": "https://example.com/big.png"})


def test_opt_out_env_var_allows_private_urls(monkeypatch):
    monkeypatch.setenv("UNSLOTH_ALLOW_PRIVATE_URL_FETCH", "1")
    seen = []

    def fake_get(url, **kwargs):
        seen.append(url)
        return _FakeResponse(_png_bytes())

    monkeypatch.setattr(vision_utils.requests, "get", fake_get)

    image = vision_utils.fetch_image({"image": "http://127.0.0.1:8000/local.png"})
    assert seen == ["http://127.0.0.1:8000/local.png"]
    assert image.size[0] > 0


def test_local_paths_and_data_uris_are_untouched(tmp_path, no_network):
    path = tmp_path / "img.png"
    path.write_bytes(_png_bytes(color=(9, 9, 9)))

    assert vision_utils.fetch_image({"image": str(path)}).size[0] > 0
    assert vision_utils.fetch_image({"image": path.as_uri()}).size[0] > 0
    assert vision_utils.fetch_image({"image": {"path": str(path)}}).size[0] > 0
    assert vision_utils.fetch_image({"image": {"bytes": _png_bytes()}}).size[0] > 0

    import base64
    encoded = base64.b64encode(_png_bytes()).decode()
    assert vision_utils.fetch_image({"image": f"data:image/png;base64,{encoded}"}).size[0] > 0


@pytest.mark.parametrize(
    "address,blocked",
    [
        ("127.0.0.1", True), ("10.1.2.3", True), ("172.20.0.1", True),
        ("192.168.0.1", True), ("169.254.169.254", True), ("100.64.0.1", True),
        ("0.0.0.0", True), ("224.0.0.1", True), ("240.0.0.1", True),
        ("192.0.0.1", True), ("198.18.0.1", True), ("::1", True),
        ("fd00::1", True), ("fe80::1", True), ("::ffff:10.0.0.1", True), ("::", True),
        ("1.1.1.1", False), ("8.8.8.8", False), ("93.184.216.34", False),
        ("140.82.121.4", False), ("2606:4700:4700::1111", False),
        ("2001:4860:4860::8888", False),
    ],
)
def test_address_classification_is_explicit(address, blocked):
    """Pinned so a stdlib change to ipaddress cannot silently widen the guard."""
    import ipaddress
    assert vision_utils._is_blocked_ip(ipaddress.ip_address(address)) is blocked


def test_host_lookup_is_cached(monkeypatch):
    calls = []
    real = __import__("socket").getaddrinfo

    def counting(host, *args, **kwargs):
        calls.append(host)
        return real("127.0.0.1", *args, **kwargs)

    monkeypatch.setattr(__import__("socket"), "getaddrinfo", counting)
    vision_utils._is_blocked_address.cache_clear()
    for _ in range(20):
        assert vision_utils._is_blocked_address("repeated.example") is True
    assert len(calls) == 1, "the collator must not re-resolve once per image"


def test_response_is_closed_on_every_path(monkeypatch, public_dns):
    closed = []

    class Tracking(_FakeResponse):
        def close(self):
            closed.append(True)

    monkeypatch.setattr(
        vision_utils.requests, "get",
        lambda url, **kwargs: Tracking(_png_bytes()),
    )
    vision_utils.fetch_image({"image": "https://example.com/a.png"})
    assert closed, "a streaming response must not be left open"


def test_fetch_video_rejects_internal_url_before_touching_a_backend(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("video backend must not be reached")

    monkeypatch.setattr(vision_utils, "get_video_reader_backend", boom)
    with pytest.raises(ValueError):
        vision_utils.fetch_video({"video": "http://127.0.0.1:9/clip.mp4"})


@pytest.mark.parametrize(
    "url",
    [
        "http://%31%32%37.0.0.1:8080/x.png",
        "http://127%2e0%2e0%2e1:8080/x.png",
        "http://%6c%6f%63%61%6c%68%6f%73%74:8080/x.png",
        "http://example.com\\@127.0.0.1/x.png",
        "http://%31%32%37.0.0.1/x.png",
    ],
)
def test_percent_encoded_authority_is_refused(url, no_network):
    """urlparse leaves the authority encoded while requests decodes it before
    connecting, so an encoded host would be checked as an unresolvable name and
    then fetched from 127.0.0.1."""
    with pytest.raises(ValueError):
        vision_utils.fetch_image({"image": url})
    assert no_network == []


def test_percent_escapes_in_the_path_are_still_fine(monkeypatch, public_dns):
    seen = []
    monkeypatch.setattr(
        vision_utils.requests, "get",
        lambda url, **kwargs: seen.append(url) or _FakeResponse(_png_bytes()),
    )
    image = vision_utils.fetch_image({"image": "https://example.com/a%20b/logo%20black.png"})
    assert image.size[0] > 0 and seen == ["https://example.com/a%20b/logo%20black.png"]


@pytest.mark.parametrize("value", ["inf", "INF", "nan", "1e400", "-inf", "abc", "0"])
def test_non_finite_size_limits_do_not_break_every_fetch(monkeypatch, public_dns, value):
    """float() accepts inf and nan, int() then refuses them, which used to make
    every remote image raise before its request was issued."""
    monkeypatch.setenv("UNSLOTH_MAX_MEDIA_DOWNLOAD_MB", value)
    monkeypatch.setattr(
        vision_utils.requests, "get",
        lambda url, **kwargs: _FakeResponse(_png_bytes()),
    )
    assert vision_utils.fetch_image({"image": "https://example.com/a.png"}).size[0] > 0


def test_video_download_is_guarded_not_delegated(monkeypatch, public_dns):
    """The decoders follow redirects with no way to refuse, so a server that
    answers our probe cleanly could still bounce them internally. The bytes are
    fetched here instead and the decoder only ever sees a local file."""
    hops = []

    def fake_get(url, **kwargs):
        hops.append(url)
        if url.endswith("/public.mp4"):
            return _FakeResponse(redirect_to="http://127.0.0.1:9/internal.mp4")
        return _FakeResponse(b"")

    monkeypatch.setattr(vision_utils.requests, "get", fake_get)
    monkeypatch.setattr(
        vision_utils, "get_video_reader_backend",
        lambda: pytest.fail("the decoder must not be reached"),
    )
    with pytest.raises(ValueError):
        vision_utils.fetch_video({"video": "https://example.com/public.mp4"})
    assert hops == ["https://example.com/public.mp4"]


def test_video_decoder_receives_a_local_file_not_a_url(monkeypatch, public_dns, tmp_path):
    """Even for a wholly public chain the decoder gets a downloaded file, so a
    second request it makes cannot be answered differently."""
    seen = {}
    payload = b"\x00\x00\x00\x20ftypisom fake container bytes"

    def fake_get(url, **kwargs):
        if url.endswith("/public.mp4"):
            return _FakeResponse(redirect_to="https://cdn.example.com/real.mp4")
        return _FakeResponse(payload)

    monkeypatch.setattr(vision_utils.requests, "get", fake_get)

    def backend(ele):
        seen["path"] = ele["video"]
        seen["bytes"] = open(ele["video"], "rb").read()
        raise RuntimeError("stop here, the handed-over path is what we assert")

    monkeypatch.setattr(vision_utils, "get_video_reader_backend", lambda: "torchvision")
    monkeypatch.setitem(vision_utils.VIDEO_READER_BACKENDS, "torchvision", backend)
    with pytest.raises(Exception):
        vision_utils.fetch_video({"video": "https://example.com/public.mp4"})

    assert not seen["path"].startswith("http"), seen["path"]
    assert seen["path"].endswith(".mp4"), "the container extension must survive"
    assert seen["bytes"] == payload
    assert not os.path.exists(seen["path"]), "the temp file must be cleaned up"


def test_video_download_honours_the_size_cap(monkeypatch, public_dns):
    monkeypatch.setenv("UNSLOTH_MAX_MEDIA_DOWNLOAD_MB", "0.0001")
    monkeypatch.setattr(
        vision_utils.requests, "get",
        lambda url, **kwargs: _FakeResponse(b"x" * 4096),
    )
    monkeypatch.setattr(
        vision_utils, "get_video_reader_backend",
        lambda: pytest.fail("not reached"),
    )
    with pytest.raises(ValueError):
        vision_utils.fetch_video({"video": "https://example.com/big.mp4"})


def test_failed_video_download_leaves_no_temp_file(monkeypatch, public_dns, tmp_path):
    import glob
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path), raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(
        vision_utils.requests, "get",
        lambda url, **kwargs: _FakeResponse(b"", status=500),
    )
    monkeypatch.setattr(
        vision_utils, "get_video_reader_backend",
        lambda: pytest.fail("not reached"),
    )
    with pytest.raises(Exception):
        vision_utils.fetch_video({"video": "https://example.com/broken.mp4"})
    assert glob.glob(str(tmp_path / "unsloth_media_*")) == []


def test_fetch_video_does_not_mutate_the_caller_dict(monkeypatch, public_dns):
    monkeypatch.setattr(
        vision_utils.requests, "get",
        lambda url, **kwargs: _FakeResponse(b""),
    )
    def backend(ele):
        raise RuntimeError("decoded nothing, that is fine")

    monkeypatch.setattr(vision_utils, "get_video_reader_backend", lambda: "torchvision")
    monkeypatch.setitem(vision_utils.VIDEO_READER_BACKENDS, "torchvision", backend)
    ele = {"video": "https://example.com/clip.mp4"}
    try:
        vision_utils.fetch_video(ele)
    except Exception:
        pass
    assert ele["video"] == "https://example.com/clip.mp4"

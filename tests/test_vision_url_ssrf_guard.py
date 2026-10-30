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

"""Dataset-supplied image/video URLs must not reach localhost, the LAN or cloud metadata via
fetch_image()/fetch_video(); public URLs and local files must still work."""

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
    # RFC 6598 carrier-grade NAT: ipaddress.is_private is False for it on every Python from 3.9 to 3.13, so the guard needs its own entry.
    "http://100.64.0.1/x.png",
    "http://224.0.0.1/x.png",
    "http://240.0.0.1/x.png",
]


@pytest.fixture(autouse=True)
def _default_policy(monkeypatch):
    monkeypatch.delenv("UNSLOTH_ALLOW_PRIVATE_URL_FETCH", raising=False)
    monkeypatch.delenv("UNSLOTH_MAX_MEDIA_DOWNLOAD_MB", raising=False)
    yield


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


class _FakeSession:
    """Forwards to requests.get so tests can keep patching that one symbol."""

    instances = []

    def __init__(self):
        type(self).instances.append(self)
        self.closed = False
        self.mounted = {}

    def mount(self, prefix, adapter):
        self.mounted[prefix] = adapter

    def get(self, url, **kwargs):
        return vision_utils.requests.get(url, **kwargs)

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.fixture(autouse=True)
def _fake_session(monkeypatch):
    _FakeSession.instances = []
    monkeypatch.setattr(vision_utils.requests, "Session", _FakeSession)
    return _FakeSession


def _decoder_must_not_run(monkeypatch):
    """Backend selection happens before the download, so the assertion has to sit on the decoder itself rather than on the selector."""
    def boom(ele):
        raise AssertionError(f"the decoder must not be reached, got {ele['video']!r}")

    monkeypatch.setattr(vision_utils, "get_video_reader_backend", lambda: "torchvision")
    monkeypatch.setitem(vision_utils.VIDEO_READER_BACKENDS, "torchvision", boom)


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


def test_host_resolution_is_not_cached(monkeypatch):
    """Resolution must not be cached: a host that resolved publicly once would keep passing validation while the connection resolved somewhere else."""
    calls = []
    real = __import__("socket").getaddrinfo

    def counting(host, *args, **kwargs):
        calls.append(host)
        return real("127.0.0.1", *args, **kwargs)

    monkeypatch.setattr(__import__("socket"), "getaddrinfo", counting)
    for _ in range(3):
        assert vision_utils._is_blocked_address("repeated.example") is True
    assert len(calls) == 3, "each validation must use a fresh answer"


def test_an_absolute_name_is_resolved_absolutely(monkeypatch):
    """A trailing dot makes the name absolute: _normalize_host strips it for pin matching, but the lookup must keep the
    dot or search domains (ndots:5 on Kubernetes) answer."""
    asked = []
    real = __import__("socket").getaddrinfo

    def recording(host, *args, **kwargs):
        asked.append(host)
        return real("93.184.216.34", *args, **kwargs)

    monkeypatch.setattr(__import__("socket"), "getaddrinfo", recording)
    vision_utils._check_fetchable_url("https://cdn.example./asset.png")
    assert asked == ["cdn.example."], (
        f"the absolute marker was dropped before resolution: {asked}"
    )

    # And the pin still compares against the spelling a client puts on the wire.
    assert vision_utils._normalize_host("cdn.example.") == "cdn.example"


def test_a_relative_name_is_left_alone(monkeypatch):
    """The other half: nothing gains a dot it did not have."""
    asked = []
    real = __import__("socket").getaddrinfo

    def recording(host, *args, **kwargs):
        asked.append(host)
        return real("93.184.216.34", *args, **kwargs)

    monkeypatch.setattr(__import__("socket"), "getaddrinfo", recording)
    vision_utils._check_fetchable_url("https://cdn.example/asset.png")
    assert asked == ["cdn.example"], asked


def test_resolution_failure_is_not_cached_either(monkeypatch):
    calls = []

    def failing(host, *args, **kwargs):
        calls.append(host)
        raise OSError("no such host")

    monkeypatch.setattr(__import__("socket"), "getaddrinfo", failing)
    for _ in range(3):
        assert vision_utils._resolve_host("missing.example") is None
    assert len(calls) == 3


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


def test_fetch_video_rejects_internal_url_before_touching_a_decoder(monkeypatch, no_network):
    _decoder_must_not_run(monkeypatch)
    with pytest.raises(ValueError):
        vision_utils.fetch_video({"video": "http://127.0.0.1:9/clip.mp4"})
    assert no_network == []


@pytest.mark.parametrize(
    "url",
    ["HTTP://127.0.0.1:9/clip.mp4", "Https://169.254.169.254/clip.mp4"],
)
def test_fetch_video_guards_upper_case_http_schemes(monkeypatch, no_network, url):
    _decoder_must_not_run(monkeypatch)
    with pytest.raises(ValueError):
        vision_utils.fetch_video({"video": url})
    assert no_network == []


@pytest.mark.parametrize(
    "url",
    [
        "rtsp://10.0.0.5/stream",
        "tcp://127.0.0.1:9",
        "ftp://10.0.0.5/clip.mp4",
        "smb://host/share/a.mp4",
        "tcp:127.0.0.1:9",
        "concat:/tmp/a.mp4|/tmp/b.mp4",
        "subfile,,start,0,end,0,,:/tmp/a.mp4",
    ],
)
def test_fetch_video_refuses_other_url_schemes_before_a_decoder(monkeypatch, no_network, url):
    # ffmpeg dials these protocols itself, so they would bypass the guard.
    _decoder_must_not_run(monkeypatch)
    with pytest.raises(ValueError, match = "unsupported scheme"):
        vision_utils.fetch_video({"video": url})
    assert no_network == []


@pytest.mark.parametrize(
    "path",
    [
        "C:\\data\\clip.mp4",
        "C:/data/clip.mp4",
        "/data/clip.mp4",
        "clips/a.mp4",
        "subfile,clip.mp4",
        "file://C:/data/clip.mp4",
        "\\\\?\\C:\\data\\clip.mp4",
        "\\\\?\\Volume{b75e2c83-0000-0000-0000-602f00000000}\\videos\\clip.mp4",
        "file:///tmp/clip.mp4",
        "data:video/mp4;base64,AAAAIGZ0eXBpc29t",
    ],
)
def test_fetch_video_local_paths_still_reach_the_decoder(monkeypatch, no_network, path):
    seen = []

    def decoder(ele):
        seen.append(ele["video"])
        raise RuntimeError("stop after the decoder is reached")

    monkeypatch.setattr(vision_utils, "get_video_reader_backend", lambda: "torchvision")
    monkeypatch.setitem(vision_utils.VIDEO_READER_BACKENDS, "torchvision", decoder)
    with pytest.raises(RuntimeError, match = "stop after"):
        vision_utils.fetch_video({"video": path})
    assert seen and seen[0] == path


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
    """urlparse keeps the authority percent-encoded but requests decodes it, so an encoded host would be checked as unresolvable and then fetched from 127.0.0.1."""
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
    """float() accepts inf and nan but int() refuses them, so non-finite UNSLOTH_MAX_MEDIA_DOWNLOAD_MB values must not
    make every remote image raise before its request."""
    monkeypatch.setenv("UNSLOTH_MAX_MEDIA_DOWNLOAD_MB", value)
    monkeypatch.setattr(
        vision_utils.requests, "get",
        lambda url, **kwargs: _FakeResponse(_png_bytes()),
    )
    assert vision_utils.fetch_image({"image": "https://example.com/a.png"}).size[0] > 0


def test_video_download_is_guarded_not_delegated(monkeypatch, public_dns):
    """Decoders follow redirects with no way to refuse, so video bytes are fetched here under the guard and the decoder only ever sees a local file."""
    hops = []

    def fake_get(url, **kwargs):
        hops.append(url)
        if url.endswith("/public.mp4"):
            return _FakeResponse(redirect_to="http://127.0.0.1:9/internal.mp4")
        return _FakeResponse(b"")

    monkeypatch.setattr(vision_utils.requests, "get", fake_get)
    _decoder_must_not_run(monkeypatch)
    with pytest.raises(ValueError):
        vision_utils.fetch_video({"video": "https://example.com/public.mp4"})
    assert hops == ["https://example.com/public.mp4"]


def test_video_decoder_receives_a_local_file_not_a_url(monkeypatch, public_dns, tmp_path):
    """Even for a wholly public redirect chain the decoder receives a downloaded local file, so it cannot make a second request that is answered differently."""
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
    _decoder_must_not_run(monkeypatch)
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
    _decoder_must_not_run(monkeypatch)
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


def test_site_local_ipv6_is_blocked():
    """fec0::/10 site-local IPv6 is deprecated but still routed in places, and ipaddress reports it only via is_site_local, not is_private."""
    import ipaddress
    assert vision_utils._is_blocked_ip(ipaddress.ip_address("fec0::1")) is True


def test_percent_escapes_in_credentials_are_allowed(monkeypatch, public_dns):
    """Percent-escapes in credentials, as in user:p%40ss@host, are valid; only the host portion is suspect."""
    seen = []
    monkeypatch.setattr(
        vision_utils.requests, "get",
        lambda url, **kwargs: seen.append(url) or _FakeResponse(_png_bytes()),
    )
    image = vision_utils.fetch_image({"image": "https://user:p%40ss@example.com/x.png"})
    assert image.size[0] > 0
    assert seen == ["https://user:p%40ss@example.com/x.png"]


def test_encoded_host_still_refused_even_with_credentials(no_network):
    with pytest.raises(ValueError):
        vision_utils.fetch_image({"image": "http://user:pass@%31%32%37.0.0.1/x.png"})
    assert no_network == []


def test_unresolvable_host_behind_a_proxy_fails_closed(monkeypatch):
    """A proxy resolves the name instead of us, so nothing was ever checked."""
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.internal:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    with pytest.raises(ValueError, match="proxy"):
        vision_utils.assert_fetchable_url("http://intranet.corp.invalid/x.png")


def test_unresolvable_host_without_a_proxy_is_left_to_the_client(monkeypatch):
    """With no proxy configured an unresolvable host is left to the HTTP client, which reports its own error rather than the guard inventing one."""
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    assert vision_utils.assert_fetchable_url("http://nonexistent.invalid/x.png")


def test_one_session_spans_the_whole_redirect_chain(monkeypatch, public_dns, _fake_session):
    """One requests session must span the whole redirect chain, since signed-cookie CDNs depend on the cookie jar carrying across hops."""
    def fake_get(url, **kwargs):
        if url.endswith("/signed.png"):
            return _FakeResponse(redirect_to="https://cdn.example.com/real.png")
        return _FakeResponse(_png_bytes())

    monkeypatch.setattr(vision_utils.requests, "get", fake_get)
    image = vision_utils.fetch_image({"image": "https://example.com/signed.png"})
    assert image.size[0] > 0
    assert len(_fake_session.instances) == 1, "every hop must share one session"
    assert _fake_session.instances[0].closed, "the session must be closed"
    assert sorted(_fake_session.instances[0].mounted) == ["http://", "https://"], \
        "both schemes must go through the address-pinning adapter"


def test_video_backend_missing_leaves_no_download(monkeypatch, public_dns, tmp_path):
    """Backend selection happens before the download, so a missing decoder backend leaves no downloaded file behind."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path), raising=False)
    monkeypatch.setattr(
        vision_utils.requests, "get",
        lambda url, **kwargs: _FakeResponse(b"video bytes"),
    )

    def no_backend():
        raise ValueError("Unsloth: No video reader backend available")

    monkeypatch.setattr(vision_utils, "get_video_reader_backend", no_backend)
    with pytest.raises(ValueError):
        vision_utils.fetch_video({"video": "https://example.com/clip.mp4"})
    assert glob.glob(str(tmp_path / "unsloth_media_*")) == []


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1\\@example.com/x.png",
        "http://127.0.0.1:8080\\@example.com/x.png",
        "http://example.com\\@127.0.0.1/x.png",
        "https://user\\@evil@example.com/x.png",
    ],
)
def test_backslash_anywhere_in_the_authority_is_refused(url, no_network):
    """urlparse reads the prefix as userinfo and reports the trailing host, but the client treats the backslash as a
    path separator and dials the prefix, so any backslash in the authority is refused."""
    with pytest.raises(ValueError, match="backslash"):
        vision_utils.fetch_image({"image": url})
    assert no_network == []


@pytest.mark.parametrize(
    "path",
    [
        "\\\\attacker\\share\\clip.mp4",
        "//attacker/share/clip.mp4",
        "file://attacker/share/clip.mp4",
        "file:///%5cattacker/share/clip.mp4",
        "\\\\?\\UNC\\attacker\\share\\clip.mp4",
        "\\\\?\\GLOBALROOT\\Device\\Mup\\attacker\\share\\clip.mp4",
        "\\\\.\\GLOBALROOT\\Device\\LanmanRedirector\\attacker\\share\\clip.mp4",
    ],
)
def test_fetch_video_refuses_network_share_paths(monkeypatch, no_network, path):
    monkeypatch.setattr(vision_utils, "_WINDOWS", True)
    # Windows would open an SMB connection (and may send credentials) for these.
    _decoder_must_not_run(monkeypatch)
    with pytest.raises(ValueError, match = "network share"):
        vision_utils.fetch_video({"video": path})


@pytest.mark.parametrize(
    "body",
    [
        b"#EXTM3U\n#EXTINF:2.0,\nfile://attacker/share/seg0.ts\n",
        b'<?xml version="1.0"?>\n<!-- ' + b"x" * 4096 + b' -->\n<MPD xmlns="urn:mpeg:dash:schema:mpd:2011"></MPD>',
        b"ffconcat version 1.0\nfile //attacker/share/a.mp4\n",
        b" " * 70000 + b'<MPD xmlns="urn:mpeg:dash:schema:mpd:2011"></MPD>',
        b"\xef\xbb\xbf#EXTM3U\nfile://attacker/share/seg0.ts\n",
    ],
    # Short ids: Windows caps PYTEST_CURRENT_TEST at 32767 characters.
    ids = ["hls", "dash", "ffconcat", "padded_dash", "bom_hls"],
)
def test_a_downloaded_playlist_never_reaches_the_decoder(monkeypatch, public_dns, tmp_path, body):
    # ffmpeg opens a playlist's entries itself; a file:// segment would reach a share on Windows.
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(vision_utils.requests, "get", lambda url, **kwargs: _FakeResponse(body))
    _decoder_must_not_run(monkeypatch)
    with pytest.raises(ValueError, match = "playlist"):
        vision_utils.fetch_video({"video": "https://example.com/live.m3u8"})
    assert not list(tmp_path.glob("unsloth_media_*"))


@pytest.mark.parametrize(
    "body",
    [
        b"#EXTM3U\n#EXTINF:2.0,\nfile://attacker/share/seg0.ts\n",
        b"ffconcat version 1.0\nfile //attacker/share/a.mp4\n",
        b" " * 4096 + b'<MPD xmlns="urn:mpeg:dash:schema:mpd:2011"></MPD>',
    ],
)
@pytest.mark.parametrize("encoding", ["base64", "percent"])
def test_an_inline_data_playlist_never_reaches_the_decoder(monkeypatch, no_network, body, encoding):
    import base64
    from urllib.parse import quote_from_bytes

    if encoding == "base64":
        uri = "data:video/mp4;base64," + base64.b64encode(body).decode()
    else:
        uri = "data:video/mp4," + quote_from_bytes(body)
    _decoder_must_not_run(monkeypatch)
    with pytest.raises(ValueError, match = "playlist"):
        vision_utils.fetch_video({"video": uri})


@pytest.mark.parametrize(
    "path",
    [
        "\\\\wsl$\\Ubuntu\\home\\u\\clip.mp4",
        "\\\\wsl.localhost\\Ubuntu\\home\\u\\clip.mp4",
        "file://wsl.localhost/Ubuntu/home/u/clip.mp4",
        "\\\\?\\UNC\\wsl.localhost\\Ubuntu\\clip.mp4",
    ],
)
def test_wsl_paths_are_local_on_windows(monkeypatch, no_network, path):
    monkeypatch.setattr(vision_utils, "_WINDOWS", True)
    _reaches_the_decoder(monkeypatch, path)


def test_double_slash_paths_are_ordinary_paths_off_windows(monkeypatch, no_network):
    monkeypatch.setattr(vision_utils, "_WINDOWS", False)
    _reaches_the_decoder(monkeypatch, "//data/videos/clip.mp4")


def test_the_opt_in_allows_a_share_the_owner_trusts(monkeypatch, no_network):
    monkeypatch.setattr(vision_utils, "_WINDOWS", True)
    _decoder_must_not_run(monkeypatch)
    with pytest.raises(ValueError, match = "UNSLOTH_ALLOW_PRIVATE_URL_FETCH=1"):
        vision_utils.fetch_video({"video": "\\\\nas\\videos\\clip.mp4"})
    monkeypatch.setenv("UNSLOTH_ALLOW_PRIVATE_URL_FETCH", "1")
    _reaches_the_decoder(monkeypatch, "\\\\nas\\videos\\clip.mp4")


def _reaches_the_decoder(monkeypatch, path):
    seen = []

    def decoder(ele):
        seen.append(ele["video"])
        raise RuntimeError("stop after the decoder is reached")

    monkeypatch.setattr(vision_utils, "get_video_reader_backend", lambda: "torchvision")
    monkeypatch.setitem(vision_utils.VIDEO_READER_BACKENDS, "torchvision", decoder)
    with pytest.raises(RuntimeError, match = "stop after"):
        vision_utils.fetch_video({"video": path})
    assert seen and seen[0] == path

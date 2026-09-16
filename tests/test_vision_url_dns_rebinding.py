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

"""The media guard must connect to the address it checked.

Checking a hostname and then handing the name to the HTTP client resolves DNS
twice, and whoever serves the zone for a hostname in a dataset row answers both
queries: a public address for the check, loopback or RFC1918 for the connection.
The bytes are consumed as an image, so the payoff is a blind internal GET rather
than a read primitive, but that is still enough for dataset content to probe
services inside the victim's network.

These tests drive the real requests/urllib3 stack over a loopback HTTP server
with a stubbed resolver, so the address actually connected to is observed rather
than assumed. Nothing here talks to an external name server or host: the
resolver is monkeypatched and every non-loopback connect is refused in-process.
"""

from __future__ import annotations

import http.server
import io
import socket
import threading

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

import requests  # noqa: E402

from unsloth_zoo import vision_utils  # noqa: E402


# example.com's historical address, the same stand-in the guard's other tests use
PUBLIC_IP = "93.184.216.34"


def _png_bytes(size=(32, 32), color=(4, 5, 6)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def _default_policy(monkeypatch):
    monkeypatch.delenv("UNSLOTH_ALLOW_PRIVATE_URL_FETCH", raising=False)
    monkeypatch.delenv("UNSLOTH_MAX_MEDIA_DOWNLOAD_MB", raising=False)
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    yield


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self.server.seen.append((self.path, self.headers.get("Host")))
        if self.path.startswith("/redirect"):
            target = self.server.redirect_to
            self.send_response(302)
            self.send_header("Location", target)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = _png_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    """A loopback HTTP server standing in for whatever listens on 127.0.0.1."""
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.seen = []
    httpd.redirect_to = ""
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture
def resolver(monkeypatch):
    """A resolver that can answer the same name differently on each lookup.

    `answers("host.invalid", "1.2.3.4", "127.0.0.1")` hands out the first address
    once and the second one from then on, which is exactly what an attacker
    controlling the authoritative zone does: the guard sees a public address, the
    connection sees loopback.
    """
    real = socket.getaddrinfo
    scripted = {}
    failing_first = set()
    lookups = []

    def fake_getaddrinfo(host, port=None, *args, **kwargs):
        if host in scripted:
            lookups.append(host)
            if host in failing_first and lookups.count(host) == 1:
                raise socket.gaierror(f"scripted failure for {host}")
            addresses = scripted[host]
            index = min(lookups.count(host) - 1, len(addresses) - 1)
            chosen = addresses[index]
            if not isinstance(chosen, tuple):
                chosen = (chosen,)
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port or 0))
                for address in chosen
            ]
        return real(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    class Resolver:
        @staticmethod
        def answers(host, *addresses):
            scripted[host] = addresses

        @staticmethod
        def fails_then_answers(host, *addresses):
            """Fail the first lookup, answer every later one."""
            scripted[host] = addresses
            failing_first.add(host)

        @staticmethod
        def count(host):
            return sum(1 for seen in lookups if seen == host)

    return Resolver


@pytest.fixture
def connects(monkeypatch):
    """Record every TCP connect and refuse anything that is not loopback.

    Keeps the suite offline: a fetch that escapes to a routable address is
    reported as a refused connection to that address instead of leaving the
    test hanging on a real network round trip.
    """
    targets = []
    real_connect = socket.socket.connect

    def connect(self, address):
        targets.append((address[0], address[1]))
        if address[0] not in ("127.0.0.1", "::1"):
            raise ConnectionRefusedError(f"refused, this test is offline: {address[0]}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    return targets


@pytest.fixture
def loopback_counts_as_public(monkeypatch):
    """Let 127.0.0.1 pass the classifier so a real server can stand in for a
    public host. Only the transport is under test here; which addresses are
    internal is pinned exhaustively in test_vision_url_ssrf_guard.py."""
    import ipaddress

    real = vision_utils._is_blocked_ip
    allowed = ipaddress.ip_address("127.0.0.1")

    def is_blocked(ip):
        if ip == allowed: return False
        return real(ip)

    monkeypatch.setattr(vision_utils, "_is_blocked_ip", is_blocked)


def test_rebinding_to_loopback_after_the_check_is_refused(server, resolver, connects):
    """The fix under test: the guard checks the public answer, and the fetch goes
    to that same address rather than re-resolving into 127.0.0.1."""
    resolver.answers("rebind.invalid", PUBLIC_IP, "127.0.0.1")
    url = f"http://rebind.invalid:{server.server_address[1]}/x.png"

    with pytest.raises(Exception):
        vision_utils.fetch_image({"image": url})

    assert server.seen == [], "the loopback service must never be reached"
    assert (PUBLIC_IP, server.server_address[1]) in connects, connects
    assert not any(host == "127.0.0.1" for host, _ in connects), connects
    assert resolver.count("rebind.invalid") == 1, "the name must be resolved once, not twice"


def test_without_pinning_the_same_setup_does_reach_loopback(server, resolver, connects, monkeypatch):
    """The differential that makes the test above mean something.

    Validation alone, with the connection left to resolve the name for itself,
    is the pre-fix behaviour, and against the very same scripted resolver it
    lands on the loopback server and returns its bytes.
    """
    real_check = vision_utils._check_fetchable_url

    def validate_only(url):
        real_check(url)
        # Pin nothing, exactly like handing the client the hostname and letting
        # it resolve for itself, which is what the guard used to do.
        return vision_utils._UNPINNED

    monkeypatch.setattr(vision_utils, "_check_fetchable_url", validate_only)
    resolver.answers("rebind.invalid", PUBLIC_IP, "127.0.0.1")
    url = f"http://rebind.invalid:{server.server_address[1]}/x.png"

    image = vision_utils.fetch_image({"image": url})
    assert image.size[0] > 0 and image.size[1] > 0
    assert [path for path, _ in server.seen] == ["/x.png"]
    assert resolver.count("rebind.invalid") == 2, "two resolutions is the hole itself"


def test_a_public_hostname_still_fetches_and_keeps_its_host_header(
    server, resolver, connects, loopback_counts_as_public
):
    """Ordinary fetches keep working, and the request that arrives still carries
    the hostname, not the pinned address."""
    resolver.answers("images.invalid", "127.0.0.1")
    port = server.server_address[1]

    image = vision_utils.fetch_image({"image": f"http://images.invalid:{port}/cat.png"})

    assert image.size[0] > 0 and image.size[1] > 0
    assert server.seen == [("/cat.png", f"images.invalid:{port}")]
    assert connects == [("127.0.0.1", port)]


def test_a_mixed_case_hostname_is_pinned_too(server, resolver, connects):
    """Hostnames are case insensitive, so the pin must be matched that way or a
    capital letter in a dataset row would opt the fetch back out of pinning."""
    resolver.answers("rebind.invalid", PUBLIC_IP, "127.0.0.1")
    url = f"http://ReBiNd.InVaLiD:{server.server_address[1]}/x.png"

    with pytest.raises(Exception):
        vision_utils.fetch_image({"image": url})

    assert server.seen == []
    assert (PUBLIC_IP, server.server_address[1]) in connects, connects
    assert not any(host == "127.0.0.1" for host, _ in connects), connects


def test_every_checked_address_is_tried_in_order(server, resolver, connects, loopback_counts_as_public):
    """A multi-address answer must not lose failover: the first address refuses
    the connection here and the fetch still completes on the second."""
    resolver.answers("multi.invalid", (PUBLIC_IP, "127.0.0.1"))
    port = server.server_address[1]

    image = vision_utils.fetch_image({"image": f"http://multi.invalid:{port}/cat.png"})

    assert image.size[0] > 0 and image.size[1] > 0
    assert connects == [(PUBLIC_IP, port), ("127.0.0.1", port)]


def test_a_redirect_hop_is_pinned_to_its_own_checked_address(
    server, resolver, connects, loopback_counts_as_public
):
    """Per-hop re-validation still holds, and each hop connects to the address
    its own check resolved, so the second hop cannot be rebound either."""
    port = server.server_address[1]
    resolver.answers("images.invalid", "127.0.0.1")
    resolver.answers("rebind.invalid", PUBLIC_IP, "127.0.0.1")
    server.redirect_to = f"http://rebind.invalid:{port}/internal.png"

    with pytest.raises(Exception):
        vision_utils.fetch_image({"image": f"http://images.invalid:{port}/redirect.png"})

    assert [path for path, _ in server.seen] == ["/redirect.png"], server.seen
    assert (PUBLIC_IP, port) in connects, connects
    assert resolver.count("rebind.invalid") == 1


def test_a_redirect_to_an_internal_address_is_still_blocked_before_connecting(
    server, resolver, connects, loopback_counts_as_public
):
    """The existing per-hop check is untouched: a hop that resolves internally is
    refused by the guard, with no connection attempted at all."""
    port = server.server_address[1]
    resolver.answers("images.invalid", "127.0.0.1")
    resolver.answers("internal.invalid", "10.1.2.3")
    server.redirect_to = f"http://internal.invalid:{port}/secret.png"

    with pytest.raises(ValueError, match="loopback, private"):
        vision_utils.fetch_image({"image": f"http://images.invalid:{port}/redirect.png"})

    assert [target for target in connects if target[0] != "127.0.0.1"] == [], connects


def test_a_directly_private_hostname_is_rejected_before_any_connect(resolver, connects):
    resolver.answers("intranet.invalid", "192.168.4.4")
    with pytest.raises(ValueError, match="loopback, private"):
        vision_utils.fetch_image({"image": "http://intranet.invalid/x.png"})
    assert connects == [], "nothing may be dialled once the guard refuses"


def test_the_opt_out_still_fetches_a_loopback_url(server, connects):
    """UNSLOTH_ALLOW_PRIVATE_URL_FETCH=1 is the documented way to serve media
    from a private host, and it must keep working with pinning in place."""
    import os
    port = server.server_address[1]
    os.environ["UNSLOTH_ALLOW_PRIVATE_URL_FETCH"] = "1"
    try:
        image = vision_utils.fetch_image({"image": f"http://127.0.0.1:{port}/cat.png"})
    finally:
        del os.environ["UNSLOTH_ALLOW_PRIVATE_URL_FETCH"]
    assert image.size[0] > 0 and image.size[1] > 0
    assert connects == [("127.0.0.1", port)]


def test_tls_is_verified_against_the_hostname_not_the_pinned_address(
    server_tls, resolver, connects, loopback_counts_as_public
):
    """Pinning must not turn into certificate verification against an IP: the
    handshake here succeeds only if SNI and hostname matching still use the
    hostname, and a certificate for another name must still be rejected."""
    httpd, certificate = server_tls
    port = httpd.server_address[1]
    resolver.answers("secure.invalid", "127.0.0.1")
    resolver.answers("wrongname.invalid", "127.0.0.1")

    pinned = vision_utils._check_fetchable_url(f"https://secure.invalid:{port}/cat.png")
    assert pinned == (("secure.invalid",), ("127.0.0.1",)), pinned

    with vision_utils._guarded_session() as session:
        token = vision_utils._PINNED_ADDRESSES.set(pinned)
        try:
            response = session.get(f"https://secure.invalid:{port}/cat.png", verify=certificate, timeout=10)
            assert response.status_code == 200
            assert Image.open(io.BytesIO(response.content)).size == (32, 32)
        finally:
            vision_utils._PINNED_ADDRESSES.reset(token)

        mismatched = (("wrongname.invalid",), ("127.0.0.1",))
        token = vision_utils._PINNED_ADDRESSES.set(mismatched)
        try:
            with pytest.raises(requests.exceptions.SSLError):
                session.get(f"https://wrongname.invalid:{port}/cat.png", verify=certificate, timeout=10)
        finally:
            vision_utils._PINNED_ADDRESSES.reset(token)

    assert httpd.seen[0][1] == f"secure.invalid:{port}", httpd.seen


@pytest.fixture
def server_tls(tmp_path):
    """A loopback HTTPS server holding a certificate for `secure.invalid`."""
    cryptography = pytest.importorskip("cryptography")
    import datetime
    import ssl
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    del cryptography
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "secure.invalid")])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("secure.invalid")]), critical=False)
        .sign(key, hashes.SHA256())
    )
    certificate_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(certificate_path), str(key_path))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.seen = []
    httpd.redirect_to = ""
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd, str(certificate_path)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_the_pinned_connection_restores_the_hostname_before_tls_runs():
    """urllib3 reads `host` for SNI, hostname matching and the Host header after
    `_new_conn` returns, so the address may only be in place across the connect."""
    observed = []

    class Base:
        def __init__(self):
            self._dns_host = "example.invalid"

        def _new_conn(self):
            observed.append(self._dns_host)
            if self._dns_host == "93.184.216.34":
                raise OSError("first address refuses")
            return "socket"

    class Pinned(vision_utils._PinnedConnectionMixin, Base):
        pass

    connection = Pinned()
    token = vision_utils._PINNED_ADDRESSES.set(("example.invalid", (PUBLIC_IP, "198.51.100.7")))
    try:
        assert connection._new_conn() == "socket"
    finally:
        vision_utils._PINNED_ADDRESSES.reset(token)

    assert observed == [PUBLIC_IP, "198.51.100.7"], "each checked address, in order"
    assert connection._dns_host == "example.invalid", "the hostname must be back in place"


def _pinned_probe_class(observed):
    class Base:
        def __init__(self, host):
            self._dns_host = host

        def _new_conn(self):
            observed.append(self._dns_host)
            return "socket"

    class Pinned(vision_utils._PinnedConnectionMixin, Base):
        pass

    return Pinned


def test_a_missing_or_foreign_pin_is_refused_rather_than_resolved_again():
    """Falling back to the client's own DNS answer is the bug, so every state
    other than a matching pin has to refuse. `_UNPINNED` is the one exception:
    the opt-out and the proxy path really do leave the address to someone else.
    """
    observed = []
    pinned_class = _pinned_probe_class(observed)

    # No pin in this context at all, e.g. the state was lost crossing a thread
    with pytest.raises(ValueError, match="nothing pins"):
        pinned_class("images.invalid")._new_conn()

    # A host that is not the one checked, a proxy hop for instance
    token = vision_utils._PINNED_ADDRESSES.set((("example.invalid",), (PUBLIC_IP,)))
    try:
        with pytest.raises(ValueError, match="checked host was"):
            pinned_class("proxy.invalid")._new_conn()
    finally:
        vision_utils._PINNED_ADDRESSES.reset(token)

    # Checked, but the name did not resolve then, so nothing may be dialled now
    token = vision_utils._PINNED_ADDRESSES.set((("flaky.invalid",), ()))
    try:
        with pytest.raises(ValueError, match="did not resolve"):
            pinned_class("flaky.invalid")._new_conn()
    finally:
        vision_utils._PINNED_ADDRESSES.reset(token)

    assert observed == [], "none of those may reach the resolver"

    token = vision_utils._PINNED_ADDRESSES.set(vision_utils._UNPINNED)
    try:
        assert pinned_class("proxy.invalid")._new_conn() == "socket"
    finally:
        vision_utils._PINNED_ADDRESSES.reset(token)
    assert observed == ["proxy.invalid"]


def test_a_public_ipv6_literal_url_still_connects_to_that_address(connects):
    """There is no name to rebind here, and the pin must recognise the address
    as its own host rather than refusing a perfectly ordinary URL."""
    with pytest.raises(requests.exceptions.ConnectionError):
        vision_utils.fetch_image({"image": "http://[2606:4700:4700::1111]/x.png"})
    assert connects == [("2606:4700:4700::1111", 80)], connects


def test_a_bracketed_ipv6_host_matches_the_pin(connects):
    """urllib3 1.x leaves an IPv6 literal in its brackets where 2.x strips them,
    and either spelling has to match or the fetch would be refused outright."""
    observed = []
    pinned_class = _pinned_probe_class(observed)
    token = vision_utils._PINNED_ADDRESSES.set(
        (("2606:4700:4700::1111",), ("2606:4700:4700::1111",))
    )
    try:
        assert pinned_class("[2606:4700:4700::1111]")._new_conn() == "socket"
    finally:
        vision_utils._PINNED_ADDRESSES.reset(token)
    assert observed == ["2606:4700:4700::1111"]


def test_a_punycoded_hostname_is_still_pinned(server, resolver, connects):
    """requests punycodes a non-ASCII host before it builds the connection, so a
    pin held only under the unicode spelling would never match and the fetch
    would resolve the name for itself again. One accented letter in a dataset
    row must not be a way back to the old behaviour."""
    port = server.server_address[1]
    resolver.answers("rébind.invalid", PUBLIC_IP)
    resolver.answers("xn--rbind-bsa.invalid", "127.0.0.1")

    assert vision_utils._check_fetchable_url(f"http://rébind.invalid:{port}/x.png") == (
        ("rébind.invalid", "xn--rbind-bsa.invalid"), (PUBLIC_IP,)
    )

    with pytest.raises(Exception):
        vision_utils.fetch_image({"image": f"http://rébind.invalid:{port}/x.png"})

    assert server.seen == [], "the loopback service must never be reached"
    assert not any(host == "127.0.0.1" for host, _ in connects), connects


def test_a_name_that_fails_the_check_lookup_cannot_resolve_at_connect_time(server, resolver, connects):
    """Answering the guard's query with a failure and the client's with loopback
    is the same two-answer trick, so an unresolvable name pins to nothing."""
    port = server.server_address[1]
    resolver.fails_then_answers("flaky.invalid", "127.0.0.1")

    with pytest.raises(ValueError, match="did not resolve"):
        vision_utils.fetch_image({"image": f"http://flaky.invalid:{port}/x.png"})

    assert server.seen == []
    assert connects == []


def test_the_checked_addresses_are_not_dialled_once_per_socket_type(monkeypatch):
    """getaddrinfo answers once per socket type unless it is told otherwise, and
    dialling each address three times would treble the wait on a dead host."""
    import socket as socket_module

    seen = {}
    real = socket_module.getaddrinfo

    def recording(host, port=None, *args, **kwargs):
        seen["type"] = kwargs.get("type", args[2] if len(args) > 2 else 0)
        return [
            (socket_module.AF_INET, family, proto, "", ("93.184.216.34", 0))
            for family, proto in ((socket_module.SOCK_STREAM, 6), (socket_module.SOCK_DGRAM, 17))
        ]

    monkeypatch.setattr(socket_module, "getaddrinfo", recording)
    del real
    assert vision_utils._resolve_host("dupes.invalid") == (
        __import__("ipaddress").ip_address("93.184.216.34"),
    ), "duplicate rows must collapse"
    assert seen["type"] == socket_module.SOCK_STREAM


def test_the_guarded_session_pins_both_schemes():
    with vision_utils._guarded_session() as session:
        for prefix in ("http://", "https://"):
            adapter = session.get_adapter(prefix + "example.com/x.png")
            classes = adapter.poolmanager.pool_classes_by_scheme
            for scheme in ("http", "https"):
                assert issubclass(
                    classes[scheme].ConnectionCls, vision_utils._PinnedConnectionMixin
                ), (prefix, scheme)

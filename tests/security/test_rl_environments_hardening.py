# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Two boundaries in `rl_environments.py`, both of them load bearing.

`create_locked_down_function` is handed source the user did not write (a model
or an RL environment did), so the allowlist policy is the only thing between
that source and `__import__`. `_reject_dunder_access` enforces it by walking
the AST, which means any construct that spells an attribute as a plain
identifier string rather than an `ast.Attribute` node is invisible to it. A
PEP 634 class pattern is exactly that: `case object(__class__=x)` is a getattr.

`launch_openenv` spawns the OpenEnv server the trainer exchanges observations,
actions and rewards with. It must bind loopback rather than every interface,
and it must not accept any local process that happens to answer `/health` as
that server.

CPU-only and network-free; the one real listener here is on 127.0.0.1, which
the suite's network blocker allows.
"""

import contextlib
import http.server
import socket
import subprocess
import threading
import time
import types
from urllib.parse import urlsplit

import pytest

import unsloth_zoo.rl_environments as rl_env
from unsloth_zoo.rl_environments import create_locked_down_function


# --- the AST allowlist: attribute names that are not Attribute nodes ---------

MATCH_ESCAPES = [
    # The subclasses walk, one getattr at a time, with no Attribute anywhere.
    ("class", "    match (1,):\n        case object(__class__=t):\n            pass\n    return t\n"),
    ("subclasses", "    match A:\n        case object(__subclasses__=g):\n            pass\n    return g\n"),
    ("globals", "    match A:\n        case object(__globals__=g):\n            pass\n    return g\n"),
    # Private, not dunder: the same rule the Attribute check applies.
    ("private", "    match A:\n        case object(_os=g):\n            pass\n    return g\n"),
    # Frame walk names are denied by value, not by leading underscore.
    ("frame", "    match A:\n        case object(f_globals=g):\n            pass\n    return g\n"),
    # Nested inside a sequence pattern, so a shallow check would miss it.
    ("nested", "    match [A]:\n        case [object(__class__=t)]:\n            pass\n    return t\n"),
]


@pytest.mark.parametrize("name,body", MATCH_ESCAPES, ids = [n for n, _ in MATCH_ESCAPES])
def test_match_class_pattern_attribute_rejected(name, body):
    with pytest.raises(RuntimeError, match = "not allowed in generated code"):
        create_locked_down_function("def matmul(A, B):\n" + body)


def test_match_class_pattern_chain_cannot_reach_import():
    """The end to end escape: pattern getattrs recover the real builtins dict."""
    source = (
        "def matmul(A, B):\n"
        "    match (1,):\n"
        "        case object(__class__=t):\n"
        "            pass\n"
        "    match t:\n"
        "        case object(__base__=obj):\n"
        "            pass\n"
        "    match obj:\n"
        "        case object(__subclasses__=gs):\n"
        "            pass\n"
        "    return gs()\n"
    )
    with pytest.raises(RuntimeError, match = "not allowed in generated code"):
        create_locked_down_function(source)


# `kwd_attrs` is the only identifier-as-string field that READS an attribute.
# Every other one just binds a name, and a bound dunder is inert because reading
# it back is an ast.Name that the Name rule refuses. These cases pin exactly
# that, so the reason the allowlist does not need a rule per binding form is
# asserted rather than assumed: if the Name rule ever stopped covering a read,
# one of these fails instead of a hole opening quietly.
BOUND_NAME_READS = [
    ("match capture", "    match A:\n        case [x] as __evil:\n            pass\n    return __evil\n"),
    ("match star", "    match A:\n        case [x, *__evil]:\n            pass\n    return __evil\n"),
    ("match rest", "    match A:\n        case {'a': x, **__evil}:\n            pass\n    return __evil\n"),
    ("except", "    try:\n        return A[0]\n    except IndexError as __evil:\n        return str(__evil)\n"),
    ("global", "    global __builtins__\n    __builtins__ = 1\n    return 0\n"),
    ("keyword", "    d = dict(__class__=1)\n    return __class__\n"),
    ("import alias", "    import math as __evil\n    return __evil.pi\n"),
    ("parameter", "    return __class__\n"),
]


@pytest.mark.parametrize("name,body", BOUND_NAME_READS, ids = [n for n, _ in BOUND_NAME_READS])
def test_reading_a_dunder_binding_is_rejected(name, body):
    """A dunder bound by any spelling is unusable, because the read is an ast.Name."""
    with pytest.raises(RuntimeError, match = "not allowed in generated code"):
        create_locked_down_function("def matmul(A, B):\n" + body)


# The flip side, and the reason the binding forms are not rejected outright:
# every one of these is a working program that a rule on the binding broke.
BOUND_NAME_FALSE_POSITIVES = [
    ("dict with a dunder key", "def f():\n    return dict(__class__ = 'label')\n", (), {"__class__": "label"}),
    ("unused dunder parameter", "def f(__class__ = None):\n    return 7\n", (), 7),
    ("star args", "def f(*__args):\n    return 7\n", (1, 2), 7),
    ("kwargs", "def f(**__kw):\n    return 7\n", (), 7),
    ("unused import alias", "def f():\n    import math as __m\n    return 7\n", (), 7),
    ("unused except capture", "def f():\n    try:\n        1 / 0\n    except ZeroDivisionError as __e:\n        return 7\n", (), 7),
    ("unused match capture", "def f(x):\n    match x:\n        case [v] as __w:\n            return v\n", ([7],), 7),
    ("unused match star", "def f(x):\n    match x:\n        case [v, *__r]:\n            return v\n", ([7, 8],), 7),
    ("unused match rest", "def f(x):\n    match x:\n        case {'v': v, **__r}:\n            return v\n", ({"v": 7, "o": 8},), 7),
]


@pytest.mark.parametrize(
    "name,source,args,expected", BOUND_NAME_FALSE_POSITIVES,
    ids = [n for n, _, _, _ in BOUND_NAME_FALSE_POSITIVES],
)
def test_binding_a_dunder_name_is_not_itself_refused(name, source, args, expected):
    """Binding reaches nothing, so refusing it only broke generated code that worked."""
    assert create_locked_down_function(source)(*args) == expected


# --- none of the hardening may cost ordinary generated code ------------------

def test_ordinary_match_statements_still_work():
    """Public attributes, capture names and every benign pattern kind."""
    source = (
        "def matmul(A, B):\n"
        "    total = 0\n"
        "    for item in A:\n"
        "        match item:\n"
        "            case complex(real=r, imag=i):\n"
        "                total += int(r + i)\n"
        "            case {'score': s}:\n"
        "                total += s\n"
        "            case [head, *rest] as whole:\n"
        "                total += head + len(rest) + len(whole)\n"
        "            case int() | float():\n"
        "                total += int(item)\n"
        "            case _:\n"
        "                total += 1\n"
        "    return total\n"
    )
    matmul = create_locked_down_function(source)
    assert matmul([complex(2, 3), {"score": 4}, [1, 2, 3], 5, "x"], []) == 21


def test_ordinary_bindings_still_work():
    """Private and plain names bind as before; only dunders are refused."""
    source = (
        "def matmul(A, B):\n"
        "    import math as m\n"
        "    _total = 0\n"
        "    try:\n"
        "        _total += A[0]\n"
        "    except IndexError as error:\n"
        "        _total += len(str(error)) * 0\n"
        "    return int(m.floor(_total)) + sorted([2, 1], reverse=True)[0]\n"
    )
    assert create_locked_down_function(source)([5], []) == 7


# --- launch_openenv ----------------------------------------------------------

class _HealthyHandler(http.server.BaseHTTPRequestHandler):
    """A local process that answers /health and nothing else, i.e. a squatter."""
    def do_GET(self):
        body = b"healthy"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def foreign_listener():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _HealthyHandler)
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout = 5)


@pytest.fixture(autouse = True)
def no_leaked_children():
    """The launcher's registry is a module global, so isolate it per test."""
    registry = getattr(rl_env, "_OPENENV_CHILDREN", None)
    if registry is not None: registry.clear()
    yield
    if registry is not None: registry.clear()


class _DeadChild:
    """A uvicorn child that exited, e.g. because the port was already taken."""
    def poll(self): return 1


class _LiveChild:
    def poll(self): return None


class _DummyClient:
    def __init__(self, base_url): self.base_url = base_url


def _fake_launcher(monkeypatch, ports, child = None, bound = None):
    """Drive launch_openenv without a real uvicorn: fixed ports, fake child."""
    bound = set() if bound is None else bound
    calls = []

    def fake_popen(argv, **kwargs):
        calls.append(argv)
        if child is None or isinstance(child, _LiveChild):
            bound.add(int(argv[argv.index("--port") + 1]))
        return _LiveChild() if child is None else child

    queue = list(ports)
    monkeypatch.setattr(rl_env, "subprocess", types.SimpleNamespace(
        Popen = fake_popen, PIPE = subprocess.PIPE,
    ))
    monkeypatch.setattr(rl_env, "random", types.SimpleNamespace(
        randint = lambda low, high: queue.pop(0) if queue else 65000,
    ))
    monkeypatch.setattr(rl_env, "is_port_open", lambda host, port: port in bound)
    monkeypatch.setattr(rl_env, "requests", types.SimpleNamespace(
        get = lambda url, **kwargs: types.SimpleNamespace(content = b"healthy"),
    ))
    monkeypatch.setattr(rl_env, "time", types.SimpleNamespace(sleep = lambda seconds: None))
    return calls


def test_server_binds_loopback_by_default(monkeypatch, tmp_path):
    """The bind address is what an unauthenticated peer needs; loopback denies it."""
    calls = _fake_launcher(monkeypatch, ports = [31517])
    port, client = rl_env.launch_openenv(
        working_directory = str(tmp_path), openenv_class = _DummyClient,
    )
    assert port == 31517
    assert isinstance(client, _DummyClient)
    assert len(calls) == 1
    argv = calls[0]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert "0.0.0.0" not in argv


def test_explicit_host_is_still_honoured(monkeypatch, tmp_path):
    """Widening the bind stays possible, but it is now recorded at the call site."""
    calls = _fake_launcher(monkeypatch, ports = [31519])
    port, client = rl_env.launch_openenv(
        working_directory = str(tmp_path), openenv_class = _DummyClient,
        host = "0.0.0.0",
    )
    argv = calls[0]
    assert argv[argv.index("--host") + 1] == "0.0.0.0"
    # A wildcard bind is still dialled through a real address of its own family.
    assert client.base_url == f"http://127.0.0.1:{port}"


def test_foreign_listener_answering_healthy_is_not_adopted(foreign_listener, monkeypatch, tmp_path):
    """Seven bytes from a process we did not start must not become the environment."""
    stale_client = _DummyClient(base_url = f"http://localhost:{foreign_listener}")
    monkeypatch.setattr(rl_env, "subprocess", types.SimpleNamespace(
        Popen = lambda argv, **kwargs: _DeadChild(), PIPE = subprocess.PIPE,
    ))
    monkeypatch.setattr(rl_env, "time", types.SimpleNamespace(sleep = lambda seconds: None))

    with pytest.raises(TimeoutError, match = "30 times"):
        rl_env.launch_openenv(
            port = foreign_listener,
            openenv_process = stale_client,
            working_directory = str(tmp_path),
            openenv_class = _DummyClient,
        )


def test_child_that_exited_is_not_confused_with_a_squatter(monkeypatch, tmp_path):
    """A dead child is detected instead of being waited on for 60 seconds."""
    calls = _fake_launcher(monkeypatch, ports = [], child = _DeadChild())
    with pytest.raises(TimeoutError, match = "30 times"):
        rl_env.launch_openenv(
            working_directory = str(tmp_path), openenv_class = _DummyClient,
        )
    assert len(calls) == 30


def test_our_own_live_child_is_reused(monkeypatch, tmp_path):
    """The reuse branch still works: no second uvicorn for a server we own."""
    calls = _fake_launcher(monkeypatch, ports = [31521])
    port, client = rl_env.launch_openenv(
        working_directory = str(tmp_path), openenv_class = _DummyClient,
    )
    again_port, again_client = rl_env.launch_openenv(
        port = port, openenv_process = client,
        working_directory = str(tmp_path), openenv_class = _DummyClient,
    )
    assert (again_port, again_client) == (port, client)
    assert len(calls) == 1


# --- the readiness probe: an IPv6 bind must be probed over IPv6 --------------

@pytest.fixture
def ipv4_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(5)
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


@pytest.fixture
def ipv6_listener():
    """A loopback listener reachable only over IPv6, as `--host ::1` gives."""
    if not socket.has_ipv6: pytest.skip("Python built without IPv6")
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    try:
        sock.bind(("::1", 0))
    except OSError:
        sock.close()
        pytest.skip("No IPv6 loopback on this host")
    sock.listen(5)
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


def test_ipv6_listener_is_probed_over_ipv6(ipv6_listener):
    """AF_INET sent every ::1 probe to 127.0.0.1, so the port read as closed."""
    assert rl_env.is_port_open("::1", ipv6_listener)
    # Whether `localhost` also covers it is the resolver's call, not ours: this
    # host's /etc/hosts gives ::1 to ip6-localhost only. Where it is dual stack,
    # the probe must not stop at the first candidate that refuses.
    dual_stack = any(
        info[0] == socket.AF_INET6
        for info in socket.getaddrinfo("localhost", ipv6_listener, type = socket.SOCK_STREAM)
    )
    if dual_stack: assert rl_env.is_port_open("localhost", ipv6_listener)


def test_ipv4_listener_still_probes_open(ipv4_listener):
    """The widened probe must not have cost the IPv4 path."""
    assert rl_env.is_port_open("127.0.0.1", ipv4_listener)
    assert rl_env.is_port_open("localhost", ipv4_listener)


def test_closed_port_is_still_closed():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    assert not rl_env.is_port_open("127.0.0.1", port)


def test_unresolvable_host_returns_false(monkeypatch):
    """Resolution failure is a closed port, not a gaierror out of the probe."""
    def boom(*args, **kwargs): raise socket.gaierror(-2, "Name or service not known")
    monkeypatch.setattr(rl_env.socket, "getaddrinfo", boom)
    assert not rl_env.is_port_open("unsloth-no-such-host.invalid", 9000)


def test_ipv6_host_is_bracketed_in_the_client_url(monkeypatch, tmp_path):
    """http://::1:9000 has no host and an unparseable port; RFC 3986 3.2.2."""
    calls = _fake_launcher(monkeypatch, ports = [31523])
    port, client = rl_env.launch_openenv(
        working_directory = str(tmp_path), openenv_class = _DummyClient,
        host = "::1",
    )
    argv = calls[0]
    assert argv[argv.index("--host") + 1] == "::1"
    assert client.base_url == f"http://[::1]:{port}"
    assert urlsplit(client.base_url).hostname == "::1"
    assert urlsplit(client.base_url).port == port


def test_ipv6_wildcard_is_dialled_through_ipv6_loopback(monkeypatch, tmp_path):
    """asyncio sets IPV6_V6ONLY on every AF_INET6 listener, so `::` is not on 127.0.0.1."""
    calls = _fake_launcher(monkeypatch, ports = [31525])
    port, client = rl_env.launch_openenv(
        working_directory = str(tmp_path), openenv_class = _DummyClient,
        host = "::",
    )
    assert calls[0][calls[0].index("--host") + 1] == "::"
    assert client.base_url == f"http://[::1]:{port}"


def test_asyncio_really_makes_an_ipv6_wildcard_listener_v6_only():
    """The premise above, measured rather than assumed: CPython base_events.create_server
    calls setsockopt(IPPROTO_IPV6, IPV6_V6ONLY, True) whatever net.ipv6.bindv6only says."""
    if not socket.has_ipv6: pytest.skip("Python built without IPv6")
    import asyncio

    async def bind():
        server = await asyncio.start_server(lambda r, w: w.close(), "::", 0)
        try:
            sock = server.sockets[0]
            return sock.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY)
        finally:
            server.close()

    try:
        assert asyncio.run(bind()) == 1
    except OSError:
        pytest.skip("No IPv6 loopback on this host")


def test_ipv4_loopback_host_is_not_bracketed(monkeypatch, tmp_path):
    calls = _fake_launcher(monkeypatch, ports = [31527])
    port, client = rl_env.launch_openenv(
        working_directory = str(tmp_path), openenv_class = _DummyClient,
        host = "127.0.0.1",
    )
    assert client.base_url == f"http://127.0.0.1:{port}"


# --- the registry identifies an endpoint, not a port number -------------------

def test_two_hosts_on_one_port_are_tracked_separately(monkeypatch, tmp_path):
    """A port-only key lost one of two live children and misassigned the other."""
    listening = set()
    spawned = []
    dialled = {"0.0.0.0": "127.0.0.1", "::": "::1"}

    def fake_popen(argv, **kwargs):
        host = argv[argv.index("--host") + 1]
        port = int(argv[argv.index("--port") + 1])
        spawned.append((host, port))
        # uvicorn on a wildcard answers on the loopback of that same family.
        listening.add((dialled.get(host, host), port))
        return _LiveChild()

    monkeypatch.setattr(rl_env, "subprocess", types.SimpleNamespace(
        Popen = fake_popen, PIPE = subprocess.PIPE,
    ))
    monkeypatch.setattr(rl_env, "random", types.SimpleNamespace(randint = lambda a, b: 45000))
    monkeypatch.setattr(rl_env, "is_port_open", lambda host, port: (host, port) in listening)
    monkeypatch.setattr(rl_env, "requests", types.SimpleNamespace(
        get = lambda url, **kwargs: types.SimpleNamespace(content = b"healthy"),
    ))
    monkeypatch.setattr(rl_env, "time", types.SimpleNamespace(sleep = lambda seconds: None))

    port_v4, client_v4 = rl_env.launch_openenv(
        working_directory = str(tmp_path), openenv_class = _DummyClient, host = "127.0.0.1",
    )
    port_v6, client_v6 = rl_env.launch_openenv(
        working_directory = str(tmp_path), openenv_class = _DummyClient, host = "::",
    )
    assert port_v4 == port_v6 == 45000
    assert client_v4.base_url == "http://127.0.0.1:45000"
    assert client_v6.base_url == "http://[::1]:45000"
    assert len(spawned) == 2
    # Both live children are still tracked; neither overwrote the other.
    assert set(rl_env._OPENENV_CHILDREN) == {("127.0.0.1", 45000), ("::1", 45000)}


def test_ownership_does_not_transfer_between_addresses():
    """Our 127.0.0.1 child is not evidence that we own [::1] on the same port."""
    rl_env._OPENENV_CHILDREN[("127.0.0.1", 47000)] = _LiveChild()
    assert rl_env._openenv_child_alive("127.0.0.1", 47000)
    assert not rl_env._openenv_child_alive("::1", 47000)
    assert not rl_env._openenv_child_alive("localhost", 47000)


def test_exited_children_are_reaped_rather_than_retained(monkeypatch, tmp_path):
    """Holding the Popen suppresses the interpreter's own reap, so sweep instead."""
    rl_env._OPENENV_CHILDREN[("127.0.0.1", 48000)] = _DeadChild()
    rl_env._OPENENV_CHILDREN[("::1", 48001)] = _DeadChild()
    live = _LiveChild()
    rl_env._OPENENV_CHILDREN[("127.0.0.1", 48002)] = live
    _fake_launcher(monkeypatch, ports = [49000])
    rl_env.launch_openenv(working_directory = str(tmp_path), openenv_class = _DummyClient)
    # The two exited children are gone, the live one and the new one remain.
    assert ("127.0.0.1", 48000) not in rl_env._OPENENV_CHILDREN
    assert ("::1", 48001) not in rl_env._OPENENV_CHILDREN
    assert rl_env._OPENENV_CHILDREN[("127.0.0.1", 48002)] is live


def test_probe_advances_past_a_family_the_host_cannot_open(monkeypatch):
    """A container without IPv6 raises EAFNOSUPPORT on socket(); try the next one."""
    reached = []

    class _Connected:
        def settimeout(self, timeout): pass
        def connect_ex(self, address):
            reached.append(address)
            return 0
        def close(self): pass

    def refuse_ipv6(family, socktype, proto):
        if family == socket.AF_INET6:
            raise OSError(97, "Address family not supported by protocol")
        return _Connected()

    monkeypatch.setattr(rl_env, "socket", types.SimpleNamespace(
        getaddrinfo = lambda *a, **k: [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 9000, 0, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 9000)),
        ],
        socket = refuse_ipv6,
        SOCK_STREAM = socket.SOCK_STREAM,
        AF_INET = socket.AF_INET, AF_INET6 = socket.AF_INET6,
        error = OSError, gaierror = socket.gaierror,
    ))
    assert rl_env.is_port_open("localhost", 9000)
    assert reached == [("127.0.0.1", 9000)]


# --- the registry is shared mutable state, so it needs a lock ------------------

class _SlowPollChild:
    """poll() calls waitpid, which releases the GIL; this models that window."""
    def __init__(self, dead = False): self.dead = dead
    def poll(self):
        time.sleep(0.0005)
        return 1 if self.dead else None


@contextlib.contextmanager
def _registry_guard():
    """Whatever serialization the module offers, which is the point of the test.

    Taking the real lock when it exists and nothing when it does not is what keeps
    these two falsifiable: a helper that required the lock would raise inside its
    own thread on unlocked code, the contended access would never happen, and the
    test would pass by doing nothing.
    """
    lock = getattr(rl_env, "_OPENENV_CHILDREN_LOCK", None)
    if lock is None:
        yield
    else:
        with lock:
            yield


def test_reaping_survives_a_concurrent_launch():
    """Iterating while another thread registers raised 'dictionary changed size'."""
    for index in range(200):
        rl_env._OPENENV_CHILDREN[("127.0.0.1", 50000 + index)] = _SlowPollChild(dead = index % 2 == 0)
    errors = []

    def reap():
        try:
            rl_env._reap_exited_openenv_children()
        except BaseException as error:      # the failure mode is a RuntimeError
            errors.append(error)

    def register():
        for index in range(400):
            with _registry_guard():
                rl_env._OPENENV_CHILDREN[("::1", 60000 + index)] = _SlowPollChild()
            time.sleep(0.0001)

    reaper = threading.Thread(target = reap)
    registrar = threading.Thread(target = register)
    registrar.start(); reaper.start()
    reaper.join(); registrar.join()
    assert errors == [], f"reaping raised {errors!r}"
    # The sweep still did its job: every dead child is gone, every live one stays.
    assert all(child.poll() is None for child in rl_env._OPENENV_CHILDREN.values())


def test_a_liveness_check_cannot_delete_another_threads_live_child():
    """get-poll-pop is check-then-mutate: the pop must not outlive the check."""
    endpoint = ("127.0.0.1", 51000)
    rl_env._OPENENV_CHILDREN[endpoint] = _SlowPollChild(dead = True)
    replacement = _SlowPollChild(dead = False)

    def check(): rl_env._openenv_child_alive(*endpoint)

    def replace():
        time.sleep(0.0002)                  # lands inside the checker's poll()
        with _registry_guard():
            rl_env._OPENENV_CHILDREN[endpoint] = replacement

    checker = threading.Thread(target = check)
    replacer = threading.Thread(target = replace)
    checker.start(); replacer.start()
    checker.join(); replacer.join()
    assert rl_env._OPENENV_CHILDREN.get(endpoint) is replacement

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Static malicious-pattern scan for the llama.cpp converter we download and run.

GGUF export fetches ``convert_hf_to_gguf.py`` from llama.cpp master at runtime
(``llama_cpp.LLAMA_CPP_CONVERT_FILE``) and then hands those bytes to two
execution primitives:

  * the argparse defaults scraped out of them are passed to ``eval()`` inside the
    Unsloth process, and
  * the patched script is written under ``~/.unsloth/llama.cpp/`` and run with
    ``subprocess.run([sys.executable, ...])``.

There is no integrity check on the fetch, so whatever master holds at that moment
runs on the user's machine. This module is defence in depth for that window: it
is NOT a substitute for pinning the converter to a reviewed commit plus a sha256,
or for vendoring it. A static regex scan can be evaded by anyone who knows it is
there; what it buys is that an off-the-shelf stealer payload of the kind seen in
recent supply-chain waves does not run silently.

Scope is that one download and nothing else. On llama.cpp's current layout the
downloaded entrypoint imports a ``conversion`` package, and those modules come
from the local checkout or from a prebuilt bundle whose asset is checked against
a published sha256, so they are not part of the unverified fetch and are not
scanned here.

Why the logic lives here and not in ``scripts/``:
``pyproject.toml`` excludes ``scripts*`` from the wheel, so
``scripts/scan_packages.py`` does not exist on a pip-installed unsloth_zoo, which
is the only install where this check matters. The patterns below are therefore
vendored verbatim from that scanner; ``tests/security/test_converter_scan_pins_scan_packages.py``
fails if the two copies drift.

Behaviour on a match is a loud warning naming the rule, the evidence line and the
converter path, plus the ``UNSLOTH_LLAMA_CPP_SCRIPTS_DIR`` escape hatch. It does
not block by default: these are heuristics, and a false positive that refuses to
export GGUF would be a worse regression than the warning is a win. Set
``UNSLOTH_CONVERTER_SCAN_STRICT=1`` to turn a finding on downloaded bytes into a
hard failure, or ``UNSLOTH_DISABLE_CONVERTER_SCAN=1`` to skip the scan entirely.
"""

import ast
import logging
import os
import re
import threading
import string
from dataclasses import dataclass

__all__ = [
    "CRITICAL",
    "HIGH",
    "ConverterScanError",
    "ConverterScanFinding",
    "RE_ARGPARSE_DEFAULT",
    "VENDORED_PATTERNS",
    "PATTERNS_NOT_VENDORED",
    "scan_converter_source",
    "scan_is_disabled",
    "scan_is_strict",
    "warn_on_suspicious_converter",
]

logger = logging.getLogger(__name__)

CRITICAL = "CRITICAL"
HIGH = "HIGH"

# Env switches, read per call so a caller can flip them between exports.
ENV_DISABLE_SCAN = "UNSLOTH_DISABLE_CONVERTER_SCAN"
ENV_STRICT_SCAN = "UNSLOTH_CONVERTER_SCAN_STRICT"

# Cap the scanned text. The converter entrypoint is tens of KB on the package
# layout and ~400 KB on the old monolith; anything far past that is either not a
# converter or is trying to make the DOTALL patterns below quadratic.
MAX_SCAN_BYTES = 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# Patterns, vendored verbatim from scripts/scan_packages.py
# ---------------------------------------------------------------------------
# Do not edit these in place. They are byte-pinned to the canonical scanner by
# tests/security/test_converter_scan_pins_scan_packages.py: change one there
# first, then copy it here. Only the patterns check_py_file() consumes are
# vendored; the .pth / .js / shell / workflow patterns have no meaning for a
# single downloaded Python file.

# Subprocess / OS exec patterns
RE_SUBPROCESS = re.compile(
    r"\bsubprocess\s*\.\s*(Popen|call|run|check_call|check_output)\b"
    r"|\bos\s*\.\s*(system|popen|exec[lv]p?e?)\b",
)

# Encoding / obfuscation
RE_BASE64 = re.compile(
    r"\bbase64\s*\.\s*(b64decode|decodebytes|b32decode|b16decode)\b"
    r"|\bcodecs\s*\.\s*decode\b",
)

# exec / eval
RE_EXEC_EVAL = re.compile(r"\b(exec|eval)\s*\(")

# Network APIs (excludes urllib.parse which is pure string manipulation)
RE_NETWORK = re.compile(
    r"\burllib\.request\b"
    r"|\burlopen\s*\("
    r"|\brequests\s*\.\s*(get|post|put|patch|delete|head|Session)\b"
    r"|\bhttpx\s*\.\s*(get|post|put|patch|delete|Client|AsyncClient)\b"
    r"|\bsocket\s*\.\s*(socket|create_connection)\b"
    r"|\bhttp\.client\b"
    r"|\bhttp\.server\b",
)

# Hosts a converter legitimately talks to. Upstream's own gguf-py/gguf/utility.py
# reads HF_TOKEN and sends it to huggingface.co as an Authorization header, which
# is what downloading a gated model looks like, so the env-harvest rule fired on
# every clean checkout of llama.cpp master and UNSLOTH_CONVERTER_SCAN_STRICT
# refused the export.
MODEL_HUB_HOSTS = frozenset(("huggingface.co", "hf.co"))

# The scheme only. The authority is taken from there to the first URL delimiter
# and then required to BE a hostname, rather than matched by a pattern that
# tries to know where a URL ends. Four review rounds found five spellings that
# ended it early and left only the hub recorded: user information before an @,
# an uppercase scheme, a bytes literal, a backslash, and a space. Each is a
# character this now keeps inside the authority, where it fails to look like a
# hostname and the allowance is refused. Enumerating the spellings was losing;
# saying what an allowed destination looks like is not.
RE_URL_SCHEME = re.compile(r"https?://", re.IGNORECASE)
RE_AUTHORITY_END = re.compile(r"[/?#]")
RE_HOSTNAME = re.compile(r"^[A-Za-z0-9.\-]+(?::[0-9]+)?$")
# A URL whose authority is already over: what follows cannot change the host.
RE_AUTHORITY_CLOSED = re.compile(r"https?://[^/?#]*[/?#]", re.IGNORECASE)


# Network APIs the allowance cannot vouch for, so their presence refuses it.
#
# socket and http.client name their destination as a bare host rather than a
# URL: a file could carry a hub URL and open socket.create_connection((
# "evil.example", 443)) beside it, and the allowance would call that talking
# only to the hub.
#
# urllib is here for the other half of the question. It expresses a write as
# Request(..., data = ...), Request(..., method = "POST") or urlopen(..., data =
# ...), none of which is an attribute called post, so the write check below
# cannot see them. Neither API appears anywhere in the real gguf-py or
# conversion packages, so refusing on them costs nothing upstream.
RE_UNVOUCHABLE_NETWORK = re.compile(
    r"\bsocket\s*\.\s*(?:socket|create_connection)\b"
    r"|\bhttp\.(?:client|server)\b"
    r"|\burllib\.request\b"
    r"|\burlopen\s*\(",
)


# Primitives that build a string, and so a host, out of view of the folding in
# _literal_text: bytes.fromhex("68747470733a2f2f6576696c2e6578616d706c65")
# .decode() is "https://evil.example" with no URL anywhere in the source for the
# literal walk to read. Folding each of them would mean an evaluator; refusing
# them costs nothing, because none appears anywhere in the real gguf-py or
# conversion packages, and the allowance is for a file that names its
# destination plainly, which none of these does.
RE_UNVOUCHABLE_STRING_BUILD = re.compile(
    r"\bfromhex\s*\("
    r"|\bunhexlify\s*\("
    r"|\bmaketrans\s*\("
    r"|\.translate\s*\("
    r"|\.to_bytes\s*\("
    r"|\bcodecs\s*\.\s*(?:decode|encode)\b"
    r"|\bbase64\s*\.\s*\w+\s*\("
    # unquote("https%3A%2F%2Fevil.example%2Fc") is a destination too, and the
    # literal that spells it carries no scheme for the walk to find. urlparse is
    # NOT here: upstream's utility.py parses URLs with it, and parsing one is not
    # building one.
    r"|\bunquote(?:_plus|_to_bytes)?\s*\(",
)


# The shape the allowance is actually for: a token-authenticated READ from the
# hub. Anything else sent to a writable multi-tenant host is not obviously
# benign, since a write-capable token can create a public repo there and use it
# as a channel any attacker can read back.
HUB_TOKEN_ENV_NAMES = frozenset((
    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN",
))

# `request`, `send` and `stream` are here because session.request("POST", ...),
# Session.send(Request("POST", ...).prepare()) and httpx Client.stream("POST",
# ...) are writes the named methods do not cover, and reading the method out of
# any of them would mean following an argument that need not be a literal.
# Upstream names none of these anywhere at all, in any position, so taking every
# one of them as a write costs nothing there.
WRITE_METHODS = frozenset((
    "post", "put", "patch", "delete", "request", "send", "stream",
))

# The environment under another name. os.environ and os.getenv are attributes
# and are read normally; these are the bare names an alias or a from-import
# leaves behind, which _env_reads does not inspect.
ENV_ALIAS_NAMES = frozenset(("environ", "getenv", "environb", "getenvb"))

# Introspection, which is the only way a docstring becomes a value. Spelling
# them out one at a time was losing: __doc__, fn.__doc__, getattr(fn,
# "__doc__"), vars(fn)["__doc__"] and inspect.getdoc(fn) all read the same
# string. Naming any of these at all keeps docstrings in the scan.
DOCSTRING_READERS = frozenset((
    "__doc__", "vars", "__dict__", "inspect", "pydoc", "getdoc", "help",
    "__getattribute__",
))

RE_WHOLE_ENV = re.compile(
    r"\bos\.environ\s*\.\s*copy\s*\("
    r"|\bdict\s*\(\s*os\.environ\s*\)"
    r"|\bos\.environ\.items\s*\(",
)


def _is_os_environ(node):
    return isinstance(node, ast.Attribute) and node.attr == "environ"


def _env_reads(tree):
    """`(names, dynamic)` for the environment this file reads by name.

    `dynamic` is set when a read names its variable at runtime, as in
    os.environ[SECRET_NAME]. Nothing here can say which variable that is, and an
    unattributable read used to pass the allow-list by leaving the set empty.

    Only os.environ and os.getenv count. Any object's .get() used to, so an
    ordinary config.get("API_KEY") beside a normal hub download read as an
    environment secret and put the false positive back.
    """
    names, dynamic = set(), False
    # Every os.environ in the file, and the ones this walk actually accounts
    # for. `env = os.environ` followed by env["AWS_SECRET_ACCESS_KEY"] reads a
    # credential this never sees, and the short name set then satisfied the
    # allow-list. An os.environ that is passed around rather than subscripted
    # here is a read this cannot attribute, which is the same as a dynamic one.
    environs = {
        id(node) for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "environ"
    }
    accounted = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and _is_os_environ(node.value):
            accounted.add(id(node.value))
            index = node.slice
            if isinstance(index, ast.Constant) and isinstance(index.value, str):
                names.add(index.value)
            else:
                dynamic = True
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            func = node.func
            is_env_get = func.attr == "get" and _is_os_environ(func.value)
            is_getenv = func.attr in ("getenv", "getenvb") and (
                (isinstance(func.value, ast.Name) and func.value.id == "os")
                or (isinstance(func.value, ast.Attribute) and func.value.attr == "os")
            )
            if is_env_get:
                accounted.add(id(func.value))
            # Only .get(). Marking every method on os.environ as accounted for
            # let .values() collect every secret and .pop("AWS_SECRET_ACCESS_KEY")
            # take a named one, with the collected set still reading HF_TOKEN
            # alone. Anything else leaves that os.environ unaccounted, which
            # makes the reads dynamic and refuses the allowance.
            if func.attr in ("getenv", "getenvb") and not is_getenv:
                # import os as o, then o.getenv("AWS_SECRET_ACCESS_KEY"). The
                # receiver says nothing, so neither does the name it reads.
                dynamic = True
                continue
            if not (is_env_get or is_getenv):
                continue
            # os.environ.get(key = "HF_TOKEN") is the same read spelled with a
            # keyword, and calling it dynamic refused a legitimate download.
            key = node.args[0] if node.args else next(
                (
                    keyword.value for keyword in node.keywords
                    if keyword.arg == "key"
                ),
                None,
            )
            # bytes as well as str: os.getenvb(b"AWS_SECRET_ACCESS_KEY") is
            # the same read on Unix, and exempting it left the collected set
            # holding the hub token alone.
            name = _literal_text(key) if key is not None else None
            if name is not None:
                names.add(name)
            else:
                dynamic = True
    if environs - accounted:
        dynamic = True
    return names, dynamic


def _writes_anything(tree):
    """Whether this file so much as NAMES a write method.

    Matching only receivers spelled `session` or `client` missed
    `s = requests.Session(); s.post(...)`, which is the writable-hub channel
    this is here to refuse. Matching only CALLS then missed the same write one
    rename later: `post = requests.post` and `from requests import post` both
    leave the call site an ast.Name, so post(HUB, data = os.environ["HF_TOKEN"])
    read as no write at all. Following the alias means following assignment
    through every shape; naming one at all is enough to refuse, and the 21 real
    gguf-py and converter modules contain no reference to any of these names,
    in any position, so it costs nothing upstream.
    """
    return any(
        (isinstance(node, ast.Attribute) and node.attr in WRITE_METHODS)
        or (isinstance(node, ast.Name) and node.id in WRITE_METHODS)
        # vars(requests)["post"] and requests.__dict__["post"] leave the method
        # name as a string and nothing else, and "po" + "st" is the same string
        # spelled to miss a Constant check. None of these names appears as a
        # string constant anywhere in the 21 real modules either.
        or _literal_text(node) in WRITE_METHODS
        or (
            isinstance(node, ast.ImportFrom)
            and any(alias.name in WRITE_METHODS for alias in node.names)
        )
        for node in ast.walk(tree)
    )


# Modules whose attributes are the things this refuses by name. A getattr on
# one of them with a name this cannot read is a lookup this cannot vouch for.
SENSITIVE_MODULES = frozenset(("os", "requests", "httpx", "urllib", "socket"))


# Modules whose members name a destination as a bare host, or write in a way
# the method names cannot see. urllib.parse is deliberately absent: it is string
# manipulation, RE_NETWORK excludes it for the same reason, and upstream's
# utility.py imports urlparse from it twice.
# Clients whose destination is a bare host rather than a URL this can read, so
# no walk over the literals can say where they send. smtplib.SMTP(
# "evil.example").sendmail(..., os.environ["HF_TOKEN"]) is a whole exfiltration
# beside an honest hub download. requests, httpx and aiohttp are deliberately
# absent: they take a URL, which this reads.
UNVOUCHABLE_MODULES = (
    "socket", "http.client", "http.server", "urllib.request",
    "smtplib", "ftplib", "poplib", "imaplib", "nntplib", "telnetlib",
    "xmlrpc.client", "paramiko", "pysftp", "websocket", "websockets",
)

# The same door for the string builders. RE_UNVOUCHABLE_STRING_BUILD reads
# qualified spellings, so `from base64 import b64decode as d` left it nothing to
# match and `d("aHR0cHM6...")` decoded an attacker URL beside an unused hub
# literal. Modules whose members rebuild a string, and the member names that do
# it under any module. Upstream imports none of these anywhere.
STRING_BUILDER_MODULES = ("base64", "binascii", "codecs", "quopri", "uu")
STRING_BUILDER_NAMES = frozenset((
    "b64decode", "b64encode", "b32decode", "b16decode", "b85decode",
    "a85decode", "standard_b64decode", "urlsafe_b64decode", "decodebytes",
    "unhexlify", "hexlify", "decode", "encode",
    "unquote", "unquote_plus", "unquote_to_bytes",
))

# Functions that BUILD a URL out of one. urlparse and urlsplit are not here:
# upstream reads its URL with urlparse, and reading one is not rebuilding it.
# The methods a URL OBJECT is rewritten by. httpx.URL(HUB).copy_with(host =
# "evil.example") and yarl.URL(HUB).with_host(...) wrap the carrier in a call
# first, and a carrier deliberately does not survive a call: upstream writes
# cls.get_list_tensors(url).items() and response = requests.get(url), and
# tracking what came back out of those refused the very file this exists for.
# So the receiver is searched for a carrier only when the method is one that
# rebuilds a URL, which none of upstream's calls on a URL are. _replace is
# here for the namedtuple urlparse returns.
# The result objects urlsplit and urlparse return, which assemble a whole
# destination out of fields with no URL literal anywhere:
# ParseResult("https", "evil.example", "/c", "", "", "").geturl(). Upstream
# reads .scheme and .netloc off a parse result and never calls geturl.
URL_ASSEMBLERS = frozenset((
    "ParseResult", "SplitResult", "ParseResultBytes", "SplitResultBytes",
    "DefragResult", "DefragResultBytes", "geturl",
))

URL_REWRITE_METHODS = frozenset((
    "copy_with", "copy_set_param", "copy_add_param", "copy_merge_params",
    "with_host", "with_scheme", "with_path", "with_query", "with_port",
    "with_user", "with_password", "with_fragment", "with_netloc",
    "set_host", "set_scheme", "update_query", "_replace",
))

URL_BUILDERS = frozenset((
    "urljoin", "urlunparse", "urlunsplit", "urldefrag",
))


def _dynamic_imports(tree, aliases = None):
    """`(names, unreadable)` for every import spelled as a call.

    __import__("smtplib") and importlib.import_module("smtplib") bind a module
    with no Import node anywhere, so every check that reads the import lines saw
    nothing at all. A name this cannot read is worse than a known one, so it
    counts as unreadable rather than as no import.
    """
    if aliases is None:
        aliases = _call_aliases(tree)
    names, unreadable = set(), False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _called_name(node, aliases) not in ("__import__", "import_module"):
            continue
        text = _literal_text(node.args[0]) if node.args else None
        if text is None or UNKNOWN_PIECE in text:
            unreadable = True
        else:
            names.add(text)
    return names, unreadable


def _imports_an_unvouchable_api(tree, aliases = None):
    """Whether a connection API arrives by from-import.

    RE_UNVOUCHABLE_NETWORK reads qualified spellings, so
    `from socket import create_connection` left nothing for it to match, and
    `import socket as s` left it only `s.create_connection`, which it does not
    recognise either. Both let a token go to a bare host beside a benign hub GET.
    """
    def unvouchable(name):
        return any(
            name == module or name.startswith(module + ".")
            for module in UNVOUCHABLE_MODULES
        )

    dynamic, unreadable = _dynamic_imports(tree, aliases)
    if unreadable or any(unvouchable(name) for name in dynamic):
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if unvouchable(node.module):
                return True
        elif isinstance(node, ast.Import):
            # import socket as s: the qualified pattern sees no socket. call
            # anywhere, so the import itself is the only place it is named.
            if any(unvouchable(alias.name) for alias in node.names):
                return True
    return False


def _builds_text_from_numbers(tree):
    """Whether text is assembled out of character codes.

    bytearray([104, 116, 116, 112, 115]).decode() is a URL with no URL in it,
    and so is "".join(chr(c) for c in codes). The existing obfuscation rule
    wants exec or eval beside it, which this needs nothing of: the destination
    is simply spelled in numbers.

    Narrow at the receiver, because upstream really does write
    bytearray(get_data_by_range(...)) in the one file this allowance reaches:
    only a container of values written out in the source counts, never bytes
    that came back from a call. chr is refused outright, and appears in
    vocab.py and nowhere that harvests the environment.
    """
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name) and node.id == "chr"
        ) or (
            isinstance(node, ast.Attribute) and node.attr == "chr"
        ):
            return True
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.attr if isinstance(node.func, ast.Attribute)
            else node.func.id if isinstance(node.func, ast.Name) else ""
        )
        if name in ("bytes", "bytearray") and node.args and isinstance(
            node.args[0],
            (ast.List, ast.Tuple, ast.Set, ast.ListComp, ast.SetComp,
             ast.GeneratorExp, ast.DictComp),
        ):
            return True
    return False


def _templates_are_oversized(tree):
    """Whether any formatting template is longer than this will parse.

    Checked by length alone, before anything materialises the fields: the
    template is the input, and a megabyte of repeated {} is half a million
    tuples out of a file the scan accepts at up to 8 MiB.
    """
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("format", "format_map")
            and isinstance(node.func.value, ast.Constant)
            and isinstance(node.func.value.value, str)
            and len(node.func.value.value) > MAX_TEMPLATE
        ):
            return True
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Mod)
            and isinstance(node.left, ast.Constant)
            and isinstance(node.left.value, str)
            and len(node.left.value) > MAX_TEMPLATE
        ):
            return True
    return False


def _joins_something_unreadable(tree):
    """Whether a literal separator joins pieces this cannot enumerate.

    "".join(x for x in ("https", "://evil.example/c")) is a whole URL, and the
    walk sees two pieces neither of which carries a scheme. Only a sequence
    written out is folded, and returning no text for anything else left the
    destination unnamed rather than unreadable.

    The separator is read too: str().join(("htt", "ps://evil.example/c")) is a
    URL whose scheme is broken in the middle by the hole the separator folded
    to, so nothing matched a scheme there either.

    One argument, which leaves os.path.join(a, b) alone. No file that reaches
    this allowance joins anything: gguf-py/gguf/utility.py has no .join at all.
    """
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "join"
            and len(node.args) == 1
            and not node.keywords
        ):
            continue
        if _join_parts(node) is None:
            return True
        if _literal_text(node.func.value) is None:
            # A separator this cannot read, folded as a hole, hid a whole URL
            # between two readable pieces: str().join(("htt", "ps://evil.
            # example/c")) reads as htt<hole>ps://..., where the hole breaks
            # the scheme and nothing was recorded as a destination.
            return True
    return False


def _call_aliases(tree, assignments = None):
    """`{local name: original name}` for imports and for rebinding assignments.

    `from importlib import import_module as im` leaves the dynamic import
    spelled im(...), and `j = urljoin` rebinds a builder with no import in
    sight. Both are the same question a call site asks: what is this really.
    """
    aliases = _import_aliases(tree)
    for name, value in (assignments or _single_assignments(tree)).items():
        if isinstance(value, ast.Name):
            aliases[name] = aliases.get(value.id, value.id)
        elif isinstance(value, ast.Attribute):
            aliases[name] = aliases.get(value.attr, value.attr)
    return aliases


def _called_name(node, aliases):
    """The name a call site uses, read through any alias."""
    name = (
        node.func.attr if isinstance(node.func, ast.Attribute)
        else node.func.id if isinstance(node.func, ast.Name) else ""
    )
    return aliases.get(name, name)


def _import_aliases(tree):
    """`{local name: imported name}` for every alias an import binds.

    `from urllib.parse import urljoin as j` leaves the call spelled j(...), and
    comparing the call site against a set of builder names missed it, exactly as
    the decoder and docstring checks missed their own aliases.
    """
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name.split(".")[-1]
    return aliases


def _imports_a_string_builder(tree, aliases = None):
    """Whether a string decoder arrives by import, under any name.

    An alias erases the only spelling RE_UNVOUCHABLE_STRING_BUILD can read: with
    `from base64 import b64decode as d` the call is `d(...)`, and with
    `import base64 as b` it is `b.b64decode(...)`. Either one decoded an
    attacker URL while an unused hub literal still granted the allowance.
    """
    def builder_module(name):
        return any(
            name == module or name.startswith(module + ".")
            for module in STRING_BUILDER_MODULES
        )

    dynamic, _ = _dynamic_imports(tree, aliases)
    if any(builder_module(name) for name in dynamic):
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and builder_module(node.module):
                return True
            if any(alias.name in STRING_BUILDER_NAMES for alias in node.names):
                return True
        elif isinstance(node, ast.Import):
            if any(builder_module(alias.name) for alias in node.names):
                return True
    return False


def _reaches_through_getattr(tree):
    """Whether a write or the environment is reached by a computed lookup.

    `f = getattr(requests, "post")` holds no Attribute and no Name spelled post,
    so the write checks saw nothing and a POST of HF_TOKEN to the hub scanned
    clean. A constant name is read here and refused; a name that is NOT constant
    is refused only on the modules above, because upstream calls getattr with a
    computed name nine times, all on its own objects, and refusing those would
    refuse real converter modules.
    """
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
        ):
            continue
        name = _literal_text(node.args[1])
        if name is not None:
            if name in WRITE_METHODS or name in ENV_ALIAS_NAMES:
                return True
            continue
        target = node.args[0]
        if isinstance(target, ast.Name) and target.id in SENSITIVE_MODULES:
            return True
        if isinstance(target, ast.Attribute) and target.attr in SENSITIVE_MODULES:
            return True
    return False


def _aliases_the_environment(tree):
    """Whether the environment is reachable here under another name.

    `getenv = os.getenv` and `from os import getenv` both leave the read an
    ast.Name, which _env_reads does not inspect, so getenv("AWS_SECRET_ACCESS_KEY")
    beside a legitimate os.environ["HF_TOKEN"] left the collected set holding the
    hub token alone and the allowance was granted. `from os import environ` does
    the same to the subscript form.

    `os.environ` and `os.getenv` themselves are attributes and are unaffected:
    this is about the bare name. Upstream reads the environment only through
    os.environ, so refusing the aliases costs nothing.
    """
    called = {
        id(node.func) for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    return any(
        (isinstance(node, ast.Name) and node.id in ENV_ALIAS_NAMES)
        # vars(os)["environ"] and os.__dict__["environ"] leave the name as a
        # string and nothing else, and "en" + "viron" is the same string spelled
        # to miss a Constant check. None of these appears as a string constant
        # in the 21 real modules.
        or _literal_text(node) in ENV_ALIAS_NAMES
        or (
            isinstance(node, ast.ImportFrom)
            and any(alias.name in ENV_ALIAS_NAMES for alias in node.names)
        )
        # read_secret = os.getenv keeps no reserved spelling anywhere, so the
        # name it lands under says nothing. An os.getenv that is not called
        # right here is one going somewhere this does not follow. os.environ is
        # already covered: _env_reads counts every one it did not account for.
        or (
            isinstance(node, ast.Attribute)
            and node.attr in ("getenv", "getenvb", "environb")
            and id(node) not in called
        )
        for node in ast.walk(tree)
    )


# Only %s, %r and %% are folded below, under a mapping key or without one. A
# width turns "%2000000000d" % 1 into a two gigabyte string, and reading a file
# is not a reason to allocate one. The key has to be skipped over rather than
# read as the conversion: "%(host)s" is an ordinary %s, and calling it unsafe
# left every mapping template unfoldable, holes and all.
RE_UNSAFE_PERCENT = re.compile(r"%(?:\([^)]*\)[^sr%]|(?!\()[^sr%])")

# Stands in for a piece of a string this cannot read. It is not a hostname
# character, so a destination interrupted by one fails the hostname check
# instead of being read as the text on either side of the hole.
UNKNOWN_PIECE = "\x00"

# Passes over the tree allowed to settle which names carry a URL. See
# _reshapes_a_url: real files need one or two.
MAX_CARRIER_PASSES = 16

# Ceiling on a folded join. A join multiplies, unlike + and %, whose result is
# bounded by the source that spells it: a 100 KB separator with 10000 elements
# is 130 KB of source and a 1 GB string, which took 15.7 seconds and 1.7 GB of
# resident memory inside the decision about whether an export may run. Over the
# ceiling the join is simply not folded, so the walk reads its separator and its
# elements individually and a destination inside one is still seen.
MAX_FOLDED_JOIN = 1 << 20


# A template longer than this is not parsed at all. Formatter().parse turns a
# 1 MiB template of repeated {} into half a million tuples before any output
# check can run, and the scan accepts sources up to 8 MiB. No converter writes
# a 64 KiB format string, and one that does is refused rather than read.
MAX_TEMPLATE = 1 << 16


class _FoldBudgetExceeded(Exception):
    """Raised when one allowance decision has folded more text than it may.

    Each fold carries its own output ceiling, and the destination walk charges
    what it reads against a shared budget, but the folding itself happens in
    several places: the carrier fixpoint alone reads every assignment in the
    file up to sixteen times. Two hundred joins of a megabyte apiece are inside
    every individual ceiling and still hundreds of megabytes of work. The
    budget is on the work, and running out of it refuses, since the caller
    turns any exception here into "do not suppress anything".
    """


# Generous next to a real converter, which folds a few kilobytes: the whole
# gguf-py package folds well under one megabyte across every pass.
MAX_TOTAL_FOLD = 32 * MAX_FOLDED_JOIN

# Per thread, not per module: two exports running at once shared one counter,
# so the first to finish cleared it under the second, their charges were added
# together, and the bound stopped meaning anything for either.
_fold_state = threading.local()


def _charge_fold(text):
    budget = getattr(_fold_state, "budget", None)
    if budget is None or text is None:
        return text
    budget -= len(text)
    _fold_state.budget = budget
    if budget < 0:
        raise _FoldBudgetExceeded("folded more text than one decision may")
    return text


def _fields_are_plain(template):
    """Whether every replacement field is a bare name, with no spec at all.

    A spec is not folded: "{:>1000000000}".format("x") is a gigabyte, and a
    converter that needs one of those in a URL does not exist. Read with the
    formatter rather than a pattern, because a pattern cannot see into a NESTED
    spec: "{0:{1}}".format("x", "10000000000") has one, the character classes
    could not span the inner braces, the length estimate stayed tiny, and
    formatting it took 8.5 seconds and ten gigabytes.

    Upstream formats with plain fields in 15 places, so folding rather than
    refusing is what keeps those files readable here.
    """
    if len(template) > MAX_TEMPLATE:
        return False                    # too long to read, so never folded
    try:
        fields = list(string.Formatter().parse(template))
    except Exception:
        return False                    # unparseable is unfoldable
    return not any(
        name is not None and (spec or conversion)
        for _, name, spec, conversion in fields
    )


def _longest_argument(node):
    """Length of the longest literal argument of a call, 0 when there is none."""
    texts = [_literal_text(argument) for argument in node.args]
    texts += [_literal_text(keyword.value) for keyword in node.keywords]
    return max((len(text) for text in texts if text is not None), default = 0)


# One percent conversion, in the full printf shape the operator accepts:
# "%(name)-#010.3lf" is one field and "%%" is an escaped sign, not a field.
RE_PERCENT_FIELD = re.compile(
    r"%(?:\((?P<key>[^)]*)\))?[-#0 +]*(?:\d+|\*)?(?:\.(?:\d+|\*))?[hlL]?"
    r"[diouxXeEfFgGcrsa]"
)


def _percent_is_oversized(template, values):
    """Whether this formatting would materialise more than the fold ceiling.

    The aggregate budget is spent on what the walk READS, and the operator runs
    before that: a 53 KB source with a thousand %(x)s fields and a 50 KB value
    built 50 MB first, and both factors scale inside the 8 MiB the scan accepts.
    The estimate is one longest value per field, which is exact for %s and %r
    and the only conversions folded here.
    """
    if len(template) > MAX_TEMPLATE:
        return True                     # too long to read, so never folded
    fields = len(RE_PERCENT_FIELD.findall(template))
    longest = max((len(value) for value in values), default = 0)
    return len(template) + fields * longest > MAX_FOLDED_JOIN


def _percent_holes(template):
    """The template with each percent conversion replaced by a hole.

    Returning None for a formatting this cannot evaluate read as no text at all,
    where the join and the f-string leave holes: "%s://%s" % (scheme, host)
    spells its own scheme that way and no literal carried a URL. It also refused
    honest code, since "/api/models/%s" % name appended to the hub URL was text
    the authority check could not read either.
    """
    pieces, index = [], 0
    for match in RE_PERCENT_FIELD.finditer(template):
        pieces.append(template[index:match.start()])
        pieces.append(UNKNOWN_PIECE)
        index = match.end()
    pieces.append(template[index:])
    return "".join(pieces).replace("%%", "%")


# The callable spellings of the plus operator, which build the same string it
# does. set.add is not one of them: these take two arguments.
ADD_FUNCTIONS = frozenset(("add", "concat", "iadd", "iconcat"))


def _added_text(node):
    """The text of a `+` spelled as a call, or None.

    operator.add("https", "://evil.example/c") and
    "https".__add__("://evil.example/c") build the same string the operator
    does, and reading only the operator left neither literal holding a URL.
    """
    if not isinstance(node, ast.Call) or node.keywords:
        return None
    if isinstance(node.func, ast.Attribute) and node.func.attr == "__add__":
        operands = [node.func.value, *node.args]
    elif isinstance(node.func, ast.Attribute) and node.func.attr in ADD_FUNCTIONS:
        operands = list(node.args)
    else:
        return None
    if len(operands) != 2:
        return None
    left = _literal_text(operands[0])
    right = _literal_text(operands[1])
    if left is None or right is None:
        return None
    if len(left) + len(right) > MAX_FOLDED_JOIN:
        return None                     # the same output ceiling as the join
    return left + right


def _replace_text(node):
    """The text of a literal `"...".replace(old, new)`, or None.

    "httpsX//evil.example/c".replace("X", ":") is a URL whose scheme does not
    exist until the call runs, so the walk read a literal with no scheme in it
    and recorded no destination at all while a hub literal elsewhere granted the
    allowance. Folding it reads the destination the way the join, the f-string
    and the format fold do.

    Only a literal receiver with literal arguments. Upstream's replaces are all
    `parameter.replace(" ", "-")`, whose receiver is not a literal, so this
    never reaches them; a literal receiver whose arguments cannot be read is
    refused outright in _reshapes_a_url instead.
    """
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "replace"
        and not node.keywords
        and len(node.args) == 2
    ):
        return None
    receiver = _literal_text(node.func.value)
    old = _literal_text(node.args[0])
    new = _literal_text(node.args[1])
    if receiver is None or old is None or new is None:
        return None
    if len(receiver) * (len(new) + 1) > MAX_FOLDED_JOIN:
        return None                     # the same output ceiling as the join
    try:
        return receiver.replace(old, new)
    except (TypeError, ValueError):
        return None


def _rewrites_a_constant(node):
    """Whether a literal string is rewritten by arguments this cannot read.

    "httpsX//evil.example/c".replace(marker, colon) manufactures its scheme out
    of names, and _replace_text can only fold the arguments it can read. The
    receiver being a literal is what makes this narrow: upstream rewrites
    parameters, never constants.
    """
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("replace", "translate")
    ):
        return False
    if _literal_text(node.func.value) is None:
        return False
    return any(
        _literal_text(argument) is None
        for argument in [*node.args, *(k.value for k in node.keywords)]
    )


def _format_holes(node, template):
    """The template with every field replaced by a hole, or None.

    Returning None for a format this cannot evaluate was the one builder that
    read as no text at all, where the join and the f-string both leave holes.
    "{p[s]}://{p[h]}".format(p = parts) spells its own scheme that way, and no
    literal in the file then carried a URL for the walk to refuse.
    """
    if len(template) > MAX_TEMPLATE:
        return None
    try:
        fields = list(string.Formatter().parse(template))
    except Exception:
        return None
    pieces = []
    for literal, name, _spec, _conversion in fields:
        pieces.append(literal)
        if name is not None:
            pieces.append(UNKNOWN_PIECE)
    return "".join(pieces)


def _format_text(node):
    """The text of a literal `"...".format(...)`, or None.

    Upstream spells plenty of strings this way, so refusing the builder would
    refuse real converter modules. Folding it reads the destination instead:
    "{}://{}".format("https", "evil.example/collect") carries its scheme in no
    single piece, exactly like the join and f-string splits.
    """
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("format", "format_map")
    ):
        return None
    template = _literal_text(node.func.value)
    if template is None:
        return None
    if not _fields_are_plain(template):
        return _format_holes(node, template)
    if len(template) * (1 + _longest_argument(node)) > MAX_FOLDED_JOIN:
        # One argument referenced by thousands of {0} fields expands far past
        # the input cap, so the ceiling is on the OUTPUT, as it is for joins.
        return _format_holes(node, template)
    if node.func.attr == "format_map":
        if node.keywords or len(node.args) != 1 or not isinstance(node.args[0], ast.Dict):
            return _format_holes(node, template)
        mapping = {}
        for key, value in zip(node.args[0].keys, node.args[0].values):
            name = _literal_text(key) if key is not None else None
            text = _literal_text(value)
            if name is None or text is None:
                return _format_holes(node, template)
            mapping[name] = text
        try:
            return template.format_map(mapping)
        except Exception:
            # Any exception at all: "{0[x]}".format("a") raises TypeError, and
            # one unreachable expression like it made the whole scan raise,
            # which the caller turns into "continue with no findings".
            return _format_holes(node, template)
    arguments = [_literal_text(argument) for argument in node.args]
    keywords = {
        keyword.arg: _literal_text(keyword.value)
        for keyword in node.keywords if keyword.arg is not None
    }
    if any(value is None for value in arguments) or any(
        value is None for value in keywords.values()
    ) or len(node.keywords) != len(keywords):
        return _format_holes(node, template)
    try:
        return template.format(*arguments, **keywords)
    except Exception:
        return _format_holes(node, template)    # see format_map above


def _join_nodes(node):
    """`(separator, elements)` for `sep.join([...])`, or None.

    Only a sequence written out here: `"".join(parts)` names elements this
    cannot enumerate, and nothing is gained by pretending otherwise.
    """
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and not node.keywords
        and len(node.args) == 1
        and isinstance(node.args[0], (ast.List, ast.Tuple))
    ):
        return None
    return node.func.value, node.args[0].elts


def _join_parts(node):
    """The text of a `sep.join([...])`, with holes for what cannot be read."""
    nodes = _join_nodes(node)
    if nodes is None:
        return None
    separator_node, element_nodes = nodes
    separator = _literal_text(separator_node)
    separator = UNKNOWN_PIECE if separator is None else separator
    elements = []
    for element in element_nodes:
        # `or` here turned a folded "" into the sentinel, and
        # "".join(("https://huggingface.co", "")) then read as a host with an
        # unreadable piece stuck to it: a CRITICAL on a benign download.
        text = _literal_text(element)
        elements.append(UNKNOWN_PIECE if text is None else text)
    size = sum(map(len, elements)) + len(separator) * max(len(elements) - 1, 0)
    if size > MAX_FOLDED_JOIN:
        return None
    return separator, elements


def _literal_text(node):
    """As _literal_text_uncharged, with every fold charged to the work budget."""
    return _charge_fold(_literal_text_uncharged(node))


def _literal_text_uncharged(node):
    """The text a constant expression evaluates to, or None when it is not one.

    `+` and `%` are folded because ast.parse does not fold them: "https://" +
    "evil.example/collect" is two Constants, neither of which names a host, so
    that destination was recorded as no destination at all while a hub literal
    elsewhere in the file still granted the allowance. Same for "%s://%s" %
    ("https", "evil.example"), which spells the scheme itself past the check.
    An operand that is not a literal makes the whole expression unreadable here,
    which leaves it exactly where the dynamic destinations already sit.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return node.value
        if isinstance(node.value, bytes):
            return node.value.decode("utf-8", "replace")
        return None
    if isinstance(node, ast.JoinedStr):
        # An f-string is the one construction that arrived here already split.
        # f"https://{''}evil.example/log" left the scheme in one constant piece
        # and the whole attacker hostname, in plain sight, in another that no
        # longer had a scheme in front of it, so no host was read from it at all
        # and a hub literal elsewhere granted the allowance. A piece whose value
        # is not a literal becomes UNKNOWN_PIECE rather than nothing, so the
        # text around a hole is never read as though the hole were not there.
        pieces = []
        for part in node.values:
            if isinstance(part, ast.FormattedValue):
                known = (
                    None if (part.conversion not in (-1, None) or part.format_spec)
                    else _literal_text(part.value)
                )
            else:
                known = _literal_text(part)
            pieces.append(UNKNOWN_PIECE if known is None else known)
        return "".join(pieces)
    rewritten = _replace_text(node)
    if rewritten is not None:
        return rewritten
    formatted = _format_text(node)
    if formatted is not None:
        return formatted
    parts = _join_parts(node)
    if parts is not None:
        # "".join(("htt", "ps://ev", "il.exa", "mple/c")) is a URL spelled out in
        # full whose scheme never appears in any one piece, so nothing matched
        # RE_URL_SCHEME and nothing was recorded as a destination. Read like an
        # f-string: a piece that cannot be read is a hole, never a reason to
        # treat the pieces around it as the whole string.
        separator, elements = parts
        return separator.join(elements)
    added = _added_text(node)
    if added is not None:
        return added
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        # "https" * 1 + "://evil.example/c" carries its scheme in a repetition,
        # which was neither folded nor refused, so no literal in the file held
        # a URL. Either operand may be the count, and one this cannot read
        # leaves a hole, exactly as an unreadable f-string piece does.
        for text_side, count_side in (
            (node.left, node.right), (node.right, node.left),
        ):
            text = _literal_text_uncharged(text_side)
            if text is None:
                continue
            count = count_side.value if (
                isinstance(count_side, ast.Constant)
                and isinstance(count_side.value, int)
                and not isinstance(count_side.value, bool)
            ) else None
            if count is None:
                return UNKNOWN_PIECE
            if count < 0:
                return ""
            if len(text) * count > MAX_FOLDED_JOIN:
                return None             # the same output ceiling as the join
            return text * count
        return None
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod))):
        return None
    left = _literal_text(node.left)
    if left is None:
        return None
    if isinstance(node.op, ast.Add):
        right = _literal_text(node.right)
        if right is None:
            return None
        if len(left) + len(right) > MAX_FOLDED_JOIN:
            return None                 # the same output ceiling as the join
        return left + right
    if RE_UNSAFE_PERCENT.search(left):
        return _percent_holes(left)
    if isinstance(node.right, ast.Dict):
        # "%(scheme)s://%(host)s/c" % {"scheme": "https", "host": "evil.example"}
        # is a whole URL whose scheme is spelled by the template and whose host
        # is spelled by the mapping, and neither half is a destination on its
        # own. Only the tuple form was folded, so this one read as no URL at all.
        mapping = {}
        for key, value in zip(node.right.keys, node.right.values):
            name = _literal_text(key) if key is not None else None
            text = _literal_text(value)
            if name is None or text is None:
                return _percent_holes(left)
            mapping[name] = text
        if _percent_is_oversized(left, mapping.values()):
            return _percent_holes(left)
        try:
            return left % mapping
        except (TypeError, ValueError, KeyError):
            return _percent_holes(left)
    operands = node.right.elts if isinstance(node.right, ast.Tuple) else [node.right]
    values = [_literal_text(operand) for operand in operands]
    if any(value is None for value in values):
        return _percent_holes(left)
    if _percent_is_oversized(left, values):
        return _percent_holes(left)
    try:
        return left % tuple(values)
    except (TypeError, ValueError):
        return _percent_holes(left)


def _mapping_key(node):
    """The literal string key of a Subscript, or None for anything else.

    A dictionary lookup by name reads a value out. A slice, an index, or a key
    this cannot read takes the string apart, which is what BASE[:8] does, so
    only the named lookup is treated as a read rather than a reshape.
    """
    if not isinstance(node, ast.Subscript):
        return None
    key = _literal_text(node.slice)
    return key if key is not None else None


def _bound_names(target):
    """Every name an assignment target binds, unpacking included.

    `BASE, = ("https://huggingface.co",)` binds BASE through a Tuple, and
    reading only bare Name targets lost the carrier. An attribute target gives
    its attribute name, which is what carries() matches for cls.BASE_DOMAIN.
    """
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Attribute):
        return {target.attr}
    if isinstance(target, ast.Starred):
        return _bound_names(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        names = set()
        for element in target.elts:
            names |= _bound_names(element)
        return names
    return set()


def _reshapes_a_url(tree, assignments = None, aliases = None):
    """Whether a URL this file spells out is transformed before it is used.

    Reading every literal and asking whether each names a hub host assumes a
    literal reaches the request as written. It need not:

        BASE = "https://huggingface.co"
        requests.get(BASE.replace("huggingface.co", "evil.example"), ...)

    fetches evil.example, and every literal in that file is either the hub or a
    bare name with no scheme in front of it, so nothing was recorded as a
    destination but the hub. Slicing does the same with no method call at all.
    So a URL literal, or a name that carries one, may not be the receiver of a
    call or be subscripted here. Upstream is unaffected: in gguf-py/gguf/utility.py
    every .replace(), .strip() and .format() receiver is a parameter, and
    BASE_DOMAIN is only ever interpolated into an f-string.

    A name carries through STRING BUILDING only: url = f"{BASE}/x" carries, and
    url.replace(...) after it is refused. It deliberately does not carry through
    a call, because upstream writes response = requests.get(url) and then reads
    response.raise_for_status(), index_json["weight_map"], raw_data[:8]: tracking
    what came BACK from the hub made ordinary parsing of the download look like a
    reshaped URL and refused the very file this allowance exists for. A fixed
    point, because an assignment can precede the one that makes its value carry.

    A parameter is not tracked: following one means following a call, which is
    the interprocedural residual this allowance already documents.
    """
    carriers = set()
    aliases = aliases if aliases is not None else _call_aliases(tree, assignments)

    def called_name(node):
        return _called_name(node, aliases)

    def carries(node):
        while isinstance(node, ast.NamedExpr):
            node = node.value                   # (u := BASE).replace(...)
        if isinstance(node, ast.Name) and node.id in carriers:
            return True
        if isinstance(node, ast.Attribute) and node.attr in carriers:
            return True                         # cls.BASE_DOMAIN, self.BASE ...
        text = _literal_text(node)
        return bool(text) and bool(RE_URL_SCHEME.search(text))

    def built_from_a_carrier(node):
        if carries(node):
            return True
        if isinstance(node, ast.JoinedStr):
            return any(
                built_from_a_carrier(
                    part.value if isinstance(part, ast.FormattedValue) else part
                )
                for part in node.values
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
            operands = [node.left]
            operands += (
                node.right.elts if isinstance(node.right, ast.Tuple) else [node.right]
            )
            return any(built_from_a_carrier(operand) for operand in operands)
        if isinstance(node, ast.Dict):
            # URLS = {"hub": "https://huggingface.co"} then URLS["hub"].
            return any(
                value is not None and built_from_a_carrier(value)
                for value in node.values
            )
        if isinstance(node, ast.Subscript) and _mapping_key(node) is not None:
            # A lookup by name reads the value out; it does not reshape it.
            return built_from_a_carrier(node.value)
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            # BASE, = ("https://huggingface.co",): the value is a sequence, and
            # which element lands on which name is not worth tracking, so every
            # name the target binds carries when any element does.
            return any(built_from_a_carrier(element) for element in node.elts)
        nodes = _join_nodes(node)
        if nodes is not None:
            # "".join([BASE, "/x"]) carries: without this the join laundered the
            # carrier and .replace() on the result was not a reshape of anything.
            separator_node, element_nodes = nodes
            return any(
                built_from_a_carrier(part)
                for part in [separator_node, *element_nodes]
            )
        return False

    # Bounded, because each pass walks the whole tree and a chain of assignments
    # that each carry the previous one needs a pass apiece: 2000 of them took 25
    # seconds, on the path that decides whether an export may run. Real files
    # settle in one or two. A file that has not settled by then is refused
    # rather than spun on, which is the same answer this gives to everything
    # else it cannot read in reasonable time.
    # Collected once. Walking the whole tree per pass made a chain of 10000
    # assignments take about two seconds, on a file the scan accepts at up to
    # 8 MiB, and the chain is what forces the passes in the first place.
    assignments = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value:
            targets, value = [node.target], node.value
        else:
            continue
        names = set()
        for target in targets:
            names |= _bound_names(target)
        if names:
            assignments.append((names, value))

    for _ in range(MAX_CARRIER_PASSES):
        found = set()
        for names, value in assignments:
            if names <= carriers:
                continue                # already carrying, nothing to learn
            if built_from_a_carrier(value):
                found |= names
        if found <= carriers:
            break
        carriers |= found
    else:
        return True
    def extends_the_authority(node):
        """Whether this expression appends to a carrier past its authority.

        HUB + "@evil.example/collect" is fetched from evil.example, everything
        before the @ being user information, while the walk sees only the hub
        literal the name still holds. Appending a PATH is what upstream does,
        so text beginning with a URL delimiter is fine and anything else, text
        this cannot read included, is not.
        """
        whole = _literal_text(node)
        if whole is not None and UNKNOWN_PIECE not in whole:
            # It folds COMPLETELY, so the walk reads the URL it really builds:
            # "https://hugging" + "face.co/api/models" is the hub, and judging
            # that by its pieces refuses a file that names nothing else. A fold
            # with a hole in it is not read by anyone, which is the case this
            # check exists for.
            return False
        def flatten(node):
            # HUB + "/api/" + name parses as (HUB + "/api/") + name, so reading
            # two parts put the path inside the first one and the check never
            # saw the delimiter that had already ended the authority. It then
            # refused ordinary upstream-shaped code, which is the false positive
            # this rule has to stay clear of. Iteratively, because a long chain
            # of appends is exactly what nests deepest.
            parts, stack = [], [node]
            while stack:
                item = stack.pop()
                if isinstance(item, ast.BinOp) and isinstance(item.op, ast.Add):
                    stack.extend([item.right, item.left])
                else:
                    parts.append(item)
            return parts

        if isinstance(node, ast.JoinedStr):
            parts = node.values
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            parts = flatten(node)
        else:
            return False
        appended = False
        for part in parts:
            if isinstance(part, ast.FormattedValue):
                part = part.value
            if not appended:
                appended = built_from_a_carrier(part)
                if appended:
                    carried = _literal_text(part)
                    if carried and RE_AUTHORITY_CLOSED.search(carried):
                        break           # this piece already ended the authority
                continue
            text = _literal_text(part)
            if text == "":
                continue
            if text and RE_AUTHORITY_END.match(text):
                break                   # a path, a query or a fragment: the
                                        # authority ended here, and what comes
                                        # after it cannot change the host
            return True
        return False

    for node in ast.walk(tree):
        if extends_the_authority(node):
            return True
        if (
            isinstance(node, ast.Subscript)
            and _mapping_key(node) is None
            and _literal_text(node.value) is not None
        ):
            # A literal that is indexed or sliced:
            # "c/tcelloc/elpmaxe.live//:sptth"[::-1] is a whole URL written
            # backwards, and no rule here matched a scheme in it. Subscripting
            # a name that CARRIES a URL was already refused; a literal that
            # becomes one only when it is sliced is the same reshape with the
            # text written out. Upstream slices what came back from the hub,
            # raw_data[:8], which is not a literal.
            return True
        if _rewrites_a_constant(node):
            return True
        if isinstance(node, ast.Call) and called_name(node) in ("reduce", "accumulate"):
            # functools.reduce(operator.add, ["https", "://evil.example/c"])
            # builds a string out of pieces this walk reads one at a time, and
            # the fold cannot follow a callable applied pairwise. No file in
            # llama.cpp imports functools at all.
            return True
        if isinstance(node, ast.Call) and called_name(node) in URL_ASSEMBLERS:
            # No carrier and no URL literal: the destination is spelled field by
            # field and assembled by the object itself.
            return True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in URL_REWRITE_METHODS
            and any(
                built_from_a_carrier(inner)
                for inner in ast.walk(node.func.value)
            )
        ):
            # httpx.URL(HUB).copy_with(host = "evil.example") sends the token to
            # evil.example while every literal in the file is still the hub.
            return True
        if isinstance(node, ast.Call) and any(
            built_from_a_carrier(argument)
            for argument in [*node.args, *(k.value for k in node.keywords)]
        ):
            # A function can rewrite a URL as well as a method can:
            # urljoin(HUB, "//evil.example/collect") resolves to evil.example.
            # Only the builders, since upstream passes its URL to urlparse and
            # to its own helpers, and those read it rather than rebuild it.
            if called_name(node) in URL_BUILDERS:
                return True
        if isinstance(node, ast.AugAssign) and (
            carries(node.target) or built_from_a_carrier(node.value)
        ):
            return True                 # url += "@evil.example/collect"
        # built_from_a_carrier, not carries: the receiver may be the string
        # building itself, as in "".join([BASE, "/x"]).replace(...) or
        # f"{BASE}"[:8], with no name in between to have been tainted.
        if (
            isinstance(node, ast.Subscript)
            and _mapping_key(node) is None
            and built_from_a_carrier(node.value)
        ):
            return True                 # BASE[:8], and any index this cannot read
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and built_from_a_carrier(node.func.value)
        ):
            return True
    return False


def _docstrings(tree):
    """Docstring nodes, which document a destination rather than name one.

    Comments never reach here, since the walk reads the AST, so upstream's
    `# Reference: https://github.com/...` costs nothing. A docstring saying the
    same thing IS a Constant, and counting it as a destination refuses a clean
    hub-only converter over a link in its own documentation, which is the false
    positive this whole narrowing exists to remove.

    Unless the file can REACH a docstring, in which case it is a value like any
    other and the argument above stops holding. Enumerating the spellings was
    losing: __doc__, fn.__doc__, getattr(fn, "__doc__") and vars(fn)["__doc__"]
    all read the same string. So anything that names __doc__, vars or __dict__
    at all keeps docstrings in the scan. Upstream names none of them.
    """
    if any(
        (isinstance(node, ast.Name) and node.id in DOCSTRING_READERS)
        or (isinstance(node, ast.Attribute) and node.attr in DOCSTRING_READERS)
        # `from inspect import getdoc as g` names the reader in an alias, which
        # is neither, and g(fetch) then read a docstring this had skipped.
        or (
            isinstance(node, ast.alias)
            and node.name.split(".")[0] in DOCSTRING_READERS
        )
        or (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").split(".")[0] in DOCSTRING_READERS
        )
        # Folded, like the write and environment names: getattr(fn, "__" +
        # "doc__") reads the same attribute.
        or _literal_text(node) in DOCSTRING_READERS
        for node in ast.walk(tree)
    ):
        # Read from the AST: the word in a comment is not a read, and taking it
        # for one put the false positive straight back.
        return frozenset()
    nodes = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            nodes.add(id(first.value))
    return frozenset(nodes)


# The match patterns capture a name, and they do not exist before Python 3.10,
# which this package still supports. Named at import rather than reached for
# per node: ast.MatchAs raised AttributeError there, the allowance caught it and
# refused, and the honest converter produced a CRITICAL finding on 3.9 alone.
# isinstance against an empty tuple is simply False.
MATCH_CAPTURES = tuple(
    pattern for pattern in
    (getattr(ast, name, None) for name in ("MatchAs", "MatchStar"))
    if pattern is not None
)
MATCH_MAPPING = tuple(
    pattern for pattern in (getattr(ast, "MatchMapping", None),)
    if pattern is not None
)


def _single_assignments(tree):
    """`{name: value}` for every name this module binds exactly once.

    Bindings of every kind are counted, not just assignments: a parameter, a
    loop variable, an import or a `for` target that reuses the name means the
    name is not one value, and anything but exactly one Assign is left alone.
    """
    counts, values = {}, {}

    def bind(name, value = None):
        counts[name] = counts.get(name, 0) + 1
        if value is not None:
            values[name] = value

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                bind(node.targets[0].id, node.value)
            else:
                for target in node.targets:
                    for name in _bound_names(target):
                        bind(name)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value is not None:
                bind(node.target.id, node.value)
            else:
                for name in _bound_names(node.target):
                    bind(name)
        elif isinstance(node, (ast.AugAssign, ast.NamedExpr, ast.For,
                               ast.AsyncFor, ast.comprehension)):
            for name in _bound_names(node.target):
                bind(name)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            for name in _bound_names(node.optional_vars):
                bind(name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            bind(node.name)
        elif isinstance(node, ast.arg):
            bind(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bind((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bind(node.name)
        elif isinstance(node, MATCH_CAPTURES) and node.name:
            bind(node.name)             # case (scheme, HOST) rebinds HOST
        elif isinstance(node, MATCH_MAPPING) and node.rest:
            bind(node.rest)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                bind(name)
                bind(name)              # never one value, whatever else it is
    return {
        name: value for name, value in values.items() if counts.get(name) == 1
    }


# The string methods this file folds or refuses, which are also the ones worth
# spelling in unbound form to miss those rules.
DESCRIPTOR_METHODS = frozenset((
    "format", "format_map", "join", "replace", "translate", "__add__",
))


def _bind_descriptor_calls(tree, assignments = None, aliases = None):
    """The tree with `str.format(t, x)` rewritten as `t.format(x)`.

    Called unbound, the receiver is the type rather than the template, so every
    rule that reads a receiver saw `str` and the template went past as an
    ordinary argument: str.format("{}://{}", "https", "evil.example/c") named
    no destination at all. Rewriting it once here is what keeps the folds and
    the refusals written one way.
    """
    # r = str.replace then r(HUB, "huggingface.co", "evil.example") is the same
    # call with the method behind a name, which no rule that reads a call site
    # could see.
    if assignments is None:
        assignments = _single_assignments(tree)
    if aliases is None:
        aliases = _call_aliases(tree, assignments)
    rebound = {}
    for name, value in assignments.items():
        if (
            isinstance(value, ast.Attribute)
            and value.attr in DESCRIPTOR_METHODS
            and isinstance(value.value, ast.Name)
            and value.value.id in ("str", "bytes", "bytearray")
        ):
            rebound[name] = value.attr

    # Rewritten in place, in one walk. A NodeTransformer over the whole tree
    # cost 3.7 seconds of the 11 this allowance spent on a 6.4 MB file, and
    # rebinding a call's function changes no structure for a parent to rewire.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if isinstance(node.func, ast.Name) and node.func.id in rebound:
            attribute = rebound[node.func.id]
        elif (
            _called_name(node, aliases) in ADD_FUNCTIONS
            and len(node.args) == 2
        ):
            # from operator import add, then add("https", "://evil.example/c").
            # Written as the method it is, so one fold reads every spelling.
            attribute = "__add__"
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in DESCRIPTOR_METHODS
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in ("str", "bytes", "bytearray")
        ):
            attribute = node.func.attr
        else:
            continue
        node.func = ast.copy_location(
            ast.Attribute(value = node.args[0], attr = attribute, ctx = ast.Load()),
            node.func,
        )
        node.args = node.args[1:]
    return tree


def _inline_constants(tree, assignments = None):
    """The tree with every single-assignment constant name read as its text.

    scheme = "https"; separator = "://"; host = "evil.example" assembles a URL
    that no literal in the file spells, and nothing here reads a name, so the
    only destination recorded was an unused hub literal beside it. Substituting
    the text a name can only ever hold puts the assembled URL back in front of
    every rule that reads one, the destination walk included.

    Only names bound exactly once, and only to text that folds with no hole in
    it, under the same ceiling the folds use: this replaces a name with a value
    it demonstrably has, never with a guess.
    """
    uses = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            uses[node.id] = uses.get(node.id, 0) + 1

    if assignments is None:
        assignments = _single_assignments(tree)
    texts, budget = {}, MAX_FOLDED_JOIN
    for name, value in assignments.items():
        text = _literal_text(value)
        if text is None or UNKNOWN_PIECE in text:
            continue
        # Per USE, not per name: one 50 KB constant loaded a hundred times is
        # five megabytes of text out of a 49 KB file, and folding the growing
        # prefixes of that took six seconds on a file the scan accepts at up to
        # 8 MiB. The budget is on what the substitution materialises.
        budget -= len(text) * uses.get(name, 0)
        if budget < 0:
            # Not a break: leaving the rest of the file unsubstituted analyses a
            # tree this could not read, and a 600 KB constant loaded twice ahead
            # of scheme, separator and host bought exactly that.
            raise _FoldBudgetExceeded("substituted more text than one file may")
        texts[name] = text
    if not texts:
        return tree

    class Inliner(ast.NodeTransformer):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id in texts:
                return ast.copy_location(ast.Constant(texts[node.id]), node)
            return node

    return Inliner().visit(tree)


def _literal_texts(tree, skip = frozenset()):
    """Every whole literal expression in `tree`, folded ones in place of parts.

    A folded expression is not descended into: reading "https://hugging" +
    "face.co/api/models" as the hub AND as a host called `hugging` refuses the
    allowance over a file that only ever names the hub, which is the false
    positive this whole narrowing exists to remove.
    """
    stack = [tree]
    while stack:
        node = stack.pop()
        if id(node) in skip:
            continue
        text = _literal_text(node)
        if text is not None:
            yield text
            continue
        stack.extend(ast.iter_child_nodes(node))


def _authority_host(authority):
    """The hostname an authority names, "" when it names none, None when unclear.

    Empty is the bare "https://" that upstream's metadata.py hands to startswith:
    it names no destination, so it says nothing either way. Anything else has to
    look like a plain host with an optional port. If it does not, this cannot say
    where the request goes, and not knowing is not a reason to allow it.
    """
    if not authority:
        return ""
    if not RE_HOSTNAME.match(authority):
        return None
    return authority.split(":")[0].lower()


def _talks_only_to_the_model_hub(text):
    """Whether every URL this file names in code is a model-hub host.

    Read from string literals via the AST, not from the raw text, so the github
    links upstream carries in comments do not count as destinations. Requires at
    least one hub host: a file that names no destination at all has built it
    some other way, and that is not evidence of anything except that this cannot
    see it. An authority that does not look like a plain hostname refuses the
    allowance outright, whatever it contains, and so does any use of a network
    API that takes a bare host rather than a URL.

    This is a deliberate narrowing of a CRITICAL rule, and the evasion is not
    hypothetical: taking upstream's utility.py and changing one call to
    `requests.get(os.environ["X"] + "/collect", ...)` leaves the hub literal in
    place, so this suppresses the finding. Adding a literal exfil URL is caught,
    as is a payload that names no host at all, since suppression needs a hub host
    to be named.

    Kept anyway, because the alternative measured worse: with the rule as it
    was, UNSLOTH_CONVERTER_SCAN_STRICT refused every clean checkout of llama.cpp
    master over upstream's own file, and a control that rejects what it is meant
    to protect is one people switch off. The scan says of itself that it raises
    the cost of an opportunistic payload and is not a boundary; this is inside
    that claim, not a departure from it.
    """
    if _matches(RE_UNVOUCHABLE_NETWORK, text):
        # A destination this allowance never looks at, so it cannot vouch for it.
        return False
    if _matches(RE_UNVOUCHABLE_STRING_BUILD, text):
        # A destination decoded out of a constant is one this cannot read.
        return False
    if _matches(RE_WHOLE_ENV, text):
        # Reading the whole environment is not the token-authenticated download
        # this allowance is for.
        return False
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False                # cannot tell, so do not suppress anything
    try:
        return _talks_only_to_the_model_hub_tree(tree, text)
    except Exception:
        # Nothing in here may raise, for ANY reason. 600 operands in one
        # addition exhausts the recursion limit and "{0[x]}".format("a") raises
        # TypeError, both from expressions that need never run;
        # scan_converter_source does not catch either, and
        # warn_on_suspicious_converter catches everything and CONTINUES, so a
        # payload could append one such expression to itself and have the whole
        # scan report nothing, in strict mode included. Refusing is the answer
        # to every failure here, so the class of failure does not matter.
        return False


def _talks_only_to_the_model_hub_tree(tree, text):
    """The parsed half of the allowance. See the caller."""
    previous = getattr(_fold_state, "budget", None)
    _fold_state.budget = MAX_TOTAL_FOLD
    try:
        return _talks_only_to_the_model_hub_parsed(tree, text)
    finally:
        _fold_state.budget = previous


def _talks_only_to_the_model_hub_parsed(tree, text):
    """The allowance proper, under the fold budget its caller opened."""
    # Read once and handed on. Every binding form is counted by walking the
    # whole tree, and doing that per transformation was over half the time this
    # allowance spent on a 6.4 MB file: 9.3 seconds of 16.
    assignments = _single_assignments(tree)
    aliases = _call_aliases(tree, assignments)
    tree = _bind_descriptor_calls(tree, assignments, aliases)
    tree = _inline_constants(tree, assignments)
    if _writes_anything(tree):
        # Sending TO the hub is not downloading from it. The hub is writable and
        # multi-tenant: a token with write scope can create a public repository
        # there and make it a channel anyone can read back, so "the destination
        # is the hub" is not on its own a reason to say nothing.
        return False
    if _imports_an_unvouchable_api(tree, aliases):
        # A destination this allowance never looks at, arriving by another door.
        return False
    if _imports_a_string_builder(tree, aliases):
        # A decoder whose only readable name is the import line itself.
        return False
    if _templates_are_oversized(tree):
        # Text no rule here will read, which is not a reason to say nothing.
        return False
    if _joins_something_unreadable(tree):
        # Pieces this cannot enumerate, joined into one string by a literal.
        return False
    if _builds_text_from_numbers(tree):
        # A destination spelled in character codes rather than in characters.
        return False
    if _reaches_through_getattr(tree):
        # A lookup this cannot read is a write or a read it cannot see.
        return False
    if _aliases_the_environment(tree):
        # A read this cannot attribute is the same as one it cannot see.
        return False
    if _reshapes_a_url(tree, assignments, aliases):
        # A literal that is rewritten before it is sent names the host it was,
        # not the host it becomes.
        return False
    names, dynamic = _env_reads(tree)
    if dynamic:
        # A read whose variable is chosen at runtime cannot be attributed, and
        # an unattributable one used to pass by leaving the set empty.
        return False
    if not names or not names <= HUB_TOKEN_ENV_NAMES:
        # EVERY name, not the ones that look secret. Filtering on keywords let
        # GITHUB_PAT through, because "PAT" is not one of them, and a file
        # reading HF_TOKEN and GITHUB_PAT then sending the second one to the hub
        # produced no finding at all. Nothing here can tell a credential from a
        # setting by its name, so the allowance covers reading the hub's own
        # token and nothing else. The only upstream file this applies to reads
        # exactly HF_TOKEN.
        return False
    hosts = set()
    # bytes as well as str: requests decodes b"https://evil.example/collect" and
    # accepts it, so skipping bytes constants let a destination hide in one while
    # a str hub literal stayed in the file.
    # Iterated, not materialised, and budgeted in aggregate: MAX_FOLDED_JOIN
    # bounds one expansion, and an 8 MiB file holds thousands of them.
    def folded_literals():
        budget = MAX_FOLDED_JOIN
        for literal in _literal_texts(tree, skip = _docstrings(tree)):
            budget -= len(literal)
            if budget < 0:
                raise MemoryError("folded literal budget")
            yield literal

    try:
        for literal in folded_literals():
            if UNKNOWN_PIECE + "://" in literal:
                # The scheme itself came out of a piece this cannot read, so
                # the destination is not merely unnamed, it is hidden: nothing
                # in the file spells a URL for the walk below to look at. Every
                # upstream "://" is preceded by a readable http, https or ssh.
                return False
            for match in RE_URL_SCHEME.finditer(literal):
                rest = literal[match.end():]
                end = RE_AUTHORITY_END.search(rest)
                if end is None and rest:
                    # No delimiter, so the authority runs to the end of the literal:
                    # "https://huggingface.co", which is upstream's BASE_DOMAIN.
                    host = _authority_host(rest)
                elif end is None:
                    # Nothing at all after the scheme. Reading that as "names no
                    # destination" is what every splitting trick was built on:
                    # "https://" + host, "".join(("https://", host)),
                    # "{}evil.example".format("https://"), each leaves a bare scheme
                    # in one literal and the host somewhere this does not connect to
                    # it. A scheme with no authority means the destination is
                    # assembled, and an assembled one is not a destination this can
                    # vouch for. Upstream's bare "https://" literals are in
                    # metadata.py, which never reaches this allowance: only
                    # gguf-py/gguf/utility.py does, measured on master, and it has
                    # none.
                    host = None
                elif end.start() == 0:
                    # A path with no host, as in https:///collect. requests will not
                    # send that anywhere useful, but this cannot say where it goes,
                    # and not knowing is not a reason to allow it.
                    host = None
                else:
                    host = _authority_host(rest[: end.start()])
                if host is None:
                    return False        # cannot tell, so do not suppress anything
                if host:
                    hosts.add(host)
    except (UnicodeError, AttributeError, MemoryError, RecursionError):
        return False                # cannot tell, so do not suppress anything
    return bool(hosts) and hosts <= MODEL_HUB_HOSTS


# Large base64 blob (>200 chars of contiguous base64 alphabet)
RE_LARGE_BLOB = re.compile(r"[A-Za-z0-9+/=]{200,}")

# Credential path access (requires file-access context, not just string mentions)
RE_CRED_ACCESS = re.compile(
    r"(?:open|Path|read_text|read_bytes)\s*\([^)]*?"
    r"(?:\.ssh[/\\]|\.aws[/\\]|\.kube[/\\]|\.gnupg[/\\]|\.docker[/\\]"
    r"|\.azure[/\\]|\.gcp[/\\]"
    r"|credentials\.json|\.git-credentials|\.npmrc|\.pypirc|wallet\.dat"
    r"|/etc/shadow|/etc/passwd"
    r"|id_rsa|id_ed25519|id_ecdsa"
    r"|kubeconfig|service-account-token)"
    r"|os\.path\.(?:join|expanduser)\([^)]*?"
    r"(?:\.ssh|\.aws|\.kube|\.gnupg|\.docker|\.azure|\.gcp|credentials)"
    r"|(?:open|Path)\(\s*['\"]\.env['\"]\s*[,)]",
    re.DOTALL,
)

# Chained / advanced obfuscation (marshal, compile, zlib, nested decode)
RE_OBFUSCATION = re.compile(
    r"\bmarshal\s*\.\s*(loads|load)\b"
    r"|\bcompile\s*\([^)]*['\"]exec['\"]\s*\)"
    r"|\bzlib\s*\.\s*decompress\b"
    r"|\blzma\s*\.\s*decompress\b"
    r"|\bbz2\s*\.\s*decompress\b"
    r"|\bbytearray\s*\(\s*\[.*?\]\s*\)"  # bytearray([104,101,...])
    r"|\bchr\s*\(\s*\d+\s*\).*chr\s*\(\s*\d+\s*\)"  # chr() obfuscation chains
    r"|\b__import__\s*\("  # dynamic import
    r"|\bgetattr\s*\(\s*__builtins__"  # getattr(__builtins__, ...)
    r"|\brotate\s*=.*\blambda\b.*\bchr\b"  # rotation ciphers
    r"|\b(?:b64decode|decodebytes)\s*\(.*(?:b64decode|decodebytes)\s*\(",  # double base64
    re.DOTALL,
)

# Embedded cryptographic keys (PEM-encoded)
RE_EMBEDDED_KEYS = re.compile(
    r"-----BEGIN\s+(?:RSA\s+)?(?:PUBLIC|PRIVATE|ENCRYPTED|EC|DSA|OPENSSH)\s+KEY-----"
    r"|\bRSA\s+PUBLIC\s+KEY\b.*[A-Za-z0-9+/=]{64,}"
    r"|\bMII[A-Za-z0-9+/]{20,}",  # DER-encoded key prefix (base64)
    re.DOTALL,
)

# Cloud metadata / IMDS endpoints
RE_CLOUD_METADATA = re.compile(
    r"169\.254\.169\.254"  # AWS/Azure/GCP IMDS
    r"|metadata\.google\.internal"  # GCP metadata
    r"|169\.254\.170\.2"  # AWS ECS task metadata
    r"|100\.100\.100\.200"  # Alibaba Cloud metadata
    r"|/latest/meta-data"  # AWS IMDS path
    r"|/metadata/instance"  # GCP metadata path
    r"|/metadata/identity"  # Azure managed identity
    r"|\bIMDSv[12]\b",
)

# Persistence mechanisms (systemd, cron, launchd, registry, startup dirs)
RE_PERSISTENCE = re.compile(
    r"/etc/systemd/"
    r"|systemctl\s+(enable|start|daemon-reload)"
    r"|\.service\b.*\[Service\]"  # systemd unit content
    r"|/etc/cron"
    r"|crontab\s"
    r"|/etc/init\.d/"
    r"|/Library/LaunchDaemons"
    r"|/Library/LaunchAgents"
    r"|~/\.config/autostart"
    r"|~/.local/share/systemd"
    r"|~/\.config/systemd/user/"  # user-level systemd
    r"|HKEY_LOCAL_MACHINE.*\\\\Run"  # Windows registry autorun
    r"|HKEY_CURRENT_USER.*\\\\Run"
    r"|\\\\Start Menu\\\\Programs\\\\Startup"
    r"|schtasks\s",  # Windows scheduled tasks
    re.IGNORECASE,
)

# Container / orchestration abuse
RE_CONTAINER_ABUSE = re.compile(
    r"/var/run/docker\.sock"
    r"|\bdocker\s+(run|exec|cp|build)\b"
    r"|\bkubectl\s+(apply|create|exec|run|cp)\b"
    r"|\bkubernetes\.client\b"
    r"|\bfrom_incluster_config\b"
    r"|\blist_namespaced_secret\b"
    r"|\bcreate_namespaced_pod\b"
    r"|\bcreate_namespaced_daemon_set\b"
    r"|\bcreate_namespaced_secret\b"
    r"|\bkube-system\b"
    r"|\bhostPID\s*:\s*true"
    r"|\bprivileged\s*:\s*true"
    r"|\bhostNetwork\s*:\s*true"
    r"|\bhostPath\b.*\bpath\s*:\s*/",  # k8s hostPath mounts
    re.IGNORECASE,
)

# Environment variable harvesting (bulk access or known secret vars)
RE_ENV_HARVEST = re.compile(
    r"\bos\.environ\s*\.\s*copy\s*\("  # full env copy
    r"|\bdict\s*\(\s*os\.environ\s*\)"
    r"|\bjson\.dumps\s*\(\s*(?:dict\s*\(\s*)?os\.environ"
    r"|\bfor\s+\w+\s*,\s*\w+\s+in\s+os\.environ\.items\(\)"  # iterating all env vars
    r"|\bos\.environ\b.*(?:SECRET|TOKEN|KEY|PASSWORD|CREDENTIAL|API_KEY|PRIVATE)"
    r"|\b(?:SECRET|TOKEN|PASSWORD|API_KEY|PRIVATE_KEY)\b.*os\.environ",
    re.IGNORECASE,
)

# Archive staging / exfiltration prep (create archive + network send)
RE_ARCHIVE_STAGING = re.compile(
    r"\btarfile\s*\.\s*open\s*\("
    r"|\bzipfile\s*\.\s*ZipFile\s*\([^)]*['\"]w['\"]\s*\)"
    r"|\bshutil\s*\.\s*make_archive\b"
    r"|\b\.add\s*\([^)]*(?:\.ssh|\.aws|\.env|\.kube|credentials|\.gnupg|\.docker)"
    r"|\b\.write\s*\([^)]*(?:\.ssh|\.aws|\.env|\.kube|credentials|\.gnupg|\.docker)",
    re.DOTALL,
)

# Anti-analysis / sandbox evasion / debugger detection
RE_ANTI_ANALYSIS = re.compile(
    r"\bptrace\b"
    r"|\bsys\s*\.\s*gettrace\s*\("
    r"|\bsys\s*\.\s*settrace\b"
    r"|\bTracerPid\b"
    r"|\b/proc/self/status\b"
    r"|\bIsDebuggerPresent\b"
    r"|\bvirtualbox\b.*\bhardware\b"
    r"|\bvmware\b.*\bdetect\b"
    r"|\btime\.sleep\s*\(\s*(?:[3-9]\d{2,}|[1-9]\d{3,})\s*\)"  # long sleep (anti-sandbox)
    r"|\bplatform\.\s*system\b.*\bif\b.*\b(?:Linux|Windows|Darwin)\b",
    re.IGNORECASE | re.DOTALL,
)

# DNS exfiltration / tunneling
RE_DNS_EXFIL = re.compile(
    r"\bdns\.resolver\b"
    r"|\bsocket\.getaddrinfo\s*\([^)]*\+[^)]*\)"  # dynamic hostname construction
    r"|\bdnspython\b"
    r"|\bTXT\b.*\bresolver\b"
    r"|\bresolver\b.*\bTXT\b"
    r"|\bnslookup\b"
    r"|\bdig\s+",
)

# File system enumeration / bulk file theft
RE_FS_ENUM = re.compile(
    r"\bos\.walk\s*\(\s*['\"](?:/|~|/home|/root|/Users|C:\\\\)"
    r"|\bglob\s*\.\s*glob\s*\([^)]*(?:\*\*|\*\.pem|\*\.key|\*\.cer|\*\.pfx|\*\.p12)"
    r"|\bos\.listdir\s*\(\s*['\"](?:/home|/root|/Users|/etc)"
    r"|\bPath\s*\(\s*['\"]~['\"]\s*\)\s*\.\s*glob\b"
    r"|\bhistory\b.*\bread\b"  # reading shell history
    r"|\b\.bash_history\b"
    r"|\b\.zsh_history\b"
    r"|/etc/shadow"
    r"|/etc/passwd",
    re.DOTALL,
)

# Reverse shell / bind shell patterns
RE_REVERSE_SHELL = re.compile(
    r"\bsocket\b.*\bconnect\b.*\bsubprocess\b"
    r"|\bsocket\b.*\bconnect\b.*\b(?:sh|bash|cmd)\b"
    r"|\b/bin/(?:sh|bash)\b.*\bsocket\b"
    r"|\bpty\s*\.\s*spawn\b"
    r"|\bos\s*\.\s*dup2\s*\("
    r"|\bwebbrowser\s*\.\s*open\b.*\bdata:\b",  # data: URI abuse
    re.DOTALL,
)

# Process injection / code loading from remote
RE_REMOTE_CODE = re.compile(
    r"\bexec\s*\(\s*(?:urllib|requests|httpx|urlopen)"  # exec(requests.get(...))
    r"|\bexec\s*\([^)]*\.(?:text|content|read)\s*\("
    r"|\beval\s*\([^)]*\.(?:text|content|read)\s*\("
    r"|\bimportlib\s*\.\s*import_module\s*\([^)]*\+"  # dynamic import with concatenation
    r"|\b__import__\s*\([^)]*\+",  # __import__ with concatenation
    re.DOTALL,
)

# Crypto wallet / cryptocurrency theft
RE_CRYPTO_THEFT = re.compile(
    r"\bwallet\.dat\b"
    r"|\b\.bitcoin[/\\]"
    r"|\b\.ethereum[/\\]"
    r"|\b\.solana[/\\]"
    r"|\b\.monero[/\\]"
    r"|\b\.litecoin[/\\]"
    r"|\b\.config/solana[/\\]"
    r"|\bkeystore[/\\]UTC--"
    r"|\bseed\s*phrase\b"
    r"|\bmnemonic\b.*\b(?:word|phrase|recover|restore)\b"
    r"|\b(?:xprv|xpub|bc1|0x[a-fA-F0-9]{40})\b",
    re.IGNORECASE,
)

# openssl CLI invocations via subprocess (encrypted exfiltration)
RE_OPENSSL_CLI = re.compile(
    r"\bopenssl\s+(enc|rand|rsautl|pkeyutl|genrsa|dgst|s_client)\b"
)

# Write to /tmp then execute (staged dropper)
RE_TEMP_EXEC = re.compile(
    r"/tmp/\S+.*(?:subprocess|os\.system|os\.popen|Popen|chmod.*\+x)",
    re.DOTALL,
)

# C2 polling / beaconing loop
RE_C2_POLLING = re.compile(
    r"while\s+True.*(?:time\.sleep|sleep)\s*\(.*(?:urlopen|requests\.|httpx\.)",
    re.DOTALL,
)

# Mini Shai-Hulud May-12 2026 wave indicators. The dropper artifact name
# `transformers.pyz` is high-confidence (no legit PyPI package ships a `.pyz`
# named after `transformers`); the host + slogans are CRITICAL.
RE_MAY12_IOC = re.compile(
    r"(git-tanstack\.com|/tmp/transformers\.pyz|transformers\.pyz"
    r"|With Love TeamPCP|We've been online over 2 hours)",
    re.IGNORECASE,
)

# The exact regex llama_cpp.py uses to scrape argparse defaults out of the
# downloaded bytes before eval()ing them. Defined here, and imported there, so
# the scanner and the eval can never look at different tokens.
RE_ARGPARSE_DEFAULT = re.compile(
    rb"parser\.add_argument\([\s]*[\"\']([^\"\']{1,})[\'\"][^\)]*(?:action=|default=)[\s]*([^,\s\)]+)"
)

# A default token that is a literal, or a plain (possibly dotted) name. Anything
# else reaching eval() is a call, an operator, a subscript or a comprehension.
RE_PLAIN_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")

# Vendored pattern registry. The pin test walks this mapping.
VENDORED_PATTERNS = {
    "RE_SUBPROCESS": RE_SUBPROCESS,
    "RE_BASE64": RE_BASE64,
    "RE_EXEC_EVAL": RE_EXEC_EVAL,
    "RE_NETWORK": RE_NETWORK,
    "RE_LARGE_BLOB": RE_LARGE_BLOB,
    "RE_CRED_ACCESS": RE_CRED_ACCESS,
    "RE_OBFUSCATION": RE_OBFUSCATION,
    "RE_EMBEDDED_KEYS": RE_EMBEDDED_KEYS,
    "RE_CLOUD_METADATA": RE_CLOUD_METADATA,
    "RE_PERSISTENCE": RE_PERSISTENCE,
    "RE_CONTAINER_ABUSE": RE_CONTAINER_ABUSE,
    "RE_ENV_HARVEST": RE_ENV_HARVEST,
    "RE_ARCHIVE_STAGING": RE_ARCHIVE_STAGING,
    "RE_ANTI_ANALYSIS": RE_ANTI_ANALYSIS,
    "RE_DNS_EXFIL": RE_DNS_EXFIL,
    "RE_FS_ENUM": RE_FS_ENUM,
    "RE_REVERSE_SHELL": RE_REVERSE_SHELL,
    "RE_REMOTE_CODE": RE_REMOTE_CODE,
    "RE_CRYPTO_THEFT": RE_CRYPTO_THEFT,
    "RE_OPENSSL_CLI": RE_OPENSSL_CLI,
    "RE_TEMP_EXEC": RE_TEMP_EXEC,
    "RE_C2_POLLING": RE_C2_POLLING,
    "RE_MAY12_IOC": RE_MAY12_IOC,
}

# Canonical patterns deliberately left out, with the reason. The pin test asserts
# this set plus VENDORED_PATTERNS accounts for every RE_* in the canonical
# scanner, so a new pattern there forces a decision here instead of being missed.
PATTERNS_NOT_VENDORED = {
    "RE_PTH_IMPORT": "belongs to check_pth_file; a converter is not a .pth file",
    "RE_DEV_TOOL_HIJACK": "check_py_file does not consume it",
    "RE_TOKEN_REGEX": "check_py_file does not consume it",
    "RE_JS_OBFUSCATION": "belongs to check_js_file",
    "RE_WEB3_HIJACK": "belongs to check_js_file",
    "RE_WORKFLOW_INJECT": "belongs to check_js_file / check_shell_file / check_workflow_file",
    "RE_SHELL_DROPPER": "belongs to check_shell_file",
}


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------
@dataclass(frozen = True)
class ConverterScanFinding:
    severity : str
    check    : str
    evidence : str = ""


class ConverterScanError(RuntimeError):
    """Raised instead of running the converter when UNSLOTH_CONVERTER_SCAN_STRICT=1."""


# ---------------------------------------------------------------------------
# Linear-time evaluation of the whole-file patterns
# ---------------------------------------------------------------------------
# Several canonical patterns are shaped `A.*B.*C` under re.DOTALL. Run against a
# file that holds many A and B but no C, the engine tries every (A, B) pair
# before it can report "no match": measured at 2.0s on 5 KB and 17.3s on 11 KB,
# growing with the cube of the input. The bytes being scanned are the ones we do
# not trust, so an attacker who cannot beat the rules could still hang every GGUF
# export with a file full of the word "socket". A try/except cannot catch a hang.
#
# So the boolean is computed without backtracking across the `.*` joins. For
# existence, `A.*B` under DOTALL holds exactly when some A is followed by some B,
# which is what searching for B from the end of the earliest A answers, in linear
# time. Top-level `|` is split the same way, since "either alternative matches"
# is the same question. The compiled pattern is untouched and still byte-pinned
# to scan_packages; only how its answer is computed changes, and
# test_llama_cpp_converter_scan.py fuzzes the two against each other.
#
# Anything this decomposition cannot handle safely (a `.` wildcard nested inside
# a group, a `.+` whose "at least one character" would be lost, a segment that
# does not compile on its own) falls back to the pattern itself.


def _split_top_level_alternatives(source):
    """Split a regex source on `|` at paren depth 0, outside character classes."""
    parts, start, depth, in_class, i = [], 0, 0, False, 0
    while i < len(source):
        char = source[i]
        if char == "\\":
            i += 2
            continue
        if in_class:
            if char == "]":
                in_class = False
        elif char == "[":
            in_class = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "|" and depth == 0:
            parts.append(source[start:i])
            start = i + 1
        i += 1
    parts.append(source[start:])
    return parts


def _split_on_dot_star(alternative):
    """Split one alternative on top-level `.*` / `.*?`. None if it is not safe.

    Returns None for a `.+`, whose "at least one character" the split would drop,
    and for a `.` wildcard inside a group, where the split would break the group.
    """
    segments, start, depth, in_class, i = [], 0, 0, False, 0
    while i < len(alternative):
        char = alternative[i]
        if char == "\\":
            i += 2
            continue
        if in_class:
            if char == "]":
                in_class = False
            i += 1
            continue
        if char == "[":
            in_class = True
            i += 1
            continue
        if char == "(":
            depth += 1
            i += 1
            continue
        if char == ")":
            depth -= 1
            i += 1
            continue
        if char == "." and i + 1 < len(alternative) and alternative[i + 1] in "*+":
            if depth != 0:
                # Nested in a group, so not a top-level join. Left in place: it
                # still backtracks, but only inside one segment.
                i += 2
                continue
            if alternative[i + 1] == "+":
                # `.+` demands at least one character, which a split would drop,
                # and relaxing it to `.*` would invent matches. Use the pattern.
                return None
            end = i + 2
            if end < len(alternative) and alternative[end] == "?":
                end += 1
            segments.append(alternative[start:i])
            start = end
            i = end
            continue
        i += 1
    segments.append(alternative[start:])
    return [segment for segment in segments if segment]


def _end_is_pinned(segment):
    """True when this segment's match end is fixed by whatever it ends with.

    A non-final segment whose own tail can stretch, `/tmp/\\S+` for example, is
    not safe to chain from: the greedy match runs to the end of the token and the
    next segment is then searched from too far right, which loses matches the
    pattern would find by giving the tail back. Such a tail is made lazy instead,
    so the earliest match also has the earliest end.
    """
    stripped = segment[:-1] if segment.endswith("?") else segment
    return not stripped.endswith(("*", "+", "}"))


def _make_tail_lazy(segment):
    """`\\S+` to `\\S+?`. Minimal repetitions, so the match ends as early as it can."""
    if _end_is_pinned(segment):
        return segment
    return segment + "?"


# An unbounded negated class scans to end of input when its terminator is absent:
# `(?:open|Path)\s*\([^)]*?(?:\.ssh/...)` on a file of `open(` with no closing
# paren took 68s over 218 KB, quadratic in the input. Bounding the span keeps the
# rule's intent, which is "inside this call", and makes the cost linear. The cost
# is that a match needing more than this many characters between the two halves
# is missed; inside one function call that is not a shape worth paying 68s for.
MAX_CLASS_SPAN = 512


def _bound_dot_star(source, span = MAX_CLASS_SPAN):
    """Rewrite a surviving `.*` as `.{0,span}`, and `.+` as `.{1,span}`.

    _split_on_dot_star only splits the wildcards at the top level of an
    alternative; one nested inside a group is left whole on purpose, because the
    ordered-segment evaluation cannot represent it. That leaves it free to
    backtrack: RE_TEMP_EXEC's `(?:...|chmod.*\\+x)` over `/tmp/a ` followed by
    `chmod ` with no `+x` anywhere restarts the wildcard at every chmod and runs
    to the end each time, measured quadratic at 0.32s / 1.27s / 5.04s for
    96 / 192 / 384 KB, which is hours at the 8 MiB scan cap.

    Same trade as MAX_CLASS_SPAN and for the same reason: the rule means "these
    two near each other", a match needing more than this much between them is
    given up, and the cost becomes linear.
    """
    out, i = [], 0
    while i < len(source):
        char = source[i]
        if char == "\\":
            out.append(source[i:i + 2])
            i += 2
            continue
        if char == "[":
            end = i + 1
            if end < len(source) and source[end] == "^":
                end += 1
            if end < len(source) and source[end] == "]":
                end += 1
            while end < len(source) and source[end] != "]":
                end += 2 if source[end] == "\\" else 1
            out.append(source[i:end + 1])
            i = end + 1
            continue
        if char == "." and i + 1 < len(source) and source[i + 1] in "*+":
            low = 0 if source[i + 1] == "*" else 1
            i += 2
            lazy = ""
            if i < len(source) and source[i] == "?":
                lazy = "?"
                i += 1
            out.append(f".{{{low},{span}}}{lazy}")
            continue
        out.append(char)
        i += 1
    return "".join(out)


def _bound_class_repeats(source, span = MAX_CLASS_SPAN):
    """Rewrite `[^)]*` as `[^)]{0,span}`, preserving a lazy `?`."""
    out, i = [], 0
    while i < len(source):
        char = source[i]
        if char == "\\":
            out.append(source[i:i + 2])
            i += 2
            continue
        if char != "[":
            out.append(char)
            i += 1
            continue
        end = i + 1
        if end < len(source) and source[end] == "^":
            end += 1
        if end < len(source) and source[end] == "]":
            end += 1
        while end < len(source) and source[end] != "]":
            end += 2 if source[end] == "\\" else 1
        if end >= len(source):
            out.append(source[i:])
            break
        body = source[i:end + 1]
        i = end + 1
        if i < len(source) and source[i] == "*":
            i += 1
            lazy = ""
            if i < len(source) and source[i] == "?":
                lazy = "?"
                i += 1
            out.append(f"{body}{{0,{span}}}{lazy}")
        else:
            out.append(body)
    return "".join(out)


# Only the string anchors. A word boundary is decided by the characters either
# side of it, not by where the subject begins, and the spans below are cut at
# newlines, which are non-word characters on both readings. Treating \b as an
# anchor here excluded `\.service\b.*\[Service\]`, which is the one alternative
# that actually needed splitting.
_ANCHOR_TOKENS = ("^", "$", "\\A", "\\Z")


def _has_anchor(alternative):
    """Whether an alternative's meaning depends on where the subject starts or ends.

    Per-line evaluation reinterprets those, so an anchored alternative keeps the
    whole-text path even though it is the slower one.
    """
    text = alternative.decode("latin-1") if isinstance(alternative, bytes) else alternative
    return any(token in text for token in _ANCHOR_TOKENS)


def _literal_prefix(branch):
    """The leading run of `branch` that any match must contain verbatim.

    Stops at the first construct that is not a literal character, and gives back
    a character a quantifier could drop, so the result is always required.
    """
    out, i = [], 0
    while i < len(branch):
        char = branch[i]
        if char == "\\":
            if i + 1 >= len(branch):
                break
            nxt = branch[i + 1]
            if nxt.isalnum():
                break            # \s, \d, \b ... a class or an assertion
            out.append(nxt)
            i += 2
            continue
        if char in "[](){}|.^$":
            break
        if char in "*+?{":
            # Applies to the character just read, which is therefore optional.
            if out:
                out.pop()
            break
        out.append(char)
        i += 1
    return "".join(out)


def _literal_groups(alternative):
    """Every `(?:a|b|c)` in `alternative` whose match is mandatory, as branch lists.

    Character classes are stepped over rather than skipped: the group this exists
    for is the credential-marker alternation, and every branch of it ends in one
    (`\\.ssh[/\\\\]`), so a scanner that gave up at `[` never saw the only group
    worth probing.
    """
    groups, i = [], 0
    while i < len(alternative):
        if alternative[i] == "\\":
            i += 2
            continue
        if alternative[i] == "[":
            i += 1
            if i < len(alternative) and alternative[i] == "^":
                i += 1
            if i < len(alternative) and alternative[i] == "]":
                i += 1
            while i < len(alternative) and alternative[i] != "]":
                i += 2 if alternative[i] == "\\" else 1
            i += 1
            continue
        if not alternative.startswith("(?:", i):
            i += 1
            continue
        body, j, depth = [], i + 3, 0
        while j < len(alternative):
            char = alternative[j]
            if char == "\\":
                body.append(alternative[j:j + 2])
                j += 2
                continue
            if char == "[":
                k = j + 1
                if k < len(alternative) and alternative[k] == "^":
                    k += 1
                if k < len(alternative) and alternative[k] == "]":
                    k += 1
                while k < len(alternative) and alternative[k] != "]":
                    k += 2 if alternative[k] == "\\" else 1
                body.append(alternative[j:k + 1])
                j = k + 1
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                if depth == 0:
                    break
                depth -= 1
            body.append(char)
            j += 1
        if j >= len(alternative):
            break
        # A group a quantifier can skip is not required.
        if j + 1 >= len(alternative) or alternative[j + 1] not in "?*{":
            groups.append("".join(body).split("|"))
        i = j + 1
    return groups


def _required_literal_probes(alternative, flags):
    """Regexes matching literals that EVERY match of `alternative` must contain.

    Bounding the negated class made this pattern's cost linear in the input, but
    the number of candidate starts stays attacker-controlled: RE_CRED_ACCESS over
    a single line of repeated `open(` ran about 3.2s per MB, so roughly 26s at the
    8 MiB scan cap, synchronously and before the converter runs. A credential rule
    cannot match without one of its credential markers, and that is one literal
    pass, so a file carrying none is rejected up front instead of being walked
    once per call prefix.

    Every qualifying group, not the most promising one: the marker group and the
    call-prefix group are both required, so probing both is both exact and more
    selective than either. Exact rather than heuristic throughout, so nothing that
    would have matched is skipped.
    """
    probes = []
    for branches in _literal_groups(alternative):
        if len(branches) < 2:
            continue
        prefixes = [_literal_prefix(branch) for branch in branches]
        if not all(len(prefix) >= 3 for prefix in prefixes):
            continue         # too short to be worth a pass, or not all literal
        probes.append(re.compile("|".join(re.escape(x) for x in prefixes), flags))
    return probes


def _build_evaluator(pattern):
    """Compile `pattern` into alternatives of ordered, individually bounded segments.

    Returns `(alternatives, per_line_flags, probes)`, one of each per alternative. A set
    flag says the caller must apply that alternative's segments to one line at a
    time: a non-DOTALL `.*` cannot cross a newline, so that is what the pattern
    already meant, and it is the only thing that bounds the span when the file is
    one enormous line. Bounding by "a line" was the original reasoning here and it
    is only true when lines are short; a crafted 352 KB single line took over five
    seconds through RE_PERSISTENCE.

    Per alternative, not per pattern. One flag for the whole pattern meant a
    single `.*` alternative put every OTHER alternative on the line-at-a-time
    path, including ones that legitimately span lines through `\\s*`: in
    RE_ENV_HARVEST the last two alternatives carry the `.*` and the third is
    `json.dumps(\\s*os.environ`, so `json.dumps(\\n    os.environ\\n)` matched
    re.search and not _matches. Paired with a network send that is a finding the
    scan silently dropped, in strict mode too.
    """
    dotall = bool(pattern.flags & re.DOTALL)
    alternatives = []
    per_line_flags = []
    probes = []
    for alternative in _split_top_level_alternatives(pattern.pattern):
        segments = None
        per_line = False
        probes.append(_required_literal_probes(
            alternative.decode("latin-1") if isinstance(alternative, bytes) else alternative,
            pattern.flags,
        ))
        if ".*" in (
            alternative.decode("latin-1") if isinstance(alternative, bytes) else alternative
        ):
            if dotall:
                segments = _split_on_dot_star(alternative)
            elif not _has_anchor(alternative):
                segments = _split_on_dot_star(alternative)
                if segments is not None and len(segments) > 1:
                    per_line = True
        if segments is None:
            segments = [alternative]
        segments = [_make_tail_lazy(s) for s in segments[:-1]] + segments[-1:]
        alternatives.append(
            [
                re.compile(_bound_dot_star(_bound_class_repeats(s)), pattern.flags)
                for s in segments
            ]
        )
        per_line_flags.append(per_line)
    return alternatives, per_line_flags, probes


_EVALUATORS = {}


def _evaluator(pattern):
    if pattern not in _EVALUATORS:
        try:
            _EVALUATORS[pattern] = _build_evaluator(pattern)
        except Exception:
            _EVALUATORS[pattern] = ([[pattern]], [False], [[]])
    return _EVALUATORS[pattern]


def _segments_match(segments, text, start = 0, end = None):
    position = start
    limit = len(text) if end is None else end
    for segment in segments:
        found = segment.search(text, position, limit)
        if found is None:
            return False
        position = found.end()
    return True


def _matches(pattern, text):
    """`bool(pattern.search(text))`, without the backtracking blowup."""
    alternatives, per_line_flags, probes = _evaluator(pattern)
    line_bound = []
    for segments, per_line, probe in zip(alternatives, per_line_flags, probes):
        if any(not required.search(text) for required in probe):
            continue      # a literal this alternative requires is nowhere in the file
        if per_line:
            line_bound.append(segments)
        elif _segments_match(segments, text):
            return True
    if not line_bound:
        return False
    # Walk line spans in place rather than materialising them: the whole match
    # has to sit inside one line for a non-DOTALL pattern anyway.
    newline = b"\n" if isinstance(text, (bytes, bytearray)) else "\n"
    start = 0
    length = len(text)
    while start <= length:
        stop = text.find(newline, start)
        if stop == -1:
            stop = length
        for segments in line_bound:
            if _segments_match(segments, text, start, stop):
                return True
        start = stop + 1
    return False


# Evidence is extracted a line at a time, which bounds the same patterns, unless
# a "line" is the whole file. Probe a prefix rather than reintroduce the hang.
MAX_EVIDENCE_LINE_CHARS = 4096


def _matching_lines(lines, probe, max_matches):
    found = []
    for i, line in enumerate(lines, 1):
        if _matches(probe, line[:MAX_EVIDENCE_LINE_CHARS]):
            snippet = line.strip()
            if len(snippet) > 160:
                snippet = snippet[:160] + "..."
            found.append(f"L{i}: {snippet}")
            if len(found) >= max_matches:
                break
    return found


def _extract_evidence(content, pattern, max_matches = 3):
    """Pull matching lines as evidence snippets. Ported from scan_packages.py.

    With one addition: a rule like `socket .* connect .* subprocess` matches the
    file but no single line, and canonical then reports the finding with no
    evidence at all. Where the whole pattern finds nothing line by line, the
    lines matching its individual parts are reported instead, so a warning always
    points somewhere the reader can look.
    """
    lines = content.splitlines()
    matches = _matching_lines(lines, pattern, max_matches)
    if matches:
        return " | ".join(matches)
    for segments in _evaluator(pattern)[0]:
        if len(segments) < 2:
            continue
        parts, located = [], 0
        for segment in segments:
            for snippet in _matching_lines(lines, segment, 1):
                located += 1
                if snippet not in parts:      # two parts often share one line
                    parts.append(snippet)
            if located >= max_matches:
                break
        # Only report this alternative if every part of it was located, or the
        # snippet budget ran out first. A partial alternative is not why the
        # pattern matched.
        if located >= min(len(segments), max_matches):
            return " | ".join(parts[:max_matches])
    return ""


def _default_is_inert(node):
    """True when evaluating this expression cannot do anything but produce a value.

    Literals are inert. So is a plain name or dotted name: with no call syntax
    there is nothing to invoke, and these are what a converter that defines its
    default next to a constant looks like.
    """
    try:
        ast.literal_eval(node)
        return True
    except Exception:
        pass
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name)


def _argparse_default_findings(raw):
    """Flag argparse defaults that would do something when eval() reaches them.

    llama_cpp.py scrapes these tokens out of the downloaded file and eval()s them
    in the Unsloth process. The capture stops at the first comma, whitespace or
    closing paren, so a complete call cannot survive it and neither can most
    ordinary defaults: `default=[0, 1]` arrives as `[0`, `default=os.cpu_count()`
    as `os.cpu_count(`. Those raise SyntaxError inside llama_cpp.py's own
    try/except, which logs and falls back to None. A token that cannot be parsed
    cannot run, so it is not reported: flagging it would warn on a converter that
    did nothing wrong.

    What survives the capture and still evaluates is subscripting and attribute
    access, so an expression that parses and is neither a literal nor a name is
    what gets reported.
    """
    findings = []
    for flag_bytes, default_bytes in RE_ARGPARSE_DEFAULT.findall(raw):
        try:
            flag = flag_bytes.decode("utf-8", errors = "replace")
            default = default_bytes.decode("utf-8", errors = "replace")
        except Exception:
            continue
        if RE_PLAIN_NAME.match(default):
            continue
        try:
            node = ast.parse(default, mode = "eval").body
        except Exception:
            # Not a parseable expression, so eval() raises and nothing runs.
            continue
        if _default_is_inert(node):
            continue
        findings.append(
            ConverterScanFinding(
                CRITICAL,
                "argparse default evaluates to more than a literal, and Unsloth eval()s it in-process",
                f"{flag}: default={default}",
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------
def _decode_as_python_would(raw):
    """Decode `raw` the way the interpreter that runs it will.

    Returns (text, declared_encoding), the second being None when the file makes
    no PEP 263 declaration and there is nothing to report about it.

    The scanner must read the same characters the interpreter does, and a file
    gets to choose that: `# coding: utf-7` on line one makes
    `+AHM-+AHU-+AGI-+AHA-+AHI-+AG8-+AGM-+AGU-+AHM-+AHM-.run(...)` the source text
    `subprocess.run(...)`. Decoded as UTF-8 that is a string of plus signs and
    capital letters, matching nothing here, while the file Unsloth then writes
    and executes spawns the process. Confirmed against the scanner as it stood:
    zero findings on bytes whose declared decoding is a subprocess call.

    A cookie CPython itself rejects reports nothing: the file will not parse, so
    there is no decoding for the scan to disagree with and no execution to warn
    about. What gets reported is the case that DOES run and runs as something
    else.
    """
    import io
    import tokenize

    try:
        encoding, _lines = tokenize.detect_encoding(io.BytesIO(raw).readline)
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return raw.decode("utf-8", errors = "replace"), None
    if encoding.lower().replace("_", "-") in ("utf-8", "utf-8-sig"):
        return raw.decode(encoding, errors = "replace"), None
    try:
        return raw.decode(encoding, errors = "replace"), encoding
    except (LookupError, UnicodeDecodeError):
        return raw.decode("utf-8", errors = "replace"), encoding


def scan_converter_source(content, filename = "convert_hf_to_gguf.py"):
    """Return a list of ConverterScanFinding for one converter script.

    ``content`` may be bytes or str. The rules are the CRITICAL and HIGH tiers of
    ``scan_packages.check_py_file``. The MEDIUM tier there is documented as
    "informational, may be legitimate" (standalone wallet strings, a filesystem
    walk, an env-var copy); on a path that runs on every GGUF export, a warning
    nobody can act on trains people to ignore the ones that matter, so it is not
    ported. The realistic staged-payload shapes are still covered: base64 plus
    subprocess is CRITICAL, obfuscation plus exec/eval is HIGH. The setup.py rule
    is not ported either, since the scanned file is never a setup.py.
    """
    declared_encoding = None
    if isinstance(content, bytes):
        raw = content
        text, declared_encoding = _decode_as_python_would(content)
    else:
        raw = content.encode("utf-8", errors = "replace")
        text = content

    truncated = None
    if len(raw) > MAX_SCAN_BYTES:
        # The whole file is still patched, written and executed, so a partial
        # scan has to be said out loud rather than left to look like a pass.
        # Reported at the end, so it cannot perturb the one order-dependent rule
        # carried over from check_py_file.
        truncated = ConverterScanFinding(
            HIGH,
            "Converter is larger than the scan cap, so only its first "
            f"{MAX_SCAN_BYTES} bytes were scanned",
            f"{len(raw)} bytes; a real converter is under 500 KB",
        )
        raw = raw[:MAX_SCAN_BYTES]
        text = text[:MAX_SCAN_BYTES]

    findings = []

    has_network      = _matches(RE_NETWORK, text)
    has_subprocess   = _matches(RE_SUBPROCESS, text)
    has_base64       = _matches(RE_BASE64, text)
    has_exec_eval    = _matches(RE_EXEC_EVAL, text)
    has_creds        = _matches(RE_CRED_ACCESS, text)
    has_blob         = _matches(RE_LARGE_BLOB, text)
    has_obfuscation  = _matches(RE_OBFUSCATION, text)
    has_keys         = _matches(RE_EMBEDDED_KEYS, text)
    has_cloud_meta   = _matches(RE_CLOUD_METADATA, text)
    has_persistence  = _matches(RE_PERSISTENCE, text)
    has_container    = _matches(RE_CONTAINER_ABUSE, text)
    has_env_harvest  = _matches(RE_ENV_HARVEST, text)
    has_archive      = _matches(RE_ARCHIVE_STAGING, text)
    has_anti         = _matches(RE_ANTI_ANALYSIS, text)
    has_dns_exfil    = _matches(RE_DNS_EXFIL, text)
    has_fs_enum      = _matches(RE_FS_ENUM, text)
    has_rev_shell    = _matches(RE_REVERSE_SHELL, text)
    has_remote_code  = _matches(RE_REMOTE_CODE, text)
    has_crypto_theft = _matches(RE_CRYPTO_THEFT, text)
    has_openssl_cli  = _matches(RE_OPENSSL_CLI, text)
    has_temp_exec    = _matches(RE_TEMP_EXEC, text)
    has_c2_polling   = _matches(RE_C2_POLLING, text)
    has_may12_ioc    = _matches(RE_MAY12_IOC, text)

    def _add(severity, check, *patterns):
        evidence = "\n".join(
            e for e in (_extract_evidence(text, p) for p in patterns) if e
        )
        findings.append(ConverterScanFinding(severity, check, evidence))

    # --- CRITICAL: combinations that strongly indicate malice ---------------
    if has_base64 and has_subprocess:
        _add(CRITICAL, "base64 decode + subprocess execution (staged payload)",
             RE_BASE64, RE_SUBPROCESS)
    if has_openssl_cli and (has_network or has_keys):
        _add(CRITICAL, "openssl encryption + network/key material (encrypted exfiltration)",
             RE_OPENSSL_CLI, RE_NETWORK)
    if has_temp_exec:
        _add(CRITICAL, "Writes to /tmp and executes (staged dropper)", RE_TEMP_EXEC)
    if has_may12_ioc:
        _add(CRITICAL, "May-12 Shai-Hulud IOC string present in Python file", RE_MAY12_IOC)
    if has_c2_polling:
        _add(CRITICAL, "C2 polling/beaconing loop detected", RE_C2_POLLING)
    if has_creds and has_network:
        _add(CRITICAL, "Reads credential paths AND makes network calls",
             RE_CRED_ACCESS, RE_NETWORK)
    if has_rev_shell:
        _add(CRITICAL, "Reverse shell / bind shell pattern", RE_REVERSE_SHELL)
    if has_remote_code:
        _add(CRITICAL, "Downloads and executes remote code", RE_REMOTE_CODE)
    if has_env_harvest and has_network and not _talks_only_to_the_model_hub(text):
        _add(CRITICAL, "Harvests environment variables/secrets AND makes network calls",
             RE_ENV_HARVEST, RE_NETWORK)
    if has_fs_enum and has_network:
        _add(CRITICAL, "Enumerates filesystem AND makes network calls",
             RE_FS_ENUM, RE_NETWORK)
    if has_cloud_meta and has_network:
        _add(CRITICAL, "Accesses cloud metadata/IMDS AND makes network calls",
             RE_CLOUD_METADATA, RE_NETWORK)
    if has_crypto_theft and has_network:
        _add(CRITICAL, "Targets cryptocurrency wallets AND makes network calls",
             RE_CRYPTO_THEFT, RE_NETWORK)
    if has_archive and has_network:
        _add(CRITICAL, "Creates archive with sensitive data AND makes network calls",
             RE_ARCHIVE_STAGING, RE_NETWORK)
    if has_persistence and has_network:
        _add(CRITICAL, "Installs persistence AND makes network calls (backdoor pattern)",
             RE_PERSISTENCE, RE_NETWORK)
    if has_container and has_network:
        _add(CRITICAL, "Container/orchestration abuse AND makes network calls",
             RE_CONTAINER_ABUSE, RE_NETWORK)

    # --- HIGH: single strong signals or weaker combinations -----------------
    if has_base64 and has_exec_eval and has_blob:
        _add(HIGH, "base64 decode + exec/eval + large encoded blob",
             RE_BASE64, RE_EXEC_EVAL)
    if has_obfuscation and has_exec_eval:
        _add(HIGH, "Advanced obfuscation (marshal/compile/zlib) + exec/eval",
             RE_OBFUSCATION, RE_EXEC_EVAL)
    if has_keys and has_network:
        _add(HIGH, "Embedded cryptographic key + network calls (encrypted exfil pattern)",
             RE_EMBEDDED_KEYS, RE_NETWORK)
    if has_anti and (has_network or has_subprocess or has_exec_eval):
        _add(HIGH, "Anti-analysis/sandbox evasion + suspicious behavior", RE_ANTI_ANALYSIS)
    if has_dns_exfil and (has_base64 or has_network or has_creds):
        _add(HIGH, "DNS exfiltration / tunneling patterns", RE_DNS_EXFIL)
    # Ordering carried over from check_py_file: standalone IMDS access is only
    # worth reporting when nothing stronger already fired.
    if has_cloud_meta and not findings:
        _add(HIGH, "Accesses cloud metadata / IMDS endpoints", RE_CLOUD_METADATA)
    if has_persistence and not has_network:
        _add(HIGH, "Installs persistence mechanism (systemd/cron/launchd/registry)",
             RE_PERSISTENCE)
    if has_container and not has_network:
        _add(HIGH, "Interacts with container/orchestration runtime", RE_CONTAINER_ABUSE)
    if has_openssl_cli and not (has_network or has_keys):
        _add(HIGH, "Invokes openssl CLI (uncommon in a model converter)", RE_OPENSSL_CLI)

    # --- Converter-specific: any process execution at all -------------------
    # check_py_file only reports subprocess use in combination with base64 or
    # anti-analysis markers, which is right for a general package and wrong here.
    # A converter reads tensors and writes a GGUF; it spawns nothing. So
    # os.system("curl ... | sh") on its own produced no finding, and neither did
    # subprocess.run(["rm", "-rf", path]), because a shell-level curl does not
    # match RE_NETWORK either. Verified against the current entrypoint and the
    # b4000 monolith: neither matches RE_SUBPROCESS, so this cannot fire on a
    # genuine converter.
    if has_subprocess and not has_base64 and not has_anti:
        _add(HIGH, "Spawns a process (a converter reads tensors and writes a GGUF)",
             RE_SUBPROCESS)

    # --- Converter-specific: the in-process eval sink -----------------------
    findings.extend(_argparse_default_findings(raw))

    if truncated is not None:
        findings.append(truncated)

    if declared_encoding is not None:
        # Reported whether or not anything else matched. Every real converter is
        # UTF-8, so a declaration is already odd, and its whole effect is to make
        # the scanner and the interpreter read different characters -- which is
        # worth a line in the warning even on the run where the rules above,
        # reading the declared decoding, came back clean.
        findings.append(ConverterScanFinding(
            HIGH,
            "Converter declares a non-UTF-8 source encoding, so its text is not "
            "what it looks like on the wire",
            f"PEP 263 cookie: {declared_encoding}",
        ))

    return findings


def _scan_is_disabled():
    return os.environ.get(ENV_DISABLE_SCAN, "0") == "1"


def _scan_is_strict():
    return os.environ.get(ENV_STRICT_SCAN, "0") == "1"


def scan_is_strict():
    """Whether a finding in downloaded bytes should fail the export.

    Public because llama_cpp.py has one refusal that cannot travel through
    warn_on_suspicious_converter: a conversion/ package holding more modules than
    the scan will read is a fact about the DIRECTORY, not about any bytes, and
    that function takes the bytes it scans.
    """
    return _scan_is_strict()


def scan_is_disabled():
    """Whether the scan is switched off entirely, for the same caller."""
    return _scan_is_disabled()


def warn_on_suspicious_converter(
    content,
    source,
    is_local_copy = False,
    log = None,
):
    """Scan converter bytes and warn about anything the scan flags.

    ``source`` is the URL or path the bytes came from and is named in the
    warning. ``is_local_copy`` marks bytes that did not come off this download: a
    converter pinned with UNSLOTH_LLAMA_CPP_SCRIPTS_DIR, or one hydrated from a
    prebuilt bundle whose asset was checked against the published sha256 list.
    Both are still scanned, because a stale or tampered checkout is exactly as
    dangerous as a bad download, but neither ever hard-fails: pinning a local
    converter is the remedy this warning recommends, and hard-failing the remedy
    would leave the user with no way forward.

    Returns the findings. Raises ConverterScanError only when there is a finding,
    UNSLOTH_CONVERTER_SCAN_STRICT=1, and the bytes were downloaded.
    """
    log = log if log is not None else logger
    if _scan_is_disabled():
        return []

    # A bug in the scanner, or a pattern that misbehaves on some future
    # converter, must not be able to break a legitimate export. Only the scan
    # is inside this guard; the strict-mode raise below is deliberately outside
    # it so it cannot be swallowed.
    try:
        findings = scan_converter_source(content, source)
    except Exception as error:
        log.warning(
            "Unsloth: The converter safety scan could not run on %s (%s: %s). "
            "Continuing with the conversion; the scan is advisory.",
            source, type(error).__name__, error,
        )
        return []

    if not findings:
        return []

    log.warning(
        "Unsloth: %d suspicious pattern(s) in the llama.cpp converter at %s. "
        "This file is executed to build your GGUF, so read it before continuing.",
        len(findings), source,
    )
    for finding in findings:
        log.warning("Unsloth:   [%s] %s", finding.severity, finding.check)
        for line in finding.evidence.splitlines():
            if line.strip():
                log.warning("Unsloth:     %s", line)
    log.warning(
        "Unsloth: To pin a converter you have reviewed, put convert_hf_to_gguf.py in a "
        "directory and set UNSLOTH_LLAMA_CPP_SCRIPTS_DIR to it; Unsloth then reads that "
        "file instead of downloading one. Set UNSLOTH_CONVERTER_SCAN_STRICT=1 to make "
        "findings in a downloaded converter fail the export instead of warning, or "
        "UNSLOTH_DISABLE_CONVERTER_SCAN=1 to silence this scan."
    )

    if _scan_is_strict() and not is_local_copy:
        raise ConverterScanError(
            f"Unsloth: Refusing to run the downloaded llama.cpp converter from {source}: "
            f"{len(findings)} suspicious pattern(s) and UNSLOTH_CONVERTER_SCAN_STRICT=1. "
            f"Review the warnings above, then either unset "
            f"UNSLOTH_CONVERTER_SCAN_STRICT or pin a reviewed converter with "
            f"UNSLOTH_LLAMA_CPP_SCRIPTS_DIR."
        )
    return findings

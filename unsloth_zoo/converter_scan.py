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


def _build_evaluator(pattern):
    """Compile `pattern` into alternatives of ordered, individually bounded segments.

    Returns `(alternatives, per_line)`. `per_line` says the caller must apply the
    segments to one line at a time: a non-DOTALL `.*` cannot cross a newline, so
    that is what the pattern already meant, and it is the only thing that bounds
    the span when the file is one enormous line. Bounding by "a line" was the
    original reasoning here and it is only true when lines are short; a crafted
    352 KB single line took over five seconds through RE_PERSISTENCE.
    """
    dotall = bool(pattern.flags & re.DOTALL)
    alternatives = []
    per_line = False
    for alternative in _split_top_level_alternatives(pattern.pattern):
        segments = None
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
            [re.compile(_bound_class_repeats(s), pattern.flags) for s in segments]
        )
    return alternatives, per_line


_EVALUATORS = {}


def _evaluator(pattern):
    if pattern not in _EVALUATORS:
        try:
            _EVALUATORS[pattern] = _build_evaluator(pattern)
        except Exception:
            _EVALUATORS[pattern] = ([[pattern]], False)
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
    alternatives, per_line = _evaluator(pattern)
    if not per_line:
        for segments in alternatives:
            if _segments_match(segments, text):
                return True
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
        for segments in alternatives:
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
    if isinstance(content, bytes):
        raw = content
        text = content.decode("utf-8", errors = "replace")
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
    if has_env_harvest and has_network:
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

    return findings


def _scan_is_disabled():
    return os.environ.get(ENV_DISABLE_SCAN, "0") == "1"


def _scan_is_strict():
    return os.environ.get(ENV_STRICT_SCAN, "0") == "1"


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

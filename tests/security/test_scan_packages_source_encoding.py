# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Python honours a PEP 263 coding cookie, so the scanner must read `.py` members in that encoding: a
`# coding: utf-7` module read as UTF-8 runs `os.system` while every pattern sees `+AG8AcwAu-...`."""

from __future__ import annotations

import base64
import io
import sys
import tarfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import scan_packages as sp  # noqa: E402

PAYLOAD = "import base64, os\nos.system(base64.b64decode('aWQ=').decode())\n"


def _utf7_shifted(text: str) -> bytes:
    # UTF-7 may base64 any character, letters included; Python's encoder only shifts the ones it must.
    out = b""
    for line in text.splitlines(keepends = True):
        body = line.rstrip("\n")
        out += b"+" + base64.b64encode(body.encode("utf-16-be")).rstrip(b"=") + b"-" + b"\n" * line.endswith("\n")
    return out


UTF7 = b"# coding: utf-7\n" + _utf7_shifted(PAYLOAD)


def _severe(findings):
    return [f for f in findings if f.severity in (sp.CRITICAL, sp.HIGH)]


def test_payload_really_runs_as_utf7():
    assert b"system" not in UTF7 and b"base64" not in UTF7
    assert compile(UTF7, "x.py", "exec").co_names[:4] == ("base64", "os", "system", "b64decode")


def test_plain_payload_is_flagged(tmp_path):
    whl = tmp_path / "pkg-1.0-py3-none-any.whl"
    with zipfile.ZipFile(whl, "w") as zf:
        zf.writestr("pkg/__init__.py", PAYLOAD)
    assert _severe(sp.scan_archive(str(whl), "pkg"))


def test_utf7_wheel_member_is_flagged(tmp_path):
    whl = tmp_path / "pkg-1.0-py3-none-any.whl"
    with zipfile.ZipFile(whl, "w") as zf:
        zf.writestr("pkg/__init__.py", UTF7)
    assert _severe(sp.scan_archive(str(whl), "pkg"))


def test_utf7_sdist_member_is_flagged(tmp_path):
    sdist = tmp_path / "pkg-1.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as tf:
        info = tarfile.TarInfo("pkg-1.0/pkg/__init__.py")
        info.size = len(UTF7)
        tf.addfile(info, io.BytesIO(UTF7))
    assert _severe(sp.scan_archive(str(sdist), "pkg"))


def test_plain_and_bad_cookie_members_still_decode():
    assert sp._decode_member("a.py", b"x = 1\n") == "x = 1\n"
    assert sp._decode_member("a.py", b"# coding: no-such-codec\nx = 1\n").endswith("x = 1\n")
    assert sp._decode_member("a.js", b"# coding: utf-7\n+AGk-") == "# coding: utf-7\n+AGk-"

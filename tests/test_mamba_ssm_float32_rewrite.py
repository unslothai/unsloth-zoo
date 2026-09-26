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

"""fix_mamba_ssm_float32 must rewrite ssd_chunk_scan.py atomically: an in-place write let a
concurrent reader see an empty module (issue: "cannot import name '_chunk_scan_fwd'")."""

import ast
import builtins
import importlib
import inspect
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

MISC = Path(__file__).resolve().parents[1] / "unsloth_zoo" / "temporary_patches" / "misc.py"
_SRC = MISC.read_text(encoding = "utf-8")
_NAME = "fix_mamba_ssm_float32"
_MODS = ("mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton", "mamba_ssm.ops.triton.ssd_chunk_scan")

_KERNEL = "".join(
    f"@_Autotune\n"
    f"@_JIT\n"
    f"def _kernel_{i}(a, b, cb, x, dout, c):\n"
    f"    acc = tl.dot(cb, x)\n"
    f"    acc += tl.dot(dout, c)\n"
    f"    tmp = tl.dot(a, b)\n"
    f"    return acc, tmp  # {'pad ' * 40}\n\n"
    for i in range(400)
)
_MODULE = (
    "import inspect\n"
    "from pathlib import Path\n"
    "UPCAST = ('to(tl.' + 'float32)') in Path(__file__).read_text()  # split, so this line never matches\n"
    "class _JIT:\n"
    "    # Like triton.JITFunction: keeps the source the kernel was compiled from.\n"
    "    def __init__(self, fn):\n"
    "        self.src = inspect.getsource(fn)\n"
    "class _Autotune:\n"
    "    # Like triton.autotune: wraps the JIT function in `.fn`.\n"
    "    def __init__(self, fn):\n"
    "        self.fn = fn\n"
    "def _chunk_scan_fwd():\n"
    "    return 1\n\n"
    + _KERNEL
)


def _function_source():
    for node in ast.parse(_SRC).body:
        if isinstance(node, ast.FunctionDef) and node.name == _NAME:
            return ast.get_source_segment(_SRC, node)
    raise AssertionError(f"{_NAME} not found")


def _load(open_fn = open):
    ns = {"inspect": inspect, "importlib": importlib, "re": re, "os": os,
          "raise_error": lambda *a, **k: None, "open": open_fn}
    exec(_function_source(), ns)
    return ns[_NAME]


def _make_package(root: Path) -> Path:
    triton = root / "mamba_ssm" / "ops" / "triton"
    triton.mkdir(parents = True)
    for d in (root / "mamba_ssm", root / "mamba_ssm" / "ops", triton):
        (d / "__init__.py").write_text("")
    target = triton / "ssd_chunk_scan.py"
    target.write_text(_MODULE, encoding = "utf-8")
    return target


@pytest.fixture
def fake_mamba(tmp_path, monkeypatch):
    target = _make_package(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    saved = {m: sys.modules.pop(m, None) for m in _MODS}
    importlib.invalidate_caches()
    yield target
    for m in _MODS:
        sys.modules.pop(m, None)
        if saved[m] is not None:
            sys.modules[m] = saved[m]


def test_upcasts_every_dot_and_reloads(fake_mamba):
    _load()()
    src = fake_mamba.read_text(encoding = "utf-8")
    assert " acc = tl.dot(cb.to(tl.float32), x.to(tl.float32))" in src
    assert " acc = tl.dot(dout.to(tl.float32), c.to(tl.float32), acc = acc)" in src
    assert " tmp = tl.dot(a.to(tl.float32), b.to(tl.float32))" in src
    assert re.search(r"tl\.dot\([a-z]+, [a-z]+\)", src) is None
    assert "def _chunk_scan_fwd" in src
    assert sys.modules["mamba_ssm.ops.triton.ssd_chunk_scan"].UPCAST is True


def test_already_upcast_file_is_not_rewritten(fake_mamba):
    patch = _load()
    patch()
    before = os.stat(fake_mamba)
    content = fake_mamba.read_bytes()
    os.utime(fake_mamba, ns = (before.st_atime_ns, before.st_mtime_ns - 10_000_000_000))
    before = os.stat(fake_mamba)
    patch()
    after = os.stat(fake_mamba)
    assert fake_mamba.read_bytes() == content
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


def test_installed_file_is_never_opened_for_writing(fake_mamba):
    opened = []
    def recording_open(file, mode = "r", *args, **kwargs):
        opened.append((os.path.realpath(file) if isinstance(file, (str, Path)) else file, mode))
        return builtins.open(file, mode, *args, **kwargs)
    _load(recording_open)()
    target = os.path.realpath(fake_mamba)
    assert not [m for f, m in opened if f == target and any(c in m for c in "wax+")]
    assert "tl.float32" in fake_mamba.read_text(encoding = "utf-8")


def test_keeps_permissions_and_leaves_no_temp_files(fake_mamba):
    os.chmod(fake_mamba, 0o644)
    _load()()
    assert os.stat(fake_mamba).st_mode & 0o777 == 0o644
    assert sorted(p.name for p in fake_mamba.parent.iterdir()) == ["__init__.py", "ssd_chunk_scan.py"]


def test_unwritable_directory_leaves_the_file_intact(fake_mamba):
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    content = fake_mamba.read_bytes()
    mode = os.stat(fake_mamba.parent).st_mode
    os.chmod(fake_mamba.parent, 0o555)
    try:
        _load()()
    finally:
        os.chmod(fake_mamba.parent, mode)
    assert fake_mamba.read_bytes() == content


_WORKER = textwrap.dedent("""
    import ast, importlib, inspect, os, re, sys
    misc, root, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3])
    sys.path.insert(0, root)
    sys.dont_write_bytecode = True
    src = open(misc, encoding = "utf-8").read()
    node = next(n for n in ast.parse(src).body
                if isinstance(n, ast.FunctionDef) and n.name == "fix_mamba_ssm_float32")
    ns = {"inspect": inspect, "importlib": importlib, "re": re, "os": os,
          "raise_error": lambda *a, **k: None}
    exec(ast.get_source_segment(src, node), ns)
    for _ in range(rounds):
        for m in [m for m in sys.modules if m.startswith("mamba_ssm")]:
            del sys.modules[m]
        importlib.invalidate_caches()
        try:
            ns["fix_mamba_ssm_float32"]()
        except Exception:
            pass
""")


def test_concurrent_processes_never_truncate_the_module(tmp_path):
    target = _make_package(tmp_path)
    workers = [
        subprocess.Popen([sys.executable, "-c", _WORKER, str(MISC), str(tmp_path), "40"])
        for _ in range(8)
    ]
    short_reads = 0
    try:
        while any(w.poll() is None for w in workers):
            if "def _chunk_scan_fwd" not in target.read_text(encoding = "utf-8"):
                short_reads += 1
    finally:
        for w in workers:
            try:
                w.wait(timeout = 120)
            except subprocess.TimeoutExpired:
                w.kill()
                w.wait()
    final = target.read_text(encoding = "utf-8")
    assert "def _chunk_scan_fwd" in final, "the module was left truncated"
    assert "tl.float32" in final
    assert short_reads == 0, f"a reader saw a truncated module {short_reads} times"


def _upcast_text(text):
    patch_ns = {}
    exec("import re", patch_ns)
    return re.sub(
        r" ([a-zA-Z0-9\_]{1,}) (\=|\+\=) tl\.dot\(([a-zA-Z0-9\_]{1,})\, ([a-zA-Z0-9\_]{1,})\)",
        lambda m: f" {m.group(1)} = tl.dot({m.group(3)}.to(tl.float32), {m.group(4)}.to(tl.float32)"
                  + ("" if m.group(2) == "=" else f", acc = {m.group(1)}") + ")",
        text,
    )


def test_a_module_imported_before_a_peer_rewrite_is_reloaded(fake_mamba):
    import mamba_ssm.ops.triton.ssd_chunk_scan as mod
    assert mod.UPCAST is False
    tmp = fake_mamba.with_name(".peer.tmp")
    tmp.write_text(_upcast_text(fake_mamba.read_text(encoding = "utf-8")), encoding = "utf-8")
    os.replace(tmp, fake_mamba)
    _load()()
    mod = sys.modules["mamba_ssm.ops.triton.ssd_chunk_scan"]
    assert mod.UPCAST is True
    assert "to(tl.float32)" in mod._kernel_0.fn.src


def test_an_upcast_module_is_not_reloaded(fake_mamba, monkeypatch):
    patch = _load()
    patch()
    reloads = []
    real_reload = importlib.reload
    monkeypatch.setattr(importlib, "reload", lambda m: reloads.append(m) or real_reload(m))
    patch()
    assert reloads == []

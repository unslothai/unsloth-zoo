"""keep_newest_blackwell_ptxas: pin Triton's bundled ptxas-blackwell only when flashinfer would swap in an older
system ptxas. CPU only: find_spec, torch.version, get_ptxas, shutil.which and subprocess are mocked."""

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "unsloth_zoo" / "triton_ptxas.py"
_spec = importlib.util.spec_from_file_location("_zoo_triton_ptxas_under_test", _PATH)
tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tp)

ENV = "TRITON_PTXAS_BLACKWELL_PATH"


class _Tool:
    def __init__(self, path, version):
        self.path, self.version = path, version


def _flashinfer_rule():
    """What importing flashinfer does: system ptxas whenever it is CUDA 13+ and the variable is unset."""
    if os.environ.get(ENV):
        return
    ptxas = shutil.which("ptxas")
    if not ptxas:
        return
    out = subprocess.check_output([ptxas, "--version"], stderr = subprocess.STDOUT).decode()
    m = re.search(r"release (\d+)\.", out)
    if m and int(m.group(1)) >= 13:
        os.environ[ENV] = ptxas


@pytest.fixture
def host(monkeypatch, tmp_path):
    bundled = tmp_path / "ptxas-blackwell"
    bundled.write_text("")
    system = tmp_path / "ptxas"
    system.write_text("")
    state = {"bundled": str(bundled), "bundled_ver": "13.3", "system": str(system), "system_ver": "13.1",
             "flashinfer": True, "hip": None, "cuda": "13.0", "get_ptxas": True, "calls": 0}
    monkeypatch.delenv(ENV, raising = False)

    real_find_spec = importlib.util.find_spec
    def find_spec(name, *a, **k):
        if name == "flashinfer":
            return object() if state["flashinfer"] else None
        if name == "triton":
            return object()
        return real_find_spec(name, *a, **k)
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(hip = None, cuda = None)
    monkeypatch.setitem(sys.modules, "torch", torch)

    compiler = types.ModuleType("triton.backends.nvidia.compiler")
    def get_ptxas(arch):
        assert arch == 100
        path = os.environ.get(ENV) or state["bundled"]
        return _Tool(path, state["system_ver"] if path == state["system"] else state["bundled_ver"])
    for name in ("triton", "triton.backends", "triton.backends.nvidia"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "triton.backends.nvidia.compiler", compiler)

    monkeypatch.setattr(shutil, "which", lambda name: state["system"] if name == "ptxas" and state["system"] else None)

    def version_text(cmd):
        state["calls"] += 1
        assert cmd[0] == state["system"] and cmd[1:] == ["--version"]
        return f"Cuda compilation tools, release {state['system_ver']}, V{state['system_ver']}.0\n"
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: types.SimpleNamespace(stdout = version_text(cmd), stderr = ""))
    monkeypatch.setattr(subprocess, "check_output", lambda cmd, **k: version_text(cmd).encode())

    def apply():
        torch.version.hip, torch.version.cuda = state["hip"], state["cuda"]
        if state["get_ptxas"]:
            compiler.get_ptxas = get_ptxas
        elif hasattr(compiler, "get_ptxas"):
            del compiler.get_ptxas
    state["apply"] = apply
    return state


def _run(host, pin):
    host["apply"]()
    got = tp.keep_newest_blackwell_ptxas() if pin else None
    _flashinfer_rule()
    return got, os.environ.get(ENV)


def test_flip_older_system_ptxas(host):
    # Without the pin flashinfer swaps to the older system ptxas; with it Triton keeps the bundled one.
    _, without = _run(host, pin = False)
    assert without == host["system"]
    os.environ.pop(ENV)
    got, with_pin = _run(host, pin = True)
    assert got == with_pin == host["bundled"]


@pytest.mark.parametrize("system_ver", ["13.3", "13.4", "14.0"])
def test_newer_or_equal_system_left_alone(host, system_ver):
    host["system_ver"] = system_ver
    got, final = _run(host, pin = True)
    assert got is None and final == host["system"]


def test_cuda12_system_left_alone(host):
    host["system_ver"] = "12.8"
    got, final = _run(host, pin = True)
    assert got is None and final is None


def test_user_value_never_overridden(host, monkeypatch):
    monkeypatch.setenv(ENV, "/opt/my/ptxas")
    got, final = _run(host, pin = True)
    assert got is None and final == "/opt/my/ptxas" and host["calls"] == 0


@pytest.mark.parametrize("knob", ["flashinfer", "hip", "cuda", "get_ptxas", "system"])
def test_noop_paths(host, knob):
    host.update({"flashinfer": False} if knob == "flashinfer" else {"hip": "6.4"} if knob == "hip" else
                {"cuda": None} if knob == "cuda" else {"get_ptxas": False} if knob == "get_ptxas" else {"system": None})
    host["apply"]()
    assert tp.keep_newest_blackwell_ptxas() is None
    assert os.environ.get(ENV) is None


def test_never_raises(host, monkeypatch):
    host["apply"]()
    def boom(*a, **k):
        raise OSError("no exec")
    monkeypatch.setattr(subprocess, "run", boom)
    assert tp.keep_newest_blackwell_ptxas() is None
    assert os.environ.get(ENV) is None


def test_tuple_return_older_triton_is_noop(host):
    host["apply"]()
    sys.modules["triton.backends.nvidia.compiler"].get_ptxas = lambda arch: ("/x/ptxas", "13.3")
    assert tp.keep_newest_blackwell_ptxas() is None
    assert os.environ.get(ENV) is None


def test_init_calls_it_before_temporary_patches():
    src = (Path(__file__).resolve().parents[1] / "unsloth_zoo" / "__init__.py").read_text()
    assert src.index("_keep_newest_blackwell_ptxas()") < src.index("from unsloth_zoo.temporary_patches import")

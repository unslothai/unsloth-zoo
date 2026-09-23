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

"""The Windows ROCm torchao stub must read as absent. Only Windows + ROCm installs it, so these
exec the stub classes out of utils.py and install the finder by hand, as the guard there does."""

import ast
import importlib
import importlib.metadata
import inspect
import sys
from pathlib import Path

import pytest

UTILS = (Path(__file__).resolve().parents[1] / "unsloth_zoo"
         / "temporary_patches" / "utils.py")
WANTED = {
    "_ROCmSentinelMeta", "_rocm_make_sentinel", "_rocm_make_torchao_stub",
    "_ROCmTorchaoLoader", "_ROCmTorchaoFinder",
}


def _load_stub():
    tree = ast.parse(UTILS.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if getattr(n, "name", None) in WANTED]
    ns: dict = {}
    exec("from importlib.abc import MetaPathFinder as _MetaPathFinder, Loader as _Loader", ns)
    exec(compile(ast.Module(nodes, []), "<utils>", "exec"), ns)
    missing = WANTED - set(ns)
    assert not missing, f"stub pieces not found in utils.py: {missing}"
    return ns


STUB = _load_stub()


@pytest.fixture
def stubbed_torchao(monkeypatch):
    saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] == "torchao"}
    for name in saved:
        del sys.modules[name]
    monkeypatch.setattr(sys, "meta_path", [STUB["_ROCmTorchaoFinder"]()] + list(sys.meta_path))

    real_version = importlib.metadata.version

    def no_torchao_metadata(name):
        if name.lower().startswith("torchao"):
            raise importlib.metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(importlib.metadata, "version", no_torchao_metadata)
    yield importlib.import_module("torchao")
    for name in [k for k in sys.modules if k.split(".")[0] == "torchao"]:
        del sys.modules[name]
    sys.modules.update(saved)


def test_stub_version_is_below_every_minimum(stubbed_torchao):
    from packaging.version import Version

    version = stubbed_torchao.__version__
    assert isinstance(version, str), version
    assert Version(version) < Version("0.0.1")


def test_stub_dunders_are_real_misses(stubbed_torchao):
    assert getattr(stubbed_torchao, "__file__", None) is None
    assert not hasattr(stubbed_torchao, "__wrapped__")
    sentinel = stubbed_torchao.quantization.Float8Tensor
    assert not hasattr(sentinel, "__wrapped__")
    assert inspect.unwrap(sentinel) is sentinel


def test_a_source_lookup_walks_past_the_stub(stubbed_torchao):
    """torch.library's fake-op registration takes this path; a sentinel __file__ raised TypeError."""
    import torchao.quantization  # noqa: F401  a submodule stub in sys.modules as well

    ns: dict = {}
    exec(compile("def probe():\n    pass\n", "/nowhere/zoo_rocm_stub_probe.py", "exec"), ns)
    assert inspect.getsourcefile(ns["probe"].__code__) is None


def test_stub_still_answers_the_names_callers_import(stubbed_torchao):
    from torchao.quantization import Float8Tensor, quantize_  # noqa: F401

    assert not isinstance(object(), stubbed_torchao.dtypes.AffineQuantizedTensor)
    assert stubbed_torchao.dtypes.AffineQuantizedTensor.child is not None


def test_transformers_reads_the_stub_as_unavailable(stubbed_torchao):
    # Without torch (Apple Silicon) the probe answers False before reading torchao: vacuous.
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from packaging.version import Version

    if Version(transformers.__version__).major < 5:
        pytest.skip("transformers 4.x reads torchao once, at its own import")
    import_utils = importlib.import_module("transformers.utils.import_utils")
    probe = import_utils.is_torchao_available
    clear = getattr(probe, "cache_clear", lambda: None)
    clear()
    try:
        assert probe() is False
    finally:
        clear()


_FRESH = r'''
import ast, importlib, importlib.metadata as md, sys
tree = ast.parse(open(sys.argv[1], encoding="utf-8").read())
nodes = [n for n in tree.body if getattr(n, "name", None) in set(sys.argv[2:])]
ns = {}
exec("from importlib.abc import MetaPathFinder as _MetaPathFinder, Loader as _Loader", ns)
exec(compile(ast.Module(nodes, []), "<utils>", "exec"), ns)
real = md.version
def version(name):
    if name.lower().startswith("torchao"):
        raise md.PackageNotFoundError(name)
    return real(name)
md.version = version
sys.meta_path.insert(0, ns["_ROCmTorchaoFinder"]())
import transformers.modeling_utils
print("OK")
'''


def test_transformers_modeling_utils_imports_against_the_stub():
    """Fresh interpreter: an earlier import would hide the failure."""
    import subprocess

    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    out = subprocess.run(
        [sys.executable, "-c", _FRESH, str(UTILS), *sorted(WANTED)],
        capture_output = True, text = True, timeout = 600,
    )
    assert out.returncode == 0 and out.stdout.strip().endswith("OK"), out.stderr[-2000:]

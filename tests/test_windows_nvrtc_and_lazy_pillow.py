"""Windows NVRTC builtins lookup and a pillow-free import of the vision utilities."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from unsloth_zoo.windows_nvrtc import expose_torch_lib_to_nvrtc

ROOT = Path(__file__).resolve().parents[1]


def _torch_lib(tmp_path, dlls = ("nvrtc-builtins64_135.dll",)):
    lib = tmp_path / "torch" / "lib"
    lib.mkdir(parents = True)
    for name in dlls:
        (lib / name).write_bytes(b"")
    return str(lib)


def test_unloadable_builtins_put_torch_lib_first_on_path(tmp_path):
    lib = _torch_lib(tmp_path)
    env = {"PATH": r"C:\Windows\system32"}
    tried = []

    def loads(name):
        tried.append(name)
        return False

    assert expose_torch_lib_to_nvrtc("win32", lambda: lib, loads, env) is True
    assert env["PATH"].split(os.pathsep)[0] == lib
    assert env["PATH"].endswith(r"C:\Windows\system32")
    assert tried == ["nvrtc-builtins64_135.dll"]


def test_loadable_builtins_leave_path_alone(tmp_path):
    lib = _torch_lib(tmp_path)
    env = {"PATH": "x"}
    assert expose_torch_lib_to_nvrtc("win32", lambda: lib, lambda name: True, env) is False
    assert env["PATH"] == "x"


def test_torch_lib_already_on_path_is_not_added_twice(tmp_path):
    lib = _torch_lib(tmp_path)
    env = {"PATH": os.pathsep.join(["a", lib + os.sep])}
    assert expose_torch_lib_to_nvrtc("win32", lambda: lib, lambda name: False, env) is False
    assert env["PATH"] == os.pathsep.join(["a", lib + os.sep])


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_other_platforms_are_untouched(tmp_path, platform):
    lib = _torch_lib(tmp_path)
    env = {"PATH": "x"}
    assert expose_torch_lib_to_nvrtc(platform, lambda: lib, lambda name: False, env) is False
    assert env["PATH"] == "x"


def test_no_builtins_dll_or_no_torch_changes_nothing(tmp_path):
    lib = _torch_lib(tmp_path, dlls = ("nvrtc64_130_0.dll",))
    env = {"PATH": "x"}
    assert expose_torch_lib_to_nvrtc("win32", lambda: lib, lambda name: False, env) is False
    assert expose_torch_lib_to_nvrtc("win32", lambda: None, lambda name: False, env) is False
    assert env["PATH"] == "x"


def test_a_failing_check_never_raises(tmp_path):
    lib = _torch_lib(tmp_path)
    env = {"PATH": "x"}

    def boom(name):
        raise RuntimeError("loader exploded")

    assert expose_torch_lib_to_nvrtc("win32", lambda: lib, boom, env) is False
    assert env["PATH"] == "x"


def test_package_import_runs_it_before_torch():
    src = (ROOT / "unsloth_zoo" / "__init__.py").read_text(encoding = "utf-8")
    call = src.index("_expose_torch_lib_to_nvrtc()")
    first_torch = min(
        i for i in (src.find("\nimport torch"), src.find("import torch as _torch")) if i >= 0
    )
    assert call < first_torch


def test_vision_utils_imports_without_pillow():
    # PIL blocked: the module still imports, and only an image operation names the missing package.
    code = textwrap.dedent(
        """
        import sys
        sys.modules["PIL"] = None
        sys.modules["PIL.Image"] = None
        sys.modules["PIL.ImageOps"] = None
        import unsloth_zoo.vision_utils as v
        assert v.LANCZOS is None
        try:
            v.Image.open
        except ImportError as e:
            assert "pip install pillow" in str(e), e
        else:
            raise SystemExit("expected ImportError")
        try:
            v._decode_image(b"x")
        except ImportError as e:
            assert "pillow" in str(e)
        else:
            raise SystemExit("expected ImportError")
        print("OK")
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd = ROOT,
        env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        capture_output = True,
        text = True,
        timeout = 600,
    )
    assert out.returncode == 0 and out.stdout.strip().endswith("OK"), out.stderr[-3000:]

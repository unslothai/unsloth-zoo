#!/usr/bin/env python3
"""Manual integration verification for unslothai/unsloth#8933."""

from __future__ import annotations

import os

os.environ.setdefault("UNSLOTH_ZOO_DISABLE_GPU_INIT", "1")

import subprocess
import sys
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _banner(title: str) -> None:
    print(f"\n=== {title} ===")


def test_old_transformers_fails_without_stub() -> bool:
    _banner("1. Studio finegrained_fp8 bind WITHOUT stub (expect fail)")
    import torch

    if hasattr(torch, "float8_e8m0fnu"):
        print("SKIP: need torch < 2.7")
        return True
    try:
        ns = {"torch": torch}
        exec("_UE8M0_SF_DTYPE = torch.float8_e8m0fnu", ns)  # noqa: S102
        print("UNEXPECTED: bind succeeded without stub")
        return False
    except AttributeError as exc:
        if "float8_e8m0fnu" in str(exc):
            print("OK: failed as expected:", exc)
            return True
        print("FAIL: wrong AttributeError:", exc)
        return False


def test_old_transformers_succeeds_with_stub() -> bool:
    _banner("2. Studio finegrained_fp8 bind WITH temporary stub (expect pass)")
    import ast
    import torch

    if hasattr(torch, "float8_e8m0fnu"):
        print("SKIP: need torch < 2.7")
        return True

    utils_path = os.path.join(ROOT, "unsloth_zoo", "temporary_patches", "utils.py")
    tree = ast.parse(open(utils_path, encoding="utf-8").read())
    ns = {
        "torch": torch,
        "contextlib": __import__("contextlib"),
        "_E8M0_IMPORT_STUB_ACTIVE": False,
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "torch_supports_float8_e8m0fnu",
            "require_native_float8_e8m0fnu",
            "_temporary_float8_e8m0fnu_import_stub",
        ):
            exec(compile(ast.Module([node], []), "<utils>", "exec"), ns)
    if "require_native_float8_e8m0fnu" not in ns:
        print("FAIL: could not load require_native_float8_e8m0fnu from utils.py")
        return False
    temporary_stub = ns["_temporary_float8_e8m0fnu_import_stub"]
    require_native = ns["require_native_float8_e8m0fnu"]

    try:
        with temporary_stub() as used:
            assert used is True
            bind_ns = {"torch": torch}
            exec("_UE8M0_SF_DTYPE = torch.float8_e8m0fnu", bind_ns)  # noqa: S102
            assert bind_ns["_UE8M0_SF_DTYPE"] is torch.float8_e4m3fn
        assert not hasattr(torch, "float8_e8m0fnu")
        try:
            require_native()
            print("FAIL: require_native should raise after stub exits")
            return False
        except RuntimeError as exc:
            if "PyTorch >= 2.7" not in str(exc):
                print("FAIL: unexpected error:", exc)
                return False
        print("OK: temporary bind works; UE8M0 rejected after import")
        return True
    except Exception as exc:
        print("FAIL:", exc)
        traceback.print_exc()
        return False


def test_temporary_patches_utils_import() -> bool:
    _banner("3. unsloth_zoo.temporary_patches.utils import on torch 2.6")
    venv_py = os.path.join(ROOT, ".test-e8m0-venv", "bin", "python")
    if not os.path.isfile(venv_py):
        print("SKIP: .test-e8m0-venv not found")
        return True

    code = r'''
import os, sys, types, importlib.util
os.environ["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
root = os.environ["UNSLOTH_ZOO_ROOT"]
sys.path.insert(0, root)
import torch
pkg = "unsloth_zoo"
pkg_path = os.path.join(root, pkg)
pkg_mod = types.ModuleType(pkg)
pkg_mod.__path__ = [pkg_path]
pkg_mod.__package__ = pkg
sys.modules[pkg] = pkg_mod
for prereq in ("utils",):
    full = f"{pkg}.{prereq}"
    path = os.path.join(pkg_path, f"{prereq}.py")
    spec = importlib.util.spec_from_file_location(full, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
dt_path = os.path.join(pkg_path, "device_type.py")
dt_spec = importlib.util.spec_from_file_location(f"{pkg}.device_type", dt_path)
dt_mod = importlib.util.module_from_spec(dt_spec)
sys.modules[f"{pkg}.device_type"] = dt_mod
_orig = torch.cuda.is_available
torch.cuda.is_available = lambda: True
try:
    dt_spec.loader.exec_module(dt_mod)
finally:
    torch.cuda.is_available = _orig
utils_path = os.path.join(pkg_path, "temporary_patches", "utils.py")
utils_spec = importlib.util.spec_from_file_location(
    "unsloth_zoo.temporary_patches.utils", utils_path,
    submodule_search_locations=[os.path.join(pkg_path, "temporary_patches")],
)
utils_mod = importlib.util.module_from_spec(utils_spec)
sys.modules["unsloth_zoo.temporary_patches.utils"] = utils_mod
utils_spec.loader.exec_module(utils_mod)
assert not utils_mod.torch_supports_float8_e8m0fnu()
assert hasattr(utils_mod, "Unpack")
print("ok")
'''
    env = {**os.environ, "UNSLOTH_ZOO_ROOT": ROOT, "UNSLOTH_ZOO_DISABLE_GPU_INIT": "1"}
    result = subprocess.run([venv_py, "-c", code], env=env, capture_output=True, text=True)
    if result.returncode == 0 and "ok" in result.stdout:
        print("OK: temporary_patches.utils imported with Unpack")
        return True
    print("FAIL:", result.stderr or result.stdout)
    return False


def test_ue8m0_on_capable_torch() -> bool:
    _banner("4. UE8M0 dtype available on torch 2.12+ (Studio FP8 path)")
    py = "/home/ubuntu/workspace/unsloth/.venv/bin/python"
    if not os.path.isfile(py):
        print("SKIP: unsloth venv not available")
        return True
    code = """
import torch
from transformers.integrations import finegrained_fp8 as fg
assert hasattr(torch, "float8_e8m0fnu")
getter = getattr(fg, "_get_ue8m0_dtype", None)
if getter is not None:
    assert getter() is torch.float8_e8m0fnu
    print("ok getter")
elif hasattr(fg, "_UE8M0_SF_DTYPE"):
    assert fg._UE8M0_SF_DTYPE is torch.float8_e8m0fnu
    print("ok module bind")
else:
    print("skip no ue8m0 hook")
"""
    env = {**os.environ}
    result = subprocess.run([py, "-c", code], env=env, capture_output=True, text=True)
    if result.returncode == 0:
        out = (result.stdout or "").strip()
        print("OK:", out or "passed")
        return True
    print("FAIL:", result.stderr or result.stdout)
    return False


def main() -> int:
    sys.path.insert(0, ROOT)
    import torch

    print("torch", torch.__version__)
    print("e8m0 native", hasattr(torch, "float8_e8m0fnu"))

    results = [
        test_old_transformers_fails_without_stub(),
        test_old_transformers_succeeds_with_stub(),
        test_temporary_patches_utils_import(),
        test_ue8m0_on_capable_torch(),
    ]
    passed = sum(results)
    print(f"\n=== SUMMARY: {passed}/{len(results)} passed ===")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

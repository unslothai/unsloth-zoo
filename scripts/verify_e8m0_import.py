#!/usr/bin/env python3
"""Manual integration verification for unslothai/unsloth#8933."""

from __future__ import annotations

import os
import sys
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STUDIO_TRANSFORMERS_ROOT = os.path.expanduser(
    "~/.unsloth/studio/.venv_t5_510"
)


def _banner(title: str) -> None:
    print(f"\n=== {title} ===")


def test_old_transformers_fails_without_stub() -> bool:
    _banner("1. Studio finegrained_fp8 bind WITHOUT stub (expect fail)")
    import torch

    assert not hasattr(torch, "float8_e8m0fnu"), "need torch < 2.7 for this check"
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


def _load_ensure_helper():
    import ast

    path = os.path.join(ROOT, "unsloth_zoo", "temporary_patches", "utils.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_ensure_torch_float8_e8m0fnu":
            ns = {"torch": __import__("torch")}
            exec(compile(ast.Module([node], []), "<utils>", "exec"), ns)
            return ns["_ensure_torch_float8_e8m0fnu"]
    raise RuntimeError("helper not found")


def test_old_transformers_succeeds_with_stub() -> bool:
    _banner("2. Studio finegrained_fp8 bind WITH stub (expect pass)")
    import torch

    ensure = _load_ensure_helper()
    ensure()
    assert torch.float8_e8m0fnu is torch.float8_e4m3fn

    try:
        ns = {"torch": torch}
        exec("_UE8M0_SF_DTYPE = torch.float8_e8m0fnu", ns)  # noqa: S102
        assert ns["_UE8M0_SF_DTYPE"] is torch.float8_e4m3fn
        print("OK: Studio bind line succeeds after stub")
        return True
    except Exception as exc:
        print("FAIL:", exc)
        traceback.print_exc()
        return False


def test_temporary_patches_utils_import() -> bool:
    _banner("3. unsloth_zoo.temporary_patches.utils import on torch 2.6")
    import subprocess

    code = r'''
import os, sys, types
os.environ["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
root = os.environ["UNSLOTH_ZOO_ROOT"]
sys.path.insert(0, root)

import torch
assert not hasattr(torch, "float8_e8m0fnu")
assert hasattr(torch, "float8_e4m3fn")

# conftest-style device_type preload
import importlib.util
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

from unsloth_zoo.temporary_patches.utils import Unpack
assert torch.float8_e8m0fnu is torch.float8_e4m3fn
print("ok")
'''
    env = dict(os.environ)
    env["UNSLOTH_ZOO_ROOT"] = ROOT
    # Load utils.py directly (same entry point as the Studio crash) without
    # pulling every temporary_patches submodule.
    code = code.replace(
        "from unsloth_zoo.temporary_patches import utils as tp_utils\n"
        "assert torch.float8_e8m0fnu is torch.float8_e4m3fn\n"
        "assert hasattr(tp_utils, \"Unpack\")",
        "utils_path = os.path.join(pkg_path, \"temporary_patches\", \"utils.py\")\n"
        "utils_spec = importlib.util.spec_from_file_location(\n"
        "    \"unsloth_zoo.temporary_patches.utils\", utils_path,\n"
        "    submodule_search_locations=[os.path.join(pkg_path, \"temporary_patches\")],\n"
        ")\n"
        "utils_mod = importlib.util.module_from_spec(utils_spec)\n"
        "sys.modules[\"unsloth_zoo.temporary_patches.utils\"] = utils_mod\n"
        "utils_spec.loader.exec_module(utils_mod)\n"
        "assert torch.float8_e8m0fnu is torch.float8_e4m3fn\n"
        "assert hasattr(utils_mod, \"Unpack\")",
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and "ok" in result.stdout:
        print("OK: temporary_patches.utils imported with Unpack")
        return True
    print("FAIL:", result.stderr or result.stdout)
    return False


def test_ue8m0_on_capable_torch() -> bool:
    _banner("4. UE8M0 dtype available on torch 2.12+ (Studio FP8 path)")
    import subprocess

    code = """
import torch
from transformers.integrations import finegrained_fp8 as fg
if hasattr(torch, "float8_e8m0fnu"):
    getter = getattr(fg, "_get_ue8m0_dtype", None)
    if getter is not None:
        dtype = getter()
        assert dtype is torch.float8_e8m0fnu
        print("ok getter")
    elif hasattr(fg, "_UE8M0_SF_DTYPE"):
        assert fg._UE8M0_SF_DTYPE is torch.float8_e8m0fnu
        print("ok module bind")
    else:
        print("skip no ue8m0 hook")
else:
    raise RuntimeError("torch lacks float8_e8m0fnu on a build that should have it")
"""
    py = "/home/ubuntu/workspace/unsloth/.venv/bin/python"
    if not os.path.isfile(py):
        print("SKIP: unsloth venv not available")
        return True
    result = subprocess.run([py, "-c", code], capture_output=True, text=True)
    if result.returncode == 0:
        print("OK:", (result.stdout or "").strip())
        return True
    print("FAIL:", result.stderr or result.stdout)
    return False


def main() -> int:
    os.environ.setdefault("UNSLOTH_ZOO_DISABLE_GPU_INIT", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch

    print("torch", torch.__version__)
    print("e8m0", hasattr(torch, "float8_e8m0fnu"))
    print("e4m3", hasattr(torch, "float8_e4m3fn"))
    print("studio transformers root:", STUDIO_TRANSFORMERS_ROOT)

    results = [
        test_old_transformers_fails_without_stub(),
        test_old_transformers_succeeds_with_stub(),
        test_temporary_patches_utils_import(),
    ]

    # UE8M0-capable torch check uses the workspace torch 2.12 venv.
    results.append(test_ue8m0_on_capable_torch())

    passed = sum(results)
    total = len(results)
    print(f"\n=== SUMMARY: {passed}/{total} passed ===")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

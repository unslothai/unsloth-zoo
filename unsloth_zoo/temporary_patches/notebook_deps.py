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

"""One-shot pip install of allow-listed optional deps that transformers asks for
(requires_backends, trust_remote_code check_imports / submodule imports).
Opt out with UNSLOTH_AUTO_INSTALL=0; skipped under UNSLOTH_OFFLINE / HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE."""

import functools
import importlib
import importlib.util
import os
import shutil
import site
import subprocess
import sys

from ..log import logger

__all__ = []

# pypi name -> import name
_ALLOW_LIST = {
    "timm": "timm",
    "addict": "addict",
    "einops": "einops",
    "easydict": "easydict",
    "snac": "snac",
    "torchcodec": "torchcodec",
    "matplotlib": "matplotlib",
    "soundfile": "soundfile",
    "librosa": "librosa",
    "scipy": "scipy",
    "pyctcdecode": "pyctcdecode",
    "tiktoken": "tiktoken",
    "blobfile": "blobfile",
    "pillow_heif": "pillow_heif",
    "decord": "decord",
    "av": "av",
    "num2words": "num2words",
    "jieba": "jieba",
    "sentencepiece": "sentencepiece",
}
_BY_IMPORT_NAME = {v: k for k, v in _ALLOW_LIST.items()}
_TRUE = frozenset({"1", "ON", "TRUE", "YES"})
_attempted = set()


def _env_true(name, default = "0"):
    return os.environ.get(name, default).strip().upper() in _TRUE


def _enabled():
    # Read per attempt, like llama_cpp's UNSLOTH_AUTO_INSTALL.
    if not _env_true("UNSLOTH_AUTO_INSTALL", "1"):
        return False
    return not any(_env_true(v) for v in ("UNSLOTH_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))


def _in_venv():
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix) or bool(os.environ.get("CONDA_PREFIX"))


def _pip_install(pkg):
    if pkg in _attempted:
        return False
    _attempted.add(pkg)
    if shutil.which("uv") and _in_venv():
        # --python: a bare `uv pip install` targets $VIRTUAL_ENV / ./.venv, not necessarily this interpreter.
        cmd = ["uv", "pip", "install", "--quiet", "--python", sys.executable, pkg]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--no-input", pkg]
        if not _in_venv() and hasattr(os, "geteuid") and os.geteuid() != 0:
            try:
                writable = any(os.access(sp, os.W_OK) for sp in site.getsitepackages())
            except Exception:
                writable = False
            if not writable:
                cmd.append("--user")
    logger.warning(
        f"Unsloth: auto-installing missing optional dependency `{pkg}` via `{' '.join(cmd)}`. "
        f"Set UNSLOTH_AUTO_INSTALL=0 to disable."
    )
    try:
        r = subprocess.run(cmd, capture_output = True, text = True, timeout = 300)
    except Exception as e:
        logger.warning(f"Unsloth: auto-install of `{pkg}` failed to launch: {e}")
        return False
    if r.returncode != 0:
        logger.warning(f"Unsloth: auto-install of `{pkg}` failed:\n{(r.stderr or '')[-500:]}")
        return False
    importlib.invalidate_caches()
    return True


def _allowed(name):
    """Allow-listed pypi name for a pypi or (possibly dotted) import name, else None."""
    name = (name or "").strip().split(".")[0]
    return name if name in _ALLOW_LIST else _BY_IMPORT_NAME.get(name)


def _install(name):
    """True if `name` is allow-listed and importable afterwards."""
    pkg = _allowed(name)
    if pkg is None or not _enabled():
        return False
    import_name = _ALLOW_LIST[pkg]
    if importlib.util.find_spec(import_name) is not None:
        return True
    return _pip_install(pkg) and importlib.util.find_spec(import_name) is not None


def _rebind_guarded_imports(obj, backend):
    """Replay `if is_<backend>_available(): import ...` blocks in obj's model package: they ran
    while the dep was missing (modeling_timm_wrapper's `import timm`), so the names are unbound."""
    import ast
    import inspect
    cls = obj if isinstance(obj, type) else type(obj)
    packages = {c.__module__.rpartition(".")[0] for c in cls.__mro__ if c.__module__.startswith("transformers.models.")}
    guard = f"is_{backend}_available"
    for name, module in list(sys.modules.items()):
        if module is None or name.rpartition(".")[0] not in packages:
            continue
        try:
            tree = ast.parse(inspect.getsource(module))
        except Exception:
            continue
        for node in tree.body:
            if not (isinstance(node, ast.If) and isinstance(node.test, ast.Call)
                    and getattr(node.test.func, "id", None) == guard):
                continue
            for stmt in node.body:
                try:
                    if isinstance(stmt, ast.Import):
                        for a in stmt.names:
                            top = importlib.import_module(a.name if a.asname else a.name.split(".")[0])
                            if a.asname:
                                setattr(module, a.asname, top)
                            else:
                                importlib.import_module(a.name)
                                setattr(module, a.name.split(".")[0], top)
                    elif isinstance(stmt, ast.ImportFrom) and stmt.level == 0 and stmt.module:
                        src = importlib.import_module(stmt.module)
                        for a in stmt.names:
                            value = getattr(src, a.name, None)
                            if value is None:
                                value = importlib.import_module(f"{stmt.module}.{a.name}")
                            setattr(module, a.asname or a.name, value)
                except Exception as e:
                    logger.info(f"Unsloth: could not rebind `{backend}` imports in {name} ({e})")


def patch_requires_backends_autoinstall():
    try:
        import transformers.utils as tu
        from transformers.utils import import_utils as iu
    except Exception:
        return
    original = iu.requires_backends
    if getattr(original, "_unsloth_patched", False):
        return

    @functools.wraps(original)
    def requires_backends(obj, backends):
        try:
            return original(obj, backends)
        except ImportError:
            wanted = backends if isinstance(backends, (list, tuple)) else [backends]
            wanted = [b for b in wanted if isinstance(b, str) and b in _ALLOW_LIST]
            if not wanted or not any([_install(b) for b in wanted]):
                raise
            # is_<backend>_available is lru_cached with the pre-install False.
            for b in wanted:
                check = getattr(iu, "BACKENDS_MAPPING", {}).get(b, (None,))[0]
                if hasattr(check, "cache_clear"):
                    check.cache_clear()
                _rebind_guarded_imports(obj, b)
            return original(obj, backends)

    requires_backends._unsloth_patched = True
    # Model files do `from ...utils import requires_backends`, binding the name per module,
    # so rebinding import_utils alone never reaches them (TimmWrapperModel for Gemma 3n).
    for name, module in list(sys.modules.items()):
        # __dict__, not getattr: getattr on a transformers _LazyModule imports submodules (~5 s).
        if (name == "transformers" or name.startswith("transformers.")) and \
                getattr(module, "__dict__", {}).get("requires_backends") is original:
            try:
                setattr(module, "requires_backends", requires_backends)
            except Exception:
                pass
    iu.requires_backends = requires_backends
    tu.requires_backends = requires_backends


def patch_check_imports_autoinstall():
    try:
        from transformers import dynamic_module_utils as dmu
    except Exception:
        return
    original_check = dmu.check_imports
    if getattr(original_check, "_unsloth_patched", False):
        return

    @functools.wraps(original_check)
    def check_imports(filename):
        try:
            return original_check(filename)
        except ImportError as e:
            # "... not found in your environment: pkg1, pkg2. Run `pip install pkg1 pkg2`"
            msg = str(e)
            if "environment:" not in msg:
                raise
            names = [p.strip() for p in msg.split("environment:", 1)[1].split(".", 1)[0].split(",")]
            if not names or not all(_allowed(p) for p in names) or not all([_install(p) for p in names]):
                raise
            return original_check(filename)

    check_imports._unsloth_patched = True
    dmu.check_imports = check_imports

    # check_imports only scans the top-level file; sibling files it imports
    # (DeepSeek-OCR deepencoder.py -> easydict) fail inside get_class_in_module.
    original_get = getattr(dmu, "get_class_in_module", None)
    if original_get is None:
        return

    @functools.wraps(original_get)
    def get_class_in_module(*args, **kwargs):
        for _ in range(len(_ALLOW_LIST)):
            try:
                return original_get(*args, **kwargs)
            except ModuleNotFoundError as e:
                if not _install(getattr(e, "name", None)):
                    raise
        return original_get(*args, **kwargs)

    get_class_in_module._unsloth_patched = True
    dmu.get_class_in_module = get_class_in_module


patch_requires_backends_autoinstall()
patch_check_imports_autoinstall()

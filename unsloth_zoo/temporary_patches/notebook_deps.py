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

# Auto-install missing notebook-only deps: three call sites raise (requires_backends,
# check_imports, and a bare ModuleNotFoundError from the IPython chain), so all three
# are wrapped with an allow-listed pip retry.

import ast
import importlib
import importlib.metadata
import importlib.util
import os
import shutil
import site
import subprocess
import sys

# Absolute on purpose: `from ..log` makes transformers' custom_object_save write
# temporary_patches/.log.py and crash. See PR #1045.
from unsloth_zoo.log import logger
from .common import TEMPORARY_PATCHES

# pypi-name -> import-name (None means same).
_ALLOW_LIST = {
    "timm":          None,
    "addict":        None,
    "einops":        None,
    "easydict":      None,
    "snac":          None,
    "torchcodec":    None,
    "matplotlib":    None,
    "traitlets":     None,
    "soundfile":     None,
    "librosa":       None,
    "scipy":         None,
    "pyctcdecode":   None,
    "tiktoken":      None,
    "blobfile":      None,
    "pillow_heif":   "pillow_heif",
    "decord":        None,
    "av":            "av",
    "num2words":     None,
    "jieba":         None,
    "sentencepiece": None,
}

# huggingface_hub.constants.ENV_VARS_TRUE_VALUES: the hub owns HF_HUB_OFFLINE and
# TRANSFORMERS_OFFLINE, so accepting only "1" would run pip against `=true`.
_TRUE_VALUES = frozenset({"1", "ON", "TRUE", "YES"})


def _env_is_true(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().upper() in _TRUE_VALUES


def _auto_install_enabled() -> bool:
    # Read at the attempt, not at import: `import unsloth` imports this module, so a
    # user acting on the _run_install warning sets the variable after a constant froze.
    return _env_is_true("UNSLOTH_AUTO_INSTALL", "1")


def _no_network() -> bool:
    # Read at the attempt (a notebook can go offline after importing unsloth). Only the
    # two flags huggingface_hub itself honours; HF_DATASETS_OFFLINE is deliberately
    # absent because unsloth_zoo/__init__.py cross-syncs it into these two.
    return (
        _env_is_true("UNSLOTH_OFFLINE")
        or _env_is_true("HF_HUB_OFFLINE")
        or _env_is_true("TRANSFORMERS_OFFLINE")
    )
_attempted: set = set()


def _is_running_prefix(root: str) -> bool:
    # samefile, not string compare: symlinks, and raise-on-missing is the right
    # answer for a variable left over from a deleted environment.
    try:
        return os.path.samefile(root, sys.prefix)
    except Exception:
        return False


def _in_venv() -> bool:
    # From the RUNNING interpreter, never an inherited activation variable alone: a
    # kernel runs A with VIRTUAL_ENV from B, and trusting B skips the --user fallback.
    if hasattr(sys, "real_prefix"):
        return True
    if getattr(sys, "base_prefix", sys.prefix) != sys.prefix:
        return True
    # conda reports `base_prefix == prefix`, so the variable is its only marker.
    return any(
        bool(root) and _is_running_prefix(root)
        for root in (os.environ.get("VIRTUAL_ENV"), os.environ.get("CONDA_PREFIX"))
    )


def _uv_command(pkg: str) -> list:
    # `--python` is required: uv otherwise targets VIRTUAL_ENV/CONDA_PREFIX/a .venv.
    return ["uv", "pip", "install", "--quiet", "--python", sys.executable, pkg]


def _pip_command(pkg: str) -> list:
    cmd = [
        sys.executable, "-m", "pip", "install", "--quiet",
        "--disable-pip-version-check", "--no-input", pkg,
    ]
    if not _in_venv() and hasattr(os, "geteuid") and os.geteuid() != 0:
        try:
            sp = site.getsitepackages()[0]
            probe = os.path.join(sp, ".unsloth_write_probe")
            open(probe, "w").close()
            os.remove(probe)
        except Exception:
            cmd.append("--user")
    return cmd


def _run_install(pkg: str, cmd: list) -> tuple:
    """Run one installer command. Returns ``(succeeded, retry_with_pip)``."""
    logger.warning(
        f"Unsloth: auto-installing missing notebook dep `{pkg}` via "
        f"`{' '.join(cmd)}`. Set UNSLOTH_AUTO_INSTALL=0 to disable."
    )
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except Exception as e:
        logger.warning(f"Unsloth: auto-install of `{pkg}` failed to launch: {e}")
        return False, False
    if r.returncode == 0:
        importlib.invalidate_caches()
        try:
            list(importlib.metadata.distributions())
        except Exception:
            pass
        return True, False
    stderr = r.stderr or ""
    logger.warning(f"Unsloth: auto-install of `{pkg}` failed:\n{stderr[-500:]}")
    # Retry via pip only when uv could not be aimed at this interpreter at all; a real
    # resolution or build failure must NOT be, or every failure costs two installs.
    lowered = stderr.lower()
    retry_with_pip = (
        "unexpected argument" in lowered
        or "unrecognized" in lowered
        or "no virtual environment or system python installation found" in lowered
    )
    return False, retry_with_pip


def _pip_install(pkg: str) -> bool:
    if pkg in _attempted:
        return False
    _attempted.add(pkg)
    if shutil.which("uv") and _in_venv():
        ok, retry_with_pip = _run_install(pkg, _uv_command(pkg))
        if ok:
            return True
        if not retry_with_pip:
            return False
    return _run_install(pkg, _pip_command(pkg))[0]


def _importable(import_name: str) -> bool:
    """Whether ``import_name`` imports, not merely resolves: find_spec passes a broken
    native extension, and calling that a success downgrades ImportError to NameError."""
    try:
        importlib.import_module(import_name)
    except Exception:
        return False
    return True


def _try_install_and_import(pkg: str) -> bool:
    if pkg not in _ALLOW_LIST:
        return False
    if not _auto_install_enabled() or _no_network():
        return False
    import_name = _ALLOW_LIST[pkg] or pkg.replace("-", "_")
    if importlib.util.find_spec(import_name) is not None and _importable(import_name):
        return True
    if not _pip_install(pkg):
        return False
    return _importable(import_name)


def _rebind_requires_backends(wrapper, original) -> None:
    """Point every alias of ``requires_backends`` at the wrapper: modeling files do
    ``from ...utils import requires_backends``, binding their own name, so patching
    only ``import_utils`` leaves the installer unreachable. ``vars()`` not
    ``getattr``, or lazy module shims import submodules as a side effect."""
    if original is None or wrapper is None:
        return
    for module in list(sys.modules.values()):
        try:
            namespace = vars(module)
            if namespace.get("requires_backends", None) is original:
                namespace["requires_backends"] = wrapper
        except Exception:
            continue


def _refresh_backend_availability(iu, backend) -> None:
    """Make transformers re-evaluate ``backend``: 5.x caches the probe behind
    ``lru_cache`` (already ``False``), 4.x used a ``_<backend>_available`` flag.
    Both are handled and neither is required to be present."""
    flag = f"_{backend.replace('-', '_')}_available"
    if hasattr(iu, flag):
        setattr(iu, flag, True)
    try:
        available = iu.BACKENDS_MAPPING[backend][0]
    except Exception:
        return
    cache_clear = getattr(available, "cache_clear", None)
    if cache_clear is None:
        return
    try:
        cache_clear()
    except Exception:
        pass


def _names_bound_by(statement) -> list:
    if isinstance(statement, ast.Import):
        return [alias.asname or alias.name.split(".")[0] for alias in statement.names]
    return [alias.asname or alias.name for alias in statement.names]


def _perform_import(statement, module) -> None:
    """Carry out one parsed import statement in ``module``'s namespace. importlib, not
    ``exec(compile(...))``, so the replay cannot run anything else in the file."""
    namespace = vars(module)
    if isinstance(statement, ast.Import):
        for alias in statement.names:
            importlib.import_module(alias.name)
            if alias.asname:
                namespace[alias.asname] = sys.modules[alias.name]
            else:
                top = alias.name.split(".")[0]
                namespace[top] = sys.modules[top]
        return
    name = "." * statement.level + (statement.module or "")
    source = importlib.import_module(
        name, package = getattr(module, "__package__", None)
    )
    for alias in statement.names:
        try:
            value = getattr(source, alias.name)
        except AttributeError:
            # Re-raise the attribute error, not the submodule one: "cannot import name
            # 'X' from 'timm.data'" beats "No module named 'timm.data.X'".
            try:
                value = importlib.import_module(f"{source.__name__}.{alias.name}")
            except ImportError:
                raise ImportError(
                    f"cannot import name {alias.name!r} from {source.__name__!r}"
                ) from None
        namespace[alias.asname or alias.name] = value


def _statement_imports(statement, import_name) -> bool:
    """Whether `statement` imports `import_name` itself, not a sibling."""
    if isinstance(statement, ast.Import):
        return any(
            alias.name == import_name or alias.name.startswith(import_name + ".")
            for alias in statement.names
        )
    return statement.level == 0 and (statement.module or "").split(".")[0] == import_name


def _skipped_import_statements(tree, guard, import_name) -> list:
    """Top-level import statements conditional on this backend. Two shapes: the ``if
    is_<backend>_available():`` guard leaves the name absent, while ``try: import x /
    except ImportError: x = None`` BINDS it to None (see ``_needs_rebinding``). Negated
    and compound guards are skipped rather than guessed at."""
    out = []
    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            if not (
                isinstance(test, ast.Call)
                and isinstance(test.func, ast.Name)
                and test.func.id == guard
                and not test.args
                and not test.keywords
            ):
                continue
            out.extend(
                statement for statement in node.body
                if isinstance(statement, (ast.Import, ast.ImportFrom))
            )
        elif isinstance(node, ast.Try):
            # Only an import-guard try; a body with real logic must not be re-run.
            if not node.body or not all(
                isinstance(statement, (ast.Import, ast.ImportFrom))
                for statement in node.body
            ):
                continue
            out.extend(
                statement for statement in node.body
                if _statement_imports(statement, import_name)
            )
    return out


def _needs_rebinding(module, names) -> bool:
    # None counts as missing: `except ImportError: spm = None` binds the name, so
    # hasattr says present and the caller gets AttributeError on None, not NameError.
    return any(getattr(module, each, None) is None for each in names)


def _replay_skipped_guarded_imports(iu, backend) -> bool:
    """Run the guarded import blocks skipped at first import; False if any raised.

    Refreshing the availability flag alone is not enough: the module scope import never
    ran, so ``requires_backends`` starts succeeding and the body dies on a bare
    ``NameError`` instead of the ImportError the install replaced."""
    guard = f"is_{backend.replace('-', '_')}_available"
    import_name = _ALLOW_LIST.get(backend) or backend.replace("-", "_")
    try:
        passes = bool(getattr(iu, guard)())
    except Exception:
        return True
    if not passes:
        return True
    ok = True
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if name != "transformers" and not name.startswith("transformers."):
            continue
        path = getattr(module, "__file__", None)
        if not path or not path.endswith(".py"):
            continue
        try:
            with open(path, encoding = "utf-8") as handle:
                source = handle.read()
        except OSError:
            continue
        if guard not in source and import_name not in source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for statement in _skipped_import_statements(tree, guard, import_name):
            if any(alias.name == "*" for alias in statement.names):
                # No name to check a star import against.
                continue
            bound = _names_bound_by(statement)
            if not bound or not _needs_rebinding(module, bound):
                continue
            try:
                _perform_import(statement, module)
            except Exception as exception:
                ok = False
                logger.warning(
                    f"Unsloth: {backend} installed, but replaying "
                    f"`{ast.unparse(statement)}` in {module.__name__} failed: "
                    f"{type(exception).__name__}: {exception}"
                )
    return ok


def patch_requires_backends_autoinstall():
    """Wrap ``requires_backends`` so an allow-listed missing backend triggers a one-shot
    pip install and a retry. The original ImportError is preserved on failure, so error
    bytes stay identical to upstream under ``UNSLOTH_AUTO_INSTALL=0``."""
    try:
        from transformers.utils import import_utils as iu
    except Exception:
        return
    current = getattr(iu, "requires_backends", None)
    if current is None:
        return
    if getattr(current, "_unsloth_patched", False):
        # Re-broadcast rather than return early: a transformers module imported since
        # the previous pass still holds its own copy of the original.
        _rebind_requires_backends(current, getattr(current, "_unsloth_original", None))
        return
    _orig = current

    def requires_backends(obj, backends):
        try:
            return _orig(obj, backends)
        except ImportError:
            if not _auto_install_enabled() or _no_network():
                raise
            wanted_iter = backends if isinstance(backends, (list, tuple)) else [backends]
            wanted = [b for b in wanted_iter if isinstance(b, str) and b in _ALLOW_LIST]
            if not wanted:
                raise
            # Only genuinely importable backends: on 4.x the refresh flips
            # `_<backend>_available` unconditionally and would wave the retry through.
            installed = [b for b in wanted if _try_install_and_import(b)]
            if not installed:
                raise
            for b in installed:
                _refresh_backend_availability(iu, b)
                # On replay failure the consumer is still unbound; the original
                # ImportError at least names the package.
                if not _replay_skipped_guarded_imports(iu, b):
                    raise
            return _orig(obj, backends)

    requires_backends._unsloth_patched = True
    requires_backends._unsloth_original = _orig
    iu.requires_backends = requires_backends
    _rebind_requires_backends(requires_backends, _orig)


def patch_check_imports_autoinstall():
    """trust_remote_code modeling files raise via ``dynamic_module_utils.check_imports``,
    which never reaches ``requires_backends``, so wrap it too."""
    try:
        from transformers import dynamic_module_utils as dmu
    except Exception:
        return
    if getattr(dmu.check_imports, "_unsloth_patched", False):
        return
    _orig = dmu.check_imports

    def check_imports(filename):
        try:
            return _orig(filename)
        except ImportError as e:
            if not _auto_install_enabled() or _no_network():
                raise
            msg = str(e)
            if "This modeling file requires" not in msg:
                raise
            # Message format: "... environment: pkg1, pkg2. Run `pip install...`"
            try:
                tail = msg.split("environment:", 1)[1]
                pkgs_str = tail.split(".", 1)[0]
            except Exception:
                raise
            pkgs = [p.strip() for p in pkgs_str.split(",") if p.strip() in _ALLOW_LIST]
            if not pkgs:
                raise
            ok = all(_try_install_and_import(p) for p in pkgs)
            if not ok:
                raise
            return _orig(filename)

    check_imports._unsloth_patched = True
    dmu.check_imports = check_imports


def _ipython_chain_is_broken() -> bool:
    """True only when IPython is installed but ``traitlets`` is missing: a plain
    ``import unsloth`` in a container that never had IPython must not reach the package
    manager. ``find_spec`` keeps the probe offline and avoids importing IPython."""
    return (
        importlib.util.find_spec("IPython") is not None
        and importlib.util.find_spec("traitlets") is None
    )


def _ensure_notebook_chain():
    """Repair deps that raise a bare ModuleNotFoundError outside transformers, where no
    wrapper hook can catch them."""
    if not _auto_install_enabled() or _no_network():
        return
    if not _ipython_chain_is_broken():
        return
    for pkg in ("traitlets",):
        if importlib.util.find_spec(pkg) is None:
            _try_install_and_import(pkg)


def patch_notebook_deps_autoinstall():
    patch_requires_backends_autoinstall()
    patch_check_imports_autoinstall()
    _ensure_notebook_chain()


TEMPORARY_PATCHES.append(patch_notebook_deps_autoinstall)

# Also run at import time: a trust_remote_code modeling file can load before the
# TEMPORARY_PATCHES pass. UNSLOTH_NOTEBOOK_DEPS_NO_AUTORUN=1 suppresses only this run.
if os.environ.get("UNSLOTH_NOTEBOOK_DEPS_NO_AUTORUN", "0") != "1":
    try:
        patch_notebook_deps_autoinstall()
    except Exception as _e:
        logger.warning(f"Unsloth: notebook dependency hooks deferred: {_e}")

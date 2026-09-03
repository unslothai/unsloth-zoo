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

# Auto-install missing notebook-only Python deps on first use.
#
# Four notebooks failed in the Blackwell docker validation because the slim
# venv shipped without timm / traitlets / addict / matplotlib, and the
# raising frame is buried inside HF code (`transformers.utils.import_utils.
# requires_backends` for TimmWrapper, `transformers.dynamic_module_utils.
# check_imports` for the Deepseek-OCR trust_remote_code modeling file, and
# a bare ModuleNotFoundError for traitlets from the IPython chain). Wrap
# all three call sites with a thin retry that pip-installs the offending
# package (allow-list only) and re-tries the original import. Honours the
# existing `UNSLOTH_AUTO_INSTALL=0` opt-out (used by `llama_cpp.py`) and
# the standard offline flags so air-gapped envs keep emitting the
# upstream ImportError verbatim.

import importlib
import importlib.metadata
import importlib.util
import os
import shutil
import site
import subprocess
import sys

# Absolute on purpose: transformers' custom_object_save walks relative imports
# with a regex that pastes the raw capture onto the directory, so `from ..log`
# becomes temporary_patches/.log.py and crashes the save. See PR #1045.
from unsloth_zoo.log import logger
from .common import TEMPORARY_PATCHES

# pypi-name -> import-name (None means same).
_ALLOW_LIST = {
    "timm":          None,           # vision backbones (TimmWrapperModel)
    "addict":        None,           # Deepseek-OCR config dicts
    "einops":        None,           # Deepseek-OCR deepencoder + many other vision models
    "easydict":      None,           # Deepseek-OCR deepencoder.py:12 `from easydict import EasyDict`
    "snac":          None,           # Orpheus TTS neural audio codec
    "torchcodec":    None,           # HF datasets audio Feature decoder (>= datasets 4.x)
    "matplotlib":    None,           # Deepseek-OCR + a few HF image utils
    "traitlets":     None,           # Jupyter/IPython widget chain
    "soundfile":     None,           # audio processors
    "librosa":       None,           # audio processors
    "scipy":         None,           # several processors
    "pyctcdecode":   None,           # ASR
    "tiktoken":      None,           # tokenizer remote-code paths
    "blobfile":      None,           # tiktoken backing store
    "pillow_heif":   "pillow_heif",  # HEIF images
    "decord":        None,           # video processors
    "av":            "av",           # pyav (video processors)
    "num2words":     None,           # speech text norm
    "jieba":         None,           # zh tokenizer
    "sentencepiece": None,           # tokenizers
}

_AUTO_INSTALL = os.environ.get("UNSLOTH_AUTO_INSTALL", "1") == "1"
_NO_NETWORK = (
    os.environ.get("UNSLOTH_OFFLINE", "0") == "1"
    or os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
)
_attempted: set = set()


def _is_running_prefix(root: str) -> bool:
    # Does `root` name the environment the running interpreter actually lives in?
    # `samefile` resolves symlinks and differing spellings of the same directory,
    # and raises when either side does not exist, which is the answer we want for
    # a variable left over from an environment that has since been deleted.
    try:
        return os.path.samefile(root, sys.prefix)
    except Exception:
        return False


def _in_venv() -> bool:
    # Decided from the RUNNING interpreter, never from an inherited activation
    # variable on its own. A notebook kernel routinely runs interpreter A while
    # the process inherited VIRTUAL_ENV / CONDA_PREFIX from a different
    # environment B, which is the exact mismatch `_uv_command` pins `--python`
    # for. Counting B as "in a venv" made `_pip_install` hand the install to uv
    # and `_pip_command` skip both the write probe and `--user`, while every
    # installer still targeted A's site-packages: on a system A a non-root
    # kernel then failed outright, even though plain pip would have installed
    # into the user site.
    if hasattr(sys, "real_prefix"):
        return True
    if getattr(sys, "base_prefix", sys.prefix) != sys.prefix:
        return True
    # conda environments report `base_prefix == prefix`, so the variable is the
    # only marker they have. Trust it only when it names the prefix we run in.
    return any(
        bool(root) and _is_running_prefix(root)
        for root in (os.environ.get("VIRTUAL_ENV"), os.environ.get("CONDA_PREFIX"))
    )


def _uv_command(pkg: str) -> list:
    # `--python` pins the install to the interpreter that is actually running
    # Unsloth. Without it `uv pip install` picks its target from VIRTUAL_ENV /
    # CONDA_PREFIX / a discovered `.venv`, and a notebook kernel frequently
    # runs a different interpreter than the environment its process inherited.
    # In that case the package lands in the activated environment, the
    # follow-up `find_spec` still fails, and an unrelated environment has been
    # modified for nothing. This mirrors what the pip fallback below already
    # does implicitly by invoking `sys.executable -m pip`.
    return ["uv", "pip", "install", "--quiet", "--python", sys.executable, pkg]


def _pip_command(pkg: str) -> list:
    cmd = [
        sys.executable, "-m", "pip", "install", "--quiet",
        "--disable-pip-version-check", "--no-input", pkg,
    ]
    # Outside a venv on Linux/Mac as non-root: probe write access to
    # site-packages and fall back to --user. Windows has no geteuid;
    # site-packages there is usually writable inside the venv anyway.
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
    # Retry through pip only when uv could not be aimed at this interpreter at
    # all: a uv predating `--python` exits with an argument-parser error
    # ("unexpected argument '--python' found"), and a uv that cannot resolve
    # the given path exits with a discovery error. Both happen before any
    # network access, and `sys.executable -m pip` targets the right interpreter
    # by construction, so falling back there is strictly better than giving up.
    # A genuine resolution or build failure is NOT retried, otherwise every
    # real failure would cost two installs.
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


def _try_install_and_import(pkg: str) -> bool:
    if pkg not in _ALLOW_LIST:
        return False
    if not _AUTO_INSTALL or _NO_NETWORK:
        return False
    import_name = _ALLOW_LIST[pkg] or pkg.replace("-", "_")
    if importlib.util.find_spec(import_name) is not None:
        return True
    if not _pip_install(pkg):
        return False
    return importlib.util.find_spec(import_name) is not None


def _rebind_requires_backends(wrapper, original) -> None:
    """
    ``transformers/utils/__init__.py`` re-exports ``requires_backends``, and
    modeling files import it from there rather than from ``import_utils``
    (``models/timm_wrapper/modeling_timm_wrapper.py`` does ``from ...utils
    import auto_docstring, is_timm_available, requires_backends``). Both of
    those bind their own name to the function object, so rebinding only
    ``transformers.utils.import_utils`` leaves the public alias and every
    already-imported copy on the unwrapped original -- the TimmWrapper path
    then raises for a missing `timm` without the installer ever running.
    Point every such alias at the wrapper instead.

    ``vars(module)`` is used rather than ``getattr``: lazy module shims resolve
    unknown attributes by importing submodules, and this must not trigger
    imports as a side effect. The ``is original`` identity test means only
    aliases of the exact function we wrapped are touched, so an unrelated
    ``requires_backends`` in some other package is left alone.
    """
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
    """
    Make transformers re-evaluate whether ``backend`` is importable.

    The retry re-enters the original ``requires_backends``, which decides from
    ``BACKENDS_MAPPING[backend][0]()``. On transformers 5.x that entry is an
    ``functools.lru_cache`` wrapper around ``is_<backend>_available`` which has
    already cached ``False``, so without clearing it the freshly installed
    package is still reported missing and the retry raises the very ImportError
    the install was meant to remove. transformers 4.x instead kept a module
    level ``_<backend>_available`` flag, which no longer exists in 5.x, so both
    are handled and neither is required to be present.
    """
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


def patch_requires_backends_autoinstall():
    """
    Wrap ``transformers.utils.import_utils.requires_backends`` so that an
    allow-listed missing backend triggers a one-shot pip install and a
    second attempt. Preserves the original ImportError when the install
    fails or the dep isn't on the allow-list, so user-facing error bytes
    stay identical to upstream when ``UNSLOTH_AUTO_INSTALL=0``.
    """
    try:
        from transformers.utils import import_utils as iu
    except Exception:
        return  # transformers absent (MLX-only path) -- nothing to patch.
    current = getattr(iu, "requires_backends", None)
    if current is None:
        return  # transformers version without this helper -- nothing to patch.
    if getattr(current, "_unsloth_patched", False):
        # Already wrapped. Re-broadcast rather than returning early, because a
        # transformers module imported since the previous pass holds its own
        # copy of the original. The wrapper object is the same everywhere, so
        # `_unsloth_patched` stays a single sentinel rather than becoming one
        # sentinel per location.
        _rebind_requires_backends(current, getattr(current, "_unsloth_original", None))
        return
    _orig = current

    def requires_backends(obj, backends):
        try:
            return _orig(obj, backends)
        except ImportError:
            if not _AUTO_INSTALL or _NO_NETWORK:
                raise
            wanted_iter = backends if isinstance(backends, (list, tuple)) else [backends]
            wanted = [b for b in wanted_iter if isinstance(b, str) and b in _ALLOW_LIST]
            if not wanted:
                raise
            # Refresh only the backends that are genuinely importable now.
            # On transformers 4.x `_refresh_backend_availability` flips the
            # module level `_<backend>_available` flag that
            # `is_<backend>_available` returns, so refreshing a backend whose
            # install failed makes the retry below report it as present. The
            # caller then proceeds into an import of a package that is still
            # missing and fails later with a much less actionable error.
            installed = [b for b in wanted if _try_install_and_import(b)]
            if not installed:
                raise
            for b in installed:
                _refresh_backend_availability(iu, b)
            return _orig(obj, backends)

    requires_backends._unsloth_patched = True
    requires_backends._unsloth_original = _orig
    iu.requires_backends = requires_backends
    _rebind_requires_backends(requires_backends, _orig)


def patch_check_imports_autoinstall():
    """
    trust_remote_code modeling files (e.g. Deepseek-OCR's modeling_deepseekocr.py)
    declare their import requirements at the top of the file and raise via
    ``dynamic_module_utils.check_imports`` (ImportError "This modeling file
    requires the following packages..."). That call site never reaches
    ``requires_backends``, so wrap it too.
    """
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
            if not _AUTO_INSTALL or _NO_NETWORK:
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
    """
    True only when IPython is installed but its hard dependency ``traitlets``
    is missing, i.e. the environment is already broken and the very next
    ``from IPython...`` import will die with a bare ModuleNotFoundError.

    This is deliberately narrow. A plain ``import unsloth`` in a script,
    container or CI job that never had IPython in the first place must not
    reach the package manager at all. ``find_spec`` only inspects the local
    metadata, so the probe itself is offline and cannot import IPython (which
    would need the very traitlets we are checking for).
    """
    return (
        importlib.util.find_spec("IPython") is not None
        and importlib.util.find_spec("traitlets") is None
    )


def _ensure_notebook_chain():
    """
    Repair deps that raise a bare ModuleNotFoundError outside transformers
    (the Jupyter/IPython chain), so no wrapper hook can catch them. Kept tiny:
    only ``traitlets`` is touched today; expand only when a new failure mode
    appears.
    """
    if not _AUTO_INSTALL or _NO_NETWORK:
        return
    if not _ipython_chain_is_broken():
        return
    for pkg in ("traitlets",):
        if importlib.util.find_spec(pkg) is None:
            _try_install_and_import(pkg)


def patch_notebook_deps_autoinstall():
    """Install all three hooks. Idempotent, so it is safe to run at import time
    and again from the ``TEMPORARY_PATCHES`` pass."""
    patch_requires_backends_autoinstall()
    patch_check_imports_autoinstall()
    _ensure_notebook_chain()


TEMPORARY_PATCHES.append(patch_notebook_deps_autoinstall)

# Install the two wrappers at import time as well, since a trust_remote_code
# modeling file can be loaded before the TEMPORARY_PATCHES pass runs. Both are
# pure monkeypatches with no network or package-manager side effects; the only
# hook that can reach pip is `_ensure_notebook_chain`, and it no-ops unless the
# IPython chain is already broken. Set UNSLOTH_NOTEBOOK_DEPS_NO_AUTORUN=1 to
# suppress just this import-time run (the TEMPORARY_PATCHES pass still applies),
# mirroring UNSLOTH_VENDORED_FLA_NO_AUTORUN in fla_vendor.py.
if os.environ.get("UNSLOTH_NOTEBOOK_DEPS_NO_AUTORUN", "0") != "1":
    try:
        patch_notebook_deps_autoinstall()
    except Exception as _e:
        logger.warning(f"Unsloth: notebook dependency hooks deferred: {_e}")

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

from ..log import logger

# pypi-name -> import-name (None means same).
_ALLOW_LIST = {
    "timm":          None,           # vision backbones (TimmWrapperModel)
    "addict":        None,           # Deepseek-OCR config dicts
    "einops":        None,           # Deepseek-OCR deepencoder + many other vision models
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


def _in_venv() -> bool:
    return (
        hasattr(sys, "real_prefix")
        or (getattr(sys, "base_prefix", sys.prefix) != sys.prefix)
        or bool(os.environ.get("VIRTUAL_ENV"))
        or bool(os.environ.get("CONDA_PREFIX"))
    )


def _pip_install(pkg: str) -> bool:
    if pkg in _attempted:
        return False
    _attempted.add(pkg)
    if shutil.which("uv") and _in_venv():
        cmd = ["uv", "pip", "install", "--quiet", pkg]
    else:
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
    logger.warning(
        f"Unsloth: auto-installing missing notebook dep `{pkg}` via "
        f"`{' '.join(cmd)}`. Set UNSLOTH_AUTO_INSTALL=0 to disable."
    )
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except Exception as e:
        logger.warning(f"Unsloth: auto-install of `{pkg}` failed to launch: {e}")
        return False
    if r.returncode != 0:
        tail = (r.stderr or "")[-500:]
        logger.warning(f"Unsloth: auto-install of `{pkg}` failed:\n{tail}")
        return False
    importlib.invalidate_caches()
    try:
        list(importlib.metadata.distributions())
    except Exception:
        pass
    return True


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
    if getattr(iu.requires_backends, "_unsloth_patched", False):
        return
    _orig = iu.requires_backends

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
            installed_any = False
            for b in wanted:
                if _try_install_and_import(b):
                    installed_any = True
            if not installed_any:
                raise
            for b in wanted:
                flag = f"_{b.replace('-', '_')}_available"
                if hasattr(iu, flag):
                    setattr(iu, flag, True)
            return _orig(obj, backends)

    requires_backends._unsloth_patched = True
    iu.requires_backends = requires_backends


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


def _ensure_notebook_chain():
    """
    Pre-emptive ensure for deps that raise bare ModuleNotFoundError outside
    transformers (the Jupyter/IPython chain). Kept tiny: only ``traitlets``
    is touched today; expand only when a new failure mode appears.
    """
    if not _AUTO_INSTALL or _NO_NETWORK:
        return
    for pkg in ("traitlets",):
        if importlib.util.find_spec(pkg) is None:
            _try_install_and_import(pkg)


patch_requires_backends_autoinstall()
patch_check_imports_autoinstall()
_ensure_notebook_chain()

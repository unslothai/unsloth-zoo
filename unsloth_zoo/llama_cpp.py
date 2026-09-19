# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "convert_to_gguf",
    "quantize_gguf",
    "resolve_imatrix_file",
    "quant_requires_imatrix",
    "IMATRIX_REQUIRED_QUANTS",
    "use_local_gguf",
    "assert_correct_gguf",
    "gguf_metadata_problems",
    "gguf_metadata_warnings",
    "gguf_tensor_problems",
    "install_llama_cpp",
    "check_llama_cpp",
    "_download_convert_hf_to_gguf",
    "UNSLOTH_HOME",
    "LLAMA_CPP_DEFAULT_DIR",
    "IS_WINDOWS",
]

import collections
import errno
import hashlib
import marshal
import threading
import subprocess
import sys
import os
import time
import re
import ast
import requests
import json
from tqdm.auto import tqdm as ProgressBar
from functools import lru_cache
import contextlib
import importlib.util
import tempfile
import logging
import shlex
import shutil
import tarfile
import zipfile
import platform
try:
    import torch
except ImportError:
    torch = None
from pathlib import Path
import psutil
try:
    from .device_type import device_is_bf16_supported
except (ImportError, NotImplementedError):
    # ImportError when torch is absent; NotImplementedError when
    # get_device_type() runs at import on an unrecognised platform.
    # Fall through to the platform probe either way.
    import platform as _platform
    _IS_APPLE_SILICON = (
        _platform.system() == "Darwin" and _platform.machine() == "arm64"
    )
    def device_is_bf16_supported():
        return _IS_APPLE_SILICON

logger = logging.getLogger(__name__)
# Configure basic logging if not already configured elsewhere
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')

LLAMA_CPP_CONVERT_FILE = \
    "https://github.com/ggerganov/llama.cpp/raw/refs/heads/master/convert_hf_to_gguf.py"

LLAMA_CPP_CONVERTER_FILENAMES = ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py")

COMMANDS_NOT_FOUND = (
    "command not found",
    "not found",
    "No such file or directory",
)
PIP_MODULE_NOT_FOUND = (
    "no module named pip",
    "no module named 'pip'",
    "no module named pip.__main__",
    "modulenotfounderror: no module named 'pip'",
)

# llama.cpp specific targets - all takes 90s. Below takes 60s
LLAMA_CPP_TARGETS = [
    "llama-quantize",
    # "llama-export-lora",
    "llama-cli",
    # "llama-llava-cli",
    "llama-mtmd-cli",
    "llama-gguf-split",
    "llama-server",
]

PIP_OPTIONS = [
    f'"{sys.executable}" -m pip',  # Always prefer the running interpreter's pip
    "uv pip", # Astral's uv
    "pip",
    "pip3",
    "python3 -m pip", # Python standalone installation
    "py -m pip", # Windows
    "poetry", # Poetry
]

BAD_OUTCOMES = {
    "undefined reference"        : "Please report this ASAP!",
    "Unknown argument"           : "Please report this ASAP!",
    "[FAIL]"                     : "Please report this ASAP!",
    "--break-system-packages"    : "You need to redo the command manually with elevated permissions.",
    "establish a new connection" : "You do not have internet connection!",
    "fatal: unable to access"    : "You do not have internet connection!",
    "failure resolving"          : "You do not have internet connection!",
    "fatal "                     : "",
    "Err:"                       : "",
    "Failed "                    : "",
}

# Detection lives in disk_utils so unsloth and unsloth_zoo cannot drift apart
# on it, and so a KAGGLE_USERNAME exported for the Kaggle CLI on a laptop no
# longer looks like a Kaggle kernel.
try:
    from .disk_utils import (
        is_colab_environment as _is_colab_environment,
        is_kaggle_environment as _is_kaggle_environment,
    )
except ImportError:
    # Loaded as a standalone file with no package context, which is how the
    # tests skip unsloth_zoo's import-time device detection. Load the sibling
    # by path rather than duplicating it.
    import importlib.util as _importlib_util
    _disk_utils_spec = _importlib_util.spec_from_file_location(
        "_unsloth_zoo_disk_utils",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "disk_utils.py"),
    )
    _disk_utils = _importlib_util.module_from_spec(_disk_utils_spec)
    _disk_utils_spec.loader.exec_module(_disk_utils)
    _is_colab_environment = _disk_utils.is_colab_environment
    _is_kaggle_environment = _disk_utils.is_kaggle_environment

# Static scan of the converter we download and execute. Vendored inside the
# package rather than left in scripts/, which pyproject excludes from the wheel.
try:
    from .converter_scan import (
        RE_ARGPARSE_DEFAULT,
        ConverterScanError,
        scan_is_disabled,
        scan_is_strict,
        warn_on_suspicious_converter,
    )
except ImportError:
    # Standalone file load with no package context, as above.
    import importlib.util as _importlib_util
    _converter_scan_spec = _importlib_util.spec_from_file_location(
        "_unsloth_zoo_converter_scan",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "converter_scan.py"),
    )
    _converter_scan = _importlib_util.module_from_spec(_converter_scan_spec)
    _converter_scan_spec.loader.exec_module(_converter_scan)
    RE_ARGPARSE_DEFAULT = _converter_scan.RE_ARGPARSE_DEFAULT
    ConverterScanError = _converter_scan.ConverterScanError
    scan_is_disabled = _converter_scan.scan_is_disabled
    scan_is_strict = _converter_scan.scan_is_strict
    warn_on_suspicious_converter = _converter_scan.warn_on_suspicious_converter

IS_COLAB_ENVIRONMENT  = _is_colab_environment()
IS_KAGGLE_ENVIRONMENT = _is_kaggle_environment()
IS_WINDOWS = sys.platform == "win32"

# Default llama.cpp location: ~/.unsloth/llama.cpp
# Override with UNSLOTH_LLAMA_CPP_PATH env var to use a custom llama.cpp install
#
# Deliberately does NOT move on Kaggle: only /kaggle/working is small there. A
# probe kernel measured home on the same large overlay as /tmp (1026.8GB free
# of 8062.4GB on both), so the checkout and build tree already have room.
UNSLOTH_HOME = os.path.join(str(Path.home()), ".unsloth")
LLAMA_CPP_DEFAULT_DIR = os.environ.get(
    "UNSLOTH_LLAMA_CPP_PATH",
    os.path.join(UNSLOTH_HOME, "llama.cpp"),
)

# Prebuilt llama.cpp binaries. CPU builds come from upstream ggml-org
# releases; GPU (CUDA/ROCm/Metal) bundles come from the unslothai/llama.cpp
# fork that Unsloth Studio also installs from, selected via its manifest and
# verified against its published sha256 list. Marker file distinguishes a
# prebuilt install from a corrupted source checkout.
UNSLOTH_PREBUILT_INFO_FILENAME = "UNSLOTH_PREBUILT_INFO.json"
LLAMA_CPP_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
LLAMA_CPP_PUBLISHED_RELEASES_API = "https://api.github.com/repos/unslothai/llama.cpp/releases"
LLAMA_CPP_SOURCE_TARBALL = "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/tags/{tag}"
LLAMA_CPP_PREBUILT_MANIFEST_ASSET = "llama-prebuilt-manifest.json"
LLAMA_CPP_PREBUILT_SHA256_ASSET = "llama-prebuilt-sha256.json"


def _resolve_local_convert_script():
    """Return (abs_path, mtime_ns, size) for a local convert_hf_to_gguf.py if
    UNSLOTH_LLAMA_CPP_SCRIPTS_DIR holds one, else None. mtime_ns/size are part
    of the cache key so in-place updates are honored. An invalid env var raises
    RuntimeError (an explicit pin fails closed rather than hitting the network).
    """
    scripts_dir = os.environ.get("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR")
    if not scripts_dir:
        return None
    scripts_dir = os.path.abspath(os.path.expanduser(scripts_dir))
    if not os.path.isdir(scripts_dir):
        raise RuntimeError(
            f"Unsloth: UNSLOTH_LLAMA_CPP_SCRIPTS_DIR='{scripts_dir}' is not a directory. "
            f"Unset UNSLOTH_LLAMA_CPP_SCRIPTS_DIR to use the network converter, "
            f"or point it at a directory containing convert_hf_to_gguf.py."
        )
    for name in LLAMA_CPP_CONVERTER_FILENAMES:
        candidate = os.path.join(scripts_dir, name)
        try:
            if not os.path.isfile(candidate):
                continue
            stat = os.stat(candidate)
        except OSError as exc:
            raise RuntimeError(
                f"Unsloth: Could not inspect local llama.cpp converter at '{candidate}': {exc}"
            ) from exc
        return (candidate, stat.st_mtime_ns, stat.st_size)
    raise RuntimeError(
        f"Unsloth: UNSLOTH_LLAMA_CPP_SCRIPTS_DIR='{scripts_dir}' has no "
        f"convert_hf_to_gguf.py or convert-hf-to-gguf.py. Unset the env var "
        f"to use the network converter."
    )


def _resolve_bundle_convert_script():
    """Fallback when UNSLOTH_LLAMA_CPP_SCRIPTS_DIR is unset: a prebuilt llama.cpp
    bundle ships convert_hf_to_gguf.py alongside its own conversion/ package, so
    the two are co-versioned. Downloading the latest entrypoint instead runs it
    against the bundle's older conversion/ModelBase and crashes (e.g. unexpected
    target_model_dir kwarg). Prefer the bundle's converter when, and only when,
    that paired conversion/ package is present. We require both __init__.py and
    base.py, the same signal _detect_converter_layout uses, so selection and
    layout detection never disagree. Returns (path, mtime_ns, size) or None
    (monolith installs / trees without a paired conversion/ fall through)."""
    bundle_dir = LLAMA_CPP_DEFAULT_DIR
    if not bundle_dir or not os.path.isdir(bundle_dir):
        return None
    conversion_dir = os.path.join(bundle_dir, "conversion")
    if not (os.path.isfile(os.path.join(conversion_dir, "__init__.py")) and
            os.path.isfile(os.path.join(conversion_dir, "base.py"))):
        return None
    for name in LLAMA_CPP_CONVERTER_FILENAMES:
        candidate = os.path.join(bundle_dir, name)
        try:
            if not os.path.isfile(candidate):
                continue
            stat = os.stat(candidate)
        except OSError:
            continue
        logger.info(
            f"Unsloth: Using bundle convert_hf_to_gguf.py from {candidate} "
            f"(co-versioned with its conversion/ package)"
        )
        return (candidate, stat.st_mtime_ns, stat.st_size)
    return None
pass


# `sys.path` and `sys.modules` are process-global, so two conversions swapping different
# gguf trees at once would interleave: one read back parses with the other's package, and
# whichever cleanup runs last restores a temporary package over the process's own. Nothing
# stopped that before, and the read back now runs on every export rather than only when a
# caller asked for it, so the swap is serialised. Reentrant, since the body of one swap
# can reach code that opens another with the same tree.
_GGUF_MODULE_SWAP_LOCK = threading.RLock()


@contextlib.contextmanager
def use_local_gguf(gguf_py_path = None):
    """Context manager to temporarily use llama.cpp's local gguf-py

    `gguf_py_path` names the tree to use. It defaults to the one beside the
    default install, which is the pre-existing behaviour and what every
    existing caller gets. `convert_to_gguf` passes the tree it actually
    resolved for the converter child, so the file is read back with the same
    `gguf` that wrote it rather than with whatever the parent happens to have.

    Only one swap runs at a time: see `_GGUF_MODULE_SWAP_LOCK`.
    """
    with _GGUF_MODULE_SWAP_LOCK:
        yield from _use_local_gguf(gguf_py_path)


def _use_local_gguf(gguf_py_path):
    """The swap itself, run under `_GGUF_MODULE_SWAP_LOCK`."""
    # Store original state
    original_sys_path = sys.path.copy()
    original_modules = set(sys.modules.keys())
    if gguf_py_path is None:
        gguf_py_path = os.path.join(LLAMA_CPP_DEFAULT_DIR, "gguf-py")

    original_gguf_modules = {}

    try:
        if os.path.exists(gguf_py_path):
            logger.debug(f"Adding {gguf_py_path} to sys.path")
            # Index 0, ahead of the script directory Python puts there: a process
            # launched from inside another gguf-py checkout has that checkout at
            # sys.path[0], and the point of the read back is to parse the file with
            # the `gguf` that wrote it.
            sys.path.insert(0, gguf_py_path)

            # Drop system gguf modules to force a reimport from gguf-py
            gguf_modules = [key for key in sys.modules.keys() if key.startswith('gguf')]
            for module in gguf_modules:
                original_gguf_modules[module] = sys.modules[module]
                del sys.modules[module]
                logger.debug(f"Removed system module {module}")

        yield

    finally:
        sys.path[:] = original_sys_path

        # Remove any newly imported gguf modules
        new_modules = set(sys.modules.keys()) - original_modules
        gguf_modules_to_remove = [m for m in new_modules if m.startswith('gguf')]
        for module in gguf_modules_to_remove:
            del sys.modules[module]
            logger.debug(f"Cleaned up module {module}")

        for module_name, module_obj in original_gguf_modules.items():
            sys.modules[module_name] = module_obj
            logger.debug(f"Restored original module {module_name}")

        logger.debug("Restored original Python environment")
pass

_AUTO_INSTALL_TRUE_VALUES = frozenset({"1", "ON", "TRUE", "YES"})


def _auto_install_enabled() -> bool:
    """Read at the attempt, not at import, so setting it after `import unsloth` works."""
    return os.environ.get("UNSLOTH_AUTO_INSTALL", "1").strip().upper() \
        in _AUTO_INSTALL_TRUE_VALUES


def _stdin_is_usable() -> bool:
    """Whether sys.stdin still refers to a live descriptor we could have read from."""
    try:
        os.fstat(sys.stdin.fileno())
    except Exception:
        # None, closed, detached, or a wrapper with no real fd. All mean no one to ask.
        return False
    return True


def install_package(package, sudo = False, print_output = False, print_outputs = None, system_type = "debian"):
    # All Unsloth Zoo code licensed under LGPLv3

    # Checked before the platform branch. The Windows arm returns early and the Colab
    # and Kaggle paths skip the prompt, so an opt out placed any lower would miss them.
    if not _auto_install_enabled():
        raise RuntimeError(
            f"Unsloth: Installation of `{package}` was cancelled (UNSLOTH_AUTO_INSTALL=0)!\n"\
            "Please install llama.cpp manually via https://docs.unsloth.ai/basics/troubleshooting-and-faqs#how-do-i-manually-save-to-gguf"
        )

    if IS_WINDOWS:
        # Per-package winget config aligned with setup.ps1
        # Each entry: (winget_id, extra_args_list)
        WINGET_PACKAGES = {
            'git': ('Git.Git', []),
            'cmake': ('Kitware.CMake', []),
            'build-essential': (
                'Microsoft.VisualStudio.2022.BuildTools',
                ['--override',
                 '--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive --wait'],
            ),
            'openssl': ('ShiningLight.OpenSSL.Dev', []),
        }

        # Handle space-separated multi-package strings
        packages = package.strip().split()
        for pkg in packages:
            pkg_lower = pkg.lower()
            entry = WINGET_PACKAGES.get(pkg_lower)
            if entry is None:
                print(f"Unsloth: Package '{pkg}' not applicable on Windows, skipping.")
                continue

            winget_id, extra_args = entry
            if shutil.which('winget') is None:
                raise RuntimeError(
                    f"Unsloth: Missing '{pkg}' and winget not available.\n"
                    f"Install manually: winget install {winget_id}"
                )

            print(f"Unsloth: Installing {pkg} via winget ({winget_id})...")
            cmd = [
                'winget', 'install', '-e', '--id', winget_id,
                '--source', 'winget',
                '--accept-package-agreements',
                '--accept-source-agreements',
            ] + extra_args

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Unsloth: Failed to install {winget_id} via winget.\n"
                    f"Install manually: winget install {winget_id}"
                )
            if print_output: print(f"Unsloth: Successfully installed {winget_id}", flush=True)
            if print_outputs is not None: print_outputs.append(f"Installed {winget_id}")
        return

    # Choose package manager based on system type
    if system_type == "rpm":
        pkg_manager = "yum" if os.path.exists('/usr/bin/yum') else "dnf"
        install_cmd = f"{'sudo ' if sudo else ''}{pkg_manager} install {package} -y"
    elif system_type == "arch":
        install_cmd = f"{'sudo ' if sudo else ''}pacman -S --noconfirm {package}"
    else:  # Default to debian/apt-get
        install_cmd = f"{'sudo ' if sudo else ''}apt-get install {package} -y"

    print(f"Unsloth: Installing packages: {package}")
    if not (IS_COLAB_ENVIRONMENT or IS_KAGGLE_ENVIRONMENT):
        # input() raises in non-interactive contexts. Under `docker run` without -i, or with
        # stdin from /dev/null, this used to propagate through save_pretrained_gguf as
        # `RuntimeError: Unsloth: GGUF conversion failed: EOF when reading a line`.
        try:
            acceptance = input(f"Missing system packages. We need to execute `{install_cmd}` - do you accept? Press ENTER. Type NO if not.")
        except (EOFError, RuntimeError, ValueError, OSError) as exception:
            # A stdin closed AFTER interpreter start raises neither EOFError nor
            # RuntimeError. Measured on CPython 3.13:
            #   sys.stdin.close()  -> ValueError: I/O operation on closed file.
            #   os.close(0)        -> OSError: [Errno 9] Bad file descriptor
            #   sys.stdin = None   -> RuntimeError: lost sys.stdin
            #   stdin=/dev/null    -> EOFError
            # Daemon and process wrappers close fd 0, so without the first two a headless
            # export still dies on the input() call this whole branch exists to survive.
            # All four mean the same thing: there is no one to ask. The terminal check
            # below still distinguishes Ctrl-D, and it holds for these too, since a closed
            # sys.stdin makes isatty() raise and a closed fd 0 makes it return False.
            #
            # Each type is accepted only in the exact state that means "stdin is gone".
            # An unrelated I/O error on a live stdin is not consent: EIO from a serial
            # console, or a ValueError out of a custom stdin wrapper, must still propagate.
            # CPython raises RuntimeError for a lost stdout or stderr too.
            if isinstance(exception, RuntimeError) and sys.stdin is not None:
                raise
            # EBADF on its own is not enough. input() writes the prompt to stdout before
            # it reads, so a closed fd 1 raises EBADF while fd 0 is open, non-tty and
            # holding an unread answer. Reproduced on CPython: `python -u` with fd 1
            # closed gives OSError(9) with sys.stdin.closed and isatty() both False.
            # So ask stdin itself, not the errno.
            if isinstance(exception, OSError) and (
                exception.errno != errno.EBADF or _stdin_is_usable()
            ):
                raise
            if isinstance(exception, ValueError) and not getattr(sys.stdin, "closed", False):
                raise
            # EOFError on a terminal is Ctrl-D and still cancels. EOFError with no terminal
            # means there was never anyone to ask, which is the same implicit ENTER the
            # prompt already documents. A stdin whose isatty() raises counts as no terminal.
            try:
                _stdin_is_a_tty = sys.stdin is not None and sys.stdin.isatty()
            except Exception:
                _stdin_is_a_tty = False
            if _stdin_is_a_tty:
                raise RuntimeError(
                    f"Unsloth: Execution of `{install_cmd}` was cancelled!\n"\
                    "Please install llama.cpp manually via https://docs.unsloth.ai/basics/troubleshooting-and-faqs#how-do-i-manually-save-to-gguf"
                )
            acceptance = ""
        if "no" in str(acceptance).lower():
            raise RuntimeError(
                f"Unsloth: Execution of `{install_cmd}` was cancelled!\n"\
                "Please install llama.cpp manually via https://docs.unsloth.ai/basics/troubleshooting-and-faqs#how-do-i-manually-save-to-gguf"
            )
    with subprocess.Popen(install_cmd, shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT) as sp:
        for line in sp.stdout:
            line = line.decode("utf-8", errors = "replace").rstrip()

            if "Permission denied" in line or "not open lock file" in line or "are you root?" in line or "fatal" in line:
                sp.terminate()
                raise RuntimeError(f"[FAIL] Unsloth: Permission denied when installing package {package}\n"\
                                   "This operation requires elevated sudo/root permissions. Please manually install missing packages and retry again"
                    )
            elif line.endswith(COMMANDS_NOT_FOUND):
                sp.terminate()
                pkg_mgr_name = {"rpm": "yum/dnf", "arch": "pacman"}.get(system_type, "apt-get")
                raise RuntimeError(f"[FAIL] Unsloth: {pkg_mgr_name} does not exist when installing {package}? Is this NOT a Linux / Mac based computer?")
            elif "Unable to locate package" in line:
                sp.terminate()
                raise RuntimeError(f"[FAIL] Unsloth: Could not install package {package} since it does not exist.")
            if print_output: print(line, flush = True, end = "")
            if print_outputs is not None: print_outputs.append(line)
        pass
    pass
pass


def do_we_need_sudo(system_type="debian"):
    # All Unsloth Zoo code licensed under LGPLv3
    if IS_WINDOWS:
        return False

    # Check apt-get updating
    sudo = False
    print("Unsloth: Updating system package directories")

    # Choose update command based on system type
    if system_type == "rpm":
        pkg_manager = "yum" if os.path.exists('/usr/bin/yum') else "dnf"
        update_cmd = f"{pkg_manager} check-update"
    elif system_type == "arch":
        update_cmd = "pacman -Sy"
    else:
        update_cmd = "apt-get update -y"

    start_time = time.time()
    with subprocess.Popen(update_cmd, shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT) as sp:
        for line in sp.stdout:
            line = line.decode("utf-8", errors = "replace").rstrip()
            if "Permission denied" in line or "not open lock file" in line or "are you root?" in line or "fatal" in line:
                sp.terminate()
                sudo = True
                break
            elif line.endswith(COMMANDS_NOT_FOUND):
                sp.terminate()
                pkg_mgr_name = {"rpm": "yum/dnf", "arch": "pacman"}.get(system_type, "apt-get")
                raise RuntimeError(f"[FAIL] Unsloth: {pkg_mgr_name} does not exist? Is this NOT a Linux / Mac based computer?")
            elif "failure resolving" in line or "Err:" in line:
                sp.terminate()
                raise RuntimeError("[FAIL] Unsloth: You do not have internet connection!")
            elif time.time() - start_time >= 180:
                # Failure if longer than 3 minutes
                sp.terminate()
                raise RuntimeError("[FAIL] Unsloth: You do not have internet connection!")
        pass
    pass

    # Update all package lists as well
    update_cmd_sudo = f"sudo {update_cmd}"

    start_time = time.time()
    with subprocess.Popen(update_cmd_sudo, shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT) as sp:
        for line in sp.stdout:
            line = line.decode("utf-8", errors = "replace").rstrip()
            if "Permission denied" in line or "not open lock file" in line or "are you root?" in line or "fatal" in line:
                sp.terminate()
                raise RuntimeError("[FAIL] Unsloth: Tried with sudo, but still failed?")
            elif "failure resolving" in line or "Err:" in line:
                sp.terminate()
                raise RuntimeError("[FAIL] Unsloth: You do not have internet connection!")
            elif time.time() - start_time >= 180:
                # Failure if longer than 3 minutes
                sp.terminate()
                raise RuntimeError("[FAIL] Unsloth: You do not have internet connection!")
        pass
    pass

    #if sudo: print("Unsloth: All commands will now use admin permissions (sudo)")
    return sudo
pass


def check_pip():
    # All Unsloth Zoo code licensed under LGPLv3
    def _is_safe_candidate(pip):
        # Guard against malformed or shell-injected candidates.
        if any(char in pip for char in (";", "|", "&", ">", "<", "`", "$", "\n", "\r")):
            return False
        try:
            tokens = shlex.split(pip)
        except ValueError:
            return False
        if tokens in (["pip"], ["pip3"], ["uv", "pip"], ["poetry"]):
            return True
        if len(tokens) == 3 and tokens[1] == "-m" and tokens[2] == "pip":
            return True
        return False

    def _is_missing_command(output):
        markers = tuple(marker.lower() for marker in COMMANDS_NOT_FOUND)
        for line in output.splitlines():
            lowered = line.rstrip().lower()
            if lowered.endswith(markers):
                return True
        return False

    def _is_missing_pip_module(output):
        lowered = output.lower()
        return any(marker in lowered for marker in PIP_MODULE_NOT_FOUND)

    for pip in PIP_OPTIONS:
        if not _is_safe_candidate(pip):
            continue
        # Probe each candidate in a way that reflects real usage and avoids false positives.
        # uv pip expects a subcommand, so --help is the stable probe there.
        probe_command = f"{pip} --help" if pip.startswith("uv pip") else f"{pip} --version"
        probe = subprocess.run(
            probe_command,
            shell = True,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
        )
        output = probe.stdout or ""

        if _is_missing_command(output): continue
        if _is_missing_pip_module(output): continue
        if probe.returncode != 0: continue
        # For non-uv candidates, require pip-like output to avoid selecting arbitrary commands.
        if not pip.startswith("uv pip") and "pip" not in output.lower():
            continue
        return pip
    pass
    raise RuntimeError(f"[FAIL] Unsloth: Tried all of `{', '.join(PIP_OPTIONS)}` but failed.")
pass


def try_execute(command, sudo = False, print_output = False, print_outputs = None, cwd = None, system_type = "debian", ignore_deprecation = False):
    # All Unsloth Zoo code licensed under LGPLv3

    with subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, cwd = cwd, text=True) as sp:
        stdout, stderr = sp.communicate()
        stdout = stdout or ""
        stderr = stderr or ""
        all_output = stdout + stderr

        # Check exit code
        if sp.returncode != 0:
            error_msg = f"[FAIL] Command `{command}` failed with exit code {sp.returncode}\n"
            if stdout: error_msg += f"stdout: {stdout}\n"
            if stderr: error_msg += f"stderr: {stderr}\n"
            raise RuntimeError(error_msg)

        # Process output
        for line in all_output.splitlines(keepends=True):
            # Check for command not found
            if line.rstrip().endswith(COMMANDS_NOT_FOUND):
                raise RuntimeError(f"Command not found: {command}")

            # Check for other bad outcomes
            for key, value in BAD_OUTCOMES.items():
                if key in line:
                    error_msg = f"[FAIL] Command `{command}` failed with error `{line.strip()}`\n"
                    raise RuntimeError(error_msg + value)
            key, value = "is deprecated", "Command is deprecated!"
            if not ignore_deprecation and key in line:
                error_msg = f"[FAIL] Command `{command}` failed with error `{line.strip()}`\n"
                raise RuntimeError(error_msg + value)

            if print_output:
                print(line, flush=True, end="")
            if print_outputs is not None:
                print_outputs.append(line)
pass


def try_execute_with_auto_install(command, sudo=False, print_output=False, print_outputs=None, cwd = None, system_type = "debian", ignore_deprecation = False):
    """Try to execute a command, and if it fails due to missing package, try to install it"""
    try:
        try_execute(command, sudo, print_output, print_outputs, cwd, system_type, ignore_deprecation)
    except RuntimeError as e:
        if "Command not found" in str(e):
            package_name = command.split(" ", 1)[0]
            print(f"Trying to install missing package: {package_name}")
            install_package(package_name, sudo, print_output, print_outputs, system_type)
            # Retry once
            try_execute(command, sudo, print_output, print_outputs, cwd, system_type, ignore_deprecation)
        else:
            raise
pass


def _find_visual_studio():
    """Detect VS Build Tools (aligned with setup.ps1 Find-VsBuildTools).
    Returns (cmake_generator, vs_install_path) or (None, None)."""
    program_files = [
        os.environ.get('ProgramFiles', r'C:\Program Files'),
        os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)'),
    ]
    editions = ['BuildTools', 'Community', 'Professional', 'Enterprise']
    vs_map = {'2022': '17', '2019': '16', '2017': '15'}
    for year, ver in vs_map.items():
        for pf in program_files:
            for edition in editions:
                candidate = os.path.join(pf, 'Microsoft Visual Studio', year, edition)
                vc_dir = os.path.join(candidate, 'VC', 'Tools', 'MSVC')
                if os.path.isdir(vc_dir):
                    return f"Visual Studio {ver} {year}", candidate
    return None, None


def _find_openssl_root():
    """Find OpenSSL dev on Windows (aligned with setup.ps1 $OpenSslRoots).
    Returns the root path or None."""
    openssl_roots = [
        r'C:\Program Files\OpenSSL-Win64',
        r'C:\Program Files\OpenSSL',
        r'C:\OpenSSL-Win64',
    ]
    for root in openssl_roots:
        if os.path.exists(os.path.join(root, 'include', 'openssl', 'ssl.h')):
            return root
    return None


def _find_lib_path(lib_name):
    """Find a shared library path via gcc's linker search; abs path or None."""
    try:
        result = subprocess.run(
            ['gcc', f'-print-file-name={lib_name}'],
            capture_output=True, text=True
        )
        path = os.path.realpath(result.stdout.strip())
        if os.path.isabs(path) and os.path.exists(path):
            return path
    except Exception as exc:
        # Treat any error during probing as "library not found" but log for debugging purposes.
        logger.debug("Failed to locate shared library %r via gcc: %s", lib_name, exc)
    return None


def _is_cmake_only_llama_cpp(llama_cpp_folder):
    """True if llama.cpp's Makefile is the post-CMake-migration deprecation
    stub (or missing entirely), so `make` cannot build it."""
    makefile = os.path.join(llama_cpp_folder, "Makefile")
    if not os.path.exists(makefile):
        return True
    try:
        with open(makefile, "r", encoding = "utf-8", errors = "ignore") as f:
            content = f.read(4096)
    except OSError:
        return False
    lowered = content.lower()
    return "build system changed" in lowered or ("cmake" in lowered and "deprecated" in lowered)


def check_llama_cpp(llama_cpp_folder = LLAMA_CPP_DEFAULT_DIR):
    # All Unsloth Zoo code licensed under LGPLv3
    # Check if the folder exists
    if not os.path.exists(llama_cpp_folder):
        raise RuntimeError(f"llama.cpp folder '{llama_cpp_folder}' does not exist")

    quantizer_location = None
    converter_location = None

    # On Windows, binaries have .exe extension and live in build/bin/Release/
    if IS_WINDOWS:
        quantizer_names = ["llama-quantize.exe", "quantize.exe"]
        search_dirs = [
            llama_cpp_folder,
            os.path.join(llama_cpp_folder, "build", "bin", "Release"),
        ]
    else:
        quantizer_names = ["llama-quantize", "quantize"]
        search_dirs = [llama_cpp_folder]

    # Check for quantizer binary
    for quantizer in quantizer_names:
        for search_dir in search_dirs:
            location = os.path.join(search_dir, quantizer)
            if not os.path.exists(location):
                continue
            # os.access(X_OK) is unreliable on Windows — skip it
            if not IS_WINDOWS and not os.access(location, os.X_OK):
                continue
            try:
                result = subprocess.run(
                    [location, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 or "usage" in result.stdout.lower() or "usage" in result.stderr.lower():
                    quantizer_location = location
                    break
            except Exception as e: print(f"Found {quantizer} at {location} but couldn't run it: {e}")
        if quantizer_location is not None:
            break
    pass

    if quantizer_location is None:
        # List what files are actually there for debugging
        import glob
        all_files = []
        for search_dir in search_dirs:
            all_files.extend(glob.glob(os.path.join(search_dir, "*")))
        raise RuntimeError(
            f"Unsloth: No working quantizer found in {', '.join(search_dirs)}\n"
            f"Files found: {', '.join(os.path.basename(f) for f in all_files[:20])}"
        )
    pass

    # Check for converter script
    for converter in LLAMA_CPP_CONVERTER_FILENAMES:
        location = os.path.join(llama_cpp_folder, converter)
        if os.path.isfile(location):
            converter_location = location
            break
    pass

    if converter_location is None:
        raise RuntimeError(f"Unsloth: Failed to find converter script in {llama_cpp_folder}")
    pass

    return quantizer_location, converter_location
pass


def _is_safe_to_delete(path):
    """Check if a path is safe to delete (must be under UNSLOTH_HOME or be a llama.cpp dir)."""
    try:
        real_path = os.path.realpath(path)
        real_home = os.path.realpath(UNSLOTH_HOME)
        # Safe if under ~/.unsloth/
        if real_path.startswith(real_home + os.sep):
            return True
        # Safe if it's the CWD-relative llama.cpp (backward-compat path)
        cwd_llama = os.path.realpath(os.path.join(os.getcwd(), "llama.cpp"))
        if real_path == cwd_llama:
            return True
    except Exception as exc:
        # On any unexpected error, treat the path as unsafe but log for debugging.
        logger.debug("Failed to check if path %r is safe to delete: %s", path, exc)
    return False


def _github_auth_headers():
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _requests_get_with_retries(url, timeout = (10, 120), headers = None, stream = False, max_attempts = 3):
    last_error = None
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout = timeout, headers = headers, stream = stream)
            if response.status_code in (403, 429):
                logger.warning(
                    "Unsloth: GitHub returned HTTP %s for %s. "
                    "Set GH_TOKEN or GITHUB_TOKEN to raise the rate limit.",
                    response.status_code, url,
                )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt + 1 < max_attempts:
                time.sleep(2 ** attempt)
    raise last_error


def _resolve_llama_cpp_release(releases_api = LLAMA_CPP_RELEASES_API):
    """Return (tag, {asset_name: download_url}) for the UNSLOTH_LLAMA_TAG
    pinned release or the latest one, or None when resolution fails."""
    tag = os.environ.get("UNSLOTH_LLAMA_TAG", "").strip()
    url = f"{releases_api}/tags/{tag}" if tag else f"{releases_api}/latest"
    try:
        release = _requests_get_with_retries(url, headers = _github_auth_headers()).json()
        assets = {a["name"]: a["browser_download_url"] for a in release.get("assets", [])}
        return release["tag_name"], assets
    except Exception as e:
        logger.warning("Unsloth: Could not resolve a llama.cpp release (%s).", e)
        return None


def _fetch_release_json_asset(assets, asset_name):
    url = assets.get(asset_name)
    if not url:
        return None
    try:
        return _requests_get_with_retries(url, headers = _github_auth_headers()).json()
    except Exception as e:
        logger.warning("Unsloth: Could not fetch %s (%s).", asset_name, e)
        return None


def _detect_gpu_target():
    """Return ("cuda", sm, "cuda12"/"cuda13"/None) or ("rocm", "gfxNNNN")
    from torch, or None when no GPU target is detectable."""
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        if getattr(torch.version, "hip", None):
            gfx = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or ""
            gfx = gfx.split(":")[0]
            return ("rocm", gfx) if gfx.startswith("gfx") else None
        major, minor = torch.cuda.get_device_capability(0)
        cuda_version = getattr(torch.version, "cuda", None)
        line = f"cuda{cuda_version.split('.')[0]}" if cuda_version else None
        return ("cuda", major * 10 + minor, line)
    except Exception:
        return None


def _rocm_gfx_family(gfx):
    """Map a gcnArchName to the fork's per-family ROCm bundle suffix."""
    if gfx in ("gfx1150", "gfx1151"):
        return gfx
    for prefix, family in (("gfx103", "gfx103X"), ("gfx110", "gfx110X"), ("gfx120", "gfx120X")):
        if gfx.startswith(prefix):
            return family
    return None


def _select_gpu_assets(tag, assets, manifest, target = None):
    """Ordered download attempts [(asset_name, url), ...] of unslothai/llama.cpp
    GPU bundles for this host: narrowest CUDA coverage for the torch runtime
    line first, that line's portable build next, then the other line; ROCm by
    gfx family; macOS by the fork's Metal bundles. Empty list = compile.
    target is the _detect_gpu_target() result; pass it to reuse an already-probed
    value (the caller's gate probes it once), else it is detected here."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"): arch = "x64"
    elif machine in ("aarch64", "arm64"): arch = "arm64"
    else: return []
    system = platform.system()

    if system == "Darwin":
        name = f"llama-{tag}-bin-macos-{arch}.tar.gz"
        return [(name, assets[name])] if name in assets else []

    if target is None:
        target = _detect_gpu_target()
    if target is None:
        return []
    artifacts = (manifest or {}).get("artifacts", [])

    if target[0] == "rocm":
        if arch != "x64":
            return []
        family = _rocm_gfx_family(target[1])
        kind = "windows-rocm" if system == "Windows" else "linux-rocm"
        if family is None:
            return []
        return [
            (a["asset_name"], assets[a["asset_name"]])
            for a in artifacts
            if a.get("install_kind") == kind
            and family in a.get("asset_name", "")
            and a.get("asset_name") in assets
        ]

    _, sm, preferred_line = target
    if system == "Windows":
        kind = "windows-cuda"
    elif system == "Linux":
        kind = "linux-arm64-cuda" if arch == "arm64" else "linux-cuda"
    else:
        return []
    kind_artifacts = [
        a for a in artifacts
        if a.get("install_kind") == kind and a.get("asset_name") in assets
    ]
    lines = [preferred_line] if preferred_line else []
    lines += [l for l in ("cuda13", "cuda12") if l not in lines]

    attempts = []
    for line in lines:
        covering = []
        portable = None
        for a in (a for a in kind_artifacts if a.get("runtime_line") == line):
            supported = {int(s) for s in a.get("supported_sms", []) if str(s).isdigit()}
            if sm not in supported:
                continue
            if a.get("coverage_class") == "portable":
                portable = a
            else:
                covering.append(a)
        covering.sort(key = lambda a: ((a.get("max_sm") or 0) - (a.get("min_sm") or 0), a.get("rank") or 0))
        for a in covering[:1] + ([portable] if portable else []):
            entry = (a["asset_name"], assets[a["asset_name"]])
            if entry not in attempts:
                attempts.append(entry)
    return attempts


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_prebuilt_asset(tag, assets):
    """Map this host to the official CPU archive. llama-quantize is CPU-only,
    so the CPU bundle suffices even on GPU machines. Returns (name, url) or
    None for unsupported platforms or releases missing the asset."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"): arch = "x64"
    elif machine in ("aarch64", "arm64"): arch = "arm64"
    else: return None
    name = {
        ("Linux",   "x64")   : f"llama-{tag}-bin-ubuntu-x64.tar.gz",
        ("Linux",   "arm64") : f"llama-{tag}-bin-ubuntu-arm64.tar.gz",
        ("Darwin",  "x64")   : f"llama-{tag}-bin-macos-x64.tar.gz",
        ("Darwin",  "arm64") : f"llama-{tag}-bin-macos-arm64.tar.gz",
        ("Windows", "x64")   : f"llama-{tag}-bin-win-cpu-x64.zip",
        ("Windows", "arm64") : f"llama-{tag}-bin-win-cpu-arm64.zip",
    }.get((platform.system(), arch))
    if name is None or name not in assets:
        return None
    return name, assets[name]


def _select_cpu_assets(tag, assets, manifest):
    """Ordered download attempts [(asset_name, url), ...] of the unslothai/llama.cpp
    fork's CPU bundle for this host -- the final prebuilt fallback before a source
    compile (its app-*-cpu archive still ships llama-quantize, the only binary the
    export path needs). On macOS the CPU and GPU bundle are the same Metal archive,
    named by convention. On Linux/Windows the fork CPU asset names carry a
    commit-hash suffix, so they are looked up by the manifest's install_kind rather
    than constructed. Empty list = no fork CPU bundle for this host."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"): arch = "x64"
    elif machine in ("aarch64", "arm64"): arch = "arm64"
    else: return []
    system = platform.system()

    if system == "Darwin":
        name = f"llama-{tag}-bin-macos-{arch}.tar.gz"
        return [(name, assets[name])] if name in assets else []

    kind = {
        ("Linux",   "x64")   : "linux-cpu",
        ("Linux",   "arm64") : "linux-arm64",
        ("Windows", "x64")   : "windows-cpu",
        ("Windows", "arm64") : "windows-arm64",
    }.get((system, arch))
    if kind is None:
        return []
    artifacts = (manifest or {}).get("artifacts", [])
    return [
        (a["asset_name"], assets[a["asset_name"]])
        for a in artifacts
        if a.get("install_kind") == kind and a.get("asset_name") in assets
    ]


def _download_archive(url, dest_path):
    response = _requests_get_with_retries(url, headers = _github_auth_headers(), stream = True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size = 1 << 20):
            f.write(chunk)


def _extract_archive(archive_path, extract_dir):
    """Extract a release .zip / .tar.gz, refusing path-escaping members."""
    real_root = os.path.realpath(extract_dir)
    def _escapes(target):
        try:
            return os.path.commonpath([real_root, target]) != real_root
        except ValueError:
            return True
    def _check(name):
        target = os.path.realpath(os.path.join(extract_dir, name))
        if _escapes(target):
            raise RuntimeError(f"Unsloth: Archive member escapes extraction dir: {name}")
        return target
    def _check_tar_member(member):
        member_target = _check(member.name)
        if member.issym() or member.islnk():
            # Hardlink targets are archive-relative (resolve from the root);
            # symlink targets resolve from the link's own directory.
            link_base = real_root if member.islnk() else os.path.dirname(member_target)
            link_target = member.linkname if os.path.isabs(member.linkname) else os.path.join(link_base, member.linkname)
            if _escapes(os.path.realpath(link_target)):
                raise RuntimeError(f"Unsloth: Archive link escapes extraction dir: {member.name} -> {member.linkname}")
        elif not (member.isfile() or member.isdir()):
            raise RuntimeError(f"Unsloth: Unsupported archive member type: {member.name}")
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                _check(member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RuntimeError(f"Unsloth: Archive contains an unsupported symlink: {member.filename}")
            archive.extractall(extract_dir)
    else:
        tar_kwargs = {"filter": "data"} if sys.version_info >= (3, 12) else {}
        with tarfile.open(archive_path, "r:gz") as archive:
            # Validate every member (rejecting links whose targets escape) before
            # extracting anything, so no escaping symlink is ever written for a
            # later member to traverse through. extractall defers directory attrs
            # until contents are written, which per-member extract would break.
            members = archive.getmembers()
            for member in members: _check_tar_member(member)
            archive.extractall(extract_dir, members = members, **tar_kwargs)


def _single_extracted_root(extract_dir):
    """Release archives nest contents under llama-{tag}/ (source tarballs
    under llama.cpp-{tag}/); flat archives extract in place."""
    entries = [os.path.join(extract_dir, e) for e in os.listdir(extract_dir)]
    dirs = [e for e in entries if os.path.isdir(e)]
    if len(dirs) == 1 and len(entries) == 1:
        return dirs[0]
    return extract_dir


def _place_prebuilt_binaries(extracted_root, install_folder):
    """Copy executables + shared libs where check_llama_cpp/quantize_gguf
    look: folder root on Linux/macOS (RPATH $ORIGIN needs libs as siblings),
    build/bin/Release on Windows. ROCm bundles also carry hipblaslt/ and
    rocblas/ Tensile kernel trees that must sit next to the libs."""
    dest = os.path.join(install_folder, "build", "bin", "Release") if IS_WINDOWS else install_folder
    os.makedirs(dest, exist_ok = True)
    n_executables = 0
    lib_suffixes = (".so", ".dylib", ".dll", ".metal", ".txt", ".md", ".json")
    for entry in sorted(os.listdir(extracted_root)):
        source = os.path.join(extracted_root, entry)
        if os.path.isdir(source):
            if entry in ("hipblaslt", "rocblas"):
                shutil.copytree(source, os.path.join(dest, entry), dirs_exist_ok = True)
            continue
        target = os.path.join(dest, entry)
        shutil.copy2(source, target)
        is_lib = entry.startswith("lib") or any(s in entry for s in lib_suffixes)
        if not is_lib:
            if not IS_WINDOWS:
                os.chmod(target, 0o755)
            n_executables += 1
    if n_executables == 0:
        raise RuntimeError("Unsloth: No executables found in the prebuilt archive.")


def _hydrate_converter_sources(tag, install_folder, source_assets = None, checksums = None):
    """Copy convert_hf_to_gguf.py, conversion/ and gguf-py/ from the same-tag
    source tarball so check_llama_cpp and the converter machinery work
    without a git checkout, and tensor mappings match the binaries.

    Fork releases use "mix" tags (e.g. b9739-mix-2d6bd50) that do NOT exist on
    ggml-org, so a verbatim ggml-org download 404s and the whole prebuilt install
    fails into a source compile. Prefer the fork release's own source asset
    (llama.cpp-source-{tag}.tar.gz, passed in via source_assets) so the converter
    exactly matches the fork build; otherwise strip the -mix-... suffix and pull
    the matching upstream tag from ggml-org. Plain ggml-org tags carry no suffix,
    so this is a no-op for them (upstream_tag == tag).

    The archive is verified against the release's own published sha256 when the
    release publishes one for it. These are the files the converter subprocess
    executes, and they were arriving unverified: the sha256 in
    _stage_prebuilt_install covers the BINARY asset, and this is a second,
    separate download. The fork's llama-prebuilt-sha256.json does carry an entry
    for llama.cpp-source-{tag}.tar.gz, so on that path there is something to
    check against. The ggml-org codeload fallback publishes no digest, so it is
    reported rather than silently trusted."""
    fork_source_name = f"llama.cpp-source-{tag}.tar.gz"
    expected_sha256 = None
    if source_assets and fork_source_name in source_assets:
        source_url = source_assets[fork_source_name]
        expected_sha256 = ((checksums or {}).get(fork_source_name) or {}).get("sha256")
        if not expected_sha256:
            logger.warning(
                "Unsloth: The %s release publishes no sha256 for %s, so the "
                "converter sources it holds cannot be verified.", tag, fork_source_name,
            )
    else:
        upstream_tag = tag.split("-mix-")[0]
        source_url = LLAMA_CPP_SOURCE_TARBALL.format(tag = upstream_tag)
        logger.warning(
            "Unsloth: Falling back to the ggml-org source archive for %s, which "
            "publishes no sha256, so the converter sources cannot be verified.",
            upstream_tag,
        )
    with tempfile.TemporaryDirectory(dir = os.path.dirname(install_folder) or ".") as source_dir:
        archive_path = os.path.join(source_dir, "source.tar.gz")
        _download_archive(source_url, archive_path)
        if expected_sha256:
            actual = _sha256_file(archive_path)
            if actual != expected_sha256:
                raise RuntimeError(
                    f"Unsloth: sha256 mismatch for {fork_source_name}: expected "
                    f"{expected_sha256}, got {actual}"
                )
        extract_dir = os.path.join(source_dir, "extracted")
        os.makedirs(extract_dir)
        _extract_archive(archive_path, extract_dir)
        root = _single_extracted_root(extract_dir)
        converter = os.path.join(root, "convert_hf_to_gguf.py")
        gguf_py = os.path.join(root, "gguf-py")
        if not (os.path.isfile(converter) and os.path.isdir(gguf_py)):
            raise RuntimeError(f"Unsloth: Source tarball for {tag} is missing converter files.")
        shutil.copy2(converter, os.path.join(install_folder, "convert_hf_to_gguf.py"))
        shutil.copytree(gguf_py, os.path.join(install_folder, "gguf-py"), dirs_exist_ok = True)
        conversion = os.path.join(root, "conversion")
        if os.path.isdir(conversion):
            shutil.copytree(conversion, os.path.join(install_folder, "conversion"), dirs_exist_ok = True)


def _write_prebuilt_marker(install_folder, tag, asset_name, repo = "ggml-org/llama.cpp"):
    try:
        info = {
            "source"           : f"{repo} prebuilt release",
            "repo"             : repo,
            "tag"              : tag,
            "asset"            : asset_name,
            "installed_at_utc" : time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(os.path.join(install_folder, UNSLOTH_PREBUILT_INFO_FILENAME), "w", encoding = "utf-8") as f:
            json.dump(info, f, indent = 2)
    except Exception as e:
        logger.warning("Unsloth: Could not write prebuilt marker (%s).", e)


def _stage_prebuilt_install(llama_cpp_folder, tag, asset_name, asset_url, expected_sha256 = None, repo = "ggml-org/llama.cpp", source_assets = None, checksums = None):
    """Download one prebuilt asset, verify, hydrate, validate in staging,
    then activate into llama_cpp_folder. Raises on any failure. source_assets is
    the release's asset map, used so the converter sources hydrate from the fork's
    own source tarball for "mix" tags (see _hydrate_converter_sources)."""
    parent_dir = os.path.dirname(llama_cpp_folder) or "."
    os.makedirs(parent_dir, exist_ok = True)
    # Stage next to the target so activation is an atomic same-fs move,
    # and validate before touching the real folder.
    staging = tempfile.mkdtemp(prefix = ".llama_cpp_prebuilt_", dir = parent_dir)
    try:
        archive_path = os.path.join(staging, asset_name)
        _download_archive(asset_url, archive_path)
        if expected_sha256:
            actual = _sha256_file(archive_path)
            if actual != expected_sha256:
                raise RuntimeError(f"Unsloth: sha256 mismatch for {asset_name}: expected {expected_sha256}, got {actual}")
        extract_dir = os.path.join(staging, "extracted")
        os.makedirs(extract_dir)
        _extract_archive(archive_path, extract_dir)
        staged_install = os.path.join(staging, "install")
        os.makedirs(staged_install)
        _place_prebuilt_binaries(_single_extracted_root(extract_dir), staged_install)
        _hydrate_converter_sources(
            tag, staged_install, source_assets = source_assets, checksums = checksums,
        )
        _write_prebuilt_marker(staged_install, tag, asset_name, repo = repo)
        check_llama_cpp(llama_cpp_folder = staged_install)

        if not os.path.exists(llama_cpp_folder):
            shutil.move(staged_install, llama_cpp_folder)
        else:
            # Folder exists with broken/missing binaries: merge into it
            # rather than deleting a tree the user may own.
            shutil.copytree(staged_install, llama_cpp_folder, dirs_exist_ok = True)
        return check_llama_cpp(llama_cpp_folder = llama_cpp_folder)
    finally:
        shutil.rmtree(staging, ignore_errors = True)


def _install_llama_cpp_prebuilt(llama_cpp_folder, gpu_support = False, print_output = False):
    """Install prebuilt llama.cpp binaries plus same-tag converter sources into
    llama_cpp_folder, always preferring the unslothai/llama.cpp fork. Tries, in
    order: the fork GPU bundle (CUDA/ROCm/Metal) when a GPU target is present, the
    fork CPU bundle (the final prebuilt fallback -- its app-*-cpu archive also ships
    llama-quantize), then ggml-org's upstream CPU build for extra resilience on
    non-macOS hosts. Returns (quantizer, converter) on the first asset that installs,
    else None so the caller compiles from source as before. ggml-org is skipped on
    macOS: its recent CPU build targets a newer macOS and fails to load on 14/15."""
    try:
        is_darwin = platform.system() == "Darwin"
        # Each attempt carries its own (repo, tag, checksums, source_assets) so
        # staging verifies the right sha256 and hydrates the converter from the
        # matching source. (repo, asset_name) dedups the macOS bundle, which both
        # fork selectors return.
        attempts = []          # [(repo, tag, checksums, source_assets, name, url), ...]
        seen = set()           # {(repo, asset_name)}

        def _extend(repo, tag, checksums, source_assets, selected):
            for asset_name, asset_url in selected:
                key = (repo, asset_name)
                if key in seen:
                    continue
                seen.add(key)
                attempts.append((repo, tag, checksums, source_assets, asset_name, asset_url))

        # 1 + 2: unslothai/llama.cpp fork bundles (GPU then CPU). Best-effort: a
        # failed fork release resolution still lets ggml-org be tried below.
        fork_repo = "unslothai/llama.cpp"
        resolved = _resolve_llama_cpp_release(LLAMA_CPP_PUBLISHED_RELEASES_API)
        if resolved is not None:
            fork_tag, fork_assets = resolved
            manifest = _fetch_release_json_asset(fork_assets, LLAMA_CPP_PREBUILT_MANIFEST_ASSET)
            fork_checksums = _fetch_release_json_asset(fork_assets, LLAMA_CPP_PREBUILT_SHA256_ASSET) or {}
            fork_checksums = fork_checksums.get("artifacts", {})
            # 1: GPU bundle, only with a usable GPU target (or macOS Metal).
            # Probe the GPU target once and reuse it inside _select_gpu_assets.
            gpu_target = _detect_gpu_target() if (gpu_support and not is_darwin) else None
            if gpu_support and (is_darwin or gpu_target is not None):
                _extend(fork_repo, fork_tag, fork_checksums, fork_assets,
                        _select_gpu_assets(fork_tag, fork_assets, manifest, target = gpu_target))
            # 2: CPU bundle -- the final prebuilt fallback for CPU-oriented
            # installs. Skipped for an explicit GPU request so a failed GPU
            # prebuilt compiles a GPU-enabled build (the pre-prebuilt behavior)
            # rather than silently landing on a CPU-only prebuilt. macOS export
            # passes gpu_support=False and still gets the right archive: on Darwin
            # the CPU selector returns the same universal macOS/Metal bundle.
            if not gpu_support:
                _extend(fork_repo, fork_tag, fork_checksums, fork_assets,
                        _select_cpu_assets(fork_tag, fork_assets, manifest))
        else:
            logger.warning("Unsloth: Could not resolve a unslothai/llama.cpp release - "
                           "trying upstream ggml-org instead.")

        # 3: ggml-org upstream CPU, non-Darwin CPU installs only (its Darwin CPU
        # build is unusable on macOS 14/15, and a GPU request must not be shadowed
        # by a CPU-only prebuilt -- it falls through to a source GPU build).
        if not is_darwin and not gpu_support:
            ggml_repo = "ggml-org/llama.cpp"
            resolved = _resolve_llama_cpp_release()
            if resolved is not None:
                ggml_tag, ggml_assets = resolved
                selected = _select_prebuilt_asset(ggml_tag, ggml_assets)
                if selected is not None:
                    _extend(ggml_repo, ggml_tag, {}, None, [selected])

        if not attempts:
            logger.warning("Unsloth: No prebuilt llama.cpp bundle matches this host - "
                           "falling back to source build.")
            return None

        for repo, tag, checksums, source_assets, asset_name, asset_url in attempts:
            print(f"Unsloth: Installing prebuilt llama.cpp {tag} ({asset_name}) - skipping compilation.")
            try:
                result = _stage_prebuilt_install(
                    llama_cpp_folder, tag, asset_name, asset_url,
                    expected_sha256 = (checksums.get(asset_name) or {}).get("sha256"),
                    repo = repo,
                    source_assets = source_assets,
                    checksums = checksums,
                )
            except Exception as e:
                logger.warning("Unsloth: Prebuilt %s failed (%s) - trying next option.", asset_name, e)
                continue
            try:
                try_execute(f"{check_pip()} install gguf protobuf sentencepiece mistral_common", print_output = print_output)
            except Exception as e:
                logger.warning("Unsloth: Converter dependency install failed (%s); conversion self-heals if needed.", e)
            return result
        return None
    except Exception as e:
        logger.warning("Unsloth: Prebuilt llama.cpp install failed (%s) - falling back to source build.", e)
        return None


def _maybe_install_llama_cpp_prebuilt(llama_cpp_folder, gpu_support = False, print_output = False):
    """Gate for the prebuilt path; UNSLOTH_LLAMA_FORCE_COMPILE=1 always
    compiles. No exception ever propagates to install_llama_cpp."""
    try:
        if os.environ.get("UNSLOTH_LLAMA_FORCE_COMPILE", "0").lower() in ("1", "true", "yes", "on"):
            return None
        return _install_llama_cpp_prebuilt(llama_cpp_folder, gpu_support = gpu_support, print_output = print_output)
    except Exception as e:
        logger.warning("Unsloth: Prebuilt llama.cpp path errored (%s) - falling back to source build.", e)
        return None


def install_llama_cpp(
    llama_cpp_folder = LLAMA_CPP_DEFAULT_DIR,
    llama_cpp_targets = LLAMA_CPP_TARGETS,
    print_output = False,
    gpu_support = False,
    just_clone_repo = False,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Installs llama.cpp
    quantizer = None
    converter = None

    gpu_support = "ON" if gpu_support else "OFF"

    needs_clone = False
    needs_build = False
    needs_wipe  = False

    # Ensure ~/.unsloth/ exists before we try to use it
    os.makedirs(UNSLOTH_HOME, exist_ok=True)

    # C3: Backward compat -- if using the new default location but CWD has a working ./llama.cpp, use it
    cwd_llama_cpp = os.path.join(os.getcwd(), "llama.cpp")
    if (
        llama_cpp_folder == LLAMA_CPP_DEFAULT_DIR
        and os.path.exists(cwd_llama_cpp)
        and cwd_llama_cpp != os.path.realpath(llama_cpp_folder)
    ):
        try:
            q, c = check_llama_cpp(llama_cpp_folder=cwd_llama_cpp)
            print(
                f"Unsloth: Found existing llama.cpp at `{cwd_llama_cpp}` -- using it.\n"
                f"Unsloth: Note: the default location has moved to `{LLAMA_CPP_DEFAULT_DIR}`."
            )
            return q, c
        except Exception:
            pass  # CWD copy is broken, proceed with default location

    if os.path.exists(llama_cpp_folder):
        # Repo integrity check -- a source checkout has src/ggml/common; a
        # prebuilt install instead carries the UNSLOTH_PREBUILT_INFO.json marker
        required_dirs = ['src', 'ggml', 'common']
        is_source_checkout = all(os.path.isdir(os.path.join(llama_cpp_folder, d)) for d in required_dirs)
        is_prebuilt_install = os.path.isfile(os.path.join(llama_cpp_folder, UNSLOTH_PREBUILT_INFO_FILENAME))
        if not (is_source_checkout or is_prebuilt_install):
            print("Unsloth: llama.cpp repo appears corrupted (missing src/ggml/common) - will re-clone")
            # Deleting is part of installing, so it waits for the opt-out check below.
            # `llama_cpp_folder` can be a custom path holding the user's own files, and
            # wiping it only to then report that installation was declined would be the
            # worst of both outcomes.
            needs_wipe = True
            needs_clone = True
            needs_build = True
        else:
            # Repo is intact -- check for existing binaries
            try:
                quantizer, converter = check_llama_cpp(llama_cpp_folder=llama_cpp_folder)
                # C2: If binaries work, use them directly (no auto-update)
                print(f"Unsloth: llama.cpp found at `{llama_cpp_folder}` -- using existing install.")
                return quantizer, converter
            except Exception:
                print("Unsloth: llama.cpp folder exists but binaries not found - will build")
                needs_build = True
    else:
        needs_clone = True
        needs_build = True
    pass

    # Everything below this point installs something: the prebuilt download, the clone, and
    # do_we_need_sudo, which probes by RUNNING `apt-get update -y` / `pacman -Sy` / a yum
    # check-update and then retrying under sudo. Checking the opt-out only inside
    # install_package left all of that reachable, so UNSLOTH_AUTO_INSTALL=0 still fetched and
    # activated a prebuilt llama.cpp, and still ran a package-manager update as root.
    # An existing install is unaffected: that returns above without reaching here.
    if (needs_build or needs_clone) and not _auto_install_enabled():
        raise RuntimeError(
            "Unsloth: llama.cpp is not installed and automatic installation was declined "
            "(UNSLOTH_AUTO_INSTALL=0)!\n"\
            "Please install llama.cpp manually via https://docs.unsloth.ai/basics/troubleshooting-and-faqs#how-do-i-manually-save-to-gguf"
        )

    if needs_wipe:
        # C4: Only delete if the path is safe
        if _is_safe_to_delete(llama_cpp_folder):
            shutil.rmtree(llama_cpp_folder)
        else:
            raise RuntimeError(
                f"Unsloth: llama.cpp at `{llama_cpp_folder}` appears corrupted but is not in a safe location to delete.\n"
                f"Please manually remove or fix it."
            )

    # Prefer official prebuilt binaries before any source-build work
    # (no system package installs, no clone, no compile).
    if needs_build and not just_clone_repo:
        prebuilt = _maybe_install_llama_cpp_prebuilt(
            llama_cpp_folder,
            gpu_support = (gpu_support == "ON"),
            print_output = print_output,
        )
        if prebuilt is not None:
            return prebuilt

    print_outputs = []
    missing_packages, system_type = check_build_requirements()
    sudo = do_we_need_sudo(system_type)
    kwargs = {"sudo" : sudo, "print_output" : print_output, "print_outputs" : print_outputs, "system_type": system_type}

    if not missing_packages:
        if print_output: print("Unsloth: All required system packages already installed!")
    else:
        packages_to_install = " ".join(missing_packages)
        print(f"Unsloth: Missing packages: {packages_to_install}")
        print(f"Unsloth: Will attempt to install missing system packages.")
        install_package(packages_to_install, sudo, system_type = system_type)

    # Clone repo if needed
    if needs_clone:
        parent_dir = os.path.dirname(llama_cpp_folder)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        print("Unsloth: Cloning llama.cpp repository...")
        # H2: Quote path to handle spaces in directory names
        quoted_folder = shlex.quote(llama_cpp_folder) if not IS_WINDOWS else f'"{llama_cpp_folder}"'
        try_execute_with_auto_install(
            f"git clone https://github.com/ggml-org/llama.cpp {quoted_folder}",
            **kwargs
        )
    pass

    pip = check_pip()

    # Install Python packages (only if not already satisfied)
    try_execute(f"{pip} install gguf protobuf sentencepiece mistral_common", **kwargs)
    if just_clone_repo: return llama_cpp_folder

    if needs_build:
        print("Unsloth: Building llama.cpp - please wait 1 to 3 minutes")
    if gpu_support == "ON":
        print("Unsloth: Building llama.cpp with GPU support")

    build_success = False
    build_errors = []

    # Check for Colab / Kaggle, and deduct some CPUs to conserve memory
    cpu_count = psutil.cpu_count() or 1
    if IS_COLAB_ENVIRONMENT or IS_KAGGLE_ENVIRONMENT:
        cpu_count = cpu_count - 1
        cpu_count = max(cpu_count, 1)

    if IS_WINDOWS:
        # Windows: cmake-only build with Visual Studio generator
        # Aligned with setup.ps1 Phase 4 build logic
        try:
            build_dir = os.path.join(llama_cpp_folder, "build")

            # Clean up any partial build
            if os.path.exists(build_dir):
                shutil.rmtree(build_dir)

            # Detect Visual Studio generator
            cmake_generator, vs_install_path = _find_visual_studio()

            if not cmake_generator:
                raise RuntimeError(
                    "Unsloth: Visual Studio Build Tools not found.\n"
                    "Install via: winget install Microsoft.VisualStudio.2022.BuildTools "
                    '--override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive --wait"'
                )

            # cmake configure
            cmake_args = [
                "cmake", "-S", llama_cpp_folder, "-B", build_dir,
                "-G", cmake_generator,
                "-Wno-dev",
                "-DBUILD_SHARED_LIBS=OFF",
                f"-DGGML_CUDA={gpu_support}",
            ]
            if vs_install_path:
                cmake_args.append(f"-DCMAKE_GENERATOR_INSTANCE={vs_install_path}")

            # Check for OpenSSL (enables HTTPS for llama-server)
            openssl_root = _find_openssl_root()
            if openssl_root:
                cmake_args.extend([
                    f"-DOPENSSL_ROOT_DIR={openssl_root}",
                    "-DLLAMA_OPENSSL=ON",  # Defined in common/CMakeLists.txt
                ])

            if print_output:
                print(f"Unsloth: cmake configure with {cmake_generator}")
                print(f"Unsloth: cmake args: {' '.join(cmake_args)}")

            result = subprocess.run(cmake_args, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"cmake configure failed (exit {result.returncode}):\n"
                    f"{result.stdout}\n{result.stderr}"
                )

            # cmake build
            build_cmd = [
                "cmake", "--build", build_dir, "--config", "Release",
                f"-j{cpu_count}", "--target",
            ] + list(llama_cpp_targets)

            if print_output: print("Unsloth: Building llama.cpp (this may take several minutes)...")

            result = subprocess.run(build_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"cmake build failed (exit {result.returncode}):\n"
                    f"{result.stdout}\n{result.stderr}"
                )

            # On Windows, binaries stay in build/bin/Release/ — no copy needed
            build_success = True
            if print_output: print("Unsloth: Successfully built with cmake (Visual Studio)")

        except Exception as e:
            build_errors.append(f"Windows cmake build failed: {str(e)}")

    else:
        # Linux/macOS: Try make first, then cmake. Modern llama.cpp Makefiles
        # are CMake migration stubs; skip make there so the misleading
        # "Build system changed" error never surfaces (unslothai/unsloth#5832).
        try_make = not _is_cmake_only_llama_cpp(llama_cpp_folder)
        if try_make:
            try:
                if print_output: print("Trying to build with make...")
                try_execute("make clean", cwd = llama_cpp_folder, **kwargs)
                try_execute(f"make all -j{cpu_count}", cwd = llama_cpp_folder, **kwargs)
                build_success = True
                print("Successfully built with make")
            except Exception as e:
                build_errors.append(f"Make failed: {str(e)}")
                if print_output: print(f"Make failed, trying cmake...")
        elif print_output:
            print("CMake-only llama.cpp checkout detected; skipping make...")
        if not build_success:
            # Use cmake instead
            try:
                # Clean up any partial build
                try_execute(f"rm -rf build", cwd = llama_cpp_folder, **kwargs)

                # Build cmake configure command with library detection.
                # Set CMAKE_BUILD_TYPE=Release at configure time: on single-config
                # generators (Unix Makefiles / Ninja on Linux/macOS) the build
                # step's `--config Release` is ignored, so without this the
                # binaries are built unoptimized.
                cmake_configure = (
                    f"cmake . -B build "
                    f"-DCMAKE_BUILD_TYPE=Release "
                    f"-DBUILD_SHARED_LIBS=OFF -DGGML_CUDA={gpu_support}"
                )

                # Detect OpenMP library path (fixes GOMP linker errors)
                gomp_path = _find_lib_path('libgomp.so')
                if gomp_path:
                    cmake_configure += (
                        f" -DOpenMP_C_LIB_NAMES=gomp"
                        f" -DOpenMP_CXX_LIB_NAMES=gomp"
                        f" -DOpenMP_gomp_LIBRARY={gomp_path}"
                    )

                # Detect OpenSSL library paths
                ssl_path = _find_lib_path('libssl.so')
                crypto_path = _find_lib_path('libcrypto.so')
                if ssl_path and crypto_path:
                    cmake_configure += (
                        f" -DOPENSSL_ROOT_DIR=/usr"
                        f" -DOPENSSL_SSL_LIBRARY={ssl_path}"
                        f" -DOPENSSL_CRYPTO_LIBRARY={crypto_path}"
                    )

                # LLAMA_CURL is deprecated upstream (ggml-org/llama.cpp#18791),
                # so we pass ignore_deprecation=True to handle any deprecation warnings.
                try_execute(
                    cmake_configure,
                    cwd = llama_cpp_folder,
                    ignore_deprecation = True,
                    **kwargs
                )
                try_execute(
                    f"cmake --build build --config Release "\
                    f"-j{cpu_count} --clean-first --target "\
                    f"{' '.join(llama_cpp_targets)}",
                    cwd = llama_cpp_folder,
                    **kwargs
                )
                # Move compiled objects to main folder.
                # Remove only the target binaries first to avoid
                # "same file" errors when symlinks point into build/bin/.
                try_execute(
                    "rm -f " + " ".join(llama_cpp_targets) + " && cp build/bin/llama-* .",
                    cwd = llama_cpp_folder,
                    **kwargs
                )
                build_success = True
                # Remove build folder
                try_execute(f"rm -rf build", cwd = llama_cpp_folder, **kwargs)
                if print_output: print("Successfully built with cmake")
            except Exception as e:
                build_errors.append(f"CMake failed: {str(e)}")

    if not build_success:
        error_msg = "=== Unsloth: FAILED building llama.cpp ===\n"
        error_msg += "\n".join(build_errors)
        error_msg += "\n=== Full output log: ===\n"
        error_msg += "".join(print_outputs)
        raise RuntimeError(error_msg)

    # Check if it installed correctly
    try:
        quantizer, converter = check_llama_cpp(llama_cpp_folder)
        print(f"Unsloth: Successfully installed llama.cpp!")
        return quantizer, converter
    except Exception as e:
        raise RuntimeError(
            f"Build appeared to succeed but can't find binaries: {str(e)}\n"
            f"Check the {llama_cpp_folder} directory for compiled binaries."
        )
pass


def _extract_archs_from_monolith_source(source_bytes):
    """Read (text_archs, vision_archs) out of a monolithic convert_hf_to_gguf.py.

    Parsed, not imported: the file is downloaded from llama.cpp master at
    runtime, so importing it would execute whatever the download contained.
    Mirrors _extract_dict_keys_from_conversion_init for the package layout.
    """
    try:
        tree = ast.parse(source_bytes)
    except Exception:
        return set(), set()

    text_archs   = set()
    vision_archs = set()

    def _is_register_call(node):
        # Matches ModelBase.register(...) / Model.register(...) / <X>.register(...)
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register"
            and isinstance(node.func.value, ast.Name)
        )

    def _base_names(class_node):
        names = []
        for base in class_node.bases:
            if   isinstance(base, ast.Attribute): names.append(base.attr)
            elif isinstance(base, ast.Name):      names.append(base.id)
        return names

    # class -> bases, so a class two hops below MmprojModel still counts as vision.
    class_bases = {
        node.name : _base_names(node)
        for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }

    def _inherits_mmproj(class_name, _seen = None):
        if _seen is None: _seen = set()
        if class_name in _seen: return False
        _seen.add(class_name)
        for base in class_bases.get(class_name, []):
            if base.lower() in ("mmprojmodel", "visionmodel"): return True
            if _inherits_mmproj(base, _seen): return True
        return False

    def _is_mmproj(class_node, call_node):
        for keyword in call_node.keywords:
            if keyword.arg != "model_type": continue
            value = keyword.value
            # model_type=ModelType.MMPROJ, or a bare MMPROJ / "mmproj"
            name = None
            if isinstance(value, ast.Attribute): name = value.attr
            elif isinstance(value, ast.Name): name = value.id
            elif isinstance(value, ast.Constant) and isinstance(value.value, str):
                name = value.value
            # An explicit model_type wins over the base classes.
            if name is not None: return "mmproj" in name.lower()
        return _inherits_mmproj(class_node.name)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef): continue
        for decorator in node.decorator_list:
            if not _is_register_call(decorator): continue
            names = [
                arg.value for arg in decorator.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            ]
            if not names: continue
            if _is_mmproj(node, decorator): vision_archs.update(names)
            else: text_archs.update(names)

    # A converter may seed `_model_classes` literally instead of decorating. The
    # import path saw those entries, so harvest them rather than report nothing.
    def _bucket_for(key_node):
        name = None
        if   isinstance(key_node, ast.Attribute): name = key_node.attr
        elif isinstance(key_node, ast.Name):      name = key_node.id
        elif isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            name = key_node.value
        if name is None: return None
        return vision_archs if "mmproj" in name.lower() else text_archs

    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else \
                  [node.target] if isinstance(node, ast.AnnAssign) else []
        named = any(
            (isinstance(t, ast.Name) and t.id == "_model_classes") or
            (isinstance(t, ast.Attribute) and t.attr == "_model_classes")
            for t in targets
        )
        if not named or not isinstance(node.value, ast.Dict): continue
        for key, value in zip(node.value.keys, node.value.values):
            bucket = _bucket_for(key)
            if bucket is None or not isinstance(value, ast.Dict): continue
            bucket.update(
                k.value for k in value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            )
    return text_archs, vision_archs
pass


_UNSLOTH_BRANDING_MARKER = b"# UNSLOTH_BRANDING_APPLIED"

# Idempotency marker for the monolith branding patch: it edits the in-memory
# entrypoint, so it has nowhere to put a marker comment of its own.
_UNSLOTH_BRANDING_LINE = b"self.metadata.quantized_by = 'Unsloth'"

# Same for the gguf attribute guard patch.
_GGUF_GUARD_LINE = b"except AttributeError: gguf."
# Reads back the names already guarded. Anchored on the `except` half: `try: gguf.X`
# is also what the reference scan looks for, so a pattern matching both could not
# tell "already guarded" from "needs guarding".
_GGUF_GUARD_PATTERN = re.compile(rb"except AttributeError: gguf\.([\.A-Z_0-9]{3,}) = None")
# The whole guard, so the reference scan can run with previous guards stripped out.
# Otherwise it finds `gguf.X` inside its own guards and re-guards every name.
_GGUF_GUARD_BLOCK_PATTERN = re.compile(
    rb"try: gguf\.[\.A-Z_0-9]{3,}\r?\n?except AttributeError: gguf\.[\.A-Z_0-9]{3,} = None\r?\n?"
)
_BRANDING_PATTERN = re.compile(
    rb"(self\.metadata \= gguf\.Metadata\.load\(.+?\))([\n\r]+([\s\t]{4,}))",
    flags = re.MULTILINE,
)


def _get_llama_cpp_dir(local_script_info):
    """Directory holding the converter being patched: UNSLOTH_LLAMA_CPP_SCRIPTS_DIR
    when set, else ~/.unsloth/llama.cpp. Single anchor for layout detection,
    branding patch, Qwen check, and sibling-info cache key."""
    if local_script_info is not None:
        return os.path.dirname(local_script_info[0])
    return LLAMA_CPP_DEFAULT_DIR
pass


# Enough to cover the package a converter imports without walking a tree an
# attacker chooses the size of. Measured against llama.cpp master: conversion/
# holds 94 modules today, so the original 64 refused the real package on every
# export and, under UNSLOTH_CONVERTER_SCAN_STRICT, refused the export itself.
# A cap below the thing it is meant to read is not a safety margin.
MAX_CONVERSION_PACKAGE_FILES = 256
# The same bound on the traversal itself. Counting only .py files bounded what
# gets READ but not what gets WALKED: a tree with sixty modules and a million
# empty directories never reaches the file cap, so the whole attacker-sized
# directory was still traversed, once per export through the cache key. Generous
# next to any real converter package, which is dozens of entries.
MAX_CONVERSION_PACKAGE_ENTRIES = 4096
# And a bound on how many places get that allowance. Every root directory became
# a location of its own, with no limit on how many there could be, so a tree with
# thousands of top-level directories multiplied the per-location budget by
# thousands. llama.cpp master has about twenty. Crossing this is not reported
# separately: a root that wide trips the root location's own entry budget, which
# already says the tree is too big to read.
MAX_SCAN_LOCATIONS = 64

# The patched converter this module writes, which for the package layout lands in
# the llama.cpp root. It is this scan's own output, not an input: keying on it
# meant every cache miss rewrote it, changed its mtime, and missed again on the
# next call, so with the scan disabled each export re-fetched and re-patched the
# converter forever. Excluded by exact name and only where it is written, never
# by prefix: matching a prefix in every directory meant a module named
# conversion/unsloth_convert_hf_to_gguf_payload.py was skipped by the scan and
# the key both, which a clean __init__.py could then import and run.
GENERATED_CONVERTER_NAME = "unsloth_convert_hf_to_gguf.py"

# No single module is read whole. The converter's own entrypoint is tens of KB and
# the scanner caps its input at 8 MiB anyway, so a module past this is one the
# scan could not have judged in full regardless; reading it to find that out is
# how an unverified package with one enormous or sparse module exhausted memory
# before any finding was produced.
MAX_MODULE_BYTES = 8 * 1024 * 1024
MODULE_READ_CHUNK = 1024 * 1024

# Everything beside the entrypoint that the entrypoint puts on sys.path and
# imports, and so executes with exactly the privileges of the converter itself.
# gguf-py is copied out of the same source tarball as conversion/, and that
# tarball arrives with no digest at all, so scanning only conversion/ left a
# payload in gguf-py/gguf/__init__.py running under a clean entrypoint and a
# clean conversion/, strict mode included.
IMPORTED_PACKAGE_SUBDIRS = ("conversion", "gguf-py/gguf")

# The directories that end up on sys.path: the converter's own, which Python puts
# at index 0 for the subprocess, and gguf-py, which the entrypoint inserts itself.
# Anything directly under either is importable by its own name.
IMPORT_ROOTS = (".", "gguf-py")
# The packages imported by name. Only in these is a native file a Python module
# rather than one of the prebuilt bundle's runtime libraries.
NAMED_IMPORT_PACKAGES = frozenset(IMPORTED_PACKAGE_SUBDIRS)

# Files Python will execute that a source scan cannot read. Native extensions
# are dlopened; a .pyc in a module's own place, with no source beside it, is a
# sourceless import and CPython runs the bytecode as the module.
NATIVE_MODULE_SUFFIXES = (".so", ".pyd", ".dll", ".dylib")
BYTECODE_SUFFIXES = (".pyc", ".pyo")
COLLECTED_MODULE_SUFFIXES = (".py",) + BYTECODE_SUFFIXES + NATIVE_MODULE_SUFFIXES


def _purge_regenerable_bytecode(
    package_dir, entry_limit = None, recursive = True, boundary = None,
):
    """Delete bytecode caches that came with an unverified package.

    CPython's cache validation proves that a .pyc CLAIMS to belong to the source
    beside it, not that its bytecode was compiled from that source: a timestamp
    cache is accepted when the source's mtime and size match the header, and a
    checked-hash cache when the source hashes to the value in the header. Anyone
    who ships both files sets both, so a clean .py can be paired with arbitrary
    marshalled code and that code is what runs. Verified directly: a
    timestamp-validated cache whose source reads `VALUE = "clean"` imported as
    `PWNED`. So the invalidation mode says nothing about trust, and an earlier
    version of this file was wrong to read it that way.

    Deleting is better than reporting for any cache that has a source, because
    Python simply rebuilds it from the .py this scan did read: nothing is lost,
    nothing legitimate breaks, and an ordinary export's own caches are removed
    and rebuilt without a word. A sourceless .pyc is left alone and reported
    instead, since deleting that one would break a package that needs it.

    Returns the caches it could not remove, which are reported like any other
    file Python will execute and this scan cannot read.
    """
    stuck = []
    entries = 0
    pending = [package_dir]
    seen_directories = set()
    while pending:
        root = pending.pop()
        identity = _directory_identity(root)
        if identity is not None:
            if identity in seen_directories:
                continue
            seen_directories.add(identity)
        try:
            with os.scandir(root) as scanner:
                for entry in scanner:
                    entries += 1
                    if entry_limit is not None and entries >= entry_limit:
                        return stuck
                    try:
                        if entry.is_dir():
                            # Containment, unlike the scan: the scan FOLLOWS a
                            # symlink out of the tree because the converter's
                            # import would, and reading is harmless. Deleting is
                            # not. A checkout can carry a symlink pointing at an
                            # unrelated directory, and removing caches there would
                            # rewrite something that has nothing to do with this
                            # export, before strict mode ever gets to refuse.
                            if not _stays_within(boundary or package_dir, entry.path):
                                continue
                            if recursive:
                                pending.append(entry.path)
                            elif entry.name == "__pycache__" and root == package_dir:
                                # As above: the cache belonging to this directory.
                                pending.append(entry.path)
                            continue
                    except OSError:
                        continue
                    name = entry.name
                    if os.path.splitext(name)[1].lower() not in BYTECODE_SUFFIXES:
                        continue
                    if not _has_a_source(root, name):
                        continue        # sourceless: reported, not removed
                    try:
                        os.remove(entry.path)
                    except OSError:
                        stuck.append(
                            os.path.relpath(entry.path, package_dir).replace(os.sep, "/")
                        )
        except OSError:
            continue
    return stuck


# Named for the same reason PackageWalk is: this grew a field and a positional
# read of the old shape would have been silently wrong.
ScanLocation = collections.namedtuple("ScanLocation", "label path recursive natives skip")
# Named too: `truncated` has to reach the caller, and a bare list could not say it.
ScanPlan = collections.namedtuple("ScanPlan", "locations truncated")


def _imported_top_level_names(directory, only = None, recursive = False):
    """The top-level names imported by the modules under `directory`.

    `only` names the files to read, or None for every module. `recursive` walks
    nested packages too, because `conversion/__init__.py` can import
    `conversion.nested.mod` and that module's own imports are just as much a part
    of what the converter runs.

    Returns None when nothing here could be parsed, which the caller reads as "do
    not narrow anything on the strength of this": a syntax error must not be a way
    to choose what gets looked at.
    """
    names, parsed_any, read = set(), False, 0
    pending = [directory]
    seen = set()
    # ONE budget for the whole traversal, not one per directory. Counting per
    # directory bounded nothing: `read` advances only for .py files, so a tree
    # that branches without holding any modules was walked in full. A checkout
    # with 14400 empty directories, three and a half times this budget, was
    # traversed entirely, and nothing about that shape is hard to build at a
    # scale that stalls the export before strict mode can refuse the checkout.
    examined = 0
    while pending and read < MAX_CONVERSION_PACKAGE_FILES:
        if examined > MAX_CONVERSION_PACKAGE_ENTRIES:
            # An early exit, not the bound. The check inside the scandir loop is
            # what stops the walk; without this the queue would still drain, one
            # scandir per directory that breaks on its first entry.
            break
        current = pending.pop()
        identity = _directory_identity(current)
        if identity is not None:
            if identity in seen:
                continue
            seen.add(identity)
        try:
            with os.scandir(current) as scanner:
                for entry in scanner:
                    # Iterated lazily and counted, not materialized: a checkout
                    # with millions of entries in one directory would otherwise
                    # exhaust memory here, before any budget was consulted, and
                    # this runs on a tree nothing has verified yet.
                    examined += 1
                    if examined > MAX_CONVERSION_PACKAGE_ENTRIES:
                        break
                    if read >= MAX_CONVERSION_PACKAGE_FILES:
                        break
                    try:
                        if recursive and entry.is_dir():
                            if entry.name != "__pycache__":
                                pending.append(entry.path)
                            continue
                    except OSError:
                        continue
                    if not entry.name.endswith(".py"):
                        continue
                    if only is not None and entry.name not in only:
                        continue
                    read += 1
                    try:
                        with open(entry.path, "rb") as handle:
                            source = handle.read(MAX_MODULE_BYTES)
                        tree = ast.parse(source, entry.path)
                    except (OSError, SyntaxError, ValueError, RecursionError, MemoryError):
                        continue
                    parsed_any = True
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            names.update(alias.name.split(".")[0] for alias in node.names)
                        elif (
                            isinstance(node, ast.ImportFrom)
                            and node.module and not node.level
                        ):
                            names.add(node.module.split(".")[0])
        except OSError:
            continue
    return None if not parsed_any else names


def _scanned_locations(llama_cpp_dir):
    """Every place the converter can import from, as a ScanPlan of ScanLocations.

    Two directories go on sys.path: the one the converter script sits in, which
    Python puts at index 0 for the subprocess, and gguf-py, which the entrypoint
    inserts at index 1 itself. Both are IMPORT ROOTS, meaning anything directly
    under them is importable by its own name. Scanning gguf-py/gguf while leaving
    the rest of gguf-py alone left `gguf-py/payload.py` importable as `payload`
    from a clean-looking gguf/__init__.py, and nothing read it.

    Each import root contributes its own modules, walked WITHOUT recursion, and
    each directory under it as a location of its own. The root's subdirectories
    are the rest of the llama.cpp checkout, which is enormous and not importable
    as one thing; walking it whole would cross every budget on an ordinary
    install and refuse the export. A directory that is itself an import root is
    not also taken as a child of another, so gguf-py is walked once.

    `natives` is true only for the packages the converter imports by name. A
    prebuilt install copies the bundle's own .so and .dylib into the root and
    beside it, so calling those Python modules refused every ordinary install.

    One list, used by the scan, the bytecode purge and the cache key alike.
    Keeping two of these in step by hand is what left gguf-py scanned but
    unkeyed earlier in this branch.
    """
    if not llama_cpp_dir or not os.path.isdir(llama_cpp_dir):
        return ScanPlan([], False)

    def _path_of(label):
        return llama_cpp_dir if label == "." else os.path.join(llama_cpp_dir, *label.split("/"))

    locations, truncated = [], False
    root_labels = [label for label in IMPORT_ROOTS if os.path.isdir(_path_of(label))]
    # The converter's own sources, not every script in the checkout. Reading the
    # root wholesale put `examples` back in the scan set, because llama.cpp's
    # unrelated convert_llama_ggml_to_gguf.py imports examples.convert_legacy_llama
    # and Unsloth never runs that script.
    # Seeded from the entrypoint AND the packages it imports. Seeding from the
    # entrypoint alone looks equivalent, since conversion/ and gguf-py/gguf are
    # taken by name below and feed their own imports back, but it is not: a
    # checkout whose entrypoint is missing, renamed or unparseable then yields
    # nothing at all, and nothing means "widen to every directory". Reading the
    # packages too is what keeps such a checkout narrow.
    seeds = [(llama_cpp_dir, LLAMA_CPP_CONVERTER_FILENAMES + (GENERATED_CONVERTER_NAME,))]
    seeds += [
        (_path_of(label), None) for label in IMPORTED_PACKAGE_SUBDIRS
        if os.path.isdir(_path_of(label))
    ]
    imported_names = set()
    unreadable = True
    walked = set()
    for path, only in seeds:
        # Recursively for the packages, and remembered: these are admitted below
        # by name and their imports collected again there, and parsing each of
        # them twice was 250 ms of the 560 ms this took per export on a checkout
        # the size of llama.cpp master.
        recursive = only is None
        found = _imported_top_level_names(path, only = only, recursive = recursive)
        if recursive:
            walked.add(os.path.realpath(path))
        if found is None:
            continue
        unreadable = False
        imported_names |= found
    if unreadable:
        # Nothing could be parsed, so there is no closure to narrow by and every
        # directory is a candidate again.
        imported_names = None
    for label in root_labels:
        locations.append(
            ScanLocation(
                label,
                _path_of(label),
                False,
                False,
                (GENERATED_CONVERTER_NAME,) if label == "." else (),
            )
        )
    # Every child directory of every import root, as candidates. Collected first
    # and then filtered in passes, because the closure grows as it is walked: a
    # directory admitted late can import the name of one passed over early, and a
    # single scandir pass would have already skipped it. conversion/nested/mod.py
    # importing a root `payload` package is exactly that shape.
    candidates = []
    for label in root_labels:
        try:
            with os.scandir(_path_of(label)) as scanner:
                examined = 0
                for entry in scanner:
                    # Counted as they arrive. Collecting every child first and
                    # capping afterwards meant a downloaded tree with millions of
                    # entries in its root exhausted memory before the cap, the
                    # truncation finding or a strict-mode refusal could say
                    # anything at all.
                    examined += 1
                    if examined > MAX_CONVERSION_PACKAGE_ENTRIES:
                        truncated = True
                        break
                    child = entry.name if label == "." else f"{label}/{entry.name}"
                    if child in root_labels or entry.name == "__pycache__":
                        # An import root is walked as itself, and a cache belongs
                        # to the directory whose modules it holds.
                        continue
                    try:
                        if not entry.is_dir():
                            continue
                    except OSError:
                        continue
                    candidates.append((child, entry.name, entry.path))
        except OSError:
            continue

    # A directory is scanned when the converter imports its name. Scanning every
    # directory in the checkout instead read files no import can reach, and
    # measured against a real clone of llama.cpp master that was not a
    # theoretical cost: scripts/server-bench.py polls a /health endpoint in a
    # while loop and examples/llama-eval/llama-eval.py spawns a process, so the
    # scan reported both on every export and UNSLOTH_CONVERTER_SCAN_STRICT
    # refused the export outright. A control that rejects every clean upstream
    # checkout is not a control.
    #
    # Name-matched rather than skipped, because a directory called gguf or torch
    # beside the converter SHADOWS the real package for the subprocess:
    # llama.cpp's own directory is sys.path[0] there, so it wins over
    # site-packages. That is the case these directories were added for, and it is
    # the one kept.
    taken = set()
    progressed = True
    while progressed and not truncated:
        progressed = False
        for child, name, path in candidates:
            if child in taken:
                continue
            if child not in NAMED_IMPORT_PACKAGES and (
                imported_names is not None and name not in imported_names
            ):
                continue
            taken.add(child)
            progressed = True
            locations.append(
                ScanLocation(child, path, True, child in NAMED_IMPORT_PACKAGES, ())
            )
            if imported_names is not None and os.path.realpath(path) not in walked:
                # This directory is part of what runs, so what IT imports is too.
                walked.add(os.path.realpath(path))
                reached = _imported_top_level_names(path, recursive = True)
                if reached:
                    imported_names |= reached
            if len(locations) > MAX_SCAN_LOCATIONS:
                # One past, then trim: stopping AT the cap called a root of
                # exactly that many directories truncated and refused it under
                # strict mode, the same off-by-one the file and entry budgets
                # avoid by asking for one more than they keep. Reported rather
                # than dropped quietly, because a root wide enough to reach this
                # is nowhere near the entry budget that would otherwise have said
                # something.
                truncated = True
                locations = locations[:MAX_SCAN_LOCATIONS]
                break
    return ScanPlan(locations, truncated)


def _purge_imported_package_bytecode(llama_cpp_dir, is_local_copy = False):
    """Purge every imported package's supplied bytecode, and say what is stuck.

    Called before the patcher cache is consulted, not only inside it. The purge
    used to run within the cached function, so once strict mode had completed one
    clean export in a long-lived process, dropping a fresh valid .pyc beside an
    unchanged source left every component of the key identical: the cached result
    came back, nothing purged it, and the next converter subprocess executed it.

    The names it could not delete travel into the key, so bytecode that appears
    and cannot be removed re-runs the scan that reports it, rather than hiding
    behind a cache entry made when the tree was clean.
    """
    if not llama_cpp_dir:
        return ()
    if scan_is_disabled():
        # The opt-out means this does nothing, and deleting files inside a
        # checkout the user pinned is the last thing it should still be doing.
        return ()
    if is_local_copy:
        # A pin says "I chose this directory". Reporting what is in it is fair;
        # rewriting it is not, and a checkout someone works in has caches of its
        # own that are none of this scan's business.
        return ()
    stuck = []
    for location in _scanned_locations(llama_cpp_dir).locations:
        stuck.extend(
            f"{location.label}/{name}"
            for name in _purge_regenerable_bytecode(
                location.path,
                entry_limit = MAX_CONVERSION_PACKAGE_ENTRIES + 1,
                recursive = location.recursive,
                # The checkout, not this location. _scanned_locations promotes a
                # top-level directory symlink to a location of its own, and
                # containment measured from there calls the symlink's target
                # internal: a checkout carrying a link to someone's source tree
                # had that tree's caches deleted.
                boundary = llama_cpp_dir,
            )
        )
    return tuple(sorted(stuck))


def _bytecode_matches_its_source(cache_path, source_path):
    """Whether this cache really is the compiled form of that source.

    Only asked when the cache is being left in place, which is what happens for a
    checkout the user pinned. "A cache exists" is not a finding there: an ordinary
    working tree has one per module, and warning about all of them on every export
    is the false positive this module says is worse than the warning is a win.
    Measured on a tree shaped like llama.cpp master, that was seven warnings
    naming 215 files, every time.

    "This cache disagrees with the source beside it" IS a finding, and it is the
    whole of the attack: CPython checks the header against the source and never
    checks that the bytecode came from it. Compiling the source and comparing is
    exact for the interpreter that will run the converter, which is this one.
    Anything unreadable or unparseable counts as disagreement, since then nothing
    here can say it agrees.
    """
    try:
        with open(cache_path, "rb") as handle:
            cached = handle.read(MAX_MODULE_BYTES + 1)
        with open(source_path, "rb") as handle:
            source = handle.read(MAX_MODULE_BYTES + 1)
    except OSError:
        return False
    if len(cached) <= 16:
        return False
    if cached[:4] != importlib.util.MAGIC_NUMBER:
        # Built by a different Python. This interpreter is the one that runs the
        # converter and it will never load this file, so it is not what executes
        # and reporting it would warn about every checkout used with two Pythons.
        return True
    payload = cached[16:]
    try:
        # dont_inherit, as importlib's own loader compiles: otherwise the future
        # flags in effect wherever this is called from land in the code object
        # and every cache reads as a mismatch.
        code = compile(source, source_path, "exec", dont_inherit = True)
        if payload == marshal.dumps(code):
            return True
        # marshal tags the outermost object with FLAG_REF only when its refcount
        # is above one at dump time, and that tag renumbers every reference after
        # it. So the same code marshals to two byte strings depending on whether
        # the writer was holding it: importlib was, the line above is, a writer
        # that dumps the result of a call was not. Recompiling into that second
        # form is what keeps this a comparison of the code rather than of who
        # produced it. Done only on mismatch, which is the rare path.
        del code
        return payload == marshal.dumps(
            compile(source, source_path, "exec", dont_inherit = True)
        )
    except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError):
        return False


def _stays_within(root, path):
    """Whether `path` resolves to somewhere still under `root`."""
    try:
        resolved_root = os.path.realpath(root)
        resolved = os.path.realpath(path)
    except OSError:
        return False
    return (
        resolved == resolved_root
        or resolved.startswith(resolved_root.rstrip(os.sep) + os.sep)
    )


def _source_beside(root, name):
    """The .py this cache would be rebuilt from, or None when there is none."""
    if os.path.basename(root) == "__pycache__":
        # base.cpython-313.pyc -> ../base.py
        candidate = os.path.join(os.path.dirname(root), name.split(".")[0] + ".py")
    else:
        suffix = os.path.splitext(name)[1]
        candidate = os.path.join(root, name[: -len(suffix)] + ".py")
    return candidate if os.path.isfile(candidate) else None


def _has_a_source(root, name):
    """Whether this cache has a .py beside it that Python can rebuild it from."""
    return _source_beside(root, name) is not None


def _counts_as_a_module(root, name, purged = True, verify = True):
    """Whether this file is one the scan has to account for.

    A .py is read; a native extension and a sourceless .pyc cannot be read and so
    are reported. Everything else here is a bytecode cache of a .py that is
    already counted, and those must NOT count: an ordinary export leaves one per
    module behind, so counting them made a package of 40 modules with 25 caches
    read as 65 files, over the cap. That is a false security warning in advisory
    mode and, under UNSLOTH_CONVERTER_SCAN_STRICT, a refusal of every export
    after the first, which is the one outcome this scan must never produce.

    Decided here rather than after the walk because the cap stops the walk, so a
    file that does not count must not consume the budget either.
    """
    suffix = os.path.splitext(name)[1].lower()
    if suffix == ".py" or suffix in NATIVE_MODULE_SUFFIXES:
        return True
    if suffix not in BYTECODE_SUFFIXES:
        return False
    # A cache with a source is deleted before the converter runs and rebuilt by
    # Python from the .py this scan read, so it is not a module of its own. One
    # without a source is, and gets reported.
    #
    # Unless nothing was deleted. For a pinned checkout this scan does not touch
    # the user's files, and a cache left in place executes instead of the source
    # beside it, whatever that source says. Reported then, but only when it
    # actually disagrees with that source: every working tree carries one cache
    # per module, and reporting those was seven warnings naming 215 files on
    # every export of a tree shaped like llama.cpp master.
    source = _source_beside(root, name)
    if source is None:
        if os.path.basename(root) == "__pycache__":
            # Not importable, so not code that runs. CPython loads a sourceless
            # cache only from the legacy location, mod.pyc in the package
            # directory itself; a __pycache__ entry whose .py is gone is dead
            # weight an upstream update leaves behind. Verified directly: with
            # the source deleted, importing it raises ModuleNotFoundError, while
            # the same bytes copied to the legacy path import and run.
            return False
        return True                 # nothing could rebuild it: reported
    if purged:
        return False                # deleted before the converter runs
    if not verify:
        # The cache key asks a different question: not "is this a finding" but
        # "has anything the converter will execute changed since the last scan".
        # Every cache a pin keeps answers that one, and answering it by compiling
        # each source would put the scan's cost on every export instead of on
        # the cache miss.
        return True
    # Left in place, so it is what executes. Reported only when it disagrees with
    # the source that was scanned, which is the difference between a finding and
    # an ordinary working tree.
    return not _bytecode_matches_its_source(os.path.join(root, name), source)


def _unscannable_modules(package_dir, names, natives = True):
    """The collected names Python can execute and `warn_on_suspicious_converter` cannot read.

    Native extensions, and bytecode with no source to rebuild it from. A cache
    that HAS a source never reaches here: _purge_regenerable_bytecode removes it
    before the converter runs, and anything it could not remove is reported by
    that function instead.
    """
    found = []
    for name in names:
        suffix = os.path.splitext(name)[1].lower()
        if suffix in NATIVE_MODULE_SUFFIXES:
            if natives:
                found.append(name)
        elif suffix in BYTECODE_SUFFIXES:
            found.append(name)
    return found


def _conversion_sibling_info(llama_cpp_dir, is_local_copy = False):
    """Hashable (path, digest) pairs for EVERY module the converter imports,
    folded into the patcher cache key so re-pulled checkouts re-patch and
    re-scan. None only when there is nothing on disk to look at.

    Covers the same places the scan does, the converter's own directory included.
    A monolith checkout used to key on nothing, which is exactly where a module
    that shadows one of the converter's imports would sit.

    Every package in IMPORTED_PACKAGE_SUBDIRS, not just conversion/: the key is
    what decides whether the scan runs again, so a package it does not cover is
    one a long-lived process will re-import without rescanning.

    Every module, not the three the patcher edits. The key also decides whether
    _scan_conversion_package runs again, and that reads the whole directory: with
    only __init__.py, base.py and qwen.py in the key, changing any other module
    in a long-lived process left the key identical, so the next export returned
    the cached converter without rescanning and then executed the changed file.
    The same MAX_CONVERSION_PACKAGE_FILES cap applies, for the same reason it
    applies there, and the count travels in the key so crossing the cap is itself
    a change. The count saturates one past the cap, because the walk stops there;
    above the cap the package is reported as unread on every pass that scans it,
    so the key is not what protects anything there and buying a finer count would
    cost an unbounded walk of a directory an attacker sized."""
    # Every directory that exists is keyed, on the layout test or not. Keying
    # conversion/ only when it holds __init__.py AND base.py was a hole of the
    # same shape as the one that used to skip gguf-py on a monolith: the scan
    # reads the directory either way, because the converter can import from it
    # either way, so a conversion/ without base.py was scanned once and then
    # recorded as an empty header with no digests at all. After the first export
    # a changed module there left the key identical, the patcher came back from
    # cache, and the subprocess imported the new bytes unscanned, strict mode
    # included.
    # With the scan off, the digest buys nothing: it exists to decide whether to
    # re-scan, and there is no scan. The key still has to move when a checkout is
    # re-pulled, so it falls back to the (mtime, size) identity this used before
    # digests, which is the cheap half of the same question. Flipping the switch
    # changes the key regardless, because _converter_scan_mode is in it too.
    _scan_off = scan_is_disabled()

    def _identity(p):
        if _scan_off:
            try:
                stat = os.stat(p)
                return (p, stat.st_size, stat.st_mtime_ns)
            except OSError:
                return (p, -1, 0)
        # The bytes, not (mtime, size): a module replaced with same-sized content
        # under a preserved mtime left the key identical, so the next export
        # returned the cached converter without rescanning and the subprocess
        # imported bytes nothing had read. That is a metadata copy away from
        # deliberate, and free on a coarse-timestamp filesystem. Hashing at most
        # MAX_CONVERSION_PACKAGE_FILES small modules is nothing beside the export
        # this key gates, and the scan reads the same bytes anyway.
        try:
            size = os.path.getsize(p)
            digest = hashlib.sha256()
            read = 0
            with open(p, "rb") as handle:
                while read < MAX_MODULE_BYTES:
                    chunk = handle.read(min(MODULE_READ_CHUNK, MAX_MODULE_BYTES - read))
                    if not chunk:
                        break
                    digest.update(chunk)
                    read += len(chunk)
            # The size travels beside the digest of the prefix, so a file too big
            # for this scan to read still moves the key when it changes length.
            return (p, size, digest.hexdigest())
        except OSError:
            return (p, -1, "")
    header, entries = [], []
    found_any = False
    for location in _scanned_locations(llama_cpp_dir).locations:
        subdir, package_dir, recursive = location.label, location.path, location.recursive
        found_any = True
        walk = _conversion_package_modules(
            package_dir,
            file_limit = MAX_CONVERSION_PACKAGE_FILES + 1,
            entry_limit = MAX_CONVERSION_PACKAGE_ENTRIES + 1,
            recursive = recursive,
            skip_names = location.skip,
            # A pin's caches are deliberately left in place, so they are part of
            # what the next converter subprocess executes. Left out of the key,
            # a cache that changed after the first export returned the patcher
            # from cache, skipped the scan, and ran unreported. Not verified
            # here, only identified: the key has to move when they change, and
            # deciding whether a change is a finding is the scan's job. Both
            # answer the question, but this walk runs on every export, and on a
            # tree shaped like llama.cpp master verifying costs 59ms against 6ms
            # for identifying.
            purged = not is_local_copy,
            verify = False,
        )
        names = walk.names if walk.names is not None else ["__init__.py", "base.py"]
        # The unreadable directories travel too: one becoming readable, or a new
        # one appearing, changes what this scan can say and so has to re-run it.
        header.append((subdir, len(names), walk.complete, walk.unreadable))
        entries.extend(
            _identity(os.path.join(package_dir, *name.split("/")))
            for name in names[:MAX_CONVERSION_PACKAGE_FILES]
        )
    if not found_any:
        # Nothing on disk at all, not even a directory to look in.
        return None
    # The header is ONE element, however much it comes to carry: callers read the
    # per-module entries as info[1:], so widening it in place silently fed them a
    # count or a flag where they expected a (path, digest) pair.
    return (tuple(header),) + tuple(entries)
pass


def _detect_converter_layout(entry_content_bytes, llama_cpp_dir):
    """Return 'package' for the new conversion/ layout, else 'monolith'.
    Structural: entrypoint must contain `from conversion import` AND
    conversion/__init__.py + conversion/base.py must exist on disk."""
    try:
        if b"from conversion import" not in entry_content_bytes:
            return "monolith"
        init_py = os.path.join(llama_cpp_dir, "conversion", "__init__.py")
        base_py = os.path.join(llama_cpp_dir, "conversion", "base.py")
        if os.path.isfile(init_py) and os.path.isfile(base_py):
            return "package"
    except Exception:
        # Detection is best-effort; on any I/O or attribute error fall back
        # to monolith so the legacy regex patches still run.
        pass
    return "monolith"
pass


def _extract_dict_keys_from_conversion_init(conv_init_path, dict_name):
    """AST-parse conversion/__init__.py for TEXT_MODEL_MAP / MMPROJ_MODEL_MAP
    keys. Used as the arch allowlist on the new layout, where
    ModelBase._model_classes is empty until load_all_models() runs."""
    try:
        with open(conv_init_path, "rb") as f:
            tree = ast.parse(f.read())
    except Exception:
        return set()
    keys = set()
    def _harvest(value):
        if isinstance(value, ast.Dict):
            for k in value.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == dict_name:
                    _harvest(node.value)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == dict_name:
                _harvest(node.value)
    return keys
pass


def _dominant_newline(content):
    """The line ending the file mostly uses, b"\r\n" or b"\n".

    The patches below insert whole lines, so a CRLF checkout must stay CRLF rather
    than come out mixed. A lone CR is not a line ending in Python 3, so counting
    is enough."""
    # All Unsloth Zoo code licensed under LGPLv3
    crlf = content.count(b"\r\n")
    if crlf == 0: return b"\n"
    return b"\r\n" if crlf >= (content.count(b"\n") - crlf) else b"\n"
pass


def _apply_branding_patch_to_base(conv_base_path):
    """Insert Unsloth metadata branding after `self.metadata = gguf.Metadata.load(...)`
    in conversion/base.py (idempotent via a one-line marker).
    Returns 'applied' / 'already-applied' / 'pattern-missing'."""
    try:
        with open(conv_base_path, "rb") as f:
            content = f.read()
    except OSError:
        return "pattern-missing"
    if _UNSLOTH_BRANDING_MARKER in content:
        return "already-applied"

    eol = _dominant_newline(content)

    def _replace(match):
        load_call = match.group(1)
        suffix    = match.group(2)   # already starts with newline + indent
        indent    = match.group(3)
        return (
            load_call + eol
            + indent + _UNSLOTH_BRANDING_MARKER + eol
            + indent + b"if hasattr(self.metadata, 'quantized_by'): self.metadata.quantized_by = 'Unsloth'" + eol
            + indent + b"if hasattr(self.metadata, 'repo_url'): self.metadata.repo_url = 'https://huggingface.co/unsloth'" + eol
            + indent + b"if hasattr(self.metadata, 'tags'): self.metadata.tags = ['unsloth', 'llama.cpp']"
            + suffix
        )

    new_content, n = _BRANDING_PATTERN.subn(_replace, content, count = 1)
    if n == 0:
        return "pattern-missing"
    try:
        with open(conv_base_path, "wb") as f:
            f.write(new_content)
    except OSError:
        return "pattern-missing"
    return "applied"
pass


_NUM_EXPERTS_PATTERN = re.compile(
    rb'^([ \t]*)n_experts = self\.hparams\[(["\'])num_experts\2\]'
    rb'([ \t]*(?:\#[^\r\n]*)?)(\r?\n|$)',
    re.MULTILINE,
)


def _patch_num_experts(content, eol_fallback = None):
    """Rewrite `n_experts = self.hparams["num_experts"]` to also accept
    num_local_experts, reusing the captured indent (the converter is not always
    written at the same depth) and line ending. Returns (new_content, applied)."""
    # A match on the last line carries no terminator to reuse; inherit the file's own.
    fallback = eol_fallback or _dominant_newline(content)

    def _replace(match):
        indent, trailer, newline = match.group(1), match.group(3), match.group(4)
        eol = newline or fallback
        # Keep `trailer` (trailing spaces or an inline comment) so a checkout that
        # carries one still gets patched, as it did under the old unanchored regex.
        return (
            indent + b"# Qwen3MoE seems to use num_local_experts instead of num_experts" + eol +
            indent + b"n_experts = self.hparams.get('num_experts', None) or self.hparams.get('num_local_experts')" +
            trailer + newline
        )
    new_content = _NUM_EXPERTS_PATTERN.sub(_replace, content)
    return new_content, (new_content != content)
pass


def _patched_content_parses(content):
    """True iff the patched converter is still syntactically valid Python.

    Parse the bytes, not a decoded string, so a utf-8 BOM or a PEP 263 coding
    cookie is honoured the way CPython honours it on import."""
    try:
        ast.parse(content)
        return True
    except (SyntaxError, ValueError, RecursionError):
        # RecursionError: deeply nested expressions can exhaust the stack.
        return False
pass


def _choose_content_to_write(stages):
    """Pick the most patched converter content that is still valid Python.

    `stages` is [(label, content), ...] oldest first, starting with the
    untouched upstream script. Returns (label, content, dropped_labels).

      * Only blame our own patches: if the untouched script does not parse on
        this interpreter, dropping them fixes nothing, so keep them all.
      * Drop the fewest possible, by walking back one stage at a time."""
    # All Unsloth Zoo code licensed under LGPLv3
    base_label, base_content = stages[0]
    final_label, final_content = stages[-1]
    if final_content == base_content:
        return final_label, final_content, []
    if _patched_content_parses(final_content):
        return final_label, final_content, []
    if not _patched_content_parses(base_content):
        return final_label, final_content, []
    for index in range(len(stages) - 2, -1, -1):
        label, content = stages[index]
        if _patched_content_parses(content):
            dropped = [name for name, _ in stages[index + 1:]]
            return label, content, dropped
        pass
    pass
    return base_label, base_content, [name for name, _ in stages[1:]]
pass


def _qwen_already_handles_expert_aliases(conv_qwen_path):
    """True iff conversion/qwen.py already searches both num_local_experts and
    num_experts (upstream master uses
    find_hparam(["num_local_experts", "num_experts"])), making the legacy
    patch a no-op with a misleading warning."""
    try:
        with open(conv_qwen_path, "rb") as f:
            content = f.read()
    except OSError:
        return False
    return (b"num_local_experts" in content) and (b"num_experts" in content)
pass


def _refuse_unscannable_conversion_package(conversion_dir, reason, is_local_copy = False):
    """Warn, or under strict mode refuse, a conversion/ package too big to read.

    Mirrors warn_on_suspicious_converter's contract, but cannot go through it:
    this is a fact about the DIRECTORY and that function takes the bytes it
    scans. Silent when the scan is switched off entirely, as everything here is.

    Says "more than", not how many: counting them all is the unbounded walk the
    cap exists to avoid, so the walk stops one past the cap and the exact size of
    an oversized tree is deliberately never learned.
    """
    if scan_is_disabled():
        return
    message = (
        f"Unsloth: The converter package at {conversion_dir} {reason}, so some "
        f"of the modules the converter imports have not been checked."
    )
    logger.warning(message)
    # is_local_copy carries the same meaning it has in warn_on_suspicious_converter:
    # the user pinned this directory, so it is reported and never refused.
    if scan_is_strict() and not is_local_copy:
        raise ConverterScanError(
            f"{message} Refusing to run it with UNSLOTH_CONVERTER_SCAN_STRICT=1. Pin a "
            f"converter you have reviewed with UNSLOTH_LLAMA_CPP_SCRIPTS_DIR, or unset "
            f"UNSLOTH_CONVERTER_SCAN_STRICT."
        )


def _directory_identity(path):
    """(device, inode) for a directory, or None when the filesystem has no usable one.

    Only ever used to notice a directory reached twice. None on anything that
    cannot answer, so an unidentifiable directory is walked rather than pruned:
    losing the loop guard is recoverable (the cap stops it), pruning a real
    subtree would drop modules from the scan.
    """
    try:
        info = os.stat(path)
    except OSError:
        return None
    # st_ino is 0 on filesystems that do not report one; every directory would
    # then share an identity and the first would prune all the rest.
    if not info.st_ino:
        return None
    return (info.st_dev, info.st_ino)


# Named, because this result has now grown three times and twice a caller read a
# new field as though it were an old one.
PackageWalk = collections.namedtuple("PackageWalk", "names complete unreadable")


def _conversion_package_modules(
    conversion_dir,
    file_limit = None,
    entry_limit = None,
    recursive = True,
    skip_names = (),
    purged = True,
    verify = True,
):
    """`(names, complete)` for the .py files under `conversion_dir`, nested included.

    Relative POSIX paths, sorted, so the caller's cap is stable across platforms.
    `complete` is False when a limit stopped the walk early, which is the caller's
    signal that the package is too big to have been read, not that it is clean.
    `names` is None when the directory could not be walked at all, which the
    caller treats as nothing to scan rather than as a clean package.

    Recursive, because `os.listdir` saw immediate children only: a clean
    `conversion/__init__.py` doing `from .nested import x` fronted
    `conversion/nested/__init__.py`, which was neither scanned nor counted
    against the cap, and Python imported and ran it all the same. Subdirectories
    are walked whether or not they hold an `__init__.py`, since a namespace
    package imports just as well.

    followlinks, because the import machinery follows directory symlinks and
    os.walk does not. Without it a clean `conversion/__init__.py` importing
    `conversion.linked`, where `linked` is a symlink to a directory holding the
    payload, returned only the clean initializer: the whole package read as
    scanned and strict mode let the payload run. Following them means the walk
    can be sent round a loop, so a directory reached a second time is pruned.

    The limits are how the caller's cap becomes a bound on the WORK and not just
    on what gets read: the directory is attacker-supplied, so traversing all of it
    to discover it was too big hands over exactly the unbounded time and memory
    the cap denies. `file_limit` stops once that many modules are in hand;
    `entry_limit` stops on entries visited, because a tree can be enormous while
    holding almost no Python at all and only the second bound sees that.
    """
    found = []
    seen_directories = set()
    unreadable = []
    entries = 0
    pending = [conversion_dir]
    while pending:
        root = pending.pop()
        identity = _directory_identity(root)
        if identity is not None:
            if identity in seen_directories:
                continue
            seen_directories.add(identity)
        try:
            # scandir rather than walk: walk hands back a whole directory's names
            # at once, so a single directory with a great many entries was fully
            # listed and copied before either budget was consulted, and the
            # advertised bounds bounded nothing for it. This iterator is lazy, so
            # the budget is checked per entry.
            with os.scandir(root) as scanner:
                for entry in scanner:
                    entries += 1
                    if entry_limit is not None and entries >= entry_limit:
                        return PackageWalk(sorted(found), False, tuple(unreadable))
                    try:
                        # Follows symlinks, because the import machinery does.
                        is_directory = entry.is_dir()
                    except OSError:
                        continue
                    if is_directory:
                        if recursive:
                            pending.append(entry.path)
                        elif entry.name == "__pycache__" and root == conversion_dir:
                            # Not a subtree, the cache for THIS directory's own
                            # modules. Skipping it left a clean root gguf.py beside
                            # an attacker's __pycache__/gguf.<tag>.pyc, which
                            # CPython validates and runs in place of the source
                            # that was scanned.
                            pending.append(entry.path)
                        continue
                    name = entry.name
                    if name in skip_names and root == conversion_dir:
                        continue      # this scan's own output, where it writes it
                    if not name.lower().endswith(COLLECTED_MODULE_SUFFIXES):
                        continue
                    if not _counts_as_a_module(
                        root, name, purged = purged, verify = verify,
                    ):
                        continue
                    found.append(
                        os.path.relpath(entry.path, conversion_dir).replace(os.sep, "/")
                    )
                    if file_limit is not None and len(found) >= file_limit:
                        # Sorted first: which names survive stays deterministic
                        # even though which directories were reached does not.
                        return PackageWalk(
                            sorted(found)[:file_limit], False, tuple(unreadable),
                        )
        except OSError:
            # One unreadable directory, not the whole package. Aborting the walk
            # here and reporting nothing meant an unreadable directory beside a
            # readable malicious module silenced the scan entirely, strict mode
            # included. Keep what was collected and name what could not be read.
            if root == conversion_dir:
                return PackageWalk(None, False, (".",))
            unreadable.append(
                os.path.relpath(root, conversion_dir).replace(os.sep, "/")
            )
            continue
    return PackageWalk(sorted(found), not unreadable, tuple(unreadable))


def _scan_conversion_package(llama_cpp_dir, is_local_copy = False):
    """Scan every package the converter imports, beside an unverified converter.

    Same warn-or-raise contract as the entrypoint: these files are imported and
    executed by it, so leaving them unscanned let a clean entrypoint front a
    payload in conversion/__init__.py, or in a nested module below it, or in
    gguf-py/gguf, which arrives in the same undigested tarball and which the
    entrypoint puts on sys.path itself.
    """
    # Cost, measured against a real clone of llama.cpp master rather than a tree
    # shaped like one: 3.1s for the four locations it now reads, paid inside the
    # cached patcher, so once per process rather than per export, against an
    # export that runs for minutes.
    #
    # What runs on EVERY export is the cache key, and that is no longer the 11ms
    # it once was: it builds the scan plan, which parses the converter's import
    # closure to decide which directories are reachable, and that is 296ms of the
    # 314ms the key takes. Parsing each package twice, as a seed and again as an
    # admitted location, was another 250ms on top until it was deduplicated.
    # Memoizing the plan across exports would take it to about 1ms, since the
    # signature that would invalidate it costs 0.3ms, and that is the thing to do
    # if this ever needs to be cheaper.
    #
    # Worth re-measuring, against a real clone, before widening what gets scanned
    # any further.
    if not llama_cpp_dir:
        return
    if scan_is_disabled():
        # Each per-file check returns early anyway, but only after the whole tree
        # has been walked and every module read. Opting out should cost nothing.
        return
    plan = _scanned_locations(llama_cpp_dir)
    if plan.truncated:
        _refuse_unscannable_conversion_package(
            llama_cpp_dir,
            f"holds more than the {MAX_SCAN_LOCATIONS} directories this scan "
            f"looks in, so the ones past that were not read",
            is_local_copy = is_local_copy,
        )
    for location in plan.locations:
        _scan_imported_package(
            location.path,
            recursive = location.recursive,
            natives = location.natives,
            is_local_copy = is_local_copy,
            skip_names = location.skip,
            boundary = llama_cpp_dir,
        )


def _scan_imported_package(
    package_dir, recursive = True, natives = True, is_local_copy = False,
    skip_names = (), boundary = None,
):
    """Read every module in one imported package, or report why it could not be.

    `natives` is False outside the packages the converter imports by name. A
    prebuilt install copies the bundle's own .so/.dylib/.dll into the llama.cpp
    root and into directories beside it, so treating every native file there as
    an opaque Python module warned on an ordinary prebuilt install and, under
    strict mode, refused the export before the converter ran. Inside conversion/
    or gguf-py/gguf a native file has no business being there and is still
    reported. The cost of that line is stated plainly: a native file planted in
    the root under exactly the name of a module the converter imports is not
    reported, and refusing every prebuilt install is not a price worth paying for
    it.
    """
    # First, before the walk and long before the converter runs: bytecode that
    # came with the package executes in place of the source this scan reads, and
    # CPython's validation does not prove otherwise. Anything with a source is
    # removed and rebuilt from the .py; what could not be removed is reported.
    # Not for a pin: the outer purge already declines to touch a checkout the
    # user chose, and this path reached straight past that and rewrote it anyway.
    stuck_bytecode = [] if is_local_copy else _purge_regenerable_bytecode(
        package_dir,
        entry_limit = MAX_CONVERSION_PACKAGE_ENTRIES + 1,
        recursive = recursive,
        boundary = boundary,
    )
    # One past each cap: enough to establish it was crossed, and no more. Both
    # limits, because stopping AT the limit reports a package of exactly that
    # many entries as holding more than it does, which strict mode then refuses.
    walk = _conversion_package_modules(
        package_dir,
        file_limit = MAX_CONVERSION_PACKAGE_FILES + 1,
        entry_limit = MAX_CONVERSION_PACKAGE_ENTRIES + 1,
        recursive = recursive,
        skip_names = skip_names,
        purged = not is_local_copy,
    )
    names, complete = walk.names, walk.complete
    if walk.unreadable:
        # Silence here was the hole: an unreadable directory beside a readable
        # module its initializer imports meant the whole package went unscanned
        # and unreported, under strict mode as well.
        shown = ", ".join(walk.unreadable[:5]) + ("..." if len(walk.unreadable) > 5 else "")
        _refuse_unscannable_conversion_package(
            package_dir,
            f"holds {len(walk.unreadable)} director(y/ies) this scan could not "
            f"read ({shown})",
            is_local_copy = is_local_copy,
        )
    if names is None:
        return
    if not complete and len(names) <= MAX_CONVERSION_PACKAGE_FILES and not walk.unreadable:
        # Stopped on the traversal budget rather than the file cap: few enough
        # modules, far too many entries to have walked. Unread either way.
        _refuse_unscannable_conversion_package(
            package_dir,
            f"holds more than the {MAX_CONVERSION_PACKAGE_ENTRIES} directory "
            f"entries this scan walks",
            is_local_copy = is_local_copy,
        )
    if len(names) > MAX_CONVERSION_PACKAGE_FILES:
        # Truncating the list silently was the hole: a payload in a late-sorting
        # module (z_payload.py) imported from an otherwise clean __init__.py went
        # unscanned, and under strict mode ran with nothing reported. The cap
        # stays, because an attacker must not get to choose how much work this
        # does, so exceeding it becomes the finding rather than a quiet skip.
        _refuse_unscannable_conversion_package(
            package_dir,
            f"holds more than the {MAX_CONVERSION_PACKAGE_FILES} Python files "
            f"this scan reads",
            is_local_copy = is_local_copy,
        )
        names = names[:MAX_CONVERSION_PACKAGE_FILES]
    opaque = _unscannable_modules(package_dir, names, natives = natives) + stuck_bytecode
    if opaque:
        # Reporting is the whole answer available here: this scan reads source,
        # and these are the files it provably cannot. Passing over them quietly
        # is what let bytecode and native modules ride in under a clean package.
        shown = ", ".join(opaque[:5]) + ("..." if len(opaque) > 5 else "")
        _refuse_unscannable_conversion_package(
            package_dir,
            f"holds {len(opaque)} file(s) Python will execute but this scan "
            f"cannot read ({shown})",
            is_local_copy = is_local_copy,
        )
    for name in names:
        if os.path.splitext(name)[1].lower() != ".py":
            continue
        path = os.path.join(package_dir, *name.split("/"))
        try:
            with open(path, "rb") as handle:
                # One byte past the ceiling, which is how a module too big to
                # judge is told apart from one that merely fills it. Reading it
                # whole first was the memory the ceiling exists to deny.
                content = handle.read(MAX_MODULE_BYTES + 1)
        except OSError:
            continue
        if len(content) > MAX_MODULE_BYTES:
            content = content[:MAX_MODULE_BYTES]
            _refuse_unscannable_conversion_package(
                package_dir,
                f"holds a module ({name}) larger than the {MAX_MODULE_BYTES} "
                f"bytes this scan reads, so only its first part was checked",
                    is_local_copy = is_local_copy,
            )
        warn_on_suspicious_converter(
            content, path, is_local_copy = is_local_copy, log = logger,
        )


# A pin Unsloth set itself, to route the patcher at an install it has just made.
# Not the same thing as a pin the user set to choose a converter they reviewed,
# and only the second is a reason to skip the scan.
_INTERNAL_SCRIPTS_DIR_LOCK = threading.Lock()
_internal_scripts_dir_pin = None


@contextlib.contextmanager
def internal_scripts_dir_pin(folder):
    """Point the patcher at `folder` without that counting as the user's choice.

    MLX export installs llama.cpp itself and then sets
    UNSLOTH_LLAMA_CPP_SCRIPTS_DIR so the patcher resolves against that install.
    Trust was read from the variable alone, so a converter Unsloth had just
    downloaded looked exactly like one the user had pinned and reviewed:
    UNSLOTH_CONVERTER_SCAN_STRICT only logged the entrypoint's findings instead
    of raising, and the imported packages were not scanned at all. The whole
    strict control was therefore off for save_pretrained_gguf.

    A pin already in the environment is left untouched, because that one IS the
    user's and carries their exemption.
    """
    global _internal_scripts_dir_pin
    with _INTERNAL_SCRIPTS_DIR_LOCK:
        existing = os.environ.get("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR")
        previous = _internal_scripts_dir_pin
        if existing is None:
            os.environ["UNSLOTH_LLAMA_CPP_SCRIPTS_DIR"] = folder
            _internal_scripts_dir_pin = os.path.abspath(os.path.expanduser(folder))
        try:
            yield
        finally:
            if existing is None:
                os.environ.pop("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", None)
            else:
                os.environ["UNSLOTH_LLAMA_CPP_SCRIPTS_DIR"] = existing
            _internal_scripts_dir_pin = previous


def _converter_is_trusted_local(script_path):
    """Whether a local converter was pinned deliberately by the user.

    The strict-mode exemption means "you chose this file", so it cannot cover
    every local path. UNSLOTH_LLAMA_CPP_SCRIPTS_DIR is an explicit pin and is the
    only thing that is, and then only when Unsloth did not set it itself: MLX
    export points it at the llama.cpp it has just installed, which is routing and
    not a judgement about the converter, so internal_scripts_dir_pin marks that
    case and it gets no exemption either. When no prebuilt is available install_llama_cpp falls
    back to an unpinned `git clone` of upstream master, and
    _resolve_bundle_convert_script accepts that checkout on the strength of a
    conversion/ package alone. A converter fetched automatically from upstream is
    not a converter the user pinned, so it gets no exemption.

    UNSLOTH_PREBUILT_INFO.json is NOT accepted, though it used to be. The marker
    reads like proof that these bytes were verified and it is not: _stage_prebuilt
    _install checks a sha256 for the BINARY asset only, and even that only when
    the release published one, while _hydrate_converter_sources downloads the
    source tarball separately with no digest at all and the marker is written
    afterwards. So a replaced source tarball wore a "verified" marker, took the
    exemption, and skipped the conversion/ scan with it. The converter from a
    prebuilt bundle is now scanned like any other download, which is what it is.
    """
    if not script_path:
        return False
    script_path = os.path.abspath(os.path.expanduser(script_path))
    scripts_dir = os.environ.get("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR")
    if not scripts_dir:
        return False
    # expanduser to match _resolve_local_convert_script, which accepts the pin
    # after expanding it. Comparing an unexpanded "~/llama.cpp" against the
    # expanded script path made a deliberate pin fail this test and be refused
    # under strict mode as though it had been downloaded.
    scripts_dir = os.path.abspath(os.path.expanduser(scripts_dir))
    if _internal_scripts_dir_pin is not None and scripts_dir == _internal_scripts_dir_pin:
        # Unsloth's own routing, not a choice anyone made about these bytes.
        return False
    try:
        return os.path.commonpath([scripts_dir, script_path]) == scripts_dir
    except ValueError:
        return False


def _download_convert_hf_to_gguf(name = "unsloth_convert_hf_to_gguf"):
    # Resolve env vars + sibling mtimes each call; both feed the @lru_cache key
    # so re-pulled checkouts re-run the patcher. Anchor conversion/ to the
    # converter being patched (matters when UNSLOTH_LLAMA_CPP_SCRIPTS_DIR points
    # at a different checkout), not always LLAMA_CPP_DEFAULT_DIR.
    local_script_info = _resolve_local_convert_script()
    if local_script_info is None:
        local_script_info = _resolve_bundle_convert_script()
    # Outside the cache on purpose: cheap, idempotent, and a checkout pulled
    # or replaced after the first conversion still gets the Qwen3.5 aliases.
    _llama_cpp_dir = _get_llama_cpp_dir(local_script_info)
    _patch_tensor_mapping_for_qwen35(_llama_cpp_dir)
    # Before the cache is consulted and before the key is built, so the key
    # describes the tree the converter will actually run against.
    trusted_local = _converter_is_trusted_local(
        local_script_info[0] if local_script_info is not None else None
    )
    stuck_bytecode = _purge_imported_package_bytecode(
        _llama_cpp_dir, is_local_copy = trusted_local,
    )
    return _download_convert_hf_to_gguf_cached(
        name,
        local_script_info,
        _conversion_sibling_info(_llama_cpp_dir, is_local_copy = trusted_local),
        _converter_scan_mode(),
        trusted_local,
        stuck_bytecode,
    )


def _converter_scan_mode():
    """The scan switches, read at call time so they are part of the cache key.

    The scan runs inside the cached patcher below, so without this a user who
    sees a warning, sets UNSLOTH_CONVERTER_SCAN_STRICT=1 and retries in the same
    session gets the cached converter back and no second scan, which reads as the
    strict mode having accepted the file.

    Whether the converter is a user pin travels in the key for the same reason
    and is computed beside this one: a first call under an explicit pin could
    accept a flagged converter as trusted and cache it, and MLX routing to that
    same folder afterwards matched every other component of the key, so the
    cached result came back without the trust test or the package scan running
    again. The distinction internal_scripts_dir_pin draws is only as good as the
    key that carries it.
    """
    return (
        os.environ.get("UNSLOTH_CONVERTER_SCAN_STRICT", ""),
        os.environ.get("UNSLOTH_DISABLE_CONVERTER_SCAN", ""),
    )


@lru_cache(1)
def _download_convert_hf_to_gguf_cached(
    name, _local_script_info, _conversion_info, _scan_mode = None, _trusted_local = False,
    _stuck_bytecode = (),
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Download from llama.cpp's GitHub, or read a local copy when
    # UNSLOTH_LLAMA_CPP_SCRIPTS_DIR is set. _local_script_info is
    # (path, mtime_ns, size); mtime/size in the cache key invalidate stale
    # entries on in-place updates. Cache size is 1 because the patched script
    # is written to one shared on-disk path, so a second entry would read stale
    # bytes.

    # Ensure llama.cpp directory exists
    os.makedirs(LLAMA_CPP_DEFAULT_DIR, exist_ok=True)

    supported_types = set()
    text_archs = set()
    vision_archs = set()
    # Default to 'monolith' so a failed introspection still drives the legacy
    # patches; set by introspection and read by Patch 2 + Patch 3 below.
    _layout = "monolith"
    _llama_cpp_dir = _get_llama_cpp_dir(_local_script_info)

    _local_script = _local_script_info[0] if _local_script_info is not None else None

    try:
        # 1. Obtain the file (local override takes precedence over network)
        if _local_script is not None:
            logger.info(f"Unsloth: Using local convert_hf_to_gguf.py from {_local_script}")
            with open(_local_script, "rb") as f:
                original_content = f.read()
        else:
            # Retry with exponential backoff: the upstream host can
            # exceed the default read timeout on slower networks.
            _last_err = None
            original_content = None
            for _attempt in range(3):
                try:
                    response = requests.get(
                        LLAMA_CPP_CONVERT_FILE, timeout = (10, 120)
                    )
                    response.raise_for_status()
                    original_content = response.content
                    break
                except requests.exceptions.RequestException as _err:
                    _last_err = _err
                    logger.warning(
                        f"Unsloth: convert_hf_to_gguf.py download attempt "
                        f"{_attempt + 1}/3 failed ({type(_err).__name__}: {_err}); retrying"
                    )
                    time.sleep(2 ** _attempt)
            if original_content is None:
                raise _last_err  # type: ignore[misc]

        # 1b. Scan the bytes we are about to patch, eval defaults out of, write
        # to disk and run. Deliberately before every one of those steps. Warns
        # by default (see converter_scan for why it does not block), and raises
        # ConverterScanError only under UNSLOTH_CONVERTER_SCAN_STRICT=1 on
        # downloaded bytes.
        # Passed in, not recomputed: it is a cache key component, so deciding it
        # again in here could disagree with the entry that was looked up.
        warn_on_suspicious_converter(
            original_content,
            _local_script if _local_script is not None else LLAMA_CPP_CONVERT_FILE,
            is_local_copy = _trusted_local,
            log = logger,
        )
        # The package entrypoint runs `from conversion import ...` on import, so
        # those files execute too. They are covered by a sha256 only when the
        # release published one for the SOURCE archive they came out of, which
        # is a second download from the one the bundle's own sha256 covers. An
        # earlier version of this comment said the bundle's digest covered them;
        # it does not, and until that archive was verified as well nothing did.
        # From an unpinned `git clone`, or the ggml-org codeload fallback, there
        # is still no digest to check, and a payload can sit in
        # conversion/__init__.py behind a clean entrypoint.
        # Always, pinned or not. The entrypoint itself is scanned either way and
        # only the refusal is waived for a pin; skipping the sibling packages
        # entirely meant a stale or tampered conversion/ beside a pinned
        # entrypoint executed without even the advisory warning.
        _scan_conversion_package(_llama_cpp_dir, is_local_copy = _trusted_local)

        # 2. Detect layout BEFORE importing: the package entrypoint does
        # `from conversion import ...`, which a temp-file import resolves
        # against LLAMA_CPP_DEFAULT_DIR; with a different
        # UNSLOTH_LLAMA_CPP_SCRIPTS_DIR that would ModuleNotFoundError and
        # abort before the AST-based arch extraction path.
        _layout = _detect_converter_layout(original_content, _llama_cpp_dir)
        logger.info(f"Unsloth: convert_hf_to_gguf layout detected: {_layout}")
        logger.info("Unsloth: Identifying llama.cpp gguf supported architectures...")

        if _layout == "package":
            # Package layout: archs come from AST-parsing the static
            # TEXT_MODEL_MAP / MMPROJ_MODEL_MAP in conversion/__init__.py.
            # No module import required, so we skip the temp-write entirely.
            conv_init_py = os.path.join(_llama_cpp_dir, "conversion", "__init__.py")
            text_archs   = _extract_dict_keys_from_conversion_init(conv_init_py, "TEXT_MODEL_MAP")
            vision_archs = _extract_dict_keys_from_conversion_init(conv_init_py, "MMPROJ_MODEL_MAP")
            supported_types.update(text_archs)
            supported_types.update(vision_archs)
            if not supported_types:
                logger.warning(
                    "Unsloth: conversion/__init__.py parsed but TEXT_MODEL_MAP / "
                    "MMPROJ_MODEL_MAP yielded no architecture keys. The arch "
                    "allowlist will be empty; conversion will still attempt to run."
                )
        else:
            # Monolith layout: read the registrations out of the entrypoint. It is
            # downloaded from llama.cpp master at runtime, so parse, never import.
            text_archs, vision_archs = _extract_archs_from_monolith_source(original_content)
            supported_types.update(text_archs)
            supported_types.update(vision_archs)
            if not text_archs:
                logger.info("Unsloth: No TEXT model architectures found registered in the original script.")
            if not vision_archs:
                logger.info("Unsloth: No VISION model architectures found registered in the original script.")
        # --- End Architecture Extraction ---

        # Convert final set to frozenset for immutability (good practice for cache keys/return values)
        text_archs = frozenset(text_archs)
        vision_archs = frozenset(vision_archs)
        supported_types = frozenset(supported_types)

        if not supported_types:
             logger.warning(
                f"Unsloth: No supported architectures (TEXT or VISION) could be determined from the original script."
            )

    except ConverterScanError:
        # A deliberate refusal under UNSLOTH_CONVERTER_SCAN_STRICT=1. Propagate it
        # with its own message instead of relabelling it an introspection failure.
        raise
    except Exception as e:
         logger.error(f"Unsloth: Error during loading or introspecting the original script: {e}", exc_info=True)
         raise RuntimeError(f"Failed during loading/introspection of original script: {e}") from e


    # --- Proceed with patching and saving ---
    try:
        patched_content = original_content # Start patching from original
        # Snapshot after every patch so a patch that breaks the file can be dropped
        # on its own instead of discarding the others (see _choose_content_to_write).
        _patch_stages = [("unpatched upstream script", original_content)]

        # 3. Apply Patches (gguf attributes, metadata branding - same logic as before)
        logger.info("Unsloth: Applying patches...")
        # Patch 1: gguf Attribute Handling
        try:
            # Scan with any previous guards stripped out: they spell `gguf.X` themselves,
            # so a scan over the patched file re-finds every name it already covers. The
            # old fix was to bail out on the first guard line, which left a newly
            # referenced enum unguarded on a converter that had been patched once.
            _unguarded_source = _GGUF_GUARD_BLOCK_PATTERN.sub(b"", patched_content)
            archs = list(set(re.findall(rb"[\n\s]gguf\.([\.A-Z\_0-9]{3,})[\n\s\,]", _unguarded_source)))
            archs = [x.decode("utf-8") for x in archs if not x.startswith(b"_")]
            _already_guarded = {
                name.decode("utf-8")
                for name in _GGUF_GUARD_PATTERN.findall(patched_content)
            }
            # Sorted, so a re-patch writes the same bytes rather than a set's order.
            archs = sorted(name for name in archs if name not in _already_guarded)
            if not archs and _GGUF_GUARD_LINE in patched_content:
                # Everything is already covered, so the file converges instead of
                # growing one copy of the block per patch.
                logger.info(
                    "Unsloth: gguf attribute guards already cover every referenced "
                    "attribute (idempotent skip)."
                )
            elif archs:
                _eol = _dominant_newline(patched_content)
                _eol_text = _eol.decode("utf-8")
                all_edits = _eol_text.join(
                    f"try: gguf.{x}{_eol_text}except AttributeError: gguf.{x} = None" for x in archs
                ).encode("utf-8")
                patched_content = re.sub(rb"(import gguf[ \t]*\r?\n)", rb"\1" + all_edits + _eol + _eol, patched_content, count=1)
                if original_content == patched_content and archs: logger.warning("Unsloth: gguf attribute patch did not seem to apply.")
            else: logger.info("Unsloth: No specific gguf attributes found to patch.")
        except Exception as e: logger.error(f"Unsloth: Error applying gguf attribute patch: {e}", exc_info=True); raise
        _patch_stages.append(("gguf attribute guards", patched_content))



        # Patch 2: Metadata Branding.
        # Monolith: target lives in the entrypoint; patch the in-memory bytes.
        # Package: target moved to conversion/base.py; patch that file in place
        # (idempotent via _UNSLOTH_BRANDING_MARKER) since the entrypoint just
        # imports ModelBase from it at runtime.
        try:
            if _layout == "package":
                conv_base_py = os.path.join(_llama_cpp_dir, "conversion", "base.py")
                _branding_status = _apply_branding_patch_to_base(conv_base_py)
                if _branding_status == "applied":
                    logger.info(f"Unsloth: Metadata branding patch applied to {conv_base_py}.")
                elif _branding_status == "already-applied":
                    logger.info(f"Unsloth: Metadata branding patch already present in {conv_base_py} (idempotent skip).")
                else:
                    logger.warning(
                        f"Unsloth: Metadata branding patch target not found in {conv_base_py}. "
                        f"Upstream may have refactored Metadata.load again."
                    )
            elif _UNSLOTH_BRANDING_LINE in patched_content:
                # This patch has no marker of its own, so without this check the regex
                # matches `Metadata.load(...)` again and appends another copy of the
                # branding on every conversion. Upstream never ships this line, so a
                # pristine checkout is unaffected.
                logger.info(
                    "Unsloth: Metadata branding patch already present in the converter "
                    "(idempotent skip)."
                )
            else:
                metadata_patch_applied = False
                _eol = _dominant_newline(patched_content)
                new_patched_content = re.sub(
                    rb"(self\.metadata \= gguf\.Metadata\.load\(.+?\))([\n\r]+([\s\t]{4,}))",
                    rb"\1" + _eol +
                    rb"\3if hasattr(self.metadata, 'quantized_by'): self.metadata.quantized_by = 'Unsloth'" + _eol +
                    rb"\3if hasattr(self.metadata, 'repo_url'): self.metadata.repo_url = 'https://huggingface.co/unsloth'" + _eol +
                    rb"\3if hasattr(self.metadata, 'tags'): self.metadata.tags = ['unsloth', 'llama.cpp']" + _eol +
                    rb"\2",
                    patched_content, count=1, flags=re.MULTILINE
                )
                if new_patched_content != patched_content: patched_content = new_patched_content; metadata_patch_applied = True
                if not metadata_patch_applied:
                     if re.search(rb"self\.metadata \= gguf\.Metadata\.load\(", patched_content): logger.warning("Unsloth: Metadata branding patch target found, but regex failed to apply.")
                     else: logger.warning("Unsloth: Metadata branding patch target 'self.metadata = gguf.Metadata.load(...)' not found.")
        except Exception as e: logger.error(f"Unsloth: Error applying metadata branding patch: {e}", exc_info=True); raise
        _patch_stages.append(("metadata branding", patched_content))


        # Patch 3: Qwen2MoE / Qwen3MoE num_experts fix.
        # Package layout uses find_hparam(["num_local_experts", "num_experts"])
        # already, so the legacy patch is obsolete and its warning misleading.
        # Skip it (info-log) on new layout; run unchanged on monolith.
        try:
            _qwen_handled = False
            if _layout == "package":
                conv_qwen_py = os.path.join(_llama_cpp_dir, "conversion", "qwen.py")
                if os.path.isfile(conv_qwen_py) and _qwen_already_handles_expert_aliases(conv_qwen_py):
                    logger.info(
                        "Unsloth: Qwen2MoE expert-key alias already handled upstream "
                        "(conversion/qwen.py uses find_hparam([num_local_experts, num_experts])) "
                        "-- legacy patch skipped."
                    )
                    _qwen_handled = True

            if not _qwen_handled:
                new_patched_content, num_experts_patch_applied = _patch_num_experts(patched_content)

                if num_experts_patch_applied:
                    patched_content = new_patched_content
                else:
                    logger.warning("Unsloth: Qwen2MoE num_experts patch target not found.")

        except Exception as e:
            logger.error(f"Unsloth: Error applying Qwen2MoE num_experts patch: {e}", exc_info=True)
            raise
        _patch_stages.append(("Qwen2MoE num_experts alias", patched_content))


        # 4. Write Patched File
        # Keep package-layout entrypoints beside conversion/ so subprocess
        # execution resolves `from conversion import ...`.
        patched_dir = _llama_cpp_dir if _layout == "package" else LLAMA_CPP_DEFAULT_DIR
        os.makedirs(patched_dir, exist_ok=True)
        patched_filename = os.path.join(patched_dir, f"{name}.py")

        # Never write a converter we just broke: fall back to the last content that
        # still parses, rather than failing later far from the cause.
        _kept_label, _kept_content, _dropped_labels = _choose_content_to_write(_patch_stages)
        if _dropped_labels:
            logger.warning(
                f"Unsloth: Patched converter script is not valid Python - dropping "
                f"{', '.join(_dropped_labels)} and writing the content after "
                f"{_kept_label} instead."
            )
            patched_content = _kept_content
        pass

        logger.info(f"Unsloth: Saving patched script to {patched_filename}")
        with open(patched_filename, "wb") as file:
            file.write(patched_content)

        # 5. Parse Flags from Patched Content (same logic as before)
        logger.info("Unsloth: Parsing arguments from patched script...")
        flags = re.findall(rb"parser\.add_argument\([\s]*[\"\']([^\"\']{1,})[\'\"]", patched_content)
        if not flags: raise RuntimeError(f"Unsloth: Failed parsing {patched_filename} - no arguments found.")
        # The same compiled regex converter_scan vets these tokens with, so the
        # two cannot drift apart. The scan reads the pre-patch bytes; the patches
        # above only insert Unsloth-authored lines and never an add_argument call,
        # so both see the same set of defaults.
        defaults = RE_ARGPARSE_DEFAULT.findall(patched_content)
        all_flags = {}
        for flag_bytes, default_bytes in defaults:
            flag = flag_bytes.decode("utf-8").lstrip('-').replace("-", "_")
            default_str = default_bytes.decode("utf-8")
            try:
                if default_str == "store_true": default_val = False
                elif default_str == "store_false": default_val = True
                elif default_str == "None": default_val = None
                else: default_val = eval(default_str)
            except Exception: logger.warning(f"Could not eval default '{default_str}' for '{flag}'. Setting None."); default_val = None
            all_flags[flag] = default_val
        rest_flags = [fb.decode("utf-8").lstrip('-').replace("-", "_") for fb in flags if fb.decode("utf-8").lstrip('-').replace("-", "_") not in all_flags]
        essential_flags = ["model", "outfile", "outtype"]
        for flag in rest_flags:
            if flag not in essential_flags: all_flags[flag] = None
        for flag in essential_flags:
             if flag not in all_flags and flag not in rest_flags: logger.warning(f"Essential flag '{flag}' potentially missing."); all_flags[flag] = None
        logger.info("Unsloth: Successfully processed convert_hf_to_gguf.py.")
        # Return path to PATCHED file and combined architectures set
        return patched_filename, text_archs, vision_archs

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Unsloth: Network error downloading `{LLAMA_CPP_CONVERT_FILE}`: {e}") from e
    except ImportError as e:
         raise RuntimeError(f"Unsloth: Import error during module loading: {e}") from e
    except Exception as e:
        logger.error(f"Unsloth: Unexpected error after introspection: {e}", exc_info=True)
        raise RuntimeError(f"Unsloth: Failed during patching/parsing of script content: {e}") from e
pass


# Preserve the pre-split lru_cache surface (cache_clear, cache_info,
# cache_parameters) so external callers keep working. __wrapped__ is not
# forwarded because the inner function takes a private (name, _local_script_info)
# pair while the public wrapper is a single-arg callable.
_download_convert_hf_to_gguf.cache_clear = _download_convert_hf_to_gguf_cached.cache_clear
_download_convert_hf_to_gguf.cache_info = _download_convert_hf_to_gguf_cached.cache_info
_download_convert_hf_to_gguf.cache_parameters = _download_convert_hf_to_gguf_cached.cache_parameters


# Qwen3.5 HF tensor names emitted by empty_model.py's GDN export, keyed by
# the tensor_mapping.py block each belongs to on llama.cpp master. dt_bias
# needs no entry: the converter renames it to dt_proj.bias before mapping.
_QWEN35_TENSOR_MAPPINGS = (
    ("ATTN_QKV",   "model.layers.{bid}.linear_attn.in_proj_qkv"),
    ("ATTN_GATE",  "model.layers.{bid}.linear_attn.in_proj_z"),
    ("SSM_BETA",   "model.layers.{bid}.linear_attn.in_proj_b"),
    ("SSM_ALPHA",  "model.layers.{bid}.linear_attn.in_proj_a"),
    ("SSM_CONV1D", "model.layers.{bid}.linear_attn.conv1d"),
    ("SSM_DT",     "model.layers.{bid}.linear_attn.dt_proj"),
    ("SSM_A",      "model.layers.{bid}.linear_attn.A_log"),
    ("SSM_NORM",   "model.layers.{bid}.linear_attn.norm"),
    ("SSM_OUT",    "model.layers.{bid}.linear_attn.out_proj"),
)


def _patch_tensor_mapping_for_qwen35(llama_cpp_dir: str):
    """Insert missing Qwen3.5 linear_attn aliases into a stale
    gguf-py/gguf/tensor_mapping.py. The converter script is fetched from
    llama.cpp master, but gguf-py comes from the local checkout, so a
    checkout predating Qwen3.5 cannot map the split GDN projections.
    Idempotent per entry; blocks absent from old checkouts are skipped."""
    tensor_mapping_path = os.path.join(llama_cpp_dir, "gguf-py", "gguf", "tensor_mapping.py")
    if not os.path.isfile(tensor_mapping_path):
        return
    try:
        with open(tensor_mapping_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return

    content = "".join(lines)
    missing = [(block, name) for block, name in _QWEN35_TENSOR_MAPPINGS if f'"{name}"' not in content]
    if not missing:
        return

    new_lines = []
    for line in lines:
        new_lines.append(line)
        stripped = line.strip()
        for block, name in missing:
            if stripped == f"MODEL_TENSOR.{block}: (":
                indent = line[: len(line) - len(line.lstrip())] + "    "
                new_lines.append(f'{indent}"{name}",  # qwen3.5\n')
                break

    if new_lines == lines:
        return
    patched = "".join(new_lines)
    try:
        ast.parse(patched)
    except SyntaxError:
        logger.warning("Unsloth: Qwen3.5 tensor_mapping.py patch produced invalid syntax, leaving file unchanged.")
        return
    with open(tensor_mapping_path, "w", encoding="utf-8") as f:
        f.write(patched)


def _split_str_to_n_bytes(split_str: str) -> int:
    # All Unsloth Zoo code licensed under LGPLv3
    # Converts 50G to bytes
    if split_str.endswith("K"):
        n = float(split_str[:-1]) * 1000
    elif split_str.endswith("M"):
        n = float(split_str[:-1]) * 1000 * 1000
    elif split_str.endswith("G"):
        n = float(split_str[:-1]) * 1000 * 1000 * 1000
    elif split_str.isnumeric():
        n = float(split_str)
    else:
        raise ValueError(f"Invalid split size: {split_str}, must be a number, optionally followed by K, M, or G")

    if n < 0:
        raise ValueError(f"Invalid split size: {split_str}, must be positive")

    return n
pass


def _convert_to_gguf(command, output_filename, print_output = False, print_outputs = None):
    # All Unsloth Zoo code licensed under LGPLv3
    # Filter warnings / errors with dates
    import datetime
    datetime = datetime.datetime.today().strftime("%Y-%m-%d")

    popen = subprocess.Popen(
        command,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        universal_newlines = True,
        shell = True,
    )
    ProgressBar._instances.clear()

    progress_bar = None
    chat_template_line = 0
    stop_chat_template = False
    metadata = {}

    for line in iter(popen.stdout.readline, ""):
        if line.startswith("Writing:"):
            if progress_bar is None:
                progress_bar = ProgressBar(total = 100, position = 0, leave = True, desc = "Unsloth: GGUF conversion")

            desc = re.findall(r"([\d]{1,3})\%.+?([\d\.].+?\])", line)
            if len(desc) == 1 and len(desc[0]) == 2:
                percentage, info = desc[0]
                progress_bar.update(int(percentage) - progress_bar.n)
                info = re.findall(r"([\d\.]{1,}(?:K|M|G)\/[\d\.]{1,}(?:K|M|G))", info)
                if len(info) != 0: progress_bar.set_postfix_str(info[0])
                continue
            pass

        elif line.startswith("INFO:gguf.gguf_writer") and "total_size = " in line:
            # Get name of file as well
            name = re.findall(r"INFO:gguf\.gguf_writer:([^\:]{1,})\:", line)
            if len(name) == 1:
                name = name[0]
                # Save final size of model
                x = re.findall(r"total_size = ([\d\.]{1,}(?:K|M|G))", line)
                if len(x) == 1:
                    try:
                        total_size = _split_str_to_n_bytes(x[0])
                    except Exception as error:
                        popen.terminate()
                        raise RuntimeError(error)
                    metadata[name] = (total_size, x[0],)
                pass
            pass

        elif line.startswith((datetime, "WARNING:", "INFO:numexpr")):
            # Skip warnings / errors
            continue

        elif line.startswith("INFO:hf-to-gguf:blk"):
            # Skip showcasing conversions - unnecessary
            continue

        elif line.startswith("INFO:gguf.vocab:Setting chat_template"):
            # Do not print super long chat templates - allow 5 lines
            chat_template_line = 1

        if chat_template_line != 0: chat_template_line += 1

        if chat_template_line >= 10:
            # Restart if possible
            if line.startswith("INFO:hf-to-gguf:"):
                chat_template_line = 0
            else:
                if not stop_chat_template:
                    print("..... Chat template truncated .....\n")
                stop_chat_template = True
                continue
            pass
        pass

        # Fix up start of strings
        if line.startswith("INFO:"): line = "Unsloth GGUF:" + line[len("INFO:"):]

        if print_output: print(line, flush = True, end = "")
        if print_outputs is not None: print_outputs.append(line)
    pass

    if progress_bar is not None: progress_bar.close()
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    pass

    # Check final size approximately
    if len(metadata) != 0:
        for output_filename, (total_size, x,) in metadata.items():
            actual_size = os.path.getsize(output_filename)

            ratio = actual_size / total_size
            if ratio <= 0.9 or ratio >= 1.1:
                raise RuntimeError(
                    "Unsloth: Failed converting to GGUF since we do not have enough disk space!\n"\
                    f"We need {total_size} bytes but we managed to find only {actual_size} bytes!"
                )
            pass

            line = f"Unsloth: Converted to {output_filename} with size = {x}\n"
            if print_output: print(line, flush = True, end = "")
            if print_outputs is not None: print_outputs.append(line)
        pass
    else:
        raise RuntimeError(
            "Unsloth: Failed converting to GGUF since we did not create an GGUF files?"
        )
    return list(metadata.keys())
pass


def check_quantization_type(quantization_type = "Q8_0"):
    # All Unsloth Zoo code licensed under LGPLv3
    # Gets quantization and multiplier
    assert(type(quantization_type) is str)
    quantization_type = quantization_type.lower()
    SUPPORTED_GGUF_TYPES = frozenset(("f32", "f16", "bf16", "q8_0"))
    if quantization_type not in SUPPORTED_GGUF_TYPES:
        raise RuntimeError(
            f"Unsloth: `{quantization_type}` quantization type is not supported.\n"\
            f"The following quantization types are supported: `{list(SUPPORTED_GGUF_TYPES)}`"
        )
    pass
    size_multiplier = {
        "q8_0" : 0.5,
        "f32"  : 2.0,
        "f16"  : 1.0,
        "bf16" : 1.0,
    }
    return quantization_type, size_multiplier[quantization_type]
pass


def check_max_shard_size(max_shard_size = "50GB"):
    # All Unsloth Zoo code licensed under LGPLv3
    assert(type(max_shard_size) is str)
    if max_shard_size.endswith("B"): max_shard_size = max_shard_size[:-1]
    try:
        _split_str_to_n_bytes(max_shard_size)
    except:
        raise TypeError(f"Unsloth: Shard size must be in GB, but `{max_shard_size}` is not")
    return max_shard_size
pass


# Converter deps, only installed in install_llama_cpp() (skipped when llama.cpp
# already exists), so a stale `gguf` can fail with exit 1.
_CONVERTER_PYTHON_DEPS = ("gguf", "protobuf", "sentencepiece", "mistral_common")

# Markers meaning the converter env (not the model) is broken; only these
# trigger auto-repair, genuine model errors surface as-is.
_CONVERTER_DEP_ERROR_MARKERS = (
    "ModuleNotFoundError",
    "No module named",
    "ImportError",
    "cannot import name",
    "DLL load failed",      # common Windows broken-package symptom
    "undefined symbol",
)


def _looks_like_converter_dep_error(text):
    # All Unsloth Zoo code licensed under LGPLv3
    if not text: return False
    return any(marker in text for marker in _CONVERTER_DEP_ERROR_MARKERS)


def _reinstall_converter_deps(python_exe, print_output = False):
    # All Unsloth Zoo code licensed under LGPLv3
    # Force-reinstall converter deps into their interpreter to self-heal.
    if print_output:
        print(
            f"Unsloth: The GGUF converter environment looks broken (stale/missing "
            f"package). Reinstalling {', '.join(_CONVERTER_PYTHON_DEPS)} and retrying..."
        )
    def _run(cmd):
        return subprocess.run(cmd, encoding="utf-8", errors="replace",
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    install = [python_exe, "-m", "pip", "install", "--upgrade", "--force-reinstall",
               *_CONVERTER_PYTHON_DEPS]
    result = _run(install)
    # Some envs (eg uv-created venvs) ship without pip; bootstrap it once.
    if result.returncode != 0 and "no module named pip" in (result.stdout or "").lower():
        _run([python_exe, "-m", "ensurepip", "--upgrade"])
        result = _run(install)
    return result


def _has_mtp_weight_tensors(input_folder, num_layers):
    """Return whether the checkpoint's tensor names include an MTP layer."""
    input_folder = Path(input_folder)
    _layer_re = re.compile(r"^(?:model\.)?layers\.(\d+)\.")

    def _is_mtp(name):
        # Match the converter's TextModel.filter_tensors normalization.
        name = name.replace("language_model.", "")
        if name.startswith(("mtp.", "model.mtp.")):
            return True
        m = _layer_re.match(name)
        return m is not None and int(m.group(1)) >= num_layers

    def _inspection_error(path):
        return RuntimeError(
            f"Unsloth: Could not inspect `{path.name}` for MTP tensors; "
            "`config.json` was not changed."
        )

    def _names_from_index(index_path):
        try:
            with index_path.open("r", encoding = "utf-8") as f:
                index = json.load(f)
        except Exception as error:
            raise _inspection_error(index_path) from error
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict):
            raise _inspection_error(index_path)
        return weight_map.keys()

    parts = sorted(input_folder.glob("model*.safetensors"))
    if parts:
        index_path = input_folder / "model.safetensors.index.json"
        # llama.cpp gives the canonical index precedence whenever any
        # safetensors part exists, including alongside model.safetensors.
        if index_path.is_file():
            return any(_is_mtp(name) for name in _names_from_index(index_path))
        from safetensors import safe_open

        for part in parts:
            try:
                with safe_open(part, framework = "pt", device = "cpu") as f:
                    if any(_is_mtp(name) for name in f.keys()):
                        return True
            except Exception as error:
                raise _inspection_error(part) from error
        return False

    parts = sorted(input_folder.glob("pytorch_model*.bin"))
    if parts:
        index_path = input_folder / "pytorch_model.bin.index.json"
        if index_path.is_file():
            return any(_is_mtp(name) for name in _names_from_index(index_path))
        if torch is None:
            raise RuntimeError("Unsloth: PyTorch is required to inspect `.bin` model weights.")
        for part in parts:
            try:
                state_dict = torch.load(
                    part,
                    map_location = "cpu",
                    mmap = True,
                    weights_only = True,
                )
            except Exception as error:
                raise _inspection_error(part) from error
            if isinstance(state_dict, dict) and isinstance(state_dict.get("state_dict"), dict):
                state_dict = state_dict["state_dict"]
            if not isinstance(state_dict, dict):
                raise _inspection_error(part)
            if any(_is_mtp(name) for name in state_dict):
                return True
    return False


def _converter_supports_no_mtp(converter_location):
    """Return whether the selected converter declares the `--no-mtp` option."""
    try:
        source = Path(converter_location).read_bytes()
    # Missing/unreadable path, or None and NUL-bearing ones: all mean "cannot
    # prove support", and omitting --no-mtp just restores the old behaviour.
    except (OSError, TypeError, ValueError):
        return False
    return re.search(
        rb"parser\.add_argument\([^)]*[\"']--no-mtp[\"']",
        source,
        flags = re.DOTALL,
    ) is not None


def _find_bitsandbytes_quantization(config, _path = "config.json"):
    """Where a bitsandbytes `quantization_config` sits, or None.

    Recursive: VLMs keep theirs under a sub-config like `text_config`.
    """
    if not isinstance(config, dict):
        return None
    quant = config.get("quantization_config")
    if isinstance(quant, dict):
        method = quant.get("quant_method")
        # Older checkpoints omit quant_method and only carry the bnb flags.
        if method == "bitsandbytes" or (
            method is None
            and ("load_in_4bit" in quant or "load_in_8bit" in quant)
        ):
            return _path
    for key, value in config.items():
        if key == "quantization_config" or not isinstance(value, dict):
            continue
        found = _find_bitsandbytes_quantization(value, f"{_path}[{key!r}]")
        if found is not None:
            return found
    return None


def _converter_was_oom_killed(exc):
    """Was the converter OOM-killed (SIGKILL), rather than failing on its own?"""
    if getattr(exc, "returncode", None) in (-9, 137):
        return True
    return "sigkill" in f"{exc}".lower()


# One line only: a failure that merely names the counter must not match, and
# `[^\S\r\n]` (not `\s`, which spans newlines) keeps the comparison off the
# stderr/stdout join.
_MTP_INFERENCE_ASSERTION = re.compile(
    r"\bassert\b[^\r\n]*\bopt_num_mtp_layers[^\S\r\n]*!=[^\S\r\n]*0"
)


def _converter_needs_no_mtp(text):
    """Did the converter abort while inferring an MTP head it could not find?"""
    if not text: return False
    return _MTP_INFERENCE_ASSERTION.search(text) is not None


def _converter_rejected_no_mtp(text):
    """Did the converter refuse `--no-mtp` because this architecture has no MTP?"""
    if not text: return False
    for line in text.splitlines():
        low = line.lower()
        if "--mtp" not in low and "--no-mtp" not in low and "--no-nextn" not in low:
            continue
        if "not supported" in low or "only supported" in low:
            return True
    return False


def _drop_no_mtp(command):
    """The same command without `--no-mtp`, or None when it has none to drop."""
    if "--no-mtp" not in command: return None
    return [token for token in command if token != "--no-mtp"]


def _add_no_mtp(command):
    """The same command with `--no-mtp`, or None when it already has it."""
    if "--no-mtp" in command: return None
    # The positional model directory must stay last.
    return command[:-1] + ["--no-mtp", command[-1]]


def _checkpoint_has_mtp_tensors(input_folder, num_layers):
    """Does the checkpoint carry an MTP head, whatever it declares?

    Unlike `_keep_mtp` this ignores the declaration, since the checkpoints that
    reach the assertion declare nothing. Only consulted after a failure, so an
    unreadable index answers "no evidence" and lets the retry proceed.
    """
    if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers <= 0:
        return False
    try:
        return _has_mtp_weight_tensors(input_folder, num_layers)
    except Exception:
        return False


def _retry_with_temp_file(command):
    """The same command spooling tensors to disk instead of holding them in RAM.

    llama.cpp refuses `--use-temp-file` alongside splitting ("Cannot use temp
    file when splitting"), so the split options are dropped. None when the
    command already has the flag, so the retry cannot loop.
    """
    if "--use-temp-file" in command:
        return None
    out = []
    drop_value = False
    for token in command:
        if drop_value:
            drop_value = False
            # Only its value: a flag here means the previous one had none.
            if not str(token).startswith("--"):
                continue
        if token in ("--split-max-size", "--split-max-tensors"):
            drop_value = True
            continue
        out.append(token)
    # The trailing model path must stay last.
    return out[:-1] + ["--use-temp-file"] + out[-1:]


def _gguf_output_paths(output_file):
    """`output_file` plus any shards llama.cpp names after it."""
    basename_without_gguf = os.path.splitext(output_file)[0]
    shard_pattern = re.compile(
        re.escape(os.path.basename(basename_without_gguf)) + r'-(\d{5})-of-(\d{5})\.gguf'
    )
    parent_dir = os.path.dirname(output_file) or '.'
    paths = [output_file]
    try:
        # fullmatch, not search: these get os.remove'd, and an unanchored match
        # would take a neighbour like "old-model.BF16-00001-of-00002".
        paths += sorted(os.path.join(parent_dir, f) for f in os.listdir(parent_dir)
                        if shard_pattern.fullmatch(f))
    except OSError:
        pass
    return paths


def _remove_gguf_outputs(output_file):
    """Delete what a failed or abandoned conversion left at `output_file`.

    The writer opens every shard with "wb" before any tensor byte and cleans up
    nothing on failure, and callers upload every save_directory/*.gguf, so a
    truncated leftover gets published as a valid artifact.
    """
    for path in _gguf_output_paths(output_file):
        try:
            os.remove(path)
        except OSError:
            pass


# GGUF converter / gguf-py version skew (unsloth#3581). Every converter
# entrypoint self-locates with `sys.path.insert(1, __file__/../gguf-py)`, which
# outranks PYTHONPATH and site-packages, so the child's gguf is decided by
# whatever tree sits beside the entrypoint, not by `use_local_gguf()`.
# Version numbers cannot settle which tree is right: the fork's gguf-py calls
# itself 0.19.0 while carrying architectures PyPI 0.19.0 lacks. So probe for the
# symbols the entrypoint needs and pin the child's gguf to the tree with them.

# How many conversion/*.py files to read when collecting requirements. A cap so
# an unexpected directory cannot turn the preflight into a filesystem walk.
_GGUF_REQUIREMENT_SCAN_LIMIT = 200

# The skew signatures seen in the wild, matched on the child's own output. Every
# one is anchored on a `gguf`-rooted name: a bare "type object 'X' has no
# attribute 'Y'" is the shape of an ordinary model-side AttributeError too, and
# the `"gguf" in text` prefilter does not separate them, because the converter is
# named convert_hf_to_gguf.py and logs every write as `INFO:gguf.gguf_writer:`.
_GGUF_SKEW_SIGNATURES = (
    re.compile(r"cannot import name ['\"]([^'\"]+)['\"] from ['\"](gguf[\w\.]*)['\"]"),
    re.compile(r"module ['\"](gguf[\w\.]*)['\"] has no attribute ['\"]([^'\"]+)['\"]"),
    re.compile(r"No module named ['\"](gguf[\w\.]*)['\"]"),
)

# `gguf.MODEL_ARCH.GEMMA4` raises "type object 'MODEL_ARCH' has no attribute
# 'GEMMA4'", which names the class but not the package, so the class has to be
# recognised some other way. The preflight already knows which `gguf` attribute
# chains the converter needs, so use those: an AttributeError on a class that
# appears in a requirement is ours, one on anything else is the model's.
_GGUF_ATTRIBUTE_ERROR = re.compile(
    r"(?:type object|object) ['\"]([\w]+)['\"] has no attribute ['\"]([^'\"]+)['\"]"
)


def _looks_like_gguf_skew(text, requirements = ()):
    """Whether the converter's output is the gguf version-skew failure.

    `requirements` is the `gguf` chains the preflight collected for this
    converter. Without them the bare attribute-error shape is not attributed to
    gguf at all, so a model-side AttributeError never collects the skew advice.
    """
    if not text:
        return False
    if "gguf" not in text:
        return False
    if any(pattern.search(text) for pattern in _GGUF_SKEW_SIGNATURES):
        return True
    owners = set()
    for requirement in requirements or ():
        # "gguf.MODEL_ARCH.GEMMA4" -> the intermediate names gguf owns.
        parts = str(requirement).split(".")
        owners.update(parts[1:-1])
    if not owners:
        return False
    return any(
        match.group(1) in owners for match in _GGUF_ATTRIBUTE_ERROR.finditer(text)
    )


def _importable_gguf_py(directory):
    """`<directory>/gguf-py` when it really holds an importable `gguf` package.

    A gguf-py directory can exist and still be useless (a wheel-layout folder
    with no package inside, or a leftover empty dir), so require the package
    __init__ rather than the parent.
    """
    if not directory:
        return None
    candidate = os.path.join(directory, "gguf-py")
    if os.path.isfile(os.path.join(candidate, "gguf", "__init__.py")):
        return candidate
    return None


def _gguf_requirements_from_source(source_bytes):
    """Requirements a single module places on `gguf`, as dotted expressions.

    Returns `(certain, advisory)`. Certain means it runs the moment the module is
    imported, so its absence is a guaranteed failure: module-level
    `import gguf...` / `from gguf... import X`, and any attribute chain rooted at
    a bare `gguf` evaluated at module or class-body scope (a class body executes
    on import, which is how `model_arch = gguf.MODEL_ARCH.GEMMA4` in
    conversion/gemma.py brings the whole converter down). Advisory means it may
    never be reached: inside a function body, or under a `try` that may be
    guarding for exactly this. Never raises; an unparseable file contributes
    nothing.
    """
    certain, advisory = set(), set()
    try:
        tree = ast.parse(source_bytes)
    except Exception:
        return certain, advisory

    def chain(node):
        """Longest dotted expression rooted at the bare name `gguf`, else None."""
        parts = []
        cursor = node
        while isinstance(cursor, ast.Attribute):
            parts.append(cursor.attr)
            cursor = cursor.value
        if isinstance(cursor, ast.Name) and cursor.id == "gguf" and parts:
            return "gguf." + ".".join(reversed(parts))
        return None

    # PEP 563: with `from __future__ import annotations` every annotation is a string and
    # nothing in it is evaluated at import time, so an annotation naming a missing symbol
    # costs nothing. Without it, an annotation is an ordinary expression in the signature.
    #
    # PEP 649 makes that the default from 3.14 on: an annotation becomes a lazily built
    # `__annotate__` function that import never calls, future import or not. The version
    # tested is this interpreter's because the converter child is launched as
    # `[sys.executable, converter_location]`, so the process that will evaluate these
    # annotations is this one.
    annotations_eager = sys.version_info < (3, 14) and not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in getattr(tree, "body", [])
    )

    def _eagerly_evaluated_signature(function, annotations_eager):
        """The expressions in a function's signature that run when the module is imported."""
        pieces = []
        pieces.extend(getattr(function, "decorator_list", []) or [])
        arguments = getattr(function, "args", None)
        if arguments is not None:
            pieces.extend(arguments.defaults or [])
            pieces.extend([item for item in (arguments.kw_defaults or []) if item is not None])
            if annotations_eager:
                if getattr(function, "returns", None) is not None:
                    pieces.append(function.returns)
                for group in ("posonlyargs", "args", "kwonlyargs"):
                    for argument in getattr(arguments, group, []) or []:
                        if argument.annotation is not None:
                            pieces.append(argument.annotation)
                for single in (arguments.vararg, arguments.kwarg):
                    if single is not None and single.annotation is not None:
                        pieces.append(single.annotation)
        return pieces

    def visit_expression(expression, eager):
        """Run the walk over one expression, as if it stood on its own at this level."""
        visit(ast.Module(body = [ast.Expr(value = expression)], type_ignores = []), eager)

    def visit_body(function, eager):
        """The statements of a function, without its signature."""
        for statement in function.body if isinstance(function.body, list) else [
            ast.Expr(value = function.body)
        ]:
            visit(ast.Module(body = [statement], type_ignores = []), eager)

    def visit(node, eager):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ImportFrom):
                module = child.module or ""
                if module == "gguf" or module.startswith("gguf."):
                    for alias in child.names:
                        target = certain if eager else advisory
                        target.add(module if alias.name == "*" else f"{module}.{alias.name}")
                continue
            if isinstance(child, ast.Import):
                for alias in child.names:
                    if alias.name == "gguf" or alias.name.startswith("gguf."):
                        (certain if eager else advisory).add(alias.name)
                continue
            if isinstance(child, ast.Attribute):
                expression = chain(child)
                if expression is not None:
                    (certain if eager else advisory).add(expression)
                    # Inner Attribute nodes are prefixes of this one; the probe
                    # resolves prefixes itself, so do not descend.
                    continue
            # A function body may never run, and a try body may be guarding for
            # a missing symbol on purpose. Both demote to advisory.
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                # The SIGNATURE is not the body. `@gguf.register` and
                # `def convert(kind = gguf.NEW_KIND)` are evaluated while the module is
                # imported, exactly like a module-level expression, so a symbol missing
                # from either one makes the import fail however the body is written. Only
                # the body keeps the demotion.
                for piece in _eagerly_evaluated_signature(child, annotations_eager):
                    visit_expression(piece, eager)
                visit_body(child, False)
            elif isinstance(child, ast.Try):
                # Body and handlers are advisory: a `try` body may be probing for a
                # symbol on purpose. `finally` is not, since it runs on every path,
                # so at module level a `gguf` name it reads fails the import as surely
                # as a plain statement would. Demoting it let the resolver keep an
                # incompatible baseline without probing a pin that would have worked.
                for part in (child.body, child.handlers, child.orelse):
                    for statement in part:
                        visit(ast.Module(body = [statement], type_ignores = []), False)
                for statement in child.finalbody:
                    visit(ast.Module(body = [statement], type_ignores = []), eager)
            elif isinstance(child, ast.If) and _is_type_checking_test(child.test):
                # `if TYPE_CHECKING: from gguf... import X` is the documented way to import a
                # name for annotations only, and the branch is FALSE at run time, so the
                # import never executes and a `gguf` without X still imports this module
                # fine. Counting it as certain made a working converter look unusable and
                # could repin or abandon it for a symbol nothing reads. Advisory, like a try
                # body. The else branch is the one that really runs, so it keeps `eager`.
                for statement in child.body:
                    visit(ast.Module(body = [statement], type_ignores = []), False)
                for statement in child.orelse:
                    visit(ast.Module(body = [statement], type_ignores = []), eager)
            elif isinstance(child, ast.If):
                # A module-level guard (`if sys.platform == "win32"`, a find_spec or
                # version test) runs at most one branch, so counting both as certain let
                # a symbol only the dead branch reads pin a different gguf. The test runs
                # either way and stays eager; the branches are advisory unless provable
                # above.
                visit_expression(child.test, eager)
                for statement in list(child.body) + list(child.orelse):
                    visit(ast.Module(body = [statement], type_ignores = []), False)
            else:
                visit(child, eager)

    visit(tree, True)
    return certain, advisory - certain


def _is_type_checking_test(test):
    """True for an `if` whose body cannot run at import time.

    `TYPE_CHECKING` is False at run time by definition, under any of the spellings the
    converters use -- the bare name, `typing.TYPE_CHECKING`, `t.TYPE_CHECKING` -- and so is a
    literal `if False:`. Nothing else is claimed: an `if` this cannot prove false keeps its
    eager reading, which is the conservative direction.
    """
    if isinstance(test, ast.Constant):
        return test.value is False
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _extract_dict_values_from_conversion_init(conv_init_path, dict_name):
    """`{architecture: module}` out of conversion/__init__.py's TEXT_MODEL_MAP /
    MMPROJ_MODEL_MAP. The sibling of _extract_dict_keys_from_conversion_init,
    which needs only the keys."""
    mapping = {}
    try:
        with open(conv_init_path, "rb") as f:
            tree = ast.parse(f.read())
    except Exception:
        return mapping
    def _harvest(value):
        if not isinstance(value, ast.Dict):
            return
        for key, item in zip(value.keys, value.values):
            if (isinstance(key, ast.Constant) and isinstance(key.value, str)
                    and isinstance(item, ast.Constant) and isinstance(item.value, str)):
                mapping[key.value] = item.value
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == dict_name:
                    _harvest(node.value)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == dict_name:
                _harvest(node.value)
    return mapping


def _conversion_modules_for(conversion_dir, architecture, is_vlm = None):
    """The conversion/ modules this conversion will really import.

    Only these place a CERTAIN requirement on `gguf`. conversion/__init__.py
    imports base.py eagerly, and `get_model_class` then imports the one module
    the architecture maps to; `load_all_models` imports the rest inside a
    per-module `try/except Exception` that only warns. So a name referenced in
    some other architecture's module cannot break this conversion, and treating
    it as certain downgrades a working export to an older converter for nothing.
    An unknown architecture scopes to the two eager modules, which is the
    conservative direction: fewer certain names means fewer reasons to switch.

    `is_vlm` is which halves this call converts. A text-only export never imports
    the projector module, so counting its names as certain would let a symbol only
    the projector needs move a working text conversion onto another converter.
    None means both, which is the answer when the caller does not know.
    """
    eager = [os.path.join(conversion_dir, "__init__.py"),
             os.path.join(conversion_dir, "base.py")]
    modules = [path for path in eager if os.path.isfile(path)]
    if not architecture:
        return modules
    conv_init = os.path.join(conversion_dir, "__init__.py")
    selected = set()
    for dict_name in _conversion_maps_in_play(is_vlm):
        module = _extract_dict_values_from_conversion_init(conv_init, dict_name).get(architecture)
        if module:
            selected.add(module)
    for module in sorted(selected):
        path = os.path.join(conversion_dir, f"{module}.py")
        if os.path.isfile(path):
            modules.append(path)
    return modules


def _conversion_maps_in_play(is_vlm):
    """The conversion/__init__.py maps a call with this `is_vlm` really dispatches
    through. Every export runs the text half; only a VLM export adds the projector.
    None means the caller does not know, so both stand."""
    if is_vlm is False:
        return ("TEXT_MODEL_MAP",)
    return ("TEXT_MODEL_MAP", "MMPROJ_MODEL_MAP")


def _converter_architecture_maps(script_path, architecture, is_vlm = None):
    """Which of a converter's maps name *architecture*, as a set of dict names.

    The pair is not interchangeable. ``TEXT_MODEL_MAP`` dispatches the text conversion, which
    every export runs, and ``MMPROJ_MODEL_MAP`` dispatches the projector, which a VLM export
    runs in addition. A fallback that maps the architecture in only one of them therefore
    fails the other half outright, and the support check that would have caught it describes
    the converter that was REQUESTED rather than the one substituted in.

    Empty for a monolith entrypoint (no package to read) and for an unreadable __init__.py,
    which are failures to look rather than answers.
    """
    if not architecture:
        return set()
    try:
        with open(script_path, "rb") as f:
            entry_source = f.read()
    except OSError:
        return set()
    if b"from conversion import" not in entry_source:
        return set()
    conversion_dir = os.path.join(os.path.dirname(script_path) or ".", "conversion")
    conv_init = os.path.join(conversion_dir, "__init__.py")
    if not os.path.isfile(conv_init):
        return set()
    return {
        dict_name
        for dict_name in _conversion_maps_in_play(is_vlm)
        if architecture in _extract_dict_values_from_conversion_init(conv_init, dict_name)
    }


def _converter_maps_architecture(script_path, architecture, required_maps = None):
    """Whether this converter's OWN maps name *architecture*.

    Asked of a fallback before it is chosen. The requirement scan is scoped to the modules a
    conversion will really import, and a converter that predates the architecture maps it to
    nothing: the scan then sees only the two eager modules, the candidate misses nothing
    certain, and it ranks as a perfect match for a model it cannot convert at all. The support
    check that would have caught it reads the arch sets of the converter that was REQUESTED,
    not of the one being substituted, so it does not fire either, and the export dies on
    "unsupported model" with a working candidate left unused further down the ranking.

    ``required_maps`` is which of them have to name it, read off the converter that was
    REQUESTED: the text map dispatches the text conversion that every export runs, the
    projector map the mmproj half a VLM export adds, and a fallback that maps the
    architecture in only one of them fails the other outright. Taken from the request rather
    than from a flag, because the request is what this is falling back FROM and it already
    knows which halves this architecture has. Absent, presence in either map stands, which is
    the answer for a request whose own maps could not be read.

    True on anything that is not positive evidence of absence: a monolith entrypoint with no
    conversion package dispatches by class registration rather than by these maps, an
    unreadable __init__.py is a failure to look, and no architecture to check is not a
    question. Only a converter whose maps are readable, non-empty and do not contain the
    architecture is refused.
    """
    if not architecture:
        return True
    # The same structural signal the requirement scan uses: a package sitting beside a
    # monolith entrypoint (left by a newer install in the same directory, which is where every
    # sibling candidate lives) is not that entrypoint's dispatch table, and reading it would
    # answer for the wrong converter in both directions.
    try:
        with open(script_path, "rb") as f:
            entry_source = f.read()
    except OSError:
        return True
    if b"from conversion import" not in entry_source:
        return True
    conversion_dir = os.path.join(os.path.dirname(script_path) or ".", "conversion")
    conv_init = os.path.join(conversion_dir, "__init__.py")
    if not os.path.isfile(conv_init):
        return True
    mapped = {
        dict_name: _extract_dict_values_from_conversion_init(conv_init, dict_name)
        for dict_name in ("TEXT_MODEL_MAP", "MMPROJ_MODEL_MAP")
    }
    if not any(mapped.values()):
        return True
    if required_maps:
        return all(architecture in mapped.get(name, {}) for name in required_maps)
    return any(architecture in one for one in mapped.values())


def _converter_gguf_requirements(script_path, architecture = None, is_vlm = None):
    """Everything the entrypoint and the conversion/ modules this conversion will
    actually import need from `gguf`. Returns `(certain, advisory)` as sorted
    tuples so the result is hashable and stable for the probe cache.

    Modules that will not be imported for `architecture` still contribute, but
    only as advisory: they cannot break this export.
    """
    certain, advisory = set(), set()
    sources = [script_path]
    advisory_sources = []
    try:
        with open(script_path, "rb") as f:
            entry_source = f.read()
    except OSError:
        entry_source = b""
    # A monolith entrypoint does not import conversion/, so a package sitting
    # beside it (left by a newer install) places no requirement on it. Same
    # structural signal _detect_converter_layout uses.
    conversion_dir = os.path.join(os.path.dirname(script_path) or ".", "conversion")
    if b"from conversion import" in entry_source and os.path.isdir(conversion_dir):
        imported = _conversion_modules_for(conversion_dir, architecture, is_vlm)
        sources.extend(imported)
        imported_set = {os.path.abspath(path) for path in imported}
        try:
            names = sorted(os.listdir(conversion_dir))
        except OSError:
            names = []
        for name in names[:_GGUF_REQUIREMENT_SCAN_LIMIT]:
            path = os.path.join(conversion_dir, name)
            if name.endswith(".py") and os.path.abspath(path) not in imported_set:
                advisory_sources.append(path)
    for path in sources:
        try:
            with open(path, "rb") as f:
                source = f.read()
        except OSError:
            continue
        module_certain, module_advisory = _gguf_requirements_from_source(source)
        certain |= module_certain
        advisory |= module_advisory
    # Another architecture's module: it is imported only by load_all_models,
    # which swallows the failure, so nothing here can break this conversion.
    for path in advisory_sources:
        try:
            with open(path, "rb") as f:
                source = f.read()
        except OSError:
            continue
        module_certain, module_advisory = _gguf_requirements_from_source(source)
        advisory |= module_certain
        advisory |= module_advisory
    # An attribute chain that is also a certain requirement adds nothing here.
    return tuple(sorted(certain)), tuple(sorted(advisory - certain))


# Resolves a dotted expression rooted at `gguf` inside the child interpreter:
# import the longest importable module prefix, then getattr the remainder.
_GGUF_PROBE_SOURCE = r"""
import importlib, json, os, sys

converter = sys.argv[1]
# On stdin, not in argv. The requirement list is one name per conversion/ module
# and grows with every architecture llama.cpp adds (462 names, 16.8 KB of argv on
# the b10909 bundle), while Windows caps a CreateProcess command line at 32767
# characters. stdin has no such ceiling.
requirements = json.loads(sys.stdin.read() or "[]")

# Reproduce the entrypoint's own self-location exactly, or the probe would
# measure a different `gguf` than the real run: every llama.cpp converter (and
# its conversion/base.py) does this before `import gguf`, and `python -c` has no
# script directory of its own to do it for us.
if converter and "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(converter)), "gguf-py"))

def resolve(expression):
    parts = expression.split(".")
    module = None
    index = 0
    for stop in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:stop]))
        except Exception:
            continue
        index = stop
        break
    if module is None:
        return False
    target = module
    for attribute in parts[index:]:
        try:
            target = getattr(target, attribute)
        except Exception:
            return False
    return True

# The sentinel is what makes the report identifiable as ours. The parent reads the
# last brace-line of stdout, and anything the converter's imports print there would
# otherwise be parsed as a report: a dict with no "missing" key reads as "nothing
# missing", which ranks an unprobed candidate as a perfect match and switches the
# converter on the strength of a line we never wrote.
report = {"unsloth_gguf_probe": 1, "missing": [], "location": None, "version": None}
try:
    import gguf
    report["location"] = getattr(gguf, "__file__", None)
except Exception as error:
    report["error"] = "%s: %s" % (type(error).__name__, error)
    print(json.dumps(report))
    raise SystemExit(0)
try:
    import importlib.metadata as metadata
    report["version"] = metadata.version("gguf")
except Exception:
    pass
try:
    tree = os.path.dirname(os.path.dirname(os.path.abspath(gguf.__file__)))
    with open(os.path.join(tree, "pyproject.toml")) as handle:
        for line in handle:
            if line.strip().startswith("version"):
                report["version"] = line.split("=", 1)[1].strip().strip('"\'')
                break
except Exception:
    pass
for expression in requirements:
    if not resolve(expression):
        report["missing"].append(expression)
print(json.dumps(report))
"""


def _probe_child_gguf(python_exe, env, requirements, converter_location = None, timeout = 120):
    """Ask the child interpreter which `gguf` it resolves and what it cannot
    satisfy. Returns a dict with `missing`, `location`, `version`, `error`, or
    None when the probe itself could not run (never fail an export because a
    diagnostic did not work)."""
    try:
        completed = subprocess.run(
            [python_exe, "-c", _GGUF_PROBE_SOURCE, str(converter_location or "")],
            # The names go on stdin so the command line stays a fixed size; see
            # the comment in _GGUF_PROBE_SOURCE for the Windows argv ceiling.
            input = json.dumps(list(requirements)),
            env = env,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            encoding = "utf-8",
            errors = "replace",
            timeout = timeout,
        )
    except Exception as error:
        logger.debug("Unsloth: gguf preflight probe could not run (%s).", error)
        return None
    for line in reversed((completed.stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        # Only our own report counts. Without the sentinel any JSON the converter's
        # imports happen to print is read as a report, and one with no "missing" key
        # scores a perfect zero, so a candidate nothing ever probed can be adopted
        # and announced as matching.
        if isinstance(parsed, dict) and parsed.get("unsloth_gguf_probe") == 1:
            return parsed
    logger.debug(
        "Unsloth: gguf preflight probe returned no report (exit %s).", completed.returncode
    )
    return None


def _converter_child_env(gguf_py_dir = None, base_env = None):
    """The environment a converter child should run with.

    Always a copy of the parent environment, so PATH, HOME, HF_*, CUDA_* and the
    active virtualenv keep working. When `gguf_py_dir` is given it is prepended
    to PYTHONPATH with os.pathsep (Windows uses ';'), and NO_LOCAL_GGUF is set so
    the entrypoint stops inserting its own sibling tree ahead of our choice.
    Idempotent: prepending a directory already at the front changes nothing.
    """
    env = dict(os.environ if base_env is None else base_env)
    if not gguf_py_dir:
        return env
    gguf_py_dir = os.path.abspath(gguf_py_dir)
    existing = env.get("PYTHONPATH") or ""
    entries = [entry for entry in existing.split(os.pathsep) if entry]
    entries = [entry for entry in entries if os.path.abspath(entry) != gguf_py_dir]
    env["PYTHONPATH"] = os.pathsep.join([gguf_py_dir] + entries)
    # Only meaningful together with the pin above: on its own it would demote the
    # entrypoint to whatever site-packages happens to hold.
    env["NO_LOCAL_GGUF"] = "1"
    return env


def _gguf_report_summary(report, blocking):
    """Short human phrase for what a probe report found wrong. `blocking` is the
    subset that certainly runs, which is what actually breaks the conversion, so
    it is named rather than the full advisory list."""
    if not report:
        return "reason unknown"
    if report.get("error"):
        return f"gguf is not importable: {report['error']}"
    missing = list(blocking or ())
    if missing:
        return "missing " + ", ".join(missing[:3])
    return "reason unknown"


def _gguf_blocking_missing(report, certain):
    """The unsatisfied names that certainly run, i.e. the ones that break it."""
    return sorted(set(report.get("missing") or ()) & set(certain or ()))


def _installed_gguf_tree(python_exe):
    """The directory holding the installed `gguf` package, as the child sees it with
    the converter's own sibling tree suppressed, or None.

    This is the candidate the other three cannot express. Someone who upgraded the
    `gguf` wheel past the llama.cpp checkout on disk has a satisfying `gguf` already
    installed, but every llama.cpp entrypoint puts its sibling `gguf-py` ahead of it
    with `sys.path.insert(1, ...)`, so the wheel only wins if it is pinned like any
    other tree. Discovered by asking the child rather than by importing `gguf` here:
    the parent may resolve a different one, and importing it would be a side effect.
    """
    # All Unsloth Zoo code licensed under LGPLv3
    env = dict(os.environ)
    env["NO_LOCAL_GGUF"] = "1"
    report = _probe_child_gguf(python_exe, env, ())
    location = (report or {}).get("location")
    if not location:
        return None
    tree = os.path.dirname(os.path.dirname(os.path.abspath(location)))
    if not os.path.isfile(os.path.join(tree, "gguf", "__init__.py")):
        return None
    return tree
pass


def _gguf_candidate_converters(converter_location):
    """(converter, gguf_py_dir, label) candidates, best first.

    0. The converter as asked for, with the environment untouched. This is the
       pre-existing behaviour and is always tried first, so a consistent install
       sees no change at all.
    1. The same converter with an explicit gguf-py pinned: its own sibling tree,
       then the installed bundle's. Keeps the newer converter, which is what we
       want whenever a satisfying tree exists anywhere.
    2. A sibling entrypoint co-versioned with its own gguf-py. Last resort: it
       may be considerably older than the downloaded one, so switching to it is
       announced rather than silent.
    """
    converter_dir = os.path.dirname(os.path.abspath(converter_location)) or "."
    candidates = [(converter_location, None, "as configured")]

    seen = set()
    for directory, label in (
        (converter_dir, "the converter's own gguf-py"),
        (LLAMA_CPP_DEFAULT_DIR, "the installed llama.cpp gguf-py"),
    ):
        gguf_py = _importable_gguf_py(directory)
        if gguf_py is None or gguf_py in seen:
            continue
        seen.add(gguf_py)
        candidates.append((converter_location, gguf_py, label))

    requested = os.path.abspath(converter_location)
    for name in LLAMA_CPP_CONVERTER_FILENAMES:
        sibling = os.path.join(converter_dir, name)
        if not os.path.isfile(sibling) or os.path.abspath(sibling) == requested:
            continue
        candidates.append((
            sibling,
            _importable_gguf_py(converter_dir),
            "the llama.cpp checkout's own converter",
        ))
    return candidates


def _gguf_tree_of_location(location):
    """The sys.path entry that makes the `gguf` at `location` importable, or None.

    `location` is the child's `gguf.__file__`, so `<tree>/gguf/__init__.py`. Anything that
    is not laid out as a package directory named `gguf` is rejected rather than guessed at:
    putting the wrong directory on the parent's sys.path is worse than not reading the file
    back with the child's tree.
    """
    if not isinstance(location, str) or not location:
        return None
    package = os.path.dirname(os.path.abspath(location))
    if os.path.basename(package) != "gguf":
        return None
    if not os.path.isfile(os.path.join(package, "__init__.py")):
        return None
    return os.path.dirname(package)


def _gguf_readback_tree(report):
    """The tree to read a fresh GGUF back with: the one the converter child used.

    Candidate zero -- the requested converter with the environment untouched -- is the
    common path and deliberately pins nothing, because changing a working child's
    environment is the one thing this resolver must not do. But the pin was also what told
    `_verify_converted_gguf` which `gguf` wrote the file, so on that path the read-back ran
    against whatever the PARENT happens to have. Where the parent has a different package,
    every reader comes back None and the whole gate -- the required metadata, the tensor
    sanity pass and the missing-shard check -- degrades into one warning and the file is
    published unverified.

    The probe report already names the package the child resolved, so the read-back tree is
    derivable without touching the child at all.

    Returned whenever it is derivable, with no version comparison. Version ordering cannot
    establish compatibility here: llama.cpp's vendored `gguf-py` and the PyPI `gguf` wheel
    both report their own numbers, a fork and a release can report the SAME number with
    different contents, and a parent that omits `__version__` says nothing at all. The only
    thing that is actually known is which package wrote the bytes, and that is what the
    file has to be read with. It is also what the pinned path already does unconditionally,
    so the two paths now agree rather than applying different rules to the same question.

    `use_local_gguf` snapshots and restores `sys.path` and every `gguf` module, so this
    scopes the pin to the read-back and leaves an outer caller's resolution alone.
    """
    return _gguf_tree_of_location((report or {}).get("location"))


def _resolve_converter_and_gguf(converter_location, python_exe, architecture = None,
                                is_vlm = None):
    """Pick the (converter, gguf-py) pair whose `gguf` can satisfy the converter.

    Returns `(converter_location, gguf_py_dir, report)`.

    `gguf_py_dir` is the gguf-py to pin, or None to leave the child's resolution
    exactly as it is today. A directory rather than a finished environment on
    purpose: the launch sites build the env from the live os.environ at launch
    time, so anything that edits it in between is still honoured.

    `report` is the probe result for the pair we chose, kept for the failure
    diagnosis.

    Conservative by design, in two steps.

    Candidate 0 is the requested pair with the environment untouched. If nothing
    it CERTAINLY needs is missing we return it and change nothing, so a false
    positive in the requirement scan can never turn a working export into a
    refusal and an installed gguf-py that is merely one unrelated architecture
    behind never costs anyone their converter.

    Only once candidate 0 is provably broken do we look at the rest, and then we
    rank them rather than taking the first that clears the certain bar: fewest
    certain misses, then fewest advisory misses. The tie-break matters. A
    candidate can satisfy every name that certainly runs and still fail on one
    that runs in practice, because a reference inside a function body of
    `conversion/base.py` is advisory by construction and yet executes on every
    conversion (`gguf.LazyChunkedTensor` is exactly this). Preferring the
    candidate with no advisory misses at all picks the genuinely co-versioned
    tree instead of a merely plausible one.
    """
    requested_certain, requested_advisory = _converter_gguf_requirements(
        converter_location, architecture, is_vlm,
    )
    if not requested_certain and not requested_advisory:
        return converter_location, None, None

    # Which halves of the conversion this architecture has, read off the converter that was
    # asked for. A fallback has to serve the same ones.
    requested_maps = _converter_architecture_maps(converter_location, architecture, is_vlm)
    candidates = _gguf_candidate_converters(converter_location)
    baseline_report = None
    baseline_blocking = ()
    ranked = []
    for index, (candidate, gguf_py, label) in enumerate(candidates):
        # Loop-local, so no iteration can read another candidate's requirements.
        if candidate == converter_location:
            certain, advisory = requested_certain, requested_advisory
        else:
            # Before the probe, which costs a subprocess: a converter that does not map this
            # architecture cannot convert this model however well its gguf matches, and its
            # empty requirement scan is exactly what makes it rank first.
            if not _converter_maps_architecture(candidate, architecture, requested_maps):
                continue
            certain, advisory = _converter_gguf_requirements(candidate, architecture, is_vlm)
        env = _converter_child_env(gguf_py)
        report = _probe_child_gguf(
            python_exe, env, tuple(certain) + tuple(advisory), candidate,
        )
        if report is None:
            # Probe unavailable: keep today's behaviour.
            return converter_location, None, None
        blocking = _gguf_blocking_missing(report, certain)
        if index == 0:
            baseline_report = report
            baseline_blocking = blocking
            if not (report.get("error") or blocking):
                # Either nothing is missing, or only names that may never be
                # reached. Never move off the requested pair on a maybe.
                return converter_location, gguf_py, report
            # Only now, and only once: the installed `gguf` wheel, with the
            # converter's sibling tree out of the way. Probing for it costs a
            # subprocess, so it is not paid by an install that already works.
            installed_tree = _installed_gguf_tree(python_exe)
            known = {
                os.path.abspath(tree) for _candidate, tree, _label in candidates if tree
            }
            if installed_tree and os.path.abspath(installed_tree) not in known:
                # Ahead of the older trees: the requested converter is the newest
                # thing here, so the newest gguf is the one most likely to match it.
                candidates.insert(1, (
                    converter_location, installed_tree, "the installed gguf package",
                ))
            continue
        if report.get("error"):
            continue
        missing = set(report.get("missing") or ())
        ranked.append((
            len(blocking), len(missing & set(advisory)), index,
            candidate, gguf_py, label, report,
        ))

    ranked.sort(key = lambda row: row[:3])
    # Only a candidate that misses nothing certain is an improvement. One that
    # still misses a certainly-executed name is no better than the request, and
    # pinning it would change the environment for nothing.
    if ranked and ranked[0][0] == 0:
        _certain_miss, _advisory_miss, _index, candidate, gguf_py, label, report = ranked[0]
        if candidate != converter_location:
            print(
                f"Unsloth: The GGUF converter at {converter_location} needs a newer "
                f"gguf than this llama.cpp install provides "
                f"({_gguf_report_summary(baseline_report, baseline_blocking)}). Falling back to "
                f"{label} at {candidate}, which matches it. The GGUF will be produced "
                f"by that older converter; update llama.cpp to use the newer one."
            )
        else:
            logger.info(
                "Unsloth: Pinning the GGUF converter's gguf to %s (%s).", gguf_py, label
            )
        return candidate, gguf_py, report

    # Nothing satisfies it. Run exactly what was asked for, as before, and let
    # the failure path attach the diagnosis.
    return converter_location, None, baseline_report


def _gguf_skew_diagnosis(converter_location, python_exe, report = None, architecture = None):
    """One actionable paragraph naming the gguf the child resolved, what it could
    not satisfy, and the remedy. Never raises."""
    try:
        certain, advisory = _converter_gguf_requirements(converter_location, architecture)
        if report is None:
            report = _probe_child_gguf(
                python_exe, _converter_child_env(), tuple(certain) + tuple(advisory),
                converter_location,
            )
        if report is None:
            return ""
        lines = [
            "",
            "--- unsloth: gguf version skew ---",
            f"converter : {converter_location}",
            f"gguf      : {report.get('location') or 'not importable'}"
            + (f" (version {report['version']})" if report.get("version") else ""),
        ]
        if report.get("error"):
            lines.append(f"import    : {report['error']}")
        blocking = _gguf_blocking_missing(report, certain)
        missing = list(blocking) or (report.get("missing") or [])
        if missing:
            lines.append(f"missing   : {', '.join(missing[:8])}")
        lines.append(
            "This converter needs a newer gguf than the one it resolved. Delete the "
            "llama.cpp folder so Unsloth reinstalls a matching one, or point "
            "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR at a llama.cpp checkout whose gguf-py "
            "matches its convert_hf_to_gguf.py."
        )
        return "\n".join(lines)
    except Exception:
        return ""


def _verify_run_outputs(files, description, required, quantization_type,
                        print_output = False, gguf_py_dir = None):
    """Read one converter run's outputs back. True to keep them, False to drop them.

    A REQUIRED run raises, which is the export failing. An OPTIONAL run, i.e. the
    projector, does not: a converter that exits 0 while writing an unreadable or
    structurally invalid GGUF is the same outcome for the user as one that exits
    non-zero, and that case is already handled as a text-only downgrade. Aborting
    here instead would throw away a text model that is present and valid, which is
    the opposite of what the optional handling exists to do. The bad projector is
    removed so that nothing uploads it."""
    try:
        _verify_converted_gguf(
            files, quantization_type, print_output = print_output,
            gguf_py_dir = gguf_py_dir,
        )
        return True
    except Exception as error:
        if required: raise
        for path in files: _remove_gguf_outputs(path)
        print(
            f"Unsloth: The {description} converted but did not pass verification "
            f"({error}). It has been removed. The text model was converted "
            f"successfully and is usable; only multimodal (image/audio) input is "
            f"unavailable in this GGUF."
        )
        return False


def convert_to_gguf(
    model_name,
    input_folder,
    model_dtype = "bf16",
    quantization_type = "bf16", # dequantizing from q8_0 disallow, setting default to bf16
    converter_location = os.path.join(LLAMA_CPP_DEFAULT_DIR, "unsloth_convert_hf_to_gguf.py"),
    supported_text_archs = None,
    supported_vision_archs = None,
    is_vlm = False,
    is_gpt_oss = False,
    max_shard_size = "50GB",
    print_output = False,
    print_outputs = None,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Converts to GGUF using convert_hf_to_gguf.py. Quantization handled by quantize_gguf.

    max_shard_size = check_max_shard_size(max_shard_size)
    if quantization_type not in ["None", "f32", "f16", "bf16", "q8_0"]:
        quantization_type, _ = check_quantization_type(quantization_type)

    if not os.path.exists(input_folder):
        raise RuntimeError(f"Unsloth: `{input_folder}` does not exist?")

    config_path = os.path.join(input_folder, "config.json")
    if not os.path.exists(config_path):
        raise RuntimeError(f"Unsloth: `config.json` does not exist inside `{input_folder}`.")

    # Load config.json
    with open(config_path, "r", encoding = "utf-8") as f:
        config_file = json.load(f)

    _bnb_where = _find_bitsandbytes_quantization(config_file)
    if _bnb_where is not None:
        # llama.cpp has no bitsandbytes dequantizer and only refuses after
        # reading the whole model, so fail here instead of after a multi-GB
        # download. Both flags are named: 8bit checkpoints hit this too.
        raise RuntimeError(
            f"Unsloth: `{input_folder}` still holds bitsandbytes quantized "
            f"weights (`quantization_config` at {_bnb_where}), and llama.cpp "
            f"cannot convert those to GGUF.\n"
            f"GGUF export needs dequantized 16bit weights. Either load the "
            f"model with `load_in_4bit = False` and `load_in_8bit = False` "
            f"before saving, or merge a LoRA adapter with "
            f"`save_method = \"merged_16bit\"`, which downloads the original "
            f"16bit weights. Saving a quantized model that has no adapter does "
            f"not dequantize it."
        )

    # Decide text-only versus text-plus-projector BEFORE resolving the converter, so the
    # resolution is scoped to the halves this call will really convert. Downgrading
    # afterwards would leave the resolver having treated the projector module's `gguf`
    # names as certain for a conversion that never imports it.
    if is_vlm and supported_vision_archs is not None:
        if "architectures" in config_file:
            arch = config_file["architectures"][0]
        else:
            arch = None  # MLX-style config; skip mmproj arch check
        if arch is not None and arch not in supported_vision_archs:
                is_vlm = False
                print(f"Unsloth: {arch} is not supported for MMPROJ conversion. Converting as text-only model.")

    # Resolve, once, which converter and which `gguf` the child will run with.
    # Untouched when the install is consistent; see _resolve_converter_and_gguf.
    # The architecture goes in so that only the conversion/ module this export
    # really imports counts as a hard requirement: every other architecture's
    # module is imported by load_all_models, which swallows the failure.
    _architectures = config_file.get("architectures") or ()
    _architecture = _architectures[0] if _architectures else None
    converter_location, _gguf_py_pin, _gguf_report = _resolve_converter_and_gguf(
        converter_location, sys.executable, _architecture, is_vlm,
    )

    # The converter sizes block_count from the config, so keep `mtp_num_hidden_layers`
    # only when the weights still carry the MTP layer (else it crashes on the extra
    # tensor), and strip it when they don't (else it errors on the missing one).
    # `unsloth_fixed_mtp` is an internal marker, always dropped. Only Qwen3.5/3.6 set
    # these keys, so other arches are untouched.
    _tc = config_file.get("text_config")
    if not isinstance(_tc, dict):
        _tc = {}
    _mtp_declared = "mtp_num_hidden_layers" in config_file or "mtp_num_hidden_layers" in _tc
    _num_layers = _tc.get("num_hidden_layers", config_file.get("num_hidden_layers"))
    if _mtp_declared and (
        not isinstance(_num_layers, int)
        or isinstance(_num_layers, bool)
        or _num_layers <= 0
    ):
        raise ValueError(
            "Unsloth: `num_hidden_layers` must be a positive integer to reconcile MTP tensors; "
            "`config.json` was not changed."
        )
    _keep_mtp = _mtp_declared and _has_mtp_weight_tensors(input_folder, _num_layers)
    _no_mtp = (
        _mtp_declared
        and not _keep_mtp
        and _converter_supports_no_mtp(converter_location)
    )
    # Keep the declaration when `--no-mtp` carries the intent: deleting it made
    # a retry or second export read none, omit the flag, and hit the assertion.
    _strip_keys = (
        ("unsloth_fixed_mtp",)
        if _keep_mtp or _no_mtp
        else ("unsloth_fixed_mtp", "mtp_num_hidden_layers")
    )
    _changed = False
    for _cfg in (config_file, _tc):
        if not _cfg:
            continue
        for _key in _strip_keys:
            if _key in _cfg:
                _cfg.pop(_key)
                _changed = True
    if _changed:
        with open(config_path, "w", encoding = "utf-8") as f:
            json.dump(config_file, f, indent = 2)
    pass

    # Check if arch is supported
    supported_types = (supported_vision_archs or set()) | (supported_text_archs or set())
    if supported_types and "architectures" in config_file:
        arch = config_file["architectures"][0]
        if arch not in supported_types:
            raise NotImplementedError(
                f"Unsloth: llama.cpp GGUF conversion does not yet support "\
                f"converting model types of `{arch}`."
            )
    pass

    all_output_files = []
    # Which files came from which run, and whether that run was required. The
    # read-back below needs it: a projector is an OPTIONAL run, and its failure is
    # already a text-only downgrade rather than an aborted export.
    verify_groups = []
    runs_to_do = []

    if is_vlm:
        # VLM: dual conversion (text + mmproj)
        if not model_name.endswith(".gguf") and quantization_type == "None":
            text_output = f"{model_name}.{model_dtype.upper()}.gguf"
            mmproj_output = f"{model_name}.{model_dtype.upper()}-mmproj.gguf"
        else:
            if model_name.endswith(".gguf"):
                base_name = model_name[:-5]
                text_output = model_name
                # Fix: mmproj should always include dtype since it's not quantized
                mmproj_dtype = model_dtype if model_dtype else ("bf16" if device_is_bf16_supported() else "f16")
                mmproj_output = f"{base_name}.{mmproj_dtype.upper()}-mmproj.gguf"
            else:
                text_output = f"{model_name}.{quantization_type.upper()}.gguf"
                mmproj_dtype = model_dtype if model_dtype else ("bf16" if device_is_bf16_supported() else "f16")
                mmproj_output = f"{model_name}.{mmproj_dtype.upper()}-mmproj.gguf"

        # Text model conversion
        if quantization_type == "None":
            text_args = {
                "--outfile"        : text_output,
                "--split-max-size" : max_shard_size,
            }
        else:
            text_args = {
                "--outfile"        : text_output,
                "--outtype"        : quantization_type,
                "--split-max-size" : max_shard_size,
            }
        if _no_mtp:
            text_args["--no-mtp"] = ""
        runs_to_do.append((text_args, text_output, "text model", True))

        # Vision projector conversion
        mmproj_args = {
            "--outfile"        : mmproj_output,
            "--outtype"        : model_dtype if model_dtype else "bf16" if device_is_bf16_supported() else "f16",
            "--mmproj"         : "",
            "--split-max-size" : max_shard_size,
        }
        # Optional: a projector failure must not discard the text GGUF above.
        runs_to_do.append((mmproj_args, mmproj_output, "vision projector", False))

    else:
        if is_gpt_oss:
        # GPT-OSS models always preserve mxfp4 quantization regardless of user input
            final_output = f"{model_name}.MXFP4.gguf"
        # Non-VLM: single conversion
        elif quantization_type == "None":
            if is_gpt_oss:
                final_output = f"{model_name}.MXFP4.gguf"
            else:
                final_output = f"{model_name}.{model_dtype.upper()}.gguf"
        else:
            final_output = model_name if model_name.endswith(".gguf") else f"{model_name}.{quantization_type.upper()}.gguf"

        if quantization_type == "None":
            args = {
                "--outfile"        : final_output,
                "--split-max-size" : max_shard_size,
            }
        else:
            args = {
                "--outfile"        : final_output,
                "--outtype"        : quantization_type,
                "--split-max-size" : max_shard_size,
            }
        if _no_mtp:
            args["--no-mtp"] = ""
        runs_to_do.append((args, final_output, "model", True))

    # A bare --outfile lands in the process CWD. On Windows that CWD is often not writable
    # (app launched from a protected dir), so the final write failed with PermissionError
    # [Errno 13] even though conversion succeeded. Only then redirect the bare name into
    # input_folder; a writable CWD (Linux/Mac/Colab) is left unchanged.
    def _dir_is_writable(d):
        # mkstemp is exclusive: never truncates an existing file or follows a symlink.
        try:
            fd, probe = tempfile.mkstemp(prefix=".unsloth_write_test_", dir=d)
            os.close(fd)
            os.remove(probe)
            return True
        except Exception:
            return False
    _cwd_writable = _dir_is_writable(os.getcwd())

    # Execute conversions
    for args, output_file, description, required in runs_to_do:
        # Redirect only a bare filename under an unwritable CWD. Absolute paths and
        # relative paths with a directory are the caller's choice; input_folder is probed
        # too since it may be a read-only model source.
        if (not _cwd_writable
                and not os.path.isabs(output_file)
                and os.path.dirname(output_file) == ""):
            _dst = os.path.abspath(input_folder)
            if _dir_is_writable(_dst):
                output_file = os.path.join(_dst, output_file)
                args = {**args, "--outfile": output_file}
        if print_output: print(f"\nUnsloth: Converting {description}...")
        command = [sys.executable, converter_location]
        for key, value in args.items():
            # Keep flag-only options (eg `--mmproj`) as standalone args.
            if value in (None, ""):
                command.append(str(key))
            else:
                command.extend([str(key), str(value)])
        command.append(str(input_folder))

        # Run the converter; self-heal and retry once if the env (not the model)
        # is broken. No cost on the happy path.
        attempted_repair = False
        attempted_temp_file = False
        attempted_no_mtp_drop = False
        attempted_no_mtp_add = False
        repair_note = ""
        no_mtp_note = ""
        optional_failed = False
        while True:
            try:
                # encoding/errors pinned so non-UTF8 output never crashes decoding.
                if print_output:
                    result = subprocess.run(command, shell=False, check=True,
                                          encoding="utf-8", errors="replace",
                                          env=_converter_child_env(_gguf_py_pin),
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    print(result.stdout)
                else:
                    # Capture so a failure surfaces the real traceback.
                    subprocess.run(command, shell=False, check=True,
                                   encoding="utf-8", errors="replace",
                                   env=_converter_child_env(_gguf_py_pin),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                break
            except subprocess.CalledProcessError as e:
                # Joined, never concatenated: an unterminated stderr line glued
                # to stdout's first splices two harmless fragments into a line
                # that matches a self-heal signature.
                captured_streams = []
                for stream in (getattr(e, "stderr", None), getattr(e, "stdout", None)):
                    if stream:
                        captured_streams.append(
                            stream if isinstance(stream, str) else stream.decode("utf-8", errors="replace"))
                captured = "\n".join(captured_streams)

                # Self-heal: reinstall the converter deps (command[0] = its
                # interpreter) and retry once instead of failing.
                # `_auto_install_enabled` gates this too: the repair runs pip with
                # --upgrade --force-reinstall, so it mutates the environment just as much
                # as the installer does. An existing checkout returns out of
                # install_llama_cpp before its gate, so this is the one remaining way a
                # refusal could still reach pip.
                if (not attempted_repair and _auto_install_enabled()
                        and _looks_like_converter_dep_error(captured)):
                    attempted_repair = True
                    try:
                        repair = _reinstall_converter_deps(command[0], print_output = print_output)
                        if repair.returncode == 0:
                            continue
                        repair_note = f"\n--- dependency reinstall failed ---\n{(repair.stdout or '').strip()}"
                    except Exception as repair_error:
                        repair_note = f"\n--- dependency reinstall failed ---\n{repair_error}"

                # `--no-mtp` is architecture-gated, so a config key alone cannot
                # prove it is accepted. Retry once without it: the pre-existing
                # behaviour, and correct for an arch with no MTP block to strip.
                if not attempted_no_mtp_drop and _converter_rejected_no_mtp(captured):
                    retry = _drop_no_mtp(command)
                    if retry is not None:
                        attempted_no_mtp_drop = True
                        command = retry
                        continue

                # Text runs only: `--no-mtp` never belonged on the projector
                # command, and a projector failure is degraded, not repaired.
                if not attempted_no_mtp_add and required and _converter_needs_no_mtp(captured):
                    retry = _add_no_mtp(command)
                    if retry is not None and _checkpoint_has_mtp_tensors(input_folder, _num_layers):
                        # `--no-mtp` would discard the `mtp.*` tensors that are
                        # here and call the lossy export a success.
                        no_mtp_note = (
                            "\n--- unsloth ---\nThe converter could not infer this "
                            "checkpoint's multi-token prediction (MTP) head, but the "
                            "checkpoint does contain MTP tensors, so retrying with "
                            "--no-mtp would silently drop them. Convert with an "
                            "explicit `mtp_num_hidden_layers` in config.json, or "
                            "remove the MTP tensors, and try again."
                        )
                        attempted_no_mtp_add = True
                    elif retry is not None:
                        attempted_no_mtp_add = True
                        # The one self-heal that changes what the GGUF contains.
                        print(
                            "Unsloth: The GGUF converter recognised no multi-token "
                            "prediction (MTP) head in this checkpoint. Retrying with "
                            "--no-mtp; if it succeeds the GGUF will have no MTP head."
                        )
                        # The failed attempt leaves a truncated --outfile and,
                        # when splitting, shards beside it. Callers scan the
                        # output directory for *.gguf and would ship them.
                        _remove_gguf_outputs(output_file)
                        # Else a retry that fails for its own reason discards
                        # the assertion that caused it.
                        no_mtp_note = (
                            f"\n--- first attempt, before retrying with --no-mtp ---\n"
                            f"{captured.strip()}"
                        )
                        command = retry
                        continue

                # OOM-killed: retry once spooling to disk, the one resource
                # these machines have. Only for a kill, so a converter that
                # failed on its own is not quietly run twice.
                if not attempted_temp_file and _converter_was_oom_killed(e):
                    retry = _retry_with_temp_file(command)
                    if retry is not None:
                        attempted_temp_file = True
                        print(
                            "Unsloth: The GGUF converter ran out of host RAM "
                            "and was killed. Retrying with --use-temp-file, "
                            "which spools tensors to disk instead."
                        )
                        # The retry drops --split-max-size, so the killed
                        # run's shards would linger beside the good file and
                        # be uploaded with it.
                        _remove_gguf_outputs(output_file)
                        command = retry
                        continue

                if print_output and getattr(e, 'stdout', None):
                    print(e.stdout)
                cmd = " ".join(str(x) for x in command)
                # Surface the converter output, else the failure is just "exit
                # status 1" with the real traceback discarded.
                details = ""
                for label, stream in (("stderr", getattr(e, "stderr", None)),
                                      ("stdout", getattr(e, "stdout", None))):
                    if not stream: continue
                    text = stream if isinstance(stream, str) else stream.decode("utf-8", errors="replace")
                    text = text.strip()
                    if text: details += f"\n--- converter {label} ---\n{text}"
                if not required:
                    # Degrade like the "Converting as text-only model" path,
                    # but for listed archs whose projector conversion fails.
                    reason = ""
                    for line in reversed((details or "").splitlines()):
                        line = line.strip()
                        if line and ("Error" in line or "Exception" in line):
                            reason = line
                            break
                    # The converter truncates its --outfile at header time,
                    # so the failed run leaves a partial projector that callers
                    # would upload as if it were valid.
                    _remove_gguf_outputs(output_file)
                    is_vlm = False
                    optional_failed = True
                    print(
                        f"Unsloth: Could not convert the {description} to GGUF "
                        f"({reason or e}). The text model was converted "
                        f"successfully and is usable; only multimodal "
                        f"(image/audio) input is unavailable in this GGUF."
                    )
                    break
                skew_note = ""
                # The requirements are what tell a gguf AttributeError apart from
                # an ordinary model-side one; see _looks_like_gguf_skew.
                if _looks_like_gguf_skew(
                    captured,
                    _converter_gguf_requirements(converter_location, _architecture)[0],
                ):
                    skew_note = _gguf_skew_diagnosis(
                        converter_location, command[0], _gguf_report, _architecture,
                    )
                raise RuntimeError(f"Unsloth: Failed to convert {description} to GGUF with command `{cmd}`: {e}{details}{repair_note}{no_mtp_note}{skew_note}")

        # Its partial output was just removed, so validation would fail for nothing.
        if optional_failed:
            continue

        # Simple validation using native Python - check for main file or sharded files
        if os.path.exists(output_file):
            all_output_files.append(output_file)
            found_files = [output_file]
        else:
            # llama.cpp uses SHARD_NAME_FORMAT = "{:s}-{:05d}-of-{:05d}.gguf"
            basename_without_gguf = os.path.splitext(output_file)[0]
            shard_pattern = re.compile(
                re.escape(os.path.basename(basename_without_gguf)) + r'-(\d{5})-of-(\d{5})\.gguf$'
            )
            parent_dir = os.path.dirname(output_file) or '.'
            shard_files = sorted(
                os.path.join(parent_dir, f)
                for f in os.listdir(parent_dir)
                if shard_pattern.search(f)
            )

            if not shard_files:
                raise RuntimeError(
                    f"Unsloth: Failed to convert {description} - "
                    f"output file {output_file} not created"
                )

            # Validate shard completeness
            shard_numbers = []
            for f in shard_files:
                m = shard_pattern.search(os.path.basename(f))
                shard_numbers.append((int(m.group(1)), int(m.group(2))))

            expected_total = shard_numbers[0][1]
            if not all(n[1] == expected_total for n in shard_numbers):
                raise RuntimeError(f"Shards have mismatched total counts in {description}")

            actual = sorted(n[0] for n in shard_numbers)
            if actual != list(range(1, expected_total + 1)):
                missing = set(range(1, expected_total + 1)) - set(actual)
                raise RuntimeError(f"Missing shards for {description}: {missing}")

            print(f"Found {len(shard_files)} sharded output files for {description}")
            all_output_files.extend(shard_files)
            found_files = shard_files
        verify_groups.append((list(found_files), description, required))
        pass

        if print_output:
            file_size_bytes = sum(os.path.getsize(f) for f in found_files)
            if file_size_bytes >= 1024**3:  # GB
                size_str = f"{file_size_bytes / (1024**3):.1f}G"
            elif file_size_bytes >= 1024**2:  # MB
                size_str = f"{file_size_bytes / (1024**2):.1f}M"
            else:
                size_str = f"{file_size_bytes / 1024:.1f}K"
            if len(found_files) == 1:
                print(f"Unsloth: Successfully saved {description} GGUF to: {found_files[0]} (size: {size_str})")
            else:
                print(f"Unsloth: Successfully saved {description} GGUF as {len(found_files)} shards (total size: {size_str})")

    # The converter never re-reads what it wrote, so it reports success on a file
    # no llama.cpp build can load (unsloth#6056, unsloth#8360, unsloth#8513).
    # `_gguf_py_pin` is None whenever nothing needed pinning, so the read back derives
    # the writer's tree from the probe report instead.
    _readback_tree = _gguf_py_pin or _gguf_readback_tree(_gguf_report)
    for _files, _description, _required in verify_groups:
        if not _verify_run_outputs(
            _files, _description, _required, quantization_type,
            print_output = print_output, gguf_py_dir = _readback_tree,
        ):
            all_output_files = [f for f in all_output_files if f not in set(_files)]
            is_vlm = False

    return all_output_files, is_vlm
pass


# Quants tensor_requires_imatrix rejects for any transformer, measured with
# `llama-quantize --dry-run`, not read off the ftype defaults. iq3_xs is here despite defaulting
# to IQ3_S: the attention Q/K overrides promote to IQ3_XXS unconditionally, failing on blk.0.
# iq3_s, iq3_m, iq4_nl, iq4_xs, q2_k, q3_k_s and tq*_0 all quantize fine without one.
IMATRIX_REQUIRED_QUANTS = frozenset((
    "iq1_s", "iq1_m", "iq1_xs", "iq1_xxs", "iq1_xxxs",
    "iq2_xxs", "iq2_xs", "iq2_s", "iq2_m",
    "iq3_xxs", "iq3_xs",
    "q2_k_s",
))

# All three are in live use across unsloth/*-GGUF, and --imatrix reads any of them. .gguf_file is
# a GGUF imatrix named so the Hub does not list it as a model; plain .gguf skips that guard.
IMATRIX_UPSTREAM_NAMES = (
    "imatrix_unsloth.dat", "imatrix_unsloth.gguf_file", "imatrix_unsloth.gguf",
)


def quant_requires_imatrix(quant_type):
    return str(quant_type).strip().lower() in IMATRIX_REQUIRED_QUANTS


def _materialize_imatrix(path, dest_dir):
    """Copy into dest_dir (never mutate the HF cache), renaming *.gguf_file -> *.gguf."""
    os.makedirs(dest_dir, exist_ok = True)
    base = os.path.basename(path)
    if base.endswith(".gguf_file"):
        base = base[: -len(".gguf_file")] + ".gguf"
    local = os.path.join(dest_dir, base)
    # dest_dir can be the file's own directory: the helper is public and Studio calls it directly.
    if os.path.exists(local) and os.path.samefile(path, local):
        return local
    shutil.copyfile(path, local)
    return local


def resolve_imatrix_file(imatrix_file, dest_dir, repo_candidates = (), token = None):
    """Resolve imatrix_file to a local path, or None.

    None/False -> None; a path -> a copy in dest_dir (*.gguf_file renamed to .gguf); True -> the
    first upstream imatrix across repo_candidates, or a RuntimeError naming what was searched.
    """
    if imatrix_file is None or imatrix_file is False:
        return None

    if isinstance(imatrix_file, (str, os.PathLike)):
        path = os.path.expanduser(os.fspath(imatrix_file))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Unsloth: imatrix_file '{path}' does not exist.")
        # Copy even when it is already local: an imatrix under save_directory would be uploaded
        # and reported as an exported model, since both paths glob save_directory/*.gguf.
        return _materialize_imatrix(path, dest_dir)

    if imatrix_file is not True:
        raise TypeError(
            "Unsloth: imatrix_file must be None, a path string, or True "
            f"(got {type(imatrix_file).__name__})."
        )

    from huggingface_hub import HfApi, hf_hub_download

    repos = []
    for repo in repo_candidates:
        if repo and repo not in repos: repos.append(repo)
    api = HfApi(token = token)
    lookup_error = None
    for repo in repos:
        try:
            files = set(api.list_repo_files(repo))
        except Exception as error:
            # Keep the first cause, reported only if no candidate hits: otherwise a bad token
            # or an outage reads as "no imatrix exists".
            if lookup_error is None: lookup_error = f"{repo}: {type(error).__name__}: {error}"
            continue
        for name in IMATRIX_UPSTREAM_NAMES:
            if name not in files: continue
            downloaded = hf_hub_download(repo_id = repo, filename = name, token = token)
            local = _materialize_imatrix(downloaded, dest_dir)
            print(f"Unsloth: Using imatrix '{name}' from '{repo}' -> '{local}'")
            return local
    raise RuntimeError(
        "Unsloth: imatrix_file=True but no upstream Unsloth imatrix was found.\n"
        f"  Searched repos: {repos or '(none derived from the base model)'}\n"
        f"  Searched files: {list(IMATRIX_UPSTREAM_NAMES)}\n"
        + (f"  First lookup failure: {lookup_error}\n" if lookup_error else "")
        + "Pass imatrix_file='/path/to/imatrix.(dat|gguf)' to use your own."
    )


def quantize_gguf(
    input_gguf,
    output_gguf,
    quant_type,
    quantizer_location = os.path.join(LLAMA_CPP_DEFAULT_DIR, "llama-quantize"),
    n_threads = None,
    print_output = True,
    imatrix = None,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Use llama-quantize for fast quantization of GGUF files.

    # quant_type reaches the shell command below straight from the user facing
    # quantization_method, and every real value is a bare token, so require one.
    if not isinstance(quant_type, str) or \
        re.fullmatch(r"[A-Za-z0-9_.\-]+", quant_type.strip()) is None:
        raise ValueError(
            f"Unsloth: Invalid quantization type `{quant_type}`. Quantization types are "
            f"single tokens like `q4_k_m`, `q8_0`, `iq4_xs`, `bf16`."
        )
    quant_type = quant_type.strip()

    # Fix default path on Windows: binaries are in build/bin/Release/
    default_quantizer = os.path.join(LLAMA_CPP_DEFAULT_DIR, "llama-quantize")
    # H3: Use normpath for reliable path comparison on Windows (/ vs \)
    if IS_WINDOWS and os.path.normpath(quantizer_location) == os.path.normpath(default_quantizer):
        quantizer_location = os.path.join(
            LLAMA_CPP_DEFAULT_DIR, "build", "bin", "Release", "llama-quantize.exe"
        )

    if n_threads is None:
        n_threads = psutil.cpu_count()
        if n_threads is None:
            n_threads = 1
        n_threads *= 2
    n_threads = int(n_threads)

    def _quote(s):
        """Quote a path for shell usage (the command runs under shell=True)."""
        s = str(s)
        if IS_WINDOWS:
            # cmd.exe: always wrap so spaces and metachars (& | ^) stay literal.
            return f'"{s}"'
        import shlex
        return shlex.quote(s)

    # Q2_K_L is an Unsloth preset (q2_k base + selective upcasts), not a native
    # llama.cpp ftype. Recipe: token_embd->Q4_K, output->Q6_K, every
    # ffn_down/ffn_down_exps->Q3_K. llama-quantize matches --tensor-type via
    # regex_search first-match-wins, so chain the more-specific MoE pattern
    # first; the leading `\.` anchors on the GGUF path separator so the
    # override doesn't leak into other tensors containing "ffn_down".
    _display_quant_type = quant_type
    _extra_flags = ""
    _is_q2_k_l_preset = str(quant_type).strip().lower() == "q2_k_l"
    if _is_q2_k_l_preset:
        _extra_flags = (
            '--tensor-type "\\.ffn_down_exps=Q3_K" '
            '--tensor-type "\\.ffn_down=Q3_K" '
            '--output-tensor-type Q6_K '
            '--token-embedding-type Q4_K '
        )
        quant_type = "q2_k"

    # An imatrix unlocks the IQ low-bit quants; prepend it (before positional args) and quote.
    if imatrix is not None and str(imatrix).strip() != "":
        # Normalize to a clean string first: os.path.exists(True) would treat a bool as a file
        # descriptor, and a space-padded path must be trimmed before the existence check / quoting.
        imatrix = str(imatrix).strip()
        if not os.path.exists(imatrix):
            raise FileNotFoundError(f"Unsloth: imatrix file `{imatrix}` does not exist.")
        _extra_flags = f"--imatrix {_quote(imatrix)} " + _extra_flags

    command = (
        f"{_quote(quantizer_location)} {_extra_flags}"
        # Validated above as a bare token, so quoting would only add a pair of
        # quotes that cmd.exe did not see in previous releases.
        f"{_quote(input_gguf)} {_quote(output_gguf)} {quant_type} {n_threads}"
    )

    if print_output:
        print(f"Unsloth: Quantizing to {_display_quant_type}...")
        if _is_q2_k_l_preset:
            print(
                "Unsloth: Expanding Q2_K_L preset "
                "(q2_k base, .ffn_down_exps=Q3_K + .ffn_down=Q3_K, "
                "output=Q6_K, token_embd=Q4_K)."
            )

    try:
        if print_output:
            result = subprocess.run(command, shell=True, check=True,
                                  encoding="utf-8", errors="replace",
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            print(result.stdout)
        else:
            # Capture so llama-quantize's output can be surfaced on failure.
            subprocess.run(command, shell=True, check=True,
                           encoding="utf-8", errors="replace",
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    except subprocess.CalledProcessError as e:
        if print_output and hasattr(e, 'stdout') and e.stdout:
            print(e.stdout)
        details = ""
        for label, stream in (("stderr", getattr(e, "stderr", None)),
                              ("stdout", getattr(e, "stdout", None))):
            if not stream: continue
            text = stream if isinstance(stream, str) else stream.decode("utf-8", errors="replace")
            text = text.strip()
            if text: details += f"\n--- llama-quantize {label} ---\n{text}"
        raise RuntimeError(f"Failed to quantize {input_gguf} to {_display_quant_type}: {e}{details}")

    # Verify output exists and get size using pathlib
    output_path = Path(output_gguf)
    if not output_path.exists():
        raise RuntimeError(f"Quantization failed - output file {output_gguf} not created")

    if print_output:
        file_size_bytes = output_path.stat().st_size
        file_size_gb = file_size_bytes / (1024**3)
        print(f"Unsloth: Successfully quantized to {output_gguf} (size: {file_size_gb:.2f}GB)")
    return output_gguf
pass


# Post-conversion GGUF verification. The converter does not check its own
# output, so a GGUF missing a key llama.cpp reads unconditionally is published
# and loads nowhere (unsloth#8360, unsloth#8513: MiniMax M3 quants without
# `{arch}.attention.indexer.head_count`). Key strings below come from
# `src/llama-arch.cpp`'s LLM_KV_NAMES, not from a guess.

# `general.architecture` names the arch; llama.cpp cannot choose a model
# loader without it (src/llama-arch.cpp maps it to LLM_ARCH_*).
GGUF_ARCHITECTURE_KEY = "general.architecture"

# Read by `llama_model_base::load_hparams` (src/llama-model.cpp) with no
# `required = false` argument, so a missing one throws before any tensor is
# touched. `{arch}.attention.head_count` and `{arch}.feed_forward_length` are
# deliberately NOT here: the same function reads those through
# `get_key_or_arr(..., false)`, i.e. they are optional.
GGUF_UNIVERSAL_REQUIRED_KEYS = (
    "{arch}.context_length",    # ml.get_key(LLM_KV_CONTEXT_LENGTH,   hparams.n_ctx_train)
    "{arch}.embedding_length",  # ml.get_key(LLM_KV_EMBEDDING_LENGTH, hparams.n_embd)
    "{arch}.block_count",       # ml.get_key(LLM_KV_BLOCK_COUNT,      hparams.n_layer_all)
)

# A tensor namespace implies the metadata namespace describing it:
# (tensor marker, required keys, exempt architectures, why).
# A key qualifies only if every architecture carrying the namespace reads it
# with no trailing `false` in `src/models/*.cpp`; ones reading it optionally go
# in the exempt set, anything weaker in GGUF_CONDITIONAL_ADVISORY_KEYS. A hit
# here refuses to publish, and rejecting a good GGUF is worse than no gate.
# Verified against llama.cpp b49650adb31f2e49a0d76113aeb1792134fd8413.
GGUF_CONDITIONAL_REQUIRED_KEYS = (
    (
        # `blk.N.indexer.*` and `blk.N.indexer_compressor_*`
        # (src/llama-arch.cpp LLM_TENSOR_INDEXER_*).
        ".indexer",
        (
            "{arch}.attention.indexer.head_count",
            "{arch}.attention.indexer.key_length",
            "{arch}.attention.indexer.top_k",
        ),
        # src/models/hy-v4.cpp:46-48 reads all three with a trailing `false`
        # and creates the indexer tensors only when top_k came out non-zero, so
        # a hy_v4 file without them loads (with the indexer off) rather than
        # failing. The arch string is src/llama-arch.cpp:126.
        frozenset(("hy_v4",)),
        "DeepSeek sparse attention indexer tensors. Seven architectures define "
        "them and six read these three keys as required: deepseek32.cpp, "
        "deepseek4.cpp, dots3note.cpp, glm-dsa.cpp, minimax-m3.cpp and "
        "qwen4exp.cpp. The seventh, hy-v4.cpp, reads them "
        "optionally and is exempt. The remaining indexer keys (block_size, "
        "local_blocks, kpool, types) are read by only some of the seven, so "
        "they are not required here.",
    ),
    (
        # `blk.N.ssm_*` (src/llama-arch.cpp LLM_TENSOR_SSM_*).
        ".ssm_",
        ("{arch}.ssm.conv_kernel",),
        frozenset(),
        "State space and linear attention tensors. All fifteen architectures "
        "that create them read ssm.conv_kernel with no `false`, checked by "
        "grepping LLM_KV_SSM_CONV_KERNEL across src/models/. inner_size, "
        "state_size and time_step_rank are required by most but not all, so "
        "they are not here.",
    ),
)

# Same shape, but these are WARNINGS. llama.cpp loads the file, so refusing to
# publish it could turn an export that works today into a hard failure; the
# result is still likely to be wrong, so it is worth saying out loud.
GGUF_CONDITIONAL_ADVISORY_KEYS = (
    (
        # `blk.N.ffn_{gate,up,down}_exps` (LLM_TENSOR_FFN_*_EXPS).
        "_exps",
        ("{arch}.expert_count", "{arch}.expert_used_count"),
        frozenset(),
        "Mixture of experts tensors. `load_hparams` reads expert_count with a "
        "trailing `false` and defaults it to 0, so without it no architecture "
        "creates an expert tensor and llama-model-loader.cpp logs 'model has "
        "unused tensor ... -- ignoring' for every one of them: the model loads "
        "and is silently wrong. Advisory rather than fatal because llama.cpp "
        "does load it, and because a converter can legitimately reach this "
        "state: conversion/hunyuan.py pops `num_experts` before calling "
        "super().set_gguf_parameters() and restores it afterwards, so the key "
        "is never written while modify_tensors still emits the `_exps` "
        "tensors.",
    ),
)

# `load_hparams` returns before reading any of the above for these, so the gate
# must not fire on a projector. "clip" is the dummy architecture every mmproj
# file carries (gguf-py/gguf/constants.py maps MODEL_ARCH.MMPROJ to "clip",
# src/llama-arch.cpp maps LLM_ARCH_CLIP to "clip"), and src/llama-model.cpp
# reads `if (hparams.vocab_only || ml.get_arch() == LLM_ARCH_CLIP) return;`.
GGUF_METADATA_EXEMPT_ARCHITECTURES = frozenset(("clip",))

_GGUF_SHARD_PATTERN = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$")

# Default number of tensors sampled by the data checks. The first and last are
# always added on top, so the floor is 2.
GGUF_VERIFY_SAMPLE_DEFAULT = 8


def _gguf_verify_enabled():
    """Whether post-conversion verification runs. Read at the call, not at
    import, so setting it after `import unsloth` works."""
    return os.environ.get("UNSLOTH_GGUF_VERIFY", "1").strip().upper() \
        not in ("0", "OFF", "FALSE", "NO")


def _gguf_sample_size():
    """`UNSLOTH_GGUF_VERIFY_TENSORS`, clamped to >= 0. Junk reads as the default."""
    raw = os.environ.get("UNSLOTH_GGUF_VERIFY_TENSORS", "")
    if not raw.strip():
        return GGUF_VERIFY_SAMPLE_DEFAULT
    try:
        return max(0, int(raw.strip()))
    except ValueError:
        logger.warning(
            f"Unsloth: UNSLOTH_GGUF_VERIFY_TENSORS='{raw}' is not an integer; "
            f"using {GGUF_VERIFY_SAMPLE_DEFAULT}."
        )
        return GGUF_VERIFY_SAMPLE_DEFAULT


def _gguf_shard_siblings(path):
    """Every shard of a split GGUF, in order, given any one of them.

    Only shard 1 carries the model's KV metadata, so a gate handed shard 3
    would find no `general.architecture` and reject a file that is in fact
    fine. Returns `[path]` for a single file, and for a shard set the full
    list, so the tensor names are the union across shards rather than
    whichever slice happened to be passed in.
    """
    match = _GGUF_SHARD_PATTERN.match(os.path.basename(path))
    if match is None:
        return [path]
    directory = os.path.dirname(path) or "."
    stem, total = match.group("stem"), int(match.group("total"))
    shards = []
    for index in range(1, total + 1):
        candidate = os.path.join(directory, f"{stem}-{index:05d}-of-{total:05d}.gguf")
        if os.path.isfile(candidate):
            shards.append(candidate)
    return shards or [path]


def _gguf_missing_shards(path):
    """Basenames the split set declares but does not have on disk, in order.

    `_gguf_shard_siblings` drops an absent shard silently, which is right for building
    the union of tensor names but means an incomplete set is indistinguishable from a
    complete smaller one: the count is only in the filename. Without this, shard 1 being
    present was enough for every check below to run on whatever survived and pass, so a
    split model missing its middle was reported ready to publish or quantize.
    """
    match = _GGUF_SHARD_PATTERN.match(os.path.basename(path))
    if match is None:
        return []
    directory = os.path.dirname(path) or "."
    stem, total = match.group("stem"), int(match.group("total"))
    missing = []
    for index in range(1, total + 1):
        name = f"{stem}-{index:05d}-of-{total:05d}.gguf"
        if not os.path.isfile(os.path.join(directory, name)):
            missing.append(name)
    return missing


def _gguf_field_text(reader, key):
    """A string KV value, or None when absent or not readable as one."""
    field = reader.fields.get(key)
    if field is None:
        return None
    try:
        value = field.contents()
    except Exception:
        return None
    if isinstance(value, (bytes, bytearray)):
        return str(bytes(value), encoding = "utf-8", errors = "replace")
    return None if value is None else str(value)


def _gguf_field_int(reader, key):
    """An integer KV value, or None when absent or not readable as one."""
    field = reader.fields.get(key)
    if field is None:
        return None
    try:
        value = field.contents()
    except Exception:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return int(value)
    except Exception:
        return None


# A block index in a parameter name: `model.layers.31.mlp...`, `transformer.h.5...`.
_HF_BLOCK_INDEX_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks|block)\.(\d+)(?:\.|$)")


def _model_block_count(shapes):
    """Blocks inferred from parameter names, or None. The fallback for a missing KV."""
    highest = -1
    for name in shapes:
        match = _HF_BLOCK_INDEX_RE.search(name)
        if match is None:
            continue
        try:
            index = int(match.group(1))
        except ValueError:
            continue
        if index > highest:
            highest = index
    return highest + 1 if highest >= 0 else None


def _open_gguf_reader(path):
    """A plain `GGUFReader`, imported at the call site.

    There used to be a `Partial_GGUFReader` here, built by running
    `inspect.getsource(GGUFReader.__init__)` through `exec` with
    `self._build_tensors(offs, tensors_fields` rewritten to
    `...tensors_fields[-1:]`, so only the final tensor was ever built. It was
    removed because it bought nothing and hid everything: measured on a 542 MB
    export the plain reader takes 6.69 s and 736 MB peak RSS against the
    rewritten one's 7.48 s and the same 736 MB, because all of the cost is
    `_build_fields` parsing a 262k entry vocabulary while `_build_tensors`
    only creates lazy memmap views.

    Neither the time nor the memory is free, and both are the vocabulary's
    rather than the model's. Reading a file back costs the parent 0.8 GB of
    anonymous heap on a 262k vocabulary and 1.5 GB on a vocab-only file of
    15.8 MB, none of it reclaimable, and that spike lands in the parent, which
    has no OOM retry, unlike the converter child. The tensor sample itself is
    file-backed and reclaimable: the windowed pass touches about 16 KB per
    tensor, flat, but a tensor in the deep sample is read whole up to
    GGUF_VERIFY_DEEP_MAX_ELEMENTS, so the sample can fault in a gigabyte, and a
    tensor ABOVE that cap is skipped entirely. The effect is non monotonic: a
    model whose tensors all cross the cap gets no deep check at all. The rewrite also raised
    `RuntimeError: Reader has no self._build_tensors(offs, tensors_fields`
    whenever gguf-py refactored that one line.
    """
    from gguf.gguf_reader import GGUFReader  # type: ignore
    return GGUFReader(path, "r")


def _gguf_open_shards(gguf_file):
    """`[(path, reader or None), ...]` for a GGUF and, if it is split, its siblings.

    Built once and threaded through the checks below. `GGUFReader.__init__`
    parses every KV entry eagerly, which on a 262k entry vocabulary costs 6.7 s
    and about 1.5 GB of transient numpy views, so opening the file once per
    check made verification cost three times what it needs to. A reader is None
    when that shard could not be read; the caller decides whether that is a
    problem or a warning.
    """
    opened = []
    for path in _gguf_shard_siblings(gguf_file):
        try:
            opened.append((path, _open_gguf_reader(path)))
        except Exception as error:
            logger.debug(f"Unsloth: could not read {path} ({type(error).__name__}: {error})")
            opened.append((path, None))
    return opened


def _gguf_metadata_scan(gguf_file, readers, conditional, include_universal):
    """Shared body of gguf_metadata_problems and gguf_metadata_warnings.

    Returns a list of strings. `conditional` is one of the two tables above;
    `include_universal` adds the keys llama.cpp reads for every architecture,
    which belong to the fatal pass only.
    """
    if readers is None:
        readers = _gguf_open_shards(gguf_file)
    # A file this unreadable, or a shard set this incomplete, is already a fatal
    # problem, so only the fatal pass reports it. Saying it twice, once as a
    # problem and once as a warning, reads like two separate faults.
    report_structural = include_universal
    reader = readers[0][1]
    if reader is None:
        if not report_structural:
            return []
        return [f"`{os.path.basename(readers[0][0])}` could not be read as a GGUF file"]

    # Only shard 1 carries the KV metadata, so if it is not on disk every key
    # below reads as missing. Say which file is absent instead, or the report
    # sends the reader looking for metadata that was never supposed to be here.
    first = _GGUF_SHARD_PATTERN.match(os.path.basename(readers[0][0]))
    if first is not None and int(first.group("index")) != 1:
        if not report_structural:
            return []
        total = int(first.group("total"))
        return [
            f"shard 1 of {total} (`{first.group('stem')}-00001-of-{total:05d}.gguf`) "
            f"is not present, and it is the only shard that carries the model's "
            f"metadata"
        ]

    # Shard 1 is here, so the metadata reads fine and every check below would run happily
    # on the shards that remain. A later shard being absent is just as unloadable, and the
    # only record of how many there should be is the `-of-NNNNN` in the name.
    absent = _gguf_missing_shards(readers[0][0])
    if absent:
        if not report_structural:
            return []
        total = int(first.group("total"))
        listed = ", ".join(f"`{name}`" for name in absent)
        return [
            f"{len(absent)} of the {total} shards this split model declares are not "
            f"present ({listed}), so it is incomplete and llama.cpp cannot load it"
        ]

    architecture = _gguf_field_text(reader, GGUF_ARCHITECTURE_KEY)
    if not architecture:
        if not report_structural:
            return []
        return [
            f"`{GGUF_ARCHITECTURE_KEY}` is missing, so llama.cpp cannot choose "
            f"a model loader for this file"
        ]
    tensor_names = []
    unreadable_shards = []
    for path, shard_reader in readers:
        if shard_reader is None:
            unreadable_shards.append(os.path.basename(path))
            logger.warning(
                f"Unsloth: could not read shard {path} while checking GGUF "
                f"metadata; its tensors are not part of the check."
            )
            continue
        tensor_names.extend(tensor.name for tensor in shard_reader.tensors)

    # BEFORE the architecture exemption, not after it. A shard that cannot be read is a
    # structural fault, and the exemption is about which metadata KEYS a loader reads: a
    # projector whose second shard is corrupt is as unloadable as any other model with a
    # corrupt shard, and returning [] on the exemption first told a caller of the exported
    # API that an unloadable projector was fine. Same for the missing-shard and unreadable
    # first-shard answers above, which already run before it.
    if unreadable_shards and report_structural:
        # A later shard that cannot be read is exactly as unloadable as a missing one, and
        # this function's documented empty list is what a caller uses to ACCEPT a downloaded
        # split model. Logging it and returning [] told that caller the shard set was fine.
        # Fatal pass only, like every other structural answer here: the advisory pass would
        # otherwise report the same fault twice.
        listed = ", ".join(f"`{name}`" for name in unreadable_shards)
        return [
            f"{len(unreadable_shards)} of this split model's shards could not be read "
            f"({listed}), so llama.cpp cannot load it"
        ]

    if architecture in GGUF_METADATA_EXEMPT_ARCHITECTURES:
        # Multimodal projector. llama.cpp returns from load_hparams before
        # reading any of the keys below, so requiring them would reject every
        # mmproj file Unsloth produces.
        return []

    if not any(name.startswith("blk.") for name in tensor_names):
        # No transformer blocks: a vocabulary-only or otherwise non-model GGUF.
        # llama.cpp skips the hparams entirely for those (the `vocab_only`
        # half of the early return in load_hparams).
        return []

    present = reader.fields.keys()
    problems = []
    if include_universal:
        for template in GGUF_UNIVERSAL_REQUIRED_KEYS:
            key = template.format(arch = architecture)
            if key not in present:
                problems.append(f"`{key}` is missing (llama.cpp reads it for every architecture)")

    for marker, templates, exempt, reason in conditional:
        if architecture in exempt:
            # This architecture's loader reads the keys optionally, so the file
            # is valid without them and the namespace implies nothing.
            continue
        example = next((name for name in tensor_names if marker in name), None)
        if example is None:
            continue
        for template in templates:
            key = template.format(arch = architecture)
            if key not in present:
                problems.append(
                    f"`{key}` is missing even though the file contains "
                    f"`{example}`. {reason}"
                )
    return problems


def gguf_metadata_problems(gguf_file, readers = None):
    """Architecture-required metadata missing from a GGUF, as a list of strings.

    Empty means the file carries everything llama.cpp reads unconditionally for
    the architecture it declares and for the tensor namespaces it contains. No
    torch model, tokenizer or source checkpoint needed, so this also runs on a
    GGUF downloaded from anywhere.

    Every entry here is something llama.cpp refuses to load without, which is
    what makes it safe to refuse to publish the file. Keys that only make the
    model silently wrong are reported by `gguf_metadata_warnings` instead.

    Split exports are resolved to their whole shard set first, so passing any
    shard is the same as passing the first.
    """
    return _gguf_metadata_scan(
        gguf_file, readers, GGUF_CONDITIONAL_REQUIRED_KEYS, True,
    )


def gguf_metadata_warnings(gguf_file, readers = None):
    """Metadata a GGUF should carry but that llama.cpp will load without.

    Same shape as `gguf_metadata_problems`, and deliberately separate: these are
    logged rather than raised, so a file llama.cpp accepts is never refused.
    """
    return _gguf_metadata_scan(
        gguf_file, readers, GGUF_CONDITIONAL_ADVISORY_KEYS, False,
    )


def _gguf_sample_indices(count, sample_size, seed_material):
    """`sample_size` indices into `range(count)`, plus the first and last.

    Deterministic in the file it describes, so a rejection reproduces: the seed
    comes from `seed_material` (the file's own name and tensor count) rather
    than from the clock or `PYTHONHASHSEED`.
    """
    if count <= 0:
        return []
    if count <= sample_size + 2:
        return list(range(count))
    import random
    digest = hashlib.sha256(str(seed_material).encode("utf-8")).hexdigest()
    chosen = {0, count - 1}
    chosen.update(random.Random(int(digest[:16], 16)).sample(range(count), sample_size))
    return sorted(chosen)


# Elements read from each end of every tensor by the cheap pass. 4096 floats
# is 8 or 16 KB, so scanning every tensor in a 4000 tensor export costs tens of
# megabytes of reads no matter how large the export is.
GGUF_VERIFY_WINDOW_DEFAULT = 4096

# A tensor above this many elements is windowed even when the deterministic
# sample picks it, so verification never reads gigabytes. 64M elements is
# 128 MB at two bytes each; `token_embd.weight` alone is 168M elements on a
# 262k vocabulary, and it is always in the sample because the first and last
# tensors always are.
GGUF_VERIFY_DEEP_MAX_ELEMENTS = 64 * 1024 * 1024

# The sublayer output projections, the only weights inside a block that a
# published method deliberately sets to exactly zero. LLaMA Pro block expansion
# (arXiv 2401.02415) zero-initialises o_proj and down_proj so that a copied block
# computes the identity, and llama.cpp loads the result, so an all-zero one is not
# evidence of a bad conversion. gguf-py's TensorNameMap spells them like this.
GGUF_IDENTITY_INIT_SUFFIXES = (".attn_output.weight", ".ffn_down.weight")


def _gguf_window_size():
    """`UNSLOTH_GGUF_VERIFY_ELEMENTS`, clamped to >= 1. Junk reads as the default."""
    raw = os.environ.get("UNSLOTH_GGUF_VERIFY_ELEMENTS", "")
    if not raw.strip():
        return GGUF_VERIFY_WINDOW_DEFAULT
    try:
        return max(1, int(raw.strip()))
    except ValueError:
        logger.warning(
            f"Unsloth: UNSLOTH_GGUF_VERIFY_ELEMENTS='{raw}' is not an integer; "
            f"using {GGUF_VERIFY_WINDOW_DEFAULT}."
        )
        return GGUF_VERIFY_WINDOW_DEFAULT


# GGUF float types gguf-py returns as raw bytes, with the unsigned view exposing
# their exponent bits. numpy has no bfloat16, so BF16 arrives as uint8 and a
# plain `np.issubdtype(..., np.floating)` test skipped it; F32, F64 and F16 come
# back as real float dtypes and take that fast path instead. Values from
# gguf-py/gguf/constants.py GGMLQuantizationType.
_GGUF_BYTEWISE_FLOAT_TYPES = {
    30: ("uint16", 0x7F80),  # BF16: 8 exponent bits
}


def _gguf_float_view(tensor):
    """`(values, exponent_mask)` for a tensor whose bits can be examined, else
    `(None, None)`.

    `exponent_mask` is None when `values` is a real float dtype and numpy can
    answer `isfinite` directly; otherwise it is the mask that isolates the
    exponent field of the integer view, and an all-ones exponent means NaN or
    Inf.
    """
    import numpy as np
    data = np.asarray(tensor.data)
    if np.issubdtype(data.dtype, np.floating):
        return data.reshape(-1), None
    entry = _GGUF_BYTEWISE_FLOAT_TYPES.get(int(tensor.tensor_type))
    if entry is None:
        # A quantized block format. Its bytes are not values, so only the
        # all-zero test below applies, on the raw bytes.
        return None, None
    view_dtype, mask = entry
    return data.reshape(-1).view(view_dtype), mask


def _gguf_degenerate_problem(tensor, sample_deeply, window, check_zero = True):
    """Why a GGUF tensor cannot be a real weight, or None.

    Two passes. Every tensor gets the cheap one: a window from each end, which
    catches a zeroed or NaN-filled tensor with certainty because damage of that
    kind is never confined to the middle. A tensor in the deterministic deep
    sample is read whole, which also catches a non-finite value that happens to
    sit away from both ends.

    The all-zero test is gated by `check_zero`, because zero is legitimate
    outside the transformer blocks: BERT-family `token_types.weight` is all
    zeros for a single-segment model, and a projector can carry zeroed entries
    too. It works on raw bytes, so it covers quantized types as well. The NaN
    and Inf test needs to know where the exponent is, so it runs for F32, F64,
    F16 and BF16 only, and it runs on every tensor because a non-finite value is
    never legitimate anywhere.
    """
    import numpy as np
    values, exponent_mask = _gguf_float_view(tensor)
    raw = np.asarray(tensor.data).reshape(-1)
    if raw.size == 0:
        return None

    def clip(array):
        if array.size <= 2 * window:
            return array, True
        if sample_deeply and array.size <= GGUF_VERIFY_DEEP_MAX_ELEMENTS:
            return array, True
        return np.concatenate((array[:window], array[-window:])), False

    if values is not None:
        checked, _ = clip(values)
        if exponent_mask is None:
            bad = not np.isfinite(checked).all()
        else:
            bad = bool((checked & exponent_mask == exponent_mask).any())
        if bad:
            return "contains NaN or Inf"

    if not check_zero:
        return None
    checked_raw, whole = clip(raw)
    if checked_raw.any():
        return None
    if whole:
        return "is entirely zero"
    # The ends are zero. Confirm against the whole tensor before rejecting, so
    # a legitimately zero-padded weight is not reported.
    return None if raw.any() else "is entirely zero"


def gguf_tensor_problems(
    gguf_file, sample_size = None, readers = None, float_tensors_only = False,
):
    """Tensors in a GGUF that cannot be real weights, as a list of strings.

    Rejects only what no architecture produces legitimately: a weight that is
    entirely zero, or one holding NaN or Inf. Every tensor is checked through a
    bounded window, and `sample_size` of them, chosen deterministically, are
    read in full.

    It deliberately does not compare values against the source checkpoint,
    because the converters transform tensors on the way out
    (`conversion/gemma.py` adds 1 to every norm weight, the llama path permutes
    `attn_q` and `attn_k`, expert weights are concatenated) and no value
    comparison can tell a transform from damage without reproducing each
    architecture's `modify_tensors`.

    `float_tensors_only` restricts the pass to tensors that have a float view,
    which is what the export gate wants: one conversion writes both, so a
    `q4_k_m` file still carries float tensors (the projector, and often
    `token_embd` and the norms), and those are exactly the ones a value check
    means something for. The quantized tensors in that file are then left alone
    rather than having the all-zero test run over their block bytes. Off by
    default so `verify_gguf` on a user's file keeps checking every tensor it
    can, quantized blocks included.
    """
    if sample_size is None:
        sample_size = _gguf_sample_size()
    if readers is None:
        readers = _gguf_open_shards(gguf_file)
    window = _gguf_window_size()
    problems = []
    for shard, reader in readers:
        if reader is None:
            problems.append(f"`{os.path.basename(shard)}` could not be read as a GGUF file")
            continue
        tensors = reader.tensors
        deep = set(_gguf_sample_indices(
            len(tensors), sample_size, (os.path.basename(shard), len(tensors)),
        ))
        for index, tensor in enumerate(tensors):
            if float_tensors_only:
                try:
                    values, _mask = _gguf_float_view(tensor)
                except Exception:
                    values = None
                if values is None: continue
            # Zero is a legitimate weight outside `blk.` (BERT `token_types.weight`,
            # projector padding) and for a bias, so all-zero is only a defect inside
            # a block, which is where unsloth#6056's damage lives. o_proj and
            # down_proj are excluded too: LLaMA Pro block expansion (arXiv
            # 2401.02415) zero-initialises them and llama.cpp loads that file.
            # NaN and Inf stay checked everywhere.
            check_zero = (
                tensor.name.startswith("blk.")
                and not tensor.name.endswith(".bias")
                and not tensor.name.endswith(GGUF_IDENTITY_INIT_SUFFIXES)
            )
            try:
                problem = _gguf_degenerate_problem(
                    tensor, index in deep, window, check_zero,
                )
            except Exception as error:
                logger.debug(
                    f"Unsloth: could not inspect `{tensor.name}` in "
                    f"{os.path.basename(shard)} ({type(error).__name__}: {error})"
                )
                continue
            if problem is not None:
                problems.append(f"`{tensor.name}` {problem}")
    return problems


def _gguf_holds_any_float_tensor(readers):
    """Whether any tensor in this file has a float view, so a value check means something.

    The export gate asks this rather than "are they all floats", because one
    conversion writes both. A `q4_k_m` VLM still carries its projector at bf16
    or f16, and usually `token_embd` and the norms at f32, and NaN or Inf in
    those is exactly as fatal as it would be in a plain f16 export. Requiring
    every tensor to be float meant a single quantized block turned the whole
    file's value check off, which on a q4_k_m or MXFP4 export is every file
    most people publish.
    """
    for _path, reader in readers:
        if reader is None:
            continue
        for tensor in reader.tensors:
            try:
                values, _mask = _gguf_float_view(tensor)
            except Exception:
                continue
            if values is not None:
                return True
    return False


def _verify_converted_gguf(
    output_files, quantization_type = None, print_output = False, gguf_py_dir = None,
    _writer_tree_known = False,
):
    """The gate `convert_to_gguf` runs on what it just wrote.

    Metadata always; tensor sanity only for a file whose tensors are plain
    floats, which is the only case where a float view of the bytes means
    anything. Raises on a problem, because the alternative is publishing the
    file. `quantization_type` is accepted and ignored, since the file itself is
    the authority on what it holds.

    `gguf_py_dir` is the tree the converter child was pinned to. Reading the
    file back with the same `gguf` that wrote it is what keeps the checks from
    degrading into the "could not read it, so it was not verified" warning
    whenever the parent's `gguf` is the older of the two. It also decides what an
    unreadable file MEANS, which is why `_writer_tree_known` is carried into the
    recursion below rather than re-derived there.
    """
    if not _gguf_verify_enabled():
        logger.info("Unsloth: UNSLOTH_GGUF_VERIFY is off; skipping GGUF verification.")
        return
    if gguf_py_dir:
        # Nesting is safe: the context manager snapshots and restores sys.path
        # and every `gguf` module, so an outer use_local_gguf() is unaffected.
        with use_local_gguf(gguf_py_dir):
            return _verify_converted_gguf(
                output_files, quantization_type, print_output = print_output,
                _writer_tree_known = True,
            )
    # One entry per split set, so a 40 shard export is checked once.
    checked = set()
    # Separate from `checked`, which is only there to stop a 40 shard export being reopened
    # 40 times. This one is what the closing message counts.
    verified = set()
    # Sets whose tensor values could not be inspected, so the closing line can say so.
    values_skipped = set()
    started = time.perf_counter()
    for output_file in output_files:
        shards = _gguf_shard_siblings(output_file)
        if shards[0] in checked:
            continue
        # Say what the wait is for before paying for it. Almost all of the cost is
        # gguf.GGUFReader parsing the vocabulary, which is seconds on a large one, and a
        # silent stall right after "Successfully saved" reads like a hang. Measured on
        # this design: 8 ms for a file with no vocabulary, 0.9 s at 32k entries, 15 s at
        # 262k, and the same 15 s whether that file is 16 MB or 4.2 GB, because the cost
        # is the vocabulary rather than the weights.
        if not checked:
            message = (
                "Unsloth: Reading the GGUF back to check it before anything publishes or "
                "quantizes it. This parses the file's metadata, which takes a few seconds "
                "on a large vocabulary and does not grow with the model. Set "
                "UNSLOTH_GGUF_VERIFY=0 to skip it."
            )
            if print_output: print(message)
            else: logger.info(message)
        pass
        checked.add(shards[0])
        readers = _gguf_open_shards(output_file)
        if any(reader is None for _, reader in readers):
            unreadable = ", ".join(
                os.path.basename(path) for path, reader in readers if reader is None
            )
            detail = ""
            # An ImportError never reached the file's bytes, so it says nothing about
            # the file. The reader runs in the PARENT, and `use_local_gguf` only puts
            # the writer's directory on sys.path: it does not reproduce the child's
            # PYTHONPATH or NO_LOCAL_GGUF, so that tree can import in the child and
            # fail in the parent over a dependency the child had. Refusing there would
            # reject a healthy export and offer a re-run that cannot help.
            reader_unavailable = False
            for path, reader in readers:
                if reader is not None: continue
                try:
                    _open_gguf_reader(path)
                except ImportError as error:
                    reader_unavailable = True
                    detail = f" ({type(error).__name__}: {error})"
                except Exception as error:
                    detail = f" ({type(error).__name__}: {error})"
                break
            if _writer_tree_known and not reader_unavailable:
                # Elsewhere this is only a warning because an older installed `gguf`
                # fails on a perfectly good export. Not here: this IS the tree the child
                # wrote with, so a file its own writer cannot reopen is malformed, and
                # everything above checks existence and shard numbering only.
                raise RuntimeError(
                    f"Unsloth: the GGUF converter wrote {unreadable}, and the same `gguf` "
                    f"package that wrote it cannot read it back{detail}. The file is "
                    f"malformed rather than merely newer than the reader, so it is not "
                    f"being published. Re-run the conversion, and set "
                    f"UNSLOTH_GGUF_VERIFY=0 if you need to keep the file anyway."
                )
            # Warn rather than refuse. The installed `gguf` can be older than
            # the converter that wrote this file, in which case the reader
            # fails on an export that is perfectly good, and refusing here
            # would break conversions that work today. `convert_to_gguf`
            # already rejects a missing or truncated output above.
            logger.warning(
                f"Unsloth: could not read {unreadable} with the installed gguf "
                f"package{detail}, so it was not verified. Upgrade `gguf` if you want "
                f"GGUF exports checked before they are published."
            )
            continue
        # Counted here, after the unreadable check, not when the set was first seen: a set
        # that could not be read had no metadata or tensor check run on it, and reporting
        # it as verified tells the user a gate passed that never executed.
        verified.add(shards[0])
        problems = gguf_metadata_problems(output_file, readers = readers)
        if _gguf_holds_any_float_tensor(readers):
            problems = problems + gguf_tensor_problems(
                output_file, readers = readers, float_tensors_only = True,
            )
        else:
            # Nothing in this set has a float view, so the tensor pass has nothing to
            # read and the set got the metadata gate only. Counted so the closing line
            # cannot report a value check that never happened. A mixed file does not
            # land here: its float tensors are checked and only its quantized blocks,
            # which hold bytes rather than values, are passed over.
            values_skipped.add(shards[0])
        # llama.cpp loads these, so they never refuse the file.
        for advisory in gguf_metadata_warnings(output_file, readers = readers):
            logger.warning(
                f"Unsloth: `{os.path.basename(output_file)}`: {advisory}"
            )
        if not problems:
            continue
        listed = "\n".join(f"  - {problem}" for problem in problems)
        raise RuntimeError(
            f"Unsloth: `{os.path.basename(output_file)}` did not pass post "
            f"conversion verification, so it was not published:\n{listed}\n"
            f"This is a problem with the conversion, not with your training "
            f"run. Please report it with the model name at "
            f"https://github.com/unslothai/unsloth/issues, and set "
            f"UNSLOTH_GGUF_VERIFY=0 if you need the file anyway."
        )
    if print_output and verified:
        scope = ""
        if values_skipped:
            scope = (f", metadata only for {len(values_skipped)} of them because a "
                     f"quantized block format holds bytes rather than values")
        print(f"Unsloth: Verified {len(verified)} GGUF file(s){scope} in "
              f"{time.perf_counter() - started:.1f}s.")
    elif print_output and checked:
        # Every set was skipped, so there is nothing to report as verified and saying
        # nothing at all would read as "it passed". The per-file warning above already
        # named which ones and why.
        print("Unsloth: No GGUF file could be read back, so none was verified.")


# GGUF special token id key -> the tokenizer attribute holding the same id.
# Integers, so unlike the token strings they are not rewritten on the way out
# and can be compared exactly. See `_gguf_tokenizer_problems` for why the
# strings are not compared.
_GGUF_SPECIAL_TOKEN_IDS = (
    ("tokenizer.ggml.bos_token_id",     "bos_token_id"),
    ("tokenizer.ggml.eos_token_id",     "eos_token_id"),
    ("tokenizer.ggml.padding_token_id", "pad_token_id"),
    ("tokenizer.ggml.unknown_token_id", "unk_token_id"),
)


def _gguf_tokenizer_problems(tokenizer, reader):
    """Disagreements between a tokenizer and a GGUF's tokenizer metadata.

    Checks the vocabulary is there at all, then compares the special token ids,
    which is the failure people actually hit: a GGUF written with the base
    `<eos>` instead of the instruct model's chat EOS never stops generating.

    It does NOT compare the token strings. The code this replaces tried to, with
    `saved_vocab != vocab`, and could not have worked: llama.cpp rewrites SPM
    pieces on the way out. Measured on a flawless gemma-3-270m-it export, that
    comparison reports 30 differences in 262144 tokens, every one a run of two
    or more U+2581 written out as plain spaces, and normalising those away
    leaves the lone U+2581 at id 236743 still differing because llama.cpp
    rewrites runs but not singletons. `get_vocab()` also legitimately returns
    one entry more than the file holds, because `<image_soft_token>` sits at id
    262144 against a `vocab_size` of 262144. Reproducing llama.cpp's
    per-tokenizer-type rewriting here would be a second implementation of it,
    wrong in a different way, and every error it made would reject a correct
    export. That is presumably why the original check was written so that
    `hasattr` on a dict made it unreachable.
    """
    problems = []
    field = reader.fields.get("tokenizer.ggml.tokens")
    if field is None or not field.data:
        problems.append("`tokenizer.ggml.tokens` is missing or empty, so the "
                        "GGUF carries no vocabulary")
        return problems
    for key, attribute in _GGUF_SPECIAL_TOKEN_IDS:
        gguf_field = reader.fields.get(key)
        if gguf_field is None:
            continue
        expected = getattr(tokenizer, attribute, None)
        if expected is None:
            continue
        try:
            written = int(gguf_field.contents())
        except Exception:
            continue
        if written != int(expected):
            problems.append(
                f"`{key}` is {written} in the GGUF but the tokenizer's "
                f"{attribute} is {int(expected)}"
            )
    return problems


def _model_tensor_shapes(model):
    """`{parameter name: shape tuple}` for a model, or `{}` if it has none."""
    try:
        return {name: tuple(parameter.shape) for name, parameter in model.named_parameters()}
    except Exception:
        return {}


def _gguf_shape_problems(model, readers, sample_size):
    """Sampled GGUF tensors whose shape contradicts the model's, as strings.

    `readers` is the whole shard set, not one file. A split export puts most of its
    tensors in shards 1..N, so checking only shard 0 left the bulk of the model
    unvalidated while every other check in `assert_correct_gguf` covered all of it, and a
    tensor whose dimensions were wrong past the first shard passed verification. The
    architecture and the name map come from the first readable shard, which is where the
    KV block lives; the tensor walk then runs over every readable shard, sampled per shard
    the way `gguf_tensor_problems` samples.

    Matched by NAME, through gguf-py's own `TensorNameMap` inverted, rather
    than by `shape[0]` as before: that matched `output_norm.weight` of shape
    (640,) against a (640, 2048) parameter and then ran
    `x = torch.empty_like(param); x[:] = tensor_data[:]`, which raised
    `RuntimeError: The expanded size of the tensor (2048) must match the
    existing size (640)` on a perfectly good export, which is why nothing ever
    called this function. Values are not compared: the converters transform
    tensors on the way out, so only the shape, which they preserve up to
    GGUF's reversed dimension order, is decidable here.
    """
    shapes = _model_tensor_shapes(model)
    if not shapes:
        return []
    try:
        from gguf.constants import MODEL_ARCH_NAMES  # type: ignore
        from gguf.tensor_mapping import TensorNameMap  # type: ignore
    except Exception:
        return []
    readable = [(shard, reader) for shard, reader in readers if reader is not None]
    if not readable:
        # An unreadable shard set is already reported as fatal by the metadata pass, and
        # saying it again here would only duplicate it.
        return []
    reader = readable[0][1]
    architecture = _gguf_field_text(reader, GGUF_ARCHITECTURE_KEY)
    arch_enum = next(
        (enum for enum, name in MODEL_ARCH_NAMES.items() if name == architecture), None,
    )
    if arch_enum is None:
        return []
    # `TensorNameMap`'s second argument is the BLOCK count and it expands every
    # per-block template once per index, so passing the parameter count instead builds
    # hundreds of megabytes of names for layers that do not exist (an MoE with named
    # experts has tens of thousands of parameters and a few dozen blocks). llama.cpp
    # refuses to load without `{arch}.block_count`; parameter names are the fallback.
    block_count = _gguf_field_int(reader, f"{architecture}.block_count")
    if block_count is None or block_count <= 0:
        block_count = _model_block_count(shapes)
    if block_count is None or block_count <= 0:
        return []
    try:
        name_map = TensorNameMap(arch_enum, block_count)
    except Exception:
        return []
    # HF name -> GGUF name, inverted so a sampled GGUF tensor finds its parameter.
    gguf_to_hf = {}
    for hf_name, (_, gguf_name) in name_map.mapping.items():
        for suffix in (".weight", ".bias", ""):
            candidate = f"{hf_name}{suffix}"
            if candidate in shapes:
                gguf_to_hf.setdefault(f"{gguf_name}{suffix}", candidate)
                break

    problems = []
    for shard, shard_reader in readable:
        tensors = shard_reader.tensors
        # Keyed on the shard name as well, so two shards holding the same number of
        # tensors do not sample the same positions and leave the same gaps.
        indices = _gguf_sample_indices(
            len(tensors), sample_size, ("shapes", os.path.basename(shard), len(tensors)),
        )
        for index in indices:
            tensor = tensors[index]
            hf_name = gguf_to_hf.get(tensor.name)
            if hf_name is None:
                continue
            # GGUF stores dimensions in reverse order.
            gguf_shape = tuple(int(x) for x in reversed(list(tensor.shape)))
            model_shape = shapes[hf_name]
            if gguf_shape != model_shape:
                problems.append(
                    f"`{tensor.name}` has shape {gguf_shape} but the model's "
                    f"`{hf_name}` has shape {model_shape}"
                )
    return problems


def _assert_correct_gguf(model_name, model, tokenizer, sample_size = None):
    # All Unsloth Zoo code licensed under LGPLv3
    # Verify a conversion against the model and tokenizer it came from.
    if sample_size is None:
        sample_size = _gguf_sample_size()
    readers = _gguf_open_shards(model_name)
    reader = readers[0][1]
    if reader is None:
        raise RuntimeError(
            f"Unsloth: `{os.path.basename(model_name)}` could not be read as a GGUF file."
        )

    problems = list(gguf_metadata_problems(model_name, readers = readers))
    # Not on a projector. A VLM conversion returns the text model AND the `clip` mmproj,
    # and callers hand that whole list straight to this function. An mmproj legitimately
    # carries no `tokenizer.ggml.tokens` -- it holds a vision encoder, not a vocabulary --
    # so the tokenizer pass reported a missing vocabulary and every otherwise valid
    # multimodal conversion was rejected. Same exemption the metadata pass already applies,
    # read from the same constant rather than re-spelled, so the two cannot drift.
    if _gguf_field_text(reader, GGUF_ARCHITECTURE_KEY) not in GGUF_METADATA_EXEMPT_ARCHITECTURES:
        problems += _gguf_tokenizer_problems(tokenizer, reader)
    problems += _gguf_shape_problems(model, readers, sample_size)
    problems += gguf_tensor_problems(model_name, sample_size = sample_size, readers = readers)

    if problems:
        listed = "\n".join(f"  - {problem}" for problem in problems)
        raise RuntimeError(
            f"Unsloth: `{os.path.basename(model_name)}` failed GGUF "
            f"verification:\n{listed}"
        )


def assert_correct_gguf(model_name, model, tokenizer, sample_size = None):
    """Check one or more converted GGUF files against their source model.

    OPT-IN, and deliberately not wired into the export path: it needs the live
    torch model and tokenizer, and `unsloth.save.save_to_gguf` has neither in
    scope by the time the files exist, only a directory. Call it yourself when
    you do hold both, for example straight after `save_pretrained_merged`.

    The gate that does run on every conversion is `_verify_converted_gguf`,
    which covers architecture-required metadata (`gguf_metadata_problems`) and
    tensor sanity (`gguf_tensor_problems`). Neither of those needs a model, and
    neither compares the special token ids, which is what this function adds.

    Raises on the first file that does not check out.
    """
    # All Unsloth Zoo code licensed under LGPLv3
    if type(model_name) not in (list, tuple,):
        model_name = [model_name,]
    # One entry per split set, as `_verify_converted_gguf` already does. `convert_to_gguf`
    # returns the shard LIST, so handing that straight back to this function made every
    # shard resolve to the same complete set and revalidate it: a 40 shard model was
    # reopened and reparsed 40 times, and almost all of that cost is GGUFReader parsing
    # shard 1's vocabulary, which is 15 s on its own at 262k entries.
    checked = set()
    for name in model_name:
        shards = _gguf_shard_siblings(name)
        if shards[0] in checked:
            continue
        checked.add(shards[0])
        _assert_correct_gguf(name, model, tokenizer, sample_size = sample_size)


def check_build_requirements():
    """Check if build requirements are available (tool-based approach)"""

    if IS_WINDOWS:
        missing = []

        # Check git (setup.ps1 L266: Get-Command git)
        if shutil.which('git') is None:
            missing.append('git')

        # Check cmake (setup.ps1 L290: Get-Command cmake)
        if shutil.which('cmake') is None:
            missing.append('cmake')

        # Check VS Build Tools
        cmake_generator, _ = _find_visual_studio()
        if cmake_generator is None:
            missing.append('build-essential')

        # Check OpenSSL dev
        is_installed, package_name = check_libcurl_dev()
        if not is_installed:
            missing.append(package_name)

        return missing, "windows"

    required_tools = {
        'gcc': 'build-essential',
        'cmake': 'cmake',
        'curl': 'curl',
        'git': 'git',
    }

    missing_packages = []
    system_type = check_linux_type()  # Get system type first

    for tool, package in required_tools.items():
        try:
            result = subprocess.run(['which', tool], capture_output=True, text=True)
            if result.returncode != 0:
                # Adjust package names for non-Debian systems
                if system_type == "rpm":
                    distro_packages = {
                        'build-essential': 'gcc gcc-c++ make',
                        'cmake': 'cmake',
                        'curl': 'curl',
                        'git': 'git',
                    }
                    package = distro_packages.get(package, package)
                elif system_type == "arch":
                    distro_packages = {
                        'build-essential': 'base-devel',
                        'cmake': 'cmake',
                        'curl': 'curl',
                        'git': 'git',
                    }
                    package = distro_packages.get(package, package)
                missing_packages.append(package)
        except Exception:
            missing_packages.append(package)

    # Check for libgomp (OpenMP runtime) - needed for llama.cpp CPU backend linking
    gomp_path = _find_lib_path('libgomp.so')
    if gomp_path is None:
        gomp_packages = {'debian': 'libgomp1', 'rpm': 'libgomp-devel', 'arch': 'gcc'}
        missing_packages.append(gomp_packages.get(system_type, 'libgomp1'))

    # Check for libssl-dev (OpenSSL development) - needed for HTTPS support
    ssl_path = _find_lib_path('libssl.so')
    if ssl_path is None:
        ssl_packages = {'debian': 'libssl-dev', 'rpm': 'openssl-devel', 'arch': 'openssl'}
        missing_packages.append(ssl_packages.get(system_type, 'libssl-dev'))

    # Check for libcurl development headers
    is_installed, package_name = check_libcurl_dev()
    if not is_installed:
        missing_packages.append(package_name)

    return list(set(missing_packages)), system_type  # Remove duplicates
pass

def check_libcurl_dev():
    """Check if required libcurl dev package is installed (cross-platform)"""
    system_type = check_linux_type()

    if system_type == "windows":
        root = _find_openssl_root()
        if root is not None:
            return True, "OpenSSL"
        return False, "openssl"

    if system_type == "debian":
        package_name = "libcurl4-openssl-dev"
        try:
            result = subprocess.run(['dpkg','-l', package_name], capture_output = True, text = True)
            is_installed = result.returncode == 0 and 'ii' in result.stdout
            return is_installed, package_name
        except Exception:
            return False, package_name

    elif system_type == "rpm":
        package_name = "libcurl-devel"
        try:
            result = subprocess.run(['rpm', '-q', package_name], capture_output = True, text = True)
            is_installed = result.returncode == 0
            return is_installed, package_name
        except Exception:
            return False, package_name

    elif system_type == "arch":
        package_name = "curl"
        try:
            result = subprocess.run(['pacman', '-Q', package_name], capture_output=True, text=True)
            is_installed = result.returncode == 0
            return is_installed, package_name
        except Exception:
            return False, package_name

    return False, "libcurl4-openssl-dev"
pass

def check_linux_type():
    """Determine the linux distribution type"""
    import platform

    system = platform.system().lower()

    if system == "windows":
        return "windows"

    if system != "linux":
        return "unknown"

    # Check if it's Debian/Ubuntu-based:
    if os.path.exists('/etc/debian_version'):
        return 'debian'

    # Check if it's RPM-based (CentOS/RHEL/Fedora):
    elif any(os.path.exists(f) for f in ['/etc/redhat-release', '/etc/fedora-release']):
        return 'rpm'

    # Check if it's Arch-based (Arch/Manjaro):
    elif os.path.exists('/etc/arch-release'):
        return 'arch'

    return 'unknown'
pass


@lru_cache(1)
def _check_llama_cpp_appended_system_message():
    # See https://github.com/ggml-org/llama.cpp/issues/18323
    # See https://docs.unsloth.ai/basics/inference-and-deployment/llama-server-and-openai-endpoint#llama-server-quirks
    llama_cpp_chat_file = "https://raw.githubusercontent.com/ggml-org/llama.cpp/refs/heads/master/common/chat.cpp"
    llama_cpp_appended = '''Respond in JSON format, either with `tool_call` (a request to call tools) or with `response` reply to the user's request'''
    check = requests.get(llama_cpp_chat_file, timeout = 5)
    try:
        check.raise_for_status()
        check = check.content.decode("utf-8")
        if llama_cpp_appended in check:
            logger.info("llama.cpp appends an extra system message for tools. You should consider this.")
            return llama_cpp_appended
    except:
        pass
    return ""


def add_llama_cpp_system_message(messages, tools, inplace = False):
    # See https://github.com/ggml-org/llama.cpp/issues/18323
    # See https://docs.unsloth.ai/basics/inference-and-deployment/llama-server-and-openai-endpoint#llama-server-quirks
    extra = _check_llama_cpp_appended_system_message()
    if len(messages) == 0 or messages is None:
        return messages
    if tools is None or len(tools) == 0:
        # Does not affect non tools
        return messages
    if extra == "":
        return messages
    if messages[0]["role"] == "system":
        if inplace:
            messages[0]["content"] = messages[0]["content"] + "\n\n" + extra
        else:
            messages = [{"role" : "system", "content" : messages[0]["content"]}] + messages[1:]
    else:
        if inplace:
            messages.insert(0, {"role" : "system", "content" : extra})
        else:
            messages = [{"role" : "system", "content" : extra}] + messages
    return messages

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

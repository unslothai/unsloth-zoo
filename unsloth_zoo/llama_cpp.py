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
    "use_local_gguf",
    "install_llama_cpp",
    "check_llama_cpp",
    "_download_convert_hf_to_gguf",
    "UNSLOTH_HOME",
    "LLAMA_CPP_DEFAULT_DIR",
    "IS_WINDOWS",
]

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
import inspect
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

# Check environments
keynames = "\n" + "\n".join(os.environ.keys())
IS_COLAB_ENVIRONMENT  = "\nCOLAB_"  in keynames
IS_KAGGLE_ENVIRONMENT = "\nKAGGLE_" in keynames
IS_WINDOWS = sys.platform == "win32"
KAGGLE_TMP = "/tmp"
del keynames

# Default llama.cpp location: ~/.unsloth/llama.cpp
# Override with UNSLOTH_LLAMA_CPP_PATH env var to use a custom llama.cpp install
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


@contextlib.contextmanager
def use_local_gguf():
    """Context manager to temporarily use llama.cpp's local gguf-py"""
    # Store original state
    original_sys_path = sys.path.copy()
    original_modules = set(sys.modules.keys())
    gguf_py_path = os.path.join(LLAMA_CPP_DEFAULT_DIR, "gguf-py")

    original_gguf_modules = {}

    try:
        if os.path.exists(gguf_py_path):
            logger.debug(f"Adding {gguf_py_path} to sys.path")
            sys.path.insert(1, gguf_py_path)

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

def install_package(package, sudo = False, print_output = False, print_outputs = None, system_type = "debian"):
    # All Unsloth Zoo code licensed under LGPLv3

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
        acceptance = input(f"Missing system packages. We need to execute `{install_cmd}` - do you accept? Press ENTER. Type NO if not.")
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
    import hashlib
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


def _hydrate_converter_sources(tag, install_folder, source_assets = None):
    """Copy convert_hf_to_gguf.py, conversion/ and gguf-py/ from the same-tag
    source tarball so check_llama_cpp and the converter machinery work
    without a git checkout, and tensor mappings match the binaries.

    Fork releases use "mix" tags (e.g. b9739-mix-2d6bd50) that do NOT exist on
    ggml-org, so a verbatim ggml-org download 404s and the whole prebuilt install
    fails into a source compile. Prefer the fork release's own source asset
    (llama.cpp-source-{tag}.tar.gz, passed in via source_assets) so the converter
    exactly matches the fork build; otherwise strip the -mix-... suffix and pull
    the matching upstream tag from ggml-org. Plain ggml-org tags carry no suffix,
    so this is a no-op for them (upstream_tag == tag)."""
    fork_source_name = f"llama.cpp-source-{tag}.tar.gz"
    if source_assets and fork_source_name in source_assets:
        source_url = source_assets[fork_source_name]
    else:
        upstream_tag = tag.split("-mix-")[0]
        source_url = LLAMA_CPP_SOURCE_TARBALL.format(tag = upstream_tag)
    with tempfile.TemporaryDirectory(dir = os.path.dirname(install_folder) or ".") as source_dir:
        archive_path = os.path.join(source_dir, "source.tar.gz")
        _download_archive(source_url, archive_path)
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


def _stage_prebuilt_install(llama_cpp_folder, tag, asset_name, asset_url, expected_sha256 = None, repo = "ggml-org/llama.cpp", source_assets = None):
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
        _hydrate_converter_sources(tag, staged_install, source_assets = source_assets)
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
            # C4: Only delete if the path is safe
            if _is_safe_to_delete(llama_cpp_folder):
                shutil.rmtree(llama_cpp_folder)
            else:
                raise RuntimeError(
                    f"Unsloth: llama.cpp at `{llama_cpp_folder}` appears corrupted but is not in a safe location to delete.\n"
                    f"Please manually remove or fix it."
                )
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


def _load_module_from_path(filepath, module_name):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for module {module_name} at {filepath}")
    module = importlib.util.module_from_spec(spec)
    script_dir = os.path.dirname(os.path.abspath(filepath))
    original_path = sys.path[:]
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    # Register module before execution to handle circular imports within the script if any
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        # Clean up registry if exec fails
        del sys.modules[module_name]
        raise ImportError(f"Failed to execute module {module_name} from {filepath}") from e
    finally:
        sys.path[:] = original_path
    return module
pass


_UNSLOTH_BRANDING_MARKER = b"# UNSLOTH_BRANDING_APPLIED"
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


def _conversion_sibling_info(llama_cpp_dir):
    """Hashable (path, mtime, size) tuples for conversion/{__init__,base,qwen}.py,
    folded into the patcher cache key so re-pulled checkouts re-patch. None on
    the monolithic layout."""
    conv_dir = os.path.join(llama_cpp_dir, "conversion")
    init_py  = os.path.join(conv_dir, "__init__.py")
    base_py  = os.path.join(conv_dir, "base.py")
    qwen_py  = os.path.join(conv_dir, "qwen.py")
    if not (os.path.isfile(init_py) and os.path.isfile(base_py)):
        return None
    def _stat(p):
        try:
            s = os.stat(p)
            return (p, s.st_mtime_ns, s.st_size)
        except OSError:
            return (p, 0, 0)
    return (
        _stat(init_py),
        _stat(base_py),
        _stat(qwen_py) if os.path.isfile(qwen_py) else None,
    )
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

    def _replace(match):
        load_call = match.group(1)
        suffix    = match.group(2)   # already starts with newline + indent
        indent    = match.group(3)
        return (
            load_call + b"\n"
            + indent + _UNSLOTH_BRANDING_MARKER + b"\n"
            + indent + b"if hasattr(self.metadata, 'quantized_by'): self.metadata.quantized_by = 'Unsloth'\n"
            + indent + b"if hasattr(self.metadata, 'repo_url'): self.metadata.repo_url = 'https://huggingface.co/unsloth'\n"
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
    _patch_tensor_mapping_for_qwen35(_get_llama_cpp_dir(local_script_info))
    return _download_convert_hf_to_gguf_cached(
        name,
        local_script_info,
        _conversion_sibling_info(_get_llama_cpp_dir(local_script_info)),
    )


@lru_cache(1)
def _download_convert_hf_to_gguf_cached(name, _local_script_info, _conversion_info):
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
    temp_original_file_path = None # for the finally block
    original_module_name = None    # Only set on the monolith branch
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
            # Monolith layout: original behaviour. Write the entrypoint to a
            # temp file under LLAMA_CPP_DEFAULT_DIR and import it to read
            # ModelBase._model_classes.
            with tempfile.NamedTemporaryFile(
                mode='wb', suffix=".py", prefix="original_gguf_", dir=LLAMA_CPP_DEFAULT_DIR, delete=False
            ) as temp_file:
                temp_original_file_path = temp_file.name
                temp_file.write(original_content)
                temp_file.flush()

            logger.debug(f"Loading module from temporary file: {temp_original_file_path}")
            original_module_name = f"convert_hf_to_gguf_{os.path.basename(temp_original_file_path).split('.')[0]}"

            # Set NO_LOCAL_GGUF to prevent the script from adding path again
            old_env = os.environ.get('NO_LOCAL_GGUF')
            os.environ['NO_LOCAL_GGUF'] = '1'

            try:
                module = _load_module_from_path(temp_original_file_path, original_module_name)
            finally:
                if old_env is None:
                    os.environ.pop('NO_LOCAL_GGUF', None)
                else:
                    os.environ['NO_LOCAL_GGUF'] = old_env
            ModelBase = getattr(module, 'ModelBase', None)
            ModelType = getattr(module, 'ModelType', None)

            if ModelBase is None or ModelType is None:
                logger.warning(
                    f"Unsloth: Failed to find 'ModelBase' or 'ModelType' in the original downloaded script. "
                    f"Structure might have changed. Cannot determine supported architectures."
                )
            elif not hasattr(ModelBase, '_model_classes') or not isinstance(ModelBase._model_classes, dict):
                 logger.warning(
                    f"Unsloth: 'ModelBase._model_classes' not found or not a dictionary in original script."
                     " Cannot determine supported architectures."
                )
            else:
                # Check for TEXT models
                if hasattr(ModelType, 'TEXT') and ModelType.TEXT in ModelBase._model_classes:
                    if isinstance(ModelBase._model_classes[ModelType.TEXT], dict):
                        text_archs = set(ModelBase._model_classes[ModelType.TEXT].keys())
                        supported_types.update(text_archs)
                    else:
                        logger.warning("Unsloth: ModelBase._model_classes[ModelType.TEXT] is not a dictionary.")
                else:
                    logger.info("Unsloth: No TEXT model architectures found registered in the original script.")

                # Check for VISION models
                if hasattr(ModelType, 'MMPROJ') and ModelType.MMPROJ in ModelBase._model_classes:
                    if isinstance(ModelBase._model_classes[ModelType.MMPROJ], dict):
                        vision_archs = set(ModelBase._model_classes[ModelType.MMPROJ].keys())
                        supported_types.update(vision_archs)
                    else:
                        logger.warning("Unsloth: ModelBase._model_classes[ModelType.MMPROJ] is not a dictionary.")
                else:
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

        # Cleanup module reference (only set on the monolith branch)
        if original_module_name is not None and original_module_name in sys.modules:
             del sys.modules[original_module_name]

    except Exception as e:
         logger.error(f"Unsloth: Error during loading or introspecting the original script: {e}", exc_info=True)
         if temp_original_file_path and os.path.exists(temp_original_file_path):
             try: os.remove(temp_original_file_path)
             except OSError as remove_error: logger.warning(f"Could not remove temp file {temp_original_file_path}: {remove_error}")
         raise RuntimeError(f"Failed during loading/introspection of original script: {e}") from e
    finally:
        if temp_original_file_path and os.path.exists(temp_original_file_path):
            try:
                os.remove(temp_original_file_path)
                logger.debug(f"Cleaned up temporary file: {temp_original_file_path}")
            except OSError as remove_error:
                logger.warning(f"Could not remove temporary file {temp_original_file_path}: {remove_error}")


    # --- Proceed with patching and saving ---
    try:
        patched_content = original_content # Start patching from original

        # 3. Apply Patches (gguf attributes, metadata branding - same logic as before)
        logger.info("Unsloth: Applying patches...")
        # Patch 1: gguf Attribute Handling
        try:
            archs = list(set(re.findall(rb"[\n\s]gguf\.([\.A-Z\_0-9]{3,})[\n\s\,]", patched_content)))
            archs = [x.decode("utf-8") for x in archs if not x.startswith(b"_")]
            if archs:
                all_edits = "\n".join(f"try: gguf.{x}\nexcept AttributeError: gguf.{x} = None" for x in archs).encode("utf-8")
                patched_content = re.sub(rb"(import gguf\s*\n)", rb"\1" + all_edits + b"\n\n", patched_content, count=1)
                if original_content == patched_content and archs: logger.warning("Unsloth: gguf attribute patch did not seem to apply.")
            else: logger.info("Unsloth: No specific gguf attributes found to patch.")
        except Exception as e: logger.error(f"Unsloth: Error applying gguf attribute patch: {e}", exc_info=True); raise



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
            else:
                metadata_patch_applied = False
                new_patched_content = re.sub(
                    rb"(self\.metadata \= gguf\.Metadata\.load\(.+?\))([\n\r]+([\s\t]{4,}))",
                    rb"\1\n"
                    rb"\3if hasattr(self.metadata, 'quantized_by'): self.metadata.quantized_by = 'Unsloth'\n"
                    rb"\3if hasattr(self.metadata, 'repo_url'): self.metadata.repo_url = 'https://huggingface.co/unsloth'\n"
                    rb"\3if hasattr(self.metadata, 'tags'): self.metadata.tags = ['unsloth', 'llama.cpp']\n"
                    rb"\2",
                    patched_content, count=1, flags=re.MULTILINE
                )
                if new_patched_content != patched_content: patched_content = new_patched_content; metadata_patch_applied = True
                if not metadata_patch_applied:
                     if re.search(rb"self\.metadata \= gguf\.Metadata\.load\(", patched_content): logger.warning("Unsloth: Metadata branding patch target found, but regex failed to apply.")
                     else: logger.warning("Unsloth: Metadata branding patch target 'self.metadata = gguf.Metadata.load(...)' not found.")
        except Exception as e: logger.error(f"Unsloth: Error applying metadata branding patch: {e}", exc_info=True); raise


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
                # Use a single regex to handle both quote styles
                num_experts_pattern = rb'n_experts = self\.hparams\[(["\'])num_experts\1\]'
                replacement = (
                    b"# Qwen3MoE seems to use num_local_experts instead of num_experts\n"
                    b"            n_experts = self.hparams.get('num_experts', None) or self.hparams.get('num_local_experts')"
                )

                new_patched_content = re.sub(num_experts_pattern, replacement, patched_content)
                num_experts_patch_applied = (new_patched_content != patched_content)

                if num_experts_patch_applied:
                    patched_content = new_patched_content
                else:
                    logger.warning("Unsloth: Qwen2MoE num_experts patch target not found.")

        except Exception as e:
            logger.error(f"Unsloth: Error applying Qwen2MoE num_experts patch: {e}", exc_info=True)
            raise


        # 4. Write Patched File
        # Keep package-layout entrypoints beside conversion/ so subprocess
        # execution resolves `from conversion import ...`.
        patched_dir = _llama_cpp_dir if _layout == "package" else LLAMA_CPP_DEFAULT_DIR
        os.makedirs(patched_dir, exist_ok=True)
        patched_filename = os.path.join(patched_dir, f"{name}.py")
        logger.info(f"Unsloth: Saving patched script to {patched_filename}")
        with open(patched_filename, "wb") as file:
            file.write(patched_content)

        # 5. Parse Flags from Patched Content (same logic as before)
        logger.info("Unsloth: Parsing arguments from patched script...")
        flags = re.findall(rb"parser\.add_argument\([\s]*[\"\']([^\"\']{1,})[\'\"]", patched_content)
        if not flags: raise RuntimeError(f"Unsloth: Failed parsing {patched_filename} - no arguments found.")
        defaults = re.findall(rb"parser\.add_argument\([\s]*[\"\']([^\"\']{1,})[\'\"][^\)]*(?:action=|default=)[\s]*([^,\s\)]+)", patched_content)
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

    # Strip MTP / nextn config keys so the downstream convert_hf_to_gguf.py
    # doesn't inflate block_count / inject nextn_predict_layers.
    _changed = False
    for _key in ("mtp_num_hidden_layers", "unsloth_fixed_mtp"):
        if config_file.pop(_key, None) is not None:
            _changed = True
        _tc = config_file.get("text_config")
        if _tc and _tc.pop(_key, None) is not None:
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

    if is_vlm and supported_vision_archs is not None:
        if "architectures" in config_file:
            arch = config_file["architectures"][0]
        else:
            arch = None  # MLX-style config; skip mmproj arch check
        if arch is not None and arch not in supported_vision_archs:
                is_vlm = False
                print(f"Unsloth: {arch} is not supported for MMPROJ conversion. Converting as text-only model.")

    all_output_files = []
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
        runs_to_do.append((text_args, text_output, "text model"))

        # Vision projector conversion
        mmproj_args = {
            "--outfile"        : mmproj_output,
            "--outtype"        : model_dtype if model_dtype else "bf16" if device_is_bf16_supported() else "f16",
            "--mmproj"         : "",
            "--split-max-size" : max_shard_size,
        }
        runs_to_do.append((mmproj_args, mmproj_output, "vision projector"))

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
        runs_to_do.append((args, final_output, "model"))

    # Execute conversions
    for args, output_file, description in runs_to_do:
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
        repair_note = ""
        while True:
            try:
                # encoding/errors pinned so non-UTF8 output never crashes decoding.
                if print_output:
                    result = subprocess.run(command, shell=False, check=True,
                                          encoding="utf-8", errors="replace",
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    print(result.stdout)
                else:
                    # Capture so a failure surfaces the real traceback.
                    subprocess.run(command, shell=False, check=True,
                                   encoding="utf-8", errors="replace",
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                break
            except subprocess.CalledProcessError as e:
                captured = ""
                for stream in (getattr(e, "stderr", None), getattr(e, "stdout", None)):
                    if stream:
                        captured += stream if isinstance(stream, str) else stream.decode("utf-8", errors="replace")

                # Self-heal: reinstall the converter deps (command[0] = its
                # interpreter) and retry once instead of failing.
                if not attempted_repair and _looks_like_converter_dep_error(captured):
                    attempted_repair = True
                    try:
                        repair = _reinstall_converter_deps(command[0], print_output = print_output)
                        if repair.returncode == 0:
                            continue
                        repair_note = f"\n--- dependency reinstall failed ---\n{(repair.stdout or '').strip()}"
                    except Exception as repair_error:
                        repair_note = f"\n--- dependency reinstall failed ---\n{repair_error}"

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
                raise RuntimeError(f"Unsloth: Failed to convert {description} to GGUF with command `{cmd}`: {e}{details}{repair_note}")

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

    return all_output_files, is_vlm
pass


def quantize_gguf(
    input_gguf,
    output_gguf,
    quant_type,
    quantizer_location = os.path.join(LLAMA_CPP_DEFAULT_DIR, "llama-quantize"),
    n_threads = None,
    print_output = True,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Use llama-quantize for fast quantization of GGUF files.

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

    def _quote(s):
        """Quote a path for shell usage, handling both Windows and Unix."""
        s = str(s)
        if IS_WINDOWS:
            # On Windows cmd, wrap in double quotes if path contains spaces
            return f'"{s}"' if ' ' in s else s
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
    if str(quant_type).strip().lower() == "q2_k_l":
        _extra_flags = (
            '--tensor-type "\\.ffn_down_exps=Q3_K" '
            '--tensor-type "\\.ffn_down=Q3_K" '
            '--output-tensor-type Q6_K '
            '--token-embedding-type Q4_K '
        )
        quant_type = "q2_k"

    command = (
        f"{_quote(quantizer_location)} {_extra_flags}"
        f"{_quote(input_gguf)} {_quote(output_gguf)} {quant_type} {n_threads}"
    )

    if print_output:
        print(f"Unsloth: Quantizing to {_display_quant_type}...")
        if _extra_flags:
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


def _assert_correct_gguf(model_name, model, tokenizer):
    # All Unsloth Zoo code licensed under LGPLv3
    # Verify if conversion is in fact correct by checking tokenizer and last tensor
    import gguf.gguf_reader  # type: ignore
    from gguf.gguf_reader import GGUFReader  # type: ignore

    # Stop until building tensors
    if not hasattr(GGUFReader, "__init__"):
        raise RuntimeError("Unsloth: Failed to verify GGUF: GGUFReader has no __init__")
    init_source = inspect.getsource(GGUFReader.__init__)
    text = "self._build_tensors(offs, tensors_fields"
    stop = init_source.find(text)
    if text not in init_source:
        raise RuntimeError(f"Unsloth: Failed to verify GGUF: Reader has no `{text}`")
    init_source = init_source.replace(text, text + "[-1:]")

    # Execute source and run partial GGUF reader
    source = f"class Partial_GGUFReader(GGUFReader):\n{init_source}"

    functions = dir(gguf.gguf_reader)
    functions = [x for x in functions if x in source]
    functions = f"from gguf.gguf_reader import ({','.join(functions)})"
    all_functions = {}
    exec(functions, all_functions)
    exec(source, all_functions)

    # Check if tokenizer is the same
    def check_gguf_tokenizer(tokenizer, reader):
        vocab = tokenizer.get_vocab()
        if not hasattr(reader, "fields"): return
        if not hasattr(reader.fields, "tokenizer.ggml.tokens"): return

        field = reader.fields["tokenizer.ggml.tokens"].data
        saved_vocab = [str(bytes(x), encoding = "utf-8") for x in field]

        vocab = [k for k, v in sorted(vocab.items(), key = lambda item: item[1])]
        if saved_vocab != vocab:
            raise RuntimeError("Unsloth: Failed converting to GGUF due to corrupted tokenizer.")
    pass

    # Get last tensor in file and check for exactness
    def check_gguf_last_tensor(model, reader):
        if not hasattr(reader, "tensors"): return

        last_tensor = reader.tensors[-1]
        last_tensor_data = torch.tensor(last_tensor.data)
        parameters = list(model.parameters())[-10:]

        distances = torch.ones(len(parameters), device = parameters[-1].device)
        found = False
        for k, param in enumerate(parameters):
            if param.shape[0] == last_tensor.shape[0]:
                x = torch.empty_like(param)
                x[:] = last_tensor_data[:]
                distances[k] = torch.dist(x, param)
                found = True
            pass
        pass
        if found:
            torch._assert(
                distances.min() == 0,
                "Unsloth: Failed converting to GGUF due to corrupted files."
            )
        pass
    pass

    Partial_GGUFReader = all_functions['Partial_GGUFReader']
    reader = Partial_GGUFReader(model_name, "r")
    check_gguf_last_tensor(model, reader)
    check_gguf_tokenizer(tokenizer, reader)

    # Try parsing metadata
    try:
        from gguf.scripts.gguf_dump import dump_metadata_json  # type: ignore
        class Arguments: pass
        args = Arguments()

        args.no_tensors = True
        args.model = model_name
        args.json_array = False

        # Stop prints
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            metadata = dump_metadata_json(reader, args)
        return
    except:
        pass
pass


def assert_correct_gguf(model_name, model, tokenizer):
    # All Unsloth Zoo code licensed under LGPLv3
    # Verify if conversion is in fact correct by checking tokenizer and last tensor
    if type(model_name) not in (list, tuple,):
        model_name = [model_name,]
    for name in model_name:
        _assert_correct_gguf(name, model, tokenizer)
    pass
pass


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

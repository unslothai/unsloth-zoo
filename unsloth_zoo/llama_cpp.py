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
    "_download_convert_hf_to_gguf"
]

import subprocess
import sys
import os
import time
import re
import requests
import json
from tqdm.auto import tqdm as ProgressBar
from functools import lru_cache
import inspect
import contextlib
import importlib.util
import tempfile
import logging
import torch
from pathlib import Path
import psutil

# Get a logger instance
logger = logging.getLogger(__name__)
# Configure logging basic level if not already configured elsewhere
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')

LLAMA_CPP_CONVERT_FILE = \
    "https://github.com/ggerganov/llama.cpp/raw/refs/heads/master/convert_hf_to_gguf.py"

COMMANDS_NOT_FOUND = (
    "command not found",
    "not found",
    "No such file or directory",
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
    "is deprecated"              : "Command is deprecated!",
}

# Check environments
keynames = "\n" + "\n".join(os.environ.keys())
IS_COLAB_ENVIRONMENT  = "\nCOLAB_"  in keynames
IS_KAGGLE_ENVIRONMENT = "\nKAGGLE_" in keynames
KAGGLE_TMP = "/tmp"
del keynames


@contextlib.contextmanager
def use_local_gguf():
    """Context manager to temporarily use llama.cpp's local gguf-py"""
    # Store original state
    original_sys_path = sys.path.copy()
    original_modules = set(sys.modules.keys())
    gguf_py_path = os.path.join("llama.cpp", "gguf-py")

    original_gguf_modules = {}

    try:
        # Add gguf-py to sys.path if it exists
        if os.path.exists(gguf_py_path):
            logger.debug(f"Adding {gguf_py_path} to sys.path")
            sys.path.insert(1, gguf_py_path)

            # Remove system gguf modules to force reimport
            gguf_modules = [key for key in sys.modules.keys() if key.startswith('gguf')]
            for module in gguf_modules:
                original_gguf_modules[module] = sys.modules[module]  # Store original
                del sys.modules[module]
                logger.debug(f"Removed system module {module}")

        yield  # Let the conversion happen

    finally:
        # Restore original sys.path
        sys.path[:] = original_sys_path

        # Remove any new gguf modules that were imported
        new_modules = set(sys.modules.keys()) - original_modules
        gguf_modules_to_remove = [m for m in new_modules if m.startswith('gguf')]
        for module in gguf_modules_to_remove:
            del sys.modules[module]
            logger.debug(f"Cleaned up module {module}")

        # Restore original gguf modules
        for module_name, module_obj in original_gguf_modules.items():
            sys.modules[module_name] = module_obj
            logger.debug(f"Restored original module {module_name}")

        logger.debug("Restored original Python environment")
pass

def install_package(package, sudo = False, print_output = False, print_outputs = None, system_type = "debian"):
    # All Unsloth Zoo code licensed under LGPLv3
    # Choose package manager based on system type
    if system_type == "rpm":
        pkg_manager = "yum" if os.path.exists('/usr/bin/yum') else "dnf"
        install_cmd = f"{'sudo ' if sudo else ''}{pkg_manager} install {package} -y"
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
                pkg_mgr_name = "yum/dnf" if system_type == "rpm" else "apt-get"
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
    # Check apt-get updating
    sudo = False
    print("Unsloth: Updating system package directories")

    # Choose update command based on system type
    if system_type == "rpm":
        pkg_manager = "yum" if os.path.exists('/usr/bin/yum') else "dnf"
        update_cmd = f"{pkg_manager} check-update"
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
                pkg_mgr_name = "yum/dnf" if system_type == "rpm" else "apt-get"
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
    for pip in PIP_OPTIONS:
        final_pip = pip
        with subprocess.Popen(pip, shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT) as sp:
            for line in sp.stdout:
                if line.decode("utf-8", errors = "replace").rstrip().endswith(COMMANDS_NOT_FOUND):
                    final_pip = None
                    sp.terminate()
                    break
            pass
        pass
        if final_pip is not None: return final_pip
    pass
    raise RuntimeError(f"[FAIL] Unsloth: Tried all of `{', '.join(PIP_OPTIONS)}` but failed.")
pass


def try_execute(command, sudo = False, print_output = False, print_outputs = None, cwd = None, system_type = "debian"):
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

            if print_output:
                print(line, flush=True, end="")
            if print_outputs is not None:
                print_outputs.append(line)
pass


def try_execute_with_auto_install(command, sudo=False, print_output=False, print_outputs=None, cwd = None, system_type = "debian"):
    """Try to execute a command, and if it fails due to missing package, try to install it"""
    try:
        try_execute(command, sudo, print_output, print_outputs, cwd, system_type)
    except RuntimeError as e:
        if "Command not found" in str(e):
            package_name = command.split(" ", 1)[0]
            print(f"Trying to install missing package: {package_name}")
            install_package(package_name, sudo, print_output, print_outputs, system_type)
            # Retry once
            try_execute(command, sudo, print_output, print_outputs, cwd, system_type)
        else:
            raise
pass


def check_llama_cpp(llama_cpp_folder = "llama.cpp"):
    # All Unsloth Zoo code licensed under LGPLv3
    # Check if the folder exists
    if not os.path.exists(llama_cpp_folder):
        raise RuntimeError(f"llama.cpp folder '{llama_cpp_folder}' does not exist")

    quantizer_location = None
    converter_location = None

    # Check for quantizer binary
    for quantizer in ["llama-quantize", "quantize"]:
        location = os.path.join(llama_cpp_folder, quantizer)
        if os.path.exists(location) and os.access(location, os.X_OK):
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
        pass
    pass

    if quantizer_location is None:
        # List what files are actually there for debugging
        import glob
        files_found = glob.glob(os.path.join(llama_cpp_folder, "*"))
        raise RuntimeError(
            f"Unsloth: No working quantizer found in {llama_cpp_folder}\n"
            f"Files in directory: {', '.join(os.path.basename(f) for f in files_found[:20])}"
        )
    pass

    # Check for converter script
    for converter in ["convert-hf-to-gguf.py", "convert_hf_to_gguf.py"]:
        location = os.path.join(llama_cpp_folder, converter)
        if os.path.exists(location):
            converter_location = location
            break
    pass

    if converter_location is None:
        raise RuntimeError(f"Unsloth: Failed to find converter script in {llama_cpp_folder}")
    pass

    return quantizer_location, converter_location
pass


def install_llama_cpp(
    llama_cpp_folder = "llama.cpp",
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

    if os.path.exists(llama_cpp_folder):
        try:
            quantizer, converter = check_llama_cpp(llama_cpp_folder = llama_cpp_folder)
            print(f"Unsloth: llama.cpp folder already exists - will use `{llama_cpp_folder}`")
            return quantizer, converter
        except: print(f"Unsloth: llama.cpp folder exists but binaries not found - will rebuild")
    pass

    print_outputs = []
    missing_packages, system_type = check_build_requirements()
    sudo = do_we_need_sudo()
    kwargs = {"sudo" : sudo, "print_output" : print_output, "print_outputs" : print_outputs, "system_type": system_type}

    if not missing_packages:
        print("Unsloth: All required system packages already installed!")
    else:
        packages_to_install = " ".join(missing_packages)
        print(f"Unsloth: Missing packages: {packages_to_install}")
        print(f"Unsloth: Will attempt to install missing system packages.")
        install_package(packages_to_install, sudo, system_type = system_type)

    print("Unsloth: Install llama.cpp and building - please wait 1 to 3 minutes")
    if gpu_support == "ON":
        print("Unsloth: Building llama.cpp with GPU support")

    # Clone repo if it doesn't exist
    if not os.path.exists(llama_cpp_folder):
        print("Unsloth: Cloning llama.cpp repository")
        try_execute_with_auto_install(
            f"git clone https://github.com/ggml-org/llama.cpp {llama_cpp_folder}",
            **kwargs
        )

    pip = check_pip()

    print("Unsloth: Install GGUF and other packages")
    try_execute(f"{pip} install gguf protobuf sentencepiece mistral_common", **kwargs)
    if just_clone_repo: return llama_cpp_folder

    build_success = False
    build_errors = []

    # Check for Colab / Kaggle, and deduct some CPUs to conserve memory
    cpu_count = psutil.cpu_count() or 1
    if IS_COLAB_ENVIRONMENT or IS_KAGGLE_ENVIRONMENT:
        cpu_count = cpu_count - 1
        cpu_count = max(cpu_count, 1)

    # Try make first
    try:
        if print_output: print("Trying to build with make...")
        try_execute(f"make clean", cwd = llama_cpp_folder, **kwargs)
        try_execute(f"make all -j{cpu_count}", cwd = llama_cpp_folder, **kwargs)
        build_success = True
        print("Successfully built with make")
    except Exception as e:
        build_errors.append(f"Make failed: {str(e)}")
        if print_output: print(f"Make failed, trying cmake...")
        # Use cmake instead
        try:
            # Clean up any partial build
            try_execute(f"rm -rf build", cwd = llama_cpp_folder, **kwargs)

            try_execute(
                f"cmake . -B build "\
                f"-DBUILD_SHARED_LIBS=OFF -DGGML_CUDA={gpu_support} -DLLAMA_CURL=ON",
                cwd = llama_cpp_folder,
                **kwargs
            )
            try_execute(
                f"cmake --build build --config Release "\
                f"-j{cpu_count} --clean-first --target "\
                f"{' '.join(llama_cpp_targets)}",
                cwd = llama_cpp_folder,
                **kwargs
            )
            # Move compiled objects to main folder
            try_execute(
                f"cp build/bin/llama-* .",
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
    # Register module before execution to handle circular imports within the script if any
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        # Clean up registry if exec fails
        del sys.modules[module_name]
        raise ImportError(f"Failed to execute module {module_name} from {filepath}") from e
    return module
pass


@lru_cache(1)
def _download_convert_hf_to_gguf(
    name = "unsloth_convert_hf_to_gguf",
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Downloads from llama.cpp's Github report

    # Ensure llama.cpp directory exists
    os.makedirs("llama.cpp", exist_ok=True)

    supported_types = set() # Initialize outside try block
    temp_original_file_path = None # Initialize for finally block

    try:
        # 1. Download the file
        response = requests.get(LLAMA_CPP_CONVERT_FILE)
        response.raise_for_status()
        original_content = response.content

        # 2. Introspect Original Script for Supported Architectures
        logger.info("Unsloth: Identifying llama.cpp gguf supported architectures...")
        with tempfile.NamedTemporaryFile(
            mode='wb', suffix=".py", prefix="original_gguf_", dir="llama.cpp", delete=False
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
            # Restore environment
            if old_env is None:
                os.environ.pop('NO_LOCAL_GGUF', None)
            else:
                os.environ['NO_LOCAL_GGUF'] = old_env

        # --- Extract Supported Architectures (TEXT and VISION) ---
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

        # Cleanup module reference
        if original_module_name in sys.modules:
             del sys.modules[original_module_name]

    except Exception as e:
         logger.error(f"Unsloth: Error during download or introspection of original script: {e}", exc_info=True)
         if temp_original_file_path and os.path.exists(temp_original_file_path):
             try: os.remove(temp_original_file_path)
             except OSError as remove_error: logger.warning(f"Could not remove temp file {temp_original_file_path}: {remove_error}")
         raise RuntimeError(f"Failed during download/introspection of original script: {e}") from e
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



        # Patch 2: Metadata Branding
        try:
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


        # 4. Write Patched File
        patched_filename = f"llama.cpp/{name}.py"
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


def convert_to_gguf(
    model_name,
    input_folder,
    model_dtype = "bf16",
    quantization_type = "bf16", # dequantizing from q8_0 disallow, setting default to bf16
    converter_location = "llama.cpp/unsloth_convert_hf_to_gguf.py",
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

    config_file = os.path.join(input_folder, "config.json")
    if not os.path.exists(config_file):
        raise RuntimeError(f"Unsloth: `config.json` does not exist inside `{input_folder}`.")

    # Load config.json
    with open(config_file, "r", encoding = "utf-8") as config_file:
        config_file = json.load(config_file)
    pass

    # Check if arch is supported
    supported_types = supported_vision_archs | supported_text_archs
    if supported_types is not None:
        assert("architectures" in config_file)
        arch = config_file["architectures"][0]
        if arch not in supported_types:
            raise NotImplementedError(
                f"Unsloth: llama.cpp GGUF conversion does not yet support "\
                f"converting model types of `{arch}`."
            )
    pass

    if is_vlm and supported_vision_archs is not None:
        arch = config_file["architectures"][0]
        if arch not in supported_vision_archs:
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
                mmproj_dtype = model_dtype if model_dtype else ("bf16" if torch.cuda.is_bf16_supported() else "f16")
                mmproj_output = f"{base_name}.{mmproj_dtype.upper()}-mmproj.gguf"
            else:
                text_output = f"{model_name}.{quantization_type.upper()}.gguf"
                mmproj_dtype = model_dtype if model_dtype else ("bf16" if torch.cuda.is_bf16_supported() else "f16")
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
            "--outtype"        : model_dtype if model_dtype else "bf16" if torch.cuda.is_bf16_supported() else "f16",
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
        args_str = " ".join(f"{k} {v}" for k, v in args.items())
        command = f"python {converter_location} {args_str} {input_folder}"

        try:
            if print_output:
                result = subprocess.run(command, shell=True, check=True, text=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                print(result.stdout)
            else:
                subprocess.run(command, shell=True, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            if print_output and hasattr(e, 'stdout') and e.stdout:
                print(e.stdout)
            raise RuntimeError(f"Unsloth: Failed to convert {description} to GGUF: {e}")

        # Simple validation using native Python
        if not os.path.exists(output_file):
            raise RuntimeError(f"Unsloth: Failed to convert {description} - output file {output_file} not created")

        all_output_files.append(output_file)

        if print_output:
            file_size_bytes = os.path.getsize(output_file)
            if file_size_bytes >= 1024**3:  # GB
                size_str = f"{file_size_bytes / (1024**3):.1f}G"
            elif file_size_bytes >= 1024**2:  # MB
                size_str = f"{file_size_bytes / (1024**2):.1f}M"
            else:
                size_str = f"{file_size_bytes / 1024:.1f}K"
            print(f"Unsloth: Successfully saved {description} GGUF to: {output_file} (size: {size_str})")

    return all_output_files, is_vlm
pass


def quantize_gguf(
    input_gguf,
    output_gguf,
    quant_type,
    quantizer_location = "llama.cpp/llama-quantize",
    n_threads = None,
    print_output = True,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Use llama-quantize for fast quantization of GGUF files.

    if n_threads is None:
        n_threads = psutil.cpu_count()
        if n_threads is None:
            n_threads = 1
        n_threads *= 2

    command = f"{quantizer_location} {input_gguf} {output_gguf} {quant_type} {n_threads}"

    if print_output:
        print(f"Unsloth: Quantizing to {quant_type}...")

    try:
        if print_output:
            result = subprocess.run(command, shell=True, check=True, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            print(result.stdout)
        else:
            subprocess.run(command, shell=True, check=True, capture_output=True)

    except subprocess.CalledProcessError as e:
        if print_output and hasattr(e, 'stdout') and e.stdout:
            print(e.stdout)
        raise RuntimeError(f"Failed to quantize {input_gguf} to {quant_type}: {e}")

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
    import gguf.gguf_reader
    from gguf.gguf_reader import GGUFReader

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

    reader = Partial_GGUFReader(model_name, "r")
    check_gguf_last_tensor(model, reader)
    check_gguf_tokenizer(tokenizer, reader)

    # Try parsing metadata
    try:
        from gguf.scripts.gguf_dump import dump_metadata_json
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
                # Adjust package names for RPM-based systems
                if system_type == "rpm":
                    rpm_packages = {
                        'build-essential': 'gcc gcc-c++ make',
                        'cmake': 'cmake',
                        'curl': 'curl',
                        'git': 'git',
                    }
                    package = rpm_packages.get(package, package)
                missing_packages.append(package)
        except Exception:
            missing_packages.append(package)

    # Check for libcurl development headers
    is_installed, package_name = check_libcurl_dev()
    if not is_installed:
        missing_packages.append(package_name)

    return list(set(missing_packages)), system_type  # Remove duplicates
pass

def check_libcurl_dev():
    """Check if required libcurl dev package is installed (cross-platform)"""
    system_type = check_linux_type()

    if system_type == "debian":
        package_name = "libcurl4-openssl-dev"
        try:
            result = subprocess.run(['dpkg','-l', package_name], capture_output = True, text = True)
            is_installed = result.returncode == 0 and 'ii' in result.stdout
            return is_installed, package_name
        except Exception:
            return False, package_name

    elif system_type == "rpm":
        package_name = "libcurl-dev"
        try:
            result = subprocess.run(['rpm', '-q', package_name], capture_output = True, text = True)
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

    if system != "linux":
        return "unknown"

    # Check if it's Debian/Ubuntu-based:
    if os.path.exists('/etc/debian_version'):
        return 'debian'

    # Check if it's RPM-based (CentOS/RHEL/Fedora):
    elif any(os.path.exists(f) for f in ['/etc/redhat-release', '/etc/fedora-release']):
        return 'rpm'

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

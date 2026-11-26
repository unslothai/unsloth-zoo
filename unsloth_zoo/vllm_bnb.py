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

"""vLLM's bitsandbytes: in tree on vLLM <= 0.27.1, the vllm-bnb-plugin package from vLLM 0.28."""

import importlib
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import sys

__all__ = ["resolve", "reset", "install_plugin", "INSTALL_HINT", "PLUGIN_SPEC"]

PLUGIN_SPEC = "vllm-bnb-plugin==0.0.3"
INSTALL_HINT = f"pip install {PLUGIN_SPEC}"

_IN_TREE = (
    "vllm.model_executor.layers.quantization.bitsandbytes",
    "vllm.model_executor.model_loader.bitsandbytes_loader",
)
_PLUGIN = ("vllm_bnb_plugin.bitsandbytes", "vllm_bnb_plugin.bitsandbytes_loader")
_PLUGIN_NAME = "bitsandbytes"  # its vllm.general_plugins entry point
_RESOLVED = None

_find_spec = importlib.util.find_spec
_import = importlib.import_module


def _import_pair(pair):
    return _import(pair[0]), _import(pair[1])


def _alias_rename_mapper():
    # Plugin 0.0.3 calls get_rename_mapper; vLLM 0.28.0 (where Llama / Qwen have a mapper) names it get_unstacked_mapper.
    WeightsMapper = _import("vllm.model_executor.models.utils").WeightsMapper
    if not hasattr(WeightsMapper, "get_rename_mapper") and hasattr(WeightsMapper, "get_unstacked_mapper"):
        WeightsMapper.get_rename_mapper = WeightsMapper.get_unstacked_mapper
        return True
    return False


def _keep_plugin_off():
    # vLLM imports every installed plugin at engine build; with built-in bitsandbytes that registers
    # vllm::apply_bnb_4bit twice. VLLM_PLUGINS is vLLM's allow-list, so list everything but it.
    if _find_spec("vllm_bnb_plugin") is None:
        return False
    current = os.environ.get("VLLM_PLUGINS")
    if current is None:
        # Per distribution: a bare entry_points() is a dict of groups on Python 3.9 to 3.11.
        names = sorted({ep.name for dist in importlib.metadata.distributions()
                        for ep in dist.entry_points if ep.group.startswith("vllm.")})
    else:
        names = current.split(",")
    os.environ["VLLM_PLUGINS"] = ",".join(name for name in names if name and name != _PLUGIN_NAME)
    return True


def reset():
    global _RESOLVED
    _RESOLVED = None


def install_plugin():
    """pip / uv install PLUGIN_SPEC with --no-deps into this interpreter. (ok, output tail)."""
    if _find_spec("pip") is not None:
        cmd = [sys.executable, "-m", "pip", "install", "--no-deps", PLUGIN_SPEC]
    elif shutil.which("uv") is not None:
        cmd = ["uv", "pip", "install", "--python", sys.executable, "--no-deps", PLUGIN_SPEC]
    else:
        return False, "neither pip nor uv is available"
    try:
        out = subprocess.run(cmd, capture_output = True, text = True, timeout = 600)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    importlib.invalidate_caches()
    text = (out.stdout + out.stderr).strip()[-500:]
    return out.returncode == 0 and _find_spec("vllm_bnb_plugin") is not None, text


def resolve():
    """(quant module, loader module, "vllm" | "vllm-bnb-plugin"), or Nones when neither is installed."""
    global _RESOLVED
    if _RESOLVED is not None:
        return _RESOLVED
    if _find_spec("vllm") is None:
        return (None, None, None)
    try:
        _RESOLVED = (*_import_pair(_IN_TREE), "vllm")
    except ImportError:
        pass
    else:
        _keep_plugin_off()
        return _RESOLVED
    if _find_spec("vllm_bnb_plugin") is None:
        return (None, None, None)
    _alias_rename_mapper()
    # Before load_vllm patches its config; idempotent, and loads it even when VLLM_PLUGINS leaves it out.
    _import("vllm_bnb_plugin").register()
    _RESOLVED = (*_import_pair(_PLUGIN), "vllm-bnb-plugin")
    return _RESOLVED

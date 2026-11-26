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

"""vLLM's bitsandbytes: in tree on vLLM <= 0.27.1, vllm-bnb-plugin from 0.28 (vllm-project/vllm#43529)."""

import importlib
import importlib.util

__all__ = ["resolve", "INSTALL_HINT"]

INSTALL_HINT = "pip install vllm-bnb-plugin"

_IN_TREE = (
    "vllm.model_executor.layers.quantization.bitsandbytes",
    "vllm.model_executor.model_loader.bitsandbytes_loader",
)
_PLUGIN = ("vllm_bnb_plugin.bitsandbytes", "vllm_bnb_plugin.bitsandbytes_loader")
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


def resolve():
    """(quant module, loader module, "vllm" | "vllm-bnb-plugin"), or Nones when neither is installed."""
    global _RESOLVED
    if _RESOLVED is not None:
        return _RESOLVED
    if _find_spec("vllm") is None:
        return (None, None, None)
    try:
        _RESOLVED = (*_import_pair(_IN_TREE), "vllm")
        return _RESOLVED
    except ImportError:
        pass
    if _find_spec("vllm_bnb_plugin") is None:
        return (None, None, None)
    _alias_rename_mapper()
    # Before load_vllm patches its config; idempotent, and loads it even when VLLM_PLUGINS leaves it out.
    _import("vllm_bnb_plugin").register()
    _RESOLVED = (*_import_pair(_PLUGIN), "vllm-bnb-plugin")
    return _RESOLVED

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored by Unsloth from vllm-project/vllm-bnb-plugin v0.0.3 (commit d4914a9), Apache-2.0 (LICENSE in this
# directory). The rest of Unsloth Zoo is LGPLv3; the files in this directory stay under Apache-2.0.
"""bitsandbytes for vLLM >= 0.28 (vllm-project/vllm#43529 moved it to vllm-bnb-plugin), used when neither
vLLM nor an installed plugin provides it.

Changes from upstream v0.0.3:
- this `__init__` replaces the plugin's (adds `resolve()`, no entry point).
- custom op renamed `torch.ops.vllm.unsloth_apply_bnb_4bit` to avoid colliding with the plugin's.
- loader falls back to `get_unstacked_mapper` (vLLM 0.28.0) when `get_rename_mapper` is missing.
"""

import importlib
import importlib.util
import os
from typing import Any

__all__ = ["register", "resolve"]

_REGISTERED = False
_RESOLVED = None

_IN_TREE = (
    "vllm.model_executor.layers.quantization.bitsandbytes",  # vLLM <= 0.27.1
    "vllm.model_executor.model_loader.bitsandbytes_loader",
)
_PLUGIN = ("vllm_bnb_plugin.bitsandbytes", "vllm_bnb_plugin.bitsandbytes_loader")
_VENDORED = ("unsloth_zoo.vllm_bnb.bitsandbytes", "unsloth_zoo.vllm_bnb.bitsandbytes_loader")


def _enforce_eager_for_8bit(model_config: Any) -> None:
    from .quantization.utils import logger

    quant_config = model_config.model_arch_config.quantization_config
    if quant_config is not None and quant_config.get("load_in_8bit", False) and not model_config.enforce_eager:
        logger.warning("CUDA graph is not supported on BitsAndBytes 8bit yet, fallback to the eager mode.")
        model_config.enforce_eager = True


def _patch_engine_args() -> None:
    """Pre-quantized bnb checkpoints also need the bnb loader (upstream plugin behaviour)."""
    from vllm.engine.arg_utils import EngineArgs

    if getattr(EngineArgs, "_unsloth_vllm_bnb_patched", False):
        return
    _orig_create_model_config = EngineArgs.create_model_config
    _orig_create_engine_config = EngineArgs.create_engine_config

    def _patched_create_model_config(self, *args, **kwargs):
        result = _orig_create_model_config(self, *args, **kwargs)
        if result.quantization == "bitsandbytes":
            _enforce_eager_for_8bit(result)
        return result

    def _patched_create_engine_config(self, *args, **kwargs):
        result = _orig_create_engine_config(self, *args, **kwargs)
        if result.model_config.quantization == "bitsandbytes":
            self.quantization = "bitsandbytes"
            self.load_format = "bitsandbytes"
            result.load_config.load_format = "bitsandbytes"
        return result

    EngineArgs.create_model_config = _patched_create_model_config
    EngineArgs.create_engine_config = _patched_create_engine_config
    EngineArgs._unsloth_vllm_bnb_patched = True


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.model_loader import register_model_loader

    from .bitsandbytes import BitsAndBytesConfig
    from .bitsandbytes_loader import BitsAndBytesModelLoader

    register_quantization_config("bitsandbytes")(BitsAndBytesConfig)
    register_model_loader("bitsandbytes")(BitsAndBytesModelLoader)
    _patch_engine_args()
    _REGISTERED = True


_find_spec = importlib.util.find_spec
_import = importlib.import_module


def _import_pair(pair):
    return _import(pair[0]), _import(pair[1])


def _register_plugin():
    _import("vllm_bnb_plugin").register()


def resolve():
    """(quant module, loader module, source in "vllm" / "vllm-bnb-plugin" / "unsloth"), or Nones.

    An installed plugin wins: vLLM loads it anyway and a second registration would silently replace it.
    """
    global _RESOLVED
    if _RESOLVED is not None:
        return _RESOLVED
    if _find_spec("vllm") is None:
        return (None, None, None)
    # UNSLOTH_VLLM_BNB forces a provider (unsloth / vllm-bnb-plugin / vllm); one that cannot load raises.
    forced = os.environ.get("UNSLOTH_VLLM_BNB", "").strip().lower()
    if forced == "unsloth":
        register()
        quant, loader = _import_pair(_VENDORED)
        _RESOLVED = (quant, loader, "unsloth")
        return _RESOLVED
    if forced == "vllm-bnb-plugin":
        _register_plugin()
        quant, loader = _import_pair(_PLUGIN)
        _RESOLVED = (quant, loader, "vllm-bnb-plugin")
        return _RESOLVED
    if forced == "vllm":
        quant, loader = _import_pair(_IN_TREE)
        _RESOLVED = (quant, loader, "vllm")
        return _RESOLVED
    try:
        quant, loader = _import_pair(_IN_TREE)
        _RESOLVED = (quant, loader, "vllm")
        return _RESOLVED
    except ImportError:
        pass
    if _find_spec("bitsandbytes") is None:
        return (None, None, None)
    if _find_spec("vllm_bnb_plugin") is not None:
        try:
            _register_plugin()
            quant, loader = _import_pair(_PLUGIN)
            _RESOLVED = (quant, loader, "vllm-bnb-plugin")
            return _RESOLVED
        except Exception as e:  # plugin built for another vLLM
            print(f"Unsloth: vllm-bnb-plugin failed to load ({type(e).__name__}: {e}); using Unsloth's copy.")
    register()
    quant, loader = _import_pair(_VENDORED)
    _RESOLVED = (quant, loader, "unsloth")
    return _RESOLVED

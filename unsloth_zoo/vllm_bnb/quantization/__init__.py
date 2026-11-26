# SPDX-License-Identifier: Apache-2.0
# Vendored by Unsloth from vllm-project/vllm-bnb-plugin v0.0.3 (commit d4914a9), Apache-2.0 (LICENSE in this
# directory). Changes: relocated under unsloth_zoo.vllm_bnb; see unsloth_zoo/vllm_bnb/__init__.py for the list.

from .config import BitsAndBytesConfig
from .fused_moe import BitsAndBytesMoEMethod
from .linear import BitsAndBytesLinearMethod, apply_bnb_4bit
from .params import BitsAndBytes4bitParameter, BitsAndBytes8bitParameter
from .utils import (
    _adjust_bitsandbytes_4bit_shard,
    _check_bitsandbytes_version,
    _get_min_bitsandbytes_version,
    calculate_quant_ratio,
    is_layer_skipped_bnb,
    logger,
)

__all__ = [
    "BitsAndBytes4bitParameter",
    "BitsAndBytes8bitParameter",
    "BitsAndBytesConfig",
    "BitsAndBytesLinearMethod",
    "BitsAndBytesMoEMethod",
    "_adjust_bitsandbytes_4bit_shard",
    "_check_bitsandbytes_version",
    "_get_min_bitsandbytes_version",
    "apply_bnb_4bit",
    "calculate_quant_ratio",
    "is_layer_skipped_bnb",
    "logger",
]

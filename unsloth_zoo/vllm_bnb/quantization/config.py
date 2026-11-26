# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored by Unsloth from vllm-project/vllm-bnb-plugin v0.0.3 (commit d4914a9), Apache-2.0 (LICENSE in this
# directory). Changes: relocated under unsloth_zoo.vllm_bnb; see unsloth_zoo/vllm_bnb/__init__.py for the list.

from typing import Any

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig,
    QuantizationMethods,
)

from .utils import is_layer_skipped_bnb


class BitsAndBytesConfig(QuantizationConfig):
    """Config class for BitsAndBytes Quantization.

    Reference: https://arxiv.org/abs/2305.14314
    """

    def __init__(
        self,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "float32",
        bnb_4bit_quant_storage: str = "uint8",
        bnb_4bit_quant_type: str = "fp4",
        bnb_4bit_use_double_quant: bool = False,
        llm_int8_enable_fp32_cpu_offload: bool = False,
        llm_int8_has_fp16_weight: bool = False,
        llm_int8_skip_modules: list[str] | None = None,
        llm_int8_threshold: float = 6.0,
    ) -> None:
        super().__init__()
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_quant_storage = bnb_4bit_quant_storage
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        self.llm_int8_enable_fp32_cpu_offload = llm_int8_enable_fp32_cpu_offload
        self.llm_int8_has_fp16_weight = llm_int8_has_fp16_weight
        self.llm_int8_skip_modules = llm_int8_skip_modules or []
        self.llm_int8_threshold = llm_int8_threshold

        if self.bnb_4bit_quant_storage not in ["uint8"]:
            raise ValueError(
                f"Unsupported bnb_4bit_quant_storage: {self.bnb_4bit_quant_storage}"
            )

    def __repr__(self) -> str:
        return (
            f"BitsAndBytesConfig(load_in_8bit={self.load_in_8bit}, "
            f"load_in_4bit={self.load_in_4bit}, "
            f"bnb_4bit_compute_dtype={self.bnb_4bit_compute_dtype}, "
            f"bnb_4bit_quant_storage={self.bnb_4bit_quant_storage}, "
            f"bnb_4bit_quant_type={self.bnb_4bit_quant_type}, "
            f"llm_int8_skip_modules={self.llm_int8_skip_modules})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "bitsandbytes"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BitsAndBytesConfig":
        def get_safe_value(config: dict[str, Any], keys: list[str], default_value=None):
            try:
                value = cls.get_from_keys(config, keys)
                return value if value is not None else default_value
            except ValueError:
                return default_value

        load_in_8bit = get_safe_value(config, ["load_in_8bit"], default_value=False)
        load_in_4bit = get_safe_value(config, ["load_in_4bit"], default_value=True)
        bnb_4bit_compute_dtype = get_safe_value(
            config, ["bnb_4bit_compute_dtype"], default_value="float32"
        )
        bnb_4bit_quant_storage = get_safe_value(
            config, ["bnb_4bit_quant_storage"], default_value="uint8"
        )
        bnb_4bit_quant_type = get_safe_value(
            config, ["bnb_4bit_quant_type"], default_value="fp4"
        )
        bnb_4bit_use_double_quant = get_safe_value(
            config, ["bnb_4bit_use_double_quant"], default_value=False
        )
        llm_int8_enable_fp32_cpu_offload = get_safe_value(
            config, ["llm_int8_enable_fp32_cpu_offload"], default_value=False
        )
        llm_int8_has_fp16_weight = get_safe_value(
            config, ["llm_int8_has_fp16_weight"], default_value=False
        )
        llm_int8_skip_modules = get_safe_value(
            config, ["llm_int8_skip_modules"], default_value=[]
        )
        llm_int8_threshold = get_safe_value(
            config, ["llm_int8_threshold"], default_value=6.0
        )

        return cls(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_quant_storage=bnb_4bit_quant_storage,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload,
            llm_int8_has_fp16_weight=llm_int8_has_fp16_weight,
            llm_int8_skip_modules=llm_int8_skip_modules,
            llm_int8_threshold=llm_int8_threshold,
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> LinearMethodBase | object | None:
        from .fused_moe import BitsAndBytesMoEMethod
        from .linear import BitsAndBytesLinearMethod

        if isinstance(layer, LinearBase):
            if is_layer_skipped_bnb(prefix, self.llm_int8_skip_modules):
                return UnquantizedLinearMethod()
            return BitsAndBytesLinearMethod(self)
        if isinstance(layer, RoutedExperts):
            return BitsAndBytesMoEMethod(self, layer.moe_config)
        return None

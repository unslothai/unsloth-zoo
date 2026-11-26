# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored by Unsloth from vllm-project/vllm-bnb-plugin v0.0.3 (commit d4914a9), Apache-2.0 (LICENSE in this
# directory). Changes: relocated under unsloth_zoo.vllm_bnb; see unsloth_zoo/vllm_bnb/__init__.py for the list.

import torch
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.linear import set_weight_attrs

from .config import BitsAndBytesConfig
from .utils import _check_bitsandbytes_version, calculate_quant_ratio


class BitsAndBytesMoEMethod(FusedMoEMethodBase):
    """MoE method for BitsAndBytes."""

    def __init__(
        self,
        quant_config: BitsAndBytesConfig,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        _check_bitsandbytes_version()
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if self.quant_config.load_in_8bit:
            call_fun = self._create_weights_8bit
        else:
            call_fun = self._create_weights_4bit
        call_fun(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        del layer
        return None

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe import fused_experts

        del shared_experts, shared_experts_input
        if self.quant_config.load_in_8bit:
            w13, w2 = self._apply_8bit_dequant(layer)
        else:
            w13, w2 = self._apply_4bit_dequnt(layer)
        return fused_experts(
            hidden_states=x,
            w1=w13,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
        )

    def _create_weights_4bit(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        quant_ratio = calculate_quant_ratio(params_dtype)
        w13_total_size = (
            hidden_size * 2 * intermediate_size_per_partition
        ) // quant_ratio
        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_total_size,
                1,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        set_weight_attrs(
            w13_qweight,
            {
                "num_experts": num_experts,
                "input_dim": hidden_size,
                "output_dim": 2 * intermediate_size_per_partition,
                "experts_shape": (
                    num_experts,
                    intermediate_size_per_partition * 2,
                    hidden_size,
                ),
                "pack_factor": quant_ratio,
                "is_sharded_weight": True,
            },
        )

        w2_total_size = (hidden_size * intermediate_size_per_partition) // quant_ratio
        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w2_total_size,
                1,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            w2_qweight,
            {
                "num_experts": num_experts,
                "input_dim": intermediate_size_per_partition,
                "output_dim": hidden_size,
                "experts_shape": (
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                ),
                "pack_factor": quant_ratio,
                "is_sharded_weight": True,
            },
        )
        layer.register_parameter("w2_weight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

    def _create_weights_8bit(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del layer, num_experts, hidden_size, intermediate_size_per_partition
        del params_dtype, extra_weight_attrs
        raise NotImplementedError

    def _apply_4bit_dequnt(
        self, layer: torch.nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from bitsandbytes.functional import dequantize_4bit

        w13 = dequantize_4bit(
            layer.w13_weight.reshape(-1, 1),
            layer.w13_weight.bnb_quant_state,
        )
        w2 = dequantize_4bit(
            layer.w2_weight.reshape(-1, 1),
            layer.w2_weight.bnb_quant_state,
        )
        w13 = w13.reshape(layer.w13_weight.experts_shape)
        w2 = w2.reshape(layer.w2_weight.experts_shape)
        return w13, w2

    def _apply_8bit_dequant(
        self, layer: torch.nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del layer
        raise NotImplementedError

    def load_expert_weight(
        self,
        layer: RoutedExperts,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        del weight_name

        shard_dim = 0
        expert_data = param.data[expert_id]

        if shard_id == "w2":
            if expert_data.shape != loaded_weight.shape:
                raise ValueError(
                    "BitsAndBytes quantization with padded hidden_size "
                    "(e.g., from DeepEP) is not supported. "
                    f"Parameter shape {tuple(expert_data.shape)} != "
                    f"checkpoint shape {tuple(loaded_weight.shape)}"
                )
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            layer._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=layer.moe_config.tp_rank,
                load_full=True,
            )
        else:
            return None

        return True if return_success else None

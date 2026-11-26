# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored by Unsloth from vllm-project/vllm-bnb-plugin v0.0.3 (commit d4914a9), Apache-2.0 (LICENSE in this
# directory). Changes: relocated under unsloth_zoo.vllm_bnb; see unsloth_zoo/vllm_bnb/__init__.py for the list.

import torch
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
    set_weight_attrs,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .config import BitsAndBytesConfig
from .params import BitsAndBytes4bitParameter, BitsAndBytes8bitParameter
from .utils import _check_bitsandbytes_version, calculate_quant_ratio


def _apply_bnb_4bit(
    x: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
    out: torch.Tensor,
) -> None:
    from bitsandbytes import matmul_4bit

    quant_states = weight.bnb_quant_state
    current_index = 0
    for i in range(len(quant_states)):
        output_size = quant_states[i].shape[0]
        out[:, current_index : current_index + output_size] = matmul_4bit(
            x,
            weight[offsets[i] : offsets[i + 1]].t(),
            quant_states[i],
        )
        current_index += output_size


def _apply_bnb_4bit_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
    out: torch.Tensor,
) -> None:
    del x, weight, offsets, out
    return


try:
    direct_register_custom_op(
        op_name="unsloth_apply_bnb_4bit",
        op_func=_apply_bnb_4bit,
        mutates_args=["out"],
        fake_impl=_apply_bnb_4bit_fake,
        dispatch_key=current_platform.dispatch_key,
    )
    apply_bnb_4bit = torch.ops.vllm.unsloth_apply_bnb_4bit
except AttributeError as error:
    raise error


@register_weight_loader_v2_supported_method
class BitsAndBytesLinearMethod(LinearMethodBase):
    """Linear method for BitsAndBytes."""

    def __init__(self, quant_config: BitsAndBytesConfig):
        _check_bitsandbytes_version()
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size

        def create_qweight_for_8bit() -> BitsAndBytes8bitParameter:
            qweight = BitsAndBytes8bitParameter(
                data=torch.empty(
                    sum(output_partition_sizes),
                    input_size_per_partition,
                    dtype=torch.int8,
                ),
                input_dim=0,
                output_dim=0,
                weight_loader=extra_weight_attrs["weight_loader"],
            )
            set_weight_attrs(
                qweight,
                {
                    "pack_factor": 1,
                    "generation": 0,
                },
            )
            return qweight

        def create_qweight_for_4bit() -> BitsAndBytes4bitParameter:
            quant_ratio = calculate_quant_ratio(params_dtype)
            total_size = input_size_per_partition * sum(output_partition_sizes)
            if total_size % quant_ratio != 0:
                raise ValueError(
                    "The input size is not aligned with the quantized weight shape."
                )

            qweight = BitsAndBytes4bitParameter(
                data=torch.empty(total_size // quant_ratio, 1, dtype=torch.uint8),
                input_dim=0,
                output_dim=0,
                packed_factor=quant_ratio,
                packed_dim=0,
                logical_width=sum(output_partition_sizes),
                weight_loader=extra_weight_attrs["weight_loader"],
            )
            set_weight_attrs(
                qweight,
                {
                    "pack_factor": quant_ratio,
                },
            )
            return qweight

        qweight = (
            create_qweight_for_8bit()
            if self.quant_config.load_in_8bit
            else create_qweight_for_4bit()
        )
        layer.register_parameter("weight", qweight)
        extra_weight_attrs = {
            key: value
            for key, value in extra_weight_attrs.items()
            if key != "weight_loader"
        }
        set_weight_attrs(qweight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.quant_config.load_in_8bit:
            return self._apply_8bit_weight(layer, x, bias)
        return self._apply_4bit_weight(layer, x, bias)

    def _apply_8bit_weight(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from bitsandbytes import MatmulLtState, matmul

        original_type = x.dtype
        original_shape = x.shape
        reshape_after_matmul = False
        if x.ndim > 2:
            x = x.reshape(-1, x.size(-1))
            reshape_after_matmul = True
        bf_x = x.to(torch.bfloat16)

        qweight = layer.weight
        offsets = qweight.bnb_shard_offsets
        quant_states = qweight.bnb_quant_state
        matmul_states = qweight.matmul_state
        generation = qweight.generation

        out_dim_0 = x.shape[0]
        out_dim_1 = sum(quant_state[1].shape[0] for quant_state in quant_states.items())
        out = torch.empty(out_dim_0, out_dim_1, dtype=torch.float16, device=x.device)

        current_index = 0
        for i in range(len(quant_states)):
            output_size = quant_states[i].shape[0]

            if generation in (0, 1):
                matmul_states[i] = MatmulLtState()
                matmul_states[i].CB = qweight[offsets[i] : offsets[i + 1]]
                matmul_states[i].SCB = quant_states[i].to(x.device)
                matmul_states[i].threshold = self.quant_config.llm_int8_threshold
                matmul_states[
                    i
                ].has_fp16_weights = self.quant_config.llm_int8_has_fp16_weight
                matmul_states[i].is_training = False
                if (
                    matmul_states[i].threshold > 0.0
                    and not matmul_states[i].has_fp16_weights
                ):
                    matmul_states[i].use_pool = True

            new_x = bf_x.unsqueeze(0)
            out[:, current_index : current_index + output_size] = matmul(
                new_x,
                qweight[offsets[i] : offsets[i + 1]],
                state=matmul_states[i],
            )
            current_index += output_size

        out = out.to(original_type)
        if reshape_after_matmul:
            out = out.view(*original_shape[:-1], out.size(-1))
        if bias is not None:
            out += bias

        qweight.generation += 1
        return out

    def _apply_4bit_weight(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_type = x.dtype
        original_shape = x.shape
        reshape_after_matmul = False
        if x.ndim > 2:
            x = x.reshape(-1, x.size(-1))
            reshape_after_matmul = True
        bf_x = x.to(torch.bfloat16)

        qweight = layer.weight
        quant_states = qweight.bnb_quant_state
        offsets = qweight.bnb_shard_offsets

        out_dim_0 = x.shape[0]
        out_dim_1 = sum(quant_state[1].shape[0] for quant_state in quant_states.items())
        out = torch.empty(out_dim_0, out_dim_1, dtype=torch.bfloat16, device=x.device)
        apply_bnb_4bit(bf_x, qweight, offsets, out)
        out = out.to(original_type)

        if reshape_after_matmul:
            out = out.view(*original_shape[:-1], out.size(-1))
        if bias is not None:
            out += bias

        return out

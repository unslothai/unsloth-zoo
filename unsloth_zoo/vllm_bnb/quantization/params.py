# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored by Unsloth from vllm-project/vllm-bnb-plugin v0.0.3 (commit d4914a9), Apache-2.0 (LICENSE in this
# directory). Changes: relocated under unsloth_zoo.vllm_bnb; see unsloth_zoo/vllm_bnb/__init__.py for the list.

import torch
from vllm.model_executor.parameter import ModelWeightParameter, PackedvLLMParameter


class BitsAndBytes8bitParameter(ModelWeightParameter):
    """BNB 8-bit parameters loaded through vLLM's weight_loader_v2 API."""

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor) -> None:
        self._assert_and_load(loaded_weight)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor) -> None:
        self._assert_and_load(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs) -> None:
        shard_offset = kwargs["shard_offset"]
        shard_size = kwargs["shard_size"]

        param_data = self.data.narrow(self.output_dim, shard_offset, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs) -> None:
        shard_offset = kwargs["shard_offset"]
        shard_size = kwargs["shard_size"]

        param_data = self.data.narrow(self.output_dim, shard_offset, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class BitsAndBytes4bitParameter(PackedvLLMParameter):
    """BNB 4-bit packed parameters loaded through vLLM's weight_loader_v2 API."""

    def __init__(self, logical_width: int, **kwargs):
        self.logical_width = logical_width
        super().__init__(**kwargs)

    def adjust_shard_indexes_for_packing(
        self, shard_size: int, shard_offset: int
    ) -> tuple[int, int]:
        quantized_total = self.data.shape[self.packed_dim]
        quantized_offset = shard_offset * quantized_total // self.logical_width
        quantized_size = shard_size * quantized_total // self.logical_width
        return quantized_size, quantized_offset

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor) -> None:
        self._assert_and_load(loaded_weight)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor) -> None:
        self._assert_and_load(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs) -> None:
        shard_offset = kwargs["shard_offset"]
        shard_size = kwargs["shard_size"]

        shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
            shard_size=shard_size,
            shard_offset=shard_offset,
        )
        param_data = self.data.narrow(self.output_dim, shard_offset, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs) -> None:
        shard_offset = kwargs["shard_offset"]
        shard_size = kwargs["shard_size"]

        shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
            shard_size=shard_size,
            shard_offset=shard_offset,
        )
        param_data = self.data.narrow(self.output_dim, shard_offset, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

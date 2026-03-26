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
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from .common import (
    TEMPORARY_PATCHES,
    UNSLOTH_ENABLE_LOGGING,
)
from .utils import (
    patch_function,
    logger,
)
import torch


def patch_qwen3_5():
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    except Exception:
        return

    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        can_compute_mrope = input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None)

        if can_compute_mrope and (self.rope_deltas is None or past_key_values_length == 0):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        elif self.rope_deltas is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
                position_ids = position_ids.view(1, batch_size, -1).repeat(3, 1, 1).to(inputs_embeds.device)
            else:
                position_ids = torch.arange(past_key_values_length, past_key_values_length + seq_length)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1).to(inputs_embeds.device)

            delta = self.rope_deltas
            if delta.shape[0] != batch_size:
                # GRPO can re-forward smaller text-only chunks than the prior
                # generation batch. Reusing the older rope_deltas tensor here can
                # create an empty batch via repeat_interleave(0), which then makes
                # Qwen3.5 RoPE produce batch-0 cos/sin while hidden_states stay
                # batch-1. Text-only calls should use zero deltas for the current
                # batch, exactly like the generation setup path does.
                if not can_compute_mrope:
                    delta = torch.zeros(batch_size, 1, dtype=position_ids.dtype, device=position_ids.device)
                    self.rope_deltas = delta
                elif delta.shape[0] == 1:
                    delta = delta.expand(batch_size, -1)
                elif batch_size % delta.shape[0] == 0:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                else:
                    delta = delta[:1].expand(batch_size, -1)
            position_ids = position_ids + delta.to(device=position_ids.device)
        else:
            # Can't build correct 3D positions. Let the model infer it from `cache_position`
            position_ids = None
        return position_ids

    patched = patch_function(
        Qwen3_5Model,
        "compute_3d_position_ids",
        compute_3d_position_ids,
    )
    if (not patched) and UNSLOTH_ENABLE_LOGGING:
        logger.warning("Unsloth: Could not patch Qwen3_5Model.compute_3d_position_ids.")


TEMPORARY_PATCHES.append(patch_qwen3_5)

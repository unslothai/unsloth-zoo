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

import torch
import torch.nn as nn
from typing import Optional
from .common import TEMPORARY_PATCHES, torch_compile_options
from .utils import (
    patch_function,
    raise_error,
)

def patch_FalconH1Mixer_torch_forward():
    try:
        import transformers.models.falcon_h1.modeling_falcon_h1
        from transformers.models.falcon_h1.modeling_falcon_h1 import (
            FalconHybridMambaAttentionDynamicCache,
            apply_mask_to_padding_states,
            pad_tensor_by_size,
            reshape_into_chunks,
            segment_sum,
        )
    except Exception as e:
        return raise_error("FalconH1Mixer.torch_forward", e)

    # this patch is largely adapted from the original torch_forward implementation
    # The idea is to make the training portion faster by torch compiling the heavy parts
    def _get_data_hidden_states_dt(self, input_states):
        input_states = input_states * self.ssm_in_multiplier
        projected_states = self.in_proj(input_states)
        projected_states = projected_states * self.mup_vector
        gate, hidden_states_B_C, dt = projected_states.split([
                self.intermediate_size, self.conv_dim, self.num_heads
            ], dim=-1)
        return gate, hidden_states_B_C, dt

    def _conv1d(self, hidden_states_B_C, seq_len):
        hidden_states_B_C = self.act(self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2))
        return hidden_states_B_C

    def _kern_dt_and_A_and_hs(self, dt, A_log, hs, time_lim):
        dt = torch.nn.functional.softplus(dt + self.dt_bias)
        dt = torch.clamp(dt, time_lim[0], time_lim[1])
        hs = hs * dt[..., None]
        A  = -torch.exp(A_log).to(torch.float32) * dt
        return dt, A, hs

    def _kern_intra_chunk(self, hs, B, C, A, dt):
        A = A.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A, dim=-1)

        L = torch.exp(segment_sum(A))

        G = (C[:, :, :, None, :, :] * B[:, :, None, :, :, :]).sum(dim=-1)
        M = (G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]).sum(dim=-1)
        Y_diag = (M[..., None] * hs[:, :, None]).sum(dim=3)

        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
        states = (B_decay[..., None, :] * hs[..., None]).sum(dim=2)

        return Y_diag, states, A_cumsum

    def _kern_inter_chunk(
        self,
        states,
        A_cumsum,
        padded_A_cumsum,
        C_chunks,
    ):
        decay_chunk = torch.exp(padded_A_cumsum).transpose(1, 3)
        new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
        states, ssm_state = new_states[:, :-1], new_states[:, -1]

        state_decay_out = torch.exp(A_cumsum)
        C_times_states = (C_chunks[..., None, :] *
                          states[:, :, None, ...])
        Y_off = (C_times_states.sum(-1) *
                 state_decay_out.permute(0, 2, 3, 1)[..., None])
        return Y_off, ssm_state

    _get_data_hidden_states_dt = torch.compile(_get_data_hidden_states_dt, fullgraph = True, dynamic = True, options = torch_compile_options)
    _conv1d                    = torch.compile(_conv1d, fullgraph = True, dynamic = True, options = torch_compile_options)
    _kern_dt_and_A_and_hs      = torch.compile(_kern_dt_and_A_and_hs, fullgraph = True, dynamic = True, options = torch_compile_options)
    _kern_intra_chunk          = torch.compile(_kern_intra_chunk, fullgraph = True, dynamic = True, options = torch_compile_options)
    _kern_inter_chunk          = torch.compile(_kern_inter_chunk, fullgraph = False, dynamic = True)

    def torch_forward(
        self,
        input_states,
        cache_params: Optional[FalconHybridMambaAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        input_states = apply_mask_to_padding_states(input_states, attention_mask)
        gate, hidden_states_B_C, dt = _get_data_hidden_states_dt(self, input_states)

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_params.conv_states[self.layer_idx].shape[0]
            == cache_params.ssm_states[self.layer_idx].shape[0]
            == batch_size
            and cache_position is not None
            and cache_position[0] > 0
        )

        if use_precomputed_states:
            cache_params.conv_states[self.layer_idx] = cache_params.conv_states[self.layer_idx].roll(shifts=-1, dims=-1)
            cache_params.conv_states[self.layer_idx][:, :, -1] = hidden_states_B_C[:, 0, :].to(cache_params.conv_states[self.layer_idx].device)

            conv_states = cache_params.conv_states[self.layer_idx].to(device=self.conv1d.weight.device)

            hidden_states_B_C = torch.sum(
                conv_states * self.conv1d.weight.squeeze(1), dim=-1
            )
            if self.use_conv_bias:
                hidden_states_B_C = hidden_states_B_C + self.conv1d.bias
            hidden_states_B_C = self.act(hidden_states_B_C)
        else:
            # Init cache
            if cache_params is not None:
                hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
                conv_states = nn.functional.pad(
                    hidden_states_B_C_transposed, (self.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx].copy_(conv_states)

            hidden_states_B_C = _conv1d(self, hidden_states_B_C, seq_len)

        hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size],
            dim=-1
        )

        if use_precomputed_states:
            A = -torch.exp(self.A_log.float())
            cache_device = cache_params.ssm_states[self.layer_idx].device

            dt = dt[:, 0, :][:, None, ...]
            dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
            dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)

            dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            A = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dA = (torch.exp(dt[..., None] * A)).to(device=cache_device)

            B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
            B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
            B = B.reshape(batch_size, -1, B.shape[-1])
            dB = dt[..., None] * B[..., None, :]

            hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
            dBx = (dB * hidden_states[..., None]).to(device=cache_device)

            cache_params.ssm_states[self.layer_idx].copy_(
                cache_params.ssm_states[self.layer_idx] * dA + dBx
            )

            C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
            C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
            C = C.reshape(batch_size, -1, C.shape[-1])

            ssm_states = cache_params.ssm_states[self.layer_idx].to(device=C.device, dtype=C.dtype)  # Shape: [b, h, d, n]
            ssm_states_reshaped = ssm_states.view(batch_size * self.num_heads, self.head_dim, self.ssm_state_size)  # Shape: [b*h, d, n]
            C_reshaped = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)  # Shape: [b*h, n, 1]
            y = torch.bmm(ssm_states_reshaped, C_reshaped)
            y = y.view(batch_size, self.num_heads, self.head_dim)

            D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
            y = (y + hidden_states * D).to(y.dtype)

            y = y.reshape(batch_size, -1)[:, None, ...]
        else:
            H, D        = self.num_heads, self.head_dim
            S           = self.ssm_state_size
            pad_size    = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

            hs = hidden_states.view(batch_size, seq_len, H, D).float()
            B  = B.view(batch_size, seq_len, self.n_groups, S).float()
            C  = C.view(batch_size, seq_len, self.n_groups, S).float()
            A_log = self.A_log.float()
            heads_per_group = H // self.n_groups
            B = B.repeat_interleave(heads_per_group, dim=2, output_size=H)
            C = C.repeat_interleave(heads_per_group, dim=2, output_size=H)

            D_residual = self.D[..., None] * pad_tensor_by_size(hs, pad_size)

            dt_scaled, A, hs = _kern_dt_and_A_and_hs(
                self, dt, A_log, hs, self.time_step_limit
            )

            hs, A, B, C = [reshape_into_chunks(t, pad_size, self.chunk_size)
                                for t in (hs, A, B, C)]

            Y_diag, states_chunks, A_cumsum = _kern_intra_chunk(
                self, hs, B, C, A, dt_scaled
            )

            # if use_precomputed_states:
            #     prev_states = cache_params.ssm_states[self.layer_idx][:, None, ...].to(device=hidden_states.device)
            # else:
            prev_states = torch.zeros_like(states_chunks[:, :1])
            states_chunks = torch.cat([prev_states, states_chunks], dim=1)  # prepend
            padded_A_cumsum = torch.nn.functional.pad(A_cumsum[:, :, :, -1], (1,0))
            padded_A_cumsum = segment_sum(padded_A_cumsum)

            Y_off, ssm_state = _kern_inter_chunk(
                self, states_chunks, A_cumsum, padded_A_cumsum, C
            )

            y = Y_diag + Y_off
            y = y.view(batch_size, -1, H, D)
            y = y + D_residual
            if pad_size:
                y = y[:, :seq_len]
            y = y.view(batch_size, seq_len, -1)

            if ssm_state is not None and cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        if self.mamba_rms_norm:
            scan_output = self.norm(y, gate)
        else:
            scan_output = y * torch.nn.functional.silu(gate)


        contextualized_states = self.out_proj(scan_output.to(dtype))  # [batch, seq_len, hidden_size]
        return contextualized_states


    # only patch if bf16 is not supported
    major_version, minor_version = torch.cuda.get_device_capability()
    SUPPORTS_BFLOAT16 = (major_version >= 8)
    if not SUPPORTS_BFLOAT16:
        return patch_function(
            transformers.models.falcon_h1.modeling_falcon_h1.FalconH1Mixer, "torch_forward", torch_forward,
        )
    else:
        return True  # return True if bf16 is not supported since we don't need to patch
pass

TEMPORARY_PATCHES.append(patch_FalconH1Mixer_torch_forward)

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

import os
import math
from collections import OrderedDict
from types import MethodType
import functools
import torch
from torch.utils.checkpoint import (
    _infer_device_type,
    _get_device_module,
    get_device_states,
)
from unsloth_zoo.gradient_checkpointing import set_device_states
from unsloth_zoo.device_type import DEVICE_TYPE

__all__ = [
    "patch_tiled_mlp",
    "patch_mlp",
]

FIRST_PASS = True
UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"
UNSLOTH_ENABLE_TILED_LOGGING = UNSLOTH_ENABLE_LOGGING and os.environ.get("UNSLOTH_ENABLE_TILED_LOGGING", "0") == "1"

torch_amp_custom_fwd = torch.amp.custom_fwd(device_type = DEVICE_TYPE)
torch_amp_custom_bwd = torch.amp.custom_bwd(device_type = DEVICE_TYPE)

@functools.cache
def get_max_flat_qlen(
    hd = 4096,
    mlp_size = 14336,
    nbytes = 2,
    target_gb = 0.5,
    padded_length = 64,
):
    # flat_qlen = bsz*qlen
    # flat_qlen = torch.arange(0, 512*1024, 1024)
    # forward_memory_usage  = 3*flat_qlen*mlp_size + flat_qlen*hd
    # backward_memory_usage = forward_memory_usage + 3*hd*mlp_size
    # saved_tensors = 10*flat_qlen*hd # 2 norms, 2 residual, 4 QKVO, 2 attention
    # total_memory_usage = saved_tensors + backward_memory_usage
    # total_memory_usage = total_memory_usage * nbytes / 1024 / 1024 / 1024
    numerator = target_gb * 1024 * 1024 * 1024 / nbytes - (3*hd*mlp_size)
    denominator = (10*hd + 3*mlp_size + hd)
    max_flat_qlen = math.ceil(numerator / denominator)
    max_flat_qlen = max(padded_length, (max_flat_qlen // padded_length) * padded_length)
    return max_flat_qlen
pass

class TiledMLP(torch.autograd.Function):
    @staticmethod
    def handle_output(output, extra_lists):
        """Extract main output and append extras to their lists"""
        if isinstance(output, tuple):
            # Initialize lists on first tuple
            if not extra_lists:
                for _ in output[1:]:
                    extra_lists.append([])
            # Append extras
            for i, extra in enumerate(output[1:]):
                extra_lists[i].append(extra)
            return output[0]  # Return main output
        return output  # Single tensor

    @staticmethod
    def structure_output(main_output, extra_lists):
        """Reconstruct original structure"""
        if not extra_lists:
            return main_output
        # Cat extras along seq dim and return tuple
        extras = [torch.cat(extra_list, dim=-2) for extra_list in extra_lists]
        return (main_output, *extras)

    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, mlp_forward, mlp_module, x, preserve_rng_state, num_shards, max_flat_qlen):
        # num_shards is probably the wrong name and it should be n_splits
        # num_shards is also not guaranteed. It could end up having num_shards + 1
        # the main thing is the shard seq length is all the same unless it's not
        # evenly divisible with sequence length. Then the last shard will have the remainder.
        ctx.shard_dim = -2
        B, S, H = x.shape
        # ctx.num_shards = int(max(1, min(S, math.ceil(S / max(1, H)))))
        ctx.num_shards = num_shards
        if max_flat_qlen:
            qlen_chunk_size = min(max_flat_qlen, B*S)
            remainder = max(0, B*S - qlen_chunk_size * num_shards)
        else:
            qlen_chunk_size, remainder = divmod(B*S, min(max(1, num_shards), B*S))
        split_sizes = [qlen_chunk_size]*num_shards
        if remainder != 0: split_sizes.append(remainder)
        ctx.split_sizes = split_sizes
        global FIRST_PASS
        if (FIRST_PASS and UNSLOTH_ENABLE_LOGGING) or UNSLOTH_ENABLE_TILED_LOGGING:
            print(f"Unsloth: Enabling TiledMLP to reduce VRAM usage! chunk size: {split_sizes[0]}")
            FIRST_PASS = False

        ctx.device_type = _infer_device_type(x)
        ctx.mlp_forward = mlp_forward
        ctx.mlp_module = mlp_module

        if preserve_rng_state is None:
            preserve_rng_state = True
        ctx.preserve_rng_state = bool(preserve_rng_state)

        # Save tensors needed in backward
        ctx.save_for_backward(x)

        # RNG state capture if requested
        if ctx.preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device_type)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(x)

        # with device?
        # preserve rng_state should be false if not dropout in mlp
        start_idx = 0
        final_output = None
        extra_outputs = []
        x = x.view(-1, H)
        with torch.no_grad():
            x_splits = torch.split(x, ctx.split_sizes, dim=0)
            for i, x_split in enumerate(x_splits):
                x_split = x_split.unsqueeze(0)
                out = TiledMLP.handle_output(mlp_forward(x_split), extra_outputs)

                if final_output is None:
                    final_output = torch.empty(B, S, H, device=out.device, dtype=out.dtype)
                split_size = x_split.numel()
                final_output.view(-1).narrow(
                    dim=0,
                    start=start_idx,
                    length=split_size,
                ).view_as(x_split).copy_(out)
                start_idx += split_size
        return TiledMLP.structure_output(final_output, extra_outputs)

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, grad_output, *args):
        rng_devices = []
        x = ctx.saved_tensors[0]
        B, S, H = x.shape
        if ctx.preserve_rng_state and ctx.had_device_in_fwd:
            rng_devices = ctx.fwd_devices
        with torch.random.fork_rng(
            devices=rng_devices, enabled=ctx.preserve_rng_state, device_type=ctx.device_type
        ):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    set_device_states(ctx.fwd_devices, ctx.fwd_device_states, device_type=ctx.device_type)

            x_gradients = torch.zeros_like(x, memory_format=torch.preserve_format)
            x = x.view(-1, H)
            # chunk on seq length, assume index before last
            x_splits = torch.split(x, ctx.split_sizes, dim=0)
            start_idx = 0
            extra_outputs = []
            for i, x_split in enumerate(x_splits):
                x_split = x_split.unsqueeze(0)
                split_size = x_split.numel()
                x_grad_slice = x_gradients.view(-1).narrow(
                    dim=0,
                    start=start_idx,
                    length=split_size,
                ).view_as(x_split)

                grad_output_shard = grad_output.view(-1).narrow(
                    dim=0,
                    start=start_idx,
                    length=split_size,
                ).view_as(x_split)

                x_split.requires_grad_(True)
                x_split.grad = x_grad_slice
                with torch.enable_grad():
                    outputs = TiledMLP.handle_output(ctx.mlp_forward(x_split), extra_outputs)

                torch.autograd.backward(outputs, grad_output_shard)
                start_idx += split_size

        return None, None, x_gradients, None, None, None

def patch_mlp(mlp_module, target_arctic = True, target_gb = None, padded_length = 128):
    preserve_rng_state = False
    for n, m in mlp_module.named_modules():
        if isinstance(m, torch.nn.Dropout):
            preserve_rng_state = True
            break

    # unbound
    mlp_module._original_forward = mlp_module.__class__.forward
    # second is what llama style patch uses
    mlp_module._unsloth_forward = mlp_module.__class__.forward


    def tiled_forward_target_gb(self, x):
        nonlocal target_gb
        bsz, qlen, hd = x.shape
        flat_qlen = bsz*qlen
        try:
            intermediate_size = mlp_module.config.intermediate_size
            if isinstance(intermediate_size, (list, tuple)):
                intermediate_size = intermediate_size[0]
        except:
            intermediate_size = hd * 4

        if target_gb is None:
            free, total = torch.cuda.mem_get_info(0)
            free_gb = free / 1024 / 1024 / 1024
            free_gb = free_gb * 0.5
            target_gb = free_gb

        max_flat_qlen = get_max_flat_qlen(
            hd = hd,
            mlp_size = intermediate_size,
            nbytes = x.element_size(),
            target_gb = target_gb,
            padded_length = padded_length,
        )
        n_shards, remainder = divmod(flat_qlen, max_flat_qlen)
        n_shards = max(1, n_shards)

        # this call binds
        inner_forward = self._unsloth_forward.__get__(self, self.__class__)
        return TiledMLP.apply(inner_forward, mlp_module, x, preserve_rng_state, n_shards, max_flat_qlen)

    def tiled_forward_arctic_size(self, x):
        B, S, H = x.shape
        chunk_size = max(1, H)
        n_shards, remainder = divmod(S, chunk_size)
        n_shards = max(1, n_shards)
        # remainder gets added to the last shard in the forward pass

        # this call binds
        inner_forward = self._unsloth_forward.__get__(self, self.__class__)
        return TiledMLP.apply(inner_forward, mlp_module, x, preserve_rng_state, n_shards, chunk_size)

    if target_arctic:
        mlp_module.forward = MethodType(tiled_forward_arctic_size, mlp_module)
    else:
        mlp_module.forward = MethodType(tiled_forward_target_gb, mlp_module)
    return mlp_module

def patch_tiled_mlp(model, patch_options_str = "arctic", padded_length = 128):
    patch_options_strs = patch_options_str.split(":")
    if patch_options_strs[0] in ["arctic", "1"]:
        target_arctic = True
    else:
        target_arctic = False
    if len(patch_options_strs) > 1:
        try:
            target_gb = float(patch_options_strs[-1])
        except:
            target_gb = None
    else:
        target_gb = None
    for name, module in model.named_modules():
        if name.lower().endswith(".mlp") or type(module).__name__.lower().endswith("mlp"):
            patch_mlp(module, target_arctic = target_arctic, target_gb = target_gb, padded_length = padded_length)
    return model

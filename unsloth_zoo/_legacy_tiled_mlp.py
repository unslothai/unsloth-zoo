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

"""The previous tiled MLP autograd Function, kept importable for released Unsloth versions that import it.

Nothing in Unsloth Zoo uses it; `unsloth_zoo.tiling_registry` replaces it."""

import os
import sys
from fnmatch import fnmatch
import torch
from torch.utils.checkpoint import (
    _infer_device_type,
    _get_device_module,
    get_device_states,
)
from unsloth_zoo.gradient_checkpointing import set_device_states
from unsloth_zoo.device_type import DEVICE_TYPE_TORCH


__all__ = ["TiledMLP", "torch_amp_custom_fwd", "torch_amp_custom_bwd", "is_custom_module"]

FIRST_PASS = True
UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"
UNSLOTH_ENABLE_TILED_LOGGING = UNSLOTH_ENABLE_LOGGING and os.environ.get("UNSLOTH_ENABLE_TILED_LOGGING", "0") == "1"

# amp rejects DEVICE_TYPE "hip" / "mlx"; torch < 2.5 also lacks mps autocast, hence the cpu fallback.
def _amp_device_type():
    try:
        torch.get_autocast_dtype(DEVICE_TYPE_TORCH)
        return DEVICE_TYPE_TORCH
    except Exception:
        return "cpu"
pass
_AMP_DEVICE_TYPE = _amp_device_type()
torch_amp_custom_fwd = torch.amp.custom_fwd(device_type = _AMP_DEVICE_TYPE)
torch_amp_custom_bwd = torch.amp.custom_bwd(device_type = _AMP_DEVICE_TYPE)


class TiledMLP(torch.autograd.Function):
    @staticmethod
    def handle_output(output, extra_lists):
        """Extract main output, append extras to their lists."""
        if isinstance(output, tuple):
            if not extra_lists:
                for _ in output[1:]:
                    extra_lists.append([])
            for i, extra in enumerate(output[1:]):
                extra_lists[i].append(extra)
            return output[0]
        return output

    @staticmethod
    def structure_output(main_output, extra_lists):
        """Reconstruct original structure: cat extras along seq dim."""
        if not extra_lists:
            return main_output
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

        ctx.save_for_backward(x)

        if ctx.preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device_type)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(x)

        # preserve_rng_state should be False without MLP dropout; the with-device question is open.
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
        # x_split is a no_grad view and the fused LoRA MLP writes dX into it in place; if x requires grad
        # (a compiled block's output), AccumulateGrad raises "A view was created in no_grad mode".
        x = ctx.saved_tensors[0].detach()
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

def _mxfp4_instance_forward(mlp_module):
    # transformers binds mxfp4.mlp_forward on each MXFP4 GptOssMLP; the class forward skips it (unsloth-zoo#385).
    mxfp4 = sys.modules.get("transformers.integrations.mxfp4")
    mlp_forward = getattr(mxfp4, "mlp_forward", None)
    if mlp_forward is None:
        return None
    for name in ("forward", "_original_forward"):
        bound = mlp_module.__dict__.get(name)
        func = getattr(bound, "__func__", bound)
        if func is mlp_forward:
            return func
    return None




def is_custom_module(name, custom_modules):
    name_lower = name.lower()
    for custom_module in custom_modules:
        custom_module_lower = custom_module.lower()
        if '*' in custom_module or '?' in custom_module:
            if fnmatch(name_lower, custom_module_lower):
                return True
        else:
            if name_lower.endswith(custom_module_lower):
                return True
    return False


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

"""flex_attention backward shared-mem OOM: adds configs only when over limit; no-op on torch < 2.10."""

import torch

__all__ = ["patch_flex_attention_bwd_configs"]

# Fixed Triton runtime and buffer overhead; 9728 bytes measured as 140800 actual minus 131072 base.
_SHMEM_OVERHEAD = 10240  # 10KB, slightly conservative


# Linear term measured on GB10 (sm_121) against Triton's Required; stage-independent there.
_SHMEM_TILE_FACTOR = 3.5


def _dtype_size(dtype):
    if dtype in (torch.bfloat16, torch.float16):
        return 2
    # getattr, because the float8 dtypes do not exist on every supported torch.
    if dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
        return 1
    return 4


def _staged_dtype_size(dtype):
    """Returns 4 for non-bf16/fp16 dtypes (float8 too); kept as a floor that only raises the estimate."""
    return 2 if dtype in (torch.bfloat16, torch.float16) else 4


def _estimate_shmem(block_size, num_stages, head_dim, dtype):
    """Max of staged/linear shmem formulas; biased high: an underestimate returns an uncompilable config."""
    tile = block_size * head_dim
    staged = num_stages * tile * _staged_dtype_size(dtype) * 2 + _SHMEM_OVERHEAD
    linear = _SHMEM_TILE_FACTOR * tile * _dtype_size(dtype)
    return int(max(staged, linear))


def _shmem_limit(device):
    """Min of per-block opt-in and per-SM smem, since Triton enforces opt-in; tolerates either attr absent"""
    props = torch.cuda.get_device_properties(device)
    limits = [
        getattr(props, name, None)
        for name in ("shared_memory_per_block_optin", "shared_memory_per_multiprocessor")
    ]
    limits = [x for x in limits if isinstance(x, int) and x > 0]
    return min(limits) if limits else 0


def _generate_safe_configs(FlexBwDConfig, shmem_limit, head_dim, dtype, worst_block = 0):
    """Offer all smaller blocks, not just estimate-approved ones; Inductor drops configs that don't fit."""
    configs = []
    block = worst_block if worst_block else 256
    while block > 16:
        block //= 2
        for stages in [2, 1]:
            for warps in [4, 8]:
                configs.append(FlexBwDConfig(block, block, block, block, stages, warps))
    # Always include the smallest safe fallback
    fallback = FlexBwDConfig(16, 16, 16, 16, 1, 4)
    if fallback not in configs:
        configs.append(fallback)
    return configs


def patch_flex_attention_bwd_configs():
    """Patch flex_attention bwd config selection against shared memory OOM; returns True if applied."""
    try:
        from torch._inductor.template_heuristics.triton import FlexBwDConfig
    except ImportError:
        return False  # torch < 2.10 -- FlexBwDConfig does not exist

    if not torch.cuda.is_available():
        return False

    import torch._inductor.template_heuristics.triton as triton_heuristics

    # Find all non-ROCm heuristic classes with get_flex_attn_bwd_configs
    classes_to_patch = []
    for name in dir(triton_heuristics):
        obj = getattr(triton_heuristics, name)
        if isinstance(obj, type) and hasattr(obj, "get_flex_attn_bwd_configs"):
            if "ROCm" not in name:
                classes_to_patch.append((name, obj))

    if not classes_to_patch:
        return False

    for _cls_name, cls in classes_to_patch:
        original_method = cls.get_flex_attn_bwd_configs

        def make_patched(orig):
            def patched_get_flex_attn_bwd_configs(self, head_dim, dtype):
                # Per-device limit so heterogeneous multi-GPU setups work.
                try:
                    shmem_limit = _shmem_limit(torch.cuda.current_device())
                except Exception:
                    shmem_limit = 0

                try:
                    configs = list(orig(self, head_dim, dtype))
                except Exception:
                    configs = []

                if not shmem_limit:
                    return configs

                # Patch only if some original config exceeds shared memory.
                needs_patch = False
                worst_block = 0
                for c in configs:
                    max_block = max(c.block_m1, c.block_n1, c.block_m2, c.block_n2)
                    est = _estimate_shmem(max_block, c.num_stages, head_dim, dtype)
                    if est > shmem_limit:
                        needs_patch = True
                        worst_block = max(worst_block, max_block)

                if not needs_patch and configs:
                    return configs

                # Add safe fallback configs for this GPU / head_dim / dtype
                safe = _generate_safe_configs(
                    FlexBwDConfig, shmem_limit, head_dim, dtype, worst_block,
                )
                for c in safe:
                    if c not in configs:
                        configs.append(c)

                return configs
            patched_get_flex_attn_bwd_configs._unsloth_flex_bwd = True
            return patched_get_flex_attn_bwd_configs

        # Skip already-patched methods; a second wrap would make every later call walk the logic twice.
        if getattr(original_method, "_unsloth_flex_bwd", False):
            continue
        cls.get_flex_attn_bwd_configs = make_patched(original_method)

    return True


# Apply the patch at import time so it is active before any compilation.
try:
    patch_flex_attention_bwd_configs()
except Exception:
    pass

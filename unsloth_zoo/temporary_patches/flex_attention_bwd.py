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

"""
Patch for flex_attention backward kernel shared memory OOM on constrained GPUs.

On GPUs with limited shared memory per SM (RTX 3090/4090/5090, RTX PRO 6000,
etc. -- typically ~100KB), the flex_attention backward kernel configs can exceed
shared memory limits for large head_dims (e.g., 256 in Gemma3). For example,
the default config (BLOCK=64, num_stages=2) requires ~141KB, but the hardware
only has ~101KB.

This patch intercepts the backward config selection and adds safe fallback
configs estimated to fit within the GPU's actual shared memory limit. The
shared memory estimation uses:
    shmem ~ num_stages * max_block * head_dim * dtype_size * 2 + OVERHEAD
where OVERHEAD is ~10KB from Triton runtime. This formula was validated
against real crashes (estimated 141,312 vs actual 140,800 bytes).

If the default configs already fit in shared memory, they are returned
unchanged. Safe fallbacks are only added when a config is estimated to exceed
the limit.

This is a no-op on torch versions that do not have FlexBwDConfig (< 2.10).
"""

import torch

__all__ = ["patch_flex_attention_bwd_configs"]

# Estimated fixed overhead from Triton runtime + small kernel buffers.
# Derived empirically: actual_shmem(140800) - formula_base(131072) = 9728 bytes
_SHMEM_OVERHEAD = 10240  # 10KB, slightly conservative


def _estimate_shmem(block_size, num_stages, head_dim, dtype):
    """Estimate shared memory for a flex_attention backward config.

    The backward kernel pipelines two tensor loads (qT + dO or kT + vT)
    with num_stages copies each in shared memory.
    """
    dtype_size = 2 if dtype in (torch.bfloat16, torch.float16) else 4
    return num_stages * block_size * head_dim * dtype_size * 2 + _SHMEM_OVERHEAD


def _generate_safe_configs(FlexBwDConfig, shmem_limit, head_dim, dtype):
    """Generate FlexBwDConfig candidates estimated to fit within shared memory."""
    configs = []
    for block in [128, 64, 32, 16]:
        for stages in [3, 2, 1]:
            est = _estimate_shmem(block, stages, head_dim, dtype)
            if est <= shmem_limit:
                for warps in [4, 8]:
                    configs.append(FlexBwDConfig(block, block, block, block, stages, warps))
    # Always include the smallest safe fallback
    fallback = FlexBwDConfig(16, 16, 16, 16, 1, 4)
    if fallback not in configs:
        configs.append(fallback)
    return configs


def patch_flex_attention_bwd_configs():
    """Patch flex_attention backward config selection to avoid shared memory OOM.

    Returns True if the patch was applied, False otherwise.
    """
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
                # Query shared memory limit from the current device so this
                # works correctly in heterogeneous multi-GPU setups.
                device = torch.cuda.current_device()
                shmem_limit = torch.cuda.get_device_properties(device).shared_memory_per_multiprocessor

                # Get original configs first
                try:
                    configs = list(orig(self, head_dim, dtype))
                except Exception:
                    configs = []

                # Check if any original config exceeds shared memory
                needs_patch = False
                for c in configs:
                    max_block = max(c.block_m1, c.block_n1, c.block_m2, c.block_n2)
                    est = _estimate_shmem(max_block, c.num_stages, head_dim, dtype)
                    if est > shmem_limit:
                        needs_patch = True
                        break

                if not needs_patch and configs:
                    return configs

                # Add safe fallback configs for this GPU / head_dim / dtype
                safe = _generate_safe_configs(FlexBwDConfig, shmem_limit, head_dim, dtype)
                for c in safe:
                    if c not in configs:
                        configs.append(c)

                return configs
            return patched_get_flex_attn_bwd_configs

        cls.get_flex_attn_bwd_configs = make_patched(original_method)

    return True


# Apply the patch at import time so it is active before any compilation.
try:
    patch_flex_attention_bwd_configs()
except Exception:
    pass

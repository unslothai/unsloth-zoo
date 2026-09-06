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
Patch flex_attention backward shared-memory OOM on constrained GPUs.

On GPUs with limited shared memory per SM (RTX 3090/4090/5090, RTX PRO 6000,
~100KB), the backward configs can exceed the limit for large head_dims (e.g.
256 in Gemma3): the default BLOCK=64, num_stages=2 needs ~141KB vs ~101KB.

We intercept backward config selection and add safe fallbacks. Fallbacks are only
added when a config is estimated to exceed the limit, so a GPU with room to spare
keeps exactly the configs upstream chose and pays no extra autotune time.

The estimate is the maximum of two formulas, because each was fitted on a different
GPU and neither generalises on its own:
    staged ~ num_stages * max_block * head_dim * dtype_size * 2 + OVERHEAD
    linear ~ 3.5 * max_block * head_dim * dtype_size
The first was validated on a ~100KB-per-SM sm8x part (141,312 estimated vs 140,800
actual, BLOCK=64/num_stages=2/head_dim=256). The second was measured on GB10
(sm_121, torch 2.14, Triton 3.8) by forcing single configs and reading Triton's own
"Required:": BLOCK 64 -> 114,688 and BLOCK 128 -> 229,376, IDENTICAL for num_stages
1 and 2. So the requirement is linear in block size and, on that part, independent
of num_stages.

Taking the maximum matters in both directions. Under-estimating is fatal and silent:
the gate does not fire, the offending config is returned untouched, and the compile
dies with "No valid triton configs". Over-estimating only offers configs that
Inductor discards. So the estimate is deliberately biased high, and it is never
lower than the original formula, which means every GPU where this patch fires today
still gets it.

No-op on torch < 2.10 (no FlexBwDConfig).
"""

import torch

__all__ = ["patch_flex_attention_bwd_configs"]

# Estimated fixed overhead from Triton runtime + small kernel buffers.
# Derived empirically: actual_shmem(140800) - formula_base(131072) = 9728 bytes
_SHMEM_OVERHEAD = 10240  # 10KB, slightly conservative


# Measured on GB10 (sm_121) against Triton's reported "Required:", and stage-independent
# there. See the module docstring for why both this and the staged formula are kept.
_SHMEM_TILE_FACTOR = 3.5


def _dtype_size(dtype):
    if dtype in (torch.bfloat16, torch.float16):
        return 2
    if dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
        return 1
    return 4


def _estimate_shmem(block_size, num_stages, head_dim, dtype):
    """Estimate shared memory for a backward config, biased high on purpose.

    The kernel pipelines two tensor loads (qT+dO or kT+vT). How many copies it keeps
    live is not the same across architectures: on sm8x it scales with num_stages, and
    on sm_121 it measurably does not. Neither formula is safe alone, so take whichever
    is larger -- an over-estimate costs a few discarded autotune candidates, while an
    under-estimate silently returns a config that cannot compile.
    """
    tile = block_size * head_dim * _dtype_size(dtype)
    staged = num_stages * tile * 2 + _SHMEM_OVERHEAD
    linear = _SHMEM_TILE_FACTOR * tile
    return int(max(staged, linear))


def _shmem_limit(device):
    """The limit Triton actually enforces.

    It refuses a kernel on the per-block opt-in maximum, which is SMALLER than the
    per-SM figure on recent parts (101,376 vs 102,400 on GB10) -- and the 1KB between
    them is enough to bless a config Triton then rejects. Take the smaller of the two,
    and tolerate either attribute being absent on older torch.
    """
    props = torch.cuda.get_device_properties(device)
    limits = [
        getattr(props, name, None)
        for name in ("shared_memory_per_block_optin", "shared_memory_per_multiprocessor")
    ]
    limits = [x for x in limits if isinstance(x, int) and x > 0]
    return min(limits) if limits else 0


def _generate_safe_configs(FlexBwDConfig, shmem_limit, head_dim, dtype, worst_block = 0):
    """Candidates to offer alongside the ones upstream chose.

    Every block strictly smaller than the offending one is offered, rather than only
    those the estimate blesses. The estimate decides WHETHER to intervene, but it is a
    model of somebody else's kernel and it should not also decide what survives:
    Inductor's autotuner catches OutOfResources per candidate and keeps the ones that
    compile, so offering a config that does not fit costs one discarded candidate,
    while failing to offer one that does is the whole bug.
    """
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

        # Calling this twice would otherwise wrap the wrapper, so every later call
        # would walk two copies of the same logic.
        if getattr(original_method, "_unsloth_flex_bwd", False):
            continue
        cls.get_flex_attn_bwd_configs = make_patched(original_method)

    return True


# Apply the patch at import time so it is active before any compilation.
try:
    patch_flex_attention_bwd_configs()
except Exception:
    pass

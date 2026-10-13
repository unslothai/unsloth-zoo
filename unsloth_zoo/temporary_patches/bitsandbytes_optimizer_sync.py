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

"""Skip bitsandbytes' per-parameter device sync in non-paged 8-bit / 32-bit optimizer steps.

`Optimizer8bit.step` calls `torch.cuda.synchronize()` (0.45 / 0.46) or `sync_gpu(p)`
(0.47+) after every parameter's update. It arrived with paged optimizers (commit 41a9c70,
"Changed prefetching.", May 2023) next to the per-parameter `prefetch_state`: paged state
lives in managed memory and `cprefetch` / `cudaMemPrefetchAsync` run on stream 0 without
ordering against the host. Non-paged optimizers never prefetch, and their update kernels
(`kOptimizer32bit*`, `kOptimizerStatic8bit*Blockwise`) are launched without a stream
argument, i.e. on the legacy default stream, which is also torch's default stream. When the
parameter's device runs torch on that default stream, stream order already serialises the
update after backward and before the next forward, and the caching allocator's reuse of
freed gradients is stream ordered too. The host-side sync then only blocks the CPU, which
costs one full device drain per parameter (192 per step for a 4-bit LoRA on Qwen3.5-2B).

Scope, deliberately narrow:
  * paged optimizers (`self.is_paged`) run bitsandbytes' own `step` untouched;
  * the sync is kept for non-CUDA params (XPU, CPU, ...) and whenever the param's device has
    a non-default current stream (side streams are non-blocking against the legacy stream);
  * only applied when the installed `Optimizer8bit.step` is AST-identical to one of the two
    known shapes, inside the version window below;
  * `UNSLOTH_BNB_OPTIMIZER_SYNC=1` restores the per-parameter sync (read every step).
"""

import ast
import inspect
import os
from textwrap import dedent

import torch

from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING
from .utils import logger, raise_error

__all__ = ["patch_bitsandbytes_optimizer_step_sync"]

_BNB_MIN_VERSION = "0.45.0"
_BNB_MAX_VERSION = "0.51.0"  # exclusive; bump after checking the new step shape
_KILL_SWITCH = "UNSLOTH_BNB_OPTIMIZER_SYNC"
_PATCHED_FLAG = "_unsloth_no_per_param_sync"

# bitsandbytes 0.45.0 - 0.46.x
_STEP_TORCH_SYNC = '''
@torch.no_grad()
def step(self, closure=None):
    """doc"""
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    if not self.initialized:
        self.check_overrides()
        self.to_gpu()  # needed for fairseq pure fp16 training
        self.initialized = True

    for gindex, group in enumerate(self.param_groups):
        for pindex, p in enumerate(group["params"]):
            if p.grad is None:
                continue
            state = self.state[p]
            if len(state) == 0:
                self.init_state(group, p, gindex, pindex)

            self.prefetch_state(p)
            self.update_step(group, p, gindex, pindex)
            torch.cuda.synchronize()
    if self.is_paged:
        torch.cuda.synchronize()

    return loss
'''

# bitsandbytes 0.47.0 - 0.50.x
_STEP_SYNC_GPU = '''
@torch.no_grad()
def step(self, closure=None):
    """doc"""
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    if not self.initialized:
        self.check_overrides()
        self.to_gpu()  # needed for fairseq pure fp16 training
        self.initialized = True

    p = None
    for gindex, group in enumerate(self.param_groups):
        for pindex, p in enumerate(group["params"]):
            if p.grad is None:
                continue
            state = self.state[p]
            if len(state) == 0:
                self.init_state(group, p, gindex, pindex)

            self.prefetch_state(p)
            self.update_step(group, p, gindex, pindex)
            sync_gpu(p)
    if self.is_paged and p is not None:
        sync_gpu(p)

    return loss
'''


def _step_shape(source):
    """AST dump of a step function: decorators + body without the docstring."""
    function = ast.parse(dedent(source)).body[0]
    body = function.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    # 0.45 - 0.47 also carry a dead `overflows = []` (removed by bitsandbytes #1727).
    body = [
        node for node in body
        if not (
            isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "overflows"
            and isinstance(node.value, ast.List) and not node.value.elts
        )
    ]
    return ast.dump(ast.Module(body = function.decorator_list + body, type_ignores = []))


def _bnb_version_in_window():
    try:
        from importlib.metadata import version as _version
        from packaging.version import Version
        installed = Version(_version("bitsandbytes"))
        return Version(_BNB_MIN_VERSION) <= installed < Version(_BNB_MAX_VERSION)
    except Exception:
        return False


def _default_stream_on(p):
    # bitsandbytes launches its optimizer kernels on the legacy default stream (raw handle 0).
    # Skipping the sync is only order-preserving if torch also runs on it for p's device.
    return p.device.type == "cuda" and torch._C._cuda_getCurrentRawStream(p.device.index) == 0


def patch_bitsandbytes_optimizer_step_sync():
    try:
        import bitsandbytes.optim.optimizer as bnb_optimizer
        Optimizer8bit = bnb_optimizer.Optimizer8bit
    except Exception as e:
        return raise_error("bitsandbytes.optim.optimizer.Optimizer8bit.step", e)

    original_step = Optimizer8bit.__dict__.get("step", None)
    if original_step is None or getattr(original_step, _PATCHED_FLAG, False):
        return
    if not _bnb_version_in_window():
        return
    if not hasattr(torch._C, "_cuda_getCurrentRawStream"):
        return

    try:
        shape = _step_shape(inspect.getsource(original_step))
    except Exception as e:
        return raise_error("bitsandbytes.optim.optimizer.Optimizer8bit.step (source)", e)

    if shape == _step_shape(_STEP_TORCH_SYNC):
        def per_param_sync(p):
            torch.cuda.synchronize()
    elif shape == _step_shape(_STEP_SYNC_GPU):
        def per_param_sync(p):
            # Module-level lookup so later patches of bnb's sync_gpu still apply.
            bnb_optimizer.sync_gpu(p)
    else:
        if UNSLOTH_ENABLE_LOGGING:
            logger.info("Unsloth: bitsandbytes Optimizer8bit.step has an unknown shape; keeping its per-parameter sync.")
        return

    @torch.no_grad()
    def step(self, closure = None):
        # Paged optimizers prefetch managed memory on stream 0 and rely on every sync: untouched.
        if self.is_paged or os.environ.get(_KILL_SWITCH, "0") == "1":
            return original_step(self, closure)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self.initialized:
            self.check_overrides()
            self.to_gpu()  # needed for fairseq pure fp16 training
            self.initialized = True

        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    self.init_state(group, p, gindex, pindex)

                self.prefetch_state(p)
                self.update_step(group, p, gindex, pindex)
                if not _default_stream_on(p):
                    per_param_sync(p)
        return loss

    step.__doc__ = getattr(original_step, "__doc__", None)
    step.__qualname__ = getattr(original_step, "__qualname__", "Optimizer8bit.step")
    setattr(step, _PATCHED_FLAG, True)
    step._unsloth_original_step = original_step
    Optimizer8bit.step = step
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched bitsandbytes Optimizer8bit.step to skip the per-parameter sync for non-paged optimizers.")
pass
TEMPORARY_PATCHES.append(patch_bitsandbytes_optimizer_step_sync)

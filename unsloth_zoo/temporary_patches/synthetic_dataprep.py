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

"""Harden ``unsloth.dataprep.synthetic.SyntheticDataKit`` against silent
vLLM-subprocess failures.

Observed on Meta_Synthetic_Data_Llama3_2_(3B) inside an unsloth-blackwell
container: the parent already touched CUDA via ``patch_vllm`` /
``patch_vllm_graph_capture`` (which imports
``vllm.v1.worker.gpu_model_runner``). Forking the child ``vllm serve``
subprocess after CUDA init leaves the child with a contaminated NVML/Triton
state. The child logs ``Triton is installed but 0 active driver(s) found``
and dies inside ``EngineCore._initialize_kv_caches``. The parent
``PipeCapture`` has ``echo=False`` for stderr, so the actual error is
silenced until the 1200s timeout fires.

This patch:
  * runs ``torch.cuda.empty_cache() / synchronize() / gc.collect()`` before
    the fork so the child boots with a clean CUDA context,
  * pops ``LD_PRELOAD`` and ``VLLM_LOGGING_LEVEL`` from the env it inherits
    (CUDA_VISIBLE_DEVICES is preserved),
  * forces stderr ``PipeCapture`` instances to ``echo=True`` so users see
    the real failure immediately.

Idempotent via the ``_unsloth_patched`` sentinel. Cross-platform: the CUDA
empty-cache path is a no-op on Mac/MLX since ``torch.cuda.is_available()``
returns False. LD_PRELOAD pop is harmless when unset.
"""

import os
import gc
import functools

from .common import TEMPORARY_PATCHES


def patch_synthetic_data_kit_subprocess():
    try:
        from unsloth.dataprep import synthetic as _syn
    except Exception:
        return
    SyntheticDataKit = getattr(_syn, "SyntheticDataKit", None)
    PipeCapture = getattr(_syn, "PipeCapture", None)
    if SyntheticDataKit is None or PipeCapture is None:
        return
    orig_init = getattr(SyntheticDataKit, "__init__", None)
    if orig_init is None or getattr(orig_init, "_unsloth_patched", False):
        return

    # Flip stderr PipeCapture instances to echo so the user sees the real
    # subprocess failure. Wrap once; idempotent.
    if not getattr(PipeCapture, "_unsloth_echo_stderr", False):
        _orig_pc_init = PipeCapture.__init__

        @functools.wraps(_orig_pc_init)
        def _pc_init(self, pipe, *a, **kw):
            name = kw.get("name", "")
            if isinstance(name, str) and name.upper().endswith("STDERR"):
                kw["echo"] = True
            _orig_pc_init(self, pipe, *a, **kw)

        PipeCapture.__init__ = _pc_init
        PipeCapture._unsloth_echo_stderr = True

    # Wrap subprocess.Popen inside the synthetic module's globals so the
    # CUDA-clear / env-sanitise fires RIGHT BEFORE the actual fork+exec of
    # `vllm serve`, after `patch_vllm()` has re-touched CUDA. Clearing
    # CUDA at __init__ entry was insufficient because patch_vllm() runs
    # AFTER the wrapper.
    mod_globals = getattr(orig_init, "__globals__", None)
    if isinstance(mod_globals, dict) and not getattr(_syn, "_unsloth_popen_patched", False):
        import subprocess
        _orig_popen = mod_globals.get("subprocess") or subprocess

        class _CudaClearingPopen(subprocess.Popen):
            def __init__(self, *a, **kw):
                # Drop the parent's CUDA context so the child vLLM
                # subprocess can boot with a clean NVML / Triton state.
                try:
                    import torch
                    if hasattr(torch, "cuda") and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                except Exception:
                    pass
                for _ in range(3):
                    gc.collect()
                # vLLM EngineCore subprocess often hits
                # `cudaErrorDevicesUnavailable` when a parent holds an
                # initialised CUDA context inherited via fork+exec. Force
                # the child to a clean spawn-style boot by clearing the
                # vars that would otherwise carry NVML state forward.
                # CUDA_VISIBLE_DEVICES MUST be preserved.
                env = kw.get("env")
                if env is None:
                    env = dict(os.environ)
                else:
                    env = dict(env)
                for var in (
                    "LD_PRELOAD",
                    "VLLM_LOGGING_LEVEL",
                    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
                ):
                    env.pop(var, None)
                # Force vLLM to use spawn for its EngineCore worker so
                # the child gets a fresh interpreter not a forked clone.
                env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
                # If the parent already imported torchao etc. and held a
                # CUDA context, a stale lazy-init flag can survive
                # fork+exec. start_new_session=True isolates the
                # subprocess's PGID so a CUDA-busy SIGINT in the child
                # can't bubble back into the parent.
                kw["env"] = env
                kw.setdefault("start_new_session", True)
                super().__init__(*a, **kw)

        # Surgical replacement: only swap `subprocess.Popen` as visible
        # from the synthetic module. Other importers of `subprocess`
        # see the unmodified original.
        class _SubprocessShim:
            __slots__ = ("_real",)
            def __init__(self, real):
                object.__setattr__(self, "_real", real)
            def __getattr__(self, name):
                if name == "Popen":
                    return _CudaClearingPopen
                return getattr(object.__getattribute__(self, "_real"), name)

        mod_globals["subprocess"] = _SubprocessShim(_orig_popen)
        _syn._unsloth_popen_patched = True

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        # Drop CUDA context held by the parent before patch_vllm()
        # re-touches CUDA inside the original __init__. Defensive
        # double-clear; the Popen wrapper above is the primary fix.
        try:
            import torch
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
        except Exception:
            pass
        for _ in range(3):
            gc.collect()
        for var in ("LD_PRELOAD", "VLLM_LOGGING_LEVEL"):
            os.environ.pop(var, None)
        return orig_init(self, *args, **kwargs)

    __init__._unsloth_patched = True
    SyntheticDataKit.__init__ = __init__
pass
TEMPORARY_PATCHES.append(patch_synthetic_data_kit_subprocess)

# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified by Unsloth: rewritten. Upstream TritonTuner delegates to
# attn_gym._backends.cute.run_tunable (CuTe DSL, tvm_ffi and an on-disk winner
# cache), which is not vendored. This keeps the same TritonTuner interface
# (kernel, configs, key, prune, reset; launch as tuner[grid](*args, **kwargs))
# with an in-process tuner: the first launch per key times every surviving
# candidate with triton.testing.do_bench (restoring mutated outputs through
# `reset` before each trial and before the final launch, so the caller observes
# exactly one launch), skips candidates that fail to compile or load, and
# memoizes the winner. UNSLOTH_ATTN_GYM_AUTOTUNE=0 skips timing and pins the
# first candidate that launches. Never tunes while a CUDA graph is captured.
"""Tune Triton kernels over explicit candidates without ``triton.autotune``."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Sequence
from typing import Any, Dict, Tuple, Union

import triton
from triton.compiler.errors import CompileTimeAssertionFailure
from triton.runtime.errors import OutOfResources

try:
    from triton.runtime.errors import PTXASError
except ImportError:  # older Triton
    PTXASError = OutOfResources

# Modified by Unsloth: typing.Union runtime alias (Python 3.9).
Grid = Union[Tuple[int, ...], Callable[[Dict[str, Any]], Tuple[int, ...]]]


def _autotune_enabled() -> bool:
    return os.environ.get("UNSLOTH_ATTN_GYM_AUTOTUNE", "1") != "0"


def _capturing() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


class TritonTuner:
    """Launch a ``@triton.jit`` kernel with the best of ``configs`` for each ``key``."""

    benchmark_errors = (OutOfResources, CompileTimeAssertionFailure, PTXASError)

    def __init__(
        self,
        kernel: Any,
        configs: Sequence[triton.Config],
        *,
        key: Callable[[dict[str, Any]], tuple[Any, ...]],
        prune: Callable[[list[triton.Config], dict[str, Any]], list[triton.Config]] | None = None,
        reset: Callable[[Grid, dict[str, Any]], Callable[[], Any]] | None = None,
    ):
        self.kernel = kernel
        self._arg_names = kernel.arg_names
        self._configs = list(configs)
        self._key = key
        self._prune = prune
        self._reset = reset
        self._winners: dict[tuple[Any, ...], triton.Config] = {}
        self._lock = threading.Lock()

    def __getitem__(self, grid: Grid) -> Callable[..., None]:
        return lambda *args, **kwargs: self._run(grid, self.bind(args, kwargs))

    def bind(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        """Name positional launch arguments by the kernel's parameters."""
        return {**dict(zip(self._arg_names, args)), **kwargs}

    def _launch(self, config: triton.Config, grid: Grid, args: dict[str, Any]) -> None:
        meta = dict(config.all_kwargs())
        launch_grid = grid({**args, **meta}) if callable(grid) else grid
        self.kernel[launch_grid](**args, **meta)

    def _candidates(self, args: dict[str, Any]) -> list[triton.Config]:
        configs = self._configs if self._prune is None else self._prune(self._configs, args)
        return list(configs) or list(self._configs)

    def _run(self, grid: Grid, args: dict[str, Any]) -> None:
        key = self._key(args)
        winner = self._winners.get(key)
        if winner is not None:
            self._launch(winner, grid, args)
            return
        with self._lock:
            winner = self._winners.get(key)
            if winner is not None:
                self._launch(winner, grid, args)
                return
            # The snapshot is taken before any trial, so the final launch sees the caller's state.
            restore = self._reset(grid, args) if self._reset is not None else None
            winner, already_ran = self._select(grid, args, restore)
            self._winners[key] = winner
            # An untimed selection already ran the winner on the caller's state: launching again would repeat
            # the work (and be captured twice into a CUDA graph).
            if not already_ran:
                if restore is not None:
                    restore()
                self._launch(winner, grid, args)

    def _select(self, grid: Grid, args: dict[str, Any], restore) -> tuple[triton.Config, bool]:
        candidates = self._candidates(args)
        timed = _autotune_enabled() and len(candidates) > 1 and not _capturing()
        best, best_ms, last_error = None, float("inf"), None
        for config in candidates:
            try:
                if restore is not None:
                    restore()
                if not timed:
                    self._launch(config, grid, args)
                    return config, True
                from triton.testing import do_bench

                def trial(config=config):
                    if restore is not None:
                        restore()
                    self._launch(config, grid, args)

                ms = do_bench(trial, warmup=3, rep=10)
            except self.benchmark_errors as error:
                last_error = error
                continue
            if ms < best_ms:
                best, best_ms = config, ms
        if best is None:
            raise RuntimeError("no gather-attention Triton config could launch") from last_error
        return best, False


__all__ = ["TritonTuner"]

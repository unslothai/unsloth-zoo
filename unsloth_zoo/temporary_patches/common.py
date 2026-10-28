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

__all__ = [
    "TEMPORARY_PATCHES", 
    "torch_compile_options",
    "UNSLOTH_ENABLE_LOGGING",
    "UNSLOTH_COMPILE_DISABLE",
    "UNSLOTH_COMPILE_DISABLE_PARTIAL",
    "get_torch_compile_options",
    "is_transformers_v5_moe_quantization_available",
    "logger",
    "torch_compile",
    "_torch_compile",
    "_raw_torch_compile",
    "unwrap_already_compiled",
    "flatten_for_elementwise_norm",
    "unwrap_norm_weight",
    "CHUNK_ALIGN_ROWS",
    "chunk_row_bounds",
    "publish_to_modeling_module",
    "RESCOPE_PATCH_FLAG",
    "WRAPPER_INNER_ATTR",
]

import os
import sys
import logging
from unsloth_zoo.log import logger
import functools
UNSLOTH_ENABLE_LOGGING  = os.environ.get("UNSLOTH_ENABLE_LOGGING",  "0") == "1"
UNSLOTH_COMPILE_DISABLE = os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") in ("1", "partial",)
# "partial" keeps the source rewrites but turns torch.compile off, like compiler.py does.
UNSLOTH_COMPILE_DISABLE_PARTIAL = os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") == "partial"

# Marks this package's wrapper on transformers.conversion_mapping.get_model_conversion_mapping; set in
# conversion_mapping_rescope.py and read by bitsandbytes.py before it blames the transformers version.
RESCOPE_PATCH_FLAG = "_unsloth_zoo_patched_composite_prefix_renaming"

# Links a wrapper to the callable it wraps; deliberately not __wrapped__, because rescope unwraps __wrapped__ to pick
# its target and would replace a surviving wrapper, dropping its behaviour.
WRAPPER_INNER_ATTR = "_unsloth_wrapper_inner"

# Get only allowed options
import inspect
import torch
inductor_config_source = inspect.getsource(torch._inductor.config)

def _persistent_tma_matmul_ok():
    # torch 2.12+ guards every GEMM size when Inductor considers TMA templates (pytorch/pytorch issue 172925), so a
    # compiled torch._scaled_mm recompiles per new token count.
    import re
    found = re.match(r"(\d+)\.(\d+)", torch.__version__)
    return found is not None and (int(found.group(1)), int(found.group(2))) < (2, 12)
pass

@functools.lru_cache(1)
def determine_compile_threads():
    # See https://github.com/pytorch/pytorch/blob/ab2294d8289a7757a2fc321cdefac88e2b378edf/torch/_inductor/config.py#L771
    # Windows thread count = 1. See https://github.com/unslothai/unsloth-zoo/pull/187
    if sys.platform == "win32": return 1
    # get_torch_compile_options feeds this into the Inductor options dict, which outranks
    # TORCHINDUCTOR_COMPILE_THREADS, so returning the cpu count would undo _gpu_init's forcing.
    # Gated on our sentinel and NOT on TORCHINDUCTOR_COMPILE_THREADS: vLLM sets that to "1"
    # unconditionally at import (vllm/env_override.py), which would drop every vLLM user to
    # a single compile worker.
    if os.environ.get("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", "0") == "1": return 1
    cpu_count = os.cpu_count()
    return min(32, max(4, cpu_count))
pass

@functools.lru_cache(1)
def is_transformers_v5_moe_quantization_available():
    """True when Transformers exposes the v5 MoE weight-loading/dispatcher APIs
    that PR #659 patches (v4 doesn't use bare nn.Parameter MoE experts here)."""
    try:
        from transformers.core_model_loading import WeightConverter
        from transformers.integrations import moe as transformers_moe
        from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
    except (ImportError, AttributeError):
        return False

    return (
        WeightConverter is not None
        and getattr(Bnb4BitHfQuantizer, "param_needs_quantization", None) is not None
        and getattr(transformers_moe, "ALL_EXPERTS_FUNCTIONS", None) is not None
        and (
            getattr(transformers_moe, "grouped_mm_experts_forward", None) is not None
            or getattr(transformers_moe, "batched_mm_experts_forward", None) is not None
        )
    )

def allow_inductor_pool_aliasing():
    """
    torch >= 2.11 checks custom op outputs for aliasing on a compiled function's first call
    (torch/_functorch/_aot_autograd/runtime_wrappers.py, _AnalyzeCustomOpInputOutputMode). Under
    inference_mode it also sees Inductor's own memory_planning op inductor::_alloc_from_pool, which
    returns a view of its pool by design but declares no alias in its schema. That is a UserWarning,
    and a RuntimeError when the CI env var is set (error_on_custom_op_aliasing = bool(CI)), so every
    memory_planning=True graph under inference_mode fails in CI. Skip only that op; every other
    custom op is still checked.
    """
    try:
        # Import torch._dynamo first: importing runtime_wrappers alone is a circular import that leaves torch._dynamo
        # half initialised, which breaks torch.compile for the whole process.
        import torch._dynamo
        import torch._functorch._aot_autograd.runtime_wrappers as runtime_wrappers
    except Exception:
        return
    check = getattr(runtime_wrappers, "_check_custom_op_aliasing", None)
    if check is None or getattr(check, "_unsloth_skips_alloc_from_pool", False):
        return

    def _check_custom_op_aliasing(name, *args, **kwargs):
        if name == "inductor::_alloc_from_pool":
            return None
        return check(name, *args, **kwargs)
    _check_custom_op_aliasing._unsloth_skips_alloc_from_pool = True
    runtime_wrappers._check_custom_op_aliasing = _check_custom_op_aliasing
pass

def get_torch_compile_options(
    epilogue_fusion = True,
    max_autotune = False,
    shape_padding = True,
    debug = False,
    cudagraphs = False,
    coordinate_descent_tuning = False,
    logging = False,
    combo_kernels = False,
    group_fusion = True,
    memory_planning = True,
    multi_kernel = False,
    use_block_ptr = False,
):
    UNSLOTH_COMPILE_DEBUG         = os.environ.get("UNSLOTH_COMPILE_DEBUG",         "0") == "1"
    UNSLOTH_COMPILE_MAXIMUM       = os.environ.get("UNSLOTH_COMPILE_MAXIMUM",       "0") == "1"
    UNSLOTH_COMPILE_IGNORE_ERRORS = os.environ.get("UNSLOTH_COMPILE_IGNORE_ERRORS", "0") == "1"
    if UNSLOTH_ENABLE_LOGGING: logging = True

    # multi_kernel needs an integer, per torch/_inductor/config.py at L1283, so bools are converted to 1 or 0.
    multi_kernel = 1 if multi_kernel else 0
    if memory_planning: allow_inductor_pool_aliasing()

    # Relabel Inductor's compile progress bar
    try:
        import torch._inductor.async_compile
        from torch.hub import tqdm
        def replaced_tqdm(*args, **kwargs):
            kwargs["desc"] = "Unsloth: Compiling kernels"
            return tqdm(*args, **kwargs)
        torch._inductor.async_compile.tqdm = replaced_tqdm
    except:
        print("Unsloth: Failed editing tqdm to replace Inductor Compilation:")
    pass

    torch_compile_options = {
        "epilogue_fusion"           : epilogue_fusion,
        "max_autotune"              : max_autotune,
        "shape_padding"             : shape_padding,
        "trace.enabled"             : UNSLOTH_COMPILE_DEBUG or debug,
        "triton.cudagraphs"         : cudagraphs,
        "debug"                     : UNSLOTH_COMPILE_DEBUG or debug,
        "dce"                       : True,
        "memory_planning"           : memory_planning,
        "coordinate_descent_tuning" : coordinate_descent_tuning or UNSLOTH_COMPILE_MAXIMUM,
        "trace.graph_diagram"       : UNSLOTH_COMPILE_DEBUG or debug,
        "compile_threads"           : determine_compile_threads(), # Auto detects via https://github.com/unslothai/unsloth-zoo/pull/187
        "group_fusion"              : group_fusion, # [DEPRECATED]
        "disable_progress"          : not logging,
        "verbose_progress"          : logging,

        "triton.multi_kernel"       : multi_kernel, # RuntimeError: name 'multi_kernel_0' is not defined
        "triton.use_block_ptr"      : use_block_ptr,
        "triton.enable_persistent_tma_matmul" : _persistent_tma_matmul_ok(),
        "triton.autotune_at_compile_time"     : False,
        "triton.cooperative_reductions"       : False,
        # "reorder_for_compute_comm_overlap"  : True, # Fails for single GPU
        "cuda.compile_opt_level"              : "-O2",
        "cuda.enable_cuda_lto"                : True,
        # cuda.use_fast_math is left off: it causes incompatible gradient sizes on 2.6 and TypeError: bad operand type for unary -: 'SymbolicCallArg'.
        "combo_kernels"                       : combo_kernels,
        "benchmark_combo_kernel"              : True,
        "combo_kernel_foreach_dynamic_shapes" : True,
    }
    final_torch_compile_options = {}
    for key, value in torch_compile_options.items():
        splits = key.split(".")
        if all(k in inductor_config_source for k in splits):
            final_torch_compile_options[key] = value
    return final_torch_compile_options
pass
torch_compile_options = get_torch_compile_options(
    epilogue_fusion = True,
    max_autotune = False,
    shape_padding = True,
    debug = False,
    cudagraphs = False,
    coordinate_descent_tuning = False,
    logging = UNSLOTH_ENABLE_LOGGING,
    combo_kernels = False,
    memory_planning = False,
    multi_kernel = False,
    use_block_ptr = False,
)

from typing import Any, Callable, TypeVar
F = TypeVar("F", bound=Callable[..., Any])
def noop(*args: Any, **kwargs: Any):
    """No-op decorator/adapter usable as @noop, @noop(...), or noop(func, ...)."""
    # @noop / noop(func, ...): first positional arg is the function.
    if args and callable(args[0]):
        return torch.compiler.disable(args[0]) # type: ignore[return-value]

    # @noop(...): return a decorator.
    def _decorator(func: F) -> F:
        return torch.compiler.disable(func)
    return _decorator
pass

def unwrap_already_compiled(function):
    """Hand `torch.compile` the eager original when given an already-compiled one.

    `torch.compile` builds its wrapper with `functools.wraps(fn)`, which copies
    `fn.__dict__`, and then asserts the copy did not bring `get_compiler_config`
    along:

        torch/_dynamo/eval_frame.py: assert not hasattr(compile_wrapper, "get_compiler_config")

    a bare assert, so it surfaces as `AssertionError:` with no message at all.
    Dynamo unwraps its OWN wrappers before that point (`innermost_fn`), so the
    assert can only be reached through a wrapper that carries
    `get_compiler_config` without being a Dynamo wrapper. That is exactly what
    `_fall_back_to_eager_on_recompile_limit` returns: it forwards
    `get_compiler_config` and `_torchdynamo_orig_callable` on purpose, so that
    "is this compiled?" checks keep answering yes after the fallback is
    installed.

    Up to torch 2.10 `innermost_fn` followed `_torchdynamo_orig_callable`
    unconditionally and unwrapped those too, so recompiling one was harmless.
    torch 2.11 additionally requires `_torchdynamo_wrapper_id == id(fn)`,
    which a `functools.wraps` copy cannot satisfy by construction, and the
    assert became reachable. Under GPT-OSS it is: the generated
    `unsloth_compiled_module_gpt_oss.py` installs a fullgraph-fallback
    `apply_rotary_pos_emb` onto the modeling module during `pre_compile`, and
    `patch_GptOssAttention` re-compiles that same name during `post_compile`.

    `patch_function` has always done this unwrap by hand before compiling.
    Doing it in the shared funnel covers the bare decorators too. `__wrapped__`
    is what `functools.wraps` sets, and it is the eager original for both the
    fallback wrapper and a plain `torch.compile` result, so compiling it
    reproduces the pre-2.11 outcome exactly rather than degrading to eager.
    """
    seen = set()
    while callable(function) and hasattr(function, "get_compiler_config"):
        # A bound method forwards attribute lookups, so __wrapped__ is the UNBOUND original and following it would drop
        # the receiver; torch's innermost_fn stops on a bound method for that reason.
        if getattr(function, "__self__", None) is not None: break
        inner = getattr(function, "__wrapped__", None)
        # No __wrapped__ (an OptimizedModule, say) or a cycle: leave it and let the caller's guard deal with whatever torch makes of it.
        if not callable(inner) or id(inner) in seen: break
        seen.add(id(function))
        function = inner
    return function
pass


# Warn once per call site, not per compile attempt: temporary patches re-run at every phase and for every model, so
# repeats would print the same line several times per load.
_UNCOMPILED_CALL_SITES: set = set()


def _compile_or_fall_back(*args, **kwargs):
    """`torch.compile`, routed through the eager fallback under fullgraph.

    The alias below is what the bare decorators use (gpt_oss, qwen3_vl_moe and
    gemma each have `@torch_compile(..., fullgraph = True)` regions), and a
    `functools.partial(torch.compile)` reaches Dynamo directly, so cache
    exhaustion there stayed fatal while `patch_function`'s did not. Fixed here
    rather than per call site so a new one cannot miss it.

    The same goes for the compile call itself. `fullgraph = False` used to go
    straight to `torch.compile` with nothing around it, so a compiler that
    refuses to decorate at all (see `unwrap_already_compiled`) took the whole
    model load down. Decorating is the only thing guarded here: it either hands
    back the compiled callable it always did, or, if torch declines to build
    one, the eager function. Compilation itself still happens on the first
    call, so a setup where compile works keeps working, at the same speed.

    Both spellings are in use: `@torch_compile(...)` as a decorator factory, and
    `torch_compile(fn, ...)` applied directly (gemma.py, gpt_oss.py). Imported
    lazily: utils imports this module."""
    # `torch.compile`'s first parameter is named `model`, so `torch_compile(model = fn)`
    # is a third legal spelling. Take it out of the compile kwargs, where it would
    # collide with the function passed positionally, and treat it as that function.
    # Popped whether or not it is callable: `torch_compile(model = None, ...)` is the
    # decorator-factory spelling, and a `model` left behind collides with the function
    # `_compile` passes positionally, which the guard below would then swallow as a
    # refusal to compile.
    function = args[0] if args and callable(args[0]) else None
    if function is None and "model" in kwargs:
        model = kwargs.pop("model")
        if callable(model): function = model

    if kwargs.get("fullgraph"):
        from .utils import torch_compile_with_fallback
        _compile = torch_compile_with_fallback(**kwargs)
    else:
        def _compile(function):
            from .utils import eager_under_zero3
            return eager_under_zero3(function, torch.compile(function, **kwargs))

    def decorate(function):
        function = unwrap_already_compiled(function)
        try:
            return _compile(function)
        except Exception as exception:
            # Never fatal: a temporary patch that cannot be compiled is a
            # performance problem, and the uncompiled function is still correct.
            label = getattr(function, "__qualname__", None) or repr(function)
            if label not in _UNCOMPILED_CALL_SITES:
                _UNCOMPILED_CALL_SITES.add(label)
                logger.warning(
                    f"Unsloth: torch.compile refused to wrap {label}; running "
                    f"it eagerly. Training and inference are unaffected apart "
                    f"from speed. ({type(exception).__name__}: {exception})"
                )
            return function
    if function is not None:
        return decorate(function)
    return decorate


if UNSLOTH_COMPILE_DISABLE:
    torch_compile = noop
    # _raw_torch_compile is for the one caller that applies the fallback itself, so the alias routing does not wrap it twice.
    _raw_torch_compile = noop
else:
    torch_compile = functools.partial(
        _compile_or_fall_back,
        options = torch_compile_options,
    )
    _raw_torch_compile = functools.partial(
        torch.compile,
        options = torch_compile_options,
    )

if UNSLOTH_COMPILE_DISABLE:
    _torch_compile = noop
else:
    _torch_compile = functools.partial(
        _compile_or_fall_back,
    )

def flatten_for_elementwise_norm(hidden_states):
    """``(..., H)`` -> ``((N, H), original_shape)``.

    Norms only touch the last dim, so making every caller rank 2 drops the rank and
    leading-dim guards from the shared kernel's Dynamo cache.
    """
    shape = hidden_states.shape
    return hidden_states.reshape(-1, shape[-1]), shape
pass


# Opt-in: aligned boundaries reorder the float32 per-chunk sums.
CHUNK_ALIGN_ROWS = 128 if os.environ.get("UNSLOTH_CE_CHUNK_ALIGN", "0").strip() == "1" else 0

def chunk_row_bounds(n_rows, chunks, align = CHUNK_ALIGN_ROWS):
    """``align``-multiple row ranges no larger than ``ceil(n_rows / chunks)`` (may add ranges).
    None means keep torch.chunk."""
    if align <= 1 or chunks <= 1:
        return None
    cap = -(-n_rows // chunks)
    size = (cap // align) * align
    # A 1-row tail hits the K=1 head-gradient matmul bug (see test_chunked_log_softmax_one_row_chunk).
    if size == 0 or 4 * size < 3 * cap or n_rows % size == 1:
        return None
    return list(range(0, n_rows, size)) + [n_rows]
pass


def unwrap_norm_weight(weight):
    """Hand a norm weight to a compiled kernel as a plain Tensor view.

    Dynamo pins a static shape for anything whose ``type`` is ``nn.Parameter`` even
    under ``dynamic = True``, so every distinct norm width would be another cache
    entry. A view takes dynamic shapes, and autograd still reaches the Parameter.
    """
    if weight is None: return None
    return weight.reshape(-1)
pass

def publish_to_modeling_module(modeling_module, **names):
    """Make helper names used by a patched forward importable from the modeling module.

    ``create_standalone_class`` copies the patched forward's source into
    ``unsloth_compiled_cache`` and resolves its free names against the modeling
    module, so unsloth_zoo helpers must be published here or the generated module
    raises NameError on import.
    """
    for name, value in names.items():
        try:
            setattr(modeling_module, name, value)
        except Exception as e:
            # Log, never swallow: silence reappears as an unexplained NameError from
            # generated cache code. `warning`, not `warning_once`, which transformers
            # monkeypatches on later and may not exist yet.
            logger.warning(
                f"Unsloth: could not publish `{name}` to "
                f"{getattr(modeling_module, '__name__', modeling_module)}: {e}"
            )
pass

global TEMPORARY_PATCHES
TEMPORARY_PATCHES = []


def _maybe_compile(**kwargs):
    """torch.compile, unless UNSLOTH_COMPILE_DISABLE asks for it to be off.

    Defined here, not beside its callers: the compiler copies decorated source
    verbatim into generated trainer modules, where a name local to
    rl_replacements.py NameErrors at import and the failure is swallowed.
    compiler.py emits the import when the name appears.
    """
    if UNSLOTH_COMPILE_DISABLE or UNSLOTH_COMPILE_DISABLE_PARTIAL:
        return lambda fn: fn
    if not kwargs.get("fullgraph"):
        # `unwrap_already_compiled` for the same reason the funnel needs it:
        # `torch.compile(**kwargs)` is Dynamo's own decorator, and handing it a
        # `functools.wraps` copy of a compiled function is a bare AssertionError
        # on torch 2.11+.
        def decorate(function):
            return torch.compile(unwrap_already_compiled(function), **kwargs)
        return decorate
    # Under fullgraph Dynamo makes cache exhaustion fatal, so these regions get
    # `patch_function`'s eager fallback, which unwraps on its own. Lazy import:
    # utils imports this module.
    from .utils import torch_compile_with_fallback
    return torch_compile_with_fallback(**kwargs)

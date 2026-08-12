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
    "publish_to_modeling_module",
]

import os
import sys
import logging
from ..log import logger
import functools
UNSLOTH_ENABLE_LOGGING  = os.environ.get("UNSLOTH_ENABLE_LOGGING",  "0") == "1"
UNSLOTH_COMPILE_DISABLE = os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") in ("1", "partial",)
# "partial" keeps the source rewrites but turns torch.compile off, like compiler.py does.
UNSLOTH_COMPILE_DISABLE_PARTIAL = os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") == "partial"

# Get only allowed options
import inspect
import torch
inductor_config_source = inspect.getsource(torch._inductor.config)

@functools.lru_cache(1)
def determine_compile_threads():
    # See https://github.com/pytorch/pytorch/blob/ab2294d8289a7757a2fc321cdefac88e2b378edf/torch/_inductor/config.py#L771
    # Windows thread count = 1. See https://github.com/unslothai/unsloth-zoo/pull/187
    if sys.platform == "win32": return 1
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

    # https://github.com/pytorch/pytorch/blob/c665594c1edca9a507b0ec8b18ab74a0ecb65bc3/torch/_inductor/config.py#L1283
    # Needs integer
    multi_kernel = 1 if multi_kernel else 0

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
        "triton.enable_persistent_tma_matmul" : True,
        "triton.autotune_at_compile_time"     : False,
        "triton.cooperative_reductions"       : False,
        # "reorder_for_compute_comm_overlap"  : True, # Fails for single GPU
        "cuda.compile_opt_level"              : "-O2",
        "cuda.enable_cuda_lto"                : True,
        # "cuda.use_fast_math"                : True, # Disable fast math
        # Causes incompatible gradient sizes on 2.6
        # And TypeError: bad operand type for unary -: 'SymbolicCallArg'
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
        # A bound method forwards attribute lookups to the function underneath it,
        # so `__wrapped__` here is the UNBOUND original and following it would drop
        # the receiver. torch's own `innermost_fn` stops on a bound method for that
        # exact reason ("id(bound_method) != id(wrapper_function), so we won't
        # unwrap through __func__ and lose the self binding").
        if getattr(function, "__self__", None) is not None: break
        inner = getattr(function, "__wrapped__", None)
        # No `__wrapped__` (an OptimizedModule, say) or a cycle: leave it be and
        # let the caller's guard deal with whatever torch makes of it.
        if not callable(inner) or id(inner) in seen: break
        seen.add(id(function))
        function = inner
    return function
pass


# Warn once per call site, not once per compile attempt: the temporary patches
# re-run at every phase and for every model, so an unwarned repeat would print
# the same line several times per load.
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
            return torch.compile(function, **kwargs)

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
    # For the one caller that applies the fallback itself, so the alias's
    # routing does not wrap it twice.
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

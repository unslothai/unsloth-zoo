# SPDX-License-Identifier: AGPL-3.0-only
"""Dynamic MLX prefill with a fixed-step arithmetic grid."""

import ast
import functools
import importlib
import inspect
import textwrap
from contextlib import contextmanager
from contextvars import ContextVar
from types import FunctionType

import mlx.core as mx
import mlx.nn as nn


_PREFILL_LEGACY_STEP_SIZE = 256


_PREFILL_ARITHMETIC_STEP = ContextVar("prefill_arithmetic_step", default = _PREFILL_LEGACY_STEP_SIZE)


_PREFILL_MAX_CHUNK_SIZE = 2048


_PREFILL_MAX_QUANTIZED_OUTPUT_DIM = 1024


_PREFILL_GENERATE_STEP = None


_PREFILL_ATTENTION_CLASSES = {}


_PREFILL_KV_OWNERS = ContextVar("prefill_kv_owners", default = None)


_PREFILL_BATCH_SCHEDULE = ContextVar("prefill_batch_schedule", default = None)


_PREFILL_BATCH_METHODS = None


def _prefill_attention(original, queries, keys, values, cache, scale, mask, **kwargs):
    from mlx_vlm.models.cache import BatchKVCache, BatchQuantizedKVCache, BatchRotatingKVCache, RotatingKVCache

    step = _PREFILL_ARITHMETIC_STEP.get()
    owners = _PREFILL_KV_OWNERS.get()
    rotating = type(cache) in (RotatingKVCache, BatchRotatingKVCache)
    batched = type(cache) in (BatchKVCache, BatchQuantizedKVCache)
    window = cache.max_size if rotating else None
    keep = cache.keep if type(cache) is RotatingKVCache else 0
    identity = (id(keys), id(values))
    if (rotating or batched) and owners is not None:
        owners[identity] = (keys, values, window, keep, batched)
    elif cache is None and owners is not None and identity in owners:
        window, keep, batched = owners[identity][2:]
    causal = isinstance(mask, str) and mask == "causal"
    batched = batched and isinstance(mask, mx.array) and mask.ndim >= 2
    if queries.shape[-2] <= step or not (causal or window is not None or batched):
        return original(queries, keys, values, cache = cache, scale = scale, mask = mask, **kwargs)
    prefix = (keys[0] if isinstance(keys, (tuple, list)) else keys).shape[-2] - queries.shape[-2]
    outputs = []
    for start in range(0, queries.shape[-2], step):
        end = min(start + step, queries.shape[-2])
        first = max(0, prefix + start - window + 1) if window is not None else 0
        # Native window masks use compact positions, including retained prefix keys.
        part_mask = mask if causal else mask[..., start:end, first:prefix + end]
        outputs.append(original(
            queries[..., start:end, :], _slice_prefill_keys(keys, first, prefix + end, keep),
            _slice_prefill_keys(values, first, prefix + end, keep), cache = cache, scale = scale, mask = part_mask, **kwargs,
        ))
    if rotating:
        # Keep the same excess tail and ring cursor as the last fixed-size append.
        first = max(0, cache.keys.shape[-2] - queries.shape[-2] + start - window + 1)
        cache.keys = _slice_prefill_keys(cache.keys, first, None, keep)
        cache.values = _slice_prefill_keys(cache.values, first, None, keep)
        cache._idx = cache.keys.shape[-2]
        if type(cache) is BatchRotatingKVCache:
            cache.left_padding -= first
    return mx.concatenate(outputs, axis = -2)


def _slice_prefill_keys(value, first, end, keep):
    if isinstance(value, (tuple, list)):
        return type(value)(_slice_prefill_keys(part, first, end, keep) for part in value)
    if first > 0 and keep:
        return mx.concatenate([value[..., :keep, :], value[..., first + keep:end, :]], axis = -2)
    return value[..., first:end, :]


@functools.cache
def _prefill_attention_methods(base):
    methods = {}
    for cls in reversed(base.__mro__):
        methods.update(vars(cls))
    return {
        name: call for name, call in methods.items()
        if isinstance(call, FunctionType)
        and any(
            name == "scaled_dot_product_attention"
            or getattr(call.__globals__.get(name), "__name__", None) == "scaled_dot_product_attention"
            for name in call.__code__.co_names
        )
    }


def _has_supported_prefill_attention(base):
    methods = _prefill_attention_methods(base)
    return bool(methods) and all(
        callable(call.__globals__.get("scaled_dot_product_attention"))
        and "state" not in call.__code__.co_names
        for call in methods.values()
    )


def _prefill_attention_method(original):
    def attention(*args, **kwargs):
        return _prefill_attention(
            original.__globals__["scaled_dot_product_attention"], *args, **kwargs,
        )

    namespace = {**original.__globals__, "scaled_dot_product_attention": attention}
    call = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    call.__kwdefaults__ = original.__kwdefaults__
    return call


def _prefill_attention_class(base):
    if base not in _PREFILL_ATTENTION_CLASSES:
        _PREFILL_ATTENTION_CLASSES[base] = type(f"_Prefill{base.__name__}", (base,), {
            name: _prefill_attention_method(call) for name, call in _prefill_attention_methods(base).items()
        })
    return _PREFILL_ATTENTION_CLASSES[base]


class DynamicPrefillSchedule:
    """Schedule prefill on a fixed arithmetic grid; optionally consume largest chunks first.

    Prefix-cache integrations enabling ``largest_first`` must use ``steps_until``
    to count forwards, since tail chunks depend on the remaining prompt length.
    """

    def __init__(self, model = None, step_size = _PREFILL_LEGACY_STEP_SIZE, quantized_kv_start = None, *, largest_first = False):
        if type(step_size) is not int or step_size <= 0:
            raise ValueError("prefill step_size must be a positive integer")
        self.model = model
        self.step_size = step_size
        self.quantized_kv_start = quantized_kv_start
        self.largest_first = largest_first
        # Conversion follows the first fixed-step append reaching this threshold.
        self._quantization_boundary = None if quantized_kv_start is None else max(
            step_size, ((quantized_kv_start + step_size - 1) // step_size) * step_size,
        )

    def _grid_boundary(self, offset):
        boundary, chunk = 0, self.step_size
        limit = max(self.step_size, _PREFILL_MAX_CHUNK_SIZE // self.step_size * self.step_size)
        while chunk < limit:
            boundary += chunk
            if offset < boundary:
                return boundary
            chunk = min(chunk * 2, limit)
        return boundary + ((offset - boundary) // limit + 1) * limit

    def next_boundary(self, offset):
        boundary = self._grid_boundary(offset)
        if self._quantization_boundary is not None and offset < self._quantization_boundary:
            boundary = min(boundary, self._quantization_boundary)
        return boundary

    def boundary_at(self, rows):
        if self.largest_first:
            return max(0, rows // self.step_size * self.step_size)
        boundary, chunk = 0, self.step_size
        limit = max(self.step_size, _PREFILL_MAX_CHUNK_SIZE // self.step_size * self.step_size)
        while chunk < limit and boundary + chunk <= rows:
            boundary += chunk
            chunk = min(chunk * 2, limit)
        boundary += max(0, (rows - boundary) // chunk) * chunk
        if self._quantization_boundary is not None and self._quantization_boundary <= rows:
            boundary = max(boundary, self._quantization_boundary)
        return boundary

    def steps_until(self, offset, boundary):
        """Count prefill forwards from a cached offset through a snapshot boundary."""
        count = 0
        while offset < boundary:
            offset += self._chunk_size_at(boundary - offset, offset)
            count += 1
        return count

    def chunk_size(self, remaining, cache):
        offsets = [entry.offset for entry in cache if type(getattr(entry, "offset", None)) is int]
        if not offsets or any(offset != offsets[0] for offset in offsets):
            raise ValueError("Dynamic prefill requires a consistent absolute cache offset")
        return self._chunk_size_at(remaining, offsets[0])

    def _chunk_size_at(self, remaining, offset):
        owners = _PREFILL_KV_OWNERS.get()
        if owners is not None:
            owners.clear()
        if self.largest_first:
            limit = max(self.step_size, _PREFILL_MAX_CHUNK_SIZE // self.step_size * self.step_size)
            budget = min(remaining, limit)
            if self._quantization_boundary is not None and offset < self._quantization_boundary:
                budget = min(budget, self._quantization_boundary - offset)
            if offset % self.step_size:
                return min(budget, self.step_size - offset % self.step_size)
            if budget < self.step_size:
                return budget
            return self.step_size * (1 << ((budget // self.step_size).bit_length() - 1))
        chunk = min(self.next_boundary(offset) - offset, remaining)
        # Preserve the small final forward's kernel dispatch, including wide projections.
        if chunk == remaining and remaining > self.step_size and remaining % self.step_size:
            chunk = remaining // self.step_size * self.step_size
        return chunk

    @staticmethod
    @contextmanager
    def arithmetic(model, step_size = _PREFILL_LEGACY_STEP_SIZE):
        changed = []
        token = _PREFILL_KV_OWNERS.set({})
        step_token = _PREFILL_ARITHMETIC_STEP.set(step_size)
        try:
            for _, module in model.language_model.named_modules():
                base = type(module)
                if _has_supported_prefill_attention(base):
                    module.__class__ = _prefill_attention_class(base)
                elif base is nn.QuantizedLinear and (
                    step_size != _PREFILL_LEGACY_STEP_SIZE
                    or module.weight.shape[0] <= _PREFILL_MAX_QUANTIZED_OUTPUT_DIM
                ):
                    module.__class__ = _PrefillQuantizedLinear
                elif base is nn.Linear:
                    module.__class__ = _PrefillLinear
                else:
                    continue
                changed.append((module, base))
            yield
        finally:
            for module, base in reversed(changed):
                module.__class__ = base
            _PREFILL_KV_OWNERS.reset(token)
            _PREFILL_ARITHMETIC_STEP.reset(step_token)


class _PrefillQuantizedLinear(nn.QuantizedLinear):
    def __call__(self, x):
        step = _PREFILL_ARITHMETIC_STEP.get()
        if x.ndim == 3 and x.shape[1] > step:
            # Preserve the native batch-by-step matrix layout and kernel dispatch.
            return mx.concatenate([
                nn.QuantizedLinear.__call__(self, mx.contiguous(x[:, start : start + step]))
                for start in range(0, x.shape[1], step)
            ], axis = 1)
        return nn.QuantizedLinear.__call__(self, x)


class _PrefillLinear(nn.Linear):
    def __call__(self, x):
        step = _PREFILL_ARITHMETIC_STEP.get()
        if x.ndim == 3 and x.shape[1] > step:
            return mx.concatenate([
                nn.Linear.__call__(self, mx.contiguous(x[:, start : start + step]))
                for start in range(0, x.shape[1], step)
            ], axis = 1)
        return nn.Linear.__call__(self, x)


def _adapt_prefill_generate_step(original):
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    except (OSError, TypeError, SyntaxError):
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return None
    function = tree.body[0]
    if function.decorator_list or function.args.kwarg is None:
        return None
    expected = ast.dump(ast.parse("min(prefill_step_size, inputs_embeds.shape[1] - 1)", mode = "eval").body)
    assignments = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "n_to_process"
        and ast.dump(node.value) == expected
    ]
    branches = [
        node for node in ast.walk(function) if isinstance(node, ast.If)
        and ast.dump(node.test) == ast.dump(ast.parse(
            "prefill_step_size is not None and should_chunk", mode = "eval"
        ).body)
    ]
    if len(assignments) != 1 or len(branches) != 1 or assignments[0] not in list(ast.walk(branches[0])):
        return None
    assignment = assignments[0]
    assignment.value = ast.parse(
        "_unsloth_prefill_schedule.chunk_size(inputs_embeds.shape[1] - 1, prompt_cache)",
        mode = "eval",
    ).body
    branch = branches[0]
    branch.body = [ast.With(
        items = [ast.withitem(context_expr = ast.parse(
            "_unsloth_prefill_schedule.arithmetic(model, _unsloth_prefill_schedule.step_size)", mode = "eval"
        ).body)], body = branch.body,
    )]
    function.args.kwonlyargs.append(ast.arg(arg = "_unsloth_prefill_schedule"))
    function.args.kw_defaults.append(None)
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    try:
        exec(compile(tree, inspect.getsourcefile(original) or "<mlx-vlm-prefill>", "exec"), namespace)
    except (TypeError, ValueError, NameError):
        return None
    return namespace[function.name]


def _install_dynamic_prefill():
    global _PREFILL_GENERATE_STEP
    try:
        ar = importlib.import_module("mlx_vlm.generate.ar")
        dispatch = importlib.import_module("mlx_vlm.generate.dispatch")
    except ModuleNotFoundError:
        return False
    if _PREFILL_GENERATE_STEP is not None:
        return ar.generate_step is _PREFILL_GENERATE_STEP and dispatch.generate_step is _PREFILL_GENERATE_STEP
    original = ar.generate_step
    if dispatch.generate_step is not original:
        return False
    adapted = _adapt_prefill_generate_step(original)
    if adapted is None:
        return False

    @functools.wraps(original)
    def generate_step(*args, _unsloth_prefill_schedule = None, **kwargs):
        if _unsloth_prefill_schedule is None:
            yield from original(*args, **kwargs)
            return
        bound = inspect.signature(original).bind(*args, **kwargs)
        input_ids = bound.arguments["input_ids"]
        model = bound.arguments["model"]
        if (
            type(_unsloth_prefill_schedule) is not DynamicPrefillSchedule
            or _unsloth_prefill_schedule.model is not model or model.training
        ):
            raise ValueError("Dynamic prefill requires an inference schedule for this model")
        from mlx_vlm.models.cache import ArraysCache, KVCache, QuantizedKVCache, RotatingKVCache

        caches = kwargs.get("prompt_cache") or ()
        supported_cache = all(type(cache) in (ArraysCache, KVCache, QuantizedKVCache, RotatingKVCache) for cache in caches)
        step = kwargs.get("prefill_step_size", _unsloth_prefill_schedule.step_size)
        unsupported = ("draft_model", "prompt_cache_checkpoint", "prompt_cache_checkpoint_len")
        uniform_kv = (
            kwargs.get("kv_quant_scheme", "uniform") == "uniform"
            and all(kwargs.get(name) is None for name in (
                "kv_key_bits", "kv_value_bits", "kv_key_scheme", "kv_value_scheme",
            ))
        )
        if (
            not supported_cache or input_ids.ndim != 2 or input_ids.shape[0] != 1
            or step is None
            or any(kwargs.get(name) is not None for name in unsupported)
            or (kwargs.get("kv_bits") is not None and not uniform_kv)
        ):
            yield from original(*args, **kwargs)
            return
        quantized_start = None
        if kwargs.get("kv_bits") is not None:
            quantized_start = kwargs.get("quantized_kv_start", original.__globals__.get("DEFAULT_QUANTIZED_KV_START", 0))
        schedule = DynamicPrefillSchedule(model, step, quantized_start, largest_first = _unsloth_prefill_schedule.largest_first)
        kwargs["prefill_step_size"] = step
        yield from adapted(
            *args,
            _unsloth_prefill_schedule = schedule, **kwargs,
        )

    ar.generate_step = dispatch.generate_step = _PREFILL_GENERATE_STEP = generate_step
    return True


def _install_dynamic_batch_prefill():
    global _PREFILL_BATCH_METHODS
    from mlx_vlm.generate.ar import BatchGenerator, PromptProcessingBatch
    from mlx_vlm.models.cache import ArraysCache, BatchKVCache, BatchQuantizedKVCache, BatchRotatingKVCache, KVCache, QuantizedKVCache, RotatingKVCache

    if _PREFILL_BATCH_METHODS is not None:
        return
    original_init = BatchGenerator.__init__
    original_next = BatchGenerator._next
    original_step = PromptProcessingBatch.prompt_step

    @functools.wraps(original_init)
    def initialize(self, *args, _unsloth_prefill_schedule = None, **kwargs):
        if _unsloth_prefill_schedule is not None:
            model = inspect.signature(original_init).bind(self, *args, **kwargs).arguments["model"]
            if (
                type(_unsloth_prefill_schedule) is not DynamicPrefillSchedule
                or getattr(_unsloth_prefill_schedule.model, "language_model", None) is not model or model.training
            ):
                raise ValueError("Dynamic prefill requires an inference schedule for this model")
            kwargs.setdefault("prefill_step_size", _unsloth_prefill_schedule.step_size)
        original_init(self, *args, **kwargs)
        self._unsloth_prefill_schedule = _unsloth_prefill_schedule

    @functools.wraps(original_next)
    def advance(self, *args, **kwargs):
        token = _PREFILL_BATCH_SCHEDULE.set(getattr(self, "_unsloth_prefill_schedule", None))
        try:
            return original_next(self, *args, **kwargs)
        finally:
            _PREFILL_BATCH_SCHEDULE.reset(token)

    @functools.wraps(original_step)
    def prompt_step(self):
        schedule = _PREFILL_BATCH_SCHEDULE.get()
        step = self.prefill_step_size
        if (
            schedule is None or step is None or not self.needs_processing()
            or self.model is not schedule.model.language_model or self.model.training
            or self.draft_model is not None or self._apc_manager is not None
            or self._right_pad_per_row is not None
            or any(type(cache) not in (ArraysCache, BatchKVCache, BatchQuantizedKVCache, BatchRotatingKVCache, KVCache, QuantizedKVCache, RotatingKVCache) for cache in self.prompt_cache)
        ):
            return original_step(self)
        # Batch offsets include per-row padding; the native arithmetic grid uses columns.
        batch_schedule = DynamicPrefillSchedule(schedule.model, step, largest_first = schedule.largest_first)
        self.prefill_step_size = batch_schedule._chunk_size_at(
            self._inputs_embeds.shape[1] - 1, self._processed_prompt_columns,
        )
        # Some models strip each row's padding before choosing attention kernels.
        if self._processed_prompt_columns < max(self._left_padding_per_row):
            self.prefill_step_size = min(self.prefill_step_size, step)
        try:
            with schedule.arithmetic(schedule.model, step):
                return original_step(self)
        finally:
            self.prefill_step_size = step

    BatchGenerator.__init__ = initialize
    BatchGenerator._next = advance
    PromptProcessingBatch.prompt_step = prompt_step
    _PREFILL_BATCH_METHODS = (initialize, advance, prompt_step)


def create_dynamic_prefill_schedule(model, step_size = _PREFILL_LEGACY_STEP_SIZE, quantized_kv_start = None, *, largest_first = False):
    """Return a dynamic prefill schedule, or None to retain native prefill.

    Native modality policies decide whether embeddings can be chunked. Uniform
    quantized KV is supported; other quantization schemes, speculative decoding,
    and checkpoints retain the native path. Cold batches support unequal prompt
    lengths; batched prefix reuse and speculative decoding retain native prefill.
    Pass the schedule as ``_unsloth_prefill_schedule`` to ``batch_generate`` or
    ``BatchGenerator``, with ``prefill_step_size=schedule.step_size``. Cache reuse must
    use this schedule's step size, boundaries, and a separate cache namespace.
    Enable ``largest_first`` to start at the largest fitting chunk and shrink the
    tail; prefix-cache integrations must then count forwards with ``steps_until``.
    The factory keeps its legacy 256-step default for older cache integrations.
    Pass ``step_size=2048`` to match native mlx-vlm arithmetic; with a 2048 cap,
    largest-first uses full 2048 chunks followed by one exact remainder.
    Obtain a new schedule after changing the model.
    """
    language = getattr(model, "language_model", None)
    if language is None or model.training or getattr(model, "_unsloth_mlx_distributed_parallel_mode", None):
        return None
    modules = list(language.named_modules())
    if any(
        hasattr(module, "lora_a") or hasattr(module, "lora_b")
        for _, module in modules
    ):
        return None
    if any(_prefill_attention_methods(type(module)) and not _has_supported_prefill_attention(type(module)) for _, module in modules):
        return None
    projections = [module for _, module in modules if isinstance(module, (nn.Linear, nn.QuantizedLinear))]
    if not projections or any(
        type(module) not in (nn.Linear, nn.QuantizedLinear) or module.training
        for module in projections
    ):
        return None
    if not _install_dynamic_prefill():
        return None
    from mlx_vlm.models.cache import ArraysCache, KVCache, QuantizedKVCache, RotatingKVCache, make_prompt_cache

    caches = make_prompt_cache(language)
    if not any(type(getattr(cache, "offset", None)) is int for cache in caches):
        return None
    if any(type(cache) not in (ArraysCache, KVCache, QuantizedKVCache, RotatingKVCache) for cache in caches):
        return None
    _install_dynamic_batch_prefill()
    return DynamicPrefillSchedule(model, step_size, quantized_kv_start, largest_first = largest_first)

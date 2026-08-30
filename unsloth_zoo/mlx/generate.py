# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Batched generation primitives for training-resident MLX models."""

from __future__ import annotations

import ast
import asyncio
import importlib
import inspect
import textwrap
import math
import os
import sys
import threading
import types
import warnings
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import Any, Iterator, Literal, Sequence


_TEXT_EVENT_FIELDS = frozenset(("uid", "token", "logprobs", "finish_reason"))


def _installed_mlx_lm_version() -> str:
    """Name the installed mlx-lm, or an unversioned phrase if metadata is unusable."""

    try:
        from importlib.metadata import version

        installed = version("mlx-lm")
    except Exception:
        installed = None
    if not isinstance(installed, str) or not installed.strip():
        return "the installed mlx-lm"
    return f"mlx-lm {installed.strip().splitlines()[0]}"


def _current_async_task():
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


class _GenerationModeLock:
    """Thread-reentrant lock that rejects unsafe same-thread task overlap."""

    def __init__(self):
        self._lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._owner_thread = None
        self._owner_task = None
        self._depth = 0

    def acquire(self, blocking=True, timeout=-1):
        owner_thread = threading.get_ident()
        owner_task = _current_async_task()
        with self._state_lock:
            if (
                self._depth
                and self._owner_thread == owner_thread
                and self._owner_task is not owner_task
            ):
                raise RuntimeError(
                    "generation_mode contexts from different async tasks cannot "
                    "overlap on the same thread."
                )
        acquired = (
            self._lock.acquire(blocking)
            if timeout == -1
            else self._lock.acquire(blocking, timeout)
        )
        if not acquired:
            return False
        with self._state_lock:
            if self._depth == 0:
                self._owner_thread = owner_thread
                self._owner_task = owner_task
            self._depth += 1
        return True

    def release(self):
        owner_thread = threading.get_ident()
        owner_task = _current_async_task()
        with self._state_lock:
            if (
                not self._depth
                or self._owner_thread != owner_thread
                or self._owner_task is not owner_task
            ):
                raise RuntimeError("generation_mode lock released by a non-owner.")
            self._depth -= 1
            if self._depth == 0:
                self._owner_thread = None
                self._owner_task = None
        self._lock.release()


_GENERATION_MODE_LOCK = _GenerationModeLock()
_GENERATION_MODE_DEPTH = 0
_GENERATION_LIMIT_SNAPSHOT: dict[str, int] | None = None


_KV_QUANT_CONTROLS = ("kv_bits", "kv_group_size", "kv_quant_scheme", "quantized_kv_start")


@dataclass(frozen=True)
class SamplingParams:
    """Sampling controls understood by mlx-lm's sampler factory."""

    temperature: float = 0.0
    top_p: float = 0.0
    top_k: int = 0
    min_p: float = 0.0
    seed: int | None = None

    def __post_init__(self):
        temperature = float(self.temperature)
        top_p = float(self.top_p)
        min_p = float(self.min_p)
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
                raise TypeError("seed must be an integer or None.")
            object.__setattr__(self, "seed", int(self.seed) % (2**64))
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be a finite value >= 0.")
        if not math.isfinite(top_p) or not 0 <= top_p <= 1:
            raise ValueError("top_p must be a finite value between 0 and 1.")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, Integral):
            raise TypeError("top_k must be an integer.")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0.")
        if not math.isfinite(min_p) or not 0 <= min_p <= 1:
            raise ValueError("min_p must be a finite value between 0 and 1.")
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "top_p", top_p)
        object.__setattr__(self, "top_k", int(self.top_k))
        object.__setattr__(self, "min_p", min_p)


@dataclass(frozen=True)
class GenerationDefaults:
    """Batch-wide defaults used when a request does not override a value."""

    max_tokens: int = 256
    sampling: SamplingParams = field(default_factory=SamplingParams)
    stop_strings: tuple[str, ...] = ()
    prefill_batch_size: int = 8
    completion_batch_size: int = 32
    max_kv_size: int | None = None
    kv_bits: float | None = None
    kv_group_size: int | None = None
    kv_quant_scheme: str | None = None
    quantized_kv_start: int | None = None

    def __post_init__(self):
        _validate_positive_int(self.max_tokens, "defaults.max_tokens")
        _validate_positive_int(
            self.prefill_batch_size,
            "defaults.prefill_batch_size",
        )
        _validate_positive_int(
            self.completion_batch_size,
            "defaults.completion_batch_size",
        )
        if self.max_kv_size is not None:
            _validate_positive_int(self.max_kv_size, "defaults.max_kv_size")
        if not isinstance(self.sampling, SamplingParams):
            raise TypeError("defaults.sampling must be SamplingParams.")
        stops = _normalize_stop_strings(self.stop_strings)
        object.__setattr__(self, "max_tokens", int(self.max_tokens))
        object.__setattr__(self, "prefill_batch_size", int(self.prefill_batch_size))
        object.__setattr__(
            self,
            "completion_batch_size",
            int(self.completion_batch_size),
        )
        if self.max_kv_size is not None:
            object.__setattr__(self, "max_kv_size", int(self.max_kv_size))
        object.__setattr__(self, "stop_strings", stops)


@dataclass(frozen=True)
class GenerationRequest:
    """One generation request.

    Exactly one of ``prompt`` and ``prompt_token_ids`` must be provided.
    Rendered prompt strings are encoded as-is; chat templating belongs to the
    caller. ``image`` (vision models only) accepts a PIL image, an array, or a
    str/os.PathLike filesystem path or URL, decoded once before grouping; raw
    bytes and file-like objects are rejected. ``audio`` (vision models only) is
    accepted but decodes one request at a time, since no supported mlx-vlm
    release batches it. At most one of ``image`` and ``audio`` per request.
    """

    prompt: str | None = None
    prompt_token_ids: Sequence[int] | None = None
    image: Any | None = None
    audio: Any | None = None
    max_tokens: int | None = None
    sampling: SamplingParams | None = None


@dataclass(frozen=True)
class GenerationResult:
    """One row's outcome: its sampled tokens, their text, and what the prompt cost."""

    token_ids: list[int]
    text: str
    logprobs: list[float]
    finish_reason: Literal["stop", "length", "stop_string"]
    stop_match: str | None = None
    prompt_token_count: int = 0


def _validate_positive_int(value: Any, name: str):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _normalize_stop_strings(stop_strings: Sequence[str] | str | None) -> tuple[str, ...]:
    if stop_strings is None:
        return ()
    if isinstance(stop_strings, str):
        stop_strings = (stop_strings,)
    try:
        stops = tuple(stop_strings)
    except TypeError as exc:
        raise TypeError("stop_strings must be a string or a sequence of strings.") from exc
    if any(not isinstance(stop, str) for stop in stops):
        raise TypeError("Every stop string must be a string.")
    if any(not stop for stop in stops):
        raise ValueError("Stop strings must not be empty.")
    return tuple(dict.fromkeys(stops))


def _validate_text_requests(
    requests: Sequence[GenerationRequest],
    defaults: GenerationDefaults,
) -> list[GenerationRequest]:
    validated = list(requests)
    for index, request in enumerate(validated):
        if not isinstance(request, GenerationRequest):
            raise TypeError(f"requests[{index}] must be GenerationRequest.")
        has_prompt = request.prompt is not None
        has_ids = request.prompt_token_ids is not None
        if has_prompt == has_ids:
            raise ValueError(
                f"requests[{index}] must provide exactly one of prompt or "
                "prompt_token_ids."
            )
        if has_prompt and not isinstance(request.prompt, str):
            raise TypeError(f"requests[{index}].prompt must be a string.")
        if has_prompt and not request.prompt:
            raise ValueError(f"requests[{index}].prompt must not be empty.")
        if has_ids:
            try:
                prompt_ids = list(request.prompt_token_ids)
            except TypeError as exc:
                raise TypeError(
                    f"requests[{index}].prompt_token_ids must be a sequence "
                    "of integers."
                ) from exc
            if not prompt_ids:
                raise ValueError(
                    f"requests[{index}].prompt_token_ids must not be empty."
                )
            if any(
                isinstance(token, bool) or not isinstance(token, Integral)
                for token in prompt_ids
            ):
                raise TypeError(
                    f"requests[{index}].prompt_token_ids must contain only "
                    "integers."
                )
            validated[index] = replace(
                request,
                prompt_token_ids=tuple(int(token) for token in prompt_ids),
            )
        if request.max_tokens is not None:
            _validate_positive_int(request.max_tokens, f"requests[{index}].max_tokens")
        if request.sampling is not None and not isinstance(
            request.sampling, SamplingParams
        ):
            raise TypeError(f"requests[{index}].sampling must be SamplingParams.")
        if request.image is not None or request.audio is not None:
            raise ValueError(
                f"requests[{index}] includes media, but text models accept "
                "text prompts only."
            )
    refused_kv = [
        name for name in _KV_QUANT_CONTROLS if getattr(defaults, name) is not None
    ]
    if refused_kv:
        raise ValueError(
            f"{' and '.join(refused_kv)} are not forwarded by this engine's "
            f"mlx-lm text path (installed: {_installed_mlx_lm_version()}); "
            "omit these controls."
        )
    return validated


def _validate_vlm_requests(
    requests: Sequence[GenerationRequest],
    defaults: GenerationDefaults,
) -> list[GenerationRequest]:
    validated = list(requests)
    for index, request in enumerate(validated):
        if not isinstance(request, GenerationRequest):
            raise TypeError(f"requests[{index}] must be GenerationRequest.")
        if request.prompt is None or request.prompt_token_ids is not None:
            raise ValueError(
                f"requests[{index}] must provide a rendered prompt string; "
                "token-id prompts are text-model only."
            )
        if not isinstance(request.prompt, str):
            raise TypeError(f"requests[{index}].prompt must be a string.")
        if not request.prompt:
            raise ValueError(f"requests[{index}].prompt must not be empty.")
        if request.image is not None and request.audio is not None:
            raise ValueError(
                f"requests[{index}] carries both an image and audio; batched "
                "vision generation takes at most one medium per request."
            )
        if request.max_tokens is not None:
            _validate_positive_int(request.max_tokens, f"requests[{index}].max_tokens")
        if request.sampling is not None and not isinstance(
            request.sampling, SamplingParams
        ):
            raise TypeError(f"requests[{index}].sampling must be SamplingParams.")
        if isinstance(request.image, os.PathLike):
            # mlx-vlm's loader opens str paths only.
            validated[index] = replace(request, image=os.fspath(request.image))
    if defaults.prefill_batch_size > defaults.completion_batch_size:
        # An inverted pair never admits anything on mlx-vlm, so generation would
        # hang. mlx-lm normalizes the two, so this is vision-specific.
        raise ValueError(
            "prefill_batch_size must not exceed completion_batch_size for "
            "batched vision generation (got "
            f"{defaults.prefill_batch_size} > {defaults.completion_batch_size})."
        )
    refused_kv = [
        name for name in _KV_QUANT_CONTROLS if getattr(defaults, name) is not None
    ]
    if refused_kv:
        # Whether these bound anything depends on the release, the rest of the
        # configuration, and the model's own cache classes, which upstream
        # resolves per layer. None is forwarded rather than promise a bound
        # that may not exist.
        raise ValueError(
            f"{' and '.join(refused_kv)} are not forwarded by this engine's "
            "batched vision path; omit these controls, or use model.generate."
        )
    if defaults.max_kv_size is not None:
        # No supported BatchGenerator takes a KV-window control, so say so rather
        # than drop a memory constraint the caller believes is in force.
        raise ValueError(
            "max_kv_size is not supported for batched vision generation; "
            f"{_installed_mlx_vlm_version()} exposes no KV-window control on "
            "its batch generator. Omit it, or use model.generate."
        )
    return validated


def _api_shape_error(details: str) -> RuntimeError:
    return RuntimeError(
        "Unsupported mlx-lm batch-generation API shape "
        f"({details}) in {_installed_mlx_lm_version()}. Batched generation "
        "needs a BatchGenerator that streams per-token events, which requires "
        "mlx-lm 0.31.2 or newer; upgrade mlx-lm, or use model.generate for "
        "sequential decoding."
    )


def _probe_text_api(generate_module):
    """Verify the event-level mlx-lm control surface used by this adapter."""

    batch_generator = getattr(generate_module, "BatchGenerator", None)
    generation_batch = getattr(generate_module, "GenerationBatch", None)
    response_type = getattr(generation_batch, "Response", None)
    if batch_generator is None or response_type is None:
        raise _api_shape_error("BatchGenerator or GenerationBatch.Response missing")
    for name in ("insert", "next_generated", "remove", "close"):
        if not callable(getattr(batch_generator, name, None)):
            raise _api_shape_error(f"BatchGenerator.{name} missing")

    try:
        constructor_signature = inspect.signature(batch_generator)
        insert_parameters = inspect.signature(batch_generator.insert).parameters
        insert_signature = inspect.signature(batch_generator.insert)
        next_signature = inspect.signature(batch_generator.next_generated)
        remove_signature = inspect.signature(batch_generator.remove)
        close_signature = inspect.signature(batch_generator.close)
    except AttributeError as exc:
        raise _api_shape_error("required BatchGenerator method missing") from exc
    except (TypeError, ValueError) as exc:
        raise _api_shape_error("BatchGenerator callable signature unavailable") from exc

    required_constructor = {
        "max_tokens",
        "stop_tokens",
        "prefill_batch_size",
        "completion_batch_size",
        "max_kv_size",
    }
    missing_constructor = required_constructor.difference(
        constructor_signature.parameters
    )
    if missing_constructor:
        names = ", ".join(sorted(missing_constructor))
        raise _api_shape_error(f"BatchGenerator constructor missing {names}")
    required_insert = {"prompts", "max_tokens", "samplers", "logits_processors"}
    missing_insert = required_insert.difference(insert_parameters)
    if missing_insert:
        names = ", ".join(sorted(missing_insert))
        raise _api_shape_error(f"BatchGenerator.insert missing {names}")
    try:
        constructor_signature.bind(
            object(),
            max_tokens=1,
            stop_tokens=[],
            prefill_batch_size=1,
            completion_batch_size=1,
            max_kv_size=None,
        )
        insert_signature.bind(
            object(),
            [[1]],
            max_tokens=[1],
            samplers=[None],
            logits_processors=[[]],
        )
        next_signature.bind(object())
        remove_signature.bind(object(), [0])
        close_signature.bind(object())
    except TypeError as exc:
        raise _api_shape_error("BatchGenerator call signatures are incompatible") from exc

    fields = set(getattr(response_type, "__dataclass_fields__", ()))
    if not fields:
        fields = set(getattr(response_type, "__annotations__", ()))
    missing_fields = _TEXT_EVENT_FIELDS.difference(fields)
    if missing_fields:
        names = ", ".join(sorted(missing_fields))
        raise _api_shape_error(f"GenerationBatch.Response missing {names}")


def _probe_sampler_api(sample_utils_module):
    make_sampler = getattr(sample_utils_module, "make_sampler", None)
    if not callable(make_sampler):
        raise _api_shape_error("sample_utils.make_sampler missing")
    try:
        signature = inspect.signature(make_sampler)
    except (TypeError, ValueError) as exc:
        raise _api_shape_error("make_sampler signature unavailable") from exc
    required = {"temp", "top_p", "top_k", "min_p"}
    missing = required.difference(signature.parameters)
    if missing:
        names = ", ".join(sorted(missing))
        raise _api_shape_error(f"make_sampler missing {names}")
    try:
        signature.bind(temp=0.0, top_p=0.0, top_k=0, min_p=0.0)
    except TypeError as exc:
        raise _api_shape_error("make_sampler call signature is incompatible") from exc


def _draws_seeded(sampling: SamplingParams) -> bool:
    """Whether this request's seed changes anything: argmax draws nothing."""
    return sampling.seed is not None and sampling.temperature != 0


def _seeded_sampler(sample_utils_module, params: SamplingParams):
    """``make_sampler``'s chain drawing from a request key instead of global RNG."""
    import mlx.core as mx

    if params.temperature == 0:
        return lambda logprobs: mx.argmax(logprobs, axis=-1)

    stages = []
    if 0 < params.top_p < 1.0:
        stages.append(lambda x: sample_utils_module.apply_top_p(x, params.top_p))
    if params.min_p != 0.0:
        stages.append(lambda x: sample_utils_module.apply_min_p(x, params.min_p, 1))
    if params.top_k > 0:
        stages.append(lambda x: sample_utils_module.apply_top_k(x, params.top_k))

    state = {"key": mx.random.key(params.seed)}

    def sampler(logprobs):
        for stage in stages:
            logprobs = stage(logprobs)
        state["key"], subkey = mx.random.split(state["key"])
        return mx.random.categorical(logprobs * (1 / params.temperature), key=subkey)

    return sampler


def _sampler_key(params: SamplingParams):
    """The settings one sampler can serve. A seed is not among them: rows"""
    return (params.temperature, params.top_p, params.top_k, params.min_p)


def _sampler_for(sample_utils_module, params: SamplingParams, make_sampler):
    """The sampler one request's settings ask for."""
    if _draws_seeded(params):
        return _seeded_sampler(sample_utils_module, params)
    return make_sampler(
        temp=params.temperature,
        top_p=params.top_p,
        top_k=params.top_k,
        min_p=params.min_p,
    )


class _PerRowSampler:
    """One sampler per row for a backend that takes only one for the batch."""

    def __init__(self, params, *, sample_utils_module, make_sampler):
        self._sample_utils = sample_utils_module
        self._make_sampler = make_sampler
        self._params = list(params)
        self._by_uid: dict[Any, Any] = {}
        self._samplers: dict[Any, Any] = {}
        self._generator = None
        self.row_uids: list[Any] = []

    def bind_uids(self, uids) -> None:
        if len(uids) != len(self._params):
            raise RuntimeError(
                f"batch admitted {len(uids)} rows for {len(self._params)} "
                "requests; per-row sampling cannot be matched to rows."
            )
        self._by_uid.clear()
        self.row_uids.clear()
        for uid, params in zip(uids, self._params):
            self.register(uid, params)

    def register(self, uid, params: SamplingParams) -> None:
        """Take one more row, for a caller that learns them one at a time."""
        self._by_uid[uid] = params
        self.row_uids.append(uid)

    def release(self, uid) -> None:
        """Forget a row the batch can no longer draw."""
        self._by_uid.pop(uid, None)
        self._samplers.pop(uid, None)
        if uid in self.row_uids:
            self.row_uids.remove(uid)

    def release_all(self) -> None:
        """Forget every row at once, for a batch that has gone."""
        self._by_uid.clear()
        self._samplers.clear()
        self.row_uids.clear()

    def bind_generator(self, generator) -> None:
        self._generator = generator

    def _row_sampler(self, uid):
        sampler = self._samplers.get(uid)
        if sampler is None:
            params = self._by_uid.get(uid)
            if params is None:
                raise RuntimeError(
                    f"{_installed_mlx_vlm_version()} drew a row this batch was "
                    "not given, so per-row sampling cannot be matched to rows."
                )
            sampler = _sampler_for(self._sample_utils, params, self._make_sampler)
            self._samplers[uid] = sampler
        return sampler

    def _drawing_uids(self, width, positions):
        """The rows of this draw, in the order their logprobs are stacked."""

        if self._generator is not None and positions is not None:
            name = (
                "_prompt_batch"
                if all(position == 0 for position in positions)
                else "_generation_batch"
            )
            batch = getattr(self._generator, name, _MISSING)
            if batch is not _MISSING:
                uids = list(getattr(batch, "uids", ()) or ())
                if len(uids) != width:
                    raise RuntimeError(
                        f"{_installed_mlx_vlm_version()} drew {width} rows from a "
                        f"{name.lstrip('_').replace('_', ' ')} holding {len(uids)}, "
                        "so per-row sampling cannot be matched to rows."
                    )
                return uids
        if width != len(self.row_uids):
            raise RuntimeError(
                f"{_installed_mlx_vlm_version()} stepped a batch of {width} rows "
                f"while {len(self.row_uids)} were tracked as live, so per-row "
                "sampling cannot be matched to rows. Give every request in a "
                "batch the same sampling settings and no seed, or pin a release "
                "whose batch admits and retires rows through its event stream."
            )
        return self.row_uids

    def sample_target(self, logprobs, *, row_ids=None, positions=None):
        return self._draw(logprobs, positions)

    def __call__(self, logprobs):
        return self._draw(logprobs, None)

    def _draw(self, logprobs, positions):
        import mlx.core as mx

        uids = self._drawing_uids(logprobs.shape[0], positions)
        return mx.concatenate(
            [
                self._row_sampler(uid)(logprobs[row : row + 1])
                for row, uid in enumerate(uids)
            ],
            axis=0,
        )


def _iter_model_modules(model) -> list[Any]:
    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        modules = [module for _, module in named_modules()]
        if modules:
            return modules
    return [model]


def _snapshot_training_flags(model) -> list[tuple[Any, bool]]:
    states = []
    seen = set()
    for module in _iter_model_modules(model):
        if id(module) in seen or not hasattr(module, "training"):
            continue
        seen.add(id(module))
        states.append((module, bool(module.training)))
    return states


def _restore_training_flags(states: Sequence[tuple[Any, bool]]):
    first_error = None
    for module, training in states:
        try:
            set_mode = getattr(module, "_set_training_mode", None)
            if callable(set_mode):
                set_mode(training)
            elif isinstance(getattr(type(module), "training", None), property):
                # mlx.nn.Module exposes training read-only; _set_training_mode
                # only arrived in mlx 0.30.2, so write the backing attribute.
                module._training = training
            else:
                setattr(module, "training", training)
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _snapshot_metal_limits() -> dict[str, int]:
    """Read process-global Metal limits through their return-previous setters."""

    import mlx.core as mx

    if not mx.metal.is_available():
        return {}
    snapshot = {}
    try:
        for name in ("memory", "cache", "wired"):
            setter = getattr(mx, f"set_{name}_limit")
            previous = setter(0)
            snapshot[name] = int(previous)
            setter(previous)
    except BaseException:
        _restore_metal_limits(snapshot)
        raise
    return snapshot


def _restore_metal_limits(snapshot: dict[str, int] | None):
    if not snapshot:
        return
    import mlx.core as mx

    first_error = None
    for name in ("memory", "cache", "wired"):
        if name in snapshot:
            try:
                getattr(mx, f"set_{name}_limit")(int(snapshot[name]))
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
    if first_error is not None:
        raise first_error


@contextmanager
def generation_mode(model):
    """Temporarily switch a model to eval mode and steward global MLX limits.

    The outermost context snapshots the process-global memory, cache, and wired
    limits. Nested contexts only track their own model flags; the outer context
    restores the limits after all nested generation work completes. Gradient
    checkpointing patches are intentionally left untouched because they are
    class-level and may belong to a live compiled training step.
    """

    global _GENERATION_MODE_DEPTH, _GENERATION_LIMIT_SNAPSHOT

    _GENERATION_MODE_LOCK.acquire()
    training_states: list[tuple[Any, bool]] = []
    entered = False
    active_error = None
    try:
        if _GENERATION_MODE_DEPTH == 0:
            _GENERATION_LIMIT_SNAPSHOT = _snapshot_metal_limits()
        training_states = _snapshot_training_flags(model)
        eval_model = getattr(model, "eval", None)
        if not callable(eval_model):
            raise TypeError("generation_mode requires a model with eval().")
        eval_model()
        _GENERATION_MODE_DEPTH += 1
        entered = True
        yield model
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        restoration_error = None
        try:
            if training_states:
                _restore_training_flags(training_states)
        except BaseException as exc:
            restoration_error = exc
        if entered:
            _GENERATION_MODE_DEPTH -= 1
        try:
            if _GENERATION_MODE_DEPTH == 0:
                snapshot = _GENERATION_LIMIT_SNAPSHOT
                _GENERATION_LIMIT_SNAPSHOT = None
                try:
                    _restore_metal_limits(snapshot)
                except BaseException as exc:
                    if restoration_error is None:
                        restoration_error = exc
        finally:
            _GENERATION_MODE_LOCK.release()
        if restoration_error is not None:
            if active_error is None:
                raise restoration_error
            try:
                warnings.warn(
                    "generation_mode could not fully restore model or MLX "
                    f"state after {type(active_error).__name__}: "
                    f"{restoration_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            except BaseException:
                pass


# Speculative decoding's stream belongs here but not in _VLM_STREAM_MODULES, which may
# only hold modules exporting wired_limit.
def _drain_stream_modules():
    # A function, not a constant: _VLM_STREAM_MODULES is defined further down.
    return ("mlx_lm.generate",) + _VLM_STREAM_MODULES + ("mlx_vlm.speculative.common",)


def _drain_generation_streams(mx):
    """Drain the generation streams, best effort, before the caller clears the cache.

    A no-argument mx.synchronize() waits on the default stream, not these. Best effort
    because they can fail: at the mlx-vlm pin floor generation_stream is a plain
    mx.new_stream, which raises when synchronized off its creating thread, and losing
    the drain must never cost the caller its clear. Only effective on the generating
    thread: elsewhere a thread-local stream drains nothing instead of raising.
    """

    synchronize = getattr(mx, "synchronize", None)
    if not callable(synchronize):
        return
    drained = []
    for name in _drain_stream_modules():
        try:
            # sys.modules only, so cleanup never imports mlx-vlm; a module __getattr__
            # runs arbitrary code and getattr swallows only AttributeError.
            stream = getattr(sys.modules.get(name), "generation_stream", None)
        except Exception:
            continue
        # 0.6.x aliases one object across every mlx_vlm.generate name.
        if stream is None or any(stream is seen for seen in drained):
            continue
        drained.append(stream)
        try:
            synchronize(stream)
        except Exception:
            continue
    try:
        synchronize()
    except Exception:
        pass


@contextmanager
def _generation_cache_hygiene():
    """Cache hygiene around one burst; limits stay owned by ``generation_mode``."""

    import mlx.core as mx

    clear_cache = getattr(mx, "clear_cache", None)
    if not callable(clear_cache):
        raise RuntimeError(
            "Unsupported MLX runtime API shape (mlx.core.clear_cache missing)."
        )
    # MLX pins buffers a live command buffer reads, but not an output array dropped mid-flight.
    _drain_generation_streams(mx)
    clear_cache()
    active_error = None
    try:
        yield
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        try:
            _drain_generation_streams(mx)
            clear_cache()
        except BaseException as cleanup_error:
            if active_error is None:
                raise
            try:
                warnings.warn(
                    "mlx.core.synchronize()/clear_cache() failed while preserving an "
                    f"active {type(active_error).__name__}: {cleanup_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            except BaseException:
                pass


_ARRAYS_CACHE_ADVANCE_LOCK = threading.Lock()
_ARRAYS_CACHE_ADVANCE_RESOLVED = False
_VLM_ARRAYS_CACHE_ADVANCE_RESOLVED = False


def _adopt_deferred_metadata(cache):
    """Move a cache built before the patch onto the deferred slots.

    Mediated fields only: the two properties are installed one at a time, and moving
    the other one mid-window would strip it into a slot nothing reads.
    """

    state = cache.__dict__
    for name in ("lengths", "left_padding"):
        if name in state and isinstance(getattr(type(cache), name, None), property):
            state["_" + name] = state.pop(name)
            state["_" + name + "_pending"] = 0


def _deferred_metadata(name):
    stored = "_" + name
    pending = stored + "_pending"

    def read(self):
        _adopt_deferred_metadata(self)
        value = getattr(self, stored)
        outstanding = getattr(self, pending)
        if outstanding and value is not None:
            value = value - outstanding
            setattr(self, stored, value)
            setattr(self, pending, 0)
        return value

    def write(self, value):
        _adopt_deferred_metadata(self)
        setattr(self, stored, value)
        setattr(self, pending, 0)

    return property(read, write)


def _deferred_advance(self, N):
    _adopt_deferred_metadata(self)
    if not isinstance(N, int):
        # Stock takes anything the subtraction takes, including a per-row array;
        # counting that would force the sync this patch exists to avoid.
        if self._lengths is not None:
            self.lengths = self.lengths - N
        if self._left_padding is not None:
            self.left_padding = self.left_padding - N
        return
    if self._lengths is not None:
        self._lengths_pending += N
    if self._left_padding is not None:
        self._left_padding_pending += N


# Never installed, never called: compiled body + signature identify the implementation
# this patch reproduces, independent of source files and formatting.
def _stock_advance(self, N):
    if self.lengths is not None:
        self.lengths -= N
    if self.left_padding is not None:
        self.left_padding -= N


def _advance_identity(function):
    code = function.__code__
    return (
        inspect.signature(function),
        code.co_code,
        code.co_consts,
        code.co_names,
        code.co_varnames,
    )


def _has_replaceable_advance(arrays_cache):
    """Identify the exact ``advance`` this patch reproduces."""

    if any(
        hasattr(type(inspect.getattr_static(arrays_cache, name, None)), "__set__")
        for name in ("lengths", "left_padding")
    ):
        return False
    # A plain read unwraps staticmethod, judging a body bound differently from ours.
    if not isinstance(
        inspect.getattr_static(arrays_cache, "advance", None), types.FunctionType
    ):
        return False
    return _advance_identity(arrays_cache.advance) == _advance_identity(_stock_advance)


def _install_deferred_metadata(arrays_cache):
    """Install the deferred slots, descriptors and ``advance`` on one cache class."""

    arrays_cache._lengths = None
    arrays_cache._lengths_pending = 0
    arrays_cache._left_padding = None
    arrays_cache._left_padding_pending = 0
    arrays_cache.lengths = _deferred_metadata("lengths")
    arrays_cache.left_padding = _deferred_metadata("left_padding")
    # Marker + original, so a patched class is recognisable when diffing upstream.
    arrays_cache._unsloth_stock_advance = arrays_cache.advance
    arrays_cache.advance = _deferred_advance
    arrays_cache._unsloth_advance_patched = True


def _install_arrays_cache_advance_fix():
    """Keep mlx-lm's ``ArraysCache.advance`` from stranding a Metal buffer per call.

    ``-=`` rebinds to an unevaluated subtract whose scalar operand owns a live buffer.
    The metadata is absent from ``ArraysCache.state``, so during batched decode only
    the mask-owning cache is forced and every other linear-attention layer strands one
    buffer per token. A Python counter keeps the arithmetic and allocates nothing.

    mlx-vlm needs visiting separately: it re-exports mlx-lm's class up to 0.5.x, vendors
    this same body from 0.6.4, and only defers for itself from 0.6.17. Visited only once
    already imported, so a text-only run does not pull in an optional dependency.
    """

    global _ARRAYS_CACHE_ADVANCE_RESOLVED, _VLM_ARRAYS_CACHE_ADVANCE_RESOLVED
    with _ARRAYS_CACHE_ADVANCE_LOCK:
        if _ARRAYS_CACHE_ADVANCE_RESOLVED and _VLM_ARRAYS_CACHE_ADVANCE_RESOLVED:
            return
        seen = []
        if not _ARRAYS_CACHE_ADVANCE_RESOLVED:
            try:
                arrays_cache = importlib.import_module("mlx_lm.models.cache").ArraysCache
                replaceable = _has_replaceable_advance(arrays_cache)
            except Exception:
                # A failed attempt is not a decision: retry on the next call.
                return
            if replaceable:
                _install_deferred_metadata(arrays_cache)
            seen.append(arrays_cache)
            _ARRAYS_CACHE_ADVANCE_RESOLVED = True
        if not _VLM_ARRAYS_CACHE_ADVANCE_RESOLVED and "mlx_vlm" in sys.modules:
            try:
                vlm_cache = importlib.import_module("mlx_vlm.models.cache").ArraysCache
                # Up to mlx-vlm 0.5.x this is mlx-lm's class, already decided above.
                replaceable = vlm_cache not in seen and _has_replaceable_advance(
                    vlm_cache
                )
            except Exception:
                return
            if replaceable:
                _install_deferred_metadata(vlm_cache)
            _VLM_ARRAYS_CACHE_ADVANCE_RESOLVED = True


@dataclass(frozen=True)
class GenerationEvent:
    """One row's progress through a batch, in the order the batch produced it."""

    index: int
    delta: str = ""
    result: GenerationResult | None = None


def _text_events(index: int, state: "_PendingResult") -> Iterator[GenerationEvent]:
    delta = state.release()
    if delta:
        yield GenerationEvent(index=index, delta=delta)


def _finished_events(
    index: int,
    state: "_PendingResult",
    tokenizer,
) -> Iterator[GenerationEvent]:
    yield GenerationEvent(
        index=index,
        delta=state.release(),
        result=state.result(tokenizer),
    )


class _StopStringScanner:
    """Incrementally search new detokenized text with longest-stop lookback."""

    def __init__(self, stop_strings: Sequence[str]):
        self.stop_strings = tuple(stop_strings)
        self.max_stop_length = max(map(len, self.stop_strings), default=0)
        self.previous_length = 0

    def feed(self, text: str) -> tuple[int, str] | None:
        if not self.stop_strings:
            self.previous_length = len(text)
            return None
        search_start = max(0, self.previous_length - self.max_stop_length + 1)
        matches = []
        for order, stop in enumerate(self.stop_strings):
            position = text.find(stop, search_start)
            if position >= 0:
                matches.append((position, -len(stop), order, stop))
        self.previous_length = len(text)
        if not matches:
            return None
        position, _, _, stop = min(matches)
        return position, stop


class _DecodeStreamingDetokenizer:
    """Pinned mlx-lm naive-streaming semantics for raw tokenizers."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.reset()

    def reset(self):
        self.offset = 0
        self.tokens: list[int] = []
        self._text = ""
        self._current_tokens: list[int] = []
        self._current_text = ""

    def add_token(self, token: int):
        self._current_tokens.append(token)
        self.tokens.append(token)

    def finalize(self):
        self._text += self.tokenizer.decode(self._current_tokens)
        self._current_tokens = []
        self._current_text = ""

    @property
    def text(self):
        if self._current_tokens:
            self._current_text = self.tokenizer.decode(self._current_tokens)
            if self._current_text.endswith("\ufffd") or (
                getattr(self.tokenizer, "clean_up_tokenization_spaces", False)
                and self._current_text.endswith(" ")
            ):
                self._current_text = self._current_text[:-1]
        if self._current_text.endswith("\n"):
            self._text += self._current_text
            self._current_tokens.clear()
            self._current_text = ""
        return self._text + self._current_text

    @property
    def last_segment(self):
        text = self.text
        if self.offset > len(text):
            # A decode can shorten when a trailing character becomes incomplete.
            # The offset only advances; lowering it would re-emit that tail.
            return ""
        segment = text[self.offset :]
        self.offset = len(text)
        return segment


def _new_detokenizer(tokenizer, *, require_independent: bool = False):
    """A detokenizer owned by one sequence.

    Some processors return one shared instance every time, which would
    interleave concurrent sequences into a single text buffer.
    """

    try:
        detokenizer = tokenizer.detokenizer
    except (AttributeError, TypeError):
        detokenizer = None
    if detokenizer is None:
        return _DecodeStreamingDetokenizer(tokenizer)
    if require_independent:
        try:
            if tokenizer.detokenizer is detokenizer:
                return _DecodeStreamingDetokenizer(tokenizer)
        except (AttributeError, TypeError):
            return _DecodeStreamingDetokenizer(tokenizer)
    reset = getattr(detokenizer, "reset", None)
    if callable(reset):
        reset()
    return detokenizer


def _replay_stream_text(detokenizer, token_ids: Sequence[int]) -> str:
    detokenizer.reset()
    text = ""
    for token in token_ids:
        detokenizer.add_token(int(token))
        text += detokenizer.last_segment
    detokenizer.finalize()
    return text + detokenizer.last_segment


@dataclass
class _PendingResult:
    detokenizer: Any
    scanner: _StopStringScanner
    prompt_token_count: int = 0
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    text: str = ""
    released: int = 0
    finish_reason: Literal["stop", "length", "stop_string"] | None = None
    stop_match: str | None = None

    def release(self) -> str:
        """Text this row has not handed out yet."""
        delta = self.text[self.released :]
        self.released = len(self.text)
        return delta

    def _apply_stop_match(
        self,
        tokenizer,
        match: tuple[int, str],
    ):
        stop_start, self.stop_match = match
        target_prefix = self.text[:stop_start]
        boundary_detokenizer = _new_detokenizer(tokenizer)
        keep = 0
        for token_count in range(len(self.token_ids) - 1, -1, -1):
            candidate_text = _replay_stream_text(
                boundary_detokenizer,
                self.token_ids[:token_count],
            )
            if target_prefix.startswith(candidate_text):
                keep = token_count
                break
        del self.token_ids[keep:]
        del self.logprobs[keep:]
        self.text = _replay_stream_text(self.detokenizer, self.token_ids)
        self.released = min(self.released, len(self.text))
        self.finish_reason = "stop_string"

    def add_terminal(self, token: int, logprob: float):
        self.token_ids.append(token)
        self.logprobs.append(logprob)
        self.detokenizer.add_token(token)

    def append(self, tokenizer, token: int, logprob: float) -> bool:
        self.add_terminal(token, logprob)
        self.text += self.detokenizer.last_segment
        match = self.scanner.feed(self.text)
        if match is None:
            return False
        self._apply_stop_match(tokenizer, match)
        return True

    def finish(self, tokenizer, reason: Literal["stop", "length"]):
        self.detokenizer.finalize()
        self.text += self.detokenizer.last_segment
        match = self.scanner.feed(self.text)
        if match is not None:
            self._apply_stop_match(tokenizer, match)
        else:
            self.finish_reason = reason

    def result(self, tokenizer) -> GenerationResult:
        if self.finish_reason is None:
            raise RuntimeError("Internal error: generation ended without a finish reason.")
        return GenerationResult(
            token_ids=list(self.token_ids),
            text=self.text,
            logprobs=list(self.logprobs),
            finish_reason=self.finish_reason,
            stop_match=self.stop_match,
            prompt_token_count=self.prompt_token_count,
        )


def _encode_prompt(tokenizer, request: GenerationRequest) -> list[int]:
    if request.prompt_token_ids is not None:
        return [int(token) for token in request.prompt_token_ids]
    try:
        prompt_ids = tokenizer.encode(request.prompt, add_special_tokens=False)
    except TypeError:
        prompt_ids = tokenizer.encode(request.prompt)
    prompt_ids = [int(token) for token in prompt_ids]
    if not prompt_ids:
        raise ValueError("Encoded prompts must contain at least one token.")
    return prompt_ids


def _eos_stop_tokens(tokenizer) -> list[list[int]]:
    eos_ids = getattr(tokenizer, "eos_token_ids", None)
    if eos_ids is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        eos_ids = () if eos_id is None else (eos_id,)
    if isinstance(eos_ids, Integral):
        eos_ids = (eos_ids,)
    return [[int(token)] for token in eos_ids if token is not None]


def _event_logprob(event) -> float:
    """Sampled-token logprob, from a scalar field or a vocabulary vector."""

    scalar = getattr(event, "token_logprob", None)
    if scalar is not None:
        return float(scalar)
    return _sampled_logprob(event)


def _sampled_logprob(event) -> float:
    try:
        value = event.logprobs[int(event.token)]
    except (IndexError, KeyError, TypeError) as exc:
        raise RuntimeError(
            "the backend emitted a logprob vector that does not contain the "
            f"sampled token {event.token}."
        ) from exc
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _warn_preserving(
    what: str,
    active_error: BaseException,
    secondary: BaseException,
) -> None:
    """Report a second failure that must not replace the error already in flight."""
    try:
        warnings.warn(
            f"{what} failed while preserving an active "
            f"{type(active_error).__name__}: {secondary}",
            RuntimeWarning,
            stacklevel=3,
        )
    except BaseException:
        pass


class _TextBatchSession:
    """One mlx-lm ``BatchGenerator`` with its row set left open."""

    def __init__(self, adapter: "_TextBatchAdapter"):
        self.adapter = adapter
        defaults = adapter.defaults
        self.generator = adapter.batch_generator_type(
            adapter.model,
            max_tokens=defaults.max_tokens,
            stop_tokens=_eos_stop_tokens(adapter.tokenizer),
            prefill_batch_size=defaults.prefill_batch_size,
            completion_batch_size=defaults.completion_batch_size,
            max_kv_size=defaults.max_kv_size,
        )
        self._row_signature = None
        self._pending: dict[int, _PendingResult] = {}
        self._row_of: dict[int, int] = {}
        self._uid_of: dict[int, int] = {}
        self._next_row = 0
        self.usable = True

    @property
    def rows_in_flight(self) -> int:
        return len(self._pending)

    def add(self, request: GenerationRequest) -> int:
        adapter = self.adapter
        defaults = adapter.defaults
        prompt = _encode_prompt(adapter.tokenizer, request)
        params = request.sampling or defaults.sampling
        sampler = (
            _seeded_sampler(adapter.sample_utils, params)
            if params.seed is not None
            else adapter.make_sampler(
                temp=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                min_p=params.min_p,
            )
        )
        try:
            uids = self.generator.insert(
                [prompt],
                max_tokens=[int(request.max_tokens or defaults.max_tokens)],
                samplers=[sampler],
                logits_processors=[[]],
            )
        except BaseException:
            self.usable = False
            raise
        try:
            if len(uids) != 1:
                raise RuntimeError(
                    f"mlx-lm answered a one-prompt insert with {len(uids)} uids."
                )
            uid = uids[0]
            state = _PendingResult(
                detokenizer=_new_detokenizer(adapter.tokenizer),
                scanner=_StopStringScanner(defaults.stop_strings),
                prompt_token_count=len(prompt),
            )
        except BaseException as active_error:
            try:
                self.generator.remove(list(uids))
            except BaseException as rollback_error:
                self.usable = False
                _warn_preserving(
                    "handing a rejected row back to mlx-lm",
                    active_error,
                    rollback_error,
                )
            raise
        row = self._next_row
        self._next_row += 1
        self._pending[uid] = state
        self._row_of[uid] = row
        self._uid_of[row] = uid
        return row

    def cancel(self, row: int) -> bool:
        """Withdraw a row. False if it was never added, or has already ended."""
        uid = self._uid_of.get(row)
        if uid is None or uid not in self._pending:
            return False
        try:
            self.generator.remove([uid])
        except BaseException:
            self.usable = False
            raise
        self._retire(uid)
        return True

    def step(self) -> Iterator[GenerationEvent]:
        """One decode step's worth of events, or none while no row is in flight."""
        if not self._pending:
            return
        try:
            responses = self.generator.next_generated()
            if not responses:
                raise RuntimeError(
                    "mlx-lm ended its event stream before every request "
                    "reported a finish reason."
                )
            for response in responses:
                yield from self._consume(response)
        except BaseException:
            self.usable = False
            raise

    def close(self):
        self.generator.close()

    def _retire(self, uid: int):
        row = self._row_of.pop(uid, None)
        self._pending.pop(uid, None)
        if row is not None:
            self._uid_of.pop(row, None)

    def _consume(self, response) -> Iterator[GenerationEvent]:
        state = self._pending.get(response.uid)
        if state is None:
            return
        row = self._row_of[response.uid]
        finish_reason = response.finish_reason
        if finish_reason is None:
            stopped = state.append(
                self.adapter.tokenizer,
                int(response.token),
                _sampled_logprob(response),
            )
            if not stopped:
                yield from _text_events(row, state)
                return
            self.generator.remove([response.uid])
            yield from _finished_events(row, state, self.adapter.tokenizer)
            self._retire(response.uid)
            return
        if finish_reason not in ("stop", "length"):
            raise RuntimeError(
                "mlx-lm emitted an unsupported finish reason: "
                f"{finish_reason!r}."
            )
        if finish_reason == "length":
            state.add_terminal(int(response.token), _sampled_logprob(response))
        state.finish(self.adapter.tokenizer, finish_reason)
        yield from _finished_events(row, state, self.adapter.tokenizer)
        self._retire(response.uid)


class _TextBatchAdapter:
    def __init__(self, model, tokenizer, defaults: GenerationDefaults):
        generate_module = importlib.import_module("mlx_lm.generate")
        sample_utils_module = importlib.import_module("mlx_lm.sample_utils")
        _probe_text_api(generate_module)
        _probe_sampler_api(sample_utils_module)
        self.batch_generator_type = generate_module.BatchGenerator
        self.make_sampler = sample_utils_module.make_sampler
        self.sample_utils = sample_utils_module
        self.model = model
        self.tokenizer = tokenizer
        self.defaults = defaults

    def stream(self, requests: Sequence[GenerationRequest]) -> Iterator[GenerationEvent]:
        session = _TextBatchSession(self)
        active_error = None
        try:
            for request in requests:
                session.add(request)
            while session.rows_in_flight:
                yield from session.step()
        except BaseException as exc:
            active_error = exc
            raise
        finally:
            try:
                session.close()
            except BaseException as close_error:
                if active_error is None:
                    raise
                try:
                    warnings.warn(
                        "BatchGenerator.close() failed while preserving an "
                        f"active {type(active_error).__name__}: {close_error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                except BaseException:
                    pass


_VLM_BATCH_MODULES = ("mlx_vlm.generate", "mlx_vlm.generate.ar")
_VLM_STREAM_MODULES = ("mlx_vlm.generate", "mlx_vlm.generate.dispatch", "mlx_vlm.generate.ar")
_MISSING = object()


def _installed_mlx_vlm_version() -> str:
    """Describe the installed mlx-vlm for diagnostics (see the mlx-lm twin)."""

    try:
        from importlib.metadata import version

        installed = version("mlx-vlm")
    except Exception:
        installed = None
    if not isinstance(installed, str) or not installed.strip():
        return "the installed mlx-vlm"
    return f"mlx-vlm {installed.strip().splitlines()[0]}"


def _vlm_api_shape_error(details: str) -> RuntimeError:
    return RuntimeError(
        "Unsupported mlx-vlm batch-generation API shape "
        f"({details}) in {_installed_mlx_vlm_version()}. Batched vision "
        "generation needs a BatchGenerator that streams per-token events; "
        "upgrade or reinstall mlx-vlm, or use model.generate for sequential "
        "decoding."
    )


def _resolve_module_attr(candidates: Sequence[str], attribute: str):
    """First importable candidate module exposing ``attribute``.

    Which module holds a symbol differs by release, and attribute access on the
    package can shadow the submodule, so candidates are imported by name.
    """

    broken = None
    for name in candidates:
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            # Absent candidate, unless what is missing is a dependency of it.
            missing = exc.name
            if broken is None and missing and not name.startswith(missing):
                broken = exc
            continue
        except Exception as exc:
            if broken is None:
                broken = exc
            continue
        if getattr(module, attribute, None) is not None:
            return module
    if broken is not None:
        # A partial install is not an unsupported release; say which it is.
        raise broken
    return None


def _probe_vlm_api(batch_module):
    """Verify the event-level mlx-vlm control surface used by this adapter."""

    generator = getattr(batch_module, "BatchGenerator", None)
    if generator is None:
        raise _vlm_api_shape_error("BatchGenerator missing")
    response = getattr(generator, "Response", None)
    if response is None:
        # Newer releases move the event to the generation batch.
        batch = getattr(batch_module, "GenerationBatch", None)
        response = getattr(batch, "Response", None) if batch is not None else None
    if response is None:
        raise _vlm_api_shape_error("no per-token Response class found")
    fields = set(getattr(response, "__dataclass_fields__", {}))
    required = {"uid", "token", "finish_reason"}
    missing = sorted(required - fields)
    if missing:
        raise _vlm_api_shape_error(
            f"per-token Response lacks {', '.join(missing)}"
        )
    if not fields & {"logprobs", "token_logprob"}:
        raise _vlm_api_shape_error(
            "per-token Response exposes neither logprobs nor token_logprob"
        )
    for name in ("insert", "next"):
        if not callable(getattr(generator, name, None)):
            raise _vlm_api_shape_error(f"BatchGenerator.{name} missing")
    try:
        constructor = inspect.signature(generator.__init__).parameters
        insert = inspect.signature(generator.insert).parameters
    except (TypeError, ValueError) as exc:
        raise _vlm_api_shape_error(
            "BatchGenerator call signatures are unavailable"
        ) from exc
    required = {"prefill_batch_size", "completion_batch_size"}
    absent = sorted(required - set(constructor))
    if absent:
        raise _vlm_api_shape_error(
            f"BatchGenerator constructor missing {', '.join(absent)}"
        )
    return constructor, insert, _resolve_cancel(generator)


def _statements(body):
    """Every node written in this body, skipping nested definitions."""

    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield node
        yield from _statements(list(ast.iter_child_nodes(node)))


def _draws_by_position(owner, method) -> bool:
    """Whether this draw hands the sampler the rows and positions it draws at."""

    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(owner, method))))
    except (AttributeError, OSError, TypeError, SyntaxError, IndentationError):
        return False
    definition = tree.body[0] if tree.body else None
    if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for node in _statements(definition.body):
        if not isinstance(node, ast.Call):
            continue
        called = node.func
        name = (
            called.attr if isinstance(called, ast.Attribute)
            else called.id if isinstance(called, ast.Name)
            else None
        )
        if name != "_sample_with_positions":
            continue
        given = {keyword.arg: keyword.value for keyword in node.keywords}
        if not {"row_ids", "positions"}.issubset(given):
            continue
        if any(
            isinstance(given[argument], ast.Constant) and given[argument].value is None
            for argument in ("row_ids", "positions")
        ):
            continue
        return True
    return False


def _vlm_batches_observable(batch_module) -> bool:
    """Whether a sampler can tell which of a release's rows a draw is for."""

    generator = getattr(batch_module, "BatchGenerator", None)
    code = getattr(getattr(generator, "__init__", None), "__code__", None)
    if not {"_prompt_batch", "_generation_batch"}.issubset(
        getattr(code, "co_names", ())
    ):
        return False
    return _draws_by_position(
        getattr(batch_module, "PromptProcessingBatch", None), "generate"
    ) and _draws_by_position(
        getattr(batch_module, "GenerationBatch", None), "_step"
    )


_LAYER_MAJOR_PROMPT_KWARGS = frozenset({"deepstack_visual_embeds"})


def _padded_prompt_kwargs(batch_module) -> frozenset:
    """The keys this release stretches to the batch's longest prompt."""

    return frozenset({
        "inputs_embeds",
        *(getattr(batch_module, "_SEQUENCE_ALIGNED_PROMPT_KWARGS", None) or ()),
    })


def _row_shape(key, value, padded: frozenset):
    """What a row's value has to match in the batch's other rows."""

    shape = getattr(value, "shape", None)
    if shape is None:
        if isinstance(value, (list, tuple)):
            return (type(value).__name__,
                    tuple(_row_shape(key, item, padded) for item in value))
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return value
        return type(value).__name__
    shape = list(shape)
    if key in padded:
        axis = 2 if key == "position_ids" and len(shape) == 3 else 1
        if axis < len(shape):
            shape[axis] = None
    return tuple(shape)


def _row_signature(prepared: dict, padded: frozenset) -> dict:
    return {key: _row_shape(key, value, padded) for key, value in prepared.items()}


def _own_body(method):
    """One method's own statements, minus anything nested inside it."""

    source = textwrap.dedent(inspect.getsource(method))
    definition = ast.parse(source).body[0]
    if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise TypeError("not a function definition")
    nested = {
        inner for node in ast.walk(definition)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        and node is not definition
        for inner in ast.walk(node)
    }
    return [node for node in ast.walk(definition) if node not in nested]


def _keeps_what_it_prepared(model, method = "get_input_embeddings") -> bool:
    """Whether preparing a request leaves anything on this model.

    Follows the helpers the method calls on itself: several models do the
    assignment one call down, where reading the method alone would not see it.
    A ``self.<name>(...)`` with no readable body is a submodule being run --
    every vision model runs its tower that way -- and is stepped over. Only the
    prepare method itself being unreadable counts as keeping something, since
    then nothing at all is known about it.
    """

    seen = set()

    def follows(name, entry = False) -> bool:
        if name in seen:
            return False
        seen.add(name)
        try:
            body = _own_body(getattr(model, name))
        except (AttributeError, OSError, TypeError, SyntaxError, IndentationError):
            return entry
        calls = set()
        for node in body:
            targets = (
                list(node.targets) if isinstance(node, ast.Assign)
                else [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign))
                else []
            )
            for target in targets:
                while isinstance(target, ast.Attribute):
                    target = target.value
                if isinstance(target, ast.Name) and target.id == "self":
                    return True
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
            ):
                calls.add(node.func.attr)
        return any(follows(call) for call in sorted(calls))

    return follows(method, entry = True)


def _require_streamable_vlm(adapter: "_VLMBatchAdapter") -> None:
    """Refuse a release that cannot keep a vision batch open."""

    if not adapter.per_row_prompt_kwargs:
        raise ValueError(
            f"{_installed_mlx_vlm_version()} admits one set of embeddings for a "
            "whole prefill batch, so a vision row cannot join a batch it was not "
            "preprocessed with. Generate with generate_batch or stream_batch, or "
            "upgrade mlx-vlm."
        )
    if not adapter.batches_observable:
        raise ValueError(
            f"{_installed_mlx_vlm_version()} does not let a sampler tell which "
            "row it is drawing for, which a batch that widens while it decodes "
            "needs. Generate with generate_batch or stream_batch, or upgrade "
            "mlx-vlm."
        )
    if not callable(getattr(adapter.batch_module, "_chunked_prefill_enabled", None)):
        raise ValueError(
            f"{_installed_mlx_vlm_version()} settles how to prefill when the "
            "batch is built, before any prompt has arrived, so a row whose model "
            "needs its prompt prefilled in one pass would be chunked anyway. "
            "Generate with generate_batch or stream_batch, which build a batch "
            "around the prompts it holds, or upgrade mlx-vlm."
        )
    if _keeps_what_it_prepared(adapter.model):
        raise ValueError(
            "This model keeps what it worked out about one request, which a "
            "batch admitting requests one at a time would hand to whichever "
            "prepared last. Generate it with generate_batch or stream_batch, "
            "which prepare a whole batch together."
        )


def _resolve_cancel(generator):
    """How this generator cancels one sequence, decided once.

    Retrying the other form on TypeError would re-run a partially applied
    cancel against an already mutated batch.
    """

    remove = getattr(generator, "remove", None)
    if not callable(remove):
        return None
    try:
        params = [
            name
            for name, param in inspect.signature(remove).parameters.items()
            if name != "self"
            and param.kind
            in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        return None
    if not params:
        return None
    plural = params[0].endswith("s")
    return (lambda gen, uid: gen.remove([uid])) if plural else (
        lambda gen, uid: gen.remove(uid)
    )


def _is_mrope_position_ids(key: str, value) -> bool:
    shape = getattr(value, "shape", None)
    return (key == "position_ids" and getattr(value, "ndim", 0) == 3
            and shape is not None and shape[0] == 3)


def _split_prompt_kwargs_fallback(prompt_kwargs: dict, batch_size: int) -> list[dict]:
    """Per-row split for releases without upstream's splitter.

    MRoPE ``position_ids`` carry the batch on axis 1, so an unconditional
    axis-0 slice would corrupt them.
    """

    if batch_size <= 1:
        return [dict(prompt_kwargs or {})]
    rows: list[dict] = [{} for _ in range(batch_size)]
    for key, value in (prompt_kwargs or {}).items():
        ndim = getattr(value, "ndim", None)
        if ndim is None or ndim == 0:
            for row in rows:
                row[key] = value
            continue
        if _is_mrope_position_ids(key, value):
            for index in range(batch_size):
                rows[index][key] = (
                    value[:, index : index + 1, :]
                    if value.shape[1] == batch_size
                    else value[:, :1, :]
                )
            continue
        for index in range(batch_size):
            rows[index][key] = (
                value[index : index + 1]
                if value.shape[0] == batch_size
                else value[:1]
            )
    return rows


class _VLMBatchAdapter:
    """Event-level adapter over mlx-vlm's BatchGenerator.

    Bypasses the public batch helper, which templates the prompt and discards
    per-token data, and reproduces what it owns: image-shape grouping, the
    release's chunked-prefill policy, and the wired-limit window.
    """

    def __init__(
        self, model, processor, defaults: GenerationDefaults, *, audio_warn_stacklevel = 3,
    ):
        batch_module = _resolve_module_attr(_VLM_BATCH_MODULES, "BatchGenerator")
        if batch_module is None:
            raise _vlm_api_shape_error("no importable module exposes BatchGenerator")
        # Bind the module the class is DEFINED in: that is where the private
        # helpers and event class live, however the package re-exports them.
        defining = getattr(batch_module.BatchGenerator, "__module__", None)
        if defining and defining != batch_module.__name__:
            try:
                batch_module = importlib.import_module(defining)
            except Exception:
                pass
        (
            self.constructor_params,
            insert_params,
            self.cancel,
        ) = _probe_vlm_api(batch_module)
        self.batch_module = batch_module
        self.generator_type = batch_module.BatchGenerator
        self.per_row_prompt_kwargs = "prompt_kwargs" in insert_params
        self.batches_observable = _vlm_batches_observable(batch_module)
        self.padded_prompt_kwargs = _padded_prompt_kwargs(batch_module)
        self.stream_module = _resolve_module_attr(_VLM_STREAM_MODULES, "wired_limit")
        utils = importlib.import_module("mlx_vlm.utils")
        self.prepare_inputs = utils.prepare_inputs
        self.process_image = getattr(utils, "process_image", None)
        sample_utils = importlib.import_module("mlx_lm.sample_utils")
        _probe_sampler_api(sample_utils)
        self.make_sampler = sample_utils.make_sampler
        self.sample_utils = sample_utils
        self.model = model
        self.processor = processor
        self.defaults = defaults
        self.audio_warn_stacklevel = audio_warn_stacklevel

    def _wired_limit(self):
        """Raise the wired limit, which ``generation_mode`` restores but never raises."""

        module = self.stream_module
        if module is None:
            return _null_context()
        # Unconditional: the chunked-prefill policy needs embeddings before the
        # generator exists, so a release whose constructor raises the limit
        # would otherwise embed under the trainer's cap. Those releases take one
        # redundant nested raise.
        stream = getattr(module, "generation_stream", None)
        try:
            return module.wired_limit(self.model, [stream] if stream else None)
        except TypeError:
            # Only a signature change reaches here: wired_limit is a context
            # manager, so it sets the limit at __enter__, not on this call.
            warnings.warn(
                f"{_installed_mlx_vlm_version()} exposes an incompatible "
                "wired_limit(); batched vision generation runs under the "
                "caller's wired-memory limit.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _null_context()

    def _add_special_tokens(self) -> bool:
        config = getattr(self.model, "config", None)
        model_type = getattr(config, "model_type", None)
        if model_type in ("gemma3", "gemma3n", "gemma4", "gemma4_unified"):
            return getattr(self.processor, "chat_template", None) is None
        return True

    def _chunked_prefill_kwargs(self, *, input_ids=None, prefill_kwargs=None) -> dict:
        """Apply the release's chunked-prefill policy, which needs real inputs."""

        enabled = getattr(self.batch_module, "_chunked_prefill_enabled", None)
        if callable(enabled):
            embeds = (prefill_kwargs or {}).get("inputs_embeds")
            # Every release shipping this helper takes the full keyword form, so
            # any error is a real policy failure and propagates.
            allowed = enabled(
                self.model,
                input_ids=input_ids,
                inputs_embeds=embeds,
                prefill_kwargs=prefill_kwargs,
            )
            return {} if allowed else {"prefill_step_size": None}
        if getattr(self.model, "no_chunked_prefill", False):
            return {"prefill_step_size": None}
        return {}

    def _split_prompt_kwargs(self, prompt_kwargs: dict, batch_size: int) -> list[dict]:
        upstream = getattr(self.batch_module, "_split_prompt_kwargs_per_row", None)
        mrope_aware = getattr(
            self.batch_module,
            "_is_mrope_position_ids_prompt_kwarg",
            None,
        )
        # Only the MRoPE-aware upstream splitter is used; an axis-0 one would
        # corrupt position ids.
        if callable(upstream) and callable(mrope_aware):
            return upstream(prompt_kwargs, batch_size)
        return _split_prompt_kwargs_fallback(prompt_kwargs, batch_size)

    def _admission_stalled(self, generator) -> bool | None:
        """True when no prompt can ever be admitted; None when unobservable.

        Prefill legitimately yields eventless polls for as long as a long
        prompt needs, so a stall is judged by whether a prompt batch is in
        flight, never by counting empty polls.
        """

        prompt_batch = getattr(generator, "_prompt_batch", _MISSING)
        generation_batch = getattr(generator, "_generation_batch", _MISSING)
        queued = getattr(generator, "_unprocessed_sequences", _MISSING)
        if _MISSING in (prompt_batch, generation_batch, queued):
            return None
        try:
            return (
                prompt_batch is None
                and len(generation_batch) == 0
                and len(queued) > 0
            )
        except TypeError:
            return None

    def _stall_error(self) -> RuntimeError:
        return RuntimeError(
            "mlx-vlm reported pending generation work but has admitted no "
            "prompt and has no prefill in flight, so generation cannot "
            "progress. This happens when the free completion capacity is "
            "smaller than the prefill batch size; check any batch-size "
            "overrides passed to batched generation."
        )

    def _group_key(self, request: GenerationRequest):
        # Prompt length is part of the key: preprocessing stacks a group without
        # padding, so mixing lengths raises instead of generating. Grouping by
        # length keeps varied batches working, at the cost of smaller groups.
        shape = self._image_shape(request.image)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        key = (request.image is None, shape, len(tokenizer.encode(request.prompt)))
        if self.batches_observable:
            return key
        sampling = request.sampling or self.defaults.sampling
        return (_sampler_key(sampling), *key)

    def _decode_image(self, request: GenerationRequest) -> GenerationRequest:
        """Load a path or URL image once, before grouping.

        A second fetch would break one-use URLs and could return a different
        size than grouping saw.
        """

        image = request.image
        if image is None or not isinstance(image, (str, bytes)):
            return request
        if isinstance(image, bytes) or self.process_image is None:
            raise ValueError(
                "Batched vision generation accepts a PIL image, an array, or a "
                "str/os.PathLike path or URL; raw bytes and file-like objects "
                "are not supported."
            )
        try:
            decoded = self.process_image(
                image,
                None,
                getattr(self.processor, "image_processor", None),
            )
        except Exception as exc:
            raise ValueError(
                f"Could not load the image {image!r} for batched vision "
                "generation."
            ) from exc
        return replace(request, image=decoded)

    def _image_shape(self, image):
        """Spatial shape of an already-decoded request image."""

        if image is None:
            return None
        shape = getattr(image, "shape", None)
        if shape is not None:
            return tuple(shape)
        size = getattr(image, "size", None)
        if isinstance(size, tuple):
            return size
        if isinstance(size, list):
            return tuple(size)
        raise ValueError(
            "Batched vision generation needs an image whose size is known "
            "after decoding: pass a PIL image, an array, or a path/URL. "
            f"Got {type(image).__name__}."
        )

    def stream(self, requests: Sequence[GenerationRequest]) -> Iterator[GenerationEvent]:
        requests = [self._decode_image(request) for request in requests]
        audio_rows = [
            index for index, request in enumerate(requests) if request.audio is not None
        ]
        if audio_rows:
            warnings.warn(
                f"{len(audio_rows)} audio request(s) decode one at a time: "
                f"{_installed_mlx_vlm_version()} batches text and images only.",
                RuntimeWarning,
                stacklevel = self.audio_warn_stacklevel,
            )
        for index in audio_rows:
            yield from self._stream_sequentially(index, requests[index])
        groups: dict[Any, list[int]] = {}
        for index, request in enumerate(requests):
            if request.audio is None:
                groups.setdefault(self._group_key(request), []).append(index)
        for indices in groups.values():
            inferred_rows = not self.batches_observable and any(
                _draws_seeded(requests[index].sampling or self.defaults.sampling)
                for index in indices
            )
            capacity = (
                None
                if self.per_row_prompt_kwargs and not inferred_rows
                else min(
                    self.defaults.prefill_batch_size,
                    self.defaults.completion_batch_size,
                )
            )
            step = capacity or len(indices)
            for start in range(0, len(indices), step):
                yield from self._run_chunk(requests, indices[start : start + step])

    def _stream_sequentially(
        self,
        index: int,
        request: GenerationRequest,
    ) -> Iterator[GenerationEvent]:
        """One audio request through the release's sequential stream.

        The stream's terminal event carries no new token: it repeats the last
        one on budget exhaustion, or is the stopping token, which this contract
        excludes. So an event is committed only once a successor arrives, and
        the terminal one contributes text alone.
        """

        stream = getattr(self.stream_module, "stream_generate", None)
        if not callable(stream):
            raise _vlm_api_shape_error("no importable module exposes stream_generate")
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        sampling = request.sampling or self.defaults.sampling
        state = _PendingResult(
            detokenizer=_new_detokenizer(tokenizer, require_independent=True),
            scanner=_StopStringScanner(self.defaults.stop_strings),
        )
        previous = None
        # No wrap here: the sequential stream owns its own wired-limit window.
        events = stream(
            self.model,
            self.processor,
            request.prompt,
            audio=request.audio,
            max_tokens=int(request.max_tokens or self.defaults.max_tokens),
            sampler=(
                _seeded_sampler(self.sample_utils, sampling)
                if _draws_seeded(sampling)
                else self.make_sampler(
                    temp=sampling.temperature,
                    top_p=sampling.top_p,
                    top_k=sampling.top_k,
                    min_p=sampling.min_p,
                )
            ),
        )
        for event in events:
            if previous is None:
                state.prompt_token_count = int(getattr(event, "prompt_tokens", 0) or 0)
            if previous is not None and state.append(
                tokenizer,
                int(previous.token),
                _event_logprob(previous),
            ):
                yield from _finished_events(index, state, tokenizer)
                return
            if previous is not None:
                yield from _text_events(index, state)
            previous = event
        if previous is None:
            raise RuntimeError(
                "mlx-vlm produced no events for an audio request."
            )
        state.finish(tokenizer, self._terminal_reason(previous, tokenizer))
        yield from _finished_events(index, state, tokenizer)

    @staticmethod
    def _terminal_reason(event, tokenizer) -> Literal["stop", "length"]:
        reason = getattr(event, "finish_reason", None)
        if reason in ("stop", "length"):
            return reason
        if reason is not None:
            raise RuntimeError(
                f"mlx-vlm emitted an unsupported finish reason: {reason!r}."
            )
        token = getattr(event, "token", None)
        if token is None:
            return "length"
        # Ask the criteria object that actually ended the loop: processors carry
        # stop ids the tokenizer's EOS attributes do not list, so rebuilding the
        # set would report a genuine stop as a length cut-off.
        criteria = getattr(tokenizer, "stopping_criteria", None)
        if callable(criteria):
            try:
                return "stop" if criteria(int(token)) else "length"
            except Exception:
                pass
        eos = {ids[0] for ids in _eos_stop_tokens(tokenizer)}
        return "stop" if int(token) in eos else "length"

    def _run_chunk(
        self,
        requests: Sequence[GenerationRequest],
        indices: Sequence[int],
    ) -> Iterator[GenerationEvent]:
        chunk = [requests[index] for index in indices]
        batch_size = len(chunk)
        prompts = [request.prompt for request in chunk]
        images = [request.image for request in chunk if request.image is not None]
        row_sampling = [
            request.sampling or self.defaults.sampling for request in chunk
        ]
        sampling = row_sampling[0]
        config = getattr(self.model, "config", None)
        inputs = self.prepare_inputs(
            self.processor,
            images=images or None,
            audio=None,
            prompts=prompts,
            image_token_index=getattr(config, "image_token_index", None),
            resize_shape=None,
            add_special_tokens=self._add_special_tokens(),
            pad_to_uniform_size=False,
        )
        input_ids = inputs.get("input_ids")
        pixel_values = inputs.get("pixel_values")
        mask = inputs.get("attention_mask")
        data_kwargs = {
            key: value
            for key, value in inputs.items()
            if key not in ("input_ids", "pixel_values", "attention_mask")
        }
        options = {
            "prefill_batch_size": (
                self.defaults.prefill_batch_size
                if self.per_row_prompt_kwargs
                else batch_size
            ),
            "completion_batch_size": (
                self.defaults.completion_batch_size
                if self.per_row_prompt_kwargs
                else batch_size
            ),
            "sampler": self.make_sampler(
                temp=sampling.temperature,
                top_p=sampling.top_p,
                top_k=sampling.top_k,
                min_p=sampling.min_p,
            ),
        }
        row_sampler = None
        if any(_draws_seeded(params) for params in row_sampling) or len(
            {_sampler_key(params) for params in row_sampling}
        ) > 1:
            row_sampler = _PerRowSampler(
                row_sampling,
                sample_utils_module=self.sample_utils,
                make_sampler=self.make_sampler,
            )
            options["sampler"] = row_sampler
        if "compute_logprobs" in self.constructor_params:
            options["compute_logprobs"] = True
        max_tokens = [
            int(request.max_tokens or self.defaults.max_tokens) for request in chunk
        ]
        generator = None
        active_error = None
        try:
            with self._wired_limit():
                embedding_output = self.model.get_input_embeddings(
                    input_ids,
                    pixel_values,
                    mask=mask,
                    **data_kwargs,
                )
                gen_kwargs = {**data_kwargs, **{
                    key: value
                    for key, value in embedding_output.to_dict().items()
                    if value is not None
                }}
                options.update(
                    self._chunked_prefill_kwargs(
                        input_ids=input_ids,
                        prefill_kwargs=gen_kwargs,
                    )
                )
                options = {
                    key: value
                    for key, value in options.items()
                    if key in self.constructor_params
                }
                generator = self.generator_type(
                    self.model.language_model,
                    self.processor,
                    **options,
                )
                token_ids = input_ids.tolist()
                if self.per_row_prompt_kwargs:
                    uids = generator.insert(
                        token_ids,
                        max_tokens,
                        prompt_kwargs=self._split_prompt_kwargs(
                            gen_kwargs,
                            batch_size,
                        ),
                    )
                else:
                    uids = generator.insert(token_ids, max_tokens)
                if row_sampler is not None:
                    row_sampler.bind_uids(uids)
                    row_sampler.bind_generator(generator)
                yield from self._drive(
                    generator, uids, indices, gen_kwargs, row_sampler,
                    prompt_token_count=len(token_ids[0]),
                )
        except BaseException as exc:
            active_error = exc
            raise
        finally:
            closer = getattr(generator, "close", None) if generator else None
            if callable(closer):
                try:
                    closer()
                except BaseException as close_error:
                    if active_error is None:
                        raise
                    try:
                        warnings.warn(
                            "BatchGenerator.close() failed while preserving an "
                            f"active {type(active_error).__name__}: {close_error}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    except BaseException:
                        pass

    def _drive(
        self,
        generator,
        uids,
        indices,
        gen_kwargs,
        row_sampler=None,
        prompt_token_count=0,
    ) -> Iterator[GenerationEvent]:
        """Report the chunk's rows. ``prompt_token_count`` is what it prefilled,"""
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        pending = {
            uid: _PendingResult(
                detokenizer=_new_detokenizer(tokenizer, require_independent=True),
                scanner=_StopStringScanner(self.defaults.stop_strings),
                prompt_token_count=prompt_token_count,
            )
            for uid in uids
        }
        row_of = dict(zip(uids, indices))
        live = list(uids)
        while pending:
            if row_sampler is not None:
                row_sampler.row_uids = list(live)
            if self.per_row_prompt_kwargs:
                if not generator.has_work:
                    raise RuntimeError(
                        "mlx-vlm ended its event stream before every request "
                        "reported a finish reason."
                    )
                _, events = generator.next()
            else:
                events = generator.next(**gen_kwargs)
            if not events:
                stalled = self._admission_stalled(generator)
                if stalled:
                    raise self._stall_error()
                if stalled is None and not self.per_row_prompt_kwargs:
                    raise RuntimeError(
                        "mlx-vlm ended its event stream before every request "
                        "reported a finish reason."
                    )
                continue
            for event in events:
                finish_reason = event.finish_reason
                if finish_reason is not None and event.uid in live:
                    live.remove(event.uid)
                state = pending.get(event.uid)
                if state is None:
                    continue
                if finish_reason is None:
                    stopped = state.append(
                        tokenizer,
                        int(event.token),
                        _event_logprob(event),
                    )
                    if not stopped:
                        yield from _text_events(row_of[event.uid], state)
                        continue
                    if self.cancel is not None:
                        self.cancel(generator, event.uid)
                        if event.uid in live:
                            live.remove(event.uid)
                    yield from _finished_events(row_of[event.uid], state, tokenizer)
                    del pending[event.uid]
                    continue
                if finish_reason not in ("stop", "length"):
                    raise RuntimeError(
                        "mlx-vlm emitted an unsupported finish reason: "
                        f"{finish_reason!r}."
                    )
                if finish_reason == "length":
                    state.add_terminal(int(event.token), _event_logprob(event))
                state.finish(tokenizer, finish_reason)
                yield from _finished_events(row_of[event.uid], state, tokenizer)
                del pending[event.uid]


@contextmanager
def _null_context():
    yield


def generate_batch(
    model,
    tokenizer_or_processor,
    requests: Sequence[GenerationRequest],
    *,
    defaults: GenerationDefaults | None = None,
) -> list[GenerationResult]:
    """Generate a batch from a training-resident MLX text or vision model.

    Results preserve input order. ``token_ids`` come directly from the
    backend's sampled-token events, and ``logprobs`` are aligned one-to-one
    with them.
    """

    results: list[GenerationResult | None] = [None] * len(requests)
    for event in _stream_batch(
        model, tokenizer_or_processor, requests, defaults=defaults,
        audio_warn_stacklevel=3,
    ):
        if event.result is not None:
            results[event.index] = event.result
    if any(result is None for result in results):
        raise RuntimeError(
            "Internal error: batched generation produced no result for "
            f"{sum(result is None for result in results)} of {len(results)} requests."
        )
    return results


class _VLMBatchSession:
    """One mlx-vlm ``BatchGenerator`` with its row set left open.

    The vision counterpart of ``_TextBatchSession``, under the same contract:
    rows keep the number they were given, and ``usable`` goes false where a call
    may already have changed the batch.

    A row is let go only once the batch can no longer draw it, since its
    settings go with it. mlx-vlm will not withdraw a row being prefilled beside
    others, so a cancelled one stays the caller's until it is decoding and can
    be withdrawn; where it cannot stay -- its text is already out, or it was
    never recorded -- the batch is no longer answerable and ``usable`` goes
    false.

    Each row is preprocessed alone and admitted with its own embeddings, which
    is what lets rows of unrelated image shapes and prompt lengths merge: the
    release left-pads them to the batch's longest before the prefill forward,
    which preprocessing a whole chunk at once cannot do.

    Preparing rows apart is what the merging costs. It serves only models that
    keep nothing from preparing one request, which is settled before any row is
    admitted, and requests whose preparation lines up with what the batch's
    other rows prepared -- the same values of the same shapes, but for the
    length the release evens out. What a request prepares follows the request
    and not the model, so that second condition is asked of every row.
    """

    def __init__(self, adapter: "_VLMBatchAdapter"):
        self.adapter = adapter
        self.tokenizer = getattr(adapter.processor, "tokenizer", adapter.processor)
        self.sampler = _PerRowSampler(
            (),
            sample_utils_module = adapter.sample_utils,
            make_sampler = adapter.make_sampler,
        )
        self._row_signature = None
        self._pending: dict[int, _PendingResult] = {}
        self._row_of: dict[int, int] = {}
        self._uid_of: dict[int, int] = {}
        self._next_row = 0
        self.usable = True
        self._stack = ExitStack()
        try:
            self._stack.enter_context(adapter._wired_limit())
            self.generator = self._open()
        except BaseException:
            self._stack.close()
            raise

    @property
    def rows_in_flight(self) -> int:
        return len(self._pending)

    def add(self, request: GenerationRequest) -> int:
        adapter = self.adapter
        defaults = adapter.defaults
        if request.audio is not None:
            raise ValueError(
                "An audio request decodes on its own and cannot join a batch; "
                "generate it with generate_batch or stream_batch."
            )
        input_ids, prompt_kwargs = self._prepare(request)
        token_ids = input_ids.tolist()
        try:
            uids = self.generator.insert(
                token_ids,
                [int(request.max_tokens or defaults.max_tokens)],
                prompt_kwargs = adapter._split_prompt_kwargs(prompt_kwargs, 1),
            )
        except BaseException:
            self.usable = False
            raise
        try:
            if len(uids) != 1:
                raise RuntimeError(
                    f"mlx-vlm answered a one-prompt insert with {len(uids)} uids."
                )
            uid = uids[0]
            state = _PendingResult(
                detokenizer = _new_detokenizer(
                    self.tokenizer, require_independent = True,
                ),
                scanner = _StopStringScanner(defaults.stop_strings),
                prompt_token_count = len(token_ids[0]),
            )
        except BaseException as active_error:
            try:
                for admitted in uids:
                    if not self._withdraw(admitted):
                        self.usable = False
            except BaseException as rollback_error:
                self.usable = False
                _warn_preserving(
                    "handing a rejected row back to mlx-vlm",
                    active_error,
                    rollback_error,
                )
            raise
        row = self._next_row
        self._next_row += 1
        self._pending[uid] = state
        self._row_of[uid] = row
        self._uid_of[row] = uid
        self.sampler.register(uid, request.sampling or defaults.sampling)
        self._row_signature = _row_signature(
            prompt_kwargs, adapter.padded_prompt_kwargs)
        return row

    def cancel(self, row: int) -> bool:
        """Withdraw a row. False if it was never added, or has already ended.

        Also false where the batch would not give it back, which mlx-vlm answers
        for a row being prefilled beside others. The row is still the caller's
        then -- it keeps reporting, and cancelling it once it is decoding works
        -- because retiring it here would leave it decoding where nothing can
        see it and nothing can end it.
        """
        uid = self._uid_of.get(row)
        if uid is None or uid not in self._pending:
            return False
        try:
            withdrawn = self._withdraw(uid)
        except BaseException:
            self.usable = False
            raise
        if not withdrawn:
            return False
        self._retire(uid)
        return True

    def step(self) -> Iterator[GenerationEvent]:
        """One decode step's worth of events, or none while no row is in flight."""
        if not self._pending:
            return
        try:
            if not self.generator.has_work:
                raise RuntimeError(
                    "mlx-vlm ended its event stream before every request "
                    "reported a finish reason."
                )
            _, events = self.generator.next()
            if not events:
                if self.adapter._admission_stalled(self.generator):
                    raise self.adapter._stall_error()
                return
            for event in events:
                yield from self._consume(event)
        except BaseException:
            self.usable = False
            raise

    def close(self):
        closer = getattr(self.generator, "close", None)
        try:
            if callable(closer):
                closer()
        finally:
            self.sampler.bind_generator(None)
            self.sampler.release_all()
            self.generator = None
            self._row_signature = None
            self._pending.clear()
            self._row_of.clear()
            self._uid_of.clear()
            self._stack.close()

    def _prepare(self, request: GenerationRequest):
        """One request's token ids and the embeddings it is admitted with."""

        adapter = self.adapter
        request = adapter._decode_image(request)
        config = getattr(adapter.model, "config", None)
        inputs = adapter.prepare_inputs(
            adapter.processor,
            images = None if request.image is None else [request.image],
            audio = None,
            prompts = [request.prompt],
            image_token_index = getattr(config, "image_token_index", None),
            resize_shape = None,
            add_special_tokens = adapter._add_special_tokens(),
            pad_to_uniform_size = False,
        )
        input_ids = inputs.get("input_ids")
        data_kwargs = {
            key: value
            for key, value in inputs.items()
            if key not in ("input_ids", "pixel_values", "attention_mask")
        }
        embedding_output = adapter.model.get_input_embeddings(
            input_ids,
            inputs.get("pixel_values"),
            mask = inputs.get("attention_mask"),
            **data_kwargs,
        )
        embeddings = embedding_output.to_dict()
        prepared = {**data_kwargs, **{
            key: value
            for key, value in embeddings.items()
            if value is not None
        }}
        layered = sorted(_LAYER_MAJOR_PROMPT_KWARGS.intersection(prepared))
        if layered:
            raise ValueError(
                f"This request comes back with {', '.join(layered)} per layer, "
                "which merging lines up as though the layers were rows. "
                "Generate it with generate_batch or stream_batch, which prepare "
                "a whole batch together."
            )
        signature = _row_signature(prepared, adapter.padded_prompt_kwargs)
        if self._row_signature is not None and signature != self._row_signature:
            differing = sorted(
                key for key in set(signature) | set(self._row_signature)
                if signature.get(key, _MISSING) != self._row_signature.get(key, _MISSING)
            )
            raise ValueError(
                f"This request prepares {', '.join(differing)} unlike the "
                "batch's other requests, which merging lines up against rows "
                "that do not match it. Generate it with generate_batch or "
                "stream_batch, which prepare a whole batch together."
            )
        return input_ids, prepared

    def _open(self):
        """The batch, built for the whole session."""

        adapter = self.adapter
        defaults = adapter.defaults
        options = {
            "prefill_batch_size": defaults.prefill_batch_size,
            "completion_batch_size": defaults.completion_batch_size,
            "sampler": self.sampler,
            "compute_logprobs": True,
        }
        generator = adapter.generator_type(
            adapter.model.language_model,
            adapter.processor,
            **{
                key: value
                for key, value in options.items()
                if key in adapter.constructor_params
            },
        )
        self.sampler.bind_generator(generator)
        return generator

    def _withdraw(self, uid: int) -> bool:
        """Hand a row back, and say whether the batch took it."""

        if self.adapter.cancel is None:
            return False
        return self.adapter.cancel(self.generator, uid) is not False

    def _retire(self, uid: int):
        """Let a row go, once the batch can no longer draw it."""

        row = self._row_of.pop(uid, None)
        self._pending.pop(uid, None)
        if row is not None:
            self._uid_of.pop(row, None)
        self.sampler.release(uid)
        if not self._pending:
            self._row_signature = None

    def _consume(self, event) -> Iterator[GenerationEvent]:
        tokenizer = self.tokenizer
        state = self._pending.get(event.uid)
        if state is None:
            return
        row = self._row_of[event.uid]
        finish_reason = event.finish_reason
        if finish_reason is None:
            stopped = state.append(tokenizer, int(event.token), _event_logprob(event))
            if not stopped:
                yield from _text_events(row, state)
                return
            if not self._withdraw(event.uid):
                self.usable = False
            yield from _finished_events(row, state, tokenizer)
            self._retire(event.uid)
            return
        if finish_reason not in ("stop", "length"):
            raise RuntimeError(
                f"mlx-vlm emitted an unsupported finish reason: {finish_reason!r}."
            )
        if finish_reason == "length":
            state.add_terminal(int(event.token), _event_logprob(event))
        state.finish(tokenizer, finish_reason)
        yield from _finished_events(row, state, tokenizer)
        self._retire(event.uid)


class BatchStream:
    """A batch left open: rows join and leave while the batch decodes.

    ``stream_batch`` decodes one fixed set of requests and ends. This is the
    same decode with the set open, so a server can merge requests that arrived
    separately into one batch, and retire each row where it ends rather than at
    the end of the batch. Rows are numbered in the order they were added and
    keep their number for the stream's life.

    The process-wide generation lock is held from construction until ``close``,
    and it belongs to the thread *and* the asyncio task that took it: the same
    one constructs, adds, steps, and closes. Use it as a context manager, or
    close it in a ``finally``. A call from anywhere else is refused before
    anything is torn down, so the stream stays closable by its owner.

    Vision models are served too, on a release whose batch can place a drawn row
    and take a row's embeddings with it; where it cannot, a vision model is
    refused rather than mis-sampled and ``stream_batch`` serves it. Audio rows
    decode on their own and never join.

    ``stop_strings`` are unavailable here for the reason ``stream_batch`` gives:
    they cut on token boundaries, which would retract text already handed out.

    Any call that raises where it may already have changed the batch retires the
    whole stream: the engine takes and releases a row in several steps, so what
    it holds afterwards is a question neither side can answer. ``add``, ``step`` and
    ``cancel`` then refuse; only ``close`` still works. Open another.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        defaults: GenerationDefaults | None = None,
    ):
        if defaults is None:
            defaults = GenerationDefaults()
        if not isinstance(defaults, GenerationDefaults):
            raise TypeError("defaults must be GenerationDefaults.")
        _install_arrays_cache_advance_fix()
        if defaults.stop_strings:
            raise ValueError(
                "BatchStream cannot apply stop_strings: they cut on token "
                "boundaries, which would retract text already streamed. Scan "
                "the deltas for them instead."
            )
        # Routing follows the model, not the request: a vision model
        # preprocesses text-only prompts too.
        is_vlm = bool(getattr(model, "_is_vlm_model", False))
        if is_vlm:
            # A text-only multimodal load stays on the vision path but publishes
            # its inner tokenizer, which cannot drive mlx-vlm preprocessing.
            tokenizer = getattr(model, "_processor", None) or tokenizer
        if tokenizer is None:
            raise ValueError(
                "Batched generation requires a processor."
                if is_vlm
                else "Text batched generation requires a tokenizer."
            )
        self._stack = ExitStack()
        self._session = None
        self._is_vlm = is_vlm
        self._closed = False
        self._owner = (threading.get_ident(), _current_async_task())
        try:
            self._stack.enter_context(generation_mode(model))
            self._stack.enter_context(_generation_cache_hygiene())
            if is_vlm:
                adapter = _VLMBatchAdapter(model, tokenizer, defaults)
                _require_streamable_vlm(adapter)
                self._session = _VLMBatchSession(adapter)
            else:
                self._session = _TextBatchSession(
                    _TextBatchAdapter(model, tokenizer, defaults)
                )
            self._stack.callback(self._session.close)
        except BaseException as active_error:
            try:
                self._stack.close()
            except BaseException as close_error:
                _warn_preserving("tearing the batch down", active_error, close_error)
            raise

    def __enter__(self) -> "BatchStream":
        return self

    def __exit__(self, _exc_type, active_error, _traceback) -> None:
        try:
            self.close()
        except BaseException as close_error:
            if active_error is None:
                raise
            _warn_preserving("tearing the batch down", active_error, close_error)



    @property
    def rows_in_flight(self) -> int:
        """Rows added and not yet finished or cancelled."""
        return 0 if self._session is None else self._session.rows_in_flight

    def add(self, request: GenerationRequest) -> int:
        """Admit one request, and answer with the row number reporting it."""
        session = self._require_open()
        validate = _validate_vlm_requests if self._is_vlm else _validate_text_requests
        (validated,) = validate([request], session.adapter.defaults)
        return session.add(validated)

    def cancel(self, row: int) -> bool:
        """Withdraw a row, answering whether it is gone."""
        return self._require_open().cancel(row)

    def step(self) -> list[GenerationEvent]:
        """What the batch produced in one decode step."""
        return list(self._require_open().step())

    def close(self) -> None:
        """Release the batch and the generation lock. Safe to call twice."""
        if self._closed:
            return
        self._require_owner()
        self._closed = True
        self._session = None
        self._stack.close()

    def _require_open(self):
        if self._session is None:
            raise RuntimeError("This BatchStream is closed.")
        self._require_owner()
        if not self._session.usable:
            raise RuntimeError(
                "This BatchStream holds a batch neither it nor the engine can "
                "answer for, left by an earlier call that failed partway. "
                "Close it and open another."
            )
        return self._session

    def _require_owner(self) -> None:
        current = (threading.get_ident(), _current_async_task())
        if current == self._owner:
            return
        raise RuntimeError(
            "This BatchStream belongs to the thread and task that opened it: "
            "the generation lock it holds can only be released there."
        )


def stream_batch(
    model,
    tokenizer_or_processor,
    requests: Sequence[GenerationRequest],
    *,
    defaults: GenerationDefaults | None = None,
) -> Iterator[GenerationEvent]:
    """``generate_batch``, reporting each row as it goes."""
    if defaults is not None and defaults.stop_strings:
        raise ValueError(
            "stream_batch cannot apply stop_strings: they cut on token "
            "boundaries, which would retract text already streamed. Scan the "
            "deltas for them instead, or use generate_batch."
        )
    return _stream_batch(
        model, tokenizer_or_processor, requests, defaults=defaults,
    )


def _stream_batch(
    model,
    tokenizer_or_processor,
    requests: Sequence[GenerationRequest],
    *,
    defaults: GenerationDefaults | None = None,
    audio_warn_stacklevel: int = 2,
) -> Iterator[GenerationEvent]:
    if defaults is None:
        defaults = GenerationDefaults()
    if not isinstance(defaults, GenerationDefaults):
        raise TypeError("defaults must be GenerationDefaults.")
    is_vlm = bool(getattr(model, "_is_vlm_model", False))
    validated = (
        _validate_vlm_requests(requests, defaults)
        if is_vlm
        else _validate_text_requests(requests, defaults)
    )
    if not validated:
        return
    if is_vlm:
        tokenizer_or_processor = (
            getattr(model, "_processor", None) or tokenizer_or_processor
        )
    if tokenizer_or_processor is None:
        raise ValueError(
            "Batched generation requires a processor."
            if is_vlm
            else "Text batched generation requires a tokenizer."
        )
    _install_arrays_cache_advance_fix()
    with generation_mode(model):
        with _generation_cache_hygiene():
            adapter = (
                _VLMBatchAdapter(
                    model, tokenizer_or_processor, defaults,
                    audio_warn_stacklevel = audio_warn_stacklevel + 1,
                )
                if is_vlm
                else _TextBatchAdapter(model, tokenizer_or_processor, defaults)
            )
            yield from adapter.stream(validated)


def fast_generate(
    self,
    prompts,
    *,
    max_tokens=256,
    temperature=0.0,
    top_p=0.0,
    top_k=0,
    min_p=0.0,
    stop_strings=(),
    prefill_batch_size=8,
    completion_batch_size=32,
    max_kv_size=None,
):
    """Batch-generate text rollouts from a training-resident MLX model.

    Accepts one rendered prompt or a sequence of rendered prompts and always
    returns a list of ``GenerationResult`` objects in input order. Callers that
    need token-id prompts, images or audio, or heterogeneous per-request
    controls should use ``generate_batch`` directly. KV-cache quantisation is
    not forwarded by this engine; use ``model.generate`` for that.
    """

    tokenizer = getattr(self, "_tokenizer", None)
    if tokenizer is None:
        raise ValueError("Unsloth MLX: fast_generate requires model._tokenizer.")
    if isinstance(prompts, str):
        prompts = [prompts]
    elif isinstance(prompts, Mapping):
        # A vLLM prompt dict iterates to its keys, which are strings, so every
        # guard below would pass and generation would run on the key names.
        raise TypeError(
            "Unsloth MLX: fast_generate takes rendered prompt strings, not "
            "vLLM prompt dicts. Pass the rendered text, and use generate_batch "
            "with GenerationRequest(prompt=..., image=...) for media."
        )
    else:
        try:
            prompts = list(prompts)
        except TypeError as exc:
            raise TypeError(
                "Unsloth MLX: fast_generate prompts must be a string or a "
                "sequence of strings."
            ) from exc
    if any(not isinstance(prompt, str) for prompt in prompts):
        raise TypeError(
            "Unsloth MLX: every fast_generate prompt must be a string."
        )

    defaults = GenerationDefaults(
        max_tokens=max_tokens,
        sampling=SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        ),
        stop_strings=stop_strings,
        prefill_batch_size=prefill_batch_size,
        completion_batch_size=completion_batch_size,
        max_kv_size=max_kv_size,
    )
    requests = [GenerationRequest(prompt=prompt) for prompt in prompts]
    return generate_batch(self, tokenizer, requests, defaults=defaults)


__all__ = [
    "BatchStream",
    "GenerationDefaults",
    "GenerationEvent",
    "GenerationRequest",
    "GenerationResult",
    "SamplingParams",
    "fast_generate",
    "generate_batch",
    "generation_mode",
    "stream_batch",
]

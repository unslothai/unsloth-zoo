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
from numbers import Integral, Real
from typing import Any, Callable, Iterator, Literal, Sequence


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
        if self.kv_bits is not None:
            if isinstance(self.kv_bits, bool) or not isinstance(self.kv_bits, Real):
                raise TypeError("defaults.kv_bits must be a number.")
            if not math.isfinite(self.kv_bits) or self.kv_bits <= 0:
                raise ValueError("defaults.kv_bits must be a finite positive number.")
        if self.kv_group_size is not None:
            _validate_positive_int(self.kv_group_size, "defaults.kv_group_size")
        if self.kv_quant_scheme is not None and not isinstance(self.kv_quant_scheme, str):
            raise TypeError("defaults.kv_quant_scheme must be a string.")
        if self.quantized_kv_start is not None:
            if isinstance(self.quantized_kv_start, bool) or not isinstance(
                self.quantized_kv_start, Integral
            ):
                raise TypeError("defaults.quantized_kv_start must be an integer.")
            if self.quantized_kv_start < 0:
                raise ValueError("defaults.quantized_kv_start must not be negative.")
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

    ``logits_processors`` are this row's own ``(tokens, logits) -> logits`` callables,
    run in order on its slice before it draws, and shown the same history on every path
    so a row answers the same batched or alone. They keep state: never share them.
    """

    prompt: str | None = None
    prompt_token_ids: Sequence[int] | None = None
    image: Any | None = None
    audio: Any | None = None
    max_tokens: int | None = None
    sampling: SamplingParams | None = None
    logits_processors: Sequence[Callable[[Any, Any], Any]] | None = None

    def __post_init__(self):
        if self.logits_processors is None:
            return
        try:
            processors = tuple(self.logits_processors)
        except TypeError as exc:
            raise TypeError(
                "logits_processors must be a sequence of callables."
            ) from exc
        if any(not callable(processor) for processor in processors):
            raise TypeError("Every logits processor must be callable.")
        object.__setattr__(self, "logits_processors", processors)


@dataclass(frozen=True)
class GenerationResult:

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
    _validate_text_defaults(defaults)
    return validated


def _validate_text_defaults(defaults: GenerationDefaults) -> None:
    refused_kv = [
        name for name in _KV_QUANT_CONTROLS if getattr(defaults, name) is not None
    ]
    if refused_kv:
        raise ValueError(
            f"{' and '.join(refused_kv)} are not forwarded by this engine's "
            f"mlx-lm text path (installed: {_installed_mlx_lm_version()}); "
            "omit these controls."
        )


def _kv_quant_options(defaults: GenerationDefaults) -> dict:
    """The caller's KV-quant controls, which validation has already vetted."""
    return {
        name: getattr(defaults, name)
        for name in _KV_QUANT_CONTROLS
        if getattr(defaults, name) is not None
    }


def _batch_kv_quant_options(defaults: GenerationDefaults) -> dict:
    """The controls a batch cache is built with.

    A sequential decode quantizes a turboquant cache once it reaches the start; a batch
    decides at construction, so mlx-vlm's default start (5000) leaves a shorter batch
    unquantized for good. Building from the first token is what was asked.
    """
    options = _kv_quant_options(defaults)
    if "quantized_kv_start" not in options and _vlm_turboquant(defaults):
        options["quantized_kv_start"] = 0
    return options


def _vlm_turboquant(defaults: GenerationDefaults) -> bool:
    """Whether these controls make mlx-vlm build a turboquant cache."""
    if defaults.kv_bits is None:
        return False
    try:
        return bool(_resolve_vlm_batch_module().turboquant_enabled(
            defaults.kv_bits, defaults.kv_quant_scheme))
    except Exception:
        return False


def _vlm_kv_controls_refused(defaults: GenerationDefaults) -> list[str]:
    """The KV-quant controls set here that mlx-vlm would drop rather than apply."""
    requested = [
        name for name in _KV_QUANT_CONTROLS if getattr(defaults, name) is not None
    ]
    if not requested:
        return []
    try:
        accepted = inspect.signature(
            _resolve_vlm_batch_module().BatchGenerator.__init__
        ).parameters
    except Exception:
        return requested
    refused = [name for name in requested if name not in accepted]
    scheme = defaults.kv_quant_scheme
    if scheme is not None and "kv_quant_scheme" not in refused and scheme not in _vlm_kv_quant_schemes():
        # Any other name quietly resolves to uniform.
        refused.append("kv_quant_scheme")
    if defaults.kv_bits is None:
        # Every other control reads off kv_bits, so without it none of them build anything.
        refused += [name for name in requested if name != "kv_bits" and name not in refused]
        return refused
    turbo = _vlm_turboquant(defaults)
    # Fractional bits make the cache turboquant whatever the scheme says; a turboquant cache
    # takes no group size; quantized_kv_start gates only a turboquant cache.
    dropped = ["kv_group_size"] if turbo else ["quantized_kv_start"]
    if turbo and scheme is not None and scheme != "turboquant":
        dropped.append("kv_quant_scheme")
    refused += [name for name in dropped if name in requested and name not in refused]
    return refused


def _vlm_kv_quant_schemes() -> tuple[str, ...]:
    """The scheme names the installed mlx-vlm resolves."""
    try:
        from mlx_vlm import kv_quant

        return (kv_quant.UNIFORM_SCHEME, kv_quant.TURBOQUANT_SCHEME)
    except Exception:
        return ("uniform", "turboquant")


def vlm_batch_adds_special_tokens(model, processor) -> bool:
    """Whether a batched prompt is tokenized with the tokenizer's own special tokens.

    A Gemma template writes ``<bos>`` itself, so a processor carrying one must not add
    another. A caller whose rendered prompt already opens with the BOS text strips it
    when this is true, or the batch would see two.
    """
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type in ("gemma3", "gemma3n", "gemma4", "gemma4_unified"):
        return getattr(processor, "chat_template", None) is None
    return True


def _vlm_quantized_cache_gap(model, defaults: GenerationDefaults) -> str | None:
    """Why mlx-vlm would refuse to quantize this model's batch cache, or None.

    It refuses while building the batch, after the rows were admitted, for a cache
    that converts itself (``to_batch``); asked here so a caller can decline first.
    """
    if defaults.kv_bits is None:
        return None
    make_cache = getattr(getattr(model, "language_model", model), "make_cache", None)
    if not callable(make_cache):
        return None
    try:
        from mlx_vlm.models.cache import should_quantize_kv_layer
    except ImportError:
        # mlx-vlm only exported this policy as a helper from 0.6.6. Before that the
        # same rule is inline in generate/ar.py, which builds each layer's batch cache
        # with `quantize=(i < n - 1 if n > 2 else True)`: a deep stack leaves its last
        # layer unquantized. Reproducing it keeps the refusal in step with what the
        # installed release actually does, where assuming every layer quantizes made
        # this refuse a model mlx-vlm would have batched.
        def should_quantize_kv_layer(index, count):
            return True if count <= 2 else index < count - 1
    try:
        entries = list(make_cache())
    except Exception:
        return None
    for index, entry in enumerate(entries):
        if hasattr(entry, "to_batch") and should_quantize_kv_layer(index, len(entries)):
            return (
                f"{type(entry).__name__} keeps model-specific cache state that "
                f"{_installed_mlx_vlm_version()} cannot quantize across a batch; "
                "omit kv_bits."
            )
    return None


def _unpadded_lengths(mask) -> list[int] | None:
    """Each row's own prompt length from the attention mask, or None without one.

    mlx-vlm left-pads a prefill batch to its widest prompt, so the padded width is
    not what a shorter row cost.
    """
    if mask is None:
        return None
    try:
        return [int(n) for n in mask.sum(axis=-1).tolist()]
    except Exception:
        return None


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
    _validate_vlm_defaults(defaults)
    return validated


def _validate_vlm_defaults(defaults: GenerationDefaults) -> None:
    if defaults.prefill_batch_size > defaults.completion_batch_size:
        # An inverted pair never admits anything on mlx-vlm, so generation would
        # hang. mlx-lm normalizes the two, so this is vision-specific.
        raise ValueError(
            "prefill_batch_size must not exceed completion_batch_size for "
            "batched vision generation (got "
            f"{defaults.prefill_batch_size} > {defaults.completion_batch_size})."
        )
    refused_kv = _vlm_kv_controls_refused(defaults)
    if refused_kv:
        raise ValueError(
            f"{' and '.join(refused_kv)} would be dropped rather than applied by "
            f"{_installed_mlx_vlm_version()}'s batch generator; omit these "
            "controls, or use model.generate."
        )
    if defaults.max_kv_size is not None:
        # No supported BatchGenerator takes a KV-window control, so say so rather
        # than drop a memory constraint the caller believes is in force.
        raise ValueError(
            "max_kv_size is not supported for batched vision generation; "
            f"{_installed_mlx_vlm_version()} exposes no KV-window control on "
            "its batch generator. Omit it, or use model.generate."
        )


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
    return (params.temperature, params.top_p, params.top_k, params.min_p)


def _sampler_for(sample_utils_module, params: SamplingParams, make_sampler):
    if _draws_seeded(params):
        return _seeded_sampler(sample_utils_module, params)
    return make_sampler(
        temp=params.temperature,
        top_p=params.top_p,
        top_k=params.top_k,
        min_p=params.min_p,
    )


def _sequential_history(processor):
    """A processor shown only the last prompt token and what followed it.

    How much prompt a processor sees is otherwise an accident of the prefill, and a
    repetition penalty windows recent tokens, so that accident would decide the reply.
    """
    state = {"offset": None}

    def _shown(tokens, logits):
        if state["offset"] is None:
            state["offset"] = max(int(tokens.shape[0]) - 1, 0)
        return processor(tokens[state["offset"] :], logits)

    return _shown


def _row_processors(request: GenerationRequest) -> list:
    return [_sequential_history(fn) for fn in (request.logits_processors or ())]


class _PerRowSampler:

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
        self._by_uid[uid] = params
        self.row_uids.append(uid)

    def release(self, uid) -> None:
        self._by_uid.pop(uid, None)
        self._samplers.pop(uid, None)
        if uid in self.row_uids:
            self.row_uids.remove(uid)

    def release_all(self) -> None:
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


def _require_evaluable(model):
    """The model's eval(), or a refusal: what switches a model out of training."""
    switch = getattr(model, "eval", None)
    if not callable(switch):
        raise TypeError("generation_mode requires a model with eval().")
    return switch


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
        _require_evaluable(model)()
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
    """One row's progress through a batch, in the order the batch produced it.

    ``prompt_tokens`` and ``generated_tokens`` are the row's counts so far, carried on
    every event so a caller that stops a batch early can still report usage. ``result``
    arrives only when the row ends on its own.
    """

    index: int
    delta: str = ""
    result: GenerationResult | None = None
    prompt_tokens: int = 0
    generated_tokens: int = 0


class BatchRowRefused(ValueError):
    """A row a batch will not take, for what it is rather than for being wrong."""


def _text_events(index: int, state: "_PendingResult") -> Iterator[GenerationEvent]:
    # Emitted even when the token decoded to no text. A tokenizer buffering an
    # incomplete byte sequence is ordinary, not an edge case, and the counts ride on
    # every event so a caller that cancels or withdraws right then still reports what
    # the row actually consumed.
    yield GenerationEvent(
        index=index,
        delta=state.release(),
        prompt_tokens=state.prompt_token_count,
        generated_tokens=len(state.token_ids),
    )


def _finished_events(
    index: int,
    state: "_PendingResult",
    tokenizer,
) -> Iterator[GenerationEvent]:
    yield GenerationEvent(
        index=index,
        delta=state.release(),
        result=state.result(tokenizer),
        prompt_tokens=state.prompt_token_count,
        generated_tokens=len(state.token_ids),
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


def _cancelled_result(tokenizer, state: "_PendingResult") -> GenerationResult:
    if state.finish_reason is None:
        state.finish(tokenizer, "stop")
    return state.result(tokenizer)


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
                logits_processors=[_row_processors(request)],
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
        return self._take_back(row) is not None

    def withdraw(self, row: int) -> GenerationResult | None:
        state = self._take_back(row)
        return None if state is None else _cancelled_result(self.adapter.tokenizer, state)

    def _take_back(self, row: int) -> "_PendingResult | None":
        uid = self._uid_of.get(row)
        if uid is None or uid not in self._pending:
            return None
        try:
            self.generator.remove([uid])
        except BaseException:
            self.usable = False
            raise
        state = self._pending[uid]
        self._retire(uid)
        return state

    def step(self) -> Iterator[GenerationEvent]:
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

    return frozenset({
        "inputs_embeds",
        *(getattr(batch_module, "_SEQUENCE_ALIGNED_PROMPT_KWARGS", None) or ()),
    })


def _row_shape(key, value, padded: frozenset):

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

    Follows the helpers the method calls on itself, since several models assign one call
    down. A ``self.<name>(...)`` with no readable body is a submodule being run -- every
    vision model runs its tower that way -- and is stepped over. A binding counts wherever
    its target leads back to ``self``; mutating a container the model already holds does
    not, nor does reaching ``self`` indirectly.
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
                else [node.target]
                if isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For, ast.AsyncFor,
                                     ast.comprehension))
                else [item.optional_vars for item in node.items if item.optional_vars]
                if isinstance(node, (ast.With, ast.AsyncWith))
                else []
            )
            while targets:
                target = targets.pop()
                if isinstance(target, (ast.Tuple, ast.List)):
                    targets.extend(target.elts)
                    continue
                if isinstance(target, ast.Starred):
                    targets.append(target.value)
                    continue
                while isinstance(target, (ast.Attribute, ast.Subscript)):
                    target = target.value
                if isinstance(target, ast.Name) and target.id == "self":
                    return True
            if isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "setattr"
                    and node.args
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "self"
                ):
                    return True
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                ):
                    calls.add(node.func.attr)
        return any(follows(call) for call in sorted(calls))

    return follows(method, entry = True)


def _require_streamable_vlm(adapter: "_VLMBatchAdapter") -> None:

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


def _resolve_vlm_batch_module():

    batch_module = _resolve_module_attr(_VLM_BATCH_MODULES, "BatchGenerator")
    if batch_module is None:
        raise _vlm_api_shape_error("no importable module exposes BatchGenerator")
    # Bind the module the class is DEFINED in, where the private helpers live.
    defining = getattr(batch_module.BatchGenerator, "__module__", None)
    if defining and defining != batch_module.__name__:
        try:
            batch_module = importlib.import_module(defining)
        except Exception:
            pass
    return batch_module


def row_logits_processors_unavailable_reason() -> str | None:
    """Why a batched vision row here cannot carry its own logits processors."""

    try:
        _, insert_params, _ = _probe_vlm_api(_resolve_vlm_batch_module())
    except Exception as refusal:
        return str(refusal)
    if "logits_processors" in insert_params:
        return None
    return (
        f"{_installed_mlx_vlm_version()} takes no per-row logits processors, "
        "so a batched vision reply cannot carry a penalty or bias"
    )


class _VLMBatchAdapter:
    """Event-level adapter over mlx-vlm's BatchGenerator.

    Bypasses the public batch helper, which templates the prompt and discards
    per-token data, and reproduces what it owns: image-shape grouping, the
    release's chunked-prefill policy, and the wired-limit window.
    """

    def __init__(
        self, model, processor, defaults: GenerationDefaults, *, audio_warn_stacklevel = 3,
    ):
        batch_module = _resolve_vlm_batch_module()
        (
            self.constructor_params,
            insert_params,
            self.cancel,
        ) = _probe_vlm_api(batch_module)
        self.batch_module = batch_module
        self.generator_type = batch_module.BatchGenerator
        self.per_row_prompt_kwargs = "prompt_kwargs" in insert_params
        self.row_logits_processors = "logits_processors" in insert_params
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
        return vlm_batch_adds_special_tokens(self.model, self.processor)

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

    def _row_logits_processors(self, requests) -> list[list] | None:
        rows = [_row_processors(request) for request in requests]
        if not any(rows):
            return None
        if not self.row_logits_processors:
            raise _vlm_api_shape_error(
                "BatchGenerator.insert does not take logits_processors, so a "
                "request cannot carry a penalty or bias into a batch"
            )
        return rows

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
            logits_processors=_row_processors(request),
            **_kv_quant_options(self.defaults),
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
        options.update(_batch_kv_quant_options(self.defaults))
        max_tokens = [
            int(request.max_tokens or self.defaults.max_tokens) for request in chunk
        ]
        row_processors = self._row_logits_processors(chunk)
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
                insert_kwargs = {}
                if self.per_row_prompt_kwargs:
                    insert_kwargs["prompt_kwargs"] = self._split_prompt_kwargs(
                        gen_kwargs,
                        batch_size,
                    )
                if row_processors is not None:
                    insert_kwargs["logits_processors"] = row_processors
                uids = generator.insert(token_ids, max_tokens, **insert_kwargs)
                if row_sampler is not None:
                    row_sampler.bind_uids(uids)
                    row_sampler.bind_generator(generator)
                yield from self._drive(
                    generator, uids, indices, gen_kwargs, row_sampler,
                    prompt_token_count=len(token_ids[0]),
                    prompt_token_counts=_unpadded_lengths(mask),
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
        prompt_token_counts=None,
    ) -> Iterator[GenerationEvent]:
        """Report the chunk's rows, charging each its own prompt count, or
        ``prompt_token_count`` to every row when none are given."""
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if prompt_token_counts is None:
            prompt_token_counts = [prompt_token_count] * len(uids)
        pending = {
            uid: _PendingResult(
                detokenizer=_new_detokenizer(tokenizer, require_independent=True),
                scanner=_StopStringScanner(self.defaults.stop_strings),
                prompt_token_count=count,
            )
            for uid, count in zip(uids, prompt_token_counts)
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
    """One mlx-vlm ``BatchGenerator`` with its row set left open."""

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
            raise BatchRowRefused(
                "An audio request decodes on its own and cannot join a batch; "
                "generate it with generate_batch or stream_batch."
            )
        if request.logits_processors and not adapter.row_logits_processors:
            raise BatchRowRefused(
                "This mlx-vlm's batch cannot carry a row's own logits processors."
            )
        input_ids, prompt_kwargs = self._prepare(request)
        try:
            row_processors = adapter._row_logits_processors([request])
        except RuntimeError as refusal:
            # Reachable when the penalties come from the defaults rather than the
            # request, so the check above passes. add() refuses a row without
            # disturbing the batch, and nothing here has been inserted yet, so this
            # is a row refusal rather than the API-shape error raised when a whole
            # batch is being built.
            raise BatchRowRefused(str(refusal)) from refusal
        token_ids = input_ids.tolist()
        insert_kwargs = {
            "prompt_kwargs": adapter._split_prompt_kwargs(prompt_kwargs, 1),
        }
        if row_processors is not None:
            insert_kwargs["logits_processors"] = row_processors
        try:
            uids = self.generator.insert(
                token_ids,
                [int(request.max_tokens or defaults.max_tokens)],
                **insert_kwargs,
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
        return self._take_back(row) is not None

    def withdraw(self, row: int) -> GenerationResult | None:
        state = self._take_back(row)
        return None if state is None else _cancelled_result(self.tokenizer, state)

    def _take_back(self, row: int) -> "_PendingResult | None":
        uid = self._uid_of.get(row)
        if uid is None or uid not in self._pending:
            return None
        try:
            withdrawn = self._withdraw(uid)
        except BaseException:
            self.usable = False
            raise
        if not withdrawn:
            return None
        state = self._pending[uid]
        self._retire(uid)
        return state

    def step(self) -> Iterator[GenerationEvent]:
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
            raise BatchRowRefused(
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
            raise BatchRowRefused(
                f"This request prepares {', '.join(differing)} unlike the "
                "batch's other requests, which merging lines up against rows "
                "that do not match it. Generate it with generate_batch or "
                "stream_batch, which prepare a whole batch together."
            )
        return input_ids, prepared

    def _open(self):

        adapter = self.adapter
        defaults = adapter.defaults
        options = {
            "prefill_batch_size": defaults.prefill_batch_size,
            "completion_batch_size": defaults.completion_batch_size,
            "sampler": self.sampler,
            "compute_logprobs": True,
            **_batch_kv_quant_options(defaults),
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

        if self.adapter.cancel is None:
            return False
        return self.adapter.cancel(self.generator, uid) is not False

    def _retire(self, uid: int):

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


def _stream_setup(model, tokenizer, defaults: GenerationDefaults):
    """What a stream settles before it takes anything: vision or not, and the adapter."""
    if bool(getattr(model, "_is_vlm_model", False)):
        _validate_vlm_defaults(defaults)
        gap = _vlm_quantized_cache_gap(model, defaults)
        if gap is not None:
            raise ValueError(gap)
    else:
        _validate_text_defaults(defaults)
    if defaults.stop_strings:
        raise ValueError(
            "BatchStream cannot apply stop_strings: they cut on token "
            "boundaries, which would retract text already streamed. Scan "
            "the deltas for them instead."
        )
    is_vlm = bool(getattr(model, "_is_vlm_model", False))
    if is_vlm:
        tokenizer = getattr(model, "_processor", None) or tokenizer
    if tokenizer is None:
        raise ValueError(
            "Batched generation requires a processor."
            if is_vlm
            else "Text batched generation requires a tokenizer."
        )
    if is_vlm:
        adapter = _VLMBatchAdapter(model, tokenizer, defaults)
        _require_streamable_vlm(adapter)
    else:
        adapter = _TextBatchAdapter(model, tokenizer, defaults)
    return is_vlm, adapter


def stream_unavailable_reason(
    model,
    tokenizer = None,
    *,
    defaults: GenerationDefaults | None = None,
) -> str | None:
    """Why ``BatchStream`` would refuse this model, or None. Asked before committing."""
    if defaults is None:
        defaults = GenerationDefaults()
    if not isinstance(defaults, GenerationDefaults):
        raise TypeError("defaults must be GenerationDefaults.")
    try:
        _stream_setup(model, tokenizer, defaults)
        _require_evaluable(model)
    except Exception as refusal:
        return str(refusal)
    return None


class BatchStream:
    """A batch left open: rows join and leave while the batch decodes.

    A row joins immediately, but which admitted row prefills next is the underlying
    generator's decision, not this one's. mlx-lm takes them in arrival order; mlx-vlm
    sorts its unprocessed prompts shortest-first on every insert, so a long vision
    prompt waits behind shorter ones that arrived after it. Rows already decoding are
    unaffected either way.
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
        is_vlm, adapter = _stream_setup(model, tokenizer, defaults)
        self._stack = ExitStack()
        self._session = None
        self._is_vlm = is_vlm
        self._closed = False
        self._owner = (threading.get_ident(), _current_async_task())
        try:
            self._stack.enter_context(generation_mode(model))
            self._stack.enter_context(_generation_cache_hygiene())
            self._session = (
                _VLMBatchSession(adapter) if is_vlm else _TextBatchSession(adapter)
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
        return 0 if self._session is None else self._session.rows_in_flight

    def add(self, request: GenerationRequest) -> int:
        session = self._require_open()
        validate = _validate_vlm_requests if self._is_vlm else _validate_text_requests
        (validated,) = validate([request], session.adapter.defaults)
        return session.add(validated)

    def cancel(self, row: int) -> bool:
        return self._require_open().cancel(row)

    def withdraw(self, row: int) -> GenerationResult | None:
        """The same, answering with what the row managed rather than whether it was
        still there to withdraw.
        """
        return self._require_open().withdraw(row)

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

    def __del__(self):
        # An open stream dropped without close() unwinds here instead. ExitStack has
        # no finalizer, but the generation_mode generator it holds does, so the
        # cleanup runs in whatever thread collected this. On the owner, which is where
        # refcounting collects, closing repairs it; off-owner nothing can release an
        # RLock the owner holds, so swallow the non-owner release rather than let it
        # surface from unrelated code, and name the leak.
        if self.__dict__.get("_closed", True):
            return
        self._closed = True
        self._session = None
        stack = self.__dict__.get("_stack")
        if stack is None:
            return
        if (threading.get_ident(), _current_async_task()) == self._owner:
            stack.close()
            return
        try:
            stack.close()
        except RuntimeError:
            pass
        warnings.warn(
            "A BatchStream was dropped by a thread or task other than the one that "
            "opened it, so the generation lock it holds cannot be released and later "
            "generation will block. Close it on the thread that opened it.",
            RuntimeWarning,
            stacklevel=2,
        )

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


class _OwnedStream:
    """A ``_stream_batch`` iterator pinned to the thread and task that opened it.

    The generator holds ``generation_mode``'s process-wide lock across its yields, and
    that lock refuses a release from anyone else. Resuming or closing the iterator
    elsewhere therefore runs the cleanup on the wrong owner: the release raises before
    it can decrement the depth, the underlying RLock is never dropped, and every later
    generation in the process blocks forever. ``BatchStream`` pins itself the same way.
    """

    def __init__(self, events: Iterator[GenerationEvent]):
        self._events = events
        self._owner = (threading.get_ident(), _current_async_task())

    def _require_owner(self, action: str) -> None:
        if (threading.get_ident(), _current_async_task()) == self._owner:
            return
        raise RuntimeError(
            f"This stream_batch iterator belongs to the thread and task that "
            f"opened it: the generation lock it holds can only be released there, "
            f"so it cannot be {action} from another one."
        )

    def __iter__(self):
        return self

    def __next__(self) -> GenerationEvent:
        self._require_owner("advanced")
        return next(self._events)

    def send(self, value):
        self._require_owner("advanced")
        return self._events.send(value)

    def throw(self, *args):
        self._require_owner("advanced")
        return self._events.throw(*args)

    def close(self):
        self._require_owner("closed")
        self._events.close()

    def __del__(self):
        # Dropping the last reference finalizes the generator in whatever thread ran
        # the collection, which does not come through close(). Off-owner that runs
        # generation_mode's cleanup as a non-owner: the release raises before dropping
        # the RLock, leaving the depth inconsistent on top of a lock only the owner
        # could ever have released. Refcounting usually collects on the thread that
        # dropped the reference, so closing here repairs the ordinary abandonment.
        events = self.__dict__.pop("_events", None)
        if events is None:
            return
        if (threading.get_ident(), _current_async_task()) == self._owner:
            events.close()
            return
        # Off-owner there is nothing that can release the lock: an RLock is only
        # releasable by the thread holding it. Let the generator finalize, swallow the
        # non-owner release so it does not surface as an unraisable exception from
        # whatever code happened to trigger the collection, and say plainly what
        # leaked, since the alternative is a process that blocks with no explanation.
        try:
            events.close()
        except RuntimeError:
            pass
        warnings.warn(
            "A stream_batch iterator was dropped by a thread or task other than "
            "the one that opened it, so the generation lock it holds cannot be "
            "released and later generation will block. Close it on the thread that "
            "opened it.",
            RuntimeWarning,
            stacklevel=2,
        )


def stream_batch(
    model,
    tokenizer_or_processor,
    requests: Sequence[GenerationRequest],
    *,
    defaults: GenerationDefaults | None = None,
) -> Iterator[GenerationEvent]:
    """``generate_batch``, reporting each row as it goes."""
    if defaults is not None and not isinstance(defaults, GenerationDefaults):
        raise TypeError("defaults must be GenerationDefaults.")
    if defaults is not None and defaults.stop_strings:
        raise ValueError(
            "stream_batch cannot apply stop_strings: they cut on token "
            "boundaries, which would retract text already streamed. Scan the "
            "deltas for them instead, or use generate_batch."
        )
    return _OwnedStream(_stream_batch(
        model, tokenizer_or_processor, requests, defaults=defaults,
        # _OwnedStream.__next__ sits between the caller and the generator body, so
        # the audio warning needs one more frame to reach the caller.
        audio_warn_stacklevel=3,
    ))


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
    if is_vlm and (gap := _vlm_quantized_cache_gap(model, defaults)) is not None:
        raise ValueError(gap)
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
    "BatchRowRefused",
    "BatchStream",
    "GenerationDefaults",
    "GenerationEvent",
    "GenerationRequest",
    "GenerationResult",
    "SamplingParams",
    "fast_generate",
    "generate_batch",
    "generation_mode",
    "row_logits_processors_unavailable_reason",
    "stream_batch",
    "stream_unavailable_reason",
    "vlm_batch_adds_special_tokens",
]

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

import asyncio
import importlib
import inspect
import math
import os
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import Any, Literal, Sequence


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

    def __post_init__(self):
        temperature = float(self.temperature)
        top_p = float(self.top_p)
        min_p = float(self.min_p)
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
    """Sampled-token output with text from the backend's streaming detokenizer."""

    token_ids: list[int]
    text: str
    logprobs: list[float] | None
    finish_reason: Literal["stop", "length", "stop_string"]
    stop_match: str | None = None


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
        "needs a BatchGenerator that streams per-token events; upgrade or "
        "reinstall mlx-lm, or use model.generate for sequential decoding."
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


@contextmanager
def _generation_cache_hygiene():
    """Cache hygiene around one burst; limits stay owned by ``generation_mode``."""

    import mlx.core as mx

    clear_cache = getattr(mx, "clear_cache", None)
    if not callable(clear_cache):
        raise RuntimeError(
            "Unsupported MLX runtime API shape (mlx.core.clear_cache missing)."
        )
    clear_cache()
    active_error = None
    try:
        yield
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        try:
            clear_cache()
        except BaseException as clear_error:
            if active_error is None:
                raise
            try:
                warnings.warn(
                    "mlx.core.clear_cache() failed while preserving an active "
                    f"{type(active_error).__name__}: {clear_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            except BaseException:
                pass


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
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    text: str = ""
    finish_reason: Literal["stop", "length", "stop_string"] | None = None
    stop_match: str | None = None

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


class _TextBatchAdapter:
    def __init__(self, model, tokenizer, defaults: GenerationDefaults):
        generate_module = importlib.import_module("mlx_lm.generate")
        sample_utils_module = importlib.import_module("mlx_lm.sample_utils")
        _probe_text_api(generate_module)
        _probe_sampler_api(sample_utils_module)
        self.batch_generator_type = generate_module.BatchGenerator
        self.make_sampler = sample_utils_module.make_sampler
        self.model = model
        self.tokenizer = tokenizer
        self.defaults = defaults

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        prompts = [_encode_prompt(self.tokenizer, request) for request in requests]
        max_tokens = [
            int(request.max_tokens or self.defaults.max_tokens)
            for request in requests
        ]
        sampling = [
            request.sampling or self.defaults.sampling
            for request in requests
        ]
        samplers = [
            self.make_sampler(
                temp=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                min_p=params.min_p,
            )
            for params in sampling
        ]
        generator = self.batch_generator_type(
            self.model,
            max_tokens=self.defaults.max_tokens,
            stop_tokens=_eos_stop_tokens(self.tokenizer),
            prefill_batch_size=self.defaults.prefill_batch_size,
            completion_batch_size=self.defaults.completion_batch_size,
            max_kv_size=self.defaults.max_kv_size,
        )
        active_error = None
        try:
            uids = generator.insert(
                prompts,
                max_tokens=max_tokens,
                samplers=samplers,
                logits_processors=[[] for _ in requests],
            )
            pending = {
                uid: _PendingResult(
                    detokenizer=_new_detokenizer(self.tokenizer),
                    scanner=_StopStringScanner(self.defaults.stop_strings),
                )
                for uid in uids
            }
            completed: dict[int, GenerationResult] = {}
            while pending:
                events = generator.next_generated()
                if not events:
                    raise RuntimeError(
                        "mlx-lm ended its event stream before every request "
                        "reported a finish reason."
                    )
                for event in events:
                    state = pending.get(event.uid)
                    if state is None:
                        continue
                    finish_reason = event.finish_reason
                    if finish_reason is None:
                        stopped = state.append(
                            self.tokenizer,
                            int(event.token),
                            _sampled_logprob(event),
                        )
                        if stopped:
                            generator.remove([event.uid])
                            completed[event.uid] = state.result(self.tokenizer)
                            del pending[event.uid]
                            continue
                        continue
                    if finish_reason not in ("stop", "length"):
                        raise RuntimeError(
                            "mlx-lm emitted an unsupported finish reason: "
                            f"{finish_reason!r}."
                        )
                    if finish_reason == "length":
                        state.add_terminal(
                            int(event.token),
                            _sampled_logprob(event),
                        )
                    state.finish(self.tokenizer, finish_reason)
                    completed[event.uid] = state.result(self.tokenizer)
                    del pending[event.uid]
            return [completed[uid] for uid in uids]
        except BaseException as exc:
            active_error = exc
            raise
        finally:
            try:
                generator.close()
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

    for name in candidates:
        try:
            module = importlib.import_module(name)
        except Exception:
            continue
        if getattr(module, attribute, None) is not None:
            return module
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

    def __init__(self, model, processor, defaults: GenerationDefaults):
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
        self.stream_module = _resolve_module_attr(_VLM_STREAM_MODULES, "wired_limit")
        utils = importlib.import_module("mlx_vlm.utils")
        self.prepare_inputs = utils.prepare_inputs
        self.process_image = getattr(utils, "process_image", None)
        sample_utils = importlib.import_module("mlx_lm.sample_utils")
        _probe_sampler_api(sample_utils)
        self.make_sampler = sample_utils.make_sampler
        self.model = model
        self.processor = processor
        self.defaults = defaults

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
        except Exception:
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
        sampling = request.sampling or self.defaults.sampling
        shape = self._image_shape(request.image)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        return (
            sampling.temperature,
            sampling.top_p,
            sampling.top_k,
            sampling.min_p,
            request.image is None,
            shape,
            len(tokenizer.encode(request.prompt)),
        )

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

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        requests = [self._decode_image(request) for request in requests]
        results: list[GenerationResult | None] = [None] * len(requests)
        audio_indices = [
            index for index, request in enumerate(requests) if request.audio is not None
        ]
        if audio_indices:
            # No supported release batches audio.
            warnings.warn(
                f"{len(audio_indices)} audio request(s) decode one at a time: "
                f"{_installed_mlx_vlm_version()} batches text and images only.",
                RuntimeWarning,
                stacklevel=3,
            )
            for index in audio_indices:
                results[index] = self._generate_sequentially(requests[index])
        groups: dict[Any, list[int]] = {}
        for index, request in enumerate(requests):
            if request.audio is None:
                groups.setdefault(self._group_key(request), []).append(index)
        # Older releases slice shared kwargs from row zero on every prefill
        # batch, pairing later prompts with earlier embeddings, so they run one
        # chunk per generator. Per-row releases keep the configured sizes.
        capacity = (
            None
            if self.per_row_prompt_kwargs
            else min(
                self.defaults.prefill_batch_size,
                self.defaults.completion_batch_size,
            )
        )
        for indices in groups.values():
            step = capacity or len(indices)
            for start in range(0, len(indices), step):
                chunk = indices[start : start + step]
                for index, result in zip(chunk, self._run_chunk(requests, chunk)):
                    results[index] = result
        if any(result is None for result in results):
            raise RuntimeError(
                "Internal error: batched vision generation produced no result "
                f"for {sum(result is None for result in results)} of "
                f"{len(results)} requests."
            )
        return results

    def _generate_sequentially(self, request: GenerationRequest) -> GenerationResult:
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
            sampler=self.make_sampler(
                temp=sampling.temperature,
                top_p=sampling.top_p,
                top_k=sampling.top_k,
                min_p=sampling.min_p,
            ),
        )
        for event in events:
            if previous is not None and state.append(
                tokenizer,
                int(previous.token),
                _event_logprob(previous),
            ):
                return state.result(tokenizer)
            previous = event
        if previous is None:
            raise RuntimeError(
                "mlx-vlm produced no events for an audio request."
            )
        state.finish(tokenizer, self._terminal_reason(previous, tokenizer))
        return state.result(tokenizer)

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
    ) -> list[GenerationResult]:
        chunk = [requests[index] for index in indices]
        batch_size = len(chunk)
        prompts = [request.prompt for request in chunk]
        images = [request.image for request in chunk if request.image is not None]
        sampling = chunk[0].sampling or self.defaults.sampling
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
        if "compute_logprobs" in self.constructor_params:
            # The public helper turns this off, but logprobs are contract here.
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
                # Optional fields enumerate as None and would overwrite valid
                # prepare_inputs values; upstream filters them the same way.
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
                return self._drive(generator, uids, gen_kwargs)
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

    def _drive(self, generator, uids, gen_kwargs) -> list[GenerationResult]:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        pending = {
            uid: _PendingResult(
                detokenizer=_new_detokenizer(tokenizer, require_independent=True),
                scanner=_StopStringScanner(self.defaults.stop_strings),
            )
            for uid in uids
        }
        completed: dict[int, GenerationResult] = {}
        while pending:
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
                state = pending.get(event.uid)
                if state is None:
                    continue
                finish_reason = event.finish_reason
                if finish_reason is None:
                    stopped = state.append(
                        tokenizer,
                        int(event.token),
                        _event_logprob(event),
                    )
                    if stopped:
                        # Where the release cannot cancel, the sequence keeps
                        # decoding upstream and its later events are ignored;
                        # results match either way, only wasted decode differs.
                        if self.cancel is not None:
                            self.cancel(generator, event.uid)
                        completed[event.uid] = state.result(tokenizer)
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
                completed[event.uid] = state.result(tokenizer)
                del pending[event.uid]
        return [completed[uid] for uid in uids]


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

    if defaults is None:
        defaults = GenerationDefaults()
    if not isinstance(defaults, GenerationDefaults):
        raise TypeError("defaults must be GenerationDefaults.")
    # Routing follows the model, not the request: a vision model preprocesses
    # text-only prompts too.
    is_vlm = bool(getattr(model, "_is_vlm_model", False))
    validated = (
        _validate_vlm_requests(requests, defaults)
        if is_vlm
        else _validate_text_requests(requests, defaults)
    )
    if not validated:
        return []
    if is_vlm:
        # A text-only multimodal load stays on the vision path but publishes its
        # inner tokenizer, which cannot drive mlx-vlm preprocessing.
        tokenizer_or_processor = (
            getattr(model, "_processor", None) or tokenizer_or_processor
        )
    if tokenizer_or_processor is None:
        raise ValueError(
            "Batched generation requires a processor."
            if is_vlm
            else "Text batched generation requires a tokenizer."
        )
    with generation_mode(model):
        with _generation_cache_hygiene():
            adapter = (
                _VLMBatchAdapter(model, tokenizer_or_processor, defaults)
                if is_vlm
                else _TextBatchAdapter(model, tokenizer_or_processor, defaults)
            )
            return adapter.generate(validated)


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
    kv_bits=None,
    kv_group_size=None,
):
    """Batch-generate text rollouts from a training-resident MLX model.

    Accepts one rendered prompt or a sequence of rendered prompts and always
    returns a list of ``GenerationResult`` objects in input order. Callers that
    need token-id prompts or heterogeneous per-request controls should use
    ``generate_batch`` directly.
    """

    tokenizer = getattr(self, "_tokenizer", None)
    if tokenizer is None:
        raise ValueError("Unsloth MLX: fast_generate requires model._tokenizer.")
    if isinstance(prompts, str):
        prompts = [prompts]
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
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
    )
    requests = [GenerationRequest(prompt=prompt) for prompt in prompts]
    return generate_batch(self, tokenizer, requests, defaults=defaults)


__all__ = [
    "GenerationDefaults",
    "GenerationRequest",
    "GenerationResult",
    "SamplingParams",
    "fast_generate",
    "generate_batch",
    "generation_mode",
]

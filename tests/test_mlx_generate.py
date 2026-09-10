import concurrent.futures
import contextlib
import importlib
import importlib.util
import inspect
import pathlib
import sys
import threading
import types
import warnings
from dataclasses import make_dataclass
import pytest

# Logic tests, but a few cross into mlx.core and mlx_vlm, so off Apple Silicon
# they run against the torch-backed shim.
if importlib.util.find_spec("mlx") is None:
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()

from unsloth_zoo.mlx.generate import (  # noqa: E402
    GenerationDefaults, GenerationEvent, GenerationRequest, SamplingParams,
    _GENERATION_MODE_LOCK, _PendingResult, _PerRowSampler,
    _StopStringScanner, _TextBatchAdapter,
    _eos_stop_tokens, _new_detokenizer, _probe_sampler_api, _probe_text_api,
    _generation_cache_hygiene, _restore_training_flags,
    _validate_text_requests, generate_batch, generation_mode, stream_batch,
)

class _CharDetokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.reset()
    def reset(self):
        self.offset = 0
        self.tokens = []
        self.text = ""
    def add_token(self, token):
        self.tokens.append(token)
        self.text = "".join(self.tokenizer.pieces[item] for item in self.tokens)
    def finalize(self):
        pass
    @property
    def last_segment(self):
        segment = self.text[self.offset:]
        self.offset = len(self.text)
        return segment

class _CharTokenizer:
    pieces = {1: "hello ", 2: "<ST", 3: "OP>", 4: "tailSTOPsuffix"}
    @property
    def detokenizer(self):
        return _CharDetokenizer(self)
    def decode(self, tokens):
        return "".join(self.pieces[item] for item in tokens)
    def encode(self, text, **kwargs):
        return [0] * len(text)

class _TableTokenizer:
    def __init__(self, pieces, clean=False):
        self.pieces, self.clean_up_tokenization_spaces = pieces, clean
    def decode(self, tokens):
        return self.pieces.get(tuple(tokens), "")
def test_stop_scanner_trims_to_token_boundary():
    tokenizer = _CharTokenizer()
    pending = _PendingResult(
        _new_detokenizer(tokenizer),
        _StopStringScanner(("<STOP>", "STOP")),
    )
    assert pending.append(tokenizer, 1, -0.1) is False
    assert pending.append(tokenizer, 2, -0.2) is False
    assert pending.append(tokenizer, 3, -0.3) is True
    result = pending.result(tokenizer)
    assert (result.token_ids, result.logprobs, result.text) == ([1], [-0.1], "hello ")
    assert (result.finish_reason, result.stop_match) == ("stop_string", "<STOP>")
    mid_token = _PendingResult(
        _new_detokenizer(tokenizer),
        _StopStringScanner(("STOP",)),
    )
    assert mid_token.append(tokenizer, 4, -0.4) is True
    assert mid_token.result(tokenizer).token_ids == []
    canonical = _TableTokenizer({(1,): " hello", (2,): "\ufffd", (2, 3): "¡STOP"})
    leading = _PendingResult(_new_detokenizer(canonical), _StopStringScanner((" hello",)))
    assert leading.append(canonical, 1, -0.1) is True
    assert leading.result(canonical).token_ids == []

@pytest.mark.parametrize(
    "generation_request,error",
    [
        (GenerationRequest(), "exactly one"),
        (GenerationRequest(prompt="x", prompt_token_ids=[1]), "exactly one"),
        (GenerationRequest(prompt_token_ids=[]), "must not be empty"),
        (GenerationRequest(prompt=""), "must not be empty"),
        (GenerationRequest(prompt="x", image=object()), "text prompts only"),
        (GenerationRequest(prompt="x", max_tokens=0), "must be positive"),
    ],
)
def test_request_invariants(generation_request, error):
    with pytest.raises((TypeError, ValueError), match=error):
        _validate_text_requests([generation_request], GenerationDefaults())

def test_one_shot_prompt_ids_are_normalized_during_validation():
    validated = _validate_text_requests([GenerationRequest(prompt_token_ids=iter((11, 12)))], GenerationDefaults())  # noqa: E501
    assert validated[0].prompt_token_ids == (11, 12)

def test_text_api_probe_accepts_supported_shape_and_names_gap_on_mismatch():
    Response = make_dataclass(
        "Response",
        # make_dataclass evaluates these, so no PEP 604 union: the package supports 3.9.
        [("uid", int), ("token", int), ("logprobs", object), ("finish_reason", object)],
    )
    GenerationBatch = type("GenerationBatch", (), {"Response": Response})
    class BatchGenerator:
        def __init__(self, model, max_tokens=1, stop_tokens=None,
                     prefill_batch_size=8, completion_batch_size=32, max_kv_size=None):
            pass
        def insert(self, prompts, max_tokens=None, samplers=None,
                   logits_processors=None):
            pass
        def next_generated(self):
            pass
        def remove(self, uids):
            pass
        def close(self):
            pass
    module = type(
        "PinnedShape", (), {"BatchGenerator": BatchGenerator, "GenerationBatch": GenerationBatch}
    )
    _probe_text_api(module)
    signature = inspect.signature(BatchGenerator)
    for name in ("prefill_batch_size", "completion_batch_size", "max_kv_size"):
        BatchGenerator.__signature__ = signature.replace(
            parameters=[item for item in signature.parameters.values() if item.name != name])
        with pytest.raises(RuntimeError, match=rf"constructor missing {name}"):
            _probe_text_api(module)
    del BatchGenerator.__signature__
    del BatchGenerator.remove
    with pytest.raises(RuntimeError, match=r"remove missing"):
        _probe_text_api(module)
    with pytest.raises(RuntimeError, match=r"make_sampler missing min_p"):
        _probe_sampler_api(type("Sample", (), {"make_sampler": lambda temp: None}))
    assert _eos_stop_tokens(types.SimpleNamespace(eos_token_ids={2, None})) == [[2]]

def test_falsey_defaults_are_not_silently_replaced(monkeypatch):
    with pytest.raises(TypeError, match="defaults"):
        generate_batch(object(), None, [], defaults={})
    from unsloth_zoo.mlx.generate import _validate_text_requests
    for name in ("kv_bits", "kv_group_size", "kv_quant_scheme", "quantized_kv_start"):
        value = "affine" if name == "kv_quant_scheme" else 4
        with pytest.raises(ValueError, match=rf"{name} are not forwarded by this engine"):
            _validate_text_requests([GenerationRequest(prompt="a")],
                                    GenerationDefaults(**{name: value}))
    for name in ("prefill_batch_size", "completion_batch_size", "max_kv_size"):
        for value in (-1, 0, True, 1.5):
            with pytest.raises((TypeError, ValueError), match=name):
                GenerationDefaults(**{name: value})
    calls = []
    def clear():
        calls.append(None)
        if len(calls) == 2:
            raise RuntimeError("clear")
    monkeypatch.setattr("mlx.core.clear_cache", clear)
    with pytest.warns(RuntimeWarning, match="clear_cache"), pytest.raises(ValueError, match="body"):
        with _generation_cache_hygiene():
            raise ValueError("body")

def _record_cache_calls(monkeypatch, *, has_clear_cache=True):
    events = []
    monkeypatch.setattr(
        "mlx.core.synchronize",
        # `is None`: a falsey stream labelled "default" would hide an over-drain.
        lambda stream=None: events.append(
            f"synchronize:{'default' if stream is None else stream}"
        ),
    )
    if has_clear_cache:
        monkeypatch.setattr("mlx.core.clear_cache", lambda: events.append("clear_cache"))
    else:
        # None, not delattr: tests/mlx_simulation trampolines any missing name, so a
        # deleted attribute stays callable and the guard never fires.
        monkeypatch.setattr("mlx.core.clear_cache", None, raising=False)
    return events

def test_both_cache_clears_drain_gpu_work_first(monkeypatch):
    events = _record_cache_calls(monkeypatch)
    with _generation_cache_hygiene():
        events.append("body")
    assert events == [
        "synchronize:default", "clear_cache", "body", "synchronize:default", "clear_cache",
    ]

def test_the_generation_stream_is_drained_not_just_the_default_one(monkeypatch):
    events = _record_cache_calls(monkeypatch)
    for name in ("mlx_lm.generate", "mlx_vlm.generate.dispatch"):
        monkeypatch.setitem(
            sys.modules, name,
            types.SimpleNamespace(generation_stream=f"stream<{name}>"),
        )

    with _generation_cache_hygiene():
        pass

    assert events.count("synchronize:stream<mlx_lm.generate>") == 2
    assert events.count("synchronize:stream<mlx_vlm.generate.dispatch>") == 2
    assert events.index("synchronize:stream<mlx_lm.generate>") < events.index("clear_cache")

def _fake_stream_module(monkeypatch, name, stream):
    monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(generation_stream=stream))

def test_the_speculative_decoding_stream_is_drained_too(monkeypatch):
    # Created on import and never routed through wired_limit: nothing else drains it.
    events = _record_cache_calls(monkeypatch)
    _fake_stream_module(monkeypatch, "mlx_vlm.speculative.common", "stream<speculative>")

    with _generation_cache_hygiene():
        pass

    assert events.count("synchronize:stream<speculative>") == 2

def test_one_stream_shared_by_several_modules_is_drained_once(monkeypatch):
    # 0.6.x re-exports one stream from every candidate name, so identity decides.
    events = _record_cache_calls(monkeypatch)
    shared = "stream<shared>"
    for name in ("mlx_vlm.generate", "mlx_vlm.generate.dispatch", "mlx_vlm.generate.ar"):
        _fake_stream_module(monkeypatch, name, shared)

    with _generation_cache_hygiene():
        pass

    assert events.count("synchronize:stream<shared>") == 2

def test_a_stream_that_cannot_be_drained_does_not_stop_the_clear(monkeypatch):
    # A plain mx.new_stream (the mlx-vlm pin floor) raises off its creating thread.
    events = _record_cache_calls(monkeypatch)
    def synchronize(stream=None):
        if stream == "stream<foreign>":
            raise RuntimeError("There is no Stream(gpu, 0) in current thread")
        events.append(f"synchronize:{'default' if stream is None else stream}")
    monkeypatch.setattr("mlx.core.synchronize", synchronize)
    _fake_stream_module(monkeypatch, "mlx_lm.generate", "stream<foreign>")

    with _generation_cache_hygiene():
        events.append("body")

    assert events == [
        "synchronize:default", "clear_cache", "body", "synchronize:default", "clear_cache",
    ]

def test_a_module_getattr_that_raises_does_not_stop_the_clear(monkeypatch):
    events = _record_cache_calls(monkeypatch)
    module = types.ModuleType("mlx_vlm.generate")
    def module_getattr(name):
        raise RuntimeError(f"lazy attribute {name} exploded")
    module.__getattr__ = module_getattr
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", module)

    with _generation_cache_hygiene():
        events.append("body")

    assert events == [
        "synchronize:default", "clear_cache", "body", "synchronize:default", "clear_cache",
    ]

def test_a_runtime_without_synchronize_still_clears(monkeypatch):
    events = _record_cache_calls(monkeypatch)
    monkeypatch.setattr("mlx.core.synchronize", None, raising=False)
    _fake_stream_module(monkeypatch, "mlx_lm.generate", "stream<mlx_lm>")

    with _generation_cache_hygiene():
        events.append("body")

    assert events == ["clear_cache", "body", "clear_cache"]

def test_the_exit_clear_still_drains_when_the_body_raised(monkeypatch):
    events = _record_cache_calls(monkeypatch)
    with pytest.raises(ValueError, match="body"):
        with _generation_cache_hygiene():
            raise ValueError("body")
    assert events == [
        "synchronize:default", "clear_cache", "synchronize:default", "clear_cache",
    ]

@pytest.mark.parametrize("metal", [
    types.SimpleNamespace(),
    # Refused even when the pre-0.22 shape is present: this path never accepted it.
    types.SimpleNamespace(clear_cache=lambda: None),
])
def test_missing_clear_cache_is_rejected_before_any_work(monkeypatch, metal):
    events = _record_cache_calls(monkeypatch, has_clear_cache=False)
    monkeypatch.setattr("mlx.core.metal", metal, raising=False)
    with pytest.raises(RuntimeError, match="clear_cache missing"):
        with _generation_cache_hygiene():
            pytest.fail("body must not run when the runtime cannot clear its cache")
    assert events == []

def test_training_flag_restore_attempts_every_module():
    class Module:
        def __init__(self, fails=False):
            self.training, self.fails = False, fails
        def _set_training_mode(self, mode):
            self.training = mode
            if self.fails:
                raise RuntimeError("module restore failed")

    first, later = Module(True), Module()
    with pytest.raises(RuntimeError, match="module restore failed"):
        _restore_training_flags([(first, True), (later, True)])
    assert later.training is True

def test_adapter_removes_stop_matches_and_closes_on_failure():
    class Generator:
        instances, fail, eos, close_fail = [], False, False, False
        def __init__(self, *_args, **_kwargs):
            self.removed, self.closed, self.kwargs = [], False, _kwargs
            self.instances.append(self)
        def insert(self, *_args, **_kwargs):
            return [7]
        def next_generated(self):
            if self.fail:
                raise RuntimeError("event failure")
            def event(token):
                return types.SimpleNamespace(
                    uid=7, token=token, logprobs=[-9.0, -0.1, -0.2, -0.3],
                    finish_reason=None,
                )
            if self.eos:
                terminal = event(3)
                terminal.finish_reason = "stop"
                return [event(1), terminal]
            return [event(1), event(2), event(3)]
        def remove(self, uids):
            self.removed.extend(uids)
        def close(self):
            self.closed = True
            if self.close_fail:
                raise RuntimeError("close failure")

    tokenizer, adapter = _CharTokenizer(), object.__new__(_TextBatchAdapter)
    tokenizer.eos_token_ids = ()
    adapter.model, adapter.tokenizer = object(), tokenizer
    adapter.defaults = GenerationDefaults(stop_strings=("<STOP>",), prefill_batch_size=2, completion_batch_size=3, max_kv_size=16)
    adapter.batch_generator_type = Generator
    adapter.make_sampler = lambda **_kwargs: None
    result = _results(adapter.stream([GenerationRequest(prompt_token_ids=[9])]))[0]
    assert result.token_ids == [1] and Generator.instances[-1].removed == [7]
    assert tuple(Generator.instances[-1].kwargs[name] for name in ("prefill_batch_size", "completion_batch_size", "max_kv_size")) == (2, 3, 16)
    assert Generator.instances[-1].closed is True
    Generator.eos = True
    eos = _results(adapter.stream([GenerationRequest(prompt_token_ids=[9])]))[0]
    assert eos.token_ids == [1] and eos.finish_reason == "stop"
    Generator.eos = False
    Generator.fail = True
    Generator.close_fail = True
    with pytest.warns(RuntimeWarning, match=r"close\(\) failed"):
        with pytest.raises(RuntimeError, match="event failure"):
            _results(adapter.stream([GenerationRequest(prompt_token_ids=[9])]))
    assert Generator.instances[-1].closed is True

def test_generation_mode_releases_lock_when_limit_restore_raises(monkeypatch):
    model = types.SimpleNamespace(training=True)
    model.named_modules = lambda: [("", model)]
    model.eval = lambda: setattr(model, "training", False)

    # Patch the loaded module: a string target reimports unsloth_zoo, whose
    # install guard fires on a CPU checkout without the unsloth package.
    from unsloth_zoo.mlx import generate as generate_module
    monkeypatch.setattr(generate_module, "_snapshot_metal_limits", lambda: {"memory": 1})
    monkeypatch.setattr(
        generate_module,
        "_restore_metal_limits",
        lambda _snapshot: (_ for _ in ()).throw(RuntimeError("restore failed")),
    )
    with pytest.raises(RuntimeError, match="restore failed"):
        with generation_mode(model):
            pass
    def acquire_elsewhere():
        acquired = _GENERATION_MODE_LOCK.acquire(timeout=1)
        if acquired:
            _GENERATION_MODE_LOCK.release()
        return acquired
    with concurrent.futures.ThreadPoolExecutor() as executor:
        assert executor.submit(acquire_elsewhere).result()
    assert model.training is True

def test_fast_generate_binds_text_and_vision_models_and_maps_shared_controls(monkeypatch):
    from unsloth_zoo.mlx import generate as generate_module
    from unsloth_zoo.mlx.loader import _patch_mlx_saving
    captured, expected = {}, [object()]
    def fake_generate(model, tokenizer, requests, *, defaults):
        captured.update(model=model, tokenizer=tokenizer, requests=requests, defaults=defaults)
        return expected
    monkeypatch.setattr(generate_module, "generate_batch", fake_generate)
    tokenizer, model = object(), types.SimpleNamespace(_is_vlm_model=False)
    _patch_mlx_saving(model, tokenizer)
    result = model.fast_generate(
        ("one", "two"), max_tokens=7, temperature=0.5,
        prefill_batch_size=2, completion_batch_size=3, max_kv_size=32,
    )
    assert result is expected
    assert [request.prompt for request in captured["requests"]] == ["one", "two"]
    assert (captured["defaults"].max_tokens, captured["defaults"].sampling.temperature) == (7, 0.5)
    assert (captured["defaults"].prefill_batch_size, captured["defaults"].completion_batch_size, captured["defaults"].max_kv_size) == (2, 3, 32)
    with pytest.raises(TypeError, match="every fast_generate prompt"):
        model.fast_generate(["valid", 3])
    # A vLLM prompt dict iterates to its keys, so it must be refused by type.
    with pytest.raises(TypeError, match="not vLLM prompt dicts"):
        model.fast_generate({"prompt": "hi", "multi_modal_data": {}})
    # KV-cache quantisation is never forwarded, so it is not a parameter.
    for control in ("kv_bits", "kv_group_size"):
        with pytest.raises(TypeError):
            model.fast_generate(["one"], **{control: 4})
    vlm = types.SimpleNamespace(_is_vlm_model=True)
    _patch_mlx_saving(vlm, tokenizer)
    assert vlm.fast_generate(["look"]) is expected
    assert vlm.fast_generate(["look"]) is expected


@contextlib.contextmanager
def _entry_ctx(order):
    order.append("wired")
    yield


@contextlib.contextmanager
def _wired_ctx(order):
    order.append("held")
    try:
        yield
    finally:
        order.append("released")


def _vlm_stub(per_row, *, events):
    """BatchGenerator stand-in: pre-0.5 nests Response with a logprob vector,
    0.5+ moves it to GenerationBatch with a scalar token_logprob."""
    class _Response:
        __dataclass_fields__ = dict.fromkeys(
            ("uid", "token", "finish_reason",
             "token_logprob" if per_row else "logprobs"))
        def __init__(self, uid, token, finish_reason):
            self.uid, self.token, self.finish_reason = uid, token, finish_reason
            if per_row:
                self.token_logprob = -0.5
            else:
                self.logprobs = {token: -0.5}
    class Generator:
        if not per_row:
            Response = _Response
        GenerationBatch = types.SimpleNamespace(Response=_Response)
        def __init__(self, model, processor, **kwargs):
            self._prompt_batch = None
            self._generation_batch = []
            self._unprocessed_sequences = []
            self._polls = 0
        def insert(self, prompts, max_tokens, **kwargs):
            return list(range(len(prompts)))
        @property
        def has_work(self):
            return self._polls < len(events)
        def next(self, **kwargs):
            batch = events[self._polls] if self._polls < len(events) else []
            self._polls += 1
            emitted = [_Response(*item) for item in batch]
            return (([], emitted) if per_row else emitted)
        def close(self):
            pass
    if per_row:
        Generator.insert.__signature__ = inspect.Signature([
            inspect.Parameter("prompts", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("max_tokens", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("prompt_kwargs", inspect.Parameter.KEYWORD_ONLY, default=None),
        ])
    return Generator


def test_inverted_batch_sizes_are_rejected_for_vision_but_kept_for_text():
    # Vision refuses an inverted pair rather than hang; mlx-lm normalizes it.
    from unsloth_zoo.mlx.generate import _validate_text_requests, _validate_vlm_requests
    inverted = GenerationDefaults(prefill_batch_size=16, completion_batch_size=8)  # noqa: E501
    with pytest.raises(ValueError, match="must not exceed completion_batch_size"):
        _validate_vlm_requests([GenerationRequest(prompt="a")], inverted)
    assert _validate_text_requests([GenerationRequest(prompt="a")], inverted)


def test_vlm_requests_accept_stop_strings():
    # Reinstating the old blanket rejection left every _drive-level test green.
    from unsloth_zoo.mlx.generate import _validate_vlm_requests
    assert _validate_vlm_requests([GenerationRequest(prompt="a")],
                                  GenerationDefaults(stop_strings=("STOP",)))


def test_vlm_requests_reject_token_id_prompts_and_unsupported_controls():
    from unsloth_zoo.mlx.generate import _validate_vlm_requests
    cases = ((GenerationRequest(prompt_token_ids=[1]), GenerationDefaults(), "rendered prompt string"),
        (GenerationRequest(prompt=""), GenerationDefaults(), "must not be empty"),
        (GenerationRequest(prompt="a"), GenerationDefaults(max_kv_size=64), "max_kv_size is not supported"),
        # One mlx-vlm would drop rather than apply is refused: an inert start or bogus scheme,
        # a group size on a turboquant cache, an explicit scheme that fractional bits override.
        *((GenerationRequest(prompt="a"), GenerationDefaults(**kw), f"{name} would be dropped")
          for kw, name in (({"kv_bits": 4, "quantized_kv_start": 100}, "quantized_kv_start"), ({"kv_bits": 4, "kv_quant_scheme": "bogus"}, "kv_quant_scheme"), ({"kv_bits": 4, "kv_quant_scheme": "turboquant", "kv_group_size": 32}, "kv_group_size"), ({"kv_bits": 4.5, "kv_quant_scheme": "uniform"}, "kv_quant_scheme"))))  # noqa: E501
    for request, defaults, message in cases:
        with pytest.raises(ValueError, match=message):
            _validate_vlm_requests([request], defaults)
    # Forwarded, not refused: mlx-vlm's cache builder applies these.
    accepted = ({"kv_group_size": 32}, {"kv_quant_scheme": "uniform"}, {"kv_quant_scheme": "turboquant"}, {"kv_bits": 4.5}, {"kv_bits": 4.5, "quantized_kv_start": 9})  # noqa: E501
    assert all(_validate_vlm_requests([GenerationRequest(prompt="a")], GenerationDefaults(**{"kv_bits": 4, **kw})) for kw in accepted)  # noqa: E501
    # Path images normalize to str once: mlx-vlm's loader opens str only, and
    # grouping and preprocessing must see the same value.
    assert _validate_vlm_requests([GenerationRequest(prompt="a", image=pathlib.Path("/tmp/i.png"))], GenerationDefaults())[0].image == "/tmp/i.png"  # noqa: E501


def test_prompt_kwargs_fallback_splits_mrope_on_the_batch_axis():
    # MRoPE position_ids are [3, batch, sequence]; axis 0 would hand every row
    # the same rope section.
    from unsloth_zoo.mlx.generate import _split_prompt_kwargs_fallback
    class Fake:
        def __init__(self, shape): self.shape, self.ndim = shape, len(shape)
        def __getitem__(self, key): return ("sliced", key)
    rows = _split_prompt_kwargs_fallback(
        {"position_ids": Fake((3, 2, 5)), "image_position_ids": Fake((2, 4, 5)),
         "inputs_embeds": Fake((2, 5, 7)), "scale": 3}, 2)
    assert rows[1]["position_ids"][1] == (slice(None), slice(1, 2), slice(None))
    assert rows[1]["image_position_ids"][1] == slice(1, 2)  # batch-first, not MRoPE
    assert rows[1]["inputs_embeds"][1] == slice(1, 2) and rows[0]["scale"] == 3


def test_vlm_adapter_reports_admission_stall_without_counting_polls():
    # Prefill yields eventless polls, so only impossible admission is a stall.
    from unsloth_zoo.mlx.generate import _VLMBatchAdapter
    adapter = _VLMBatchAdapter.__new__(_VLMBatchAdapter)
    adapter.defaults = GenerationDefaults()
    def state(prompt_batch):
        return types.SimpleNamespace(_prompt_batch=prompt_batch,
                                     _generation_batch=[], _unprocessed_sequences=[1])
    assert (adapter._admission_stalled(state(object())), adapter._admission_stalled(
        state(None)), adapter._admission_stalled(types.SimpleNamespace())) == (
        False, True, None)


@pytest.mark.parametrize("per_row", (False, True))
def test_vlm_adapter_drives_both_upstream_protocols(per_row):
    # Pre-0.5 polls next() until empty; later ones own has_work and return pairs.
    from unsloth_zoo.mlx.generate import _VLMBatchAdapter
    generator = _vlm_stub(per_row, events=[[(0, 1, None)], [(0, 3, "stop")]])(
        object(), object())
    adapter = _VLMBatchAdapter.__new__(_VLMBatchAdapter)
    adapter.per_row_prompt_kwargs = per_row
    adapter.defaults = GenerationDefaults()
    adapter.processor = _CharTokenizer()
    results = _results(adapter._drive(generator, [0], [0], {}))
    assert ([r.token_ids for r in results], results[0].finish_reason) == ([[1]], "stop")


def test_vlm_adapter_isolates_detokenizers_and_reads_scalar_logprobs():
    # A shared-detokenizer processor must not interleave sequences into one buffer.
    from unsloth_zoo.mlx.generate import _VLMBatchAdapter
    class SharedDetokenizerTokenizer(_CharTokenizer):
        def __init__(self):
            self._shared = _CharDetokenizer(self)
        @property
        def detokenizer(self):
            return self._shared
    processor = types.SimpleNamespace(tokenizer=SharedDetokenizerTokenizer())
    adapter = _VLMBatchAdapter.__new__(_VLMBatchAdapter)
    adapter.per_row_prompt_kwargs = True
    adapter.defaults = GenerationDefaults()
    adapter.processor = processor
    generator = _vlm_stub(True, events=[
        [(0, 1, None), (1, 4, None)], [(0, 3, "stop"), (1, 3, "stop")]])(
        object(), object())
    results = _results(adapter._drive(generator, [0, 1], [0, 1], {}, prompt_token_count=37))
    # Distinct buffers: a shared one concatenates both sequences into the first.
    assert [(r.token_ids, r.text) for r in results] == [
        ([1], "hello "), ([4], "tailSTOPsuffix")] and results[0].logprobs == [-0.5]
    assert [r.prompt_token_count for r in results] == [37, 37]


def test_vlm_chunking_respects_release_capabilities():
    # Older releases run one chunk per generator to avoid row-zero reuse.
    from unsloth_zoo.mlx.generate import _VLMBatchAdapter
    adapter = _VLMBatchAdapter.__new__(_VLMBatchAdapter)
    adapter.defaults = GenerationDefaults(prefill_batch_size=2, completion_batch_size=4)
    adapter.batches_observable = True
    chunks = []
    adapter._group_key = lambda request: "same"
    adapter._run_chunk = lambda requests, indices: (
        chunks.append(list(indices))
        or [GenerationEvent(index = index, result = object()) for index in indices])
    requests = [GenerationRequest(prompt="p") for _ in range(5)]
    for per_row, expected in ((True, [[0, 1, 2, 3, 4]]), (False, [[0, 1], [2, 3], [4]])):
        chunks.clear(); adapter.per_row_prompt_kwargs = per_row  # noqa: E702
        _results(adapter.stream(requests))
        assert chunks == expected


def test_vlm_run_chunk_builds_the_generator_after_embeddings():
    from unsloth_zoo.mlx.generate import _VLMBatchAdapter
    order, seen = [], {}
    ids, embeds = (types.SimpleNamespace(tolist=lambda: [[1], [2]]),
                   types.SimpleNamespace(to_dict=lambda: {"inputs_embeds": "E", "position_ids": None}))
    adapter = _VLMBatchAdapter.__new__(_VLMBatchAdapter)
    adapter.defaults = GenerationDefaults(kv_bits=4, kv_quant_scheme="turboquant")
    adapter.per_row_prompt_kwargs = True
    adapter.processor = types.SimpleNamespace(tokenizer=_CharTokenizer())
    adapter.constructor_params = {"prefill_batch_size", "completion_batch_size", "sampler", "stop_tokens", "kv_bits", "kv_quant_scheme", "quantized_kv_start"}  # noqa: E501
    adapter.prepare_inputs = lambda *a, **k: {"input_ids": ids, "extra": "D", "position_ids": "P"}  # noqa: E501
    adapter.make_sampler = lambda **k: object()
    adapter._add_special_tokens = lambda: True
    adapter._drive = lambda *a, **k: iter(
        [GenerationEvent(index = row, result = k["prompt_token_count"]) for row in a[2]]
    )
    adapter._split_prompt_kwargs = lambda kw, n: (seen.update(kwargs=kw), [{}] * n)[1]
    # Recording on ENTRY proves embeddings run inside the wired-limit context.
    adapter._wired_limit = lambda: _entry_ctx(order)
    adapter._chunked_prefill_kwargs = lambda **kw: (
        order.append("policy"), seen.update(policy=kw), {})[2]
    adapter.model = types.SimpleNamespace(
        config=types.SimpleNamespace(image_token_index=None), language_model=object(),
        get_input_embeddings=lambda *a, **k: (order.append("embed"), embeds)[1])
    adapter.generator_type = lambda *a, **k: (order.append("construct"),
        seen.update(ctor=k), _vlm_stub(True, events=[[(0, 1, "stop")]])(*a[:2]))[2]
    assert [event.result for event in
            adapter._run_chunk([GenerationRequest(prompt = "p")] * 2, [0, 1])] == [1, 1]
    assert order == ["wired", "embed", "policy", "construct"]
    assert seen["policy"]["prefill_kwargs"]["inputs_embeds"] == "E"
    assert seen["kwargs"]["extra"] == "D"
    # A None embedding field must not overwrite a valid preprocessing value.
    assert seen["kwargs"]["position_ids"] == "P"
    # The KV-quant controls reach the constructor that builds the cache, turboquant from token one.
    assert [seen["ctor"][n] for n in ("kv_bits", "kv_quant_scheme", "quantized_kv_start")] == [4, "turboquant", 0]
    # Per-row keeps configured sizes; EOS stays with the processor.
    assert (seen["ctor"]["prefill_batch_size"], seen["ctor"]["completion_batch_size"]) == (8, 32)
    assert "stop_tokens" not in seen["ctor"]


@pytest.mark.parametrize("nested", (True, False))
def test_vlm_probe_accepts_both_event_class_locations(nested):
    from unsloth_zoo.mlx.generate import _probe_vlm_api
    response = type("Response", (), {"__dataclass_fields__": dict.fromkeys(
        ("uid", "token", "finish_reason", "logprobs" if nested else "token_logprob"))})
    class Generator:
        def __init__(self, model, processor, prefill_batch_size=1, completion_batch_size=1): pass  # noqa: E501
        def insert(self, prompts, max_tokens): pass
        def next(self): pass
    if nested:
        Generator.Response = response
    module = types.SimpleNamespace(BatchGenerator=Generator, **(
        {} if nested else {"GenerationBatch": types.SimpleNamespace(Response=response)}))
    assert _probe_vlm_api(module)


def test_vlm_prompt_kwargs_prefer_upstream_only_when_mrope_aware():
    # An axis-0 upstream splitter is refused in favour of the local MRoPE one.
    from unsloth_zoo.mlx.generate import _VLMBatchAdapter
    adapter, calls = _VLMBatchAdapter.__new__(_VLMBatchAdapter), []
    adapter.batch_module = types.SimpleNamespace(
        _split_prompt_kwargs_per_row=lambda kwargs, size: calls.append(size) or ["up"])
    assert adapter._split_prompt_kwargs({}, 2) == [{}, {}] and not calls
    adapter.batch_module._is_mrope_position_ids_prompt_kwarg = lambda key, value: False
    assert adapter._split_prompt_kwargs({}, 2) == ["up"] and calls == [2]


def test_vlm_images_are_decoded_once_and_flow_to_preprocessing():
    from unsloth_zoo.mlx.generate import _VLMBatchAdapter
    loads, seen = [], {}
    decoded = types.SimpleNamespace(size=(8, 8))
    adapter = _VLMBatchAdapter.__new__(_VLMBatchAdapter)
    adapter.defaults, adapter.per_row_prompt_kwargs = GenerationDefaults(), True
    adapter.processor = types.SimpleNamespace(tokenizer=_CharTokenizer())
    adapter.batches_observable = True
    adapter.process_image = lambda src, shape, proc: loads.append(src) or decoded
    adapter._run_chunk = lambda requests, indices: (
        seen.update(image=requests[indices[0]].image),
        [GenerationEvent(index=indices[0], result=object())])[1]
    _results(adapter.stream([GenerationRequest(prompt="p", image="/tmp/one-use.png")]))
    assert loads == ["/tmp/one-use.png"] and seen["image"] is decoded
    with pytest.raises(ValueError, match="raw bytes"):
        _results(adapter.stream([GenerationRequest(prompt="p", image=b"\x89PNG")]))


def test_vlm_adapter_initializes_against_newer_module_layout(monkeypatch):
    # Newer layout: the package re-exports BatchGenerator while the helpers and the
    # event class live on the defining module. __getattr__ delegation is left out so
    # this proves the defining-module binding alone.
    import sys
    import mlx_vlm.utils  # noqa: F401  (cache the real module before shadowing)
    from unsloth_zoo.mlx import generate as engine
    response = type("Response", (), {"__dataclass_fields__": dict.fromkeys(
        ("uid", "token", "finish_reason", "token_logprob"))})
    class Generator:
        def __init__(self, model, processor, prefill_batch_size=1,
                     completion_batch_size=1, compute_logprobs=False): pass
        def insert(self, prompts, max_tokens, prompt_kwargs=None,
                   logits_processors=None): pass
        def next(self): pass
        def remove(self, uid): pass
        def close(self): pass
    Generator.__module__ = "mlx_vlm.generate.ar"
    ar = types.ModuleType("mlx_vlm.generate.ar")
    ar.BatchGenerator = Generator
    ar.GenerationBatch = types.SimpleNamespace(Response=response)
    ar._split_prompt_kwargs_per_row = lambda kwargs, size: [{}] * size
    ar._is_mrope_position_ids_prompt_kwarg = lambda key, value: False
    policy_calls = []
    ar._chunked_prefill_enabled = lambda model, **kwargs: policy_calls.append(model) or True
    bare = types.ModuleType("mlx_vlm.generate")
    bare.BatchGenerator = Generator
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", bare)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate.ar", ar)
    adapter = engine._VLMBatchAdapter(object(), _CharTokenizer(), GenerationDefaults())
    # Binds the DEFINING module although the re-exporting one matched first.
    assert adapter.batch_module is ar and adapter.per_row_prompt_kwargs
    assert adapter._split_prompt_kwargs({}, 3) == [{}, {}, {}]  # upstream splitter chosen
    assert adapter._chunked_prefill_kwargs(input_ids=None, prefill_kwargs={}) == {}
    assert policy_calls
    ar._chunked_prefill_enabled = lambda model, **kwargs: False
    assert adapter._chunked_prefill_kwargs(
        input_ids=None, prefill_kwargs={}) == {"prefill_step_size": None}
    # The probed capability must reach the adapter, not merely exist on the module.
    assert adapter.cancel is not None


def test_module_resolution_separates_an_absent_release_from_a_broken_install(monkeypatch):
    # A missing dependency of a candidate must not be reported as an old release.
    import importlib as importlib_module
    from unsloth_zoo.mlx.generate import _resolve_module_attr
    missing = {"mlx_vlm.generate": "timm", "mlx_vlm.generate.ar": "mlx_vlm.generate.ar"}
    def fake_import(name):
        raise ModuleNotFoundError(f"No module named {missing[name]!r}", name=missing[name])
    monkeypatch.setattr(importlib_module, "import_module", fake_import)
    with pytest.raises(ModuleNotFoundError, match="timm"):
        _resolve_module_attr(("mlx_vlm.generate",), "BatchGenerator")
    assert _resolve_module_attr(("mlx_vlm.generate.ar",), "BatchGenerator") is None


def test_vlm_drive_raises_on_true_stall_and_survives_long_prefill():
    from unsloth_zoo.mlx.generate import _VLMBatchAdapter
    adapter = _VLMBatchAdapter.__new__(_VLMBatchAdapter)
    adapter.defaults, adapter.per_row_prompt_kwargs = GenerationDefaults(), True
    adapter.processor = types.SimpleNamespace(tokenizer=_CharTokenizer())
    base = _vlm_stub(True, events=[[(0, 1, "stop")]])
    class LongPrefill(base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._prompt_batch, self._unprocessed_sequences = object(), [1]
            self._empty_polls = 3
        @property
        def has_work(self):
            return self._empty_polls > 0 or self._polls < 1
        def next(self, **kwargs):
            if self._empty_polls:
                self._empty_polls -= 1
                if not self._empty_polls:
                    self._prompt_batch, self._generation_batch = None, [1]
                return ([], [])
            return super().next(**kwargs)
    assert _results(
        adapter._drive(LongPrefill(object(), object()), [0], [0], {})
    )[0].finish_reason == "stop"
    class Stalled(base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._prompt_batch, self._generation_batch = None, []
            self._unprocessed_sequences = [1]
        has_work = True
        def next(self, **kwargs):
            return ([], [])
    with pytest.raises(RuntimeError, match="admitted no prompt"):
        _results(adapter._drive(Stalled(object(), object()), [0], [0], {}))


@pytest.mark.parametrize("plural", (True, False))
def test_vlm_stop_strings_trim_and_cancel_in_the_release_form(plural):
    from unsloth_zoo.mlx.generate import _VLMBatchAdapter, _resolve_cancel
    cancelled = []
    class Generator(_vlm_stub(True, events=[[(0, 2, None)], [(0, 3, None)]])):
        if plural:
            def remove(self, uids): cancelled.append(list(uids))
        else:
            def remove(self, uid): cancelled.append([uid])
    adapter = _VLMBatchAdapter.__new__(_VLMBatchAdapter)
    adapter.defaults = GenerationDefaults(stop_strings=("STOP",))
    adapter.per_row_prompt_kwargs = True
    adapter.processor = types.SimpleNamespace(tokenizer=_CharTokenizer())
    adapter.cancel = _resolve_cancel(Generator)
    result = _results(adapter._drive(Generator(object(), object()), [0], [0], {}))[0]
    # "<ST" + "OP>" completes the stop string: trimmed back to the boundary.
    assert (result.token_ids, result.finish_reason, result.stop_match) == ([], "stop_string", "STOP")
    assert cancelled == [[0]]


def test_vlm_stop_strings_drop_later_events_when_the_release_cannot_cancel():
    # Without remove(), a stopped sequence keeps decoding: its later events must
    # not reach its result, and the sibling must still complete.
    from unsloth_zoo.mlx.generate import _VLMBatchAdapter, _resolve_cancel
    Generator = _vlm_stub(False, events=[
        [(0, 2, None), (1, 1, None)],
        [(0, 3, None), (1, 1, None)],       # uid 0 completes "<STOP>" here
        [(0, 1, None), (1, 1, "length")]])
    adapter = _VLMBatchAdapter.__new__(_VLMBatchAdapter)
    adapter.defaults = GenerationDefaults(stop_strings=("STOP",))
    adapter.per_row_prompt_kwargs = False
    adapter.processor = types.SimpleNamespace(tokenizer=_CharTokenizer())
    adapter.cancel = _resolve_cancel(Generator)
    assert adapter.cancel is None
    stopped, sibling = _results(
        adapter._drive(Generator(object(), object()), [0, 1], [0, 1], {}))
    assert (stopped.token_ids, stopped.finish_reason, stopped.stop_match) == (
        [], "stop_string", "STOP")
    assert (sibling.token_ids, sibling.finish_reason) == ([1, 1, 1], "length")


def test_generation_mode_serializes_threads_and_rejects_overlapping_tasks():
    # Process-global MLX state is shared by neither threads nor async tasks.
    import asyncio
    import concurrent.futures
    import threading
    from unsloth_zoo.mlx.generate import generation_mode
    model = types.SimpleNamespace(training=True, eval=lambda: None)
    model.named_modules = lambda: [("", model)]
    held, release = threading.Event(), threading.Event()
    def hold():
        with generation_mode(model):
            held.set()
            release.wait(5)  # held open until the probe has run: no timing race
    with concurrent.futures.ThreadPoolExecutor() as pool:
        task = pool.submit(hold)
        try:
            assert held.wait(5)
            acquired = _GENERATION_MODE_LOCK.acquire(True, 0.05)
            if acquired:
                _GENERATION_MODE_LOCK.release()
            assert acquired is False
        finally:
            release.set()
            task.result()
    async def overlap():
        async def inner():
            with generation_mode(model):
                pass
        with generation_mode(model):
            with pytest.raises(RuntimeError, match="async tasks"):
                await asyncio.create_task(inner())
    asyncio.run(overlap())


def test_detokenizer_buffers_partial_characters_and_emits_once():
    # A replacement character is an incomplete sequence: it contributes nothing
    # until resolved, then appears exactly once.
    from unsloth_zoo.mlx.generate import _PendingResult, _StopStringScanner
    tokenizer = _TableTokenizer({(1,): "hi ", (1, 2): "hi \ufffd", (1, 2, 3): "hi ok"})
    state = _PendingResult(detokenizer=_new_detokenizer(tokenizer),
                           scanner=_StopStringScanner(()))
    for token in (1, 2, 3):
        state.append(tokenizer, token, -0.1)
    state.finish(tokenizer, "length")
    assert state.result(tokenizer).text == "hi ok"


def test_detokenizer_never_re_emits_after_a_shortening_decode():
    from unsloth_zoo.mlx.generate import _PendingResult, _StopStringScanner
    tokenizer = _TableTokenizer({(1,): "hello", (1, 2): "he", (1, 2, 3): "hello there"})
    state = _PendingResult(detokenizer=_new_detokenizer(tokenizer),
                           scanner=_StopStringScanner(()))
    for token in (1, 2, 3):
        state.append(tokenizer, token, -0.1)
    state.finish(tokenizer, "length")
    assert state.result(tokenizer).text == "hello there"


class _EosTokenizer(_CharTokenizer):
    eos_token_id = 3

def _sample_utils():
    """The release's filtering stages, as a seeded sampler reaches for them."""
    return types.SimpleNamespace(
        apply_top_p=lambda x, p: x, apply_min_p=lambda x, p, n: x,
        apply_top_k=lambda x, k: x,
    )

def _results(events):
    out = {}
    for event in events:
        if event.result is not None:
            out[event.index] = event.result
    return [out[index] for index in sorted(out)]


def _audio_events(*specs):
    """Sequential events; `finish` omitted models releases carrying no reason.

    Each event carries a distinct logprob, so committing a successor's value
    against its predecessor's token shifts alignment visibly.
    """
    made = []
    for position, (token, finish) in enumerate(specs):
        fields = {
            "token": token,
            "logprobs": {token: -0.5 - position},
            "prompt_tokens": 40,
        }
        if finish is not None:
            fields["finish_reason"] = finish
        made.append(types.SimpleNamespace(**fields))
    return made

def _audio_adapter(events, tokenizer=None, stops=()):
    from unsloth_zoo.mlx.generate import _VLMBatchAdapter
    adapter = _VLMBatchAdapter.__new__(_VLMBatchAdapter)
    adapter.defaults = GenerationDefaults(stop_strings=stops)
    adapter.processor = types.SimpleNamespace(tokenizer=tokenizer or _CharTokenizer())
    adapter.model = object()
    adapter.sampler = object()
    adapter.sampler_kwargs = {}
    adapter.make_sampler = lambda **k: (adapter.sampler_kwargs.update(k), adapter.sampler)[1]
    adapter.sample_utils = _sample_utils()
    adapter._wired_limit = contextlib.nullcontext
    adapter.stream_calls = []
    adapter.audio_warn_stacklevel = 3
    def stream(*args, **kwargs):
        adapter.stream_calls.append((args, kwargs))
        return iter(events)
    adapter.stream_module = types.SimpleNamespace(stream_generate=stream)
    return adapter


def _audio_row(adapter, request):
    """The one row an audio request produces, and the deltas it streamed."""
    events = list(adapter._stream_sequentially(0, request))
    assert all(event.index == 0 for event in events)
    return events[-1].result, "".join(event.delta for event in events)


class _CriteriaTokenizer(_CharTokenizer):
    # Processors stop on ids their eos attribute never lists, so these disagree.
    eos_token_id = 1
    stopping_criteria = staticmethod(lambda token: token == 3)


@pytest.mark.parametrize("specs,tokenizer,expected", [
    # The terminal event repeats the last token; committing it would duplicate.
    (((1, None), (2, None), (2, None)), None, ([1, 2], "length")),
    # No finish reason on the terminal event, but its token is end-of-sequence.
    (((1, None), (3, None)), _EosTokenizer(), ([1], "stop")),
    # The release stops through criteria the tokenizer's eos id does not list.
    (((1, None), (3, None)), _CriteriaTokenizer(), ([1], "stop")),
    # ... and that same authority reports an ordinary token as a length cut-off.
    (((1, None), (2, None), (2, None)), _CriteriaTokenizer(), ([1, 2], "length")),
    (((1, None), (2, "stop")), None, ([1], "stop")),
    (((1, None), (3, "length")), _CriteriaTokenizer(), ([1], "length")),
    # Nothing generated at all: newer releases emit one event with no token.
    (((None, "length"),), None, ([], "length")),
])
def test_audio_fallback_keeps_the_batched_result_contract(specs, tokenizer, expected):
    result, streamed = _audio_row(
        _audio_adapter(_audio_events(*specs), tokenizer),
        GenerationRequest(prompt="p", audio="a.wav"))
    assert (result.token_ids, result.finish_reason) == expected
    assert streamed == result.text
    # Logprobs must belong to the tokens they sit beside, not to their successors.
    assert result.logprobs == [-0.5 - n for n in range(len(result.token_ids))]
    assert result.text == "".join(_CharTokenizer.pieces[t] for t in result.token_ids)


def test_audio_terminal_event_flushes_text_held_back_during_streaming():
    # Cleanup withholds a trailing space until finalize; losing it truncates.
    tokenizer = _TableTokenizer({(1,): "a ", (1, 2): "a b "}, clean=True)
    result, streamed = _audio_row(
        _audio_adapter(_audio_events((1, None), (2, None), (2, None)), tokenizer),
        GenerationRequest(prompt="p", audio="a.wav"))
    assert (result.token_ids, result.text) == ([1, 2], "a b ")
    assert streamed == result.text


def _shows(processor, ids):
    import mlx.core as mx

    return processor(mx.array(ids), None)


def _pin(token_id, shown = None):
    """A processor with a visible effect, recording the history it was shown."""
    def _processor(tokens, logits):
        if shown is not None:
            shown.append(tokens.tolist())
        return token_id
    return _processor


def test_audio_fallback_forwards_the_request_to_the_stream():
    # A dropped audio payload or ignored control shows up in the stream call.
    adapter = _audio_adapter(_audio_events((1, None), (2, "length")))
    result, _ = _audio_row(adapter, GenerationRequest(
        prompt="p", audio="a.wav", max_tokens=9,
        sampling=SamplingParams(temperature=0.25, top_k=7)))
    assert result.prompt_token_count == 40
    args, kwargs = adapter.stream_calls[0]
    assert args[2] == "p"  # the rendered prompt, after model and processor
    assert (kwargs["audio"], kwargs["max_tokens"]) == ("a.wav", 9)
    assert kwargs["sampler"] is adapter.sampler
    assert (adapter.sampler_kwargs["temp"], adapter.sampler_kwargs["top_k"]) == (0.25, 7)
    assert kwargs["logits_processors"] == []
    # Decoding alone is not a reason to drop what the request asked for.
    shown = []
    _result, _ = _audio_row(adapter, GenerationRequest(
        prompt="p", audio="a.wav", logits_processors=[_pin(7, shown)]))
    audio = adapter.stream_calls[1][1]["logits_processors"]
    assert [_shows(f, [4, 5, 6]) for f in audio] == [7]
    # An audio row starts its history where every other row does.
    assert shown == [[6]]


def test_an_audio_row_decodes_with_the_quantization_the_batch_was_given():
    from unsloth_zoo.mlx.generate import _KV_QUANT_CONTROLS, _validate_vlm_requests
    # Each one validation accepts, as given: a sequential turboquant row keeps mlx-vlm's own start.
    for kw in ({"kv_bits": 4, "kv_group_size": 32, "kv_quant_scheme": "uniform"}, {"kv_bits": 4, "kv_quant_scheme": "turboquant", "quantized_kv_start": 9}, {"kv_bits": 4.5}):  # noqa: E501
        adapter = _audio_adapter(_audio_events((1, None), (2, "length")))
        adapter.defaults = GenerationDefaults(**kw)
        assert _validate_vlm_requests([GenerationRequest(prompt = "p", audio = "a.wav")], adapter.defaults)
        _audio_row(adapter, GenerationRequest(prompt = "p", audio = "a.wav"))
        assert {k: v for k, v in adapter.stream_calls[0][1].items() if k in _KV_QUANT_CONTROLS} == kw


def test_a_turboquant_batch_without_a_start_quantizes_from_the_first_token():
    from unsloth_zoo.mlx.generate import _batch_kv_quant_options
    for kw in ({"kv_bits": 4.5}, {"kv_bits": 4, "kv_quant_scheme": "turboquant"}):
        assert _batch_kv_quant_options(GenerationDefaults(**kw)) == {**kw, "quantized_kv_start": 0}
    assert _batch_kv_quant_options(GenerationDefaults(kv_bits = 4)) == {"kv_bits": 4}  # uniform: start stays mlx-vlm's


def test_audio_fallback_honours_stop_strings():
    # Audio reaches the same scanner: "<ST" + "OP>" completes the stop string.
    result, _streamed = _audio_row(
        _audio_adapter(_audio_events((2, None), (3, None), (3, "length")),
                       stops=("STOP",)),
        GenerationRequest(prompt="p", audio="a.wav"))
    assert (result.token_ids, result.finish_reason, result.stop_match) == (
        [], "stop_string", "STOP")


def test_audio_requests_reach_the_fallback_through_the_public_entry_point(monkeypatch):
    # The tests above drive the adapter directly and would stay green if the old
    # blanket audio rejection came back.
    from unsloth_zoo.mlx import generate as engine
    seen = []
    monkeypatch.setattr(engine._VLMBatchAdapter, "_decode_image", lambda self, r: r)
    monkeypatch.setattr(
        engine._VLMBatchAdapter, "_stream_sequentially",
        lambda self, index, request: iter(
            [GenerationEvent(index=index,
                             result=seen.append(request.audio) or "audio")]))
    model = types.SimpleNamespace(_is_vlm_model=True, training=False, eval=lambda: None)
    model.named_modules = lambda: [("", model)]
    request = [GenerationRequest(prompt="p", audio="a.wav")]
    with pytest.warns(RuntimeWarning, match="decode one at a time") as collected:
        results = engine.generate_batch(model, object(), request)
    assert (results, seen) == (["audio"], ["a.wav"])
    assert collected[0].filename == __file__
    with pytest.warns(RuntimeWarning, match="decode one at a time") as collected:
        _results(engine.stream_batch(model, object(), request))
    assert collected[0].filename == __file__
    # The at-most-one-media invariant has to survive audio becoming supported.
    with pytest.raises(ValueError, match="both an image and audio"):
        engine.generate_batch(model, object(),
                              [GenerationRequest(prompt="p", image=object(), audio="a.wav")])


def test_audio_requests_decode_alone_while_the_rest_of_the_batch_still_batches():
    from unsloth_zoo.mlx.generate import _VLMBatchAdapter
    adapter = _audio_adapter(_audio_events((1, None), (2, "length")))
    batched = []
    adapter._decode_image = lambda request: request
    adapter._group_key = lambda request: "g"
    adapter.per_row_prompt_kwargs = True
    adapter.batches_observable = True
    adapter._run_chunk = lambda chunk, indices: (
        batched.append(list(indices)),
        [GenerationEvent(index = index, result = "batched") for index in indices])[1]
    requests = [GenerationRequest(prompt="a"), GenerationRequest(prompt="b", audio="a.wav"),
                GenerationRequest(prompt="c")]
    with pytest.warns(RuntimeWarning, match="decode one at a time"):
        results = _results(_VLMBatchAdapter.stream(adapter, requests))
    assert batched == [[0, 2]]
    assert [results[0], results[2]] == ["batched", "batched"]
    assert results[1].token_ids == [1]


def test_vision_generation_resolves_the_saved_processor(monkeypatch):
    # A text-only multimodal load publishes its inner tokenizer, so the typed
    # core resolves the saved processor for both it and fast_generate.
    from unsloth_zoo.mlx.generate import generate_batch as engine_generate_batch
    from unsloth_zoo.mlx import generate as engine
    seen = {}
    class _Adapter:
        def __init__(self, model, tok, defaults, **kwargs): seen["tok"] = tok
        def stream(self, requests):
            return iter([GenerationEvent(index=0, result="ok")])
    monkeypatch.setattr(engine, "_VLMBatchAdapter", _Adapter)
    monkeypatch.setattr(engine, "generation_mode", lambda model: contextlib.nullcontext())
    processor = object()
    model = types.SimpleNamespace(_is_vlm_model=True, _processor=processor)
    assert engine_generate_batch(model, object(), [GenerationRequest(prompt="a")]) == ["ok"]
    assert seen["tok"] is processor


def _knob_sampler(**knobs):
    import mlx.core as mx
    token = (int(round(knobs["temp"] * 10)) + 10 * int(round(knobs["top_p"] * 10))
             + 100 * knobs["top_k"] + 1000 * int(round(knobs["min_p"] * 10)))
    return lambda logprobs: mx.array([token])


def _batched_generator(prompt_uids = None, generation_uids = None):
    def batch(uids):
        return None if uids is None else types.SimpleNamespace(uids = list(uids))

    return types.SimpleNamespace(_prompt_batch = batch(prompt_uids),
                                 _generation_batch = batch(generation_uids))


def _warm(temps):
    return [SamplingParams(temperature = temp) for temp in temps]


def test_a_per_row_sampler_places_each_row_by_the_batch_and_builds_its_own_knobs():
    """The batch's own row order decides placement, and every knob reaches its row."""
    import mlx.core as mx
    sampler = _PerRowSampler(
        _warm([0.1, 0.2, 0.3]),
        sample_utils_module = types.SimpleNamespace(),
        make_sampler = _knob_sampler,
    )
    sampler.bind_uids([10, 20, 30])
    sampler.row_uids = [30, 20, 10]
    logprobs = mx.zeros((3, 4))

    sampler.bind_generator(_batched_generator(generation_uids = [10, 20, 30]))
    assert sampler.sample_target(logprobs, row_ids = [0] * 3,
                                 positions = [1, 2, 3]).tolist() == [1, 2, 3]
    sampler.bind_generator(_batched_generator(generation_uids = [30, 10, 20]))
    assert sampler.sample_target(logprobs, row_ids = [0] * 3,
                                 positions = [4, 5, 6]).tolist() == [3, 1, 2]
    sampler.bind_generator(_batched_generator(generation_uids = [30, 20]))
    assert sampler.sample_target(mx.zeros((2, 4)), row_ids = [0] * 2,
                                 positions = [7, 8]).tolist() == [3, 2]

    knobs = _PerRowSampler(
        [SamplingParams(temperature = 0.5), SamplingParams(temperature = 0.5, top_p = 0.8),
         SamplingParams(temperature = 0.5, top_k = 3),
         SamplingParams(temperature = 0.5, min_p = 0.2)],
        sample_utils_module = types.SimpleNamespace(), make_sampler = _knob_sampler,
    )
    knobs.bind_uids([10, 20, 30, 40])
    knobs.bind_generator(_batched_generator(generation_uids = [10, 20, 30, 40]))
    assert knobs.sample_target(mx.zeros((4, 4)), row_ids = [0] * 4,
                               positions = [1, 2, 3, 4]).tolist() == [5, 85, 305, 2005]


_UNREADABLE: dict = {}
exec(compile(
    "def draw(self):\n"
    "    return _sample_with_positions(self.sampler, None, row_ids=[0], positions=[1])\n",
    "<generated>", "exec"), _UNREADABLE)


def test_a_row_is_reported_as_it_goes_and_says_what_it_consumed_and_produced():
    """Deltas as they settle then the result once, with running counts on every event."""
    from unsloth_zoo.mlx.generate import _VLMBatchAdapter

    adapter = _VLMBatchAdapter.__new__(_VLMBatchAdapter)
    adapter.per_row_prompt_kwargs, adapter.cancel = True, None
    adapter.defaults = GenerationDefaults()
    adapter.processor = types.SimpleNamespace(tokenizer=_CharTokenizer())
    generator = _vlm_stub(True, events=[
        [(0, 1, None), (1, 1, None)],
        [(0, 2, "length"), (1, 1, None)],
        [(1, 2, "length")],
    ])(object(), object())
    events = list(adapter._drive(generator, [0, 1], [7, 9], {}, prompt_token_count = 5))
    finished = [index for index, event in enumerate(events) if event.result is not None]
    nine = [index for index, event in enumerate(events) if event.index == 9]

    assert {event.index for event in events} == {7, 9}
    assert events[finished[0]].index == 7
    assert any(index < finished[0] for index in nine)
    assert any(index > finished[0] for index in nine)
    assert [event.prompt_tokens for event in events] == [5] * len(events)
    # The exact running count: a fixed one would answer every stop the same way.
    assert [event.generated_tokens for event in events if event.index == 7] == [1, 2]
    assert [event.generated_tokens for event in events if event.index == 9] == [1, 2, 3]
    for index in (7, 9):
        row = [event for event in events if event.index == index]
        assert "".join(event.delta for event in row) == row[-1].result.text
        assert row[-1].generated_tokens == len(row[-1].result.token_ids)


class _OpenBatchGenerator:
    """A BatchGenerator reporting one token per row per step, while it has tokens left."""

    def __init__(self, *_args, **_kwargs):
        self.rows: dict[int, int] = {}
        self.removed: list[int] = []
        self.closed, self.steps, self._next_uid = False, 0, 100

    def insert(self, prompts, max_tokens=None, samplers=None, logits_processors=None):
        uids = []
        for index in range(len(prompts)):
            self._next_uid += 1
            self.rows[self._next_uid] = int((max_tokens or [1])[index])
            uids.append(self._next_uid)
        return uids

    def next_generated(self):
        self.steps += 1
        events = []
        for uid in list(self.rows):
            self.rows[uid] -= 1
            events.append(types.SimpleNamespace(
                uid=uid, token=1, logprobs=[-9.0, -0.1, -0.2, -0.3],
                finish_reason="length" if self.rows[uid] <= 0 else None,
            ))
            if self.rows[uid] <= 0:
                del self.rows[uid]
        return events

    def remove(self, uids):
        self.removed.extend(uids)
        for uid in uids:
            self.rows.pop(uid, None)

    def close(self):
        self.closed = True


def _open_stream(monkeypatch, generator_type=_OpenBatchGenerator, **defaults):
    """A BatchStream over a fake generator, with the generation lock stubbed out."""
    from unsloth_zoo.mlx import generate as module
    from unsloth_zoo.mlx.generate import BatchStream

    def adapter(model, tok, defaults_):
        built = object.__new__(_TextBatchAdapter)
        built.model, built.tokenizer, built.defaults = model, tok, defaults_
        built.batch_generator_type, built.sample_utils = generator_type, None
        built.make_sampler = lambda **_kwargs: None
        return built

    monkeypatch.setattr(module, "_TextBatchAdapter", adapter)
    monkeypatch.setattr(module, "_generation_cache_hygiene", contextlib.nullcontext)
    monkeypatch.setattr(module, "_snapshot_metal_limits", dict)
    monkeypatch.setattr(module, "_restore_metal_limits", lambda _snapshot: None)
    model = types.SimpleNamespace(training=False, eval=lambda: None)
    model.named_modules = lambda: [("", model)]
    tokenizer = _CharTokenizer()
    tokenizer.eos_token_ids = ()
    return BatchStream(model, tokenizer, defaults=GenerationDefaults(**defaults))


def test_a_batch_left_open_patches_the_cache_before_it_decodes(monkeypatch):
    """mlx-lm's ArraysCache.advance strands a Metal buffer per token until patched."""
    from unsloth_zoo.mlx import generate as engine

    installed = []
    monkeypatch.setattr(
        engine, "_install_arrays_cache_advance_fix", lambda: installed.append(True))
    stream = _open_stream(monkeypatch, max_tokens=2)
    try:
        assert installed == [True]
    finally:
        stream.close()


def test_rows_join_and_leave_a_batch_while_the_rest_of_it_runs_on(monkeypatch):
    """A later request joins the batch already running, and a cancelled row leaves it."""
    stream = _open_stream(monkeypatch, max_tokens=8)
    try:
        stream.add(GenerationRequest(prompt_token_ids=[9], max_tokens=8))
        assert [event.index for _ in range(2) for event in stream.step()] == [0, 0]
        assert stream.add(GenerationRequest(prompt_token_ids=[9, 9, 9], max_tokens=8)) == 1
        assert [event.index for _ in range(2) for event in stream.step()] == [0, 1, 0, 1]

        managed = stream.withdraw(1)
        assert managed is not None and managed.finish_reason == "stop"
        assert managed.prompt_token_count == 3
        assert managed.text and len(managed.token_ids) == len(managed.logprobs) > 0
        assert stream._session.generator.removed == [102]
        assert [event.index for event in stream.step()] == [0]
        assert stream.rows_in_flight == 1
        assert stream.withdraw(1) is None and stream.withdraw(7) is None
        assert stream.cancel(1) is False, "and the older answer is the same answer"
    finally:
        stream.close()


class _OpenVLMGenerator:
    """mlx-vlm's open-batch surface: insert answers with uids, next reports."""

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.inserted, self.removed, self.events = [], [], []
        self.closed = False
        self._next_uid = 0
        self._prompt_batch = None
        self._generation_batch = types.SimpleNamespace(uids = [])
        self._unprocessed_sequences = []

    def insert(self, prompts, max_tokens, **kwargs):
        self.inserted.append((list(prompts), list(max_tokens), kwargs))
        uids = list(range(self._next_uid, self._next_uid + len(prompts)))
        self._next_uid += len(prompts)
        self._generation_batch.uids.extend(uids)
        return uids

    @property
    def has_work(self):
        return bool(self.events)

    def next(self):
        return [], self.events.pop(0) if self.events else []

    def remove(self, uids):
        self.removed.append(list(uids))

    def close(self):
        self.closed = True


def _vlm_adapter(generator_type = None):
    """The release surface a vision session actually uses, over fakes."""
    from unsloth_zoo.mlx.generate import _VLMBatchAdapter

    def keeps_nothing(input_ids, *a, **k):
        width = len(input_ids.tolist()[0])
        embeds = types.SimpleNamespace(shape = (1, width, 4), width = width)
        return types.SimpleNamespace(to_dict = lambda: {"inputs_embeds": embeds})

    adapter = types.SimpleNamespace(
        defaults = GenerationDefaults(max_tokens = 5),
        processor = types.SimpleNamespace(tokenizer = _CharTokenizer()),
        sample_utils = types.SimpleNamespace(),
        make_sampler = lambda **knobs: (lambda logprobs: logprobs),
        constructor_params = {"prefill_batch_size", "completion_batch_size",
                              "sampler", "compute_logprobs", "prefill_step_size"},
        per_row_prompt_kwargs = True,
        batches_observable = True,
        padded_prompt_kwargs = frozenset({"inputs_embeds", "position_ids"}),
        cancel = lambda generator, uid: generator.remove([uid]),
        model = types.SimpleNamespace(
            config = types.SimpleNamespace(image_token_index = None),
            language_model = object(),
            get_input_embeddings = keeps_nothing,
        ),
        prepare_inputs = lambda processor, **k: {
            "input_ids": types.SimpleNamespace(
                tolist = lambda: [[1] * len(k["prompts"][0])]),
        },
        generator_type = generator_type or _OpenVLMGenerator,
    )
    adapter.row_logits_processors, adapter.wired = True, []
    adapter._row_logits_processors = types.MethodType(
        _VLMBatchAdapter._row_logits_processors, adapter)
    adapter._wired_limit = lambda: _wired_ctx(adapter.wired)
    adapter._decode_image = lambda request: request
    adapter._add_special_tokens = lambda: True
    adapter._split_prompt_kwargs = lambda kwargs, n: [dict(kwargs)] * n
    adapter._admission_stalled = lambda generator: False
    adapter._stall_error = lambda: RuntimeError("stalled")
    return adapter


def _vlm_session(generator_type = None, **defaults):
    from unsloth_zoo.mlx.generate import _VLMBatchSession

    adapter = _vlm_adapter(generator_type)
    adapter.defaults = GenerationDefaults(max_tokens = 5, **defaults)
    adapter.constructor_params = adapter.constructor_params | set(defaults)
    return _VLMBatchSession(adapter)


def test_an_open_batch_builds_its_cache_with_the_quantization_it_was_given():
    with contextlib.closing(_vlm_session(kv_bits = 8, kv_group_size = 32)) as session:
        assert [session.generator.kwargs[n] for n in ("kv_bits", "kv_group_size")] == [8, 32]
    with contextlib.closing(_vlm_session(kv_bits = 4, kv_quant_scheme = "turboquant", quantized_kv_start = None)) as session:  # noqa: E501
        assert session.generator.kwargs["quantized_kv_start"] == 0  # from the first token, not mlx-vlm's 5000


def test_a_model_keeping_state_is_seen_to_keep_it_however_it_says_so():
    """A model may do the assignment in a helper, or spell it as something other than a name."""
    from unsloth_zoo.mlx.generate import _keeps_what_it_prepared

    class Submodule:
        def __call__(self, pixels): return pixels

    class Clean:
        def _measure(self, ids): return len(ids)
        def get_input_embeddings(self, ids): return self._measure(ids)

    class Deep:
        def _remember(self, ids): self.language_model._position_ids = ids
        def _prepare(self, ids): self._remember(ids)
        def get_input_embeddings(self, ids): self._prepare(ids)

    class Circular:
        def _left(self, ids): return self._right(ids)
        def _right(self, ids): return self._left(ids)
        def get_input_embeddings(self, ids): return self._left(ids)

    class Tower:
        def __init__(self): self.vision_tower = Submodule()
        def get_input_embeddings(self, ids, pixels): return self.vision_tower(pixels)

    class Opaque:
        get_input_embeddings = Submodule()

    class Subscript:
        def get_input_embeddings(self, ids): self.state["ids"] = ids

    class Unpacked:
        def get_input_embeddings(self, ids): self._ids, self._mask = ids, None

    keeping = [Deep, Opaque, Subscript, Unpacked]
    for model in (Clean, Deep, Circular, Tower, Opaque, Subscript, Unpacked):
        assert _keeps_what_it_prepared(model()) is (model in keeping), model.__name__


def test_a_vision_request_preparing_unlike_the_batch_s_others_cannot_join():
    """A key only some rows carry cannot be merged into a batch-wide keyword."""
    session = _vlm_session()
    plain = session.adapter.model.get_input_embeddings
    assert session.add(GenerationRequest(prompt = "abc")) == 0
    def with_positions(input_ids, *a, **k):
        out = plain(input_ids, *a, **k).to_dict()
        return types.SimpleNamespace(
            to_dict = lambda: {**out, "position_ids": "P"})
    session.adapter.model.get_input_embeddings = with_positions
    with pytest.raises(ValueError, match = "unlike the batch's other requests"):
        session.add(GenerationRequest(prompt = "de"))
    assert (session.rows_in_flight, len(session.generator.inserted)) == (1, 1)
    session.adapter.model.get_input_embeddings = plain
    assert session.add(GenerationRequest(prompt = "fg")) == 1


def test_a_processor_is_shown_its_own_row_s_history_from_its_own_offset():
    """Rows rarely share a prompt length, so each latches its own offset."""
    from unsloth_zoo.mlx.generate import _row_processors

    adapter = _vlm_adapter()
    shown = [[], []]
    rows = adapter._row_logits_processors([
        GenerationRequest(prompt = "p", logits_processors = [_pin(7, seen)])
        for seen in shown
    ])
    for processor, history in zip(rows, ([1, 2, 3], [1, 2, 3, 4, 5, 6, 7])):
        _shows(processor[0], history)
    assert shown == [[[3]], [[7]]]

    alone = []
    processor, = _row_processors(
        GenerationRequest(prompt = "p", logits_processors = [_pin(7, alone)])
    )
    # A prompt of five, then one sampled token, then a second.
    for history in ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]):
        _shows(processor, history)
    assert alone == [[5], [5, 6], [5, 6, 7]]


@pytest.mark.parametrize("carrying", (0, 1))
def test_a_text_row_carries_its_own_logits_processors_into_the_batch(monkeypatch, carrying):
    """The text path admits rows one at a time too, and each keeps its own list."""
    seen = []

    class _Recording(_OpenBatchGenerator):
        def insert(self, prompts, max_tokens = None, samplers = None,
                   logits_processors = None):
            seen.append(logits_processors)
            return super().insert(prompts, max_tokens, samplers, logits_processors)

    lists = [None, None]
    lists[carrying] = [_pin(7)]
    stream = _open_stream(monkeypatch, generator_type = _Recording, max_tokens = 4)
    try:
        for processors in lists:
            stream.add(GenerationRequest(
                prompt_token_ids = [9], max_tokens = 4, logits_processors = processors,
            ))
    finally:
        stream.close()
    assert seen[1 - carrying] == [[]]
    assert [_shows(processor, [4, 5, 6]) for processor in seen[carrying][0]] == [7]

def test_stream_batch_checks_the_defaults_type_before_reading_stop_strings():
    with pytest.raises(TypeError, match="must be GenerationDefaults"):
        stream_batch(object(), object(), [], defaults = {"stop_strings": ()})
    with pytest.raises(ValueError, match="cannot apply stop_strings"):
        stream_batch(object(), object(), [],
                     defaults = GenerationDefaults(stop_strings = ("X",)))


def test_kv_quant_controls_and_processor_rows_are_refused_before_they_reach_a_cache():
    assert [type(pytest.raises((TypeError, ValueError), GenerationDefaults, **kw).value) for kw in ({"kv_bits": True}, {"kv_bits": float("nan")}, {"kv_bits": float("inf")}, {"kv_group_size": True}, {"kv_group_size": -1}, {"kv_quant_scheme": 3}, {"quantized_kv_start": -1})] == [TypeError, ValueError, ValueError, TypeError, ValueError, TypeError, ValueError] and GenerationDefaults(kv_bits = 4.5, kv_group_size = 32, kv_quant_scheme = "turboquant", quantized_kv_start = 0).kv_bits == 4.5  # noqa: E501
    with contextlib.closing(_vlm_session()) as session:
        session.adapter.row_logits_processors = False
        with pytest.raises(ValueError, match = "cannot carry") as refusal:
            session.add(GenerationRequest(prompt = "p", logits_processors = [lambda tokens, logits: logits]))
        assert type(refusal.value).__name__ == "BatchRowRefused" and session.usable and session.add(GenerationRequest(prompt = "p")) == 0  # noqa: E501


def test_the_preflight_refuses_defaults_every_add_would():
    from unsloth_zoo.mlx.generate import stream_unavailable_reason
    assert all(m in stream_unavailable_reason(types.SimpleNamespace(**model), object(), defaults = GenerationDefaults(**kw)) for model, kw, m in (({}, {"kv_bits": 8}, "not forwarded"), ({"_is_vlm_model": True, "language_model": object()}, {"max_kv_size": 64}, "max_kv_size"), ({"_is_vlm_model": True, "language_model": object()}, {"prefill_batch_size": 4, "completion_batch_size": 2}, "must not exceed")))  # noqa: E501


def test_a_cache_with_model_specific_state_refuses_a_quantized_batch_before_it_starts():
    from unsloth_zoo.mlx.generate import stream_unavailable_reason

    _SideCache = type("_SideCache", (), {"to_batch": lambda self, left_padding: self})
    caches = [object(), _SideCache(), object()]
    model = types.SimpleNamespace(_is_vlm_model = True, language_model = types.SimpleNamespace(make_cache = lambda: list(caches)))  # noqa: E501
    gap = lambda **kw: stream_unavailable_reason(model, object(), defaults = GenerationDefaults(kv_bits = 8, **kw)) or ""  # noqa: E501,E731
    assert "cannot quantize across a batch" in gap()
    assert "kv_quant_scheme would be dropped" in gap(kv_quant_scheme = "bogus")  # refused before the cache is looked at
    caches[1], caches[2] = caches[2], caches[1]  # only in the last layer, which stays unquantized
    assert "cannot quantize" not in gap()


def test_a_prefill_batch_charges_each_row_its_own_prompt_and_not_the_padded_width():
    from unsloth_zoo.mlx.generate import _VLMBatchAdapter, _unpadded_lengths

    adapter = _VLMBatchAdapter.__new__(_VLMBatchAdapter)
    adapter.per_row_prompt_kwargs, adapter.defaults = True, GenerationDefaults()
    adapter.processor = types.SimpleNamespace(tokenizer = _CharTokenizer())
    generator = _vlm_stub(True, events = [[(0, 3, "stop"), (1, 3, "stop")]])(object(), object())
    mask = types.SimpleNamespace(sum = lambda axis: {-1: types.SimpleNamespace(tolist = lambda: [1.0, 11.0])}[axis])  # noqa: E501
    results = _results(adapter._drive(generator, [0, 1], [0, 1], {}, prompt_token_counts = _unpadded_lengths(mask)))  # noqa: E501
    assert [r.prompt_token_count for r in results] == [1, 11]

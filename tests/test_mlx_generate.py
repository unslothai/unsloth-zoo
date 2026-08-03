import concurrent.futures
import contextlib
import importlib.util
import inspect
import pathlib
import types
from dataclasses import make_dataclass
import pytest

# These are logic tests, but they cross into mlx.core and mlx_vlm at a few
# points, so off Apple Silicon they run against the torch-backed shim.
if importlib.util.find_spec("mlx") is None:
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()

from unsloth_zoo.mlx.generate import (  # noqa: E402
    GenerationDefaults, GenerationRequest, SamplingParams,
    _GENERATION_MODE_LOCK, _PendingResult, _StopStringScanner, _TextBatchAdapter,
    _eos_stop_tokens, _new_detokenizer, _probe_sampler_api, _probe_text_api,
    _generation_cache_hygiene, _restore_training_flags,
    _validate_text_requests, generate_batch, generation_mode,
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
    # Refused per backend, not by the container, so each reason is true.
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
    result = adapter.generate([GenerationRequest(prompt_token_ids=[9])])[0]
    assert result.token_ids == [1] and Generator.instances[-1].removed == [7]
    assert tuple(Generator.instances[-1].kwargs[name] for name in ("prefill_batch_size", "completion_batch_size", "max_kv_size")) == (2, 3, 16)
    assert Generator.instances[-1].closed is True
    Generator.eos = True
    eos = adapter.generate([GenerationRequest(prompt_token_ids=[9])])[0]
    assert eos.token_ids == [1] and eos.finish_reason == "stop"
    Generator.eos = False
    Generator.fail = True
    Generator.close_fail = True
    with pytest.warns(RuntimeWarning, match=r"close\(\) failed"):
        with pytest.raises(RuntimeError, match="event failure"):
            adapter.generate([GenerationRequest(prompt_token_ids=[9])])
    assert Generator.instances[-1].closed is True

def test_generation_mode_releases_lock_when_limit_restore_raises(monkeypatch):
    model = types.SimpleNamespace(training=True)
    model.named_modules = lambda: [("", model)]
    model.eval = lambda: setattr(model, "training", False)

    monkeypatch.setattr(
        "unsloth_zoo.mlx.generate._snapshot_metal_limits", lambda: {"memory": 1}
    )
    monkeypatch.setattr(
        "unsloth_zoo.mlx.generate._restore_metal_limits",
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
    # A vLLM prompt dict iterates to its keys, so it must be refused by type
    # rather than silently generating from the key names.
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
        # One that stopped being refused would reach generation and be dropped.
        *((GenerationRequest(prompt="a"), GenerationDefaults(**{name: value}),
           "not forwarded by this engine")
          for name, value in (("kv_bits", 4), ("kv_group_size", 64),
                              ("kv_quant_scheme", "affine"), ("quantized_kv_start", 100))))
    for request, defaults, message in cases:
        with pytest.raises(ValueError, match=message):
            _validate_vlm_requests([request], defaults)
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
    results = adapter._drive(generator, [0], {})
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
    results = adapter._drive(generator, [0, 1], {})
    # Distinct buffers: a shared one concatenates both sequences into the first.
    assert [(r.token_ids, r.text) for r in results] == [
        ([1], "hello "), ([4], "tailSTOPsuffix")] and results[0].logprobs == [-0.5]


def test_vlm_chunking_respects_release_capabilities():
    # Older releases run one chunk per generator to avoid row-zero reuse.
    from unsloth_zoo.mlx.generate import _VLMBatchAdapter
    adapter = _VLMBatchAdapter.__new__(_VLMBatchAdapter)
    adapter.defaults = GenerationDefaults(prefill_batch_size=2, completion_batch_size=4)
    chunks = []
    adapter._group_key = lambda request: "same"
    adapter._run_chunk = lambda requests, indices: (
        chunks.append(list(indices)) or [object() for _ in indices])
    requests = [GenerationRequest(prompt="p") for _ in range(5)]
    for per_row, expected in ((True, [[0, 1, 2, 3, 4]]), (False, [[0, 1], [2, 3], [4]])):
        chunks.clear(); adapter.per_row_prompt_kwargs = per_row  # noqa: E702
        adapter.generate(requests)
        assert chunks == expected


def test_vlm_run_chunk_builds_the_generator_after_embeddings():
    # The policy needs real inputs, so the generator is built after embeddings.
    from unsloth_zoo.mlx.generate import _VLMBatchAdapter
    order, seen = [], {}
    ids, embeds = (types.SimpleNamespace(tolist=lambda: [[1], [2]]),
                   types.SimpleNamespace(to_dict=lambda: {"inputs_embeds": "E", "position_ids": None}))
    adapter = _VLMBatchAdapter.__new__(_VLMBatchAdapter)
    adapter.defaults, adapter.per_row_prompt_kwargs = GenerationDefaults(), True
    adapter.processor = types.SimpleNamespace(tokenizer=_CharTokenizer())
    adapter.constructor_params = {"prefill_batch_size", "completion_batch_size", "sampler", "stop_tokens"}  # noqa: E501
    adapter.prepare_inputs = lambda *a, **k: {"input_ids": ids, "extra": "D", "position_ids": "P"}  # noqa: E501
    adapter.make_sampler = lambda **k: object()
    adapter._add_special_tokens, adapter._drive = (lambda: True), (lambda *a: ["ok"] * 2)
    adapter._split_prompt_kwargs = lambda kw, n: (seen.update(kwargs=kw), [{}] * n)[1]
    # Recording on ENTRY proves embeddings run inside the wired-limit context.
    adapter._wired_limit = lambda: _entry_ctx(order)  # records on ENTRY
    adapter._chunked_prefill_kwargs = lambda **kw: (
        order.append("policy"), seen.update(policy=kw), {})[2]
    adapter.model = types.SimpleNamespace(
        config=types.SimpleNamespace(image_token_index=None), language_model=object(),
        get_input_embeddings=lambda *a, **k: (order.append("embed"), embeds)[1])
    adapter.generator_type = lambda *a, **k: (order.append("construct"),
        seen.update(ctor=k), _vlm_stub(True, events=[[(0, 1, "stop")]])(*a[:2]))[2]
    assert adapter._run_chunk([GenerationRequest(prompt="p")] * 2, [0, 1]) == ["ok", "ok"]
    assert order == ["wired", "embed", "policy", "construct"]
    assert seen["policy"]["prefill_kwargs"]["inputs_embeds"] == "E"
    assert seen["kwargs"]["extra"] == "D"
    # A None embedding field must not overwrite a valid preprocessing value.
    assert seen["kwargs"]["position_ids"] == "P"
    # Per-row keeps configured sizes; EOS stays with the processor.
    assert (seen["ctor"]["prefill_batch_size"], seen["ctor"]["completion_batch_size"]) == (8, 32)
    assert "stop_tokens" not in seen["ctor"]


@pytest.mark.parametrize("nested", (True, False))
def test_vlm_probe_accepts_both_event_class_locations(nested):
    # Pre-0.5 nests Response with a logprob vector; 0.5+ moves it to
    # GenerationBatch with a scalar token_logprob.
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
    # One fetch per request; the decoded object reaches grouping and
    # prepare_inputs; raw bytes are rejected.
    from unsloth_zoo.mlx.generate import _VLMBatchAdapter
    loads, seen = [], {}
    decoded = types.SimpleNamespace(size=(8, 8))
    adapter = _VLMBatchAdapter.__new__(_VLMBatchAdapter)
    adapter.defaults, adapter.per_row_prompt_kwargs = GenerationDefaults(), True
    adapter.processor = types.SimpleNamespace(tokenizer=_CharTokenizer())
    adapter.process_image = lambda src, shape, proc: loads.append(src) or decoded
    adapter._run_chunk = lambda requests, indices: (
        seen.update(image=requests[indices[0]].image), [object()])[1]
    adapter.generate([GenerationRequest(prompt="p", image="/tmp/one-use.png")])
    assert loads == ["/tmp/one-use.png"] and seen["image"] is decoded
    with pytest.raises(ValueError, match="raw bytes"):
        adapter.generate([GenerationRequest(prompt="p", image=b"\x89PNG")])


def test_vlm_adapter_initializes_against_newer_module_layout(monkeypatch):
    # Newer layout: the package re-exports BatchGenerator while helpers and the
    # event class live on the defining module. Delegation via __getattr__ is
    # omitted so this proves the defining-module binding alone.
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
    bare.BatchGenerator = Generator  # re-export only; no delegation here
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", bare)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate.ar", ar)
    adapter = engine._VLMBatchAdapter(object(), _CharTokenizer(), GenerationDefaults())
    # Binds the DEFINING module although the re-exporting one matched first.
    assert adapter.batch_module is ar and adapter.per_row_prompt_kwargs
    assert adapter._split_prompt_kwargs({}, 3) == [{}, {}, {}]  # upstream splitter chosen
    assert adapter._chunked_prefill_kwargs(input_ids=None, prefill_kwargs={}) == {}
    assert policy_calls  # the policy helper was consulted, not bypassed
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
    # The candidate itself being absent still degrades to None.
    assert _resolve_module_attr(("mlx_vlm.generate.ar",), "BatchGenerator") is None


def test_vlm_drive_raises_on_true_stall_and_survives_long_prefill():
    # Impossible admission raises; a multi-poll prefill must still complete.
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
                    # Prefill done: the sequence is promoted into decoding.
                    self._prompt_batch, self._generation_batch = None, [1]
                return ([], [])
            return super().next(**kwargs)
    assert adapter._drive(LongPrefill(object(), object()), [0], {})[0].finish_reason == "stop"
    class Stalled(base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._prompt_batch, self._generation_batch = None, []
            self._unprocessed_sequences = [1]
        has_work = True
        def next(self, **kwargs):
            return ([], [])
    with pytest.raises(RuntimeError, match="admitted no prompt"):
        adapter._drive(Stalled(object(), object()), [0], {})


def test_vlm_requests_group_by_sampling_params():
    # Differing sampling must not share one constructor-level sampler.
    from unsloth_zoo.mlx.generate import _VLMBatchAdapter
    adapter = _VLMBatchAdapter.__new__(_VLMBatchAdapter)
    adapter.defaults, adapter.per_row_prompt_kwargs = GenerationDefaults(), True
    adapter.process_image = None
    chunks = []
    adapter._run_chunk = lambda requests, indices: (
        chunks.append(list(indices)) or [object() for _ in indices])
    hot = SamplingParams(temperature=0.9)
    adapter.processor = types.SimpleNamespace(tokenizer=_CharTokenizer())
    adapter.generate([GenerationRequest(prompt="a"), GenerationRequest(prompt="b", sampling=hot),
                      GenerationRequest(prompt="c")])
    assert sorted(chunks) == [[0, 2], [1]]
    # Preprocessing stacks a group without padding, so differing prompt lengths
    # must not share a chunk; upstream raises when they do.
    chunks.clear()
    adapter.generate([GenerationRequest(prompt="a"), GenerationRequest(prompt="bb"),
                      GenerationRequest(prompt="c")])
    assert sorted(chunks) == [[0, 2], [1]]


@pytest.mark.parametrize("plural", (True, False))
def test_vlm_stop_strings_trim_and_cancel_in_the_release_form(plural):
    # Both signature forms, collection and single uid, must drive correctly.
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
    result = adapter._drive(Generator(object(), object()), [0], {})[0]
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
        [(0, 1, None), (1, 1, "length")]])  # uid 0's late token must be ignored
    adapter = _VLMBatchAdapter.__new__(_VLMBatchAdapter)
    adapter.defaults = GenerationDefaults(stop_strings=("STOP",))
    adapter.per_row_prompt_kwargs = False
    adapter.processor = types.SimpleNamespace(tokenizer=_CharTokenizer())
    adapter.cancel = _resolve_cancel(Generator)
    assert adapter.cancel is None  # no cancellation on this release
    stopped, sibling = adapter._drive(Generator(object(), object()), [0, 1], {})
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
            if acquired:  # never leak global lock state on failure
                _GENERATION_MODE_LOCK.release()
            assert acquired is False  # serialized across threads
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
    # A shortening decode must not push the offset back and re-emit that tail.
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

def _audio_events(*specs):
    """Sequential events; `finish` omitted models releases carrying no reason.

    Each event carries a distinct logprob, so committing a successor's value
    against its predecessor's token shifts alignment visibly.
    """
    made = []
    for position, (token, finish) in enumerate(specs):
        fields = {"token": token, "logprobs": {token: -0.5 - position}}
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
    adapter._wired_limit = contextlib.nullcontext
    adapter.stream_calls = []
    def stream(*args, **kwargs):
        adapter.stream_calls.append((args, kwargs))
        return iter(events)
    adapter.stream_module = types.SimpleNamespace(stream_generate=stream)
    return adapter


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
    # A reported reason is believed, not re-inferred, in both directions.
    (((1, None), (2, "stop")), None, ([1], "stop")),
    (((1, None), (3, "length")), _CriteriaTokenizer(), ([1], "length")),
    # Nothing generated at all: newer releases emit one event with no token.
    (((None, "length"),), None, ([], "length")),
])
def test_audio_fallback_keeps_the_batched_result_contract(specs, tokenizer, expected):
    result = _audio_adapter(_audio_events(*specs), tokenizer)._generate_sequentially(
        GenerationRequest(prompt="p", audio="a.wav"))
    assert (result.token_ids, result.finish_reason) == expected
    # Logprobs must belong to the tokens they sit beside, not to their successors.
    assert result.logprobs == [-0.5 - n for n in range(len(result.token_ids))]
    assert result.text == "".join(_CharTokenizer.pieces[t] for t in result.token_ids)


def test_audio_terminal_event_flushes_text_held_back_during_streaming():
    # Cleanup withholds a trailing space until finalize; losing it truncates.
    tokenizer = _TableTokenizer({(1,): "a ", (1, 2): "a b "}, clean=True)
    result = _audio_adapter(_audio_events((1, None), (2, None), (2, None)),
                            tokenizer)._generate_sequentially(
        GenerationRequest(prompt="p", audio="a.wav"))
    assert (result.token_ids, result.text) == ([1, 2], "a b ")


def test_audio_fallback_forwards_the_request_to_the_stream():
    # A dropped audio payload or ignored control shows up in the stream call.
    adapter = _audio_adapter(_audio_events((1, None), (2, "length")))
    adapter._generate_sequentially(GenerationRequest(
        prompt="p", audio="a.wav", max_tokens=9,
        sampling=SamplingParams(temperature=0.25, top_k=7)))
    args, kwargs = adapter.stream_calls[0]
    assert args[2] == "p"  # the rendered prompt, after model and processor
    assert (kwargs["audio"], kwargs["max_tokens"]) == ("a.wav", 9)
    assert kwargs["sampler"] is adapter.sampler
    # Per-request sampling, not the batch-wide default.
    assert (adapter.sampler_kwargs["temp"], adapter.sampler_kwargs["top_k"]) == (0.25, 7)


def test_audio_fallback_honours_stop_strings():
    # Audio reaches the same scanner: "<ST" + "OP>" completes the stop string.
    result = _audio_adapter(_audio_events((2, None), (3, None), (3, "length")),
                            stops=("STOP",))._generate_sequentially(
        GenerationRequest(prompt="p", audio="a.wav"))
    assert (result.token_ids, result.finish_reason, result.stop_match) == (
        [], "stop_string", "STOP")


def test_audio_requests_reach_the_fallback_through_the_public_entry_point(monkeypatch):
    # The tests above drive the adapter directly and would stay green if the
    # old blanket audio rejection came back.
    from unsloth_zoo.mlx import generate as engine
    seen = []
    monkeypatch.setattr(engine._VLMBatchAdapter, "__init__",
                        lambda self, *a: setattr(self, "per_row_prompt_kwargs", True))
    monkeypatch.setattr(engine._VLMBatchAdapter, "_decode_image", lambda self, r: r)
    monkeypatch.setattr(engine._VLMBatchAdapter, "_generate_sequentially",
                        lambda self, request: seen.append(request.audio) or "audio")
    model = types.SimpleNamespace(_is_vlm_model=True, training=False, eval=lambda: None)
    model.named_modules = lambda: [("", model)]
    with pytest.warns(RuntimeWarning, match="decode one at a time"):
        results = engine.generate_batch(
            model, object(), [GenerationRequest(prompt="p", audio="a.wav")])
    assert (results, seen) == (["audio"], ["a.wav"])
    # The at-most-one-media invariant has to survive audio becoming supported.
    with pytest.raises(ValueError, match="both an image and audio"):
        engine.generate_batch(model, object(),
                              [GenerationRequest(prompt="p", image=object(), audio="a.wav")])


def test_audio_requests_decode_alone_while_the_rest_of_the_batch_still_batches():
    # Audio has no batched path, but must not drag the others out of theirs.
    from unsloth_zoo.mlx.generate import _VLMBatchAdapter
    adapter = _audio_adapter(_audio_events((1, None), (2, "length")))
    batched = []
    adapter._decode_image = lambda request: request
    adapter._group_key = lambda request: "g"
    adapter.per_row_prompt_kwargs = True
    adapter._run_chunk = lambda chunk, indices: (
        batched.append(list(indices)), ["batched"] * len(chunk))[1]
    requests = [GenerationRequest(prompt="a"), GenerationRequest(prompt="b", audio="a.wav"),
                GenerationRequest(prompt="c")]
    with pytest.warns(RuntimeWarning, match="decode one at a time"):
        results = _VLMBatchAdapter.generate(adapter, requests)
    assert batched == [[0, 2]]  # the audio row never reached the batched path
    assert [results[0], results[2]] == ["batched", "batched"]
    assert results[1].token_ids == [1]


def test_vision_generation_resolves_the_saved_processor(monkeypatch):
    # A text-only multimodal load publishes its inner tokenizer, so the typed
    # core resolves the saved processor for both it and fast_generate.
    from unsloth_zoo.mlx.generate import generate_batch as engine_generate_batch
    from unsloth_zoo.mlx import generate as engine
    seen = {}
    class _Adapter:
        def __init__(self, model, tok, defaults): seen["tok"] = tok
        def generate(self, requests): return ["ok"]
    monkeypatch.setattr(engine, "_VLMBatchAdapter", _Adapter)
    monkeypatch.setattr(engine, "generation_mode", lambda model: contextlib.nullcontext())
    processor = object()
    model = types.SimpleNamespace(_is_vlm_model=True, _processor=processor)
    assert engine_generate_batch(model, object(), [GenerationRequest(prompt="a")]) == ["ok"]
    assert seen["tok"] is processor

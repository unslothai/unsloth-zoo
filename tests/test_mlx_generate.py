import asyncio
import concurrent.futures
import inspect
import threading
import types
from dataclasses import make_dataclass
import pytest
from unsloth_zoo.mlx.generate import (
    GenerationDefaults, GenerationRequest,
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
    scanner = _StopStringScanner(("STOP",))
    assert scanner.feed("abcde") is None
    assert scanner.feed("abcdeSTOP") == (5, "STOP")
    canonical = _TableTokenizer({(1,): " hello", (2,): "\ufffd", (2, 3): "¡STOP"})
    leading = _PendingResult(_new_detokenizer(canonical), _StopStringScanner((" hello",)))
    assert leading.append(canonical, 1, -0.1) is True
    assert leading.result(canonical).token_ids == []
    byte_stop = _PendingResult(_new_detokenizer(canonical), _StopStringScanner(("STOP",)))
    assert byte_stop.append(canonical, 2, -0.2) is False
    assert byte_stop.append(canonical, 3, -0.3) is True
    assert (byte_stop.result(canonical).token_ids, byte_stop.text) == ([], "")
    unicode_text = _PendingResult(_new_detokenizer(canonical), _StopStringScanner(()))
    unicode_text.append(canonical, 2, -0.2)
    unicode_text.append(canonical, 3, -0.3)
    unicode_text.finish(canonical, "length")
    assert unicode_text.text == "¡STOP"

    rewriting = _TableTokenizer({(1,): "abcST", (1, 2): "abcXYOP"})
    rewritten = _PendingResult(_new_detokenizer(rewriting), _StopStringScanner(("STOP",)))
    assert rewritten.append(rewriting, 1, -0.1) is False
    assert rewritten.append(rewriting, 2, -0.2) is True
    assert rewritten.result(rewriting).text == ""

    shrinking = _TableTokenizer({(1,): "abcde", (1, 2): "x "}, clean=True)
    terminal = _PendingResult(_new_detokenizer(shrinking), _StopStringScanner((" ",)))
    terminal.append(shrinking, 1, -0.1)
    terminal.add_terminal(2, -0.2)
    terminal.finish(shrinking, "length")
    assert (terminal.text, terminal.finish_reason) == ("abcde", "length")

@pytest.mark.parametrize(
    "generation_request,error",
    [
        (GenerationRequest(), "exactly one"),
        (GenerationRequest(prompt="x", prompt_token_ids=[1]), "exactly one"),
        (GenerationRequest(prompt_token_ids=[]), "must not be empty"),
        (GenerationRequest(prompt="x", image=object()), "text prompts only"),
        (GenerationRequest(prompt="x", max_tokens=0), "must be positive"),
    ],
)
def test_request_invariants(generation_request, error):
    with pytest.raises((TypeError, ValueError), match=error):
        _validate_text_requests([generation_request], GenerationDefaults())

def test_one_shot_prompt_ids_are_normalized_during_validation():
    validated = _validate_text_requests(
        [GenerationRequest(prompt_token_ids=iter((11, 12)))], GenerationDefaults())
    assert validated[0].prompt_token_ids == (11, 12)

def test_text_api_probe_accepts_pinned_shape_and_names_pin_on_mismatch():
    Response = make_dataclass(
        "Response",
        [("uid", int), ("token", int), ("logprobs", object), ("finish_reason", str | None)],
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
    with pytest.raises(RuntimeError, match=r"remove missing.*mlx-lm==0\.31\.2"):
        _probe_text_api(module)
    with pytest.raises(RuntimeError, match=r"make_sampler missing min_p"):
        _probe_sampler_api(type("Sample", (), {"make_sampler": lambda temp: None}))
    assert _eos_stop_tokens(types.SimpleNamespace(eos_token_ids={2, None})) == [[2]]

def test_falsey_defaults_are_not_silently_replaced(monkeypatch):
    with pytest.raises(TypeError, match="defaults"):
        generate_batch(object(), None, [], defaults={})
    for name in ("kv_bits", "kv_group_size"):
        with pytest.raises(ValueError, match=rf"{name} are not supported by mlx-lm==0\.31\.2"):
            GenerationDefaults(**{name: 4})
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
    monkeypatch.setattr(
        "unsloth_zoo.mlx.generate._restore_metal_limits", lambda _state: None
    )
    async def overlap():
        async def enter_mode():
            with generation_mode(model):
                pass
        with generation_mode(model):
            with pytest.raises(RuntimeError, match="async tasks"):
                await asyncio.create_task(enter_mode())
    asyncio.run(overlap())
    assert model.training is True
    entered, release = threading.Event(), threading.Event()
    def hold_mode():
        with generation_mode(model):
            entered.set()
            release.wait(1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        holder = executor.submit(hold_mode)
        assert entered.wait(1)
        blocked = executor.submit(_GENERATION_MODE_LOCK.acquire, True, 0.05)
        assert blocked.result() is False
        release.set()
        holder.result()
    assert model.training is True

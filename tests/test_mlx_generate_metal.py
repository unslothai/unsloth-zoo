import types

import pytest
try:
    import mlx.core as mx
    import mlx.nn as nn
    _METAL = mx.metal.is_available()
except Exception:  # module-level nn.Module subclasses below need mlx to exist
    pytest.skip("requires mlx", allow_module_level=True)
metal_only = pytest.mark.skipif(not _METAL, reason="requires Apple Silicon Metal")
# Importable is not real: a sibling module installs the mlx simulation process-wide while it is
# collected, so these can import mlx against the shim (see tests/test_mlx_attention_metal.py).
from mlx_simulation import mlx_is_simulated  # noqa: E402

_HAS_REAL_MLX = not mlx_is_simulated()
real_mlx_only = pytest.mark.skipif(
    not _HAS_REAL_MLX, reason="needs real mlx.nn; the simulation has no set_dtype or Sequential"
)


def _nax_available():
    if not _METAL:
        return False
    from unsloth_zoo.mlx import nax
    return nax.nax_available()


# The NAX kernels need MetalPerformancePrimitives tensor ops: macOS 15 cannot build them and
# paravirtual or pre-M5 GPUs cannot load them, so they run only where the product would route.
nax_only = pytest.mark.skipif(not _nax_available(), reason="requires an Apple GPU with neural accelerators")
MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"
VLM_MODEL = "mlx-community/FastVLM-0.5B-bf16"

class _CompileBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear, self.dropout, self.seen_states = nn.Linear(8, 8), nn.Dropout(0.2), []
    def __call__(self, hidden):
        self.seen_states.append((self.training, hasattr(type(self), "_orig_call")))
        return self.dropout(nn.relu(self.linear(hidden)))

class _CompileLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed, self.layers, self.proj = (
            nn.Embedding(16, 8), [_CompileBlock()], nn.Linear(8, 16, bias=False))
    def make_cache(self):
        return []
    def __call__(self, tokens, cache=None):
        return self.proj(self.layers[0](self.embed(tokens)))

def _current_limit(name, expected):
    setter = getattr(mx, f"set_{name}_limit")
    current = setter(expected)
    setter(current)
    return current

@metal_only
def test_generation_mode_restores_flags_and_limits_when_nested_or_raised():
    from unsloth_zoo.mlx.generate import generation_mode
    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.2))
    model.train()
    model.modules()[-1]._set_training_mode(False)
    original_training = [module.training for module in model.modules()]
    recommended = int(mx.device_info()["max_recommended_working_set_size"])
    targets = {
        "memory": int(recommended * 0.75),
        "cache": int(recommended * 0.10),
        "wired": int(recommended * 0.50),
    }
    previous = {
        name: getattr(mx, f"set_{name}_limit")(value)
        for name, value in targets.items()
    }
    try:
        with pytest.raises(RuntimeError, match="injected"):
            with generation_mode(model):
                assert all(not module.training for module in model.modules())
                inner = nn.Linear(4, 4)
                inner.train()
                with generation_mode(inner):
                    assert inner.training is False
                assert inner.training is True
                changed = {
                    "memory": int(recommended * 0.65),
                    "cache": int(recommended * 0.05),
                    "wired": int(recommended * 0.40),
                }
                for name, value in changed.items():
                    getattr(mx, f"set_{name}_limit")(value)
                    assert _current_limit(name, value) == value
                raise RuntimeError("injected")
        assert [module.training for module in model.modules()] == original_training
        for name, target in targets.items():
            assert _current_limit(name, target) == target
    finally:
        for name, value in previous.items():
            getattr(mx, f"set_{name}_limit")(value)

@metal_only
def test_batched_greedy_matches_sequential_and_preserves_sampled_ids():
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    import unsloth_zoo.mlx.generate as generate_module
    from unsloth_zoo.mlx.generate import (
        GenerationDefaults,
        GenerationRequest,
        generate_batch,
    )
    model, tokenizer = load(MODEL)
    generate_module._ARRAYS_CACHE_ADVANCE_RESOLVED = False
    requests = [
        GenerationRequest(prompt="The capital of France is", max_tokens=8),
        GenerationRequest(prompt="Two plus two equals", max_tokens=8),
    ]
    defaults = GenerationDefaults()
    order = []
    real_install = generate_module._install_arrays_cache_advance_fix
    real_adapter_stream = generate_module._TextBatchAdapter.stream

    def record_install():
        order.append("install")
        return real_install()

    def record_stream(self, requests):
        order.append("stream")
        return real_adapter_stream(self, requests)

    generate_module._install_arrays_cache_advance_fix = record_install
    generate_module._TextBatchAdapter.stream = record_stream
    try:
        batched = generate_batch(model, tokenizer, requests, defaults=defaults)
    finally:
        generate_module._install_arrays_cache_advance_fix = real_install
        generate_module._TextBatchAdapter.stream = real_adapter_stream
    from mlx_lm.models.cache import ArraysCache
    assert order == ["install", "stream"]
    assert generate_module._ARRAYS_CACHE_ADVANCE_RESOLVED
    assert isinstance(ArraysCache.left_padding, property)
    sequential = []
    for request in requests:
        prompt_ids = tokenizer.encode(request.prompt, add_special_tokens=False)
        events = list(stream_generate(
            model,
            tokenizer,
            prompt_ids,
            max_tokens=request.max_tokens,
            sampler=make_sampler(temp=0.0),
        ))
        token_ids = [
            int(event.token) for event in events
            if event.finish_reason != "stop"
        ]
        logprobs = [float(event.logprobs[event.token].item()) for event in events
                    if event.finish_reason != "stop"]
        text = "".join(event.text for event in events)
        sequential.append((token_ids, logprobs, events[-1].finish_reason, text))
    for result, (token_ids, logprobs, reason, text) in zip(batched, sequential):
        assert result.token_ids == token_ids
        assert result.logprobs == pytest.approx(logprobs, abs=0.02)
        assert result.finish_reason == reason
        assert result.text == text
    from unsloth_zoo.mlx.loader import _patch_mlx_saving
    _patch_mlx_saving(model, tokenizer)
    smoke = model.fast_generate([request.prompt for request in requests], max_tokens=2)
    assert len(smoke) == 2 and all(result.token_ids for result in smoke)

@metal_only
def test_compiled_training_state_survives_generation():
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from unsloth_zoo.mlx.generate import (
        GenerationRequest, SamplingParams, generate_batch,
    )
    from unsloth_zoo.mlx.utils import (
        apply_gradient_checkpointing, remove_gradient_checkpointing,
    )
    model, optimizer = _CompileLM(), optim.SGD(learning_rate=1e-2)
    optimizer.init(model.trainable_parameters())
    loss_and_grad = nn.value_and_grad(
        model, lambda current, x, y: nn.losses.cross_entropy(
            current(x), y, reduction="mean"),
    )
    def step(x, y):
        loss, gradients = loss_and_grad(model, x, y)
        optimizer.update(model, gradients)
        return loss
    state = [model.state, optimizer.state, mx.random.state]
    compiled_step = mx.compile(step, inputs=state, outputs=state)
    x, y = mx.array([[1, 2, 3]]), mx.array([[2, 3, 4]])
    tokenizer = type("Tok", (), {"eos_token_ids": [0], "decode": lambda _, ids: str(ids)})()
    apply_gradient_checkpointing(model)
    try:
        mx.eval(state, before := compiled_step(x, y))
        def snapshot():
            return [(name, value.tolist()) for name, value in tree_flatten(model.parameters())]
        state_after_train = snapshot()
        model.train()
        model.layers[0].seen_states.clear()
        def sample(seed):
            mx.random.seed(seed)
            return generate_batch(model, tokenizer, [GenerationRequest(
                prompt_token_ids=[1, 2], max_tokens=3,
                sampling=SamplingParams(temperature=0.7))])[0].token_ids
        # Sampling consumes the global RNG stream, so the same seed reproduces.
        result, repeat = [types.SimpleNamespace(token_ids=sample(11))], sample(11)
        assert result[0].token_ids and repeat == result[0].token_ids
        assert snapshot() == state_after_train
        assert (False, True) in model.layers[0].seen_states
        assert hasattr(_CompileBlock, "_orig_call")
        assert all(module.training for module in model.modules())
        mx.eval(state, after := compiled_step(x, y))
        assert snapshot() != state_after_train
        assert optimizer.state["step"].item() == 2
        assert all(mx.isfinite(value).item() for value in (before, after))
    finally:
        remove_gradient_checkpointing(model)


@metal_only
def test_vlm_batched_generation_is_ordered_and_aligned():
    from PIL import Image
    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template
    from unsloth_zoo.mlx.generate import (
        GenerationDefaults,
        GenerationRequest,
        generate_batch,
    )
    model, processor = load(VLM_MODEL)
    model._is_vlm_model = True
    # One prompt, several images: many processors reject unequal prompt lengths,
    # so batching is exercised through the images.
    prompt = apply_chat_template(processor, model.config, "Name the colour.", num_images=1)
    requests = [
        GenerationRequest(prompt=prompt, image=Image.new("RGB", size, colour), max_tokens=4)
        for colour, size in (("red", (64, 64)), ("blue", (64, 64)), ("green", (96, 96)))
    ]
    defaults = GenerationDefaults(max_tokens=4)
    results = generate_batch(model, processor, requests, defaults=defaults)
    assert len(results) == 3
    for result in results:
        assert result.token_ids
        assert result.logprobs is None or len(result.logprobs) == len(result.token_ids)
        assert result.finish_reason in ("stop", "length")
    # Against upstream's own sequential decoding, not just against ourselves:
    # this is what proves the bypassed preprocessing agrees with mlx-vlm.
    from mlx_vlm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    events = [event for event in stream_generate(
        model, processor, requests[0].prompt, image=[requests[0].image],
        max_tokens=4, sampler=make_sampler(temp=0.0))]
    tail = events[-1]
    body = events[:-1]
    assert results[0].token_ids == [int(event.token) for event in body]
    # bf16 logprobs near 19 sit on a 0.125 grid; M1 + mlx-vlm 0.7.x lands one step apart.
    assert results[0].logprobs == pytest.approx(
        [float(event.logprobs[event.token].item()) for event in body], abs=0.125)
    assert results[0].text == "".join(event.text for event in events)
    assert tail is not None
    assert results[0].finish_reason == ("stop" if len(body) < 4 else "length")
    # Chunking must not pair a prompt with an earlier chunk's embeddings.
    chunked = generate_batch(model, processor, requests, defaults=GenerationDefaults(
        max_tokens=4, prefill_batch_size=1, completion_batch_size=1))
    assert [item.token_ids for item in chunked] == [item.token_ids for item in results]
    # Concurrent sequences must not share a detokenizer buffer, so text must
    # match too, not just ids.
    assert [item.text for item in chunked] == [item.text for item in results]
    assert all(r.text == "" or not r.text.startswith(results[0].text + results[1].text) for r in results)  # noqa: E501


def _own_copy(value):
    if isinstance(value, mx.array):
        return value + 0
    if isinstance(value, (list, tuple)):
        items = [_own_copy(item) for item in value]
        return type(value)(*items) if hasattr(value, "_fields") else type(value)(items)
    if hasattr(value, "__dict__") and hasattr(type(value), "state"):
        duplicate = type(value).__new__(type(value))
        duplicate.__dict__.update({k: _own_copy(v) for k, v in vars(value).items()})
        return duplicate
    return value


class _RowCacheState:
    def __init__(self, cache, lengths=None):
        self.cache, self.lengths, self.kept = cache, lengths, {}

    def open(self, token_ids):
        return self.cache, self.lengths or range(1, len(token_ids) + 1)

    def checkpoint(self, token_count, cache):
        self.kept[token_count] = _own_copy(cache)


class _SharedPrefixState(_RowCacheState):
    """Resumes the longest prefix any row banked into ``banked``, as a caller's store does."""

    def __init__(self, banked, make_cache):
        super().__init__(None)
        self.banked, self.make_cache = banked, make_cache

    def open(self, token_ids):
        self.ids = tuple(token_ids)
        hits = [ids for ids in self.banked if len(ids) < len(token_ids) and self.ids[:len(ids)] == ids]
        best = max(hits, key=len, default=())
        cache = _own_copy(self.banked[best]) if best else self.make_cache()
        return cache, range(len(best) + 1, len(token_ids))

    def checkpoint(self, token_count, cache):
        super().checkpoint(token_count, cache)
        self.banked[self.ids[:token_count]] = self.kept[token_count]


@metal_only
def test_vlm_stream_rows_resume_from_their_own_cache_bitwise():
    from mlx_vlm import load
    from mlx_vlm.models.cache import make_prompt_cache
    from mlx_vlm.prompt_utils import apply_chat_template
    import mlx_vlm
    from mlx.utils import tree_flatten
    from packaging.version import Version
    from PIL import Image
    from unsloth_zoo.mlx.generate import (
        BatchRowRefused, BatchStream, GenerationDefaults, GenerationRequest,
        row_prompt_cache_unavailable_reason,
    )
    model, processor = load(VLM_MODEL)
    model._is_vlm_model = True
    assert row_prompt_cache_unavailable_reason() is None
    rules = " ".join(f"Rule {i}: answer tersely." for i in range(420))
    prompt = apply_chat_template(processor, model.config, f"Name a colour. {rules}", num_images=0)
    state = _RowCacheState
    lm = model.language_model

    def run(*requests):
        defaults = GenerationDefaults(max_tokens=12, prefill_batch_size=1, completion_batch_size=4)
        with BatchStream(model, processor, defaults=defaults) as stream:
            rows = [stream.add(request) for request in requests]
            results = {}
            while len(results) < len(rows):
                results.update((e.index, e.result) for e in stream.step() if e.result is not None)
        return [results[row] for row in rows]

    cold_state = state(make_prompt_cache(lm))
    (cold,) = run(GenerationRequest(prompt=prompt, prompt_cache_state=cold_state))
    length, kept = cold.prompt_token_count, cold_state.kept
    # Checkpoints land only where a prefill chunk ends: the 2048 grid, and from mlx-vlm 0.7.0,
    # which chunks a tail shorter than a step, the held-back last token.
    held_back = [length - 1] if Version(mlx_vlm.__version__) >= Version("0.7.0") else []
    assert sorted(kept) == [*range(2048, length - 1, 2048), *held_back]
    # Alone, as the cold row decoded: batched decode depends on what decodes beside a row.
    for prefix in sorted(kept):
        warm_state = state(_own_copy(kept[prefix]))
        (warm,) = run(GenerationRequest(prompt=prompt, prompt_cache_state=warm_state))
        assert (warm.token_ids, warm.logprobs) == (cold.token_ids, cold.logprobs)
        assert (warm.cached_token_count, warm.prompt_token_count) == (prefix, length)
        assert sorted(warm_state.kept) == [n for n in sorted(kept) if n > prefix]
    # Rows decoding together keep their own caches and checkpoint only the lengths they ask for.
    fruit = apply_chat_template(processor, model.config, f"Name a fruit. {rules}", num_images=0)
    run(GenerationRequest(prompt=fruit, prompt_cache_state=(alone := state(make_prompt_cache(lm), {2048}))))
    first, second = state(_own_copy(kept[2048]), {4096}), state(make_prompt_cache(lm), {2048})
    rows = run(GenerationRequest(prompt=prompt, prompt_cache_state=first),
               GenerationRequest(prompt=fruit, prompt_cache_state=second))
    assert [row.cached_token_count for row in rows] == [2048, 0]
    assert (sorted(first.kept), sorted(second.kept)) == ([4096], [2048])
    states = lambda cache: [v for _, v in tree_flatten([entry.state for entry in cache])]
    for got, want in ((first.kept[4096], kept[4096]), (second.kept[2048], alone.kept[2048])):
        assert all(mx.array_equal(a, b).item() for a, b in zip(states(got), states(want), strict=True))
    ask = lambda question: apply_chat_template(processor, model.config, f"{rules} {question}", num_images=0)
    short, long = ask("Name a colour."), ask("Name a fruit, then a vegetable.")
    seed, make = {}, lambda: make_prompt_cache(lm)
    (cold,) = run(GenerationRequest(prompt=long, prompt_cache_state=_SharedPrefixState(seed, make)))
    # Both rows are added resuming 2048 tokens; the later one takes the 4096 the earlier banks.
    banked = {ids: cache for ids, cache in seed.items() if len(ids) == 2048}
    rows = run(*(GenerationRequest(prompt=p, prompt_cache_state=_SharedPrefixState(banked, make))
                 for p in (long, short)))
    assert [row.cached_token_count for row in rows] == [4096, 2048]
    # Its prefill, which yields the first token, is the cold row's; decode depends on neighbours.
    assert (rows[0].token_ids[0], rows[0].logprobs[0]) == (cold.token_ids[0], cold.logprobs[0])
    # Rows merge by cache class and keep the receiving cache's window.
    for cache in (None, make_prompt_cache(lm, max_kv_size=8192)):
        other = None if cache is None else state(cache)
        with pytest.raises(BatchRowRefused, match="laid out unlike"):
            run(GenerationRequest(prompt=prompt, prompt_cache_state=state(make_prompt_cache(lm, max_kv_size=4096))),
                GenerationRequest(prompt=prompt, prompt_cache_state=other))
    # FastVLM expands its image placeholder, so its ids cannot name cache offsets.
    image = Image.new("RGB", (64, 64), (200, 40, 40))
    pictured = apply_chat_template(processor, model.config, f"Describe it. {rules}", num_images=1)
    cold_state = state(make_prompt_cache(lm))
    run(GenerationRequest(prompt=pictured, image=image, prompt_cache_state=cold_state))
    assert cold_state.kept == {}
    with pytest.raises(BatchRowRefused, match="expands past"):
        run(GenerationRequest(prompt=pictured, image=image, prompt_cache_state=state(_own_copy(kept[2048]))))


@metal_only
def test_vlm_stream_rows_resume_from_their_own_quantized_cache_bitwise(monkeypatch):
    from mlx_vlm import load
    from mlx_vlm.models.cache import QuantizedKVCache, make_prompt_cache
    from mlx_vlm.prompt_utils import apply_chat_template
    from unsloth_zoo.mlx.generate import BatchRowRefused, BatchStream, GenerationDefaults, GenerationRequest
    model, processor = load(VLM_MODEL)
    model._is_vlm_model = True
    rules = " ".join(f"Rule {i}: answer tersely." for i in range(420))
    prompt = apply_chat_template(processor, model.config, f"Name a colour. {rules}", num_images=0)
    fruit = apply_chat_template(processor, model.config, "Name a fruit.", num_images=0)
    quantized = lambda bits: [e.to_quantized(group_size=64, bits=bits) for e in make_prompt_cache(model.language_model)]

    def run(*requests):
        defaults = GenerationDefaults(max_tokens=12, prefill_batch_size=1, completion_batch_size=4)
        with BatchStream(model, processor, defaults=defaults) as stream:
            rows = [stream.add(request) for request in requests]
            results = {}
            while len(results) < len(rows):
                results.update((e.index, e.result) for e in stream.step() if e.result is not None)
        return [results[row] for row in rows]

    # mlx-vlm before 0.7 cannot turn a quantized row into a batch cache.
    mergeable = "prefix_cache_merge" in vars(QuantizedKVCache)
    with monkeypatch.context() as patch:
        patch.delattr(QuantizedKVCache, "prefix_cache_merge", raising=False)
        with pytest.raises(BatchRowRefused, match="cannot batch a row's own quantized cache"):
            run(GenerationRequest(prompt=fruit, prompt_cache_state=_RowCacheState(quantized(4))))
    if not mergeable:
        return
    cold_state = _RowCacheState(quantized(4), {2048})
    (cold,) = run(GenerationRequest(prompt=prompt, prompt_cache_state=cold_state))
    warm_state = _RowCacheState(_own_copy(cold_state.kept[2048]))
    (warm,) = run(GenerationRequest(prompt=prompt, prompt_cache_state=warm_state))
    assert (warm.token_ids, warm.logprobs, warm.cached_token_count) == (cold.token_ids, cold.logprobs, 2048)
    # The short row prefills while the resumed one decodes, so it joins a quantized batch.
    rows = run(GenerationRequest(prompt=prompt, prompt_cache_state=_RowCacheState(_own_copy(cold_state.kept[2048]))),
               GenerationRequest(prompt=fruit, prompt_cache_state=_RowCacheState(quantized(4))))
    assert [row.cached_token_count for row in rows] == [2048, 0]
    assert all(row.finish_reason in ("stop", "length") for row in rows)
    with pytest.raises(BatchRowRefused, match="laid out unlike"):
        run(GenerationRequest(prompt=fruit, prompt_cache_state=_RowCacheState(quantized(4))),
            GenerationRequest(prompt=fruit, prompt_cache_state=_RowCacheState(quantized(8))))
    # A uniformly quantizing stream converts a float row from its first token, as mlx-vlm's batch does.
    streamed = _RowCacheState(make_prompt_cache(model.language_model), {2048})
    defaults = GenerationDefaults(max_tokens=12, prefill_batch_size=1, completion_batch_size=4, kv_bits=4)
    with BatchStream(model, processor, defaults=defaults) as stream:
        row = stream.add(GenerationRequest(prompt=prompt, prompt_cache_state=streamed))
        results = {}
        while row not in results:
            results.update((e.index, e.result) for e in stream.step() if e.result is not None)
    assert results[row].finish_reason in ("stop", "length")
    assert isinstance(streamed.kept[2048][0], QuantizedKVCache)
    ask = lambda question: apply_chat_template(processor, model.config, f"{rules} {question}", num_images=0)
    banked, make = {}, lambda: make_prompt_cache(model.language_model)
    with BatchStream(model, processor, defaults=defaults) as stream:
        rows = [stream.add(GenerationRequest(prompt=ask(q), prompt_cache_state=_SharedPrefixState(banked, make)))
                for q in ("Name a fruit, then a vegetable.", "Name a colour.")]
        results = {}
        while len(results) < len(rows):
            results.update((e.index, e.result) for e in stream.step() if e.result is not None)
    assert results[rows[0]].cached_token_count >= 2048 and results[rows[1]].cached_token_count == 0


@metal_only
def test_vlm_stream_turboquant_rows_quantize_their_own_cache_as_a_single_decode_does(monkeypatch):
    from mlx_vlm import load
    from mlx_vlm.models.cache import QuantizedKVCache, make_prompt_cache
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.turboquant import TurboQuantKVCache
    from unsloth_zoo.mlx.generate import BatchRowRefused, BatchStream, GenerationDefaults, GenerationRequest
    model, processor = load(VLM_MODEL)
    model._is_vlm_model = True
    rules = " ".join(f"Rule {i}: answer tersely." for i in range(420))
    prompt = apply_chat_template(processor, model.config, f"Name a colour. {rules}", num_images=0)
    fruit = apply_chat_template(processor, model.config, "Name a fruit.", num_images=0)
    fresh = lambda: make_prompt_cache(model.language_model)

    def run(*requests, start=0):
        defaults = GenerationDefaults(max_tokens=12, prefill_batch_size=1, completion_batch_size=4,
                                      kv_bits=3.5, kv_quant_scheme="turboquant", quantized_kv_start=start)
        with BatchStream(model, processor, defaults=defaults) as stream:
            rows = [stream.add(request) for request in requests]
            results = {}
            while len(results) < len(rows):
                results.update((e.index, e.result) for e in stream.step() if e.result is not None)
        return [results[row] for row in rows]

    mergeable = "prefix_cache_merge" in vars(QuantizedKVCache)
    with monkeypatch.context() as patch:
        patch.delattr(QuantizedKVCache, "prefix_cache_merge", raising=False)
        with pytest.raises(BatchRowRefused, match="cannot batch a row's own quantized cache"):
            run(GenerationRequest(prompt=fruit, prompt_cache_state=_RowCacheState(fresh())))
    if not mergeable:
        return
    # A float row cache is quantized after each prefill forward, before its checkpoint.
    cold_state = _RowCacheState(fresh(), {2048})
    (cold,) = run(GenerationRequest(prompt=prompt, prompt_cache_state=cold_state))
    assert isinstance(cold_state.kept[2048][0], TurboQuantKVCache)
    (warm,) = run(GenerationRequest(prompt=prompt, prompt_cache_state=_RowCacheState(_own_copy(cold_state.kept[2048]))))
    assert (warm.token_ids, warm.logprobs, warm.cached_token_count) == (cold.token_ids, cold.logprobs, 2048)
    # A cold row opens float but decodes quantized, so it joins the resumed one; so does a
    # one-token row, whose only forward is its last.
    rows = run(GenerationRequest(prompt=prompt, prompt_cache_state=_RowCacheState(_own_copy(cold_state.kept[2048]))),
               GenerationRequest(prompt=fruit, prompt_cache_state=_RowCacheState(fresh())),
               GenerationRequest(prompt="Hi", prompt_cache_state=_RowCacheState(fresh())))
    assert [row.cached_token_count for row in rows] == [2048, 0, 0] and rows[2].prompt_token_count == 1
    assert all(row.finish_reason in ("stop", "length") for row in rows)
    # Rows share one codec, and convert at one length.
    seeded = fresh()
    seeded[:-1] = [TurboQuantKVCache(bits=3.5, seed=1) for _ in seeded[:-1]]
    with pytest.raises(BatchRowRefused, match="laid out unlike"):
        run(GenerationRequest(prompt=fruit, prompt_cache_state=_RowCacheState(fresh())),
            GenerationRequest(prompt=fruit, prompt_cache_state=_RowCacheState(seeded)))
    with pytest.raises(BatchRowRefused, match="quantized_kv_start=0"):
        run(GenerationRequest(prompt=fruit, prompt_cache_state=_RowCacheState(fresh())), start=100)


@metal_only
def test_vlm_stream_prefills_a_short_arrival_ahead_of_a_long_prefill_bitwise():
    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template
    from unsloth_zoo.mlx.generate import BatchStream, GenerationDefaults, GenerationRequest
    model, processor = load(VLM_MODEL)
    model._is_vlm_model = True
    rules = " ".join(f"Rule {i}: answer tersely." for i in range(700))
    long = GenerationRequest(prompt=apply_chat_template(processor, model.config, f"{rules} Name a colour.", num_images=0))
    short = GenerationRequest(prompt=apply_chat_template(processor, model.config, "Name a fruit.", num_images=0))
    defaults = GenerationDefaults(max_tokens=4, prefill_batch_size=1, completion_batch_size=4)

    def run(*requests):
        with BatchStream(model, processor, defaults=defaults) as stream:
            rows, results, first = [stream.add(requests[0])], {}, []
            while len(results) < len(requests):
                for e in stream.step():
                    first += [e.index] if e.index not in first else []
                    if e.result is not None:
                        results[e.index] = e.result
                if len(rows) < len(requests):  # after the long row's first chunk
                    rows.append(stream.add(requests[1]))
        return [results[row] for row in rows], [rows.index(index) for index in first]

    (alone_long,), _ = run(long)
    (alone_short,), _ = run(short)
    (got_long, got_short), order = run(long, short)
    assert got_long.prompt_token_count > 3 * 2048 and order == [1, 0]
    for got, alone in ((got_long, alone_long), (got_short, alone_short)):
        assert (got.token_ids[0], got.logprobs[0]) == (alone.token_ids[0], alone.logprobs[0])


@metal_only
def test_vlm_stop_strings_cut_generation_through_the_public_path():
    from PIL import Image
    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template
    from unsloth_zoo.mlx.generate import (
        GenerationDefaults,
        GenerationRequest,
        generate_batch,
    )
    model, processor = load(VLM_MODEL)
    model._is_vlm_model = True
    prompt = apply_chat_template(processor, model.config, "Name the colour.", num_images=1)
    request = GenerationRequest(prompt=prompt, image=Image.new("RGB", (64, 64), "red"))
    free = generate_batch(model, processor, [request],
                          defaults=GenerationDefaults(max_tokens=12))[0]
    assert len(free.text) > 1
    stop = free.text[1]
    stopped = generate_batch(model, processor, [request], defaults=GenerationDefaults(
        max_tokens=12, stop_strings=(stop,)))[0]
    assert (stopped.finish_reason, stopped.stop_match) == ("stop_string", stop)
    assert free.text.startswith(stopped.text) and stop not in stopped.text
    assert len(stopped.token_ids) < len(free.token_ids)

@metal_only
def test_arrays_cache_advance_patch_only_replaces_the_body_it_reproduces():
    from mlx_lm.models.cache import ArraysCache
    import unsloth_zoo.mlx.generate as generate_module
    from unsloth_zoo.mlx.generate import (
        _has_replaceable_advance,
        _install_arrays_cache_advance_fix,
    )

    _install_arrays_cache_advance_fix()
    # Descriptors now mediate these reads, so a second install must be a no-op.
    assert not _has_replaceable_advance(ArraysCache)
    installed = (ArraysCache.left_padding, ArraysCache.lengths)
    generate_module._ARRAYS_CACHE_ADVANCE_RESOLVED = False
    _install_arrays_cache_advance_fix()
    assert (ArraysCache.left_padding, ArraysCache.lengths) == installed

    # Candidacy follows the compiled body, however similar a different one reads.
    class Incrementing:
        lengths = left_padding = None

        def advance(self, N):
            if self.lengths is not None:
                self.lengths += N
            if self.left_padding is not None:
                self.left_padding += N

    class Decrementing(Incrementing):
        def __init__(self):
            raise AssertionError("candidacy must not instantiate the candidate")

        def advance(self, N):
            if self.lengths is not None:
                self.lengths -= N
            if self.left_padding is not None:
                self.left_padding -= N

    class DefaultedStep(Incrementing):
        def advance(self, N=1):
            if self.lengths is not None:
                self.lengths -= N
            if self.left_padding is not None:
                self.left_padding -= N

    assert not _has_replaceable_advance(Incrementing)
    assert not _has_replaceable_advance(DefaultedStep)
    assert _has_replaceable_advance(Decrementing)

    # A failed attempt is not a decision, so installation stays available afterwards.
    def raise_transient(arrays_cache):
        raise RuntimeError("transient")

    generate_module._ARRAYS_CACHE_ADVANCE_RESOLVED = False
    generate_module._has_replaceable_advance = raise_transient
    try:
        _install_arrays_cache_advance_fix()
        assert not generate_module._ARRAYS_CACHE_ADVANCE_RESOLVED
    finally:
        generate_module._has_replaceable_advance = _has_replaceable_advance
    _install_arrays_cache_advance_fix()
    assert generate_module._ARRAYS_CACHE_ADVANCE_RESOLVED

@metal_only
def test_arrays_cache_advance_defers_instead_of_stranding_metal_buffers():
    from mlx_lm.models.cache import ArraysCache
    from unsloth_zoo.mlx.generate import _install_arrays_cache_advance_fix

    _install_arrays_cache_advance_fix()
    cache = ArraysCache(1)
    cache[0] = mx.zeros((2, 4))
    cache.left_padding, cache.lengths = mx.array([0, 3]), mx.array([9, 5])
    mx.eval(cache[0], cache.left_padding, cache.lengths)
    mx.clear_cache()
    before = mx.get_active_memory()
    advances = [1] * 4000 + [2] * 4000
    for step in advances:
        cache.advance(step)
    # Stock strands a live scalar per field per call; clear_cache() cannot reclaim it.
    assert mx.get_active_memory() - before < len(advances)
    total = sum(advances)
    assert cache.left_padding.tolist() == [-total, 3 - total]
    assert cache.lengths.tolist() == [9 - total, 5 - total]

    # Deferred advances have to survive mlx-lm's own metadata plumbing.
    batch = ArraysCache.merge([ArraysCache(1), ArraysCache(1)])
    batch[0] = mx.zeros((2, 4))
    batch.left_padding = mx.array([0, 2])
    batch.advance(1)
    assert batch.make_mask(3).tolist() == [[True] * 3, [False, True, True]]
    batch.filter(mx.array([1]))
    assert batch.left_padding.tolist() == [1] and batch.batch_size == 1
    batch.prepare(lengths=[4])
    batch.advance(2)
    assert (batch.lengths.tolist(), batch.left_padding.tolist()) == ([2], [-1])
    batch.advance(3)
    batch.prepare(lengths=[7])
    assert batch.lengths.tolist() == [7]
    batch.finalize()
    batch.advance(5)
    assert batch.lengths is None and batch.left_padding is None

    # A pre-patch cache keeps its metadata despite the descriptors now shadowing it.
    legacy = ArraysCache(1)
    legacy.__dict__.clear()
    legacy.__dict__.update(cache=[None], left_padding=mx.array([2]), lengths=None)
    legacy.advance(1)
    assert legacy.left_padding.tolist() == [1] and legacy.lengths is None

    left = ArraysCache(1)
    left[0] = mx.zeros((2, 4))
    left.left_padding, left.lengths = mx.array([0, 4]), mx.array([6, 9])
    right = ArraysCache(1)
    right[0] = mx.zeros((1, 4))
    right.left_padding, right.lengths = mx.array([7]), mx.array([3])
    left.advance(2)
    right.advance(5)
    left.extend(right)
    assert left.left_padding.tolist() == [-2, 2, 2]
    assert left.lengths.tolist() == [4, 7, -2]


@metal_only
def test_a_penalised_row_answers_the_same_batched_and_alone():
    """A row shown its whole prompt is penalised against text it never sees alone, and the
    two prompts differ in length, so one offset cannot serve both rows."""
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    from unsloth_zoo.mlx.generate import (
        GenerationDefaults, GenerationRequest, generate_batch,
    )

    model, tokenizer = load(MODEL)
    prompts = ("red red red red red red red red. Name a colour:", "one one one. Count:")
    penalty = dict(repetition_penalty = 1.6, repetition_context_size = 20)
    batched = generate_batch(model, tokenizer, [
        GenerationRequest(prompt = prompt, max_tokens = 12,
                          logits_processors = make_logits_processors(**penalty))
        for prompt in prompts
    ], defaults = GenerationDefaults())

    for prompt, result in zip(prompts, batched):
        alone = [int(event.token) for event in stream_generate(
            model, tokenizer, tokenizer.encode(prompt, add_special_tokens = False),
            max_tokens = 12, sampler = make_sampler(temp = 0.0),
            logits_processors = make_logits_processors(**penalty),
        ) if event.finish_reason != "stop"]
        assert result.token_ids == alone, prompt


def _residual_equal(actual, expected):
    assert bool(mx.array_equal(actual.view(mx.uint8), expected.view(mx.uint8)))


class ResidualNormBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.RMSNorm(width)
        self.tail_norm = nn.RMSNorm(width)
        self.layer_scalar = mx.array([0.75])

    def __call__(self, h, tail = True):
        residual = h
        h = self.norm(h)
        h = residual + h
        if tail:
            residual = h
            gate = h * 0.5
            gate = self.tail_norm(gate)
            h = residual + gate
        if self.layer_scalar is not None:
            h = h * self.layer_scalar
        return h, tail


@pytest.mark.parametrize("cancel", [False, True])
@metal_only
def test_generation_mode_applies_residual_norm_and_restores(cancel):
    from contextlib import nullcontext
    from unsloth_zoo.mlx.generate import generation_mode

    models = [ResidualNormBlock(128), ResidualNormBlock(128)]
    root = nn.Sequential(*models)
    root.train()
    models[1].eval()
    flags = [module.training for module in root.modules()]
    x = mx.random.normal((1, 1, 128))
    expected = [model(x)[0] for model in models]
    mx.eval(expected)
    with pytest.raises(RuntimeError, match = "cancel") if cancel else nullcontext():
        with generation_mode(root):
            with generation_mode(root):
                assert all(not module.training for module in root.modules())
                for model, native in zip(models, expected):
                    assert type(model) is not ResidualNormBlock
                    _residual_equal(model(x)[0], native)
            assert all(type(model) is not ResidualNormBlock for model in models)
            if cancel:
                raise RuntimeError("cancel")
    assert all(type(model) is ResidualNormBlock for model in models)
    assert [module.training for module in root.modules()] == flags


@pytest.mark.parametrize("vlm", [False, True], ids = ["text", "vlm"])
@pytest.mark.parametrize("cancel", [False, True], ids = ["complete", "cancel"])
@metal_only
def test_loader_generate_applies_residual_norm_and_restores(monkeypatch, vlm, cancel):
    from contextlib import nullcontext
    import mlx_lm
    import mlx_vlm
    from unsloth_zoo.mlx import loader

    models = [ResidualNormBlock(128), ResidualNormBlock(128)]
    root = nn.Sequential(*models)
    root._tokenizer = types.SimpleNamespace(eos_token_ids = {2})
    root._is_vlm_model = vlm
    root.eval()
    x = mx.random.normal((1, 1, 128))
    expected = [model(x)[0] for model in models]
    mx.eval(expected)

    def stream(model, *args, **kwargs):
        assert model is root
        for index, (block, native) in enumerate(zip(models, expected)):
            assert type(block) is not ResidualNormBlock
            _residual_equal(block(x)[0], native)
            yield types.SimpleNamespace(token = 7 + index)
        if cancel:
            raise RuntimeError("cancel")

    monkeypatch.setattr(mlx_vlm if vlm else mlx_lm, "stream_generate", stream)
    with pytest.raises(RuntimeError, match = "cancel") if cancel else nullcontext():
        output = loader._mlx_generate(root, input_ids = [[1, 2]], max_new_tokens = 2)
        assert output.tolist() == [[1, 2, 7, 8]]
    assert all(type(model) is ResidualNormBlock for model in models)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@metal_only
def test_residual_norm_matches_reduction_rounding_and_scale(dtype):
    from unsloth_zoo.mlx import inference as decode
    for width in (63, 127, 128, 129, 1536, 4095, 4096):
        mx.random.seed(width)
        norm = nn.RMSNorm(width, eps = 1e-6)
        norm.weight = mx.random.normal((width,)).astype(dtype)
        norm.eval()
        # The last entry spreads magnitudes within the row rather than scaling the whole
        # row. Uniform rows agree bitwise even if the squares accumulate without fma.
        for magnitude in (0.01, 1., 10., None):
            spread = mx.power(10., mx.random.uniform(-4, 4, (1, 1, width)))
            x = (mx.random.normal((1, 1, width)) * (spread if magnitude is None else magnitude)).astype(dtype)
            residual = mx.random.normal(x.shape).astype(dtype)
            for scale in (None, mx.array(0.7, dtype), mx.array([1.5], dtype)):
                expected = residual + norm(x)
                if scale is not None:
                    expected = expected * scale
                _residual_equal(decode._residual_norm_add(norm, x, residual, scale), expected)


@metal_only
def test_residual_norm_scope_mutations_and_restore(monkeypatch):
    from unsloth_zoo.mlx import inference as decode
    models = [ResidualNormBlock(128), ResidualNormBlock(128)]
    root = nn.Sequential(*models)
    root.set_dtype(mx.bfloat16)
    root.eval()
    x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
    calls = []
    apply = decode._norm_add_apply
    def observed(*args):
        calls.append(None)
        return apply(*args)
    monkeypatch.setattr(decode, "_norm_add_apply", observed)
    with pytest.raises(RuntimeError, match = "cancel"):
        with decode.fused_residual_norm(root):
            assert all(type(m) is not ResidualNormBlock for m in models)
            with decode.fused_residual_norm(root):
                for model in models:
                    for factor in (2., .25):
                        model.norm.weight = model.norm.weight * factor
                        model.tail_norm.weight = model.tail_norm.weight / factor
                        for tail in (False, True):
                            calls.clear()
                            expected = ResidualNormBlock.__call__(model, x, tail)[0]
                            _residual_equal(model(x, tail)[0], expected)
                            assert len(calls) == 1 + int(tail)
            original = nn.RMSNorm.__call__
            monkeypatch.setattr(nn.RMSNorm, "__call__", lambda self, value: original(self, value) * 0.5)
            for model in models:
                calls.clear()
                _residual_equal(model(x)[0], ResidualNormBlock.__call__(model, x)[0])
                assert not calls
            raise RuntimeError("cancel")
    assert all(type(m) is ResidualNormBlock for m in models)


@metal_only
def test_residual_norm_fallbacks(monkeypatch):
    from unsloth_zoo.mlx import inference as decode
    model = ResidualNormBlock(128)
    model.eval()
    def unexpected(*args):
        pytest.fail("unsupported input reached residual norm kernel")
    monkeypatch.setattr(decode, "_norm_add_apply", unexpected)
    with decode.fused_residual_norm(model):
        for shape in ((2, 1, 128), (1, 3, 128)):
            x = mx.random.normal(shape)
            _residual_equal(model(x)[0], ResidualNormBlock.__call__(model, x)[0])
        x = mx.random.normal((1, 1, 128))
        model.train()
        _residual_equal(model(x)[0], ResidualNormBlock.__call__(model, x)[0])
        model.eval()
        model.norm.train()
        model.tail_norm.train()
        _residual_equal(model(x)[0], ResidualNormBlock.__call__(model, x)[0])
    monkeypatch.setattr(decode, "_residual_norm_kernel", lambda: None)
    with decode.fused_residual_norm(model):
        assert type(model) is ResidualNormBlock


@metal_only
def test_residual_norm_respects_an_existing_native_patch(monkeypatch):
    from unsloth_zoo.mlx import inference as decode
    class FreshBlock(ResidualNormBlock):
        pass
    model = FreshBlock(128)
    model.eval()
    original = mx.fast.rms_norm
    monkeypatch.setattr(mx.fast, "rms_norm", lambda *args, **kwargs: original(*args, **kwargs) * 0.5)
    x = mx.random.normal((1, 1, 128))
    expected = model(x)[0]
    with decode.fused_residual_norm(model):
        _residual_equal(model(x)[0], expected)


def test_residual_norm_scope_tolerates_a_stand_in_without_training(monkeypatch):
    """named_modules() yields whatever the generation API was handed, including the plain
    stand-ins it deliberately tolerates. Reading `.training` on one raises out of generation
    instead of skipping it, which is why the decode-fusion scope checks the type first."""
    from unsloth_zoo.mlx import inference as decode

    class _StandIn:
        pass

    class _Root:
        def named_modules(self):
            return [("stand_in", _StandIn())]

    # Off Metal the scope returns before the loop, so give it a kernel to get past that.
    monkeypatch.setattr(decode, "_residual_norm_kernel", lambda: object())
    root = _Root()
    with decode.fused_residual_norm(root) as yielded:
        assert yielded is root


def _quantized(N, K, group_size, dtype, bits = 4, seed = 0):
    w = mx.random.normal((N, K), key = mx.random.key(seed)) * 0.05
    return mx.quantize(w.astype(dtype), group_size = group_size, bits = bits)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("N", [384, 320])   # 128- and 64-column tiles
@pytest.mark.parametrize("bits", [4, 8])
@nax_only
def test_nax_small_m_qmm_matches_native(monkeypatch, bits, N, group_size, dtype):
    from unsloth_zoo.mlx import nax

    monkeypatch.setattr(nax, "_QMM_THREADGROUPS", {8: 20, 16: 20})   # several uneven K steps per split
    monkeypatch.setattr(nax, "_gpu_core_count", lambda: nax._QMM_MEASURED_CORES)
    for groups in (68, 17, 2):   # the last one runs unsplit
        K = group_size * groups
        w, scales, biases = _quantized(N, K, group_size, dtype, bits, seed = groups)
        for M in (2, 3, 8, 9, 16):
            # Rows differ in scale and the rows after M are NaN, so a row mix-up or a read past M shows.
            padded = mx.random.normal((M + 16, K), key = mx.random.key(M)) * (1 + mx.arange(M + 16)[:, None])
            x = mx.where(mx.arange(M + 16)[:, None] < M, padded, mx.nan).astype(dtype)[:M]
            got = nax.small_m_qmm(x, w, scales, biases, group_size, bits).astype(mx.float32)
            with mx.stream(mx.cpu):
                weight = mx.dequantize(w, scales.astype(mx.float32), biases.astype(mx.float32),
                                       group_size = group_size, bits = bits)
                exact = x.astype(mx.float32) @ weight.T
                terms = mx.abs(x.astype(mx.float32)) @ mx.abs(weight).T
                # Rounded once to dtype after an fp32 sum that only reorders the K products.
                bound = mx.finfo(dtype).eps * mx.abs(exact) + K * 2.0 ** -23 * terms
                mx.eval(exact, bound)
            assert mx.all(mx.abs(got - exact) <= bound).item(), (groups, M)


def test_nax_small_m_qmm_rejects_what_it_does_not_cover():
    from unsloth_zoo.mlx import nax

    for args in ((384, 512, 64, 4, "affine"), (320, 512, 32, 8, "affine")):
        assert nax.small_m_qmm_supported(*args)
    for args in ((352, 512, 64, 4, "affine"), (384, 480, 64, 4, "affine"), (384, 512, 16, 4, "affine"),
                 (384, 512, 64, 6, "affine"), (384, 512, 32, 4, "mxfp4")):
        assert not nax.small_m_qmm_supported(*args)


def test_nax_small_m_qmm_geometry_covers_k_within_threadgroup_memory():
    import itertools
    from unsloth_zoo.mlx import nax

    for M, N, K, group_size in itertools.product((2, 8, 9, 16), (320, 4096, 262144), (256, 2112, 5376, 16384),
                                                 (32, 64, 128)):
        if K % group_size:
            continue
        row_tile, column_tile, splits, unroll = nax.small_m_qmm_geometry(M, N, K, group_size)
        assert unroll in (2, 4, 8)
        groups, steps = K // group_size, -(-(K // group_size) // unroll)
        split_groups = -(-steps // splits) * unroll
        starts = [s * steps // splits * unroll for s in range(splits)] + [groups]   # the kernel's g0 per split
        assert starts[0] == 0 and starts == sorted(starts) and starts[-2] < groups
        assert max(b - a for a, b in zip(starts, starts[1:])) <= split_groups
        assert splits == 1 or -(-steps // (splits - 1)) > -(-steps // splits)   # every split shortens the longest
        assert row_tile * (split_groups + column_tile) * 4 <= nax._QMM_THREADGROUP_MEMORY
    # A vocabulary-wide head at large K needs more splits than the threadgroup target gives.
    assert nax.small_m_qmm_geometry(16, 262144, 16384, 32)[2] == 2


def test_nax_small_m_qmm_row_range_matches_bits_group_size_and_shape(monkeypatch):
    from unsloth_zoo.mlx import nax

    table = ((8, 32, 1 << 20, 4096, 9, 30), (8, None, 1 << 20, 4096, 1, 12), (4, None, 1 << 20, None, 5, 16))
    monkeypatch.setattr(nax, "_QMM_ROWS_BY_GPU", {nax._gpu_generation(): table})
    shapes = ((4096, 256, 32, 8), (4096, 256, 64, 8), (8192, 256, 64, 8), (4096, 128, 64, 8), (8192, 256, 64, 4))
    assert [nax.small_m_qmm_row_range(*shape) for shape in shapes] == [(9, 16), (2, 12), (0, -1), (0, -1), (5, 16)]
    monkeypatch.setattr(nax, "_QMM_ROWS_BY_GPU", {})
    monkeypatch.setattr(nax, "_QMM_ROWS_UNMEASURED", table[2:])
    assert nax.small_m_qmm_row_range(4096, 256, 64, 4) == (5, 16)


def test_nax_small_m_qmm_scales_with_gpu_cores_and_keys_rows_by_generation(monkeypatch):
    from unsloth_zoo.mlx import nax

    splits = {}
    for cores in (None, 10, 16, 40):
        monkeypatch.setattr(nax, "_gpu_core_count", lambda: cores)
        splits[cores] = [nax.small_m_qmm_geometry(M, 4096, 16384, 32)[2] for M in (2, 16)]
    assert splits == {None: [32, 19], 10: [22, 13], 16: [32, 19], 40: [64, 43]}
    for architecture, rows in (("applegpu_g17s", (6, 16)), ("applegpu_g17c", (6, 16)),
                               ("applegpu_g18s", (11, 16)), ("", (11, 16))):
        monkeypatch.setattr(nax, "_gpu_architecture", lambda: architecture)
        assert nax.small_m_qmm_row_range(4096, 4096, 64, 4) == rows, architecture


_EVERY_ROW = ((4, None, 0, None, 1, 16), (8, None, 0, None, 1, 16))


class _QuantizedHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.QuantizedEmbedding(512, 512, group_size = 64, bits = 4)
        self.proj = nn.QuantizedLinear(512, 512, bias = True, group_size = 64, bits = 8)
        self.odd = nn.QuantizedLinear(512, 352, bias = False, group_size = 64, bits = 4)
        self.set_dtype(mx.bfloat16)
        self.eval()

    def __call__(self, x):
        return self.embed.as_linear(self.proj(x)), self.odd(x)


@nax_only
def test_nax_quantized_linear_scope_routes_restores_and_falls_back(monkeypatch, caplog):
    import functools
    from unsloth_zoo.mlx import inference, nax
    from unsloth_zoo.mlx.generate import generation_mode

    model = _QuantizedHead()
    calls = []
    kernel = nax.small_m_qmm
    monkeypatch.setattr(nax, "small_m_qmm", lambda *args: calls.append((len(args[0]), args[5])) or kernel(*args))
    monkeypatch.setattr(nax, "nax_available", lambda: True)
    monkeypatch.setattr(nax, "kernel_probe_passed", lambda *args: True)
    # The 8-bit projection routes up to 8 rows only, so at 12 only the 4-bit embedding does.
    monkeypatch.setattr(nax, "_QMM_ROWS_BY_GPU", {nax._gpu_generation(): (_EVERY_ROW[0], (8, None, 0, None, 1, 8))})
    monkeypatch.setattr(inference, "_NAX_QMM_VERIFIED", {})
    caplog.set_level("INFO", logger = inference.__name__)

    def run(rows):
        calls.clear()
        return model(mx.random.normal((1, rows, 512), key = mx.random.key(rows)).astype(mx.bfloat16))

    natives = {rows: run(rows) for rows in (1, 4, 8, 12)}   # 8 fills the row tile: its own verified key
    linear_call = nn.QuantizedLinear.__call__
    with monkeypatch.context() as patch:   # a transient wrapper, e.g. training patches during eval sampling
        patch.setattr(nn.QuantizedLinear, "__call__", functools.wraps(linear_call)(lambda *a: linear_call(*a)))
        with inference.nax_quantized_linear(model):
            assert type(model.proj) is nn.QuantizedLinear
    with generation_mode(model):
        with inference.nax_quantized_linear(model):
            assert [type(m).__name__ for m in (model.embed, model.proj, model.odd)] == [
                "_NaxSmallMQuantizedEmbedding", "_NaxSmallMQuantizedLinear", "QuantizedLinear"]
        assert type(model.proj) is not nn.QuantizedLinear   # the outer scope still owns it
        for rows, native in natives.items():
            routed = run(rows)
            assert calls == {1: [], 4: [(4, 8), (4, 4)], 8: [(8, 8), (8, 4)], 12: [(12, 4)]}[rows]   # (rows, bits)
            for a, b in zip(routed, native):
                assert mx.allclose(a, b, rtol = 2e-2, atol = 2e-2).item()
        assert len(inference._NAX_QMM_VERIFIED) == 5 and all(inference._NAX_QMM_VERIFIED.values())
        assert sum("NAX kernel" in r.getMessage() for r in caplog.records) == 1
        model.train()
        run(4)
        assert not calls   # training stays native
        model.eval()
        monkeypatch.setattr(nn.QuantizedLinear, "__call__", lambda self, x: linear_call(self, x) * 0)
        assert not any(mx.any(out).item() for out in run(4)) and not calls   # drifted: native, new body
        monkeypatch.undo()
    assert (type(model.embed), type(model.proj)) == (nn.QuantizedEmbedding, nn.QuantizedLinear)
    assert "_unsloth_nax_qmm_rows" not in model.proj


@nax_only
def test_nax_quantized_linear_first_use_check_rejects_a_wrong_kernel(monkeypatch):
    from unsloth_zoo.mlx import inference, nax

    model = _QuantizedHead()
    kernel = nax.small_m_qmm
    monkeypatch.setattr(nax, "small_m_qmm", lambda *args: kernel(*args) * 1.1)
    monkeypatch.setattr(nax, "nax_available", lambda: True)
    monkeypatch.setattr(nax, "kernel_probe_passed", lambda *args: True)
    monkeypatch.setattr(nax, "_QMM_ROWS_BY_GPU", {nax._gpu_generation(): _EVERY_ROW})
    monkeypatch.setattr(inference, "_NAX_QMM_VERIFIED", {})
    native = model.proj(x := mx.random.normal((3, 512), key = mx.random.key(3)).astype(mx.bfloat16))
    compiled = mx.compile(model.proj)(x)   # stock compiled and eager can differ in the last bit
    with inference.nax_quantized_linear(model):
        assert mx.array_equal(mx.compile(model.proj)(x), compiled).item()   # unverifiable in a transform
        assert mx.array_equal(model.proj(x), native).item()
    assert list(inference._NAX_QMM_VERIFIED.values()) == [False]


@real_mlx_only
@pytest.mark.parametrize("blocker", [None, "kill switch", "no NAX", "gap closed", "distributed", "probe failed"])
def test_nax_quantized_linear_scope_stays_native(monkeypatch, blocker):
    from unsloth_zoo.mlx import inference, nax

    model = _QuantizedHead()
    monkeypatch.delenv("UNSLOTH_MLX_NAX_QMM", raising = False)
    monkeypatch.setattr(nax, "_QMM_ROWS_BY_GPU", {nax._gpu_generation(): _EVERY_ROW})
    monkeypatch.setattr(nax, "nax_available", lambda: blocker != "no NAX")
    monkeypatch.setattr(nax, "kernel_probe_passed", lambda *args: blocker != "probe failed")
    monkeypatch.setattr(nax, "gap_open", lambda name: blocker != "gap closed")
    if blocker == "kill switch":
        monkeypatch.setenv("UNSLOTH_MLX_NAX_QMM", "0")
    if blocker == "distributed":
        model._unsloth_mlx_distributed_parallel_mode = "tensor"
    with inference.nax_quantized_linear(model):
        assert (type(model.proj) is nn.QuantizedLinear) is (blocker is not None)


@real_mlx_only
def test_nax_quantized_linear_scope_swaps_a_shared_module_once(monkeypatch):
    from unsloth_zoo.mlx import inference, nax

    model = _QuantizedHead()
    model.alias = model.proj   # reachable under two paths, as tied or shared modules are
    monkeypatch.setattr(nax, "_QMM_ROWS_BY_GPU", {nax._gpu_generation(): _EVERY_ROW})
    monkeypatch.setattr(nax, "nax_available", lambda: True)
    monkeypatch.setattr(nax, "kernel_probe_passed", lambda *args: True)
    with inference.nax_quantized_linear(model):
        assert type(model.alias).__name__ == "_NaxSmallMQuantizedLinear"
        assert model.proj._unsloth_nax_qmm_scopes == 1
    assert type(model.proj) is nn.QuantizedLinear and "_unsloth_nax_qmm_rows" not in model.proj


class _PrefillProjections(nn.Module):
    def __init__(self):
        super().__init__()
        self.wide = nn.QuantizedLinear(512, 2048, bias = True, group_size = 64, bits = 8)
        self.small = nn.QuantizedLinear(512, 32, bias = False, group_size = 64, bits = 4)  # too few output tiles
        self.wide.bias = mx.random.normal((2048,), key = mx.random.key(5))
        self.set_dtype(mx.bfloat16)
        self.eval()

    def __call__(self, x):
        return self.wide(x), self.small(x)


def _same_bits(outputs, natives):
    return all(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item() for a, b in zip(outputs, natives))


@real_mlx_only
@metal_only
def test_dense_prefill_linear_matches_native_and_restores(monkeypatch):
    from unsloth_zoo.mlx import inference
    from unsloth_zoo.mlx.generate import generation_mode

    model = _PrefillProjections()
    # below the row floor, ragged, two tile-aligned row counts, and a batch whose row count MLX decides
    xs = [mx.random.normal(shape, key = mx.random.key(shape[1])).astype(mx.bfloat16)
          for shape in ((1, 1023, 512), (1, 1025, 512), (1, 2048, 512), (1, 1088, 512), (2, 1100, 512))]
    natives = [model(x) for x in xs]
    mx.eval(natives)
    dequantized, dequantize = [], mx.dequantize
    monkeypatch.setattr(mx, "dequantize", lambda *a, **k: dequantized.append(None) or dequantize(*a, **k))
    monkeypatch.setattr(inference, "_DENSE_QMM_VERIFIED", {})
    with generation_mode(model):
        with inference.dense_prefill_linear(model):
            pass
        assert type(model.wide) is type(model.small) is not nn.QuantizedLinear  # until the outer scope exits
        for x in xs[1:]:
            mx.eval(model(x))  # the first call of each row count is compared with the native matmul
        for x, native, dense in zip(xs, natives, (0, 1, 1, 1, 0)):
            dequantized.clear()
            assert _same_bits(model(x), native)
            assert len(dequantized) == dense
        model.train()
        dequantized.clear()
        mx.eval(model(xs[2]))
        assert not dequantized
        model.eval()
        linear_call = nn.QuantizedLinear.__call__
        with monkeypatch.context() as patch:  # a wrapper installed after the scope resolved the native body
            patch.setattr(nn.QuantizedLinear, "__call__", lambda self, x: linear_call(self, x) + 1)
            assert _same_bits(model(xs[2]), [native + 1 for native in natives[2]])
            assert not dequantized
        with monkeypatch.context() as patch:
            patch.setattr(inference, "_DENSE_QMM_MAX_WEIGHTS", 2048 * 512 - 1)
            mx.eval(model(xs[2]))
            assert not dequantized
        model.wide.biases = model.wide.biases.astype(mx.float32)  # the native result is float32 now
        assert model(xs[2])[0].dtype == mx.float32
    assert type(model.wide) is type(model.small) is nn.QuantizedLinear
    assert "_unsloth_dense_qmm_scopes" not in model.wide.__dict__
    assert sorted(inference._DENSE_QMM_VERIFIED.values()) == [False, False, True, True, True]
    monkeypatch.setenv("UNSLOTH_MLX_DENSE_PREFILL", "0")
    with inference.dense_prefill_linear(model):
        assert type(model.wide) is nn.QuantizedLinear


@real_mlx_only
@metal_only
def test_dense_prefill_linear_composes_with_the_nax_scope(monkeypatch):
    from unsloth_zoo.mlx import inference, nax
    from unsloth_zoo.mlx.generate import generation_mode

    model = _PrefillProjections()
    x = mx.random.normal((1, 2048, 512), key = mx.random.key(2)).astype(mx.bfloat16)
    few = mx.random.normal((1, 8, 512), key = mx.random.key(3)).astype(mx.bfloat16)
    native = model(x)
    mx.eval(native)
    monkeypatch.setattr(nax, "_QMM_ROWS_BY_GPU", {nax._gpu_generation(): _EVERY_ROW})
    monkeypatch.setattr(nax, "nax_available", lambda: True)
    monkeypatch.setattr(nax, "kernel_probe_passed", lambda *args: True)
    small_rows, small = [], inference._nax_small_m_qmm
    monkeypatch.setattr(inference, "_nax_small_m_qmm", lambda module, x, bindings: small_rows.append(x.shape[-2]) or None)
    dequantized, dequantize = [], mx.dequantize
    monkeypatch.setattr(mx, "dequantize", lambda *a, **k: dequantized.append(None) or dequantize(*a, **k))
    monkeypatch.setattr(inference, "_DENSE_QMM_VERIFIED", {})
    with inference.nax_quantized_linear(model):
        assert type(model.wide).__name__ == "_NaxSmallMQuantizedLinear"
        mx.eval(model(x))
        assert not dequantized   # outside the dense scope the NAX-routed linear falls back to the native call
        with generation_mode(model):
            assert type(model.wide).__name__ == "_NaxSmallMQuantizedLinear"
            mx.eval(model(x))
            dequantized.clear()
            assert _same_bits(model(x), native) and len(dequantized) == 1
            small_rows.clear()
            mx.eval(model(few))   # the NAX route is still asked first, and small calls never dequantize
            assert small_rows and len(dequantized) == 1
        assert type(model.wide).__name__ == "_NaxSmallMQuantizedLinear"
        assert "_unsloth_dense_qmm_scopes" not in model.wide.__dict__
        dequantized.clear()
        mx.eval(model(x))
        assert not dequantized
    assert type(model.wide) is nn.QuantizedLinear


@real_mlx_only
@metal_only
def test_dense_prefill_linear_first_use_check_rejects_a_different_result(monkeypatch, caplog):
    from unsloth_zoo.mlx import inference

    model = _PrefillProjections()
    x = mx.random.normal((1, 2048, 512)).astype(mx.bfloat16)
    native = model(x)
    mx.eval(native)
    dequantize = mx.dequantize
    monkeypatch.setattr(mx, "dequantize", lambda *a, **k: dequantize(*a, **k) * 1.5)
    monkeypatch.setattr(inference, "_DENSE_QMM_VERIFIED", {})
    with inference.dense_prefill_linear(model):
        for rows in (2048, 2048, 1536):   # the differing shape is not tried again at another row count
            assert _same_bits(model(x[:, :rows]), [out[:, :rows] for out in native])
    assert list(inference._DENSE_QMM_VERIFIED.values()) == [False, False]
    assert "the native call stays in use" in caplog.text


@metal_only
def test_vlm_cache_materialization_finishes_the_previous_snapshot(monkeypatch):
    from unsloth_zoo.mlx.generate import _VLMCacheMaterializer

    monkeypatch.setenv("MLX_VLM_BATCH_CACHE_EVAL_INTERVAL", "2")
    states = [mx.array([0.0]), mx.array([10.0])]
    generator = types.SimpleNamespace(
        _cache_eval_interval=2, _steps_counter=0, stream=mx.new_stream(mx.gpu),
        _generation_batch=types.SimpleNamespace(cache_states=lambda: [states], prompt_cache=[]),
    )
    def advance():
        with mx.stream(generator.stream):
            states[0] = states[0] + 1
            states[1][:] = states[1] - 3
            generator._steps_counter += 1
        return generator._steps_counter
    generator.next = advance
    evaluations, submissions = [], []
    real_eval, real_async = mx.eval, mx.async_eval
    monkeypatch.setattr(mx, "eval", lambda *xs: (evaluations.append(tuple(xs)), real_eval(*xs))[-1])
    monkeypatch.setattr(mx, "async_eval", lambda *xs: (submissions.append(tuple(xs)), real_async(*xs))[-1])
    materializer = _VLMCacheMaterializer(generator)
    try:
        assert generator._cache_eval_interval == 0
        assert materializer.next() == 1
        assert not submissions and not evaluations
        assert materializer.next() == 2
        snapshot = submissions[0]
        assert len(submissions) == 1 and len(snapshot) == 2 and not evaluations
        assert materializer.next() == 3
        assert len(evaluations) == 1 and len(evaluations[0]) == 2
        assert all(a is b for a, b in zip(evaluations[0], snapshot))
        assert [x.item() for x in snapshot] == [2, 4]
        assert [x.item() for x in states] == [3, 1]
        assert materializer.next() == 4
        final = submissions[-1]
    finally:
        materializer.close()
    assert len(evaluations) == 2 and len(evaluations[1]) == 2
    assert all(a is b for a, b in zip(evaluations[1], final))
    assert [x.item() for x in final] == [4, -2]
    assert generator._cache_eval_interval == 2 and not materializer.pending
    assert materializer.generator is None
    materializer.close()


@metal_only
@pytest.mark.parametrize("interval", [0, None, 2, "missing"])
def test_vlm_cache_materialization_preserves_unknown_or_disabled_generators(interval):
    from unsloth_zoo.mlx.generate import _VLMCacheMaterializer

    generator = types.SimpleNamespace(
        _cache_eval_interval=interval, _steps_counter=0, stream=mx.new_stream(mx.gpu),
        _generation_batch=types.SimpleNamespace(prompt_cache=[]), next=lambda: "unchanged",
    )
    if interval == 2:
        del generator.stream
    elif interval == "missing":
        del generator._cache_eval_interval
    materializer = _VLMCacheMaterializer(generator)
    assert materializer.next() == "unchanged"
    materializer.close()
    assert getattr(generator, "_cache_eval_interval", "missing") == interval


@metal_only
def test_vlm_cache_materialization_drains_before_a_failed_step(monkeypatch):
    from unsloth_zoo.mlx.generate import _VLMCacheMaterializer

    monkeypatch.setenv("MLX_VLM_BATCH_CACHE_EVAL_INTERVAL", "1")
    state = mx.array([4.0, -3.0]) * 2
    generator = types.SimpleNamespace(
        _cache_eval_interval=1, _steps_counter=0, stream=mx.new_stream(mx.gpu),
        _generation_batch=types.SimpleNamespace(prompt_cache=[types.SimpleNamespace(state=[state])]),
    )
    def advance():
        if generator._steps_counter:
            raise RuntimeError("decode failed")
        generator._steps_counter += 1
        return 1
    generator.next = advance
    materializer = _VLMCacheMaterializer(generator)
    assert materializer.next() == 1
    pending = materializer.pending[0]
    assert pending is not state
    evaluations = []
    real_eval = mx.eval
    monkeypatch.setattr(mx, "eval", lambda *xs: (evaluations.append(tuple(xs)), real_eval(*xs))[-1])
    with pytest.raises(RuntimeError, match="decode failed"):
        materializer.next()
    assert len(evaluations) == 1 and evaluations[0][0] is pending
    materializer.close()
    assert len(evaluations) == 1 and evaluations[0][0] is pending
    assert not materializer.pending and state.tolist() == [8, -6]
    assert generator._cache_eval_interval == 1


@metal_only
@pytest.mark.parametrize("configured, upstream, steps", [(None, 50, 256), ("50", 50, 50), (None, 300, 300)])
def test_vlm_cache_materialization_spaces_out_the_default_interval(monkeypatch, configured, upstream, steps):
    from unsloth_zoo.mlx.generate import _VLMCacheMaterializer

    if configured is None:
        monkeypatch.delenv("MLX_VLM_BATCH_CACHE_EVAL_INTERVAL", raising=False)
    else:
        monkeypatch.setenv("MLX_VLM_BATCH_CACHE_EVAL_INTERVAL", configured)
    state = mx.array([1.0])
    generator = types.SimpleNamespace(
        _cache_eval_interval=upstream, _steps_counter=0, stream=mx.new_stream(mx.gpu),
        _generation_batch=types.SimpleNamespace(cache_states=lambda: [state], prompt_cache=[]),
    )
    def advance():
        generator._steps_counter += 1
    generator.next = advance
    materializer = _VLMCacheMaterializer(generator)
    flushed = []
    for _ in range(600):
        materializer.next()
        if materializer.pending:
            flushed.append(generator._steps_counter)
    materializer.close()
    assert flushed[0] == steps and len(flushed) == 600 // steps
    assert generator._cache_eval_interval == upstream


@metal_only
def test_vlm_cache_materialization_reuses_cache_buffers(monkeypatch):
    from unsloth_zoo.mlx.generate import _VLMCacheMaterializer

    monkeypatch.setenv("MLX_VLM_BATCH_CACHE_EVAL_INTERVAL", "1")
    states = [mx.ones((1024, 1024)), mx.full((1024, 1024), 2.0)]
    mx.eval(states)
    generator = types.SimpleNamespace(
        _cache_eval_interval=1, _steps_counter=0, stream=mx.new_stream(mx.gpu),
        _generation_batch=types.SimpleNamespace(cache_states=lambda: [states], prompt_cache=[]),
    )
    def advance():
        with mx.stream(generator.stream):
            generator._steps_counter += 1
            for index, state in enumerate(states):
                state[index, index + 1] = generator._steps_counter * (1 if index else -1)
            mx.eval(states)
        return generator._steps_counter
    generator.next = advance
    materializer = _VLMCacheMaterializer(generator)
    try:
        initial = mx.get_active_memory()
        materializer.next()
        mx.eval(materializer.pending)
        assert mx.get_active_memory() - initial < 1024 * 1024
        mx.reset_peak_memory()
        initial = mx.get_active_memory()
        materializer.next()
        mx.eval(materializer.pending)
        assert mx.get_peak_memory() - initial < 1024 * 1024
        assert [state[index, index + 1].item() for index, state in enumerate(states)] == [-2, 2]
    finally:
        materializer.close()


@real_mlx_only
@pytest.mark.parametrize("vlm", [False, True], ids = ["text", "vlm"])
@pytest.mark.parametrize("scope_name", ["nax_quantized_linear", "dense_prefill_linear"])
def test_loader_generate_enters_the_quantized_linear_scopes(monkeypatch, vlm, scope_name):
    from contextlib import contextmanager
    import mlx_lm
    import mlx_vlm
    from unsloth_zoo.mlx import loader

    root = nn.Sequential(nn.Linear(4, 4))
    root._tokenizer = types.SimpleNamespace(eos_token_ids = {2})
    root._is_vlm_model = vlm
    entered = []

    @contextmanager
    def scope(model, *args):
        entered.append((model, *args))
        yield model

    def stream(model, *args, **kwargs):
        assert entered == [(root, vlm) if scope_name == "nax_quantized_linear" else (root,)]
        assert "int8_prefill" not in kwargs
        yield types.SimpleNamespace(token = 7)

    monkeypatch.setattr(loader, scope_name, scope)
    monkeypatch.setattr(mlx_vlm if vlm else mlx_lm, "stream_generate", stream)
    generated = loader._mlx_generate(root, input_ids = [[1, 2]], max_new_tokens = 1, int8_prefill = vlm)
    assert generated.tolist() == [[1, 2, 7]]


def _packed_codes(codes, bits):   # MLX's layout: a little-endian bitstream of `bits`-wide codes
    import numpy as np
    stream = (codes[..., None].astype(np.uint8) >> np.arange(bits, dtype = np.uint8)) & 1
    packed = np.packbits(stream.reshape(*codes.shape[:-1], -1), axis = -1, bitorder = "little")
    return mx.array(np.ascontiguousarray(packed).view(np.uint32))


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("bits", [3, 4, 5, 6, 8])
@nax_only
def test_nax_int8_prefill_partial_is_exact(bits, group_size):
    import numpy as np
    from unsloth_zoo.mlx import nax

    # Integer activations peaking at 127 per activation group (at most 64 wide) quantize to themselves,
    # scaled by 1 or 2 per row and activation group; zero weights under the peaks and two +-1 entries
    # per row keep every output (at most 2 * 2 * 510) exact in float16. Code sums of 379 to 383 per
    # activation group need both bfloat16 halves, and 17 quant groups span a full and a partial block.
    rng = np.random.default_rng(bits)
    E, N, K, G, span = 3, 320, 17 * group_size, 17, min(group_size, 64)
    peaks = (np.arange(0, K, span)[:, None] + [0, 11, 22]).ravel()
    codes = rng.integers(0, 1 << bits, (E, N, K), dtype = np.uint8)
    s = rng.integers(1, 3, (E, N, G)).astype(np.float32)
    codes[..., peaks] = np.repeat(rng.integers(0, 4, (E, N, G)), len(peaks) // G, -1)
    b = -s * codes[..., ::group_size]
    a = np.zeros((131, K), np.float32)
    a[:, peaks] = 127
    off_peak = np.setdiff1d(np.arange(K), peaks)
    for row in a:
        row[rng.choice(off_peak, 2, replace = False)] = rng.choice([-1, 1], 2)
    a *= np.repeat(rng.choice([1, 2], (131, K // span)), span, -1)
    W = codes * np.repeat(s, group_size, -1) + np.repeat(b, group_size, -1)
    exact = np.einsum("mk,enk->emn", a.astype(np.float64), W.astype(np.float64))
    a, s, b = (mx.array(v).astype(mx.float16) for v in (a, s, b))
    w = _packed_codes(codes, bits)
    dense = nax.int8_qmm(a, w[1], s[1], b[1], bits)
    assert np.array_equal(np.array(dense.astype(mx.float32)), exact[1])
    experts = np.repeat([0, 2], [60, 71]).astype(np.uint32)   # a ragged segment and an empty expert
    gathered = nax.int8_gather_qmm(a, w, s, b, mx.array(experts), bits)
    assert np.array_equal(np.array(gathered.astype(mx.float32)), exact[experts, np.arange(131)])
    experts, tokens = np.repeat([0, 2], [400, 648]).astype(np.uint32), rng.integers(0, 131, 1048).astype(np.uint32)
    gathered = nax.int8_gather_qmm(a, w, s, b, mx.array(experts), bits, mx.array(tokens))
    assert np.array_equal(np.array(gathered.astype(mx.float32)), exact[experts, tokens])
    x = mx.where(mx.arange(131)[:, None] == 70, mx.nan, a).astype(mx.bfloat16)
    y = nax.int8_qmm(x, w[0], *(v[0].astype(mx.bfloat16) for v in (s, b)), bits)
    rows = mx.isnan(y).any(axis = 1)
    assert rows.tolist() == [i == 70 for i in range(131)]


def _int8_prefill_dense_from_64_rows(monkeypatch):
    from unsloth_zoo.mlx import nax
    for name, value in (("_INT8_PREFILL_MIN_ROWS", 64), ("_INT8_PREFILL_MIN_N", 0), ("_INT8_PREFILL_MIN_K", 0)):
        monkeypatch.setattr(nax, name, value)


class _QuantizedMoE(nn.Module):
    def __init__(self):
        super().__init__()
        from mlx_lm.models.switch_layers import SwitchGLU
        self.proj = nn.QuantizedLinear(256, 320, bias = True, group_size = 64, bits = 8)
        self.small = nn.QuantizedLinear(256, 256, bias = False, group_size = 128, bits = 4)
        self.odd = nn.QuantizedLinear(256, 256, bias = False, group_size = 32, bits = 8, mode = "mxfp8")
        self.moe = SwitchGLU(256, 128, 8, bias = False)
        nn.quantize(self.moe, group_size = 64, bits = 8)
        self.set_dtype(mx.bfloat16)
        self.eval()

    def __call__(self, x, indices):
        return self.proj(x), self.small(x) + self.odd(x), self.moe(x, indices)


@nax_only
def test_nax_int8_prefill_routes_dense_and_expert_calls_from_their_row_minimum(monkeypatch):
    from unsloth_zoo.mlx import inference, nax
    from unsloth_zoo.mlx.generate import generation_mode

    model = _QuantizedMoE()
    calls = []
    for name in ("int8_qmm", "int8_gather_qmm"):
        kernel = getattr(nax, name)
        monkeypatch.setattr(nax, name, lambda *a, kernel = kernel, name = name: calls.append(
            ("gather" if "gather" in name else "dense", a[0].shape[0], a[1].shape[-2])) or kernel(*a))
    monkeypatch.setattr(nax, "nax_available", lambda: True)
    monkeypatch.setattr(nax, "kernel_probe_passed", lambda *args: True)
    _int8_prefill_dense_from_64_rows(monkeypatch)
    monkeypatch.setattr(nax, "_INT8_PREFILL_EXPERT_ROWS", {8: (20, 30)})
    monkeypatch.setattr(nax, "_QMM_ROWS_BY_GPU", {})
    monkeypatch.setattr(inference, "_NAX_INT8_QMM_VERIFIED", {})

    def run(rows, top = 2):
        calls.clear()
        x = mx.random.normal((1, rows, 256), key = mx.random.key(rows)).astype(mx.bfloat16)
        indices = mx.random.randint(0, 8, (1, rows, top), key = mx.random.key(rows + 1)).astype(mx.uint32)
        return model(x, indices)

    natives = {rows: run(rows) for rows in (1, 20, 64, 100, 128)}
    monkeypatch.delenv("UNSLOTH_MLX_INT8_PREFILL", raising = False)
    with generation_mode(model):   # off by default: bitwise stock
        assert all(mx.array_equal(a, b).item() for a, b in zip(run(100), natives[100])) and not calls
    monkeypatch.setenv("UNSLOTH_MLX_INT8_PREFILL", "1")
    with inference.nax_quantized_linear(model, False), inference.nax_quantized_linear(model, True), \
            inference.nax_quantized_linear(model):   # the outermost scope decides
        assert all(mx.array_equal(a, b).item() for a, b in zip(run(100), natives[100])) and not calls
    monkeypatch.delenv("UNSLOTH_MLX_INT8_PREFILL")
    for scope, packed in ((inference.nax_quantized_linear, False), (generation_mode, True)):
        with scope(model, True):   # alone, and with the gate and up projections packed into one gathered call
            assert not type(model.odd).__name__.startswith("_NaxInt8Prefill")
            assert type(model.moe.down_proj).__name__.startswith("_NaxInt8Prefill")
            assert (type(model.moe).__name__ != "SwitchGLU") is packed
            for rows, native in natives.items():
                routed = run(rows)
                # Dense projections route from 64 rows; the 2 * rows expert rows sort from 32 and route from
                # 30 per expert (240), or 20 (160) where the packed gate and up quantize each token once.
                T = 2 * rows
                down = [("gather", T, 256)] * (T >= 240)
                unpacked = [("gather", T, 128)] * 2 * (T >= 240)
                experts = ([("gather", rows, 256)] * (T >= 160) if packed else unpacked) + down
                assert calls == [("dense", rows, 320), ("dense", rows, 256)] * (rows >= 64) + experts
                for a, b in zip(routed, native):
                    assert mx.allclose(a, b, rtol = 5e-2, atol = 5e-2).item()
            model.train()
            run(100)
            assert not calls
            model.eval()
    assert type(model.moe.down_proj).__name__ == "QuantizedSwitchLinear" and type(model.proj) is nn.QuantizedLinear
    assert "_unsloth_nax_int8_prefill_rows" not in model.proj.__dict__
    assert "_unsloth_nax_int8_prefill" not in model.moe.up_proj.__dict__
    verified = inference._NAX_INT8_QMM_VERIFIED
    assert sorted(key[0] for key in verified) == ["dense", "dense", "gather", "gather", "gather"]
    assert all(inference._NAX_INT8_QMM_VERIFIED.values())


@nax_only
def test_nax_int8_prefill_packs_moe_gate_and_up_in_either_scope_order(monkeypatch):
    import contextlib
    from unsloth_zoo.mlx import inference, nax

    model = _QuantizedMoE()
    calls = []
    kernel = nax.int8_gather_qmm
    monkeypatch.setattr(nax, "int8_gather_qmm", lambda *a: calls.append(a[0].shape[0]) or kernel(*a))
    monkeypatch.setattr(nax, "nax_available", lambda: True)
    monkeypatch.setattr(nax, "kernel_probe_passed", lambda *args: True)
    monkeypatch.setattr(nax, "_QMM_ROWS_BY_GPU", {})
    # Per-token rows only: the gate and up projections route once packed.
    monkeypatch.setattr(nax, "_INT8_PREFILL_EXPERT_ROWS", {8: (20, 0)})
    monkeypatch.setattr(inference, "_NAX_INT8_QMM_VERIFIED", {})
    x = mx.random.normal((1, 100, 256), key = mx.random.key(0)).astype(mx.bfloat16)
    indices = mx.random.randint(0, 8, (1, 100, 2), key = mx.random.key(1)).astype(mx.uint32)
    int8 = lambda m: inference.nax_quantized_linear(m, True)
    with int8(model):   # swapped ahead of a pack, but unpacked calls stay native
        assert type(model.moe.gate_proj).__name__.startswith("_NaxInt8Prefill")
        mx.eval(model.moe(x, indices))
        assert not calls
    outputs = []
    for scopes in ((inference.fused_moe_gate_up, int8), (int8, inference.fused_moe_gate_up)):
        calls.clear()
        with contextlib.ExitStack() as stack:
            for scope in scopes:
                stack.enter_context(scope(model))
            assert type(model.moe).__name__.startswith("_FusedMoEGateUp")
            outputs.append(model.moe(x, indices))
        assert calls == [100]   # one packed call over the tokens, quantized once each
        assert type(model.moe).__name__ == "SwitchGLU" and "_unsloth_moe_gate_up" not in model.moe.__dict__
        assert all(type(p).__name__ == "QuantizedSwitchLinear" and "_unsloth_nax_int8_prefill" not in p.__dict__
                   for p in (model.moe.gate_proj, model.moe.up_proj, model.moe.down_proj))
    assert mx.array_equal(*outputs).item()


@nax_only
def test_nax_int8_prefill_checks_each_group_size_apart(monkeypatch):
    from mlx_lm.models.switch_layers import SwitchGLU
    from unsloth_zoo.mlx import inference, nax
    from unsloth_zoo.mlx.generate import generation_mode

    class Pairs(nn.Module):   # projections that differ only in group size
        def __init__(self):
            super().__init__()
            self.dense = [nn.QuantizedLinear(256, 256, bias = False, group_size = g, bits = 4) for g in (64, 128)]
            self.moe = [SwitchGLU(256, 128, 8, bias = False) for _ in range(2)]
            for moe, group_size in zip(self.moe, (64, 128)):
                nn.quantize(moe, group_size = group_size, bits = 8)
            self.set_dtype(mx.bfloat16)
            self.eval()

        def __call__(self, x, indices):
            return [layer(x) for layer in self.dense] + [moe(x, indices) for moe in self.moe]

    monkeypatch.setattr(nax, "nax_available", lambda: True)
    monkeypatch.setattr(nax, "kernel_probe_passed", lambda *args: True)
    _int8_prefill_dense_from_64_rows(monkeypatch)
    monkeypatch.setattr(nax, "_INT8_PREFILL_EXPERT_ROWS", {8: (20, 30)})
    monkeypatch.setattr(nax, "_QMM_ROWS_BY_GPU", {})
    monkeypatch.setattr(inference, "_NAX_INT8_QMM_VERIFIED", {})
    model = Pairs()
    x = mx.random.normal((1, 128, 256), key = mx.random.key(0)).astype(mx.bfloat16)
    indices = mx.random.randint(0, 8, (1, 128, 2), key = mx.random.key(1)).astype(mx.uint32)
    with generation_mode(model, True):
        mx.eval(model(x, indices))
    # Each kernel variant gets its own first-use check: two dense, and a packed gate/up and a down per MoE.
    assert sorted(key[0] for key in inference._NAX_INT8_QMM_VERIFIED) == ["dense"] * 2 + ["gather"] * 4
    assert all(inference._NAX_INT8_QMM_VERIFIED.values())


@nax_only
def test_nax_int8_prefill_first_use_check_rejects_a_wrong_kernel(monkeypatch):
    from unsloth_zoo.mlx import inference, nax

    model = _QuantizedMoE()
    kernel = nax.int8_qmm
    monkeypatch.setattr(nax, "int8_qmm", lambda *args: kernel(*args) * 1.05)
    monkeypatch.setattr(nax, "nax_available", lambda: True)
    monkeypatch.setattr(nax, "kernel_probe_passed", lambda *args: True)
    _int8_prefill_dense_from_64_rows(monkeypatch)
    monkeypatch.setattr(inference, "_NAX_INT8_QMM_VERIFIED", {})
    x = mx.random.normal((96, 256), key = mx.random.key(3)).astype(mx.bfloat16)
    native = model.proj(x)
    with inference.nax_quantized_linear(model, int8_prefill = True):
        assert mx.array_equal(model.proj(x), native).item()
    assert list(inference._NAX_INT8_QMM_VERIFIED.values()) == [False]


@nax_only
def test_nax_int8_prefill_first_use_check_accepts_all_zero_rows():
    import itertools
    from unsloth_zoo.mlx import inference, nax

    # mx.quantize stores an all-zero row as codes 0, biases 0 and a tiny scale.
    w = mx.random.normal((256, 512), key = mx.random.key(0)) * 0.05
    w = mx.where(mx.arange(256)[:, None] % 7 == 0, 0.0, w).astype(mx.bfloat16)
    x = mx.random.normal((200, 512), key = mx.random.key(1)).astype(mx.bfloat16)
    for bits, group_size in itertools.product((3, 4, 5, 6, 8), (32, 64, 128)):
        wq, s, b = mx.quantize(w, group_size = group_size, bits = bits)
        routed = nax.int8_qmm(x, wq, s, b, bits)
        assert inference._nax_int8_qmm_matches_native(x, wq, s, b, group_size, bits, routed, None)
        assert not inference._nax_int8_qmm_matches_native(x, wq, s, b, group_size, bits, routed * 1.05, None)
    # The allowance for those rows stays small enough to reject a 5% error on deeper fp16 rows too.
    x = mx.random.normal((160, 3072), key = mx.random.key(1)).astype(mx.float16)
    wq, s, b = mx.quantize((mx.random.normal((1024, 3072), key = mx.random.key(2)) * 0.03).astype(mx.float16),
                           group_size = 64, bits = 8)
    routed = nax.int8_qmm(x, wq, s, b, 8)
    assert inference._nax_int8_qmm_matches_native(x, wq, s, b, 64, 8, routed, None)
    assert not inference._nax_int8_qmm_matches_native(x, wq, s, b, 64, 8, routed * 1.05, None)


@nax_only
def test_nax_int8_prefill_differentiates_through_the_stock_op(monkeypatch):
    from unsloth_zoo.mlx import inference, nax
    from mlx.utils import tree_flatten

    model = _QuantizedMoE()
    calls = []
    kernel = nax._int8_qmm
    monkeypatch.setattr(nax, "_int8_qmm", lambda *args: calls.append(args[1].ndim) or kernel(*args))
    monkeypatch.setattr(nax, "nax_available", lambda: True)
    monkeypatch.setattr(nax, "kernel_probe_passed", lambda *args: True)
    _int8_prefill_dense_from_64_rows(monkeypatch)
    monkeypatch.setattr(nax, "_INT8_PREFILL_EXPERT_ROWS", {8: (20, 20)})
    monkeypatch.setattr(inference, "_NAX_INT8_QMM_VERIFIED", {})
    x = mx.random.normal((1, 100, 256), key = mx.random.key(5)).astype(mx.bfloat16)
    indices = mx.random.randint(0, 8, (1, 100, 2), key = mx.random.key(6)).astype(mx.uint32)
    model.proj.unfreeze(keys = ["scales", "biases"], recurse = False)
    weight = mx.random.normal((1, 100, 320), key = mx.random.key(7))   # a non-uniform cotangent
    loss = lambda x: (model.proj(x).astype(mx.float32) * weight).sum()
    dense = lambda x: {"x": mx.grad(loss)(x), **dict(tree_flatten(nn.value_and_grad(model.proj, loss)(x)[1]))}
    moe = mx.grad(lambda x: model.moe(x, indices).astype(mx.float32).sum())
    native_dense, native_moe = dense(x), moe(x)
    state = tree_flatten(model.parameters())
    with inference.nax_quantized_linear(model, int8_prefill = True):
        model(x, indices)
        calls.clear()
        routed_dense = dense(x)
        routed_moe = mx.compile(moe, inputs = model.state, outputs = model.state)(x)
    assert 2 in calls and 3 in calls
    # The dense gradients never read the forward output, so they match stock bitwise.
    assert sorted(routed_dense) == sorted(native_dense) == ["biases", "scales", "x"]
    assert all(mx.array_equal(routed_dense[k], native_dense[k]).item() for k in native_dense)
    assert mx.allclose(routed_moe, native_moe, rtol = 5e-2, atol = 5e-2).item()
    w, s, b = (model.moe.up_proj[name] for name in ("weight", "scales", "biases"))
    rows = mx.sort(indices.reshape(-1))
    tokens = mx.random.randint(0, 100, (200,), key = mx.random.key(8)).astype(mx.uint32)
    token_loss = lambda f: mx.grad(lambda x: (f(x).astype(mx.float32) * mx.arange(128)).sum())(x[0])
    mapped = token_loss(lambda x: nax.int8_gather_qmm(x, w, s, b, rows, 8, tokens))
    stock = token_loss(lambda x: mx.gather_qmm(x[tokens][:, None], w, s, b, rhs_indices = rows, transpose = True,
                                               bits = 8, sorted_indices = True)[:, 0])
    assert mx.allclose(mapped, stock, rtol = 1e-2, atol = 1.0).item()   # bf16 scatter-adds in either order
    after = tree_flatten(model.parameters())
    assert [k for k, _ in after] == [k for k, _ in state]
    assert all(mx.array_equal(a, b).item() for (_, a), (_, b) in zip(after, state))


@metal_only
def test_nax_int8_prefill_probes_gate_their_own_route(monkeypatch):
    from unsloth_zoo.mlx import inference, nax

    model = _QuantizedMoE()
    monkeypatch.setattr(nax, "nax_available", lambda: True)
    _int8_prefill_dense_from_64_rows(monkeypatch)
    monkeypatch.setattr(nax, "_QMM_ROWS_BY_GPU", {nax._gpu_generation(): _EVERY_ROW})
    monkeypatch.setattr(nax, "_INT8_PREFILL_EXPERT_ROWS", {8: (20, 20)})
    for failed in (nax.QMM_PROBE_KEY, nax.INT8_QMM_PROBE_KEY):
        monkeypatch.setattr(nax, "kernel_probe_passed", lambda key, *args: key != failed)
        with inference.nax_quantized_linear(model, True):
            assert (model.proj._unsloth_nax_qmm_rows != (0, -1)) is (failed != nax.QMM_PROBE_KEY)
            assert bool(model.proj._unsloth_nax_int8_prefill_rows) is (failed != nax.INT8_QMM_PROBE_KEY)
            assert (type(model.moe.down_proj).__name__.startswith("_NaxInt8Prefill")
                    is (failed != nax.INT8_QMM_PROBE_KEY))


@metal_only
def test_nax_int8_prefill_available_reports_what_the_route_takes(monkeypatch):
    from unsloth_zoo.mlx import inference, nax
    from unsloth_zoo.mlx.generate import generation_mode

    model = _QuantizedMoE()
    monkeypatch.setattr(nax, "nax_available", lambda: False)
    assert inference.int8_prefill_available(model) == (False, "nax_unavailable", 0)
    monkeypatch.setattr(nax, "nax_available", lambda: True)
    monkeypatch.setattr(nax, "kernel_probe_passed", lambda *args: True)
    monkeypatch.setattr(nax, "_INT8_PREFILL_EXPERT_ROWS", {})
    status = inference.int8_prefill_available(model)   # linears too small, and no expert threshold
    assert status == (False, "no_eligible_projections", 0) and not status
    _int8_prefill_dense_from_64_rows(monkeypatch)
    monkeypatch.setattr(nax, "_INT8_PREFILL_EXPERT_ROWS", {8: (20, 20)})

    def routed(model):
        with generation_mode(model, True):
            assert inference.int8_prefill_available(model) == status
            return sum("_unsloth_nax_int8_prefill_rows" in m.__dict__ or "_unsloth_nax_int8_prefill" in m.__dict__
                       for _, m in model.named_modules())

    model.train()   # a model loaded for training is reported as it would generate
    status = inference.int8_prefill_available(model)
    with inference.nax_quantized_linear(model, True):   # while a scope entered in training swaps nothing
        assert not any("_unsloth_nax_int8_prefill_rows" in m.__dict__ or "_unsloth_nax_int8_prefill" in m.__dict__
                       for _, m in model.named_modules())
    model.eval()
    assert status == (True, "", 5) and status and routed(model) == 5   # two dense linears and three experts
    monkeypatch.setattr(nax, "kernel_probe_passed", lambda key, *args: key != nax.INT8_QMM_PROBE_KEY)
    assert inference.int8_prefill_available(model) == (False, "probe_failed", 5)
    monkeypatch.setattr(nax, "kernel_probe_passed", lambda *args: True)
    # Per-token calls only: the gate and up qualify packed, the down projection never.
    monkeypatch.setattr(nax, "_INT8_PREFILL_EXPERT_ROWS", {8: (20, 0)})
    status = inference.int8_prefill_available(model)
    assert status == (True, "", 4) and routed(model) == 4
    trainable = _QuantizedMoE()   # a projection with trainable parameters never packs
    trainable.moe.gate_proj.unfreeze(keys = ["scales"], recurse = False)
    status = inference.int8_prefill_available(trainable)
    assert status == (True, "", 2) and routed(trainable) == 2
    custom = _QuantizedMoE()   # nor does a custom callable projection, which must not break generation
    gate = custom.moe.gate_proj
    custom.moe.gate_proj = lambda x, indices, sorted_indices = False: gate(x, indices, sorted_indices)
    custom.floats = [nn.Linear(256, 256), nn.Embedding(128, 256)]   # unquantized layers are skipped
    status = inference.int8_prefill_available(custom)
    assert status == (True, "", 2) and routed(custom) == 2
    mixed = _QuantizedMoE()   # gate and up quantized differently never pack
    up = mixed.moe.up_proj
    w = mx.dequantize(up.weight, up.scales, up.biases, group_size = 64, bits = 8)
    up.weight, up.scales, up.biases = mx.quantize(w, group_size = 64, bits = 4)
    up.bits = 4
    status = inference.int8_prefill_available(mixed)
    assert status == (True, "", 2) and routed(mixed) == 2
    model.proj.scales = model.proj.scales.astype(mx.float32)   # the kernel takes 16-bit scales only
    status = inference.int8_prefill_available(model)
    assert status == (True, "", 3) and routed(model) == 3
    call = nn.QuantizedLinear.__call__   # a rebound linear call is left to its owner
    monkeypatch.setattr(nn.QuantizedLinear, "__call__", lambda self, x: call(self, x))
    assert inference.int8_prefill_available(model) == (True, "", 2)
    model._unsloth_mlx_distributed_parallel_mode = "tensor"
    assert inference.int8_prefill_available(model) == (False, "distributed", 0)


@metal_only
def test_nax_int8_prefill_checkpoint_available_reads_the_headers_alone(monkeypatch, tmp_path):
    import json
    from unsloth_zoo.mlx import inference, nax
    model = _QuantizedMoE()
    model.save_weights(str(tmp_path / "model.safetensors"))
    for name, value in (("nax_available", lambda: True), ("kernel_probe_passed", lambda *args: True),
                        ("_INT8_PREFILL_EXPERT_ROWS", {8: (20, 30)}), ("_INT8_PREFILL_MIN_N", 0),
                        ("_INT8_PREFILL_MIN_K", 0)):
        monkeypatch.setattr(nax, name, value)
    def available(**layers):
        layers = {"group_size": 64, "bits": 8, "small": {"group_size": 128}, "odd": {"mode": "mxfp8"}, **layers}
        (tmp_path / "config.json").write_text(json.dumps({"quantization": layers}))
        return tuple(inference.int8_prefill_checkpoint_available(str(tmp_path)))

    assert available() == tuple(inference.int8_prefill_available(model)) == (True, "", 5)
    assert available(small = False) == available(small = {"mode": "mxfp4"}) == (True, "", 4)
    (tmp_path / "model.safetensors").unlink()
    assert available() == (False, "not_downloaded", 0)


def test_nax_int8_prefill_row_thresholds():
    from unsloth_zoo.mlx import nax

    dense, expert = nax.int8_prefill_min_rows, nax.int8_prefill_expert_min_rows
    cases = (   # (lookup, arguments, rows)
        (dense, (1024, 1024), 17), (dense, (4096, 4096), 17), (dense, (65536, 1024), 17),
        (dense, (8192, 512), 0), (dense, (4096, 1023), 0), (dense, (512, 4096), 0), (dense, (960, 4096), 0),
        (dense, (65600, 1024), 0), (dense, (151936, 2048), 0),
        (expert, (8, 2, True), 0), (expert, (8, 2), 0), (expert, (8, 3, True), 64), (expert, (8, 3), 256),
        (expert, (8, 4, True), 64), (expert, (8, 4), 64), (expert, (8, 5, True), 64), (expert, (8, 5), 256),
        (expert, (8, 6, True), 64), (expert, (8, 6), 256), (expert, (8, 8, True), 128), (expert, (8, 8), 256),
        (expert, (16, 8), 512), (expert, (16, 4, True), 128))
    assert [lookup(*args) for lookup, args, _ in cases] == [rows for _, _, rows in cases]



@real_mlx_only
@metal_only
def test_generation_mode_discovers_fusion_modules_once_per_entry(monkeypatch):
    from unsloth_zoo.mlx.generate import generation_mode

    root = nn.Sequential(ResidualNormBlock(128), ResidualNormBlock(128))
    named_modules = type(root).named_modules
    walks = []

    def counted(model):
        walks.append(model)
        return named_modules(model)

    monkeypatch.setattr(type(root), "named_modules", counted)
    x = mx.random.normal((1, 1, 128))
    for _ in range(2):
        root.train()
        root.layers[1].eval()
        flags = [module.training for module in root.modules()]
        expected = [layer(x)[0] for layer in root.layers]
        mx.eval(expected)
        before = len(walks)
        with generation_mode(root):
            assert len(walks) - before == 2  # Training flags, then post-eval fusion discovery.
            with generation_mode(root):
                assert len(walks) - before == 4
                for layer, native in zip(root.layers, expected):
                    assert type(layer) is not ResidualNormBlock
                    _residual_equal(layer(x)[0], native)
            assert all(type(layer) is not ResidualNormBlock for layer in root.layers)
        assert all(type(layer) is ResidualNormBlock for layer in root.layers)
        assert [module.training for module in root.modules()] == flags
        root.layers[1] = ResidualNormBlock(128)

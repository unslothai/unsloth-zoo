import types

import pytest
try:
    import mlx.core as mx
    import mlx.nn as nn
    _METAL = mx.metal.is_available()
except Exception:  # module-level nn.Module subclasses below need mlx to exist
    pytest.skip("requires mlx", allow_module_level=True)
metal_only = pytest.mark.skipif(not _METAL, reason="requires Apple Silicon Metal")
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
    # The cache patch must be installed before any decoding runs.
    order = []
    real_install = generate_module._install_arrays_cache_advance_fix
    real_adapter_generate = generate_module._TextBatchAdapter.generate

    def record_install():
        order.append("install")
        return real_install()

    def record_generate(self, requests):
        order.append("generate")
        return real_adapter_generate(self, requests)

    generate_module._install_arrays_cache_advance_fix = record_install
    generate_module._TextBatchAdapter.generate = record_generate
    try:
        batched = generate_batch(model, tokenizer, requests, defaults=defaults)
    finally:
        generate_module._install_arrays_cache_advance_fix = real_install
        generate_module._TextBatchAdapter.generate = real_adapter_generate
    from mlx_lm.models.cache import ArraysCache
    assert order == ["install", "generate"]
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
        # Parameters untouched, block runs in eval with the checkpoint patch
        # installed, flags restored, and the compiled step keeps training.
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
        # The RL contract: ids come from sampled events, logprobs align to them.
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
    # The terminal event contributes text only: its token is the stopping token,
    # or a repeat of the last body token when the budget ran out.
    tail = events[-1]
    body = events[:-1]
    assert results[0].token_ids == [int(event.token) for event in body]
    assert results[0].logprobs == pytest.approx(
        [float(event.logprobs[event.token].item()) for event in body], abs=0.02)
    assert results[0].text == "".join(event.text for event in events)
    # Derived from the stream, not our own rule: fewer body events than the
    # budget means it stopped early.
    assert tail is not None
    assert results[0].finish_reason == ("stop" if len(body) < 4 else "length")
    # Chunking must not pair a prompt with an earlier chunk's embeddings.
    chunked = generate_batch(model, processor, requests, defaults=GenerationDefaults(
        max_tokens=4, prefill_batch_size=1, completion_batch_size=1))
    assert [item.token_ids for item in chunked] == [item.token_ids for item in results]
    # Concurrent sequences must not share a detokenizer buffer, so text must
    # match too, not just ids.
    assert [item.text for item in chunked] == [item.text for item in results]
    # Independent buffers: no result may carry another's decoded text appended.
    assert all(r.text == "" or not r.text.startswith(results[0].text + results[1].text) for r in results)  # noqa: E501


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
    # The whole public path must carry stop strings, not just the event loop.
    stop = free.text[1]
    stopped = generate_batch(model, processor, [request], defaults=GenerationDefaults(
        max_tokens=12, stop_strings=(stop,)))[0]
    assert (stopped.finish_reason, stopped.stop_match) == ("stop_string", stop)
    # Trimming is token-aligned: whole tokens before the match survive, possibly
    # none, and never the stop string itself.
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
        # A matcher that reached for an instance trips this instead of passing quietly.
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
    # Re-arming a field replaces it outright, deferred count included.
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

    # extend() concatenates two caches carrying different deferred counts.
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

import pytest
try:
    import mlx.core as mx
    import mlx.nn as nn
    _METAL = mx.metal.is_available()
except Exception:
    _METAL = False
metal_only = pytest.mark.skipif(not _METAL, reason="requires Apple Silicon Metal")
MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"

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
    from unsloth_zoo.mlx.generate import (
        GenerationDefaults,
        GenerationRequest,
        generate_batch,
    )
    model, tokenizer = load(MODEL)
    requests = [
        GenerationRequest(prompt="The capital of France is", max_tokens=8),
        GenerationRequest(prompt="Two plus two equals", max_tokens=8),
    ]
    defaults = GenerationDefaults()
    batched = generate_batch(model, tokenizer, requests, defaults=defaults)
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

@metal_only
def test_compiled_training_state_survives_generation(monkeypatch):
    import functools
    import importlib
    import mlx.optimizers as optim
    mlx_lm_generate = importlib.import_module("mlx_lm.generate")
    from mlx.utils import tree_flatten
    from mlx_lm.sample_utils import make_sampler
    from unsloth_zoo.mlx.generate import (
        GenerationRequest, SamplingParams, generate_batch,
    )
    from unsloth_zoo.mlx.utils import (
        apply_gradient_checkpointing, remove_gradient_checkpointing,
    )
    events = []
    real_clear, real_close = mx.clear_cache, mlx_lm_generate.BatchGenerator.close
    real_insert = mlx_lm_generate.BatchGenerator.insert
    real_next = mlx_lm_generate.BatchGenerator.next_generated
    def clear():
        events.append("clear")
        real_clear()
    @functools.wraps(real_close)
    def close(generator):
        result = real_close(generator)
        events.append(("close", _current_limit("wired", wired)))
        return result
    @functools.wraps(real_insert)
    def insert(generator, *args, **kwargs):
        events.append("insert")
        return real_insert(generator, *args, **kwargs)
    @functools.wraps(real_next)
    def next_generated(generator):
        events.append("decode")
        return real_next(generator)
    monkeypatch.setattr(mx, "clear_cache", clear)
    monkeypatch.setattr(mlx_lm_generate.BatchGenerator, "close", close)
    monkeypatch.setattr(mlx_lm_generate.BatchGenerator, "insert", insert)
    monkeypatch.setattr(mlx_lm_generate.BatchGenerator, "next_generated", next_generated)
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
    wired = int(mx.device_info()["max_recommended_working_set_size"] * 0.5)
    previous_wired = mx.set_wired_limit(wired)
    apply_gradient_checkpointing(model)
    try:
        mx.eval(state, before := compiled_step(x, y))
        rng_before = tuple(mx.random.state[0].tolist())
        def snapshot():
            return [(name, value.tolist()) for name, value in tree_flatten(model.parameters())]
        state_after_train = snapshot()
        tokenizer = type("Tok", (), {"eos_token_ids": [0], "decode": lambda _, ids: str(ids)})()
        model.eval()
        oracle = []
        for token, _ in mlx_lm_generate.generate_step(mx.array([1, 2]), model, max_tokens=3, sampler=make_sampler(temp=0.7)):
            if int(token) == 0:
                break
            oracle.append(int(token))
        expected_rng = tuple(mx.random.state[0].tolist())
        model.train()
        model.layers[0].seen_states.clear()
        events.clear()
        mx.random.state[0] = mx.array(rng_before, dtype=mx.uint32)
        result = generate_batch(model, tokenizer, [GenerationRequest(prompt_token_ids=[1, 2], max_tokens=3, sampling=SamplingParams(temperature=0.7))])
        assert result[0].token_ids == oracle
        assert tuple(mx.random.state[0].tolist()) == expected_rng
        assert snapshot() == state_after_train
        mx.eval(state, after := compiled_step(x, y))
        assert snapshot() != state_after_train
        assert all(module.training for module in model.modules())
        assert (False, True) in model.layers[0].seen_states
        assert hasattr(_CompileBlock, "_orig_call") and optimizer.state["step"].item() == 2
        labels = [event[0] if isinstance(event, tuple) else event for event in events]
        assert labels[0] == "clear" and labels.index("insert") < labels.index("decode") < labels.index("close") < len(labels) - 1 and labels[-1] == "clear"
        assert ("close", wired) in events and _current_limit("wired", wired) == wired
        assert all(mx.isfinite(value).item() for value in (before, after))
    finally:
        remove_gradient_checkpointing(model)
        mx.set_wired_limit(previous_wired)

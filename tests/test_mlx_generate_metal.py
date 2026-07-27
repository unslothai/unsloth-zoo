import types

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

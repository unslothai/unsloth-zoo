import pytest
try:
    import mlx.core as mx
    import mlx.nn as nn
    _METAL = mx.metal.is_available()
except Exception:
    _METAL = False
metal_only = pytest.mark.skipif(not _METAL, reason="requires Apple Silicon Metal")
MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"

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

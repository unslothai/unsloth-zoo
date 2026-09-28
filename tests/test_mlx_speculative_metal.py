import pytest

try:
    import mlx.core as mx
except Exception:
    pytest.skip("requires mlx", allow_module_level = True)
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason = "Requires Metal")

MODEL = "mlx-community/Qwen3.5-0.8B-bf16"
CODE = (
    "def merge_sort(values):\n    if len(values) <= 1:\n        return values\n    middle = len(values) // 2\n"
    "    left = merge_sort(values[:middle])\n    right = merge_sort(values[middle:])\n    return merge(left, right)\n"
)
PROMPTS = [f"Copy this code exactly, then add a docstring to it:\n{CODE}", "Explain what a hash map is in two sentences."]


@pytest.fixture(scope = "module")
def qwen():
    from mlx_vlm import load
    model, processor = load(MODEL)
    tok = processor.tokenizer
    ids = [
        tok.encode(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt = True, tokenize = False), add_special_tokens = False)
        for p in PROMPTS
    ]
    return model, ids


def _prefill(lm, prompt, sampling, processors = ()):
    from mlx_vlm.models.cache import make_prompt_cache
    from unsloth_zoo.mlx.speculative import _RowSampler
    cache = make_prompt_cache(lm)
    logits = lm(mx.array([prompt]), cache = cache).logits[:, -1]
    for processor in processors:
        logits = processor(mx.array(prompt), logits)
    return cache, int(_RowSampler(sampling)(logits - mx.logsumexp(logits, -1, keepdims = True), 0).item())


def _solo(model, prompt, n, sampling, processors = ()):
    from unsloth_zoo.mlx.speculative import _RowSampler
    lm = model.language_model
    cache, token = _prefill(lm, prompt, sampling, processors)
    sample, out = _RowSampler(sampling), [token]
    for position in range(1, n):
        logits = lm(mx.array([[token]]), cache = cache).logits[:, -1]
        for processor in processors:
            logits = processor(mx.array(list(prompt) + out), logits)
        token = int(sample(logits - mx.logsumexp(logits, -1, keepdims = True), position).item())
        out.append(token)
    return out


def _run(model, prompts, n, controller, sampling, processors = None, patch = None):
    from unsloth_zoo.mlx.speculative import EngineRow, SpeculativeEngine
    engine = SpeculativeEngine(model, controller)
    if patch:
        patch(engine)
    out = {}
    for uid, prompt in enumerate(prompts):
        extra = (processors or {}).get(uid, ())
        cache, first = _prefill(model.language_model, prompt, sampling, extra)
        engine.add(EngineRow(cache = cache, pending = first, prompt = prompt, sampling = sampling, max_tokens = n, processors = extra, uid = uid))
        out[uid] = [first]
    drafted = {}
    while engine.rows:
        for step in engine.step():
            out[step.uid].extend(step.tokens)
            drafted[step.uid] = step.draft_n
    return [out[uid] for uid in range(len(prompts))], drafted, engine


def _rounds_only():
    from unsloth_zoo.mlx.speculative import DraftController, RoundPlan, RowPlan

    class RoundsOnly(DraftController):
        def plan(self, rows):
            return RoundPlan("round", rows = tuple(RowPlan("copy", min(s.copy_available, 4, max(0, s.remaining - 1))) if s.copy_available else RowPlan() for s in rows))
    return RoundsOnly(max_depth = 0, max_copy = 16)


@pytest.mark.parametrize("seed", [None, 1234])
def test_single_row_matches_solo_decoding_across_rounds_and_windows(qwen, seed):
    from unsloth_zoo.mlx.generate import SamplingParams
    from unsloth_zoo.mlx.speculative import DraftController, _RowSampler
    model, ids = qwen
    sampling = SamplingParams() if seed is None else SamplingParams(temperature = 0.8, top_p = 0.95, seed = seed)
    (out,), drafted, _ = _run(model, ids[:1], 160, DraftController(max_depth = 0, max_copy = 16), sampling)
    assert drafted[0] > 0
    assert out == _solo(model, ids[0], 160, sampling)
    if seed is not None:
        # Uniform logits: a key that ignored the position would draw one token everywhere.
        assert len({_RowSampler(sampling)(mx.zeros((1, 4096)), position).item() for position in range(8)}) > 1


def test_ragged_rounds_match_each_row_decoded_alone(qwen):
    from unsloth_zoo.mlx.generate import SamplingParams
    model, ids = qwen
    ban = lambda history, logits: logits + mx.where(mx.arange(logits.shape[-1]) == 13, -1e9, 0)
    out, drafted, _ = _run(model, [ids[0], ids[1], ids[0]], 128, _rounds_only(), SamplingParams(), processors = {2: [ban]})
    assert drafted[0] > 0 and drafted[2] == 0
    assert out == [_solo(model, ids[0], 128, SamplingParams()), _solo(model, ids[1], 128, SamplingParams()), _solo(model, ids[0], 128, SamplingParams(), [ban])]


def test_verify_that_replaces_caches_is_refused_and_decoding_continues(qwen):
    from unsloth_zoo.mlx.generate import SamplingParams
    model, ids = qwen

    def patch(engine):
        verify = engine._verify
        def replacing(inputs):
            engine.cache[-1] = type(engine.cache[-1])()
            return verify(inputs)
        engine._verify = replacing
    (out,), _, engine = _run(model, ids[:1], 48, _rounds_only(), SamplingParams(), patch = patch)
    assert "replaced its cache objects" in engine.round_refusal
    assert out == _solo(model, ids[0], 48, SamplingParams())

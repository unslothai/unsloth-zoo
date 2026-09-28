import pytest

try:
    import mlx.core as mx
except Exception:
    pytest.skip("requires mlx", allow_module_level = True)
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason = "Requires Metal")

MODEL = "mlx-community/Qwen3.5-0.8B-bf16"
MTP_MODEL = "Qwen/Qwen3.5-0.8B"
CODE = (
    "def merge_sort(values):\n    if len(values) <= 1:\n        return values\n    middle = len(values) // 2\n"
    "    left = merge_sort(values[:middle])\n    right = merge_sort(values[middle:])\n    return merge(left, right)\n"
)
PROMPTS = [f"Copy this code exactly, then add a docstring to it:\n{CODE}", "Explain what a hash map is in two sentences."]


def _load(repo):
    from mlx_vlm import load
    model, processor = load(repo)
    tok = processor.tokenizer
    ids = [
        tok.encode(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt = True, tokenize = False), add_special_tokens = False)
        for p in PROMPTS
    ]
    return model, ids


@pytest.fixture(scope = "module")
def qwen():
    return _load(MODEL)


@pytest.fixture(scope = "module")
def qwen_mtp():
    return _load(MTP_MODEL)


def _drafter(model, **kwargs):
    from mlx_vlm.utils import get_model_path
    from unsloth_zoo.mlx.speculative import native_mtp_drafter
    drafter = native_mtp_drafter(get_model_path(MTP_MODEL), model, **kwargs)
    drafter.dropped, push = {}, drafter.push
    def counting(row, tokens, hidden):
        drafter.dropped[id(row)] = drafter.dropped.get(id(row), 0) + max(0, len(row.tokens) + len(tokens) - drafter.max_lag)
        push(row, tokens, hidden)
    drafter.push = counting
    return drafter


def _prefill(lm, prompt, sampling, processors = ()):
    from mlx_vlm.models.cache import make_prompt_cache
    from unsloth_zoo.mlx.speculative import _RowSampler
    cache = make_prompt_cache(lm)
    out = lm(mx.array([prompt]), cache = cache, return_hidden = True)
    logits = out.logits[:, -1]
    for processor in processors:
        logits = processor(mx.array(prompt), logits)
    return cache, int(_RowSampler(sampling)(logits - mx.logsumexp(logits, -1, keepdims = True), 0).item()), out.hidden_states[-1]


def _solo(model, prompt, n, sampling, processors = ()):
    from unsloth_zoo.mlx.speculative import _RowSampler
    lm = model.language_model
    cache, token, _ = _prefill(lm, prompt, sampling, processors)
    sample, out = _RowSampler(sampling), [token]
    for position in range(1, n):
        logits = lm(mx.array([[token]]), cache = cache).logits[:, -1]
        for processor in processors:
            logits = processor(mx.array(list(prompt) + out), logits)
        token = int(sample(logits - mx.logsumexp(logits, -1, keepdims = True), position).item())
        out.append(token)
    return out


def _head_in_step(engine):
    offset = mx.array(engine.draft_cache[0].offset).reshape(-1).tolist()
    for i, row in enumerate(engine._rows):
        assert offset[i] == row.draft.position - engine.drafter.dropped.get(id(row.draft), 0)
        assert row.draft.position + len(row.draft.tokens) == len(row.tokens) - 1


def _run(model, prompts, n, controller, sampling, processors = None, patch = None, drafter = None):
    from unsloth_zoo.mlx.speculative import EngineRow, SpeculativeEngine
    engine = SpeculativeEngine(model, controller, drafter)
    if patch:
        patch(engine)
    out = {}
    for uid, prompt in enumerate(prompts):
        extra = (processors or {}).get(uid, ())
        cache, first, hidden = _prefill(model.language_model, prompt, sampling, extra)
        engine.add(EngineRow(cache = cache, pending = first, prompt = prompt, sampling = sampling, max_tokens = n, processors = extra, hidden = hidden, uid = uid))
        out[uid] = [first]
    drafted = {}
    while engine.rows:
        for step in engine.step():
            out[step.uid].extend(step.tokens)
            drafted[step.uid] = (step.draft_n, step.draft_n_accepted)
        if drafter is not None and engine.rows:
            _head_in_step(engine)
    return [out[uid] for uid in range(len(prompts))], drafted, engine


def _script(*cycle):
    """A controller replaying ``cycle``: ("plain", window) or (source, length) rounds."""
    from unsloth_zoo.mlx.speculative import DraftController, RoundPlan, RowPlan

    class Script(DraftController):
        def plan(self, rows):
            kind, n = cycle[self.steps_planned % len(cycle)]
            self.steps_planned += 1
            if kind == "plain":
                return RoundPlan("plain", n)
            available = lambda s: s.copy_available if kind == "copy" else n * s.can_draft
            return RoundPlan("round", rows = tuple(RowPlan(kind, min(available(s), n, max(0, s.remaining - 1))) if available(s) else RowPlan() for s in rows))
    controller = Script(max_depth = 4, max_copy = 16)
    controller.steps_planned = 0
    return controller


def _rounds_only():
    return _script(("copy", 4))


@pytest.mark.parametrize("seed", [None, 1234])
def test_single_row_matches_solo_decoding_across_rounds_and_windows(qwen, seed):
    from unsloth_zoo.mlx.generate import SamplingParams
    from unsloth_zoo.mlx.speculative import DraftController, _RowSampler
    model, ids = qwen
    sampling = SamplingParams() if seed is None else SamplingParams(temperature = 0.8, top_p = 0.6, seed = seed)
    (out,), drafted, _ = _run(model, ids[:1], 160, DraftController(max_depth = 0, max_copy = 16), sampling)
    assert drafted[0][0] > 0
    assert out == _solo(model, ids[0], 160, sampling)
    if seed is not None:
        # Uniform logits: a key that ignored the position would draw one token everywhere.
        assert len({_RowSampler(sampling)(mx.zeros((1, 4096)), position).item() for position in range(8)}) > 1


def test_ragged_rounds_match_each_row_decoded_alone(qwen):
    from unsloth_zoo.mlx.generate import SamplingParams
    model, ids = qwen
    ban = lambda history, logits: logits + mx.where(mx.arange(logits.shape[-1]) == 13, -1e9, 0)
    out, drafted, _ = _run(model, [ids[0], ids[1], ids[0]], 128, _rounds_only(), SamplingParams(), processors = {2: [ban]})
    assert drafted[0][0] > 0 and drafted[2][0] == 0
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


def test_mtp_rounds_match_solo_decoding(qwen_mtp):
    from unsloth_zoo.mlx.generate import SamplingParams
    from unsloth_zoo.mlx.speculative import DraftController
    model, ids = qwen_mtp
    greedy = SamplingParams()
    solo = [_solo(model, prompt, 96, greedy) for prompt in ids]
    out, drafted, _ = _run(model, ids, 96, _script(("draft", 3), ("copy", 4)), greedy, drafter = _drafter(model))
    assert out == solo and all(accepted >= 0.6 * count for count, accepted in drafted.values())
    (out,), drafted, _ = _run(model, ids[:1], 96, DraftController(max_depth = 4, max_copy = 16), greedy, drafter = _drafter(model))
    assert out == solo[0] and drafted[0][1] > 0


@pytest.mark.parametrize("max_lag", [128, 4])
def test_mtp_catches_up_after_plain_windows_and_copies(qwen_mtp, max_lag):
    from unsloth_zoo.mlx.generate import SamplingParams
    model, ids = qwen_mtp
    controller = _script(("plain", 8), ("draft", 3), ("copy", 4), ("draft", 3))
    accepted = []
    record = controller.record_round
    def record_drafts(plan, rows, counts, **kwargs):
        accepted.extend((count, row.length) for row, count in zip(plan.rows, counts) if row.source == "draft")
        record(plan, rows, counts, **kwargs)
    controller.record_round = record_drafts
    drafter = _drafter(model, max_lag = max_lag)
    (out,), _, _ = _run(model, ids[:1], 128, controller, SamplingParams(), drafter = drafter)
    assert out == _solo(model, ids[0], 128, SamplingParams())
    assert any(drafter.dropped.values()) == (max_lag < 8)
    assert sum(count for count, _ in accepted) >= 0.6 * sum(length for _, length in accepted)


def test_native_mtp_head_is_built_in_memory_from_mtp_tensors_only(qwen_mtp, monkeypatch):
    from mlx_vlm.speculative.drafters.mtp_split import MTPSplitter
    model, _ = qwen_mtp
    loaded, load_shard = [], MTPSplitter.load_shard
    monkeypatch.setattr(MTPSplitter, "load_shard", lambda self, file, keys: loaded.extend(keys) or load_shard(self, file, keys))
    monkeypatch.setattr(mx, "save_safetensors", lambda *args, **kwargs: pytest.fail("the build wrote a checkpoint"))
    assert _drafter(model) is not None
    assert loaded and all(key.startswith("mtp.") for key in loaded)

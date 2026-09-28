import pytest

try:
    import mlx.core as mx
except Exception:
    pytest.skip("requires mlx", allow_module_level = True)
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason = "Requires Metal")

MODEL = "mlx-community/Qwen3.5-0.8B-bf16"
MTP_MODEL = "Qwen/Qwen3.5-0.8B"
GEMMA, GEMMA_ASSISTANT = "mlx-community/gemma-4-e4b-it-bf16", "mlx-community/gemma-4-E4B-it-assistant-bf16"
CODE = (
    "def merge_sort(values):\n    if len(values) <= 1:\n        return values\n    middle = len(values) // 2\n"
    "    left = merge_sort(values[:middle])\n    right = merge_sort(values[middle:])\n    return merge(left, right)\n"
)
PROMPTS = [f"Copy this code exactly, then add a docstring to it:\n{CODE}", "Explain what a hash map is in two sentences."]
PAST_WINDOW = "Summarize these notes in one paragraph:\n" + "The planets orbit the sun in order: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune. " * 30


def _load(repo, *extra):
    from mlx_vlm import load
    model, processor = load(repo)
    tok = processor.tokenizer
    ids = [
        tok.encode(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt = True, tokenize = False), add_special_tokens = False)
        for p in [*PROMPTS, *extra]
    ]
    return model, ids


@pytest.fixture(scope = "module")
def qwen():
    return _load(MODEL)


@pytest.fixture(scope = "module")
def qwen_mtp():
    return _load(MTP_MODEL)


@pytest.fixture(scope = "module")
def gemma():
    from mlx_vlm.speculative.drafters import load_drafter
    from unsloth_zoo.mlx.speculative import AssistantDrafter
    model, ids = _load(GEMMA, PAST_WINDOW)
    return model, ids, AssistantDrafter(load_drafter(GEMMA_ASSISTANT)[0], model)


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
        if engine.draft_cache:
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



def test_assistant_drafts_from_the_target_kv_alone_and_in_batches(gemma, monkeypatch):
    from unsloth_zoo.mlx.generate import SamplingParams
    model, ids, drafter = gemma
    lm, cache = model.language_model, model.language_model.make_cache()
    out = lm(mx.array([ids[0]]), cache = cache, return_hidden = True, return_shared_kv = True)
    bound, bind, n = [], drafter.model.set_shared_kv, len(ids[0])
    monkeypatch.setattr(drafter.model, "set_shared_kv", lambda shared, offset, **kwargs: bound.append((shared, kwargs)) or bind(shared, offset, **kwargs))
    drafter.draft([], [drafter.start(ids[0], out.hidden_states[-1], 0)[0]], 3, cache)
    assert [where["position"].tolist() + where["kv_valid_len"].tolist() for _, where in bound] == [[n - 1, n]]
    assert all(mx.array_equal(bound[0][0][kind][0], keys) for kind, (keys, _) in out.shared_kv_states.items())

    compared, draft = [], drafter.draft
    def against_each_row(cache, rows, depth, target):
        drafts = draft(cache, rows, depth, target)
        alone = [draft(cache, [row], depth, [entry.extract(i) for entry in target]) for i, row in enumerate(rows) if len(rows) > 1 and row]
        compared.extend(row.tolist()[0] == batch for row, batch in zip(alone, drafts.tolist()))
        return drafts
    monkeypatch.setattr(drafter, "draft", against_each_row)
    _run(model, ids, 64, _script(("plain", 4), ("draft", 3)), SamplingParams(), drafter = drafter)
    assert len(compared) > 20 and sum(compared) >= 0.9 * len(compared)
    _, drafted, _ = _run(model, ids[:1], 64, _script(("draft", 3)), SamplingParams(), drafter = drafter)
    assert drafted[0][1] >= 0.4 * drafted[0][0]




def test_rounds_the_transaction_cannot_record_replay_the_accepted_prefix(qwen, monkeypatch):
    from mlx_vlm.speculative.cache_state import SpeculativeCacheTransaction
    from unsloth_zoo.mlx.generate import SamplingParams
    model, ids = qwen
    monkeypatch.setattr(SpeculativeCacheTransaction, "validate", lambda self, lengths: (_ for _ in ()).throw(RuntimeError("no record")))
    (out,), drafted, _ = _run(model, ids[:1], 128, _rounds_only(), SamplingParams())
    assert drafted[0][0] > 0 and out == _solo(model, ids[0], 128, SamplingParams())

def test_other_mtp_heads_match_a_head_that_never_drafted_even_in_a_wrapping_window():
    import copy
    from mlx_vlm.speculative.drafters import deepseek_v4_mtp
    from mlx_vlm.utils import get_model_and_args
    from unsloth_zoo.mlx.generate import SamplingParams
    from unsloth_zoo.mlx.speculative import HeadDrafter
    config = dict(model_type = "deepseek_v4", vocab_size = 256, hidden_size = 64, moe_intermediate_size = 32, num_hidden_layers = 2, compress_ratios = [0, 4],
                  num_attention_heads = 2, head_dim = 16, qk_rope_head_dim = 8, q_lora_rank = 16, o_groups = 1, o_lora_rank = 16, n_routed_experts = 4,
                  num_experts_per_tok = 2, index_n_heads = 1, index_head_dim = 16, num_hash_layers = 0, hc_mult = 2, sliding_window = 16)
    model = (arch := get_model_and_args(dict(config))[0]).Model(arch.ModelConfig.from_dict(config))
    head = deepseek_v4_mtp.Model(deepseek_v4_mtp.ModelConfig.from_dict({"model_type": "deepseek_v4_mtp", "text_config": config}))
    drafter, chunks, compared, prompt = HeadDrafter(head, model, max_lag = 4), [[]], [], [(7 * i + 3) % 200 + 1 for i in range(37)]
    start, push, catch_up, draft = drafter.start, drafter.push, drafter.catch_up, drafter.draft
    drafter.start = lambda prompt, hidden, token: start(prompt, hidden[:, -20:], token)  # as after a reused prompt prefix
    drafter.push = lambda row, tokens, hidden: chunks[-1].append((list(tokens), hidden)) or push(row, tokens, hidden)
    drafter.catch_up = lambda cache, rows: (catch_up(cache, rows), chunks.append([]))[0]
    def against_replay(cache, rows, depth, target):
        (fresh := copy.copy(head)).reset(model)
        fresh._next_position = len(prompt) - 20
        for chunk in filter(None, chunks):
            tokens, hidden = sum((t for t, _ in chunk), []), mx.concatenate([h for _, h in chunk])[None]
            fresh.accept_verified_tokens(hidden, mx.array([tokens[:-1]]), len(tokens) - 1, tokens[-1:], None, greedy = True)
        compared.append(all(mx.array_equal(a, b).item() for a, b in zip(*[(h._seed_hidden, h._cache[0].keys, h._cache[0].values) for h in (rows[0].head, fresh)])))
        return draft(cache, rows, depth, target)
    drafter.draft = against_replay
    _run(model, [prompt], 96, _script(("draft", 3), ("draft", 3), ("plain", 12), ("copy", 4), ("draft", 2)), SamplingParams(), drafter = drafter)
    assert len(compared) > 6 and all(compared)
    assert max(sum(len(t) for t, _ in chunk) for chunk in chunks[1:]) <= 8  # max_lag - 1 pending plus one 5-token round

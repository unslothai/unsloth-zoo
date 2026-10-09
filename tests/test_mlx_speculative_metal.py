from types import SimpleNamespace

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
    model._processor, tok = processor, processor.tokenizer
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
    from unsloth_zoo.mlx.speculative import companion_drafter
    # The bf16 per-layer embedding table is one 5.6 GB buffer, past the Metal buffer cap of 8 GB machines.
    limit = (getattr(mx, "device_info", None) or mx.metal.device_info)().get("max_buffer_length", 1 << 62)
    if limit < 6 << 30:
        pytest.skip(f"{GEMMA} needs a 5.6 GB Metal buffer; this device allows {limit} bytes")
    model, ids = _load(GEMMA, PAST_WINDOW)
    return model, ids, companion_drafter(GEMMA_ASSISTANT, model)


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


def _probe_matches_a_started_row(model, drafter, prompt, held):
    from mlx_vlm.models.cache import make_prompt_cache
    from unsloth_zoo.mlx.speculative import _capture, _features, probe_drafter
    hidden = _features(drafter, (lm := getattr(model, "language_model", model))(mx.array([prompt]), cache = make_prompt_cache(lm), **_capture(drafter)))
    shapes = lambda entries: [(type(entry), getattr(entry, "max_size", None), tuple(a.shape), a.dtype) for entry in entries for a in entry.state if a is not None]
    return shapes(probe_drafter(model, drafter, len(prompt))) == shapes(held(drafter.start(prompt, hidden, 0))) != []


def _prefill(lm, prompt, sampling, processors = (), drafter = None):
    from mlx_vlm.models.cache import make_prompt_cache
    from unsloth_zoo.mlx.speculative import _RowSampler, _capture, _features
    cache = make_prompt_cache(lm)
    out = lm(mx.array([prompt]), cache = cache, **_capture(drafter))
    logits = out.logits[:, -1]
    for processor in processors:
        logits = processor(mx.array(prompt), logits)
    return cache, int(_RowSampler(sampling)(logits - mx.logsumexp(logits, -1, keepdims = True), 0).item()), _features(drafter, out)


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


def _presence():
    """History-only: charges the reply's tokens, counted from its first call, and favours half the vocabulary by the last token's parity."""
    def penalty(tokens, logits):
        start, ids = penalty.__dict__.setdefault("start", tokens.shape[0]), mx.arange(logits.shape[-1])
        return logits - 3 * (ids[:, None] == tokens[start:]).any(-1) + 2 * ((ids + tokens[-1]) % 2 == 0)
    penalty.history_only = True
    return penalty


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
        cache, first, hidden = _prefill(model.language_model, prompt, sampling, extra, drafter)
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
            available = lambda kind, s: s.copy_available if kind == "copy" else n * s.can_draft
            return RoundPlan("round", rows = tuple(RowPlan(k, min(available(k, s), n, max(0, s.remaining - 1))) if available(k, s) else RowPlan() for k, s in zip(kind.split("/") * len(rows), rows)))
    controller = Script(max_depth = 4, max_copy = 16)
    controller.steps_planned = 0
    return controller


def _rounds_only():
    return _script(("copy", 4))


@pytest.mark.parametrize("seed", [None, 1234])
def test_single_row_matches_solo_decoding_across_rounds_and_windows(qwen, seed):
    from unsloth_zoo.mlx.generate import SamplingParams
    from unsloth_zoo.mlx.speculative import DraftController, _RowSampler
    (model, ids), seen = qwen, []
    sampling = SamplingParams() if seed is None else SamplingParams(temperature = 0.8, top_p = 0.6, seed = seed)
    (out,), drafted, _ = _run(model, ids[:1], 160, DraftController(max_depth = 0, max_copy = 16), sampling)
    assert drafted[0][0] > 0
    assert out == _solo(model, ids[0], 160, sampling)
    (cut,), _, engine = _run(model, ids[:1], 40, DraftController(max_depth = 0), sampling, patch = lambda e: setattr(e, "step", lambda step = e.step, calls = __import__("itertools").count(1): step(lambda: next(calls) % 3 == 0)))
    assert cut == out[:40] and engine.controller.plain_cost[1].count >= 10
    _run(model, ids[:1], 24, DraftController(max_depth = 0, can_copy = False), sampling, patch = lambda e: setattr(e, "step", lambda step = e.step: (seen.append((len(o.tokens), e._rows[0].emitted)) or o for o in step())))
    assert seen == [(1, emitted) for emitted in range(2, 25)]
    if seed is not None:
        # Uniform logits: a key that ignored the position would draw one token everywhere.
        assert len({_RowSampler(sampling)(mx.zeros((1, 4096)), position).item() for position in range(8)}) > 1


def test_ragged_rounds_match_each_row_decoded_alone(qwen):
    from unsloth_zoo.mlx.generate import SamplingParams
    model, ids = qwen
    ban = lambda history, logits: logits + mx.where(mx.arange(logits.shape[-1]) == 13, -1e9, 0)
    out, drafted, _ = _run(model, [ids[0], ids[1], ids[0], ids[0]], 128, _rounds_only(), SamplingParams(), processors = {2: [ban], 3: [_presence()]})
    assert drafted[0][0] > 0 and drafted[2][0] == 0 and drafted[3][1] > 0
    assert out == [_solo(model, ids[0], 128, SamplingParams()), _solo(model, ids[1], 128, SamplingParams()), *(_solo(model, ids[0], 128, SamplingParams(), [fn]) for fn in (ban, _presence()))]
    (window := _script(("plain", 16))).interrupts_plain, seen = lambda rows: False, []
    short = lambda e: setattr(e, "add", lambda row, add = e.add: add(__import__("dataclasses").replace(row, max_tokens = 3) if row.uid else row)) or setattr(e, "step", lambda step = e.step: seen.append(len(e._rows)) or step())
    out, _, _ = _run(model, ids[:2], 12, window, SamplingParams(), patch = short)
    assert seen == [2, 1] and out == [_solo(model, ids[0], 12, SamplingParams()), _solo(model, ids[1], 3, SamplingParams())]


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
    # A head that dropped context past max_lag accepts less; a misaligned catch-up would accept next to nothing.
    assert sum(count for count, _ in accepted) >= (0.6 if max_lag >= 8 else 0.5) * sum(length for _, length in accepted)


def test_native_mtp_head_is_built_in_memory_from_mtp_tensors_only(qwen_mtp, monkeypatch):
    from mlx_vlm.speculative.drafters.mtp_split import MTPSplitter
    model, _ = qwen_mtp
    loaded, load_shard = [], MTPSplitter.load_shard
    monkeypatch.setattr(MTPSplitter, "load_shard", lambda self, file, keys: loaded.extend(keys) or load_shard(self, file, keys))
    monkeypatch.setattr(mx, "save_safetensors", lambda *args, **kwargs: pytest.fail("the build wrote a checkpoint"))
    assert _drafter(model) is not None
    assert loaded and all(key.startswith("mtp.") for key in loaded)


def test_a_separate_mtp_head_checkpoint_is_wrapped_as_the_built_in_head(qwen_mtp, tmp_path):
    from mlx_vlm.speculative.drafters.mtp_split import detect_mtp_splitter, get_model_path
    from unsloth_zoo.mlx.speculative import MTPDrafter, companion_drafter
    assert detect_mtp_splitter(source := get_model_path(MTP_MODEL)).split(str(source), str(tmp_path)) and type(companion_drafter(tmp_path, qwen_mtp[0])) is MTPDrafter



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
    packed = pytest.importorskip("mlx_vlm.speculative.drafters").load_drafter(GEMMA_ASSISTANT)[0]
    packed.model.embed_tokens = packed.model.embed_tokens.to_quantized(32, 4, mode = "mxfp4")
    assert type(drafter)(pytest.importorskip("unsloth_zoo.mlx.speculative")._dense_table(packed), model).draft([], [drafter.start(ids[0], out.hidden_states[-1], 0)[0]], 3, cache).shape == (1, 3)

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




@pytest.mark.parametrize("native", [False, True])
def test_generate_step_decodes_our_drafter_through_the_engine(request, monkeypatch, native):
    import importlib
    from unsloth_zoo.mlx.generate import SamplingParams
    from unsloth_zoo.mlx.speculative import SpeculativeDraft, install_speculative_seam
    ar, (model, ids) = importlib.import_module("mlx_vlm.generate.ar"), request.getfixturevalue("qwen_mtp" if native else "qwen")
    monkeypatch.setattr(ar, "run_speculative_rounds", lambda *args, **kwargs: iter([("upstream", None)]))
    monkeypatch.setattr(ar, "SpeculativePrefill", ar.SpeculativePrefill)
    install_speculative_seam()
    install_speculative_seam()
    with monkeypatch.context() as patch:
        patch.setattr("unsloth_zoo.mlx.speculative.speculative_unavailable_reason", lambda: "too old") or pytest.raises(RuntimeError, install_speculative_seam)
    assert list(ar.run_speculative_rounds(model, object(), max_tokens = 1)) == [("upstream", None)]
    draft, rounds, started = SpeculativeDraft(_script(("draft", 3)) if native else _script(("copy", 4), ("plain", 3)), _drafter(model, max_lag = 32) if native else None), [], []
    monkeypatch.setattr(draft.controller, "record_round", lambda plan, *args, **kwargs: rounds.append(plan))
    if native:
        start = draft.drafter.start
        monkeypatch.setattr(draft.drafter, "start", lambda prompt, hidden, pending: started.append(hidden) or start(prompt, hidden, pending))
    generate = lambda **kwargs: [int(token) for token, _ in ar.generate_step(mx.array([ids[0]]), model, None, None, max_tokens = 96, temperature = 0.0, **kwargs)]
    for prefill_step_size in (16, None):  # unchunked, the final forward returns every prompt position's hidden
        # Chunking the prefill alone can flip a bf16 near-tie on Metal (M1 runners), so the reference is prefilled alike.
        plain = generate(prefill_step_size = prefill_step_size)
        penalized = generate(prefill_step_size = prefill_step_size, logits_processors = [_presence()])
        draft.prepare(ids[0], SamplingParams())
        assert generate(draft_model = draft, draft_kind = draft.draft_kind, prefill_step_size = prefill_step_size) == plain
        draft.prepare(ids[0], SamplingParams(), [penalty := _presence()])
        assert generate(draft_model = draft, draft_kind = draft.draft_kind, prefill_step_size = prefill_step_size, logits_processors = [penalty]) == penalized != plain
    assert rounds and [hidden.shape[1] for hidden in started] == ([32] * 4 if native else []) and draft.draft_n >= draft.draft_n_accepted > 0
    # The same positions from six chunks as from one forward: chunking moves them ~0.07 on average, a one-position shift ~3.3.
    assert not native or (started[0] - started[2]).abs().mean().item() < 0.5
    with pytest.raises(RuntimeError, match = "prepare"):
        generate(draft_model = draft, draft_kind = draft.draft_kind)

@pytest.mark.parametrize("companion", [None, "z-lab/Qwen3.5-4B-DFlash"])
def test_a_lazy_drafter_weighs_its_checkpoint_and_holds_a_started_row_without_allocating(request, companion):
    from mlx_vlm.utils import get_model_path, load
    from unsloth_zoo.mlx import speculative
    model = request.getfixturevalue("qwen_mtp")[0] if companion is None else load("mlx-community/Qwen3.5-4B-4bit", lazy = True)[0]
    before, lazy = mx.get_active_memory(), _drafter(model, lazy = True) if companion is None else speculative.companion_drafter(companion, model, lazy = True)
    grown = speculative.probe_drafter(model, lazy, 40) and mx.get_active_memory() - before  # before any eager build below
    stored = _drafter(model).weight_bytes if companion is None else sum(a.nbytes for a in mx.load(str(get_model_path(companion) / "model.safetensors")).values())
    assert grown < lazy.weight_bytes / 100 and lazy.weight_bytes == stored > 0 and (companion or _probe_matches_a_started_row(model, _drafter(model), list(range(1, 41)), lambda started: started[1]))


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
    assert _probe_matches_a_started_row(model, HeadDrafter(head, model), prompt, lambda started: started[0].head._cache)


def test_eagle3_narrower_than_its_target_drafts_from_the_concatenated_layers_row_by_row(qwen):
    from mlx_vlm.speculative.drafters import eagle3
    from mlx_vlm.utils import load_config
    from unsloth_zoo.mlx import generate, speculative
    (model, ids), config, text = qwen, load_config("RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3"), qwen[0].language_model.config.text_config
    config["transformer_layer_config"].update(hidden_size = 256, vocab_size = text.vocab_size, intermediate_size = 256, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 64)
    config.update(target_hidden_size = text.hidden_size, draft_vocab_size = 4096, eagle_aux_hidden_state_layer_ids = [2, text.num_hidden_layers // 2, text.num_hidden_layers - 1])
    drafter, SamplingParams = speculative.Eagle3Drafter(eagle3.Model(eagle3.ModelConfig.from_dict(config)), model.language_model), generate.SamplingParams
    for prompts, script in ((ids[:1], (("draft", 3), ("plain", 9), ("copy", 4), ("draft", 4))), (ids[:2], (("copy/draft", 4), ("draft", 3), ("copy", 4)))):
        out, drafted, _ = _run(model, prompts, 48, _script(*script), SamplingParams(), drafter = drafter)
        assert out == [_solo(model, prompt, 48, SamplingParams()) for prompt in prompts] and all(n for n, _ in drafted.values()) and drafter.start(prompts[0], None, 0) == (None, [])


@pytest.mark.parametrize("companion", [None, "z-lab/Qwen3.5-4B-DFlash"])
def test_speculative_batch_stream_rows_join_and_leave_as_if_decoded_alone(qwen, monkeypatch, companion):
    from copy import deepcopy
    from mlx_vlm.models.cache import make_prompt_cache
    from unsloth_zoo.mlx.generate import BatchStream, GenerationRequest, SamplingParams
    from unsloth_zoo.mlx.speculative import ContextDrafter, SpeculativeDraft
    (model, _), ar = qwen, __import__("mlx_vlm.generate.ar", fromlist = ["ar"])
    monkeypatch.setattr(model, "_is_vlm_model", True, raising = False)
    tok = model._processor.tokenizer
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt = True, tokenize = False) for p in PROMPTS]
    banked, size = {}, len(tok.encode(texts[0], add_special_tokens = False))
    state = SimpleNamespace(checkpoint = lambda count, cache: banked.update({count: deepcopy(cache)}))
    state.open = lambda prompt: (deepcopy(banked[max(banked)]) if banked else make_prompt_cache(model.language_model), [size - 9, size - 1])
    resumed = GenerationRequest(prompt = texts[0], max_tokens = 24, prompt_cache_state = state)
    greedy = [GenerationRequest(prompt = texts[0], max_tokens = 40), GenerationRequest(prompt = texts[1], max_tokens = 200)]
    sampled = GenerationRequest(prompt = texts[1], max_tokens = 32, sampling = SamplingParams(temperature = 0.8, top_k = 20, seed = 7))
    reference = lambda request: list(ar.generate_step(mx.array([tok.encode(request.prompt, add_special_tokens = False)]), model, None, None,
                                                      max_tokens = request.max_tokens, temperature = 0.0, logits_processors = request.logits_processors))
    upstream = [reference(request) for request in greedy]
    # History-free, so mlx-vlm applies it as the batch does: it bans the second reply's greedy opening.
    bias = mx.zeros_like(upstream[1][0][1]).at[upstream[1][0][0]].add(-1e9)
    greedy.append(GenerationRequest(prompt = texts[1], max_tokens = 24, logits_processors = [lambda tokens, logits: logits + bias]))
    upstream.append(reference(greedy[2]))

    def run(schedule):
        # Rounds only: batched plain windows carry ordinary batched numerics.
        drafter = companion and ContextDrafter(_tiny_companion(model, companion), model.language_model)
        with BatchStream(model, None, speculative = SpeculativeDraft(_script(("draft", 3), ("copy", 4)) if drafter else _rounds_only(), drafter)) as stream:
            rows, results, step = {}, {}, 0
            while step <= max(schedule) or stream.rows_in_flight:
                if step in schedule:
                    rows[stream.add(schedule[step])] = step
                results.update({rows[event.index]: event.result for event in stream.step() if event.result is not None})
                step += 1
        return results

    batched, alone = run({0: greedy[0], 3: greedy[1], 6: sampled, 7: greedy[2], 8: resumed}), run({0: sampled})
    for request, result, expected in zip(greedy, (batched[0], batched[3], batched[7]), upstream):
        n = len(result.token_ids)
        # A stop token ends the reply without joining its text.
        assert result.token_ids == [token for token, _ in expected[:n]] and (n == request.max_tokens or expected[n][0] in tok.stopping_criteria.eos_token_ids)
        assert result.logprobs == pytest.approx([logprobs[token].item() for token, logprobs in expected[:n]], abs = 0.02)
    assert (batched[6].token_ids, batched[6].logprobs) == (alone[0].token_ids, alone[0].logprobs)
    assert [batched[0].finish_reason, batched[3].finish_reason] == ["length", "stop"]
    assert batched[0].draft_tokens >= batched[0].accepted_draft_tokens > 0
    assert (batched[8].cached_token_count, sorted(banked)) == (0, [size - 9, size - 1])
    del banked[size - 1]
    warm = run({0: resumed})[0]
    assert (warm.cached_token_count, sorted(banked), warm.token_ids) == (size - 9, [size - 9, size - 1], batched[8].token_ids) and warm.draft_tokens > 0
    # Rows that join mid-flight draft from the features their own prefill captured.
    assert not companion or all(batched[row].draft_tokens > 0 for row in (3, 6))



def test_speculative_batch_stream_leaves_no_row_behind_when_admission_fails(qwen, monkeypatch):
    from dataclasses import replace
    from unsloth_zoo.mlx import generate
    from unsloth_zoo.mlx.speculative import DraftController, SpeculativeDraft
    model, request = qwen[0], generate.GenerationRequest(prompt = "Name three colours.", max_tokens = 4)
    call = type(model.language_model).__call__
    monkeypatch.setattr(model, "_is_vlm_model", True, raising = False)
    with generate.BatchStream(model, None, speculative = SpeculativeDraft(_rounds_only())) as stream:
        with monkeypatch.context() as patch:
            patch.setattr(generate, "_new_detokenizer", lambda *args, **kwargs: 1 / 0)
            pytest.raises(ZeroDivisionError, stream.add, request)
        with monkeypatch.context() as patch:
            # Encoder-decoder models hand encoder state to every later forward.
            patch.setattr(type(model.language_model), "__call__", lambda self, *args, **kwargs: replace(call(self, *args, **kwargs), encoder_outputs = mx.zeros(1)))
            pytest.raises(generate.BatchRowRefused, stream.add, request)
        assert (stream.rows_in_flight, stream._session.engine.rows, stream._session.usable) == (0, [], True)
        assert stream.add(request) is not None and next(events := stream.iter_step()) and (owner := stream._owner) and setattr(stream, "_owner", None) is None and pytest.raises(RuntimeError, next, events) and setattr(stream, "_owner", owner) is None
    with generate.BatchStream(model, None, speculative = SpeculativeDraft(DraftController(max_depth = 0, can_copy = False))) as stream:
        row, events = stream.add(replace(request, max_tokens = 12)), stream.iter_step()
        assert next(events).index == row and stream._session.engine._rows[0].emitted <= 3
        pytest.raises(RuntimeError, stream.withdraw, row)
        assert len(list(events)) >= 8 and len(stream.withdraw(row).token_ids) >= 9
        assert stream.add(replace(request, max_tokens = 12)) is not None and (state := stream._session.engine._rows[0]) and next(events := stream.iter_step()) and (before := state.emitted) and stream.close() is None and (list(events), state.emitted) == ([], before)


def _tiny_companion(model, repo):
    from mlx_vlm.speculative.drafters import dflash2, dspark, qwen3_dflash
    from mlx_vlm.utils import load_config
    config, n = load_config(repo), (text := model.language_model.config.text_config).num_hidden_layers
    config.update(hidden_size = text.hidden_size, vocab_size = text.vocab_size, num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2,
                  head_dim = 64, intermediate_size = 256, num_target_layers = n, layer_types = config.get("layer_types", [])[:2])
    config["dflash_config"].update(target_layer_ids = [1, n // 2, n - 2], num_target_layers = n, selector_rank = 16, markov_rank = 16)
    module = {"DFlash2DraftModel": dflash2, "DSparkDraftModel": dspark}.get(config["architectures"][0], qwen3_dflash)
    return module.Model(module.ModelConfig.from_dict(config))


@pytest.mark.parametrize("repo", ["z-lab/Qwen3.5-4B-DFlash", "incoai/Qwen3.8-27B-DFlash2", "RadixArk/Qwen3.8-27B-DSpark"])
def test_dflash_family_drafts_from_every_committed_feature_alone_batched_and_through_the_seam(qwen, repo, monkeypatch):
    from unsloth_zoo.mlx import generate, speculative
    (model, ids), ar, SamplingParams = qwen, __import__("mlx_vlm.generate.ar", fromlist = ["ar"]), generate.SamplingParams
    drafter, blocks, rows, sequences = speculative.ContextDrafter(_tiny_companion(model, repo), model.language_model, max_lag = 8), [], [], []
    block, start = drafter._block, drafter.start
    drafter._block = lambda row, depth: blocks.append((row, row.pending, mx.concatenate(row.features), got := block(row, depth))) or got
    drafter.start = lambda prompt, hidden, pending: rows.append((started := start(prompt, hidden, pending))[0]) or started
    # Only the lone row mixes in plain windows: batched ones carry ordinary batched numerics.
    for prompts, script in ((ids[:1], (("draft", 3), ("plain", 9), ("copy", 4), ("draft", 5))), (ids[:2], (("draft", 3), ("copy", 4), ("draft", 5)))):
        out, drafted, _ = _run(model, prompts, 48, _script(*script), SamplingParams(), drafter = drafter)
        assert out == [_solo(model, prompt, 48, SamplingParams()) for prompt in prompts] and all(n for n, _ in drafted.values())
        sequences += [[*prompt, *tokens] for prompt, tokens in zip(prompts, out)]
    [monkeypatch.setattr(ar, name, getattr(ar, name)) for name in ("run_speculative_rounds", "SpeculativePrefill")]
    speculative.install_speculative_seam()
    (draft := speculative.SpeculativeDraft(_script(("draft", 4)), drafter)).prepare(ids[0], SamplingParams())
    step = lambda **kwargs: [*ids[0], *(int(token) for token, _ in ar.generate_step(
        mx.array([ids[0]]), model, None, None, max_tokens = 48, temperature = 0.0, prefill_step_size = 16, **kwargs))]
    sequences.append(step(draft_model = draft, draft_kind = draft.draft_kind))
    # Against plain decoding chunked alike: chunking alone can flip a bf16 near-tie on Metal.
    assert sequences[-1] == step() and start(ids[0], None, 0) == (None, [])
    # Contexts run on from the prompt's first position (six prefill chunks); anchors follow; drafts equal a fresh drafter's.
    for row, sequence in zip(rows, sequences, strict = True):
        reference, fed, fresh = drafter.features(model.language_model(mx.array([sequence]), **drafter.capture))[0], 0, drafter.model.make_cache()
        for pending, context, got in [block[1:] for block in blocks if block[0] is row]:
            fed += len(context)
            assert pending == sequence[fed] and (context - reference[fed - len(context) : fed]).abs().mean().item() < 0.5
            assert mx.array_equal(got, drafter.model.draft_block(pending, context[None], fresh, got.shape[-1] + 1, lambda logits: mx.argmax(logits, axis = -1)))
    assert _probe_matches_a_started_row(model, speculative.ContextDrafter(drafter.model, model.language_model, max_lag = 8), ids[0][:40], lambda started: started[0].cache)

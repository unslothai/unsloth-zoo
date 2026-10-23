# Unsloth Zoo - Utilities for Unsloth
# Pin MLXTrainer's batch padding to match mlx-lm's iterate_batches:
# `1 + _PAD_MULTIPLE * ceil(L / _PAD_MULTIPLE)`.
#
# mlx-lm (mlx_lm/tuner/trainer.py:158) pads to `1 + 32*ceil(max_len/32)`;
# default_loss then slices [:, :-1] / [:, 1:]. unsloth_zoo's
# create_text_batches previously dropped the `+1`, leaving the input one
# token short after the autoregressive shift, which shifted small fixtures
# into a different convergence basin (probe 31 = 67% vs probe 33-37 = 40-53%,
# see danielhanchen/unsloth-staging-2).

import contextlib
from types import SimpleNamespace

import pytest

try:
    import mlx.core as mx
except Exception:
    pytest.skip("requires mlx", allow_module_level = True)
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason = "Requires Metal")

MODEL = "mlx-community/Qwen3.5-0.8B-bf16"
QUANTIZED = "mlx-community/Qwen3.5-0.8B-4bit"
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
    from unsloth_zoo.mlx.loader import _ensure_exact_verify_sums_promoted, _ensure_verify_qk_norm_matches_decode
    _ensure_exact_verify_sums_promoted()  # as FastMLXModel's VLM load does
    _ensure_verify_qk_norm_matches_decode()
    model, processor = load(repo)
    model._processor, tok = processor, processor.tokenizer
    ids = [
        tok.encode(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt = True, tokenize = False), add_special_tokens = False)
        for p in [*PROMPTS, *extra]
    ]
    return model, ids


# Every mlx-vlm 0.7.8 kernel source that copies MLX's qmv input sum.
EXACT_VERIFY_SUMS = [("mlx_vlm.models.quantized_verifier", "_AFFINE_MOE_HC_HEADER"),
                     ("mlx_vlm.models.quantized_verifier", "_target_verify_qlinear_header"),
                     ("mlx_vlm.models.fast_ops", "_AFFINE5_SWITCH_HEADER"), ("mlx_vlm.models.fast_ops", "_AFFINE4_SWITCH_HEADER"),
                     ("mlx_vlm.models.glm_moe_dsa.speculative_verifier", "_quantized_head_header")]


def _verify_sources(monkeypatch):
    """The listed sources as they stand, restored after the test, with promotion not yet done."""
    import importlib
    from unsloth_zoo.mlx import loader
    monkeypatch.setattr(loader, "_exact_verify_sums_promoted", False)
    modules = {module: importlib.import_module(module) for module, _ in EXACT_VERIFY_SUMS}
    before = {(module, name): getattr(modules[module], name) for module, name in EXACT_VERIFY_SUMS}
    for (module, name), source in before.items():
        monkeypatch.setattr(modules[module], name, source)
    return modules, before


def _assert_promoted(modules, before, promoted):
    for (module, name), source in before.items():
        after = getattr(modules[module], name)
        if not promoted:  # MLX before 0.32.4 adds in the input dtype, as the kernels do
            assert after is source, name
        else:
            text = after if isinstance(after, str) else after(4, 64)
            assert "sum += x[i] + x[i + 1]" not in text and "sum += float(x[i]) + float(x[i + 1])" in text, name


# First in the file: the model fixtures promote the sums for the rest of the session.
@pytest.mark.parametrize("version", ["0.32.3", "0.32.4"])
def test_exact_verify_kernels_sum_their_input_as_the_installed_mlx_decode_does(monkeypatch, version):
    from unsloth_zoo.mlx import loader
    modules, before = _verify_sources(monkeypatch)
    monkeypatch.setattr(mx, "__version__", version)
    built = modules["mlx_vlm.models.quantized_verifier"]._target_verify_qmv_kernel
    built(4, 64, mx.bfloat16, 2, 64, 64)
    loader._ensure_exact_verify_sums_promoted()
    assert (built.cache_info().currsize > 0) == (version == "0.32.3")  # a kernel built earlier keeps its sums
    _assert_promoted(modules, before, version == "0.32.4")


def test_a_fast_mlx_model_load_promotes_the_verify_sums(monkeypatch):
    from packaging.version import Version
    from unsloth_zoo.mlx import loader
    from unsloth_zoo.mlx.loader import FastMLXModel
    modules, before = _verify_sources(monkeypatch)
    repointed, repoint = [], loader._ensure_verify_qk_norm_matches_decode
    monkeypatch.setattr(loader, "_ensure_verify_qk_norm_matches_decode", lambda: repointed.append(1) or repoint())
    FastMLXModel.from_pretrained(QUANTIZED)
    _assert_promoted(modules, before, Version(mx.__version__) >= Version("0.32.4"))
    assert repointed


@pytest.fixture(scope = "module")
def qwen():
    return _load(MODEL)


@pytest.fixture(scope = "module")
def qwen_mtp():
    return _load(MTP_MODEL)


@pytest.fixture(scope = "module")
def qwen_4bit():
    from mlx_vlm.models.cache import make_prompt_cache
    model, ids = _load(QUANTIZED)
    cache = make_prompt_cache(model.language_model)
    mx.eval(model.language_model(mx.array([ids[0][:24]]), cache = cache).logits)
    return model, ids, cache


@pytest.fixture(scope = "module")
def qwen4():
    """A small random qwen4_exp quantized like the 4-bit releases, so its decode takes the batch-invariant forward, with
    their expert and hyper-connection widths, which pick the projections' kernels."""
    import mlx.nn as nn
    from mlx.utils import tree_map
    from mlx_vlm.models import qwen4_exp
    from unsloth_zoo.mlx.loader import _ensure_exact_verify_sums_promoted
    _ensure_exact_verify_sums_promoted()  # as FastMLXModel's VLM load does
    text = dict(model_type = "qwen4_exp_text", hidden_size = 512, num_hidden_layers = 4, num_attention_heads = 4, num_key_value_heads = 2,
                head_dim = 128, linear_num_key_heads = 2, linear_num_value_heads = 6, linear_key_head_dim = 128, linear_value_head_dim = 128,
                linear_conv_kernel_dim = 4, num_experts = 64, num_experts_per_tok = 10, moe_intermediate_size = 640,
                shared_expert_intermediate_size = 640, hc_lowrank = 320, rms_norm_eps = 1e-6, vocab_size = 1024, max_position_embeddings = 4096,
                ple_layer_ids = [2], ngram_vocab_size_base = 4096, split_ngram_parts = 4, eos_token_id = 1)
    vision = dict(depth = 1, hidden_size = 64, num_heads = 2, intermediate_size = 128, out_hidden_size = 512)
    model = qwen4_exp.Model(qwen4_exp.ModelConfig.from_dict(dict(model_type = "qwen4_exp", text_config = text, vision_config = vision)))
    # Constructors zero gates and residual scales, which would cut the paths under test.
    model.update(tree_map(lambda a: mx.random.normal(a.shape, key = mx.random.key(a.size)) * 0.05
                          if mx.issubdtype(a.dtype, mx.floating) and not mx.any(a).item() else a, model.parameters()))
    model.set_dtype(mx.bfloat16)
    gates = {"group_size": 64, "bits": 8, "mode": "affine"}
    nn.quantize(model.language_model, group_size = 32, bits = 4, class_predicate = lambda path, module: hasattr(module, "to_quantized") and (
        gates if path.endswith(("mlp.gate", "shared_expert_gate")) else True))
    model.eval()
    mx.eval(model.parameters())
    assert model.language_model._supports_batch_invariant_decode()
    ids = [[(7 * i + 3) % 1000 + 2 for i in range(29)] * 2, [(11 * i + 5) % 1000 + 2 for i in range(23)]]
    return model, ids


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


def _clone(cache):
    import copy
    from mlx.utils import tree_map
    copies = [copy.copy(entry) for entry in cache]
    for entry in copies:
        entry.__dict__.update(tree_map(lambda leaf: mx.array(leaf) if isinstance(leaf, mx.array) else leaf, vars(entry)))
    return copies


def _state(out, cache):
    """Output tensors, then every cache entry's state and bookkeeping."""
    from mlx.utils import tree_flatten
    found = tree_flatten((out.logits, out.hidden_states))
    for i, entry in enumerate(cache):
        found += [(f"{i}.{key}", leaf) for key, leaf in tree_flatten(entry.state)]
        found += [(f"{i}.{name}", getattr(entry, name, None)) for name in ("meta_state", "lengths", "left_padding")]
    return found


def _same(a, b):
    return [key for key, _ in a] == [key for key, _ in b] and all(
        (x.dtype == y.dtype and x.shape == y.shape and mx.array_equal(x.view(mx.uint8), y.view(mx.uint8)).item())
        if isinstance(x, mx.array) else x == y for (_, x), (_, y) in zip(a, b))


def _captured(lm):
    return dict(capture_layer_ids = [0, len(lm.layers) // 2], return_hidden = True, rope_deltas = mx.array([[3]]))


def _verified(lm, cache, block, keep):
    """A verify block on a copy of `cache`, aborted or committed at `keep` rows."""
    run = _clone(cache)
    out = lm(block, cache = run, speculative_verify = True, **_captured(lm))
    out.gdn_states.commit(keep) if keep else out.gdn_states.abort()
    return _state(out, run)


def _serving(monkeypatch):
    """Keys whose replacement ran, in order."""
    from unsloth_zoo.mlx import inference
    served, admitted = [], inference._verify_admitted
    monkeypatch.setattr(inference, "_verify_admitted", lambda key, native, fused: admitted(key, native, lambda: served.append(key) or fused()))
    inference._VERIFY_VERDICTS.clear()
    return served


def test_verify_admission_serves_a_key_only_after_its_first_block_matched_bit_for_bit(monkeypatch):
    from unsloth_zoo.mlx import inference
    monkeypatch.setattr(inference, "_VERIFY_VERDICTS", {})
    admit, (a, b) = inference._verify_admitted, (mx.ones((1, 2), dtype = mx.float16) for _ in range(2))
    zero, minus = mx.array([0.0], dtype = mx.float16), mx.array([-0.0], dtype = mx.float16)
    assert admit("one", lambda: a, lambda: b) is a and admit("one", lambda: a, lambda: b) is b
    assert admit("pair", lambda: (a, zero), lambda: (b, zero))[0] is a and admit("pair", lambda: (a, zero), lambda: (b, zero))[0] is b
    # The second output differs in its sign bit only.
    assert all(admit("signed", lambda: (a, zero), lambda: (b, minus))[0] is a for _ in range(2))
    assert admit("refused", lambda: a, lambda: None) is a and admit("one", lambda: a, lambda: None) is a
    assert inference._VERIFY_VERDICTS == {"one": True, "pair": True, "signed": False}


def test_wide_verify_blocks_are_cut_into_balanced_slices_within_the_fused_bound(monkeypatch):
    import mlx.nn as nn
    from unsloth_zoo.mlx import inference
    monkeypatch.setattr(inference, "_VERIFY_VERDICTS", {})
    monkeypatch.setattr(inference._VERIFY_STATE, "serving", True, raising = False)
    linears, widths = (nn.QuantizedLinear(64, 8, bias = False),) * 2, []

    def cut(width, eligible = True):
        x = mx.arange(2 * width * 4).reshape(2, width, 4).astype(mx.float16)
        widths.clear()
        for _ in range(2):
            out = inference._verify_sliced(lambda rows: widths.append(rows.shape[1]) or (rows, rows * 2), linears, x, lambda linear, rows: eligible)
        assert mx.array_equal(out[0], x).item() and mx.array_equal(out[1], x * 2).item()
        return list(widths)
    assert cut(17) == [17, 6, 6, 5, 6, 6, 5] and cut(9) == [9, 5, 4, 5, 4] and cut(16) == [16, 8, 8, 8, 8]
    assert cut(8) == [8, 8] and cut(17, eligible = False) == [17, 17]


@pytest.mark.parametrize("width", [9, 17])
def test_wide_verify_blocks_in_slices_match_the_native_verifier(qwen_4bit, monkeypatch, width):
    from unsloth_zoo.mlx import inference
    monkeypatch.setenv("UNSLOTH_MLX_VERIFY_QMV_ROWS", "0")  # slices serve the layouts `matmul.qmv_rows` does not cover
    (model, ids, cache), served = qwen_4bit, _serving(monkeypatch)
    lm, block = model.language_model, mx.array([ids[0][24 : 24 + width]])
    first = -(-width // -(-width // 8))
    native = {keep: _verified(lm, cache, block, keep) for keep in (0, 1, first, first + 1, width)}
    with inference.exact_speculative_verify(model):
        assert _same(native[0], _verified(lm, cache, block, 0))
        verdicts = [verdict for key, verdict in inference._VERIFY_VERDICTS.items() if key[0] == (1, width)]
        assert verdicts and all(verdicts) and not served.clear()
        assert all(_same(expected, _verified(lm, cache, block, keep)) for keep, expected in native.items())
    assert len(served) >= len(native) * len(verdicts)


def test_a_sliced_verify_block_matches_one_token_decode_steps(qwen_4bit, monkeypatch):
    from unsloth_zoo.mlx import inference
    monkeypatch.setenv("UNSLOTH_MLX_VERIFY_QMV_ROWS", "0")
    (model, ids, cache), served = qwen_4bit, _serving(monkeypatch)
    lm, block = model.language_model, mx.array([ids[0][24:39]])
    steps, snapshots, decoded = _clone(cache), {0: cache}, []
    for t in range(block.shape[1]):
        decoded.append(lm(block[:, t : t + 1], cache = steps, **_captured(lm)))
        if t + 1 in (1, 8, 15):
            snapshots[t + 1] = _clone(steps)
    stepped = SimpleNamespace(logits = mx.concatenate([out.logits for out in decoded], axis = 1), hidden_states = [mx.concatenate(parts, axis = 1) for parts in zip(*(out.hidden_states for out in decoded))])
    with inference.exact_speculative_verify(model):
        _verified(lm, cache, block, 0)
        assert not served.clear() and all(_same(_state(stepped, entries), _verified(lm, cache, block, keep)) for keep, entries in snapshots.items())
    assert served


@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_a_verify_block_runs_its_rows_together_and_matches_one_token_decode_steps(monkeypatch, bits):
    import mlx.nn as nn
    from mlx_vlm.models.cache import make_prompt_cache
    from unsloth_zoo.mlx import inference, matmul, nax
    model, ids = _load(MODEL)
    lm = model.language_model
    nn.quantize(lm, group_size = 64, bits = bits)
    cache = make_prompt_cache(lm)
    mx.eval(lm(mx.array([ids[0][:24]]), cache = cache).logits)
    block, calls = mx.array([ids[0][24:35]]), []
    kernel = matmul.qmv_rows
    kind = lambda inputs, staged: "prepared" if inputs is not None else "staged" if staged else "inline"
    monkeypatch.setattr(matmul, "qmv_rows", lambda x, *args: calls.append((*x.shape, kind(*args[5:]))) or kernel(x, *args))
    matmul.compile_qmv_rows.cache_clear()
    # Launches are timed once per kernel; here staged launches are cheapest up to 16 rows, or prepared ones up to 4.
    timed, cheaper = [], "staged" if bits in (4, 6) else "prepared"
    costs = _spilling_costs(16, staged = 0.8) if cheaper == "staged" else _spilling_costs(4)
    monkeypatch.setattr(inference, "_QMV_ROWS_COSTS", {})
    monkeypatch.setattr(inference, "_QMV_ROWS_LAYOUTS", {})
    monkeypatch.setattr(matmul, "qmv_rows_launch_costs", lambda *kernel: timed.append(kernel) or costs)
    steps, snapshots, decoded = _clone(cache), {0: cache}, []
    for t in range(block.shape[1]):
        decoded.append(lm(block[:, t : t + 1], cache = steps, **_captured(lm)))
        if t + 1 in (1, 6, 11):
            snapshots[t + 1] = _clone(steps)
    stepped = SimpleNamespace(logits = mx.concatenate([out.logits for out in decoded], axis = 1), hidden_states = [mx.concatenate(parts, axis = 1) for parts in zip(*(out.hidden_states for out in decoded))])
    projections = [p for _, p in lm.named_modules() if isinstance(p, nn.QuantizedLinear)]
    model.generate, model._unsloth_speculates = lambda *args, **kwargs: None, True  # marked as building a drafter does
    # Settled at load, outside any verify scope: every row count built, launches timed once, each layout checked.
    inference.warm_inference(model)
    with inference.exact_speculative_verify(model):
        step = 512 if bits in (4, 5) else 256
        assert {M for M, K, _ in calls if K == step} == set(range(1, 17)) and timed == [(mx.bfloat16, 64, bits)]
        supported, support = [], matmul.qmv_rows_supported
        monkeypatch.setattr(matmul, "qmv_rows_supported", lambda *args: supported.append(args) or support(*args))
        calls.clear()
        assert all(_same(_state(stepped, entries), _verified(lm, cache, block, keep)) for keep, entries in snapshots.items())
        assert {(M, launched) for M, K, launched in calls} == {(M, cheaper) for M in ({11} if cheaper == "staged" else {4, 3})}  # 11 or 4 + 4 + 3
        _verified(lm, cache, block[:, :5], 0)
        assert (5 if cheaper == "staged" else 2, lm.args.hidden_size, cheaper) in calls and not supported
        assert timed == [(mx.bfloat16, 64, bits)]
        probe, recorded = nax.kernel_probe_passed, matmul.qmv_rows  # replaced scales settle again, natively if these fail
        fresh = lambda: [setattr(p, "scales", mx.array(p.scales)) for p in projections]
        for failed in ((nax, "kernel_probe_passed", lambda key, *args: not key.startswith(matmul.QMV_ROWS_PROBE_KEY) and probe(key, *args)),
                       (matmul, "qmv_rows", lambda x, *args: recorded(x, *args) + (kind(*args[5:]) == "staged"))):
            with monkeypatch.context() as patch:
                patch.setattr(inference, "_QMV_ROWS_LAYOUTS", {})
                patch.setattr(*failed)
                fresh()
                inference._warm_qmv_rows(model)
                calls.clear()
                _verified(lm, cache, block, 0)
                assert not calls and all(vars(p)["_unsloth_qmv_rows"][2] is None for p in projections)
        fresh()  # without a warm-up the first block settles them, unless the kill switch is set
        count, checked = len(calls), len(supported)
        monkeypatch.setenv("UNSLOTH_MLX_VERIFY_QMV_ROWS", "0")
        inference.warm_inference(model)
        _verified(lm, cache, block, 0)
        assert len(calls) == count and len(supported) == checked
        monkeypatch.delenv("UNSLOTH_MLX_VERIFY_QMV_ROWS")
        _verified(lm, cache, block, 0)
        assert len(calls) > count


def _spilling_costs(most, prepared = 0.9, staged = 1):  # one unit per launch and per row; launches over `most` rows spill
    per_row = {"inline": 1, "prepared": prepared, "staged": staged}
    return {(m, kind): 1 + m * per_row[kind] if m <= most else 100 for kind in per_row
            for m in range(1, 17 if kind == "staged" else 9)}


def test_each_verify_rows_kernel_has_its_own_probe_building_only_that_kernel(monkeypatch):
    import itertools
    from unsloth_zoo.mlx import matmul
    probed = []
    monkeypatch.setattr(matmul, "_probe_qmv_rows", lambda *kernel: probed.append(kernel))
    kernels = list(itertools.product((mx.bfloat16, mx.float16), (32, 64, 128), (4, 5, 6, 8)))
    keys = set()
    for kernel in kernels:
        key, function = matmul.qmv_rows_probe(*kernel)
        keys.add(key)
        getattr(matmul, function)()
        assert key.startswith(matmul.QMV_ROWS_PROBE_KEY + ":") and probed[-1] == kernel
    assert len(keys) == len(kernels) == len(probed)
    for name in ("probe_qmv_rows_bfloat16_64_3", "probe_qmv_rows_float32_64_4", "probe_qmv_rows_float16_48_4"):
        with pytest.raises(AttributeError):
            getattr(matmul, name)


def test_verify_rows_serve_only_projections_whose_one_row_call_is_stock(monkeypatch):
    import mlx.nn as nn
    from unsloth_zoo.mlx import inference, matmul
    monkeypatch.setattr(inference, "_QMV_ROWS_COSTS", {})
    monkeypatch.setattr(inference, "_QMV_ROWS_LAYOUTS", {})
    monkeypatch.setattr(matmul, "qmv_rows_launch_costs", lambda *kernel: _spilling_costs(8))
    supported, support = [], matmul.qmv_rows_supported
    monkeypatch.setattr(matmul, "qmv_rows_supported", lambda *args: supported.append(args) or support(*args))
    linear = nn.Linear(512, 64, bias = False)
    linear.set_dtype(mx.bfloat16)
    p = nn.QuantizedLinear.from_linear(linear, group_size = 64, bits = 4)
    assert inference._qmv_rows_kernel(p) == (mx.bfloat16, 64, 4) and len(supported) == 1
    # A wrapper zoo installs during generation, stock on one row, keeps the verdict settled at load.
    stock_call = lambda self, x: nn.QuantizedLinear.__call__(self, x)
    monkeypatch.setattr(inference, "_ONE_ROW_STOCK_CALLS", {stock_call})
    p.__class__ = type("Wrapped", (nn.QuantizedLinear,), {"__call__": stock_call})
    assert inference._qmv_rows_kernel(p) == (mx.bfloat16, 64, 4) and len(supported) == 1
    # A subclass whose call changes the projection is never bypassed.
    p.__class__ = type("Scaled", (nn.QuantizedLinear,), {"__call__": lambda self, x: nn.QuantizedLinear.__call__(self, x) * 2})
    assert inference._qmv_rows_kernel(p) is None
    p.__class__ = nn.QuantizedLinear
    assert inference._qmv_rows_kernel(p) == (mx.bfloat16, 64, 4)


def test_verify_rows_launch_the_plan_their_launch_costs_make_cheapest():
    from unsloth_zoo.mlx import matmul
    plan = lambda costs, rows: matmul.qmv_rows_plan(tuple(costs.items()), rows)
    # As few launches as fit under the spill, balanced, with the cheaper inputs.
    assert plan(_spilling_costs(6), 16) == (6, "prepared") and matmul.qmv_rows_sizes(16, 6) == [6, 5, 5]
    assert plan(_spilling_costs(8, prepared = 1.1), 16) == (8, "inline")
    assert plan(_spilling_costs(8, staged = 0.8), 16) == (8, "staged")
    assert plan(_spilling_costs(16, staged = 0.8), 16) == (16, "staged")
    assert plan(_spilling_costs(4), 3) in {(3, "prepared"), (4, "prepared")}


def test_verify_row_launches_are_costed_by_the_median_of_three_timed_runs(monkeypatch):
    from unsloth_zoo.mlx import matmul
    now, spent, runs = [0.0], [0.0], {}

    def clock():
        now[0], spent[0] = now[0] + spent[0], 0.0
        return now[0]

    def launch(x, w, scales, biases, group_size, bits, inputs, staged):
        rows, prepared = x.shape[0], inputs is not None
        durations = (0, rows, 50, rows + 0.5 * prepared + 0.25 * staged)
        spent[0] += next(runs.setdefault((rows, prepared, staged), iter([d for d in durations for _ in range(4)])))
        return x
    monkeypatch.setattr(matmul, "qmv_rows", launch)
    monkeypatch.setattr(matmul, "qmv_rows_inputs", lambda x, bits: x)
    # A warm run, then three timed runs of four launches each, every launch advancing the clock by its own time.
    costs = matmul.qmv_rows_launch_costs(mx.bfloat16, 64, 4, clock)
    extra = {"inline": 0, "prepared": 0.5, "staged": 0.25}
    assert costs == {(m, kind): m + extra[kind] for kind in extra for m in range(1, 17 if kind == "staged" else 9)}


def test_the_verify_scope_serves_only_its_own_models_and_restores_the_verifier(qwen_4bit, monkeypatch):
    import copy
    import mlx.nn as nn
    from mlx_vlm.models.qwen3_5 import language, speculative_verifier
    from mlx_vlm.speculative.ops import linear as linear_ops
    from unsloth_zoo.mlx import inference
    (model, ids, cache), verdicts = qwen_4bit, inference._VERIFY_VERDICTS
    monkeypatch.setenv("UNSLOTH_MLX_VERIFY_QMV_ROWS", "0")
    lm, verifier, native = model.language_model, language._EXACT_SPECULATIVE_VERIFIER, speculative_verifier.Qwen3_5BatchInvariantForward
    unscoped, block = copy.copy(lm), mx.array([ids[0][24:36]])
    quantized = [module for _, module in lm.named_modules() if isinstance(module, nn.QuantizedLinear)]
    verdicts.clear()
    with inference.exact_speculative_verify(model):
        with pytest.raises(ZeroDivisionError), inference.exact_speculative_verify(model):
            assert type(verifier) is not native and (lm._unsloth_verify_scopes, verifier._unsloth_verify_scopes) == (2, 2)
            1 / 0
        assert type(verifier) is not native and lm._unsloth_verify_scopes == 1
        layer, pending = lm.model.layers[0], [block[:, :11]]
        norm = layer.input_layernorm

        def nested(x):
            if pending:
                _verified(unscoped, cache, pending.pop(), 0)
            return norm(x)
        with monkeypatch.context() as patch:
            # An unscoped model's verify call made inside a served one stays native, and the served one goes on.
            patch.setattr(layer, "input_layernorm", nested)
            assert _verified(lm, cache, block, 0) and {key[0][1] for key in verdicts} == {12} and not verdicts.clear()
        # Native: an unscoped model, a block inside mlx-vlm's fused bound, training, thawed projections, a drifted binding.
        _verified(unscoped, cache, block, 0)
        _verified(lm, cache, block[:, :8], 0)
        assert lm.train() and _verified(lm, cache, block, 0) and lm.eval()
        assert [module.unfreeze() for module in quantized] and _verified(lm, cache, block, 0) and [module.freeze() for module in quantized]
        with monkeypatch.context() as patch:
            patch.setattr(linear_ops, "_target_verify_quantized_linear", lambda linear, x: None)
            _verified(lm, cache, block, 0)
        assert not verdicts and _verified(lm, cache, block, 0) and verdicts
    assert type(verifier) is native and not any("_unsloth_verify_scopes" in vars(owner) for owner in (verifier, lm))
    with monkeypatch.context() as patch:
        patch.setattr(model, "_unsloth_mlx_distributed_parallel_mode", True, raising = False)
        with inference.exact_speculative_verify(model):
            assert type(verifier) is native
    for name in ("_linears", "_layer", "_feed_forward"):
        with monkeypatch.context() as patch:
            patch.setattr(native, name, lambda self, *args: None)
            with inference.exact_speculative_verify(model):
                assert type(verifier) is native
    # A drifted attention body leaves only the attention route native.
    with monkeypatch.context() as patch:
        patch.setattr(native, "_attention", lambda self, *args: None)
        with inference.exact_speculative_verify(model):
            assert type(verifier) is not native and "_attention" not in vars(type(verifier))


def _verify_call(lm, cache, block):
    run = _clone(cache)
    return run, lm(block, cache = run, speculative_verify = True, **_captured(lm))


@pytest.mark.parametrize("width", [3, 5])
def test_a_taped_verify_block_commits_what_stock_commits_for_every_row(qwen_4bit, monkeypatch, width):
    from mlx_vlm.models.cache import make_prompt_cache
    from unsloth_zoo.mlx import inference
    from unsloth_zoo.mlx.speculative import _join
    model, ids, _ = qwen_4bit
    lm, cache, fresh = model.language_model, None, make_prompt_cache(model.language_model)
    for row, prompt in enumerate([ids[0][:24], ids[0][:17], ids[0][:31]]):
        entries = make_prompt_cache(lm)
        mx.eval(lm(mx.array([prompt]), cache = entries).logits)
        cache = _join(cache, entries, row)
    block = mx.array([ids[0][start : start + width] for start in (24, 30, 36)])
    keeps = [[1, width, 1 + width // 2], [width] * 3, [1] * 3, [0, width, 1], [0] * 3]
    native = [_verified(lm, cache, block, keep) for keep in keeps]
    # A row with no recurrent state yet.
    native_fresh = [_verified(lm, fresh, block[:1], keep) for keep in ([0], [1])]
    monkeypatch.setattr(inference, "_TAPE_VERDICTS", {})
    with inference.exact_speculative_verify(model):
        _verified(lm, cache, block, 0)
        _verified(lm, fresh, block[:1], 0)
        run, out = _verify_call(lm, cache, block)
        assert len(out.gdn_states._unsloth_tapes) == sum(layer.is_linear for layer in lm.model.layers)
        # A refused commit keeps the tapes, and the retained lengths may come from any iterable.
        with pytest.raises(ValueError):
            out.gdn_states.commit(width + 1)
        out.gdn_states.commit(iter(keeps[0]))
        assert _same(native[0], _state(out, run))
        run, out = _verify_call(lm, cache, block[:, :2])
        assert "_unsloth_tapes" not in vars(out.gdn_states)
        out.gdn_states.abort()
        assert all(_same(expected, _verified(lm, cache, block, keep)) for keep, expected in zip(keeps, native))
        _, out = _verify_call(lm, fresh, block[:1])
        assert len(out.gdn_states._unsloth_tapes) == sum(layer.is_linear for layer in lm.model.layers)
        out.gdn_states.abort()
        assert all(_same(expected, _verified(lm, fresh, block[:1], keep)) for keep, expected in zip(([0], [1]), native_fresh))
        assert _same(native[2], _verified(lm, cache, block, 1))
        run, out = _verify_call(lm, cache, block)
        out.gdn_states.commit(mx.array(keeps[0]))
        assert _same(native[0], _state(out, run))
        assert set(inference._TAPE_VERDICTS.values()) == {True}
        # A replay that does not reproduce stock keeps that key on stock.
        inference._TAPE_VERDICTS.clear()
        monkeypatch.setattr(inference, "_replay_matches", lambda *args: False)
        work = inference.first_use_work()
        _verified(lm, cache, block, 0)
        run, out = _verify_call(lm, cache, block)
        assert "_unsloth_tapes" not in vars(out.gdn_states) and list(inference._TAPE_VERDICTS.values()) == [False]
        assert inference.first_use_work() > work
        out.gdn_states.commit(keeps[0])
        assert _same(native[0], _state(out, run))


def _recurrence(rows, steps, heads = 4, width = 128):
    q, k, v = (mx.random.normal((rows, steps, heads, width), key = mx.random.key(seed)).astype(mx.bfloat16) for seed in range(3))
    a, b = (mx.random.normal((rows, steps, heads), key = mx.random.key(seed)).astype(mx.bfloat16) for seed in (3, 4))
    return q, k, v, a, b, mx.zeros((heads,)), mx.ones((heads,))


def test_replay_admission_refuses_a_replay_one_bit_off_or_misreading_retained_lengths():
    from functools import partial
    from mlx_vlm.models.qwen3_5.gated_delta import gated_delta_update_with_states
    from unsloth_zoo.mlx.inference import _replay_matches
    rows, steps = 3, 4
    q, k, v, a, b, A_log, dt_bias = _recurrence(rows, steps)
    initial = mx.random.normal((rows, *v.shape[2:], k.shape[3]), key = mx.random.key(5))
    stock = gated_delta_update_with_states(q, k, v, a, b, A_log, dt_bias, initial, state_steps = steps - 1)
    replay = partial(gated_delta_update_with_states, q, k, v, a, b, A_log, dt_bias, initial, state_steps = 0)
    bits = stock[2].view(mx.uint32)
    nudged = mx.where(mx.arange(bits.size).reshape(bits.shape) == 7, bits + 1, bits).view(mx.float32)
    assert _replay_matches(stock, replay, None, initial)
    assert not _replay_matches((*stock[:2], nudged), replay, None, initial)
    assert not _replay_matches(stock, lambda mask: replay(mask = None), None, initial)
    # Rows all retaining the first row's length, or retaining one step where they retained none.
    assert not _replay_matches(stock, lambda mask: replay(mask = None if mask is None else mx.broadcast_to(mask[:1], mask.shape)), None, initial)
    assert not _replay_matches(stock, lambda mask: replay(mask = None if mask is None else mask | (mx.arange(steps) == 0)), None, initial)


def test_a_block_on_an_entry_state_of_another_batch_commits_what_stock_commits(monkeypatch):
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.qwen3_5 import gated_delta
    from mlx_vlm.speculative.cache_state import start_speculative_cache
    from unsloth_zoo.mlx import inference
    route = inference._taped_update(gated_delta.gated_delta_update, gated_delta.gated_delta_update_with_states)
    inputs = _recurrence(3, 4)

    def committed(tapes):
        entry = ArraysCache(2)
        entry[1] = mx.full((1, *inputs[2].shape[2:], inputs[1].shape[3]), 10.0)
        transaction = start_speculative_cache([entry], 4)
        monkeypatch.setattr(inference._VERIFY_STATE, "tapes", tapes, raising = False)
        route(*inputs, cache = entry)
        if tapes:
            transaction.__class__ = inference._taped_transaction_class(type(transaction))
            transaction._unsloth_tapes = tapes
        transaction.commit([0, 1, 4])
        return entry[1]
    monkeypatch.setattr(inference, "_TAPE_VERDICTS", {})
    stock, tapes = committed(None), []
    committed([])
    assert mx.array_equal(stock[0], mx.full(stock.shape[1:], 10.0)).item()
    assert mx.array_equal(stock.view(mx.uint8), committed(tapes).view(mx.uint8)).item() and not tapes
    assert not inference._TAPE_VERDICTS


def _submissions(monkeypatch):
    submitted, submit = [], mx.async_eval
    monkeypatch.setattr(mx, "async_eval", lambda *arrays: (submit(*arrays), submitted.append(arrays))[0])
    return submitted


def test_a_served_verify_call_submits_each_layer_as_it_is_built(qwen_4bit, monkeypatch):
    import copy
    from unsloth_zoo.mlx import inference
    model, ids, cache = qwen_4bit
    lm, block = model.language_model, mx.array([ids[0][24:29]])
    unscoped, native = copy.copy(lm), {keep: _verified(lm, cache, block, keep) for keep in (0, 3)}
    submitted = _submissions(monkeypatch)
    with inference.exact_speculative_verify(model):
        assert _verified(unscoped, cache, block, 0) and not submitted
        assert all(_same(expected, _verified(lm, cache, block, keep)) for keep, expected in native.items())
        assert len(submitted) == 2 * len(lm.model.layers) and not submitted.clear()
        # Each submission is that layer's own output: the captured layers sit at their own positions.
        out = lm(block, cache = _clone(cache), speculative_verify = True, **_captured(lm))
        assert [index for index, (layer,) in enumerate(submitted) if any(layer is hidden for hidden in out.hidden_states)] == _captured(lm)["capture_layer_ids"]
        # A function transformation cannot evaluate: the layers are built and nothing is submitted.
        assert mx.compile(lambda x: lm(x, cache = _clone(cache), speculative_verify = True).logits)(block).shape[1] == 5
    assert len(submitted) == len(lm.model.layers)


def test_engine_tokens_and_scores_are_unchanged_by_the_verify_scope_over_wide_ragged_rounds(qwen_4bit, monkeypatch):
    from unsloth_zoo.mlx import inference
    from unsloth_zoo.mlx.generate import SamplingParams
    (model, ids, _), _ = qwen_4bit, _serving(monkeypatch)
    ban = lambda history, logits: logits + mx.where(mx.arange(logits.shape[-1]) == 13, -1e9, 0)

    def run():
        scores, first_use, verified = [], [], []
        def scored(engine):
            engine.logprobs, step, record, verify = True, engine.step, engine.controller.record_round, engine._verify
            engine.step = lambda *args: (scores.append(out.logprobs) or out for out in step(*args))
            engine.controller.record_round = lambda *args, **kwargs: first_use.append(kwargs["first_use"]) or record(*args, **kwargs)
            engine._verify = lambda inputs: (work := inference.first_use_work(), out := verify(inputs), verified.append(inference.first_use_work() != work))[1]
        out, drafted, _ = _run(model, [ids[0], ids[1], ids[0]], 96, _script(("copy", 16)), SamplingParams(), processors = {2: [ban]}, patch = scored)
        return out, scores, drafted, (first_use, verified)
    closed, scores, drafted, _ = run()
    inference._VERIFY_VERDICTS.clear()
    # Projections earlier tests settled settle again, so the first blocks run their first-use checks.
    monkeypatch.setattr(inference, "_QMV_ROWS_LAYOUTS", {})
    for _, module in model.named_modules():
        vars(module).pop("_unsloth_qmv_rows", None)
    with inference.exact_speculative_verify(model):
        opened = run()
    assert opened[:2] == (closed, scores) and any(scores) and drafted[0][1] > 8 and len({accepted for _, accepted in drafted.values()}) > 1
    # Exactly the rounds whose verify ran a first-use check are flagged, so the controller leaves just those out of its costs.
    assert opened[3][0] == opened[3][1] and any(opened[3][0]) and not all(opened[3][0])


def _decoded(lm, prompt, steps, block):
    """`steps` one-token decode steps after `prompt`, then a verify block on a copy of that cache, and that copy's state."""
    from mlx_vlm.models.cache import make_prompt_cache
    cache = make_prompt_cache(lm)
    lm(mx.array([prompt]), cache = cache)
    outs = [lm(mx.array([[token]]), cache = cache, return_hidden = True) for token in steps]
    hidden, _, transaction, logits = lm.speculative_verify_logits(mx.array([block]), run := _clone(cache), lambda logits: logits)
    transaction.commit([len(block)])
    stepped = SimpleNamespace(logits = mx.concatenate([out.logits for out in outs], 1), hidden_states = [mx.concatenate([out.hidden_states[-1] for out in outs], 1)])
    return _state(stepped, cache) + _state(SimpleNamespace(logits = logits, hidden_states = [hidden]), run)


def test_qwen4_decode_steps_and_verify_blocks_run_the_routed_experts_bit_for_bit(qwen4, monkeypatch):
    from mlx_vlm.models.qwen4_exp import language
    from unsloth_zoo.mlx import inference
    (model, ids), served = qwen4, _serving(monkeypatch)
    lm, native = model.language_model, type(language._QWEN4_BATCH_INVARIANT_FORWARD)
    args = (lm, ids[0][:20], ids[0][20:26], ids[0][26:31])
    expected = _decoded(*args)
    # The scopes a plain, undrafted generation holds.
    with inference.inference_scopes(model, ()):
        assert type(language._QWEN4_BATCH_INVARIANT_FORWARD) is not native
        assert _same(expected, _decoded(*args))
    assert type(language._QWEN4_BATCH_INVARIANT_FORWARD) is native and "_unsloth_verify_scopes" not in vars(lm)
    # The first decode step and the verify block each matched on their first layer, then served every other one.
    admitted = [key for key, verdict in inference._VERIFY_VERDICTS.items() if verdict]
    assert {key[0] for key in admitted} == {(1, 1), (1, 5)} and all(served.count(key) == (6 if key[0] == (1, 1) else 1) * len(lm.layers) for key in admitted)
    # A drafted target settles its projections' multi-row kernels at load, through the batch-invariant forward too.
    projections = [p for _, p in lm.named_modules() if type(p).__name__ == "QuantizedLinear"]
    for p in projections:
        p.__dict__.pop("_unsloth_qmv_rows", None)
    monkeypatch.setattr(model, "generate", lambda *args, **kwargs: None, raising = False)
    monkeypatch.setattr(model, "_unsloth_speculates", True, raising = False)
    inference.warm_inference(model)
    tags = [vars(p).get("_unsloth_qmv_rows") for p in projections]
    assert projections and all(tags) and any(tag[2] for tag in tags)


def test_qwen4_rounds_verify_and_commit_ragged_rows_as_their_own_decode_steps(qwen4):
    from mlx_vlm.models.cache import make_prompt_cache
    from unsloth_zoo.mlx.generate import SamplingParams
    from unsloth_zoo.mlx.speculative import EngineRow, SpeculativeEngine
    (model, ids), greedy = qwen4, SamplingParams()
    lm, engine, stepped = model.language_model, SpeculativeEngine(model, _rounds_only()), []
    prompts, streams = [ids[0][:30], ids[1][:16], ids[0][:23]], [ids[0][30:44], ids[0][2:16], ids[0][23:37]]
    # Rows of different MRoPE deltas, as image prompts leave them; the last row's is the model's own afterwards.
    for uid, (prompt, stream, delta) in enumerate(zip(prompts, streams, (0, -4, 7))):
        cache = make_prompt_cache(lm)
        lm(mx.array([prompt]), cache = cache)
        engine.add(EngineRow(cache = _clone(cache), pending = stream[0], prompt = prompt, sampling = greedy, max_tokens = 32,
                             uid = uid, rope_delta = delta))
        outs = [lm(mx.array([[token]]), cache = cache, rope_deltas = mx.array([[delta]]), return_hidden = True) for token in stream]
        stepped.append((mx.concatenate([out.logits for out in outs], 1), mx.concatenate([out.hidden_states[-1] for out in outs], 1)))
    for row in engine._rows:
        row.draft = object()  # a drafting row, so verify returns the hidden states an MTP head reads
    assert engine._invariant and engine._batch_rounds
    # Blocks of each row's own stream, kept raggedly, so a rejected tail left in any cache shows in the next block.
    starts = [0, 0, 0]
    for keeps in ((5, 2, 3), (1, 5, 2), (4, 3, 1)):
        logits, hidden, transaction = engine._verify(mx.array([stream[start : start + 5] for stream, start in zip(streams, starts)]))
        assert all(_same([("logits", want_logits[:, start : start + 5]), ("hidden", want_hidden[:, start : start + 5])],
                         [("logits", logits[i : i + 1]), ("hidden", hidden[i : i + 1])])
                   for i, ((want_logits, want_hidden), start) in enumerate(zip(stepped, starts)))
        transaction.commit(list(keeps))
        for entry in engine.cache:  # as a round does after a ragged commit
            padding = getattr(entry, "left_padding", None)
            if isinstance(padding, mx.array) and not hasattr(entry, "metadata_revision"):
                entry.left_padding = padding + 0
        starts = [start + keep for start, keep in zip(starts, keeps)]
    assert not any("keys" in vars(entry) for entry in engine.cache)  # the commit's view of the QSA keys is gone


@pytest.mark.parametrize("heads", [(24, 4), (16, 2), (8, 2)])
def test_a_block_read_in_one_pass_over_the_kv_prefix_matches_one_row_sdpa_calls_bit_for_bit(heads, monkeypatch):
    from unsloth_zoo.mlx import inference
    gqa = heads[0] // heads[1]
    monkeypatch.setattr(inference, "_EXACT_ATTENTION_THREADS", {})
    first, second = inference._exact_attention_kernels()
    templates = set()
    monkeypatch.setattr(inference, "_exact_attention_kernels", lambda: (lambda **kw: templates.add(tuple(kw["template"])) or first(**kw), second))
    # Batched rows, a row count past one threadgroup's, and key counts on each side of MLX's block-count steps.
    cases = ((1, 3, 5000, mx.bfloat16), (2, 5, 9000, mx.float16), (1, 8, 40000, mx.bfloat16), (1, 13, 70000, mx.bfloat16), (1, 4, 70000, mx.bfloat16))
    for batch, width, keys, dtype in cases:
        blocks = inference._sdpa_vector_blocks(gqa, keys - width + 1)
        assert blocks == inference._sdpa_vector_blocks(gqa, keys)
        # Cache-shaped: a capacity-padded buffer cut to `keys`, and queries transposed from (B, L, H, D).
        k, v = (mx.random.normal((batch, heads[1], keys + 100, 256)).astype(dtype)[..., :keys, :] for _ in range(2))
        q = (2 * mx.random.normal((batch, width, heads[0], 256))).astype(dtype).transpose(0, 2, 1, 3)
        rows = mx.concatenate([mx.fast.scaled_dot_product_attention(
            q[:, :, i : i + 1], k[:, :, : keys - width + i + 1], v[:, :, : keys - width + i + 1], scale = 0.0625)
            for i in range(width)], axis = 2)
        try:
            out = inference._exact_attention_rows(q, k, v, 0.0625, blocks)
            mx.eval(out)
        except ValueError as error:
            # A device whose limit refuses the widest threadgroup (M1-class GPUs) launches fewer row pairs per threadgroup.
            assert inference._fit_threadgroup(error, dtype, gqa)
            out = inference._exact_attention_rows(q, k, v, 0.0625, blocks)
        assert mx.array_equal(out.view(mx.uint16), rows.view(mx.uint16)).item()
        # One row pair per threadgroup, the narrowest any limit leaves, reads the same bits.
        with monkeypatch.context() as patch:
            patch.setitem(inference._EXACT_ATTENTION_THREADS, (dtype, gqa), 32 * gqa)
            narrow = inference._exact_attention_rows(q, k, v, 0.0625, blocks)
            assert mx.array_equal(narrow.view(mx.uint16), rows.view(mx.uint16)).item()
    # One compile per layout serves every width: the row count is an input, not a template parameter.
    assert templates == {(("T", dtype), ("D", 256), ("G", gqa), ("BLOCKS", inference._sdpa_vector_blocks(gqa, keys - width + 1)))
                         for _, width, keys, dtype in cases}
    monkeypatch.undo()
    # Rows on both sides of a block-count step stay one-row calls, as do array masks, quantized caches and float32 rows.
    monkeypatch.setattr(inference, "_EXACT_ATTENTION_MIN_KEYS", 0)
    monkeypatch.setattr(inference, "_VERIFY_VERDICTS", {})
    monkeypatch.setattr(inference, "_exact_attention_faster", lambda per_row, fused: True)
    monkeypatch.setattr(inference, "_EXACT_ATTENTION_TIMED", {})
    blocks = lambda n: inference._sdpa_vector_blocks(gqa, n)
    q, k = mx.zeros((1, heads[0], 4, 256), mx.bfloat16), mx.zeros((1, heads[1], 70000, 256), mx.bfloat16)
    for n in (1024, 4096, 8192, 16384, 32768, 65536):
        if blocks(n - 1) != blocks(n) or blocks(n) != blocks(n + 1):
            assert inference._exact_attention(q, k[:, :, : n + 1], k[:, :, : n + 1], 0.0625, None, None, lambda: "rows") == "rows"
    for args in ((q, k, k, 0.0625, None, mx.ones((1, 1, 4, k.shape[2]), mx.bool_)), (q, k, k, 0.0625, SimpleNamespace(bits = 8), None),
                 (q.astype(mx.float32), k.astype(mx.float32), k.astype(mx.float32), 0.0625, None, None)):
        assert inference._exact_attention(*args, lambda: "rows") == "rows"
    assert isinstance(inference._exact_attention(q, k, k, 0.0625, None, None, lambda: q), mx.array) and inference._VERIFY_VERDICTS


def test_exact_attention_engages_where_timed_faster_and_caches_each_bucket_verdict(monkeypatch):
    from unsloth_zoo.mlx import inference
    timings, timed = [], inference._exact_attention_faster
    faster = lambda per_row, fused: timings.append(per_row()) or timings[-1] in wins
    monkeypatch.setattr(inference, "_exact_attention_faster", faster)
    monkeypatch.setattr(inference, "_verify_admitted", lambda key, native, fused: "kernel")
    monkeypatch.setattr(inference, "_EXACT_ATTENTION_TIMED", {})
    q, k, wins = mx.zeros((1, 24, 4, 256), mx.bfloat16), mx.zeros((1, 4, 40000, 256), mx.bfloat16), {20000, 9500}
    attend = lambda keys, rows = q: inference._exact_attention(rows, k[:, :, :keys], k[:, :, :keys], 0.0625, None, None, lambda: keys)
    work = inference.first_use_work()
    # Below the bottom limit nothing is timed; above it each power-of-two bucket is timed once, and its verdict holds for it alone.
    assert [attend(n) for n in (5000, 9000, 12000, 20000, 30000, 40000, 10000)] == [5000, 9000, 12000, "kernel", "kernel", 40000, 10000]
    assert timings == [9000, 20000, 40000] and inference.first_use_work() == work + 3
    # Each width is timed apart, and a loss in a larger bucket leaves a smaller one to its own timing.
    assert attend(40000, q[:, :, :3]) == 40000 and attend(9500, q[:, :, :3]) == "kernel" and timings == [9000, 20000, 40000, 40000, 9500]
    monkeypatch.setattr(inference, "_EXACT_ATTENTION_RETRY", 1)
    q5 = mx.zeros((1, 24, 5, 256), mx.bfloat16)
    del timings[:]
    seen = [(attend(9800, q5), len(timings)) for _ in range(20)]
    assert {out for out, _ in seen} == {9800} and [n for _, n in seen] == [1, 1, 2, 2, 2, 2] + [3] * 14
    first = attend(17000, q5)
    wins.add(17000)
    assert [first] + [attend(17000, q5) for _ in range(4)] == [17000, 17000, "kernel", "kernel", "kernel"] and timings == [9800] * 3 + [17000] * 2
    # A kernel that cannot build is a loss, while a failed command buffer still raises.
    for message, raised in (("Unable to build metal library from source", False), (inference._FAILED_BUFFER, True)):
        monkeypatch.setattr(inference, "_EXACT_ATTENTION_TIMED", {})
        monkeypatch.setattr(inference, "_exact_attention_faster", lambda per_row, fused: (_ for _ in ()).throw(RuntimeError(message)))
        with pytest.raises(RuntimeError) if raised else contextlib.nullcontext():
            assert attend(17000, q5) == 17000 and list(inference._EXACT_ATTENTION_TIMED.values()) == [[1, 2]]
    # A threadgroup over the device's limit is retimed at the limit MLX quotes; a limit below one row pair's is a loss.
    refusal = "Thread group size (1024) is greater than  the maximum allowed threads per threadgroup (%d)."
    for limit, kernel in ((640, True), (96, False)):
        monkeypatch.setattr(inference, "_EXACT_ATTENTION_TIMED", {})
        monkeypatch.setattr(inference, "_EXACT_ATTENTION_THREADS", {})
        refused = []
        def faster(per_row, fused):
            if not refused:
                refused.append(1)
                raise ValueError(refusal % limit)
            return True
        monkeypatch.setattr(inference, "_exact_attention_faster", faster)
        assert (attend(17000, q5) == "kernel") is kernel
        assert inference._EXACT_ATTENTION_THREADS == ({(mx.bfloat16, 6): 640} if kernel else {})
    # The kernel must beat the one-row calls by 5% on the medians of three alternating runs; each run advances the clock by its own time.
    for rows, kernel, verdict in (((10, 30, 10), (9.4, 9.4, 30), True), ((10, 10, 40), (9.4, 9.6, 9.6), False)):
        now, spent, order = [0.0], [0.0], []
        def clock():
            now[0], spent[0] = now[0] + spent[0], 0.0
            return now[0]
        run = lambda durations, value: (lambda it: lambda: order.append(value.item()) or spent.__setitem__(0, next(it)) or value)(iter((0,) + durations))
        assert timed(run(rows, mx.zeros(())), run(kernel, mx.ones(())), clock) is verdict and order == [0, 1] * 4


def test_long_verify_blocks_read_attention_in_one_pass_and_match_the_native_verifier(qwen_4bit, monkeypatch):
    from mlx_vlm.models.cache import make_prompt_cache
    from unsloth_zoo.mlx import inference
    model, ids, _ = qwen_4bit
    lm, block = model.language_model, mx.array([ids[0][24:30]])
    cache, prompt = make_prompt_cache(lm), (ids[0] * 64)[:4200]
    lm(mx.array([prompt]), cache = cache, skip_logits = True)
    mx.eval([entry.state for entry in cache])
    native, unscoped = {keep: _verified(lm, cache, block, keep) for keep in (0, 2, 6)}, __import__("copy").copy(lm)
    attention = lambda: [key for key in served if key[1] == "attention"]
    served = _serving(monkeypatch)
    with inference.exact_speculative_verify(model):
        assert _same(native[0], _verified(lm, cache, block, 0)) and not attention()
        monkeypatch.setattr(inference, "_EXACT_ATTENTION_MIN_KEYS", 4096)
        monkeypatch.setattr(inference, "_exact_attention_faster", lambda per_row, fused: True)
        monkeypatch.setattr(inference, "_EXACT_ATTENTION_TIMED", {})
        # An unscoped copy of the model stays on the one-row calls.
        assert _same(native[0], _verified(unscoped, cache, block, 0)) and not attention()
        assert all(_same(expected, _verified(lm, cache, block, keep)) for keep, expected in native.items())
    layers = sum(not layer.is_linear for layer in lm.model.layers)
    assert len(attention()) == layers * len(native) and all(inference._VERIFY_VERDICTS[key] for key in attention())


def test_mtp_rounds_match_solo_decoding(qwen_mtp, monkeypatch):
    from unsloth_zoo.mlx.generate import SamplingParams
    from unsloth_zoo.mlx.inference import exact_speculative_verify
    from unsloth_zoo.mlx.speculative import DraftController, _ShortlistHead, tree_flatten
    model, ids = qwen_mtp
    greedy = SamplingParams()
    solo = [_solo(model, prompt, 96, greedy) for prompt in ids]
    submitted = _submissions(monkeypatch)
    out, drafted, _ = _run(model, ids, 96, _script(("draft", 3), ("copy", 4)), greedy, drafter = _drafter(model))
    assert out == solo and all(accepted >= 0.6 * count for count, accepted in drafted.values())
    unscoped = len(submitted)
    with exact_speculative_verify(model):
        assert _run(model, ids, 96, _script(("draft", 3), ("copy", 4)), greedy, drafter = _drafter(model))[0] == solo
    assert len(submitted) > 2 * unscoped + len(model.language_model.model.layers)
    picked, pick = [], _ShortlistHead.__call__
    monkeypatch.setattr(_ShortlistHead, "__call__", lambda self, hidden: picked.append(self._source is self.table.weight) or pick(self, hidden))
    (out,), drafted, _ = _run(model, ids[:1], 96, DraftController(max_depth = 4, max_copy = 16), greedy, drafter = _drafter(model))
    assert out == solo[0] and drafted[0][1] > 0 and picked and all(picked)
    built = _drafter(model)
    assert built.weight_bytes == sum(a.nbytes for _, a in tree_flatten(built.model.parameters())) + built._shortlist.nbytes


def test_the_shortlist_head_takes_the_tables_own_best_of_its_coarse_copys_few(monkeypatch):
    from unsloth_zoo.mlx import speculative
    [monkeypatch.setattr(speculative, name, value) for name, value in (("_SHORTLIST_CHUNK", 100), ("_SHORTLIST_KEEP", 8), ("_SHORTLIST_COLUMNS", 128))]
    nn, _, hidden = speculative.nn, mx.random.seed(0), mx.random.normal((6, 1, 192))
    dense, narrow, packed = nn.Linear(192, 512, bias = False), nn.Linear(96, 512, bias = False), lambda linear: nn.QuantizedLinear.from_linear(linear, group_size = 32, bits = 8)
    lm = lambda table, tied: SimpleNamespace(args = SimpleNamespace(tie_word_embeddings = tied), lm_head = None if tied else table, model = SimpleNamespace(embed_tokens = table if tied else None))
    for tied, (group, table) in enumerate(((64, dense), (32, packed(dense)))):
        (head := speculative._ShortlistHead.of(lm(table, bool(tied)))).prepare()
        table.update({name: table[name][::-1] for name in ("weight", "scales", "biases") if name in table})
        head.prepare()
        rows = mx.dequantize(table.weight, table.scales, table.biases, group, 8) if tied else table.weight
        scan = mx.quantize(rows[:, :128], 64, 2)
        assert all(mx.array_equal(a, b) for a, b in zip(head._scan, scan)) and head.nbytes == sum(a.nbytes for a in scan)
        coarse, logits = mx.quantized_matmul(hidden[..., :128], *scan, True, 64, 2), hidden @ rows.T
        few = mx.argsort(-coarse, axis = -1)[..., :8]
        best = mx.take_along_axis(few, mx.argmax(mx.take_along_axis(logits, few, axis = -1), axis = -1, keepdims = True), axis = -1)
        assert head(hidden).tolist() == best.squeeze(-1).tolist() != mx.argmax(coarse, axis = -1).tolist()
    assert (small := speculative._ShortlistHead.of(lm(nn.Linear(64, 512, bias = False), False))).prepare() is None and small.nbytes == sum(a.nbytes for a in small._scan) and speculative._ShortlistHead.of(lm(packed(dense), False)).nbytes == speculative._ShortlistHead.of(lm(dense, False)).nbytes and all(speculative._ShortlistHead.of(lm(uncovered, False)) is None for uncovered in (nn.Linear(192, 512), narrow, packed(narrow), nn.Linear(192, 8, bias = False), nn.QuantizedLinear(192, 512, bias = False, bits = 2)))


@pytest.mark.parametrize("max_lag", [128, 4])
def test_mtp_catches_up_after_plain_windows_and_copies(qwen_mtp, max_lag, monkeypatch):
    from mlx_vlm.speculative.common import generation_stream
    from unsloth_zoo.mlx.generate import SamplingParams
    model, ids = qwen_mtp
    controller = _script(("plain", 8), ("draft", 3), ("copy", 4), ("draft", 3))
    accepted, apart, synced = [], [], []
    monkeypatch.setattr(mx, "synchronize", lambda stream = None, sync = mx.synchronize: synced.append(stream) or sync(stream))
    record = controller.record_round
    def record_drafts(plan, rows, counts, **kwargs):
        accepted.extend((count, row.length) for row, count in zip(plan.rows, counts) if row.source == "draft")
        if any(row.source == "draft" for row in plan.rows):
            apart.append(kwargs["catch_up_seconds"] > 0 and synced == [generation_stream])
        synced.clear()
        record(plan, rows, counts, **kwargs)
    controller.record_round, controller.interrupts_plain = record_drafts, lambda rows: False
    drafter = _drafter(model, max_lag = max_lag)
    (out,), _, _ = _run(model, ids[:1], 128, controller, SamplingParams(), drafter = drafter)
    assert out == _solo(model, ids[0], 128, SamplingParams())
    assert any(drafter.dropped.values()) == (max_lag < 8)
    # Only a plain window leaves more pairs than a round does: its replay is timed apart from drafting.
    assert apart[:4] == [max_lag >= 8, False] * 2
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
    assert detect_mtp_splitter(source := get_model_path(MTP_MODEL)).split(str(source), str(tmp_path)) and type(companion_drafter(tmp_path, qwen_mtp[0])) is MTPDrafter and qwen_mtp[0]._unsloth_speculates



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

    by_ids = GenerationRequest(prompt_token_ids = tok.encode(texts[0], add_special_tokens = False), max_tokens = 40)
    batched, alone = run({0: greedy[0], 3: greedy[1], 6: sampled, 7: greedy[2], 8: resumed, 9: by_ids}), run({0: sampled})
    for request, result, expected in zip(greedy, (batched[0], batched[3], batched[7]), upstream):
        n = len(result.token_ids)
        # A stop token ends the reply without joining its text.
        assert result.token_ids == [token for token, _ in expected[:n]] and (n == request.max_tokens or expected[n][0] in tok.stopping_criteria.eos_token_ids)
        assert result.logprobs == pytest.approx([logprobs[token].item() for token, logprobs in expected[:n]], abs = 0.02)
    assert (batched[6].token_ids, batched[6].logprobs) == (alone[0].token_ids, alone[0].logprobs)
    assert batched[9].token_ids == batched[0].token_ids  # a token-id prompt prefills as its rendered twin
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


def test_speculative_batch_rows_prefill_on_the_batch_step(qwen, monkeypatch):
    from unsloth_zoo.mlx import generate
    from unsloth_zoo.mlx.speculative import SpeculativeDraft
    model, seen, admit = qwen[0], [], SpeculativeDraft.admit
    monkeypatch.setattr(model, "_is_vlm_model", True, raising = False)
    monkeypatch.setattr(SpeculativeDraft, "admit", lambda self, *args, **kwargs: (seen.append(kwargs.get("prefill_step_size")), admit(self, *args, **kwargs))[1])
    for step in (None, 8):
        with generate.BatchStream(model, None, defaults = generate.GenerationDefaults(max_tokens = 2, prefill_step_size = step), speculative = SpeculativeDraft(_rounds_only())) as stream:
            stream.add(generate.GenerationRequest(prompt = "Name three colours."))
    assert seen == [None, 8]


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

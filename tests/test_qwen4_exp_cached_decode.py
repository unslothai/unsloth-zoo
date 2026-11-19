# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Qwen4Exp QSA cached decode: decoding through the cache (block keys cached, selected K / V rows gathered) must give
the full-prefix recompute's logits, and every step must keep the reference indexer's tokens on the same inputs and
cache exactly the reference's pooled block keys. Covers prompts below / at / above the budget, blocks completing
inside multi-token steps, left-padded batches, a LoRA adapter on and off, eager masks, fp16 / bf16, the kill switch,
and that a block pooled before all its tokens are written (SGLang #46125) is caught."""

import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")

from unsloth_zoo.temporary_patches import qwen4_exp as zoo  # noqa: E402

BUDGET, RATIO = 16, 4
_REAL_CACHED_DECODE = zoo._qwen4_exp_cached_decode


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE", "gpu")
    monkeypatch.delenv("UNSLOTH_QWEN4_EXP_CACHED_DECODE", raising = False)
    # The tiny model's contexts are far below the default minimum: engage from the budget on.
    monkeypatch.setenv("UNSLOTH_QWEN4_EXP_CACHED_DECODE_MIN_CONTEXT", "0")
    zoo.patch_qwen4_exp()
    assert modeling.Qwen4ExpTextAttention.forward is zoo.qwen4_exp_attention_forward
    # CPU: the torch linear-attention paths (the GPU host's Triton kernels reject CPU tensors).
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
                 "causal_conv1d_update"):
        original = getattr(getattr(modeling, name, None), "__wrapped__", None)
        if original is not None:
            monkeypatch.setattr(modeling, name, original)
    threads = torch.get_num_threads()
    torch.set_num_threads(min(threads, 4))
    yield monkeypatch
    torch.set_num_threads(threads)


def _model(dtype = torch.float32, attn = "sdpa", lora = False, seed = 0):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    config = Qwen4ExpTextConfig(
        hidden_size = 64, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 32,
        indexer_budget = BUDGET, indexer_compress_ratio = RATIO, indexer_head_dim = 16, indexer_n_heads = 16,
        indexer_kv_heads = 1, partial_rotary_factor = 0.25, num_hidden_layers = 4, vocab_size = 128,
        linear_num_key_heads = 2, linear_num_value_heads = 2, linear_key_head_dim = 16, linear_value_head_dim = 16,
        num_experts = 4, num_experts_per_tok = 2, moe_intermediate_size = 16, shared_expert_intermediate_size = 32,
        hc_count = 2, hc_lowrank = 8, ple_layer_ids = [],
        # 16 index heads make an all-zero block score (a top-k tie, order unspecified) a 2^-16 event.
        layer_types = ["linear_attention", "full_attention", "linear_attention", "full_attention"],
    )
    config._attn_implementation = attn
    torch.manual_seed(seed)
    model = modeling.Qwen4ExpForCausalLM(config)
    for layer in model.model.layers:
        layer.mlp = layer.mlp.shared_expert
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "index_qk_proj" in name:
                p.normal_(0, 0.3)
    if lora:
        from peft import LoraConfig, inject_adapter_in_model
        inject_adapter_in_model(LoraConfig(r = 4, lora_alpha = 8, init_lora_weights = False,
                                           target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "index_qk_proj"]),
                                model)
    return model.to(dtype).eval()


def _set_adapters(model, enabled):
    from peft.tuners.tuners_utils import BaseTunerLayer
    n = 0
    for m in model.modules():
        if isinstance(m, BaseTunerLayer):
            m.enable_adapters(enabled)
            n += 1
    assert n > 0


def _positions(mask):
    # What generate passes: cumulative positions from the first real token; pads get 1.
    return (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)


def _batch(lengths, total, seed = 3):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, 128, (len(lengths), total), generator = g)
    mask = torch.ones_like(ids)
    width = max(lengths)
    for b, n in enumerate(lengths):
        mask[b, : width - n] = 0
        ids[b, : width - n] = 0
    return ids, mask, width


def _decode(model, ids, mask, prompt, steps, greedy):
    """Prefill, then the given step sizes through the model's own cache. Greedy fills ids from the logits."""
    ids = ids.clone()
    pos = _positions(mask)
    with torch.no_grad():
        out = model(input_ids = ids[:, :prompt], attention_mask = mask[:, :prompt], position_ids = pos[:, :prompt],
                    use_cache = True)
        logits, past, cur, filled = [out.logits], out.past_key_values, prompt, []
        for n in steps:
            if greedy:
                ids[:, cur] = logits[-1][:, -1].argmax(-1)
                filled.append(cur)
            out = model(input_ids = ids[:, cur : cur + n], attention_mask = mask[:, : cur + n],
                        position_ids = pos[:, cur : cur + n], past_key_values = past, use_cache = True)
            logits.append(out.logits)
            past, cur = out.past_key_values, cur + n
    return ids[:, :cur], torch.cat(logits, dim = 1).float(), filled


def _recompute(model, ids, mask, monkeypatch):
    """The reference: the whole prefix in one no-cache forward through transformers' per-query indexer loop."""
    monkeypatch.setenv("UNSLOTH_QWEN4_EXP_SPARSE_ATTN", "0")
    monkeypatch.setenv("UNSLOTH_QWEN4_EXP_VECTORISED_QSA", "0")
    with torch.no_grad():
        out = model(input_ids = ids, attention_mask = mask[:, : ids.shape[1]],
                    position_ids = _positions(mask[:, : ids.shape[1]]), use_cache = False)
    monkeypatch.delenv("UNSLOTH_QWEN4_EXP_SPARSE_ATTN")
    monkeypatch.delenv("UNSLOTH_QWEN4_EXP_VECTORISED_QSA")
    return out.logits.float()


class _RawKeys:
    """A cache whose indexer already holds this step's keys: the reference indexer reads them without a write."""

    def __init__(self, raw):
        self.raw = raw

    def update_indexer(self, raw_keys, layer_idx):
        return self.raw


def _reference_block_keys(indexer, raw, full_cos, full_sin, start, n_blocks):
    """The reference indexer's pooled, normed, roped block keys of one row (fp32)."""
    d = indexer.index_head_dim
    slots = start + torch.arange(n_blocks * RATIO, device = raw.device)
    pooled = raw.index_select(0, slots).view(n_blocks, RATIO, d).float().mean(dim = 1).to(raw.dtype)
    pooled = indexer.k_layernorm(pooled)
    first = slots.view(n_blocks, RATIO)[:, 0]
    return modeling.apply_rotary_pos_emb(pooled.unsqueeze(1), cos = full_cos.index_select(0, first),
                                         sin = full_sin.index_select(0, first)).squeeze(1).float()


def _install_checker(monkeypatch, record):
    """Around every cached-decode step: the selection must equal the reference indexer's on the same inputs, and the
    cached block keys must equal the reference's pooling of the complete blocks."""
    real = _REAL_CACHED_DECODE

    def checked(module, hidden_states, position_embeddings, attention_mask, past_key_values, kwargs):
        trace = []
        zoo._QSA_DECODE_TRACE = trace
        try:
            out = real(module, hidden_states, position_embeddings, attention_mask, past_key_values, kwargs)
        finally:
            zoo._QSA_DECODE_TRACE = None
        if out is None:
            return None
        layer = past_key_values.layers[module.layer_idx]
        raw = layer.indexer_keys
        total = raw.shape[1]
        full_cos, full_sin = position_embeddings
        if hidden_states.dtype in (torch.float16, torch.bfloat16):
            full_cos, full_sin = full_cos.to(hidden_states.dtype), full_sin.to(hidden_states.dtype)
        with torch.no_grad():
            ref = zoo._reference_qsa_forward(module.indexer, hidden_states, (full_cos, full_sin), attention_mask,
                                             _RawKeys(raw))
        ref = ref if ref.dtype == torch.bool else ref == 0
        visible = attention_mask[:, 0] if attention_mask.dtype == torch.bool else attention_mask[:, 0] == 0
        ref = ref[:, 0] & visible
        (_, selected), = trace
        ours = torch.zeros(*selected.shape[:2], total + 1, dtype = torch.bool, device = selected.device)
        ours.scatter_(-1, torch.where(selected >= 0, selected, total), True)
        assert torch.equal(ours[..., :total], ref), f"layer {module.layer_idx}: selection differs at {total} keys"
        blocks = layer._unsloth_qsa_decode["blocks"]
        for b in range(hidden_states.shape[0]):
            start = int(visible[b, -1].nonzero()[0])
            n = (total - start) // RATIO
            want = _reference_block_keys(module.indexer, raw[b], full_cos[b], full_sin[b], start, n)
            torch.testing.assert_close(blocks[b, :n], want, rtol = 1e-5, atol = 1e-5,
                                       msg = f"layer {module.layer_idx} row {b}: cached block keys at {total} keys")
        record.append(total)
        return out

    monkeypatch.setattr(zoo, "_qwen4_exp_cached_decode", checked)


TOL = {torch.float32: dict(rtol = 1e-4, atol = 1e-4), torch.bfloat16: dict(rtol = 0, atol = 6e-2),
       torch.float16: dict(rtol = 0, atol = 1e-2)}


def _assert_matches(model, ids, mask, logits, monkeypatch, dtype, filled):
    want = _recompute(model, ids, mask, monkeypatch)
    real = mask[:, : ids.shape[1]].bool()   # pad rows are not compared
    torch.testing.assert_close(logits[real], want[real], **TOL[dtype])
    if filled:
        # Every greedy token is the recompute's argmax wherever its top two are apart by more than the tolerance.
        at = torch.tensor(filled, device = ids.device)
        top2 = want[:, at - 1].topk(2, dim = -1).values
        clear = (top2[..., 0] - top2[..., 1]) > 2 * TOL[dtype]["atol"]
        agree = want[:, at - 1].argmax(-1) == ids[:, at]
        assert bool((agree | ~clear).all()), (agree, clear)


def _run(monkeypatch, *, lengths, steps, greedy = True, dtype = torch.float32, attn = "sdpa", lora = None,
         kill = False, device = "cpu", rope_type = None):
    total = max(lengths) + sum(steps)
    ids, mask, prompt = _batch(lengths, total)
    ids, mask = ids.to(device), mask.to(device)
    model = _model(dtype, attn, lora = lora is not None).to(device)
    if rope_type is not None:
        model.config.rope_parameters = {**(model.config.rope_parameters or {}), "rope_type": rope_type}
    if lora is not None:
        _set_adapters(model, lora)
    if kill:
        monkeypatch.setenv("UNSLOTH_QWEN4_EXP_CACHED_DECODE", "0")
    record = []
    _install_checker(monkeypatch, record)
    before = zoo.QWEN4_EXP_SPARSE_STATS["cached_decode"]
    out_ids, logits, filled = _decode(model, ids, mask, prompt, steps, greedy)
    engaged = zoo.QWEN4_EXP_SPARSE_STATS["cached_decode"] - before
    monkeypatch.delenv("UNSLOTH_QWEN4_EXP_CACHED_DECODE", raising = False)
    _assert_matches(model, out_ids, mask, logits, monkeypatch, dtype, filled)
    counter = zoo.QWEN4_EXP_CACHED_DECODE_MISMATCH.get(ids.device)
    assert counter is None or int(counter) == 0
    return engaged, record


@pytest.mark.parametrize("prompt", [10, 16, 40], ids = ["below_budget", "at_budget", "above_budget"])
def test_greedy_decode_equals_full_recompute(patched, prompt):
    # Left-padded batch (row 2 is 5 shorter); 30 steps cross the budget and the 64-token linear-attention chunk.
    steps = [1] * 30
    engaged, record = _run(patched, lengths = [prompt, prompt - 5], steps = steps)
    # Two QSA layers per step from the first step whose keys reach budget + ratio.
    expected = 2 * sum(1 for i in range(len(steps)) if prompt + i + 1 >= BUDGET + RATIO)
    assert engaged == expected == len(record) and engaged > 0


def test_multi_token_steps_complete_blocks_inside_the_step(patched):
    # Steps of 2 - 7 tokens: blocks complete in the middle of a step, and earlier rows of the step must not see them.
    engaged, record = _run(patched, lengths = [23, 18], steps = [3, 1, 5, 2, 7, 1, 6, 3], greedy = False)
    assert engaged == 16 and len(record) == 16


def test_lora_adapter_on_and_off(patched):
    for enabled in (True, False):
        engaged, _ = _run(patched, lengths = [26, 21], steps = [1] * 12 + [3, 2], greedy = False, lora = enabled)
        assert engaged == 28


def test_eager_float_masks(patched):
    engaged, _ = _run(patched, lengths = [30, 27], steps = [1] * 10 + [5], greedy = False, attn = "eager")
    assert engaged == 22


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids = ["bf16", "fp16"])
def test_half_precision(patched, dtype):
    engaged, _ = _run(patched, lengths = [30, 24], steps = [1] * 12 + [3], greedy = True, dtype = dtype)
    assert engaged == 26


def test_kill_switch_keeps_the_reference(patched):
    engaged, record = _run(patched, lengths = [30, 25], steps = [1] * 8 + [3], kill = True)
    assert engaged == 0 and record == []


def test_min_context_gates_engagement(patched):
    # Below MIN_CONTEXT the reference runs; from it on the cached path rebuilds and still matches the recompute.
    assert zoo._CACHED_DECODE_MIN_CONTEXT >= 16384
    patched.delenv("UNSLOTH_QWEN4_EXP_CACHED_DECODE_MIN_CONTEXT")
    engaged, record = _run(patched, lengths = [30, 25], steps = [1] * 8 + [3])
    assert engaged == 0 and record == []
    # Threshold 36 keys: prompt 30, steps end at 31 .. 38 then 41 keys: 4 steps x 2 QSA layers engage.
    patched.setenv("UNSLOTH_QWEN4_EXP_CACHED_DECODE_MIN_CONTEXT", "36")
    engaged, record = _run(patched, lengths = [30, 25], steps = [1] * 8 + [3])
    assert engaged == 8 == len(record) and min(record) == 36


def test_block_pooled_before_it_completes_is_caught(patched):
    # The SGLang #46125 bug: pool the block a step's last token falls in, before its remaining tokens are written.
    patched.setattr(zoo, "_qsa_pooled_blocks",
                    lambda n, start, ratio: torch.div(n - start + ratio - 1, ratio, rounding_mode = "floor").clamp_min(0))
    with pytest.raises(AssertionError, match = "selection differs|cached block keys"):
        _run(patched, lengths = [30, 25], steps = [1] * 8)


def test_cache_edits_rebuild_the_block_cache(patched):
    # A beam reorder swaps the raw key tensor (rows exchanged): the block cache is rebuilt, never reused stale.
    ids, mask, prompt = _batch([30, 26], 44)
    model = _model()
    record = []
    _install_checker(patched, record)
    pos = _positions(mask)
    with torch.no_grad():
        out = model(input_ids = ids[:, :prompt], attention_mask = mask[:, :prompt], position_ids = pos[:, :prompt])
        past, cur = out.past_key_values, prompt
        for n in (3, 4, 1, 1, 5):
            if n == 1 and cur == prompt + 8:
                swap = torch.tensor([1, 0])
                past.reorder_cache(swap)
                ids, mask, pos = ids[swap], mask[swap], pos[swap]
            out = model(input_ids = ids[:, cur : cur + n], attention_mask = mask[:, : cur + n],
                        position_ids = pos[:, cur : cur + n], past_key_values = past)
            past, cur = out.past_key_values, cur + n
    assert len(record) == 10


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_cuda_decode_step_has_no_host_sync(monkeypatch):
    """Every cached-decode call runs under torch.cuda.set_sync_debug_mode('error'): any GPU -> CPU sync raises."""
    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE", "gpu")
    monkeypatch.setenv("UNSLOTH_QWEN4_EXP_CACHED_DECODE_MIN_CONTEXT", "0")
    zoo.patch_qwen4_exp()
    real = _REAL_CACHED_DECODE
    calls = []

    def no_sync(*args):
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            out = real(*args)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        calls.append(out is not None)
        return out

    monkeypatch.setattr(zoo, "_qwen4_exp_cached_decode", no_sync)
    for dtype in (torch.bfloat16, torch.float16):
        model = _model(dtype).cuda()
        ids, mask, prompt = _batch([40, 33], 60)
        ids, mask = ids.cuda(), mask.cuda()
        _decode(model, ids, mask, prompt, [1] * 12 + [3, 5], greedy = True)
    # Per dtype and QSA layer: the prefill (an empty cache: not taken, no sync either) and 14 cached steps.
    assert len(calls) == 2 * 2 * 15 and sum(calls) == 2 * 2 * 14
    counter = zoo.QWEN4_EXP_CACHED_DECODE_MISMATCH.get(torch.device("cuda", torch.cuda.current_device()))
    assert counter is not None and int(counter) == 0



def test_ties_keep_the_count(patched):
    # All-zero index keys: every block ties; still budget / ratio distinct visible blocks plus the tail.
    lengths, steps = [30, 25], [1] * 6 + [4]
    ids, mask, prompt = _batch(lengths, max(lengths) + sum(steps))
    model = _model()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "index_qk_proj" in name:
                p[-model.config.indexer_head_dim:] = 0
    trace = []
    patched.setattr(zoo, "_QSA_DECODE_TRACE", trace)
    _decode(model, ids, mask, prompt, steps, greedy = False)
    assert len(trace) == 2 * len(steps)
    cur = prompt
    for step, n in enumerate(steps):
        for _, selected in trace[2 * step : 2 * step + 2]:
            for b, length in enumerate(lengths):
                start = prompt - length
                for i in range(n):
                    visible = cur + i - start + 1
                    tokens = selected[b, i][selected[b, i] >= 0]
                    want = min(visible // RATIO, BUDGET // RATIO) * RATIO + visible % RATIO
                    assert tokens.numel() == want == tokens.unique().numel()
                    assert bool(((tokens >= start) & (tokens <= cur + i)).all())
                    # Whole blocks counted from the row's first visible key, then the tail.
                    blocks = (tokens - start) // RATIO
                    whole = blocks[blocks < visible // RATIO]
                    assert bool((whole.bincount()[whole.unique()] == RATIO).all())
        cur += n


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids = ["bf16", "fp16"])
def test_cuda_decode_equals_full_recompute(patched, dtype):
    engaged, record = _run(patched, lengths = [40, 33], steps = [1] * 16 + [3, 5], dtype = dtype, device = "cuda")
    assert engaged == 36 == len(record)


@pytest.mark.parametrize("rope_type", ["dynamic", "longrope"])
def test_dynamic_rope_keeps_the_reference(patched, rope_type):
    engaged, record = _run(patched, lengths = [30, 25], steps = [1] * 8 + [3], rope_type = rope_type)
    assert engaged == 0 and record == []


def _install_selection_checker(monkeypatch, record):
    """Around every cached-decode step: the selection must equal the reference indexer's on the same inputs."""
    real = _REAL_CACHED_DECODE

    def checked(module, hidden_states, position_embeddings, attention_mask, past_key_values, kwargs):
        trace = []
        zoo._QSA_DECODE_TRACE = trace
        try:
            out = real(module, hidden_states, position_embeddings, attention_mask, past_key_values, kwargs)
        finally:
            zoo._QSA_DECODE_TRACE = None
        if out is None:
            return None
        raw = past_key_values.layers[module.layer_idx].indexer_keys
        total = raw.shape[1]
        with torch.no_grad():
            ref = zoo._reference_qsa_forward(module.indexer, hidden_states, position_embeddings, attention_mask,
                                             _RawKeys(raw))
        ref = ref if ref.dtype == torch.bool else ref == 0
        (_, selected), = trace
        ours = torch.zeros(*selected.shape[:2], total + 1, dtype = torch.bool, device = selected.device)
        ours.scatter_(-1, torch.where(selected >= 0, selected, total), True)
        assert torch.equal(ours[..., :total], ref[:, 0]), f"layer {module.layer_idx}: selection differs at {total} keys"
        record.append(total)
        return out

    monkeypatch.setattr(zoo, "_qwen4_exp_cached_decode", checked)


def test_padding_mask_with_gaps_matches_the_reference(patched):
    # Right padding and holes inside the prompt (a 2D mask with interior zeros): transformers forms each query's
    # blocks from its visible slots in order, not from contiguous slots after the first visible one.
    total, steps = 44, [1] * 10 + [3, 1]
    ids, mask, _ = _batch([30, 30], total)
    mask[0, 5:8] = 0
    mask[1, :2] = 0
    mask[1, 20:22] = 0
    ids[mask == 0] = 0
    model = _model()
    record = []
    _install_selection_checker(patched, record)
    before = zoo.QWEN4_EXP_SPARSE_STATS["cached_decode"]
    out_ids, logits, filled = _decode(model, ids, mask, 30, steps, greedy = True)
    engaged = zoo.QWEN4_EXP_SPARSE_STATS["cached_decode"] - before
    assert engaged == 2 * len(steps) == len(record)
    _assert_matches(model, out_ids, mask, logits, patched, torch.float32, filled)
    counter = zoo.QWEN4_EXP_CACHED_DECODE_MISMATCH.get(ids.device)
    assert counter is None or int(counter) == 0


def test_interior_gap_selection_equals_the_reference(patched):
    # One hidden slot (3) in a 40-key row: 9 slots differed when blocks were taken from contiguous slots.
    from transformers.cache_utils import DynamicIndexedLayer
    model = _model()
    indexer = model.model.layers[1].self_attn.indexer
    torch.manual_seed(42)
    layer = DynamicIndexedLayer()
    layer.update_indexer(torch.randn(1, 39, indexer.index_head_dim))
    hidden = torch.randn(1, 1, model.config.hidden_size)
    cos, sin = torch.randn(1, 40, 8), torch.randn(1, 40, 8)
    mask = torch.ones(1, 1, 1, 40, dtype = torch.bool)
    mask[..., 3] = False
    ref = zoo._reference_qsa_forward(indexer, hidden, (cos, sin), mask, _RawKeys(torch.cat(
        [layer.indexer_keys, indexer.index_qk_proj(hidden)[..., -indexer.index_head_dim:]], dim = 1)))
    selected = zoo._qsa_cached_select(indexer, layer, hidden, cos, sin, mask, modeling.apply_rotary_pos_emb, None)
    assert sorted(selected[selected >= 0].tolist()) == ref[0, 0, 0].nonzero().flatten().tolist()


def test_mask_outside_the_padding_shape_keeps_the_reference_on_cpu(patched):
    # A 2-token step whose first row sees slot 3 but whose last row does not (a caller-built 4D mask): no cached
    # selection can serve both rows, so a host-resident mask takes the reference before any cache write.
    from transformers.cache_utils import DynamicCache
    model = _model()
    module = model.model.layers[1].self_attn
    torch.manual_seed(5)
    n = 40
    past = DynamicCache(config = model.config)
    past.update(torch.randn(1, 2, n - 2, 32), torch.randn(1, 2, n - 2, 32), module.layer_idx)
    past.update_indexer(torch.randn(1, n - 2, 16), module.layer_idx)
    mask = torch.ones(1, 1, 2, n, dtype = torch.bool)
    mask[:, :, 0, -1] = False
    mask[:, :, 1, 3] = False
    counter = zoo.QWEN4_EXP_CACHED_DECODE_MISMATCH.get(torch.device("cpu"))
    before = 0 if counter is None else int(counter)
    with torch.no_grad():
        out = _REAL_CACHED_DECODE(module, torch.randn(1, 2, 64), (torch.randn(1, n, 8), torch.randn(1, n, 8)), mask,
                                  past, {})
    assert out is None
    assert past.layers[module.layer_idx].indexer_keys.shape[1] == n - 2   # untouched: the reference appends
    counter = zoo.QWEN4_EXP_CACHED_DECODE_MISMATCH.get(torch.device("cpu"))
    assert (0 if counter is None else int(counter)) == before


def test_inference_mode_then_no_grad_decode(patched):
    # The mismatch counter is created on the first cached step; under inference_mode it must not become an
    # inference tensor, or a later cached step under plain no_grad raises on counter.add_().
    zoo.QWEN4_EXP_CACHED_DECODE_MISMATCH.clear()
    model = _model()
    ids, mask, prompt = _batch([30, 30], 40)
    pos = _positions(mask)

    def step(ctx):
        before = zoo.QWEN4_EXP_SPARSE_STATS["cached_decode"]
        with ctx():
            out = model(input_ids = ids[:, :prompt], attention_mask = mask[:, :prompt],
                        position_ids = pos[:, :prompt], use_cache = True)
            model(input_ids = ids[:, prompt : prompt + 1], attention_mask = mask[:, : prompt + 1],
                  position_ids = pos[:, prompt : prompt + 1], past_key_values = out.past_key_values, use_cache = True)
        return zoo.QWEN4_EXP_SPARSE_STATS["cached_decode"] - before

    assert step(torch.inference_mode) > 0
    assert step(torch.no_grad) > 0
    counter = zoo.QWEN4_EXP_CACHED_DECODE_MISMATCH[ids.device]
    assert not counter.is_inference() and int(counter) == 0

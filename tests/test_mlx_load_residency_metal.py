# SPDX-License-Identifier: AGPL-3.0-only
"""What a load holds. A finished load must own its weights, measured as the memory evaluating them
allocates, over several layers so evaluating only the first cannot reach the total. An offloaded
one reads PLE rows and experts from the checkpoint instead, computing the stock modules' outputs
bit for bit."""

import gc
import inspect
import json
import os
import sys
import threading
import warnings
import weakref

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from mlx_simulation import mlx_is_simulated

if mlx_is_simulated():
    pytest.skip("Requires native MLX", allow_module_level = True)
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason = "Requires Metal")

import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
from unsloth_zoo.mlx import offload as zoo_offload
from unsloth_zoo.mlx.loader import (
    FastMLXModel, _MLXQuantizationSpec, _apply_mlx_quantization, _apply_offload, _auto_offload,
    _convert_mlx_dtype, _finish_load, _load_mlx_vlm_with_extra_weight_filter, _mlx_save_pretrained_merged, _settle_offload,
)
from unsloth_zoo.mlx.offload import (
    DiskEmbedding, OffloadSwitchGLU, context_bytes, offload_experts, offload_footprint, offload_inventory,
    offload_ple, offload_setting, offloadable, plan_offload,
)

_SIDES = (2048, 1024, 512)
_BYTES = sum(side * side * 4 for side in _SIDES)
_SMALLEST_LAYER = min(side * side * 4 for side in _SIDES)


@pytest.fixture(autouse = True)
def _real_mlx_still_installed():
    # A sibling file's torch shim would make these measure nothing.
    if sys.modules.get("mlx.core") is not mx or mx.default_device() != mx.gpu:
        pytest.skip("another test replaced mlx.core with the simulation shim")


def _lazy_model(tmp_path):
    path = str(tmp_path / "weights.safetensors")
    mx.save_safetensors(path, {
        str(i): mx.random.normal((side, side)).astype(mx.float32)
        for i, side in enumerate(_SIDES)
    })
    mx.clear_cache()
    mapped = mx.load(path)
    model = nn.Sequential(*(nn.Linear(side, side, bias = False) for side in _SIDES))
    for i, layer in enumerate(model.layers):
        layer.update({"weight": mapped[str(i)]})
    return model


def test_runtime_quantization_reads_each_weight_before_its_kernel(tmp_path, monkeypatch):
    """A quantize kernel evaluated while its source is still being read keeps a GPU command
    buffer waiting on the disk: by then, reading each source must allocate nothing."""
    quantize, evaluate = mx.quantize, mx.eval
    pending, reads = {}, []

    def recording_quantize(weight, *args, **kwargs):
        quantized = quantize(weight, *args, **kwargs)
        pending[id(quantized[0])] = (quantized[0], weight)  # held, so the id is never reused
        return quantized

    def checking_eval(*arrays):
        for _, array in tree_flatten(list(arrays)):
            _, source = pending.pop(id(array), (None, None))
            if source is not None:
                before = mx.get_active_memory()
                evaluate(source)
                reads.append(mx.get_active_memory() - before)
        return evaluate(*arrays)

    model = _lazy_model(tmp_path)
    monkeypatch.setattr(mx, "quantize", recording_quantize)
    monkeypatch.setattr(mx, "eval", checking_eval)
    spec = _MLXQuantizationSpec(enabled = True, bits = 4, group_size = 64)
    model, _ = _apply_mlx_quantization(model, {}, spec, is_vlm = False)
    mx.eval(model.parameters())  # as the load branches do after quantizing
    assert len(reads) == len(_SIDES) and not any(reads), reads


def test_finish_load_returns_resident_weights(tmp_path):
    model = _lazy_model(tmp_path)
    before = mx.get_active_memory()
    returned, tokenizer = _finish_load(model, "tokenizer")
    assert (returned, tokenizer) == (model, "tokenizer")
    assert mx.get_active_memory() - before >= _BYTES


def test_tables_mlx_vlm_reads_from_disk_stay_there(tmp_path):  # as it exposes DeepSeek V4.1's Engram
    model, before = _lazy_model(tmp_path), mx.get_active_memory()
    for layer in model.layers[:2]:  # V4.1 has two
        layer.__class__ = type("OffloadedEngramEmbedding", (nn.Linear,), {})
    _finish_load(_convert_mlx_dtype(model, mx.bfloat16) or model, None)
    assert 0 < mx.get_active_memory() - before < _SIDES[1] ** 2 * 2


def test_glm5_next_fused_layers_take_their_checkpoint_bits(monkeypatch):
    glm = pytest.importorskip("mlx_vlm.models.glm5_next.glm5_next")
    from mlx_vlm.utils import _quantization_for_module_path

    monkeypatch.delattr(glm.Model, "quantization_path_aliases", raising = False)
    fused = {"mlp.shared_experts.gate_up_proj": "mlp.shared_experts.gate_proj", "self_attn.qkv_proj": "self_attn.q_proj",
             "self_attn.fbg_a_proj": "self_attn.f_a_proj", "self_attn.qkv_a_proj": "self_attn.q_a_proj"}
    # Bits of their own, as GLM-5.3-Flash-4bit's 6-bit shared experts beside its 4-bit default.
    bits = [{"group_size": 64, "bits": 5 + i} for i in range(len(fused))]
    quantization = {f"model.layers.3.{source}": b for source, b in zip(fused.values(), bits)}
    model = glm.Model.__new__(glm.Model)
    load = lambda *_, **__: [_quantization_for_module_path(quantization, f"language_model.model.layers.3.{p}", model) for p in fused]
    assert _load_mlx_vlm_with_extra_weight_filter("glm", "glm5_next", load, {}) == bits


def test_lazy_weights_env_leaves_the_load_lazy(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_MLX_LAZY_WEIGHTS", "1")
    model = _lazy_model(tmp_path)
    before = mx.get_active_memory()
    _finish_load(model, None)
    # Under the smallest layer, not under the total: any one layer evaluated is a leak.
    assert mx.get_active_memory() - before < _SMALLEST_LAYER


def test_unwalkable_weights_report_but_do_not_fail_the_load(tmp_path, capsys):
    model = _lazy_model(tmp_path)

    def parameters():
        raise RuntimeError("cannot walk")

    model.parameters = parameters
    assert _finish_load(model, None) == (model, None)
    assert "cannot walk" in capsys.readouterr().out


def test_an_unreadable_weight_reports_but_does_not_fail_the_load(tmp_path, capsys, monkeypatch):
    model = _lazy_model(tmp_path)

    def refuse(*_args, **_kwargs):
        raise RuntimeError("cannot read shard")

    monkeypatch.setattr(mx, "eval", refuse)
    assert _finish_load(model, None) == (model, None)
    assert "cannot read shard" in capsys.readouterr().out


# Offload.

def _with_negative_zeros(module):
    weight = module.weight.astype(mx.bfloat16)
    module.weight = mx.where(mx.arange(weight.shape[-1]) % 7 == 3, mx.array(-0.0, mx.bfloat16), weight)
    return module


def _table(kind):
    if kind.startswith("sharded"):
        ShardedEmbedding = pytest.importorskip("mlx_vlm.models.qwen4_exp.language").ShardedEmbedding
        module = ShardedEmbedding(1000, 64, 7)
        for shard in module.shards:
            _with_negative_zeros(shard)
        if kind == "sharded_quantized":
            nn.quantize(module, group_size = 32, bits = 4)
        return "ngram_embedding", module
    module = _with_negative_zeros(nn.Embedding(500, 96))
    if kind in ("quantized", "mxfp4"):
        mode = "mxfp4" if kind == "mxfp4" else "affine"
        module = nn.QuantizedEmbedding.from_embedding(module, group_size = 32, bits = 4, mode = mode)
    return "embed_tokens_per_layer", module


_MLX_PREFIX, _HF_PREFIX = "language_model.model.", "model.language_model."


def _lazy_copy(build, directory, layout = _MLX_PREFIX):
    """``build()``'s model with its weights lazily mapped from ``directory``, saved there once, its
    ``language_model.model.`` names under ``layout``."""
    model = build()
    path = os.path.join(directory, "model.safetensors")
    rename = lambda name, a, b: b + name[len(a):] if name.startswith(a) else name
    if not os.path.exists(path):
        mx.save_safetensors(path, {
            rename(name, _MLX_PREFIX, layout): value for name, value in tree_flatten(model.parameters())
        })
    model.load_weights([(rename(name, layout, _MLX_PREFIX), value) for name, value in mx.load(path).items()])
    return model


def _holder(name, module):
    """``module`` at ``language_model.model.<name>``, where the VLM families keep their tables."""
    inner = nn.Module()
    setattr(inner, name, module)
    model = nn.Module()
    model.language_model = nn.Module()
    model.language_model.model = inner
    return model


def _site(model, name):
    return getattr(model.language_model.model, name)


def _cache_limit():
    limit = mx.set_cache_limit(0)
    mx.set_cache_limit(limit)
    return limit


def _bits(array):
    return np.array(array.view(mx.uint16) if array.dtype == mx.bfloat16 else array)


@pytest.mark.parametrize("kind", ["dense", "quantized", "mxfp4", "sharded", "sharded_quantized"])
@pytest.mark.parametrize("fraction", [1.0, 0.5])
def test_offloaded_table_matches_stock(tmp_path, kind, fraction):
    name, module = _table(kind)
    # Checkpoint layouts: Hugging Face's VLM, and a text checkpoint mlx-vlm wraps as its language model.
    layout = {"quantized": _HF_PREFIX, "mxfp4": "model."}.get(kind, _MLX_PREFIX)
    model = _lazy_copy(lambda: _holder(name, module), str(tmp_path), layout)
    # Repeats, unsorted, and rows on both sides of the held prefix of several shards; the second
    # lookup mixes rows the first one read with new ones.
    rng = np.random.default_rng(0)
    first = rng.integers(0, 480, (3, 41))
    lookups = [mx.array(first), mx.array(np.concatenate([first[:, :20], rng.integers(0, 480, (3, 21))], 1))]
    expected = [_site(model, name)(ids) for ids in lookups]
    table_bytes = sum(value.nbytes for _, value in tree_flatten(model.parameters()))
    listing = sorted(os.listdir(tmp_path))

    report = offload_ple(model, str(tmp_path), fraction)

    replacement = _site(model, name)
    assert isinstance(replacement, DiskEmbedding) and not tree_flatten(model.parameters())
    assert report["tables"] == 1 and report["skipped"] == []
    assert report["on_disk_bytes"] + report["resident_bytes"] == table_bytes
    assert abs(report["resident_bytes"] - table_bytes * (1 - fraction)) <= table_bytes * 0.02
    for ids, reference in zip(lookups, expected):
        actual = replacement(ids)
        assert actual.dtype == reference.dtype and actual.shape == reference.shape
        np.testing.assert_array_equal(_bits(actual), _bits(reference))
    assert sorted(os.listdir(tmp_path)) == listing  # nothing is written next to the checkpoint


def test_offloaded_rows_the_page_cache_dropped_are_read_again(tmp_path, monkeypatch):
    name, module = _table("sharded_quantized")
    model = _lazy_copy(lambda: _holder(name, module), str(tmp_path))
    ids, many = mx.array([[7, 400, 999, 3]]), mx.arange(0, 1000, 5)[None]
    expected, expected_many = _site(model, name)(ids), _site(model, name)(many)
    offload_ple(model, str(tmp_path), 1.0)
    reads, read, in_core = [], zoo_offload.read_ranges, zoo_offload._in_core
    monkeypatch.setattr(zoo_offload, "read_ranges", lambda jobs: (reads.append(len(jobs)), read(jobs)))
    table = _site(model, name)
    fields = sum(len(t.fields) for t in table._tables[:1])
    for evicted in (False, False, True):
        if evicted:
            monkeypatch.setattr(zoo_offload, "_in_core", lambda mapped, rows, row_bytes: rows >= 10)
        np.testing.assert_array_equal(_bits(table(ids)), _bits(expected))
    assert reads == [4 * fields, 0, 2 * fields]
    # More seen rows than the sample: a spread sample of every field checked row by row; all rows when one
    # was evicted (here late rows of the scales and biases).
    table(many)
    for evicted in (False, True):
        checks = []  # (field, rows checked, first row, last row)
        monkeypatch.setattr(zoo_offload, "_in_core", lambda mapped, rows, row_bytes: (checks.append((id(mapped), rows.size, rows[0], rows[-1])), (rows < 100) | (row_bytes > 8) | (not evicted))[1])
        np.testing.assert_array_equal(_bits(table(many)), _bits(expected_many))
        assert (max(check[1] for check in checks) > 1) == evicted
        if not evicted:  # every field of every shard, from its first seen rows to its last
            assert len({check[0] for check in checks}) == sum(len(shard.fields) for shard in table._tables)
            assert min(check[2] for check in checks) < 10 and max(check[3] for check in checks) > 130
    import mmap
    pages = np.frombuffer(mmap.mmap(-1, 6 * mmap.PAGESIZE), dtype = np.uint8)
    pages[[2 * mmap.PAGESIZE + 5, 3 * mmap.PAGESIZE, 5 * mmap.PAGESIZE]] = 1
    assert in_core(pages, np.array([0, 2, 3]), mmap.PAGESIZE).tolist() == [False, True, True]
    assert in_core(pages, np.array([0, 1]), 3 * mmap.PAGESIZE).tolist() == [False, False]


def test_offloaded_rows_are_never_loaded(tmp_path):
    rows, dims = 8192, 2048  # 64 MiB of float32
    model = _lazy_copy(lambda: _holder("embed_tokens_per_layer", nn.Embedding(rows, dims)), str(tmp_path))
    site = model.language_model.model
    mx.synchronize()
    mx.clear_cache()
    before = mx.get_active_memory()
    mx.reset_peak_memory()
    offload_ple(model, str(tmp_path), 1.0)
    mx.eval(site.embed_tokens_per_layer(mx.array([[0, rows - 1, 4321]])))
    assert mx.get_peak_memory() - before < rows * dims * 4 // 16
    assert site.embed_tokens_per_layer(mx.zeros((1, 0), mx.int32)).shape == (1, 0, dims)
    for outside in (-1, rows):
        with pytest.raises(IndexError, match = "outside the table"):
            site.embed_tokens_per_layer(mx.array([outside]))


@pytest.mark.parametrize("cast", [True, False])
def test_offloaded_floats_get_the_load_cast(tmp_path, cast):
    def build():
        model = _holder("embed_tokens_per_layer", nn.QuantizedEmbedding.from_embedding(
            _with_negative_zeros(nn.Embedding(300, 64)), group_size = 32, bits = 4,
        ))
        model.cast_predicate = lambda _key: cast
        return model

    stock = _lazy_copy(build, str(tmp_path))
    offloaded = _lazy_copy(build, str(tmp_path))
    _convert_mlx_dtype(stock, mx.float32)
    report = offload_ple(offloaded, str(tmp_path), 0.5, cast_dtype = mx.float32)
    for ids in ([[5, 299, 150, 149, 0]], [[5, 149, 0]]):  # mixed, then held rows only
        expected = _site(stock, "embed_tokens_per_layer")(mx.array(ids))
        actual = _site(offloaded, "embed_tokens_per_layer")(mx.array(ids))
        assert expected.dtype == actual.dtype == (mx.float32 if cast else mx.bfloat16)
        np.testing.assert_array_equal(_bits(actual), _bits(expected))
    held = _site(offloaded, "embed_tokens_per_layer")._tables[0].held
    assert report["resident_bytes"] == sum(column.nbytes for column in held) > 0


def _moe(lib, dims = 512, hidden = 32, scales = mx.bfloat16, eval = True, quantized = True, experts = 64, v41 = False):
    """One MoE layer of 64 quantized experts, top-4, from mlx-vlm's or mlx-lm's ``SwitchGLU``; 512
    wide, as mlx-vlm's decode-block kernels require."""
    switch_layers = pytest.importorskip(f"mlx_{lib}.models.switch_layers")
    mx.random.seed(0)
    block = nn.Module()
    block.top_k = 4
    v41 = v41 and pytest.importorskip("mlx_vlm.models.deepseek_v41.language")
    block.switch_mlp = v41.DeepseekV41SwitchGLU(dims, hidden, experts, activation = v41.DeepseekV41SwiGLU(10.0)) if v41 else switch_layers.SwitchGLU(dims, hidden, experts)
    model = nn.Module()
    model.layers = [block]
    if not quantized:
        return model
    nn.quantize(block, group_size = 32, bits = 4, class_predicate = lambda _, module: hasattr(module, "to_quantized"))
    for name in ("gate_proj", "up_proj", "down_proj"):
        linear = block.switch_mlp[name]
        linear.scales, linear.biases = linear.scales.astype(scales), linear.biases.astype(scales)
    model.train(not eval)
    return model


def _routes(shape, seed, among = 64):
    """Top-4 of 64 experts per position, drawn from the first ``among``."""
    logits = mx.random.normal((*shape, 64), key = mx.random.key(seed))
    logits = mx.where(mx.arange(64) < among, logits + 100, logits)
    return mx.argpartition(logits, kth = -4, axis = -1)[..., -4:]


def _moe_calls(dtype):
    # A decode block whose positions route to disjoint experts, so over 8 slots its last chunk holds
    # one position; then (batch, length, experts routed among): decode, batched decode, decode blocks
    # (mlx-vlm's block kernels; one position at a time, here on gather_qmm_rhs, once batch x top-k
    # reaches 64), prefill on the per-route kernel, prefill on gather_qmm_rhs (from exactly 4 routes
    # per expert), prefill among 12 experts, which a 16-slot table alone would run on gather_qmm_rhs
    # and 64 experts do not, and a batch's decode blocks under 64 routes.
    disjoint = mx.arange(12, dtype = mx.uint32).reshape(1, 3, 4)
    calls = [(mx.random.normal((1, 3, 512), key = mx.random.key(99)).astype(dtype), disjoint)]
    shapes = [(1, 1, 64), (8, 1, 64), (1, 4, 64), (64, 2, 64), (1, 20, 64), (1, 64, 64), (2, 300, 64), (1, 30, 12), (8, 2, 64)]
    calls += [
        (mx.random.normal((batch, length, 512), key = mx.random.key(i)).astype(dtype), _routes((batch, length), 100 + i, among))
        for i, (batch, length, among) in enumerate(shapes)
    ]
    # gather_qmm_rhs prefill whose rarely routed experts leave a chunk too small for that kernel.
    skewed = mx.concatenate([_routes((1, 290), 7, 12), _routes((1, 10), 8)], axis = 1)
    return calls + [(mx.random.normal((1, 300, 512), key = mx.random.key(9)).astype(dtype), skewed)]


def _verifier_steps(glu, x, indices):
    """mlx-vlm's batch-invariant verifier runs the projections itself, unsorted; then a down
    projection of another input."""
    x, indices = mx.expand_dims(x.reshape(-1, x.shape[-1]), (-2, -3)), indices.reshape(-1, 4)
    up = glu.up_proj(x, indices, sorted_indices = False)
    gate = glu.gate_proj(x, indices, sorted_indices = False)
    activated = glu.activation(up, gate)
    down = glu.down_proj(activated, indices, sorted_indices = False)
    return mx.concatenate([down, glu.down_proj(activated * 2, indices, sorted_indices = False)])


@pytest.mark.parametrize("lib", ["vlm", "lm"])
@pytest.mark.parametrize("fraction", [1.0, 0.75, 0.1])
def test_offloaded_experts_match_stock(tmp_path, lib, fraction):
    stock = _lazy_copy(lambda: _moe(lib), str(tmp_path))
    calls = _moe_calls(mx.bfloat16)
    expected = [stock.layers[0].switch_mlp(*call) for call in calls]
    expected += [_verifier_steps(stock.layers[0].switch_mlp, *call) for call in calls[5:7]]
    table_bytes = sum(value.nbytes for _, value in tree_flatten(stock.parameters()))

    model = _lazy_copy(lambda: _moe(lib), str(tmp_path))
    report = offload_experts(model, str(tmp_path), fraction)

    glu = model.layers[0].switch_mlp
    assert isinstance(glu, OffloadSwitchGLU) and not tree_flatten(model.parameters())
    assert report["layers"] == 1 and report["on_disk_bytes"] == table_bytes
    slots = max(8, int(64 * (1 - fraction)))
    assert glu._slots.views is not None
    assert glu._slots.capacity == slots and report["resident_bytes"] == table_bytes * slots // 64
    # Only a downstream expression of each output is kept, while later calls evict its experts.
    actual = [glu(*call) * 1 for call in calls]
    actual += [_verifier_steps(glu, *call) for call in calls[5:7]]
    for got, want in zip(actual, expected):
        assert got.shape == want.shape
        np.testing.assert_array_equal(_bits(got), _bits(want))


def test_decode_blocks_follow_the_installed_mlx_vlm(monkeypatch):
    switch_layers = pytest.importorskip("mlx_vlm.models.switch_layers")
    glu = _moe("vlm").layers[0].switch_mlp
    short = mx.zeros((1, 2, 512), mx.bfloat16)
    monkeypatch.setattr(switch_layers, "DECODE_BLOCK_SIZE", 8, raising = False)  # as from 0.7.1
    assert zoo_offload._decode_block(glu, short)
    monkeypatch.delattr(switch_layers, "DECODE_BLOCK_SIZE")  # as before mlx-vlm 0.7.1
    assert not zoo_offload._decode_block(glu, short)


def test_offloaded_experts_run_on_another_thread(tmp_path):
    # Loaded here and generated on a server's request thread, which has its own MLX stream.
    stock = _lazy_copy(lambda: _moe("lm"), str(tmp_path))
    model = _lazy_copy(lambda: _moe("lm"), str(tmp_path))
    offload_experts(model, str(tmp_path), 1.0)
    calls = _moe_calls(mx.bfloat16)[:4]
    expected = [stock.layers[0].switch_mlp(*call) for call in calls]
    mx.eval(calls, expected)
    results = []

    def run():
        with mx.stream(mx.new_stream(mx.gpu)):
            for call in calls:
                out = model.layers[0].switch_mlp(*call)
                mx.eval(out)
                results.append(_bits(out))

    thread = threading.Thread(target = run)
    thread.start()
    thread.join(timeout = 120)
    assert len(results) == len(calls)
    for got, want in zip(results, expected):
        np.testing.assert_array_equal(got, _bits(want))


def test_a_request_threads_lazy_output_does_not_fail_the_next(tmp_path):
    # A prefill drops its last layer's output unevaluated; the next request runs on another thread.
    stock = _lazy_copy(lambda: _moe("lm"), str(tmp_path))
    model = _lazy_copy(lambda: _moe("lm"), str(tmp_path))
    offload_experts(model, str(tmp_path), 0.75)
    glu = model.layers[0].switch_mlp
    calls = _moe_calls(mx.bfloat16)
    expected = [stock.layers[0].switch_mlp(*call) for call in calls[1:4]]
    mx.eval(calls, expected)
    results, errors = [], []

    def run(evaluate, *chosen):
        try:
            with mx.stream(mx.new_stream(mx.gpu)):
                for call in chosen:
                    out = glu(*call)
                    if evaluate:
                        mx.eval(out)
                        results.append(_bits(out))
        except Exception as error:
            errors.append(error)

    misses = []
    for args in ((False, calls[0]), (True, *calls[1:4])):
        misses.append(glu._slots.misses)
        thread = threading.Thread(target = run, args = args)
        thread.start()
        thread.join(timeout = 120)
    assert not errors and len(results) == 3 and glu._slots.misses > misses[1]
    for got, want in zip(results, expected):
        np.testing.assert_array_equal(got, _bits(want))


@pytest.mark.parametrize("v41", [False, True])
def test_experts_a_checkpoint_stores_apart(tmp_path, v41):
    # As GLM-5's: one tensor per expert, which mlx-vlm stacks at load; DeepSeek V4.1's, under reference names.
    stock = _moe("vlm" if v41 else "lm", v41 = v41)
    stacked = dict(tree_flatten(stock.parameters()))
    names = {p: (p, n) for p, n in (("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3"))} if v41 else {}
    for shard in range(2):  # an expert's tensors may sit in any shard
        mx.save_safetensors(str(tmp_path / f"model-{shard}.safetensors"), {
            name.replace("switch_mlp.", f"experts.{expert}.").replace(*names.get(name.split(".")[-2], ("", ""))): value[expert]
            for name, value in stacked.items() for expert in range(shard, value.shape[0], 2)
        })
    (tmp_path / "config.json").write_text(json.dumps({"num_experts_per_tok": 4}))
    model = _moe("vlm" if v41 else "lm", v41 = v41).load_weights(list(stacked.items()))
    table_bytes = sum(value.nbytes for value in stacked.values())
    inventory = offload_inventory(str(tmp_path))
    assert (inventory["expert_bytes"], inventory["experts_per_layer"]) == (table_bytes, 64)
    report = offload_experts(model, str(tmp_path), 0.75)
    assert report["layers"] == 1 and report["on_disk_bytes"] == table_bytes
    for x, indices in _moe_calls(mx.bfloat16):  # V4.1's on 3 routes a token, not the layer's top-k
        indices = indices[..., :3] if v41 else indices
        call = (x, indices, mx.random.uniform(shape = indices.shape, key = mx.random.key(indices.size)))[:3 if v41 else 2]
        np.testing.assert_array_equal(_bits(model.layers[0].switch_mlp(*call)), _bits(stock.layers[0].switch_mlp(*call)))


def _qwen3_moe(mlp_only_layers = (), bits = lambda path: 4):
    qwen3_moe = pytest.importorskip("mlx_lm.models.qwen3_moe")
    mx.random.seed(0)
    model = qwen3_moe.Model(qwen3_moe.ModelArgs(
        model_type = "qwen3_moe", hidden_size = 256, num_hidden_layers = 3, intermediate_size = 256,
        num_attention_heads = 4, num_experts = 16, num_experts_per_tok = 4, decoder_sparse_step = 1,
        mlp_only_layers = list(mlp_only_layers), moe_intermediate_size = 64, rms_norm_eps = 1e-6, vocab_size = 512,
        num_key_value_heads = 2, head_dim = 64, rope_theta = 1e6, tie_word_embeddings = False,
        max_position_embeddings = 8192, norm_topk_prob = True,
    ))
    nn.quantize(model, class_predicate = lambda path, module: (type(module).__name__ == "QuantizedSwitchLinear" or hasattr(module, "num_experts"))
                and {"group_size": 32, "bits": bits(path)})
    model.eval()
    return model


@pytest.mark.parametrize("apart", [False, True])
def test_a_wide_prefill_step_runs_layer_by_layer_on_full_tables(tmp_path, monkeypatch, apart):
    # Runs of adjacent experts read in pieces that end mid-expert.
    monkeypatch.setattr(zoo_offload, "_RUN_BYTES", 3000)
    cache_module = pytest.importorskip("mlx_lm.models.cache")
    tokens = mx.random.randint(0, 512, (1, 2 * 2048 + 5), key = mx.random.key(3))
    stock = _lazy_copy(_qwen3_moe, str(tmp_path))
    if apart:  # one tensor per expert, half the experts in each of two shards, with gaps between them
        stacked = mx.load(str(tmp_path / "model.safetensors"))
        os.remove(tmp_path / "model.safetensors")
        for shard in range(2):
            mx.save_safetensors(str(tmp_path / f"model-{shard}.safetensors"), {
                name.replace("switch_mlp.", f"experts.{expert}."): value[expert]
                for name, value in stacked.items() if "switch_mlp." in name for expert in range(shard * 8, shard * 8 + 8)
            } | ({name: value for name, value in stacked.items() if "switch_mlp." not in name} if shard else {}))
    expected_cache = cache_module.make_prompt_cache(stock)
    expected = []
    copy = lambda c: [_bits(a) if isinstance(a, mx.array) else a for _, a in tree_flatten(c.state)]
    for start in range(0, tokens.shape[1], 2048):
        expected.append(stock.model(tokens[:, start:start + 2048], cache = expected_cache))
        mx.eval(expected[-1], [c.state for c in expected_cache])
        if start == 0:
            first = [copy(c) for c in expected_cache]
        if start == 2048:
            second = [copy(c) for c in expected_cache]

    model = _qwen3_moe().load_weights(list(tree_flatten(stock.parameters())))
    offload_experts(model, str(tmp_path), 0.75)
    switches = [layer.mlp.switch_mlp for layer in model.model.layers]
    assert all(isinstance(glu, OffloadSwitchGLU) and (glu._slots.fields[0][2].apart is not None) == apart for glu in switches)
    assert [glu._next for glu in switches] == switches[1:] + [None]
    # Some experts resident first, which the full tables copy from the slots.
    mx.eval(model(tokens[:, :3], cache = cache_module.make_prompt_cache(model)))
    slots = lambda: [(s.misses, s.calls, s.slot_of.tolist(), s.score.tolist()) for s in (glu._slots for glu in switches)]
    before = slots()
    assert all(misses for misses, *_ in before)
    reads, read = [], zoo_offload.read_ranges
    monkeypatch.setattr(zoo_offload, "read_ranges", lambda jobs: (reads.extend(len(view) for *_, view in jobs), read(jobs)))
    tables, table = [], OffloadSwitchGLU._table
    monkeypatch.setattr(OffloadSwitchGLU, "_table", lambda self, spare = None: (made := table(self, spare), tables.append(
        spare is not None and made[0].down_proj["weight"] is spare["down_proj", "weight"]))[0])
    cache = cache_module.make_prompt_cache(model)
    assert zoo_offload.wide_prefill_step(model) == 2048 and zoo_offload.wide_prefill_step(stock) is None
    # Each layer's cache as a 2048-chunked prefill holds it after two chunks.
    with zoo_offload.capture_prefill_rows(4096, copy) as captured:
        hidden = model.model(tokens, cache = cache)
        mx.eval(hidden, [c.state for c in cache])
    assert list(captured) == [id(c) for c in cache]
    # The last layer refilled the first's table, and none is left spare.
    assert tables == [False, False, True] and all(glu._spare is None for glu in switches)
    # Each missing expert read once; adjacent experts of a stacked tensor merged, never those stored apart.
    missing = [(glu._slots.slot_of < 0).sum() for glu in switches]
    rows = [field.row_bytes for _, _, field in switches[0]._slots.fields]
    assert max(reads) <= 3000 and sum(reads) == sum(missing) * sum(rows)
    unmerged = sum(missing) * sum(-(-row // 3000) for row in rows)
    assert len(reads) == unmerged if apart else len(reads) < unmerged
    # Every expert came from a full table: the slots decode uses were left alone.
    assert slots() == before
    assert all(glu._ahead is None and glu._full is None for glu in switches)
    np.testing.assert_array_equal(_bits(hidden), _bits(mx.concatenate(expected, axis = 1)))
    for (_, got), (_, want) in zip(tree_flatten([c.state for c in cache]), tree_flatten([c.state for c in expected_cache]), strict = True):
        np.testing.assert_array_equal(*(_bits(a) if isinstance(a, mx.array) else a for a in (got, want)))
    step = tokens[:, -1:]
    with zoo_offload.capture_prefill_rows(2048, copy) as narrow:
        np.testing.assert_array_equal(_bits(model(step, cache = cache)), _bits(stock(step, cache = expected_cache)))
    assert not narrow
    # A call no wider than a step runs whole and fills a request it ends at.
    with zoo_offload.capture_prefill_rows(2048, copy) as short:
        mx.eval(model.model(tokens[:, :2048], cache = cache_module.make_prompt_cache(model)))
    for request, want in ((captured, second), (short, first)):
        assert len(request) == len(cache)
        for got, held in zip(request.values(), want, strict = True):
            assert all(np.array_equal(a, b) for a, b in zip(got, held, strict = True))
    # Without a causal mask the layer attends to the whole call, which consecutive steps cannot.
    x = stock.model.embed_tokens(tokens)
    layers = (model.model.layers[0], stock.model.layers[0])
    full = mx.ones((x.shape[1], x.shape[1]), dtype = mx.bool_)
    monkeypatch.setattr(type(layers[0]), "_wide_masks", (None, "causal"))
    rotating = lambda: cache_module.RotatingKVCache(2048)
    scores = rotating().make_mask(x.shape[1], return_array = False, window_size = None).astype(mx.float32)  # additive, not kept
    for mask, cache in ((None, cache_module.KVCache), (full, cache_module.KVCache), (full, rotating), (scores, rotating)):
        got, want = (layer(x, mask, cache()) for layer in layers)
        np.testing.assert_array_equal(_bits(got), _bits(want))
    monkeypatch.setattr(model.model.layers[-1], "__class__", type(stock.model.layers[-1]))
    monkeypatch.setattr(model.model.layers[-1], "mlp", stock.model.layers[-1].mlp)
    assert zoo_offload.wide_prefill_step(model) is None  # an unwidened layer runs a call whole


def test_a_failed_wide_pass_leaves_no_table_behind(tmp_path, monkeypatch):
    model = _qwen3_moe().load_weights(list(tree_flatten(_lazy_copy(_qwen3_moe, str(tmp_path)).parameters())))
    offload_experts(model, str(tmp_path), 0.75)
    switches = [layer.mlp.switch_mlp for layer in model.model.layers]
    for glu in switches[:2]:  # a pass aborted after the second layer: the third holds its read-ahead
        assert glu._wide(read_next = True)
        glu._narrow()
    assert switches[2]._ahead is not None
    table = OffloadSwitchGLU._table

    def fail(self, spare = None):
        if self is switches[1]:
            raise MemoryError
        return table(self, spare)

    monkeypatch.setattr(OffloadSwitchGLU, "_table", fail)
    with pytest.raises(MemoryError):
        switches[0]._wide(read_next = True)
    # The retry dropped the stale read-ahead, and its failed read-ahead left the first layer on its slots.
    assert switches[2]._ahead is None and switches[0]._full is None


@pytest.mark.parametrize("window", [None, 2048])  # a rotating cache past its window masks with an array
def test_a_layer_without_a_table_steps_on_its_slots(tmp_path, monkeypatch, window):
    cache_module = pytest.importorskip("mlx_lm.models.cache")
    tokens = mx.random.randint(0, 512, (1, 2 * 2048 + 5), key = mx.random.key(4))
    stock = _lazy_copy(lambda: _qwen3_moe([0]), str(tmp_path))
    expected_cache = cache_module.make_prompt_cache(stock, max_kv_size = window)
    expected = [stock.model(tokens[:, i:i + 2048], cache = expected_cache) for i in range(0, tokens.shape[1], 2048)]
    model = _qwen3_moe([0]).load_weights(list(tree_flatten(stock.parameters())))
    offload_experts(model, str(tmp_path), 0.75)
    dense, cast, _ = model.model.layers
    assert isinstance(dense, zoo_offload._WideLayer) and dense._offload_switch is None
    assert zoo_offload.wide_prefill_step(model) == 2048
    monkeypatch.setattr(cast.mlp.switch_mlp, "_table", lambda spare = None: None)
    cache = cache_module.make_prompt_cache(model, max_kv_size = window)
    np.testing.assert_array_equal(_bits(model.model(tokens, cache = cache)), _bits(mx.concatenate(expected, axis = 1)))
    for (_, got), (_, want) in zip(tree_flatten([c.state for c in cache]), tree_flatten([c.state for c in expected_cache]), strict = True):
        np.testing.assert_array_equal(*(_bits(a) if isinstance(a, mx.array) else a for a in (got, want)))


def test_every_wide_layer_still_takes_its_token_arguments():
    for name, (axes, _) in zoo_offload._WIDE_LAYERS.items():
        module, _, layer = name.rpartition(".")
        assert {*axes, "cache"} <= set(inspect.signature(getattr(pytest.importorskip(module), layer).__call__).parameters), name
    assert zoo_offload._cache_mask(pytest.importorskip("mlx_lm.models.cache").ArraysCache(2), 4097) is None
    rotary = zoo_offload._token_slice(tuple(mx.arange(6).reshape(1, 6, 1) + shift for shift in (0, 10)), -2, 2, 4)
    assert [part.tolist() for part in rotary] == [[[[2], [3]]], [[[12], [13]]]]


def test_a_spare_table_of_another_layout_is_gone_before_the_next_is_made(tmp_path, monkeypatch):
    build = lambda: _qwen3_moe(bits = lambda path: 4 if ".layers.0." in path else 8)
    model = build().load_weights(list(tree_flatten(_lazy_copy(build, str(tmp_path)).parameters())))
    offload_experts(model, str(tmp_path), 0.75)
    alive, made, table = [], [], OffloadSwitchGLU._table

    def spy(self, spare = None):
        gc.collect()
        alive.append(sum(ref() is not None for ref in made))
        out = table(self, spare)
        made.append(weakref.ref(out[0].down_proj["weight"]))
        return out

    monkeypatch.setattr(OffloadSwitchGLU, "_table", spy)
    tokens = mx.random.randint(0, 512, (1, 2 * 2048 + 5), key = mx.random.key(3))
    mx.eval(model.model(tokens, cache = pytest.importorskip("mlx_lm.models.cache").make_prompt_cache(model)))
    # The 4-bit first layer's table fits neither 8-bit layer: two tables at most, as without reuse.
    assert alive == [0, 1, 1]
    alive.clear()
    switches = [layer.mlp.switch_mlp for layer in model.model.layers]
    switches[0]._wide(read_next = True)
    switches[0]._narrow()
    # A pass aborted after the first layer left the second its read-ahead and a spare: its retry holds neither.
    switches[0]._wide(read_next = True)
    assert alive == [0, 1, 0, 1]


def test_deepseek_v4_blocks_prefill_layer_by_layer(tmp_path):
    v4 = pytest.importorskip("mlx_vlm.models.deepseek_v4.deepseek_v4")
    lm = v4.Model(v4.ModelConfig(vocab_size = 512, hidden_size = 512, moe_intermediate_size = 64, num_hidden_layers = 4, n_routed_experts = 16, head_dim = 64,
        num_attention_heads = 4, q_lora_rank = 64, qk_rope_head_dim = 32, compress_ratios = [0, 4, 128, 0], o_groups = 2, o_lora_rank = 32, index_topk = 64)).language_model.eval()
    # Hash-routed blocks pick experts by token id: distinct ids, distinct experts.
    lm.load_weights([(f"model.layers.{i}.ffn.gate.tid2eid", mx.argsort(mx.random.uniform(shape = (512, 16)), axis = -1)[:, :6]) for i in range(3)], strict = False)
    nn.quantize(lm, group_size = 32, bits = 4, class_predicate = lambda _, module: type(module).__name__ == "SwitchLinear")
    mx.save_safetensors(str(tmp_path / "model.safetensors"), dict(tree_flatten(lm.parameters())))
    tokens = mx.random.randint(0, 512, (1, 2 * 2048 + 5), key = mx.random.key(3))
    expected = [lm.model(tokens[:, i:i + 2048], cache = cache, skip_final_norm = True) for cache in [lm.make_cache()] for i in range(0, tokens.shape[1], 2048)]
    offload_experts(lm, str(tmp_path), 0.75)
    assert zoo_offload.wide_prefill_step(lm) is None  # a call with image rows runs whole
    np.testing.assert_array_equal(_bits(lm.model(tokens, cache = lm.make_cache(), skip_final_norm = True)), _bits(mx.concatenate(expected, axis = 1)))
    assert [block.ffn.switch_mlp._slots.misses for block in lm.model.layers] == [0] * 4  # full tables, the slots untouched


def test_offloaded_experts_are_never_loaded(tmp_path):
    model = _lazy_copy(lambda: _moe("lm", hidden = 512), str(tmp_path))
    table_bytes = sum(value.nbytes for _, value in tree_flatten(model.parameters()))
    mx.synchronize()
    mx.clear_cache()
    before = mx.get_active_memory()
    mx.reset_peak_memory()
    offload_experts(model, str(tmp_path), 1.0)
    mx.eval(model.layers[0].switch_mlp(mx.ones((1, 1, 512), mx.bfloat16), _routes((1, 1), 0)))
    assert mx.get_peak_memory() - before < table_bytes // 4


def test_offloaded_expert_floats_get_the_load_cast(tmp_path):
    # Loaded in training mode and switched to eval afterwards, as generation does.
    stock = _lazy_copy(lambda: _moe("vlm", scales = mx.float16, eval = False), str(tmp_path))
    offloaded = _lazy_copy(lambda: _moe("vlm", scales = mx.float16, eval = False), str(tmp_path))
    _convert_mlx_dtype(stock, mx.bfloat16)
    offload_experts(offloaded, str(tmp_path), 0.75, cast_dtype = mx.bfloat16)
    assert offloaded.layers[0].switch_mlp._slots.views is None
    stock.eval()
    offloaded.eval()
    for x, indices in _moe_calls(mx.bfloat16)[::2]:
        expected = stock.layers[0].switch_mlp(x, indices)
        actual = offloaded.layers[0].switch_mlp(x, indices)
        assert actual.dtype == expected.dtype == mx.bfloat16
        np.testing.assert_array_equal(_bits(actual), _bits(expected))
    assert not offloaded.layers[0].switch_mlp._slots.readers


def test_slot_readers_kept_are_bounded_in_bytes(monkeypatch):
    monkeypatch.setattr(zoo_offload, "_PENDING_BYTES", 1000)
    slots = zoo_offload._ExpertSlots([], 4, 2)
    for _ in range(3):
        slots.read_by(mx.zeros((100,), dtype = mx.float32) + 1)
    assert len(slots.readers) == 0


def test_concurrent_offloaded_loads_restore_the_cache_limit(monkeypatch):
    # Two loads finishing at once must not both take the cap as the first model's: the second
    # would save the capped limit as the one to restore.
    import unsloth_zoo.mlx.loader as zoo_loader

    state = {"limit": 8 << 30}
    both_first = threading.Barrier(2, timeout = 1)

    def set_cache_limit(value):
        if value == zoo_loader._OFFLOAD_CACHE_LIMIT:
            try:
                both_first.wait()
            except threading.BrokenBarrierError:
                pass
        state["limit"], old = value, state["limit"]
        return old

    monkeypatch.setattr(mx, "set_cache_limit", set_cache_limit)
    monkeypatch.setattr(zoo_loader, "_offload_cache_cap", {"models": 0, "previous": None, "applied": None})
    models = [type("Model", (), {})() for _ in range(2)]
    threads = [threading.Thread(target = zoo_loader._cap_cache_while_alive, args = (m,)) for m in models]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout = 10)
    assert state["limit"] == zoo_loader._OFFLOAD_CACHE_LIMIT
    del models, threads
    gc.collect()
    assert state["limit"] == 8 << 30


def _parts(ple = None, experts = None):
    return {"ple": ple, "experts": experts}


def test_offload_controls_and_refusals(tmp_path):
    for off in (None, False, 0, "off"):
        assert offload_setting(off) == _parts()
    assert offload_setting(0.25) == _parts(0.25, 0.25)
    assert offload_setting(1) == _parts(1.0, 1.0)
    assert offload_setting("auto") == offload_setting(True) == _parts("auto", "auto")
    assert offload_setting({"experts": 0.5}) == _parts(None, 0.5)
    assert offload_setting(print) is print
    for bad, name in ((1.5, "offload"), (-0.1, "offload"), ("on", "offload"), ("0.5", "offload"), ({"ple": 2}, "offload\\['ple'\\]")):
        with pytest.raises(ValueError, match = name + " must be"):
            offload_setting(bad)
    with pytest.raises(ValueError, match = "'ple' and 'experts'"):
        offload_setting({"expert": 0.5})

    for value in (0.5, {"experts": 0.5}):
        with pytest.raises(ValueError, match = "full_finetuning"):
            FastMLXModel.from_pretrained("unused", full_finetuning = True, offload = value)

    dense = tmp_path / "dense"
    dense.mkdir()
    model = _lazy_copy(lambda: _moe("vlm", quantized = False), str(dense))
    assert offload_experts(model, str(dense), 1.0) == {"layers": 0, "on_disk_bytes": 0, "resident_bytes": 0, "skipped": []}
    top_12 = tmp_path / "top_12"
    top_12.mkdir()
    model = _lazy_copy(lambda: _moe("lm"), str(top_12))
    model.layers[0].top_k = 12  # the slots hold at least one token's routes
    offload_experts(model, str(top_12), 1.0)
    assert model.layers[0].switch_mlp._slots.capacity == 12
    model = _lazy_copy(lambda: _moe("lm"), str(top_12))
    model.layers[0].gate = nn.Module()  # on the router, as GLM-5 and DeepSeek V4 keep it
    model.layers[0].gate.top_k, model.layers[0].top_k = 12, None
    offload_experts(model, str(top_12), 1.0)
    assert model.layers[0].switch_mlp._slots.capacity == 12
    # A planned fraction keeps exactly its whole experts, whatever the float rounding.
    sixty = tmp_path / "sixty"
    sixty.mkdir()
    model = _lazy_copy(lambda: _moe("lm", experts = 60), str(sixty))
    offload_experts(model, str(sixty), (60 - 25) / 60)
    assert model.layers[0].switch_mlp._slots.capacity == 25

    # An offloaded load caps MLX's buffer cache at 1 GiB (keeping a lower limit) while it lives;
    # one that offloads nothing leaves it alone.
    previous = mx.set_cache_limit(4 << 30)
    try:
        for limit, capped in ((4 << 30, 1 << 30), (512 << 20, 512 << 20)):
            mx.set_cache_limit(limit)
            models = [_lazy_copy(lambda: _moe("lm"), str(top_12)) for _ in range(2)]
            for offloaded in models:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")  # one knob for both: a part it lacks is not news
                    _apply_offload(offloaded, str(top_12), offload_setting(1.0), None, 2048)
            del offloaded
            assert _cache_limit() == capped
            del models[0]
            gc.collect()
            assert _cache_limit() == capped
            del models[0]
            gc.collect()
            assert _cache_limit() == limit
        mx.set_cache_limit(4 << 30)
        ple = _lazy_copy(lambda: _holder("embed_tokens_per_layer", nn.Embedding(16, 8)), str(tmp_path))
        _apply_offload(ple, str(tmp_path), _parts(ple = 1.0), None, 2048)
        assert _cache_limit() == 1 << 30
        del ple
        gc.collect()
        model = _lazy_copy(lambda: _moe("lm", quantized = False), str(dense))
        with pytest.warns(UserWarning, match = "offload is set, but this model has no PLE table or quantized MoE experts"):
            _apply_offload(model, str(dense), offload_setting(1.0), None, 2048)
        assert _cache_limit() == 4 << 30
        # A limit set since the cap, lower or higher, stays when the model goes.
        for newer in (256 << 20, 2 << 30):
            ple = _lazy_copy(lambda: _holder("embed_tokens_per_layer", nn.Embedding(16, 8)), str(tmp_path))
            _apply_offload(ple, str(tmp_path), _parts(ple = 1.0), None, 2048)
            mx.set_cache_limit(newer)
            del ple
            gc.collect()
            assert _cache_limit() == newer
    finally:
        mx.set_cache_limit(previous)

    model = _lazy_copy(lambda: _holder("embed_tokens_per_layer", nn.Embedding(16, 8)), str(tmp_path))
    model.update(tree_map(lambda value: value.astype(mx.bfloat16), model.parameters()))
    report = offload_ple(model, str(tmp_path), 1.0)  # the checkpoint holds float32, the model bf16
    assert report["tables"] == 0 and len(report["skipped"]) == 1
    assert isinstance(_site(model, "embed_tokens_per_layer"), nn.Embedding)

    model = _lazy_copy(lambda: _holder("embed_tokens_per_layer", nn.Embedding(16, 8)), str(tmp_path))
    model._unsloth_offload = {"ple": offload_ple(model, str(tmp_path), 1.0)}
    with pytest.raises(ValueError, match = "inference only"):
        FastMLXModel.get_peft_model(model, r = 8)
    with pytest.raises(ValueError, match = "inference only"):
        _mlx_save_pretrained_merged(model, str(tmp_path / "merged"), tokenizer = object())
    assert not (tmp_path / "merged").exists()


def test_offloadable(tmp_path):
    def build():
        model = _moe("lm")
        model.language_model = _holder(*_table("quantized")).language_model
        return model

    # What it counts is what the offload then takes, and in float32 the bytes the load would hold.
    model = _lazy_copy(build, str(tmp_path))
    found = offloadable(model, str(tmp_path))
    wide = lambda prefix: sum(
        v.size * 4 if mx.issubdtype(v.dtype, mx.floating) else v.nbytes
        for k, v in tree_flatten(model.parameters()) if k.startswith(prefix)
    )
    assert offloadable(model, str(tmp_path), cast_dtype = mx.float32) == {
        **found, "ple_bytes": wide("language_model"), "expert_bytes": wide("layers"),
        "expert_floor_bytes": wide("layers") * 8 // 64, "expert_layer_bytes": wide("layers"),
    }
    assert wide("layers") > found["expert_bytes"]
    ple, experts = (offload(model, str(tmp_path), 1.0) for offload in (offload_ple, offload_experts))
    assert found == {
        "ple_bytes": ple["on_disk_bytes"], "expert_bytes": experts["on_disk_bytes"],
        "expert_floor_bytes": experts["resident_bytes"], "experts_per_layer": 64,
        "expert_layer_bytes": experts["on_disk_bytes"], "ple": None, "experts": None,
    }
    # Experts the offload keeps in memory are not counted, with its reason.
    for change, reason in ((lambda block: delattr(block, "top_k"), "its routing top-k is unknown"),
                           (lambda block: setattr(block, "top_k", 64), "its 64 experts per layer all stay resident")):
        model = _lazy_copy(build, str(tmp_path))
        change(model.layers[0])
        found = offloadable(model, str(tmp_path))
        assert (found["expert_bytes"], found["experts"]) == (0, reason) and found["ple"] is None


def _inventory(ple = 30, experts = 64, floor = 2, per_layer = 64):
    return {
        "total_bytes": 0, "ple_bytes": ple, "expert_bytes": experts, "expert_floor_bytes": floor,
        "experts_per_layer": per_layer, "expert_layer_bytes": 0, "ple": None, "experts": None,
    }


@pytest.fixture
def units(monkeypatch):
    # Sizes in arbitrary units: PLE offload's working memory is 1, an expert offload's 2, no margin;
    # experts stay resident while 4096 tokens of context (16) fit beside them.
    monkeypatch.setattr(zoo_offload, "_OFFLOAD_CONTEXT", 4096)
    monkeypatch.setattr(zoo_offload, "_PLE_OVERHEAD", 1)
    monkeypatch.setattr(zoo_offload, "_CACHE_CAP", 2)
    monkeypatch.setattr(zoo_offload, "SAFETY_MARGIN", (0, 0))
    return monkeypatch


def _plan(budget, inventory = None, cost = lambda t, step: t // 256 + (step // 2048 - 1) * 4, **kwargs):
    # A unit per 256 tokens of context, 4 more per 2048 tokens of prefill step beyond the first.
    return plan_offload(
        budget = budget, resident = 4, inventory = inventory or _inventory(), context_cost = cost,
        ceiling = 8192, minimum = 1024, **kwargs,
    )


def test_offload_inventory(tmp_path):
    lazy = _lazy_copy(lambda: _moe("lm"), str(tmp_path))
    experts = sum(value.nbytes for _, value in tree_flatten(lazy.parameters()))
    table = mx.zeros((32, 8), mx.bfloat16)
    mx.save_safetensors(str(tmp_path / "extra.safetensors"), {
        "language_model.model.embed_tokens_per_layer.weight": table,
        "language_model.layers.1.engram.embed.weight": table,  # mlx-vlm maps it, never resident
        # the drafter head is never offloaded
        **{f"mtp.layers.0.switch_mlp.{p}.{f}": table for p in ("gate_proj", "up_proj", "down_proj") for f in ("weight", "scales")},
    })
    config = {"text_config": {"num_experts_per_tok": 4}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    inventory = offload_inventory(str(tmp_path))
    assert inventory == {
        "total_bytes": experts + 7 * table.nbytes, "ple_bytes": table.nbytes, "expert_bytes": experts,
        "expert_floor_bytes": experts * 8 // 64, "experts_per_layer": 64, "expert_layer_bytes": experts,
        "ple": None, "experts": None,
    }
    # Loaded at four bytes an element, floating-point tensors (bf16 here) double; packed weights do not.
    widened = sum(v.size * 4 if mx.issubdtype(v.dtype, mx.floating) else v.nbytes for _, v in tree_flatten(lazy.parameters()))
    inventory = offload_inventory(str(tmp_path), 4)
    assert (inventory["ple_bytes"], inventory["expert_bytes"]) == (2 * table.nbytes, widened) and widened > experts

    (tmp_path / "config.json").write_text("{}")
    assert offload_inventory(str(tmp_path))["experts"] == "its routing top-k is unknown"
    dense = tmp_path / "dense"
    dense.mkdir()
    _lazy_copy(lambda: _moe("lm", quantized = False), str(dense))
    (dense / "config.json").write_text(json.dumps(config))
    inventory = offload_inventory(str(dense))
    assert inventory["experts"] == "its routed experts are not quantized" and inventory["expert_bytes"] == 0
    assert inventory["ple"] == "the model has no per-layer embedding table"
    # offload_experts keeps experts with global scales in memory, so they are not offloadable.
    scaled = tmp_path / "scaled"
    scaled.mkdir()
    lazy = _lazy_copy(lambda: _moe("lm"), str(scaled))
    up = next(k for k, _ in tree_flatten(lazy.parameters()) if k.endswith("up_proj.weight"))
    mx.save_safetensors(str(scaled / "scale.safetensors"), {up.replace(".weight", ".global_scale"): mx.ones((64,))})
    (scaled / "config.json").write_text(json.dumps(config))
    inventory = offload_inventory(str(scaled))
    assert inventory["experts"] == "its experts have global scales" and inventory["expert_bytes"] == 0
    # A layer counts only with all three projections quantized.
    partial = tmp_path / "partial"
    partial.mkdir()
    mx.save_safetensors(str(partial / "model.safetensors"), {
        k: v for k, v in tree_flatten(lazy.parameters()) if not k.startswith(up.replace("up_proj.weight", "down_proj.s"))
    })
    (partial / "config.json").write_text(json.dumps(config))
    assert offload_inventory(str(partial))["experts"] == "its routed experts are not quantized"


def test_plan_offload_rule(units):
    plan = lambda ple, experts, context, step: {"ple": ple, "experts": experts, "context": context, "prefill_step": step}
    # 8192 tokens (32 units) is the ceiling; without offloaded experts the step stays 2048.
    assert _plan(200) == plan(None, None, 8192, 2048)
    assert _plan(4 + 32 + 94) == plan(None, None, 8192, 2048)
    assert _plan(4 + 32 + 93) == plan(1.0, None, 8192, 2048)
    # Experts stay resident (65 with PLE offloaded) while 4096 tokens fit beside them, and the
    # context then takes all that is left.
    assert _plan(4 + 65 + 16) == plan(1.0, None, 4096, 2048)
    assert _plan(4 + 65 + 23) == plan(1.0, None, 5888, 2048)
    assert _plan(4 + 65 + 23, _inventory(ple = 4)) == plan(None, None, 5120, 2048)  # the least offload
    # Short of that, experts pay for 4096 tokens, in whole experts: 80 - 16 - PLE overhead 1 -
    # expert overhead 2 = 61 of 64 stay at step 2048 (3/2048 per token; 7/4096 at step 4096).
    assert _plan(4 + 65 + 15) == plan(1.0, 3 / 64, 4096, 2048)
    # Whole experts of 4 units each leave slack the context then takes: 24 - (1 + 4 + 2) = 17 units.
    assert _plan(28, _inventory(per_layer = 16)) == plan(1.0, 15 / 16, 4352, 2048)
    assert _plan(4 + 65 + 15, context = 8192) == plan(1.0, 31 / 64, 8192, 8192)  # a pinned context is kept
    # 40 units of room: at step 2048, 24 - 3 = 21 stay (43/2048 per token); at 4096, 17 stay
    # (47/4096: fewer re-read bytes); 8192 is past the context.
    assert _plan(44) == plan(1.0, 47 / 64, 4096, 4096)
    assert _plan(44, prefill_step = 2048) == plan(1.0, 43 / 64, 4096, 2048)
    # The context experts pay for stays within the minimum and the ceiling.
    units.setattr(zoo_offload, "_OFFLOAD_CONTEXT", 512)
    assert _plan(44) == plan(1.0, 31 / 64, 1024, 2048)
    units.setattr(zoo_offload, "_OFFLOAD_CONTEXT", 16384)
    assert _plan(4 + 65 + 15) == plan(1.0, 31 / 64, 8192, 8192)
    units.setattr(zoo_offload, "_OFFLOAD_CONTEXT", 4096)
    # A pinned context keeps its size; with the PLE knob off, experts alone pay; 8192 does not fit.
    assert _plan(4 + 32 + 42, context = 8192, offload = _parts(experts = "auto")) == plan(None, 58 / 64, 8192, 4096)
    assert _plan(40, context = 8192) == plan(None, None, None, None)
    # Nothing fits next to half the room: context drops to what the smallest footprint leaves.
    assert _plan(24, _inventory(floor = 8)) == plan(1.0, 56 / 64, 2304, 2048)
    assert _plan(18, _inventory(floor = 8)) == plan(None, None, None, None)
    # The safety margin, absolute or a fraction of the estimate, comes off the budget first.
    units.setattr(zoo_offload, "SAFETY_MARGIN", (10, 0))
    assert _plan(4 + 32 + 94 + 10) == plan(None, None, 8192, 2048)
    assert _plan(4 + 32 + 94 + 9) == plan(1.0, None, 8192, 2048)
    units.setattr(zoo_offload, "SAFETY_MARGIN", (0, 0.03))
    assert _plan(134) == plan(None, None, 8192, 2048)
    assert _plan(133) == plan(1.0, None, 8192, 2048)
    units.setattr(zoo_offload, "SAFETY_MARGIN", (0, 0))
    # A hand-set fraction is kept; only the auto part and the step are chosen.
    assert _plan(200, offload = _parts("auto", 0.5)) == plan(None, 0.5, 8192, 8192)
    # What a resolved offload holds: PLE rows and experts kept (never under the floor) plus working memory.
    inventory = _inventory()
    assert offload_footprint(inventory, None, None) == 30 + 64
    assert offload_footprint(inventory, 1.0, 0.5) == 1 + 32 + 2
    assert offload_footprint(inventory, 0.5, 1.0) == 15 + 1 + 2 + 2
    assert offload_footprint({**inventory, "ple": "no table", "expert_bytes": 0}, 1.0, 1.0) == 0


def test_context_bytes(tmp_path):
    config = {"num_hidden_layers": 8, "full_attention_interval": 4, "num_key_value_heads": 2, "head_dim": 64, "vocab_size": 0}
    (tmp_path / "config.json").write_text(json.dumps({"text_config": config}))
    assert context_bytes(str(tmp_path), 1000) == 2 * 2 * 2 * 64 * 2 * 2 * 1000 + (2 << 30)
    # Omni models keep the decoder's config under the thinker; the talker's is not the one priced.
    talker = {**config, "num_key_value_heads": 64}
    (tmp_path / "config.json").write_text(json.dumps({"talker_config": {"text_config": talker}, "thinker_config": {"text_config": config}}))
    assert context_bytes(str(tmp_path), 1000) == 2 * 2 * 2 * 64 * 2 * 2 * 1000 + (2 << 30)
    (tmp_path / "config.json").write_text(json.dumps({"vision_config": config}))
    with pytest.raises(ValueError, match = "cannot be priced"):
        context_bytes(str(tmp_path), 1000)
    config.update(layer_types = ["sliding_attention", "full_attention", "linear_attention"], sliding_window = 100, vocab_size = 10)
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert context_bytes(str(tmp_path), 1000, 4096) == 2 * 2 * 2 * 64 * 2 * (1000 + 100) + 4 * 4096 * 10 + (2 << 30)


def test_auto_offload(tmp_path, monkeypatch):
    # The rule keeps 4 GiB of margin and reserve, so a small runner's own budget fits nothing.
    device_info = mx.device_info
    monkeypatch.setattr(mx, "device_info", lambda: {**device_info(), "max_recommended_working_set_size": 64 << 30})
    model = _lazy_copy(lambda: _moe("lm"), str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"num_experts_per_tok": 4, "vocab_size": 16, "num_hidden_layers": 2, "num_key_value_heads": 2, "head_dim": 8}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing to offload, nothing said
        assert _auto_offload(model, str(tmp_path), _parts("auto", "auto"), 4096, mx.bfloat16) == _parts()
    seen = {}
    monkeypatch.setattr(zoo_offload, "plan_offload", lambda **k: seen.update(k) or {"ple": None, "experts": 0.75, "context": 4096, "prefill_step": 8192})
    with pytest.warns(UserWarning, match = r"offload 'auto' leaves 75% .*fit in .*prefill_step_size=8192.*'experts' to 'off'"):
        assert _auto_offload(model, str(tmp_path), _parts(0.5, "auto"), 4096, mx.float32) == _parts(0.5, 0.75)
    assert seen["offload"] == _parts(0.5, "auto")
    # The checkpoint's size and what this model can offload, at the dtype the load converts to.
    assert seen["inventory"] == {**offload_inventory(str(tmp_path), 4), **offloadable(model, str(tmp_path), cast_dtype = mx.float32)}
    assert seen["inventory"]["expert_bytes"] > offloadable(model, str(tmp_path))["expert_bytes"]
    assert seen["context_cost"](1000, 4096) == context_bytes(str(tmp_path), 1000, 4096, 4)
    monkeypatch.setattr(zoo_offload, "plan_offload", lambda **_: {"ple": None, "experts": None, "context": None, "prefill_step": None})
    with pytest.warns(UserWarning) as record:  # all that can go to disk does; there is no PLE table
        assert _auto_offload(model, str(tmp_path), _parts("auto", "auto"), 4096, mx.bfloat16) == _parts(None, 1.0)
    said = [str(w.message) for w in record]
    assert len(said) == 2 and "do not fit" in said[0] and "leaves 100%" in said[1] and "fit in" not in said[1]

    monkeypatch.setattr(zoo_offload, "plan_offload", lambda **k: seen.update(k) or {"ple": None, "experts": None, "context": 4096, "prefill_step": 2048})
    del model.layers[0].top_k  # the experts this model's offload keeps, the headers alone would count
    assert _auto_offload(model, str(tmp_path), _parts(None, "auto"), 4096, mx.bfloat16) == _parts()
    assert seen["inventory"]["expert_bytes"] == 0 < offload_inventory(str(tmp_path))["expert_bytes"]
    (tmp_path / "config.json").write_text(json.dumps({"num_experts_per_tok": 4}))  # a context it cannot price
    with pytest.warns(UserWarning, match = "cannot be priced. .*'auto' stays off"):
        assert _auto_offload(model, str(tmp_path), _parts(0.5, "auto"), 4096, mx.bfloat16) == _parts(0.5)

    # A caller's own plan decides in place of the rule, with the downloaded checkpoint.
    monkeypatch.setattr(zoo_offload, "plan_offload", None)
    asked = []
    fresh = _lazy_copy(lambda: _moe("lm"), str(tmp_path))
    _apply_offload(fresh, str(tmp_path), lambda *a: asked.append(a) or {"experts": 0.75}, None, 4096)
    assert asked == [(str(tmp_path),)] and set(fresh._unsloth_offload) == {"experts"}
    # What it leaves "auto" goes to the rule, which here cannot price the context.
    with pytest.warns(UserWarning, match = "cannot be priced. .*'auto' stays off"):
        _apply_offload(_lazy_copy(lambda: _moe("lm"), str(tmp_path)), str(tmp_path), lambda _: "auto", None, 4096)

    setting = _parts("auto", 0.5)
    assert _settle_offload(setting, False, "refused") is setting
    assert _settle_offload(_parts("auto"), True, "refused") == _settle_offload(print, True, "refused") == _parts()
    with pytest.raises(ValueError, match = "refused"):
        _settle_offload(setting, True, "refused")

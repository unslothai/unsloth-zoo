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

"""Disk-backed frozen tables: gather == F.embedding oracle, madvise ranges, fp8 + e8m0 planes, a
parameter-free module, memo reuse under gradient checkpointing, and a tiny Qwen4Exp load that never
reads the n-gram shards yet matches a normal load bit for bit."""

import json
import mmap
import os

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from unsloth_zoo import frozen_table_store as fts
from unsloth_zoo.frozen_table_store import (
    FROZEN_TABLE_STATS,
    FrozenTableStore,
    UnslothFrozenTableEmbedding,
    make_fp8_block_decode,
    reset_frozen_table_stats,
    resolve_tensor_locations,
)


def _write_sharded(tmp_path, shards, prefix="t", files=2, extra=True):
    """Write ``shards`` (list of tensors) as ``{prefix}.shard_i.weight`` across ``files`` safetensors
    files with an index json. Unrelated tensors are interleaved so offsets are not page aligned."""
    weight_map = {}
    per_file = [{} for _ in range(files)]
    for i, t in enumerate(shards):
        f = i % files
        per_file[f][f"{prefix}.shard_{i}.weight"] = t.contiguous()
        if extra:
            per_file[f][f"pad_{i}"] = torch.arange(7 + i, dtype=torch.int8)
    for f, tensors in enumerate(per_file):
        name = f"model-{f:05d}.safetensors"
        save_file(tensors, str(tmp_path / name))
        weight_map.update(dict.fromkeys(tensors, name))
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return str(index), [f"{prefix}.shard_{i}.weight" for i in range(len(shards))]


@pytest.fixture
def bf16_table(tmp_path):
    torch.manual_seed(0)
    sizes = [37, 64, 1, 50]
    shards = [torch.randn(n, 20).to(torch.bfloat16) for n in sizes]
    index, names = _write_sharded(tmp_path, shards)
    return index, names, torch.cat(shards)


def _ids_with_edges(full):
    n = full.shape[0]
    edges = torch.tensor([0, 36, 37, 100, 101, 102, n - 1, 0, 37, 37, n - 1])
    rand = torch.randint(0, n, (5, 13))
    return [edges, rand, edges.view(1, -1).repeat(3, 1), torch.empty(0, dtype=torch.long)]


@pytest.mark.parametrize("reader", ["mmap", "pread"])
@pytest.mark.parametrize("mode", ["disk", "ram", "gpu"])
def test_gather_matches_embedding_oracle(bf16_table, reader, mode):
    index, names, full = bf16_table
    store = FrozenTableStore.from_checkpoint(index, {"weight": names}, mode=mode, reader=reader)
    assert store.num_rows == full.shape[0]
    for ids in _ids_with_edges(full):
        got = store.gather(ids)
        ref = F.embedding(ids, full)
        assert got.dtype == torch.bfloat16 and got.shape == ref.shape
        assert torch.equal(got, ref)
        assert torch.equal(store.gather(ids, dtype=torch.float32), ref.float())
    with pytest.raises(IndexError):
        store.gather(torch.tensor([full.shape[0]]))
    with pytest.raises(IndexError):
        store.gather(torch.tensor([-1]))
    store.close()


def test_madvise_page_aligned_and_real(bf16_table, monkeypatch):
    index, names, _ = bf16_table
    locs = resolve_tensor_locations(index, names=names)
    assert any(loc.offset % mmap.PAGESIZE for loc in locs.values())  # the test must cover unaligned tensors
    calls = []
    real = fts._madvise

    def spy(mm, start, length, advice):
        calls.append((start, length, advice))
        return real(mm, start, length, advice)

    monkeypatch.setattr(fts, "_madvise", spy)
    store = FrozenTableStore.from_checkpoint(index, {"weight": names}, mode="disk")
    rand = [c for c in calls if c[2] == getattr(mmap, "MADV_RANDOM", None)]
    if hasattr(mmap, "MADV_RANDOM"):
        assert len(rand) == len(names)
        for start, length, _ in rand:
            assert start % mmap.PAGESIZE == 0
            covered = [loc for loc in locs.values() if start <= loc.offset and loc.offset + loc.nbytes <= start + length]
            assert covered, (start, length)
        # madvise really succeeded on this platform
        assert len(store.madvise_calls) == len(names)
    store.close()

    calls.clear()
    FrozenTableStore.from_checkpoint(index, {"weight": names}, mode="ram").close()
    if hasattr(mmap, "MADV_WILLNEED"):
        assert any(c[2] == mmap.MADV_WILLNEED and c[0] % mmap.PAGESIZE == 0 for c in calls)


def test_fp8_e8m0_planes_match_reference(tmp_path):
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("torch without float8_e8m0fnu")
    torch.manual_seed(0)
    n, dim, block = 300, 64, 32
    weight = (torch.randn(n, dim) * 4).to(torch.float8_e4m3fn)
    scale = torch.randint(110, 140, (n, dim // block), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    save_file({"layers.1.engram.embed.weight": weight, "x": torch.ones(3), "layers.1.engram.embed.scale": scale},
              str(tmp_path / "model.safetensors"))
    store = FrozenTableStore.from_checkpoint(
        str(tmp_path), {"weight": ["layers.1.engram.embed.weight"], "scale": ["layers.1.engram.embed.scale"]},
        decode=make_fp8_block_decode(block), mode="disk",
    )
    ids = torch.randint(0, n, (4, 9, 6))
    ids[0, 0, :3] = torch.tensor([0, n - 1, 0])
    # DeepseekV41EngramEmbedding.forward, fp8 branch
    values = F.embedding(ids, weight)
    scales = F.embedding(ids, scale).float()
    ref = (values.float().unflatten(-1, (-1, block)) * scales.unsqueeze(-1)).flatten(-2)
    got = store.gather(ids)
    assert got.dtype == torch.float32
    assert torch.equal(got, ref)
    store.close()


def test_module_has_no_tensors_and_memoizes_under_checkpoint(bf16_table):
    index, names, full = bf16_table
    emb = UnslothFrozenTableEmbedding(full.shape[0], 20, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="not bound"):
        emb(torch.tensor([1]))
    emb.bind(FrozenTableStore.from_checkpoint(index, {"weight": names}, mode="disk"))
    proj = torch.nn.Linear(20, 3)
    model = torch.nn.ModuleDict({"emb": emb, "proj": proj})
    assert [n for n, _ in model.named_parameters()] == ["proj.weight", "proj.bias"]
    assert list(model.named_buffers()) == []
    assert set(model.state_dict()) == {"proj.weight", "proj.bias"}
    assert emb.weight.is_meta
    model.to(torch.float64)
    assert emb.table_dtype == torch.float64
    model.to(torch.float32)

    tokens = torch.randint(0, 50, (2, 7))

    def block(x):
        ids = (tokens * 3 + 1) % full.shape[0]  # fresh tensor on every (re)compute, as in Qwen4Exp
        return proj(emb(ids) * x).sum()

    x = torch.ones(2, 7, 20, requires_grad=True)
    reset_frozen_table_stats()
    out = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
    out.backward()
    assert FROZEN_TABLE_STATS["gathers"] == 1
    assert FROZEN_TABLE_STATS["memo_hits"] == 1
    ref_rows = F.embedding((tokens * 3 + 1) % full.shape[0], full.float())
    assert torch.allclose(x.grad, ref_rows * proj.weight.sum(0))
    assert torch.equal(emb((tokens * 3 + 1) % full.shape[0]), ref_rows)
    # a new step or different ids is a miss
    fts.frozen_table_next_step()
    emb((tokens * 3 + 1) % full.shape[0])
    emb((tokens * 3 + 2) % full.shape[0])
    assert FROZEN_TABLE_STATS["gathers"] == 3
    assert proj.weight.grad is not None and x.grad is not None


def test_memo_never_serves_stale_rows_and_runs_under_inference_mode(bf16_table):
    index, names, full = bf16_table
    emb = UnslothFrozenTableEmbedding(full.shape[0], 20, dtype=torch.float32)
    emb.bind(FrozenTableStore.from_checkpoint(index, {"weight": names}, mode="disk"))
    ref = full.float()
    ids = torch.tensor([0, 3])
    assert torch.equal(emb(ids), F.embedding(ids, ref))
    ids.fill_(5)  # updated in place: the cached rows belong to the old values
    assert torch.equal(emb(ids), F.embedding(ids, ref))
    with torch.inference_mode():
        inf_ids = torch.tensor([[1, 2], [7, 7]])
        assert torch.equal(emb(inf_ids), F.embedding(inf_ids, ref))
        assert torch.equal(emb(inf_ids), F.embedding(inf_ids, ref))


def test_bind_errors_are_loud(bf16_table, tmp_path):
    index, names, full = bf16_table
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
    model.layers[1].emb = UnslothFrozenTableEmbedding(full.shape[0], 20)
    with pytest.raises(RuntimeError, match="could not find"):
        fts.bind_frozen_tables(model, index)
    store = FrozenTableStore.from_checkpoint(index, {"weight": names}, mode="disk")
    with pytest.raises(ValueError, match="rows"):
        UnslothFrozenTableEmbedding(full.shape[0] + 1, 20).bind(store)
    with pytest.raises(TypeError):
        import pickle
        pickle.dumps(store)


# ------------------------------------------------------------------------------------------------
# Tiny Qwen4Exp
# ------------------------------------------------------------------------------------------------
modeling = None
try:
    import transformers.models.qwen4_exp.modeling_qwen4_exp as modeling  # noqa: F811
except Exception:
    pass
needs_qwen4 = pytest.mark.skipif(modeling is None, reason="transformers without qwen4_exp")


def _tiny_config():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
    text = dict(
        model_type="qwen4_exp_text", num_hidden_layers=4, hidden_size=64, head_dim=16, num_attention_heads=4,
        num_key_value_heads=2, hc_count=2, hc_lowrank=16, full_attention_interval=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"], linear_conv_kernel_dim=4,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        moe_intermediate_size=32, shared_expert_intermediate_size=32, num_experts=4, num_experts_per_tok=2,
        ple_embed_dim=128, ple_layer_ids=[2], ple_conv_kernel_size=4, ngram_size=3, heads_per_ngram=8,
        ngram_vocab_size_base=1009, split_ngram_parts=4, make_ngram_vocab_size_divisible_by=4,
        indexer_head_dim=16, indexer_n_heads=2, indexer_kv_heads=1, indexer_budget=8, indexer_compress_ratio=4,
        mtp_num_hidden_layers=0, vocab_size=1000, bos_token_id=1, eos_token_id=2, pad_token_id=None,
        partial_rotary_factor=0.25, max_position_embeddings=512, tie_word_embeddings=False,
        rope_parameters={"mrope_interleaved": True, "mrope_section": [1, 1, 0], "partial_rotary_factor": 0.25,
                         "rope_theta": 10000000, "rope_type": "default"},
    )
    vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                  num_position_embeddings=64, patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
                  in_channels=3, deepstack_visual_indexes=[])
    return Qwen4ExpConfig(text_config=text, vision_config=vision, tie_word_embeddings=False)


@pytest.fixture(scope="module")
def tiny_qwen4(tmp_path_factory):
    if modeling is None:
        pytest.skip("transformers without qwen4_exp")
    threads = torch.get_num_threads()
    torch.set_num_threads(min(threads, 4))  # the reference GDN kernels crawl when oversubscribed
    from unsloth_zoo.temporary_patches.qwen4_exp import patch_qwen4_exp_frozen_table
    assert patch_qwen4_exp_frozen_table(modeling)
    path = tmp_path_factory.mktemp("tiny_qwen4")
    torch.manual_seed(0)
    model = modeling.Qwen4ExpForConditionalGeneration(_tiny_config())  # from_config: a real table
    emb = model.model.language_model.layers[1].ple.ple_embedding.ngram_embedding
    assert type(emb) is torch.nn.Embedding
    with torch.no_grad():
        emb.weight.normal_(0, 1)
    model.to(torch.bfloat16).save_pretrained(str(path), max_shard_size="40KB")
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    shard_keys = sorted(k for k in index if "ngram_embedding.shard_" in k)
    assert len(shard_keys) == 4 and len(set(index.values())) > 1  # sharded checkpoint
    yield str(path), shard_keys
    torch.set_num_threads(threads)


class _SliceSpy:
    def __init__(self, key, real, log):
        self._key, self._real, self._log = key, real, log

    def __getitem__(self, idx):
        self._log.append(self._key)
        return self._real[idx]

    def __getattr__(self, name):
        return getattr(self._real, name)


class _OpenSpy:
    def __init__(self, real, log):
        self._real, self._log = real, log

    def get_slice(self, key):
        return _SliceSpy(key, self._real.get_slice(key), self._log)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _load(path, mode, monkeypatch, dtype, log=None):
    import transformers.modeling_utils as mu
    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE", mode)
    if log is not None:
        real_open = mu.safe_open
        monkeypatch.setattr(mu, "safe_open", lambda *a, **k: _OpenSpy(real_open(*a, **k), log))
    model = modeling.Qwen4ExpForConditionalGeneration.from_pretrained(path, dtype=dtype)
    monkeypatch.undo() if log is not None else None
    return model


@needs_qwen4
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tiny_qwen4_disk_load_matches_normal_load(tiny_qwen4, monkeypatch, dtype):
    path, shard_keys = tiny_qwen4
    ref_log = []
    ref = _load(path, "gpu", monkeypatch, dtype, log=ref_log)
    assert set(shard_keys) <= set(ref_log)  # the spy sees shard reads on the normal path
    ref_emb = ref.model.language_model.layers[1].ple.ple_embedding.ngram_embedding
    assert type(ref_emb) is torch.nn.Embedding and ref_emb.weight.dtype == dtype

    log = []
    model = _load(path, "disk", monkeypatch, dtype, log=log)
    emb = model.model.language_model.layers[1].ple.ple_embedding.ngram_embedding
    assert isinstance(emb, UnslothFrozenTableEmbedding) and emb.is_bound
    assert emb.frozen_table_store.mode == "disk"
    assert emb.table_dtype == dtype
    # Every other tensor was read, no n-gram shard was.
    assert log and not [k for k in log if "ngram_embedding" in k and "shard_" in k], sorted(set(log))[:5]
    assert not [n for n, _ in model.named_parameters() if "ngram_embedding" in n]
    assert not [n for n, _ in model.named_buffers() if "ngram_embedding.weight" in n]
    # state_dict carries the checkpoint's own shard tensors (zero-copy views of the mapped files), not one table.
    sd = model.state_dict()
    assert sorted(k for k in sd if "ngram_embedding" in k) == shard_keys
    table_key = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight"
    assert set(ref.state_dict()) - set(sd) == {table_key}
    assert torch.equal(torch.cat([sd[k] for k in shard_keys]).to(dtype), ref.state_dict()[table_key])
    # Nothing left for the no-placement / block-swap machinery to keep on the host.
    from unsloth_zoo.device_map_planner import drop_no_placement_modules
    assert drop_no_placement_modules(model) == []
    assert model.model.language_model.layers[1].ple.ple_embedding.ngram_embedding is emb
    for k, v in model.state_dict().items():
        if k not in shard_keys:
            assert torch.equal(v, ref.state_dict()[k]), k

    torch.manual_seed(1)
    ids = torch.randint(3, 1000, (2, 24))
    ids[0, 10] = 2  # eos inside the sequence: n-gram reset
    labels = ids.clone()
    reset_frozen_table_stats()
    for m in (ref, model):
        m.train()
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    out_ref = ref(input_ids=ids, labels=labels)
    out_ref.loss.backward()
    out = model(input_ids=ids, labels=labels)
    out.loss.backward()
    assert torch.equal(out.logits, out_ref.logits)
    assert torch.equal(out.loss, out_ref.loss)
    grads_ref = {n: p.grad for n, p in ref.named_parameters() if p.grad is not None}
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    grads_ref.pop("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight", None)
    assert set(grads) == set(grads_ref)
    for n in grads:
        assert torch.equal(grads[n], grads_ref[n]), n
    assert grads["model.language_model.layers.1.ple.key_proj.weight"].abs().sum() > 0
    # one disk gather for forward + checkpoint recompute
    assert FROZEN_TABLE_STATS["gathers"] == 1
    assert FROZEN_TABLE_STATS["memo_hits"] >= 1
    assert FROZEN_TABLE_STATS["unique_rows"] <= 2 * 24 * 16


@needs_qwen4
def test_tiny_qwen4_disk_model_saves_and_reloads(tiny_qwen4, monkeypatch, tmp_path):
    path, shard_keys = tiny_qwen4
    model = _load(path, "disk", monkeypatch, torch.bfloat16)
    model.load_state_dict(model.state_dict())  # the exported shard views load back without unexpected keys
    out = tmp_path / "saved"
    model.save_pretrained(str(out), max_shard_size="40KB")
    with pytest.raises(ValueError, match="Save to a different directory"):
        model.save_pretrained(path)   # would overwrite the files the table streams from
    saved = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert sorted(k for k in saved if "ngram_embedding" in k) == shard_keys
    ids = torch.randint(3, 1000, (1, 16))
    with torch.no_grad():
        want = _load(path, "gpu", monkeypatch, torch.bfloat16)(input_ids=ids).logits
        for mode in ("gpu", "disk"):
            got = _load(str(out), mode, monkeypatch, torch.bfloat16)(input_ids=ids).logits
            assert torch.equal(got, want), mode


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_table_is_read_in_its_checkpoint_dtype(tmp_path, dtype):
    torch.manual_seed(0)
    shards = [torch.randn(n, 6).to(dtype) for n in (5, 9)]
    index, names = _write_sharded(tmp_path, shards)
    full = torch.cat(shards)
    store = FrozenTableStore.from_checkpoint(index, {"weight": names}, mode="disk")
    ids = torch.tensor([0, 4, 5, 13, 4])
    assert torch.equal(store.gather(ids), full[ids])
    assert torch.equal(store.gather(ids, dtype=torch.float64), full[ids].double())
    store.close()


@needs_qwen4
def test_tiny_qwen4_from_config_keeps_real_table(tiny_qwen4, monkeypatch):
    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE", "disk")
    model = modeling.Qwen4ExpForConditionalGeneration(_tiny_config())
    emb = model.model.language_model.layers[1].ple.ple_embedding.ngram_embedding
    assert type(emb) is torch.nn.Embedding


@needs_qwen4
def test_tiny_qwen4_missing_table_is_loud(tiny_qwen4, monkeypatch, tmp_path):
    path, shard_keys = tiny_qwen4
    from safetensors.torch import load_file
    index = json.loads((open(os.path.join(path, "model.safetensors.index.json")).read()))
    tensors = {}
    for f in sorted(set(index["weight_map"].values())):
        tensors.update(load_file(os.path.join(path, f)))
    for k in shard_keys:
        tensors.pop(k)
    save_file(tensors, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
    for f in os.listdir(path):
        if f.endswith(".json") and "index" not in f:
            (tmp_path / f).write_text(open(os.path.join(path, f)).read())
    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE", "disk")
    with pytest.raises(RuntimeError, match="could not find frozen table"):
        modeling.Qwen4ExpForConditionalGeneration.from_pretrained(str(tmp_path))


def test_lookup_compiles_fullgraph_without_recompiles(bf16_table):
    index, names, full = bf16_table
    emb = UnslothFrozenTableEmbedding(full.shape[0], 20, dtype=torch.float32)
    emb.bind(FrozenTableStore.from_checkpoint(index, {"weight": names}, mode="disk"))
    torch._dynamo.reset()
    fn = torch.compile(lambda ids, x: emb(ids) * x, fullgraph=True, backend="aot_eager", dynamic=True)
    for n in (5, 9, 17):
        ids = torch.randint(0, full.shape[0], (2, n))
        assert torch.equal(fn(ids, torch.ones(2, n, 20)), F.embedding(ids, full.float()))


def test_direct_bind_checks_the_row_shape(bf16_table):
    index, names, full = bf16_table
    emb = UnslothFrozenTableEmbedding(full.shape[0], 7, dtype=torch.float32)
    with pytest.raises(ValueError, match="rows decode to"):
        emb.bind(FrozenTableStore.from_checkpoint(index, {"weight": names}, mode="disk"))


@pytest.mark.skipif(os.name == "nt", reason="Windows refuses to replace a file that is mapped, so a stale mapping cannot arise")
def test_a_file_replaced_at_the_same_path_is_mapped_again(tmp_path):
    shards = [torch.full((3, 4), 1.0, dtype=torch.bfloat16)]
    index, names = _write_sharded(tmp_path, shards, files=1)
    first = FrozenTableStore.from_checkpoint(index, {"weight": names}, mode="disk")
    import os as _os, time as _t
    _t.sleep(0.01)
    _write_sharded(tmp_path, [torch.full((3, 4), 7.0, dtype=torch.bfloat16)], files=1)
    second = FrozenTableStore.from_checkpoint(index, {"weight": names}, mode="disk")
    assert second.gather(torch.tensor([0])).float().unique().tolist() == [7.0]
    first.close(), second.close()


def test_full_finetuning_keeps_the_ngram_table_trainable_in_auto_mode(monkeypatch):
    from unsloth_zoo.temporary_patches.qwen4_exp import _ngram_table_wanted
    monkeypatch.setattr(fts, "_SMALL_TABLE_BYTES", 0)
    monkeypatch.setattr(fts, "_available_host_bytes", lambda: 1 << 40)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE", "auto")
    monkeypatch.setenv("UNSLOTH_ENABLE_FULL_FINETUNING", "0")
    assert _ngram_table_wanted(1000, 16, torch.bfloat16)
    monkeypatch.setenv("UNSLOTH_ENABLE_FULL_FINETUNING", "1")
    assert not _ngram_table_wanted(1000, 16, torch.bfloat16)
    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE", "disk")
    assert _ngram_table_wanted(1000, 16, torch.bfloat16)


def test_exported_shards_do_not_share_a_storage(bf16_table):
    index, names, full = bf16_table
    emb = UnslothFrozenTableEmbedding(full.shape[0], 20, dtype=torch.float32)
    store = FrozenTableStore.from_checkpoint(index, {"weight": names}, mode="disk")
    emb.bind(store)
    emb.checkpoint_suffixes = {"weight": [f"shard_{i}.weight" for i in range(len(names))]}
    sd = emb.state_dict()
    ptrs = [t.untyped_storage().data_ptr() for t in sd.values()]
    assert len(set(ptrs)) == len(ptrs)   # a save must not see tied views and clone them into RAM
    assert torch.equal(torch.cat(list(sd.values())), full)


def test_deepcopy_registers_its_own_lookup(bf16_table):
    import copy, gc
    index, names, full = bf16_table
    emb = UnslothFrozenTableEmbedding(full.shape[0], 20, dtype=torch.float32)
    emb.bind(FrozenTableStore.from_checkpoint(index, {"weight": names}, mode="disk"))
    twin = copy.deepcopy(emb)
    assert twin._handle != emb._handle and twin.frozen_table_store is emb.frozen_table_store
    twin.to(torch.float64)
    ids = torch.tensor([0, 5, 99])
    assert emb(ids).dtype == torch.float32 and twin(ids).dtype == torch.float64
    del emb
    gc.collect()
    assert torch.equal(twin(ids), full[ids].double())


def test_lookup_op_is_tagged_cudagraph_unsafe():
    tag = getattr(torch._C.Tag, "cudagraph_unsafe", None)
    if tag is None or fts._LOOKUP_OP is None:
        pytest.skip("torch without the cudagraph_unsafe tag or custom_op")
    assert tag in torch.ops.unsloth_zoo.frozen_table_lookup.default.tags


def test_exported_tensors_outlive_the_store(bf16_table):
    import gc
    index, names, full = bf16_table
    emb = UnslothFrozenTableEmbedding(full.shape[0], 20, dtype=torch.float32)
    emb.bind(FrozenTableStore.from_checkpoint(index, {"weight": names}, mode="disk"))
    emb.checkpoint_suffixes = {"weight": [f"shard_{i}.weight" for i in range(len(names))]}
    sd = emb.state_dict()
    del emb
    gc.collect()
    assert torch.equal(torch.cat([t.clone() for t in sd.values()]), full)   # still mapped, no dangling pointer

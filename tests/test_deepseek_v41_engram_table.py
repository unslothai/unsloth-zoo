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

"""DeepSeek-V4.1's engram tables streamed from the checkpoint (the frozen table store) instead of loaded into
host RAM: a tiny model in the community transformers port, bf16 and fp8 + e8m0 tables, loaded in disk mode
matches the port's own DeepseekV41EngramEmbedding load bit for bit and never reads a table tensor."""

import json
import os

import pytest
import torch
from safetensors.torch import load_file, save_file

try:
    from transformers.models.deepseek_v41 import modeling_deepseek_v41 as modeling
except Exception:  # transformers without the DeepSeek-V4.1 port
    modeling = None
pytestmark = pytest.mark.skipif(modeling is None, reason="transformers without deepseek_v41 (community port)")

from unsloth_zoo import frozen_table_store as fts  # noqa: E402
from unsloth_zoo.frozen_table_store import FROZEN_TABLE_STATS, UnslothFrozenTableEmbedding, reset_frozen_table_stats  # noqa: E402

needs_fp8 = pytest.mark.skipif(not hasattr(torch, "float8_e8m0fnu"), reason="torch without float8_e8m0fnu")


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocab = {t: i for i, t in enumerate("abcdefgh")}
    vocab["<pad>"] = 8
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="a"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend)


def _config(tok):
    from transformers import DeepseekV41TextConfig
    _, compressed = modeling.build_compressed_token_map(tok)
    return DeepseekV41TextConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=1, num_hidden_layers=5,
        compress_ratios=[0, 2, 2, 1, 1], num_experts_per_tok=2, moe_intermediate_size=64, max_position_embeddings=128,
        vocab_size=len(tok), head_dim=32, qk_rope_head_dim=8, q_lora_rank=32, o_groups=2, o_lora_rank=16,
        n_routed_experts=4, n_shared_experts=1, sliding_window=8, hc_mult=2, index_n_heads=2, index_head_dim=32,
        index_topk=2, candidate_topk_blocks=4, candidate_block_size=4, num_nextn_predict_layers=0,
        tie_word_embeddings=False, engram_layer_ids=[1, 2], engram_num_embeddings=[700, 700], engram_vocab_size=64,
        engram_n_heads=2, engram_head_dim=32, engram_max_ngram_size=4, engram_pad_id=8,
        engram_compressed_vocab_size=compressed, dspark_noise_token_id=len(tok) - 1,
    )


TABLES = ("model.engram_tables.1", "model.engram_tables.2")


@pytest.fixture(scope="module")
def tiny_dsv41(tmp_path_factory):
    from unsloth_zoo.temporary_patches.deepseek_v41 import patch_deepseek_v41_engram_table
    os.environ["UNSLOTH_FROZEN_TABLE"] = "disk"
    try:
        assert patch_deepseek_v41_engram_table()
    finally:
        os.environ.pop("UNSLOTH_FROZEN_TABLE", None)
    tok = _tokenizer()
    torch.manual_seed(0)
    model = modeling.DeepseekV41ForCausalLM(_config(tok))   # from_config: real tables
    assert type(model.model.engram_tables["1"]) is modeling.DeepseekV41EngramEmbedding
    with torch.no_grad():
        for t in model.model.engram_tables.values():
            t.weight.normal_(0, 0.5)
    bf16 = tmp_path_factory.mktemp("dsv41_bf16")
    model.to(torch.bfloat16).save_pretrained(str(bf16))
    tok.save_pretrained(str(bf16))
    # fp8 e4m3 rows + one e8m0 scale per 32 channels, as the released checkpoint stores the tables.
    fp8 = tmp_path_factory.mktemp("dsv41_fp8")
    files = [f for f in os.listdir(bf16) if f.endswith(".safetensors")]
    linears = sorted(n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear))
    for f in os.listdir(bf16):
        src = os.path.join(bf16, f)
        if f in files:
            sd, new = load_file(src), {}
            for k, v in sd.items():
                if any(k == t + ".weight_scale_inv" for t in TABLES):
                    continue
                if any(k == t + ".weight" for t in TABLES):
                    blocks = v.float().unflatten(-1, (-1, 32))
                    scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-12) / 448.0)))
                    new[k] = (blocks / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).flatten(-2)
                    new[k[: -len("weight")] + "weight_scale_inv"] = scale.to(torch.float8_e8m0fnu)
                else:
                    new[k] = v
            save_file(new, os.path.join(fp8, f), metadata={"format": "pt"})
        elif os.path.isfile(src):
            with open(src, "rb") as a, open(os.path.join(fp8, f), "wb") as b:
                b.write(a.read())
    cfg = json.loads(open(os.path.join(fp8, "config.json")).read())
    cfg["quantization_config"] = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [32, 32],
                                  "scale_fmt": "ue8m0", "modules_to_not_convert": linears + ["lm_head"]}
    open(os.path.join(fp8, "config.json"), "w").write(json.dumps(cfg))
    return {"bf16": str(bf16), "fp8": str(fp8)}


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

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, *exc):
        return self._real.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _load(path, mode, monkeypatch, log=None):
    import transformers.modeling_utils as mu
    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE", mode)
    if log is not None:
        real_open = mu.safe_open
        monkeypatch.setattr(mu, "safe_open", lambda *a, **k: _OpenSpy(real_open(*a, **k), log))
    model = modeling.DeepseekV41ForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    if log is not None:
        monkeypatch.setattr(mu, "safe_open", real_open)
    return model


@pytest.mark.parametrize("kind", ["bf16", pytest.param("fp8", marks=needs_fp8)])
def test_disk_engram_matches_the_ports_table(tiny_dsv41, monkeypatch, kind):
    path = tiny_dsv41[kind]
    ref = _load(path, "gpu", monkeypatch)
    assert type(ref.model.engram_tables["1"]) is modeling.DeepseekV41EngramEmbedding
    log = []
    model = _load(path, "disk", monkeypatch, log=log)
    tables = model.model.engram_tables
    assert all(isinstance(t, UnslothFrozenTableEmbedding) and t.is_bound for t in tables.values())
    assert {t.frozen_table_store.mode for t in tables.values()} == {"disk"}
    assert log and not [k for k in log if "engram_tables" in k or "engram.embed" in k], sorted(set(log))[:6]
    assert not [n for n, _ in model.named_parameters() if "engram_tables" in n]
    assert len(tables["1"].frozen_table_store.planes) == (2 if kind == "fp8" else 1)   # a bf16 save's scale is unused
    ids = torch.arange(0, 700, 7).view(1, 4, 25)
    with torch.no_grad():
        for k in tables:
            assert torch.equal(tables[k](ids), ref.model.engram_tables[k](ids)), k
    # Training: logits, loss and every gradient bitwise equal. The model gathers each table once per forward, outside
    # the decoder layers, so a layer recompute never reads it again.
    x = torch.randint(0, 8, (2, 24))
    reset_frozen_table_stats()
    for m in (ref, model):
        m.train()
    out_ref = ref(input_ids=x, labels=x)
    out_ref.loss.backward()
    out = model(input_ids=x, labels=x)
    out.loss.backward()
    assert torch.equal(out.logits, out_ref.logits) and torch.equal(out.loss, out_ref.loss)
    grads_ref = {n: p.grad for n, p in ref.named_parameters() if p.grad is not None and "engram_tables" not in n}
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert set(grads) == set(grads_ref)
    for n in grads:
        assert torch.equal(grads[n], grads_ref[n]), n
    assert grads["model.layers.1.engram.wkv.weight"].abs().sum() > 0
    assert FROZEN_TABLE_STATS["gathers"] == 2
    # The export. A bf16 table: zero-copy views under the port's names. An fp8 table on a dequantized load (no GPU
    # here, so transformers dequantizes): the model holds decoded rows and drops its quantization config, so fp8
    # rows + scales would reload as raw rows. Outside a full save nothing is exported (an adapter save must not
    # decode the table); inside one, the decoded rows equal the port's dequantized table.
    sd = model.state_dict()
    if kind == "bf16":
        assert {k for k in sd if "engram_tables" in k} == {f"{t}.weight" for t in TABLES}
    else:
        assert not [k for k in sd if "engram_tables" in k]
        with fts.full_table_export():
            sd = model.state_dict()
        assert {k for k in sd if "engram_tables" in k} == {f"{t}.weight" for t in TABLES}
        for t in TABLES:
            assert torch.equal(sd[t + ".weight"], ref.state_dict()[t + ".weight"]), t


@needs_fp8
def test_native_engram_names_bind_and_decode_like_the_port(tmp_path):
    # The released files: `layers.<i>.engram.embed.{weight,scale}`, bound to the model-level engram_tables.<i>.
    from unsloth_zoo.temporary_patches.deepseek_v41 import _frozen_engram
    emb = modeling.DeepseekV41EngramEmbedding(96, 64)
    ref = modeling.DeepseekV41EngramEmbedding(96, 64)
    g = torch.Generator().manual_seed(0)
    q = (torch.randn(96, 64, generator=g) * 50).clamp(-448, 448).to(torch.float8_e4m3fn)
    s = torch.exp2(torch.randint(-8, 4, (96, 2), generator=g).float()).to(torch.float8_e8m0fnu)
    save_file({"layers.14.engram.embed.weight": q, "layers.14.engram.embed.scale": s,
               "layers.14.engram.wkv.weight": torch.zeros(4, 4)}, str(tmp_path / "m.safetensors"))
    with torch.no_grad():
        ref.weight = torch.nn.Parameter(q, requires_grad=False)
        ref.weight_scale_inv = torch.nn.Parameter(s, requires_grad=False)
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.engram_tables = torch.nn.ModuleDict({"14": _frozen_engram(modeling, "14", emb)})
    # A dequantize=False load: the quantizer is the load's (model.hf_quantizer is attached only afterwards).
    import types
    kept = types.SimpleNamespace(quantization_config=types.SimpleNamespace(quant_method="fp8", dequantize=False))
    assert getattr(model, "hf_quantizer", None) is None
    fts.bind_frozen_tables(model, str(tmp_path / "m.safetensors"), mode="disk", quantizer=kept)
    table = model.model.engram_tables["14"]
    ids = torch.tensor([[0, 5, 95, 40]])
    with torch.no_grad():
        want = ref(ids)                       # float32 rows: the port keeps the fp8 table (dequantize=False)
        got = table(ids)
        assert got.dtype == torch.float32 and torch.equal(got, want)
    # Kept fp8: the export is the fp8 rows and scales themselves, which reload with the quantization config.
    sd = model.state_dict()
    assert torch.equal(sd["model.engram_tables.14.weight"].view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(sd["model.engram_tables.14.weight_scale_inv"].view(torch.uint8), s.view(torch.uint8))


def test_fp8_kept_load_returns_float32_rows():
    from unsloth_zoo.temporary_patches.deepseek_v41 import _fp8_weights_kept
    import types
    qc = types.SimpleNamespace(quant_method="fp8", dequantize=False)
    assert _fp8_weights_kept(types.SimpleNamespace(hf_quantizer=types.SimpleNamespace(quantization_config=qc)))
    qc.dequantize = True
    assert not _fp8_weights_kept(types.SimpleNamespace(hf_quantizer=types.SimpleNamespace(quantization_config=qc)))
    assert not _fp8_weights_kept(torch.nn.Module())


def test_from_config_and_gpu_mode_keep_the_ports_table(tiny_dsv41, monkeypatch):
    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE", "disk")
    model = modeling.DeepseekV41ForCausalLM(_config(_tokenizer()))
    assert type(model.model.engram_tables["1"]) is modeling.DeepseekV41EngramEmbedding
    ref = _load(tiny_dsv41["bf16"], "gpu", monkeypatch)
    assert type(ref.model.engram_tables["2"]) is modeling.DeepseekV41EngramEmbedding

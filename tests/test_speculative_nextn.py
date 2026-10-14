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

import dataclasses
import json
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.glm4_moe")

from safetensors.torch import load_file, save_file

from unsloth_zoo.speculative.mtp import build_mtp_head, load_mtp_state_dict, mtp_family_name
from unsloth_zoo.speculative.mtp_families import MTP_FAMILIES, mtp_family_for_config
from unsloth_zoo.speculative.nextn import (
    NextNHead,
    _ChainKV,
    _window_mask,
    build_nextn_head,
    export_nextn_state_dict,
    pack_experts,
    unpack_experts,
)
from unsloth_zoo.speculative.stitch import write_mtp_tensors

VOCAB, HIDDEN = 97, 64


def _glm_config(**kwargs):
    from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
    base = dict(
        vocab_size = VOCAB, hidden_size = HIDDEN, intermediate_size = 128, moe_intermediate_size = 32,
        num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16, num_hidden_layers = 2,
        n_routed_experts = 4, num_experts_per_tok = 2, n_shared_experts = 1, first_k_dense_replace = 1,
        num_nextn_predict_layers = 1, partial_rotary_factor = 0.5,
    )
    base.update(kwargs)
    return Glm4MoeConfig(**base)


def _randomise(head, seed = 0):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in head.named_parameters():
            if param.ndim == 1:
                param.copy_(1.0 + 0.3 * torch.randn(param.shape, generator = gen))
            else:
                param.copy_(torch.randn(param.shape, generator = gen) / math.sqrt(param.shape[-1]))
    return head.eval()


def _chained(family, heads):
    return dataclasses.replace(family, chained = True, max_heads = heads)


def test_matches_the_qwen35_head_it_generalises():
    pytest.importorskip("transformers.models.qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    config = Qwen3_5TextConfig(
        vocab_size = VOCAB, hidden_size = HIDDEN, intermediate_size = 128, num_attention_heads = 4,
        num_key_value_heads = 2, head_dim = 16, num_hidden_layers = 2,
        layer_types = ["linear_attention", "full_attention"], mtp_num_hidden_layers = 1,
    )
    reference = _randomise(build_mtp_head(config).float())
    family = mtp_family_for_config(config)
    canonical = {family.from_ckpt("mtp." + k, 2)[1]: v for k, v in reference.state_dict().items() if "inv_freq" not in k}
    head = build_nextn_head(config, state_dict = canonical, dtype = torch.float32).eval()
    embed = torch.nn.Embedding(VOCAB, HIDDEN)
    ids = torch.randint(0, VOCAB, (2, 12))
    mask = torch.ones(2, 12, dtype = torch.long)
    mask[1, 9:] = 0
    hidden = torch.randn(2, 12, HIDDEN)
    with torch.no_grad():
        a = reference.forward_train(hidden, ids, embed, attention_mask = mask, num_steps = 3)
        b = head.forward_train(hidden, ids, embed, attention_mask = mask, num_steps = 3)
        head._embedding_first = False
        swapped = head.forward_train(hidden, ids, embed, attention_mask = mask, num_steps = 1)
    for x, y in zip(a, b):
        assert torch.equal(x["labels"], y["labels"])
        assert torch.allclose(x["hidden"][x["valid"]], y["hidden"][x["valid"]], atol = 1e-5)
    assert not torch.allclose(a[0]["hidden"][a[0]["valid"]], swapped[0]["hidden"][a[0]["valid"]], atol = 1e-2)


@pytest.mark.parametrize("family_name", sorted({f.name for f in MTP_FAMILIES.values()}))
def test_checkpoint_names_round_trip(family_name):
    family = next(f for f in MTP_FAMILIES.values() if f.name == family_name)
    n = 4
    canon = [
        "heads.0.eh_proj.weight", "heads.0.enorm.weight", "heads.0.hnorm.weight",
        "heads.0.shared_head_norm.weight", "heads.0.block.0.self_attn.q_proj.weight",
        "heads.0.block.0.mlp.experts.3.down_proj.weight",
    ]
    if family.style == "nemotron":
        family = dataclasses.replace(family, blocks = ({}, {}))
        canon.append("heads.0.block.1.mixer.gate.weight")
    prefix = family.default_prefix()
    for name in canon:
        ckpt = family.to_ckpt(name, prefix, n)
        assert family.is_mtp_name(ckpt, n), ckpt
        assert family.from_ckpt(ckpt, n) == (prefix, name), (ckpt, family.from_ckpt(ckpt, n))
    if family.style == "layers":
        assert not family.is_mtp_name(f"{prefix}layers.{n - 1}.mlp.gate.weight", n)


def test_real_checkpoint_spellings():
    glm = MTP_FAMILIES["glm4_moe"]
    assert glm.from_ckpt("model.layers.46.shared_head.head.weight", 46) == ("model.", "heads.0.shared_head_head.weight")
    assert glm.from_ckpt("model.layers.46.embed_tokens.weight", 46) == ("model.", "heads.0.embed_tokens.weight")
    step = MTP_FAMILIES["step3p5"]
    assert step.from_ckpt("model.layers.47.transformer.shared_head.output.weight", 45) == \
        ("model.", "heads.2.shared_head_head.weight")
    hy = MTP_FAMILIES["hy_v3"]
    assert hy.from_ckpt("model.layers.80.final_layernorm.weight", 80)[1] == "heads.0.shared_head_norm.weight"
    assert hy.from_ckpt("model.layers.80.mlp.router.gate.weight", 80)[1] == "heads.0.block.0.mlp.gate.weight"
    assert hy.to_ckpt("heads.0.block.0.mlp.gate.weight", "model.", 80) == "model.layers.80.mlp.router.gate.weight"
    qwen = MTP_FAMILIES["qwen3_next"]
    assert qwen.from_ckpt("mtp.fc.weight", 48)[1] == "heads.0.eh_proj.weight"
    nemotron = MTP_FAMILIES["nemotron_h"]
    assert nemotron.from_ckpt("mtp.layers.1.final_layernorm.weight", 88)[1] == "heads.0.shared_head_norm.weight"
    assert nemotron.from_ckpt("mtp.layers.0.eh_proj.weight", 88)[1] == "heads.0.eh_proj.weight"
    assert nemotron.from_ckpt("mtp.layers.1.mixer.gate.weight", 88)[1] == "heads.0.block.1.mixer.gate.weight"


def test_expert_pack_round_trip():
    head = build_nextn_head(_glm_config())
    module = head.state_dict()
    per_expert = unpack_experts(module, ["heads.0.block.0.mlp.experts"])
    assert "heads.0.block.0.mlp.experts.3.gate_proj.weight" in per_expert
    packed, prefixes = pack_experts(per_expert, module)
    assert prefixes == ["heads.0.block.0.mlp.experts"]
    assert set(packed) == set(module)
    assert all(torch.equal(packed[k], module[k]) for k in module)


def _write_target_checkpoint(path, config, head = None):
    path.mkdir()
    tensors = {"model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN), "lm_head.weight": torch.randn(VOCAB, HIDDEN)}
    tensors["model.layers.0.input_layernorm.weight"] = torch.ones(HIDDEN)
    if head is not None:
        tensors.update({k: v.contiguous() for k, v in export_nextn_state_dict(head).items()})
    save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(config.to_dict()))
    return path


def test_load_build_export_round_trip(tmp_path):
    config = _glm_config()
    head = _randomise(build_nextn_head(config))
    ckpt = _write_target_checkpoint(tmp_path / "ckpt", config, head)
    state_dict, info = load_mtp_state_dict(str(ckpt))
    assert info["family"] == "glm4_moe" and info["prefix"] == "" and info["ckpt_prefix"] == "model."
    assert all(name.startswith("model.layers.2.") for name in state_dict)
    loaded = build_mtp_head(config, state_dict = state_dict, dtype = torch.float32)
    assert isinstance(loaded, NextNHead)
    assert set(loaded.mtp_state_dict()) == set(state_dict)
    assert all(torch.equal(loaded.mtp_state_dict()[k], v) for k, v in state_dict.items())
    assert mtp_family_name(config) == "glm4_moe"


def test_head_serialized_copies(tmp_path):
    config = _glm_config()
    head = build_nextn_head(config)
    sd = {k: v.clone() for k, v in head.state_dict().items()}
    embed, lm = torch.randn(VOCAB, HIDDEN), torch.randn(VOCAB, HIDDEN)
    sd["heads.0.embed_tokens.weight"] = embed.clone()
    own_head = torch.randn(VOCAB, HIDDEN)
    sd["heads.0.shared_head_head.weight"] = own_head.clone()
    built = build_nextn_head(config, state_dict = sd, target_embed = embed, target_lm_head = lm, dtype = torch.bfloat16)
    assert built._frozen_kinds == {"heads.0.embed_tokens.weight": "duplicate", "heads.0.shared_head_head.weight": "owned"}
    assert built.heads[0].embed_weight is None
    # The owned copy keeps its values through to_empty / dtype casts (it was once zeroed there).
    assert torch.equal(built.heads[0].lm_head_weight.float(), own_head)
    assert not any("shared_head_head" in k or "embed_tokens" in k for k in built.mtp_state_dict())


def test_stitch_replace_and_append(tmp_path):
    config = _glm_config()
    stock = _randomise(build_nextn_head(config), seed = 1)
    ckpt = _write_target_checkpoint(tmp_path / "with", config, stock)
    trained = _randomise(build_nextn_head(config), seed = 2)
    report = write_mtp_tensors(str(ckpt), trained.mtp_state_dict(), **trained.stitch_kwargs())
    assert report["mode"] == "replaced"
    on_disk = load_file(str(ckpt / "model.safetensors"))
    for name, tensor in trained.mtp_state_dict().items():
        assert torch.equal(on_disk[name], tensor)
    assert torch.equal(on_disk["model.layers.0.input_layernorm.weight"], torch.ones(HIDDEN))

    bare = _write_target_checkpoint(tmp_path / "without", config)
    cfg = json.loads((bare / "config.json").read_text())
    cfg.pop("num_nextn_predict_layers", None)
    (bare / "config.json").write_text(json.dumps(cfg))
    report = write_mtp_tensors(str(bare), trained.mtp_state_dict(), **trained.stitch_kwargs())
    assert report["mode"] == "appended" and report["config_declared"]
    assert json.loads((bare / "config.json").read_text())["num_nextn_predict_layers"] == 1
    index = json.loads((bare / "model.safetensors.index.json").read_text())
    assert all(index["weight_map"][k] == "model-mtp.safetensors" for k in trained.mtp_state_dict())


def test_stitch_append_keeps_owned_copies(tmp_path):
    """A full-model save drops the MTP layer: appending writes owned copies back, never duplicates."""
    config = _glm_config()
    head = build_nextn_head(config)
    sd = {k: v.clone() for k, v in _randomise(head).state_dict().items()}
    embed, lm, own_head = torch.randn(VOCAB, HIDDEN), torch.randn(VOCAB, HIDDEN), torch.randn(VOCAB, HIDDEN)
    sd["heads.0.embed_tokens.weight"] = embed.clone()
    sd["heads.0.shared_head_head.weight"] = own_head.clone()
    built = build_nextn_head(config, state_dict = sd, target_embed = embed, target_lm_head = lm)
    bare = _write_target_checkpoint(tmp_path / "bare", config)
    report = write_mtp_tensors(str(bare), built.mtp_state_dict(), **built.stitch_kwargs())
    assert report["mode"] == "appended"
    on_disk = load_file(str(bare / "model-mtp.safetensors"))
    assert torch.equal(on_disk["model.layers.2.shared_head.head.weight"], own_head)
    assert "model.layers.2.embed_tokens.weight" not in on_disk
    # Saving again over it replaces only the trained tensors and leaves the copy as written.
    report = write_mtp_tensors(str(bare), built.mtp_state_dict(), **built.stitch_kwargs())
    assert report["mode"] == "replaced"
    assert torch.equal(load_file(str(bare / "model-mtp.safetensors"))["model.layers.2.shared_head.head.weight"], own_head)
    state_dict, info = load_mtp_state_dict(str(bare))
    again = build_mtp_head(config, state_dict = state_dict, target_embed = embed, target_lm_head = lm,
                           ckpt_prefix = info.get("ckpt_prefix"))
    assert again._frozen_kinds == {"heads.0.shared_head_head.weight": "owned"}


def _greedy_consistency(head, steps = 3):
    """forward_train on a sequence built from the head's own greedy drafts reproduces them."""
    torch.manual_seed(0)
    embed = torch.nn.Embedding(VOCAB, HIDDEN)
    lm = torch.randn(VOCAB, HIDDEN)
    S, anchor = 10, 6
    ids = torch.randint(0, VOCAB, (1, S))
    hidden = torch.randn(1, S + steps, HIDDEN)
    with torch.no_grad():
        drafts = head.draft_greedy(hidden, ids, embed, lm, anchor = anchor, num_steps = steps)[0].tolist()
        seq = torch.cat([ids[:, : anchor + 1], torch.tensor([drafts])], dim = 1)
        out = head.forward_train(hidden[:, : seq.shape[1]], seq, embed, num_steps = steps)
    teacher = [int((out[k]["hidden"][0, anchor] @ lm.T).argmax()) for k in range(steps)]
    return drafts, teacher


def test_training_forward_matches_incremental_drafting():
    head = _randomise(build_nextn_head(_glm_config()))
    drafts, teacher = _greedy_consistency(head)
    assert drafts == teacher


def test_chained_heads_match_incremental_drafting_and_differ_from_one_head():
    config = _glm_config(num_nextn_predict_layers = 3)
    family = _chained(mtp_family_for_config(config), 3)
    head = _randomise(build_nextn_head(config, family = family))
    assert len(head.heads) == 3 and head.chained
    drafts, teacher = _greedy_consistency(head)
    assert drafts == teacher
    # The same weights drafted with head 0 at every step must give other drafts.
    head.family = dataclasses.replace(head.family, chained = False)
    torch.manual_seed(0)
    assert _greedy_consistency(head)[0] != drafts


def test_window_mask_limits_keys_to_the_window():
    cache = _ChainKV()
    cache.add_positions(torch.arange(6)[None])
    positions = torch.tensor([[6]])
    mask = torch.zeros(1, 1, 1, 7)
    out = _window_mask(mask, positions, cache, window = 3)
    allowed = (out[0, 0, 0] == 0).tolist()
    assert allowed == [False, False, False, False, True, True, True]


def test_pre_norm_family_trains_on_last_layer_hidden(tmp_path):
    from datasets import Dataset
    from transformers import Glm4MoeForCausalLM
    from unsloth_zoo.speculative.trainer import MTPDataCollator, MTPTrainingArguments, UnslothMTPTrainer
    torch.manual_seed(0)
    config = _glm_config()
    target = Glm4MoeForCausalLM(config).eval()
    family = dataclasses.replace(mtp_family_for_config(config), h_source = "last_layer", feedback = "raw")
    head = build_nextn_head(config, family = family)
    head._unsloth_mtp_info = {"prefix": "", "num_layers": 1}
    calls = []
    original = head.forward_train

    def spy(hidden_in, *args, **kwargs):
        calls.append(hidden_in.detach().clone())
        return original(hidden_in, *args, **kwargs)

    head.forward_train = spy
    before = {k: v.detach().clone() for k, v in target.state_dict().items()}
    rows = [{"input_ids": torch.randint(1, VOCAB, (int(n),)).tolist()} for n in (9, 12)]
    args = MTPTrainingArguments(
        output_dir = str(tmp_path), per_device_train_batch_size = 2, max_steps = 1, learning_rate = 1e-2,
        logging_steps = 1, save_strategy = "no", report_to = "none", use_cpu = True, mtp_num_steps = 2,
    )
    UnslothMTPTrainer(model = head, target_model = target, args = args, train_dataset = Dataset.from_list(rows),
                      data_collator = MTPDataCollator(pad_token_id = 0)).train()
    assert all(torch.equal(before[k], v) for k, v in target.state_dict().items())
    ids = torch.tensor([rows[0]["input_ids"] + [0] * 3])
    with torch.no_grad():
        out = target.model(input_ids = ids[:, :12], output_hidden_states = True)
    # transformers' last hidden_states entry is post-norm; the pre-norm residual is the last layer's output.
    assert calls and not torch.allclose(calls[0][0, :9], out.last_hidden_state[0, :9], atol = 1e-4)

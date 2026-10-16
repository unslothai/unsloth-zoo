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

import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from datasets import Dataset  # noqa: E402

from unsloth_zoo.speculative.capture import run_frozen_target  # noqa: E402
from unsloth_zoo.speculative.dspark import (  # noqa: E402
    DSparkTrainingArguments,
    Qwen3DSparkDrafter,
    UnslothDSparkTrainer,
    dspark_loss,
    sample_anchor_positions,
)
from unsloth_zoo.speculative.trainer import MTPDataCollator  # noqa: E402

VOCAB = 97
LAYERS = [0, 2]
BS = 4


def _target():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size = VOCAB, hidden_size = 64, intermediate_size = 128, num_hidden_layers = 4,
        num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16, max_position_embeddings = 256,
    )
    return Qwen3ForCausalLM(config).eval()


def _drafter(target, **kwargs):
    torch.manual_seed(1)
    defaults = dict(target_layer_ids = LAYERS, mask_token_id = VOCAB - 1, num_draft_layers = 2, block_size = BS,
                    num_anchors = 6, markov_rank = 8, dtype = torch.float64)
    defaults.update(kwargs)
    return Qwen3DSparkDrafter.from_target(target.double(), **defaults).eval()


def _batch(S = 20, prompt = 5, seed = 0):
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB - 1, (1, S), generator = gen)
    loss_mask = torch.zeros_like(ids)
    loss_mask[:, prompt:] = 1
    return ids, loss_mask


def _run(drafter, target, ids, loss_mask, anchors = None, gen_seed = 0):
    cap = run_frozen_target(target, ids, layer_ids = LAYERS)
    gen = torch.Generator().manual_seed(gen_seed)
    ctx = drafter.context_states(cap)
    return cap, ctx, drafter.forward_train(ctx, ids, loss_mask, generator = gen, anchors = anchors)


def test_anchor_sampler_picks_supervised_pairs_sorted():
    loss_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 1, 1, 1, 1]])
    anchors, keep = sample_anchor_positions(loss_mask, 8, torch.Generator().manual_seed(0))
    valid = {a for a in range(9) if loss_mask[0, a] and loss_mask[0, a + 1]}
    picked = anchors[0][keep[0]].tolist()
    assert set(picked) == valid and picked == sorted(picked)
    assert int(keep.sum()) == len(valid) and anchors[0][~keep[0]].eq(0).all()


def test_labels_prev_tokens_and_positions_follow_the_anchor():
    target = _target()
    drafter = _drafter(target)
    ids, lm = _batch()
    _, _, out = _run(drafter, target, ids, lm)
    S = ids.shape[1]
    for n, a in enumerate(out["anchors"][0].tolist()):
        if not out["keep"][0, n]:
            continue
        for j in range(BS):
            if a + j + 1 < S:
                assert out["target_ids"][0, n, j] == ids[0, a + j + 1]
                assert out["prev_ids"][0, n, j] == ids[0, a + j]
            # eval_mask is a contiguous in-range supervised prefix
            assert bool(out["eval_mask"][0, n, j]) == all(a + i + 1 < S and lm[0, a + i + 1] for i in range(j + 1))


def test_block_sees_only_its_own_block_and_context_before_the_anchor():
    target = _target()
    drafter = _drafter(target)
    ids, lm = _batch(S = 24)
    cap, ctx, out = _run(drafter, target, ids, lm)
    anchors = (out["anchors"], out["keep"])
    n = int(out["keep"][0].nonzero()[len(out["keep"][0].nonzero()) // 2])
    a = int(out["anchors"][0, n])
    base = out["hidden"][0, n]
    # Context at or after the anchor, and every other token in the sequence beyond the anchor, is invisible.
    ctx2 = ctx.clone()
    ctx2[:, a:] += torch.randn_like(ctx2[:, a:])
    ids2 = ids.clone()
    ids2[:, a + 1:] = (ids2[:, a + 1:] + 7) % (VOCAB - 1)
    out2 = drafter.forward_train(ctx2, ids2, lm, anchors = anchors)
    assert torch.allclose(out2["hidden"][0, n], base, atol = 1e-10)
    # Context strictly before the anchor is visible.
    if a > 0:
        ctx3 = ctx.clone()
        ctx3[:, a - 1] += 1.0
        out3 = drafter.forward_train(ctx3, ids, lm, anchors = anchors)
        assert not torch.allclose(out3["hidden"][0, n], base)


def test_markov_bias_uses_the_teacher_forced_predecessor():
    target = _target()
    drafter = _drafter(target)
    ids, lm = _batch()
    _, _, out = _run(drafter, target, ids, lm)
    prev = out["prev_ids"].clone()
    base = drafter.logits(out["hidden"], prev)
    prev[..., 2] = (prev[..., 2] + 1) % VOCAB
    moved = drafter.logits(out["hidden"], prev)
    assert torch.equal(base[..., [0, 1, 3], :], moved[..., [0, 1, 3], :])
    assert not torch.allclose(base[..., 2, :], moved[..., 2, :])


def _reference_loss(drafter, out, final, gamma = 4.0, ce_a = 0.1, l1_a = 0.9, conf_a = 1.0, teacher_w = None):
    em = out["eval_mask"]
    logits = drafter.logits(out["hidden"], out["prev_ids"])
    teacher = torch.gather(final[:, None].expand(-1, em.shape[1], -1, -1), 2,
                           (out["label_index"] - 1).clamp(min = 0)[..., None].expand(-1, -1, -1, final.shape[-1]))
    p = (teacher @ (drafter.lm_head.weight if teacher_w is None else teacher_w).T).softmax(-1)
    q = logits.softmax(-1)
    w = em.double() * torch.exp(-torch.arange(BS, dtype = torch.float64) / gamma)
    ce = torch.nn.functional.cross_entropy(logits.flatten(0, 2), out["target_ids"].flatten(), reduction = "none").view_as(w)
    l1 = (q - p).abs().sum(-1)
    accept = (1 - 0.5 * l1).clamp(0, 1).detach()
    conf = drafter.confidence_logits(out["hidden"], out["prev_ids"]).double()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(conf, accept, reduction = "none")
    den = w.sum() + 1e-6
    return ce_a * (ce * w).sum() / den + l1_a * (l1 * w).sum() / den + conf_a * (bce * w).sum() / den


@pytest.mark.parametrize("chunk_size", [1, 3, 1024])
def test_loss_and_gradients_match_an_unchunked_reference(chunk_size):
    target = _target()
    drafter = _drafter(target).freeze_shared()
    ids, lm = _batch()
    cap, _, out = _run(drafter, target, ids, lm)
    final = cap["final"].double()
    loss, stats = dspark_loss(drafter, out, teacher_hidden = final, chunk_size = chunk_size)
    grads = torch.autograd.grad(loss, [p for p in drafter.parameters() if p.requires_grad])
    _, _, out_ref = _run(drafter, target, ids, lm)
    ref = _reference_loss(drafter, out_ref, final)
    ref_grads = torch.autograd.grad(ref, [p for p in drafter.parameters() if p.requires_grad])
    assert torch.allclose(loss.double(), ref, atol = 1e-6)
    for g, r in zip(grads, ref_grads):
        assert torch.allclose(g, r, atol = 1e-6)
    assert stats["n_tokens"] == int(out["eval_mask"].sum())


def test_teacher_is_projected_with_the_target_lm_head():
    target = _target()
    drafter = _drafter(target).freeze_shared()
    ids, lm = _batch()
    cap, _, out = _run(drafter, target, ids, lm)
    final = cap["final"].double()
    torch.manual_seed(3)
    target_w = drafter.lm_head.weight.detach() + 0.5 * torch.randn_like(drafter.lm_head.weight)
    loss, _ = dspark_loss(drafter, out, teacher_hidden = final, teacher_lm_head_weight = target_w)
    _, _, out_ref = _run(drafter, target, ids, lm)
    assert torch.allclose(loss.double(), _reference_loss(drafter, out_ref, final, teacher_w = target_w), atol = 1e-6)
    assert not torch.allclose(loss.double(), _reference_loss(drafter, out_ref, final), atol = 1e-4)


def test_chunks_respect_the_logits_budget(monkeypatch):
    from unsloth_zoo.speculative import dspark, losses
    target = _target()
    drafter = _drafter(target).freeze_shared()
    ids, lm = _batch()
    cap, _, out = _run(drafter, target, ids, lm)
    final = cap["final"].double()
    full, _ = dspark_loss(drafter, out, teacher_hidden = final, chunk_size = 1024)
    sizes, real = [], dspark._row_terms
    monkeypatch.setattr(dspark, "_row_terms", lambda *a: sizes.append(a[0].shape[0]) or real(*a))
    monkeypatch.setattr(losses, "LOGITS_BUDGET_BYTES", VOCAB * 4 * 6 * 2)
    capped, _ = dspark_loss(drafter, out, teacher_hidden = final, chunk_size = 1024)
    assert sizes and max(sizes) == 2 and sum(sizes) == int(out["eval_mask"].sum())
    assert torch.allclose(capped, full, atol = 1e-9)


def test_save_round_trip_and_schema(tmp_path):
    target = _target()
    drafter = _drafter(target)
    drafter.save_pretrained(str(tmp_path), dtype = torch.float32)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["architectures"] == ["Qwen3DSparkModel"]
    for key in ("block_size", "target_layer_ids", "mask_token_id", "num_anchors", "markov_rank",
                "markov_head_type", "enable_confidence_head", "confidence_head_with_markov", "num_target_layers"):
        assert key in config, key
    assert config["markov_head_type"] == "vanilla" and config["target_layer_ids"] == LAYERS
    loaded = Qwen3DSparkDrafter.from_pretrained(str(tmp_path), dtype = torch.float64)
    sd, sd2 = drafter.state_dict(), loaded.state_dict()
    assert set(sd2) == set(sd)
    assert all(torch.allclose(sd[k].float(), sd2[k].float()) for k in sd if not k.startswith("rotary_emb."))
    names = set(sd2)
    for required in ("fc.weight", "hidden_norm.weight", "norm.weight", "embed_tokens.weight", "lm_head.weight",
                     "markov_head.markov_w1.weight", "markov_head.markov_w2.weight", "confidence_head.proj.weight",
                     "layers.0.self_attn.q_norm.weight", "layers.1.mlp.down_proj.weight"):
        assert required in names, required


def test_gguf_converter_accepts_the_saved_drafter(tmp_path):
    llama = os.environ.get("UNSLOTH_TEST_LLAMA_CPP") or os.path.join(os.environ.get("WORKSPACE", ""), "temp", "llama_build")
    converter = os.path.join(llama, "convert_hf_to_gguf.py")
    if not os.path.isfile(converter):
        pytest.skip("no llama.cpp checkout with convert_hf_to_gguf.py")
    try:
        from huggingface_hub import snapshot_download
        target_dir = snapshot_download("trl-internal-testing/tiny-Qwen3ForCausalLM")
    except Exception as error:
        pytest.skip(f"tiny Qwen3 unavailable: {type(error).__name__}")
    from transformers import AutoModelForCausalLM
    target = AutoModelForCausalLM.from_pretrained(target_dir, dtype = torch.float32).eval()
    vocab = target.config.vocab_size
    drafter = Qwen3DSparkDrafter.from_target(target, target_layer_ids = [0, 1], mask_token_id = vocab - 1,
                                             num_draft_layers = 1, block_size = BS, markov_rank = 8)
    drafter.save_pretrained(str(tmp_path / "draft"))
    out = tmp_path / "draft.gguf"
    proc = subprocess.run(
        [sys.executable, converter, str(tmp_path / "draft"), "--target-model-dir", target_dir, "--outtype", "f32", "--outfile", str(out)],
        capture_output = True, text = True, env = dict(os.environ, PYTHONPATH = os.path.join(llama, "gguf-py")),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    sys.path.insert(0, os.path.join(llama, "gguf-py"))
    import gguf
    reader = gguf.GGUFReader(str(out))
    names = {t.name for t in reader.tensors}
    assert {"markov_w1.weight", "markov_w2.weight", "fc.weight"} <= names
    assert any("conf" in n for n in names)
    fields = {f.name: f.contents() for f in reader.fields.values() if f.name.startswith("dflash.")}
    assert fields.get("dflash.block_size") == BS
    assert fields.get("dflash.sample_from_anchor") is True
    assert list(fields.get("dflash.target_layers")) == [1, 2]


def test_trainer_moves_only_the_draft_body(tmp_path):
    target = _target().float()
    drafter = Qwen3DSparkDrafter.from_target(target, target_layer_ids = LAYERS, mask_token_id = VOCAB - 1,
                                             num_draft_layers = 1, block_size = BS, num_anchors = 6, markov_rank = 8)
    before = {k: v.detach().clone() for k, v in drafter.state_dict().items()}
    before_target = {k: v.detach().clone() for k, v in target.state_dict().items()}
    gen = torch.Generator().manual_seed(0)
    rows = []
    for n in (14, 18, 16, 20):
        ids = torch.randint(0, VOCAB - 1, (n,), generator = gen).tolist()
        rows.append({"input_ids": ids, "labels": [-100] * 4 + ids[4:]})
    args = DSparkTrainingArguments(
        output_dir = str(tmp_path), per_device_train_batch_size = 2, max_steps = 3, learning_rate = 1e-2,
        logging_steps = 1, save_strategy = "no", report_to = "none", use_cpu = True, dspark_num_anchors = 6, seed = 0,
    )
    trainer = UnslothDSparkTrainer(model = drafter, target_model = target, args = args, train_dataset = Dataset.from_list(rows),
                                   data_collator = MTPDataCollator(pad_token_id = 0))
    trainer.train()
    after = drafter.state_dict()
    assert torch.equal(before["embed_tokens.weight"], after["embed_tokens.weight"])
    assert torch.equal(before["lm_head.weight"], after["lm_head.weight"])
    assert not torch.equal(before["fc.weight"], after["fc.weight"])
    assert not torch.equal(before["markov_head.markov_w2.weight"], after["markov_head.markov_w2.weight"])
    assert all(torch.equal(before_target[k], v) for k, v in target.state_dict().items())
    logs = [h for h in trainer.state.log_history if "dspark_ce_loss" in h]
    assert logs and all(torch.isfinite(torch.tensor(h["loss"])) for h in trainer.state.log_history if "loss" in h)
    trainer.save_model(str(tmp_path / "saved"))
    assert (tmp_path / "saved" / "config.json").is_file()


def test_trainer_evaluates_and_resumes(tmp_path):
    target = _target().float()

    def drafter():
        return Qwen3DSparkDrafter.from_target(target, target_layer_ids = LAYERS, mask_token_id = VOCAB - 1,
                                              num_draft_layers = 1, block_size = BS, num_anchors = 6, markov_rank = 8)

    gen = torch.Generator().manual_seed(1)
    rows = Dataset.from_list([{"input_ids": torch.randint(0, VOCAB - 1, (16,), generator = gen).tolist()} for _ in range(8)])
    args = DSparkTrainingArguments(
        output_dir = str(tmp_path), per_device_train_batch_size = 2, max_steps = 4, learning_rate = 1e-2,
        save_strategy = "steps", save_steps = 2, eval_strategy = "steps", eval_steps = 2, logging_steps = 1,
        report_to = "none", use_cpu = True, dspark_num_anchors = 6, seed = 0,
    )
    first = drafter()
    trainer = UnslothDSparkTrainer(model = first, target_model = target, args = args, train_dataset = rows,
                                   eval_dataset = rows, data_collator = MTPDataCollator(pad_token_id = 0))
    trainer.train()
    assert trainer.evaluate()["eval_loss"] > 0
    resumed = UnslothDSparkTrainer(model = drafter(), target_model = target, args = args, train_dataset = rows,
                                   eval_dataset = rows, data_collator = MTPDataCollator(pad_token_id = 0))
    resumed.train(resume_from_checkpoint = str(tmp_path / "checkpoint-2"))
    assert resumed.state.global_step == 4


def test_default_ddp_reduces_the_loss_only_heads(tmp_path):
    """Trainer wraps an nn.Module in DDP(find_unused_parameters=True): the Markov and confidence heads, used
    only by the loss, must be in forward()'s output graph or backward raises "mark a variable ready only once"."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    target = _target().float()
    drafter = Qwen3DSparkDrafter.from_target(target, target_layer_ids = LAYERS, mask_token_id = VOCAB - 1,
                                             num_draft_layers = 1, block_size = BS, num_anchors = 6, markov_rank = 8,
                                             confidence_head = True).train()
    ids = torch.randint(0, VOCAB - 1, (2, 16), generator = torch.Generator().manual_seed(0))
    args = DSparkTrainingArguments(output_dir = str(tmp_path), report_to = "none", use_cpu = True,
                                   dspark_num_anchors = 6, save_strategy = "no")
    trainer = UnslothDSparkTrainer(model = drafter, target_model = target, args = args,
                                   train_dataset = Dataset.from_list([{"input_ids": ids[0].tolist(), "labels": ids[0].tolist()}]),
                                   data_collator = MTPDataCollator(pad_token_id = 0))
    dist.init_process_group("gloo", store = dist.HashStore(), rank = 0, world_size = 1)
    try:
        ddp = DistributedDataParallel(drafter, find_unused_parameters = True)
        trainer.compute_loss(ddp, {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}).backward()
    finally:
        dist.destroy_process_group()
    assert drafter.markov_head.markov_w2.weight.grad is not None
    assert all(p.grad is not None for p in drafter.confidence_head.parameters())


def test_layer_minus_one_is_the_embedding_output(tmp_path):
    """DeepSpec's target layer -1 is the embedding output, not the last decoder layer."""
    target = _target().float()
    drafter = Qwen3DSparkDrafter.from_target(target, target_layer_ids = [-1, 1], mask_token_id = VOCAB - 1,
                                             num_draft_layers = 1, block_size = BS, num_anchors = 6, markov_rank = 8)
    seen = {}
    real = drafter.context_states

    def spy(captured):
        seen.update(captured)
        return real(captured)

    drafter.context_states = spy
    ids = torch.randint(0, VOCAB - 1, (1, 16), generator = torch.Generator().manual_seed(0))
    args = DSparkTrainingArguments(output_dir = str(tmp_path), report_to = "none", use_cpu = True,
                                   dspark_num_anchors = 6, save_strategy = "no")
    trainer = UnslothDSparkTrainer(model = drafter, target_model = target, args = args,
                                   train_dataset = Dataset.from_list([{"input_ids": ids[0].tolist(), "labels": ids[0].tolist()}]),
                                   data_collator = MTPDataCollator(pad_token_id = 0))
    batch = {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}
    trainer.compute_loss(drafter, batch)
    with torch.no_grad():
        reference = target(input_ids = ids, output_hidden_states = True).hidden_states
    assert torch.allclose(seen[-1].float(), reference[0].float())
    assert torch.allclose(seen[1].float(), reference[2].float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU to put target layers on two devices")
def test_context_states_from_layers_on_different_devices():
    target = _target()
    drafter = _drafter(target).cuda()
    cap = run_frozen_target(target, _batch()[0], layer_ids = LAYERS)
    cap[LAYERS[1]] = cap[LAYERS[1]].cuda()
    ctx = drafter.context_states(cap)
    assert ctx.device.type == "cuda" and ctx.shape[-1] == 2 * cap[LAYERS[0]].shape[-1]

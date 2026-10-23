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

"""Speculative training must project through the target's effective lm_head, unmerged LoRA included."""

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

from peft import LoraConfig, get_peft_model  # noqa: E402

from unsloth_zoo.speculative.capture import effective_lm_head_weight, resolve_text_parts  # noqa: E402
from unsloth_zoo.speculative.trainer import mtp_step_losses  # noqa: E402

VOCAB = 97


def _target():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size = VOCAB, hidden_size = 64, intermediate_size = 128, num_hidden_layers = 4,
        num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16, max_position_embeddings = 256,
        tie_word_embeddings = False,
    )
    return Qwen3ForCausalLM(config).eval()


def _lora_target(head_bias = False, dtype = None, **lora_kwargs):
    torch.manual_seed(1)
    config = LoraConfig(r = 4, lora_alpha = 16, lora_dropout = 0.0, target_modules = ["lm_head"],
                        init_lora_weights = False, **lora_kwargs)
    target = _target()
    if head_bias:
        old = target.lm_head
        target.lm_head = torch.nn.Linear(old.in_features, old.out_features, bias = True)
        with torch.no_grad():
            target.lm_head.weight.copy_(old.weight)
            target.lm_head.bias.zero_()
    if dtype is not None:
        target = target.to(dtype)
    model = get_peft_model(target, config).eval()
    with torch.no_grad():
        # A large delta so the adapter visibly changes the argmax.
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.mul_(20.0)
    return model


def _head(model):
    return resolve_text_parts(model)["lm_head"]


def test_effective_weight_matches_the_adapted_forward():
    model = _lora_target()
    head = _head(model)
    x = torch.randn(8, 64)
    w = effective_lm_head_weight(head)
    with torch.no_grad():
        expected = head(x)
        base = head.get_base_layer()(x)
    assert torch.allclose(x @ w.T, expected, atol = 1e-4)
    assert not torch.allclose(base, expected, atol = 1e-2)
    assert not w.requires_grad


def test_plain_merged_and_disabled_heads_return_the_layer_weight():
    plain = _target()
    assert effective_lm_head_weight(plain.lm_head) is plain.lm_head.weight
    model = _lora_target()
    head = _head(model)
    with model.disable_adapter():
        assert effective_lm_head_weight(head) is head.get_base_layer().weight
    head.merge()
    assert effective_lm_head_weight(head) is head.get_base_layer().weight


def test_effective_weight_is_cached_and_refreshed_on_change():
    head = _head(_lora_target())
    first = effective_lm_head_weight(head)
    assert effective_lm_head_weight(head) is first
    with torch.no_grad():
        head.lora_B["default"].weight.mul_(2.0)
    second = effective_lm_head_weight(head)
    assert second is not first
    x = torch.randn(4, 64)
    with torch.no_grad():
        assert torch.allclose(x @ second.T, head(x), atol = 1e-4)


def test_dora_head_is_refused():
    head = _head(_lora_target(use_dora = True))
    with pytest.raises(NotImplementedError, match = "DoRA"):
        effective_lm_head_weight(head)


class _FakeDraft:
    """One MTP step whose drafted hidden is fixed, so the loss depends only on the head matrices."""

    def __init__(self, hidden, labels):
        self.hidden, self.labels = hidden, labels

    def __call__(self, *args, **kwargs):
        return [dict(hidden = self.hidden, labels = self.labels)]


@pytest.mark.parametrize("loss_type", ["ce", "kl_div"])
def test_mtp_teacher_follows_the_unmerged_adapter(loss_type):
    model = _lora_target()
    merged = copy.deepcopy(model)
    _head(merged).merge()
    torch.manual_seed(2)
    S = 12
    final_hidden = torch.randn(1, S, 64)
    ids = torch.randint(0, VOCAB, (1, S))
    draft_hidden = torch.randn(1, S - 1, 64)
    args = SimpleNamespace(mtp_num_steps = 1, mtp_step_beta = 0.6, mtp_label_source = "teacher",
                           mtp_loss_type = loss_type, mtp_chunk_size = 1024)

    def run(target):
        draft = _FakeDraft(draft_hidden, ids[:, 1:].clone())
        parts = dict(embed_tokens = None, lm_head = _head(target))
        loss, stats = mtp_step_losses(draft, parts, final_hidden, ids, torch.ones_like(ids), ids, args)
        return loss, stats

    loss, stats = run(model)
    ref_loss, ref_stats = run(merged)
    base_loss, _ = run(_target())
    assert torch.allclose(loss, ref_loss, atol = 1e-4)
    assert stats[0][1:] == ref_stats[0][1:]
    # The adapter matters for this target: the bare base head gives a different objective.
    assert not torch.allclose(loss, base_loss, atol = 1e-3)


def test_dspark_copies_and_scores_with_the_effective_head(tmp_path, monkeypatch):
    from datasets import Dataset
    from unsloth_zoo.speculative import dspark
    from unsloth_zoo.speculative.dspark import DSparkTrainingArguments, Qwen3DSparkDrafter, UnslothDSparkTrainer
    from unsloth_zoo.speculative.trainer import MTPDataCollator

    model = _lora_target()
    effective = effective_lm_head_weight(_head(model)).detach().clone()
    drafter = Qwen3DSparkDrafter.from_target(model, target_layer_ids = [0, 2], mask_token_id = VOCAB - 1,
                                             num_draft_layers = 1, block_size = 4, num_anchors = 6, markov_rank = 8)
    assert torch.allclose(drafter.lm_head.weight.detach(), effective.float())

    seen = []
    original = dspark.dspark_loss

    def spy(*args, **kwargs):
        seen.append(kwargs["teacher_lm_head_weight"])
        return original(*args, **kwargs)

    monkeypatch.setattr(dspark, "dspark_loss", spy)
    gen = torch.Generator().manual_seed(0)
    rows = [{"input_ids": (ids := torch.randint(0, VOCAB - 1, (n,), generator = gen).tolist()),
             "labels": [-100] * 4 + ids[4:]} for n in (14, 18)]
    args = DSparkTrainingArguments(
        output_dir = str(tmp_path), per_device_train_batch_size = 2, max_steps = 1, learning_rate = 1e-2,
        save_strategy = "no", report_to = "none", use_cpu = True, dspark_num_anchors = 6, seed = 0,
    )
    trainer = UnslothDSparkTrainer(model = drafter, target_model = model, args = args, train_dataset = Dataset.from_list(rows),
                                   data_collator = MTPDataCollator(pad_token_id = 0))
    trainer.train()
    assert seen and all(torch.allclose(w, effective) for w in seen)


def test_eagle3_trainer_uses_the_effective_head(tmp_path):
    from unsloth_zoo.speculative.eagle3 import (
        Eagle3TrainingArguments,
        UnslothEagle3Trainer,
        build_eagle3_drafter,
        build_eagle3_vocab_map,
    )

    model = _lora_target()
    effective = effective_lm_head_weight(_head(model))
    gen = torch.Generator().manual_seed(1)
    d2t, t2d = build_eagle3_vocab_map([torch.randint(0, VOCAB, (500,), generator = gen).tolist()], VOCAB, 32)
    drafter = build_eagle3_drafter(model, d2t, t2d, aux_layer_ids = [1, 2, 3])
    rows = effective[t2d.bool()] if drafter.uses_draft_vocab else effective
    assert torch.allclose(drafter.lm_head.weight.detach(), rows.float())
    args = Eagle3TrainingArguments(output_dir = str(tmp_path), report_to = "none", save_strategy = "no", use_cpu = True)
    trainer = UnslothEagle3Trainer(model = drafter, target_model = model, args = args)
    assert torch.allclose(trainer._lm_head_weight, effective)


def test_effective_weight_accumulates_in_place_in_the_base_dtype(monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode
    from unsloth_zoo.speculative import capture

    model = _lora_target(dtype = torch.bfloat16)
    head = _head(model)
    shape = tuple(head.get_base_layer().weight.shape)
    fp32_full = []
    merged = copy.deepcopy(model)
    _head(merged).merge()
    # Same representation as PEFT's merge: the delta added into the base dtype.
    torch.testing.assert_close(effective_lm_head_weight(head), _head(merged).get_base_layer().weight,
                               atol = 0, rtol = 0)
    with torch.no_grad():
        head.lora_B["default"].weight.mul_(1.5)  # invalidate the cache

    def delta(name):
        return (head.lora_B[name].weight.float() @ head.lora_A[name].weight.float()) * head.scaling[name]

    class Watch(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args = (), kwargs = None):
            out = func(*args, **(kwargs or {}))
            if isinstance(out, torch.Tensor) and tuple(out.shape) == shape and out.dtype == torch.float32 \
                    and not func.__name__.split(".")[0].endswith("_"):
                fp32_full.append(func.__name__)
            return out

    monkeypatch.setattr(capture, "_DELTA_BLOCK_BYTES", 16 * 64 * 4)  # 16-row delta blocks
    with Watch():
        w = effective_lm_head_weight(head)
    # No full-size fp32 temporaries (the old path held a base copy, the delta, its cast and the sum).
    assert fp32_full == []
    assert w.dtype == torch.bfloat16
    with torch.no_grad():
        expected = (head.get_base_layer().weight.float() + delta("default")).to(torch.bfloat16)
    torch.testing.assert_close(w, expected, atol = 0, rtol = 0)


def test_lora_bias_is_refused_as_unsupported():
    model = _lora_target(head_bias = True, lora_bias = True)
    with torch.no_grad():
        _head(model).lora_B["default"].bias.fill_(0.25)
    with pytest.raises(NotImplementedError, match = "lora_bias = True.*does not support"):
        effective_lm_head_weight(_head(model))
    # Merging folds the adapter bias into the head bias, which a weight-only projection drops.
    _head(model).merge()
    with pytest.raises(NotImplementedError, match = "nonzero bias"):
        effective_lm_head_weight(_head(model))


def test_merged_head_with_a_bias_is_checked():
    model = _lora_target(head_bias = True)
    head = _head(model)
    head.merge()
    # A zero bias changes nothing: the merged weight is the whole head.
    assert effective_lm_head_weight(head) is head.get_base_layer().weight
    with torch.no_grad():
        head.get_base_layer().bias.fill_(0.5)
    with pytest.raises(NotImplementedError, match = "nonzero bias"):
        effective_lm_head_weight(head)

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

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen3_5")

from datasets import Dataset

from unsloth_zoo.speculative.capture import resolve_text_parts, run_frozen_target
from unsloth_zoo.speculative.mtp import build_mtp_head
from unsloth_zoo.speculative.stitch import load_saved_mtp_head
from unsloth_zoo.speculative.trainer import (
    MTPDataCollator,
    MTPTrainingArguments,
    UnslothMTPTrainer,
    mtp_step_losses,
)

# With a GPU visible, zoo routes Qwen3.5's gated delta rule to Triton, which needs CUDA tensors.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB = 97


def _target():
    from transformers import Qwen3_5ForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size = VOCAB, hidden_size = 64, intermediate_size = 128, num_attention_heads = 4,
        num_key_value_heads = 2, head_dim = 16, num_hidden_layers = 2,
        layer_types = ["linear_attention", "full_attention"], mtp_num_hidden_layers = 1,
    )
    return Qwen3_5ForCausalLM(config).to(DEVICE).eval()


def _args(tmp_path, **kwargs):
    defaults = dict(
        output_dir = str(tmp_path), per_device_train_batch_size = 2, max_steps = 3, learning_rate = 1e-2,
        logging_steps = 1, save_strategy = "no", report_to = "none", use_cpu = DEVICE == "cpu",
        mtp_num_steps = 3, seed = 0,
    )
    defaults.update(kwargs)
    return MTPTrainingArguments(**defaults)


def test_supervised_counts_follow_the_response_mask(tmp_path):
    target = _target()
    parts = resolve_text_parts(target)
    head = build_mtp_head(parts["config"], device = DEVICE)
    S, prompt = 12, 4
    input_ids = torch.randint(1, VOCAB, (1, S), device = DEVICE)
    labels = input_ids.clone()
    labels[:, :prompt] = -100
    final = run_frozen_target(target, input_ids)["final"]
    _, stats = mtp_step_losses(head, parts, final, input_ids, torch.ones_like(input_ids), labels, _args(tmp_path))
    # Step k predicts x[p+k+1] for chains p >= 1: response tokens are indices prompt..S-1.
    for k, (_, n_tokens, _) in enumerate(stats):
        expected = sum(1 for p in range(1, S) if p + k + 1 < S and p + k + 1 >= prompt)
        assert n_tokens == expected, (k, n_tokens, expected)


@pytest.mark.parametrize("loss_type", ["ce", "kl_div", "lk_hybrid"])
def test_trainer_moves_only_the_head_and_saves_it(tmp_path, loss_type):
    target = _target()
    head = build_mtp_head(resolve_text_parts(target)["config"], device = DEVICE)
    head._unsloth_mtp_info = {"prefix": "mtp.", "num_layers": 1}
    before_head = {k: v.detach().clone() for k, v in head.state_dict().items()}
    before_target = {k: v.detach().clone() for k, v in target.state_dict().items()}
    gen = torch.Generator().manual_seed(0)
    rows = [{"input_ids": torch.randint(1, VOCAB, (int(n),), generator = gen).tolist()} for n in (9, 14, 11, 16)]
    trainer = UnslothMTPTrainer(
        model = head, target_model = target, args = _args(tmp_path, mtp_loss_type = loss_type),
        train_dataset = Dataset.from_list(rows), data_collator = MTPDataCollator(pad_token_id = 0),
    )
    trainer.train()
    assert any(not torch.equal(before_head[k], v) for k, v in head.state_dict().items())
    assert all(torch.equal(before_target[k], v) for k, v in target.state_dict().items())
    assert not any(p.requires_grad for p in target.parameters())
    logs = [h for h in trainer.state.log_history if "mtp_loss_0" in h]
    assert logs and all(0 <= h["mtp_top1_0"] <= 1 for h in logs)
    trainer.save_model(str(tmp_path / "saved"))
    state_dict, info = load_saved_mtp_head(str(tmp_path / "saved"))
    assert set(state_dict) == set(head.state_dict())
    assert info["prefix"] == "mtp."


def test_collator_pads_and_masks():
    batch = MTPDataCollator(pad_token_id = 0)([
        {"input_ids": [5, 6, 7], "labels": [-100, 6, 7]},
        {"input_ids": [8]},
    ])
    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 6, 7], [8, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_evaluate_and_resume(tmp_path):
    target = _target()
    head = build_mtp_head(resolve_text_parts(target)["config"], device = DEVICE)
    head._unsloth_mtp_info = {"prefix": "mtp.", "num_layers": 1}
    gen = torch.Generator().manual_seed(1)
    rows = Dataset.from_list([{"input_ids": torch.randint(1, VOCAB, (12,), generator = gen).tolist()} for _ in range(8)])
    args = _args(tmp_path, max_steps = 4, save_strategy = "steps", save_steps = 2, eval_strategy = "steps", eval_steps = 2)
    trainer = UnslothMTPTrainer(
        model = head, target_model = target, args = args, train_dataset = rows, eval_dataset = rows,
        data_collator = MTPDataCollator(pad_token_id = 0),
    )
    trainer.train()
    metrics = trainer.evaluate()
    assert metrics["eval_loss"] > 0
    checkpoint = str(tmp_path / "checkpoint-2")
    resumed_head = build_mtp_head(resolve_text_parts(target)["config"], device = DEVICE)
    resumed = UnslothMTPTrainer(
        model = resumed_head, target_model = target, args = args, train_dataset = rows, eval_dataset = rows,
        data_collator = MTPDataCollator(pad_token_id = 0),
    )
    resumed.train(resume_from_checkpoint = checkpoint)
    assert resumed.state.global_step == 4


def test_gradient_accumulation_matches_one_big_batch(tmp_path):
    # Unequal lengths: averaging microbatch means would weight short rows more than long ones.
    target = _target()
    config = resolve_text_parts(target)["config"]
    gen = torch.Generator().manual_seed(2)
    rows = Dataset.from_list([{"input_ids": torch.randint(1, VOCAB, (int(n),), generator = gen).tolist()} for n in (6, 20, 9, 31)])
    torch.manual_seed(0)
    init = build_mtp_head(config, device = DEVICE).state_dict()
    results = []
    for bs, ga in ((4, 1), (1, 4)):
        head = build_mtp_head(config, device = DEVICE)
        head.load_state_dict(init)
        args = _args(tmp_path / f"{bs}_{ga}", per_device_train_batch_size = bs, gradient_accumulation_steps = ga,
                     max_steps = 1, optim = "sgd", learning_rate = 1.0, max_grad_norm = 0.0)
        UnslothMTPTrainer(model = head, target_model = target, args = args, train_dataset = rows,
                          data_collator = MTPDataCollator(pad_token_id = 0)).train()
        results.append(head.state_dict())
    for k in init:
        assert torch.allclose(results[0][k], results[1][k], atol = 1e-5), k

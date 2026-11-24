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

"""CPU ORPO: log-probs from the last hidden state in row chunks, equal to TRL's full-logits forward."""

import dataclasses
import os

import pytest
import torch

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def _orpo():
    import trl
    if hasattr(trl, "ORPOTrainer"):
        return trl.ORPOConfig, trl.ORPOTrainer
    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    try:
        from trl.experimental.orpo import ORPOConfig, ORPOTrainer
    except ImportError:
        pytest.skip("this TRL has no ORPOTrainer")
    return ORPOConfig, ORPOTrainer


def _trainer(tmp_path, dtype, bf16):
    from datasets import Dataset
    from unsloth_zoo.cpu import loader
    from unsloth_zoo.cpu.trainer import install_cpu_training_defaults
    install_cpu_training_defaults()
    try:
        model, tokenizer = loader.FastCPUModel.from_pretrained(TINY, dtype = dtype, full_finetuning = True, load_in_4bit = False)
    except OSError as error:  # offline without the tiny model cached
        pytest.skip(f"{TINY} unavailable: {error}")
    assert getattr(model, "_unsloth_cpu_chunked_ce", False)
    config_cls, trainer_cls = _orpo()
    words = "the a cat dog sat ran on under mat tree big small red blue".split()
    g = torch.Generator().manual_seed(0)
    text = lambda n: " ".join(words[int(i)] for i in torch.randint(0, len(words), (n,), generator = g))  # noqa: E731
    # Uneven lengths: padding inside both the chosen and the rejected halves.
    rows = [{"prompt": text(3 + i), "chosen": text(4 + 2 * i), "rejected": text(9 - i)} for i in range(4)]
    values = dict(output_dir = str(tmp_path), per_device_train_batch_size = 4, max_length = 48, max_prompt_length = 16,
                  report_to = "none", bf16 = bf16, beta = 0.1, remove_unused_columns = False, use_cpu = True)
    fields = {f.name for f in dataclasses.fields(config_cls)}
    args = config_cls(**{k: v for k, v in values.items() if k in fields})
    trainer = trainer_cls(model = model, args = args, train_dataset = Dataset.from_list(rows), processing_class = tokenizer)
    return trainer, model


def _step(trainer, model, forward):
    trainer.concatenated_forward = forward
    model.zero_grad(set_to_none = True)
    calls = []
    hook = model.get_output_embeddings().register_forward_hook(lambda *a: calls.append(1))
    try:
        with trainer.accelerator.autocast():
            loss, metrics = trainer.get_batch_loss_metrics(model, trainer._unsloth_test_batch, train_eval = "train")
        loss.backward()
    finally:
        hook.remove()
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
    metrics = {k: float(v) for k, v in metrics.items()}
    return float(loss), metrics, grads, len(calls)


@pytest.mark.parametrize("dtype, bf16, tol", [(torch.float32, False, 1e-5), (torch.bfloat16, True, 2e-2)])
def test_chunked_orpo_matches_trl_full_logits(tmp_path, dtype, bf16, tol):
    from unsloth_zoo.cpu.runtime import cpu_bf16_supported
    if bf16 and not cpu_bf16_supported():
        pytest.skip("no native bfloat16")
    trainer, model = _trainer(tmp_path, dtype, bf16)
    chunked = trainer.concatenated_forward
    assert getattr(chunked, "_unsloth_cpu", False), "the CPU ORPO forward was not installed"
    stock = type(trainer).concatenated_forward.__get__(trainer)
    trainer._unsloth_test_batch = next(iter(trainer.get_train_dataloader()))
    assert int((trainer._unsloth_test_batch["chosen_attention_mask"] == 0).sum()) > 0
    loss_c, metrics_c, grads_c, head_calls_c = _step(trainer, model, chunked)
    loss_s, metrics_s, grads_s, head_calls_s = _step(trainer, model, stock)
    # The chunked path never runs lm_head (no [2B, L, V] logits); TRL's runs it once.
    assert (head_calls_c, head_calls_s) == (0, 1)
    assert loss_c == pytest.approx(loss_s, rel = tol, abs = tol)
    assert metrics_c.keys() == metrics_s.keys()
    for key in metrics_s:
        if "accuracies" in key and dtype != torch.float32:
            continue  # chosen > rejected flips on bf16 rounding of near-equal random-model log-probs
        assert metrics_c[key] == pytest.approx(metrics_s[key], rel = tol, abs = tol), key
    assert grads_c.keys() == grads_s.keys() and len(grads_s) > 0
    for name, grad in grads_s.items():
        scale = float(grad.abs().max()) or 1.0
        assert float((grads_c[name] - grad).abs().max()) / scale < (1e-4 if dtype == torch.float32 else 0.1), name


def test_opt_out_and_subclasses_keep_trl_forward(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_CPU_CHUNKED_PREFERENCE", "0")
    trainer, _ = _trainer(tmp_path, torch.float32, False)
    assert not getattr(trainer.concatenated_forward, "_unsloth_cpu", False)
    monkeypatch.delenv("UNSLOTH_CPU_CHUNKED_PREFERENCE")
    from unsloth_zoo.cpu.trainer import _install_chunked_preference_forward
    _, trainer_cls = _orpo()

    class Custom(trainer_cls):
        def concatenated_forward(self, model, batch):
            return super().concatenated_forward(model, batch)

    trainer.__class__ = Custom
    trainer.__dict__.pop("concatenated_forward", None)
    assert _install_chunked_preference_forward(trainer) is False

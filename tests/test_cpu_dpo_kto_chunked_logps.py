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

"""CPU DPO / KTO on TRL 0.2x: log-probs from the last hidden state in row chunks, equal to TRL's full-logits
forwards. TRL 1.x has neither forward (its fused LM head is already chunked on CPU), so these tests skip there."""

import dataclasses
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tiny_qwen3 import tiny_snapshot  # noqa: E402



def _trl_cls(kind):
    import trl
    config_cls = getattr(trl, f"{kind}Config")
    trainer_cls = getattr(trl, f"{kind}Trainer")
    method = "concatenated_forward" if kind == "DPO" else "forward"
    if getattr(trainer_cls, method, None) is None:
        pytest.skip(f"this TRL's {kind}Trainer has no {method} (fused LM head)")
    return config_cls, trainer_cls, method


def _rows(kind):
    words = "the a cat dog sat ran on under mat tree big small red blue".split()
    g = torch.Generator().manual_seed(0)
    text = lambda n: " ".join(words[int(i)] for i in torch.randint(0, len(words), (n,), generator = g))  # noqa: E731
    # Uneven lengths: padding inside the prompt, chosen and rejected halves.
    if kind == "KTO":
        return [{"prompt": text(3 + i), "completion": text(4 + 3 * (i % 3)), "label": i % 2 == 0} for i in range(4)]
    return [{"prompt": text(3 + i), "chosen": text(4 + 2 * i), "rejected": text(9 - i)} for i in range(4)]


def _trainer(tmp_path, kind, dtype = torch.float32, bf16 = False, lora = True, **config):
    from datasets import Dataset
    from unsloth_zoo.cpu import loader
    from unsloth_zoo.cpu.trainer import install_cpu_training_defaults
    install_cpu_training_defaults()
    model, tokenizer = loader.FastCPUModel.from_pretrained(tiny_snapshot(), dtype = dtype, full_finetuning = not lora,
                                                           load_in_4bit = False)
    assert getattr(model, "_unsloth_cpu_chunked_ce", False)
    if lora:  # frozen lm_head; the reference forward runs with the adapters disabled
        model = loader.FastCPUModel.get_peft_model(model, r = 8, target_modules = ["q_proj", "v_proj"])
    config_cls, trainer_cls, method = _trl_cls(kind)
    values = dict(output_dir = str(tmp_path), per_device_train_batch_size = 4, max_length = 48, max_prompt_length = 16,
                  report_to = "none", bf16 = bf16, beta = 0.1, remove_unused_columns = False, use_cpu = True)
    values.update(config)
    fields = {f.name for f in dataclasses.fields(config_cls)}
    args = config_cls(**{k: v for k, v in values.items() if k in fields})
    trainer = trainer_cls(model = model, ref_model = None, args = args, train_dataset = Dataset.from_list(_rows(kind)),
                          processing_class = tokenizer)
    return trainer, model, method


def _base_head(model):
    head = model.get_output_embeddings()
    return getattr(head, "original_module", head)


def _count_head_calls(model, fn):
    calls = []
    head = _base_head(model)
    linear_forward = torch.nn.Linear.forward

    def counting(self, x):
        if self is head:
            calls.append(1)
        return linear_forward(self, x)

    torch.nn.Linear.forward = counting
    try:
        result = fn()
    finally:
        torch.nn.Linear.forward = linear_forward
    return result, len(calls)


def _step(trainer, model, method, forward, batch, backward = True):
    setattr(trainer, method, forward)
    model.zero_grad(set_to_none = True)

    def run():
        with trainer.accelerator.autocast(), torch.set_grad_enabled(backward):
            loss, metrics = trainer.get_batch_loss_metrics(model, batch)
        if backward:
            loss.backward()
        return loss, metrics

    (loss, metrics), calls = _count_head_calls(model, run)
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
    return float(loss.detach()), {k: float(v) for k, v in metrics.items()}, grads, calls


def _close(a, b, tol, key):
    if isinstance(a, torch.Tensor):
        assert a.shape == b.shape, key
        assert torch.allclose(a.float(), b.float(), rtol = tol, atol = tol, equal_nan = True), (key, a, b)
    else:
        assert a == pytest.approx(b, rel = tol, abs = tol, nan_ok = True), key


@pytest.fixture(autouse = True)
def _restore_accelerate_env():
    """TrainingArguments exports ACCELERATE_MIXED_PRECISION: a bf16 test would autocast every later trainer."""
    saved = {k: v for k, v in os.environ.items() if k.startswith("ACCELERATE_")}
    yield
    for key in [k for k in os.environ if k.startswith("ACCELERATE_")]:
        del os.environ[key]
    os.environ.update(saved)


def _compare(trainer, model, method, tol, grad_tol, chunk_rows = None, monkeypatch = None):
    chunked = getattr(trainer, method)
    assert getattr(chunked, "_unsloth_cpu", False), "the chunked CPU forward was not installed"
    stock = getattr(type(trainer), method).__get__(trainer)
    if chunk_rows is not None:  # several row chunks with a ragged last one
        monkeypatch.setenv("UNSLOTH_CPU_CE_CHUNK_ELEMENTS", str(chunk_rows * _base_head(model).weight.shape[0]))
    batch = next(iter(trainer.get_train_dataloader()))

    # Per-sample log-probs straight from the forward (policy, no grad), to 1e-5 in fp32.
    with torch.no_grad(), trainer.accelerator.autocast():
        out_c, calls_c = _count_head_calls(model, lambda: chunked(model, batch))
        out_s, calls_s = _count_head_calls(model, lambda: stock(model, batch))
    assert (calls_c, calls_s) == (0, 2 if method == "forward" and trainer.calculate_KL else 1)
    if isinstance(out_s, dict):
        assert out_c.keys() == out_s.keys()
        for key in out_s:
            _close(out_c[key], out_s[key], tol, key)
    else:
        assert len(out_c) == len(out_s)
        for i, (c, s) in enumerate(zip(out_c, out_s)):
            if i in (2, 3):  # KTO's logits: TRL only ever takes .nansum()
                _close(c.float().nansum(), s.float().nansum(), max(tol, 1e-5) * 10, f"logits_sum[{i}]")
            elif s is None:
                assert c is None
            else:
                _close(c, s, tol, f"output[{i}]")

    loss_c, metrics_c, grads_c, head_c = _step(trainer, model, method, chunked, batch)
    loss_s, metrics_s, grads_s, head_s = _step(trainer, model, method, stock, batch)
    assert head_c == 0 and head_s > 0
    _close(loss_c, loss_s, tol, "loss")
    assert metrics_c.keys() == metrics_s.keys()
    for key in metrics_s:
        if "accuracies" in key and (tol > 1e-4 or abs(metrics_s.get("rewards/margins", 1.0)) < 1e-4):
            # chosen > rejected flips on rounding: bf16, or a full finetune's first step (policy == reference
            # copy, every reward about 0; the trainable-head policy and the frozen reference use different kernels)
            continue
        _close(metrics_c[key], metrics_s[key], tol * (10 if "logits" in key else 1), key)
    assert grads_c.keys() == grads_s.keys() and len(grads_s) > 0
    for name, grad in grads_s.items():
        scale = float(grad.abs().max()) or 1.0
        assert float((grads_c[name] - grad).abs().max()) / scale < grad_tol, name
    setattr(trainer, method, chunked)


DPO_VARIANTS = [
    dict(),
    dict(rpo_alpha = 0.5, use_weighting = True),
    dict(loss_type = "ipo", ld_alpha = 0.5),
    dict(use_logits_to_keep = True),
    dict(max_length = 20, truncation_mode = "keep_end"),
    dict(max_length = 20, truncation_mode = "keep_start", loss_type = ["sigmoid", "sft"], loss_weights = [1.0, 1.0]),
]


@pytest.mark.parametrize("config", DPO_VARIANTS, ids = lambda c: ",".join(f"{k}={v}" for k, v in c.items()) or "default")
@pytest.mark.parametrize("lora", [True, False], ids = ["lora", "full"])
def test_chunked_dpo_matches_trl_full_logits(tmp_path, monkeypatch, config, lora):
    if isinstance(config.get("loss_type"), list):
        import trl.trainer.dpo_config as dpo_config
        if "loss_weights" not in {f.name for f in dataclasses.fields(dpo_config.DPOConfig)}:
            pytest.skip("this TRL has a single DPO loss_type")
    trainer, model, method = _trainer(tmp_path, "DPO", lora = lora, **config)
    _compare(trainer, model, method, 1e-5, 1e-4, chunk_rows = 37, monkeypatch = monkeypatch)


@pytest.mark.parametrize("lora", [True, False], ids = ["lora", "full"])
def test_chunked_kto_matches_trl_full_logits(tmp_path, monkeypatch, lora):
    trainer, model, method = _trainer(tmp_path, "KTO", lora = lora)
    assert trainer.calculate_KL
    _compare(trainer, model, method, 1e-5, 1e-4, chunk_rows = 37, monkeypatch = monkeypatch)


@pytest.mark.parametrize("kind", ["DPO", "KTO"])
def test_chunked_bf16_close_to_trl(tmp_path, kind):
    from unsloth_zoo.cpu.runtime import cpu_bf16_supported
    if not cpu_bf16_supported():
        pytest.skip("no native bfloat16")
    trainer, model, method = _trainer(tmp_path, kind, dtype = torch.bfloat16, bf16 = True)
    _compare(trainer, model, method, 2e-2, 0.1)


@pytest.mark.parametrize("preference", ["1", "0"], ids = ["chunked", "trl"])
def test_bf16_kto_lora_step_trains_the_adapters(tmp_path, monkeypatch, preference):
    """KTO scores the KL completions under no_grad before the policy pass, inside the trainer's one autocast region:
    a cached cast of the fp32 adapters left them without gradients, and gradient checkpointing's recompute failed.
    UNSLOTH_CPU_BF16=1 forces bf16 so runners without native bf16 run it too (emulated, slower)."""
    from unsloth_zoo.cpu import runtime
    monkeypatch.setenv("UNSLOTH_CPU_BF16", "1")
    monkeypatch.setenv("UNSLOTH_CPU_CHUNKED_PREFERENCE", preference)
    runtime.cpu_bf16_supported.cache_clear()
    try:
        trainer, model, method = _trainer(tmp_path, "KTO", dtype = torch.bfloat16, bf16 = True, max_steps = 1,
                                          learning_rate = 1e-2, save_strategy = "no", logging_steps = 1)
        assert trainer.args.bf16 and trainer.accelerator.mixed_precision == "bf16"
        assert getattr(getattr(trainer, method), "_unsloth_cpu", False) is (preference == "1")
        before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        assert before and all(p.dtype == torch.float32 for p in before.values())
        trainer.train()
    finally:
        runtime.cpu_bf16_supported.cache_clear()
    moved = [n for n, p in model.named_parameters() if n in before and not torch.equal(before[n], p.detach())]
    # LoRA B starts at zero, so the first step moves exactly the B matrices.
    assert sorted(moved) == sorted(n for n in before if "lora_B" in n)


@pytest.mark.parametrize("kind", ["DPO", "KTO"])
def test_opt_out_and_subclasses_keep_trl_forward(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("UNSLOTH_CPU_CHUNKED_PREFERENCE", "0")
    trainer, _, method = _trainer(tmp_path, kind)
    assert not getattr(getattr(trainer, method), "_unsloth_cpu", False)
    monkeypatch.delenv("UNSLOTH_CPU_CHUNKED_PREFERENCE")
    from unsloth_zoo.cpu.trainer import _install_chunked_preference_forward
    trainer_cls = type(trainer)
    overrides = [method, "get_batch_loss_metrics"] + (["compute_ref_log_probs"] if kind == "DPO" else ["_compute_kl_logps"])
    classes = []
    for name in overrides:
        original = getattr(trainer_cls, name)
        classes.append(type(f"Custom_{name}", (trainer_cls,), {name: lambda self, *a, _o = original, **k: _o(self, *a, **k)}))
    for cls in (*classes, trainer_cls):
        trainer.__class__ = cls
        trainer.__dict__.pop(method, None)
        assert _install_chunked_preference_forward(trainer) is (cls is trainer_cls), cls.__name__


def test_hooked_head_keeps_trl_scoring(tmp_path):
    trainer, model, method = _trainer(tmp_path, "DPO", lora = False)
    chunked = getattr(trainer, method)
    stock = getattr(type(trainer), method).__get__(trainer)
    batch = next(iter(trainer.get_train_dataloader()))
    handle = model.get_output_embeddings().register_forward_hook(lambda mod, i, o: o * 2)
    try:
        with torch.no_grad():
            c, s = chunked(model, batch), stock(model, batch)
    finally:
        handle.remove()
    for key in s:
        _close(c[key], s[key], 1e-6, key)

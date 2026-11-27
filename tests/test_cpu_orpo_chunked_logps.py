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
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tiny_qwen3 import tiny_snapshot  # noqa: E402


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


def _trainer(tmp_path, dtype, bf16, lora = False, trainer_cls = None):
    from datasets import Dataset
    from unsloth_zoo.cpu import loader
    from unsloth_zoo.cpu.trainer import install_cpu_training_defaults
    install_cpu_training_defaults()
    model, tokenizer = loader.FastCPUModel.from_pretrained(tiny_snapshot(), dtype = dtype, full_finetuning = not lora,
                                                           load_in_4bit = False)
    assert getattr(model, "_unsloth_cpu_chunked_ce", False)
    if lora:  # frozen lm_head: the _ChunkedLogProbs path behind PEFT's wrappers
        model = loader.FastCPUModel.get_peft_model(model, r = 8, target_modules = ["q_proj", "v_proj"])
    config_cls, default_cls = _orpo()
    trainer_cls = trainer_cls or default_cls
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


def _base_head(model):
    head = model.get_output_embeddings()
    return getattr(head, "original_module", head)


def _step(trainer, model, forward, backward = True):
    trainer.concatenated_forward = forward
    model.zero_grad(set_to_none = True)
    calls = []
    head = _base_head(model)
    # Counted on the class: a hook or an instance forward on the head itself sends it back to TRL's forward.
    linear_forward = torch.nn.Linear.forward

    def counting(self, x):
        if self is head:
            calls.append(1)
        return linear_forward(self, x)

    torch.nn.Linear.forward = counting
    try:
        with trainer.accelerator.autocast(), torch.set_grad_enabled(backward):
            loss, metrics = trainer.get_batch_loss_metrics(model, trainer._unsloth_test_batch, train_eval = "train")
        if backward:
            loss.backward()
    finally:
        torch.nn.Linear.forward = linear_forward
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
    metrics = {k: float(v) for k, v in metrics.items()}
    return float(loss.detach()), metrics, grads, len(calls)


@pytest.fixture(autouse = True)
def _restore_accelerate_env():
    """TrainingArguments exports ACCELERATE_MIXED_PRECISION: a bf16 test would autocast every later trainer."""
    saved = {k: v for k, v in os.environ.items() if k.startswith("ACCELERATE_")}
    yield
    for key in [k for k in os.environ if k.startswith("ACCELERATE_")]:
        del os.environ[key]
    os.environ.update(saved)


@pytest.mark.parametrize("dtype, bf16, tol, lora, chunk_rows", [
    (torch.float32, False, 1e-5, False, 37), (torch.bfloat16, True, 2e-2, False, None),
    (torch.float32, False, 1e-5, True, 37),
])
def test_chunked_orpo_matches_trl_full_logits(tmp_path, monkeypatch, dtype, bf16, tol, lora, chunk_rows):
    from unsloth_zoo.cpu.runtime import cpu_bf16_supported
    if bf16 and not cpu_bf16_supported():
        pytest.skip("no native bfloat16")
    trainer, model = _trainer(tmp_path, dtype, bf16, lora = lora)
    if chunk_rows is not None:  # several row chunks with a ragged last one
        monkeypatch.setenv("UNSLOTH_CPU_CE_CHUNK_ELEMENTS", str(chunk_rows * _base_head(model).weight.shape[0]))
    assert _base_head(model).weight.requires_grad is not lora
    chunked = trainer.concatenated_forward
    assert getattr(chunked, "_unsloth_cpu", False), "the CPU ORPO forward was not installed"
    stock = type(trainer).concatenated_forward.__get__(trainer)
    trainer._unsloth_test_batch = next(iter(trainer.get_train_dataloader()))
    assert int((trainer._unsloth_test_batch["chosen_attention_mask"] == 0).sum()) > 0
    if chunk_rows is not None:
        batch = trainer._unsloth_test_batch
        length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        rows = 2 * batch["chosen_input_ids"].shape[0] * (length - 1)
        assert rows > 2 * chunk_rows and rows % chunk_rows, rows
    loss_c, metrics_c, grads_c, head_calls_c = _step(trainer, model, chunked)
    loss_s, metrics_s, grads_s, head_calls_s = _step(trainer, model, stock)
    # The chunked path never runs lm_head (no [2B, L, V] logits); TRL's runs it once.
    assert (head_calls_c, head_calls_s) == (0, 1)
    assert loss_c == pytest.approx(loss_s, rel = tol, abs = tol)
    # Evaluation (no grad): the frozen-scorer path.
    eval_c, _, _, eval_calls_c = _step(trainer, model, chunked, backward = False)
    eval_s, _, _, _ = _step(trainer, model, stock, backward = False)
    assert eval_calls_c == 0 and eval_c == pytest.approx(eval_s, rel = tol, abs = tol)
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

    # Overriding what the chunked forward replaces (the scoring) or feeds (the [B, L, V] logits consumer).
    class CustomLogps(trainer_cls):
        @staticmethod
        def get_batch_logps(logits, labels, average_log_prob = False, **kwargs):
            return trainer_cls.get_batch_logps(logits, labels, average_log_prob = False, **kwargs)

    class CustomMetrics(trainer_cls):
        def get_batch_loss_metrics(self, model, batch, train_eval = "train"):
            return super().get_batch_loss_metrics(model, batch, train_eval)

    for cls in (CustomLogps, CustomMetrics, trainer_cls):
        trainer.__class__ = cls
        trainer.__dict__.pop("concatenated_forward", None)
        assert _install_chunked_preference_forward(trainer) is (cls is trainer_cls), cls.__name__


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    from unsloth_zoo.cpu.ops import patch_chunked_causal_lm_loss
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size = 64, hidden_size = 16, intermediate_size = 32, num_hidden_layers = 1,
                         num_attention_heads = 2, num_key_value_heads = 2, tie_word_embeddings = False)
    model = LlamaForCausalLM(config)
    assert patch_chunked_causal_lm_loss(model)
    return model


def test_plain_head_weight_disabled_adapters_and_hooks():
    """The raw-weight projection must equal calling the head: PEFT's disable_adapter() runs the base head, and hooks
    or an instance forward act on the module call."""
    from peft import LoraConfig, get_peft_model
    from unsloth_zoo.cpu.ops import _plain_head_weight
    model = get_peft_model(_tiny_llama(), LoraConfig(r = 2, target_modules = ["q_proj"], modules_to_save = ["lm_head"]))
    head = model.get_output_embeddings()
    assert _plain_head_weight(head) is head.modules_to_save["default"].weight
    with model.disable_adapter():
        assert _plain_head_weight(head) is head.original_module.weight
    assert _plain_head_weight(head) is head.modules_to_save["default"].weight

    linear = torch.nn.Linear(4, 8, bias = False)
    assert _plain_head_weight(linear) is linear.weight
    for register in (lambda m: m.register_forward_hook(lambda mod, i, o: o * 2),
                     lambda m: m.register_forward_pre_hook(lambda mod, i: i),
                     lambda m: m.register_full_backward_hook(lambda mod, gi, go: None)):
        handle = register(linear)
        assert _plain_head_weight(linear) is None
        handle.remove()
    linear.forward = lambda x: torch.nn.functional.linear(x, linear.weight) * 2
    assert _plain_head_weight(linear) is None


def test_hooked_head_keeps_trl_scoring(tmp_path):
    trainer, model = _trainer(tmp_path, torch.float32, False)
    chunked = trainer.concatenated_forward
    stock = type(trainer).concatenated_forward.__get__(trainer)
    trainer._unsloth_test_batch = next(iter(trainer.get_train_dataloader()))
    handle = model.get_output_embeddings().register_forward_hook(lambda mod, i, o: o * 2)
    try:
        loss_c = _step(trainer, model, chunked, backward = False)[0]
        loss_s = _step(trainer, model, stock, backward = False)[0]
    finally:
        handle.remove()
    assert loss_c == pytest.approx(loss_s, rel = 1e-6, abs = 1e-6)


def test_one_token_rows_match_trl(tmp_path):
    """One-token rows leave no next-token target: TRL returns NaN (0 / 0), the chunked forward must not raise."""
    trainer, model = _trainer(tmp_path, torch.float32, False)
    chunked = trainer.concatenated_forward
    stock = type(trainer).concatenated_forward.__get__(trainer)
    batch = next(iter(trainer.get_train_dataloader()))
    trainer._unsloth_test_batch = {k: v[:, :1] if torch.is_tensor(v) and v.ndim == 2 else v for k, v in batch.items()}
    loss_c, metrics_c = _step(trainer, model, chunked, backward = False)[:2]
    loss_s, metrics_s = _step(trainer, model, stock, backward = False)[:2]
    assert torch.isnan(torch.tensor(loss_s)) and torch.isnan(torch.tensor(loss_c))
    assert metrics_c.keys() == metrics_s.keys()


def test_head_off_cpu_keeps_trl_forward():
    """The chunked scorer allocates CPU buffers: a CPU-patched model moved to another device keeps TRL's forward."""
    import types
    from unsloth_zoo.cpu.trainer import _orpo_chunked_forward
    model = _tiny_llama().to("meta")
    trainer = types.SimpleNamespace(is_encoder_decoder = False, aux_loss_enabled = False, label_pad_token_id = -100)
    forward = _orpo_chunked_forward(trainer, lambda model, batch: "trl")
    assert forward(model, {}) == "trl"


def test_trainable_head_metrics_only_requested(monkeypatch):
    """Trainable head: only the requested detached metrics are computed (ORPO asks for mean_logits alone), and the
    values match an fp64 oracle."""
    from unsloth_zoo.cpu.ops import cpu_chunked_log_probs
    torch.manual_seed(0)
    hidden = torch.randn(13, 8, requires_grad = True)
    weight = torch.nn.Parameter(torch.randn(32, 8))
    targets = torch.randint(0, 32, (13,))
    logits = hidden.double() @ weight.double().t()
    oracle_lp = torch.log_softmax(logits, -1).gather(-1, targets[:, None]).squeeze(-1)
    oracle_mean = logits.mean(-1)
    monkeypatch.setenv("UNSLOTH_CPU_CE_CHUNK_ELEMENTS", str(5 * 32))
    calls = []
    for name in ("log_softmax", "logsumexp"):
        original = getattr(torch, name)
        monkeypatch.setattr(torch, name, lambda *a, _f = original, _n = name, **k: calls.append(_n) or _f(*a, **k))
    log_probs, entropy, sq, mean, top1 = cpu_chunked_log_probs(hidden, weight, None, targets, 1.0, None,
                                                               outputs = ("mean_logits",))
    monkeypatch.undo()
    assert entropy is None and sq is None and top1 is None
    # 3 row chunks: one logsumexp each in the differentiable pass, none for the metrics pass.
    assert calls == ["logsumexp"] * 3, calls
    torch.testing.assert_close(log_probs.double(), oracle_lp, rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(mean.double(), oracle_mean, rtol = 1e-5, atol = 1e-5)
    log_probs.sum().backward()
    oracle_hidden, oracle_weight = hidden.detach().double().requires_grad_(), weight.detach().double().requires_grad_()
    torch.log_softmax(oracle_hidden @ oracle_weight.t(), -1).gather(-1, targets[:, None]).sum().backward()
    torch.testing.assert_close(hidden.grad.double(), oracle_hidden.grad, rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(weight.grad.double(), oracle_weight.grad, rtol = 1e-5, atol = 1e-5)


_DDP_SCRIPT = r'''
import dataclasses, json, os, sys, tempfile
import torch, torch.distributed as dist
from datasets import Dataset
from transformers import TrainerCallback
from unsloth_zoo.cpu import loader
from unsloth_zoo.cpu.trainer import install_cpu_training_defaults
install_cpu_training_defaults()
model, tok = loader.FastCPUModel.from_pretrained(sys.argv[1], dtype = torch.float32, full_finetuning = True,
                                                 load_in_4bit = False)
head = model.get_output_embeddings()
head.weight = torch.nn.Parameter(head.weight.detach().clone())  # untied, trainable lm_head
model.config.tie_word_embeddings = False
import trl
if hasattr(trl, "ORPOTrainer"):
    C, T = trl.ORPOConfig, trl.ORPOTrainer
else:
    from trl.experimental.orpo import ORPOConfig as C, ORPOTrainer as T
words = "the a cat dog sat ran on under mat tree big small red blue".split()
g = torch.Generator().manual_seed(0)
text = lambda n: " ".join(words[int(i)] for i in torch.randint(0, len(words), (n,), generator = g))
rows = [{"prompt": text(3 + i % 4), "chosen": text(4 + 2 * (i % 4)), "rejected": text(9 - i % 4)} for i in range(8)]
values = dict(output_dir = tempfile.mkdtemp(), per_device_train_batch_size = 4, max_length = 48, max_prompt_length = 16,
              report_to = "none", beta = 0.1, remove_unused_columns = False, use_cpu = True, max_steps = 1, bf16 = False,
              ddp_find_unused_parameters = True, ddp_backend = "gloo", save_strategy = "no", seed = 0)
fields = {f.name for f in dataclasses.fields(C)}
seen = {}
head.weight.register_hook(lambda grad: seen.setdefault("local", grad.detach().clone()))

class Grab(TrainerCallback):
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        seen.setdefault("reduced", head.weight.grad.detach().clone())

trainer = T(model = model, args = C(**{k: v for k, v in values.items() if k in fields}), callbacks = [Grab()],
            train_dataset = Dataset.from_list(rows), processing_class = tok)
trainer.train()


def gather(tensor):
    parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, tensor)
    return parts


local, reduced = gather(seen["local"]), gather(seen["reduced"])
if dist.get_rank() == 0:
    torch.save(dict(chunked = getattr(trainer.concatenated_forward, "_unsloth_cpu", False),
                    wrapped = type(trainer.model_wrapped).__name__, local = local, reduced = reduced), sys.argv[2])
'''


def _ddp_step(tmp_path, chunked, model_dir):
    import socket
    import subprocess
    import sys
    script = tmp_path / "ddp_orpo.py"
    script.write_text(_DDP_SCRIPT)
    result = tmp_path / f"result_{chunked}.pt"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Not ACCELERATE_*: TrainingArguments exports ACCELERATE_MIXED_PRECISION, so earlier tests would set bf16 here.
    env = {k: v for k, v in os.environ.items() if not k.startswith("ACCELERATE_")}
    env.update(UNSLOTH_CPU_CHUNKED_PREFERENCE = chunked, CUDA_VISIBLE_DEVICES = "", OMP_NUM_THREADS = "2",
               PYTHONPATH = os.pathsep.join([root, os.environ.get("PYTHONPATH", "")]))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    run = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2",
                          "--master_port", str(port), str(script), model_dir, str(result)],
                         env = env, capture_output = True, text = True, timeout = 900)
    assert run.returncode == 0, run.stderr[-3000:]
    return torch.load(result)


def test_ddp_trainable_head_grads_synced(tmp_path):
    """CPU gloo DDP, 2 ranks with different rows, untied trainable lm_head, find_unused_parameters = True (the Trainer
    default without gradient checkpointing): the head runs inside DDP's forward, so the step runs and the head
    gradient is all-reduced across ranks, equal to TRL's own forward."""
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("no gloo")
    model_dir = tiny_snapshot()
    # The ranks load this snapshot, not the repo id: conftest points HF_HOME at an empty per-test dir that the
    # torchrun children inherit, and an HF_HUB_OFFLINE=1 set earlier in this worker would leave them nothing to load.
    chunked, stock = _ddp_step(tmp_path, "1", model_dir), _ddp_step(tmp_path, "0", model_dir)
    assert (chunked["chunked"], stock["chunked"]) == (True, False)
    for out in (chunked, stock):
        assert out["wrapped"] == "DistributedDataParallel"
        local, reduced = out["local"], out["reduced"]
        assert not torch.equal(local[0], local[1]), "ranks saw the same rows"
        assert torch.equal(reduced[0], reduced[1]), "lm_head gradient not synced across ranks"
        torch.testing.assert_close(reduced[0], (local[0] + local[1]) / 2, rtol = 1e-4, atol = 1e-7)
    scale = float(stock["reduced"][0].abs().max())
    assert float((chunked["reduced"][0] - stock["reduced"][0]).abs().max()) / scale < 1e-4

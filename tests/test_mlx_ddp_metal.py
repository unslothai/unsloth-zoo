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

import json, os, subprocess, sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
if "mlx_simulation" in str(getattr(mx, "__file__", "")):
    pytest.skip("requires real MLX runtime", allow_module_level=True)


def test_mlx_launch_ordered_text_batches_are_rank_sharded(tmp_path):
    launcher = Path(sys.executable).with_name("mlx.launch")
    if not launcher.exists():
        pytest.skip("mlx.launch not found beside current Python executable")

    script = tmp_path / "ddp_batch_probe.py"
    script.write_text("""import json, sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from unsloth_zoo import dataset_utils
import unsloth_zoo.mlx.trainer as trainer_mod
from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig, _create_labeled_batches, train_on_responses_only
from unsloth_zoo.mlx.utils import _MLXIterableTokenizedDatasetView, create_batches, create_ordered_batches, create_vlm_batches, iterate_training_batches, iterate_vlm_training_batches, make_baseline_loss_fn

class TinyTokenizer:
    pad_token_id = eos_token_id = 2
    def encode(self, text): return [int(part) for part in str(text).split()]
    def __call__(self, text, **_kwargs): return {"input_ids": self.encode(text)}

class TinyProcessor:
    tokenizer = TinyTokenizer()
    image_processor = object()
    def __call__(self, text, **_kwargs):
        raw_rows = []
        for item in text:
            value = int(str(item))
            raw_rows.append([value, 20, 2] if value % 2 else [value, 20, value + 1, 2])
        width = max(len(row) for row in raw_rows)
        rows = [row + [2] * (width - len(row)) for row in raw_rows]
        masks = [[1] * len(row) + [0] * (width - len(row)) for row in raw_rows]
        return {"input_ids": mx.array(rows, dtype=mx.int32), "attention_mask": mx.array(masks, dtype=mx.int32)}

def keep_all_labels(d): return {"labels": [list(d["input_ids"][0])]}
def first_token_rows(batches):
    return [[int(row[0]) for row in batch.tolist()] for batch, _, _ in batches]
def take_stream_rows(iterator, count):
    rows = []
    for _ in range(count):
        item = next(iterator); batch = item["input_ids"] if isinstance(item, dict) else item[0]; rows.append([int(row[0]) for row in batch.tolist()])
    return rows
def eval_loss_for_batches(model, batches):
    loss_fn = make_baseline_loss_fn()
    all_losses = mx.array(0.0)
    ntokens = mx.array(0)
    for batch, lengths, labels in batches:
        loss, ntoks = loss_fn(model, batch, lengths, labels)
        all_losses += loss * ntoks
        ntokens += ntoks
    mx.eval(all_losses, ntokens)
    return float((all_losses / ntokens).item())

class ReplayableStream:
    @property
    def _ex_iterable(self):
        if int(world.rank()) != 0: raise AssertionError("non-owner inspected eval stream")
        return None
    def __iter__(self):
        if int(world.rank()) != 0: raise AssertionError("non-owner consumed text stream")
        return ({"text": f"{i} {i + 20} {i + 30}"} for i in range(10, 15))

class ReplayableLabeledStream:
    def __iter__(self):
        if int(world.rank()) != 0: raise AssertionError("non-owner consumed labeled stream")
        return ({"input_ids": [i, i + 20], "labels": [-100, i + 20]} for i in range(10, 13))

class ReplayableVariableStream:
    def __iter__(self):
        if int(world.rank()) != 0: raise AssertionError("non-owner consumed variable stream")
        return iter(({"input_ids": [10, 30]}, {"input_ids": [11, 31, 41, 51]}, {"input_ids": [12, 32]}))

class RaisingMetadataStream:
    @property
    def _distributed(self): raise RuntimeError("owner metadata failure")
    def __iter__(self): raise AssertionError("metadata failure must precede consumption")

class RankOwnedLengthStream:
    def __len__(self):
        if int(world.rank()) != 0: raise AssertionError("non-owner inspected stream length")
        return 4
    def __iter__(self):
        if int(world.rank()) != 0: raise AssertionError("non-owner consumed length stream")
        return ({"text": f"{i} {i + 20}"} for i in range(10, 14))

class RankFailingStream:
    def __iter__(self):
        if int(world.rank()) == 0:
            raise KeyError("rank0 stream failure")
        return ({"text": f"{i} {i + 20} {i + 30}"} for i in range(10, 15))

class ReplayableVLMStream:
    def __iter__(self): return ({"text": str(i)} for i in range(10, 15))

world = mx.distributed.init()
trainer = MLXTrainer.__new__(MLXTrainer)
trainer._distributed_world = world; trainer._distributed_initialized = True
trainer._distributed_rank = int(world.rank()); trainer._distributed_world_size = int(world.size())
trainer._distributed_is_main_process = world.rank() == 0
data = [{"text": f"{i} {i + 20} {i + 30}"} for i in range(10, 15)]
common = dict(batch_size=2, max_seq_length=8, comm_group=world)
trainer.stop_requested = world.rank() == 0
synced_stop = trainer._distributed_should_stop()
trainer.stop_requested = False

def stream_replay_probe():
    full = take_stream_rows(iterate_training_batches(ReplayableStream(), TinyTokenizer(), batch_size=1, max_seq_length=8, dataset_order="sequential", comm_group=world), 3)
    resumed = iterate_training_batches(ReplayableStream(), TinyTokenizer(), batch_size=1, max_seq_length=8, dataset_order="sequential", comm_group=world)
    next(resumed); next(resumed)
    return [full[2], take_stream_rows(resumed, 1)[0]]

def presharded_stream_error():
    from datasets import IterableDataset
    from datasets.distributed import split_dataset_by_node
    source = IterableDataset.from_generator(lambda: ({"text": f"{i} {i + 20}"} for i in range(4)))
    source = split_dataset_by_node(source, rank=int(world.rank()), world_size=int(world.size()))
    try:
        next(iterate_training_batches(source, TinyTokenizer(), batch_size=1, max_seq_length=8, dataset_order="sequential", comm_group=world))
    except (RuntimeError, ValueError) as exc:
        return str(exc)
    return ""

def owner_metadata_error():
    try:
        next(iterate_training_batches(RaisingMetadataStream(), TinyTokenizer(), batch_size=1, max_seq_length=8, dataset_order="sequential", comm_group=world))
    except RuntimeError as exc:
        return str(exc)
    return ""

def prepared_override_batches():
    override = TinyTokenizer(); override.pad_token_id = 9
    view = _MLXIterableTokenizedDatasetView(ReplayableVariableStream(), override, max_seq_length=8)
    return [batch.tolist() for batch, _lengths, _labels in iterate_training_batches(view, TinyTokenizer(), batch_size=1, max_seq_length=8, dataset_order="sequential", comm_group=world, repeat=False)]

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__(); self.embed = nn.Embedding(64, 8)
        self.proj = nn.Linear(8, 64, bias=False); self._config = {"model_type": "tiny"}
        self._record_eval_rows = False; self.eval_first_tokens = []
    def train(self, mode=True):
        self._record_eval_rows = not bool(mode)
        return super().train(mode)
    def eval(self):
        self._record_eval_rows = True
        return super().eval()
    def __call__(self, x):
        if self._record_eval_rows:
            self.eval_first_tokens.extend([int(row[0]) for row in x.tolist()])
        return self.proj(self.embed(x))

def epoch_owned_probe():
    epoch_args = MLXTrainingConfig(per_device_train_batch_size=1, gradient_accumulation_steps=1, max_steps=-1, num_train_epochs=1, max_seq_length=8, output_dir=str(Path(sys.argv[1], "epoch_owned")), use_cce=False, gradient_checkpointing=False, completion_only_loss=False, dataset_order="sequential", streaming=True)
    epoch_trainer = MLXTrainer(TinyLM(), TinyTokenizer(), RankOwnedLengthStream(), args=epoch_args)
    _batches, batch_iter = epoch_trainer._prepare_data(False)
    return [int(epoch_trainer._streaming_epoch_batch_count), take_stream_rows(batch_iter, 1)[0]]

class BrokenTinyLM(TinyLM):
    def __call__(self, x):
        raise RuntimeError("eval side failure from model")

class TinyVLM(nn.Module):
    _is_vlm_model = True
    __module__ = "mlx_vlm.models.qwen2_vl.fake"
    def __init__(self):
        super().__init__(); self.embed = nn.Embedding(64, 8)
        self.proj = nn.Linear(8, 64, bias=False); self._config = {"model_type": "tiny_vlm", "image_token_id": 20}
    def __call__(self, input_ids, **_kwargs): return self.proj(self.embed(input_ids))

def mark(name):
    def inner(*_args, **_kwargs):
        Path(sys.argv[1], f"{name}_rank{world.rank()}").write_text("1")
    return inner

real_save_trainable_adapters, real_save_optimizer_state, real_save_trainer_state = trainer_mod.save_trainable_adapters, trainer_mod.save_optimizer_state, trainer_mod.save_trainer_state
trainer_mod.save_trainable_adapters = mark("ckpt_adapter")
trainer_mod.save_optimizer_state = mark("ckpt_opt")
trainer_mod.save_trainer_state = mark("ckpt_state")
mx.random.seed(0)
train_data = [{"text": f"{i} {i + 1} {i + 2}"} for i in range(10, 18)]
args = MLXTrainingConfig(per_device_train_batch_size=1, gradient_accumulation_steps=2, max_steps=3, logging_steps=1, eval_steps=1, save_steps=1, learning_rate=1e-3, max_seq_length=8, output_dir=str(Path(sys.argv[1], "train_out")), use_cce=False, gradient_checkpointing=False, cast_norm_output_to_input_dtype=False, dataset_order="sequential")
loop_trainer = MLXTrainer(TinyLM(), TinyTokenizer(), train_data, eval_dataset=train_data[:3], args=args)
events = []
loop_trainer.save_model = mark("final")
loop_trainer.add_step_callback(lambda *args: events.append(["step", int(args[7])]))
loop_trainer.add_eval_callback(lambda *_args: (events.append(["eval"]), setattr(loop_trainer, "stop_requested", True)))
dataset_utils.train_on_responses_only = lambda *_args, **_kwargs: keep_all_labels
train_on_responses_only(loop_trainer, instruction_part="user", response_part="assistant")
response_rows = first_token_rows(loop_trainer._batches)
result = loop_trainer.train()
param_sum = mx.sum(loop_trainer.model.embed.weight) + mx.sum(loop_trainer.model.proj.weight)
loop_eval_reference_loss = eval_loss_for_batches(loop_trainer.model, _create_labeled_batches(train_data[:3], TinyTokenizer(), keep_all_labels, batch_size=1, max_seq_length=8, seed=args.seed, dataset_order="sequential"))
mx.random.seed(1)
stream_args = MLXTrainingConfig(per_device_train_batch_size=1, gradient_accumulation_steps=1, max_steps=1, logging_steps=1, eval_steps=1, save_steps=0, learning_rate=1e-3, max_seq_length=8, output_dir=str(Path(sys.argv[1], "stream_train_out")), use_cce=False, gradient_checkpointing=False, cast_norm_output_to_input_dtype=False, dataset_order="sequential", streaming=True)
stream_eval_data = [{"text": f"{i} {i + 20} {i + 30}"} for i in range(10, 15)]
stream_trainer = MLXTrainer(TinyLM(), TinyTokenizer(), ReplayableStream(), eval_dataset=ReplayableStream(), args=stream_args)
stream_events = []
stream_trainer.save_model = mark("stream_final")
stream_trainer.add_step_callback(lambda step, *_args: stream_events.append(["step", int(step)]))
stream_trainer.add_eval_callback(lambda step, loss, _ppl: stream_events.append(["eval", int(step), float(loss)]))
stream_result = stream_trainer.train()
stream_param_sum = mx.sum(stream_trainer.model.embed.weight) + mx.sum(stream_trainer.model.proj.weight)
stream_eval_reference_loss = eval_loss_for_batches(stream_trainer.model, create_batches(stream_eval_data, TinyTokenizer(), batch_size=1, max_seq_length=8, seed=stream_args.seed))
trainer_mod.save_trainable_adapters, trainer_mod.save_optimizer_state, trainer_mod.save_trainer_state = real_save_trainable_adapters, real_save_optimizer_state, real_save_trainer_state
parity_data = [{"text": f"{i} {i + 1} {i + 2}"} for i in range(10, 18)]
def make_parity_trainer(output_name, seed=123):
    mx.random.seed(seed)
    parity_args = MLXTrainingConfig(per_device_train_batch_size=1, gradient_accumulation_steps=2, max_steps=3, logging_steps=1, eval_steps=0, save_steps=2, learning_rate=1e-3, max_seq_length=8, output_dir=str(Path(sys.argv[1], output_name)), use_cce=False, gradient_checkpointing=False, cast_norm_output_to_input_dtype=False, dataset_order="sequential")
    out = MLXTrainer(TinyLM(), TinyTokenizer(), parity_data, args=parity_args)
    out.save_model = mark(f"{output_name}_final")
    return out
def max_trainable_delta(left, right):
    left_params = dict(tree_flatten(left.model.trainable_parameters()))
    right_params = dict(tree_flatten(right.model.trainable_parameters()))
    delta = mx.array(0.0, dtype=mx.float32)
    for name, value in left_params.items():
        other = right_params[name]
        leaf_delta = mx.max(mx.abs(value.astype(mx.float32) - other.astype(mx.float32)))
        delta = mx.maximum(delta, leaf_delta)
    mx.eval(delta)
    return float(delta.item())
parity_lengths_data = [
    {"text": "10 11 12 13 14"},
    {"text": "15 16 17"},
    {"text": "18 19 20 21"},
    {"text": "22 23"},
    {"text": "24 25 26 27 28"},
    {"text": "29 30 31"},
    {"text": "32 33 34 35"},
    {"text": "36 37"},
]
def make_ddp_compile_parity_trainer(output_name, compile_enabled):
    mx.random.seed(456)
    parity_args = MLXTrainingConfig(per_device_train_batch_size=1, gradient_accumulation_steps=2, max_steps=3, logging_steps=1, eval_steps=0, save_steps=0, learning_rate=1e-3, max_grad_norm=1.0, max_seq_length=8, output_dir=str(Path(sys.argv[1], output_name)), use_cce=False, compile=compile_enabled, gradient_checkpointing=False, cast_norm_output_to_input_dtype=False, dataset_order="sequential")
    out = MLXTrainer(TinyLM(), TinyTokenizer(), parity_lengths_data, args=parity_args)
    out.save_model = mark(f"{output_name}_final")
    return out
ddp_eager_trainer = make_ddp_compile_parity_trainer("ddp_parity_eager", False)
ddp_eager_result = ddp_eager_trainer.train()
sync = trainer._distributed_all_sum(mx.array(1, dtype=mx.int32), stream=mx.cpu)
mx.eval(sync)
ddp_compile_trainer = make_ddp_compile_parity_trainer("ddp_parity_compile", True)
ddp_compile_result = ddp_compile_trainer.train()
ddp_compile_param_delta = max_trainable_delta(ddp_eager_trainer, ddp_compile_trainer)
def strict_compile_failure_message():
    real_compile = mx.compile
    def failing_compile(_fn, *_args, **_kwargs):
        def fail(*_call_args, **_call_kwargs):
            raise RuntimeError("forced compile failure")
        return fail
    mx.compile = failing_compile
    try:
        mx.random.seed(789)
        strict_args = MLXTrainingConfig(per_device_train_batch_size=1, gradient_accumulation_steps=1, max_steps=1, logging_steps=1, eval_steps=0, save_steps=0, learning_rate=1e-3, max_seq_length=8, output_dir=str(Path(sys.argv[1], "strict_compile_failure")), use_cce=False, compile=True, compile_mode="strict", gradient_checkpointing=False, cast_norm_output_to_input_dtype=False, dataset_order="sequential")
        strict_trainer = MLXTrainer(TinyLM(), TinyTokenizer(), parity_lengths_data, args=strict_args)
        strict_trainer.save_model = mark("strict_compile_failure_final")
        strict_trainer.train()
    except RuntimeError as exc:
        return str(exc)
    finally:
        mx.compile = real_compile
    return ""
strict_error = strict_compile_failure_message()
def best_effort_compile_failure_result():
    real_compile = mx.compile
    def failing_compile(_fn, *_args, **_kwargs):
        def fail(*_call_args, **_call_kwargs):
            raise RuntimeError("forced compile failure")
        return fail
    mx.compile = failing_compile
    try:
        mx.random.seed(790)
        fallback_args = MLXTrainingConfig(per_device_train_batch_size=1, gradient_accumulation_steps=1, max_steps=1, logging_steps=1, eval_steps=0, save_steps=0, learning_rate=1e-3, max_seq_length=8, output_dir=str(Path(sys.argv[1], "best_effort_compile_failure")), use_cce=False, compile=True, gradient_checkpointing=False, cast_norm_output_to_input_dtype=False, dataset_order="sequential")
        fallback_trainer = MLXTrainer(TinyLM(), TinyTokenizer(), parity_lengths_data, args=fallback_args)
        fallback_trainer.save_model = mark("best_effort_compile_failure_final")
        return fallback_trainer.train()
    finally:
        mx.compile = real_compile
best_effort_fallback_result = best_effort_compile_failure_result()
def non_compile_runtime_error_message():
    mx.random.seed(792)
    error_args = MLXTrainingConfig(per_device_train_batch_size=1, gradient_accumulation_steps=1, max_steps=1, logging_steps=1, eval_steps=0, save_steps=0, learning_rate=1e-3, max_seq_length=8, output_dir=str(Path(sys.argv[1], "non_compile_runtime_error")), use_cce=False, compile=True, gradient_checkpointing=False, cast_norm_output_to_input_dtype=False, dataset_order="sequential")
    error_trainer = MLXTrainer(BrokenTinyLM(), TinyTokenizer(), parity_lengths_data, args=error_args)
    error_trainer.save_model = mark("non_compile_runtime_error_final")
    try:
        error_trainer.train()
    except RuntimeError as exc:
        return str(exc)
    return ""
non_compile_error = non_compile_runtime_error_message()
def asymmetric_data_error_message():
    mx.random.seed(793)
    error_args = MLXTrainingConfig(per_device_train_batch_size=1, gradient_accumulation_steps=1, max_steps=1, logging_steps=1, eval_steps=0, save_steps=0, learning_rate=1e-3, max_seq_length=8, output_dir=str(Path(sys.argv[1], "asymmetric_data_error")), use_cce=False, compile=False, gradient_checkpointing=False, cast_norm_output_to_input_dtype=False, dataset_order="sequential", streaming=True)
    error_trainer = MLXTrainer(TinyLM(), TinyTokenizer(), RankFailingStream(), args=error_args)
    error_trainer.save_model = mark("asymmetric_data_error_final")
    try:
        error_trainer.train()
    except RuntimeError as exc:
        return str(exc)
    return ""
asymmetric_data_error = asymmetric_data_error_message()
def zero_token_error_message():
    mx.random.seed(794)
    zero_args = MLXTrainingConfig(per_device_train_batch_size=1, gradient_accumulation_steps=1, max_steps=1, logging_steps=1, eval_steps=0, save_steps=0, learning_rate=1e-3, max_seq_length=8, output_dir=str(Path(sys.argv[1], "zero_token_error")), use_cce=False, compile=False, gradient_checkpointing=False, cast_norm_output_to_input_dtype=False, dataset_order="sequential")
    zero_trainer = MLXTrainer(TinyLM(), TinyTokenizer(), train_data, args=zero_args)
    zero_trainer._batches = [(mx.array([[0, 0]], dtype=mx.int32), mx.array([[0, 0]], dtype=mx.int32), None)]
    zero_trainer.save_model = mark("zero_token_error_final")
    try:
        zero_trainer.train()
    except (RuntimeError, ValueError) as exc:
        return str(exc)
    return ""
zero_token_error = zero_token_error_message()
fresh_trainer = make_parity_trainer("resume_fresh")
fresh_trainer.train()
partial_trainer = make_parity_trainer("resume_partial")
partial_trainer.add_step_callback(lambda step, *_args: setattr(partial_trainer, "stop_requested", step >= 2))
partial_trainer.train()
sync = trainer._distributed_all_sum(mx.array(1, dtype=mx.int32), stream=mx.cpu)
mx.eval(sync)
resumed_trainer = make_parity_trainer("resume_resumed", seed=987)
resumed_trainer.train(resume_from_checkpoint=str(Path(sys.argv[1], "resume_partial", "checkpoint-2")))
resume_param_delta = max_trainable_delta(fresh_trainer, resumed_trainer)
trainer_mod.save_trainable_adapters = mark("vlm_ckpt_adapter")
trainer_mod.save_optimizer_state = mark("vlm_ckpt_opt")
trainer_mod.save_trainer_state = mark("vlm_ckpt_state")
mx.random.seed(1)
vlm_args = MLXTrainingConfig(per_device_train_batch_size=1, gradient_accumulation_steps=1, max_steps=1, logging_steps=1, eval_steps=1, save_steps=1, learning_rate=1e-3, max_seq_length=8, output_dir=str(Path(sys.argv[1], "vlm_train_out")), use_cce=False, gradient_checkpointing=False, cast_norm_output_to_input_dtype=False, dataset_order="sequential", streaming=True)
vlm_processor = TinyProcessor()
vlm_train_data = [{"text": str(i)} for i in (10, 13, 11, 12, 14, 17, 15, 16)]
vlm_trainer = MLXTrainer(TinyVLM(), vlm_processor, vlm_train_data, eval_dataset=vlm_train_data[:4], args=vlm_args, processor=vlm_processor)
vlm_events = []
vlm_trainer.save_model = mark("vlm_final")
vlm_trainer.add_step_callback(lambda step, *_args: vlm_events.append(["step", int(step)]))
vlm_trainer.add_eval_callback(lambda step, loss, _ppl: vlm_events.append(["eval", int(step), float(loss)]))
vlm_result = vlm_trainer.train()
vlm_param_sum = mx.sum(vlm_trainer.model.embed.weight) + mx.sum(vlm_trainer.model.proj.weight)
def make_vlm_compile_parity_trainer(output_name, compile_enabled):
    mx.random.seed(2)
    parity_args = MLXTrainingConfig(per_device_train_batch_size=1, gradient_accumulation_steps=2, max_steps=2, logging_steps=1, eval_steps=0, save_steps=0, learning_rate=1e-3, max_grad_norm=1.0, max_seq_length=8, output_dir=str(Path(sys.argv[1], output_name)), use_cce=False, compile=compile_enabled, gradient_checkpointing=False, cast_norm_output_to_input_dtype=False, dataset_order="sequential")
    out = MLXTrainer(TinyVLM(), vlm_processor, vlm_train_data, args=parity_args, processor=vlm_processor)
    out.save_model = mark(f"{output_name}_final")
    return out
vlm_eager_parity_trainer = make_vlm_compile_parity_trainer("vlm_parity_eager", False)
vlm_eager_parity_result = vlm_eager_parity_trainer.train()
sync = trainer._distributed_all_sum(mx.array(1, dtype=mx.int32), stream=mx.cpu)
mx.eval(sync)
vlm_compile_parity_trainer = make_vlm_compile_parity_trainer("vlm_parity_compile", True)
vlm_compile_parity_result = vlm_compile_parity_trainer.train()
vlm_compile_param_delta = max_trainable_delta(vlm_eager_parity_trainer, vlm_compile_parity_trainer)
ckpt = Path(sys.argv[1], "resume_ckpt"); ckpt.mkdir(exist_ok=True)
for name in ("adapters.safetensors", "optimizer_state.safetensors", "trainer_state.json"):
    (ckpt / name).write_text("x")
try:
    trainer._validate_distributed_resume_checkpoint(ckpt if world.rank() == 0 else None)
    resume_mismatch = ""
except RuntimeError as exc:
    resume_mismatch = str(exc)
def vlm_stream_rejection():
    try:
        next(iter(iterate_vlm_training_batches(ReplayableVLMStream(), TinyProcessor(), {"image_token_id": 20}, batch_size=2, max_seq_length=8, dataset_order="sequential", comm_group=world)))
        return "no-error"
    except ValueError as exc:
        return "rejected" if "DDP training" in str(exc) else f"wrong: {exc}"

payload = {
    "rank": int(world.rank()),
    "size": int(world.size()),
    "synced_stop": bool(synced_stop),
    "loop_events": events,
    "loop_step": int(loop_trainer._global_step),
    "loop_compile_enabled": bool(result["compile_enabled"]),
    "loop_compile_scope": result["compile_scope"],
    "loop_compile_fallback": bool(result["compile_fallback"]),
    "loop_compile_fallback_reason": result["compile_fallback_reason"],
    "loop_text_eval_rows": list(loop_trainer.model.eval_first_tokens),
    "loop_eval_metrics": result["eval_metrics"],
    "loop_eval_loss": float(loop_trainer._last_eval_metrics["eval_loss"]),
    "loop_eval_reference_loss": loop_eval_reference_loss,
    "loop_host_rank_map": result["distributed_host_rank_map"],
    "param_sum": float(param_sum.item()),
    "loop_main": bool(result["distributed_is_main_process"]),
    "response_rows": response_rows,
    "stream_events": stream_events,
    "stream_step": int(stream_trainer._global_step),
    "stream_compile_enabled": bool(stream_result["compile_enabled"]),
    "stream_compile_scope": stream_result["compile_scope"],
    "stream_text_eval_rows": list(stream_trainer.model.eval_first_tokens),
    "stream_eval_loss": float(stream_trainer._last_eval_metrics["eval_loss"]),
    "stream_eval_reference_loss": stream_eval_reference_loss,
    "stream_param_sum": float(stream_param_sum.item()),
    "ddp_eager_losses": [float(x) for x in ddp_eager_trainer._train_loss_history],
    "ddp_compile_losses": [float(x) for x in ddp_compile_trainer._train_loss_history],
    "ddp_compile_param_delta": ddp_compile_param_delta,
    "ddp_compile_enabled": bool(ddp_compile_result["compile_enabled"]),
    "ddp_compile_scope": ddp_compile_result["compile_scope"],
    "ddp_eager_tokens": int(ddp_eager_result["trained_tokens"]),
    "ddp_compile_tokens": int(ddp_compile_result["trained_tokens"]),
    "strict_error": strict_error,
    "best_effort_compile_enabled": bool(best_effort_fallback_result["compile_enabled"]),
    "best_effort_compile_scope": best_effort_fallback_result["compile_scope"],
    "best_effort_compile_fallback": bool(best_effort_fallback_result["compile_fallback"]),
    "best_effort_compile_fallback_reason": best_effort_fallback_result["compile_fallback_reason"],
    "non_compile_error": non_compile_error,
    "asymmetric_data_error": asymmetric_data_error,
    "zero_token_error": zero_token_error,
    "fresh_losses": [float(x) for x in fresh_trainer._train_loss_history],
    "resumed_losses": [float(x) for x in resumed_trainer._train_loss_history],
    "resume_param_delta": resume_param_delta,
    "vlm_events": vlm_events,
    "vlm_step": int(vlm_trainer._global_step),
    "vlm_compile_enabled": bool(vlm_result["compile_enabled"]),
    "vlm_compile_scope": vlm_result["compile_scope"],
    "vlm_param_sum": float(vlm_param_sum.item()),
    "vlm_eager_parity_losses": [float(x) for x in vlm_eager_parity_trainer._train_loss_history],
    "vlm_compile_parity_losses": [float(x) for x in vlm_compile_parity_trainer._train_loss_history],
    "vlm_compile_param_delta": vlm_compile_param_delta,
    "vlm_compile_parity_enabled": bool(vlm_compile_parity_result["compile_enabled"]),
    "vlm_compile_parity_scope": vlm_compile_parity_result["compile_scope"],
    "vlm_eager_parity_tokens": int(vlm_eager_parity_result["trained_tokens"]),
    "vlm_compile_parity_tokens": int(vlm_compile_parity_result["trained_tokens"]),
    "resume_ok": Path(trainer._validate_distributed_resume_checkpoint(ckpt)).name,
    "resume_mismatch": resume_mismatch,
    "default": first_token_rows(create_batches(data, TinyTokenizer(), seed=1, **common)),
    "ordered": first_token_rows(create_ordered_batches(data, TinyTokenizer(), dataset_order="sequential", **common)),
    "labeled": first_token_rows(_create_labeled_batches(data, TinyTokenizer(), keep_all_labels, dataset_order="sequential", **common)),
    "stream_text": take_stream_rows(iterate_training_batches(ReplayableStream(), TinyTokenizer(), batch_size=2, max_seq_length=8, dataset_order="sequential", comm_group=world), 2),
    "stream_replay": stream_replay_probe(),
    "presharded_stream_error": presharded_stream_error(),
    "owner_metadata_error": owner_metadata_error(),
    "prepared_override_batches": prepared_override_batches(),
    "epoch_owned": epoch_owned_probe(),
    "stream_labeled_empty": [{"ids": batch.tolist(), "lengths": lengths.tolist(), "labels": labels.tolist()} for batch, lengths, labels in list(iterate_training_batches(ReplayableLabeledStream(), TinyTokenizer(), batch_size=1, max_seq_length=8, dataset_order="sequential", comm_group=world, repeat=False, distributed_pad_mode="empty"))],
    "vlm": [[int(row[0]) for row in batch["input_ids"].tolist()] for batch in create_vlm_batches([{"text": str(i)} for i in range(10, 15)], TinyProcessor(), {"image_token_id": 20}, batch_size=2, max_seq_length=8, dataset_order="sequential", comm_group=world)],
    "vlm_empty_eval": [{"ids": batch["input_ids"].tolist(), "mask": batch["attention_mask"].tolist(), "labels": batch["labels"].tolist()} for batch in create_vlm_batches([{"text": str(i)} for i in range(10, 13)], TinyProcessor(), {"image_token_id": 20}, batch_size=1, max_seq_length=8, dataset_order="sequential", comm_group=world, distributed_pad_mode="empty")],
    "stream_vlm_rejection": vlm_stream_rejection(),
    "stream_window": take_stream_rows(iterate_training_batches(ReplayableStream(), TinyTokenizer(), batch_size=1, max_seq_length=8, dataset_order="default", comm_group=world, repeat=False, length_window_batches=2), 3),
}
Path(sys.argv[1], f"rank{world.rank()}.json").write_text(json.dumps(payload))
""",
        encoding="utf-8")

    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    # The ring launcher bash-execs the command array verbatim, so pin the
    # interpreter as argv[0] instead of relying on a shebang + chmod.
    cmd = [
        str(launcher), "-n", "2", "--backend", "ring",
        "--cwd", str(repo_root),
        sys.executable, str(script), str(tmp_path),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, env=env, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    outputs = [tmp_path / f"rank{rank}.json" for rank in (0, 1)]
    assert all(path.exists() for path in outputs), proc.stdout + proc.stderr
    ranks = [json.loads(path.read_text()) for path in outputs]
    expected = [[[10, 12], [14, 11]], [[11, 13], [10, 12]]]
    assert [rank["size"] for rank in ranks] == [2, 2]
    assert [rank["synced_stop"] for rank in ranks] == [True, True]
    assert [rank["loop_main"] for rank in ranks] == [True, False]
    assert [rank["loop_events"] for rank in ranks] == [[["step", 12], ["eval"]], []]
    assert [rank["loop_step"] for rank in ranks] == [1, 1]
    assert [rank["loop_compile_enabled"] for rank in ranks] == [True, True]
    assert [rank["loop_compile_scope"] for rank in ranks] == ["ddp_local_grad", "ddp_local_grad"]
    assert [rank["loop_compile_fallback"] for rank in ranks] == [False, False]
    assert [rank["loop_compile_fallback_reason"] for rank in ranks] == ["", ""]
    assert [rank["loop_text_eval_rows"] for rank in ranks] == [[10, 12], [11, 2]]
    assert ranks[0]["loop_eval_metrics"]["eval_loss"] == pytest.approx(ranks[0]["loop_eval_loss"], abs=1e-6)
    assert ranks[1]["loop_eval_metrics"]["eval_loss"] == pytest.approx(ranks[1]["loop_eval_loss"], abs=1e-6)
    assert ranks[0]["loop_eval_loss"] == pytest.approx(ranks[1]["loop_eval_loss"], abs=1e-6)
    assert ranks[0]["loop_eval_loss"] == pytest.approx(ranks[0]["loop_eval_reference_loss"], abs=1e-6)
    assert ranks[1]["loop_eval_loss"] == pytest.approx(ranks[1]["loop_eval_reference_loss"], abs=1e-6)
    assert ranks[0]["loop_host_rank_map"] == ranks[1]["loop_host_rank_map"]
    assert [entry["rank"] for entry in ranks[0]["loop_host_rank_map"]] == [0, 1]
    assert all(entry["hostname"] for entry in ranks[0]["loop_host_rank_map"])
    assert abs(ranks[0]["param_sum"] - ranks[1]["param_sum"]) < 1e-5
    assert [rank["response_rows"] for rank in ranks] == [[[10], [12], [14], [16]], [[11], [13], [15], [17]]]
    assert [rank["stream_step"] for rank in ranks] == [1, 1]
    assert [rank["stream_compile_enabled"] for rank in ranks] == [True, True]
    assert [rank["stream_compile_scope"] for rank in ranks] == ["ddp_local_grad", "ddp_local_grad"]
    assert [rank["stream_text_eval_rows"] for rank in ranks] == [[10, 12, 14], [11, 13, 2]]
    assert ranks[0]["stream_eval_loss"] == pytest.approx(ranks[1]["stream_eval_loss"], abs=1e-6)
    assert ranks[0]["stream_eval_loss"] == pytest.approx(ranks[0]["stream_eval_reference_loss"], abs=1e-6)
    assert ranks[1]["stream_eval_loss"] == pytest.approx(ranks[1]["stream_eval_reference_loss"], abs=1e-6)
    assert ranks[0]["stream_events"][0] == ["step", 1]
    assert ranks[0]["stream_events"][1][0:2] == ["eval", 1]
    assert ranks[0]["stream_events"][1][2] > 0
    assert ranks[1]["stream_events"] == []
    assert abs(ranks[0]["stream_param_sum"] - ranks[1]["stream_param_sum"]) < 1e-5
    assert ranks[0]["ddp_compile_losses"] == pytest.approx(ranks[0]["ddp_eager_losses"], abs=1e-6)
    assert ranks[1]["ddp_compile_losses"] == pytest.approx(ranks[1]["ddp_eager_losses"], abs=1e-6)
    assert ranks[0]["ddp_compile_param_delta"] < 1e-5
    assert ranks[1]["ddp_compile_param_delta"] < 1e-5
    assert [rank["ddp_compile_enabled"] for rank in ranks] == [True, True]
    assert [rank["ddp_compile_scope"] for rank in ranks] == ["ddp_local_grad", "ddp_local_grad"]
    assert [rank["ddp_eager_tokens"] for rank in ranks] == [rank["ddp_compile_tokens"] for rank in ranks]
    assert all("runtime fallback is disabled" in rank["strict_error"] for rank in ranks)
    assert [rank["best_effort_compile_enabled"] for rank in ranks] == [False, False]
    assert [rank["best_effort_compile_scope"] for rank in ranks] == ["fallback_eager", "fallback_eager"]
    assert [rank["best_effort_compile_fallback"] for rank in ranks] == [True, True]
    assert [rank["best_effort_compile_fallback_reason"] for rank in ranks] == ["runtime_error", "runtime_error"]
    assert all("eval side failure from model" in rank["non_compile_error"] for rank in ranks)
    assert all("runtime fallback is disabled" not in rank["non_compile_error"] for rank in ranks)
    assert all("fetching training batch" in rank["asymmetric_data_error"] for rank in ranks)
    assert "rank 0 failed" in ranks[0]["asymmetric_data_error"]
    assert "rank 0 failed while reading" in ranks[1]["asymmetric_data_error"]
    assert all("zero supervised tokens" in rank["zero_token_error"] for rank in ranks)
    assert ranks[0]["fresh_losses"] == pytest.approx(ranks[0]["resumed_losses"], abs=1e-6)
    assert ranks[0]["resume_param_delta"] < 1e-5
    assert ranks[1]["resume_param_delta"] < 1e-5
    assert [rank["vlm_step"] for rank in ranks] == [1, 1]
    assert [rank["vlm_compile_enabled"] for rank in ranks] == [True, True]
    assert [rank["vlm_compile_scope"] for rank in ranks] == ["ddp_local_grad", "ddp_local_grad"]
    assert ranks[0]["vlm_compile_parity_losses"] == pytest.approx(ranks[0]["vlm_eager_parity_losses"], abs=1e-6)
    assert ranks[1]["vlm_compile_parity_losses"] == pytest.approx(ranks[1]["vlm_eager_parity_losses"], abs=1e-6)
    assert ranks[0]["vlm_compile_param_delta"] < 1e-5
    assert ranks[1]["vlm_compile_param_delta"] < 1e-5
    assert [rank["vlm_compile_parity_enabled"] for rank in ranks] == [True, True]
    assert [rank["vlm_compile_parity_scope"] for rank in ranks] == ["ddp_local_grad", "ddp_local_grad"]
    assert [rank["vlm_eager_parity_tokens"] for rank in ranks] == [rank["vlm_compile_parity_tokens"] for rank in ranks]
    assert ranks[0]["vlm_events"][0] == ["step", 1]
    assert ranks[0]["vlm_events"][1][0:2] == ["eval", 1]
    assert ranks[0]["vlm_events"][1][2] > 0
    assert ranks[1]["vlm_events"] == []
    assert abs(ranks[0]["vlm_param_sum"] - ranks[1]["vlm_param_sum"]) < 1e-5
    assert [rank["resume_ok"] for rank in ranks] == ["resume_ckpt", "resume_ckpt"]
    assert all("all ranks must either resume" in rank["resume_mismatch"] for rank in ranks)
    for prefix in ("ckpt_adapter", "ckpt_opt", "ckpt_state", "final"):
        assert (tmp_path / f"{prefix}_rank0").exists()
        assert not (tmp_path / f"{prefix}_rank1").exists()
    for prefix in ("vlm_ckpt_adapter", "vlm_ckpt_opt", "vlm_ckpt_state", "vlm_final"):
        assert (tmp_path / f"{prefix}_rank0").exists()
        assert not (tmp_path / f"{prefix}_rank1").exists()
    assert (tmp_path / "stream_final_rank0").exists()
    assert not (tmp_path / "stream_final_rank1").exists()
    assert [rank["default"] for rank in ranks] == expected
    assert [rank["ordered"] for rank in ranks] == expected
    assert [rank["labeled"] for rank in ranks] == expected
    assert [rank["vlm"] for rank in ranks] == expected
    assert ranks[0]["vlm_empty_eval"][1]["ids"][0][0] == 12
    assert ranks[1]["vlm_empty_eval"][1]["ids"][0][0] == 12
    assert ranks[1]["vlm_empty_eval"][1]["ids"][0][1] == 20
    assert ranks[1]["vlm_empty_eval"][1]["mask"][0] == [1, 1, 1, 1]
    assert all(value == -100 for value in ranks[1]["vlm_empty_eval"][1]["labels"][0])
    assert [rank["stream_text"] for rank in ranks] == expected
    assert all(rank["stream_replay"][0] == rank["stream_replay"][1] for rank in ranks)
    assert "global unsharded" in ranks[0]["presharded_stream_error"]
    assert "rank 0 failed" in ranks[1]["presharded_stream_error"]
    assert "owner metadata failure" in ranks[0]["owner_metadata_error"]
    assert "rank 0 failed" in ranks[1]["owner_metadata_error"]
    assert ranks[0]["prepared_override_batches"][1][0] == [12, 32, 9, 9]
    assert ranks[1]["prepared_override_batches"][1][0] == [10, 30, 9, 9]
    assert [rank["epoch_owned"] for rank in ranks] == [[2, [10]], [2, [11]]]
    assert [rank["stream_labeled_empty"][1]["ids"][0][0] for rank in ranks] == [12, 2]
    assert ranks[0]["stream_labeled_empty"][1]["labels"][0] == [-100, 32]
    assert ranks[1]["stream_labeled_empty"][1]["lengths"][0] == [0, 0]
    assert ranks[1]["stream_labeled_empty"][1]["labels"][0] == [-100, -100]
    # Intentional compatibility break: every-rank lazy-VLM consumption is
    # replaced by a synchronized pre-consumption rejection on all ranks.
    assert [rank["stream_vlm_rejection"] for rank in ranks] == ["rejected", "rejected"]
    # W=2 owner-side window: rows 10..14 (equal length) pool into chunks
    # [10,11],[12,13]; seed-42 permutation emits [12,13] first, so the pass-tail
    # partial [14] cycle-pads from that first dispatched batch.
    assert [rank["stream_window"] for rank in ranks] == [
        [[12], [10], [14]], [[13], [11], [12]],
    ]

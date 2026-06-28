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
from unsloth_zoo.mlx.utils import create_batches, create_ordered_batches

class TinyTokenizer:
    pad_token_id = eos_token_id = 2
    def encode(self, text): return [int(part) for part in str(text).split()]
    def __call__(self, text, **_kwargs): return {"input_ids": self.encode(text)}

def keep_all_labels(d): return {"labels": [list(d["input_ids"][0])]}
def first_token_rows(batches):
    return [[int(row[0]) for row in batch.tolist()] for batch, _, _ in batches]

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

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__(); self.embed = nn.Embedding(64, 8)
        self.proj = nn.Linear(8, 64, bias=False); self._config = {"model_type": "tiny"}
    def __call__(self, x): return self.proj(self.embed(x))

def mark(name):
    def inner(*_args, **_kwargs):
        Path(sys.argv[1], f"{name}_rank{world.rank()}").write_text("1")
    return inner

real_save_trainable_adapters = trainer_mod.save_trainable_adapters
real_save_optimizer_state = trainer_mod.save_optimizer_state
real_save_trainer_state = trainer_mod.save_trainer_state
trainer_mod.save_trainable_adapters = mark("ckpt_adapter")
trainer_mod.save_optimizer_state = mark("ckpt_opt")
trainer_mod.save_trainer_state = mark("ckpt_state")
mx.random.seed(0)
train_data = [{"text": f"{i} {i + 1} {i + 2}"} for i in range(10, 18)]
args = MLXTrainingConfig(
    per_device_train_batch_size=1, gradient_accumulation_steps=2, max_steps=3,
    logging_steps=1, eval_steps=1, save_steps=1, learning_rate=1e-3,
    max_seq_length=8, output_dir=str(Path(sys.argv[1], "train_out")),
    use_cce=False, gradient_checkpointing=False,
    cast_norm_output_to_input_dtype=False, dataset_order="sequential",
)
loop_trainer = MLXTrainer(TinyLM(), TinyTokenizer(), train_data, eval_dataset=train_data[:4], args=args)
events = []
loop_trainer.save_model = mark("final")
loop_trainer.add_step_callback(lambda *args: events.append(["step", int(args[7])]))
loop_trainer.add_eval_callback(
    lambda *_args: (events.append(["eval"]), setattr(loop_trainer, "stop_requested", True))
)
dataset_utils.train_on_responses_only = lambda *_args, **_kwargs: keep_all_labels
train_on_responses_only(loop_trainer, instruction_part="user", response_part="assistant")
response_rows = first_token_rows(loop_trainer._batches)
result = loop_trainer.train()
param_sum = mx.sum(loop_trainer.model.embed.weight) + mx.sum(loop_trainer.model.proj.weight)
trainer_mod.save_trainable_adapters = real_save_trainable_adapters
trainer_mod.save_optimizer_state = real_save_optimizer_state
trainer_mod.save_trainer_state = real_save_trainer_state
parity_data = [{"text": f"{i} {i + 1} {i + 2}"} for i in range(10, 18)]
def make_parity_trainer(output_name, seed=123):
    mx.random.seed(seed)
    parity_args = MLXTrainingConfig(
        per_device_train_batch_size=1, gradient_accumulation_steps=2, max_steps=3,
        logging_steps=1, eval_steps=0, save_steps=2, learning_rate=1e-3,
        max_seq_length=8, output_dir=str(Path(sys.argv[1], output_name)),
        use_cce=False, gradient_checkpointing=False,
        cast_norm_output_to_input_dtype=False, dataset_order="sequential",
    )
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
fresh_trainer = make_parity_trainer("resume_fresh")
fresh_trainer.train()
partial_trainer = make_parity_trainer("resume_partial")
partial_trainer.add_step_callback(
    lambda step, *_args: setattr(partial_trainer, "stop_requested", step >= 2)
)
partial_trainer.train()
sync = trainer._distributed_all_sum(mx.array(1, dtype=mx.int32), stream=mx.cpu)
mx.eval(sync)
resumed_trainer = make_parity_trainer("resume_resumed", seed=987)
resumed_trainer.train(
    resume_from_checkpoint=str(Path(sys.argv[1], "resume_partial", "checkpoint-2"))
)
resume_param_delta = max_trainable_delta(fresh_trainer, resumed_trainer)
ckpt = Path(sys.argv[1], "resume_ckpt"); ckpt.mkdir(exist_ok=True)
for name in ("adapters.safetensors", "optimizer_state.safetensors", "trainer_state.json"):
    (ckpt / name).write_text("x")
try:
    trainer._validate_distributed_resume_checkpoint(ckpt if world.rank() == 0 else None)
    resume_mismatch = ""
except RuntimeError as exc:
    resume_mismatch = str(exc)
payload = {
    "rank": int(world.rank()),
    "size": int(world.size()),
    "synced_stop": bool(synced_stop),
    "loop_events": events,
    "loop_step": int(loop_trainer._global_step),
    "param_sum": float(param_sum.item()),
    "loop_main": bool(result["distributed_is_main_process"]),
    "response_rows": response_rows,
    "fresh_losses": [float(x) for x in fresh_trainer._train_loss_history],
    "resumed_losses": [float(x) for x in resumed_trainer._train_loss_history],
    "resume_param_delta": resume_param_delta,
    "resume_ok": Path(trainer._validate_distributed_resume_checkpoint(ckpt)).name,
    "resume_mismatch": resume_mismatch,
    "default": first_token_rows(create_batches(data, TinyTokenizer(), seed=1, **common)),
    "ordered": first_token_rows(
        create_ordered_batches(data, TinyTokenizer(), dataset_order="sequential", **common)
    ),
    "labeled": first_token_rows(
        _create_labeled_batches(
            data, TinyTokenizer(), keep_all_labels,
            dataset_order="sequential", **common,
        )
    ),
}
Path(sys.argv[1], f"rank{world.rank()}.json").write_text(json.dumps(payload))
""",
        encoding="utf-8")

    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        str(launcher), "-n", "2", "--backend", "ring",
        "--python", sys.executable, "--cwd", str(repo_root),
        str(script), str(tmp_path),
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
    assert abs(ranks[0]["param_sum"] - ranks[1]["param_sum"]) < 1e-5
    assert [rank["response_rows"] for rank in ranks] == [[[10], [12], [14], [16]], [[11], [13], [15], [17]]]
    assert ranks[0]["fresh_losses"] == pytest.approx(ranks[0]["resumed_losses"], abs=1e-6)
    assert ranks[0]["resume_param_delta"] < 1e-5
    assert ranks[1]["resume_param_delta"] < 1e-5
    assert [rank["resume_ok"] for rank in ranks] == ["resume_ckpt", "resume_ckpt"]
    assert all("all ranks must either resume" in rank["resume_mismatch"] for rank in ranks)
    for prefix in ("ckpt_adapter", "ckpt_opt", "ckpt_state", "final"):
        assert (tmp_path / f"{prefix}_rank0").exists()
        assert not (tmp_path / f"{prefix}_rank1").exists()
    assert [rank["default"] for rank in ranks] == expected
    assert [rank["ordered"] for rank in ranks] == expected
    assert [rank["labeled"] for rank in ranks] == expected

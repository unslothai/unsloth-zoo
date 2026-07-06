"""Real MLX LoRA training smoke on Apple Silicon for the PR 684 trainer rework.

Downloads a tiny 4-bit model (~80MB), runs short FastMLXModel + MLXTrainer
LoRA fits, and checks losses, gradients-driven progress, and adapter saving.
Exercises the reworked grad-clip resolution (default leaf-norm path and the
explicit elementwise path), decoupled weight decay, batching, and both loss
functions (CCE and baseline).
"""

import glob
import os

import pytest

try:
    import mlx.core as mx
    _METAL = mx.metal.is_available()
except Exception:
    _METAL = False

if not _METAL:
    print("NOTICE: Metal unavailable; all MLX e2e training tests will be skipped.")

metal_only = pytest.mark.skipif(not _METAL, reason="requires Apple Silicon Metal")

MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"


def _dataset(n=24):
    return [
        {"text": f"### Question: what is {i} plus {i}?\n### Answer: {2 * i}."}
        for i in range(n)
    ]


def _train(tmp_path, **config_overrides):
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    model, tokenizer = FastMLXModel.from_pretrained(MODEL, max_seq_length=256)
    model = FastMLXModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0)
    config = dict(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=8,
        warmup_steps=2,
        learning_rate=5e-4,
        logging_steps=1,
        output_dir=str(tmp_path),
        seed=3407,
        report_to="none",
    )
    config.update(config_overrides)
    trainer = MLXTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=_dataset(),
        args=MLXTrainingConfig(**config),
    )
    trainer.train()
    return trainer


def _assert_history(trainer, min_steps):
    hist = trainer._train_loss_history
    assert len(hist) >= min_steps, f"only {len(hist)} logged losses"
    assert all(
        isinstance(l, float) and l == l and abs(l) != float("inf") for l in hist
    ), f"non-finite losses: {hist}"
    return hist


def _callback_batch():
    """Build a tiny labeled MLX batch for callback lifecycle tests."""
    tokens = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
    return tokens, mx.array([[0, 4]], dtype=mx.int32), tokens


def _callback_trainer(
    tmp_path,
    callbacks,
    max_steps=3,
    eval_steps=1,
    logging_steps=1,
    with_eval=False,
):
    """Create a minimal MLXTrainer with prebuilt batches for callback tests."""
    import mlx.nn as nn
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(8, 4)
            self.lm_head = nn.Linear(4, 8, bias=False)

        def __call__(self, input_ids):
            return self.lm_head(self.embed(input_ids))

    trainer = MLXTrainer(
        model=TinyLM(),
        tokenizer=None,
        train_dataset=[],
        eval_dataset=[{}] if (eval_steps or with_eval) else None,
        args=MLXTrainingConfig(
            max_steps=max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            output_dir=str(tmp_path),
            use_cce=False,
            compile=False,
            gradient_checkpointing=False,
            report_to="none",
        ),
        callbacks=callbacks,
    )
    trainer._batches = [_callback_batch()]
    if eval_steps or with_eval:
        trainer._eval_batches_labeled = [_callback_batch()]
    trainer._saved = []
    trainer.save_model = lambda output_dir=None: trainer._saved.append(
        output_dir or trainer.args.output_dir
    )
    return trainer


@metal_only
def test_hf_callbacks_receive_mlx_trainer_lifecycle(tmp_path):
    from transformers import TrainerCallback

    class Recorder(TrainerCallback):
        def __init__(self):
            self.events, self.eval_metrics = [], None

        def on_init_end(self, args, state, control, **_kwargs):
            self.events.append(("init", state.global_step, args.eval_strategy))

        def on_train_begin(self, args, state, control, **kwargs):
            self.events.append((
                "train_begin",
                state.global_step,
                kwargs["train_dataloader"] is not None,
            ))

        def on_step_begin(self, args, state, control, **_kwargs):
            self.events.append(("step_begin", state.global_step))

        def on_optimizer_step(self, args, state, control, **_kwargs):
            self.events.append(("optimizer", state.global_step))

        def on_step_end(self, args, state, control, **_kwargs):
            self.events.append(("step_end", state.global_step))

        def on_log(self, args, state, control, logs, **_kwargs):
            self.events.append(("log", state.global_step, "loss" in logs))

        def on_save(self, args, state, control, **_kwargs):
            self.events.append(("save", state.global_step))

        def on_train_end(self, args, state, control, **_kwargs):
            self.events.append(("train_end", state.global_step))

        def on_epoch_begin(self, args, state, control, **_kwargs):
            self.events.append(("epoch_begin", state.epoch))

        def on_epoch_end(self, args, state, control, **_kwargs):
            self.events.append(("epoch_end", state.epoch))

        def on_evaluate(self, args, state, control, metrics, **_kwargs):
            self.eval_metrics = dict(metrics)
            self.events.append(("eval", state.global_step))

    class ClassCallback(TrainerCallback):
        calls = []

        def on_train_begin(self, args, state, control, **_kwargs):
            type(self).calls.append(state.global_step)

    recorder = Recorder()
    ClassCallback.calls = []
    trainer = _callback_trainer(tmp_path, [recorder, ClassCallback])
    output = trainer.train()
    names = {event[0] for event in recorder.events}
    assert {
        "init", "train_begin", "optimizer", "step_end", "log", "eval",
        "save", "train_end", "epoch_begin", "epoch_end",
    } <= names
    assert recorder.events[0] == ("init", 0, "steps")
    assert ("train_begin", 0, True) in recorder.events
    assert recorder.eval_metrics["eval_loss"] >= 0
    assert ClassCallback.calls == [0]
    assert trainer._saved == [str(tmp_path)]
    assert output.global_step == 3


@metal_only
def test_hf_callback_control_can_stop_mlx_training(tmp_path):
    from transformers import TrainerCallback

    class StopAfterFirstStep(TrainerCallback):
        def __init__(self):
            self.events = []

        def on_step_end(self, args, state, control, **_kwargs):
            self.events.append(("step_end", state.global_step))
            control.should_training_stop = state.global_step == 1
            return control

        def on_epoch_end(self, args, state, control, **_kwargs):
            self.events.append(("epoch_end", state.global_step, state.epoch))

    callback = StopAfterFirstStep()
    output = _callback_trainer(tmp_path, [callback], max_steps=5, eval_steps=0).train()
    assert output.global_step == 1
    assert ("step_end", 1) in callback.events


@metal_only
def test_hf_callback_control_can_force_log_and_eval(tmp_path):
    from transformers import TrainerCallback

    class RequestLogAndEval(TrainerCallback):
        def __init__(self):
            self.logs, self.evals = [], []

        def on_step_end(self, args, state, control, **_kwargs):
            if state.global_step == 1:
                control.should_log = True
                control.should_evaluate = True
            return control

        def on_log(self, args, state, control, logs, **_kwargs):
            self.logs.append((state.global_step, dict(logs)))

        def on_evaluate(self, args, state, control, metrics, **_kwargs):
            self.evals.append((state.global_step, dict(metrics)))

    callback = RequestLogAndEval()
    _callback_trainer(
        tmp_path,
        [callback],
        max_steps=2,
        eval_steps=0,
        logging_steps=0,
        with_eval=True,
    ).train()

    assert any(step == 1 and "loss" in logs for step, logs in callback.logs)
    assert callback.evals and callback.evals[0][0] == 1
    assert "eval_loss" in callback.evals[0][1]


@metal_only
def test_hf_eval_callbacks_see_prior_best_metric(tmp_path):
    from transformers import TrainerCallback

    class BestMetricRecorder(TrainerCallback):
        def __init__(self):
            self.best_before_eval = []

        def on_evaluate(self, args, state, control, metrics, **_kwargs):
            self.best_before_eval.append(state.best_metric)

    callback = BestMetricRecorder()
    trainer = _callback_trainer(tmp_path, [callback], max_steps=2, eval_steps=1)
    eval_losses = iter((2.0, 3.0))

    def fake_evaluate(_eval_batches, _loss_fn, is_vlm=False):
        loss = next(eval_losses)
        trainer._last_eval_metrics = {
            "eval_loss": loss,
            "eval_perplexity": 1.0,
        }
        return loss, 1.0

    trainer._evaluate = fake_evaluate
    trainer.train()

    assert callback.best_before_eval == [None, 2.0]
    assert trainer.state.best_metric == 2.0
    assert trainer.state.best_global_step == 1


@metal_only
def test_mlx_trainer_import_keeps_torch_unloaded():
    import subprocess
    import sys
    from pathlib import Path

    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    code = (
        "import sys; "
        "import unsloth_zoo.mlx.trainer; "
        "raise SystemExit(1 if 'torch' in sys.modules else 0)"
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


@metal_only
def test_lora_sft_cce_default_clip(tmp_path):
    """Default config: CCE loss, leaf-norm clip default, decoupled decay."""
    trainer = _train(tmp_path, use_cce=True)
    hist = _assert_history(trainer, min_steps=8)
    assert hist[-1] < hist[0], f"loss did not improve: {hist}"
    saved = glob.glob(os.path.join(str(tmp_path), "**", "*.safetensors"), recursive=True)
    assert saved, "no adapter safetensors saved at end of training"


@metal_only
def test_lora_sft_baseline_loss_value_clip(tmp_path):
    """Baseline (non-CCE) loss with explicit elementwise grad clip."""
    trainer = _train(
        tmp_path,
        use_cce=False,
        max_grad_value=0.5,
        max_steps=4,
    )
    _assert_history(trainer, min_steps=4)

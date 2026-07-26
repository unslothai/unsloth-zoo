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

if _METAL:
    # Module scope: leaked mlx-simulation shims must not hijack test-time imports.
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_map
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    from unsloth_zoo.mlx.utils import (
        FiniteTextBatchPlan, _FiniteTextRow, make_baseline_loss_fn,
    )

MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"


def _dataset(n=24):
    return [
        {"text": f"### Question: what is {i} plus {i}?\n### Answer: {2 * i}."}
        for i in range(n)
    ]


def _train(tmp_path, **config_overrides):
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
    save_steps=0,
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
            save_steps=save_steps,
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
            config = args.to_dict()
            assert config["output_dir"] == str(tmp_path)
            assert args.logging_dir == os.path.join(str(tmp_path), "runs")
            assert args.run_name == str(tmp_path)
            assert '"output_dir"' in args.to_json_string()
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
        "train_end", "epoch_begin", "epoch_end",
    } <= names
    assert recorder.events[0] == ("init", 0, "steps")
    assert ("train_begin", 0, True) in recorder.events
    assert recorder.eval_metrics["eval_loss"] >= 0
    assert ClassCallback.calls == [0]
    assert trainer._saved == [str(tmp_path)]
    assert output.global_step == 3


@metal_only
def test_hf_callback_on_save_only_fires_for_checkpoints(tmp_path, monkeypatch):
    from pathlib import Path
    from transformers import TrainerCallback
    import unsloth_zoo.mlx.trainer as mlx_trainer

    class SaveRecorder(TrainerCallback):
        def __init__(self):
            self.saves = []

        def on_save(self, args, state, control, **_kwargs):
            self.saves.append((state.global_step, Path(args.output_dir)))

    def fake_save_trainable_adapters(_model, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        mlx_trainer, "save_trainable_adapters", fake_save_trainable_adapters,
    )
    monkeypatch.setattr(mlx_trainer, "save_optimizer_state", lambda *_args: None)
    monkeypatch.setattr(mlx_trainer, "save_trainer_state", lambda *_args: None)

    callback = SaveRecorder()
    trainer = _callback_trainer(
        tmp_path, [callback], max_steps=1, eval_steps=0, save_steps=1,
    )
    trainer.train()

    assert callback.saves == [(1, tmp_path)]
    assert trainer._saved == [str(tmp_path)]


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
def test_hf_callback_stop_allows_same_step_eval(tmp_path):
    from transformers import TrainerCallback

    class StopAndEval(TrainerCallback):
        def __init__(self):
            self.evals = []

        def on_step_end(self, args, state, control, **_kwargs):
            control.should_evaluate = True
            control.should_training_stop = True
            return control

        def on_evaluate(self, args, state, control, metrics, **_kwargs):
            self.evals.append((state.global_step, metrics["eval_loss"]))

    callback = StopAndEval()
    output = _callback_trainer(
        tmp_path,
        [callback],
        max_steps=1,
        eval_steps=0,
        with_eval=True,
    ).train()

    assert output.global_step == 1
    assert len(callback.evals) == 1
    assert callback.evals[0][0] == 1


@metal_only
def test_hf_callback_add_remove_pop_support_class_and_instance(tmp_path):
    from transformers import TrainerCallback

    class ClassCallback(TrainerCallback):
        pass

    class InstanceCallback(TrainerCallback):
        pass

    trainer = _callback_trainer(tmp_path, [])
    instance = InstanceCallback()

    trainer.add_callback(ClassCallback)
    trainer.add_callback(instance)
    assert any(isinstance(cb, ClassCallback) for cb in trainer.callback_handler.callbacks)
    assert instance in trainer.callback_handler.callbacks

    removed_class = trainer.pop_callback(ClassCallback)
    assert isinstance(removed_class, ClassCallback)
    assert not any(isinstance(cb, ClassCallback) for cb in trainer.callback_handler.callbacks)

    trainer.remove_callback(instance)
    assert instance not in trainer.callback_handler.callbacks

    assert trainer.pop_callback(ClassCallback) is None
    trainer.remove_callback(instance)


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


_NormTok = type("Tok", (), {"pad_token_id": 0, "eos_token_id": 0})


def _norm_model(seed=77, dtype=None):
    class _TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(32, 4)
            self.proj = nn.Linear(4, 32, bias=False)
            self._config = {"model_type": "tiny"}

        def __call__(self, input_ids):
            return self.proj(self.embed(input_ids))

    mx.random.seed(seed)
    model = _TinyLM()
    if dtype is not None:
        model.set_dtype(dtype)
    mx.eval(model.parameters())
    return model


def _norm_batches(count):
    # A CPU batch plan (not a raw list) keeps compiled runs compile-eligible.
    rows = ([1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15])
    return FiniteTextBatchPlan(
        [_FiniteTextRow(input_ids=tuple(row)) for row in rows],
        [(i % 3,) for i in range(count)],
        max_seq_length=8, pad_id=0,
    )


_CLIP = {
    "none": dict(max_grad_norm=0.0, max_grad_value=0.0, max_grad_leaf_norm=0.0),
    "leaf": dict(max_grad_norm=0.0, max_grad_value=None, max_grad_leaf_norm=0.01),
    "global": dict(max_grad_norm=0.01, max_grad_value=None, max_grad_leaf_norm=None),
}


def _norm_train(tmp_path, mode, *, report=False, compiled=False, accum=1,
                optim="adamw", weight_decay=0.0, dtype=None, max_steps=1,
                batches=None, overrides=None):
    import numpy as np

    model = _norm_model(dtype=dtype)
    config = dict(
        per_device_train_batch_size=1, gradient_accumulation_steps=accum,
        max_steps=max_steps, warmup_steps=0, learning_rate=1e-3,
        weight_decay=weight_decay, optim=optim, logging_steps=1, eval_steps=0,
        save_steps=0, max_seq_length=8, output_dir=str(tmp_path),
        compile=compiled, compile_mode="strict" if compiled else "eager",
        gradient_checkpointing=False, cast_norm_output_to_input_dtype=False,
        dataset_order="sequential", disable_memory_limits=True,
        use_cce=False, report_grad_norm=report, **_CLIP[mode],
    )
    config.update(overrides or {})
    args = MLXTrainingConfig(**config)

    class _Capturing(MLXTrainer):
        def _build_optimizer(self, total_steps):
            optimizer = super()._build_optimizer(total_steps)
            self.captured_optimizer = optimizer
            return optimizer

    trainer = _Capturing(model, _NormTok(), [], args=args)
    trainer._batches = batches if batches is not None else _norm_batches(max_steps * accum)
    trainer.save_model = lambda *_a, **_k: None
    callbacks = []
    trainer.add_step_callback(lambda *v: callbacks.append(v))
    result = trainer.train()
    snap = {
        f"param.{name}": (str(v.dtype), np.asarray(v.tolist()))
        for name, v in tree_flatten(trainer.model.trainable_parameters())
    }
    snap.update({
        f"opt.{name}": (str(v.dtype), np.asarray(v.tolist()))
        for name, v in tree_flatten(trainer.captured_optimizer.state)
        if hasattr(v, "dtype")
    })
    return trainer, result, callbacks, snap


def _oracle_norm(batches, seed=77):
    batches = batches.materialize_all()
    model = _norm_model(seed)
    acc, toks = None, mx.array(0.0, dtype=mx.float32)
    for batch in batches:
        (_l, n), grad = nn.value_and_grad(model, make_baseline_loss_fn())(model, *batch)
        weighted = tree_map(lambda g: g * n.astype(g.dtype), grad)
        acc = weighted if acc is None else tree_map(lambda a, b: a + b, acc, weighted)
        toks = toks + n.astype(mx.float32)
    sq = mx.array(0.0, dtype=mx.float32)
    for _n, v in tree_flatten(acc):
        sq = sq + mx.sum((v.astype(mx.float32) / toks) ** 2)
    return float(mx.sqrt(sq).item())


@metal_only
@pytest.mark.parametrize("mode,report,compiled,accum,optim,wd,expect", [
    ("global", False, True, 2, "adamw", 0.0, "oracle"),
    ("none", True, False, 3, "sgd", 0.5, "oracle"),  # decay excluded from norm
    ("none", True, True, 1, "lion", 0.0, "reported"),  # no Adam second moment
    ("leaf", False, False, 1, "adamw", 0.0, "absent"),
])
def test_grad_norm_reporting_matrix(tmp_path, mode, report, compiled, accum, optim, wd, expect):
    batches = _norm_batches(accum)
    trainer, _result, callbacks, _snap = _norm_train(
        tmp_path, mode, report=report, compiled=compiled, accum=accum,
        optim=optim, weight_decay=wd, batches=batches,
    )
    history = trainer._grad_norm_history
    if expect == "absent":
        assert history == [] and callbacks[0][8] is None
    else:
        assert len(history) == 1 and callbacks[0][8] == history[0]
        if expect == "oracle":
            assert history[0] == pytest.approx(_oracle_norm(batches), abs=1e-6)


@metal_only
def test_reporting_flag_never_changes_update_numerics(tmp_path):
    import numpy as np

    runs = {
        r: _norm_train(tmp_path / str(r), "none", report=r, compiled=True,
                       accum=1, max_steps=2, dtype=mx.bfloat16,
                       batches=_norm_batches(2))
        for r in (False, True)
    }
    (off_t, _res, _cb, off_snap), (on_t, _res2, _cb2, on_snap) = runs[False], runs[True]
    assert off_t._train_loss_history == on_t._train_loss_history
    for key in off_snap:
        if key.startswith("param."):
            assert off_snap[key][0] == on_snap[key][0] == "mlx.core.bfloat16"
        assert off_snap[key][0] == on_snap[key][0], key
        assert np.array_equal(off_snap[key][1], on_snap[key][1]), key
    assert on_t._grad_norm_history and not off_t._grad_norm_history


# ---- Compiled global-norm clipping with gradient accumulation ----

@metal_only
def test_compiled_clip_accum_matches_eager_bitwise(tmp_path, capsys):
    import numpy as np

    runs = {
        c: _norm_train(tmp_path / str(c), "global", compiled=c, accum=3,
                       max_steps=2, dtype=mx.bfloat16, batches=_norm_batches(6))
        for c in (False, True)
    }
    assert "mx.compile disabled because MLX global norm" not in capsys.readouterr().out
    (eager_t, _r, _cb, eager_snap), (comp_t, comp_res, _cb2, comp_snap) = runs[False], runs[True]
    assert comp_res["compile_enabled"] is True
    assert comp_res["compile_scope"] == "full_step"
    assert eager_t._train_loss_history == comp_t._train_loss_history
    assert eager_t._grad_norm_history == comp_t._grad_norm_history
    for key in eager_snap:
        assert eager_snap[key][0] == comp_snap[key][0], key
        assert np.array_equal(eager_snap[key][1], comp_snap[key][1]), key



@metal_only
def test_evaluation_failure_propagates_without_eager_retry(tmp_path, monkeypatch, capsys):
    import functools

    real_eval, real_compile = mx.eval, mx.compile
    state = {"armed": False, "raised": False, "step_ran": False}

    def failing_eval(*a, **k):
        if state["armed"] and not state["raised"]:
            state["raised"] = True
            raise RuntimeError("injected compile evaluation failure")
        return real_eval(*a, **k)

    def arming_compile(fn, *ca, **ck):
        compiled = real_compile(fn, *ca, **ck)

        @functools.wraps(fn)
        def wrapper(*fa, **fk):
            # Arm AFTER the compiled step returns so the next mx.eval is the
            # unified post-call boundary; a regression moving that eval inside
            # the fallback try would retry eagerly and fail this test.
            result = compiled(*fa, **fk)
            state["step_ran"] = True
            state["armed"] = True
            return result
        return wrapper

    monkeypatch.setattr(mx, "eval", failing_eval)
    monkeypatch.setattr(mx, "compile", arming_compile)
    with pytest.raises(RuntimeError, match="injected compile evaluation failure"):
        _norm_train(tmp_path, "global", compiled=True, accum=2,
                    batches=_norm_batches(2),
                    overrides={"compile_mode": "best_effort"})
    assert state["step_ran"] and state["raised"]
    assert "falling back to eager" not in capsys.readouterr().out

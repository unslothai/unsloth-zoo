"""Real MLX LoRA training smoke on Apple Silicon for the PR 684 trainer rework.

Downloads a tiny 4-bit model (~80MB), runs short FastMLXModel + MLXTrainer
LoRA fits, and checks losses, gradients-driven progress, and adapter saving.
Exercises the reworked grad-clip resolution (default leaf-norm path and the
explicit elementwise path), decoupled weight decay, batching, and both loss
functions (CCE and baseline).
"""

import glob
import json
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
        FiniteTextBatchPlan, _FiniteTextRow, collect_mlx_lora_adapter_tensors,
        make_baseline_loss_fn,
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



@metal_only
def test_lora_plus_ratio_scales_the_lora_b_step(tmp_path):
    """The LoRA+ ratio must actually scale lora_b's realized step.

    The old gradient pre-scale was an AdamW no-op. lora_b starts at zero, so
    its final L2 norm is its total movement. Same mechanism as embedding LR.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten

    def _lora_b_norm(trainer):
        total = 0.0
        for k, v in tree_flatten(trainer.model.trainable_parameters()):
            if k == "lora_b" or k.endswith(".lora_b"):
                total += float(mx.sqrt(mx.sum(v.astype(mx.float32) ** 2)).item())
        return total

    base = _lora_b_norm(_train(tmp_path / "r1", lora_plus_ratio=1.0, max_steps=6))
    boosted = _lora_b_norm(_train(tmp_path / "r8", lora_plus_ratio=8.0, max_steps=6))
    assert base > 0.0, "lora_b never moved at ratio=1"
    # Under the old gradient-scale no-op, boosted/base would be ~1.0.
    assert boosted > 3.0 * base, (
        f"LoRA+ ratio did not scale the step (fix regressed): "
        f"base={base:.4f} boosted={boosted:.4f} ratio={boosted / base:.2f}"
    )


@metal_only
@pytest.mark.parametrize("nested", [True, False])
def test_lora_plus_scales_layer_wrapped_lora_b_weight(tmp_path, nested):
    """mlx-lm may wrap the LoRA halves in nn.Linear children, flattening
    lora_b to `...lora_b.weight`. The scoped rescale must scale that layout
    too, both nested (`proj.lora_b.weight`) and root (`lora_b.weight`).
    """
    key = "proj.lora_b.weight" if nested else "lora_b.weight"

    def _b_weight_norm(ratio):
        class _WrappedLoRA(nn.Module):
            def __init__(s):
                super().__init__()
                s.embed = nn.Embedding(32, 4)
                host = nn.Module() if nested else s
                host.lora_a = mx.random.normal((4, 8)) * 0.2   # frozen, non-zero
                host.lora_b = nn.Linear(8, 32, bias=False)     # -> lora_b.weight
                host.lora_b.weight = mx.zeros((32, 8))          # zero-init B
                if nested:
                    s.proj = host
                s._config = {"model_type": "tiny"}

            def __call__(s, input_ids):
                host = s.proj if nested else s
                return host.lora_b(s.embed(input_ids) @ host.lora_a)

        mx.random.seed(77)
        m = _WrappedLoRA()
        mx.eval(m.parameters())
        m.freeze()
        (m.proj.lora_b if nested else m.lora_b).unfreeze(recurse=True)
        args = MLXTrainingConfig(
            per_device_train_batch_size=1, gradient_accumulation_steps=1,
            max_steps=6, warmup_steps=0, learning_rate=1e-3, optim="adamw",
            logging_steps=1, eval_steps=0, save_steps=0, max_seq_length=8,
            output_dir=str(tmp_path / str(ratio)), compile=False,
            compile_mode="eager", gradient_checkpointing=False,
            cast_norm_output_to_input_dtype=False, dataset_order="sequential",
            disable_memory_limits=True, use_cce=False, lora_plus_ratio=ratio,
            max_grad_norm=0.0, max_grad_value=0.0, max_grad_leaf_norm=0.0,
        )
        t = MLXTrainer(m, _NormTok(), [], args=args)
        t._batches = _norm_batches(6)
        t.save_model = lambda *_a, **_k: None
        t.train()
        w = dict(tree_flatten(t.model.trainable_parameters()))[key]
        return float(mx.sqrt(mx.sum(w.astype(mx.float32) ** 2)).item())

    base = _b_weight_norm(1.0)
    boosted = _b_weight_norm(8.0)
    assert base > 0.0, f"wrapped {key} never moved at ratio=1"
    assert boosted > 3.0 * base, (
        f"LoRA+ did not scale the wrapped {key} step: "
        f"base={base:.4f} boosted={boosted:.4f}"
    )


# ---------------------------------------------------------------------------
# Warm-starting continued training from a saved adapter: reloading a LoRA/DoRA
# adapter via FastMLXModel.from_pretrained must freeze the base and leave the
# adapter parameters trainable, together with any non-adapter tensors the
# checkpoint itself recorded as trainable. Uses a tiny locally-built Llama so
# the full_finetuning and DoRA branches stay cheap.
# ---------------------------------------------------------------------------


def _trainable_names(model):
    return {name for name, _ in tree_flatten(model.trainable_parameters())}


def _adapter_keys(model):
    # lora_a/lora_b for every LoRA module, plus m for DoRA modules only.
    return set(collect_mlx_lora_adapter_tensors(model).keys())


def _tiny_base(path):
    """Write a tiny unquantized HF Llama + tokenizer to ``path``."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
    # vocab_size matches hf-internal-testing/llama-tokenizer so token ids stay
    # in range (Metal indexing is unchecked).
    cfg = LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=32000,
        max_position_embeddings=128, tie_word_embeddings=False,
    )
    LlamaForCausalLM(cfg).save_pretrained(path, safe_serialization=True)
    AutoTokenizer.from_pretrained(
        "hf-internal-testing/llama-tokenizer"
    ).save_pretrained(path)
    return path


def _save_lora_adapter(base_path, adapter_path):
    """Attach LoRA to the tiny base and save an adapter directory."""
    from unsloth_zoo.mlx.utils import save_lora_adapters
    model, _ = FastMLXModel.from_pretrained(
        str(base_path), load_in_4bit=False, max_seq_length=64,
    )
    model = FastMLXModel.get_peft_model(
        model, r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
    )
    save_lora_adapters(model, str(adapter_path))
    return adapter_path


def _save_dora_adapter(base_path, adapter_path):
    """Build a DoRA adapter with mlx-lm and save it in reloadable form."""
    from mlx_lm.tuner.utils import linear_to_lora_layers
    model, _ = FastMLXModel.from_pretrained(
        str(base_path), load_in_4bit=False, max_seq_length=64,
    )
    num_layers = 2
    lora_params = {"rank": 8, "scale": 16.0, "dropout": 0.0,
                   "keys": ["self_attn.q_proj", "self_attn.v_proj"]}
    model.freeze()
    linear_to_lora_layers(model, num_layers, lora_params, use_dora=True)
    os.makedirs(adapter_path, exist_ok=True)
    mx.save_safetensors(
        os.path.join(str(adapter_path), "adapters.safetensors"),
        dict(tree_flatten(model.trainable_parameters())),
    )
    with open(os.path.join(str(adapter_path), "adapter_config.json"), "w") as f:
        json.dump({"fine_tune_type": "dora", "num_layers": num_layers,
                   "lora_parameters": lora_params,
                   "base_model_name_or_path": str(base_path)}, f)
    return adapter_path


@metal_only
def test_adapter_reload_freezes_base(tmp_path):
    base = _tiny_base(tmp_path / "base")
    adapter = _save_lora_adapter(base, tmp_path / "adapter")

    model, _ = FastMLXModel.from_pretrained(
        str(adapter), load_in_4bit=False, max_seq_length=64,
    )
    adapter_keys = _adapter_keys(model)
    assert any(n.endswith("lora_a") for n in adapter_keys)
    # The regression: the whole base used to come back trainable.
    assert _trainable_names(model) == adapter_keys


@metal_only
def test_warm_start_trains_only_adapter(tmp_path):
    base = _tiny_base(tmp_path / "base")
    adapter = _save_lora_adapter(base, tmp_path / "adapter")

    model, tok = FastMLXModel.from_pretrained(
        str(adapter), load_in_4bit=False, max_seq_length=64,
    )
    # A base weight must stay untouched while an adapter tensor must change
    # (else a no-op run would pass).
    probe_key = "model.layers.0.mlp.down_proj.weight"
    lora_key = next(n for n in _trainable_names(model) if n.endswith("lora_b"))
    params_before = dict(tree_flatten(model.parameters()))
    base_before = params_before[probe_key]
    lora_before = params_before[lora_key]
    cfg = MLXTrainingConfig(
        output_dir=str(tmp_path / "out"), per_device_train_batch_size=2,
        max_steps=2, learning_rate=1e-3, compile=False, use_cce=False,
        report_to="none",
    )
    MLXTrainer(
        model=model, tokenizer=tok,
        train_dataset=[{"text": f"warm start {i}"} for i in range(6)],
        args=cfg,
    ).train()
    params_after = dict(tree_flatten(model.parameters()))
    assert mx.array_equal(base_before, params_after[probe_key])
    assert not mx.array_equal(lora_before, params_after[lora_key])
    assert _trainable_names(model) == _adapter_keys(model)


@metal_only
def test_dora_reload_keeps_magnitude_trainable(tmp_path):
    base = _tiny_base(tmp_path / "base")
    adapter = _save_dora_adapter(base, tmp_path / "dora")

    from unsloth_zoo.mlx.utils import iter_mlx_lora_modules
    model, _ = FastMLXModel.from_pretrained(
        str(adapter), load_in_4bit=False, max_seq_length=64,
    )
    trainable = _trainable_names(model)
    dora_modules = [n for n, m in iter_mlx_lora_modules(model)
                    if type(m).__name__.startswith("DoRA")]
    assert len(dora_modules) > 0
    # Every DoRA magnitude must stay trainable (not just one).
    assert len([n for n in trainable if n.endswith(".m")]) == len(dora_modules)
    # Exactly the adapter tensors, so no base weight leaks in. (A base parameter
    # literally named "m" does not exist on this fixture, so that pathological
    # case is left to follow-up.)
    assert trainable == _adapter_keys(model)


@metal_only
def test_full_finetuning_reload_keeps_base_trainable(tmp_path):
    base = _tiny_base(tmp_path / "base")
    adapter = _save_lora_adapter(base, tmp_path / "adapter")

    model, _ = FastMLXModel.from_pretrained(
        str(adapter), load_in_4bit=False, max_seq_length=64,
        full_finetuning=True,
    )
    # full_finetuning is an explicit full-training request, so no freeze.
    assert _trainable_names(model) > _adapter_keys(model)


@metal_only
@pytest.mark.parametrize("prefetch", [False, True])
def test_resume_from_adapter_dir_names_warm_start(tmp_path, prefetch):
    base = _tiny_base(tmp_path / "base")
    adapter = _save_lora_adapter(base, tmp_path / "adapter")

    model, tok = FastMLXModel.from_pretrained(
        str(adapter), load_in_4bit=False, max_seq_length=64,
    )
    # Single-process streaming prefetch reads resume state early, before the
    # main resume block; that read must not pre-empt the completeness check
    # with a raw FileNotFoundError for trainer_state.json.
    extra = (
        dict(streaming=True, streaming_prefetch_batches=2) if prefetch else {}
    )
    cfg = MLXTrainingConfig(
        output_dir=str(tmp_path / "out"), per_device_train_batch_size=2,
        max_steps=2, learning_rate=1e-3, compile=False, use_cce=False,
        report_to="none", **extra,
    )
    trainer = MLXTrainer(
        model=model, tokenizer=tok,
        train_dataset=[{"text": f"row {i}"} for i in range(6)], args=cfg,
    )
    # No optimizer_state.safetensors, so resume must fail and name warm-start
    # rather than silently restarting.
    with pytest.raises(RuntimeError, match="from_pretrained"):
        trainer.train(resume_from_checkpoint=str(adapter))


@metal_only
def test_reload_keeps_saved_non_adapter_trainables(tmp_path):
    from unsloth_zoo.mlx.utils import save_trainable_adapters

    base = _tiny_base(tmp_path / "base")
    model, _ = FastMLXModel.from_pretrained(
        str(base), load_in_4bit=False, max_seq_length=64,
    )
    model = FastMLXModel.get_peft_model(
        model, r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
    )
    # Non-adapter tensors a user legitimately trains alongside LoRA.
    aux = set()
    modules = dict(model.named_modules())
    for path in ("model.embed_tokens", "lm_head", "model.norm"):
        modules[path].unfreeze(recurse=True)
        aux.update(
            f"{path}.{name}" for name, _ in tree_flatten(modules[path].parameters())
        )
    assert aux <= _trainable_names(model)

    adapter = tmp_path / "adapter"
    save_trainable_adapters(model, str(adapter))
    saved = set(mx.load(str(adapter / "adapters.safetensors")).keys())
    assert aux <= saved

    reloaded, _ = FastMLXModel.from_pretrained(
        str(adapter), load_in_4bit=False, max_seq_length=64,
    )
    trainable = _trainable_names(reloaded)
    # The base freeze must not silently drop the saved auxiliary trainables.
    assert aux <= trainable, sorted(aux - trainable)
    assert _adapter_keys(reloaded) <= trainable
    # Still a warm start, not a full finetune: nothing beyond adapters + aux.
    assert trainable == _adapter_keys(reloaded) | aux


@metal_only
def test_vlm_planned_vs_unplanned_training_parity(monkeypatch, tmp_path):
    """Real-runtime contract for planned VLM training: with a qualified
    compile decision the trainer surveys, installs a width plan, and runs
    the compiled path over planned widths only; losses and token counts
    match the unplanned eager run exactly (padded tails are inert), and
    every compiled input width is an admitted endpoint."""
    import os as _os
    import sys as _sys
    import types

    import mlx.nn as nn
    import unsloth_zoo.mlx.trainer as trainer_mod
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from test_mlx_batching_and_decay import _WidthOnlyProcessor as _Proc

    class TinyVLM(nn.Module):
        # Keep the genuine train()/state members: the compiled step threads
        # model.state, so overriding it would leak traced parameters out.
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(260, 8)
            self.proj = nn.Linear(8, 260, bias=False)
            self._config = {"model_type": "tiny"}

        def __call__(self, inputs, pixel_values=None, mask=None, **_kwargs):
            return self.proj(self.embed(inputs))

    seen_widths = []
    original_call = TinyVLM.__call__

    def recording_call(self, inputs, *args, **kwargs):
        seen_widths.append(int(inputs.shape[1]))
        return original_call(self, inputs, *args, **kwargs)

    compiled_invocations = []
    real_compile = mx.compile

    def counting_compile(fn, **kwargs):
        compiled = real_compile(fn, **kwargs)

        def tracked(*args):
            compiled_invocations.append(1)
            return compiled(*args)

        return tracked

    def run(planned):
        mx.random.seed(7)
        seen_widths.clear()
        compiled_invocations.clear()
        plan = _create_vlm_batch_plan(
            dataset=[{"text": str(i)} for i in range(4)],
            processor=_Proc(),
            config={"image_size": 16, "image_token_id": 200},
            batch_size=1,
            max_seq_length=8,
        )
        args = MLXTrainingConfig(
            max_steps=4,
            gradient_accumulation_steps=1,
            compile=planned,
            use_cce=False,
            gradient_checkpointing=False,
            cast_norm_output_to_input_dtype=False,
            max_grad_norm=0.0,
            max_grad_leaf_norm=0.0,
            disable_memory_limits=True,
            logging_steps=1000,
            output_dir=str(tmp_path),
        )
        trainer = MLXTrainer(
            TinyVLM(),
            types.SimpleNamespace(pad_token_id=2, eos_token_id=2),
            [],
            args=args,
        )
        trainer._is_vlm = True
        trainer.processor = _Proc()
        trainer._batches = plan
        if planned:
            enabled_decision = types.SimpleNamespace(
                should_raise=False, enabled=True, arch="tiny",
                reason="mocked", setting_recommendations=(),
                fallback_allowed=True, support_state="supported_verified",
                strict=False, policy_mode="auto",
            )
            monkeypatch.setattr(
                trainer_mod, "resolve_training_compile",
                lambda *_a, **_k: enabled_decision,
            )
            monkeypatch.setattr(
                trainer_mod, "trace_compile_application",
                lambda *_a, **_k: None,
            )
            monkeypatch.setattr(
                trainer_mod, "explain_compile_support", lambda *_a, **_k: "",
            )
            monkeypatch.setattr(
                trainer_mod, "get_compile_qualification",
                lambda *_a, **_k: None,
            )
        trainer.save_model = lambda *_a, **_k: None
        result = trainer.train()
        return result, plan, list(seen_widths)

    monkeypatch.setattr(TinyVLM, "__call__", recording_call)
    monkeypatch.setattr(mx, "compile", counting_compile)
    unplanned_result, _plan, unplanned_widths = run(planned=False)
    unplanned_compiled_calls = len(compiled_invocations)
    planned_result, planned_plan, planned_widths = run(planned=True)
    planned_compiled_calls = len(compiled_invocations)

    # The planned run really executes through mx.compile, one invocation per
    # training step, while the unplanned eager run never compiles.
    assert unplanned_compiled_calls == 0
    assert planned_compiled_calls == 4
    assert planned_result["compile_enabled"] is True
    assert unplanned_result["compile_enabled"] is False
    assert planned_result["trained_tokens"] == (
        unplanned_result["trained_tokens"]
    )
    assert planned_result["train_loss"] == pytest.approx(
        unplanned_result["train_loss"], rel=1e-4,
    )
    # Every compiled width is a planned endpoint at or above the raw width, and
    # the planned run exposed at most the admitted variants.
    endpoints = {
        planned_plan._shape_plan.endpoint_for(
            planned_plan.batch_family(index),
            planned_plan._planned_widths[index],
        )
        for index in range(len(planned_plan))
    }
    assert set(planned_widths) == endpoints
    assert all(
        width >= raw
        for width, raw in zip(
            planned_widths,
            (
                planned_plan.batch_width(
                    planned_plan.batch_index_for_visit(visit)
                )
                for visit in range(len(planned_widths))
            ),
        )
    )
    assert len(set(planned_widths)) <= len(set(unplanned_widths)) + 1

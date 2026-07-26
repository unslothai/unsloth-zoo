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

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
    from unsloth_zoo.mlx.utils import make_baseline_loss_fn

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
    rows = ([1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15])
    return [
        (mx.array([rows[i % 3]], dtype=mx.int32),
         mx.array([[0, len(rows[i % 3])]], dtype=mx.int32), None)
        for i in range(count)
    ]


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

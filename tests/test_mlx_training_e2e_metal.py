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
    _U, _CPU = lambda *s: None, None   # parametrization evaluates before the skip

metal_only = pytest.mark.skipif(not _METAL, reason="requires Apple Silicon Metal")
_K_REM, _ROW_OVF = "k_remainder", "row_overflow"
_UNTRANSPOSED = dict(k=192, rows=1, out_width=128, transpose=False, rhs_indices=None)

if _METAL:
    # Module scope: leaked mlx-simulation shims must not hijack test-time imports.
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_map
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    from unsloth_zoo.mlx import utils as mlx_utils
    from unsloth_zoo.mlx.utils import (
        FiniteTextBatchPlan, _FiniteTextRow, collect_mlx_lora_adapter_tensors,
        make_baseline_loss_fn,
    )
    _U, _CPU = lambda *s: mx.zeros(s, dtype=mx.uint32), mx.default_stream(mx.cpu)

MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"


@metal_only
@pytest.mark.parametrize("family", ["qwen2_vl", "glm4v"])
def test_cacheless_text_positions_match_language_wrapper(family):
    import importlib
    from types import SimpleNamespace

    configs = importlib.import_module(f"mlx_vlm.models.{family}.config")
    language = importlib.import_module(f"mlx_vlm.models.{family}.language")
    args = configs.TextConfig.from_dict(dict(
        model_type=family, hidden_size=512, num_hidden_layers=2,
        intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
        rms_norm_eps=1e-5, vocab_size=32, max_position_embeddings=128,
        rope_scaling={"type": "default", "rope_type": "default",
                      "mrope_section": [8, 12, 12] if family == "glm4v" else [16, 24, 24]},
    ))
    config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2),
                             image_token_id=40, video_token_id=41, vision_start_token_id=42)
    lm = language.LanguageModel(args, config)
    for rows in ([[2, 3, 4], [5, 6, 7]], [[7, 6], [4, 3]], [[2, 3, 4], [5, 6, 8]]):
        inputs = mx.array(rows)
        positions, _ = lm.get_rope_index(inputs)
        expected = mlx_utils._model_logits(lm(inputs, position_ids=positions))
        lm._position_ids = mx.full((3, 1, 1), 97)
        hidden = mlx_utils._forward_text_hidden_states(lm, inputs)
        actual = lm.lm_head(hidden)
        mx.eval(actual, expected)
        assert mx.allclose(actual, expected, atol=1e-5).item()
        assert lm._position_ids.shape == (3, 1, 1)
        explicit = positions + mx.array([0, 1, 2])[:, None, None]
        expected_hidden = lm.model(inputs, position_ids=explicit)
        actual_hidden = mlx_utils._forward_text_hidden_states(lm, inputs, position_ids=explicit)
        assert mx.array_equal(actual_hidden, expected_hidden).item()


@metal_only
def test_cached_position_attribute_does_not_imply_multiaxis_rope():
    from mlx_vlm.models.minimax_m3_vl.config import TextConfig
    from mlx_vlm.models.minimax_m3_vl.language import LanguageModel

    args = TextConfig(hidden_size=64, intermediate_size=64, dense_intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                      head_dim=32, vocab_size=32, num_local_experts=2, num_experts_per_tok=1,
                      shared_intermediate_size=64, moe_layer_freq=[0, 0])
    lm = LanguageModel(args)
    inputs = mx.array([[2, 3, 4], [5, 6, 7]])
    expected = lm(inputs).logits
    actual = lm.lm_head(mlx_utils._forward_text_hidden_states(lm, inputs))
    assert mx.allclose(actual, expected, atol=1e-5).item()


@metal_only
@pytest.mark.parametrize("static", [False, True])
def test_native_vlm_names_preserve_source_remaps_and_transforms(monkeypatch, static):
    import inspect
    from unsloth_zoo.mlx import loader

    class RenamingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Linear(3, 3)
            self.other = nn.Linear(3, 3)

    def sanitize(weights):
        return {
            ("decoder." + k if k.startswith("decoder.") else k.replace("source.", "other.")):
            (v.T if k == "decoder.bias" else v)
            for k, v in weights.items()
        }

    RenamingModel.sanitize = staticmethod(sanitize) if static else lambda self, weights: sanitize(weights)

    original_class_call = RenamingModel.sanitize
    original_signature = inspect.signature(RenamingModel().sanitize)
    monkeypatch.setattr(loader, "_resolve_mlx_vlm_model_class", lambda _: RenamingModel)
    loader._ensure_native_vlm_weight_names("arbitrary")
    loader._ensure_native_vlm_weight_names("arbitrary")
    model = RenamingModel()
    assert RenamingModel.sanitize is original_class_call
    assert inspect.signature(model.sanitize) == original_signature
    weights = {"decoder.weight": mx.ones((3, 3)), "source.weight": mx.zeros((3, 3)),
               "decoder.scales": mx.ones((3, 1)), "decoder.biases": mx.zeros((3, 1)),
               "unknown.weight": mx.ones((2, 2)), "decoder.bias": mx.ones((3,))}
    result = model.sanitize(weights)
    assert result["decoder.weight"] is weights["decoder.weight"]
    for key in ("decoder.scales", "decoder.biases"):
        assert result[key] is weights[key]
    assert result["other.weight"] is weights["source.weight"]
    assert result["unknown.weight"] is weights["unknown.weight"]
    assert "decoder.decoder.bias" in result
    assert "decoder.bias" not in result
    if static:
        assert RenamingModel.sanitize(weights).keys() == sanitize(weights).keys()
        assert mlx_utils._call_mlx_vlm_sanitize(RenamingModel, {}, weights).keys() == sanitize(weights).keys()


@metal_only
@pytest.mark.parametrize("per_layer", [False, True])
def test_recurrent_language_layers_receive_one_adapter_each(per_layer):
    from unsloth_zoo.mlx.loader import linear_to_lora_layers
    from mlx_lm.tuner.lora import LoRALinear

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(4, 4, bias=False)
            self.v_proj = nn.Linear(4, 4, bias=False)

    class RecurrentModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.low, self.high = Layer(), Layer()

        @property
        def layers(self):
            return [self.low, self.low, self.high, self.low, self.high]

    model = RecurrentModel()
    config = dict(keys=["q_proj"], rank=2, scale=2, dropout=0)
    if per_layer:
        config.update(keys=["q_proj", "v_proj"],
                      layer_keys=[["q_proj"], ["q_proj"], ["v_proj"], ["q_proj"], ["v_proj"]])
    assert linear_to_lora_layers(model, 5, config) == 2
    assert isinstance(model.low.q_proj, LoRALinear)
    if per_layer:
        assert isinstance(model.high.v_proj, LoRALinear)
        assert not isinstance(model.high.q_proj, LoRALinear)
        assert not isinstance(model.low.v_proj, LoRALinear)
    else:
        assert isinstance(model.high.q_proj, LoRALinear)
        assert model.low.q_proj is not model.high.q_proj


@metal_only
@pytest.mark.parametrize("window, T, windowed", [
    (256, 1600, True), (640, 2000, True), (640, 1280, False), (None, 4100, False)])
def test_training_attention_visits_only_the_window(monkeypatch, window, T, windowed):
    from mlx_vlm.models import base as vlm_base

    mx.random.seed(0)
    B, heads, dim = 2, 4, 32

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(dim, 2 * dim)

        def __call__(self, x, mask=None):
            q, kv = mx.split(self.qkv(x), [dim], axis=-1)
            q = q.reshape(B, T, heads, -1).transpose(0, 2, 1, 3)
            k, v = (t.reshape(B, T, heads // 2, -1).transpose(0, 2, 1, 3)
                    for t in mx.split(kv, 2, axis=-1))
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.3, mask=mask)
            return out.transpose(0, 2, 1, 3).reshape(B, T, dim)

    mlx_utils._patch_layer_class_for_gc(Attention)
    layer, x = Attention(), mx.random.normal((B, T, dim))
    windows = []
    original = mlx_utils._windowed_attention
    monkeypatch.setattr(mlx_utils, "_windowed_attention",
                        lambda *a: windows.append(a[-2]) or original(*a))

    def step():
        def loss(params, x):
            layer.update(params)
            mask = "causal" if window is None else vlm_base.create_attention_mask(x, None, window_size=window)
            return (layer(x, mask=mask) ** 2).sum()

        params = layer.trainable_parameters()
        out = mx.compile(mx.value_and_grad(loss, argnums=(0, 1)))(params, x)
        mx.eval(out)
        layer.update(params)
        return out

    try:
        dense = step()
        # A second run keeps the wrapper installed while this one evaluates.
        mlx_utils.acquire_mlx_training_patches()
        mlx_utils.acquire_mlx_training_patches()
        try:
            trained = step()
            calls = len(windows)
            paused = mlx_utils.pause_mlx_training_patches()
            try:
                step()
            finally:
                mlx_utils.resume_mlx_training_patches(paused)
        finally:
            mlx_utils.release_mlx_training_patches()
            mlx_utils.release_mlx_training_patches()
    finally:
        mlx_utils._unpatch_layer_class_gc(Attention)
    assert len(windows) == calls
    assert set(windows) == ({window} if windowed else set())
    assert not hasattr(mx.fast.scaled_dot_product_attention, "_unsloth_original")
    assert not hasattr(vlm_base.create_causal_mask, "_unsloth_original")
    (loss, (grads, dx)), (ref_loss, (ref_grads, ref_dx)) = trained, dense
    assert mx.allclose(loss, ref_loss, rtol=1e-5)
    assert mx.allclose(dx, ref_dx, atol=1e-4)
    for key in ("weight", "bias"):
        assert mx.allclose(grads["qkv"][key], ref_grads["qkv"][key], rtol=1e-4, atol=1e-4)


@metal_only
@pytest.mark.parametrize("heads, T, dim, pays", [
    (8, 1536, 256, False), (8, 2048, 256, True), (8, 2048, 512, False), (8, 4096, 512, True)])
def test_windowing_pays_only_past_the_break_even(heads, T, dim, pays):
    q, k = mx.zeros((1, heads, T, dim)), mx.zeros((1, 1, T, dim))
    assert mlx_utils._windowing_pays(q, k, 512) is pays


@metal_only
@pytest.mark.parametrize("family", ["gemma4", "gemma3n"])
@pytest.mark.parametrize("T", [2048, 5200])
def test_windowed_attention_trains_kv_shared_gemma(monkeypatch, family, T):
    # gemma3n shares K/V through the zoo's slots, gemma4 through mlx-vlm itself.
    import importlib
    from mlx.utils import tree_flatten

    config_module = importlib.import_module(f"mlx_vlm.models.{family}.config")
    language = importlib.import_module(f"mlx_vlm.models.{family}.language")
    mx.random.seed(0)
    layers, types = 15, (["sliding_attention"] * 4 + ["full_attention"]) * 3
    shape = dict(
        num_hidden_layers=layers, num_kv_shared_layers=10, sliding_window=512, layer_types=types,
        hidden_size=64, head_dim=32, num_attention_heads=2, num_key_value_heads=1,
        hidden_size_per_layer_input=16, vocab_size=512, vocab_size_per_layer_input=512)
    if family == "gemma4":
        shape.update(intermediate_size=128, global_head_dim=64)
    else:
        shape.update(model_type="gemma3n_text", intermediate_size=[128] * layers, laurel_rank=8,
                     activation_sparsity_pattern=[0.0] * layers)
    model = language.LanguageModel(config_module.TextConfig(**shape))
    ids = mx.random.randint(0, 512, (1, T))
    windows = []
    original = mlx_utils._windowed_attention
    monkeypatch.setattr(mlx_utils, "_windowed_attention",
                        lambda *a: windows.append(a[-2]) or original(*a))

    def step():
        def loss(params):
            model.update(params)
            caches = mlx_utils._build_shared_kv_caches(model)
            return model(ids, cache=caches).logits.astype(mx.float32).mean()

        params = model.trainable_parameters()
        out = mx.compile(mx.value_and_grad(loss))(params)
        mx.eval(out)
        model.update(params)
        return out

    layer_class = type(model.model.layers[0])
    mlx_utils._patch_layer_class_for_gc(layer_class)
    try:
        dense = step()
        mlx_utils.acquire_mlx_training_patches()
        try:
            trained = step()
        finally:
            mlx_utils.release_mlx_training_patches()
    finally:
        mlx_utils._unpatch_layer_class_gc(layer_class)
    # Every sliding layer, in the forward and in its checkpoint recompute.
    assert windows == [512] * 2 * types.count("sliding_attention")
    (loss, grads), (ref_loss, ref_grads) = trained, dense
    assert mx.allclose(loss, ref_loss, rtol=1e-5)
    ref = dict(tree_flatten(ref_grads))
    for name, grad in tree_flatten(grads):
        assert (grad - ref[name]).abs().max() <= 1e-2 * ref[name].abs().max() + 1e-8, name


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


def _cce_text_model(rows, dim, *, quantized, lora=False, calls=None, softcap=0.0):
    from types import SimpleNamespace
    from mlx_lm.tuner.lora import LoRALinear
    from unsloth_zoo.mlx.cce.runtime_cce import _apply_softcap

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(rows, dim)

        def __call__(self, ids):
            if calls is not None:
                calls.append("backbone")
            return self.embed_tokens(ids)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()
            self.lm_head = LoRALinear(dim, 8192, r=8, scale=2.0) if lora else nn.Linear(dim, 8192, bias=False)
            base = self.lm_head.linear if lora else self.lm_head
            if lora and (softcap or dim > 1024):
                dtype = mx.float16 if softcap and not quantized else mx.bfloat16
                self.model.set_dtype(dtype)
                base.set_dtype(dtype)
                self.model.embed_tokens.weight *= 50 if softcap else 1
                if softcap and not quantized:
                    self.model.embed_tokens.weight = mx.full((rows, dim), 200, dtype=dtype)
                    base.weight = mx.full((8192, dim), 200, dtype=dtype)
                    base.bias = mx.full((8192,), -40000 * dim, dtype=mx.float32)
            if quantized:
                base = nn.QuantizedLinear.from_linear(base)
            if lora or quantized:
                base.freeze()
            if lora:
                self.lm_head.linear = base
                self.lm_head.lora_b = mx.random.normal((8, 8192)) * 0.02
            else:
                self.lm_head = base
            self.args = SimpleNamespace(tie_word_embeddings=False, final_logit_softcapping=softcap)

        def __call__(self, ids):
            if calls is not None:
                calls.append("model")
            return _apply_softcap(self.lm_head(self.model(ids)), softcap)

    return Model()


@metal_only
@pytest.mark.parametrize("quantized", [False, True])
def test_cce_compacts_finite_supervision_with_one_trace(monkeypatch, quantized):
    import numpy as np
    from types import SimpleNamespace
    from mlx.utils import tree_flatten

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(2053, 64)

        def __call__(self, ids):
            return self.embed_tokens(ids)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()
            self.lm_head = nn.Linear(64, 8192, bias=False)
            self.args = SimpleNamespace(tie_word_embeddings=False)
            if quantized:
                self.lm_head = nn.QuantizedLinear.from_linear(self.lm_head)
                self.lm_head.freeze()

    rows = []
    for row in range(4):
        labels = np.full(513, -100, dtype=np.int64)
        positions = np.arange(20 + row, 513, 11 + row)
        labels[positions] = (positions * 17 + row) % 2048
        labels[1] = 7
        rows.append(_FiniteTextRow(
            tuple(range(row * 513, (row + 1) * 513)), offset=10, labels=tuple(labels),
        ))
    plan = FiniteTextBatchPlan(rows, [(0, 1), (2, 3)], max_seq_length=513, pad_id=0)
    plan.configure_cce_compaction()
    kernel_rows = []
    original = mlx_utils._get_runtime_cce

    def factory(**kwargs):
        runtime = original(**kwargs)

        def record(hidden, *args):
            kernel_rows.append(hidden.shape[0])
            return runtime(hidden, *args)

        return record

    monkeypatch.setattr(mlx_utils, "_get_runtime_cce", factory)
    mx.random.seed(853)
    model = Model()
    loss_fn = mlx_utils.make_cce_loss_fn(model)
    assert loss_fn._unsloth_cce_compaction
    grad = nn.value_and_grad(model, loss_fn)
    compiled = mx.compile(lambda *batch: grad(model, *batch), inputs=model.state, outputs=model.state)
    for index in range(2):
        batch = plan[index]
        prepared = plan.prepare_cce_batch(index, batch)
        assert prepared[3].shape == (256,)
        assert mx.any(prepared[3] >= 512).item()
        reference = grad(model, *batch)
        kernel_rows.clear()
        actual = compiled(*prepared)
        mx.eval(reference, actual)
        assert kernel_rows == ([256] if index == 0 else [])
        assert actual[0][0].item() == pytest.approx(reference[0][0].item(), abs=3e-5)
        assert actual[0][1].item() == reference[0][1].item()
        for (_, expected), (_, got) in zip(tree_flatten(reference[1]), tree_flatten(actual[1])):
            assert mx.allclose(expected, got, atol=2e-5, rtol=2e-4).item()
    plan.configure_cce_compaction(False)
    assert plan.prepare_cce_batch(1, batch) is batch


@metal_only
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("softcap", [0.0, 7.0])
def test_frozen_dense_cce_preserves_gradients_with_lower_peak(monkeypatch, compiled, softcap):
    import gc
    from unsloth_zoo.mlx.cce import runtime_cce
    from unsloth_zoo.mlx.cce.runtime_cce import make_chunked_cross_entropy_loss

    stored, forward = [], runtime_cce._forward_with_hidden_gradient
    monkeypatch.setattr(runtime_cce, "_forward_with_hidden_gradient",
                        lambda *args, **kwargs: (out := forward(*args, **kwargs), stored.append(out[1].dtype))[0])

    mx.random.seed(719)
    hidden = (mx.random.normal((256, 128)) * 0.2).astype(mx.bfloat16)
    weight = (mx.random.normal((8192, 128)) * 0.1).astype(mx.bfloat16)
    targets = mx.where(mx.arange(256) % 5 == 0, -100, mx.arange(256) * 29)
    cotangent = mx.where(mx.arange(256) % 2 == 0, 1.0, -0.5)
    mx.eval(hidden, weight, targets, cotangent)
    functions = []
    for frozen, precompute in ((False, False), (True, False), (True, True)):
        runtime, _ = make_chunked_cross_entropy_loss(
            chunk_size=2048, weight_is_frozen=frozen, logit_softcap=softcap,
            precompute_hidden_gradient=precompute,
        )
        def loss(h, runtime=runtime):
            return (runtime(h, weight, targets) * cotangent).sum()
        grad = mx.value_and_grad(loss)
        functions.append(mx.compile(grad) if compiled else grad)
    expected, actual, precomputed = [fn(hidden) for fn in functions]
    reference, _ = make_chunked_cross_entropy_loss(chunk_size=2048, logit_softcap=softcap)
    exact = mx.grad(lambda h: (reference(h, weight.astype(mx.float32), targets) * cotangent).sum())(
        hidden.astype(mx.float32)
    )
    mx.eval(expected, actual, precomputed, exact)
    for left, right in zip(expected, actual):
        assert mx.array_equal(left, right).item()
    assert mx.array_equal(precomputed[0], expected[0]).item()
    assert mx.allclose(precomputed[1].astype(mx.float32), exact, atol=2e-3, rtol=0).item()
    # Held until the backward, the gradient is stored at the hidden dtype.
    assert stored == [mx.bfloat16]
    assert mx.all(actual[1][::5] == 0).item()
    assert mx.any(actual[1][1:] > 0).item() and mx.any(actual[1][1:] < 0).item()
    del expected, actual, precomputed, exact, left, right
    # Head-only peaks cover the release; precomputing saves memory only across a compiled model step.
    peaks = []
    for fn in functions[:2]:
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        resident = mx.get_active_memory()
        mx.reset_peak_memory()
        result = fn(hidden)
        mx.eval(result)
        peaks.append(mx.get_peak_memory() - resident)
        del result
    assert peaks[1] < peaks[0]


@metal_only
def test_trainer_compiled_step_uses_lora_head_cce(monkeypatch, tmp_path):
    calls = []
    factory = mlx_utils._make_text_lora_cce_loss_fn

    def recording(*args, **kwargs):
        loss = factory(*args, **kwargs)

        def record(model, *batch):
            calls.append(len(batch))
            return loss(model, *batch)

        return record

    monkeypatch.setattr(mlx_utils, "_make_text_lora_cce_loss_fn", recording)
    mx.random.seed(919)
    model = _cce_text_model(2049, 1024, quantized=False, lora=True)
    ids = tuple(range(2049))
    labels = tuple(i if i % 10 == 0 else -100 for i in ids)
    trainer = MLXTrainer(model=model, tokenizer=None, train_dataset=[], args=MLXTrainingConfig(
        max_steps=1, per_device_train_batch_size=1, gradient_accumulation_steps=1,
        learning_rate=1e-4, logging_steps=1, save_steps=0, output_dir=str(tmp_path),
        compile=True, gradient_checkpointing=False, report_to="none",
    ))
    trainer._batches = FiniteTextBatchPlan(
        [_FiniteTextRow(ids, offset=0, labels=labels)], [(0,)], max_seq_length=2049, pad_id=0,
    )
    trainer.save_model = lambda output_dir=None: None
    trainer.train()
    assert calls == [3]


@metal_only
def test_lora_head_cce_without_metal_keeps_baseline(monkeypatch):
    model = _cce_text_model(2049, 1024, quantized=False, lora=True)
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    loss = mlx_utils.make_cce_loss_fn(model)
    assert not hasattr(loss, "_unsloth_compiled_loss_fn")


@metal_only
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("tokens, softcap, dim, promoted", [(128, 0.0, 1024, False), (2048, 0.0, 1024, False), (2048, 5.0, 1024, False), (2048, 0.0, 4096, False), (2048, 0.0, 4096, True)])
def test_lora_head_cce_live_gradients_and_early_fallback(quantized, tokens, softcap, dim, promoted):
    calls = []
    low_precision = softcap > 0 or (dim > 1024 and not promoted)
    mx.random.seed(917)
    model = _cce_text_model(2049, dim, quantized=quantized, lora=True, calls=calls, softcap=softcap)
    if promoted:
        model.model.embed_tokens.weight = model.model.embed_tokens.weight.astype(mx.float32)
    ids = mx.arange(tokens + 1, dtype=mx.int32)[None, :]
    labels = mx.where(ids % 5 == 0, -100, ids)
    batch = (ids, mx.array([[0, tokens + 1]], dtype=mx.int32), labels)
    loss = mlx_utils.make_cce_loss_fn(model)
    grad = nn.value_and_grad(model, getattr(loss, "_unsloth_compiled_loss_fn", loss))
    compiled = mx.compile(lambda *b: grad(model, *b), inputs=model.state, outputs=model.state)
    native_loss = make_baseline_loss_fn()
    reference = nn.value_and_grad(model, lambda m, *b: native_loss(lambda ids: m(ids).astype(mx.float32), *b))
    first_loss = None
    for iteration in range(2):
        calls.clear()
        actual = compiled(*batch)
        mx.eval(actual)
        assert calls == ([] if iteration else (["model", "backbone"] if tokens == 128 else ["backbone"]))
        expected = reference(model, *batch)
        mx.eval(expected)
        assert actual[0][1].item() == expected[0][1].item()
        assert actual[0][0].item() == pytest.approx(expected[0][0].item(), abs=(0.005 if low_precision else 1e-5) if tokens > 128 else 0, rel=0)
        for (_, want), (_, got) in zip(tree_flatten(expected[1]), tree_flatten(actual[1])):
            assert (mx.allclose(want, got, atol=2e-5 if low_precision else 2e-6, rtol=0.02 if low_precision else 2e-4) if tokens > 128 else mx.array_equal(want, got)).item()
        for key in ("lora_a", "lora_b"):
            assert mx.any(actual[1]["lm_head"][key] != 0).item()
        if iteration:
            assert actual[0][0].item() != first_loss
        first_loss = actual[0][0].item()
        model.lm_head.lora_a = model.lm_head.lora_a * 2.0
        model.lm_head.lora_b = model.lm_head.lora_b * 3.0


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


# Warm-starting continued training from a saved adapter: reloading a LoRA/DoRA
# adapter via FastMLXModel.from_pretrained must freeze the base and leave the
# adapter parameters trainable, together with any non-adapter tensors the
# checkpoint itself recorded as trainable. Uses a tiny locally-built Llama so
# the full_finetuning and DoRA branches stay cheap.


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
    assert trainable == _adapter_keys(reloaded) | aux
    assert reloaded._unsloth_reloaded_parameter_keys == aux


def _record_cce_rows(monkeypatch):
    rows, original = [], mlx_utils._get_runtime_cce

    def factory(**kwargs):
        runtime = original(**kwargs)
        return lambda hidden, *args: (rows.append(hidden.shape[0]), runtime(hidden, *args))[1]

    monkeypatch.setattr(mlx_utils, "_get_runtime_cce", factory)
    return rows


@metal_only
@pytest.mark.parametrize("head", ["dense", "quantized", "softcap"])
@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("prompt_share", ["low", "high"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("kind", ["dpo", "orpo"])
def test_preference_cce_scores_hidden_states_like_the_logits(monkeypatch, head, reference, prompt_share, dtype, kind):
    from unsloth_zoo.mlx import preference as p
    from mlx_lm.tuner.lora import LoRAEmbedding
    mx.random.seed(937)
    kernel_rows, calls = _record_cce_rows(monkeypatch), []
    model = _cce_text_model(2053, 64, quantized=head == "quantized", calls=calls,
                            softcap=0.5 if head == "softcap" else 0.0)
    policy = None
    if reference:
        adapter = LoRAEmbedding.from_base(model.model.embed_tokens, r=4, scale=2.0)
        adapter.lora_b = mx.random.normal(adapter.lora_b.shape) * 0.02
        model.model.embed_tokens = adapter
        model.freeze()
        adapter.unfreeze(keys=["lora_a", "lora_b"])
        policy = p.ReferencePolicy(scales=[(adapter, 0.0)])
    model.set_dtype(getattr(mx, dtype))
    rows = []
    for i in range(8):
        chosen = tuple((j * 7 + i) % 2053 for j in range(513 if i % 4 == 0 else 45 + i))
        rejected = tuple((j * 11 + i) % 2053 for j in range(507 if i % 4 == 0 else 43 + i))
        cut, other = (-13 - i, -14 + i) if prompt_share == "high" else (13 + i, 14 - i)
        rows.append(p.TokenizedPreferenceRow(chosen[:cut], chosen[cut:], rejected[:other], rejected[other:]))
    normalizers = [(sum(len(row.chosen) - 1 for row in rows), 8, 2)] * 2
    plan = p.FinitePreferenceBatchPlan(rows, [(0, 1, 2, 3), (4, 5, 6, 7)], normalizers=normalizers,
                                     cycle_length=2, max_seq_length=513, pad_id=0)
    plan.configure_cce_compaction(kind=kind)
    if kind == "dpo":
        objective = p.resolve_preference_objective("dpo", beta=0.2, reference_free=not reference,
            loss_type=["sigmoid", "ipo"] if reference else ["robust", "hinge"])
        dense = p.make_dpo_loss_fn(objective, reference_policy=policy)
        loss = p.make_dpo_cce_loss_fn(model, objective, reference_policy=policy)
    else:
        objective = p.resolve_preference_objective("orpo", beta=0.2)
        dense, loss = p.make_orpo_loss_fn(objective), p.make_orpo_cce_loss_fn(model, objective)
    grad = nn.value_and_grad(model, loss)
    run = mx.compile(lambda *b: grad(model, *b), inputs=model.state, outputs=model.state)
    for index in range(2):
        batch = plan[index]
        compact = plan.prepare_cce_batch(index, batch)
        # bfloat16 logits lose the logit sums to rounding, so there the uncompacted kernel is the reference.
        eager_rows = len(kernel_rows)
        expected = nn.value_and_grad(model, dense if dtype == "float32" else loss)(model, *batch)
        mx.eval(expected)
        calls.clear()
        del kernel_rows[eager_rows:]
        actual = run(*compact)
        mx.eval(actual)
        assert "model" not in calls
        capacity = compact[3].shape[0]
        assert capacity < batch[0].shape[0] * (batch[0].shape[1] - 1)
        assert kernel_rows == [capacity] * (2 if reference and kind == "dpo" else 1)
        for want, got in zip(expected[0], actual[0]):
            assert mx.allclose(want, got, atol=1e-4, rtol=1e-5).item()
        # A bfloat16 run accumulates chunks in a different order; the softcap derivative amplifies that.
        rtol = 2e-4 if dtype == "float32" else 2e-2 if head == "softcap" else 5e-3
        for (_, want), (_, got) in zip(tree_flatten(expected[1]), tree_flatten(actual[1])):
            assert mx.allclose(want, got, atol=2e-5, rtol=rtol).item()
        if reference:
            assert adapter.scale == 2.0


@metal_only
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("labeled", [False, True])
def test_text_eval_compacts_finite_batches(monkeypatch, quantized, labeled):
    from types import SimpleNamespace
    import numpy as np

    rows = []
    for row, width in enumerate((513, 507, 769)):
        ids = (np.arange(width) * (7 + row) + row) % 2053
        offset = width - 55 - row * 3
        labels = None
        if labeled:
            labels = np.full(width, -100, dtype=np.int32)
            positions = np.arange(offset + row, width, row + 2)
            labels[positions] = (positions * 19 + row) % 8192
            labels[1] = 7
            labels = tuple(labels)
        rows.append(_FiniteTextRow(tuple(ids), offset=offset, labels=labels))
    plan = FiniteTextBatchPlan(rows, [(0, None, 1), (2,), (None, None)],
                              max_seq_length=769, pad_id=0, minimum_width=2)
    mx.random.seed(927)
    model = _cce_text_model(2053, 64, quantized=quantized)
    model.set_dtype(mx.bfloat16)
    original, projected = mlx_utils._get_runtime_cce, []
    def factory(**kwargs):
        runtime = original(**kwargs)
        def record(hidden, *args):
            projected.append(hidden.shape[0])
            return runtime(hidden, *args)
        return record
    monkeypatch.setattr(mlx_utils, "_get_runtime_cce", factory)
    candidate = mlx_utils.make_cce_loss_fn(model)
    def baseline(model, *batch):
        return candidate(model, *batch)
    def fail(failed, _context, error):
        if failed:
            raise error
    trainer = SimpleNamespace(model=model, stop_requested=False,
        _distributed_eval_status=lambda failed=False: (False, failed),
        _raise_distributed_failure_from_any=fail, _fire_prediction_step=lambda: None)
    trainer.args = SimpleNamespace(use_cce=True, streaming=False, max_seq_length=769,
                                  seed=42, dataset_text_field="text", append_eos=False)
    trainer.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=None)
    trainer.formatting_func = None
    trainer.distributed_world = None
    dataset = [{"input_ids": list(row.input_ids), **({"labels": list(row.labels)} if labeled else {})}
               for row in rows]
    prepared = MLXTrainer._create_text_eval_batches(trainer, dataset, 2, False, False)
    assert isinstance(prepared, FiniteTextBatchPlan)
    expected = MLXTrainer._evaluate_batch_totals(trainer, plan, baseline)
    dense_shapes = list(projected)
    projected.clear()
    actual = MLXTrainer._evaluate_batch_totals(trainer, plan, candidate)
    mx.eval(expected[:2], actual[:2])
    assert projected[:2] == [256, 256] and all(n > 256 for n in dense_shapes[:2])
    assert len(projected) == 3 and projected[2] == dense_shapes[2]
    assert expected[2] is None and actual[2] is None
    expected_tokens = sum(
        sum(label != -100 for label in row.labels[row.offset:]) if labeled else
        len(row.input_ids) - row.offset for row in rows
    )
    assert actual[1].item() == expected_tokens
    for want, got in zip(expected[:2], actual[:2]):
        assert mx.allclose(want, got, atol=2e-5, rtol=2e-6).item()


@metal_only
@pytest.mark.parametrize("quantized", [False, True])
def test_preference_logit_sums_keep_float32_precision(quantized):
    from unsloth_zoo.mlx import preference as p

    mx.random.seed(739)
    hidden = mx.random.normal((37, 64)).astype(mx.bfloat16)
    weight = (mx.random.normal((8192, 64)) * 0.5).astype(mx.bfloat16)
    head, quantization = (weight, None, None), {}
    if quantized:
        quantization = dict(group_size=64, bits=4, mode="affine")
        head = mx.quantize(weight, group_size=64, bits=4)
        weight = mx.dequantize(*head, group_size=64, bits=4)
    expected = (hidden.astype(mx.float32) @ weight.astype(mx.float32).T).sum(axis=-1)
    actual = p._head_logit_sums(hidden, *head, quantization, 0.0)
    assert mx.allclose(expected, actual, atol=1e-3, rtol=0).item()


@metal_only
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("kind", ["dpo", "orpo"])
def test_preference_eval_compacts_unequal_batches(monkeypatch, quantized, reference, dtype, kind):
    from types import SimpleNamespace
    from mlx_lm.tuner.lora import LoRAEmbedding
    from unsloth_zoo.mlx import preference as p

    mx.random.seed(738)
    kernel_rows = _record_cce_rows(monkeypatch)
    model = _cce_text_model(2053, 64, quantized=quantized)
    policy = None
    if reference:
        adapter = LoRAEmbedding.from_base(model.model.embed_tokens, r=4, scale=2.0)
        adapter.lora_b = mx.random.normal(adapter.lora_b.shape) * .02
        model.model.embed_tokens = adapter
        policy = p.ReferencePolicy(scales=[(adapter, 0.0)])
    model.set_dtype(getattr(mx, dtype))
    model.eval()
    rows = []
    for i in range(7):
        chosen = tuple((j * 7 + i) % 2053 for j in range(513 if i % 4 == 0 else 45 + i))
        rejected = tuple((j * 11 + i) % 2053 for j in range(507 if i % 4 == 0 else 43 + i))
        rows.append(p.TokenizedPreferenceRow(chosen[:-13-i], chosen[-13-i:],
                    rejected[:-14+i], rejected[-14+i:]))
    plan = p.FinitePreferenceBatchPlan(rows, [(0, 1, 2, 3), (4, 5, 6)],
        normalizers=[(1234, 37, 9)] * 2, cycle_length=2, max_seq_length=513, pad_id=0)
    objective = p.resolve_preference_objective(kind, beta=.2,
        **({"reference_free": not reference} if kind == "dpo" else {}))
    baseline = p.make_preference_eval_fn(objective, reference_policy=policy)
    candidate = p.make_preference_eval_fn(objective, reference_policy=policy, model=model)
    assert not baseline._unsloth_cce_compaction and candidate._unsloth_cce_compaction
    if dtype == "bfloat16":
        # bfloat16 logits lose the logit sums to rounding, so there the uncompacted kernel is the reference.
        baseline = p.make_preference_eval_fn(objective, reference_policy=policy, model=model)
        baseline._unsloth_cce_compaction = False
    calls = []
    def fail(failed, _context, error):
        if failed:
            raise error
    trainer = SimpleNamespace(model=model, stop_requested=False,
        _distributed_eval_status=lambda failed=False: (False, failed),
        _raise_distributed_failure_from_any=fail, _fire_prediction_step=lambda: calls.append(1))
    expected = MLXTrainer._evaluate_batch_totals(trainer, plan, baseline)
    kernel_rows.clear()
    actual = MLXTrainer._evaluate_batch_totals(trainer, plan, candidate)
    mx.eval(expected, actual)
    capacity = 256 if kind == "dpo" else 768
    assert kernel_rows == [capacity] * (4 if kind == "dpo" and reference else 2)
    assert len(calls) == 4 and actual[1].item() == 7
    for want, got in zip(expected, actual):
        assert mx.allclose(want, got, atol=2e-5, rtol=2e-5).item()
    if reference:
        assert adapter.scale == 2.0


@metal_only
@pytest.mark.parametrize("quantized", [False, True])
def test_vlm_cce_compaction_preserves_aligned_rows(monkeypatch, quantized):
    mx.random.seed(735)
    model = _cce_text_model(2053, 64, quantized=quantized)
    def embed_forward(self, ids, inputs_embeds=None):
        return self.embed_tokens(ids) if inputs_embeds is None else inputs_embeds
    monkeypatch.setattr(type(model.model), "__call__", embed_forward)
    model.get_input_embeddings = lambda *args, **kwargs: None
    ids = mx.arange(1026).reshape(2, 513)
    batch = {"input_ids": ids, "labels": mx.where((ids % 11) == 5, ids, -100)}
    loss = mlx_utils.make_vlm_cce_loss_fn(model)
    small_limit = getattr(loss, "_unsloth_cce_small_capacity_limit", 0)
    assert bool(small_limit) == quantized
    plan = mlx_utils.FiniteVLMBatchPlan([], [], None, processor=None, config={}, max_seq_length=1024, image_size=None)
    plan.configure_cce_compaction(small_capacity_limit=small_limit)
    compact = plan.prepare_cce_batch(0, batch)
    assert compact["_unsloth_cce_indices"].shape == (256 if quantized else 512, 2) and "_unsloth_cce_indices" not in batch
    def forward(m, b, **kwargs):
        target = b["labels"][:, 1:-4]
        return m.model.embed_tokens(b["input_ids"])[:, :-5], target, (target != -100).sum()
    monkeypatch.setattr(mlx_utils, "_vlm_cce_forward", forward)
    loss = mlx_utils.make_vlm_cce_loss_fn(model)
    grad = nn.value_and_grad(model, loss)
    run = mx.compile(lambda b: grad(model, b), inputs=model.state, outputs=model.state)
    expected, actual = grad(model, batch), run(compact)
    mx.eval(expected, actual)
    assert actual[0][1].item() == expected[0][1].item()
    assert actual[0][0].item() == pytest.approx(expected[0][0].item(), abs=2e-5)
    for (_, want), (_, got) in zip(tree_flatten(expected[1]), tree_flatten(actual[1])):
        assert mx.allclose(want, got, atol=2e-5, rtol=2e-4).item()


@metal_only
def test_vlm_cce_small_capacity_admission_and_rebuilds():
    import numpy as np
    from unsloth_zoo.mlx.shape_guard import TextShapeGuardReport, TextShapePlan

    plan = mlx_utils.FiniteVLMBatchPlan([], [], None, processor=None, config={}, max_seq_length=2048, image_size=None)
    catalog = frozenset({("full_step", "update", ("vlm",), 1025)})
    report = TextShapeGuardReport("exact", "test", 3, "full_step", 1, 1, 1)
    plan._shape_plan = TextShapePlan(report, catalog, catalog)
    ids = mx.arange(2050).reshape(2, 1025)
    sparse = {"labels": mx.where(ids % 17 == 5, ids, -100)}
    for cap, count, expected in ((2, 2, 1024), (3, 3, 256), (3, 3, 256)):
        result = plan.configure_cce_compaction(max_variants=cap, small_capacity_limit=1024)
        assert result.planned_signatures == count
        prepared = plan.prepare_cce_batch(0, sparse)["_unsloth_cce_indices"]
        assert prepared.shape == (expected, 2)
        selected = np.argwhere(np.asarray(sparse["labels"])[:, 1:] != -100)
        assert mx.array_equal(prepared[:selected.shape[0]], mx.array(selected)).item()
        assert mx.all(prepared[selected.shape[0]:] == -1).item()
    medium = {"labels": mx.where(ids % 5 == 0, ids, -100)}
    assert plan.prepare_cce_batch(0, medium)["_unsloth_cce_indices"].shape == (1024, 2)
    assert plan.prepare_cce_batch(0, sparse)["_unsloth_cce_indices"].shape == (256, 2)
    dense = {"labels": ids}
    assert plan.prepare_cce_batch(0, dense) is dense
    assert plan.prepare_cce_batch(0, sparse) is sparse
    plan.configure_cce_compaction(max_variants=3, small_capacity_limit=512)
    assert plan.prepare_cce_batch(0, sparse)["_unsloth_cce_indices"].shape == (1024, 2)
    result = plan.configure_cce_compaction(max_variants=3)
    assert result.planned_signatures == 2
    assert plan.prepare_cce_batch(0, sparse)["_unsloth_cce_indices"].shape == (1024, 2)


@metal_only
@pytest.mark.parametrize("quantized", [False, True])
def test_vlm_evaluation_compacts_sparse_batches(monkeypatch, quantized):
    from types import SimpleNamespace
    from unsloth_zoo.mlx.trainer import MLXTrainer
    from unsloth_zoo.mlx.cce import runtime_cce

    # The 256-row capacity is admitted only when it shares vocabulary chunks with
    # the half capacity, and that limit is derived from _get_memory_budget(), which
    # is 0.1% of the device's recommended working set. On a 128 GB machine the limit
    # is 16384 and the assertion below holds; on an 8 GB M1, the hosted Apple Silicon
    # runner, the budget is 6 MB, the limit is 768, the 1024-row capacity no longer
    # fits under it and the assertion reads [1024, 1024] == [256, 256]. Pin the budget
    # so this tests the admission rule rather than how much memory the runner has.
    monkeypatch.setattr(runtime_cce, "_CHUNK_BUDGET", 128 * 1024 * 1024)

    mx.random.seed(412)
    model = _cce_text_model(2053, 64, quantized=quantized)
    def embed_forward(self, ids, inputs_embeds=None):
        return self.embed_tokens(ids) if inputs_embeds is None else inputs_embeds
    monkeypatch.setattr(type(model.model), "__call__", embed_forward)
    model.get_input_embeddings = lambda *args, **kwargs: None
    ids = mx.arange(2050).reshape(2, 1025)
    batch = {"input_ids": ids, "labels": mx.where(ids % 17 == 5, ids, -100)}

    def forward(m, b, **kwargs):
        target = b["labels"][:, 1:]
        return (m.model.embed_tokens(b["input_ids"])[:, :-1], target,
                (target != -100).sum())

    monkeypatch.setattr(mlx_utils, "_vlm_cce_forward", forward)
    projected = []
    original = mlx_utils._get_runtime_cce

    def factory(**kwargs):
        runtime = original(**kwargs)

        def record(hidden, *args):
            projected.append(hidden.shape[0])
            return runtime(hidden, *args)

        return record

    monkeypatch.setattr(mlx_utils, "_get_runtime_cce", factory)
    loss_fn = mlx_utils.make_vlm_cce_loss_fn(model)
    assert loss_fn._unsloth_cce_compaction
    steps = []
    trainer = SimpleNamespace(
        model=model, stop_requested=False,
        _distributed_eval_status=lambda failed=False: (False, failed),
        _raise_distributed_failure_from_any=lambda failed, _context, error: None,
        _fire_prediction_step=lambda: steps.append(1),
    )

    def dense_fn(model, batch):
        return loss_fn(model, batch)

    def totals(fn):
        projected.clear()
        # Trainer VLM evaluation batches are an eager list, not a plan.
        result = MLXTrainer._evaluate_batch_totals(trainer, [batch, batch], fn, is_vlm=True)
        mx.eval(result[:2])
        return result, list(projected)

    expected, dense_rows = totals(dense_fn)
    actual, compact_rows = totals(loss_fn)
    # The plan admits half the tokens, or 256 once a quantized head raises the
    # small-capacity limit; either way evaluation must stop projecting all of them.
    assert dense_rows == [2048] * 2
    assert compact_rows == [256 if quantized else 1024] * 2
    assert len(steps) == 4 and actual[1].item() == expected[1].item()
    assert actual[0].item() == pytest.approx(expected[0].item(), rel=2e-5)


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


@metal_only
def test_preference_generate_during_eval_samples_through_the_real_engine(tmp_path):
    """A referenced DPO run samples held-out prompts and keeps training.

    Only a real model reaches the backend's capability probe and streaming
    detokenizer, so this is the one place the wiring runs unmocked.
    """
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXDPOTrainer
    from unsloth_zoo.mlx.utils import iter_mlx_lora_modules

    def pairs(count):
        return [
            {
                "prompt": f"### Question: what is {i} plus {i}?\n### Answer:",
                "chosen": f" {2 * i}.",
                "rejected": f" {2 * i + 3}.",
            }
            for i in range(count)
        ]

    model, tokenizer = FastMLXModel.from_pretrained(MODEL, max_seq_length=256)
    model = FastMLXModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0)
    trainer = MLXDPOTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=pairs(8), eval_dataset=pairs(3),
        args=MLXDPOConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=2,
            warmup_steps=0,
            learning_rate=5e-6,
            logging_steps=1,
            output_dir=str(tmp_path),
            report_to="none",
            max_seq_length=256,
            seed=3407,
            beta=0.1,
            eval_steps=1,
            generate_during_eval=True,
            num_generation_prompts=2,
            generation_max_tokens=8,
        ),
    )
    result = trainer.train()

    samples = trainer.last_generation_samples
    assert len(samples) == 2
    for sample in samples:
        assert sample["prompt"].startswith("### Question:")
        # Batched decoding is not batch-invariant, so only presence is asserted.
        assert isinstance(sample["policy"], str) and sample["policy"]
        assert isinstance(sample["reference"], str) and sample["reference"]

    assert result["train_steps"] == 2, "training continues past the sampling pass"
    assert all(
        module.scale != 0.0 for _, module in iter_mlx_lora_modules(model)
    ), "the reference sample restored every adapter scale"


@metal_only
def test_neftune_noise_is_gated_out_of_the_preference_eval_forward(tmp_path):
    """Evaluation must score the model, not a noised copy of it.

    The gate is the embedding's own training flag, which only a real
    mlx.nn.Module tree propagates, so nothing short of a real run tells a
    working gate from an absent one -- and a broken one moves an eval number
    without failing anything.
    """
    import mlx.core as mx
    from unsloth_zoo.mlx.trainer import MLXORPOConfig, MLXORPOTrainer

    def pairs(start, stop):
        return [{"prompt": f"### Question: {i} plus {i}?\n### Answer:",
                 "chosen": f" {2 * i}.", "rejected": f" {2 * i + 3}."}
                for i in range(start, stop)]

    model, tokenizer = FastMLXModel.from_pretrained(MODEL, max_seq_length=256)
    model = FastMLXModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0)
    trainer = MLXORPOTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=pairs(0, 4), eval_dataset=pairs(4, 6),
        args=MLXORPOConfig(
            per_device_train_batch_size=2, gradient_accumulation_steps=1,
            max_steps=1, learning_rate=1e-5, logging_steps=1,
            output_dir=str(tmp_path), report_to="none", max_seq_length=256,
            seed=3407, eval_steps=1, neftune_noise_alpha=5.0,
        ),
    )

    observed, restored, noised = [], [], []
    evaluating = {"now": False}
    install = trainer._install_neftune

    def install_and_watch():
        install()
        embed = trainer._neftune_emb
        assert embed is not None, "NEFTune never attached to the embedding"
        noisy, base = type(embed), trainer._neftune_base_cls
        # Eager, before any transform traces the forward: without this the run
        # below cannot tell a working gate from an embedding that never noises.
        probe, was_training = mx.array([[1, 2, 3]]), embed.training
        embed.train(True)
        noised.append(
            not mx.allclose(embed(probe), base.__call__(embed, probe)).item())
        embed.train(was_training)

        class _Recorder(noisy):
            def __call__(self, x):
                observed.append((bool(self.training), evaluating["now"]))
                return noisy.__call__(self, x)

        _Recorder.__name__ = noisy.__name__
        embed.__class__ = _Recorder

    trainer._install_neftune = install_and_watch
    evaluate = trainer._evaluate

    def evaluate_and_watch(*args, **kwargs):
        evaluating["now"] = True
        try:
            return evaluate(*args, **kwargs)
        finally:
            evaluating["now"] = False
            restored.append(bool(trainer.model.training))

    trainer._evaluate = evaluate_and_watch
    trainer.train()

    assert noised == [True], "the embedding never noised, so the run proves nothing"
    in_eval = [training for training, evaluated in observed if evaluated]
    assert any(t for t, evaluated in observed if not evaluated), "never trained"
    assert in_eval, "evaluation never reached the embedding"
    assert not any(in_eval), "NEFTune noise leaked into the eval forward"
    assert restored == [True], "_evaluate left the model in eval mode"


@metal_only
def test_every_text_loss_accepts_a_wrapped_model_output():
    """Text-only VLM loads take the text losses, but mlx-vlm wrappers return a
    LanguageModelOutput where mlx_lm models return the array. Every loss that
    feeds a model call to cross-entropy has to accept both."""
    import mlx.core as mx

    from unsloth_zoo.mlx.preference import (
        make_dpo_loss_fn, make_orpo_loss_fn, resolve_preference_objective)
    from unsloth_zoo.mlx.utils import make_baseline_loss_fn

    vocab = 8
    ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    lengths = mx.array([[0, ids.shape[1]]], dtype=mx.int32)
    labels = mx.array([[-100, 2, 3, 4]], dtype=mx.int32)
    logits = mx.random.normal((1, ids.shape[1] - 1, vocab))

    class _LanguageModelOutput:
        def __init__(self, logits):
            self.logits = logits

    def bare(_inputs):
        return logits

    def wrapped(_inputs):
        return _LanguageModelOutput(logits)

    baseline = make_baseline_loss_fn()
    orpo = make_orpo_loss_fn(resolve_preference_objective("orpo", beta=0.1))
    dpo = make_dpo_loss_fn(resolve_preference_objective(
        "dpo", beta=0.1, reference_free=True))
    # Chosen and rejected rows differ, so an unwrap that reordered the batch
    # axis would reverse the preference signal instead of comparing equal.
    rejected_ids = mx.array([[1, 5, 6, 7]], dtype=mx.int32)
    pair = mx.concatenate([ids, rejected_ids], axis=0)
    pair_lengths = mx.concatenate([lengths, lengths], axis=0)
    pair_logits = mx.concatenate([logits, mx.random.normal(logits.shape)], axis=0)
    # (supervised tokens, pairs, microbatches) for the window normalizers.
    norms = (mx.array(3), mx.array(1), mx.array(1))

    def pair_bare(_inputs):
        return pair_logits

    def pair_wrapped(_inputs):
        return _LanguageModelOutput(pair_logits)

    for name, call, models in (
        ("sft", lambda m: baseline(m, ids, lengths), (bare, wrapped)),
        ("sft-labels", lambda m: baseline(m, ids, lengths, labels), (bare, wrapped)),
        ("orpo", lambda m: orpo(m, pair, pair_lengths, norms), (pair_bare, pair_wrapped)),
        ("dpo", lambda m: dpo(m, pair, pair_lengths, norms), (pair_bare, pair_wrapped)),
    ):
        from_bare = call(models[0])[0]
        from_wrapped = call(models[1])[0]
        assert mx.allclose(from_bare, from_wrapped), name

    # The comparisons above only pin the two forms to each other. This pins the
    # supervised one to a value computed outside the loss.
    expected = nn.losses.cross_entropy(logits, ids[:, 1:]).mean()
    assert mx.allclose(baseline(wrapped, ids, lengths)[0], expected)


@metal_only
def test_cce_loss_precomputes_the_hidden_gradient_only_in_training(monkeypatch):
    from unsloth_zoo.mlx.cce import runtime_cce
    from unsloth_zoo.mlx.utils import make_cce_loss_fn

    model, tokenizer = FastMLXModel.from_pretrained(MODEL, max_seq_length=256)
    ids = mx.array([tokenizer.encode("the capital of France is Paris")])
    lengths = mx.array([[0, ids.shape[1]]], dtype=mx.int32)
    calls, forward = [], runtime_cce._forward_with_hidden_gradient
    monkeypatch.setattr(
        runtime_cce, "_forward_with_hidden_gradient",
        lambda *args, **kwargs: (calls.append(model.training), forward(*args, **kwargs))[1],
    )
    loss_fn = make_cce_loss_fn(model)
    losses = []
    for training in (True, False):
        model.train(training)
        losses.append(loss_fn(model, ids, lengths)[0])
    mx.eval(losses)
    assert calls == [True]
    assert mx.array_equal(*losses).item()


@metal_only
@pytest.mark.parametrize("softcap", (0.0, 20.0), ids=("no-softcap", "softcap"))
def test_post_head_multiplier_reaches_the_fused_cce_loss(softcap):
    """A model whose forward scales logits after the head must have that scale
    reproduced by fused CCE, which rebuilds the logits itself.

    Guards the silent failure mode: without the multiply, CCE softcaps logits
    that are far too large, so the loss stays plausible while the gradients it
    produces come from a saturated tanh. The softcap case is the composition
    that failure needs, so it is exercised too.
    """
    from unsloth_zoo.mlx.utils import _get_text_model, make_cce_loss_fn

    model, tokenizer = FastMLXModel.from_pretrained(MODEL, max_seq_length=256)
    ids = mx.array([tokenizer.encode("the capital of France is Paris")])
    # (start, end) per row: covers every shifted target position.
    lengths = mx.array([[0, ids.shape[1]]], dtype=mx.int32)

    # The knob is read off the resolved text model, which is not `model.model`.
    text_model = _get_text_model(model)
    multiplier = 0.5
    if softcap:
        text_model.final_logit_softcapping = softcap

    try:
        unscaled, _ = make_cce_loss_fn(model)(model, ids, lengths)
        text_model.output_multiplier = multiplier
        try:
            scaled, _ = make_cce_loss_fn(model)(model, ids, lengths)
        finally:
            del text_model.output_multiplier
    finally:
        if softcap:
            del text_model.final_logit_softcapping

    logits = model(ids[:, :-1]).astype(mx.float32) * multiplier
    if softcap:
        logits = mx.tanh(logits / softcap) * softcap
    reference = float(
        nn.losses.cross_entropy(logits, ids[:, 1:], reduction="mean")
    )

    assert float(scaled) == pytest.approx(reference, rel=2e-2)
    assert abs(float(scaled) - float(unscaled)) > 1e-3


def _gather_qmm_call(k=96, rows=64, m=1, out_width=64, pack_bits=4, **overrides):
    """One sorted call; an untransposed one is judged on `out_width`."""
    kw = {"rhs_indices": _U(rows), "transpose": True, "sorted_indices": True,
          "bits": 4, **overrides}
    w = ((8, 64, k * pack_bits // 32) if kw["transpose"]
         else (8, k, out_width * pack_bits // 32))
    return mx.zeros((rows, m, k), dtype=mx.bfloat16), _U(*w), kw


@pytest.fixture
def raw_gather_qmm(monkeypatch):
    raw = mlx_utils._MLX_GATHER_QMM_ORIGINAL or mx.gather_qmm
    monkeypatch.setattr(mx, "gather_qmm", raw)
    monkeypatch.setattr(mlx_utils, "_MLX_GATHER_QMM_ORIGINAL", raw)
    monkeypatch.setattr(mlx_utils, "_MLX_GATHER_QMM_CANARIES", {})
    return raw


@metal_only
@pytest.mark.parametrize("case, expected", [
    (dict(k=96), (_K_REM,)),
    (dict(k=64, rows=32769), (_ROW_OVF,)),
    (dict(k=96, rows=32769), (_K_REM, _ROW_OVF)),
    (dict(k=96, sorted_indices=False), ()),
    (dict(k=96, lhs_indices=_U(64, 1)), ()),
    (dict(k=96, rhs_indices=None, lhs_indices=_U(64, 1)), ()),
    (dict(k=96, m=2), ()),
    (dict(k=96, stream=_CPU), ()),
    (dict(k=192, out_width=96, transpose=False), (_K_REM,)),
    (dict(k=192, out_width=96, pack_bits=8, bits=None, mode="mxfp8", transpose=False),
     (_K_REM,)),
    ({**_UNTRANSPOSED, "rows": 32769, "rhs_indices": _U(1)}, (_ROW_OVF,)),
    ({**_UNTRANSPOSED, "lhs_indices": _U(4097, 1)}, (_ROW_OVF,)),
    ({**_UNTRANSPOSED, "lhs_indices": _U(4096, 8)}, ()),
])
def test_gather_qmm_guard_predicate(case, expected):
    """4097 lhs indices over 8 experts broadcast to 32776 rows; 4096 x 8 is 32768."""
    assert (mlx_utils._MLX_MAX_SORTED_ROWS, mlx_utils._MLX_K_REMAINDER) == (32768, _K_REM)
    x, w, kw = _gather_qmm_call(**case)
    assert mlx_utils._gather_qmm_conditions(x, w, (), kw) == expected


@metal_only
@pytest.mark.parametrize("mode, group_size",
                         [("affine", 32), ("mxfp4", 32), ("mxfp8", 32), ("nvfp4", 16)])
def test_gather_qmm_canary_probes_the_real_kernel(mode, group_size, monkeypatch, raw_gather_qmm):
    """Magnitude alone is no oracle: the unsorted path measures just as small."""
    probe_k, rows = mlx_utils._gather_qmm_probe_k, mlx_utils._MLX_PROBE_ROWS
    k, row_k = probe_k(_K_REM, group_size), probe_k(_ROW_OVF, group_size)
    assert k % 64 and k % group_size == 0 and k >= 96 and probe_k(_K_REM, 64) is None
    assert row_k % 64 == 0 and rows[_ROW_OVF] % 2 and rows[_ROW_OVF] > 32768
    seen = {}
    monkeypatch.setattr(mlx_utils, "_MLX_GATHER_QMM_ORIGINAL",
                        lambda *a, **kw: seen.update(kw) or raw_gather_qmm(*a, **kw))
    dev = mx.default_device()
    error = mlx_utils._gather_qmm_canary_error(_K_REM, group_size, None, mode, dev)
    assert (seen["mode"], seen["bits"], seen["sorted_indices"]) == (mode, None, True)
    assert mx.array_equal(seen["rhs_indices"], mx.sort(seen["rhs_indices"])).item()
    assert (error < 0.25 or error > 4.0) and mlx_utils._MLX_CANARY_ERROR_LIMIT == 1.0
    monkeypatch.setattr(mlx_utils, "_gather_qmm_canary_error", lambda *a: float("nan"))
    assert mlx_utils._gather_qmm_canary_defective(_K_REM, group_size, None, mode, dev)
    assert [*mlx_utils._MLX_GATHER_QMM_CANARIES] == [(_K_REM, group_size, None, mode, str(dev))]


@metal_only
@pytest.mark.parametrize("defective", [False, True])
def test_gather_qmm_reroutes_only_when_defective(defective, monkeypatch, raw_gather_qmm):
    """Injects the verdict, not corruption."""
    seen = {}
    monkeypatch.setattr(mlx_utils, "_MLX_GATHER_QMM_ORIGINAL", lambda *a, **kw: seen.update(kw))
    monkeypatch.setattr(mlx_utils, "_gather_qmm_canary_error",
                        lambda c, *a: 99.0 if defective and c == _ROW_OVF else 0.0)
    scales = mx.zeros((8, 64, 3), dtype=mx.bfloat16)
    x, w, kw = _gather_qmm_call(k=96, rows=32769)
    mlx_utils._gather_qmm_guarded(x, w, scales, None, group_size=32, **kw)
    assert seen["sorted_indices"] is not defective
    monkeypatch.setattr(mlx_utils, "_gather_qmm_quantization", lambda *a: pytest.fail("probed"))
    x, w, kw = _gather_qmm_call(k=64)
    mlx_utils._gather_qmm_guarded(x, w, scales, None, group_size=32, **kw)
    assert seen["sorted_indices"] is True


@metal_only
def test_gather_qmm_guard_install(monkeypatch, raw_gather_qmm):
    """The guard must end up beneath the index-stop wrapper, even mid-training."""
    monkeypatch.setenv("UNSLOTH_MLX_GATHER_QMM_GUARD", "0")
    assert mlx_utils.apply_gather_qmm_nax_guard() is False
    monkeypatch.delenv("UNSLOTH_MLX_GATHER_QMM_GUARD")
    mlx_utils.acquire_mlx_training_patches()
    try:
        assert mlx_utils.apply_gather_qmm_nax_guard() is True
        assert mlx_utils.apply_gather_qmm_nax_guard() is False
        assert mx.gather_qmm._unsloth_index_original._unsloth_gather_qmm_guard
    finally:
        mlx_utils.release_mlx_training_patches()
    assert mx.gather_qmm._unsloth_gather_qmm_guard
    import inspect   # the only production install site
    assert "apply_gather_qmm_nax_guard()" in inspect.getsource(FastMLXModel.from_pretrained)


@metal_only
@pytest.mark.parametrize("failing", ["_gather_qmm_conditions",
                                     "_gather_qmm_quantization",
                                     "_gather_qmm_target_device"])
def test_gather_qmm_guard_never_breaks_a_working_call(failing, monkeypatch,
                                                      raw_gather_qmm):
    """Not worth failing a call: the mlx that raises here predates the NAX kernels."""
    monkeypatch.setattr(mlx_utils, "_MLX_GATHER_QMM_UNREADABLE", False)
    seen = {}
    monkeypatch.setattr(mlx_utils, "_MLX_GATHER_QMM_ORIGINAL",
                        lambda *a, **kw: seen.update(kw) or "result")

    def boom(*args, **kwargs):
        raise TypeError("quantize(): incompatible function arguments")

    monkeypatch.setattr(mlx_utils, failing, boom)
    x, w, kw = _gather_qmm_call(k=96)
    assert mlx_utils._gather_qmm_guarded(
        x, w, mx.zeros((8, 64, 3), dtype=mx.bfloat16), None, group_size=32,
        **kw) == "result"
    assert seen["sorted_indices"] is True
    assert mlx_utils._MLX_GATHER_QMM_UNREADABLE is True   # warns once, not per call


@metal_only
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"], ids=["bf16", "fp16"])
def test_the_logit_sum_holds_the_widest_vocabulary_in_float16(dtype):
    """float16 stops at 65504, far below the 1e9 a wide row of logits sums to."""
    from unsloth_zoo.mlx.preference import _row_logit_sum

    wide = mx.full((1, 2, 262144), 4000.0, dtype=getattr(mx, dtype))
    rows = _row_logit_sum(wide)
    assert bool(mx.all(mx.isfinite(rows))), "the scale does not cover 256K"
    assert float(rows[0, 0]) == pytest.approx(262144 * 4000.0, rel=1e-3)


@metal_only
@pytest.mark.parametrize("targets,attention", [(["lm_head"], True),
                                               (["q_proj", "lm_head"], False)])
def test_bitlinear_can_feed_a_downstream_head_adapter(targets, attention):
    from test_mlx_lora_group_selection import HIDDEN, VOCAB, _adapters, _peft
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.bitnet import Model, ModelArgs
    model = Model(ModelArgs(
        model_type="bitnet", hidden_size=HIDDEN, intermediate_size=HIDDEN * 2,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        rms_norm_eps=1e-5, vocab_size=VOCAB, tie_word_embeddings=False,
    ))
    _peft(model, target_modules=targets, finetune_attention_modules=attention)
    _, grads = nn.value_and_grad(model, lambda m: m(mx.array([[1, 2]])).sum())(model)
    assert _adapters(model) == ["lm_head"]
    assert mx.abs(grads["lm_head"]["lora_b"]).max().item() > 0


@metal_only
def test_checkpointed_kv_shared_layers_hold_what_unshared_layers_hold():
    """Borrowed K/V's cotangent is consumed by the source layer's backward; left
    to run by that consumer, each borrowing layer keeps its T x T attention
    cotangents alive until then."""
    from mlx_vlm.models.gemma4.config import TextConfig
    from mlx_vlm.models.gemma4.language import DecoderLayer, Gemma4TextModel

    seq_len = 512
    ids = mx.random.randint(0, 64, (1, seq_len))

    def build(num_kv_shared_layers):
        mx.random.seed(0)
        model = Gemma4TextModel(TextConfig(
            hidden_size=64, num_hidden_layers=15, intermediate_size=128,
            num_attention_heads=8, head_dim=16, global_head_dim=32,
            vocab_size=64, vocab_size_per_layer_input=64,
            hidden_size_per_layer_input=8, sliding_window=seq_len // 4,
            num_kv_shared_layers=num_kv_shared_layers))
        mx.eval(model.parameters())
        return model

    def peak_and_grads(model, repeats = 3):
        """Peak above resident, taken as the MINIMUM over `repeats` measurements.

        A peak is a maximum over a run, so noise can only push it up: allocator state
        left by whatever ran before it, a cache the runtime had not reclaimed yet. The
        floor is the algorithmic requirement, and the floor is what the ratio below is
        about, so the minimum of a few readings estimates it where a single reading does
        not. One reading is what made this flake: run 35502075009 on main reported
        shared 197663512 against unshared 151210728, a ratio of 1.307 over the 1.25
        bound, and the same commit with the same pins passed on re-run.

        The bound itself is unchanged. This makes the measurement less noisy rather than
        the claim weaker.
        """
        loss_and_grad = nn.value_and_grad(model, lambda m: (m(ids) ** 2).sum())
        mx.eval(loss_and_grad(model))
        floor, grads = None, None
        for _ in range(repeats):
            # Drop the previous repeat's gradients BEFORE `resident` is sampled. Holding
            # them would put a tree in `resident` that the assignment below releases
            # before the measured run, so the subtraction would remove memory that was
            # not resident during it and the repeat would read low. A floor made of
            # readings like that is lower than the algorithmic floor, which is the one
            # direction this must not move in.
            grads = None
            mx.clear_cache()
            resident = mx.get_active_memory()
            mx.reset_peak_memory()
            grads = loss_and_grad(model)[1]
            mx.eval(grads)
            peak = mx.get_peak_memory() - resident
            floor = peak if floor is None else min(floor, peak)
        return floor, grads

    shared, unshared = build(10), build(0)
    _, reference = peak_and_grads(shared, repeats = 1)
    mlx_utils._patch_layer_class_for_gc(DecoderLayer)
    try:
        unshared_peak, _ = peak_and_grads(unshared)
        shared_peak, grads = peak_and_grads(shared)
    finally:
        mlx_utils._unpatch_layer_class_gc(DecoderLayer)
    assert shared_peak < 1.25 * unshared_peak, (shared_peak, unshared_peak)
    for (name, got), (_, want) in zip(tree_flatten(grads), tree_flatten(reference)):
        assert mx.allclose(got, want, rtol=1e-5, atol=1e-7).item(), name


@metal_only
def test_checkpointing_keeps_cacheless_kv_shared_gradients():
    from types import SimpleNamespace
    from mlx_vlm.models.gemma3n.config import TextConfig
    from mlx_vlm.models.gemma3n.language import Gemma3Model, Gemma3nDecoderLayer
    from unsloth_zoo.mlx.loader import _fix_gemma4_kv_sharing

    mx.random.seed(0)
    backbone = Gemma3Model(TextConfig(
        model_type="gemma3n_text", hidden_size=64, num_hidden_layers=6,
        intermediate_size=[128] * 6, activation_sparsity_pattern=[0.0] * 6,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        vocab_size=64, vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8, laurel_rank=4, sliding_window=16,
        num_kv_shared_layers=2,
        layer_types=["sliding_attention", "full_attention"] * 3))
    mx.eval(backbone.parameters())
    ids = mx.random.randint(0, 64, (1, 32))
    loss_and_grad = nn.value_and_grad(backbone, lambda m: (m(ids) ** 2).sum())

    original_call = Gemma3Model.__call__
    _fix_gemma4_kv_sharing(SimpleNamespace(language_model=SimpleNamespace(model=backbone)))
    # Seven early returns in the shim leave the class unpatched, and an unpatched
    # class takes the plain checkpoint branch, which is the state this test exists
    # to reject. Without this the test would pass having proved nothing.
    assert "_kv_sharing_patched" in Gemma3Model.__dict__, "the shim did not install"
    try:
        reference = loss_and_grad(backbone)[1]
        mlx_utils._patch_layer_class_for_gc(Gemma3nDecoderLayer)
        try:
            grads = loss_and_grad(backbone)[1]
        finally:
            mlx_utils._unpatch_layer_class_gc(Gemma3nDecoderLayer)
    finally:
        Gemma3Model.__call__ = original_call
        # Delete only a flag this test set. `del` on an absent attribute raises
        # from the finally and buries the real failure behind an AttributeError,
        # and `hasattr` would consume a flag inherited from an earlier patch,
        # disarming the shim's own idempotence guard for whatever runs next.
        if "_kv_sharing_patched" in Gemma3Model.__dict__:
            del Gemma3Model._kv_sharing_patched
    for (name, got), (_, want) in zip(tree_flatten(grads), tree_flatten(reference)):
        assert mx.allclose(got, want, rtol=1e-5, atol=1e-7).item(), name


@metal_only
def test_cce_hidden_forward_preserves_wrapper_embeddings_and_image_mask():
    from types import SimpleNamespace

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(32, 64)

        def __call__(self, ids, inputs_embeds=None, per_layer_inputs=None, image_mask=None):
            h = self.embed_tokens(ids) * 8 if inputs_embeds is None else inputs_embeds
            if per_layer_inputs is not None:
                h = h + per_layer_inputs
            if image_mask is not None:
                h = h * mx.where(image_mask[..., None], 2, 1)
            return h

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Module()
            self.language_model.model = Backbone()
            self._unsloth_text_only_vlm = True

        def get_input_embeddings(self, ids, pixels):
            h = self.language_model.model.embed_tokens(ids)
            return SimpleNamespace(inputs_embeds=h, per_layer_inputs=h * 0.3)

    model = Wrapper()
    for rows in ([[2, 3, 4], [5, 6, 7]], [[7, 6, 5], [4, 3, 2]]):
        ids = mx.array(rows)
        features = model.get_input_embeddings(ids, None)
        mask = ids % 2 == 0
        expected = model.language_model.model(ids, inputs_embeds=features.inputs_embeds,
            per_layer_inputs=features.per_layer_inputs, image_mask=mask)
        actual = mlx_utils._forward_text_hidden_states(model, ids, visual_pos_masks=mask)
        assert mx.array_equal(actual, expected).item()
        inverse = mlx_utils._forward_text_hidden_states(model, ids, visual_pos_masks=~mask)
        assert not mx.array_equal(inverse, expected).item()


@metal_only
@pytest.mark.parametrize("media_key", ["pixel_values", "pixel_values_videos"])
def test_cce_embedding_signature_uses_ids_and_preserves_media(capsys, monkeypatch, media_key):
    calls = []
    factory = mlx_utils._get_runtime_cce
    def counted_factory(**kwargs):
        runtime = factory(**kwargs)
        def run(*args):
            calls.append(True)
            return runtime(*args)
        return run
    monkeypatch.setattr(mlx_utils, "_get_runtime_cce", counted_factory)
    base = _cce_text_model(32, 64, quantized=False)
    class Model(type(base)):
        def __init__(self):
            super().__init__()
            self.model.encoder = nn.Identity()
        def __call__(self, ids, pixel_values=None, **kwargs):
            pixel_values = kwargs.get("pixel_values_videos", pixel_values)
            h = self.model(ids)
            return self.lm_head(h if pixel_values is None else h + pixel_values)
        def get_input_embeddings(self, *args, **kwargs):
            pytest.fail("token-ID CCE must not request merged embeddings")
    model = Model()
    assert mlx_utils._get_backbone_embed_kwarg(model.model) is None
    loss = mlx_utils.make_vlm_cce_loss_fn(model)
    assert loss._unsloth_cce_backend == "runtime-cce"
    baseline = mlx_utils.make_vlm_baseline_loss_fn(model)
    for pixels in (None, mx.ones((2, 3, 64)), None):
        batch = dict(input_ids=mx.array([[2, 3, 4], [7, 6, 5]]), **{media_key: pixels})
        before = len(calls)
        actual, ntoks = loss(model, batch)
        expected, expected_ntoks = baseline(model, batch)
        assert mx.allclose(actual, expected, atol=1e-5).item()
        assert ntoks.item() == expected_ntoks.item() == 4
        assert len(calls) - before == int(pixels is None)
    assert capsys.readouterr().out.count("cannot accept multimodal embeddings") == 1

    class Alias:
        def __call__(self, ids, input_embeddings=None):
            pass
    class Unknown:
        def __call__(self, ids, **kwargs):
            pass
    class Positional:
        def __call__(self, ids, inputs_embeds=None, /):
            pass
    assert mlx_utils._get_backbone_embed_kwarg(Alias()) == "input_embeddings"
    assert mlx_utils._get_backbone_embed_kwarg(Unknown()) is None
    assert mlx_utils._get_backbone_embed_kwarg(Positional()) is None


@metal_only
@pytest.mark.parametrize("conditioning", ["self_conditioning_logits", "self_conditioning_embeddings"])
def test_encoder_decoder_cce_matches_full_forward_and_gradients(conditioning, request):
    mlx_utils.acquire_mlx_training_patches()
    request.addfinalizer(mlx_utils.release_mlx_training_patches)
    config = pytest.importorskip("mlx_vlm.models.diffusion_gemma.config")
    module = pytest.importorskip("mlx_vlm.models.diffusion_gemma.diffusion_gemma")
    model = module.Model(config.ModelConfig(canvas_length=4, text_config=config.TextConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        num_global_key_value_heads=1, head_dim=32, global_head_dim=32,
        num_experts=2, top_k_experts=1, final_logit_softcapping=0.5,
        layer_types=["sliding_attention", "full_attention"],
    )))
    cce = mlx_utils.make_vlm_cce_loss_fn(model, ignore_token_ids=[])
    assert cce._unsloth_cce_backend == "runtime-cce"
    ce = mlx_utils.make_vlm_baseline_loss_fn(model, ignore_token_ids=[])
    for rows in ([[2, 3, 4, 5], [6, 7, 8, 9]], [[9, 8, 7, 6], [5, 4, 3, 2]]):
        ids = mx.array(rows)
        batch = dict(input_ids=ids, labels=mx.where(ids % 3 == 0, -100, ids),
                     attention_mask=mx.ones_like(ids), canvas_ids=ids[:, ::-1],
                     decoder_attention_mask=mx.ones((2, 8), dtype=mx.bool_))
        batch[conditioning] = mx.random.normal((2, 4, 64)) * 0.1
        (expected, n1), g1 = nn.value_and_grad(model, ce)(model, batch)
        (actual, n2), g2 = nn.value_and_grad(model, cce)(model, batch)
        assert n1.item() == n2.item()
        assert mx.allclose(actual, expected, atol=1e-5).item()
        for (name, a), (_, b) in zip(tree_flatten(g1), tree_flatten(g2)):
            assert mx.allclose(a, b, atol=2e-5, rtol=2e-3).item(), name


@metal_only
def test_training_refuses_batch_axis_vocabulary_mask():
    class Unsafe(nn.Module):
        def __init__(self):
            super().__init__()
            self._dummy_tokenizer_ids = mx.array([3, 7])

        def __call__(self, logits):
            logits[self._dummy_tokenizer_ids] = -float("inf")
            return logits

    class Corrected(Unsafe):
        def __call__(self, logits):
            logits[..., self._dummy_tokenizer_ids] = -float("inf")
            return logits

    with pytest.raises(ValueError, match="unsafe output token mask.*batch axis"):
        mlx_utils._validate_output_token_mask(Unsafe())
    mlx_utils._validate_output_token_mask(Corrected())
    empty = Unsafe()
    empty._dummy_tokenizer_ids = mx.array([], mx.int32)
    mlx_utils._validate_output_token_mask(empty)
    logits = Corrected()(mx.zeros((2, 3, 8)))
    assert mx.isneginf(logits[..., 3]).all().item()
    assert mx.isneginf(logits[..., 7]).all().item()
    assert (logits[..., 2] == 0).all().item()


def _ragged_compact_prefix_rows(features, valid_mask):
    """mlx-vlm 0.7.1's `_compact_prefix_rows`, verbatim."""
    rows = []
    for batch_idx, row in enumerate(valid_mask.tolist()):
        length = sum(bool(v) for v in row)
        if length:
            rows.append(features[batch_idx, :length])
    if not rows:
        return features.reshape(-1, features.shape[-1])[:0]
    return mx.concatenate(rows, axis=0)


def _padded_features(counts, width=8, dim=6):
    mx.random.seed(0)
    features = mx.random.normal((len(counts), width, dim))
    valid = mx.stack([mx.arange(width) < n for n in counts])
    return features, valid


@metal_only
@pytest.mark.parametrize("counts", [
    (5,), (8,), (0,), (5, 3), (3, 5), (8, 4, 1), (1, 4, 8), (5, 0, 6), (0, 7), (7, 0), (0, 0), (8, 8),
])
def test_gemma4_unified_reorder_matches_the_ragged_prefix(counts):
    from unsloth_zoo.mlx.compile import _static_shape_prefix_rows

    features, valid = _padded_features(counts)
    reference = _ragged_compact_prefix_rows(features, valid)
    reordered = _static_shape_prefix_rows(features, valid)

    n_valid = int(reference.shape[0])
    assert n_valid == sum(counts)
    assert reordered.shape == (features.shape[0] * features.shape[1], features.shape[-1])
    if n_valid:
        assert mx.array_equal(reference, reordered[:n_valid])


@metal_only
def test_gemma4_unified_reorder_carries_every_valid_row():
    from unsloth_zoo.mlx.compile import _static_shape_prefix_rows

    counts = (2, 7, 4)
    features, valid = _padded_features(counts)
    reordered = _static_shape_prefix_rows(features, valid)
    expected = mx.concatenate(
        [features[row, :n] for row, n in enumerate(counts)], axis=0)
    for index in range(sum(counts)):
        assert mx.array_equal(reordered[index], expected[index]), f"row {index}"


@metal_only
def test_gemma4_unified_reorder_moves_padding_past_the_prefix():
    from unsloth_zoo.mlx.compile import _static_shape_prefix_rows

    counts = (2, 7, 4)
    features, valid = _padded_features(counts)
    reordered = _static_shape_prefix_rows(features, valid)
    expected = mx.concatenate(
        [features[row, n:] for row, n in enumerate(counts)], axis=0)
    assert mx.array_equal(reordered[sum(counts):], expected)


@metal_only
def test_gemma4_unified_reorder_scatters_the_same_embeddings():
    from mlx_vlm.models.gemma4.gemma4 import masked_scatter
    from unsloth_zoo.mlx.compile import _static_shape_prefix_rows

    counts = (5, 3)
    features, valid = _padded_features(counts)
    total = sum(counts)
    embeds = mx.zeros((1, total + 4, features.shape[-1]))
    placeholder = mx.arange(total + 4) < total
    mask = mx.broadcast_to(mx.expand_dims(mx.expand_dims(placeholder, 0), -1), embeds.shape)

    reference = masked_scatter(embeds, mask, _ragged_compact_prefix_rows(features, valid))
    reordered = masked_scatter(embeds, mask, _static_shape_prefix_rows(features, valid))
    assert mx.array_equal(reference, reordered)


@metal_only
@pytest.mark.parametrize("counts", [(3, 5), (8, 0, 4), (0, 0), (8, 8)])
def test_gemma4_unified_sort_key_never_ties(counts):
    from unsloth_zoo.mlx.compile import _valid_first_sort_key

    _, valid = _padded_features(counts)
    rows = valid.shape[0] * valid.shape[1]
    keys = sorted(_valid_first_sort_key(valid.reshape(rows), rows).tolist())
    assert len(set(keys)) == rows, f"tied keys: {keys}"

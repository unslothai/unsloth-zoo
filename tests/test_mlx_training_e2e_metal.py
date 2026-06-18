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

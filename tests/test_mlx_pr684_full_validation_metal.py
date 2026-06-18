"""Deep behavioral validation of the PR 684 MLX trainer rework on Apple Silicon.

Complements test_mlx_training_e2e_metal.py (basic text LoRA smoke) with the
paths that only real Metal training can prove:

1. resume_from_checkpoint determinism: a stop+resume run reproduces the
   fresh run's losses step for step (validates the #751 resume logic inside
   the reworked training loop: optimizer state restore, batch fast-forward,
   LR schedule offset).
2. train_on_responses_only completion-only training: exact step count for
   epoch-based runs (pins the epoch double-counting fix) and finite losses
   through the labeled-batch path.
3. Epoch-based unlabeled runs: num_train_epochs drives the step count when
   max_steps is disabled.
4. SGD with gradient-coupled weight decay end to end.
5. Real VLM LoRA training: tiny 4-bit SmolVLM through the VLM collation,
   label masking, CCE loss, and adapter save pipeline.
"""

import gc
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
    print("NOTICE: Metal unavailable; PR 684 full-validation tests will be skipped.")

metal_only = pytest.mark.skipif(not _METAL, reason="requires Apple Silicon Metal")

TEXT_MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"
# Qwen2-VL: smallest VLM whose processor resolves cleanly under current
# transformers (the mlx-community SmolVLM-256M repo ships a preprocessor
# config AutoImageProcessor cannot map).
VLM_MODEL = "mlx-community/Qwen2-VL-2B-Instruct-4bit"


def _chat_dataset(n=12):
    # ChatML matches SmolLM-Instruct's template so response masking can
    # anchor on the literal role markers.
    return [
        {
            "text": (
                f"<|im_start|>user\nWhat is {i} plus {i}?<|im_end|>\n"
                f"<|im_start|>assistant\nThe answer is {2 * i}.<|im_end|>\n"
            )
        }
        for i in range(n)
    ]


def _make_text_trainer(tmp_path, dataset, **config_overrides):
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    model, tokenizer = FastMLXModel.from_pretrained(TEXT_MODEL, max_seq_length=256)
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
    return MLXTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=MLXTrainingConfig(**config),
    )


def _assert_finite(hist):
    assert all(
        isinstance(l, float) and l == l and abs(l) != float("inf") for l in hist
    ), f"non-finite losses: {hist}"


@metal_only
def test_resume_from_checkpoint_matches_fresh_run(tmp_path):
    """Stop+resume reproduces the fresh run's losses step for step."""
    fresh_dir = tmp_path / "fresh"
    resume_dir = tmp_path / "resume"

    trainer = _make_text_trainer(
        fresh_dir, _chat_dataset(), max_steps=6, save_steps=3,
    )
    trainer.train()
    fresh_hist = list(trainer._train_loss_history)
    assert len(fresh_hist) == 6, f"fresh run logged {len(fresh_hist)} losses"
    _assert_finite(fresh_hist)

    ckpt = str(fresh_dir / "checkpoint-3")
    assert os.path.isfile(os.path.join(ckpt, "adapters.safetensors"))
    assert os.path.isfile(os.path.join(ckpt, "optimizer_state.safetensors"))
    assert os.path.isfile(os.path.join(ckpt, "trainer_state.json"))
    with open(os.path.join(ckpt, "trainer_state.json")) as f:
        saved_state = json.load(f)
    assert saved_state["global_step"] == 3

    # Fresh process state: new base model, same seeds, resume from step 3.
    resumed = _make_text_trainer(
        resume_dir, _chat_dataset(), max_steps=6, save_steps=0,
    )
    resumed.train(resume_from_checkpoint=ckpt)
    resumed_hist = list(resumed._train_loss_history)
    assert len(resumed_hist) == 6, f"resumed run logged {len(resumed_hist)} losses"
    _assert_finite(resumed_hist)

    # Restored prefix is the checkpointed history; post-resume steps must
    # track the fresh run. Same seeds + restored Adam moments mean the only
    # tolerated difference is float accumulation noise.
    for i, (a, b) in enumerate(zip(fresh_hist, resumed_hist), start=1):
        assert abs(a - b) <= 1e-5 * max(1.0, abs(a)), (
            f"step {i}: fresh={a!r} resumed={b!r}\n"
            f"fresh={fresh_hist}\nresumed={resumed_hist}"
        )


@metal_only
def test_train_on_responses_only_epoch_step_count(tmp_path):
    """Completion-only 3-epoch run executes exactly 3 epochs of steps."""
    from unsloth_zoo.mlx.trainer import train_on_responses_only

    trainer = _make_text_trainer(
        tmp_path, _chat_dataset(12), max_steps=0, num_train_epochs=3,
    )
    train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    trainer.train()
    hist = trainer._train_loss_history
    # 12 samples / bs 2 = 6 batches per epoch, 3 epochs = 18 steps. The
    # pre-fix epoch double-count produced 54.
    assert len(hist) == 18, f"expected 18 steps, got {len(hist)}"
    _assert_finite(hist)


@metal_only
def test_epoch_based_unlabeled_step_count(tmp_path):
    """num_train_epochs drives total steps when max_steps is disabled."""
    trainer = _make_text_trainer(
        tmp_path, _chat_dataset(12), max_steps=0, num_train_epochs=2,
    )
    trainer.train()
    hist = trainer._train_loss_history
    assert len(hist) == 12, f"expected 12 steps (6 batches x 2 epochs), got {len(hist)}"
    _assert_finite(hist)


@metal_only
def test_sgd_coupled_weight_decay_e2e(tmp_path):
    """SGD path trains with momentum and gradient-coupled weight decay."""
    trainer = _make_text_trainer(
        tmp_path, _chat_dataset(), max_steps=4,
        optim="sgd", weight_decay=0.01, learning_rate=1e-3,
    )
    trainer.train()
    hist = trainer._train_loss_history
    assert len(hist) == 4
    _assert_finite(hist)


@metal_only
def test_vlm_lora_training_e2e(tmp_path):
    """Real VLM LoRA fit: collation, label masking, CCE, save."""
    from PIL import Image

    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    colors = ["red", "green", "blue", "yellow"]
    dataset = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "What color is this square?"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"This square is {color}."}],
                },
            ],
            "images": [Image.new("RGB", (64, 64), color)],
        }
        for color in colors
    ]

    model, processor = FastMLXModel.from_pretrained(VLM_MODEL, max_seq_length=512)
    model = FastMLXModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0)
    trainer = MLXTrainer(
        model=model,
        tokenizer=processor,
        train_dataset=dataset,
        args=MLXTrainingConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=4,
            warmup_steps=1,
            learning_rate=1e-4,
            logging_steps=1,
            output_dir=str(tmp_path),
            seed=3407,
            report_to="none",
        ),
    )
    assert trainer._is_vlm, "SmolVLM was not detected as a VLM"
    trainer.train()
    hist = trainer._train_loss_history
    assert len(hist) == 4
    _assert_finite(hist)
    saved = glob.glob(os.path.join(str(tmp_path), "**", "*.safetensors"), recursive=True)
    assert saved, "no adapter safetensors saved at end of VLM training"


def _color_square_dataset(colors):
    """Synthetic VLM dataset: one solid-color 64x64 square per message."""
    from PIL import Image
    return [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "What color is this square?"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"This square is {color}."}],
                },
            ],
            "images": [Image.new("RGB", (64, 64), color)],
        }
        for color in colors
    ]


def _make_vlm_trainer(tmp_path, dataset, **config_overrides):
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    model, processor = FastMLXModel.from_pretrained(VLM_MODEL, max_seq_length=512)
    model = FastMLXModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0)
    config = dict(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=4,
        warmup_steps=1,
        learning_rate=1e-4,
        logging_steps=1,
        output_dir=str(tmp_path),
        seed=3407,
        report_to="none",
    )
    config.update(config_overrides)
    return MLXTrainer(
        model=model,
        tokenizer=processor,
        train_dataset=dataset,
        args=MLXTrainingConfig(**config),
    )


@metal_only
def test_vlm_resume_from_checkpoint_matches_fresh_run(tmp_path):
    """VLM stop+resume reproduces the fresh run's losses step for step.

    Mirrors test_resume_from_checkpoint_matches_fresh_run for a VLM model
    so the resume code path (optimizer state restore, batch fast-forward,
    LR schedule offset) is exercised through the multimodal collator and
    image processor in addition to the text-only path.
    """
    fresh_dir = tmp_path / "fresh"
    resume_dir = tmp_path / "resume"

    colors = ["red", "green", "blue", "yellow", "purple", "orange"]
    dataset = _color_square_dataset(colors)

    trainer = _make_vlm_trainer(
        fresh_dir, dataset, max_steps=6, save_steps=3,
    )
    assert trainer._is_vlm, f"{VLM_MODEL} was not detected as a VLM"
    trainer.train()
    fresh_hist = list(trainer._train_loss_history)
    assert len(fresh_hist) == 6, f"fresh run logged {len(fresh_hist)} losses"
    _assert_finite(fresh_hist)

    ckpt_dir = fresh_dir / "checkpoint-3"
    assert (ckpt_dir / "adapters.safetensors").is_file()
    assert (ckpt_dir / "optimizer_state.safetensors").is_file()
    assert (ckpt_dir / "trainer_state.json").is_file()
    with open(ckpt_dir / "trainer_state.json") as f:
        saved_state = json.load(f)
    assert saved_state["global_step"] == 3
    ckpt = str(ckpt_dir)

    # Free the fresh trainer before loading the second 2B model (memory-tight runners).
    del trainer
    gc.collect()

    # Fresh process state: new base model, same seeds, resume from step 3.
    resumed = _make_vlm_trainer(
        resume_dir, _color_square_dataset(colors), max_steps=6, save_steps=0,
    )
    resumed.train(resume_from_checkpoint=ckpt)
    resumed_hist = list(resumed._train_loss_history)
    assert len(resumed_hist) == 6, f"resumed run logged {len(resumed_hist)} losses"
    _assert_finite(resumed_hist)

    # Restored prefix is the checkpointed history; post-resume steps must
    # track the fresh run. Same seeds + restored Adam moments mean the only
    # tolerated difference is float accumulation noise.
    for i, (a, b) in enumerate(zip(fresh_hist, resumed_hist), start=1):
        assert abs(a - b) <= 1e-5 * max(1.0, abs(a)), (
            f"step {i}: fresh={a!r} resumed={b!r}\n"
            f"fresh={fresh_hist}\nresumed={resumed_hist}"
        )


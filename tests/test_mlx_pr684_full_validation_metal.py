"""Metal end-to-end checks for the PR 684 trainer rework: resume matches fresh-run losses (#751), exact epoch step
counts (pins the double-count fix), SGD weight decay, and real VLM LoRA training."""

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
# Qwen2-VL is the smallest VLM whose processor resolves under current transformers; SmolVLM-256M's preprocessor config cannot be mapped by AutoImageProcessor.
VLM_MODEL = "mlx-community/Qwen2-VL-2B-Instruct-4bit"


def _chat_dataset(n=12):
    # ChatML matches SmolLM-Instruct's template so response masking can anchor on the literal role markers.
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

    # Restored prefix is checkpointed history; post-resume losses must track the fresh run, since same seeds and restored Adam moments leave only float noise.
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
    # 12 samples at batch size 2 is 6 batches per epoch, so 3 epochs must give 18 steps; an epoch double-count bug once produced 54.
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
@pytest.mark.parametrize("optim_name", ["adam", "adam_8bit"])
def test_adam_weight_decay_moves_weights_e2e(tmp_path, optim_name):
    """wd > 0 must move LoRA weights well past a repeated wd = 0 run (noise floor)."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    def _final_lora(subdir, wd):
        trainer = _make_text_trainer(
            tmp_path / subdir, _chat_dataset(), max_steps=4,
            optim=optim_name, weight_decay=wd, learning_rate=1e-3,
        )
        trainer.train()
        _assert_finite(trainer._train_loss_history)
        return {
            k: v.astype(mx.float32)
            for k, v in tree_flatten(trainer.model.trainable_parameters())
            if "lora" in k
        }

    def _max_diff(a, b):
        return max(mx.abs(a[k] - b[k]).max().item() for k in a)

    base = _final_lora("wd0", 0.0)
    noise = _max_diff(base, _final_lora("wd0_again", 0.0))
    decayed = _max_diff(base, _final_lora("wd", 0.1))
    assert base, "no LoRA parameters were trained"
    assert decayed > max(10 * noise, 1e-6), (optim_name, decayed, noise)


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
    """VLM stop+resume must match the fresh run's losses, covering optimizer restore, batch fast-forward and LR offset through the multimodal collator."""
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

    # Restored prefix is checkpointed history; post-resume losses must track the fresh run, since same seeds and restored Adam moments leave only float noise.
    for i, (a, b) in enumerate(zip(fresh_hist, resumed_hist), start=1):
        assert abs(a - b) <= 1e-5 * max(1.0, abs(a)), (
            f"step {i}: fresh={a!r} resumed={b!r}\n"
            f"fresh={fresh_hist}\nresumed={resumed_hist}"
        )

#!/usr/bin/env python3
"""
Unsloth MLX Benchmark: Baseline CE vs CCE
==========================================
Simple sequential benchmark. Runs CCE then baseline for each model.
"""

import gc
import time

import mlx.core as mx
from mlx.utils import tree_map
from datasets import load_dataset

from unsloth import FastLanguageModel
from unsloth_zoo.mlx_trainer import MLXTrainer, MLXTrainingConfig
from unsloth_zoo.mlx_utils import create_batches

# =============================================================================
# CONFIG
# =============================================================================

MODELS = [
    # (model_name, display_name, use_lora)
    ("mlx-community/Llama-3.2-1B-Instruct-bf16", "Llama-1B-full", False),
    ("mlx-community/gemma-2-2b-it-4bit", "Gemma2-2B-4bit", True),
    ("mlx-community/Llama-3.2-3B-Instruct-4bit", "Llama-3B-4bit", True),
    ("mlx-community/Qwen2.5-3B-Instruct-4bit", "Qwen2.5-3B-4bit", True),
    ("mlx-community/Llama-3.2-3B-Instruct-bf16", "Llama-3B-LoRA", True),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-full", False),
]

BATCH_SIZE = 8
SEQ_LEN = 1024
WARMUP_STEPS = 10
MEASURE_STEPS = 20
SEED = 42
LR = 1e-5

LORA_RANK = 8
LORA_ALPHA = 16


def run_one(model, tokenizer, batches, use_cce, dataset):
    """Train and return (losses, peak_memory_gb, time_ms)."""
    trainer = MLXTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=MLXTrainingConfig(
            per_device_train_batch_size=BATCH_SIZE,
            max_steps=WARMUP_STEPS + MEASURE_STEPS,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            learning_rate=LR,
            optim="adafactor",
            logging_steps=1,
            max_seq_length=SEQ_LEN,
            output_dir=f"./tmp_bench_{'cce' if use_cce else 'baseline'}",
            use_cce=use_cce,
        ),
    )
    trainer._batches = batches

    gc.collect()
    mx.synchronize()
    mx.reset_peak_memory()

    stats = trainer.train()

    mx.synchronize()
    peak_gb = mx.get_peak_memory() / 1e9
    total_time = stats["train_runtime"]
    time_per_step = total_time / (WARMUP_STEPS + MEASURE_STEPS)
    measure_ms = time_per_step * MEASURE_STEPS * 1000

    losses = trainer._train_loss_history[WARMUP_STEPS:]
    return losses, peak_gb, measure_ms


def main():
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    print("=" * 70)
    print("Unsloth MLX Benchmark: Baseline CE vs CCE")
    print("=" * 70)
    print(f"Batch: {BATCH_SIZE}, Seq: {SEQ_LEN}, Warmup: {WARMUP_STEPS}, Measure: {MEASURE_STEPS}")
    print(f"Optimizer: Adafactor (lr={LR})")
    print(f"Models: {len(MODELS)}")
    print("=" * 70)

    results = []

    for i, (model_name, display_name, use_lora) in enumerate(MODELS):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(MODELS)}] {display_name}")
        print(f"  Repo: {model_name}, LoRA: {use_lora}")
        print("=" * 70)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name, max_seq_length=SEQ_LEN,
        )

        if use_lora:
            model = FastLanguageModel.get_peft_model(
                model, r=LORA_RANK, lora_alpha=LORA_ALPHA,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
            )

        total_batches = WARMUP_STEPS + MEASURE_STEPS + 5
        batches = create_batches(
            dataset, tokenizer,
            batch_size=BATCH_SIZE,
            max_seq_length=SEQ_LEN,
            num_batches=total_batches,
            seed=SEED,
        )

        # Save initial weights to restore between runs
        init_w = tree_map(
            lambda x: mx.array(x) if isinstance(x, mx.array) else x,
            model.parameters(),
        )
        mx.eval(init_w)

        # --- CCE ---
        print("\n--- CCE ---")
        model.update(init_w); mx.eval(model.parameters())
        cce_losses, cce_mem, cce_ms = run_one(model, tokenizer, batches, True, dataset)

        # --- BASELINE ---
        print("\n--- BASELINE ---")
        model.update(init_w); mx.eval(model.parameters())
        bl_losses, bl_mem, bl_ms = run_one(model, tokenizer, batches, False, dataset)

        # Compare
        loss_diffs = [abs(b - c) for b, c in zip(bl_losses, cce_losses)]
        max_diff = max(loss_diffs)
        speedup = bl_ms / cce_ms if cce_ms > 0 else 0
        mem_save = (bl_mem - cce_mem) / bl_mem * 100 if bl_mem > 0 else 0
        status = "PASS" if max_diff < 0.1 else "FAIL"

        print(f"\n  {'='*60}")
        print(f"  {display_name}")
        print(f"  Baseline: {bl_ms:.0f}ms | {bl_mem:.2f}GB | loss={bl_losses[-1]:.4f}")
        print(f"  CCE:      {cce_ms:.0f}ms | {cce_mem:.2f}GB | loss={cce_losses[-1]:.4f}")
        print(f"  Speedup: {speedup:.2f}x | Mem Saved: {mem_save:.1f}% | Max Diff: {max_diff:.4f} | {status}")
        print(f"  {'='*60}")

        results.append((display_name, speedup, mem_save, max_diff, status))

        # Cleanup before next model
        del model, tokenizer, batches, init_w
        gc.collect()

    # Summary
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<22} {'Speedup':>8} {'MemSave':>8} {'MaxDiff':>8} {'Status':>7}")
    print("-" * 70)
    for name, spd, mem, diff, st in results:
        print(f"{name:<22} {spd:>7.2f}x {mem:>7.1f}% {diff:>8.4f} {st:>7}")
    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()

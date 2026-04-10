#!/usr/bin/env python3
"""
Unsloth MLX Benchmark: Quantized CCE vs Baseline CE
====================================================
Tests 4-bit quantized models where CCE uses fused quantized matmul kernels
(no dequantization needed) vs standard cross-entropy through the full LM head.

For each model: reset weights, run CCE, reset weights, run baseline.
Compare speed, memory, and loss agreement.
"""

import gc
import time

import mlx.core as mx
from mlx.utils import tree_map
from datasets import load_dataset

from unsloth import FastLanguageModel
from unsloth_zoo.mlx_trainer import MLXTrainer, MLXTrainingConfig
from unsloth_zoo.mlx_utils import create_batches, _get_lm_head_layer, _is_quantized_layer

# =============================================================================
# CONFIG
# =============================================================================

MODELS = [
    # All 4-bit quantized — the quantized CCE kernel path
    # (model_name, display_name)
    ("mlx-community/Llama-3.2-1B-Instruct-4bit",  "Llama-1B-4bit"),
    ("mlx-community/gemma-2-2b-it-4bit",           "Gemma2-2B-4bit"),
    ("mlx-community/Llama-3.2-3B-Instruct-4bit",   "Llama-3B-4bit"),
    ("mlx-community/Qwen2.5-3B-Instruct-4bit",     "Qwen2.5-3B-4bit"),
]

BATCH_SIZE = 4
SEQ_LEN = 512
WARMUP_STEPS = 5
MEASURE_STEPS = 15
SEED = 42
LR = 2e-4

LORA_RANK = 16
LORA_ALPHA = 16


def run_one(model, tokenizer, batches, use_cce, dataset):
    """Train and return (losses, peak_memory_gb, step_time_ms)."""
    total_steps = WARMUP_STEPS + MEASURE_STEPS
    trainer = MLXTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=MLXTrainingConfig(
            per_device_train_batch_size=BATCH_SIZE,
            max_steps=total_steps,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            learning_rate=LR,
            optim="adafactor",
            logging_steps=1,
            max_seq_length=SEQ_LEN,
            output_dir=f"./tmp_bench_{'qcce' if use_cce else 'baseline'}",
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
    # Per-step time from measured steps only
    total_time = stats["train_runtime"]
    step_ms = (total_time / total_steps) * 1000

    losses = trainer._train_loss_history[WARMUP_STEPS:]
    return losses, peak_gb, step_ms


def main():
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    print("=" * 72)
    print("Unsloth MLX Benchmark: Quantized CCE vs Baseline CE")
    print("=" * 72)
    print(f"Batch: {BATCH_SIZE}  |  Seq: {SEQ_LEN}  |  Warmup: {WARMUP_STEPS}  |  Measure: {MEASURE_STEPS}")
    print(f"LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}")
    print(f"Optimizer: Adafactor (lr={LR})")
    print(f"Models: {len(MODELS)} (all 4-bit quantized)")
    print("=" * 72)

    results = []

    for i, (model_name, display_name) in enumerate(MODELS):
        print(f"\n{'='*72}")
        print(f"[{i+1}/{len(MODELS)}] {display_name}")
        print(f"  Repo: {model_name}")
        print("=" * 72)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name, max_seq_length=SEQ_LEN,
        )

        # Verify it's quantized
        lm_layer = _get_lm_head_layer(model)
        is_quant = _is_quantized_layer(lm_layer)
        bits = getattr(lm_layer, "bits", "?")
        gs = getattr(lm_layer, "group_size", "?")
        print(f"  Quantized: {is_quant} (bits={bits}, group_size={gs})")

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

        # --- Quantized CCE ---
        print("\n--- Quantized CCE ---")
        model.update(init_w); mx.eval(model.parameters())
        cce_losses, cce_mem, cce_ms = run_one(model, tokenizer, batches, True, dataset)

        # --- Baseline CE ---
        print("\n--- Baseline CE ---")
        model.update(init_w); mx.eval(model.parameters())
        bl_losses, bl_mem, bl_ms = run_one(model, tokenizer, batches, False, dataset)

        # Compare
        loss_diffs = [abs(b - c) for b, c in zip(bl_losses, cce_losses)]
        max_diff = max(loss_diffs)
        avg_diff = sum(loss_diffs) / len(loss_diffs)
        speedup = bl_ms / cce_ms if cce_ms > 0 else 0
        mem_save_gb = bl_mem - cce_mem
        mem_save_pct = mem_save_gb / bl_mem * 100 if bl_mem > 0 else 0
        status = "PASS" if max_diff < 0.1 else "WARN"

        print(f"\n  {'='*60}")
        print(f"  {display_name}")
        print(f"  Baseline: {bl_ms:.0f} ms/step | {bl_mem:.2f} GB | final loss={bl_losses[-1]:.4f}")
        print(f"  Qnt CCE:  {cce_ms:.0f} ms/step | {cce_mem:.2f} GB | final loss={cce_losses[-1]:.4f}")
        print(f"  Speedup: {speedup:.2f}x | Mem saved: {mem_save_gb:.2f} GB ({mem_save_pct:.1f}%)")
        print(f"  Loss diff: max={max_diff:.4f}, avg={avg_diff:.4f} | {status}")
        print(f"  {'='*60}")

        results.append({
            "name": display_name,
            "speedup": speedup,
            "mem_save_gb": mem_save_gb,
            "mem_save_pct": mem_save_pct,
            "max_diff": max_diff,
            "avg_diff": avg_diff,
            "cce_ms": cce_ms,
            "bl_ms": bl_ms,
            "cce_mem": cce_mem,
            "bl_mem": bl_mem,
            "cce_final_loss": cce_losses[-1],
            "bl_final_loss": bl_losses[-1],
            "status": status,
        })

        # Cleanup before next model
        del model, tokenizer, batches, init_w
        gc.collect()

    # Summary table
    print(f"\n\n{'='*72}")
    print("FINAL SUMMARY — Quantized CCE vs Baseline CE (4-bit models)")
    print(f"{'='*72}")
    print(f"{'Model':<20} {'ms/step':>14} {'Peak Mem (GB)':>16} {'Speedup':>8} {'MemSave':>8} {'MaxDiff':>8} {'Status':>7}")
    print(f"{'':20} {'CCE':>7}{'BL':>7} {'CCE':>8}{'BL':>8}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['name']:<20} "
            f"{r['cce_ms']:>6.0f} {r['bl_ms']:>6.0f} "
            f"{r['cce_mem']:>7.2f} {r['bl_mem']:>7.2f} "
            f"{r['speedup']:>7.2f}x "
            f"{r['mem_save_pct']:>6.1f}% "
            f"{r['max_diff']:>8.4f} "
            f"{r['status']:>6}"
        )
    print("=" * 72)
    print("Done!")


if __name__ == "__main__":
    main()

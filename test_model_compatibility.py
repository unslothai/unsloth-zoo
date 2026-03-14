#!/usr/bin/env python3
"""
Unsloth MLX+CCE Model Compatibility & Benchmark
================================================
Runs 15 warmup + 50 training steps per (model, loss_mode) config.
Each run is an isolated subprocess. Stores every loss for curve plotting,
plus peak memory and throughput for CCE vs baseline comparison.

Results saved to test_results/ with per-test log files and results.json.
"""

import json
import shutil
import subprocess
import sys
import os
import time

# Remove cwd from sys.path to prevent the repo directory (e.g. Desktop/unsloth/)
# from shadowing the editable-installed unsloth package as a namespace package.
sys.path = [p for p in sys.path if p not in ("", ".")]

# =============================================================================
# CONFIG
# =============================================================================

ATTN_ONLY = ["q_proj", "k_proj", "v_proj", "o_proj"]
ATTN_MLP = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Models that need adamw (adafactor breaks on 3D LoRA tensors)
ADAMW_MODELS = {"gpt-oss-20B-4bit"}

# (model_repo, display_name, training_mode, lora_targets)
MODELS = [
    # --- Llama ---
    ("mlx-community/Llama-3.2-1B-Instruct-4bit",            "Llama-1B-4bit",       "qlora", ATTN_MLP),
    ("mlx-community/Llama-3.2-3B-Instruct-4bit",            "Llama-3B-4bit",       "qlora", ATTN_MLP),
    ("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",       "Llama-8B-4bit",       "qlora", ATTN_MLP),
    ("mlx-community/Llama-3.2-1B-Instruct-bf16",            "Llama-1B-bf16-lora",  "lora",  ATTN_MLP),
    ("mlx-community/Llama-3.2-1B-Instruct-bf16",            "Llama-1B-bf16-full",  "full",  None),

    # --- Qwen (text-only) ---
    ("mlx-community/Qwen2.5-3B-Instruct-4bit",              "Qwen2.5-3B-4bit",    "qlora", ATTN_MLP),
    ("mlx-community/Qwen3-0.6B-4bit",                       "Qwen3-0.6B-4bit",    "qlora", ATTN_MLP),

    # --- Gemma ---
    ("mlx-community/gemma-2-2b-it-4bit",                    "Gemma2-2B-4bit",      "qlora", ATTN_MLP),
    ("mlx-community/gemma-3-1b-it-4bit",                    "Gemma3-1B-4bit",      "qlora", ATTN_MLP),

    # --- Phi ---
    ("mlx-community/Phi-3.5-mini-instruct-4bit",            "Phi3.5-mini-4bit",    "qlora", ATTN_ONLY),
    ("mlx-community/Phi-4-mini-instruct-4bit",              "Phi4-mini-4bit",      "qlora", ATTN_ONLY),

    # --- Mistral ---
    ("mlx-community/Mistral-7B-Instruct-v0.3-4bit",        "Mistral-7B-4bit",     "qlora", ATTN_MLP),

    # --- gpt-oss ---
    ("mlx-community/gpt-oss-20b-MXFP4-Q4",                 "gpt-oss-20B-4bit",    "qlora", ATTN_ONLY),

    # --- DeepSeek (MoE architecture) ---
    ("mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx",        "DeepSeek-V2L-4bit",   "qlora", ATTN_ONLY),
]

BATCH_SIZE = 8
SEQ_LEN = 1024
WARMUP_STEPS = 15
TRAIN_STEPS = 50
MAX_STEPS = WARMUP_STEPS + TRAIN_STEPS  # 65 total
SEED = 42
LR = 2e-5
LORA_RANK = 8
LORA_ALPHA = 16
TIMEOUT = 600  # 10 minutes per subprocess

# Results directory
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results")


# =============================================================================
# WORKER — runs in subprocess
# =============================================================================

def run_worker(model_name, display_name, training_mode, use_cce, lora_targets_json):
    """Run a single (model, mode, loss) benchmark. Prints JSON result to stdout."""
    import traceback
    try:
        import gc
        import mlx.core as mx
        from datasets import load_dataset
        from unsloth import FastLanguageModel
        from unsloth_zoo.mlx_trainer import MLXTrainer, MLXTrainingConfig
        from unsloth_zoo.mlx_utils import create_batches

        lora_targets = json.loads(lora_targets_json) if lora_targets_json else None
        optimizer = "adamw" if display_name in ADAMW_MODELS else "adafactor"

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name, max_seq_length=SEQ_LEN,
        )

        if training_mode in ("qlora", "lora"):
            targets = lora_targets or ATTN_ONLY
            model = FastLanguageModel.get_peft_model(
                model, r=LORA_RANK, lora_alpha=LORA_ALPHA,
                target_modules=targets,
            )

        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        batches = create_batches(
            dataset, tokenizer,
            batch_size=BATCH_SIZE, max_seq_length=SEQ_LEN,
            num_batches=MAX_STEPS + 10, seed=SEED,
        )

        trainer = MLXTrainer(
            model=model, tokenizer=tokenizer, train_dataset=dataset,
            args=MLXTrainingConfig(
                per_device_train_batch_size=BATCH_SIZE,
                max_steps=MAX_STEPS,
                gradient_accumulation_steps=1,
                learning_rate=LR,
                warmup_steps=WARMUP_STEPS,
                lr_scheduler_type="cosine",
                optim=optimizer,
                logging_steps=1,
                max_seq_length=SEQ_LEN,
                seed=SEED,
                use_cce=use_cce,
                compile=True,
                gradient_checkpointing=True,
            ),
        )
        trainer._batches = batches

        gc.collect()
        mx.synchronize()
        mx.reset_peak_memory()

        train_start = time.time()
        trainer.train()
        mx.synchronize()
        train_time = time.time() - train_start

        peak_gb = mx.get_peak_memory() / 1e9
        losses = trainer._train_loss_history
        nan_count = sum(1 for l in losses if l != l)
        total_tokens = getattr(trainer, '_total_tokens', 0)
        tokens_per_sec = total_tokens / train_time if train_time > 0 else 0

        result = {
            "status": "ok",
            "losses": losses,
            "nan_count": nan_count,
            "peak_gb": peak_gb,
            "train_time_s": round(train_time, 2),
            "total_tokens": total_tokens,
            "tokens_per_sec": round(tokens_per_sec, 1),
            "optimizer": optimizer,
            "error": None,
        }

    except Exception as e:
        result = {
            "status": "error",
            "losses": [],
            "nan_count": 0,
            "peak_gb": 0,
            "train_time_s": 0,
            "total_tokens": 0,
            "tokens_per_sec": 0,
            "optimizer": "adamw" if display_name in ADAMW_MODELS else "adafactor",
            "error": f"{type(e).__name__}: {e}",
        }
        traceback.print_exc()

    print(f"__RESULT__{json.dumps(result)}", flush=True)


# =============================================================================
# ORCHESTRATOR
# =============================================================================

def classify_result(result):
    if result is None:
        return "CRASH"
    if result["status"] == "error":
        err = result["error"] or ""
        if "OOM" in err or "memory" in err.lower() or "MemoryError" in err:
            return "OOM"
        return "FAIL"
    if result["nan_count"] > 0:
        return "NaN"
    losses = result.get("losses", [])
    if len(losses) < 2:
        return "FAIL"
    if not (0.001 < losses[0] < 50.0) or not (0.001 < losses[-1] < 50.0):
        return "WARN"
    return "PASS"


def make_log_filename(display_name, loss_label):
    safe_name = display_name.replace(" ", "_").replace("/", "_")
    return f"{safe_name}_{loss_label.lower()}.log"


def run_subprocess(cmd, log_path, timeout=TIMEOUT):
    """Run subprocess, stream to console AND save to log file."""
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        lines = []
        with open(log_path, "w") as log_file:
            for line in proc.stdout:
                line = line.rstrip("\n")
                lines.append(line)
                log_file.write(line + "\n")
                log_file.flush()
                if not line.startswith("__RESULT__"):
                    print(f"    {line}", flush=True)
        proc.wait(timeout=timeout)
        return lines, proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return [], -1
    except Exception as e:
        return [str(e)], -2


def _cleanup_hf_cache(model_name):
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    folder_name = "models--" + model_name.replace("/", "--")
    cache_path = os.path.join(cache_dir, folder_name)
    if os.path.exists(cache_path):
        size_gb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(cache_path) for f in fns
        ) / 1e9
        shutil.rmtree(cache_path, ignore_errors=True)
        print(f"  [cleanup] Removed {folder_name} ({size_gb:.1f}GB)")
    else:
        print(f"  [cleanup] {folder_name} not in cache")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 90)
    print("Unsloth MLX+CCE Model Benchmark")
    print("=" * 90)
    print(f"BS={BATCH_SIZE} | Seq={SEQ_LEN} | Warmup={WARMUP_STEPS} | "
          f"Train={TRAIN_STEPS} | Total={MAX_STEPS} steps")
    print(f"LR={LR} | LR_sched=cosine | Opt=adafactor (adamw for gpt-oss) | Compile+GC=ON")
    print(f"Models: {len(MODELS)} configs | Loss modes: CCE + Baseline (sequential)")
    print(f"Timeout: {TIMEOUT}s | Logs: {RESULTS_DIR}/")
    print("=" * 90)

    script_path = os.path.abspath(__file__)
    python = sys.executable
    all_results = []

    # Group by model_repo for cache cleanup
    model_groups = []
    for model_name, display_name, training_mode, lora_targets in MODELS:
        if not model_groups or model_groups[-1][0] != model_name:
            model_groups.append((model_name, []))
        model_groups[-1][1].append((display_name, training_mode, lora_targets))

    run_idx = 0
    total_runs = len(MODELS) * 2

    for model_name, configs in model_groups:
        for display_name, training_mode, lora_targets in configs:
            for loss_label, use_cce in [("CCE", True), ("Baseline", False)]:
                run_idx += 1
                test_name = f"{display_name} | {training_mode} | {loss_label}"
                log_file = make_log_filename(display_name, loss_label)
                log_path = os.path.join(RESULTS_DIR, log_file)

                print(f"\n--- [{run_idx}/{total_runs}] {test_name} ---")
                print(f"    Log: {log_file}")

                lora_json = json.dumps(lora_targets) if lora_targets else ""
                cmd = [
                    python, script_path, "--worker",
                    "--model", model_name,
                    "--display_name", display_name,
                    "--mode", training_mode,
                    "--use_cce", str(int(use_cce)),
                    "--lora_targets", lora_json,
                ]

                start = time.time()
                lines, returncode = run_subprocess(cmd, log_path)
                elapsed = time.time() - start

                # Parse result
                result = None
                for line in lines:
                    if line.startswith("__RESULT__"):
                        try:
                            result = json.loads(line[len("__RESULT__"):])
                        except json.JSONDecodeError:
                            pass

                if returncode == -1:
                    result = {"status": "error", "losses": [], "nan_count": 0,
                              "peak_gb": 0, "train_time_s": 0, "total_tokens": 0,
                              "tokens_per_sec": 0, "optimizer": "", "error": "TIMEOUT"}

                status = classify_result(result)
                peak = result["peak_gb"] if result else 0
                tps = result["tokens_per_sec"] if result else 0
                n_losses = len(result["losses"]) if result else 0
                final_loss = result["losses"][-1] if result and result["losses"] else None
                err = result.get("error", "") if result else "No output"

                final_str = f"{final_loss:.4f}" if final_loss is not None else "---"

                print(f"  >> {status:>5} | Steps={n_losses} | Final={final_str} | "
                      f"Peak={peak:.1f}GB | {tps:.0f} tok/s | {elapsed:.0f}s"
                      + (f" | {err}" if status not in ("PASS", "WARN") else ""))

                entry = {
                    "display_name": display_name,
                    "model": model_name,
                    "mode": training_mode,
                    "loss_mode": loss_label,
                    "losses": result["losses"] if result else [],
                    "nan_count": result["nan_count"] if result else 0,
                    "peak_gb": peak,
                    "train_time_s": result["train_time_s"] if result else 0,
                    "total_tokens": result["total_tokens"] if result else 0,
                    "tokens_per_sec": tps,
                    "optimizer": result["optimizer"] if result else "",
                    "status": status,
                    "error": err if status not in ("PASS", "WARN") else None,
                    "elapsed_s": round(elapsed, 2),
                    "log_file": log_file,
                }
                all_results.append(entry)

                # Save after every single test
                results_path = os.path.join(RESULTS_DIR, "results.json")
                with open(results_path, "w") as f:
                    json.dump(all_results, f, indent=2)

        # Clean cache after all configs for this model repo
        _cleanup_hf_cache(model_name)

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print(f"\n\n{'='*100}")
    print("SUMMARY — MLX+CCE Benchmark")
    print("=" * 100)
    print(f"{'Model':<24} {'Mode':<6} {'Loss':<9} {'Steps':>5} {'Final':>8} "
          f"{'Peak':>7} {'Tok/s':>7} {'Time':>6} {'Status':>6}")
    print("-" * 100)

    pass_count = warn_count = fail_count = 0
    for r in all_results:
        losses = r.get("losses", [])
        final = f"{losses[-1]:.4f}" if losses else "---"
        peak = f"{r['peak_gb']:.1f}GB" if r['peak_gb'] > 0 else "---"
        tps = f"{r['tokens_per_sec']:.0f}" if r['tokens_per_sec'] > 0 else "---"
        t = f"{r['train_time_s']:.0f}s" if r['train_time_s'] > 0 else "---"
        err_suffix = f"  ({r['error'][:35]})" if r.get('error') else ""

        print(f"{r['display_name']:<24} {r['mode']:<6} {r['loss_mode']:<9} "
              f"{len(losses):>5} {final:>8} {peak:>7} {tps:>7} {t:>6} "
              f"{r['status']:>6}{err_suffix}")

        if r['status'] == 'PASS':
            pass_count += 1
        elif r['status'] == 'WARN':
            warn_count += 1
        else:
            fail_count += 1

    # CCE vs Baseline comparison
    print(f"\n{'='*100}")
    print("CCE vs BASELINE Comparison")
    print("=" * 100)
    print(f"{'Model':<24} {'CCE Peak':>9} {'BL Peak':>9} {'Mem Save':>9} "
          f"{'CCE Tok/s':>9} {'BL Tok/s':>9} {'Speedup':>8}")
    print("-" * 100)

    # Pair up CCE and Baseline results
    by_key = {}
    for r in all_results:
        key = (r['display_name'], r['mode'])
        by_key.setdefault(key, {})[r['loss_mode']] = r

    for key in by_key:
        cce = by_key[key].get("CCE")
        bl = by_key[key].get("Baseline")
        if not cce or not bl:
            continue
        if cce['status'] not in ('PASS', 'WARN') or bl['status'] not in ('PASS', 'WARN'):
            continue

        cce_peak = cce['peak_gb']
        bl_peak = bl['peak_gb']
        mem_save = (1 - cce_peak / bl_peak) * 100 if bl_peak > 0 else 0
        cce_tps = cce['tokens_per_sec']
        bl_tps = bl['tokens_per_sec']
        speedup = cce_tps / bl_tps if bl_tps > 0 else 0

        print(f"{key[0]:<24} {cce_peak:>8.1f}GB {bl_peak:>8.1f}GB {mem_save:>8.1f}% "
              f"{cce_tps:>8.0f} {bl_tps:>8.0f} {speedup:>7.2f}x")

    total = len(all_results)
    print("=" * 100)
    print(f"TOTAL: {pass_count}/{total} PASS | {warn_count}/{total} WARN | "
          f"{fail_count}/{total} FAIL")
    print("=" * 100)
    print(f"\nResults: {os.path.join(RESULTS_DIR, 'results.json')}")
    print(f"Logs:    {RESULTS_DIR}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", type=str)
    parser.add_argument("--display_name", type=str, default="")
    parser.add_argument("--mode", type=str, default="qlora")
    parser.add_argument("--use_cce", type=int, default=0)
    parser.add_argument("--lora_targets", type=str, default="")
    args = parser.parse_args()

    if args.worker:
        run_worker(args.model, args.display_name, args.mode,
                   bool(args.use_cce), args.lora_targets or None)
    else:
        main()
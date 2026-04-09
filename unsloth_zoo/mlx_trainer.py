# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
MLXTrainer — drop-in trainer for Apple Silicon, mirroring SFTTrainer's API.

Usage mirrors TRL notebooks:

    from unsloth_zoo.mlx_trainer import MLXTrainer, MLXTrainingConfig

    trainer = MLXTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=MLXTrainingConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_steps=60,
            learning_rate=2e-4,
            use_cce=True,
        ),
    )
    trainer.train()
"""

from dataclasses import dataclass
import concurrent.futures
import math
import os
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten

_PAD_MULTIPLE = 32

from .mlx_utils import (
    make_cce_loss_fn,
    make_baseline_loss_fn,
    make_vlm_cce_loss_fn,
    make_vlm_baseline_loss_fn,
    create_batches,
    iterate_training_batches,
    create_vlm_batches,
    iterate_vlm_training_batches,
    save_lora_adapters,
    apply_gradient_checkpointing,
    remove_gradient_checkpointing,
    _is_vlm_model,
)


@dataclass
class MLXTrainingConfig:
    """Training configuration mirroring SFTConfig / TrainingArguments field names."""

    # Core training
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_steps: int = 60
    num_train_epochs: int = -1  # -1 means use max_steps instead
    warmup_steps: int = 5
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"  # "cosine", "linear", "constant"

    # Optimization
    optim: str = "adafactor"  # "adafactor", "adamw", "adam", "sgd", "muon", "lion"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # matches HuggingFace TrainingArguments default
    seed: int = 3407
    lora_plus_ratio: float = 0.0  # 0 = disabled, 16.0 = recommended
    embedding_learning_rate: float = 0.0  # 0 = disabled, 5e-5 = recommended

    # Logging & output
    logging_steps: int = 1
    output_dir: str = "./outputs"
    report_to: str = "none"
    save_steps: int = 0  # 0 = only save at end
    save_total_limit: int = -1  # -1 = no limit

    # Eval
    eval_steps: int = 0  # 0 = disabled

    # SFT-specific (from SFTConfig, for API compat)
    dataset_text_field: str = "text"
    max_seq_length: int = 2048
    packing: bool = False
    dataset_num_proc: int = 2

    # MLX-specific
    use_cce: bool = True
    compile: bool = True
    gradient_checkpointing: bool = True
    streaming: bool = False  # Use streaming iterator instead of materializing batches

    # VLM / completion masking
    train_on_completions: bool = False  # Mask prompt tokens in loss
    assistant_token_id: int = 0  # Token ID marking start of assistant response


class MLXTrainer:
    """MLX-native trainer for Apple Silicon, mirroring SFTTrainer's constructor API."""

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset=None,
        dataset_text_field=None,
        max_seq_length=None,
        packing=None,
        data_collator=None,
        args=None,
        formatting_func=None,
        processor=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.formatting_func = formatting_func
        # Use args or defaults
        self.args = args or MLXTrainingConfig()

        # Auto-detect VLM
        self._is_vlm = _is_vlm_model(model)

        # Constructor params override args if provided
        if dataset_text_field is not None:
            self.args.dataset_text_field = dataset_text_field
        if max_seq_length is not None:
            self.args.max_seq_length = max_seq_length
        if packing is not None:
            self.args.packing = packing

        if self.args.packing:
            print(
                "Unsloth: packing=True is not yet supported on MLX. "
                "Falling back to packing=False (standard padding)."
            )
            self.args.packing = False

        # Safety: freeze non-LoRA parameters if LoRA layers are detected.
        # mlx-lm calls model.freeze() BEFORE linear_to_lora_layers(), but users
        # might forget. Without this, LayerNorm weights remain trainable and
        # adaptive optimizers (Adafactor, AdamW) produce NaN on the first step
        # because their 1D second-moment initialization is numerically unstable.
        self._ensure_lora_frozen(model)

        # Training state
        self._global_step = 0
        self._train_loss_history = []
        self._step_times = []
        self._batches = None  # Pre-created batches (skips internal batch creation)

    @staticmethod
    def _ensure_lora_frozen(model):
        """Freeze accidentally trainable norm parameters when LoRA is active.

        Without this, LayerNorm/RMSNorm weights remain trainable, and
        adaptive optimizers produce NaN on 1D tensors at initialization
        (the second-moment estimate starts at 0, causing division by ~eps).

        Only freezes norm parameters — does NOT touch projector, vision, or
        other intentionally trainable non-LoRA parameters.
        """
        trainable = dict(tree_flatten(model.trainable_parameters()))
        has_lora = any("lora" in k for k in trainable)
        if not has_lora:
            return  # Not a LoRA model — don't touch

        # Only freeze params that look like accidentally unfrozen norms.
        # Projector weights, vision tower weights, etc. are intentionally
        # trainable when train_projector/train_vision is used.
        _NORM_FRAGMENTS = (".norm.", "norm.weight", "norm.bias",
                           ".ln_", "ln_f.weight", "ln_f.bias")
        # Don't freeze norms inside components the user explicitly unfroze
        # (projector via train_projector, vision tower via train_vision).
        _INTENTIONAL_COMPONENTS = (
            "multi_modal_projector", "mm_projector", "connector", "aligner",
            "vision_tower", "vision_model", "vision_encoder",
        )
        suspect = [
            k for k in trainable
            if "lora" not in k
            and any(frag in k for frag in _NORM_FRAGMENTS)
            and not any(comp in k for comp in _INTENTIONAL_COMPONENTS)
        ]
        if not suspect:
            return  # No accidental norms — nothing to fix

        for key in suspect:
            parts = key.split(".")
            obj = model
            for p in parts[:-1]:
                try:
                    obj = obj[int(p)]
                except (ValueError, TypeError):
                    obj = getattr(obj, p)
            obj.freeze(keys=[parts[-1]], recurse=False)

        print(
            f"Unsloth: Froze {len(suspect)} accidentally trainable norm "
            f"parameters to prevent optimizer NaN."
        )

    def _build_schedule(self, total_steps):
        """Build LR schedule from config. Returns a callable or float."""
        lr = self.args.learning_rate
        warmup = self.args.warmup_steps
        sched_type = self.args.lr_scheduler_type.lower()

        if sched_type == "constant" and warmup == 0:
            return lr

        decay_steps = max(total_steps - warmup, 1)

        if sched_type == "cosine":
            end_lr = lr * 0.1
            main_schedule = optim.cosine_decay(lr, decay_steps, end=end_lr)
        elif sched_type == "linear":
            main_schedule = optim.linear_schedule(lr, 0.0, decay_steps)
        else:  # constant
            main_schedule = lr

        if warmup > 0:
            warmup_fn = optim.linear_schedule(0.0, lr, warmup)
            if callable(main_schedule):
                return optim.join_schedules(
                    [warmup_fn, main_schedule], [warmup]
                )
            else:
                const_fn = optim.linear_schedule(lr, lr, decay_steps)
                return optim.join_schedules(
                    [warmup_fn, const_fn], [warmup]
                )

        return main_schedule

    def _build_optimizer(self, total_steps):
        """Create MLX optimizer with LR schedule from config.

        For optimizers that support weight_decay, wraps with
        optim.decay_weight to exclude bias and norm parameters
        (matching HuggingFace Trainer behavior).
        """
        schedule = self._build_schedule(total_steps)
        wd = self.args.weight_decay

        opt_name = self.args.optim.lower()
        if opt_name == "adafactor":
            optimizer = optim.Adafactor(
                learning_rate=schedule,
                relative_step=False,
                scale_parameter=False,
            )
        elif opt_name == "adamw":
            optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.0)
        elif opt_name == "adam":
            optimizer = optim.Adam(learning_rate=schedule)
        elif opt_name == "sgd":
            optimizer = optim.SGD(learning_rate=schedule, weight_decay=0.0)
        elif opt_name == "muon":
            optimizer = optim.Muon(learning_rate=schedule, weight_decay=0.0)
        elif opt_name == "lion":
            optimizer = optim.Lion(learning_rate=schedule, weight_decay=0.0)
        else:
            print(f"Unknown optimizer '{opt_name}', falling back to Adafactor.")
            optimizer = optim.Adafactor(
                learning_rate=schedule,
                relative_step=False,
                scale_parameter=False,
            )

        # Apply weight decay only to 2D+ weight tensors (not biases, norms,
        # embeddings). Matches HuggingFace Trainer.get_decay_parameter_names.
        if wd > 0 and opt_name not in ("adafactor", "adam"):
            def _wd_predicate(module, key, value):
                return key == "weight" and value.ndim >= 2
            optimizer = optim.chain(
                optimizer,
                optim.decay_weight(wd, predicate=_wd_predicate),
            )

        return optimizer

    def _evaluate(self, eval_batches, loss_fn, is_vlm=False):
        """Run evaluation loop.

        Returns:
            (avg_loss, perplexity) tuple.
        """
        self.model.eval()
        all_losses = mx.array(0.0)
        ntokens = mx.array(0)

        for batch_data in eval_batches:
            if is_vlm:
                loss, ntoks = loss_fn(self.model, batch_data)
            else:
                batch, lengths, labels = batch_data
                loss, ntoks = loss_fn(self.model, batch, lengths, labels)
            all_losses += loss * ntoks
            ntokens += ntoks
            mx.eval(all_losses, ntokens)

        self.model.train()
        avg_loss = (all_losses / ntokens).item() if ntokens.item() > 0 else 0.0
        perplexity = math.exp(min(avg_loss, 100))
        return avg_loss, perplexity

    def train(self):
        """Run MLX-native training loop.

        Follows mlx-lm's compiled step pattern with proper gradient accumulation:
        one compiled function handles forward+backward, accumulates gradients across
        micro-batches, and conditionally applies optimizer update.

        Returns:
            dict with training metrics (train_loss, train_runtime, etc.)
        """
        args = self.args
        model = self.model

        # Set wired memory limit (reduces page faults)
        if mx.metal.is_available():
            mx.set_wired_limit(
                mx.device_info()["max_recommended_working_set_size"]
            )

        # Apply gradient checkpointing if requested
        if args.gradient_checkpointing:
            apply_gradient_checkpointing(model)
            print("Unsloth: Using gradient checkpointing to reduce memory.")

        try:
            return self._train_inner()
        finally:
            if args.gradient_checkpointing:
                remove_gradient_checkpointing(model)

    def _train_inner(self):
        """Inner training loop, separated for GC cleanup in finally block."""
        args = self.args
        model = self.model
        is_vlm = self._is_vlm

        # Pick loss function — returns (loss, ntoks) tuples
        use_cce = args.use_cce

        if is_vlm:
            _atid = args.assistant_token_id if args.train_on_completions else 0
            if use_cce:
                loss_fn = make_vlm_cce_loss_fn(model, assistant_token_id=_atid)
                cce_backend = getattr(loss_fn, "_unsloth_cce_backend", "unknown")
                print(f"Unsloth: Using VLM CCE loss ({cce_backend}) for memory-efficient training.")
            else:
                loss_fn = make_vlm_baseline_loss_fn(model, assistant_token_id=_atid)
                print("Unsloth: Using VLM standard cross-entropy loss.")
        else:
            if use_cce:
                loss_fn = make_cce_loss_fn(model)
                cce_backend = getattr(loss_fn, "_unsloth_cce_backend", "unknown")
                print(f"Unsloth: Using CCE loss ({cce_backend}) for memory-efficient training.")
            else:
                loss_fn = make_baseline_loss_fn()
                print("Unsloth: Using standard cross-entropy loss.")

        # Prepare data — determine total_steps first
        if is_vlm:
            batches, batch_iter = self._prepare_vlm_data()
        else:
            batches, batch_iter = self._prepare_text_data()

        if batches is not None and not batches:
            raise ValueError(
                "No training batches created. Check your dataset and batch_size."
            )

        grad_accum = args.gradient_accumulation_steps
        if args.max_steps > 0:
            total_steps = args.max_steps
        elif batches is not None:
            n_batches = len(batches)
            if args.num_train_epochs > 0:
                # Epoch-based: total micro-batches = epochs * batches_per_epoch
                total_steps = (n_batches * args.num_train_epochs) // grad_accum
            else:
                total_steps = n_batches // grad_accum
            total_steps = max(1, total_steps)
        else:
            # Streaming mode — must have max_steps
            if args.num_train_epochs > 0:
                raise ValueError(
                    "num_train_epochs requires a finite dataset (not streaming). "
                    "Use max_steps instead, or disable streaming."
                )
            raise ValueError(
                "max_steps must be > 0 when using streaming mode."
            )

        # Build optimizer with LR schedule
        optimizer = self._build_optimizer(total_steps)

        # Build loss+grad function — returns ((loss, ntoks), grads)
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

        # Per-parameter gradient scaling (LoRA+, embedding LR)
        lora_plus_ratio = args.lora_plus_ratio
        use_lora_plus = lora_plus_ratio > 0
        if use_lora_plus:
            print(f"Unsloth: LoRA+ enabled (ratio={lora_plus_ratio}).")

        embedding_lr = args.embedding_learning_rate
        main_lr = args.learning_rate
        # Ratio < 1 slows embeddings down; 0 = disabled
        use_embedding_lr = embedding_lr > 0 and main_lr > 0
        embedding_lr_ratio = embedding_lr / main_lr if use_embedding_lr else 1.0
        if use_embedding_lr:
            print(f"Unsloth: Embedding LR = {embedding_lr:.2e} "
                  f"(ratio={embedding_lr_ratio:.3f} of main LR {main_lr:.2e}).")

        _needs_grad_scaling = use_lora_plus or use_embedding_lr

        # Build step functions following mlx-lm's pattern
        max_grad_norm = args.max_grad_norm
        state = [model.state, optimizer.state, mx.random.state]

        def _apply_update(grad, toks_f):
            """Common gradient post-processing and optimizer update."""
            grad = tree_map(lambda g: g / toks_f, grad)
            if _needs_grad_scaling:
                flat = tree_flatten(grad)
                scaled = []
                for k, v in flat:
                    if use_lora_plus and "lora_b" in k:
                        v = v * lora_plus_ratio
                    if use_embedding_lr and ("embed_tokens" in k or "lm_head" in k):
                        v = v * embedding_lr_ratio
                    scaled.append((k, v))
                grad = tree_unflatten(scaled)
            if max_grad_norm > 0:
                grad, _ = optim.clip_grad_norm(grad, max_norm=max_grad_norm)
            optimizer.update(model, grad)

        if is_vlm:
            def step_fn(batch_dict, prev_state, do_update):
                (lvalue, toks), grad = loss_and_grad_fn(model, batch_dict)

                toks_f = toks.astype(mx.float32)
                grad = tree_map(lambda g: g * toks_f, grad)

                if prev_state is not None:
                    prev_grad, prev_toks = prev_state
                    grad = tree_map(lambda x, y: x + y, grad, prev_grad)
                    toks_f = toks_f + prev_toks

                if do_update:
                    _apply_update(grad, toks_f)
                    return lvalue, toks, None

                return lvalue, toks, (grad, toks_f)
        else:
            def step_fn(batch, lengths, labels, prev_state, do_update):
                (lvalue, toks), grad = loss_and_grad_fn(model, batch, lengths, labels)

                toks_f = toks.astype(mx.float32)
                grad = tree_map(lambda g: g * toks_f, grad)

                if prev_state is not None:
                    prev_grad, prev_toks = prev_state
                    grad = tree_map(lambda x, y: x + y, grad, prev_grad)
                    toks_f = toks_f + prev_toks

                if do_update:
                    _apply_update(grad, toks_f)
                    return lvalue, toks, None

                return lvalue, toks, (grad, toks_f)

        if args.compile:
            step_fn = mx.compile(step_fn, inputs=state, outputs=state)

        # Prepare eval batches
        eval_batches = None
        if args.eval_steps > 0 and self.eval_dataset is not None:
            # Use pre-built labeled eval batches if available
            _labeled_eval = getattr(self, '_eval_batches_labeled', None)
            if _labeled_eval is not None:
                eval_batches = _labeled_eval
            elif is_vlm:
                processor = self.processor or getattr(self.model, "_processor", None)
                config = getattr(self.model, "_config", {})
                _vlm_mask_fn = getattr(self, '_vlm_response_mask_fn', None)
                eval_batches = create_vlm_batches(
                    dataset=self.eval_dataset,
                    processor=processor,
                    config=config,
                    batch_size=args.per_device_train_batch_size,
                    max_seq_length=args.max_seq_length,
                    seed=args.seed,
                    response_mask_fn=_vlm_mask_fn,
                )
            else:
                eval_batches = create_batches(
                    dataset=self.eval_dataset,
                    tokenizer=self.tokenizer,
                    batch_size=args.per_device_train_batch_size,
                    max_seq_length=args.max_seq_length,
                    seed=args.seed,
                    dataset_text_field=args.dataset_text_field,
                    formatting_func=self.formatting_func,
                )
            if eval_batches:
                print(f"Unsloth: Eval enabled every {args.eval_steps} steps "
                      f"({len(eval_batches)} eval batches).")

        features = []
        if is_vlm:
            features.append("VLM")
        if use_cce:
            features.append("CCE")
        if args.gradient_checkpointing:
            features.append("GC")
        if args.compile:
            features.append("compile")
        if use_lora_plus:
            features.append(f"LoRA+(r={lora_plus_ratio})")
        features.append(f"LR={args.lr_scheduler_type}")
        features.append(f"opt={args.optim}")

        print(f"Unsloth: Training for {total_steps} steps, "
              f"BS={args.per_device_train_batch_size}, "
              f"grad_accum={grad_accum}, "
              f"seq_len={args.max_seq_length}")
        print(f"Unsloth: Features: {', '.join(features)}")

        # Training loop — mlx-lm pattern
        model.train()
        start_time = time.perf_counter()
        losses = 0
        n_tokens = 0
        steps = 0
        trained_tokens = 0
        train_time = 0
        grad_accum_state = None
        batch_idx = 0

        for it in range(1, total_steps * grad_accum + 1):
            tic = time.perf_counter()

            # Get next batch
            if batch_iter is not None:
                batch_data = next(batch_iter)
            else:
                batch_data = batches[batch_idx % len(batches)]
                batch_idx += 1

            do_update = (it % grad_accum == 0)

            if is_vlm:
                lvalue, toks, grad_accum_state = step_fn(
                    batch_data,
                    grad_accum_state,
                    do_update,
                )
            else:
                lvalue, toks, grad_accum_state = step_fn(
                    batch_data[0], batch_data[1], batch_data[2],
                    grad_accum_state,
                    do_update,
                )

            losses += lvalue * toks
            n_tokens += toks
            steps += 1
            if grad_accum_state is not None:
                mx.eval(state, losses, n_tokens, grad_accum_state[0], grad_accum_state[1])
            else:
                mx.eval(state, losses, n_tokens)
            train_time += time.perf_counter() - tic

            # Only log/eval on actual optimizer steps
            if not do_update:
                continue

            self._global_step = it // grad_accum
            current_step = self._global_step

            # Logging
            if current_step % args.logging_steps == 0 or current_step == total_steps:
                train_loss = (losses / n_tokens).item()
                tok_count = n_tokens.item()
                trained_tokens += tok_count
                lr_val = optimizer.learning_rate.item()
                tokens_sec = tok_count / train_time if train_time > 0 else 0
                peak_mem = mx.get_peak_memory() / 1e9

                self._train_loss_history.append(train_loss)
                self._step_times.append(train_time / steps if steps > 0 else 0)

                # Benchmark hook: reset peak memory after warmup
                reset_after = getattr(self, '_benchmark_reset_peak_after_step', 0)
                if reset_after > 0 and current_step == reset_after:
                    mx.synchronize()
                    mx.reset_peak_memory()

                print(
                    f"  Step {current_step}/{total_steps} | "
                    f"Loss: {train_loss:.4f} | "
                    f"LR: {lr_val:.2e} | "
                    f"Tok/s: {tokens_sec:.0f} | "
                    f"Peak: {peak_mem:.2f} GB"
                )

                losses = 0
                n_tokens = 0
                steps = 0
                train_time = 0

            # Eval
            if (eval_batches and args.eval_steps > 0
                    and current_step % args.eval_steps == 0):
                val_loss, ppl = self._evaluate(
                    eval_batches, loss_fn, is_vlm=is_vlm)
                model.train()
                print(
                    f"  Eval  {current_step}/{total_steps} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Perplexity: {ppl:.2f}"
                )

            # Checkpointing
            if args.save_steps > 0 and current_step % args.save_steps == 0:
                ckpt_dir = f"{args.output_dir}/checkpoint-{current_step}"
                save_lora_adapters(model, ckpt_dir)
                print(f"  Saved checkpoint to {ckpt_dir}")

        total_time = time.perf_counter() - start_time
        avg_loss = (
            sum(self._train_loss_history) / len(self._train_loss_history)
            if self._train_loss_history else 0.0
        )

        print(f"\nUnsloth: Training complete! "
              f"Avg loss: {avg_loss:.4f} | "
              f"Total time: {total_time:.1f}s | "
              f"Steps: {total_steps} | "
              f"Tokens: {trained_tokens}")

        return {
            "train_loss": avg_loss,
            "train_runtime": total_time,
            "train_steps": total_steps,
            "trained_tokens": trained_tokens,
            "train_samples_per_second": (
                trained_tokens / total_time if total_time > 0 else 0
            ),
        }

    def _prepare_text_data(self):
        """Prepare text training data. Returns (batches, batch_iter)."""
        args = self.args
        if self._batches is not None:
            return self._batches, None
        elif args.streaming:
            return None, iterate_training_batches(
                dataset=self.train_dataset,
                tokenizer=self.tokenizer,
                batch_size=args.per_device_train_batch_size,
                max_seq_length=args.max_seq_length,
                seed=args.seed,
                dataset_text_field=args.dataset_text_field,
                formatting_func=self.formatting_func,
            )
        else:
            total_batches_needed = (
                args.max_steps * args.gradient_accumulation_steps
                if args.max_steps > 0 else None
            )
            batches = create_batches(
                dataset=self.train_dataset,
                tokenizer=self.tokenizer,
                batch_size=args.per_device_train_batch_size,
                max_seq_length=args.max_seq_length,
                num_batches=total_batches_needed,
                seed=args.seed,
                dataset_text_field=args.dataset_text_field,
                formatting_func=self.formatting_func,
            )
            return batches, None

    def _prepare_vlm_data(self):
        """Prepare VLM training data. Returns (batches, batch_iter)."""
        args = self.args
        processor = self.processor or getattr(self.model, "_processor", None)
        if processor is None:
            raise ValueError(
                "VLM training requires a processor. Pass processor= to MLXTrainer "
                "or load the model with FastLanguageModel.from_pretrained()."
            )
        config = getattr(self.model, "_config", {})
        _vlm_mask_fn = getattr(self, '_vlm_response_mask_fn', None)

        if self._batches is not None:
            return self._batches, None
        elif args.streaming:
            return None, iterate_vlm_training_batches(
                dataset=self.train_dataset,
                processor=processor,
                config=config,
                batch_size=args.per_device_train_batch_size,
                max_seq_length=args.max_seq_length,
                seed=args.seed,
                response_mask_fn=_vlm_mask_fn,
            )
        else:
            total_batches_needed = (
                args.max_steps * args.gradient_accumulation_steps
                if args.max_steps > 0 else None
            )
            batches = create_vlm_batches(
                dataset=self.train_dataset,
                processor=processor,
                config=config,
                batch_size=args.per_device_train_batch_size,
                max_seq_length=args.max_seq_length,
                num_batches=total_batches_needed,
                seed=args.seed,
                response_mask_fn=_vlm_mask_fn,
            )
            # Safety check: detect all-masked VLM labels early
            if _vlm_mask_fn is not None and batches:
                _check_vlm_all_masked(batches)
            return batches, None

    def save_model(self, output_dir=None):
        """Save LoRA adapters or full merged model (if no LoRA)."""
        from .mlx_utils import save_merged_model
        output_dir = output_dir or self.args.output_dir

        trainable = dict(tree_flatten(self.model.trainable_parameters()))
        has_lora = any("lora" in k for k in trainable)

        if has_lora:
            self.model.save_lora_adapters(output_dir, adapter_config={
                "learning_rate": self.args.learning_rate,
                "max_steps": self.args.max_steps,
                "max_seq_length": self.args.max_seq_length,
                "use_cce": self.args.use_cce,
            })
            self.tokenizer.save_pretrained(output_dir)
            # VLMs: also save the processor (image preprocessor config)
            # so the adapter directory is complete for inference.
            _processor = self.processor or getattr(self.model, "_processor", None)
            if _processor is not None and hasattr(_processor, "save_pretrained"):
                _processor.save_pretrained(output_dir)
            print(f"Unsloth: LoRA adapters saved to {output_dir}")
        else:
            save_merged_model(self.model, self.tokenizer, output_dir)


def _create_labeled_batches(dataset, tokenizer, mask_fn, batch_size,
                            max_seq_length, formatting_func=None,
                            dataset_text_field="text", num_batches=None,
                            seed=42):
    """Create padded batches with label masks for train_on_responses_only.

    Tokenizes each dataset item, applies the masking closure to get labels,
    sorts by length, and produces right-padded 3-tuple batches.

    Returns:
        List of (batch, lengths, labels) tuples where:
        - batch: mx.array (BS, padded_len) — input_ids padded with 0
        - lengths: mx.array (BS, 2) — [1, actual_len - 1] per sequence
        - labels: mx.array (BS, padded_len) — labels padded with -100
    """
    eos_id = tokenizer.eos_token_id

    # 1. Gather all text strings (serial, fast)
    all_texts = []
    for item in dataset:
        if formatting_func is not None:
            result = formatting_func(item)
            texts = result if isinstance(result, list) else [result]
        elif isinstance(item, dict):
            text = item.get(dataset_text_field, "")
            texts = [text] if text else []
        elif isinstance(item, str):
            texts = [item]
        else:
            continue

        for text in texts:
            if text:
                all_texts.append(text)

    # 2. Tokenize + mask in parallel (HF fast tokenizers are thread-safe;
    #    slow tokenizers degrade gracefully via the GIL)
    def _process_text(text):
        encoded = tokenizer.encode(text)
        if eos_id is not None and (not encoded or encoded[-1] != eos_id):
            encoded.append(eos_id)
        if len(encoded) > max_seq_length:
            encoded = encoded[:max_seq_length]
        if len(encoded) < 2:
            return None
        result = mask_fn({"input_ids": [encoded]})
        labels = result["labels"]
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        return (encoded, labels[0])

    max_workers = min(4, os.cpu_count() or 1)
    all_items = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(_process_text, all_texts):
            if result is not None:
                all_items.append(result)

    if not all_items:
        raise ValueError(
            "No training data found after tokenization. "
            "Check your dataset and formatting_func."
        )

    # 2. Sort by length for efficient padding
    all_items.sort(key=lambda x: len(x[0]))

    # 3. Create padded batches
    rng = random.Random(seed)
    batches = []
    for start in range(0, len(all_items), batch_size):
        batch_items = all_items[start:start + batch_size]
        if not batch_items:
            continue
        max_len = max(len(ids) for ids, _ in batch_items)
        # Round up to nearest multiple of _PAD_MULTIPLE (matching mlx-lm)
        padded_len = ((max_len + _PAD_MULTIPLE - 1) // _PAD_MULTIPLE) * _PAD_MULTIPLE
        padded_len = min(padded_len, max_seq_length)

        batch_ids = []
        batch_labels = []
        batch_lengths = []
        for ids, lbls in batch_items:
            L = min(len(ids), padded_len)
            pad_len = padded_len - L
            batch_ids.append(ids[:L] + [0] * pad_len)
            batch_labels.append(lbls[:L] + [-100] * pad_len)
            batch_lengths.append([1, L - 1])

        batches.append((
            mx.array(batch_ids),
            mx.array(batch_lengths),
            mx.array(batch_labels),
        ))

    # 4. Shuffle batches
    rng.shuffle(batches)

    # Limit if needed
    if num_batches is not None and len(batches) > num_batches:
        batches = batches[:num_batches]

    # Evaluate all tensors
    all_tensors = []
    for b, l, lb in batches:
        all_tensors.extend([b, l, lb])
    mx.eval(all_tensors)

    return batches


def _check_all_masked(batches, max_check=100):
    """Safety check: raise if all labels in the first N batches are -100.

    Mirrors fix_zero_training_loss from the HF path.
    """
    seen_bad = 0
    seen_good = 0
    checked = 0
    for batch_ids, batch_lengths, batch_labels in batches:
        labels_list = batch_labels.tolist()
        for row in labels_list:
            unique = set(row)
            if unique == {-100}:
                seen_bad += 1
            else:
                seen_good += 1
            checked += 1
            if checked >= max_check:
                break
        if checked >= max_check:
            break

    if seen_bad == 0 and seen_good == 0:
        return
    ratio = seen_bad / (seen_bad + seen_good)
    # ZeroDivisionError matches fix_zero_training_loss in the HF/CUDA path
    if ratio == 1.0:
        raise ZeroDivisionError(
            "Unsloth: All labels in your dataset are -100. Training losses will be all 0.\n"
            "Are you sure you used `train_on_responses_only` correctly?\n"
            "Check that your instruction_part and response_part strings match "
            "the chat template used by your tokenizer."
        )
    elif ratio >= 0.9:
        import warnings
        warnings.warn(
            f"Unsloth: {seen_bad}/{seen_bad + seen_good} samples have all -100 labels "
            f"({ratio:.0%}). Your instruction_part / response_part may not match "
            f"the chat template correctly.",
            UserWarning,
        )


def _check_vlm_all_masked(batches, max_check=100):
    """Safety check for VLM batches: raise if all labels are -100.

    Same purpose as _check_all_masked but for VLM batch dicts
    (which have a "labels" key instead of a 3-tuple structure).
    """
    seen_bad = 0
    seen_good = 0
    checked = 0
    for batch_dict in batches:
        labels = batch_dict.get("labels")
        if labels is None:
            continue
        labels_list = labels.tolist()
        for row in labels_list:
            unique = set(row)
            if unique == {-100}:
                seen_bad += 1
            else:
                seen_good += 1
            checked += 1
            if checked >= max_check:
                break
        if checked >= max_check:
            break

    if seen_bad == 0 and seen_good == 0:
        return
    ratio = seen_bad / (seen_bad + seen_good)
    # ZeroDivisionError matches fix_zero_training_loss in the HF/CUDA path
    if ratio == 1.0:
        raise ZeroDivisionError(
            "Unsloth: All VLM labels in your dataset are -100. Training losses will be all 0.\n"
            "Are you sure you used `train_on_responses_only` correctly?\n"
            "Check that your instruction_part and response_part strings match "
            "the chat template used by your processor."
        )
    elif ratio >= 0.9:
        import warnings
        warnings.warn(
            f"Unsloth: {seen_bad}/{seen_bad + seen_good} VLM samples have all -100 labels "
            f"({ratio:.0%}). Your instruction_part / response_part may not match "
            f"the chat template correctly.",
            UserWarning,
        )


def train_on_responses_only(
    trainer,
    instruction_part=None,
    response_part=None,
    force_match=True,
    tokenizer=None,
    return_function=False,
    num_proc=None,
):
    """Mask instruction tokens from loss — train only on assistant responses.

    Call after MLXTrainer(...), before trainer.train(). Works for both
    text and VLM models. Mirrors the HF/unsloth API.

    Args:
        trainer: MLXTrainer instance (can be None when return_function=True
            and tokenizer is provided).
        instruction_part: String marking start of user/instruction turns
            (e.g. "<|start_header_id|>user<|end_header_id|>\\n\\n").
        response_part: String marking start of assistant/response turns
            (e.g. "<|start_header_id|>assistant<|end_header_id|>\\n\\n").
        force_match: Match newlines as well (forwarded to HF implementation).
        tokenizer: Optional tokenizer override. If None, uses trainer.tokenizer.
        return_function: If True, return the masking closure without touching
            the trainer.
        num_proc: Accepted for API compatibility with the HF path, unused on MLX.

    Returns:
        The trainer (for chaining), or the masking closure if return_function=True.
    """
    from .dataset_utils import (
        train_on_responses_only as _hf_train_on_responses_only,
    )

    # Resolve tokenizer: kwarg > trainer.tokenizer
    _tokenizer = tokenizer
    if _tokenizer is None and trainer is not None:
        _tokenizer = trainer.tokenizer
    if _tokenizer is None:
        raise ValueError(
            "Unsloth: A tokenizer must be provided either via the `tokenizer` "
            "kwarg or via trainer.tokenizer."
        )

    # Unwrap to get a callable HF tokenizer.
    # mlx-lm: TokenizerWrapper._tokenizer -> HF tokenizer
    # VLM processors: processor.tokenizer -> HF tokenizer
    if hasattr(_tokenizer, "_tokenizer"):
        _tokenizer = _tokenizer._tokenizer
    elif hasattr(_tokenizer, "tokenizer"):
        _tokenizer = _tokenizer.tokenizer

    # Get masking closure from the HF/CUDA implementation
    mask_fn = _hf_train_on_responses_only(
        None,
        instruction_part=instruction_part,
        response_part=response_part,
        force_match=force_match,
        tokenizer=_tokenizer,
        return_function=True,
    )

    if return_function:
        return mask_fn

    if trainer is None:
        raise ValueError(
            "trainer is required when return_function=False. "
            "Pass return_function=True to get the masking closure, "
            "or provide an MLXTrainer instance."
        )

    if trainer._is_vlm:
        # VLM path: store mask_fn for application during batch creation
        trainer._vlm_response_mask_fn = mask_fn
        print("Unsloth: train_on_responses_only enabled (VLM mode).")
    else:
        # Text path: tokenize, mask, and create batches now
        args = trainer.args
        total_batches_needed = (
            args.max_steps * args.gradient_accumulation_steps
            if args.max_steps > 0 else None
        )
        batches = _create_labeled_batches(
            dataset=trainer.train_dataset,
            tokenizer=_tokenizer,
            mask_fn=mask_fn,
            batch_size=args.per_device_train_batch_size,
            max_seq_length=args.max_seq_length,
            formatting_func=trainer.formatting_func,
            dataset_text_field=args.dataset_text_field,
            num_batches=total_batches_needed,
            seed=args.seed,
        )

        # Safety check: detect all-masked labels early
        _check_all_masked(batches)

        trainer._batches = batches

        # Process eval dataset too
        if trainer.eval_dataset is not None:
            eval_batches = _create_labeled_batches(
                dataset=trainer.eval_dataset,
                tokenizer=_tokenizer,
                mask_fn=mask_fn,
                batch_size=args.per_device_train_batch_size,
                max_seq_length=args.max_seq_length,
                formatting_func=trainer.formatting_func,
                dataset_text_field=args.dataset_text_field,
                seed=args.seed,
            )
            trainer._eval_batches_labeled = eval_batches

        print(f"Unsloth: train_on_responses_only enabled "
              f"({len(batches)} batches prepared).")

    return trainer


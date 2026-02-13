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
import time
import gc

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .mlx_utils import (
    make_cce_loss_fn,
    make_baseline_loss_fn,
    has_cce_kernel,
    create_batches,
    save_lora_adapters,
    save_merged_model,
    apply_gradient_checkpointing,
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
    lr_scheduler_type: str = "linear"

    # Optimization
    optim: str = "adafactor"  # "adafactor", "adamw", "adam", "sgd"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # gradient clipping (matches HF Trainer default)
    seed: int = 3407

    # Logging & output
    logging_steps: int = 1
    output_dir: str = "./outputs"
    report_to: str = "none"
    save_steps: int = 0  # 0 = only save at end
    save_total_limit: int = -1  # -1 = no limit

    # SFT-specific (from SFTConfig, for API compat)
    dataset_text_field: str = "text"
    max_seq_length: int = 2048
    packing: bool = False
    dataset_num_proc: int = 2

    # MLX-specific
    use_cce: bool = True   # Use CCE from mlx-cce (key feature)
    compile: bool = False  # mx.compile for graph fusion
    gradient_checkpointing: bool = False  # Recompute activations to save memory


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
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.formatting_func = formatting_func
        self.data_collator = data_collator

        # Use args or defaults
        self.args = args or MLXTrainingConfig()

        # Constructor params override args if provided
        if dataset_text_field is not None:
            self.args.dataset_text_field = dataset_text_field
        if max_seq_length is not None:
            self.args.max_seq_length = max_seq_length
        if packing is not None:
            self.args.packing = packing

        # Training state
        self._global_step = 0
        self._train_loss_history = []
        self._step_times = []  # Per-step wall-clock times
        self._batches = None  # Pre-created batches (skips internal batch creation)

    def _build_optimizer(self):
        """Create MLX optimizer from config."""
        lr = self.args.learning_rate
        wd = self.args.weight_decay

        opt_name = self.args.optim.lower()
        if opt_name == "adafactor":
            return optim.Adafactor(
                learning_rate=lr,
                relative_step=False,
                scale_parameter=False,
            )
        elif opt_name == "adamw":
            return optim.AdamW(learning_rate=lr, weight_decay=wd)
        elif opt_name == "adam":
            return optim.Adam(learning_rate=lr)
        elif opt_name == "sgd":
            return optim.SGD(learning_rate=lr, weight_decay=wd)
        else:
            print(f"Unknown optimizer '{opt_name}', falling back to Adafactor.")
            return optim.Adafactor(
                learning_rate=lr,
                relative_step=False,
                scale_parameter=False,
            )

    def train(self):
        """Run MLX-native training loop.

        Returns:
            dict with training metrics (train_loss, train_runtime, etc.)
        """
        args = self.args
        model = self.model

        # Apply gradient checkpointing if requested
        if args.gradient_checkpointing:
            apply_gradient_checkpointing(model)
            print("Unsloth: Using gradient checkpointing to reduce memory.")

        # Pick loss function
        use_cce = args.use_cce and has_cce_kernel()
        if args.use_cce and not has_cce_kernel():
            print(
                "Unsloth: mx.fast.cce_loss not found. "
                "Install mlx-cce for memory-efficient CCE. "
                "Falling back to standard cross-entropy."
            )

        if use_cce:
            loss_fn = make_cce_loss_fn(model)
            print("Unsloth: Using CCE loss (mx.fast.cce_loss) for memory-efficient training.")
        else:
            loss_fn = make_baseline_loss_fn()
            print("Unsloth: Using standard cross-entropy loss.")

        # Build loss+grad function
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

        # Build optimizer
        optimizer = self._build_optimizer()

        # Build step function (optionally compiled)
        max_grad_norm = args.max_grad_norm
        if args.compile:
            from functools import partial as _partial
            state = [model.state, optimizer.state]

            @_partial(mx.compile, inputs=state, outputs=state)
            def _step(batch, lengths):
                loss, grads = loss_and_grad_fn(model, batch, lengths)
                if max_grad_norm > 0:
                    grads, _ = optim.clip_grad_norm(grads, max_norm=max_grad_norm)
                optimizer.update(model, grads)
                return loss

            def train_step(batch, lengths):
                loss = _step(batch, lengths)
                mx.eval(state)
                return loss

            print("Unsloth: Using mx.compile for graph fusion.")
        else:
            def train_step(batch, lengths):
                loss, grads = loss_and_grad_fn(model, batch, lengths)
                if max_grad_norm > 0:
                    grads, _ = optim.clip_grad_norm(grads, max_norm=max_grad_norm)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state, loss)
                return loss

        # Prepare data — use pre-created batches if available
        if self._batches is not None:
            batches = self._batches
        else:
            total_batches_needed = args.max_steps * args.gradient_accumulation_steps if args.max_steps > 0 else None
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

        if not batches:
            raise ValueError("No training batches created. Check your dataset and batch_size.")

        total_steps = args.max_steps if args.max_steps > 0 else len(batches) // args.gradient_accumulation_steps
        grad_accum = args.gradient_accumulation_steps

        print(f"Unsloth: Training for {total_steps} steps, "
              f"batch_size={args.per_device_train_batch_size}, "
              f"grad_accum={grad_accum}, "
              f"seq_len={args.max_seq_length}")

        # Training loop
        model.train()
        start_time = time.time()
        accumulated_loss = 0.0
        batch_idx = 0

        for step in range(1, total_steps + 1):
            self._global_step = step
            step_loss = 0.0
            step_start = time.time()

            for _micro in range(grad_accum):
                batch, lengths = batches[batch_idx % len(batches)]
                batch_idx += 1

                loss = train_step(batch, lengths)
                step_loss += loss.item()

            step_loss /= grad_accum
            accumulated_loss += step_loss
            self._train_loss_history.append(step_loss)
            self._step_times.append(time.time() - step_start)

            # Logging
            if step % args.logging_steps == 0:
                avg_loss = accumulated_loss / step
                elapsed = time.time() - start_time
                print(
                    f"  Step {step}/{total_steps} | "
                    f"Loss: {step_loss:.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

            # Checkpointing
            if args.save_steps > 0 and step % args.save_steps == 0:
                ckpt_dir = f"{args.output_dir}/checkpoint-{step}"
                save_lora_adapters(model, ckpt_dir)
                print(f"  Saved checkpoint to {ckpt_dir}")

        total_time = time.time() - start_time
        avg_loss = accumulated_loss / total_steps if total_steps > 0 else 0.0

        print(f"\nUnsloth: Training complete! "
              f"Avg loss: {avg_loss:.4f} | "
              f"Total time: {total_time:.1f}s | "
              f"Steps: {total_steps}")

        return {
            "train_loss": avg_loss,
            "train_runtime": total_time,
            "train_steps": total_steps,
            "train_samples_per_second": (total_steps * args.per_device_train_batch_size * grad_accum) / total_time if total_time > 0 else 0,
        }

    def save_model(self, output_dir=None):
        """Save LoRA adapters (or full model if no LoRA)."""
        output_dir = output_dir or self.args.output_dir
        save_lora_adapters(self.model, output_dir, adapter_config={
            "learning_rate": self.args.learning_rate,
            "max_steps": self.args.max_steps,
            "max_seq_length": self.args.max_seq_length,
            "use_cce": self.args.use_cce,
        })
        self.tokenizer.save_pretrained(output_dir)
        print(f"Unsloth: Model saved to {output_dir}")

    def save_merged_model(self, output_dir=None):
        """Fuse LoRA weights and save the full merged model."""
        output_dir = output_dir or self.args.output_dir
        save_merged_model(self.model, self.tokenizer, output_dir)
        print(f"Unsloth: Merged model saved to {output_dir}")

    def push_to_hub(self, repo_id, **kwargs):
        """Upload model to HuggingFace Hub."""
        try:
            from mlx_lm import upload_to_hub
            upload_to_hub(
                model=self.model,
                tokenizer=self.tokenizer,
                repo_id=repo_id,
                **kwargs,
            )
            print(f"Unsloth: Uploaded to https://huggingface.co/{repo_id}")
        except ImportError:
            # Fallback: save locally then use huggingface_hub
            import tempfile
            from huggingface_hub import HfApi
            with tempfile.TemporaryDirectory() as tmp:
                self.save_model(tmp)
                api = HfApi()
                api.upload_folder(folder_path=tmp, repo_id=repo_id, **kwargs)
            print(f"Unsloth: Uploaded to https://huggingface.co/{repo_id}")

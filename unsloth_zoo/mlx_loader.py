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
Lightweight FastLanguageModel for Apple Silicon / MLX.

No GPU dependencies — uses mlx-lm for model loading and LoRA.
This avoids importing unsloth.models (which pulls in CUDA kernels).
"""


class FastMLXModel:
    """MLX model loader for Apple Silicon.

    Mirrors the unsloth GPU API so notebooks work with minimal changes:
        model, tokenizer = FastLanguageModel.from_pretrained(...)
        model = FastLanguageModel.get_peft_model(model, r=16)

    Pass any HuggingFace model name directly — mlx-lm handles loading:
        "mlx-community/Llama-3.2-1B-Instruct-4bit"   (pre-quantized MLX)
        "mlx-community/Llama-3.2-1B-Instruct-8bit"   (8-bit MLX)
        "meta-llama/Llama-3.2-1B-Instruct"            (full precision)
        "Qwen/Qwen2.5-7B-Instruct"                    (any HF model)
    """

    @staticmethod
    def from_pretrained(
        model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
        token=None,
        trust_remote_code=False,
        **kwargs,  # Accept and ignore GPU-only kwargs
    ):
        """Load a model via mlx-lm on Apple Silicon.

        Args:
            model_name: Any HuggingFace repo name. mlx-lm supports
                pre-converted MLX models (mlx-community/*) and standard
                PyTorch models (converted on-the-fly).
            max_seq_length: Maximum sequence length for training.
            load_in_4bit: Accepted for API compat with CUDA unsloth.
                On MLX, quantization is determined by the model repo itself.
            token: HuggingFace token for gated models.
        """
        try:
            from mlx_lm import load as mlx_load
        except ImportError:
            raise ImportError(
                "Unsloth: mlx-lm is required for Apple Silicon. "
                "Install via: pip install unsloth-zoo[mlx]"
            )

        print(f"Unsloth: Loading {model_name} via mlx-lm...")

        tokenizer_config = {}
        if token:
            tokenizer_config["token"] = token

        model, tokenizer, config = mlx_load(
            model_name,
            tokenizer_config=tokenizer_config if tokenizer_config else None,
            return_config=True,
        )

        # Stash metadata on the model for saving later
        model._config = config
        model._hf_repo = model_name
        # Resolve local cache path for copying auxiliary files during save
        try:
            from mlx_lm.utils import _download
            model._src_path = str(_download(model_name))
        except Exception:
            model._src_path = None

        model.max_seq_length = max_seq_length
        return model, tokenizer

    @staticmethod
    def get_peft_model(
        model,
        r=16,
        target_modules=None,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        max_seq_length=2048,
        **kwargs,  # Accept and ignore GPU-only kwargs
    ):
        """Apply LoRA via mlx-lm on Apple Silicon."""
        try:
            from mlx_lm.tuner.utils import linear_to_lora_layers
        except ImportError:
            raise ImportError(
                "Unsloth: mlx-lm is required for LoRA on Apple Silicon. "
                "Install via: pip install unsloth-zoo[mlx]"
            )

        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

        lora_config = {
            "rank": r,
            "alpha": lora_alpha,
            "dropout": 0.0,
            "scale": lora_alpha / r,
        }

        num_layers = 0
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            num_layers = len(model.model.layers)

        linear_to_lora_layers(
            model,
            num_layers=num_layers,
            config=lora_config,
            use_dora=False,
        )

        # Freeze all, then selectively unfreeze LoRA weights
        model.freeze()
        model.unfreeze(keys=["lora_a", "lora_b"], strict=False)

        import mlx.utils
        trainable = sum(v.size for _, v in mlx.utils.tree_flatten(model.trainable_parameters()))
        total = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
        pct = 100.0 * trainable / total if total > 0 else 0
        print(
            f"Unsloth: LoRA applied — {trainable:,} trainable params "
            f"({pct:.2f}% of {total:,} total)"
        )
        return model


# Aliases for backward compat
FastLanguageModel = FastMLXModel
FastModel = FastMLXModel

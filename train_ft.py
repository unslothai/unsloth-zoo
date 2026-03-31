"""Full fine-tuning: all weights trainable, no LoRA."""

from unsloth_zoo.mlx_loader import FastLanguageModel
from unsloth_zoo.mlx_trainer import MLXTrainer, MLXTrainingConfig
from unsloth_zoo.mlx_utils import save_merged_model
from datasets import load_dataset

MODEL = "mlx-community/SmolLM2-135M-Instruct"
OUTPUT_DIR = "trained_models/ft"

# Load model (all weights will be trainable)
model, tokenizer = FastLanguageModel.from_pretrained(MODEL, max_seq_length=512)

# NO get_peft_model — all weights stay trainable

# Small dataset for demo
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:200]")

def format_sample(example):
    if example.get("input"):
        return f"### Instruction:\n{example['instruction']}\n### Input:\n{example['input']}\n### Response:\n{example['output']}"
    return f"### Instruction:\n{example['instruction']}\n### Response:\n{example['output']}"

trainer = MLXTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=format_sample,
    args=MLXTrainingConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=5,
        learning_rate=1e-5,  # lower LR for full FT
        max_seq_length=512,
        output_dir=OUTPUT_DIR,
        logging_steps=5,
        use_cce=True,
        gradient_checkpointing=True,
    ),
)

print(f"\n{'='*50}")
print(f"Full Fine-Tuning: {MODEL}")
print(f"{'='*50}\n")

trainer.train()

# Save the full model
save_merged_model(model, tokenizer, OUTPUT_DIR)
print(f"\nModel saved to {OUTPUT_DIR}/")

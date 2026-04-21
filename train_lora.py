"""LoRA training: full-precision base model + LoRA adapters."""

from unsloth_zoo.mlx_loader import FastLanguageModel
from unsloth_zoo.mlx_trainer import MLXTrainer, MLXTrainingConfig
from datasets import load_dataset

MODEL = "mlx-community/SmolLM2-135M-Instruct"
OUTPUT_DIR = "trained_models/lora"

# Load full-precision model
model, tokenizer = FastLanguageModel.from_pretrained(MODEL, max_seq_length=512)

# Apply LoRA adapters
model = FastLanguageModel.get_peft_model(model, r=8, lora_alpha=16)

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
        learning_rate=2e-4,
        max_seq_length=512,
        output_dir=OUTPUT_DIR,
        logging_steps=5,
        use_cce=True,
    ),
)

print(f"\n{'='*50}")
print(f"LoRA Training: {MODEL}")
print(f"{'='*50}\n")

trainer.train()

# Save: both LoRA adapters and merged model
trainer.save_model(f"{OUTPUT_DIR}_adapters")  # just adapters
trainer.save_pretrained_merged(OUTPUT_DIR)     # fully merged
print(f"\nAdapters saved to {OUTPUT_DIR}_adapters/")
print(f"Merged model saved to {OUTPUT_DIR}/")

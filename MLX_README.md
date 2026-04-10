# MLX Training & Export (Apple Silicon)

Train and export LLMs on Apple Silicon using MLX + unsloth-zoo.

## Setup

```bash
# Create venv with Python 3.10+
python3 -m venv venv
source venv/bin/activate

# Install MLX from local source (with CCE support)
cd ../mlx && pip install -e ".[dev]" && cd ../unsloth-zoo

# Install unsloth-zoo + dependencies
pip install -e .
pip install mlx-lm datasets
```

## Training

Three training modes are supported. All scripts use `SmolLM2-135M-Instruct` by default — change `MODEL` to use any HuggingFace or mlx-community model.

### QLoRA (4-bit quantized base + LoRA)
```bash
python train_qlora.py
# Output: trained_models/qlora/
```

### LoRA (full-precision base + LoRA)
```bash
python train_lora.py
# Output: trained_models/lora/          (merged model)
#         trained_models/lora_adapters/ (adapters only)
```

### Full Fine-Tuning (all weights trainable)
```bash
python train_ft.py
# Output: trained_models/ft/
```

## GGUF Export

Export any trained model to GGUF format for use with llama.cpp, Ollama, etc.

```bash
# Default (q8_0 — good balance of size and quality)
python export_gguf.py trained_models/qlora

# Quantized (q4_k_m — smallest, fast inference)
python export_gguf.py trained_models/lora --quant q4_k_m

# Unquantized (bf16 — full precision)
python export_gguf.py trained_models/ft --quant not_quantized

# Custom output directory
python export_gguf.py trained_models/ft --quant q8_0 --output my_gguf_export/
```

### Quantization Options

| Option | Alias | Description |
|--------|-------|-------------|
| `not_quantized` | bf16 | Full precision, largest file |
| `fast_quantized` | q8_0 | 8-bit, good quality (default) |
| `quantized` | q4_k_m | 4-bit, smallest file |
| `q2_k` | - | 2-bit, aggressive compression |
| `q3_k_m` | - | 3-bit |
| `q5_k_m` | - | 5-bit |
| `q6_k` | - | 6-bit |
| `f16` | - | Float16 |

## Output Structure

```
trained_models/
├── qlora/                    # QLoRA trained model (HF-compatible)
│   ├── config.json
│   ├── model.safetensors
│   ├── model.safetensors.index.json
│   ├── tokenizer.json
│   └── ...
├── qlora_gguf/               # GGUF export
│   └── model.BF16.gguf
├── lora/                     # LoRA merged model
├── lora_adapters/            # LoRA adapters only (small)
│   ├── adapters.safetensors
│   └── adapter_config.json
├── lora_gguf/
│   └── model.Q4_K_M.gguf
├── ft/                       # Full fine-tune model
└── ft_gguf/
    └── model.BF16.gguf
```

## Programmatic Usage

```python
from unsloth_zoo.mlx_loader import FastLanguageModel
from unsloth_zoo.mlx_trainer import MLXTrainer, MLXTrainingConfig
from unsloth_zoo.mlx_utils import save_merged_model, save_pretrained_gguf

# Load
model, tokenizer = FastLanguageModel.from_pretrained("mlx-community/Llama-3.2-1B-Instruct-4bit")
model = FastLanguageModel.get_peft_model(model, r=16)

# Train
trainer = MLXTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset,
                     args=MLXTrainingConfig(max_steps=100, use_cce=True))
trainer.train()

# Save
trainer.save_pretrained_merged("my_model")           # HF-compatible safetensors
trainer.save_pretrained_gguf("my_gguf", quantization_method="q4_k_m")  # GGUF

# Or save from a previously trained model directory
save_pretrained_gguf(model, tokenizer, "output_gguf", quantization_method="q8_0")
```

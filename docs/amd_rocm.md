# AMD ROCm Support

Unsloth supports AMD GPUs (Instinct MI300X, MI325X, MI355X, Radeon RX 7900 series and newer)
on Linux, WSL, and Windows via ROCm.

## Quick Start

```bash
# Install ROCm PyTorch (replace rocm6.2 with your ROCm version)
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

# Install Unsloth
pip install unsloth
```

Then train exactly as you would on NVIDIA - no code changes needed.

## Recommended Configuration for AMD Instinct (MI300X / MI325X)

The following settings are benchmarked on AMD Instinct MI325X (256 GB HBM3e, gfx942, ROCm 6.2).

### 1. Use SDPA attention (most important)

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3-8B",
    attn_implementation = "sdpa",   # recommended on ROCm
)
```

PyTorch SDPA routes to MIOpen fused attention on ROCm - functionally equivalent
to flash_attention_2 with zero extra packages.

Benchmark results (MI325X, TinyLlama-1.1B, inference):

| Implementation | Throughput     | VRAM    |
|----------------|----------------|---------|
| sdpa           | 108,505 tok/s  | 2.22 GB |
| eager          | 78,914 tok/s   | 2.45 GB |

SDPA is +37% faster and uses 9% less VRAM than eager.

At sequence lengths >= 2048, SDPA is effectively required. Eager attention allocates
an O(n^2) attention matrix and runs out of memory approximately 3x sooner than SDPA.
At seq=2048, SDPA gives +41% throughput over eager.

### 2. Scale batch size - AMD GPUs have large VRAM

AMD Instinct MI300X/MI325X have 192-256 GB HBM3e. Default batch_size=4 uses less
than 5% of available VRAM. Scale up for free throughput gains:

| Batch size  | Throughput     | VRAM used              |
|-------------|----------------|------------------------|
| 4 (default) | 27,418 tok/s   | 10.6 GB (4% of 256 GB) |
| 16          | 41,271 tok/s   | 20.5 GB (8% of 256 GB) |

+51% throughput with no other changes. Rule of thumb: start at batch=16 on
MI300X/MI325X and increase until VRAM is around 50% utilized.

```python
trainer = SFTTrainer(
    ...
    args = TrainingArguments(
        per_device_train_batch_size = 16,   # was 4
        ...
    ),
)
```

### 3. Target all projection matrices in LoRA

Targeting q_proj, k_proj, v_proj, and o_proj (full QKV+O) instead of only
q_proj + v_proj reduces VRAM by 28% at essentially the same throughput:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 32,
    lora_dropout = 0.05,
)
```

| Target modules            | Throughput   | VRAM    |
|---------------------------|--------------|---------|
| q_proj + v_proj (default) | 27,418 tok/s | 10.6 GB |
| q+k+v+o (full QKV+O)      | 26,793 tok/s | 7.6 GB  |

### 4. Gradient checkpointing for very large models

```python
model = FastLanguageModel.get_peft_model(
    model,
    use_gradient_checkpointing = "unsloth",
    ...
)
```

With full QKV+O LoRA + gradient checkpointing, VRAM drops to 3.1 GB for a
1B-parameter model - enabling very long context or large batch training on any
AMD Instinct GPU.

### 5. Account for ROCm JIT warm-up in benchmarks

ROCm compiles HIP kernels on first use. Throughput ramps significantly over the
first ~15 training steps:

| Step | Tok/s |
|------|-------|
| 1    | 78    |
| 5    | 386   |
| 10   | 756   |
| 15   | 1,111 |
| 20+  | 1,454 |

Exclude steps 1-10 from benchmark measurements. Set
MIOPEN_USER_DB_PATH=/path/to/persistent/cache to avoid recompilation across runs.

## Validated Hardware

| GPU                 | Architecture | VRAM         | ROCm | Status      |
|---------------------|--------------|--------------|------|-------------|
| AMD Instinct MI325X | gfx942       | 256 GB HBM3e | 6.2+ | Validated   |
| AMD Instinct MI300X | gfx942       | 192 GB HBM3  | 6.2+ | Validated   |
| AMD Instinct MI355X | gfx950       | 288 GB HBM3e | 7.0+ | Validated   |

## Troubleshooting

**torch.cuda.is_available() returns False**

ROCm PyTorch aliases torch.cuda - ensure you installed the ROCm wheel:
```bash
python -c "import torch; print(torch.version.hip)"  # should print ROCm version
```

**FlashInfer errors on ROCm**

FlashInfer requires NVIDIA nvcc and is automatically skipped on ROCm. No action needed.

**Slow first run**

ROCm JIT kernel compilation on first use is normal. See the warm-up note in section 5.

**UNSLOTH_IS_PRESENT error**

If running unsloth-zoo directly without the unsloth package installed:
```bash
export UNSLOTH_IS_PRESENT=1
```

## Running the AMD Validation Suite

```bash
# Quick checks - no model download, ~10 seconds
UNSLOTH_IS_PRESENT=1 python -m pytest tests/test_rocm_compatibility.py -v
```

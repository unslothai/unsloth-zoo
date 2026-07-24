# AMD ROCm Support

Unsloth supports AMD GPUs (Instinct MI300X, MI325X, MI355X, Radeon RX 7900 series and newer)
on Linux, WSL, and Windows via ROCm.

## Quick Start

**Linux / WSL:**

The `download.pytorch.org/whl/` index may not carry a wheel for every ROCm minor version.
Use the version resolver in `unsloth_zoo/device_type.py` (which checks available indices
and falls back to the nearest supported version), or check the
[PyTorch ROCm wheel index list](https://download.pytorch.org/whl/) directly for an
available `rocmX.Y` entry that matches your installed ROCm.

Install `torch`, `torchvision`, and `torchaudio` together from the ROCm index so all
three come from the same ROCm build (mixing ROCm and default-PyPI wheels causes
incompatible-import errors, especially for multimodal workloads):

```bash
# Example for ROCm 6.2; substitute the nearest available index for your ROCm version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install unsloth
```

**Windows (native ROCm):**
AMD provides ROCm-enabled PyTorch wheels for Windows via the TheRock project.
Follow the [AMD ROCm Windows installation guide](https://rocm.docs.amd.com/en/latest/install/windows.html)
to install the correct wheel for your GPU and ROCm version, then:
```bash
pip install unsloth
```

Then train exactly as you would on NVIDIA - no code changes needed.

---

## Recommended Configuration for AMD Instinct (MI300X / MI325X)

The following settings are benchmarked on AMD Instinct MI325X (256 GB HBM3e, gfx942,
ROCm 6.2, TinyLlama-1.1B, LoRA r=16, batch=4, seq=512, bfloat16).

### 1. Use SDPA attention (most important)

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3-8B",
    attn_implementation = "sdpa",   # recommended on ROCm
)
```

PyTorch SDPA is a dispatcher that selects the best available kernel for the
given inputs. On ROCm with the benchmark inputs (bfloat16, standard head dimensions,
no custom attention mask), SDPA dispatches to MIOpen's fused attention kernel,
which delivers flash_attention_2-level performance with zero extra packages.
For inputs with unsupported dtypes or mask shapes, SDPA may fall back to a
math or memory-efficient implementation with different performance characteristics.

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

AMD Instinct MI300X/MI325X have 192-256 GB HBM3e. At the benchmark workload
(TinyLlama-1.1B, LoRA r=16, seq=512, bfloat16), default batch=4 uses less than
5% of available VRAM:

| Batch size  | Throughput     | VRAM used              |
|-------------|----------------|------------------------|
| 4 (default) | 27,418 tok/s   | 10.6 GB (4% of 256 GB) |
| 16          | 41,271 tok/s   | 20.5 GB (8% of 256 GB) |

**+51% throughput for this benchmark workload.** These numbers are specific to the
benchmark conditions above. For larger models or longer sequences, batch=16 may
not fit — start from a batch size you know fits your workload and increase from
there until VRAM is around 50% utilized.

```python
trainer = SFTTrainer(
    ...
    args = TrainingArguments(
        per_device_train_batch_size = 16,   # was 4 — adjust to your model/seq
        ...
    ),
)
```

### 3. Use the standard full-module LoRA target set

Use Unsloth's recommended full target_modules (attention + MLP) with lora_dropout=0
for the optimized training path:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,    # 0 is the optimized path in Unsloth
    bias = "none",       # "none" is the optimized path in Unsloth
    use_gradient_checkpointing = "unsloth",
)
```

**Note on attention-only LoRA (q+k+v+o):** Targeting only the four attention
projection matrices saves ~28% VRAM compared to attention-only baselines but
excludes MLP layers from training. This trades model quality for memory and
is not equivalent to the standard configuration above.

### 4. Use gradient checkpointing for very large models

```python
model = FastLanguageModel.get_peft_model(
    model,
    use_gradient_checkpointing = "unsloth",
    ...
)
```

With full LoRA + gradient checkpointing, VRAM drops dramatically, enabling
very long context or large model training on any AMD Instinct GPU.

### 5. Account for ROCm JIT warm-up in benchmarks

ROCm compiles HIP kernels on first use. Throughput ramps significantly over the
first ~20 training steps before reaching steady state:

| Step | Tok/s  |
|------|--------|
| 1    | 78     |
| 5    | 386    |
| 10   | 756    |
| 15   | 1,111  |
| 20+  | 1,454  |

**Exclude steps 1-20 from benchmark measurements** — throughput is still rising
at step 10 (756 tok/s vs 1,454 steady-state). Start measuring only after
throughput has plateaued (step 20+ in the benchmark above).

### 6. Persist compiled kernels and Unsloth artifacts across runs

ROCm and Unsloth each have their own compilation artifacts. Set both to avoid
recompilation costs on every restart:

**Linux / WSL:**
```bash
# MIOpen compiled kernel binaries (HIP kernel compilation cost)
export MIOPEN_CUSTOM_CACHE_DIR=/path/to/persistent/miopen_cache

# MIOpen performance tuning database (separate from kernel binaries)
export MIOPEN_USER_DB_PATH=/path/to/persistent/miopen_userdb

# Unsloth Mega-cache: persists Dynamo/AOTAutograd/Inductor/Triton artifacts
# (torch >= 2.7 required; enabled by default on POSIX)
export UNSLOTH_MEGA_CACHE_DIR=/path/to/persistent/unsloth_mega_cache
```

**Windows (native ROCm):**
```powershell
$env:MIOPEN_CUSTOM_CACHE_DIR = "C:\path\to\persistent\miopen_cache"
$env:MIOPEN_USER_DB_PATH     = "C:\path\to\persistent\miopen_userdb"
# Mega-cache defaults to DISABLED on non-POSIX — must opt in explicitly
$env:UNSLOTH_MEGA_CACHE      = "1"
$env:UNSLOTH_MEGA_CACHE_DIR  = "C:\path\to\persistent\unsloth_mega_cache"
```

Notes:
- `MIOPEN_USER_DB_PATH` alone does not avoid kernel recompilation — set
  `MIOPEN_CUSTOM_CACHE_DIR` for that.
- `TORCHINDUCTOR_CACHE_DIR` is cleared internally by Unsloth's compile setup
  (`patch_torch_compile` in `patching_utils.py`). Use `UNSLOTH_MEGA_CACHE_DIR`
  instead to persist Inductor/AOTAutograd/Triton artifacts across runs (torch >= 2.7).
- On Linux/WSL, Mega-cache is enabled by default when `UNSLOTH_MEGA_CACHE_DIR` is set.
  On Windows, you must also set `UNSLOTH_MEGA_CACHE=1` to opt in (non-POSIX platforms
  default to disabled; see `unsloth_zoo/compile_cache.py`).
- On torch < 2.7, Mega-cache is not available; the one-time compile cost is paid
  on every new process.

---

## Validated Hardware

| GPU                 | Architecture | VRAM         | ROCm | Status    |
|---------------------|--------------|--------------|------|-----------|
| AMD Instinct MI325X | gfx942       | 256 GB HBM3e | 6.2+ | Validated |
| AMD Instinct MI300X | gfx942       | 192 GB HBM3  | 6.2+ | Validated |
| AMD Instinct MI355X | gfx950       | 288 GB HBM3e | 7.0+ | Validated |

---

## Troubleshooting

**torch.cuda.is_available() returns False**

ROCm PyTorch aliases torch.cuda - ensure you installed the ROCm wheel:
```bash
python -c "import torch; print(torch.version.hip)"  # should print ROCm version
```

**FlashInfer errors on ROCm**

FlashInfer requires NVIDIA nvcc and is automatically skipped on ROCm. No action needed.

**Slow first run / recompilation on every restart**

See section 6 for how to persist MIOpen and Unsloth compilation artifacts.

**ImportError: unsloth package not found**

`unsloth_zoo` requires the `unsloth` package to be installed. The env var
`UNSLOTH_IS_PRESENT` does not bypass this check — install `unsloth` directly:
```bash
pip install unsloth
```

---

## Running the AMD Validation Suite

The mock-based test suite (added in PR #943) runs on every CI build without AMD hardware:

```bash
UNSLOTH_IS_PRESENT=1 python -m pytest tests/test_rocm_compatibility.py -v
```

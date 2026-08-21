# Tests

Pytest suite for `unsloth_zoo`. The whole tree runs in roughly 8 seconds on a
Linux+CPU box. None of the tests need a GPU, ROCm runtime, Apple Silicon, or
Metal install: the hardware-aware suites simulate every accelerator family by
spoofing `torch.cuda.is_available`, `torch.xpu.is_available`,
`torch.backends.mps.is_available`, `torch.version.hip`, `platform.system`,
`platform.machine`, and `sys.modules['mlx']`.

## Layout

```
tests/
  conftest.py                                 GPU-free harness + sys.path setup
  mlx_simulation/                             torch-on-MLX shim package
  test_active_merge_device_matrix.py          accelerator cascade matrix
  test_backend_device_helpers.py              device_synchronize edge cases
  test_forward_native_moe_loop_lora.py        MoE LoRA forward parity
  test_gemma4_moe_lora_registration.py        Gemma-4 MoE LoRA extractor wiring
  test_mlx_torch_shim_smoke.py                shim infrastructure smoke
  test_moe_lora_extractor_coverage.py         MoE LoRA extractor coverage
  test_pr_a_components.py                     CCE forward / dequant / VJP / monkey-patches
  test_pr_a_deep_components.py                MLXTrainingConfig / mlx_compile / SDPA / tree
  test_pr_a_dequantize.py                     end-to-end _dequantize_selected_mlx_modules
  test_pr_a_gated_delta.py                    gated_delta_ops_efficient forward + backward
  test_pr_a_imports.py                        PR-A module import surface
  test_qwen_moe_lora_extractor.py             Qwen MoE LoRA extractor
  test_unsloth_zoo_lora_merge.py              pure-torch LoRA + MoE merge regression
```

## Quick start

```bash
pip install -e . pytest torch --index-url https://download.pytorch.org/whl/cpu
pytest tests/ -v
```

## Hardware-aware tests without hardware

Two parametrized suites cover every supported accelerator combination on a
single CPU box:

### `test_active_merge_device_matrix.py`

Drives `unsloth_zoo.saving_utils._active_merge_device()` through every
profile of the `cuda > xpu > mps > cpu` cascade. Profiles include
`nvidia_cuda`, `amd_rocm` (cuda + `torch.version.hip="6.1"`), `intel_xpu`,
`apple_silicon_mps`, and `cpu_only`. Plus three negative-space canaries:

- `test_cuda_takes_priority_over_xpu_mps` (cuda + xpu + mps -> cuda)
- `test_xpu_takes_priority_over_mps` (xpu + mps -> xpu)
- `test_lru_cache_freezes_first_result` (verifies cache discipline)

To extend coverage, add a row to the `PROFILES` list inside the file. Pytest's
parametrize automatically picks up new entries and asserts:

1. `_active_merge_device()` returns the right backend string.
2. The `@functools.lru_cache(maxsize=1)` is cleared before AND after each
   test so subsequent tests in the session see a fresh probe.

### `test_mlx_torch_shim_smoke.py` and the `test_pr_a_*` family

Activate `mlx_simulation/` to install torch-backed shims for `mlx`, `mlx.core`,
`mlx.nn`, `mlx.optimizers`, `mlx.utils`, `mlx_lm`, and `mlx_vlm`, then
exercise every PR-A code path on Linux+CPU. The shim is opt-in: a test must
explicitly call `simulate_mlx_on_torch()` to activate it. Production
`import unsloth_zoo` is unaffected and pulls zero MLX modules into
`sys.modules` on a non-Apple host.

## Running a subset

```bash
# accelerator cascade only
pytest tests/test_active_merge_device_matrix.py -v

# MLX shim infrastructure only
pytest tests/test_mlx_torch_shim_smoke.py -v

# all PR-A simulation tests
pytest tests/test_pr_a_*.py tests/test_mlx_torch_shim_smoke.py -v

# pre-PR LoRA / MoE merge regression suite
pytest tests/test_unsloth_zoo_lora_merge.py tests/test_forward_native_moe_loop_lora.py -v
```

## How the GPU-free harness works

`tests/conftest.py` pre-loads `unsloth_zoo.device_type` under a temporarily
True `torch.cuda.is_available()` so the package import chain succeeds on a
runner without CUDA / XPU / HIP visible. The `@functools.cache` on
`get_device_type` permanently captures `"cuda"` and the patch is reverted
before any test runs. When a real accelerator IS available the pre-load is
skipped and detection runs normally. The conftest also adds `tests/` to
`sys.path` so the bundled `mlx_simulation` shim package is importable.

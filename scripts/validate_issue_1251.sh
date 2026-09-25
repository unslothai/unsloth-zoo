#!/usr/bin/env bash
# CPU regression bundle for unsloth-zoo #1251 (PRs #1294–#1296).
set -euo pipefail
cd "$(dirname "$0")/.."
export UNSLOTH_ZOO_DISABLE_GPU_INIT=1
export UNSLOTH_ALLOW_CPU=1
export UNSLOTH_IS_PRESENT=1

echo "=== #1251 CPU regression tests ==="
python3 -m pytest \
  tests/test_gpt_oss_mxfp4_expert_property_dequantize.py \
  tests/test_mxfp4_training_config_env.py \
  tests/test_gpt_oss_mxfp4_lora_forward_dispatch.py \
  tests/test_mxfp4_load_path_layout.py \
  -q "$@"

echo "=== file-based dispatch tests (no pytest) ==="
python3 - <<'PY'
import importlib.util

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

base = "tests"
for fn in (
    "test_gpt_oss_mxfp4_lora_forward_dispatch",
    "test_gpt_oss_mxfp4_expert_property_dequantize",
):
    m = load(fn, f"{base}/{fn}.py")
    for attr in sorted(dir(m)):
        if attr.startswith("test_"):
            getattr(m, attr)()
print("file-based OK")
PY

echo "=== optional GPU smoke (skipped if no CUDA/triton_kernels) ==="
python3 -m pytest tests/test_mxfp4_issue_1251_gpu_smoke.py -q "$@" || true

echo "Done."

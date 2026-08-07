PYTHON ?= python3

.PHONY: help test test-fast test-matrix test-mlx-sim test-merge install-test format

help:
	@echo "Targets:"
	@echo "  install-test   pip install -e . + pytest + CPU torch (no GPU needed)"
	@echo "  test           run the full pytest suite under tests/"
	@echo "  test-fast      run the full suite quietly (one line per file)"
	@echo "  test-matrix    accelerator cascade dispatch matrix only"
	@echo "  test-mlx-sim   MLX-on-torch shim suite only (PR-A regression)"
	@echo "  test-merge     pure-torch LoRA / MoE merge regression only (Tier 0)"
	@echo "  format         run ruff format + kwarg spacing pipeline"

install-test:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -e . pytest

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-fast:
	$(PYTHON) -m pytest tests/ --tb=line -q

test-matrix:
	$(PYTHON) -m pytest tests/test_active_merge_device_matrix.py -v

test-mlx-sim:
	$(PYTHON) -m pytest tests/test_mlx_torch_shim_smoke.py tests/test_pr_a_*.py -v

test-merge:
	$(PYTHON) -m pytest tests/test_unsloth_zoo_lora_merge.py tests/test_forward_native_moe_loop_lora.py -v

format:
	$(PYTHON) scripts/run_ruff_format.py $$(git ls-files '*.py')

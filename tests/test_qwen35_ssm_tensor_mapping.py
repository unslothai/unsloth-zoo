# Tests for _patch_tensor_mapping_for_qwen35: inserting Qwen3.5 linear_attn
# aliases into a stale llama.cpp gguf-py/gguf/tensor_mapping.py.

from __future__ import annotations

import ast
import importlib.util
import os
import sys
from pathlib import Path


def _load_llama_cpp_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "unsloth_zoo" / "llama_cpp.py"
    spec = importlib.util.spec_from_file_location("llama_cpp_under_test_qwen35", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


llama_cpp = _load_llama_cpp_module()


# Mirrors a qwen3next-era checkout: shared linear_attn names already present,
# Qwen3.5 split projections (in_proj_qkv/z/a/b) missing.
STALE_TENSOR_MAPPING = '''class TensorNameMap:
    mappings_cfg = {
        MODEL_TENSOR.ATTN_QKV: (
            "model.layers.{bid}.self_attn.qkv_proj",  # phi3
        ),

        MODEL_TENSOR.ATTN_GATE: (
            "model.layers.{bid}.self_attn.gate_proj",  # afmoe
        ),

        MODEL_TENSOR.SSM_CONV1D: (
            "model.layers.{bid}.conv1d",               # mamba-hf
            "model.layers.{bid}.linear_attn.conv1d",   # qwen3next
        ),

        MODEL_TENSOR.SSM_DT: (
            "model.layers.{bid}.dt_proj",               # mamba-hf
            "model.layers.{bid}.linear_attn.dt_proj",   # qwen3next
        ),

        MODEL_TENSOR.SSM_A: (
            "model.layers.{bid}.A_log",               # mamba-hf
            "model.layers.{bid}.linear_attn.A_log",   # qwen3next
        ),

        MODEL_TENSOR.SSM_NORM: (
            "model.layers.{bid}.mamba.norm",        # falcon-h1
            "model.layers.{bid}.linear_attn.norm",  # qwen3next
        ),

        MODEL_TENSOR.SSM_OUT: (
            "model.layers.{bid}.out_proj",               # mamba-hf
            "model.layers.{bid}.linear_attn.out_proj",   # qwen3next
        ),

        MODEL_TENSOR.SSM_BETA: (
            "model.layers.{bid}.self_attn.b_proj",       # Kimi Linear
        ),

        MODEL_TENSOR.SSM_ALPHA: (
        ),
    }
'''


def _write_stale(tmpdir, content = STALE_TENSOR_MAPPING):
    gguf_dir = os.path.join(tmpdir, "gguf-py", "gguf")
    os.makedirs(gguf_dir, exist_ok = True)
    path = os.path.join(gguf_dir, "tensor_mapping.py")
    with open(path, "w", encoding = "utf-8") as f:
        f.write(content)
    return path


def _block(content, name):
    start = content.index(f"MODEL_TENSOR.{name}: (")
    return content[start : content.index("),", start)]


def test_inserts_missing_qwen35_aliases(tmp_path):
    path = _write_stale(str(tmp_path))
    llama_cpp._patch_tensor_mapping_for_qwen35(str(tmp_path))
    result = Path(path).read_text(encoding = "utf-8")

    # The split GDN projections land in their exact blocks.
    assert "linear_attn.in_proj_qkv" in _block(result, "ATTN_QKV")
    assert "linear_attn.in_proj_z" in _block(result, "ATTN_GATE")
    assert "linear_attn.in_proj_b" in _block(result, "SSM_BETA")
    assert "linear_attn.in_proj_a" in _block(result, "SSM_ALPHA")
    # Names already covered by qwen3next entries are not duplicated.
    for shared in ("conv1d", "dt_proj", "A_log", "norm", "out_proj"):
        assert result.count(f'"model.layers.{{bid}}.linear_attn.{shared}"') == 1
    # Patched file stays valid Python.
    ast.parse(result)


def test_idempotent_and_no_op_when_current(tmp_path):
    path = _write_stale(str(tmp_path))
    llama_cpp._patch_tensor_mapping_for_qwen35(str(tmp_path))
    once = Path(path).read_text(encoding = "utf-8")
    llama_cpp._patch_tensor_mapping_for_qwen35(str(tmp_path))
    assert Path(path).read_text(encoding = "utf-8") == once
    for _, name in llama_cpp._QWEN35_TENSOR_MAPPINGS:
        assert once.count(f'"{name}"') == 1


def test_partial_patch_completes_missing_entries(tmp_path):
    # A file already holding one qwen3.5 alias still receives the rest.
    partial = STALE_TENSOR_MAPPING.replace(
        '            "model.layers.{bid}.self_attn.b_proj",       # Kimi Linear\n',
        '            "model.layers.{bid}.linear_attn.in_proj_b",  # qwen3.5\n'
        '            "model.layers.{bid}.self_attn.b_proj",       # Kimi Linear\n',
    )
    path = _write_stale(str(tmp_path), partial)
    llama_cpp._patch_tensor_mapping_for_qwen35(str(tmp_path))
    result = Path(path).read_text(encoding = "utf-8")
    assert result.count('"model.layers.{bid}.linear_attn.in_proj_b"') == 1
    assert "linear_attn.in_proj_qkv" in _block(result, "ATTN_QKV")
    assert "linear_attn.in_proj_a" in _block(result, "SSM_ALPHA")


def test_skips_blocks_absent_from_old_checkouts(tmp_path):
    # Pre-SSM_BETA/SSM_ALPHA file: patch what exists, never invent blocks.
    old = STALE_TENSOR_MAPPING[: STALE_TENSOR_MAPPING.index("        MODEL_TENSOR.SSM_BETA")] + "    }\n"
    path = _write_stale(str(tmp_path), old)
    llama_cpp._patch_tensor_mapping_for_qwen35(str(tmp_path))
    result = Path(path).read_text(encoding = "utf-8")
    assert "linear_attn.in_proj_qkv" in _block(result, "ATTN_QKV")
    assert "in_proj_b" not in result and "in_proj_a" not in result
    ast.parse(result)


def test_missing_file_is_a_no_op(tmp_path):
    llama_cpp._patch_tensor_mapping_for_qwen35(str(tmp_path))

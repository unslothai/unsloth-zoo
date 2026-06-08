import os
import tempfile
import sys

SOURCE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unsloth_zoo', 'llama_cpp.py')


def _load_patch_function():
    with open(SOURCE_FILE, 'r') as f:
        content = f.read()

    start = content.find('def _patch_tensor_mapping_for_qwen35')
    if start == -1:
        raise RuntimeError("Function not found")

    end = content.find('\ndef ', start + 1)
    if end == -1:
        end = len(content)

    func_code = content[start:end]

    namespace = {'os': os}
    exec(func_code, namespace)
    return namespace['_patch_tensor_mapping_for_qwen35']


_patch_tensor_mapping_for_qwen35 = _load_patch_function()


def test_patch_tensor_mapping_for_qwen35():
    with tempfile.TemporaryDirectory() as tmpdir:
        gguf_dir = os.path.join(tmpdir, 'gguf-py', 'gguf')
        os.makedirs(gguf_dir)

        mock_content = '''MODEL_TENSOR.SSM_CONV1D: (
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
'''
        mapping_path = os.path.join(gguf_dir, 'tensor_mapping.py')
        with open(mapping_path, 'w') as f:
            f.write(mock_content)

        _patch_tensor_mapping_for_qwen35(tmpdir)

        with open(mapping_path, 'r') as f:
            result = f.read()

        assert 'qwen3.5' in result.lower()
        assert result.lower().count('qwen3.5') >= 5

        expected_patterns = [
            'linear_attn.conv1d',
            'linear_attn.dt_proj',
            'linear_attn.A_log',
            'linear_attn.norm',
            'linear_attn.out_proj',
        ]
        for pattern in expected_patterns:
            assert pattern in result


def test_patch_tensor_mapping_idempotent():
    with tempfile.TemporaryDirectory() as tmpdir:
        gguf_dir = os.path.join(tmpdir, 'gguf-py', 'gguf')
        os.makedirs(gguf_dir)

        mock_content = '''MODEL_TENSOR.SSM_CONV1D: (
    "model.layers.{bid}.conv1d",               # mamba-hf
    "model.layers.{bid}.linear_attn.conv1d",   # qwen3next
),
'''
        mapping_path = os.path.join(gguf_dir, 'tensor_mapping.py')
        with open(mapping_path, 'w') as f:
            f.write(mock_content)

        _patch_tensor_mapping_for_qwen35(tmpdir)
        _patch_tensor_mapping_for_qwen35(tmpdir)

        with open(mapping_path, 'r') as f:
            result = f.read()

        count = result.lower().count('qwen3.5')
        assert count == 1


def test_patch_tensor_mapping_skips_when_already_patched():
    with tempfile.TemporaryDirectory() as tmpdir:
        gguf_dir = os.path.join(tmpdir, 'gguf-py', 'gguf')
        os.makedirs(gguf_dir)

        mock_content = '''MODEL_TENSOR.SSM_CONV1D: (
    "model.layers.{bid}.conv1d",               # mamba-hf
    "model.layers.{bid}.linear_attn.conv1d",   # qwen3next
            "model.layers.{bid}.linear_attn.conv1d",   # qwen3.5
),
'''
        mapping_path = os.path.join(gguf_dir, 'tensor_mapping.py')
        with open(mapping_path, 'w') as f:
            f.write(mock_content)

        _patch_tensor_mapping_for_qwen35(tmpdir)

        with open(mapping_path, 'r') as f:
            result = f.read()

        count = result.lower().count('qwen3.5')
        assert count == 1


def test_patch_tensor_mapping_graceful_missing_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        _patch_tensor_mapping_for_qwen35(tmpdir)
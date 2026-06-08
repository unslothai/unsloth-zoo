import os
import tempfile
import shutil
import sys

# Import the function directly without triggering unsloth_zoo package imports
sys.path.insert(0, '/home/datta0/repos/unsloth-zoo/worktrees/issue-6071-qwen35-ssm-fix')

# Read and exec the function from the source file to avoid package import issues
def _get_patch_function():
    with open('/home/datta0/repos/unsloth-zoo/worktrees/issue-6071-qwen35-ssm-fix/unsloth_zoo/llama_cpp.py', 'r') as f:
        content = f.read()
    
    # Extract the function definition
    start = content.find('def _patch_tensor_mapping_for_qwen35')
    if start == -1:
        raise RuntimeError("Function not found")
    
    # Find the end of the function (next function def or end of file)
    end = content.find('\ndef ', start + 1)
    if end == -1:
        end = len(content)
    
    func_code = content[start:end]
    
    # Create a namespace with required imports
    import logging
    logger = logging.getLogger('test')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    
    namespace = {
        'os': os,
        'logging': logging,
        'logger': logger,
    }
    exec(func_code, namespace)
    return namespace['_patch_tensor_mapping_for_qwen35']

_patch_tensor_mapping_for_qwen35 = _get_patch_function()


def test_patch_tensor_mapping_for_qwen35():
    """Test that _patch_tensor_mapping_for_qwen35 correctly adds Qwen3.5 patterns to tensor_mapping.py"""

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

        assert 'qwen3.5' in result.lower(), 'Qwen3.5 pattern not found'
        assert result.lower().count('qwen3.5') >= 5, f'Expected at least 5 qwen3.5 patterns, got {result.lower().count("qwen3.5")}'

        expected_patterns = [
            'linear_attn.conv1d',
            'linear_attn.dt_proj',
            'linear_attn.A_log',
            'linear_attn.norm',
            'linear_attn.out_proj',
        ]
        for pattern in expected_patterns:
            assert pattern in result, f'Pattern {pattern} not found in patched file'

        print('test_patch_tensor_mapping_for_qwen35 PASSED')


def test_patch_tensor_mapping_idempotent():
    """Test that running the patch twice doesn't duplicate patterns"""

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
        assert count == 1, f'Expected 1 qwen3.5 pattern after idempotent run, got {count}'

        print('test_patch_tensor_mapping_idempotent PASSED')


def test_patch_tensor_mapping_skips_when_already_patched():
    """Test that patch is skipped when Qwen3.5 patterns already present"""

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
        assert count == 1, f'Expected 1 qwen3.5 pattern, got {count}'

        print('test_patch_tensor_mapping_skips_when_already_patched PASSED')


def test_patch_tensor_mapping_graceful_missing_file():
    """Test that function handles missing tensor_mapping.py gracefully"""

    with tempfile.TemporaryDirectory() as tmpdir:
        _patch_tensor_mapping_for_qwen35(tmpdir)
        print('test_patch_tensor_mapping_graceful_missing_file PASSED')


if __name__ == '__main__':
    test_patch_tensor_mapping_for_qwen35()
    test_patch_tensor_mapping_idempotent()
    test_patch_tensor_mapping_skips_when_already_patched()
    test_patch_tensor_mapping_graceful_missing_file()
    print('\\nAll tests PASSED!')
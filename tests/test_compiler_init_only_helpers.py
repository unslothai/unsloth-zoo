"""Module-level helpers that only build constructor buffers must not be compiled.

Qwen4Exp's `_build_layer_multipliers(unigram_vocab_size, ngram_size, ple_layer_index, seed)`
has unannotated arguments, so `function_has_tensor_inputs` cannot rule it out, and compiling
it made `Qwen4ExpTextNGramEmbedding.__init__` fail under the meta device at load
(Dynamo: "'NotImplementedType' object has no attribute 'detach_'" in torch.tensor)."""
import importlib
import inspect

import pytest

SOURCE = '''
def _mix(value):
    return value * 3

def _build_layer_multipliers(vocab, n, index, seed: int):
    return torch.tensor([_mix(seed + i) for i in range(n)])

def rotate_half(x):
    return x

def used_in_forward(x):
    return x

TABLE = _mix(3)

class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.buf = _build_layer_multipliers(config.v, 3, 0, config.seed)

    def forward(self, x):
        return rotate_half(used_in_forward(x))

class Other(nn.Module):
    def _init_weights(self, module):
        module.buf = _build_layer_multipliers(1, 2, 3, 4)
'''


def test_init_only_helpers_detected():
    from unsloth_zoo.compiler import function_only_called_at_init
    assert function_only_called_at_init(SOURCE, "_build_layer_multipliers")
    # _mix is also called at module import time (TABLE), so it is not init-only.
    assert not function_only_called_at_init(SOURCE, "_mix")
    assert not function_only_called_at_init(SOURCE, "rotate_half")
    assert not function_only_called_at_init(SOURCE, "used_in_forward")
    # Never called in this source: unknown callers, keep compiling.
    assert not function_only_called_at_init(SOURCE, "missing_function")
    assert not function_only_called_at_init("not python (", "_mix")


def test_transitive_init_only_helper():
    from unsloth_zoo.compiler import function_only_called_at_init
    src = SOURCE.replace("TABLE = _mix(3)\n", "")
    assert function_only_called_at_init(src, "_mix")


def test_qwen4_exp_ngram_helpers_are_init_only():
    try:
        modeling = importlib.import_module("transformers.models.qwen4_exp.modeling_qwen4_exp")
    except Exception:
        pytest.skip("transformers without qwen4_exp")
    from unsloth_zoo.compiler import function_only_called_at_init
    source = inspect.getsource(modeling)
    for name in ("_build_layer_multipliers", "_find_nth_prime_after", "_is_prime", "_splitmix64"):
        assert function_only_called_at_init(source, name), name
    for name in ("rotate_half", "apply_rotary_pos_emb", "l2norm", "repeat_kv"):
        assert not function_only_called_at_init(source, name), name

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""MLX QAT: the trained forward is exactly what merged_4bit ships; refusals mutate nothing.

Runs on real mlx (CPU or Metal), no downloads: a tiny mlx-lm llama is built in-process.
"""

import importlib
import sys

import pytest


def _real_mlx_runtime():
    try:
        importlib.import_module("mlx_lm.tuner.lora")
    except Exception:
        return False
    return "mlx_simulation" not in (getattr(sys.modules.get("mlx.core"), "__file__", "") or "")


if not _real_mlx_runtime():
    pytest.skip("needs the real mlx runtime (tests/mlx_simulation is active or mlx absent)",
                allow_module_level=True)

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm.tuner.lora import LoRALinear

from unsloth_zoo.mlx import qat
from unsloth_zoo.mlx.loader import FastMLXModel


def _lora(bias, dtype, dims=128):
    mx.random.seed(1)
    lin = nn.Linear(dims, dims, bias=bias)
    lin.set_dtype(dtype)
    layer = LoRALinear.from_base(nn.QuantizedLinear.from_linear(lin, 64, 4), r=8, scale=2.0)
    layer.lora_b = (mx.random.normal(layer.lora_b.shape) * 0.05).astype(dtype)
    layer.lora_a = layer.lora_a.astype(dtype)
    return layer


def _tiny_llama(quantized=True):
    from mlx_lm.models import llama
    args = llama.ModelArgs(
        model_type="llama", hidden_size=128, num_hidden_layers=2, intermediate_size=256,
        num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=512,
    )
    mx.random.seed(0)
    model = llama.Model(args)
    model.update(tree_map(lambda v: v.astype(mx.bfloat16), model.parameters()))
    if quantized:
        nn.quantize(model, group_size=64, bits=4)
    model._config = {**vars(args), "torch_dtype": "bfloat16"}
    if quantized:
        model._config["quantization"] = {"group_size": 64, "bits": 4, "mode": "affine"}
    return model


def _peft(model, **kwargs):
    kwargs = {"r": 8, "lora_alpha": 16, "lora_dropout": 0, "qat_scheme": "auto",
              "use_gradient_checkpointing": False, **kwargs}
    return FastMLXModel.get_peft_model(model, **kwargs)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
def test_qat_forward_is_bit_exact_to_the_fused_module(bias, dtype):
    # bf16 is where a dense GEMM and quantized_matmul round differently.
    layer = _lora(bias, dtype)
    fused = layer.fuse(dequantize=False)
    qat.apply_mlx_qat(layer)
    x = mx.random.normal((64, 128)).astype(dtype)
    assert mx.array_equal(layer(x), fused(x)).item()


def test_straight_through_gradients_match_the_dense_reference():
    layer = _lora(True, mx.float32)
    x = mx.random.normal((3, 5, 128))

    def dense_ste(a, b, xx):
        base = layer.linear
        w = mx.dequantize(base.weight, base.scales, base.biases, group_size=64, bits=4)
        merged = w + (layer.scale * b.T) @ a.T
        fake = mx.dequantize(*mx.quantize(merged, group_size=64, bits=4), group_size=64, bits=4)
        return (xx @ (merged + mx.stop_gradient(fake - merged)).T + base.bias).square().sum()

    def ours(a, b, xx):
        layer.lora_a, layer.lora_b = a, b
        return layer(xx).square().sum()

    ref = mx.grad(dense_ste, argnums=(0, 1, 2))(layer.lora_a, layer.lora_b, x)
    qat.apply_mlx_qat(layer)
    got = mx.grad(ours, argnums=(0, 1, 2))(layer.lora_a, layer.lora_b, x)
    for g, r in zip(got, ref):
        assert mx.allclose(g, r, rtol=1e-4, atol=1e-4).item()
        assert float(mx.abs(g).max()) > 0


def test_get_peft_model_qat_trains_compiled_and_survives_the_merged_4bit_save(tmp_path):
    from mlx_lm.utils import load_model

    from unsloth_zoo.mlx.utils import save_merged_model

    model = _peft(_tiny_llama())
    assert sum(getattr(m, qat._QAT_FLAG, False) for _, m in model.named_modules()) == 14

    ids = mx.array([[3, 11, 19, 27, 35, 43, 51, 59] * 4])

    def loss(m):
        return nn.losses.cross_entropy(m(ids[:, :-1]).astype(mx.float32), ids[:, 1:]).mean()

    opt = optim.Adam(learning_rate=1e-2)
    grad_fn = nn.value_and_grad(model, loss)
    state = [model.state, opt.state]

    @mx.compile
    def step():
        value, grads = grad_fn(model)
        opt.update(model, grads)
        return value

    first = None
    for _ in range(5):
        value = step()
        mx.eval(state, value)
        first = first if first is not None else float(value)
    assert float(loss(model)) < first

    logits = model(ids)
    mx.eval(logits)

    class _Tok:
        def save_pretrained(self, path):
            pass

    save_merged_model(model, _Tok(), tmp_path, dequantize=False, quantize_unquantized=True)
    reloaded, _ = load_model(tmp_path)
    assert mx.array_equal(reloaded(ids), logits).item()


def _set_full_finetuning(model):
    model._unsloth_full_finetuning = True
    return model


_REFUSALS = [
    ("torchao scheme", {"qat_scheme": "fp8-int4"}, NotImplementedError, None),
    ("unknown scheme", {"qat_scheme": "int3"}, NotImplementedError, None),
    ("non-string scheme", {"qat_scheme": 4}, TypeError, None),
    ("bit mismatch", {"qat_scheme": "int8"}, ValueError, None),
    ("lora_dropout", {"lora_dropout": 0.1}, NotImplementedError, None),
    ("dora", {"use_dora": True}, NotImplementedError, None),
    ("no targets", {"finetune_language_layers": False}, ValueError, None),
    ("unquantized base", {}, ValueError, lambda: _tiny_llama(quantized=False)),
    ("full_finetuning", {}, NotImplementedError, lambda: _set_full_finetuning(_tiny_llama())),
]


@pytest.mark.parametrize("kwargs,error,make", [r[1:] for r in _REFUSALS], ids=[r[0] for r in _REFUSALS])
def test_refused_requests_leave_the_model_untouched(kwargs, error, make):
    model = make() if make else _tiny_llama()
    before = tree_flatten(model.trainable_parameters())
    with pytest.raises(error):
        _peft(model, **kwargs)
    assert not any(isinstance(m, LoRALinear) for _, m in model.named_modules())
    assert [k for k, _ in tree_flatten(model.trainable_parameters())] == [k for k, _ in before]


def test_vlm_is_refused(monkeypatch):
    from unsloth_zoo.mlx import utils
    monkeypatch.setattr(utils, "_is_vlm_model", lambda model: True)
    with pytest.raises(NotImplementedError, match="VLM"):
        qat.validate_mlx_qat_request(_tiny_llama(), "auto")

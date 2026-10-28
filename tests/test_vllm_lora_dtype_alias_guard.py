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

"""Adapter dtype must never equal the vLLM hot-load dtype (model dtype, since unsloth never sets lora_dtype); equal
dtypes alias training weights and optimize() scales them in place when lora_alpha != r; the fp32 upcast prevents
that, so never drop it to save memory."""
import pytest
import torch
import torch.nn as nn

from unsloth_zoo.training_utils import prepare_model_for_training


class _Cfg:
    def __init__(self, dtype):
        self.torch_dtype = dtype
        self.model_type = "llama"


class _LoRALinear(nn.Module):
    def __init__(self, d_in, d_out, r):
        super().__init__()
        self.base_layer = nn.Linear(d_in, d_out, bias = False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias = False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias = False)})

    def forward(self, x):
        return self.base_layer(x) + self.lora_B["default"](self.lora_A["default"](x))


class _TinyLoRA(nn.Module):
    """Frozen base plus trainable adapter named as peft names it: the upcast keys on the substring .lora_A. with the
    leading dot, so a flat lora_A.weight would not be upcast."""

    def __init__(self, dtype, d_in = 16, d_out = 12, r = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, d_in)
        self.q_proj = _LoRALinear(d_in, d_out, r)
        self.to(dtype)
        for name, param in self.named_parameters():
            param.requires_grad_(".lora_A." in name or ".lora_B." in name)
        self.config = _Cfg(dtype)

    # prepare_model_for_training registers a require-grad hook on the input embedding
    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, x):
        return self.q_proj(x)


def _lora_dtypes(model):
    return {p.dtype for n, p in model.named_parameters() if "lora_" in n}


def _vllm_lora_dtype_for(model_dtype):
    """vLLM resolves lora_dtype None or auto to model_config.dtype (vllm/config/lora.py); unsloth never passes lora_dtype, so this is always the model dtype."""
    return model_dtype


@pytest.mark.parametrize("model_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("float32_mixed_precision", [True, False])
def test_adapter_dtype_never_matches_the_vllm_hot_load_dtype(
    model_dtype, float32_mixed_precision,
):
    """The invariant: if this fails, load_lora ships aliases of the live training weights and GRPO corrupts silently whenever lora_alpha != r."""
    model = _TinyLoRA(model_dtype)
    prepare_model_for_training(
        model,
        use_gradient_checkpointing = False,
        float32_mixed_precision = float32_mixed_precision,
    )

    engine_dtype = _vllm_lora_dtype_for(model_dtype)
    for dtype in _lora_dtypes(model):
        assert dtype != engine_dtype, (
            f"LoRA adapter is {dtype}, which is exactly what vLLM will hot-load it as "
            f"({engine_dtype}). tensor.to(dtype) becomes a no-op, the engine aliases "
            "the training weights, and LoRALayerWeights.optimize() scales them in "
            "place on every generation. Either restore the float32 upcast in "
            "prepare_model_for_training, or make load_lora ship detached clones."
        )


@pytest.mark.parametrize("model_dtype", [torch.bfloat16, torch.float16])
def test_adapter_is_upcast_to_float32(model_dtype):
    """The upcast alone, isolated from the mixed-precision default: float32_mixed_precision = False is deliberate, since
    True makes the adapter fp32 via mixed_precision_dtype and the test would pin nothing."""
    model = _TinyLoRA(model_dtype)
    prepare_model_for_training(
        model, use_gradient_checkpointing = False, float32_mixed_precision = False,
    )
    assert _lora_dtypes(model) == {torch.float32}, (
        f"expected the adapter upcast to float32, got {_lora_dtypes(model)}. The "
        "`upcast = True` arm for .lora_A./.lora_B. in prepare_model_for_training is "
        "what keeps the adapter dtype away from vLLM's lora_dtype."
    )


def test_frozen_base_weights_are_not_upcast():
    """The upcast must stay scoped to trainable params: upcasting the frozen base would blow up memory and is not what protects against aliasing."""
    model = _TinyLoRA(torch.bfloat16)
    prepare_model_for_training(model, use_gradient_checkpointing = False)
    assert model.q_proj.base_layer.weight.dtype == torch.bfloat16


def test_aliasing_is_what_the_dtype_gap_prevents():
    """Why the dtype gap matters: torch .to() returns the same object on a dtype match and a fresh one otherwise; asserted so the docs cannot quietly go stale."""
    training_weight = torch.randn(8, 4, dtype = torch.float32)

    same = training_weight.to(device = None, dtype = torch.float32)
    assert same.data_ptr() == training_weight.data_ptr(), \
        "matching dtype must alias, or the premise of this file is wrong"

    different = training_weight.to(device = None, dtype = torch.bfloat16)
    assert different.data_ptr() != training_weight.data_ptr()

    # and the alias is writable straight through to the training weight
    before = training_weight.clone()
    same *= 2.0
    assert not torch.equal(training_weight, before), \
        "in-place scaling through the alias must reach the training weight"

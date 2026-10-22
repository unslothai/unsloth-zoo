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

"""The LoRA adapter dtype must never equal the dtype vLLM will hot-load it as.

`load_lora(load_tensors = True)` passes the live training tensors into
`LoRARequest(lora_tensors = ...)`. vLLM stores `tensor.to(device = None, dtype = ...)`,
which returns the SAME OBJECT when the dtype already matches, and
`LoRALayerWeights.optimize()` then runs `lora_b *= scaling` in place. With
`lora_alpha != r` that multiply lands on the training weights and compounds once per
generation, silently rotting a GRPO run.

Two things keep that from happening, and only one of them is ours:

  1. vLLM resolves `lora_dtype = "auto"` to the MODEL dtype (bf16 or fp16 here).
     unsloth never sets `lora_dtype`, so this is always the model dtype.
  2. `prepare_model_for_training` upcasts every trainable parameter, i.e. exactly the
     LoRA adapter, to float32.

fp32 != bf16, so `.to(dtype)` allocates a copy and the engine scales its own tensor.
That is the ONLY reason stock GRPO is safe: the two facts live in different repos,
neither documents the coupling, and dropping the upcast to save adapter memory looks
like a free optimisation. It is not, and this test is the tripwire.

Measured on a real engine (Llama-3.2-1B, r=16, 4 generation rounds, vLLM 0.15.1):

    adapter fp32 / lora_dtype bf16   224/224 aliased, 0/448 optimize() hits, 0/224 drifted
    adapter bf16 / lora_dtype bf16   224/224 aliased, 448/448 hits, 112/224 DRIFTED

CPU only; does not import vLLM or build an engine.
"""
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
    """A frozen base plus a trainable adapter, named the way peft names one.

    The naming is load-bearing: prepare_model_for_training keys the upcast on the
    substring ".lora_A." WITH the leading dot, so a flat `lora_A.weight` is not
    upcast and the test would pass for the wrong reason.
    """

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
    """What vLLM resolves `lora_dtype = "auto"` to. unsloth never passes lora_dtype,
    so this is always the model dtype (vllm/config/lora.py: `if self.lora_dtype in
    (None, "auto"): self.lora_dtype = model_config.dtype`)."""
    return model_dtype


@pytest.mark.parametrize("model_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("float32_mixed_precision", [True, False])
def test_adapter_dtype_never_matches_the_vllm_hot_load_dtype(
    model_dtype, float32_mixed_precision,
):
    """The invariant. If this fails, `load_lora` starts shipping aliases of the live
    training weights and GRPO corrupts silently whenever lora_alpha != r."""
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
    """The mechanism, isolated from the mixed-precision default.

    float32_mixed_precision = False is deliberate: with it True the adapter would be
    fp32 anyway via mixed_precision_dtype, so the test would pass even with the
    adapter upcast deleted and would pin nothing.
    """
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
    """The upcast must stay scoped to trainable params: upcasting the frozen base
    would blow up memory and is not what protects us here."""
    model = _TinyLoRA(torch.bfloat16)
    prepare_model_for_training(model, use_gradient_checkpointing = False)
    assert model.q_proj.base_layer.weight.dtype == torch.bfloat16


def test_aliasing_is_what_the_dtype_gap_prevents():
    """Why the dtype gap matters at all: `.to()` returns the same object on a match
    and a fresh one otherwise. This is torch behaviour, asserted so the docstring
    above cannot quietly go stale."""
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

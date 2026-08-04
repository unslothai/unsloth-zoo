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

"""Tests for the vLLM direct LoRA hot-load path.

Drives the real `prepare_vllm_lora_loading` / `load_lora_directly` against fake PEFT
modules and fake vLLM stacked buffers. No GPU, no real vLLM needed.

Asserts the contract:
  - every projection lands in its own vLLM slot, gate_up slot 1 being up_proj (gate and
    up share shapes, so the shape asserts alone cannot catch a mis-pairing);
  - the effective delta each slot carries equals scaling * B @ A;
  - the destination is scaled, never the training weights.
"""

from __future__ import annotations

import types

import pytest
import torch


# Imported at module level on purpose: this file is a CI hard gate needing neither a GPU nor
# vLLM (tests/conftest.py handles the device probe), so an import failure is a regression
# and must fail the gate instead of skipping it.
import unsloth_zoo.vllm_utils as vllm_utils


HIDDEN, INTER, KV, RANK = 8, 12, 4, 4
SHAPES = {
    "q": (HIDDEN, HIDDEN), "k": (KV, HIDDEN), "v": (KV, HIDDEN), "o": (HIDDEN, HIDDEN),
    "gate": (INTER, HIDDEN), "up": (INTER, HIDDEN), "down": (HIDDEN, INTER),
}
SCALING = {"q": 1.0, "k": 1.0, "v": 1.0, "o": 1.0, "gate": 3.0, "up": 2.0, "down": 1.0}


def _peft_linear(name):
    """PEFT-wrapped nn.Linear stand-in, with weights unique to this projection."""
    out_features, in_features = SHAPES[name]
    return types.SimpleNamespace(
        lora_A=types.SimpleNamespace(default=types.SimpleNamespace(
            weight=torch.randn(RANK, in_features))),
        lora_B=types.SimpleNamespace(default=types.SimpleNamespace(
            weight=torch.randn(out_features, RANK))),
        scaling={"default": SCALING[name]},
    )


def _vllm_slot(name):
    """vLLM allocates (max_loras, 1, r, in) / (max_loras, 1, out, r) zero buffers."""
    out_features, in_features = SHAPES[name]
    return (torch.zeros(1, 1, RANK, in_features), torch.zeros(1, 1, out_features, RANK))


def _make_model(n_layers=2):
    m_layers, v_layers = [], []
    for _ in range(n_layers):
        m_layers.append(types.SimpleNamespace(
            self_attn=types.SimpleNamespace(**{
                f"{n}_proj": _peft_linear(n) for n in ("q", "k", "v", "o")}),
            mlp=types.SimpleNamespace(**{
                f"{n}_proj": _peft_linear(n) for n in ("gate", "up", "down")}),
        ))
        qkv_a, qkv_b = zip(*(_vllm_slot(n) for n in ("q", "k", "v")))
        gu_a, gu_b = zip(*(_vllm_slot(n) for n in ("gate", "up")))
        o_a, o_b = _vllm_slot("o")
        d_a, d_b = _vllm_slot("down")
        v_layers.append(types.SimpleNamespace(
            self_attn=types.SimpleNamespace(
                qkv_proj=types.SimpleNamespace(lora_a_stacked=qkv_a, lora_b_stacked=qkv_b),
                o_proj=types.SimpleNamespace(lora_a_stacked=(o_a,), lora_b_stacked=(o_b,))),
            mlp=types.SimpleNamespace(
                gate_up_proj=types.SimpleNamespace(lora_a_stacked=gu_a, lora_b_stacked=gu_b),
                down_proj=types.SimpleNamespace(lora_a_stacked=(d_a,), lora_b_stacked=(d_b,))),
        ))
    vllm_model = types.SimpleNamespace(model=types.SimpleNamespace(layers=v_layers))
    return types.SimpleNamespace(
        model=types.SimpleNamespace(model=types.SimpleNamespace(layers=m_layers)),
        vllm_engine=types.SimpleNamespace(llm_engine=types.SimpleNamespace(
            model_executor=types.SimpleNamespace(driver_worker=types.SimpleNamespace(
                model_runner=types.SimpleNamespace(model=vllm_model))))),
    )


def _slots(model, layer):
    v = model.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
    L = v.model.layers[layer]
    return {
        "q": (L.self_attn.qkv_proj.lora_a_stacked[0], L.self_attn.qkv_proj.lora_b_stacked[0]),
        "k": (L.self_attn.qkv_proj.lora_a_stacked[1], L.self_attn.qkv_proj.lora_b_stacked[1]),
        "v": (L.self_attn.qkv_proj.lora_a_stacked[2], L.self_attn.qkv_proj.lora_b_stacked[2]),
        "o": (L.self_attn.o_proj.lora_a_stacked[0], L.self_attn.o_proj.lora_b_stacked[0]),
        "gate": (L.mlp.gate_up_proj.lora_a_stacked[0], L.mlp.gate_up_proj.lora_b_stacked[0]),
        "up": (L.mlp.gate_up_proj.lora_a_stacked[1], L.mlp.gate_up_proj.lora_b_stacked[1]),
        "down": (L.mlp.down_proj.lora_a_stacked[0], L.mlp.down_proj.lora_b_stacked[0]),
    }


def _module(model, layer, name):
    holder = (model.model.model.layers[layer].self_attn if name in ("q", "k", "v", "o")
              else model.model.model.layers[layer].mlp)
    return getattr(holder, f"{name}_proj")


def _pairs(model_B, vllm_B, scalings, model_A=None, vllm_A=None):
    return types.SimpleNamespace(
        model_loras_A=model_A or [], vllm_loras_A=vllm_A or [],
        model_loras_B=model_B, vllm_loras_B=list(zip(vllm_B, scalings)))


@pytest.fixture(autouse=True)
def _no_cuda_sync(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)


def test_every_projection_reaches_its_own_slot():
    torch.manual_seed(0)
    model = _make_model()
    vllm_utils.prepare_vllm_lora_loading(model)
    vllm_utils.load_lora_directly(model)

    for layer in range(len(model.model.model.layers)):
        for name, (slot_a, slot_b) in _slots(model, layer).items():
            module = _module(model, layer, name)
            lora_a = module.lora_A.default.weight
            lora_b = module.lora_B.default.weight
            assert torch.equal(slot_a.squeeze(0).squeeze(0), lora_a), name
            # vLLM folds scaling into B, so the slot holds scaling * B
            assert torch.allclose(slot_b.squeeze(0).squeeze(0), SCALING[name] * lora_b), name
            delta = slot_b.squeeze(0).squeeze(0) @ slot_a.squeeze(0).squeeze(0)
            assert torch.allclose(delta, SCALING[name] * (lora_b @ lora_a), atol=1e-5), name


def test_independent_buffers_scale_the_destination_only():
    src, dst = torch.full((4, 2), 3.0), torch.zeros(1, 1, 4, 2)
    for _ in range(3):
        vllm_utils.load_lora_directly(_pairs([src], [dst], [2.0]))
    assert dst.max().item() == 6.0
    assert src.max().item() == 3.0

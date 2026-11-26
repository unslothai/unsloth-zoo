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

"""vLLM LoRA handoff: one clone per expert stack (same keys and bits, detached, disjoint); other state_dicts unchanged."""
import os

import pytest
import torch
import torch.nn as nn

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model  # noqa: E402

import transformers  # noqa: E402
if int(transformers.__version__.split(".")[0]) >= 5:
    pytest.skip("transformers 5 fuses the experts: no zoo stacking", allow_module_level = True)
VU = pytest.importorskip("unsloth_zoo.vllm_utils")
from unsloth_zoo.temporary_patches import moe_grouped_modulelist as ML  # noqa: E402

E = 8
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(params = DEVICES)
def dev(request, monkeypatch):
    # Host stacks are left unpacked on purpose (allocation cost); pack them here to test the same code on CPU.
    if hasattr(ML, "_UNPACKED_DEVICE_TYPES"):
        monkeypatch.setattr(ML, "_UNPACKED_DEVICE_TYPES", ())
    return request.param


def _model(dev):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(vocab_size = 64, hidden_size = 32, intermediate_size = 64, moe_intermediate_size = 24,
                         num_experts = E, num_experts_per_tok = 2, num_hidden_layers = 2, num_attention_heads = 4,
                         num_key_value_heads = 2, head_dim = 8, max_position_embeddings = 32,
                         norm_topk_prob = True, mlp_only_layers = [], decoder_sparse_step = 1)
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(cfg).to(dev, torch.bfloat16)
    if not isinstance(model.model.layers[0].mlp.experts, nn.ModuleList):
        pytest.skip("experts are not an nn.ModuleList on this transformers (no zoo stacking)")
    model = get_peft_model(model, LoraConfig(r = 4, lora_alpha = 8, target_modules = [
        "q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]))
    with torch.no_grad():
        torch.manual_seed(1)
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.normal_(0, 0.5)
    ref = {k: v.detach().clone() for k, v in VU._lora_state_dict(model).items()}
    n = 0
    for mod in model.modules():
        spec = ML._block_is_eligible(mod) if hasattr(mod, "experts") else None
        if spec is not None:
            n += ML._stack_block_lora(mod, spec)
    assert n == 2 * 3   # 2 layers x gate / up / down
    stacks = [p for k, p in model.named_parameters() if k.endswith("." + ML._STACK_NAME)]
    assert len(stacks) == 2 * 3 * 2
    return model, ref, stacks


def _handoff(model):
    return getattr(VU, "_lora_tensors_for_vllm", VU._lora_state_dict)(model)


def _count_clones(fn):
    n = [0]
    orig = torch.Tensor.clone

    def counted(self, *a, **k):
        n[0] += 1
        return orig(self, *a, **k)
    torch.Tensor.clone = counted
    try:
        out = fn()
    finally:
        torch.Tensor.clone = orig
    return out, n[0]


def test_handoff_one_clone_per_stack_same_keys_and_bits(dev):
    model, ref, stacks = _model(dev)
    sd, clones = _count_clones(lambda: _handoff(model))
    assert list(sd) == list(ref)
    for k, v in ref.items():
        assert sd[k].dtype == v.dtype and sd[k].shape == v.shape and torch.equal(sd[k], v), k
    assert clones == len(stacks), clones   # base: one per expert (2 * 3 * 2 * E)
    expert = [k for k in sd if ".experts." in k]
    assert len(expert) == 2 * 3 * 2 * E
    assert len({sd[k].untyped_storage().data_ptr() for k in expert}) == len(stacks)


def test_handoff_detached_and_slices_disjoint(dev):
    model, _, stacks = _model(dev)
    sd = _handoff(model)
    snap = {k: v.clone() for k, v in sd.items()}
    live_ptrs = {p.untyped_storage().data_ptr() for p in stacks}
    expert = [k for k in sd if ".experts." in k]
    assert not any(sd[k].untyped_storage().data_ptr() in live_ptrs for k in expert)
    with torch.no_grad():   # an optimizer step on the live stacks does not reach the handed-off tensors
        for p in stacks:
            p.add_(1.0)
    assert all(torch.equal(sd[k], snap[k]) for k in sd)
    # the receiver scaling one expert in place leaves its stack neighbours and the live stacks alone
    live = {k: v.clone() for k, v in VU._lora_state_dict(model).items()}
    k0 = next(k for k in expert if ".experts.0." in k and "lora_A" in k)
    sd[k0].mul_(3.0)
    assert all(torch.equal(sd[k], snap[k]) for k in sd if k != k0)
    assert torch.equal(sd[k0], snap[k0] * 3.0)
    assert all(torch.equal(v, live[k]) for k, v in VU._lora_state_dict(model).items())


def test_other_state_dicts_keep_independent_clones(dev):
    model, ref, stacks = _model(dev)
    _handoff(model)   # leaves no packing behind
    for sd in (VU._lora_state_dict(model), {k: v for k, v in model.state_dict().items() if "lora_" in k}):
        expert = [k for k in sd if ".experts." in k]
        assert len({sd[k].untyped_storage().data_ptr() for k in expert}) == len(expert)
        assert all(torch.equal(sd[k], ref[k]) for k in expert)
    live_ptrs = {p.untyped_storage().data_ptr() for p in stacks}
    sd = model.state_dict(keep_vars = True)
    assert all(sd[k].untyped_storage().data_ptr() in live_ptrs for k in sd if ".experts." in k and "lora_" in k)


def test_host_stacks_stay_unpacked():
    model, ref, stacks = _model("cpu")
    sd, clones = _count_clones(lambda: _handoff(model))
    assert clones == 2 * 3 * 2 * E
    assert all(torch.equal(sd[k], ref[k]) for k in ref)

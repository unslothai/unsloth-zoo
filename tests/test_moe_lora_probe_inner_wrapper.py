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

"""The stash probe of an outer expert ParamWrapper must measure the experts forward, not a decode route.

gpt-oss with LoRA on gate_up_proj and down_proj nests two ParamWrappers. The outer one's first training call
measures whether the experts forward reads its stash with a no_grad call of the inner wrapper. On sm80+ CUDA in
bf16 the inner wrapper took the routed decode kernel there (it only skips routing once marked inner, which
happened only when the outer routed first), so the experts forward never ran, the verdict said "stash unread",
and the outer adapter went to PEFT's parametrization. Its first-step grads were then None: the inner wrapper's
probe ran inside PEFT's parametrize.cached() and cached the folded weight without autograd history.

The routed kernel is CUDA / Triton only, so it is replaced by a stand-in with the same contract (no stash reads,
every adapter of the chain applied) and the eligibility check answers as on an L4.
"""

from __future__ import annotations

import sys

import pytest
import torch

peft = pytest.importorskip("peft")
pytest.importorskip("peft.tuners.lora.layer").ParamWrapper  # noqa: B018

from unsloth_zoo.temporary_patches import moe_utils as MU  # noqa: E402


def _copies():
    out = [MU]
    for name in ("unsloth_cached_moe_utils", "moe_utils"):
        module = sys.modules.get(name)
        if module is not None and module is not MU and hasattr(module, "_patched_param_wrapper_forward"):
            out.append(module)
    return out


def _apply_patches():
    import unsloth_zoo.temporary_patches  # noqa: F401
    from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES

    for fn in TEMPORARY_PATCHES:
        try:
            fn()
        except Exception:
            pass


def _tiny_gpt_oss():
    from transformers import GptOssConfig, GptOssForCausalLM

    config = GptOssConfig(
        vocab_size = 128, hidden_size = 32, intermediate_size = 32, num_hidden_layers = 2,
        num_attention_heads = 4, num_key_value_heads = 2, head_dim = 8, num_local_experts = 4,
        num_experts_per_tok = 2, layer_types = ["sliding_attention", "full_attention"], sliding_window = 8,
        max_position_embeddings = 64,
    )
    torch.manual_seed(0)
    return GptOssForCausalLM(config).float().train()


def _lora(model):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(7)
    config = LoraConfig(
        r = 4, lora_alpha = 8, lora_dropout = 0.0, target_modules = [], init_lora_weights = False,
        target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    )
    return get_peft_model(model, config)


def _outer_lora_grads(model):
    return {
        name: p.grad for name, p in model.named_parameters()
        if "lora_" in name and "experts.lora_" in name and "base_layer" not in name
    }


@pytest.fixture
def routed_like_l4(monkeypatch):
    gr = pytest.importorskip("unsloth_zoo.temporary_patches.gpt_oss_routed")
    from unsloth_zoo.temporary_patches import gpt_oss as G

    if not G._is_transformers_v5():
        pytest.skip("the split-LoRA gpt-oss experts forward is transformers 5.x only")
    # patch_gpt_oss_moe_for_lora (the stash-reading experts forward) installs only for a 16-bit gpt-oss load.
    monkeypatch.setenv("UNSLOTH_MODEL_NAME", "gpt_oss")
    calls = []

    def fake_routed(wrapper, x, router_indices, routing_weights):
        # The Triton kernel reads the adapters off the wrapper chain and never runs the experts forward.
        calls.append(wrapper)
        original = next(m._original_param_wrapper_forward for m in _copies()
                        if getattr(m, "_original_param_wrapper_forward", None) is not None)
        experts = wrapper.get_base_layer()
        saved = {k: experts.__dict__.pop(k) for k in list(experts.__dict__) if k.startswith("_unsloth_lora_")}
        # The stand-in reaches the experts forward, which records a stash read the real kernel never makes: keep
        # the read record as it was, else the probe sees a read and the test passes without the fix.
        reads = experts.__dict__.get(MU._MOE_LORA_STASH_READ_ATTR)
        reads = dict(reads) if reads is not None else None
        try:
            return original(wrapper, x, router_indices, routing_weights)
        finally:
            experts.__dict__.update(saved)
            if reads is None:
                experts.__dict__.pop(MU._MOE_LORA_STASH_READ_ATTR, None)
            else:
                experts.__dict__[MU._MOE_LORA_STASH_READ_ATTR] = reads

    monkeypatch.setattr(gr, "routed_bf16_eligible", lambda experts, x: not torch.is_grad_enabled())
    monkeypatch.setattr(gr, "routed_bf16_forward", fake_routed)
    return calls


def test_outer_expert_adapter_gets_grads_when_the_inner_wrapper_could_route(routed_like_l4):
    _apply_patches()
    ids = torch.randint(0, 128, (2, 12), generator = torch.Generator().manual_seed(0))

    reference = _lora(_tiny_gpt_oss())
    # Reference: the same adapters with routing off (what T4 / CPU run).
    gr = sys.modules["unsloth_zoo.temporary_patches.gpt_oss_routed"]
    eligible = gr.routed_bf16_eligible
    gr.routed_bf16_eligible = lambda experts, x: False
    try:
        reference(input_ids = ids, labels = ids).loss.backward()
    finally:
        gr.routed_bf16_eligible = eligible
    want = _outer_lora_grads(reference)
    assert want and all(g is not None for g in want.values())

    model = _lora(_tiny_gpt_oss())
    model.load_state_dict(reference.state_dict())
    model(input_ids = ids, labels = ids).loss.backward()
    got = _outer_lora_grads(model)
    assert set(got) == set(want)
    missing = [name for name, g in got.items() if g is None]
    assert not missing, f"outer expert adapter got no grads: {missing}"
    for name in want:
        torch.testing.assert_close(got[name], want[name], rtol = 1e-4, atol = 1e-6)
    # The outer adapter stays on the separated (stash) path, as without routing.
    experts = model.base_model.model.model.layers[0].mlp.experts.get_base_layer()
    assert MU.moe_lora_forward_applies_stash(experts, "down_proj") is True
    # Eval decode still routes at the outermost wrapper only.
    routed_like_l4.clear()
    with torch.no_grad():
        model(input_ids = ids[:, :1])
    assert routed_like_l4 and all(not getattr(w, "_unsloth_inner_expert_wrapper", False) for w in routed_like_l4)


def test_probe_inside_parametrize_cache_leaves_no_detached_entry():
    """A no_grad probe inside PEFT's parametrize.cached() must not leave its no-history weight in the cache."""
    import torch.nn.utils.parametrize as P

    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = torch.nn.Parameter(torch.randn(2, 4, 4))

        def forward(self, x, *args, **kwargs):
            return x @ self.down_proj[0]

    class Delta(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))

        def forward(self, w):
            return w * self.scale

    class Wrapper(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base_layer = base

    experts = Experts()
    delta = Delta()
    P.register_parametrization(experts, "down_proj", delta, unsafe = True)
    wrapper = Wrapper(experts)
    x = torch.randn(3, 4)
    with P.cached():
        MU._measure_moe_lora_stash_read(wrapper, experts, "gate_up_proj", x, (), {})
        out = experts(x).sum()
    out.backward()
    assert delta.scale.grad is not None

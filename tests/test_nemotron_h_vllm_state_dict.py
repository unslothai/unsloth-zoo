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


import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch


def _linear(weight, output_sizes = None, bias = None):
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(weight.clone(), requires_grad = False)
    module.bias = None if bias is None else torch.nn.Parameter(bias.clone(), requires_grad = False)
    if output_sizes is not None:
        module.output_sizes = output_sizes
    return module


def _pad_vocab(weight, multiple = 64):
    # vLLM pads embed_tokens / lm_head rows to a multiple of 64 even at TP = 1
    rows = -(-weight.shape[0] // multiple) * multiple
    return torch.cat([weight, weight.new_zeros(rows - weight.shape[0], *weight.shape[1:])])


def _vllm_nemotron_h_from_hf(hf, backend = "TRITON"):
    """Mirrors vllm/model_executor/models/nemotron_h.py and mamba_mixer2.py at TP = 1."""
    sd = hf.state_dict()
    config = hf.config
    root = torch.nn.Module()
    model = torch.nn.Module()
    model.embed_tokens = _linear(_pad_vocab(sd["model.embeddings.weight"]))
    model.norm_f = _linear(sd["model.norm_f.weight"])
    layers = torch.nn.ModuleList()
    for kk, block in enumerate(config.hybrid_override_pattern):
        p = f"model.layers.{kk}.mixer"
        layer = torch.nn.Module()
        layer.norm = _linear(sd[f"model.layers.{kk}.norm.weight"])
        mixer = torch.nn.Module()
        if block == "M":
            mixer.in_proj = _linear(sd[f"{p}.in_proj.weight"], [1, 1, 1, 1, 1])
            mixer.conv1d = _linear(sd[f"{p}.conv1d.weight"], [1, 1, 1], sd.get(f"{p}.conv1d.bias"))
            mixer.out_proj = _linear(sd[f"{p}.out_proj.weight"])
            mixer.norm = _linear(sd[f"{p}.norm.weight"])
            mixer.A = torch.nn.Parameter(-torch.exp(sd[f"{p}.A_log"].float()), requires_grad = False)
            mixer.D = torch.nn.Parameter(sd[f"{p}.D"].clone(), requires_grad = False)
            mixer.dt_bias = torch.nn.Parameter(sd[f"{p}.dt_bias"].clone(), requires_grad = False)
        elif block == "*":
            q, k, v = (sd[f"{p}.{x}_proj.weight"] for x in "qkv")
            mixer.qkv_proj = _linear(torch.cat([q, k, v]), [q.shape[0], k.shape[0], v.shape[0]])
            mixer.o_proj = _linear(sd[f"{p}.o_proj.weight"])
        elif block == "E":
            mixer.gate = _linear(sd[f"{p}.gate.weight"])
            mixer.gate.e_score_correction_bias = torch.nn.Parameter(
                sd[f"{p}.gate.e_score_correction_bias"].float().clone(), requires_grad = False,
            )
            mixer.shared_experts = torch.nn.Module()
            mixer.shared_experts.up_proj = _linear(sd[f"{p}.shared_experts.up_proj.weight"])
            mixer.shared_experts.down_proj = _linear(sd[f"{p}.shared_experts.down_proj.weight"])
            mixer.fc1_latent_proj = mixer.fc2_latent_proj = None
            if f"{p}.fc1_latent_proj.weight" in sd:
                mixer.fc1_latent_proj = _linear(sd[f"{p}.fc1_latent_proj.weight"])
                mixer.fc2_latent_proj = _linear(sd[f"{p}.fc2_latent_proj.weight"])
            routed = torch.nn.Module()
            # Non-gated relu2 experts: w13 holds only up_proj
            routed.w13_weight = torch.nn.Parameter(sd[f"{p}.experts.up_proj"].clone(), requires_grad = False)
            routed.w2_weight = torch.nn.Parameter(sd[f"{p}.experts.down_proj"].clone(), requires_grad = False)
            routed.quant_method = types.SimpleNamespace(unquantized_backend = types.SimpleNamespace(name = backend))
            mixer.experts = torch.nn.Module()
            mixer.experts.routed_experts = routed
        else:
            mixer.up_proj = _linear(sd[f"{p}.up_proj.weight"])
            mixer.down_proj = _linear(sd[f"{p}.down_proj.weight"])
        layer.mixer = mixer
        layers.append(layer)
    model.layers = layers
    root.model = model
    root.lm_head = _linear(_pad_vocab(sd["lm_head.weight"]))
    root.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    engine = types.SimpleNamespace(model_executor = types.SimpleNamespace(
        driver_worker = types.SimpleNamespace(model_runner = types.SimpleNamespace(model = root))))
    return types.SimpleNamespace(llm_engine = engine)


def _tiny_nemotron_h(monkeypatch, vocab_size = 64, moe_latent_size = None):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "NemotronHConfig"):
        pytest.skip("transformers has no Nemotron-H")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))
    config = transformers.NemotronHConfig(
        vocab_size = vocab_size, moe_latent_size = moe_latent_size, hidden_size = 16, intermediate_size = 32, num_attention_heads = 4,
        num_key_value_heads = 2, head_dim = 4, mamba_num_heads = 4, mamba_head_dim = 4, n_groups = 1,
        ssm_state_size = 8, conv_kernel = 4, expand = 1, n_routed_experts = 4, num_experts_per_tok = 2,
        moe_intermediate_size = 8, moe_shared_expert_intermediate_size = 8, n_shared_experts = 1,
        layers_block_type = ["mamba", "attention", "moe", "mlp"], tie_word_embeddings = False,
    )
    torch.manual_seed(0)
    hf = transformers.NemotronHForCausalLM(config)
    for param in hf.parameters():
        torch.nn.init.normal_(param)
    for name, buffer in hf.named_buffers():
        if name.endswith("e_score_correction_bias"):
            torch.nn.init.normal_(buffer)
    return hf, config


# vocab 60: vLLM's padded rows must be trimmed; moe_latent_size: Super / Ultra experts run at the latent width
@pytest.mark.parametrize("vocab_size, moe_latent_size", [(64, None), (60, None), (64, 12)])
def test_nemotron_h_vllm_state_dict_matches_hf(monkeypatch, vocab_size, moe_latent_size):
    from unsloth_zoo import vllm_utils
    hf, config = _tiny_nemotron_h(monkeypatch, vocab_size, moe_latent_size)
    state_dict, _ = vllm_utils._get_vllm_state_dict(
        _vllm_nemotron_h_from_hf(hf), return_state_dict = True, config = config,
    )
    expected = hf.state_dict()
    assert set(expected) - set(state_dict) == set()
    for key, value in expected.items():
        if key.endswith(".A_log"):
            # vLLM keeps A = -exp(A_log), so A_log comes back through a float32 log
            torch.testing.assert_close(state_dict[key].to(value.dtype), value)
        else:
            assert torch.equal(state_dict[key], value), key


def test_nemotron_h_reordered_expert_layout_raises(monkeypatch):
    from unsloth_zoo import vllm_utils
    hf, config = _tiny_nemotron_h(monkeypatch)
    with pytest.raises(NotImplementedError, match = "FLASHINFER_CUTLASS"):
        vllm_utils._get_vllm_state_dict(
            _vllm_nemotron_h_from_hf(hf, backend = "FLASHINFER_CUTLASS"), return_state_dict = True, config = config,
        )

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

"""Inkling applies embed_norm once before the first decoder layer (transformers 5.17 norms twice, #47827)."""
import importlib
import os
import subprocess
import sys
import textwrap

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

modeling_inkling = pytest.importorskip("transformers.models.inkling.modeling_inkling")
from transformers.models.inkling.configuration_inkling import InklingConfig

# thinkingmachines/Inkling-Small config.json shrunk: same flags and layer schedule, tiny widths
INKLING_SMALL_SHRUNK = {
    "architectures": ["InklingForConditionalGeneration"],
    "model_type": "inkling_mm_model",
    "eos_token_id": 200006,
    "text_config": {
        "model_max_length": 1048576, "hidden_size": 64, "num_hidden_layers": 6, "vocab_size": 160,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "d_rel": 4, "rel_extent": 16,
        "q_bias": False, "o_bias": False, "log_scaling_n_floor": 128000, "log_scaling_alpha": 0.1,
        "rms_norm_eps": 1e-06, "use_embed_norm": True, "local_layer_ids": [0, 1, 2, 3, 4],
        "dense_mlp_idx": 2, "use_sconv": True, "sconv_kernel_size": 4, "unpadded_vocab_size": 150,
        "logits_mup_width_multiplier": 16.0, "final_logit_softcapping": None, "swa_head_dim": 16,
        "swa_num_attention_heads": 4, "swa_num_key_value_heads": 2, "sliding_window_size": 8,
        "n_routed_experts": 8, "num_experts_per_tok": 2, "n_shared_experts": 2, "shared_expert_sink": True,
        "dense_intermediate_size": 128, "intermediate_size": 32, "route_scale": 8.0, "use_gate_bias": True,
        "gate_activation": "sigmoid", "norm_after_topk": True, "use_global_scale": True,
    },
    "audio_config": {"decoder_dmodel": 64, "n_mel_bins": 4, "mel_vocab_size": 16},
    "vision_config": {"decoder_dmodel": 64, "patch_size": 40, "temporal_patch_size": 2, "n_channels": 3,
                      "n_layers": 6, "hidden_size": 32, "num_attention_heads": 2},
    "mtp_config": {"num_nextn_predict_layers": 2, "chain_hidden_post_norm": False, "local_layer_ids": [0]},
}


def _apply_zoo_patches():
    module = importlib.import_module("unsloth_zoo.temporary_patches.inkling")
    for name in ("patch_inkling_text_config", "patch_inkling_double_embed_norm"):
        patch = getattr(module, name, None)
        if patch is not None:
            patch()


def _device():
    # causal_conv1d, used when installed, is CUDA only
    return "cuda" if torch.cuda.is_available() else "cpu"


def _tiny_model(device = None, dtype = torch.float32):
    config = InklingConfig(**{k: v for k, v in INKLING_SMALL_SHRUNK.items() if k not in ("architectures", "model_type")})
    torch.manual_seed(0)
    model = modeling_inkling.InklingForConditionalGeneration(config).to(device or _device(), dtype).eval()
    norm_weight = next(p for n, p in model.named_parameters() if n.endswith("embed_norm.weight"))
    with torch.no_grad():
        # the real checkpoints hold an embed_norm far from 1 (Inkling-Small: mean 0.07, max 3.45)
        norm_weight.copy_(torch.rand_like(norm_weight) * 3 + 0.05)
    return model


def _text_path_logits(model, input_ids):
    language_model = model.model.language_model
    raw = torch.nn.functional.embedding(input_ids, model.get_input_embeddings().weight)
    norm_weight = next(p for n, p in model.named_parameters() if n.endswith("embed_norm.weight"))
    eps = model.config.text_config.rms_norm_eps
    normed = raw * torch.rsqrt(raw.pow(2).mean(-1, keepdim=True) + eps) * norm_weight
    return normed


def test_inkling_small_config_moe_width():
    _apply_zoo_patches()
    config = InklingConfig(**{k: v for k, v in INKLING_SMALL_SHRUNK.items() if k not in ("architectures", "model_type")})
    assert config.text_config.intermediate_size == 128
    assert config.text_config.moe_intermediate_size == 32


@pytest.mark.parametrize("entry", ["conditional_generation", "inkling_model"])
def test_inkling_embed_norm_applied_once(entry):
    _apply_zoo_patches()
    model = _tiny_model()
    device = next(model.parameters()).device
    captured = {}

    def capture(module, args, kwargs):
        captured.setdefault("h", (args[0] if args else kwargs["hidden_states"]).detach())

    handle = model.model.language_model.layers[0].register_forward_pre_hook(capture, with_kwargs = True)
    input_ids = torch.randint(0, 150, (2, 12), device = device)
    try:
        with torch.no_grad():
            if entry == "conditional_generation":
                model(input_ids = input_ids, use_cache = False)
            else:
                model.model(input_ids = input_ids, use_cache = False)
    finally:
        handle.remove()
    with torch.no_grad():
        expected = _text_path_logits(model, input_ids)
    torch.testing.assert_close(captured["h"], expected, rtol = 1e-5, atol = 1e-5)


def test_inkling_logits_match_single_norm_reference():
    _apply_zoo_patches()
    model = _tiny_model()
    device = next(model.parameters()).device
    input_ids = torch.randint(0, 150, (2, 20), device = device)
    with torch.no_grad():
        logits = model(input_ids = input_ids, use_cache = False).logits
        text_out = model.model.language_model(input_ids = input_ids, use_cache = False).last_hidden_state
        reference = model.lm_head(text_out / model.config.text_config.logits_mup_width_multiplier)
        reference = reference[..., : model.config.text_config.unpadded_vocab_size]
    torch.testing.assert_close(logits, reference, rtol = 1e-5, atol = 1e-5)


def test_inkling_generate_matches_single_norm_reference():
    _apply_zoo_patches()
    model = _tiny_model()
    device = next(model.parameters()).device
    input_ids = torch.randint(0, 150, (1, 10), device = device)
    with torch.no_grad():
        out = model.generate(input_ids = input_ids, max_new_tokens = 6, do_sample = False)
        full = model(input_ids = out[:, :-1], use_cache = False).logits
    assert torch.equal(full[0, 9:].argmax(-1), out[0, 10:])


def test_inkling_patch_idempotent():
    _apply_zoo_patches()
    first = modeling_inkling.InklingModel.forward
    _apply_zoo_patches()
    assert modeling_inkling.InklingModel.forward is first


def test_inkling_patch_gate():
    module = importlib.import_module("unsloth_zoo.temporary_patches.inkling")
    gate = module._transformers_has_double_embed_norm_release
    assert gate("5.17.0") and gate("5.17.1") and gate("5.17.0.dev0")
    for version in ("4.57.6", "5.14.0", "5.16.0", "5.18.0", "5.18.0.dev0", "6.0.0"):
        assert not gate(version)


def test_inkling_patch_noop_when_gate_off():
    code = textwrap.dedent("""
        import os
        os.environ["UNSLOTH_IS_PRESENT"] = "1"
        import unsloth_zoo.temporary_patches.inkling as inkling_patches
        from transformers.models.inkling import modeling_inkling as mi
        # simulate a release outside 5.17.x (importing unsloth_zoo re-reads transformers.__version__)
        inkling_patches._transformers_has_double_embed_norm_release = lambda version = None: False
        before = (mi.InklingModel.forward, mi.InklingTextModel.forward)
        assert not getattr(before[0], "_unsloth_patched", False)
        inkling_patches.patch_inkling_double_embed_norm()
        assert (mi.InklingModel.forward, mi.InklingTextModel.forward) == before
        print("NOOP_OK")
    """)
    env = dict(os.environ, PYTHONPATH = os.pathsep.join([os.path.dirname(os.path.dirname(os.path.abspath(__file__)))] + sys.path))
    result = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True, env = env)
    assert "NOOP_OK" in result.stdout, result.stderr[-2000:]

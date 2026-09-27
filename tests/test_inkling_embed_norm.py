"""Inkling must apply embed_norm exactly once before the first decoder layer.

transformers 5.17.0 norms the token embeddings in InklingModel.forward and again in
InklingTextModel.forward, which breaks every real checkpoint (Inkling-Small wikitext PPL 1099
instead of about 31). temporary_patches/inkling.py removes the duplicate norm.
"""
import importlib
import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

modeling_inkling = pytest.importorskip("transformers.models.inkling.modeling_inkling")
from transformers.models.inkling.configuration_inkling import InklingConfig


def _apply_zoo_patches():
    module = importlib.import_module("unsloth_zoo.temporary_patches.inkling")
    for name in ("patch_inkling_text_config", "patch_inkling_double_embed_norm"):
        patch = getattr(module, name, None)
        if patch is not None:
            patch()


def _tiny_model(device):
    text_config = dict(
        vocab_size=128, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=16,
        sliding_window_size=8, rel_extent=16, d_rel=4, local_layer_ids=[0],
        dense_mlp_idx=1, dense_intermediate_size=96, intermediate_size=32,
        n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
    )
    config = InklingConfig(
        text_config=text_config,
        vision_config=dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=2),
        audio_config=dict(n_mel_bins=4, mel_vocab_size=8),
    )
    torch.manual_seed(0)
    model = modeling_inkling.InklingForConditionalGeneration(config).to(device).eval()
    return model


@pytest.mark.parametrize("entry", ["conditional_generation", "inkling_model"])
def test_inkling_embed_norm_applied_once(entry):
    _apply_zoo_patches()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _tiny_model(device)
    text_config = model.config.text_config
    # The published MoE width comes from intermediate_size in the original config layout.
    assert text_config.moe_intermediate_size == 32

    norm_weight = next(p for n, p in model.named_parameters() if n.endswith("embed_norm.weight"))
    with torch.no_grad():
        norm_weight.copy_(torch.rand_like(norm_weight) * 3 + 0.1)  # far from 1, as in the checkpoints
    embed_weight = model.get_input_embeddings().weight

    captured = {}
    first_layer = model.model.language_model.layers[0]

    def capture(module, args, kwargs):
        captured.setdefault("h", (args[0] if args else kwargs["hidden_states"]).detach())

    handle = first_layer.register_forward_pre_hook(capture, with_kwargs=True)
    input_ids = torch.randint(0, text_config.vocab_size, (2, 12), device=device)
    try:
        with torch.no_grad():
            if entry == "conditional_generation":
                model(input_ids=input_ids, use_cache=False)
            else:
                model.model(input_ids=input_ids, use_cache=False)
    finally:
        handle.remove()

    raw = torch.nn.functional.embedding(input_ids, embed_weight).float()
    eps = text_config.rms_norm_eps
    expected = raw * torch.rsqrt(raw.pow(2).mean(-1, keepdim=True) + eps) * norm_weight.float()
    torch.testing.assert_close(captured["h"].float(), expected, rtol=1e-4, atol=1e-4)


def test_inkling_patch_idempotent():
    _apply_zoo_patches()
    first = modeling_inkling.InklingModel.forward
    _apply_zoo_patches()
    assert modeling_inkling.InklingModel.forward is first

"""A norm upcast by _pre_set_compute_dtype must still run in a plain forward.

The loader marks every norm of a model whose RMSNorm computes in float32 (gemma-3, gemma-3n,
gemma-4) with `_pre_set_compute_dtype = torch.float32`, and patch_model_and_tokenizer casts
them. RMSNorm-style modules upcast internally, but nn.LayerNorm / nn.GroupNorm call
F.layer_norm / F.group_norm, which need input and weight in one dtype. So gemma-3's SigLIP
tower raised "expected scalar type BFloat16 but found Float" in any forward outside autocast
(custom loop, manual eval); the trainer hid it under autocast.

Tiny random SigLIP vision tower on CUDA: the CPU layer_norm kernel accepts mixed dtypes, so
the failure only shows on the GPU.
"""

import os

import pytest
import torch

_GATE = "UNSLOTH_ZOO_DISABLE_GPU_INIT"
_prev = os.environ.get(_GATE)
try:
    os.environ.setdefault(_GATE, "1")
    from unsloth_zoo.patching_utils import patch_model_and_tokenizer  # noqa: E402
finally:
    if _prev is None:
        os.environ.pop(_GATE, None)
    else:
        os.environ[_GATE] = _prev

transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"


def _tiny_siglip():
    from transformers import SiglipVisionConfig, SiglipVisionModel
    cfg = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=2, image_size=28, patch_size=14)
    torch.manual_seed(0)
    model = SiglipVisionModel(cfg).to(DEV, torch.bfloat16).eval()
    # What unsloth/models/vision.py does when UNSLOTH_HIGH_PRECISION_LAYERNORM=1.
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            module._pre_set_compute_dtype = torch.float32
    return model


def _pixels():
    torch.manual_seed(1)
    return torch.randn(1, 3, 28, 28, dtype=torch.bfloat16, device=DEV)


def test_plain_forward_after_upcast():
    model = _tiny_siglip()
    patch_model_and_tokenizer(model, None, downcast_rope=False, fix_embeddings=False)
    norms = [m for m in model.modules() if isinstance(m, torch.nn.LayerNorm)]
    assert norms and all(m.weight.dtype == torch.float32 for m in norms)
    out = model(pixel_values=_pixels()).last_hidden_state
    assert out.dtype == torch.bfloat16  # the caller's dtype, not the norm's
    assert torch.isfinite(out.float()).all()


def test_autocast_forward_unchanged():
    ref = _tiny_siglip()
    for m in ref.modules():
        if hasattr(m, "_pre_set_compute_dtype"):
            m.to(m._pre_set_compute_dtype)  # the cast alone, no hooks: the old behaviour
    fixed = _tiny_siglip()
    patch_model_and_tokenizer(fixed, None, downcast_rope=False, fix_embeddings=False)
    with torch.autocast(DEV, dtype=torch.bfloat16):
        a = ref(pixel_values=_pixels()).last_hidden_state
        b = fixed(pixel_values=_pixels()).last_hidden_state
    assert a.dtype == b.dtype
    assert torch.equal(a, b)


def test_matching_dtype_is_untouched():
    model = _tiny_siglip()
    patch_model_and_tokenizer(model, None, downcast_rope=False, fix_embeddings=False)
    ln = next(m for m in model.modules() if isinstance(m, torch.nn.LayerNorm))
    x = torch.randn(2, 32, device=DEV)
    assert torch.equal(ln(x), torch.nn.functional.layer_norm(x, (32,), ln.weight, ln.bias, ln.eps))
    assert "_unsloth_norm_input_dtype" not in ln.__dict__

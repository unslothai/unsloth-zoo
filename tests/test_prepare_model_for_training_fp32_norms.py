"""Regression tests for the minimal fp32-norms-in-full-FT fix.

Pre-fix bug: a dangling `else:` attached to `if train_lm_head:` clobbered
`upcast = True` for layernorms above it, so norm weights ended up in bf16
for bf16 full finetuning. Empirically that caused ~60% of bf16-storage
norm-weight updates to round to zero on writeback when the adam step
(`lr * m / sqrt(v)`) fell below the bf16 ULP at the current weight magnitude.

Scope (intentionally narrow, byte-identical to pristine main outside full-FT):
  * `full_finetuning=True` AND `train_layernorms=True`: norm weights identified
    by the existing matcher (`"norm." in name or "_layernorm" in name`) are
    upcast to float32.
  * `full_finetuning=False` (LoRA / QLoRA): unchanged.
  * `bias`, `lm_head`, `embed_tokens`: NOT in scope; stay at compute dtype
    (matches pristine main's intent for those classes).
  * If another mechanism already manages a norm module via
    `_pre_set_compute_dtype` (e.g. UNSLOTH_HIGH_PRECISION_LAYERNORM), the fix
    defers and does NOT overwrite that decision.
  * `UNSLOTH_DISABLE_FLOAT32_UPCAST=1` reproduces pre-fix bf16-norm behaviour.

The mock mirrors HF's `LlamaForCausalLM.model.layers[0].input_layernorm`-style
shape so the existing `exec("model.layers[0]...")` path in the loop can
actually navigate it (the loop tries `name` first, then falls back to
`model.{name}` -- which is what real HF models hit since named_parameters
already includes the top `model.` prefix).
"""
import os

import pytest
import torch
import torch.nn as nn


class _Cfg:
    def __init__(self, dtype=torch.bfloat16):
        self.torch_dtype = dtype


class _Attn(nn.Module):
    def __init__(self, with_bias=False):
        super().__init__()
        self.q_norm = nn.LayerNorm(8)
        self.k_norm = nn.LayerNorm(8)
        self.q_proj = nn.Linear(8, 8, bias=with_bias)


class _Layer(nn.Module):
    def __init__(self, with_bias=False):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(8)
        self.post_attention_layernorm = nn.LayerNorm(8)
        self.self_attn = _Attn(with_bias=with_bias)


class _VitBlock(nn.Module):
    """ViT-style block with `norm1` / `norm2` naming (Qwen3-VL visual, DINO)."""

    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(8)
        self.norm2 = nn.LayerNorm(8)


class _Visual(nn.Module):
    """ViT-style visual encoder root: `model.visual.blocks[0].norm1/2`."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_VitBlock()])


class _Inner(nn.Module):
    def __init__(self, with_bias=False):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)
        self.layers = nn.ModuleList([_Layer(with_bias=with_bias)])
        self.norm = nn.LayerNorm(8)
        self.visual = _Visual()


class _Tiny(nn.Module):
    """`LlamaForCausalLM`-shaped wrapper: `.model.layers[0]...` paths resolve
    so the existing exec-based caster in `prepare_model_for_training` works."""

    def __init__(self, dtype=torch.bfloat16, with_bias=False):
        super().__init__()
        self.model = _Inner(with_bias=with_bias)
        # Tied lm_head: same Parameter object as embed_tokens.weight.
        self.lm_head = nn.Linear(8, 16, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.to(dtype)
        self.config = _Cfg(dtype=dtype)


def _params(m):
    return dict(m.named_parameters())


@pytest.fixture(autouse=True)
def _clean_env():
    keys = ("UNSLOTH_DISABLE_FLOAT32_UPCAST", "UNSLOTH_HIGH_PRECISION_LAYERNORM",
            "UNSLOTH_UPCAST_LAYERNORM", "UNSLOTH_MIXED_PRECISION")
    for k in keys:
        os.environ.pop(k, None)
    yield
    for k in keys:
        os.environ.pop(k, None)


def _run(full_finetuning, with_bias=False, env=None,
         tag_pre_set_compute_dtype=()):
    """Run prepare_model_for_training on a fresh _Tiny.

    `tag_pre_set_compute_dtype` is an iterable of dotted attribute paths
    (relative to the inner model) to tag with `_pre_set_compute_dtype = fp32`
    BEFORE the loop runs, simulating UNSLOTH_HIGH_PRECISION_LAYERNORM having
    already claimed those modules.
    """
    from unsloth_zoo.training_utils import prepare_model_for_training
    for k, v in (env or {}).items():
        os.environ[k] = v
    m = _Tiny(dtype=torch.bfloat16, with_bias=with_bias)
    for path in tag_pre_set_compute_dtype:
        node = m
        for part in path.split("."):
            node = node[int(part)] if part.isdigit() else getattr(node, part)
        node._pre_set_compute_dtype = torch.float32
    prepare_model_for_training(
        m,
        use_gradient_checkpointing=False,
        use_reentrant=False,
        full_finetuning=full_finetuning,
        train_layernorms=full_finetuning,
        train_embedding=full_finetuning,
        train_lm_head=full_finetuning,
        float32_mixed_precision=(not full_finetuning),
        patch_modules_to_save=False,
    )
    return m, _params(m)


# ---------------- full-FT: norm.weights become fp32 ----------------

def test_full_ft_norm_weights_become_fp32():
    m, p = _run(full_finetuning=True)
    for n in (
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.norm.weight",
    ):
        assert p[n].dtype == torch.float32, \
            f"{n} expected fp32, got {p[n].dtype}"


def test_full_ft_norm_biases_follow_module_cast():
    # The existing exec path is `module.to(dtype)`, so the LayerNorm's bias is
    # cast alongside its weight even though the matcher only inspects names
    # ending in `.weight`. This mirrors pristine main's behaviour and is fine
    # for our purposes -- both halves of the LayerNorm op stay in lockstep.
    m, p = _run(full_finetuning=True)
    for n in (
        "model.layers.0.input_layernorm.bias",
        "model.layers.0.post_attention_layernorm.bias",
        "model.layers.0.self_attn.q_norm.bias",
        "model.layers.0.self_attn.k_norm.bias",
        "model.norm.bias",
    ):
        assert p[n].dtype == torch.float32, \
            f"{n} should follow its norm module's dtype cast, got {p[n].dtype}"


# ---------------- deferral to external _pre_set_compute_dtype ----------------

def test_full_ft_defers_to_pre_set_compute_dtype_on_norm():
    # A norm module owned by an external policy keeps its loaded dtype; the
    # fix MUST NOT overwrite it. Other norms still upcast.
    m, p = _run(full_finetuning=True,
                tag_pre_set_compute_dtype=("model.layers.0.input_layernorm",))
    assert p["model.layers.0.input_layernorm.weight"].dtype == torch.bfloat16
    # Untagged norms still upcast.
    assert p["model.norm.weight"].dtype == torch.float32
    assert p["model.layers.0.self_attn.q_norm.weight"].dtype == torch.float32


def test_full_ft_defers_per_module_not_globally():
    m, p = _run(full_finetuning=True,
                tag_pre_set_compute_dtype=("model.layers.0.self_attn.q_norm",
                                           "model.layers.0.self_attn.k_norm"))
    assert p["model.layers.0.self_attn.q_norm.weight"].dtype == torch.bfloat16
    assert p["model.layers.0.self_attn.k_norm.weight"].dtype == torch.bfloat16
    assert p["model.layers.0.input_layernorm.weight"].dtype == torch.float32


# ---------------- LoRA / QLoRA path: zero behaviour change ----------------

def test_lora_norms_stay_bf16_frozen():
    m, p = _run(full_finetuning=False)
    for n in (
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.norm.weight",
    ):
        assert p[n].dtype == torch.bfloat16
        assert p[n].requires_grad is False


# ---------------- rollback env switch ----------------

def test_disable_float32_upcast_reproduces_pre_fix_bf16():
    m, p = _run(full_finetuning=True,
                env={"UNSLOTH_DISABLE_FLOAT32_UPCAST": "1"})
    for n in (
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.norm.weight",
    ):
        assert p[n].dtype == torch.bfloat16


# ---------------- matcher guard (no broadening from pristine main) ----------

def test_matcher_catches_q_norm_and_k_norm():
    m, p = _run(full_finetuning=True)
    assert p["model.layers.0.self_attn.q_norm.weight"].dtype == torch.float32
    assert p["model.layers.0.self_attn.k_norm.weight"].dtype == torch.float32


def test_matcher_catches_final_model_norm():
    m, p = _run(full_finetuning=True)
    assert p["model.norm.weight"].dtype == torch.float32


def test_matcher_catches_vit_style_norm1_norm2():
    # ViT/DINO/Qwen3-VL visual encoders use `norm1`/`norm2` without a dot
    # separator. The matcher must catch both .weight and .bias of these.
    m, p = _run(full_finetuning=True)
    assert p["model.visual.blocks.0.norm1.weight"].dtype == torch.float32
    assert p["model.visual.blocks.0.norm1.bias"].dtype == torch.float32
    assert p["model.visual.blocks.0.norm2.weight"].dtype == torch.float32
    assert p["model.visual.blocks.0.norm2.bias"].dtype == torch.float32


# ---------------- autocast wrapper: signature / meta-device / deepcopy ------


class _TinyForSig(nn.Module):
    """Model with an `input_ids`/`labels` style forward so we can inspect
    the signature `Trainer._set_signature_columns_if_needed` would see."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(8)
        self.proj = nn.Linear(8, 8)
        self.to(torch.bfloat16)
        self.config = _Cfg(dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.norm(input_ids.to(self.norm.weight.dtype))
        return self.proj(x)


def test_wrapper_preserves_forward_signature():
    """Regression: HF Trainer reads `inspect.signature(model.forward)` to
    decide which dataset columns to keep under `remove_unused_columns=True`.
    A bare `(*args, **kwargs)` wrapper would drop every named column."""
    import inspect
    from unsloth_zoo.training_utils import _wrap_forward_in_bf16_autocast

    m = _TinyForSig()
    _wrap_forward_in_bf16_autocast(m, torch.bfloat16)
    sig = inspect.signature(m.forward)
    names = list(sig.parameters.keys())
    assert "input_ids" in names, names
    assert "attention_mask" in names, names
    assert "labels" in names, names


def test_wrapper_meta_device_does_not_crash():
    """Regression: torch.is_autocast_enabled('meta') raises on
    unsupported device types. We must probe is_autocast_available first."""
    from unsloth_zoo.training_utils import _wrap_forward_in_bf16_autocast

    class _MetaModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Cfg(dtype=torch.bfloat16)

        def forward(self, x):
            return x

    m = _MetaModel()
    _wrap_forward_in_bf16_autocast(m, torch.bfloat16)
    # Passing a meta tensor would previously crash inside the wrapper
    # at torch.is_autocast_enabled('meta'); the reordered guards must
    # treat 'meta' as "autocast unavailable, fall through".
    out = m(torch.zeros(2, device="meta"))
    assert out.device.type == "meta"


def test_wrapper_survives_deepcopy_and_uses_copy_weights():
    """Regression: EMA / model averaging deepcopies the model. The closure-
    based capture pinned `_orig_forward` to the original instance, so the
    copy ran forward against the ORIGINAL weights. Subclass+self-bind via
    types.MethodType-equivalent (class override) must rebind cleanly."""
    import copy
    from unsloth_zoo.training_utils import _wrap_forward_in_bf16_autocast

    class _Owner(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4, bias=False)
            self.config = _Cfg(dtype=torch.bfloat16)

        def forward(self, x):
            return self.lin(x)

    m = _Owner()
    with torch.no_grad():
        m.lin.weight.fill_(1.0)
    _wrap_forward_in_bf16_autocast(m, torch.bfloat16)

    m2 = copy.deepcopy(m)
    with torch.no_grad():
        m2.lin.weight.fill_(2.0)

    # Distinct parameter storage after deepcopy.
    assert m.lin.weight.data_ptr() != m2.lin.weight.data_ptr()

    x = torch.zeros(1, 4, dtype=torch.bfloat16)
    x[0, 0] = 1.0
    # Original sees fills of 1.0 (row sum = 1).
    y1 = m(x)
    # Copy with weights overwritten to 2.0 must see 2.0 (row sum = 2),
    # NOT the original's 1.0. The pre-fix closure leaked to original.
    y2 = m2(x)
    assert torch.allclose(y1.float(), torch.full_like(y1, 1.0, dtype=torch.float32))
    assert torch.allclose(y2.float(), torch.full_like(y2, 2.0, dtype=torch.float32))


def test_wrapper_idempotent():
    """Calling _wrap twice does NOT double-wrap (and the second call is a
    no-op, returning the same model object with the same class)."""
    from unsloth_zoo.training_utils import _wrap_forward_in_bf16_autocast

    m = _TinyForSig()
    _wrap_forward_in_bf16_autocast(m, torch.bfloat16)
    cls_after_first = type(m)
    _wrap_forward_in_bf16_autocast(m, torch.bfloat16)
    assert type(m) is cls_after_first

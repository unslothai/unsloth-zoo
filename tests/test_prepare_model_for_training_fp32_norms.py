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


def test_full_ft_embed_stays_compute_dtype():
    m, p = _run(full_finetuning=True)
    assert p["model.embed_tokens.weight"].dtype == torch.bfloat16


def test_full_ft_lm_head_stays_compute_dtype():
    # lm_head shares storage with embed_tokens (weight tying).
    m, p = _run(full_finetuning=True)
    assert m.lm_head.weight.dtype == torch.bfloat16
    assert m.lm_head.weight is m.model.embed_tokens.weight


def test_full_ft_linear_bias_stays_compute_dtype():
    m, p = _run(full_finetuning=True, with_bias=True)
    assert p["model.layers.0.self_attn.q_proj.bias"].dtype == torch.bfloat16


def test_full_ft_base_linear_stays_compute_dtype():
    m, p = _run(full_finetuning=True)
    assert p["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16


def test_full_ft_all_params_remain_trainable():
    m, p = _run(full_finetuning=True)
    for n, param in p.items():
        assert param.requires_grad, f"{n} should be trainable in full-FT"


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


def test_lora_embed_stays_bf16_frozen():
    m, p = _run(full_finetuning=False)
    qp = p["model.embed_tokens.weight"]
    assert qp.dtype == torch.bfloat16
    assert qp.requires_grad is False


def test_lora_base_linear_stays_bf16_frozen():
    m, p = _run(full_finetuning=False)
    qp = p["model.layers.0.self_attn.q_proj.weight"]
    assert qp.dtype == torch.bfloat16
    assert qp.requires_grad is False


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


def test_full_ft_weight_tying_preserved():
    m, p = _run(full_finetuning=True)
    assert m.lm_head.weight is m.model.embed_tokens.weight

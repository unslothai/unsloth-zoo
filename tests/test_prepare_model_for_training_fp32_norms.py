"""Tests full-FT norm fp32 upcast (dangling else left bf16); norms claimed via _pre_set_compute_dtype are never
overwritten; UNSLOTH_DISABLE_FLOAT32_UPCAST=1 reverts."""
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
    """Llama-shaped so .model.layers[0] paths resolve for prepare_model_for_training's exec-based caster."""

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
    """Runs prepare_model_for_training on _Tiny; tag_pre_set_compute_dtype paths are pre-claimed as fp32 before the loop, as UNSLOTH_HIGH_PRECISION_LAYERNORM would."""
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
    # module.to(dtype) also casts a LayerNorm bias though the matcher only checks .weight names; this matches upstream main and keeps both halves in lockstep.
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


def test_full_ft_defers_to_pre_set_compute_dtype_on_norm():
    # A norm owned by an external policy keeps its loaded dtype and must not be overwritten; other norms still upcast.
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


def test_lora_norms_stay_bf16_frozen():
    m, p = _run(full_finetuning=False)
    for n in (
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.norm.weight",
    ):
        assert p[n].dtype == torch.bfloat16
        assert p[n].requires_grad is False


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
    # ViT/DINO/Qwen3-VL visual encoders name norms norm1/norm2 with no dot separator; the matcher must catch both .weight and .bias of these.
    m, p = _run(full_finetuning=True)
    assert p["model.visual.blocks.0.norm1.weight"].dtype == torch.float32
    assert p["model.visual.blocks.0.norm1.bias"].dtype == torch.float32
    assert p["model.visual.blocks.0.norm2.weight"].dtype == torch.float32
    assert p["model.visual.blocks.0.norm2.bias"].dtype == torch.float32


class _TinyForSig(nn.Module):
    """Forward with input_ids/labels so Trainer._set_signature_columns_if_needed sees a real signature."""

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
    """Trainer reads inspect.signature(model.forward) for columns; a (*args, **kwargs) wrapper drops them."""
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
    """torch.is_autocast_enabled('meta') raises on unsupported devices; probe is_autocast_available first."""
    from unsloth_zoo.training_utils import _wrap_forward_in_bf16_autocast

    class _MetaModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Cfg(dtype=torch.bfloat16)

        def forward(self, x):
            return x

    m = _MetaModel()
    _wrap_forward_in_bf16_autocast(m, torch.bfloat16)
    # Meta device must be treated as autocast unavailable and fall through, guarding against a crash in torch.is_autocast_enabled('meta') inside the wrapper.
    out = m(torch.zeros(2, device="meta"))
    assert out.device.type == "meta"


def test_wrapper_survives_deepcopy_and_uses_copy_weights():
    """EMA deepcopies the model; a closure-pinned _orig_forward ran copies on the original weights, so a subclass with self-bound forward must rebind cleanly."""
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

    assert m.lin.weight.data_ptr() != m2.lin.weight.data_ptr()

    x = torch.zeros(1, 4, dtype=torch.bfloat16)
    x[0, 0] = 1.0
    # Original sees fills of 1.0 (row sum = 1).
    y1 = m(x)
    # The copy must see its own overwritten weights (row sum 2), not the original's 1.0; guards against the forward closure leaking to the original.
    y2 = m2(x)
    assert torch.allclose(y1.float(), torch.full_like(y1, 1.0, dtype=torch.float32))
    assert torch.allclose(y2.float(), torch.full_like(y2, 2.0, dtype=torch.float32))


def test_wrapper_idempotent():
    """Wrapping twice is a no-op: no double-wrap, same model object and class."""
    from unsloth_zoo.training_utils import _wrap_forward_in_bf16_autocast

    m = _TinyForSig()
    _wrap_forward_in_bf16_autocast(m, torch.bfloat16)
    cls_after_first = type(m)
    _wrap_forward_in_bf16_autocast(m, torch.bfloat16)
    assert type(m) is cls_after_first


def test_wrapper_is_picklable():
    """Generated subclass must be module-level so torch.save(model) can pickle it (else PicklingError)."""
    import io
    from unsloth_zoo.training_utils import _wrap_forward_in_bf16_autocast

    m = _TinyForSig()
    _wrap_forward_in_bf16_autocast(m, torch.bfloat16)

    buf = io.BytesIO()
    torch.save(m, buf)  # would raise PicklingError pre-fix
    buf.seek(0)
    m2 = torch.load(buf, weights_only=False)
    assert type(m2).__name__ == type(m).__name__
    out = m2(torch.zeros(2, 8, dtype=torch.bfloat16))
    assert out.shape == (2, 8)


def test_wrapper_unpickles_in_fresh_interpreter(tmp_path):
    """Pickles load in fresh processes via __reduce__ on the importable base, not the runtime symbol."""
    import os, subprocess, sys
    from unsloth_zoo.training_utils import _wrap_forward_in_bf16_autocast
    from _pickle_base import PickleBaseNet

    m = PickleBaseNet()
    _wrap_forward_in_bf16_autocast(m, torch.bfloat16)
    blob = tmp_path / "m.pt"
    torch.save(m, str(blob))

    tests_dir = os.path.dirname(os.path.abspath(__file__))
    # Import unsloth_zoo, not unsloth: the __reduce__ target lives in unsloth_zoo.training_utils, and unsloth is GPU-only so the CPU CI runner cannot import it.
    script = (
        "import sys; sys.path.insert(0, %r);"
        "import unsloth_zoo, torch;"
        "m = torch.load(%r, weights_only=False);"
        "assert type(m).__qualname__.endswith('WithUnslothBf16Autocast'), type(m).__qualname__;"
        "out = m(torch.zeros(2, 8, dtype=torch.bfloat16));"
        "assert tuple(out.shape) == (2, 8);"
        "print('FRESH_OK')"
    ) % (tests_dir, str(blob))
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert "FRESH_OK" in r.stdout, (r.stdout + "\n" + r.stderr)[-2000:]


def test_wrapper_subclass_is_cached_across_instances():
    """Instances of one base class share one generated subclass, giving a single stable pickle identity."""
    from unsloth_zoo.training_utils import _wrap_forward_in_bf16_autocast

    a = _TinyForSig()
    b = _TinyForSig()
    _wrap_forward_in_bf16_autocast(a, torch.bfloat16)
    _wrap_forward_in_bf16_autocast(b, torch.bfloat16)
    assert type(a) is type(b)


def test_matcher_catches_top_level_norm1_norm2():
    """Top-level norm1/norm2 with no leading dot must still match; the pattern uses a trailing dot only."""
    from unsloth_zoo.training_utils import prepare_model_for_training

    class _TopLevelNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(8)
            self.norm2 = nn.LayerNorm(8)
            self.embed_tokens = nn.Embedding(16, 8)
            self.proj = nn.Linear(8, 8)
            self.to(torch.bfloat16)
            self.config = _Cfg(dtype=torch.bfloat16)

        def get_input_embeddings(self):
            return self.embed_tokens

        def forward(self, x):
            return self.proj(x)

    m = _TopLevelNorm()
    prepare_model_for_training(
        m, use_gradient_checkpointing=False,
        full_finetuning=True, train_layernorms=True,
        float32_mixed_precision=False,
    )
    assert m.norm1.weight.dtype == torch.float32
    assert m.norm2.weight.dtype == torch.float32


class _AudioRMSNorm(nn.Module):
    """RMSNorm whose names (norm_out, norm_pre_attn) match no name pattern, like Gemma-4/3n audio."""

    def __init__(self, dim=8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.weight


def test_matcher_catches_norm_by_module_class_when_name_unmatched():
    """Unmatched norm names must fall back to the owning module's class name (Gemma audio RMSNorm)."""
    from unsloth_zoo.training_utils import prepare_model_for_training

    class _AudioBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm_pre_attn = _AudioRMSNorm(8)
            self.norm_post_attn = _AudioRMSNorm(8)
            self.norm_out = _AudioRMSNorm(8)
            self.proj = nn.Linear(8, 8)

    class _AudioModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_tower = _AudioBlock()
            self.embed_tokens = nn.Embedding(16, 8)
            self.to(torch.bfloat16)
            self.config = _Cfg(dtype=torch.bfloat16)

        def get_input_embeddings(self):
            return self.embed_tokens

        def forward(self, x):
            return self.audio_tower.proj(x)

    m = _AudioModel()
    prepare_model_for_training(
        m, use_gradient_checkpointing=False,
        full_finetuning=True, train_layernorms=True,
        float32_mixed_precision=False,
    )
    assert m.audio_tower.norm_pre_attn.weight.dtype == torch.float32
    assert m.audio_tower.norm_post_attn.weight.dtype == torch.float32
    assert m.audio_tower.norm_out.weight.dtype == torch.float32
    assert m.audio_tower.proj.weight.dtype == torch.bfloat16


# ---------------- reviewer-driven regression tests ----------------


def test_wrapper_intercepts_instance_level_forward():
    """Instance-level forward (e.g. runtime patching) must be wrapped, else fp32 norms crash bf16 linears."""
    from unsloth_zoo.training_utils import _wrap_forward_in_bf16_autocast

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(8).to(torch.float32)
            self.lin = nn.Linear(8, 8).to(torch.bfloat16)
            self.config = _Cfg(dtype=torch.bfloat16)

        def _impl(self, x):
            return self.lin(self.norm(x.float()))

        def forward(self, x):
            return self._impl(x)

    m = _M()
    m.forward = m._impl
    _wrap_forward_in_bf16_autocast(m, torch.bfloat16)
    # Must not raise a dtype mismatch: autocast downcasts fp32 norm out to bf16.
    out = m(torch.randn(2, 8))
    assert out.dtype == torch.bfloat16


def test_wrapper_not_installed_when_no_fp32_norm():
    """Do not class-mutate or wrap a bf16 full-FT model without fp32 norms (e.g. train_layernorms=False)."""
    from unsloth_zoo.training_utils import prepare_model_for_training

    m = _Tiny(dtype=torch.bfloat16)
    prepare_model_for_training(
        m, use_gradient_checkpointing=False, use_reentrant=False,
        full_finetuning=True, train_layernorms=False,
        float32_mixed_precision=False,
    )
    assert not getattr(m, "_unsloth_bf16_autocast_wrapped", False)
    assert not type(m).__qualname__.endswith("WithUnslothBf16Autocast")


def test_wrapper_installed_for_external_fp32_norm_even_when_upcast_disabled():
    """External fp32 norms need the wrapper with UNSLOTH_DISABLE_FLOAT32_UPCAST=1, or bf16 linears crash."""
    from unsloth_zoo.training_utils import prepare_model_for_training

    class _RMSNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(8))

        def forward(self, x):
            return self.weight * x

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = _RMSNorm()
            self.proj = nn.Linear(8, 8, bias=False)
            self.embed_tokens = nn.Embedding(16, 8)
            self.to(torch.bfloat16)
            self.norm.to(torch.float32)
            self.norm._pre_set_compute_dtype = torch.float32
            self.config = _Cfg(dtype=torch.bfloat16)

        def get_input_embeddings(self):
            return self.embed_tokens

        def forward(self, x):
            return self.proj(self.norm(x))

    os.environ["UNSLOTH_DISABLE_FLOAT32_UPCAST"] = "1"
    m = _M()
    prepare_model_for_training(
        m, use_gradient_checkpointing=False, use_reentrant=False,
        full_finetuning=True, train_layernorms=True,
        float32_mixed_precision=False,
    )
    assert m.norm.weight.dtype == torch.float32  # external policy preserved
    assert getattr(m, "_unsloth_bf16_autocast_wrapped", False)
    assert m(torch.ones(2, 8, dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_wrapper_preserves_save_pretrained_architecture(tmp_path):
    """save_pretrained must record the base architecture, not the generated wrapper class name."""
    import json
    from transformers import LlamaConfig, LlamaForCausalLM
    from unsloth_zoo.training_utils import prepare_model_for_training

    cfg = LlamaConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
                      num_hidden_layers=1, num_attention_heads=2,
                      num_key_value_heads=2, torch_dtype=torch.bfloat16)
    model = LlamaForCausalLM(cfg).to(torch.bfloat16)
    prepare_model_for_training(
        model, use_gradient_checkpointing=False, use_reentrant=False,
        full_finetuning=True, train_layernorms=True,
        float32_mixed_precision=False,
    )
    assert getattr(model, "_unsloth_bf16_autocast_wrapped", False)
    model.save_pretrained(str(tmp_path))
    arch = json.loads((tmp_path / "config.json").read_text())["architectures"]
    assert arch == ["LlamaForCausalLM"], arch


def test_wrapper_can_be_removed_on_reprepare(tmp_path):
    """Preparing the same object bf16 then fp32 must drop the bf16 wrapper so fp32 compute is not silently downcast to bf16."""
    from unsloth_zoo.training_utils import (
        _wrap_forward_in_bf16_autocast, _unwrap_forward_in_bf16_autocast)

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)
            self.config = _Cfg(dtype=torch.bfloat16)

        def forward(self, x):
            return self.lin(x)

    m = _M()
    _wrap_forward_in_bf16_autocast(m, torch.bfloat16)
    assert getattr(m, "_unsloth_bf16_autocast_wrapped", False)
    _unwrap_forward_in_bf16_autocast(m)
    assert not getattr(m, "_unsloth_bf16_autocast_wrapped", False)
    assert not type(m).__qualname__.endswith("WithUnslothBf16Autocast")


def test_wrapper_device_detection_in_nested_container():
    """A CPU dict/list batch must not be mis-detected as cuda; finding the tensor device avoids crashing on cuda autocast with CPU inputs."""
    from unsloth_zoo.training_utils import _find_tensor_device_type

    x = torch.zeros(2, 8)
    assert _find_tensor_device_type({"input_ids": x}) == "cpu"
    assert _find_tensor_device_type([{"a": [x]}]) == "cpu"
    assert _find_tensor_device_type({"no": "tensors"}) is None


def test_legacy_upcast_layernorm_defers_to_external_policy():
    """The legacy UNSLOTH_UPCAST_LAYERNORM path must honour an external _pre_set_compute_dtype deferral and the broadened norm matcher."""
    from unsloth_zoo.training_utils import prepare_model_for_training

    class _RMSNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(8))

        def forward(self, x):
            return self.weight * x

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = nn.LayerNorm(8)   # name-matched
            self.norm1 = nn.LayerNorm(8)             # union-matched (norm1.)
            self.managed_norm = _RMSNorm()           # externally managed
            self.proj = nn.Linear(8, 8)
            self.embed_tokens = nn.Embedding(16, 8)
            self.to(torch.bfloat16)
            self.managed_norm.to(torch.float32)
            self.managed_norm._pre_set_compute_dtype = torch.float32
            self.config = _Cfg(dtype=torch.bfloat16)

        def get_input_embeddings(self):
            return self.embed_tokens

        def forward(self, x):
            return self.proj(x)

    os.environ["UNSLOTH_UPCAST_LAYERNORM"] = "1"
    os.environ["UNSLOTH_DISABLE_FLOAT32_UPCAST"] = "1"  # only legacy path upcasts
    m = _M()
    prepare_model_for_training(
        m, use_gradient_checkpointing=False, use_reentrant=False,
        full_finetuning=True, train_layernorms=True,
        float32_mixed_precision=False,
    )
    # legacy path upcasts both name- and union-matched norms
    assert m.input_layernorm.weight.dtype == torch.float32
    assert m.norm1.weight.dtype == torch.float32
    # externally managed norm preserved (fp32, untouched)
    assert m.managed_norm.weight.dtype == torch.float32


def test_full_ft_handles_bias_substring_module_names():
    """Module names containing .bias or .weight must not crash prepare_model_for_training; the caster sets param.data directly, so no name-to-module lookup."""
    from unsloth_zoo.training_utils import prepare_model_for_training

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = nn.LayerNorm(8)
            self.bias_proj = nn.Linear(8, 8)      # name contains ".bias"
            self.weight_scale = nn.Linear(8, 8)   # name contains ".weight"
            self.embed_tokens = nn.Embedding(16, 8)
            self.to(torch.bfloat16)
            self.config = _Cfg(dtype=torch.bfloat16)

        def get_input_embeddings(self):
            return self.embed_tokens

        def forward(self, x):
            return self.weight_scale(self.bias_proj(self.input_layernorm(x)))

    m = _M()
    prepare_model_for_training(  # must not raise
        m, use_gradient_checkpointing=False, use_reentrant=False,
        full_finetuning=True, train_layernorms=True,
        float32_mixed_precision=False,
    )
    assert m.input_layernorm.weight.dtype == torch.float32
    assert m.bias_proj.weight.dtype == torch.bfloat16
    assert m.weight_scale.weight.dtype == torch.bfloat16


def test_instance_forward_wrapper_rebinds_on_deepcopy():
    """With an instance-level forward wrapper, deepcopy must rebind the wrapper to the copy's parameters, not leak back to the original."""
    import copy
    from unsloth_zoo.training_utils import _wrap_forward_in_bf16_autocast

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4, bias=False)
            self.config = _Cfg(dtype=torch.bfloat16)

        def _impl(self, x):
            return self.lin(x)

        def forward(self, x):
            return self._impl(x)

    m = _M()
    with torch.no_grad():
        m.lin.weight.fill_(1.0)
    m.forward = m._impl
    _wrap_forward_in_bf16_autocast(m, torch.bfloat16)

    m2 = copy.deepcopy(m)
    with torch.no_grad():
        m2.lin.weight.fill_(2.0)
    assert m.lin.weight.data_ptr() != m2.lin.weight.data_ptr()

    x = torch.zeros(1, 4, dtype=torch.bfloat16)
    x[0, 0] = 1.0
    y1 = m(x)   # original weights = 1.0
    y2 = m2(x)  # copy weights = 2.0 -- must NOT see the original's 1.0
    assert torch.allclose(y1.float(), torch.full_like(y1, 1.0, dtype=torch.float32))
    assert torch.allclose(y2.float(), torch.full_like(y2, 2.0, dtype=torch.float32))


def test_instance_forward_wrapper_is_picklable():
    """Instance-forward wrapper must stay picklable: no instance-bound function in __dict__ (torch.load cannot resolve
    it by import path); forward must be a class attribute rebuilt by __reduce__."""
    import io
    from unsloth_zoo.training_utils import _wrap_forward_in_bf16_autocast
    from _pickle_base import PickleBaseNet

    m = PickleBaseNet()
    m.forward = m._impl  # instance-level forward (bound method, importable base)
    _wrap_forward_in_bf16_autocast(m, torch.bfloat16)
    assert getattr(m, "_unsloth_bf16_autocast_wrapped", False)
    assert "forward" not in m.__dict__  # routed through the subclass, not __dict__

    buf = io.BytesIO()
    torch.save(m, buf)  # would raise (local function) without the subclass route
    buf.seek(0)
    m2 = torch.load(buf, weights_only=False)
    out = m2(torch.zeros(2, 8, dtype=torch.bfloat16))
    assert out.shape == (2, 8)


def test_instance_forward_wrapper_unpickles_in_fresh_interpreter(tmp_path):
    """An instance-forward-wrapped model must load in a fresh interpreter that never ran _wrap_forward_in_bf16_autocast."""
    import os, subprocess, sys
    from unsloth_zoo.training_utils import _wrap_forward_in_bf16_autocast
    from _pickle_base import PickleBaseNet

    m = PickleBaseNet()
    m.forward = m._impl
    _wrap_forward_in_bf16_autocast(m, torch.bfloat16)
    blob = tmp_path / "m.pt"
    torch.save(m, str(blob))

    tests_dir = os.path.dirname(os.path.abspath(__file__))
    # Import unsloth_zoo, not unsloth: the __reduce__ target lives in unsloth_zoo.training_utils, and unsloth is GPU-only so the CPU CI runner cannot import it.
    script = (
        "import sys; sys.path.insert(0, %r);"
        "import unsloth_zoo, torch;"
        "m = torch.load(%r, weights_only=False);"
        "out = m(torch.zeros(2, 8, dtype=torch.bfloat16));"
        "assert tuple(out.shape) == (2, 8);"
        "print('FRESH_OK')"
    ) % (tests_dir, str(blob))
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert "FRESH_OK" in r.stdout, (r.stdout + "\n" + r.stderr)[-2000:]

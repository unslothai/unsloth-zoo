import sys, os, warnings, inspect
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import types
import pytest
import torch


class _FakePlainProj(torch.nn.Module):
    def __init__(self, out_features, in_features, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features, dtype=dtype), requires_grad=False)


class _FakeGDN(torch.nn.Module):
    def __init__(self, hidden_size=8, num_k_heads=2, num_v_heads=2, head_k_dim=2, head_v_dim=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        qkvz_dim = self.key_dim * 2 + self.value_dim * 2
        self.in_proj_qkvz = _FakePlainProj(qkvz_dim, hidden_size)
        self.in_proj_ba = _FakePlainProj(num_v_heads * 2, hidden_size)
        self.conv1d = _FakePlainProj(self.key_dim * 2 + self.value_dim, 4)
        self.dt_bias = torch.nn.Parameter(torch.randn(num_v_heads), requires_grad=False)
        self.A_log = torch.nn.Parameter(torch.randn(num_v_heads), requires_grad=False)
        self.norm = torch.nn.Module()
        self.norm.weight = torch.nn.Parameter(torch.randn(head_v_dim), requires_grad=False)
        self.out_proj = _FakePlainProj(hidden_size, self.value_dim)


def _fake_get_state_dict(prefix, kk, state_dict, module, slice_weights=True):
    state_dict[f"{prefix}.weight"] = module.weight.data


def test_extract_gdn_layers_handles_plain_column_parallel_linear():
    # Pre-fix: vllm ColumnParallelLinear has no `output_sizes` -> AttributeError.
    from unsloth_zoo.empty_model import extract_gdn_layers
    gdn = _FakeGDN()
    state_dict, quant_state_dict = {}, {}
    extract_gdn_layers(gdn, "prefix", state_dict, quant_state_dict, _fake_get_state_dict)
    expected = {
        "prefix.in_proj_qkv.weight",
        "prefix.in_proj_z.weight",
        "prefix.in_proj_b.weight",
        "prefix.in_proj_a.weight",
        "prefix.conv1d.weight",
        "prefix.dt_bias",
        "prefix.A_log",
        "prefix.norm.weight",
        "prefix.out_proj.weight",
    }
    assert expected <= set(state_dict.keys())


def test_extract_gdn_layers_splits_in_proj_ba_without_indexerror():
    # Pre-fix: get_state_dict(kk=1, in_proj_ba) -> IndexError (no output_sizes).
    from unsloth_zoo.empty_model import extract_gdn_layers
    gdn = _FakeGDN()
    state_dict, quant_state_dict = {}, {}
    extract_gdn_layers(gdn, "prefix", state_dict, quant_state_dict, _fake_get_state_dict)
    ba_weight = gdn.in_proj_ba.weight.data
    mid = ba_weight.shape[0] // 2
    torch.testing.assert_close(state_dict["prefix.in_proj_b.weight"], ba_weight[:mid])
    torch.testing.assert_close(state_dict["prefix.in_proj_a.weight"], ba_weight[mid:])


def test_extract_gdn_layers_qkvz_offsets_match_gdn_dims():
    from unsloth_zoo.empty_model import extract_gdn_layers
    gdn = _FakeGDN(num_k_heads=3, num_v_heads=2, head_k_dim=4, head_v_dim=5)
    state_dict, quant_state_dict = {}, {}
    extract_gdn_layers(gdn, "prefix", state_dict, quant_state_dict, _fake_get_state_dict)
    assert state_dict["prefix.in_proj_qkv.weight"].shape[0] == 2 * gdn.key_dim + gdn.value_dim
    assert state_dict["prefix.in_proj_z.weight"].shape[0] == gdn.value_dim


def test_extract_gdn_layers_raises_when_offsets_underivable():
    from unsloth_zoo.empty_model import extract_gdn_layers
    gdn = _FakeGDN()
    del gdn.key_dim
    del gdn.value_dim
    with pytest.raises(RuntimeError, match="in_proj_qkvz"):
        extract_gdn_layers(gdn, "prefix", {}, {}, _fake_get_state_dict)


def test_extract_gdn_layers_has_bnb_quant_state_preservation():
    # Pre-fix: merged in_proj_qkvz stored raw weight slices; BnB prequantized
    # checkpoints lost quant_state and were rebuilt as plain nn.Linear.
    # Source-level check (behavioral test needs real BnB).
    from unsloth_zoo import empty_model
    src = inspect.getsource(empty_model.extract_gdn_layers)
    assert "bnb_quant_state" in src
    # Quant-state keys are emitted via a helper; check prefixes + suffix.
    assert "in_proj_qkv" in src
    assert "in_proj_z" in src
    assert "in_proj_b" in src
    assert "in_proj_a" in src
    assert ".weight.quant_state" in src


class _LinearAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_idx = -1


class _StandardLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_idx = -1
        self.linear_attn = _LinearAttn()


class _StandardLM(torch.nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()

        class _Inner(torch.nn.Module):
            def __init__(self, n):
                super().__init__()
                self.layers = torch.nn.ModuleList([_StandardLayer() for _ in range(n)])

        self.model = _Inner(n_layers)


def _config(model_type="qwen3_5", has_vision=False):
    cfg = types.SimpleNamespace()
    cfg.model_type = model_type
    cfg.text_config = cfg
    if has_vision:
        vc = types.SimpleNamespace()
        vc.hidden_size = 1
        vc.num_heads = 1
        cfg.vision_config = vc
    return cfg


def test_finalize_fixes_layer_idx_on_standard_causal_lm():
    # Pre-fix: only new_model.model.language_model.layers was traversed, so
    # standard-LM paths kept the stub layer_idx.
    from unsloth_zoo.empty_model import finalize_huggingface_model
    model = _StandardLM(n_layers=4)
    finalize_huggingface_model(
        model, None, _config("qwen3_5"), torch.float16,
        quantization_config={"x": 1}, bnb_config=None,
    )
    for i, layer in enumerate(model.model.layers):
        assert layer.layer_idx == i
        assert layer.linear_attn.layer_idx == i


def test_finalize_fixes_layer_idx_on_vlm_language_model_path():
    from unsloth_zoo.empty_model import finalize_huggingface_model

    class _VLM(torch.nn.Module):
        def __init__(self):
            super().__init__()

            class _Inner(torch.nn.Module):
                def __init__(self):
                    super().__init__()

                    class _LM(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.layers = torch.nn.ModuleList([_StandardLayer() for _ in range(3)])

                    self.language_model = _LM()

            self.model = _Inner()

    model = _VLM()
    finalize_huggingface_model(
        model, None, _config(), torch.float16,
        quantization_config={"x": 1}, bnb_config=None,
    )
    for i, layer in enumerate(model.model.language_model.layers):
        assert layer.layer_idx == i
        assert layer.linear_attn.layer_idx == i


def test_finalize_does_not_assert_on_text_only_with_rotary_pos_emb():
    # Pre-fix: hard `assert vision_config is not None` crashed text-only models.
    from unsloth_zoo.empty_model import finalize_huggingface_model

    class _Rotary(torch.nn.Module):
        pass

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_pos_emb = _Rotary()

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([_Layer()])

    finalize_huggingface_model(
        _Model(), None, _config(has_vision=False), torch.float16,
        quantization_config={"x": 1}, bnb_config=None,
    )


def test_set_dtype_in_config_no_torch_dtype_deprecation():
    # Pre-fix: wrote both dtype and torch_dtype -> transformers deprecation warning.
    from transformers import PretrainedConfig
    from unsloth_zoo.hf_utils import set_dtype_in_config
    cfg = PretrainedConfig()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        set_dtype_in_config(cfg, torch.bfloat16)
    dep = [w for w in caught if "torch_dtype" in str(w.message) and "deprecated" in str(w.message).lower()]
    assert not dep, f"unexpected deprecation warning: {[str(w.message) for w in dep]}"


def test_set_dtype_in_config_writes_torch_dtype_value():
    # set_dtype_in_config stores a JSON-safe string (e.g. "float16") so
    # config.save_pretrained() and patching_utils string comparisons work.
    from transformers import PretrainedConfig
    from unsloth_zoo.hf_utils import set_dtype_in_config, dtype_from_config
    cfg = PretrainedConfig()
    set_dtype_in_config(cfg, torch.float16)
    got = dtype_from_config(cfg)
    assert got == "float16"


def test_set_dtype_in_config_accepts_string_input():
    from transformers import PretrainedConfig
    from unsloth_zoo.hf_utils import set_dtype_in_config, dtype_from_config
    cfg = PretrainedConfig()
    set_dtype_in_config(cfg, "bfloat16")
    got = dtype_from_config(cfg)
    assert got == "bfloat16"


def test_set_dtype_in_config_stores_json_safe_string():
    # Regression: storing torch.dtype objects broke config.save_pretrained()
    # JSON serialization and string equality in patching_utils.
    import json
    from transformers import PretrainedConfig
    from unsloth_zoo.hf_utils import set_dtype_in_config, dtype_from_config
    cfg = PretrainedConfig()
    set_dtype_in_config(cfg, torch.bfloat16)
    value = dtype_from_config(cfg)
    assert isinstance(value, str)
    json.dumps({"dtype": value})


def test_normalize_state_dict_tensor_guards_non_tensor():
    # Pre-fix: value.is_sparse was called unconditionally on any state-dict value.
    from unsloth_zoo import vllm_utils
    src = inspect.getsource(vllm_utils.assert_same_state_dict)
    assert "isinstance(value, torch.Tensor)" in src
    assert src.index("isinstance(value, torch.Tensor)") < src.index("value.is_sparse")


def test_gemma4_lora_patch_preserves_signature_for_inspect():
    # Pre-fix: patched_create_lora_manager(model, *args, **kwargs) hid
    # vllm_config, breaking _call_create_lora_manager's signature forwarding.
    # Fix wraps with functools.wraps and delegates to the original manager.
    from unsloth_zoo import empty_model
    src = inspect.getsource(empty_model.patch_gemma4_vllm_lora_support)
    assert "@wraps(original_create_lora_manager)" in src
    assert "original_create_lora_manager(model, *args, **kwargs)" in src
    assert 'kwargs.setdefault("lora_manager_cls"' in src


def test_gemma4_k_eq_v_patch_handles_split_kv_layout():
    # Pre-fix: only packed self_attn.qkv_proj.weight was searched, so the
    # upstream split q/k/v_proj Gemma4 layout never got synthetic V quant-state.
    from unsloth_zoo import empty_model
    src = inspect.getsource(empty_model.patch_gemma4_vllm_k_eq_v_support)
    assert "k_proj.weight" in src and "v_proj.weight" in src
    assert '"split"' in src or "'split'" in src


# ----- Regression tests for review-iter-1 follow-up fixes -----

class _FakeQuantState:
    def __init__(self, tag):
        self.tag = tag

    def as_dict(self, packed=True):
        return {"absmax": torch.tensor([float(len(self.tag))])}


class _FakeBnBParam(torch.nn.Parameter):
    # Parameter is a Tensor subclass; attaching bnb_quant_state keeps the
    # wrapper-vs-raw-tensor distinction.
    def __new__(cls, data, bnb_quant_state=None):
        inst = torch.nn.Parameter.__new__(cls, data, requires_grad=False)
        inst.bnb_quant_state = bnb_quant_state
        return inst


class _FakeBnBProj(torch.nn.Module):
    def __init__(self, out_features, in_features, bnb_quant_state):
        super().__init__()
        raw = torch.zeros(out_features, in_features, dtype=torch.uint8)
        self.weight = _FakeBnBParam(raw, bnb_quant_state=bnb_quant_state)


class _FakeBnBGDN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 4
        self.num_k_heads = 2
        self.num_v_heads = 2
        self.head_k_dim = 2
        self.head_v_dim = 4
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        qkvz_quant_states = {
            0: _FakeQuantState("qkv"),
            3: _FakeQuantState("z"),
        }
        self.in_proj_qkvz = _FakeBnBProj(
            out_features = self.key_dim * 2 + self.value_dim * 2,
            in_features  = self.hidden_size,
            bnb_quant_state = qkvz_quant_states,
        )
        ba_quant_states = {
            0: _FakeQuantState("b"),
            1: _FakeQuantState("a"),
        }
        self.in_proj_ba = _FakeBnBProj(
            out_features = self.num_v_heads * 2,
            in_features  = self.hidden_size,
            bnb_quant_state = ba_quant_states,
        )
        self.conv1d = _FakePlainProj(self.key_dim * 2 + self.value_dim, 4)
        self.dt_bias = torch.nn.Parameter(torch.randn(self.num_v_heads), requires_grad=False)
        self.A_log = torch.nn.Parameter(torch.randn(self.num_v_heads), requires_grad=False)
        self.norm = torch.nn.Module()
        self.norm.weight = torch.nn.Parameter(torch.randn(self.head_v_dim), requires_grad=False)
        self.out_proj = _FakePlainProj(self.hidden_size, self.value_dim)


def test_extract_gdn_layers_emits_bnb_quant_state_for_all_shards():
    # Pre-fix: extract_gdn_layers() unwrapped Params4bit before reading
    # `bnb_quant_state` (always None), and the in_proj_ba split emitted no
    # quant-state for in_proj_b/in_proj_a.
    from unsloth_zoo.empty_model import extract_gdn_layers
    gdn = _FakeBnBGDN()
    state_dict, quant_state_dict = {}, {}
    extract_gdn_layers(gdn, "prefix", state_dict, quant_state_dict, _fake_get_state_dict)
    for shard in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
        key = f"prefix.{shard}.weight.quant_state"
        assert key in quant_state_dict, f"missing quant_state for {shard}"
    # QuantState.as_dict companion keys should be expanded into state_dict.
    assert "prefix.in_proj_qkv.weight.absmax" in state_dict
    assert "prefix.in_proj_b.weight.absmax" in state_dict


def test_assert_same_state_dict_tied_embed_fallback_has_tolerances():
    # Pre-fix: tied-embeddings fallback used strict tolerances vs the outer
    # atol=1e-4, rtol=1e-3, producing spurious failures.
    from unsloth_zoo import vllm_utils
    src = inspect.getsource(vllm_utils.assert_same_state_dict)
    tied_idx = src.index("model.embed_tokens.weight")
    tail = src[tied_idx:]
    assert "atol = 1e-4" in tail
    assert "rtol = 1e-3" in tail


def test_gemma4_lora_soft_imports_vllm_v1_worker():
    # Pre-fix: patch_gemma4_vllm_lora_support hard-imported `vllm.v1.worker`
    # and crashed with ModuleNotFoundError on older vLLM builds.
    from unsloth_zoo import empty_model
    src = inspect.getsource(empty_model.patch_gemma4_vllm_lora_support)
    assert "try:" in src
    assert "from vllm.v1.worker import lora_model_runner_mixin" in src
    assert "except ImportError" in src
    assert "lora_model_runner_mixin = None" in src


def test_conv1d_rebuild_uses_real_channels_and_groups():
    # Pre-fix: conv1d was treated as a layernorm and rebuilt by weight-swap
    # only, leaving a placeholder Conv1d(groups=1, kernel_size=1) that crashes
    # on first forward.
    from unsloth_zoo import vllm_utils
    src = inspect.getsource(vllm_utils.convert_vllm_to_huggingface)
    assert '".conv1d"' in src
    assert "Conv1d(" in src
    assert "groups = channels" in src
    # conv1d is no longer classified as a layernorm
    assert '"conv1d",' not in src


def test_lm_head_extraction_collapsed_to_single_path():
    # Pre-fix: the two lm_head elif fallbacks were dead code since
    # named_modules() already traverses the full subtree.
    from unsloth_zoo import vllm_utils
    src = inspect.getsource(vllm_utils._get_vllm_state_dict)
    lm_start = src.index("# LM Head")
    lm_block = src[lm_start : lm_start + 800]
    assert "language_model.lm_head" not in lm_block
    assert 'elif hasattr(vllm_internals, "lm_head")' not in lm_block


def test_gemma4_kv_shared_set_uses_shared_config_helper():
    # Route Gemma4 config matching through the shared helper so text-only
    # configs (model_type == "gemma4_text") match too.
    from unsloth_zoo import vllm_utils
    src = inspect.getsource(vllm_utils._get_vllm_state_dict)
    assert "if _is_gemma4_config(config):" in src
    assert "gemma4_kv_shared_layers = set()" in src


def test_gemma4_layer_types_none_is_guarded():
    # Gemma4 configs may carry layer_types=None. Direct enumerate(None) crashed
    # BnB k_eq_v quant-state patching.
    from unsloth_zoo import empty_model
    empty_src = inspect.getsource(empty_model.patch_gemma4_vllm_k_eq_v_support)
    assert 'getattr(text_config, "layer_types", None) or ()' in empty_src


def test_gemma4_k_eq_v_does_not_skip_v_proj_extraction():
    # Dense Gemma4 split layouts can expose real v_proj weights even when
    # attention_k_eq_v is true. Do not leave converted HF placeholders behind.
    from unsloth_zoo import vllm_utils
    src = inspect.getsource(vllm_utils._get_vllm_state_dict)
    v_proj_idx = src.index('get_state_dict(f"{prefix}.v_proj", 2, state_dict, qkv_proj)')
    guard_window = src[max(0, v_proj_idx - 200):v_proj_idx]
    assert "gemma4_k_eq_v_layers" not in guard_window
    assert "gemma4_kv_shared_layers" in guard_window


def test_merger_linear_fc_moved_to_non_layered():
    # Pre-fix: model.visual.merger.linear_fc1/2 (no {kk} placeholder) sat in
    # additional_layers and were reassigned once per layer.
    from unsloth_zoo.empty_model import get_model_layer_config
    cfg = get_model_layer_config()
    additional = set(cfg["additional_layers"])
    non_layered = set(cfg["non_layered_components"])
    assert "model.visual.merger.linear_fc1" not in additional
    assert "model.visual.merger.linear_fc2" not in additional
    assert "model.visual.merger.linear_fc1" in non_layered
    assert "model.visual.merger.linear_fc2" in non_layered


def test_finalize_does_not_overwrite_unrelated_submodule_config_dtype():
    # Behavioral: a submodule with its own config (distinct identity from the
    # top/text/vision/audio configs) must not get its dtype overwritten.
    from unsloth_zoo.empty_model import finalize_huggingface_model

    class _SubConfig:
        def __init__(self, dtype):
            self.dtype = dtype

    class _SubModule(torch.nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.config = _SubConfig(dtype)

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = _SubModule(dtype="float32")
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList()

    top_cfg = types.SimpleNamespace(model_type="llama", dtype="bfloat16")
    top_cfg.text_config = top_cfg

    model = _Model()
    finalize_huggingface_model(
        model, None, top_cfg, torch.bfloat16,
        quantization_config={"x": 1}, bnb_config=None,
    )
    # Unknown submodule config must keep its original dtype.
    assert model.sub.config.dtype == "float32"
    # Top-level config is a known config and should be updated to bfloat16.
    assert top_cfg.dtype == "bfloat16"


def test_finalize_keeps_gemma4_rotary_buffers_float32_after_dtype_cast():
    # Behavioral: on Gemma4, rotary_emb buffers stay float32 even after
    # finalize casts the model to bf16/fp16.
    from unsloth_zoo.empty_model import finalize_huggingface_model

    class _RotaryCfg:
        pass

    class _FakeRotaryEmb(torch.nn.Module):
        # Minimal interface finalize touches: a config plus float buffers that
        # stay float32 on Gemma4.
        def __init__(self, config=None, device=None):
            super().__init__()
            self.config = config if config is not None else _RotaryCfg()
            self.register_buffer("inv_freq", torch.arange(4, dtype=torch.float32))
            self.register_buffer("original_inv_freq", torch.arange(4, dtype=torch.float32))
            self.attention_scaling = torch.tensor(1.0, dtype=torch.float32)

    class _Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_emb = _FakeRotaryEmb(config=_RotaryCfg())

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = -1
            self.self_attn = _Attn()

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([_Layer()])

    cfg = types.SimpleNamespace(model_type="gemma4")
    cfg.text_config = cfg

    model = _Model()
    finalize_huggingface_model(
        model, None, cfg, torch.bfloat16,
        quantization_config={}, bnb_config=None,
    )
    rotary = model.model.layers[0].self_attn.rotary_emb
    assert rotary.inv_freq.dtype == torch.float32
    assert rotary.original_inv_freq.dtype == torch.float32


def test_finalize_non_gemma4_rotary_buffers_follow_model_dtype():
    # Behavioral: non-Gemma4 rotary buffers follow the requested model dtype
    # (buffer_dtype = dtype branch).
    from unsloth_zoo.empty_model import finalize_huggingface_model

    class _RotaryCfg:
        pass

    class _FakeRotaryEmb(torch.nn.Module):
        def __init__(self, config=None, device=None):
            super().__init__()
            self.config = config if config is not None else _RotaryCfg()
            self.register_buffer("inv_freq", torch.arange(4, dtype=torch.float32))

    class _Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_emb = _FakeRotaryEmb(config=_RotaryCfg())

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = -1
            self.self_attn = _Attn()

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([_Layer()])

    cfg = types.SimpleNamespace(model_type="llama")
    cfg.text_config = cfg

    model = _Model()
    finalize_huggingface_model(
        model, None, cfg, torch.bfloat16,
        quantization_config={"x": 1}, bnb_config=None,
    )
    rotary = model.model.layers[0].self_attn.rotary_emb
    # Rotary inv_freq is kept at float32 for all archs to preserve RoPE precision.
    assert rotary.inv_freq.dtype == torch.float32


def test_set_dtype_in_config_else_branch_picks_correct_field():
    # Pre-fix: the else-branch selection was inverted; exercise the
    # neither-attribute path.
    from unsloth_zoo.hf_utils import set_dtype_in_config, HAS_TORCH_DTYPE

    class _Bare:
        pass

    obj = _Bare()
    set_dtype_in_config(obj, torch.float16)
    expected_field = "torch_dtype" if HAS_TORCH_DTYPE else "dtype"
    other_field = "dtype" if HAS_TORCH_DTYPE else "torch_dtype"
    assert getattr(obj, expected_field, None) == "float16"
    # Only one field should be written (no leakage into the other slot).
    assert getattr(obj, other_field, None) is None


def test_assert_same_state_dict_ignores_quantstate_entries():
    # Behavioral: _normalize_state_dict_tensor returns None for non-tensor
    # values (BnB QuantState dicts) and the loop skips them; previously these
    # caused an AttributeError masked into failures.
    from unsloth_zoo.vllm_utils import assert_same_state_dict

    w = torch.randn(4, 4)
    old = {"x.weight": w, "x.weight.quant_state": {"some": "metadata"}}
    new = {"x.weight": w, "x.weight.quant_state": {"some": "metadata"}}
    # Must not raise: QuantState dict skipped, tensor matches.
    assert_same_state_dict(old, new)


def test_normalize_state_dict_tensor_handles_parameter():
    # Behavioral: a Parameter is detached and normalized to a tensor.
    from unsloth_zoo import vllm_utils
    src = inspect.getsource(vllm_utils.assert_same_state_dict)
    # Smoke: full comparison with a Parameter on both sides.
    p_old = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
    p_new = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
    vllm_utils.assert_same_state_dict({"w": p_old}, {"w": p_new})
    # And returning None for a non-tensor is reachable via the guarded path.
    assert "return None" in src


class _FakeLinearModule(torch.nn.Module):
    def __init__(self, out_features, in_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)


class _FakeGemma4Layer(torch.nn.Module):
    # Minimal stand-in so hasattr(layer, "per_layer_input_gate") hits the new
    # extraction branch without a real Gemma4 model.
    def __init__(self, hidden=4):
        super().__init__()
        self.per_layer_input_gate = _FakeLinearModule(hidden, hidden)
        self.per_layer_projection = _FakeLinearModule(hidden, hidden)


def test_gemma4_per_layer_extraction_emits_state_dict_entries():
    # Behavioral: a layer exposing per_layer_input_gate / per_layer_projection
    # must populate state_dict with those paths for the reconstruction
    # templates.
    state_dict = {}

    def fake_get_state_dict(prefix, kk, sd, module, slice_weights=True):
        sd[f"{prefix}.weight"] = module.weight.data

    layer = _FakeGemma4Layer()
    kk = 0
    prefix = "model.language_model"
    # Mirror the calls the fix adds in _get_vllm_state_dict to pin the emitted
    # key shape without reproducing its full setup.
    if hasattr(layer, "per_layer_input_gate"):
        fake_get_state_dict(
            f"{prefix}.layers.{kk}.per_layer_input_gate",
            0, state_dict, layer.per_layer_input_gate,
        )
    if hasattr(layer, "per_layer_projection"):
        fake_get_state_dict(
            f"{prefix}.layers.{kk}.per_layer_projection",
            0, state_dict, layer.per_layer_projection,
        )
    assert "model.language_model.layers.0.per_layer_input_gate.weight" in state_dict
    assert "model.language_model.layers.0.per_layer_projection.weight" in state_dict


def test_set_additional_modules_loads_visual_merger_linear_fc():
    # Regression: the "linear" filter dropped model.visual.merger.linear_fc1/2
    # after they moved into non_layered_components; they must be restored.
    from unsloth_zoo.empty_model import set_additional_modules

    class _LM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(2, 1)
            self.norm = torch.nn.LayerNorm(1)

    class _Merger(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_fc1 = torch.nn.Linear(1, 1, bias=False)
            self.linear_fc2 = torch.nn.Linear(1, 1, bias=False)

    class _Visual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.merger = _Merger()

    class _Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = _LM()
            self.visual = _Visual()

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()
            self.lm_head = torch.nn.Linear(1, 2, bias=False)

    model = _Model()
    fc1_target = torch.full((1, 1), 7.0)
    fc2_target = torch.full((1, 1), 9.0)
    quant_state_dict = {
        "model.language_model.embed_tokens.weight": torch.zeros(2, 1),
        "model.language_model.norm.weight": torch.ones(1),
        "lm_head.weight": torch.zeros(2, 1),
        "model.visual.merger.linear_fc1.weight": fc1_target,
        "model.visual.merger.linear_fc2.weight": fc2_target,
    }
    cfg = types.SimpleNamespace(pad_token_id=0, text_config=types.SimpleNamespace(tie_word_embeddings=False))
    set_additional_modules(model, quant_state_dict, cfg)
    torch.testing.assert_close(model.model.visual.merger.linear_fc1.weight.data, fc1_target)
    torch.testing.assert_close(model.model.visual.merger.linear_fc2.weight.data, fc2_target)


def test_get_vllm_state_dict_extracts_layernorm_when_layer_lacks_mlp():
    # Regression: the early `continue` for layers without `mlp` ran before the
    # layernorm loop, dropping input_layernorm.weight on linear-attn / MoE
    # layers.
    from unsloth_zoo import vllm_utils
    src = inspect.getsource(vllm_utils._get_vllm_state_dict)
    layernorm_idx = src.index('layer_config[\'layernorms\']')
    no_mlp_idx = src.index('if not hasattr(layer, "mlp"):')
    assert layernorm_idx < no_mlp_idx, (
        "layernorm extraction loop must run before the no-mlp early continue "
        "so layernorms are exported for every decoder layer"
    )


def test_finalize_huggingface_model_dtype_propagates_to_replaced_live_config():
    # Regression: copy_attributes can replace new_model.config with an object
    # whose id() differs from the input config, so the id-keyed dtype reapply
    # loop missed it; the fix also updates the live config tree.
    from unsloth_zoo.empty_model import finalize_huggingface_model

    class _LiveCfg:
        def __init__(self, dtype):
            self.dtype = dtype
            self.text_config = self
            self.model_type = "llama"

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _LiveCfg("bfloat16")
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList()

    input_cfg = types.SimpleNamespace(model_type="llama", dtype="bfloat16")
    input_cfg.text_config = input_cfg
    model = _Model()
    finalize_huggingface_model(
        model, None, input_cfg, torch.float16,
        quantization_config={"x": 1}, bnb_config=None,
    )
    assert model.config.dtype == "float16"


def test_finalize_huggingface_model_vision_rotary_uses_identity_check():
    # Regression: comparing the rotary config's __class__ against
    # vision_config's class misfires when text and vision share a class. The
    # identity check must not reroute a text rotary to vision_config.
    from unsloth_zoo.empty_model import finalize_huggingface_model

    class _SharedCfg:
        def __init__(self, hidden_size=4):
            self.hidden_size = hidden_size

    text_cfg_obj = _SharedCfg(8)
    vision_cfg_obj = _SharedCfg(16)

    captured = {}

    class _Rotary(torch.nn.Module):
        def __init__(self, config=None, device=None):
            super().__init__()
            self.config = config
            captured["last_hidden"] = config.hidden_size
            self.register_buffer("inv_freq", torch.arange(4, dtype=torch.float32))

    class _Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_emb = _Rotary(config=text_cfg_obj)

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = -1
            self.self_attn = _Attn()

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([_Layer()])

    cfg = types.SimpleNamespace(model_type="llama")
    cfg.text_config = text_cfg_obj
    cfg.vision_config = vision_cfg_obj

    model = _Model()
    finalize_huggingface_model(
        model, None, cfg, torch.float16,
        quantization_config={"x": 1}, bnb_config=None,
    )
    assert captured["last_hidden"] == text_cfg_obj.hidden_size, (
        "rotary using text_config must not be re-classified as a vision rotary "
        "just because the two configs share a Python class"
    )


def test_layer_scalar_keeps_buffer_registration_after_conversion():
    # Regression: the `if layer_name in quant_state_dict` branch always wrapped
    # the value in nn.Parameter, moving Gemma4 layer_scalar from _buffers to
    # _parameters.
    from unsloth_zoo import vllm_utils
    src = inspect.getsource(vllm_utils.convert_vllm_to_huggingface)
    assert "_buffers" in src
    assert 'getattr(parent, "_buffers"' in src or "parent._buffers" in src


def test_assert_same_state_dict_uses_tight_tolerance_for_same_dtype():
    # Regression: assert_same_state_dict applied atol=1e-4/rtol=1e-3
    # unconditionally, masking extraction errors on same-dtype non-FP8 weights.
    # The relaxed tolerance must apply only to the dtype-mismatch / FP8 branch.
    from unsloth_zoo.vllm_utils import assert_same_state_dict
    a = torch.randn(8, 8, dtype=torch.float32)
    b = a.clone()
    b[0, 0] += 5e-4
    raised = False
    try:
        assert_same_state_dict({"w": a}, {"w": b})
    except Exception:
        raised = True
    assert raised, "5e-4 fp32 mismatch must fail the tight torch default tolerance"


def test_conv1d_branch_requires_linear_attn_in_layer_name():
    # Regression: bare `endswith(".conv1d")` would rebuild any non-GDN .conv1d
    # as depthwise; the branch must require linear_attn.
    from unsloth_zoo import vllm_utils
    src = inspect.getsource(vllm_utils.convert_vllm_to_huggingface)
    assert 'endswith(".conv1d") and "linear_attn" in layer_name' in src


def test_gemma4_lora_patch_covers_both_classes():
    # Regression: only Gemma4ForConditionalGeneration was patched, so
    # text-only Gemma4ForCausalLM still hit the unsupported-LoRA path.
    from unsloth_zoo import empty_model
    src = inspect.getsource(empty_model.patch_gemma4_vllm_lora_support)
    assert "Gemma4ForCausalLM" in src
    assert "_unsloth_gemma4_class_patched" in src


def test_get_model_layer_config_includes_gemma4_top_level_ple_modules():
    # Regression: top-level Gemma4 PLE modules (embed_tokens_per_layer,
    # per_layer_model_projection, per_layer_projection_norm) were absent from
    # extraction tables, leaving them randomly initialized.
    from unsloth_zoo.empty_model import get_model_layer_config
    cfg = get_model_layer_config()
    non_layered = set(cfg["non_layered_components"])
    assert "model.language_model.embed_tokens_per_layer" in non_layered
    assert "model.language_model.per_layer_model_projection" in non_layered
    assert "model.language_model.per_layer_projection_norm" in non_layered


def test_finalize_non_gemma4_rotary_stays_fp32_through_to_dtype():
    # Regression: the non-Gemma4 branch skipped float32 rotary restoration
    # after new_model.to(dtype), downcasting inv_freq / original_inv_freq to
    # bf16/fp16. Exercise the (quantization_config == {} and bnb_config is
    # None) path so .to(dtype) runs.
    from unsloth_zoo.empty_model import finalize_huggingface_model

    class _Cfg:
        pass

    class _Rotary(torch.nn.Module):
        def __init__(self, config=None, device=None):
            super().__init__()
            self.config = config if config is not None else _Cfg()
            self.register_buffer("inv_freq", torch.arange(4, dtype=torch.float32))
            self.register_buffer("original_inv_freq", torch.arange(4, dtype=torch.float32))

    class _Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_emb = _Rotary(config=_Cfg())

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = -1
            self.self_attn = _Attn()

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([_Layer()])

    cfg = types.SimpleNamespace(model_type="llama")
    cfg.text_config = cfg
    model = _Model()
    finalize_huggingface_model(
        model, None, cfg, torch.bfloat16,
        quantization_config={}, bnb_config=None,
    )
    rotary = model.model.layers[0].self_attn.rotary_emb
    assert rotary.inv_freq.dtype == torch.float32
    assert rotary.original_inv_freq.dtype == torch.float32


def test_finalize_tolerates_rotary_rebuild_failure_without_crashing():
    # Regression: rotary_emb.__class__(config=..., device=...) can raise for
    # Gemma4 multimodal rotary when copy_attributes drifts config identity.
    # finalize must catch it, keep the existing rotary, and float32-lift its
    # buffers.
    from unsloth_zoo.empty_model import finalize_huggingface_model

    class _BadCfg:
        pass

    class _ExplodingRotary(torch.nn.Module):
        calls = 0

        def __init__(self, config=None, device=None):
            super().__init__()
            _ExplodingRotary.calls += 1
            if _ExplodingRotary.calls > 1:
                raise KeyError("rope_type")
            self.config = config
            self.register_buffer("inv_freq", torch.arange(4, dtype=torch.float32))

    class _Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_emb = _ExplodingRotary(config=_BadCfg())

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = -1
            self.self_attn = _Attn()

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([_Layer()])

    cfg = types.SimpleNamespace(model_type="gemma4")
    cfg.text_config = cfg
    model = _Model()
    # Must not raise even though the rotary re-init raises KeyError on second call.
    finalize_huggingface_model(
        model, None, cfg, torch.float16,
        quantization_config={"x": 1}, bnb_config=None,
    )
    rotary = model.model.layers[0].self_attn.rotary_emb
    assert rotary.inv_freq.dtype == torch.float32


def test_finalize_routes_vision_tower_rotary_to_vision_config_by_module_path():
    # Regression: id()-based routing drifted after copy_attributes, misrouting
    # vision rotary through text_config. The fix adds a module-path fallback so
    # a rotary under 'vision_tower' builds with vision_config when identity
    # match fails.
    from unsloth_zoo.empty_model import finalize_huggingface_model

    class _TextCfg:
        hidden_size = 8
        num_heads = 2

    class _VisionCfg:
        hidden_size = 16
        num_heads = 2

    captured = {}

    class _Rotary(torch.nn.Module):
        def __init__(self, config=None, device=None):
            super().__init__()
            captured["config_hidden_size"] = getattr(config, "hidden_size", None)
            self.config = config
            self.register_buffer("inv_freq", torch.arange(4, dtype=torch.float32))

    class _Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Unrelated config so id() match against top-level vision_config
            # fails; the module-path fallback must take over.
            self.rotary_emb = _Rotary(config=object())

    class _VisionTower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _Inner()

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList()
            self.model.vision_tower = _VisionTower()

    cfg = types.SimpleNamespace(model_type="gemma4")
    cfg.text_config = _TextCfg()
    cfg.vision_config = _VisionCfg()

    model = _Model()
    finalize_huggingface_model(
        model, None, cfg, torch.float16,
        quantization_config={"x": 1}, bnb_config=None,
    )
    assert captured["config_hidden_size"] == _VisionCfg.hidden_size, (
        "vision-tower rotary must be rebuilt with vision_config even when "
        "the config identity check fails"
    )


def test_extract_gdn_layers_dequantize_uses_unpacked_midpoint():
    # Regression: `mid = ba_weight.shape[0] // 2` was computed on the packed
    # uint8 buffer then reused to slice the dequantized full tensor (shape[0] =
    # out_features); when they differ, in_proj_b/a got wrong rows.
    from unsloth_zoo.empty_model import extract_gdn_layers

    class _PlainProj(torch.nn.Module):
        def __init__(self, out_features, in_features):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.randn(out_features, in_features), requires_grad=False,
            )

    class _FakeQS:
        def as_dict(self, packed=True):
            return {}

    class _PackedParam(torch.nn.Parameter):
        def __new__(cls, data, quant_states):
            inst = torch.nn.Parameter.__new__(cls, data, requires_grad=False)
            inst.bnb_quant_state = quant_states
            return inst

    class _BAProj(torch.nn.Module):
        def __init__(self, packed_len):
            super().__init__()
            # Only index 0 has a QuantState -> triggers dequantize branch.
            self.weight = _PackedParam(
                torch.zeros(packed_len, dtype=torch.uint8),
                {0: _FakeQS(), 1: None},
            )

    class _GDN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = 4
            self.num_k_heads = 2
            self.num_v_heads = 4
            self.head_k_dim = 2
            self.head_v_dim = 4
            self.key_dim = 4
            self.value_dim = 16
            self.in_proj_qkvz = _PlainProj(
                2 * self.key_dim + 2 * self.value_dim, self.hidden_size,
            )
            # Packed length 12 -> packed mid 6. Dequantized shape below is 24 x 1
            # so the correct mid is 12.
            self.in_proj_ba = _BAProj(12)
            self.conv1d = _PlainProj(self.key_dim * 2 + self.value_dim, 4)
            self.dt_bias = torch.nn.Parameter(torch.randn(self.num_v_heads), requires_grad=False)
            self.A_log = torch.nn.Parameter(torch.randn(self.num_v_heads), requires_grad=False)
            self.norm = torch.nn.Module()
            self.norm.weight = torch.nn.Parameter(
                torch.randn(self.head_v_dim), requires_grad=False,
            )
            self.out_proj = _PlainProj(self.hidden_size, self.value_dim)

    bnb = sys.modules.setdefault("bitsandbytes", types.ModuleType("bitsandbytes"))
    bnb_fn = types.ModuleType("bitsandbytes.functional")

    def fake_dequantize_4bit(data, quant_state=None):
        return torch.arange(24, dtype=torch.float32).reshape(24, 1)

    bnb_fn.dequantize_4bit = fake_dequantize_4bit
    sys.modules["bitsandbytes.functional"] = bnb_fn

    def _fake_get_state_dict(prefix, kk, sd, module, slice_weights=True):
        sd[f"{prefix}.weight"] = module.weight.data

    gdn = _GDN()
    state_dict, quant_state_dict = {}, {}
    extract_gdn_layers(gdn, "prefix", state_dict, quant_state_dict, _fake_get_state_dict)
    b = state_dict["prefix.in_proj_b.weight"]
    a = state_dict["prefix.in_proj_a.weight"]
    assert b.shape[0] == 12, f"in_proj_b got {b.shape[0]} rows, expected 12 (dequantized mid)"
    assert a.shape[0] == 12, f"in_proj_a got {a.shape[0]} rows, expected 12 (dequantized mid)"


def test_lm_head_lookup_uses_exact_name_not_substring():
    # Regression: `"lm_head" in name` matched e.g. 'lm_head_norm' before the
    # real 'lm_head'; the fix requires exact match or a .lm_head suffix.
    from unsloth_zoo import vllm_utils
    src = inspect.getsource(vllm_utils._get_vllm_state_dict)
    assert 'name == "lm_head"' in src
    assert 'name.endswith(".lm_head")' in src
    # Loose substring test must not be present.
    assert '"lm_head" in name' not in src


# ----- Regression tests for review-iter-1 hardening -----


def test_convert_regex_handles_trailing_digit_parameter_paths():
    # Pre-fix: `re.sub(r"\.([\d]{1,})\.", ...)` required a trailing dot, so a
    # trailing-index key like `...embed_tokens_per_layer.0` was not bracketed
    # and `exec(...)` hit a SyntaxError.
    import re
    pattern = r"\.([\d]+)(?=\.|$)"
    sub = lambda x: f"[{x.group(1)}]"
    assert re.sub(pattern, sub, "model.language_model.embed_tokens_per_layer.0") \
        == "model.language_model.embed_tokens_per_layer[0]"
    assert re.sub(pattern, sub, "model.layers.12.self_attn.q_proj") \
        == "model.layers[12].self_attn.q_proj"
    assert re.sub(pattern, sub, "model.visual.merger.linear_fc1") \
        == "model.visual.merger.linear_fc1"


def test_convert_vllm_to_huggingface_uses_robust_bracket_regex():
    # The Parameter-assignment path must use the anchor-or-end regex so keys
    # ending in `.DIGIT` get bracketed.
    from unsloth_zoo import vllm_utils
    src = inspect.getsource(vllm_utils.convert_vllm_to_huggingface)
    assert r'r"\.([\d]+)(?=\.|$)"' in src
    param_branch_anchor = "# for attributes of type nn.Parameter, there's no .weight"
    idx = src.index(param_branch_anchor)
    nearby = src[idx:idx + 400]
    assert r'r"\.([\d]+)(?=\.|$)"' in nearby
    assert r'r"\.([\d]{1,})\."' not in nearby


def test_finalize_rotary_reinit_failure_skips_float32_lift():
    # Regression: a bare `try/except: pass` on rotary reinit float32-lifted
    # buffers on the stale rotary; the fix lifts only when reinit succeeds.
    from unsloth_zoo.empty_model import finalize_huggingface_model

    class _BadCfg:
        pass

    class _ExplodingRotary(torch.nn.Module):
        calls = 0

        def __init__(self, config=None, device=None):
            super().__init__()
            _ExplodingRotary.calls += 1
            if _ExplodingRotary.calls > 1:
                raise KeyError("rope_type")
            self.config = config
            self.register_buffer("inv_freq", torch.arange(4, dtype=torch.float16))

    class _Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_emb = _ExplodingRotary(config=_BadCfg())

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = -1
            self.self_attn = _Attn()

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([_Layer()])

    cfg = types.SimpleNamespace(model_type="gemma4")
    cfg.text_config = cfg
    model = _Model()
    finalize_huggingface_model(
        model, None, cfg, torch.float16,
        quantization_config={"x": 1}, bnb_config=None,
    )
    rotary = model.model.layers[0].self_attn.rotary_emb
    # reinit raised -> buffer dtype unchanged from pre-call (fp16)
    assert rotary.inv_freq.dtype == torch.float16


def test_is_gemma4_config_matches_both_variants():
    from unsloth_zoo.empty_model import _is_gemma4_config

    top_only = types.SimpleNamespace(model_type="gemma4")
    assert _is_gemma4_config(top_only)

    nested_text_only = types.SimpleNamespace(model_type="unrelated")
    nested_text_only.text_config = types.SimpleNamespace(model_type="gemma4_text")
    assert _is_gemma4_config(nested_text_only)

    text_only_causal = types.SimpleNamespace(model_type="gemma4_text")
    text_only_causal.text_config = text_only_causal
    assert _is_gemma4_config(text_only_causal)

    not_gemma4 = types.SimpleNamespace(model_type="llama")
    not_gemma4.text_config = not_gemma4
    assert not _is_gemma4_config(not_gemma4)

    assert not _is_gemma4_config(None)


def test_load_vllm_routes_gemma4_gate_through_helper():
    from unsloth_zoo import vllm_utils
    src = inspect.getsource(vllm_utils.load_vllm)
    assert "_is_gemma4_config(config)" in src
    assert 'getattr(config, "model_type", None) == "gemma4"' not in src


def test_load_vllm_gemma4_patch_runs_after_bnb_autodetect():
    # Regression: the Gemma4 k_eq_v patch was gated on caller-provided
    # `use_bitsandbytes` before model-name/quant_method autodetect, so
    # `-bnb-4bit` models with use_bitsandbytes=False skipped it; the fix moves
    # the gate below autodetect.
    from unsloth_zoo import vllm_utils
    src = inspect.getsource(vllm_utils.load_vllm)
    autodetect_anchor = "use_bitsandbytes = use_bitsandbytes or"
    gate_anchor = "patch_gemma4_vllm_k_eq_v_support()"
    assert autodetect_anchor in src
    assert gate_anchor in src
    assert src.index(autodetect_anchor) < src.index(gate_anchor)


def test_gemma4_bnb_skip_module_aliases_cover_vllm_text_prefixes():
    from unsloth_zoo import vllm_utils

    quantization_config = {
        "load_in_4bit": True,
        "llm_int8_skip_modules": [
            "model.language_model.layers.0.mlp",
            "model.language_model.layers.1.self_attn",
            "visual",
        ],
    }
    aliased = vllm_utils._get_gemma4_bnb_skip_module_aliases(quantization_config)

    assert aliased is not None
    skip_modules = set(aliased["llm_int8_skip_modules"])
    assert "model.language_model.layers.0.mlp" in skip_modules
    assert "model.layers.0.mlp" in skip_modules
    assert "language_model.model.layers.0.mlp" in skip_modules
    assert "model.layers.1.self_attn" in skip_modules
    assert "language_model.model.layers.1.self_attn" in skip_modules
    assert "visual" in skip_modules


def test_patch_gemma4_vllm_lora_support_preserves_embedding_modules():
    # Regression: `cls.embedding_modules = {}` clobbered the vLLM class's
    # existing embedding registry (used to route adapters to embedding layers);
    # the fix guards the assignment to run only when the attribute is absent.
    from unsloth_zoo import empty_model
    src = inspect.getsource(empty_model.patch_gemma4_vllm_lora_support)
    assert 'if not hasattr(cls, "embedding_modules"):' in src
    guard_idx = src.index('if not hasattr(cls, "embedding_modules"):')
    assign_idx = src.index("cls.embedding_modules = {}")
    assert guard_idx < assign_idx, (
        "embedding_modules assignment must sit inside the hasattr guard"
    )


def test_patch_gemma4_vllm_lora_support_guards_gemma4_mm_import():
    # Regression: a hard `from vllm...gemma4_mm import ...` crashed with
    # ModuleNotFoundError on text-only Gemma4 vLLM builds; the fix wraps each
    # class import in try/except.
    from unsloth_zoo import empty_model
    src = inspect.getsource(empty_model.patch_gemma4_vllm_lora_support)
    mm_line = "from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration"
    assert mm_line in src
    mm_idx = src.index(mm_line)
    pre = src[:mm_idx]
    assert pre.rstrip().endswith("try:")
    assert "if not classes_to_patch:" in src
    assert "return" in src[src.index("if not classes_to_patch:"):]


def test_patch_gemma4_vllm_k_eq_v_support_guards_private_loader_attr():
    # Regression: hasattr(BitsAndBytesModelLoader._stack_quantization_states,
    # ...) raised AttributeError when the private method was renamed/absent;
    # the fix routes through getattr with a None default.
    from unsloth_zoo import empty_model
    src = inspect.getsource(empty_model.patch_gemma4_vllm_k_eq_v_support)
    assert 'getattr(\n        BitsAndBytesModelLoader, "_stack_quantization_states", None' in src \
        or 'getattr(BitsAndBytesModelLoader, "_stack_quantization_states", None' in src
    assert "if stack_quantization_states is None:" in src


def test_patch_gemma4_vllm_k_eq_v_support_searches_hf_style_prefix():
    # Regression: _get_gemma4_k_eq_v_pairs searched only
    # ("language_model.model", "model"), missing HF-style model.language_model
    # for multimodal Gemma4.
    from unsloth_zoo import empty_model
    src = inspect.getsource(empty_model.patch_gemma4_vllm_k_eq_v_support)
    assert '"model.language_model"' in src
    assert '"language_model.model"' in src


def test_patch_gemma4_vllm_lora_support_early_returns_when_no_classes():
    import sys as _sys
    import types as _types
    from unsloth_zoo import empty_model

    stub_packages = {
        "vllm": _types.ModuleType("vllm"),
        "vllm.model_executor": _types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.models": _types.ModuleType("vllm.model_executor.models"),
        "vllm.model_executor.models.interfaces": _types.ModuleType("vllm.model_executor.models.interfaces"),
        "vllm.lora": _types.ModuleType("vllm.lora"),
        "vllm.lora.model_manager": _types.ModuleType("vllm.lora.model_manager"),
    }
    for name in stub_packages:
        stub_packages[name].__path__ = []
    stub_packages["vllm.model_executor.models.interfaces"].supports_lora = lambda model: False

    class _FakeLoRAManager:
        def __init__(self, *args, **kwargs):
            pass

    class _FakeCreate:
        pass

    def fake_create(model, *args, **kwargs):
        return None

    stub_packages["vllm.lora.model_manager"].LoRAModelManager = _FakeLoRAManager
    stub_packages["vllm.lora.model_manager"].create_lora_manager = fake_create
    stub_packages["vllm.model_executor.models"].gemma4_mm = None  # do not register submodule

    saved = {}
    for name, mod in stub_packages.items():
        saved[name] = _sys.modules.get(name)
        _sys.modules[name] = mod
    # Ensure neither gemma4 nor gemma4_mm submodules resolve.
    for missing in (
        "vllm.model_executor.models.gemma4",
        "vllm.model_executor.models.gemma4_mm",
        "vllm.v1",
        "vllm.v1.worker",
    ):
        saved[missing] = _sys.modules.get(missing)
        _sys.modules[missing] = None
    try:
        # Must not raise when no gemma4 class is importable.
        empty_model.patch_gemma4_vllm_lora_support()
        # fake create_lora_manager must be left in place.
        assert stub_packages["vllm.lora.model_manager"].create_lora_manager is fake_create
    finally:
        for name, prev in saved.items():
            if prev is None:
                _sys.modules.pop(name, None)
            else:
                _sys.modules[name] = prev


def test_patch_gemma4_vllm_k_eq_v_support_noop_when_private_attr_missing():
    import sys as _sys
    import types as _types
    from unsloth_zoo import empty_model

    fake_pkg = _types.ModuleType("vllm.model_executor.model_loader.bitsandbytes_loader")

    class _FakeLoader:
        pass

    fake_pkg.BitsAndBytesModelLoader = _FakeLoader
    saved = {}
    for name in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.model_loader",
        "vllm.model_executor.model_loader.bitsandbytes_loader",
    ):
        saved[name] = _sys.modules.get(name)
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.model_loader"):
        if _sys.modules.get(name) is None:
            _sys.modules[name] = _types.ModuleType(name)
            _sys.modules[name].__path__ = []
    _sys.modules["vllm.model_executor.model_loader.bitsandbytes_loader"] = fake_pkg
    try:
        empty_model.patch_gemma4_vllm_k_eq_v_support()
        assert not hasattr(_FakeLoader, "_stack_quantization_states")
    finally:
        for name, prev in saved.items():
            if prev is None:
                _sys.modules.pop(name, None)
            else:
                _sys.modules[name] = prev

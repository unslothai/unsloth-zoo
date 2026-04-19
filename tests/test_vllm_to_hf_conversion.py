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
    # Pre-fix: merged in_proj_qkvz path only stored raw weight slices; BnB prequantized
    # checkpoints lost quant_state metadata and were rebuilt as plain nn.Linear.
    # Behavioral test requires real BnB; source-level check confirms the branch exists.
    from unsloth_zoo import empty_model
    src = inspect.getsource(empty_model.extract_gdn_layers)
    assert "bnb_quant_state" in src
    assert "in_proj_qkv.weight.quant_state" in src
    assert "in_proj_z.weight.quant_state" in src


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
    # standard-LM paths kept layer_idx at the empty-model stub value.
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
    from transformers import PretrainedConfig
    from unsloth_zoo.hf_utils import set_dtype_in_config
    cfg = PretrainedConfig()
    set_dtype_in_config(cfg, torch.float16)
    got = getattr(cfg, "dtype", None) or getattr(cfg, "torch_dtype", None)
    assert got == torch.float16


def test_set_dtype_in_config_accepts_string_input():
    from transformers import PretrainedConfig
    from unsloth_zoo.hf_utils import set_dtype_in_config
    cfg = PretrainedConfig()
    set_dtype_in_config(cfg, "bfloat16")
    got = getattr(cfg, "dtype", None) or getattr(cfg, "torch_dtype", None)
    assert got == torch.bfloat16


def test_normalize_state_dict_tensor_guards_non_tensor():
    # Pre-fix: value.is_sparse was called unconditionally on any state-dict value.
    from unsloth_zoo import vllm_utils
    src = inspect.getsource(vllm_utils.assert_same_state_dict)
    assert "isinstance(value, torch.Tensor)" in src
    assert src.index("isinstance(value, torch.Tensor)") < src.index("value.is_sparse")


def test_gemma4_lora_patch_preserves_signature_for_inspect():
    # Pre-fix: patched_create_lora_manager(model, *args, **kwargs) hid vllm_config,
    # breaking _call_create_lora_manager's signature-based forwarding.
    from unsloth_zoo import empty_model
    src = inspect.getsource(empty_model.patch_gemma4_vllm_lora_support)
    assert "@wraps(original_create_lora_manager)" in src
    assert "lora_manager_cls(model, *args, **kwargs)" in src


def test_gemma4_k_eq_v_patch_handles_split_kv_layout():
    # Pre-fix: only packed self_attn.qkv_proj.weight was searched, so current upstream
    # Gemma4 split q_proj/k_proj/v_proj layout never got synthetic V quant-state.
    from unsloth_zoo import empty_model
    src = inspect.getsource(empty_model.patch_gemma4_vllm_k_eq_v_support)
    assert "k_proj.weight" in src and "v_proj.weight" in src
    assert '"split"' in src or "'split'" in src

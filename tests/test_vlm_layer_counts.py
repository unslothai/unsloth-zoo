"""get_model_layer_counts must match the model_type get_model_type actually returns.

get_model_type prefers `vision_config.model_type`, so a branch written against the
top level name ("mllama", "gemma3", "gemma4") never runs. The counts then fall through
to the causal-LM default of 32, and extract_vision_layers iterates to 32 instead of the
real layer count, silently skipping every weight above index 31. On Llama-3.2-11B-Vision
that drops the cross-attention gates on text layers 33 and 38.

transformers also renames these strings between majors (qwen3_vl -> qwen3_vl_vision in
transformers 5), so both spellings have to be accepted.
"""
import pytest
import torch

from unsloth_zoo.empty_model import (
    QWEN_VL_MERGED_QKV_TYPES,
    extract_vision_layers,
    get_model_layer_counts,
    get_model_type,
)


class Cfg:
    """Minimal stand-in for a transformers config (attribute access only)."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def mllama(text_layers = 40, vision_layers = 32, global_layers = 8, cross = (3, 33, 38)):
    return Cfg(
        model_type = "mllama",
        text_config = Cfg(
            model_type = "mllama_text_model",
            num_hidden_layers = text_layers,
            cross_attention_layers = list(cross),
        ),
        vision_config = Cfg(
            model_type = "mllama_vision_model",
            num_hidden_layers = vision_layers,
            num_global_layers = global_layers,
        ),
    )


def iterate_to(config):
    counts = get_model_layer_counts(config)
    return max(counts.values()) if isinstance(counts, dict) else counts


# ---------------------------------------------------------------- model_type

@pytest.mark.parametrize("top, vision, expected", [
    ("mllama",         "mllama_vision_model",   "mllama_vision_model"),
    ("gemma3",         "siglip_vision_model",   "siglip_vision_model"),
    ("gemma4",         "gemma4_vision",         "gemma4_vision"),
    ("gemma4_unified", "gemma4_unified_vision", "gemma4_unified_vision"),
    ("qwen3_vl",       "qwen3_vl_vision",       "qwen3_vl_vision"),
    ("qwen2_5_vl",     "qwen2_5_vl",            "qwen2_5_vl"),
])
def test_get_model_type_prefers_the_vision_name(top, vision, expected):
    cfg = Cfg(model_type = top, vision_config = Cfg(model_type = vision))
    assert get_model_type(cfg) == expected


# ---------------------------------------------------------------- layer counts

@pytest.mark.parametrize("vision_name", ["mllama", "mllama_vision_model"])
def test_mllama_counts_under_both_spellings(vision_name):
    cfg = mllama()
    cfg.vision_config.model_type = vision_name
    assert get_model_layer_counts(cfg) == {
        "text_layers": 40, "vision_layers": 32, "global_layers": 8,
    }


@pytest.mark.parametrize("vision_name", ["gemma3", "siglip_vision_model"])
def test_gemma3_counts_under_both_spellings(vision_name):
    cfg = Cfg(
        model_type = "gemma3",
        text_config = Cfg(num_hidden_layers = 62),
        vision_config = Cfg(model_type = vision_name, num_hidden_layers = 27),
    )
    assert get_model_layer_counts(cfg) == {"text_layers": 62, "vision_layers": 27}


@pytest.mark.parametrize("vision_name", [
    "gemma4", "gemma4_vision", "gemma4_unified", "gemma4_unified_vision",
])
def test_gemma4_counts_under_every_spelling(vision_name):
    cfg = Cfg(
        model_type = "gemma4",
        text_config = Cfg(num_hidden_layers = 60),
        vision_config = Cfg(model_type = vision_name, num_hidden_layers = 27),
    )
    assert get_model_layer_counts(cfg) == {"text_layers": 60, "vision_layers": 27}


@pytest.mark.parametrize("vision_name", ["qwen3_vl", "qwen3_vl_vision"])
def test_qwen3_vl_counts_under_both_spellings(vision_name):
    cfg = Cfg(
        model_type = "qwen3_vl",
        num_hidden_layers = 36,
        vision_config = Cfg(model_type = vision_name, depth = 27, deepstack_depth = 3),
    )
    assert get_model_layer_counts(cfg) == {
        "text_layers": 36, "vision_layers": 27, "deepstack_layers": 3,
    }


def test_qwen3_vl_vision_still_takes_the_merged_qkv_path():
    # transformers 5 renamed this; dropping out of the tuple would split a qkv that
    # HF keeps merged, producing q/k/v keys vLLM never emits.
    assert "qwen3_vl" in QWEN_VL_MERGED_QKV_TYPES
    assert "qwen3_vl_vision" in QWEN_VL_MERGED_QKV_TYPES
    assert "qwen2_5_vl" in QWEN_VL_MERGED_QKV_TYPES


def test_plain_causal_lm_still_returns_an_int():
    assert get_model_layer_counts(Cfg(model_type = "llama", num_hidden_layers = 16)) == 16


@pytest.mark.parametrize("model_type", ["mllama", "gemma3", "gemma4"])
def test_missing_sub_configs_do_not_raise(model_type):
    # A vision model_type with no text_config used to raise AttributeError.
    assert isinstance(get_model_layer_counts(Cfg(model_type = model_type)), dict)


def test_unknown_vision_model_is_untouched():
    cfg = Cfg(
        model_type = "smolvlm",
        num_hidden_layers = 24,
        vision_config = Cfg(model_type = "smolvlm_vision", num_hidden_layers = 27),
    )
    assert get_model_layer_counts(cfg) == 24


# ---------------------------------------------------------------- end to end

def test_cross_attention_gates_above_layer_31_are_extracted():
    """The regression this all exists for: gates on text layers 33 and 38."""
    cross = [3, 8, 13, 18, 23, 28, 33, 38]
    cfg = mllama(text_layers = 40, cross = cross)

    root = torch.nn.Module()
    root.config = cfg
    root.model = torch.nn.Module()
    lm = torch.nn.Module()
    lm.model = torch.nn.Module()
    lm.model.layers = torch.nn.ModuleList()
    for i in range(40):
        layer = torch.nn.Module()
        if i in cross:
            layer.cross_attn_mlp_gate = torch.nn.Parameter(torch.zeros(1))
            layer.cross_attn_attn_gate = torch.nn.Parameter(torch.zeros(1))
        lm.model.layers.append(layer)
    root.model.language_model = lm

    seen = {}
    state_dict, quant_state_dict = {}, {}

    def record(prefix, kk, sd, proj, slice_weights = True, slice_indices = None):
        seen[prefix] = True

    extract_vision_layers(root, state_dict, quant_state_dict, record)

    found = set(seen) | set(state_dict)
    gates = {k for k in found if "cross_attn" in k}
    assert len(gates) == 2 * len(cross), sorted(gates)
    for layer in (33, 38):
        assert any(f".layers.{layer}.cross_attn_attn_gate" in k for k in gates)
        assert any(f".layers.{layer}.cross_attn_mlp_gate" in k for k in gates)


def test_iteration_bound_covers_the_deepest_layer():
    assert iterate_to(mllama(text_layers = 100)) == 100
    assert iterate_to(mllama(text_layers = 40)) == 40

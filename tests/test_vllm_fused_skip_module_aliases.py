"""Merged 4bit exports must carry skip-module names vLLM can actually match.

Regression coverage for unslothai/unsloth#1886 and #464.

`find_skipped_quantized_modules` walks a Transformers module tree, so a dynamic
4bit merge records leaf projections (`model.layers.0.mlp.gate_proj`). vLLM fuses
those siblings before it consults `llm_int8_skip_modules`, and its
`is_layer_skipped_bnb` matches a skip entry against a module's own dotted
ancestors. `model.layers.0.mlp.gate_up_proj` has no ancestor named
`...mlp.gate_proj`, so vLLM quantizes a layer that was written to disk dense and
dies in `vllm/model_executor/layers/linear.py` on

    assert param_data.shape == loaded_weight.shape

The checkpoints Unsloth publishes to the Hub record the parent module
(`model.layers.0.mlp`), which does match -- which is why the published repo
serves and a user's own merge of the same model does not.
"""

import pytest

try:
    from unsloth_zoo.saving_utils import add_vllm_fused_skip_module_aliases
except ImportError:
    # Pre-fix tree: the merged 4bit save wrote the leaf list through untouched,
    # so identity is exactly the old behaviour and the assertions below report
    # the real defect rather than a collection error.
    def add_vllm_fused_skip_module_aliases(skipped_modules):
        return skipped_modules


def is_layer_skipped_bnb(prefix, llm_int8_skip_modules):
    """Verbatim copy of vLLM's matcher.

    vllm/model_executor/layers/quantization/bitsandbytes.py::is_layer_skipped_bnb
    (identical in vllm 0.11.2 and 0.29.0). Copied so the test pins the contract
    without importing vLLM.
    """
    components = prefix.split(".")
    substr_check = any(m in components for m in llm_int8_skip_modules)
    set_components = set(".".join(components[: i + 1]) for i in range(len(components)))
    prefix_check = len(set(llm_int8_skip_modules) & set_components) != 0
    return substr_check or prefix_check


# What a merged dynamic 4bit save of Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit
# records today: leaf projections only.
MERGED_LEAF_SKIP_MODULES = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
    "model.layers.2.mlp.gate_proj",
    "model.layers.2.mlp.up_proj",
    "model.layers.2.mlp.down_proj",
    "lm_head",
]

# The modules vLLM actually builds for those layers.
VLLM_FUSED_PREFIXES = [
    "model.layers.0.self_attn.qkv_proj",
    "model.layers.0.mlp.gate_up_proj",
    "model.layers.2.mlp.gate_up_proj",
]


def test_raw_leaf_skip_list_is_invisible_to_vllm():
    """Documents the defect: this is why the assertion fires."""
    for prefix in VLLM_FUSED_PREFIXES:
        assert not is_layer_skipped_bnb(prefix, MERGED_LEAF_SKIP_MODULES)


@pytest.mark.parametrize("prefix", VLLM_FUSED_PREFIXES)
def test_aliased_skip_list_is_matched_by_vllm(prefix):
    aliased = add_vllm_fused_skip_module_aliases(MERGED_LEAF_SKIP_MODULES)
    assert is_layer_skipped_bnb(prefix, aliased), (
        f"vLLM would quantize {prefix}, whose weights were saved dense"
    )


@pytest.mark.parametrize(
    "prefix",
    [
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.2.mlp.down_proj",
        "lm_head",
    ],
)
def test_unfused_skipped_modules_still_match(prefix):
    aliased = add_vllm_fused_skip_module_aliases(MERGED_LEAF_SKIP_MODULES)
    assert is_layer_skipped_bnb(prefix, aliased)


@pytest.mark.parametrize(
    "prefix",
    [
        "model.layers.1.self_attn.qkv_proj",
        "model.layers.1.mlp.gate_up_proj",
        "model.layers.1.mlp.down_proj",
        "model.layers.3.self_attn.qkv_proj",
    ],
)
def test_quantized_modules_are_not_skipped(prefix):
    """The aliases must not widen the skip set onto quantized layers."""
    aliased = add_vllm_fused_skip_module_aliases(MERGED_LEAF_SKIP_MODULES)
    assert not is_layer_skipped_bnb(prefix, aliased)


def test_aliases_are_additive_only():
    aliased = add_vllm_fused_skip_module_aliases(MERGED_LEAF_SKIP_MODULES)
    assert set(MERGED_LEAF_SKIP_MODULES).issubset(set(aliased))
    assert set(aliased) - set(MERGED_LEAF_SKIP_MODULES) == {
        "model.layers.0.self_attn.qkv_proj",
        "model.layers.0.mlp.gate_up_proj",
        "model.layers.2.mlp.gate_up_proj",
    }


def test_partially_skipped_fused_module_gets_no_alias():
    """Only gate_proj skipped, up_proj quantized: vLLM cannot express that.

    Emitting the alias would tell vLLM to leave up_proj dense too, which does
    not match what is on disk. Better to leave it unmatched than to corrupt it.
    """
    partial = ["model.layers.7.mlp.gate_proj"]
    assert add_vllm_fused_skip_module_aliases(partial) == partial

    partial_qkv = ["model.layers.7.self_attn.q_proj", "model.layers.7.self_attn.k_proj"]
    assert add_vllm_fused_skip_module_aliases(partial_qkv) == partial_qkv


def test_parent_module_form_is_left_alone():
    """The published-Hub form already works; do not churn it."""
    parent_form = ["lm_head", "model.layers.0.self_attn", "model.layers.0.mlp"]
    assert add_vllm_fused_skip_module_aliases(parent_form) == parent_form
    for prefix in ("model.layers.0.self_attn.qkv_proj", "model.layers.0.mlp.gate_up_proj"):
        assert is_layer_skipped_bnb(prefix, parent_form)


@pytest.mark.parametrize("value", [None, [], ["lm_head"]])
def test_degenerate_inputs(value):
    assert add_vllm_fused_skip_module_aliases(value) == value


def test_idempotent():
    once = add_vllm_fused_skip_module_aliases(MERGED_LEAF_SKIP_MODULES)
    assert add_vllm_fused_skip_module_aliases(once) == once


# --- second mechanism: Transformers >= 4.52 multimodal nesting vs vLLM's -------
# Real, verified on GPU: unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit writes
# `model.language_model.layers.0.mlp`, vLLM builds
# `language_model.model.layers.0.mlp.down_proj`, and vllm 0.11.2 dies with
#   param_data (12451840, 1) uint8  vs  loaded_weight (2560, 9728) bfloat16
# at vllm/model_executor/layers/linear.py:1376 (RowParallelLinear.weight_loader).

try:
    from unsloth_zoo.saving_utils import vllm_compatible_skip_modules
except ImportError:
    vllm_compatible_skip_modules = add_vllm_fused_skip_module_aliases

QWEN3VL_SKIP_MODULES = [
    "lm_head",
    "visual",
    "model.language_model.layers.0.mlp",
    "model.language_model.layers.11.self_attn",
]

QWEN3VL_VLLM_PREFIXES = [
    "language_model.model.layers.0.mlp.down_proj",
    "language_model.model.layers.0.mlp.gate_up_proj",
    "language_model.model.layers.11.self_attn.qkv_proj",
    "language_model.model.layers.11.self_attn.o_proj",
]


def test_multimodal_nesting_is_invisible_to_vllm_without_aliases():
    for prefix in QWEN3VL_VLLM_PREFIXES:
        assert not is_layer_skipped_bnb(prefix, QWEN3VL_SKIP_MODULES)


@pytest.mark.parametrize("prefix", QWEN3VL_VLLM_PREFIXES)
def test_multimodal_nesting_matches_after_aliasing(prefix):
    aliased = vllm_compatible_skip_modules(QWEN3VL_SKIP_MODULES)
    assert is_layer_skipped_bnb(prefix, aliased)


@pytest.mark.parametrize(
    "prefix",
    [
        "language_model.model.layers.1.mlp.down_proj",
        "language_model.model.layers.1.mlp.gate_up_proj",
        "language_model.model.layers.10.self_attn.qkv_proj",
    ],
)
def test_multimodal_aliasing_does_not_widen_onto_quantized_layers(prefix):
    aliased = vllm_compatible_skip_modules(QWEN3VL_SKIP_MODULES)
    assert not is_layer_skipped_bnb(prefix, aliased)


def test_combined_is_additive_and_idempotent():
    once = vllm_compatible_skip_modules(QWEN3VL_SKIP_MODULES)
    assert set(QWEN3VL_SKIP_MODULES).issubset(set(once))
    assert vllm_compatible_skip_modules(once) == once

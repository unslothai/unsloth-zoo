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

# The identity fallback below exists so this file reports the real defect on a
# pre-fix tree instead of a collection error. It must fire ONLY when the module
# imports and the symbol is absent. `unsloth_zoo.saving_utils` imports torch at
# module scope, so on a runner without torch a bare `except ImportError` would
# swap in the stub and turn "this runner ships no torch" into 21 assertion
# failures that read exactly like "the fix is missing" (observed on macos-15).
# Import the module first, and skip the file outright if that is what failed.
try:
    import unsloth_zoo.saving_utils as _saving_utils
except ImportError as _e:
    pytest.skip(
        f"unsloth_zoo.saving_utils is not importable here ({_e}); "
        "this is an environment gap, not a result about the fix",
        allow_module_level = True,
    )

add_vllm_fused_skip_module_aliases = getattr(
    _saving_utils, "add_vllm_fused_skip_module_aliases",
    # Pre-fix tree: the merged 4bit save wrote the leaf list through untouched,
    # so identity is exactly the old behaviour.
    lambda skipped_modules: skipped_modules,
)


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

vllm_compatible_skip_modules = getattr(
    _saving_utils, "vllm_compatible_skip_modules", add_vllm_fused_skip_module_aliases,
)

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


def test_merge_path_actually_writes_the_vllm_compatible_names():
    """The call site is the only production change; everything else is a helper.

    Reverting just `vllm_compatible_skip_modules(...)` back to `skipped_modules`
    at the merge writer leaves every helper and every other test in this file
    passing, so without this test a refactor that drops the wiring ships green
    and silently re-introduces the vLLM shape assertion.

    Asserted against the source of the writer rather than a full merge, which
    needs a real model on a GPU.
    """
    import inspect
    from unsloth_zoo import saving_utils

    source = inspect.getsource(saving_utils)
    marker = 'quantization_config["llm_int8_skip_modules"]'
    assignments = [
        line.strip() for line in source.splitlines() if marker in line
    ]
    assert assignments, "the merge writer no longer sets llm_int8_skip_modules"
    for line in assignments:
        assert "vllm_compatible_skip_modules" in line or line.endswith("\\"), (
            f"llm_int8_skip_modules written without the vLLM aliases: {line}"
        )

    # And the continuation form, which is what the writer currently uses.
    idx = source.index(marker)
    window = source[idx : idx + 200]
    assert "vllm_compatible_skip_modules" in window, (
        "the merge writer does not pass skipped modules through "
        "vllm_compatible_skip_modules"
    )


def test_namespace_alias_never_emits_a_bare_namespace_root():
    """A bare root matches every module beneath it, unquantizing the whole model."""
    from unsloth_zoo.saving_utils import vllm_compatible_skip_modules

    for entry in ("model.language_model.", "model.", "language_model.model."):
        out = vllm_compatible_skip_modules([entry])
        # The caller's own entries are passed through untouched; only the names
        # this helper adds are its responsibility.
        added = [name for name in out if name != entry]
        for name in added:
            assert name.strip(), f"empty alias from {entry!r}"
            assert not name.endswith("."), f"bare namespace root {name!r} from {entry!r}"


# --- the other half of the contract: the aliases must be inert for Transformers ---
# Every assertion above is about vLLM matching more. Nothing above stops an alias
# from also making *Transformers* skip more, and Transformers does not match by
# exact name: `should_convert_module` ends in an unanchored
# `full_name.endswith(key)`, so an alias only has to be a suffix of a real
# Linear's dotted path to silently leave a layer in 16 bit on reload. These
# module trees are the real `nn.Linear` naming of the architectures they name,
# enumerated on the meta device under transformers 5.x.

MODULE_TREES = {
    # text only
    "qwen2": [
        f"model.layers.{i}.{rest}"
        for i in range(4)
        for rest in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                     "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    ] + ["lm_head"],
    # multimodal: SigLIP-style tower, `encoder.layers.N` under `model.vision_tower`
    "gemma3": [
        f"model.language_model.layers.{i}.{rest}"
        for i in range(4)
        for rest in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                     "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    ] + [
        f"model.vision_tower.encoder.layers.{i}.{rest}"
        for i in range(2)
        for rest in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                     "self_attn.out_proj", "mlp.fc1", "mlp.fc2")
    ] + ["model.multi_modal_projector.linear", "lm_head"],
    # multimodal: ViT-style tower with pre-fused qkv, `blocks.N` under `model.visual`
    "qwen3_vl": [
        f"model.language_model.layers.{i}.{rest}"
        for i in range(4)
        for rest in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                     "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    ] + [
        f"model.visual.blocks.{i}.{rest}"
        for i in range(2)
        for rest in ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2")
    ] + ["lm_head"],
    # a tower that deliberately repeats the text stack's own suffixes
    "adversarial_suffix": [
        f"model.language_model.layers.{i}.{rest}"
        for i in range(2)
        for rest in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    ] + [
        # ends in "...model.layers.0.mlp.gate_proj", the exact suffix the
        # `model.` namespace alias of the text entry collapses to
        f"model.vision_tower.vision_model.layers.{i}.{rest}"
        for i in range(2)
        for rest in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    ] + ["lm_head"],
}


def _text_layer_prefixes(tree):
    prefixes = []
    for name in tree:
        if ".layers." not in name: continue
        head, _, tail = name.partition(".layers.")
        if any(t in head for t in ("visual", "vision", "audio")): continue
        prefix = f"{head}.layers.{tail.split('.')[0]}"
        if prefix not in prefixes: prefixes.append(prefix)
    return prefixes[:2]


def _skip_lists(tree):
    """The two shapes that reach `llm_int8_skip_modules` in practice."""
    layers = _text_layer_prefixes(tree)
    leaf = sorted({n for n in tree if any(n.startswith(p + ".") for p in layers)}
                  | {n for n in tree if n.endswith("lm_head")})
    parent = sorted({f"{p}.mlp" for p in layers} | {f"{p}.self_attn" for p in layers}
                    | {"lm_head"})
    return {"leaf": leaf, "parent": parent}


@pytest.mark.parametrize("arch", sorted(MODULE_TREES))
@pytest.mark.parametrize("shape", ["leaf", "parent"])
def test_aliases_do_not_widen_the_transformers_skip_set(arch, shape):
    """No alias may change whether Transformers quantizes a real Linear."""
    should_convert_module = pytest.importorskip(
        "transformers.quantizers.quantizers_utils",
        reason = "needs a transformers exposing should_convert_module",
    ).should_convert_module
    from unsloth_zoo.saving_utils import vllm_compatible_skip_modules

    tree = MODULE_TREES[arch]
    skip = _skip_lists(tree)[shape]
    aliased = vllm_compatible_skip_modules(skip, module_names = tree)

    changed = [
        name for name in tree
        if should_convert_module(name, skip) != should_convert_module(name, aliased)
    ]
    assert not changed, (
        f"{arch}/{shape}: aliases changed the Transformers conversion decision for "
        f"{changed}. Added names were {[a for a in aliased if a not in set(skip)]}"
    )
    # The guard must drop only what it has to: every architecture here still
    # gets the fused aliases that are the point of the change.
    added = [a for a in aliased if a not in set(skip)]
    if shape == "leaf":
        assert any(a.endswith(".gate_up_proj") for a in added), (
            f"{arch}/{shape}: the guard dropped the fused aliases too, added={added}"
        )


def test_the_guard_is_what_keeps_the_adversarial_tower_safe():
    """Without the live tree the `model.` alias does reach the tower.

    This is the behaviour callers get when they cannot supply `module_names`,
    and the reason the merge writer supplies it.
    """
    should_convert_module = pytest.importorskip(
        "transformers.quantizers.quantizers_utils",
        reason = "needs a transformers exposing should_convert_module",
    ).should_convert_module
    from unsloth_zoo.saving_utils import vllm_compatible_skip_modules

    tree = MODULE_TREES["adversarial_suffix"]
    skip = _skip_lists(tree)["leaf"]

    unguarded = vllm_compatible_skip_modules(skip)
    reached = [
        name for name in tree
        if should_convert_module(name, skip) != should_convert_module(name, unguarded)
    ]
    assert reached, "the adversarial tree no longer exercises the suffix rule"
    assert all("vision" in name for name in reached)

    guarded = vllm_compatible_skip_modules(skip, module_names = tree)
    assert not [
        name for name in tree
        if should_convert_module(name, skip) != should_convert_module(name, guarded)
    ]


def test_the_widening_probe_can_actually_detect_widening():
    """Guard the guard: a deliberately over-broad entry must trip the assertion."""
    should_convert_module = pytest.importorskip(
        "transformers.quantizers.quantizers_utils",
        reason = "needs a transformers exposing should_convert_module",
    ).should_convert_module

    tree = MODULE_TREES["adversarial_suffix"]
    skip = _skip_lists(tree)["leaf"]
    over_broad = list(skip) + ["mlp.gate_proj"]  # bare leaf, matches every tower
    changed = [
        name for name in tree
        if should_convert_module(name, skip) != should_convert_module(name, over_broad)
    ]
    assert changed, "the probe cannot see widening, so its passes mean nothing"


def test_vanilla_skip_matcher_matches_the_stock_transformers_clauses():
    """The guard folds transformers' three clauses into one precompiled pass.

    Compared against the clauses written out literally rather than against the
    live `should_convert_module`, because unsloth_zoo.patching_utils replaces
    that function with a broader one at import time and the comparison would
    then depend on import order.
    """
    import random
    import re as _re

    from unsloth_zoo.saving_utils import _vanilla_skip_matcher

    def stock(name, entries):
        return any(
            _re.match(f"{k}\\.", name) or _re.match(f"{k}", name) or name.endswith(k)
            for k in entries
        )

    random.seed(7)
    parts = ["model", "language_model", "layers", "0", "1", "10", "mlp", "self_attn",
             "gate_proj", "up_proj", "qkv_proj", "lm_head", "visual", "vision_model",
             "blocks", "fc1", "proj"]

    def rnd(n):
        return ".".join(random.choice(parts) for _ in range(random.randint(1, n)))

    names = [rnd(6) for _ in range(500)]
    for _ in range(150):
        entries = [rnd(4) for _ in range(random.randint(1, 5))]
        matches = _vanilla_skip_matcher(entries)
        for name in random.sample(names, 20):
            assert matches(name) == bool(stock(name, entries)), (
                f"fast matcher diverged for entries={entries} name={name!r}"
            )


def test_guard_is_inert_when_no_module_names_are_given():
    """Callers that cannot supply the tree keep exactly the previous behaviour."""
    from unsloth_zoo.saving_utils import vllm_compatible_skip_modules

    assert (vllm_compatible_skip_modules(MERGED_LEAF_SKIP_MODULES) ==
            vllm_compatible_skip_modules(MERGED_LEAF_SKIP_MODULES, module_names = None))

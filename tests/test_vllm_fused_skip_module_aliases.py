# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Merged 4bit exports must carry skip-module names vLLM can match (unslothai/unsloth issues 1886, 464): vLLM fuses
leaves and is_layer_skipped_bnb only matches dotted ancestors, so a dense layer gets quantized and fails a shape
assert."""

import pytest

# Identity fallback fires only when the module imports but the symbol is absent; a bare except ImportError would turn a
# torch-less runner into 21 failures that look like a missing fix (macos-15), so import first and skip the file on
# failure.
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
    # Pre-fix tree: the merged 4bit save wrote the leaf list through untouched, so identity is exactly the old behaviour.
    lambda skipped_modules: skipped_modules,
)


def is_layer_skipped_bnb(prefix, llm_int8_skip_modules):
    """Verbatim copy of vllm/model_executor/layers/quantization/bitsandbytes.py::is_layer_skipped_bnb (identical in vllm
    0.11.2 and 0.29.0), so the test pins it without importing vLLM."""
    components = prefix.split(".")
    substr_check = any(m in components for m in llm_int8_skip_modules)
    set_components = set(".".join(components[: i + 1]) for i in range(len(components)))
    prefix_check = len(set(llm_int8_skip_modules) & set_components) != 0
    return substr_check or prefix_check


# A merged dynamic 4bit save of Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit currently records only leaf projections.
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
    """Only gate_proj skipped (up_proj quantized) cannot be expressed in vLLM: an alias would leave up_proj dense too, so better unmatched than corrupt."""
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


# Second mechanism (Transformers >= 4.52 multimodal nesting, verified on GPU):
# unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit writes model.language_model.layers.0.mlp, but vLLM 0.11.2 builds
# language_model.model.layers.0.mlp.down_proj and dies on a uint8 vs bfloat16 shape.

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
    """Reverting only the wiring leaves other tests green, so this guards against a silent drop; it asserts on the
    writer's source since a full merge needs a GPU model."""
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
        # The caller's own entries are passed through untouched; only the names this helper adds are its responsibility.
        added = [name for name in out if name != entry]
        for name in added:
            assert name.strip(), f"empty alias from {entry!r}"
            assert not name.endswith("."), f"bare namespace root {name!r} from {entry!r}"


# Other half of the contract: an alias can also make Transformers skip more, since matching is by prefix and suffix, so
# an alias that is a suffix of a real Linear keeps it 16-bit on reload; the trees below are real nn.Linear names
# enumerated on the meta device.

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
    # a tower deliberately repeating the text stack's own suffixes
    "adversarial_suffix": [
        f"model.language_model.layers.{i}.{rest}"
        for i in range(2)
        for rest in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    ] + [
        # ends in the exact suffix the text entry's `model.` alias collapses to
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


def _should_convert_module():
    """Uses transformers' own should_convert_module or skips where this release lacks it: transformers 4.57.6 ships
    quantizers.quantizers_utils without it, so importorskip on the module alone fails on the attribute instead of
    skipping."""
    utils = pytest.importorskip(
        "transformers.quantizers.quantizers_utils",
        reason = "needs a transformers exposing should_convert_module",
    )
    function = getattr(utils, "should_convert_module", None)
    if function is None:
        pytest.skip("this transformers has quantizers_utils but no should_convert_module")
    return function


@pytest.mark.parametrize("arch", sorted(MODULE_TREES))
@pytest.mark.parametrize("shape", ["leaf", "parent"])
def test_aliases_do_not_widen_the_transformers_skip_set(arch, shape):
    """No alias may change whether Transformers quantizes a real Linear."""
    should_convert_module = _should_convert_module()
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
    # The guard must drop only what it has to: every architecture here still gets the fused aliases that are the point of the change.
    added = [a for a in aliased if a not in set(skip)]
    if shape == "leaf":
        assert any(a.endswith(".gate_up_proj") for a in added), (
            f"{arch}/{shape}: the guard dropped the fused aliases too, added={added}"
        )


def test_the_guard_is_what_keeps_the_adversarial_tower_safe():
    """Without the live tree the `model.` alias does reach the tower, which is why the merge writer supplies it."""
    should_convert_module = _should_convert_module()
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
    should_convert_module = _should_convert_module()

    tree = MODULE_TREES["adversarial_suffix"]
    skip = _skip_lists(tree)["leaf"]
    over_broad = list(skip) + ["mlp.gate_proj"]  # bare leaf, matches every tower
    changed = [
        name for name in tree
        if should_convert_module(name, skip) != should_convert_module(name, over_broad)
    ]
    assert changed, "the probe cannot see widening, so its passes mean nothing"


def test_transformers_5x_skip_matcher_matches_the_stock_transformers_clauses():
    """Compared against the clauses written out literally, not the live should_convert_module, since patching_utils
    replaces that at import time and the result would depend on import order."""
    import random
    import re as _re

    from unsloth_zoo.saving_utils import _transformers_5x_skip_matcher

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
        matches = _transformers_5x_skip_matcher(entries)
        for name in random.sample(names, 20):
            assert matches(name) == bool(stock(name, entries)), (
                f"fast matcher diverged for entries={entries} name={name!r}"
            )


def test_guard_is_inert_when_no_module_names_are_given():
    """Callers that cannot supply the tree keep exactly the previous behaviour."""
    from unsloth_zoo.saving_utils import vllm_compatible_skip_modules

    assert (vllm_compatible_skip_modules(MERGED_LEAF_SKIP_MODULES) ==
            vllm_compatible_skip_modules(MERGED_LEAF_SKIP_MODULES, module_names = None))


# A checkpoint is written by one transformers and loaded by another; 4.x and 5.x match differently and neither is a subset of the other.

def test_transformers_4x_skip_matcher_matches_the_literal_4_57_expression():
    """Pinned to the literal skip clause in transformers 4.57.6 integrations/bitsandbytes.py:167-172: a key followed by
    a dot occurs in the name, or the key equals the name."""
    import random

    from unsloth_zoo.saving_utils import _transformers_4x_skip_matcher

    def stock_4x(full_name, entries):
        short_name = full_name.rpartition(".")[2]
        if short_name in entries: return True
        return any((key + "." in full_name) or (key == full_name) for key in entries)

    random.seed(11)
    parts = ["model", "language_model", "layers", "0", "1", "10", "mlp", "self_attn",
             "gate_proj", "up_proj", "qkv_proj", "lm_head", "visual", "vision_model",
             "blocks", "fc1", "proj"]

    def rnd(n):
        return ".".join(random.choice(parts) for _ in range(random.randint(1, n)))

    names = [rnd(6) for _ in range(300)]
    for _ in range(150):
        entries = [rnd(4) for _ in range(random.randint(1, 5))]
        matches = _transformers_4x_skip_matcher(entries)
        for name in random.sample(names, 20):
            assert matches(name) == bool(stock_4x(name, entries)), (
                f"4.x matcher diverged for entries={entries} name={name!r}"
            )


def test_guard_covers_the_4x_only_widening_vector():
    """A parent-form alias reaches the tower under transformers 4.x but not 5.x, so a guard tested only against the
    installed matcher waves it through; reachable by saving on 5.x and reloading on 4.57.6."""
    from unsloth_zoo.saving_utils import (
        _transformers_4x_skip_matcher, _transformers_5x_skip_matcher,
        vllm_compatible_skip_modules,
    )

    tree = MODULE_TREES["adversarial_suffix"]
    skip = ["model.language_model.layers.0.mlp", "lm_head"]
    unguarded = vllm_compatible_skip_modules(skip)
    added = [a for a in unguarded if a not in set(skip)]

    def newly_matched(build):
        base = build(skip)
        extra = build(added)
        return [n for n in tree if extra(n) and not base(n)]

    assert newly_matched(_transformers_5x_skip_matcher) == [], (
        "this case is supposed to be invisible to 5.x; the fixture has drifted"
    )
    reached_4x = newly_matched(_transformers_4x_skip_matcher)
    assert reached_4x, "the fixture no longer exercises the 4.x substring clause"
    assert all("vision" in n for n in reached_4x)

    guarded = vllm_compatible_skip_modules(skip, module_names = tree)
    guarded_added = [a for a in guarded if a not in set(skip)]
    for build in (_transformers_4x_skip_matcher, _transformers_5x_skip_matcher):
        base = build(skip)
        extra = build(guarded_added) if guarded_added else (lambda n: False)
        assert [n for n in tree if extra(n) and not base(n)] == []


@pytest.mark.parametrize("arch", sorted(MODULE_TREES))
@pytest.mark.parametrize("shape", ["leaf", "parent"])
def test_no_widening_under_any_transformers_generation(arch, shape):
    """The per-architecture check, run against all three matchers."""
    from unsloth_zoo.saving_utils import _SKIP_MATCHERS, vllm_compatible_skip_modules

    tree = MODULE_TREES[arch]
    skip = _skip_lists(tree)[shape]
    aliased = vllm_compatible_skip_modules(skip, module_names = tree)
    added = [a for a in aliased if a not in set(skip)]

    for build in _SKIP_MATCHERS:
        base = build(skip)
        extra = build(added) if added else (lambda n: False)
        widened = [n for n in tree if extra(n) and not base(n)]
        assert not widened, (
            f"{arch}/{shape} widened under {build.__name__}: {widened}"
        )
    if shape == "leaf":
        assert any(a.endswith(".gate_up_proj") for a in added), (
            f"{arch}/{shape}: the guard dropped the fused aliases too, added={added}"
        )

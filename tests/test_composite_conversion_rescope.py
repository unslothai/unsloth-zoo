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

"""The conversion-mapping re-scope, and the guard message that depends on it.

transformers 5.4.0 (PR #44300) merges a submodule's own prefix renaming into a composite
model's conversion mapping, where it renames every checkpoint key off the map and the
bitsandbytes quant_state sidecars are dropped. PR #45567 fixed it in 5.6.0.
`temporary_patches/conversion_mapping_rescope.py` re-scopes the leaked renaming for the
releases in between.

Measured on one B200 with `unsloth/qwen3.8-27b-unsloth-bnb-4bit`, 352 Linear4bit modules:

    transformers   repair off            repair on
    5.3.0          0/352  forward ok     0/352  (declines to install)
    5.4.0          352/352 forward dies  0/352  forward ok
    5.5.4          352/352 forward dies  0/352  forward ok
    5.6.2          0/352  forward ok     0/352  (declines to install)
    5.17.0         0/352  forward ok     0/352  (declines to install)

and the repaired arms reproduce the known-good releases' logits mean to all 16 digits
(-2.540154457092285), so the recovered state is the same state and not merely a non-null one.

CPU only, no GPU, no network, no model downloads: every model here is built on the meta
device from a config.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from unsloth_zoo.temporary_patches import conversion_mapping_rescope as rescope
from unsloth_zoo.temporary_patches import bitsandbytes as bnb_patch
from unsloth_zoo.temporary_patches.common import RESCOPE_PATCH_FLAG, WRAPPER_INNER_ATTR


def _core_model_loading():
    core = pytest.importorskip("transformers.core_model_loading")
    # On a transformers that predates the module (4.x), importing unsloth installs an inert
    # stand-in under this name so peft can import (`_build_transformers_core_model_loading_stub`
    # in unsloth/import_fixes.py). Its WeightRenaming stores its patterns and renames nothing,
    # so every assertion below would measure the stand-in rather than transformers. A real
    # module is a file on disk; the stand-in's `__file__` is `<unsloth stub: ...>`.
    if not os.path.isfile(getattr(core, "__file__", None) or ""):
        pytest.skip(
            f"transformers.core_model_loading is a stand-in ({getattr(core, '__file__', None)!r}), "
            "not a module this transformers ships"
        )
    if not hasattr(core, "WeightRenaming"):
        pytest.skip("this transformers has no WeightRenaming")
    return core


def _renaming(source, target):
    core = _core_model_loading()
    return core.WeightRenaming(source_patterns = source, target_patterns = target)


def _requires_submodule_extraction():
    """The leak detector needs `extract_weight_conversions_for_model` to exist.

    It first appears in 5.4.0. Before that there is no per-submodule recursion to correct, the
    install gate answers "leave transformers alone", and nothing here is ever called in anger --
    so these tests skip rather than assert against a function the release does not have.
    """
    conversion_mapping = pytest.importorskip("transformers.conversion_mapping")
    if not hasattr(conversion_mapping, "extract_weight_conversions_for_model"):
        pytest.skip("this transformers has no per-submodule conversion extraction")
    return conversion_mapping


def _meta_model(model_type, factory_name, **config_kwargs):
    """A real model of `model_type`, on the meta device, with the layers shrunk to nothing."""
    from transformers import CONFIG_MAPPING
    if model_type not in CONFIG_MAPPING:
        pytest.skip(f"this transformers has no {model_type}")
    factory = getattr(transformers, factory_name, None)
    if factory is None:
        pytest.skip(f"this transformers has no {factory_name}")
    config = CONFIG_MAPPING[model_type]()
    for key, value in config_kwargs.items():
        target = config
        *path, leaf = key.split(".")
        for step in path:
            target = getattr(target, step, None)
            if target is None:
                break
        if target is not None:
            setattr(target, leaf, value)
    try:
        with torch.device("meta"):
            return factory.from_config(config)
    except Exception as e:
        pytest.skip(f"cannot build a meta {model_type}: {e}")


@pytest.fixture
def composite_model():
    model = _meta_model(
        "qwen3_5", "AutoModelForImageTextToText",
        **{"text_config.num_hidden_layers": 2, "vision_config.depth": 1},
    )
    names = [name for name, _ in model.named_parameters()]
    if not any(name.startswith("model.language_model.") for name in names):
        pytest.skip("this build does not nest the text model under model.language_model")
    return model


# ---------------------------------------------------------------- pattern helpers


def test_prefixed_pattern_keeps_an_anchor_anchored():
    assert rescope._prefixed_pattern("^model.", "model.language_model") == \
        "^model.language_model.model."
    # and does not invent one
    assert rescope._prefixed_pattern("gate.", "model.language_model") == \
        "model.language_model.gate."


def test_renaming_signature_compares_by_value_not_identity():
    one = _renaming("^model.language_model.", "^model.")
    two = _renaming("^model.language_model.", "^model.")
    assert one is not two
    assert rescope._renaming_signature(one) == rescope._renaming_signature(two)
    other = _renaming("^model.vision_tower.", "^model.")
    assert rescope._renaming_signature(one) != rescope._renaming_signature(other)


def test_patterns_survive_a_release_without_the_original_fields():
    """transformers 5.4.0 has no `_original_source_patterns`; 5.5.x does. Both must work."""
    renaming = _renaming("^model.language_model.", "^model.")
    source, target = rescope._patterns(renaming)
    assert source and target
    assert isinstance(source, list) and isinstance(target, list)


# ---------------------------------------------------------------- the discriminator


def test_a_renaming_that_lands_off_the_map_is_destructive():
    renaming = _renaming("^model.language_model.", "^model.")
    sample = ["model.language_model.layers.0.self_attn.q_proj.weight"]
    model_keys = set(sample)
    assert rescope._renaming_destroys_keys(renaming, sample, model_keys) is True


def test_a_renaming_that_matches_nothing_is_not_destructive():
    renaming = _renaming("^nothing.at.all.", "^model.")
    sample = ["model.language_model.layers.0.self_attn.q_proj.weight"]
    assert rescope._renaming_destroys_keys(renaming, sample, set(sample)) is False


def test_one_rename_onto_a_real_key_outvotes_the_rest():
    """A mapping doing its job is never destructive, however much other evidence there is."""
    renaming = _renaming("^model.language_model.", "^model.")
    sample = [
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.embed_tokens.weight",
    ]
    # The model really has the SHORT name, i.e. this renaming is doing what it was written for.
    model_keys = set(sample) | {"model.layers.0.self_attn.q_proj.weight"}
    assert rescope._renaming_destroys_keys(renaming, sample, model_keys) is False


# ---------------------------------------------------------------- re-scoping


def test_rescoped_renaming_reproduces_the_doubled_prefix():
    # Spelled the way a real prefix strip is: the SOURCE is anchored, the target is not.
    renaming = _renaming("^model.language_model.", "model.")
    sample = ["model.language_model.layers.0.self_attn.q_proj.weight"]
    model_keys = set(sample)
    scoped = rescope._rescoped_renaming(renaming, "model.language_model", sample, model_keys)
    assert scoped is not None
    source, target = rescope._patterns(scoped)
    assert source[0] == "^model.language_model.model.language_model."
    assert target[0] == "model.language_model.model."
    # and it is inert on the real keys, which is the whole point
    for key in sample:
        renamed, _ = scoped.rename_source_key(key)
        assert renamed == key


def test_rescoped_renaming_refuses_an_unanchored_pattern():
    """Pushing a prefix in front of "wherever this appears" means neither thing."""
    renaming = _renaming(r"\.gate\.", ".router.")
    sample = ["model.language_model.layers.0.gate.weight"]
    assert rescope._rescoped_renaming(renaming, "model.language_model", sample, set(sample)) is None


# ---------------------------------------------------------------- install gate


def test_on_a_fixed_transformers_the_rescope_changes_nothing(composite_model):
    """The stronger claim, and the one that survives the install gate being wrong.

    On a release that already scopes submodule mappings, the leaked SIGNATURE is still present
    in the mapping -- the renaming really was collected from the text model -- but transformers
    has set `scope_prefix` on it, so the entry is skipped and the caller's own list comes back.
    Measured by identity, so this cannot pass by rebuilding an equal list.
    """
    conversion_mapping = _requires_submodule_extraction()
    if not rescope._transformers_rescopes_submodule_prefix_renamings():
        pytest.skip("this transformers is inside the defect window; see the real-checkpoint matrix")
    unpatched = getattr(
        conversion_mapping.get_model_conversion_mapping, "__wrapped__",
        conversion_mapping.get_model_conversion_mapping,
    )
    conversions = unpatched(composite_model)
    assert conversions, "expected this model to have a conversion mapping at all"
    assert rescope._rescope_conversions(composite_model, conversions) is conversions


def test_the_leaked_entry_is_the_language_model_prefix_strip(composite_model):
    """Name what the detector finds, so a silent change of target shows up as a failure.

    Only asserted where a leak exists at all. The releases differ in where they scope: 5.4/5.5
    hand the entry over unscoped, which is the defect; 5.6 to 5.9 rewrite its patterns at
    extraction time, so nothing reaches here to leak; 5.10+ set `scope_prefix` on the
    extracted object, which still looks destructive on its own and is skipped downstream
    instead (see test_on_a_fixed_transformers_the_rescope_changes_nothing).
    """
    _requires_submodule_extraction()
    leaked, _ = rescope._leaked_submodule_prefix_renamings(composite_model)
    if not leaked:
        pytest.skip("this transformers scopes submodule mappings at extraction time")
    assert len(leaked) == 1
    (_name, source, target), (prefix, _keys) = next(iter(leaked.items()))
    assert prefix == "model.language_model"
    assert any("language_model" in pattern for pattern in source)
    assert target


def test_a_non_composite_model_gets_its_own_list_back():
    """Measurably untouched, not merely equal."""
    _requires_submodule_extraction()
    model = _meta_model("llama", "AutoModelForCausalLM", **{"num_hidden_layers": 1})
    sentinel = [_renaming("^nothing.", "^also_nothing.")]
    assert rescope._rescope_conversions(model, sentinel) is sentinel


def _base_mapping_fn(conversion_mapping):
    """The genuinely unwrapped `get_model_conversion_mapping`.

    conftest applies TEMPORARY_PATCHES, so by the time a test runs the module attribute is
    already this package's wrappers. Walking to the base is what lets a test install onto a
    clean function instead of being refused by the already-installed check.
    """
    function = conversion_mapping.get_model_conversion_mapping
    for _ in range(8):
        following = (
            getattr(function, "__wrapped__", None)
            or getattr(function, "_unsloth_wrapper_inner", None)
        )
        if following is None:
            break
        function = following
    return function


def test_install_is_idempotent_and_undoable():
    conversion_mapping = pytest.importorskip("transformers.conversion_mapping")
    live = conversion_mapping.get_model_conversion_mapping
    before = _base_mapping_fn(conversion_mapping)
    try:
        conversion_mapping.get_model_conversion_mapping = before
        rescope.patch_transformers_composite_prefix_renaming()
        after = conversion_mapping.get_model_conversion_mapping
        if rescope._transformers_rescopes_submodule_prefix_renamings():
            # This build carries the upstream fix, so the patch must have declined.
            assert after is before
            return
        assert getattr(after, RESCOPE_PATCH_FLAG, False) is True
        rescope.patch_transformers_composite_prefix_renaming()
        assert conversion_mapping.get_model_conversion_mapping is after
        assert after.__wrapped__ is before
    finally:
        conversion_mapping.get_model_conversion_mapping = live


def test_repair_detection_sees_both_packages_marks():
    def bare():
        pass
    assert rescope._repair_already_installed(bare) is False

    def zoo_marked():
        pass
    setattr(zoo_marked, RESCOPE_PATCH_FLAG, True)
    assert rescope._repair_already_installed(zoo_marked) is True

    def unsloth_marked():
        pass
    setattr(unsloth_marked, "_unsloth_patched_composite_prefix_renaming", True)
    assert rescope._repair_already_installed(unsloth_marked) is True

    # Under a wrapper that publishes __wrapped__.
    def on_top():
        pass
    on_top.__wrapped__ = unsloth_marked
    assert rescope._repair_already_installed(on_top) is True

    # And under one that publishes only WRAPPER_INNER_ATTR, which is what
    # moe_utils_bnb4bit.py installs. It cannot use __wrapped__: the installer unwraps that
    # to choose what to wrap, so publishing it would have the rescope REPLACE the MoE
    # wrapper instead of sitting on top of it, dropping the per-expert converters.
    def moe_style():
        pass
    setattr(moe_style, WRAPPER_INNER_ATTR, zoo_marked)
    assert rescope._repair_already_installed(moe_style) is True


def test_the_moe_wrapper_does_not_hide_the_repair_from_either_probe():
    """Regression: the MoE wrapper used to be an opaque lid over the repair.

    `TEMPORARY_PATCHES` runs three times (init, pre_compile, post_compile) and the rescope
    installs before `moe_utils_bnb4bit`, so from pass 1 onward the MoE wrapper was outermost
    with no link down. Measured on transformers 5.5.4 before the fix: pass 1 left
    `_composite_renaming_repair_installed()` False, so the Linear4bit error blamed the
    transformers version for a load the repair had already fixed, and pass 2 stacked a
    second rescope wrapper.
    """
    # importorskip, not a bare import: transformers 4.x has no conversion_mapping at all, so
    # on the 4.57.6 CI lane this was a collection-time ModuleNotFoundError rather than a skip.
    cm = pytest.importorskip("transformers.conversion_mapping")
    moe = pytest.importorskip("unsloth_zoo.temporary_patches.moe_utils_bnb4bit")
    patch_bnb4bit_model_conversion_mapping = moe.patch_bnb4bit_model_conversion_mapping

    pristine = cm.get_model_conversion_mapping
    try:
        def repaired(*args, **kwargs):
            return pristine(*args, **kwargs)
        setattr(repaired, RESCOPE_PATCH_FLAG, True)
        cm.get_model_conversion_mapping = repaired

        patch_bnb4bit_model_conversion_mapping()
        installed = cm.get_model_conversion_mapping
        assert installed is not repaired, "the MoE patch did not install"
        assert getattr(installed, WRAPPER_INNER_ATTR, None) is repaired
        assert rescope._repair_already_installed(installed) is True
        assert bnb_patch._composite_renaming_repair_installed() is True
    finally:
        cm.get_model_conversion_mapping = pristine


def test_repair_detection_cannot_spin_on_a_cycle():
    def a():
        pass
    def b():
        pass
    a.__wrapped__ = b
    b.__wrapped__ = a
    assert rescope._repair_already_installed(a) is False


# ---------------------------------------------------------------- the guard message


@pytest.mark.parametrize(
    "version,expected",
    [
        ("5.3.0", False),
        ("5.4.0", True),
        ("5.5.0", True),
        ("5.5.4", True),
        ("5.6.0", False),
        ("5.17.0", False),
        ("4.57.6", False),
        # A prerelease sorts with the release line it belongs to. Without the `.dev0` bounds
        # `Version("5.6.0.dev0") < Version("5.6.0")` accuses a build that carries the fix.
        ("5.6.0.dev0", False),
        ("5.6.0rc1", False),
        ("5.4.0.dev0", True),
    ],
)
def test_defect_window_bounds_including_prereleases(monkeypatch, version, expected):
    # The raw string, as `importlib.metadata.version` returns it. Deliberately NOT routed
    # through `unsloth_zoo.utils.Version`, which answers `Version('5.6.0.1')` for the string
    # "5.6.0.dev0" and would sort a prerelease after the release it belongs before.
    monkeypatch.setattr(bnb_patch, "_installed_transformers_version", lambda: version)
    assert bnb_patch._transformers_drops_prequantized_quant_state() is expected


class _FakeLinear4bit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.zeros((8192, 1), dtype = torch.uint8), requires_grad = False,
        )
        self.quant_state = None


def test_message_blames_the_version_only_when_the_repair_is_not_installed(monkeypatch):
    monkeypatch.setattr(bnb_patch, "_transformers_drops_prequantized_quant_state", lambda: True)
    monkeypatch.setattr(bnb_patch, "_composite_renaming_repair_installed", lambda: False)
    message = bnb_patch._packed_weight_without_quant_state_error(_FakeLinear4bit())
    assert "#45567" in message
    assert "lost while LOADING" in message
    assert "before regenerating anything" in message


def test_message_stops_blaming_the_version_once_the_repair_is_installed(monkeypatch):
    monkeypatch.setattr(bnb_patch, "_transformers_drops_prequantized_quant_state", lambda: True)
    monkeypatch.setattr(bnb_patch, "_composite_renaming_repair_installed", lambda: True)
    message = bnb_patch._packed_weight_without_quant_state_error(_FakeLinear4bit())
    assert "not the explanation here" in message
    assert "#45567" not in message
    # and specifically not the advice to downgrade or re-quantize
    assert "regenerating anything" not in message


def test_message_stays_generic_outside_the_window(monkeypatch):
    monkeypatch.setattr(bnb_patch, "_transformers_drops_prequantized_quant_state", lambda: False)
    message = bnb_patch._packed_weight_without_quant_state_error(_FakeLinear4bit())
    assert "absmax" in message
    assert "#45567" not in message


def test_repair_probe_never_raises_without_transformers(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _boom(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _boom)
    assert bnb_patch._composite_renaming_repair_installed() is False


def test_the_sweep_leaves_a_wrapper_it_does_not_supersede(monkeypatch):
    """Someone else's wrapper on an imported alias must survive the rebinding sweep.

    The sweep exists because a module that did `from .conversion_mapping import
    get_model_conversion_mapping` holds the OBJECT, so rebinding the module attribute alone
    leaves it on the unrepaired function. But a binding under that name is not necessarily the
    function we wrapped: another integration may have wrapped the alias and left the module
    attribute alone, and our wrapper closes over the module attribute, so overwriting the alias
    drops their behaviour with no trace -- silently disabling unrelated conversion logic.

    Reproduced before the fix: `modeling_utils.get_model_conversion_mapping` came back as ours
    and `third_party` was nowhere in the chain.
    """
    conversion_mapping = _requires_submodule_extraction()
    if rescope._transformers_rescopes_submodule_prefix_renamings():
        pytest.skip("this transformers carries the upstream fix; the re-scope declines")
    modeling_utils = pytest.importorskip("transformers.modeling_utils")

    pristine = _base_mapping_fn(conversion_mapping)

    def third_party(*args, **kwargs):
        return pristine(*args, **kwargs)

    monkeypatch.setattr(conversion_mapping, "get_model_conversion_mapping", pristine)
    monkeypatch.setattr(modeling_utils, "get_model_conversion_mapping", third_party)

    rescope.patch_transformers_composite_prefix_renaming()

    assert conversion_mapping.get_model_conversion_mapping is not pristine, "expected an install"
    assert modeling_utils.get_model_conversion_mapping is third_party


def test_the_sweep_still_rebinds_a_plain_alias(monkeypatch):
    """The other half: an alias holding exactly what we wrapped must be moved onto ours."""
    conversion_mapping = _requires_submodule_extraction()
    if rescope._transformers_rescopes_submodule_prefix_renamings():
        pytest.skip("this transformers carries the upstream fix; the re-scope declines")
    modeling_utils = pytest.importorskip("transformers.modeling_utils")

    pristine = _base_mapping_fn(conversion_mapping)
    monkeypatch.setattr(conversion_mapping, "get_model_conversion_mapping", pristine)
    monkeypatch.setattr(modeling_utils, "get_model_conversion_mapping", pristine)

    rescope.patch_transformers_composite_prefix_renaming()

    live = conversion_mapping.get_model_conversion_mapping
    assert live is not pristine
    assert modeling_utils.get_model_conversion_mapping is live


def test_a_third_party_wrapper_on_the_module_attribute_survives(monkeypatch):
    """The canonical-binding half of the same hazard.

    Installing used to unwrap `__wrapped__` before wrapping, to avoid re-wrapping a wrapper of
    ours that had lost its mark. But the already-installed check declines whenever either mark
    is anywhere in the chain, so that case cannot reach the unwrap, while a third party's
    `functools.wraps` wrapper on the module attribute WAS unwrapped past and then replaced,
    dropping their behaviour with nothing left in the chain.
    """
    conversion_mapping = _requires_submodule_extraction()
    if rescope._transformers_rescopes_submodule_prefix_renamings():
        pytest.skip("this transformers carries the upstream fix; the re-scope declines")
    import functools

    pristine = _base_mapping_fn(conversion_mapping)

    @functools.wraps(pristine)
    def third_party(*args, **kwargs):
        return pristine(*args, **kwargs)

    monkeypatch.setattr(conversion_mapping, "get_model_conversion_mapping", third_party)
    rescope.patch_transformers_composite_prefix_renaming()

    live = conversion_mapping.get_model_conversion_mapping
    assert live is not third_party, "expected an install"
    chain = rescope._wrapper_chain(live)
    assert any(f is third_party for f in chain), \
        "the third party's wrapper must still run, not be unwrapped past and discarded"

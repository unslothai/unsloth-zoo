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

"""A standalone function that builds attention masks must never be compiled fullgraph.

`transformers.masking_utils` branches on tensor VALUES -- `flex_attention_mask` does
`if attention_mask is not None and not fast_all(attention_mask)`, and `fast_all` returns
a 0-dim tensor -- so a caller captured whole dies with `Unsupported: Data-dependent
branching`.

`DISABLED_KEYWORDS` used to hold the literal `create_causal_mask(**mask_kwargs)` for
this. transformers then added `block_sequence_ids=` to that call in Gemma3, the `)`
moved, the literal stopped matching, and Gemma3 vision inference crashed on the first
generate while gemma4, whose spelling was untouched, kept working. These tests pin the
call-shaped detector that replaced it, so the same drift cannot land twice.
"""

import inspect

import pytest
import torch

from unsloth_zoo import compiler as compiler_module
from unsloth_zoo.compiler import (
    DISABLED_KEYWORDS,
    calls_mask_creation_function,
    get_mask_functions,
)

# Every model whose modeling file defines a module-level mask builder. gemma4 /
# gemma4_unified do not exist on older transformers, so each is probed, not required.
_VISION_MASK_MODELS = ("gemma3", "gemma4", "gemma4_unified")


def _vision_mask_builder(model):
    """`create_masks_for_vision_model` from an installed modeling file, or None."""
    try:
        module = __import__(
            f"transformers.models.{model}.modeling_{model}", fromlist=["_"]
        )
    except Exception:
        return None
    return getattr(module, "create_masks_for_vision_model", None)


def _installed_vision_mask_builders():
    found = {}
    for model in _VISION_MASK_MODELS:
        function = _vision_mask_builder(model)
        if function is not None:
            found[model] = function
    return found


def test_mask_factories_are_discoverable():
    """Everything below is vacuous if the installed transformers exports no factory."""
    factories = get_mask_functions()

    assert "create_causal_mask" in factories, (
        "transformers.masking_utils no longer exports create_causal_mask, so "
        "calls_mask_creation_function cannot recognise a mask builder and every "
        f"builder goes back to fullgraph = True. Found: {sorted(factories)}"
    )


@pytest.mark.parametrize("model", _VISION_MASK_MODELS)
def test_vision_mask_builders_are_not_compiled(model):
    """The regression guard. Fails on the commit that shipped the drifted literal."""
    function = _vision_mask_builder(model)
    if function is None:
        pytest.skip(f"transformers has no {model}.create_masks_for_vision_model")

    source = inspect.getsource(function)
    matched = calls_mask_creation_function(source)

    assert len(matched) != 0, (
        f"transformers.models.{model}.modeling_{model}.create_masks_for_vision_model "
        "calls a transformers.masking_utils factory, but the compiler does not see "
        "it, so the rewriter stamps @torch_compile_with_fallback(fullgraph = True, "
        "...) on it and the first vision generate dies with `Unsupported: "
        "Data-dependent branching` inside flex_attention_mask. The detector has "
        "drifted away from how upstream now spells the call."
    )


def test_a_defined_vision_mask_builder_is_always_resolved():
    """Skipping everything is only legitimate when no modeling file defines one.

    transformers 4.57.6 has no create_masks_for_vision_model at all, so the cases
    above skip and the rule is correctly a no-op. A release that DEFINES one but
    exposes it under another name would also skip, silently guarding nothing, so
    the modeling source is asked directly rather than trusting the skips."""
    defines = []
    for model in _VISION_MASK_MODELS:
        try:
            module = __import__(
                f"transformers.models.{model}.modeling_{model}", fromlist=["_"]
            )
            source = inspect.getsource(module)
        except Exception:
            continue
        if "def create_masks_for_vision_model" in source:
            defines.append(model)

    if len(defines) == 0:
        pytest.skip("this transformers defines no create_masks_for_vision_model")

    found = _installed_vision_mask_builders()

    assert sorted(found) == sorted(defines), (
        f"{sorted(defines)} define create_masks_for_vision_model but only "
        f"{sorted(found)} could be resolved by name, so the cases above skipped a "
        "builder that really is compiled. The lookup has drifted from upstream."
    )


def test_a_mask_builder_really_is_uncapturable_fullgraph():
    """Why the rule exists, rather than only that it fires.

    Builds the real Gemma3 flex-attention mask eagerly first -- so a harness that
    drifted out of shape fails loudly instead of passing through the `except` -- then
    asserts that compiling the same call fullgraph either raises, in which case the
    compiler must be excluding it, or succeeds, which is allowed and means upstream
    became traceable."""
    torch = pytest.importorskip("torch")
    gemma3 = pytest.importorskip("transformers.models.gemma3.modeling_gemma3")
    builder = getattr(gemma3, "create_masks_for_vision_model", None)
    if builder is None:
        pytest.skip("transformers has no gemma3.create_masks_for_vision_model")

    from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

    config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        sliding_window=8,
    )
    config._attn_implementation = "flex_attention"

    length = 16
    inputs_embeds = torch.zeros(1, length, config.hidden_size)
    # A padded row is the point: `fast_all` is False, so `flex_attention_mask` takes
    # the branch that cannot be captured.
    attention_mask = torch.ones(1, length, dtype=torch.long)
    attention_mask[:, -4:] = 0
    position_ids = torch.arange(length).unsqueeze(0)
    token_type_ids = torch.zeros(1, length, dtype=torch.long)
    token_type_ids[:, 4:8] = 1
    block_sequence_ids = gemma3.get_block_sequence_ids_for_mask(token_type_ids)

    kwargs = dict(
        config=config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=position_ids,
        block_sequence_ids=block_sequence_ids,
    )

    eager = builder(**kwargs)
    assert set(eager) == {"full_attention", "sliding_attention"}

    torch._dynamo.reset()
    try:
        torch.compile(builder, fullgraph=True, dynamic=True)(**kwargs)
    except Exception as exception:
        assert len(calls_mask_creation_function(inspect.getsource(builder))) != 0, (
            "create_masks_for_vision_model cannot be traced with fullgraph = True "
            f"({type(exception).__name__}: "
            f"{str(exception).strip().splitlines()[0][:200]}), and the compiler does "
            "not exclude it, so it is compiled and Gemma3 vision generate crashes."
        )
    finally:
        torch._dynamo.reset()


def test_the_call_is_matched_whatever_its_arguments():
    """The exact drift that caused the bug: a kwarg added inside the parentheses."""
    drifted = "    full_mask = create_causal_mask(**mask_kwargs, block_sequence_ids=ids)\n"
    original = "    full_mask = create_causal_mask(**mask_kwargs)\n"
    split = "    mask = create_causal_mask(\n        **mask_kwargs,\n    )\n"
    spaced = "    mask = create_causal_mask (**mask_kwargs)\n"

    for source in (drifted, original, split, spaced):
        assert calls_mask_creation_function(source) == ["create_causal_mask"], source

    assert original not in "".join(DISABLED_KEYWORDS), (
        "the spelling-sensitive literal is back in DISABLED_KEYWORDS; it silently "
        "stops matching the moment upstream adds a keyword argument"
    )


def test_unrelated_sources_are_not_matched():
    """Over-matching would stop compiling functions that are fine today."""
    assert calls_mask_creation_function("    m = self.create_causal_mask(x)\n") == []
    assert calls_mask_creation_function("    m = utils.create_causal_mask(x)\n") == []
    assert calls_mask_creation_function("    m = my_create_causal_mask(x)\n") == []
    assert calls_mask_creation_function('    """See create_causal_mask."""\n') == []
    assert calls_mask_creation_function("    x = create_causal_mask\n") == []


@pytest.mark.parametrize(
    "name", ["rotate_half", "apply_rotary_pos_emb", "eager_attention_forward"]
)
def test_ordinary_lifted_functions_still_compile(name):
    """Negative control: the rule must not sweep up the hot helpers."""
    gemma3 = pytest.importorskip("transformers.models.gemma3.modeling_gemma3")
    function = getattr(gemma3, name, None)
    if function is None:
        pytest.skip(f"transformers has no gemma3.{name}")

    assert calls_mask_creation_function(inspect.getsource(function)) == []


def test_disable_compile_functions_outranks_the_mask_rule():
    """`@torch.compiler.disable` is a stronger guarantee than no decorator.

    No decorator only stops us compiling the function; the disable decorator also
    stops Dynamo inlining it into a compiled caller. A name on
    DISABLE_COMPILE_FUNCTIONS is an explicit instruction to emit that decorator, so
    a mask call inside it must not silently downgrade to 'emit bare'. Nothing on
    the list builds masks today, which is exactly why this needs pinning."""
    source = inspect.getsource(compiler_module)

    # Loop B: the mask rule is skipped for a listed name, so the branch below that
    # emits @torch.compiler.disable is still reached.
    assert "if not bad and module not in disable_compile_functions:" in source, (
        "the copy loop applies the mask rule to names in DISABLE_COMPILE_FUNCTIONS, "
        "so such a function is emitted bare instead of with "
        "@torch.compiler.disable(recursive = False) and can be inlined into a "
        "compiled caller"
    )

    # Loop A: membership is tested before the mask branch.
    fixup = source.index("_mask_builders = calls_mask_creation_function(")
    window = source[fixup:fixup + 400]
    assert window.index("if module in disable_compile_functions:") < window.index(
        "elif len(_mask_builders) != 0:"
    ), "the signature-fixup loop tests the mask rule before DISABLE_COMPILE_FUNCTIONS"


def test_both_standalone_emit_sites_consult_the_detector():
    """Miss one and a mask builder is stamped fullgraph again from the other path."""
    source = inspect.getsource(compiler_module)

    assert source.count("calls_mask_creation_function(") == 3, (
        "expected one definition plus the two standalone-function emit sites in "
        "unsloth_compile_transformers (the signature-fixup loop and the copy loop); "
        "a site that stopped consulting the detector will stamp "
        "@torch_compile_with_fallback(fullgraph = True, ...) on a mask builder again"
    )

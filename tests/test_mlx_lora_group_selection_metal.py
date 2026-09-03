# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Which linears each MLX `get_peft_model` group flag adapts."""

import warnings

import pytest

pytest.importorskip("mlx.core")


@pytest.fixture(autouse=True)
def _require_real_metal():
    import mlx.core as _mx   # re-import: the shim may have swapped it
    if not (getattr(_mx, "metal", None) and _mx.metal.is_available()
            and _mx.default_device() == _mx.gpu):
        pytest.skip("real Metal required; shim active or no GPU")


HIDDEN, VOCAB, VISION = 64, 128, 32


def _build(spec):
    # Module trees as data: a dict is a module, a tuple is `Linear(in, out)`.
    import mlx.nn as nn
    if isinstance(spec, nn.Module):
        return spec
    if callable(spec):
        return spec()
    if isinstance(spec, tuple):
        return nn.Linear(*spec)
    if isinstance(spec, list):
        return [_build(item) for item in spec]
    module = type("Module", (nn.Module,), {})()
    for name, child in spec.items():
        setattr(module, name, _build(child))
    return module


_KIMI_BLOCK = {"wqkv": (VISION, VISION * 3), "wo": (VISION, VISION),
               "mlp": {"fc0": (VISION, VISION), "fc1": (VISION, VISION)}}
_GLM_MERGER = {"proj": (VISION, VISION), "gate_proj": (VISION, HIDDEN),
               "down_proj": (HIDDEN, HIDDEN)}
_MIXER = {"token_mixer": {"qkv": (VISION, VISION * 3), "proj": (VISION, VISION)}}
_TEXT_BLOCK = {"self_attn": {"q_proj": (HIDDEN, HIDDEN), "o_proj": (HIDDEN, HIDDEN)},
               "mlp": {"gate_proj": (HIDDEN, HIDDEN), "down_proj": (HIDDEN, HIDDEN)}}
_MOLMO_BLOCK = {"att_proj": (HIDDEN, HIDDEN), "attn_out": (HIDDEN, HIDDEN),
                "ff_proj": (HIDDEN, HIDDEN), "ff_out": (HIDDEN, HIDDEN)}
def _tower(block=_KIMI_BLOCK, merger=None):
    spec = {"blocks": [block], "patch_embed": {}}
    if merger is not None:
        spec["merger"] = merger
    return _build(spec)


def _text_model(blocks=None):
    import mlx.nn as nn
    model = _build({"model": {"embed_tokens": lambda: nn.Embedding(VOCAB, HIDDEN),
                              "layers": blocks or [_TEXT_BLOCK] * 2},
                    "lm_head": (HIDDEN, VOCAB)})
    model.config = type("C", (), {})()
    return model


def _vlm(tower_attr="vision_tower", tower=None, extra=None, decoder=None):
    config, text = type("C", (), {})(), type("C", (), {})()
    text.hidden_size, config.hidden_size, config.text_config = HIDDEN, HIDDEN, text
    model, model.config = _build({}), config
    setattr(model, tower_attr, _tower() if tower is None else tower)
    for name, child in [extra] if extra else []:
        setattr(model, name, child)
    model.language_model = _text_model(decoder)
    model._is_vlm_model = True
    return model


def _peft(model, **kwargs):
    from unsloth_zoo.mlx.loader import FastMLXModel
    return FastMLXModel.get_peft_model(
        model, **{"r": 4, "use_gradient_checkpointing": False, **kwargs})


def _adapters(model, prefix=""):
    cut = len(prefix) + 1 if prefix else 0
    return sorted(name[cut:] for name, module in model.named_modules()
                  if hasattr(module, "lora_a") and name.startswith(prefix))


# What each path reads as. The enclosing block outranks the leaf below it, a bare
# projection names no role, and a `gate` naming what it projects into is an MLP.
_ROLE_CASES = {
    "query_key_value": "attention", "dense": None, "mlp.gate": "gate",
    "attn_pool.mlp.fc1": "mlp", "mlp.dense_h_to_4h": "mlp", "mlp.gate_1": "gate",
    "self_attn.gate_proj": "attention", "q_proj.linear": "attention",
    "linear_attn.in_proj_qkv": "attention", "per_layer_input_gate": "gate",
    "block_sparse_moe.switch_mlp.gate_up_proj": "mlp",
    "mlp.zaya_block.router.down_proj": "gate",
}


@pytest.mark.parametrize("path,role", _ROLE_CASES.items(), ids=list(_ROLE_CASES))
def test_what_a_linear_name_reads_as(path, role):
    from unsloth_zoo.mlx.loader import _linear_role
    assert _linear_role(path) == role


_LOAN_CASES = {
    "mlp, from three agreeing siblings": (
        _GLM_MERGER, True, True, ["down_proj", "gate_proj", "proj"]),
    "and not when the mlp flag is off": (_GLM_MERGER, True, False, []),
    "attention, from the one linear beside it": (
        _MIXER, True, True, ["token_mixer.proj", "token_mixer.qkv"]),
    "and not when the attention flag is off": (_MIXER, False, True, []),
    # Exclusion, not borrowing: a gate decides, and one unit is a scale.
    "and a gate or a one-unit scale is selected by neither flag": (
        {"mlp": {"up": (HIDDEN, HIDDEN), "gate": (HIDDEN, 4),
                 "shared_expert_gate": (HIDDEN, 1)},
         "mlp_res_proj": (HIDDEN, 1)}, True, True, ["mlp.up"]),
}


@pytest.mark.parametrize("spec,attention,mlp,expected", _LOAN_CASES.values(),
                         ids=list(_LOAN_CASES))
def test_role_selection_borrows_beside_and_excludes_what_decides(
        spec, attention, mlp, expected):
    from unsloth_zoo.mlx.loader import _role_selected_paths
    assert sorted(_role_selected_paths(_build(spec), attention, mlp)) == expected


_SELECTION_CASES = {
    "a tower under an unlisted name, by role and by flag": (
        lambda: _vlm(tower_attr="visual"),
        {"finetune_vision_layers": True, "finetune_mlp_modules": False},
        "visual.blocks.0", ["wo", "wqkv"]),
    # A nested connector reads as MLP, so the tower pass must leave it alone.
    "a nested connector is adapted once, by its own pass": (
        lambda: _vlm(tower=_tower(merger=_GLM_MERGER)),
        {"finetune_vision_layers": True, "train_projector": True},
        "vision_tower.merger", ["down_proj", "gate_proj", "proj"]),
    # Qwen3.5 alternates attention kinds, so reading one layer misses half.
    "a decoder spelled unlike a canonical one, layer by layer": (
        lambda: _text_model([_MOLMO_BLOCK, _TEXT_BLOCK]), {}, "model.layers",
        ["0.att_proj", "0.attn_out", "0.ff_out", "0.ff_proj", "1.mlp.down_proj",
         "1.mlp.gate_proj", "1.self_attn.o_proj", "1.self_attn.q_proj"]),
    # Each call site must pass the flags on, and the decoder has two of them.
    "a tower with attention off": (
        lambda: _vlm(tower_attr="visual"), {"finetune_vision_layers": True,
        "finetune_attention_modules": False}, "visual.blocks.0", ["mlp.fc0", "mlp.fc1"]),
    "a decoder with attention off": (
        lambda: _text_model([_MOLMO_BLOCK]), {"finetune_attention_modules": False},
        "model.layers.0", ["ff_out", "ff_proj"]),
    "a vlm decoder with mlp off": (
        lambda: _vlm(decoder=[_MOLMO_BLOCK]), {"finetune_vision_layers": False,
        "finetune_mlp_modules": False}, "language_model.model.layers.0",
        ["att_proj", "attn_out"]),
}


@pytest.mark.parametrize("build,kwargs,prefix,expected", _SELECTION_CASES.values(),
                         ids=list(_SELECTION_CASES))
def test_a_group_flag_adapts_exactly_its_own_linears(build, kwargs, prefix,
                                                     expected):
    model = build()
    with warnings.catch_warnings():   # role selection announces nothing
        warnings.simplefilter("error", UserWarning)
        _peft(model, **kwargs)
    assert _adapters(model, prefix) == expected


# Selecting nothing must raise, naming the flag and enough of the tree to act on.
_EMPTY_GROUP_CASES = {
    # "sam" occurs inside `itok_upsampler`, so tokens are matched whole.
    "unresolved tower": (
        lambda: _vlm(tower_attr="itok_upsampler"), {"train_vision": True},
        ["train_vision=True", "'itok_upsampler'"]),
    # Refuses after the tower walk matched, so an adapting pass would leave
    # those adapters behind.
    "a connector with no linear, beside an adaptable tower": (
        lambda: _vlm(extra=("mm_projector", _build({}))),
        {"finetune_vision_layers": True, "train_projector": True},
        ["holds no linear layer to adapt"]),
    # An explicit vocabulary is the caller's, so a tower it misses is theirs.
    "tower matches no requested target": (
        lambda: _vlm(tower=_tower(merger=_GLM_MERGER)),
        {"finetune_vision_layers": True, "train_projector": True,
         "target_modules": ["q_proj"]},
        ["finetune_vision_layers", "'vision_tower'", "q_proj", "'wqkv'"]),
    "a tower whose linears read as neither role": (
        lambda: _vlm(tower=_build({"patch_ln1": {}, "patch_dense": (VISION, HIDDEN)})),
        {"finetune_vision_layers": True},
        ["finetune_vision_layers", "'patch_dense'"]),
}


@pytest.mark.parametrize("build,kwargs,expected", _EMPTY_GROUP_CASES.values(),
                         ids=list(_EMPTY_GROUP_CASES))
def test_a_group_flag_selecting_nothing_raises_and_changes_nothing(
        build, kwargs, expected):
    model = build()
    model._unsloth_cpt_full_module_weight_keys = "untouched"
    with pytest.raises(ValueError) as excinfo:
        _peft(model, **kwargs)
    assert all(word in str(excinfo.value) for word in expected)
    assert _adapters(model) == []
    assert model._unsloth_cpt_full_module_weight_keys == "untouched"
    _peft(model)                      # the retry the message asks for
    assert len(_adapters(model, "language_model")) == 8
    assert _adapters(model, "vision_tower") == []


def _text_only_vlm():
    model = _vlm(tower=_tower(merger=_GLM_MERGER))
    model._unsloth_text_only_vlm = True
    return model


# A warning is a message, not a decision, so it precedes any modification.
@pytest.mark.parametrize("build,prefix,adapted", [
    (_text_only_vlm, "language_model", "vision_tower"),
    (_text_model, "model", None)],
    ids=["a vlm the forward pass never reaches", "no vision path at all"])
@pytest.mark.parametrize("flag", ["finetune_vision_layers", "train_vision",
                                  "train_projector"])
def test_a_flag_that_will_not_train_warns_before_the_model_is_touched(
        build, flag, prefix, adapted):
    model = build()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning):
            _peft(model, **{flag: True})
    assert _adapters(model) == []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _peft(model, **{flag: True})
    said = [str(w.message) for w in caught if "text_only" in str(w.message)]
    assert len(said) == 1, "the executing pass must not warn as well"
    assert f"{flag}=True" in said[0]
    assert ("text_only=True" if adapted else "no vision path") in said[0]
    assert len(_adapters(model, prefix)) == 8
    # Warning, not refusing: the tower really is wrapped, just never trained.
    assert bool(_adapters(model, adapted)) if adapted else True

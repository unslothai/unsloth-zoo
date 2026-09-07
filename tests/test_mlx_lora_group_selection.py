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
def _require_real_mlx():
    # Selection reads module trees, never a kernel, so the CPU backend answers
    # it as well as Metal does. The torch shim cannot: its module classes are
    # not the ones the selection isinstance-checks against.
    import mlx.core as _mx   # re-import: the shim may have swapped it
    if "mlx_simulation" in str(getattr(_mx, "__file__", "")):
        pytest.skip("requires the real MLX runtime; shim active")


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
    "a tower with only normalization and scalar gates": (
        lambda: _vlm(tower=_build({"patch_ln1": {}, "patch_dense": (VISION, 1)})),
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


# A connector nested in the tower is skipped so the projector pass can adapt it
# once. With that pass switched off there is no second pass, and skipping would
# drop linears the tower walk used to reach.
@pytest.mark.parametrize("train_projector,adapted_by", [
    (False, "the tower pass"), (True, "its own pass")],
    ids=["projector off", "projector on"])
def test_a_nested_connector_is_adapted_whichever_pass_owns_it(
        train_projector, adapted_by):
    model = _vlm(tower=_tower(merger=_GLM_MERGER))
    _peft(model, finetune_vision_layers=True, train_projector=train_projector)
    merger = _adapters(model, "vision_tower.merger")
    assert merger == ["down_proj", "gate_proj", "proj"], adapted_by
    # Once, not twice: an adapter stacked on an adapter would show a wrapped base.
    assert not [name for name, module in model.named_modules()
                if hasattr(module, "lora_a")
                and hasattr(getattr(module, "linear", None), "lora_a")]


def test_a_refusal_on_the_defaulted_path_names_no_target_modules():
    # The canonical list is substituted internally, so quoting it back sends
    # the caller to change an argument they never passed.
    model = _vlm(tower=_build({"patch_ln1": {}, "patch_dense": (VISION, 1)}))
    with pytest.raises(ValueError) as excinfo:
        _peft(model, finetune_vision_layers=True)
    # Suggesting one is fine; quoting back a list they never passed is not.
    assert "target_modules=" not in str(excinfo.value)
    assert "'patch_dense'" in str(excinfo.value)
    # A vocabulary the caller did pass is still worth naming back to them.
    model = _vlm(tower=_tower(merger=_GLM_MERGER))
    with pytest.raises(ValueError) as excinfo:
        _peft(model, finetune_vision_layers=True, target_modules=["q_proj"])
    assert "target_modules=['q_proj']" in str(excinfo.value)


# Routed experts are selectable, so the pass that adapts them must know the same
# module types selection does: a mixed tower silently trained only its ordinary
# linears, and a tower holding nothing else refused as though it were empty.
@pytest.mark.parametrize("mixed", [True, False],
                         ids=["beside ordinary linears", "alone in the tower"])
def test_a_tower_holding_routed_experts_adapts_them(mixed):
    switch_layers = pytest.importorskip("mlx_lm.models.switch_layers")
    block = {"mlp": {"switch_mlp": lambda: switch_layers.SwitchLinear(VISION, VISION, 4)}}
    if mixed:
        block["attn"] = {"qkv": (VISION, VISION * 3), "proj": (VISION, VISION)}
    model = _vlm(tower=_build({"blocks": [block], "patch_embed": {}}))
    _peft(model, finetune_vision_layers=True)
    adapted = _adapters(model, "vision_tower")
    assert "blocks.0.mlp.switch_mlp" in adapted
    assert ("blocks.0.attn.qkv" in adapted) is mixed


# `feed_forward` splits into two tokens, so the joined spelling in the MLP table
# did not reach it and the whole block lost its role.
@pytest.mark.parametrize("block_name,role", [
    ("feed_forward", "mlp"), ("feedforward", "mlp"), ("ffn", "mlp"),
    ("mlp", "mlp"), ("self_attn", "attention")])
def test_an_enclosing_block_names_its_role_however_it_is_spelled(block_name, role):
    from unsloth_zoo.mlx.loader import _linear_role
    assert _linear_role(f"{block_name}.dense_h_to_4h") == role


def test_a_block_spelled_feed_forward_is_adapted_by_the_mlp_flag():
    model = _text_model([{"self_attn": {"q_proj": (HIDDEN, HIDDEN)},
                          "feed_forward": {"dense_h_to_4h": (HIDDEN, HIDDEN),
                                           "dense_4h_to_h": (HIDDEN, HIDDEN)}}])
    _peft(model)
    assert _adapters(model, "model.layers.0") == [
        "feed_forward.dense_4h_to_h", "feed_forward.dense_h_to_4h",
        "self_attn.q_proj"]


# A role is read per layer, so the same generic name can be attention beside a
# qkv and MLP beside an fc1. Unioning the layers wrapped it in both.
def test_a_role_read_in_one_layer_does_not_reach_another():
    model = _text_model([{"mixer": {"qkv": (HIDDEN, HIDDEN * 3),
                                    "proj": (HIDDEN, HIDDEN)}},
                         {"mixer": {"fc1": (HIDDEN, HIDDEN),
                                    "proj": (HIDDEN, HIDDEN)}}])
    _peft(model, finetune_mlp_modules=False)
    assert _adapters(model, "model.layers.0") == ["mixer.proj", "mixer.qkv"]
    assert _adapters(model, "model.layers.1") == []


def test_a_fused_expert_stack_reports_output_width_not_expert_count():
    switch_layers = pytest.importorskip("mlx_lm.models.switch_layers")
    from unsloth_zoo.mlx.loader import _semantic_dims
    # (experts, out, in): the first axis counts experts, not outputs.
    assert _semantic_dims(switch_layers.SwitchLinear(HIDDEN, 7, 4)) == (7, HIDDEN)
    import mlx.nn as nn
    assert _semantic_dims(nn.Linear(HIDDEN, 7)) == (7, HIDDEN)


@pytest.mark.parametrize("module_name", ["mlx_lm.models.bitlinear_layers",
                                         "mlx_vlm.models.bitnet.bitlinear"])
@pytest.mark.parametrize("targets", [None, "q_proj", ["q_proj"], {"q_proj"},
                                     frozenset({"q_proj"}), "all-linear", ["all-linear"]])
def test_bitlinear_refuses_before_mutating_other_targets(module_name, targets):
    import importlib
    BitLinear = importlib.import_module(module_name).BitLinear
    model = _text_model([_TEXT_BLOCK, {"self_attn": {
        "q_proj": BitLinear(HIDDEN, HIDDEN, bias=False)}}])
    with pytest.raises(ValueError, match="BitLinear.*CustomKernel.*VJP"):
        _peft(model, target_modules=targets)
    assert not _adapters(model)


def test_vlm_bitlinear_refusal_preserves_trainability():
    from mlx_lm.models.bitlinear_layers import BitLinear
    from mlx.utils import tree_flatten
    model = _vlm(decoder=[_TEXT_BLOCK, {"self_attn": {
        "q_proj": BitLinear(HIDDEN, HIDDEN, bias=False)}}])
    before = [name for name, _ in tree_flatten(model.trainable_parameters())]
    with pytest.raises(ValueError, match="BitLinear.*CustomKernel.*VJP"):
        _peft(model)
    assert [name for name, _ in tree_flatten(model.trainable_parameters())] == before
    assert not _adapters(model)


@pytest.mark.parametrize("targets,attention", [(["lm_head"], True),
                                               (["q_proj", "lm_head"], False)])
def test_bitlinear_can_feed_a_downstream_head_adapter(targets, attention):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.bitnet import Model, ModelArgs
    model = Model(ModelArgs(
        model_type="bitnet", hidden_size=HIDDEN, intermediate_size=HIDDEN * 2,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        rms_norm_eps=1e-5, vocab_size=VOCAB, tie_word_embeddings=False,
    ))
    _peft(model, target_modules=targets, finetune_attention_modules=attention)
    _, grads = nn.value_and_grad(model, lambda m: m(mx.array([[1, 2]])).sum())(model)
    assert _adapters(model) == ["lm_head"]
    assert mx.abs(grads["lm_head"]["lora_b"]).max().item() > 0


def test_unselected_bitlinear_groups_are_left_alone():
    from mlx_lm.models.bitlinear_layers import BitLinear
    tower = _build({"q_proj": BitLinear(VISION, VISION, bias=False)})
    vlm = _vlm(tower=tower)
    _peft(vlm)
    assert not _adapters(vlm, "vision_tower")
    model = _text_model([{"self_attn": {"q_proj": BitLinear(HIDDEN, HIDDEN)}},
                         _TEXT_BLOCK])
    _peft(model, finetune_last_n_layers=1)
    assert not _adapters(model, "model.layers.0")
    assert _adapters(model, "model.layers.1")


def test_roleless_vision_projections_follow_default_group_flags():
    model = _vlm(tower=_build({"patch_ln1": {}, "patch_dense": (VISION, HIDDEN)}))
    _peft(model, finetune_vision_layers=True)
    assert _adapters(model, "vision_tower") == ["patch_dense"]
    for flag in ("finetune_attention_modules", "finetune_mlp_modules"):
        model = _vlm(tower=_build({"patch_dense": (VISION, HIDDEN)}))
        with pytest.raises(ValueError, match="read as attention or MLP"):
            _peft(model, finetune_vision_layers=True, **{flag: False})


def test_pointwise_vision_adapters_train_reload_and_fuse(tmp_path):
    import copy
    import json
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from unsloth_zoo.mlx.loader import _apply_lora_at_paths
    from unsloth_zoo.mlx.utils import save_lora_adapters

    tower = _build({"project": nn.Conv2d(4, 8, 1, stride=(2, 1), padding=(1, 0)),
                    "depthwise": nn.Conv2d(4, 4, 1, groups=4),
                    "spatial": nn.Conv2d(4, 8, 3)})
    model = _vlm(tower=tower)
    restored = copy.deepcopy(model)
    x = mx.random.normal((2, 7, 8, 4))
    before = tower.project(x)
    _peft(model, finetune_vision_layers=True)
    assert _adapters(model, "vision_tower") == ["project"]
    assert mx.array_equal(tower.project(x), before).item()
    loss = lambda m: mx.square(m.vision_tower.project(x)).mean()
    _, grads = nn.value_and_grad(model, loss)(model)
    optim.SGD(0.1).update(model, grads)
    mx.eval(model.parameters())
    assert mx.abs(tower.project.lora_b).max().item() > 0
    assert all("lora_" in name for name, _ in tree_flatten(model.trainable_parameters()))
    expected = tower.project(x)
    fused = tower.project.fuse()
    assert type(fused) is nn.Conv2d
    assert (fused.stride, fused.padding, fused.dilation) == ((2, 1), (1, 0), 1)
    assert mx.allclose(fused(x), expected, atol=1e-5).item()
    save_lora_adapters(model, tmp_path)
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    _apply_lora_at_paths(restored, _adapters(model), config)
    restored.load_weights(str(tmp_path / "adapters.safetensors"), strict=False)
    assert mx.array_equal(restored.vision_tower.project(x), expected).item()


@pytest.mark.parametrize("pixel_key", ["images", "pixel_values"])
def test_vision_lora_b_moves_through_the_processed_image_merge(pixel_key):
    from types import SimpleNamespace
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from unsloth_zoo.mlx.compile import _merge_special_token_features_only
    from unsloth_zoo.mlx.utils import _to_mx_vlm_batch, make_vlm_baseline_loss_fn

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=HIDDEN, image_token_index=7)
            self.language_model = _text_model([{"q_proj": (HIDDEN, HIDDEN)}])
            self.vision_tower = _build({"q_proj": (VISION, HIDDEN)})
            self._is_vlm_model = True

        def get_input_embeddings(self, ids, pixel_values=None):
            text = self.language_model.model.embed_tokens(ids)
            if pixel_values is not None:
                features = self.vision_tower.q_proj(pixel_values)
                text = _merge_special_token_features_only(7, None, features, text, ids)
            return SimpleNamespace(inputs_embeds=text)

        def __call__(self, ids, pixel_values=None, **kwargs):
            embeds = self.get_input_embeddings(ids, pixel_values).inputs_embeds
            hidden = self.language_model.model.layers[0].q_proj(mx.cumsum(embeds, axis=1))
            return self.language_model.lm_head(nn.tanh(hidden))

    model = _peft(Wrapper(), finetune_vision_layers=True)
    loss_fn = make_vlm_baseline_loss_fn(model)
    optimizer = optim.SGD(0.1)
    for ids, sign in (([[1, 7, 2, 3]], 1), ([[1, 2, 7, 3]], -1)):
        before = mx.array(model.vision_tower.q_proj.lora_b)
        batch = _to_mx_vlm_batch({"input_ids": mx.array(ids),
                                  pixel_key: sign * mx.ones((1, 1, VISION))})
        _, grads = nn.value_and_grad(model, lambda m: loss_fn(m, batch)[0])(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters())
        assert mx.abs(model.vision_tower.q_proj.lora_b - before).max().item() > 0


def test_native_string_message_format_keeps_numbered_images():
    from types import SimpleNamespace
    from unsloth_zoo.mlx.utils import _vlm_token_messages
    processor = SimpleNamespace(_unsloth_model_type="phi3_v")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "image"},
                 {"type": "text", "text": "Compare these."}]}]
    output = _vlm_token_messages(processor, messages)
    assert output[0]["content"] == "<|image_1|><|image_2|>Compare these."
    assert isinstance(messages[0]["content"], list)
    messages += [{"role": "assistant", "content": "Two images."},
                 {"role": "user", "content": [{"type": "image"},
                  {"type": "text", "text": "Compare with <|image_1|>."}]}]
    output = _vlm_token_messages(processor, messages)
    assert output[2]["content"] == "<|image_3|>Compare with <|image_1|>."


# An all-caps prefix runs into the next word, so a tower found only by its class
# name needs the acronym boundary as well as the camelCase one.
@pytest.mark.parametrize("name,expected", [
    ("CLIPVisionModel", {"clip", "vision", "model"}),
    ("SiglipVisionTransformer", {"siglip", "vision", "transformer"}),
    ("vision_tower", {"vision", "tower"})])
def test_a_class_name_splits_on_acronym_boundaries(name, expected):
    from unsloth_zoo.mlx.loader import _role_tokens
    assert expected <= _role_tokens(name)


def test_a_tower_named_only_by_its_class_still_resolves():
    import mlx.nn as nn
    from unsloth_zoo.mlx.loader import _resolve_vision_group
    # `encoder` says nothing, so only the class name marks this as a tower.
    tower = _tower()
    tower.__class__ = type("CLIPVisionModel", (nn.Module,), {})
    assert _resolve_vision_group(_vlm(tower_attr="encoder", tower=tower))[1] == "encoder"


def test_a_module_registered_in_a_mapping_is_reachable():
    from unsloth_zoo.mlx.loader import _named_child_modules, _navigate
    model = _vlm(tower_attr="placeholder", tower=_build({"x": (VISION, VISION)}))
    model.encoders = {"vision": _tower()}
    assert "encoders.vision" in [name for name, _ in _named_child_modules(model)]
    assert _navigate(model, "encoders.vision") is not None


def test_a_text_side_projection_is_not_the_vision_connector():
    from unsloth_zoo.mlx.loader import _resolve_vision_group, _resolve_projector_group
    model = _vlm()
    model.text_projection = _build({"linear_1": (VISION, HIDDEN)})
    owner, _, path, tower = _resolve_vision_group(model)
    assert _resolve_projector_group(model, owner, tower, path) == []


def test_a_root_module_sharing_a_layer_local_name_is_left_alone():
    import mlx.nn as nn
    model = _text_model([_MOLMO_BLOCK])
    model.ff_proj = nn.Linear(HIDDEN, HIDDEN)   # an unrelated auxiliary head
    _peft(model)
    assert _adapters(model) == [
        "model.layers.0.att_proj", "model.layers.0.attn_out",
        "model.layers.0.ff_out", "model.layers.0.ff_proj"]


def test_every_container_discovery_yields_can_be_written_back_into():
    # Selection is only useful if the pass that adapts can replace what it
    # found, so discovery must not name a container the writer cannot mutate.
    import mlx.nn as nn
    from unsloth_zoo.mlx.loader import _named_child_modules, _navigate, _set_child
    holder = _build({})
    holder.as_list = [nn.Linear(4, 4)]
    holder.as_dict = {"vision": nn.Linear(4, 4)}
    found = [name for name, _ in _named_child_modules(holder)]
    assert sorted(found) == ["as_dict.vision", "as_list.0"]
    for path in found:
        parent_path, _, leaf = path.rpartition(".")
        parent = _navigate(holder, parent_path)
        _set_child(parent, leaf, nn.Linear(4, 4))       # must not raise
        assert _navigate(holder, path) is not None


@pytest.mark.parametrize("pixel_key", ["images", "pixel_values"])
def test_generated_image_labels_ignore_negative_placeholders_only(pixel_key):
    import numpy as np
    import mlx.core as mx
    from unsloth_zoo.mlx.utils import (_stage_vlm_label_mask_np,
        _apply_vlm_label_masks, _to_mx_vlm_batch)
    ids = np.array([[1, -1, 2, -2], [-2, 3, -1, 4]], dtype=np.int32)
    inputs = {"input_ids": ids, pixel_key: np.ones((2, 1, VISION))}
    mask = _stage_vlm_label_mask_np(inputs)
    assert np.array_equal(mask, ids < 0)
    batch = _to_mx_vlm_batch(inputs)
    labels = _apply_vlm_label_masks(batch)
    assert np.array_equal(np.asarray(labels), np.where(ids < 0, -100, ids))
    assert not _stage_vlm_label_mask_np(inputs, labels=ids).any()
    assert mx.array_equal(_apply_vlm_label_masks(batch, labels=mx.array(ids)), mx.array(ids))
    assert not _stage_vlm_label_mask_np({"input_ids": ids}).any()
    assert mx.array_equal(_apply_vlm_label_masks({"input_ids": mx.array(ids)}), mx.array(ids))


@pytest.mark.parametrize("row", [0, 1])
def test_legacy_image_validation_checks_each_row(row):
    from unsloth_zoo.mlx.legacy_vision import validate_legacy_image_batch
    batch = {"input_ids": [[1, -200, -200, 2], [-200, -200, 3, 4]],
             "_unsloth_legacy_image_spec": (-200, 2)}
    validate_legacy_image_batch(batch)
    batch["input_ids"][row][1] = 5
    with pytest.raises(ValueError, match="split or removed"):
        validate_legacy_image_batch(batch)

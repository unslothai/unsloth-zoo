# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Regression tests for get_peft_regex's audio/vision-embedder LoRA targeting.

Goals (mirrors the #798 byte-identical methodology):
  * finetune_audio_layers defaults False and is strictly additive: for every
    existing architecture the produced regex never contains the audio branches
    when the flag is off, and is byte-identical with the flag on when the model
    has no audio_tower / Gemma embedder modules.
  * With the flag on, Gemma 4 / Gemma 3N attach their audio_tower Linears and the
    embed_audio.embedding_projection projector (0 -> N); the vision flag attaches
    embed_vision.embedding_projection. conv / *norm modules are never matched
    (they are not nn.Linear, so not even candidates).

Module names are taken verbatim from the real checkpoint tensor names
(`get_safetensors_metadata` on unsloth/gemma-4-E2B-it, unsloth/gemma-4-12B-it,
unsloth/gemma-3n-E4B-it), so the synthetic trees are faithful. We use synthetic
trees for determinism / no downloads; get_peft_regex only inspects named_modules()
+ isinstance(_, nn.Linear).
"""
import re
import pytest
import torch.nn as nn

from unsloth_zoo.peft_utils import get_peft_regex


# ----------------------------------------------------------------------------- helpers
def _add(root: nn.Module, path: str, leaf_module: nn.Module):
    *mids, leaf = path.split(".")
    cur = root
    for m in mids:
        if not hasattr(cur, m):
            setattr(cur, m, nn.Module())
        cur = getattr(cur, m)
    setattr(cur, leaf, leaf_module)


class FakeModel(nn.Module):
    """Builds a module tree from (path -> module-kind) specs."""
    def __init__(self, linear_paths, other_paths=(), name="fake", model_type="fake"):
        super().__init__()
        self.model = nn.Module()
        for p in linear_paths:
            _add(self.model, p, nn.Linear(8, 8, bias=False))
        for p, kind in other_paths:
            _add(self.model, p, kind())
        class _Cfg:
            pass
        cfg = _Cfg()
        cfg._name_or_path = name
        cfg.model_type = model_type
        self.config = cfg


def linear_names(model):
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]


def matched(regex, names):
    # PEFT uses re.fullmatch; assert on that (no DOTALL needed - names have no newlines).
    return {n for n in names if re.fullmatch(regex, n)}


def leaf(name):
    # Strip a trailing Gemma4ClippableLinear ".linear" suffix only (do NOT touch a
    # leaf literally named e.g. "linear_start"), then take the last path segment.
    if name.endswith(".linear"):
        name = name[: -len(".linear")]
    return name.rsplit(".", 1)[-1]


# ----------------------------------------------------------------------------- fixtures
def _text_stack(prefix, n=2, heads=("q_proj", "k_proj", "v_proj", "o_proj"),
                mlp=("gate_proj", "up_proj", "down_proj")):
    out = []
    for i in range(n):
        out += [f"{prefix}.{i}.self_attn.{h}" for h in heads]
        out += [f"{prefix}.{i}.mlp.{m}" for m in mlp]
    return out


# Existing architectures (no Gemma embedder / no audio LoRA target today).
LLAMA      = _text_stack("language_model.layers") + ["lm_head"]
QWEN3      = _text_stack("language_model.layers") + ["lm_head"]
QWEN2_VL   = (_text_stack("language_model.layers")
              + [f"visual.blocks.{i}.attn.{p}" for i in range(2) for p in ("qkv", "proj")]
              + [f"visual.blocks.{i}.mlp.{p}" for i in range(2) for p in ("fc1", "fc2")]
              + ["visual.merger.mlp.0", "visual.merger.mlp.2"])
LLAVA      = (_text_stack("language_model.layers")
              + [f"vision_tower.vision_model.encoder.layers.{i}.self_attn.{p}"
                 for i in range(2) for p in ("q_proj", "k_proj", "v_proj", "out_proj")]
              + ["multi_modal_projector.linear_1", "multi_modal_projector.linear_2"])
GEMMA3_VL  = (_text_stack("language_model.layers")
              + [f"vision_tower.vision_model.encoder.layers.{i}.self_attn.{p}"
                 for i in range(2) for p in ("q_proj", "k_proj", "v_proj", "out_proj")]
              + [f"vision_tower.vision_model.encoder.layers.{i}.mlp.{p}"
                 for i in range(2) for p in ("fc1", "fc2")]
              + ["multi_modal_projector.mm_input_projection"])
# Audio-bearing existing models: MUST stay byte-identical when the flag is OFF.
QWEN2_AUDIO = (_text_stack("language_model.layers")
               + [f"audio_tower.layers.{i}.self_attn.{p}"
                  for i in range(2) for p in ("q_proj", "k_proj", "v_proj", "out_proj")]
               + [f"audio_tower.layers.{i}.fc1" for i in range(2)]
               + [f"audio_tower.layers.{i}.fc2" for i in range(2)]
               + ["multi_modal_projector.linear"])
# Whisper-style standalone audio model: a language stack so get_peft_regex applies,
# plus an audio encoder named `audio_model.encoder.layers` (NO `audio_tower` segment),
# to prove the audio branch keys on the `audio_tower` anchor specifically.
WHISPER     = (_text_stack("language_model.layers")
               + [f"audio_model.encoder.layers.{i}.self_attn.{p}"
                  for i in range(2) for p in ("q_proj", "k_proj", "v_proj", "out_proj")])

NON_GEMMA_ARCHS = {
    "llama": LLAMA, "qwen3": QWEN3, "qwen2_vl": QWEN2_VL,
    "llava": LLAVA, "gemma3_vl": GEMMA3_VL,
}
AUDIO_ARCHS = {"qwen2_audio": QWEN2_AUDIO, "whisper": WHISPER}

FLAG_COMBOS = [
    dict(finetune_vision_layers=True,  finetune_language_layers=True,  finetune_attention_modules=True,  finetune_mlp_modules=True),
    dict(finetune_vision_layers=False, finetune_language_layers=True,  finetune_attention_modules=True,  finetune_mlp_modules=True),
    dict(finetune_vision_layers=True,  finetune_language_layers=False, finetune_attention_modules=True,  finetune_mlp_modules=True),
    dict(finetune_vision_layers=True,  finetune_language_layers=True,  finetune_attention_modules=True,  finetune_mlp_modules=False),
    dict(finetune_vision_layers=True,  finetune_language_layers=True,  finetune_attention_modules=False, finetune_mlp_modules=True),
]


# Gemma audio/vision trees with REAL leaf names (verified from checkpoint metadata).
def _gemma3n():
    lin = _text_stack("language_model.layers")
    conf = []
    others = []
    for i in range(2):
        b = f"audio_tower.conformer.{i}"
        conf += [f"{b}.attention.attn.{p}" for p in ("q_proj", "k_proj", "v_proj")]
        conf += [f"{b}.attention.attn.relative_position_embedding.pos_proj"]
        conf += [f"{b}.attention.post"]
        conf += [f"{b}.ffw_layer_start.ffw_layer_1", f"{b}.ffw_layer_start.ffw_layer_2"]
        conf += [f"{b}.ffw_layer_end.ffw_layer_1",   f"{b}.ffw_layer_end.ffw_layer_2"]
        conf += [f"{b}.lconv1d.linear_start", f"{b}.lconv1d.linear_end"]
        # non-Linear decoys under audio_tower:
        others += [(f"{b}.lconv1d.depthwise_conv1d", lambda: nn.Conv1d(8, 8, 3)),
                   (f"{b}.lconv1d.conv_norm", lambda: nn.LayerNorm(8)),
                   (f"{b}.attention.post_norm", lambda: nn.LayerNorm(8))]
    conf += ["audio_tower.subsample_conv_projection.input_proj_linear"]
    others += [("audio_tower.subsample_conv_projection.conv_0.conv", lambda: nn.Conv2d(1, 8, 3)),
               ("embed_audio.hard_embedding_norm", lambda: nn.LayerNorm(8)),
               ("embed_vision.soft_embedding_norm", lambda: nn.LayerNorm(8))]
    # Decoy encoder named audio_model.* (NOT audio_tower) to prove the audio branch
    # keys on the audio_tower segment and never sweeps a differently-named encoder.
    lin += [f"audio_model.encoder.layers.{i}.self_attn.q_proj" for i in range(2)]
    lin += conf + ["embed_audio.embedding_projection", "embed_vision.embedding_projection"]
    return FakeModel(lin, others, name="unsloth/gemma-3n-E4B-it", model_type="gemma3n")


def _gemma4(size="E2B"):
    lin = _text_stack("language_model.layers")
    others = []
    # vision_tower encoder (already matched today via 'vision' tag) - ClippableLinear .linear children
    for i in range(2):
        b = f"vision_tower.encoder.layers.{i}"
        lin += [f"{b}.self_attn.{p}.linear" for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
        lin += [f"{b}.mlp.{p}.linear" for p in ("gate_proj", "up_proj", "down_proj")]
    lin += ["vision_tower.patch_embedder.input_proj"]
    # audio_tower conformer (Gemma4 ClippableLinear -> .linear children)
    for i in range(2):
        b = f"audio_tower.layers.{i}"
        lin += [f"{b}.feed_forward1.ffw_layer_1.linear", f"{b}.feed_forward1.ffw_layer_2.linear"]
        lin += [f"{b}.feed_forward2.ffw_layer_1.linear", f"{b}.feed_forward2.ffw_layer_2.linear"]
        lin += [f"{b}.attention.attn.q_proj.linear", f"{b}.attention.attn.k_proj.linear",
                f"{b}.attention.attn.v_proj.linear", f"{b}.attention.attn.relative_k_proj.linear"]
        lin += [f"{b}.attention.post.linear", f"{b}.lconv1d.linear_start.linear",
                f"{b}.lconv1d.linear_end.linear"]
        others += [(f"{b}.lconv1d.depthwise_conv1d", lambda: nn.Conv1d(8, 8, 3)),
                   (f"{b}.attention.norm_post_attn", lambda: nn.LayerNorm(8))]
    lin += ["audio_tower.subsample_conv_projection.input_proj_linear",
            "audio_tower.output_proj"]
    lin += ["embed_vision.embedding_projection", "embed_audio.embedding_projection"]
    if size == "12B":
        lin += ["vision_embedder.patch_dense"]
    return FakeModel(lin, others, name=f"unsloth/gemma-4-{size}-it", model_type="gemma4")


# ----------------------------------------------------------------------------- tests
@pytest.mark.parametrize("arch_name", list(NON_GEMMA_ARCHS) + list(AUDIO_ARCHS))
@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_audio_flag_off_is_inert(arch_name, flags):
    """With finetune_audio_layers=False the regex must not contain audio branches,
    and the matched set is identical to omitting the kwarg entirely."""
    names_map = {**NON_GEMMA_ARCHS, **AUDIO_ARCHS}
    model = FakeModel(names_map[arch_name])
    ns = linear_names(model)
    r_absent = get_peft_regex(model, **flags)
    r_off    = get_peft_regex(model, **flags, finetune_audio_layers=False)
    assert r_absent == r_off, "audio=False must equal omitting the kwarg"
    assert "audio_tower" not in r_off, "audio branch leaked with flag off"
    assert matched(r_off, ns) == matched(r_absent, ns)


@pytest.mark.parametrize("arch_name", list(NON_GEMMA_ARCHS))
@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_audio_flag_additive_for_non_audio_archs(arch_name, flags):
    """Models with no audio_tower / Gemma embedder are byte-identical whether the
    audio flag is on or off (branches only appended when they actually match)."""
    model = FakeModel(NON_GEMMA_ARCHS[arch_name])
    r_off = get_peft_regex(model, **flags, finetune_audio_layers=False)
    r_on  = get_peft_regex(model, **flags, finetune_audio_layers=True)
    assert r_off == r_on


def test_qwen2audio_off_no_leak_and_no_qwen_specific_leaves():
    # finetune_audio_layers is scoped to the models we ship notebooks for
    # (Gemma 4 / Gemma 3N) via a model_type / architectures gate. On a non-notebook
    # audio_tower (Qwen2-Audio, model_type != gemma) the audio branch is never even
    # built, so the flag is a no-op whether off OR on -- in particular it never
    # targets Qwen-specific leaves (out_proj / fc1 / fc2).
    model = FakeModel(QWEN2_AUDIO, model_type="qwen2_audio")
    ns = linear_names(model)
    off = matched(get_peft_regex(model, finetune_audio_layers=False), ns)
    on  = matched(get_peft_regex(model, finetune_audio_layers=True), ns)
    assert not any("audio_tower" in n for n in off)
    assert off == on, "non-Gemma audio model must be unaffected by finetune_audio_layers"
    on_leaves = {leaf(n) for n in on if "audio_tower" in n}
    assert not ({"out_proj", "fc1", "fc2"} & on_leaves), \
        f"unexpectedly targeted Qwen-specific audio leaves: {on_leaves}"


def test_audio_branch_keys_on_audio_tower_anchor():
    # The audio branch is anchored on the `audio_tower` segment. A differently-named
    # encoder (audio_model.encoder.*) sitting on the SAME Gemma model must NOT be
    # swept in -- the anchor, not just the model-type gate, has to hold.
    model = _gemma3n()  # carries an audio_model.encoder decoy alongside audio_tower
    ns = linear_names(model)
    on = matched(get_peft_regex(model, finetune_vision_layers=False,
                                finetune_language_layers=True,
                                finetune_audio_layers=True), ns)
    assert any(".language_model." in n for n in on)
    assert any(".audio_tower." in n for n in on)
    assert not any("audio_model" in n for n in on), \
        f"audio branch leaked into a non-audio_tower encoder: {[n for n in on if 'audio_model' in n]}"


@pytest.mark.parametrize("builder,name", [(_gemma3n, "gemma3n"),
                                          (lambda: _gemma4("E2B"), "gemma4_e2b"),
                                          (lambda: _gemma4("12B"), "gemma4_12b")])
def test_gemma_audio_attaches(builder, name):
    model = builder()
    ns = linear_names(model)

    # Notebook-style config BEFORE the fix: audio off -> 0 audio modules.
    off = matched(get_peft_regex(model, finetune_vision_layers=False,
                                 finetune_language_layers=True,
                                 finetune_audio_layers=False), ns)
    assert not any(".audio_tower." in n or n.endswith("embed_audio.embedding_projection") for n in off)

    # WITH the fix.
    on = matched(get_peft_regex(model, finetune_vision_layers=False,
                                finetune_language_layers=True,
                                finetune_audio_layers=True), ns)
    audio_hits = {n for n in on if ".audio_tower." in n}
    assert audio_hits, f"{name}: audio_tower got 0 LoRA targets"
    assert any(n.endswith("embed_audio.embedding_projection") for n in on), f"{name}: projector missed"

    # The conformer feed-forward + attention + lconv leaves attach.
    leaves = {leaf(n) for n in audio_hits}
    assert {"ffw_layer_1", "ffw_layer_2"} <= leaves
    assert {"linear_start", "linear_end"} <= leaves
    assert "post" in leaves
    assert {"q_proj", "k_proj", "v_proj"} <= leaves

    # NEGATIVE: conv / norm leaves never matched (not nn.Linear -> not candidates).
    assert not any(n.endswith("depthwise_conv1d") for n in on)
    assert not any(n.endswith(".conv") or n.endswith("_norm") or n.endswith(".norm") for n in on)

    # Language stack unchanged off<->on.
    assert {n for n in off if ".language_model." in n} == {n for n in on if ".language_model." in n}


@pytest.mark.parametrize("builder", [_gemma3n, lambda: _gemma4("E2B"), lambda: _gemma4("12B")])
def test_vision_flag_attaches_vision_projector(builder):
    model = builder()
    ns = linear_names(model)
    on = matched(get_peft_regex(model, finetune_vision_layers=True,
                                finetune_language_layers=True,
                                finetune_audio_layers=False), ns)
    assert any(n.endswith("embed_vision.embedding_projection") for n in on)


def test_audio_only_does_not_match_language():
    """Audio-only (vision=False, language=False, audio=True) must not degenerate
    into matching every attn/mlp leaf in the language stack."""
    model = _gemma3n()
    ns = linear_names(model)
    on = matched(get_peft_regex(model, finetune_vision_layers=False,
                                finetune_language_layers=False,
                                finetune_audio_layers=True), ns)
    assert on, "audio-only should still match audio modules"
    assert not any(".language_model." in n for n in on), "audio-only leaked into language stack"


def test_vision_only_no_language_does_not_touch_language():
    # finetune_vision_layers=True, language=False on Gemma 3N: the only vision
    # Linear is the flat embed_vision.embedding_projection. The component-only
    # fallback must NOT fire before the embedder branch and broaden into the
    # language stack.
    model = _gemma3n()
    ns = linear_names(model)
    on = matched(get_peft_regex(model, finetune_vision_layers=True,
                                finetune_language_layers=False,
                                finetune_audio_layers=False), ns)
    assert any(n.endswith("embed_vision.embedding_projection") for n in on)
    assert not any(".language_model." in n for n in on), \
        f"vision-only leaked into the language stack: {[n for n in on if '.language_model.' in n]}"


def test_audio_respects_explicit_target_modules():
    # With an explicit target_modules list, the audio branch must intersect with it
    # rather than attach every Gemma audio leaf.
    model = _gemma3n()
    ns = linear_names(model)
    on = matched(get_peft_regex(model, finetune_vision_layers=False,
                                finetune_language_layers=True,
                                finetune_audio_layers=True,
                                target_modules=["q_proj", "v_proj"]), ns)
    at = {leaf(n) for n in on if ".audio_tower." in n}
    assert {"q_proj", "v_proj"} <= at
    assert "k_proj" not in at and "ffw_layer_1" not in at and "post" not in at, \
        f"explicit target_modules not respected by audio branch: {at}"
    # embedding_projection was not in the list -> the projector is not attached
    assert not any(n.endswith("embed_audio.embedding_projection") for n in on)


def test_explicit_leaf_matches_full_segment_not_prefix():
    # target_modules=["k_proj"] must match audio_tower ...attn.k_proj but NOT
    # ...attn.relative_k_proj (the ".*?" before the leaf must stop at a "." boundary).
    model = _gemma4("E2B")  # Gemma 4 audio_tower has relative_k_proj (+ .linear children)
    ns = linear_names(model)
    on = matched(get_peft_regex(model, finetune_vision_layers=False,
                                finetune_language_layers=True,
                                finetune_audio_layers=True,
                                target_modules=["k_proj"]), ns)
    at = [n for n in on if ".audio_tower." in n]
    assert any(leaf(n) == "k_proj" for n in at), "k_proj should match in audio_tower"
    assert not any(leaf(n) == "relative_k_proj" for n in at), \
        f"k_proj wrongly matched relative_k_proj: {at}"


def test_audio_respects_attn_mlp_flags():
    # attention off, mlp on: the conformer attention leaves (q/k/v/post) must not
    # attach, but the feed-forward leaves must.
    model = _gemma3n()
    ns = linear_names(model)
    on = matched(get_peft_regex(model, finetune_vision_layers=False,
                                finetune_language_layers=True,
                                finetune_attention_modules=False,
                                finetune_mlp_modules=True,
                                finetune_audio_layers=True), ns)
    at = {leaf(n) for n in on if ".audio_tower." in n}
    assert {"ffw_layer_1", "ffw_layer_2"} <= at
    assert "q_proj" not in at and "k_proj" not in at and "post" not in at, \
        f"attn-off should drop audio attention leaves: {at}"


def test_positional_target_modules_not_shadowed():
    # finetune_audio_layers lives at the END of the signature, so the 6th positional
    # argument is still target_modules and positional callers are unaffected.
    model = FakeModel(LLAMA)
    ns = linear_names(model)
    kw = get_peft_regex(model, True, True, True, True, ["q_proj", "v_proj"])  # 6th positional = target_modules
    pos = get_peft_regex(model, finetune_vision_layers=True, finetune_language_layers=True,
                         finetune_attention_modules=True, finetune_mlp_modules=True,
                         target_modules=["q_proj", "v_proj"])
    assert kw == pos
    m = matched(kw, ns)
    assert m and all(leaf(n) in ("q_proj", "v_proj") for n in m)


def test_positional_tag_overrides_not_shadowed():
    # The tag-override params (vision_tags/language_tags/...) keep their positional
    # slots: passing a list as the 7th positional must bind to vision_tags, NOT to
    # finetune_audio_layers (which now sits after the tag params).
    model = FakeModel(LLAMA)
    # 7th positional = vision_tags. A made-up tag should appear verbatim in the regex.
    r = get_peft_regex(model, True, True, True, True, None, ["zzvision"])
    assert "zzvision" in r, "7th positional did not bind to vision_tags"
    assert "audio_tower" not in r, "list mis-bound to finetune_audio_layers"


def test_embedder_branches_gated_to_gemma_model_type():
    # empty_model.py lists embed_vision.embedding_projection and
    # vision_tower.patch_embedder.input_proj as Mistral3 components too. On a
    # non-Gemma checkpoint that happens to carry those exact Linears, the default
    # finetune_vision_layers=True must NOT start matching them (byte-identical set).
    shared = ["embed_vision.embedding_projection",
              "vision_tower.patch_embedder.input_proj"]
    mistral = FakeModel(_text_stack("language_model.layers") + shared,
                        name="mistralai/Mistral-Small-3", model_type="mistral3")
    ns = linear_names(mistral)
    on = matched(get_peft_regex(mistral, finetune_vision_layers=True,
                                finetune_language_layers=True), ns)
    assert not any(n.endswith(tuple(shared)) for n in on), \
        f"non-Gemma embedder Linears wrongly targeted: {[n for n in on if n.endswith(tuple(shared))]}"
    # The very same module names DO attach on a Gemma model.
    gem = _gemma4("12B")
    gon = matched(get_peft_regex(gem, finetune_vision_layers=True,
                                 finetune_language_layers=True), linear_names(gem))
    assert any(n.endswith("embed_vision.embedding_projection") for n in gon)


def test_projector_branches_gated_under_mlp_flag():
    # The vision/audio projectors are feed-forward projections, so attention-only
    # runs (finetune_mlp_modules=False) must not pick them up; mlp-on must.
    model = _gemma3n()
    ns = linear_names(model)
    # attention-only -> no projectors
    attn_only = matched(get_peft_regex(model, finetune_vision_layers=True,
                                       finetune_language_layers=True,
                                       finetune_attention_modules=True,
                                       finetune_mlp_modules=False,
                                       finetune_audio_layers=True), ns)
    assert not any(n.endswith("embed_vision.embedding_projection") for n in attn_only), \
        "vision projector attached under attention-only"
    assert not any(n.endswith("embed_audio.embedding_projection") for n in attn_only), \
        "audio projector attached under attention-only"
    # mlp-on -> projectors present
    mlp_on = matched(get_peft_regex(model, finetune_vision_layers=True,
                                    finetune_language_layers=True,
                                    finetune_attention_modules=False,
                                    finetune_mlp_modules=True,
                                    finetune_audio_layers=True), ns)
    assert any(n.endswith("embed_vision.embedding_projection") for n in mlp_on)
    assert any(n.endswith("embed_audio.embedding_projection") for n in mlp_on)


def test_guard_allows_audio_only_and_blocks_nothing_selected():
    model = _gemma3n()
    # nothing selected -> raises
    with pytest.raises(RuntimeError):
        get_peft_regex(model, finetune_vision_layers=False,
                       finetune_language_layers=False, finetune_audio_layers=False)
    # audio-only -> must NOT raise
    get_peft_regex(model, finetune_vision_layers=False,
                   finetune_language_layers=False, finetune_audio_layers=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

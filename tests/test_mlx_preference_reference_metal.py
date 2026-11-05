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

"""Real-runtime contracts of MLX preference training."""

import functools

import pytest

try:
    import mlx.core as mx
    _METAL = mx.metal.is_available()
except Exception:
    _METAL = False

pytestmark = pytest.mark.skipif(not _METAL, reason="requires Apple Silicon Metal")


def _tiny():
    import mlx.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(16, 8)
            self.proj = nn.Linear(8, 8, bias=False)
            self.drop = nn.Dropout(0.0)
            self.head = nn.Linear(8, 16, bias=False)

        def __call__(self, tokens):
            return self.head(self.drop(self.proj(self.embed(tokens))))

    mx.random.seed(0)
    model = Tiny()
    mx.eval(model.parameters())
    return model


def _adapted(*, dora=False, train_embedding=False):
    from mlx_lm.tuner.dora import DoRAEmbedding, DoRALinear
    from mlx_lm.tuner.lora import LoRALinear

    model = _tiny()
    keys = ["lora_a", "lora_b"] + (["m"] if dora else [])
    for name in ("proj", "head"):
        wrapped = (DoRALinear if dora else LoRALinear).from_base(
            getattr(model, name), r=2, scale=1.0,
        )
        setattr(model, name, wrapped)
    if dora:
        model.embed = DoRAEmbedding.from_base(model.embed, r=2, scale=1.0)
    model.freeze()
    for name in ("proj", "head") + (("embed",) if dora else ()):
        getattr(model, name).unfreeze(keys=keys)
    if train_embedding:
        model.embed.unfreeze()
    mx.eval(model.parameters())
    return model


def _batch():
    tokens = mx.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 7, 8, 0], [1, 2, 9, 10, 0, 0],
                       [1, 2, 11, 12, 13, 0]], dtype=mx.int32)
    return tokens, mx.array([[3, 6], [3, 5], [2, 4], [2, 5]], dtype=mx.int32)


def _build(model, **options):
    from unsloth_zoo.mlx.preference import build_reference_policy
    return build_reference_policy(
        model, reference_free=False, resume_provenance=None, **options,
    )[0]


@pytest.mark.parametrize("mode", [
    "adapter", "full", "ref_model", "synced_full", "synced_model",
    "synced_adapter", "synced_dora", "synced_adapted_model",
])
def test_reference_overrides_hold_inside_a_compiled_step(mode):
    """The compiled step scores against the initial weights or the synced mix."""
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_map
    from unsloth_zoo.mlx.preference import _response_logps

    synced = mode.startswith("synced")
    with_model = mode in ("ref_model", "synced_model", "synced_adapted_model")
    adapted = mode in ("adapter", "ref_model", "synced_adapter", "synced_dora",
                       "synced_adapted_model")
    build = functools.partial(
        _adapted, dora=mode == "synced_dora", train_embedding=True,
    ) if adapted else _tiny
    model = build()
    pristine = build() if synced and adapted else _tiny()
    tokens, lengths = _batch()
    options = {"sync_ref_model": synced}
    if with_model:
        reference = build() if mode == "synced_adapted_model" else _tiny()
        # Inert only because the build puts the reference in eval mode.
        reference.drop = nn.Dropout(0.5)
        options.update(ref_model=reference, force_use_ref_model=True)
    policy = _build(model, **options)
    if with_model:
        assert policy.state is reference.state
    optimizer = optim.SGD(learning_rate=0.5)

    def loss_fn(model, tokens, lengths):
        scored = policy.forward(model, tokens, lengths)
        return (_response_logps(model, tokens, lengths) - scored).sum(), scored

    value_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state, policy.state]

    @functools.partial(mx.compile, inputs=state, outputs=state)
    def step(tokens, lengths):
        (loss, scored), grads = value_and_grad(model, tokens, lengths)
        optimizer.update(model, grads)
        return loss, scored

    embedding = model.embed.embedding if mode == "synced_dora" else model.embed
    initial = embedding.weight
    for index in range(3):
        loss, scored = step(tokens, lengths)
        mx.eval(loss, scored, model.parameters())
        if with_model and index == 0:
            scaled = pristine.embed.weight * 1.5
            reference.embed.weight = scaled
            pristine.embed.weight = scaled
        if synced and index == 1:
            policy.sync(model, 0.25)
            pristine.update(tree_map(
                lambda held, trained: 0.75 * held + 0.25 * trained,
                pristine.trainable_parameters(), model.trainable_parameters(),
            ))

    assert not mx.allclose(embedding.weight, initial).item(), "the policy moved"
    if mode == "adapter":
        assert (model.proj.scale, model.head.scale) == (1.0, 1.0)
    if with_model:
        assert reference.training is False
    expected = _response_logps(pristine, tokens, lengths)
    assert mx.allclose(scored, expected, atol=1e-5).item()
    assert loss.item() != 0.0, "the loss sees the gap"


def test_a_qualified_vision_plan_is_surveyed_and_planned_for_compile():
    from types import SimpleNamespace
    from unsloth_zoo.mlx.compile import build_compile_policy
    from unsloth_zoo.mlx.preference import FiniteVisionPreferenceBatchPlan
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, _plan_single_process_text_shapes

    class Plan(FiniteVisionPreferenceBatchPlan):
        pixel_width = 33
        def _inputs(self, rows, batch, lengths):
            inputs = {"input_ids": mx.array(batch), "attention_mask": mx.array(
                (batch != self.pad_id).astype("int32"))}
            if rows[0].image:
                inputs["pixel_values"] = mx.zeros((len(batch), self.pixel_width))
            return inputs

    row = lambda length, image: SimpleNamespace(
        chosen=(1,) * length, rejected=(1,) * (length - 2), image=image,
        chosen_prompt_ids=(1, 1), rejected_prompt_ids=(1, 1))

    plan = Plan(
        [row(20, False), row(50, True)], [[0], [1]], processor=None, model_config={},
        normalizers=[(1, 1, 1)] * 2, cycle_length=2, max_seq_length=64, pad_id=0)
    args = MLXDPOConfig(max_steps=2, gradient_accumulation_steps=1, max_grad_norm=0.0)

    def planned(decision):
        return _plan_single_process_text_shapes(
            plan, None, args=args, total_steps=2, is_vlm=True, distributed_world_size=1,
            compile_policy=build_compile_policy(args=args), vlm_compile_decision=decision)

    assert planned(None)[1].reason == "vlm_compile_unqualified" and plan.batch_width(0) == 33
    assert planned(SimpleNamespace(enabled=True))[1:3] == (plan._shape_plan.report, True)
    # The text batch steps off the width the image batch's pixels already use.
    assert [plan.batch_width(index) for index in (0, 1)] == [34, 64]
    assert plan.batch_family(0)[4] != plan.batch_family(1)[4]
    text, image = (plan.materialize(index, phase="single")[0] for index in (0, 1))
    assert text["input_ids"].shape == (2, 34) and "pixel_values" not in text
    assert image["input_ids"].shape == (2, 64) and image["pixel_values"].shape == (2, 33)
    assert plan[0][0]["input_ids"].shape == (2, 33)
    with pytest.raises(RuntimeError, match="not admitted"):
        plan.materialize(1, phase="update")
    Plan.pixel_width = 16
    with pytest.raises(RuntimeError, match="drifted from its surveyed compile family"):
        plan.materialize(1, phase="single")


def _tiny_gemma4_vlm(seed):
    from mlx_vlm.models import gemma4

    mx.random.seed(seed)
    return gemma4.Model(gemma4.ModelConfig(image_token_id=500, vocab_size=512, text_config=gemma4.TextConfig(
        hidden_size=64, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, head_dim=32, global_head_dim=64, vocab_size=512,
        vocab_size_per_layer_input=512, hidden_size_per_layer_input=16, num_kv_shared_layers=0), vision_config=gemma4.VisionConfig(hidden_size=32,
        intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=16, global_head_dim=16, position_embedding_size=16)))


def _tiny_vision_batches(objective):
    """Two rows of an image pooled to 4 tokens per branch (chosen/rejected, or completion/KL), laid out as _vision_batch does."""
    ids, pixels = mx.array([[3, *[500] * 4, 7, 8 + row, 9] for row in range(4)]), mx.random.normal((2, 3, 96, 96))
    if objective == "dpo":
        return {"input_ids": ids, "pixel_values": mx.concatenate([pixels] * 2), "_unsloth_image_copies": 2}, {}
    return {"input_ids": ids[:2], "pixel_values": pixels}, {"kl_batch": {"input_ids": ids[2:], "pixel_values": pixels}}


@pytest.mark.parametrize("objective", ["dpo", "kto"])
@pytest.mark.parametrize("tower", ["frozen", "trained", "adapted", "overridden", "neftune", "ref_model"])
def test_vision_preference_reuses_image_features_exactly(tower, objective, monkeypatch):
    import types, mlx.nn as nn, mlx.utils
    from mlx_lm.tuner.lora import LoRALinear
    from unsloth_zoo.mlx import preference, utils

    model = _tiny_gemma4_vlm(0)
    batch, kl = _tiny_vision_batches(objective)
    attention = model.language_model.model.layers[0].self_attn
    text = attention.q_proj = LoRALinear.from_base(attention.q_proj, r=2)
    vision = model.embed_vision.embedding_projection = LoRALinear.from_base(model.embed_vision.embedding_projection, r=2)
    for lora in (text, vision):
        lora.lora_b = mx.full(lora.lora_b.shape, 0.1)
    model.freeze()
    model.language_model.model.norm.unfreeze()
    if tower == "trained":
        model.vision_tower.unfreeze()
    # Every reference drops the text adapter, so policy and reference differ from the first step.
    reference = {
        "adapted": dict(scales=[(text, 0.0), (vision, 0.0)]),
        "overridden": dict(scales=[(text, 0.0)], overrides=[(vision, "lora_b", mx.zeros_like(vision.lora_b))]),
        "neftune": dict(scales=[(text, 0.0)], neftune_modules=[vision]),
        "ref_model": dict(model=_tiny_gemma4_vlm(1)),
    }.get(tower, dict(scales=[(text, 0.0)]))
    policy = preference.ReferencePolicy(**reference)
    fn = getattr(preference, f"make_{objective}_loss_fn")(preference.resolve_preference_objective(objective, beta=0.1, loss_type=(
        "kto" if objective == "kto" else "sigmoid")), reference_policy=policy)
    args = [mx.array(value) for value in ([[6, 8]] * 4, [0, 2, 1], [True, False])][:2 + (objective == "kto")]
    if objective == "dpo":
        # One copy's features repeated for both branches, one row per image token.
        tiled = utils._vlm_image_features(model, batch, 2)
        assert tiled.shape[-2] == (batch["input_ids"] == 500).sum().item() == 16 and mx.array_equal(tiled[:, :8], tiled[:, 8:]).item()
    rows, call = [], type(model.vision_tower).__call__
    monkeypatch.setattr(type(model.vision_tower), "__call__", lambda self, x, *a, **k: rows.append(len(x)) or call(self, x, *a, **k))
    # A KTO precompute scores the same split dicts, one chunk of two rows.
    plan = types.SimpleNamespace(rows=[0, 1], pack=lambda chunk: (None, [[6, 8]] * 4), set_reference_logps=lambda table: None,
                                 _scored_inputs=lambda *_a, **_k: [batch, kl["kl_batch"]])

    def step():
        rows.clear()
        loss, grads = nn.value_and_grad(model, lambda: fn(model, batch, *args, **kl)[0])()
        mx.eval(loss, grads)
        tower_rows = sum(rows)
        rows.clear()
        table = mx.array(preference.precompute_reference_logps(plan, model, policy, batch_size=2)) if kl else None
        return types.SimpleNamespace(loss=loss, grads=dict(mlx.utils.tree_flatten(grads)), tower_rows=tower_rows,
                                     precompute_rows=sum(rows), table=table)

    reused = step()
    monkeypatch.setattr(utils, "_VLM_IMAGE_ENCODERS", {})
    recomputed = step()
    assert "cached_image_features" not in {**batch, **kl.get("kl_batch", {})}
    # Today every forward runs the tower: 4 forwards x 2 images. KTO's KL forwards read their completion's features.
    assert recomputed.tower_rows == 8 and reused.tower_rows == (2 if tower == "frozen" else 4)
    if kl:
        assert recomputed.precompute_rows == 4 and reused.precompute_rows == 2
        assert mx.allclose(reused.table, recomputed.table, rtol=1e-6, atol=0).item()
    assert abs(recomputed.loss.item() - (0.6931472 if objective == "dpo" else 0.5)) > 1e-3
    assert reused.grads.keys() == recomputed.grads.keys()
    assert mx.allclose(reused.loss, recomputed.loss, rtol=1e-6, atol=0).item() and all(
        mx.allclose(grad, recomputed.grads[path], rtol=1e-5, atol=1e-7).item() for path, grad in reused.grads.items())


def test_vision_image_features_refuse_active_dropout():
    from mlx_lm.tuner.lora import LoRALinear
    from unsloth_zoo.mlx import utils

    model = _tiny_gemma4_vlm(0)
    lora = model.embed_vision.embedding_projection = LoRALinear.from_base(model.embed_vision.embedding_projection, r=2, dropout=0.5)
    batch, _ = _tiny_vision_batches("dpo")
    # Each copy must draw its own mask, so the batch keeps a tower run per copy.
    assert utils._vlm_image_encoder(model) is None and utils._vlm_image_features(model, batch, 2) is None
    model.eval()
    assert utils._vlm_image_encoder(model) is not None
    model.train()
    # Toggled in place, as PreferenceRunContext.restore() does, with no change of mode or frozen keys.
    lora.dropout._p_1 = 1.0
    assert utils._vlm_image_encoder(model) is not None
    lora.dropout._p_1 = 0.5
    assert utils._vlm_image_encoder(model) is None


def test_qwen35_image_features_match_the_tower_over_every_copy():
    from mlx_vlm.models.qwen3_5 import config, qwen3_5
    from unsloth_zoo.mlx import utils

    mx.random.seed(0)
    model = qwen3_5.Model(config.ModelConfig(
        model_type="qwen3_5", vocab_size=64, image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63,
        text_config=config.TextConfig(model_type="qwen3_5", hidden_size=32, intermediate_size=64, linear_num_value_heads=2,
                                      linear_num_key_heads=2, linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
                                      num_hidden_layers=1, num_attention_heads=2, rms_norm_eps=1e-6, vocab_size=64, num_key_value_heads=1,
                                      max_position_embeddings=128, head_dim=16),
        vision_config=config.VisionConfig(depth=1, hidden_size=32, intermediate_size=64, out_hidden_size=32, num_heads=2, patch_size=16,
                                          num_position_embeddings=64)))
    model.vision_tower.set_dtype(mx.bfloat16)
    # Per copy, one row of two images of 16 and 32 patches merged 2x2 into 4 + 8 tokens; float32 pixels as the processor returns them.
    grid, pixels = mx.array([[1, 4, 4], [1, 4, 8]]), mx.random.normal((48, 3 * 2 * 16 * 16))
    ids = mx.array([[62, *[60] * 4, 63, 62, *[60] * 8, 63, 5, 6]] * 2)
    batch = {"input_ids": ids, "pixel_values": mx.concatenate([pixels] * 2), "image_grid_thw": mx.concatenate([grid] * 2)}
    features = utils._vlm_image_features(model, batch, 2)
    expected = model.vision_tower(batch["pixel_values"].astype(mx.bfloat16), batch["image_grid_thw"])[0]
    assert features.dtype == expected.dtype == mx.bfloat16 and features.shape == expected.shape == (24, 32)
    # Through the embedder's cached-features path, as the preference forwards hand them over.
    fresh = model.get_input_embeddings(ids, batch["pixel_values"], image_grid_thw=batch["image_grid_thw"])
    cached = model.get_input_embeddings(ids, batch["pixel_values"], image_grid_thw=batch["image_grid_thw"], cached_image_features=features)
    assert mx.array_equal(features, expected).item() and mx.array_equal(fresh.inputs_embeds, cached.inputs_embeds).item()
    assert mx.array_equal(fresh.position_ids, cached.position_ids).item()

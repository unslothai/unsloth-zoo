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

"""CUDA-parity continued pretraining, end to end on real MLX (Apple Silicon).

Drives get_peft_model + save routing (LR key, full-module unfreeze, save/reload).
Fine-grained matrix: analysis_artifacts/2026-07-21_mlx-cpt-validation/.
"""

from __future__ import annotations

import os
import tempfile

import pytest

# Skip before importing mlx.nn/utils so non-MLX collection does not error.
pytest.importorskip("mlx.core")
import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mu


@pytest.fixture(autouse=True)
def _require_real_metal():
    # Re-import: the shim may have swapped mlx.core in sys.modules since import.
    import mlx.core as _mx
    if not (getattr(_mx, "metal", None) and _mx.metal.is_available()
            and _mx.default_device() == _mx.gpu):
        pytest.skip("real Metal required; shim active or no GPU")


class _Attn(nn.Module):
    def __init__(s):
        super().__init__()
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(s, n, nn.Linear(32, 32, bias=False))


class _Core(nn.Module):
    def __init__(s, tied, emb):
        super().__init__()
        s._emb, s._tied = emb, tied
        s.model = nn.Module()
        setattr(s.model, emb, nn.Embedding(64, 32))
        lyr = nn.Module(); lyr.self_attn = _Attn(); s.model.layers = [lyr]
        if tied:
            s.args = type("A", (), {"tie_word_embeddings": True})()
        else:
            s.lm_head = nn.Linear(32, 64, bias=False)

    @property
    def layers(s):
        return s.model.layers

    def __call__(s, x):
        e = getattr(s.model, s._emb); h = e(x)
        return e.as_linear(h) if s._tied else s.lm_head(h)


def _tiny(tied=False, emb="embed_tokens"):
    return _Core(tied, emb)


def _peft(model, **kw):
    from unsloth_zoo.mlx.loader import FastMLXModel
    return FastMLXModel.get_peft_model(
        model, r=4, use_gradient_checkpointing="none", **kw)


def test_untied_cpt_partition_lr_key_and_save_reload():
    m = _tiny()
    _peft(m, target_modules=["q_proj", "embed_tokens", "lm_head"])
    trainable = set(dict(mu.tree_flatten(m.trainable_parameters())))
    assert "model.embed_tokens.weight" in trainable
    assert {"lm_head.lora_a", "lm_head.lora_b"} <= trainable
    assert "lm_head.weight" not in trainable
    assert m._unsloth_cpt_full_module_weight_keys == {"model.embed_tokens.weight"}
    # Save keeps the full module; a reloaded model (no marker) saves it too.
    from unsloth_zoo.mlx.loader import _mlx_save_lora_adapters  # noqa: E402
    d, d2 = tempfile.mkdtemp(), tempfile.mkdtemp()
    _mlx_save_lora_adapters(m, d)
    assert "model.embed_tokens.weight" in mx.load(os.path.join(d, "adapters.safetensors"))
    r = _tiny(); r.freeze(); r.model.embed_tokens.unfreeze(recurse=True)
    _mlx_save_lora_adapters(r, d2)
    assert "model.embed_tokens.weight" in mx.load(os.path.join(d2, "adapters.safetensors"))


def test_tied_trains_shared_matrix_and_rejects_lm_head():
    m = _tiny(tied=True)
    _peft(m, target_modules=["embed_tokens"], finetune_language_layers=False)
    assert m._unsloth_cpt_full_module_weight_keys == {"model.embed_tokens.weight"}
    # Co-requesting lm_head trains the shared matrix rather than erroring.
    m2 = _tiny(tied=True)
    _peft(m2, target_modules=["embed_tokens", "lm_head"], finetune_language_layers=False)
    assert m2._unsloth_cpt_full_module_weight_keys == {"model.embed_tokens.weight"}
    trn = set(dict(mu.tree_flatten(m2.trainable_parameters())))
    assert not any(k.startswith("lm_head") for k in trn)  # tied: no separate head
    with pytest.raises(ValueError, match="tied"):
        _peft(_tiny(tied=True), target_modules=["lm_head"])


def test_set_valued_target_modules_are_not_silently_emptied():
    # A set/frozenset selection must be honored, not dropped to keys=None.
    m = _tiny()
    _peft(m, target_modules={"q_proj", "embed_tokens"})
    assert m._unsloth_cpt_full_module_weight_keys == {"model.embed_tokens.weight"}
    # A bare-string modules_to_save must not iterate characters.
    ms = _tiny()
    _peft(ms, target_modules=["q_proj"], modules_to_save="embed_tokens")
    assert ms._unsloth_cpt_full_module_weight_keys == {"model.embed_tokens.weight"}


def test_all_linear_sentinel_accepts_set_form():
    # {"all-linear"} must expand like the string sentinel, not be a literal name.
    m = _tiny()
    _peft(m, target_modules={"all-linear"})
    trn = set(dict(mu.tree_flatten(m.trainable_parameters())))
    assert any(k.endswith("q_proj.lora_a") for k in trn)


def test_all_linear_expands_inside_a_mixed_cpt_target_list():
    # ["all-linear", "embed_tokens"] is the CPT spelling of the sentinel. A
    # literal "all-linear" left in the list matches no module, and the CPT
    # entry keeps the no-target guard quiet, so every layer adapter vanished.
    m = _tiny()
    _peft(m, target_modules=["all-linear", "embed_tokens"])
    trn = set(dict(mu.tree_flatten(m.trainable_parameters())))
    assert "model.embed_tokens.weight" in trn
    for leaf in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert any(k.endswith(f"{leaf}.lora_a") for k in trn), leaf
    assert m._unsloth_cpt_full_module_weight_keys == {"model.embed_tokens.weight"}

    # The head entry survives the PEFT-parity exclusion of the output layer.
    h = _tiny()
    _peft(h, target_modules=["all-linear", "lm_head"])
    trn = set(dict(mu.tree_flatten(h.trainable_parameters())))
    assert {"lm_head.lora_a", "lm_head.lora_b"} <= trn
    assert any(k.endswith("q_proj.lora_a") for k in trn)

    # Set form, and a duplicate sentinel, resolve the same way.
    s = _tiny()
    _peft(s, target_modules={"all-linear", "embed_tokens", "lm_head"})
    trn = set(dict(mu.tree_flatten(s.trainable_parameters())))
    assert {"lm_head.lora_a", "model.embed_tokens.weight"} <= trn
    assert any(k.endswith("v_proj.lora_a") for k in trn)


def test_full_module_only_checkpoint_reloads_without_unfreezing_the_base():
    # No LoRA anywhere, so the artifact is stamped fine_tune_type="full" with
    # no exact LoRA paths and reload takes the pathless route. mlx-lm's
    # load_adapters never freezes, so without a selective freeze every base
    # tensor came back trainable and the scoped-LR keys were lost.
    import json

    from unsloth_zoo.mlx.loader import (
        _is_partial_mlx_checkpoint,
        _load_pathless_mlx_adapter,
        _mlx_save_lora_adapters,
        _normalize_mlx_lora_module_paths,
    )

    m = _tiny()
    _peft(m, target_modules=["embed_tokens"])
    assert set(dict(mu.tree_flatten(m.trainable_parameters()))) == {
        "model.embed_tokens.weight"}
    d = tempfile.mkdtemp()
    _mlx_save_lora_adapters(m, d)
    weights = os.path.join(d, "adapters.safetensors")
    assert sorted(mx.load(weights)) == ["model.embed_tokens.weight"]
    with open(os.path.join(d, "adapter_config.json")) as fh:
        cfg = json.load(fh)
    assert cfg["fine_tune_type"] == "full"
    assert not _normalize_mlx_lora_module_paths(
        cfg.get("unsloth_mlx_lora_module_paths"))

    r = _load_pathless_mlx_adapter(_tiny(), d, weights, cfg, False)
    assert set(dict(mu.tree_flatten(r.trainable_parameters()))) == {
        "model.embed_tokens.weight"}
    assert getattr(r, "_unsloth_cpt_full_module_weight_keys", set()) == {
        "model.embed_tokens.weight"}
    assert mx.array_equal(r.model.embed_tokens.weight, m.model.embed_tokens.weight)

    # A full-module head-only checkpoint restores the head, nothing else.
    hm = _tiny()
    _peft(hm, target_modules=[], modules_to_save=["lm_head"])
    hd = tempfile.mkdtemp()
    _mlx_save_lora_adapters(hm, hd)
    hw = os.path.join(hd, "adapters.safetensors")
    with open(os.path.join(hd, "adapter_config.json")) as fh:
        hcfg = json.load(fh)
    hr = _load_pathless_mlx_adapter(_tiny(), hd, hw, hcfg, False)
    assert set(dict(mu.tree_flatten(hr.trainable_parameters()))) == {
        "lm_head.weight"}
    assert getattr(hr, "_unsloth_cpt_full_module_weight_keys", set()) == {
        "lm_head.weight"}

    # A whole-model artifact is a real full fine-tune: keep it unfrozen.
    fd = tempfile.mkdtemp()
    whole = _tiny()
    mx.save_safetensors(
        os.path.join(fd, "adapters.safetensors"),
        dict(mu.tree_flatten(whole.parameters())),
    )
    assert not _is_partial_mlx_checkpoint(
        _tiny(), os.path.join(fd, "adapters.safetensors"))
    # full_finetuning=True never freezes either.
    ff = _load_pathless_mlx_adapter(_tiny(), d, weights, cfg, True)
    assert len(dict(mu.tree_flatten(ff.trainable_parameters()))) == len(
        dict(mu.tree_flatten(ff.parameters())))


def test_set_target_modules_respects_finetune_filters():
    # A set selection must still honor finetune_attention_modules=False.
    m = _tiny()
    _peft(m, target_modules={"q_proj", "embed_tokens"},
          finetune_attention_modules=False)
    trn = set(dict(mu.tree_flatten(m.trainable_parameters())))
    assert "model.embed_tokens.weight" in trn
    assert not any("q_proj.lora" in k for k in trn)


class _AltHead(nn.Module):
    """No lm_head and no tie flag: the head descriptor stays unresolved."""

    def __init__(s):
        super().__init__()
        s.model = nn.Module()
        s.model.embed_tokens = nn.Embedding(64, 32)
        lyr = nn.Module(); lyr.self_attn = _Attn(); s.model.layers = [lyr]
        s.embed_out = nn.Linear(32, 64, bias=False)

    @property
    def layers(s):
        return s.model.layers

    def __call__(s, x):
        return s.embed_out(s.model.embed_tokens(x))


def test_unusable_lm_head_keeps_the_other_lora_targets():
    # An untrainable lm_head must warn and drop, not abort, so the other
    # targets keep the pre-CPT behaviour.
    m = _tiny(tied=True)
    with pytest.warns(UserWarning, match="tied"):
        _peft(m, target_modules=["q_proj", "lm_head"])
    trn = set(dict(mu.tree_flatten(m.trainable_parameters())))
    assert any(k.endswith("q_proj.lora_a") for k in trn)
    assert not any(k.startswith("lm_head") for k in trn)

    alt = _AltHead()
    with pytest.warns(UserWarning, match="output head could not be resolved"):
        _peft(alt, target_modules=["q_proj", "lm_head"])
    assert any(k.endswith("q_proj.lora_a")
               for k in dict(mu.tree_flatten(alt.trainable_parameters())))


class _WrappedHead(nn.Module):
    """Phixtral's OutputHead shape: a LayerNorm+Linear module, not a Linear."""

    def __init__(s, d=32, v=64):
        super().__init__()
        s.ln = nn.LayerNorm(d); s.linear = nn.Linear(d, v, bias=False)

    def __call__(s, x):
        return s.linear(s.ln(x))


def test_lm_head_wrapper_mlx_lm_cannot_lora_is_dropped_not_fatal():
    # An unwrappable head must warn, not abort; modules_to_save still trains it.
    m = _tiny(); m.lm_head = _WrappedHead()
    with pytest.warns(UserWarning, match="cannot wrap as a LoRA layer"):
        _peft(m, target_modules=["q_proj", "lm_head"])
    assert any(k.endswith("q_proj.lora_a")
               for k in dict(mu.tree_flatten(m.trainable_parameters())))

    full = _tiny(); full.lm_head = _WrappedHead()
    _peft(full, target_modules=["q_proj"], modules_to_save=["lm_head"])
    trn = set(dict(mu.tree_flatten(full.trainable_parameters())))
    assert "lm_head.linear.weight" in trn


def test_wrapper_head_records_its_real_weight_keys_for_the_scoped_lr():
    # A wrapper head owns no `.weight`, so `lm_head.weight` named nothing live
    # and embedding_learning_rate never reached the head.
    m = _tiny(); m.lm_head = _WrappedHead()
    _peft(m, target_modules=["q_proj"], modules_to_save=["lm_head"])
    keys = m._unsloth_cpt_full_module_weight_keys
    assert keys == {"lm_head.ln.weight", "lm_head.linear.weight"}
    trainable = set(dict(mu.tree_flatten(m.trainable_parameters())))
    assert keys <= trainable
    # A plain Linear / Embedding full module keeps the exact previous key.
    plain = _tiny()
    _peft(plain, target_modules=["q_proj", "embed_tokens"],
          modules_to_save=["lm_head"])
    assert plain._unsloth_cpt_full_module_weight_keys == {
        "model.embed_tokens.weight", "lm_head.weight",
    }


def test_quantized_child_of_a_wrapper_head_is_rejected():
    # A wrapper head carries `scales` on a descendant, which a leaf-only check
    # missed until MLX raised at the first backward.
    m = _tiny(); m.lm_head = _WrappedHead()
    m.lm_head.linear = nn.QuantizedLinear.from_linear(
        m.lm_head.linear, group_size=32, bits=4)
    with pytest.raises(ValueError, match=r"quantized module at 'lm_head\.linear'"):
        _peft(m, target_modules=["q_proj"], modules_to_save=["lm_head"])
    # A directly quantized full module still reports its own path.
    e = _tiny()
    e.model.embed_tokens = nn.QuantizedEmbedding.from_embedding(
        e.model.embed_tokens, group_size=32, bits=4)
    with pytest.raises(
        ValueError, match=r"quantized module at 'model\.embed_tokens'",
    ):
        _peft(e, target_modules=["q_proj", "embed_tokens"])


def test_unusable_lm_head_still_raises_when_nothing_else_trains():
    # Dropping lm_head must never leave an empty selection for auto-discovery.
    with pytest.raises(ValueError, match="tied"):
        _peft(_tiny(tied=True), target_modules=["lm_head"])
    with pytest.raises(ValueError, match="output head could not be resolved"):
        _peft(_AltHead(), target_modules=["lm_head"])
    with pytest.raises(ValueError, match="tied"):
        _peft(_tiny(tied=True), target_modules=["q_proj"],
              modules_to_save=["lm_head"])


class _VlmCore(nn.Module):
    """VLM wrapper: the head is a root module of the language stack."""

    def __init__(s):
        super().__init__()
        s.language_model = _Core(False, "embed_tokens")
        s._is_vlm_model = True

    @property
    def layers(s):
        return s.language_model.layers

    def __call__(s, x):
        return s.language_model(x)


def test_root_lm_head_lora_is_attached_outside_the_layer_walk():
    # The head is a root module, so a layer-only walk never reaches it.
    head_only = _tiny()
    _peft(head_only, target_modules=["lm_head"])
    assert {"lm_head.lora_a", "lm_head.lora_b"} <= set(
        dict(mu.tree_flatten(head_only.trainable_parameters())))

    mixed = _tiny()
    _peft(mixed, target_modules=["q_proj", "lm_head"])
    trn = set(dict(mu.tree_flatten(mixed.trainable_parameters())))
    assert {"lm_head.lora_a", "lm_head.lora_b"} <= trn
    assert any(k.endswith("q_proj.lora_a") for k in trn)

    # CPT recipe: an lm_head adapter plus a full embedding, no layer targets.
    cpt = _tiny()
    _peft(cpt, target_modules=["embed_tokens", "lm_head"])
    trn = set(dict(mu.tree_flatten(cpt.trainable_parameters())))
    assert {"lm_head.lora_a", "lm_head.lora_b",
            "model.embed_tokens.weight"} <= trn

    vlm = _VlmCore()
    _peft(vlm, target_modules=["q_proj", "lm_head"],
          finetune_vision_layers=False, train_projector=False)
    trn = set(dict(mu.tree_flatten(vlm.trainable_parameters())))
    assert {"language_model.lm_head.lora_a",
            "language_model.lm_head.lora_b"} <= trn


def test_reloaded_cpt_adapter_rebuilds_the_scoped_lr_keys():
    # Reload restores trainability, so the scoped-LR keys must come back too or
    # embedding_learning_rate degrades to the main LR for alt-named embeddings.
    import json

    from unsloth_zoo.mlx.loader import (
        _apply_lora_at_paths,
        _mlx_save_lora_adapters,
        _unfreeze_saved_mlx_non_adapter_parameters,
    )

    m = _tiny(emb="tok_embeddings")
    _peft(m, target_modules=["q_proj", "embed_tokens"])
    assert m._unsloth_cpt_full_module_weight_keys == {
        "model.tok_embeddings.weight"}
    d = tempfile.mkdtemp()
    _mlx_save_lora_adapters(m, d)
    weights = os.path.join(d, "adapters.safetensors")
    with open(os.path.join(d, "adapter_config.json")) as fh:
        cfg = json.load(fh)

    r = _tiny(emb="tok_embeddings")
    r.freeze()
    _apply_lora_at_paths(r, cfg.get("unsloth_mlx_lora_module_paths"), cfg,
                         adapter_weights_file=weights)
    r.load_weights(weights, strict=False)
    _unfreeze_saved_mlx_non_adapter_parameters(r, weights)
    assert "model.tok_embeddings.weight" in set(
        dict(mu.tree_flatten(r.trainable_parameters())))
    assert getattr(r, "_unsloth_cpt_full_module_weight_keys", set()) == {
        "model.tok_embeddings.weight"}

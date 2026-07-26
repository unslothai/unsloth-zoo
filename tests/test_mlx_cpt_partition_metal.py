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

"""CUDA-parity continued pretraining, end-to-end on real MLX (Apple Silicon).

Drives get_peft_model + save routing (LR key, full-module unfreeze, save/reload).
Fine-grained matrix: analysis_artifacts/2026-07-21_mlx-cpt-validation/.
"""

from __future__ import annotations

import os
import tempfile

import pytest

# Skip the whole module (before importing mlx.nn/utils) when MLX is absent, so
# collection on Linux / non-MLX environments does not raise ModuleNotFoundError.
pytest.importorskip("mlx.core")
import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mu


@pytest.fixture(autouse=True)
def _require_real_metal():
    # Re-import: the shim swaps mlx.core in sys.modules at run time, after this
    # module imported real MLX; these must skip (not fail) under the shim.
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
    # embed -> full module (weight trainable); lm_head -> LoRA; q_proj -> LoRA.
    trainable = set(dict(mu.tree_flatten(m.trainable_parameters())))
    assert "model.embed_tokens.weight" in trainable
    assert {"lm_head.lora_a", "lm_head.lora_b"} <= trainable
    assert "lm_head.weight" not in trainable
    # AD-7: recorded LR keys are the exact registered full-module weight keys.
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
    # The documented recipe co-requests embed_tokens + lm_head; on a tied model
    # that trains the shared matrix via embed_tokens rather than erroring.
    m2 = _tiny(tied=True)
    _peft(m2, target_modules=["embed_tokens", "lm_head"], finetune_language_layers=False)
    assert m2._unsloth_cpt_full_module_weight_keys == {"model.embed_tokens.weight"}
    trn = set(dict(mu.tree_flatten(m2.trainable_parameters())))
    assert not any(k.startswith("lm_head") for k in trn)  # tied: no separate head
    # A standalone lm_head request is still rejected.
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
    # target_modules={"all-linear"} must expand like the string sentinel rather
    # than be treated as a literal module name that matches nothing.
    m = _tiny()
    _peft(m, target_modules={"all-linear"})
    trn = set(dict(mu.tree_flatten(m.trainable_parameters())))
    assert any(k.endswith("q_proj.lora_a") for k in trn)


def test_set_target_modules_respects_finetune_filters():
    # A set selection must still honor finetune_attention_modules=False:
    # q_proj (attention) is filtered out while embed_tokens still trains.
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
    # target_modules=[..., "lm_head"] LoRA'd the other targets before CPT
    # existed. An lm_head this backend cannot train must not abort the run:
    # warn and drop it, so tied models (Llama/Qwen/Gemma) and unresolved-head
    # models (GPT-NeoX embed_out, InternLM2 output) still train.
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
    # mlx-lm's to_lora() raises for a head it cannot wrap, which used to take
    # the whole run down; modules_to_save still trains it as a full module.
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
    # A wrapper head owns no `.weight` of its own, so recording `lm_head.weight`
    # named a tensor that does not exist: it matches neither the trainer's
    # recorded-key lookup nor its `<...>.embed_tokens/lm_head.weight` leaf
    # fallback, and embedding_learning_rate silently never reached the head.
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
    # Quantization on a wrapper head lives on a descendant, so a leaf-only
    # `scales` check accepted it and MLX then raised
    # "[QuantizedMatmul::vjp] no gradient wrt the quantized weights" at the
    # first backward instead of this actionable error.
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
    # Dropping lm_head must never leave an empty selection to fall through to
    # mlx-lm's auto-discovery, and an explicit modules_to_save request names
    # one module, so both keep raising.
    with pytest.raises(ValueError, match="tied"):
        _peft(_tiny(tied=True), target_modules=["lm_head"])
    with pytest.raises(ValueError, match="output head could not be resolved"):
        _peft(_AltHead(), target_modules=["lm_head"])
    with pytest.raises(ValueError, match="tied"):
        _peft(_tiny(tied=True), target_modules=["q_proj"],
              modules_to_save=["lm_head"])

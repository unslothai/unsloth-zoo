# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License, version 3 or (at your
# option) any later version. See <https://www.gnu.org/licenses/>.

"""CUDA-parity continued pretraining, end-to-end on real MLX (Apple Silicon).

Drives get_peft_model + save routing (LR key, full-module unfreeze, save/reload).
Fine-grained matrix: analysis_artifacts/2026-07-21_mlx-cpt-validation/.
"""

from __future__ import annotations

import os
import tempfile

import pytest
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
    with pytest.raises(ValueError, match="tied"):
        _peft(_tiny(tied=True), target_modules=["lm_head"])

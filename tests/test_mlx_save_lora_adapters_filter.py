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

"""Regression coverage for save_lora_adapters / save_trainable_adapters.

- save_lora_adapters keeps only module-anchored lora_a / lora_b tensors,
  even when base weights are listed as trainable, and raises if no
  LoRA modules are present.
- save_trainable_adapters preserves every trainable tensor for in-loop
  checkpoints.
- The module-anchored filter does not leak paths that merely contain
  "lora_" (e.g. router.lora_gate.weight).

Runs on Linux + Windows via the mlx_simulation shim, on macOS against
real MLX.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    import sys
    if "mlx" in sys.modules:
        return
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


class _MockDropoutKeepProb:
    def __init__(self, p):
        self._p_1 = 1.0 - p


class _MockLoRALinear:
    def __init__(self, in_features, out_features, rank, scale, dropout):
        self.weight = torch.zeros(out_features, in_features)
        # mlx-lm convention: lora_a (in_features, rank), lora_b (rank, out_features)
        self.lora_a = torch.zeros(in_features, rank)
        self.lora_b = torch.zeros(rank, out_features)
        self.scale = scale
        self.dropout = dropout


class _MockPlainLinear:
    def __init__(self, in_features, out_features):
        self.weight = torch.zeros(out_features, in_features)


def _make_model(layers, **attrs):
    class _M:
        def __init__(self):
            for k, v in layers.items():
                setattr(self, k, v)
            self._hf_repo = attrs.get("_hf_repo", None)
            self._config = None
            self._unsloth_quantization_config = None
            self._unsloth_quantization_policy = None
            self._unsloth_quantized_source = None
            self._unsloth_base_revision = None
            self._unsloth_base_commit_hash = None
            self._src_path = None

        def parameters(self):
            out = {}
            for name, mod in layers.items():
                for attr in ("weight", "bias", "lora_a", "lora_b"):
                    v = getattr(mod, attr, None)
                    if isinstance(v, torch.Tensor):
                        out[f"{name}.{attr}"] = v
            return out

        def trainable_parameters(self):
            return self.parameters()

        def named_modules(self):
            yield "", self
            for name, mod in layers.items():
                yield name, mod

    return _M()


def test_save_lora_adapters_keeps_only_lora_tensors(tmp_path):
    from unsloth_zoo.mlx.utils import save_lora_adapters

    model = _make_model({
        "q_proj": _MockLoRALinear(8, 16, 4, 2.5, _MockDropoutKeepProb(0.0)),
        "up_proj": _MockPlainLinear(16, 32),
    })
    save_lora_adapters(model, tmp_path)

    from safetensors.torch import load_file
    keys = set(load_file(str(tmp_path / "adapters.safetensors")).keys())
    assert keys == {"q_proj.lora_a", "q_proj.lora_b"}, sorted(keys)


def test_save_lora_adapters_does_not_leak_paths_containing_lora_(tmp_path):
    # Anchor-on-modules filter must drop a non-LoRA tensor whose path
    # happens to contain "lora_" (e.g. a routing layer literally named
    # `lora_router`).
    from unsloth_zoo.mlx.utils import save_lora_adapters

    real = _MockLoRALinear(8, 16, 4, 1.0, _MockDropoutKeepProb(0.0))
    fake = _MockPlainLinear(8, 8)
    model = _make_model({"q_proj": real, "lora_router": fake})
    save_lora_adapters(model, tmp_path)

    from safetensors.torch import load_file
    keys = set(load_file(str(tmp_path / "adapters.safetensors")).keys())
    assert keys == {"q_proj.lora_a", "q_proj.lora_b"}, sorted(keys)


def test_save_lora_adapters_raises_when_no_lora_modules(tmp_path):
    from unsloth_zoo.mlx.utils import save_lora_adapters

    model = _make_model({"up_proj": _MockPlainLinear(16, 32)})
    with pytest.raises(ValueError, match="LoRA adapter tensors"):
        save_lora_adapters(model, tmp_path)


def test_save_trainable_adapters_preserves_full_trainable_tree(tmp_path):
    from unsloth_zoo.mlx.utils import save_trainable_adapters

    model = _make_model({
        "q_proj": _MockLoRALinear(8, 16, 4, 1.0, _MockDropoutKeepProb(0.0)),
        "up_proj": _MockPlainLinear(16, 32),
    })
    save_trainable_adapters(model, tmp_path)

    from safetensors.torch import load_file
    keys = set(load_file(str(tmp_path / "adapters.safetensors")).keys())
    assert keys == {
        "q_proj.weight", "q_proj.lora_a", "q_proj.lora_b",
        "up_proj.weight",
    }, sorted(keys)


class _MockLoRALinearUpper:
    # Some external adapters expose uppercase tensor names. mlx-lm reload
    # only recreates lowercase wrappers so the collector should skip these,
    # not silently emit unreloadable adapter weights.
    def __init__(self, in_features, out_features, rank, scale, dropout):
        self.weight = torch.zeros(out_features, in_features)
        self.lora_A = torch.zeros(in_features, rank)
        self.lora_B = torch.zeros(rank, out_features)
        self.scale = scale
        self.dropout = dropout


class _MockHalfLoRA:
    # Module that only exposes one side of the LoRA pair (e.g. half-built
    # during construction). Saving such modules would produce unreloadable
    # adapter files; the collector must drop them.
    def __init__(self, in_features, rank):
        self.weight = torch.zeros(in_features, in_features)
        self.lora_a = torch.zeros(in_features, rank)


def _make_model_with_named_modules(layers):
    # Variant of _make_model that exposes upper- and lower-case LoRA attr
    # pairs when listing parameters.
    class _M:
        def __init__(self):
            for k, v in layers.items():
                setattr(self, k, v)
            self._hf_repo = None
            self._config = None
            self._unsloth_quantization_config = None
            self._unsloth_quantization_policy = None
            self._unsloth_quantized_source = None
            self._unsloth_base_revision = None
            self._unsloth_base_commit_hash = None
            self._src_path = None

        def parameters(self):
            out = {}
            for name, mod in layers.items():
                for attr in ("weight", "bias", "lora_a", "lora_b",
                             "lora_A", "lora_B"):
                    v = getattr(mod, attr, None)
                    if isinstance(v, torch.Tensor):
                        out[f"{name}.{attr}"] = v
            return out

        def trainable_parameters(self):
            return self.parameters()

        def named_modules(self):
            yield "", self
            for name, mod in layers.items():
                yield name, mod

    return _M()


def test_collect_lora_helper_skips_uppercase_only_module(tmp_path):
    # mlx-lm reload only recreates lowercase lora_a/lora_b wrappers, so
    # adapter tensors saved under uppercase keys can never bind back to
    # the recreated wrappers. The collector must skip these modules and
    # save_lora_adapters must raise rather than silently produce an
    # unreloadable artifact.
    from unsloth_zoo.mlx.utils import (
        collect_mlx_lora_adapter_tensors,
        save_lora_adapters,
    )

    model = _make_model_with_named_modules({
        "q_proj": _MockLoRALinearUpper(8, 16, 4, 1.0, _MockDropoutKeepProb(0.0)),
        "up_proj": _MockPlainLinear(16, 32),
    })
    assert collect_mlx_lora_adapter_tensors(model) == {}
    with pytest.raises(ValueError, match="LoRA adapter tensors"):
        save_lora_adapters(model, tmp_path)


def test_collect_lora_helper_drops_half_adapter_module(tmp_path):
    # A module that only exposes lora_a (no lora_b) cannot be reloaded; the
    # collector must skip it so save_lora_adapters surfaces the empty-set
    # error rather than write a half-broken adapter file.
    from unsloth_zoo.mlx.utils import (
        collect_mlx_lora_adapter_tensors,
        save_lora_adapters,
    )

    model = _make_model_with_named_modules({
        "broken": _MockHalfLoRA(8, 4),
    })
    assert collect_mlx_lora_adapter_tensors(model) == {}
    with pytest.raises(ValueError, match="LoRA adapter tensors"):
        save_lora_adapters(model, tmp_path)


def test_iter_mlx_lora_modules_yields_only_lowercase_pairs():
    from unsloth_zoo.mlx.utils import iter_mlx_lora_modules

    model = _make_model_with_named_modules({
        "q_proj": _MockLoRALinear(8, 16, 4, 1.0, _MockDropoutKeepProb(0.0)),
        "v_proj": _MockLoRALinearUpper(8, 16, 4, 1.0, _MockDropoutKeepProb(0.0)),
        "up_proj": _MockPlainLinear(16, 32),
    })
    names = [name for name, _m in iter_mlx_lora_modules(model)]
    assert names == ["q_proj"], names


def test_enrich_adapter_config_skips_uppercase_only_modules():
    # mlx-lm cannot reload uppercase tensors, so the enrich helper must not
    # advertise uppercase-only modules as reloadable LoRA paths.
    from unsloth_zoo.mlx.utils import _enrich_mlx_adapter_config

    model = _make_model_with_named_modules({
        "q_proj": _MockLoRALinearUpper(8, 16, 4, 1.0, _MockDropoutKeepProb(0.0)),
    })
    enriched = _enrich_mlx_adapter_config(model, {})
    assert "unsloth_mlx_lora_module_paths" not in enriched


def test_collect_lora_helper_finds_adapters_after_reload():
    # After a reload/freeze, LoRA tensors live in parameters() but are
    # not always listed in trainable_parameters(). The module-anchored
    # helper must still find them so MLXTrainer.save_model routes to
    # the adapter exporter (not save_merged_model).
    from unsloth_zoo.mlx.utils import collect_mlx_lora_adapter_tensors

    real = _MockLoRALinear(8, 16, 4, 1.0, _MockDropoutKeepProb(0.0))
    plain = _MockPlainLinear(16, 32)

    class _ReloadedModel:
        def __init__(self):
            self.q_proj = real
            self.up_proj = plain

        def parameters(self):
            return {
                "q_proj.weight": real.weight,
                "q_proj.lora_a": real.lora_a,
                "q_proj.lora_b": real.lora_b,
                "up_proj.weight": plain.weight,
            }

        def trainable_parameters(self):
            # mimic the post-reload state where adapter tensors are not
            # explicitly marked trainable.
            return {"up_proj.weight": plain.weight}

        def named_modules(self):
            yield "", self
            yield "q_proj", real
            yield "up_proj", plain

    found = collect_mlx_lora_adapter_tensors(_ReloadedModel())
    assert set(found.keys()) == {"q_proj.lora_a", "q_proj.lora_b"}, found


def test_ensure_lora_frozen_freezes_norm_when_lora_is_actively_trained():
    # Active LoRA training with an accidentally trainable norm. The
    # module-anchored detector finds q_proj.lora_a/lora_b in the
    # trainable map, so the NaN safeguard runs and freezes norm.weight.
    from unsloth_zoo.mlx.trainer import MLXTrainer

    lora = _MockLoRALinear(8, 16, 4, 1.0, _MockDropoutKeepProb(0.0))

    freeze_calls = []

    class _NormStub:
        def __init__(self):
            self.weight = torch.zeros(16)
        def freeze(self, keys=None, recurse=False):
            freeze_calls.append((list(keys or []), recurse))

    norm_stub = _NormStub()

    class _ActiveLora:
        def __init__(self):
            self.q_proj = lora
            self.norm = norm_stub
        def parameters(self):
            return {
                "q_proj.weight": lora.weight,
                "q_proj.lora_a": lora.lora_a,
                "q_proj.lora_b": lora.lora_b,
                "norm.weight": norm_stub.weight,
            }
        def trainable_parameters(self):
            return {
                "q_proj.lora_a": lora.lora_a,
                "q_proj.lora_b": lora.lora_b,
                "norm.weight": norm_stub.weight,
            }
        def named_modules(self):
            yield "", self
            yield "q_proj", lora
            yield "norm", norm_stub

    MLXTrainer._ensure_lora_frozen(_ActiveLora())
    assert freeze_calls == [(["weight"], False)], freeze_calls


def test_ensure_lora_frozen_skips_when_lora_tensors_present_but_not_trainable():
    # Reloaded model where adapter tensors live in parameters() but none
    # are trainable; the user is doing a non-LoRA fine-tune (norm-only)
    # so the safeguard must NOT freeze the intentionally trainable norm.
    from unsloth_zoo.mlx.trainer import MLXTrainer

    lora = _MockLoRALinear(8, 16, 4, 1.0, _MockDropoutKeepProb(0.0))

    freeze_calls = []

    class _NormStub:
        def __init__(self):
            self.weight = torch.zeros(16)
        def freeze(self, keys=None, recurse=False):
            freeze_calls.append((list(keys or []), recurse))

    norm_stub = _NormStub()

    class _ReloadedFrozenLora:
        def __init__(self):
            self.q_proj = lora
            self.norm = norm_stub
        def parameters(self):
            return {
                "q_proj.weight": lora.weight,
                "q_proj.lora_a": lora.lora_a,
                "q_proj.lora_b": lora.lora_b,
                "norm.weight": norm_stub.weight,
            }
        def trainable_parameters(self):
            return {"norm.weight": norm_stub.weight}
        def named_modules(self):
            yield "", self
            yield "q_proj", lora
            yield "norm", norm_stub

    MLXTrainer._ensure_lora_frozen(_ReloadedFrozenLora())
    assert freeze_calls == [], freeze_calls


def test_ensure_lora_frozen_skips_when_no_lora_modules_present():
    from unsloth_zoo.mlx.trainer import MLXTrainer

    freeze_calls = []

    class _NormStub:
        def __init__(self):
            self.weight = torch.zeros(16)
        def freeze(self, keys=None, recurse=False):
            freeze_calls.append((list(keys or []), recurse))

    norm_stub = _NormStub()

    class _NoLora:
        def __init__(self):
            self.norm = norm_stub
        def parameters(self):
            return {"norm.weight": norm_stub.weight}
        def trainable_parameters(self):
            return {"norm.weight": norm_stub.weight}
        def named_modules(self):
            yield "", self
            yield "norm", norm_stub

    MLXTrainer._ensure_lora_frozen(_NoLora())
    assert freeze_calls == [], freeze_calls


def test_save_trainable_adapters_omits_lora_metadata_for_full_checkpoint(tmp_path):
    # A full fine-tune (no LoRA modules) must not stamp fine_tune_type='lora'
    # or default lora_parameters on its adapter_config.json; mlx-lm would
    # otherwise inject LoRA wrappers before binding the saved full weights.
    from unsloth_zoo.mlx.utils import save_trainable_adapters
    import json

    plain = _MockPlainLinear(16, 32)
    model = _make_model({"dense": plain})

    save_trainable_adapters(model, tmp_path)
    with open(tmp_path / "adapter_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert "lora_parameters" not in cfg, cfg
    assert "rank" not in cfg, cfg
    assert "num_layers" not in cfg, cfg
    assert cfg.get("fine_tune_type") != "lora", cfg
    assert cfg.get("peft_type") != "LORA", cfg


def test_save_pretrained_merged_lora_method_raises_when_no_adapter_tensors(tmp_path):
    # The new outer gate uses collect_mlx_lora_adapter_tensors, so the
    # "no LoRA layers" ValueError is raised at the gate with the
    # user-facing message rather than slipping past hasattr(m, "fuse")
    # and bubbling the lower-level "no MLX LoRA adapter tensors" error.
    from unsloth_zoo.mlx.utils import save_pretrained_merged

    plain = _MockPlainLinear(16, 32)

    class _NoLora:
        def __init__(self):
            self.up_proj = plain
            self._hf_repo = None
            self._config = None
            self._unsloth_quantization_config = None
            self._unsloth_quantization_policy = None
            self._unsloth_quantized_source = None
            self._unsloth_base_revision = None
            self._unsloth_base_commit_hash = None
            self._src_path = None
        def parameters(self):
            return {"up_proj.weight": plain.weight}
        def trainable_parameters(self):
            return self.parameters()
        def named_modules(self):
            yield "", self
            yield "up_proj", plain

    class _Tok:
        def save_pretrained(self, *_a, **_k):
            pass

    with pytest.raises(ValueError, match="no LoRA layers"):
        save_pretrained_merged(_NoLora(), _Tok(), tmp_path, save_method="lora")


def test_save_pretrained_merged_lora_strips_accidental_trainable_base_tensors(tmp_path):
    # After a reload, base weights such as q_proj.weight may end up
    # marked trainable. save_method="lora" must still ship a lean
    # adapter file rather than leaking those base tensors.
    from unsloth_zoo.mlx.utils import save_pretrained_merged

    lora = _MockLoRALinear(8, 16, 4, 1.0, _MockDropoutKeepProb(0.0))

    class _ReloadedModel:
        def __init__(self):
            self.q_proj = lora
            self._hf_repo = None
            self._config = None
            self._unsloth_quantization_config = None
            self._unsloth_quantization_policy = None
            self._unsloth_quantized_source = None
            self._unsloth_base_revision = None
            self._unsloth_base_commit_hash = None
            self._src_path = None
        def parameters(self):
            return {
                "q_proj.weight": lora.weight,
                "q_proj.lora_a": lora.lora_a,
                "q_proj.lora_b": lora.lora_b,
            }
        def trainable_parameters(self):
            return self.parameters()
        def named_modules(self):
            yield "", self
            yield "q_proj", lora

    class _Tok:
        def save_pretrained(self, *_a, **_k):
            pass

    save_pretrained_merged(_ReloadedModel(), _Tok(), tmp_path, save_method="lora")

    from safetensors.torch import load_file
    keys = set(load_file(str(tmp_path / "adapters.safetensors")).keys())
    assert keys == {"q_proj.lora_a", "q_proj.lora_b"}, sorted(keys)


def test_save_pretrained_merged_merged_methods_skip_lora_collection(tmp_path, monkeypatch):
    # Merged exports (merged_16bit / merged_4bit) must not call the
    # adapter collector; the lora-only presence check belongs inside
    # the method == "lora" branch.
    from unsloth_zoo.mlx import utils as mlx_utils

    collect_calls = []
    merged_calls = []

    def _spy_collect(model):
        collect_calls.append(model)
        return {}

    def _stub_save_merged(model, tokenizer, path, dequantize=False):
        merged_calls.append((path, dequantize))

    monkeypatch.setattr(mlx_utils, "collect_mlx_lora_adapter_tensors", _spy_collect)
    monkeypatch.setattr(mlx_utils, "save_merged_model", _stub_save_merged)

    class _Model:
        def parameters(self):
            return {}
        def named_modules(self):
            yield "", self

    class _Tok:
        def save_pretrained(self, *_a, **_k):
            pass

    mlx_utils.save_pretrained_merged(_Model(), _Tok(), tmp_path, save_method="merged_16bit")
    mlx_utils.save_pretrained_merged(_Model(), _Tok(), tmp_path, save_method="merged_4bit")

    assert collect_calls == [], collect_calls
    assert len(merged_calls) == 2
    assert merged_calls[0][1] is True   # merged_16bit dequantizes
    assert merged_calls[1][1] is False  # merged_4bit keeps quantization


def test_save_trainable_adapters_raises_when_no_trainable_params(tmp_path):
    # A fully frozen model has no trainable tensors, so writing a checkpoint
    # would leave adapter_config.json next to a missing adapters.safetensors;
    # the loader cannot consume that. The exporter must surface the empty
    # case explicitly the same way save_lora_adapters() does.
    from unsloth_zoo.mlx.utils import save_trainable_adapters

    class _FrozenModel:
        def __init__(self):
            self._hf_repo = None
            self._config = None
            self._unsloth_quantization_config = None
            self._unsloth_quantization_policy = None
            self._unsloth_quantized_source = None
            self._unsloth_base_revision = None
            self._unsloth_base_commit_hash = None
            self._src_path = None
        def parameters(self):
            return {}
        def trainable_parameters(self):
            return {}
        def named_modules(self):
            yield "", self

    with pytest.raises(ValueError, match="no trainable or LoRA parameters"):
        save_trainable_adapters(_FrozenModel(), tmp_path)
    assert not (tmp_path / "adapters.safetensors").exists()
    assert not (tmp_path / "adapter_config.json").exists()


def test_ensure_lora_frozen_freezes_norm_whose_name_contains_lora_substring():
    # Anchor-on-modules detection lets the norm safeguard flag accidentally
    # trainable norms even when the parameter path happens to contain the
    # literal "lora" substring (e.g. a norm sitting inside a module named
    # `lora_router`). The previous "lora" not in k check let those through.
    from unsloth_zoo.mlx.trainer import MLXTrainer

    lora = _MockLoRALinear(8, 16, 4, 1.0, _MockDropoutKeepProb(0.0))

    freeze_calls = []

    class _NormStub:
        def __init__(self):
            self.weight = torch.zeros(16)
        def freeze(self, keys=None, recurse=False):
            freeze_calls.append((list(keys or []), recurse))

    fake_norm = _NormStub()

    class _ModelWithLoraNamedNorm:
        def __init__(self):
            self.q_proj = lora
            self.lora_router_norm = fake_norm
        def parameters(self):
            return {
                "q_proj.weight": lora.weight,
                "q_proj.lora_a": lora.lora_a,
                "q_proj.lora_b": lora.lora_b,
                "lora_router_norm.weight": fake_norm.weight,
            }
        def trainable_parameters(self):
            return {
                "q_proj.lora_a": lora.lora_a,
                "q_proj.lora_b": lora.lora_b,
                "lora_router_norm.weight": fake_norm.weight,
            }
        def named_modules(self):
            yield "", self
            yield "q_proj", lora
            yield "lora_router_norm", fake_norm

    MLXTrainer._ensure_lora_frozen(_ModelWithLoraNamedNorm())
    assert freeze_calls == [(["weight"], False)], freeze_calls


def test_save_pretrained_merged_lora_method_preserves_external_trainables(
    tmp_path, monkeypatch
):
    # When the user intentionally trains non-LoRA tensors OUTSIDE a
    # LoRA-wrapped module (embed_tokens, lm_head, projector, vision,
    # norm), save_method='lora' must preserve them via the trainable
    # writer. Base weights INSIDE a LoRA module are still excluded so
    # accidentally-trainable q_proj.weight under a wrapped q_proj does
    # not leak (covered by the reload regression test above).
    from unsloth_zoo.mlx import utils as mlx_utils

    lora = _MockLoRALinear(8, 16, 4, 1.0, _MockDropoutKeepProb(0.0))
    embed = torch.zeros(32, 16)

    class _MixedModel:
        def __init__(self):
            self.q_proj = lora
            self._hf_repo = None
            self._config = None
            self._unsloth_quantization_config = None
            self._unsloth_quantization_policy = None
            self._unsloth_quantized_source = None
            self._unsloth_base_revision = None
            self._unsloth_base_commit_hash = None
            self._src_path = None
        def parameters(self):
            return {
                "q_proj.weight": lora.weight,
                "q_proj.lora_a": lora.lora_a,
                "q_proj.lora_b": lora.lora_b,
                "embed_tokens.weight": embed,
            }
        def trainable_parameters(self):
            return {
                "q_proj.lora_a": lora.lora_a,
                "q_proj.lora_b": lora.lora_b,
                "embed_tokens.weight": embed,
            }
        def named_modules(self):
            yield "", self
            yield "q_proj", lora

    class _Tok:
        def save_pretrained(self, *_a, **_k):
            pass

    routed = []

    def _spy_lora(model, path, adapter_config=None):
        routed.append("lora")
    def _spy_trainable(model, path, adapter_config=None):
        routed.append("trainable")

    monkeypatch.setattr(mlx_utils, "save_lora_adapters", _spy_lora)
    monkeypatch.setattr(mlx_utils, "save_trainable_adapters", _spy_trainable)

    mlx_utils.save_pretrained_merged(
        _MixedModel(), _Tok(), tmp_path, save_method="lora",
    )
    assert routed == ["trainable"], routed


def test_save_pretrained_merged_lora_method_pure_lora_uses_lean_writer(
    tmp_path, monkeypatch
):
    # When no non-LoRA tensor is trainable, the public API keeps the lean
    # adapter-only artifact (no over-eager routing to the trainable writer).
    from unsloth_zoo.mlx import utils as mlx_utils

    lora = _MockLoRALinear(8, 16, 4, 1.0, _MockDropoutKeepProb(0.0))

    class _PureLoraModel:
        def __init__(self):
            self.q_proj = lora
            self._hf_repo = None
            self._config = None
            self._unsloth_quantization_config = None
            self._unsloth_quantization_policy = None
            self._unsloth_quantized_source = None
            self._unsloth_base_revision = None
            self._unsloth_base_commit_hash = None
            self._src_path = None
        def parameters(self):
            return {
                "q_proj.weight": lora.weight,
                "q_proj.lora_a": lora.lora_a,
                "q_proj.lora_b": lora.lora_b,
            }
        def trainable_parameters(self):
            return {
                "q_proj.lora_a": lora.lora_a,
                "q_proj.lora_b": lora.lora_b,
            }
        def named_modules(self):
            yield "", self
            yield "q_proj", lora

    class _Tok:
        def save_pretrained(self, *_a, **_k):
            pass

    routed = []
    monkeypatch.setattr(
        mlx_utils, "save_lora_adapters",
        lambda model, path, adapter_config=None: routed.append("lora"),
    )
    monkeypatch.setattr(
        mlx_utils, "save_trainable_adapters",
        lambda model, path, adapter_config=None: routed.append("trainable"),
    )

    mlx_utils.save_pretrained_merged(
        _PureLoraModel(), _Tok(), tmp_path, save_method="lora",
    )
    assert routed == ["lora"], routed


def test_save_trainable_adapters_preserves_frozen_lora_alongside_trainable_norm(
    tmp_path,
):
    # After a checkpoint reload + norm-only fine-tune, the LoRA pair lives
    # in parameters() but is absent from trainable_parameters(). The
    # checkpoint writer must still emit the LoRA tensors so the saved
    # artifact remains a valid, reloadable adapter file.
    from unsloth_zoo.mlx.utils import save_trainable_adapters

    lora = _MockLoRALinear(8, 16, 4, 1.0, _MockDropoutKeepProb(0.0))
    norm_w = torch.zeros(16)

    class _FrozenLoraTrainableNorm:
        def __init__(self):
            self.q_proj = lora
            self.norm = self
            self.weight = norm_w
            self._hf_repo = None
            self._config = None
            self._unsloth_quantization_config = None
            self._unsloth_quantization_policy = None
            self._unsloth_quantized_source = None
            self._unsloth_base_revision = None
            self._unsloth_base_commit_hash = None
            self._src_path = None
        def parameters(self):
            return {
                "q_proj.weight": lora.weight,
                "q_proj.lora_a": lora.lora_a,
                "q_proj.lora_b": lora.lora_b,
                "norm.weight": norm_w,
            }
        def trainable_parameters(self):
            return {"norm.weight": norm_w}
        def named_modules(self):
            yield "", self
            yield "q_proj", lora

    save_trainable_adapters(_FrozenLoraTrainableNorm(), tmp_path)

    from safetensors.torch import load_file
    keys = set(load_file(str(tmp_path / "adapters.safetensors")).keys())
    assert keys == {"q_proj.lora_a", "q_proj.lora_b", "norm.weight"}, sorted(keys)


def test_save_pretrained_merged_lora_writes_complete_adapter_config(tmp_path):
    # adapter_config.json shipped from save_pretrained_merged(save_method='lora')
    # must include lora_parameters with rank/scale/dropout so mlx-lm
    # load_adapters can recreate the wrappers; previously these were absent
    # whenever the caller did not pass an explicit adapter_config.
    import json
    from unsloth_zoo.mlx.utils import save_pretrained_merged

    lora = _MockLoRALinear(8, 16, 4, 2.5, _MockDropoutKeepProb(0.1))

    class _PureLoraModel:
        def __init__(self):
            self.q_proj = lora
            self._hf_repo = None
            self._config = None
            self._unsloth_quantization_config = None
            self._unsloth_quantization_policy = None
            self._unsloth_quantized_source = None
            self._unsloth_base_revision = None
            self._unsloth_base_commit_hash = None
            self._src_path = None
        def parameters(self):
            return {
                "q_proj.weight": lora.weight,
                "q_proj.lora_a": lora.lora_a,
                "q_proj.lora_b": lora.lora_b,
            }
        def trainable_parameters(self):
            return {
                "q_proj.lora_a": lora.lora_a,
                "q_proj.lora_b": lora.lora_b,
            }
        def named_modules(self):
            yield "", self
            yield "q_proj", lora

    class _Tok:
        def save_pretrained(self, *_a, **_k):
            pass

    save_pretrained_merged(_PureLoraModel(), _Tok(), tmp_path, save_method="lora")
    cfg = json.loads((tmp_path / "adapter_config.json").read_text())
    assert "lora_parameters" in cfg, cfg
    assert cfg["lora_parameters"]["rank"] == 4, cfg
    assert cfg["lora_parameters"]["scale"] == 2.5, cfg

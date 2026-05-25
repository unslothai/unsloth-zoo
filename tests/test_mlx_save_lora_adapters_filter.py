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


def test_save_trainable_adapters_preserves_external_trainables_and_lora(tmp_path):
    # External trainables (up_proj.weight, outside any LoRA module) survive
    # alongside the LoRA tensors. A reload-leaked base weight INSIDE a LoRA
    # module (q_proj.weight under wrapped q_proj) is dropped so the adapter
    # file stays loadable as an adapter rather than a partial base dump.
    from unsloth_zoo.mlx.utils import save_trainable_adapters

    model = _make_model({
        "q_proj": _MockLoRALinear(8, 16, 4, 1.0, _MockDropoutKeepProb(0.0)),
        "up_proj": _MockPlainLinear(16, 32),
    })
    save_trainable_adapters(model, tmp_path)

    from safetensors.torch import load_file
    keys = set(load_file(str(tmp_path / "adapters.safetensors")).keys())
    assert keys == {
        "q_proj.lora_a", "q_proj.lora_b", "up_proj.weight",
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


def test_save_pretrained_merged_lora_mixed_external_drops_inside_lora_base(tmp_path):
    # When the user trains intentional non-LoRA tensors OUTSIDE a LoRA
    # module (embed_tokens) AND a reload-leaked base weight INSIDE a LoRA
    # module is also marked trainable (q_proj.weight under wrapped q_proj),
    # the public save_method='lora' path must preserve the external
    # trainable WITHOUT leaking the inside-LoRA base weight.
    import torch
    from unsloth_zoo.mlx.utils import save_pretrained_merged

    lora = _MockLoRALinear(8, 16, 4, 1.0, _MockDropoutKeepProb(0.0))
    embed = torch.zeros(32, 16)

    class _MixedReloadedModel:
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
            # mimic reload: base weight inside the LoRA module is unfrozen
            # AND user intentionally trains the embedding.
            return self.parameters()
        def named_modules(self):
            yield "", self
            yield "q_proj", lora

    class _Tok:
        def save_pretrained(self, *_a, **_k):
            pass

    save_pretrained_merged(_MixedReloadedModel(), _Tok(), tmp_path, save_method="lora")

    from safetensors.torch import load_file
    keys = set(load_file(str(tmp_path / "adapters.safetensors")).keys())
    assert keys == {
        "q_proj.lora_a", "q_proj.lora_b", "embed_tokens.weight",
    }, sorted(keys)


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


def test_push_lora_adapters_routes_commit_metadata_through_upload_folder(
    tmp_path, monkeypatch,
):
    # Regression: huggingface_hub>=0.34's upload_large_folder() does NOT
    # accept commit_message / commit_description / create_pr / revision in
    # any way that lands them on the commit. Routing LoRA pushes through
    # upload_folder (which honors those kwargs) is the only way the
    # caller's commit string and PR flag survive to the Hub.
    import huggingface_hub
    from unsloth_zoo.mlx.utils import _push_lora_adapters_to_hub

    (tmp_path / "adapters.safetensors").write_bytes(b"\x00")
    (tmp_path / "adapter_config.json").write_text("{}")

    calls = {"folder": [], "large": []}

    class _FakeApi:
        def __init__(self, token=None):
            self.token = token

        def create_repo(self, **kwargs):
            return None

        def update_repo_settings(self, **kwargs):
            return None

        def upload_folder(self, **kwargs):
            calls["folder"].append(kwargs)

        def upload_large_folder(self, **kwargs):
            calls["large"].append(kwargs)

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    _push_lora_adapters_to_hub(
        tmp_path,
        repo_id="me/adapter",
        token="hf_dummy",
        commit_message="Release v1",
        commit_description="Custom desc",
        create_pr=True,
        revision="main",
    )

    # upload_folder must be the primary path, with all caller kwargs
    # threaded through; upload_large_folder must not be hit in the happy
    # case (it would commit with the default "Upload N LFS files" string).
    assert len(calls["folder"]) == 1, calls
    assert calls["large"] == [], calls
    sent = calls["folder"][0]
    # Commit message gets a " (Trained with Unsloth)" suffix per Unsloth
    # commit-history convention; substring check tolerates the suffix.
    assert "Release v1" in sent["commit_message"], sent
    assert "Custom desc" in sent["commit_description"], sent
    assert sent["create_pr"] is True
    assert sent["revision"] == "main"
    assert sent["repo_id"] == "me/adapter"


def test_save_trainable_adapters_drops_quantized_base_scales_biases(tmp_path):
    # Reload-leaked state on mlx-lm QuantizedLinear includes .scales and
    # .biases alongside .weight; the filter must drop ALL three under
    # LoRA prefixes so a QLoRA reload-trainable layer's quantization
    # tensors don't slip into the adapter save.
    from unsloth_zoo.mlx.utils import save_trainable_adapters
    import torch as _t
    from safetensors.torch import load_file

    class _QLoRAModule:
        def __init__(self):
            self.lora_a = _t.zeros(4, 8)
            self.lora_b = _t.zeros(8, 4)
            self.weight = _t.zeros(8, 8)
            self.scales = _t.zeros(8)
            self.biases = _t.zeros(8)
            self.bias = _t.zeros(8)  # legitimate Linear bias (preserved)

    class _Model:
        def __init__(self):
            self._lora = _QLoRAModule()
        def named_modules(self):
            yield "q_proj", self._lora
        def parameters(self):
            return {"q_proj": {
                "lora_a": self._lora.lora_a,
                "lora_b": self._lora.lora_b,
                "weight": self._lora.weight,
                "scales": self._lora.scales,
                "biases": self._lora.biases,
                "bias": self._lora.bias,
            }}
        def trainable_parameters(self):
            return self.parameters()

    out = tmp_path / "qlora_ckpt"
    save_trainable_adapters(_Model(), out)
    keys = set(load_file(out / "adapters.safetensors").keys())

    assert "q_proj.lora_a" in keys
    assert "q_proj.lora_b" in keys
    assert "q_proj.weight" not in keys
    assert "q_proj.scales" not in keys
    assert "q_proj.biases" not in keys
    # Singular `.bias` is a legitimate Linear bias, not quantization state.
    assert "q_proj.bias" in keys


def test_enrich_overrides_caller_full_when_lora_modules_present(tmp_path):
    # If the caller passes adapter_config={"fine_tune_type": "full"}
    # but the model has LoRA modules, the saved tensors are *.lora_a /
    # *.lora_b. mlx-lm reload would skip LoRA wrapping because the
    # config says "full", silently dropping the adapter. Override to
    # "lora" so the artifact stays consistent.
    from unsloth_zoo.mlx.utils import _enrich_mlx_adapter_config
    import torch as _t

    class _LoRAModule:
        def __init__(self):
            self.lora_a = _t.zeros(4, 8)
            self.lora_b = _t.zeros(8, 4)

    class _Model:
        def __init__(self):
            self._lora = _LoRAModule()
        def named_modules(self):
            yield "q_proj", self._lora
        def parameters(self):
            return {"q_proj": {
                "lora_a": self._lora.lora_a,
                "lora_b": self._lora.lora_b,
            }}

    cfg = _enrich_mlx_adapter_config(_Model(), {"fine_tune_type": "full"})
    assert cfg["fine_tune_type"] == "lora", cfg


def test_enrich_strips_stale_lora_fields_on_full_finetune():
    # A no-LoRA save reusing a config dict from a prior LoRA save would
    # otherwise carry stale peft_type / lora_parameters / rank etc.
    # alongside the new fine_tune_type=full stamp.
    from unsloth_zoo.mlx.utils import _enrich_mlx_adapter_config

    class _Model:
        def named_modules(self):
            return iter(())
        def parameters(self):
            return {}

    stale_cfg = {
        "peft_type": "LORA",
        "lora_parameters": {"rank": 8, "scale": 1.0, "dropout": 0.0},
        "rank": 8,
        "scale": 1.0,
        "dropout": 0.0,
        "num_layers": 24,
        "unsloth_mlx_lora_module_paths": ["layers.0.q_proj"],
    }
    cfg = _enrich_mlx_adapter_config(_Model(), stale_cfg)
    assert cfg.get("fine_tune_type") == "full"
    for stale_key in (
        "peft_type", "lora_parameters", "rank", "scale", "dropout",
        "num_layers", "unsloth_mlx_lora_module_paths",
    ):
        assert stale_key not in cfg, f"stale {stale_key} should be stripped: {cfg}"


def test_push_lora_adapters_create_pr_failure_raises_instead_of_silent_main(
    tmp_path, monkeypatch,
):
    # When upload_folder raises on a create_pr=True request, falling
    # back to upload_large_folder would silently push to main. Refuse
    # to do that and raise a clear RuntimeError instead.
    import pytest as _pytest
    import huggingface_hub
    from unsloth_zoo.mlx.utils import _push_lora_adapters_to_hub

    (tmp_path / "adapters.safetensors").write_bytes(b"\x00")

    class _FakeApi:
        def __init__(self, token=None):
            pass
        def create_repo(self, **kwargs):
            return None
        def update_repo_settings(self, **kwargs):
            return None
        def upload_folder(self, **kwargs):
            raise TypeError("simulated old huggingface_hub signature")
        def upload_large_folder(self, **kwargs):
            raise AssertionError("must not silently push to main on create_pr=True")

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    with _pytest.raises(RuntimeError, match="create_pr=True"):
        _push_lora_adapters_to_hub(tmp_path, repo_id="me/adapter", create_pr=True)


def test_save_trainable_adapters_preserves_bias_under_lora_wrapped_linear(tmp_path):
    # Refinement of the lora_module_prefixes filter: only the wrapped
    # base `.weight` is a reload-leak risk (the V x H matmul gradient).
    # Other trainable params under the LoRA module (e.g. q_proj.bias
    # when bias=True is explicitly trained) must survive the filter so
    # the user's bias training isn't silently dropped on checkpoint.
    from unsloth_zoo.mlx.utils import save_trainable_adapters
    import torch as _t
    from safetensors.torch import load_file

    class _LoRALinearWithBias:
        # Path: q_proj has lora_a / lora_b plus the wrapped base
        # Linear's weight + bias (flattened to q_proj.weight, q_proj.bias).
        def __init__(self):
            self.lora_a = _t.zeros(4, 8)
            self.lora_b = _t.zeros(8, 4)
            self.weight = _t.zeros(8, 8)
            self.bias = _t.zeros(8)

    class _Model:
        def __init__(self):
            self._lora = _LoRALinearWithBias()
            self._up = type("UpProj", (), {"weight": _t.zeros(8, 8)})()
        def named_modules(self):
            yield "q_proj", self._lora
            yield "up_proj", self._up
        def parameters(self):
            return {
                "q_proj": {
                    "lora_a": self._lora.lora_a,
                    "lora_b": self._lora.lora_b,
                    "weight": self._lora.weight,
                    "bias": self._lora.bias,
                },
                "up_proj": {"weight": self._up.weight},
            }
        def trainable_parameters(self):
            return self.parameters()

    out = tmp_path / "ckpt"
    save_trainable_adapters(_Model(), out)
    keys = set(load_file(out / "adapters.safetensors").keys())

    # LoRA tensors kept; base weight inside LoRA dropped; bias kept;
    # external weight kept.
    assert "q_proj.lora_a" in keys
    assert "q_proj.lora_b" in keys
    assert "q_proj.weight" not in keys
    assert "q_proj.bias" in keys, "trainable .bias under LoRA module must not be dropped"
    assert "up_proj.weight" in keys


def test_enrich_stamps_fine_tune_type_full_when_no_lora_modules(tmp_path):
    # Full-finetune / no-LoRA checkpoints must NOT default to lora on
    # reload. mlx-lm's load_adapters() defaults missing fine_tune_type
    # to "lora" and then reads num_layers / lora_parameters, so a no-LoRA
    # save with no fine_tune_type stamp fails to reload as a full model.
    from unsloth_zoo.mlx.utils import _enrich_mlx_adapter_config

    class _Model:
        def named_modules(self):
            return iter(())  # no LoRA modules
        def parameters(self):
            return {}

    cfg = _enrich_mlx_adapter_config(_Model(), {})
    assert cfg.get("fine_tune_type") == "full", cfg
    # Should NOT have peft_type=LORA on a full-finetune save.
    assert cfg.get("peft_type") is None or cfg.get("peft_type") != "LORA", cfg


def test_enrich_stamps_fine_tune_type_dora_when_dora_modules_present():
    # mlx-lm's linear_to_lora_layers(..., use_dora=(fine_tune_type=="dora"))
    # only recreates DoRA wrappers when the config says "dora". Without
    # the right stamp, the saved q_proj.m magnitude tensor cannot bind
    # via DoRALinear on reload, dropping the learned DoRA magnitudes.
    from unsloth_zoo.mlx.utils import _enrich_mlx_adapter_config
    import torch as _t

    class DoRALinear:
        def __init__(self):
            self.lora_a = _t.zeros(8, 4)
            self.lora_b = _t.zeros(4, 8)
            self.m = _t.zeros(8)
        # mlx-lm names this attr `m`; the gate on type(module).__name__
        # is what triggers the dora stamp.

    class _DoRAModel:
        def __init__(self):
            self._mod = DoRALinear()
        def named_modules(self):
            yield "q_proj", self._mod
        def parameters(self):
            return {
                "q_proj": {
                    "lora_a": self._mod.lora_a,
                    "lora_b": self._mod.lora_b,
                    "m": self._mod.m,
                }
            }

    cfg = _enrich_mlx_adapter_config(_DoRAModel(), {})
    assert cfg.get("fine_tune_type") == "dora", cfg


def test_is_lm_head_trainable_skips_base_weight_under_lora_wrapped_lm_head():
    # After reload, mlx-lm wrappers may leak the inner base .weight as
    # trainable. For a LoRA-wrapped lm_head the leaked lm_head.weight is
    # not real user intent; treating it as trainable defeats the CCE
    # memory guard and forces a V x H weight gradient per chunk.
    from unsloth_zoo.mlx.utils import _is_lm_head_trainable
    import torch as _t

    class _LoRAlmHead:
        def __init__(self):
            self.lora_a = _t.zeros(4, 1024)
            self.lora_b = _t.zeros(1024, 4)
            self.weight = _t.zeros(1024, 1024)  # leaked base

    class _Model:
        def __init__(self):
            self._lm = _LoRAlmHead()
        def trainable_parameters(self):
            return {
                "lm_head": {
                    "lora_a": self._lm.lora_a,
                    "lora_b": self._lm.lora_b,
                    "weight": self._lm.weight,
                }
            }
        def parameters(self):
            return self.trainable_parameters()
        def named_modules(self):
            yield "lm_head", self._lm

    # lm_head.weight under a LoRA-wrapped lm_head must be filtered out;
    # the trainable check should return False (LoRA-only training).
    assert _is_lm_head_trainable(_Model()) is False


def test_push_lora_adapters_uses_allow_patterns_to_avoid_stale_uploads(
    tmp_path, monkeypatch,
):
    # If the save_directory already contains stale full-model files
    # (e.g. from a prior save_pretrained_merged(save_method='merged_16bit')
    # run), the LoRA push must NOT upload them. Public-by-default repos
    # would otherwise expose merged weights under a "LoRA adapter" repo.
    import huggingface_hub
    from unsloth_zoo.mlx.utils import _push_lora_adapters_to_hub

    (tmp_path / "adapters.safetensors").write_bytes(b"\x00")
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"stale-full-model")
    (tmp_path / "tokenizer.json").write_text("{}")

    calls = {"folder": []}

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kwargs):
            return None

        def update_repo_settings(self, **kwargs):
            return None

        def upload_folder(self, **kwargs):
            calls["folder"].append(kwargs)

        def upload_large_folder(self, **kwargs):
            pass

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    _push_lora_adapters_to_hub(tmp_path, repo_id="me/adapter")

    assert len(calls["folder"]) == 1, calls
    sent = calls["folder"][0]
    assert "allow_patterns" in sent, sent
    patterns = sent["allow_patterns"]
    assert "adapters.safetensors" in patterns
    assert "adapter_config.json" in patterns
    # No catch-all that would re-include the stale merged-model shard.
    assert "*.safetensors" not in patterns
    assert "*" not in patterns


def test_collect_lora_includes_m_on_real_dora_class():
    # Positive twin of the DoRA-gate negative test: if a module's class
    # name starts with "DoRA" (matches mlx-lm's DoRALinear / DoRAEmbedding),
    # collect_mlx_lora_adapter_tensors MUST include the m magnitude
    # tensor. A future typo (e.g. startswith("DORA"), wrong attr name)
    # would silently drop DoRA magnitudes from every export.
    from unsloth_zoo.mlx.utils import collect_mlx_lora_adapter_tensors
    import torch as _t

    class DoRALinear:
        def __init__(self):
            self.lora_a = _t.zeros(8, 4)
            self.lora_b = _t.zeros(4, 8)
            self.m = _t.zeros(8)

    class _Model:
        def __init__(self):
            self._mod = DoRALinear()
        def parameters(self):
            return {
                "q_proj": {
                    "lora_a": self._mod.lora_a,
                    "lora_b": self._mod.lora_b,
                    "m": self._mod.m,
                }
            }
        def named_modules(self):
            yield "q_proj", self._mod

    tensors = collect_mlx_lora_adapter_tensors(_Model())
    assert "q_proj.lora_a" in tensors
    assert "q_proj.lora_b" in tensors
    assert "q_proj.m" in tensors, "DoRA magnitude tensor must be exported"


def test_collect_lora_skips_unrelated_m_attribute_on_non_dora_module():
    # `m` is a generic 1-letter attr name; if a future LoRA wrapper
    # exposes self.m as a learned mixing scalar that is not a DoRA
    # magnitude vector, we must not ship it under DoRA semantics.
    # Gate on the class name starting with "DoRA".
    from unsloth_zoo.mlx.utils import collect_mlx_lora_adapter_tensors
    import torch as _t

    class _MockLoRAWithUnrelatedM:
        def __init__(self):
            self.lora_a = _t.zeros(8, 4)
            self.lora_b = _t.zeros(4, 8)
            self.m = _t.zeros(1)  # unrelated; not a DoRA magnitude

    class _Model:
        def parameters(self):
            mod = _MockLoRAWithUnrelatedM()
            return {
                "q_proj": {
                    "lora_a": mod.lora_a,
                    "lora_b": mod.lora_b,
                    "m": mod.m,
                }
            }
        def named_modules(self):
            yield "q_proj", _MockLoRAWithUnrelatedM()

    tensors = collect_mlx_lora_adapter_tensors(_Model())
    # lora_a / lora_b must be collected; `m` must not (non-DoRA class).
    assert "q_proj.lora_a" in tensors
    assert "q_proj.lora_b" in tensors
    assert "q_proj.m" not in tensors


def test_save_adapter_artifacts_rejects_empty_tensors():
    # Defensive: any future direct caller of the private helper with
    # tensors={} must hit a clear error, not silently write an
    # adapter_config.json without weights next to it.
    import pytest as _pytest
    from unsloth_zoo.mlx.utils import _save_adapter_artifacts

    class _Model:
        def parameters(self):
            return {}
        def named_modules(self):
            return iter(())

    with _pytest.raises(ValueError, match="non-empty"):
        _save_adapter_artifacts(_Model(), "/tmp/zzz_unsloth_test_empty", tensors={})


def test_push_to_hub_merged_honors_create_pr_via_upload_folder(
    tmp_path, monkeypatch,
):
    # Regression for the same upload_large_folder kwarg-drop defect on the
    # merged-save path: with create_pr=True or a custom commit_message,
    # the merged push must route through upload_folder so the kwargs land.
    import huggingface_hub
    from unsloth_zoo.mlx.utils import push_to_hub_merged

    # Pretend the model was already saved so push_to_hub_merged skips the
    # merge step and only runs the upload path under test.
    (tmp_path / "model.safetensors.index.json").write_text("{}")

    calls = {"folder": [], "large": []}

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kwargs):
            return None

        def update_repo_settings(self, **kwargs):
            return None

        def upload_folder(self, **kwargs):
            calls["folder"].append(kwargs)

        def upload_large_folder(self, **kwargs):
            calls["large"].append(kwargs)

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    push_to_hub_merged(
        model=None,
        tokenizer=None,
        save_directory=tmp_path,
        repo_id="me/merged",
        commit_message="Release v2",
        create_pr=True,
    )

    # Custom commit_message + create_pr=True must route through
    # upload_folder so both reach the Hub. upload_large_folder would
    # silently drop them.
    assert len(calls["folder"]) == 1, calls
    assert calls["large"] == [], calls
    sent = calls["folder"][0]
    assert "Release v2" in sent["commit_message"], sent
    assert sent["create_pr"] is True


def test_push_to_hub_merged_revision_alone_keeps_large_folder_route(
    tmp_path, monkeypatch,
):
    # upload_large_folder natively supports revision, so a revision-only
    # push should keep the resumable/chunked large-folder route for
    # multi-GB merged models. Only commit_message / commit_description /
    # create_pr force the upload_folder route (those are the kwargs
    # upload_large_folder cannot represent).
    import huggingface_hub
    from unsloth_zoo.mlx.utils import push_to_hub_merged

    (tmp_path / "model.safetensors.index.json").write_text("{}")

    calls = {"folder": [], "large": []}

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kwargs):
            return None

        def update_repo_settings(self, **kwargs):
            return None

        def upload_folder(self, **kwargs):
            calls["folder"].append(kwargs)

        def upload_large_folder(self, **kwargs):
            calls["large"].append(kwargs)

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    push_to_hub_merged(
        model=None,
        tokenizer=None,
        save_directory=tmp_path,
        repo_id="me/merged",
        revision="release-v3",
    )

    assert calls["folder"] == [], calls
    assert len(calls["large"]) == 1, calls
    assert calls["large"][0]["revision"] == "release-v3"


def test_push_to_hub_merged_uses_large_folder_when_no_custom_metadata(
    tmp_path, monkeypatch,
):
    # When the caller did NOT pass custom commit metadata, prefer
    # upload_large_folder so multi-GB merged dirs get chunked uploads
    # with resume. This preserves the original large-merge behavior.
    import huggingface_hub
    from unsloth_zoo.mlx.utils import push_to_hub_merged

    (tmp_path / "model.safetensors.index.json").write_text("{}")

    calls = {"folder": [], "large": []}

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kwargs):
            return None

        def update_repo_settings(self, **kwargs):
            return None

        def upload_folder(self, **kwargs):
            calls["folder"].append(kwargs)

        def upload_large_folder(self, **kwargs):
            calls["large"].append(kwargs)

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    push_to_hub_merged(
        model=None,
        tokenizer=None,
        save_directory=tmp_path,
        repo_id="me/merged",
    )

    assert calls["folder"] == [], calls
    assert len(calls["large"]) == 1, calls


def test_push_lora_adapters_falls_back_to_large_folder_when_unavailable(
    tmp_path, monkeypatch,
):
    # On a hypothetical environment without upload_folder (or with a
    # TypeError signature mismatch), the helper should still complete the
    # upload via upload_large_folder rather than crash silently.
    import huggingface_hub
    from unsloth_zoo.mlx.utils import _push_lora_adapters_to_hub

    (tmp_path / "adapters.safetensors").write_bytes(b"\x00")

    calls = {"folder": 0, "large": []}

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kwargs):
            return None

        def update_repo_settings(self, **kwargs):
            return None

        def upload_folder(self, **kwargs):
            calls["folder"] += 1
            raise TypeError("simulated old huggingface_hub signature")

        def upload_large_folder(self, **kwargs):
            calls["large"].append(kwargs)

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    _push_lora_adapters_to_hub(tmp_path, repo_id="me/adapter")

    assert calls["folder"] == 1
    assert len(calls["large"]) == 1, calls
    assert calls["large"][0]["repo_id"] == "me/adapter"

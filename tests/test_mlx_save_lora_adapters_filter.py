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
    # Some PEFT-style adapters expose uppercase tensor names.
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


def test_collect_lora_helper_accepts_uppercase_pair(tmp_path):
    # PEFT-style adapters expose lora_A / lora_B; collector must yield
    # those tensors so save_lora_adapters writes them intact.
    from unsloth_zoo.mlx.utils import (
        collect_mlx_lora_adapter_tensors,
        save_lora_adapters,
    )

    model = _make_model_with_named_modules({
        "q_proj": _MockLoRALinearUpper(8, 16, 4, 1.0, _MockDropoutKeepProb(0.0)),
        "up_proj": _MockPlainLinear(16, 32),
    })

    found = collect_mlx_lora_adapter_tensors(model)
    assert set(found.keys()) == {"q_proj.lora_A", "q_proj.lora_B"}, found

    save_lora_adapters(model, tmp_path)
    from safetensors.torch import load_file
    keys = set(load_file(str(tmp_path / "adapters.safetensors")).keys())
    assert keys == {"q_proj.lora_A", "q_proj.lora_B"}, sorted(keys)


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


def test_iter_mlx_lora_modules_reports_attr_pair():
    from unsloth_zoo.mlx.utils import iter_mlx_lora_modules

    model = _make_model_with_named_modules({
        "q_proj": _MockLoRALinear(8, 16, 4, 1.0, _MockDropoutKeepProb(0.0)),
        "v_proj": _MockLoRALinearUpper(8, 16, 4, 1.0, _MockDropoutKeepProb(0.0)),
        "up_proj": _MockPlainLinear(16, 32),
    })
    rows = [(name, a, b) for name, _m, a, b in iter_mlx_lora_modules(model)]
    assert sorted(rows) == [
        ("q_proj", "lora_a", "lora_b"),
        ("v_proj", "lora_A", "lora_B"),
    ], rows


def test_enrich_adapter_config_records_uppercase_lora_paths():
    # _enrich_mlx_adapter_config must record uppercase LoRA module paths so
    # reload can recreate the matching wrappers.
    from unsloth_zoo.mlx.utils import _enrich_mlx_adapter_config

    model = _make_model_with_named_modules({
        "q_proj": _MockLoRALinearUpper(8, 16, 4, 1.0, _MockDropoutKeepProb(0.0)),
    })
    enriched = _enrich_mlx_adapter_config(model, {})
    assert enriched.get("unsloth_mlx_lora_module_paths") == ["q_proj"]


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

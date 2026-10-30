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

"""Adapter save tests: module-anchored lora_a/lora_b only; no name-substring leaks; trainables kept."""

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
    # External trainables survive; leaked base weights inside LoRA modules are dropped to stay loadable.
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
    # Uppercase lora_A/lora_B: mlx-lm reload recreates only lowercase wrappers, so they are skipped.
    def __init__(self, in_features, out_features, rank, scale, dropout):
        self.weight = torch.zeros(out_features, in_features)
        self.lora_A = torch.zeros(in_features, rank)
        self.lora_B = torch.zeros(rank, out_features)
        self.scale = scale
        self.dropout = dropout


class _MockHalfLoRA:
    # A half-built LoRA pair (one side only) must be dropped; saving it gives unreloadable adapters.
    def __init__(self, in_features, rank):
        self.weight = torch.zeros(in_features, in_features)
        self.lora_a = torch.zeros(in_features, rank)


def _make_model_with_named_modules(layers):
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
    # mlx-lm cannot reload uppercase tensors, so enrich must not advertise uppercase-only modules.
    from unsloth_zoo.mlx.utils import _enrich_mlx_adapter_config

    model = _make_model_with_named_modules({
        "q_proj": _MockLoRALinearUpper(8, 16, 4, 1.0, _MockDropoutKeepProb(0.0)),
    })
    enriched = _enrich_mlx_adapter_config(model, {})
    assert "unsloth_mlx_lora_module_paths" not in enriched


def test_collect_lora_helper_finds_adapters_after_reload():
    # After reload LoRA tensors may be absent from trainable_parameters(); detect them by module anchor.
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
            return {"up_proj.weight": plain.weight}

        def named_modules(self):
            yield "", self
            yield "q_proj", real
            yield "up_proj", plain

    found = collect_mlx_lora_adapter_tensors(_ReloadedModel())
    assert set(found.keys()) == {"q_proj.lora_a", "q_proj.lora_b"}, found


def test_collect_lora_helper_finds_linear_backed_adapter_factors():
    from unsloth_zoo.mlx.utils import collect_mlx_lora_adapter_tensors

    class _Factor:
        def __init__(self, shape):
            self.weight = torch.zeros(*shape)

    class _LinearBackedLoRA:
        def __init__(self):
            self.lora_a = _Factor((8, 4))
            self.lora_b = _Factor((4, 16))

    adapter = _LinearBackedLoRA()

    class _Model:
        def parameters(self):
            return {
                "q_proj": {
                    "lora_a": {"weight": adapter.lora_a.weight},
                    "lora_b": {"weight": adapter.lora_b.weight},
                }
            }

        def named_modules(self):
            yield "q_proj", adapter

    found = collect_mlx_lora_adapter_tensors(_Model())
    assert set(found) == {
        "q_proj.lora_a.weight", "q_proj.lora_b.weight",
    }


def test_ensure_lora_frozen_freezes_norm_when_lora_is_actively_trained():
    # Active LoRA: module-anchored detection makes the NaN safeguard freeze an accidental trainable norm.
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
    # Norm-only fine-tune: LoRA tensors present but untrainable, so the safeguard must not freeze norm.
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
    # No-LoRA full fine-tunes must not stamp fine_tune_type='lora' or lora_parameters; mlx-lm would inject LoRA wrappers before binding the full weights.
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
    # Gate on collect_mlx_lora_adapter_tensors so the user-facing "no LoRA layers" error fires instead of the lower-level one hasattr(m, "fuse") lets through.
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
    # save_method='lora' keeps user-trainable tensors outside LoRA modules (embed_tokens) but drops reload-leaked base weights inside them (q_proj.weight).
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
    from unsloth_zoo.mlx import utils as mlx_utils

    collect_calls = []
    merged_calls = []

    def _spy_collect(model):
        collect_calls.append(model)
        return {}

    def _stub_save_merged(model, tokenizer, path, dequantize=False,
                          quantize_unquantized=False):
        merged_calls.append((path, dequantize, quantize_unquantized))

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
    # Only merged_4bit opts into quantizing an unquantized merge; the flag is not inferred from dequantize=False, which push_to_hub_merged also uses.
    assert merged_calls[0][2] is False
    assert merged_calls[1][2] is True


def test_save_trainable_adapters_raises_when_no_trainable_params(tmp_path):
    # A fully frozen model has no trainable tensors, so the checkpoint would lack adapters.safetensors; raise explicitly as save_lora_adapters() does.
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
    # Anchor LoRA detection on modules, not the "lora" substring, so a norm inside a module named lora_router is still frozen when accidentally trainable.
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
    # save_method='lora' keeps intentional trainable non-LoRA tensors (embed_tokens, lm_head, projector, vision, norm) but not base weights inside LoRA modules.
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
    # save_pretrained_merged lora output must write lora_parameters (rank/scale/dropout) to adapter_config.json so mlx-lm load_adapters can recreate wrappers.
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
    # huggingface_hub>=0.34 upload_large_folder() cannot land commit_message, commit_description, create_pr or revision on the commit, so use upload_folder.
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

    # upload_folder is the primary path with kwargs threaded through; upload_large_folder would commit under the default "Upload N LFS files" message.
    assert len(calls["folder"]) == 1, calls
    assert calls["large"] == [], calls
    sent = calls["folder"][0]
    # Commit messages get a " (Trained with Unsloth)" suffix per Unsloth convention; the assertions check substrings to tolerate it.
    assert "Release v1" in sent["commit_message"], sent
    assert "Custom desc" in sent["commit_description"], sent
    assert sent["create_pr"] is True
    assert sent["revision"] == "main"
    assert sent["repo_id"] == "me/adapter"


def test_save_trainable_adapters_drops_quantized_base_scales_biases(tmp_path):
    # Reload-leaked mlx-lm QuantizedLinear state includes .scales and .biases alongside .weight; the filter must drop all three under LoRA prefixes.
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
    assert "q_proj.bias" in keys


def test_enrich_overrides_caller_full_when_lora_modules_present(tmp_path):
    # A caller-supplied fine_tune_type of full is overridden to lora when LoRA modules exist, since mlx-lm reload would skip wrapping and drop the adapter.
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
    # A no-LoRA save reusing a config from a prior LoRA save must strip stale peft_type, lora_parameters, rank before stamping fine_tune_type=full.
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
    # If upload_folder fails on a create_pr=True request, do not fall back to upload_large_folder, which would silently push to main; raise RuntimeError.
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
    # lora_module_prefixes drops only the wrapped base .weight (a reload leak with a V x H gradient); other trainable params such as q_proj.bias must survive.
    from unsloth_zoo.mlx.utils import save_trainable_adapters
    import torch as _t
    from safetensors.torch import load_file

    class _LoRALinearWithBias:
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

    assert "q_proj.lora_a" in keys
    assert "q_proj.lora_b" in keys
    assert "q_proj.weight" not in keys
    assert "q_proj.bias" in keys, "trainable .bias under LoRA module must not be dropped"
    assert "up_proj.weight" in keys


def test_enrich_stamps_fine_tune_type_full_when_no_lora_modules(tmp_path):
    # No-LoRA checkpoints must stamp fine_tune_type full, since mlx-lm load_adapters() defaults a missing one to lora and then needs num_layers/lora_parameters.
    from unsloth_zoo.mlx.utils import _enrich_mlx_adapter_config

    class _Model:
        def named_modules(self):
            return iter(())  # no LoRA modules
        def parameters(self):
            return {}

    cfg = _enrich_mlx_adapter_config(_Model(), {})
    assert cfg.get("fine_tune_type") == "full", cfg
    assert cfg.get("peft_type") is None or cfg.get("peft_type") != "LORA", cfg


def test_enrich_stamps_fine_tune_type_dora_when_dora_modules_present():
    # mlx-lm recreates DoRA wrappers only when fine_tune_type is "dora"; without that stamp the saved q_proj.m magnitudes cannot bind on reload.
    from unsloth_zoo.mlx.utils import _enrich_mlx_adapter_config
    import torch as _t

    class DoRALinear:
        def __init__(self):
            self.lora_a = _t.zeros(8, 4)
            self.lora_b = _t.zeros(4, 8)
            self.m = _t.zeros(8)
        # mlx-lm names the DoRA magnitude attribute m; the dora stamp is triggered by the gate on type(module).__name__.

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


def test_is_lm_head_trainable_skips_base_weight_under_lora_wrapped_lm_head(monkeypatch):
    # After reload mlx-lm may leak a LoRA-wrapped lm_head's base .weight as trainable; that defeats the CCE memory guard with a V x H gradient per chunk.
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

    # Trainable must be False for LoRA-only training; stub the descriptor to resolve the head, else an unresolved head returns False before the leak filter.
    import unsloth_zoo.mlx.utils as _U

    model = _Model()
    monkeypatch.setattr(
        _U, "describe_output_head",
        lambda m: type("_Desc", (), {"module": model._lm})(),
    )
    assert _is_lm_head_trainable(model) is False


def test_push_lora_adapters_uses_allow_patterns_to_avoid_stale_uploads(
    tmp_path, monkeypatch,
):
    # The LoRA push must not upload stale full-model files left in save_directory by a prior merged_16bit save; public repos would expose merged weights.
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
    assert "*.safetensors" not in patterns
    assert "*" not in patterns


def test_collect_lora_includes_m_on_real_dora_class():
    # Classes named DoRA* (DoRALinear, DoRAEmbedding) must keep their m magnitude in collect_mlx_lora_adapter_tensors; a typo would drop it from every export.
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
    # m is a generic attribute name, so gate DoRA magnitudes on the class name starting with DoRA; a non-DoRA self.m must not ship under DoRA semantics.
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
    assert "q_proj.lora_a" in tensors
    assert "q_proj.lora_b" in tensors
    assert "q_proj.m" not in tensors


def test_save_adapter_artifacts_rejects_empty_tensors():
    # Empty tensors={} must raise a clear error, not write an adapter_config.json without weights beside it.
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
    # Regression for the upload_large_folder kwarg drop on the merged-save path: create_pr=True or a custom commit_message must route through upload_folder.
    import huggingface_hub
    from unsloth_zoo.mlx.utils import push_to_hub_merged

    # Pretend the model was already saved so push_to_hub_merged skips the merge step and only runs the upload path under test.
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

    assert len(calls["folder"]) == 1, calls
    assert calls["large"] == [], calls
    sent = calls["folder"][0]
    assert "Release v2" in sent["commit_message"], sent
    assert sent["create_pr"] is True


def test_push_to_hub_merged_revision_alone_keeps_large_folder_route(
    tmp_path, monkeypatch,
):
    # upload_large_folder natively supports revision, so revision-only pushes stay resumable; commit_message,
    # commit_description or create_pr force upload_folder.
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


def test_save_lora_adapters_method_ignores_a_fully_unfrozen_tree(tmp_path):
    # load_adapters() never freezes, so every base weight reads trainable, which is not CPT state; keep the LoRA-only writer.
    from unsloth_zoo.mlx.loader import _mlx_save_lora_adapters

    model = _make_model({
        "q_proj": _MockLoRALinear(8, 16, 4, 1.0, _MockDropoutKeepProb(0.0)),
        "up_proj": _MockPlainLinear(16, 32),
    })
    _mlx_save_lora_adapters(model, tmp_path)

    from safetensors.torch import load_file
    keys = set(load_file(str(tmp_path / "adapters.safetensors")).keys())
    assert keys == {"q_proj.lora_a", "q_proj.lora_b"}, sorted(keys)


def test_save_lora_adapters_method_raises_when_no_lora_modules(tmp_path):
    # A base model must still error, not write a whole-model "adapter".
    from unsloth_zoo.mlx.loader import _mlx_save_lora_adapters

    model = _make_model({"up_proj": _MockPlainLinear(16, 32)})
    with pytest.raises(ValueError, match="LoRA adapter tensors"):
        _mlx_save_lora_adapters(model, tmp_path)


def test_save_lora_adapters_method_keeps_continued_pretraining_tensors(tmp_path):
    # Real CPT state: embed_tokens.weight trainable, the rest frozen.
    from unsloth_zoo.mlx.loader import _mlx_save_lora_adapters

    model = _make_model({
        "q_proj": _MockLoRALinear(8, 16, 4, 1.0, _MockDropoutKeepProb(0.0)),
        "embed_tokens": _MockPlainLinear(16, 32),
    })
    everything = model.parameters()
    model.trainable_parameters = lambda: {
        key: everything[key] for key in (
            "embed_tokens.weight", "q_proj.lora_a", "q_proj.lora_b",
        )
    }
    _mlx_save_lora_adapters(model, tmp_path)

    from safetensors.torch import load_file
    keys = set(load_file(str(tmp_path / "adapters.safetensors")).keys())
    assert keys == {
        "q_proj.lora_a", "q_proj.lora_b", "embed_tokens.weight",
    }, sorted(keys)

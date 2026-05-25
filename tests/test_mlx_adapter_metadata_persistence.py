"""Coverage for MLX LoRA adapter save/reload metadata persistence.

Targets the helpers that infer rank/scale/dropout from live MLX LoRA
modules and the reload-time wrapper that recreates non-language LoRA
paths before loading adapter weights:

  - _get_mlx_dropout_probability prefers MLX Dropout._p_1 keep-probability
    over the stale .p attribute used by compatibility shims.
  - _infer_mlx_lora_rank reads MoE/Switch rank from lora_a.shape[-2] and
    cross-checks lora_b shape, returning None on mismatch and handling
    None/missing-shape inputs without raising.
  - _enrich_mlx_adapter_config persists rank/scale/dropout under an
    explicit filter that does NOT borrow metadata from unselected modules
    while still honoring caller-provided topology metadata exactly.
  - _apply_lora_at_paths TypeError-fallback restores both scale AND
    dropout when older mlx-lm wrappers reject those kwargs.
"""

import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)


def _load_utils():
    if "mlx" not in sys.modules:
        from mlx_simulation import simulate_mlx_on_torch
        simulate_mlx_on_torch()
    from unsloth_zoo.mlx import utils as mlx_utils
    return mlx_utils


class _FakeArray:
    def __init__(self, shape):
        self.shape = tuple(shape)


class _FakeDropout:
    def __init__(self, keep_prob):
        self._p_1 = keep_prob


class _ShimDropout:
    def __init__(self, p):
        self.p = p


class _FakeLoRAModule:
    def __init__(self, lora_a, lora_b, scale=1.0, dropout=None):
        self.lora_a = _FakeArray(lora_a)
        self.lora_b = _FakeArray(lora_b)
        self.scale = scale
        self.dropout = dropout


class _FakeModel:
    def __init__(self, modules):
        self._modules = modules

    def named_modules(self):
        return list(self._modules)


def test_dropout_prefers_keep_prob_over_stale_p():
    mlx_utils = _load_utils()
    drop = types.SimpleNamespace(p=0.0, _p_1=0.8)
    assert abs(mlx_utils._get_mlx_dropout_probability(drop) - 0.2) < 1e-9


def test_dropout_falls_back_to_p_attribute_for_shims():
    mlx_utils = _load_utils()
    drop = types.SimpleNamespace(p=0.3)
    assert abs(mlx_utils._get_mlx_dropout_probability(drop) - 0.3) < 1e-9


def test_dropout_handles_none_module():
    mlx_utils = _load_utils()
    assert mlx_utils._get_mlx_dropout_probability(None) == 0.0


def test_infer_rank_moe_uses_axis_minus_two():
    mlx_utils = _load_utils()
    moe = _FakeLoRAModule((8, 4, 512), (8, 512, 4))
    assert mlx_utils._infer_mlx_lora_rank(moe) == 4


class _FakeWeightLayer:
    def __init__(self, shape):
        self.weight = _FakeArray(shape)


def test_infer_rank_supports_nn_linear_layers():
    # mlx-lm uses nn.Linear layers for lora_a/lora_b.
    mlx_utils = _load_utils()
    mod = types.SimpleNamespace(
        lora_a=_FakeWeightLayer((4, 512)),
        lora_b=_FakeWeightLayer((1024, 4)),
    )
    assert mlx_utils._infer_mlx_lora_rank(mod) == 4


def test_infer_rank_returns_none_on_shape_mismatch():
    mlx_utils = _load_utils()
    bad = _FakeLoRAModule((512, 4), (8, 512))
    assert mlx_utils._infer_mlx_lora_rank(bad) is None


def test_infer_rank_handles_none_tensors_without_raising():
    mlx_utils = _load_utils()
    mod = types.SimpleNamespace(lora_a=None, lora_b=None)
    assert mlx_utils._infer_mlx_lora_rank(mod) is None


def test_infer_rank_handles_missing_shape_attribute():
    mlx_utils = _load_utils()
    mod = types.SimpleNamespace(lora_a=object(), lora_b=_FakeArray((4, 512)))
    assert mlx_utils._infer_mlx_lora_rank(mod) is None


def test_enrich_no_explicit_paths_writes_metadata():
    mlx_utils = _load_utils()
    model = _FakeModel([
        ("layers.0.q_proj", _FakeLoRAModule((512, 4), (4, 512), scale=2.5,
                                            dropout=_FakeDropout(0.75))),
    ])
    cfg = mlx_utils._enrich_mlx_adapter_config(model, {})
    assert cfg["rank"] == 4
    assert cfg["scale"] == 2.5
    assert abs(cfg["dropout"] - 0.25) < 1e-9
    assert cfg["lora_parameters"]["rank"] == 4
    assert cfg["unsloth_mlx_lora_module_paths"] == ["layers.0.q_proj"]
    assert cfg["peft_type"] == "LORA"


def test_enrich_explicit_empty_paths_still_writes_metadata():
    mlx_utils = _load_utils()
    model = _FakeModel([
        ("layers.0.q_proj", _FakeLoRAModule((512, 4), (4, 512), scale=2.5,
                                            dropout=_FakeDropout(0.75))),
    ])
    cfg = mlx_utils._enrich_mlx_adapter_config(
        model, {"unsloth_mlx_lora_module_paths": []}
    )
    assert cfg["unsloth_mlx_lora_module_paths"] == []
    assert cfg["rank"] == 4
    assert cfg["scale"] == 2.5
    assert abs(cfg["dropout"] - 0.25) < 1e-9


def test_enrich_explicit_filter_uses_selected_module():
    mlx_utils = _load_utils()
    model = _FakeModel([
        ("vision.proj", _FakeLoRAModule((1024, 8), (8, 1024), scale=4.0,
                                         dropout=_FakeDropout(0.9))),
        ("language.q", _FakeLoRAModule((512, 4), (4, 512), scale=2.0,
                                        dropout=_FakeDropout(0.8))),
    ])
    cfg = mlx_utils._enrich_mlx_adapter_config(
        model, {"unsloth_mlx_lora_module_paths": ["language.q"]}
    )
    assert cfg["unsloth_mlx_lora_module_paths"] == ["language.q"]
    assert cfg["rank"] == 4
    assert cfg["scale"] == 2.0


def test_enrich_explicit_filter_does_not_borrow_from_unselected():
    mlx_utils = _load_utils()
    bad = _FakeLoRAModule((512, 4), (8, 512), scale=9.9,
                          dropout=_FakeDropout(0.5))
    good = _FakeLoRAModule((512, 2), (2, 512), scale=2.0,
                            dropout=_FakeDropout(0.8))
    model = _FakeModel([("bad", bad), ("good", good)])
    cfg = mlx_utils._enrich_mlx_adapter_config(
        model, {"unsloth_mlx_lora_module_paths": ["bad"]}
    )
    assert "rank" not in cfg
    assert "scale" not in cfg
    assert "dropout" not in cfg
    assert cfg["unsloth_mlx_lora_module_paths"] == ["bad"]


def test_enrich_skips_invalid_rank_then_uses_next_valid():
    mlx_utils = _load_utils()
    bad = _FakeLoRAModule((512, 4), (8, 512), scale=9.9,
                          dropout=_FakeDropout(0.5))
    good = _FakeLoRAModule((512, 4), (4, 512), scale=2.0,
                           dropout=_FakeDropout(0.75))
    model = _FakeModel([("layers.0.bad", bad), ("layers.0.q_proj", good)])
    cfg = mlx_utils._enrich_mlx_adapter_config(model, {})
    assert cfg["rank"] == 4
    assert cfg["scale"] == 2.0


def test_enrich_does_not_raise_on_module_with_none_tensors():
    mlx_utils = _load_utils()
    broken = types.SimpleNamespace(lora_a=None, lora_b=None, scale=1.0,
                                    dropout=None)
    good = _FakeLoRAModule((512, 4), (4, 512), scale=2.0,
                            dropout=_FakeDropout(0.75))
    model = _FakeModel([("broken", broken), ("good", good)])
    cfg = mlx_utils._enrich_mlx_adapter_config(model, {})
    assert cfg["rank"] == 4
    assert cfg["scale"] == 2.0


def test_infer_rank_rejects_one_dimensional_lora_b():
    # A bare 1D lora_b means the pair is half-built; rank inference would
    # otherwise read shape[-1] from lora_a and persist nonsense metadata.
    mlx_utils = _load_utils()
    bad = _FakeLoRAModule((512, 4), (4,))
    assert mlx_utils._infer_mlx_lora_rank(bad) is None


def test_normalize_mlx_lora_module_paths_handles_string_input():
    # Hand-authored adapter_config.json files sometimes store a single
    # path as a bare string; the loader must convert it to a single-
    # element list rather than iterating its characters.
    _load_utils()
    from unsloth_zoo.mlx.loader import _normalize_mlx_lora_module_paths

    assert _normalize_mlx_lora_module_paths(None) == []
    assert _normalize_mlx_lora_module_paths("") == []
    assert _normalize_mlx_lora_module_paths("vision.proj") == ["vision.proj"]
    assert _normalize_mlx_lora_module_paths(["a", "", "b"]) == ["a", "b"]
    assert _normalize_mlx_lora_module_paths(("a", "b")) == ["a", "b"]
    assert _normalize_mlx_lora_module_paths(123) == []


def test_infer_rank_rejects_switch_lora_b_without_expert_prefix():
    # MoE/Switch lora_a (num_experts, rank, in_dims) paired with bare 2D
    # lora_b (out_dims, rank) is malformed; the helper must reject it
    # instead of returning a plausible rank.
    mlx_utils = _load_utils()
    bad = _FakeLoRAModule((8, 4, 512), (512, 4))
    assert mlx_utils._infer_mlx_lora_rank(bad) is None


def test_enrich_normalizes_string_explicit_path():
    # A caller-supplied single-string path must be normalized to a list so
    # the loader does not iterate it character-by-character.
    mlx_utils = _load_utils()
    model = _FakeModel([
        ("layers.0.q_proj", _FakeLoRAModule((512, 4), (4, 512), scale=2.0,
                                            dropout=_FakeDropout(0.8))),
    ])
    cfg = mlx_utils._enrich_mlx_adapter_config(
        model, {"unsloth_mlx_lora_module_paths": "layers.0.q_proj"},
    )
    assert cfg["unsloth_mlx_lora_module_paths"] == ["layers.0.q_proj"]
    assert cfg["rank"] == 4


def test_typeerror_fallback_restores_scale_and_dropout_via_p1():
    _load_utils()
    from unsloth_zoo.mlx.loader import _lora_from_base_compat

    class _NoKwargCls:
        @classmethod
        def from_base(cls, module, r=None, scale=None, dropout=None):
            if scale is not None or dropout is not None:
                raise TypeError("old mlx-lm: scale/dropout not accepted")
            return types.SimpleNamespace(scale=1.0,
                                          dropout=_FakeDropout(1.0))

    wrapped = _lora_from_base_compat(
        _NoKwargCls, object(), rank=4, scale=2.5, dropout=0.3,
    )
    assert wrapped.scale == 2.5
    assert abs(wrapped.dropout._p_1 - 0.7) < 1e-9


def test_typeerror_fallback_handles_shim_dropout_with_p_attr():
    _load_utils()
    from unsloth_zoo.mlx.loader import _apply_lora_metadata_to_wrapper

    wrapped = types.SimpleNamespace(scale=1.0, dropout=_ShimDropout(0.0))
    _apply_lora_metadata_to_wrapper(wrapped, scale=2.5, dropout=0.4)
    assert wrapped.scale == 2.5
    assert abs(wrapped.dropout.p - 0.4) < 1e-9


def _trainer_loop_metadata(model):
    # Mirror MLXTrainer.save_model()'s metadata extraction so the test
    # tracks the trainer logic without depending on a real MLX runtime.
    mlx_utils = _load_utils()
    rank, scale, dropout = 8, 1.0, 0.0
    for _, m in model.named_modules():
        if not (hasattr(m, "lora_a") and hasattr(m, "lora_b")):
            continue
        inferred_rank = mlx_utils._infer_mlx_lora_rank(m)
        if inferred_rank is None:
            continue
        rank = inferred_rank
        scale = getattr(m, "scale", 1.0)
        dropout = mlx_utils._get_mlx_dropout_probability(getattr(m, "dropout", None))
        break
    return rank, scale, dropout


def test_trainer_metadata_loop_no_lora_uses_defaults():
    model = _FakeModel([
        ("layers.0.q_proj", types.SimpleNamespace()),
    ])
    rank, scale, dropout = _trainer_loop_metadata(model)
    assert (rank, scale, dropout) == (8, 1.0, 0.0)

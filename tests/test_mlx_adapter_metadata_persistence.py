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
    rank = scale = dropout = None
    for _, m in model.named_modules():
        if not (hasattr(m, "lora_a") and hasattr(m, "lora_b")):
            continue
        inferred_rank = mlx_utils._infer_mlx_lora_rank(m)
        if inferred_rank is None:
            continue
        rank = inferred_rank
        _scale = getattr(m, "scale", 1.0)
        if hasattr(_scale, "item"):
            try:
                scale = float(_scale.item())
            except Exception:
                scale = 1.0
        else:
            try:
                scale = float(_scale)
            except Exception:
                scale = 1.0
        dropout = mlx_utils._get_mlx_dropout_probability(getattr(m, "dropout", None))
        break
    return rank, scale, dropout


def test_trainer_metadata_loop_returns_none_when_no_lora_modules():
    # No LoRA modules means we cannot produce trustworthy rank/scale/dropout.
    # Returning None lets the caller omit the keys from adapter_config so
    # reload fails loudly instead of silently scaling with rank=8.
    model = _FakeModel([
        ("layers.0.q_proj", types.SimpleNamespace()),
    ])
    assert _trainer_loop_metadata(model) == (None, None, None)


def test_trainer_metadata_loop_returns_none_when_rank_inference_fails():
    # Module has lora_a/lora_b but _infer_mlx_lora_rank returns None
    # (e.g. 1-D lora_b). Trainer must not persist the old placeholder
    # rank=8 / scale=1.0 / dropout=0.0 defaults.
    one_d_lora = _FakeLoRAModule(lora_a=(8, 4), lora_b=(8,))
    model = _FakeModel([("layers.0.q_proj", one_d_lora)])
    assert _trainer_loop_metadata(model) == (None, None, None)


def test_trainer_metadata_loop_coerces_mxarray_scale_safely():
    # LoRASwitchLinear stores scale as a per-expert mx.array. float() on a
    # non-0-D array raises; the trainer loop must coerce via .item() and
    # fall back to 1.0 if the array is wider than 0-D.
    class _ZeroDArray:
        def item(self):
            return 2.5

        @property
        def shape(self):
            return ()

    class _MultiExpertArray:
        def item(self):
            raise ValueError("only one element tensors can be converted")

        @property
        def shape(self):
            return (4,)

    m_zero_d = _FakeLoRAModule(lora_a=(8, 4), lora_b=(4, 8), scale=_ZeroDArray())
    m_wide   = _FakeLoRAModule(lora_a=(8, 4), lora_b=(4, 8), scale=_MultiExpertArray())

    _, scale_zero_d, _ = _trainer_loop_metadata(_FakeModel([("q", m_zero_d)]))
    _, scale_wide,   _ = _trainer_loop_metadata(_FakeModel([("q", m_wide)]))

    assert scale_zero_d == 2.5
    assert scale_wide == 1.0  # safe fallback, not a crash


def _build_trainer_adapter_dict(rank, scale, dropout, num_layers, hf_repo=""):
    # Mirror the dict-building gate in MLXTrainer.save_model: keys are only
    # included when their source values are valid, so reload sees no
    # placeholder sentinels.
    cfg = {
        "fine_tune_type": "lora",
        "peft_type": "LORA",
        "base_model_name_or_path": hf_repo,
    }
    if num_layers is not None and num_layers > 0:
        cfg["num_layers"] = num_layers
    if rank is not None:
        cfg["lora_parameters"] = {"rank": rank, "scale": scale, "dropout": dropout}
        cfg["rank"] = rank
        cfg["scale"] = scale
        cfg["dropout"] = dropout
    return cfg


def test_trainer_adapter_dict_omits_num_layers_sentinel():
    # _num_layers=None must not land in adapter_config as num_layers=-1
    # or 0; mlx-lm's load_adapters would slice range(-1) and apply zero
    # LoRA layers on reload.
    cfg = _build_trainer_adapter_dict(rank=8, scale=1.0, dropout=0.0, num_layers=None)
    assert "num_layers" not in cfg
    cfg2 = _build_trainer_adapter_dict(rank=8, scale=1.0, dropout=0.0, num_layers=0)
    assert "num_layers" not in cfg2


def test_trainer_adapter_dict_omits_rank_when_inference_failed():
    # No LoRA found -> trainer must NOT write the placeholder rank=8 /
    # scale=1.0 / dropout=0.0 trio. Reload should fail loud instead.
    cfg = _build_trainer_adapter_dict(rank=None, scale=None, dropout=None, num_layers=24)
    assert "rank" not in cfg
    assert "scale" not in cfg
    assert "dropout" not in cfg
    assert "lora_parameters" not in cfg
    assert cfg["num_layers"] == 24
    # Identity keys still present so HF PEFT and mlx-lm can route the load.
    assert cfg["peft_type"] == "LORA"
    assert cfg["fine_tune_type"] == "lora"


def test_normalize_mlx_lora_module_paths_handles_dict_input():
    # Older / hand-authored configs sometimes group paths by tower.
    # Dict input must flatten into a list, not silently return [].
    _load_utils()  # installs mlx torch simulation so loader imports cleanly
    from unsloth_zoo.mlx.loader import _normalize_mlx_lora_module_paths

    grouped = {"language": ["layers.0.q_proj"], "vision": ["vision.proj"]}
    paths = _normalize_mlx_lora_module_paths(grouped)
    assert set(paths) == {"layers.0.q_proj", "vision.proj"}


def test_normalize_mlx_lora_module_paths_handles_pathlike_elements():
    # pathlib.Path elements in the saved list must coerce via os.fspath,
    # not be silently dropped by the isinstance(p, str) gate.
    import pathlib
    _load_utils()
    from unsloth_zoo.mlx.loader import _normalize_mlx_lora_module_paths

    paths = _normalize_mlx_lora_module_paths(
        [pathlib.PurePosixPath("layers.0.q_proj"), "layers.1.q_proj"]
    )
    assert paths == ["layers.0.q_proj", "layers.1.q_proj"]


def test_normalize_mlx_lora_module_paths_handles_bare_pathlike():
    # A bare pathlib.Path (not wrapped in a list) should also coerce.
    import pathlib
    _load_utils()
    from unsloth_zoo.mlx.loader import _normalize_mlx_lora_module_paths

    paths = _normalize_mlx_lora_module_paths(
        pathlib.PurePosixPath("layers.0.q_proj")
    )
    assert paths == ["layers.0.q_proj"]


def test_enrich_mlx_adapter_config_normalizes_dict_explicit_paths():
    # Save-side path normalization must accept the same shapes the load
    # side does: dict-grouped paths, pathlib.Path elements, set, etc.
    # Otherwise vision/projector LoRA topology is silently erased before
    # it reaches adapter_config.json.
    mlx_utils = _load_utils()
    model = _FakeModel([
        ("layers.0.q_proj", _FakeLoRAModule((512, 4), (4, 512), scale=2.0)),
        ("vision.proj", _FakeLoRAModule((512, 4), (4, 512), scale=2.0)),
    ])
    cfg = mlx_utils._enrich_mlx_adapter_config(
        model,
        {"unsloth_mlx_lora_module_paths": {"language": ["layers.0.q_proj"],
                                            "vision": ["vision.proj"]}},
    )
    assert set(cfg["unsloth_mlx_lora_module_paths"]) == {
        "layers.0.q_proj", "vision.proj",
    }


def test_enrich_mlx_adapter_config_normalizes_pathlike_explicit_paths():
    import pathlib
    mlx_utils = _load_utils()
    model = _FakeModel([
        ("layers.0.q_proj", _FakeLoRAModule((512, 4), (4, 512), scale=2.0)),
    ])
    cfg = mlx_utils._enrich_mlx_adapter_config(
        model,
        {"unsloth_mlx_lora_module_paths": [
            pathlib.PurePosixPath("layers.0.q_proj"),
        ]},
    )
    assert cfg["unsloth_mlx_lora_module_paths"] == ["layers.0.q_proj"]


def test_enrich_mlx_adapter_config_coerces_mxarray_scale_without_aborting():
    # _enrich_mlx_adapter_config previously used raw float(module.scale).
    # A LoRASwitchLinear that exposes scale as a multi-element mx.array
    # made float() raise, which the outer try/except: pass swallowed -
    # silently dropping unsloth_mlx_lora_module_paths so the reload path
    # could not re-attach vision/projector LoRA wrappers. Mirror the
    # trainer-side .item() / fallback-1.0 coercion in enrich too.
    mlx_utils = _load_utils()

    class _MultiExpertScale:
        def item(self):
            raise ValueError("only one element tensors can be converted")

        @property
        def shape(self):
            return (4,)

    class _ZeroDScale:
        def item(self):
            return 0.5

        @property
        def shape(self):
            return ()

    m_wide = _FakeLoRAModule(lora_a=(8, 4), lora_b=(4, 8), scale=_MultiExpertScale())
    m_zerod = _FakeLoRAModule(lora_a=(8, 4), lora_b=(4, 8), scale=_ZeroDScale())

    cfg_wide = mlx_utils._enrich_mlx_adapter_config(
        _FakeModel([("vision.q_proj", m_wide)]), {},
    )
    cfg_zd = mlx_utils._enrich_mlx_adapter_config(
        _FakeModel([("vision.q_proj", m_zerod)]), {},
    )

    # Both must complete with metadata and module-path list, not silently
    # abandon after the float() raise.
    assert cfg_wide.get("unsloth_mlx_lora_module_paths") == ["vision.q_proj"]
    assert cfg_wide.get("scale") == 1.0  # safe fallback
    assert cfg_wide.get("rank") == 4
    assert cfg_zd.get("unsloth_mlx_lora_module_paths") == ["vision.q_proj"]
    assert cfg_zd.get("scale") == 0.5
    assert cfg_zd.get("rank") == 4

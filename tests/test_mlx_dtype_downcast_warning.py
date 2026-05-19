# Unsloth Zoo - Utilities for Unsloth
# Test the bf16->fp16 downcast warning in unsloth_zoo.mlx.loader._convert_mlx_dtype.
# Warning gated on model_type being in unsloth_zoo.FORCE_FLOAT32.

from __future__ import annotations

import pytest
import warnings


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


class _FakeArray:
    """Minimal stand-in for an mx.array carrying just .dtype/.astype."""
    def __init__(self, dtype):
        import torch
        self._t = torch.zeros((1,), dtype=dtype)

    @property
    def dtype(self):
        return self._t.dtype

    def astype(self, target_dtype):
        return _FakeArray(target_dtype)


class _FakeModel:
    def __init__(self, params):
        self._params = params

    def parameters(self):
        return self._params

    def update(self, new_params):
        self._params = new_params


def _make_model(dtype_map):
    return _FakeModel({name: _FakeArray(dt) for name, dt in dtype_map.items()})


def _bf_fp_msgs(caught):
    return [
        str(w.message) for w in caught
        if "bfloat16" in str(w.message) and "float16" in str(w.message)
    ]


def test_force_float32_list_exported():
    """FORCE_FLOAT32 is importable from the top-level unsloth_zoo namespace."""
    import unsloth_zoo
    assert "gemma3," in unsloth_zoo.FORCE_FLOAT32
    assert "gpt_oss" in unsloth_zoo.FORCE_FLOAT32
    assert "qwen3_5" in unsloth_zoo.FORCE_FLOAT32


def test_force_float32_matches_config_json_model_types():
    """Entries are HuggingFace `config.json` `model_type` values (the same
    strings returned by `get_transformers_model_type`). The MLX matcher
    must accept the real on-disk variants verbatim."""
    from unsloth_zoo.mlx.loader import _is_force_float32_arch
    # Real config.json model_type strings as they appear on the Hub.
    real_world = [
        "gemma3",       # google/gemma-3-*
        "gemma3_text",  # google/embeddinggemma-*
        "gemma3n",      # google/gemma-3n-*
        "gpt_oss",      # openai/gpt-oss-*
        "qwen3_5",      # Qwen/Qwen3.5-*
    ]
    for model_type in real_world:
        assert _is_force_float32_arch(model_type) is True, (
            f"FORCE_FLOAT32 must match real config.json model_type {model_type!r}"
        )


def test_warns_on_bf16_to_fp16_for_gemma3():
    """The production scenario: gemma3 bf16 weights downcast to fp16."""
    import torch
    from unsloth_zoo.mlx.loader import _convert_mlx_dtype

    model = _make_model({"w1": torch.bfloat16, "w2": torch.bfloat16})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _convert_mlx_dtype(model, torch.float16, model_type="gemma3")

    assert _bf_fp_msgs(caught), (
        f"Expected a bf16->fp16 downcast warning for gemma3, got: "
        f"{[str(w.message) for w in caught]}"
    )


def test_warns_for_other_force_float32_archs():
    """gpt_oss and qwen3_5 also need the warning."""
    import torch
    from unsloth_zoo.mlx.loader import _convert_mlx_dtype

    for mt in ("gpt_oss", "qwen3_5", "gemma3n", "gemma3text"):
        model = _make_model({"w1": torch.bfloat16})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _convert_mlx_dtype(model, torch.float16, model_type=mt)
        assert _bf_fp_msgs(caught), f"Expected warning for {mt!r}"


def test_no_warning_for_safe_architecture():
    """A model NOT in FORCE_FLOAT32 (e.g. llama) downcasts silently."""
    import torch
    from unsloth_zoo.mlx.loader import _convert_mlx_dtype

    model = _make_model({"w1": torch.bfloat16})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _convert_mlx_dtype(model, torch.float16, model_type="llama")

    assert _bf_fp_msgs(caught) == [], (
        f"Did not expect a warning for llama; got: "
        f"{[str(w.message) for w in caught]}"
    )


def test_gemma3_comma_does_not_match_gemma3n():
    """The trailing-comma trick: 'gemma3,' must NOT match 'gemma3n_audio'."""
    import torch
    from unsloth_zoo.mlx.loader import _is_force_float32_arch
    # gemma3n is its own entry and DOES match; we're testing that the
    # 'gemma3,' entry doesn't accidentally swallow gemma3n variants by
    # prefix match. gemma3n itself still matches via its own entry.
    assert _is_force_float32_arch("gemma3") is True
    assert _is_force_float32_arch("gemma3n") is True
    assert _is_force_float32_arch("gemma3text") is True
    # An invented gemma3 variant not in the list returns False
    assert _is_force_float32_arch("gemma3_audio_only_pretend") is False


def test_no_warning_on_bf16_to_fp32_upcast():
    """Upcasts are safe; no warning even for gemma3."""
    import torch
    from unsloth_zoo.mlx.loader import _convert_mlx_dtype

    model = _make_model({"w1": torch.bfloat16})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _convert_mlx_dtype(model, torch.float32, model_type="gemma3")

    assert _bf_fp_msgs(caught) == []


def test_no_warning_on_fp32_to_fp16_downcast():
    """fp32->fp16 is a different (explicit) regime; no warning."""
    import torch
    from unsloth_zoo.mlx.loader import _convert_mlx_dtype

    model = _make_model({"w1": torch.float32})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _convert_mlx_dtype(model, torch.float16, model_type="gemma3")

    assert _bf_fp_msgs(caught) == []


def test_no_warning_when_no_cast_needed():
    """All params already at target_dtype -> early return."""
    import torch
    from unsloth_zoo.mlx.loader import _convert_mlx_dtype

    model = _make_model({"w1": torch.float16})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _convert_mlx_dtype(model, torch.float16, model_type="gemma3")

    assert _bf_fp_msgs(caught) == []


def test_cast_still_happens_after_warning():
    """The warning is advisory only — the cast still occurs."""
    import torch
    from unsloth_zoo.mlx.loader import _convert_mlx_dtype

    model = _make_model({"w1": torch.bfloat16})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _convert_mlx_dtype(model, torch.float16, model_type="gemma3")

    assert model._params["w1"].dtype == torch.float16

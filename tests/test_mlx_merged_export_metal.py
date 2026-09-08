"""Real-Metal checks on a merged MLX export: they need `mx.quantize`, which the
mlx simulation shim does not implement."""

import json

import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="requires Apple Silicon Metal")
from unsloth_zoo.mlx.loader import FastMLXModel, _canonical_mlx_quantization_path
from unsloth_zoo.mlx.utils import save_pretrained_merged

# Dense bf16: an already-quantized base keeps its widths through fuse(), so a
# merge off one never reaches the path these cover.
MODEL = "mlx-community/SmolLM2-135M-Instruct"


def _widths(model):
    """Every packed module's own width, keyed by its pre-LoRA path."""
    return {_canonical_mlx_quantization_path(path): (module.bits, module.group_size)
            for path, module in model.named_modules()
            if path and hasattr(module, "bits") and hasattr(module, "group_size")}


def _quantized_linear():
    import mlx.nn as nn

    return nn.QuantizedLinear.from_linear(
        nn.Linear(64, 64, bias=False), group_size=64, bits=4)


def _wrap(built):
    import mlx.nn as nn

    model = nn.Module()
    model.fc = built
    return model


def test_merged_4bit_quantizes_a_base_that_was_never_quantized(tmp_path):
    model, tokenizer = FastMLXModel.from_pretrained(
        model_name=MODEL, max_seq_length=256, dtype=None, load_in_16bit=True)
    assert _widths(model) == {}
    model = FastMLXModel.get_peft_model(
        model, r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
    export = tmp_path / "merged"
    save_pretrained_merged(model, tokenizer, str(export), save_method="merged_4bit")

    # Read off the checkpoint: a reload would quantize a dense one on the way in.
    weights = {name: array for shard in export.glob("*.safetensors")
               for name, array in mx.load(str(shard)).items()}
    assert weights["model.layers.0.self_attn.q_proj.weight"].dtype == mx.uint32
    # The width itself, since uint32 is how every packed width is stored.
    quantization = json.loads((export / "config.json").read_text())["quantization"]
    assert (quantization["bits"], quantization["group_size"]) == (4, 64)
    # The output head and embeddings follow the loader's skip rules.
    assert "model.embed_tokens.scales" not in weights


def test_an_already_quantized_fused_model_is_left_alone(monkeypatch):
    # fuse() has already restored each module's own width by this point.
    from unsloth_zoo.mlx import utils as mutils

    monkeypatch.setattr("mlx_lm.utils.quantize_model", lambda *a, **k: pytest.fail(
        "an already-quantized model must not be quantized a second time"))
    mutils._quantize_merged_model(
        _wrap(_quantized_linear()), mutils._MERGED_4BIT_QUANTIZATION)


def test_a_merge_that_quantized_nothing_is_refused():
    # mlx-lm writes the quantization block before deciding which layers are
    # eligible, so a checkpoint can carry one over entirely dense weights.
    import mlx.nn as nn
    from unsloth_zoo.mlx import utils as mutils

    # 10 is not a multiple of any supported group size, so no layer qualifies.
    model = _wrap(nn.Linear(10, 10))
    model._config = {}
    with pytest.raises(RuntimeError, match="no layer could be quantized"):
        mutils._quantize_merged_model(model, mutils._MERGED_4BIT_QUANTIZATION)

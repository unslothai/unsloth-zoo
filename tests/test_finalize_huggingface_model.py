import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import types
import torch
from unsloth_zoo.empty_model import finalize_huggingface_model


class _LinearAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_idx = -1


class _StandardLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_idx = -1
        self.linear_attn = _LinearAttn()


class _StandardLM(torch.nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()

        class _Inner(torch.nn.Module):
            def __init__(self, n):
                super().__init__()
                self.layers = torch.nn.ModuleList([_StandardLayer() for _ in range(n)])

        self.model = _Inner(n_layers)


def _make_config(model_type="qwen3_5", has_vision=False):
    cfg = types.SimpleNamespace()
    cfg.model_type = model_type
    cfg.text_config = cfg
    if has_vision:
        vc = types.SimpleNamespace()
        vc.hidden_size = 1
        vc.num_heads = 1
        cfg.vision_config = vc
    return cfg


def test_finalize_fixes_layer_idx_on_standard_causal_lm():
    # Pre-fix: finalize_huggingface_model only touched new_model.model.language_model.layers,
    # so standard-LM paths kept layer_idx at the empty-model stub value.
    model = _StandardLM(n_layers=4)
    cfg = _make_config(model_type="qwen3_5")
    finalize_huggingface_model(
        model, None, cfg, torch.float16,
        quantization_config={"x": 1}, bnb_config=None,  # skip .to() to avoid meta tensors
    )
    for i, layer in enumerate(model.model.layers):
        assert layer.layer_idx == i
        assert layer.linear_attn.layer_idx == i


def test_finalize_also_handles_vlm_language_model_path():
    # Original VLM path should still work.
    class _VLM(torch.nn.Module):
        def __init__(self):
            super().__init__()

            class _Inner(torch.nn.Module):
                def __init__(self):
                    super().__init__()

                    class _LM(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.layers = torch.nn.ModuleList([_StandardLayer() for _ in range(3)])

                    self.language_model = _LM()

            self.model = _Inner()

    model = _VLM()
    cfg = _make_config()
    finalize_huggingface_model(
        model, None, cfg, torch.float16,
        quantization_config={"x": 1}, bnb_config=None,
    )
    for i, layer in enumerate(model.model.language_model.layers):
        assert layer.layer_idx == i
        assert layer.linear_attn.layer_idx == i


def test_finalize_does_not_assert_when_rotary_pos_emb_without_vision_config():
    # Pre-fix: hard `assert vision_config is not None` crashed text-only models that
    # happened to expose a rotary_pos_emb attr. Post-fix: skip silently.
    class _Rotary(torch.nn.Module):
        def __init__(self):
            super().__init__()

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_pos_emb = _Rotary()

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([_Layer()])

    model = _Model()
    cfg = _make_config(has_vision=False)
    # Should not raise AssertionError
    finalize_huggingface_model(
        model, None, cfg, torch.float16,
        quantization_config={"x": 1}, bnb_config=None,
    )

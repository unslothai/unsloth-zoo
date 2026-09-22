# Small helpers behind the structural MoE dispatch: LoRA target regex skips
# Linear subclasses that return tuples, the compiler leaves scalar-only helper
# functions alone, compiled classes stay native, Inkling's config split.
import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers")


def test_linear_forward_returns_tensor():
    from unsloth_zoo.peft_utils import _linear_forward_returns_tensor, get_peft_regex

    class Router(nn.Linear):
        def forward(self, x):
            logits = super().forward(x)
            return torch.sigmoid(logits), logits

    class Scaled(nn.Linear):
        def forward(self, x):
            return super().forward(x) * 2

    assert _linear_forward_returns_tensor(nn.Linear)
    assert _linear_forward_returns_tensor(Scaled)
    assert not _linear_forward_returns_tensor(Router)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.router = Router(8, 4)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"model_type": "x", "architectures": [], "_name_or_path": "x"})()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
            for layer in self.model.layers:
                layer.self_attn = Block()
                layer.mlp = Block()

    import re
    regex = get_peft_regex(Model())
    names = [n for n, m in Model().named_modules() if isinstance(m, nn.Linear)]
    matched = {n for n in names if re.fullmatch(regex, n)}
    assert all("router" not in n for n in matched), matched
    assert any(n.endswith("q_proj") for n in matched)


def test_function_has_tensor_inputs():
    from unsloth_zoo.compiler import function_has_tensor_inputs
    scalar = 'def plan_out_scales(temporal_patch_size: int, patch_size: int, n_layers: int, n_channels: int, device="cpu") -> torch.LongTensor:\n    pass'
    assert not function_has_tensor_inputs(scalar)
    assert not function_has_tensor_inputs("def prime_factors(number: int) -> list[int]:\n    pass")
    assert function_has_tensor_inputs("def rotate_half(x):\n    pass")
    assert function_has_tensor_inputs("def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:\n    pass")
    assert function_has_tensor_inputs("def f(module: nn.Module, query, key, scaling: float, **kwargs):\n    pass")
    assert function_has_tensor_inputs("def g(config: Optional[PreTrainedConfig] = None, seq_len: Optional[int] = None):\n    pass")
    assert function_has_tensor_inputs("not python at all")


def test_compiled_model_is_not_custom_code():
    from transformers.modeling_utils import PreTrainedModel
    from transformers.configuration_utils import PreTrainedConfig
    from unsloth_zoo.temporary_patches.compiled_model_identity import patch_compiled_model_is_custom_code
    saved = PreTrainedModel.__dict__["is_custom_code"]
    patch_compiled_model_is_custom_code()

    Native = type("NativeModel", (PreTrainedModel,), {"__module__": "transformers.models.x.modeling_x", "config_class": PreTrainedConfig})
    Compiled = type("NativeModel", (PreTrainedModel,), {"__module__": "unsloth_compiled_module_x", "config_class": PreTrainedConfig})
    User = type("UserModel", (PreTrainedModel,), {"__module__": "__main__", "config_class": PreTrainedConfig})
    try:
        assert not Native.is_custom_code()
        assert not Compiled.is_custom_code()
        assert User.is_custom_code()
    finally:
        PreTrainedModel.is_custom_code = saved


def test_inkling_config_moe_width_from_split():
    try:
        from transformers.models.inkling.configuration_inkling import InklingTextConfig
    except Exception:
        pytest.skip("no Inkling in this transformers")
    from unsloth_zoo.temporary_patches.inkling import patch_inkling_text_config
    saved_init = InklingTextConfig.__init__
    patch_inkling_text_config()
    # The published Inkling-Small shape: MoE width in intermediate_size, dense width separate.
    c = InklingTextConfig(hidden_size = 64, num_hidden_layers = 2, intermediate_size = 2048, dense_intermediate_size = 16384)
    assert c.moe_intermediate_size == 2048
    assert c.intermediate_size == 16384
    # Explicit MoE width wins.
    c = InklingTextConfig(hidden_size = 64, num_hidden_layers = 2, intermediate_size = 2048, dense_intermediate_size = 16384, moe_intermediate_size = 512)
    assert c.moe_intermediate_size == 512
    # No split: defaults untouched.
    c = InklingTextConfig(hidden_size = 64, num_hidden_layers = 2)
    assert c.moe_intermediate_size == 3072
    InklingTextConfig.__init__ = saved_init

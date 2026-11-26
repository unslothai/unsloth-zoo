# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""vLLM >= 0.29 runs block FP8 linears on FP8 Marlin on sm89 (L4), which repacks the weight training shares into
int32 tiles. load_vllm pins vLLM's Triton block FP8 kernel there, and get_vllm_state_dict refuses a repacked weight
with a clear error instead of handing training an int32 tensor."""

import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch


@pytest.fixture
def vu():
    from unsloth_zoo import vllm_utils
    return vllm_utils


class _Kernel:
    supported = True
    @classmethod
    def is_supported(cls, compute_capability = None):
        return cls.supported, None


def _kernel(name, supported):
    return type(name, (_Kernel,), {"supported": supported})


def _install_fake_vllm(monkeypatch, kernels, disabled = (), per_quant = True):
    """The vLLM 0.34 names the selector reads: the CUDA block FP8 kernel list (priority order), the platform,
    VLLM_DISABLED_KERNELS and KernelConfig."""
    cuda = object()
    linear = types.ModuleType("vllm.model_executor.kernels.linear")
    linear._POSSIBLE_FP8_BLOCK_KERNELS = {cuda: kernels}
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = types.SimpleNamespace(_enum = cuda, get_device_capability = lambda: (8, 9))
    envs = types.ModuleType("vllm.envs")
    envs.VLLM_DISABLED_KERNELS = list(disabled)
    config = types.ModuleType("vllm.config")
    fields = {"linear_backend": None}
    if per_quant:
        fields["linear_backend_per_quant"] = None
    config.KernelConfig = type("KernelConfig", (), {"__dataclass_fields__": fields})
    vllm = types.ModuleType("vllm")
    vllm.envs, vllm.config, vllm.platforms = envs, config, platforms
    for name, module in {
        "vllm": vllm, "vllm.envs": envs, "vllm.config": config, "vllm.platforms": platforms,
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.kernels": types.ModuleType("vllm.model_executor.kernels"),
        "vllm.model_executor.kernels.linear": linear,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def _vllm_034_cuda_list(sm89):
    # vLLM 0.34 _POSSIBLE_FP8_BLOCK_KERNELS[CUDA], with what each kernel's is_supported says on sm89 vs sm90 / sm100.
    return [
        _kernel("FlashInferFp8DeepGEMMDynamicBlockScaledKernel", not sm89),
        _kernel("DeepGemmFp8BlockScaledMMKernel", not sm89),
        _kernel("CutlassFp8BlockScaledMMKernel", not sm89),
        _kernel("B12xFp8BlockScaledMMKernel", False),
        _kernel("MarlinFP8ScaledMMLinearKernel", True),
        _kernel("HummingFP8ScaledMMLinearKernel", False),
        _kernel("TritonFp8BlockScaledMMKernel", True),
    ]


def test_sm89_picks_marlin_and_pins_triton_for_block_fp8_only(vu, monkeypatch):
    _install_fake_vllm(monkeypatch, _vllm_034_cuda_list(sm89 = True))
    assert vu._vllm_fp8_block_auto_kernel(89) == "MarlinFP8ScaledMMLinearKernel"
    args, message = vu._vllm_fp8_block_share_backend_args({}, {"kernel_config", "linear_backend"}, 89)
    assert args == {"kernel_config": {"linear_backend_per_quant": {"fp8_block_w8a8": "triton"}}}
    assert "MarlinFP8ScaledMMLinearKernel" in message and "Triton" in message


def test_hopper_and_blackwell_keep_automatic_selection(vu, monkeypatch):
    _install_fake_vllm(monkeypatch, _vllm_034_cuda_list(sm89 = False))
    assert vu._vllm_fp8_block_auto_kernel(100) == "FlashInferFp8DeepGEMMDynamicBlockScaledKernel"
    assert vu._vllm_fp8_block_share_backend_args({}, {"kernel_config"}, 100) == ({}, None)


def test_disabled_kernels_are_skipped_like_vllm(vu, monkeypatch):
    kernels = _vllm_034_cuda_list(sm89 = False)
    _install_fake_vllm(monkeypatch, kernels, disabled = [k.__name__ for k in kernels[:3]])
    assert vu._vllm_fp8_block_auto_kernel(100) == "MarlinFP8ScaledMMLinearKernel"


def test_user_kernel_config_is_merged_and_an_explicit_choice_wins(vu, monkeypatch):
    _install_fake_vllm(monkeypatch, _vllm_034_cuda_list(sm89 = True))
    user = {"enable_flashinfer_autotune": False, "linear_backend_per_quant": {"nvfp4_w4a16": "humming"}}
    args, _ = vu._vllm_fp8_block_share_backend_args({"kernel_config": user}, {"kernel_config"}, 89)
    assert args["kernel_config"] == {
        "enable_flashinfer_autotune": False,
        "linear_backend_per_quant": {"nvfp4_w4a16": "humming", "fp8_block_w8a8": "triton"},
    }
    assert user["linear_backend_per_quant"] == {"nvfp4_w4a16": "humming"}
    for engine_args in (
        {"linear_backend": "marlin"},
        {"kernel_config": {"linear_backend": "cutlass"}},
        {"kernel_config": {"linear_backend_per_quant": {"fp8_block_w8a8": "marlin"}}},
    ):
        assert vu._vllm_fp8_block_share_backend_args(engine_args, {"kernel_config", "linear_backend"}, 89) == ({}, None)


def test_vllm_029_without_per_quant_override_uses_linear_backend(vu, monkeypatch):
    _install_fake_vllm(monkeypatch, _vllm_034_cuda_list(sm89 = True), per_quant = False)
    args, _ = vu._vllm_fp8_block_share_backend_args({}, {"kernel_config", "linear_backend"}, 89)
    assert args == {"linear_backend": "triton"}


def test_vllm_without_the_kernel_list_is_a_no_op(vu, monkeypatch):
    # vLLM <= 0.28 picks Marlin only below sm89, and the oldest have no kernels.linear package at all.
    monkeypatch.setitem(sys.modules, "vllm.model_executor.kernels.linear", None)
    assert vu._vllm_fp8_block_auto_kernel(89) is None
    assert vu._vllm_fp8_block_share_backend_args({}, {"kernel_config"}, 89) == ({}, None)


def test_load_vllm_applies_the_pin_only_when_training_shares_the_weights(vu):
    import inspect
    source = inspect.getsource(vu.load_vllm)
    i = source.index("_vllm_fp8_block_share_backend_args(")
    window = source[source.rindex("\n        if ", 0, i): i]
    assert "is_fp8 or native_fp8 is not None" in window and "not (return_args or use_async)" in window
    # After the FlashInfer autotune default, which skips itself when kernel_config is set, and before the key filter.
    assert source.index('engine_args["enable_flashinfer_autotune"] = False') < i
    assert i < source.index("good_keys = inspect.signature(AsyncEngineArgs if use_async else EngineArgs)")


# A vLLM Qwen3 engine model, one layer, as get_vllm_state_dict walks it.
def _linear(weight, output_sizes = None, quant_method = None, block = None, scale = None):
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(weight, requires_grad = False)
    module.bias = None
    if output_sizes is not None:
        module.output_sizes = output_sizes
        module.logical_widths = output_sizes
    if quant_method is not None:
        module.quant_method = quant_method
        module.weight_block_size = block
        module.weight_scale_inv = torch.nn.Parameter(scale, requires_grad = False)
        module.logical_widths = output_sizes or [weight.shape[0]]
    return module


def _fp8_method(kernel):
    method = type("Fp8LinearMethod", (), {})()
    method.fp8_linear = type(kernel, (), {})()
    method.use_marlin = kernel == "MarlinFP8ScaledMMLinearKernel"
    return method


def _fake_llm(marlin):
    hidden, heads, kv_heads, head_dim, inter, vocab, block = 128, 2, 1, 64, 256, 64, [128, 128]
    q, kv = heads * head_dim, kv_heads * head_dim
    kernel = "MarlinFP8ScaledMMLinearKernel" if marlin else "TritonFp8BlockScaledMMKernel"

    def fp8(n, k, sizes = None):
        weight = torch.randn(n, k).to(torch.float8_e4m3fn)
        scale = torch.ones(-(-n // 128), -(-k // 128))
        if marlin:
            # gptq_marlin_repack: [N, K] float8 -> [K / 16, N * 16 / 4] int32, same bytes.
            weight = weight.view(torch.int32).reshape(k // 16, n * 4)
        return _linear(weight, sizes, _fp8_method(kernel), block, scale)

    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.qkv_proj = fp8(q + 2 * kv, hidden, [q, kv, kv])
    layer.self_attn.o_proj = fp8(hidden, q)
    layer.self_attn.q_norm = _linear(torch.ones(head_dim))
    layer.self_attn.k_norm = _linear(torch.ones(head_dim))
    layer.mlp = torch.nn.Module()
    layer.mlp.gate_up_proj = fp8(2 * inter, hidden, [inter, inter])
    layer.mlp.down_proj = fp8(hidden, inter)
    layer.input_layernorm = _linear(torch.ones(hidden))
    layer.post_attention_layernorm = _linear(torch.ones(hidden))
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.embed_tokens = _linear(torch.randn(vocab, hidden).to(torch.bfloat16))
    model.model.layers = torch.nn.ModuleList([layer])
    model.model.norm = _linear(torch.ones(hidden))
    model.lm_head = _linear(torch.randn(vocab, hidden).to(torch.bfloat16))
    runner = types.SimpleNamespace(model = model)
    engine = types.SimpleNamespace(model_executor = types.SimpleNamespace(driver_worker = types.SimpleNamespace(model_runner = runner)))
    config = types.SimpleNamespace(
        model_type = "qwen3", vocab_size = vocab, num_hidden_layers = 1, hidden_size = hidden,
        num_attention_heads = heads, num_key_value_heads = kv_heads, head_dim = head_dim, intermediate_size = inter,
        tie_word_embeddings = False, quantization_config = {"quant_method": "fp8", "weight_block_size": block},
    )
    return types.SimpleNamespace(llm_engine = engine), config


def test_state_dict_shares_the_float8_weights_triton_keeps(vu):
    llm, config = _fake_llm(marlin = False)
    _, quant_state_dict = vu.get_vllm_state_dict(llm, config = config, load_in_fp8 = True)
    qkv = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[0].self_attn.qkv_proj.weight
    q = quant_state_dict["model.layers.0.self_attn.q_proj.weight"]
    assert q.dtype == torch.float8_e4m3fn and q.shape == (128, 128)
    assert q.untyped_storage().data_ptr() == qkv.untyped_storage().data_ptr()


def test_state_dict_refuses_a_marlin_repacked_weight(vu):
    llm, config = _fake_llm(marlin = True)
    with pytest.raises(RuntimeError, match = r"repacked the FP8 weight of model\.layers\.0\.self_attn\.q_proj.*MarlinFP8ScaledMMLinearKernel.*torch\.int32"):
        vu.get_vllm_state_dict(llm, config = config, load_in_fp8 = True)

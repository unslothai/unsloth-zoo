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

"""CPU backend (unsloth_zoo.cpu): selection rules and the chunked cross entropy."""

import importlib
import os
import math

import pytest
import torch
import torch.nn.functional as F


def _device_type_module():
    return importlib.import_module("unsloth_zoo.device_type")


@pytest.mark.parametrize(
    "env, rocm, nvidia, expected",
    [
        ({}, False, False, True),                                     # plain CPU host
        ({"UNSLOTH_DISABLE_CPU_FALLBACK": "1"}, False, False, False),  # opt-out keeps the error
        ({}, True, False, False),                                     # ROCm host, wrong torch: keep AMD hint
        ({}, False, True, False),                                     # NVIDIA driver, CPU wheel: keep hint
        ({"CUDA_VISIBLE_DEVICES": ""}, False, True, True),            # GPUs hidden on purpose
        ({"CUDA_VISIBLE_DEVICES": "-1"}, True, False, True),
        ({}, False, "ascend", False),                                 # Ascend driver, no torch_npu: keep error
        ({"ROCR_VISIBLE_DEVICES": "-1"}, False, True, False),         # another vendor's mask: NVIDIA still seen
        ({"HIP_VISIBLE_DEVICES": ""}, True, False, True),             # the ROCm mask hides AMD
        ({"CUDA_VISIBLE_DEVICES": "-1"}, False, "ascend", False),     # the CUDA mask does not hide Ascend
        ({"ASCEND_RT_VISIBLE_DEVICES": ""}, False, "ascend", True),
    ],
)
def test_cpu_backend_allowed(monkeypatch, env, rocm, nvidia, expected):
    dt = _device_type_module()
    for key in ("UNSLOTH_DISABLE_CPU_FALLBACK", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                "ASCEND_RT_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising = False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    ascend = nvidia == "ascend"
    nvidia = nvidia is True
    monkeypatch.setattr(dt, "_detect_amd_rocm_runtime", lambda: rocm)
    monkeypatch.setattr(dt, "_intel_gpu_present", lambda: False)
    real_exists = dt.os.path.exists
    monkeypatch.setattr(dt.os.path, "exists", lambda p: nvidia if p == "/proc/driver/nvidia/version"
                        else ascend if p == "/dev/davinci0" else real_exists(p))
    monkeypatch.setattr(dt.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if nvidia and name == "nvidia-smi" else None)
    assert dt.cpu_backend_allowed() is expected


def test_intel_gpu_without_xpu_torch_trains_on_cpu_with_a_hint(monkeypatch):
    dt = _device_type_module()
    for key in ("UNSLOTH_DISABLE_CPU_FALLBACK", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising = False)
    monkeypatch.setattr(dt, "_detect_amd_rocm_runtime", lambda: False)
    monkeypatch.setattr(dt, "_intel_gpu_present", lambda: True)
    real_exists = dt.os.path.exists
    monkeypatch.setattr(dt.os.path, "exists", lambda p: False if p.startswith(("/proc/driver/nvidia", "/usr/lib/wsl", "/dev/davinci")) else real_exists(p))
    monkeypatch.setattr(dt.shutil, "which", lambda name: None)
    with pytest.warns(UserWarning, match = "whl/xpu"):
        assert dt.cpu_backend_allowed() is True


def test_allow_cpu_spoof_still_wins(monkeypatch):
    dt = _device_type_module()
    monkeypatch.setenv("UNSLOTH_ALLOW_CPU", "1")
    assert dt._cpu_fallback() == "cuda"


def _reference(hidden, weight, labels, num_items_in_batch = None):
    logits = (hidden @ weight.t()).float()
    labels = F.pad(labels, (0, 1), value = -100)[..., 1:]
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index = -100, reduction = reduction)
    return loss / num_items_in_batch if num_items_in_batch is not None else loss


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_items", [None, 37])
@pytest.mark.parametrize("weight_grad", [False, True])
def test_chunked_linear_cross_entropy_matches_reference(dtype, num_items, weight_grad):
    from unsloth_zoo.cpu.ops import chunked_linear_cross_entropy
    torch.manual_seed(0)
    hidden = torch.randn(3, 11, 16, dtype = dtype)
    weight = torch.randn(97, 16, dtype = dtype) * 0.2
    labels = torch.randint(0, 97, (3, 11))
    labels[0, :4] = -100
    labels[2, -2:] = -100
    h1 = hidden.clone().requires_grad_(); w1 = weight.clone().requires_grad_(weight_grad)
    h2 = hidden.clone().requires_grad_(); w2 = weight.clone().requires_grad_(weight_grad)
    out = chunked_linear_cross_entropy(h1, w1, labels, num_items_in_batch = num_items, chunk_rows = 5)
    ref = _reference(h2, w2, labels, num_items)
    (out * 1.5).backward(); (ref * 1.5).backward()
    tol = dict(atol = 1e-5, rtol = 1e-5) if dtype == torch.float32 else dict(atol = 2e-2, rtol = 2e-2)
    torch.testing.assert_close(out.float(), ref.float(), **tol)
    torch.testing.assert_close(h1.grad.float(), h2.grad.float(), **tol)
    if weight_grad:
        torch.testing.assert_close(w1.grad.float(), w2.grad.float(), **tol)


def test_chunked_ce_all_ignored_matches_mean_nan():
    from unsloth_zoo.cpu.ops import chunked_linear_cross_entropy
    hidden = torch.randn(1, 4, 8, requires_grad = True)
    weight = torch.randn(10, 8)
    labels = torch.full((1, 4), -100)
    loss = chunked_linear_cross_entropy(hidden, weight, labels)
    assert torch.isnan(loss)
    loss.backward()
    assert torch.equal(hidden.grad, torch.zeros_like(hidden.grad))
    assert chunked_linear_cross_entropy(hidden, weight, labels, num_items_in_batch = 1).item() == 0.0


def test_bf16_policy_override(monkeypatch):
    from unsloth_zoo.cpu import runtime
    runtime.cpu_bf16_supported.cache_clear()
    monkeypatch.setenv("UNSLOTH_CPU_BF16", "0")
    assert runtime.cpu_bf16_supported() is False
    runtime.cpu_bf16_supported.cache_clear()
    monkeypatch.setenv("UNSLOTH_CPU_BF16", "1")
    assert runtime.cpu_bf16_supported() is True
    runtime.cpu_bf16_supported.cache_clear()


def test_bf16_policy_times_bf16_when_cpu_flags_are_unreadable(monkeypatch):
    """Windows has no /proc/cpuinfo: the answer comes from timing bf16 autocast against fp32."""
    from unsloth_zoo.cpu import runtime
    monkeypatch.delenv("UNSLOTH_CPU_BF16", raising = False)
    monkeypatch.setattr(runtime, "cpu_flags", lambda: frozenset())
    for fast in (True, False):
        monkeypatch.setattr(runtime, "_bf16_autocast_is_fast", lambda fast = fast: fast)
        runtime.cpu_bf16_supported.cache_clear()
        assert runtime.cpu_bf16_supported() is fast
    runtime.cpu_bf16_supported.cache_clear()


def test_bnb_cpu_4bit_grads_survive_eval_forward():
    """bitsandbytes repacks Linear4bit weights for its AVX512-BF16 kernel on an eval no-grad forward,
    after which backward is wrong. The CPU loader turns that repacking off; check grads after an eval."""
    from unsloth_zoo.cpu.runtime import cpu_bnb_4bit_supported
    if not cpu_bnb_4bit_supported():
        pytest.skip("no working bitsandbytes CPU 4-bit backend on this host")
    import bitsandbytes as bnb
    from unsloth_zoo.cpu.loader import _disable_bnb_cpu_weight_packing
    torch.manual_seed(0)
    layer = bnb.nn.Linear4bit(256, 128, bias = False, compute_dtype = torch.bfloat16, quant_type = "nf4")
    layer.weight = layer.weight._quantize("cpu")
    model = torch.nn.Sequential(layer)
    assert _disable_bnb_cpu_weight_packing(model) == 1
    dense = bnb.functional.dequantize_4bit(layer.weight.data, layer.weight.quant_state).to(torch.bfloat16)
    model.eval()
    with torch.no_grad():
        model(torch.randn(3, 256, dtype = torch.bfloat16))
    model.train()
    x = torch.randn(8, 256, dtype = torch.bfloat16, requires_grad = True)
    grad = torch.randn(8, 128, dtype = torch.bfloat16)
    (model(x) * grad).sum().backward()
    x_ref = x.detach().clone().requires_grad_()
    ((x_ref @ dense.t()) * grad).sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad, x_ref.grad)


@pytest.fixture
def dtype_policy(monkeypatch):
    """Pin the CPU bf16 / fp16 answers via their env overrides; the cached answers are dropped after."""
    from unsloth_zoo.cpu import runtime

    def pin(bf16, fp16):
        monkeypatch.setenv("UNSLOTH_CPU_BF16", bf16)
        monkeypatch.setenv("UNSLOTH_CPU_FP16", fp16)
        runtime.cpu_bf16_supported.cache_clear()
        runtime.cpu_fp16_supported.cache_clear()
    yield pin
    runtime.cpu_bf16_supported.cache_clear()
    runtime.cpu_fp16_supported.cache_clear()


@pytest.mark.parametrize("bf16, fp16, expected", [
    ("1", "1", (False, True)),   # native bf16 wins: no loss scaling needed
    ("0", "1", (True, False)),   # native fp16 only: fp16 autocast + CPU GradScaler
    ("0", "0", (False, False)),  # neither: fp32
])
def test_fp16_request_picks_the_native_half_precision(dtype_policy, bf16, fp16, expected):
    from types import SimpleNamespace
    from unsloth_zoo.cpu import runtime
    from unsloth_zoo.cpu.trainer import normalize_cpu_training_arguments
    dtype_policy(bf16, fp16)
    args = normalize_cpu_training_arguments(SimpleNamespace(use_cpu = False, fp16 = True, bf16 = False, optim = "adamw_torch"))
    assert (args.fp16, args.bf16) == expected
    assert runtime.cpu_autocast_dtype() == {"1": torch.bfloat16}.get(bf16, torch.float16 if fp16 == "1" else None)


def test_fp16_trainer_gets_cpu_autocast_and_grad_scaler(monkeypatch, tmp_path, dtype_policy):
    """accelerate leaves fp16 on CPU as a silent fp32 run; the CPU backend enables autocast + GradScaler."""
    from unsloth_zoo.cpu.trainer import _patch_cpu_fp16_accelerator, _patch_post_init
    from transformers import Trainer, TrainingArguments
    dtype_policy("0", "1")
    monkeypatch.setattr(Trainer, "create_accelerator_and_postprocess", Trainer.create_accelerator_and_postprocess)
    monkeypatch.setattr(TrainingArguments, "__post_init__", TrainingArguments.__post_init__)
    _patch_cpu_fp16_accelerator(Trainer)
    _patch_post_init(TrainingArguments)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8))
    seen = []
    model[0].register_forward_hook(lambda m, i, o: seen.append(o.dtype))
    args = TrainingArguments(output_dir = str(tmp_path), fp16 = True, report_to = [], max_steps = 1)
    trainer = Trainer(model = model, args = args)
    assert args.fp16 and trainer.accelerator.native_amp
    assert isinstance(trainer.accelerator.scaler, torch.amp.GradScaler)
    prepared = trainer.accelerator.prepare(model)
    prepared(torch.randn(2, 8))
    assert seen == [torch.float16]


def test_cpu_training_arguments_turn_off_packing_and_gpu_optimizers(dtype_policy):
    from types import SimpleNamespace
    from unsloth_zoo.cpu.trainer import normalize_cpu_training_arguments
    dtype_policy("0", "0")
    args = SimpleNamespace(use_cpu = False, fp16 = True, bf16 = False, optim = "paged_adamw_8bit",
                           packing = True, eval_packing = True, padding_free = True, dataloader_pin_memory = True)
    normalize_cpu_training_arguments(args)
    assert args.use_cpu and not args.fp16 and args.optim == "adamw_torch"
    assert not args.packing and not args.eval_packing and not args.padding_free and not args.dataloader_pin_memory


def test_moe_models_keep_their_own_loss():
    from types import SimpleNamespace
    from unsloth_zoo.cpu.ops import _plain_lm_head

    class Model(torch.nn.Module):
        base_model_prefix = "model"

        def __init__(self, config):
            super().__init__()
            self.config = config
            self.model = torch.nn.Linear(4, 4)
            self.lm_head = torch.nn.Linear(4, 10, bias = False)

        def get_output_embeddings(self):
            return self.lm_head

    assert _plain_lm_head(Model(SimpleNamespace(model_type = "llama")))
    assert not _plain_lm_head(Model(SimpleNamespace(model_type = "llama", num_experts = 8)))
    assert not _plain_lm_head(Model(SimpleNamespace(model_type = "xlstm")))


def test_cpu_backend_installs_the_datasets_path_guard():
    folder = pytest.importorskip("datasets.packaged_modules.folder_based_builder.folder_based_builder")
    from unsloth_zoo.cpu.runtime import apply_cpu_security_patches
    assert apply_cpu_security_patches() == ["patch_datasets_folder_metadata_file_name"]
    assert getattr(folder.FolderBasedBuilder._read_metadata, "_unsloth_file_name_guard", False)
    apply_cpu_security_patches()  # idempotent


def test_generate_accepts_a_batch_encoding_as_input_ids():
    """Notebook inference cells pass apply_chat_template(..., return_tensors = "pt") as input_ids."""
    from transformers import BatchEncoding
    from unsloth_zoo.cpu.loader import _unpack_encoding_generate
    seen = {}
    wrapped = _unpack_encoding_generate(lambda self, **kwargs: seen.update(kwargs, owner = self) or "out")
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(ids)
    assert wrapped("model", input_ids = BatchEncoding({"input_ids": ids, "attention_mask": mask}), max_new_tokens = 4) == "out"
    assert seen["input_ids"] is ids and seen["attention_mask"] is mask and seen["max_new_tokens"] == 4
    assert seen["owner"] == "model"
    wrapped("model", input_ids = ids)
    assert seen["input_ids"] is ids


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size = 64, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 1,
                         num_attention_heads = 4, num_key_value_heads = 4, max_position_embeddings = 32,
                         tie_word_embeddings = False)
    return LlamaForCausalLM(config)


def test_deepcopied_models_run_their_own_weights():
    """TRL reference models and merged copies are deepcopies; the patched forward and generate must
    run the copy, not the model they were copied from."""
    import copy
    from unsloth_zoo.cpu.loader import FastCPUModel
    from unsloth_zoo.cpu.ops import patch_chunked_causal_lm_loss
    model = _tiny_llama()
    assert patch_chunked_causal_lm_loss(model)
    FastCPUModel._attach(model)
    clone = copy.deepcopy(model)
    with torch.no_grad():
        clone.lm_head.weight.zero_()
    ids = torch.tensor([[1, 2, 3, 4]])
    uniform = math.log(64)  # zero head -> uniform logits over the vocab
    with torch.no_grad():
        assert abs(clone(input_ids = ids, labels = ids).loss.item() - uniform) < 1e-5
        assert abs(model(input_ids = ids, labels = ids).loss.item() - uniform) > 1e-3
    assert clone.generate.__self__ is clone and model.generate.__self__ is model


def test_for_training_turns_the_kv_cache_back_off():
    from unsloth_zoo.cpu.loader import FastCPUModel
    model = _tiny_llama()
    FastCPUModel.for_inference(model)
    assert model.config.use_cache is True and not model.training
    FastCPUModel.for_training(model)
    assert model.config.use_cache is False and model.training


def test_push_to_hub_merged_forwards_branch_and_pr_options():
    from types import SimpleNamespace
    from unsloth_zoo.cpu.loader import push_to_hub_merged
    calls = []
    model = SimpleNamespace(push_to_hub = lambda repo, **kw: calls.append(("model", repo, kw)))
    tokenizer = SimpleNamespace(push_to_hub = lambda repo, **kw: calls.append(("tokenizer", repo, kw)))
    push_to_hub_merged(model, "user/repo", tokenizer, save_method = "lora", token = "t",
                       revision = "dev", create_pr = True, commit_message = "m", max_shard_size = "1GB")
    for _, repo, kw in calls:
        assert repo == "user/repo"
        assert kw == {"token": "t", "private": None, "revision": "dev", "create_pr": True, "commit_message": "m"}
    assert [c[0] for c in calls] == ["model", "tokenizer"]


def test_lora_targets_include_fused_projections():
    """Phi-3 fuses Q/K/V and gate/up; a Llama-style target list must not leave them frozen."""
    from transformers import Phi3Config, Phi3ForCausalLM
    from unsloth_zoo.cpu.loader import FastCPUModel
    config = Phi3Config(vocab_size = 64, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 1,
                        num_attention_heads = 4, num_key_value_heads = 4, max_position_embeddings = 32,
                        pad_token_id = 0)
    for targets in (None, ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
        peft_model = FastCPUModel.get_peft_model(Phi3ForCausalLM(config), r = 4, target_modules = targets,
                                                 use_gradient_checkpointing = False)
        lora = {name.split(".lora_A")[0].rsplit(".", 1)[-1] for name, _ in peft_model.named_parameters() if ".lora_A" in name}
        assert lora == {"qkv_proj", "o_proj", "gate_up_proj", "down_proj"}


def test_inference_mode_left_pads_and_training_right_pads():
    from types import SimpleNamespace
    from unsloth_zoo.cpu.loader import FastCPUModel
    model = _tiny_llama()
    model._saved_temp_tokenizer = SimpleNamespace(padding_side = "right")
    FastCPUModel.for_inference(model)
    assert model._saved_temp_tokenizer.padding_side == "left"
    FastCPUModel.for_training(model)
    assert model._saved_temp_tokenizer.padding_side == "right"


def test_chunked_sft_step_keeps_token_count():
    from types import SimpleNamespace
    from unsloth_zoo.cpu.trainer import _count_sft_tokens
    trainer = SimpleNamespace(_metrics = {"train": {}, "eval": {}}, _total_train_tokens = 5,
                              accelerator = SimpleNamespace(gather_for_metrics = lambda t: t))
    model = torch.nn.Linear(2, 2).train()
    _count_sft_tokens(trainer, model, {"attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]])})
    assert trainer._total_train_tokens == 10 and trainer._metrics["train"]["num_tokens"] == [10]


def test_chunked_loss_honours_a_custom_ignore_index():
    from unsloth_zoo.cpu.ops import patch_chunked_causal_lm_loss
    model = _tiny_llama()
    assert patch_chunked_causal_lm_loss(model)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    labels = torch.tensor([[1, 2, -1, -1, 5]])
    with torch.no_grad():
        logits = model.lm_head(model.model(input_ids = ids).last_hidden_state)
    expected = F.cross_entropy(logits[0, :-1], labels[0, 1:], ignore_index = -1)
    model._unsloth_cpu_chunked_ce_active = True
    loss = model(input_ids = ids, labels = labels, ignore_index = -1).loss
    torch.testing.assert_close(loss, expected, rtol = 1e-5, atol = 1e-5)


def test_tokenizer_name_loads_the_tokenizer_not_the_model(monkeypatch):
    import transformers
    from types import SimpleNamespace
    from unsloth_zoo.cpu.loader import FastCPUModel
    seen = {}

    def fake_model(name, **kwargs):
        seen["model"] = (name, kwargs)
        return _tiny_llama()

    def fake_tokenizer(name, **kwargs):
        seen["tokenizer"] = name
        return SimpleNamespace(pad_token = "<pad>", pad_token_id = 0, eos_token = "</s>", padding_side = "right")

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fake_model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_tokenizer)
    FastCPUModel.from_pretrained("some/model", tokenizer_name = "some/tokenizer", load_in_4bit = False)
    assert seen["tokenizer"] == "some/tokenizer"
    assert seen["model"][0] == "some/model" and "tokenizer_name" not in seen["model"][1]


def test_kv_cache_is_off_for_training_without_checkpointing():
    from unsloth_zoo.cpu.loader import FastCPUModel
    model = _tiny_llama()
    model.config.use_cache = True
    FastCPUModel._prepare(model, use_gradient_checkpointing = False)
    assert model.config.use_cache is False


def test_partial_llama_style_match_falls_back_to_the_peft_default():
    """MPT-like: Wqkv + out_proj + up_proj/down_proj. up/down alone must not become the target set."""
    import torch.nn as nn
    from unsloth_zoo.cpu.loader import FastCPUModel
    import peft
    captured = {}
    real = peft.LoraConfig.__init__

    def spy(self, *args, **kwargs):
        captured["target_modules"] = kwargs.get("target_modules")
        raise RuntimeError("stop")

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.Wqkv, self.out_proj = nn.Linear(8, 24), nn.Linear(8, 8)
            self.up_proj, self.down_proj = nn.Linear(8, 16), nn.Linear(16, 8)

    model = nn.Sequential(Block())
    peft.LoraConfig.__init__ = spy
    try:
        with pytest.raises(RuntimeError, match = "stop"):
            FastCPUModel.get_peft_model(model, r = 4, use_gradient_checkpointing = False)
    finally:
        peft.LoraConfig.__init__ = real
    assert captured["target_modules"] is None


def test_sft_evaluation_keeps_the_stock_loss_and_metrics():
    """Under no_grad the chunked path is off, so TRL's own compute_loss (entropy / accuracy) must run."""
    from types import SimpleNamespace
    from unsloth_zoo.cpu.trainer import _patch_sft_compute_loss

    class FakeSFT:
        def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None, **kwargs):
            return "trl"

    _patch_sft_compute_loss(FakeSFT)
    model = _tiny_llama()
    model._unsloth_cpu_chunked_ce = True
    trainer = FakeSFT()
    trainer.accelerator = SimpleNamespace(unwrap_model = lambda m: m)
    trainer.args = SimpleNamespace(loss_type = "nll")
    with torch.no_grad():
        assert trainer.compute_loss(model, {"input_ids": torch.tensor([[1, 2]])}) == "trl"


def test_gguf_export_drops_the_unrequested_bf16_intermediate(monkeypatch, tmp_path):
    import unsloth_zoo.llama_cpp as llama_cpp
    from unsloth_zoo.cpu import loader
    made = []

    def convert(prefix, merged, model_dtype, quantization_type, converter_location):
        path = f"{prefix}.{quantization_type.upper()}.gguf"
        open(path, "w").close()
        made.append(path)
        return [path], None

    def quantize(src, dst, method, quantizer_location, print_output):
        open(dst, "w").close()

    monkeypatch.setattr(loader, "save_pretrained_merged", lambda *a, **k: None)
    monkeypatch.setattr(llama_cpp, "install_llama_cpp", lambda print_output = False: ("q", "c"))
    monkeypatch.setattr(llama_cpp, "convert_to_gguf", convert)
    monkeypatch.setattr(llama_cpp, "quantize_gguf", quantize)
    out = loader.save_pretrained_gguf(None, str(tmp_path / "m"), quantization_method = "q4_k_m")
    assert [os.path.basename(p) for p in out] == ["m.Q4_K_M.gguf"]
    assert not os.path.exists(made[0]) and os.path.exists(out[0])
    out = loader.save_pretrained_gguf(None, str(tmp_path / "k"), quantization_method = ["bf16", "q4_k_m"])
    assert sorted(os.path.basename(p) for p in out) == ["k.BF16.gguf", "k.Q4_K_M.gguf"]
    assert all(os.path.exists(p) for p in out)


def test_for_training_switches_gradient_checkpointing_only_when_asked():
    from unsloth_zoo.cpu.loader import FastCPUModel
    model = _tiny_llama()
    FastCPUModel.for_training(model)
    assert not model.is_gradient_checkpointing
    FastCPUModel.for_training(model, use_gradient_checkpointing = True)
    assert model.is_gradient_checkpointing
    FastCPUModel.for_training(model)
    assert model.is_gradient_checkpointing
    FastCPUModel.for_training(model, use_gradient_checkpointing = False)
    assert not model.is_gradient_checkpointing


def test_zero_item_batch_gives_zero_gradients_not_nan():
    from unsloth_zoo.cpu.ops import chunked_linear_cross_entropy
    hidden = torch.randn(2, 4, 8, requires_grad = True)
    weight = torch.randn(16, 8)
    labels = torch.full((2, 4), -100)
    loss = chunked_linear_cross_entropy(hidden, weight, labels, num_items_in_batch = torch.tensor(0))
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(hidden.grad, torch.zeros_like(hidden.grad))


@pytest.mark.parametrize("accelerator, expected", [("mps", "cpu"), ("hpu", None)])
def test_unsupported_active_accelerator_keeps_its_error(monkeypatch, accelerator, expected):
    dt = _device_type_module()
    for key in ("UNSLOTH_DISABLE_CPU_FALLBACK", "UNSLOTH_ALLOW_CPU", "UNSLOTH_FORCE_CPU", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising = False)
    monkeypatch.setattr(dt, "_IS_MLX", False)
    # On Apple Silicon with MLX the module never imported torch; give the non-MLX path its global.
    monkeypatch.setattr(dt, "torch", torch, raising = False)
    monkeypatch.setattr(dt, "_GPU_INIT_SKIPPED", False)
    monkeypatch.setattr(dt.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(dt.torch.xpu, "is_available", lambda: False)
    monkeypatch.setattr(dt, "npu_is_available", lambda: False)
    monkeypatch.setattr(dt.torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(dt.torch.accelerator, "current_accelerator", lambda: accelerator)
    monkeypatch.setattr(dt, "cpu_backend_allowed", lambda: True)
    monkeypatch.setattr(dt, "_amd_installation_hint", lambda: None)
    if expected is None:
        with pytest.raises(NotImplementedError):
            dt.get_device_type.__wrapped__()
    else:
        assert dt.get_device_type.__wrapped__() == expected


def test_create_pr_puts_the_tokenizer_in_the_model_pr():
    from types import SimpleNamespace
    from unsloth_zoo.cpu.loader import push_to_hub_merged
    calls = []
    model = SimpleNamespace(push_to_hub = lambda repo, **kw: calls.append(("model", kw)) or SimpleNamespace(pr_revision = "refs/pr/7"))
    tokenizer = SimpleNamespace(push_to_hub = lambda repo, **kw: calls.append(("tokenizer", kw)))
    push_to_hub_merged(model, "user/repo", tokenizer, save_method = "lora", create_pr = True)
    assert calls[0][1]["create_pr"] is True
    assert calls[1][1]["create_pr"] is False and calls[1][1]["revision"] == "refs/pr/7"


def test_gguf_imatrix_quants_fail_before_the_merge(monkeypatch, tmp_path):
    from unsloth_zoo.cpu import loader
    merged = []
    monkeypatch.setattr(loader, "save_pretrained_merged", lambda *a, **k: merged.append(1))
    with pytest.raises(NotImplementedError, match = "importance matrix"):
        loader.save_pretrained_gguf(None, str(tmp_path / "m"), quantization_method = "iq2_xxs")
    assert merged == []


class _StubTokenizer:
    pad_token, pad_token_id, eos_token, padding_side = "<pad>", 0, "</s>", "right"

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def save_pretrained(self, *a, **k):
        pass


def test_reloaded_lora_checkpoint_is_trainable_and_keeps_its_weights(monkeypatch, tmp_path):
    """FastLanguageModel.from_pretrained("lora_model") must give back a trainable PeftModel with the
    saved adapter weights; get_peft_model on it keeps them; merged_16bit then writes a full model."""
    import transformers
    from types import SimpleNamespace
    from peft import LoraConfig, PeftModel, get_peft_model
    from unsloth_zoo.cpu.loader import FastCPUModel
    base_dir, adapter_dir, merged_dir = tmp_path / "base", tmp_path / "lora", tmp_path / "merged"
    _tiny_llama().save_pretrained(base_dir)
    trained = get_peft_model(transformers.AutoModelForCausalLM.from_pretrained(base_dir),
                             LoraConfig(r = 4, target_modules = ["q_proj", "v_proj"], task_type = "CAUSAL_LM"))
    with torch.no_grad():
        for name, p in trained.named_parameters():
            if "lora_B" in name:
                p.fill_(0.123)
    trained.save_pretrained(adapter_dir)

    tok = _StubTokenizer(64)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tok)
    model, _ = FastCPUModel.from_pretrained(str(adapter_dir), load_in_4bit = False, dtype = torch.float32)
    assert isinstance(model, PeftModel)
    lora_b = [p for n, p in model.named_parameters() if "lora_B" in n]
    assert lora_b and all(p.requires_grad for p in lora_b)
    assert all(torch.allclose(p, torch.full_like(p, 0.123)) for p in lora_b)
    assert FastCPUModel.get_peft_model(model, r = 4) is model
    model.save_pretrained_merged(str(merged_dir), tok, save_method = "merged_16bit")
    assert (merged_dir / "config.json").is_file()


def test_reloaded_lora_with_added_tokens_resizes_the_base_first(monkeypatch, tmp_path):
    import transformers
    from peft import LoraConfig, PeftModel, get_peft_model
    from unsloth_zoo.cpu.loader import FastCPUModel
    base_dir, adapter_dir = tmp_path / "base", tmp_path / "lora"
    _tiny_llama().save_pretrained(base_dir)
    grown = transformers.AutoModelForCausalLM.from_pretrained(base_dir)
    grown.resize_token_embeddings(70)
    trained = get_peft_model(grown, LoraConfig(r = 4, target_modules = ["q_proj"], task_type = "CAUSAL_LM",
                                               modules_to_save = ["embed_tokens", "lm_head"]))
    trained.save_pretrained(adapter_dir)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _StubTokenizer(70))
    model, _ = FastCPUModel.from_pretrained(str(adapter_dir), load_in_4bit = False, dtype = torch.float32)
    assert isinstance(model, PeftModel)
    assert model.get_input_embeddings().num_embeddings == 70


def test_adapter_and_base_revisions_stay_separate(monkeypatch):
    import peft
    import transformers
    from unsloth_zoo.cpu import loader
    seen = {}
    monkeypatch.setattr(loader, "_adapter_base_model", lambda name, token, revision, **k: ("org/base", "base-rev"))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        lambda name, **k: seen.update(base = (name, k.get("revision"))) or _tiny_llama())
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda name, **k: seen.update(tok = (name, k.get("revision"))) or _StubTokenizer(64))

    def fake_peft(model, name, **k):
        seen["adapter"] = (name, k.get("revision"))
        return model
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", fake_peft)
    loader.FastCPUModel.from_pretrained("org/lora", revision = "adapter-rev", load_in_4bit = False)
    assert seen["base"] == ("org/base", "base-rev")
    assert seen["adapter"] == ("org/lora", "adapter-rev")
    assert seen["tok"] == ("org/lora", "adapter-rev")


def test_adapter_in_a_subfolder_is_found_and_loaded_from_it(monkeypatch, tmp_path):
    import peft
    import transformers
    from unsloth_zoo.cpu import loader
    base_dir, repo = tmp_path / "base", tmp_path / "repo"
    _tiny_llama().save_pretrained(base_dir)
    trained = peft.get_peft_model(transformers.AutoModelForCausalLM.from_pretrained(base_dir),
                                  peft.LoraConfig(r = 4, target_modules = ["q_proj"], task_type = "CAUSAL_LM"))
    trained.save_pretrained(repo / "checkpoint")
    seen = {}
    real_load = transformers.AutoModelForCausalLM.from_pretrained
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        lambda name, **k: seen.update(base = (name, k.get("subfolder"))) or real_load(name, **k))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _StubTokenizer(64))
    model, _ = loader.FastCPUModel.from_pretrained(str(repo), subfolder = "checkpoint", load_in_4bit = False,
                                                   dtype = torch.float32)
    assert isinstance(model, peft.PeftModel)
    assert seen["base"] == (str(base_dir), None)


def test_torch_dtype_alias_is_used_and_not_passed_twice(monkeypatch):
    import transformers
    from unsloth_zoo.cpu import loader
    seen = {}
    monkeypatch.setattr(loader, "_adapter_base_model", lambda *a, **k: None)

    def fake_load(name, **k):
        seen.update(k)
        return _tiny_llama()
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fake_load)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _StubTokenizer(64))
    loader.FastCPUModel.from_pretrained("org/model", torch_dtype = torch.float32, load_in_4bit = False)
    assert "torch_dtype" not in seen or "dtype" not in seen
    assert torch.float32 in (seen.get("dtype"), seen.get("torch_dtype"))


def test_fp16_autocast_chunked_loss_keeps_small_gradients_for_the_scaler():
    """Under fp16 autocast the cached hidden gradient must survive until GradScaler's multiplier is applied."""
    from unsloth_zoo.cpu.ops import chunked_linear_cross_entropy
    torch.manual_seed(0)
    hidden = torch.randn(2, 8, 16, requires_grad = True)
    weight = torch.randn(512, 16) * 1e-9
    labels = torch.randint(0, 512, (2, 8))
    reference = hidden.detach().clone().requires_grad_(True)
    (F.cross_entropy((reference[:, :-1] @ weight.t()).reshape(-1, 512), labels[:, 1:].reshape(-1)) * 65536).backward()
    with torch.autocast("cpu", dtype = torch.float16):
        loss = chunked_linear_cross_entropy(hidden, weight, labels)
    (loss * 65536).backward()
    assert reference.grad.abs().max() > 0
    torch.testing.assert_close(hidden.grad, reference.grad, rtol = 1e-3, atol = 1e-9)


def test_json_gradient_checkpointing_kwargs_are_decoded(dtype_policy):
    from types import SimpleNamespace
    from unsloth_zoo.cpu.trainer import normalize_cpu_training_arguments
    dtype_policy("0", "0")
    args = SimpleNamespace(use_cpu = True, fp16 = False, bf16 = False, optim = "adamw_torch", gradient_checkpointing = True,
                           gradient_checkpointing_kwargs = '{"use_reentrant": true}')
    normalize_cpu_training_arguments(args)
    assert args.gradient_checkpointing_kwargs == {"use_reentrant": True}


def test_resize_model_vocab_is_applied(monkeypatch, tmp_path):
    import transformers
    from unsloth_zoo.cpu.loader import FastCPUModel
    _tiny_llama().save_pretrained(tmp_path / "base")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _StubTokenizer(70))
    model, _ = FastCPUModel.from_pretrained(str(tmp_path / "base"), load_in_4bit = False, dtype = torch.float32,
                                            resize_model_vocab = 70)
    assert model.get_input_embeddings().num_embeddings == 70
    assert model.get_output_embeddings().weight.shape[0] == 70


def test_f32_gguf_is_staged_from_fp32_weights(monkeypatch, tmp_path):
    import json
    import unsloth_zoo.llama_cpp as llama_cpp
    from unsloth_zoo.cpu import loader
    staged = {}

    def convert(prefix, merged, model_dtype, quantization_type, converter_location):
        staged["dtype"] = json.load(open(os.path.join(merged, "config.json"))).get("torch_dtype") or \
            json.load(open(os.path.join(merged, "config.json"))).get("dtype")
        path = f"{prefix}.{quantization_type.upper()}.gguf"
        open(path, "w").close()
        return [path], None

    monkeypatch.setattr(llama_cpp, "install_llama_cpp", lambda print_output = False: ("q", "c"))
    monkeypatch.setattr(llama_cpp, "convert_to_gguf", convert)
    out = loader.save_pretrained_gguf(_tiny_llama().float(), str(tmp_path / "m"), quantization_method = "f32")
    assert [os.path.basename(p) for p in out] == ["m.F32.gguf"]
    assert staged["dtype"] == "float32"


def test_learned_position_tables_cap_max_seq_length(monkeypatch, tmp_path):
    import transformers
    from transformers import GPT2Config, GPT2LMHeadModel
    from unsloth_zoo.cpu.loader import FastCPUModel
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _StubTokenizer(64))
    GPT2LMHeadModel(GPT2Config(vocab_size = 64, n_positions = 32, n_embd = 32, n_layer = 1, n_head = 4)).save_pretrained(
        tmp_path / "gpt2")
    model, _ = FastCPUModel.from_pretrained(str(tmp_path / "gpt2"), max_seq_length = 128, load_in_4bit = False)
    assert model.max_seq_length == 32
    # RoPE models keep the requested length.
    _tiny_llama().save_pretrained(tmp_path / "llama")
    model, _ = FastCPUModel.from_pretrained(str(tmp_path / "llama"), max_seq_length = 128, load_in_4bit = False)
    assert model.max_seq_length == 128


def test_fp16_autocast_probe_works_with_the_device_less_api(monkeypatch):
    from unsloth_zoo.cpu import ops

    def device_less(*args):
        if args:
            raise TypeError("is_autocast_enabled() takes 0 positional arguments")
        return False
    with monkeypatch.context() as m:
        m.setattr(torch, "is_autocast_enabled", device_less)
        assert ops._cpu_fp16_autocast() is False
    with torch.autocast("cpu", dtype = torch.float16):
        with monkeypatch.context() as m:
            # torch.autocast itself calls the real API on entry and exit, so stub only inside.
            m.setattr(torch, "is_autocast_enabled", device_less)
            assert ops._cpu_fp16_autocast() is True


def test_f32_in_a_multi_format_gguf_export_is_converted_not_quantized(monkeypatch, tmp_path):
    import unsloth_zoo.llama_cpp as llama_cpp
    from unsloth_zoo.cpu import loader
    converted, quantized = [], []

    def convert(prefix, merged, model_dtype, quantization_type, converter_location):
        converted.append(quantization_type)
        path = f"{prefix}.{quantization_type.upper()}.gguf"
        open(path, "w").close()
        return [path], None

    def quantize(src, dst, method, quantizer_location, print_output):
        quantized.append(method)
        open(dst, "w").close()

    monkeypatch.setattr(llama_cpp, "install_llama_cpp", lambda print_output = False: ("q", "c"))
    monkeypatch.setattr(llama_cpp, "convert_to_gguf", convert)
    monkeypatch.setattr(llama_cpp, "quantize_gguf", quantize)
    out = loader.save_pretrained_gguf(_tiny_llama().float(), str(tmp_path / "m"), quantization_method = ["f32", "q4_k_m"])
    assert sorted(os.path.basename(p) for p in out) == ["m.F32.gguf", "m.Q4_K_M.gguf"]
    assert "f32" in converted and quantized == ["q4_k_m"]


@pytest.mark.parametrize("saved, expected", [(torch.float32, torch.float32), (torch.float16, torch.bfloat16)])
def test_auto_dtype_keeps_the_checkpoint_dtype(monkeypatch, tmp_path, dtype_policy, saved, expected):
    import transformers
    from unsloth_zoo.cpu.loader import FastCPUModel
    dtype_policy("1", "0")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _StubTokenizer(64))
    _tiny_llama().to(saved).save_pretrained(tmp_path / "m")
    for key in ("dtype", "torch_dtype"):
        model, _ = FastCPUModel.from_pretrained(str(tmp_path / "m"), load_in_4bit = False, **{key: "auto"})
        assert model.dtype == expected


def test_a_separate_tokenizer_keeps_the_callers_revision(monkeypatch):
    import peft
    import transformers
    from unsloth_zoo.cpu import loader
    seen = {}
    monkeypatch.setattr(loader, "_adapter_base_model", lambda name, token, revision, **k: ("org/base", "base-rev"))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda name, **k: _tiny_llama())
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda name, **k: seen.update(tok = (name, k.get("revision"))) or _StubTokenizer(64))
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", lambda model, name, **k: model)
    loader.FastCPUModel.from_pretrained("org/lora", tokenizer_name = "org/tok", revision = "v2", load_in_4bit = False)
    assert seen["tok"] == ("org/tok", "v2")

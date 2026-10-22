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
    ],
)
def test_cpu_backend_allowed(monkeypatch, env, rocm, nvidia, expected):
    dt = _device_type_module()
    for key in ("UNSLOTH_DISABLE_CPU_FALLBACK", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising = False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(dt, "_detect_amd_rocm_runtime", lambda: rocm)
    real_exists = dt.os.path.exists
    monkeypatch.setattr(dt.os.path, "exists",
                        lambda p: nvidia if p == "/proc/driver/nvidia/version" else real_exists(p))
    monkeypatch.setattr(dt.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if nvidia and name == "nvidia-smi" else None)
    assert dt.cpu_backend_allowed() is expected


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


def test_cpu_training_arguments_turn_off_packing_and_gpu_optimizers():
    from types import SimpleNamespace
    from unsloth_zoo.cpu.trainer import normalize_cpu_training_arguments
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

    assert _plain_lm_head(Model(SimpleNamespace()))
    assert not _plain_lm_head(Model(SimpleNamespace(num_experts = 8, output_router_logits = False)))


def test_cpu_backend_installs_the_datasets_path_guard():
    folder = pytest.importorskip("datasets.packaged_modules.folder_based_builder.folder_based_builder")
    from unsloth_zoo.cpu.runtime import apply_cpu_security_patches
    assert apply_cpu_security_patches() == ["patch_datasets_folder_metadata_file_name"]
    assert getattr(folder.FolderBasedBuilder._read_metadata, "_unsloth_file_name_guard", False)
    apply_cpu_security_patches()  # idempotent

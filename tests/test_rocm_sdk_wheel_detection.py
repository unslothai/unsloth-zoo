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

"""AMD SDK / Radeon wheels (torch 2.11.0+rocm7.14.0 from repo.amd.com) leave torch.version.hip
unset and carry the ROCm tag only in torch.__version__. Every ROCm gate in the files below
must still see ROCm there. CPU only: the SDK wheel is faked."""

import ast
import json
import os
import pathlib
import subprocess
import sys
import types

import pytest

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_VENDORED_FLA_NO_AUTORUN", "1")

torch = pytest.importorskip("torch")

from unsloth_zoo.rocm_build import is_rocm_torch
from unsloth_zoo.temporary_patches import fla_vendor, moe_utils
from unsloth_zoo import rocm_flex_static
from unsloth_zoo.fused_losses import preference_loss

ROOT = pathlib.Path(__file__).resolve().parents[1]
SDK_VERSION = "2.11.0+rocm7.14.0"


@pytest.fixture
def sdk_wheel(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch, "__version__", SDK_VERSION)


@pytest.fixture
def cuda_wheel(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch, "__version__", "2.11.0+cu130")


def _fake_torch(version, hip = None, archs = ("gfx1151",)):
    return types.SimpleNamespace(
        __version__ = version,
        version = types.SimpleNamespace(hip = hip),
        cuda = types.SimpleNamespace(
            is_available = lambda: True,
            device_count = lambda: len(archs),
            get_device_properties = lambda i: types.SimpleNamespace(gcnArchName = archs[i]),
        ),
    )


def test_is_rocm_torch_reads_hip_or_the_version_tag():
    assert is_rocm_torch(_fake_torch(SDK_VERSION)) is True
    assert is_rocm_torch(_fake_torch("2.9.1+rocm6.4", hip = "6.4.43482")) is True
    assert is_rocm_torch(_fake_torch("2.11.0", hip = "7.13.99004")) is True
    for version in ("2.11.0+cu130", "2.11.0", "2.11.0+cpu", ""):
        assert is_rocm_torch(_fake_torch(version)) is False, version
    assert is_rocm_torch(types.SimpleNamespace()) is False


def test_fla_gates_see_an_sdk_wheel(sdk_wheel, monkeypatch):
    assert fla_vendor._rocm_has_fp32_faulting_gpu(_fake_torch(SDK_VERSION)) is True
    assert fla_vendor._gpu_lacks_dot_instructions(_fake_torch(SDK_VERSION, archs = ("gfx1010",))) is True
    monkeypatch.setitem(sys.modules, "fla.ops.kda", sys.modules.get("fla.ops.kda"))
    assert fla_vendor._withhold_kda_on_rocm() is True
    assert sys.modules["fla.ops.kda"] is None


def test_fla_fp32_route_takes_an_sdk_wheel_gfx11_tensor(sdk_wheel, monkeypatch):
    class _OnGpu(torch.Tensor):
        @property
        def device(self):
            return torch.device("cuda", 0)

    t = torch.zeros(1).as_subclass(_OnGpu)
    monkeypatch.setitem(fla_vendor._rocm_fp32_state, "arch", {0: "gfx1151"})
    assert fla_vendor._rocm_fp32_on_faulting_gpu(t) is True


def test_flex_static_and_fp32_out_gemm_gates_see_an_sdk_wheel(sdk_wheel, monkeypatch):
    monkeypatch.delenv("UNSLOTH_ROCM_FLEX_STATIC", raising = False)
    assert rocm_flex_static.rocm_flex_static_enabled() is True
    grad = types.SimpleNamespace(is_cuda = True, dtype = torch.bfloat16)
    assert moe_utils._mm_fp32_out_ok(grad) is False
    assert moe_utils._unsloth_triton_beats_loop() is True

    # The fp32-out GEMM is skipped, so its per-device verdict is never written.
    monkeypatch.setattr(preference_loss, "_MM_OUT_DTYPE_OK", {})
    hidden, weight = torch.randn(2, 4, dtype = torch.bfloat16), torch.randn(3, 4, dtype = torch.bfloat16)
    preference_loss._linear_float32(hidden, weight, None)
    assert preference_loss._MM_OUT_DTYPE_OK == {}


def test_cuda_wheel_keeps_the_cuda_routes(cuda_wheel, monkeypatch):
    assert fla_vendor._rocm_has_fp32_faulting_gpu(_fake_torch("2.11.0+cu130")) is False
    monkeypatch.delenv("UNSLOTH_ROCM_FLEX_STATIC", raising = False)
    assert rocm_flex_static.rocm_flex_static_enabled() is False
    if moe_utils._MM_OUT_DTYPE:
        assert moe_utils._mm_fp32_out_ok(types.SimpleNamespace(is_cuda = True, dtype = torch.bfloat16)) is True


_IMPORT_FLAGS = """
import json, os, sys, torch
os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_VENDORED_FLA_NO_AUTORUN", "1")
torch.version.hip = None
torch.__version__ = str(torch.__version__).split("+")[0] + sys.argv[1]
from unsloth_zoo.fused_losses import chunked_logprob, fused_linear
from unsloth_zoo.temporary_patches import moe_utils
print("FLAGS " + json.dumps({
    "fp32_out_gemm": chunked_logprob._FP32_OUT_GEMM,
    "mm_out_dtype": chunked_logprob._MM_OUT_DTYPE,
    "hip_tiles": fused_linear._HIP_TILES,
    "is_hip": fused_linear._IS_HIP,
    "bwd_rows": fused_linear._BWD_ROWS,
    "cuda_build": moe_utils._TORCH_IS_CUDA_BUILD,
}))
"""


def _import_flags(local_tag):
    env = dict(os.environ, PYTHONPATH = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.pop("UNSLOTH_FUSED_LINEAR_BWD_ROWS", None)
    out = subprocess.run(
        [sys.executable, "-c", _IMPORT_FLAGS, local_tag], capture_output = True, text = True, env = env, timeout = 600,
    )
    lines = [line for line in out.stdout.splitlines() if line.startswith("FLAGS ")]
    assert lines, out.stderr[-2000:]
    return json.loads(lines[-1][6:])


def test_import_time_rocm_flags_on_an_sdk_wheel():
    # aten::mm.dtype (float32-out 16-bit GEMM) is used on CUDA only, so it is on for a CUDA build whose torch has it.
    mm_dtype_op = hasattr(torch.ops.aten, "mm") and hasattr(torch.ops.aten.mm, "dtype")
    assert _import_flags("+rocm7.14.0") == {
        "fp32_out_gemm": False, "mm_out_dtype": False, "hip_tiles": True, "is_hip": True, "bwd_rows": 1024,
        "cuda_build": False,
    }
    assert _import_flags("+cu130") == {
        "fp32_out_gemm": True, "mm_out_dtype": mm_dtype_op, "hip_tiles": False, "is_hip": False, "bwd_rows": 4096,
        "cuda_build": True,
    }


# Files whose ROCm gates go through is_rocm_torch; a raw torch.version.hip read here is a regression.
_GUARDED = (
    "unsloth_zoo/rocm_flex_static.py",
    "unsloth_zoo/fused_losses/chunked_logprob.py",
    "unsloth_zoo/fused_losses/fused_linear.py",
    "unsloth_zoo/fused_losses/preference_loss.py",
    "unsloth_zoo/score_centering.py",
    "unsloth_zoo/temporary_patches/fla_vendor.py",
    "unsloth_zoo/temporary_patches/moe_utils.py",
)


def _raw_hip_reads(path):
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "hip" and isinstance(node.value, ast.Attribute) \
                and node.value.attr == "version":
            lines.append(node.lineno)
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "getattr" and len(node.args) >= 2 \
                and isinstance(node.args[1], ast.Constant) and node.args[1].value == "hip":
            lines.append(node.lineno)
    return sorted(lines)


def test_guarded_files_never_read_version_hip_directly():
    raw = {p: _raw_hip_reads(ROOT / p) for p in _GUARDED}
    assert not any(raw.values()), f"use unsloth_zoo.device_type.is_rocm_torch: {raw}"


def test_the_raw_read_check_can_fail(tmp_path):
    p = tmp_path / "x.py"
    p.write_text("import torch\na = torch.version.hip is None\nb = getattr(torch.version, 'hip', None)\n", encoding = "utf-8")
    assert _raw_hip_reads(p) == [2, 3]


def _score_centering_head_mode(monkeypatch):
    from unsloth_zoo import score_centering
    from unsloth_zoo.fused_losses import chunked_logprob

    monkeypatch.delenv("UNSLOTH_VLLM_FP32_LM_HEAD", raising = False)
    monkeypatch.delenv("UNSLOTH_LOGPROB_FP32_LOGITS", raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # A 16-bit CUDA head on CPU: the auto default's scope, and no float32-out GEMM (as on every ROCm build).
    monkeypatch.setattr(chunked_logprob, "fp32_lm_head_auto_ok", lambda weight: True)
    monkeypatch.setattr(chunked_logprob, "fp32_out_linear_ok", lambda weight: False)
    head = torch.zeros(4, 3, dtype = torch.bfloat16)
    mode, weight, _ = score_centering._head_projection(head, None)
    return mode, weight


@pytest.mark.skipif(not hasattr(torch.ops.aten.mm, "dtype"), reason = "the fp32 default needs aten::mm.dtype")
def test_score_centering_sdk_wheel_keeps_the_head_dtype(sdk_wheel, monkeypatch):
    # A raw torch.version.hip read took the fp32 CUDA default here, copying the whole head to float32 per call.
    mode, weight = _score_centering_head_mode(monkeypatch)
    assert mode == "plain" and weight.dtype == torch.bfloat16


@pytest.mark.skipif(not hasattr(torch.ops.aten.mm, "dtype"), reason = "the fp32 default needs aten::mm.dtype")
def test_score_centering_cuda_wheel_keeps_the_fp32_default(cuda_wheel, monkeypatch):
    mode, weight = _score_centering_head_mode(monkeypatch)
    assert mode == "fp32" and weight.dtype == torch.float32


def test_truncated_support_auto_is_cuda_only_on_an_sdk_wheel(sdk_wheel, monkeypatch):
    from unsloth_zoo import rl_replacements

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert rl_replacements._grpo_truncated_support_auto_platform() is False


def test_truncated_support_auto_stays_on_for_a_cuda_wheel(cuda_wheel, monkeypatch):
    from unsloth_zoo import rl_replacements

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert rl_replacements._grpo_truncated_support_auto_platform() is True

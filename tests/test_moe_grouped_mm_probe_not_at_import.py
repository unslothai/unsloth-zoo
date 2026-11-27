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

"""The torch._grouped_mm probe must not run while unsloth imports: a worker that picks its card afterwards
(torch.multiprocessing.spawn + set_device) would open a CUDA context on GPU 0 for it. It runs at model load, on the
worker's own card. CPU only: torch.cuda and the kernel probes are mocked, in a child process (the patches it
applies are process-wide)."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CHILD = textwrap.dedent(
    """
    import json, os, sys, types
    sys.path.insert(0, sys.argv[1])
    import torch
    import unsloth_zoo.temporary_patches.moe_utils as M
    import unsloth_zoo.temporary_patches.qwen3_vl_moe as qwen3_vl_moe
    assert os.path.abspath(M.__file__).startswith(sys.argv[1] + os.sep), M.__file__

    spec = types.SimpleNamespace(_initializing = True)  # `import unsloth` in progress
    fake = types.ModuleType("unsloth"); fake.__spec__ = spec; sys.modules["unsloth"] = fake
    os.environ.pop("UNSLOTH_MOE_BACKEND", None)
    current = [0]
    props = types.SimpleNamespace(name = "B200", major = 10, minor = 0, gcnArchName = None)
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 2
    torch.cuda.current_device = lambda: current[0]
    torch.cuda.get_device_properties = lambda index: props
    if hasattr(torch, "xpu"):
        torch.xpu.is_available = lambda: False
    probed = []
    M._TORCH_GROUPED_MM_AVAILABLE = True
    M.is_mlx_available = lambda: False
    M._load_cached_moe_utils_module = lambda: None
    M._grouped_mm_survives_out_of_process = lambda device: True
    M._grouped_mm_runs_on = lambda device: probed.append(str(device)) or True
    M._transposed_view_probe = lambda device: True
    qwen3_vl_moe.patch_param_wrapper_for_moe = lambda: None
    M.select_moe_backend.cache_clear()

    out = {}
    # During the import: the experts forward is resolved (Qwen3 MoE, Llama 4) and Qwen3-VL MoE is patched.
    M.get_forward_moe_backend()
    qwen3_vl_moe.patch_qwen3_vl_moe()
    out["import"] = {"probed": list(probed), "home": M._GROUPED_MM_HOME, "flag": M._TORCH_GROUPED_MM_SUPPORTED}

    # Import done; the spawned worker picks card 1, then loads a model (the temporary patches run again).
    spec._initializing = False
    current[0] = 1
    getattr(M, "warm_moe_backend_selection", M.select_moe_backend)()
    getattr(M, "warm_moe_backend_selection", M.select_moe_backend)()
    out["load"] = {"probed": list(probed), "home": M._GROUPED_MM_HOME, "backend": M.select_moe_backend()}
    print("RESULT " + json.dumps(out))
    """
)


def test_grouped_mm_probe_waits_for_model_load_on_the_worker_card():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "")
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, ROOT], capture_output = True, text = True, env = env, timeout = 600, cwd = ROOT,
    )
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line is not None, proc.stdout[-2000:] + proc.stderr[-4000:]
    out = json.loads(line[len("RESULT "):])
    assert out["import"] == {"probed": [], "home": None, "flag": None}, out
    # Probed once (then cached), on the worker's own card.
    assert out["load"] == {"probed": ["cuda:1"], "home": ["cuda", 1], "backend": "grouped_mm"}, out

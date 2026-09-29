# SPDX-License-Identifier: AGPL-3.0-only
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

"""A 16bit gpt-oss load after a 4bit one (or the reverse) in one process must not reuse the compiled module's classes."""
import json
import os
import subprocess
import sys
import textwrap


_RUNNER = textwrap.dedent(
    """
    import os, sys, json, types
    os.environ["UNSLOTH_ALLOW_CPU"] = "1"
    os.environ["UNSLOTH_MODEL_NAME"] = "gpt_oss,"
    os.environ["UNSLOTH_COMPILE_LOCATION"] = sys.argv[1]
    import torch
    if not torch.cuda.is_available():
        import torch.cuda.memory as _cm
        _cm.mem_get_info = lambda *a, **k: (0, 80 * 1024**3)
        torch.cuda.device_count = lambda: 1
        torch.cuda.get_device_capability = lambda *a, **k: (8, 0)
        class _P:
            major = 8; minor = 0; total_memory = 80 * 1024**3
            multi_processor_count = 108; name = "stub"
        torch.cuda.get_device_properties = lambda *a, **k: _P()
    import transformers.models.gpt_oss.modeling_gpt_oss as modeling
    from unsloth_zoo.temporary_patches import gpt_oss as G

    stock_experts, stock_router = modeling.GptOssExperts, modeling.GptOssTopKRouter
    # Stand-in for the compiled module: its GptOssMLP resolves the classes through its own globals.
    compiled = types.ModuleType("unsloth_compiled_module_gpt_oss")
    exec("class GptOssMLP:\\n    def __init__(self):\\n        self.classes = (GptOssExperts, GptOssTopKRouter)\\n", compiled.__dict__)
    compiled.GptOssExperts, compiled.GptOssTopKRouter = stock_experts, stock_router
    modeling.GptOssMLP = compiled.GptOssMLP
    res = {}

    # Compiled for 16bit first, then a 4bit load.
    os.environ["UNSLOTH_MODEL_NAME"] = "gpt_oss,_load_in_4bit_"
    G.patch_gpt_oss_bnb4bit_auto()
    res["after_4bit"] = compiled.GptOssMLP().classes == (G.GptOssExpertsBnb4bit, G.GptOssTopKRouter)

    # Back to 16bit, with the env flag cleared the way a fresh environment or a test harness would.
    os.environ.pop("UNSLOTH_GPT_OSS_BNB4BIT_PATCHED", None)
    os.environ["UNSLOTH_MODEL_NAME"] = "gpt_oss,"
    G.patch_gpt_oss_bnb4bit_auto()
    res["after_16bit"] = compiled.GptOssMLP().classes == (stock_experts, stock_router)
    res["transformers_restored"] = modeling.GptOssExperts is stock_experts

    # A compiled class of the flavor being loaded is kept, not replaced by the uncompiled one.
    class CompiledBnbRouter:
        def __init__(self):
            self.linear = None
    compiled.GptOssTopKRouter = CompiledBnbRouter
    compiled.GptOssExperts = G.GptOssExpertsBnb4bit
    os.environ["UNSLOTH_MODEL_NAME"] = "gpt_oss,_load_in_4bit_"
    G.patch_gpt_oss_bnb4bit_auto()
    res["matching_compiled_kept"] = compiled.GptOssTopKRouter is CompiledBnbRouter
    print("RESULT " + json.dumps(res))
    """
)


def test_flavor_switch_rebinds_compiled_module_classes(tmp_path):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(tmp_path)],
        capture_output = True, text = True, timeout = 600, env = env,
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-2000:] + proc.stderr[-4000:]
    res = json.loads(lines[-1][len("RESULT "):])
    assert res == {
        "after_4bit": True, "after_16bit": True,
        "transformers_restored": True, "matching_compiled_kept": True,
    }, res

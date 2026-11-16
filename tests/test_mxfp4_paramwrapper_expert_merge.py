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

"""merged_16bit of an MXFP4 GPT-OSS checkpoint must fold PEFT target_parameters expert LoRA
(ParamWrapper on mlp.experts) into the dequantized experts; PEFT's own merge is the oracle."""
import os
import subprocess
import sys
import tempfile
import textwrap

import pytest

_RUNNER = textwrap.dedent(
    """
    import os, sys, tempfile
    from collections import defaultdict
    import torch, torch.nn as nn
    from safetensors.torch import save_file, safe_open
    from peft import LoraConfig, get_peft_model
    import unsloth_zoo.saving_utils as sv

    E, H, I = (int(x) for x in sys.argv[1:4])
    torch.manual_seed(0)
    pre = "model.layers.0.mlp.experts"

    def packed(rows, cols):
        # blocks (E, rows, G, B) dequantize to (E, cols, rows) with cols = G * B * 2
        B = 16 if cols % 32 == 0 else 8
        G = cols // (2 * B)
        return (torch.randint(0, 256, (E, rows, G, B), dtype = torch.uint8),
                torch.randint(124, 130, (E, rows, G), dtype = torch.uint8))

    (gb, gs), (db, ds) = packed(2 * I, H), packed(H, I)
    shard = {pre + ".gate_up_proj_blocks": gb, pre + ".gate_up_proj_scales": gs,
             pre + ".down_proj_blocks": db, pre + ".down_proj_scales": ds}
    fname = "model.safetensors"

    def run(lora_weights):
        d = tempfile.mkdtemp(prefix = "mxfp4pw_")
        save_file(shard, os.path.join(d, fname), metadata = {"format": "pt"})
        count, _ = sv._merge_and_overwrite_lora_mxfp4(d, fname, lora_weights, torch.bfloat16, "GptOssForCausalLM")
        with safe_open(os.path.join(d, fname), "pt") as f:
            return count, {k: f.get_tensor(pre + "." + k).float() for k in ("gate_up_proj", "down_proj")}

    _, W0 = run(defaultdict(lambda: None))
    assert tuple(W0["gate_up_proj"].shape) == (E, H, 2 * I) and tuple(W0["down_proj"].shape) == (E, I, H), \
        {k: tuple(v.shape) for k, v in W0.items()}

    class Experts(nn.Module):
        is_transposed = True   # GptOssExperts: (E, in, out), which PEFT reads off this flag
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(W0["gate_up_proj"].clone())
            self.down_proj = nn.Parameter(W0["down_proj"].clone())
    class Mlp(nn.Module):
        def __init__(self):
            super().__init__(); self.experts = Experts()
    class Layer(nn.Module):
        def __init__(self):
            super().__init__(); self.mlp = Mlp()
    class Inner(nn.Module):
        def __init__(self):
            super().__init__(); self.layers = nn.ModuleList([Layer()])
    class Top(nn.Module):
        def __init__(self):
            super().__init__(); self.model = Inner()

    peft_model = get_peft_model(Top(), LoraConfig(r = 4, lora_alpha = 8, target_modules = [],
                                target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]))
    lora_weights = defaultdict(lambda: None)
    for name, mod in peft_model.base_model.model.named_modules():
        if hasattr(mod, "parameter_name") and hasattr(mod, "lora_A"):
            nn.init.normal_(mod.lora_B["default"].weight, std = 0.5)
            lora_weights[name] = sv.LoraStats(
                module = None, lora_A = mod.lora_A["default"].weight.detach().clone(),
                lora_B = mod.lora_B["default"].weight.detach().clone(),
                alpha = mod.scaling["default"], parameter_name = mod.parameter_name)
    assert sorted(lora_weights) == [pre, pre + ".base_layer"], sorted(lora_weights)
    merged = peft_model.merge_and_unload().model.layers[0].mlp.experts

    count, W1 = run(lora_weights)
    print("COUNT", count, sv.__file__)
    for k in ("gate_up_proj", "down_proj"):
        ref = getattr(merged, k).detach().float()
        delta = (ref - W0[k]).norm()
        err = (W1[k] - ref).norm()
        print(k, "delta", float(delta), "err", float(err))
        assert delta > 0.1, k
        assert err <= 0.02 * delta + 2 ** -7 * ref.norm(), (k, float(err), float(delta))
    assert count == 2, count
    print("OK")
    """
)


# (2, 32, 16): gate_up is square (H == 2I), so only the GPT-OSS layout hint can orient the delta.
@pytest.mark.parametrize("dims", [(2, 64, 16), (2, 32, 16)])
def test_mxfp4_merge_folds_paramwrapper_expert_lora(dims):
    pytest.importorskip("transformers.integrations.mxfp4")
    peft = pytest.importorskip("peft")
    from peft import LoraConfig
    if "target_parameters" not in LoraConfig.__dataclass_fields__:
        pytest.skip(f"peft {peft.__version__} has no target_parameters")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_IS_PRESENT = "1")
    # Not the repo root as cwd: `python -c` would import unsloth_zoo from there, not from PYTHONPATH.
    r = subprocess.run([sys.executable, "-c", _RUNNER, *map(str, dims)], capture_output = True, text = True,
                       env = env, timeout = 600, cwd = tempfile.gettempdir())
    assert r.returncode == 0 and "OK" in r.stdout, f"stdout={r.stdout[-3000:]}\nstderr={r.stderr[-3000:]}"

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

"""float16 grouped GEMM under torch.compile. The eager torch._grouped_mm kernel takes float16, but its fake impl
rejects it, so a compiled float16 MoE frame failed to trace and (Unsloth sets suppress_errors) silently ran eager.
_grouped_mm_with_backward_fix now routes compiled float16 calls through an opaque custom op; bf16 is untouched."""

import json
import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not hasattr(torch, "_grouped_mm"),
                                reason = "needs CUDA torch._grouped_mm")

from unsloth_zoo.temporary_patches import moe_utils as MU


def _supported():
    if not MU._check_torch_grouped_mm_supported():
        pytest.skip("torch._grouped_mm unsupported on this device")


def _case(dtype, seed = 0):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    counts = torch.tensor([10, 0, 18, 12], device = "cuda", dtype = torch.int32)   # one empty expert
    offs = torch.cumsum(counts, 0, dtype = torch.int32)
    M, K, N, R = int(counts.sum()), 32, 48, 16
    x = (torch.randn(M, K, device = "cuda", generator = g) * 0.5).to(dtype).requires_grad_(True)
    base = (torch.randn(4, N, K, device = "cuda", generator = g) * 0.1).to(dtype)           # frozen [E, out, in]
    lora = (torch.randn(4, K, R, device = "cuda", generator = g) * 0.1).to(dtype).requires_grad_(True)
    return x, base, lora, offs


def _fn(x, base, lora, offs):
    y = MU._grouped_mm_with_backward_fix(x, base.transpose(-2, -1), offs)   # transposed view, as the MoE path
    h = MU._grouped_mm_with_backward_fix(x, lora, offs)
    return y, h


def _run(fn, dtype):
    x, base, lora, offs = _case(dtype)
    y, h = fn(x, base, lora, offs)
    (y.float().square().sum() + h.float().sum()).backward()
    return y.detach(), h.detach(), x.grad.detach(), lora.grad.detach()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_compiled_grouped_mm_matches_eager_fullgraph(dtype):
    _supported()
    torch._dynamo.reset()
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(str(gm.graph))
        from torch._dynamo.backends.common import aot_autograd
        return aot_autograd(fw_compiler = lambda g, _: g, bw_compiler = lambda g, _: g)(gm, example_inputs)

    eager = _run(_fn, dtype)
    compiled = _run(torch.compile(_fn, fullgraph = True, backend = backend), dtype)
    names = ("out", "lora_out", "grad_x", "grad_lora")
    for name, a, b in zip(names, eager, compiled):
        assert torch.isfinite(b).all(), name
        if name in ("out", "lora_out"):
            assert torch.equal(a, b), f"{name}: compiled differs from eager"
        else:
            torch.testing.assert_close(b.float(), a.float(), rtol = 1e-2, atol = 1e-2, msg = name)
    # the empty expert gets no LoRA gradient
    assert compiled[3][1].abs().max() == 0
    joined = "\n".join(graphs)
    if dtype == torch.float16:
        assert "grouped_mm_fp16" in joined, "compiled float16 call did not take the custom op"
    else:
        assert "grouped_mm_fp16" not in joined, "bf16 must keep aten._grouped_mm"


def test_eager_float16_does_not_use_custom_op(monkeypatch):
    _supported()
    calls = []
    monkeypatch.setattr(MU, "_GROUPED_MM_FP16_OP", lambda *a: calls.append(1))
    _run(_fn, torch.float16)
    assert calls == []


def test_custom_op_backward_matches_per_group_fp32():
    """dX and dW of the op against an fp32 per-group reference, including the empty group."""
    _supported()
    op = MU._GROUPED_MM_FP16_OP
    assert op is not None
    x, base, lora, offs = _case(torch.float16, seed = 3)
    out = op(x, lora, offs)
    gy = torch.randn_like(out)
    out.backward(gy)
    ends = [0] + offs.tolist()
    ref_out = torch.cat([x[s:e].float() @ lora[i].float() for i, (s, e) in enumerate(zip(ends, ends[1:]))])
    ref_gx = torch.cat([gy[s:e].float() @ lora[i].float().t() for i, (s, e) in enumerate(zip(ends, ends[1:]))])
    ref_gw = torch.stack([x[s:e].float().t() @ gy[s:e].float() for s, e in zip(ends, ends[1:])])
    torch.testing.assert_close(out.float(), ref_out, rtol = 2e-3, atol = 2e-3)
    torch.testing.assert_close(x.grad.float(), ref_gx, rtol = 2e-3, atol = 2e-3)
    torch.testing.assert_close(lora.grad.float(), ref_gw, rtol = 2e-3, atol = 2e-3)


_CHILD = r'''
import os, sys, json, copy
dtype_name, out_dir = sys.argv[1], sys.argv[2]
os.environ["UNSLOTH_COMPILE_LOCATION"] = os.path.join(out_dir, "unsloth_compiled_cache")
import unsloth
from unsloth import FastModel
import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
dtype = getattr(torch, dtype_name)
ckpt = os.path.join(out_dir, "tiny")
cfg = AutoConfig.for_model("mixtral", vocab_size = 512, bos_token_id = 0, eos_token_id = 1, pad_token_id = 2,
                           hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2, num_attention_heads = 4,
                           num_key_value_heads = 2, num_local_experts = 4, num_experts_per_tok = 2)
torch.manual_seed(0)
AutoModelForCausalLM.from_config(cfg).to(dtype).save_pretrained(ckpt)
tk = Tokenizer(WordLevel({f"t{i}": i for i in range(512)}, unk_token = "t3")); tk.pre_tokenizer = Whitespace()
PreTrainedTokenizerFast(tokenizer_object = tk, bos_token = "t0", eos_token = "t1", pad_token = "t2",
                        unk_token = "t3").save_pretrained(ckpt)
model, _ = FastModel.from_pretrained(ckpt, max_seq_length = 128, dtype = dtype, load_in_4bit = False,
                                     load_in_16bit = True, device_map = {"": 0})
model = FastModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0, bias = "none",
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing = "unsloth", random_state = 3407)
torch.manual_seed(1)
for n, p in model.named_parameters():
    if "lora_B" in n:
        p.data.normal_(0, 0.02)
# A frame that fails to trace must fail the test, not fall back to eager.
torch._dynamo.config.suppress_errors = False
from unsloth_zoo.temporary_patches import moe_utils
from torch._dynamo.utils import counters
counters.clear()
model.train()
ids = torch.randint(4, 500, (2, 48), device = "cuda")
err = None
try:
    with torch.autocast("cuda", dtype = dtype):
        loss = model(input_ids = ids, labels = ids).loss
    loss.backward()
except Exception as e:
    err = f"{type(e).__name__}: {str(e)[:400]}"
grads = [float(p.grad.abs().max()) if p.grad is not None else 0.0
         for n, p in model.named_parameters() if "lora_" in n and ".experts." in n] if err is None else []
print("RESULT " + json.dumps({"err": err, "grads": grads, "frames": dict(counters["frames"]),
                              "loss": None if err else float(loss)}))
'''


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_compiled_mixtral_expert_lora_traces(tmp_path, dtype_name):
    """End to end: Mixtral expert LoRA step compiles with suppress_errors off (float16 used to raise in the fake
    impl of aten._grouped_mm); expert LoRA grads finite and non-zero."""
    _supported()
    script = tmp_path / "child.py"
    script.write_text(_CHILD)
    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join([repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    env.pop("UNSLOTH_MOE_BACKEND", None)
    proc = subprocess.run([sys.executable, str(script), dtype_name, str(tmp_path)],
                          capture_output = True, text = True, timeout = 1200, env = env)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line is not None, proc.stdout[-3000:] + proc.stderr[-3000:]
    r = json.loads(line[len("RESULT "):])
    assert r["err"] is None, r["err"]
    if not r["grads"]:
        # transformers 4.x keeps Mixtral experts as per-expert ModuleLists: no stacked expert LoRA to check.
        pytest.skip(reason = "no stacked expert LoRA in this transformers")
    assert r["grads"] and all(0 < g < float("inf") for g in r["grads"]), r["grads"]

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

"""grpo_accumulated_loss under DeepSpeed ZeRO-3 reads the output head after the model forward.

ZeRO-3 gathers a parameter only around its own module's forward; afterwards ``lm_head.weight`` is a
0-element placeholder (``ds_status`` NOT_AVAILABLE). The GRPO scoring head never calls lm_head, so it
used to read that placeholder (``lm_head.shape[1]`` IndexError, or a matmul against nothing). A fake
ZeRO-3 parameter (the attributes DeepSpeed sets plus a ``GatheredParameters`` that restores the data)
stands in for DeepSpeed, so this runs on CPU without it. Loss and LoRA-side gradients must match an
unpartitioned reference; a trainable head is refused rather than silently left without a gradient.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

_WORKER = textwrap.dedent(
    r"""
    import contextlib, json, sys, types
    import torch

    bias, train_head, out = sys.argv[1] == "1", sys.argv[2] == "1", sys.argv[3]
    gathers = []

    def _fake_deepspeed():
        @contextlib.contextmanager
        def GatheredParameters(params, modifier_rank = None, **kwargs):
            gathers.append(modifier_rank)
            for p in params:
                p.data = p._fake_full
                p.ds_status = "AVAILABLE"
            try:
                yield
            finally:
                for p in params:
                    p.data = torch.empty(0, dtype = p._fake_full.dtype)
                    p.ds_status = "NOT_AVAILABLE"
        ds = types.ModuleType("deepspeed")
        ds.zero = types.SimpleNamespace(GatheredParameters = GatheredParameters)
        return ds

    def partition(p, ds_id):
        p._fake_full = p.data.clone()
        p.ds_id = ds_id
        p.ds_status = "NOT_AVAILABLE"
        p.ds_persist = False
        p.data = torch.empty(0, dtype = p.dtype)

    import unsloth_zoo.rl_replacements as rl
    from transformers.modeling_outputs import CausalLMOutput
    # After the imports, which probe for the real package; the helper imports deepspeed lazily.
    sys.modules["deepspeed"] = _fake_deepspeed()

    V, H = 32, 8
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.embed = torch.nn.Embedding(V, H)
            self.proj = torch.nn.Linear(H, H)
            self.lm_head = torch.nn.Linear(H, V, bias = bias)
            self.lm_head.weight.requires_grad_(train_head)
            if bias:
                self.lm_head.bias.requires_grad_(train_head)
        def get_output_embeddings(self):
            return self.lm_head
        def forward(self, input_ids = None, attention_mask = None, logits_to_keep = None, **kwargs):
            hidden = self.proj(self.embed(input_ids))
            if logits_to_keep:
                hidden = hidden[:, -logits_to_keep:]
            return CausalLMOutput(logits = hidden)

    def trainer_for(model):
        return types.SimpleNamespace(
            args = types.SimpleNamespace(unsloth_grpo_mini_batch = 1, unsloth_logit_chunk_multiplier = 1),
            processing_class = types.SimpleNamespace(pad_token_id = 0),
            model = model,
            accelerator = types.SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None),
            use_vllm = False, _autocast_dtype = None, beta = 0.0,
        )

    g = torch.Generator().manual_seed(100)
    batches = []
    for _ in range(2):
        ids = torch.randint(1, V, (3, 7), generator = g)
        ids[2, 0] = 0
        batches.append((ids, torch.randn(3, generator = g)))

    def run(model, trainer):
        losses = []
        for ids, adv in batches:
            mask = torch.ones(3, 4, dtype = torch.long)
            loss = rl.grpo_accumulated_loss(
                trainer, ids, (ids != 0).long(), 4, mask, adv, None, None,
                loss_type = "dapo", num_items_in_batch = int(mask.sum()), num_processes = 1,
                current_gradient_accumulation_steps = 2, max_completion_length = 4,
            )[0]
            (loss / 2).backward()
            losses.append(loss.item())
        return losses

    model = Toy()
    partition(model.lm_head.weight, 1)
    if bias:
        partition(model.lm_head.bias, 2)
    result = {}
    try:
        losses = run(model, trainer_for(model))
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    else:
        ref = Toy()
        ref_losses = run(ref, trainer_for(ref))
        result["loss_err"] = max(abs(a - b) for a, b in zip(losses, ref_losses))
        result["grad_err"] = max(
            (p.grad - rp.grad).abs().max().item()
            for p, rp in zip(model.proj.parameters(), ref.proj.parameters())
        )
        result["placeholder_after"] = model.lm_head.weight.numel() == 0
        result["read_only_gathers"] = all(r is None for r in gathers) and len(gathers) > 0
        result["n_gathers"] = len(gathers)
    with open(out, "w") as f:
        json.dump(result, f)
    """
)


def _run(tmp_path, bias, train_head):
    script = tmp_path / "worker.py"
    script.write_text(_WORKER)
    env = dict(os.environ)
    for key in ("UNSLOTH_RETURN_HIDDEN_STATES", "UNSLOTH_GRPO_LOGPROB_RECOMPUTE"):
        env.pop(key, None)
    env.update(
        CUDA_VISIBLE_DEVICES = "",
        TORCHDYNAMO_DISABLE = "1",
        UNSLOTH_IS_PRESENT = "1",
        UNSLOTH_ZOO_DISABLE_GPU_INIT = "1",
        UNSLOTH_GRPO_SEQ_PACKING = "0",
        UNSLOTH_GRPO_PREFIX_GROUPER = "0",
        PYTHONPATH = os.pathsep.join([str(REPO), env.get("PYTHONPATH", "")]),
    )
    out = tmp_path / "r.json"
    proc = subprocess.run(
        [sys.executable, str(script), str(int(bias)), str(int(train_head)), str(out)],
        env = env, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True, timeout = 600,
    )
    assert out.exists(), proc.stdout[-4000:]
    return json.loads(out.read_text())


@pytest.mark.parametrize("bias", [False, True])
def test_grpo_loss_gathers_the_released_zero3_head(tmp_path, bias):
    result = _run(tmp_path, bias, train_head = False)
    assert "error" not in result, result["error"]
    assert result["loss_err"] < 1e-6, result
    assert result["grad_err"] < 1e-6, result
    # Read-only gathers (modifier_rank None), one per loss call, and the head is released again after.
    assert result["read_only_gathers"], result
    assert result["n_gathers"] == 2 * (1 + bias), result
    assert result["placeholder_after"], result


def test_grpo_loss_refuses_a_trainable_zero3_head(tmp_path):
    result = _run(tmp_path, bias = False, train_head = True)
    assert "NotImplementedError" in result.get("error", ""), result
    assert "ZeRO-3" in result["error"], result


def test_gather_helper_leaves_ordinary_tensors_alone():
    import torch
    from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_zero3_gathered, unsloth_zero3_partitioned

    w = torch.nn.Parameter(torch.randn(4, 3))
    assert unsloth_zero3_gathered(w) is w
    assert unsloth_zero3_gathered(None) is None
    assert not unsloth_zero3_partitioned(w)
    # A persistent ZeRO-3 parameter (below stage3_param_persistence_threshold) is never released.
    w.ds_id, w.ds_status, w.ds_persist = 1, "AVAILABLE", True
    assert not unsloth_zero3_partitioned(w)
    assert unsloth_zero3_gathered(w) is w

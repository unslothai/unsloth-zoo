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

"""CPU backend, MoE router aux loss under gradient accumulation: a model FastCPUModel loads weights the
router_aux_loss_coef x aux_loss term its stock forward adds by the micro-batch's token share,
local / num_items_in_batch, as zoo's GPU forwards do, so G accumulated micro-batches carry it once, not G
times. Without a count (evaluation, generation) the forward is unchanged. Fresh interpreter: the CPU route
keeps the stock transformers forward, which a process that installed zoo's GPU forwards would not."""

import json
import os
import subprocess
import sys
import tempfile

import pytest

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3MoeForCausalLM"):
    pytest.skip("transformers without Qwen3-MoE", allow_module_level = True)

_CODE = r"""
import copy, json, sys, torch, transformers
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from unsloth_zoo.cpu.loader import FastCPUModel

COEF = 0.5

class Tok:
    pad_token, pad_token_id, eos_token, padding_side = "<pad>", 0, "</s>", "right"
    def __len__(self):
        return 64
    def save_pretrained(self, *a, **k):
        pass

transformers.AutoTokenizer.from_pretrained = lambda *a, **k: Tok()
torch.manual_seed(0)
config = Qwen3MoeConfig(
    vocab_size = 64, hidden_size = 32, intermediate_size = 64, moe_intermediate_size = 32,
    num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 8,
    num_experts = 4, num_experts_per_tok = 2, decoder_sparse_step = 1, max_position_embeddings = 64,
    output_router_logits = True, router_aux_loss_coef = COEF, tie_word_embeddings = False,
)
Qwen3MoeForCausalLM(config).save_pretrained(sys.argv[1])
model, _ = FastCPUModel.from_pretrained(sys.argv[1], load_in_4bit = False, dtype = torch.float32)

def terms(m, ids, labels, n):
    with torch.no_grad():
        out = m(input_ids = ids, labels = labels, num_items_in_batch = n)
        m.router_aux_loss_coef = 0.0
        try:
            ce = m(input_ids = ids, labels = labels, num_items_in_batch = n).loss
        finally:
            m.router_aux_loss_coef = COEF
    return float(out.loss - ce), float(out.aux_loss)

g = torch.Generator().manual_seed(1)
ids = torch.randint(1, 64, (2, 9), generator = g)
labels = ids.clone()
labels[0, :3] = -100
local = int((labels[:, 1:] != -100).sum())
share, aux = terms(model, ids, labels, torch.tensor(4 * local))  # one micro-batch of four
plain, aux_plain = terms(model, ids, labels, None)
clone = copy.deepcopy(model)
with torch.no_grad():
    clone.lm_head.weight.zero_()
    a = model(input_ids = ids, labels = labels, num_items_in_batch = torch.tensor(local)).loss
    b = clone(input_ids = ids, labels = labels, num_items_in_batch = torch.tensor(local)).loss
print("RESULT " + json.dumps({
    "share": share, "aux": aux, "plain": plain, "aux_plain": aux_plain, "copy_differs": not torch.allclose(a, b),
}), flush = True)
"""


def _run(extra = None):
    # The CPU route itself: UNSLOTH_ALLOW_CPU (the test suite's GPU-stack-on-CPU mode) installs zoo's GPU forwards.
    env = {k: v for k, v in os.environ.items() if k not in ("UNSLOTH_ALLOW_CPU", "UNSLOTH_DISABLE_CPU_FALLBACK")}
    env.update(CUDA_VISIBLE_DEVICES = "", UNSLOTH_FORCE_CPU = "1", OMP_NUM_THREADS = "2", TOKENIZERS_PARALLELISM = "false")
    env.update(extra or {})
    with tempfile.TemporaryDirectory() as tmp:
        out = subprocess.run(
            [sys.executable, "-c", _CODE, os.path.join(tmp, "moe")],
            capture_output = True, text = True, env = env, timeout = 600, cwd = tmp,
        )
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    pytest.fail(f"no RESULT line\nstdout:\n{out.stdout[-3000:]}\nstderr:\n{out.stderr[-5000:]}")


def test_router_aux_carries_the_micro_batch_token_share():
    res = _run()
    assert res["aux"] > 0, res
    # loss - cross entropy = coef x aux x local / num_items_in_batch (a quarter here), not coef x aux.
    assert res["share"] == pytest.approx(0.5 * res["aux"] / 4, rel = 1e-4), res
    # No count (evaluation, generation): the stock term.
    assert res["plain"] == pytest.approx(0.5 * res["aux_plain"], rel = 1e-4), res
    assert res["copy_differs"], res  # a deepcopy (TRL reference model) runs its own weights


def test_kill_switch_keeps_the_stock_term():
    res = _run({"UNSLOTH_CPU_ROUTER_AUX_GA": "0"})
    assert res["share"] == pytest.approx(0.5 * res["aux"], rel = 1e-4), res

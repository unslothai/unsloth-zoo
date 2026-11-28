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

"""CPU backend: num_items_in_batch counts the targets a causal LM's shifted loss sees (labels[..., 1:]), as on
the GPU route, also where transformers 4.x counts the unshifted labels (one per supervised row start more).
Fresh interpreter: install_cpu_training_defaults patches transformers' Trainer process-wide."""

import json
import os
import subprocess
import sys
import tempfile

import pytest

pytest.importorskip("transformers")

_CODE = r"""
import json, sys, torch
from unsloth_zoo.cpu.trainer import install_cpu_training_defaults
install_cpu_training_defaults()
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

torch.manual_seed(0)
model = LlamaForCausalLM(LlamaConfig(
    vocab_size = 64, hidden_size = 16, intermediate_size = 32, num_hidden_layers = 1,
    num_attention_heads = 2, num_key_value_heads = 1,
))
args = TrainingArguments(output_dir = sys.argv[1], use_cpu = True, report_to = [], per_device_train_batch_size = 2)
trainer = Trainer(model = model, args = args)

def batch(lengths, width):
    ids = torch.randint(1, 64, (len(lengths), width))
    labels = ids.clone()
    mask = torch.zeros_like(ids)
    for row, n in enumerate(lengths):
        labels[row, n:] = -100  # right padding
        mask[row, :n] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}

# Every token supervised, so position 0 of each row is a label the shifted loss never predicts.
batches = [batch([5, 9], 9), batch([3, 7], 7)]
_, count = trainer.get_batch_samples(iter(batches), 2, torch.device("cpu"))
# A classification head keeps transformers' count (its labels are not shifted targets); a custom
# compute_loss_func is what makes stock Trainer count at all.
from transformers import LlamaForSequenceClassification
cls_model = LlamaForSequenceClassification(LlamaConfig(
    vocab_size = 64, hidden_size = 16, intermediate_size = 32, num_hidden_layers = 1,
    num_attention_heads = 2, num_key_value_heads = 1, num_labels = 2, pad_token_id = 0,
))
cls_trainer = Trainer(model = cls_model, args = args, compute_loss_func = lambda outputs, labels, num_items_in_batch = None: outputs.logits.sum())
cls_batches = [{"input_ids": torch.randint(1, 64, (2, 5)), "labels": torch.tensor([0, 1])} for _ in range(2)]
_, cls_count = cls_trainer.get_batch_samples(iter(cls_batches), 2, torch.device("cpu"))
print("RESULT " + json.dumps({
    "cls_count": None if cls_count is None else int(cls_count),
    "count": None if count is None else int(count),
    "shifted": int(sum((b["labels"][:, 1:] != -100).sum() for b in batches)),
    "unshifted": int(sum((b["labels"] != -100).sum() for b in batches)),
    "patched": bool(getattr(Trainer.get_batch_samples, "_unsloth_cpu", False)),
}), flush = True)
"""


def _run(extra = None):
    # The CPU route itself: UNSLOTH_ALLOW_CPU (the test suite's GPU-stack-on-CPU mode) installs zoo's GPU forwards.
    env = {k: v for k, v in os.environ.items() if k not in ("UNSLOTH_ALLOW_CPU", "UNSLOTH_DISABLE_CPU_FALLBACK")}
    env.update(CUDA_VISIBLE_DEVICES = "", UNSLOTH_FORCE_CPU = "1", OMP_NUM_THREADS = "2", TOKENIZERS_PARALLELISM = "false")
    env.update(extra or {})
    with tempfile.TemporaryDirectory() as tmp:
        out = subprocess.run(
            [sys.executable, "-c", _CODE, tmp], capture_output = True, text = True, env = env, timeout = 600, cwd = tmp,
        )
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    pytest.fail(f"no RESULT line\nstdout:\n{out.stdout[-3000:]}\nstderr:\n{out.stderr[-5000:]}")


def test_cpu_route_counts_shifted_causal_targets():
    res = _run()
    assert res["patched"], res
    assert res["shifted"] == res["unshifted"] - 4, res  # 4 rows, each supervised from position 0
    assert res["count"] == res["shifted"], res
    assert res["cls_count"] == 4, res  # 2 batches x 2 labels, as stock transformers counts them


def test_kill_switch_keeps_transformers_count():
    res = _run({"UNSLOTH_CPU_NUM_ITEMS": "0"})
    assert not res["patched"], res
    assert res["count"] in (res["shifted"], res["unshifted"]), res  # transformers' own convention (4.x: unshifted)

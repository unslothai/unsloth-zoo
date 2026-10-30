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

"""`UNSLOTH_COMPILE_DISABLE=partial` must keep the fast LoRA forward; it does not need torch.compile."""
import inspect
import json
import os
import re
import subprocess
import sys

import pytest


def test_lora_patch_is_gated_on_full_disable_only():
    from unsloth_zoo import compiler
    src = inspect.getsource(compiler.unsloth_compile_transformers)
    gate = re.search(r"if \(not (\w+)\) and fast_lora_forwards:", src)
    assert gate is not None, "the fast LoRA forward gate moved; update this test with it"
    assert gate.group(1) == "full_disable", gate.group(0)


_CHILD = r"""
import json
import torch
import unsloth  # noqa: F401
from unsloth import FastModel
model, _ = FastModel.from_pretrained(
    "tiny-random/gemma-4-moe", max_seq_length=256, dtype=torch.bfloat16, load_in_4bit=True,
)
model = FastModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0, bias="none")
from peft.tuners.lora.bnb import Linear4bit
print("RESULT " + json.dumps([Linear4bit.forward.__name__, Linear4bit.forward.__module__]))
"""


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs a GPU to load a 4-bit model")
def test_partial_disable_still_patches_lora_forward(tmp_path):
    pytest.importorskip("transformers.models.gemma4")
    # A child process: `import unsloth` and from_pretrained patch peft / transformers classes and export
    # UNSLOTH_MODEL_NAME etc. for good, which broke later tests in a shared session (gpt_oss patches skipped,
    # peft's Linear4bit.forward replaced, generated gemma4 modules leaving the eager_on_recompile stance).
    script = tmp_path / "child.py"
    script.write_text(_CHILD)
    env = dict(os.environ, UNSLOTH_COMPILE_DISABLE="partial", UNSLOTH_IS_PRESENT="1")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(x for x in (root, env.get("PYTHONPATH")) if x)
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, cwd=tmp_path,
                          timeout=1200)
    line = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
    assert proc.returncode == 0 and line, proc.stdout[-2000:] + proc.stderr[-3000:]
    name, module = json.loads(line[-1][len("RESULT "):])
    assert name == "unsloth_forward", module

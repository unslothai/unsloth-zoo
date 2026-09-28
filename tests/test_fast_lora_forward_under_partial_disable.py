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
import re

import pytest


def test_lora_patch_is_gated_on_full_disable_only():
    from unsloth_zoo import compiler
    src = inspect.getsource(compiler.unsloth_compile_transformers)
    gate = re.search(r"if \(not (\w+)\) and fast_lora_forwards:", src)
    assert gate is not None, "the fast LoRA forward gate moved; update this test with it"
    assert gate.group(1) == "full_disable", gate.group(0)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs a GPU to load a 4-bit model")
def test_partial_disable_still_patches_lora_forward(tmp_path, monkeypatch):
    pytest.importorskip("transformers.models.gemma4")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "partial")
    import torch
    import unsloth  # noqa: F401
    from unsloth import FastModel
    model, _ = FastModel.from_pretrained(
        "tiny-random/gemma-4-moe", max_seq_length=256, dtype=torch.bfloat16, load_in_4bit=True,
    )
    model = FastModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0, bias="none")
    from peft.tuners.lora.bnb import Linear4bit
    assert Linear4bit.forward.__name__ == "unsloth_forward", Linear4bit.forward.__module__

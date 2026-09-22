"""The fast LoRA forward does not depend on torch.compile, so `UNSLOTH_COMPILE_DISABLE=partial`
must keep it. Before, "partial" (set for aya_vision, modernbert, granite-vision, csm and
Gemma-3 on RDNA) also dropped the addmm LoRA forward and left PEFT's, which casts the
activation to the float32 LoRA dtype and runs the LoRA matmuls as fp32 SIMT GEMMs.
"""
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

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""The rewritten fused forward compiles on every torch version.

The rewritten forward runs in a copy of the modeling module globals plus EMPTY_LOGITS, os and the
fused helpers. Dynamo before torch 2.14 resolves an inlined function's globals through
sys.modules[__name__] whenever __name__ is set, so torch.compile of the model failed with
"modeling_qwen3 has no attribute EMPTY_LOGITS" (or "os") on torch 2.13.
"""

import inspect
import os
import sys

import pytest
import torch

_PREVIOUS_ALLOW_CPU = os.environ.get("UNSLOTH_ALLOW_CPU")
try:
    os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
    from unsloth_zoo.fused_losses import forward_install
finally:
    if _PREVIOUS_ALLOW_CPU is None:
        os.environ.pop("UNSLOTH_ALLOW_CPU", None)
    else:
        os.environ["UNSLOTH_ALLOW_CPU"] = _PREVIOUS_ALLOW_CPU


def _patched_qwen3():
    import transformers.models.qwen3.modeling_qwen3 as modeling
    cls = modeling.Qwen3ForCausalLM
    forward_install.install_for_class(cls)
    if "EMPTY_LOGITS" not in inspect.unwrap(cls.forward).__code__.co_names:
        pytest.fail(f"fused forward not installed on Qwen3ForCausalLM: {forward_install.audit()['failed']}")
    return modeling, cls


def test_fused_forward_globals_not_mistaken_for_module():
    modeling, cls = _patched_qwen3()
    forward = inspect.unwrap(cls.forward)
    fglobals = forward.__globals__
    assert forward.__module__ == modeling.__name__
    # Either no __name__ (Dynamo guards on the dict itself) or the dict really is the module's.
    name = fglobals.get("__name__")
    assert name is None or sys.modules[name].__dict__ is fglobals
    assert fglobals["EMPTY_LOGITS"] is not None
    assert not hasattr(modeling, "EMPTY_LOGITS")


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "fused loss kernel needs a GPU")
@pytest.mark.parametrize("fullgraph", [False, True])
def test_torch_compile_fused_forward_tiny_qwen3(fullgraph):
    _, cls = _patched_qwen3()
    from transformers import Qwen3Config
    torch._dynamo.reset()
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size = 128, hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2,
        num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16,
    )
    model = cls(config).cuda().train()
    ids = torch.randint(0, 128, (2, 8), device = "cuda")
    eager = model(input_ids = ids, labels = ids).loss
    compiled = torch.compile(model, backend = "aot_eager", fullgraph = fullgraph)
    loss = compiled(input_ids = ids, labels = ids).loss
    loss.backward()
    torch.testing.assert_close(loss.detach(), eager.detach(), rtol = 1e-4, atol = 1e-4)
    torch._dynamo.reset()

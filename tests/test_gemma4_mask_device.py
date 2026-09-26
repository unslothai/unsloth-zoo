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

"""Gemma 4's legacy vision mask must follow the embedding execution device.

Load the patch function alone so these checks do not depend on CUDA-heavy
package initialization or unrelated patches. The mask builders and model are
the installed Transformers implementation.
"""
import ast
from contextlib import nullcontext
import functools
import inspect
import os
from typing import Callable
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_patch(name, filename, **namespace):
    path = ROOT / "unsloth_zoo" / "temporary_patches" / filename
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {"torch": torch, "inspect": inspect, "os": os, **namespace}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


@pytest.fixture(params=[
    "cpu",
    pytest.param("cuda:1", marks=pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="Requires two physical CUDA GPUs",
    )),
])
def source_device(request):
    return request.param


@pytest.fixture
def legacy(monkeypatch):
    gemma = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    original = getattr(gemma, "token_type_ids_mask_function", None)
    if original is None:
        pytest.skip("Legacy Gemma vision-mask helper is absent on this Transformers version")
    # Package initialization may already have applied the patch. Read the
    # upstream helper from its source for an independent negative control.
    tree = ast.parse(Path(gemma.__file__).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "token_type_ids_mask_function")
    scope = {"torch": torch, "Callable": Callable}
    exec(compile(ast.Module(body=[node], type_ignores=[]), gemma.__file__, "exec"), scope)
    original = scope["token_type_ids_mask_function"]
    monkeypatch.setattr(gemma, "token_type_ids_mask_function", original)
    return gemma, original


def install():
    load_patch("patch_Gemma4_token_type_ids_mask", "gemma4.py")()


def test_missing_legacy_helper_is_untouched(monkeypatch):
    gemma = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    monkeypatch.delattr(gemma, "token_type_ids_mask_function", raising=False)
    install()
    assert not hasattr(gemma, "token_type_ids_mask_function")


def test_unknown_signature_is_untouched(legacy, monkeypatch):
    gemma, _ = legacy
    def future_helper(token_type_ids, image_group_ids, new_argument):
        return new_argument
    monkeypatch.setattr(gemma, "token_type_ids_mask_function", future_helper)
    install()
    assert gemma.token_type_ids_mask_function is future_helper


def test_registration_is_idempotent(legacy):
    gemma, _ = legacy
    install()
    patched = gemma.token_type_ids_mask_function
    install()
    assert gemma.token_type_ids_mask_function is patched
    assert patched(None, None) is None


@pytest.mark.parametrize("groups", [[-1]*8, [-1,0,0,-1,1,1,-1,-1]])
def test_mask_semantics_and_static_cache_bounds(legacy, groups):
    gemma, original = legacy
    tt = torch.tensor([[int(v >= 0) for v in groups]])
    ids = torch.tensor([groups])
    before = ids.clone()
    expected = original(tt, ids)
    install()
    actual = gemma.token_type_ids_mask_function(tt, ids)
    # Beyond the current sequence: static-cache slots must remain masked.
    q = torch.arange(11)[:, None]
    k = torch.arange(11)[None, :]
    torch.testing.assert_close(actual(0, 0, q, k), expected(0, 0, q, k))
    torch.testing.assert_close(ids, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Cross-device regression needs a CUDA GPU")
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("vision_tokens", [False, True])
def test_real_mask_builder_with_existing_unsloth_guards(legacy, monkeypatch, compiled, vision_tokens, source_device):
    import transformers.masking_utils as mu
    import transformers.generation.utils as gu
    from transformers import Gemma4Config, Gemma4TextConfig

    gemma, original = legacy
    # Restore every module attribute the existing generic guard installs.
    mu_before, gu_before = dict(vars(mu)), dict(vars(gu))
    try:
        mu.__dict__.pop("__unsloth_mask_patch__", None)
        def fail(*args):
            raise AssertionError(args)
        compile_fn = functools.partial(torch.compile, backend="eager") if compiled else lambda f, **kw: f
        load_patch("patch_transformers_masks", "misc.py", _torch_compile=compile_fn, raise_error=fail)()
        monkeypatch.setattr(gemma, "create_causal_mask", mu.create_causal_mask)
        monkeypatch.setattr(gemma, "create_sliding_window_causal_mask", mu.create_sliding_window_causal_mask)
        config = Gemma4Config(
            text_config=Gemma4TextConfig(use_bidirectional_attention="vision", sliding_window=4).to_dict(),
            vision_config=None, audio_config=None,
        )
        config.text_config._attn_implementation = "sdpa"
        embeddings = torch.zeros(1, 8, 16, device="cuda")
        positions = torch.arange(8, device="cuda").unsqueeze(0)
        token_types = torch.tensor([[0,1,1,0,2,2,0,0] if vision_tokens else [0]*8], device=source_device)
        def build(types):
            return gemma.create_causal_mask_mapping(
                config, embeddings, None, None, positions, types, is_training=True,
            )

        expected = build(token_types.cuda())
        # Existing generic packed-sequence guards do not cover the vision overlay.
        with pytest.raises(Exception, match="indices should be|[Dd]evice"):
            build(token_types)
        install()
        actual = build(token_types)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
        # Unsloth copies standalone function source into its generated module.
        # The replacement must survive that without a captured original function.
        import textwrap
        copied = {"torch": torch}
        exec(textwrap.dedent(inspect.getsource(gemma.token_type_ids_mask_function)), copied)
        monkeypatch.setattr(gemma, "token_type_ids_mask_function", copied["token_type_ids_mask_function"])
        for key, value in build(token_types).items():
            torch.testing.assert_close(value, expected[key])
    finally:
        for module, before in ((mu, mu_before), (gu, gu_before)):
            for key in set(vars(module)) - set(before):
                if key.startswith(("_unsloth", "__unsloth")):
                    delattr(module, key)
            for key, value in before.items():
                setattr(module, key, value)
        torch._dynamo.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Cross-device training surrogate needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_lora_checkpointed_training_matches_baseline(legacy, dtype, monkeypatch, source_device):
    from accelerate import dispatch_model
    from peft import LoraConfig, get_peft_model
    from transformers import Gemma4Config, Gemma4TextConfig, Gemma4ForConditionalGeneration

    gemma, original = legacy
    config = Gemma4TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, global_head_dim=16, hidden_size_per_layer_input=0,
        num_kv_shared_layers=0, layer_types=["sliding_attention", "full_attention"],
        sliding_window=4, use_bidirectional_attention="vision",
    )
    config = Gemma4Config(text_config=config.to_dict(), vision_config=None, audio_config=None)
    config._attn_implementation = "sdpa"
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]], device="cuda")
    types = torch.zeros_like(ids)
    # One sample, as in the reporter's retry. Vision overlay is still constructed.
    positions = torch.arange(8, device="cuda").unsqueeze(0)

    def build(split):
        torch.manual_seed(3407)
        model = Gemma4ForConditionalGeneration(config).to(device="cuda", dtype=dtype).train()
        if split:
            # CPU substitutes for the unavailable second CUDA device.
            model = dispatch_model(
                model, {"model.language_model": "cuda:0", "lm_head": "cuda:0"},
                main_device=source_device, force_hooks=True,
            )
        model = get_peft_model(model, LoraConfig(
            r=4, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
        ))
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        return model

    def step(model):
        model.zero_grad(set_to_none=True)
        # Keep BF16 arithmetic identical across differently captured graphs.
        # Compiled mask creation is exercised separately above.
        context = torch.compiler.set_stance("force_eager") if dtype == torch.bfloat16 else nullcontext()
        with context:
            loss = model(
                input_ids=ids, mm_token_type_ids=types, position_ids=positions,
                labels=ids, use_cache=False,
            ).loss
            loss.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        assert grads and all(torch.isfinite(g).all() for g in grads.values())
        return loss.detach(), grads

    expected_loss, expected_grads = step(build(False))
    split = build(True)
    with pytest.raises(Exception, match="indices should be|[Dd]evice"):
        step(split)
    # Reference uses the same dispatch hooks but moves multimodal input
    # metadata before upstream constructs its group IDs. This separates
    # patch correctness from BF16 differences between execution graphs.
    original_builder = gemma.create_causal_mask_mapping
    signature = inspect.signature(original_builder)
    def reference_builder(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        types = bound.arguments.get("mm_token_type_ids")
        if types is not None:
            bound.arguments["mm_token_type_ids"] = types.to(bound.arguments["inputs_embeds"].device)
        return original_builder(*bound.args, **bound.kwargs)
    with monkeypatch.context() as context:
        context.setattr(gemma, "create_causal_mask_mapping", reference_builder)
        reference_loss, reference_grads = step(build(True))

    install()
    unchanged_loss, unchanged_grads = step(build(False))
    torch.testing.assert_close(unchanged_loss, expected_loss, rtol=0, atol=0)
    for name in unchanged_grads:
        torch.testing.assert_close(unchanged_grads[name], expected_grads[name], rtol=0, atol=0)
    loss, grads = step(split)
    # Different device-hook graphs can produce BF16 rounding differences.
    tolerance = {"rtol": 1.6e-2, "atol": 1e-4} if dtype == torch.bfloat16 else {"rtol": 0, "atol": 0}
    torch.testing.assert_close(loss, expected_loss, **tolerance)
    torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
    assert grads.keys() == reference_grads.keys()
    for name in grads:
        torch.testing.assert_close(grads[name], reference_grads[name], rtol=0, atol=0)

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

import importlib
import importlib.util
import os

import pytest
import torch


def _load(name):
    try:
        return importlib.import_module(f"unsloth_zoo.speculative.{name}")
    except ImportError:
        path = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth_zoo", "speculative", f"{name}.py")
        spec = importlib.util.spec_from_file_location(f"_speculative_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


capture = _load("capture")

QWEN3 = "trl-internal-testing/tiny-Qwen3ForCausalLM"
QWEN35_VLM = "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-NoThink"
QWEN35_TEXT = os.path.join(
    os.environ.get("WORKSPACE", ""), "temp", "tiny_cache", "Qwen3_5ForCausalLM-84305c791e79",
)


def _load_model(name):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    try:
        if name == QWEN35_VLM:
            model = AutoModelForImageTextToText.from_pretrained(name, dtype = torch.float32)
        else:
            model = AutoModelForCausalLM.from_pretrained(name, dtype = torch.float32)
    except Exception as error:
        pytest.skip(f"{name} unavailable offline: {type(error).__name__}")
    return model.to(DEVICE).eval()


MODELS = [QWEN3, QWEN35_TEXT, QWEN35_VLM]
# With a GPU visible, zoo routes Qwen3.5's gated delta rule to Triton, which needs CUDA tensors.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _ids(model, length = 9):
    torch.manual_seed(0)
    return torch.randint(3, min(200, model.config.get_text_config().vocab_size), (2, length), device = DEVICE)


@pytest.mark.parametrize("name", MODELS)
def test_final_matches_hf_post_norm_and_layers(name):
    if name == QWEN35_TEXT and not os.path.isdir(name):
        pytest.skip("tiny Qwen3.5 text fixture not built")
    model = _load_model(name)
    ids = _ids(model)
    with torch.no_grad():
        ref = model(input_ids = ids, output_hidden_states = True, use_cache = False)
    parts = capture.resolve_text_parts(model)
    n_layers = len(parts["backbone"].layers)
    got = capture.run_frozen_target(model, ids, attention_mask = torch.ones_like(ids), layer_ids = (0, -1))
    # transformers 5 returns the post-norm state as hidden_states[-1] (== last_hidden_state).
    assert torch.allclose(got["final"], ref.hidden_states[-1], atol = 1e-5)
    assert torch.allclose(got[0], ref.hidden_states[1], atol = 1e-5)
    assert torch.allclose(parts["final_norm"](got[-1]), got["final"], atol = 1e-5)
    assert set(got) == {"final", 0, -1}
    assert parts["lm_head"] is model.get_output_embeddings()
    assert parts["embed_tokens"] is model.get_input_embeddings()
    assert parts["config"].num_hidden_layers == n_layers


def test_hooks_removed_on_exception_and_overwrite():
    model = _load_model(QWEN3)
    parts = capture.resolve_text_parts(model)
    modules = [parts["final_norm"], parts["backbone"], parts["backbone"].layers[0]]
    before = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in modules]
    with pytest.raises(RuntimeError, match = "boom"):
        with capture.capture_hidden_states(model, layer_ids = (0,)):
            raise RuntimeError("boom")
    assert [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in modules] == before
    with torch.no_grad(), capture.capture_hidden_states(model, layer_ids = (0,)) as captured:
        model(input_ids = _ids(model, 5))
        model(input_ids = _ids(model, 7))
    assert captured["final"].shape[1] == 7 and captured[0].shape[1] == 7 and len(captured) == 2
    assert [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in modules] == before


def test_inference_mode_tensors_are_autograd_safe():
    model = _load_model(QWEN3)
    with torch.inference_mode(), capture.capture_hidden_states(model) as captured:
        model(input_ids = _ids(model))
    final = captured["final"]
    assert not final.is_inference()
    weight = torch.nn.Parameter(torch.ones(final.shape[-1], device = final.device))
    (final * weight).pow(2).sum().backward()
    assert weight.grad is not None


def test_functional_norm_fallback():
    # Fast forwards apply the norm without calling the module; the backbone output is used.
    model = _load_model(QWEN3)
    parts = capture.resolve_text_parts(model)
    norm = parts["final_norm"]
    original_call = type(norm).__call__
    ids = _ids(model)
    with torch.no_grad():
        ref = model(input_ids = ids, output_hidden_states = True).hidden_states[-1]
    class Bypass(type(norm)):
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)
    norm.__class__ = Bypass
    try:
        got = capture.run_frozen_target(model, ids)
    finally:
        norm.__class__ = type(norm).__mro__[1]
    assert type(norm).__call__ is original_call
    assert torch.allclose(got["final"], ref, atol = 1e-5)


def test_peft_unwrap_and_errors():
    peft = pytest.importorskip("peft")
    model = _load_model(QWEN3)
    base_parts = capture.resolve_text_parts(model)
    peft_model = peft.get_peft_model(model, peft.LoraConfig(r = 4, target_modules = ["q_proj", "v_proj"]))
    parts = capture.resolve_text_parts(peft_model)
    assert parts["backbone"] is base_parts["backbone"] and parts["lm_head"] is base_parts["lm_head"]
    got = capture.run_frozen_target(peft_model, _ids(model))
    assert got["final"].shape[-1] == model.config.hidden_size
    with pytest.raises(ValueError, match = "Linear"):
        capture.resolve_text_parts(torch.nn.Linear(2, 2))
    with pytest.raises(ValueError, match = "out of range"):
        with capture.capture_hidden_states(model, layer_ids = (99,)):
            pass

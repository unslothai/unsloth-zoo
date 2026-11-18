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

"""vision_dedup on tiny VLMs: dedup matches per-row logprobs/grads; kill switch and dropout gate disable it."""

import pytest
import torch

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
from PIL import Image, ImageDraw  # noqa: E402

from unsloth_zoo import vision_dedup  # noqa: E402

MODELS = {
    "gemma3": "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
    "qwen2_5_vl": "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
}
G = 4


def _load(key):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    try:
        proc = AutoProcessor.from_pretrained(MODELS[key])
        model = AutoModelForImageTextToText.from_pretrained(MODELS[key], dtype = torch.float32)
    except Exception as e:  # offline without the cached tiny repo
        pytest.skip(f"{MODELS[key]} unavailable: {e}")
    return proc, model


def _tower(model):
    for name, module in model.named_modules():
        if "language_model" in name:
            continue
        if name.endswith("vision_tower") or name.endswith("visual"):
            return module
    raise AssertionError("no vision tower")


def _batch(proc, prompts = 2):
    images, texts = [], []
    for p in range(prompts):
        img = Image.new("RGB", (64, 64), (255, 40 + 100 * p, 255))
        ImageDraw.Draw(img).text((4, 20), f"{p + 2}+{p + 5}", fill = (0, 0, 0))
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Sum {p}?"}]}]
        prompt = proc.apply_chat_template(msgs, add_generation_prompt = True)
        for g in range(G):
            images.append([img])
            texts.append(prompt + " answer" * (g + 1) + f" {g}")
    tok = getattr(proc, "tokenizer", proc)
    tok.padding_side = "right"
    return proc(text = texts, images = images, padding = True, return_tensors = "pt")


def _logps(model, inputs):
    logits = model(**inputs).logits[:, :-1].float()
    target = inputs["input_ids"][:, 1:]
    lp = torch.log_softmax(logits, -1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return lp * inputs["attention_mask"][:, 1:]


def _run(model, inputs, tower, monkeypatch, enabled):
    monkeypatch.setenv("UNSLOTH_GRPO_IMAGE_DEDUP", "1" if enabled else "0")
    seen = []
    handle = tower.register_forward_pre_hook(
        lambda m, a, k: seen.append(
            int(k["grid_thw"].shape[0]) if k.get("grid_thw", None) is not None
            else int((k.get("pixel_values", a[0] if a else None)).shape[0])
        ),
        with_kwargs = True,
    )
    try:
        model.zero_grad(set_to_none = True)
        lp = _logps(model, inputs)
        lp.sum().backward()
    finally:
        handle.remove()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return lp.detach(), grads, seen


def _peft(model, key, vision_lora, dropout = 0.0):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(0)
    if vision_lora:
        target = "all-linear"
    else:
        target = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    cfg = LoraConfig(r = 4, lora_alpha = 8, lora_dropout = dropout, target_modules = target,
                     init_lora_weights = False)
    return get_peft_model(model, cfg)


@pytest.mark.parametrize("key", list(MODELS))
@pytest.mark.parametrize("vision_lora", [False, True], ids = ["tower_frozen", "tower_lora"])
def test_dedup_parity_and_engagement(key, vision_lora, monkeypatch):
    proc, model = _load(key)
    model = _peft(model, key, vision_lora)
    model.train()
    inputs = _batch(proc)
    tower = _tower(model)
    assert vision_dedup.install_image_dedup(model) == 1
    assert vision_dedup.install_image_dedup(model) == 0  # idempotent

    lp_base, g_base, seen_base = _run(model, inputs, tower, monkeypatch, enabled = False)
    # First engaged call per module also encodes in full once, to self-check the mapping.
    v0 = vision_dedup.DEDUP_STATS["verified"]
    _, _, seen_first = _run(model, inputs, tower, monkeypatch, enabled = True)
    assert vision_dedup.DEDUP_STATS["verified"] == v0 + 1 and sorted(seen_first) == [2, 2 * G], seen_first
    before = dict(vision_dedup.DEDUP_STATS)
    lp_head, g_head, seen_head = _run(model, inputs, tower, monkeypatch, enabled = True)
    after = dict(vision_dedup.DEDUP_STATS)

    assert sum(seen_base) == 2 * G, seen_base     # kill switch: every row encoded
    assert sum(seen_head) == 2, seen_head         # one encode per distinct image
    assert after["deduped_calls"] - before["deduped_calls"] == 1
    assert after["fallbacks"] == before["fallbacks"]

    assert g_base.keys() == g_head.keys() and len(g_base) > 0
    if vision_lora:
        assert any("lora" in n and ("vision" in n or "visual" in n) for n in g_head), "no vision LoRA grads"
    lp_diff = (lp_head - lp_base).abs().max().item()
    g_diff = max(((g_head[n] - g_base[n]).abs().max() / g_base[n].abs().max().clamp_min(1e-12)).item() for n in g_base)
    bitwise = torch.equal(lp_head, lp_base) and all(torch.equal(g_head[n], g_base[n]) for n in g_base)
    print(f"[vision_dedup] {key} {'tower_lora' if vision_lora else 'tower_frozen'}: logp max|d| = {lp_diff:.3e}, "
          f"grad max|d|/max|g| = {g_diff:.3e}, bitwise = {bitwise}")
    torch.testing.assert_close(lp_head, lp_base, rtol = 1e-5, atol = 1e-5)
    for n in g_base:
        torch.testing.assert_close(g_head[n], g_base[n], rtol = 1e-4, atol = 1e-5, msg = lambda m: f"{n}: {m}")


@pytest.mark.parametrize("key", list(MODELS))
def test_dropout_gate(key, monkeypatch):
    """Vision LoRA dropout in train mode draws per row: no dedup. In eval mode it is inert: dedup."""
    proc, model = _load(key)
    model = _peft(model, key, vision_lora = True, dropout = 0.1)
    inputs = _batch(proc)
    vision_dedup.install_image_dedup(model)
    model.train()
    monkeypatch.setenv("UNSLOTH_GRPO_IMAGE_DEDUP", "1")
    monkeypatch.setenv("UNSLOTH_GRPO_IMAGE_DEDUP_VERIFY", "0")
    before = vision_dedup.DEDUP_STATS["deduped_calls"]
    with torch.no_grad():
        model(**inputs)
    assert vision_dedup.DEDUP_STATS["deduped_calls"] == before  # gated in train mode
    model.eval()
    with torch.no_grad():
        model(**inputs)
    assert vision_dedup.DEDUP_STATS["deduped_calls"] == before + 1


@pytest.mark.parametrize("key", list(MODELS))
def test_single_distinct_image_encodes_two(key, monkeypatch):
    """One distinct image is encoded at batch 2, not 1, and logprobs still match."""
    proc, model = _load(key)
    model = _peft(model, key, vision_lora = False)
    model.train()
    inputs = _batch(proc, prompts = 1)
    tower = _tower(model)
    vision_dedup.install_image_dedup(model)
    monkeypatch.setenv("UNSLOTH_GRPO_IMAGE_DEDUP_VERIFY", "0")
    lp_base, _, seen_base = _run(model, inputs, tower, monkeypatch, enabled = False)
    lp_head, _, seen_head = _run(model, inputs, tower, monkeypatch, enabled = True)
    assert sum(seen_base) == G and sum(seen_head) == 2, (seen_base, seen_head)
    torch.testing.assert_close(lp_head, lp_base, rtol = 1e-5, atol = 1e-5)


def test_self_check_catches_a_wrong_mapping():
    """The first-use self-check is structural: batch-size noise passes, swapped images do not."""
    torch.manual_seed(0)
    full = torch.randn(2, 16).repeat_interleave(2, 0)  # images [a, a, b, b]
    inverse = [0, 0, 1, 1]
    noisy = full + 0.1 * torch.randn_like(full)
    assert vision_dedup._outputs_mismatch(noisy, full, inverse) is None
    swapped = full[[2, 3, 0, 1]]
    assert vision_dedup._outputs_mismatch(swapped, full, inverse) is not None


@pytest.mark.parametrize("key", list(MODELS))
def test_wrapper_keeps_signature_and_generate_runs(key, monkeypatch):
    # transformers >= 5.23 generate pops encoder inputs by get_image_features' signature.
    import inspect
    proc, model = _load(key)
    owners = vision_dedup._image_modules(model)
    before = [inspect.signature(m.get_image_features) for m in owners]
    assert vision_dedup.install_image_dedup(model) == len(owners) > 0
    assert [inspect.signature(m.get_image_features) for m in owners] == before
    inputs = _batch(proc, prompts = 1)
    tok = getattr(proc, "tokenizer", proc)
    tok.padding_side = "left"
    inputs = proc(text = [proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Hi"}]}], add_generation_prompt = True)],
                  images = [[Image.new("RGB", (64, 64), (200, 30, 30))]], return_tensors = "pt")
    out = model.generate(**inputs, max_new_tokens = 2, do_sample = False)
    assert out.shape[1] == inputs["input_ids"].shape[1] + 2


def test_kill_switch_skips_install(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_IMAGE_DEDUP", "0")
    _, model = _load("qwen2_5_vl")
    assert vision_dedup.install_image_dedup(model) == 0
    assert all(not m.__dict__.get(vision_dedup._INSTALLED, False) for m in vision_dedup._image_modules(model))


def test_collective_sharded_forwards_never_dedup(monkeypatch):
    # Resharding FSDP2 towers all-gather per forward; rank-local extra forwards would hang.
    import unsloth_zoo.rl_replacements as rr
    _, model = _load("qwen2_5_vl")
    monkeypatch.setattr(rr, "grpo_forwards_are_collective", lambda m: True)
    assert vision_dedup.install_image_dedup(model) == 0
    monkeypatch.setattr(rr, "grpo_forwards_are_collective", lambda m: False)
    proc, model = _load("qwen2_5_vl")
    assert vision_dedup.install_image_dedup(model) > 0
    inputs = _batch(proc)
    tower = _tower(model)
    calls = []
    hook = tower.register_forward_hook(lambda m, a, o: calls.append(1))
    monkeypatch.setattr(rr, "grpo_forwards_are_collective", lambda m: True)
    with torch.no_grad():
        model(**inputs)
    hook.remove()
    assert len(calls) == 1  # one per-row encode, no self-check forward
    assert not any(m.__dict__.get(vision_dedup._VERIFIED, False) for m in vision_dedup._image_modules(model))

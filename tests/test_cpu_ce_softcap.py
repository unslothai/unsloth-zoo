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

"""CPU chunked CE for heads whose forward transforms the logits (Gemma 2 / 3 soft cap, Cohere / Granite scale):
same loss and grads as the model's own forward, without its [B, S, V] logits, and ignored rows never reach the
LM head GEMMs, also under torch.compile."""

import pytest
import torch
import torch.nn.functional as F


def _reference(hidden, weight, labels, softcap, logit_scale, num_items = None):
    logits = (hidden.double() @ weight.double().t()) * logit_scale
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    shift = F.pad(labels, (0, 1), value = -100)[..., 1:].reshape(-1)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), shift, ignore_index = -100, reduction = "sum")
    return loss / (num_items if num_items is not None else (shift != -100).sum())


def _inputs(dtype, masked = 0.75, seed = 0):
    torch.manual_seed(seed)
    hidden = (torch.randn(3, 16, 24) * 3).to(dtype).requires_grad_()
    weight = (torch.randn(80, 24) * 2).to(dtype).requires_grad_()
    labels = torch.randint(0, 80, (3, 16))
    labels[torch.rand(3, 16) < masked] = -100
    return hidden, weight, labels


@pytest.mark.parametrize("softcap, logit_scale", [(None, 1.0), (5.0, 1.0), (30.0, 1.0), (None, 0.0625), (4.0, 0.5)])
@pytest.mark.parametrize("num_items", [None, 7])
def test_transformed_loss_and_grads_match_autograd(softcap, logit_scale, num_items):
    from unsloth_zoo.cpu.ops import chunked_linear_cross_entropy
    hidden, weight, labels = _inputs(torch.float32)
    loss = chunked_linear_cross_entropy(hidden, weight, labels, num_items_in_batch = num_items, chunk_rows = 5,
                                        softcap = softcap, logit_scale = logit_scale)
    loss.backward()
    h2, w2 = hidden.detach().double().requires_grad_(), weight.detach().double().requires_grad_()
    ref = _reference(h2, w2, labels, softcap, logit_scale, num_items)
    ref.backward()
    assert torch.allclose(loss.double(), ref, rtol = 1e-5, atol = 1e-6), (loss, ref)
    assert torch.allclose(hidden.grad.double(), h2.grad, rtol = 1e-4, atol = 1e-6)
    assert torch.allclose(weight.grad.double(), w2.grad, rtol = 1e-4, atol = 1e-6)
    if softcap:  # the cap really bites: raw logits exceed it
        raw = (hidden.detach() @ weight.detach().t()) * logit_scale
        assert float(raw.abs().max()) > softcap


def _count_gemm_rows(monkeypatch):
    from unsloth_zoo.cpu import ops
    rows = []
    loop = ops._chunked_ce_forward_loop

    def spy(hidden, *args, **kwargs):
        rows.append(hidden.shape[0])
        return loop(hidden, *args, **kwargs)

    monkeypatch.setattr(ops, "_chunked_ce_forward_loop", spy)
    return rows


@pytest.mark.parametrize("softcap", [None, 5.0])
def test_compiled_loss_drops_ignored_rows(monkeypatch, softcap):
    from unsloth_zoo.cpu.ops import chunked_linear_cross_entropy
    rows = _count_gemm_rows(monkeypatch)
    hidden, weight, labels = _inputs(torch.float32)

    def fn(h, w, y):
        kwargs = {"softcap": softcap} if softcap else {}
        return chunked_linear_cross_entropy(h, w, y, num_items_in_batch = 9, chunk_rows = 4, **kwargs)

    torch._dynamo.reset()
    loss = torch.compile(fn, backend = "aot_eager", fullgraph = True)(hidden, weight, labels)
    loss.backward()
    shift = F.pad(labels, (0, 1), value = -100)[..., 1:]
    valid = int((shift != -100).sum())
    # Before: the compiled path ran the LM head over all 48 rows, ignored ones included.
    assert rows and rows[-1] == valid < shift.numel(), (rows, valid)
    h2, w2 = hidden.detach().double().requires_grad_(), weight.detach().double().requires_grad_()
    ref = _reference(h2, w2, labels, softcap, 1.0, 9)
    ref.backward()
    assert torch.allclose(loss.double(), ref, rtol = 1e-5, atol = 1e-6)
    assert torch.allclose(hidden.grad.double(), h2.grad, rtol = 1e-4, atol = 1e-6)
    assert torch.allclose(weight.grad.double(), w2.grad, rtol = 1e-4, atol = 1e-6)


def _tiny_model(name, softcap = None):
    import transformers
    torch.manual_seed(0)
    common = dict(vocab_size = 96, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
                  num_attention_heads = 4, num_key_value_heads = 2, head_dim = 8, tie_word_embeddings = False)
    if name == "gemma2":
        config = transformers.Gemma2Config(**common, final_logit_softcapping = softcap or 30.0)
        cls = transformers.Gemma2ForCausalLM
    elif name == "gemma3_text":
        config = transformers.Gemma3TextConfig(**common, final_logit_softcapping = softcap)
        cls = transformers.Gemma3ForCausalLM
    elif name == "cohere":
        config = transformers.CohereConfig(**common, logit_scale = 0.25)
        cls = transformers.CohereForCausalLM
    elif name == "granite":
        config = transformers.GraniteConfig(**common, logits_scaling = 4.0)
        cls = transformers.GraniteForCausalLM
    else:
        config = transformers.Qwen3Config(**common)
        cls = transformers.Qwen3ForCausalLM
    config._attn_implementation = "eager"
    model = cls(config).float()
    if softcap or name == "gemma2":
        with torch.no_grad():  # logits large enough that the soft cap is active
            model.get_output_embeddings().weight.mul_(40.0)
    return model


_FAMILIES = [("gemma2", 2.0), ("gemma3_text", 3.0), ("gemma3_text", None), ("cohere", None), ("granite", None),
             ("qwen3", None)]


@pytest.mark.parametrize("name, softcap", _FAMILIES)
def test_patched_forward_matches_stock_loss_and_grads(name, softcap, monkeypatch):
    from unsloth_zoo.cpu import ops
    applied = []
    apply = ops._ChunkedLinearCrossEntropy.apply
    monkeypatch.setattr(ops._ChunkedLinearCrossEntropy, "apply", lambda *a: applied.append(a[6:]) or apply(*a))
    stock = _tiny_model(name, softcap)
    model = _tiny_model(name, softcap)
    model.load_state_dict(stock.state_dict())
    assert ops.patch_chunked_causal_lm_loss(model)
    model._unsloth_cpu_chunked_ce_active = True
    torch.manual_seed(1)
    ids = torch.randint(0, 96, (2, 12))
    labels = ids.clone()
    labels[:, :8] = -100  # response-only: two thirds masked
    out = model(input_ids = ids, labels = labels)
    ref = stock(input_ids = ids, labels = labels)
    assert out.logits is None
    out.loss.backward()
    ref.loss.backward()
    assert len(applied) == 1  # engagement: the chunked loss ran
    if softcap:
        assert applied[0][0] == softcap
    assert torch.allclose(out.loss, ref.loss, rtol = 1e-5, atol = 1e-6), (out.loss, ref.loss)
    for (n, p), (_, q) in zip(model.named_parameters(), stock.named_parameters()):
        # Relative to the whole tensor: fp32 summation order differs by BLAS build (Windows vs Linux).
        assert float((p.grad - q.grad).norm()) <= 2e-5 * float(q.grad.norm()) + 1e-7, n


def test_kill_switch_keeps_stock_forward(monkeypatch):
    from unsloth_zoo.cpu import ops
    monkeypatch.setenv("UNSLOTH_CPU_CHUNKED_CE", "0")
    model = _tiny_model("gemma2")
    assert not ops.patch_chunked_causal_lm_loss(model)
    assert not vars(model).get("_unsloth_cpu_chunked_ce", False)


@pytest.mark.parametrize("change", ["composite", "extra_knob", "bad_cap", "moe"])
def test_unmatched_transforms_keep_stock_forward(change):
    from unsloth_zoo.cpu import ops
    model = _tiny_model("gemma2")
    if change == "composite":
        model.config.model_type = "gemma3"  # a vision composite keeps its own forward
    elif change == "extra_knob":
        model.config.logit_scale = 2.0  # a knob this family's forward does not read
    elif change == "bad_cap":
        model.config.final_logit_softcapping = 0.0
    else:
        model.config.num_local_experts = 4
    assert ops._lm_head_transform(model) is None
    assert not ops.patch_chunked_causal_lm_loss(model)


@pytest.mark.parametrize("name, softcap", [("gemma2", 2.0), ("cohere", None)])
def test_preference_logps_use_the_transformed_logits(name, softcap):
    from unsloth_zoo.cpu import ops
    model = _tiny_model(name, softcap)
    assert ops.patch_chunked_causal_lm_loss(model)
    torch.manual_seed(2)
    ids = torch.randint(0, 96, (2, 10))
    targets = torch.randint(0, 96, (2, 10))
    model._unsloth_cpu_logps_targets = targets
    try:
        with torch.no_grad():
            out = model(input_ids = ids, use_cache = False)
    finally:
        model.__dict__.pop("_unsloth_cpu_logps_targets")
    with torch.no_grad():
        logits = model(input_ids = ids).logits.double()
    expected = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(out["log_probs"].double(), expected, rtol = 1e-5, atol = 1e-5)
    assert torch.allclose(out["mean_logits"].double(), logits.mean(-1), rtol = 1e-5, atol = 1e-5)


@pytest.mark.parametrize("name, knob, value", [("gemma2", "final_logit_softcapping", 5.0),
                                              ("gemma2", "final_logit_softcapping", None),
                                              ("granite", "logits_scaling", 1.5),
                                              ("gemma2", "final_logit_softcapping", 0.0)])
def test_cap_or_scale_changed_after_patching(name, knob, value):
    from unsloth_zoo.cpu import ops
    stock = _tiny_model(name, 2.0)
    model = _tiny_model(name, 2.0)
    model.load_state_dict(stock.state_dict())
    assert ops.patch_chunked_causal_lm_loss(model)
    model._unsloth_cpu_chunked_ce_active = True
    setattr(model.config, knob, value)
    setattr(stock.config, knob, value)
    torch.manual_seed(1)
    ids = torch.randint(0, 96, (2, 12))
    labels = ids.clone()
    labels[:, :6] = -100
    if value == 0.0:  # a cap the chunked loss cannot apply: the stock forward runs (and divides by zero alike)
        assert model(input_ids = ids, labels = labels).logits is not None
        return
    out, ref = model(input_ids = ids, labels = labels), stock(input_ids = ids, labels = labels)
    assert out.logits is None
    assert torch.allclose(out.loss, ref.loss, rtol = 1e-5, atol = 1e-6), (out.loss, ref.loss)

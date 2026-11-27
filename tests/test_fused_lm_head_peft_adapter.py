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

"""Fused CE on an lm_head with a live PEFT adapter must match lm_head.forward(): the kernel reads
only the base weight, so the adapter got no gradient; such heads now score via their own forward
with the kernel's reduction; plain heads, modules_to_save and merged adapters stay fused."""

import inspect
import os
import textwrap
import types

import pytest
import torch

peft = pytest.importorskip("peft")
gemma3 = pytest.importorskip("transformers.models.gemma3.modeling_gemma3")
from peft.tuners.lora.layer import Linear as LoraLinear  # noqa: E402
from transformers.loss.loss_utils import ForCausalLMLoss  # noqa: E402
from unsloth_zoo import compiler  # noqa: E402
from unsloth_zoo.fused_losses import cross_entropy_loss as ce  # noqa: E402
from unsloth_zoo.fused_losses import unsloth_fused_ce_loss, unsloth_fused_lm_head_loss  # noqa: E402
from unsloth_zoo.fused_losses.forward_adapter import lm_head_needs_module_forward  # noqa: E402
from unsloth_zoo.fused_losses.forward_install import _can_fuse_loss, _fuses_model  # noqa: E402

V, H = 96, 32


@pytest.fixture(autouse = True)
def _fresh_state(monkeypatch):
    # The CE dispatcher reads its compile flag at import: set the module value, not the environment.
    monkeypatch.setattr(ce, "_FUSED_CE_COMPILE_DISABLED", True)
    monkeypatch.setattr(ce, "_WARNED_MODULE_HEADS", set())


class _Spy:
    """Counts calls of the module-forward route."""

    def __init__(self, monkeypatch):
        self.calls = 0
        real = ce._module_head_ce_loss

        def spy(*args, **kwargs):
            self.calls += 1
            return real(*args, **kwargs)
        monkeypatch.setattr(ce, "_module_head_ce_loss", spy)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(V, H)
        self.proj = torch.nn.Linear(H, H)
        self.lm_head = torch.nn.Linear(H, V, bias = False)

    def forward(self, ids):
        return self.lm_head(self.proj(self.embed(ids)))


def _ids():
    return torch.randint(0, V, (2, 9), generator = torch.Generator().manual_seed(1))


def _wrap(dropout = 0.0, mts = False, **extra):
    torch.manual_seed(0)
    if mts:
        cfg = peft.LoraConfig(r = 4, target_modules = ["proj"], modules_to_save = ["lm_head"])
    else:
        cfg = peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = ["proj", "lm_head"], lora_dropout = dropout,
                              init_lora_weights = False, **extra)
    model = peft.get_peft_model(Toy(), cfg).train()
    inner = model.base_model.model
    if mts:
        with torch.no_grad():  # a per-row perturbation, so which head scored shows in the loss
            weight = inner.lm_head.modules_to_save["default"].weight
            weight.add_(0.5 * torch.randn(weight.shape, generator = torch.Generator().manual_seed(7)))
    return model, inner


def _grad(loss, params):
    return torch.autograd.grad(loss, params, allow_unused = True, materialize_grads = True)


def _assert_match(inner, model, ids, rtol = 1e-4, atol = 1e-6, hidden_fn = None):
    params = [p for p in model.parameters() if p.requires_grad]
    hidden = hidden_fn or (lambda: inner.proj(inner.embed(ids)))
    torch.manual_seed(3)
    want = ForCausalLMLoss(inner.lm_head(hidden()).float(), ids, vocab_size = V)
    want_grads = _grad(want, params)
    torch.manual_seed(3)
    got = unsloth_fused_lm_head_loss(hidden(), inner.lm_head, ids, vocab_size = V, torch_compile = False)
    got_grads = _grad(got, params)
    torch.testing.assert_close(got, want, rtol = rtol, atol = atol)
    for g, w in zip(got_grads, want_grads):
        torch.testing.assert_close(g, w, rtol = rtol, atol = atol)
    return dict(zip([n for n, p in model.named_parameters() if p.requires_grad], got_grads))


@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_lora_head_trains_through_its_forward(dropout, monkeypatch):
    spy = _Spy(monkeypatch)
    model, inner = _wrap(dropout = dropout)
    grads = _assert_match(inner, model, _ids())
    assert grads["base_model.model.lm_head.lora_B.default.weight"].abs().sum() > 0
    assert spy.calls == 1


@pytest.mark.parametrize("extra", [dict(use_dora = True), dict(lora_bias = True)])
def test_lora_variants_train_through_the_heads_forward(extra, capsys):
    model, inner = _wrap(**extra)
    _assert_match(inner, model, _ids())
    assert "the loss calls the head's own forward" in capsys.readouterr().out


def test_adalora_head_uses_its_projection():
    torch.manual_seed(0)
    model = peft.get_peft_model(
        Toy(), peft.AdaLoraConfig(target_modules = ["proj", "lm_head"], total_step = 10, init_r = 4,
                                  init_lora_weights = False),
    ).train()
    inner = model.base_model.model
    assert lm_head_needs_module_forward(inner.lm_head)
    _assert_match(inner, model, _ids())


class _FakeQuantLinear(torch.nn.Linear):
    """A custom base projection whose forward is not x @ weight.T (int8 round trip)."""

    def forward(self, x):
        scale = self.weight.detach().abs().amax(dim = 1, keepdim = True) / 127
        return torch.nn.functional.linear(x, torch.round(self.weight / scale).clamp(-127, 127) * scale)


def _quant_base(inner):
    quant = _FakeQuantLinear(H, V, bias = False)
    with torch.no_grad():
        quant.weight.copy_(inner.lm_head.base_layer.weight)
    quant.weight.requires_grad_(False)
    inner.lm_head.base_layer = quant


def test_quantized_base_under_live_lora_uses_the_heads_forward():
    model, inner = _wrap()
    _quant_base(inner)
    _assert_match(inner, model, _ids())


def test_disabled_lora_over_custom_base_uses_the_heads_forward():
    """Row 14: a disabled adapter runs base_layer(x), which a custom base computes its own way."""
    model, inner = _wrap()
    _quant_base(inner)
    inner.proj.base_layer.weight.requires_grad_(True)  # disabling freezes every adapter: keep one grad to compare
    with model.disable_adapter():
        assert lm_head_needs_module_forward(inner.lm_head)
        _assert_match(inner, model, _ids())


def test_merged_lora_over_custom_base_uses_the_heads_forward():
    """Row 15: merged into a custom base, the stored weight is not what the base forward computes."""
    model, inner = _wrap()
    _quant_base(inner)
    inner.lm_head.merge()
    inner.proj.base_layer.weight.requires_grad_(True)
    assert lm_head_needs_module_forward(inner.lm_head)
    _assert_match(inner, model, _ids())


def test_fp16_lora_head_matches_stock():
    """A large LoRA scale: PEFT applies it after B in the base dtype; the head's own forward keeps that order."""
    torch.manual_seed(0)
    model = peft.get_peft_model(
        Toy(), peft.LoraConfig(r = 1, lora_alpha = 64, target_modules = ["lm_head"], init_lora_weights = False)
    ).half().train()
    inner = model.base_model.model
    _assert_match(inner, model, _ids(), rtol = 1e-2, atol = 1e-3)


def test_fp16_overflow_on_an_ignored_row_stays_out_of_backward():
    """Row 12: an ignored row whose half precision logits overflow must not turn the gradients into NaN."""
    torch.manual_seed(0)
    model = peft.get_peft_model(
        Toy(), peft.LoraConfig(r = 4, target_modules = ["lm_head"], init_lora_weights = False)
    ).half().train()
    inner = model.base_model.model
    ids = _ids()
    labels = ids.clone()
    labels[:, 1] = -100  # shifted: row 0 of each sequence is ignored
    hidden = inner.embed(ids).detach().clone()
    lora_a = inner.lm_head.lora_A["default"]
    with torch.no_grad():
        lora_a.weight.fill_(2)
        hidden[:, 0] = 3e4  # finite, but overflows inside lora_A (saved for lora_B's backward) on the ignored row
        assert torch.isinf(lora_a(hidden[:, 0])).all()
    hidden.requires_grad_(True)
    loss = unsloth_fused_lm_head_loss(hidden, inner.lm_head, labels, torch_compile = False)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(hidden.grad).all()
    assert torch.isfinite(inner.lm_head.lora_B["default"].weight.grad).all()
    # The valid rows alone give the same loss and adapter gradient.
    grad = inner.lm_head.lora_B["default"].weight.grad.clone()
    inner.lm_head.zero_grad()
    keep = hidden.detach().clone()
    keep[:, 0] = 0
    want = unsloth_fused_lm_head_loss(keep, inner.lm_head, labels, torch_compile = False)
    want.backward()
    torch.testing.assert_close(loss, want)
    torch.testing.assert_close(grad, inner.lm_head.lora_B["default"].weight.grad)


def test_half_hidden_states_feed_an_fp32_lora_head():
    """Row 13: the kernel casts the head input to the weight dtype outside autocast; the module route does too."""
    model, inner = _wrap()
    ids = _ids()
    hidden = inner.proj(inner.embed(ids)).to(torch.bfloat16)
    loss = unsloth_fused_lm_head_loss(hidden, inner.lm_head, ids, torch_compile = False)
    want = ForCausalLMLoss(inner.lm_head(hidden.float()).float(), ids, vocab_size = V)
    torch.testing.assert_close(loss, want)


def test_confident_rows_keep_their_target_gradient():
    """Row 18: a 20-nat margin gives loss exp(-20), not 0, with the stable row nll the kernel uses."""
    base = torch.nn.Linear(2, 2, bias = False)
    with torch.no_grad():
        base.weight.copy_(torch.eye(2))  # logits = the input
    head = LoraLinear(base, "default", config = peft.LoraConfig(), r = 1, lora_alpha = 1)  # B = 0: adapter adds nothing
    hidden = torch.tensor([[[20.0, 0.0], [0.0, 0.0]]], requires_grad = True)
    labels = torch.tensor([[0, 0]])
    loss = unsloth_fused_lm_head_loss(hidden, head, labels, torch_compile = False)
    loss.backward()
    oracle = torch.nn.functional.cross_entropy(torch.tensor([[20.0, 0.0]], dtype = torch.float64), torch.tensor([0]))
    torch.testing.assert_close(loss.double(), oracle, rtol = 1e-4, atol = 0)
    assert hidden.grad[0, 0, 0] < 0  # the target logit still gets its gradient


def test_merged_and_disabled_heads_stay_fused(monkeypatch):
    spy = _Spy(monkeypatch)
    model, inner = _wrap()
    ids = _ids()
    inner.proj.base_layer.weight.requires_grad_(True)
    with model.disable_adapter():
        assert not lm_head_needs_module_forward(inner.lm_head)
        _assert_match(inner, model, ids)  # disabled and unmerged: forward is base_layer(x), the kernel's product
    assert spy.calls == 0
    inner.lm_head.merge()
    model.base_model.model.proj.merge()
    _assert_match(inner, model, ids)  # fully merged into a plain Linear: the kernel
    assert spy.calls == 0


def test_merged_then_disabled_head_unmerges_like_its_forward():
    model, inner = _wrap()
    ids = _ids()
    inner.proj.base_layer.weight.requires_grad_(True)
    inner.lm_head.merge()
    with model.disable_adapter():
        assert lm_head_needs_module_forward(inner.lm_head)
        _assert_match(inner, model, ids)
        assert not inner.lm_head.merged


def test_modules_to_save_head_trains_the_copy(monkeypatch):
    spy = _Spy(monkeypatch)
    model, inner = _wrap(mts = True)
    assert not lm_head_needs_module_forward(inner.lm_head)
    grads = _assert_match(inner, model, _ids())
    assert "base_model.model.lm_head.original_module.weight" not in grads
    assert grads["base_model.model.lm_head.modules_to_save.default.weight"].abs().sum() > 0
    assert spy.calls == 0


@pytest.mark.parametrize("mts", [False, True])
def test_per_example_adapter_names_take_the_heads_forward(mts):
    """Row 1: the adapter_names hook sits on a LoRA head or on a modules_to_save wrapper; each row picks its head."""
    model, inner = _wrap(mts = mts)
    model.eval()  # PEFT takes adapter_names only in eval (labelled evaluation)
    with model.base_model._enable_peft_forward_hooks(adapter_names = ["__base__", "default"]):
        assert ce._lm_head_module_reason(inner.lm_head) == "per-example adapter_names"
        _assert_match(inner, model, _ids())


def _nest(outer_active):
    """LoRA inside LoRA: an outer tuner whose own adapter is inactive, or merged, over a live inner LoRA."""
    torch.manual_seed(0)
    cfg = peft.LoraConfig(init_lora_weights = False)
    # config by keyword: PEFT 0.18 has no config parameter (a positional one binds to r), later versions do.
    inner_head = LoraLinear(torch.nn.Linear(H, V, bias = False), "inner", config = cfg, r = 2, lora_alpha = 4,
                            init_lora_weights = False)
    outer = LoraLinear(inner_head, "outer", config = cfg, r = 2, lora_alpha = 4, init_lora_weights = False)
    if not outer_active:
        outer._active_adapter = ["nothing"]
    return outer


@pytest.mark.parametrize("state", ["inactive", "merged"])
def test_outer_tuner_does_not_hide_a_live_inner_one(state):
    """Rows 5 and 17: an inactive or merged outer tuner still runs the inner LoRA in its forward."""
    head = _nest(outer_active = state != "inactive")
    if state == "merged":
        head.merge(adapter_names = ["outer"])
    assert lm_head_needs_module_forward(head)
    hidden = torch.randn(2, 9, H, requires_grad = True)
    ids = _ids()
    want = ForCausalLMLoss(head(hidden).float(), ids, vocab_size = V)
    got = unsloth_fused_lm_head_loss(hidden, head, ids, torch_compile = False)
    torch.testing.assert_close(got, want)
    (g,) = torch.autograd.grad(got, [head.base_layer.lora_B["inner"].weight])
    (w,) = torch.autograd.grad(want, [head.base_layer.lora_B["inner"].weight])
    torch.testing.assert_close(g, w)


def test_dotted_adapter_containers_are_resolved():
    """Row 28: PEFT layer names such as token_adapter.trainable_tokens_delta are attribute paths."""
    holder = types.SimpleNamespace(trainable_tokens_delta = {"default": None})
    layer = types.SimpleNamespace(adapter_layer_names = ("token_adapter.trainable_tokens_delta",),
                                  token_adapter = holder, merged_adapters = ["default"])
    assert ce._adapter_containers_fully_merged(layer)
    layer.merged_adapters = []
    assert not ce._adapter_containers_fully_merged(layer)


def _module_vs_kernel(head, hidden, labels, **kw):
    want = unsloth_fused_ce_loss(trainer = None, hidden_states = hidden, lm_head_weight = head.weight,
                                 lm_head_bias = None, labels = labels, torch_compile = False, **kw)
    mask = kw.pop("mask", None)
    n_items = kw.pop("n_items", None)
    got = ce._module_head_ce_loss(head, hidden, labels, mask, n_items, kw.pop("shift_labels", True), kw)
    torch.testing.assert_close(got, want, rtol = 1e-5, atol = 1e-6)


@pytest.mark.parametrize("shift", [True, False])
def test_module_ce_masks_shifted_and_pre_aligned_targets(shift):
    torch.manual_seed(0)
    head = torch.nn.Linear(H, V, bias = False)
    hidden = torch.randn(2, 9, H)
    labels = _ids().to(torch.int32)
    mask = torch.ones(2, 9, dtype = torch.long)
    mask[0, 4:] = 0
    got = ce._module_head_ce_loss(head, hidden, labels, mask, torch.tensor(7), shift, {})
    targets, keep = labels.long(), mask
    if shift:
        targets = torch.cat((targets[:, 1:], torch.full_like(targets[:, :1], -100)), dim = 1)
        keep = torch.cat((mask[:, 1:], torch.ones_like(mask[:, :1])), dim = 1)
    targets = targets.masked_fill(keep == 0, -100)
    want = torch.nn.functional.cross_entropy(head(hidden).reshape(-1, V), targets.reshape(-1), reduction = "sum") / 7
    torch.testing.assert_close(got, want)
    if shift:
        _module_vs_kernel(head, hidden, labels, mask = mask, n_items = torch.tensor(7))


def test_module_ce_matches_the_kernel_transforms_bias_and_smoothing():
    torch.manual_seed(0)
    head = torch.nn.Linear(H, V, bias = False)
    _module_vs_kernel(head, torch.randn(2, 9, H), _ids(), logit_softcapping = 5.0, logit_scale_multiply = 2.0,
                      logit_scale_divide = 3.0, label_smoothing = 0.1, logits_bias = torch.randn(V))


def test_module_ce_all_ignored_is_zero_and_row_mismatch_raises():
    torch.manual_seed(0)
    head = torch.nn.Linear(H, V, bias = False)
    hidden = torch.randn(2, 9, H)
    ignored = torch.full((2, 9), -100)
    assert ce._module_head_ce_loss(head, hidden, ignored, None, None, True, {}).item() == 0.0
    assert ce._module_head_ce_loss(head, hidden, ignored, None, 0, True, {}).item() == 0.0
    with pytest.raises(ValueError, match = "rows for"):
        ce._module_head_ce_loss(head, hidden[:, -4:], _ids(), None, None, True, {})


def test_module_ce_moves_the_count_to_the_loss_device(monkeypatch):
    # n_items on another GPU than the head (a split model): the divisor must be moved to the logits' device.
    moved = []

    class _Count(torch.Tensor):
        def to(self, *args, **kwargs):
            moved.append(kwargs.get("device", args[0] if args else None))
            return torch.Tensor.to(self.as_subclass(torch.Tensor), *args, **kwargs)
    torch.manual_seed(0)
    head = torch.nn.Linear(H, V, bias = False)
    hidden = torch.randn(2, 9, H)
    got = ce._module_head_ce_loss(head, hidden, _ids(), None, torch.tensor([7.0, 7.0]).as_subclass(_Count), True, {})
    assert moved and all(torch.device(d) == head.weight.device for d in moved)
    want = ce._module_head_ce_loss(head, hidden, _ids(), None, 7, True, {})
    torch.testing.assert_close(got.as_subclass(torch.Tensor), want)
    single = ce._module_head_ce_loss(head, hidden, _ids(), None, torch.tensor([7.0], dtype = torch.float64), True, {})
    assert single.shape == torch.Size([]) and single.dtype == torch.float32
    torch.testing.assert_close(single, want)


def test_module_route_follows_the_label_smoothing_override():
    model, inner = _wrap()
    ids = _ids()
    hidden = inner.proj(inner.embed(ids))
    with ce.unsloth_label_smoothing_override(0.2, torch.tensor(5.0)) as state:
        got = unsloth_fused_lm_head_loss(hidden, inner.lm_head, ids, torch_compile = False)
    assert state.engaged == 1
    logits = inner.lm_head(hidden).float()
    want = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1), reduction = "sum", label_smoothing = 0.2,
    ) / 5.0
    torch.testing.assert_close(got, want)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason = "needs a second device for the head")
def test_split_device_hidden_states_move_to_the_head():
    model, inner = _wrap()
    inner.lm_head.cuda()
    hidden = inner.proj(inner.embed(_ids()))
    loss = unsloth_fused_lm_head_loss(hidden, inner.lm_head, _ids(), torch_compile = False)
    loss.backward()
    assert inner.lm_head.lora_B["default"].weight.grad is not None


def test_fullgraph_trace_with_the_notice(capsys):
    model, inner = _wrap()
    ids = _ids()

    def loss_fn(hidden):
        return unsloth_fused_lm_head_loss(hidden, inner.lm_head, ids, torch_compile = False)

    hidden = inner.proj(inner.embed(ids)).detach()
    compiled = torch.compile(loss_fn, fullgraph = True, backend = "eager")
    torch.testing.assert_close(compiled(hidden), loss_fn(hidden))
    assert capsys.readouterr().out.count("the loss calls the head's own forward") == 1


def _fused_gemma3_forward():
    # Gemma 3 as in test_fused_lm_head_qat.py: the conftest's unsloth import leaves its forward stock.
    source = inspect.getsource(gemma3.Gemma3ForCausalLM.forward)
    source = compiler.fixup_fused_lm_head(source)
    source, _ = compiler.apply_fused_lm_head(source, "Gemma3ForCausalLM")
    assert "explicit_shift_labels" in source  # cross_entropy_replacement_2
    return _exec_forward(source, dict(vars(gemma3)))


def _gemma3_model(softcap, use_dora, dtype = torch.float32):
    config = gemma3.Gemma3TextConfig(
        vocab_size = V, hidden_size = H, intermediate_size = 64, num_hidden_layers = 1, num_attention_heads = 2,
        num_key_value_heads = 1, head_dim = 16, final_logit_softcapping = softcap, attn_implementation = "eager",
    )
    torch.manual_seed(0)
    model = peft.get_peft_model(
        gemma3.Gemma3ForCausalLM(config).to(dtype),
        peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj", "lm_head"], init_lora_weights = False,
                        use_dora = use_dora),
    ).train()
    inner = model.base_model.model
    inner.loss_function = ForCausalLMLoss
    return model, inner


@pytest.mark.parametrize("use_dora", [False, True])
@pytest.mark.parametrize("softcap", [None, 30.0])
def test_pattern2_forward_trains_the_head_adapter(softcap, use_dora, monkeypatch):
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    model, inner = _gemma3_model(softcap, use_dora)
    ids = _ids()
    params = [p for p in model.parameters() if p.requires_grad]
    want = model(input_ids = ids, labels = ids).loss
    want_grads = _grad(want, params)
    out = _fused_gemma3_forward()(inner, input_ids = ids, labels = ids)
    assert out.logits.numel() == 0  # row 2: logits stay the EMPTY_LOGITS sentinel when not requested
    got_grads = _grad(out.loss, params)
    torch.testing.assert_close(out.loss, want)
    for g, w in zip(got_grads, want_grads):
        torch.testing.assert_close(g, w, rtol = 1e-4, atol = 1e-6)


def test_pattern2_bf16_softcap_rounds_like_stock(monkeypatch):
    """Row 3: the softcap runs in the head's dtype before the fp32 loss, as in the stock forward."""
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    model, inner = _gemma3_model(30.0, False, torch.bfloat16)
    ids = _ids()
    want = model(input_ids = ids, labels = ids).loss
    got = _fused_gemma3_forward()(inner, input_ids = ids, labels = ids).loss
    torch.testing.assert_close(got.float(), want.float(), rtol = 2e-5, atol = 0)


def test_pattern2_follows_the_label_smoothing_override_and_ignored_batches(monkeypatch):
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    model, inner = _gemma3_model(None, True)
    forward = _fused_gemma3_forward()
    ids = _ids()
    with ce.unsloth_label_smoothing_override(0.1, torch.tensor(8.0)) as state:
        got = forward(inner, input_ids = ids, labels = ids).loss
    assert state.engaged == 1
    logits = model(input_ids = ids).logits.float()
    want = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1), reduction = "sum", label_smoothing = 0.1,
    ) / 8.0
    torch.testing.assert_close(got, want)
    assert forward(inner, input_ids = ids, labels = torch.full_like(ids, -100)).loss.item() == 0.0


PATTERN_1 = """
def forward(self, input_ids=None, labels=None, **kwargs):
    hidden_states = self.proj(self.embed(input_ids))
    logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        logits = logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
    return types.SimpleNamespace(loss=loss, logits=logits)
"""

# A loss call without **kwargs (Qwen3-VL, ViP-LLaVA on transformers 4.57).
PATTERN_2_NO_KWARGS = """
def forward(self, input_ids=None, labels=None):
    hidden_states = self.proj(self.embed(input_ids))
    logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
    return types.SimpleNamespace(loss=loss, logits=logits)
"""

PATTERN_3 = """
def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
    hidden_states = self.proj(self.embed(input_ids))
    logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:]
            shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
            shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
        )
    return types.SimpleNamespace(loss=loss, logits=logits)
"""

AST_CANONICAL = """
def forward(self, input_ids=None, labels=None, **kwargs):
    hidden_states = self.proj(self.embed(input_ids))
    logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size, **kwargs)
    return types.SimpleNamespace(loss=loss, logits=logits)
"""


def _exec_forward(source, base_scope = None, header = True):
    from _compiled_module_names import fused_loss_header_names

    scope = dict(base_scope or {})
    scope.update(torch = torch, nn = torch.nn, CrossEntropyLoss = torch.nn.CrossEntropyLoss, types = types, os = os)
    if header:
        scope.update(fused_loss_header_names())
        scope.update(
            UNSLOTH_ENABLE_CCE = False, HAS_CUT_CROSS_ENTROPY = False, UNSLOTH_COMPILE_DISABLE = True,
            unsloth_fused_ce_loss = unsloth_fused_ce_loss, _can_fuse_loss = _can_fuse_loss, _fuses_model = _fuses_model,
            unsloth_training_stance = lambda: None, EMPTY_LOGITS = torch.empty(0),
        )
    source = textwrap.dedent(source)
    exec(source[source.index("def forward"):], scope)
    return scope["forward"]


def _check_against_stock(stock, rewritten, model, inner, **kw):
    params = [p for p in model.parameters() if p.requires_grad]
    want = stock(inner, **kw).loss
    want_grads = torch.autograd.grad(want, params)
    out = rewritten(inner, **kw)
    got_grads = torch.autograd.grad(out.loss, params)
    torch.testing.assert_close(out.loss, want)
    for g, w in zip(got_grads, want_grads):
        torch.testing.assert_close(g, w, rtol = 1e-4, atol = 1e-6)
    return out


@pytest.mark.parametrize("use_dora", [False, True])
@pytest.mark.parametrize("pattern", [PATTERN_1, PATTERN_2_NO_KWARGS, PATTERN_3], ids = ["p1", "p2_no_kwargs", "p3"])
def test_regex_patterns_train_the_head_adapter(pattern, use_dora, monkeypatch):
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    new, fused = compiler.apply_fused_lm_head(pattern, "ToyForCausalLM")
    assert fused and "lm_head_module       = self.lm_head" in new
    model, inner = _wrap(use_dora = use_dora)
    inner.config = types.SimpleNamespace(vocab_size = V)
    inner.loss_function = ForCausalLMLoss
    ids = _ids()
    kw = dict(input_ids = ids, labels = ids)
    if pattern is PATTERN_3:
        mask = torch.ones_like(ids)
        mask[0, 5:] = 0
        kw["attention_mask"] = mask
    out = _check_against_stock(_exec_forward(pattern, header = False), _exec_forward(new), model, inner, **kw)
    assert out.logits.numel() == 0


@pytest.mark.parametrize("use_dora", [False, True])
def test_ast_rewritten_forward_trains_the_head_adapter(use_dora, monkeypatch):
    from unsloth_zoo.fused_losses.ast_rewriter import rewrite_forward_source

    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    new, _ = rewrite_forward_source(AST_CANONICAL)
    assert "unsloth_fused_lm_head_loss(" in new
    # Row 25: only the names forward_install puts in the namespace, no compiler header.
    rewritten = _exec_forward(new, dict(
        unsloth_fused_lm_head_loss = unsloth_fused_lm_head_loss, _can_fuse_loss = _can_fuse_loss, _fuses_model = _fuses_model,
        EMPTY_LOGITS = torch.empty(0),
    ), header = False)
    model, inner = _wrap(use_dora = use_dora)
    inner.config = types.SimpleNamespace(vocab_size = V)
    inner.loss_function = ForCausalLMLoss
    ids = _ids()
    out = _check_against_stock(_exec_forward(AST_CANONICAL, header = False), rewritten, model, inner,
                               input_ids = ids, labels = ids)
    assert out.logits.numel() == 0


def test_module_ce_rejects_out_of_vocabulary_targets():
    """Row 9: like F.cross_entropy and the kernel, a labelled id outside the vocabulary raises."""
    head = torch.nn.Linear(H, V, bias = False)
    hidden = torch.randn(2, 9, H)
    for bad in (V, -3):
        labels = _ids()
        labels[0, 3] = bad
        with pytest.raises(IndexError, match = "out of bounds"):
            ce._module_head_ce_loss(head, hidden, labels, None, None, True, {})
    labels = _ids()
    labels[0, 0] = V  # dropped by the shift: never a target
    ce._module_head_ce_loss(head, hidden, labels, None, None, True, {})


@pytest.mark.parametrize("smoothing", [None, -0.1, 1.5])
def test_module_ce_label_smoothing_matches_the_kernel_contract(smoothing):
    """Row 10: None means no smoothing and a coefficient outside [0, 1] raises, as in fused_linear."""
    torch.manual_seed(0)
    head = torch.nn.Linear(H, V, bias = False)
    hidden = torch.randn(2, 9, H)
    if smoothing is None:
        got = ce._module_head_ce_loss(head, hidden, _ids(), None, None, True, dict(label_smoothing = None))
        torch.testing.assert_close(got, ce._module_head_ce_loss(head, hidden, _ids(), None, None, True, {}))
    else:
        with pytest.raises(ValueError, match = "label_smoothing"):
            ce._module_head_ce_loss(head, hidden, _ids(), None, None, True, dict(label_smoothing = smoothing))


class _FP8Linear(torch.nn.Linear):
    """FP8 storage dequantized in forward: the input must keep its own dtype."""

    def __init__(self, src):
        super().__init__(src.in_features, src.out_features, bias = False)
        self.weight = torch.nn.Parameter(src.weight.detach().to(torch.float8_e4m3fn), requires_grad = False)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight.to(x.dtype))


def test_storage_dtype_is_not_an_activation_dtype():
    """Row 13: an FP8 stored base must not get FP8 activations (stock lm_head(hidden) works)."""
    torch.manual_seed(0)
    head = LoraLinear(_FP8Linear(torch.nn.Linear(H, V, bias = False)), "default", config = peft.LoraConfig(),
                      r = 2, lora_alpha = 4, init_lora_weights = False)
    head.lora_A.float(), head.lora_B.float()  # PEFT moves the adapter to the base's (FP8) dtype
    hidden = torch.randn(2, 9, H)
    ids = _ids()
    got = unsloth_fused_lm_head_loss(hidden, head, ids, torch_compile = False)
    torch.testing.assert_close(got, ForCausalLMLoss(head(hidden).float(), ids, vocab_size = V))


def test_accelerate_hooked_head_still_gets_its_input_dtype():
    """Row 17: an Accelerate hook places the input, but the dtype cast for a resident fp32 head still applies."""
    model, inner = _wrap()
    inner.lm_head._hf_hook = types.SimpleNamespace()
    ids = _ids()
    hidden = inner.proj(inner.embed(ids)).to(torch.bfloat16)
    loss = unsloth_fused_lm_head_loss(hidden, inner.lm_head, ids, torch_compile = False)
    torch.testing.assert_close(loss, ForCausalLMLoss(inner.lm_head(hidden.float()).float(), ids, vocab_size = V))


def test_every_merged_tuner_base_is_checked():
    """Row 14: a merged tuner over a custom base is caught even when a merged plain one is visited first."""

    class _Pair(torch.nn.Module):
        def __init__(self, first, second):
            super().__init__()
            self.first, self.second = first, second

    def merged(base):
        layer = LoraLinear(base, "default", config = peft.LoraConfig(), r = 2, lora_alpha = 4,
                           init_lora_weights = False)
        layer.merge()
        return layer
    plain, custom = merged(torch.nn.Linear(H, V, bias = False)), merged(_FakeQuantLinear(H, V, bias = False))
    assert ce._lm_head_module_reason(_Pair(plain, custom)) is not None
    assert ce._lm_head_module_reason(_Pair(custom, plain)) is not None
    assert ce._lm_head_module_reason(_Pair(plain, merged(torch.nn.Linear(H, V, bias = False)))) is None


def test_merged_trainable_tokens_head_keeps_its_bias():
    """Row 18: a merged TrainableTokensWrapper exposes weight only; its base bias must still reach the loss."""
    from peft.utils.other import TrainableTokensWrapper

    torch.manual_seed(0)
    head = TrainableTokensWrapper(torch.nn.Linear(H, V, bias = True), "default", token_indices = [0, 1])
    with torch.no_grad():
        head.token_adapter.trainable_tokens_delta["default"].add_(1.0)
    head.token_adapter.merge()
    assert getattr(head, "bias", None) is None
    hidden = torch.randn(2, 9, H)
    ids = _ids()
    got = unsloth_fused_lm_head_loss(hidden, head, ids, torch_compile = False)
    torch.testing.assert_close(got, ForCausalLMLoss(head(hidden).float(), ids, vocab_size = V))


def test_unknown_adapter_container_is_treated_as_live():
    """Row 23: a container without keys() (a bare Parameter) is not a layout the merge check knows."""
    layer = types.SimpleNamespace(adapter_layer_names = ("delta",), delta = torch.nn.Parameter(torch.zeros(1)),
                                  merged_adapters = ["default"])
    assert not ce._adapter_containers_fully_merged(layer)


def test_detector_is_exported_from_the_package():
    """Row 29: fused_losses star-imports cross_entropy_loss."""
    from unsloth_zoo.fused_losses import lm_head_needs_module_forward as exported
    assert exported is lm_head_needs_module_forward


class _ActQuant(torch.nn.Module):
    """A non-idempotent stand-in for a TorchAO activation fake quantizer, counting its calls."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x * 0.5


class _QATLinear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation_fake_quantizer = _ActQuant()

    def forward(self, x):
        return torch.nn.functional.linear(self.activation_fake_quantizer(x), self.weight)


class _QATToy(Toy):
    def __init__(self):
        super().__init__()
        self.lm_head = _QATLinear(H, V, bias = False)


def _qat_mixed_batch():
    """modules_to_save over a QAT head, scored per row by adapter_names: each selected head quantizes once."""
    torch.manual_seed(0)
    cfg = peft.LoraConfig(r = 4, target_modules = ["proj"], modules_to_save = ["lm_head"])
    model = peft.get_peft_model(_QATToy(), cfg).eval()
    inner = model.base_model.model
    with torch.no_grad():
        weight = inner.lm_head.modules_to_save["default"].weight
        weight.add_(0.5 * torch.randn(weight.shape, generator = torch.Generator().manual_seed(7)))
    quantizers = [m for m in inner.lm_head.modules() if isinstance(m, _ActQuant)]
    assert len(quantizers) == 2
    return model, inner, quantizers


@pytest.mark.parametrize("route", ["ast", "p1", "p2_no_kwargs", "p3"])
def test_qat_head_is_quantized_once_on_the_module_route(route, monkeypatch):
    """Rows 2-5: the module route passes raw activations; the selected heads apply their own quantizer once."""
    from unsloth_zoo.fused_losses.ast_rewriter import rewrite_forward_source

    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    pattern = dict(ast = AST_CANONICAL, p1 = PATTERN_1, p2_no_kwargs = PATTERN_2_NO_KWARGS, p3 = PATTERN_3)[route]
    if route == "ast":
        new, _ = rewrite_forward_source(pattern)
        rewritten = _exec_forward(new, dict(
            unsloth_fused_lm_head_loss = unsloth_fused_lm_head_loss, _can_fuse_loss = _can_fuse_loss, _fuses_model = _fuses_model,
            EMPTY_LOGITS = torch.empty(0),
        ), header = False)
    else:
        new, fused = compiler.apply_fused_lm_head(pattern, "ToyForCausalLM")
        assert fused
        rewritten = _exec_forward(new)
    model, inner, quantizers = _qat_mixed_batch()
    inner.config = types.SimpleNamespace(vocab_size = V)
    inner.loss_function = ForCausalLMLoss
    ids = _ids()
    with model.base_model._enable_peft_forward_hooks(adapter_names = ["__base__", "default"]):
        want = _exec_forward(pattern, header = False)(inner, input_ids = ids, labels = ids).loss
        assert [q.calls for q in quantizers] == [1, 1]
        got = rewritten(inner, input_ids = ids, labels = ids).loss
    assert [q.calls for q in quantizers] == [2, 2]
    torch.testing.assert_close(got, want)


@pytest.mark.parametrize("pattern", [PATTERN_1, PATTERN_2_NO_KWARGS], ids = ["p1", "p2_no_kwargs"])
def test_cce_gate_skips_a_live_lora_head_and_admits_a_merged_one(pattern, monkeypatch):
    """Row 30: with Cut Cross Entropy on, a live head adapter stays off CCE (which reads only the weight)."""
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    new, fused = compiler.apply_fused_lm_head(pattern, "ToyForCausalLM")
    assert fused
    calls = []

    def fake_cce(**kwargs):
        calls.append(kwargs)
        return torch.zeros(())
    forward = _exec_forward(new)
    forward.__globals__.update(UNSLOTH_ENABLE_CCE = True, HAS_CUT_CROSS_ENTROPY = True,
                               fused_linear_cross_entropy = fake_cce)
    model, inner = _wrap()
    model.to(torch.bfloat16)
    inner.config = types.SimpleNamespace(vocab_size = V)
    inner.loss_function = ForCausalLMLoss
    ids = _ids()
    forward(inner, input_ids = ids, labels = ids).loss.backward()
    assert calls == []
    assert inner.lm_head.lora_B["default"].weight.grad is not None
    inner.lm_head.merge()
    forward(inner, input_ids = ids, labels = ids)
    assert len(calls) == 1


def test_plain_head_needs_no_module_forward():
    assert not lm_head_needs_module_forward(torch.nn.Linear(H, V))
    assert not lm_head_needs_module_forward(None)

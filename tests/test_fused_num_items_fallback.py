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

"""Every loss branch of a fused forward divides by num_items_in_batch, not only the fused one."""
import ast
import os
import textwrap
import types

import pytest
import torch
import torch.nn.functional as F

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from unsloth_zoo.compiler import apply_fused_lm_head  # noqa: E402
from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_count_aware_cross_entropy  # noqa: E402
from unsloth_zoo.fused_losses.ast_rewriter import (  # noqa: E402
    rewrite_forward_source,
    rewrite_forward_source_spliced,
)


QWEN3_VL_5_16_1 = '''
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen3VLCausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )
'''

MAMBA_5_16_1 = '''
def forward(self, input_ids=None, labels=None, logits_to_keep=0, **kwargs):
    mamba_outputs = self.backbone(input_ids)
    hidden_states = mamba_outputs[0]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :].to(self.lm_head.weight.dtype)).float()
    loss = None
    if labels is not None:
        labels = labels.to(logits.device)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss, logits
'''

BART_ALIGNED = '''
def forward(self, input_ids=None, labels=None, **kwargs):
    outputs = self.model.decoder(input_ids)
    logits = self.lm_head(outputs[0])
    loss = None
    if labels is not None:
        labels = labels.to(logits.device)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
    return loss, logits
'''

T5_ENC_DEC = '''
def forward(self, input_ids=None, decoder_input_ids=None, labels=None, **kwargs):
    if labels is not None and decoder_input_ids is None:
        decoder_input_ids = self._shift_right(labels)
    decoder_outputs = self.decoder(input_ids=decoder_input_ids, **kwargs)
    sequence_output = decoder_outputs[0]
    lm_logits = self.lm_head(sequence_output)
    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        labels = labels.to(lm_logits.device)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
    return loss, lm_logits
'''

XGLM_KWARGLESS = '''
def forward(self, input_ids=None, labels=None, logits_to_keep=0, **kwargs):
    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])
    loss = None
    if labels is not None:
        loss = self.loss_function(
            logits,
            labels,
            vocab_size=self.config.vocab_size,
            pad_token_id=self.config.pad_token_id,
        )
    return loss, logits
'''


def _loss_function_calls(source):
    tree = ast.parse(textwrap.dedent(source))
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "loss_function"
    ]


def test_kwargless_vlm_fallback_branches_get_the_count():
    new, ok = apply_fused_lm_head(QWEN3_VL_5_16_1, "Qwen3VLForConditionalGeneration")
    assert ok
    compile(textwrap.dedent(new), "<fused>", "exec")
    calls = _loss_function_calls(new)
    assert len(calls) == 2
    causal, non_causal = calls
    for call in (causal, non_causal):
        assert not any(kw.arg == "num_items_in_batch" for kw in call.keywords), ast.unparse(call)
        starred = [ast.unparse(kw.value) for kw in call.keywords if kw.arg is None]
        assert starred == ["unsloth_loss_count_kwargs(self.loss_function, n_items)"], ast.unparse(call)


def test_non_causal_fallback_runs_a_loss_without_the_count_parameter():
    new, ok = apply_fused_lm_head(QWEN3_VL_5_16_1, "Qwen3VLForConditionalGeneration")
    non_causal = _loss_function_calls(new)[1]
    call = ast.Expression(non_causal)
    seen = {}

    class Holder:
        def loss_function(self, logits, labels, vocab_size):
            seen["called"] = True
            return 0.0

    from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_loss_count_kwargs
    labels = types.SimpleNamespace(to = lambda device: "labels")
    for n_items in (None, 7):
        seen.clear()
        env = dict(self = Holder(), n_items = n_items, logits = "logits", labels = labels,
                   unsloth_loss_count_kwargs = unsloth_loss_count_kwargs)
        env["self"].lm_head = types.SimpleNamespace(weight = types.SimpleNamespace(device = "cpu"))
        env["self"].config = types.SimpleNamespace(text_config = types.SimpleNamespace(vocab_size = 1))
        eval(compile(ast.fix_missing_locations(call), "<call>", "eval"), env)
        assert seen.get("called")
    assert "n_items              = n_items" in new


def test_captured_kwargs_are_left_as_they_were():
    source = QWEN3_VL_5_16_1.replace(
        "vocab_size=self.config.text_config.vocab_size)",
        "vocab_size=self.config.text_config.vocab_size, **kwargs)",
    )
    new, ok = apply_fused_lm_head(source, "Qwen3VLForConditionalGeneration")
    assert ok
    calls = _loss_function_calls(new)
    assert calls and all(
        any(kw.arg is None and ast.unparse(kw.value) == "kwargs" for kw in call.keywords)
        and not any(kw.arg == "num_items_in_batch" for kw in call.keywords)
        for call in calls
    )


def test_legacy_shifted_ce_threads_kwargs():
    new = rewrite_forward_source_spliced(MAMBA_5_16_1)[0]
    assert new is not None
    compile(new, "<fused>", "exec")
    assert "unsloth_fused_lm_head_loss(" in new
    fused = [line for line in new.splitlines() if "unsloth_fused_lm_head_loss(" in line]
    assert fused and all("**kwargs" in line for line in fused)
    assert "unsloth_count_aware_cross_entropy(" in new
    assert "n_items=kwargs.get('num_items_in_batch', None) if kwargs.get('num_items_in_batch', None) is not None else kwargs.get('n_items', None), shift=False)" in new


def test_decoder_only_aligned_ce_gets_the_count():
    new, cap = rewrite_forward_source_spliced(BART_ALIGNED)
    assert new is not None and cap.aligned
    compile(textwrap.dedent(new), "<fused>", "exec")
    fused = new.split("unsloth_fused_lm_head_loss(", 1)[1].split("\n", 1)[0]
    assert "shift_labels=False" in fused and "**kwargs" in fused
    assert "n_items=kwargs.get('num_items_in_batch', None) if kwargs.get('num_items_in_batch', None) is not None else kwargs.get('n_items', None), shift=False)" in new


def test_kwargless_hook_forward_gets_the_count():
    new = rewrite_forward_source(XGLM_KWARGLESS)[0]
    assert new is not None
    compile(new, "<fused>", "exec")
    count = ("kwargs.get('num_items_in_batch', None) if kwargs.get('num_items_in_batch', None) is not None "
             "else kwargs.get('n_items', None)")
    assert new.count(f"num_items_in_batch={count}") == 1, new
    assert f"**unsloth_loss_count_kwargs(self.loss_function, {count})" in new, new


def _run_legacy_unfused(n_items):
    new = rewrite_forward_source_spliced(MAMBA_5_16_1)[0]
    torch.manual_seed(0)
    vocab, hidden = 11, 8

    class Backbone(torch.nn.Module):
        def forward(self, input_ids):
            return (torch.randn(input_ids.shape[0], input_ids.shape[1], hidden, generator = gen),)

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Backbone()
            self.lm_head = torch.nn.Linear(hidden, vocab, bias = False)

    ns = {"torch": torch, "CrossEntropyLoss": torch.nn.CrossEntropyLoss, "os": os,
          "EMPTY_LOGITS": None, "unsloth_fused_lm_head_loss": None,
          "unsloth_count_aware_cross_entropy": unsloth_count_aware_cross_entropy}
    exec(new, ns)
    head = Head()
    input_ids = torch.zeros(2, 6, dtype = torch.long)
    labels = torch.randint(0, vocab, (2, 6))
    labels[0, :4] = -100
    kwargs = {} if n_items is None else {"num_items_in_batch": torch.tensor(n_items)}
    gen = torch.Generator().manual_seed(1)
    old = os.environ.get("UNSLOTH_RETURN_LOGITS")
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    try:
        loss, logits = ns["forward"](head, input_ids = input_ids, labels = labels, **kwargs)
    finally:
        if old is None:
            os.environ.pop("UNSLOTH_RETURN_LOGITS")
        else:
            os.environ["UNSLOTH_RETURN_LOGITS"] = old
    summed = F.cross_entropy(
        logits[..., :-1, :].reshape(-1, vocab), labels[..., 1:].reshape(-1), reduction = "sum",
    )
    count = (labels[..., 1:] != -100).sum()
    return loss, summed, count


def test_legacy_unfused_branch_divides_by_the_count():
    loss, summed, count = _run_legacy_unfused(n_items = 17)
    torch.testing.assert_close(loss, summed / 17)
    loss, summed, count = _run_legacy_unfused(n_items = None)
    torch.testing.assert_close(loss, summed / count)


def test_encoder_decoder_aligned_ce_threads_kwargs():
    new = rewrite_forward_source_spliced(T5_ENC_DEC)[0]
    assert new is not None
    compile(textwrap.dedent(new), "<fused>", "exec")
    fused = [line for line in new.splitlines() if "unsloth_fused_lm_head_loss(" in line]
    assert fused and all("shift_labels=False" in line and "**kwargs" in line for line in fused)
    assert "unsloth_count_aware_cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), n_items=kwargs.get('num_items_in_batch', None) if kwargs.get('num_items_in_batch', None) is not None else kwargs.get('n_items', None), shift=False)" in new

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

"""Heads no fused route can take still divide by num_items_in_batch.

A forward whose own mean token CrossEntropyLoss stays unfused (Llama 4 vision on <= 5.16.1 reads
logits off an inner model; ModernBertDecoder has no Linear head; deprecated models are never
compiled) is rewritten to route that CE through `unsloth_count_aware_cross_entropy`, which returns
sum / num_items_in_batch when Trainer passes the count and the stock mean otherwise. Aligned-label
heads (BartForCausalLM, encoder-decoders) are marked `_unsloth_counts_unshifted_labels` so the batch
counter counts what their CE averages over.
"""
import ast
import importlib.util
import os
import textwrap

import pytest
import torch
import torch.nn.functional as F
from torch import nn

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from unsloth_zoo.fused_losses.ast_rewriter import rewrite_count_aware_ce_spliced  # noqa: E402
from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_count_aware_cross_entropy  # noqa: E402


# Llama4ForConditionalGeneration.forward on transformers 5.16.1, docstring and blank lines dropped.
LLAMA4_CG_5_16_1 = """
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        vision_feature_select_strategy: str | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Llama4CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_select_strategy=vision_feature_select_strategy,
                return_dict=True,
            ).last_hidden_state
            vision_flat = image_features.view(-1, image_features.size(-1))
            projected_vision_flat = self.multi_modal_projector(vision_flat).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=projected_vision_flat
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, projected_vision_flat)
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        logits = outputs[0]
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return Llama4CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )
"""

# OpenLlamaForCausalLM.forward on transformers 4.57.6 (deprecated, no **kwargs).
OPEN_LLAMA_4_57_6 = """
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        if self.config.shared_input_output_embedding:
            logits = torch.einsum(
                "blh,vh->blv", hidden_states.to(self.model.embed_tokens.weight.device), self.model.embed_tokens.weight
            )
        else:
            logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
"""

# Two token CEs (GPT2DoubleHeadsModel shape): the multiple choice loss counts other labels.
TWO_CES = """
    def forward(self, input_ids=None, labels=None, mc_labels=None, **kwargs):
        lm_logits = self.lm_head(self.transformer(input_ids)[0])
        mc_logits = self.multiple_choice_head(lm_logits)
        mc_loss = None
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            mc_loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
        lm_loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return lm_loss, mc_loss
"""

RELABELLED = """
    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = self.lm_head(input_ids)
        loss = None
        if labels is not None:
            labels = labels.masked_fill(labels == self.config.audio_vocab_size, -100).reshape(-1)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.config.audio_vocab_size), labels)
        return loss
"""

WEIGHTED = """
    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = self.lm_head(input_ids)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss
"""


def _labels_block(source):
    fn = ast.parse(textwrap.dedent(source)).body[0]
    block = next(s for s in fn.body if isinstance(s, ast.If) and ast.unparse(s.test) == "labels is not None")
    return compile(ast.Module(body = block.body, type_ignores = []), "<labels>", "exec")


def _run(source, **env):
    ns = dict(torch = torch, nn = nn, CrossEntropyLoss = nn.CrossEntropyLoss,
              unsloth_count_aware_cross_entropy = unsloth_count_aware_cross_entropy, **env)
    exec(_labels_block(source), ns)
    return ns["loss"]


def _batch(vocab = 13, mask_row = True):
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 7, vocab, generator = gen)
    labels = torch.randint(0, vocab, (2, 7), generator = gen)
    labels[0, :3] = -100
    attention_mask = torch.ones(2, 7, dtype = torch.long)
    if mask_row:
        attention_mask[1, :2] = 0
    return logits, labels, attention_mask


def test_helper_matches_the_stock_mean_and_divides_by_the_count():
    logits, labels, _ = _batch()
    stock = nn.CrossEntropyLoss()(logits[..., :-1, :].reshape(-1, 13), labels[..., 1:].reshape(-1))
    torch.testing.assert_close(unsloth_count_aware_cross_entropy(logits, labels), stock)
    summed = F.cross_entropy(logits[..., :-1, :].reshape(-1, 13), labels[..., 1:].reshape(-1), reduction = "sum")
    torch.testing.assert_close(unsloth_count_aware_cross_entropy(logits, labels, 29), summed / 29)
    torch.testing.assert_close(unsloth_count_aware_cross_entropy(logits, labels, torch.tensor(29)), summed / 29)
    aligned = F.cross_entropy(logits.reshape(-1, 13), labels.reshape(-1), reduction = "sum")
    torch.testing.assert_close(unsloth_count_aware_cross_entropy(logits, labels, 5, shift = False), aligned / 5)
    ignored = torch.full_like(labels, -100)
    assert torch.isnan(unsloth_count_aware_cross_entropy(logits, ignored))
    assert unsloth_count_aware_cross_entropy(logits, ignored, 4) == 0


def test_helper_keeps_the_stock_dtype_and_sums_in_float32():
    # The stock block it replaces runs CE on the logits as they are (no float32 copy of a
    # vocab-sized tensor): the mean is that exact call, and only the per-token losses of the
    # counted sum are accumulated in float32, so a half precision sum cannot overflow.
    logits, labels, _ = _batch()
    half = logits.to(torch.bfloat16)
    stock = nn.CrossEntropyLoss()(half[..., :-1, :].reshape(-1, 13), labels[..., 1:].reshape(-1))
    mean = unsloth_count_aware_cross_entropy(half, labels)
    assert mean.dtype == torch.bfloat16 and torch.equal(mean, stock)
    per_token = F.cross_entropy(half[..., :-1, :].reshape(-1, 13), labels[..., 1:].reshape(-1),
                                reduction = "none").float().sum()
    counted = unsloth_count_aware_cross_entropy(half, labels, torch.tensor(29))
    assert counted.dtype == torch.float32
    torch.testing.assert_close(counted, per_token / 29)
    # 6000 tokens of loss ~ln(50000) sum past float16's 65504 limit; the float32 sum stays finite.
    big = torch.zeros(1, 6001, 4, dtype = torch.float16)
    big_labels = torch.zeros(1, 6001, dtype = torch.long)
    big[..., 0] = -12.0
    assert torch.isfinite(unsloth_count_aware_cross_entropy(big, big_labels, 6000))


def test_helper_mask_matches_the_filtered_stock_block():
    logits, labels, attention_mask = _batch()
    keep = attention_mask[:, -(logits.shape[1] - 1):] != 0
    stock = nn.CrossEntropyLoss()(logits[..., :-1, :][keep], labels[..., 1:][keep])
    torch.testing.assert_close(unsloth_count_aware_cross_entropy(logits, labels, mask = attention_mask), stock)


@pytest.mark.parametrize("source", [LLAMA4_CG_5_16_1, OPEN_LLAMA_4_57_6], ids = ["llama4_cg", "open_llama"])
def test_unfused_mean_ce_becomes_count_aware(source):
    new, shifted = rewrite_count_aware_ce_spliced(source)
    assert new is not None and shifted
    compile(textwrap.dedent(new), "<count>", "exec")
    assert "CrossEntropyLoss" not in new
    assert new.count("unsloth_count_aware_cross_entropy(") == 1
    assert "n_items=(kwargs.get('num_items_in_batch', None) if kwargs.get('num_items_in_batch', None) is not None else kwargs.get('n_items', None)), shift=False)" in new
    fn = ast.parse(textwrap.dedent(new)).body[0]
    assert fn.args.kwarg is not None and fn.args.kwarg.arg == "kwargs"
    # Everything outside the labels block is untouched.
    assert new.split("loss = None", 1)[0] == source.split("loss = None", 1)[0].replace(
        "return_dict: Optional[bool] = None,\n    ) ->", "return_dict: Optional[bool] = None, **kwargs,\n    ) ->"
    )


def test_llama4_filtered_block_numerics():
    new, _ = rewrite_count_aware_ce_spliced(LLAMA4_CG_5_16_1)
    logits, labels, attention_mask = _batch()
    for mask in (attention_mask, None):
        env = dict(logits = logits, labels = labels, attention_mask = mask)
        stock = _run(LLAMA4_CG_5_16_1, **env)
        torch.testing.assert_close(_run(new, kwargs = {}, **env), stock)
        keep = (mask if mask is not None else torch.ones_like(labels))[:, 1:] != 0
        valid = (labels[..., 1:] != -100) & keep
        summed = stock * valid.sum()
        torch.testing.assert_close(_run(new, kwargs = {"num_items_in_batch": torch.tensor(40)}, **env), summed / 40)


def test_open_llama_block_numerics():
    new, _ = rewrite_count_aware_ce_spliced(OPEN_LLAMA_4_57_6)
    logits, labels, _ = _batch()
    self = type("S", (), {"config": type("C", (), {"vocab_size": 13})()})()
    stock = _run(OPEN_LLAMA_4_57_6, logits = logits, labels = labels, self = self)
    torch.testing.assert_close(_run(new, logits = logits, labels = labels, self = self, kwargs = {}), stock)
    count = (labels[..., 1:] != -100).sum()
    out = _run(new, logits = logits, labels = labels, self = self, kwargs = {"num_items_in_batch": 50})
    torch.testing.assert_close(out, stock * count / 50)


@pytest.mark.parametrize(
    "source", [TWO_CES, WEIGHTED, RELABELLED], ids = ["auxiliary_ce", "label_smoothing", "relabelled"],
)
def test_other_objectives_are_left_alone(source):
    assert rewrite_count_aware_ce_spliced(source) == (None, None)


def test_deprecated_module_gets_the_count_aware_forward_from_the_hook(tmp_path):
    from unsloth_zoo.fused_losses import forward_install
    if not forward_install._transformers_version_ok():
        pytest.skip(reason = "hook needs transformers >= 4.56")
    header = (
        "from typing import Optional, Union\n"
        "import torch\n"
        "from torch.nn import CrossEntropyLoss\n"
        "from transformers.cache_utils import Cache\n"
        "from transformers.modeling_outputs import CausalLMOutputWithPast\n\n"
    )
    import inspect
    for name, cls_name in (
        ("transformers.models.deprecated.fake_open_llama.modeling_fake_open_llama", "FakeOpenLlamaForCausalLM"),
        ("transformers.models.fake_llama.modeling_fake_llama", "FakeLlamaForCausalLM"),
    ):
        path = tmp_path / (name.rsplit(".", 1)[-1] + ".py")
        path.write_text(
            header + f"class {cls_name}(torch.nn.Module):\n"
            + textwrap.indent(textwrap.dedent(OPEN_LLAMA_4_57_6), "    ")
        )
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = getattr(module, cls_name)
        installed = forward_install.install_for_class(cls)
        if ".deprecated." in name:
            assert installed
            assert "unsloth_count_aware_cross_entropy(" in inspect.getsource(cls.forward)
            assert "kwargs" in inspect.signature(cls.forward).parameters
            assert not getattr(cls, "_unsloth_counts_unshifted_labels", False)
        else:
            # Outside deprecated/ the compiler owns the rewrite, so it can still fuse it.
            assert not installed


def _fused(cls):
    import inspect
    from unsloth_zoo.compiler import fused_lm_head_forward
    return fused_lm_head_forward(cls.__name__, cls, cls.__module__, inspect.getsource(cls.forward))


def test_aligned_decoder_only_head_is_marked_unshifted():
    transformers = pytest.importorskip("transformers")
    cls = getattr(transformers, "BartForCausalLM")
    had = "_unsloth_counts_unshifted_labels" in cls.__dict__
    try:
        new, route, _ = _fused(cls)
        assert route == "ast"
        assert "shift_labels=False" in new and "num_items_in_batch" in new
        assert cls._unsloth_counts_unshifted_labels is True
    finally:
        if not had:
            cls.__dict__.get("_unsloth_counts_unshifted_labels") and delattr(cls, "_unsloth_counts_unshifted_labels")


@pytest.mark.parametrize("name", ["LlamaForCausalLM", "Qwen2_5_VLForConditionalGeneration"])
def test_fused_heads_do_not_take_the_count_route(name):
    transformers = pytest.importorskip("transformers")
    cls = getattr(transformers, name, None)
    if cls is None:
        pytest.skip(reason = f"{name} not in this transformers")
    new, route, _ = _fused(cls)
    assert route != "count" and "unsloth_count_aware_cross_entropy" not in new
    assert not getattr(cls, "_unsloth_counts_unshifted_labels", False)

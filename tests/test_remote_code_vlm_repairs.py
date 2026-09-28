# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Remote-code repairs, on a module copying the failing parts of Nemotron-3-Nano-Omni
loaded through transformers' dynamic module loader. CPU only."""

import inspect
import os
import textwrap

import pytest
import torch

_MODELING = textwrap.dedent('''
    import torch
    import torch.nn as nn
    from torch.nn import CrossEntropyLoss
    from transformers import LlamaConfig, LlamaForCausalLM, PretrainedConfig, PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast


    class TinyOmniConfig(PretrainedConfig):
        model_type = "tiny_omni_zoo_test"

        def __init__(self, img_context_token_id = 5, **kwargs):
            super().__init__(**kwargs)
            self.img_context_token_id = img_context_token_id
            self.llm_config = LlamaConfig(
                vocab_size = 32, hidden_size = 16, intermediate_size = 32, num_hidden_layers = 1,
                num_attention_heads = 2, num_key_value_heads = 1, max_position_embeddings = 64,
            )
            self.llm_config._attn_implementation = "eager"


    class LegacyHybridCache:
        """Like NemotronHHybridDynamicCache: not a transformers Cache, no get_query_offset."""

        is_compileable = False

        def __init__(self, seen = 0):
            self.seen = seen

        def get_seq_length(self, layer_idx = 0):
            return self.seen

        def get_mask_sizes(self, query_length, layer_idx):
            if isinstance(query_length, torch.Tensor):
                query_length = query_length.shape[0]
            return self.seen + query_length, 0


    class TinyOmni(PreTrainedModel):
        config_class = TinyOmniConfig
        main_input_name = "pixel_values"

        def __init__(self, config):
            super().__init__(config)
            self.language_model = LlamaForCausalLM(config.llm_config)
            self.vision_model = nn.Conv2d(3, 16, kernel_size = 4, stride = 4, bias = False)
            self.img_context_token_id = config.img_context_token_id
            self.post_init()

        def extract_feature(self, pixel_values):
            if isinstance(pixel_values, (list, tuple)):
                return torch.cat([self._extract_feature_single(pv) for pv in pixel_values], dim = 0)
            return self._extract_feature_single(pixel_values)

        def _extract_feature_single(self, pixel_values):
            return self.vision_model(pixel_values).flatten(2).transpose(1, 2)

        # Same body as the Nemotron-3-Nano-Omni remote forward (modeling.py:172-257), print dropped.
        def forward(
                self,
                pixel_values,
                input_ids = None,
                attention_mask = None,
                position_ids = None,
                image_flags = None,
                past_key_values = None,
                labels = None,
                inputs_embeds = None,
                use_cache = None,
                output_attentions = None,
                output_hidden_states = None,
                return_dict = None,
        ):
            return_dict = return_dict if return_dict is not None else True
            if inputs_embeds is None:
                inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            image_flags = image_flags.squeeze(-1)
            B, N, C = inputs_embeds.shape
            inputs_embeds = inputs_embeds.reshape(B * N, C)
            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            vit_batch_size = pixel_values.shape[0]
            vit_embeds = self.extract_feature(pixel_values)
            del pixel_values
            vit_embeds = vit_embeds[image_flags == 1]
            try:
                inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                n_token = selected.sum()
                inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds[:n_token]
            del vit_embeds
            inputs_embeds = inputs_embeds.reshape(B, N, C)
            outputs = self.language_model(
                inputs_embeds = inputs_embeds,
                attention_mask = attention_mask,
                position_ids = position_ids,
                past_key_values = past_key_values,
                use_cache = use_cache,
                output_attentions = output_attentions,
                output_hidden_states = output_hidden_states,
                return_dict = return_dict,
            )
            logits = outputs.logits
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
            return CausalLMOutputWithPast(loss = loss, logits = logits, past_key_values = outputs.past_key_values)


    class TinyRadioConfig(PretrainedConfig):
        model_type = "tiny_radio_zoo_test"

        def __init__(self, args = None, **kwargs):
            super().__init__(**kwargs)
            self.args = args or {"teachers": [{"name": "a"}, {"name": "b"}, {"name": "c"}, {"name": "d", "use_summary": False}]}


    class _RadioBase(nn.Module):
        def __init__(self, summary_idxs):
            super().__init__()
            self.register_buffer("summary_idxs", summary_idxs)

        def forward(self, x):
            return x[:, self.summary_idxs]


    class RADIOModel(PreTrainedModel):
        """Like nvidia/C-RADIOv4-H hf_model.RADIOModel: summary_idxs is computed, not stored."""
        config_class = TinyRadioConfig

        def __init__(self, config):
            super().__init__(config)
            summary_idxs = torch.tensor(
                [i for i, t in enumerate(config.args["teachers"]) if t.get("use_summary", True)],
                dtype = torch.int64,
            )
            self.radio_model = _RadioBase(summary_idxs = summary_idxs)
            self.post_init()

        def forward(self, x):
            return self.radio_model.forward(x)


    class PlainRemote(PreTrainedModel):
        """A remote class without the InternVL merge: must be left alone."""
        config_class = TinyOmniConfig

        def __init__(self, config):
            super().__init__(config)
            self.proj = nn.Linear(4, 4)
            self.post_init()

        def extract_feature(self, pixel_values):
            return pixel_values

        def forward(self, pixel_values, input_ids = None, image_flags = None, labels = None):
            return self.proj(pixel_values)
''')


@pytest.fixture(scope = "module")
def remote(tmp_path_factory):
    import transformers.dynamic_module_utils as dmu
    try:
        from unsloth_zoo.temporary_patches.remote_code_vlm import patch_remote_code_vlm
    except ImportError:  # unsloth-zoo without the repair: the tests show the original failures
        patch_remote_code_vlm = lambda: None
    patch_remote_code_vlm()
    repo = tmp_path_factory.mktemp("tiny_omni_repo")
    (repo / "modeling_tiny_omni.py").write_text(_MODELING)
    (repo / "config.json").write_text("{}")
    load = lambda name: dmu.get_class_from_dynamic_module(f"modeling_tiny_omni.{name}", str(repo))
    Omni = load("TinyOmni")
    import sys
    return sys.modules[Omni.__module__]


def _model(remote):
    torch.manual_seed(0)
    return remote.TinyOmni(remote.TinyOmniConfig()).float()


def _batch(token_counts, seq_len = 10, image_token = 5):
    rows = []
    for n in token_counts:
        row = [image_token] * n + list(range(6, 6 + seq_len - n))
        rows.append(row)
    return torch.tensor(rows)


def test_legacy_cache_gets_query_offset(remote):
    cache = remote.LegacyHybridCache(seen = 3)
    assert cache.get_query_offset(0) == cache.get_seq_length(0) == 3


def test_legacy_cache_builds_causal_mask(remote):
    from transformers import masking_utils
    original = getattr(masking_utils, "_unsloth_original_create_causal_mask", masking_utils.create_causal_mask)
    parameters = inspect.signature(original).parameters
    cfg = remote.TinyOmniConfig().llm_config
    embeds_name = "inputs_embeds" if "inputs_embeds" in parameters else "input_embeds"
    kwargs = {"config": cfg, embeds_name: torch.zeros(1, 2, 16),
              "attention_mask": torch.ones(1, 5, dtype = torch.long),
              "past_key_values": remote.LegacyHybridCache(seen = 3), "position_ids": torch.tensor([[3, 4]])}
    if "cache_position" in parameters:
        kwargs["cache_position"] = torch.tensor([3, 4])
    mask = original(**kwargs)
    if mask is not None:
        assert mask.shape[-1] == 5


def test_inplace_merge_trains_with_frozen_embedding(remote):
    model = _model(remote)
    model.language_model.get_input_embeddings().weight.requires_grad_(False)
    # Gradient checkpointing setups make the frozen embedding's output a leaf that requires grad.
    model.language_model.enable_input_require_grads()
    model.train()
    input_ids = _batch([4])
    out = model(pixel_values = torch.randn(1, 3, 8, 8), input_ids = input_ids,
                image_flags = torch.ones(1, 1, dtype = torch.long), labels = input_ids)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert model.vision_model.weight.grad is not None and model.vision_model.weight.grad.abs().sum() > 0


def test_ragged_pixel_values_list(remote):
    model = _model(remote).eval()
    images = [torch.randn(3, 8, 8), torch.randn(3, 8, 12)]  # 4 and 6 image tokens
    input_ids = _batch([4, 6])
    flags = torch.ones(2, 1, dtype = torch.long)
    with torch.no_grad():
        both = model(pixel_values = images, input_ids = input_ids, image_flags = flags).logits
        for i, image in enumerate(images):
            one = model(pixel_values = image.unsqueeze(0), input_ids = input_ids[i:i + 1],
                        image_flags = flags[:1]).logits
            torch.testing.assert_close(both[i:i + 1], one, rtol = 1e-5, atol = 1e-5)


def test_same_values_as_original_forward(remote):
    model = _model(remote).eval()
    original = getattr(type(model), "_unsloth_original_internvl_forward", None)
    assert original is not None, "InternVL-style forward was not repaired"
    kwargs = dict(pixel_values = torch.randn(2, 3, 8, 8), input_ids = _batch([4, 4]),
                  image_flags = torch.ones(2, 1, dtype = torch.long))
    with torch.no_grad():
        new = model(**kwargs, labels = kwargs["input_ids"])
        old = original(model, **kwargs, labels = kwargs["input_ids"])
    assert torch.equal(new.logits, old.logits)
    assert torch.equal(new.loss, old.loss)
    assert inspect.signature(type(model).forward) == inspect.signature(original)


def test_image_flags_drop_tiles(remote):
    model = _model(remote).eval()
    pixel_values = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        dropped = model(pixel_values = pixel_values, input_ids = _batch([4]),
                        image_flags = torch.tensor([[0], [1]])).logits
        kept = model(pixel_values = pixel_values[1:], input_ids = _batch([4]),
                     image_flags = torch.ones(1, 1, dtype = torch.long)).logits
    torch.testing.assert_close(dropped, kept)


def test_other_remote_classes_untouched(remote):
    from unsloth_zoo.temporary_patches.remote_code_vlm import repair_internvl_style_forward, repair_remote_modules
    assert "_unsloth_original_internvl_forward" not in remote.PlainRemote.__dict__
    forward = remote.TinyOmni.forward
    repair_remote_modules()
    assert repair_internvl_style_forward(remote.TinyOmni) is False
    assert remote.TinyOmni.forward is forward


def test_radio_summary_idxs_restored(remote):
    model = remote.RADIOModel(remote.TinyRadioConfig())
    with torch.no_grad():
        model.radio_model.summary_idxs.zero_()
    x = torch.arange(24.0).reshape(2, 4, 3)
    out = model(x)
    assert model.radio_model.summary_idxs.tolist() == [0, 1, 2]
    torch.testing.assert_close(out, x[:, [0, 1, 2]])

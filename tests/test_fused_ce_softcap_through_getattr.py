# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""PaliGemma on transformers 5.23 reads its softcap through getattr (PaliGemma 1's Gemma text config has none): the
fused lm_head rewrite must still match it, and keep the getattr so a config without the field does not raise."""
import types

from unsloth_zoo.compiler import apply_fused_lm_head, fixup_fused_lm_head

FORWARD = '''
def forward(self, input_ids=None, labels=None, logits_to_keep=0, **kwargs):
    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs[0]
    hidden_states = slice_logits_to_keep(hidden_states, logits_to_keep)
    logits = self.lm_head(hidden_states)
    final_logit_softcapping = getattr(self.config.get_text_config(), "final_logit_softcapping", None)
    if final_logit_softcapping is not None:
        logits = logits / final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * final_logit_softcapping

    loss = None
    if labels is not None:
        loss = self.loss_function(
            logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
        )

    return PaliGemmaCausalLMOutputWithPast(loss=loss, logits=logits)
'''

SOFTCAP = 'getattr(self.config.get_text_config(),"final_logit_softcapping",None)'


def test_the_getattr_softcap_is_fused():
    fused_source, fused = apply_fused_lm_head(fixup_fused_lm_head(FORWARD))
    assert fused, "the PaliGemma softcap form no longer reaches the fused lm_head + CE path"
    assert f"logits = logits / ({SOFTCAP})" in fused_source


def test_the_softcap_expression_is_safe_without_the_field():
    without = types.SimpleNamespace(config = types.SimpleNamespace(get_text_config = lambda: types.SimpleNamespace()))
    with_cap = types.SimpleNamespace(
        config = types.SimpleNamespace(get_text_config = lambda: types.SimpleNamespace(final_logit_softcapping = 30.0))
    )
    assert eval(SOFTCAP, {}, {"self": without}) is None
    assert eval(SOFTCAP, {}, {"self": with_cap}) == 30.0

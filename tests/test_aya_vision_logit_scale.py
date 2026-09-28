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

"""AyaVision drops its Cohere text logit_scale; the compiled forward must apply it (CPU, source only)."""

import inspect
import re

import pytest

aya = pytest.importorskip("transformers.models.aya_vision.modeling_aya_vision")
from unsloth_zoo import compiler  # noqa: E402

NAME = "AyaVisionForConditionalGeneration"


def _compiled_source(fix):
    source = compiler.fixup_fused_lm_head(inspect.getsource(aya.AyaVisionForConditionalGeneration.forward))
    if fix:
        source = compiler.fixup_dropped_logit_scale(source, NAME)
    out, _ = compiler.apply_fused_lm_head(source, NAME)
    return out


def test_premise_upstream_forward_has_no_logit_scale():
    assert "logit_scale" not in inspect.getsource(aya.AyaVisionForConditionalGeneration.forward)


def test_fused_loss_and_logits_carry_the_text_logit_scale():
    out = _compiled_source(fix = True)
    assert "unsloth_fused_ce_loss" in out
    scale = re.findall(r"logit_scale_multiply\s*=\s*\(([^)]*)\)", out)
    assert scale and all(s == "self.config.text_config.logit_scale" for s in scale)
    assert "logits = logits * (self.config.text_config.logit_scale)" in out


def test_other_models_and_already_scaled_sources_are_untouched():
    source = inspect.getsource(aya.AyaVisionForConditionalGeneration.forward)
    assert compiler.fixup_dropped_logit_scale(source, "LlavaForConditionalGeneration") == source
    scaled = source.replace("logits = self.lm_head(", "logit_scale = 1\n        logits = self.lm_head(")
    assert compiler.fixup_dropped_logit_scale(scaled, NAME) == scaled

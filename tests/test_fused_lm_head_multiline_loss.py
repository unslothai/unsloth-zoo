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

"""Multi-line `self.loss_function(...)` calls (Granite MoE family) must get fused CE (CPU, source only)."""

import importlib
import inspect

import pytest

from unsloth_zoo import compiler

PREFIX = """    def forward(self, hidden_states, labels=None, **kwargs):
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits = logits / self.config.logits_scaling

        loss = None
        if labels is not None:
"""

CALLS = {
    "single_line": "            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)\n",
    "multi_line_trailing_comma": (
        "            # Flatten the tokens\n"
        "            loss = self.loss_function(\n"
        "                logits,\n"
        "                labels,\n"
        "                vocab_size=self.config.vocab_size,\n"
        "                **kwargs,\n"
        "            )\n"
    ),
    "multi_line_after_upcast": (
        "            # Upcast to float if we need to compute the loss to avoid potential precision issues\n"
        "            logits = logits.float()\n"
        "            # Flatten the tokens\n"
        "            loss = self.loss_function(\n"
        "                logits,\n"
        "                labels,\n"
        "                vocab_size=self.config.vocab_size,\n"
        "                **kwargs,\n"
        "            )\n"
    ),
    "multi_line_no_trailing_comma": (
        "            loss = self.loss_function(\n"
        "                logits=logits,\n"
        "                labels=labels,\n"
        "                vocab_size=self.config.vocab_size,\n"
        "                **lm_kwargs\n"
        "            )\n"
    ),
}


@pytest.mark.parametrize("name", list(CALLS))
def test_loss_function_call_layouts_are_fused(name):
    out, _ = compiler.apply_fused_lm_head(PREFIX + CALLS[name] + "        return loss\n", name)
    assert "unsloth_fused_ce_loss" in out
    assert "logit_scale_divide   = (self.config.logits_scaling)" in out
    kwargs = "lm_kwargs" if "lm_kwargs" in CALLS[name] else "kwargs"
    assert f"if ({kwargs}) != () and type({kwargs}) is dict:" in out


@pytest.mark.parametrize(
    "module, cls",
    [
        ("granitemoehybrid", "GraniteMoeHybridForCausalLM"),
        ("granitemoe", "GraniteMoeForCausalLM"),
        ("granitemoeshared", "GraniteMoeSharedForCausalLM"),
    ],
)
def test_granite_moe_forwards_are_fused(module, cls):
    try:
        mod = importlib.import_module(f"transformers.models.{module}.modeling_{module}")
    except ImportError:
        pytest.skip(f"{module} not in this transformers")
    source = compiler.fixup_fused_lm_head(inspect.getsource(getattr(mod, cls).forward))
    out, _ = compiler.apply_fused_lm_head(source, cls)
    assert "unsloth_fused_ce_loss" in out
    assert "self.router_aux_loss_coef * aux_loss" in out

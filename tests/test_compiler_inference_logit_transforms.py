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
"""The compiled inference branch must apply the logit transforms.

`compiler.py` emits several forward templates. Each has an `elif labels is
None:` arm that returns logits straight to the caller, and training arms that
apply the captured logit scale and softcap. When the inference arm skips those
transforms the returned distribution is wrong for any model configuring them:
greedy decoding is unaffected because tanh is monotonic, but sampling,
logprobs and logit-based scoring are not.

This is a source guard rather than a runtime test, because reaching the
generated code needs a real checkpoint and a GPU.
"""
import re

import unsloth_zoo.compiler as compiler_module


def _template_source() -> str:
    import inspect
    return inspect.getsource(compiler_module)


def test_every_inference_branch_applies_the_softcap():
    src = _template_source()
    marker = "elif labels is None:"
    assert src.count(marker) >= 1, "inference branch template disappeared"

    for match in re.finditer(re.escape(marker), src):
        # Look at the arm only, i.e. up to the next branch at the same level.
        tail = src[match.end(): match.end() + 2000]
        head_call = tail.find("logits = self.lm_head(")
        assert head_call != -1, "inference arm no longer calls the lm_head"
        arm = tail[head_call:]
        for stop in ("\nelif ", "\nelse:"):
            cut = arm.find(stop)
            if cut != -1:
                arm = arm[:cut]
        assert "torch.tanh(logits)" in arm, (
            "an inference arm returns logits without applying the softcap; "
            "the training arms below it do, so the two disagree"
        )


def test_training_and_inference_use_the_same_transform_order():
    src = _template_source()
    # scale multiply, then scale divide, then softcap. Any arm that applies the
    # transforms must do so in this order, otherwise the two paths differ.
    order = re.compile(
        r"logits = logits \* \(\\\\2\).*?logits = logits / \(\\\\3\).*?torch\.tanh\(logits\)",
        re.S,
    )
    assert order.search(src), "logit transform order changed in the templates"

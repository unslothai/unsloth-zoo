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

"""Training forwards must not inherit the eager_on_recompile stance that inference sets."""

import logging
import os
import textwrap

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch.compiler, "set_stance"):
    pytest.skip("torch without set_stance", allow_module_level = True)

from unsloth_zoo import compiler
from unsloth_zoo.temporary_patches.utils import UNSLOTH_DECODE_COMPILE


def _stance():
    import torch._dynamo.eval_frame as eval_frame
    return eval_frame._stance.stance


def _generated_module(compiled):
    """The stance globals of a generated module plus a forward built from the same snippets."""
    header = compiler._license_header
    start = header.index("global INFERENCE_RUNS")
    end = header.index("from unsloth_zoo import DEVICE_TYPE_TORCH")
    namespace = {
        "torch": torch,
        "UNSLOTH_DECODE_COMPILE": UNSLOTH_DECODE_COMPILE,
        "UNSLOTH_ENABLE_LOGGING": False,
        "logger_compiler": logging.getLogger(__name__),
        "compiled": compiled,
    }
    exec(header[start:end], namespace)
    forward = (
        "def forward(self, x, labels = None):\n"
        + textwrap.indent(compiler.__DYNAMO__TRAINING_STANCE__, "    ")
        + "\n    if labels is None:\n"
        + textwrap.indent(compiler.__DYNAMO__RECOMPILING__, "    ")
        + "    return compiled(x)\n"
    )
    exec(forward, namespace)
    return namespace


@pytest.fixture(autouse = True)
def _reset_stance():
    torch._dynamo.reset()
    torch.compiler.set_stance("default")
    yield
    torch.compiler.set_stance("default")
    torch._dynamo.reset()


def _counting_compile():
    frames = []
    def backend(gm, example_inputs):
        frames.append(gm)
        return gm.forward
    return frames, torch.compile(lambda x: x.sin() * 2 + 1, backend = backend)


def test_every_lm_head_template_resets_before_branching():
    for template in (
        compiler.cross_entropy_replacement_1,
        compiler.cross_entropy_replacement_2,
        compiler.cross_entropy_replacement_3,
    ):
        assert "unsloth_training_stance()" in template
        assert template.index("unsloth_training_stance()") < template.index("if RETURN_HIDDEN_STATES:")


def test_training_after_inference_compiles_again():
    frames, compiled = _counting_compile()
    ns = _generated_module(compiled)
    module = torch.nn.Linear(1, 1)

    module.eval()
    with torch.no_grad():
        for _ in range(3):
            ns["forward"](module, torch.randn(8))
    assert _stance() == "eager_on_recompile"
    assert len(frames) == 1

    # model.train() alone, no for_training: grad mode changes, so the frame must recompile.
    module.train()
    x = torch.randn(8, requires_grad = True)
    ns["forward"](module, x, labels = x).sum().backward()
    assert _stance() == "default"
    assert len(frames) == 2, "the training step ran the eager body"

    # Inference afterwards still warms up once, then freezes recompiles again.
    module.eval()
    with torch.no_grad():
        for _ in range(3):
            ns["forward"](module, torch.randn(8))
    assert _stance() == "eager_on_recompile"


def test_training_with_labels_none_does_not_flip_the_stance():
    # GRPO-style logits forwards: train mode, grad on, no labels.
    frames, compiled = _counting_compile()
    ns = _generated_module(compiled)
    module = torch.nn.Linear(1, 1).train()
    for _ in range(4):
        ns["forward"](module, torch.randn(8, requires_grad = True))
    assert _stance() == "default"


def test_user_stance_is_left_alone():
    frames, compiled = _counting_compile()
    ns = _generated_module(compiled)
    module = torch.nn.Linear(1, 1).train()
    torch.compiler.set_stance("eager_on_recompile")
    x = torch.randn(8, requires_grad = True)
    ns["forward"](module, x, labels = x)
    assert _stance() == "eager_on_recompile"


def test_no_graph_break_when_the_forward_is_traced():
    frames, compiled = _counting_compile()
    ns = _generated_module(compiled)
    module = torch.nn.Linear(1, 1).train()
    traced = torch.compile(ns["forward"], fullgraph = True, backend = "eager")
    x = torch.randn(8, requires_grad = True)
    traced(module, x, labels = x)
    with torch.no_grad():
        traced(module.eval(), torch.randn(8))

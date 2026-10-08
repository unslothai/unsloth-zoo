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
import sys
import textwrap
import types

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch.compiler, "set_stance"):
    pytest.skip("torch without set_stance", allow_module_level = True)

from unsloth_zoo import compiler
from unsloth_zoo.temporary_patches.utils import (
    UNSLOTH_DECODE_COMPILE,
    UNSLOTH_EAGER_STANCE_OWNED,
    unsloth_decode_compile,
)


def _stance():
    import torch._dynamo.eval_frame as eval_frame
    return eval_frame._stance.stance


# A causal LM forward in the shape transformers writes them: decoder body, then the lm_head and
# loss. Production runs it through `apply_fused_lm_head`; so does the harness, so the stance
# hooks land exactly where generated modules have them.
CAUSAL_LM_FORWARD = """    def forward(self, input_ids, labels=None, logits_to_keep=0, **kwargs):
        \"\"\"Mirrors a transformers *ForCausalLM.forward.\"\"\"
        hidden_states = self.model(input_ids)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return loss, logits
"""


def ForCausalLMLoss(logits, labels, vocab_size = None, **kwargs):
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(), labels[:, 1:].reshape(-1), ignore_index = -100,
    )


def _fused_ce(hidden_states, lm_head_weight, labels, **kwargs):
    return ForCausalLMLoss(hidden_states @ lm_head_weight.t(), labels)


class _CausalLM(torch.nn.Module):
    def __init__(self, body):
        super().__init__()
        self.model = body
        self.lm_head = torch.nn.Linear(8, 16, bias = False)
        self.loss_function = ForCausalLMLoss
        self.config = types.SimpleNamespace(vocab_size = 16)


def _generated_forward_source():
    source, fused = compiler.apply_fused_lm_head(CAUSAL_LM_FORWARD, "TestForCausalLM")
    assert fused
    return textwrap.dedent(source)


def _generated_module(compiled, name = "unsloth_compiled_module_test"):
    """A module object with the stance globals of a generated module and the forward
    `apply_fused_lm_head` produces. Separate calls give separate modules, as two model types
    would. `compiled` is the decoder body; the returned `forward(module, x, labels)` follows
    `module`'s train / eval mode."""
    header = compiler._license_header
    start = header.index("global INFERENCE_RUNS")
    end = header.index("from unsloth_zoo import DEVICE_TYPE_TORCH")
    module = types.ModuleType(name)
    sys.modules[name] = module
    namespace = module.__dict__
    namespace.update({
        "torch": torch,
        "os": os,
        "EMPTY_LOGITS": torch.empty(0),
        "UNSLOTH_ENABLE_CCE": False,
        "HAS_CUT_CROSS_ENTROPY": False,
        "UNSLOTH_COMPILE_DISABLE": True,
        "unsloth_fused_ce_loss": _fused_ce,
        "UNSLOTH_DECODE_COMPILE": UNSLOTH_DECODE_COMPILE,
        "UNSLOTH_ENABLE_LOGGING": False,
        "logger_compiler": logging.getLogger(__name__),
    })
    exec(header[start:end], namespace)
    exec(_generated_forward_source(), namespace)
    causal_lm = _CausalLM(lambda ids: compiled(ids).view(1, 2, 8))

    generated_forward = namespace["forward"]
    assert x_shape_guard(generated_forward)

    def forward(owner, x, labels = None):
        # Calls the generated forward captured above, never the "forward" slot this replaces.
        assert x.numel() == 8, "harness input must stay (8,)"
        causal_lm.train(owner.training)
        ids = x.repeat(2)
        if labels is None:
            return generated_forward(causal_lm, ids)[1]
        return generated_forward(causal_lm, ids, labels = torch.zeros(1, 2, dtype = torch.long))[0]

    namespace["generated_forward"] = generated_forward
    namespace["forward"] = forward
    return namespace


def x_shape_guard(generated_forward):
    # The wrapper below must not be its own target.
    return generated_forward.__name__ == "forward" and "owner" not in generated_forward.__code__.co_varnames


def _snippet_module(compiled):
    """Only the stance hooks around a compiled call, small enough to trace with fullgraph."""
    header = compiler._license_header
    start = header.index("global INFERENCE_RUNS")
    end = header.index("from unsloth_zoo import DEVICE_TYPE_TORCH")
    module = types.ModuleType("unsloth_compiled_module_snippet")
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    namespace.update({
        "torch": torch,
        "UNSLOTH_DECODE_COMPILE": UNSLOTH_DECODE_COMPILE,
        "UNSLOTH_ENABLE_LOGGING": False,
        "logger_compiler": logging.getLogger(__name__),
        "compiled": compiled,
    })
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
    UNSLOTH_EAGER_STANCE_OWNED.clear()
    yield
    UNSLOTH_EAGER_STANCE_OWNED.clear()
    torch.compiler.set_stance("default")
    torch._dynamo.reset()


def _counting_compile():
    frames = []
    def backend(gm, example_inputs):
        frames.append(gm)
        return gm.forward
    return frames, torch.compile(lambda x: x.sin() * 2 + 1, backend = backend)


def test_reset_runs_before_the_decoder_body():
    source = _generated_forward_source()
    assert source.count("unsloth_training_stance()") == 1
    assert source.index("unsloth_training_stance()") < source.index("self.model(input_ids)")
    assert compiler._TRAINING_STANCE_MARKER not in source
    for template in (
        compiler.cross_entropy_replacement_1,
        compiler.cross_entropy_replacement_2,
        compiler.cross_entropy_replacement_3,
    ):
        assert "unsloth_training_stance()" not in template


def test_reset_falls_back_to_the_logits_site():
    # A forward the parser cannot place it in still gets the reset where the marker was.
    source = "def forward(self, x): (\n    " + compiler._TRAINING_STANCE_MARKER + "\n"
    placed = compiler._place_training_stance(source)
    assert "unsloth_training_stance()" in placed and compiler._TRAINING_STANCE_MARKER not in placed


def test_first_training_forward_after_inference_compiles_the_body():
    frames, compiled = _counting_compile()
    ns = _generated_module(compiled)
    module = torch.nn.Linear(1, 1)
    _infer(ns, module)
    assert _stance() == "eager_on_recompile"
    assert len(frames) == 1
    _train(ns, module)
    assert len(frames) == 2, "the first training step ran the decoder body eager"
    assert _stance() == "default"


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
    ns = _snippet_module(compiled)
    module = torch.nn.Linear(1, 1).train()
    traced = torch.compile(ns["forward"], fullgraph = True, backend = "eager")
    x = torch.randn(8, requires_grad = True)
    traced(module, x, labels = x)
    with torch.no_grad():
        traced(module.eval(), torch.randn(8))


def _infer(ns, module, calls = 3):
    module.eval()
    with torch.no_grad():
        for _ in range(calls):
            ns["forward"](module, torch.randn(8))


def _train(ns, module):
    module.train()
    x = torch.randn(8, requires_grad = True)
    ns["forward"](module, x, labels = x).sum().backward()


def test_inference_in_one_module_training_in_another():
    frames_a, compiled_a = _counting_compile()
    frames_b, compiled_b = _counting_compile()
    ns_a = _generated_module(compiled_a, "unsloth_compiled_module_gemma3")
    ns_b = _generated_module(compiled_b, "unsloth_compiled_module_qwen3_5")
    assert ns_a is not ns_b
    _infer(ns_a, torch.nn.Linear(1, 1))
    assert _stance() == "eager_on_recompile"
    _train(ns_b, torch.nn.Linear(1, 1))
    assert _stance() == "default"
    assert len(frames_b) == 1, "model B's training step ran the eager body"


def test_user_stance_survives_unsloth_inference_then_training():
    frames, compiled = _counting_compile()
    ns = _generated_module(compiled)
    module = torch.nn.Linear(1, 1)
    torch.compiler.set_stance("eager_on_recompile")
    _infer(ns, module)
    assert _stance() == "eager_on_recompile"
    _train(ns, module)
    assert _stance() == "eager_on_recompile"


def test_user_stance_set_after_unsloth_claimed_it_is_kept():
    frames, compiled = _counting_compile()
    ns = _generated_module(compiled)
    module = torch.nn.Linear(1, 1)
    _infer(ns, module)
    assert _stance() == "eager_on_recompile"
    torch.compiler.set_stance("eager_on_recompile")  # the user's own choice now
    _train(ns, module)
    assert _stance() == "eager_on_recompile"


def test_user_force_eager_then_eager_on_recompile_is_kept():
    frames, compiled = _counting_compile()
    ns = _generated_module(compiled)
    module = torch.nn.Linear(1, 1)
    _infer(ns, module)
    torch.compiler.set_stance("force_eager")
    torch.compiler.set_stance("eager_on_recompile")
    _train(ns, module)
    assert _stance() == "eager_on_recompile"


def test_unsloth_stance_still_reset_after_a_scoped_user_stance():
    # A `with set_stance(...)` block restores the very object Unsloth installed.
    frames, compiled = _counting_compile()
    ns = _generated_module(compiled)
    module = torch.nn.Linear(1, 1)
    _infer(ns, module)
    with torch.compiler.set_stance("force_eager"):
        pass
    _train(ns, module)
    assert _stance() == "default"


def test_unsloth_stance_still_reset_after_a_compiled_decode_scope():
    # The decode scope switches to default and back, installing a new stance object.
    frames, compiled = _counting_compile()
    ns = _generated_module(compiled)
    module = torch.nn.Linear(1, 1)
    _infer(ns, module)
    with unsloth_decode_compile():
        assert _stance() == "default"
    assert _stance() == "eager_on_recompile"
    _train(ns, module)
    assert _stance() == "default"


def test_inference_in_two_modules_then_training_compiles():
    # Model B's inference finds the stance already eager and must not take it over unclaimed.
    frames_a, compiled_a = _counting_compile()
    frames_b, compiled_b = _counting_compile()
    ns_a = _generated_module(compiled_a, "unsloth_compiled_module_gemma3")
    ns_b = _generated_module(compiled_b, "unsloth_compiled_module_qwen3_5")
    _infer(ns_a, torch.nn.Linear(1, 1))
    _infer(ns_b, torch.nn.Linear(1, 1))
    assert _stance() == "eager_on_recompile"
    _train(ns_b, torch.nn.Linear(1, 1))
    assert _stance() == "default"
    assert len(frames_b) == 1, "model B's training step ran the eager body"


def test_training_inside_a_scoped_user_stance_keeps_ownership():
    # The scope's exit restores Unsloth's object, so a later training forward still resets it.
    frames, compiled = _counting_compile()
    ns = _generated_module(compiled)
    module = torch.nn.Linear(1, 1)
    _infer(ns, module)
    with torch.compiler.set_stance("force_eager"):
        _train(ns, module)
    assert _stance() == "eager_on_recompile"
    _train(ns, module)
    assert _stance() == "default"


def test_scoped_inference_in_a_second_module_keeps_the_first_claim():
    # A owns the eager stance; B's claiming inference runs inside a temporary default scope,
    # whose exit restores A's object. Training must still recognise and reset it.
    frames_a, compiled_a = _counting_compile()
    frames_b, compiled_b = _counting_compile()
    ns_a = _generated_module(compiled_a, "unsloth_compiled_module_gemma3")
    ns_b = _generated_module(compiled_b, "unsloth_compiled_module_qwen3_5")
    module_b = torch.nn.Linear(1, 1)
    _infer(ns_a, torch.nn.Linear(1, 1))
    _infer(ns_b, module_b, calls = 1)
    with torch.compiler.set_stance("default"):
        _infer(ns_b, module_b, calls = 1)
    assert _stance() == "eager_on_recompile"
    _train(ns_b, module_b)
    assert _stance() == "default"


def test_training_inside_a_scope_keeps_the_outer_owned_stance():
    # Outer Unsloth stance saved by a default scope; inference inside claims a second one and a
    # training forward resets it. The scope's exit restores the outer one, which is still ours.
    frames, compiled = _counting_compile()
    ns = _generated_module(compiled)
    module = torch.nn.Linear(1, 1)
    _infer(ns, module)
    with torch.compiler.set_stance("default"):
        _infer(ns, module)
        assert _stance() == "eager_on_recompile"
        _train(ns, module)
        assert _stance() == "default"
    assert _stance() == "eager_on_recompile"
    _train(ns, module)
    assert _stance() == "default"

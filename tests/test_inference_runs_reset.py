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

"""Every generate() starts its inference count from 0, so decode step 1 still compiles.

The generated logits site switches Dynamo to eager_on_recompile once one inference forward has run.
A count left at 1 by a max_new_tokens=1 generate made the next generate switch right after its
prefill, so any decode graph not compiled yet ran eagerly for the whole generate.
"""

import logging
import os
import sys
import types

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch.compiler, "set_stance"):
    pytest.skip("torch < 2.6 has no torch.compiler.set_stance", allow_module_level = True)

from unsloth_zoo import compiler
from unsloth_zoo.temporary_patches import utils as stance_utils
from unsloth_zoo.temporary_patches.utils import UNSLOTH_DECODE_COMPILE, UNSLOTH_EAGER_STANCE_OWNED


def _current_stance():
    import torch._dynamo.eval_frame as eval_frame

    return eval_frame._stance.stance


def _generated_module(body):
    """The header's stance globals plus a forward whose logits site carries the generated block."""
    header = compiler._license_header
    start = header.index("global INFERENCE_RUNS")
    end = header.index("from unsloth_zoo import DEVICE_TYPE_TORCH")
    module = types.ModuleType("unsloth_compiled_module_inference_runs_test")
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    namespace.update({
        "torch": torch,
        "UNSLOTH_DECODE_COMPILE": UNSLOTH_DECODE_COMPILE,
        "UNSLOTH_ENABLE_LOGGING": False,
        "logger_compiler": logging.getLogger(__name__),
        "body": body,
        "stances": [],
    })
    exec(header[start:end], namespace)
    exec(
        "def forward(x):\n"
        "    stances.append(torch_dynamo_eval_frame._stance.stance)\n"
        "    out = body(x)\n" + compiler.__DYNAMO__RECOMPILING__ + "    return out\n",
        namespace,
    )
    return namespace


def _generate(namespace, prompt_len, max_new_tokens):
    """What FastBaseModel.for_inference + generate() do: default stance, reset (where zoo has it), forwards."""
    torch.compiler.set_stance("default")
    reset = getattr(stance_utils, "unsloth_reset_inference_runs", None)
    if reset is not None:
        reset()
    namespace["stances"].clear()
    namespace["forward"](torch.ones(prompt_len))
    for _ in range(max_new_tokens - 1):
        namespace["forward"](torch.ones(1))
    return list(namespace["stances"])


@pytest.fixture(autouse = True)
def _clean_stance():
    torch._dynamo.reset()
    torch.compiler.set_stance("default")
    UNSLOTH_EAGER_STANCE_OWNED.clear()
    yield
    UNSLOTH_EAGER_STANCE_OWNED.clear()
    torch.compiler.set_stance("default")
    torch._dynamo.reset()
    sys.modules.pop("unsloth_compiled_module_inference_runs_test", None)
    sys.modules.pop("unsloth_compiled_module_legacy_test", None)


def _counting_body():
    frames = []

    def backend(gm, example_inputs):
        frames.append(tuple(example_inputs[0].shape))
        return gm.forward

    return torch.compile(lambda x: x * 2, backend = backend, dynamic = False), frames


def test_decode_step_one_compiles_after_a_one_token_generate():
    body, frames = _counting_body()
    namespace = _generated_module(body)
    assert _generate(namespace, 8, 1) == ["default"]
    stances = _generate(namespace, 8, 4)
    # Prefill and decode step 1 run under the default stance; eager_on_recompile only from step 2.
    assert stances == ["default", "default", "eager_on_recompile", "eager_on_recompile"]
    assert frames == [(8,), (1,)], "the decode graph never compiled"


def test_alternating_one_token_generates_never_skip_the_decode_compile():
    body, frames = _counting_body()
    namespace = _generated_module(body)
    for _ in range(3):
        _generate(namespace, 8, 1)
    assert _generate(namespace, 8, 3)[1] == "default"
    assert frames == [(8,), (1,)]


def test_repeated_generates_keep_eager_from_decode_step_two():
    body, frames = _counting_body()
    namespace = _generated_module(body)
    for _ in range(3):
        assert _generate(namespace, 8, 4) == ["default", "default", "eager_on_recompile", "eager_on_recompile"]
    assert frames == [(8,), (1,)], "same shapes must not recompile"


def test_count_is_shared_and_reset_by_training_stance():
    namespace = _generated_module(lambda x: x)
    namespace["forward"](torch.ones(2))
    assert stance_utils.UNSLOTH_INFERENCE_RUNS[0] == 1
    namespace["unsloth_training_stance"]()
    assert stance_utils.UNSLOTH_INFERENCE_RUNS[0] == 0
    assert _current_stance() == "default"


def test_reset_also_clears_modules_from_an_older_zoo():
    legacy = types.ModuleType("unsloth_compiled_module_legacy_test")
    legacy.INFERENCE_RUNS = 1
    sys.modules[legacy.__name__] = legacy
    stance_utils.UNSLOTH_INFERENCE_RUNS[0] = 5
    stance_utils.unsloth_reset_inference_runs()
    assert stance_utils.UNSLOTH_INFERENCE_RUNS[0] == 0
    assert legacy.INFERENCE_RUNS == 0

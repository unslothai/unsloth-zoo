# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""patch_torch_compile turns off Inductor's pointwise launch-config autotuning (cold start: one config per pointwise
kernel instead of 2 or 6 compiled and timed at first launch). max_autotune still tunes them."""

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("torch")

_CHILD = textwrap.dedent("""
    import sys
    sys.path.insert(0, sys.argv[1])
    import torch._inductor.config as config
    from unsloth_zoo.patching_utils import patch_torch_compile
    before = config.triton.autotune_pointwise
    patch_torch_compile(debug = False, O3 = sys.argv[2] == "1", ignore_errors = True)
    print("RESULT", before, config.triton.autotune_pointwise, config.max_autotune_pointwise)
""")


@pytest.mark.parametrize("O3", ["0", "1"])
def test_patch_torch_compile_turns_pointwise_autotune_off(O3):
    env = dict(os.environ, UNSLOTH_ALLOW_CPU = "1", CUDA_VISIBLE_DEVICES = "")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-c", _CHILD, root, O3], env = env, capture_output = True, text = True,
                         timeout = 600)
    line = [x for x in out.stdout.splitlines() if x.startswith("RESULT")]
    assert out.returncode == 0 and line, out.stderr[-4000:]
    _, before, after, max_pointwise = line[-1].split()
    assert before == "True" and after == "False"
    # O3 keeps tuning pointwise kernels through max_autotune_pointwise, which outranks autotune_pointwise.
    assert max_pointwise == ("True" if O3 == "1" else "False")

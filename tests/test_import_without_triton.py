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

"""AMD Windows wheels ship no Triton: the modules Unsloth imports must load without it and the fused losses take
their torch path."""

import os
import subprocess
import sys
import textwrap

# Triton made unfindable for the child, as on a host where the package is not installed.
_NO_TRITON = textwrap.dedent('''
    import importlib.machinery, sys
    _find = importlib.machinery.PathFinder.find_spec.__func__

    def _find_spec(cls, name, path = None, target = None):
        if name == "triton" or name.startswith("triton."):
            return None
        return _find(cls, name, path, target)

    importlib.machinery.PathFinder.find_spec = classmethod(_find_spec)
    for _k in [k for k in sys.modules if k == "triton" or k.startswith("triton.")]:
        del sys.modules[_k]
''')

_CHECK = textwrap.dedent('''
    import importlib.util
    import torch
    import unsloth_zoo.compiler as compiler
    import unsloth_zoo.loss_utils as loss_utils
    import unsloth_zoo.patching_utils  # noqa: F401
    import unsloth_zoo.fused_losses.fused_linear as fl
    assert importlib.util.find_spec("triton") is None
    assert compiler.triton is None and compiler.OLD_TRITON_VERSION
    assert loss_utils.triton_version is None and not loss_utils.HAS_CUT_CROSS_ENTROPY
    assert not fl._HAS_TRITON
    torch.manual_seed(0)
    h = torch.randn(48, 16, dtype = torch.float64, requires_grad = True)
    w = torch.randn(101, 16, dtype = torch.float64)
    t = torch.randint(0, 101, (48,))
    logps = fl.fused_linear_logprobs(h, w, t)
    logps.sum().backward()
    ref = torch.log_softmax(h.detach() @ w.T, -1).gather(1, t[:, None])[:, 0]
    assert torch.allclose(logps.double(), ref, atol = 1e-6), (logps - ref).abs().max()
    assert torch.isfinite(h.grad).all() and fl.CALLS["torch"] > 0 and fl.CALLS["triton"] == 0
    print("NO_TRITON_OK")
''')


def test_zoo_imports_and_fused_losses_run_without_triton():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_IS_PRESENT = "1")
    out = subprocess.run([sys.executable, "-c", _NO_TRITON + _CHECK], env = env, capture_output = True,
                         text = True, timeout = 600)
    assert out.returncode == 0 and "NO_TRITON_OK" in out.stdout, (out.stdout[-2000:], out.stderr[-4000:])

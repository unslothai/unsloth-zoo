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

"""The rand4x-off cache key tag must reach a compile on another thread (a reentrant checkpoint's
recompute compiles on autograd's device thread; torch >= 2.12 keeps config per thread)."""

import json
import os
import pathlib
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
from torch._inductor.codegen.triton import TritonOverrides  # noqa: E402
import torch._inductor.config as inductor_config  # noqa: E402

AFFECTED = "_can_use_4x_random" in TritonOverrides.__dict__ and not hasattr(inductor_config, "use_rand4x")
ROOT = str(pathlib.Path(__file__).resolve().parents[1])

CODE = """
import json, sys, threading
sys.path.insert(0, %r)
import torch, unsloth_zoo, unsloth_zoo.temporary_patches
assert unsloth_zoo.__file__.startswith(%r), unsloth_zoo.__file__
out = {}
t = threading.Thread(target = lambda: out.setdefault("tag", torch.compiler.config.cache_key_tag))
t.start(); t.join()
print(json.dumps({"main": torch.compiler.config.cache_key_tag, "thread": out["tag"]}))
"""


@pytest.mark.skipif(not AFFECTED, reason = "torch without the unfixed rand4x path")
def test_rand4x_cache_tag_reaches_other_threads():
    env = dict(os.environ, UNSLOTH_NO_LOCAL_GPU = "1", CUDA_VISIBLE_DEVICES = "")
    env.pop("TORCHINDUCTOR_USE_RAND4X", None)
    r = subprocess.run(
        [sys.executable, "-B", "-c", CODE % (ROOT, ROOT)], capture_output = True, text = True, env = env, timeout = 900,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    main_tags = [t for t in out["main"].split(",") if t]
    assert "unsloth/inductor-rand4x-off" in main_tags, out
    assert set(main_tags) <= {t for t in out["thread"].split(",") if t}, out

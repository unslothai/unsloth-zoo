# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Regression: unsloth_zoo.tokenizer_utils must import and run patch_tokenizer
without torch, so get_chat_template works on a torch-free MLX install. The module
holds torch-only embedding-fix helpers, but patch_tokenizer itself is torch-free;
a top-level `import torch` or `@torch.inference_mode` decorator would break the
MLX tokenizer patch path at import time.
"""

import subprocess
import sys
import textwrap


def test_tokenizer_utils_imports_and_patches_without_torch():
    # Runs in a subprocess: initialise the package normally, then block torch at
    # load and re-exec only tokenizer_utils to prove it is import-safe torch-free.
    script = textwrap.dedent(
        """
        import importlib, importlib.abc, importlib.machinery, sys
        import unsloth_zoo.tokenizer_utils  # init package (torch may be present)

        class _FailLoader(importlib.abc.Loader):
            def create_module(self, spec): return None
            def exec_module(self, module): raise ImportError("torch blocked")

        class _Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "torch" or name.startswith("torch."):
                    return importlib.machinery.ModuleSpec(name, _FailLoader())
                return None

        sys.meta_path.insert(0, _Block())
        for m in [k for k in list(sys.modules) if k == "torch" or k.startswith("torch.")]:
            del sys.modules[m]
        del sys.modules["unsloth_zoo.tokenizer_utils"]

        import unsloth_zoo.tokenizer_utils as tu
        assert tu.torch is None, tu.torch
        assert tu.patch_tokenizer(model=None, tokenizer=None) == (None, None)

        class FakeTok:
            pad_token = "<pad>"; pad_token_id = 0
            eos_token = "</s>"; eos_token_id = 1
            def __len__(self): return 32

        _, tok = tu.patch_tokenizer(model=None, tokenizer=FakeTok())
        assert tok is not None
        print("TORCH_FREE_OK")
        """
    )
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert "TORCH_FREE_OK" in r.stdout, f"stdout={r.stdout}\n---\nstderr={r.stderr}"

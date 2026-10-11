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

"""The compiled gemma4 module must load with the patched mask builders (unslothai/unsloth#13245)."""

import json
import os
import subprocess
import sys

import pytest

from unsloth_zoo import compiler

PROBE_NAME = "UnslothCompileTransformersImportProbe"
GEMMA4_LOCATION = "transformers.models.gemma4.modeling_gemma4"


@pytest.mark.parametrize(
    ("source", "expect_import"),
    [
        pytest.param(
            "def _probe(config: transformers.cache_utils.Cache):\n    return config\n",
            True,
            id = "references-transformers",
        ),
        pytest.param("def _probe(config):\n    return config\n", False, id = "no-transformers-ref"),
    ],
)
def test_emitted_module_imports_transformers_iff_referenced(
    tmp_path, monkeypatch, source, expect_import
):
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_LOCATION", str(tmp_path))
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_USE_TEMP", False)
    compiler.create_new_function(PROBE_NAME, source, GEMMA4_LOCATION, [], overwrite = True)
    emitted = (tmp_path / f"{PROBE_NAME}.py").read_text()
    assert ("import transformers\n" in emitted.split("def _probe")[0]) is expect_import


_GEMMA4_CHILD = r'''
import os, io, json, contextlib
os.environ["UNSLOTH_COMPILE_LOCATION"] = "unsloth_compiled_cache"
import inspect
import transformers.masking_utils as masking_utils
from unsloth_zoo.temporary_patches.misc import patch_transformers_masks
from unsloth_zoo.compiler import unsloth_compile_transformers

patch_transformers_masks()
buf = io.StringIO()
error = None
try:
    with contextlib.redirect_stdout(buf):
        unsloth_compile_transformers(
            model_type = "gemma4", fast_lora_forwards = False, fullgraph = True,
            import_from_cache = False, disable = False, supports_sdpa = [None],
        )
except Exception as e:
    error = repr(e)
path = os.path.join("unsloth_compiled_cache", "unsloth_compiled_module_gemma4.py")
generated = open(path).read() if os.path.exists(path) else ""
print("@@@" + json.dumps({
    "error": error,
    "emits_wrapper": "def create_causal_mask(" in generated,
    "signature_kept": inspect.signature(masking_utils.create_causal_mask)
        == inspect.signature(masking_utils._unsloth_original_create_causal_mask),
}))
'''


def test_gemma4_compiles_with_patched_mask_builders(tmp_path):
    pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    env = dict(os.environ)
    env["UNSLOTH_ALLOW_CPU"] = "1"
    env["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    result = subprocess.run(
        [sys.executable, "-c", _GEMMA4_CHILD],
        cwd = str(tmp_path), env = env, capture_output = True, text = True, timeout = 1800,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    line = [row for row in result.stdout.splitlines() if row.startswith("@@@")]
    assert line, result.stdout[-4000:] + result.stderr[-4000:]
    payload = json.loads(line[-1][3:])
    assert payload == {"error": None, "emits_wrapper": False, "signature_kept": True}, payload

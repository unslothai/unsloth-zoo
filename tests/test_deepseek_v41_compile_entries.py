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

"""DeepSeek-V4.1 port: host helpers and QAT stay eager; rope compiles only via the V4 `split` rewrite."""

import importlib.util
import json
import os
import subprocess
import sys

import pytest
import torch

from unsloth_zoo.compiler import (
    DISABLE_COMPILE_MODEL_FUNCTIONS,
    DISABLE_COMPILE_MODULES,
    MODEL_FUNCTION_SOURCE_REWRITES,
    model_function_source_rewrites,
)

_DISABLED = {
    "apply_rotary_pos_emb",
    "build_compressed_token_map",
    "compute_hash_multipliers",
    "_is_prime",
    "_find_next_prime",
    "_mesh_process_group",
    "_tp_grad_group",
    "_validate_attention_tp_divisibility",
    "_quantize_qat",
}


def test_deepseek_v41_entries_registered():
    assert set(DISABLE_COMPILE_MODEL_FUNCTIONS["deepseek_v41"]) == _DISABLED
    assert "DeepseekV41HyperConnection" in DISABLE_COMPILE_MODULES
    assert (
        MODEL_FUNCTION_SOURCE_REWRITES["deepseek_v41"]["apply_rotary_pos_emb"]
        == MODEL_FUNCTION_SOURCE_REWRITES["deepseek_v4"]["apply_rotary_pos_emb"]
    )


def test_rope_rewrite_is_value_identical():
    old, new = MODEL_FUNCTION_SOURCE_REWRITES["deepseek_v41"]["apply_rotary_pos_emb"]
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, 512).transpose(1, 2)
    scope_old, scope_new = {"x": x, "rope_dim": 64}, {"x": x, "rope_dim": 64}
    exec(old, scope_old)
    exec(new, scope_new)
    for name in ("nope", "rope"):
        assert torch.equal(scope_old[name], scope_new[name])


modeling = None
if importlib.util.find_spec("transformers.models.deepseek_v41") is not None:
    import transformers.models.deepseek_v41.modeling_deepseek_v41 as modeling

needs_port = pytest.mark.skipif(modeling is None, reason = "needs the community deepseek_v41 port")


@needs_port
def test_port_defines_every_entry_and_the_rope_rewrite_matches():
    missing = sorted(name for name in _DISABLED if not callable(getattr(modeling, name, None)))
    assert not missing, missing
    assert hasattr(modeling, "DeepseekV41HyperConnection")
    assert "apply_rotary_pos_emb" in model_function_source_rewrites(modeling, "deepseek_v41")


_CHILD = r'''
import os, sys, json, io, contextlib
os.environ["UNSLOTH_COMPILE_LOCATION"] = os.path.abspath("unsloth_compiled_cache")
from unsloth_zoo.compiler import unsloth_compile_transformers
with contextlib.redirect_stdout(io.StringIO()):
    unsloth_compile_transformers(
        model_type = "deepseek_v41", fast_lora_forwards = False, fullgraph = True,
        import_from_cache = False, disable = False, supports_sdpa = [None],
    )
path = os.path.join(os.environ["UNSLOTH_COMPILE_LOCATION"], "unsloth_compiled_module_deepseek_v41.py")
generated = open(path, encoding = "utf-8").read()
out = {}
for name in sys.argv[1:]:
    where = generated.find(f"def {name}(")
    out[name] = None if where == -1 else generated[:where].rsplit("\n@", 1)[-1].strip()
where = generated.find("def apply_rotary_pos_emb(")
out["rope_split"] = "x.split([x.shape[-1] - rope_dim, rope_dim]" in generated[where:where + 3000]
print("@@@" + json.dumps(out))
'''


@needs_port
def test_generated_cache_keeps_helpers_eager(tmp_path):
    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    env["UNSLOTH_ALLOW_CPU"] = "1"
    if importlib.util.find_spec("unsloth") is None:
        env["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, *sorted(_DISABLED)], cwd = str(tmp_path), env = env,
        capture_output = True, text = True, timeout = 1800,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    line = next(l for l in result.stdout.splitlines() if l.startswith("@@@"))
    out = json.loads(line[3:])
    assert out.pop("rope_split"), out
    assert (out.pop("apply_rotary_pos_emb") or "").startswith("torch_compile_with_fallback("), out
    compiled = {k: v for k, v in out.items() if v is not None and "compile" in v and "disable" not in v}
    assert not compiled, compiled

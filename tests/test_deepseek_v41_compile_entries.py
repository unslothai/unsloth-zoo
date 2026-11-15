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
    "_fake_quant_fp8_block",
    "_fake_quant_fp4_block",
    # Unsloth's native module: host-side load / layout / registry helpers
    "engram_layout",
    "make_engram_table",
    "is_parameter_free_table",
    "_checkpoint_key_names",
    "has_core_model_loading",
    "_is_fp8_quant_config",
    "_released_layout",
    "_engram_cut_safe",
    "pipeline_safe_cut_points",
    "_resolve_sparse_attn",
    "ensure_sparse_attn",
    "_drop_engram_tables_outside_full_save",
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


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dsv41_impl import NATIVE_ALIAS, available  # noqa: E402

IMPLS = available()
needs_impl = pytest.mark.skipif(
    not IMPLS, reason = "needs a DeepSeek-V4.1 implementation (transformers port, or unsloth's native module)"
)
impl_param = pytest.mark.parametrize("impl", IMPLS, ids = lambda impl: impl.name)

# Host helpers of the native module (the rest of _DISABLED are the port's names).
_NATIVE_DISABLED = {
    "build_compressed_token_map", "compute_hash_multipliers", "_is_prime", "engram_layout", "make_engram_table",
    "is_parameter_free_table", "_checkpoint_key_names", "has_core_model_loading", "_is_fp8_quant_config",
    "_released_layout", "_engram_cut_safe", "pipeline_safe_cut_points", "_resolve_sparse_attn",
    "ensure_sparse_attn", "_drop_engram_tables_outside_full_save",
}
_PORT_DISABLED = _DISABLED - (_NATIVE_DISABLED - {"build_compressed_token_map", "compute_hash_multipliers", "_is_prime"})


@needs_impl
@impl_param
def test_implementation_defines_its_entries_and_the_rewrites_match(impl, monkeypatch):
    from unsloth_zoo.compiler import module_forward_eager_rewrites, module_forward_source_rewrites

    modeling = impl.modeling
    names = _PORT_DISABLED if impl.name == "port" else _NATIVE_DISABLED
    missing = sorted(name for name in names if not callable(getattr(modeling, name, None)))
    assert not missing, missing
    assert hasattr(modeling, "DeepseekV41HyperConnection")
    if impl.name == "port":
        assert "apply_rotary_pos_emb" in model_function_source_rewrites(modeling, "deepseek_v41")
    else:
        # The native mixer's projection and Sinkhorn loop match the native alternatives exactly once.
        monkeypatch.setenv("UNSLOTH_DSV4_MHC_FAST", "unrolled")
        eager = module_forward_eager_rewrites(modeling, "deepseek_v41")
        assert "unsloth_mhc_projection_rsqrt(" in eager["DeepseekV41HyperConnection"]
        source = module_forward_source_rewrites(modeling, "deepseek_v41")["DeepseekV41HyperConnection"]
        assert "unsloth_sinkhorn_knopp(" in source and "unsloth_mhc_projection_rsqrt(" in source


_CHILD = r'''
import os, sys, json, io, contextlib
os.environ["UNSLOTH_COMPILE_LOCATION"] = os.path.abspath("unsloth_compiled_cache")
impl_name, tests_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, tests_dir)
from unsloth_zoo.compiler import register_native_modeling, resolve_modeling_module, unsloth_compile_transformers
if impl_name == "native":
    from _dsv41_impl import NATIVE_ALIAS, native
    native()
    register_native_modeling("deepseek_v41", NATIVE_ALIAS + ".modeling")
location = resolve_modeling_module("deepseek_v41")[0]
with contextlib.redirect_stdout(io.StringIO()):
    unsloth_compile_transformers(
        model_type = "deepseek_v41", fast_lora_forwards = False, fullgraph = True,
        import_from_cache = False, disable = False, supports_sdpa = [None],
    )
path = os.path.join(os.environ["UNSLOTH_COMPILE_LOCATION"], "unsloth_compiled_module_deepseek_v41.py")
generated = open(path, encoding = "utf-8").read()
out = {"location": location}
for name in sys.argv[3:]:
    where = generated.find(f"def {name}(")
    out[name] = None if where == -1 else generated[:where].rsplit("\n@", 1)[-1].strip()
where = generated.find("def apply_rotary_pos_emb(")
out["rope_split"] = where != -1 and "x.split([x.shape[-1] - rope_dim, rope_dim]" in generated[where:where + 3000]
out["mhc"] = "unsloth_mhc_projection_rsqrt(" in generated or "unsloth_mhc_projection(" in generated
import re as _re
helpers = ("unsloth_sinkhorn_knopp", "unsloth_mhc_projection", "unsloth_mhc_projection_rsqrt")
model_imports = " ".join(_re.findall(r"from " + _re.escape(location) + r" import \(([^)]*)\)", generated))
out["helpers_from_model"] = [h for h in helpers if _re.search(rf"\b{h}\b", model_imports)]
out["helpers_missing"] = [
    h for h in helpers
    if _re.search(rf"\b{h}\(", generated)
    and f"from unsloth_zoo.temporary_patches.mhc_sinkhorn import {h}\n" not in generated
]
print("@@@" + json.dumps(out))
'''

# A fresh process: the generated cache must import on its own (the mHC helpers the compiler injects into the
# modeling module at runtime are not there yet).
_FRESH = r'''
import importlib.util, os, sys
impl_name, tests_dir, path = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, tests_dir)
if impl_name == "native":
    from _dsv41_impl import native
    native()
spec = importlib.util.spec_from_file_location("unsloth_compiled_module_deepseek_v41_fresh", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print("@@@ok")
'''


@needs_impl
@impl_param
def test_generated_cache_keeps_helpers_eager(impl, tmp_path):
    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    env["UNSLOTH_ALLOW_CPU"] = "1"
    if importlib.util.find_spec("unsloth") is None or impl.name == "native":
        env["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    names = sorted(_PORT_DISABLED if impl.name == "port" else _NATIVE_DISABLED)
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, impl.name, os.path.dirname(os.path.abspath(__file__)), *names],
        cwd = str(tmp_path), env = env, capture_output = True, text = True, timeout = 1800,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    line = next(l for l in result.stdout.splitlines() if l.startswith("@@@"))
    out = json.loads(line[3:])
    location = out.pop("location")
    mhc = out.pop("mhc")
    rope_split = out.pop("rope_split")
    assert out.pop("helpers_from_model") == [], "injected mHC helpers imported from the model module"
    assert out.pop("helpers_missing") == [], "an mHC helper used without its zoo import"
    fresh = subprocess.run(
        [sys.executable, "-c", _FRESH, impl.name, os.path.dirname(os.path.abspath(__file__)),
         str(tmp_path / "unsloth_compiled_cache" / "unsloth_compiled_module_deepseek_v41.py")],
        cwd = str(tmp_path), env = env, capture_output = True, text = True, timeout = 1800,
    )
    assert "@@@ok" in fresh.stdout, fresh.stdout[-3000:] + fresh.stderr[-3000:]
    if impl.name == "port":
        assert location.startswith("transformers.models.deepseek_v41"), location
        assert rope_split, out
        assert (out.pop("apply_rotary_pos_emb") or "").startswith("torch_compile_with_fallback("), out
    else:
        # The compiler found the native module (it used to return quietly) and applied the mHC rewrite.
        assert location == NATIVE_ALIAS + ".modeling", location
        assert mhc, out
    assert any(v is not None for v in out.values()), out
    compiled = {k: v for k, v in out.items() if v is not None and "compile" in v and "disable" not in v}
    assert not compiled, compiled

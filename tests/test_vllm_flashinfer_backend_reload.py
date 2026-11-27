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

"""vLLM < 0.13: earlier load_vllm FLASHINFER backend is no user choice; head size: head_dim first."""

import ast
import importlib
import inspect
import textwrap
import types

from packaging.version import Version

vllm_utils = importlib.import_module("unsloth_zoo.vllm_utils")


def _is_backend_assign(stmt):
    return (
        isinstance(stmt, ast.Assign)
        and any(
            isinstance(t, ast.Subscript) and getattr(t.slice, "value", "") == "VLLM_ATTENTION_BACKEND"
            for t in stmt.targets
        )
        and getattr(stmt.value, "value", "") == "FLASHINFER"
    )


def _chain():
    tree = ast.parse(textwrap.dedent(inspect.getsource(vllm_utils.load_vllm)))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "in os.environ" in ast.unparse(node.test)
        and any(_is_backend_assign(n) for n in ast.walk(node))
    )


def _load(env, config, major_version = 9, vllm_version = "0.11.2"):
    namespace = {
        "os": types.SimpleNamespace(environ = env),
        "Version": Version,
        "config": config,
        "model_name": "m",
        "vllm_supports_flashinfer": lambda config: True,
        "use_bitsandbytes": False,
        "major_version": major_version,
        "vllm_version": vllm_version,
    }
    namespace.update({k: v for k, v in vars(vllm_utils).items() if k.startswith("_vllm_flashinfer")})
    exec(compile(ast.Module(body = [_chain()], type_ignores = []), "<backend chain>", "exec"), namespace)
    return env.get("VLLM_ATTENTION_BACKEND")


LLAMA = types.SimpleNamespace(hidden_size = 4096, num_attention_heads = 32)  # 128
PHI3 = types.SimpleNamespace(hidden_size = 3072, num_attention_heads = 32)  # 96


def test_second_load_in_one_process_does_not_inherit_flashinfer():
    env = {}
    assert _load(env, LLAMA) == "FLASHINFER"
    assert _load(env, PHI3) is None  # vLLM 0.11.2 / 0.12 would reject a forced FLASHINFER for head size 96
    assert _load(env, LLAMA) == "FLASHINFER"


def test_user_backend_is_still_kept_across_loads():
    env = {"VLLM_ATTENTION_BACKEND": "FLASHINFER"}
    assert _load(env, PHI3) == "FLASHINFER"
    env = {"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"}
    assert _load(env, LLAMA) == "FLASH_ATTN"


def test_hidden_size_per_head_is_read_before_the_division():
    config = types.SimpleNamespace(hidden_size = 2048, num_attention_heads = 16, hidden_size_per_head = 192)
    assert not vllm_utils._vllm_flashinfer_head_size_ok(config)
    assert _load({}, config) is None

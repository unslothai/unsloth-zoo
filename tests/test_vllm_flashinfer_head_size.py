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

"""load_vllm must not force FLASHINFER for head sizes it rejects (only 64 / 128 / 256)."""

import ast
import importlib
import inspect
import textwrap
import types

import pytest
from packaging.version import Version

vllm_utils = importlib.import_module("unsloth_zoo.vllm_utils")


def _forces_flashinfer(stmt):
    return (
        isinstance(stmt, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and getattr(target.slice, "value", "") == "VLLM_ATTENTION_BACKEND"
            for target in stmt.targets
        )
        and getattr(stmt.value, "value", "") == "FLASHINFER"
    )


def _backend_env(config, major_version = 9, vllm_version = "0.11.2", use_bitsandbytes = False, inherited = None):
    tree = ast.parse(textwrap.dedent(inspect.getsource(vllm_utils.load_vllm)))
    chain = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "in os.environ" in ast.unparse(node.test)
        and any(_forces_flashinfer(n) for n in ast.walk(node))
    )
    env = {} if inherited is None else {"VLLM_ATTENTION_BACKEND": inherited}
    namespace = {
        "os": types.SimpleNamespace(environ = env),
        "Version": Version,
        "config": config,
        "model_name": "m",
        "vllm_supports_flashinfer": lambda config: True,
        "use_bitsandbytes": use_bitsandbytes,
        "major_version": major_version,
        "vllm_version": vllm_version,
    }
    namespace.update({k: v for k, v in vars(vllm_utils).items() if k.startswith("_vllm_flashinfer")})
    exec(compile(ast.Module(body = [chain], type_ignores = []), "<backend chain>", "exec"), namespace)
    return env.get("VLLM_ATTENTION_BACKEND")


def _config(**kw):
    return types.SimpleNamespace(**kw)


PHI3 = _config(hidden_size = 3072, num_attention_heads = 32)  # head_dim 96, no head_dim field
TINY_QWEN3 = _config(hidden_size = 8, num_attention_heads = 4, head_dim = 16)
LLAMA = _config(hidden_size = 4096, num_attention_heads = 32)  # 128
QWEN3 = _config(hidden_size = 1024, num_attention_heads = 16, head_dim = 128)
GEMMA2 = _config(hidden_size = 2304, num_attention_heads = 8, head_dim = 256)
QWEN25_05B = _config(hidden_size = 896, num_attention_heads = 14)  # 64
DEEPSEEK_MLA = _config(hidden_size = 7168, num_attention_heads = 128, kv_lora_rank = 512, q_lora_rank = 1536)
GEMMA4 = _config(hidden_size = 2560, num_attention_heads = 8, head_dim = 256, global_head_dim = 512)


@pytest.mark.parametrize("config", [PHI3, TINY_QWEN3, DEEPSEEK_MLA, GEMMA4], ids = ["phi3_96", "tiny_16", "mla", "gemma4_512"])
@pytest.mark.parametrize("major_version,use_bitsandbytes", [(9, False), (8, True), (7, False)])
def test_unsupported_head_size_is_not_forced(config, major_version, use_bitsandbytes):
    assert _backend_env(config, major_version = major_version, use_bitsandbytes = use_bitsandbytes) is None


@pytest.mark.parametrize("config", [LLAMA, QWEN3, GEMMA2, QWEN25_05B], ids = ["llama_128", "qwen3_128", "gemma2_256", "qwen25_64"])
def test_supported_head_sizes_still_get_flashinfer(config):
    assert _backend_env(config) == "FLASHINFER"


def test_a_user_choice_is_kept():
    assert _backend_env(PHI3, inherited = "FLASH_ATTN") == "FLASH_ATTN"


def test_vision_config_reads_the_text_head_size():
    vlm = types.SimpleNamespace(get_text_config = lambda: PHI3, hidden_size = 4096, num_attention_heads = 32)
    assert _backend_env(vlm) is None

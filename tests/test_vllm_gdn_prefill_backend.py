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

"""vLLM picks FlashInfer GDN prefill by GPU alone, so a flashinfer lacking the keywords vLLM passes must route GDN prefill to Triton, not fail at first prefill."""

import ast
import functools
import inspect
import types
from unittest import mock

from unsloth_zoo import vllm_utils


# vLLM 0.31's caller, verbatim in shape: the probe reads the keywords from its source.
def fi_chunk_gated_delta_rule(q, k, v, g, beta, initial_state, output_final_state, cu_seqlens = None):
    from flashinfer.gdn_prefill import chunk_gated_delta_rule as chunk_gated_delta_rule_fi
    return chunk_gated_delta_rule_fi(
        q = q, k = k, v = v, g = g, beta = beta, initial_state = initial_state,
        output_final_state = output_final_state, cu_seqlens = cu_seqlens, backend = "flashinfer",
    )


def flashinfer_0_6(q, k, v, g = None, beta = None, scale = None, initial_state = None,
                   output_final_state = False, cu_seqlens = None, use_qk_l2norm_in_kernel = False):
    pass


def flashinfer_0_7(q, k, v, g = None, beta = None, scale = None, initial_state = None,
                   output_final_state = False, cu_seqlens = None, use_qk_l2norm_in_kernel = False,
                   backend = "auto"):
    pass


def _logged(fn):
    # flashinfer wraps its API in a *args / **kwargs logger that keeps __wrapped__.
    @functools.wraps(fn)
    def _auto_dump_wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return _auto_dump_wrapper


def test_missing_keyword_is_reported_through_the_logging_wrapper():
    assert vllm_utils._gdn_prefill_kwarg_mismatch(fi_chunk_gated_delta_rule, _logged(flashinfer_0_6)) == ["backend"]


def test_matching_flashinfer_is_left_alone():
    assert vllm_utils._gdn_prefill_kwarg_mismatch(fi_chunk_gated_delta_rule, _logged(flashinfer_0_7)) is None


def test_var_keyword_callee_accepts_anything():
    def any_kwargs(q, k, v, **kwargs): pass
    assert vllm_utils._gdn_prefill_kwarg_mismatch(fi_chunk_gated_delta_rule, any_kwargs) is None


def test_unreadable_source_is_not_a_mismatch():
    assert vllm_utils._gdn_prefill_kwarg_mismatch(len, flashinfer_0_6) is None


def test_unimportable_flashinfer_routes_to_triton(monkeypatch):
    gdn = types.ModuleType("qwen_gdn_linear_attn")
    gdn.fi_chunk_gated_delta_rule = fi_chunk_gated_delta_rule
    real_import = __import__
    def fake_import(name, globals = None, locals = None, fromlist = (), level = 0):
        if name == "vllm.model_executor.layers.mamba.gdn":
            return types.SimpleNamespace(qwen_gdn_linear_attn = gdn)
        if name == "flashinfer.gdn_prefill":
            raise ImportError("blocked")
        return real_import(name, globals, locals, fromlist, level)
    monkeypatch.setattr("builtins.__import__", fake_import)
    assert "not importable" in vllm_utils._flashinfer_gdn_prefill_mismatch()


def test_gdn_layer_detection():
    qwen35 = types.SimpleNamespace(text_config = types.SimpleNamespace(linear_key_head_dim = 128))
    layer_types = types.SimpleNamespace(layer_types = ["linear_attention", "full_attention"])
    dense = types.SimpleNamespace(model_type = "qwen3_moe")
    assert vllm_utils._config_has_gdn_layers(qwen35)
    assert vllm_utils._config_has_gdn_layers(layer_types)
    assert not vllm_utils._config_has_gdn_layers(dense)


def test_load_vllm_sets_triton_only_for_gdn_models_without_an_explicit_choice():
    source = inspect.getsource(vllm_utils.load_vllm)
    guards = [
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.If) and "_config_has_gdn_layers" in ast.unparse(node.test)
    ]
    assert len(guards) == 1
    assert ast.unparse(guards[0].test) == "_config_has_gdn_layers(config) and 'gdn_prefill_backend' not in engine_args"
    assert "engine_args['gdn_prefill_backend'] = 'triton'" in ast.unparse(guards[0])

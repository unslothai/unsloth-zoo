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

"""load_vllm(speculative_config = ...) / UNSLOTH_VLLM_SPECULATIVE_CONFIG: any vLLM speculative method, resolved before
the memory estimate (a draft model's weights come out of the KV cache budget), then through the same LoRA lookahead
fitting, memory retry and padded-vocab guard as suffix decoding. load_vllm needs a GPU, so wiring is checked on the AST."""

import ast
import json
import os
import pathlib
import sys
import types
import typing

import pytest

import unsloth_zoo

ROOT = pathlib.Path(unsloth_zoo.__file__).resolve().parents[1]
_FUNCS = (
    "_vllm_suffix_decoding_config", "_vllm_speculative_methods", "_check_vllm_suffix_decoding",
    "_vllm_speculative_config", "_speculative_draft_gb", "_speculation_lookahead",
    "_fit_max_num_seqs_to_speculation", "_fit_max_num_batched_tokens_to_speculation", "_shrink_for_memory_retry",
)
_CONSTS = ("_SPECULATIVE_CONFIG_ENV", "_MRV2_UNSUPPORTED_SPECULATIVE_METHODS", "_SPECULATIVE_DRAFT_METHODS")
_METHODS = typing.Literal["ngram", "medusa", "draft_model", "suffix", typing.Literal["eagle", "eagle3", typing.Literal["mtp"]], typing.Literal["ngram_gpu"]]


def _tree():
    return ast.parse((ROOT / "unsloth_zoo" / "vllm_utils.py").read_text(encoding = "utf-8-sig"))


def _load_vllm():
    return next(n for n in ast.walk(_tree()) if isinstance(n, ast.FunctionDef) and n.name == "load_vllm")


@pytest.fixture
def ns(monkeypatch):
    def make(methods = _METHODS, has_arctic = True, hf_api = None):
        body = []
        for n in _tree().body:
            if isinstance(n, ast.FunctionDef) and n.name in _FUNCS: body.append(n)
            elif isinstance(n, ast.Assign) and any(getattr(t, "id", None) in _CONSTS for t in n.targets): body.append(n)
        found = {n.name for n in body if isinstance(n, ast.FunctionDef)} | {t.id for n in body if isinstance(n, ast.Assign) for t in n.targets}
        assert found == set(_FUNCS) | set(_CONSTS), f"missing from vllm_utils.py: {set(_FUNCS) | set(_CONSTS) - found}"
        util = types.SimpleNamespace(find_spec = lambda name: object() if (name == "arctic_inference" and has_arctic) else None)
        out = {"os": os, "json": json, "importlib": types.SimpleNamespace(util = util), "vllm_version": "0.29.0"}
        exec(compile(ast.Module(body = body, type_ignores = []), "vllm_utils_spec", "exec"), out)
        for name in ("vllm", "vllm.config", "vllm.config.speculative"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        if methods is not None:
            sys.modules["vllm.config.speculative"].SpeculativeMethod = methods
        if hf_api is not None:
            hub = types.ModuleType("huggingface_hub"); hub.HfApi = hf_api
            monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
        return out
    for env in ("UNSLOTH_VLLM_SPECULATIVE_CONFIG", "UNSLOTH_VLLM_SUFFIX_DECODING", "VLLM_USE_V2_MODEL_RUNNER"):
        monkeypatch.delenv(env, raising = False)
    return make


def test_off_by_default(ns):
    assert ns(methods = None, has_arctic = False)["_vllm_speculative_config"](None) is None


def test_nested_literal_methods_are_flattened(ns):
    assert ns()["_vllm_speculative_methods"]() == {"ngram", "medusa", "draft_model", "suffix", "eagle", "eagle3", "mtp", "ngram_gpu"}
    assert ns(methods = None)["_vllm_speculative_methods"]() is None


@pytest.mark.parametrize("value", [
    {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4},
    {"method": "eagle3", "model": "org/draft", "num_speculative_tokens": 3},
    {"model": "org/draft", "num_speculative_tokens": 4},  # vLLM infers the method from the draft
    {"method": "mtp", "num_speculative_tokens": 1},
])
def test_kwarg_dict_passes_through_as_a_copy(ns, value):
    got = ns()["_vllm_speculative_config"](value)
    assert got == value and got is not value


def test_json_string_kwarg_and_env(ns, monkeypatch):
    fn = ns()["_vllm_speculative_config"]
    assert fn('{"method": "ngram", "num_speculative_tokens": 2}') == {"method": "ngram", "num_speculative_tokens": 2}
    monkeypatch.setenv("UNSLOTH_VLLM_SPECULATIVE_CONFIG", '{"method": "eagle3", "model": "x", "num_speculative_tokens": 3}')
    assert fn(None)["method"] == "eagle3"
    # The explicit kwarg beats the env var.
    assert fn({"method": "ngram", "num_speculative_tokens": 5})["method"] == "ngram"


def test_suffix_env_still_works_and_loses_to_the_new_config(ns, monkeypatch, capsys):
    fn = ns()["_vllm_speculative_config"]
    monkeypatch.setenv("UNSLOTH_VLLM_SUFFIX_DECODING", "1")
    assert fn(None) == {"method": "suffix"}
    monkeypatch.setenv("UNSLOTH_VLLM_SPECULATIVE_CONFIG", "0")
    assert fn(None) == {"method": "suffix"}
    assert fn({"method": "ngram", "num_speculative_tokens": 4})["method"] == "ngram"
    assert "UNSLOTH_VLLM_SUFFIX_DECODING is ignored" in capsys.readouterr().out


@pytest.mark.parametrize("value, match", [
    ("{bad", "not valid JSON"),
    ("[1, 2]", "must be a dict"),
    (5, "must be a dict"),
    ({"num_speculative_tokens": 4}, "needs a 'method'"),
    ({"method": "warp_drive", "num_speculative_tokens": 4}, "not one this vLLM"),
])
def test_bad_values_raise(ns, value, match):
    with pytest.raises(ValueError, match = match):
        ns()["_vllm_speculative_config"](value)


def test_bad_env_names_the_env(ns, monkeypatch):
    monkeypatch.setenv("UNSLOTH_VLLM_SPECULATIVE_CONFIG", "ngram")
    with pytest.raises(ValueError, match = "UNSLOTH_VLLM_SPECULATIVE_CONFIG is not valid JSON"):
        ns()["_vllm_speculative_config"](None)


def test_suffix_method_keeps_the_arctic_check(ns):
    with pytest.raises(RuntimeError, match = "arctic-inference.*drop speculative_config"):
        ns(has_arctic = False)["_vllm_speculative_config"]({"method": "suffix"})
    with pytest.raises(RuntimeError, match = "vLLM >= 0.11.1"):
        ns(methods = typing.Literal["ngram"])["_vllm_speculative_config"]({"method": "suffix"})


def test_unknown_method_list_is_left_to_vllm(ns):
    # A vLLM without the Literal (or a renamed module) cannot be checked here; vLLM's own config validation decides.
    assert ns(methods = None)["_vllm_speculative_config"]({"method": "eagle3", "model": "x"})["method"] == "eagle3"


@pytest.mark.parametrize("method, refused", [("ngram", True), ("suffix", True), ("eagle3", False), ("mtp", False)])
def test_forced_model_runner_v2_refuses_v1_only_methods(ns, monkeypatch, method, refused):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    fn = ns()["_vllm_speculative_config"]
    if refused:
        with pytest.raises(ValueError, match = "Model Runner V2"):
            fn({"method": method, "num_speculative_tokens": 2})
    else:
        assert fn({"method": method, "num_speculative_tokens": 2})["method"] == method


@pytest.mark.parametrize("config, k", [
    ({"method": "ngram", "num_speculative_tokens": 4}, 4),
    ({"method": "eagle3", "model": "x"}, None),        # k from the draft config: only vLLM knows it
    ({"model": "x"}, None),
    ({"method": "suffix"}, 24),
    ({"method": "suffix", "suffix_decoding_max_tree_depth": 8}, 8),
])
def test_lookahead_per_method(ns, config, k):
    out = ns()
    assert out["_speculation_lookahead"](config) == k
    seqs = out["_fit_max_num_seqs_to_speculation"](256, 512, config)
    tokens = out["_fit_max_num_batched_tokens_to_speculation"](seqs, 512, config)
    if k is None:
        assert (seqs, tokens) == (256, 512)
    else:
        assert tokens >= seqs * (k + 1)
    seqs, tokens = out["_shrink_for_memory_retry"](64, 512, 128, config)
    assert k is None or tokens >= seqs * (k + 1)


def test_draft_gb(ns, tmp_path):
    class Sib:
        def __init__(self, f, n): self.rfilename, self.size = f, n
    class Api:
        def model_info(self, repo, revision = None, files_metadata = False):
            assert files_metadata
            if repo == "org/missing": raise OSError("offline")
            return types.SimpleNamespace(siblings = [Sib("model.safetensors", 3 << 30), Sib("pytorch_model.bin", 3 << 30), Sib("config.json", 10)])
    fn = ns(hf_api = Api)["_speculative_draft_gb"]
    assert fn({"method": "ngram", "num_speculative_tokens": 4}, "t") == 0.0
    assert fn({"method": "mtp", "model": "org/draft"}, "t") == 0.0
    assert fn({"method": "eagle3", "model": "t"}, "t") == 0.0
    assert fn({"method": "eagle3", "model": "org/draft"}, "t") == 3.0      # safetensors only, not the .bin copy
    assert fn({"model": "org/draft"}, "t") == 3.0
    assert fn({"method": "draft_model", "model": "org/missing"}, "t") is None
    (tmp_path / "model.safetensors").write_bytes(b"\0" * (1 << 20))
    (tmp_path / "config.json").write_text("{}")
    assert fn({"method": "draft_model", "model": str(tmp_path)}, "t") == pytest.approx(1 / 1024)


def test_load_vllm_resolves_before_sizing_and_budgets_the_draft():
    fn = _load_vllm()
    assert "speculative_config" in [a.arg for a in fn.args.args]
    src = ast.unparse(fn)
    resolve = src.find("user_speculative_config = None if return_args else _vllm_speculative_config(speculative_config)")
    # The draft is sized before the capacity estimate and counted inside it, so the KV estimate, max_num_seqs and
    # the context-length fit all see the draft weights (subtracting afterwards left max_num_batched_tokens stale).
    budget = src.find("draft_gb = _speculative_draft_gb(user_speculative_config, model_name)", max(resolve, 0))
    sizing = src.find("approximate_vllm_memory_usage(", max(budget, 0))
    counted = src.find("extra_weight_bytes=int(draft_gb * 1024 ** 3)", max(sizing, 0))
    tiers = src.find("if memory_left_for_kv_cache_gb <= 2:", max(counted, 0))
    assign = src.find("engine_args['speculative_config'] = user_speculative_config", max(tiers, 0))
    arg_filter = src.find("good_keys = inspect.signature(AsyncEngineArgs", max(assign, 0))
    assert -1 < resolve < budget < sizing < counted < tiers < assign < arg_filter
    # Padded-vocab guard and memory retry read the engine's config, so they cover every method.
    assert "speculative=engine_args.get('speculative_config', None) is not None" in src
    assert "engine_args.get('speculative_config') if enable_lora else None" in src


def test_real_vllm_methods_if_installed():
    pytest.importorskip("vllm.config.speculative")
    nodes = [n for n in _tree().body if isinstance(n, ast.FunctionDef) and n.name == "_vllm_speculative_methods"]
    out = {}
    exec(compile(ast.Module(body = nodes, type_ignores = []), "vllm_utils_real", "exec"), out)
    methods = out["_vllm_speculative_methods"]()
    assert {"ngram", "draft_model", "eagle3"} <= methods


def test_draft_weights_shrink_the_capacity_estimate(monkeypatch):
    # The draft is counted inside the estimate, so max_num_batched_tokens (which the context-length fit reads) and the
    # KV budget both shrink with it, instead of only the KV budget being lowered afterwards.
    import types
    import unsloth_zoo.vllm_utils as vu
    monkeypatch.setattr(vu, "get_mem_info", lambda: (80 * 1024**3, 80 * 1024**3))
    config = types.SimpleNamespace(
        vocab_size = 32000, hidden_size = 2048, max_position_embeddings = 131072, num_hidden_layers = 24,
        num_key_value_heads = 8, num_attention_heads = 16, intermediate_size = 8192, tie_word_embeddings = True,
    )
    kwargs = dict(max_seq_length = 131072, gpu_memory_utilization = 0.5, max_num_batched_tokens = 8192)
    base = vu.approximate_vllm_memory_usage(config, **kwargs)
    with_draft = vu.approximate_vllm_memory_usage(config, extra_weight_bytes = 3 * 1024**3, **kwargs)
    assert abs((base[3] - with_draft[3]) - 3.0) < 1e-6, (base, with_draft)
    assert with_draft[0] <= base[0] and with_draft[1] <= base[1]

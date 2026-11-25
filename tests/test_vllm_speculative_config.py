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

"""UNSLOTH_VLLM_SPECULATIVE_CONFIG resolved before memory estimate; AST-checked, no GPU."""

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
    "_normalize_speculative_keys", "_vllm_suffix_decoding_config", "_vllm_speculative_methods", "_vllm_release",
    "_check_vllm_suffix_decoding", "_vllm_speculative_config", "_redact_speculative_config",
    "_speculative_config_has_secret", "_patch_vllm_log_non_default_args", "_mrv2_unsupported_speculative_methods",
    "_speculative_method", "_same_checkpoint", "_select_draft_weight_files", "_speculative_draft_load_format", "_speculative_draft_gb",
    "_routing_replay_will_drop_speculation", "_resolve_speculative_depth", "_speculation_lookahead",
    "_fit_max_num_seqs_to_speculation", "_fit_max_num_batched_tokens_to_speculation", "_shrink_for_memory_retry",
)
_CONSTS = (
    "_SPECULATIVE_CONFIG_ENV", "_SUPPORTED_SPECULATIVE_METHODS", "_LOSSLESS_REJECTION_SAMPLE_METHODS",
    "_SECRET_KEY_PATTERN", "_MRV2_UNSUPPORTED_BY_RELEASE", "_MRV2_UNSUPPORTED_BEFORE_0_31",
    "_SPECULATIVE_DRAFT_METHODS", "_NOT_FOR_INFERENCE_WEIGHTS", "_HUB_REPO_ID",
)
_METHODS = typing.Literal[
    "ngram", "medusa", "draft_model", "suffix", "custom_class",
    typing.Literal["eagle", "eagle3", "dflash", typing.Literal["mtp", "deepseek_mtp"]], typing.Literal["ngram_gpu"],
]


def _tree():
    return ast.parse((ROOT / "unsloth_zoo" / "vllm_utils.py").read_text(encoding = "utf-8-sig"))


def _load_vllm():
    return next(n for n in ast.walk(_tree()) if isinstance(n, ast.FunctionDef) and n.name == "load_vllm")


@pytest.fixture
def ns(monkeypatch):
    def make(methods = _METHODS, has_arctic = True, hf_api = None, version = "0.29.0"):
        body = []
        for n in _tree().body:
            if isinstance(n, ast.FunctionDef) and n.name in _FUNCS: body.append(n)
            elif isinstance(n, ast.Assign) and any(getattr(t, "id", None) in _CONSTS for t in n.targets): body.append(n)
        found = {n.name for n in body if isinstance(n, ast.FunctionDef)} | {t.id for n in body if isinstance(n, ast.Assign) for t in n.targets}
        assert found == set(_FUNCS) | set(_CONSTS), f"missing from vllm_utils.py: {set(_FUNCS) | set(_CONSTS) - found}"
        util = types.SimpleNamespace(find_spec = lambda name: object() if (name == "arctic_inference" and has_arctic) else None)
        import copy, functools, inspect, re
        from packaging.version import Version
        out = {"os": os, "json": json, "re": re, "copy": copy, "functools": functools, "inspect": inspect,
               "Version": Version, "importlib": types.SimpleNamespace(util = util), "vllm_version": version,
               "_UNSLOTH_SET_V1_RUNNER": False, "is_hip": lambda: False,
               "_vllm_rocm_default_v1_architectures": lambda: frozenset()}
        exec(compile(ast.Module(body = body, type_ignores = []), "vllm_utils_spec", "exec"), out)
        for name in ("vllm", "vllm.config", "vllm.config.speculative"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        if methods is not None:
            sys.modules["vllm.config.speculative"].SpeculativeMethod = methods
        if hf_api is not None:
            hub = types.ModuleType("huggingface_hub"); hub.HfApi = hf_api
            monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
        return out
    for env in ("UNSLOTH_VLLM_SPECULATIVE_CONFIG", "UNSLOTH_VLLM_SUFFIX_DECODING", "VLLM_USE_V2_MODEL_RUNNER", "UNSLOTH_ROUTING_REPLAY"):
        monkeypatch.delenv(env, raising = False)
    return make


def test_off_by_default(ns):
    assert ns(methods = None, has_arctic = False)["_vllm_speculative_config"](None) is None
    assert ns(methods = None, has_arctic = False)["_vllm_speculative_config"](False) is None


def test_draft_sizing_edge_paths(ns, tmp_path):
    called = []
    class Api:
        def model_info(self, repo, **k):
            called.append(repo); raise OSError("no")
    fn = ns(hf_api = Api)["_speculative_draft_gb"]
    assert fn({"method": "draft_model", "model": "./does/not/exist"}, "t") is None and called == []
    (tmp_path / "model.safetensors").write_bytes(b"\0" * (1 << 20))
    (tmp_path / "broken.safetensors").symlink_to(tmp_path / "gone.safetensors")
    assert fn({"method": "draft_model", "model": str(tmp_path)}, "t") == pytest.approx(1 / 1024)
    # MTP in the target's own checkpoint, by an equivalent path or the Hub snapshot the repo id resolved to.
    assert fn({"method": "mtp", "model": str(tmp_path) + "/"}, str(tmp_path)) == 0.0
    snap = tmp_path / "hub" / "models--org--target" / "snapshots" / "abc"; snap.mkdir(parents = True)
    assert fn({"method": "mtp", "model": "org/target"}, str(snap)) == 0.0


def test_nested_literal_methods_are_flattened(ns):
    assert ns()["_vllm_speculative_methods"]() == {
        "ngram", "medusa", "draft_model", "suffix", "custom_class", "eagle", "eagle3", "dflash", "mtp", "deepseek_mtp", "ngram_gpu",
    }
    assert ns(methods = None)["_vllm_speculative_methods"]() is None


@pytest.mark.parametrize("value", [
    {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4},
    {"method": "eagle3", "model": "org/draft", "num_speculative_tokens": 3},
    {"method": "eagle", "model": "org/draft", "num_speculative_tokens": 3},
    {"method": "draft_model", "model": "org/draft", "num_speculative_tokens": 4},
    {"method": "mtp", "num_speculative_tokens": 1},
    {"method": "eagle3", "model": "org/draft", "num_speculative_tokens": 3, "rejection_sample_method": "block"},
])
def test_supported_methods_pass_through_as_a_copy(ns, value):
    got = ns()["_vllm_speculative_config"](value)
    assert got == value and got is not value


def test_hyphenated_keys_are_normalized_like_vllm(ns, monkeypatch):
    # (15) vLLM's create_speculative_config rewrites '-' to '_'; the LoRA fit must read the same key.
    out = ns()
    got = out["_vllm_speculative_config"]({"method": "ngram", "num-speculative-tokens": 7})
    assert got == {"method": "ngram", "num_speculative_tokens": 7}
    assert out["_speculation_lookahead"](got) == 7
    monkeypatch.setenv("UNSLOTH_VLLM_SUFFIX_DECODING", '{"num-speculative-tokens": 5}')
    assert out["_vllm_speculative_config"](None) == {"num_speculative_tokens": 5, "method": "suffix"}


def test_json_string_kwarg_and_env(ns, monkeypatch):
    fn = ns()["_vllm_speculative_config"]
    assert fn('{"method": "ngram", "num_speculative_tokens": 2}') == {"method": "ngram", "num_speculative_tokens": 2}
    monkeypatch.setenv("UNSLOTH_VLLM_SPECULATIVE_CONFIG", '{"method": "eagle3", "model": "x", "num_speculative_tokens": 3}')
    assert fn(None)["method"] == "eagle3"
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
    # (7) / (11) / (16) vLLM would infer the method from the draft name (ngram alias, Python proposer path).
    ({"num_speculative_tokens": 4}, "explicit 'method'"),
    ({"model": "org/draft", "num_speculative_tokens": 4}, "explicit 'method'"),
    ({"model": "ngram", "num_speculative_tokens": 4}, "explicit 'method'"),
    ({"model": "pkg.module.Proposer", "num_speculative_tokens": 4}, "explicit 'method'"),
    # (24) a non-string method gets a clear error, not a TypeError from a set lookup.
    ({"method": ["ngram"], "num_speculative_tokens": 4}, "must be a string"),
    ({"method": {"x": 1}, "num_speculative_tokens": 4}, "must be a string"),
    # (6) unverified methods and plugin speculators are refused explicitly.
    ({"method": "warp_drive", "num_speculative_tokens": 4}, "not supported for Unsloth rollouts"),
    ({"method": "medusa", "model": "x", "num_speculative_tokens": 4}, "not supported for Unsloth rollouts"),
    ({"method": "custom_class", "model": "pkg.P", "num_speculative_tokens": 4}, "not supported for Unsloth rollouts"),
    ({"method": "dflash", "model": "x", "num_speculative_tokens": 4}, "not supported for Unsloth rollouts"),
    ({"method": "ngram_gpu", "num_speculative_tokens": 4}, "not supported for Unsloth rollouts"),
    ({"method": "deepseek_mtp", "num_speculative_tokens": 1}, "Use 'mtp'"),
    ({"method": "eagle3", "num_speculative_tokens": 3}, "needs the draft checkpoint"),
    ({"method": "draft_model", "model": 3, "num_speculative_tokens": 3}, "'model' must be a string"),
    ({"method": "draft_model", "model": "x", "revision": 2, "num_speculative_tokens": 3}, "'revision' must be a string"),
    # (12) lossy verification would sample rollouts from another distribution than the reported target logprobs.
    ({"method": "eagle3", "model": "x", "num_speculative_tokens": 3, "rejection_sample_method": "synthetic"}, "lossy"),
    ({"method": "ngram", "num_speculative_tokens": 3, "rejection_sample_method": "fly"}, "lossy"),
])
def test_bad_values_raise(ns, value, match):
    with pytest.raises(ValueError, match = match):
        ns()["_vllm_speculative_config"](value)


def test_lossy_verification_allowed_outside_training(ns):
    cfg = {"method": "ngram", "num_speculative_tokens": 3, "rejection_sample_method": "synthetic"}
    assert ns()["_vllm_speculative_config"](cfg, training = False) == cfg


def test_method_missing_from_the_installed_vllm_is_refused(ns):
    # (1) / (5) support comes from the installed vLLM: 0.34 dropped custom_class / medusa from the Literal.
    with pytest.raises(ValueError, match = "not one this vLLM"):
        ns(methods = typing.Literal["ngram", "suffix"])["_vllm_speculative_config"]({"method": "eagle3", "model": "x", "num_speculative_tokens": 3})


def test_bad_env_names_the_env(ns, monkeypatch):
    monkeypatch.setenv("UNSLOTH_VLLM_SPECULATIVE_CONFIG", "ngram")
    with pytest.raises(ValueError, match = "UNSLOTH_VLLM_SPECULATIVE_CONFIG is not valid JSON"):
        ns()["_vllm_speculative_config"](None)


def test_suffix_arctic_check_is_per_release(ns):
    # (8) vLLM 0.34 ships native suffix decoding; older releases need Arctic Inference.
    with pytest.raises(RuntimeError, match = "arctic-inference.*drop speculative_config"):
        ns(has_arctic = False, version = "0.33.0")["_vllm_speculative_config"]({"method": "suffix"})
    assert ns(has_arctic = False, version = "0.34.0")["_vllm_speculative_config"]({"method": "suffix"}) == {"method": "suffix"}
    with pytest.raises(ValueError, match = "not one this vLLM"):
        ns(methods = typing.Literal["ngram"])["_vllm_speculative_config"]({"method": "suffix"})


@pytest.mark.parametrize("method", ["ngram", "suffix", "draft_model", "eagle3"])
def test_forced_model_runner_v2_is_left_to_vllm(ns, monkeypatch, method):
    # (1) / (3) no hard-coded runner preflight: vLLM validates its own runner, and replay arbitrates first.
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    cfg = {"method": method, "num_speculative_tokens": 2}
    if method in ("draft_model", "eagle3"): cfg["model"] = "x"
    assert ns()["_vllm_speculative_config"](cfg)["method"] == method


@pytest.mark.parametrize("release, blocked, allowed", [
    ((0, 31), {"ngram", "suffix", "custom_class"}, {"eagle3", "draft_model", "mtp"}),
    ((0, 32), {"suffix", "medusa"}, {"ngram", "ngram_gpu", "eagle3", "draft_model"}),
    ((0, 33), {"suffix", "custom_class"}, {"ngram", "medusa", "draft_model"}),
    ((0, 34), set(), {"ngram", "suffix", "draft_model", "eagle3"}),
    ((0, 30), {"ngram", "draft_model", "suffix"}, {"eagle3", "mtp"}),
])
def test_mrv2_unsupported_methods_follow_the_installed_release(ns, release, blocked, allowed):
    got = ns()["_mrv2_unsupported_speculative_methods"](release)
    assert blocked <= got and not (allowed & got)


def test_replay_drop_is_decided_before_the_draft_is_charged(ns, monkeypatch):
    # (13) replay drops a V2-unsupported draft before the memory estimate, so it is never charged.
    out = ns(version = "0.31.0")
    rr = types.ModuleType("unsloth_zoo.routing_replay")
    rr.routing_replay_requested = lambda: True
    rr._config_is_moe = lambda config: True
    rr._vllm_at_least = lambda a, b: (0, 31) >= (a, b)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.routing_replay", rr)
    out["__name__"] = "unsloth_zoo.vllm_utils"; out["__package__"] = "unsloth_zoo"
    fn = out["_routing_replay_will_drop_speculation"]
    cfg = types.SimpleNamespace(architectures = ["Qwen3MoeForCausalLM"])
    assert fn(cfg, {"method": "ngram", "num_speculative_tokens": 4}) is True
    assert fn(cfg, {"method": "eagle3", "model": "x", "num_speculative_tokens": 3}) is False
    assert fn(cfg, {"method": "ngram", "num_speculative_tokens": 4}, enable_prefix_caching = False) is False  # replay goes off
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    assert fn(cfg, {"method": "ngram", "num_speculative_tokens": 4}) is False  # user V1: replay goes off, spec stays
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER")
    rr.routing_replay_requested = lambda: False
    assert fn(cfg, {"method": "ngram", "num_speculative_tokens": 4}) is False
    assert fn(cfg, None) is False


@pytest.mark.parametrize("config, k", [
    ({"method": "ngram", "num_speculative_tokens": 4}, 4),
    ({"method": "eagle3", "model": "x"}, None),        # resolved by _resolve_speculative_depth before the fit
    ({"method": "suffix"}, 24),
    ({"method": "suffix", "suffix_decoding_max_tree_depth": 8}, 8),
])
def test_lookahead_per_method(ns, config, k):
    out = ns()
    assert out["_speculation_lookahead"](config) == k


class _FakeEngineArgs:
    tensor_parallel_size = pipeline_parallel_size = 1

    def __init__(self, model = None, enable_lora = False, max_num_seqs = None, max_num_batched_tokens = None,
                 speculative_config = None):
        self.speculative_config = speculative_config

    def create_model_config(self):
        return object()

    def create_speculative_config(self, target, parallel):
        assert "num_speculative_tokens" not in self.speculative_config
        return types.SimpleNamespace(num_speculative_tokens = {"eagle3": 3, "suffix": 8, "mtp": 2}[self.speculative_config["method"]])


@pytest.fixture
def fake_parallel(ns):
    def make(**kw):
        out = ns(**kw)
        sys.modules["vllm.config"].ParallelConfig = lambda **k: types.SimpleNamespace(**k)
        return out
    return make


def test_checkpoint_depth_fits_lora_on_start_and_retry(fake_parallel):
    # (2) / (21) the draft's n_predict (here 3) is what vLLM checks: tokens >= seqs * (k + 1), at start and on retry.
    out = fake_parallel()
    args = {"model": "t", "enable_lora": True, "max_num_seqs": 32, "max_num_batched_tokens": 64,
            "speculative_config": {"method": "eagle3", "model": "org/draft"}}
    spec = out["_resolve_speculative_depth"](args, _FakeEngineArgs)
    assert spec["num_speculative_tokens"] == 3 and args["speculative_config"] is spec
    seqs = out["_fit_max_num_seqs_to_speculation"](32, 64, spec)
    assert (seqs, out["_fit_max_num_batched_tokens_to_speculation"](seqs, 64, spec)) == (16, 64)
    s2, t2 = out["_shrink_for_memory_retry"](16, 64, 128, args["speculative_config"])
    assert t2 >= s2 * 4
    # Before: an unresolved depth left (32, 64) unfitted, which vLLM refuses (64 < 32 * 4).
    unresolved = {"method": "eagle3", "model": "org/draft"}
    assert out["_fit_max_num_seqs_to_speculation"](32, 64, unresolved) == 32


def test_suffix_depth_comes_from_the_installed_vllm(fake_parallel):
    # (10) vLLM 0.34 native suffix defaults to 8, not Arctic's 24.
    args = {"model": "t", "speculative_config": {"method": "suffix"}}
    assert fake_parallel()["_resolve_speculative_depth"](args, _FakeEngineArgs)["num_speculative_tokens"] == 8


def test_explicit_depth_skips_the_probe(ns):
    args = {"speculative_config": {"method": "ngram", "num_speculative_tokens": 4}}
    class Boom:
        def __init__(self, **k): raise AssertionError("probe ran")
    assert ns()["_resolve_speculative_depth"](args, Boom)["num_speculative_tokens"] == 4


def test_unresolvable_depth_asks_for_it(fake_parallel):
    class NoDepth(_FakeEngineArgs):
        def create_speculative_config(self, target, parallel):
            raise ValueError("num_speculative_tokens must be provided")
    with pytest.raises(ValueError, match = "Pass num_speculative_tokens explicitly"):
        fake_parallel()["_resolve_speculative_depth"]({"model": "t", "speculative_config": {"method": "ngram"}}, NoDepth)


def test_banner_and_vllm_log_redact_nested_credentials(ns, monkeypatch):
    # (14) / (20) a credential nested in the draft loader config is never printed.
    out = ns()
    cfg = {"method": "eagle3", "model": "x", "num_speculative_tokens": 3,
           "draft_load_config": {"model_loader_extra_config": {"s3_secret_access_key": "SEKRET", "s3_endpoint": "e"}}}
    red = out["_redact_speculative_config"](cfg)
    assert "SEKRET" not in repr(red) and red["draft_load_config"]["model_loader_extra_config"]["s3_endpoint"] == "e"
    assert red["num_speculative_tokens"] == 3
    assert out["_speculative_config_has_secret"](cfg) and not out["_speculative_config_has_secret"]({"method": "ngram", "num_speculative_tokens": 3})
    logged = []
    llm_mod = types.ModuleType("vllm.entrypoints.llm")
    llm_mod.log_non_default_args = lambda args: logged.append(repr(args.speculative_config))
    monkeypatch.setitem(sys.modules, "vllm.entrypoints", types.ModuleType("vllm.entrypoints"))
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.llm", llm_mod)
    assert out["_patch_vllm_log_non_default_args"]() is True
    args = types.SimpleNamespace(speculative_config = cfg)
    llm_mod.log_non_default_args(args)
    assert logged and "SEKRET" not in logged[0] and args.speculative_config is cfg
    assert out["_patch_vllm_log_non_default_args"]() is True  # idempotent


def test_draft_weight_selection_matches_vllms_default_loader(ns):
    # (4) / (25) one format, root-level files only, index-filtered, like DefaultModelLoader.
    sel = ns()["_select_draft_weight_files"]
    G = 1 << 30
    files = [("model.safetensors", 1 * G), ("pytorch_model.bin", 1 * G), ("checkpoint-500/model.safetensors", 1 * G),
             ("training_args.bin", 10), ("config.json", 10)]
    assert sel(files, "auto") == 1.0
    assert sel([("pytorch_model.bin", 2 * G), ("training_args.bin", 10), ("optimizer.bin", G)], "auto") == pytest.approx(2.0, rel = 1e-6)
    assert sel([("consolidated.safetensors", 2 * G), ("model-00001.safetensors", G)], "auto") == 2.0
    shards = [("model-00001-of-00002.safetensors", G), ("model-00002-of-00002.safetensors", G), ("consolidated.safetensors", 2 * G)]
    assert sel(shards, "safetensors", index = {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}) == 2.0
    assert sel([("model.safetensors", None)], "auto") is None
    assert sel(files, "gguf") is None
    assert sel([("config.json", 1)], "auto") is None


def test_draft_gb(ns, tmp_path):
    class Sib:
        def __init__(self, f, n): self.rfilename, self.size = f, n
    class Api:
        def model_info(self, repo, revision = None, files_metadata = False):
            assert files_metadata
            if repo == "org/missing": raise OSError("offline")
            return types.SimpleNamespace(siblings = [Sib("model.safetensors", 3 << 30), Sib("pytorch_model.bin", 3 << 30),
                                                     Sib("ckpt/model.safetensors", 3 << 30), Sib("config.json", 10)])
    fn = ns(hf_api = Api)["_speculative_draft_gb"]
    assert fn({"method": "ngram", "num_speculative_tokens": 4}, "t") == 0.0
    assert fn({"method": "suffix"}, "t") == 0.0
    assert fn({"method": "mtp"}, "t") == 0.0                                  # MTP layers inside the target
    assert fn({"method": "mtp", "model": "t"}, "t") == 0.0
    assert fn({"method": "mtp", "model": "org/assistant"}, "t") == 3.0         # (22) a separate MTP assistant counts
    assert fn({"method": "draft_model", "model": "t"}, "t") == 3.0             # (23) a same-ID standalone draft counts
    assert fn({"method": "mtp", "model": "t", "revision": "r2"}, "t") == 3.0   # (23) a different revision is separate
    assert fn({"method": "eagle3", "model": "org/draft"}, "t") == 3.0          # (25) nested ckpt/ and .bin not counted
    assert fn({"method": "draft_model", "model": "org/missing"}, "t") is None
    (tmp_path / "model.safetensors").write_bytes(b"\0" * (1 << 20))
    (tmp_path / "pytorch_model.bin").write_bytes(b"\0" * (1 << 20))          # (4) alternate format not double-counted
    (tmp_path / "sub").mkdir(); (tmp_path / "sub" / "model.safetensors").write_bytes(b"\0" * (1 << 20))
    (tmp_path / "config.json").write_text("{}")
    assert fn({"method": "draft_model", "model": str(tmp_path)}, "t") == pytest.approx(1 / 1024)


def test_load_vllm_wiring():
    fn = _load_vllm()
    assert "speculative_config" in [a.arg for a in fn.args.args]
    src = ast.unparse(fn)
    resolve = src.find("user_speculative_config = None if return_args else _vllm_speculative_config(speculative_config, training=training)")
    replay = src.find("_routing_replay_will_drop_speculation(config, user_speculative_config, enable_prefix_caching)", max(resolve, 0))
    async_refusal = src.find("cannot mask this model's padded vocabulary", max(replay, 0))
    budget = src.find("draft_gb = _speculative_draft_gb(user_speculative_config, model_name)", max(async_refusal, 0))
    sizing = src.find("approximate_vllm_memory_usage(", max(budget, 0))
    counted = src.find("extra_weight_bytes=int(draft_gb * 1024 ** 3)", max(sizing, 0))
    tiers = src.find("if memory_left_for_kv_cache_gb <= 2:", max(counted, 0))
    bnb = src.find("LoadConfig(load_format='auto')", max(tiers, 0))
    assign = src.find("engine_args['speculative_config'] = user_speculative_config", max(bnb, 0))
    depth = src.find("_resolve_speculative_depth(engine_args, AsyncEngineArgs if use_async else EngineArgs)", max(assign, 0))
    fit = src.find("_fit_max_num_seqs_to_speculation(", max(depth, 0))
    banner = src.find("_redact_speculative_config(user_speculative_config)", max(fit, 0))
    arg_filter = src.find("good_keys = inspect.signature(AsyncEngineArgs", max(banner, 0))
    assert -1 < resolve < replay < async_refusal < budget < sizing < counted < tiers < bnb < assign < depth < fit < banner < arg_filter
    assert "speculative=engine_args.get('speculative_config', None) is not None" in src
    assert "engine_args.get('speculative_config') if enable_lora else None" in src
    assert "speculative_config = {user_speculative_config}" not in src


def test_replay_arbitration_reads_the_release_table_and_drops_adaptive_verification():
    src = ast.unparse(next(n for n in _tree().body if isinstance(n, ast.FunctionDef) and n.name == "_routing_replay_model_runner"))
    assert "_mrv2_unsupported_speculative_methods()" in src and "_MRV2_UNSUPPORTED_SPECULATIVE_METHODS" not in src
    assert "'enable_adaptive_verification': False" in src


def test_speculative_padded_vocab_never_silently_sampleable():
    # (19) speculation without an in-process model to patch raises instead of leaving the rows sampleable.
    src = ast.unparse(next(n for n in _tree().body if isinstance(n, ast.FunctionDef) and n.name == "suppress_vllm_padded_vocab"))
    assert "stay sampleable. Drop speculative_config" in src
    assert "alignment rows stay sampleable.')\n        return llm" not in src


def test_real_vllm_methods_if_installed():
    pytest.importorskip("vllm.config.speculative")
    nodes = [n for n in _tree().body if isinstance(n, ast.FunctionDef) and n.name == "_vllm_speculative_methods"]
    out = {}
    exec(compile(ast.Module(body = nodes, type_ignores = []), "vllm_utils_real", "exec"), out)
    methods = out["_vllm_speculative_methods"]()
    assert {"ngram", "draft_model", "eagle3"} <= methods


def test_draft_weights_shrink_the_capacity_estimate(monkeypatch):
    # Counting the draft up front shrinks max_num_batched_tokens and KV budget, not KV alone.
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

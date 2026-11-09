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

"""ReplaySSM gating in load_vllm (default on, UNSLOTH_VLLM_REPLAYSSM=0 off). AST-level: load_vllm needs a GPU."""

import ast
import importlib.util
import os
import inspect
import pathlib
from types import SimpleNamespace

import pytest

VLLM_UTILS = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo" / "vllm_utils.py"
TREE = ast.parse(VLLM_UTILS.read_text(encoding = "utf-8-sig"))
CONSTANTS = ("_REPLAYSSM_MAMBA2_MODELS", "_REPLAYSSM_GDN_MODEL_TYPES", "_REPLAYSSM_MODEL_TYPES")
HELPERS = ("_vllm_replayssm_mode", "_vllm_replayssm_skip_reason", "_is_vllm_replayssm_refusal")


def _helpers():
    nodes = [
        n for n in TREE.body
        if (isinstance(n, ast.FunctionDef) and n.name in HELPERS)
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in CONSTANTS for t in n.targets))
    ]
    assert len(nodes) == len(HELPERS) + len(CONSTANTS)
    ns = {"os": os}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(VLLM_UTILS), "exec"), ns)
    return ns


NS = _helpers()
KEYS = {"model", "use_replayssm"}


def _config(model_type, nested = False):
    inner = SimpleNamespace(model_type = model_type)
    return SimpleNamespace(model_type = "wrapper", text_config = inner) if nested else inner


@pytest.mark.parametrize("nested", [False, True])
def test_nemotron_h_on_cuda_is_requested(nested):
    assert NS["_vllm_replayssm_skip_reason"](_config("nemotron_h", nested), KEYS, "cuda") is None


@pytest.mark.parametrize("config, keys, device, why", [
    (_config("lfm2"), KEYS, "cuda", "model type"),
    (_config("llama"), KEYS, "cuda", "model type"),
    (_config("nemotron_h"), KEYS, "hip", "CUDA only"),
    (_config("nemotron_h"), KEYS, "xpu", "CUDA only"),
    (_config("nemotron_h"), {"model"}, "cuda", "no `use_replayssm`"),
])
def test_skip_reasons(config, keys, device, why):
    assert why in NS["_vllm_replayssm_skip_reason"](config, keys, device)


@pytest.mark.parametrize("value, mode", [
    (None, "auto"), ("", "auto"), ("auto", "auto"),
    ("0", "off"), ("false", "off"), ("OFF", "off"), ("no", "off"),
    ("1", "on"), ("true", "on"), ("On", "on"), ("yes", "on"),
])
def test_mode_parsing(monkeypatch, value, mode):
    if value is None:
        monkeypatch.delenv("UNSLOTH_VLLM_REPLAYSSM", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_VLLM_REPLAYSSM", value)
    assert NS["_vllm_replayssm_mode"]() == mode


def test_refusal_matcher_rejects_unrelated_errors():
    is_refusal = NS["_is_vllm_replayssm_refusal"]
    assert not is_refusal("CUDA out of memory. Tried to allocate 2.00 GiB")
    assert not is_refusal("Free memory on device cuda:0 (1.2/80 GiB) on startup is less than desired")


def _load_vllm():
    return next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "load_vllm")


def test_requested_only_for_engines_load_vllm_builds():
    # return_args callers (synthetic data `vllm serve`) never get the flag without the runner env.
    fn = _load_vllm()
    return_args_line = next(
        n.lineno for n in ast.walk(fn)
        if isinstance(n, ast.If) and getattr(n.test, "id", "") == "return_args"
    )
    request_lines = [
        n.lineno for n in ast.walk(fn)
        if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Subscript) and getattr(t.slice, "value", None) == "use_replayssm"
            for t in n.targets
        )
    ]
    assert request_lines and min(request_lines) > return_args_line
    src = ast.get_source_segment(VLLM_UTILS.read_text(encoding = "utf-8-sig"), fn)
    assert "_vllm_replayssm_mode()" in src
    assert '"VLLM_USE_V2_MODEL_RUNNER" not in os.environ' in src
    assert 'replayssm_mode == "auto" and replayssm_type in _REPLAYSSM_GDN_MODEL_TYPES' in src


# Drift guards against the installed vLLM, run whenever it has the engine arg load_vllm keys on.
def _vllm_root():
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        return None
    return pathlib.Path(list(spec.submodule_search_locations)[0])


def _vllm_has_replayssm():
    if _vllm_root() is None:
        return False
    try:
        from vllm.engine.arg_utils import EngineArgs
    except Exception:
        return False
    return "use_replayssm" in inspect.signature(EngineArgs).parameters


needs_replayssm_vllm = pytest.mark.skipif(
    not _vllm_has_replayssm(),
    reason = "needs a vLLM with use_replayssm (vllm-project/vllm#48018); load_vllm never requests it on older vLLM",
)


def _vllm_ast(*parts):
    return ast.parse(_vllm_root().joinpath(*parts).read_text(encoding = "utf-8"))


@needs_replayssm_vllm
def test_refusal_matcher_matches_vllm_replayssm_errors():
    messages = []
    for node in ast.walk(_vllm_ast("config", "vllm.py")):
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            text = "".join(
                c.value for a in node.exc.args for c in ast.walk(a)
                if isinstance(c, ast.Constant) and isinstance(c.value, str)
            )
            if "replayssm" in text.lower():
                messages.append(text)
    assert messages, "vLLM's ReplaySSM config validation moved; re-check _is_vllm_replayssm_refusal"
    assert any("Model Runner V1" in m for m in messages)
    assert all(NS["_is_vllm_replayssm_refusal"](m) for m in messages)


@needs_replayssm_vllm
def test_nemotron_h_declares_supports_replayssm():
    tree = _vllm_ast("model_executor", "models", "nemotron_h.py")
    bases = {
        getattr(b, "id", getattr(b, "attr", None))
        for n in ast.walk(tree) if isinstance(n, ast.ClassDef) for b in n.bases
    }
    assert "SupportsReplaySSM" in bases


@needs_replayssm_vllm
@pytest.mark.parametrize("model_type", sorted(NS["_REPLAYSSM_MAMBA2_MODELS"]))
def test_widened_models_decode_through_mamba_mixer2(model_type):
    # The widening is only sound while the layers are MambaMixer2 built with the cache config.
    module_name, class_name, _ = NS["_REPLAYSSM_MAMBA2_MODELS"][model_type]
    tree = _vllm_ast("model_executor", "models", f"{module_name}.py")
    mixers = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "MambaMixer2"
    ]
    assert mixers and all(any(k.arg == "cache_config" for k in m.keywords) for m in mixers)
    assert any(isinstance(n, ast.ClassDef) and n.name == class_name for n in ast.walk(tree))


def _ring_helpers_present():
    try:
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
    except Exception:
        return False
    return hasattr(MambaStateShapeCalculator, "append_replayssm_ring")


needs_ring_helpers = pytest.mark.skipif(
    not _ring_helpers_present(),
    reason = "needs a vLLM with the ReplaySSM ring helpers (vllm-project/vllm#48018)",
)

_HF_CONFIGS = {
    "falcon_h1": "FalconH1Config",
    "granitemoehybrid": "GraniteMoeHybridConfig",
    "zamba2": "Zamba2Config",
}


def _fake_vllm_config(hf_config, use_replayssm):
    import torch
    from vllm.config.mamba import MambaBackendEnum
    return SimpleNamespace(
        cache_config = SimpleNamespace(
            use_replayssm = use_replayssm, replayssm_buffer_len = 16,
            mamba_cache_dtype = "auto", mamba_ssm_cache_dtype = "auto",
        ),
        model_config = SimpleNamespace(hf_config = hf_config, hf_text_config = hf_config, dtype = torch.bfloat16),
        parallel_config = SimpleNamespace(tensor_parallel_size = 1),
        mamba_config = SimpleNamespace(backend = MambaBackendEnum.TRITON),
        num_speculative_tokens = 0,
        speculative_config = None,
    )


@needs_ring_helpers
@pytest.mark.parametrize("model_type", sorted(_HF_CONFIGS))
def test_mamba2_hybrid_hooks_count_the_ring(model_type):
    import importlib
    import torch
    import transformers
    from unsloth_zoo.vllm_utils import _REPLAYSSM_MAMBA2_MODELS, _patch_vllm_replayssm_mamba2
    module_name, class_name, _ = _REPLAYSSM_MAMBA2_MODELS[model_type]
    model_cls = getattr(importlib.import_module(f"vllm.model_executor.models.{module_name}"), class_name)
    hf_config = getattr(transformers, _HF_CONFIGS[model_type])()
    on, off = _fake_vllm_config(hf_config, True), _fake_vllm_config(hf_config, False)
    stock = (model_cls.get_mamba_state_shape_from_config(off), model_cls.get_mamba_state_dtype_from_config(off))

    assert _patch_vllm_replayssm_mamba2(model_type)
    assert _patch_vllm_replayssm_mamba2(model_type)  # idempotent: no second wrap

    shapes = model_cls.get_mamba_state_shape_from_config(on)
    dtypes = model_cls.get_mamba_state_dtype_from_config(on)
    assert len(shapes) == len(dtypes) == 5
    assert shapes[:2] == stock[0] and dtypes[:2] == stock[1]
    local_heads, head_dim, _ = stock[0][1]
    assert shapes[2] == (local_heads, 16, head_dim) and shapes[3] == (local_heads, 16)
    assert dtypes[3] == torch.float32
    assert (model_cls.get_mamba_state_shape_from_config(off), model_cls.get_mamba_state_dtype_from_config(off)) == stock


@needs_ring_helpers
def test_capability_widens_only_patched_architectures():
    from vllm.config.model import ModelConfig
    from unsloth_zoo.vllm_utils import _patch_vllm_replayssm_mamba2
    assert _patch_vllm_replayssm_mamba2("granitemoehybrid")
    fget = ModelConfig.supports_replayssm.fget

    def fake(arch, native):
        return SimpleNamespace(architecture = arch, _model_info = SimpleNamespace(supports_replayssm = native))

    assert fget(fake("GraniteMoeHybridForCausalLM", False))
    assert fget(fake("NemotronHForCausalLM", True))
    assert not fget(fake("LlamaForCausalLM", False))
    assert not fget(fake("Qwen3_5ForConditionalGeneration", False))


def test_refusal_matcher_matches_recoverssm_under_speculation():
    # vLLM 0.29 config/vllm.py validate_mamba_cached_kernel: use_replayssm with speculative tokens becomes RecoverSSM;
    # that is refused for all but Kimi KDA (e.g. Nemotron-H with suffix decoding).
    assert NS["_is_vllm_replayssm_refusal"]("RecoverSSM is only supported for Kimi-K3 KDA")
    assert NS["_is_vllm_replayssm_refusal"]("RecoverSSM currently requires pipeline_parallel_size=1")

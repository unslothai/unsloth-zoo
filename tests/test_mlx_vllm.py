import importlib.util
import os
import sys
import types

import pytest

if importlib.util.find_spec("mlx") is None:
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()

from unsloth_zoo.mlx import vllm_utils  # noqa: E402


@pytest.mark.parametrize(
    "fast_inference, env, error, installed, expected",
    [
        (False, "1", None, True, False),
        (True, "", None, True, True),
        (True, "", "not installed", False, False),
        (True, "", "failed to import", True, False),
        (True, "0", None, True, False),
        (True, "1", None, True, True),
    ],
)
def test_use_vllm_prefers_a_working_vllm_metal_and_honours_the_override(
    monkeypatch, recwarn, fast_inference, env, error, installed, expected,
):
    monkeypatch.setenv(vllm_utils.VLLM_ENV, env)
    monkeypatch.setattr(vllm_utils, "_metal_platform_error", lambda: error)
    monkeypatch.setattr(vllm_utils, "_installed", lambda: installed)
    assert vllm_utils.use_vllm(fast_inference) is expected
    # Only an installation that does not work is worth a warning; a missing one is the default.
    assert [str(w.message) for w in recwarn] == (["Unsloth MLX: fast_inference keeps MLX generation because failed to import."] if error == "failed to import" else [])


def test_use_vllm_refuses_a_forced_engine_that_does_not_work_and_unknown_values(monkeypatch):
    monkeypatch.setattr(vllm_utils, "_metal_platform_error", lambda: "vLLM failed to import (boom)")
    monkeypatch.setenv(vllm_utils.VLLM_ENV, "1")
    with pytest.raises(ImportError, match=r"failed to import \(boom\).*vllm-metal"):
        vllm_utils.use_vllm(True)
    monkeypatch.setenv(vllm_utils.VLLM_ENV, "yes")
    with pytest.raises(ValueError, match="UNSLOTH_MLX_VLLM"):
        vllm_utils.use_vllm(True)


@pytest.mark.parametrize(
    "installed, platform_module, expected",
    [
        (False, "vllm_metal.platform", "not installed"),
        (True, None, "failed to import"),
        (True, "no LLM", "failed to import"),
        (True, "vllm.platforms.cpu", "not vllm-metal"),
        (True, "vllm_metal.platform", None),
    ],
)
def test_metal_platform_error_needs_vllm_to_import_and_resolve_metal(monkeypatch, installed, platform_module, expected):
    if platform_module is None:
        monkeypatch.setitem(sys.modules, "vllm.platforms", None)
    elif platform_module == "no LLM":
        # vLLM resolves its platform, but its lazily imported LLM entry point fails.
        _fake_vllm(monkeypatch, "vllm_metal.platform")
        monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace())
    else:
        _fake_vllm(monkeypatch, platform_module)
    monkeypatch.setattr(vllm_utils, "_installed", lambda: installed)
    error = vllm_utils._metal_platform_error()
    assert error is None if expected is None else expected in error
    assert vllm_utils.vllm_metal_available() is (expected is None)


def _fake_vllm(monkeypatch, platform_module):
    platform = type("MetalPlatform", (), {"__module__": platform_module})()
    created = []

    class LLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.removed = []
            self.llm_engine = types.SimpleNamespace(remove_lora=self.removed.append)
            created.append(self)

        def generate(self, *args, **kwargs):
            return ("generated", args, kwargs)

    class LoRARequest:
        def __init__(self, lora_name, lora_int_id, lora_path):
            self.lora_name, self.lora_int_id, self.lora_path = lora_name, lora_int_id, lora_path

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=LLM))
    monkeypatch.setitem(sys.modules, "vllm.platforms", types.SimpleNamespace(current_platform=platform))
    monkeypatch.setitem(sys.modules, "vllm.lora", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.lora.request", types.SimpleNamespace(LoRARequest=LoRARequest))
    monkeypatch.setattr(vllm_utils, "_installed", lambda: True)
    return created


def test_load_vllm_starts_an_in_process_metal_engine_on_the_loaded_checkpoint(monkeypatch):
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    created = _fake_vllm(monkeypatch, "vllm_metal.platform")
    model = types.SimpleNamespace(_src_path="/snap/qwen", _hf_repo="Qwen/Qwen3-0.6B")
    engine = vllm_utils.load_vllm(model, max_seq_length=512, gpu_memory_utilization=0.3, max_lora_rank=32, seed=7)
    assert engine.kwargs == dict(
        model="/snap/qwen", max_model_len=512, gpu_memory_utilization=0.3, seed=7,
        enable_lora=True, max_lora_rank=32, max_loras=1,
    )
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    full = vllm_utils.load_vllm(model, max_seq_length=512, enable_lora=False)
    assert "enable_lora" not in full.kwargs and len(created) == 2
    vllm_utils.attach_vllm_engine(model, engine)
    assert model.vllm_engine is engine and model.fast_generate("hi")[0] == "generated"


def _tiny(seed, layers=2, dtype="bfloat16"):
    import mlx.core as mx
    qwen3 = pytest.importorskip("mlx_lm.models.qwen3")
    mx.random.seed(seed)
    model = qwen3.Model(qwen3.ModelArgs("qwen3", 64, layers, 128, 4, 1e-6, 128, 2, 128, 16, True, 1e4))
    return model.set_dtype(getattr(mx, dtype)) or model


@pytest.mark.parametrize("layers, dtype, multiprocess, loads", [(2, "bfloat16", "0", [True]), (1, "bfloat16", "0", [True, False]), (2, "float16", "0", [True, False]), (2, "bfloat16", "1", [False])])
def test_an_in_process_engine_shares_only_matching_weights(monkeypatch, layers, dtype, multiprocess, loads):
    from mlx.utils import tree_flatten
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", multiprocess)
    _fake_vllm(monkeypatch, "vllm_metal.platform")
    calls, model = [], _tiny(0, layers, dtype)
    life = types.SimpleNamespace(mlx_lm_load_model=lambda path, lazy=False: (calls.append(lazy), (_tiny(1), {}))[1], mlx_vlm_load_model=lambda path: _tiny(2))
    v1, original, model._src_path = types.SimpleNamespace(model_lifecycle=life), life.mlx_lm_load_model, "/snap"
    for item in (("vllm_metal", types.SimpleNamespace(v1=v1)), ("vllm_metal.v1", v1), ("vllm_metal.v1.model_lifecycle", life)):
        monkeypatch.setitem(sys.modules, *item)
    monkeypatch.setattr(sys.modules["vllm"], "LLM", lambda **kw: types.SimpleNamespace(models=(life.mlx_lm_load_model(kw["model"])[0], life.mlx_vlm_load_model(kw["model"]))))
    engine, vision = vllm_utils.load_vllm(model, max_seq_length=64).models
    aliased = [array is dict(tree_flatten(model.parameters())).get(name) for name, array in tree_flatten(engine.parameters())]
    assert calls == loads and life.mlx_lm_load_model is original and all(aliased) == (loads == [True]) == any(aliased)
    assert all(getattr(m, "_unsloth_no_checkpoint", False) == (multiprocess == "0") for x in (engine, vision) for _, m in x.named_modules())


def test_an_in_process_engine_reads_the_live_lora_from_memory(monkeypatch):
    import mlx.core as mx
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from unsloth_zoo.mlx.utils import iter_mlx_lora_modules
    worker = types.SimpleNamespace(load_peft_adapter=lambda path, *, lora_id, **kwargs: ("disk", path))
    lora = types.SimpleNamespace(worker_manager=worker, peft_loader=types.SimpleNamespace(LoadedLoRA=lambda **kwargs: kwargs, LoRALayerWeightsMLX=lambda *args: args))
    for item in (("vllm_metal", types.SimpleNamespace(v1=types.SimpleNamespace(lora=lora))), ("vllm_metal.v1.lora", lora), ("vllm_metal.v1.lora.peft_loader", lora.peft_loader)):
        monkeypatch.setitem(sys.modules, *item)
    _fake_vllm(monkeypatch, "vllm_metal.platform")
    model = _tiny(0)
    linear_to_lora_layers(model, 2, {"rank": 4, "scale": 2.0, "dropout": 0.0})
    model.vllm_engine = type("LLM", (), {"llm_engine": types.SimpleNamespace(remove_lora=lambda i: None, add_request=lambda *a, lora_request: "added")})()  # weak-referenceable, as vLLM's is
    model.vllm_engine._unsloth_in_memory_lora = vllm_utils._serve_in_memory_loras(model.vllm_engine)
    request, snapshot = vllm_utils.load_lora(model, load_tensors=True), {n: (m.lora_a.T, m.lora_b.T) for n, m in iter_mlx_lora_modules(model)}
    for _, module in iter_mlx_lora_modules(model):  # training goes on after the snapshot
        module.lora_a = module.lora_a + 1
    loaded = worker.load_peft_adapter(request.lora_path, lora_id=3)
    assert loaded["rank"] == 4 and loaded["lora_id"] == 3 and set(loaded["weights"]) == set(snapshot) and not os.path.exists(request.lora_path)
    assert all(mx.array_equal(w[2], snapshot[n][0]) and mx.array_equal(w[3], snapshot[n][1]) and w[4] == 2.0 for n, w in loaded["weights"].items())
    with pytest.raises(ValueError, match="max_lora_rank 2"):
        worker.load_peft_adapter(request.lora_path, lora_id=3, lora_config=types.SimpleNamespace(max_lora_rank=2))
    newer = vllm_utils.load_lora(model, load_tensors=True)  # replaces the unreleased live adapter
    add_request = model.vllm_engine.llm_engine.add_request
    assert add_request("1", "p", None, lora_request=newer) == add_request("2", "p", None, lora_request=None) == "added"
    with pytest.raises(ValueError, match="replaced by a newer"):
        add_request("3", "p", None, lora_request=request)
    vllm_utils.release_lora(model, newer)
    assert len(vllm_utils._IN_MEMORY_LORAS) == 0 and model.vllm_engine._unsloth_live_lora is None and worker.load_peft_adapter("/saved", lora_id=4) == ("disk", "/saved")
    vllm_utils.load_lora(model, load_tensors=True)
    model.vllm_engine = None  # an engine dropped without a release takes its live adapter with it
    assert len(vllm_utils._IN_MEMORY_LORAS) == 0


def test_load_vllm_refuses_a_non_metal_vllm(monkeypatch):
    _fake_vllm(monkeypatch, "vllm.platforms.cpu")
    with pytest.raises(ImportError, match="not vllm-metal"):
        vllm_utils.load_vllm(types.SimpleNamespace(_src_path="/snap"), max_seq_length=64)


class _AdapterModel:
    def __init__(self):
        self.version = 0

    def save_lora_adapters(self, path, adapter_format):
        assert adapter_format == "peft" and not os.path.exists(path)
        os.makedirs(path)
        self.version += 1
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            with open(os.path.join(path, name), "w") as handle:
                handle.write(str(self.version))


def test_save_lora_writes_peft_and_replaces_an_earlier_save(tmp_path):
    model, target = _AdapterModel(), tmp_path / "grpo_saved_lora"
    vllm_utils.save_lora(model, target)
    vllm_utils.save_lora(model, target)
    assert (target / "adapter_model.safetensors").read_text() == (target / "adapter_config.json").read_text() == "2"
    assert sorted(os.listdir(tmp_path)) == ["grpo_saved_lora"]
    mlx_dir = tmp_path / "mlx"
    mlx_dir.mkdir()
    (mlx_dir / "adapters.safetensors").write_text("")
    with pytest.raises(ValueError, match="MLX-format adapter"):
        vllm_utils.save_lora(model, mlx_dir)


def test_load_lora_builds_path_requests_and_releases_live_snapshots(monkeypatch, tmp_path):
    _fake_vllm(monkeypatch, "vllm_metal.platform")
    model = _AdapterModel()
    with pytest.raises(RuntimeError, match="vLLM engine"):
        vllm_utils.load_lora(model, tmp_path)
    model.vllm_engine = sys.modules["vllm"].LLM()
    saved = tmp_path / "grpo_saved_lora"
    vllm_utils.save_lora(model, saved)
    other = _AdapterModel()
    other.vllm_engine = model.vllm_engine
    first, second = vllm_utils.load_lora(model, saved), vllm_utils.load_lora(other, saved)
    assert first.lora_path == second.lora_path == str(saved)
    # vLLM reserves lora_int_id 0 for "no adapter".
    assert first.lora_int_id >= 1 and second.lora_int_id == first.lora_int_id + 1
    assert second.lora_name == str(second.lora_int_id)
    pinned = vllm_utils.load_lora(model, saved, lora_request_id=1000)
    assert pinned.lora_int_id == 1000
    assert vllm_utils.load_lora(model, saved).lora_int_id == second.lora_int_id + 2
    converted = []
    monkeypatch.setattr(vllm_utils, "convert_mlx_dir_to_peft", lambda src, dst: converted.append(src) or os.makedirs(dst))
    mlx_dir = tmp_path / "mlx"
    mlx_dir.mkdir()
    (mlx_dir / "adapters.safetensors").write_text("")
    assert vllm_utils.load_lora(model, mlx_dir).lora_path != str(mlx_dir) and converted == [str(mlx_dir)]
    live = vllm_utils.load_lora(model, load_tensors=True)
    later = vllm_utils.load_lora(model, load_tensors=True)
    assert later.lora_path != live.lora_path
    read = lambda request: open(os.path.join(request.lora_path, "adapter_model.safetensors")).read()
    assert (read(live), read(later)) == (str(model.version - 1), str(model.version))
    vllm_utils.release_lora(model, live)
    assert model.vllm_engine.removed == [live.lora_int_id] and not os.path.exists(live.lora_path)
    vllm_utils.release_lora(model, first)
    assert saved.exists()

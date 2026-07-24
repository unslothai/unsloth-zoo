"""PEFT <-> MLX adapter interop contracts, on genuine peft-generated files."""

import json
import os
import sys

import numpy as np
import pytest
from pathlib import Path

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")
nn = pytest.importorskip("mlx.nn")
load_model = pytest.importorskip("mlx_lm.utils").load_model
st_load_file = pytest.importorskip("safetensors.torch").load_file

from unsloth_zoo.mlx.utils import (
    attach_and_bind_peft_adapter,
    detect_adapter_format,
    normalize_peft_adapter_config,
)
from unsloth_zoo.saving_utils import (
    MLX_WEIGHTS_FILE,
    PEFT_WEIGHTS_FILE,
    convert_mlx_dir_to_peft,
    convert_peft_dir_to_mlx,
    _resolve_pattern,
)

VOCAB, HIDDEN, LAYERS = 256, 64, 2
IDS = [1, 5, 9, 13]


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-llama-base")
    config = transformers.LlamaConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=128,
        num_hidden_layers=LAYERS, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    transformers.LlamaForCausalLM(config).to(torch.float32).save_pretrained(
        path, safe_serialization=True
    )
    transformers.AutoTokenizer.from_pretrained(
        "hf-internal-testing/llama-tokenizer"
    ).save_pretrained(path)
    return str(path)


def _make_peft_adapter(base_dir, out, dtype=torch.float32, **lora_kwargs):
    model = transformers.LlamaForCausalLM.from_pretrained(base_dir, dtype=dtype)
    kwargs = dict(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
    kwargs.update(lora_kwargs)
    wrapped = peft.get_peft_model(model, peft.LoraConfig(**kwargs))
    torch.manual_seed(7)
    with torch.no_grad():
        for name, param in wrapped.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn_like(param) * 0.05)
            if "lora_" in name:
                # peft autocasts adapter params to fp32; force the dtype so
                # bf16 runs exercise real bf16 tensors.
                param.data = param.data.to(dtype)
    wrapped.save_pretrained(out, save_embedding_layers=False)
    cfg_path = os.path.join(out, "adapter_config.json")
    cfg = json.load(open(cfg_path))
    cfg["base_model_name_or_path"] = base_dir
    json.dump(cfg, open(cfg_path, "w"))
    return wrapped, cfg


def _peft_logits(wrapped, ids=IDS):
    with torch.no_grad():
        return wrapped(torch.tensor([ids])).logits[0, -1].float().numpy()


def _mlx_logits(model, ids=IDS):
    out = model(mx.array([ids]))
    logits = out.logits if hasattr(out, "logits") else out
    return np.array(logits[0, -1].astype(mx.float32))


def test_format_detection_and_fresh_destination(tmp_path, base_dir):
    d = tmp_path / "a"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        detect_adapter_format(str(d))
    (d / PEFT_WEIGHTS_FILE).write_bytes(b"")
    (d / MLX_WEIGHTS_FILE).write_bytes(b"")
    with pytest.raises(ValueError, match="both"):
        detect_adapter_format(str(d))
    peft_dir = str(tmp_path / "peft")
    _make_peft_adapter(base_dir, peft_dir)
    (tmp_path / "dst").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        convert_peft_dir_to_mlx(peft_dir, str(tmp_path / "dst"), {"num_hidden_layers": LAYERS})


@pytest.mark.parametrize("field,value,match", [
    ("use_dora", True, "DoRA"),
    ("modules_to_save", ["lm_head"], "modules_to_save"),
    ("target_parameters", ["experts.gate_up_proj"], "expert-parameter"),
    ("alora_invocation_tokens", [1, 2], "aLoRA"),
    ("bias", "all", "bias"),
    ("corda_config", {"x": 1}, "corda_config"),
    ("layer_replication", [[0, 1], [0, 1]], "layer_replication"),
    ("use_qalora", True, "QALoRA"),
    ("lora_bias", True, "lora_bias"),
    ("init_lora_weights", "pissa_niter_4", "mutated"),
    ("init_lora_weights", "OLoRA", "mutated"),
    ("loftq_config", {"bits": 4}, "LoftQ"),
])
def test_normalize_rejects_by_name(field, value, match):
    cfg = {"peft_type": "LORA", "r": 8, "lora_alpha": 16, field: value}
    with pytest.raises(ValueError, match=match):
        normalize_peft_adapter_config(cfg)


def test_pattern_resolution_and_eva_acceptance():
    normalize_peft_adapter_config({"peft_type": "LORA", "r": 8, "init_lora_weights": "eva", "eva_config": {"rho": 1.0}})
    path = "model.layers.0.self_attn.q_proj"
    assert _resolve_pattern({"q_proj": 4}, path) == 4
    assert _resolve_pattern({"layers.0": 4}, path) is None  # end-anchored
    assert _resolve_pattern({"layers.0.self_attn.q_proj": 2, "q_proj": 4}, path) == 2
    with pytest.raises(ValueError, match="regular expression"):
        _resolve_pattern({"(": 1}, "x")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_roundtrip_bitwise_with_revision_and_nested_layers(tmp_path, base_dir, dtype):
    peft_dir = str(tmp_path / "peft")
    _, cfg = _make_peft_adapter(base_dir, peft_dir, dtype=dtype)
    cfg["revision"] = "deadbeef"
    json.dump(cfg, open(os.path.join(peft_dir, "adapter_config.json"), "w"))
    mlx_dir, back_dir = str(tmp_path / "mlx"), str(tmp_path / "back")
    # Path-typed destination must keep working (os.PathLike contract).
    convert_peft_dir_to_mlx(
        peft_dir, tmp_path / "mlx", {"text_config": {"num_hidden_layers": LAYERS}}
    )
    mlx_cfg = json.load(open(os.path.join(mlx_dir, "adapter_config.json")))
    assert mlx_cfg["num_layers"] == LAYERS
    assert mlx_cfg["base_model_revision"] == "deadbeef"
    assert mlx_cfg["lora_parameters"]["keys"] == mlx_cfg["unsloth_mlx_lora_module_paths"]
    convert_mlx_dir_to_peft(mlx_dir, back_dir)

    orig = st_load_file(os.path.join(peft_dir, PEFT_WEIGHTS_FILE))
    back = st_load_file(os.path.join(back_dir, PEFT_WEIGHTS_FILE))
    assert set(orig) == set(back)
    for key in orig:
        assert orig[key].dtype == back[key].dtype and torch.equal(orig[key], back[key]), key
    back_cfg = json.load(open(os.path.join(back_dir, "adapter_config.json")))
    assert back_cfg["r"] == 8 and back_cfg["lora_alpha"] == pytest.approx(16)
    assert back_cfg["revision"] == "deadbeef"
    targets = back_cfg["target_modules"]
    assert len(targets) == 2 * LAYERS and all(
        t.startswith("model.layers.") for t in targets
    )
    if dtype is torch.float32:  # exported artifact must BIND in peft
        base = transformers.LlamaForCausalLM.from_pretrained(
            base_dir, dtype=torch.float32
        )
        back_cfg.pop("revision")
        json.dump(back_cfg, open(os.path.join(back_dir, "adapter_config.json"), "w"))
        reloaded = peft.PeftModel.from_pretrained(base, back_dir)
        orig_model = transformers.LlamaForCausalLM.from_pretrained(
            base_dir, dtype=torch.float32
        )
        ref = peft.PeftModel.from_pretrained(orig_model, peft_dir)
        np.testing.assert_allclose(
            _peft_logits(reloaded), _peft_logits(ref), atol=1e-5,
        )


def test_attach_matches_peft_forward_with_patterns(tmp_path, base_dir):
    peft_dir = str(tmp_path / "peft")
    wrapped, cfg = _make_peft_adapter(
        base_dir, peft_dir,
        rank_pattern={"q_proj": 4}, alpha_pattern={"v_proj": 8}, use_rslora=True,
        lora_dropout=0.1,
    )
    wrapped.eval()
    model, _ = load_model(Path(base_dir))
    assert attach_and_bind_peft_adapter(
        model, peft_dir, normalize_peft_adapter_config(cfg)
    ) == 2 * LAYERS
    modules = dict(model.named_modules())
    assert modules["model.layers.0.self_attn.q_proj"].lora_a.shape[-1] == 4
    np.testing.assert_allclose(_mlx_logits(model), _peft_logits(wrapped), atol=2e-4)
    assert model._unsloth_lora_module_scales[
        "model.layers.0.self_attn.v_proj"
    ] == pytest.approx(8 / (8 ** 0.5))
    # Eval-mode host + nonzero lora_dropout must stay deterministic.
    np.testing.assert_array_equal(_mlx_logits(model), _mlx_logits(model))


@pytest.mark.parametrize("key,shape,match", [
    ("base_model.model.language_model.model.layers.0.self_attn.q_proj.lora_A.weight",
     (8, HIDDEN), "language_model"),
    ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
     (HIDDEN + 8, 8), "out-dim"),
])
def test_attach_rejects_bad_tensors(tmp_path, base_dir, key, shape, match):
    peft_dir = str(tmp_path / "peft")
    _, cfg = _make_peft_adapter(base_dir, peft_dir)

    def mutate(tensors):
        tensors[key] = torch.zeros(*shape)
        if "language_model" in key:  # give the alien path a complete pair
            tensors[key.replace("lora_A", "lora_B")] = torch.zeros(HIDDEN, 8)

    from safetensors.torch import save_file
    wpath = os.path.join(peft_dir, PEFT_WEIGHTS_FILE)
    tensors = st_load_file(wpath)
    mutate(tensors)
    save_file(tensors, wpath)
    model, _ = load_model(Path(base_dir))
    with pytest.raises(ValueError, match=match):
        attach_and_bind_peft_adapter(
            model, peft_dir, normalize_peft_adapter_config(cfg)
        )


def test_mlx_to_peft_rejects_out_of_layer_paths(tmp_path, base_dir):
    peft_dir = str(tmp_path / "peft")
    _make_peft_adapter(base_dir, peft_dir)
    mlx_dir = str(tmp_path / "mlx")
    convert_peft_dir_to_mlx(
        peft_dir, mlx_dir,
        json.load(open(os.path.join(base_dir, "config.json"))),
    )
    tensors = dict(mx.load(os.path.join(mlx_dir, MLX_WEIGHTS_FILE)))
    tensors["model.embed_tokens.lora_a"] = mx.zeros((VOCAB, 8))
    tensors["model.embed_tokens.lora_b"] = mx.zeros((8, HIDDEN))
    mx.save_safetensors(os.path.join(mlx_dir, MLX_WEIGHTS_FILE), tensors)
    with pytest.raises(ValueError, match="transformer stack"):
        convert_mlx_dir_to_peft(mlx_dir, str(tmp_path / "back"))
    with pytest.raises(ValueError, match="embedding LoRA"):
        convert_mlx_dir_to_peft(
            mlx_dir, str(tmp_path / "b2"),
            module_types={"model.embed_tokens": "embedding"},
        )
    out = convert_mlx_dir_to_peft(
        mlx_dir, str(tmp_path / "b3"),
        module_types={"model.embed_tokens": "linear"},
    )
    assert any("embed_tokens.lora_A" in k
               for k in st_load_file(os.path.join(out, PEFT_WEIGHTS_FILE)))
    tensors["model.layers.0.mlp.experts.lora_a"] = mx.zeros((4, HIDDEN, 8))
    tensors["model.layers.0.mlp.experts.lora_b"] = mx.zeros((4, 8, HIDDEN))
    mx.save_safetensors(os.path.join(mlx_dir, MLX_WEIGHTS_FILE), tensors)
    with pytest.raises(ValueError, match="non-2-D"):
        convert_mlx_dir_to_peft(
            mlx_dir, str(tmp_path / "b4"),
            module_types={"model.embed_tokens": "linear"},
        )
    # VLM wrapper layouts reorder the stack root; they must not pass the gate.
    tensors["language_model.model.layers.0.self_attn.q_proj.lora_a"] = mx.zeros((HIDDEN, 8))
    tensors["language_model.model.layers.0.self_attn.q_proj.lora_b"] = mx.zeros((8, HIDDEN))
    mx.save_safetensors(os.path.join(mlx_dir, MLX_WEIGHTS_FILE), tensors)
    with pytest.raises(ValueError, match="transformer stack"):
        convert_mlx_dir_to_peft(mlx_dir, str(tmp_path / "b5"))


@pytest.mark.parametrize("lora_kwargs,q_rank,q_scale,v_scale", [
    ({"rank_pattern": {"q_proj": 4}, "alpha_pattern": {"v_proj": 8}}, 4, 4.0, 1.0),
    ({"alpha_pattern": {"v_proj": 8}}, 8, 2.0, 1.0),  # scale-only repair path
])
def test_converted_dir_full_loop(tmp_path, base_dir, lora_kwargs, q_rank, q_scale, v_scale):
    peft_dir = str(tmp_path / "peft")
    wrapped, _ = _make_peft_adapter(base_dir, peft_dir, **lora_kwargs)
    mlx_dir = str(tmp_path / "mlx")
    convert_peft_dir_to_mlx(
        peft_dir, mlx_dir,
        json.load(open(os.path.join(base_dir, "config.json"))),
    )
    assert json.load(open(os.path.join(mlx_dir, "adapter_config.json")))[
        "unsloth_mlx_requires_unsloth_loader"
    ] is True

    from unsloth_zoo.mlx.loader import FastMLXModel
    model, _ = FastMLXModel.from_pretrained(
        mlx_dir, load_in_4bit=False, max_seq_length=64,
    )
    modules = dict(model.named_modules())
    q = modules["model.layers.0.self_attn.q_proj"]
    assert q.lora_a.shape[-1] == q_rank and q.scale == pytest.approx(q_scale)
    assert modules["model.layers.0.self_attn.v_proj"].scale == pytest.approx(v_scale)
    wrapped_paths = {n for n, m in modules.items() if hasattr(m, "lora_a")}
    assert len(wrapped_paths) == 2 * LAYERS  # no zero-init topology growth
    np.testing.assert_allclose(_mlx_logits(model), _peft_logits(wrapped), atol=5e-3)

    from unsloth_zoo.mlx.utils import save_lora_adapters
    resaved = str(tmp_path / "resaved")
    save_lora_adapters(model, resaved)
    recfg = json.load(open(os.path.join(resaved, "adapter_config.json")))
    assert recfg.get("unsloth_mlx_requires_unsloth_loader") is True
    model2, _ = FastMLXModel.from_pretrained(
        resaved, load_in_4bit=False, max_seq_length=64,
    )
    m2 = dict(model2.named_modules())
    assert m2["model.layers.0.self_attn.q_proj"].scale == pytest.approx(q_scale)
    assert m2["model.layers.0.self_attn.v_proj"].scale == pytest.approx(v_scale)
    np.testing.assert_allclose(_mlx_logits(model2), _mlx_logits(model), atol=2e-4)


def test_loader_import_train_step_save_reload(tmp_path, base_dir):
    peft_dir = str(tmp_path / "peft")
    wrapped, _ = _make_peft_adapter(base_dir, peft_dir)
    from unsloth_zoo.mlx.loader import FastMLXModel
    model, _ = FastMLXModel.from_pretrained(
        peft_dir, load_in_4bit=False, max_seq_length=64,
    )
    np.testing.assert_allclose(_mlx_logits(model), _peft_logits(wrapped), atol=5e-3)

    import mlx.optimizers as optim
    q = dict(model.named_modules())["model.layers.0.self_attn.q_proj"]
    lora_b_before, base_before = mx.array(q.lora_b), mx.array(q.linear.weight)

    def loss_fn(m):
        logits = m(mx.array([IDS]))
        logits = logits.logits if hasattr(logits, "logits") else logits
        return nn.losses.cross_entropy(logits, mx.array([IDS[1:] + [2]])).mean()

    _, grads = nn.value_and_grad(model, loss_fn)(model)
    optim.AdamW(learning_rate=1e-2).update(model, grads)
    mx.eval(model.parameters())
    assert not mx.array_equal(q.lora_b, lora_b_before)   # LoRA moved
    assert mx.array_equal(q.linear.weight, base_before)  # base frozen

    from unsloth_zoo.mlx.utils import save_lora_adapters
    saved = str(tmp_path / "saved")
    save_lora_adapters(model, saved)
    stepped = _mlx_logits(model)
    model2, _ = FastMLXModel.from_pretrained(
        saved, load_in_4bit=False, max_seq_length=64,
    )
    np.testing.assert_allclose(_mlx_logits(model2), stepped, atol=2e-4)


def test_peft_import_honors_quantized_base_request(tmp_path, base_dir):
    peft_dir = str(tmp_path / "peft")
    wrapped, _ = _make_peft_adapter(base_dir, peft_dir)
    from unsloth_zoo.mlx.loader import FastMLXModel
    model, _ = FastMLXModel.from_pretrained(peft_dir, max_seq_length=64)
    q = dict(model.named_modules())["model.layers.0.self_attn.q_proj"]
    assert type(q.linear) is nn.QuantizedLinear  # default load_in_4bit=True
    np.testing.assert_allclose(_mlx_logits(model), _peft_logits(wrapped), atol=0.35)


def test_converter_entries_reject_ambiguous_and_subclassed(tmp_path, base_dir):
    peft_dir = str(tmp_path / "peft")
    _, cfg = _make_peft_adapter(base_dir, peft_dir)
    mlx_dir = str(tmp_path / "mlx")
    convert_peft_dir_to_mlx(
        peft_dir, mlx_dir,
        json.load(open(os.path.join(base_dir, "config.json"))),
    )
    open(os.path.join(mlx_dir, PEFT_WEIGHTS_FILE), "wb").close()
    with pytest.raises(ValueError, match="both"):
        convert_mlx_dir_to_peft(mlx_dir, str(tmp_path / "b"))
    model, _ = load_model(Path(base_dir))

    class _Fusedish(nn.Linear):
        pass
    layer0 = model.model.layers[0].self_attn
    sub = _Fusedish(HIDDEN, HIDDEN, bias=False)
    sub.weight = layer0.q_proj.weight
    layer0.q_proj = sub
    with pytest.raises(ValueError, match="_Fusedish"):
        attach_and_bind_peft_adapter(
            model, peft_dir, normalize_peft_adapter_config(cfg),
        )


def test_helpers_import_without_torch(tmp_path):
    """Default Apple installs exclude torch: detection/validation and the
    saving_utils module itself must import and run with torch blocked."""
    import subprocess, sys, textwrap
    script = textwrap.dedent("""
        import importlib.abc, sys
        class _Block(importlib.abc.MetaPathFinder):
            BLOCKED = ("torch", "bitsandbytes", "peft", "triton", "xformers")
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in self.BLOCKED:
                    raise ModuleNotFoundError("blocked: " + name)
        sys.meta_path.insert(0, _Block())
        import unsloth_zoo.saving_utils as s
        cfg = {"peft_type": "LORA", "r": 8, "lora_alpha": 16}
        assert s.normalize_peft_adapter_config(cfg)["_unsloth_peft_import"]
        import unsloth_zoo.mlx.utils as mu
        assert mu.normalize_peft_adapter_config(dict(cfg))["_unsloth_peft_import"]
        print("OK")
    """)
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr[-2000:]


def test_peft_import_quant_override_predicate():
    from unsloth_zoo.mlx.loader import _peft_import_quant_override as ov
    assert ov({"load_in_4bit": False})            # explicit full-precision
    assert ov({"load_in_8bit": True})
    assert ov({"q_bits": 8})
    assert ov({"quantization_config": {"load_in_8bit": True}})
    assert not ov({"load_in_4bit": True})         # untouched default


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_converters_torch_fallback_without_mlx(tmp_path, base_dir, monkeypatch, dtype):
    """mlx-less hosts take the torch IO fallback; it must produce identical
    artifacts (the two backends differ in save-argument order)."""
    peft_dir = str(tmp_path / "peft")
    _make_peft_adapter(base_dir, peft_dir, dtype=dtype)
    # Poison the import machinery the way a torch-only host resolves it:
    # `import mlx.core` inside _tensor_backend raises ImportError.
    for name in ("mlx", "mlx.core"):
        monkeypatch.setitem(sys.modules, name, None)
    convert_peft_dir_to_mlx(
        peft_dir, str(tmp_path / "m2"), {"num_hidden_layers": LAYERS}
    )
    convert_mlx_dir_to_peft(str(tmp_path / "m2"), str(tmp_path / "back"))
    orig = st_load_file(os.path.join(peft_dir, PEFT_WEIGHTS_FILE))
    back = st_load_file(os.path.join(str(tmp_path / "back"), PEFT_WEIGHTS_FILE))
    assert set(orig) == set(back)
    for key in orig:
        assert orig[key].dtype == back[key].dtype, key
        assert torch.equal(orig[key], back[key]), key


def test_save_lora_adapters_peft_format(tmp_path, base_dir):
    peft_dir = str(tmp_path / "peft")
    wrapped, _ = _make_peft_adapter(base_dir, peft_dir)
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.utils import save_lora_adapters
    model, _ = FastMLXModel.from_pretrained(
        peft_dir, load_in_4bit=False, max_seq_length=64,
    )
    with pytest.raises(ValueError, match="adapter_format"):
        save_lora_adapters(model, str(tmp_path / "x"), adapter_format="bogus")
    out = str(tmp_path / "exported")
    model.save_lora_adapters(out, adapter_format="peft")  # bound method
    base = transformers.LlamaForCausalLM.from_pretrained(base_dir, dtype=torch.float32)
    reexported = peft.PeftModel.from_pretrained(base, out)
    np.testing.assert_allclose(
        _peft_logits(reexported), _mlx_logits(model), atol=5e-3,
    )
    from unsloth_zoo.saving_utils import export_peft_adapter
    with pytest.raises(ValueError, match="base_weights_source"):
        export_peft_adapter(str(tmp_path / "m"), str(tmp_path / "n"),
                            base_weights_source=object())


def test_peft_export_includes_lm_head_on_mirrored_layout(tmp_path, base_dir):
    peft_dir = str(tmp_path / "peft")
    wrapped, _ = _make_peft_adapter(
        base_dir, peft_dir, target_modules=["q_proj", "lm_head"],
        rank_pattern={"q_proj": 4},
    )
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.utils import save_lora_adapters
    model, _ = FastMLXModel.from_pretrained(
        peft_dir, load_in_4bit=False, max_seq_length=64,
    )
    out = str(tmp_path / "exported")
    save_lora_adapters(model, out, adapter_format="peft")
    keys = set(st_load_file(os.path.join(out, PEFT_WEIGHTS_FILE)))
    assert any("lm_head.lora_A" in k for k in keys)
    exported_cfg_text = open(os.path.join(out, "adapter_config.json")).read()
    assert "rank_pattern" in exported_cfg_text
    assert "unsloth_mlx" not in exported_cfg_text  # no zoo-internal leaks
    base = transformers.LlamaForCausalLM.from_pretrained(base_dir, dtype=torch.float32)
    reexported = peft.PeftModel.from_pretrained(base, out)
    # An lm_head adapter amplifies the zoo loader's bf16 base rounding at
    # the output layer; allow slightly more than the in-stack cases.
    np.testing.assert_allclose(
        _peft_logits(reexported), _mlx_logits(model), atol=8e-3,
    )


def test_peft_export_rejects_unsupported_wrapper(tmp_path, base_dir):
    peft_dir = str(tmp_path / "peft")
    _make_peft_adapter(base_dir, peft_dir)
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.utils import save_lora_adapters
    model, _ = FastMLXModel.from_pretrained(
        peft_dir, load_in_4bit=False, max_seq_length=64,
    )

    class _FusedWrap(nn.Module):
        # Plain factors AND a stock .linear base: the wrapper type itself
        # is what makes the semantics non-plain.
        def __init__(self, inner):
            super().__init__()
            self.lora_a = inner.lora_a
            self.lora_b = inner.lora_b
            self.linear = inner.linear

    layers = model.model.layers
    layers[0].self_attn.q_proj = _FusedWrap(layers[0].self_attn.q_proj)
    with pytest.raises(ValueError, match="_FusedWrap"):
        save_lora_adapters(model, str(tmp_path / "x"), adapter_format="peft")


def test_conversion_failure_leaves_no_destination(tmp_path, base_dir, monkeypatch):
    peft_dir = str(tmp_path / "peft")
    _make_peft_adapter(base_dir, peft_dir)
    import unsloth_zoo.saving_utils as su
    dst = str(tmp_path / "mlx")

    def boom(*args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(su.json, "dump", boom)
    with pytest.raises(RuntimeError, match="simulated"):
        convert_peft_dir_to_mlx(peft_dir, dst, {"num_hidden_layers": LAYERS})
    monkeypatch.undo()
    assert not os.path.exists(dst)  # failed claim fully released
    convert_peft_dir_to_mlx(peft_dir, dst, {"num_hidden_layers": LAYERS})
    assert os.path.exists(os.path.join(dst, MLX_WEIGHTS_FILE))
    # Trailing-separator spelling must not break staging or publication.
    dst2 = str(tmp_path / "mlx2") + os.sep
    convert_peft_dir_to_mlx(peft_dir, dst2, {"num_hidden_layers": LAYERS})
    assert os.path.exists(os.path.join(str(tmp_path / "mlx2"), MLX_WEIGHTS_FILE))


def test_peft_export_rejects_mixed_dropout(tmp_path, base_dir):
    peft_dir = str(tmp_path / "peft")
    _make_peft_adapter(base_dir, peft_dir, lora_dropout=0.1)
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.utils import save_lora_adapters
    model, _ = FastMLXModel.from_pretrained(
        peft_dir, load_in_4bit=False, max_seq_length=64,
    )
    modules = dict(model.named_modules())
    modules["model.layers.0.self_attn.q_proj"].dropout = nn.Dropout(p=0.4)
    with pytest.raises(ValueError, match="different lora_dropout"):
        save_lora_adapters(model, str(tmp_path / "x"), adapter_format="peft")


def test_dangling_symlink_destination_refused(tmp_path, base_dir):
    peft_dir = str(tmp_path / "peft")
    _make_peft_adapter(base_dir, peft_dir)
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "missing-target")
    for spelling in (str(link), str(link) + os.sep):
        with pytest.raises(ValueError, match="already exists"):
            convert_peft_dir_to_mlx(
                peft_dir, spelling, {"num_hidden_layers": LAYERS}
            )
    assert not (tmp_path / "missing-target").exists()


def test_exported_patterns_are_exact_anchored(tmp_path, base_dir):
    peft_dir = str(tmp_path / "peft")
    _make_peft_adapter(base_dir, peft_dir, rank_pattern={"q_proj": 4})
    mlx_dir, back = str(tmp_path / "m"), str(tmp_path / "b")
    convert_peft_dir_to_mlx(
        peft_dir, mlx_dir,
        json.load(open(os.path.join(base_dir, "config.json"))),
    )
    convert_mlx_dir_to_peft(mlx_dir, back)
    cfg = json.load(open(os.path.join(back, "adapter_config.json")))
    pattern = cfg["rank_pattern"]
    assert pattern and all(k.startswith("^") for k in pattern)
    key, rank = next(iter(sorted(pattern.items())))
    # Exact under peft's matcher: hits its own path, never a dotted suffix.
    path = key[1:].replace("\\.", ".")
    assert _resolve_pattern({key: rank}, path) == rank
    assert _resolve_pattern({key: rank}, "prefix." + path) is None

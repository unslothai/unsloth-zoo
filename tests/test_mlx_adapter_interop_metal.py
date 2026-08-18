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


def _make_peft_adapter(base_dir, out, dtype=torch.float32,
                       save_embedding_layers=False, mutate=None, **lora_kwargs):
    model = transformers.LlamaForCausalLM.from_pretrained(base_dir, dtype=dtype)
    kwargs = dict(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
    kwargs.update(lora_kwargs)
    wrapped = peft.get_peft_model(model, peft.LoraConfig(**kwargs))
    torch.manual_seed(7)
    with torch.no_grad():
        for name, param in wrapped.named_parameters():
            if "lora_B" in name or "lora_embedding_A" in name:
                param.copy_(torch.randn_like(param) * 0.05)
            if "lora_" in name:
                # peft autocasts adapter params to fp32; force the dtype so
                # bf16 runs exercise real bf16 tensors.
                param.data = param.data.to(dtype)
    if mutate is not None:
        with torch.no_grad():
            mutate(wrapped)
    wrapped.save_pretrained(out, save_embedding_layers=save_embedding_layers)
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


def test_converter_refuses_disagreeing_factor_ranks(tmp_path, base_dir):
    """A factor pair whose ranks disagree is refused, not silently bound."""
    peft_dir = str(tmp_path / "peft")
    _make_peft_adapter(base_dir, peft_dir)
    wf = os.path.join(peft_dir, PEFT_WEIGHTS_FILE)
    tensors = st_load_file(wf)
    key = next(k for k in tensors if k.endswith("lora_B.weight"))
    tensors[key] = torch.zeros(HIDDEN, 4)   # rank 4 against lora_A's 8
    from safetensors.torch import save_file as _st_save
    _st_save(tensors, wf)

    with pytest.raises(ValueError, match="lora_A rank 8 but lora_B rank 4"):
        convert_peft_dir_to_mlx(
            peft_dir, str(tmp_path / "mlx"),
            json.load(open(os.path.join(base_dir, "config.json"))),
        )
    # The export direction reads the other axis, in its own key spelling.
    mlx_dir = str(tmp_path / "mlx-ok")
    _make_peft_adapter(base_dir, str(tmp_path / "p2"))
    convert_peft_dir_to_mlx(
        str(tmp_path / "p2"), mlx_dir,
        json.load(open(os.path.join(base_dir, "config.json"))),
    )
    mt = st_load_file(os.path.join(mlx_dir, MLX_WEIGHTS_FILE))
    bkey = next(k for k in mt if k.endswith(".lora_b"))
    mt[bkey] = torch.zeros(4, HIDDEN)
    _st_save(mt, os.path.join(mlx_dir, MLX_WEIGHTS_FILE))
    with pytest.raises(ValueError, match="lora_a rank 8 but lora_b rank 4"):
        convert_mlx_dir_to_peft(mlx_dir, str(tmp_path / "back"))


@pytest.fixture(scope="module")
def tied_base_dir(tmp_path_factory):
    """Most small on-device models tie word embeddings, and peft auto-saves
    both the embedding and the head for them."""
    path = tmp_path_factory.mktemp("tiny-llama-tied")
    torch.manual_seed(0)
    transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=128,
        num_hidden_layers=LAYERS, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=128,
        tie_word_embeddings=True,
    )).to(torch.float32).save_pretrained(path, safe_serialization=True)
    return str(path)


def test_tied_output_head_snapshot_folds_or_refuses(tmp_path, tied_base_dir):
    """On a tied base the auto-saved head IS the embedding weight: it folds
    away silently, and a head that does not duplicate the embedding is
    refused rather than bound as independent head state."""
    peft_dir = str(tmp_path / "peft")
    _make_peft_adapter(
        tied_base_dir, peft_dir, save_embedding_layers=True,
        mutate=lambda pm: pm.get_input_embeddings().weight.add_(0.02),
    )
    base_config = json.load(open(os.path.join(tied_base_dir, "config.json")))
    wf = os.path.join(peft_dir, PEFT_WEIGHTS_FILE)
    head = next(k for k in st_load_file(wf) if "lm_head" in k)

    convert_peft_dir_to_mlx(peft_dir, str(tmp_path / "ok"), base_config)
    mlx_cfg = json.load(open(os.path.join(tmp_path, "ok", "adapter_config.json")))
    assert "lm_head" not in json.dumps(mlx_cfg.get("full_state_modules") or {})
    # Folded, not dropped: the written artifact must still carry the trained
    # embedding, or a reload silently reverts it to the base weight.
    assert any("embed_tokens" in p for p in mlx_cfg["full_state_modules"])
    torch.testing.assert_close(
        st_load_file(os.path.join(tmp_path, "ok", MLX_WEIGHTS_FILE))[
            "model.embed_tokens.weight"
        ],
        st_load_file(wf)[head],
    )

    # The live attach path folds through the same helper, with an MLX
    # equality callback that casts before comparing.
    cfg = json.load(open(os.path.join(peft_dir, "adapter_config.json")))
    model, _ = load_model(Path(tied_base_dir))
    attach_and_bind_peft_adapter(model, peft_dir, cfg)
    assert not any(
        "lm_head" in p
        for p in getattr(model, "_unsloth_full_state_modules", {})
    )

    # Folded, not dropped: the trained embedding must survive both paths.
    assert any(
        "embed_tokens" in p
        for p in getattr(model, "_unsloth_full_state_modules", {})
    )
    emb = dict(model.named_modules())["model.embed_tokens"]
    torch.testing.assert_close(
        torch.tensor(np.array(emb.weight)),
        st_load_file(wf)[head],
    )

    # export_peft_adapter's own refusal: a modules_to_save snapshot of a tied
    # embedding has no untied PEFT form.
    from unsloth_zoo.saving_utils import export_peft_adapter
    ok_cfg = os.path.join(tmp_path, "ok", "adapter_config.json")
    _c = json.load(open(ok_cfg))
    _c["full_state_modules"] = {"model.embed_tokens": "modules_to_save"}
    json.dump(_c, open(ok_cfg, "w"))
    with pytest.raises(ValueError, match="tied embedding or output head"):
        export_peft_adapter(
            str(tmp_path / "ok"), str(tmp_path / "exp"), base_config=base_config,
        )

    tensors = st_load_file(wf)
    tensors[head] = tensors[head] + 1.0
    from safetensors.torch import save_file as _st_save
    _st_save(tensors, wf)
    with pytest.raises(ValueError, match="tied output-head snapshot"):
        convert_peft_dir_to_mlx(peft_dir, str(tmp_path / "bad"), base_config)
    model2, _ = load_model(Path(tied_base_dir))
    with pytest.raises(ValueError, match="tied output-head snapshot"):
        attach_and_bind_peft_adapter(model2, peft_dir, cfg)


@pytest.mark.parametrize("field,value,match", [
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
    ("base_model.model.decoder.blocks.0.attn.q_proj.lora_A.weight",
     (8, HIDDEN), "decoder.blocks"),
    ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
     (HIDDEN + 8, 8), "out-dim"),
])
def test_attach_rejects_bad_tensors(tmp_path, base_dir, key, shape, match):
    peft_dir = str(tmp_path / "peft")
    _, cfg = _make_peft_adapter(base_dir, peft_dir)

    def mutate(tensors):
        tensors[key] = torch.zeros(*shape)
        if "decoder.blocks" in key:  # give the alien path a complete pair
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
    # One step must move LoRA weights and a save/reload must reproduce the
    # stepped logits. A native MLX reload leaves the base trainable (longstanding
    # loader behavior, unlike the PEFT import, which freezes — asserted in
    # test_peft_import_honors_quantized_base_request).
    import mlx.optimizers as optim
    q2 = m2["model.layers.0.self_attn.q_proj"]
    lora_b_before = mx.array(q2.lora_b)
    def loss_fn(m):
        logits = m(mx.array([IDS]))
        logits = logits.logits if hasattr(logits, "logits") else logits
        return nn.losses.cross_entropy(logits, mx.array([IDS[1:] + [2]])).mean()
    _, grads = nn.value_and_grad(model2, loss_fn)(model2)
    optim.AdamW(learning_rate=1e-2).update(model2, grads)
    mx.eval(model2.parameters())
    assert not mx.array_equal(q2.lora_b, lora_b_before)
    saved2 = str(tmp_path / "stepped")
    save_lora_adapters(model2, saved2)
    model3, _ = FastMLXModel.from_pretrained(saved2, load_in_4bit=False, max_seq_length=64)
    # Adapter state round-trips bitwise. No logits assert: this path's trainable
    # base moved during the step, and adapter saves carry only adapter tensors.
    q3 = dict(model3.named_modules())["model.layers.0.self_attn.q_proj"]
    assert mx.array_equal(q3.lora_b, q2.lora_b)


def test_peft_import_honors_quantized_base_request(tmp_path, base_dir):
    peft_dir = str(tmp_path / "peft")
    wrapped, _ = _make_peft_adapter(base_dir, peft_dir)
    from unsloth_zoo.mlx.loader import FastMLXModel
    model, _ = FastMLXModel.from_pretrained(peft_dir, max_seq_length=64)
    q = dict(model.named_modules())["model.layers.0.self_attn.q_proj"]
    assert type(q.linear) is nn.QuantizedLinear  # default load_in_4bit=True
    np.testing.assert_allclose(_mlx_logits(model), _peft_logits(wrapped), atol=0.35)
    # One step moves LoRA and the stepped state survives save/reload.
    # QuantizedLinear freezes independently; the attach-freeze contract on
    # full-precision bases is asserted in the DoRA lifecycle test.
    import mlx.optimizers as optim
    lora_b_before, base_before = mx.array(q.lora_b), mx.array(q.linear.weight)
    def loss_fn(m):
        logits = m(mx.array([IDS]))
        logits = logits.logits if hasattr(logits, "logits") else logits
        return nn.losses.cross_entropy(logits, mx.array([IDS[1:] + [2]])).mean()
    _, grads = nn.value_and_grad(model, loss_fn)(model)
    optim.AdamW(learning_rate=1e-2).update(model, grads)
    mx.eval(model.parameters())
    assert not mx.array_equal(q.lora_b, lora_b_before)
    assert mx.array_equal(q.linear.weight, base_before)
    from unsloth_zoo.mlx.utils import save_lora_adapters
    from unsloth_zoo.mlx.loader import FastMLXModel as _F
    saved = str(tmp_path / "stepped")
    save_lora_adapters(model, saved)
    m2, _ = _F.from_pretrained(saved, max_seq_length=64)
    np.testing.assert_allclose(_mlx_logits(m2), _mlx_logits(model), atol=2e-3)


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


@pytest.mark.parametrize("wrapper", ["fused", "dora_embedding"])
def test_peft_export_rejects_unsupported_wrapper(tmp_path, base_dir, wrapper):
    """Only a stock LoRALinear/DoRALinear over a stock Linear is
    representable: a foreign wrapper type and a stock wrapper over an
    embedding are both refused by name."""
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
            self.lora_a, self.lora_b = inner.lora_a, inner.lora_b
            self.linear = inner.linear

    if wrapper == "fused":
        attn = model.model.layers[0].self_attn
        attn.q_proj = _FusedWrap(attn.q_proj)
        match = "_FusedWrap"
    else:
        from mlx_lm.tuner.dora import DoRAEmbedding
        model.model.embed_tokens = DoRAEmbedding.from_base(
            model.model.embed_tokens, r=4,
        )
        match = "DoRAEmbedding"
    with pytest.raises(ValueError, match=match):
        save_lora_adapters(model, str(tmp_path / "x"), adapter_format="peft")


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


def test_dora_import_roundtrip_and_training(tmp_path, base_dir):
    peft_dir = str(tmp_path / "peft")
    wrapped, cfg = _make_peft_adapter(base_dir, peft_dir, use_dora=True)
    with torch.no_grad():  # make magnitudes informative
        for n, prm in wrapped.named_parameters():
            if "lora_magnitude_vector" in n:
                prm.add_(torch.randn_like(prm) * 0.05)
    wrapped.save_pretrained(peft_dir, save_embedding_layers=False)

    from unsloth_zoo.mlx.loader import FastMLXModel
    model, _ = FastMLXModel.from_pretrained(
        peft_dir, load_in_4bit=False, max_seq_length=64,
    )
    np.testing.assert_allclose(_mlx_logits(model), _peft_logits(wrapped), atol=5e-3)

    q = dict(model.named_modules())["model.layers.0.self_attn.q_proj"]
    m_before = mx.array(q.m)
    base_before = mx.array(q.linear.weight)
    import mlx.optimizers as optim
    def loss_fn(m):
        logits = m(mx.array([IDS]))
        logits = logits.logits if hasattr(logits, "logits") else logits
        return nn.losses.cross_entropy(logits, mx.array([IDS[1:] + [2]])).mean()
    _, grads = nn.value_and_grad(model, loss_fn)(model)
    optim.AdamW(learning_rate=1e-2).update(model, grads)
    mx.eval(model.parameters())
    assert not mx.array_equal(q.m, m_before)  # magnitude trains
    assert mx.array_equal(q.linear.weight, base_before)  # fp base frozen

    from unsloth_zoo.mlx.utils import save_lora_adapters
    saved = str(tmp_path / "saved")
    save_lora_adapters(model, saved)
    stepped = _mlx_logits(model)
    model2, _ = FastMLXModel.from_pretrained(saved, load_in_4bit=False, max_seq_length=64)
    np.testing.assert_allclose(_mlx_logits(model2), stepped, atol=2e-4)

    exported = str(tmp_path / "exported")
    model2.save_lora_adapters(exported, adapter_format="peft")
    ecfg = json.load(open(os.path.join(exported, "adapter_config.json")))
    assert ecfg["use_dora"] is True
    keys = set(st_load_file(os.path.join(exported, PEFT_WEIGHTS_FILE)))
    # peft's canonical on-disk key has no .weight suffix.
    assert any(k.endswith(".lora_magnitude_vector") for k in keys)
    assert not any(".lora_magnitude_vector.weight" in k for k in keys)
    base = transformers.LlamaForCausalLM.from_pretrained(base_dir, dtype=torch.float32)
    reloaded = peft.PeftModel.from_pretrained(base, exported)
    np.testing.assert_allclose(_peft_logits(reloaded), stepped, atol=5e-3)


def test_dora_dropout_rejected():
    with pytest.raises(ValueError, match="different training-mode"):
        normalize_peft_adapter_config({
            "peft_type": "LORA", "r": 8, "use_dora": True, "lora_dropout": 0.1,
        })


def test_dora_rejects_partial_magnitudes(tmp_path, base_dir):
    peft2 = str(tmp_path / "p2")
    _make_peft_adapter(base_dir, peft2, use_dora=True)
    from safetensors.torch import save_file
    wpath = os.path.join(peft2, PEFT_WEIGHTS_FILE)
    tensors = st_load_file(wpath)
    drop = next(k for k in tensors if "lora_magnitude_vector" in k)
    tensors.pop(drop)
    save_file(tensors, wpath)
    with pytest.raises(ValueError, match="all-or-nothing|magnitude"):
        convert_peft_dir_to_mlx(
            peft2, str(tmp_path / "m"),
            json.load(open(os.path.join(base_dir, "config.json"))),
        )


def test_mixed_lora_dora_save_refused(tmp_path, base_dir):
    peft_dir = str(tmp_path / "peft")
    _make_peft_adapter(base_dir, peft_dir, use_dora=True)
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.utils import save_lora_adapters
    from mlx_lm.tuner.lora import LoRALinear
    model, _ = FastMLXModel.from_pretrained(
        peft_dir, load_in_4bit=False, max_seq_length=64,
    )
    attn = model.model.layers[1].self_attn
    attn.k_proj = LoRALinear.from_base(attn.k_proj, r=4)
    from unsloth_zoo.mlx.utils import save_trainable_adapters
    for saver in (save_lora_adapters, save_trainable_adapters):
        with pytest.raises(ValueError, match="mixes plain-LoRA and DoRA"):
            saver(model, str(tmp_path / "x"))


def test_embedding_lora_is_refused(tmp_path, base_dir):
    """Embedding LoRA is unsupported, so it is named and refused rather than
    silently dropped from the converted adapter."""
    peft_dir = str(tmp_path / "peft")
    _make_peft_adapter(
        base_dir, peft_dir, target_modules=["q_proj", "embed_tokens"],
        save_embedding_layers="auto",
    )
    assert any("lora_embedding_A" in k for k in st_load_file(
        os.path.join(peft_dir, PEFT_WEIGHTS_FILE)
    ))
    with pytest.raises(ValueError, match="does not support"):
        convert_peft_dir_to_mlx(
            peft_dir, str(tmp_path / "mlx"),
            json.load(open(os.path.join(base_dir, "config.json"))),
        )
    from unsloth_zoo.mlx.loader import FastMLXModel
    with pytest.raises(ValueError, match="does not support"):
        FastMLXModel.from_pretrained(
            peft_dir, load_in_4bit=False, max_seq_length=64,
        )


def test_full_state_modules_lifecycle(tmp_path, base_dir):
    def _mutate(pm):
        dict(pm.named_modules())["base_model.model.lm_head"].modules_to_save[
            "default"
        ].weight += 0.01
        pm.get_input_embeddings().weight += 0.02
    peft_dir = str(tmp_path / "peft")
    wrapped, _ = _make_peft_adapter(
        base_dir, peft_dir, modules_to_save=["lm_head"],
        save_embedding_layers=True, mutate=_mutate,
    )
    from unsloth_zoo.mlx.loader import FastMLXModel
    model, _ = FastMLXModel.from_pretrained(peft_dir, load_in_4bit=False, max_seq_length=64)
    assert model._unsloth_full_state_modules == {
        "lm_head": "modules_to_save", "model.embed_tokens": "embedding_auto",
    }
    mods = dict(model.named_modules())
    emb_live = mods["model.embed_tokens"].weight
    # Applied exactly, up to the loader's own base-dtype cast.
    assert mx.array_equal(
        emb_live,
        mx.array(
            wrapped.get_input_embeddings().weight.detach().numpy()
        ).astype(emb_live.dtype),
    )
    np.testing.assert_allclose(_mlx_logits(model), _peft_logits(wrapped), atol=5e-3)
    # modules_to_save is trainable, the replaced embedding is not.
    from mlx.utils import tree_flatten
    trainable = dict(tree_flatten(model.trainable_parameters()))
    assert "lm_head.weight" in trainable
    assert "model.embed_tokens.weight" not in trainable

    import mlx.optimizers as optim
    lm_before = mx.array(mods["lm_head"].weight)
    emb_before = mx.array(mods["model.embed_tokens"].weight)
    def loss_fn(m):
        logits = m(mx.array([IDS]))
        logits = logits.logits if hasattr(logits, "logits") else logits
        return nn.losses.cross_entropy(logits, mx.array([IDS[1:] + [2]])).mean()
    _, grads = nn.value_and_grad(model, loss_fn)(model)
    optim.AdamW(learning_rate=1e-2).update(model, grads)
    mx.eval(model.parameters())
    assert not mx.array_equal(mods["lm_head"].weight, lm_before)
    assert mx.array_equal(mods["model.embed_tokens"].weight, emb_before)

    from unsloth_zoo.mlx.utils import save_lora_adapters
    mlx_dir = str(tmp_path / "mlx")
    save_lora_adapters(model, mlx_dir)
    cfg = json.load(open(os.path.join(mlx_dir, "adapter_config.json")))
    assert cfg["full_state_modules"] == model._unsloth_full_state_modules
    model2, _ = FastMLXModel.from_pretrained(mlx_dir, load_in_4bit=False, max_seq_length=64)
    mods2 = dict(model2.named_modules())
    assert mx.array_equal(mods2["lm_head"].weight, mods["lm_head"].weight)
    assert mx.array_equal(mods2["model.embed_tokens"].weight, emb_before)
    assert model2._unsloth_full_state_modules == model._unsloth_full_state_modules
    tr2 = dict(tree_flatten(model2.trainable_parameters()))
    assert "lm_head.weight" in tr2
    assert "model.embed_tokens.weight" not in tr2

    exported = str(tmp_path / "exported")
    save_lora_adapters(model2, exported, adapter_format="peft")
    ecfg = json.load(open(os.path.join(exported, "adapter_config.json")))
    # Reconstructed from origin tags: only the modules_to_save entry.
    assert ecfg["modules_to_save"] == ["lm_head"]
    base = transformers.LlamaForCausalLM.from_pretrained(base_dir, dtype=torch.float32)
    reloaded = peft.PeftModel.from_pretrained(base, exported, is_trainable=True)
    m2s_param = dict(reloaded.named_parameters())[
        "base_model.model.lm_head.modules_to_save.default.weight"
    ]
    assert m2s_param.requires_grad
    np.testing.assert_allclose(
        m2s_param.detach().numpy(),
        np.array(mods["lm_head"].weight.astype(mx.float32)), atol=0,
    )
    np.testing.assert_allclose(
        reloaded.get_input_embeddings().weight.detach().to(torch.float32).numpy(),
        np.array(emb_before.astype(mx.float32)), atol=0,
    )


def test_full_state_shape_mismatch_rejected(tmp_path, base_dir):
    peft_dir = str(tmp_path / "peft")
    _make_peft_adapter(
        base_dir, peft_dir, modules_to_save=["lm_head"],
        mutate=lambda pm: None,
    )
    wf = os.path.join(peft_dir, PEFT_WEIGHTS_FILE)
    tensors = st_load_file(wf)
    key = "base_model.model.lm_head.weight"
    tensors[key] = torch.cat([tensors[key], tensors[key][:4]], dim=0)
    from safetensors.torch import save_file as _st_save
    _st_save(tensors, wf)
    from unsloth_zoo.mlx.loader import FastMLXModel
    with pytest.raises(Exception, match="shape mismatch"):
        FastMLXModel.from_pretrained(peft_dir, load_in_4bit=False, max_seq_length=64)


def test_auto_saved_base_state_refused_off_embedding_and_head(tmp_path, base_dir):
    """peft auto-saves full base state for the input embedding and the output
    head only. Every LoRA target carries a .base_layer, so a key spelled that
    way on any other module would bind and replace live base weights. Stock
    mlx-lm binds adapters.safetensors with load_weights(strict=False) and never
    reads full_state_modules, so conversion must refuse before it writes."""
    peft_dir = str(tmp_path / "peft")
    _make_peft_adapter(base_dir, peft_dir)
    wf = os.path.join(peft_dir, PEFT_WEIGHTS_FILE)
    tensors = st_load_file(wf)
    target = "model.layers.0.self_attn.q_proj"
    tensors[f"base_model.model.{target}.base_layer.weight"] = torch.zeros(
        HIDDEN, HIDDEN
    )
    from safetensors.torch import save_file as _st_save
    _st_save(tensors, wf)
    refused = pytest.raises(
        ValueError, match="not the input embedding or output head",
    )

    model, _ = load_model(Path(base_dir))
    before = np.array(dict(model.named_modules())[target].weight)
    cfg = json.load(open(os.path.join(peft_dir, "adapter_config.json")))
    with refused:
        attach_and_bind_peft_adapter(model, peft_dir, cfg)
    live = dict(model.named_modules())[target]
    assert np.array_equal(np.array(getattr(live, "linear", live).weight), before)

    mlx_dir = tmp_path / "mlx"
    with refused:
        convert_peft_dir_to_mlx(
            peft_dir, str(mlx_dir),
            json.load(open(os.path.join(base_dir, "config.json"))),
        )
    assert not mlx_dir.exists()


@pytest.mark.parametrize("path,accepted", [
    ("model.embed_tokens", True),
    ("lm_head", True),
    ("transformer.wte", True),                 # GPT-2 / MPT
    ("gpt_neox.embed_in", True),
    ("backbone.embeddings", True),             # Mamba
    ("model.transformer.ff_out", True),        # OLMo's root output head
    ("model.tok_embeddings", True),            # InternLM2
    ("model.layers.0.self_attn.q_proj", False),
    ("model.ngram.embedders.0", False),
    ("model.embed_tokens_per_layer", False),   # Gemma 4 auxiliary table
    ("model.layers.0.mlp.gate_proj", False),
    # Leaves the root head shares with a per-layer module: OLMo names every
    # block's MLP output ff_out, and `output` also spells an attention
    # projection.
    ("model.transformer.blocks.0.ff_out", False),
    ("model.layers.0.attention.output", False),
])
def test_grouping_admits_only_hugging_face_embedding_and_head_paths(path, accepted):
    """A PEFT artifact carries Hugging Face paths, whose input-embedding and
    output-head spellings are a small known set. Conversion has no live tree to
    ask and the directory it writes is loadable by stock mlx-lm, so the path is
    all this layer has; a leaf shared with a per-layer module is told apart by
    the numbered segment such a path always carries."""
    from unsloth_zoo.saving_utils import group_peft_lora_pairs
    lora = "base_model.model.model.layers.0.self_attn.q_proj"
    _, full_state, rejected = group_peft_lora_pairs({
        f"{lora}.lora_A.weight": mx.zeros((8, HIDDEN)),
        f"{lora}.lora_B.weight": mx.zeros((HIDDEN, 8)),
        f"base_model.model.{path}.base_layer.weight": mx.zeros((VOCAB, HIDDEN)),
    }, {})
    assert (not rejected) is accepted
    assert (path in full_state) is accepted


def _peft_task_type(name):
    # A config built with peft's TaskType keeps the enum, while adapter JSON
    # carries the bare string; both spellings must reach the same verdict.
    from peft import TaskType
    return getattr(TaskType, name)


@pytest.mark.parametrize("cfg_extra, refused", [
    ({"task_type": "CAUSAL_LM"}, False),
    ({"task_type": "causal_lm"}, False),
    ({}, False),
    ({"task_type": "SEQ_CLS"}, True),
    ({"task_type": "SEQ_2_SEQ_LM"}, True),
    ({"task_type": _peft_task_type("CAUSAL_LM")}, False),
])
def test_import_refuses_non_causal_task(cfg_extra, refused):
    # The backend serves causal LMs, so a differently-tasked adapter would
    # bind its backbone factors and answer with vocabulary logits.
    from unsloth_zoo.saving_utils import normalize_peft_adapter_config
    cfg = {"peft_type": "LORA", "r": 4, "lora_alpha": 8,
           "target_modules": ["q_proj"], **cfg_extra}
    if refused:
        with pytest.raises(ValueError, match="CAUSAL_LM"):
            normalize_peft_adapter_config(cfg)
    else:
        normalize_peft_adapter_config(cfg)


def test_import_accepts_a_live_peft_config():
    # LoraConfig.to_dict() hands back peft's enum for peft_type, whose str() is
    # "PeftType.LORA", so comparing it would refuse an ordinary causal adapter.
    # The task_type enum spelling is covered by the case above.
    import peft
    from unsloth_zoo.saving_utils import normalize_peft_adapter_config
    cfg = peft.LoraConfig(
        task_type="CAUSAL_LM", r=4, lora_alpha=8, target_modules=["q_proj"],
    ).to_dict()
    normalize_peft_adapter_config(cfg)

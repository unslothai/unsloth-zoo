import importlib.util
import inspect
import sys
import textwrap

import pytest
import torch

from unsloth_zoo.temporary_patches import gemma4
from unsloth_zoo.compiler import fix_gemma4_audio_feature_dtype


def _load_fake_gemma4(tmp_path, monkeypatch, name, cast_expression):
    source = textwrap.dedent(
        f"""
        import functools
        import torch

        def preserve_metadata(function):
            @functools.wraps(function)
            def wrapper(*args, **kwargs):
                return function(*args, **kwargs)
            return wrapper

        class Gemma4Model:
            def __init__(self):
                pass

            @preserve_metadata
            def forward(self, inputs_embeds, audio_features=None):
                if audio_features is None:
                    return inputs_embeds
                audio_mask = torch.ones_like(inputs_embeds, dtype=torch.bool)
                return inputs_embeds.masked_scatter(
                    audio_mask,
                    {cast_expression},
                )
        """
    )
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module.Gemma4Model


@pytest.mark.parametrize("inputs_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("audio_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_gemma4_audio_merge_aligns_to_actual_inputs_embeds_dtype(
    tmp_path, monkeypatch, inputs_dtype, audio_dtype,
):
    model_cls = _load_fake_gemma4(
        tmp_path,
        monkeypatch,
        f"fake_gemma4_{str(inputs_dtype)}_{str(audio_dtype)}".replace(".", "_"),
        "audio_features.to(inputs_embeds.device)",
    )
    assert gemma4._patch_gemma4_audio_feature_dtype_on_class(model_cls) is True
    patched_source = inspect.getsource(model_cls.forward)
    assert "audio_features.to(inputs_embeds.device)" not in patched_source
    assert "audio_features.to(inputs_embeds.device, inputs_embeds.dtype)" in patched_source

    inputs = torch.zeros(1, 2, dtype=inputs_dtype, requires_grad=True)
    audio = torch.tensor([[1.25, -2.5]], dtype=audio_dtype, requires_grad=True)
    output = model_cls().forward(inputs_embeds=inputs, audio_features=audio)

    assert output.dtype == inputs_dtype
    torch.testing.assert_close(output, audio.to(inputs_dtype))
    output.float().sum().backward()
    assert audio.grad is not None


def test_gemma4_audio_dtype_patch_preserves_no_audio_and_is_idempotent(
    tmp_path, monkeypatch,
):
    model_cls = _load_fake_gemma4(
        tmp_path,
        monkeypatch,
        "fake_gemma4_idempotent",
        "audio_features.to(inputs_embeds.device)",
    )
    original_signature = inspect.signature(model_cls.forward)
    original_forward = model_cls.forward
    assert gemma4._patch_gemma4_audio_feature_dtype_on_class(model_cls) is True
    patched_forward = model_cls.forward
    assert patched_forward is original_forward
    assert gemma4._patch_gemma4_audio_feature_dtype_on_class(model_cls) is False
    assert model_cls.forward is patched_forward
    assert inspect.signature(model_cls.forward) == original_signature

    inputs = torch.randn(1, 2, dtype=torch.float32)
    assert model_cls().forward(inputs_embeds=inputs) is inputs


def test_gemma4_audio_dtype_patch_noops_when_upstream_is_fixed(tmp_path, monkeypatch):
    model_cls = _load_fake_gemma4(
        tmp_path,
        monkeypatch,
        "fake_gemma4_fixed",
        "audio_features.to(inputs_embeds.device, inputs_embeds.dtype)",
    )
    original_forward = model_cls.forward

    assert gemma4._patch_gemma4_audio_feature_dtype_on_class(model_cls) is False
    assert model_cls.forward is original_forward
    output = model_cls().forward(
        inputs_embeds=torch.zeros(1, 2, dtype=torch.float32),
        audio_features=torch.ones(1, 2, dtype=torch.float16),
    )
    assert output.dtype == torch.float32


def test_gemma4_audio_dtype_patch_fails_closed_on_source_drift(tmp_path, monkeypatch):
    model_cls = _load_fake_gemma4(
        tmp_path,
        monkeypatch,
        "fake_gemma4_drifted",
        "audio_features.to(device=inputs_embeds.device)",
    )
    original_forward = model_cls.forward

    assert gemma4._patch_gemma4_audio_feature_dtype_on_class(model_cls) is False
    assert model_cls.forward is original_forward
    assert not hasattr(model_cls, "_unsloth_audio_feature_dtype_patched")


def test_gemma4_audio_dtype_patch_is_registered_before_model_wrapper():
    names = [patch.__name__ for patch in gemma4.TEMPORARY_PATCHES]
    assert names.index("patch_Gemma4Model_audio_feature_dtype") < names.index(
        "patch_Gemma4Model_forward_kv_shared_no_cache"
    )


def test_gemma4_compiler_transform_is_guarded_and_upstream_safe():
    buggy = "audio_features.to(inputs_embeds.device)"
    fixed = "audio_features.to(inputs_embeds.device, inputs_embeds.dtype)"

    assert fix_gemma4_audio_feature_dtype(buggy) == fixed
    assert fix_gemma4_audio_feature_dtype(fixed) == fixed
    multiline = "audio_features.to(\n    inputs_embeds.device\n)"
    assert fix_gemma4_audio_feature_dtype(multiline) == fixed
    duplicated = f"{buggy}\n{buggy}"
    assert fix_gemma4_audio_feature_dtype(duplicated) == duplicated


def test_gemma4_compiler_emits_fixed_generated_forward(tmp_path, monkeypatch):
    from unsloth_zoo import compiler

    module_name = "fake_gemma4_compiler_source"
    _load_fake_gemma4(
        tmp_path,
        monkeypatch,
        module_name,
        "audio_features.to(inputs_embeds.device)",
    )
    module = sys.modules[module_name]
    monkeypatch.setattr(compiler, module_name, module, raising=False)

    generated = compiler.create_standalone_class(
        "Gemma4Model",
        module_name,
        dir(module),
        disable=True,
    )
    assert "audio_features.to(inputs_embeds.device)" not in generated
    assert "audio_features.to(inputs_embeds.device, inputs_embeds.dtype)" in generated


def test_gemma4_audio_dtype_patch_is_torch_compile_safe(tmp_path, monkeypatch):
    model_cls = _load_fake_gemma4(
        tmp_path,
        monkeypatch,
        "fake_gemma4_compiled",
        "audio_features.to(inputs_embeds.device)",
    )
    gemma4._patch_gemma4_audio_feature_dtype_on_class(model_cls)
    model = model_cls()
    compiled = torch.compile(model.forward, backend="eager", fullgraph=True)

    inputs = torch.zeros(1, 2, dtype=torch.float32)
    audio = torch.ones(1, 2, dtype=torch.float16)
    torch.testing.assert_close(compiled(inputs, audio), model.forward(inputs, audio))

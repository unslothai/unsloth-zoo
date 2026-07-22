import importlib.util
import inspect
import sys
import textwrap

import pytest
import torch

from unsloth_zoo.compiler import (
    _GEMMA4_PLE_CAST_HELPER,
    fix_gemma4_forced_float32_ple_dtype,
)
from unsloth_zoo.temporary_patches.gemma4_float32 import _unsloth_gemma4_ple_cast_input
from unsloth_zoo.temporary_patches.gemma4_float32 import _patch_gemma4_ple_dtype_on_method


def _compiler_ple_cast_input(module, x):
    namespace = {}
    exec(_GEMMA4_PLE_CAST_HELPER, namespace)
    return namespace["_unsloth_gemma4_ple_cast_input"](module, x)


@pytest.fixture(params=[_unsloth_gemma4_ple_cast_input, _compiler_ple_cast_input])
def ple_cast_input(request):
    return request.param


class _ModuleWithWeight:
    def __init__(self, weight):
        self.weight = weight


class _PeftLikeModule:
    def __init__(self, base_layer):
        self._base_layer = base_layer

    def get_base_layer(self):
        return self._base_layer


def _load_fake_ple_module(tmp_path, monkeypatch, name, source):
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_gemma4_ple_cast_input_aligns_floating_weight_and_peft_base_layer(ple_cast_input):
    x = torch.randn(2, 3, dtype=torch.float32)
    linear = torch.nn.Linear(3, 3, bias=False, dtype=torch.float16)

    cast = ple_cast_input(_PeftLikeModule(linear), x)

    assert cast.dtype == torch.float16
    assert cast is not x
    assert linear(cast).dtype == torch.float16


def test_gemma4_ple_cast_input_leaves_quant_state_activation_exactly_unchanged(ple_cast_input):
    weight = torch.empty(3, 3, dtype=torch.uint8)
    weight.quant_state = object()
    x = torch.randn(2, 3, dtype=torch.float32)

    cast = ple_cast_input(_ModuleWithWeight(weight), x)

    assert cast is x
    assert cast.dtype == torch.float32


def test_gemma4_ple_cast_input_leaves_nonfloating_packed_weight_unchanged(ple_cast_input):
    x = torch.randn(2, 3, dtype=torch.float32)
    cast = ple_cast_input(_ModuleWithWeight(torch.empty(3, 3, dtype=torch.uint8)), x)

    assert cast is x
    assert cast.dtype == torch.float32


@pytest.mark.parametrize(
    "module, source, expected",
    [
        (
            "Gemma4TextModel",
            "return self.per_layer_model_projection(inputs_embeds)",
            ("self.per_layer_model_projection(_unsloth_gemma4_ple_cast_input(self.per_layer_model_projection, inputs_embeds))",),
        ),
        (
            "Gemma4TextDecoderLayer",
            "hidden_states = self.per_layer_input_gate(hidden_states)\n    return self.per_layer_projection(hidden_states)",
            (
                "self.per_layer_input_gate(_unsloth_gemma4_ple_cast_input(self.per_layer_input_gate, hidden_states))",
                "self.per_layer_projection(_unsloth_gemma4_ple_cast_input(self.per_layer_projection, hidden_states))",
            ),
        ),
    ],
)
def test_gemma4_ple_compiler_transform_is_gated_exact_and_idempotent(
    monkeypatch, module, source, expected,
):
    source = f"def forward(self, inputs_embeds=None, hidden_states=None):\n    {source}\n"
    monkeypatch.delenv("UNSLOTH_FORCE_FLOAT32", raising=False)
    assert fix_gemma4_forced_float32_ple_dtype(source, module) == source

    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    rewritten = fix_gemma4_forced_float32_ple_dtype(source, module)
    for call in expected:
        assert rewritten.count(call) == 1
    assert fix_gemma4_forced_float32_ple_dtype(rewritten, module) == rewritten


def test_gemma4_ple_method_patch_lifecycle_covers_all_three_boundaries(tmp_path, monkeypatch):
    module = _load_fake_ple_module(
        tmp_path,
        monkeypatch,
        "fake_gemma4_ple_lifecycle",
        """
        import torch

        class Gemma4TextModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.per_layer_model_projection = torch.nn.Linear(3, 3, bias=False, dtype=torch.float16)

            def project_per_layer_inputs(self, inputs_embeds):
                return self.per_layer_model_projection(inputs_embeds)

        class Gemma4TextDecoderLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.per_layer_input_gate = torch.nn.Linear(3, 3, bias=False, dtype=torch.float16)
                self.per_layer_projection = torch.nn.Linear(3, 3, bias=False, dtype=torch.float16)

            def forward(self, hidden_states, per_layer_input):
                hidden_states = self.per_layer_input_gate(hidden_states)
                hidden_states = hidden_states * per_layer_input
                return self.per_layer_projection(hidden_states)
        """,
    )
    model_cls = module.Gemma4TextModel
    decoder_cls = module.Gemma4TextDecoderLayer

    assert _patch_gemma4_ple_dtype_on_method(
        model_cls,
        "project_per_layer_inputs",
        (("per_layer_model_projection", "inputs_embeds"),),
    ) is True
    assert _patch_gemma4_ple_dtype_on_method(
        decoder_cls,
        "forward",
        (("per_layer_input_gate", "hidden_states"), ("per_layer_projection", "hidden_states")),
    ) is True

    assert "_unsloth_gemma4_ple_cast_input(self.per_layer_model_projection, inputs_embeds)" in inspect.getsource(
        model_cls.project_per_layer_inputs
    )
    patched_decoder = inspect.getsource(decoder_cls.forward)
    assert "_unsloth_gemma4_ple_cast_input(self.per_layer_input_gate, hidden_states)" in patched_decoder
    assert "_unsloth_gemma4_ple_cast_input(self.per_layer_projection, hidden_states)" in patched_decoder
    assert model_cls._unsloth_ple_dtype_project_per_layer_inputs_patched is True
    assert decoder_cls._unsloth_ple_dtype_forward_patched is True

    model, decoder = model_cls(), decoder_cls()
    model_dtypes, decoder_dtypes = [], []
    model.per_layer_model_projection.register_forward_pre_hook(lambda _, args: model_dtypes.append(args[0].dtype))
    decoder.per_layer_input_gate.register_forward_pre_hook(lambda _, args: decoder_dtypes.append(args[0].dtype))
    decoder.per_layer_projection.register_forward_pre_hook(lambda _, args: decoder_dtypes.append(args[0].dtype))
    inputs_embeds = torch.randn(2, 3, dtype=torch.float32, requires_grad=True)
    hidden_states = torch.randn(2, 3, dtype=torch.float32, requires_grad=True)
    per_layer_input = torch.randn(2, 3, dtype=torch.float32)
    model_output = model.project_per_layer_inputs(inputs_embeds)
    decoder_output = decoder.forward(hidden_states, per_layer_input)

    assert model_output.dtype == torch.float16
    assert decoder_output.dtype == torch.float16
    assert model_dtypes == [torch.float16]
    assert decoder_dtypes == [torch.float16, torch.float16]
    (model_output.float().sum() + decoder_output.float().sum()).backward()
    assert inputs_embeds.grad is not None and torch.isfinite(inputs_embeds.grad).all()
    assert hidden_states.grad is not None and torch.isfinite(hidden_states.grad).all()
    for parameter in (*model.parameters(), *decoder.parameters()):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()

    assert _patch_gemma4_ple_dtype_on_method(
        model_cls,
        "project_per_layer_inputs",
        (("per_layer_model_projection", "inputs_embeds"),),
    ) is False
    assert _patch_gemma4_ple_dtype_on_method(
        decoder_cls,
        "forward",
        (("per_layer_input_gate", "hidden_states"), ("per_layer_projection", "hidden_states")),
    ) is False


@pytest.mark.parametrize(
    "name, body, expected_marker",
    [
        (
            "fixed",
            "return self.per_layer_model_projection(_unsloth_gemma4_ple_cast_input(self.per_layer_model_projection, inputs_embeds))",
            True,
        ),
        (
            "drifted",
            "return self.per_layer_model_projection(inputs_embeds.to(inputs_embeds.dtype))",
            False,
        ),
    ],
)
def test_gemma4_ple_method_patch_fails_closed_for_fixed_or_drifted_source(
    tmp_path, monkeypatch, name, body, expected_marker,
):
    module = _load_fake_ple_module(
        tmp_path,
        monkeypatch,
        f"fake_gemma4_ple_{name}",
        f"""
        class Gemma4TextModel:
            def project_per_layer_inputs(self, inputs_embeds):
                {body}
        """,
    )
    model_cls = module.Gemma4TextModel

    assert _patch_gemma4_ple_dtype_on_method(
        model_cls,
        "project_per_layer_inputs",
        (("per_layer_model_projection", "inputs_embeds"),),
    ) is False
    assert getattr(model_cls, "_unsloth_ple_dtype_project_per_layer_inputs_patched", False) is expected_marker


@pytest.mark.parametrize(
    "module, source",
    [
        (
        "Gemma4TextModel",
        """
        def forward(self, inputs_embeds):
            return self.per_layer_model_projection(inputs_embeds)
        """),
        (
        "Gemma4TextDecoderLayer",
        """
        def forward(self, hidden_states):
            hidden_states = self.per_layer_input_gate(hidden_states)
            return self.per_layer_projection(hidden_states)
        """),
        (
        "Gemma4TextModel",
        """
        class Gemma4TextModel:
            def forward(self, inputs_embeds):
                return self.per_layer_model_projection(inputs_embeds)
        """),
        (
        "Gemma4TextDecoderLayer",
        """
        class Gemma4TextDecoderLayer:
            def forward(self, hidden_states):
                hidden_states = self.per_layer_input_gate(hidden_states)
                return self.per_layer_projection(hidden_states)
        """),
    ],
)
def test_gemma4_ple_compiler_transform_emits_valid_forward_and_full_class_source(
    monkeypatch, module, source,
):
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    rewritten = fix_gemma4_forced_float32_ple_dtype(textwrap.dedent(source), module)

    expected_calls = 1 if module == "Gemma4TextModel" else 2
    assert rewritten.count("_unsloth_gemma4_ple_cast_input") == expected_calls
    assert _GEMMA4_PLE_CAST_HELPER.count("def _unsloth_gemma4_ple_cast_input") == 1
    compile(rewritten + _GEMMA4_PLE_CAST_HELPER, "<gemma4-ple-generated>", "exec")


def test_gemma4_ple_three_boundary_eager_and_compiled_parity():
    class ThreePLEBoundaries(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.per_layer_model_projection = torch.nn.Linear(3, 3, bias=False, dtype=torch.float16)
            self.per_layer_input_gate = torch.nn.Linear(3, 3, bias=False, dtype=torch.float16)
            self.per_layer_projection = torch.nn.Linear(3, 3, bias=False, dtype=torch.float16)

        def forward(self, inputs_embeds, hidden_states, per_layer_input):
            self.per_layer_model_projection(
                _unsloth_gemma4_ple_cast_input(self.per_layer_model_projection, inputs_embeds)
            )
            hidden_states = self.per_layer_input_gate(
                _unsloth_gemma4_ple_cast_input(self.per_layer_input_gate, hidden_states)
            )
            hidden_states = torch.nn.functional.silu(hidden_states) * per_layer_input
            return self.per_layer_projection(
                _unsloth_gemma4_ple_cast_input(self.per_layer_projection, hidden_states)
            )

    model = ThreePLEBoundaries()
    inputs = torch.randn(2, 3, dtype=torch.float32)
    hidden_states = torch.randn(2, 3, dtype=torch.float32)
    per_layer_input = torch.randn(2, 3, dtype=torch.float32)
    eager = model(inputs, hidden_states, per_layer_input)

    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(inputs, hidden_states, per_layer_input), eager)


class _WeightStub:
    """Minimal `.weight` stand-in: the cast helper only reads dtype/quant_state."""
    pass


def test_gemma4_ple_cast_input_leaves_fp8_weight_unchanged(ple_cast_input):
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None:
        pytest.skip("float8_e4m3fn is unavailable on this torch build")
    x = torch.randn(2, 3, dtype=torch.float32)
    weight = _WeightStub()
    weight.dtype = fp8  # sub-2-byte float, no quant_state

    cast = ple_cast_input(_ModuleWithWeight(weight), x)

    # Must not downcast the fp32 carrier to fp8 (fp8 kernels scale internally).
    assert cast is x
    assert cast.dtype == torch.float32


def test_gemma4_ple_cast_input_leaves_weight_without_dtype_unchanged(ple_cast_input):
    x = torch.randn(2, 3, dtype=torch.float32)
    # A weight proxy that does not expose `.dtype` must not raise AttributeError.
    cast = ple_cast_input(_ModuleWithWeight(_WeightStub()), x)

    assert cast is x
    assert cast.dtype == torch.float32


def test_gemma4_ple_dry_run_classifies_without_mutating(tmp_path, monkeypatch):
    module = _load_fake_ple_module(
        tmp_path,
        monkeypatch,
        "fake_gemma4_ple_dryrun",
        """
        import torch

        class Gemma4TextModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.per_layer_model_projection = torch.nn.Linear(3, 3, bias=False, dtype=torch.float16)

            def project_per_layer_inputs(self, inputs_embeds):
                return self.per_layer_model_projection(inputs_embeds)

        class Gemma4TextModelDrifted(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.per_layer_model_projection = torch.nn.Linear(3, 3, bias=False, dtype=torch.float16)

            def project_per_layer_inputs(self, inputs_embeds):
                return self.per_layer_model_projection(inputs_embeds.to(inputs_embeds.dtype))
        """,
    )
    targets = (("per_layer_model_projection", "inputs_embeds"),)

    ok_cls = module.Gemma4TextModel
    original_code = ok_cls.project_per_layer_inputs.__code__
    status = _patch_gemma4_ple_dtype_on_method(
        ok_cls, "project_per_layer_inputs", targets, dry_run=True,
    )
    assert status == "PATCH"
    # Dry run must not touch bytecode, the marker, or linecache.
    assert ok_cls.project_per_layer_inputs.__code__ is original_code
    assert not getattr(ok_cls, "_unsloth_ple_dtype_project_per_layer_inputs_patched", False)

    drift_cls = module.Gemma4TextModelDrifted
    assert _patch_gemma4_ple_dtype_on_method(
        drift_cls, "project_per_layer_inputs", targets, dry_run=True,
    ) == "DRIFT"

    # A real (non-dry) call on the drifted class fails closed without mutating.
    drift_code = drift_cls.project_per_layer_inputs.__code__
    assert _patch_gemma4_ple_dtype_on_method(
        drift_cls, "project_per_layer_inputs", targets,
    ) is False
    assert drift_cls.project_per_layer_inputs.__code__ is drift_code

    # Dry run on the already-marked good class reports ALREADY.
    assert _patch_gemma4_ple_dtype_on_method(ok_cls, "project_per_layer_inputs", targets) is True
    assert _patch_gemma4_ple_dtype_on_method(
        ok_cls, "project_per_layer_inputs", targets, dry_run=True,
    ) == "ALREADY"


def _make_fake_decoder_module(tmp_path, monkeypatch, name):
    return _load_fake_ple_module(
        tmp_path,
        monkeypatch,
        name,
        """
        import torch

        class Gemma4TextDecoderLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.per_layer_input_gate = torch.nn.Linear(3, 3, bias=False, dtype=torch.float16)
                self.per_layer_projection = torch.nn.Linear(3, 3, bias=False, dtype=torch.float16)

            def forward(self, hidden_states, per_layer_input):
                hidden_states = self.per_layer_input_gate(hidden_states)
                hidden_states = hidden_states * per_layer_input
                return self.per_layer_projection(hidden_states)
        """,
    )


def test_gemma4_ple_compiler_appends_helper_only_when_call_present(tmp_path, monkeypatch):
    from unsloth_zoo import compiler

    module = _make_fake_decoder_module(tmp_path, monkeypatch, "fake_gemma4_ple_append")
    monkeypatch.setattr(compiler, "fake_gemma4_ple_append", module, raising=False)

    # Flag off: no PLE rewrite, so no helper call and no appended helper def.
    monkeypatch.delenv("UNSLOTH_FORCE_FLOAT32", raising=False)
    off = compiler.create_standalone_class(
        "Gemma4TextDecoderLayer", "fake_gemma4_ple_append", dir(module), disable=True,
    )
    assert "_unsloth_gemma4_ple_cast_input(" not in off
    assert "def _unsloth_gemma4_ple_cast_input" not in off

    # Flag on: the rewrite inserts the call, so the helper def is appended exactly once.
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    on = compiler.create_standalone_class(
        "Gemma4TextDecoderLayer", "fake_gemma4_ple_append", dir(module), disable=True,
    )
    assert "_unsloth_gemma4_ple_cast_input(" in on
    assert on.count("def _unsloth_gemma4_ple_cast_input") == 1
    compile(on, "<gemma4-ple-append>", "exec")


def test_gemma4_ple_eager_then_compile_share_one_helper_name(tmp_path, monkeypatch):
    """Guards the fixed bug: compiling AFTER the eager patch must not emit a call to
    a helper name the generated module never defines (previously a NameError)."""
    from unsloth_zoo import compiler

    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    module = _make_fake_decoder_module(tmp_path, monkeypatch, "fake_gemma4_ple_crosspath")
    decoder_cls = module.Gemma4TextDecoderLayer

    # Eager patch first: rewrites __code__ AND poisons linecache with wrapped source.
    assert _patch_gemma4_ple_dtype_on_method(
        decoder_cls, "forward",
        (("per_layer_input_gate", "hidden_states"), ("per_layer_projection", "hidden_states")),
    ) is True
    # inspect.getsource (which the compiler uses) now returns the eager-patched source.
    assert "_unsloth_gemma4_ple_cast_input(" in inspect.getsource(decoder_cls.forward)

    # The compiler reads that poisoned source; the generated module must DEFINE the
    # same helper name it CALLS.
    monkeypatch.setattr(compiler, "fake_gemma4_ple_crosspath", module, raising=False)
    generated = compiler.create_standalone_class(
        "Gemma4TextDecoderLayer", "fake_gemma4_ple_crosspath", dir(module), disable=True,
    )
    assert "_unsloth_gemma4_ple_cast_input(" in generated
    assert generated.count("def _unsloth_gemma4_ple_cast_input") == 1
    compile(generated, "<gemma4-ple-crosspath>", "exec")

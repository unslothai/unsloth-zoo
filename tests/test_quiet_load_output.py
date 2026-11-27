# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Load-time warnings a user cannot act on stay out of the default log: the Gemma 3 processor patch
no longer reads the Gemma3ProcessorKwargs alias that transformers 5.23 deprecated; the device-map
planner's sizing probe no longer prints transformers' unknown quantization type warning."""

import logging
import warnings

import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("transformers.models.gemma3.processing_gemma3")


def _future_warnings(record, needle):
    return [w for w in record if issubclass(w.category, FutureWarning) and needle in str(w.message)]


def test_gemma3_processor_kwargs_resolve_without_warning():
    import transformers.models.gemma3.processing_gemma3 as processing_gemma3
    from transformers.processing_utils import ProcessingKwargs
    from unsloth_zoo.temporary_patches.gemma import _resolve_Gemma3ProcessorKwargs

    with warnings.catch_warnings(record = True) as record:
        warnings.simplefilter("always")
        kwargs_class = _resolve_Gemma3ProcessorKwargs()
    assert _future_warnings(record, "ProcessorKwargs") == [], [str(w.message) for w in record]
    own = vars(processing_gemma3).get("Gemma3ProcessorKwargs")
    if own is not None:
        # 4.57.6 .. 5.22: the real class, carrying Gemma 3's own `_defaults`.
        assert kwargs_class is own and "images_kwargs" in kwargs_class._defaults
    else:
        # 5.23+: what the deprecated alias resolves to, and what 5.29 leaves behind.
        assert kwargs_class is ProcessingKwargs


def test_gemma3_processor_patch_does_not_warn():
    import transformers.models.gemma3.processing_gemma3 as processing_gemma3
    from unsloth_zoo.temporary_patches.gemma import patch_Gemma3Processor

    with warnings.catch_warnings(record = True) as record:
        warnings.simplefilter("always")
        patch_Gemma3Processor()
    assert _future_warnings(record, "ProcessorKwargs") == [], [str(w.message) for w in record]
    call = processing_gemma3.Gemma3Processor.__call__
    assert "unsloth_zoo" in (getattr(call, "__module__", "") or "")


def test_gemma3_processor_patch_survives_alias_removal(monkeypatch):
    # transformers 5.29 drops the alias: a module-level __getattr__ that no longer knows the name.
    import transformers.models.gemma3.processing_gemma3 as processing_gemma3
    from transformers.processing_utils import ProcessingKwargs
    from unsloth_zoo.temporary_patches import gemma

    namespace = vars(processing_gemma3)
    monkeypatch.delitem(namespace, "Gemma3ProcessorKwargs", raising = False)

    def _removed(name):
        raise AttributeError(name)

    monkeypatch.setitem(namespace, "__getattr__", _removed)
    assert gemma._resolve_Gemma3ProcessorKwargs() is ProcessingKwargs
    # And the patch itself still installs: reading the removed name would make it bail out.
    processor = processing_gemma3.Gemma3Processor

    def upstream_call(self, images = None, text = None, videos = None, audio = None, **kwargs):
        raise AssertionError("upstream __call__ was not replaced")

    monkeypatch.setattr(processor, "__call__", upstream_call)
    with warnings.catch_warnings(record = True) as record:
        warnings.simplefilter("always")
        gemma.patch_Gemma3Processor()
    assert _future_warnings(record, "ProcessorKwargs") == [], [str(w.message) for w in record]
    call = processor.__call__
    assert call is not upstream_call
    assert "unsloth_zoo" in (getattr(call, "__module__", "") or "")


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


_MODELOPT = {"quant_method": "modelopt", "quant_algo": "NVFP4"}


@pytest.fixture
def quantizer_log():
    """Capture the quantizer logger at WARNING whatever TRANSFORMERS_VERBOSITY / earlier tests set."""
    auto_logger = logging.getLogger("transformers.quantizers.auto")
    capture = _Capture()
    level = auto_logger.level
    auto_logger.setLevel(logging.WARNING)
    auto_logger.addHandler(capture)
    try:
        yield auto_logger, capture
    finally:
        auto_logger.removeHandler(capture)
        auto_logger.setLevel(level)


def _unknown(messages):
    return [m for m in messages if m.startswith("Unknown quantization type")]


def test_planner_quantization_probe_is_quiet_and_loader_check_is_not(quantizer_log):
    from transformers.quantizers import AutoHfQuantizer
    from unsloth_zoo import device_map_planner as planner

    auto_logger, capture = quantizer_log
    # transformers' own check (what from_pretrained runs) decides the case and must still warn.
    supported = bool(AutoHfQuantizer.supports_quant_method(dict(_MODELOPT)))
    loader_messages = list(capture.messages)
    if supported:
        pytest.skip("this transformers knows modelopt, so neither path warns")
    assert _unknown(loader_messages), loader_messages
    del capture.messages[:]
    # The planner gives the same answer, quietly.
    assert planner._quantization_method_is_known(dict(_MODELOPT)) is False
    assert _unknown(capture.messages) == [], capture.messages
    assert not any(isinstance(f, planner._DropUnknownQuantizationWarning) for f in auto_logger.filters)


def test_planner_probe_does_not_silence_another_thread(quantizer_log, monkeypatch):
    # A from_pretrained in another thread, overlapping the sizing probe, keeps its warning.
    import threading

    from transformers.quantizers import AutoHfQuantizer
    from unsloth_zoo import device_map_planner as planner

    auto_logger, capture = quantizer_log
    inside, release = threading.Event(), threading.Event()

    def blocking_probe(quantization_config):
        auto_logger.warning("Unknown quantization type, got modelopt (from the probe)")
        inside.set()
        assert release.wait(30)
        return False

    monkeypatch.setattr(AutoHfQuantizer, "supports_quant_method", staticmethod(blocking_probe))
    answers = []
    probe = threading.Thread(
        target = lambda: answers.append(planner._quantization_method_is_known(dict(_MODELOPT)))
    )
    probe.start()
    try:
        assert inside.wait(30)
        auto_logger.warning("Unknown quantization type, got modelopt (from the loader)")
    finally:
        release.set()
        probe.join(30)
    assert answers == [False]
    assert _unknown(capture.messages) == ["Unknown quantization type, got modelopt (from the loader)"]
    assert not any(isinstance(f, planner._DropUnknownQuantizationWarning) for f in auto_logger.filters)


_GiB = 1024 ** 3


def _plan(notes = ()):
    from unsloth_zoo.device_map_planner import DeviceMapPlan

    return DeviceMapPlan(
        device_map = {"model.embed_tokens": 0, "model.layers.0": 0, "model.layers.1": 1, "lm_head": 1},
        head_module = "lm_head",
        head_device = 1,
        headroom_bytes = 1 * _GiB,
        budgets = {0: 12 * _GiB, 1: 11 * _GiB},
        raw_budgets = {0: 14 * _GiB, 1: 14 * _GiB},
        weight_bytes = {0: int(7.26 * _GiB), 1: int(6.86 * _GiB)},
        activation_reserve_bytes = 2 * _GiB,
        total_weight_bytes = int(14.12 * _GiB),
        no_split_module_classes = ["Inkling2LoopDecoderLayer"],
        notes = list(notes),
    )


def test_plan_summary_is_one_line():
    plan = _plan(notes = [
        "Bnb4BitHfQuantizer.adjust_max_memory lowered the budget of every device to 90%",
        "sub-model towers ['model.vision_tower']: placed whole on one device",
    ])
    assert plan.summary() == (
        "Unsloth: placed the model over 2 GPUs: cuda:0 7.26 GiB, cuda:1 6.86 GiB "
        "(lm_head on cuda:1). UNSLOTH_ENABLE_LOGGING=1 shows the full placement."
    )
    # The full table is unchanged and still has everything.
    table = plan.describe()
    for detail in ("head headroom", "activation reserve", "no_split classes", "budget", "note: Bnb4Bit"):
        assert detail in table


def test_plan_summary_keeps_notes_that_warn_about_the_load():
    risks = [
        "load transient: 1x the largest merged tensor kept free per card while loading (cuda:0=1.00 GiB); "
        "the allocator reserves 2x to 3x one merged tensor while merging, so loading may still run out of memory",
        "sub-model towers ['model.vision_tower']: no plan keeps their embeddings together, so they are split "
        "per unit; a tower whose forward reads a child's weight directly may hit a device mismatch",
    ]
    lines = _plan(notes = ["tied embeddings: model.embed_tokens pinned with the head"] + risks).summary().splitlines()
    assert lines[1:] == [f"Unsloth: note: {r}" for r in risks]


def test_plan_summary_skips_empty_devices():
    plan = _plan()
    plan.weight_bytes = {0: 3 * _GiB, 1: 0, 2: 2 * _GiB}
    plan.head_device = 2
    assert plan.summary().startswith(
        "Unsloth: placed the model over 2 GPUs: cuda:0 3.00 GiB, cuda:2 2.00 GiB (lm_head on cuda:2)."
    )


def test_plan_summary_wording_for_one_or_no_gpu():
    plan = _plan()
    plan.weight_bytes = {0: 3 * _GiB, 1: 0}
    plan.head_device = 0
    assert plan.summary().startswith(
        "Unsloth: placed the model on 1 GPU: cuda:0 3.00 GiB (lm_head on cuda:0)."
    )
    plan.weight_bytes = {0: 0, 1: 0}
    line = plan.summary().splitlines()[0]
    assert line == "Unsloth: placed no weights on a GPU. UNSLOTH_ENABLE_LOGGING=1 shows the full placement."


def test_activation_checkpointing_read_without_deprecated_alias():
    import types

    from transformers import LlamaConfig, LlamaForCausalLM
    from unsloth_zoo.utils import is_activation_checkpointing

    model = LlamaForCausalLM(LlamaConfig(
        hidden_size = 16, intermediate_size = 32, num_hidden_layers = 1,
        num_attention_heads = 2, num_key_value_heads = 2, vocab_size = 32,
    ))
    assert is_activation_checkpointing(model) is False
    model.gradient_checkpointing_enable()
    with warnings.catch_warnings(record = True) as record:
        warnings.simplefilter("always")
        assert is_activation_checkpointing(model) is True
    assert not [w for w in record if "is_gradient_checkpointing" in str(w.message)]
    # Old name only (transformers < 5.23, TRL wrappers), then the default.
    assert is_activation_checkpointing(types.SimpleNamespace(is_gradient_checkpointing = True)) is True
    assert is_activation_checkpointing(object()) is False


def test_no_zoo_site_reads_the_deprecated_model_alias():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "unsloth_zoo"
    offenders = [
        str(p.relative_to(root)) for p in root.rglob("*.py")
        if '"is_gradient_checkpointing", False' in p.read_text(errors = "ignore")
    ]
    assert offenders == []

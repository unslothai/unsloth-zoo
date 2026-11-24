# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Load-time warnings a user cannot act on stay out of the default log.

* The Gemma 3 processor patch read `Gemma3ProcessorKwargs`, which transformers 5.23 turned
  into a deprecated alias (FutureWarning on every access, removed in 5.29), so every
  `import unsloth` printed it, Gemma or not.
* The device-map planner's sizing probe printed transformers' "Unknown quantization type"
  for the repo's config, which Inkling 2 loads never hand to transformers.
"""

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
    # The patch still installed its __call__.
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


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


_MODELOPT = {"quant_method": "modelopt", "quant_algo": "NVFP4"}


def test_planner_quantization_probe_is_quiet_and_loader_check_is_not():
    from transformers.quantizers import AutoHfQuantizer
    from unsloth_zoo import device_map_planner as planner

    auto_logger = logging.getLogger("transformers.quantizers.auto")
    capture = _Capture()
    auto_logger.addHandler(capture)
    try:
        known = planner._quantization_method_is_known(dict(_MODELOPT))
        probe_messages = list(capture.messages)
        # transformers' own check (what from_pretrained runs) still warns: nothing is hidden there.
        AutoHfQuantizer.supports_quant_method(dict(_MODELOPT))
    finally:
        auto_logger.removeHandler(capture)
    if known:
        pytest.skip("this transformers knows modelopt, so neither path warns")
    assert not [m for m in probe_messages if m.startswith("Unknown quantization type")], probe_messages
    assert any(m.startswith("Unknown quantization type") for m in capture.messages[len(probe_messages):])
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

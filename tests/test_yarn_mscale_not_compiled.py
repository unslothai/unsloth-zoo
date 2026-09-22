# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""YaRN mscale helpers called from MLA attention __init__ must stay uncompiled.

Under meta init a compiled float helper returns a meta tensor and `.item()` fails.
"""

import glob
import importlib
import inspect
import math
import os

import pytest
import torch

from unsloth_zoo.compiler import DISABLED_KEYWORDS

_HELPERS = ("yarn_get_mscale", "yarn_apply_mscale")


def _modeling_files_defining_helpers():
    # Discovered rather than listed so new MLA models are covered automatically.
    import transformers.models

    root = os.path.dirname(transformers.models.__file__)
    for path in sorted(glob.glob(os.path.join(root, "*", "modeling_*.py"))):
        try:
            with open(path, encoding = "utf-8") as file:
                text = file.read()
        except OSError:
            continue
        if any(f"def {name}(" in text for name in _HELPERS):
            model_dir = os.path.basename(os.path.dirname(path))
            yield model_dir, os.path.splitext(os.path.basename(path))[0]


def _shipped_helpers():
    found = []
    for model_dir, module_name in _modeling_files_defining_helpers():
        try:
            module = importlib.import_module(f"transformers.models.{model_dir}.{module_name}")
        except Exception:
            continue
        for name in _HELPERS:
            function = getattr(module, name, None)
            if callable(function):
                found.append((model_dir, name, function))
    return found


def _disabled(source):
    return any(keyword in source for keyword in DISABLED_KEYWORDS)


def test_shipped_yarn_helpers_are_disabled():
    helpers = _shipped_helpers()
    if not helpers:
        pytest.skip("this transformers has none of the MLA modeling files")
    missed = [
        f"{model_type}.{name}"
        for model_type, name, function in helpers
        if not _disabled(inspect.getsource(function))
    ]
    assert not missed, f"would be compiled and fail under meta init: {missed}"


def test_callers_are_not_disabled():
    # Keywords match the definitions only, so callers keep compiling.
    try:
        from transformers.models.deepseek_v3 import modeling_deepseek_v3 as m
    except Exception:
        pytest.skip("no deepseek_v3 in this transformers")
    assert not _disabled(inspect.getsource(m.DeepseekV3Attention.forward))


def _yarn_get_mscale(scale = 1, mscale = 1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def test_compiled_scalar_helper_fails_under_meta_init():
    # Eager is fine under meta init, compiled is not.
    with torch.device("meta"):
        assert _yarn_get_mscale(40.0, 1.0) == pytest.approx(1.3688879454113936)
        compiled = torch.compile(_yarn_get_mscale, fullgraph = True, dynamic = True)
        try:
            value = compiled(40.0, 1.0)
        except Exception:
            return
    if value != pytest.approx(1.3688879454113936):
        pytest.fail(f"compiled helper returned {value!r} under meta init")

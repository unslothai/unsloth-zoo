# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""MLA attention __init__ calls the YaRN mscale helpers under meta init, where a compiled float helper fails."""

import glob
import importlib
import inspect
import os

import pytest

from unsloth_zoo.compiler import DISABLED_KEYWORDS

_HELPERS = ("yarn_get_mscale", "yarn_apply_mscale")


def _modeling_files_defining_helpers():
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
        pytest.skip(reason = "this transformers ships no yarn_get_mscale / yarn_apply_mscale")
    missed = [
        f"{model_type}.{name}"
        for model_type, name, function in helpers
        if not _disabled(inspect.getsource(function))
    ]
    assert not missed, f"would be compiled and fail under meta init: {missed}"


def test_callers_are_not_disabled():
    try:
        from transformers.models.deepseek_v3 import modeling_deepseek_v3 as m
    except Exception:
        pytest.skip(reason = "this transformers has no deepseek_v3 to check a caller against")
    assert not _disabled(inspect.getsource(m.DeepseekV3Attention.forward))


# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

"""Generated compiled-cache modules must import `transformers` when they use it.

Copied upstream modeling source can reference the `transformers` package
itself: gemma4's `modeling_gemma4.py` uses `transformers.PreTrainedConfig`
inside a copied class body. The import block `create_new_function` prepends
only covers torch/typing/functools plus `from {model_location} import ...`,
so the emitted module referenced `transformers` without importing it, and a
direct load of the compiled cache died with
`name 'transformers' is not defined` (unslothai/unsloth#13245).
"""

import pytest

from unsloth_zoo import compiler

PROBE_NAME = "UnslothCompileTransformersImportProbe"
GEMMA4_LOCATION = "transformers.models.gemma4.modeling_gemma4"

SOURCE_WITH_TRANSFORMERS_REF = (
    "class _ZooProbeConfig(transformers.PreTrainedConfig):\n"
    "    model_type = 'gemma4'\n"
)
SOURCE_WITHOUT_TRANSFORMERS_REF = (
    "class _ZooProbeConfig:\n    model_type = 'gemma4'\n"
)


def _emit(tmp_path, monkeypatch, source):
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_LOCATION", str(tmp_path))
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_USE_TEMP", False)
    compiler.create_new_function(
        PROBE_NAME,
        source,
        GEMMA4_LOCATION,
        [],
        overwrite=True,
    )
    return tmp_path / f"{PROBE_NAME}.py"


@pytest.mark.parametrize(
    ("source", "expect_import"),
    [
        pytest.param(SOURCE_WITH_TRANSFORMERS_REF, True, id="references-transformers"),
        pytest.param(SOURCE_WITHOUT_TRANSFORMERS_REF, False, id="no-transformers-ref"),
    ],
)
def test_emitted_module_imports_transformers_iff_referenced(
    tmp_path, monkeypatch, source, expect_import
):
    path = _emit(tmp_path, monkeypatch, source)
    emitted = path.read_text()
    assert ("import transformers\n" in emitted.split("class _ZooProbeConfig")[0]) is expect_import

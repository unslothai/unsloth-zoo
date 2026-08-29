# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Regression tests for patch_gpt_oss optional-import cascade (#1119)."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GPT_OSS_PY = REPO_ROOT / "unsloth_zoo" / "temporary_patches" / "gpt_oss.py"
UTILS_PY = REPO_ROOT / "unsloth_zoo" / "temporary_patches" / "utils.py"


def _patch_gpt_oss_source() -> str:
    module = ast.parse(GPT_OSS_PY.read_text())
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "patch_gpt_oss":
            return ast.get_source_segment(GPT_OSS_PY.read_text(), node) or ""
    raise AssertionError("patch_gpt_oss not found")


def test_skip_patch_helper_exists_and_warns_once():
    utils_src = UTILS_PY.read_text()
    assert "def skip_patch(" in utils_src
    assert "logger.warning_once" in utils_src


def test_patch_gpt_oss_optional_imports_do_not_return_raise_error():
    """Optional tail imports must use skip_patch, not return raise_error."""
    src = _patch_gpt_oss_source()
    tree = ast.parse(src)

    optional_targets = {
        "triton_kernels.routing.routing",
        "triton_kernels.matmul_ogs",
        "transformers.integrations.tensor_parallel.shard_and_distribute_module",
        "transformers.integrations.mxfp4._replace_with_mxfp4_linear",
        "triton_kernels",
        "transformers.quantizers.quantizer_mxfp4.is_kernels_available",
    }

    class ReturnRaiseErrorFinder(ast.NodeVisitor):
        def __init__(self):
            self.matches: list[str] = []

        def visit_Return(self, node: ast.Return) -> None:
            if not isinstance(node.value, ast.Call):
                return
            call = node.value
            if not (isinstance(call.func, ast.Name) and call.func.id == "raise_error"):
                return
            if call.args and isinstance(call.args[0], ast.Constant):
                target = call.args[0].value
                if target in optional_targets:
                    self.matches.append(target)
            self.generic_visit(node)

    finder = ReturnRaiseErrorFinder()
    finder.visit(tree)
    assert finder.matches == [], (
        "patch_gpt_oss still abandons later patches via "
        f"return raise_error(...) for: {finder.matches}"
    )


def test_patch_gpt_oss_scopes_tensor_parallel_to_load_and_swizzle():
    src = _patch_gpt_oss_source()
    assert "load_and_swizzle_ready" in src
    assert "shard_and_distribute_module = None" in src
    assert "skip_patch(" in src

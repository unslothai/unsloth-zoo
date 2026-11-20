"""Globals a compiled module gets from its header, for tests that exec generated code by hand.

The rewriter emits calls to helpers the compiled module's header imports (unsloth_fused_ce_loss,
unsloth_fsdp_unshard, ...). A test that execs the generated source into a hand-built namespace has to
supply the same names; listing them by hand misses the next one the header adds, which is how
unsloth_fsdp_unshard (#1766) turned every such harness into a NameError. Read the header instead.
"""

from __future__ import annotations

import ast
import importlib


_LOSS_MODULES = ("unsloth_zoo.loss_utils", "unsloth_zoo.fused_losses", "unsloth_zoo.compiler")


def fused_loss_header_names() -> dict:
    """Every name the compiled header imports from the loss modules, as the real objects. Callers
    override the ones they fake (unsloth_fused_ce_loss, HAS_CUT_CROSS_ENTROPY, ...) afterwards."""
    from unsloth_zoo import compiler

    names = {}
    for node in ast.walk(ast.parse(compiler._disabled_sdpa_code)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(_LOSS_MODULES):
            module = importlib.import_module(node.module)
            for alias in node.names:
                names[alias.asname or alias.name] = getattr(module, alias.name)
    if "unsloth_fused_ce_loss" not in names:
        raise AssertionError("the compiled header no longer imports unsloth_fused_ce_loss")
    return names

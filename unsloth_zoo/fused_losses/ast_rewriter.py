# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""AST-level rewriter for the canonical HF lm_head / loss_function triplet.

What we match (structural, ignores whitespace, comments, docstrings):

    <LOGITS_NAME> = self.<HEAD>(<HIDDEN_EXPR>)
    ...
    if labels is not None:
        <LOGITS_NAME2> = self.loss_function(
            <LOGITS_NAME>,                  # or logits=<LOGITS_NAME>
            labels,                         # or labels=labels
            vocab_size=<VOCAB_EXPR>,        # or 3rd positional
            **<KWARGS_NAME>,
        )

What we rewrite to:

    if labels is not None:
        <LOSS_NAME> = unsloth_fused_lm_head_loss(
            <HIDDEN_EXPR>, self.<HEAD>, labels,
            vocab_size=<VOCAB_EXPR>, **<KWARGS_NAME>,
        )
        <LOGITS_NAME> = EMPTY_LOGITS
    else:
        <LOGITS_NAME> = self.<HEAD>(<HIDDEN_EXPR>)
        <LOSS_NAME>   = None

So the bf16 logits and the fp32 cast both disappear in the labels branch;
generation (labels is None) is untouched.

Robustness notes:

- We tolerate `.float()` / `.contiguous()` / `[slice]` wrappers around
  the `self.<HEAD>(...)` call by walking the RHS for any descendant Call
  whose func is `self.<X>`.
- We tolerate both keyword and positional `vocab_size` in the
  `loss_function` call (some VLMs pass it positionally).
- We do NOT rewrite forwards that lack the canonical triplet. Those
  classes fall through to `_UNMATCHED` and the LOSS_MAPPING patch
  remains the backstop.
"""

from __future__ import annotations

__all__ = [
    "rewrite_forward_source",
    "TripletCapture",
]

import ast
import textwrap
from dataclasses import dataclass


@dataclass
class TripletCapture:
    head_attr: str            # e.g. "lm_head"
    hidden_expr: ast.AST      # the expression passed into self.<head_attr>(...)
    logits_name: str          # the name the lm_head output was bound to
    loss_name: str            # the name the loss was bound to
    vocab_expr: ast.AST | None
    kwargs_name: str | None   # name of the **kwargs param passed to loss_function
    lm_head_assign_idx: int   # index in the function body of the `logits = self.lm_head(...)` stmt
    if_block_idx: int         # index of the `if labels is not None:` stmt
    loss_init_idx: int | None # index of the `loss = None` stmt that we delete (may be None)


def _is_self_attr_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    )


def _find_inner_self_call(value: ast.AST) -> ast.Call | None:
    """First Call descendant whose func is `self.<X>`. Lets us see through
    `.float()` / `[slice]` / `.contiguous()` chains."""
    for node in ast.walk(value):
        if _is_self_attr_call(node):
            return node
    return None


def _find_loss_function_call(if_block: ast.If) -> ast.Call | None:
    for n in ast.walk(if_block):
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "self"
            and n.func.attr == "loss_function"
        ):
            return n
    return None


def _find_loss_assign_target(if_block: ast.If, call: ast.Call) -> str | None:
    for n in ast.walk(if_block):
        if isinstance(n, ast.Assign) and n.value is call and len(n.targets) == 1:
            tgt = n.targets[0]
            if isinstance(tgt, ast.Name):
                return tgt.id
    return None


def _capture(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> TripletCapture | None:
    body = fn.body

    if_idx = None
    if_node = None
    for i, stmt in enumerate(body):
        if not isinstance(stmt, ast.If):
            continue
        t = stmt.test
        if not (isinstance(t, ast.Compare)
                and isinstance(t.left, ast.Name) and t.left.id == "labels"
                and len(t.ops) == 1 and isinstance(t.ops[0], ast.IsNot)
                and isinstance(t.comparators[0], ast.Constant)
                and t.comparators[0].value is None):
            continue
        # Must contain a self.loss_function call
        if _find_loss_function_call(stmt) is None:
            continue
        if_idx = i
        if_node = stmt
        break
    if if_node is None:
        return None

    loss_call = _find_loss_function_call(if_node)
    if loss_call is None:
        return None
    loss_name = _find_loss_assign_target(if_node, loss_call) or "loss"

    # Locate logits-bearing arg: first positional or `logits=` kw.
    logits_name = None
    if loss_call.args:
        a0 = loss_call.args[0]
        if isinstance(a0, ast.Name):
            logits_name = a0.id
    if logits_name is None:
        for kw in loss_call.keywords:
            if kw.arg == "logits" and isinstance(kw.value, ast.Name):
                logits_name = kw.value.id
                break
    if logits_name is None:
        return None

    # vocab_size: keyword preferred, else 3rd positional.
    vocab_expr = None
    for kw in loss_call.keywords:
        if kw.arg == "vocab_size":
            vocab_expr = kw.value
            break
    if vocab_expr is None and len(loss_call.args) >= 3:
        vocab_expr = loss_call.args[2]

    # **kwargs unpack
    kwargs_name = None
    for kw in loss_call.keywords:
        if kw.arg is None and isinstance(kw.value, ast.Name):
            kwargs_name = kw.value.id
            break

    # Find the lm_head assignment for logits_name (walking upward from if_idx).
    head_attr = None
    hidden_expr = None
    lm_head_assign_idx = None
    for j in range(if_idx - 1, -1, -1):
        stmt = body[j]
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            continue
        tgt = stmt.targets[0]
        if not (isinstance(tgt, ast.Name) and tgt.id == logits_name):
            continue
        inner = _find_inner_self_call(stmt.value)
        if inner is None:
            # The logits-bearing name is re-assigned by a non-lm_head
            # expression (e.g. `logits = logits * self.logit_scale` for
            # Cohere). Removing the original lm_head call would leave the
            # rebinding referencing an undefined `logits`. Bail out and
            # let the LOSS_MAPPING patch handle this class.
            return None
        head_attr = inner.func.attr
        if not inner.args:
            return None
        hidden_expr = inner.args[0]
        lm_head_assign_idx = j
        break
    if head_attr is None or hidden_expr is None or lm_head_assign_idx is None:
        return None

    # Optional `loss = None` between the lm_head assign and the if block.
    loss_init_idx = None
    for j in range(lm_head_assign_idx + 1, if_idx):
        stmt = body[j]
        if (isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == loss_name
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is None):
            loss_init_idx = j
            break

    return TripletCapture(
        head_attr=head_attr,
        hidden_expr=hidden_expr,
        logits_name=logits_name,
        loss_name=loss_name,
        vocab_expr=vocab_expr,
        kwargs_name=kwargs_name,
        lm_head_assign_idx=lm_head_assign_idx,
        if_block_idx=if_idx,
        loss_init_idx=loss_init_idx,
    )


def _build_replacement(cap: TripletCapture) -> list[ast.stmt]:
    """Build the AST nodes for the rewritten labels-branch / else-branch."""
    head_attr = cap.head_attr
    logits = cap.logits_name
    loss = cap.loss_name
    vocab = ast.unparse(cap.vocab_expr) if cap.vocab_expr is not None else "None"
    kwargs_unpack = f", **{cap.kwargs_name}" if cap.kwargs_name else ""
    hidden_src = ast.unparse(cap.hidden_expr)

    template = textwrap.dedent(f"""
        if labels is not None:
            {loss} = unsloth_fused_lm_head_loss(
                {hidden_src}, self.{head_attr}, labels,
                vocab_size={vocab}{kwargs_unpack},
            )
            {logits} = EMPTY_LOGITS
        else:
            {logits} = self.{head_attr}({hidden_src})
            {loss} = None
    """).strip()
    return ast.parse(template).body


def rewrite_forward_source(source: str) -> tuple[str | None, TripletCapture | None]:
    """Rewrite a forward function source string.

    Returns (new_source, capture) on success, (None, None) if the canonical
    triplet wasn't found (and the caller should leave the class alone).
    """
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return (None, None)
    if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return (None, None)
    fn = tree.body[0]
    cap = _capture(fn)
    if cap is None:
        return (None, None)

    new_block = _build_replacement(cap)
    body = fn.body
    # Replace `[lm_head_assign .. if_block]` (inclusive of both, plus an
    # optional `loss = None` initialiser in between) with the new branch.
    delete_indices = {cap.lm_head_assign_idx, cap.if_block_idx}
    if cap.loss_init_idx is not None:
        delete_indices.add(cap.loss_init_idx)
    new_body = []
    inserted = False
    for i, stmt in enumerate(body):
        if i in delete_indices:
            if not inserted:
                new_body.extend(new_block)
                inserted = True
            continue
        new_body.append(stmt)
    fn.body = new_body
    # Strip decorators -- they belong to the original module's globals
    # (e.g. @auto_docstring, @can_return_tuple) and we exec in a namespace
    # that may not have them visible. The decorators only add docstring
    # sugar / tuple-return handling and are not needed for the runtime
    # forward we install.
    fn.decorator_list = []
    ast.fix_missing_locations(tree)
    return (ast.unparse(tree), cap)

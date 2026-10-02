# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""AST-level rewriter for the canonical HF lm_head / loss_function triplet.

Matches ``<LOGITS> = self.<HEAD>(<HIDDEN>)`` (optional .float()/.contiguous()/
.to() wrappers, and a ``* scale`` / ``/ scale`` the fused kernel reapplies as
``logit_scale_multiply`` / ``logit_scale_divide``) followed by an ``if labels is
not None:`` branch computing ``self.loss_function(<LOGITS>, labels,
vocab_size=..., **kwargs)``. Any other wrapper bails out of the rewrite.

Rewrites the labels branch to two paths:
  - default (``UNSLOTH_RETURN_LOGITS`` unset): ``unsloth_fused_lm_head_loss``
    (no lm_head matmul on the hot path); ``logits = EMPTY_LOGITS``.
  - opt-in (``UNSLOTH_RETURN_LOGITS=1``): materialise logits once via the
    original head expr, then route loss through ``self.loss_function`` on
    them. One lm_head matmul total.

The else (generation) branch keeps the original RHS. Forwards missing the
triplet fall through; the LOSS_MAPPING sweep is the backstop.

``UNSLOTH_RETURN_HIDDEN_STATES`` is handled instead in the compiler-rewritten
forward (``unsloth_zoo/compiler.py``), which overrides the AST forward for
the supported ``*ForCausalLM`` classes used with that env var.
"""

from __future__ import annotations

__all__ = [
    "rewrite_forward_source",
    "TripletCapture",
]

import ast
import textwrap
from dataclasses import dataclass, field


@dataclass
class TripletCapture:
    head_attr: str            # e.g. "lm_head"
    hidden_expr: ast.AST      # the expression passed into self.<head_attr>(...)
    logits_rhs_src: str       # ast.unparse of the original `logits = ...` RHS
    logits_name: str          # the name the lm_head output was bound to
    loss_name: str            # the name the loss was bound to
    vocab_expr: ast.AST | None
    kwargs_name: str | None   # name of the **kwargs param passed to loss_function
    extra_loss_kws: list      # [(name, ast.AST), ...] explicit kwargs beyond vocab_size
    lm_head_assign_idx: int   # index in the function body of the `logits = self.lm_head(...)` stmt
    if_block_idx: int         # index of the `if labels is not None:` stmt
    loss_init_idx: int | None # index of the `loss = None` stmt that we delete (may be None)
    # [(name, ast.AST)] post-head scaling; fused call only. Defaulted + last for existing callers.
    scale_kws: list = field(default_factory = list)
    pre_stmts: list = field(default_factory = list)   # casts before the loss call in the labels branch
    post_stmts: list = field(default_factory = list)  # casts after it
    softcap_idx: int | None = None
    softcap_stmt: ast.stmt | None = None
    softcap_expr: ast.AST | None = None


def _is_self_attr_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    )


def _contains_self_attr_call(node: ast.AST) -> bool:
    return any(_is_self_attr_call(n) for n in ast.walk(node))


# Wrappers the fused kernel makes redundant (it casts + accumulates in fp32 itself).
_TRANSPARENT_METHODS = frozenset(("float", "contiguous", "to"))


def _unwrap_logits_rhs(value: ast.AST):
    """``(self.<HEAD>(...) call, scale_kws)``, or None for any wrapper besides ``* s`` / ``/ s``
    (peeling blindly dropped it and trained on wrongly scaled logits)."""
    scale_kws: list = []
    node = value
    while True:
        if _is_self_attr_call(node):
            return node, scale_kws
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _TRANSPARENT_METHODS
        ):
            node = node.func.value
            continue
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
            left_has, right_has = (
                _contains_self_attr_call(node.left),
                _contains_self_attr_call(node.right),
            )
            if left_has and not right_has:
                kw = "logit_scale_multiply" if isinstance(node.op, ast.Mult) else "logit_scale_divide"
                scale, node = node.right, node.left
            elif right_has and not left_has and isinstance(node.op, ast.Mult):
                # `scale / head` is not a scaling.
                kw, scale, node = "logit_scale_multiply", node.left, node.right
            else:
                return None
            # A repeat would emit a duplicate keyword argument.
            if any(name == kw for name, _ in scale_kws):
                return None
            scale_kws.append((kw, scale))
            continue
        return None


def _find_loss_function_call(if_block: ast.If) -> ast.Call | None:
    # Only direct body statements: nested ifs inside the labels branch would
    # be dropped by the wholesale rewrite.
    for stmt in if_block.body:
        if isinstance(stmt, ast.Assign):
            v = stmt.value
            if (
                isinstance(v, ast.Call)
                and isinstance(v.func, ast.Attribute)
                and isinstance(v.func.value, ast.Name)
                and v.func.value.id == "self"
                and v.func.attr == "loss_function"
            ):
                return v
    return None


def _is_loss_function_assign(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and isinstance(stmt.value.func.value, ast.Name)
        and stmt.value.func.value.id == "self"
        and stmt.value.func.attr == "loss_function"
    )


def _is_cast(stmt: ast.stmt, names) -> bool:
    """`x = x.float()` / `x = x.to(...)` / `x = x.contiguous()` for x in names."""
    if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id in names):
        return False
    v = stmt.value
    return (
        isinstance(v, ast.Call)
        and isinstance(v.func, ast.Attribute)
        and v.func.attr in _TRANSPARENT_METHODS
        and isinstance(v.func.value, ast.Name)
        and v.func.value.id == stmt.targets[0].id
    )


_TANH_SPELLINGS = frozenset(("torch.tanh", "nn.functional.tanh", "torch.nn.functional.tanh", "F.tanh"))


def _softcap_cap(stmt: ast.stmt, logits_name: str):
    """The `cap` of `logits = tanh(logits / cap) * cap`, else None."""
    if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == logits_name):
        return None
    v = stmt.value
    if not (isinstance(v, ast.BinOp) and isinstance(v.op, ast.Mult)
            and isinstance(v.left, ast.Call) and len(v.left.args) == 1 and not v.left.keywords
            and ast.unparse(v.left.func) in _TANH_SPELLINGS):
        return None
    inner = v.left.args[0]
    if not (isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Div)
            and isinstance(inner.left, ast.Name) and inner.left.id == logits_name):
        return None
    cap = inner.right
    if ast.dump(cap) != ast.dump(v.right):
        return None
    if any(isinstance(n, ast.Name) and n.id == logits_name for n in ast.walk(cap)):
        return None
    return cap


def _find_loss_assign_target(if_block: ast.If, call: ast.Call) -> str | None:
    for stmt in if_block.body:
        if isinstance(stmt, ast.Assign) and stmt.value is call and len(stmt.targets) == 1:
            tgt = stmt.targets[0]
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

    # Reject non-trivial label branches: anything beyond a single
    # `loss = self.loss_function(...)` is lost by the wholesale rewrite (e.g.
    # CSM auxiliary depth-decoder loss).
    if if_node.orelse:
        return None
    loss_positions = [k for k, s in enumerate(if_node.body) if _is_loss_function_assign(s)]
    if len(loss_positions) != 1:
        return None
    loss_k = loss_positions[0]
    loss_assign = if_node.body[loss_k]
    loss_call = loss_assign.value
    loss_name = loss_assign.targets[0].id

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
    # transformers 4.x wraps the call in casts (`labels.to(logits.device)`, `logits.float()`,
    # `loss.to(hidden_states.dtype)`); any other statement would be dropped by the rewrite.
    pre_stmts = if_node.body[:loss_k]
    post_stmts = if_node.body[loss_k + 1:]
    if not all(_is_cast(s, ("labels", logits_name)) for s in pre_stmts):
        return None
    if not all(_is_cast(s, (loss_name, logits_name)) for s in post_stmts):
        return None

    # Labels arg must be literally the plain `labels` name; aliased labels
    # (e.g. CSM `labels=backbone_labels`) need bespoke handling.
    labels_arg = None
    if len(loss_call.args) >= 2:
        if isinstance(loss_call.args[1], ast.Name):
            labels_arg = loss_call.args[1].id
    for kw in loss_call.keywords:
        if kw.arg == "labels":
            if isinstance(kw.value, ast.Name):
                labels_arg = kw.value.id
            else:
                return None
    if labels_arg != "labels":
        return None

    # vocab_size: keyword preferred, else 3rd positional.
    vocab_expr = None
    for kw in loss_call.keywords:
        if kw.arg == "vocab_size":
            vocab_expr = kw.value
            break
    if vocab_expr is None and len(loss_call.args) >= 3:
        vocab_expr = loss_call.args[2]

    # **kwargs unpack + any explicit kwargs beyond {logits, labels, vocab_size}.
    kwargs_name = None
    extra_loss_kws: list = []
    for kw in loss_call.keywords:
        if kw.arg is None:
            if isinstance(kw.value, ast.Name):
                kwargs_name = kw.value.id
            continue
        if kw.arg in ("logits", "labels", "vocab_size"):
            continue
        extra_loss_kws.append((kw.arg, kw.value))

    # Find the lm_head assignment for logits_name (walking upward from if_idx).
    head_attr = None
    hidden_expr = None
    logits_rhs_src = None
    lm_head_assign_idx = None
    scale_kws: list = []
    skipped_softcap = False
    for j in range(if_idx - 1, -1, -1):
        stmt = body[j]
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            continue
        tgt = stmt.targets[0]
        if not (isinstance(tgt, ast.Name) and tgt.id == logits_name):
            continue
        if not skipped_softcap and _softcap_cap(stmt, logits_name) is not None:
            skipped_softcap = True
            continue
        unwrapped = _unwrap_logits_rhs(stmt.value)
        if unwrapped is None:
            # Non-lm_head reassign (Cohere's `logits * self.logit_scale`) or unreproducible wrapper.
            return None
        inner, scale_kws = unwrapped
        head_attr = inner.func.attr
        if not inner.args:
            return None
        hidden_expr = inner.args[0]
        logits_rhs_src = ast.unparse(stmt.value)
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

    # Bail if any statement between lm_head and the labels-if touches logits
    # (e.g. Gemma3 final_logit_softcapping): it would run on EMPTY_LOGITS in
    # the labels branch, so fused loss would see un-softcapped logits.
    # The one exception is a single `logits = tanh(logits / cap) * cap` (RecurrentGemma), which the
    # kernel reapplies as logit_softcapping.
    softcap_idx = None
    softcap_expr = None
    for j in range(lm_head_assign_idx + 1, if_idx):
        if j == loss_init_idx:
            continue
        if softcap_idx is None:
            softcap_expr = _softcap_cap(body[j], logits_name)
            if softcap_expr is not None:
                softcap_idx = j
                continue
        for n in ast.walk(body[j]):
            if isinstance(n, ast.Name) and n.id == logits_name:
                return None

    return TripletCapture(
        head_attr=head_attr,
        hidden_expr=hidden_expr,
        logits_rhs_src=logits_rhs_src,
        logits_name=logits_name,
        loss_name=loss_name,
        vocab_expr=vocab_expr,
        kwargs_name=kwargs_name,
        extra_loss_kws=extra_loss_kws,
        scale_kws=scale_kws,
        lm_head_assign_idx=lm_head_assign_idx,
        if_block_idx=if_idx,
        loss_init_idx=loss_init_idx,
        pre_stmts=pre_stmts,
        post_stmts=post_stmts,
        softcap_idx=softcap_idx,
        softcap_stmt=body[softcap_idx] if softcap_idx is not None else None,
        softcap_expr=softcap_expr,
    )


def _build_replacement(cap: TripletCapture) -> list[ast.stmt]:
    """Build the AST nodes for the rewritten labels-branch / else-branch."""
    head_attr = cap.head_attr
    logits = cap.logits_name
    loss = cap.loss_name
    vocab = ast.unparse(cap.vocab_expr) if cap.vocab_expr is not None else "None"
    extra = "".join(
        f", {name}={ast.unparse(value)}" for name, value in cap.extra_loss_kws
    )
    # Fused call only (other branches re-evaluate the RHS); an explicit same-name kwarg wins.
    already = {name for name, _ in cap.extra_loss_kws}
    scale_extra = "".join(
        f", {name}={ast.unparse(value)}"
        for name, value in cap.scale_kws
        if name not in already
    )
    kwargs_unpack = f", **{cap.kwargs_name}" if cap.kwargs_name else ""
    hidden_src = ast.unparse(cap.hidden_expr)
    logits_rhs = cap.logits_rhs_src or f"self.{head_attr}({hidden_src})"

    # Labels branch. Default: fused kernel, logits = EMPTY_LOGITS (no lm_head
    # matmul). UNSLOTH_RETURN_LOGITS=1: run the full lm_head matmul once and
    # route loss through self.loss_function on those logits (avoids the double
    # matmul of fused-kernel + separate logits_rhs).
    if cap.softcap_expr is not None and "logit_softcapping" not in already:
        scale_extra += f", logit_softcapping={ast.unparse(cap.softcap_expr)}"
    if not (cap.pre_stmts or cap.post_stmts or cap.softcap_expr is not None):
        template = textwrap.dedent(f"""
            if labels is not None:
                if os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '1':
                    {logits} = {logits_rhs}
                    {loss} = self.loss_function({logits}, labels, vocab_size={vocab}{extra}{kwargs_unpack})
                else:
                    {loss} = unsloth_fused_lm_head_loss(
                        {hidden_src}, self.{head_attr}, labels,
                        vocab_size={vocab}{extra}{scale_extra}{kwargs_unpack},
                    )
                    {logits} = EMPTY_LOGITS
            else:
                {logits} = {logits_rhs}
                {loss} = None
        """).strip()
        return ast.parse(template).body

    # Unfused branches replay the original statements verbatim; the fused one keeps only loss casts
    # (the kernel moves labels to the head's device and accumulates in fp32 itself).
    softcap = [ast.unparse(cap.softcap_stmt)] if cap.softcap_stmt is not None else []
    pre = [ast.unparse(s) for s in cap.pre_stmts]
    post = [ast.unparse(s) for s in cap.post_stmts]
    loss_casts = [ast.unparse(s) for s in cap.post_stmts if s.targets[0].id == loss]
    unfused = [
        f"{logits} = {logits_rhs}", *softcap, *pre,
        f"{loss} = self.loss_function({logits}, labels, vocab_size={vocab}{extra}{kwargs_unpack})",
        *post,
    ]
    fused = [
        f"{loss} = unsloth_fused_lm_head_loss({hidden_src}, self.{head_attr}, labels, "
        f"vocab_size={vocab}{extra}{scale_extra}{kwargs_unpack})",
        *loss_casts,
        f"{logits} = EMPTY_LOGITS",
    ]
    def ind(lines, n):
        return "\n".join(" " * n + x for x in lines)

    template = (
        "if labels is not None:\n"
        "    if os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '1':\n"
        f"{ind(unfused, 8)}\n"
        f"    else:\n{ind(fused, 8)}\n"
        f"else:\n{ind([f'{logits} = {logits_rhs}', *softcap, f'{loss} = None'], 4)}"
    )
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
    delete_indices = {cap.lm_head_assign_idx, cap.if_block_idx}
    if cap.loss_init_idx is not None:
        delete_indices.add(cap.loss_init_idx)
    # A softcap may read names bound after the head (`cap = self.config...`): insert at the if.
    insert_at = min(delete_indices)
    if cap.softcap_idx is not None:
        delete_indices.add(cap.softcap_idx)
        insert_at = cap.if_block_idx
    new_body = []
    for i, stmt in enumerate(body):
        if i == insert_at:
            new_body.extend(new_block)
        if i in delete_indices:
            continue
        new_body.append(stmt)
    fn.body = new_body
    # @can_return_tuple carries return_dict=False semantics and must survive;
    # strip only the docstring-only decorators below.
    _DROP_DECORATORS = {
        "auto_docstring",
        "add_start_docstrings",
        "add_start_docstrings_to_model_forward",
        "add_end_docstrings",
        "replace_return_docstrings",
    }
    fn.decorator_list = [
        d for d in fn.decorator_list if _decorator_name(d) not in _DROP_DECORATORS
    ]
    ast.fix_missing_locations(tree)
    return (ast.unparse(tree), cap)


def _decorator_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return None

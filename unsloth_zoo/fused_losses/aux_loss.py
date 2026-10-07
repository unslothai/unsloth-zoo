# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Gradient accumulation weighting for MoE router auxiliary losses.

MoE heads add `loss += self.router_aux_loss_coef * aux_loss` after the LM loss (also spelt
`loss = loss + aux_loss`, `loss = loss + z_loss + aux_loss` for Switch / NLLB-MoE router terms and
`loss = loss + self.z_loss_coefficient * z_loss` for Bamba). Once num_items_in_batch reaches the
forward, the LM term of every micro-batch is sum / N_total and Trainer no longer divides by the
accumulation steps, but the extra term is still a per micro-batch mean added in full, so over G
micro-batches it carries G times its full-batch weight. Scaling it by this micro-batch's share of
the counted target tokens (local / N_total) makes the weights sum to one over the accumulated batch
(and over ranks: Trainer multiplies the loss by the world size and DDP averages). Without a count,
nothing changes.

The share is token weighted, which is exact for a term that averages over the same targets and an
approximation otherwise: a router aux loss averages routing over every token (padding included),
Bamba's z-loss over every position. Either way the accumulated total weight is one, as it is
without accumulation.
"""

__all__ = [
    "unsloth_ga_scale_aux_loss",
    "rewrite_aux_loss_ga",
    "unscaled_extra_loss_terms",
]

import ast
import re
import textwrap

import torch


def _local_count(labels, attention_mask, loss_kwargs, model):
    # The batch counter's own per micro-batch definition (loss_utils.count_batch_items), so the
    # shares of the micro-batches it summed add up to exactly one.
    from unsloth_zoo.loss_utils import count_batch_items, counts_unshifted_labels, num_items_labels_convention
    convention = num_items_labels_convention(model) if model is not None else None
    unshifted = (convention == "unshifted") if convention is not None else counts_unshifted_labels(model)
    if not (isinstance(attention_mask, torch.Tensor) and attention_mask.shape == labels.shape):
        attention_mask = None
    packed_seq_lengths = loss_kwargs.get("packed_seq_lengths", None) if isinstance(loss_kwargs, dict) else None
    # The packed-boundary drop indexes by data, which a compiled forward cannot trace; there the
    # boundaries a collator left unmasked (at most documents - 1 per micro-batch) are counted.
    packed = not torch.compiler.is_compiling()
    count, _, _ = count_batch_items(labels, attention_mask, None, packed_seq_lengths,
                                    unshifted = unshifted, packed = packed)
    return count


def unsloth_ga_scale_aux_loss(aux_loss, labels, attention_mask = None, loss_kwargs = None,
                              num_items_in_batch = None, model = None):
    # All Unsloth Zoo code licensed under LGPLv3
    """`aux_loss * local / num_items_in_batch`, or `aux_loss` unchanged without a usable count.

    The count is the explicit `num_items_in_batch` argument (a forward naming it as a parameter),
    else `loss_kwargs["num_items_in_batch"]`, else the `n_items` alias. `model` (the head) says
    whether labels are counted shifted or unshifted, as the batch counter decided.
    """
    n_items = num_items_in_batch
    if isinstance(loss_kwargs, dict):
        if n_items is None: n_items = loss_kwargs.get("num_items_in_batch", None)
        if n_items is None: n_items = loss_kwargs.get("n_items", None)
    if n_items is None or not isinstance(aux_loss, torch.Tensor) \
            or not isinstance(labels, torch.Tensor) or labels.ndim < 1:
        return aux_loss
    if not isinstance(n_items, torch.Tensor):
        if n_items <= 0: return aux_loss
        n_items = torch.tensor(n_items)
    n_items = n_items.to(device = aux_loss.device, dtype = torch.float32)
    # A DataParallel replica gets a one-element slice of the repeated count.
    if n_items.ndim > 0: n_items = n_items.reshape(-1)[0]
    local = _local_count(labels, attention_mask, loss_kwargs, model)
    local = local.to(device = aux_loss.device, dtype = torch.float32)
    # fp32 ratio: a half precision count overflows past 65504 tokens. A zero total leaves the term
    # as is, without a host sync.
    ratio = torch.where(n_items > 0, local / n_items.clamp_min(1), torch.ones_like(local))
    return (aux_loss.float() * ratio).to(aux_loss.dtype)
pass


# Coefficients of a router / z-loss term: router_aux_loss_coef, aux_loss_coef, router_z_loss_coef,
# z_loss_coefficient (Bamba), moe_loss_weight (4.x DBRX).
_COEF = re.compile(r"(aux_loss_coef|z_loss_coef|z_loss_coefficient|moe_loss_weight)$")
# The term itself: aux_loss, z_loss, router_aux_loss, ...
_TERM = re.compile(r"(^|_)(aux_loss|z_loss)$")


def _is_aux_coef(node):
    # self.router_aux_loss_coef / self.aux_loss_coef / self.config.text_config.router_aux_loss_coef
    return isinstance(node, ast.Attribute) and _COEF.search(node.attr) is not None


def _aux_operand(node):
    # `aux_loss` or `aux_loss.to(...)` (any *_aux_loss / *_z_loss name).
    if isinstance(node, ast.Name):
        return _TERM.search(node.id) is not None
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "to"
        and isinstance(node.func.value, ast.Name)
        and _TERM.search(node.func.value.id) is not None
    )


def _is_scaled(node):
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "unsloth_ga_scale_aux_loss" for n in ast.walk(node))


def _added_terms(node, loss_name = "loss"):
    """The terms a statement adds to `loss`: `loss += a [+ b]` or `loss = loss + a [+ b]`, else None."""
    def flatten(n):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            return flatten(n.left) + flatten(n.right)
        return [n]
    if (isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add)
            and isinstance(node.target, ast.Name) and node.target.id == loss_name):
        return flatten(node.value)
    if (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name) and node.targets[0].id == loss_name):
        terms = flatten(node.value)
        if len(terms) >= 2 and isinstance(terms[0], ast.Name) and terms[0].id == loss_name:
            return terms[1:]
    return None


def _term_operand(term):
    """The node to wrap in one added term (`coef * aux` -> aux, bare `aux` -> itself), else None."""
    if isinstance(term, ast.BinOp) and isinstance(term.op, ast.Mult):
        if _is_aux_coef(term.left) and _aux_operand(term.right): return term.right
        if _is_aux_coef(term.right) and _aux_operand(term.left): return term.left
        return None
    return term if _aux_operand(term) else None


def unscaled_extra_loss_terms(source):
    """Does a forward add anything to `loss` that is not GA weighted? True also when unparsable.

    A rewrite that makes the CE count-aware must not leave such a term behind: once unsloth sees the
    count consumed, Trainer stops dividing by the accumulation steps and that term would carry G
    times its weight.
    """
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        terms = _added_terms(node)
        if terms is not None and not all(_is_scaled(t) for t in terms):
            return True
    return False


def rewrite_aux_loss_ga(source):
    """Wrap the router / z-loss terms a forward adds to `loss` in `unsloth_ga_scale_aux_loss`.

    Handles `loss += <coef> * aux_loss[.to(..)]` and `loss = loss + <term> [+ <term>]` where each
    term is `<coef> * <aux>` or a bare `*aux_loss` / `*z_loss` name; a statement with any other term
    is left whole. Text splice over the original source (formatting kept, so later regex passes
    still match). Needs a `labels` parameter and somewhere the count arrives: a **kwargs parameter
    or an explicit `num_items_in_batch` one; otherwise the source is returned unchanged.
    Without a count the helper returns the term as is, so the stock loss is unchanged.
    """
    if "unsloth_ga_scale_aux_loss" in source or ("aux_loss" not in source and "z_loss" not in source):
        return source
    lines = source.splitlines(keepends = True)
    dedented = textwrap.dedent(source)
    try:
        tree = ast.parse(dedented)
    except SyntaxError:
        return source
    fn = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if fn is None:
        return source
    positional = fn.args.posonlyargs + fn.args.args
    params = {a.arg for a in positional + fn.args.kwonlyargs}
    explicit_count = "num_items_in_batch" in params
    if "labels" not in params or (fn.args.kwarg is None and not explicit_count):
        return source
    kwargs_name = fn.args.kwarg.arg if fn.args.kwarg is not None else "None"
    mask = "attention_mask" if "attention_mask" in params else "None"
    extra = ""
    if explicit_count:
        extra += ", num_items_in_batch = num_items_in_batch"
    if positional and positional[0].arg == "self":
        extra += ", model = self"

    # Columns: dedent removed the same prefix from every non-blank line.
    def indent(text):
        line = next((l for l in text.splitlines() if l.strip()), "")
        return len(line) - len(line.lstrip())

    shift = indent(source) - indent(dedented)

    spans = []
    for node in ast.walk(fn):
        terms = _added_terms(node)
        if not terms:
            continue
        operands = [_term_operand(t) for t in terms]
        if any(o is None for o in operands):
            continue
        spans.extend(operands)
    if not spans:
        return source

    # Absolute offsets into the original text, replaced back to front.
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def offset(lineno, col):
        return starts[lineno - 1] + col + shift

    out = source
    for operand in sorted(spans, key = lambda n: (n.lineno, n.col_offset), reverse = True):
        a = offset(operand.lineno, operand.col_offset)
        b = offset(operand.end_lineno, operand.end_col_offset)
        segment = out[a:b]
        out = out[:a] + (
            f"unsloth_ga_scale_aux_loss({segment}, labels, {mask}, {kwargs_name}{extra})"
        ) + out[b:]
    try:
        ast.parse(textwrap.dedent(out))
    except SyntaxError:
        return source
    return out
pass

"""CPU/CI regression guards for the Gemma 4 dtype fixes (PR #925).

These lock in the three failures we traced:

  1. Audio merge: upstream `Gemma4Model.forward` cast `audio_features.to(device)`
     without dtype (transformers #45192), while image/video aligned dtype on the
     statement that produced the features ->
     `masked_scatter_: expected self and source to have same dtypes`.
     Fixed upstream in transformers 5.15.0; the guards below still cover the
     5.5.0-5.14.1 window, which unsloth continues to support.
  2. Forced-float32 PLE: fp32 residual into the fp16 PLE Linears (unsloth-zoo #866
     forced fp32 but left PLE inputs uncast) -> `mat1 and mat2 ... float != Half`.
  3. Cross-path NameError: the eager patch rewrote calls to one helper name while
     the compiler emitted another; a compile after the eager patch called an
     undefined helper.

Everything here is CPU-only (dtype-check errors raise on CPU), no GPU required.
The real-source canaries import the real transformers gemma4 module and skip
cleanly when it is absent; when present (transformers >= 5.5.0, as on CI) they
catch upstream source drift that would silently disable the patch.
"""
import ast
import collections
import inspect
import re

import pytest
import torch

from unsloth_zoo.temporary_patches import gemma4 as g4p
from unsloth_zoo.temporary_patches import gemma4_float32 as g4f
from unsloth_zoo.temporary_patches.gemma4_float32 import _unsloth_gemma4_ple_cast_input
from unsloth_zoo import compiler as C
from unsloth_zoo.compiler import _GEMMA4_PLE_CAST_HELPER, fix_gemma4_forced_float32_ple_dtype

CANONICAL_HELPER = "_unsloth_gemma4_ple_cast_input"

try:
    from transformers.models.gemma4 import modeling_gemma4 as real_gemma4
    HAS_GEMMA4 = True
except Exception:
    real_gemma4 = None
    HAS_GEMMA4 = False

requires_gemma4 = pytest.mark.skipif(not HAS_GEMMA4, reason="transformers gemma4 not installed")


# ---------------------------------------------------------------------------
# Guard 3: the eager and compiler PLE helpers must share ONE name (string/AST).
# This is what broke as the cross-path NameError. Any reintroduction of a second
# spelling fails here instantly, with no GPU / model needed.
# ---------------------------------------------------------------------------
def _ple_cast_identifiers(source):
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*_ple_cast_input", source))


def test_ple_cast_helper_has_exactly_one_name_across_eager_and_compiler():
    eager_src = inspect.getsource(g4f)
    compiler_src = inspect.getsource(C)
    names = _ple_cast_identifiers(eager_src) | _ple_cast_identifiers(compiler_src)
    assert names == {CANONICAL_HELPER}, (
        f"PLE cast helper name diverged across eager/compiler paths: {sorted(names)}. "
        f"Both must use exactly {CANONICAL_HELPER!r} or a compile after the eager "
        f"patch will call an undefined helper (NameError)."
    )


def test_compiler_generated_helper_defines_the_name_it_calls():
    # The name the compiler REWRITE inserts must equal the name the appended
    # helper DEFINES. Extract both from the compiler source directly.
    on_flag = _run_ple_rewrite("Gemma4TextDecoderLayer")
    called = set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*_ple_cast_input)\(", on_flag))
    defined = set(re.findall(r"def ([A-Za-z_][A-Za-z0-9_]*_ple_cast_input)\b", _GEMMA4_PLE_CAST_HELPER))
    assert called and defined and called == defined == {CANONICAL_HELPER}, (
        f"compiler rewrite calls {called} but helper defines {defined}"
    )


def _run_ple_rewrite(module, monkey_env="1"):
    import os
    prev = os.environ.get("UNSLOTH_FORCE_FLOAT32")
    os.environ["UNSLOTH_FORCE_FLOAT32"] = monkey_env
    try:
        src = (
            "def forward(self, hidden_states):\n"
            "    hidden_states = self.per_layer_input_gate(hidden_states)\n"
            "    return self.per_layer_projection(hidden_states)\n"
        )
        return fix_gemma4_forced_float32_ple_dtype(src, module)
    finally:
        if prev is None:
            os.environ.pop("UNSLOTH_FORCE_FLOAT32", None)
        else:
            os.environ["UNSLOTH_FORCE_FLOAT32"] = prev


# ---------------------------------------------------------------------------
# Guard: eager and compiler helper implementations must stay behaviourally
# identical (both copies are hand-maintained; catch silent drift between them).
# ---------------------------------------------------------------------------
def _compiler_helper_callable():
    ns = {}
    exec(_GEMMA4_PLE_CAST_HELPER, ns)
    return ns[CANONICAL_HELPER]


class _ModWithWeight:
    def __init__(self, weight): self.weight = weight
class _WeightStub:
    pass


def _dtype_cases():
    cases = []
    for name, dt in (("f16", torch.float16), ("bf16", torch.bfloat16), ("f32", torch.float32)):
        w = torch.nn.Linear(3, 3, bias=False, dtype=dt)
        cases.append((f"dense_{name}", w))
    q = torch.zeros(3, 3, dtype=torch.uint8); q.quant_state = object()
    cases.append(("bnb4bit", _ModWithWeight(q)))
    cases.append(("int8", _ModWithWeight(torch.zeros(3, 3, dtype=torch.int8))))
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is not None:
        s = _WeightStub(); s.dtype = fp8
        cases.append(("fp8", _ModWithWeight(s)))
    cases.append(("no_dtype", _ModWithWeight(_WeightStub())))
    return cases


@pytest.mark.parametrize("label,module", _dtype_cases())
def test_eager_and_compiler_helpers_agree(label, module):
    eager = _unsloth_gemma4_ple_cast_input
    comp = _compiler_helper_callable()
    x = torch.randn(2, 3, dtype=torch.float32)
    e, c = eager(module, x), comp(module, x)
    assert (e is x) == (c is x), f"{label}: identity divergence"
    assert e.dtype == c.dtype, f"{label}: dtype divergence {e.dtype} vs {c.dtype}"


# ---------------------------------------------------------------------------
# Guard 1 & 2 behaviour (pure CPU torch): prove the fix actually resolves the
# two dtype crashes. No gemma4 needed - exercises the exact failing op.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dst_dt", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("src_dt", [torch.float16, torch.bfloat16, torch.float32])
def test_audio_dtype_cast_fixes_masked_scatter(dst_dt, src_dt):
    dst = torch.zeros(1, 4, dtype=dst_dt)
    mask = torch.ones(1, 4, dtype=torch.bool)
    src = torch.arange(4, dtype=src_dt).reshape(1, 4)
    if dst_dt != src_dt:
        # BEFORE the fix (device-only cast) the merge raises.
        with pytest.raises(RuntimeError, match="same dtype"):
            dst.masked_scatter(mask, src.to(dst.device))
    # AFTER the fix (device + dtype) it works and preserves values.
    out = dst.masked_scatter(mask, src.to(dst.device, dst.dtype))
    assert out.dtype == dst_dt
    torch.testing.assert_close(out, src.to(dst_dt))


@pytest.mark.parametrize("w_dt", [torch.float16, torch.bfloat16])
def test_ple_helper_fixes_fp32_into_lowprec_linear(w_dt):
    lin = torch.nn.Linear(3, 3, bias=False, dtype=w_dt)
    x = torch.randn(2, 3, dtype=torch.float32, requires_grad=True)
    # BEFORE: fp32 activation into low-precision weight raises.
    with pytest.raises(RuntimeError, match="same dtype"):
        lin(x)
    # AFTER: helper casts the input to the weight dtype; forward + backward work.
    out = lin(_unsloth_gemma4_ple_cast_input(lin, x))
    assert out.dtype == w_dt
    out.float().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_ple_helper_never_casts_to_fp8_or_packed_weight():
    x = torch.randn(2, 3, dtype=torch.float32)
    # fp8 packed storage must be left alone (fp8 kernels scale internally).
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is not None:
        s = _WeightStub(); s.dtype = fp8
        assert _unsloth_gemma4_ple_cast_input(_ModWithWeight(s), x) is x
    # bnb 4-bit: uint8 + quant_state -> unchanged (never cast activation to uint8).
    q = torch.zeros(3, 3, dtype=torch.uint8); q.quant_state = object()
    assert _unsloth_gemma4_ple_cast_input(_ModWithWeight(q), x) is x


# ---------------------------------------------------------------------------
# Real-source drift canaries: the highest-value "never again" guards. They read
# the REAL transformers gemma4 module FILE FROM DISK (no mutation, no GPU) and
# assert the exact call sites the unsloth patches target are still present.
#
# Reading from disk is deliberate: importing unsloth_zoo monkeypatches
# Gemma4Model.forward (the KV-carrier wrapper) and can poison linecache for the
# PLE methods, so inspect.getsource() on the class attributes returns the
# unsloth-wrapped source, not upstream. The on-disk file is pristine upstream.
#
# If a future transformers reshapes a call site so the patch can no longer match
# AND upstream has not fixed the dtype itself, these FAIL loudly instead of the
# patch silently no-op-ing and the dtype crash quietly returning.
# ---------------------------------------------------------------------------
def _gemma4_modeling_source():
    import pathlib
    return pathlib.Path(inspect.getsourcefile(real_gemma4)).read_text()


# Upstream is free to spell the dtype alignment wherever it likes: at the merge
# call (`audio_features.to(inputs_embeds.device, inputs_embeds.dtype)`, tf 5.15+)
# or on the statement that produced the features
# (`image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)`,
# tf 5.5-5.14, or `... = torch.cat(image_features, dim=0).to(..., inputs_embeds.dtype)`,
# tf 5.15). A substring anchor on one spelling breaks on cosmetic refactors while
# staying blind to a real regression written in another spelling, so the canaries
# below trace dtype ALIGNMENT structurally instead: find every
# `inputs_embeds.masked_scatter` that consumes the features, and require of each an
# `inputs_embeds.dtype` cast either inside the merge argument or on the last
# assignment to that name that DOMINATES the merge (same block or an enclosing one,
# so a cast stranded in a branch the merge does not run under does not count), plus
# a receiver that is still AT `inputs_embeds.dtype` and a result that is actually
# used (masked_scatter is out-of-place).
def _is_attr(node, owner, attr):
    """`<owner>.<attr>` as an AST node, e.g. `inputs_embeds.dtype`."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Name)
        and node.value.id == owner
    )


def _is_embeds_tensor(node):
    """`inputs_embeds` itself, as the argument of the tensor-to-tensor `.to()`."""
    return isinstance(node, ast.Name) and node.id == "inputs_embeds"


def _casts_to_inputs_embeds_dtype(node):
    """True if `node` contains a `.to(...)` that lands on `inputs_embeds`' dtype.

    Two spellings count. The explicit one names the dtype
    (`features.to(inputs_embeds.device, inputs_embeds.dtype)`), and the tensor
    overload names the tensor: `Tensor.to(other)` is documented as "returns a
    Tensor with same torch.dtype and torch.device as the Tensor other", so
    `features.to(inputs_embeds)` aligns dtype AND device in one call and is just
    as safe a merge source. Rejecting it would make these canaries fail on a
    healthy upstream that happened to prefer that spelling, and the failure would
    claim nobody aligns the dtype at all. Only the FIRST positional argument can
    be the tensor overload (`to(other, non_blocking=..., copy=...)`); torch has
    no `other=` keyword, so keywords are not considered for it.
    """
    for sub in ast.walk(node):
        if not (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "to"
        ):
            continue
        values = list(sub.args) + [kw.value for kw in sub.keywords]
        if any(_is_attr(v, "inputs_embeds", "dtype") for v in values):
            return True
        if sub.args and _is_embeds_tensor(sub.args[0]):
            return True
    return False


# Methods that may sit between `inputs_embeds` and `.masked_scatter(...)` without
# changing the destination dtype. Anything outside this set (`.float()`, `.half()`,
# `.type(...)`, an unknown helper) can retype the destination, which makes an
# `inputs_embeds.dtype` source cast prove nothing: the merge then has an fp32
# destination and a low-precision source and raises.
_DTYPE_PRESERVING_METHODS = frozenset({
    "clone", "contiguous", "detach", "cpu", "cuda",
    "view", "reshape", "flatten", "squeeze", "unsqueeze",
    "expand", "expand_as", "transpose", "permute", "narrow",
})


def _is_dtype_bearing(node):
    """Can `node` retype a tensor? (`torch.float32`, `other.dtype`, ...)"""
    if isinstance(node, ast.Attribute) and node.attr == "dtype":
        return not _is_attr(node, "inputs_embeds", "dtype")
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "torch"
    ):
        # `torch.float32` / `torch.bfloat16` / `torch.long` ...
        return True
    return False


def _is_device_expr(node):
    """`x.device`, `"cuda"`, `torch.device(...)` - device without a dtype."""
    if isinstance(node, ast.Attribute) and node.attr == "device":
        return True
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "device"
    ):
        return True
    return False


def _to_call_preserves_dtype(call):
    """Is this `.to(...)` device-only (or a no-op `inputs_embeds.dtype` cast)?

    `inputs_embeds.to(language_model_inputs.device)` is blip_2's real spelling and
    keeps the dtype; `.to(torch.float32)`, `.to(other.dtype)` and the tensor
    overload `.to(other_tensor)` do not.
    """
    for arg in call.args:
        if not (_is_device_expr(arg) or _is_attr(arg, "inputs_embeds", "dtype")):
            return False
    for kw in call.keywords:
        if kw.arg in ("non_blocking", "copy", "memory_format"):
            continue
        if kw.arg == "device" and _is_device_expr(kw.value):
            continue
        if kw.arg == "dtype" and _is_attr(kw.value, "inputs_embeds", "dtype"):
            continue
        return False
    return True


def _receiver_preserves_embeds_dtype(node):
    """Is the merge destination `inputs_embeds` still at `inputs_embeds.dtype`?

    The whole trace is "the source is cast to `inputs_embeds.dtype`, so the merge
    is safe", which only holds while the DESTINATION is at that dtype too. The
    receiver may be `inputs_embeds` itself or a chain of dtype-preserving
    transformations of it (upstream blip_2 really does write
    `inputs_embeds.to(language_model_inputs.device).masked_scatter(...)`), but
    `inputs_embeds.float().masked_scatter(mask, feat.to(inputs_embeds.dtype))`
    merges a low-precision source into an fp32 destination and raises.
    """
    while True:
        if isinstance(node, ast.Name):
            return node.id == "inputs_embeds"
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            return False
        method = node.func.attr
        if method == "to":
            if not _to_call_preserves_dtype(node):
                return False
        elif method in _DTYPE_PRESERVING_METHODS:
            values = list(node.args) + [kw.value for kw in node.keywords]
            if any(_is_dtype_bearing(v) for v in values):
                return False
        else:
            return False
        node = node.func.value


_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _walk_own_scope(root, descend = True):
    """`ast.walk` restricted to the code `root`'s own scope executes.

    A nested `def` / `class` / `lambda` is a separate scope with its own local
    names, so a merge written inside a helper defined in `forward` is not
    `forward`'s merge (the helper may never be invoked), and a cast in `forward`
    says nothing about a same-named parameter of that helper. The nested
    statement itself is still yielded - only its body is skipped.
    """
    if isinstance(root, _SCOPE_NODES) and not descend:
        yield root
        return
    todo = collections.deque([root])
    while todo:
        node = todo.popleft()
        yield node
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _SCOPE_NODES):
                yield child
            else:
                todo.append(child)


def _gemma4_model_forward_node(src):
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Gemma4Model":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "forward":
                    return item
    return None


def _child_blocks(stmt):
    """Every nested statement list of `stmt` (`if`/`else`, loop bodies, `try`)."""
    blocks = []
    for field in ("body", "orelse", "finalbody"):
        value = getattr(stmt, field, None)
        if isinstance(value, list) and value and all(isinstance(s, ast.stmt) for s in value):
            blocks.append(value)
    for handler in getattr(stmt, "handlers", None) or []:
        blocks.append(handler.body)
    return blocks


def _enclosing_blocks(forward, target):
    """`[(block, index)]` from the function body inwards to the stmt holding `target`.

    Only these statement lists run unconditionally *relative to the merge*. A cast
    nested in some other branch may never execute on the path that reaches the
    merge, so it cannot prove alignment.
    """
    chain = []

    def holds(node):
        return any(n is target for n in _walk_own_scope(node, descend = False))

    def descend(block):
        for index, stmt in enumerate(block):
            if not holds(stmt):
                continue
            chain.append((block, index))
            for inner in _child_blocks(stmt):
                if any(holds(s) for s in inner):
                    descend(inner)
                    break
            return

    descend(forward.body)
    return chain


def _is_dtype_guard(test, feature):
    """Exactly `<feature>.dtype != inputs_embeds.dtype` (either operand order).

    Only that comparison makes the untaken branch aligned BY DEFINITION. Any
    other test mentioning the feature dtype (`... == inputs_embeds.dtype`,
    `... != torch.bfloat16`, or the mismatch test `and`-ed with something else)
    leaves a reachable path on which the feature stays un-cast, so a cast under
    it proves nothing and must not count as unconditional alignment.
    """
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
        return False
    if not isinstance(test.ops[0], ast.NotEq):
        return False
    operands = [test.left, test.comparators[0]]
    return any(
        _is_attr(operands[i], feature, "dtype")
        and _is_attr(operands[1 - i], "inputs_embeds", "dtype")
        for i in (0, 1)
    )


def _bound_names(target):
    """Every plain name an assignment target binds.

    Recurses through destructuring targets, because
    `image_features, feature_lens = self.pack_image_features(...)` (the
    llava_next / llava_onevision / llava_next_video spelling) rebinds
    `image_features` just as much as a bare `image_features = ...` does. Missing
    it lets an earlier aligned assignment keep proving alignment for a tensor
    that has since been replaced by an unpacked, possibly mismatched one.

    `image_features[0] = ...` (Subscript) and `self.image_features = ...`
    (Attribute) are deliberately NOT name rebindings: the name still refers to
    the same object afterwards.
    """
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for elt in target.elts for name in _bound_names(elt)]
    if isinstance(target, ast.Starred):
        return _bound_names(target.value)
    return []


_ASSIGN_NODES = (ast.Assign, ast.AnnAssign, ast.AugAssign)
_LOOP_NODES = (ast.For, ast.AsyncFor)
_WITH_NODES = (ast.With, ast.AsyncWith)
# Every construct that can rebind a plain name in the code's OWN scope. An
# assignment statement is only the most common one: `for image_features in
# image_batches:`, `with self.autocast() as image_features:` and
# `if (image_features := self.pack(image_features)) is None:` all replace the
# tensor the name refers to just as completely. Counting only assignments lets an
# earlier aligned assignment keep proving alignment for a value that one of these
# has since overwritten, and the replacement reaches masked_scatter un-cast.
# (Comprehensions bind in their own scope and so cannot affect `forward`'s name.)
_BINDING_NODES = _ASSIGN_NODES + _LOOP_NODES + _WITH_NODES + (ast.NamedExpr,)


def _binding_pairs(node):
    """`[(target, value)]` this binding node establishes.

    The `value` is the expression whose dtype the bound name ends up carrying, so
    the same `_casts_to_inputs_embeds_dtype` question can be asked of every
    binding form: an assignment's RHS, a loop's iterable (the target takes its
    elements), and a `with` item's context expression.
    """
    if isinstance(node, ast.Assign):
        return [(target, node.value) for target in node.targets]
    if isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
        return [(node.target, getattr(node, "value", None))]
    if isinstance(node, _LOOP_NODES):
        return [(node.target, node.iter)]
    if isinstance(node, _WITH_NODES):
        return [
            (item.optional_vars, item.context_expr)
            for item in node.items
            if item.optional_vars is not None
        ]
    return []


def _assigns_feature(stmt, feature):
    return any(
        target is not None and feature in _bound_names(target)
        for target, _ in _binding_pairs(stmt)
    )


def _binding_value(node, feature):
    """The expression `<feature>` takes its dtype from in this binding node."""
    for target, value in _binding_pairs(node):
        if target is not None and feature in _bound_names(target):
            return value
    return None


def _binding_is_aligned(node, feature):
    value = _binding_value(node, feature)
    return value is not None and _casts_to_inputs_embeds_dtype(value)


def _feature_rebindings(stmt, feature):
    """Every rebinding of `<feature>` anywhere inside `stmt`'s own scope."""
    return [
        node
        for node in _walk_own_scope(stmt, descend = False)
        if isinstance(node, _BINDING_NODES) and _assigns_feature(node, feature)
    ]


def _rebinding_alignment(stmt, feature):
    """Does `stmt` rebind `<feature>` on the path to the merge, and is it aligned?

    `None` when `stmt` does not rebind the name (so it says nothing either way).
    """
    if isinstance(stmt, _ASSIGN_NODES) and _assigns_feature(stmt, feature):
        return _binding_is_aligned(stmt, feature)
    # A bare named expression statement, `(image_features := ...)`, rebinds
    # unconditionally exactly like an assignment does.
    if (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.NamedExpr)
        and _assigns_feature(stmt.value, feature)
    ):
        return _binding_is_aligned(stmt.value, feature)
    # `if <feature>.dtype != inputs_embeds.dtype: <feature> = <feature>.to(...)`
    # is a cast-if-needed idiom (transformers spells it in blip_2 / granite_speech):
    # the untaken branch is aligned by definition, so it counts as unconditional.
    #
    # Both branches still have to be analysed as PATHS, not as a bag of
    # rebindings whose textually last one wins. `if enabled: <feature> =
    # <feature>.to(inputs_embeds.dtype)` nested inside the guard, or an aligned
    # `else` after an unaligned body, would otherwise report the whole guard
    # aligned while the mismatch path leaves the feature un-cast.
    if isinstance(stmt, ast.If) and _is_dtype_guard(stmt.test, feature):
        if _feature_rebindings(stmt, feature):
            # Body runs only on a mismatch (starts unaligned); `orelse` only when
            # the dtypes already agree (starts aligned). Both ends must be aligned.
            return (
                _sequence_alignment(stmt.body, feature, False)
                and _sequence_alignment(stmt.orelse, feature, True)
            )
    # Any OTHER compound statement that rebinds the feature is still a rebinding
    # on the path to the merge and must not be skipped, or an earlier aligned
    # assignment would keep proving alignment for a value that has since been
    # rewritten (`if enabled: image_features = image_features.float()`).
    rebindings = _feature_rebindings(stmt, feature)
    if not rebindings:
        return None
    if isinstance(stmt, _WITH_NODES):
        # A `with` body runs unconditionally, so its rebindings dominate the
        # merge exactly like top-level ones: take the last one that says
        # anything. The `as` target binds first, before the body runs, so it
        # seeds the state.
        alignment = (
            _binding_is_aligned(stmt, feature)
            if _assigns_feature(stmt, feature) else None
        )
        for inner in stmt.body:
            inner_alignment = _rebinding_alignment(inner, feature)
            if inner_alignment is not None:
                alignment = inner_alignment
        return alignment
    # Conditionally executed (`if` / `for` / `while` / `try`): the rebinding may
    # or may not run, so the state after it is "aligned before AND aligned in
    # every branch". An unaligned branch therefore reports unaligned; a branch
    # that only ever casts to `inputs_embeds.dtype` leaves the earlier verdict
    # standing (`None`) instead of overriding it. A loop's own target counts as
    # one of those rebindings, taking its dtype from the iterable.
    for rebinding in rebindings:
        if not _binding_is_aligned(rebinding, feature):
            return False
    return None


def _sequence_alignment(block, feature, initial):
    """Alignment of `<feature>` after running `block` top to bottom.

    `initial` is the state on entry to the block. Statements that say nothing
    about the feature (`_rebinding_alignment` -> `None`, which includes a
    conditional rebinding that only ever casts to `inputs_embeds.dtype`, since
    it may not run) leave the running state alone.
    """
    state = initial
    for stmt in block:
        aligned = _rebinding_alignment(stmt, feature)
        if aligned is not None:
            state = aligned
    return state


def _dominating_alignment(forward, feature, merge_arg):
    """Is the last rebinding of `<feature>` that dominates `merge_arg` dtype-aligned?"""
    best_line, best_aligned = None, False
    for block, index in _enclosing_blocks(forward, merge_arg):
        for stmt in block[:index]:
            aligned = _rebinding_alignment(stmt, feature)
            if aligned is None:
                continue
            if best_line is None or stmt.lineno > best_line:
                best_line, best_aligned = stmt.lineno, aligned
    return best_aligned


_MERGE_METHODS = ("masked_scatter", "masked_scatter_")


def _feature_merges(forward, feature):
    """Every `inputs_embeds.masked_scatter[_](mask, ...)` call using `<feature>`.

    The receiver has to REFERENCE `inputs_embeds`: a
    `scratch.masked_scatter(mask, feature)` elsewhere in `forward` is a different
    tensor and says nothing about the dtype of the embedding merge. Discovery
    stays deliberately loose here (any receiver mentioning the name) so that a
    transformed receiver such as blip_2's
    `inputs_embeds.to(language_model_inputs.device).masked_scatter(...)` is still
    recognised AS the embedding merge; whether that transformation is dtype-safe
    is then decided by `_receiver_preserves_embeds_dtype`. Dropping unrecognised
    receivers here instead would let a second, safe merge hide a dtype-changing
    one.

    The in-place spelling is accepted too (transformers writes
    `masked_scatter_` in qwen2_5_omni and blt): it merges into the same storage
    and needs the same dtype agreement.

    Nested `def` / `class` / `lambda` bodies are skipped: a merge in a helper
    `forward` never invokes is not `forward`'s merge.
    """
    calls = []
    for node in _walk_own_scope(forward):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _MERGE_METHODS
            and len(node.args) >= 2
        ):
            continue
        if not any(
            isinstance(n, ast.Name) and n.id == "inputs_embeds"
            for n in ast.walk(node.func.value)
        ):
            continue
        if any(
            isinstance(n, ast.Name) and n.id == feature for n in ast.walk(node.args[1])
        ):
            calls.append(node)
    return sorted(calls, key=lambda n: (n.lineno, n.col_offset))


def _contains(node, target):
    return any(n is target for n in ast.walk(node))


def _references(node, names):
    return any(isinstance(n, ast.Name) and n.id in names for n in ast.walk(node))


def _statements_after(forward, call):
    """The statements that execute after `call`, innermost block outwards.

    The first one yielded is the merge's OWN statement; then the rest of the
    block holding it, then the remainder of each enclosing block. That is the
    straight-line path the merged tensor has to survive to reach the return.
    """
    chain = _enclosing_blocks(forward, call)
    if not chain:
        return None
    ordered = []
    for depth, (block, index) in enumerate(reversed(chain)):
        # The innermost entry points AT the merge's statement; every outer entry
        # points at the compound statement wrapping it, which we have already
        # accounted for.
        ordered.extend(block[index if depth == 0 else index + 1:])
    return ordered


def _merge_result_is_used(forward, call):
    """Does the merge's result reach the embeddings `forward` goes on to use?

    `masked_scatter` is OUT-OF-PLACE, so the merged embedding only exists in its
    return value. Upstream has always spelled it
    `inputs_embeds = inputs_embeds.masked_scatter(...)` on 5.5.0-5.15.0, and the
    features reach the model precisely because that result lands back on
    `inputs_embeds`.

    Asking only whether the merge is a bare expression statement is too weak: it
    passes `dead = inputs_embeds.masked_scatter(...)` and
    `self.log(inputs_embeds.masked_scatter(...))`, both of which compute the
    merged embedding, drop it on the floor and leave `forward` returning the
    UNCHANGED `inputs_embeds` - the features silently never reach the model, with
    every dtype canary in this file green. So follow the result forward instead:
    it must flow, directly or through temporaries, into `inputs_embeds` or into a
    `return`. The in-place `masked_scatter_` writes through the receiver's
    storage and needs no result at all.
    """
    if call.func.attr.endswith("_"):
        return True
    statements = _statements_after(forward, call)
    if statements is None:
        return True

    tainted = set()
    for position, statement in enumerate(statements):
        for node in _walk_own_scope(statement, descend = False):
            # A `return` carrying the merge (or anything it flowed into) is the
            # merged embedding leaving `forward`.
            if isinstance(node, ast.Return) and node.value is not None:
                if (position == 0 and _contains(node.value, call)) or _references(node.value, tainted):
                    return True
            if not isinstance(node, _BINDING_NODES):
                continue
            for target, value in _binding_pairs(node):
                if target is None or value is None:
                    continue
                bound = _bound_names(target)
                if not bound:
                    continue
                carries = (
                    (position == 0 and _contains(value, call))
                    or _references(value, tainted)
                )
                if carries:
                    tainted.update(bound)
                elif node is statement:
                    # A direct, unconditional rebinding to something the merge
                    # did NOT flow into overwrites the name: the merged value is
                    # gone. A rebinding nested inside a conditional may not run,
                    # so it must not clear the taint.
                    tainted.difference_update(bound)
        if "inputs_embeds" in tainted:
            return True
    return False


def _merge_dtype_alignment(src, feature):
    """Trace `<feature>` into `inputs_embeds.masked_scatter(...)`.

    Returns (merge_found, every_merge_aligned). EVERY merge consuming the feature
    has to be aligned: one unaligned branch is one reachable dtype crash, so an
    aligned first merge must not excuse an unaligned second one.

    A merge counts as aligned when all three hold: the destination is still at
    `inputs_embeds.dtype`, the merged result is actually used, and the source
    carries an `inputs_embeds.dtype` cast at the call or on its last dominating
    assignment.
    """
    forward = _gemma4_model_forward_node(src)
    if forward is None:
        return (False, False)
    merges = _feature_merges(forward, feature)
    if not merges:
        return (False, False)
    aligned = all(
        _receiver_preserves_embeds_dtype(merge.func.value)
        and _merge_result_is_used(forward, merge)
        and (
            _casts_to_inputs_embeds_dtype(merge.args[1])
            or _dominating_alignment(forward, feature, merge)
        )
        for merge in merges
    )
    return (True, aligned)


@requires_gemma4
def test_real_gemma4_audio_merge_site_is_recognized():
    src = _gemma4_modeling_source()
    found, aligned = _merge_dtype_alignment(src, "audio_features")
    assert found, (
        "Gemma4 audio merge site drifted: no `inputs_embeds.masked_scatter(...)` "
        "consuming `audio_features` in Gemma4Model.forward. The unsloth audio patch "
        "would silently no-op and the masked_scatter dtype crash would return. Update "
        "_patch_gemma4_audio_feature_dtype_on_class (eager) and "
        "fix_gemma4_audio_feature_dtype (compiler)."
    )
    upstream_aligned = aligned
    unsloth_can_patch = C.fix_gemma4_audio_feature_dtype(src) != src
    assert upstream_aligned or unsloth_can_patch, (
        "Gemma4 audio features reach masked_scatter with no `inputs_embeds.dtype` "
        "alignment anywhere, and the unsloth rewriter no longer matches the merge "
        "site either. `masked_scatter_: expected self and source to have same dtypes` "
        "is back. Update _patch_gemma4_audio_feature_dtype_on_class (eager) and "
        "fix_gemma4_audio_feature_dtype (compiler)."
    )


@requires_gemma4
@pytest.mark.parametrize("feature", ["image_features", "video_features"])
def test_real_gemma4_image_and_video_merges_still_cast_dtype(feature):
    # Regression anchor: image/video have always aligned dtype before the merge,
    # so unsloth patches only audio. If upstream ever drops the alignment here
    # too, that is a new modality that also needs patching.
    src = _gemma4_modeling_source()
    found, aligned = _merge_dtype_alignment(src, feature)
    assert found, (
        f"No `inputs_embeds.masked_scatter(...)` consuming `{feature}` in "
        f"Gemma4Model.forward - the merge site moved and this canary is blind."
    )
    assert aligned, (
        f"Gemma4 {feature.split('_')[0]} merge lost its `inputs_embeds.dtype` "
        f"alignment: `{feature}` reaches masked_scatter without an "
        f"`inputs_embeds.dtype` cast at the call or on its last dominating "
        f"assignment. That is "
        f"the audio bug (transformers #45192) in a new modality and needs the same "
        f"treatment as _patch_gemma4_audio_feature_dtype_on_class / "
        f"fix_gemma4_audio_feature_dtype."
    )


@requires_gemma4
def test_real_gemma4_audio_eager_patch_matches_real_merge_site():
    # The canary above only proves the COMPILER rewriter (a regex over the source)
    # still fires. The eager patch is stricter: it requires the merge's source
    # argument to BE the `audio_features.to(inputs_embeds.device, ...)` call, so a
    # wrapped spelling such as
    # `audio_features.to(inputs_embeds.device).reshape(-1, inputs_embeds.shape[-1])`
    # matches the regex but not the AST matcher - the eager patch would log a drift
    # warning, no-op, and the dtype crash would come back on the non-compiled path
    # while every other guard here stayed green. Run the patch's OWN matcher over
    # the real source so that spelling fails loudly here instead.
    src = _gemma4_modeling_source()
    forward = _gemma4_model_forward_node(src)
    assert forward is not None, "Gemma4Model.forward not found in the real source"
    buggy, fixed = g4p._gemma4_audio_merge_casts(forward)
    already_aligned = not buggy and len(fixed) == 1
    patchable = len(buggy) == 1 and not fixed
    assert already_aligned or patchable, (
        f"_patch_gemma4_audio_feature_dtype_on_class no longer recognizes the real "
        f"Gemma4 audio merge site: it found {len(buggy)} device-only and {len(fixed)} "
        f"device+dtype `audio_features.to(inputs_embeds.device, ...)` merge arguments, "
        f"and it only acts on exactly one of either. The eager patch would silently "
        f"no-op (the compiler regex may still fire, which is why the other audio "
        f"canaries stay green). Update _patch_gemma4_audio_feature_dtype_on_class."
    )


@requires_gemma4
def test_real_gemma4_ple_call_sites_are_recognized():
    src = _gemma4_modeling_source()

    def _ok(attr, arg):
        raw = f"self.{attr}({arg})"
        wrapped = f"self.{attr}({CANONICAL_HELPER}(self.{attr}, {arg})"
        return (raw in src) or (wrapped in src)

    assert _ok("per_layer_model_projection", "inputs_embeds"), (
        "Gemma4TextModel.project_per_layer_inputs PLE call drifted; PLE dtype patch "
        "would silently no-op under UNSLOTH_FORCE_FLOAT32."
    )
    assert _ok("per_layer_input_gate", "hidden_states"), "per_layer_input_gate call drifted"
    assert _ok("per_layer_projection", "hidden_states"), "per_layer_projection call drifted"


@requires_gemma4
def test_real_gemma4_audio_compiler_transform_emits_dtype_cast():
    # Drive the compiler's audio rewrite against the REAL upstream source; either
    # upstream already casts dtype, or our regex still matches and inserts it.
    src = _gemma4_modeling_source()
    out = C.fix_gemma4_audio_feature_dtype(src)
    found, aligned = _merge_dtype_alignment(out, "audio_features")
    assert found and aligned, (
        "After fix_gemma4_audio_feature_dtype, `audio_features` still reaches "
        "masked_scatter with no `inputs_embeds.dtype` alignment on upstream source "
        "(regex drift or multiple matches)."
    )


def test_audio_compiler_transform_still_fixes_the_device_only_spelling():
    # transformers 5.15.0 fixed the audio merge upstream, which makes the
    # real-source test above pass without the rewriter doing anything. Keep the
    # rewriter itself under test against the historical buggy spelling so it
    # cannot silently rot while older transformers are still supported.
    buggy = (
        "        inputs_embeds = inputs_embeds.masked_scatter(\n"
        "            audio_mask.to(inputs_embeds.device), audio_features.to(inputs_embeds.device)\n"
        "        )\n"
    )
    out = C.fix_gemma4_audio_feature_dtype(buggy)
    assert "audio_features.to(inputs_embeds.device, inputs_embeds.dtype)" in out, (
        "fix_gemma4_audio_feature_dtype no longer rewrites the device-only audio "
        "cast shipped by transformers 5.5.0-5.14.1."
    )


# ---------------------------------------------------------------------------
# Tracer unit tests. The canaries above are only as good as `_merge_dtype_alignment`,
# and a hole in it is invisible: it reports GREEN on a source that crashes. These
# feed the tracer a minimal synthetic `Gemma4Model.forward` (no transformers
# needed, so they run on every supported version) and pin the verdict on the
# spellings that used to slip through.
# ---------------------------------------------------------------------------
_DEFAULT_MERGE = (
    "inputs_embeds = inputs_embeds.masked_scatter(\n"
    "    image_mask.to(inputs_embeds.device), image_features.to(inputs_embeds.device)\n"
    ")"
)


def _indented(block):
    return "\n".join(
        "            " + line if line else "" for line in block.strip("\n").split("\n")
    )


def _synthetic_forward(body, merge = _DEFAULT_MERGE):
    """Minimal `Gemma4Model.forward` with `body` between the features and `merge`."""
    return (
        "class Gemma4Model:\n"
        "    def forward(self, inputs_embeds, pixel_values, image_mask):\n"
        "        if pixel_values is not None:\n"
        "            image_features = self.get_image_features(pixel_values)\n"
        f"{_indented(body)}\n"
        f"{_indented(merge)}\n"
        "        return inputs_embeds\n"
    )


_ALIGNED_ASSIGN = "image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)"


# Every path inside a `<feature>.dtype != inputs_embeds.dtype` guard has to end
# aligned. Taking the textually last rebinding inside the guard instead reports
# aligned for a guard whose MISMATCH path leaves the feature un-cast, which is
# exactly one reachable `masked_scatter_: expected self and source to have same
# dtypes`.
_GUARD_HOLES = [
    ("cast nested under a second condition",
     "if image_features.dtype != inputs_embeds.dtype:\n"
     "    if self.config.cast_vision_features:\n"
     f"        {_ALIGNED_ASSIGN}"),
    ("unaligned body, aligned else (textually last)",
     "if image_features.dtype != inputs_embeds.dtype:\n"
     "    image_features = image_features.to(inputs_embeds.device)\n"
     "else:\n"
     f"    {_ALIGNED_ASSIGN}"),
    ("aligned cast then a conditional rebinding that undoes it",
     "if image_features.dtype != inputs_embeds.dtype:\n"
     f"    {_ALIGNED_ASSIGN}\n"
     "    if self.config.upcast_vision_features:\n"
     "        image_features = image_features.float()"),
    ("aligned else only, mismatch path rebinds nothing aligned",
     "if image_features.dtype != inputs_embeds.dtype:\n"
     "    image_features = image_features.reshape(-1, inputs_embeds.shape[-1])\n"
     "else:\n"
     f"    {_ALIGNED_ASSIGN}"),
]


@pytest.mark.parametrize(
    "label,body", _GUARD_HOLES, ids = [case[0] for case in _GUARD_HOLES],
)
def test_conditional_cast_inside_a_dtype_guard_is_not_alignment(label, body):
    found, aligned = _merge_dtype_alignment(_synthetic_forward(body), "image_features")
    assert found, f"{label}: the synthetic merge site itself was not found"
    assert not aligned, (
        f"{label}: the tracer reports `image_features` dtype-aligned, but the "
        f"dtype-mismatch path reaches masked_scatter with no `inputs_embeds.dtype` "
        f"cast. A real upstream source spelled this way would keep every canary in "
        f"this file green while the merge crashed."
    )


_GUARD_CONTROLS = [
    ("plain cast-if-needed (blip_2 / granite_speech idiom)",
     "if image_features.dtype != inputs_embeds.dtype:\n"
     f"    {_ALIGNED_ASSIGN}"),
    ("dtype-only cast-if-needed, device carried at the merge",
     "if image_features.dtype != inputs_embeds.dtype:\n"
     "    image_features = image_features.to(inputs_embeds.dtype)"),
    ("reversed operand order",
     "if inputs_embeds.dtype != image_features.dtype:\n"
     f"    {_ALIGNED_ASSIGN}"),
    ("aligned assign then a guard that only raises",
     f"{_ALIGNED_ASSIGN}\n"
     "if image_features.dtype != inputs_embeds.dtype:\n"
     "    raise ValueError('dtype mismatch')"),
    ("aligned assign then a guard that casts again",
     f"{_ALIGNED_ASSIGN}\n"
     "if image_features.dtype != inputs_embeds.dtype:\n"
     f"    {_ALIGNED_ASSIGN}"),
]


@pytest.mark.parametrize(
    "label,body", _GUARD_CONTROLS, ids = [case[0] for case in _GUARD_CONTROLS],
)
def test_cast_if_needed_guards_still_count_as_alignment(label, body):
    # Controls for the test above: these are dtype-safe on every path and must
    # NOT turn red, or the canaries start failing on a healthy upstream.
    found, aligned = _merge_dtype_alignment(_synthetic_forward(body), "image_features")
    assert found and aligned, f"{label}: dtype-safe spelling reported unaligned"


# A destructuring assignment rebinds the feature name just like a plain one.
# `image_features, feature_lens = self.pack_image_features(...)` is upstream's
# own spelling in llava_next / llava_next_video / llava_onevision; skipping it
# lets an earlier aligned assignment keep proving alignment for a tensor that
# has since been replaced.
_UNPACK_HOLES = [
    ("tuple unpack",
     f"{_ALIGNED_ASSIGN}\n"
     "image_features, feature_lens = self.pack_image_features(image_features)"),
    ("starred unpack",
     f"{_ALIGNED_ASSIGN}\n"
     "image_features, *feature_lens = self.pack_image_features(image_features)"),
    ("list-target unpack",
     f"{_ALIGNED_ASSIGN}\n"
     "[image_features, feature_lens] = self.pack_image_features(image_features)"),
    ("nested tuple unpack",
     f"{_ALIGNED_ASSIGN}\n"
     "(image_features, feature_lens), image_sizes = self.pack_image_features(image_features)"),
    ("conditional tuple unpack",
     f"{_ALIGNED_ASSIGN}\n"
     "if self.config.pack_image_features:\n"
     "    image_features, feature_lens = self.pack_image_features(image_features)"),
]


@pytest.mark.parametrize(
    "label,body", _UNPACK_HOLES, ids = [case[0] for case in _UNPACK_HOLES],
)
def test_destructuring_rebindings_of_the_feature_are_not_skipped(label, body):
    found, aligned = _merge_dtype_alignment(_synthetic_forward(body), "image_features")
    assert found, f"{label}: the synthetic merge site itself was not found"
    assert not aligned, (
        f"{label}: the tracer still credits the earlier aligned assignment even "
        f"though `image_features` was rebound by an unpack whose value carries no "
        f"`inputs_embeds.dtype` cast. The unpacked tensor reaches masked_scatter."
    )


_UNPACK_CONTROLS = [
    ("unpack whose value casts dtype",
     f"{_ALIGNED_ASSIGN}\n"
     "image_features, feature_lens = self.pack_image_features(\n"
     "    image_features.to(inputs_embeds.device, inputs_embeds.dtype))"),
    ("unpack binding other names only",
     f"{_ALIGNED_ASSIGN}\n"
     "feature_lens, image_sizes = self.pack_image_features(image_features)"),
    ("subscript and attribute targets are not name rebindings",
     f"{_ALIGNED_ASSIGN}\n"
     "image_features[0] = image_features[0]\n"
     "self.image_features = image_features"),
]


@pytest.mark.parametrize(
    "label,body", _UNPACK_CONTROLS, ids = [case[0] for case in _UNPACK_CONTROLS],
)
def test_non_rebinding_targets_do_not_break_alignment(label, body):
    # Controls for the test above.
    found, aligned = _merge_dtype_alignment(_synthetic_forward(body), "image_features")
    assert found and aligned, f"{label}: dtype-safe spelling reported unaligned"


# The whole trace is "the source is cast to `inputs_embeds.dtype`, so the merge is
# safe". That only holds while the DESTINATION is still at `inputs_embeds.dtype`.
# A receiver transformation that retypes it turns the source cast into the wrong
# cast: `inputs_embeds.float().masked_scatter(mask, image_features.to(
# inputs_embeds.dtype))` has an fp32 destination and an fp16/bf16 source and
# raises the exact error this whole file exists to prevent.
_RECEIVER_HOLES = [
    ("receiver upcast with .float()",
     "inputs_embeds = inputs_embeds.float().masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.dtype)\n"
     ")"),
    ("receiver .to(torch.float32)",
     "inputs_embeds = inputs_embeds.to(torch.float32).masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.dtype)\n"
     ")"),
    ("receiver cast to a third tensor's dtype",
     "inputs_embeds = inputs_embeds.to(image_features.dtype).masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.dtype)\n"
     ")"),
    ("receiver .to(dtype = torch.float32) by keyword",
     "inputs_embeds = inputs_embeds.to(dtype = torch.float32).masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.dtype)\n"
     ")"),
    ("receiver retyped by an unknown helper",
     "inputs_embeds = self.upcast(inputs_embeds).masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.dtype)\n"
     ")"),
]


@pytest.mark.parametrize(
    "label,merge", _RECEIVER_HOLES, ids = [case[0] for case in _RECEIVER_HOLES],
)
def test_receiver_that_changes_dtype_is_not_alignment(label, merge):
    found, aligned = _merge_dtype_alignment(
        _synthetic_forward(_ALIGNED_ASSIGN, merge = merge), "image_features",
    )
    assert found, f"{label}: the synthetic merge site itself was not found"
    assert not aligned, (
        f"{label}: the tracer reports the merge dtype-aligned, but the merge "
        f"DESTINATION is no longer at `inputs_embeds.dtype`, so the "
        f"`inputs_embeds.dtype` source cast produces a mismatched pair. With "
        f"fp16/bf16 embeddings this is one reachable `masked_scatter_: expected "
        f"self and source to have same dtypes` with every canary green."
    )


_RECEIVER_CONTROLS = [
    # blip_2 really ships this spelling; the receiver predicate is deliberately
    # loose enough to still recognise it, and `.to(<device>)` keeps the dtype.
    ("blip_2 receiver .to(other.device)",
     "inputs_embeds = inputs_embeds.to(image_features.device).masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")"),
    ("receiver .to(device = ..., non_blocking = True)",
     "inputs_embeds = inputs_embeds.to(device = image_features.device, non_blocking = True).masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")"),
    ("receiver .to('cuda')",
     "inputs_embeds = inputs_embeds.to('cuda').masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")"),
    ("receiver .clone().contiguous()",
     "inputs_embeds = inputs_embeds.clone().contiguous().masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")"),
    ("receiver .to(inputs_embeds.dtype) is a no-op",
     "inputs_embeds = inputs_embeds.to(inputs_embeds.dtype).masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")"),
    ("plain receiver",
     "inputs_embeds = inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")"),
]


@pytest.mark.parametrize(
    "label,merge", _RECEIVER_CONTROLS, ids = [case[0] for case in _RECEIVER_CONTROLS],
)
def test_dtype_preserving_receivers_still_count_as_alignment(label, merge):
    # Controls for the test above: every one of these merges into a destination
    # that is still at `inputs_embeds.dtype`, so they must NOT turn red.
    found, aligned = _merge_dtype_alignment(
        _synthetic_forward(_ALIGNED_ASSIGN, merge = merge), "image_features",
    )
    assert found and aligned, f"{label}: dtype-safe spelling reported unaligned"


# `masked_scatter` is out-of-place. Losing the surrounding `inputs_embeds =`
# leaves every dtype cast in place - and silently drops the features, because the
# merged tensor is discarded. Upstream assigns the result on 5.5.0-5.15.0, so
# requiring it costs nothing; the in-place `masked_scatter_` spelling
# (transformers writes it in qwen2_5_omni / blt) needs no assignment and must
# still be accepted.
def test_discarded_out_of_place_merge_is_not_alignment():
    merge = (
        "inputs_embeds.masked_scatter(\n"
        "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
        ")"
    )
    found, aligned = _merge_dtype_alignment(
        _synthetic_forward(_ALIGNED_ASSIGN, merge = merge), "image_features",
    )
    assert found, "the synthetic merge site itself was not found"
    assert not aligned, (
        "the tracer reports the merge aligned, but `masked_scatter` is "
        "out-of-place and its result is thrown away: the image features never "
        "reach the model. Every dtype canary here would stay green while the "
        "merge silently did nothing."
    )


_RESULT_USE_CONTROLS = [
    ("in-place masked_scatter_ needs no assignment",
     "inputs_embeds.masked_scatter_(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")"),
    ("result bound to a temp name, then rebound",
     "merged = inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")\n"
     "inputs_embeds = merged"),
    ("result flows into a call argument",
     "inputs_embeds = self.norm(inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     "))"),
    ("result returned directly",
     "return inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")"),
]


@pytest.mark.parametrize(
    "label,merge", _RESULT_USE_CONTROLS, ids = [case[0] for case in _RESULT_USE_CONTROLS],
)
def test_used_merge_results_still_count_as_alignment(label, merge):
    # Controls for the test above.
    found, aligned = _merge_dtype_alignment(
        _synthetic_forward(_ALIGNED_ASSIGN, merge = merge), "image_features",
    )
    assert found and aligned, f"{label}: dtype-safe spelling reported unaligned"


def test_in_place_merge_with_an_unaligned_source_is_still_traced():
    # Accepting `masked_scatter_` must not become a way to skip the dtype trace:
    # the in-place op has exactly the same dtype requirement.
    merge = (
        "image_features = image_features.float()\n"
        "inputs_embeds.masked_scatter_(image_mask, image_features)"
    )
    found, aligned = _merge_dtype_alignment(
        _synthetic_forward(_ALIGNED_ASSIGN, merge = merge), "image_features",
    )
    assert found and not aligned, (
        "an in-place merge fed by a feature rebound to fp32 must report unaligned"
    )


# ---------------------------------------------------------------------------
# `Tensor.to(other)` is documented as "returns a Tensor with same torch.dtype and
# torch.device as the Tensor other", so `image_features.to(inputs_embeds)` aligns
# BOTH in one call and is a perfectly safe merge source. Reading only the
# explicit `inputs_embeds.dtype` spelling would fail these canaries on a healthy
# upstream that happened to prefer the overload, and the failure message would
# claim the dtype is never aligned - a false red that hides nothing and costs a
# release.
# ---------------------------------------------------------------------------
_TENSOR_OVERLOAD_CONTROLS = [
    ("tensor overload at the merge argument",
     "inputs_embeds = inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds)\n"
     ")"),
    ("tensor overload with non_blocking",
     "inputs_embeds = inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds, non_blocking = True)\n"
     ")"),
]


@pytest.mark.parametrize(
    "label,merge", _TENSOR_OVERLOAD_CONTROLS, ids = [c[0] for c in _TENSOR_OVERLOAD_CONTROLS],
)
def test_tensor_to_tensor_overload_counts_as_alignment(label, merge):
    body = "image_features = image_features.reshape(-1, inputs_embeds.shape[-1])"
    found, aligned = _merge_dtype_alignment(
        _synthetic_forward(body, merge = merge), "image_features",
    )
    assert found and aligned, (
        f"{label}: `image_features.to(inputs_embeds)` gives the result "
        f"`inputs_embeds`' dtype AND device, so the merge is dtype-safe. Reporting "
        f"it unaligned fails the real-source canaries on a healthy upstream."
    )


def test_tensor_overload_on_the_assignment_counts_as_alignment():
    found, aligned = _merge_dtype_alignment(
        _synthetic_forward("image_features = image_features.to(inputs_embeds)"),
        "image_features",
    )
    assert found and aligned, "the tensor overload on the dominating assignment is alignment"


_TENSOR_OVERLOAD_HOLES = [
    # Aligning to some OTHER tensor says nothing about `inputs_embeds`' dtype.
    ("overload naming a different tensor",
     "image_features = image_features.to(image_mask)"),
    ("overload naming a module attribute",
     "image_features = image_features.to(self.dummy)"),
    # `inputs_embeds` only carries the dtype as the FIRST positional argument;
    # torch has no `other=` keyword, and `.to(device, inputs_embeds)` is not a
    # signature that exists.
    ("embeds passed as a keyword, which torch rejects",
     "image_features = image_features.to(device = inputs_embeds)"),
]


@pytest.mark.parametrize(
    "label,body", _TENSOR_OVERLOAD_HOLES, ids = [c[0] for c in _TENSOR_OVERLOAD_HOLES],
)
def test_tensor_overload_to_another_tensor_is_not_alignment(label, body):
    found, aligned = _merge_dtype_alignment(_synthetic_forward(body), "image_features")
    assert found, f"{label}: the synthetic merge site itself was not found"
    assert not aligned, (
        f"{label}: only `.to(inputs_embeds)` proves the source landed on the "
        f"embedding dtype; accepting any tensor makes the overload a blanket "
        f"escape hatch."
    )


# ---------------------------------------------------------------------------
# `masked_scatter` is out-of-place, so the merged embedding exists ONLY in the
# return value. Rejecting just the bare-expression spelling is not enough: a
# result bound to a name nothing reads, or passed to a logger, is discarded just
# as completely, and `forward` then returns the UNCHANGED `inputs_embeds` with
# every dtype canary green. The result has to reach the embeddings that survive.
# ---------------------------------------------------------------------------
_DISCARDED_RESULT_HOLES = [
    ("result bound to a name nothing reads",
     "dead = inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")"),
    ("result passed to a logger",
     "self.log(inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     "))"),
    ("result bound to a temp that is then overwritten",
     "merged = inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")\n"
     "merged = inputs_embeds"),
    ("result appended to a list that never reaches the embeddings",
     "collected = [inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")]"),
]


@pytest.mark.parametrize(
    "label,merge", _DISCARDED_RESULT_HOLES, ids = [c[0] for c in _DISCARDED_RESULT_HOLES],
)
def test_merge_result_that_never_reaches_the_embeddings_is_not_alignment(label, merge):
    found, aligned = _merge_dtype_alignment(
        _synthetic_forward(_ALIGNED_ASSIGN, merge = merge), "image_features",
    )
    assert found, f"{label}: the synthetic merge site itself was not found"
    assert not aligned, (
        f"{label}: the tracer reports the merge aligned, but its out-of-place "
        f"result never reaches `inputs_embeds` or the return, so `forward` hands "
        f"back embeddings the features were never merged into. Every dtype canary "
        f"here would stay green while the features silently vanished."
    )


_RESULT_REACHES_CONTROLS = [
    ("result reaches inputs_embeds through two temporaries",
     "first = inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")\n"
     "second = self.norm(first)\n"
     "inputs_embeds = second"),
    ("result rebinds inputs_embeds inside a branch",
     "if self.config.merge:\n"
     "    inputs_embeds = inputs_embeds.masked_scatter(\n"
     "        image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     "    )"),
]


@pytest.mark.parametrize(
    "label,merge", _RESULT_REACHES_CONTROLS, ids = [c[0] for c in _RESULT_REACHES_CONTROLS],
)
def test_results_that_do_reach_the_embeddings_still_count(label, merge):
    # Controls: the merged tensor really does land back on `inputs_embeds`, so
    # these must NOT turn red.
    found, aligned = _merge_dtype_alignment(
        _synthetic_forward(_ALIGNED_ASSIGN, merge = merge), "image_features",
    )
    assert found and aligned, f"{label}: a surviving merge result reported unaligned"


# ---------------------------------------------------------------------------
# A name is not only rebound by an assignment statement. A `for` target, a `with
# ... as` target and a named expression all replace the tensor the name refers
# to, so an earlier aligned assignment stops describing it. Skipping them lets
# the replacement - which nothing cast - reach masked_scatter.
# ---------------------------------------------------------------------------
_NON_ASSIGN_BINDING_HOLES = [
    ("for target",
     f"{_ALIGNED_ASSIGN}\n"
     "for image_features in image_batches:\n"
     "    pass"),
    ("for target destructured",
     f"{_ALIGNED_ASSIGN}\n"
     "for image_features, size in image_batches:\n"
     "    pass"),
    ("async for target",
     f"{_ALIGNED_ASSIGN}\n"
     "async for image_features in image_batches:\n"
     "    pass"),
    ("with ... as target",
     f"{_ALIGNED_ASSIGN}\n"
     "with self.stream() as image_features:\n"
     "    pass"),
    ("named expression inside a condition",
     f"{_ALIGNED_ASSIGN}\n"
     "if (image_features := self.pack_image_features(image_features)) is None:\n"
     "    pass"),
    ("named expression as a statement",
     f"{_ALIGNED_ASSIGN}\n"
     "(image_features := self.pack_image_features(image_features))"),
]


@pytest.mark.parametrize(
    "label,body", _NON_ASSIGN_BINDING_HOLES, ids = [c[0] for c in _NON_ASSIGN_BINDING_HOLES],
)
def test_bindings_outside_assignments_invalidate_alignment(label, body):
    found, aligned = _merge_dtype_alignment(_synthetic_forward(body), "image_features")
    assert found, f"{label}: the synthetic merge site itself was not found"
    assert not aligned, (
        f"{label}: the tracer still credits the earlier aligned assignment, but "
        f"`image_features` has since been rebound by a construct that carries no "
        f"`inputs_embeds.dtype` cast. The rebound tensor reaches masked_scatter."
    )


_NON_ASSIGN_BINDING_CONTROLS = [
    ("for target over an aligned iterable",
     f"{_ALIGNED_ASSIGN}\n"
     "for image_features in image_batches.to(inputs_embeds.device, inputs_embeds.dtype):\n"
     "    pass"),
    ("with ... as target over an aligned context",
     f"{_ALIGNED_ASSIGN}\n"
     "with self.stream(image_features.to(inputs_embeds.dtype)) as image_features:\n"
     "    pass"),
    ("named expression whose value casts dtype",
     f"{_ALIGNED_ASSIGN}\n"
     "(image_features := image_features.to(inputs_embeds.dtype))"),
    ("for target binding an unrelated name",
     f"{_ALIGNED_ASSIGN}\n"
     "for patch in image_batches:\n"
     "    pass"),
    ("with ... as binding an unrelated name",
     f"{_ALIGNED_ASSIGN}\n"
     "with self.stream() as ctx:\n"
     "    pass"),
    ("with ... as target aligned, body leaves it alone",
     "with self.stream(image_features) as image_features:\n"
     f"    {_ALIGNED_ASSIGN}"),
]


@pytest.mark.parametrize(
    "label,body", _NON_ASSIGN_BINDING_CONTROLS, ids = [c[0] for c in _NON_ASSIGN_BINDING_CONTROLS],
)
def test_aligned_or_unrelated_bindings_do_not_break_alignment(label, body):
    # Controls for the test above.
    found, aligned = _merge_dtype_alignment(_synthetic_forward(body), "image_features")
    assert found and aligned, f"{label}: dtype-safe spelling reported unaligned"


# ---------------------------------------------------------------------------
# The eager patch's own matcher must be tied to the EMBEDDING merge. Accepting
# `masked_scatter` on any receiver let an already-aligned merge on some scratch
# tensor land in `fixed_casts`, which reads to
# `_patch_gemma4_audio_feature_dtype_on_class` as "upstream fixed it": it sets
# the patched marker and returns, leaving a real, still-device-only
# `inputs_embeds` merge unpatched and crashing.
# ---------------------------------------------------------------------------
def _audio_forward(body):
    return (
        "class Gemma4Model:\n"
        "    def forward(self, inputs_embeds, audio_features, audio_mask):\n"
        "        if audio_features is not None:\n"
        f"{_indented(body)}\n"
        "        return inputs_embeds\n"
    )


_REAL_AUDIO_MERGE = (
    "inputs_embeds = inputs_embeds.masked_scatter(\n"
    "    audio_mask.to(inputs_embeds.device), audio_features.to(inputs_embeds.device)\n"
    ")"
)
_ALIGNED_SCRATCH_MERGE = (
    "scratch = scratch.masked_scatter(\n"
    "    audio_mask, audio_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
    ")"
)


def _audio_merge_casts(body):
    forward = _gemma4_model_forward_node(_audio_forward(body))
    assert forward is not None
    return g4p._gemma4_audio_merge_casts(forward)


def test_eager_matcher_ignores_merges_onto_other_tensors():
    # An aligned decoy on a scratch tensor must not be reported as the audio
    # merge being already fixed.
    buggy, fixed = _audio_merge_casts(_ALIGNED_SCRATCH_MERGE)
    assert not buggy and not fixed, (
        f"_gemma4_audio_merge_casts counted a `scratch.masked_scatter(...)` as the "
        f"embedding merge ({len(buggy)} buggy, {len(fixed)} fixed). The eager patch "
        f"would mark Gemma4Model as already patched on the strength of a merge that "
        f"never touches `inputs_embeds`."
    )


def test_eager_matcher_still_finds_the_real_merge_next_to_a_decoy():
    # The decoy must be ignored AND the real device-only merge still picked up,
    # so the patch fires exactly once.
    buggy, fixed = _audio_merge_casts(_ALIGNED_SCRATCH_MERGE + "\n" + _REAL_AUDIO_MERGE)
    assert len(buggy) == 1 and not fixed, (
        f"expected the one real device-only `inputs_embeds` merge to be patchable "
        f"next to an aligned scratch decoy, got {len(buggy)} buggy / {len(fixed)} fixed"
    )


def test_eager_matcher_still_accepts_a_transformed_inputs_embeds_receiver():
    # Discovery must stay loose about HOW the receiver mentions `inputs_embeds`
    # (blip_2 ships `inputs_embeds.to(...).masked_scatter(...)`), or a real merge
    # would be dropped instead of patched.
    buggy, fixed = _audio_merge_casts(
        "inputs_embeds = inputs_embeds.to(audio_features.device).masked_scatter(\n"
        "    audio_mask.to(inputs_embeds.device), audio_features.to(inputs_embeds.device)\n"
        ")"
    )
    assert len(buggy) == 1 and not fixed, (
        f"a transformed `inputs_embeds` receiver must still be recognised as the "
        f"embedding merge, got {len(buggy)} buggy / {len(fixed)} fixed"
    )

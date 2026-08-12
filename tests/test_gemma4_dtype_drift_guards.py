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


# Upstream may spell the dtype alignment at the merge call
# (`audio_features.to(inputs_embeds.device, inputs_embeds.dtype)`, tf 5.15+) or on
# the statement that produced the features (tf 5.5-5.14). A substring anchor on one
# spelling breaks on cosmetic refactors while staying blind to a real regression
# written in another, so the canaries trace dtype ALIGNMENT structurally: every
# `inputs_embeds.masked_scatter` consuming the features needs an
# `inputs_embeds.dtype` cast at the call or on the last assignment that DOMINATES
# the merge (same block or an enclosing one, so a cast stranded in a branch the
# merge does not run under does not count), a receiver still AT
# `inputs_embeds.dtype`, and a result that is actually used (masked_scatter is
# out-of-place).
# Deliberate limits, none hit by any transformers 5.5.0-5.15.0 source (checked
# against every gemma4-bearing release, and the spellings against all 503 model
# families of 5.15.0):
#   * `match`/`case` bindings are not tracked - no transformers modeling file has
#     a `match` statement;
#   * the merged result only has to REACH `inputs_embeds`; a later overwrite is not
#     modelled, since requiring survival to the end would red-flag "merge, use,
#     then release the name" without a use-before-overwrite analysis;
#   * `.to(other_tensor)` counts as alignment here but not in the eager patch's own
#     matcher (unsloth_zoo/temporary_patches/gemma4.py), which reads only the
#     spelling upstream actually ships. Upstream adopting the overload would
#     surface as a drift warning telling us to update the patch, as intended.
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

    Two spellings count: the explicit `features.to(inputs_embeds.device,
    inputs_embeds.dtype)`, and the tensor overload `features.to(inputs_embeds)`,
    documented as returning a tensor with the same dtype AND device as `other`.
    Rejecting the overload would fail these canaries on a healthy upstream that
    preferred it, claiming nobody aligns the dtype at all. Only the FIRST
    positional argument can be that overload; torch has no `other=` keyword.
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
# changing the destination dtype. Anything else (`.float()`, `.type(...)`, an
# unknown helper) can retype it, leaving an fp32 destination and a low-precision
# source, so an `inputs_embeds.dtype` source cast would prove nothing.
_DTYPE_PRESERVING_METHODS = frozenset({
    "clone", "contiguous", "detach", "cpu", "cuda",
    "view", "reshape", "flatten", "squeeze", "unsqueeze",
    "expand", "expand_as", "transpose", "permute", "narrow",
})


# `torch.<name>` constants that provably are NOT dtypes, so they cannot retype the
# receiver: `clone(memory_format = torch.preserve_format)` keeps the embedding dtype
# exactly. Reading every `torch.<attr>` as a dtype turned that cosmetic upstream
# addition into a "destination was retyped" false red - the very failure this
# structural trace exists to remove. Resolved against the live torch, so an
# attribute torch does not have stays dtype-bearing: the safe default.
_NON_DTYPE_TORCH_CONSTANTS = (torch.memory_format, torch.layout)


def _is_dtype_bearing(node, aliases = frozenset()):
    """Can `node` retype a tensor? (`torch.float32`, `other.dtype`, ...)

    `aliases` are local names known to hold a dtype: `Tensor.view(dtype)`
    reinterprets its receiver whether the dtype is written out or held in a name.
    """
    if isinstance(node, ast.Name):
        return node.id in aliases
    if isinstance(node, ast.Attribute) and node.attr == "dtype":
        return not _is_attr(node, "inputs_embeds", "dtype")
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "torch"
    ):
        # A memory format or layout does not retype; `Tensor.view(torch.int32)`
        # does, so the check is on the constant, not the method.
        return not isinstance(
            getattr(torch, node.attr, None), _NON_DTYPE_TORCH_CONSTANTS
        )
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

    `inputs_embeds.to(other.device)` is blip_2's real spelling and keeps the dtype;
    `.to(torch.float32)`, `.to(other.dtype)` and `.to(other_tensor)` do not.
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


def _receiver_preserves_embeds_dtype(node, aliases = frozenset()):
    """Is the merge destination `inputs_embeds` still at `inputs_embeds.dtype`?

    The trace is "the source is cast to `inputs_embeds.dtype`, so the merge is
    safe", which only holds while the DESTINATION is at that dtype too. A chain of
    dtype-preserving transformations is fine (blip_2 really writes
    `inputs_embeds.to(other.device).masked_scatter(...)`), but
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
            if any(_is_dtype_bearing(v, aliases) for v in values):
                return False
        else:
            return False
        node = node.func.value


_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _walk_own_scope(root, descend = True):
    """`ast.walk` restricted to the code `root`'s own scope executes.

    A nested `def` / `class` / `lambda` has its own locals, so a merge inside a
    helper defined in `forward` is not `forward`'s merge (it may never be invoked),
    and a cast in `forward` says nothing about a same-named parameter of that
    helper. The nested statement itself is still yielded - only its body is skipped.
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

    Only these lists run unconditionally *relative to the merge*: a cast nested in
    another branch may never execute on the path that reaches it.
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

    Only that comparison makes the untaken branch aligned BY DEFINITION. Any other
    test (`== inputs_embeds.dtype`, `!= torch.bfloat16`, or the mismatch test
    `and`-ed with something else) leaves a reachable path on which the feature
    stays un-cast, so a cast under it is not unconditional alignment.
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

    Recurses through destructuring targets, because `image_features, feature_lens =
    self.pack_image_features(...)` (llava_next / llava_onevision) rebinds
    `image_features` just as much as a bare assignment does; missing it lets an
    earlier aligned assignment keep proving alignment for a tensor since replaced.

    `image_features[0] = ...` (Subscript) and `self.image_features = ...`
    (Attribute) are deliberately NOT rebindings: the name still refers to the same
    object afterwards.
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
# Every construct that can rebind a plain name in the code's OWN scope: `for
# image_features in ...`, `with self.autocast() as image_features:` and
# `(image_features := ...)` replace the tensor as completely as an assignment does.
# Counting only assignments lets an earlier aligned cast keep proving alignment for
# a value one of these has overwritten, and the replacement reaches masked_scatter
# un-cast. (Comprehensions bind in their own scope.)
_BINDING_NODES = _ASSIGN_NODES + _LOOP_NODES + _WITH_NODES + (ast.NamedExpr,)


def _binding_pairs(node):
    """`[(target, value)]` this binding node establishes.

    `value` is the expression whose dtype the bound name carries, so the same
    alignment question fits every form: an assignment's RHS, a loop's iterable, a
    `with` item's context expression.
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


def _element_value(target, value, feature):
    """The part of `value` that `<feature>` actually receives from `target`.

    A destructuring target consumes a literal tuple/list RHS element by element, so
    in `image_features, lens = (image_features.float(), lens.to(
    inputs_embeds.dtype))` the cast belongs to `lens` alone, and crediting it to
    `image_features` reports an fp32 tensor aligned. Anything else (a call, a
    starred target, a length mismatch) keeps the whole RHS, which is how upstream's
    llava_next `image_features, lens = self.pack_image_features(image_features.to(
    inputs_embeds.dtype))` stays aligned.
    """
    if isinstance(target, ast.Name):
        return value if target.id == feature else None
    if (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(value, (ast.Tuple, ast.List))
        and len(target.elts) == len(value.elts)
        and not any(isinstance(elt, ast.Starred) for elt in target.elts)
        and not any(isinstance(elt, ast.Starred) for elt in value.elts)
    ):
        for element, element_value in zip(target.elts, value.elts):
            if feature in _bound_names(element):
                return _element_value(element, element_value, feature)
    return value


# Only a real destructuring assignment pairs targets with RHS elements; a `for` /
# `with` target takes its value from the iterable / context manager instead.
_ELEMENTWISE_NODES = (ast.Assign, ast.AnnAssign, ast.NamedExpr)


def _binding_value(node, feature):
    """The expression `<feature>` takes its dtype from in this binding node."""
    for target, value in _binding_pairs(node):
        if target is None or feature not in _bound_names(target):
            continue
        if value is not None and isinstance(node, _ELEMENTWISE_NODES):
            return _element_value(target, value, feature)
        return value
    return None


def _binding_is_aligned(node, feature):
    value = _binding_value(node, feature)
    return value is not None and _casts_to_inputs_embeds_dtype(value)


def _feature_rebindings(stmt, feature):
    """Every rebinding of `<feature>` anywhere inside `stmt`'s own scope.

    An augmented assignment is NOT one: `image_features += delta` is `Tensor.add_`,
    which writes through the existing storage and keeps its dtype. It neither
    creates the alignment its RHS carries nor destroys one an earlier cast
    established, so the running verdict stands.
    """
    return [
        node
        for node in _walk_own_scope(stmt, descend = False)
        if isinstance(node, _BINDING_NODES)
        and not isinstance(node, ast.AugAssign)
        and _assigns_feature(node, feature)
    ]


def _rebinding_alignment(stmt, feature):
    """Does `stmt` rebind `<feature>` on the path to the merge, and is it aligned?

    `None` when `stmt` does not rebind the name, so it says nothing either way.
    """
    if (
        isinstance(stmt, _ASSIGN_NODES)
        and not isinstance(stmt, ast.AugAssign)
        and _assigns_feature(stmt, feature)
    ):
        return _binding_is_aligned(stmt, feature)
    # A bare `(image_features := ...)` statement rebinds like an assignment.
    if (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.NamedExpr)
        and _assigns_feature(stmt.value, feature)
    ):
        return _binding_is_aligned(stmt.value, feature)
    # `if <feature>.dtype != inputs_embeds.dtype: <feature> = <feature>.to(...)` is
    # the cast-if-needed idiom (blip_2 / granite_speech): the untaken branch is
    # aligned by definition, so it counts as unconditional. Both branches are still
    # analysed as PATHS, not as a bag of rebindings whose textually last one wins,
    # or an aligned `else` after an unaligned body would report the guard aligned
    # while the mismatch path leaves the feature un-cast.
    if isinstance(stmt, ast.If) and _is_dtype_guard(stmt.test, feature):
        if _feature_rebindings(stmt, feature):
            # Body runs only on a mismatch (starts unaligned); `orelse` only when
            # the dtypes already agree (starts aligned). Both ends must be aligned.
            return (
                _sequence_alignment(stmt.body, feature, False)
                and _sequence_alignment(stmt.orelse, feature, True)
            )
    # Any OTHER compound statement that rebinds the feature is still a rebinding on
    # the path to the merge (`if enabled: image_features = image_features.float()`).
    rebindings = _feature_rebindings(stmt, feature)
    if not rebindings:
        return None
    if isinstance(stmt, _WITH_NODES):
        # A `with` body runs unconditionally, so its rebindings dominate the merge
        # like top-level ones. The `as` target binds first and seeds the state.
        alignment = (
            _binding_is_aligned(stmt, feature)
            if _assigns_feature(stmt, feature) else None
        )
        for inner in stmt.body:
            inner_alignment = _rebinding_alignment(inner, feature)
            if inner_alignment is not None:
                alignment = inner_alignment
        return alignment
    # Conditionally executed (`if` / `for` / `while` / `try`): the rebinding may not
    # run, so an unaligned branch reports unaligned while a branch that only ever
    # casts to `inputs_embeds.dtype` leaves the earlier verdict standing (`None`).
    # A loop's own target is one of those rebindings, typed by the iterable.
    for rebinding in rebindings:
        if not _binding_is_aligned(rebinding, feature):
            return False
    return None


def _sequence_alignment(block, feature, initial):
    """Alignment of `<feature>` after running `block` top to bottom.

    `initial` is the state on entry. Statements that say nothing about the feature
    (`None`, which includes a conditional cast that may not run) leave it alone.
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

# Attributes / methods that read a tensor's METADATA rather than its values.
# `fallback.to(image_features.device, inputs_embeds.dtype)` mentions
# `image_features` but scatters none of it.
_METADATA_ATTRS = frozenset({
    "device", "dtype", "shape", "ndim", "size", "numel", "layout",
    "requires_grad", "is_cuda", "element_size", "itemsize", "stride",
})


def _carries_value(node, matches):
    """Do the VALUES of a `matches` expression flow into `node`?

    A bare occurrence check would answer yes to `image_features.device` or
    `merged.dtype` too. A metadata read carries no values, so its whole subtree is
    skipped; everything else still counts, keeping
    `image_features.to(...).reshape(-1, image_features.shape[-1])` a use.
    """
    if isinstance(node, ast.Attribute) and node.attr in _METADATA_ATTRS:
        return False
    if matches(node):
        return True
    return any(_carries_value(child, matches) for child in ast.iter_child_nodes(node))


def _carries_feature_value(node, feature):
    """Do `<feature>`'s VALUES flow into `node`?

    Counting a metadata read would report `fallback.to(image_features.device,
    inputs_embeds.dtype)` as the image merge, so if the real merge were ever
    dropped `_merge_dtype_alignment` would still return `(True, True)` on a source
    in which no image value reaches the model.
    """
    return _carries_value(node, lambda n: isinstance(n, ast.Name) and n.id == feature)


def _feature_merges(forward, feature):
    """Every `inputs_embeds.masked_scatter[_](mask, ...)` call using `<feature>`.

    The receiver has to REFERENCE `inputs_embeds` (a `scratch.masked_scatter(...)`
    is a different tensor), but discovery stays loose - any receiver mentioning the
    name - so a transformed one such as blip_2's
    `inputs_embeds.to(other.device).masked_scatter(...)` is still recognised AS the
    embedding merge; whether it is dtype-safe is then
    `_receiver_preserves_embeds_dtype`'s call. Dropping unrecognised receivers here
    instead would let a second, safe merge hide a dtype-changing one.

    The SOURCE, by contrast, has to carry the feature's VALUES, or a metadata
    mention could stand in for a merge upstream has removed.

    `masked_scatter_` is accepted too (qwen2_5_omni, blt): same storage, same dtype
    agreement. Nested `def` / `class` / `lambda` bodies are skipped - a merge in a
    helper `forward` never invokes is not `forward`'s merge.
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
        if _carries_feature_value(node.args[1], feature):
            calls.append(node)
    return sorted(calls, key=lambda n: (n.lineno, n.col_offset))


# The merge RESULT is followed by the same value-flow rule as the source. A
# metadata read (`merged.device`, `merged.dtype`, `merged.shape`) discards every
# scattered value, so `inputs_embeds = original.to(merged.device)` hands the
# embeddings the ORIGINAL tensor. Treating that mention as value flow reported the
# merge used, so a refactor dropping the effective merge stayed green.
def _contains(node, target):
    return _carries_value(node, lambda n: n is target)


def _references(node, names):
    return _carries_value(node, lambda n: isinstance(n, ast.Name) and n.id in names)


def _statements_after(forward, call):
    """The statements that execute after `call`, innermost block outwards.

    The merge's OWN statement first, then the rest of its block, then the remainder
    of each enclosing one: the path the merged tensor must survive to the return.
    """
    chain = _enclosing_blocks(forward, call)
    if not chain:
        return None
    ordered = []
    for depth, (block, index) in enumerate(reversed(chain)):
        # The innermost entry points AT the merge's statement; every outer entry
        # points at the compound statement wrapping it, already accounted for.
        ordered.extend(block[index if depth == 0 else index + 1:])
    return ordered


def _merge_result_is_used(forward, call):
    """Does the merge's result reach the embeddings `forward` goes on to use?

    `masked_scatter` is OUT-OF-PLACE, so the merged embedding exists only in its
    return value; upstream lands it back on `inputs_embeds` on 5.5.0-5.15.0.
    Asking only whether the merge is a bare expression statement is too weak - it
    passes `dead = inputs_embeds.masked_scatter(...)` and
    `self.log(inputs_embeds.masked_scatter(...))`, which drop the merged embedding
    on the floor and leave `forward` returning the UNCHANGED `inputs_embeds` with
    every dtype canary green. So follow the result forward: it must flow, directly
    or through temporaries, into `inputs_embeds` or into a `return`.

    `masked_scatter_` needs no result ONLY when its receiver is `inputs_embeds`
    itself. On a transformed receiver (`inputs_embeds.clone()`, or the
    device-changing `.to(...)`, both of which `_receiver_preserves_embeds_dtype`
    accepts as dtype-safe) the mutation lands in a temporary, so it has to hand its
    result back like the out-of-place spelling does - and it can, since
    `masked_scatter_` returns its receiver.
    """
    if call.func.attr.endswith("_") and _is_embeds_tensor(call.func.value):
        return True
    statements = _statements_after(forward, call)
    if statements is None:
        return True

    tainted = set()
    for position, statement in enumerate(statements):
        for node in _walk_own_scope(statement, descend = False):
            # A `return` carrying the merge, or its taint, is the merged embedding
            # leaving `forward`.
            if isinstance(node, ast.Return) and node.value is not None:
                if (position == 0 and _contains(node.value, call)) or _references(node.value, tainted):
                    return True
            if not isinstance(node, _BINDING_NODES):
                continue
            for target, value in _binding_pairs(node):
                if target is None or value is None:
                    continue
                # Each name is judged on the RHS element it actually receives.
                # Reading the whole RHS made `inputs_embeds, aux = original, merged`
                # taint `inputs_embeds` while the embeddings were handed the
                # ORIGINAL. A non-literal RHS keeps the whole value, so
                # `inputs_embeds, aux = self.pack(merged)` still counts.
                element = value if isinstance(node, _ELEMENTWISE_NODES) else None
                for name in _bound_names(target):
                    part = _element_value(target, element, name) if element is not None else value
                    carries = (
                        (position == 0 and part is not None and _contains(part, call))
                        or (part is not None and _references(part, tainted))
                    )
                    if carries:
                        tainted.add(name)
                    elif node is statement and not isinstance(node, ast.AugAssign):
                        # A direct, unconditional rebinding to something the merge
                        # did NOT flow into overwrites the name. One nested in a
                        # conditional may not run, and `+=` adds rather than
                        # replaces, so neither clears the taint.
                        tainted.discard(name)
        if "inputs_embeds" in tainted:
            return True
    return False


def _dtype_alias_names(forward):
    """Local names bound to a dtype, e.g. `target_dtype = torch.float32`.

    `inputs_embeds.view(target_dtype)` retypes the merge destination exactly like
    `view(torch.float32)` does, so a receiver holding its dtype in a variable must
    not read as dtype-preserving. Only a binding whose VALUE is dtype-bearing
    counts, which keeps the legitimate shape overload (`target_shape =
    image_features.shape` / `view(target_shape)`, real upstream spelling) out.
    Repeated to a fixpoint so an alias of an alias resolves.
    """
    names = set()
    while True:
        found = set()
        for node in _walk_own_scope(forward):
            if not isinstance(node, _ELEMENTWISE_NODES):
                continue
            for target, value in _binding_pairs(node):
                if target is None or value is None:
                    continue
                if _is_dtype_bearing(value, names):
                    found.update(_bound_names(target))
        if found <= names:
            return names
        names |= found


def _merge_dtype_alignment(src, feature):
    """Trace `<feature>` into `inputs_embeds.masked_scatter(...)`.

    Returns (merge_found, every_merge_aligned). EVERY merge consuming the feature
    has to be aligned: one unaligned branch is one reachable dtype crash. A merge
    is aligned when the destination is still at `inputs_embeds.dtype`, the result
    is actually used, and the source carries an `inputs_embeds.dtype` cast at the
    call or on its last dominating assignment.
    """
    forward = _gemma4_model_forward_node(src)
    if forward is None:
        return (False, False)
    merges = _feature_merges(forward, feature)
    if not merges:
        return (False, False)
    aliases = _dtype_alias_names(forward)
    aligned = all(
        _receiver_preserves_embeds_dtype(merge.func.value, aliases)
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
    # The canary above only proves the COMPILER rewriter (a regex) still fires. The
    # eager patch is stricter - the merge source argument must BE the
    # `audio_features.to(inputs_embeds.device, ...)` call - so a wrapped spelling
    # matches the regex but not the AST matcher, and the eager path would no-op
    # with the dtype crash back and every other guard green. Run the patch's OWN
    # matcher over the real source so that spelling fails loudly here instead.
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
    # transformers 5.15.0 fixed the audio merge upstream, so the real-source test
    # above passes without the rewriter doing anything. Keep it under test against
    # the historical buggy spelling while older transformers are still supported.
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
# Tracer unit tests. A hole in `_merge_dtype_alignment` is invisible: it reports
# GREEN on a source that crashes. These feed it a minimal synthetic
# `Gemma4Model.forward` (no transformers needed, so they run on every supported
# version) and pin the verdict on the spellings that used to slip through.
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
# aligned. Taking the textually last rebinding instead reports aligned for a guard
# whose MISMATCH path leaves the feature un-cast - one reachable
# `masked_scatter_: expected self and source to have same dtypes`.
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
    # Controls: dtype-safe on every path, so they must not turn red or the canaries
    # start failing on a healthy upstream.
    found, aligned = _merge_dtype_alignment(_synthetic_forward(body), "image_features")
    assert found and aligned, f"{label}: dtype-safe spelling reported unaligned"


# A destructuring assignment rebinds the feature name just like a plain one, and
# `image_features, feature_lens = self.pack_image_features(...)` is upstream's own
# llava_next spelling. Skipping it lets an earlier aligned assignment keep proving
# alignment for a tensor that has since been replaced.
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


# A destructuring target takes its RHS ELEMENT, not the whole RHS: reading the
# whole tuple lets a cast belonging to another name keep the feature green
# (transformers writes literal-tuple destructuring in vilt).
_UNPACK_PAIRING_HOLES = [
    ("tuple RHS whose cast belongs to the other name",
     f"{_ALIGNED_ASSIGN}\n"
     "image_features, feature_lens = (\n"
     "    image_features.float(), feature_lens.to(inputs_embeds.dtype))"),
    ("list RHS whose cast belongs to the other name",
     f"{_ALIGNED_ASSIGN}\n"
     "[image_features, feature_lens] = [\n"
     "    image_features.float(), feature_lens.to(inputs_embeds.dtype)]"),
    ("nested tuple RHS whose cast belongs to the other name",
     f"{_ALIGNED_ASSIGN}\n"
     "(image_features, feature_lens), image_sizes = (\n"
     "    (image_features.float(), feature_lens.to(inputs_embeds.dtype)), sizes)"),
]


@pytest.mark.parametrize(
    "label,body", _UNPACK_PAIRING_HOLES, ids = [case[0] for case in _UNPACK_PAIRING_HOLES],
)
def test_a_cast_belonging_to_another_target_is_not_alignment(label, body):
    found, aligned = _merge_dtype_alignment(_synthetic_forward(body), "image_features")
    assert found, f"{label}: the synthetic merge site itself was not found"
    assert not aligned, (
        f"{label}: the tracer credits `image_features` with an "
        f"`inputs_embeds.dtype` cast that the unpack hands to a DIFFERENT name. "
        f"`image_features` is fp32 when it reaches masked_scatter."
    )


def test_the_matching_element_of_a_tuple_rhs_still_counts():
    body = (
        "image_features, feature_lens = (\n"
        "    image_features.to(inputs_embeds.device, inputs_embeds.dtype), feature_lens)"
    )
    found, aligned = _merge_dtype_alignment(_synthetic_forward(body), "image_features")
    assert found and aligned, "the element the feature actually receives carries the cast"


# `image_features += delta` is `Tensor.add_`: in place, so the name keeps the dtype
# it already had. It can neither inherit the RHS's cast nor lose an earlier one.
_AUG_ASSIGN_CASES = [
    ("augmented assignment does not inherit the RHS cast",
     "image_features = image_features.float()\n"
     "image_features += delta.to(inputs_embeds.dtype)", False),
    ("augmented assignment with no earlier cast at all",
     "image_features += delta.to(inputs_embeds.dtype)", False),
    ("augmented assignment does not undo an earlier cast",
     f"{_ALIGNED_ASSIGN}\n"
     "image_features += self.position_embedding", True),
    ("augmented assignment inside a branch keeps the earlier cast",
     f"{_ALIGNED_ASSIGN}\n"
     "if self.config.add_position_embedding:\n"
     "    image_features += self.position_embedding", True),
]


@pytest.mark.parametrize(
    "label,body,expected", _AUG_ASSIGN_CASES, ids = [case[0] for case in _AUG_ASSIGN_CASES],
)
def test_augmented_assignment_preserves_the_left_hand_dtype(label, body, expected):
    found, aligned = _merge_dtype_alignment(_synthetic_forward(body), "image_features")
    assert found, f"{label}: the synthetic merge site itself was not found"
    assert aligned is expected, (
        f"{label}: expected aligned={expected}, got {aligned}. An in-place "
        f"`+=` keeps `image_features`' existing dtype, so it neither creates "
        f"alignment from its right-hand side nor destroys alignment established "
        f"before it."
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


# The trace is "the source is cast to `inputs_embeds.dtype`, so the merge is safe",
# which only holds while the DESTINATION is still at that dtype: a receiver
# transformation that retypes it leaves `inputs_embeds.float().masked_scatter(mask,
# image_features.to(inputs_embeds.dtype))` with an fp32 destination and an fp16/bf16
# source, raising the exact error this file exists to prevent.
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
    # `Tensor.view(dtype)` reinterprets the storage, so a real dtype constant must
    # still be caught now that memory formats do not count.
    ("receiver .view(torch.int32) reinterprets the dtype",
     "inputs_embeds = inputs_embeds.view(torch.int32).masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.dtype)\n"
     ")"),
    # ... just as thoroughly when the dtype arrives in a variable, which reads as a
    # plain name at the call site.
    ("receiver .view(alias of torch.float32)",
     "target_dtype = torch.float32\n"
     "inputs_embeds = inputs_embeds.view(target_dtype).masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.dtype)\n"
     ")"),
    ("receiver .view(alias of another tensor's dtype)",
     "target_dtype = image_mask.dtype\n"
     "inputs_embeds = inputs_embeds.view(target_dtype).masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.dtype)\n"
     ")"),
    ("receiver .view(alias of an alias)",
     "base_dtype = torch.float32\n"
     "target_dtype = base_dtype\n"
     "inputs_embeds = inputs_embeds.view(target_dtype).masked_scatter(\n"
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
    # blip_2 really ships this spelling; the receiver predicate stays loose enough
    # to recognise it, and `.to(<device>)` keeps the dtype.
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
    # A memory format is not a dtype, and transformers already ships
    # `.clone(memory_format = torch.contiguous_format)` (higgs_audio_v2).
    ("receiver .clone(memory_format = torch.preserve_format)",
     "inputs_embeds = inputs_embeds.clone(memory_format = torch.preserve_format).masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")"),
    ("receiver .contiguous(memory_format = torch.contiguous_format)",
     "inputs_embeds = inputs_embeds.contiguous(memory_format = torch.contiguous_format).masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")"),
    ("receiver .contiguous(torch.channels_last) positionally",
     "inputs_embeds = inputs_embeds.contiguous(torch.channels_last).masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")"),
    # `Tensor.view` also takes a shape sequence, so a plain name is more often a
    # shape than a dtype. Only a name actually BOUND to a dtype may retype.
    ("receiver .view(shape variable)",
     "target_shape = image_features.shape\n"
     "inputs_embeds = inputs_embeds.view(target_shape).masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")"),
    ("receiver .view(alias of inputs_embeds.dtype) is a no-op",
     "target_dtype = inputs_embeds.dtype\n"
     "inputs_embeds = inputs_embeds.view(target_dtype).masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")"),
]


@pytest.mark.parametrize(
    "label,merge", _RECEIVER_CONTROLS, ids = [case[0] for case in _RECEIVER_CONTROLS],
)
def test_dtype_preserving_receivers_still_count_as_alignment(label, merge):
    # Controls: every destination here is still at `inputs_embeds.dtype`.
    found, aligned = _merge_dtype_alignment(
        _synthetic_forward(_ALIGNED_ASSIGN, merge = merge), "image_features",
    )
    assert found and aligned, f"{label}: dtype-safe spelling reported unaligned"


# `masked_scatter` is out-of-place: losing the surrounding `inputs_embeds =` leaves
# every dtype cast in place and silently drops the features. Upstream assigns the
# result on 5.5.0-5.15.0, so requiring it costs nothing; in-place `masked_scatter_`
# (qwen2_5_omni / blt) needs no assignment and must still be accepted.
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
    # `masked_scatter_` returns its receiver, so a transformed in-place receiver
    # that hands the result back is as safe as the out-of-place spelling.
    ("in-place merge into a clone whose result is assigned back",
     "inputs_embeds = inputs_embeds.clone().masked_scatter_(\n"
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


# ---------------------------------------------------------------------------
# Merge DISCOVERY keys on the feature's VALUES reaching the scatter source. A source
# that only reads `<feature>.device` / `.dtype` / `.shape` merges some other tensor,
# so accepting it would let a metadata mention impersonate a merge upstream had
# deleted, reporting (True, True) while no image value reaches the model.
# ---------------------------------------------------------------------------
_METADATA_ONLY_MERGES = [
    ("source reads the feature device only",
     "inputs_embeds = inputs_embeds.masked_scatter(\n"
     "    image_mask, fallback.to(image_features.device, inputs_embeds.dtype)\n"
     ")"),
    ("source reads the feature dtype only",
     "inputs_embeds = inputs_embeds.masked_scatter(\n"
     "    image_mask, fallback.to(inputs_embeds.device, image_features.dtype)\n"
     ")"),
    ("source reads the feature shape only",
     "inputs_embeds = inputs_embeds.masked_scatter(\n"
     "    image_mask, fallback.reshape(-1, image_features.shape[-1])\n"
     ")"),
]


@pytest.mark.parametrize(
    "label,merge", _METADATA_ONLY_MERGES, ids = [c[0] for c in _METADATA_ONLY_MERGES],
)
def test_a_metadata_only_reference_is_not_a_feature_merge(label, merge):
    found, _ = _merge_dtype_alignment(
        _synthetic_forward(_ALIGNED_ASSIGN, merge = merge), "image_features",
    )
    assert not found, (
        f"{label}: the tracer counted a scatter that merely READS "
        f"`image_features` metadata as the image merge. With the real merge gone "
        f"this reports the modality present and aligned while no image value "
        f"reaches the model."
    )


_VALUE_FLOW_CONTROLS = [
    ("value and metadata in the same source",
     "inputs_embeds = inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype).reshape(\n"
     "        -1, image_features.shape[-1])\n"
     ")"),
    ("value inside a concatenation",
     "inputs_embeds = inputs_embeds.masked_scatter(\n"
     "    image_mask, torch.cat([image_features, extra]).to(\n"
     "        inputs_embeds.device, inputs_embeds.dtype)\n"
     ")"),
    ("value through a subscript",
     "inputs_embeds = inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features[0].to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")"),
]


@pytest.mark.parametrize(
    "label,merge", _VALUE_FLOW_CONTROLS, ids = [c[0] for c in _VALUE_FLOW_CONTROLS],
)
def test_feature_values_reaching_the_source_are_still_a_merge(label, merge):
    # Controls: dropping any of these would blind the canaries to a real merge.
    found, aligned = _merge_dtype_alignment(
        _synthetic_forward(_ALIGNED_ASSIGN, merge = merge), "image_features",
    )
    assert found and aligned, f"{label}: a real feature merge was not discovered"


def test_in_place_merge_with_an_unaligned_source_is_still_traced():
    # Accepting `masked_scatter_` must not skip the dtype trace: same requirement.
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
# `Tensor.to(other)` is documented as returning a tensor with the same dtype AND
# device as `other`, so `image_features.to(inputs_embeds)` aligns both in one call.
# Reading only the explicit `inputs_embeds.dtype` spelling would fail these canaries
# on a healthy upstream that preferred the overload, claiming the dtype is never
# aligned - a false red that hides nothing and costs a release.
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
    # torch has no `other=` keyword.
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
# return value. Rejecting just the bare-expression spelling is not enough: a result
# bound to a name nothing reads, or passed to a logger, is discarded as completely
# and `forward` returns the UNCHANGED `inputs_embeds` with every canary green.
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
    # `masked_scatter_` writes into the RECEIVER's storage; on a clone or a
    # device-changed copy that storage is a temporary, so the merge is lost.
    ("in-place merge into a clone of the embeddings",
     "inputs_embeds.clone().masked_scatter_(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")"),
    ("in-place merge into a device-changed copy of the embeddings",
     "inputs_embeds.to('cuda').masked_scatter_(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")"),
    # A destructuring target takes its own RHS element; crediting the whole RHS let
    # a tuple hand `inputs_embeds` the ORIGINAL while the merge went to a name
    # nothing reads - the modality silently dropped, every canary green.
    ("destructuring hands the embeddings the original, not the merge",
     "merged = inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")\n"
     "inputs_embeds, aux = original, merged"),
    ("list destructuring hands the embeddings the original",
     "merged = inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")\n"
     "[inputs_embeds, aux] = [original, merged]"),
    ("nested destructuring hands the embeddings the original",
     "merged = inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")\n"
     "(inputs_embeds, aux), sizes = (original, merged), image_sizes"),
    # A metadata read of the merge scatters nothing: the embeddings are handed the
    # ORIGINAL tensor. Propagating taint through it reported the merge used, so a
    # refactor dropping the effective merge kept every canary here green.
    ("embeddings take the original, reading only the merge device",
     "merged = inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")\n"
     "inputs_embeds = original.to(merged.device)"),
    ("embeddings take the original, reading only the merge dtype",
     "merged = inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")\n"
     "inputs_embeds = original.to(inputs_embeds.device, merged.dtype)"),
    ("return reads only the merge shape",
     "merged = inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")\n"
     "return original.reshape(-1, merged.shape[-1])"),
    ("the merge expression itself is consumed as metadata",
     "inputs_embeds = original.to(inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ").device)"),
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
    # The mirror image: element-by-element pairing must still see the merge when it
    # IS the element the embeddings receive, and a non-literal RHS keeps the whole
    # value, so an unpacking call still propagates it.
    ("destructuring hands the embeddings the merge",
     "merged = inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")\n"
     "inputs_embeds, aux = merged, original"),
    ("nested destructuring hands the embeddings the merge",
     "merged = inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")\n"
     "(inputs_embeds, aux), sizes = (merged, original), image_sizes"),
    ("unpacking call keeps the whole right-hand side",
     "merged = inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")\n"
     "inputs_embeds, aux = self.pack_image_features(merged)"),
    ("starred destructuring keeps the whole right-hand side",
     "merged = inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")\n"
     "inputs_embeds, *aux = merged, original"),
    # Skipping metadata must not skip a real use that merely mentions it alongside:
    # the merged VALUES still reach the embeddings here.
    ("result reshaped by its own shape",
     "merged = inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")\n"
     "inputs_embeds = merged.reshape(-1, merged.shape[-1])"),
    ("result returned alongside a metadata read of itself",
     "merged = inputs_embeds.masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")\n"
     "return merged.to(merged.device)"),
]


@pytest.mark.parametrize(
    "label,merge", _RESULT_REACHES_CONTROLS, ids = [c[0] for c in _RESULT_REACHES_CONTROLS],
)
def test_results_that_do_reach_the_embeddings_still_count(label, merge):
    # Controls: the merged tensor really does land back on `inputs_embeds`.
    found, aligned = _merge_dtype_alignment(
        _synthetic_forward(_ALIGNED_ASSIGN, merge = merge), "image_features",
    )
    assert found and aligned, f"{label}: a surviving merge result reported unaligned"


# ---------------------------------------------------------------------------
# A name is not only rebound by an assignment: a `for` target, a `with ... as`
# target and a named expression all replace the tensor the name refers to, so an
# earlier aligned assignment stops describing it. Skipping them lets the
# replacement, which nothing cast, reach masked_scatter.
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
# `masked_scatter` on any receiver let an aligned merge on a scratch tensor land in
# `fixed_casts`, which reads to `_patch_gemma4_audio_feature_dtype_on_class` as
# "upstream fixed it": it sets the marker and returns, leaving a real,
# still-device-only `inputs_embeds` merge unpatched and crashing.
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
# The same decoy with a receiver that MENTIONS `inputs_embeds` without being it.
# Matching the name anywhere in the receiver accepted this as the embedding merge;
# the receiver has to be anchored at its base instead.
_ALIGNED_SCRATCH_MERGE_ON_EMBEDS_DEVICE = (
    "scratch = scratch.to(inputs_embeds.device).masked_scatter(\n"
    "    audio_mask, audio_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
    ")"
)


def _audio_merge_casts(body):
    forward = _gemma4_model_forward_node(_audio_forward(body))
    assert forward is not None
    return g4p._gemma4_audio_merge_casts(forward)


def test_eager_matcher_ignores_merges_onto_other_tensors():
    # An aligned decoy on a scratch tensor is not the audio merge already fixed.
    buggy, fixed = _audio_merge_casts(_ALIGNED_SCRATCH_MERGE)
    assert not buggy and not fixed, (
        f"_gemma4_audio_merge_casts counted a `scratch.masked_scatter(...)` as the "
        f"embedding merge ({len(buggy)} buggy, {len(fixed)} fixed). The eager patch "
        f"would mark Gemma4Model as already patched on the strength of a merge that "
        f"never touches `inputs_embeds`."
    )


def test_eager_matcher_still_finds_the_real_merge_next_to_a_decoy():
    # The decoy is ignored AND the real device-only merge still picked up, so the
    # patch fires exactly once.
    buggy, fixed = _audio_merge_casts(_ALIGNED_SCRATCH_MERGE + "\n" + _REAL_AUDIO_MERGE)
    assert len(buggy) == 1 and not fixed, (
        f"expected the one real device-only `inputs_embeds` merge to be patchable "
        f"next to an aligned scratch decoy, got {len(buggy)} buggy / {len(fixed)} fixed"
    )


@pytest.mark.parametrize(
    "decoy",
    [_ALIGNED_SCRATCH_MERGE, _ALIGNED_SCRATCH_MERGE_ON_EMBEDS_DEVICE],
    ids = ["bare scratch receiver", "scratch receiver naming inputs_embeds"],
)
def test_a_decoy_merge_never_hides_the_real_one(decoy):
    # `scratch.to(inputs_embeds.device)` is a different tensor however often it
    # names `inputs_embeds`. Counting it puts an aligned cast in `fixed_casts`, and
    # `len(buggy) == 1 and fixed_casts` is exactly the branch on which the eager
    # patch returns without patching and without warning.
    buggy, fixed = _audio_merge_casts(decoy + "\n" + _REAL_AUDIO_MERGE)
    assert len(buggy) == 1 and not fixed, (
        f"expected the one real device-only `inputs_embeds` merge to be patchable "
        f"next to an aligned decoy, got {len(buggy)} buggy / {len(fixed)} fixed"
    )


def test_eager_matcher_still_accepts_a_transformed_inputs_embeds_receiver():
    # Discovery must stay loose about HOW the receiver mentions `inputs_embeds`
    # (blip_2 ships `inputs_embeds.to(...).masked_scatter(...)`), or a real merge
    # is dropped instead of patched.
    buggy, fixed = _audio_merge_casts(
        "inputs_embeds = inputs_embeds.to(audio_features.device).masked_scatter(\n"
        "    audio_mask.to(inputs_embeds.device), audio_features.to(inputs_embeds.device)\n"
        ")"
    )
    assert len(buggy) == 1 and not fixed, (
        f"a transformed `inputs_embeds` receiver must still be recognised as the "
        f"embedding merge, got {len(buggy)} buggy / {len(fixed)} fixed"
    )

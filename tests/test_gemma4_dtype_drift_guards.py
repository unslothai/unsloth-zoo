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


# Guard 3: eager and compiler PLE helpers must share one name; a second spelling is a cross-path NameError, caught here without a GPU or model.
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
    # The name the compiler REWRITE inserts must equal the one the helper DEFINES.
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


# Both helper copies are hand-maintained, so catch silent behavioural drift.
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


# Guard 1 & 2 behaviour (pure CPU torch): exercises the exact failing op.
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


# Reads the real gemma4 modeling file from disk: importing unsloth_zoo monkeypatches Gemma4Model.forward and can poison
# linecache for the PLE methods, so inspect.getsource returns unsloth-wrapped code; drift fails loudly if upstream has
# not fixed the dtype.
def _gemma4_modeling_source():
    import pathlib
    return pathlib.Path(inspect.getsourcefile(real_gemma4)).read_text()


# Traced structurally: each masked_scatter needs an inputs_embeds.dtype cast (at the merge on 5.15+, at the feature
# producer on 5.5-5.14), a receiver still at that dtype, and a used result; match/case and later overwrites untracked;
# .to(other_tensor) counts here, not in the eager matcher
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
    """True if node has a .to(...) landing on inputs_embeds' dtype: the explicit .to(inputs_embeds.device,
    inputs_embeds.dtype) or the tensor overload .to(inputs_embeds), which a healthy upstream preferred; only the
    first positional arg can be the overload."""
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


# Receiver methods that keep the destination dtype; anything else (.float(), .type(...), an unknown helper) can retype it and leave an fp32 destination.
_DTYPE_PRESERVING_METHODS = frozenset({
    "clone", "contiguous", "detach", "cpu", "cuda",
    "view", "reshape", "flatten", "squeeze", "unsqueeze",
    "expand", "expand_as", "transpose", "permute", "narrow",
})


# torch.<name> constants that are provably not dtypes (memory_format or layout instances), so they cannot retype a
# receiver; resolved against live torch, so unknown attributes count as dtypes.
_NON_DTYPE_TORCH_CONSTANTS = (torch.memory_format, torch.layout)


def _is_dtype_bearing(node, aliases = frozenset()):
    """Can node retype a tensor: a dtype (torch.float32, other.dtype, or an alias name); Tensor.view(dtype) retypes even when the dtype is held in a name."""
    if isinstance(node, ast.Name):
        return node.id in aliases
    if isinstance(node, ast.Attribute) and node.attr == "dtype":
        return not _is_attr(node, "inputs_embeds", "dtype")
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "torch"
    ):
        # `Tensor.view(torch.int32)` retypes but a memory format does not.
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
    """Is this .to(...) device-only or a no-op inputs_embeds.dtype cast? inputs_embeds.to(other.device) (blip_2) keeps
    dtype; .to(torch.float32), .to(other.dtype), .to(other_tensor) do not."""
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
    """Is the merge destination still at inputs_embeds.dtype? A cast source is only safe then: blip_2-style .to(device) chains pass, .float() receivers raise."""
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
    """ast.walk limited to root's own scope: nested def/class/lambda bodies are skipped (own locals; a helper may never
    be called), but the nested statement itself is still yielded."""
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
    """Returns [(block, index)] from the body inwards to the stmt holding target; only these run unconditionally relative to the merge (nested branches may not)."""
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
    """Only an exact <feature>.dtype != inputs_embeds.dtype test (either order) makes the untaken branch aligned; any other test leaves a reachable un-cast path."""
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
    """Plain names an assignment target binds, recursing into destructuring (llava_next unpacking rebinds); subscript
    and attribute targets are not rebindings (the name keeps its object)."""
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
# Plain-name rebinding nodes beyond assignments: for targets, with ... as name, and walrus :=, which replace the tensor and can reach masked_scatter un-cast.
_BINDING_NODES = _ASSIGN_NODES + _LOOP_NODES + _WITH_NODES + (ast.NamedExpr,)


def _binding_pairs(node):
    """[(target, value)] a binding node establishes; value is the expression whose dtype the bound name carries (RHS, loop iterable, or with context expression)."""
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
    """Part of value that feature receives: a literal tuple/list RHS splits per element (no crediting another name's
    cast); calls, starred and length mismatches keep the whole RHS."""
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


# Only assignments pair targets with RHS elements; a for or with target takes its value from the iterable or context manager instead.
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
    """Every rebinding of feature in stmt's own scope; += is not one (Tensor.add_ keeps the existing dtype), so the running alignment verdict stands."""
    return [
        node
        for node in _walk_own_scope(stmt, descend = False)
        if isinstance(node, _BINDING_NODES)
        and not isinstance(node, ast.AugAssign)
        and _assigns_feature(node, feature)
    ]


def _rebinding_alignment(stmt, feature):
    """Whether stmt rebinds feature on the path to the merge, and if so whether it is aligned; None when it does not rebind the name."""
    if (
        isinstance(stmt, _ASSIGN_NODES)
        and not isinstance(stmt, ast.AugAssign)
        and _assigns_feature(stmt, feature)
    ):
        return _binding_is_aligned(stmt, feature)
    if (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.NamedExpr)
        and _assigns_feature(stmt.value, feature)
    ):
        return _binding_is_aligned(stmt.value, feature)
    # Cast-if-needed guard (blip_2, granite_speech) is aligned by definition on the untaken branch; analysing both
    # branches as paths stops an aligned else hiding the mismatch path.
    if isinstance(stmt, ast.If) and _is_dtype_guard(stmt.test, feature):
        if _feature_rebindings(stmt, feature):
            # The body runs only on a mismatch (starts unaligned) and orelse only when dtypes agree (starts aligned); both ends must end aligned.
            return (
                _sequence_alignment(stmt.body, feature, False)
                and _sequence_alignment(stmt.orelse, feature, True)
            )
    # `if enabled: image_features = image_features.float()` still rebinds on the path.
    rebindings = _feature_rebindings(stmt, feature)
    if not rebindings:
        return None
    if isinstance(stmt, _WITH_NODES):
        # A with body runs unconditionally, so its rebindings dominate the merge; the as target binds first and seeds the alignment state.
        alignment = (
            _binding_is_aligned(stmt, feature)
            if _assigns_feature(stmt, feature) else None
        )
        for inner in stmt.body:
            inner_alignment = _rebinding_alignment(inner, feature)
            if inner_alignment is not None:
                alignment = inner_alignment
        return alignment
    # Conditional rebindings (if / for / while / try) may not run: an unaligned one reports unaligned, an aligned one leaves the earlier verdict (None).
    for rebinding in rebindings:
        if not _binding_is_aligned(rebinding, feature):
            return False
    return None


def _sequence_alignment(block, feature, initial):
    """Alignment of feature after running block top to bottom from initial; statements saying nothing about it (None, e.g. a conditional cast) leave it alone."""
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

# Attributes and methods that read tensor METADATA rather than values (device, dtype, shape, ...);
# fallback.to(image_features.device, ...) scatters no image_features value.
_METADATA_ATTRS = frozenset({
    "device", "dtype", "shape", "ndim", "size", "numel", "layout",
    "requires_grad", "is_cuda", "element_size", "itemsize", "stride",
})


def _carries_value(node, matches):
    """Whether matches-selected VALUES flow into node; metadata reads (.device, .dtype, .shape, ...) carry no values and are skipped, other uses still count."""
    if isinstance(node, ast.Attribute) and node.attr in _METADATA_ATTRS:
        return False
    if matches(node):
        return True
    return any(_carries_value(child, matches) for child in ast.iter_child_nodes(node))


def _carries_feature_value(node, feature):
    """Whether feature's VALUES flow into node; metadata reads (fallback.to(image_features.device, ...)) must not count
    or a dropped merge still reports (True, True)."""
    return _carries_value(node, lambda n: isinstance(n, ast.Name) and n.id == feature)


def _feature_merges(forward, feature):
    """Calls inputs_embeds.masked_scatter or masked_scatter_ whose receiver mentions inputs_embeds (loose, so
    transformed receivers count) and whose source carries feature's VALUES; nested defs skipped, since a helper's
    merge may never run."""
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


# The merge result follows the same value rule: a metadata read (merged.device, .dtype, .shape) discards the scattered
# values, so inputs_embeds = original.to(merged.device) returns the original.
def _contains(node, target):
    return _carries_value(node, lambda n: n is target)


def _references(node, names):
    return _carries_value(node, lambda n: isinstance(n, ast.Name) and n.id in names)


def _statements_after(forward, call):
    """Statements that run after call, innermost block outwards; the merged tensor must survive along this path to the return."""
    chain = _enclosing_blocks(forward, call)
    if not chain:
        return None
    ordered = []
    for depth, (block, index) in enumerate(reversed(chain)):
        # The innermost index points at the merge's own statement; outer indices point at the wrapping compound statement, so their slices start one past it.
        ordered.extend(block[index if depth == 0 else index + 1:])
    return ordered


def _merge_result_is_used(forward, call):
    """masked_scatter is out-of-place (upstream assigns it back on transformers 5.5.0-5.15.0), so its result must flow
    into inputs_embeds or a return, directly or via temporaries; a discarded or logged call fails; masked_scatter_ on
    a transformed receiver must hand its result back too."""
    if call.func.attr.endswith("_") and _is_embeds_tensor(call.func.value):
        return True
    statements = _statements_after(forward, call)
    if statements is None:
        return True

    tainted = set()
    for position, statement in enumerate(statements):
        for node in _walk_own_scope(statement, descend = False):
            if isinstance(node, ast.Return) and node.value is not None:
                if (position == 0 and _contains(node.value, call)) or _references(node.value, tainted):
                    return True
            if not isinstance(node, _BINDING_NODES):
                continue
            for target, value in _binding_pairs(node):
                if target is None or value is None:
                    continue
                # Each name is judged on the RHS element it receives: a whole-RHS read made inputs_embeds, aux =
                # original, merged taint inputs_embeds, though it receives the original.
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
                        # An unconditional rebinding to a value the merge did not flow into overwrites the name;
                        # conditional ones may not run, and += adds rather than replaces.
                        tainted.discard(name)
        if "inputs_embeds" in tainted:
            return True
    return False


def _dtype_alias_names(forward):
    """Local names bound to a dtype (target_dtype = torch.float32), resolved to a fixpoint so view(target_dtype)
    retypes; shape-valued names like target_shape do not."""
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
    """Returns (merge_found, every_merge_aligned); each merge consuming feature needs the destination still at
    inputs_embeds.dtype, a used result, and a cast source (at the call or last dominating assignment)."""
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
    # Image and video have always aligned dtype before the merge, so unsloth patches only audio; upstream dropping it there would be a new modality to patch.
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
    # The eager patch needs the merge source to BE the audio_features.to(inputs_embeds.device, ...) call (stricter than
    # the compiler regex), else the eager path silently no-ops.
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
    # Either upstream already casts dtype, or our regex still matches and inserts it.
    src = _gemma4_modeling_source()
    out = C.fix_gemma4_audio_feature_dtype(src)
    found, aligned = _merge_dtype_alignment(out, "audio_features")
    assert found and aligned, (
        "After fix_gemma4_audio_feature_dtype, `audio_features` still reaches "
        "masked_scatter with no `inputs_embeds.dtype` alignment on upstream source "
        "(regex drift or multiple matches)."
    )


def test_audio_compiler_transform_still_fixes_the_device_only_spelling():
    # transformers 5.15.0 fixed the audio merge upstream, so the real-source test above no-ops; the device-only spelling is still covered for 5.5.0-5.14.1.
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


# Tracer unit tests: a hole in _merge_dtype_alignment reports green on a source that crashes, so the spellings that slipped through are pinned here.
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


# Every path through a <feature>.dtype != inputs_embeds.dtype guard must end aligned; the textually last rebinding
# misses a mismatch path left un-cast (reachable masked_scatter_: expected self and source to have same dtypes).
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
    # Controls: dtype-safe on every path, so a red here fails a healthy upstream.
    found, aligned = _merge_dtype_alignment(_synthetic_forward(body), "image_features")
    assert found and aligned, f"{label}: dtype-safe spelling reported unaligned"


# A destructuring assignment rebinds the feature name like a plain one, e.g. image_features, feature_lens = self.pack_image_features(...) in llava_next.
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


# A destructuring target takes its RHS element, not the whole RHS, or a cast belonging to another name keeps the feature green (vilt writes literal tuples).
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


# image_features += delta is Tensor.add_, in place, so the name keeps its existing dtype: it neither inherits the RHS cast nor loses an earlier one.
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
    found, aligned = _merge_dtype_alignment(_synthetic_forward(body), "image_features")
    assert found and aligned, f"{label}: dtype-safe spelling reported unaligned"


# The source cast only makes a merge safe while the destination keeps inputs_embeds.dtype; a retyping receiver leaves an
# fp32 destination and fp16/bf16 source, which raises.
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
    # `Tensor.view(dtype)` reinterprets storage, so a dtype constant still counts.
    ("receiver .view(torch.int32) reinterprets the dtype",
     "inputs_embeds = inputs_embeds.view(torch.int32).masked_scatter(\n"
     "    image_mask, image_features.to(inputs_embeds.dtype)\n"
     ")"),
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
    # blip_2 really ships this spelling, and `.to(<device>)` keeps the dtype.
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
    # A memory format is not a dtype (higgs_audio_v2 ships `contiguous_format`).
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
    # `Tensor.view` also takes a shape, so only a name BOUND to a dtype may retype.
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
    found, aligned = _merge_dtype_alignment(
        _synthetic_forward(_ALIGNED_ASSIGN, merge = merge), "image_features",
    )
    assert found and aligned, f"{label}: dtype-safe spelling reported unaligned"


# masked_scatter is out-of-place: losing the surrounding inputs_embeds assignment silently drops the features; in-place
# masked_scatter_ (qwen2_5_omni, blt) needs none.
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
    # masked_scatter_ returns its receiver, so an in-place merge into a transformed receiver is as safe as out-of-place when its result is assigned back.
    ("in-place merge into a clone whose result is assigned back",
     "inputs_embeds = inputs_embeds.clone().masked_scatter_(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")"),
]


@pytest.mark.parametrize(
    "label,merge", _RESULT_USE_CONTROLS, ids = [case[0] for case in _RESULT_USE_CONTROLS],
)
def test_used_merge_results_still_count_as_alignment(label, merge):
    found, aligned = _merge_dtype_alignment(
        _synthetic_forward(_ALIGNED_ASSIGN, merge = merge), "image_features",
    )
    assert found and aligned, f"{label}: dtype-safe spelling reported unaligned"


# Merge discovery keys on feature VALUES reaching the scatter source; a metadata-only read (feature.device, .dtype,
# .shape) must not count, or a deleted merge would still report (True, True).
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


# Tensor.to(other) returns the dtype and device of other, so image_features.to(inputs_embeds) aligns both; reading only
# the explicit spelling false-reds healthy upstream.
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
    # inputs_embeds only carries the dtype as the first positional argument, since torch has no other= keyword.
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


# masked_scatter is out-of-place; a discarded result leaves forward returning unchanged inputs_embeds
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
    # masked_scatter_ writes the receiver's storage; on a clone or device-changed copy the merge is lost.
    ("in-place merge into a clone of the embeddings",
     "inputs_embeds.clone().masked_scatter_(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")"),
    ("in-place merge into a device-changed copy of the embeddings",
     "inputs_embeds.to('cuda').masked_scatter_(\n"
     "    image_mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype)\n"
     ")"),
    # Destructuring targets take their own RHS element; crediting the whole RHS hides a dead-name merge
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
    # Reading only the merge's device scatters nothing; embeddings still get the original tensor
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
    # Pairing must still see a merge that is the element embeddings get; a non-literal RHS stays whole
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
    # Skipping metadata must not skip a real use that mentions it alongside.
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
    found, aligned = _merge_dtype_alignment(
        _synthetic_forward(_ALIGNED_ASSIGN, merge = merge), "image_features",
    )
    assert found and aligned, f"{label}: a surviving merge result reported unaligned"


# for, with-as and walrus targets rebind a name, so an un-cast replacement reaches masked_scatter
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
    found, aligned = _merge_dtype_alignment(_synthetic_forward(body), "image_features")
    assert found and aligned, f"{label}: dtype-safe spelling reported unaligned"


# Eager matcher must tie to the embedding merge; an aligned scratch merge reads as upstream fixed it
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
# Matching inputs_embeds anywhere in the receiver wrongly accepts a receiver that only mentions it
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
    buggy, fixed = _audio_merge_casts(_ALIGNED_SCRATCH_MERGE)
    assert not buggy and not fixed, (
        f"_gemma4_audio_merge_casts counted a `scratch.masked_scatter(...)` as the "
        f"embedding merge ({len(buggy)} buggy, {len(fixed)} fixed). The eager patch "
        f"would mark Gemma4Model as already patched on the strength of a merge that "
        f"never touches `inputs_embeds`."
    )


def test_eager_matcher_still_finds_the_real_merge_next_to_a_decoy():
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
    # names `inputs_embeds`, and `len(buggy) == 1 and fixed_casts` is exactly the
    # branch on which the eager patch returns without patching and without warning.
    buggy, fixed = _audio_merge_casts(decoy + "\n" + _REAL_AUDIO_MERGE)
    assert len(buggy) == 1 and not fixed, (
        f"expected the one real device-only `inputs_embeds` merge to be patchable "
        f"next to an aligned decoy, got {len(buggy)} buggy / {len(fixed)} fixed"
    )


def test_eager_matcher_still_accepts_a_transformed_inputs_embeds_receiver():
    # Keep discovery loose on receivers like inputs_embeds.to(...).masked_scatter(...) or merges are lost
    buggy, fixed = _audio_merge_casts(
        "inputs_embeds = inputs_embeds.to(audio_features.device).masked_scatter(\n"
        "    audio_mask.to(inputs_embeds.device), audio_features.to(inputs_embeds.device)\n"
        ")"
    )
    assert len(buggy) == 1 and not fixed, (
        f"a transformed `inputs_embeds` receiver must still be recognised as the "
        f"embedding merge, got {len(buggy)} buggy / {len(fixed)} fixed"
    )

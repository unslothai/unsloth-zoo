"""CPU/CI regression guards for the Gemma 4 dtype fixes (PR #925).

These lock in the three failures we traced:

  1. Audio merge: upstream `Gemma4Model.forward` cast `audio_features.to(device)`
     without dtype (transformers #45192), while image/video cast dtype too ->
     `masked_scatter_: expected self and source to have same dtypes`.
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
import inspect
import re

import pytest
import torch

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


def _mentions(node, name):
    """Does the expression ``node`` read the local ``name`` anywhere inside it?"""
    return any(
        isinstance(n, ast.Name) and n.id == name for n in ast.walk(node)
    )


def _is_attr(node, owner, attr):
    """``<owner>.<attr>`` as an AST node, e.g. ``inputs_embeds.dtype``."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Name)
        and node.value.id == owner
    )


def _is_embeds_dtype(node):
    return _is_attr(node, "inputs_embeds", "dtype")


def _is_device_expr(node):
    """``x.device``, ``"cuda"`` or ``torch.device(...)``: a device, not a dtype."""
    if isinstance(node, ast.Attribute) and node.attr == "device":
        return True
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "device"
    )


def _is_dtype_bearing(node):
    """Can ``node`` be a dtype argument (``torch.float32``, ``other.dtype``)?"""
    if isinstance(node, ast.Attribute) and node.attr == "dtype":
        return True
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "torch"
    )


def _is_embeds_tensor(node):
    return isinstance(node, ast.Name) and node.id == "inputs_embeds"


def _call_aligns_dtype(call):
    """Does this call land the tensor ON ``inputs_embeds``' dtype?

    Three spellings count. `.to(inputs_embeds.device, inputs_embeds.dtype)` and
    `.to(dtype = inputs_embeds.dtype)` name the dtype; `.to(inputs_embeds)` and
    `.type_as(inputs_embeds)` take it from the tensor itself, which torch
    documents as returning a tensor with the same dtype (and, for `.to`, device)
    as the argument. Only a dtype ARGUMENT counts, so a bare mention of
    `inputs_embeds.dtype` somewhere else in the expression proves nothing.
    """
    method = call.func.attr
    if method == "type_as":
        return bool(call.args) and _is_embeds_tensor(call.args[0])
    if method != "to":
        return False
    if any(_is_embeds_dtype(a) for a in call.args):
        return True
    if any(kw.arg == "dtype" and _is_embeds_dtype(kw.value) for kw in call.keywords):
        return True
    # torch spells the tensor overload positionally only; there is no `other=`.
    return bool(call.args) and _is_embeds_tensor(call.args[0])


def _to_call_is_device_only(call):
    """Is this ``.to(...)`` a pure device move, which keeps the dtype?

    `to(device = None, dtype = None, ...)` infers `dtype` from `self` when it is
    not given, so `features.to(inputs_embeds.device)` cannot retype anything.
    """
    for arg in call.args:
        if not _is_device_expr(arg):
            return False
    for kw in call.keywords:
        if kw.arg in ("non_blocking", "copy", "memory_format"):
            continue
        if kw.arg == "device" and _is_device_expr(kw.value):
            continue
        return False
    return True


# Methods that reshape or relocate a tensor without retyping it, so the dtype the
# merge sees is still the one its reaching definition established.
_DTYPE_PRESERVING_METHODS = frozenset({
    "clone", "contiguous", "detach", "cpu", "cuda",
    "view", "reshape", "flatten", "squeeze", "unsqueeze",
    "expand", "expand_as", "transpose", "permute", "narrow",
})


def _expr_alignment(node, name):
    """Alignment of the value ``node`` produces, for the tensor called ``name``.

    True  -> the expression itself casts to inputs_embeds' dtype.
    None  -> it only moves/reshapes ``name``, so its dtype is whatever reached it.
    False -> anything else, including an expression that retypes (`.float()`) or
             one built from some other tensor entirely.
    """
    while True:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            method = node.func.attr
            if _call_aligns_dtype(node):
                return True
            if method == "to":
                if not _to_call_is_device_only(node):
                    return False
            elif method in _DTYPE_PRESERVING_METHODS:
                values = list(node.args) + [kw.value for kw in node.keywords]
                if any(_is_dtype_bearing(v) for v in values):
                    return False
            else:
                return False
            node = node.func.value
            continue
        if isinstance(node, ast.Subscript):
            node = node.value
            continue
        if isinstance(node, ast.Name):
            return None if node.id == name else False
        return False


def _bound_names(target):
    """Every plain local name an assignment target binds.

    `image_features[0] = ...` and `self.image_features = ...` are NOT bindings:
    the name keeps referring to the same object, so the RHS of a slice write is
    not the definition that reaches the merge.
    """
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [n for elt in target.elts for n in _bound_names(elt)]
    if isinstance(target, ast.Starred):
        return _bound_names(target.value)
    return []


_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _walk_own_scope(root, descend = True):
    """``ast.walk`` restricted to the code ``root``'s own scope executes.

    A nested `def` / `class` / `lambda` has its own locals, so a cast written in
    another method says nothing about the tensor this one scatters. The nested
    node itself is still yielded, only its body is skipped.
    """
    if isinstance(root, _SCOPE_NODES) and not descend:
        yield root
        return
    todo = [root]
    while todo:
        node = todo.pop(0)
        yield node
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _SCOPE_NODES):
                yield child
            else:
                todo.append(child)


def _child_blocks(stmt):
    """Every nested statement list of ``stmt`` (if/else, loop bodies, try)."""
    blocks = []
    for field in ("body", "orelse", "finalbody"):
        value = getattr(stmt, field, None)
        if isinstance(value, list) and value and all(isinstance(s, ast.stmt) for s in value):
            blocks.append(value)
    for handler in getattr(stmt, "handlers", None) or []:
        blocks.append(handler.body)
    return blocks


def _enclosing_blocks(scope, target):
    """``[(block, index)]`` from the scope body inwards to the stmt holding ``target``.

    Only these statement lists run on the path to ``target``. A cast stranded in
    a branch the merge does not run under is not a reaching definition, however
    late in the file it is written.
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

    descend(scope.body)
    return chain


def _rebinding_values(stmt, name):
    """The value expressions every rebinding of ``name`` inside ``stmt`` gives it."""
    values = []
    for node in _walk_own_scope(stmt, descend = False):
        if isinstance(node, ast.Assign):
            pairs = [(t, node.value) for t in node.targets]
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
            pairs = [(node.target, node.value)]
        else:
            continue
        for target, value in pairs:
            if name in _bound_names(target):
                values.append(value)
    return values


def _reaching_alignment(scope, name, target):
    """Fold the statements that reach ``target`` into "is ``name`` dtype-aligned?".

    Every rebinding on the way has to leave the tensor aligned. A rebinding that
    only moves or reshapes it inherits the running verdict, so a device move
    hoisted out of the merge call does not read as a regression; one that can
    retype it, or that is stranded in a branch alongside an unaligned sibling,
    makes the whole thing unaligned.
    """
    state, seen = False, False
    for block, index in _enclosing_blocks(scope, target):
        for stmt in block[:index]:
            values = _rebinding_values(stmt, name)
            if not values:
                continue
            seen = True
            resolved = [_expr_alignment(v, name) for v in values]
            state = all(state if r is None else r for r in resolved)
    return state, seen


def _feature_merges(scope, name):
    """Every ``inputs_embeds.masked_scatter[_](mask, <name> ...)`` in ``scope``.

    The receiver has to reference `inputs_embeds`: an auxiliary scatter into some
    other tensor is not the merge these guards protect, and letting one in makes
    the canary answer for the wrong call in both directions. Only the SOURCE
    operand is inspected, since that is the tensor whose dtype `masked_scatter`
    requires to match; the mask is a bool tensor and its dtype is irrelevant.
    """
    found = []
    for node in _walk_own_scope(scope):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("masked_scatter", "masked_scatter_")
        ):
            continue
        if not _mentions(node.func.value, "inputs_embeds"):
            continue
        source = node.args[1] if len(node.args) >= 2 else None
        if source is None:
            source = next((kw.value for kw in node.keywords if kw.arg == "source"), None)
        if source is not None and _mentions(source, name):
            found.append((node, source))
    return found


def _scopes(tree):
    """The module plus every function body, each as its own name scope."""
    yield tree
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _merge_is_dtype_aligned(src, modality):
    """Is the ``<modality>_features`` tensor that reaches the merge dtype-aligned?

    Structural, not textual: find the `inputs_embeds.masked_scatter` that
    consumes the tensor, then ask whether it lands on ``inputs_embeds.dtype``
    either in the scattered expression itself or in the definitions that reach
    it along the merge's own control-flow path, inside the merge's own scope.
    Answering from the reaching definition (rather than from any mention of the
    name anywhere in the file) is what stops a later retyping rebinding, or a
    cast in some unrelated method or untaken branch, from standing in for the
    cast that actually runs.

    Returns (ok, detail) so a failure can say what it saw.
    """
    tree = ast.parse(src)
    name = f"{modality}_features"
    total = 0
    for scope in _scopes(tree):
        for call, source in _feature_merges(scope, name):
            total += 1
            at_call = _expr_alignment(source, name)
            if at_call is True:
                continue
            if at_call is False:
                # The scattered expression rebuilds or retypes the tensor, so
                # whatever the reaching definition established no longer holds.
                return False, (
                    f"the expression scattered at line {call.lineno} does not "
                    f"reach inputs_embeds.dtype: {ast.unparse(source).strip()!r}"
                )
            aligned, seen = _reaching_alignment(scope, name, call)
            if not seen:
                return False, (
                    f"{name} is scattered at line {call.lineno} with no "
                    f"assignment reaching it"
                )
            if not aligned:
                return False, (
                    f"the {name} binding reaching the merge at line "
                    f"{call.lineno} does not cast to inputs_embeds.dtype: "
                    f"{ast.unparse(source).strip()!r}"
                )
    if not total:
        return False, f"no masked_scatter consuming {name}"
    return True, f"{total} merge(s) dtype-aligned"


@requires_gemma4
def test_real_gemma4_audio_merge_site_is_recognized():
    src = _gemma4_modeling_source()
    buggy = "audio_features.to(inputs_embeds.device)"
    fixed = "audio_features.to(inputs_embeds.device, inputs_embeds.dtype)"
    assert (buggy in src) or (fixed in src), (
        "Gemma4 audio merge site drifted: neither the known device-only pattern "
        "nor the fixed device+dtype pattern is present in modeling_gemma4.py. The "
        "unsloth audio patch would silently no-op and the masked_scatter dtype crash "
        "would return. Update _patch_gemma4_audio_feature_dtype_on_class (eager) and "
        "fix_gemma4_audio_feature_dtype (compiler)."
    )


@requires_gemma4
@pytest.mark.parametrize("modality", ["image", "video"])
def test_real_gemma4_image_and_video_merges_still_cast_dtype(modality):
    # Regression anchor: image/video have always cast dtype; if upstream ever
    # drops it there too, that is a new modality that also needs patching.
    #
    # Asked of the tensor that REACHES the merge rather than of one exact call
    # spelling. transformers 5.15.0 reshaped the image branch to
    #
    #   image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    #   ...
    #   inputs_embeds = inputs_embeds.masked_scatter(image_mask.to(...), image_features.to(inputs_embeds.device))
    #
    # which still casts, but no longer contains the literal
    # "image_features.to(inputs_embeds.device, inputs_embeds.dtype)" the guard
    # used to grep for. Only the spelling moved, so the substring match went red
    # on a file that is still correct. What matters is that the value the
    # masked_scatter consumes carries inputs_embeds.dtype, and that survives a
    # rename of whatever produced the tensor.
    ok, detail = _merge_is_dtype_aligned(_gemma4_modeling_source(), modality)
    assert ok, (
        f"Gemma4 {modality} merge no longer casts to inputs_embeds.dtype: {detail}. "
        f"That is a modality the unsloth dtype patches do not cover, and "
        f"masked_scatter will raise on a dtype mismatch."
    )


# Spellings the guard above has to keep straight. Kept out of the gemma4-only
# section on purpose: these run everywhere, so the guard cannot quietly go
# vacuous on the hosts that do not have a gemma4 to read.
_MERGE_SHAPES = {
    # transformers 5.15.0: the cast moved into the torch.cat that builds the tensor.
    "cat_then_merge": ("""
def forward(self, inputs_embeds):
    image_features = self.get_image_features(pixel_values).pooler_output
    image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(
        image_mask.to(inputs_embeds.device), image_features.to(inputs_embeds.device)
    )
""", True),
    # transformers 5.5.0: cast on a plain rebinding of the same name.
    "rebind_then_merge": ("""
def forward(self, inputs_embeds):
    image_features = self.get_image_features(pixel_values).pooler_output
    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(
        image_mask.to(inputs_embeds.device), image_features.to(inputs_embeds.device)
    )
""", True),
    # Cast written at the merge itself (what the audio branch does since 5.15.0).
    "cast_at_merge": ("""
def forward(self, inputs_embeds):
    image_features = self.get_image_features(pixel_values).pooler_output
    inputs_embeds = inputs_embeds.masked_scatter(
        image_mask.to(inputs_embeds.device), image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    )
""", True),
    # Keyword dtype= instead of the positional overload.
    "keyword_dtype": ("""
def forward(self, inputs_embeds):
    image_features = self.get_image_features(pixel_values).to(dtype=inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
""", True),
    # The regression itself: device only, dtype dropped.
    "device_only": ("""
def forward(self, inputs_embeds):
    image_features = self.get_image_features(pixel_values).pooler_output
    image_features = image_features.to(inputs_embeds.device)
    inputs_embeds = inputs_embeds.masked_scatter(
        image_mask.to(inputs_embeds.device), image_features.to(inputs_embeds.device)
    )
""", False),
    # A good cast that a later device-only rebinding undoes before the merge.
    # This is the case a "does the cast appear anywhere" search gets wrong.
    "cast_then_undone": ("""
def forward(self, inputs_embeds):
    image_features = self.get_image_features(pixel_values).to(inputs_embeds.device, inputs_embeds.dtype)
    image_features = postprocess(image_features).to(inputs_embeds.device)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
""", False),
    # A cast that lives in a different method entirely must not count either.
    "cast_in_another_method": ("""
def get_image_features(self, pixel_values, inputs_embeds):
    return self.vision_tower(pixel_values).to(inputs_embeds.device, inputs_embeds.dtype)

def forward(self, inputs_embeds):
    image_features = self.get_image_features(pixel_values, inputs_embeds)
    image_features = image_features.to(inputs_embeds.device)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
""", False),
    # torch's tensor overload: `.to(other)` takes other's dtype AND device, so
    # this is aligned even though it never spells `inputs_embeds.dtype`.
    "tensor_overload": ("""
def forward(self, inputs_embeds):
    image_features = self.get_image_features(pixel_values).to(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
""", True),
    # `.type_as` is the dtype-only spelling of the same thing.
    "type_as": ("""
def forward(self, inputs_embeds):
    image_features = self.get_image_features(pixel_values).type_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
""", True),
    # A device move hoisted OUT of the merge call keeps the earlier cast: `.to()`
    # infers dtype from self when none is given, so this cannot retype anything.
    "device_move_after_cast": ("""
def forward(self, inputs_embeds):
    image_features = self.get_image_features(pixel_values).pooler_output
    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    image_features = image_features.to(inputs_embeds.device)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
""", True),
    # An auxiliary scatter into some OTHER tensor is not the merge we protect,
    # and must not be able to fail the guard on a correct inputs_embeds merge.
    "auxiliary_scatter_elsewhere": ("""
def forward(self, inputs_embeds):
    image_features = self.get_image_features(pixel_values).to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
    scratch = torch.zeros_like(inputs_embeds)
    image_features = image_features.float()
    scratch = scratch.masked_scatter(image_mask, image_features)
""", True),
    # A mention of inputs_embeds.dtype that is not a cast argument proves nothing.
    "dtype_mentioned_but_not_applied": ("""
def forward(self, inputs_embeds):
    image_features = self.get_image_features(pixel_values).to(inputs_embeds.device) if inputs_embeds.dtype is not None else image_features
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
""", False),
    # A cast in the branch the merge does NOT come through is not a definition
    # that reaches it.
    "cast_in_untaken_branch": ("""
def forward(self, inputs_embeds):
    image_features = self.get_image_features(pixel_values).pooler_output
    if fast_path:
        image_features = image_features.to(inputs_embeds.device)
    else:
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
""", False),
    # The mask operand's dtype is irrelevant; only the source has to line up.
    "cast_on_the_mask_only": ("""
def forward(self, inputs_embeds):
    image_features = self.get_image_features(pixel_values).to(inputs_embeds.device)
    inputs_embeds = inputs_embeds.masked_scatter(
        image_features.to(inputs_embeds.dtype).bool(), image_features
    )
""", False),
    # A slice write does not rebind the name, so its RHS is not the definition
    # that reaches the merge.
    "subscript_write_is_not_a_binding": ("""
def forward(self, inputs_embeds):
    image_features = self.get_image_features(pixel_values).to(inputs_embeds.device)
    image_features[0] = replacement.to(inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
""", False),
    # The scattered expression retypes the tensor, so an earlier good cast is
    # already spent by the time masked_scatter sees it.
    "source_overrides_the_dtype": ("""
def forward(self, inputs_embeds):
    image_features = self.get_image_features(pixel_values).to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features.float())
""", False),
    # No merge at all is a drift we want to hear about, not a silent pass.
    "no_merge": ("""
def forward(self, inputs_embeds):
    image_features = self.get_image_features(pixel_values).to(inputs_embeds.device, inputs_embeds.dtype)
    return image_features
""", False),
}


@pytest.mark.parametrize("shape", sorted(_MERGE_SHAPES))
def test_the_dtype_cast_guard_reads_the_reaching_definition(shape):
    """Prove the guard accepts every spelling upstream has shipped, and still fails.

    Runs without gemma4 installed, unlike the guard itself, which is the point: a
    guard that only ever runs on hosts with the newest transformers is one nobody
    notices going vacuous."""
    src, expected = _MERGE_SHAPES[shape]
    ok, detail = _merge_is_dtype_aligned(src, "image")
    assert ok is expected, f"{shape}: expected ok={expected}, got {ok} ({detail})"


def test_the_dtype_cast_guard_does_not_confuse_modalities():
    src, _ = _MERGE_SHAPES["cat_then_merge"]
    ok, detail = _merge_is_dtype_aligned(src, "video")
    assert not ok and "no masked_scatter" in detail, detail


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
    assert "audio_features.to(inputs_embeds.device, inputs_embeds.dtype)" in out, (
        "fix_gemma4_audio_feature_dtype no longer produces the dtype-aligned cast on "
        "upstream source (regex drift or multiple matches)."
    )

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
# so a cast stranded in a branch the merge does not run under does not count).
def _is_attr(node, owner, attr):
    """`<owner>.<attr>` as an AST node, e.g. `inputs_embeds.dtype`."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Name)
        and node.value.id == owner
    )


def _casts_to_inputs_embeds_dtype(node):
    """True if `node` contains a `.to(...)` carrying `inputs_embeds.dtype`."""
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
    return False


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


def _assigns_feature(stmt, feature):
    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
    return any(feature in _bound_names(t) for t in targets)


_ASSIGN_NODES = (ast.Assign, ast.AnnAssign, ast.AugAssign)


def _feature_rebindings(stmt, feature):
    """Every assignment to `<feature>` anywhere inside `stmt`'s own scope."""
    return [
        node
        for node in _walk_own_scope(stmt, descend = False)
        if isinstance(node, _ASSIGN_NODES) and _assigns_feature(node, feature)
    ]


def _rebinding_alignment(stmt, feature):
    """Does `stmt` rebind `<feature>` on the path to the merge, and is it aligned?

    `None` when `stmt` does not rebind the name (so it says nothing either way).
    """
    if isinstance(stmt, _ASSIGN_NODES) and _assigns_feature(stmt, feature):
        value = getattr(stmt, "value", None)
        return value is not None and _casts_to_inputs_embeds_dtype(value)
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
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        # A `with` body runs unconditionally, so its rebindings dominate the
        # merge exactly like top-level ones: take the last one that says
        # anything.
        alignment = None
        for inner in stmt.body:
            inner_alignment = _rebinding_alignment(inner, feature)
            if inner_alignment is not None:
                alignment = inner_alignment
        return alignment
    # Conditionally executed (`if` / `for` / `while` / `try`): the rebinding may
    # or may not run, so the state after it is "aligned before AND aligned in
    # every branch". An unaligned branch therefore reports unaligned; a branch
    # that only ever casts to `inputs_embeds.dtype` leaves the earlier verdict
    # standing (`None`) instead of overriding it.
    for rebinding in rebindings:
        value = getattr(rebinding, "value", None)
        if value is None or not _casts_to_inputs_embeds_dtype(value):
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


def _feature_merge_args(forward, feature):
    """Source args of every `inputs_embeds.masked_scatter(mask, ...)` using `<feature>`.

    The receiver has to be `inputs_embeds` (or something derived from it): a
    `scratch.masked_scatter(mask, feature)` elsewhere in `forward` is a different
    tensor and says nothing about the dtype of the embedding merge. Nested `def`
    / `class` / `lambda` bodies are skipped for the same reason: a merge in a
    helper `forward` never invokes is not `forward`'s merge.
    """
    args = []
    for node in _walk_own_scope(forward):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "masked_scatter"
            and len(node.args) >= 2
        ):
            continue
        if not any(
            isinstance(n, ast.Name) and n.id == "inputs_embeds"
            for n in ast.walk(node.func.value)
        ):
            continue
        source_arg = node.args[1]
        if any(
            isinstance(n, ast.Name) and n.id == feature for n in ast.walk(source_arg)
        ):
            args.append(source_arg)
    return sorted(args, key=lambda n: (n.lineno, n.col_offset))


def _merge_dtype_alignment(src, feature):
    """Trace `<feature>` into `inputs_embeds.masked_scatter(...)`.

    Returns (merge_found, every_merge_aligned). EVERY merge consuming the feature
    has to be aligned: one unaligned branch is one reachable dtype crash, so an
    aligned first merge must not excuse an unaligned second one.
    """
    forward = _gemma4_model_forward_node(src)
    if forward is None:
        return (False, False)
    merges = _feature_merge_args(forward, feature)
    if not merges:
        return (False, False)
    aligned = all(
        _casts_to_inputs_embeds_dtype(merge)
        or _dominating_alignment(forward, feature, merge)
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
def _synthetic_forward(body):
    """Minimal `Gemma4Model.forward` with `body` between the features and the merge."""
    indented = "\n".join("            " + line if line else "" for line in body.strip("\n").split("\n"))
    return (
        "class Gemma4Model:\n"
        "    def forward(self, inputs_embeds, pixel_values, image_mask):\n"
        "        if pixel_values is not None:\n"
        "            image_features = self.get_image_features(pixel_values)\n"
        f"{indented}\n"
        "            inputs_embeds = inputs_embeds.masked_scatter(\n"
        "                image_mask.to(inputs_embeds.device), image_features.to(inputs_embeds.device)\n"
        "            )\n"
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

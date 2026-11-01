#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Refuse dangerous GitHub Actions trigger patterns at PR time.

Flags three patterns from the TanStack GHSA-g7cv-rxg3-hmpx compromise:

1.  `pull_request_target` -- runs a fork's workflow YAML against the BASE
    repo's secrets/permissions, letting the fork inject code into the base
    context. No safe use for public projects; use `pull_request` instead.

2.  `workflow_run` chained to a PR-triggered workflow -- same trust-boundary
    problem one hop later: a PR can poison artifacts/caches that the
    elevated-permission run then consumes.

3.  Shared cache keys between PR-triggered and publish/release/push
    workflows -- a fork PR can poison the cache the release run restores.
    Keys must be partitioned so secrets-holding workflows never read what a
    PR can write.

Exit codes: 0 = no findings; 1 = findings (stderr lists each with file path).

Run from repo root:
    python3 scripts/lint_workflow_triggers.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path, PurePosixPath

try:
    import yaml
except ImportError:
    print(
        "ERROR: PyYAML is required. Install with 'pip install pyyaml'", file = sys.stderr
    )
    sys.exit(2)

REPO_ROOT = Path(__file__).resolve().parents[1]
# Kept in one place because every reader below opens files the same way.
ENC = "utf-8"

DEFAULT_WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

BANNED_TRIGGERS: tuple[str, ...] = ("pull_request_target",)
RESTRICTED_TRIGGERS: tuple[str, ...] = ("workflow_run",)
PUBLISH_WORKFLOW_NAMES: tuple[str, ...] = ("release-desktop.yml",)


def _normalise_on(on_field):
    if isinstance(on_field, str):
        return {on_field}
    if isinstance(on_field, list):
        return set(on_field)
    if isinstance(on_field, dict):
        return set(on_field.keys())
    return set()


def _load_workflow(path: Path):
    try:
        return yaml.safe_load(path.read_text())
    except Exception as exc:
        print(f"ERROR: failed to parse {path}: {exc}", file = sys.stderr)
        sys.exit(2)


def _mappings(node):
    """Every mapping anywhere in a parsed document, at any depth."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _mappings(value)
    elif isinstance(node, list):
        for item in node:
            yield from _mappings(item)


def _parse(path: Path):
    try:
        return yaml.safe_load(path.read_text(ENC))
    except Exception:
        return None


def _extract_cache_keys(path: Path) -> list[str]:
    """All `key:` declarations from the parsed YAML; text scanning missed valid spellings such as `uses :` or flow style, which is a bypass here."""
    doc = _parse(path)
    keys: list[str] = []
    for mapping in _mappings(doc):
        value = mapping.get("key")
        if isinstance(value, (str, int, float)):
            text = str(value).strip()
            if text:
                keys.append(text)
    return keys


def _extract_restore_key_prefixes(path: Path) -> list[str]:
    """Fallback prefixes from restore-keys, taken from the parsed value that actions/cache receives, since restore-keys
    matches by prefix; PyYAML handles literal and folded blocks and sequences."""
    doc = _parse(path)
    prefixes: list[str] = []
    for mapping in _mappings(doc):
        value = mapping.get("restore-keys")
        if isinstance(value, str):
            prefixes.extend(line.strip() for line in value.splitlines())
        elif isinstance(value, list):
            prefixes.extend(str(item).strip() for item in value)
    return [p for p in prefixes if p]


def _local_uses(path: Path) -> list[str]:
    """Every `uses:` value in the document that points inside this repository."""
    doc = _parse(path)
    out: list[str] = []
    for mapping in _mappings(doc):
        value = mapping.get("uses")
        if isinstance(value, str) and value.strip().startswith("./"):
            out.append(value.strip())
    return out


def _call_sites(path: Path, target: str) -> list[dict]:
    """The with: mappings of every call to a local action or workflow named target, where target is the last path component of its uses: ./... reference."""
    doc = _parse(path)
    sites: list = []
    jobs = doc.get("jobs") if isinstance(doc, dict) else None
    scoped = (
        [(str(jid), job) for jid, job in jobs.items()]
        if isinstance(jobs, dict)
        else [("", doc)]
    )
    for job_id, container in scoped:
      for mapping in _mappings(container):
        value = mapping.get("uses")
        if not isinstance(value, str):
            continue
        ref = value.strip()
        if not ref.startswith("./"):
            continue
        # Compared as the FULL local reference when one is given, because two actions
        # may share a directory name: `./.github/actions/a/cache` and
        # `./.github/actions/b/cache` both end in `cache`, so matching on the last
        # component pooled their call sites and invented one action's value for the
        # other. A bare name is still accepted, which is what the predicate self-tests
        # and the shell-head narrowing pass.
        full = ref[2:].rstrip("/") if ref.startswith("./") else ref.rstrip("/")
        # ./repo/.github/actions/cache names the same action as ./.github/actions/cache; the runtime prefix must be
        # stripped here too, as reachability already does.
        segments = full.split("/")
        if ".github" in segments[1:]:
            full = "/".join(segments[segments.index(".github"):])
        if "/" in target:
            if full != target and PurePosixPath(full).parent.as_posix() != target:
                continue
        else:
            stem = full.split("/")[-1]
            if stem != target and PurePosixPath(stem).stem != PurePosixPath(target).stem:
                continue
        with_ = mapping.get("with")
        # The CALLER's scope travels with the call site, because a value it delegates
        # names a step of the caller's job, not of the target. Resolving
        # `${{ steps.pip-cache.outputs.key }}` in the target's scope looked for a step
        # the target does not have, so this repository's own pip-cache-save read as
        # unresolvable and failed the live tree.
        sites.append((
            with_ if isinstance(with_, dict) else {},
            (path.as_posix(), job_id),
        ))
    return sites



_STEP_OUTPUT = re.compile(
    r"\$\{\{\s*steps\.([A-Za-z_][\w-]*)\.outputs\.([A-Za-z_][\w-]*)\s*\}\}"
)

# Which outputs an `actions/cache` step publishes from the `key:` it was given.
_CACHE_STEP_OUTPUTS = frozenset({"cache-primary-key", "cache-matched-key", "key"})


def _recovered_outputs(text: str) -> set:
    """Output names a shell body writes with a recoverable value (a literal or a literal head before the first
    expansion), judged per output; a printf %s value is not."""
    # Shell variables with a literal head make outputs built from them recoverable, as this repository's producer does:
    # key=${prefix}${hash} with prefix=pip-v2-${name}-...; the output line alone recovered nothing.
    assigned: dict = {}
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        for name, value in re.findall(
            r"^([A-Za-z_]\w*)=[\"']?([^\"'\n]*)", stripped
        ):
            head = value.split("$", 1)[0]
            if head and "%" not in head:
                assigned[name] = head

    def _head(value: str) -> str:
        """The literal text this value starts with, after one pass of substitution."""
        expanded = re.sub(
            r"\$\{?([A-Za-z_]\w*)\}?",
            lambda m: assigned.get(m.group(1), "$"),
            value,
        )
        return expanded.split("$", 1)[0]

    found: set = set()
    for line in text.splitlines():
        stripped = line.split("#", 1)[0]
        if "GITHUB_OUTPUT" not in stripped:
            continue
        for name, value in re.findall(
            r"\b([A-Za-z_][\w-]*)=([^\s'\"]*)", stripped
        ):
            if not value or "%" in value:
                continue
            # Recovered when a value is literal or starts with a literal head once known shell variables are substituted; a printf %s value is neither.
            if _head(value):
                found.add(name)
    return found




_CACHE_KEY_OUTPUTS = ("cache-primary-key", "cache-matched-key", "key")




def _cache_withs(node):
    """The with: mappings of steps that cache or call a local ./ action, which may cache internally; a key given to unrelated actions is not a cache namespace."""
    for mapping in _mappings(node):
        uses = str(mapping.get("uses") or "").strip()
        if not uses:
            continue
        with_ = mapping.get("with")
        if not isinstance(with_, dict):
            continue
        action = uses.split("@")[0].casefold()
        if action.startswith("actions/cache") or uses.startswith("./"):
            yield with_


def _scoped_cache_keys(path: Path) -> list:
    """[(job id, key)] so a key carries the scope its step ids are resolved in.

    Step ids are unique only within a job, so a key referring to
    `${{ steps.probe.outputs.key }}` means THIS job's `probe`. Without the scope the
    only safe reading was "every step of that id, anywhere, must be readable", and this
    repository has three unrelated `id: probe` steps -- two of which emit no cache key
    at all -- so the safe reading failed a correct tree. The scope makes the question
    answerable instead of merely conservative.
    """
    doc = _parse(path)
    out = []
    jobs = doc.get("jobs") if isinstance(doc, dict) else None
    if isinstance(jobs, dict):
        for job_id, job in jobs.items():
            for mapping in _cache_withs(job):
                key = mapping.get("key")
                if key is not None and not isinstance(key, (dict, list)):
                    # Stringify the key as _extract_cache_keys does: accepting only str dropped unquoted scalars like
                    # key: 123 (a YAML int), hiding shared-key collisions.
                    out.append((str(job_id), str(key)))
        return out
    for mapping in _cache_withs(doc):
        key = mapping.get("key")
        if key is not None and not isinstance(key, (dict, list)):
            out.append(("", str(key)))
    return out


def _producer_steps(path: Path) -> dict:
    """{(document, job, step id): was its key output actually recovered}.

    Keyed by the whole scope, because a step id is unique only within its job. Merging
    bare ids let a readable `id: probe` in one job -- or in a later-sorted file -- stand
    in for an unreadable `id: probe` in another, and the unread key was then dismissed
    on the strength of a step that has nothing to do with it. The dictionary is built
    globally, so a bare id was exactly the wrong identity.

    Per producer, because "some producer somewhere was readable" is not evidence about
    THIS one. A side-wide flag let a workflow with one ordinary
    `echo 'key=safe-key' >> "$GITHUB_OUTPUT"` vouch for a second step emitting
    `printf 'key=%s\n' "shared-$GITHUB_SHA"`, whose namespace was never recovered: the
    flag was true, the delegated key was dismissed, and a publish `restore-keys:
    shared-` passed. One readable step is not a warrant for an unreadable one.
    """
    out: dict = {}
    doc = _parse(path)
    jobs = doc.get("jobs") if isinstance(doc, dict) else None
    scopes = []
    if isinstance(jobs, dict):
        scopes = [(str(jid), job) for jid, job in jobs.items()]
    else:
        # An action definition has no jobs; its steps share one scope.
        scopes = [("", doc)]
    for job_id, scope in scopes:
      for mapping in _mappings(scope):
        step_id = mapping.get("id")
        if not isinstance(step_id, str):
            continue
        ident = (path.as_posix(), job_id, step_id)
        # A step declaring its own key: publishes it as cache-primary-key, as actions/cache/restore steps do; reading only run: bodies missed these.
        with_ = mapping.get("with")
        uses_value = str(mapping.get("uses") or "")
        if (
            isinstance(with_, dict)
            and with_.get("key") is not None
            and uses_value.strip().split("@")[0].casefold().startswith("actions/cache")
        ):
            # Only for actions/cache, whose published key output is this input; marking every id-bearing step with
            # with.key readable was too generous, since a local action may take an unrelated key input while emitting
            # its own key output.
            out[ident] = set(_CACHE_STEP_OUTPUTS)
            continue
        body = mapping.get("run")
        if isinstance(body, str):
            out[ident] = _recovered_outputs(body)
            continue
        # A step may get its output from a local action call rather than a shell body (as pip-cache-restore does); the action's own shell must be readable.
        uses = mapping.get("uses")
        if not isinstance(uses, str) or not uses.strip().startswith("./"):
            continue
        # Repo root found by walking up to .github, not a fixed count: actions sit one level deeper than workflows, so a fixed three-up landed inside .github.
        root = path.parent
        for parent in path.parents:
            if parent.name == ".github":
                root = parent.parent
                break
        candidates = []
        for target in _local_ref_candidates(uses, root):
            candidates += [target / "action.yml", target / "action.yaml", target]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            text = candidate.read_text(ENC)
            # Keys declared in YAML also count as recovered, the usual case (frontend-dist-restore's cache-primary-key);
            # requiring shell-built keys wrongly failed valid configs.
            out[ident] = _local_action_outputs(candidate, text)
            break
    return out



def _local_action_outputs(action: Path, text: str) -> set:
    """Outputs of a local action judged per output: a delegated outputs.<name>.value needs that step's output recovered;
    otherwise the action's shell is the evidence."""
    doc = _parse(action)
    outputs = doc.get("outputs") if isinstance(doc, dict) else None
    inner = _producer_steps(action)
    inner_by_id: dict = {}
    for ident, names in inner.items():
        inner_by_id.setdefault(ident[2], set()).update(names)
    if isinstance(outputs, dict):
        recovered: set = set()
        for name, spec in outputs.items():
            value = spec.get("value") if isinstance(spec, dict) else None
            match = _STEP_OUTPUT.search(str(value or ""))
            if match is None:
                # Not delegated to a step: the value is literal or an expression not followed further, so the action's shell is the evidence.
                if _recovered_outputs(text) or _extract_cache_keys(action):
                    recovered.add(str(name))
                continue
            if match.group(2) in inner_by_id.get(match.group(1), set()):
                recovered.add(str(name))
        return recovered
    return set(_CACHE_STEP_OUTPUTS) if (
        _extract_cache_keys(action) or _recovered_outputs(text)
    ) else set()


def _delegation_is_read(expression: str, producers: dict, scope = None) -> bool:
    """True only if the step the expression names had that output recovered; an expression naming no step, like needs.build.outputs.key, is not evidence."""
    match = _STEP_OUTPUT.search(expression or "")
    if match is None:
        return False
    step_id, output = match.group(1), match.group(2)
    if scope is not None:
        return output in producers.get((scope[0], scope[1], step_id), set())
    # Without a scope, every step with that id must have recovered the output; one unreadable namesake is enough to say
    # no, so readable ones cannot vouch for it.
    seen = [names for ident, names in producers.items() if ident[2] == step_id]
    return bool(seen) and all(output in names for names in seen)


def _resolved_inputs(caller_paths: list, target: str, _depth: int = 0) -> dict:
    """{input: (literal values, every call site resolved)} over all callers of `target`.

    Callers are read from every PR-reachable document, workflows AND composites, not
    just the top-level workflow files. A cache action reached through a wrapper action
    gets its inputs from that wrapper, and collecting only from workflows meant such a
    call site was invisible: if the workflow ALSO called the action directly with a
    literal, every input looked resolved and the narrowing below dropped the namespace
    the wrapper passes.

    The second element of each pair is what keeps the narrowing honest. A call site
    passing `name: ${{ matrix.cache_name }}`, or omitting the input so the action default
    applies, has no literal value here. Substituting only the literals its siblings pass
    would DISCARD that caller's namespace, turning a fix for a false rejection into a
    false acceptance, which is the worse of the two.

    A value that is itself a `steps.*` or `needs.*` reference counts as neither literal
    nor unresolved: it is genuine delegation. Every live caller of pip-cache-save and
    uv-cache-save passes `key: ${{ steps.pip-cache.outputs.key }}`, whose real namespace
    was already collected from the restoring action's shell, so reporting it as
    undecidable was a false failure on this tree -- and a check that fails on a correct
    configuration is one that gets switched off.
    """
    values: dict = {}
    sites: list = []
    for pth in caller_paths:
        sites.extend(_call_sites(pth, target))
    if not sites:
        return {}
    # Every input named anywhere, collected BEFORE counting omissions. Doing both in one
    # pass made the answer depend on call-site order: a site that omitted an input had no
    # bucket yet, so the omission went unrecorded, and a later site supplying a literal
    # made the input look fully resolved. Verified before fixing -- a composite called
    # first with no `name` and then with `name: safe` reported ({'safe'}, True), so the
    # namespace narrowed to `safe` and the real default namespace was left undefended.
    names = {str(name) for site, _scope in sites for name in site}
    # Maps input name to (wrapper's own input, wrapper scope) pairs: values a wrapper forwards from its own inputs, resolved against its call sites.
    forwarded_inputs: dict = {}
    for name in names:
        literals: set = set()
        delegated: set = set()
        omitted = 0
        dynamic = 0
        for site, site_scope in sites:
            if name not in site:
                # Omitted input: the action's declared default applies, and it settles only this omission, not an explicit value the check cannot expand.
                omitted += 1
                continue
            literal = str(site[name]).strip().strip("'\"")
            if re.fullmatch(r"[A-Za-z0-9][\w.-]*", literal):
                literals.add(literal)
            elif _DELEGATED_KEY.fullmatch(literal):
                # Delegated values are kept with their scope, since which producer supplies the value decides whether the delegation settles anything.
                delegated.add((literal, site_scope))
            else:
                forwarded = _INPUT_KEY.fullmatch(literal)
                if forwarded is not None and site_scope is not None:
                    # A WRAPPER forwarding its own input: `with: {name: ${{ inputs.name }}}`
                    # inside another composite. The value is whatever that wrapper's own
                    # callers pass, so it is resolvable one level up rather than unknown,
                    # and counting it as dynamic left a nested composite undecidable and
                    # rejected an unrelated publish key -- a false failure on a valid
                    # arrangement.
                    #
                    # Recorded for the caller to resolve rather than followed here, which
                    # would mean recursing while already inside the resolution it needs.
                    forwarded_inputs.setdefault(name, set()).add(
                        (forwarded.group(1), site_scope)
                    )
                    continue
                # An explicit value this check cannot expand, like ${{ matrix.cache_key }}, counts as dynamic; no
                # default can settle it, since the caller overrode it.
                dynamic += 1
        for wrapper_input, wrapper_scope in forwarded_inputs.get(name, ()):
            # Forwarded values resolve against the wrapper's own callers, not the wrapper file; only one level is
            # followed, and past it _depth falls back to unresolved.
            outer = (
                {}
                if _depth
                else _resolved_inputs(
                    caller_paths, _target_name(Path(wrapper_scope[0])), _depth + 1
                )
            ).get(wrapper_input)
            if outer and outer[1] and outer[0]:
                literals |= outer[0]
            else:
                dynamic += 1
        values[name] = (
            literals, dynamic == 0 and omitted == 0, omitted, dynamic, delegated,
        )
    return values


def _literal_prefix(key: str) -> str:
    """Fixed-text head of a key, everything before the first ${{ expression; whole-string comparison would almost never match such keys."""
    return re.split(r"\$\{\{", key, maxsplit = 1)[0].strip().strip("'\"")


# runner.os is the only expression that routinely leads a cache key; it takes exactly Linux, Windows or macOS
# (case-sensitive), so expanding it turns the common undecidable case into three decidable ones.
_RUNNER_OS_VALUES = ("Linux", "Windows", "macOS")
_RUNNER_OS_EXPR = re.compile(r"\$\{\{\s*runner\.os\s*\}\}")

# A key that is nothing but one expression naming a step output delegates its namespace rather than declaring one.
_DELEGATED_KEY = re.compile(r"\$\{\{\s*(steps|needs)\.[^}]*\}\}")

# inputs.X is not delegation: the value comes from the caller, so it is resolved against call sites rather than dismissed.
_INPUT_KEY = re.compile(r"\$\{\{\s*inputs\.([A-Za-z_][\w-]*)\s*\}\}")


def _prefix_candidates(key: str) -> list[str]:
    """Every literal head this key could have at runtime.

    A key beginning with an expression has no literal head at all, and dropping it was a
    hole: a PR writing `${{ runner.os }}-shared-abc` and a publish job restoring
    `Linux-shared-` would never be compared, because the PR side reduced to the empty
    string and was filtered out. Expanding `runner.os` first gives `Linux-shared-abc`,
    which is comparable. Anything still expression-led afterwards is genuinely
    undecidable and is reported rather than dropped.
    """
    keys = [_RUNNER_OS_EXPR.sub(v, key) for v in _RUNNER_OS_VALUES] if _RUNNER_OS_EXPR.search(key) else [key]
    return [h for h in (_literal_prefix(k) for k in keys) if h]


def _is_truncated(key: str) -> bool:
    """Whether a key continues into an expression past its literal head; only then can a longer publish prefix still reach the runtime key."""
    keys = [_RUNNER_OS_EXPR.sub(v, key) for v in _RUNNER_OS_VALUES] if _RUNNER_OS_EXPR.search(key) else [key]
    return any("${{" in k for k in keys)

def _prefix_compatible(pr_head: str, publish_prefix: str, truncated: bool = True) -> bool:
    """Can a key with this literal head be restored by this prefix?

    `restore-keys` matching is left-anchored and exact, with no globbing, so a publish
    prefix restores a PR-written entry when the PR's runtime key starts with it.

    A head is TRUNCATED when the key continued into an expression this check cannot
    expand, and that is the only case where the reverse direction holds. A PR key
    `pip-v2-${{ runner.os }}-abc` reduces to the head `pip-v2-`, and the publish prefix
    `pip-v2-Linux-` is longer than that head, yet the runtime key `pip-v2-Linux-abc` does
    start with it, so the pairing has to be treated as compatible.

    For a head that is the WHOLE key, the reverse direction is simply wrong. A PR key
    that is exactly `shared` is saved as `shared`, and `shared`.startswith(`shared-long`)
    is false, so a publish fallback `shared-long` cannot reach it. Allowing the reverse
    unconditionally rejected every longer fallback that merely shared an opening with a
    complete key, which is a false failure on a correct configuration.
    """
    if pr_head.startswith(publish_prefix):
        return True
    return truncated and publish_prefix.startswith(pr_head)


def _shell_output_keys(text: str) -> list:
    """Fully literal `key=<value>` values written to `$GITHUB_OUTPUT`.

    The companion to `_shell_built_key_prefixes`, which recovers the literal HEAD of a
    key whose tail is assembled at runtime and therefore finds nothing in a key that is
    literal all the way through. Both are evidence that the producer was read; a step
    emitting `key=own-v1-abc` is completely resolved, and treating "no dynamic head" as
    "producer not understood" failed a correct configuration.

    These are exact keys, so they join the comparison rather than only vouching for it.
    """
    out = []
    for line in text.splitlines():
        # Comments are stripped before matching, so a commented-out echo of key=safe-key above a real printf cannot mark an unreadable producer readable.
        stripped = line.split("#", 1)[0]
        if "GITHUB_OUTPUT" not in stripped:
            continue
        for value in re.findall(r"\bkey=([A-Za-z0-9][A-Za-z0-9._-]*)", stripped):
            if "$" not in value and value not in out:
                out.append(value)
    return out


def _shell_built_key_prefixes(
    text: str, inputs: set | None = None, all_literal: bool = True,
) -> list[str]:
    """Literal key heads assembled in a composite action's shell, not in its YAML.

    The pip and uv caches build their key in a `run:` step and expose it as an output, so
    the YAML `key:` is only `${{ steps.probe.outputs.key }}` and carries no namespace at
    all. Reading YAML alone therefore learned nothing about the very composites this
    check exists to cover: pip-cache-restore's real namespace is the `pip-v2-` in
    `prefix="pip-v2-${name}-..."`, several lines away from any `key:`.

    `inputs` are the values callers actually pass for the first shell variable in such a
    prefix, which keeps the recorded namespace as narrow as the real one. See
    `_local_action_inputs` for why a broader one is not the safe direction.

    Only `key`-ish and `prefix`-ish variables are read. Taking every shell assignment
    would invent namespaces that no cache uses, and each invented one is a potential
    false rejection of a publish prefix.
    """
    heads: list[str] = []
    # A shell value only becomes a cache key by leaving the step, and `$GITHUB_OUTPUT` is
    # how it leaves. Every assignment named `key` or `prefix` was being treated as a cache
    # namespace regardless, so an unrelated `key="shared-${RANDOM}"` in a workflow that
    # caches nothing made a publish `restore-keys: shared-` fail the hard gate.
    #
    # Requiring the output channel rather than a cache step in the same file is
    # deliberate: the composite that BUILDS the key and the workflow that CACHES with it
    # are routinely different documents, which is the whole arrangement this function
    # exists to read.
    #
    # Residual, stated rather than implied: a document that writes an unrelated `key=`
    # value AND uses $GITHUB_OUTPUT elsewhere still over-collects. That direction is a
    # false rejection rather than a bypass, and narrowing it further would mean tracing
    # shell dataflow, which is past what this check should attempt.
    if "GITHUB_OUTPUT" not in text:
        return []
    pattern = re.compile(
        r"""(?:^|[\s;(])(?:[A-Za-z_]*_)?(?:key|prefix|KEY|PREFIX)\s*=\s*["']?"""
        r"""([A-Za-z0-9][A-Za-z0-9._-]*?-)(?=\$|\{)""",
        re.M,
    )
    for m in pattern.finditer(text):
        heads.append(m.group(1))
    # `echo "key=pip-v2-${hash}" >> "$GITHUB_OUTPUT"` is the same thing written inline.
    for m in re.finditer(
        r"""echo\s+["']?(?:key|prefix)=([A-Za-z0-9][A-Za-z0-9._-]*?-)(?=\$|\{)""", text
    ):
        heads.append(m.group(1))
    # A prefix whose literal head does not come FIRST yields nothing above, because the
    # pattern requires the head to begin with an alphanumeric, and
    # `prefix="${name}-pip-..."` puts its literal part after the variable. Substituting
    # the values callers actually pass makes it readable. Without this the composite
    # contributed no head at all, while the `steps.*` key reading its output was
    # dismissed as delegation, so a publish `restore-keys: shared-pip-` had nothing to be
    # compared against.
    for value in sorted(inputs or ()):
        substituted = re.sub(
            r"\$\{\s*[\w-]+\s*\}|\$\{\{\s*inputs\.[\w-]+\s*\}\}", value, text
        )
        for found in pattern.findall(substituted) + re.findall(
            r"""echo\s+["']?(?:key|prefix)=([A-Za-z0-9][A-Za-z0-9._-]*?-)(?=\$|\{)""",
            substituted,
        ):
            if found not in heads:
                heads.append(found)
    if inputs and all_literal:
        # Narrowed heads, plus heads that already carry a caller value from substitution above and need no further narrowing.
        return [f"{h}{v}-" for h in heads for v in sorted(inputs)] + [
            h for h in heads if any(h.startswith(v) for v in inputs)
        ]
    # An unresolved call site keeps the broad head, since narrowing it would silently drop the namespace that caller writes.
    return heads + [f"{h}{v}-" for h in heads for v in sorted(inputs or ())]






def _target_name(path: Path) -> str:
    """Target name of a uses: ref is the directory for an action but the file for a reusable workflow; using the directory for both matched no call site."""
    directory = path.parent if path.name.startswith("action.") else path
    for parent in path.parents:
        if parent.name == ".github":
            try:
                return directory.relative_to(parent.parent).as_posix()
            except ValueError:
                break
    return directory.name


def _namespaces_by_target(callers: list, targets: list) -> dict:
    """Maps target, then input name, to (literal values, all call sites resolved), kept per target; pooling by input
    name leaked values across targets and falsely rejected configs."""
    out: dict = {}
    for target in targets:
        resolved = _resolved_inputs(callers, _target_name(target))
        # A declared default belongs in the namespace even if no call site mentions the input; it settles only the omission, not an explicit unexpandable value.
        for field, value in _declared_defaults(target).items():
            entry = resolved.get(field, (set(), True, 0, 0, set()))
            vals, _ok, _omitted, dynamic, delegated = entry
            resolved[field] = (vals | {value}, dynamic == 0, 0, dynamic, delegated)
        out[target] = {f: (p[0], p[1], p[4]) for f, p in resolved.items()}
    return out


def _expand_key(
    key: str, namespaces: dict, producers: dict | None = None, scope = None,
) -> tuple:
    """(every literal key this can take, whether that list is complete).

    Substitutes each `${{ inputs.X }}` OCCURRENCE, rather than only a key that is
    nothing but one expression. `key: prefix-${{ inputs.name }}` called with
    `name: shared` runs as `prefix-shared`, and matching only whole-key expressions left
    that exact collision uncompared -- the commonest way a composite names a key, missed
    because it had a prefix in front of it.

    `runner.os` is expanded too, so the exact comparison sees the three keys a job really
    writes instead of one expression that equals nothing.
    """
    out = {key.strip()}
    complete = True
    # Delegated inputs: every call site hands the value off and `_shell_built_key_prefixes` records its namespace, so a leftover expression is not undecided.
    delegated_inputs: set = set()
    for _ in range(6):
        nxt: set = set()
        changed = False
        for k in out:
            match = _INPUT_KEY.search(k)
            if match is None:
                nxt.add(k)
                continue
            entry = namespaces.get(match.group(1), (set(), False, set()))
            values, resolved = entry[0], entry[1]
            delegated = entry[2] if len(entry) > 2 else set()
            if not values:
                nxt.add(k)
                # Resolved with NO literals means every call site delegates. Only
                # resolved-false is genuine doubt. Treating the two alike reported this
                # repository's own pip-cache-save, whose every caller passes
                # `${{ steps.pip-cache.outputs.key }}`, against every publish key in the
                # tree -- a false failure on a correct configuration.
                readable = bool(delegated) and all(
                    _delegation_is_read(expr, producers or {}, expr_scope)
                    for expr, expr_scope in delegated
                )
                if resolved and readable:
                    delegated_inputs.add(match.group(1))
                else:
                    # Delegation settles a key only if its producer namespace was recovered; `_shell_built_key_prefixes`
                    # reads two spellings, so printf-written keys stay undecided.
                    complete = False
                continue
            changed = True
            if not resolved:
                # An unresolved call site keeps its raw expression next to the literals, or the caller this check cannot expand is silently dropped.
                complete = False
                nxt.add(k)
            for value in sorted(values):
                nxt.add(k[: match.start()] + value + k[match.end() :])
        out = nxt
        if not changed:
            break
    expanded: set = set()
    for k in out:
        if _RUNNER_OS_EXPR.search(k):
            expanded.update(_RUNNER_OS_EXPR.sub(v, k) for v in _RUNNER_OS_VALUES)
        else:
            expanded.add(k)
    # Residue explained by a delegated input does not make a key undecided; any other surviving expression does, so mixed keys are still reported.
    for k in expanded:
        residue = k
        for name in delegated_inputs:
            residue = re.sub(
                r"\$\{\{\s*inputs\." + re.escape(name) + r"\s*\}\}", "", residue
            )
        if "${{" in residue:
            complete = False
            break
    return sorted(expanded), complete


def _input_namespaces(callers: list, targets: list) -> dict:
    """Merged view of _namespaces_by_target for shell-head narrowing only; merging is wrong for key values, but there a
    borrowed value only widens the namespace, the safe direction."""
    merged: dict = {}
    for target in targets:
        resolved = _resolved_inputs(callers, _target_name(target))
        # A declared default joins the namespace and settles only the missing-input case; an explicit caller passing an
        # unexpandable expression still leaves it undecided; treating defaults as blanket resolution hid that caller.
        defaults = _declared_defaults(target)
        for field, value in defaults.items():
            entry = resolved.get(field, (set(), True, 0, 0, set()))
            vals, _ok, _omitted, dynamic = entry[0], entry[1], entry[2], entry[3]
            resolved[field] = (vals | {value}, dynamic == 0, 0, dynamic, entry[4])
        for field, pair in resolved.items():
            vals, ok = merged.get(field, (set(), True))
            merged[field] = (vals | pair[0], ok and pair[1])
    return merged


def _expand_input_key(key: str, namespaces: dict) -> list[str]:
    """The literal keys a `key: ${{ inputs.X }}` can take, given its call sites.

    Needed on BOTH sides of the exact comparison, not just the prefix one. A pull request
    writing `shared-key` directly, against a dispatch workflow that calls a reusable
    workflow whose key is `${{ inputs.cache_key }}` with `cache_key: shared-key`, is an
    exact collision with no `restore-keys` anywhere, and comparing a literal against an
    unexpanded expression never matches. Resolution reached the prefix comparison through
    `pr_heads` and stopped there, so this plainest of all the shapes stayed open.
    """
    match = _INPUT_KEY.fullmatch(key.strip())
    if match is None:
        return [key]
    values, resolved = namespaces.get(match.group(1), (set(), False))
    if not values:
        return [key]
    # Keep unresolved call sites' raw expression beside the literals; dropping it hid a `${{ matrix.cache_key }}`
    # caller, so neither comparison saw its namespace.
    return sorted(values) if resolved else sorted(values) + [key]


def _declared_defaults(path: Path) -> dict:
    """inputs.<name>.default from an action or workflow definition; Actions applies it when a caller omits the input, so
    reading only call sites silently missed that namespace."""
    doc = _parse(path)
    out: dict = {}
    if not isinstance(doc, dict):
        return out
    declared = doc.get("inputs")
    if not isinstance(declared, dict):
        on = doc.get(True) if True in doc else doc.get("on")
        call = on.get("workflow_call") if isinstance(on, dict) else None
        declared = call.get("inputs") if isinstance(call, dict) else None
    if isinstance(declared, dict):
        for name, spec in declared.items():
            if isinstance(spec, dict) and spec.get("default") is not None:
                out[str(name)] = str(spec["default"])
    return out


def _local_ref_candidates(ref: str, root: Path) -> list:
    """Source-tree paths a ./ ref could name: as written first, then from its embedded .github part, for jobs that check
    the repo out into a subdirectory; notebooks-ci.yml and version-compat-ci.yml use that form."""
    ref = ref.strip()
    if ref.startswith("./"):
        ref = ref[2:]
    ref = ref.rstrip("/")
    out = [root / ref]
    parts = ref.split("/")
    if ".github" in parts[1:]:
        out.append(root / "/".join(parts[parts.index(".github"):]))
    return out


def _pr_reachable_action_dirs(workflows_dir: Path, pr_paths: list) -> set:
    """Composite actions a PR-triggered workflow reaches; scanning all of .github/actions wrongly counted publish-only
    composites as PR-writable namespaces, a false failure."""
    root = workflows_dir.parent
    dirs: set = set()
    seen: set = set()
    queue = [pth for pth in pr_paths]
    while queue:
        pth = queue.pop()
        if pth in seen:
            continue
        seen.add(pth)
        for ref in _local_uses(pth):
            for cand in _local_ref_candidates(ref, root.parent):
                # A local reusable workflow ref names the .yml file itself, not a directory holding an action.yml, so it must be followed as a file.
                if cand.is_file() and cand.suffix in (".yml", ".yaml"):
                    dirs.add(cand)
                    queue.append(cand)
                    break
                found = False
                for action in (cand / "action.yml", cand / "action.yaml"):
                    if action.is_file():
                        dirs.add(action)
                        queue.append(action)
                        found = True
                if found:
                    break
    return dirs

def _trigger_set(yaml_doc) -> set[str]:
    on = yaml_doc.get(True)
    if on is None:
        on = yaml_doc.get("on")
    return _normalise_on(on)


def main() -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--workflows-dir",
        type = Path,
        default = DEFAULT_WORKFLOWS_DIR,
        help = "Override the workflows directory (used by tests).",
    )
    args = parser.parse_args()
    workflows_dir = args.workflows_dir

    findings: list[str] = []
    workflows = sorted(workflows_dir.glob("*.yml"))
    pr_triggered: list[tuple[Path, list[str]]] = []
    publish_triggered: list[tuple[Path, list[str]]] = []
    publish_restore_prefixes: list[tuple[Path, list[str]]] = []

    for path in workflows:
        doc = _load_workflow(path)
        triggers = _trigger_set(doc)

        for t in BANNED_TRIGGERS:
            if t in triggers:
                findings.append(
                    f"{path.name}: BANNED trigger '{t}' (GHSA-g7cv-rxg3-hmpx "
                    "pattern: fork PRs run in base-repo context). Switch to "
                    "'pull_request' and use a deploy-on-merge workflow for "
                    "any privileged step."
                )

        for t in RESTRICTED_TRIGGERS:
            if t in triggers:
                text = path.read_text()
                if "lint:workflow_triggers-allow-workflow_run" not in text:
                    findings.append(
                        f"{path.name}: RESTRICTED trigger '{t}' requires an "
                        "explicit `# lint:workflow_triggers-allow-workflow_run` "
                        "comment somewhere in the file, with a justification."
                    )

        if "pull_request" in triggers:
            pr_triggered.append((path, _extract_cache_keys(path)))
        is_dispatch_only = "workflow_dispatch" in triggers and not (
            "push" in triggers or "pull_request" in triggers
        )
        if path.name in PUBLISH_WORKFLOW_NAMES or is_dispatch_only:
            publish_triggered.append((path, _extract_cache_keys(path)))
            publish_restore_prefixes.append((path, _extract_restore_key_prefixes(path)))

    # A PR-triggered workflow usually delegates its key to a composite action, so the
    # literal key lives in .github/actions/*/action.yml while the workflow carries only
    # `${{ steps.x.outputs.key }}`. Those count as pull-request-reachable, because the
    # workflow using them runs on pull requests. Without this the prefix rule below would
    # compare against opaque expressions and match nothing.
    pr_workflow_paths = [pth for pth, _ in pr_triggered]
    pr_reachable = sorted(_pr_reachable_action_dirs(workflows_dir, pr_workflow_paths))
    # Call sites come from every reachable document, workflows and composites alike, so a cache action behind a wrapper is narrowed by the wrapper's inputs too.
    pr_callers = pr_workflow_paths + pr_reachable
    composite_keys: list[str] = []
    shell_built: set = set()
    # A workflow can build its key in one of its OWN `run:` steps and consume it as
    # `${{ steps.probe.outputs.key }}`, with no composite involved at all. Only actions
    # were read for shell-built heads, so that key hit the delegation branch with nothing
    # recovered and a publish `restore-keys: shared-` had nothing to compare against.
    shell_literals: set = set()
    for pth in pr_workflow_paths:
        text = pth.read_text(ENC)
        built = _shell_built_key_prefixes(text)
        composite_keys.extend(built)
        shell_built.update(built)
        # Literal keys the producer writes outright; they are exact, so they belong in the comparison and show the step was understood.
        shell_literals.update(_shell_output_keys(text))
    for action_path in pr_reachable:
        composite_keys.extend(_extract_cache_keys(action_path))
        resolved = _resolved_inputs(pr_callers, _target_name(action_path))
        names, all_literal = resolved.get("name", (set(), False))[:2]
        built = _shell_built_key_prefixes(action_path.read_text(ENC), names, all_literal)
        composite_keys.extend(built)
        shell_literals.update(_shell_output_keys(action_path.read_text(ENC)))
        # A shell-built value is the literal head of a key whose tail is built at runtime, so it is truncated by
        # construction; without it, publish prefixes longer than the head (`pip-v2-shared-` vs `pip-v2-`) went
        # unexamined.
        shell_built.update(built)

    # Publish-side composites hold actions/cache/restore keys and restore-keys, so they must be read too; a comparison
    # that collects nothing on one side reports success.
    for action_path in sorted(
        _pr_reachable_action_dirs(workflows_dir, [pth for pth, _ in publish_triggered])
    ):
        publish_triggered.append((action_path, _extract_cache_keys(action_path)))
        publish_restore_prefixes.append(
            (action_path, _extract_restore_key_prefixes(action_path))
        )

    # Composite keys belong in the exact comparison too, not only prefixes; an exact collision such as shared-key is the
    # cache-poisoning shape. Input-backed keys resolve to callers' namespaces first.
    input_namespaces = _input_namespaces(pr_callers, pr_reachable)
    publish_callers = [pth for pth, _ in publish_triggered]
    publish_reachable = sorted(
        _pr_reachable_action_dirs(workflows_dir, publish_callers)
    )
    publish_namespaces = _input_namespaces(publish_callers, publish_reachable)
    # The publish side builds keys in shell too, and its heads were never collected:
    # `_shell_built_key_prefixes` ran over PR-reachable documents only. A publish
    # workflow that emits `key=shared-key` and restores `${{ steps.probe.outputs.key }}`
    # therefore had that key dismissed as delegated with nothing recovered to compare,
    # while a pull request writing the literal `shared-key` passed.
    publish_shell: set = set()
    # Keys are not unique per (path, key): setdefault kept only the first job's scope, so an unreadable second producer
    # was judged by the first job's readable one.
    pub_scope_list: dict = {}
    for pth in publish_callers + publish_reachable:
        for jid, key in _scoped_cache_keys(pth):
            pub_scope_list.setdefault((pth.as_posix(), key), []).append(
                (pth.as_posix(), jid)
            )
        text = pth.read_text(ENC)
        publish_shell.update(_shell_built_key_prefixes(text))
        publish_shell.update(_shell_output_keys(text))
    publish_producers: dict = {}
    for pth in publish_callers + publish_reachable:
        publish_producers.update(_producer_steps(pth))
    # Each definition keeps its OWN inputs. See `_namespaces_by_target`.
    pr_by_target = _namespaces_by_target(pr_callers, pr_reachable)
    publish_by_target = _namespaces_by_target(publish_callers, publish_reachable)

    # Entries are (namespace, key, scope): each key expands only with its declaring definition's inputs; scope is the
    # document and job whose step ids a delegated key refers to.
    pr_sites = [
        ({}, k, (pth.as_posix(), jid))
        for pth, _keys in pr_triggered
        for jid, k in _scoped_cache_keys(pth)
    ]
    # Only shell-built heads go here; re-adding every composite's YAML keys gave each a duplicate empty-namespace entry
    # that counted as unresolved and rejected unrelated publish keys. YAML keys belong in the target loop with their
    # inputs.
    pr_sites += [({}, k, None) for k in sorted(shell_built | shell_literals)]
    for target, namespace in pr_by_target.items():
        pr_sites += [
            (namespace, k, (target.as_posix(), jid))
            for jid, k in _scoped_cache_keys(target)
        ]

    pr_keys: set = set()
    # PR keys this check could not settle, with their literal heads; only these can collide with a publish key unseen.
    pr_undecided: list = []
    # Track readability per producing step, not per side: one readable `echo key=...` must not vouch for an unrecovered `printf` producer.
    pr_producers: dict = {}
    for pth in pr_workflow_paths + list(pr_reachable):
        pr_producers.update(_producer_steps(pth))
    for namespace, key, scope in pr_sites:
        literals, complete = _expand_key(key, namespace, pr_producers, scope)
        pr_keys.update(k for k in literals if "${{" not in k)
        # Delegation is not indecision. A key that is nothing but
        # `${{ steps.probe.outputs.key }}` names a value assembled in a shell step, and
        # `_shell_built_key_prefixes` recovers its head and records that namespace
        # separately; counting it here as well reported every publish key in the tree
        # against it, which failed the live tree on a correct configuration.
        if not complete and not (
            _DELEGATED_KEY.fullmatch(key.strip())
            and _delegation_is_read(key.strip(), pr_producers, scope)
        ):
            pr_undecided.append((key, _prefix_candidates(key)))
    pr_raw_unresolved = {raw.strip() for raw, _heads in pr_undecided}

    for pub_path, pub_keys in publish_triggered:
        # This target's own inputs, not the merged view. Two publish-reachable actions
        # sharing an input name had every key expanded with both their values, so a cache
        # composite given `publish-key` was also expanded to an unrelated action's
        # `safe-key` and reported as colliding with a PR cache of that name.
        # A top-level workflow is not a target, and its `${{ inputs.X }}` is a
        # workflow_dispatch input chosen by whoever dispatches it. Falling back to the
        # merged CHILD namespace expanded that user-controlled value using an unrelated
        # action's literal and then called it settled, so dispatching with
        # `cache_key=shared-key` restored a pull request's cache with the lint green.
        # An empty namespace leaves it undecidable, which is what it is.
        pub_ns = publish_by_target.get(pub_path, {})
        for raw in pub_keys:
            # Check every job declaring this key, not just the first; the key is settled only if every declaring scope's producer could be read.
            scopes = pub_scope_list.get((pub_path.as_posix(), raw)) or [None]
            pub_scope = next(
                (
                    sc
                    for sc in scopes
                    if sc is None
                    or not _DELEGATED_KEY.fullmatch(raw.strip())
                    or not _delegation_is_read(raw.strip(), publish_producers, sc)
                ),
                scopes[0],
            )
            literals, complete = _expand_key(
                raw, pub_ns, publish_producers, pub_scope
            )
            if _DELEGATED_KEY.fullmatch(raw.strip()):
                # Compared as the heads its own shell produced, rather than skipped.
                read = _delegation_is_read(
                    raw.strip(), publish_producers, pub_scope
                )
                literals = sorted(publish_shell) if read else [raw.strip()]
                complete = read
            # Identical spellings compare first: an unexpandable expression written the same on both sides resolves
            # alike, so skipping `${{` keys hid this collision.
            if not complete and raw.strip() in pr_raw_unresolved:
                findings.append(
                    f"{pub_path.name}: cache key {raw.strip()!r} is spelled identically "
                    "in a PR-triggered workflow. Neither value can be resolved here, "
                    "but two identical keys resolve identically at run time, so a fork "
                    "PR could write the entry this workflow restores. Add a unique "
                    "suffix (e.g. '-publish-only')."
                )
                continue
            for k in literals:
                if k in pr_keys:
                    findings.append(
                        f"{pub_path.name}: cache key {k!r} is also declared in a "
                        "PR-triggered workflow. A fork PR could poison this cache "
                        "and the publish workflow would restore it on next run. "
                        "Add a unique suffix (e.g. '-publish-only') to partition "
                        "the namespaces."
                    )
                    continue
                if "${{" in k:
                    if _DELEGATED_KEY.fullmatch(k.strip()) and _delegation_is_read(
                        k.strip(), publish_producers, pub_scope
                    ):
                        # Delegated AND read. See the PR side above.
                        continue
                    # Unresolved publish keys fail closed like PR keys; a key with no literal head is expression-led and
                    # could be any value, so every PR key is a candidate; requiring a head would exempt the least
                    # decidable keys.
                    heads = [h for h in _prefix_candidates(k) if h]
                    reported = False
                    for literal in sorted(pr_keys):
                        if not heads or any(literal.startswith(h) for h in heads):
                            findings.append(
                                f"{pub_path.name}: cache key {k!r} cannot be shown not "
                                f"to collide with the PR-written key {literal!r}, "
                                "because this check cannot resolve the publish key's "
                                "value. Give it a unique suffix (e.g. "
                                "'-publish-only'), or make it literal."
                            )
                            reported = True
                            break
                    if reported:
                        continue
                    # Two unresolved keys spelled differently are compatible if either fixed head leads to the other
                    # (prefix test both ways); resolved-only comparison missed them.
                    for raw_pr, pr_heads in pr_undecided:
                        fixed = [h for h in pr_heads if h]
                        if (
                            not heads
                            or not fixed
                            or any(
                                h.startswith(f) or f.startswith(h)
                                for h in heads
                                for f in fixed
                            )
                        ):
                            findings.append(
                                f"{pub_path.name}: cache key {k!r} and the PR-reachable "
                                f"key {raw_pr!r} can both resolve to the same value, "
                                "and neither can be resolved here. Give the publish key "
                                "a unique suffix (e.g. '-publish-only'), or make one of "
                                "them literal."
                            )
                            break
                    continue
                # Fail closed: an unsettled PR key may equal this publish key, so it is checked outside the
                # restore-prefix pass too; narrow to PR keys whose fixed head this key begins with.
                for raw_pr, heads in pr_undecided:
                    # Same rule the other way: a PR key with no literal head is unnarrowable rather than harmless, so it counts as a candidate.
                    fixed = [h for h in heads if h]
                    if not fixed or any(k.startswith(h) for h in fixed):
                        findings.append(
                            f"{pub_path.name}: cache key {k!r} cannot be shown not to "
                            f"collide with the PR-reachable key {raw_pr!r}, whose value "
                            "this check cannot resolve. Give the publish key a unique "
                            "suffix (e.g. '-publish-only'), or make the PR key literal."
                        )
                        break

    # The same trust boundary, reached by prefix instead of by an equal key.
    pr_heads: set = set()
    # Only heads whose key continued into an unexpandable expression may be reached by a publish prefix longer than the head itself.
    truncated_heads: set = set()
    undecidable_pr_keys: list[str] = []
    for k in list(pr_keys) + composite_keys:
        cands = _prefix_candidates(k)
        if cands:
            pr_heads.update(cands)
            if _is_truncated(k) or k in shell_built:
                truncated_heads.update(cands)
        elif _INPUT_KEY.fullmatch(k.strip()):
            # `key: ${{ inputs.cache_key }}` in a reusable workflow or composite names no
            # namespace here, but unlike a `steps.*` reference it is not delegation
            # either: the value comes from the CALLER, and a pull request caller passing
            # `cache_key: shared-abc` writes the `shared-` namespace. Treating it as
            # delegation dropped it silently and a publish `shared-` fallback passed.
            field = _INPUT_KEY.fullmatch(k.strip()).group(1)
            vals, resolved = input_namespaces.get(field, (set(), False))
            heads = [h for v in sorted(vals) for h in _prefix_candidates(v)]
            pr_heads.update(heads)
            truncated_heads.update(h for v in sorted(vals) if _is_truncated(v)
                                   for h in _prefix_candidates(v))
            if not resolved:
                # Some call site passes a value this check cannot expand, such as
                # `${{ matrix.cache_name }}`, or omits the input so the action default
                # applies. Say so rather than assuming it is harmless. Resolved with no
                # literals is the DELEGATION case and is fine: every live caller of
                # pip-cache-save passes `key: ${{ steps.pip-cache.outputs.key }}`, whose
                # real namespace was collected from the restoring action's shell.
                undecidable_pr_keys.append(k)
        elif _DELEGATED_KEY.fullmatch(k.strip()):
            # `key: ${{ steps.pip-cache.outputs.key }}` names no namespace of its own; it
            # hands the decision to a composite action, whose real prefix was collected
            # above from that action's own YAML and shell. Reporting it as undecidable
            # would flag every workflow that factors its cache out into an action.
            continue
        else:
            undecidable_pr_keys.append(k)

    for pub_path, prefixes in publish_restore_prefixes:
        for prefix in prefixes:
            # Expanded with the declaring target's own inputs first, exactly as its
            # exact keys are. Reducing `prefix-${{ inputs.name }}-` without them gave
            # the broad head `prefix-`, which collides with everything under it, so a
            # composite called with `name: publish` was reported against unrelated PR
            # keys whose runtime fallback it could never reach.
            prefix_ns = publish_by_target.get(pub_path, {})
            expanded_prefixes, _ok = _expand_key(prefix, prefix_ns)
            pub_heads = [
                head
                for candidate in expanded_prefixes
                for head in _prefix_candidates(candidate)
            ]
            if not pub_heads:
                findings.append(
                    f"{pub_path.name}: restore-keys entry {prefix!r} begins with an "
                    "expression, so what it can restore is not decidable here. Give it a "
                    "literal prefix."
                )
                continue
            for pub_head in pub_heads:
                hit = next(
                    (
                        h
                        for h in sorted(pr_heads)
                        if _prefix_compatible(h, pub_head, h in truncated_heads)
                    ),
                    None,
                )
                if hit is not None:
                    findings.append(
                        f"{pub_path.name}: restore-keys prefix {pub_head!r} matches "
                        f"{hit!r}, a cache key namespace a PR-triggered workflow writes. "
                        "A prefix restore takes the newest matching entry, so this "
                        "publish workflow could adopt a cache a pull request produced "
                        "even though no key is equal. Partition the namespaces, or drop "
                        "the restore-keys fallback on the publish side."
                    )
                    break
            else:
                # Reached only when no PR key matched: an undecidable PR key could still expand into this prefix, so report it rather than pass silently.
                for k in undecidable_pr_keys:
                    findings.append(
                        f"{pub_path.name}: restore-keys prefix {prefix!r} cannot be "
                        f"compared against PR cache key {k!r}, which begins with an "
                        "expression this check cannot expand, so whether the prefix "
                        "reaches that namespace is undecidable. Give the PR key a "
                        "literal prefix."
                    )
                    break

    if findings:
        print(
            "Workflow trigger lint failed with the following issues:", file = sys.stderr
        )
        for f in findings:
            print(f"  - {f}", file = sys.stderr)
        return 1

    print(
        f"OK: scanned {len(workflows)} workflow file(s); "
        f"no pull_request_target, no unjustified workflow_run, "
        f"no PR/publish cache-key collision."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

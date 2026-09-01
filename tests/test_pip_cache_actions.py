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

"""Pins the pip-cache restore/save contract for every workflow.

`actions/setup-python`'s built-in `cache: 'pip'` saves from its post-step on
whatever ref the job ran on, and a cache written on a pull_request ref can only
be restored by re-runs of that same PR. So every PR writes a copy nobody else
can read, competing for the 20 GiB budget against main's copy, which every PR
can read. Measured 2026-08-26: 19.97 of 20 GiB, merged PR #1084 holding 6.6 GiB
of it. Nothing failed; CI just got slower.

Re-adding `cache: 'pip'` looks right in review -- it is the documented way to
do this -- which is why it is asserted here instead.
"""

import re
import fnmatch
import pathlib

import pytest
import yaml


REPO = pathlib.Path(__file__).resolve().parent.parent
# Both extensions: GitHub runs `.yaml` workflows too, and a glob for `.yml` alone would
# let an unguarded cache writer added in a `.yaml` file bypass every check in this file.
WORKFLOWS = sorted(
    p for ext in ("*.yml", "*.yaml") for p in (REPO / ".github" / "workflows").glob(ext)
)
RESTORE = "./.github/actions/pip-cache-restore"
SAVE = "./.github/actions/pip-cache-save"


def _jobs(path):
    doc = yaml.safe_load(path.read_text()) or {}
    for name, job in (doc.get("jobs") or {}).items():
        yield name, (job.get("steps") or [])


def _indices(steps, action):
    return [i for i, s in enumerate(steps) if s.get("uses") == action]


def test_workflows_exist():
    # A glob that matches nothing would make every test below vacuously pass.
    assert WORKFLOWS, "no workflows found; the layout moved and this file is stale"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_no_builtin_setup_python_pip_cache(path):
    for job, steps in _jobs(path):
        for step in steps:
            uses = str(step.get("uses", ""))
            if not uses.startswith("actions/setup-python@"):
                continue
            with_ = step.get("with") or {}
            assert "cache" not in with_, (
                f"{path.name}:{job} uses setup-python's built-in cache. It saves on "
                f"the PR ref, where nothing else can read it. Use {RESTORE} plus "
                f"{SAVE} instead."
            )


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_restore_and_save_are_paired(path):
    for job, steps in _jobs(path):
        restores, saves = _indices(steps, RESTORE), _indices(steps, SAVE)
        if not restores and not saves:
            continue
        assert len(restores) == 1, f"{path.name}:{job} has {len(restores)} restores; expected 1"
        assert len(saves) == 1, (
            f"{path.name}:{job} has {len(saves)} saves; expected 1. A restore without a "
            f"save never writes the cache it just missed, so the job downloads the same "
            f"wheels on every run of main."
        )
        assert saves[0] > restores[0], f"{path.name}:{job} saves before it restores"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_save_is_not_gated_behind_a_failing_step(path):
    # A composite step with no `if:` is SKIPPED once an earlier step failed, so
    # the save's own always() covers only what sits between the pair. Anything
    # that can exit non-zero in there discards a completed install.
    for job, steps in _jobs(path):
        restores, saves = _indices(steps, RESTORE), _indices(steps, SAVE)
        if not restores or not saves:
            continue
        between = steps[restores[0] + 1 : saves[0]]
        assert len(between) <= 1, (
            f"{path.name}:{job} has {len(between)} steps between restore and save. "
            f"Put the save directly after the step that installs, or a failure in "
            f"between silently skips it."
        )


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_restore_inputs_are_well_formed(path):
    for job, steps in _jobs(path):
        for i in _indices(steps, RESTORE):
            with_ = steps[i].get("with") or {}
            name = with_.get("name", "")
            assert re.fullmatch(r"[a-z0-9-]+", name or ""), (
                f"{path.name}:{job} restore name={name!r} must be lowercase letters, "
                f"digits and dashes; it goes into the cache key verbatim."
            )
            files = [f for f in (with_.get("key-files") or "").split() if f]
            assert files, f"{path.name}:{job} passes no key-files; the key would not distinguish anything"
            for f in files:
                assert (REPO / f).exists() or "*" in f, (
                    f"{path.name}:{job} key-files entry {f!r} does not exist. hashFiles "
                    f"returns '' for a glob that matches nothing, which collapses every "
                    f"generation onto one key."
                )


def test_cache_names_are_unique_across_all_jobs():
    # A shared name is a shared prefix, which is the defect `name` exists to
    # avoid: the janitor cannot tell two live caches from two generations of one,
    # and pruning then deletes a live entry.
    seen = {}
    for path in WORKFLOWS:
        for job, steps in _jobs(path):
            for i in _indices(steps, RESTORE):
                name = ((steps[i].get("with") or {}).get("name") or "")
                where = f"{path.name}:{job}"
                assert name not in seen, (
                    f"cache name {name!r} used by both {seen[name]} and {where}; "
                    f"names must be unique so each job is its own prunable family"
                )
                seen[name] = where
    assert seen, "no pip-cache-restore call sites found; this test is stale"


def test_the_janitor_ranks_the_pip_family():
    # These keys are `pip-<name>-...`, which matched none of the janitor's
    # original arms and fell through to `*) continue`. Left that way, every
    # pyproject.toml edit strands the previous multi-GB entry on main unrankable
    # -- and once every job uses the restore/save pair nothing writes a
    # setup-python-* key again, so no pip cache would be pruned at all.
    #
    # Asserted by MATCHING a representative key against the arms, not by grepping
    # for `pip-*)`: that substring already sits inside `setup-python-*-pip-*)`,
    # so a textual check passes on the broken version and proves nothing.
    janitor = (REPO / ".github/workflows/cache-janitor.yml").read_text()
    block = re.search(r'case "\$key" in(.+?)\n\s*esac', janitor, re.S)
    assert block, "the janitor's key dispatch moved; this assertion is stale"

    patterns = []
    for line in block.group(1).splitlines():
        line = line.strip()
        if not line.endswith(")") or line.startswith("#") or line.startswith("pre="):
            continue
        patterns.extend(line[:-1].split("|"))

    for key in (
        "pip-repo-tests-Linux-X64-py3.12-" + "a" * 64,
        "pip-lint-Linux-X64-py3.12-" + "b" * 64,
    ):
        assert any(fnmatch.fnmatchcase(key, p) for p in patterns if p != "*"), (
            f"cache-janitor.yml does not rank {key!r}. Its superseded generations "
            f"would accumulate untouched, which is the pressure this PR removes."
        )


def test_save_runs_on_the_default_branch_only():
    action = yaml.safe_load((REPO / ".github/actions/pip-cache-save/action.yml").read_text())
    steps = action["runs"]["steps"]
    condition = " ".join(str(steps[0].get("if", "")).split())
    assert "github.ref == 'refs/heads/main'" in condition, (
        "the save must stay gated to main. Without it every PR writes a private "
        "copy that only that PR can read, which is the whole defect."
    )
    assert "always()" in condition, "a later step failing must not discard a completed install"


# --- every cache save, not just the pip pair ------------------------------------------
#
# The rule above is enforced on the pip-cache-save action, which is the only thing that
# was writing PR-scoped entries when it was written. It is not the only thing that CAN.
# `gemma4-audio-probe.yml` used a bare read-write `actions/cache`, which registers its own
# post-step and saves on whatever ref the job ran on, exactly like the setup-python form
# this file exists to keep out. That workflow triggers on a label, so a labelled PR wrote
# a 3.4 GB checkpoint to refs/pull/N/merge that only that PR's re-runs could restore,
# seven macOS legs racing the one key.
#
# The janitor could not have cleaned it up either: `hf-gemma4-e2b-4bit-v1` carries no
# trailing dependency hash, so cache-janitor.yml's generation ranking skips it entirely
# and only closing the PR would ever have freed the bytes.
#
# So the check is on the shape, everywhere, rather than on the one action that had the
# problem first.

_CACHE_WRITERS = ("actions/cache@", "actions/cache/save@")


def _split_top(expr, op):
    """``expr`` split on its TOP-LEVEL ``op``, ignoring occurrences in parens or quotes."""
    parts, depth, quote, buf, i = [], 0, "", [], 0
    while i < len(expr):
        ch = expr[i]
        if quote:
            if ch == quote:
                quote = ""
            buf.append(ch)
        elif ch in "'\"":
            quote = ch
            buf.append(ch)
        elif ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif depth == 0 and expr[i:i + 2] == op:
            parts.append("".join(buf))
            buf = []
            i += 2
            continue
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def _balanced(expr):
    """Whether parentheses in ``expr`` are balanced outside quotes."""
    depth, quote = 0, ""
    for ch in expr:
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


# A POSITIVE equality, in either quote style. `!=` must not match: a condition restricting
# a save to everything EXCEPT main is the exact inversion of the rule, and a substring
# search for "refs/heads/main" accepts it.
_MAIN_ONLY = re.compile(r"github\.ref\s*==\s*['\"]refs/heads/main['\"]")


def _restricted_to_main(expr):
    """Whether ``expr`` can only be true on the default branch.

    Evaluated over the whole boolean structure, not just the top level. `||` is how a
    condition GAINS refs, so an OR restricts only if EVERY branch restricts; `&&` narrows,
    so an AND restricts if ANY branch does. Parentheses matter: splitting only on
    top-level `||` accepted `always() && (github.ref == 'refs/heads/main' ||
    github.event_name == 'pull_request')`, which runs on every PR. Anything negated is
    refused outright rather than reasoned about, because `!(github.ref ==
    'refs/heads/main')` contains a positive main equality and is its exact inverse.
    """
    if not expr.strip():
        return False

    def restricted(part):
        part = part.strip()
        while part.startswith("(") and part.endswith(")") and _balanced(part[1:-1]):
            part = part[1:-1].strip()
        ors = _split_top(part, "||")
        if len(ors) > 1:
            return all(restricted(p) for p in ors)
        ands = _split_top(part, "&&")
        if len(ands) > 1:
            return any(restricted(p) for p in ands)
        # A leaf. `!` anywhere outside a `!=` is a negation we will not reason about.
        if re.search(r"!(?!=)", part):
            return False
        # Exactly one comparison, and it must be the main equality. A second one
        # is how a leaf inverts itself while still containing the equality:
        # `(github.ref == 'refs/heads/main') == false` is true off main, and
        # `... != true` is the same trick. Quoted literals are blanked first so a
        # `==` inside a string is not counted.
        bare = re.sub(r"'[^']*'|\"[^\"]*\"", "''", part)
        if len(re.findall(r"[=!]=", bare)) != 1:
            return False
        return bool(_MAIN_ONLY.search(part))

    return restricted(expr)


@pytest.mark.parametrize(
    "expr,restricted",
    [
        ("always() && github.ref == 'refs/heads/main'", True),
        ('github.ref == "refs/heads/main" && steps.x.outcome == \'success\'', True),
        ("github.ref != 'refs/heads/main'", False),
        ("github.ref == 'refs/heads/main' || github.event_name == 'pull_request'", False),
        ("", False),
        # A `||` inside parens is still a `||`. Splitting only the top level read this
        # as one alternative containing the main equality and accepted it, while it
        # actually runs on every pull request.
        (
            "always() && (github.ref == 'refs/heads/main' "
            "|| github.event_name == 'pull_request')",
            False,
        ),
        # Contains a positive main equality and means its exact opposite.
        ("!(github.ref == 'refs/heads/main')", False),
        # Parenthesised but genuinely restricted, so the fix must not over-reject.
        ("(github.ref == 'refs/heads/main') && always()", True),
        # Comparing the equality to a boolean inverts it while still containing it.
        ("(github.ref == 'refs/heads/main') == false", False),
        ("github.ref == 'refs/heads/main' != true", False),
        ("always() && (github.ref == 'refs/heads/main' && !cancelled())", True),
        # An OR restricts only when every branch does.
        (
            "(github.ref == 'refs/heads/main' && always()) "
            "|| (github.ref == 'refs/heads/main' && failure())",
            True,
        ),
        # A `||` inside a string is not a split point.
        ("github.ref == 'refs/heads/main' && contains(x, 'a||b')", True),
    ],
)
def test_the_main_only_check_reads_the_expression(expr, restricted):
    # The guard below is only as good as this predicate, so the predicate is tested too.
    assert _restricted_to_main(expr) is restricted, expr


def _cache_writer_steps():
    """(where, step) for every step that can write a cache entry, workflows and actions."""
    for path in WORKFLOWS:
        for name, steps in _jobs(path):
            for step in steps:
                yield f"{path.name}:{name}", step
    # Both spellings, for the same reason WORKFLOWS takes both: GitHub accepts
    # `action.yaml` for composite-action metadata, and a scan for `action.yml`
    # alone would let an unguarded writer live in one unseen.
    actions_dir = REPO / ".github" / "actions"
    for action in sorted(
        p for ext in ("action.yml", "action.yaml") for p in actions_dir.rglob(ext)
    ):
        doc = yaml.safe_load(action.read_text()) or {}
        for step in ((doc.get("runs") or {}).get("steps") or []):
            yield f"action {action.parent.name}", step


def test_no_cache_save_reaches_a_pull_request_ref():
    offenders = []
    for where, step in _cache_writer_steps():
        uses = str(step.get("uses", ""))
        # `actions/cache/restore@` is read-only and correct on every ref.
        if not any(w in uses for w in _CACHE_WRITERS):
            continue
        if not _restricted_to_main(" ".join(str(step.get("if", "")).split())):
            offenders.append(f"{where}: {step.get('name') or uses}")
    assert not offenders, (
        "these steps save a cache without restricting it to the default branch, so a "
        "pull request writes an entry only its own re-runs can ever restore while "
        "competing for the 20 GiB budget:\n  " + "\n  ".join(offenders)
    )


def test_a_downloaded_artifact_is_saved_after_it_is_downloaded():
    """A save placed where the restore sits stores an empty directory under the key.

    Splitting a read-write `actions/cache` is exactly where this goes wrong: the
    read-write form saves from a post-step at the END of the job, so moving to
    restore/save without also moving the save below the step that fills the directory
    poisons the key for every later job, which then hits it and finds nothing.
    """
    offenders = []
    for path in WORKFLOWS:
        for name, steps in _jobs(path):
            ids = {s.get("id"): i for i, s in enumerate(steps) if s.get("id")}
            for i, step in enumerate(steps):
                if "actions/cache/save@" not in str(step.get("uses", "")):
                    continue
                label = f"{path.name}:{name}: {step.get('name') or 'save'}"
                condition = " ".join(str(step.get("if", "")).split())
                path_saved = str((step.get("with") or {}).get("path", "")).strip()

                # The producer has to sit strictly BETWEEN the restore of the same
                # path and this save. Anything looser is satisfiable by the wrong
                # step: a `.outputs.` reference matched the restore's own
                # `cache-hit`, and merely "some earlier step's .outcome" matches the
                # restore too, so a save moved up next to the restore still passed
                # while the directory was empty. Only a step in that window can be
                # what filled it.
                restore_at = max(
                    (
                        j for j, s in enumerate(steps[:i])
                        if "actions/cache/restore@" in str(s.get("uses", ""))
                        and str((s.get("with") or {}).get("path", "")).strip() == path_saved
                    ),
                    default=None,
                )
                producers = set(re.findall(r"steps\.([A-Za-z0-9_-]+)\.outcome", condition))
                if not producers:
                    offenders.append(
                        f"{label} is not gated on the outcome of any step, so nothing "
                        f"establishes that the directory it saves was populated"
                    )
                    continue
                unknown = sorted(p for p in producers if p not in ids)
                if unknown:
                    offenders.append(f"{label} references unknown step id(s) {unknown}")
                    continue
                lower = -1 if restore_at is None else restore_at
                between = [p for p in producers if lower < ids[p] < i]
                if not between:
                    where = (
                        f"after the restore at step {restore_at}"
                        if restore_at is not None else "before it"
                    )
                    offenders.append(
                        f"{label} is gated on {sorted(producers)}, none of which runs "
                        f"{where} and before the save, so nothing between the restore "
                        f"and the save is known to have filled {path_saved!r}"
                    )
    assert not offenders, "\n  ".join(offenders)

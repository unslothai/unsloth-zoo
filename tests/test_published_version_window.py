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

"""The published version window, and every CI lane and gate that mirrors it.

`transformers<=5.5.0` and `torch<2.13.0` were never one line each. The transformers cap
was two lines here, two more in unsloth's pyproject, and a `CEILING` literal in
consolidated-tests-ci.yml that fails the build when pyproject moves past it. The torch cap
was one line plus three workflow `pip install` mirrors. A site left behind after the window
moves does not go red: the lane runs, passes, and measures a range users no longer get.

The transformers cap is now marker-split, and the two halves are different decisions:

* off darwin it is the newest release the version matrix was run against. 5.5.0 held it
  until the sweep behind unsloth #9867 / #10010 / #10017 / #10276 (prequantized bnb-4bit
  checkpoints losing `quant_state` on every `Linear4bit`) and #5355, whose Gemma 4 E4B LoRA
  fix shipped in transformers 5.5.2, one patch release above the old cap.
* on darwin + arm64 it stays at 5.5.0, because there it is what holds the joint MLX
  resolution at mlx-vlm 0.6.4. mlx-vlm 0.6.5 and up require transformers >= 5.14.0,
  tests/mlx_simulation models the 0.6.4 surface, and mlx / mlx-lm are pinned exactly. A
  CUDA-motivated bump must not make that decision for the Apple Silicon lane.

So the assertions are that the split exists, that each half is where it is supposed to be,
that the exclusions survived the rewrite, and that nothing mirroring either cap drifted.
Reads files only: no network, no torch, no transformers install.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# Newest transformers the version matrix was run against off darwin.
TESTED_CEILING = Version("5.17.0")
# What the MLX stack holds, and why: the first mlx-vlm that wants more.
MLX_CEILING = Version("5.5.0")
MLX_VLM_0_6_5_TRANSFORMERS_FLOOR = Version("5.14.0")

# Newest torch the matrix was run against, and the exclusive bound that admits it.
TESTED_TORCH = Version("2.14.0")
TORCH_BOUND = "<2.15.0"

# The transformers floor both halves must declare. peft declares no transformers floor of
# its own, and peft 0.18.0 imports `GradientCheckpointingLayer` from
# `transformers.modeling_layers` at peft/tuners/lora/model.py:26, which first exists in
# 4.52.0; at 4.51.3 `import unsloth_zoo.saving_utils` raised ModuleNotFoundError after a
# resolve that satisfied every declared constraint. 4.52.4 and not 4.52.0 because
# 4.52.0 through 4.52.3 were already rejected by name.
DECLARED_FLOOR = Version("4.52.4")

# Tested and rejected. A rewrite of the specifier must not drop one.
REJECTED = (
    "4.52.0", "4.52.1", "4.52.2", "4.52.3", "4.53.0", "4.54.0",
    "4.55.0", "4.55.1", "4.57.4", "4.57.5", "5.0.0", "5.1.0",
)

# Lanes that install torch by hand as a copy of the published ceiling. Each has to move
# when the ceiling does.
PUBLISHED_TORCH_MIRRORS = (
    "gemma4-audio-probe.yml",
    "mlx-ci.yml",
    "security-audit.yml",
)

# Lanes that hold an older torch window deliberately, with the reason. The MLX lanes need
# the torch / torchvision ABI pair that mlx-cpu was built against, which is a different
# question from what the package publishes.
OLDER_TORCH_WINDOW_BY_DESIGN = {
    "consolidated-tests-ci.yml": (
        "the interpreter, repo, MLX and core-drift lanes pin torch>=2.4.0,<2.11.0 for the "
        "CPU torch / torchvision ABI pair those suites were measured against"
    ),
}

DARWIN_ARM = {"sys_platform": "darwin", "platform_machine": "arm64"}
LINUX_X86 = {"sys_platform": "linux", "platform_machine": "x86_64"}


def _requirement_lists() -> dict[str, list[str]]:
    """{where: [raw requirement, ...]} for the base list and every extra.

    tomllib is 3.11+ and requires-python here is >=3.9, so the import is lazy and older
    interpreters skip rather than failing to collect, which would red the
    python-version-collect lane. Same shape as _find_config in
    tests/test_wheel_top_level_packages.py.
    """
    if sys.version_info < (3, 11):
        pytest.skip("tomllib needs Python 3.11+")
    import tomllib

    data = tomllib.loads(PYPROJECT.read_text(encoding = "utf-8"))
    project = data.get("project") or {}
    out = {"dependencies": list(project.get("dependencies") or [])}
    for name, reqs in (project.get("optional-dependencies") or {}).items():
        out[f"optional-dependencies.{name}"] = list(reqs)
    return out


def _named(raws: list[str], name: str) -> list[Requirement]:
    out = []
    for raw in raws:
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            continue
        if req.name.lower().replace("_", "-") == name:
            out.append(req)
    return out


def _live(reqs: list[Requirement], environment: dict[str, str]) -> list[Requirement]:
    """The requirements whose marker holds in this environment, extras aside."""
    out = []
    for req in reqs:
        if req.marker is None:
            out.append(req)
            continue
        try:
            if req.marker.evaluate(environment):
                out.append(req)
        except Exception:
            # An `extra == ...` clause has no value in a bare environment. The lists here
            # come straight from pyproject, so nothing carries one; anything that does is
            # not this test's business.
            continue
    return out


def _ceiling(spec: SpecifierSet) -> Version:
    """The version this specifier actually stops at.

    The TIGHTEST upper bound, not the loosest: bounds intersect, so `<=5.17.0,<5.16.0`
    admits nothing above 5.16.0 and `max` would report 5.17 as the ceiling of a window
    that excludes the tested release. Ties are broken by exclusivity, since `<5.16.0` is
    tighter than `<=5.16.0`.
    """
    tops = [(Version(str(s.version)), s.operator == "<") for s in spec if s.operator in ("<=", "<")]
    assert tops, f"no upper bound declared: {spec}"
    return min(tops, key = lambda pair: (pair[0], pair[1]))[0]


def _transformers_lists() -> dict[str, list[Requirement]]:
    return {
        where: reqs
        for where, raws in _requirement_lists().items()
        if (reqs := _named(raws, "transformers"))
    }


def test_every_list_that_names_transformers_carries_both_halves() -> None:
    lists = _transformers_lists()
    assert lists, "pyproject.toml declares no transformers requirement at all"
    for where, reqs in lists.items():
        general = _live(reqs, LINUX_X86)
        apple = _live(reqs, DARWIN_ARM)
        assert len(general) == 1, f"{where}: {len(general)} transformers lines apply off darwin"
        assert len(apple) == 1, f"{where}: {len(apple)} transformers lines apply on darwin + arm64"
        assert general[0].specifier != apple[0].specifier, (
            f"{where}: darwin + arm64 and everything else resolve the same transformers "
            f"window ({general[0].specifier}), so the marker split is doing nothing. Either "
            f"drop it or restore the lower Apple Silicon cap."
        )


def test_the_two_halves_are_where_they_are_supposed_to_be() -> None:
    for where, reqs in _transformers_lists().items():
        general_spec = _live(reqs, LINUX_X86)[0].specifier
        apple_spec = _live(reqs, DARWIN_ARM)[0].specifier
        general = _ceiling(general_spec)
        apple = _ceiling(apple_spec)
        # Admission as well as the ceiling number, because a second upper bound, or an
        # exclusion naming the tested release, can shut it out while the ceiling still
        # reads right.
        assert TESTED_CEILING in general_spec, (
            f"{where} declares {general_spec} off darwin, which does not admit the "
            f"{TESTED_CEILING} the matrix was run against."
        )
        assert MLX_CEILING in apple_spec, (
            f"{where} declares {apple_spec} on darwin + arm64, which does not admit the "
            f"{MLX_CEILING} the MLX stack holds."
        )
        assert general == TESTED_CEILING, (
            f"{where} caps transformers at {general} off darwin; the matrix was run "
            f"against {TESTED_CEILING}. Moving the cap means running the sweep first and "
            f"moving TESTED_CEILING here in the same commit."
        )
        assert apple == MLX_CEILING, (
            f"{where} caps transformers at {apple} on darwin + arm64, not the "
            f"{MLX_CEILING} the MLX stack holds. See the comment on the mlx-vlm pin."
        )


def test_the_apple_cap_is_what_holds_mlx_vlm_at_0_6_4() -> None:
    """The reason the Apple half is lower, asserted rather than only written down."""
    for where, reqs in _transformers_lists().items():
        apple = _ceiling(_live(reqs, DARWIN_ARM)[0].specifier)
        assert apple < MLX_VLM_0_6_5_TRANSFORMERS_FLOOR, (
            f"{where}: the darwin + arm64 cap ({apple}) now admits the "
            f"transformers >= {MLX_VLM_0_6_5_TRANSFORMERS_FLOOR} that mlx-vlm 0.6.5 and up "
            f"require, so the joint MLX resolution is free to leave 0.6.4 and "
            f"tests/mlx_simulation stops modelling what gets installed."
        )


def test_both_halves_keep_every_rejected_release_rejected() -> None:
    for where, reqs in _transformers_lists().items():
        for req in reqs:
            readmitted = [v for v in REJECTED if v in req.specifier]
            assert not readmitted, (
                f"{where}: {req.specifier} now admits {readmitted}, which were tested and "
                f"rejected."
            )


def test_the_two_halves_differ_only_in_their_ceiling() -> None:
    """Otherwise one half quietly grows an exclusion or a floor the other does not have."""
    for where, reqs in _transformers_lists().items():
        shapes = set()
        for req in reqs:
            shapes.add(frozenset(
                str(s) for s in req.specifier if s.operator not in ("<=", "<")
            ))
        assert len(shapes) == 1, (
            f"{where}: the transformers lines disagree on more than the ceiling: {shapes}"
        )


def test_the_torch_ceiling_admits_what_the_matrix_ran() -> None:
    reqs = [
        req
        for raws in _requirement_lists().values()
        for req in _named(raws, "torch")
        if str(req.specifier)
    ]
    assert reqs, "pyproject.toml no longer bounds torch; retarget this test"
    bounded = [req for req in reqs if any(s.operator in ("<", "<=") for s in req.specifier)]
    assert bounded, f"no torch requirement carries an upper bound: {[str(r) for r in reqs]}"
    for req in bounded:
        assert TESTED_TORCH in req.specifier, (
            f"pyproject.toml bounds torch as {req.specifier}, which excludes the "
            f"{TESTED_TORCH} the matrix was run against."
        )
        # The EXACT bound, not merely one that admits 2.14.0. Widening <2.15.0 to <2.16.0
        # still admits the tested release, so an admission-only check would let the package
        # publish support for a torch 2.15.x nobody ran, with this file still declaring
        # TORCH_BOUND as <2.15.0.
        assert _ceiling(req.specifier) == _ceiling(SpecifierSet(TORCH_BOUND)), (
            f"pyproject.toml bounds torch as {req.specifier}, not the {TORCH_BOUND} this "
            f"file declares. Moving the bound means running the matrix on the newly "
            f"admitted releases and moving TESTED_TORCH and TORCH_BOUND here in the same "
            f"commit."
        )


def _workflow_runs() -> list[tuple[str, str]]:
    out = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        doc = yaml.safe_load(path.read_text(encoding = "utf-8")) or {}
        for job in (doc.get("jobs") or {}).values():
            if not isinstance(job, dict):
                continue
            for step in job.get("steps") or []:
                if isinstance(step, dict) and isinstance(step.get("run"), str):
                    out.append((path.name, step["run"]))
    return out


def test_no_workflow_mirror_of_the_torch_bound_drifted() -> None:
    """Three lanes install torch by hand, before `pip install -e .` can bring it, and each
    is a copy of the published bound. The remaining lanes hold an OLDER window on purpose
    and are named, because "it is lower" is otherwise indistinguishable from "it was
    forgotten", which is the bug this file is about."""
    mirrors = {}
    for name, run in _workflow_runs():
        for raw in re.findall(r'"(torch>=[^"]*)"', run):
            try:
                req = Requirement(raw)
            except InvalidRequirement:
                continue
            if any(s.operator in ("<", "<=") for s in req.specifier):
                mirrors.setdefault(name, set()).add(raw)

    missing = [name for name in PUBLISHED_TORCH_MIRRORS if name not in mirrors]
    assert not missing, (
        f"these lanes no longer install a bounded torch at all: {missing}. Either they "
        f"stopped needing one, in which case drop them from PUBLISHED_TORCH_MIRRORS, or "
        f"the install line was lost."
    )
    for name in PUBLISHED_TORCH_MIRRORS:
        drifted = sorted(raw for raw in mirrors[name] if TESTED_TORCH not in Requirement(raw).specifier)
        assert not drifted, (
            f"{name} installs {drifted}, which excludes the {TESTED_TORCH} pyproject now "
            f"admits, so the lane tests a torch users do not get."
        )
        # A mirror is a copy of the published bound, so it has to stop where the published
        # bound stops. Admitting 2.14.0 is not the same claim: a lane widened to <2.16.0
        # would install an untested 2.15.x and still pass the check above.
        widened = sorted(
            raw
            for raw in mirrors[name]
            if _ceiling(Requirement(raw).specifier) != _ceiling(SpecifierSet(TORCH_BOUND))
        )
        assert not widened, (
            f"{name} installs {widened}, which does not stop where the published "
            f"{TORCH_BOUND} stops, so the mirror is no longer a mirror."
        )

    unexplained = sorted(
        set(mirrors) - set(PUBLISHED_TORCH_MIRRORS) - set(OLDER_TORCH_WINDOW_BY_DESIGN)
    )
    assert not unexplained, (
        f"these lanes bound torch and are neither a mirror of the published ceiling nor a "
        f"recorded older window: {unexplained}. Add them to one list or the other."
    )


def _gate_constants() -> dict[str, Version]:
    """The two policy literals in the MLX job's inline gate."""
    found = {}
    for name, run in _workflow_runs():
        if "MLX_CEILING" not in run:
            continue
        for key in ("CEILING", "MLX_CEILING"):
            match = re.search(rf"^\s*{key} = Version\(\"([^\"]+)\"\)", run, re.MULTILINE)
            if match:
                found[key] = Version(match.group(1))
    return found


def test_the_ci_policy_gate_agrees_with_pyproject() -> None:
    """consolidated-tests-ci.yml fails the build when pyproject declares a higher cap than
    the gate's literal. That is deliberate, and it means the two have to move together or
    the MLX job reds on every commit."""
    constants = _gate_constants()
    assert set(constants) == {"CEILING", "MLX_CEILING"}, (
        f"the inline policy gate no longer declares both ceilings (found {sorted(constants)}); "
        f"retarget this test or restore the gate"
    )
    assert constants["CEILING"] == TESTED_CEILING, (
        f"the CI gate holds transformers at {constants['CEILING']} while this test and "
        f"pyproject say {TESTED_CEILING}; the MLX job would red on every commit"
    )
    assert constants["MLX_CEILING"] == MLX_CEILING, (
        f"the CI gate holds the Apple Silicon cap at {constants['MLX_CEILING']} while "
        f"pyproject says {MLX_CEILING}"
    )


CONSOLIDATED_CI = WORKFLOWS / "consolidated-tests-ci.yml"

# The stdlib-only resolver inside the core-drift job that turns the
# `__from_pyproject__` sentinel into a real pip spec.
_RESOLVER = re.compile(
    r"resolve\(\)\s*\{\s*\n\s*python - \"\$@\" <<'PY'\n(.*?)\n\s*PY\n",
    re.DOTALL,
)


def _sentinel_resolver_source() -> str:
    text = CONSOLIDATED_CI.read_text(encoding = "utf-8")
    match = _RESOLVER.search(text)
    assert match, (
        f"{CONSOLIDATED_CI.name} no longer carries the `__from_pyproject__` resolver this "
        f"test exercises; retarget the test or restore the step"
    )
    return textwrap.dedent(match.group(1))


# The resolver is stdlib-only but not stdlib-ANY: it imports `tomllib`, which arrived in
# CPython 3.11. The job that runs it pins 3.12, so the workflow is fine, and
# `test_the_sentinel_resolver_runs_on_a_python_that_has_tomllib` below is what keeps that
# true. The tests here run under whatever interpreter pytest was started with, though, and
# this repository supports 3.10: under it the subprocess died with
# `ModuleNotFoundError: No module named 'tomllib'` and both tests failed for a reason that
# had nothing to do with what they assert. Worse, the negative control still entered its
# `pytest.raises(AssertionError)` and then failed on the message, which is the shape of a
# test that has quietly stopped testing.
_RESOLVER_NEEDS = (3, 11)


def _resolve_sentinel(pyproject_text: str, tmp_path: Path) -> list[str]:
    """Run the workflow's own resolver against `pyproject_text`, as the job does."""
    workdir = tmp_path / "repo"
    workdir.mkdir(exist_ok = True)
    (workdir / "pyproject.toml").write_text(pyproject_text, encoding = "utf-8")
    script = tmp_path / "resolve.py"
    script.write_text(_sentinel_resolver_source(), encoding = "utf-8")
    finished = subprocess.run(
        [sys.executable, str(script), "__from_pyproject__", "__from_pyproject__", "__from_pyproject__"],
        cwd = workdir,
        capture_output = True,
        text = True,
    )
    if finished.returncode != 0:
        raise AssertionError(
            f"the resolver exited {finished.returncode}:\n{finished.stdout}\n{finished.stderr}"
        )
    return finished.stdout.strip().splitlines()


requires_tomllib = pytest.mark.skipif(
    sys.version_info < _RESOLVER_NEEDS,
    reason = (
        "the core-drift resolver imports tomllib, added in CPython 3.11, and these tests run "
        "it under the ambient interpreter; the job itself pins 3.12, which "
        "test_the_sentinel_resolver_runs_on_a_python_that_has_tomllib asserts statically on "
        "every interpreter, so this skip cannot hide the resolver losing its python"
    ),
)


def test_the_sentinel_resolver_runs_on_a_python_that_has_tomllib() -> None:
    """The job carrying the resolver must pin a python new enough to import `tomllib`.

    Static, so it holds on 3.10 as well, which is where the two tests below cannot run.
    Without this the skip above would be a hole: someone could drop the job's
    `python-version` to 3.10 and the only tests that execute the resolver would skip
    rather than fail.
    """
    source = _sentinel_resolver_source()
    if "tomllib" not in source:
        pytest.skip("the resolver no longer imports tomllib, so there is no floor to hold")

    document = yaml.safe_load(CONSOLIDATED_CI.read_text(encoding = "utf-8"))
    owners = [
        name for name, job in (document.get("jobs") or {}).items()
        if "__from_pyproject__" in yaml.safe_dump(job)
    ]
    assert owners, (
        "no job in consolidated-tests-ci.yml carries the __from_pyproject__ sentinel any "
        "more; retarget this test or restore the step"
    )
    for name in owners:
        job = document["jobs"][name]
        pinned = [
            str(step.get("with", {}).get("python-version"))
            for step in (job.get("steps") or [])
            if str(step.get("uses", "")).startswith("actions/setup-python")
            and (step.get("with") or {}).get("python-version") is not None
        ]
        assert pinned, f"job {name} runs the tomllib resolver without pinning a python"
        for version in pinned:
            parts = tuple(int(part) for part in version.split(".")[:2])
            assert parts >= _RESOLVER_NEEDS, (
                f"job {name} pins python {version} and its resolver imports tomllib, which "
                f"needs {'.'.join(str(p) for p in _RESOLVER_NEEDS)} or newer; the lane would "
                f"die on ModuleNotFoundError before resolving the cap"
            )


@requires_tomllib
def test_the_ci_sentinel_resolves_to_the_off_darwin_half(tmp_path) -> None:
    """The core-drift lane is Linux, so it has to install the Linux half of the cap.

    Before the split there was one transformers line and any of them was the right one.
    With two, a resolver that takes whichever comes first in the file installs the 5.5.0
    Apple Silicon cap on a Linux runner and the lane silently measures the wrong range,
    which is the exact failure this whole file exists to prevent. Nothing else in the repo
    executes this step, so it is executed here.
    """
    transformers_spec, trl_spec, peft_spec = _resolve_sentinel(
        PYPROJECT.read_text(encoding = "utf-8"), tmp_path
    )
    assert Requirement(transformers_spec).name.lower() == "transformers"
    assert ";" not in transformers_spec, (
        f"the resolver left an environment marker on {transformers_spec!r}; pip install "
        f"would take it as a separate argument"
    )
    ceiling = max(
        Version(str(spec.version))
        for spec in Requirement(transformers_spec).specifier
        if spec.operator in ("<", "<=")
    )
    assert ceiling == TESTED_CEILING, (
        f"the Linux core-drift lane would install transformers {transformers_spec!r}, whose "
        f"ceiling is {ceiling}, not the {TESTED_CEILING} that half of the cap declares. A "
        f"lane pinned to the Apple Silicon half tests a range Linux users do not get."
    )
    for spec, name in ((trl_spec, "trl"), (peft_spec, "peft")):
        assert Requirement(spec).name.lower() == name
        assert ";" not in spec


@requires_tomllib
def test_the_ci_sentinel_refuses_two_different_off_darwin_specs(tmp_path) -> None:
    """Negative control, so the test above cannot pass by the resolver doing nothing.

    Widening the Apple Silicon half to a DIFFERENT off-darwin ceiling leaves the lane with
    two candidates and no rule for choosing, and it has to say so rather than pick one.
    """
    text = PYPROJECT.read_text(encoding = "utf-8")
    widened = text.replace(
        "sys_platform == 'darwin' and platform_machine == 'arm64'",
        "sys_platform != 'darwin' or platform_machine != 'arm64'",
    )
    assert widened != text, "the marker split is not spelled the way this test assumed"
    with pytest.raises(AssertionError) as raised:
        _resolve_sentinel(widened, tmp_path)
    assert "different off-darwin specs" in str(raised.value), str(raised.value)


def test_the_checker_rejects_the_window_that_shipped_the_defect() -> None:
    """Negative control. Every assertion above is a "nothing found" or "equals" shape, and
    a checker that has stopped checking reports the same thing."""
    shipped = SpecifierSet("".join(f"!={v}," for v in REJECTED) + f">={DECLARED_FLOOR},<=5.5.0")
    assert "5.5.0" in shipped
    assert "5.17.0" not in shipped
    assert Version("2.14.0") not in SpecifierSet(">=2.4.0,<2.13.0")
    assert Version("2.14.0") in SpecifierSet(f">=2.4.0,{TORCH_BOUND}")


def test_this_file_runs_in_an_executing_ci_step():
    """A gate nothing executes is not a gate.

    Across `.github/workflows` this file used to be reached only by
    `pytest tests/ --collect-only`, which proves it imports and nothing else. The
    assertions above are what stop a cap regression -- the inline gate checks only
    `general <= CEILING`, so lowering the ceiling passes it while failing here -- and
    with no executing step that regression merges green.
    """
    import re
    from pathlib import Path

    workflows = Path(__file__).resolve().parents[1] / ".github" / "workflows"
    name = Path(__file__).name
    executing = []
    for workflow in sorted(workflows.glob("*.yml")):
        text = workflow.read_text(encoding = "utf-8")
        for run_block in re.findall(r"run:\s*\|(.*?)(?=\n\s{6}[-\w]|\Z)", text, re.S):
            if "--collect-only" in run_block:
                continue
            if name in run_block and "pytest" in run_block:
                executing.append(workflow.name)
    assert executing, (
        f"{name} is not named in any executing pytest step; it is collected but never "
        f"run, so every assertion in it is inert in CI"
    )


def test_the_ceiling_helper_reports_the_tightest_upper_bound() -> None:
    """NEGATIVE CONTROL for `_ceiling`.

    Upper bounds intersect, so the effective ceiling is the tightest one. Taking the
    loosest reported 5.17 for `<=5.17.0,<5.16.0`, a window that excludes the tested
    release, and the shape comparison next door strips every upper bound, so nothing else
    in this file would have caught it.
    """
    assert _ceiling(SpecifierSet("<=5.17.0")) == Version("5.17.0")
    assert _ceiling(SpecifierSet(">=4.51.3,<=5.17.0")) == Version("5.17.0")
    assert _ceiling(SpecifierSet("<=5.17.0,<5.16.0")) == Version("5.16.0")
    assert _ceiling(SpecifierSet("<5.16.0,<=5.17.0")) == Version("5.16.0")
    # A tie on the number is broken by exclusivity: `<5.16.0` admits strictly less.
    assert _ceiling(SpecifierSet("<5.16.0,<=5.16.0")) == Version("5.16.0")
    assert Version("5.17.0") not in SpecifierSet("<=5.17.0,<5.16.0")


def test_a_widened_torch_bound_is_rejected_even_though_it_admits_the_tested_release() -> None:
    """NEGATIVE CONTROL for the exact-bound assertions.

    `<2.16.0` still contains the 2.14.0 the matrix ran, so the admission check passes on
    it. That is the drift this file exists to catch: it would publish support for a torch
    2.15.x nobody tested while TORCH_BOUND still said <2.15.0.
    """
    widened = SpecifierSet(">=2.4.0,<2.16.0")
    assert TESTED_TORCH in widened, "the premise: an admission-only check cannot see this"
    assert _ceiling(widened) != _ceiling(SpecifierSet(TORCH_BOUND))
    assert _ceiling(SpecifierSet(f">=2.4.0,{TORCH_BOUND}")) == _ceiling(SpecifierSet(TORCH_BOUND))


def test_both_halves_declare_the_floor_that_peft_needs() -> None:
    """The ceiling is pinned by several assertions above; the floor was pinned by none.

    That asymmetry is not hypothetical. This branch was cut before the floor moved and
    carried `>=4.51.3` onto a main that had already gone to 4.52.4, so the marker split
    would have shipped a REGRESSION of the floor while every ceiling assertion stayed
    green. `test_the_two_halves_differ_only_in_their_ceiling` does not catch it either: it
    only requires the two halves to agree with EACH OTHER, and a stale floor is equally
    stale on both.
    """
    lists = _transformers_lists()
    assert lists, "no requirement list names transformers"
    wrong = {}
    for where, reqs in lists.items():
        for req in reqs:
            floors = [
                Version(str(spec.version))
                for spec in req.specifier
                if spec.operator in (">=", "==", "~=")
            ]
            assert floors, f"{where}: transformers requirement {req} declares no floor"
            if max(floors) != DECLARED_FLOOR:
                wrong[f"{where}: {req}"] = str(max(floors))
    assert not wrong, (
        f"these transformers requirements do not declare the {DECLARED_FLOOR} floor "
        f"peft 0.18.0 needs: {wrong}. A floor below 4.52.0 resolves cleanly and then "
        f"fails at import with ModuleNotFoundError: transformers.modeling_layers."
    )

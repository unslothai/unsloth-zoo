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

"""Guards the published window (transformers<=5.5.0, torch<2.13.0) and CI mirrors; consolidated-tests-ci.yml CEILING
fails when pyproject moves past it; ceiling = newest tested release (CUDA: unsloth issues 9867 / 10010 / 10017 /
10276, 5355); mlx-vlm 0.6.5+ needs transformers>=5.14.0 on darwin arm64."""

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

# Newest transformers the version matrix was run against.
TESTED_CEILING = Version("5.17.0")

# Newest torch the matrix was run against, and the exclusive bound that admits it.
TESTED_TORCH = Version("2.14.0")
TORCH_BOUND = "<2.15.0"

# Floor for every transformers line, since peft declares none: peft 0.18.0 imports GradientCheckpointingLayer from
# transformers.modeling_layers (4.52.0+); on 4.51.3 import unsloth_zoo.saving_utils raised ModuleNotFoundError though
# the resolve passed; 4.52.4 not 4.52.0 as 4.52.0-4.52.3 are REJECTED.
DECLARED_FLOOR = Version("4.52.4")

# Tested and rejected. A rewrite of the specifier must not drop one.
REJECTED = (
    "4.52.0", "4.52.1", "4.52.2", "4.52.3", "4.53.0", "4.54.0",
    "4.55.0", "4.55.1", "4.57.4", "4.57.5", "5.0.0", "5.1.0",
)

# Lanes that hand-install torch as a copy of the published ceiling; each must move when the ceiling does.
PUBLISHED_TORCH_MIRRORS = (
    "gemma4-audio-probe.yml",
    "mlx-ci.yml",
    "security-audit.yml",
)

# Lanes that hold an older torch window on purpose: the MLX lanes need the torch / torchvision ABI pair mlx-cpu was
# built against, a different question from what is published.
OLDER_TORCH_WINDOW_BY_DESIGN = {
    "consolidated-tests-ci.yml": (
        "the interpreter, repo, MLX and core-drift lanes pin torch>=2.4.0,<2.11.0 for the "
        "CPU torch / torchvision ABI pair those suites were measured against"
    ),
}

DARWIN_ARM = {"sys_platform": "darwin", "platform_machine": "arm64"}
LINUX_X86 = {"sys_platform": "linux", "platform_machine": "x86_64"}


def _requirement_lists() -> dict[str, list[str]]:
    """Returns {where: [raw requirement, ...]} for the base list and every extra; tomllib is 3.11+ but requires-python
    is >=3.9, so older interpreters skip rather than fail to collect and red the python-version-collect lane. Same
    shape as _find_config in tests/test_wheel_top_level_packages.py."""
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
            # An extra == marker has no value in a bare environment; these pyproject-sourced lists carry none.
            continue
    return out


def _ceiling(spec: SpecifierSet) -> Version:
    """Returns the tightest upper bound, not the loosest: bounds intersect, so `<=5.17.0,<5.16.0` admits nothing above 5.16.0; ties go to exclusive `<`."""
    return _ceiling_bound(spec)[0]


def _ceiling_bound(spec: SpecifierSet) -> tuple[Version, bool]:
    """Returns the tightest upper bound and its exclusivity: `<2.15.0` and `<=2.15.0` share a version but admit different releases, so compare both."""
    tops = [(Version(str(s.version)), s.operator == "<") for s in spec if s.operator in ("<=", "<")]
    assert tops, f"no upper bound declared: {spec}"
    return min(tops, key = lambda pair: (pair[0], pair[1]))


def _transformers_lists() -> dict[str, list[Requirement]]:
    return {
        where: reqs
        for where, raws in _requirement_lists().items()
        if (reqs := _named(raws, "transformers"))
    }


def test_every_list_that_names_transformers_applies_one_line_per_platform() -> None:
    lists = _transformers_lists()
    assert lists, "pyproject.toml declares no transformers requirement at all"
    for where, reqs in lists.items():
        general = _live(reqs, LINUX_X86)
        apple = _live(reqs, DARWIN_ARM)
        assert len(general) == 1, f"{where}: {len(general)} transformers lines apply off darwin"
        assert len(apple) == 1, f"{where}: {len(apple)} transformers lines apply on darwin + arm64"


def test_each_platform_caps_transformers_at_the_tested_ceiling() -> None:
    for where, reqs in _transformers_lists().items():
        for platform, environment in (("off darwin", LINUX_X86), ("on darwin + arm64", DARWIN_ARM)):
            spec = _live(reqs, environment)[0].specifier
            # Check admission too, not just the ceiling number: a second upper bound, or an exclusion of the tested
            # release, can shut it out while the ceiling still reads right.
            assert TESTED_CEILING in spec, (
                f"{where} declares {spec} {platform}, which does not admit the "
                f"{TESTED_CEILING} the matrix was run against."
            )
            assert _ceiling(spec) == TESTED_CEILING, (
                f"{where} caps transformers at {_ceiling(spec)} {platform}; the matrix was "
                f"run against {TESTED_CEILING}. Moving the cap means running the sweep first "
                f"and moving TESTED_CEILING here in the same commit."
            )


def test_every_transformers_line_keeps_every_rejected_release_rejected() -> None:
    for where, reqs in _transformers_lists().items():
        for req in reqs:
            readmitted = [v for v in REJECTED if v in req.specifier]
            assert not readmitted, (
                f"{where}: {req.specifier} now admits {readmitted}, which were tested and "
                f"rejected."
            )


def test_the_transformers_lines_differ_only_in_their_ceiling() -> None:
    """If the cap is marker-split again, the lines may differ only in their ceiling, not by an exclusion or floor only one of them has."""
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
        # Compare the exact bound, not just admission: widening <2.15.0 to <2.16.0 still admits the tested 2.14.0, so an
        # admission-only check would let untested torch 2.15.x ship.
        assert _ceiling_bound(req.specifier) == _ceiling_bound(SpecifierSet(TORCH_BOUND)), (
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
    """Three lanes install torch by hand before `pip install -e .` and copy the published bound; other lanes hold an
    older window on purpose and are named, not forgotten."""
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
        # A mirror copies the published bound, so it must stop where that bound stops; a lane widened to <2.16.0 would
        # install untested 2.15.x and still pass the check above.
        widened = sorted(
            raw
            for raw in mirrors[name]
            if _ceiling_bound(Requirement(raw).specifier) != _ceiling_bound(SpecifierSet(TORCH_BOUND))
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
    """The policy literal in the MLX job's inline gate."""
    found = {}
    for name, run in _workflow_runs():
        match = re.search(r"^\s*CEILING = Version\(\"([^\"]+)\"\)", run, re.MULTILINE)
        if match:
            found[name] = Version(match.group(1))
    return found


def test_the_ci_policy_gate_agrees_with_pyproject() -> None:
    """consolidated-tests-ci.yml fails the build when pyproject declares a higher cap than its literal, so both must move together or the MLX job fails every commit."""
    constants = _gate_constants()
    assert set(constants) == {"consolidated-tests-ci.yml"}, (
        f"the inline policy gate is not where this test expects it (found {sorted(constants)}); "
        f"retarget this test or restore the gate"
    )
    assert constants["consolidated-tests-ci.yml"] == TESTED_CEILING, (
        f"the CI gate holds transformers at {constants['consolidated-tests-ci.yml']} while "
        f"this test and pyproject say {TESTED_CEILING}; the MLX job would red on every commit"
    )


CONSOLIDATED_CI = WORKFLOWS / "consolidated-tests-ci.yml"

# Stdlib-only resolver in the core-drift job that turns the `__from_pyproject__` sentinel into a real pip spec.
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


# Resolver needs tomllib (CPython 3.11+); CI pins 3.12, but these tests run on any pytest interpreter, and on 3.10 the
# subprocess dies with ModuleNotFoundError: _RESOLVER_NEEDS holds the minimum; a negative control that fails on its
# message has stopped testing.
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
    """The job running the resolver must pin Python new enough for `tomllib`; this static check also holds on 3.10,
    where the resolver tests cannot run; a 3.10 pin would make the only tests that run the resolver skip rather than
    fail."""
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


def _with_second_transformers_line(text: str, marker: str) -> str:
    """pyproject text with an extra transformers line, capped lower, under `marker`."""
    line = re.search(r'^(\s*)"(transformers[^"]*)",$', text, re.MULTILINE)
    assert line, "the transformers requirement is not spelled the way this test assumed"
    indent, spec = line.groups()
    second = spec.split(";", 1)[0].strip().replace(f"<={TESTED_CEILING}", "<=5.5.0")
    assert second != spec, "the transformers requirement does not carry the tested ceiling"
    return text.replace(line.group(0), f'{line.group(0)}\n{indent}"{second} ; {marker}",', 1)


@requires_tomllib
@pytest.mark.parametrize("darwin_only_line", [False, True])
def test_the_ci_sentinel_resolves_to_the_off_darwin_line(tmp_path, darwin_only_line) -> None:
    """The core-drift lane is Linux, so the resolver must pick the off-darwin cap line even when a darwin-only line comes first; no other test runs this step."""
    text = PYPROJECT.read_text(encoding = "utf-8")
    if darwin_only_line:
        text = _with_second_transformers_line(
            text, "sys_platform == 'darwin' and platform_machine == 'arm64'"
        )
    transformers_spec, trl_spec, peft_spec = _resolve_sentinel(text, tmp_path)
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
        f"ceiling is {ceiling}, not the {TESTED_CEILING} the Linux line declares. A lane "
        f"pinned to an Apple Silicon line tests a range Linux users do not get."
    )
    for spec, name in ((trl_spec, "trl"), (peft_spec, "peft")):
        assert Requirement(spec).name.lower() == name
        assert ";" not in spec


@requires_tomllib
def test_the_ci_sentinel_refuses_two_different_off_darwin_specs(tmp_path) -> None:
    """Negative control so the test above cannot pass trivially: two off-darwin transformers lines with different ceilings must be refused, not one picked."""
    widened = _with_second_transformers_line(
        PYPROJECT.read_text(encoding = "utf-8"), "sys_platform != 'darwin'"
    )
    with pytest.raises(AssertionError) as raised:
        _resolve_sentinel(widened, tmp_path)
    assert "different off-darwin specs" in str(raised.value), str(raised.value)


def test_the_checker_rejects_the_window_that_shipped_the_defect() -> None:
    """Negative control: every assertion above is a nothing-found or equals check, which a checker that has stopped checking would also pass."""
    shipped = SpecifierSet("".join(f"!={v}," for v in REJECTED) + f">={DECLARED_FLOOR},<=5.5.0")
    assert "5.5.0" in shipped
    assert "5.17.0" not in shipped
    assert Version("2.14.0") not in SpecifierSet(">=2.4.0,<2.13.0")
    assert Version("2.14.0") in SpecifierSet(f">=2.4.0,{TORCH_BOUND}")


def test_this_file_runs_in_an_executing_ci_step():
    """Must run in an executing CI step (collect-only proves just imports); the inline gate checks only general <=
    CEILING, so a lowered ceiling passes it but fails here."""
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
    """Negative control for _ceiling: upper bounds intersect so the tightest wins; reporting 5.17 for <=5.17.0,<5.16.0
    (which excludes the tested release) is wrong, and nothing else in this file catches it."""
    assert _ceiling(SpecifierSet("<=5.17.0")) == Version("5.17.0")
    assert _ceiling(SpecifierSet(">=4.51.3,<=5.17.0")) == Version("5.17.0")
    assert _ceiling(SpecifierSet("<=5.17.0,<5.16.0")) == Version("5.16.0")
    assert _ceiling(SpecifierSet("<5.16.0,<=5.17.0")) == Version("5.16.0")
    # A tie on the number is broken by exclusivity: `<5.16.0` admits strictly less.
    assert _ceiling(SpecifierSet("<5.16.0,<=5.16.0")) == Version("5.16.0")
    assert Version("5.17.0") not in SpecifierSet("<=5.17.0,<5.16.0")


def test_a_widened_torch_bound_is_rejected_even_though_it_admits_the_tested_release() -> None:
    """Negative control for exact-bound checks: <2.16.0 still admits the tested 2.14.0, so admission alone passes it,
    yet it would publish untested torch 2.15.x while TORCH_BOUND says <2.15.0."""
    widened = SpecifierSet(">=2.4.0,<2.16.0")
    assert TESTED_TORCH in widened, "the premise: an admission-only check cannot see this"
    assert _ceiling_bound(widened) != _ceiling_bound(SpecifierSet(TORCH_BOUND))
    assert _ceiling_bound(SpecifierSet(f">=2.4.0,{TORCH_BOUND}")) == _ceiling_bound(SpecifierSet(TORCH_BOUND))
    # The case the version-only comparison could not see: same number, different set.
    assert _ceiling(SpecifierSet("<=2.15.0")) == _ceiling(SpecifierSet("<2.15.0"))
    assert _ceiling_bound(SpecifierSet("<=2.15.0")) != _ceiling_bound(SpecifierSet("<2.15.0"))
    assert Version("2.15.0") in SpecifierSet("<=2.15.0")
    assert Version("2.15.0") not in SpecifierSet("<2.15.0")


def test_every_transformers_line_declares_the_floor_that_peft_needs() -> None:
    """Floor needs its own assertion, since no ceiling check covers it: a branch cut before the floor moved carried
    >=4.51.3 onto a main already at 4.52.4, and lines-agree tests miss it."""
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


# unsloth's patch_datasets raises at import on 4.4.0 <= v <= 4.5.0.
UNSLOTH_REFUSED_DATASETS = ("4.4.0", "4.4.1", "4.4.2", "4.5.0")


def _datasets_lists() -> dict[str, list[Requirement]]:
    return {
        where: reqs
        for where, raws in _requirement_lists().items()
        if (reqs := _named(raws, "datasets"))
    }


def _refused_admitted_by(specifier: SpecifierSet) -> list[str]:
    return [r for r in UNSLOTH_REFUSED_DATASETS if r in specifier]


# Named, not discovered: a list that drops datasets vanishes from _datasets_lists.
DATASETS_LOCATIONS = ("dependencies", "optional-dependencies.core")


def test_every_list_declares_the_same_datasets_window() -> None:
    lists = _datasets_lists()
    missing = [where for where in DATASETS_LOCATIONS if where not in lists]
    assert not missing, (
        f"{missing} no longer declares datasets, so pip would take the window from the other "
        f"list alone. Removing it on purpose means removing it from DATASETS_LOCATIONS too."
    )
    windows = {str(req.specifier) for reqs in lists.values() for req in reqs}
    assert len(windows) == 1, (
        f"pyproject.toml declares {len(windows)} different datasets windows across its "
        f"requirement lists: {sorted(windows)}"
    )


def test_the_datasets_window_excludes_what_unsloth_refuses_at_import() -> None:
    lists = _datasets_lists()
    assert lists, "pyproject.toml declares no datasets requirement at all"
    admitted = sorted(
        {
            r
            for reqs in lists.values()
            for req in reqs
            for r in _refused_admitted_by(req.specifier)
        }
    )
    assert not admitted, (
        f"the declared datasets window admits {admitted}, which unsloth's "
        f"patch_datasets refuses at import, so pip can resolve a release that cannot run"
    )


def test_the_datasets_checker_rejects_the_window_that_would_ship_the_defect() -> None:
    assert _refused_admitted_by(SpecifierSet(">=3.4.1,<5.0.0")) == list(
        UNSLOTH_REFUSED_DATASETS
    )
    assert _refused_admitted_by(SpecifierSet(">=3.4.1,!=4.4.*,<5.0.0")) == ["4.5.0"]


# Every trl 1.x declares datasets>=4.7.0 (PyPI metadata).
TRL_1_DATASETS_FLOOR = Version("4.7.0")
DATASETS_RELEASES = (
    "4.3.0", "4.4.0", "4.4.1", "4.4.2", "4.5.0", "4.6.0", "4.6.1", "4.7.0", "4.8.0", "4.8.5",
)


def test_every_trl_ceiling_is_reachable_under_the_datasets_window() -> None:
    """pip silently backtracks to trl 0.x rather than erroring on an unreachable ceiling."""
    for where, raws in _requirement_lists().items():
        datasets = _named(raws, "datasets")
        for trl in _named(raws, "trl"):
            if _ceiling(trl.specifier) < Version("1.0.0"):
                continue
            usable = [
                v
                for v in DATASETS_RELEASES
                if Version(v) >= TRL_1_DATASETS_FLOOR
                and all(v in d.specifier for d in datasets)
            ]
            assert usable, (
                f"{where}: trl ceiling {_ceiling(trl.specifier)} needs datasets>="
                f"{TRL_1_DATASETS_FLOOR}, which {[str(d.specifier) for d in datasets]} excludes"
            )


def test_the_linux_mlx_lane_mirrors_the_published_mlx_lm_pin() -> None:
    """The Linux MLX lane must repeat pyproject's darwin+arm64-only mlx-lm pin; unmirrored, it resolved mlx-lm 0.32.0
    (transformers>=5.7.0) after the cap lifted, a stack no Apple Silicon install reaches."""
    published = {
        str(req.specifier)
        for req in _live(_named(_requirement_lists()["dependencies"], "mlx-lm"), DARWIN_ARM)
    }
    assert len(published) == 1, published
    lanes = [
        str(Requirement(m).specifier)
        for name, run in _workflow_runs()
        if name == "consolidated-tests-ci.yml" and "venv-suites/bin/pip install -e" in run
        for m in re.findall(r'"(mlx-lm[^"]*)"', run)
    ]
    assert lanes and set(lanes) == published, (lanes, published)

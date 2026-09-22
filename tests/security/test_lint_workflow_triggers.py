"""Regression tests for scripts/lint_workflow_triggers.py.

Guards against future regressions that would re-introduce GHSA-g7cv-rxg3-hmpx
(TanStack) -class supply-chain vectors:
  * pull_request_target (fork PR runs in base context).
  * Shared cache keys between PR-triggered workflows and the publish workflow.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "lint_workflow_triggers.py"


def _run(workflows_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--workflows-dir", str(workflows_dir)],
        capture_output = True,
        text = True,
    )


def test_lint_passes_on_current_workflows():
    """The live `.github/workflows/` tree must pass the lint."""
    live = REPO_ROOT / ".github" / "workflows"
    proc = _run(live)
    assert (
        proc.returncode == 0
    ), f"live tree failed lint:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def test_lint_rejects_pull_request_target(tmp_path):
    """Synthetic PR_TARGET trigger must produce rc=1 with a named finding."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "bad.yml").write_text(
        "name: bad\n"
        "on:\n"
        "  pull_request_target:\n"
        "    branches: [main]\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: echo evil\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1
    assert "BANNED trigger 'pull_request_target'" in proc.stderr
    assert "GHSA-g7cv-rxg3-hmpx" in proc.stderr


def test_lint_rejects_unjustified_workflow_run(tmp_path):
    """`workflow_run` requires an explicit allow-comment in the YAML."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "chained.yml").write_text(
        "name: chained\n"
        "on:\n"
        "  workflow_run:\n"
        "    workflows: ['CI']\n"
        "    types: [completed]\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: echo elevated\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1
    assert "RESTRICTED trigger 'workflow_run'" in proc.stderr


def test_lint_allows_justified_workflow_run(tmp_path):
    """With the allow-comment, workflow_run is permitted."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "chained.yml").write_text(
        "# lint:workflow_triggers-allow-workflow_run -- justified by ticket #1234\n"
        "name: chained\n"
        "on:\n"
        "  workflow_run:\n"
        "    workflows: ['CI']\n"
        "    types: [completed]\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: echo elevated\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, f"justified workflow_run rejected:\n{proc.stderr}"


def test_lint_rejects_shared_cache_key_between_pr_and_publish(tmp_path):
    """A cache key declared in both a PR-triggered workflow and the
    publish workflow is the TanStack cache-poisoning vector."""
    wf = tmp_path / "wf"
    wf.mkdir()
    # PR-triggered: writes to a cache that the publish job will also restore.
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n"
        "          path: node_modules\n"
        "          key: shared-cache-v1\n"
    )
    # Publish workflow with the IDENTICAL cache key -- the actual attack pattern.
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n"
        "          path: node_modules\n"
        "          key: shared-cache-v1\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1
    assert "cache-key" in proc.stderr.lower() or "cache key" in proc.stderr.lower()
    assert "shared-cache-v1" in proc.stderr


def _pr_workflow(text_key: str) -> str:
    return (
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n"
        "          path: wheels\n"
        f"          key: {text_key}\n"
    )


def test_lint_rejects_a_publish_restore_keys_prefix_over_a_pr_namespace(tmp_path):
    """A prefix restore reaches the same cache an equal key would.

    `restore-keys` restores the newest entry whose key merely STARTS WITH the prefix, so
    a publish workflow can adopt an entry a pull request wrote while no two keys are
    equal. The exact-key check beside this one compares whole strings and never saw it.
    """
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("pip-v2-${{ runner.os }}-abc"))
    (wf / "wheel-smoke.yml").write_text(
        "name: wheel-smoke\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: pip-v2-publish-${{ runner.os }}\n"
        "          restore-keys: |\n"
        "            pip-\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, f"prefix restore accepted:\n{proc.stdout}\n{proc.stderr}"
    assert "restore-keys" in proc.stderr
    assert "'pip-'" in proc.stderr


def test_a_partitioned_publish_prefix_is_accepted(tmp_path):
    """The fix must pass, or the rule above is just a ban on restore-keys."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("pip-v2-${{ runner.os }}-abc"))
    (wf / "wheel-smoke.yml").write_text(
        "name: wheel-smoke\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: pip-publish-only-${{ runner.os }}\n"
        "          restore-keys: |\n"
        "            pip-publish-only-\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, f"partitioned prefix rejected:\n{proc.stderr}"


def _publish_with_restore_keys(key: str, prefixes: str) -> str:
    return (
        "name: wheel-smoke\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        f"          key: {key}\n"
        "          restore-keys: |\n" + prefixes
    )


def test_a_publish_prefix_longer_than_the_pr_literal_head_is_caught(tmp_path):
    """One-directional prefix comparison missed the common shape.

    `pip-${{ runner.os }}-abc` has the literal head `pip-`. A publish prefix `pip-Linux-`
    is longer than that head, so `head.startswith(prefix)` is False and the pairing was
    accepted, while the runtime key `pip-Linux-abc` does start with it. restore-keys
    matching is left-anchored and exact, so either string being a prefix of the other
    means they can meet.
    """
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("pip-${{ runner.os }}-abc"))
    (wf / "wheel-smoke.yml").write_text(
        _publish_with_restore_keys("pip-pub-${{ runner.os }}", "            pip-Linux-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, f"longer publish prefix accepted:\n{proc.stdout}\n{proc.stderr}"


def test_an_expression_led_pr_key_is_expanded_not_dropped(tmp_path):
    """Dropping these was a gate bypass: the PR side reduced to the empty string."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("${{ runner.os }}-shared-abc"))
    (wf / "wheel-smoke.yml").write_text(
        _publish_with_restore_keys("pub-${{ runner.os }}", "            Linux-shared-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"expression-led PR key silently dropped:\n{proc.stdout}\n{proc.stderr}"
    )


def test_a_composite_that_builds_its_key_in_shell_is_read(tmp_path):
    """This repo's pip cache names its namespace in shell, not in `key:`.

    pip-cache-restore sets `prefix="pip-${name}-..."` in a run step and exposes it as an
    output, so its YAML `key:` is only `${{ steps.probe.outputs.key }}`. Reading YAML
    alone learned nothing about the action this check exists to cover.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "pip-cache-restore"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: pip cache restore\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - id: probe\n"
        "      shell: bash\n"
        "      run: |\n"
        '        prefix="pip-${name}-${{ runner.os }}-py${pyver}-"\n'
        '        echo "key=${prefix}${hash}" >> "$GITHUB_OUTPUT"\n'
        "    - uses: actions/cache/restore@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: ${{ steps.probe.outputs.key }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pip-cache-restore\n"
    )
    (wf / "wheel-smoke.yml").write_text(
        _publish_with_restore_keys("pip-pub", "            pip-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the composite's shell-built pip- namespace was not seen:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


def test_a_publish_only_composite_is_not_treated_as_a_pr_namespace(tmp_path):
    """Scanning every action made a publish-only cache reject its own prefix."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "release-cache"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: release cache\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache/restore@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: release-only-${{ runner.os }}\n"
    )
    (wf / "pr-build.yml").write_text(_pr_workflow("pip-${{ runner.os }}-abc"))
    (wf / "wheel-smoke.yml").write_text(
        "name: wheel-smoke\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/release-cache\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: release-only-pub\n"
        "          restore-keys: |\n"
        "            release-only-\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"a publish-only composite was treated as a PR namespace:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


def test_a_restore_keys_prefix_after_a_blank_line_is_still_read(tmp_path):
    """A blank line inside the block scalar used to truncate the list silently.

    A YAML block scalar runs until the indentation drops, blank lines included, and
    actions/cache reads the value as a newline-delimited list and skips empty entries. So
    `safe-`, a blank line, then `shared-` really does offer `shared-` at runtime, while
    the reader stopped at the blank line and never compared it. That is a bypass anyone
    can reach by formatting a long restore-keys block for readability, and the dropped
    entries are exactly the ones furthest from the eye.
    """
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("shared-${{ runner.os }}-abc"))
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys(
            "release-only-${{ runner.os }}",
            "            release-only-\n"
            "\n"
            "            shared-\n",
        )
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a prefix after a blank line in the block was dropped, so the collision was "
        f"never compared:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-" in proc.stderr


def test_a_cache_key_in_a_local_reusable_workflow_is_seen(tmp_path):
    """`uses: ./.github/workflows/x.yml` names the file, not a directory with action.yml.

    Probing only for `action.yml` beneath the reference found nothing, so a reusable
    workflow called from a pull request declared keys that stayed outside the comparison
    entirely: reachable from a pull request in fact, invisible to the check. The
    composite-action case was already covered, which is what made this one easy to miss.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    wf.mkdir(parents = True)
    (wf / "shared-build.yml").write_text(
        "name: shared-build\n"
        "on:\n"
        "  workflow_call:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: reuse-v1-${{ runner.os }}-abc\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  call:\n"
        "    uses: ./.github/workflows/shared-build.yml\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("reuse-v1-pub-${{ runner.os }}", "            reuse-v1-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the reusable workflow's reuse-v1- key was not seen, so the publish prefix "
        f"matched nothing:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "reuse-v1-" in proc.stderr


def test_a_composite_key_equal_to_a_publish_key_is_caught(tmp_path):
    """An EQUAL key, not a prefix, and declared in a composite action rather than a workflow.

    Composite keys reached the prefix comparison but not the exact one, so a publish
    workflow sharing a literal key with a PR-reachable action and carrying no
    restore-keys at all passed. That is the original cache-poisoning shape, and it was
    the one route through this check with nothing watching it.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "shared-cache"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: shared cache\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: wheels-shared-key\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/shared-cache\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: wheels-shared-key\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a composite action's literal key equal to the publish key was accepted:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    assert "wheels-shared-key" in proc.stderr


def test_a_shell_built_namespace_is_narrowed_by_the_inputs_callers_pass(tmp_path):
    """Recording a namespace more broadly than the real one is a false rejection.

    The pip cache builds `prefix="pip-${name}-..."`, so reading the shell alone records
    the bare head `pip-`, which then collides with any publish prefix beginning `pip-`
    including a properly partitioned `pip-release-` that no pull request can write.
    Substituting the `name:` values callers actually pass gives `pip-mlx-`, which does
    not.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "pip-cache-restore"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: pip cache restore\n"
        "inputs:\n"
        "  name:\n"
        "    required: true\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - id: probe\n"
        "      shell: bash\n"
        "      run: |\n"
        "        name=\"${{ inputs.name }}\"\n"
        "        prefix=\"pip-${name}-${{ runner.os }}-\"\n"
        "        echo \"key=${prefix}abc\" >> \"$GITHUB_OUTPUT\"\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pip-cache-restore\n"
        "        with:\n"
        "          name: mlx\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("pip-release-pub-${{ runner.os }}", "            pip-release-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"`pip-release-` cannot be written by a pull request whose only namespace is "
        f"`pip-mlx-`, so this must pass:\n{proc.stdout}\n{proc.stderr}"
    )

    # And the narrowed namespace still has teeth: a publish prefix over `pip-mlx-` fails.
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("pip-mlx-pub-${{ runner.os }}", "            pip-mlx-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a publish prefix over the PR's real `pip-mlx-` namespace must still be "
        f"rejected:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "pip-mlx-" in proc.stderr

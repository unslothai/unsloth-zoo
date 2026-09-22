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


def test_a_publish_composite_that_restores_a_pr_namespace_is_caught(tmp_path):
    """The publish side delegates to local actions too, and that half went unread.

    Only the top-level publish workflow was parsed for `restore-keys`, so a publish
    workflow whose composite owns the `actions/cache/restore` contributed no prefixes at
    all. A pull request writing `shared-*` against a publish-only composite restoring
    `shared-` therefore passed, and a comparison that collects nothing on one side
    reports success rather than admitting it looked at nothing.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "publish-cache"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: publish cache\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache/restore@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: shared-publish-${{ runner.os }}\n"
        "        restore-keys: |\n"
        "          shared-\n"
    )
    (wf / "pr-build.yml").write_text(_pr_workflow("shared-${{ runner.os }}-abc"))
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/publish-cache\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the publish composite's `shared-` fallback was never collected, so the "
        f"collision was not compared:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-" in proc.stderr


def test_narrowing_keeps_the_broad_head_when_a_caller_is_dynamic(tmp_path):
    """Substituting only the literal call sites discards the dynamic one's namespace.

    With one caller passing `name: mlx` and another `name: ${{ matrix.cache_name }}`, the
    literal set is non-empty, so narrowing replaced the broad `pip-v2-` head with
    `pip-v2-mlx-` alone. The matrix caller can still expand to `shared`, write
    `pip-v2-shared-abc`, and a publish `restore-keys: pip-v2-shared-` would pass. The
    narrowing I added to remove a false REJECTION had therefore opened a false
    ACCEPTANCE, which is the worse of the two.
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
        "        prefix=\"pip-v2-${name}-\"\n"
        "        echo \"key=${prefix}abc\" >> \"$GITHUB_OUTPUT\"\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  literal:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pip-cache-restore\n"
        "        with:\n"
        "          name: mlx\n"
        "  dynamic:\n"
        "    runs-on: ubuntu-latest\n"
        "    strategy:\n"
        "      matrix:\n"
        "        cache_name: [shared, other]\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pip-cache-restore\n"
        "        with:\n"
        "          name: ${{ matrix.cache_name }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("pip-v2-shared-pub-${{ runner.os }}", "            pip-v2-shared-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a dynamic call site's namespace was discarded by the narrowing, so a publish "
        f"prefix over it passed:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "pip-v2-" in proc.stderr


def test_a_folded_restore_keys_block_is_one_prefix_not_several(tmp_path):
    """Folding joins the lines with spaces, so the runtime fallback is a single string.

    `restore-keys: >` over `safe-only-` and `shared-` reaches actions/cache as
    `safe-only- shared-`, which cannot restore a `shared-` key: there is no fallback
    named `shared-` at all. Reading each physical line as its own prefix invented one,
    and rejecting a configuration over an invented fallback is how a security lint earns
    the reputation that gets it switched off.
    """
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("shared-${{ runner.os }}-abc"))
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
        "          key: safe-only-${{ runner.os }}\n"
        "          restore-keys: >\n"
        "            safe-only-\n"
        "            shared-\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"a folded block is one space-joined prefix and cannot reach the `shared-` "
        f"namespace, so this must pass:\n{proc.stdout}\n{proc.stderr}"
    )

    # The literal form of the same block really does offer `shared-`, and must fail.
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys(
            "safe-only-${{ runner.os }}",
            "            safe-only-\n            shared-\n",
        )
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a literal block DOES offer `shared-` as a separate fallback and must be "
        f"rejected:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-" in proc.stderr


def test_a_quoted_restore_keys_field_is_read(tmp_path):
    """`"restore-keys": |` is valid YAML and offers the same fallback.

    The lexical reader matched only the bare token, so this spelling produced no
    prefixes at all and the collision was accepted. One of several spellings that had to
    be added one at a time before the reader was replaced with the parser, which resolves
    all of them to the same mapping key.
    """
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("shared-${{ runner.os }}-abc"))
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
        '          "key": pub-${{ runner.os }}\n'
        '          "restore-keys": |\n'
        "            shared-\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a quoted restore-keys field was not read:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-" in proc.stderr


def test_a_restore_keys_sequence_is_read(tmp_path):
    """`restore-keys: [a-, shared-]` is the sequence form, which the line reader never saw."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("shared-${{ runner.os }}-abc"))
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
        "          key: pub-${{ runner.os }}\n"
        "          restore-keys: [safe-, shared-]\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a sequence-form restore-keys was not read:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-" in proc.stderr


def test_a_flow_style_local_uses_is_followed(tmp_path):
    """`- {uses: ./.github/actions/x}` is the same step mapping in flow style.

    The traversal matched `uses:` lexically, so a flow-style or quoted-key call was never
    followed and the namespace that action declares stayed outside the comparison.
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
        "        key: flow-v1-${{ runner.os }}-abc\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - {uses: ./.github/actions/shared-cache}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("flow-v1-pub-${{ runner.os }}", "            flow-v1-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a flow-style local `uses` was not followed, so the action's namespace was "
        f"invisible:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "flow-v1-" in proc.stderr


def test_a_caller_supplied_key_is_resolved_not_dismissed(tmp_path):
    """`key: ${{ inputs.cache_key }}` is caller-supplied, which is not delegation.

    `steps.*` genuinely delegates: the real key is built in a composite's shell and is
    collected from there. `inputs.*` is different, because the value comes from the
    CALLER, so a pull request passing `cache_key: shared-abc` writes the `shared-`
    namespace. Treating the two alike dropped this key silently and a publish `shared-`
    fallback passed.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    wf.mkdir(parents = True)
    (wf / "shared-build.yml").write_text(
        "name: shared-build\n"
        "on:\n"
        "  workflow_call:\n"
        "    inputs:\n"
        "      cache_key:\n"
        "        required: true\n"
        "        type: string\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: ${{ inputs.cache_key }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  call:\n"
        "    uses: ./.github/workflows/shared-build.yml\n"
        "    with:\n"
        "      cache_key: shared-abc\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("shared-pub-${{ runner.os }}", "            shared-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a caller-supplied cache key was dismissed as delegation:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-" in proc.stderr


def test_a_key_delegated_to_a_step_output_is_still_accepted(tmp_path):
    """The other half of the rule above, and the reason it is not simply stricter.

    Every live caller of this repository's pip-cache-save passes
    `key: ${{ steps.pip-cache.outputs.key }}`, whose real namespace was already collected
    from the restoring action's shell. Reporting that as undecidable failed the live tree,
    which is the false-failure shape that gets a security check switched off, so
    resolved-with-no-literals has to mean delegation rather than doubt.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "cache-save"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: cache save\n"
        "inputs:\n"
        "  key:\n"
        "    required: true\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache/save@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: ${{ inputs.key }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: probe\n"
        "        run: echo 'key=own-v1-abc' >> \"$GITHUB_OUTPUT\"\n"
        "      - uses: ./.github/actions/cache-save\n"
        "        with:\n"
        "          key: ${{ steps.probe.outputs.key }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("unrelated-pub-${{ runner.os }}", "            unrelated-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"a key delegated to a step output, against an unrelated publish namespace, "
        f"must pass:\n{proc.stdout}\n{proc.stderr}"
    )


def test_inputs_are_collected_through_a_wrapper_action(tmp_path):
    """Call sites live in composites too, not only in workflow files.

    A cache action reached through a wrapper gets its inputs from that wrapper. Reading
    only the top-level workflows meant that call site was invisible, so if the workflow
    ALSO called the action directly with a literal, every input looked resolved and the
    narrowing dropped the namespace the wrapper passes.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    inner = root / "actions" / "pip-cache-restore"
    wrapper = root / "actions" / "setup-wrapper"
    wf.mkdir(parents = True)
    inner.mkdir(parents = True)
    wrapper.mkdir(parents = True)
    (inner / "action.yml").write_text(
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
        "        prefix=\"pip-v3-${name}-\"\n"
        "        echo \"key=${prefix}abc\" >> \"$GITHUB_OUTPUT\"\n"
    )
    (wrapper / "action.yml").write_text(
        "name: setup wrapper\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: ./.github/actions/pip-cache-restore\n"
        "      with:\n"
        "        name: wrapped\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  direct:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pip-cache-restore\n"
        "        with:\n"
        "          name: direct\n"
        "  viawrapper:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/setup-wrapper\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys(
            "pip-v3-wrapped-pub-${{ runner.os }}", "            pip-v3-wrapped-\n"
        )
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the wrapper's `name: wrapped` call site was not collected, so the narrowing "
        f"dropped that namespace:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "pip-v3-" in proc.stderr


def test_an_omission_before_the_first_literal_is_counted(tmp_path):
    """Counting omissions in the same pass made the answer depend on call-site order.

    A site that omitted an input had no bucket yet, so the omission went unrecorded, and
    a later site supplying a literal made the input look fully resolved. Verified before
    fixing: a composite called first with no `name` and then with `name: safe` reported
    `({'safe'}, True)`, so the namespace narrowed to `safe` and the action's real default
    namespace was left undefended.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "pipc"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: pipc\n"
        "inputs:\n"
        "  name:\n"
        "    default: shared\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - id: probe\n"
        "      shell: bash\n"
        "      run: |\n"
        "        name=\"${{ inputs.name }}\"\n"
        "        prefix=\"pipx-${name}-\"\n"
        "        echo \"key=${prefix}abc\" >> \"$GITHUB_OUTPUT\"\n"
    )
    # The omitting call site comes FIRST, which is the ordering that used to be lost.
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  defaulted:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pipc\n"
        "  named:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pipc\n"
        "        with:\n"
        "          name: safe\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("pipx-pub-${{ runner.os }}", "            pipx-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the first call site omits `name`, so the namespace is not fully resolved and "
        f"the broad `pipx-` head must still be defended:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "pipx-" in proc.stderr


def test_a_prefix_that_opens_with_a_variable_is_recovered(tmp_path):
    """`prefix="${name}-pip-..."` put its literal part after the variable.

    The head pattern requires an alphanumeric start, so this composite contributed no
    head at all, while the `steps.*` key reading its output was dismissed as delegation.
    A publish `restore-keys: shared-pip-` then had nothing to be compared against, which
    is the failure mode where a check reports success having looked at nothing.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "varfirst"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: varfirst\n"
        "inputs:\n"
        "  name:\n"
        "    required: true\n"
        "outputs:\n"
        "  key:\n"
        "    value: ${{ steps.probe.outputs.key }}\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - id: probe\n"
        "      shell: bash\n"
        "      run: |\n"
        "        name=\"${{ inputs.name }}\"\n"
        "        prefix=\"${name}-pip-${{ runner.os }}-\"\n"
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
        "      - id: vf\n"
        "        uses: ./.github/actions/varfirst\n"
        "        with:\n"
        "          name: shared\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: ${{ steps.vf.outputs.key }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("shared-pip-pub-${{ runner.os }}", "            shared-pip-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the namespace is `shared-pip-` once `name` is substituted, so the publish "
        f"fallback over it must be rejected:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-pip-" in proc.stderr


def test_an_input_backed_key_is_resolved_before_the_exact_comparison(tmp_path):
    """The plainest shape of all: an equal key, with no `restore-keys` anywhere.

    Input resolution reached the prefix comparison through `pr_heads` and stopped there,
    so a pull request writing `shared-key` directly, against a dispatch workflow calling
    a reusable workflow whose key is `${{ inputs.cache_key }}` with
    `cache_key: shared-key`, compared a literal against an unexpanded expression and
    matched nothing.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    wf.mkdir(parents = True)
    (wf / "shared-build.yml").write_text(
        "name: shared-build\n"
        "on:\n"
        "  workflow_call:\n"
        "    inputs:\n"
        "      cache_key:\n"
        "        required: true\n"
        "        type: string\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: ${{ inputs.cache_key }}\n"
    )
    (wf / "pr-build.yml").write_text(_pr_workflow("shared-key"))
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  call:\n"
        "    uses: ./.github/workflows/shared-build.yml\n"
        "    with:\n"
        "      cache_key: shared-key\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the publish side's key resolves to `shared-key`, which the PR writes directly, "
        f"so this exact collision must be caught:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-key" in proc.stderr
